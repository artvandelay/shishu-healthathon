"""Post-call doctor and mother summaries."""

import copy
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

import checklist
import call_state
import call_state_contract as contract
import models
from prompts import DOCTOR_MOTHER_SUMMARY_PROMPT

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

_REQUIRED_HEADINGS = (
    "### Patient Profile & Chief Complaint",
    "### Positive Red Flags & NRROID Gate Alerts",
    "### Birth & Neonatal History",
    "### Development, Growth & Physical Features",
    "### Systemic, Infections & Hematology",
    "### Antenatal (Pregnancy) History",
    "### Family & Pedigree History",
    "### Pertinent Negatives",
)


def _user_content(transcript: list[dict], findings: dict, red_flags: list[str]) -> str:
    payload = {
        "transcript": transcript,
        "findings": findings,
        "red_flags": red_flags,
    }
    return (
        "Write the post-call JSON from this call.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("summary response was not a JSON object")
    return data


def _flag_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        domain = str(item.get("domain") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if domain and detail:
            return f"{domain}: {detail}"
        return detail or domain
    return str(item).strip()


def _validate(payload: dict) -> dict:
    doctor = payload.get("doctor_markdown")
    if not isinstance(doctor, str) or not doctor.strip():
        doctor = payload.get("summary_doctor")
    mother = payload.get("mother_hindi")
    if not isinstance(mother, str) or not mother.strip():
        mother = payload.get("summary_mother")
    if not isinstance(doctor, str) or not doctor.strip():
        raise ValueError("missing doctor_markdown")
    if not isinstance(mother, str) or not mother.strip():
        raise ValueError("missing mother_hindi")
    missing = [heading for heading in _REQUIRED_HEADINGS if heading not in doctor]
    if missing:
        raise ValueError("doctor_markdown missing headings: " + ", ".join(missing))
    extracted = payload.get("extracted_findings")
    if extracted is None:
        extracted = {}
    if not isinstance(extracted, dict):
        raise ValueError("extracted_findings was not an object")
    flags = payload.get("red_flags")
    if flags is None:
        flags = []
    if not isinstance(flags, list):
        raise ValueError("red_flags was not a list")
    return {
        "doctor_markdown": doctor.strip(),
        "mother_hindi": mother.strip(),
        "extracted_findings": extracted,
        "red_flags": flags,
    }


def _user_text(transcript: list[dict]) -> str:
    parts: list[str] = []
    for turn in transcript or []:
        role = str(turn.get("role") or "").lower()
        if role in ("user", "mother"):
            parts.append(str(turn.get("text") or ""))
    return " ".join(parts)


_UNKNOWN_PHRASES = (
    "unknown",
    "does not know",
    "does not recall",
    "don't know",
    "do not know",
    "not sure",
    "not reported",
    "not discussed",
)
_SECTION_DOMAINS = (
    ("### Birth & Neonatal History", ("birth_nicu",)),
    ("### Development, Growth & Physical Features", ("milestones",)),
    ("### Systemic, Infections & Hematology", ("infections", "bleeding_joints", "metabolic_fasting")),
    ("### Antenatal (Pregnancy) History", ("pregnancy",)),
    (
        "### Family & Pedigree History",
        ("consanguinity", "family_infant_death", "family_history"),
    ),
)


def _unknown_detail(detail: str) -> bool:
    text = (detail or "").lower()
    return any(phrase in text for phrase in _UNKNOWN_PHRASES)


def _not_reported(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    detail = str(value.get("detail") or "").strip().lower()
    return (not detail) or "not reported" in detail or detail in {"not discussed", "none", "n/a"}


def _add_flag(flags: list[str], domain: str, detail: str) -> None:
    text = f"{domain}: {detail}".strip()
    if text and text not in flags:
        flags.append(text)


def _section_span(doctor: str, heading: str) -> tuple[int, int] | None:
    marker = heading + "\n"
    start = doctor.find(marker)
    if start < 0:
        return None
    body = start + len(marker)
    nxt = doctor.find("\n### ", body)
    return body, (nxt if nxt >= 0 else len(doctor))


def _fill_red_flag_section(doctor: str, red_flags: list[str]) -> str:
    if not red_flags:
        return doctor
    span = _section_span(doctor, "### Positive Red Flags & NRROID Gate Alerts")
    if span is None or "not reported" not in doctor[span[0]:span[1]].lower():
        return doctor
    lines = "\n".join(f"- {flag}" for flag in red_flags)
    return doctor[: span[0]] + lines + "\n" + doctor[span[1] :]


def _similar_detail(left: str, right: str) -> bool:
    a = (left or "").strip().lower()
    b = (right or "").strip().lower()
    return bool(a and b and (a in b or b in a))


def _dedupe_similar_flags(flags: list[str]) -> list[str]:
    kept: list[str] = []
    for flag in flags:
        domain, _, detail = flag.partition(":")
        duplicate = False
        for prev in kept:
            pdomain, _, pdetail = prev.partition(":")
            if pdomain.strip() != domain.strip():
                continue
            if _similar_detail(detail, pdetail):
                duplicate = True
                break
        if not duplicate:
            kept.append(flag)
    return kept


def _reconcile_flags(findings: dict, red_flags: list[str]) -> list[str]:
    """One red-flag line per domain, and only when that finding is actually a flag."""
    locked_off: set[str] = set()
    from_findings: list[tuple[str, str]] = []
    for domain, value in findings.items():
        if not isinstance(value, dict):
            continue
        key = str(domain)
        detail = str(value.get("detail") or "").strip()
        status = str(value.get("status") or "").lower()
        if _unknown_detail(detail) or status == "negative":
            value["is_red_flag"] = False
            locked_off.add(key)
            continue
        if value.get("is_red_flag") and detail:
            from_findings.append((key, detail))
        else:
            locked_off.add(key)
    out: list[str] = []
    covered: set[str] = set()
    for domain, detail in from_findings:
        _add_flag(out, domain, detail)
        covered.add(domain)
    for item in red_flags:
        text = _flag_text(item)
        domain, sep, detail = text.partition(":")
        domain = domain.strip()
        detail = detail.strip() if sep else ""
        if not domain or domain in covered or domain in locked_off:
            continue
        if _unknown_detail(detail or text):
            continue
        _add_flag(out, domain, detail or text)
    return _dedupe_similar_flags(out)


def _fill_empty_sections(doctor: str, findings: dict) -> str:
    for heading, domains in _SECTION_DOMAINS:
        bullets: list[str] = []
        for domain in domains:
            value = findings.get(domain)
            if not isinstance(value, dict):
                continue
            detail = str(value.get("detail") or "").strip()
            status = str(value.get("status") or "").lower()
            if not detail or status == "negative" or _unknown_detail(detail) or _not_reported(value):
                continue
            bullets.append(f"- {domain}: {detail}")
        if not bullets:
            continue
        span = _section_span(doctor, heading)
        if span is None:
            continue
        body = doctor[span[0]:span[1]].strip().lstrip("-* ").lower().rstrip(".")
        if body != "not reported":
            continue
        doctor = doctor[: span[0]] + "\n".join(bullets) + "\n" + doctor[span[1] :]
    return doctor


MOTHER_QUESTIONS_HEADING = (
    "अगली बार जब आप डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से मिलें, तो ये सवाल ज़रूर पूछें:"
)
MOTHER_FLAG_FOOTER = (
    "नोट: ये सवाल आपकी अगली विज़िट में डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से चर्चा के लिए हैं। "
    "यह किसी बीमारी की पुष्टि (डायग्नोसिस) नहीं है — केवल डॉक्टर ही जाँच करके सही सलाह दे सकते हैं।"
)
_NO_CHILD_MOTHER_QUESTIONS: tuple[str, ...] = (
    "अगली प्रेग्नेंसी से पहले मुझे कौन सी जाँच करवा लेनी चाहिए?",
    "परिवार में कोई बीमारी रही हो, तो डॉक्टर को कौन सी बात ज़रूर बतानी चाहिए?",
    "किन लक्षणों पर मुझे तुरंत डॉक्टर या स्वास्थ्य कार्यकर्ता को दिखाना चाहिए?",
)
_DEFAULT_MOTHER_QUESTIONS: tuple[str, ...] = (
    "मेरे बच्चे की उम्र के हिसाब से उसका वज़न, लंबाई और सीखना-बढ़ना (बैठना, चलना, बोलना) ठीक चल रहा है या नहीं?",
    "बच्चे के सभी ज़रूरी टीके (टीकाकरण कार्ड के अनुसार, जैसे BCG) और सुनने-देखने की सामान्य जाँच पूरी है या कोई अगला टीका बाकी है?",
    "बच्चे के रोज़ के खान-पान और सेहत के लिए अभी मुझे किन बातों का ध्यान रखना चाहिए, और किन लक्षणों के दिखने पर तुरंत डॉक्टर या स्वास्थ्य कार्यकर्ता को दिखाना चाहिए?",
    "क्या परिवार या प्रेग्नेंसी की सामान्य सेहत के लिए खून की कमी (एनीमिया) या थैलेसीमिया जैसी किसी सामान्य जाँच की ज़रूरत है?",
)

# A domain sentence that names one event must not be reused for a sibling topic.
# These topics get their own question; the shared domain sentence stays for the rest.
_TOPIC_MOTHER_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "birth_mode",
                "प्रसव ऑपरेशन से हुआ था — क्या इस बारे में डॉक्टर से कोई जाँच या सलाह पूछनी चाहिए?",
    ),
    (
        "birth_term",
                "बच्चा समय से पहले या समय के बाद पैदा हुआ था — क्या इस कारण अभी कोई जाँच ज़रूरी है?",
    ),
    (
        "body_fits_milestones",
                "बच्चे को झटके आते हैं, या बैठना-चलना-बोलना देर से हुआ — क्या इसके लिए कोई जाँच करवानी चाहिए?",
    ),
    (
        "body_transfusion",
                "बच्चे को खून चढ़ाना पड़ा था — क्या आगे खून या जाँच के बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "fam_miscarriage",
                "गर्भ गिरा था — क्या अगली गर्भावस्था के लिए कोई सामान्य जाँच या सलाह लेनी चाहिए?",
    ),
    (
        "birth_cord_bleed",
                "नाल गिरने पर खून ज़्यादा हुआ था — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "birth_cord_3weeks",
                "नाल गिरने में तीन हफ़्ते से ज़्यादा लगे — क्या इसकी जाँच ज़रूरी है?",
    ),
    (
        "birth_bcg",
                "BCG के टीके की जगह पर सूजन या निशान है — क्या इसे डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "birth_cleft",
                "होंठ या तालु कटने की बात है — क्या इसके लिए कोई जाँच या सलाह लेनी चाहिए?",
    ),
    (
        "birth_hearing_vision",
                "सुनने या देखने की जाँच में बात आई थी — क्या आगे कोई जाँच ज़रूरी है?",
    ),
    (
        "body_blue",
                "होंठ या उँगलियाँ नीली होती हैं — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "body_tummy",
                "पेट धीरे-धीरे बढ़ रहा है — क्या लिवर या तिल्ली की जाँच पूछनी चाहिए?",
    ),
    (
        "body_blood_count",
                "खून की गिनती कम या ज़्यादा बताई गई — क्या आगे जाँच करवानी चाहिए?",
    ),
    (
        "preg_thalassemia",
                "थैलेसीमिया की जाँच में बात आई थी — क्या रिपोर्ट डॉक्टर को दिखानी चाहिए?",
    ),
    (
        "preg_scan",
                "प्रेग्नेंसी के स्कैन में बात आई थी — क्या उसे डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "preg_heart",
                "प्रेग्नेंसी या जन्म पर दिल की बात आई थी — क्या बच्चे के दिल की जाँच पूछनी चाहिए?",
    ),
    (
        "preg_transfusion",
                "प्रेग्नेंसी में खून चढ़ाना पड़ा था — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "preg_delivery",
                "प्रसव में दिक्कत हुई थी — क्या माँ की आगे की जाँच ज़रूरी है?",
    ),
    (
        "ill_odor",
                "पेशाब या मल की गंध अलग है — क्या इस बारे में जाँच पूछनी चाहिए?",
    ),
    (
        "ill_sudden",
                "जन्म के बाद अचानक बहुत बीमार हुए थे — क्या इसके कारण की जाँच ज़रूरी है?",
    ),
    (
        "body_bleeding_bruise",
                "चोट पर खून देर तक बहता है, या आसानी से नील पड़ते हैं — क्या इसकी जाँच करवानी चाहिए?",
    ),
    (
        "body_limb_pain",
                "हाथ या पैर में बार-बार दर्द होता है — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "body_regression",
                "जो काम बच्चा पहले कर लेता था, अब नहीं कर पाता — क्या इसकी जाँच ज़रूरी है?",
    ),
    (
        "body_private_parts",
                "गुप्त अंग जैसे होने चाहिए वैसे नहीं दिखते — क्या डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "body_lumps_birth",
                "जन्म से शरीर पर सूजन या गाँठ है — क्या इसे डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "body_proportion",
                "हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं — क्या इसकी जाँच पूछनी चाहिए?",
    ),
    (
        "body_thumb",
                "अंगूठा या उँगली छोटी है या नहीं है — क्या डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "body_puberty",
                "बढ़ती उम्र के बदलाव समय से जल्दी या देर से आए — क्या इस बारे में सलाह लेनी चाहिए?",
    ),
)

_TOPIC_QUESTION_IDS = {item[0] for item in _TOPIC_MOTHER_QUESTIONS}


def _flagged_topic_ids(findings: dict, red_flags: list[str] | None) -> set[str]:
    found: set[str] = set()
    blobs = [str(flag) for flag in (red_flags or [])]
    for value in (findings or {}).values():
        if isinstance(value, dict) and value.get("is_red_flag"):
            blobs.append(str(value.get("detail") or ""))
    for blob in blobs:
        head = blob.split(":", 1)[0].strip()
        if head in checklist._BY_ID:
            found.add(head)
    return found


_FLAG_DOMAIN_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "family_infant_death",
                "हमारे परिवार में पहले एक बच्चा छोटी उम्र में गुजर गया था — क्या इस बच्चे की सेहत या आगे की देखभाल के लिए हमें डॉक्टर से कोई विशेष सलाह या जाँच पूछनी चाहिए?",
    ),
    (
        "bleeding_joints",
                "बच्चे के गिरने पर घुटने या जोड़ में सूजन आने (या खून/नील से जुड़ी बात) के बारे में कृपया देखकर बताएँ कि क्या कोई जाँच करवानी चाहिए?",
    ),
    (
        "infections",
                "बच्चे को बार-बार बुखार या इंफेक्शन के कारण अस्पताल में भर्ती होना पड़ा है — इससे बचाव के लिए हमें आगे क्या जाँच या सावधानी रखनी चाहिए?",
    ),
    (
        "birth_nicu",
                "जन्म के समय या BCG के टीके के बाद जो बात हुई थी (जैसे पीलिया की नीली रोशनी, NICU, नाल, या टीके की जगह पर सूजन/निशान) — क्या उसके लिए अभी बच्चे की कोई जाँच ज़रूरी है?",
    ),
    (
        "milestones",
                "बच्चे के बढ़ने, बैठने-चलने-बोलने, पेट के आकार या देखने-सुनने को जाँच कर कृपया बताएँ कि क्या किसी अतिरिक्त जाँच की ज़रूरत है?",
    ),
    (
        "metabolic_fasting",
                "खाना देर से मिलने या हल्के बुखार में बच्चे को सुस्ती, उल्टी या झटके जैसी जो दिक्कत हुई थी — ऐसे समय में हमें क्या करना चाहिए और क्या कोई जाँच ज़रूरी है?",
    ),
    (
        "pregnancy",
                "प्रेग्नेंसी के समय रही बीपी, शुगर या थायरॉइड जैसी बात के लिए अभी माँ और बच्चे की कौन-सी नियमित जाँच करवानी चाहिए?",
    ),
    (
        "consanguinity",
                "हमारी शादी रिश्तेदारी में हुई है — क्या बच्चे की सेहत के लिए डॉक्टर से कोई सामान्य परामर्श (काउंसलिंग) लेना अच्छा रहेगा?",
    ),
    (
        "family_history",
                "परिवार में पहले जो सेहत से जुड़ी समस्या रही है, उसे ध्यान में रखते हुए क्या बच्चे के लिए कोई सामान्य जाँच पूछनी चाहिए?",
    ),
)

_FORBIDDEN_DIAGNOSTIC_RE = re.compile(
    r"(?i)\b(duchenne|pompe|wiskott[- ]?aldrich|mucopolysaccharidosis|mps\b|gaucher|"
    r"hemophilia|haemophilia|primary immunodeficiency|scid\b|spinal muscular atrophy)\b|"
    r"हीमोफीलिया|इम्यूनोडेफिशिएंसी|ड्यूशेन|पॉम्पे|म्यूकोपॉलीसेकेराइडोसिस"
)
def _scrub_diagnostic_terms(text: str) -> str:
    if not text:
        return ""
    cleaned = _FORBIDDEN_DIAGNOSTIC_RE.sub("स्वास्थ्य संबंधी जाँच", text)
    return cleaned.replace("गुज़र", "गुजर")


def _drop_ungrounded_name(findings: dict, transcript: list[dict]) -> None:
    """Drop a mother-name finding whose name never appears in her own words."""
    recorded = findings.get("profile_mother_name")
    if not isinstance(recorded, dict):
        return
    user = _user_text(transcript).lower()
    skip = {"mother", "mothers", "name", "said", "info"}
    tokens = [word.strip(".'") for word in str(recorded.get("detail") or "").split()]
    names = [word for word in tokens if len(word) >= 4 and word.lower() not in skip]
    if names and not any(word.lower() in user for word in names):
        findings.pop("profile_mother_name", None)


def _extract_question_lines(text: str) -> list[str]:
    questions: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("•-* ")
        if not line:
            continue
        head, sep, rest = line.partition(" ")
        if sep and head.rstrip(".)-:").isdigit():
            line = rest.strip()
        if "?" not in line and "？" not in line:
            continue
        cleaned = _scrub_diagnostic_terms(line)
        if cleaned and cleaned not in questions:
            questions.append(cleaned)
    return questions


TOPIC_TO_DOMAIN: dict[str, str] = {
    "birth_mode": "birth_nicu",
    "birth_term": "birth_nicu",
    "birth_nicu": "birth_nicu",
    "birth_cord_bleed": "birth_nicu",
    "birth_cord_3weeks": "birth_nicu",
    "birth_bcg": "birth_nicu",
    "birth_hearing_vision": "birth_nicu",
    "birth_cleft": "birth_nicu",
    "body_fits_milestones": "milestones",
    "body_regression": "milestones",
    "body_private_parts": "milestones",
    "body_lumps_birth": "milestones",
    "body_proportion": "milestones",
    "body_thumb": "milestones",
    "body_growth": "milestones",
    "body_puberty": "milestones",
    "body_joint_swelling": "bleeding_joints",
    "body_bleeding_bruise": "bleeding_joints",
    "body_limb_pain": "bleeding_joints",
    "ill_fasting": "metabolic_fasting",
    "ill_odor": "metabolic_fasting",
    "ill_sudden": "metabolic_fasting",
    "ill_admissions": "infections",
    "ill_iv_antibiotics": "infections",
    "body_transfusion": "infections",
    "body_blood_count": "infections",
    "body_tummy": "infections",
    "body_blue": "infections",
    "preg_conditions": "pregnancy",
    "preg_thalassemia": "pregnancy",
    "preg_scan": "pregnancy",
    "preg_heart": "pregnancy",
    "preg_transfusion": "pregnancy",
    "preg_delivery": "pregnancy",
    "fam_related": "consanguinity",
    "fam_miscarriage": "family_infant_death",
    "fam_infant_death": "family_infant_death",
    "fam_fits_delay": "family_history",
    "fam_blind": "family_history",
    "fam_deaf": "family_history",
    "fam_transfusion": "family_history",
    "fam_bleeding": "family_history",
    "fam_cancer": "family_history",
    "fam_heart": "family_history",
    "fam_kidney": "family_history",
    "fam_bp_stroke": "family_history",
    "fam_maternal_side": "family_history",
    "fam_lumps": "family_history",
    "fam_same_problem": "family_history",
}


def _has_any_red_flag(findings: dict, red_flags: list[str] | None = None) -> bool:
    if red_flags:
        return True
    for value in (findings or {}).values():
        if isinstance(value, dict) and value.get("is_red_flag") and not _unknown_detail(str(value.get("detail") or "")):
            return True
    return False


def _ensure_mother_covers_flags(
    mother: str,
    findings: dict,
    red_flags: list[str] | None = None,
    fillers: tuple[str, ...] | None = None,
) -> str:
    text = _scrub_diagnostic_terms((mother or "").strip())
    questions = _extract_question_lines(text)
    combined_q = "\n".join(questions)

    flagged_topics = _flagged_topic_ids(findings, red_flags)
    for domain, fallback_q in _FLAG_DOMAIN_QUESTIONS:
        domain_topics = {tid for tid, dom in TOPIC_TO_DOMAIN.items() if dom == domain}
        entry = (findings or {}).get(domain)
        flagged_in_list = any(
            str(f).split(":", 1)[0].strip() in ({domain} | domain_topics)
            for f in (red_flags or [])
        )
        topic_flagged = any(
            isinstance((findings or {}).get(tid), dict)
            and bool((findings or {})[tid].get("is_red_flag"))
            and not _unknown_detail(str((findings or {})[tid].get("detail") or ""))
            for tid in domain_topics
        )
        is_flagged = (
            isinstance(entry, dict)
            and bool(entry.get("is_red_flag"))
            and not _unknown_detail(str(entry.get("detail") or ""))
        ) or flagged_in_list or topic_flagged
        if not is_flagged:
            continue
        domain_hits = domain_topics & flagged_topics
        # A sentence that names one event is not reused for a sibling that has its own question.
        if domain_hits and domain_hits <= _TOPIC_QUESTION_IDS:
            continue
        if fallback_q not in combined_q:
            questions.append(fallback_q)
            combined_q = "\n".join(questions)

    for topic_id, question in _TOPIC_MOTHER_QUESTIONS:
        if topic_id not in flagged_topics or question in combined_q:
            continue
        questions.append(question)
        combined_q = "\n".join(questions)

    for default_q in (fillers or _DEFAULT_MOTHER_QUESTIONS):
        if len(questions) >= 3:
            break
        if default_q not in questions:
            questions.append(default_q)

    numbered = "\n".join(f"{idx}. {q}" for idx, q in enumerate(questions[:6], start=1))
    return f"{MOTHER_QUESTIONS_HEADING}\n\n{numbered}\n\n{MOTHER_FLAG_FOOTER}"


def default_mother_questions(
    findings: dict | None = None,
    red_flags: list[str] | None = None,
    has_children: bool = True,
) -> str:
    """Build actionable Hindi questions for the mother even when no LLM summary ran."""
    fillers = _DEFAULT_MOTHER_QUESTIONS if has_children else _NO_CHILD_MOTHER_QUESTIONS
    return _ensure_mother_covers_flags("", findings or {}, red_flags or [], fillers)


def build_fallback_summaries(
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
) -> dict:
    """Deterministic fallback when both LLM summary providers are unreachable."""
    merged_findings = dict(findings or {})
    merged_flags = [_flag_text(item) for item in (red_flags or []) if _flag_text(item)]
    for domain, value in merged_findings.items():
        if isinstance(value, dict) and value.get("is_red_flag"):
            _add_flag(merged_flags, str(domain), str(value.get("detail") or "").strip())
    merged_flags = _reconcile_flags(merged_findings, merged_flags)
    shell = "\n".join(f"{heading}\nNot reported." for heading in _REQUIRED_HEADINGS)
    doctor = _fill_red_flag_section(shell, merged_flags)
    doctor = _fill_empty_sections(doctor, merged_findings)
    mother = default_mother_questions(merged_findings, merged_flags)
    return {
        "summary_doctor": doctor,
        "summary_mother": mother,
        "findings": merged_findings,
        "red_flags": merged_flags,
    }


def _assemble(payload: dict, findings: dict, red_flags: list[str], transcript: list[dict]) -> dict:
    cleaned = _validate(payload)
    merged_findings = dict(findings or {})
    for key, value in cleaned["extracted_findings"].items():
        if key not in merged_findings and not _not_reported(value):
            merged_findings[key] = value
    merged_flags: list[str] = []
    for item in list(red_flags or []) + list(cleaned["red_flags"]):
        text = _flag_text(item)
        if text and text not in merged_flags:
            merged_flags.append(text)
    for domain, value in merged_findings.items():
        if isinstance(value, dict) and value.get("is_red_flag"):
            _add_flag(merged_flags, str(domain), str(value.get("detail") or "").strip())
    _drop_ungrounded_name(merged_findings, transcript)
    merged_flags = _reconcile_flags(merged_findings, merged_flags)
    doctor = _fill_red_flag_section(cleaned["doctor_markdown"], merged_flags)
    doctor = _fill_empty_sections(doctor, merged_findings)
    mother = _ensure_mother_covers_flags(cleaned["mother_hindi"], merged_findings, merged_flags)
    return {
        "summary_doctor": doctor,
        "summary_mother": mother,
        "findings": merged_findings,
        "red_flags": merged_flags,
    }


STRUCTURED_STATE_SUMMARY_PROMPT = """
You are the post-call clinical screening auditor and summarizer for a SHISHU call.
You are given:
1. The deterministic structured call state (profile + all 54 topics) and deterministic draft summary.
2. The call transcript (when available).

Your role is to act as a sanity-check and double-checker:
- Compare the deterministic structured state against the call transcript.
- If the mother clearly stated a fact in the transcript (e.g., her name, child age/name, or a positive symptom/history/detail) that was missed or mis-parsed in the structured state, include it in `profile_corrections` or `topic_corrections`, and reflect it in `doctor_markdown`, `mother_hindi`, `extracted_findings`, and `red_flags`.
- You are the judge of the transcript. When the deterministic draft was too rigid, misheard, or misclassified, override it: set `topic_corrections` to the corrected state (`{"state": "answered", "value": true/false, "details": {...}}`) and omit any false positive from `extracted_findings` and `red_flags`. A deterministic yes is a draft, not a fact you must repeat.
- Each topic line shows the question that was asked. For yes/no topics `value: true` means she said yes to that exact question, which is worded so that yes is a concern. Only set `value: true` in `topic_corrections` when she reported that concern.
- Every profile or topic correction must include `evidence`: an exact quote of the mother's words from the transcript. A correction without that quote is ignored. `details.delivery_type` is only "normal" or "operation". `details.place` is only "home" or "hospital". `profile_corrections.marital_status` is only "married", "about_to_marry", or "unmarried".
- If a topic state is `no_response`, `not_asked`, or `not_applicable`, and the mother did NOT answer it in the transcript, keep that exact label (`no_response`, `not_asked`, or `not_applicable`). NEVER turn `unknown`, `no_response`, `not_asked`, or `not_applicable` into `no`.
- Do not invent any fact, child identity, answer, or diagnosis. Never name specific rare diseases in `mother_hindi`.
- `mother_hindi` must contain 3 to 6 clear, natural Devanagari Hindi questions (each ending with `?`) for the mother to ask her doctor or ASHA/ANM दीदी at the next visit, directly covering every positive finding from the call.
- `doctor_markdown` must be a concise English clinical summary containing all required section headings:
  ### Patient Profile & Chief Complaint
  ### Positive Red Flags & NRROID Gate Alerts
  ### Birth & Neonatal History
  ### Development, Growth & Physical Features
  ### Systemic, Infections & Hematology
  ### Antenatal (Pregnancy) History
  ### Family & Pedigree History
  ### Pertinent Negatives

Return one JSON object and nothing else. Keys:
- doctor_sections: object. Keys are the eight section titles above, without the ###. Values are the section text.
- mother_questions: list of {"topic_id", "question"}. question is one Hindi question ending in ?.
- extracted_findings: object. red_flags: list of strings.
- topic_corrections: list of {"topic_id", "state", "value", "details", "evidence"}. state is answered, unknown, no_response, or not_applicable.
- profile_corrections: {"caller_name", "marital_status", "children", "evidence"}.
""".strip()


def _format_topic_detail(topic_id: str, saved: dict) -> str:
    need = (checklist._BY_ID.get(topic_id) or {}).get("need") or topic_id
    parts: list[str] = []
    details = saved.get("details") if isinstance(saved.get("details"), dict) else {}
    for k, v in details.items():
        if v is None or isinstance(v, (dict, bool)):
            continue
        sv = str(v).strip()
        if sv and sv.lower() not in {"noted", "skip", "unknown", "no_response", "none", "true", "false"}:
            parts.append(f"{k}={sv}")
    for inst in saved.get("instances") or []:
        if not isinstance(inst, dict):
            continue
        pid = str(inst.get("person_id") or "").strip()
        idet = inst.get("details") if isinstance(inst.get("details"), dict) else {}
        ibits = [
            f"{k}={v}"
            for k, v in idet.items()
            if v is not None
            and not isinstance(v, (dict, bool))
            and str(v).strip().lower() not in {"", "noted", "skip", "unknown", "no_response", "none", "true", "false"}
        ]
        if pid or ibits:
            parts.append(f"{pid}({', '.join(ibits)})" if ibits else pid)
    if parts:
        return f"positive finding reported in structured call state ({need}; {'; '.join(parts)})"
    return "positive finding reported in structured call state"


def state_to_findings_and_flags(
    state: dict | None,
    existing_findings: dict | None = None,
    existing_flags: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """Deterministically project structured CallState into findings and red_flags."""
    findings: dict = copy.deepcopy(existing_findings) if isinstance(existing_findings, dict) else {}
    red_flags: list[str] = [
        _flag_text(item) for item in (existing_flags or []) if _flag_text(item)
    ]
    if not isinstance(state, dict):
        return findings, red_flags

    profile = state.get("profile") if isinstance(state.get("profile"), dict) else {}
    caller_name = str(profile.get("caller_name") or "").strip()
    if caller_name and caller_name.lower() not in {"not reported", "unknown", "no_response"}:
        findings.setdefault(
            "profile_mother_name",
            {"status": "info", "detail": f"Mother's name is {caller_name}.", "is_red_flag": False},
        )
    children = [c for c in (profile.get("children") or []) if isinstance(c, dict)]
    if children:
        age_bits = []
        name_bits = []
        sex_bits = []
        for c in children:
            cname = str(c.get("name") or "").strip()
            if cname:
                name_bits.append(cname)
            csex = str(c.get("sex") or "").strip()
            if csex:
                sex_bits.append(csex)
            cage = c.get("age")
            if isinstance(cage, dict) and cage.get("value") is not None:
                age_bits.append(f"{cage['value']} {cage.get('unit') or 'years'}")
            elif cage not in (None, ""):
                age_bits.append(str(cage))
        if age_bits:
            findings.setdefault(
                "profile_child_age",
                {"status": "info", "detail": ", ".join(age_bits), "is_red_flag": False},
            )
        if name_bits:
            findings.setdefault(
                "profile_child_name",
                {"status": "info", "detail": ", ".join(name_bits), "is_red_flag": False},
            )
        if sex_bits:
            findings.setdefault(
                "profile_child_gender",
                {"status": "info", "detail": ", ".join(sex_bits), "is_red_flag": False},
            )

    topics = state.get("topics") if isinstance(state.get("topics"), dict) else {}
    for topic in checklist.TOPICS:
        tid = topic["id"]
        if tid.startswith("intro_") or tid == "close_anything_else":
            continue
        saved = topics.get(tid) if isinstance(topics.get(tid), dict) else {"state": "not_asked"}
        label = call_state.topic_answer_kind(saved)
        if label != "yes":
            continue
        detail = _format_topic_detail(tid, saved)
        flag_line = f"{tid}: {detail}"
        if not any(f.startswith(f"{tid}:") for f in red_flags):
            red_flags.append(flag_line)
        domain = TOPIC_TO_DOMAIN.get(tid, tid)
        prev_dom = findings.get(domain)
        if isinstance(prev_dom, dict) and prev_dom.get("status") == "positive":
            prev_det = str(prev_dom.get("detail") or "").strip()
            if detail not in prev_det:
                prev_dom["detail"] = f"{prev_det}; {tid}: {detail}" if prev_det else f"{tid}: {detail}"
            prev_dom["is_red_flag"] = True
        else:
            findings[domain] = {
                "status": "positive",
                "detail": f"{tid}: {detail}",
                "is_red_flag": True,
            }
    return findings, red_flags


_CORRECTION_STATES = {"answered", "unknown", "no_response", "not_asked"}


def _mentioned(text: str, transcript_blob: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return cleaned.lower() in transcript_blob or cleaned in transcript_blob


def _quote_supported(evidence: object, transcript: list[dict] | None) -> bool:
    """A correction stands only when its quote is her own words, copied exactly."""
    quote = str(evidence or "").strip()
    if len(quote) < 6:
        return False
    folded = quote.lower()
    for turn in transcript or []:
        if str(turn.get("role") or "").lower() not in ("user", "mother"):
            continue
        line = str(turn.get("text") or "")
        if quote in line or folded in line.lower():
            return True
    return False


def _clean_correction_details(details: dict) -> dict:
    details = dict(details or {})
    if details.get("delivery_type") not in contract.DELIVERY_TYPES:
        details.pop("delivery_type", None)
    if details.get("place") not in contract.BIRTH_PLACES:
        details.pop("place", None)
    return details


def _merge_correction_details(prior: dict, incoming: dict) -> dict:
    """Correction keys win. Keys it does not send stay. Invalid enums are still dropped."""
    incoming = dict(incoming or {})
    cleaned = _clean_correction_details(incoming)
    merged = dict(prior or {})
    for key in incoming:
        if key in cleaned:
            merged[key] = cleaned[key]
        else:
            merged.pop(key, None)
    return merged


def _merge_instance_details(prior_instances: list, new_instances: list) -> list:
    prior_by_id = {
        item.get("person_id"): item
        for item in prior_instances
        if isinstance(item, dict) and item.get("person_id")
    }
    merged = []
    for item in new_instances:
        if not isinstance(item, dict):
            continue
        inst = dict(item)
        prior = prior_by_id.get(inst.get("person_id"))
        prior_details = prior.get("details") if isinstance(prior, dict) else {}
        if not isinstance(prior_details, dict):
            prior_details = {}
        raw_details = inst.get("details") if isinstance(inst.get("details"), dict) else {}
        inst["details"] = _merge_correction_details(prior_details, raw_details)
        merged.append(inst)
    return merged


def _apply_topic_correction(
    name: str,
    topic: dict,
    new_state: str,
    new_val: object,
    new_details: dict,
    new_instances: list,
    *,
    replace: bool,
) -> None:
    """Write a model correction. Details merge; a correction cannot erase saved keys."""
    raw_details = dict(new_details or {})
    new_details = _clean_correction_details(raw_details)
    if new_state == "answered":
        prior_details = dict(topic.get("details") or {})
        topic["state"] = "answered"
        topic["value"] = new_val if new_val is not None else (False if replace else True)
        if replace:
            topic["details"] = _merge_correction_details(prior_details, raw_details)
        elif new_details:
            merged = dict(prior_details)
            merged.update(new_details)
            topic["details"] = merged
        if name == "birth_mode":
            call_state._derive_birth_mode_value(topic)
        if new_instances:
            topic["instances"] = _merge_instance_details(topic.get("instances") or [], new_instances)
        elif replace and not call_state._is_positive(topic):
            topic["instances"] = []
    elif new_state in {"unknown", "no_response", "not_asked"}:
        topic["state"] = new_state
        topic["value"] = None
        topic["details"] = _merge_correction_details(topic.get("details") or {}, raw_details)
        topic["instances"] = []
    topic["source"] = "summary_reconciliation"
    topic["reason"] = None


def reconcile_state_with_llm_payload(
    state: dict,
    payload: dict | None,
    transcript: list[dict] | None = None,
) -> dict:
    """Apply the frontier model's corrections. An answered draft can be overridden."""
    reconciled = copy.deepcopy(state)
    if not isinstance(payload, dict):
        return reconciled
    user_blob = _user_text(transcript or []).strip().lower()
    prof_corr = payload.get("profile_corrections")
    if isinstance(prof_corr, dict) and user_blob:
        prof = reconciled.setdefault("profile", {})
        cname = str(prof_corr.get("caller_name") or "").strip()
        if cname and cname.lower() not in {"not reported", "unknown"} and _mentioned(cname, user_blob):
            prof["caller_name"] = cname
            reconciled["topics"]["intro_name"]["state"] = "answered"
            reconciled["topics"]["intro_name"]["value"] = cname
        grounded = _quote_supported(prof_corr.get("evidence"), transcript)
        mstat = prof_corr.get("marital_status")
        if grounded and mstat in ("married", "about_to_marry", "unmarried"):
            prof["marital_status"] = mstat
            reconciled["topics"]["intro_married"]["state"] = "answered"
            reconciled["topics"]["intro_married"]["value"] = mstat
        if grounded and isinstance(prof_corr.get("children"), list) and prof_corr["children"]:
            norm_kids = call_state._normalize_children(prof_corr["children"])
            if norm_kids:
                prof["children"] = norm_kids
                reconciled["topics"]["intro_children"]["state"] = "answered"
                reconciled["topics"]["intro_children"]["value"] = True

    top_corr = payload.get("topic_corrections")
    if isinstance(top_corr, list):
        mapped = {}
        for item in top_corr:
            if isinstance(item, dict) and item.get("topic_id"):
                mapped[str(item["topic_id"])] = item
        top_corr = mapped
    if isinstance(top_corr, dict):
        topics = reconciled.setdefault("topics", {})
        for tid, patch in top_corr.items():
            if tid not in checklist._BY_ID or not isinstance(patch, dict):
                continue
            cur = topics.get(tid)
            if not isinstance(cur, dict):
                continue
            raw_state = str(patch.get("state") or "").strip().lower()
            new_val = patch.get("value")
            if tid == "birth_mode" and isinstance(new_val, str) and new_val not in contract.DELIVERY_TYPES:
                continue
            if raw_state == "yes":
                new_state = "answered"
                if new_val is None:
                    new_val = True
            elif raw_state == "no":
                new_state = "answered"
                if new_val is None:
                    new_val = False
            else:
                new_state = raw_state
            new_details = patch.get("details") if isinstance(patch.get("details"), dict) else {}
            new_instances = patch.get("instances") if isinstance(patch.get("instances"), list) else []
            if not _quote_supported(patch.get("evidence"), transcript):
                continue
            prior = cur.get("state")
            if prior == "answered":
                if new_state in _CORRECTION_STATES:
                    _apply_topic_correction(
                        tid, cur, new_state, new_val, new_details, new_instances, replace=True
                    )
                elif new_details or (new_instances and not cur.get("instances")):
                    _apply_topic_correction(
                        tid, cur, "answered", cur.get("value"), new_details, new_instances, replace=False
                    )
                continue
            if prior == "not_applicable":
                if new_state in {"answered", "unknown"}:
                    _apply_topic_correction(
                        tid, cur, new_state, new_val, new_details, new_instances, replace=True
                    )
                continue
            if new_state not in _CORRECTION_STATES:
                continue
            candidate = {"state": new_state, "value": new_val, "details": new_details}
            if new_state == "answered" and call_state._is_positive(candidate):
                _apply_topic_correction(
                    tid, cur, new_state, new_val, new_details, new_instances, replace=False
                )
            elif new_state == "unknown" and prior in {"not_asked", "no_response"}:
                _apply_topic_correction(
                    tid, cur, new_state, None, new_details, new_instances, replace=True
                )
            elif new_state == "answered" and new_val is False and prior in {"not_asked", "no_response"}:
                evidence = str(patch.get("evidence") or "").strip().lower()
                if len(evidence) >= 6 and evidence in user_blob:
                    _apply_topic_correction(
                        tid, cur, new_state, False, new_details, new_instances, replace=True
                    )
    call_state._apply_rules(reconciled)
    return reconciled


def state_to_summary_context(state: dict, transcript: list[dict] | None = None) -> str:
    """Facts and deterministic draft for the summary/reconciliation model."""
    det_findings, det_flags = state_to_findings_and_flags(state)
    det_mother = default_mother_questions(det_findings, det_flags)
    lines = [
        "Structured call state.",
        "Write language only from the structured state and grounded transcript.",
        "Do not add a fact, child identity, answer, or diagnosis.",
        "Profile: " + json.dumps(state.get("profile") or {}, ensure_ascii=False),
    ]
    topics = state.get("topics") or {}
    for topic in checklist.TOPICS:
        saved = topics.get(topic["id"]) if isinstance(topics.get(topic["id"]), dict) else {}
        lines.append(
            f"{topic['id']} (asked: {topic['question']}): state={call_state.topic_answer_kind(saved)} "
            f"value={json.dumps(saved.get('value'), ensure_ascii=False)} "
            f"details={json.dumps(saved.get('details') or {}, ensure_ascii=False)} "
            f"instances={json.dumps(saved.get('instances') or [], ensure_ascii=False)}"
        )
    lines.append("Deterministic Red Flags: " + json.dumps(det_flags, ensure_ascii=False))
    lines.append("Deterministic Mother Questions Draft:\n" + det_mother)
    if transcript:
        t_lines = []
        for turn in transcript:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip()
            text = str(turn.get("text") or "").strip()
            if role and text:
                t_lines.append(f"{role}: {text}")
        if t_lines:
            lines.append("Call Transcript (for sanity-checking and fixing missed answers):\n" + "\n".join(t_lines))
    return "\n".join(lines)


def build_state_fallback(
    state: dict,
    existing_findings: dict | None = None,
    existing_flags: list[str] | None = None,
) -> dict:
    """Doctor note and mother questions from structured states when both summary providers fail."""
    findings, red_flags = state_to_findings_and_flags(state, existing_findings, existing_flags)
    profile = state.get("profile") or {}
    topics = state.get("topics") or {}

    shell = "\n".join(f"{heading}\nNot reported." for heading in _REQUIRED_HEADINGS)
    profile_lines = [f"- Profile: {json.dumps(profile, ensure_ascii=False)}"]
    shell = shell.replace(
        "### Patient Profile & Chief Complaint\nNot reported.",
        "### Patient Profile & Chief Complaint\n" + "\n".join(profile_lines),
    )
    negatives = [
        topic["id"]
        for topic in checklist.TOPICS
        if not topic["id"].startswith("intro_")
        and call_state.topic_answer_kind(topics.get(topic["id"]) if isinstance(topics.get(topic["id"]), dict) else {}) == "no"
    ]
    if negatives:
        shell = shell.replace(
            "### Pertinent Negatives\nNot reported.",
            "### Pertinent Negatives\n- Negative topics: " + ", ".join(negatives),
        )
    doctor = _fill_red_flag_section(shell, red_flags)
    doctor = _fill_empty_sections(doctor, findings)

    lines = [
        doctor,
        "",
        "Structured screening record.",
        "Profile: " + json.dumps(profile, ensure_ascii=False),
    ]
    for topic in checklist.TOPICS:
        saved = topics.get(topic["id"]) if isinstance(topics.get(topic["id"]), dict) else {"state": "not_asked"}
        label = call_state.topic_answer_kind(saved)
        lines.append(f"- {topic['id']}: {label}")

    return {
        "summary_doctor": "\n".join(lines),
        "summary_mother": default_mother_questions(
            findings,
            red_flags,
            has_children=any(isinstance(child, dict) for child in (profile.get("children") or [])),
        ),
        "findings": findings,
        "red_flags": red_flags,
        "state": state,
    }


async def _from_gemini(
    user_content: str,
    system_prompt: str = DOCTOR_MOTHER_SUMMARY_PROMPT,
    model: str | None = None,
    schema=None,
) -> dict:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is unset")
    client = models.text_client()
    response = await client.aio.models.generate_content(
        model=model or models.TEXT_MODEL,
        contents=user_content,
        config=models.text_config(
            system=system_prompt,
            schema=schema,
            max_output_tokens=16384,
            temperature=0.2,
        ),
    )
    return _parse_json(response.text or "")


def _model_findings(payload: dict, reconciled_state: dict) -> dict:
    extracted = payload.get("extracted_findings")
    if isinstance(extracted, dict):
        return {
            key: value
            for key, value in extracted.items()
            if not _not_reported(value)
        }
    findings, _flags = state_to_findings_and_flags(reconciled_state)
    return findings


def _model_flags(payload: dict, reconciled_state: dict) -> list[str]:
    incoming = payload.get("red_flags")
    if isinstance(incoming, list):
        flags: list[str] = []
        for item in incoming:
            text = _flag_text(item)
            if text and text not in flags and not _unknown_detail(text):
                flags.append(text)
        return flags
    _findings, flags = state_to_findings_and_flags(reconciled_state)
    return flags


def _markdown_from_sections(sections: dict) -> str:
    parts = []
    for heading in _REQUIRED_HEADINGS:
        title = heading.removeprefix("### ").strip()
        body = str(sections.get(title) or sections.get(heading) or "").strip()
        parts.append(f"{heading}\n{body or 'Not reported.'}")
    return "\n".join(parts)


def _mother_from_questions(payload: dict) -> str:
    asked = payload.get("mother_questions")
    if not isinstance(asked, list):
        return ""
    lines = []
    for item in asked:
        text = item.get("question") if isinstance(item, dict) else item
        text = str(text or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _judge_schema():
    from google.genai import types

    text = lambda **kwargs: types.Schema(type="STRING", **kwargs)
    correction = types.Schema(
        type="OBJECT",
        properties={
            "topic_id": text(),
            "state": text(enum=["answered", "unknown", "no_response", "not_applicable"]),
            "value": types.Schema(
                any_of=[types.Schema(type="BOOLEAN"), types.Schema(type="STRING")],
                nullable=True,
            ),
            "details": types.Schema(
                type="OBJECT",
                nullable=True,
                properties={
                    "delivery_type": text(enum=list(contract.DELIVERY_TYPES), nullable=True),
                    "place": text(enum=list(contract.BIRTH_PLACES), nullable=True),
                },
            ),
            "evidence": text(),
        },
        required=["topic_id", "state", "evidence"],
    )
    return types.Schema(
        type="OBJECT",
        properties={
            "doctor_sections": types.Schema(
                type="OBJECT",
                properties={
                    heading.removeprefix("### ").strip(): text()
                    for heading in _REQUIRED_HEADINGS
                },
            ),
            "mother_questions": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={"topic_id": text(), "question": text()},
                    required=["topic_id", "question"],
                ),
            ),
            "extracted_findings": types.Schema(type="OBJECT"),
            "red_flags": types.Schema(type="ARRAY", items=text()),
            "topic_corrections": types.Schema(type="ARRAY", items=correction),
            "profile_corrections": types.Schema(
                type="OBJECT",
                properties={
                    "caller_name": text(nullable=True),
                    "marital_status": text(
                        enum=["married", "about_to_marry", "unmarried"], nullable=True
                    ),
                    "children": types.Schema(
                        type="ARRAY", items=types.Schema(type="OBJECT"), nullable=True
                    ),
                    "evidence": text(nullable=True),
                },
            ),
        },
        required=["doctor_sections", "mother_questions", "topic_corrections"],
    )


def _finalize_structured_summary(
    payload: dict | None,
    state: dict,
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
) -> dict:
    """Use the model JSON as the record. The deterministic draft is only the fallback."""
    if not isinstance(payload, dict):
        return build_state_fallback(state, findings, red_flags)
    sections = payload.get("doctor_sections")
    doctor = _markdown_from_sections(sections) if isinstance(sections, dict) else ""
    if not doctor.strip():
        doctor = str(payload.get("doctor_markdown") or payload.get("summary_doctor") or "").strip()
    if not doctor:
        return build_state_fallback(state, findings, red_flags)
    reconciled_state = reconcile_state_with_llm_payload(state, payload, transcript)
    judged_findings = _model_findings(payload, reconciled_state)
    judged_flags = _model_flags(payload, reconciled_state)
    doctor = _fill_red_flag_section(doctor, judged_flags)
    doctor = _fill_empty_sections(doctor, judged_findings)
    raw_mother = _mother_from_questions(payload) or str(
        payload.get("mother_hindi") or payload.get("questions_hindi") or payload.get("summary_mother") or ""
    ).strip()
    if isinstance(payload.get("questions_hindi"), list) and not payload.get("mother_hindi"):
        raw_mother = "\n".join(str(line) for line in payload["questions_hindi"] if str(line).strip())
    if transcript:
        _drop_ungrounded_name(judged_findings, transcript)
    mother = _ensure_mother_covers_flags(raw_mother, judged_findings, judged_flags)
    return {
        "summary_doctor": doctor,
        "summary_mother": mother,
        "findings": judged_findings,
        "red_flags": judged_flags,
        "state": reconciled_state,
    }


async def generate_call_summaries(
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
    state: dict | None = None,
) -> dict:
    """Return doctor and mother summaries.

    With call state, the frontier Gemini pass is the judge. If the key is unset,
    every model fails, or the JSON is invalid, the deterministic draft is used.
    """
    if state:
        user_content = state_to_summary_context(state, transcript=transcript)
        try:
            payload = await _from_gemini(
                user_content, STRUCTURED_STATE_SUMMARY_PROMPT, schema=_judge_schema()
            )
            return _finalize_structured_summary(payload, state, transcript, findings, red_flags)
        except Exception:
            return build_state_fallback(state, findings, red_flags)
    user_content = _user_content(transcript, findings, red_flags)
    try:
        payload = await _from_gemini(user_content)
        return _assemble(payload, findings, red_flags, transcript)
    except Exception:
        return build_fallback_summaries(transcript, findings, red_flags)
