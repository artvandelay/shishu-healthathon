"""Post-call doctor and mother summaries."""

import copy
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

import checklist
import call_state
from prompts import DOCTOR_MOTHER_SUMMARY_PROMPT

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

# Newest public Gemini the summary key should call, then the next preview, then 2.5 pro.
# There is no gemini-4 slug. Flash is not the post-call judge.
SUMMARY_MODEL_CHAIN: tuple[str, ...] = (
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
)
os.environ.setdefault("GEMINI_SUMMARY_MODEL", SUMMARY_MODEL_CHAIN[0])

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


_UNKNOWN_RE = re.compile(
    r"\b(unknown|does not know|does not recall|don't know|do not know|not sure|not reported|not discussed)\b",
    re.I,
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
    return bool(_UNKNOWN_RE.search(detail or ""))


def _not_reported(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    detail = str(value.get("detail") or "").strip().lower()
    return (not detail) or "not reported" in detail or detail in {"not discussed", "none", "n/a"}


def _add_flag(flags: list[str], domain: str, detail: str) -> None:
    text = f"{domain}: {detail}".strip()
    if text and text not in flags:
        flags.append(text)


def _supplement_from_transcript(
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
) -> None:
    """Keep stated gates when the model drops them from red_flags."""
    text = _user_text(transcript)
    lowered = text.lower()

    def put(domain: str, detail: str, status: str, is_red_flag: bool) -> None:
        current = findings.get(domain)
        if not isinstance(current, dict) or _not_reported(current):
            findings[domain] = {
                "status": status,
                "detail": detail,
                "is_red_flag": is_red_flag,
            }
        elif is_red_flag:
            current["status"] = "positive"
            current["is_red_flag"] = True
            if not str(current.get("detail") or "").strip():
                current["detail"] = detail
        if is_red_flag:
            detail_text = str(findings[domain].get("detail") or detail)
            _add_flag(red_flags, domain, detail_text)

    if re.search(r"\bbeta\b", lowered) or "बेटा" in text:
        put("profile_child_gender", "Male. Mother said beta.", "info", False)
    elif re.search(r"\bbeti\b", lowered) or "बेटी" in text:
        put("profile_child_gender", "Female. Mother said beti.", "info", False)

    knee = re.search(
        r"(ghutn\w*|knee|joint).{0,48}(sujan|soojan|sujan|swelling)"
        r"|(sujan|soojan|sujan|swelling).{0,48}(ghutn\w*|knee|joint)",
        lowered,
    )
    if knee:
        put(
            "bleeding_joints",
            "Knee or joint swelling after a fall, resolving over days.",
            "positive",
            True,
        )

    infant_death = re.search(
        r"guzar gaya|guzar gaye|guzar gayi|mar gaya|mar gaye|mrityu|passed away",
        lowered,
    )
    young = re.search(r"mahine|month|saal se kam|under 1|1 saal", lowered)
    if infant_death and young:
        put(
            "family_infant_death",
            "A previous child died in infancy. Fits were mentioned."
            if re.search(r"fits|jhatke", lowered)
            else "A previous child died in infancy.",
            "positive",
            True,
        )

    admissions = re.search(
        r"(hospital|bharti|भर्ती).{0,48}(baar|बार|teen|three|six|chhe|mahine|month)"
        r"|(baar baar|बार-बार|repeated).{0,40}(hospital|bharti|भर्ती|admission)",
        lowered,
    )
    antibiotics = re.search(r"antibiotic|drip se|iv antibiotic", lowered)
    infections = findings.get("infections")
    infections_detail = str(infections.get("detail") or "") if isinstance(infections, dict) else ""
    if (admissions or antibiotics) and (
        not isinstance(infections, dict)
        or _not_reported(infections)
        or _unknown_detail(infections_detail)
    ):
        findings["infections"] = {
            "status": "positive",
            "detail": "Repeated hospital stays or IV antibiotics were mentioned.",
            "is_red_flag": True,
        }
        _add_flag(
            red_flags,
            "infections",
            "Repeated hospital stays or IV antibiotics were mentioned.",
        )


def _fill_red_flag_section(doctor: str, red_flags: list[str]) -> str:
    if not red_flags:
        return doctor
    pattern = (
        r"(### Positive Red Flags & NRROID Gate Alerts\n)"
        r"(.*?)(?=\n### )"
    )
    match = re.search(pattern, doctor, flags=re.S)
    if not match:
        return doctor
    body = match.group(2)
    if "not reported" not in body.lower():
        return doctor
    lines = "\n".join(f"- {flag}" for flag in red_flags)
    return doctor[: match.start(2)] + lines + "\n" + doctor[match.end(2) :]


def _similar_detail(left: str, right: str) -> bool:
    a = (left or "").strip().lower()
    b = (right or "").strip().lower()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ta = set(re.findall(r"[a-z0-9]{4,}", a))
    tb = set(re.findall(r"[a-z0-9]{4,}", b))
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.6


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
        pattern = rf"({re.escape(heading)}\n)(.*?)(?=\n### )"
        match = re.search(pattern, doctor, flags=re.S)
        if not match:
            continue
        body = re.sub(r"^[\-\*\s]+", "", match.group(2).strip()).lower().rstrip(".")
        if body != "not reported":
            continue
        doctor = doctor[: match.start(2)] + "\n".join(bullets) + "\n" + doctor[match.end(2) :]
    return doctor


_MOTHER_NAME_SENTENCE_RE = re.compile(
    r"^\s*मेरा\s+नाम\s+([^\s।,]+(?:\s+[^\s।,]+)?)\s+है\s*[।,.!?]*\s*"
)

MOTHER_QUESTIONS_HEADING = (
    "अगली बार जब आप डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से मिलें, तो ये सवाल ज़रूर पूछें:"
)
MOTHER_FLAG_FOOTER = (
    "नोट: ये सवाल आपकी अगली विज़िट में डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से चर्चा के लिए हैं। "
    "यह किसी बीमारी की पुष्टि (डायग्नोसिस) नहीं है — केवल डॉक्टर ही जाँच करके सही सलाह दे सकते हैं।"
)
MOTHER_ROUTINE_FOOTER = MOTHER_FLAG_FOOTER

_DEFAULT_MOTHER_QUESTIONS: tuple[str, ...] = (
    "मेरे बच्चे की उम्र के हिसाब से उसका वज़न, लंबाई और सीखना-बढ़ना (बैठना, चलना, बोलना) ठीक चल रहा है या नहीं?",
    "बच्चे के सभी ज़रूरी टीके (टीकाकरण कार्ड के अनुसार, जैसे BCG) और सुनने-देखने की सामान्य जाँच पूरी है या कोई अगला टीका बाकी है?",
    "बच्चे के रोज़ के खान-पान और सेहत के लिए अभी मुझे किन बातों का ध्यान रखना चाहिए, और किन लक्षणों के दिखने पर तुरंत डॉक्टर या स्वास्थ्य कार्यकर्ता को दिखाना चाहिए?",
    "क्या परिवार या प्रेग्नेंसी की सामान्य सेहत के लिए खून की कमी (एनीमिया) या थैलेसीमिया जैसी किसी सामान्य जाँच की ज़रूरत है?",
)

# A domain sentence that names one event must not be reused for a sibling topic.
# These topics get their own question; the shared domain sentence stays for the rest.
_TOPIC_MOTHER_QUESTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "birth_mode",
        re.compile(r"ऑपरेशन|प्रसव", re.I),
        "प्रसव ऑपरेशन से हुआ था — क्या इस बारे में डॉक्टर से कोई जाँच या सलाह पूछनी चाहिए?",
    ),
    (
        "birth_term",
        re.compile(r"समय से पहले|समय के बाद", re.I),
        "बच्चा समय से पहले या समय के बाद पैदा हुआ था — क्या इस कारण अभी कोई जाँच ज़रूरी है?",
    ),
    (
        "body_fits_milestones",
        re.compile(r"झटके|दौरे", re.I),
        "बच्चे को झटके आते हैं, या बैठना-चलना-बोलना देर से हुआ — क्या इसके लिए कोई जाँच करवानी चाहिए?",
    ),
    (
        "body_transfusion",
        re.compile(r"खून चढ़", re.I),
        "बच्चे को खून चढ़ाना पड़ा था — क्या आगे खून या जाँच के बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "fam_miscarriage",
        re.compile(r"गर्भ गिर", re.I),
        "गर्भ गिरा था — क्या अगली गर्भावस्था के लिए कोई सामान्य जाँच या सलाह लेनी चाहिए?",
    ),
    (
        "birth_cord_bleed",
        re.compile(r"नाल गिरने पर", re.I),
        "नाल गिरने पर खून ज़्यादा हुआ था — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "birth_cord_3weeks",
        re.compile(r"तीन हफ़्ते|तीन हफ्ते", re.I),
        "नाल गिरने में तीन हफ़्ते से ज़्यादा लगे — क्या इसकी जाँच ज़रूरी है?",
    ),
    (
        "birth_bcg",
        re.compile(r"BCG|टीके की जगह", re.I),
        "BCG के टीके की जगह पर सूजन या निशान है — क्या इसे डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "birth_cleft",
        re.compile(r"होंठ या तालु", re.I),
        "होंठ या तालु कटने की बात है — क्या इसके लिए कोई जाँच या सलाह लेनी चाहिए?",
    ),
    (
        "birth_hearing_vision",
        re.compile(r"सुनने या देखने", re.I),
        "सुनने या देखने की जाँच में बात आई थी — क्या आगे कोई जाँच ज़रूरी है?",
    ),
    (
        "body_blue",
        re.compile(r"होंठ या उँगल", re.I),
        "होंठ या उँगलियाँ नीली होती हैं — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "body_tummy",
        re.compile(r"पेट धीरे", re.I),
        "पेट धीरे-धीरे बढ़ रहा है — क्या लिवर या तिल्ली की जाँच पूछनी चाहिए?",
    ),
    (
        "body_blood_count",
        re.compile(r"खून की गिनती", re.I),
        "खून की गिनती कम या ज़्यादा बताई गई — क्या आगे जाँच करवानी चाहिए?",
    ),
    (
        "preg_thalassemia",
        re.compile(r"थैलेसीमिया", re.I),
        "थैलेसीमिया की जाँच में बात आई थी — क्या रिपोर्ट डॉक्टर को दिखानी चाहिए?",
    ),
    (
        "preg_scan",
        re.compile(r"स्कैन", re.I),
        "प्रेग्नेंसी के स्कैन में बात आई थी — क्या उसे डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "preg_heart",
        re.compile(r"दिल की", re.I),
        "प्रेग्नेंसी या जन्म पर दिल की बात आई थी — क्या बच्चे के दिल की जाँच पूछनी चाहिए?",
    ),
    (
        "preg_transfusion",
        re.compile(r"प्रेग्नेंसी में खून", re.I),
        "प्रेग्नेंसी में खून चढ़ाना पड़ा था — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "preg_delivery",
        re.compile(r"प्रसव में दिक्कत", re.I),
        "प्रसव में दिक्कत हुई थी — क्या माँ की आगे की जाँच ज़रूरी है?",
    ),
    (
        "ill_odor",
        re.compile(r"गंध", re.I),
        "पेशाब या मल की गंध अलग है — क्या इस बारे में जाँच पूछनी चाहिए?",
    ),
    (
        "ill_sudden",
        re.compile(r"अचानक बहुत", re.I),
        "जन्म के बाद अचानक बहुत बीमार हुए थे — क्या इसके कारण की जाँच ज़रूरी है?",
    ),
    (
        "body_bleeding_bruise",
        re.compile(r"चोट पर खून|नील पड़", re.I),
        "चोट पर खून देर तक बहता है, या आसानी से नील पड़ते हैं — क्या इसकी जाँच करवानी चाहिए?",
    ),
    (
        "body_limb_pain",
        re.compile(r"हाथ या पैर में दर्द", re.I),
        "हाथ या पैर में बार-बार दर्द होता है — क्या इस बारे में डॉक्टर से पूछना चाहिए?",
    ),
    (
        "body_regression",
        re.compile(r"सीखा हुआ", re.I),
        "जो काम बच्चा पहले कर लेता था, अब नहीं कर पाता — क्या इसकी जाँच ज़रूरी है?",
    ),
    (
        "body_private_parts",
        re.compile(r"गुप्त अंग", re.I),
        "गुप्त अंग जैसे होने चाहिए वैसे नहीं दिखते — क्या डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "body_lumps_birth",
        re.compile(r"जन्म से सूजन", re.I),
        "जन्म से शरीर पर सूजन या गाँठ है — क्या इसे डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "body_proportion",
        re.compile(r"हाथ-पैर छोटे", re.I),
        "हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं — क्या इसकी जाँच पूछनी चाहिए?",
    ),
    (
        "body_thumb",
        re.compile(r"अंगूठा", re.I),
        "अंगूठा या उँगली छोटी है या नहीं है — क्या डॉक्टर को दिखाना चाहिए?",
    ),
    (
        "body_puberty",
        re.compile(r"बढ़ती उम्र", re.I),
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
        for topic_id in checklist._BY_ID:
            if topic_id in blob:
                found.add(topic_id)
    return found


_FLAG_DOMAIN_QUESTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "family_infant_death",
        re.compile(r"गुजर|मृत्यु|guzar|died|death", re.I),
        "हमारे परिवार में पहले एक बच्चा छोटी उम्र में गुजर गया था — क्या इस बच्चे की सेहत या आगे की देखभाल के लिए हमें डॉक्टर से कोई विशेष सलाह या जाँच पूछनी चाहिए?",
    ),
    (
        "bleeding_joints",
        re.compile(r"घुटने|जोड़|सूजन|खून|नील|sujan|ghutn|bleed|transfusion", re.I),
        "बच्चे के गिरने पर घुटने या जोड़ में सूजन आने (या खून/नील से जुड़ी बात) के बारे में कृपया देखकर बताएँ कि क्या कोई जाँच करवानी चाहिए?",
    ),
    (
        "infections",
        re.compile(r"भर्ती|अस्पताल|इंफेक्शन|संक्रमण|बुखार|एंटीबायोटिक|निमोनिया|hospital|infection", re.I),
        "बच्चे को बार-बार बुखार या इंफेक्शन के कारण अस्पताल में भर्ती होना पड़ा है — इससे बचाव के लिए हमें आगे क्या जाँच या सावधानी रखनी चाहिए?",
    ),
    (
        "birth_nicu",
        re.compile(r"पीलिया|नीली|रोशनी|NICU|काँच|कांच|टीके|BCG|नाल|गाँठ|गांठ|मवाद", re.I),
        "जन्म के समय या BCG के टीके के बाद जो बात हुई थी (जैसे पीलिया की नीली रोशनी, NICU, नाल, या टीके की जगह पर सूजन/निशान) — क्या उसके लिए अभी बच्चे की कोई जाँच ज़रूरी है?",
    ),
    (
        "milestones",
        re.compile(r"बैठ|चल|बोल|वज़न|वजन|लंबाई|पेट|हड्डी|देखने|सुनने|बढ़त", re.I),
        "बच्चे के बढ़ने, बैठने-चलने-बोलने, पेट के आकार या देखने-सुनने को जाँच कर कृपया बताएँ कि क्या किसी अतिरिक्त जाँच की ज़रूरत है?",
    ),
    (
        "metabolic_fasting",
        re.compile(r"सुस्ती|उल्टी|झटके|दौरे|पेशाब|गंध|भूख", re.I),
        "खाना देर से मिलने या हल्के बुखार में बच्चे को सुस्ती, उल्टी या झटके जैसी जो दिक्कत हुई थी — ऐसे समय में हमें क्या करना चाहिए और क्या कोई जाँच ज़रूरी है?",
    ),
    (
        "pregnancy",
        re.compile(r"प्रेग्नेंसी|गर्भ|बीपी|BP|थायरॉइड|शुगर|स्कैन|थैलेसीमिया", re.I),
        "प्रेग्नेंसी के समय रही बीपी, शुगर या थायरॉइड जैसी बात के लिए अभी माँ और बच्चे की कौन-सी नियमित जाँच करवानी चाहिए?",
    ),
    (
        "consanguinity",
        re.compile(r"रिश्तेदार|शादी", re.I),
        "हमारी शादी रिश्तेदारी में हुई है — क्या बच्चे की सेहत के लिए डॉक्टर से कोई सामान्य परामर्श (काउंसलिंग) लेना अच्छा रहेगा?",
    ),
    (
        "family_history",
        re.compile(r"परिवार|खून|दौरे|झटके", re.I),
        "परिवार में पहले जो सेहत से जुड़ी समस्या रही है, उसे ध्यान में रखते हुए क्या बच्चे के लिए कोई सामान्य जाँच पूछनी चाहिए?",
    ),
)

_FORBIDDEN_DIAGNOSTIC_RE = re.compile(
    r"(?i)\b(duchenne|pompe|wiskott[- ]?aldrich|mucopolysaccharidosis|mps\b|gaucher|"
    r"hemophilia|haemophilia|primary immunodeficiency|scid\b|spinal muscular atrophy)\b|"
    r"हीमोफीलिया|इम्यूनोडेफिशिएंसी|ड्यूशेन|पॉम्पे|म्यूकोपॉलीसेकेराइडोसिस"
)
_REPEAT_BACK_PREFIX_RE = re.compile(
    r"^\s*(?:आपने\s+(?:यह\s+भी\s+)?बताया(?:\s+कि)?|मेरा\s+नाम\s+.*?\s+है|मेरे\s+एक\s+बच्चा\s+है)"
)


def _scrub_diagnostic_terms(text: str) -> str:
    if not text:
        return ""
    cleaned = _FORBIDDEN_DIAGNOSTIC_RE.sub("स्वास्थ्य संबंधी जाँच", text)
    return cleaned.replace("गुज़र", "गुजर")


def _strip_unspoken_mother_name(
    mother: str,
    doctor: str,
    merged_findings: dict,
    transcript: list[dict],
) -> tuple[str, str]:
    user_blob = _user_text(transcript).strip()
    user_lower = user_blob.lower()
    recorded = merged_findings.get("profile_mother_name")
    recorded_detail = str(recorded.get("detail") or "").strip() if isinstance(recorded, dict) else ""
    if _unknown_detail(recorded_detail) or "shilpa" in recorded_detail.lower() and "shilpa" not in user_lower and "शिल्पा" not in user_blob:
        merged_findings.pop("profile_mother_name", None)
        recorded_detail = ""
    match = _MOTHER_NAME_SENTENCE_RE.match(mother or "")
    if match:
        claimed = match.group(1).strip()
        if not recorded_detail and (claimed not in user_blob or user_blob in {"मेरा नाम", "मेरा नाम।"}):
            mother = (mother or "")[match.end():].strip()
        elif claimed == "शिल्पा" and "शिल्पा" not in user_blob and "shilpa" not in user_lower:
            mother = (mother or "")[match.end():].strip()
    if "shilpa" not in user_lower and "शिल्पा" not in user_blob:
        doctor = re.sub(r"(?i)\bshilpa\b", "Not reported", doctor or "")
        mother = re.sub(r"शिल्पा\s*(?:जी)?[,،:\s]*", "", mother or "").strip()
    return mother, doctor


def _extract_question_lines(text: str) -> list[str]:
    questions: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^(?:[\d१२३४५६७८९]+[\.\)\-:]|[•\-\*])\s*", "", line).strip()
        if not cleaned or _REPEAT_BACK_PREFIX_RE.match(cleaned):
            continue
        if "?" in cleaned or "？" in cleaned:
            cleaned = _scrub_diagnostic_terms(cleaned)
            if cleaned not in questions:
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
) -> str:
    text = _scrub_diagnostic_terms((mother or "").strip())
    questions = _extract_question_lines(text)
    combined_q = "\n".join(questions)

    flagged_topics = _flagged_topic_ids(findings, red_flags)
    for domain, pattern, fallback_q in _FLAG_DOMAIN_QUESTIONS:
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
        if not pattern.search(combined_q):
            questions.append(fallback_q)
            combined_q = "\n".join(questions)

    for topic_id, pattern, question in _TOPIC_MOTHER_QUESTIONS:
        if topic_id not in flagged_topics:
            continue
        if pattern.search(combined_q):
            continue
        questions.append(question)
        combined_q = "\n".join(questions)

    for default_q in _DEFAULT_MOTHER_QUESTIONS:
        if len(questions) >= 3:
            break
        if default_q not in questions:
            questions.append(default_q)

    numbered = "\n".join(f"{idx}. {q}" for idx, q in enumerate(questions[:6], start=1))
    footer = (
        MOTHER_FLAG_FOOTER
        if _has_any_red_flag(findings, red_flags)
        else MOTHER_ROUTINE_FOOTER
    )
    return f"{MOTHER_QUESTIONS_HEADING}\n\n{numbered}\n\n{footer}"


def default_mother_questions(
    findings: dict | None = None,
    red_flags: list[str] | None = None,
) -> str:
    """Build actionable Hindi questions for the mother even when no LLM summary ran."""
    return _ensure_mother_covers_flags("", findings or {}, red_flags or [])


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
    _supplement_from_transcript(transcript, merged_findings, merged_flags)
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
    _supplement_from_transcript(transcript, merged_findings, merged_flags)
    merged_flags = _reconcile_flags(merged_findings, merged_flags)
    doctor = _fill_red_flag_section(cleaned["doctor_markdown"], merged_flags)
    doctor = _fill_empty_sections(doctor, merged_findings)
    mother_raw, doctor = _strip_unspoken_mother_name(
        cleaned["mother_hindi"], doctor, merged_findings, transcript
    )
    mother = _ensure_mother_covers_flags(mother_raw, merged_findings, merged_flags)
    if merged_findings.get("profile_child_gender", {}).get("detail", "").startswith("Male"):
        doctor = doctor.replace(
            "**Child's Gender:** Not reported",
            "**Child's Gender:** Male (mother said beta)",
        )
    elif merged_findings.get("profile_child_gender", {}).get("detail", "").startswith("Female"):
        doctor = doctor.replace(
            "**Child's Gender:** Not reported",
            "**Child's Gender:** Female (mother said beti)",
        )
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
- You are the judge of the transcript. When the deterministic draft was too rigid, misheard, or misclassified, override it: set `topic_corrections` to the corrected state and omit that false positive from `extracted_findings` and `red_flags`. A deterministic yes is a draft, not a fact you must repeat.
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

Return one JSON object and nothing else. Keys must be:
{"doctor_markdown": "...", "mother_hindi": "...", "extracted_findings": {}, "red_flags": [], "topic_corrections": {}, "profile_corrections": {}}
""".strip()


def _answer_label(topic: dict, topic_id: str | None = None) -> str:
    return call_state.topic_answer_kind(topic, topic_id)


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
        label = _answer_label(saved, tid)
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


def summary_model_chain(explicit: str | None = None) -> list[str]:
    """Models to try, newest first, then down the list.

    An explicit `GEMINI_SUMMARY_MODEL` starts at that slug and continues with
    the later entries. An unknown slug is tried once, then the real chain, so
    a retired name is never the only option.
    """
    preferred = (explicit if explicit is not None else os.environ.get("GEMINI_SUMMARY_MODEL") or "").strip()
    if not preferred:
        preferred = SUMMARY_MODEL_CHAIN[0]
    if preferred in SUMMARY_MODEL_CHAIN:
        start = SUMMARY_MODEL_CHAIN.index(preferred)
        return list(SUMMARY_MODEL_CHAIN[start:])
    return [preferred, *SUMMARY_MODEL_CHAIN]


def _mentioned(text: str, transcript_blob: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return cleaned.lower() in transcript_blob or cleaned in transcript_blob


def _apply_topic_correction(
    topic: dict,
    new_state: str,
    new_val: object,
    new_details: dict,
    new_instances: list,
    *,
    replace: bool,
) -> None:
    """Write a model correction. `replace` drops draft details that would veto it."""
    if new_state == "answered":
        topic["state"] = "answered"
        topic["value"] = new_val if new_val is not None else (False if replace else True)
        if replace:
            topic["details"] = dict(new_details)
        elif new_details:
            merged = dict(topic.get("details") or {})
            merged.update(new_details)
            topic["details"] = merged
        if new_instances:
            topic["instances"] = [item for item in new_instances if isinstance(item, dict)]
        elif replace and not call_state._is_positive(topic):
            topic["instances"] = []
    elif new_state in {"unknown", "no_response", "not_asked"}:
        topic["state"] = new_state
        topic["value"] = None
        topic["details"] = dict(new_details)
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
        mstat = call_state._canonical_marital(prof_corr.get("marital_status"))
        if mstat:
            prof["marital_status"] = mstat
            reconciled["topics"]["intro_married"]["state"] = "answered"
            reconciled["topics"]["intro_married"]["value"] = mstat
        if isinstance(prof_corr.get("children"), list) and prof_corr["children"]:
            norm_kids = call_state._normalize_children(prof_corr["children"])
            if norm_kids:
                prof["children"] = norm_kids
                reconciled["topics"]["intro_children"]["state"] = "answered"
                reconciled["topics"]["intro_children"]["value"] = True

    top_corr = payload.get("topic_corrections")
    if isinstance(top_corr, dict):
        topics = reconciled.setdefault("topics", {})
        for tid, patch in top_corr.items():
            if tid not in checklist._BY_ID or not isinstance(patch, dict):
                continue
            cur = topics.get(tid)
            if not isinstance(cur, dict):
                continue
            new_state = str(patch.get("state") or "").strip()
            new_val = patch.get("value")
            new_details = patch.get("details") if isinstance(patch.get("details"), dict) else {}
            new_instances = patch.get("instances") if isinstance(patch.get("instances"), list) else []
            prior = cur.get("state")
            if prior == "answered":
                if user_blob and new_state in _CORRECTION_STATES:
                    _apply_topic_correction(
                        cur, new_state, new_val, new_details, new_instances, replace=True
                    )
                elif new_details or (new_instances and not cur.get("instances")):
                    _apply_topic_correction(
                        cur, "answered", cur.get("value"), new_details, new_instances, replace=False
                    )
                continue
            if prior == "not_applicable":
                if user_blob and new_state in {"answered", "unknown"}:
                    _apply_topic_correction(
                        cur, new_state, new_val, new_details, new_instances, replace=True
                    )
                continue
            if not user_blob or new_state not in _CORRECTION_STATES:
                continue
            candidate = {"state": new_state, "value": new_val, "details": new_details}
            if new_state == "answered" and call_state._is_positive(candidate):
                _apply_topic_correction(
                    cur, new_state, new_val, new_details, new_instances, replace=False
                )
            elif new_state == "unknown" and prior in {"not_asked", "no_response"}:
                _apply_topic_correction(
                    cur, new_state, None, new_details, new_instances, replace=True
                )
            elif new_state == "answered" and new_val is False and prior in {"not_asked", "no_response"}:
                evidence = str(patch.get("evidence") or "").strip().lower()
                if len(evidence) >= 6 and evidence in user_blob:
                    _apply_topic_correction(
                        cur, new_state, False, new_details, new_instances, replace=True
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
            f"{topic['id']}: state={_answer_label(saved, topic['id'])} "
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
        and _answer_label(
            topics.get(topic["id"]) if isinstance(topics.get(topic["id"]), dict) else {},
            topic["id"],
        ) == "no"
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
        label = _answer_label(saved, topic["id"])
        lines.append(f"- {topic['id']}: {label}")

    return {
        "summary_doctor": "\n".join(lines),
        "summary_mother": default_mother_questions(findings, red_flags),
        "findings": findings,
        "red_flags": red_flags,
        "state": state,
    }


async def _generate_accepted_model(client, models: list[str], user_content: str, system_prompt: str) -> dict:
    """Call models in order. The first response the API accepts and that parses wins."""
    last_error: Exception | None = None
    for chosen in models:
        try:
            response = await client.aio.models.generate_content(
                model=chosen,
                contents=user_content,
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                    "max_output_tokens": 8192,
                },
            )
            return _parse_json(response.text or "")
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("no summary model accepted")


async def _from_gemini(
    user_content: str,
    system_prompt: str = DOCTOR_MOTHER_SUMMARY_PROMPT,
    model: str | None = None,
) -> dict:
    from google import genai

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is unset")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        return await _generate_accepted_model(
            client, summary_model_chain(model), user_content, system_prompt
        )
    finally:
        await client.aio.aclose()
        client.close()


async def _from_openai(user_content: str, system_prompt: str = DOCTOR_MOTHER_SUMMARY_PROMPT) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content or ""
        return _parse_json(text)
    finally:
        await client.close()


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
    doctor = str(payload.get("doctor_markdown") or payload.get("summary_doctor") or "").strip()
    if not doctor:
        return build_state_fallback(state, findings, red_flags)
    reconciled_state = reconcile_state_with_llm_payload(state, payload, transcript)
    judged_findings = _model_findings(payload, reconciled_state)
    judged_flags = _model_flags(payload, reconciled_state)
    doctor = _fill_red_flag_section(doctor, judged_flags)
    doctor = _fill_empty_sections(doctor, judged_findings)
    raw_mother = str(
        payload.get("mother_hindi") or payload.get("questions_hindi") or payload.get("summary_mother") or ""
    ).strip()
    if isinstance(payload.get("questions_hindi"), list) and not payload.get("mother_hindi"):
        raw_mother = "\n".join(str(line) for line in payload["questions_hindi"] if str(line).strip())
    if transcript:
        raw_mother, doctor = _strip_unspoken_mother_name(raw_mother, doctor, judged_findings, transcript)
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
            payload = await _from_gemini(user_content, STRUCTURED_STATE_SUMMARY_PROMPT)
            return _finalize_structured_summary(payload, state, transcript, findings, red_flags)
        except Exception:
            return build_state_fallback(state, findings, red_flags)
    user_content = _user_content(transcript, findings, red_flags)
    try:
        payload = await _from_gemini(user_content)
        return _assemble(payload, findings, red_flags, transcript)
    except Exception:
        try:
            payload = await _from_openai(user_content)
            return _assemble(payload, findings, red_flags, transcript)
        except Exception:
            return build_fallback_summaries(transcript, findings, red_flags)
