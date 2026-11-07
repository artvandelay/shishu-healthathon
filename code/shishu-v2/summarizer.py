"""Post-call doctor and mother summaries."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

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
MOTHER_ROUTINE_FOOTER = (
    "नोट: इस बातचीत में कोई तुरंत चिंताजनक बात सामने नहीं आई है। "
    "ये सवाल आपकी अगली सामान्य विज़िट में डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से पूछने के लिए हैं।"
)

_DEFAULT_MOTHER_QUESTIONS: tuple[str, ...] = (
    "मेरे बच्चे की उम्र के हिसाब से उसका वज़न, लंबाई और सीखना-बढ़ना (बैठना, चलना, बोलना) ठीक चल रहा है या नहीं?",
    "बच्चे के सभी ज़रूरी टीके (टीकाकरण कार्ड के अनुसार, जैसे BCG) और सुनने-देखने की सामान्य जाँच पूरी है या कोई अगला टीका बाकी है?",
    "बच्चे के रोज़ के खान-पान और सेहत के लिए अभी मुझे किन बातों का ध्यान रखना चाहिए, और किन लक्षणों के दिखने पर तुरंत डॉक्टर या स्वास्थ्य कार्यकर्ता को दिखाना चाहिए?",
    "क्या परिवार या प्रेग्नेंसी की सामान्य सेहत के लिए खून की कमी (एनीमिया) या थैलेसीमिया जैसी किसी सामान्य जाँच की ज़रूरत है?",
)

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

    for domain, pattern, fallback_q in _FLAG_DOMAIN_QUESTIONS:
        entry = (findings or {}).get(domain)
        flagged_in_list = any(str(f).startswith(f"{domain}:") for f in (red_flags or []))
        is_flagged = (
            isinstance(entry, dict)
            and bool(entry.get("is_red_flag"))
            and not _unknown_detail(str(entry.get("detail") or ""))
        ) or flagged_in_list
        if not is_flagged:
            continue
        if not pattern.search(combined_q):
            questions.append(fallback_q)
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


async def _from_gemini(user_content: str) -> dict:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config={
                "system_instruction": DOCTOR_MOTHER_SUMMARY_PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 8192,
            },
        )
        return _parse_json(response.text or "")
    finally:
        await client.aio.aclose()
        client.close()


async def _from_openai(user_content: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": DOCTOR_MOTHER_SUMMARY_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content or ""
        return _parse_json(text)
    finally:
        await client.close()


async def generate_call_summaries(
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
) -> dict:
    """Return doctor and mother summaries, with findings and red flags merged."""
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
