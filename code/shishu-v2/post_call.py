"""Deterministic post-call builder for the mother PDF and clinician note."""

from __future__ import annotations

import json
import re

from triage_catalog import active_specs

_HEADINGS = (
    "### Patient Profile & Chief Complaint",
    "### Positive Red Flags & NRROID Gate Alerts",
    "### Birth & Neonatal History",
    "### Development, Growth & Physical Features",
    "### Systemic, Infections & Hematology",
    "### Antenatal (Pregnancy) History",
    "### Family & Pedigree History",
    "### Pertinent Negatives",
)
_POSITIVE_HEADING = _HEADINGS[1]

_SECTION = {
    "birth": _HEADINGS[2], "body": _HEADINGS[3], "illness": _HEADINGS[4],
    "pregnancy": _HEADINGS[5], "family": _HEADINGS[6], "close": _HEADINGS[0],
}

_HEMATOLOGY = frozenset({
    "body_bleeding_bruise", "body_joint_swelling", "mod_tooth_bleed",
    "mod_heavy_periods", "body_transfusion", "body_blood_count", "body_limb_pain",
})

_ROUTINE_HI = (
    "क्या मेरे बच्चे का वज़न, कद और बैठना-चलना-बोलना उम्र के हिसाब से ठीक है?",
    "क्या टीके के कार्ड के सारे टीके पूरे हैं?",
    "घर पर खान-पान और किन चेतावनी की बातों पर ध्यान रखूँ?",
)

_ROUTINE_CHILDLESS_HI = (
    "आगे प्रेग्नेंसी या स्वास्थ्य योजना से पहले मुझे कौन सी ज़रूरी जाँच करवानी चाहिए?",
    "परिवार की सेहत को देखते हुए डॉक्टर से किन बातों पर सलाह लेनी चाहिए?",
    "खान-पान, आयरन-फोलिक एसिड और किन बातों का अभी से ध्यान रखूँ?",
)

_EMERGENCY_HI = (
    "इमरजेंसी डॉक्टर से अभी पूछूँ कि यह खतरे की निशानी कितनी गंभीर है "
    "और मुझे अस्पताल या 108 पर क्या करना है?",
    "अभी साँस और हालत को स्थिर करने के लिए डॉक्टर कौन सी जाँच तुरंत करेंगे?",
    "अस्पताल पहुँचकर डॉक्टर को कौन सी चेतावनी की बातें और पहले की बीमारी बतानी चाहिए?",
)

_EMERGENCY_FLAG = "emergency: active danger sign now; hospital or 108"
_EMERGENCY_BULLET = (
    "- Active emergency. Caller was told to go to the nearest hospital or call 108."
)

_DISEASE = re.compile(
    r"(?:\b(?:Duchenne|Pompe|Wiskott[\s-]*Aldrich|Hemophilia|Haemophilia"
    r"|Gaucher|SCID|SMA|MPS|Fanconi|PKU)\b"
    r"|Sickle[\s-]*Cell|Cystic[\s-]*Fibrosis"
    r"|ड्यूशेन|हीमोफीलिया|पॉम्पे|गॉशर|इम्यूनोडेफिशिएंसी|म्यूकोपॉलीसेकेराइडोसिस"
    r"|सिकल\s*सेल|सिस्टिक\s*फाइब्रोसिस|फैंकोनी|थैलेसीमिया मेजर)",
    re.IGNORECASE,
)
_SEX_HI = {"male": "बेटा", "female": "बेटी"}


def cursor_of(call: dict | None) -> dict:
    if not isinstance(call, dict):
        return {}
    for key in ("cursor_state", "cursor_json"):
        raw = call.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict) and raw.get("active_qid"):
            return raw
    return {}


def has_cursor(call: dict | None) -> bool:
    return bool(cursor_of(call))


def _scrub_diagnostic_terms(text: str) -> str:
    cleaned = _DISEASE.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([।?])", r"\1", cleaned)
    return cleaned.strip()


def _scrub_unique(items) -> list[str]:
    out: list[str] = []
    for item in items:
        cleaned = _scrub_diagnostic_terms(str(item or "").strip())
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _section_for(spec) -> str:
    return _HEADINGS[4] if spec.id in _HEMATOLOGY else _SECTION.get(spec.group, _HEADINGS[-1])


def _children_label(profile: dict) -> str:
    bits = []
    for child in profile.get("children") or []:
        sex = _SEX_HI.get(child.get("sex") or "", "")
        age = str(child.get("age_label_hi") or "").strip()
        name = str(child.get("name") or "").strip()
        bits.append(" ".join(part for part in (name, age, sex) if part))
    return " · ".join(bit for bit in bits if bit)


def _human_notes(entry: dict) -> str:
    """Mother's words only: no follow-up ids and no clear/concern/unknown labels."""
    if not isinstance(entry, dict):
        return ""
    raw_parts: list[object] = [entry.get("notes")]
    followups = entry.get("followups")
    if isinstance(followups, dict):
        for val in followups.values():
            if isinstance(val, (dict, str)):
                raw_parts.append(val.get("note") if isinstance(val, dict) else val)
    parts: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        cleaned = " ".join(str(raw or "").split())
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            parts.append(cleaned)
    return "; ".join(parts)


def _question_stem(spec) -> str:
    body = spec.question_hi.strip().rstrip("?")
    return body[4:] if body.startswith("क्या ") else body


def _concern_sentence(spec, entry: dict) -> str:
    note = _human_notes(entry)
    stem = _question_stem(spec)
    return f"{stem} — {note}।" if note else f"{stem}।"


def _concern_question(spec, entry: dict) -> str:
    note = _human_notes(entry)
    if note:
        return f"मैंने बताया कि {note}। डॉक्टर, इसकी जाँच क्या होनी चाहिए?"
    return f"मैंने हाँ कहा कि {_question_stem(spec)}। डॉक्टर, अब क्या जाँच करनी चाहिए?"


def _fit_questions(
    questions: list[str],
    *,
    emergency: bool = False,
    has_children: bool = True,
    concern_questions: list[str] | None = None,
) -> list[str]:
    if emergency and not concern_questions:
        return list(_EMERGENCY_HI)
    routine = _ROUTINE_HI if has_children else _ROUTINE_CHILDLESS_HI
    kept: list[str] = []
    for source in (questions, concern_questions or ()):
        for raw in source:
            if len(kept) >= 6:
                break
            text = str(raw or "").strip()
            if not text or _DISEASE.search(text):
                continue
            cleaned = _scrub_diagnostic_terms(text)
            if cleaned and cleaned not in kept and not (
                emergency and (cleaned in _ROUTINE_HI or cleaned in _ROUTINE_CHILDLESS_HI)
            ):
                kept.append(cleaned)
        if kept:
            break
    for filler in (_EMERGENCY_HI if emergency else routine):
        if len(kept) >= 3:
            break
        if filler not in kept:
            kept.append(filler)
    return kept[:6]


def _rows(state: dict) -> list[tuple]:
    answers = state.get("answers") or {}
    rows = []
    for spec in active_specs():
        entry = answers.get(spec.id)
        if spec.kind in ("profile", "bridge") or not isinstance(entry, dict):
            continue
        status = entry.get("state") or ""
        if status not in ("", "not_applicable"):
            rows.append((spec, entry, status))
    return rows


def _flag_for(spec, entry: dict) -> str:
    note = _human_notes(entry)
    head = f"{spec.nrroid_gate} {spec.id}" if spec.nrroid_gate else spec.id
    flag = f"{head}: {spec.need_en}"
    return f"{flag}. {note}" if note else flag


def build_reports(call: dict) -> dict:
    """Both reports from the cursor alone. No disease names, no transliteration."""
    state = cursor_of(call)
    profile = state.get("profile") or {}
    name = str(profile.get("caller_name") or "").strip() or "माँ"
    reason = state.get("close_reason") or "incomplete"
    emergency = reason == "emergency"
    emergency_note = " ".join(str(state.get("emergency_note") or "").split())
    rows = _rows(state)
    concerns = [(spec, entry) for spec, entry, status in rows if status == "concern"]
    clears = [spec for spec, _entry, status in rows if status == "clear"]

    sections: dict[str, list[str]] = {heading: [] for heading in _HEADINGS}
    child_bits = [
        f"{child.get('id')} {child.get('sex') or 'unknown'} age {child.get('age_years')}"
        for child in (profile.get("children") or [])
    ]
    chief = concerns[0][0].need_en if concerns else "Screening call with no positive gate."
    sections[_HEADINGS[0]].extend([
        f"Caller: {name}.",
        f"Marital status: {profile.get('marital_status') or 'not reported'}.",
        f"Ever pregnant: {profile.get('ever_pregnant')}.",
        "Children: " + (", ".join(child_bits) if child_bits else "none."),
        f"Close reason: {reason}.",
        f"Chief complaint: {chief}",
    ])
    if emergency:
        suffix = f" Mother's words: {emergency_note}." if emergency_note else ""
        sections[_HEADINGS[1]].append(f"{_EMERGENCY_BULLET}{suffix}")
    for spec, entry, status in rows:
        if status == "concern":
            note = _human_notes(entry)
            gate = spec.nrroid_gate or spec.id
            words = f" Mother's words: {note}." if note else ""
            sections[_HEADINGS[1]].append(f"{gate} {spec.id}: {spec.need_en}.{words}")
            detail = f"Concern. {spec.need_en}." + (f" {note}" if note else "")
            sections[_section_for(spec)].append(detail)
        elif status in ("clear", "unknown"):
            sections[_section_for(spec)].append(f"{status.title()}. {spec.need_en}.")
    if not concerns and not emergency:
        sections[_HEADINGS[1]].append("No positive gate alerts.")
    negatives = [f"Clear: {spec.need_en}." for spec in clears] or [
        "No confirmed negatives were recorded."
    ]
    if reason != "complete":
        negatives.append("Call ended before every applicable question was asked.")
    sections[_HEADINGS[-1]].extend(negatives)

    doctor_parts = []
    for heading in _HEADINGS:
        body = sections[heading] or ["None reported in this call."]
        bullets = [
            _scrub_diagnostic_terms(line if line.startswith("-") else f"- {line}")
            for line in body
        ]
        doctor_parts.append(heading + "\n" + "\n".join(bullets))
    flags = [_flag_for(spec, entry) for spec, entry in concerns]
    if emergency:
        flags.append(f"{_EMERGENCY_FLAG}. {emergency_note}" if emergency_note else _EMERGENCY_FLAG)
    questions = [_concern_question(spec, entry) for spec, entry in concerns]
    has_children = bool(profile.get("children"))
    fitted = _fit_questions(
        questions,
        emergency=emergency,
        has_children=has_children,
        concern_questions=questions,
    )
    yes = _scrub_unique(_concern_sentence(spec, entry) for spec, entry in concerns)
    return {
        "mother_name_hi": _scrub_diagnostic_terms(name) or "माँ",
        "children_label_hi": _scrub_diagnostic_terms(_children_label(profile)),
        "mother_questions_hi": fitted,
        "yes_summaries_hi": yes,
        "doctor_markdown": "\n\n".join(doctor_parts),
        "red_flags": _scrub_unique(flags),
    }


def mother_summary_text(synthesis: dict) -> str:
    lines = [
        f"{index}. {question}"
        for index, question in enumerate(synthesis.get("mother_questions_hi") or [], 1)
    ]
    yes = [line for line in (synthesis.get("yes_summaries_hi") or []) if str(line).strip()]
    if yes:
        lines.extend(["", "इस कॉल में जो हाँ था", *(f"- {line}" for line in yes)])
    return "\n".join(lines).strip()
