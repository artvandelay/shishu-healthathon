"""Deterministic post-call builder for the mother PDF and clinician note."""

from __future__ import annotations

import call_cursor
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
    "birth": _HEADINGS[2],
    "body": _HEADINGS[3],
    "illness": _HEADINGS[4],
    "pregnancy": _HEADINGS[5],
    "family": _HEADINGS[6],
    "close": _HEADINGS[0],
}

_HEMATOLOGY = frozenset({
    "body_bleeding_bruise",
    "body_joint_swelling",
    "mod_tooth_bleed",
    "mod_heavy_periods",
    "body_transfusion",
    "body_blood_count",
    "body_limb_pain",
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
    "अस्पताल पहुँचकर डॉक्टर को कौन सी चेतावनी की बातों और पहले की बीमारी बतानी चाहिए?",
)

_EMERGENCY_FLAG = "emergency: active danger sign now; hospital or 108"
_EMERGENCY_LINE = "Active emergency. Caller was told to go to the nearest hospital or call 108."


def _unique(items) -> list[str]:
    out: list[str] = []
    for item in items:
        cleaned = " ".join(str(item or "").split())
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _section_for(spec) -> str:
    return _HEADINGS[4] if spec.id in _HEMATOLOGY else _SECTION.get(spec.group, _HEADINGS[-1])


def _child_scope_hi(profile: dict, entry: dict) -> str:
    children = profile.get("children") or []
    ids = entry.get("child_ids") or []
    if len(children) <= 1 or not ids:
        return ""
    return call_cursor.children_label_hi(profile, ids)


def _child_scope_en(profile: dict, entry: dict) -> str:
    children = profile.get("children") or []
    ids = entry.get("child_ids") or []
    if len(children) <= 1 or not ids:
        return ""
    return call_cursor.children_label_en(profile, ids)


def _concern_sentence(spec, entry: dict, profile: dict) -> str:
    note = call_cursor.mother_notes_text(entry)
    child_hi = _child_scope_hi(profile, entry)
    if spec.id == "close_anything_else":
        return f"अन्य बात — {note}।" if note else "डॉक्टर को बताने के लिए एक अन्य बात दर्ज की गई।"
    head = f"{spec.stem_hi} ({child_hi})" if child_hi else spec.stem_hi
    return f"{head} — {note}।" if note else f"{head}।"


def _concern_question(spec, entry: dict, profile: dict) -> str:
    note = call_cursor.mother_notes_text(entry)
    child_hi = _child_scope_hi(profile, entry)
    scope_prefix = f"{child_hi} के लिए: " if (child_hi and child_hi not in note) else ""
    if note:
        return f"{scope_prefix}मैंने बताया कि {note}। डॉक्टर, इसकी जाँच क्या होनी चाहिए?"
    if spec.id == "close_anything_else":
        return "मैंने कॉल के अंत में एक और बात दर्ज करवाई थी। डॉक्टर, उस पर क्या सलाह है?"
    head = f"{spec.stem_hi} ({child_hi})" if child_hi else spec.stem_hi
    return f"मैंने हाँ कहा कि {head}। डॉक्टर, अब क्या जाँच करनी चाहिए?"


def _fit_questions(
    questions: list[str],
    *,
    emergency: bool = False,
    has_children: bool = True,
) -> list[str]:
    if emergency and not questions:
        return list(_EMERGENCY_HI)
    routine = _ROUTINE_HI if has_children else _ROUTINE_CHILDLESS_HI
    kept = [
        text
        for text in _unique(questions)
        if not (emergency and (text in _ROUTINE_HI or text in _ROUTINE_CHILDLESS_HI))
    ][:6]
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


def _flag_for(spec, entry: dict, profile: dict) -> str:
    note = call_cursor.answer_notes_text(entry)
    child_en = _child_scope_en(profile, entry)
    head = f"{spec.nrroid_gate} {spec.id}" if spec.nrroid_gate else spec.id
    flag = f"{head}: {spec.need_en}"
    if child_en:
        flag = f"{flag} [{child_en}]"
    return f"{flag}. {note}" if note else flag


def build_reports(call: dict) -> dict:
    """Build clinician English note (verbatim) and mother Hindi report (non-diagnostic)."""
    state = call_cursor.cursor_from_call(call)
    profile = state.get("profile") if isinstance(state.get("profile"), dict) else {}
    name = " ".join(str(profile.get("caller_name") or "").split()) or "माँ"
    reason = state.get("close_reason") or "incomplete"
    emergency = reason == "emergency"
    emergency_note = " ".join(str(state.get("emergency_note") or "").split())
    rows = _rows(state)
    concerns = [(spec, entry) for spec, entry, status in rows if status == "concern"]
    clears = [spec for spec, _entry, status in rows if status == "clear"]

    sections: dict[str, list[str]] = {heading: [] for heading in _HEADINGS}
    children_en = call_cursor.children_label_en(profile)
    chief = concerns[0][0].need_en if concerns else "Screening call with no positive gate."
    sections[_HEADINGS[0]].extend([
        f"Caller: {name}.",
        f"Marital status: {profile.get('marital_status') or 'not reported'}.",
        f"Ever pregnant: {profile.get('ever_pregnant')}.",
        f"Children: {children_en if children_en else 'none.'}",
        f"Close reason: {reason}.",
        f"Chief complaint: {chief}",
    ])
    if emergency:
        suffix = f" Mother's words: {emergency_note}." if emergency_note else ""
        sections[_HEADINGS[1]].append(f"{_EMERGENCY_LINE}{suffix}")
    for spec, entry, status in rows:
        if status == "concern":
            note = call_cursor.answer_notes_text(entry)
            child_en = _child_scope_en(profile, entry)
            child_tag = f" [{child_en}]" if child_en else ""
            gate = spec.nrroid_gate or spec.id
            words = f" Mother's words: {note}." if note else ""
            sections[_HEADINGS[1]].append(f"{gate} {spec.id}: {spec.need_en}{child_tag}.{words}")
            detail = f"Concern. {spec.need_en}{child_tag}." + (f" {note}" if note else "")
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
        bullets = [f"- {line}" for line in body]
        doctor_parts.append(heading + "\n" + "\n".join(bullets))

    flags = [_flag_for(spec, entry, profile) for spec, entry in concerns]
    if emergency:
        flags.append(f"{_EMERGENCY_FLAG}. {emergency_note}" if emergency_note else _EMERGENCY_FLAG)

    questions = [_concern_question(spec, entry, profile) for spec, entry in concerns]
    fitted = _fit_questions(
        questions,
        emergency=emergency,
        has_children=bool(profile.get("children")),
    )
    yes = _unique(_concern_sentence(spec, entry, profile) for spec, entry in concerns)
    return {
        "mother_name_hi": name,
        "children_label_hi": call_cursor.children_label_hi(profile),
        "mother_questions_hi": fitted,
        "yes_summaries_hi": yes,
        "doctor_markdown": "\n\n".join(doctor_parts),
        "red_flags": _unique(flags),
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
