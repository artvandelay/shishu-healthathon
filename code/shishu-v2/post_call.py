"""One post-call pass for the mother PDF and the clinician note.

The live voice loop never calls this. When Gemini is unreachable, the
deterministic fallback still fills both reports from the saved cursor.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from triage_catalog import active_specs

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

MODEL = "gemini-3.8-flash"

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

_SECTION = {
    "birth": "### Birth & Neonatal History",
    "body": "### Development, Growth & Physical Features",
    "illness": "### Systemic, Infections & Hematology",
    "pregnancy": "### Antenatal (Pregnancy) History",
    "family": "### Family & Pedigree History",
    "close": "### Patient Profile & Chief Complaint",
}

_HEMATOLOGY = frozenset({
    "gate_bleeding_joints",
    "gate_transfusion_anemia",
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

_DISEASE = re.compile(
    r"(?:"
    r"\b(?:Duchenne|Pompe|Wiskott[\s-]*Aldrich|Hemophilia|Haemophilia|Gaucher|SCID|SMA|MPS)\b"
    r"|ड्यूशेन|हीमोफीलिया|पॉम्पे|गॉशर|इम्यूनोडेफिशिएंसी|म्यूकोपॉलीसेकेराइडोसिस"
    r"|थैलेसीमिया मेजर"
    r")",
    re.IGNORECASE,
)
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_SEX_HI = {"male": "बेटा", "female": "बेटी"}


def cursor_of(call: dict | None) -> dict:
    if not isinstance(call, dict):
        return {}
    for key in ("cursor_state", "cursor_json"):
        raw = call.get(key)
        if isinstance(raw, str):
            if not raw.strip():
                continue
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict) and raw.get("active_qid"):
            return raw
    return {}


def has_cursor(call: dict | None) -> bool:
    return bool(cursor_of(call))


def is_synthesis(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("mother_questions_hi"), list)
        and isinstance(payload.get("doctor_markdown"), str)
    )


def _scrub_diagnostic_terms(text: str) -> str:
    cleaned = _DISEASE.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([।?])", r"\1", cleaned)
    return cleaned.strip()


def _section_for(spec) -> str:
    if spec.id in _HEMATOLOGY:
        return "### Systemic, Infections & Hematology"
    return _SECTION.get(spec.group, "### Pertinent Negatives")


def _children_label(profile: dict) -> str:
    bits = []
    for child in profile.get("children") or []:
        sex = _SEX_HI.get(child.get("sex") or "", "")
        age = str(child.get("age_label_hi") or "").strip()
        name = str(child.get("name") or "").strip()
        bits.append(" ".join(part for part in (name, age, sex) if part))
    return " · ".join(bit for bit in bits if bit)


def _followup_bits(entry: dict) -> str:
    followups = entry.get("followups") or {}
    if not isinstance(followups, dict):
        return ""
    bits = []
    for key, value in followups.items():
        if isinstance(value, dict):
            note = str(value.get("note") or "").strip()
            state = str(value.get("state") or "").strip()
            piece = " ".join(part for part in (state, note) if part)
            bits.append(f"{key}: {piece}" if piece else str(key))
        elif value:
            bits.append(f"{key}: {value}")
    return "; ".join(bits)


def _detail_text(entry: dict) -> str:
    note = str(entry.get("notes") or "").strip()
    extra = _followup_bits(entry)
    return " — ".join(part for part in (note, extra) if part)


def _concern_sentence(spec, entry: dict) -> str:
    note = _detail_text(entry)
    body = spec.question_hi.strip().rstrip("?")
    if body.startswith("क्या "):
        body = body[4:]
    if note:
        return f"{body} — {note}।"
    return f"{body}।"


def _concern_question(spec, entry: dict) -> str:
    note = str(entry.get("notes") or "").strip()
    if note:
        return f"मैंने बताया कि {note}। डॉक्टर, इसकी जाँच क्या होनी चाहिए?"
    body = spec.question_hi.strip().rstrip("?")
    if body.startswith("क्या "):
        body = body[4:]
    return f"मैंने हाँ कहा कि {body}। डॉक्टर, अब क्या जाँच करनी चाहिए?"


def _fit_questions(questions: list[str], *, concern_count: int = 0) -> list[str]:
    cap = min(10, max(6, concern_count))
    kept: list[str] = []
    for question in questions:
        text = _scrub_diagnostic_terms(question)
        if text and text not in kept:
            kept.append(text)
        if len(kept) == cap:
            break
    for routine in _ROUTINE_HI:
        if len(kept) >= 3:
            break
        if routine not in kept:
            kept.append(routine)
    return kept[:cap]


def _rows(state: dict) -> list[tuple]:
    answers = state.get("answers") or {}
    rows = []
    for spec in active_specs():
        if spec.kind in ("profile", "bridge"):
            continue
        entry = answers.get(spec.id)
        if not isinstance(entry, dict):
            continue
        status = entry.get("state") or ""
        if status in ("", "not_applicable"):
            continue
        rows.append((spec, entry, status))
    return rows


def deterministic_fallback(call: dict) -> dict:
    """Both reports from the cursor alone. No disease names, no transliteration."""
    state = cursor_of(call)
    profile = state.get("profile") or {}
    name = str(profile.get("caller_name") or "").strip() or "माँ"
    reason = state.get("close_reason") or "incomplete"
    rows = _rows(state)
    concerns = [(spec, entry) for spec, entry, status in rows if status == "concern"]
    clears = [spec for spec, _entry, status in rows if status == "clear"]
    unknowns = [spec for spec, _entry, status in rows if status == "unknown"]

    sections = {heading: [] for heading in _HEADINGS}
    child_bits = []
    for child in profile.get("children") or []:
        child_bits.append(
            f"{child.get('id')} {child.get('sex') or 'unknown'} age {child.get('age_years')}"
        )
    sections[_HEADINGS[0]].extend([
        f"Caller: {name}.",
        f"Marital status: {profile.get('marital_status') or 'not reported'}.",
        f"Ever pregnant: {profile.get('ever_pregnant')}.",
        "Children: " + (", ".join(child_bits) if child_bits else "none."),
        f"Close reason: {reason}.",
        "Chief complaint: "
        + (
            concerns[0][0].need_en
            if concerns
            else "Screening call with no positive gate."
        ),
    ])
    if reason == "emergency":
        sections[_HEADINGS[1]].append("Active emergency. Caller was told to go to the nearest hospital or call 108.")
    if concerns:
        for spec, entry in concerns:
            note = _detail_text(entry)
            gate = spec.nrroid_gate or spec.id
            line = f"{gate} {spec.id}: {spec.need_en}."
            if note:
                line += f" Mother's words: {note}."
            sections[_HEADINGS[1]].append(line)
            sections[_section_for(spec)].append(f"Concern. {spec.need_en}." + (f" {note}" if note else ""))
    elif reason != "emergency":
        sections[_HEADINGS[1]].append("No positive gate alerts.")
    for spec in clears:
        sections[_section_for(spec)].append(f"Clear. {spec.need_en}.")
    for spec in unknowns:
        sections[_section_for(spec)].append(f"Unknown. {spec.need_en}.")
    negatives = [f"Clear: {spec.need_en}." for spec in clears]
    if not negatives:
        negatives.append("No confirmed negatives were recorded.")
    if reason != "complete":
        negatives.append("Call ended before every applicable question was asked.")
    sections[_HEADINGS[-1]].extend(negatives)

    doctor_parts = []
    for heading in _HEADINGS:
        body = sections[heading] or ["None reported in this call."]
        doctor_parts.append(heading + "\n" + "\n".join(f"- {line}" if not line.startswith("-") else line for line in body))
    flags = [
        f"{spec.nrroid_gate or spec.id} {spec.id}: {spec.need_en}"
        for spec, _entry in concerns
    ]
    if reason == "emergency":
        flags.append("emergency: active danger sign now; hospital or 108")
    return _finish({
        "mother_name_hi": name,
        "children_label_hi": _children_label(profile),
        "mother_questions_hi": [_concern_question(spec, entry) for spec, entry in concerns],
        "yes_summaries_hi": [_concern_sentence(spec, entry) for spec, entry in concerns],
        "doctor_markdown": "\n\n".join(doctor_parts),
        "red_flags": flags,
        "_concern_count": len(concerns),
    })


def mother_summary_text(synthesis: dict) -> str:
    lines = [f"{index}. {question}" for index, question in enumerate(synthesis.get("mother_questions_hi") or [], 1)]
    yes = [line for line in (synthesis.get("yes_summaries_hi") or []) if str(line).strip()]
    if yes:
        lines.append("")
        lines.append("इस कॉल में जो हाँ था")
        lines.extend(f"- {line}" for line in yes)
    return "\n".join(lines).strip()


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _usable_hindi(items: list[str]) -> bool:
    return bool(items) and all(_DEVANAGARI.search(item) for item in items)


def _concern_rows(call: dict) -> list[tuple]:
    return [
        (spec, entry)
        for spec, entry, status in _rows(cursor_of(call))
        if status == "concern"
    ]


def _summary_text(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("summary_hi") or "").strip()
    return str(value or "").strip()


def _yes_by_qid(raw: dict) -> dict[str, str]:
    items = raw.get("yes_items_hi")
    pairs: list[tuple] = []
    if isinstance(items, dict):
        pairs = list(items.items())
    elif isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                pairs.append((item.get("qid"), item))
    found: dict[str, str] = {}
    for qid, value in pairs:
        key = str(qid or "").strip()
        text = _summary_text(value)
        if key and text:
            found[key] = text
    return found


def _summaries_for(concerns: list[tuple], raw: dict) -> list[str]:
    by_qid = _yes_by_qid(raw)
    lines = []
    for spec, entry in concerns:
        summary = by_qid.get(spec.id, "")
        if summary and _usable_hindi([summary]):
            lines.append(summary)
        else:
            lines.append(_concern_sentence(spec, entry))
    return lines


def _close_uncovered(questions: list[str], entry: dict) -> bool:
    note = str(entry.get("notes") or "").strip()
    blob = "\n".join(questions)
    if note:
        return note not in blob
    return "इसके अलावा" not in blob


def _insert_line(doctor: str, heading: str, line: str) -> str:
    if line in doctor:
        return doctor
    if heading not in doctor:
        return doctor.rstrip() + f"\n\n{heading}\n{line}\n"
    head, rest = doctor.split(heading, 1)
    nxt = rest.find("\n### ")
    if nxt < 0:
        return head + heading + rest.rstrip() + "\n" + line + "\n"
    return head + heading + rest[:nxt].rstrip() + "\n" + line + "\n" + rest[nxt:]


def _concern_visible(doctor: str, spec, entry: dict) -> bool:
    if spec.id in (doctor or ""):
        return True
    note = str(entry.get("notes") or "").strip()
    return bool(note) and note in (doctor or "")


def _flag_for(spec, entry: dict) -> str:
    note = str(entry.get("notes") or "").strip()
    flag = f"{spec.nrroid_gate or spec.id} {spec.id}: {spec.need_en}"
    if note:
        flag = f"{flag}. {note}"
    return flag


def _with_every_concern(doctor: str, flags: list[str], concerns: list[tuple]) -> tuple[str, list[str]]:
    for spec, entry in concerns:
        note = str(entry.get("notes") or "").strip()
        if spec.id == "close_anything_else":
            if spec.id not in doctor or (note and note not in doctor):
                line = f"- Additional mother disclosure (close_anything_else): {note or spec.need_en}."
                doctor = _insert_line(doctor, "### Patient Profile & Chief Complaint", line)
        elif not _concern_visible(doctor, spec, entry):
            line = f"- {spec.nrroid_gate or spec.id} {spec.id}: {spec.need_en}."
            if note:
                line += f" Mother's words: {note}."
            doctor = _insert_line(doctor, _section_for(spec), line)
        blob = "\n".join(flags)
        if spec.id in blob and (spec.id != "close_anything_else" or not note or note in blob):
            continue
        flags.append(_flag_for(spec, entry))
    return doctor, flags


def _headings_ok(doctor: str) -> bool:
    return all(heading in (doctor or "") for heading in _HEADINGS)


def _coerce(raw: dict, call: dict) -> dict:
    fallback = deterministic_fallback(call)
    concerns = _concern_rows(call)
    questions = _str_list(raw.get("mother_questions_hi"))
    doctor = raw.get("doctor_markdown") if isinstance(raw.get("doctor_markdown"), str) else ""
    if not _headings_ok(doctor):
        doctor = fallback["doctor_markdown"]
    if not _usable_hindi(questions):
        questions = list(fallback["mother_questions_hi"])
    close = next(((spec, entry) for spec, entry in concerns if spec.id == "close_anything_else"), None)
    if close and _close_uncovered(questions, close[1]):
        pinned = _concern_question(close[0], close[1])
        questions = [pinned] + [item for item in questions if item != pinned]
    # The cursor decides what was हाँ. A childless or all-clear call cannot grow a yes line.
    if concerns:
        yes = _summaries_for(concerns, raw)
        doctor, flags = _with_every_concern(
            doctor,
            _str_list(raw.get("red_flags")) or list(fallback["red_flags"]),
            concerns,
        )
    else:
        yes = []
        flags = _str_list(raw.get("red_flags")) or list(fallback["red_flags"])
    profile = (cursor_of(call).get("profile") or {})
    name = raw.get("mother_name_hi")
    profile_name = str(profile.get("caller_name") or "").strip()
    if profile_name and _DEVANAGARI.search(profile_name):
        name = profile_name
    elif not isinstance(name, str) or not name.strip() or not _DEVANAGARI.search(name):
        name = fallback["mother_name_hi"]
    if not profile.get("children"):
        children = ""
    else:
        children = raw.get("children_label_hi")
        if not isinstance(children, str) or not children.strip():
            children = fallback["children_label_hi"]
    return {
        "mother_name_hi": name.strip(),
        "children_label_hi": children.strip(),
        "mother_questions_hi": questions,
        "yes_summaries_hi": yes,
        "doctor_markdown": doctor.strip(),
        "red_flags": flags,
        "_concern_count": len(concerns),
    }


def _finish(result: dict) -> dict:
    questions = _fit_questions(
        _str_list(result.get("mother_questions_hi")),
        concern_count=int(result.get("_concern_count") or 0),
    )
    yes = []
    for line in _str_list(result.get("yes_summaries_hi")):
        cleaned = _scrub_diagnostic_terms(line)
        if cleaned and cleaned not in yes:
            yes.append(cleaned)
    finished = {
        "mother_name_hi": _scrub_diagnostic_terms(str(result.get("mother_name_hi") or "माँ")) or "माँ",
        "children_label_hi": _scrub_diagnostic_terms(str(result.get("children_label_hi") or "")),
        "mother_questions_hi": questions,
        "yes_summaries_hi": yes,
        "doctor_markdown": str(result.get("doctor_markdown") or "").strip(),
        "red_flags": _str_list(result.get("red_flags")),
    }
    if not _headings_ok(finished["doctor_markdown"]):
        finished["doctor_markdown"] = "\n\n".join(
            f"{heading}\n- None reported in this call." for heading in _HEADINGS
        )
    return finished


def _answered_brief(state: dict) -> str:
    lines = []
    for spec, entry, status in _rows(state):
        notes = str(entry.get("notes") or "").replace("\n", " ")
        followups = _followup_bits(entry).replace("\n", " ")
        children = ",".join(entry.get("child_ids") or [])
        lines.append(
            "\t".join((
                spec.id,
                status,
                spec.group,
                spec.nrroid_gate,
                spec.need_en,
                f"notes={notes}",
                f"followups={followups}",
                f"children={children}",
                spec.question_hi,
            ))
        )
    return "\n".join(lines)


def _transcript_block(call: dict) -> str:
    lines = []
    for turn in call.get("transcript") or []:
        if not isinstance(turn, dict):
            continue
        qid = str(turn.get("qid") or "")
        role = str(turn.get("role") or "speaker")
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        prefix = f"[{qid}] " if qid else ""
        lines.append(f"{prefix}{role}: {text}")
    return "\n".join(lines)


def _user_content(call: dict) -> str:
    state = cursor_of(call)
    return (
        "Write the post-call JSON from this cursor and transcript.\n"
        "Answer states are authoritative. concern means she said हाँ. "
        "clear means नहीं. unknown means पता नहीं. "
        "Do not mention items that are absent or not_applicable. "
        "Do not invent a child for a childless caller.\n\n"
        "Cursor answers (qid, state, group, gate, english need, notes, child ids, hindi question):\n"
        + _answered_brief(state)
        + "\n\nProfile:\n"
        + json.dumps(state.get("profile") or {}, ensure_ascii=False)
        + "\n\nTranscript:\n"
        + _transcript_block(call)
    )


_PROMPT = """You write two reports for one SHISHU screening call.

Return JSON with exactly these fields:
- mother_name_hi: her name in Devanagari. If the profile name is already Devanagari, copy it. Do not transliterate English letter by letter.
- children_label_hi: ages and बेटा/बेटी. Empty string when she has no children.
- mother_questions_hi: 3 to 10 first-person Hindi questions she can ask her doctor or ASHA/ANM. Include one question for every concern, including close_anything_else. If there are fewer than 3 concerns, add routine questions about growth, vaccines, or home warning signs. Never say nothing concerning was found. Never say she does not need a doctor.
- yes_items_hi: one object for every concern, in cursor order. Each object has qid (the cursor question id) and summary_hi (one Devanagari sentence using her notes). Include close_anything_else when its state is concern. No English words except BCG, NICU, ASHA, ANM.
- doctor_markdown: English Markdown for a clinician. Mention every concern question id, including close_anything_else, under the matching heading. Use these headings, each exactly once, in this order:
### Patient Profile & Chief Complaint
### Positive Red Flags & NRROID Gate Alerts
### Birth & Neonatal History
### Development, Growth & Physical Features
### Systemic, Infections & Hematology
### Antenatal (Pregnancy) History
### Family & Pedigree History
### Pertinent Negatives
- red_flags: one short English string for every concern, including close_anything_else. Use the question id and the plain finding. Do not name a diagnosis.

Never name a genetic or rare disease. Never advise starting, stopping, or changing a medicine.
The PDF already tells every mother to have thalassemia screening before the next pregnancy. Do not turn that universal line into her personal concern unless preg_thalassemia state is concern.
हाँ on preg_thalassemia means a trait, minor, or abnormal result was reported.
"""


def _schema():
    from google.genai import types

    text = types.Schema(type=types.Type.STRING)
    texts = types.Schema(type=types.Type.ARRAY, items=text)
    item = types.Schema(
        type=types.Type.OBJECT,
        properties={"qid": text, "summary_hi": text},
        required=["qid", "summary_hi"],
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "mother_name_hi": text,
            "children_label_hi": text,
            "mother_questions_hi": texts,
            "yes_items_hi": types.Schema(type=types.Type.ARRAY, items=item),
            "doctor_markdown": text,
            "red_flags": texts,
        },
        required=[
            "mother_name_hi",
            "children_label_hi",
            "mother_questions_hi",
            "yes_items_hi",
            "doctor_markdown",
            "red_flags",
        ],
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
        raise ValueError("synthesis response was not a JSON object")
    return data


async def _from_gemini(call: dict) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=_user_content(call),
            config=types.GenerateContentConfig(
                system_instruction=_PROMPT,
                response_mime_type="application/json",
                response_schema=_schema(),
                temperature=0.2,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        return _parse_json(response.text or "")
    finally:
        await client.aio.aclose()
        client.close()


async def synthesize_call(call: dict) -> dict:
    """Gemini 3.8 Flash once. Any API failure uses the cursor fallback."""
    try:
        raw = await _from_gemini(call)
        if not isinstance(raw, dict):
            raise ValueError("synthesis response was not a JSON object")
        return _finish(_coerce(raw, call))
    except Exception:
        return deterministic_fallback(call)


def _pdf_pages(data: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", data))
