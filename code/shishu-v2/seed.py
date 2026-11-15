"""Seed two completed v4 clinical calls when the database has none."""

import array
import math
import os
import wave

import db
import mother_report
import post_call
import prompts
import report_service
from call_cursor import active_prompt_hi, apply_profile, apply_turn, new_cursor_state
from gemini_live import DEFAULT_LIVE_SETTINGS

_SHILPA_CONCERNS: dict[tuple[str, int | None], tuple[str, dict]] = {
    ("birth_jaundice", None): (
        "हाँ, तीन दिन नीली लाइट में रखा।",
        {"outcome": "concern", "note": "तीन दिन नीली लाइट में रखा", "followup_answered": True},
    ),
    ("body_joint_swelling", None): (
        "हाँ, गिरने पर घुटने में सूजन 5 से 7 दिन रहती है।",
        {"outcome": "concern", "note": "गिरने पर घुटने में सूजन 5 से 7 दिन रहती है"},
    ),
    ("preg_conditions", None): (
        "हाँ, हाई बीपी और थायरॉइड की दवाई डिलीवरी के बाद भी चल रही है।",
        {
            "outcome": "concern",
            "note": "हाई बीपी और थायरॉइड की दवाई डिलीवरी के बाद भी चल रही है",
            "followup_answered": True,
        },
    ),
    ("fam_infant_death", None): (
        "हाँ, 5 महीने का बेटा।",
        {"outcome": "concern", "note": "5 महीने का बेटा", "followup_answered": True},
    ),
    ("fam_infant_death", 1): (
        "झटके आने से।",
        {"outcome": "concern", "note": "झटके आने से"},
    ),
}

_GEETA_CLARIFY = {
    "preg_thalassemia": (
        "थैलेसीमिया क्या होता है?",
        "थैलेसीमिया माइनर या ट्रेट एक साधारण खून की जाँच है। इससे पता चलता है कि माता या पिता में यह गुण है या नहीं, और वे खुद बिल्कुल स्वस्थ रहते हैं। क्या आपकी या आपके पति की ऐसी जाँच में कभी कुछ बताया गया?",
    ),
}


def _write_placeholder_wav(path: str, seconds: int = 3) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rate = 16000
    frames = array.array("h")
    for i in range(rate * seconds):
        sample = int(4000 * math.sin(2 * math.pi * 440 * i / rate))
        frames.append(sample)
        frames.append(sample)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return seconds


def _walk_completed_call(
    caller_name: str,
    child: dict,
    child_reply_hi: str,
    concerns: dict[tuple[str, int | None], tuple[str, dict]] | None = None,
    clarifies: dict[str, tuple[str, str]] | None = None,
) -> tuple[dict, list[tuple[str, str, str]]]:
    state = new_cursor_state()
    turns: list[tuple[str, str, str]] = []

    turns.append(("assistant", active_prompt_hi(state), "intro_name"))
    turns.append(("user", f"मेरा नाम {caller_name} है।", "intro_name"))
    state, hold = apply_profile(state, caller_name=caller_name)
    if hold:
        raise RuntimeError(f"seed profile hold: {hold}")

    turns.append(("assistant", active_prompt_hi(state), "intro_married"))
    turns.append(("user", "हाँ, शादी हो गई है।", "intro_married"))
    state, hold = apply_profile(state, marital_status="married")
    if hold:
        raise RuntimeError(f"seed profile hold: {hold}")

    turns.append(("assistant", active_prompt_hi(state), "intro_children"))
    turns.append(("user", child_reply_hi, "intro_children"))
    state, hold = apply_profile(state, children=[child], ever_pregnant=True)
    if hold:
        raise RuntimeError(f"seed profile hold: {hold}")

    concern_map = concerns or {}
    clarify_map = dict(clarifies or {})
    for _ in range(120):
        if state["phase"] != "asking":
            break
        qid = state["active_qid"]
        pending = state.get("pending_followup_idx")
        turns.append(("assistant", active_prompt_hi(state), qid))
        if pending is None and qid in clarify_map:
            user_q, explain = clarify_map.pop(qid)
            turns.append(("user", user_q, qid))
            turns.append(("assistant", explain, qid))
        if (qid, pending) in concern_map:
            reply, kwargs = concern_map[(qid, pending)]
            turns.append(("user", reply, qid))
            state = apply_turn(state, **kwargs)
        elif qid == "intro_bridge":
            turns.append(("user", "हाँ, शुरू करें।", qid))
            state = apply_turn(state, "clear")
        else:
            turns.append(("user", "नहीं।", qid))
            state = apply_turn(state, "clear")
    if state["phase"] != "closing" or state.get("close_reason") != "complete":
        raise RuntimeError(f"seed cursor did not complete: {state['phase']} / {state.get('close_reason')}")
    turns.append(("assistant", prompts.closing_hi(state), ""))
    return state, turns


def _seed_call(
    caller_name: str,
    child: dict,
    child_reply_hi: str,
    concerns: dict[tuple[str, int | None], tuple[str, dict]] | None = None,
    clarifies: dict[str, tuple[str, str]] | None = None,
) -> int:
    state, turns = _walk_completed_call(caller_name, child, child_reply_hi, concerns, clarifies)
    call_id = db.create_call(
        caller_name=caller_name,
        user_role="user",
        model_used=DEFAULT_LIVE_SETTINGS["gemini_model"],
        voice_used=DEFAULT_LIVE_SETTINGS["gemini_voice"],
    )
    for role, text, qid in turns:
        db.append_transcript(call_id, role, text, qid=qid)
    db.save_cursor_state(call_id, state)

    synthesis = post_call.build_reports({"cursor_state": state})
    summary_doctor = synthesis.get("doctor_markdown") or ""
    summary_mother = post_call.mother_summary_text(synthesis)
    red_flags = list(synthesis.get("red_flags") or [])

    audio_file = f"call_{call_id}.wav"
    duration = _write_placeholder_wav(os.path.join(db.DATA_DIR, "audio", audio_file), 3)
    db.complete_call(
        call_id,
        duration,
        audio_file,
        summary_doctor,
        summary_mother,
        red_flags=red_flags,
    )
    call_dict = db.get_call(call_id) or {"id": call_id, "cursor_state": state, "duration_sec": duration}
    pdf = mother_report.render_pdf(mother_report.build_report(call_dict, synthesis))
    report_service.write_pdf_bytes(call_id, pdf)
    return call_id


def ensure_seeded() -> None:
    if db.list_calls():
        return
    _seed_call(
        "शिल्पा",
        {"age_years": 7, "sex": "male", "name": "आरव"},
        "एक 7 साल का बेटा आरव है।",
        concerns=_SHILPA_CONCERNS,
    )
    _seed_call(
        "गीता",
        {"age_years": 6, "sex": "female", "name": "अनन्या"},
        "एक 6 साल की बेटी अनन्या है।",
        clarifies=_GEETA_CLARIFY,
    )
