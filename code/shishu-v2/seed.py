"""Seed the two demo calls and render the three sample Mother PDFs."""

import array
import math
import os
import wave
from pathlib import Path

import db
import engine
import mother_report
import report_service
import summary
from models import (
    ChildInput,
    ChildSex,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_VOICE,
    MaritalStatus,
    Phase,
    ProfilePatch,
    SubStep,
    TurnCommand,
    TurnOutcome,
)
from triage_catalog import spec_or_none

_STARTED_AT = "2026-11-12 04:00:00"
_NO, _DONT_KNOW = "नहीं।", "पता नहीं।"
_CONCERN, _CLEAR = TurnOutcome.CONCERN, TurnOutcome.CLEAR
# name, children, seconds, scripted हाँ answers per qid, gates before she stops
_SCENARIOS: tuple[tuple, ...] = (
    (
        "SHISHU-sample-red-flags.pdf", "शिल्पा", ((7.0, ChildSex.MALE, "आरव"),), 420,
        {
            "birth_jaundice": ("हाँ, तीन दिन नीली लाइट में रखा था।",),
            "body_joint_swelling": ("हाँ, गिरने पर घुटने में सूजन 5 से 7 दिन रहती है।",),
            "preg_conditions": ("हाँ, हाई बीपी और थायरॉइड की दवा डिलीवरी के बाद भी चल रही है।",),
            "fam_infant_death": ("हाँ, 5 महीने का बेटा था।", "झटके आने से उसकी मौत हुई थी।"),
        },
        None,
    ),
    (
        "SHISHU-sample-no-flags.pdf", "गीता", ((6.0, ChildSex.FEMALE, "अनन्या"),), 360, {}, None,
    ),
    (
        "SHISHU-sample-incomplete.pdf", "सुनीता", ((4.0, ChildSex.FEMALE, None),), 95,
        {"birth_nicu": ("हाँ, दो दिन काँच के बक्से में रखा पड़ा था।", "दो दिन।")}, 3,
    ),
)


def _answer(state, queues: dict[str, list[str]]) -> tuple[str, TurnCommand]:
    """The Mother's next words, plus the command those words produce."""
    queue, qid = queues.get(state.cursor.qid) or [], state.cursor.qid
    if state.cursor.substep is SubStep.FOLLOWUP:
        words = queue.pop(0) if queue else _DONT_KNOW
        return words, TurnCommand(outcome=_CONCERN, utterance=words)
    if not queue:
        return _NO, TurnCommand(outcome=_CLEAR)
    words = queue.pop(0)
    follows = bool(spec_or_none(qid).followups)
    return words, TurnCommand(outcome=_CONCERN, utterance=words, followup_answered=follows)


def _walk(scenario: tuple) -> tuple:
    """Replay one scenario through the real engine and keep every spoken turn."""
    _sample, name, children, _seconds, scripted, stop_after = scenario
    state, turns = engine.new_state(), []
    kids = tuple(ChildInput(age_years=age, sex=sex, name=child) for age, sex, child in children)
    for patch, words in (
        (ProfilePatch(caller_name=name), f"मेरा नाम {name} है।"),
        (ProfilePatch(marital_status=MaritalStatus.MARRIED), "हाँ, शादी हो गई है।"),
        (ProfilePatch(children=kids), "एक बच्चा है।"),
    ):
        turns += [("assistant", engine.active_prompt_hi(state), state.cursor.qid)]
        turns += [("user", words, state.cursor.qid)]
        state = engine.apply_profile_patch(state, patch)

    queues = {qid: list(replies) for qid, replies in scripted.items()}
    answered = 0
    for _ in range(200):
        if state.phase is not Phase.ASKING:
            break
        if stop_after is not None and answered >= stop_after:
            state = engine.reduce_turn(state, TurnCommand(caller_stop=True))
            break
        qid, substep = state.cursor.qid, state.cursor.substep
        turns.append(("assistant", engine.active_prompt_hi(state), qid))
        words, command = _answer(state, queues)
        turns.append(("user", words, qid))
        state = engine.reduce_turn(state, command)
        spec = spec_or_none(qid)
        answered += int(substep is SubStep.MAIN and spec is not None and spec.is_routable)
    turns.append(("assistant", engine.say_hindi(state), ""))
    return state, turns


def _placeholder_wav(path: str, seconds: int = 3) -> int:
    """A stand-in stereo 16 kHz recording so the audio route has a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tone = [int(4000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000 * seconds)]
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(array.array("h", (s for s in tone for _ in (0, 1))).tobytes())
    return seconds


def _seed_call(scenario: tuple) -> int:
    state, turns = _walk(scenario)
    call_id = db.create_call(
        caller_name=scenario[1],
        user_role="user",
        model_used=DEFAULT_LIVE_MODEL,
        voice_used=DEFAULT_LIVE_VOICE,
    )
    for role, text, qid in turns:
        db.append_transcript(call_id, role, text, qid=qid)
    db.save_cursor_state(call_id, state)
    audio_file = f"call_{call_id}.wav"
    seconds = _placeholder_wav(os.path.join(db.DATA_DIR, "audio", audio_file))
    db.complete_call(call_id, seconds, audio_file)
    report_service.build_mother_pdf_sync(call_id, emit_events=False)
    return call_id


def sample_reports() -> dict[str, dict]:
    """The three documentation samples, projected from the same pure state walk."""
    return {
        scenario[0]: mother_report.build_report(
            {"id": index, "started_at": _STARTED_AT, "duration_sec": scenario[3]},
            summary.project_summary(_walk(scenario)[0]),
        )
        for index, scenario in enumerate(_SCENARIOS, start=1)
    }


def write_sample_pdfs(folder: str | os.PathLike[str]) -> list[Path]:
    """Render the three documentation samples from the single pure projection."""
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in sample_reports()]
    for path, report in zip(paths, sample_reports().values()):
        path.write_bytes(mother_report.render_pdf(report))
    return paths


def ensure_seeded() -> None:
    if not db.list_calls():
        for scenario in _SCENARIOS[:2]:
            _seed_call(scenario)
