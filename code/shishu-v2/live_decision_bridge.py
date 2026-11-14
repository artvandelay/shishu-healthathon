"""Decision API bridge for live calls: turn capture and single-engine turn execution."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import db
import decisions
import profile_extractor
from audio_recorder import pcm16_to_wav_bytes
from call_cursor import apply_interpreted_turn, apply_profile, close_technical_failure, step_id

logger = logging.getLogger(__name__)
USER_TURN_PCM_MAX_BYTES = 960_000
USER_TURN_PCM_TAIL_CHUNKS = 6
_PROFILE_REASK_QIDS = frozenset({"intro_name", "intro_married", "intro_children", "ever_pregnant"})


@dataclass
class TurnCapture:
    user_turn_buffer: str = ""
    user_turn_pcm: bytearray = field(default_factory=bytearray)
    user_turn_pcm_tail: int = 0
    user_turn_cursor_snapshot: dict | None = None
    user_turn_finalized: bool = False
    user_turn_is_quick_reply_stale: bool = False
    speech_allowed: bool = True
    handled_turn: bool = False
    last_key: tuple[str, str, int] | None = None
    thresholds: decisions.DecisionThresholds | None = None
    extractor_failures: int = 0
    extractor_failure_step: str = ""


def load_turn_thresholds(
    settings: dict | None = None, *, capture: TurnCapture | None = None,
    thresholds: decisions.DecisionThresholds | None = None,
) -> decisions.DecisionThresholds:
    """Resolve, load, and validate Decision API thresholds for a live call."""
    if (chosen := thresholds or (capture.thresholds if capture else None)) is not None:
        return decisions.validate_thresholds(chosen)
    raw = str((settings or {}).get("decision_thresholds_path")
              or os.environ.get("DECISION_THRESHOLDS_PATH") or "").strip()
    path = Path(raw) if raw else Path(__file__).resolve().with_name("decision_thresholds.json")
    return decisions.load_thresholds(path)


def merge_transcript_chunk(prev: str, chunk: str) -> str:
    prev_s, chunk_s = str(prev or "").strip(), str(chunk or "").strip()
    if not prev_s or chunk_s.startswith(prev_s):
        return chunk_s
    if not chunk_s or prev_s.endswith(chunk_s):
        return prev_s
    sep = "" if (prev_s[-1].isspace() or chunk_s[0] in ",.;:!?।") else " "
    return prev_s + sep + chunk_s


def reset_turn_capture(capture: TurnCapture, *, clear_pcm: bool = True) -> None:
    capture.user_turn_buffer = ""
    capture.user_turn_cursor_snapshot = None
    capture.user_turn_finalized = False
    capture.user_turn_is_quick_reply_stale = False
    if clear_pcm:
        capture.user_turn_pcm.clear()
        capture.user_turn_pcm_tail = 0


def append_turn_pcm(capture: TurnCapture, pcm_16k: bytes) -> None:
    avail = USER_TURN_PCM_MAX_BYTES - len(capture.user_turn_pcm)
    piece = pcm_16k[:avail] if (avail > 0 and pcm_16k) else b""
    if len(piece) % 2 == 1:
        piece = piece[:-1]
    if piece:
        capture.user_turn_pcm.extend(piece)


def consume_task_error(task: asyncio.Task) -> None:
    if not task.cancelled():
        try:
            task.exception()
        except Exception:
            return


async def extract_profile_updates(
    state: dict, utterance_text: str, audio_wav_bytes: bytes | None
) -> profile_extractor.ProfileExtractionResult:
    return await profile_extractor.extract_profile_entities(
        state, utterance_text=utterance_text, audio_wav_bytes=audio_wav_bytes
    )


def _decision_meta(
    res: decisions.DecisionCallResult | None = None, *, error_reason: str = ""
) -> dict[str, Any]:
    if res is None:
        return {"ok": False, "error_reason": error_reason, "decision_ms": 0, "attempts": 0,
                "request_bytes": 0, "question_count": 0, "catalog_option_count": 0,
                "usage_present": False, "total_input_tokens": None}
    return {"ok": bool(res.ok), "error_reason": str(res.error_reason or ""),
            "decision_ms": int(res.decision_ms), "attempts": int(res.attempts),
            "request_bytes": int(res.request_bytes), "question_count": int(res.question_count),
            "catalog_option_count": int(res.catalog_option_count),
            "usage_present": bool(res.usage_present), "total_input_tokens": res.total_input_tokens}


async def _log_event(call_id: int, event_type: str, detail: str, meta: dict[str, Any]) -> None:
    try:
        await asyncio.to_thread(db.append_call_event, call_id, event_type, detail, meta)
    except Exception:
        pass


async def _save_state_safe(call_id: int, state: dict) -> None:
    try:
        await asyncio.to_thread(db.save_cursor_state, call_id, json.loads(json.dumps(state)))
    except Exception:
        pass


def _sync_ever_pregnant(payload: dict[str, Any], updates: dict[str, Any], qid: str) -> None:
    if "ever_pregnant" not in updates:
        return
    ep = updates["ever_pregnant"]
    if payload.get("side_qid") == "ever_pregnant":
        if payload.get("side_action") != "add_or_correct":
            return
        so = str(payload.get("side_outcome") or "concern")
        if (so == "concern" and ep is True) or (so in ("clear", "retract") and ep is False):
            return
        updates.pop("ever_pregnant", None)
        for k in ("side_qid", "side_action", "side_outcome", "side_note"):
            payload.pop(k, None)
        if payload.get("current_outcome") not in ("", "not_about_active"):
            payload["current_outcome"] = "not_about_active"
    elif qid == "ever_pregnant":
        cur = str(payload.get("current_outcome") or "")
        bad = (cur == "concern" and ep is False) or (cur in ("clear", "retract") and ep is True)
        if bad or (cur == "unknown" and ep is not None):
            updates.pop("ever_pregnant", None)
            payload["current_outcome"] = "not_about_active"
        elif cur in ("concern", "clear", "retract"):
            updates.pop("ever_pregnant", None)


def _is_superseded(cursor: Any, telemetry: dict, step: str) -> bool:
    return bool(telemetry.get("end_requested") or cursor.state.get("phase") != "asking"
                or step_id(cursor.state) != step)


def _held_body(
    tool_body_fn: Callable[..., dict[str, Any]], cursor: Any, capture: TurnCapture, reason: str
) -> tuple[dict[str, Any], bool, str]:
    body = tool_body_fn(cursor.state)
    body["result"], body["reason"] = "held", reason
    capture.speech_allowed = True
    return body, False, "held"


async def _fail_decision_turn(
    cursor: Any, capture: TurnCapture, call_id: int,
    tool_body_fn: Callable[..., dict[str, Any]], mark_closing_fn: Callable[[str], Awaitable[None]],
) -> tuple[dict[str, Any], bool, str]:
    cursor.state = close_technical_failure(cursor.state)
    await _save_state_safe(call_id, cursor.state)
    await mark_closing_fn("technical_failure")
    capture.speech_allowed = True
    return tool_body_fn(cursor.state), False, "technical_failure"


async def run_decision_turn(
    cursor: Any, capture: TurnCapture, call_id: int, telemetry: dict, *,
    tool_body_fn: Callable[..., dict[str, Any]], mark_closing_fn: Callable[[str], Awaitable[None]],
    settings: dict | None = None, thresholds: decisions.DecisionThresholds | None = None,
    input_rate: int = 16000,
) -> tuple[dict[str, Any], bool, str]:
    if cursor.state.get("phase") != "asking":
        capture.speech_allowed = True
        return tool_body_fn(cursor.state), False, "closing"
    utt, pcm = capture.user_turn_buffer.strip(), bytes(capture.user_turn_pcm)
    if not utt and not pcm:
        return _held_body(tool_body_fn, cursor, capture, "empty_turn")
    step = step_id(cursor.state)
    if capture.last_key == (dedup_key := (step, " ".join(utt.split()), len(pcm))):
        return _held_body(tool_body_fn, cursor, capture, "duplicate_turn")
    try:
        resolved = load_turn_thresholds(settings, capture=capture, thresholds=thresholds)
    except Exception as exc:
        logger.exception("decision thresholds invalid on call %s", call_id)
        em = _decision_meta(error_reason=type(exc).__name__)
        await _log_event(call_id, "decision_turn", "error:invalid_thresholds", em)
        reset_turn_capture(capture)
        return await _fail_decision_turn(cursor, capture, call_id, tool_body_fn, mark_closing_fn)

    capture.last_key, capture.handled_turn = dedup_key, True
    snap = capture.user_turn_cursor_snapshot
    state_snap = copy.deepcopy(snap if snap and step_id(snap) == step else cursor.state)
    audio_wav = pcm16_to_wav_bytes(pcm, input_rate) if (not utt and pcm) else None
    reset_turn_capture(capture)
    res = await decisions.evaluate_turn(
        decisions.build_turn_request(state_snap, utterance_text=utt, audio_wav_bytes=audio_wav)
    )
    meta = _decision_meta(res)
    if _is_superseded(cursor, telemetry, step):
        await _log_event(call_id, "decision_turn", "superseded", meta)
        return _held_body(tool_body_fn, cursor, capture, "stale_turn")
    if not res.ok:
        err = f"error:{res.error_reason}" if res.error_reason else "error"
        await _log_event(call_id, "decision_turn", err, meta)
        return await _fail_decision_turn(cursor, capture, call_id, tool_body_fn, mark_closing_fn)

    interp = decisions.interpret_turn_decision(
        cursor.state, res.raw_answers, utterance_text=utt, thresholds=resolved
    )
    if interp.future_qid_match is not None:
        meta["future_qid_match"] = interp.future_qid_match
    meta["diagnostics"], payload = interp.diagnostics, dict(interp.cursor_payload)
    active_qid = str(cursor.state.get("active_qid") or "")
    outcome = ("emergency" if payload.get("emergency") else
               "caller_stop" if payload.get("caller_stop") else
               str(payload.get("current_outcome") or ""))
    ext: profile_extractor.ProfileExtractionResult | None = None
    if interp.needs_profile_extraction and outcome not in ("emergency", "caller_stop"):
        ext = await extract_profile_updates(cursor.state, utt, audio_wav)
        meta.update({"extractor_ok": bool(ext.ok), "extractor_ms": int(ext.extractor_ms),
                     "extractor_error": str(ext.error_reason or "")})
    if _is_superseded(cursor, telemetry, step):
        await _log_event(call_id, "decision_turn", "superseded", meta)
        return _held_body(tool_body_fn, cursor, capture, "stale_turn")
    dt = f"{active_qid}:{outcome}" if active_qid and outcome else (active_qid or outcome or "ok")
    await _log_event(call_id, "decision_turn", dt, meta)

    if ext is not None and ext.ok:
        capture.extractor_failures, capture.extractor_failure_step = 0, ""
        updates = dict(ext.updates)
        _sync_ever_pregnant(payload, updates, active_qid)
        if (active_qid in ("intro_name", "intro_married", "intro_children")
                and not updates and payload.get("current_outcome") != "clarify"):
            cursor.state, _ = apply_profile(cursor.state)
        if updates:
            payload["profile_updates"] = updates
    elif ext is not None:
        fails = (capture.extractor_failures if capture.extractor_failure_step == step else 0) + 1
        capture.extractor_failures, capture.extractor_failure_step = fails, step
        if fails >= 2:
            capture.extractor_failures, capture.extractor_failure_step = 0, ""
            return await _fail_decision_turn(
                cursor, capture, call_id, tool_body_fn, mark_closing_fn
            )
        payload.pop("profile_updates", None)
        if active_qid in _PROFILE_REASK_QIDS:
            capture.speech_allowed = True
            return tool_body_fn(cursor.state), False, (outcome or "reask")

    cursor.state = new_state = apply_interpreted_turn(cursor.state, payload)
    if step_id(new_state) != step:
        capture.extractor_failures, capture.extractor_failure_step = 0, ""
    await _save_state_safe(call_id, new_state)
    if new_state.get("phase") == "closing":
        await mark_closing_fn(str(new_state.get("close_reason") or "complete"))

    clarify = outcome not in ("emergency", "caller_stop") and (
        payload.get("current_outcome") == "clarify"
        or (payload.get("side_action") == "clarify_meaning"
            and cursor.state.get("active_qid") == payload.get("side_qid"))
    )
    body = tool_body_fn(cursor.state, outcome="clarify" if clarify else "")
    if clarify:
        body["clarify"] = True
    capture.speech_allowed = True
    return body, clarify, (outcome or "ok")
