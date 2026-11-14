"""Decision API bridge for live calls: turn capture, shadow evaluation, and authority execution."""

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

DECISION_MODES = frozenset({"off", "shadow", "authority"})
USER_TURN_PCM_MAX_BYTES = 960_000
USER_TURN_PCM_TAIL_CHUNKS = 6
_PROFILE_REASK_QIDS = frozenset({"intro_name", "intro_married", "intro_children", "ever_pregnant"})


@dataclass
class TurnCapture:
    user_turn_buffer: str = ""
    user_turn_pcm: bytearray = field(default_factory=bytearray)
    user_turn_pcm_tail: int = 0
    user_turn_cursor_snapshot: dict | None = None
    user_turn_shadow_sent: bool = False
    user_turn_finalized: bool = False
    user_turn_is_quick_reply_stale: bool = False
    shadow_tasks: set[asyncio.Task] = field(default_factory=set)
    decision_mode: str | None = None
    authority_speech_allowed: bool = True
    authority_handled_turn: bool = False
    authority_last_key: tuple[str, str, int] | None = None
    authority_thresholds: decisions.DecisionThresholds | None = None
    extractor_failures: int = 0
    extractor_failure_step: str = ""


def decision_mode(settings: dict | None = None, *, capture: TurnCapture | None = None) -> str:
    """Resolve decision mode ('off', 'shadow', or 'authority'). Defaults to 'off'."""
    candidates = (
        settings.get("decision_mode") if isinstance(settings, dict) else None,
        capture.decision_mode if capture is not None else None,
        os.environ.get("DECISION_MODE", "off"),
    )
    for val in candidates:
        text = str(val).strip().lower() if val is not None else ""
        if text:
            return text if text in DECISION_MODES else "off"
    return "off"


def load_authority_thresholds(
    settings: dict | None = None,
    *,
    capture: TurnCapture | None = None,
    authority_thresholds: decisions.DecisionThresholds | None = None,
) -> decisions.DecisionThresholds:
    """Resolve, load, and validate calibrated thresholds for Decision authority mode."""
    chosen = authority_thresholds or (capture.authority_thresholds if capture else None)
    if chosen is not None:
        return decisions.validate_thresholds(chosen, require_calibrated=True)
    raw_path = ""
    for candidate in (
        settings.get("decision_thresholds_path") if isinstance(settings, dict) else None,
        os.environ.get("DECISION_THRESHOLDS_PATH"),
    ):
        if candidate is not None and str(candidate).strip():
            raw_path = str(candidate).strip()
            break
    path = (
        Path(raw_path) if raw_path else Path(__file__).resolve().parent / "decision_thresholds.json"
    )
    return decisions.load_thresholds(path, require_calibrated=True)


def merge_transcript_chunk(prev: str, chunk: str) -> str:
    prev_s, chunk_s = str(prev or "").strip(), str(chunk or "").strip()
    if not prev_s:
        return chunk_s
    if not chunk_s or prev_s.endswith(chunk_s):
        return prev_s
    if chunk_s.startswith(prev_s):
        return chunk_s
    if prev_s[-1].isspace() or chunk_s[0] in ",.;:!?।":
        return prev_s + chunk_s
    return prev_s + " " + chunk_s


def reset_turn_capture(capture: TurnCapture, *, clear_pcm: bool = True) -> None:
    capture.user_turn_buffer = ""
    capture.user_turn_cursor_snapshot = None
    capture.user_turn_shadow_sent = False
    capture.user_turn_finalized = False
    capture.user_turn_is_quick_reply_stale = False
    if clear_pcm:
        capture.user_turn_pcm.clear()
        capture.user_turn_pcm_tail = 0


def append_turn_pcm(capture: TurnCapture, pcm_16k: bytes) -> None:
    avail = USER_TURN_PCM_MAX_BYTES - len(capture.user_turn_pcm)
    if avail <= 0 or not pcm_16k:
        return
    piece = pcm_16k[:avail]
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
    """Phase 5 generative profile entity extraction."""
    return await profile_extractor.extract_profile_entities(
        state, utterance_text=utterance_text, audio_wav_bytes=audio_wav_bytes
    )


def _decision_meta(
    res: decisions.DecisionCallResult | None = None, *, error_reason: str = ""
) -> dict[str, Any]:
    return {
        "ok": bool(res.ok) if res is not None else False,
        "error_reason": str(res.error_reason or "") if res is not None else error_reason,
        "decision_ms": int(res.decision_ms) if res is not None else 0,
        "attempts": int(res.attempts) if res is not None else 0,
        "request_bytes": int(res.request_bytes) if res is not None else 0,
        "question_count": int(res.question_count) if res is not None else 0,
        "catalog_option_count": int(res.catalog_option_count) if res is not None else 0,
        "usage_present": bool(res.usage_present) if res is not None else False,
        "total_input_tokens": res.total_input_tokens if res is not None else None,
    }


async def _log_event(call_id: int, event_type: str, detail: str, meta: dict[str, Any]) -> None:
    try:
        await asyncio.to_thread(db.append_call_event, call_id, event_type, detail, meta)
    except Exception:
        pass


async def _save_state_safe(call_id: int, state: dict) -> None:
    try:
        copied = json.loads(json.dumps(state, ensure_ascii=False))
        await asyncio.to_thread(db.save_cursor_state, call_id, copied)
    except Exception:
        pass


def _turn_detail(active_qid: str, outcome: str) -> str:
    if active_qid and outcome:
        return f"{active_qid}:{outcome}"
    return active_qid or outcome or "ok"


async def run_shadow_decision(call_id: int, snapshot_state: dict, utterance_text: str) -> None:
    utterance = " ".join(str(utterance_text or "").split())
    if not utterance:
        return
    try:
        request = decisions.build_turn_request(snapshot_state, utterance_text=utterance)
        result = await decisions.evaluate_turn(request)
        meta = _decision_meta(result)
        if result.ok:
            interpreted = decisions.interpret_turn_decision(
                snapshot_state,
                result.raw_answers,
                utterance_text=utterance,
                thresholds=decisions.UNCALIBRATED_THRESHOLDS,
                require_calibrated=False,
            )
            meta["diagnostics"] = interpreted.diagnostics
            outcome_diag = interpreted.diagnostics.get("current_outcome") or {}
            winner = outcome_diag.get("winner") or outcome_diag.get("top_option") or ""
            detail = _turn_detail(str(interpreted.diagnostics.get("active_qid") or ""), winner)
        else:
            detail = f"error:{result.error_reason}" if result.error_reason else "error"
        await _log_event(call_id, "decision_shadow", detail, meta)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("shadow decision failed for call %s: %s", call_id, exc)
        err_meta = _decision_meta(error_reason=type(exc).__name__)
        await _log_event(call_id, "decision_shadow", "error:exception", err_meta)


def schedule_shadow_decision(
    capture: TurnCapture,
    call_id: int | None,
    snapshot_state: dict,
    utterance_text: str,
    settings: dict | None = None,
) -> asyncio.Task | None:
    if decision_mode(settings, capture=capture) != "shadow" or not call_id or int(call_id) <= 0:
        return None
    clean = " ".join(str(utterance_text or "").split())
    if not clean:
        return None
    task = asyncio.create_task(
        run_shadow_decision(int(call_id), copy.deepcopy(snapshot_state), clean)
    )
    capture.shadow_tasks.add(task)

    def _cleanup(done_task: asyncio.Task) -> None:
        capture.shadow_tasks.discard(done_task)
        consume_task_error(done_task)

    task.add_done_callback(_cleanup)
    return task


async def drain_shadow_tasks(capture: TurnCapture | None, *, timeout_sec: float = 0.25) -> None:
    if capture is None or not capture.shadow_tasks:
        return
    pending = [task for task in list(capture.shadow_tasks) if not task.done()]
    if pending:
        if timeout_sec > 0:
            done, still_pending = await asyncio.wait(pending, timeout=timeout_sec)
            for task in done:
                consume_task_error(task)
        else:
            still_pending = pending
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.gather(*still_pending, return_exceptions=True)
    capture.shadow_tasks.clear()


def _sync_ever_pregnant(
    payload: dict[str, Any], profile_updates: dict[str, Any], *, active_qid: str = ""
) -> None:
    if "ever_pregnant" not in profile_updates:
        return
    ep_val = profile_updates["ever_pregnant"]
    if (
        payload.get("side_qid") == "ever_pregnant"
        and payload.get("side_action") == "add_or_correct"
    ):
        side_out = str(payload.get("side_outcome") or "concern")
        agrees = (side_out == "concern" and ep_val is True) or (
            side_out in ("clear", "retract") and ep_val is False
        )
        if not agrees:
            profile_updates.pop("ever_pregnant", None)
            for key in ("side_qid", "side_action", "side_outcome", "side_note"):
                payload.pop(key, None)
            if payload.get("current_outcome") not in ("", "not_about_active"):
                payload["current_outcome"] = "not_about_active"
        return

    if active_qid == "ever_pregnant":
        cur_out = str(payload.get("current_outcome") or "")
        disagrees = (
            (cur_out == "concern" and ep_val is False)
            or (cur_out in ("clear", "retract") and ep_val is True)
            or (cur_out == "unknown" and ep_val is not None)
        )
        if disagrees:
            profile_updates.pop("ever_pregnant", None)
            payload["current_outcome"] = "not_about_active"
        elif cur_out in ("concern", "clear", "retract"):
            profile_updates.pop("ever_pregnant", None)


def _is_superseded(cursor: Any, telemetry: dict, step_at_start: str) -> bool:
    return bool(
        telemetry.get("end_requested")
        or cursor.state.get("phase") != "asking"
        or step_id(cursor.state) != step_at_start
    )


def _held_body(
    tool_body_fn: Callable[..., dict[str, Any]], cursor: Any, capture: TurnCapture, reason: str
) -> tuple[dict[str, Any], bool, str]:
    body = tool_body_fn(cursor.state)
    body["result"] = "held"
    body["reason"] = reason
    capture.authority_speech_allowed = True
    return body, False, "held"


async def _fail_authority_turn(
    cursor: Any,
    capture: TurnCapture,
    call_id: int,
    tool_body_fn: Callable[..., dict[str, Any]],
    mark_closing_fn: Callable[[str], Awaitable[None]],
) -> tuple[dict[str, Any], bool, str]:
    cursor.state = close_technical_failure(cursor.state)
    await _save_state_safe(call_id, cursor.state)
    await mark_closing_fn("technical_failure")
    capture.authority_speech_allowed = True
    return tool_body_fn(cursor.state), False, "technical_failure"


async def run_authority_turn(
    cursor: Any,
    capture: TurnCapture,
    call_id: int,
    telemetry: dict,
    *,
    tool_body_fn: Callable[..., dict[str, Any]],
    mark_closing_fn: Callable[[str], Awaitable[None]],
    settings: dict | None = None,
    authority_thresholds: decisions.DecisionThresholds | None = None,
    input_rate: int = 16000,
) -> tuple[dict[str, Any], bool, str]:
    if cursor.state.get("phase") != "asking":
        capture.authority_speech_allowed = True
        return tool_body_fn(cursor.state), False, "closing"

    utterance_text = capture.user_turn_buffer.strip()
    pcm_bytes = bytes(capture.user_turn_pcm)
    if not utterance_text and not pcm_bytes:
        return _held_body(tool_body_fn, cursor, capture, "empty_turn")

    step_at_start = step_id(cursor.state)
    if capture.extractor_failure_step and capture.extractor_failure_step != step_at_start:
        capture.extractor_failures = 0
        capture.extractor_failure_step = ""
    dedup_key = (step_at_start, " ".join(utterance_text.split()), len(pcm_bytes))
    if capture.authority_last_key == dedup_key:
        return _held_body(tool_body_fn, cursor, capture, "duplicate_turn")

    try:
        thresholds = load_authority_thresholds(
            settings, capture=capture, authority_thresholds=authority_thresholds
        )
    except Exception as exc:
        logger.exception("authority thresholds invalid on call %s", call_id)
        err_meta = _decision_meta(error_reason=type(exc).__name__)
        await _log_event(call_id, "decision_authority", "error:uncalibrated_thresholds", err_meta)
        reset_turn_capture(capture)
        return await _fail_authority_turn(cursor, capture, call_id, tool_body_fn, mark_closing_fn)

    capture.authority_last_key = dedup_key
    capture.authority_handled_turn = True
    snap = capture.user_turn_cursor_snapshot
    state_snapshot = copy.deepcopy(
        snap if snap is not None and step_id(snap) == step_at_start else cursor.state
    )
    audio_wav = (
        pcm16_to_wav_bytes(pcm_bytes, input_rate) if (not utterance_text and pcm_bytes) else None
    )
    reset_turn_capture(capture)
    req = decisions.build_turn_request(
        state_snapshot, utterance_text=utterance_text, audio_wav_bytes=audio_wav
    )
    res = await decisions.evaluate_turn(req)
    meta = _decision_meta(res)

    if _is_superseded(cursor, telemetry, step_at_start):
        await _log_event(call_id, "decision_authority", "superseded", meta)
        return _held_body(tool_body_fn, cursor, capture, "stale_turn")

    if not res.ok:
        detail = f"error:{res.error_reason}" if res.error_reason else "error"
        await _log_event(call_id, "decision_authority", detail, meta)
        return await _fail_authority_turn(cursor, capture, call_id, tool_body_fn, mark_closing_fn)

    interp = decisions.interpret_for_authority(
        cursor.state, res.raw_answers, utterance_text=utterance_text, thresholds=thresholds
    )
    if interp.future_qid_match is not None:
        meta["future_qid_match"] = interp.future_qid_match
    meta["diagnostics"] = interp.diagnostics
    payload = dict(interp.cursor_payload)

    active_qid = str(cursor.state.get("active_qid") or "")
    if payload.get("emergency"):
        outcome_label = "emergency"
    elif payload.get("caller_stop"):
        outcome_label = "caller_stop"
    else:
        outcome_label = str(payload.get("current_outcome") or "")
    detail = _turn_detail(active_qid, outcome_label)

    ext: profile_extractor.ProfileExtractionResult | None = None
    if (
        interp.needs_profile_extraction
        and not payload.get("emergency")
        and not payload.get("caller_stop")
    ):
        ext = await extract_profile_updates(cursor.state, utterance_text, audio_wav)
        meta["extractor_ok"] = bool(ext.ok)
        meta["extractor_ms"] = int(ext.extractor_ms)
        meta["extractor_error"] = str(ext.error_reason or "")

    if _is_superseded(cursor, telemetry, step_at_start):
        await _log_event(call_id, "decision_authority", "superseded", meta)
        return _held_body(tool_body_fn, cursor, capture, "stale_turn")

    await _log_event(call_id, "decision_authority", detail, meta)

    if ext is not None and ext.ok:
        capture.extractor_failures = 0
        capture.extractor_failure_step = ""
        profile_updates = dict(ext.updates)
        _sync_ever_pregnant(payload, profile_updates, active_qid=active_qid)
        if (
            active_qid in ("intro_name", "intro_married", "intro_children")
            and not profile_updates
            and payload.get("current_outcome") != "clarify"
        ):
            cursor.state, _ = apply_profile(cursor.state)
        if profile_updates:
            payload["profile_updates"] = profile_updates
    elif ext is not None:
        if capture.extractor_failure_step != step_at_start:
            capture.extractor_failures = 0
            capture.extractor_failure_step = step_at_start
        capture.extractor_failures += 1
        if capture.extractor_failures >= 2:
            capture.extractor_failures = 0
            capture.extractor_failure_step = ""
            return await _fail_authority_turn(
                cursor, capture, call_id, tool_body_fn, mark_closing_fn
            )
        payload.pop("profile_updates", None)
        if active_qid in _PROFILE_REASK_QIDS:
            capture.authority_speech_allowed = True
            return tool_body_fn(cursor.state), False, (outcome_label or "reask")

    new_state = apply_interpreted_turn(cursor.state, payload)
    cursor.state = new_state
    if step_id(new_state) != step_at_start:
        capture.extractor_failures = 0
        capture.extractor_failure_step = ""
    await _save_state_safe(call_id, new_state)
    if new_state.get("phase") == "closing":
        await mark_closing_fn(str(new_state.get("close_reason") or "complete"))

    not_closing = not payload.get("emergency") and not payload.get("caller_stop")
    clarify_applied = not_closing and (
        payload.get("current_outcome") == "clarify"
        or (
            payload.get("side_action") == "clarify_meaning"
            and cursor.state.get("active_qid") == payload.get("side_qid")
        )
    )
    body = tool_body_fn(
        cursor.state,
        outcome="clarify" if clarify_applied else str(payload.get("current_outcome") or ""),
    )
    if clarify_applied:
        body["clarify"] = True
    capture.authority_speech_allowed = True
    return body, clarify_applied, (outcome_label or "ok")
