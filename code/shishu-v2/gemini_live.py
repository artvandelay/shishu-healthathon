"""Standalone Gemini Live pipeline for one SHISHU pre-consultation call (/ws/gemini)."""

import asyncio
import base64
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

import audio_recorder
import db
import decisions
import prompts

logger = logging.getLogger(__name__)
from call_cursor import (
    apply_turn,
    expect_yes_no,
    new_cursor_state,
    step_id,
)
from live_decision_bridge import (
    TurnCapture,
    consume_task_error,
    load_turn_thresholds,
    run_decision_turn,
)
from triage_catalog import get_spec
from prompts import (
    GEMINI_LIVE_SYSTEM_PROMPT,
    HINDI_MEDICAL_VOCAB,
)

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

DEFAULT_LIVE_SETTINGS: dict[str, str] = {
    "gemini_model": "gemini-3.1-flash-live-preview",
    "gemini_voice": "Kore",
    "vad_prefix_padding_ms": "300",
    "vad_silence_ms": "1000",
    "vad_start_sensitivity": "LOW",
    "vad_end_sensitivity": "LOW",
    "watchdog_timeout_sec": "8.0",
}

_LIVE_SETTING_ENV_KEYS: tuple[tuple[str, str], ...] = (
    ("GEMINI_MODEL", "gemini_model"),
    ("GEMINI_VOICE", "gemini_voice"),
    ("VAD_PREFIX_PADDING_MS", "vad_prefix_padding_ms"),
    ("VAD_SILENCE_MS", "vad_silence_ms"),
    ("VAD_START_SENSITIVITY", "vad_start_sensitivity"),
    ("VAD_END_SENSITIVITY", "vad_end_sensitivity"),
    ("WATCHDOG_TIMEOUT_SEC", "watchdog_timeout_sec"),
    ("DECISION_THRESHOLDS_PATH", "decision_thresholds_path"),
)

_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_WATCHDOG_INTERVAL_SEC = 0.5
_CHECK_IN_SEC = 15
_SILENCE_CLOSE_SEC = 30
_SOFT_LIMIT_SEC = 19 * 60
_HARD_LIMIT_SEC = 20 * 60
_CLOSING_AUDIO_GRACE_SEC = 0.8
_CLOSING_MAX_WAIT_SEC = 25
_TOOL_SPEAK_FALLBACK_SEC = 2.5

_NEXT_QUESTION_DECL = {
    "name": "next_question",
    "description": (
        "Call this on every caller utterance with no arguments to fetch the exact Hindi line to speak next. "
        "Never speak to the caller before calling next_question."
    ),
    "parameters": {"type": "OBJECT", "properties": {}},
}

_TELEMETRY_EVENTS = {
    "session_started",
    "user_transcript",
    "bot_first_audio",
    "turn_complete",
    "tool_call",
    "interrupted",
    "watchdog_nudge",
    "manual_nudge",
    "reconnecting",
    "reconnected",
    "silence_check_in",
    "auto_close",
    "error",
    "end_accepted",
}


class _TurnState:
    def __init__(self) -> None:
        self.awaiting_bot_reply = False
        self.last_user_turn_ts = 0.0
        self.bot_replied_since_user = False
        self.bot_speaking = False
        self.nudges_for_current_turn = 0
        self.quiet_from = 0.0
        self.checked_in = False
        self.call_started_ts = time.time()
        self.resume_handle = ""
        self.reconnect_requested = False
        self.capture = TurnCapture()
        self.time_cue_sent = False
        self.speech_end_waiting = False
        self.playback_ends_at = 0.0
        self.spoken_qid = ""
        self.speak_fallback: asyncio.Task | None = None
        self.ignore_bot_audio = False
        self.user_turn_step = ""
        self.pending_speak_step = ""
        self.pending_speak_kind = "speak"
        self.last_quick_reply_step = ""
        self.last_quick_reply_ts = 0.0
        self.speak_cue_step = ""
        self.go_away_pending = False
        self.active_speak_kind = "speak"
        self.end_requested: bool = False
        self.end_requested_at: float = 0.0
        self.closing_heard: bool = False
        self.ending_sent: bool = False


class _Cursor:
    def __init__(self) -> None:
        self.state = new_cursor_state()


class _SessionProxy:
    def __init__(self, session) -> None:
        self.current = session

    def __getattr__(self, name: str):
        return getattr(self.current, name)


def live_settings(overrides: dict | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_LIVE_SETTINGS)
    for env_key, setting_key in _LIVE_SETTING_ENV_KEYS:
        raw = os.environ.get(env_key)
        if raw is not None and raw.strip():
            merged[setting_key] = raw.strip()
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[str(key)] = str(value)
    return merged


def _session_settings(settings: dict | None = None) -> dict:
    merged = live_settings(settings)
    start_raw = merged.get("vad_start_sensitivity")
    if start_raw in (None, ""):
        start_raw = "LOW"
    end_raw = merged.get("vad_end_sensitivity")
    if end_raw in (None, ""):
        end_raw = "LOW"
    start_low = str(start_raw).strip().upper() == "LOW"
    end_low = str(end_raw).strip().upper() == "LOW"
    silence_raw = merged.get("vad_silence_ms")
    silence_ms = 1000 if silence_raw in (None, "") else int(float(silence_raw))
    return {
        "model": merged.get("gemini_model") or "gemini-3.1-flash-live-preview",
        "voice": str(merged.get("gemini_voice") or "Kore").strip() or "Kore",
        "prefix_padding_ms": int(float(merged.get("vad_prefix_padding_ms") or 300)),
        "silence_duration_ms": silence_ms,
        "start_sens": (
            types.StartSensitivity.START_SENSITIVITY_LOW
            if start_low
            else types.StartSensitivity.START_SENSITIVITY_HIGH
        ),
        "end_sens": (
            types.EndSensitivity.END_SENSITIVITY_LOW
            if end_low
            else types.EndSensitivity.END_SENSITIVITY_HIGH
        ),
        "watchdog_timeout_sec": float(merged.get("watchdog_timeout_sec") or 8.0),
    }


def _live_config(
    settings: dict | None = None,
    resume_handle: str | None = None,
) -> types.LiveConnectConfig:
    chosen = _session_settings(settings)
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        temperature=0.5,
        enable_affective_dialog=False,
        speech_config=types.SpeechConfig(
            language_code="hi-IN",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=chosen["voice"]
                )
            ),
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=GEMINI_LIVE_SYSTEM_PROMPT)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["hi-IN"],
            custom_vocabulary=list(HINDI_MEDICAL_VOCAB),
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["hi-IN"],
        ),
        realtime_input_config=types.RealtimeInputConfig(
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=chosen["start_sens"],
                end_of_speech_sensitivity=chosen["end_sens"],
                prefix_padding_ms=chosen["prefix_padding_ms"],
                silence_duration_ms=chosen["silence_duration_ms"],
            ),
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle or None),
        tools=[{"function_declarations": [_NEXT_QUESTION_DECL]}],
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )


def _new_telemetry(
    model: str,
    voice: str,
    silence_ms: int,
    prefix_ms: int,
    connect_ms: int,
) -> dict:
    return {
        "model": model,
        "voice": voice,
        "vad_silence_ms": silence_ms,
        "vad_prefix_padding_ms": prefix_ms,
        "connect_ms": connect_ms,
        "user_turns": 0,
        "bot_turns": 0,
        "tool_calls": 0,
        "audio_chunks_in": 0,
        "audio_chunks_out": 0,
        "interruptions": 0,
        "watchdog_nudges": 0,
        "turn_latencies_ms": [],
        "avg_latency_ms": 0,
        "max_latency_ms": 0,
        "last_interaction_status": "",
        "dropoff_qid": "",
        "reconnects": 0,
        "errors": [],
    }


def _stamp_dropoff(telemetry: dict, state: dict | None) -> None:
    """The question she was still on when the call ended before it was complete."""
    cursor_state = state or {}
    reason = str(cursor_state.get("close_reason") or "")
    if reason == "complete":
        telemetry["dropoff_qid"] = ""
    else:
        telemetry["dropoff_qid"] = str(cursor_state.get("active_qid") or "")


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _refresh_latency(telemetry: dict) -> None:
    latencies = telemetry.get("turn_latencies_ms") or []
    if latencies:
        telemetry["avg_latency_ms"] = int(round(sum(latencies) / len(latencies)))
        telemetry["max_latency_ms"] = int(max(latencies))
    else:
        telemetry["avg_latency_ms"] = 0
        telemetry["max_latency_ms"] = 0


def _ws_connected(ws: WebSocket) -> bool:
    return ws.client_state == WebSocketState.CONNECTED


async def _send_json(ws: WebSocket, lock: asyncio.Lock, payload: dict) -> None:
    if not _ws_connected(ws):
        return
    try:
        async with lock:
            if not _ws_connected(ws):
                return
            await ws.send_json(payload)
    except Exception:
        return


async def _close_ws(ws: WebSocket) -> None:
    if not _ws_connected(ws):
        return
    try:
        await ws.close()
    except Exception:
        return


def _audio_from_part(part: types.Part) -> tuple[bytes, int] | None:
    blob = part.inline_data
    if blob is None or blob.data is None:
        return None
    mime = blob.mime_type or ""
    if mime and mime.lower().partition("/")[0] != "audio":
        return None
    pcm = audio_recorder.pcm_bytes(blob.data)
    if not pcm:
        return None
    return pcm, audio_recorder.rate_from_mime(mime, _OUTPUT_RATE)


def _caller_heard(state: _TurnState) -> None:
    """She is speaking now, so this is not a silent line."""
    state.quiet_from = 0.0


def _arm_waiting_reply(
    state: _TurnState,
    *,
    speech_end_waiting: bool = False,
    now_ts: float | None = None,
) -> None:
    state.awaiting_bot_reply = True
    state.speech_end_waiting = speech_end_waiting
    state.last_user_turn_ts = now_ts if now_ts is not None else time.time()
    state.bot_replied_since_user = False
    state.nudges_for_current_turn = 0


def _arm_user_turn(state: _TurnState, telemetry: dict) -> None:
    if not state.awaiting_bot_reply or state.speech_end_waiting:
        telemetry["user_turns"] = int(telemetry.get("user_turns") or 0) + 1
    _arm_waiting_reply(state, speech_end_waiting=False)
    _caller_heard(state)
    state.checked_in = False


async def _send_realtime_safe(
    session,
    send_lock: asyncio.Lock,
    state: _TurnState,
    stop: asyncio.Event,
    **kwargs,
) -> bool:
    try:
        async with send_lock:
            await session.send_realtime_input(**kwargs)
        return True
    except Exception:
        if state.resume_handle and not stop.is_set():
            state.reconnect_requested = True
            return True
        return False



async def _emit(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    event_type: str,
    detail: str = "",
    meta: dict | None = None,
) -> None:
    if event_type not in _TELEMETRY_EVENTS:
        return
    telemetry["last_interaction_status"] = event_type
    _refresh_latency(telemetry)
    try:
        evt = await asyncio.to_thread(
            db.append_call_event, call_id, event_type, detail, meta
        )
    except Exception:
        evt = {
            "t": round(time.time(), 3),
            "type": event_type,
            "detail": detail or "",
            "meta": meta or {},
        }
    try:
        await asyncio.to_thread(db.update_call_telemetry, call_id, _json_copy(telemetry))
    except Exception:
        pass
    snapshot = json.loads(json.dumps(telemetry))
    await _send_json(
        ws,
        ws_lock,
        {"type": "telemetry", "telemetry": snapshot, "last_event": evt},
    )


async def _report_error(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int | None,
    telemetry: dict | None,
    message: str,
) -> None:
    if call_id is not None and telemetry is not None:
        errors = telemetry.setdefault("errors", [])
        errors.append(message)
        if len(errors) > 50:
            del errors[:-50]
        await _emit(ws, ws_lock, call_id, telemetry, "error", message)
    await _send_json(ws, ws_lock, {"type": "error", "message": message})


async def _forward_transcript(
    ws: WebSocket,
    lock: asyncio.Lock,
    call_id: int,
    role: str,
    transcription: types.Transcription | None,
    cursor: _Cursor,
) -> str:
    if transcription is None:
        return ""
    text = " ".join((transcription.text or "").split())
    if not text:
        return ""
    qid = str(cursor.state.get("active_qid") or "")
    await asyncio.to_thread(db.append_transcript, call_id, role, text, qid=qid)
    await _send_json(ws, lock, {"type": "transcript", "role": role, "text": text})
    return text


def _in_quick_reply_stale_window(state: _TurnState, cursor_state: dict) -> bool:
    return bool(
        state.last_quick_reply_step
        and (
            not _question_already_spoken(state, cursor_state)
            or state.ignore_bot_audio
            or time.time() - state.last_quick_reply_ts < 1.2
        )
    )


def _accept_end(state: _TurnState) -> bool:
    """Mark the call closing. True only the first time end_call is accepted."""
    first = not state.end_requested
    state.end_requested = True
    state.end_requested_at = time.time()
    return first


def _question_group(state: dict) -> str:
    qid = str(state.get("active_qid") or "")
    if qid == "ever_pregnant":
        return "pregnancy"
    if not qid:
        return ""
    try:
        return get_spec(qid).group
    except KeyError:
        return ""


def _question_state_payload(cursor_state: dict) -> dict:
    return {
        "type": "question_state",
        "step": step_id(cursor_state),
        "qid": str(cursor_state.get("active_qid") or ""),
        "group": _question_group(cursor_state),
        "expect_yes_no": expect_yes_no(cursor_state),
        "prompt_chars": len(prompts.say_hindi(cursor_state)),
    }


async def _send_greeting(
    session,
    send_lock,
    ws,
    ws_lock,
    cursor: _Cursor,
    state: _TurnState | None = None,
) -> None:
    if state is not None:
        state.capture.speech_allowed = True
    async with send_lock:
        await session.send_realtime_input(
            text=prompts.GREETING_CUE + prompts.say_hindi(cursor.state)
        )
    await _send_json(ws, ws_lock, _question_state_payload(cursor.state))


def _question_already_spoken(turn: _TurnState, cursor_state: dict) -> bool:
    step = step_id(cursor_state)
    return bool(step) and turn.spoken_qid == step


def _is_stale_user_turn(turn: _TurnState, cursor_state: dict) -> bool:
    current_step = step_id(cursor_state)
    mismatch = (
        turn.user_turn_step != current_step
        if turn.user_turn_step
        else bool(turn.last_quick_reply_step)
    )
    return bool(mismatch or turn.capture.user_turn_is_quick_reply_stale)


def _tool_body(state: dict, *, outcome: str = "") -> dict:
    body: dict[str, Any] = {
        "result": "ok",
        "say": prompts.say_hindi(state),
    }
    if state.get("phase") == "closing":
        body["stop"] = True
    qid = str(state.get("active_qid") or "")
    if outcome == "clarify" or qid in ("fam_same_problem", "close_anything_else"):
        body["recorded_concerns"] = prompts.recorded_concerns_hi(state)
    return body


def _cancel_speak_fallback(turn: _TurnState) -> None:
    task = turn.speak_fallback
    turn.speak_fallback = None
    if task is None:
        return
    if not task.done():
        task.cancel()
    task.add_done_callback(consume_task_error)


async def _send_speak_cue(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    cursor: _Cursor,
    turn: _TurnState,
    stop: asyncio.Event | None,
    *,
    kind: str,
    text: str | None = None,
    error_msg: str = "Could not prompt Gemini to speak",
    record_tool_error: bool = True,
) -> bool:
    step_now = step_id(cursor.state)
    cue_text = text if text is not None else prompts.speak_cue_text(cursor.state, kind)
    turn.active_speak_kind = kind
    turn.pending_speak_step = ""
    turn.capture.speech_allowed = True
    try:
        ok = await _send_realtime_safe(
            session, send_lock, turn, stop or asyncio.Event(), text=cue_text
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        ok = False
    if ok:
        if not turn.reconnect_requested:
            turn.speak_cue_step = step_now
        return True
    if record_tool_error:
        telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
    await _report_error(ws, ws_lock, call_id, telemetry, error_msg)
    return False


async def _fallback_speak_after_tool(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    cursor: _Cursor,
    turn: _TurnState,
    stop: asyncio.Event | None,
    *,
    step_after_tool: str,
    kind: str,
    delay_sec: float,
) -> None:
    """Speak the tool's next line only when Gemini stayed silent after the tool result."""
    try:
        if delay_sec > 0:
            if stop is None:
                await asyncio.sleep(delay_sec)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay_sec)
                    return
                except asyncio.TimeoutError:
                    pass
        if stop is not None and stop.is_set():
            return
        if turn.bot_replied_since_user or turn.bot_speaking:
            return
        if step_id(cursor.state) != step_after_tool:
            return
        await _send_speak_cue(
            session,
            send_lock,
            ws,
            ws_lock,
            call_id,
            telemetry,
            cursor,
            turn,
            stop,
            kind=kind,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
        await _report_error(ws, ws_lock, call_id, telemetry, "Could not prompt Gemini to speak")


async def _mark_closing(
    cursor: _Cursor,
    reason: str,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    turn: _TurnState,
) -> None:
    state = cursor.state
    if state.get("phase") != "closing":
        state["phase"] = "closing"
        state["close_reason"] = reason
    elif not state.get("close_reason"):
        state["close_reason"] = reason
    cursor.state = state
    _stamp_dropoff(telemetry, state)
    try:
        await asyncio.to_thread(db.save_cursor_state, call_id, _json_copy(state))
    except Exception:
        pass
    if _accept_end(turn):
        close_reason = str(state.get("close_reason") or reason)
        await _emit(
            ws,
            ws_lock,
            call_id,
            telemetry,
            "end_accepted",
            close_reason,
            {"reason": close_reason},
        )


async def _answer_function_calls(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    calls: list,
    seen: set[str],
    cancelled: set[str],
    cursor: _Cursor,
    turn: _TurnState,
    stop: asyncio.Event | None = None,
    settings: dict | None = None,
    thresholds: decisions.DecisionThresholds | None = None,
) -> bool:
    cap = turn.capture
    responses: list[types.FunctionResponse] = []
    logged: list[str] = []
    clarify_applied = False
    handled = False
    stale_only = True
    decision_ran = False
    for fc in calls:
        fc_id = getattr(fc, "id", None) or ""
        if fc_id and (fc_id in cancelled or fc_id in seen):
            continue
        if getattr(fc, "will_continue", False):
            continue
        if fc_id:
            seen.add(fc_id)
        name = getattr(fc, "name", None) or ""
        try:
            if name == "next_question":
                stale_turn = _is_stale_user_turn(turn, cursor.state)
                if cursor.state.get("phase") != "asking":
                    cap.speech_allowed = True
                    body = _tool_body(cursor.state)
                    logged.append("next_question_closing")
                    handled = True
                    stale_only = False
                elif cap.handled_turn and not _in_quick_reply_stale_window(
                    turn, cursor.state
                ):
                    body = _tool_body(cursor.state)
                    body["result"] = "held"
                    body["reason"] = "duplicate_turn"
                    logged.append("next_question_held")
                    cap.speech_allowed = True
                    handled = True
                elif stale_turn:
                    body = _tool_body(cursor.state)
                    body["result"] = "held"
                    body["reason"] = "stale_turn"
                    logged.append("next_question_held")
                    handled = True
                else:
                    stale_only = False
                    heard = _question_already_spoken(turn, cursor.state)
                    if not heard:
                        body = _tool_body(cursor.state)
                        body["result"] = "held"
                        body["reason"] = "not_spoken"
                        logged.append("next_question_held")
                        cap.speech_allowed = True
                        handled = True
                    else:
                        async def _mark_closing_cb(reason: str) -> None:
                            await _mark_closing(
                                cursor, reason, ws, ws_lock, call_id, telemetry, turn
                            )

                        body, call_clarify, event_kind = await run_decision_turn(
                            cursor,
                            cap,
                            call_id,
                            tool_body_fn=_tool_body,
                            mark_closing_fn=_mark_closing_cb,
                            is_closing=turn.end_requested,
                            settings=settings,
                            thresholds=thresholds,
                            input_rate=_INPUT_RATE,
                        )
                        decision_ran = True
                        if call_clarify:
                            clarify_applied = True
                        if event_kind == "held":
                            logged.append("next_question_held")
                        elif event_kind == "technical_failure":
                            logged.append("next_question_technical_failure")
                        else:
                            logged.append(f"next_question_{event_kind}")
                        handled = True
            else:
                body = {"result": "ignored", "say": prompts.say_hindi(cursor.state)}
                logged.append(name or "ignored")
                telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
        except Exception:
            logger.exception("tool %s failed on call %s", name, call_id)
            telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
            body = {"result": "error", "say": prompts.say_hindi(cursor.state)}
            logged.append(f"{name or 'tool'}_error")
            clarify_applied = False
            stale_only = False
            handled = True
        if cursor.state.get("phase") == "closing":
            body["stop"] = True
            body["say"] = prompts.closing_hi(cursor.state)
        if not fc_id:
            continue
        responses.append(
            types.FunctionResponse(
                id=fc_id,
                name=name or "next_question",
                response=body,
            )
        )
    send_failed = False
    if responses:
        try:
            async with send_lock:
                await session.send_tool_response(function_responses=responses)
        except Exception:
            send_failed = True
    for detail in logged:
        telemetry["tool_calls"] = int(telemetry.get("tool_calls") or 0) + 1
        await _emit(ws, ws_lock, call_id, telemetry, "tool_call", detail)
    if send_failed:
        telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
        await _report_error(ws, ws_lock, call_id, telemetry, "Could not send the tool response to Gemini")
    if handled:
        step_after_tool = step_id(cursor.state)
        cue_sent = bool(step_after_tool) and turn.speak_cue_step == step_after_tool
        already_spoken = _question_already_spoken(turn, cursor.state)
        # A late tool call after quick_reply already spoke this step. Keep the
        # FunctionResponse and the tool_call event, and do not speak it again.
        if stale_only and (cue_sent or already_spoken):
            if cap.user_turn_is_quick_reply_stale:
                cap.reset()
            return decision_ran
        await _send_json(ws, ws_lock, _question_state_payload(cursor.state))
        kind = "speak"
        if cursor.state.get("phase") != "asking":
            kind = "close"
        elif clarify_applied:
            kind = "clarify"
        delay = 0.0 if send_failed else _TOOL_SPEAK_FALLBACK_SEC
        turn.pending_speak_step = step_after_tool
        turn.pending_speak_kind = kind
        turn.active_speak_kind = kind
        if delay <= 0:
            await _fallback_speak_after_tool(
                session, send_lock, ws, ws_lock, call_id, telemetry, cursor, turn, stop,
                step_after_tool=step_after_tool, kind=kind, delay_sec=0,
            )
        else:
            _cancel_speak_fallback(turn)
            turn.speak_fallback = asyncio.create_task(
                _fallback_speak_after_tool(
                    session, send_lock, ws, ws_lock, call_id, telemetry, cursor, turn, stop,
                    step_after_tool=step_after_tool, kind=kind, delay_sec=delay,
                )
            )
        if stale_only and cap.user_turn_is_quick_reply_stale:
            cap.reset()
    return decision_ran


async def _on_bot_audio(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
    recorder: audio_recorder.CallAudioRecorder,
    pcm: bytes,
    rate: int,
    cursor: _Cursor,
) -> None:
    if state.ignore_bot_audio or not state.capture.speech_allowed:
        return
    state.pending_speak_step = ""
    telemetry["audio_chunks_out"] += 1
    first = not state.bot_replied_since_user
    latency_ms = None
    if first:
        telemetry["bot_turns"] += 1
        if state.awaiting_bot_reply and state.last_user_turn_ts > 0:
            latency_ms = int((time.time() - state.last_user_turn_ts) * 1000)
            if latency_ms < 0:
                latency_ms = 0
            telemetry["turn_latencies_ms"].append(latency_ms)
        state.nudges_for_current_turn = 0
        state.spoken_qid = step_id(cursor.state)
    state.awaiting_bot_reply = False
    state.speech_end_waiting = False
    state.bot_replied_since_user = True
    state.bot_speaking = True
    dur = len(pcm) / max(1, 2 * rate)
    state.playback_ends_at = max(time.time(), state.playback_ends_at) + dur
    recorder.add_bot_pcm(pcm, rate)
    if state.end_requested:
        state.closing_heard = True
        telemetry["closing_heard"] = True
    await _send_json(
        ws,
        ws_lock,
        {
            "type": "audio",
            "data": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": _OUTPUT_RATE,
        },
    )
    if first:
        detail = f"{latency_ms}ms" if latency_ms is not None else ""
        meta = {"spoken_qid": state.spoken_qid}
        if latency_ms is not None:
            meta["latency_ms"] = latency_ms
        await _emit(
            ws, ws_lock, call_id, telemetry, "bot_first_audio", detail, meta
        )


async def _handle_server_message(
    response,
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
    recorder: audio_recorder.CallAudioRecorder,
    seen: set[str],
    cancelled: set[str],
    cursor: _Cursor,
    stop: asyncio.Event | None = None,
    settings: dict | None = None,
    thresholds: decisions.DecisionThresholds | None = None,
) -> bool:
    """Handle one Live server message. Return False when the session should stop."""
    cap = state.capture
    decision_ran_in_msg = False
    cancellation = getattr(response, "tool_call_cancellation", None)
    if cancellation is not None and cancellation.ids:
        cancelled.update(item for item in cancellation.ids if item)

    server_content = getattr(response, "server_content", None)
    if server_content is not None and server_content.interrupted:
        state.ignore_bot_audio = False
        state.bot_speaking = False
        state.playback_ends_at = 0.0
        _caller_heard(state)
        telemetry["interruptions"] += 1
        await _send_json(ws, ws_lock, {"type": "clear"})
        await _emit(ws, ws_lock, call_id, telemetry, "interrupted", "interrupted")

    if server_content is not None:
        user_text = ""
        input_tr = getattr(server_content, "input_transcription", None)
        input_finished = (
            bool(getattr(input_tr, "finished", False))
            if input_tr is not None
            else False
        )
        if not state.end_requested:
            user_text = await _forward_transcript(
                ws,
                ws_lock,
                call_id,
                "user",
                input_tr,
                cursor,
            )
        if user_text:
            if _in_quick_reply_stale_window(state, cursor.state):
                if cap.user_turn_finalized:
                    cap.reset(clear_pcm=False)
                state.user_turn_step = state.last_quick_reply_step
                cap.user_turn_is_quick_reply_stale = True
                _caller_heard(state)
            else:
                cap.begin_transcript_chunk(cursor.state, user_text)
                state.user_turn_step = step_id(cursor.state)
                state.last_quick_reply_step = ""
                _arm_user_turn(state, telemetry)
            await _emit(
                ws, ws_lock, call_id, telemetry, "user_transcript", user_text
            )
        if input_finished and not state.end_requested:
            cap.user_turn_finalized = True
        drop_bot_output = state.ignore_bot_audio or not cap.speech_allowed
        if not drop_bot_output:
            await _forward_transcript(
                ws,
                ws_lock,
                call_id,
                "assistant",
                server_content.output_transcription,
                cursor,
            )

    parts = []
    if (
        server_content is not None
        and server_content.model_turn is not None
        and server_content.model_turn.parts
    ):
        parts = server_content.model_turn.parts

    part_calls = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        audio = _audio_from_part(part)
        if audio is not None:
            pcm, rate = audio
            await _on_bot_audio(
                ws,
                ws_lock,
                call_id,
                telemetry,
                state,
                recorder,
                pcm,
                rate,
                cursor,
            )
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            part_calls.append(function_call)

    tool_call = getattr(response, "tool_call", None)
    calls = list(part_calls)
    if tool_call is not None and tool_call.function_calls:
        calls.extend(tool_call.function_calls)
    if calls:
        decision_ran_in_msg = await _answer_function_calls(
            session,
            send_lock,
            ws,
            ws_lock,
            call_id,
            telemetry,
            calls,
            seen,
            cancelled,
            cursor,
            state,
            stop=stop,
            settings=settings,
            thresholds=thresholds,
        )
        if not state.end_requested:
            _arm_waiting_reply(state, speech_end_waiting=False)
            state.quiet_from = 0.0

    if server_content is not None and server_content.turn_complete:
        if not state.end_requested:
            stale_turn = _is_stale_user_turn(state, cursor.state)
            has_turn_input = bool(cap.user_turn_buffer.strip()) or bool(cap.user_turn_pcm)
            if (
                cursor.state.get("phase") == "asking"
                and not stale_turn
                and not cap.handled_turn
                and _question_already_spoken(state, cursor.state)
                and has_turn_input
            ):
                async def _mark_closing_cb(reason: str) -> None:
                    await _mark_closing(
                        cursor, reason, ws, ws_lock, call_id, telemetry, state
                    )

                _body, clarify_applied, _kind = await run_decision_turn(
                    cursor,
                    cap,
                    call_id,
                    tool_body_fn=_tool_body,
                    mark_closing_fn=_mark_closing_cb,
                    is_closing=state.end_requested,
                    settings=settings,
                    thresholds=thresholds,
                    input_rate=_INPUT_RATE,
                )
                decision_ran_in_msg = True
                telemetry["tool_calls"] = int(telemetry.get("tool_calls") or 0) + 1
                await _emit(ws, ws_lock, call_id, telemetry, "tool_call", f"missed_tool_{_kind}")
                await _send_json(ws, ws_lock, _question_state_payload(cursor.state))
                missed_kind = "speak"
                if cursor.state.get("phase") != "asking":
                    missed_kind = "close"
                elif clarify_applied:
                    missed_kind = "clarify"
                _cancel_speak_fallback(state)
                if not state.end_requested:
                    _arm_waiting_reply(state, speech_end_waiting=False)
                    state.quiet_from = 0.0
                if not await _send_speak_cue(
                    session,
                    send_lock,
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    cursor,
                    state,
                    stop,
                    kind=missed_kind,
                ):
                    if stop is not None:
                        stop.set()
                    return False

        if not decision_ran_in_msg and (
            cap.user_turn_is_quick_reply_stale
            or (state.user_turn_step and state.user_turn_step != step_id(cursor.state))
        ):
            cap.reset()
        cap.handled_turn = False
        cap.last_key = None
        state.ignore_bot_audio = False
        if state.go_away_pending:
            state.go_away_pending = False
            state.reconnect_requested = True
            _cancel_speak_fallback(state)
        speak_step = state.pending_speak_step
        speak_kind = state.pending_speak_kind
        stop_is_set = stop is not None and stop.is_set()
        if (
            speak_step
            and speak_step == step_id(cursor.state)
            and not state.bot_replied_since_user
            and not stop_is_set
            and not state.reconnect_requested
        ):
            _cancel_speak_fallback(state)
            if not await _send_speak_cue(
                session,
                send_lock,
                ws,
                ws_lock,
                call_id,
                telemetry,
                cursor,
                state,
                stop,
                kind=speak_kind,
            ):
                if stop is not None:
                    stop.set()
                return False
        state.bot_speaking = False
        if not state.awaiting_bot_reply:
            state.quiet_from = max(time.time(), state.playback_ends_at)
            await _send_json(
                ws,
                ws_lock,
                {
                    "type": "bot_turn_complete",
                    "step": step_id(cursor.state),
                },
            )
        if not state.awaiting_bot_reply or state.end_requested:
            await _emit(ws, ws_lock, call_id, telemetry, "turn_complete", "")

    update = getattr(response, "session_resumption_update", None)
    if update is not None and getattr(update, "resumable", False) and getattr(update, "new_handle", None):
        state.resume_handle = str(update.new_handle)
    if getattr(response, "go_away", None) is not None:
        if not state.resume_handle:
            return False
        if state.bot_speaking or state.awaiting_bot_reply:
            state.go_away_pending = True
        else:
            state.reconnect_requested = True
    return True


async def _gemini_loop(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    stop: asyncio.Event,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
    recorder: audio_recorder.CallAudioRecorder,
    cursor: _Cursor,
    reconnect,
    settings: dict | None = None,
    thresholds: decisions.DecisionThresholds | None = None,
) -> None:
    seen: set[str] = set()
    cancelled: set[str] = set()
    reconnects = 0
    while not stop.is_set():
        try:
            async for response in session.receive():
                if stop.is_set():
                    return
                keep_going = await _handle_server_message(
                    response,
                    session,
                    send_lock,
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    state,
                    recorder,
                    seen,
                    cancelled,
                    cursor,
                    stop,
                    settings=settings,
                    thresholds=thresholds,
                )
                if not keep_going:
                    stop.set()
                    return
                if state.reconnect_requested:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            if stop.is_set() or not state.resume_handle:
                stop.set()
                await _report_error(
                    ws, ws_lock, call_id, telemetry, "Gemini Live session ended"
                )
                return
            state.reconnect_requested = True
        if not state.reconnect_requested:
            continue
        state.reconnect_requested = False
        _cancel_speak_fallback(state)
        reconnects += 1
        await _send_json(ws, ws_lock, {"type": "reconnecting", "attempt": reconnects})
        await _emit(ws, ws_lock, call_id, telemetry, "reconnecting", str(reconnects))
        ok = reconnects <= 5 and await reconnect(state.resume_handle)
        if not ok:
            stop.set()
            await _report_error(
                ws, ws_lock, call_id, telemetry, "Gemini Live could not reconnect"
            )
            return
        resume_speak = (
            state.bot_speaking
            or state.awaiting_bot_reply
            or bool(state.pending_speak_step)
        )
        telemetry["reconnects"] = reconnects
        state.bot_speaking = False
        state.ignore_bot_audio = False
        state.awaiting_bot_reply = False
        state.speech_end_waiting = False
        state.playback_ends_at = 0.0
        state.quiet_from = time.time()
        state.capture.reset()
        seen.clear()
        cancelled.clear()
        await _send_json(ws, ws_lock, {"type": "reconnected", "attempt": reconnects})
        await _emit(ws, ws_lock, call_id, telemetry, "reconnected", str(reconnects))
        if (
            resume_speak
            and cursor.state.get("phase") == "asking"
            and not state.end_requested
        ):
            _arm_waiting_reply(state, speech_end_waiting=False)
            await _send_json(ws, ws_lock, _question_state_payload(cursor.state))
            if not await _send_speak_cue(
                session,
                send_lock,
                ws,
                ws_lock,
                call_id,
                telemetry,
                cursor,
                state,
                stop,
                kind=state.active_speak_kind,
                record_tool_error=False,
            ):
                stop.set()
                return


async def _watchdog_loop(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    stop: asyncio.Event,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
    cursor: _Cursor,
    watchdog_timeout_sec: float,
) -> None:
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=_WATCHDOG_INTERVAL_SEC)
                return
            except asyncio.TimeoutError:
                pass
            if stop.is_set():
                return
            now = time.time()
            if now - state.call_started_ts >= _HARD_LIMIT_SEC:
                if cursor.state.get("phase") != "closing":
                    await _mark_closing(
                        cursor, "time_limit", ws, ws_lock, call_id, telemetry, state
                    )
                stop.set()
                return
            if cursor.state.get("phase") == "closing" or state.end_requested:
                played = _closing_audio_finished(state, now)
                waited = now - float(state.end_requested_at or now)
                if played or waited > _CLOSING_MAX_WAIT_SEC:
                    await _notify_ending(ws, ws_lock, state)
                    stop.set()
                    return
                continue
            cue = ""
            reason = ""
            if now - state.call_started_ts >= _SOFT_LIMIT_SEC and not state.time_cue_sent:
                reason = "time_limit"
                state.time_cue_sent = True
            elif state.quiet_from > 0 and not state.bot_speaking and not state.awaiting_bot_reply:
                quiet = now - state.quiet_from
                if quiet >= _SILENCE_CLOSE_SEC:
                    reason = "silence"
                elif quiet >= _CHECK_IN_SEC and not state.checked_in:
                    cue = prompts.SILENCE_CHECK_IN_CUE
            if reason:
                await _mark_closing(
                    cursor, reason, ws, ws_lock, call_id, telemetry, state
                )
                cue = prompts.speak_cue_text(cursor.state, "close")
            if cue:
                if not reason:
                    state.checked_in = True
                state.capture.speech_allowed = True
                ok = await _send_realtime_safe(
                    session, send_lock, state, stop, text=cue
                )
                if not ok:
                    stop.set()
                    return
                if state.reconnect_requested:
                    continue
                await _emit(
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    "auto_close" if reason else "silence_check_in",
                    reason or "check_in",
                )
                continue
            if not state.awaiting_bot_reply or state.bot_speaking:
                continue
            if state.speech_end_waiting and _question_already_spoken(state, cursor.state):
                state.awaiting_bot_reply = False
                state.speech_end_waiting = False
                continue
            if state.bot_replied_since_user or state.last_user_turn_ts <= 0:
                continue
            effective_timeout = (
                min(watchdog_timeout_sec, 6.0)
                if state.speech_end_waiting
                else watchdog_timeout_sec
            )
            if state.nudges_for_current_turn >= 6:
                continue
            if time.time() - state.last_user_turn_ts < effective_timeout:
                continue
            state.nudges_for_current_turn += 1
            telemetry["watchdog_nudges"] += 1
            state.last_user_turn_ts = time.time()
            nudge = prompts.speak_cue_text(cursor.state, state.active_speak_kind)
            if not await _send_speak_cue(
                session,
                send_lock,
                ws,
                ws_lock,
                call_id,
                telemetry,
                cursor,
                state,
                stop,
                kind=state.active_speak_kind,
                text=nudge,
                error_msg="Watchdog nudge failed",
                record_tool_error=False,
            ):
                stop.set()
                return
            if state.reconnect_requested:
                continue
            await _emit(
                ws,
                ws_lock,
                call_id,
                telemetry,
                "watchdog_nudge",
                nudge,
            )
    except asyncio.CancelledError:
        raise


async def _browser_loop(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    stop: asyncio.Event,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
    cursor: _Cursor,
    recorder: audio_recorder.CallAudioRecorder,
) -> None:
    try:
        while not stop.is_set():
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                stop.set()
                return
            except RuntimeError:
                stop.set()
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _report_error(
                    ws, ws_lock, call_id, telemetry, "Invalid JSON"
                )
                continue
            if not isinstance(payload, dict):
                await _report_error(
                    ws, ws_lock, call_id, telemetry, "Expected a JSON object"
                )
                continue
            kind = payload.get("type")
            if state.end_requested and kind in {"audio", "text", "speech_end", "nudge"}:
                continue
            if kind == "audio":
                pcm = audio_recorder.pcm_bytes(payload.get("data") or "")
                if not pcm:
                    continue
                try:
                    rate = int(payload.get("sample_rate") or _INPUT_RATE)
                except (TypeError, ValueError):
                    rate = _INPUT_RATE
                if rate <= 0:
                    rate = _INPUT_RATE
                telemetry["audio_chunks_in"] += 1
                recorder.add_user_pcm(pcm, rate)
                pcm_16k = (
                    pcm
                    if rate == _INPUT_RATE
                    else audio_recorder.resample_int16_mono(pcm, rate, _INPUT_RATE)
                )
                in_quick_stale = _in_quick_reply_stale_window(state, cursor.state)
                cap = state.capture
                if audio_recorder.pcm_rms(pcm) >= 0.02:
                    _caller_heard(state)
                    if not in_quick_stale:
                        cap.begin_audio_speech(cursor.state, pcm_16k)
                        state.user_turn_step = step_id(cursor.state)
                        state.last_quick_reply_step = ""
                elif not in_quick_stale and cap.user_turn_pcm_tail > 0:
                    cap.user_turn_pcm_tail -= 1
                    cap.append_pcm(pcm_16k)
                ok = await _send_realtime_safe(
                    session,
                    send_lock,
                    state,
                    stop,
                    audio=types.Blob(
                        data=pcm,
                        mime_type=f"audio/pcm;rate={rate}",
                    ),
                )
                if not ok:
                    stop.set()
                    await _report_error(
                        ws,
                        ws_lock,
                        call_id,
                        telemetry,
                        "Could not forward audio to Gemini",
                    )
                    return
            elif kind == "speech_end":
                now_ts = time.time()
                already = _question_already_spoken(state, cursor.state)
                playback_done = (
                    not state.bot_speaking and now_ts >= state.playback_ends_at - 0.1
                )
                if playback_done and not already and not state.awaiting_bot_reply:
                    _arm_waiting_reply(state, speech_end_waiting=True, now_ts=now_ts)
                if playback_done:
                    state.quiet_from = max(now_ts, state.playback_ends_at)
                try:
                    await asyncio.to_thread(
                        db.append_call_event,
                        call_id,
                        "speech_end",
                        "already_spoken" if already else "",
                    )
                except Exception:
                    pass
            elif kind == "nudge":
                if state.awaiting_bot_reply:
                    state.last_user_turn_ts = time.time()
                nudge = prompts.speak_cue_text(cursor.state, state.active_speak_kind)
                if not await _send_speak_cue(
                    session,
                    send_lock,
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    cursor,
                    state,
                    stop,
                    kind=state.active_speak_kind,
                    text=nudge,
                    error_msg="Could not nudge Gemini",
                    record_tool_error=False,
                ):
                    stop.set()
                    return
                if state.reconnect_requested:
                    continue
                await _emit(
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    "manual_nudge",
                    nudge,
                )
            elif kind == "text":
                text = str(payload.get("text") or "").strip()
                if not text:
                    continue
                qid = str(cursor.state.get("active_qid") or "")
                await asyncio.to_thread(db.append_transcript, call_id, "user", text, qid=qid)
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "transcript", "role": "user", "text": text},
                )
                state.capture.begin_text_turn(cursor.state, text)
                state.user_turn_step = step_id(cursor.state)
                state.last_quick_reply_step = ""
                _arm_user_turn(state, telemetry)
                try:
                    async with send_lock:
                        await session.send_realtime_input(text=text)
                except Exception:
                    stop.set()
                    await _report_error(
                        ws,
                        ws_lock,
                        call_id,
                        telemetry,
                        "Could not forward text to Gemini",
                    )
                    return
                await _emit(
                    ws, ws_lock, call_id, telemetry, "user_transcript", text
                )
            elif kind == "quick_reply":
                if state.end_requested or cursor.state.get("phase") != "asking":
                    continue
                outcome = str(payload.get("outcome") or "").strip().lower()
                if outcome not in ("clear", "concern"):
                    continue
                if not expect_yes_no(cursor.state):
                    continue
                if payload.get("step") != step_id(cursor.state):
                    continue
                if not _question_already_spoken(state, cursor.state):
                    continue
                if state.bot_speaking:
                    state.ignore_bot_audio = True
                    state.bot_speaking = False
                    state.playback_ends_at = 0.0
                state.user_turn_step = ""
                state.pending_speak_step = ""
                state.capture.reset()
                state.last_quick_reply_step = str(payload.get("step") or "")
                state.last_quick_reply_ts = time.time()
                said = "नहीं।" if outcome == "clear" else "हाँ।"
                qid = str(cursor.state.get("active_qid") or "")
                await asyncio.to_thread(db.append_transcript, call_id, "user", said, qid=qid)
                await _send_json(
                    ws, ws_lock, {"type": "transcript", "role": "user", "text": said}
                )
                await _emit(ws, ws_lock, call_id, telemetry, "user_transcript", said)
                tool_name = "quick_reply_clear" if outcome == "clear" else "quick_reply_concern"
                telemetry["tool_calls"] = int(telemetry.get("tool_calls") or 0) + 1
                await _emit(ws, ws_lock, call_id, telemetry, "tool_call", tool_name)
                cursor.state = apply_turn(cursor.state, outcome)
                try:
                    await asyncio.to_thread(
                        db.save_cursor_state, call_id, _json_copy(cursor.state)
                    )
                except Exception:
                    pass
                await _send_json(ws, ws_lock, {"type": "clear"})
                await _send_json(ws, ws_lock, _question_state_payload(cursor.state))
                if cursor.state.get("phase") == "closing":
                    await _mark_closing(
                        cursor,
                        str(cursor.state.get("close_reason") or "complete"),
                        ws,
                        ws_lock,
                        call_id,
                        telemetry,
                        state,
                    )
                    telemetry["user_turns"] = int(telemetry.get("user_turns") or 0) + 1
                    _caller_heard(state)
                    cue_kind = "close"
                else:
                    _arm_user_turn(state, telemetry)
                    cue_kind = "speak"
                _cancel_speak_fallback(state)
                await _send_speak_cue(
                    session,
                    send_lock,
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    cursor,
                    state,
                    stop,
                    kind=cue_kind,
                    error_msg="Could not speak the next question",
                    record_tool_error=False,
                )
            elif kind == "hangup":
                stop.set()
                return
    except asyncio.CancelledError:
        raise


async def _report_is_ready(
    on_call_completed: Callable[[int], Awaitable[bool] | bool] | None,
    call_id: int,
) -> bool:
    if on_call_completed is None:
        return False
    try:
        result = on_call_completed(call_id)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:
        return False


def _closing_audio_finished(state: _TurnState, now: float) -> bool:
    return (
        bool(state.closing_heard)
        and not state.bot_speaking
        and state.playback_ends_at > 0
        and now >= state.playback_ends_at + _CLOSING_AUDIO_GRACE_SEC
    )


async def _notify_ending(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    state: _TurnState | None,
) -> None:
    """Tell the browser the outro is done so the report spinner can show."""
    if state is not None:
        if state.ending_sent:
            return
        state.ending_sent = True
    await _send_json(ws, ws_lock, {"type": "ending"})


async def _finalize(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict | None,
    recorder: audio_recorder.CallAudioRecorder,
    on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
    cursor: _Cursor | None = None,
    state: _TurnState | None = None,
) -> None:
    # Spinner first. PDF work happens after this, and a second send is skipped.
    await _notify_ending(ws, ws_lock, state)
    filename = ""
    duration = 0
    try:
        filename, duration = await asyncio.to_thread(recorder.save)
    except Exception:
        filename, duration = "", 0

    if telemetry is not None:
        _stamp_dropoff(telemetry, None if cursor is None else cursor.state)
        _refresh_latency(telemetry)
        try:
            await asyncio.to_thread(db.update_call_telemetry, call_id, _json_copy(telemetry))
        except Exception:
            pass

    await asyncio.to_thread(db.complete_call, call_id, int(duration), filename, "", "")
    ready = await _report_is_ready(on_call_completed, call_id)
    call = await asyncio.to_thread(db.get_call, call_id) or {}
    done = {
        "type": "call_completed" if ready else "report_failed",
        "call_id": call_id,
        "share_token": call.get("share_token") or "",
    }
    await _send_json(ws, ws_lock, done)


async def _stop_tasks(*tasks: asyncio.Task | None) -> None:
    pending = [task for task in tasks if task is not None and not task.done()]
    for task in pending:
        task.cancel()
    waiting = [task for task in tasks if task is not None]
    if waiting:
        await asyncio.gather(*waiting, return_exceptions=True)


async def _connect_live(client: genai.Client, model: str, config: types.LiveConnectConfig):
    started = time.perf_counter()
    ctx = client.aio.live.connect(model=model, config=config)
    session = await ctx.__aenter__()
    if getattr(session, "setup_complete", None) is None:
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            pass
        raise RuntimeError("Gemini Live setup did not complete")
    connect_ms = int((time.perf_counter() - started) * 1000)
    return ctx, session, connect_ms


async def handle_gemini_ws(
    ws: WebSocket,
    caller_name: str = "",
    user_role: str = "user",
    on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
    clinic_code: str = "",
    thresholds: decisions.DecisionThresholds | None = None,
) -> None:
    if ws.client_state == WebSocketState.CONNECTING:
        await ws.accept()

    caller_name = caller_name or ""
    user_role = user_role or "user"
    ws_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    stop = asyncio.Event()
    state = _TurnState()
    cursor = _Cursor()
    call_id: int | None = None
    telemetry: dict | None = None
    recorder: audio_recorder.CallAudioRecorder | None = None
    session_ctx = None
    gemini_task: asyncio.Task | None = None
    browser_task: asyncio.Task | None = None
    watchdog_task: asyncio.Task | None = None
    finalized = False

    try:
        settings = live_settings()
        try:
            state.capture.thresholds = load_turn_thresholds(
                settings,
                thresholds=thresholds,
            )
        except Exception:
            await _send_json(
                ws,
                ws_lock,
                {
                    "type": "error",
                    "message": "Invalid or missing Decision API thresholds",
                },
            )
            return

        if not os.environ.get("GEMINI_API_KEY"):
            await _send_json(
                ws,
                ws_lock,
                {"type": "error", "message": "GEMINI_API_KEY is not set"},
            )
            return

        chosen = _session_settings(settings)
        model = chosen["model"]
        voice = chosen["voice"]
        config = _live_config(settings)
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"api_version": "v1beta"},
        )
        try:
            session_ctx, session, connect_ms = await _connect_live(
                client, model, config
            )
        except Exception:
            await _send_json(
                ws,
                ws_lock,
                {"type": "error", "message": "Gemini Live connection failed"},
            )
            return
        session = _SessionProxy(session)

        async def _reconnect(handle: str) -> bool:
            nonlocal session_ctx
            new_ctx = None
            new_session = None
            for attempt in range(3):
                try:
                    new_ctx, new_session, _ = await _connect_live(
                        client,
                        model,
                        _live_config(settings, resume_handle=handle),
                    )
                    break
                except Exception:
                    if attempt == 2:
                        return False
                    await asyncio.sleep(0.4 * (attempt + 1))
            old_ctx = session_ctx
            async with send_lock:
                session.current = new_session
                session_ctx = new_ctx
            try:
                await old_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            return True

        call_id = db.create_call(
            caller_name=caller_name,
            user_role=user_role,
            model_used=model,
            voice_used=voice,
            clinic_code=clinic_code or "",
        )
        telemetry = _new_telemetry(
            model,
            voice,
            chosen["silence_duration_ms"],
            chosen["prefix_padding_ms"],
            connect_ms,
        )
        recorder = audio_recorder.CallAudioRecorder(call_id)
        await asyncio.to_thread(db.save_cursor_state, call_id, _json_copy(cursor.state))
        share_token = str(
            (await asyncio.to_thread(db.get_call, call_id) or {}).get("share_token") or ""
        )
        await _send_json(
            ws,
            ws_lock,
            {
                "type": "session_started",
                "call_id": call_id,
                "model": model,
                "voice": voice,
                "user_role": user_role,
                "caller_name": caller_name,
                "share_token": share_token,
                "out_sample_rate": _OUTPUT_RATE,
            },
        )
        await _emit(
            ws,
            ws_lock,
            call_id,
            telemetry,
            "session_started",
            model,
        )

        gemini_task = asyncio.create_task(
            _gemini_loop(
                session,
                send_lock,
                ws,
                ws_lock,
                stop,
                call_id,
                telemetry,
                state,
                recorder,
                cursor,
                _reconnect,
                settings=settings,
                thresholds=state.capture.thresholds,
            )
        )
        watchdog_task = asyncio.create_task(
            _watchdog_loop(
                session,
                send_lock,
                ws,
                ws_lock,
                stop,
                call_id,
                telemetry,
                state,
                cursor,
                chosen["watchdog_timeout_sec"],
            )
        )
        try:
            await _send_greeting(session, send_lock, ws, ws_lock, cursor, state=state)
        except Exception:
            stop.set()
            await _report_error(
                ws,
                ws_lock,
                call_id,
                telemetry,
                "Gemini Live greeting failed",
            )
            return

        browser_task = asyncio.create_task(
            _browser_loop(
                session,
                send_lock,
                ws,
                ws_lock,
                stop,
                call_id,
                telemetry,
                state,
                cursor,
                recorder,
            )
        )
        done, pending = await asyncio.wait(
            {gemini_task, browser_task, watchdog_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
    finally:
        stop.set()
        _cancel_speak_fallback(state)
        await _stop_tasks(gemini_task, browser_task, watchdog_task)
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if call_id is not None and recorder is not None and not finalized:
            finalized = True
            try:
                await _finalize(
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    recorder,
                    on_call_completed,
                    cursor,
                    state,
                )
            except Exception:
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "error", "message": "Could not finish the call"},
                )
        await _close_ws(ws)
