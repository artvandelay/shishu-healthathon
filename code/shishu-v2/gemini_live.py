"""Standalone Gemini Live pipeline for one SHISHU pre-consultation call (/ws/gemini)."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
from call_cursor import apply_turn, expect_yes_no, new_cursor_state, step_id
from live_decision_bridge import (
    TurnCapture,
    load_turn_thresholds,
    run_decision_turn,
    tool_body as _tool_body,
)
from prompts import GEMINI_LIVE_SYSTEM_PROMPT, HINDI_MEDICAL_VOCAB
from triage_catalog import get_spec

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_LIVE_SETTINGS: dict[str, Any] = {
    "gemini_model": "gemini-3.1-flash-live-preview",
    "gemini_voice": "Kore",
    "vad_prefix_padding_ms": 300,
    "vad_silence_ms": 1000,
    "vad_start_sensitivity": "LOW",
    "vad_end_sensitivity": "LOW",
    "watchdog_timeout_sec": 8.0,
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

_INPUT_RATE, _OUTPUT_RATE = 16000, 24000
_WATCHDOG_INTERVAL_SEC, _CHECK_IN_SEC, _SILENCE_CLOSE_SEC = 0.5, 15, 30
_SOFT_LIMIT_SEC, _HARD_LIMIT_SEC = 19 * 60, 20 * 60
_CLOSING_AUDIO_GRACE_SEC, _CLOSING_MAX_WAIT_SEC = 0.8, 25
_TOOL_SPEAK_FALLBACK_SEC = 2.5

_NEXT_QUESTION_DECL = {
    "name": "next_question",
    "description": (
        "Call this on every caller utterance with no arguments to fetch the exact Hindi line to speak next. "
        "Never speak to the caller before calling next_question."
    ),
    "parameters": {"type": "OBJECT", "properties": {}},
}

_TELEMETRY_EVENTS = frozenset({
    "session_started", "user_transcript", "bot_first_audio", "turn_complete", "tool_call",
    "interrupted", "watchdog_nudge", "manual_nudge", "reconnecting", "reconnected",
    "silence_check_in", "auto_close", "error", "end_accepted",
})


@dataclass
class _TurnState:
    awaiting_bot_reply: bool = False
    last_user_turn_ts: float = 0.0
    bot_replied_since_user: bool = False
    bot_speaking: bool = False
    nudges_for_current_turn: int = 0
    quiet_from: float = 0.0
    checked_in: bool = False
    call_started_ts: float = field(default_factory=time.time)
    resume_handle: str = ""
    reconnect_requested: bool = False
    capture: TurnCapture = field(default_factory=TurnCapture)
    time_cue_sent: bool = False
    speech_end_waiting: bool = False
    playback_ends_at: float = 0.0
    spoken_qid: str = ""
    speak_fallback: asyncio.Task | None = None
    ignore_bot_audio: bool = False
    user_turn_step: str = ""
    pending_speak_step: str = ""
    pending_speak_kind: str = "speak"
    last_quick_reply_step: str = ""
    last_quick_reply_ts: float = 0.0
    speak_cue_step: str = ""
    go_away_pending: bool = False
    active_speak_kind: str = "speak"
    end_requested: bool = False
    end_requested_at: float = 0.0
    closing_heard: bool = False
    ending_sent: bool = False


class _SessionProxy:
    def __init__(self, session) -> None:
        self.current = session

    def __getattr__(self, name: str):
        return getattr(self.current, name)


def _coerce_setting(key: str, value: Any) -> Any:
    default = DEFAULT_LIVE_SETTINGS.get(key)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return str(value).strip()


def live_settings(overrides: dict | None = None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(DEFAULT_LIVE_SETTINGS)
    for env_key, setting_key in _LIVE_SETTING_ENV_KEYS:
        if (raw := os.environ.get(env_key)) is not None and raw.strip():
            merged[setting_key] = _coerce_setting(setting_key, raw)
    if overrides:
        for key, value in overrides.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            merged[str(key)] = _coerce_setting(str(key), value)
    return merged


def _session_settings(settings: dict | None = None) -> dict[str, Any]:
    merged = live_settings(settings)
    start_low = str(merged["vad_start_sensitivity"]).upper() == "LOW"
    end_low = str(merged["vad_end_sensitivity"]).upper() == "LOW"
    return {
        "model": str(merged["gemini_model"]) or str(DEFAULT_LIVE_SETTINGS["gemini_model"]),
        "voice": str(merged["gemini_voice"]) or str(DEFAULT_LIVE_SETTINGS["gemini_voice"]),
        "prefix_padding_ms": int(merged["vad_prefix_padding_ms"]),
        "silence_duration_ms": int(merged["vad_silence_ms"]),
        "start_sens": (types.StartSensitivity.START_SENSITIVITY_LOW if start_low
                       else types.StartSensitivity.START_SENSITIVITY_HIGH),
        "end_sens": (types.EndSensitivity.END_SENSITIVITY_LOW if end_low
                     else types.EndSensitivity.END_SENSITIVITY_HIGH),
        "watchdog_timeout_sec": float(merged["watchdog_timeout_sec"]),
    }


def _live_config(settings: dict | None = None, resume_handle: str | None = None) -> types.LiveConnectConfig:
    chosen = _session_settings(settings)
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        temperature=0.5,
        enable_affective_dialog=False,
        speech_config=types.SpeechConfig(
            language_code="hi-IN",
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=chosen["voice"])),
        ),
        system_instruction=types.Content(parts=[types.Part(text=GEMINI_LIVE_SYSTEM_PROMPT)]),
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["hi-IN"], custom_vocabulary=list(HINDI_MEDICAL_VOCAB)),
        output_audio_transcription=types.AudioTranscriptionConfig(language_codes=["hi-IN"]),
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
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle or None),
        tools=[{"function_declarations": [_NEXT_QUESTION_DECL]}],
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )


def _new_telemetry(model: str, voice: str, silence_ms: int, prefix_ms: int, connect_ms: int) -> dict:
    return {
        "model": model, "voice": voice, "vad_silence_ms": silence_ms,
        "vad_prefix_padding_ms": prefix_ms, "connect_ms": connect_ms,
        "user_turns": 0, "bot_turns": 0, "tool_calls": 0,
        "audio_chunks_in": 0, "audio_chunks_out": 0, "interruptions": 0,
        "watchdog_nudges": 0, "turn_latencies_ms": [], "avg_latency_ms": 0,
        "max_latency_ms": 0, "last_interaction_status": "", "dropoff_qid": "",
        "reconnects": 0, "errors": [],
    }


def _stamp_dropoff(telemetry: dict, state: dict | None) -> None:
    cursor_state = state or {}
    reason = str(cursor_state.get("close_reason") or "")
    telemetry["dropoff_qid"] = "" if reason == "complete" else str(cursor_state.get("active_qid") or "")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _refresh_latency(telemetry: dict) -> None:
    lat = telemetry.get("turn_latencies_ms") or []
    telemetry["avg_latency_ms"] = int(round(sum(lat) / len(lat))) if lat else 0
    telemetry["max_latency_ms"] = int(max(lat)) if lat else 0


def _ws_connected(ws: WebSocket) -> bool:
    return ws.client_state == WebSocketState.CONNECTED


async def _send_json(ws: WebSocket, lock: asyncio.Lock, payload: dict) -> None:
    if not _ws_connected(ws):
        return
    try:
        async with lock:
            if _ws_connected(ws):
                await ws.send_json(payload)
    except Exception:
        return


async def _close_ws(ws: WebSocket) -> None:
    if _ws_connected(ws):
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
    return (pcm, audio_recorder.rate_from_mime(mime, _OUTPUT_RATE)) if pcm else None


def _question_state_payload(cursor_state: dict) -> dict:
    qid = str(cursor_state.get("active_qid") or "")
    try:
        group = get_spec(qid).group if qid else ""
    except KeyError:
        group = ""
    return {
        "type": "question_state",
        "step": step_id(cursor_state),
        "qid": qid,
        "group": group,
        "expect_yes_no": expect_yes_no(cursor_state),
        "prompt_chars": len(prompts.say_hindi(cursor_state)),
    }


def consume_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except Exception:
        return
    if exc is not None:
        logger.warning("background live task failed", exc_info=exc)


class LiveCallSession:
    """Encapsulates state, I/O loops, and decision bridge execution for one live call."""

    def __init__(
        self, *, ws: WebSocket, session: Any = None, call_id: int | None = None,
        telemetry: dict | None = None, recorder: Any = None, cursor_state: dict | None = None,
        settings: dict[str, Any] | None = None, thresholds: decisions.DecisionThresholds | None = None,
        stop: asyncio.Event | None = None, ws_lock: asyncio.Lock | None = None,
        send_lock: asyncio.Lock | None = None, reconnect: Callable[[str], Awaitable[bool]] | None = None,
        on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
        watchdog_timeout_sec: float | None = None,
    ) -> None:
        self.ws, self.ws_lock = ws, ws_lock or asyncio.Lock()
        self.session, self.send_lock = session, send_lock or asyncio.Lock()
        self.stop, self.call_id = stop or asyncio.Event(), call_id
        self.settings = settings if settings is not None else live_settings()
        chosen = _session_settings(self.settings)
        self.telemetry = telemetry if telemetry is not None else _new_telemetry(
            chosen["model"], chosen["voice"], chosen["silence_duration_ms"], chosen["prefix_padding_ms"], 0
        )
        self.recorder = recorder
        self.cursor_state: dict = cursor_state if cursor_state is not None else new_cursor_state()
        self.thresholds = thresholds
        self.turn = _TurnState()
        self.turn.capture.thresholds = thresholds
        self.seen: set[str] = set()
        self.cancelled: set[str] = set()
        self.reconnect, self.on_call_completed = reconnect, on_call_completed
        self.watchdog_timeout_sec = (
            float(watchdog_timeout_sec) if watchdog_timeout_sec is not None else float(chosen["watchdog_timeout_sec"])
        )

    @property
    def capture(self) -> TurnCapture:
        return self.turn.capture

    async def _send_json(self, payload: dict) -> None:
        await _send_json(self.ws, self.ws_lock, payload)

    async def _send_realtime_safe(self, **kwargs) -> bool:
        try:
            async with self.send_lock:
                await self.session.send_realtime_input(**kwargs)
            return True
        except Exception:
            if self.turn.resume_handle and not self.stop.is_set():
                self.turn.reconnect_requested = True
                return True
            return False

    async def _db_safe(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self.call_id is None:
            return None
        try:
            return await asyncio.to_thread(fn, self.call_id, *args, **kwargs)
        except Exception:
            return None

    async def _save_cursor_safe(self) -> None:
        await self._db_safe(db.save_cursor_state, _json_copy(self.cursor_state))

    async def _append_event_safe(self, event_type: str, detail: str = "", meta: dict[str, Any] | None = None) -> None:
        await self._db_safe(db.append_call_event, event_type, detail, meta)

    async def _emit_transcript(self, role: str, text: str) -> None:
        if self.call_id is not None:
            qid = str(self.cursor_state.get("active_qid") or "")
            try:
                await asyncio.to_thread(db.append_transcript, self.call_id, role, text, qid=qid)
            except Exception:
                logger.warning("append_transcript failed on call %s (%s)", self.call_id, role, exc_info=True)
        await self._send_json({"type": "transcript", "role": role, "text": text})

    async def _emit(self, event_type: str, detail: str = "", meta: dict | None = None) -> None:
        if event_type not in _TELEMETRY_EVENTS:
            return
        self.telemetry["last_interaction_status"] = event_type
        _refresh_latency(self.telemetry)
        evt = await self._db_safe(db.append_call_event, event_type, detail, meta)
        if evt is None:
            evt = {"t": round(time.time(), 3), "type": event_type, "detail": detail or "", "meta": meta or {}}
        snapshot = _json_copy(self.telemetry)
        await self._db_safe(db.update_call_telemetry, snapshot)
        await self._send_json({"type": "telemetry", "telemetry": snapshot, "last_event": evt})

    async def _report_error(self, message: str) -> None:
        if self.call_id is not None and self.telemetry is not None:
            errors = self.telemetry.setdefault("errors", [])
            errors.append(message)
            if len(errors) > 50:
                del errors[:-50]
            await self._emit("error", message)
        await self._send_json({"type": "error", "message": message})

    async def _forward_transcript(self, role: str, transcription: types.Transcription | None) -> str:
        if transcription is None or not (text := " ".join((transcription.text or "").split())):
            return ""
        await self._emit_transcript(role, text)
        return text

    def _caller_heard(self) -> None:
        self.turn.quiet_from = 0.0

    def _arm_waiting_reply(self, *, speech_end_waiting: bool = False, now_ts: float | None = None) -> None:
        self.turn.awaiting_bot_reply = True
        self.turn.speech_end_waiting = speech_end_waiting
        self.turn.last_user_turn_ts = now_ts if now_ts is not None else time.time()
        self.turn.bot_replied_since_user = False
        self.turn.nudges_for_current_turn = 0

    def _arm_user_turn(self) -> None:
        if not self.turn.awaiting_bot_reply or self.turn.speech_end_waiting:
            self.telemetry["user_turns"] = int(self.telemetry.get("user_turns") or 0) + 1
        self._arm_waiting_reply(speech_end_waiting=False)
        self._caller_heard()
        self.turn.checked_in = False
        self.turn.active_speak_kind = "speak"

    def _question_already_spoken(self) -> bool:
        step = step_id(self.cursor_state)
        return bool(step) and self.turn.spoken_qid == step

    def _in_quick_reply_stale_window(self) -> bool:
        t = self.turn
        return bool(t.last_quick_reply_step and (
            not self._question_already_spoken() or t.ignore_bot_audio or time.time() - t.last_quick_reply_ts < 1.2
        ))

    def _is_stale_user_turn(self) -> bool:
        cur = step_id(self.cursor_state)
        mismatch = self.turn.user_turn_step != cur if self.turn.user_turn_step else bool(self.turn.last_quick_reply_step)
        return bool(mismatch or self.turn.capture.user_turn_is_quick_reply_stale)

    def _cancel_speak_fallback(self) -> None:
        task = self.turn.speak_fallback
        self.turn.speak_fallback = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        task.remove_done_callback(consume_task_error)
        task.add_done_callback(consume_task_error)

    def _closing_audio_finished(self, now: float) -> bool:
        return (bool(self.turn.closing_heard) and not self.turn.bot_speaking
                and self.turn.playback_ends_at > 0 and now >= self.turn.playback_ends_at + _CLOSING_AUDIO_GRACE_SEC)

    async def _notify_ending(self) -> None:
        if not self.turn.ending_sent:
            self.turn.ending_sent = True
            await self._send_json({"type": "ending"})

    async def _mark_closing(self, reason: str) -> None:
        state = self.cursor_state
        if state.get("phase") != "closing":
            state["phase"], state["close_reason"] = "closing", reason
        elif not state.get("close_reason"):
            state["close_reason"] = reason
        _stamp_dropoff(self.telemetry, state)
        await self._save_cursor_safe()
        first = not self.turn.end_requested
        self.turn.end_requested, self.turn.end_requested_at = True, time.time()
        if first:
            close_reason = str(state.get("close_reason") or reason)
            await self._emit("end_accepted", close_reason, {"reason": close_reason})

    async def _send_greeting(self) -> None:
        self.turn.capture.speech_allowed = True
        async with self.send_lock:
            await self.session.send_realtime_input(text=prompts.GREETING_CUE + prompts.say_hindi(self.cursor_state))
        await self._send_json(_question_state_payload(self.cursor_state))

    async def _send_speak_cue(
        self, *, kind: str, text: str | None = None,
        error_msg: str = "Could not prompt Gemini to speak", record_tool_error: bool = True,
    ) -> bool:
        step_now = step_id(self.cursor_state)
        cue_text = text if text is not None else prompts.speak_cue_text(self.cursor_state, kind)
        self.turn.active_speak_kind, self.turn.pending_speak_step = kind, ""
        self.turn.capture.speech_allowed, self.turn.ignore_bot_audio = True, False
        if await self._send_realtime_safe(text=cue_text):
            if not self.turn.reconnect_requested:
                self.turn.speak_cue_step = step_now
            return True
        if record_tool_error:
            self.telemetry["tool_errors"] = int(self.telemetry.get("tool_errors") or 0) + 1
        await self._report_error(error_msg)
        return False

    async def _send_nudge(self, event_type: str, detail: str = "") -> bool:
        if event_type == "watchdog_nudge":
            self.turn.nudges_for_current_turn += 1
            self.telemetry["watchdog_nudges"] = int(self.telemetry.get("watchdog_nudges") or 0) + 1
            self.turn.last_user_turn_ts = time.time()
            err = "Watchdog nudge failed"
        else:
            if self.turn.awaiting_bot_reply:
                self.turn.last_user_turn_ts = time.time()
            err = "Could not nudge Gemini"
        cue = detail or prompts.speak_cue_text(self.cursor_state, self.turn.active_speak_kind)
        if not await self._send_speak_cue(kind=self.turn.active_speak_kind, text=cue,
                                          error_msg=err, record_tool_error=False):
            self.stop.set()
            return False
        if not self.turn.reconnect_requested:
            await self._emit(event_type, cue)
        return True

    async def _fallback_speak_after_tool(self, *, step_after_tool: str, kind: str, delay_sec: float) -> None:
        """Speak the tool's next line only when Gemini stayed silent after the tool result."""
        try:
            if delay_sec > 0:
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=delay_sec)
                    return
                except asyncio.TimeoutError:
                    pass
            if self.stop.is_set() or self.turn.bot_replied_since_user or self.turn.bot_speaking:
                return
            if step_id(self.cursor_state) != step_after_tool:
                return
            if not await self._send_speak_cue(kind=kind):
                self.stop.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.telemetry["tool_errors"] = int(self.telemetry.get("tool_errors") or 0) + 1
            await self._report_error("Could not prompt Gemini to speak")

    async def _emit_and_cue_after_decision(
        self, *, clarify: bool = False, delay_sec: float = _TOOL_SPEAK_FALLBACK_SEC, cue_now: bool = False
    ) -> bool:
        await self._send_json(_question_state_payload(self.cursor_state))
        kind = "close" if self.cursor_state.get("phase") != "asking" else ("clarify" if clarify else "speak")
        step_after = step_id(self.cursor_state)
        self._cancel_speak_fallback()
        if cue_now:
            if not self.turn.end_requested:
                self._arm_waiting_reply(speech_end_waiting=False)
                self.turn.quiet_from = 0.0
            if not await self._send_speak_cue(kind=kind):
                self.stop.set()
                return False
            return True
        self.turn.pending_speak_step, self.turn.pending_speak_kind, self.turn.active_speak_kind = step_after, kind, kind
        if delay_sec <= 0:
            await self._fallback_speak_after_tool(step_after_tool=step_after, kind=kind, delay_sec=0)
        else:
            task = asyncio.create_task(self._fallback_speak_after_tool(step_after_tool=step_after, kind=kind, delay_sec=delay_sec))
            task.add_done_callback(consume_task_error)
            self.turn.speak_fallback = task
        return True

    async def _answer_function_calls(self, calls: list) -> tuple[bool, bool]:
        cap = self.turn.capture
        responses: list[types.FunctionResponse] = []
        logged: list[str] = []
        clarify_applied = handled = decision_ran = False
        stale_only = True
        for fc in calls:
            fc_id = getattr(fc, "id", None) or ""
            if (fc_id and (fc_id in self.cancelled or fc_id in self.seen)) or getattr(fc, "will_continue", False):
                continue
            if fc_id:
                self.seen.add(fc_id)
            name = getattr(fc, "name", None) or ""
            try:
                if name == "next_question":
                    if self.cursor_state.get("phase") != "asking":
                        cap.speech_allowed = True
                        body = _tool_body(self.cursor_state)
                        logged.append("next_question_closing")
                        handled, stale_only = True, False
                    else:
                        hold_reason = (
                            "duplicate_turn" if (cap.handled_turn and not self._in_quick_reply_stale_window())
                            else "stale_turn" if self._is_stale_user_turn()
                            else "not_spoken" if not self._question_already_spoken()
                            else ""
                        )
                        if hold_reason:
                            body = {**_tool_body(self.cursor_state), "result": "held", "reason": hold_reason}
                            logged.append("next_question_held")
                            if hold_reason != "stale_turn":
                                cap.speech_allowed = True
                            handled = True
                        else:
                            body, call_clarify, event_kind = await run_decision_turn(self, input_rate=_INPUT_RATE)
                            decision_ran, clarify_applied = True, clarify_applied or call_clarify
                            if event_kind == "held":
                                logged.append("next_question_held")
                            else:
                                stale_only = False
                                logged.append(f"next_question_{event_kind}")
                            handled = True
                else:
                    body = {"result": "ignored", "say": prompts.say_hindi(self.cursor_state)}
                    logged.append(name or "ignored")
                    self.telemetry["tool_errors"] = int(self.telemetry.get("tool_errors") or 0) + 1
            except Exception:
                logger.exception("tool %s failed on call %s", name, self.call_id)
                self.telemetry["tool_errors"] = int(self.telemetry.get("tool_errors") or 0) + 1
                body = {"result": "error", "say": prompts.say_hindi(self.cursor_state)}
                logged.append(f"{name or 'tool'}_error")
                clarify_applied, stale_only, handled = False, False, True
            if self.cursor_state.get("phase") == "closing":
                body["stop"], body["say"] = True, prompts.closing_hi(self.cursor_state)
            if fc_id:
                responses.append(types.FunctionResponse(id=fc_id, name=name or "next_question", response=body))

        send_failed = False
        if responses:
            try:
                async with self.send_lock:
                    await self.session.send_tool_response(function_responses=responses)
            except Exception:
                send_failed = True
        for detail in logged:
            self.telemetry["tool_calls"] = int(self.telemetry.get("tool_calls") or 0) + 1
            await self._emit("tool_call", detail)
        if send_failed:
            self.telemetry["tool_errors"] = int(self.telemetry.get("tool_errors") or 0) + 1
            await self._report_error("Could not send the tool response to Gemini")
        if handled:
            step_after = step_id(self.cursor_state)
            cue_sent = bool(step_after) and self.turn.speak_cue_step == step_after
            if not (stale_only and (cue_sent or self._question_already_spoken())):
                await self._emit_and_cue_after_decision(
                    clarify=clarify_applied, delay_sec=(0.0 if send_failed else _TOOL_SPEAK_FALLBACK_SEC)
                )
            if stale_only and cap.user_turn_is_quick_reply_stale:
                cap.reset()
        return decision_ran, stale_only

    async def _on_bot_audio(self, pcm: bytes, rate: int) -> None:
        turn = self.turn
        if turn.ignore_bot_audio or not turn.capture.speech_allowed:
            return
        turn.pending_speak_step = ""
        self.telemetry["audio_chunks_out"] += 1
        first = not turn.bot_replied_since_user
        latency_ms = None
        if first:
            self.telemetry["bot_turns"] += 1
            if turn.awaiting_bot_reply and turn.last_user_turn_ts > 0:
                latency_ms = max(0, int((time.time() - turn.last_user_turn_ts) * 1000))
                self.telemetry["turn_latencies_ms"].append(latency_ms)
            turn.nudges_for_current_turn = 0
            turn.spoken_qid = step_id(self.cursor_state)
        turn.awaiting_bot_reply, turn.speech_end_waiting = False, False
        turn.bot_replied_since_user, turn.bot_speaking = True, True
        turn.playback_ends_at = max(time.time(), turn.playback_ends_at) + len(pcm) / max(1, 2 * rate)
        if self.recorder is not None:
            self.recorder.add_bot_pcm(pcm, rate)
        if turn.end_requested:
            turn.closing_heard = self.telemetry["closing_heard"] = True
        await self._send_json({"type": "audio", "data": base64.b64encode(pcm).decode("ascii"), "sample_rate": _OUTPUT_RATE})
        if first:
            meta: dict[str, Any] = {"spoken_qid": turn.spoken_qid}
            if latency_ms is not None:
                meta["latency_ms"] = latency_ms
            await self._emit("bot_first_audio", f"{latency_ms}ms" if latency_ms is not None else "", meta)

    async def _handle_server_message(self, response) -> bool:
        """Handle one Live server message. Return False when the session should stop."""
        turn, cap, decision_ran_in_msg = self.turn, self.turn.capture, False
        if (canc := getattr(response, "tool_call_cancellation", None)) is not None and canc.ids:
            self.cancelled.update(item for item in canc.ids if item)

        sc = getattr(response, "server_content", None)
        calls: list[Any] = []
        if sc is not None:
            if sc.interrupted:
                turn.ignore_bot_audio, turn.bot_speaking, turn.playback_ends_at = False, False, 0.0
                cap.handled_turn, cap.last_key = False, None
                self._caller_heard()
                self.telemetry["interruptions"] += 1
                await self._send_json({"type": "clear"})
                await self._emit("interrupted", "interrupted")
            input_tr = getattr(sc, "input_transcription", None)
            user_text = await self._forward_transcript("user", input_tr) if not turn.end_requested else ""
            if user_text:
                if self._in_quick_reply_stale_window():
                    if cap.user_turn_finalized:
                        cap.reset(clear_pcm=False)
                    turn.user_turn_step = turn.last_quick_reply_step
                    cap.user_turn_is_quick_reply_stale = True
                    self._caller_heard()
                else:
                    cap.begin_transcript_chunk(self.cursor_state, user_text)
                    turn.user_turn_step, turn.last_quick_reply_step = step_id(self.cursor_state), ""
                    self._arm_user_turn()
                await self._emit("user_transcript", user_text)
            if bool(getattr(input_tr, "finished", False)) and not turn.end_requested:
                cap.user_turn_finalized = True
            if not (turn.ignore_bot_audio or not cap.speech_allowed):
                await self._forward_transcript("assistant", sc.output_transcription)
            if sc.model_turn is not None and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if getattr(part, "thought", False):
                        continue
                    if (audio := _audio_from_part(part)) is not None:
                        await self._on_bot_audio(*audio)
                    if (fc := getattr(part, "function_call", None)) is not None:
                        calls.append(fc)

        if (tool_call := getattr(response, "tool_call", None)) is not None and tool_call.function_calls:
            calls.extend(tool_call.function_calls)

        if calls:
            decision_ran_in_msg, stale_only = await self._answer_function_calls(calls)
            step_now = step_id(self.cursor_state)
            cue_handled = bool(step_now) and (turn.speak_cue_step == step_now or self._question_already_spoken())
            if not turn.end_requested and not (stale_only and (turn.bot_replied_since_user or cue_handled)):
                self._arm_waiting_reply(speech_end_waiting=False)
                turn.quiet_from = 0.0

        if sc is not None and sc.turn_complete:
            has_input = bool(cap.user_turn_buffer.strip()) or bool(cap.user_turn_pcm)
            if (not turn.end_requested and self.cursor_state.get("phase") == "asking"
                    and not self._is_stale_user_turn() and not cap.handled_turn
                    and self._question_already_spoken() and has_input):
                _body, clarify_applied, kind = await run_decision_turn(self, input_rate=_INPUT_RATE)
                decision_ran_in_msg = True
                self.telemetry["tool_calls"] = int(self.telemetry.get("tool_calls") or 0) + 1
                await self._emit("tool_call", f"missed_tool_{kind}")
                if not await self._emit_and_cue_after_decision(clarify=clarify_applied, cue_now=True):
                    return False
            if not decision_ran_in_msg and (
                cap.handled_turn or cap.user_turn_is_quick_reply_stale
                or (turn.user_turn_step and turn.user_turn_step != step_id(self.cursor_state))
            ):
                cap.reset()
            cap.handled_turn, cap.last_key, turn.ignore_bot_audio = False, None, False
            if turn.go_away_pending:
                turn.go_away_pending, turn.reconnect_requested = False, True
                self._cancel_speak_fallback()
            speak_step, speak_kind = turn.pending_speak_step, turn.pending_speak_kind
            if (speak_step and speak_step == step_id(self.cursor_state) and not turn.bot_replied_since_user
                    and not self.stop.is_set() and not turn.reconnect_requested):
                self._cancel_speak_fallback()
                if not await self._send_speak_cue(kind=speak_kind):
                    self.stop.set()
                    return False
            turn.bot_speaking = False
            if not turn.awaiting_bot_reply:
                turn.quiet_from = max(time.time(), turn.playback_ends_at)
                await self._send_json({"type": "bot_turn_complete", "step": step_id(self.cursor_state)})
            if not turn.awaiting_bot_reply or turn.end_requested:
                await self._emit("turn_complete", "")

        update = getattr(response, "session_resumption_update", None)
        if update is not None and getattr(update, "resumable", False) and getattr(update, "new_handle", None):
            turn.resume_handle = str(update.new_handle)
        if getattr(response, "go_away", None) is not None:
            if not turn.resume_handle:
                return False
            if turn.bot_speaking or turn.awaiting_bot_reply:
                turn.go_away_pending = True
            else:
                turn.reconnect_requested = True
        return True

    async def _gemini_loop(self) -> None:
        reconnects = 0
        while not self.stop.is_set():
            try:
                async for response in self.session.receive():
                    if self.stop.is_set():
                        return
                    if not await self._handle_server_message(response):
                        self.stop.set()
                        return
                    if self.turn.reconnect_requested:
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.stop.is_set() or not self.turn.resume_handle:
                    self.stop.set()
                    await self._report_error("Gemini Live session ended")
                    return
                self.turn.reconnect_requested = True
            if not self.turn.reconnect_requested:
                continue
            self.turn.reconnect_requested = False
            self._cancel_speak_fallback()
            if reconnects >= 5:
                self.stop.set()
                await self._report_error("Gemini Live could not reconnect")
                return
            reconnects += 1
            await self._send_json({"type": "reconnecting", "attempt": reconnects})
            await self._emit("reconnecting", str(reconnects))
            if self.reconnect is None or not await self.reconnect(self.turn.resume_handle):
                self.stop.set()
                await self._report_error("Gemini Live could not reconnect")
                return
            resume_speak = self.turn.bot_speaking or self.turn.awaiting_bot_reply or bool(self.turn.pending_speak_step)
            self.telemetry["reconnects"] = reconnects
            self.turn.bot_speaking = self.turn.ignore_bot_audio = False
            self.turn.awaiting_bot_reply = self.turn.speech_end_waiting = False
            self.turn.playback_ends_at, self.turn.quiet_from = 0.0, time.time()
            self.turn.capture.reset()
            self.seen.clear()
            self.cancelled.clear()
            await self._send_json({"type": "reconnected", "attempt": reconnects})
            await self._emit("reconnected", str(reconnects))
            if resume_speak and self.cursor_state.get("phase") == "asking" and not self.turn.end_requested:
                self._arm_waiting_reply(speech_end_waiting=False)
                await self._send_json(_question_state_payload(self.cursor_state))
                if not await self._send_speak_cue(kind=self.turn.active_speak_kind, record_tool_error=False):
                    self.stop.set()
                    return

    async def _watchdog_loop(self) -> None:
        turn = self.turn
        try:
            while not self.stop.is_set():
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=_WATCHDOG_INTERVAL_SEC)
                    return
                except asyncio.TimeoutError:
                    pass
                if self.stop.is_set():
                    return
                now = time.time()
                if now - turn.call_started_ts >= _HARD_LIMIT_SEC:
                    if self.cursor_state.get("phase") != "closing":
                        await self._mark_closing("time_limit")
                    self.stop.set()
                    return
                if self.cursor_state.get("phase") == "closing" or turn.end_requested:
                    if self._closing_audio_finished(now) or (now - float(turn.end_requested_at or now)) > _CLOSING_MAX_WAIT_SEC:
                        await self._notify_ending()
                        self.stop.set()
                        return
                    continue
                cue = reason = ""
                if now - turn.call_started_ts >= _SOFT_LIMIT_SEC and not turn.time_cue_sent:
                    reason, turn.time_cue_sent = "time_limit", True
                elif turn.quiet_from > 0 and not turn.bot_speaking and not turn.awaiting_bot_reply:
                    quiet = now - turn.quiet_from
                    if quiet >= _SILENCE_CLOSE_SEC:
                        reason = "silence"
                    elif quiet >= _CHECK_IN_SEC and not turn.checked_in:
                        cue = prompts.SILENCE_CHECK_IN_CUE
                if reason:
                    await self._mark_closing(reason)
                    cue = prompts.speak_cue_text(self.cursor_state, "close")
                if cue:
                    if not reason:
                        turn.checked_in = True
                    turn.capture.speech_allowed = True
                    if not await self._send_realtime_safe(text=cue):
                        self.stop.set()
                        return
                    if not turn.reconnect_requested:
                        await self._emit("auto_close" if reason else "silence_check_in", reason or "check_in")
                    continue
                if not turn.awaiting_bot_reply or turn.bot_speaking:
                    continue
                if turn.speech_end_waiting and self._question_already_spoken():
                    turn.awaiting_bot_reply = turn.speech_end_waiting = False
                    continue
                if turn.bot_replied_since_user or turn.last_user_turn_ts <= 0:
                    continue
                eff = min(self.watchdog_timeout_sec, 6.0) if turn.speech_end_waiting else self.watchdog_timeout_sec
                if turn.nudges_for_current_turn >= 6 or time.time() - turn.last_user_turn_ts < eff:
                    continue
                if not await self._send_nudge("watchdog_nudge"):
                    return
        except asyncio.CancelledError:
            raise

    async def _on_browser_audio(self, payload: dict) -> bool:
        if not (pcm := audio_recorder.pcm_bytes(payload.get("data") or "")):
            return True
        try:
            rate = int(payload.get("sample_rate") or _INPUT_RATE)
        except (TypeError, ValueError):
            rate = _INPUT_RATE
        if rate <= 0:
            rate = _INPUT_RATE
        self.telemetry["audio_chunks_in"] += 1
        if self.recorder is not None:
            self.recorder.add_user_pcm(pcm, rate)
        pcm_16k = pcm if rate == _INPUT_RATE else audio_recorder.resample_int16_mono(pcm, rate, _INPUT_RATE)
        turn, cap, now = self.turn, self.turn.capture, time.time()
        current_step = step_id(self.cursor_state)
        if (not turn.bot_speaking and now >= turn.playback_ends_at
                and self._question_already_spoken() and not self._in_quick_reply_stale_window()):
            if audio_recorder.pcm_rms(pcm) >= 0.015:
                self._caller_heard()
                if not cap.in_flight:
                    new_step = turn.user_turn_step != current_step
                    fresh_audio = not cap.user_turn_pcm and not cap.user_turn_buffer
                    if new_step:
                        cap.handled_turn, cap.last_key = False, None
                        turn.user_turn_step, turn.last_quick_reply_step = current_step, ""
                    if new_step or fresh_audio:
                        self._arm_user_turn()
                    cap.begin_audio_speech(self.cursor_state, pcm_16k)
            elif not cap.in_flight and cap.user_turn_pcm_tail > 0:
                cap.user_turn_pcm_tail -= 1
                cap.append_pcm(pcm_16k)
        if not await self._send_realtime_safe(audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={rate}")):
            self.stop.set()
            await self._report_error("Could not forward audio to Gemini")
            return False
        return True

    async def _on_browser_speech_end(self) -> None:
        now_ts, already = time.time(), self._question_already_spoken()
        playback_done = not self.turn.bot_speaking and now_ts >= self.turn.playback_ends_at - 0.1
        if playback_done and not already and not self.turn.awaiting_bot_reply:
            self._arm_waiting_reply(speech_end_waiting=True, now_ts=now_ts)
        if playback_done:
            self.turn.quiet_from = max(now_ts, self.turn.playback_ends_at)
        await self._append_event_safe("speech_end", "already_spoken" if already else "")

    async def _on_browser_text(self, payload: dict) -> bool:
        if not (text := str(payload.get("text") or "").strip()):
            return True
        await self._emit_transcript("user", text)
        self.turn.capture.begin_text_turn(self.cursor_state, text)
        self.turn.user_turn_step, self.turn.last_quick_reply_step = step_id(self.cursor_state), ""
        self._arm_user_turn()
        if not await self._send_realtime_safe(text=text):
            self.stop.set()
            await self._report_error("Could not forward text to Gemini")
            return False
        await self._emit("user_transcript", text)
        return True

    async def _on_browser_quick_reply(self, payload: dict) -> bool:
        turn = self.turn
        outcome = str(payload.get("outcome") or "").strip().lower()
        if (turn.end_requested or self.cursor_state.get("phase") != "asking"
                or outcome not in ("clear", "concern") or not expect_yes_no(self.cursor_state)
                or payload.get("step") != step_id(self.cursor_state) or not self._question_already_spoken()):
            return True
        if turn.bot_speaking:
            turn.ignore_bot_audio, turn.bot_speaking, turn.playback_ends_at = True, False, 0.0
        turn.user_turn_step = turn.pending_speak_step = ""
        turn.capture.reset()
        turn.last_quick_reply_step, turn.last_quick_reply_ts = str(payload.get("step") or ""), time.time()
        said = "नहीं।" if outcome == "clear" else "हाँ।"
        await self._emit_transcript("user", said)
        await self._emit("user_transcript", said)
        self.telemetry["tool_calls"] = int(self.telemetry.get("tool_calls") or 0) + 1
        await self._emit("tool_call", "quick_reply_clear" if outcome == "clear" else "quick_reply_concern")
        self.cursor_state = apply_turn(self.cursor_state, outcome)
        if self.cursor_state.get("phase") != "closing":
            await self._save_cursor_safe()
        await self._send_json({"type": "clear"})
        await self._send_json(_question_state_payload(self.cursor_state))
        if self.cursor_state.get("phase") == "closing":
            await self._mark_closing(str(self.cursor_state.get("close_reason") or "complete"))
            self.telemetry["user_turns"] = int(self.telemetry.get("user_turns") or 0) + 1
            self._arm_waiting_reply(speech_end_waiting=False)
            self._caller_heard()
            cue_kind = "close"
        else:
            self._arm_user_turn()
            cue_kind = "speak"
        self._cancel_speak_fallback()
        if not await self._send_speak_cue(kind=cue_kind, error_msg="Could not speak the next question", record_tool_error=False):
            self.stop.set()
            return False
        return True

    async def _browser_loop(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    raw = await self.ws.receive_text()
                except (WebSocketDisconnect, RuntimeError):
                    self.stop.set()
                    return
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await self._report_error("Invalid JSON")
                    continue
                if not isinstance(payload, dict):
                    await self._report_error("Expected a JSON object")
                    continue
                kind = payload.get("type")
                if self.turn.end_requested and kind in {"audio", "text", "speech_end", "nudge"}:
                    continue
                if kind == "audio" and not await self._on_browser_audio(payload):
                    return
                elif kind == "speech_end":
                    await self._on_browser_speech_end()
                elif kind == "nudge":
                    if not await self._send_nudge("manual_nudge"):
                        return
                elif kind == "text" and not await self._on_browser_text(payload):
                    return
                elif kind == "quick_reply" and not await self._on_browser_quick_reply(payload):
                    return
                elif kind == "hangup":
                    self.stop.set()
                    return
        except asyncio.CancelledError:
            raise

    async def _finalize(self) -> None:
        await self._notify_ending()
        filename, duration = "", 0
        if self.recorder is not None:
            try:
                filename, duration = await asyncio.to_thread(self.recorder.save)
            except Exception:
                filename, duration = "", 0
        if self.telemetry is not None:
            _stamp_dropoff(self.telemetry, self.cursor_state)
            _refresh_latency(self.telemetry)
            await self._db_safe(db.update_call_telemetry, _json_copy(self.telemetry))
        if self.call_id is None:
            return
        await asyncio.to_thread(db.complete_call, self.call_id, int(duration), filename)
        ready = False
        if self.on_call_completed is not None:
            try:
                res = self.on_call_completed(self.call_id)
                ready = bool(await res if inspect.isawaitable(res) else res)
            except Exception:
                ready = False
        call = await asyncio.to_thread(db.get_call, self.call_id) or {}
        await self._send_json({
            "type": "call_completed" if ready else "report_failed",
            "call_id": self.call_id,
            "share_token": call.get("share_token") or "",
        })

    async def run(self) -> None:
        tasks: list[asyncio.Task] = []
        try:
            tasks.extend((
                asyncio.create_task(self._gemini_loop()),
                asyncio.create_task(self._watchdog_loop()),
            ))
            try:
                await self._send_greeting()
            except Exception:
                self.stop.set()
                await self._report_error("Gemini Live greeting failed")
                return
            tasks.append(asyncio.create_task(self._browser_loop()))
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.stop.set()
            self._cancel_speak_fallback()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                for res in await asyncio.gather(*tasks, return_exceptions=True):
                    if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                        logger.warning("live call task failed on call %s", self.call_id, exc_info=res)


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
    return ctx, session, int((time.perf_counter() - started) * 1000)


async def handle_gemini_ws(
    ws: WebSocket,
    caller_name: str = "",
    user_role: str = "user",
    on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
    clinic_code: str = "",
    thresholds: decisions.DecisionThresholds | None = None,
    settings_override: dict | None = None,
) -> None:
    if ws.client_state == WebSocketState.CONNECTING:
        await ws.accept()
    caller_name, user_role = caller_name or "", user_role or "user"
    ws_lock, send_lock = asyncio.Lock(), asyncio.Lock()
    live: LiveCallSession | None = None
    client = session_ctx = None

    try:
        try:
            settings = live_settings(settings_override)
            chosen = _session_settings(settings)
            config = _live_config(settings)
        except Exception:
            await _send_json(ws, ws_lock, {"type": "error", "message": "Invalid live call settings"})
            return
        try:
            loaded_thresholds = load_turn_thresholds(settings, thresholds=thresholds)
        except Exception:
            await _send_json(ws, ws_lock, {"type": "error", "message": "Invalid or missing Decision API thresholds"})
            return
        if not os.environ.get("GEMINI_API_KEY"):
            await _send_json(ws, ws_lock, {"type": "error", "message": "GEMINI_API_KEY is not set"})
            return

        model, voice = chosen["model"], chosen["voice"]
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1beta"})
        try:
            session_ctx, raw_session, connect_ms = await _connect_live(client, model, config)
        except Exception:
            await _send_json(ws, ws_lock, {"type": "error", "message": "Gemini Live connection failed"})
            return
        session = _SessionProxy(raw_session)

        async def _reconnect(handle: str) -> bool:
            nonlocal session_ctx
            new_ctx = new_session = None
            for attempt in range(3):
                try:
                    new_ctx, new_session, _ = await _connect_live(client, model, _live_config(settings, resume_handle=handle))
                    break
                except Exception:
                    if attempt == 2:
                        return False
                    await asyncio.sleep(0.4 * (attempt + 1))
            old_ctx = session_ctx
            async with send_lock:
                session.current, session_ctx = new_session, new_ctx
                if live is not None:
                    live.turn.reconnect_requested = False
            try:
                await old_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            return True

        call_id = await asyncio.to_thread(
            db.create_call, caller_name=caller_name, user_role=user_role,
            model_used=model, voice_used=voice, clinic_code=clinic_code or "",
        )
        telemetry = _new_telemetry(model, voice, chosen["silence_duration_ms"], chosen["prefix_padding_ms"], connect_ms)
        live = LiveCallSession(
            ws=ws, ws_lock=ws_lock, session=session, send_lock=send_lock, call_id=call_id,
            telemetry=telemetry, recorder=audio_recorder.CallAudioRecorder(call_id),
            settings=settings, thresholds=loaded_thresholds, reconnect=_reconnect,
            on_call_completed=on_call_completed, watchdog_timeout_sec=chosen["watchdog_timeout_sec"],
        )
        await live._save_cursor_safe()
        share_token = str((await asyncio.to_thread(db.get_call, call_id) or {}).get("share_token") or "")
        await live._send_json({
            "type": "session_started", "call_id": call_id, "model": model, "voice": voice,
            "user_role": user_role, "caller_name": caller_name, "share_token": share_token,
            "out_sample_rate": _OUTPUT_RATE,
        })
        await live._emit("session_started", model)
        await live.run()
    finally:
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if client is not None and callable(aclose := getattr(getattr(client, "aio", None), "aclose", None)):
            try:
                if inspect.isawaitable(res := aclose()):
                    await res
            except Exception:
                pass
        if live is not None:
            try:
                await live._finalize()
            except Exception:
                await _send_json(ws, ws_lock, {"type": "error", "message": "Could not finish the call"})
        await _close_ws(ws)
