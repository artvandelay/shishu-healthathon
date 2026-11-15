"""Gemini Live I/O for one SHISHU call: the only place voice meets the engine.

The live model speaks the Hindi this module hands it and never classifies an answer. Every
caller turn is classified by the Decision API, reduced by `engine`, and stored as one typed
`CallState`. `LiveTurnMachine` owns all turn-taking state, so this module is pure I/O.
"""

from __future__ import annotations

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
import copy_hi
import db
import decisions
import engine
import profile_extractor
from live_fsm import LiveTurnMachine, SpeakKind, TurnPhase, WatchdogAction
from models import (
    DEFAULT_LIVE_MODEL, DEFAULT_LIVE_VOICE, AnswerStatus, CallState, CloseReason, Phase,
    SideAction, TurnOutcome,
)
from triage_catalog import spec_or_none

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_LIVE_SETTINGS: dict[str, Any] = {
    "gemini_model": DEFAULT_LIVE_MODEL, "gemini_voice": DEFAULT_LIVE_VOICE,
    "vad_prefix_padding_ms": 300, "vad_silence_ms": 1000,
    "vad_start_sensitivity": "LOW", "vad_end_sensitivity": "LOW", "watchdog_timeout_sec": 8.0,
}
_LIVE_SETTING_ENV_KEYS = (
    ("GEMINI_MODEL", "gemini_model"), ("GEMINI_VOICE", "gemini_voice"),
    ("VAD_PREFIX_PADDING_MS", "vad_prefix_padding_ms"), ("VAD_SILENCE_MS", "vad_silence_ms"),
    ("VAD_START_SENSITIVITY", "vad_start_sensitivity"), ("VAD_END_SENSITIVITY", "vad_end_sensitivity"),
    ("WATCHDOG_TIMEOUT_SEC", "watchdog_timeout_sec"),
    ("DECISION_THRESHOLDS_PATH", "decision_thresholds_path"),
)
_INPUT_RATE, _OUTPUT_RATE = 16000, 24000
_WATCHDOG_INTERVAL_SEC, _TOOL_SPEAK_FALLBACK_SEC = 0.5, 2.5
_CLOSING_AUDIO_GRACE_SEC, _MAX_RECONNECTS = 0.8, 5
_NEXT_QUESTION_DECL = {
    "name": "next_question",
    "description": (
        "Call this on every caller utterance with no arguments to fetch the exact Hindi line to "
        "speak next. Never speak to the caller before calling next_question."
    ),
    "parameters": {"type": "OBJECT", "properties": {}},
}
_TELEMETRY_EVENTS = frozenset({
    "session_started", "user_transcript", "bot_first_audio", "turn_complete", "tool_call",
    "interrupted", "watchdog_nudge", "manual_nudge", "reconnecting", "reconnected",
    "silence_check_in", "auto_close", "error", "end_accepted",
})
# action -> (close reason or None for a check-in, event detail, hard-coded cue text)
_CLOSE_CUES = {
    WatchdogAction.SILENCE_CHECK_IN: (None, "check_in", copy_hi.SILENCE_CHECK_IN_CUE),
    WatchdogAction.AUTO_CLOSE_SILENCE: (CloseReason.SILENCE, "silence", None),
    WatchdogAction.AUTO_CLOSE_TIME: (CloseReason.TIME_LIMIT, "time_limit", None),
}


def tool_body(state: CallState, *, clarify: bool = False) -> dict[str, Any]:
    """The `next_question` response: one Hindi line plus the flags it needs."""
    body: dict[str, Any] = {"result": "ok", "say": engine.say_hindi(state)}
    if state.phase is Phase.CLOSING:
        body["stop"] = True
    spec = spec_or_none(state.cursor.qid)
    if clarify or (spec is not None and spec.shares_recorded_concerns):
        body["recorded_concerns"] = copy_hi.recorded_concerns_hi(state)
    if clarify:
        body["clarify"] = True
    return body


def _question_state_payload(state: CallState) -> dict[str, Any]:
    """Everything the browser needs to mount the हाँ / नहीं buttons for this step."""
    spec = spec_or_none(state.cursor.qid)
    return {
        "type": "question_state", "step": state.step_token, "qid": state.cursor.qid,
        "group": spec.group if spec is not None else "",
        "expect_yes_no": engine.expect_yes_no(state),
        "prompt_chars": len(engine.say_hindi(state)),
    }


def _decision_meta(
    res: decisions.DecisionCallResult | None = None, *, error_reason: str = ""
) -> dict[str, Any]:
    """Decision-call telemetry, honest about a call that never returned."""
    if res is not None:
        return {k: v for k, v in res._asdict().items() if k != "raw_answers"}
    return {
        "ok": False, "error_reason": error_reason, "usage_present": False,
        "total_input_tokens": None, "decision_ms": 0, "attempts": 0, "request_bytes": 0,
        "question_count": 0, "catalog_option_count": 0,
    }


def _coerce(key: str, value: Any) -> Any:
    default = DEFAULT_LIVE_SETTINGS.get(key)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    return float(value) if isinstance(default, float) else str(value).strip()


def live_settings(overrides: dict | None = None) -> dict[str, Any]:
    """Env first, then explicit overrides, each coerced to its default's own type."""
    merged: dict[str, Any] = dict(DEFAULT_LIVE_SETTINGS)
    for env_key, key in _LIVE_SETTING_ENV_KEYS:
        raw = os.environ.get(env_key)
        if raw is not None and raw.strip():
            merged[key] = _coerce(key, raw)
    for key, value in (overrides or {}).items():
        if value is not None and not (isinstance(value, str) and not value.strip()):
            merged[str(key)] = _coerce(str(key), value)
    return merged


def _session_settings(settings: dict | None = None) -> dict[str, Any]:
    """The resolved model, voice, VAD numbers, and watchdog timeout for one call."""
    merged = live_settings(settings)
    start, end = types.StartSensitivity, types.EndSensitivity
    loud_s = str(merged["vad_start_sensitivity"]).upper() != "LOW"
    loud_e = str(merged["vad_end_sensitivity"]).upper() != "LOW"
    return {
        "model": str(merged["gemini_model"]) or DEFAULT_LIVE_MODEL,
        "voice": str(merged["gemini_voice"]) or DEFAULT_LIVE_VOICE,
        "prefix_padding_ms": int(merged["vad_prefix_padding_ms"]),
        "silence_duration_ms": int(merged["vad_silence_ms"]),
        "start_sens": start.START_SENSITIVITY_HIGH if loud_s else start.START_SENSITIVITY_LOW,
        "end_sens": end.END_SENSITIVITY_HIGH if loud_e else end.END_SENSITIVITY_LOW,
        "watchdog_timeout_sec": float(merged["watchdog_timeout_sec"]),
    }


def _live_config(settings: dict | None = None, resume_handle: str | None = None) -> types.LiveConnectConfig:
    """The one config: Kore Hindi voice, low-sensitivity VAD, `next_question` only."""
    chosen = _session_settings(settings)
    voice = types.PrebuiltVoiceConfig(voice_name=chosen["voice"])
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO], temperature=0.5, enable_affective_dialog=False,
        speech_config=types.SpeechConfig(
            language_code="hi-IN", voice_config=types.VoiceConfig(prebuilt_voice_config=voice)
        ),
        system_instruction=types.Content(parts=[types.Part(text=copy_hi.GEMINI_LIVE_SYSTEM_PROMPT)]),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["hi-IN"], custom_vocabulary=list(copy_hi.HINDI_MEDICAL_VOCAB)
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(language_codes=["hi-IN"]),
        realtime_input_config=types.RealtimeInputConfig(
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False, start_of_speech_sensitivity=chosen["start_sens"],
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


def _new_telemetry(
    model: str, voice: str, silence_ms: int, prefix_ms: int, connect_ms: int
) -> dict[str, Any]:
    """One call's counters; `_refresh_latency` fills the rollups as turns land."""
    return {
        "model": model, "voice": voice, "connect_ms": connect_ms, "vad_silence_ms": silence_ms,
        "vad_prefix_padding_ms": prefix_ms, "user_turns": 0, "bot_turns": 0, "tool_calls": 0,
        "tool_errors": 0, "audio_chunks_in": 0, "audio_chunks_out": 0, "interruptions": 0,
        "watchdog_nudges": 0, "turn_latencies_ms": [], "avg_latency_ms": 0, "max_latency_ms": 0,
        "last_interaction_status": "", "dropoff_qid": "", "dropoff_group": "",
        "dropoff_substep": "", "reconnects": 0, "errors": [],
    }


def _refresh_latency(telemetry: dict) -> None:
    """Recompute the average and max turn latency from the raw sample list."""
    lat = telemetry.get("turn_latencies_ms") or []
    telemetry["avg_latency_ms"] = int(round(sum(lat) / len(lat))) if lat else 0
    telemetry["max_latency_ms"] = int(max(lat)) if lat else 0


def _stamp_dropoff(telemetry: dict, state: CallState) -> None:
    """Refresh the rollups, then record where she stopped. A complete call has none."""
    _refresh_latency(telemetry)
    done = state.close_reason is CloseReason.COMPLETE
    spec = spec_or_none(state.cursor.qid)
    telemetry["dropoff_qid"] = "" if done else state.cursor.qid
    telemetry["dropoff_group"] = "" if (done or spec is None) else spec.group
    telemetry["dropoff_substep"] = "" if done else state.cursor.substep.value


async def _send_json(ws: WebSocket, lock: asyncio.Lock, payload: dict) -> None:
    """Send one payload unless the socket is already gone; never raise."""
    if ws.client_state != WebSocketState.CONNECTED:
        return
    try:
        async with lock:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(payload)
    except Exception:
        return


def _audio_from_part(part: types.Part) -> tuple[bytes, int] | None:
    """One audio part as 16-bit PCM at its own rate, or None for text/thought parts."""
    blob = part.inline_data
    if blob is None or blob.data is None:
        return None
    mime = blob.mime_type or ""
    if mime and mime.lower().partition("/")[0] != "audio":
        return None
    pcm = audio_recorder.pcm_bytes(blob.data)
    return (pcm, audio_recorder.rate_from_mime(mime, _OUTPUT_RATE)) if pcm else None


def consume_task_error(task: asyncio.Task) -> None:
    """Surface a background-task failure without ever re-raising it."""
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except Exception:
        return
    if exc is not None:
        logger.warning("background live task failed", exc_info=exc)


class LiveCallSession:
    """Typed state, browser I/O, and Decision API turns for one live call."""

    def __init__(
        self, *, ws: WebSocket, session: Any = None, call_id: int | None = None,
        telemetry: dict | None = None, recorder: Any = None, state: CallState | None = None,
        settings: dict[str, Any] | None = None, thresholds: decisions.DecisionThresholds | None = None,
        stop: asyncio.Event | None = None, ws_lock: asyncio.Lock | None = None,
        send_lock: asyncio.Lock | None = None,
        reconnect: Callable[[str], Awaitable[bool]] | None = None,
        on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
        watchdog_timeout_sec: float | None = None,
    ) -> None:
        self.ws, self.ws_lock = ws, ws_lock or asyncio.Lock()
        self.session, self.send_lock = session, send_lock or asyncio.Lock()
        self.stop, self.call_id = stop or asyncio.Event(), call_id
        self.settings = live_settings() if settings is None else settings
        self.thresholds, self.reconnect = thresholds, reconnect
        self.on_call_completed, self.recorder = on_call_completed, recorder
        self.state: CallState = engine.new_state() if state is None else state
        self.fsm = LiveTurnMachine(call_started_ts=time.time())
        chosen = _session_settings(self.settings)
        self.telemetry = telemetry if telemetry is not None else _new_telemetry(
            chosen["model"], chosen["voice"], chosen["silence_duration_ms"],
            chosen["prefix_padding_ms"], 0,
        )
        self.watchdog_timeout_sec = float(
            chosen["watchdog_timeout_sec"] if watchdog_timeout_sec is None else watchdog_timeout_sec
        )
        self.seen: set[str] = set()
        self.cancelled: set[str] = set()
        self.resume_handle = ""
        self.reconnect_requested = self.go_away_pending = False
        self.speak_fallback: asyncio.Task | None = None

    async def _send_json(self, payload: dict) -> None:
        await _send_json(self.ws, self.ws_lock, payload)

    async def _send_realtime_safe(self, **kwargs) -> bool:
        try:
            async with self.send_lock:
                await self.session.send_realtime_input(**kwargs)
            return True
        except Exception:
            if self.resume_handle and not self.stop.is_set():
                self.reconnect_requested = True
                return True
            return False

    async def _db_safe(self, fn: Any, *args: Any) -> Any:
        if self.call_id is None:
            return None
        try:
            return await asyncio.to_thread(fn, self.call_id, *args)
        except Exception:
            return None

    def _bump(self, key: str) -> None:
        self.telemetry[key] = int(self.telemetry.get(key) or 0) + 1

    async def _emit(self, event_type: str, detail: str = "", meta: dict | None = None) -> None:
        """One telemetry event: DB row, aggregate counters, and the browser mirror."""
        if event_type not in _TELEMETRY_EVENTS:
            return
        self.telemetry["last_interaction_status"] = event_type
        _stamp_dropoff(self.telemetry, self.state)
        evt = await self._db_safe(db.append_call_event, event_type, detail, meta)
        if evt is None:
            evt = {"t": round(time.time(), 3), "type": event_type, "detail": detail or "", "meta": meta or {}}
        snapshot = json.loads(json.dumps(self.telemetry, ensure_ascii=False))
        await self._db_safe(db.update_call_telemetry, snapshot)
        await self._send_json({"type": "telemetry", "telemetry": snapshot, "last_event": evt})

    async def _report_error(self, message: str) -> None:
        if self.call_id is not None:
            errors = self.telemetry.setdefault("errors", [])
            errors.append(message)
            del errors[:-50]
            await self._emit("error", message)
        await self._send_json({"type": "error", "message": message})

    async def _forward_transcript(self, role: str, transcription: types.Transcription | None) -> str:
        """Mirror one transcription chunk to SQLite and the browser; return its text."""
        text = " ".join(((transcription.text if transcription else "") or "").split())
        if not text:
            return ""
        if self.call_id is not None:
            try:
                await asyncio.to_thread(
                    db.append_transcript, self.call_id, role, text, qid=self.state.cursor.qid
                )
            except Exception:
                logger.warning("append_transcript failed on call %s (%s)", self.call_id, role, exc_info=True)
        await self._send_json({"type": "transcript", "role": role, "text": text})
        return text

    def _is_superseded(self, step: str) -> bool:
        """True once a newer turn, a close, or a quick tap has moved the call on."""
        return bool(
            self.fsm.end_requested or self.state.phase is not Phase.ASKING
            or self.state.step_token != step
        )

    def _await_bot_reply(self) -> None:
        self.fsm.awaiting_bot_reply, self.fsm.bot_replied_since_user = True, False
        self.fsm.last_user_turn_ts, self.fsm.nudges_for_turn = time.time(), 0
        self.fsm.quiet_from = 0.0

    def _closing_audio_finished(self, now: float) -> bool:
        """True once the farewell finished playing inside the closing grace window."""
        fsm = self.fsm
        return bool(
            fsm.closing_heard and fsm.phase is not TurnPhase.BOT_SPEAKING
            and fsm.playback_ends_at > 0 and now >= fsm.playback_ends_at + _CLOSING_AUDIO_GRACE_SEC
        )

    async def _mark_closing(self, reason: CloseReason | str) -> None:
        reason = CloseReason(reason)
        if self.state.phase is not Phase.CLOSING:
            self.state.phase, self.state.close_reason = Phase.CLOSING, reason
        elif self.state.close_reason is None:
            self.state.close_reason = reason
        _stamp_dropoff(self.telemetry, self.state)
        await self._db_safe(db.save_cursor_state, self.state)
        if self.fsm.mark_closing(time.time()):
            closed = self.state.close_reason or reason
            await self._emit("end_accepted", closed.value, {"reason": closed.value})

    async def _persist_after_state_change(self) -> None:
        """The one place a reduced turn reaches SQLite, before any speak cue."""
        if self.state.phase is Phase.CLOSING:
            await self._mark_closing(self.state.close_reason or CloseReason.COMPLETE)
        else:
            await self._db_safe(db.save_cursor_state, self.state)

    async def _close_on_technical_failure(self) -> tuple[dict[str, Any], bool, str]:
        self.state = engine.close_technical_failure(self.state)
        await self._mark_closing(self.state.close_reason or CloseReason.TECHNICAL_FAILURE)
        self.fsm.speech_allowed = True
        self.fsm.finish_evaluation()
        return tool_body(self.state), False, "technical_failure"

    async def _send_speak_cue(
        self, kind: SpeakKind, *, text: str | None = None,
        error_msg: str = "Could not prompt Gemini to speak", record_tool_error: bool = True,
    ) -> bool:
        cue = text or copy_hi.speak_cue_text(
            engine.say_hindi(self.state), closing=self.state.phase is not Phase.ASKING,
            clarify=kind is SpeakKind.CLARIFY,
        )
        self.fsm.active_speak_kind, self.fsm.pending_speak_step = kind, ""
        self.fsm.phase = TurnPhase.AWAITING_BOT
        self.fsm.speech_allowed, self.fsm.ignore_bot_audio = True, False
        if await self._send_realtime_safe(text=cue):
            if not self.reconnect_requested:
                self.fsm.speak_cue_step = self.state.step_token
            return True
        if record_tool_error:
            self._bump("tool_errors")
        await self._report_error(error_msg)
        return False

    async def _send_greeting(self) -> None:
        self.fsm.speech_allowed = True
        async with self.send_lock:
            await self.session.send_realtime_input(text=copy_hi.GREETING_CUE + engine.say_hindi(self.state))
        await self._send_json(_question_state_payload(self.state))

    def _cancel_speak_fallback(self) -> None:
        task, self.speak_fallback = self.speak_fallback, None
        if task is not None:
            if not task.done():
                task.cancel()
            task.remove_done_callback(consume_task_error)
            task.add_done_callback(consume_task_error)

    async def _cue_after_decision(
        self, *, clarify: bool = False, delay_sec: float = _TOOL_SPEAK_FALLBACK_SEC,
        cue_now: bool = False,
    ) -> bool:
        """Publish the new step, then owe a cue. `turn_complete` may still cancel it."""
        await self._send_json(_question_state_payload(self.state))
        if self.state.phase is not Phase.ASKING:
            kind = SpeakKind.CLOSE
        else:
            kind = SpeakKind.CLARIFY if clarify else SpeakKind.SPEAK
        self._cancel_speak_fallback()
        if cue_now:
            if not self.fsm.end_requested:
                self._await_bot_reply()
            if await self._send_speak_cue(kind):
                return True
            self.stop.set()
            return False
        self.fsm.pending_speak_step = self.state.step_token
        self.fsm.pending_speak_kind = self.fsm.active_speak_kind = kind
        if delay_sec <= 0:
            await self._speak_if_silent(self.state.step_token, kind, 0)
            return True
        self.speak_fallback = asyncio.create_task(
            self._speak_if_silent(self.state.step_token, kind, delay_sec)
        )
        self.speak_fallback.add_done_callback(consume_task_error)
        return True

    async def _speak_if_silent(self, step_after_tool: str, kind: SpeakKind, delay_sec: float) -> None:
        """Speak the tool's next line only when Gemini stayed silent after the result."""
        try:
            if delay_sec > 0:
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=delay_sec)
                    return
                except asyncio.TimeoutError:
                    pass
            quiet = not (self.fsm.bot_replied_since_user or self.fsm.phase is TurnPhase.BOT_SPEAKING)
            if quiet and not self.stop.is_set() and self.state.step_token == step_after_tool:
                if not await self._send_speak_cue(kind):
                    self.stop.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._bump("tool_errors")
            await self._report_error("Could not prompt Gemini to speak")

    async def _execute_decision_turn(self) -> tuple[dict[str, Any], bool, str]:
        """One caller turn: classify, extract, reduce once, persist. Nothing else."""
        step_before = self.state.step_token
        hold, utt, pcm, snapshot, _key = self.fsm.begin_evaluation(self.state)
        self.fsm.speech_allowed = True
        if self.state.phase is not Phase.ASKING:
            return tool_body(self.state), False, "closing"
        if hold:
            return {**tool_body(self.state), "result": "held", "reason": hold}, False, "held"
        try:
            resolved = self._resolve_thresholds()
        except Exception:
            logger.exception("decision thresholds invalid on call %s", self.call_id)
            await self._db_safe(db.append_call_event, "decision_turn", "error:invalid_thresholds",
                                _decision_meta(error_reason="invalid_thresholds"))
            return await self._close_on_technical_failure()

        audio_wav = audio_recorder.pcm16_to_wav_bytes(pcm, _INPUT_RATE) if (not utt and pcm) else None
        res = await decisions.evaluate_turn(
            decisions.build_turn_request(snapshot, utterance_text=utt, audio_wav_bytes=audio_wav)
        )
        stale = ({**tool_body(self.state), "result": "held", "reason": "stale_turn"}, False, "held")
        meta = _decision_meta(res)
        if self._is_superseded(step_before):
            await self._db_safe(db.append_call_event, "decision_turn", "superseded", meta)
            self.fsm.finish_evaluation()
            return stale
        if not res.ok:
            await self._db_safe(db.append_call_event, "decision_turn",
                                f"error:{res.error_reason}" if res.error_reason else "error", meta)
            return await self._close_on_technical_failure()

        intent = decisions.interpret_turn_decision(
            self.state, res.raw_answers, thresholds=resolved, utterance_text=utt
        )
        cmd, extraction = intent.command, None
        meta["diagnostics"] = intent.diagnostics
        if intent.future_qid_match is not None:
            meta["future_qid_match"] = intent.future_qid_match
        if intent.needs_profile_extraction and not cmd.emergency and not cmd.caller_stop:
            extraction = await profile_extractor.extract_profile_entities(
                self.state, utterance_text=utt, audio_wav_bytes=audio_wav
            )
            meta |= {"extractor_ok": extraction.ok, "extractor_ms": extraction.extractor_ms,
                     "extractor_error": extraction.error_reason}
            if self._is_superseded(step_before):
                await self._db_safe(db.append_call_event, "decision_turn", "superseded", meta)
                self.fsm.finish_evaluation()
                return stale
            if extraction.ok:
                self.fsm.extractor_failures, self.fsm.extractor_failure_step = 0, ""
                cmd = cmd.model_copy(
                    update={"profile_patch": extraction.patch, "profile_turn_attempted": True}
                )

        await self._db_safe(db.append_call_event, "decision_turn",
                            f"{self.state.cursor.qid}:{cmd.outcome.value or 'ok'}", meta)
        if extraction is not None and not extraction.ok:
            if self.fsm.record_extractor_failure(step_before):
                return await self._close_on_technical_failure()
            if engine.step_kind(self.state) == "profile":
                self.fsm.finish_evaluation()
                return tool_body(self.state), False, (cmd.outcome.value or "reask")

        new_state = engine.reduce_turn(self.state, cmd)
        step_changed = new_state.step_token != step_before
        self.state = new_state
        await self._persist_after_state_change()
        clarify = bool(
            not cmd.emergency and not cmd.caller_stop and (
                cmd.outcome is TurnOutcome.CLARIFY
                or (cmd.side_action is SideAction.CLARIFY_MEANING
                    and self.state.cursor.qid == cmd.side_qid)
            )
        )
        self.fsm.finish_evaluation(step_changed=step_changed)
        body = tool_body(self.state, clarify=clarify)
        if cmd.emergency:
            return body, clarify, "emergency"
        if cmd.caller_stop:
            return body, clarify, "caller_stop"
        return body, clarify, cmd.outcome.value or "ok"

    def _resolve_thresholds(self) -> decisions.DecisionThresholds:
        if self.thresholds is not None:
            return decisions.validate_thresholds(self.thresholds)
        return decisions.load_thresholds(self.settings.get("decision_thresholds_path"))

    async def _route_call(self, name: str) -> tuple[dict[str, Any], str, bool]:
        """Route one call as (body, detail, clarify); the detail names the outcome."""
        if name != "next_question":
            self._bump("tool_errors")
            return {"result": "ignored", "say": engine.say_hindi(self.state)}, name, False
        if self.state.phase is not Phase.ASKING:
            self.fsm.speech_allowed = True
            return tool_body(self.state), "closing", False
        hold = self.fsm.hold_reason_for_tool(self.state, time.time())
        if hold:
            self.fsm.speech_allowed = True
            return {**tool_body(self.state), "result": "held", "reason": hold}, "held", False
        body, clarify, kind = await self._execute_decision_turn()
        return body, kind, clarify

    async def _answer_function_calls(self, calls: list) -> tuple[bool, bool]:
        """Answer every fresh `next_question`, then owe one cue for the new step."""
        responses: list[types.FunctionResponse] = []
        logged: list[str] = []
        decision_ran = clarify = False
        stale_only = True
        for fc in calls:
            fc_id, name = getattr(fc, "id", None) or "", getattr(fc, "name", None) or ""
            if (fc_id and (fc_id in self.cancelled or fc_id in self.seen)) or getattr(
                fc, "will_continue", False
            ):
                continue
            if fc_id:
                self.seen.add(fc_id)
            try:
                body, detail, call_clarify = await self._route_call(name)
                decision_ran = decision_ran or detail != "held"
                clarify, stale_only = clarify or call_clarify, stale_only and detail == "held"
                logged.append(f"next_question_{detail}" if name == "next_question" else detail)
            except Exception:
                logger.exception("tool %s failed on call %s", name, self.call_id)
                self._bump("tool_errors")
                body = {"result": "error", "say": engine.say_hindi(self.state)}
                logged.append(f"{name or 'tool'}_error")
                clarify, stale_only = False, False
            if self.state.phase is Phase.CLOSING:
                body["stop"], body["say"] = True, engine.say_hindi(self.state)
            if fc_id:
                responses.append(
                    types.FunctionResponse(id=fc_id, name=name or "next_question", response=body)
                )

        send_failed = False
        if responses:
            try:
                async with self.send_lock:
                    await self.session.send_tool_response(function_responses=responses)
            except Exception:
                send_failed = True
        for detail in logged:
            self._bump("tool_calls")
            await self._emit("tool_call", detail)
        if send_failed:
            self._bump("tool_errors")
            await self._report_error("Could not send the tool response to Gemini")
        elif stale_only and (
            self.fsm.speak_cue_step == self.state.step_token or self.fsm.question_already_spoken(self.state)
        ):
            return decision_ran, stale_only
        await self._cue_after_decision(
            clarify=clarify, delay_sec=0.0 if send_failed else _TOOL_SPEAK_FALLBACK_SEC
        )
        return decision_ran, stale_only

    async def _on_bot_audio(self, pcm: bytes, rate: int) -> None:
        accepted, first, latency_ms = self.fsm.on_bot_audio(self.state, len(pcm), rate, time.time())
        if not accepted:
            return
        self._bump("audio_chunks_out")
        if first:
            self._bump("bot_turns")
            if latency_ms is not None:
                self.telemetry["turn_latencies_ms"].append(latency_ms)
        if self.recorder is not None:
            self.recorder.add_bot_pcm(pcm, rate)
        if self.fsm.end_requested:
            self.fsm.closing_heard = self.telemetry["closing_heard"] = True
        await self._send_json({
            "type": "audio", "data": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": _OUTPUT_RATE,
        })
        if first:
            meta: dict[str, Any] = {"spoken_qid": self.fsm.spoken_step}
            if latency_ms is not None:
                meta["latency_ms"] = latency_ms
            await self._emit("bot_first_audio", f"{latency_ms}ms" if latency_ms else "", meta)

    async def _handle_server_message(self, response) -> bool:
        """Handle one Live server message. Return False when the session should stop."""
        fsm = self.fsm
        if (canc := getattr(response, "tool_call_cancellation", None)) is not None and canc.ids:
            self.cancelled.update(item for item in canc.ids if item)

        sc, calls = getattr(response, "server_content", None), []
        if sc is not None:
            if sc.interrupted:
                fsm.on_interrupted()
                self._bump("interruptions")
                await self._send_json({"type": "clear"})
                await self._emit("interrupted", "interrupted")
            input_tr = getattr(sc, "input_transcription", None)
            user_text = await self._forward_transcript("user", input_tr)
            if user_text:
                if fsm.on_user_transcript(self.state, user_text, time.time()):
                    self._bump("user_turns")
                await self._emit("user_transcript", user_text)
            if bool(getattr(input_tr, "finished", False)) and not fsm.end_requested:
                fsm.input_finalized = True
            if fsm.can_accept_bot_output():
                await self._forward_transcript("assistant", sc.output_transcription)
            for part in (sc.model_turn.parts if sc.model_turn is not None else None) or []:
                if getattr(part, "thought", False):
                    continue
                if (audio := _audio_from_part(part)) is not None:
                    await self._on_bot_audio(*audio)
                if (fc := getattr(part, "function_call", None)) is not None:
                    calls.append(fc)

        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None and tool_call.function_calls:
            calls.extend(tool_call.function_calls)

        decision_ran = False
        if calls:
            decision_ran, stale_only = await self._answer_function_calls(calls)
            spoken = fsm.speak_cue_step == self.state.step_token or fsm.question_already_spoken(self.state)
            if not fsm.end_requested and not (stale_only and (fsm.bot_replied_since_user or spoken)):
                self._await_bot_reply()

        if sc is not None and sc.turn_complete and not await self._on_turn_complete(decision_ran):
            return False

        update = getattr(response, "session_resumption_update", None)
        if update is not None and getattr(update, "resumable", False):
            self.resume_handle = str(getattr(update, "new_handle", "") or self.resume_handle)
        if getattr(response, "go_away", None) is not None:
            if not self.resume_handle:
                return False
            busy = fsm.phase is TurnPhase.BOT_SPEAKING or fsm.awaiting_bot_reply
            self.go_away_pending, self.reconnect_requested = busy, not busy
        return True

    async def _on_turn_complete(self, decision_ran: bool) -> bool:
        """`turn_complete` either catches a missed tool turn or closes the bot turn."""
        fsm = self.fsm
        if fsm.should_run_missed_tool_on_turn_complete(self.state):
            _body, clarify, kind = await self._execute_decision_turn()
            decision_ran = True
            self._bump("tool_calls")
            await self._emit("tool_call", f"missed_tool_{kind}")
            if not await self._cue_after_decision(clarify=clarify, cue_now=True):
                return False
        cue_kind, emit_complete = fsm.on_turn_complete_cleanup(
            self.state, time.time(), decision_ran=decision_ran
        )
        if self.go_away_pending:
            self.go_away_pending, self.reconnect_requested = False, True
            self._cancel_speak_fallback()
            cue_kind = None
        if cue_kind is not None and not self.stop.is_set() and not self.reconnect_requested:
            self._cancel_speak_fallback()
            if not await self._send_speak_cue(cue_kind):
                self.stop.set()
                return False
        if not fsm.awaiting_bot_reply:
            fsm.quiet_from = max(time.time(), fsm.playback_ends_at)
            await self._send_json({"type": "bot_turn_complete", "step": self.state.step_token})
        if emit_complete:
            await self._emit("turn_complete", "")
        return True

    async def _reconnect_once(self) -> bool:
        """Swap the Live session, then re-owe the pending cue on the same step."""
        self._cancel_speak_fallback()
        if self.reconnect is None or not await self.reconnect(self.resume_handle):
            self.stop.set()
            await self._report_error("Gemini Live could not reconnect")
            return False
        resume_speak = self.fsm.on_reconnected(time.time())
        self.seen.clear()
        self.cancelled.clear()
        if resume_speak and self.state.phase is Phase.ASKING and not self.fsm.end_requested:
            self._await_bot_reply()
            await self._send_json(_question_state_payload(self.state))
            if not await self._send_speak_cue(self.fsm.active_speak_kind, record_tool_error=False):
                self.stop.set()
                return False
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
                    if self.reconnect_requested:
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.stop.is_set() or not self.resume_handle:
                    self.stop.set()
                    await self._report_error("Gemini Live session ended")
                    return
                self.reconnect_requested = True
            if not self.reconnect_requested:
                continue
            self.reconnect_requested = False
            reconnects += 1
            if reconnects > _MAX_RECONNECTS:
                self.stop.set()
                await self._report_error("Gemini Live could not reconnect")
                return
            await self._send_json({"type": "reconnecting", "attempt": reconnects})
            await self._emit("reconnecting", str(reconnects))
            if not await self._reconnect_once():
                return
            self.telemetry["reconnects"] = reconnects
            await self._send_json({"type": "reconnected", "attempt": reconnects})
            await self._emit("reconnected", str(reconnects))

    async def _send_nudge(self, event_type: str = "watchdog_nudge", detail: str = "") -> bool:
        """Re-ask the current line; used by both the watchdog and the HUD nudge button."""
        if event_type == "watchdog_nudge":
            self._bump("watchdog_nudges")
        if self.fsm.awaiting_bot_reply:
            self.fsm.last_user_turn_ts = time.time()
        self.fsm.speech_allowed = True
        if not await self._send_speak_cue(
            self.fsm.active_speak_kind, text=detail or None,
            error_msg="Could not nudge Gemini", record_tool_error=False,
        ):
            self.stop.set()
            return False
        if not self.reconnect_requested:
            await self._emit(event_type, detail)
        return True

    async def _watchdog_step(self, action: WatchdogAction) -> bool:
        """Apply one timed action; return False when the call must stop."""
        if action is WatchdogAction.WATCHDOG_NUDGE:
            return await self._send_nudge()
        if action in (WatchdogAction.CLOSING_COMPLETE, WatchdogAction.HARD_LIMIT_STOP):
            if action is WatchdogAction.CLOSING_COMPLETE:
                if self.fsm.mark_ending_notified():
                    await self._send_json({"type": "ending"})
            elif self.state.phase is Phase.ASKING:
                await self._mark_closing(CloseReason.TIME_LIMIT)
            self.stop.set()
            return False
        reason, detail, text = _CLOSE_CUES[action]
        if reason is not None:
            await self._mark_closing(reason)
        self.fsm.speech_allowed = True
        if not await self._send_speak_cue(SpeakKind.SPEAK, text=text, record_tool_error=False):
            self.stop.set()
            return False
        if not self.reconnect_requested:
            await self._emit("auto_close" if reason is not None else "silence_check_in", detail)
        return True

    async def _watchdog_loop(self) -> None:
        """One timer tick per half second: end the call, close it, or nudge a silent model."""
        try:
            while not self.stop.is_set():
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=_WATCHDOG_INTERVAL_SEC)
                    return
                except asyncio.TimeoutError:
                    pass
                if self.stop.is_set():
                    return
                action = self.fsm.poll_watchdog(self.state, time.time(), self.watchdog_timeout_sec)
                if action is not WatchdogAction.NONE and not await self._watchdog_step(action):
                    return
        except asyncio.CancelledError:
            raise

    async def _on_browser_audio(self, payload: dict) -> bool:
        pcm = audio_recorder.pcm_bytes(payload.get("data") or "")
        if not pcm:
            return True
        try:
            rate = int(payload.get("sample_rate") or _INPUT_RATE)
        except (TypeError, ValueError):
            rate = _INPUT_RATE
        rate = rate if rate > 0 else _INPUT_RATE
        self._bump("audio_chunks_in")
        if self.recorder is not None:
            self.recorder.add_user_pcm(pcm, rate)
        pcm_16k = pcm if rate == _INPUT_RATE else audio_recorder.resample_int16_mono(pcm, rate, _INPUT_RATE)
        if self.fsm.on_browser_audio_energy(self.state, pcm_16k, audio_recorder.pcm_rms(pcm), time.time()):
            self._bump("user_turns")
        if not await self._send_realtime_safe(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={rate}")
        ):
            self.stop.set()
            await self._report_error("Could not forward audio to Gemini")
            return False
        return True

    async def _on_browser_text(self, payload: dict) -> bool:
        text = str(payload.get("text") or "").strip()
        if not text:
            return True
        await self._forward_transcript("user", types.Transcription(text=text))
        if self.fsm.on_browser_text(self.state, text, time.time()):
            self._bump("user_turns")
        if not await self._send_realtime_safe(text=text):
            self.stop.set()
            await self._report_error("Could not forward text to Gemini")
            return False
        await self._emit("user_transcript", text)
        return True

    async def _on_browser_quick_reply(self, payload: dict) -> bool:
        """A tap is a whole turn: same reducer, same persistence, same speak cue."""
        step = str(payload.get("step") or "")
        outcome = str(payload.get("outcome") or "").strip().lower()
        if not self.fsm.can_accept_quick_reply(self.state, step, outcome):
            return True
        self.fsm.on_quick_reply_accepted(step, time.time())
        said = "नहीं।" if outcome == "clear" else "हाँ।"
        await self._forward_transcript("user", types.Transcription(text=said))
        await self._emit("user_transcript", said)
        self._bump("tool_calls")
        await self._emit("tool_call", "quick_reply_clear" if outcome == "clear" else "quick_reply_concern")
        self.state = engine.reduce_turn(self.state, engine.quick_reply_command(AnswerStatus(outcome)))
        await self._persist_after_state_change()
        await self._send_json({"type": "clear"})
        await self._send_json(_question_state_payload(self.state))
        self._cancel_speak_fallback()
        closing = self.state.phase is not Phase.ASKING
        if closing:
            self._bump("user_turns")
        self._await_bot_reply()
        if not await self._send_speak_cue(
            SpeakKind.CLOSE if closing else SpeakKind.SPEAK,
            error_msg="Could not speak the next question", record_tool_error=False,
        ):
            self.stop.set()
            return False
        return True

    async def _browser_loop(self) -> None:
        handlers = {
            "audio": self._on_browser_audio,
            "text": self._on_browser_text,
            "quick_reply": self._on_browser_quick_reply,
        }
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
                if self.fsm.end_requested and kind in {"audio", "text", "speech_end", "nudge"}:
                    continue
                if kind == "speech_end":
                    spoken = self.fsm.on_speech_end(self.state, time.time())
                    await self._db_safe(db.append_call_event, "speech_end",
                                        "already_spoken" if spoken else "")
                elif kind == "hangup":
                    self.stop.set()
                    return
                elif kind == "nudge":
                    if not await self._send_nudge("manual_nudge"):
                        return
                elif kind in handlers and not await handlers[kind](payload):
                    return
        except asyncio.CancelledError:
            raise

    async def _finalize(self) -> None:
        if self.fsm.mark_ending_notified():
            await self._send_json({"type": "ending"})
        filename, duration = "", 0
        if self.recorder is not None:
            try:
                filename, duration = await asyncio.to_thread(self.recorder.save)
            except Exception:
                filename, duration = "", 0
        _stamp_dropoff(self.telemetry, self.state)
        await self._db_safe(db.update_call_telemetry, json.loads(json.dumps(self.telemetry)))
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
            "call_id": self.call_id, "share_token": call.get("share_token") or "",
        })

    async def run(self) -> None:
        tasks: list[asyncio.Task] = []
        try:
            tasks += [
                asyncio.create_task(self._gemini_loop()),
                asyncio.create_task(self._watchdog_loop()),
            ]
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
            for res in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    logger.warning("live call task failed on call %s", self.call_id, exc_info=res)


async def _connect_live(client: genai.Client, model: str, settings: dict, handle: str = "") -> tuple:
    """Open one Live context with up to three attempts, so a dropped resume survives."""
    for attempt in range(3):
        try:
            started = time.perf_counter()
            ctx = client.aio.live.connect(model=model, config=_live_config(settings, resume_handle=handle))
            session = await ctx.__aenter__()
            if getattr(session, "setup_complete", None) is None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                raise RuntimeError("Gemini Live setup did not complete")
            return ctx, session, int((time.perf_counter() - started) * 1000)
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(0.4 * (attempt + 1))
    raise RuntimeError("Gemini Live setup did not complete")


async def handle_gemini_ws(
    ws: WebSocket, caller_name: str = "", user_role: str = "user",
    on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
    clinic_code: str = "", thresholds: decisions.DecisionThresholds | None = None,
    settings_override: dict | None = None,
) -> None:
    """Open one Live call: validate, connect, then hand the socket to the session."""
    if ws.client_state == WebSocketState.CONNECTING:
        await ws.accept()
    caller_name, user_role = caller_name or "", user_role or "user"
    ws_lock, send_lock = asyncio.Lock(), asyncio.Lock()
    live: LiveCallSession | None = None
    client = session_ctx = None

    async def reject(message: str) -> None:
        await _send_json(ws, ws_lock, {"type": "error", "message": message})

    try:
        try:
            settings = live_settings(settings_override)
            chosen = _session_settings(settings)
            _live_config(settings)
        except Exception:
            await reject("Invalid live call settings")
            return
        try:
            loaded = (
                decisions.validate_thresholds(thresholds) if thresholds is not None
                else decisions.load_thresholds(settings.get("decision_thresholds_path"))
            )
        except Exception:
            await reject("Invalid or missing Decision API thresholds")
            return
        if not os.environ.get("GEMINI_API_KEY"):
            await reject("GEMINI_API_KEY is not set")
            return

        model, voice = chosen["model"], chosen["voice"]
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1beta"}
        )
        try:
            session_ctx, session, connect_ms = await _connect_live(client, model, settings)
        except Exception:
            await reject("Gemini Live connection failed")
            return

        call_id = await asyncio.to_thread(
            db.create_call, caller_name=caller_name, user_role=user_role,
            model_used=model, voice_used=voice, clinic_code=clinic_code or "",
        )
        live = LiveCallSession(
            ws=ws, ws_lock=ws_lock, session=session, send_lock=send_lock, call_id=call_id,
            telemetry=_new_telemetry(
                model, voice, chosen["silence_duration_ms"], chosen["prefix_padding_ms"], connect_ms
            ),
            recorder=audio_recorder.CallAudioRecorder(call_id), settings=settings,
            thresholds=loaded, on_call_completed=on_call_completed,
            watchdog_timeout_sec=chosen["watchdog_timeout_sec"],
        )

        async def reconnect(handle: str) -> bool:
            """Swap the session object in place: the call reads `self.session` on every send."""
            nonlocal session, session_ctx
            try:
                new_ctx, new_session, _ms = await _connect_live(client, model, settings, handle)
            except Exception:
                return False
            old_ctx = session_ctx
            async with send_lock:
                session, session_ctx = new_session, new_ctx
                live.session, live.reconnect_requested = new_session, False
            try:
                await old_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            return True

        live.reconnect = reconnect
        await live._db_safe(db.save_cursor_state, live.state)
        call = await asyncio.to_thread(db.get_call, call_id) or {}
        await live._send_json({
            "type": "session_started", "call_id": call_id, "model": model, "voice": voice,
            "user_role": user_role, "caller_name": caller_name,
            "share_token": str(call.get("share_token") or ""), "out_sample_rate": _OUTPUT_RATE,
        })
        await live._emit("session_started", model)
        await live.run()
    finally:
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        closer = getattr(getattr(client, "aio", None), "aclose", None)
        if callable(closer):
            try:
                if inspect.isawaitable(res := closer()):
                    await res
            except Exception:
                pass
        if live is not None:
            try:
                await live._finalize()
            except Exception:
                await reject("Could not finish the call")
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except Exception:
                pass
