import asyncio
import base64
import json
import os
import re
import time
from typing import Any

import websockets
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import audio_recorder
import db
import prompts
import summarizer

load_dotenv("/Users/jigar/projects/0n-going/SHISHU-healthathon-koita/.env")

OPENAI_LIVE_VOICE_PROMPT = prompts.OPENAI_LIVE_VOICE_PROMPT
OPENAI_RESPONSES_BACKEND_PROMPT = prompts.OPENAI_RESPONSES_BACKEND_PROMPT
generate_call_summaries = summarizer.generate_call_summaries

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
LIVE_MODEL = "gpt-live-1"
LIVE_VOICE = "marin"
LIVE_BACKEND_MODEL = "gpt-5.6-luna"
SAMPLE_RATE = 24000
GREETING = "Greet the mother warmly in Hindi as SHISHU and ask her name."
_NUDGE_TEXT = (
    "The mother finished speaking and is waiting. "
    "Respond warmly in Hindi now and ask the next question."
)
_WATCHDOG_POLL_SEC = 0.5
_MAX_NUDGES_PER_TURN = 2
_SILENCE_100MS = base64.b64encode(b"\x00" * (SAMPLE_RATE // 10 * 2)).decode("ascii")
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+)", re.IGNORECASE)

_USER_TRANSCRIPT_TYPES = {
    "session.input_transcript.delta",
    "session.input_transcript.done",
    "session.input_audio_transcript.delta",
    "session.input_audio_transcript.done",
    "session.input_audio_transcript.completed",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
}
_ASSISTANT_TRANSCRIPT_TYPES = {
    "session.output_transcript.delta",
    "session.output_transcript.done",
    "session.output_text.delta",
    "session.output_text.done",
    "session.output_audio_transcript.delta",
    "session.output_audio_transcript.done",
    "session.output_audio_transcript.completed",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
}
_CLEAR_EVENT_TYPES = {
    "input_audio_buffer.speech_started",
    "session.input_audio.speech_started",
    "output_audio_buffer.cleared",
    "response.cancelled",
}


class _CallState:
    def __init__(
        self,
        call_id: int,
        recorder: audio_recorder.CallAudioRecorder,
        live_model: str = LIVE_MODEL,
        voice: str = LIVE_VOICE,
        backend_model: str = LIVE_BACKEND_MODEL,
        watchdog_timeout_sec: float = 4.0,
        caller_name: str = "",
        user_role: str = "user",
    ) -> None:
        self.call_id = call_id
        self.recorder = recorder
        self.live_model = live_model
        self.voice = voice
        self.backend_model = backend_model
        self.watchdog_timeout_sec = watchdog_timeout_sec
        self.caller_name = caller_name
        self.user_role = user_role
        self.finalized = False
        self.started = False
        self.session_ready = False
        self.audio_queue: list[bytes] = []
        self.pending_text: list[str] = []
        self.bot_speaking = False
        self.streamed_roles: set[str] = set()
        self.send_lock = asyncio.Lock()
        self.tool_seq = 0
        self.nudge_seq = 0
        self.connect_t0 = 0.0
        self.awaiting_bot_reply = False
        self.last_user_turn_ts = 0.0
        self.bot_replied_since_user = True
        self.nudges_for_current_turn = 0
        self.saw_user_transcript = False
        self.telemetry: dict[str, Any] = {
            "model": live_model,
            "voice": voice,
            "backend_model": backend_model,
            "connect_ms": 0,
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
            "errors": [],
        }


def _public_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return _SECRET_RE.sub("***", text)[:500]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _pcm24(pcm_bytes: bytes, sample_rate: int) -> bytes:
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    if not pcm_bytes:
        return b""
    if sample_rate != SAMPLE_RATE:
        pcm_bytes = audio_recorder._resample_int16_mono(pcm_bytes, sample_rate, SAMPLE_RATE)
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    return pcm_bytes


def _transcript_role(event_type: str) -> str | None:
    if event_type in _USER_TRANSCRIPT_TYPES:
        return "user"
    if event_type in _ASSISTANT_TRANSCRIPT_TYPES:
        return "assistant"
    if "transcript" not in event_type:
        return None
    if "input" in event_type:
        return "user"
    return "assistant"


def _event_text(event: dict[str, Any]) -> str:
    for key in ("delta", "transcript", "text"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _session_start_payload(live_model: str, voice: str, backend_model: str) -> dict[str, Any]:
    return {
        "type": "session.start",
        "event_id": "evt_start",
        "session": {
            "model": live_model,
            "instructions": OPENAI_LIVE_VOICE_PROMPT,
            "audio": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "output": {"voice": voice},
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": backend_model,
                    "instructions": OPENAI_RESPONSES_BACKEND_PROMPT,
                    "tools": [
                        {
                            "type": "function",
                            "name": "record_finding",
                            "description": "Record a clinical intake finding, patient profile field, or red flag.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "domain": {"type": "string"},
                                    "status": {"type": "string"},
                                    "detail": {"type": "string"},
                                    "is_red_flag": {"type": "boolean"},
                                },
                                "required": ["domain", "status", "detail", "is_red_flag"],
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
            },
        },
    }


async def _send_browser(ws: WebSocket, payload: dict[str, Any]) -> None:
    if ws.client_state != WebSocketState.CONNECTED:
        return
    try:
        await ws.send_json(payload)
    except Exception:
        return


async def _oai_send(oai_ws: Any, state: _CallState, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False)
    async with state.send_lock:
        await oai_ws.send(raw)


def _telemetry_snapshot(state: _CallState) -> dict[str, Any]:
    snap = dict(state.telemetry)
    snap["turn_latencies_ms"] = list(state.telemetry["turn_latencies_ms"])
    snap["errors"] = list(state.telemetry["errors"])
    return snap


def _record_error(state: _CallState, message: str) -> None:
    safe = _SECRET_RE.sub("***", message)[:500]
    errors = state.telemetry["errors"]
    errors.append(safe)
    if len(errors) > 30:
        del errors[:-30]


def _note_latency(state: _CallState) -> None:
    if state.last_user_turn_ts <= 0:
        return
    latency = int((time.monotonic() - state.last_user_turn_ts) * 1000)
    if latency < 0:
        latency = 0
    latencies = state.telemetry["turn_latencies_ms"]
    latencies.append(latency)
    state.telemetry["avg_latency_ms"] = int(round(sum(latencies) / len(latencies)))
    state.telemetry["max_latency_ms"] = max(latencies)


async def _emit_telemetry(
    ws: WebSocket,
    state: _CallState,
    event_type: str,
    detail: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_detail = _SECRET_RE.sub("***", detail or "")[:500]
    snap = _telemetry_snapshot(state)
    try:
        evt = db.append_call_event(state.call_id, event_type, safe_detail, meta)
    except Exception as exc:
        _record_error(state, _public_error(exc))
        snap = _telemetry_snapshot(state)
        evt = {
            "t": round(time.time(), 3),
            "type": event_type,
            "detail": safe_detail,
            "meta": meta or {},
        }
    try:
        db.update_call_telemetry(state.call_id, snap)
    except Exception as exc:
        _record_error(state, _public_error(exc))
        snap = _telemetry_snapshot(state)
    await _send_browser(ws, {"type": "telemetry", "telemetry": snap, "last_event": evt})
    return evt


async def _append_commentary(oai_ws: Any, state: _CallState, content: str, event_id: str) -> None:
    payload = {
        "type": "session.commentary.append",
        "event_id": event_id,
        "delegation_id": None,
        "content": content,
    }
    async with state.send_lock:
        if not state.session_ready:
            state.pending_text.append(content)
            return
        await oai_ws.send(json.dumps(payload, ensure_ascii=False))


async def _arm_user_turn(ws: WebSocket, state: _CallState, text: str) -> None:
    new_turn = not state.awaiting_bot_reply
    state.awaiting_bot_reply = True
    state.last_user_turn_ts = time.monotonic()
    state.bot_replied_since_user = False
    state.saw_user_transcript = True
    if new_turn:
        state.nudges_for_current_turn = 0
        state.telemetry["user_turns"] += 1
        await _emit_telemetry(ws, state, "user_transcript", text)


async def _send_nudge(ws: WebSocket, oai_ws: Any, state: _CallState, *, automatic: bool) -> None:
    state.nudge_seq += 1
    event_id = f"evt_nudge_{state.nudge_seq}"
    try:
        await _append_commentary(oai_ws, state, _NUDGE_TEXT, event_id)
    except Exception as exc:
        safe = _public_error(exc)
        _record_error(state, safe)
        await _emit_telemetry(ws, state, "error", safe)
        return
    kind = "watchdog_nudge" if automatic else "manual_nudge"
    await _emit_telemetry(ws, state, kind, _NUDGE_TEXT)


async def _watchdog_loop(ws: WebSocket, oai_ws: Any, state: _CallState) -> None:
    try:
        while not state.finalized:
            await asyncio.sleep(_WATCHDOG_POLL_SEC)
            if state.finalized or not state.session_ready:
                continue
            if not state.awaiting_bot_reply or state.bot_replied_since_user:
                continue
            if not state.saw_user_transcript:
                continue
            if state.nudges_for_current_turn >= _MAX_NUDGES_PER_TURN:
                continue
            if state.last_user_turn_ts <= 0:
                continue
            if time.monotonic() - state.last_user_turn_ts < state.watchdog_timeout_sec:
                continue
            state.nudges_for_current_turn += 1
            state.telemetry["watchdog_nudges"] += 1
            state.last_user_turn_ts = time.monotonic()
            await _send_nudge(ws, oai_ws, state, automatic=True)
    except asyncio.CancelledError:
        return


def _setting_text(settings: dict[str, str], key: str, default: str) -> str:
    raw = settings.get(key)
    if not isinstance(raw, str):
        return default
    text = raw.strip()
    return text or default


async def _append_input_audio(oai_ws: Any, state: _CallState, pcm24: bytes) -> None:
    if not pcm24:
        return
    payload = {
        "type": "session.input_audio.append",
        "audio": base64.b64encode(pcm24).decode("ascii"),
    }
    async with state.send_lock:
        if not state.session_ready:
            state.audio_queue.append(pcm24)
            return
        await oai_ws.send(json.dumps(payload))


async def _browser_loop(ws: WebSocket, oai_ws: Any, state: _CallState) -> None:
    while True:
        try:
            raw = await ws.receive_text()
        except WebSocketDisconnect:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await _send_browser(ws, {"type": "error", "message": "Invalid JSON from browser"})
            continue
        if not isinstance(msg, dict):
            continue
        kind = msg.get("type")
        if kind == "audio":
            data = msg.get("data") or ""
            if not isinstance(data, str) or not data:
                continue
            try:
                pcm = base64.b64decode(data)
            except Exception:
                await _send_browser(ws, {"type": "error", "message": "Invalid audio payload"})
                continue
            try:
                sample_rate = int(msg.get("sample_rate") or SAMPLE_RATE)
            except (TypeError, ValueError):
                sample_rate = SAMPLE_RATE
            state.recorder.add_user_pcm(pcm, sample_rate)
            state.telemetry["audio_chunks_in"] += 1
            await _append_input_audio(oai_ws, state, _pcm24(pcm, sample_rate))
        elif kind == "text":
            text = msg.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            db.append_transcript(state.call_id, "user", text)
            await _send_browser(ws, {"type": "transcript", "role": "user", "text": text})
            if state.bot_speaking:
                state.bot_speaking = False
                state.telemetry["interruptions"] += 1
                await _send_browser(ws, {"type": "clear"})
                await _emit_telemetry(ws, state, "interrupted", "user text")
            commentary = {
                "type": "session.commentary.append",
                "event_id": "evt_user_text",
                "delegation_id": None,
                "content": text,
            }
            async with state.send_lock:
                if not state.session_ready:
                    state.pending_text.append(text)
                else:
                    await oai_ws.send(json.dumps(commentary, ensure_ascii=False))
            await _arm_user_turn(ws, state, text)
        elif kind == "speech_end":
            state.last_user_turn_ts = time.monotonic()
            if state.saw_user_transcript and not state.bot_replied_since_user:
                state.awaiting_bot_reply = True
            await _emit_telemetry(ws, state, "speech_end", "watchdog armed")
        elif kind == "nudge":
            state.nudges_for_current_turn += 1
            state.last_user_turn_ts = time.monotonic()
            state.awaiting_bot_reply = True
            state.bot_replied_since_user = False
            await _send_nudge(ws, oai_ws, state, automatic=False)
        elif kind == "hangup":
            return


async def _handle_transcript(ws: WebSocket, state: _CallState, event: dict[str, Any]) -> None:
    event_type = str(event.get("type") or "")
    role = _transcript_role(event_type)
    text = _event_text(event)
    if not role or not text:
        return
    is_delta = event_type.endswith(".delta")
    if not is_delta and role in state.streamed_roles:
        return
    if is_delta:
        state.streamed_roles.add(role)
    db.append_transcript(state.call_id, role, text)
    await _send_browser(ws, {"type": "transcript", "role": role, "text": text})
    if role == "user" and state.bot_speaking:
        state.bot_speaking = False
        state.telemetry["interruptions"] += 1
        await _send_browser(ws, {"type": "clear"})
        await _emit_telemetry(ws, state, "interrupted", "user transcript")
    if role == "user":
        await _arm_user_turn(ws, state, text)


async def _handle_record_finding(
    ws: WebSocket, oai_ws: Any, state: _CallState, item: dict[str, Any]
) -> None:
    raw_args = item.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    domain = str(args.get("domain") or "")
    status = str(args.get("status") or "")
    detail = str(args.get("detail") or "")
    is_red_flag = _as_bool(args.get("is_red_flag", False))
    output = '{"status":"ok"}'
    if domain:
        db.record_finding_db(state.call_id, domain, status, detail, is_red_flag)
        await _send_browser(
            ws,
            {
                "type": "finding",
                "domain": domain,
                "status": status,
                "detail": detail,
                "is_red_flag": is_red_flag,
            },
        )
    else:
        output = '{"status":"error","message":"missing domain"}'
    call_id = str(item.get("call_id") or "")
    state.tool_seq += 1
    seq = state.tool_seq
    async with state.send_lock:
        await oai_ws.send(
            json.dumps(
                {
                    "type": "response.item.create",
                    "event_id": f"evt_tool_{seq}",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    },
                }
            )
        )
        await oai_ws.send(
            json.dumps({"type": "response.create", "event_id": f"evt_response_{seq}"})
        )
    state.telemetry["tool_calls"] += 1
    await _emit_telemetry(
        ws,
        state,
        "tool_call",
        domain or "record_finding",
        {"status": status, "is_red_flag": is_red_flag},
    )


async def _on_session_started(oai_ws: Any, state: _CallState) -> None:
    async with state.send_lock:
        await oai_ws.send(
            json.dumps(
                {
                    "type": "session.commentary.append",
                    "event_id": "evt_greet",
                    "delegation_id": None,
                    "content": GREETING,
                },
                ensure_ascii=False,
            )
        )
        queued = state.audio_queue
        state.audio_queue = []
        for pcm in queued:
            await oai_ws.send(
                json.dumps(
                    {
                        "type": "session.input_audio.append",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )
            )
        if not queued:
            await oai_ws.send(
                json.dumps(
                    {
                        "type": "session.input_audio.append",
                        "event_id": "evt_warmup",
                        "audio": _SILENCE_100MS,
                    }
                )
            )
        for text in state.pending_text:
            await oai_ws.send(
                json.dumps(
                    {
                        "type": "session.commentary.append",
                        "event_id": "evt_user_text",
                        "delegation_id": None,
                        "content": text,
                    },
                    ensure_ascii=False,
                )
            )
        state.pending_text = []
        state.session_ready = True


async def _openai_loop(ws: WebSocket, oai_ws: Any, state: _CallState) -> None:
    async for raw in oai_ws:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        if event_type == "session.started":
            if state.started:
                continue
            state.started = True
            if state.connect_t0:
                state.telemetry["connect_ms"] = int((time.perf_counter() - state.connect_t0) * 1000)
            await _send_browser(
                ws,
                {
                    "type": "session_started",
                    "call_id": state.call_id,
                    "engine": state.live_model,
                    "voice": state.voice,
                    "user_role": state.user_role,
                    "caller_name": state.caller_name,
                    "out_sample_rate": SAMPLE_RATE,
                },
            )
            await _emit_telemetry(
                ws,
                state,
                "session_started",
                state.live_model,
                {"voice": state.voice, "backend_model": state.backend_model},
            )
            await _on_session_started(oai_ws, state)
            continue
        if event_type == "session.output_audio.delta":
            delta = event.get("delta")
            if not isinstance(delta, str) or not delta:
                continue
            try:
                pcm = base64.b64decode(delta)
            except Exception:
                continue
            state.recorder.add_bot_pcm(pcm, SAMPLE_RATE)
            state.telemetry["audio_chunks_out"] += 1
            reply = state.awaiting_bot_reply
            first_utterance = state.telemetry["bot_turns"] == 0 or reply
            if reply:
                _note_latency(state)
                state.awaiting_bot_reply = False
                state.bot_replied_since_user = True
                state.saw_user_transcript = False
                state.nudges_for_current_turn = 0
            if first_utterance:
                state.telemetry["bot_turns"] += 1
            state.bot_speaking = True
            await _send_browser(
                ws,
                {"type": "audio", "data": delta, "sample_rate": SAMPLE_RATE},
            )
            if first_utterance:
                await _emit_telemetry(ws, state, "bot_first_audio", "reply" if reply else "audio")
            continue
        if event_type == "response.event":
            inner = event.get("event")
            if not isinstance(inner, dict):
                continue
            if inner.get("type") != "response.output_item.done":
                continue
            item = inner.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call" and item.get("name") == "record_finding":
                await _handle_record_finding(ws, oai_ws, state, item)
            continue
        if event_type == "error":
            err = event.get("error")
            message = "OpenAI live error"
            if isinstance(err, dict):
                message = str(err.get("message") or err.get("code") or message)
            elif isinstance(err, str) and err:
                message = err
            safe = _SECRET_RE.sub("***", message)[:500]
            _record_error(state, safe)
            await _send_browser(ws, {"type": "error", "message": safe})
            await _emit_telemetry(ws, state, "error", safe)
            continue
        if event_type == "session.closed":
            return
        if event_type in _CLEAR_EVENT_TYPES or "interrupt" in event_type:
            was_speaking = state.bot_speaking
            state.bot_speaking = False
            await _send_browser(ws, {"type": "clear"})
            if was_speaking or "interrupt" in event_type:
                state.telemetry["interruptions"] += 1
                await _emit_telemetry(ws, state, "interrupted", event_type)
            continue
        await _handle_transcript(ws, state, event)


async def _finalize(ws: WebSocket, oai_ws: Any, state: _CallState) -> None:
    if state.finalized:
        return
    state.finalized = True
    if oai_ws is not None:
        try:
            await _oai_send(oai_ws, state, {"type": "session.close", "event_id": "evt_close"})
        except Exception:
            pass
    try:
        filename, duration = state.recorder.save()
    except Exception:
        filename, duration = "", 0
    call = db.get_call(state.call_id) or {}
    transcript = call.get("transcript") or []
    findings = call.get("findings") or {}
    red_flags = call.get("red_flags") or []
    summary_doctor = ""
    summary_mother = ""
    try:
        result = await generate_call_summaries(transcript, findings, red_flags)
        if isinstance(result, dict):
            summary_doctor = str(result.get("summary_doctor") or "")
            summary_mother = str(result.get("summary_mother") or "")
            extra_flags = result.get("red_flags")
            if isinstance(extra_flags, list):
                red_flags = extra_flags
    except Exception as exc:
        safe = _public_error(exc)
        _record_error(state, safe)
        await _send_browser(ws, {"type": "error", "message": safe})
    try:
        db.complete_call(state.call_id, duration, filename, summary_doctor, summary_mother)
    except Exception as exc:
        safe = _public_error(exc)
        _record_error(state, safe)
        await _send_browser(ws, {"type": "error", "message": safe})
    await _emit_telemetry(ws, state, "hangup", "call ended")
    await _send_browser(
        ws,
        {
            "type": "call_completed",
            "call_id": state.call_id,
            "summary_doctor": summary_doctor,
            "summary_mother": summary_mother,
            "red_flags": red_flags,
        },
    )


async def handle_openai_ws(
    ws: WebSocket, caller_name: str = "", user_role: str = "user"
) -> None:
    await ws.accept()
    if not os.environ.get("OPENAI_API_KEY"):
        await _send_browser(ws, {"type": "error", "message": "OPENAI_API_KEY is not configured"})
        return

    settings = db.get_all_settings()
    live_model = _setting_text(settings, "openai_model", LIVE_MODEL)
    voice = _setting_text(settings, "openai_voice", LIVE_VOICE)
    backend_model = _setting_text(settings, "openai_backend_model", LIVE_BACKEND_MODEL)
    try:
        watchdog_timeout_sec = float(settings.get("watchdog_timeout_sec") or 4.0)
    except (TypeError, ValueError):
        watchdog_timeout_sec = 4.0
    if watchdog_timeout_sec <= 0:
        watchdog_timeout_sec = 4.0

    call_id = db.create_call(
        live_model,
        caller_name=caller_name or "",
        user_role=user_role or "user",
        model_used=live_model,
        voice_used=voice,
    )
    state = _CallState(
        call_id,
        audio_recorder.CallAudioRecorder(call_id),
        live_model=live_model,
        voice=voice,
        backend_model=backend_model,
        watchdog_timeout_sec=watchdog_timeout_sec,
        caller_name=caller_name or "",
        user_role=user_role or "user",
    )
    try:
        db.update_call_telemetry(call_id, _telemetry_snapshot(state))
    except Exception as exc:
        _record_error(state, _public_error(exc))
    state.connect_t0 = time.perf_counter()
    try:
        async with websockets.connect(
            LIVE_URL,
            additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            max_size=4_000_000,
        ) as oai_ws:
            await _oai_send(
                oai_ws, state, _session_start_payload(live_model, voice, backend_model)
            )
            browser_task = asyncio.create_task(_browser_loop(ws, oai_ws, state))
            openai_task = asyncio.create_task(_openai_loop(ws, oai_ws, state))
            watchdog_task = asyncio.create_task(_watchdog_loop(ws, oai_ws, state))
            done, pending = await asyncio.wait(
                {browser_task, openai_task, watchdog_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    safe = _public_error(exc)
                    _record_error(state, safe)
                    await _send_browser(ws, {"type": "error", "message": safe})
                    await _emit_telemetry(ws, state, "error", safe)
            await _finalize(ws, oai_ws, state)
    except Exception as exc:
        if state.connect_t0 and not state.telemetry["connect_ms"]:
            state.telemetry["connect_ms"] = int((time.perf_counter() - state.connect_t0) * 1000)
        safe = _public_error(exc)
        _record_error(state, safe)
        await _send_browser(ws, {"type": "error", "message": safe})
        await _emit_telemetry(ws, state, "error", safe)
        await _finalize(ws, None, state)
