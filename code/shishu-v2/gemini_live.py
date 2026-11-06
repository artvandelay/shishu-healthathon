"""Standalone Gemini Live pipeline for one SHISHU pre-consultation call (/ws/gemini)."""

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

import audio_recorder
import db
from prompts import GEMINI_LIVE_SYSTEM_PROMPT, clean_assistant_caption
from summarizer import generate_call_summaries

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_WATCHDOG_INTERVAL_SEC = 0.5
_GREETING = (
    "Greet the caller now in Hindi: Namaste, main Shishu hoon. Aapka naam bataiyega please?"
)
_WATCHDOG_NUDGE = (
    "Speak as SHISHU in Hindi only. If your previous question was not clearly answered, "
    "gently repeat or clarify that same question. Never answer for the mother."
)
_MANUAL_NUDGE = (
    "Speak as SHISHU in Hindi only. If your previous question was not clearly answered, "
    "gently repeat or clarify that same question. Never answer for the mother."
)
_FINDING_RESULT = "ok"

_RECORD_FINDING_DECL = {
    "name": "record_finding",
    "behavior": "NON_BLOCKING",
    "description": (
        "Record a clinical intake finding, patient profile field, or red flag "
        "while continuing the Hindi conversation."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "domain": {
                "type": "STRING",
                "description": (
                    "e.g. profile_mother_name, profile_child_name, "
                    "profile_child_age, profile_child_gender, chief_complaint, "
                    "birth_nicu, milestones, bleeding_joints, metabolic_fasting, "
                    "infections, pregnancy, consanguinity, family_infant_death, "
                    "family_history"
                ),
            },
            "status": {
                "type": "STRING",
                "description": "positive, negative, or info",
            },
            "detail": {
                "type": "STRING",
                "description": "Concise clinical detail in English",
            },
            "is_red_flag": {
                "type": "BOOLEAN",
                "description": (
                    "True if this is a positive clinical red flag or gate finding"
                ),
            },
        },
        "required": ["domain", "status", "detail", "is_red_flag"],
    },
}

_SET_PROGRESS_DECL = {
    "name": "set_progress",
    "behavior": "NON_BLOCKING",
    "description": "Mark which question group you are about to ask. Call this before the first question of a group.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "group": {
                "type": "STRING",
                "description": "intro, birth, body, illness, pregnancy, family, or close",
            }
        },
        "required": ["group"],
    },
}

_END_CALL_DECL = {
    "name": "end_call",
    "behavior": "NON_BLOCKING",
    "description": "Hang up after the approved closing sentence has been spoken. Do not call this before the closing.",
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
    "error",
}


class _TurnState:
    def __init__(self) -> None:
        self.awaiting_bot_reply = False
        self.last_user_turn_ts = 0.0
        self.bot_replied_since_user = False
        self.bot_speaking = False
        self.nudges_for_current_turn = 0


def _session_settings(settings: dict) -> dict:
    start_low = settings.get("vad_start_sensitivity") == "LOW"
    end_low = settings.get("vad_end_sensitivity") == "LOW"
    return {
        "model": settings.get("gemini_model") or "gemini-3.8-live",
        "voice": settings.get("gemini_voice") or "Kore",
        "prefix_padding_ms": int(float(settings.get("vad_prefix_padding_ms") or 300)),
        "silence_duration_ms": int(float(settings.get("vad_silence_ms") or 650)),
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
        "watchdog_timeout_sec": float(settings.get("watchdog_timeout_sec") or 4.0),
    }


def _live_config(settings: dict) -> types.LiveConnectConfig:
    chosen = _session_settings(settings)
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=chosen["voice"]
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=GEMINI_LIVE_SYSTEM_PROMPT)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
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
        tools=[{"function_declarations": [_RECORD_FINDING_DECL, _SET_PROGRESS_DECL, _END_CALL_DECL]}],
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
        "errors": [],
    }


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


def _rate_from_mime(mime: str, default: int) -> int:
    for piece in (mime or "").split(";"):
        piece = piece.strip().lower()
        if not piece.startswith("rate="):
            continue
        try:
            rate = int(piece.split("=", 1)[1])
        except ValueError:
            return default
        if rate > 0:
            return rate
    return default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _call_args(fc: types.FunctionCall) -> dict:
    args = fc.args or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    if not isinstance(args, dict):
        return {}
    return args


def _pcm_bytes(raw: object) -> bytes | None:
    if isinstance(raw, str):
        try:
            raw = base64.b64decode(raw)
        except Exception:
            return None
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    data = bytes(raw)
    if len(data) % 2 == 1:
        data = data[:-1]
    return data or None


def _audio_from_part(part: types.Part) -> tuple[bytes, int] | None:
    blob = part.inline_data
    if blob is None or blob.data is None:
        return None
    mime = blob.mime_type or ""
    if mime and not mime.lower().startswith("audio/"):
        return None
    pcm = _pcm_bytes(blob.data)
    if not pcm:
        return None
    return pcm, _rate_from_mime(mime, _OUTPUT_RATE)


def _arm_user_turn(state: _TurnState, telemetry: dict) -> None:
    if not state.awaiting_bot_reply:
        telemetry["user_turns"] += 1
        state.nudges_for_current_turn = 0
    state.awaiting_bot_reply = True
    state.last_user_turn_ts = time.time()
    state.bot_replied_since_user = False


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
        evt = db.append_call_event(call_id, event_type, detail, meta)
    except Exception:
        evt = {
            "t": round(time.time(), 3),
            "type": event_type,
            "detail": detail or "",
            "meta": meta or {},
        }
    try:
        db.update_call_telemetry(call_id, telemetry)
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
) -> str:
    if transcription is None:
        return ""
    text = transcription.text or ""
    if role == "assistant":
        text = clean_assistant_caption(text)
    if not text:
        return ""
    db.append_transcript(call_id, role, text)
    await _send_json(ws, lock, {"type": "transcript", "role": role, "text": text})
    return text


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
) -> None:
    responses: list[types.FunctionResponse] = []
    logged: list[str] = []
    for fc in calls:
        fc_id = getattr(fc, "id", None) or ""
        if fc_id and (fc_id in cancelled or fc_id in seen):
            continue
        if getattr(fc, "will_continue", False):
            continue
        if fc_id:
            seen.add(fc_id)
        name = getattr(fc, "name", None) or ""
        if name == "set_progress":
            group = str(_call_args(fc).get("group") or "")
            telemetry["progress_group"] = group
            await _send_json(
                ws,
                ws_lock,
                {"type": "progress", "group": group},
            )
            body = {"result": "ok"}
            logged.append("progress:" + group)
        elif name == "end_call":
            telemetry["end_requested"] = True
            await _send_json(ws, ws_lock, {"type": "ending"})
            body = {"result": "ok"}
            logged.append("end_call")
        elif name == "record_finding":
            args = _call_args(fc)
            domain = str(args.get("domain") or "")
            status = str(args.get("status") or "")
            detail = str(args.get("detail") or "")
            is_red_flag = _as_bool(args.get("is_red_flag", False))
            db.record_finding_db(call_id, domain, status, detail, is_red_flag)
            await _send_json(
                ws,
                ws_lock,
                {
                    "type": "finding",
                    "domain": domain,
                    "status": status,
                    "detail": detail,
                    "is_red_flag": is_red_flag,
                },
            )
            body = {"result": _FINDING_RESULT}
            logged.append(domain or "record_finding")
        else:
            body = {"result": "ignored"}
            logged.append(name or "ignored")
        if not fc_id:
            continue
        responses.append(
            types.FunctionResponse(
                id=fc_id,
                name=name or "record_finding",
                response=body,
                scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
            )
        )
    for detail in logged:
        telemetry["tool_calls"] += 1
        await _emit(ws, ws_lock, call_id, telemetry, "tool_call", detail)
    if not responses:
        return
    try:
        async with send_lock:
            await session.send_tool_response(function_responses=responses)
    except Exception:
        await _report_error(
            ws,
            ws_lock,
            call_id,
            telemetry,
            "Could not send the tool response to Gemini",
        )


async def _on_bot_audio(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
    recorder: audio_recorder.CallAudioRecorder,
    pcm: bytes,
    rate: int,
) -> None:
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
    state.awaiting_bot_reply = False
    state.bot_replied_since_user = True
    state.bot_speaking = True
    recorder.add_bot_pcm(pcm, rate)
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
        meta = {"latency_ms": latency_ms} if latency_ms is not None else None
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
) -> bool:
    """Handle one Live server message. Return False when the session should stop."""
    cancellation = getattr(response, "tool_call_cancellation", None)
    if cancellation is not None and cancellation.ids:
        cancelled.update(item for item in cancellation.ids if item)

    server_content = getattr(response, "server_content", None)
    if server_content is not None and server_content.interrupted:
        state.bot_speaking = False
        telemetry["interruptions"] += 1
        await _send_json(ws, ws_lock, {"type": "clear"})
        await _emit(ws, ws_lock, call_id, telemetry, "interrupted", "interrupted")

    if server_content is not None:
        user_text = await _forward_transcript(
            ws,
            ws_lock,
            call_id,
            "user",
            server_content.input_transcription,
        )
        if user_text:
            _arm_user_turn(state, telemetry)
            await _emit(
                ws, ws_lock, call_id, telemetry, "user_transcript", user_text
            )
        await _forward_transcript(
            ws,
            ws_lock,
            call_id,
            "assistant",
            server_content.output_transcription,
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
            )
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            part_calls.append(function_call)

    tool_call = getattr(response, "tool_call", None)
    calls = list(part_calls)
    if tool_call is not None and tool_call.function_calls:
        calls.extend(tool_call.function_calls)
    if calls:
        await _answer_function_calls(
            session,
            send_lock,
            ws,
            ws_lock,
            call_id,
            telemetry,
            calls,
            seen,
            cancelled,
        )

    if server_content is not None and server_content.turn_complete:
        state.bot_speaking = False
        await _emit(ws, ws_lock, call_id, telemetry, "turn_complete", "")
        if telemetry.get("end_requested"):
            return False

    if getattr(response, "go_away", None) is not None:
        return False
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
) -> None:
    seen: set[str] = set()
    cancelled: set[str] = set()
    try:
        while not stop.is_set():
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
                )
                if not keep_going:
                    stop.set()
                    return
    except asyncio.CancelledError:
        raise
    except Exception:
        stop.set()
        await _report_error(
            ws, ws_lock, call_id, telemetry, "Gemini Live session ended"
        )


async def _watchdog_loop(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    stop: asyncio.Event,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
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
            if not state.awaiting_bot_reply or state.bot_speaking:
                continue
            if state.bot_replied_since_user or state.last_user_turn_ts <= 0:
                continue
            if state.nudges_for_current_turn >= 2:
                continue
            if time.time() - state.last_user_turn_ts < watchdog_timeout_sec:
                continue
            state.nudges_for_current_turn += 1
            telemetry["watchdog_nudges"] += 1
            state.last_user_turn_ts = time.time()
            try:
                async with send_lock:
                    await session.send_realtime_input(text=_WATCHDOG_NUDGE)
            except Exception:
                await _report_error(
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    "Watchdog nudge failed",
                )
                stop.set()
                return
            await _emit(
                ws,
                ws_lock,
                call_id,
                telemetry,
                "watchdog_nudge",
                _WATCHDOG_NUDGE,
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
            if kind == "audio":
                pcm = _pcm_bytes(payload.get("data") or "")
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
                try:
                    async with send_lock:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm,
                                mime_type=f"audio/pcm;rate={rate}",
                            )
                        )
                except Exception:
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
                _arm_user_turn(state, telemetry)
                try:
                    db.append_call_event(call_id, "speech_end", "")
                    db.update_call_telemetry(call_id, telemetry)
                except Exception:
                    pass
            elif kind == "nudge":
                if state.awaiting_bot_reply:
                    state.last_user_turn_ts = time.time()
                try:
                    async with send_lock:
                        await session.send_realtime_input(text=_MANUAL_NUDGE)
                except Exception:
                    stop.set()
                    await _report_error(
                        ws,
                        ws_lock,
                        call_id,
                        telemetry,
                        "Could not nudge Gemini",
                    )
                    return
                await _emit(
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    "manual_nudge",
                    _MANUAL_NUDGE,
                )
            elif kind == "text":
                text = str(payload.get("text") or "").strip()
                if not text:
                    continue
                db.append_transcript(call_id, "user", text)
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "transcript", "role": "user", "text": text},
                )
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
            elif kind == "hangup":
                stop.set()
                return
    except asyncio.CancelledError:
        raise


async def _finalize(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict | None,
    recorder: audio_recorder.CallAudioRecorder,
) -> None:
    filename = ""
    duration = 0
    try:
        filename, duration = recorder.save()
    except Exception:
        filename, duration = "", 0

    if telemetry is not None:
        _refresh_latency(telemetry)
        try:
            db.update_call_telemetry(call_id, telemetry)
        except Exception:
            pass

    call = db.get_call(call_id) or {}
    transcript = call.get("transcript") or []
    findings = call.get("findings") or {}
    red_flags = list(call.get("red_flags") or [])
    summary_doctor = ""
    summary_mother = ""
    try:
        result = await generate_call_summaries(transcript, findings, red_flags)
        if isinstance(result, dict):
            summary_doctor = str(result.get("summary_doctor") or "")
            summary_mother = str(result.get("summary_mother") or "")
            extra_flags = result.get("red_flags")
            if isinstance(extra_flags, list):
                red_flags = [str(item) for item in extra_flags]
    except Exception:
        await _report_error(
            ws,
            ws_lock,
            call_id,
            telemetry,
            "Post-call summary failed",
        )

    db.complete_call(
        call_id,
        int(duration),
        filename,
        summary_doctor,
        summary_mother,
    )
    await _send_json(
        ws,
        ws_lock,
        {
            "type": "call_completed",
            "call_id": call_id,
            "summary_doctor": summary_doctor,
            "summary_mother": summary_mother,
            "red_flags": red_flags,
        },
    )


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
    ws: WebSocket, caller_name: str = "", user_role: str = "user"
) -> None:
    load_dotenv(_ENV_PATH)
    if ws.client_state == WebSocketState.CONNECTING:
        await ws.accept()

    caller_name = caller_name or ""
    user_role = user_role or "user"
    ws_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    stop = asyncio.Event()
    state = _TurnState()
    call_id: int | None = None
    telemetry: dict | None = None
    recorder: audio_recorder.CallAudioRecorder | None = None
    session_ctx = None
    gemini_task: asyncio.Task | None = None
    browser_task: asyncio.Task | None = None
    watchdog_task: asyncio.Task | None = None
    finalized = False

    try:
        if not os.environ.get("GEMINI_API_KEY"):
            await _send_json(
                ws,
                ws_lock,
                {"type": "error", "message": "GEMINI_API_KEY is not set"},
            )
            return

        settings = db.get_all_settings()
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

        call_id = db.create_call(
            model,
            caller_name=caller_name,
            user_role=user_role,
            model_used=model,
            voice_used=voice,
        )
        telemetry = _new_telemetry(
            model,
            voice,
            chosen["silence_duration_ms"],
            chosen["prefix_padding_ms"],
            connect_ms,
        )
        recorder = audio_recorder.CallAudioRecorder(call_id)
        await _send_json(
            ws,
            ws_lock,
            {
                "type": "session_started",
                "call_id": call_id,
                "engine": model,
                "voice": voice,
                "user_role": user_role,
                "caller_name": caller_name,
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
                chosen["watchdog_timeout_sec"],
            )
        )
        try:
            async with send_lock:
                await session.send_realtime_input(text=_GREETING)
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
        await _stop_tasks(gemini_task, browser_task, watchdog_task)
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if call_id is not None and recorder is not None and not finalized:
            finalized = True
            try:
                await _finalize(ws, ws_lock, call_id, telemetry, recorder)
            except Exception:
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "error", "message": "Could not finish the call"},
                )
        await _close_ws(ws)
