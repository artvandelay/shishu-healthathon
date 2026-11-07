"""Standalone Gemini Live pipeline for one SHISHU pre-consultation call (/ws/gemini)."""

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

import audio_recorder
import db
import checklist
from prompts import (
    GEMINI_LIVE_SYSTEM_PROMPT,
    HINDI_MEDICAL_VOCAB,
    clean_assistant_caption,
    clean_user_caption,
)
from summarizer import default_mother_questions, generate_call_summaries

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_WATCHDOG_INTERVAL_SEC = 0.5
_GREETING = (
    "Greet the caller now in Hindi: Namaste, main Shishu hoon. Aapka naam bataiyega please?"
)
_WATCHDOG_NUDGE = (
    "SYSTEM CUE (from the app, not the mother): speak as SHISHU in Hindi only in your same warm female voice (Kore). "
    "If your previous question was not clearly answered, gently repeat or clarify that single question. "
    "NEVER say 'Nahi' or 'Haan' yourself, and never answer for the mother."
)
_MANUAL_NUDGE = _WATCHDOG_NUDGE
_FINDING_RESULT = "ok"

_RECORD_FINDING_DECL = {
    "name": "record_finding",
    "behavior": "NON_BLOCKING",
    "description": (
        "Save the mother's answer and mark checklist topics as covered. "
        "Invocation Condition: invoke once after every answer she gives, including nahi and nahi pata. "
        "The result note lists the next open topics. Never read the result aloud."
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
                    "family_history, or coverage"
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
            "topic_ids": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "TOPIC CHECKLIST ids this answer covered, or a group name to skip a whole group",
            },
        },
        "required": ["domain", "status", "detail", "is_red_flag", "topic_ids"],
    },
}

_END_CALL_DECL = {
    "name": "end_call",
    "behavior": "NON_BLOCKING",
    "description": (
        "Hang up the call. Invocation Condition: invoke only after every checklist topic is covered, "
        "open capture is done, and you have spoken the full closing. If the result is not_yet, keep asking the open topics."
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
    "reconnected",
    "silence_check_in",
    "auto_close",
    "error",
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
        self.last_user_text = ""
        self.time_cue_sent = False
        self.speech_end_waiting = False
        self.playback_ends_at = 0.0


class _SessionProxy:
    def __init__(self, session) -> None:
        self.current = session

    def __getattr__(self, name: str):
        return getattr(self.current, name)


def _session_settings(settings: dict) -> dict:
    start_low = settings.get("vad_start_sensitivity") == "LOW"
    end_low = settings.get("vad_end_sensitivity") == "LOW"
    return {
        "model": settings.get("gemini_model") or "gemini-3.1-flash-live-preview",
        "voice": "Kore",
        "prefix_padding_ms": int(float(settings.get("vad_prefix_padding_ms") or 300)),
        "silence_duration_ms": int(float(settings.get("vad_silence_ms") or 750)),
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
        "watchdog_timeout_sec": float(settings.get("watchdog_timeout_sec") or 8.0),
    }


def _live_config(settings: dict, resume_handle: str | None = None) -> types.LiveConnectConfig:
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
        input_audio_transcription=(
            types.AudioTranscriptionConfig(
                language_codes=["hi-IN"],
                custom_vocabulary=list(HINDI_MEDICAL_VOCAB),
            )
            if chosen["model"].startswith("gemini-3")
            else types.AudioTranscriptionConfig()
        ),
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
            trigger_tokens=10000,
            sliding_window=types.SlidingWindow(target_tokens=6000),
        ),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle or None),
        tools=[{"function_declarations": [_RECORD_FINDING_DECL, _END_CALL_DECL]}],
        thinking_config=(
            types.ThinkingConfig(thinking_level="MINIMAL")
            if chosen["model"].startswith("gemini-3.1")
            else None
        ),
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
        "coverage": [],
        "end_refusals": 0,
        "end_requested_at": 0.0,
        "force_close": False,
        "force_close_at": 0.0,
        "close_reason": "",
        "reconnects": 0,
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


def _pcm_rms(pcm: bytes) -> float:
    count = len(pcm) // 2
    if count <= 0:
        return 0.0
    total = 0
    for index in range(0, count * 2, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        total += sample * sample
    return (total / count) ** 0.5 / 32768.0


def _cancel_silence_close(state: _TurnState, telemetry: dict) -> None:
    """A silence close is provisional until she speaks again."""
    if telemetry.get("end_requested"):
        return
    if telemetry.get("close_reason") != "silence":
        return
    telemetry["force_close"] = False
    telemetry["force_close_at"] = 0.0
    telemetry["close_reason"] = ""
    state.checked_in = False


def _caller_heard(state: _TurnState, telemetry: dict) -> None:
    """She is speaking now, so this is not a silent line."""
    _cancel_silence_close(state, telemetry)
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
        telemetry["user_turns"] += 1
    _arm_waiting_reply(state, speech_end_waiting=False)
    _caller_heard(state, telemetry)
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


_MARRIAGE_DETAIL_RE = re.compile(
    r"\b(married|marriage|unmarried|shaadi)\b",
    re.I,
)
_CHILDREN_DETAIL_RE = re.compile(
    r"\b(children|child|kids|years?\s+old|months?\s+old|saal|mahine|beta|beti|son|daughter)\b",
    re.I,
)
_NO_MORE_USER_RE = re.compile(
    r"^\s*(?:नहीं|नही|ना|बस|बस इतना ही|और कुछ नहीं|कुछ नहीं|nahi|no|bas|nothing)(?:\s*[.,!?;:।].*|\s+.*)?$",
    re.I,
)
_NO_CHILD_RE = re.compile(
    r"\b(no\s+child|no\s+children|childless|0\s+children|zero\s+children|"
    r"unmarried|not\s+married|never\s+pregnant|to\s+be\s+married)\b"
    r"|बच्चे\s+नहीं|कोई\s+बच्चा\s+नहीं|शादी\s+होने\s+वाली",
    re.I,
)
_VALID_DOMAINS = {
    "profile_mother_name",
    "profile_child_name",
    "profile_child_age",
    "profile_child_gender",
    "chief_complaint",
    "birth_nicu",
    "milestones",
    "bleeding_joints",
    "metabolic_fasting",
    "infections",
    "pregnancy",
    "consanguinity",
    "family_infant_death",
    "family_history",
    "coverage",
}
_TOPIC_AS_DOMAIN = {
    "intro_name": "profile_mother_name",
    "intro_children": "profile_child_age",
    "profile_children": "profile_child_age",
    "profile_married": "coverage",
}
_DOMAIN_AS_TOPIC = {
    "profile_children": "intro_children",
    "profile_married": "intro_married",
}


def _normalize_finding(
    domain: str,
    detail: str,
    raw_topics: list,
    last_user_text: str = "",
) -> tuple[str, list]:
    """A checklist id in domain still counts as covered, without erasing valid clinical domains."""
    topics = list(raw_topics)
    if domain in _TOPIC_AS_DOMAIN and domain not in _VALID_DOMAINS:
        mapped_topic = _DOMAIN_AS_TOPIC.get(domain, domain)
        if mapped_topic in checklist._BY_ID and mapped_topic not in topics:
            topics.append(mapped_topic)
        domain = _TOPIC_AS_DOMAIN[domain]
    elif domain == "birth_nicu" and not topics:
        topics.append("birth_nicu")
    elif domain not in _VALID_DOMAINS and (domain in checklist._BY_ID or domain in checklist.GROUPS):
        if domain not in topics:
            topics.append(domain)
        domain = _TOPIC_AS_DOMAIN.get(domain, "coverage")
    marriage = "intro_married" in topics or bool(_MARRIAGE_DETAIL_RE.search(detail or ""))
    children_info = "intro_children" in topics or bool(_CHILDREN_DETAIL_RE.search(detail or ""))
    if domain == "profile_mother_name" and "intro_name" not in topics:
        if marriage:
            domain = "coverage"
            if "intro_married" not in topics:
                topics.append("intro_married")
        elif children_info:
            domain = "profile_child_age"
            if "intro_children" not in topics:
                topics.append("intro_children")
    allow_group_skip = bool(_NO_CHILD_RE.search(f"{detail} {last_user_text}"))
    if not allow_group_skip:
        topics = [t for t in topics if str(t).strip() not in checklist.GROUPS]
    return domain, topics


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
    elif role == "user":
        text = clean_user_caption(text)
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
    last_user_text: str = "",
) -> None:
    responses: list[types.FunctionResponse] = []
    logged: list[str] = []
    db_records: list[tuple[str, str, str, bool]] = []
    ws_messages: list[dict] = []
    for fc in calls:
        fc_id = getattr(fc, "id", None) or ""
        if fc_id and (fc_id in cancelled or fc_id in seen):
            continue
        if getattr(fc, "will_continue", False):
            continue
        if fc_id:
            seen.add(fc_id)
        name = getattr(fc, "name", None) or ""
        if name == "end_call":
            coverage = telemetry.setdefault("coverage", [])
            ok, note = checklist.can_end(coverage)
            refusals = int(telemetry.get("end_refusals") or 0)
            if not ok and refusals < 2 and not telemetry.get("force_close"):
                telemetry["end_refusals"] = refusals + 1
                body = {"result": "not_yet", "note": note}
                logged.append("end_call_refused")
            else:
                telemetry["end_requested"] = True
                telemetry["end_requested_at"] = time.time()
                ws_messages.append({"type": "ending"})
                body = {"result": "ok"}
                logged.append("end_call")
        elif name == "record_finding":
            args = _call_args(fc)
            domain = str(args.get("domain") or "")
            status = str(args.get("status") or "")
            detail = str(args.get("detail") or "")
            is_red_flag = _as_bool(args.get("is_red_flag", False))
            raw_topics = list(args.get("topic_ids") or [])
            if isinstance(args.get("topic_ids"), str):
                raw_topics = [args["topic_ids"]]
            domain, raw_topics = _normalize_finding(
                domain, detail, raw_topics, last_user_text
            )
            coverage = telemetry.setdefault("coverage", [])
            name_check = domain == "profile_mother_name" or "intro_name" in raw_topics
            name_invalid = name_check and (
                status == "negative"
                or not checklist.is_valid_name_detail(detail, last_user_text)
            )
            if name_invalid:
                raw_topics = [t for t in raw_topics if t != "intro_name"]
            open_capture_hold = False
            if (
                "close_anything_else" in raw_topics
                and not telemetry.get("open_capture_asked_more")
                and domain == "chief_complaint"
                and status in ("positive", "info")
                and (last_user_text or "").strip()
                and not _NO_MORE_USER_RE.match((last_user_text or "").strip())
            ):
                telemetry["open_capture_asked_more"] = True
                raw_topics = [t for t in raw_topics if t != "close_anything_else"]
                open_capture_hold = True
            checklist.mark(coverage, raw_topics)
            if domain and domain != "coverage" and not (name_invalid and domain == "profile_mother_name"):
                db_records.append((domain, status, detail, is_red_flag))
                ws_messages.append(
                    {
                        "type": "finding",
                        "domain": domain,
                        "status": status,
                        "detail": detail,
                        "is_red_flag": is_red_flag,
                    }
                )
            label = checklist.progress_label(coverage)
            telemetry["progress_group"] = label
            ws_messages.append({"type": "progress", "group": label})
            if name_invalid:
                note = checklist.NAME_MISSING_NOTE
            elif open_capture_hold:
                note = (
                    "Saved her concern for the doctor. Acknowledge it warmly in Hindi and ask ONLY "
                    "'Aur kuch batana chahengi?' then STOP speaking and wait for her reply. "
                    "Do NOT speak the closing yet. Keep your same warm female voice (Kore)."
                )
            else:
                note = checklist.tool_note(coverage)
            body = {"result": "ok", "note": note}
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
    send_failed = False
    if responses:
        try:
            async with send_lock:
                await session.send_tool_response(function_responses=responses)
        except Exception:
            send_failed = True
    for domain, status, detail, is_red_flag in db_records:
        db.record_finding_db(call_id, domain, status, detail, is_red_flag)
    for payload in ws_messages:
        await _send_json(ws, ws_lock, payload)
    for detail in logged:
        telemetry["tool_calls"] += 1
        await _emit(ws, ws_lock, call_id, telemetry, "tool_call", detail)
    if send_failed:
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
    state.speech_end_waiting = False
    state.bot_replied_since_user = True
    state.bot_speaking = True
    dur = len(pcm) / max(1, 2 * rate)
    state.playback_ends_at = max(time.time(), state.playback_ends_at) + dur
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
        state.playback_ends_at = 0.0
        _caller_heard(state, telemetry)
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
            state.last_user_text = user_text
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
            state.last_user_text,
        )
        if not telemetry.get("end_requested"):
            _arm_waiting_reply(state, speech_end_waiting=False)
            state.quiet_from = 0.0

    if server_content is not None and server_content.turn_complete:
        state.bot_speaking = False
        if not state.awaiting_bot_reply:
            state.quiet_from = max(time.time(), state.playback_ends_at)
        if not calls or telemetry.get("end_requested"):
            await _emit(ws, ws_lock, call_id, telemetry, "turn_complete", "")

    update = getattr(response, "session_resumption_update", None)
    if update is not None and getattr(update, "resumable", False) and getattr(update, "new_handle", None):
        state.resume_handle = str(update.new_handle)
    if getattr(response, "go_away", None) is not None:
        if not state.resume_handle:
            return False
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
    reconnect,
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
        reconnects += 1
        ok = reconnects <= 5 and await reconnect(state.resume_handle)
        if not ok:
            stop.set()
            await _report_error(
                ws, ws_lock, call_id, telemetry, "Gemini Live could not reconnect"
            )
            return
        telemetry["reconnects"] = reconnects
        state.bot_speaking = False
        state.awaiting_bot_reply = False
        state.speech_end_waiting = False
        state.playback_ends_at = 0.0
        state.quiet_from = time.time()
        seen.clear()
        cancelled.clear()
        await _emit(ws, ws_lock, call_id, telemetry, "reconnected", str(reconnects))


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
            now = time.time()
            if now - state.call_started_ts >= checklist.HARD_LIMIT_SEC:
                telemetry["close_reason"] = telemetry.get("close_reason") or "time_limit"
                stop.set()
                return
            if telemetry.get("end_requested"):
                if time.time() - float(telemetry.get("end_requested_at") or 0) > 30:
                    stop.set()
                    return
                continue
            if telemetry.get("force_close"):
                if now - float(telemetry.get("force_close_at") or 0) > checklist.FORCE_END_GRACE_SEC:
                    telemetry["end_requested"] = True
                    telemetry["end_requested_at"] = now
                    await _send_json(ws, ws_lock, {"type": "ending"})
                continue
            cue = ""
            reason = ""
            if now - state.call_started_ts >= checklist.SOFT_LIMIT_SEC and not state.time_cue_sent:
                cue, reason = checklist.TIME_CLOSE_CUE, "time_limit"
                state.time_cue_sent = True
            elif state.quiet_from > 0 and not state.bot_speaking and not state.awaiting_bot_reply:
                quiet = now - state.quiet_from
                if quiet >= checklist.SILENCE_CLOSE_SEC:
                    cue, reason = checklist.SILENCE_CLOSE_CUE, "silence"
                elif quiet >= checklist.CHECK_IN_SEC and not state.checked_in:
                    cue = checklist.CHECK_IN_CUE
            if cue:
                if reason:
                    telemetry["force_close"] = True
                    telemetry["force_close_at"] = now
                    telemetry["close_reason"] = reason
                else:
                    state.checked_in = True
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
            if state.bot_replied_since_user or state.last_user_turn_ts <= 0:
                continue
            effective_timeout = (
                min(watchdog_timeout_sec, 6.0)
                if state.speech_end_waiting
                else watchdog_timeout_sec
            )
            if state.nudges_for_current_turn >= 2:
                if time.time() - state.last_user_turn_ts >= effective_timeout:
                    state.awaiting_bot_reply = False
                    state.speech_end_waiting = False
                    if state.quiet_from <= 0:
                        state.quiet_from = time.time()
                continue
            if time.time() - state.last_user_turn_ts < effective_timeout:
                continue
            state.nudges_for_current_turn += 1
            telemetry["watchdog_nudges"] += 1
            state.last_user_turn_ts = time.time()
            ok = await _send_realtime_safe(
                session, send_lock, state, stop, text=_WATCHDOG_NUDGE
            )
            if not ok:
                await _report_error(
                    ws,
                    ws_lock,
                    call_id,
                    telemetry,
                    "Watchdog nudge failed",
                )
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
                if _pcm_rms(pcm) >= 0.02:
                    _caller_heard(state, telemetry)
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
                _cancel_silence_close(state, telemetry)
                now_ts = time.time()
                if not state.bot_speaking and now_ts >= state.playback_ends_at - 0.1:
                    if not state.awaiting_bot_reply:
                        _arm_waiting_reply(
                            state, speech_end_waiting=True, now_ts=now_ts
                        )
                    state.quiet_from = max(now_ts, state.playback_ends_at)
                try:
                    db.append_call_event(call_id, "speech_end", "")
                except Exception:
                    pass
            elif kind == "nudge":
                if state.awaiting_bot_reply:
                    state.last_user_turn_ts = time.time()
                ok = await _send_realtime_safe(
                    session, send_lock, state, stop, text=_MANUAL_NUDGE
                )
                if not ok:
                    stop.set()
                    await _report_error(
                        ws,
                        ws_lock,
                        call_id,
                        telemetry,
                        "Could not nudge Gemini",
                    )
                    return
                if state.reconnect_requested:
                    continue
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
                state.last_user_text = text
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
    merged_findings = findings
    try:
        result = await generate_call_summaries(transcript, findings, red_flags)
        if isinstance(result, dict):
            summary_doctor = str(result.get("summary_doctor") or "")
            summary_mother = str(result.get("summary_mother") or "")
            if isinstance(result.get("findings"), dict):
                merged_findings = result["findings"]
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
    if not summary_mother.strip():
        summary_mother = default_mother_questions(merged_findings, red_flags)

    if telemetry is not None:
        doctor_note, mother_note = checklist.incomplete_notes(
            telemetry.get("coverage") or [], str(telemetry.get("close_reason") or "")
        )
        if doctor_note:
            summary_doctor = doctor_note + "\n\n" + summary_doctor
        if mother_note:
            summary_mother = (summary_mother + "\n\n" + mother_note).strip()
    db.complete_call(
        call_id,
        int(duration),
        filename,
        summary_doctor,
        summary_mother,
        findings=merged_findings,
        red_flags=red_flags,
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
        session = _SessionProxy(session)

        async def _reconnect(handle: str) -> bool:
            nonlocal session_ctx
            new_ctx = None
            new_session = None
            for attempt in range(3):
                try:
                    new_ctx, new_session, _ = await _connect_live(
                        client, model, _live_config(settings, resume_handle=handle)
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
                _reconnect,
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
