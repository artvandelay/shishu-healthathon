"""Standalone Gemini Live pipeline for one SHISHU pre-consultation call (/ws/gemini)."""

import asyncio
import base64
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

import audio_recorder
import db
from call_cursor import apply_profile, apply_turn, new_cursor_state, prompt_directive
from prompts import (
    GEMINI_LIVE_SYSTEM_PROMPT,
    HINDI_MEDICAL_VOCAB,
    clean_assistant_caption,
    clean_user_caption,
    leaked_caption,
)

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_WATCHDOG_INTERVAL_SEC = 0.5
_CHECK_IN_SEC = 15
_SILENCE_CLOSE_SEC = 30
_SOFT_LIMIT_SEC = 19 * 60
_HARD_LIMIT_SEC = 20 * 60
_CLOSING_AUDIO_GRACE_SEC = 0.8
_CLOSING_MAX_WAIT_SEC = 25
_PROFILE_QIDS = frozenset({"intro_name", "intro_married", "intro_children", "ever_pregnant"})
_WATCHDOG_NUDGE = (
    "SYSTEM CUE (from the app, not the mother): she is waiting. "
    "Speak only the Hindi question you were last given, in your same warm female voice, then wait. "
    "Never answer for her."
)
_MANUAL_NUDGE = _WATCHDOG_NUDGE

_SAVE_PROFILE_DECL = {
    "name": "save_profile",
    "description": (
        "Save the profile fact she just gave: name, marital status, children, or ever pregnant. "
        "Call this only for those questions, before you speak again. Never read this tool aloud."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "caller_name": {
                "type": "STRING",
                "description": "Her name only. Not the words मेरा नाम.",
            },
            "marital_status": {
                "type": "STRING",
                "description": "married, about_to_marry, or unmarried.",
            },
            "child_count": {
                "type": "INTEGER",
                "description": "How many children she said. 0 if she has none.",
            },
            "children": {
                "type": "ARRAY",
                "description": "One object per child once she has given each age. Empty when child_count is 0.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "age_years": {"type": "NUMBER", "description": "Age in years. Six months is 0.5."},
                        "sex": {"type": "STRING", "description": "male, female, or unknown."},
                        "name": {"type": "STRING"},
                        "age_label_hi": {"type": "STRING"},
                    },
                    "required": ["age_years"],
                },
            },
            "ever_pregnant": {"type": "BOOLEAN"},
        },
    },
}

_RECORD_TURN_DECL = {
    "name": "record_turn",
    "description": (
        "Record her answer to the current screening question, then speak only the say field. "
        "Do not use this for her name, marriage, children, or ever pregnant."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "outcome": {
                "type": "STRING",
                "description": "clear, concern, unknown, clarify, emergency, or caller_stop.",
            },
            "child_ref": {
                "type": "STRING",
                "description": "child_1, oldest, youngest, both, or empty when one child or not a child question.",
            },
            "note": {
                "type": "STRING",
                "description": "Short phrase with the detail she gave, including a follow-up. Empty if she only said नहीं.",
            },
        },
        "required": ["outcome"],
    },
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
        self.last_user_text = ""
        self.last_bot_text = ""
        self.bot_text_since_user = ""
        self.time_cue_sent = False
        self.speech_end_waiting = False
        self.playback_ends_at = 0.0


class _Cursor:
    def __init__(self) -> None:
        self.state = new_cursor_state()


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
            sliding_window=types.SlidingWindow()
        ),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle or None),
        tools=[{"function_declarations": [_SAVE_PROFILE_DECL, _RECORD_TURN_DECL]}],
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
    cursor: _Cursor,
    telemetry: dict,
) -> str:
    if transcription is None:
        return ""
    raw = transcription.text or ""
    if role == "assistant":
        text = clean_assistant_caption(raw)
        if leaked_caption(raw, text):
            telemetry["prompt_leaks"] = int(telemetry.get("prompt_leaks") or 0) + 1
    elif role == "user":
        text = clean_user_caption(raw)
    else:
        text = raw
    if not text:
        return ""
    qid = str(cursor.state.get("active_qid") or "")
    db.append_transcript(call_id, role, text, qid=qid)
    await _send_json(ws, lock, {"type": "transcript", "role": role, "text": text})
    return text


def _report_snapshot(call_id: int) -> dict:
    """Immutable transcript and cursor for synthesis. No later caller turn is included."""
    call = db.get_call(call_id) or {}
    cursor = call.get("cursor_state") or {}
    snapshot = {
        "id": int(call_id),
        "transcript": call.get("transcript") or [],
        "findings": call.get("findings") or {},
        "telemetry": call.get("telemetry") or {},
        "status": call.get("status") or "",
        "started_at": call.get("started_at") or "",
        "duration_sec": call.get("duration_sec") or 0,
        "cursor_state": cursor,
        "cursor_json": cursor,
        "share_token": call.get("share_token") or "",
    }
    return json.loads(json.dumps(snapshot))


def _accept_end(telemetry: dict) -> bool:
    """Mark the call closing. True only the first time end_call is accepted."""
    first = not telemetry.get("end_requested")
    telemetry["end_requested"] = True
    telemetry["closing"] = True
    telemetry["end_requested_at"] = time.time()
    return first


async def _schedule_end_report(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    reason: str,
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None,
) -> None:
    if telemetry.get("extraction_scheduled"):
        return
    telemetry["extraction_scheduled"] = True
    await _emit(ws, ws_lock, call_id, telemetry, "end_accepted", reason, {"reason": reason})
    if on_end_accepted is None:
        return
    try:
        scheduled = on_end_accepted(int(call_id), _report_snapshot(call_id))
        if inspect.isawaitable(scheduled):
            await scheduled
    except Exception:
        telemetry["extraction_schedule_failed"] = True


def _just_name(value: str) -> str:
    cleaned = " ".join((value or "").split())
    folded = cleaned.casefold()
    for prefix in (
        "मेरा नाम है",
        "मेरा नाम",
        "जी मेरा नाम है",
        "नाम है",
        "my name is",
        "mera naam hai",
        "mera naam",
    ):
        if folded.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" .,।")
            break
    return cleaned.strip()


def _marital(value: object) -> str:
    raw = str(value or "").strip()
    text = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "married": "married",
        "yes": "married",
        "shaadi": "married",
        "unmarried": "unmarried",
        "single": "unmarried",
        "about_to_marry": "about_to_marry",
        "engaged": "about_to_marry",
        "about_to_be_married": "about_to_marry",
    }
    if text in aliases:
        return aliases[text]
    if "नहीं" in raw and "शादी" in raw:
        return "unmarried"
    if "होने वाली" in raw:
        return "about_to_marry"
    if "शादी" in raw:
        return "married"
    return ""


def _as_children(value: object) -> list | None:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    kids = []
    for item in value:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if not isinstance(item, dict):
            continue
        try:
            age = float(item.get("age_years"))
        except (TypeError, ValueError):
            continue
        sex = str(item.get("sex") or "unknown").strip().lower()
        if sex in {"male", "m", "boy", "son", "beta", "बेटा"}:
            sex = "male"
        elif sex in {"female", "f", "girl", "daughter", "beti", "बेटी"}:
            sex = "female"
        elif sex not in {"male", "female", "unknown"}:
            sex = "unknown"
        kids.append({
            "age_years": age,
            "sex": sex,
            "name": item.get("name"),
            "age_label_hi": item.get("age_label_hi"),
        })
    return kids


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "haan"}:
        return True
    if text in {"false", "no", "0", "nahi"}:
        return False
    return None


def _say_hindi(state: dict) -> str:
    if state.get("phase") != "asking":
        return _closing_hi(state)
    text = prompt_directive(state)
    marker = "then wait: "
    if marker in text:
        text = text.split(marker, 1)[1]
    if " Say " in text:
        text = text.split(" Say ", 1)[0]
    return text.strip()


def _closing_hi(state: dict) -> str:
    name = str((state.get("profile") or {}).get("caller_name") or "").strip()
    who = f"{name} जी, " if name else ""
    reason = state.get("close_reason") or "complete"
    if reason == "emergency":
        return (
            f"{who}यह तुरंत दिखाने वाली बात है। अभी बच्चे को नज़दीकी अस्पताल ले जाइए, "
            "या 108 पर कॉल कीजिए। अब तक की जानकारी मैंने डॉक्टर के लिए नोट कर ली है। नमस्ते।"
        )
    if reason == "caller_stop":
        return (
            f"कोई बात नहीं, {who}अब तक की जानकारी मैंने नोट कर ली है। "
            "जब समय मिले, दोबारा कॉल करके बाकी सवाल पूरे कर लीजिए। नमस्ते।"
        )
    if reason == "silence":
        return (
            "लगता है आप अभी बात नहीं कर पा रही हैं। कोई बात नहीं। "
            "अब तक आपने जो समय दिया, उसके लिए बहुत धन्यवाद। अभी कुछ सवाल बाकी हैं, "
            "इसलिए यह जानकारी अधूरी है। जब समय मिले, दोबारा कॉल करके इसे पूरा कर लीजिए। नमस्ते।"
        )
    if reason == "time_limit":
        return (
            f"{who}आज हमारी बात काफी लंबी हो गई। इतना समय देने के लिए बहुत धन्यवाद। "
            "अभी कुछ सवाल बाकी हैं, इसलिए यह जानकारी अधूरी है। "
            "जब समय मिले, दोबारा कॉल करके इसे पूरा कर लीजिए। नमस्ते।"
        )
    concerns = any(
        isinstance(entry, dict) and entry.get("state") == "concern"
        for entry in (state.get("answers") or {}).values()
    )
    if concerns:
        return (
            f"{who}इतना ध्यान से बताने के लिए बहुत धन्यवाद। आपने जो बताया, "
            "उसमें से कुछ बातों की जाँच डॉक्टर या स्वास्थ्य कार्यकर्ता से करवाना अच्छा रहेगा। "
            "आपकी रिपोर्ट तैयार हो रही है। यह रिपोर्ट आप अपनी ASHA, ANM या डॉक्टर को दिखा सकती हैं। नमस्ते।"
        )
    return (
        f"{who}इतना ध्यान से बताने के लिए बहुत धन्यवाद। "
        "अगर आगे इनमें से कुछ भी दिखे, तो अपने स्वास्थ्य कार्यकर्ता से बात करें। "
        "आपकी रिपोर्ट तैयार हो रही है। यह रिपोर्ट आप अपनी ASHA, ANM या डॉक्टर को दिखा सकती हैं। नमस्ते।"
    )


def _tool_body(state: dict) -> dict:
    body = {"result": "ok", "say": _say_hindi(state)}
    if state.get("phase") == "closing":
        body["stop"] = True
    else:
        body["wait"] = True
    return body


def _profile_kwargs(qid: str, args: dict) -> dict:
    kwargs: dict = {}
    if qid == "intro_name":
        name = _just_name(str(args.get("caller_name") or ""))
        if name:
            kwargs["caller_name"] = name
    elif qid == "intro_married":
        status = _marital(args.get("marital_status"))
        if status:
            kwargs["marital_status"] = status
    elif qid == "intro_children":
        if "child_count" in args and args.get("child_count") not in (None, ""):
            try:
                kwargs["child_count"] = int(args.get("child_count"))
            except (TypeError, ValueError):
                pass
        if "children" in args and args.get("children") is not None:
            kids = _as_children(args.get("children"))
            if kids is not None:
                kwargs["children"] = kids
        if kwargs.get("child_count") == 0 and "children" not in kwargs:
            kwargs["children"] = []
    elif qid == "ever_pregnant":
        flag = _optional_bool(args.get("ever_pregnant"))
        if flag is not None:
            kwargs["ever_pregnant"] = flag
    return kwargs


async def _mark_closing(
    cursor: _Cursor,
    reason: str,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None,
) -> None:
    state = cursor.state
    if state.get("phase") != "closing":
        state["phase"] = "closing"
        state["close_reason"] = reason
    elif not state.get("close_reason"):
        state["close_reason"] = reason
    cursor.state = state
    try:
        db.save_cursor_state(call_id, state)
    except Exception:
        pass
    if _accept_end(telemetry):
        await _schedule_end_report(
            ws,
            ws_lock,
            call_id,
            telemetry,
            str(state.get("close_reason") or reason),
            on_end_accepted,
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
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None = None,
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
        args = _call_args(fc)
        qid = str(cursor.state.get("active_qid") or "")
        try:
            if name == "save_profile":
                kwargs = _profile_kwargs(qid, args)
                if qid in _PROFILE_QIDS and kwargs:
                    new_state, _hold = apply_profile(cursor.state, **kwargs)
                    cursor.state = new_state
                    db.save_cursor_state(call_id, new_state)
                body = _tool_body(cursor.state)
                logged.append("save_profile")
            elif name == "record_turn":
                outcome = str(args.get("outcome") or "").strip().lower().replace(" ", "_")
                outcome = {
                    "yes": "concern",
                    "haan": "concern",
                    "positive": "concern",
                    "no": "clear",
                    "nahi": "clear",
                    "negative": "clear",
                    "dont_know": "unknown",
                    "don't_know": "unknown",
                    "confused": "clarify",
                    "stop": "caller_stop",
                }.get(outcome, outcome)
                if qid in _PROFILE_QIDS:
                    body = _tool_body(cursor.state)
                    logged.append("record_turn_held")
                else:
                    new_state = apply_turn(
                        cursor.state,
                        outcome,
                        child_ref=str(args.get("child_ref") or ""),
                        note=str(args.get("note") or ""),
                    )
                    cursor.state = new_state
                    db.save_cursor_state(call_id, new_state)
                    if new_state.get("phase") == "closing":
                        await _mark_closing(
                            cursor,
                            str(new_state.get("close_reason") or outcome),
                            ws,
                            ws_lock,
                            call_id,
                            telemetry,
                            on_end_accepted,
                        )
                    body = _tool_body(cursor.state)
                    logged.append(f"record_turn_{outcome}")
            else:
                body = {"result": "ignored", "say": _say_hindi(cursor.state), "wait": True}
                logged.append(name or "ignored")
                telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
        except Exception:
            telemetry["tool_errors"] = int(telemetry.get("tool_errors") or 0) + 1
            body = {"result": "error", "say": _say_hindi(cursor.state), "wait": True}
            logged.append(f"{name or 'tool'}_error")
        if cursor.state.get("phase") == "closing":
            body["stop"] = True
            body.pop("wait", None)
            body["say"] = _closing_hi(cursor.state)
        if not fc_id:
            continue
        responses.append(
            types.FunctionResponse(
                id=fc_id,
                name=name or "record_turn",
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
    if telemetry.get("end_requested"):
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
    cursor: _Cursor,
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None = None,
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
        user_text = ""
        if not telemetry.get("end_requested"):
            user_text = await _forward_transcript(
                ws,
                ws_lock,
                call_id,
                "user",
                server_content.input_transcription,
                cursor,
                telemetry,
            )
        if user_text:
            if state.bot_text_since_user:
                state.last_bot_text = state.bot_text_since_user
                state.bot_text_since_user = ""
            state.last_user_text = user_text
            _arm_user_turn(state, telemetry)
            await _emit(
                ws, ws_lock, call_id, telemetry, "user_transcript", user_text
            )
        bot_text = await _forward_transcript(
            ws,
            ws_lock,
            call_id,
            "assistant",
            server_content.output_transcription,
            cursor,
            telemetry,
        )
        if bot_text:
            state.bot_text_since_user = (state.bot_text_since_user + " " + bot_text).strip()[-600:]

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
            cursor,
            on_end_accepted=on_end_accepted,
        )
        if not telemetry.get("end_requested"):
            _arm_waiting_reply(state, speech_end_waiting=False)
            state.quiet_from = 0.0

    if server_content is not None and server_content.turn_complete:
        state.bot_speaking = False
        if not state.awaiting_bot_reply:
            state.quiet_from = max(time.time(), state.playback_ends_at)
        if not state.awaiting_bot_reply or telemetry.get("end_requested"):
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
    cursor: _Cursor,
    reconnect,
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None = None,
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
                    on_end_accepted,
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
        await _send_json(ws, ws_lock, {"type": "reconnecting", "attempt": reconnects})
        await _emit(ws, ws_lock, call_id, telemetry, "reconnecting", str(reconnects))
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
        await _send_json(ws, ws_lock, {"type": "reconnected", "attempt": reconnects})
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
    cursor: _Cursor,
    watchdog_timeout_sec: float,
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None = None,
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
                        cursor, "time_limit", ws, ws_lock, call_id, telemetry, on_end_accepted
                    )
                stop.set()
                return
            if cursor.state.get("phase") == "closing" or telemetry.get("end_requested"):
                played = (
                    bool(telemetry.get("closing_heard"))
                    and not state.bot_speaking
                    and state.playback_ends_at > 0
                    and now >= state.playback_ends_at + _CLOSING_AUDIO_GRACE_SEC
                )
                waited = now - float(telemetry.get("end_requested_at") or now)
                if played or waited > _CLOSING_MAX_WAIT_SEC:
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
                    cue = (
                        "SYSTEM CUE (from the app, not the mother): she has been quiet for 15 seconds. "
                        "Ask in Hindi if she can hear you, then repeat the same question in simpler words. "
                        "Do not call a tool."
                    )
            if reason:
                await _mark_closing(
                    cursor, reason, ws, ws_lock, call_id, telemetry, on_end_accepted
                )
                cue = (
                    "SYSTEM CUE (from the app, not the mother): speak only this Hindi, then stop. "
                    "Do not call a tool.\n"
                    + _closing_hi(cursor.state)
                )
            if cue:
                if not reason:
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
            if state.nudges_for_current_turn >= 6:
                continue
            if time.time() - state.last_user_turn_ts < effective_timeout:
                continue
            state.nudges_for_current_turn += 1
            telemetry["watchdog_nudges"] += 1
            state.last_user_turn_ts = time.time()
            nudge = (
                "SYSTEM CUE (from the app, not the mother): speak only this Hindi now, then wait. "
                "Do not call a tool.\n"
                + _say_hindi(cursor.state)
            )
            ok = await _send_realtime_safe(
                session, send_lock, state, stop, text=nudge
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
            if telemetry.get("end_requested") and kind in {"audio", "text", "speech_end", "nudge"}:
                continue
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
                qid = str(cursor.state.get("active_qid") or "")
                db.append_transcript(call_id, "user", text, qid=qid)
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


async def _finalize(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict | None,
    recorder: audio_recorder.CallAudioRecorder,
    on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None,
) -> None:
    # Stop the voice UI before any report work. A duplicate ending is harmless.
    await _send_json(ws, ws_lock, {"type": "ending"})
    filename = ""
    duration = 0
    try:
        filename, duration = await asyncio.to_thread(recorder.save)
    except Exception:
        filename, duration = "", 0

    if telemetry is not None:
        _refresh_latency(telemetry)
        try:
            db.update_call_telemetry(call_id, telemetry)
        except Exception:
            pass

    db.complete_call(call_id, int(duration), filename, "", "")
    ready = await _report_is_ready(on_call_completed, call_id)
    call = db.get_call(call_id) or {}
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
    on_end_accepted: Callable[[int, dict], Awaitable[None] | None] | None = None,
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
        db.save_cursor_state(call_id, cursor.state)
        share_token = str((db.get_call(call_id) or {}).get("share_token") or "")
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
                on_end_accepted,
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
                on_end_accepted,
            )
        )
        try:
            async with send_lock:
                await session.send_realtime_input(
                    text=(
                        "SYSTEM CUE (from the app, not the mother): speak only this Hindi, then wait. "
                        "Do not call a tool yet.\n"
                        + _say_hindi(cursor.state)
                    )
                )
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
                )
            except Exception:
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "error", "message": "Could not finish the call"},
                )
        await _close_ws(ws)
