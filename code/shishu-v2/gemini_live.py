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

import answer_backstop
import answer_extractor
import audio_recorder
import call_state
import answer_tool
import db
import checklist
import question_flow
from prompts import (
    GEMINI_LIVE_SYSTEM_PROMPT,
    HINDI_MEDICAL_VOCAB,
    clean_assistant_caption,
    clean_user_caption,
)
from summarizer import generate_call_summaries

_background_tasks: set[asyncio.Task] = set()

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_WATCHDOG_INTERVAL_SEC = 0.5
_GREETING = (
    "Greet the caller now in Hindi: Namaste, main Shishu hoon. Aapka naam bataiyega please?"
)
_WATCHDOG_NUDGE = (
    "Speak as SHISHU in Hindi only in your same warm female voice (Kore). "
    "If your previous question was not clearly answered, gently repeat or clarify that single question. "
    "NEVER say 'Nahi' or 'Haan' yourself, and never answer for the mother."
)
_MANUAL_NUDGE = _WATCHDOG_NUDGE
_FINDING_RESULT = "ok"
RECORD_ANSWERS_DECL = answer_tool.RECORD_ANSWERS_DECL
END_CALL_DECL = answer_tool.END_CALL_DECL
_SILENCE_CLOSE_INSTRUCTION = (
    "SYSTEM CUE (from the app, not the mother): she has been silent for 30 seconds. "
    "Speak the INCOMPLETE CLOSING for silence now, in Hindi. "
    "Do not call end_call. The server ends the call after your audio finishes."
)
_TIME_CLOSE_INSTRUCTION = (
    "SYSTEM CUE (from the app, not the mother): the call has reached its time limit. "
    "Speak the INCOMPLETE CLOSING for time now, in Hindi. "
    "Do not call end_call. The server ends the call after your audio finishes."
)

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
    "answer_backstop",
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
        self.user_lines: list[str] = []
        self.last_bot_text = ""
        self.bot_text_since_user = ""
        self.time_cue_sent = False
        self.speech_end_waiting = False
        self.playback_ends_at = 0.0
        self.call_state = None
        self.call_id = None
        self.tool_called_since_user = True
        self.backstop_checked = False


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
        tools=[{"function_declarations": [RECORD_ANSWERS_DECL, END_CALL_DECL]}],
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
    if telemetry.get("end_requested") or telemetry.get("ending_sent"):
        return
    if telemetry.get("close_reason") != "silence":
        return
    telemetry["force_close"] = False
    telemetry["force_close_at"] = 0.0
    telemetry["close_reason"] = ""
    state.checked_in = False
    live_state = state.call_state
    if isinstance(live_state, dict) and live_state.get("close_reason") == "silence":
        restored = dict(live_state)
        restored["close_reason"] = None
        restored["phase"] = "asking"
        current = dict(restored.get("current_question") or {})
        if current:
            current["phase"] = "waiting_for_answer"
            restored["current_question"] = current
        state.call_state = restored
        if state.call_id:
            try:
                db.save_call_state(state.call_id, restored)
            except Exception:
                pass


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
    state.tool_called_since_user = False
    state.backstop_checked = False
    if state.bot_text_since_user:
        state.last_bot_text = state.bot_text_since_user
        state.bot_text_since_user = ""


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
    r"\b(no\s+child|no\s+children|don'?t\s+have\s+(?:any\s+)?children|do\s+not\s+have\s+(?:any\s+)?children|"
    r"childless|0\s+children|zero\s+children|"
    r"unmarried|not\s+married|never\s+married|single|"
    r"never\s+(?:been\s+)?pregnant|haven'?t\s+been\s+pregnant|"
    r"have\s+not\s+been\s+pregnant|not\s+pregnant|wasn'?t\s+pregnant|was\s+not\s+pregnant|"
    r"to\s+be\s+married)\b"
    r"|बच्चे\s+नहीं|कोई\s+बच्चा\s+नहीं|बच्चा\s+नहीं\s+है|चिल्ड्रन"
    r"|प्रेग्नेंट\s+नहीं|गर्भवती\s+नहीं|शादी\s+नहीं\s+हुई|शादी\s+होने\s+वाली|अविवाहित",
    re.I,
)
_NO_CHILD_ONLY_RE = re.compile(
    r"\b(no\s+child|no\s+children|don'?t\s+have\s+(?:any\s+)?children|do\s+not\s+have\s+(?:any\s+)?children|"
    r"childless|0\s+children|zero\s+children)\b"
    r"|बच्चे\s+नहीं|कोई\s+बच्चा\s+नहीं|बच्चा\s+नहीं\s+है|डोंट\s+हैव\s+चिल्ड्रन",
    re.I,
)
_UNMARRIED_RE = re.compile(
    r"\b(unmarried|not\s+married|never\s+married|single|not\s+yet\s+married)\b"
    r"|शादी\s+नहीं\s+हुई|अविवाहित|नो,\s*इट्स\s+नॉट",
    re.I,
)
_EVER_PREGNANT_NO_RE = re.compile(
    r"\b(never\s+(?:been\s+)?pregnant|haven'?t\s+(?:ever\s+)?been\s+pregnant|"
    r"have\s+not\s+(?:ever\s+)?been\s+pregnant|never\s+conceived)\b"
    r"|कभी\s+प्रेग्नेंट\s+नहीं|कभी\s+गर्भवती\s+नहीं",
    re.I,
)
_NEVER_PREGNANT_RE = re.compile(
    r"\b(never\s+(?:been\s+)?pregnant|haven'?t\s+been\s+pregnant|have\s+not\s+been\s+pregnant|"
    r"not\s+pregnant|wasn'?t\s+pregnant|was\s+not\s+pregnant|no\s+pregnancy|"
    r"not\s+currently\s+pregnant)\b"
    r"|प्रेग्नेंट\s+नहीं|गर्भवती\s+नहीं",
    re.I,
)
_PREG_DELIVERY_TOPICS = (
    "preg_conditions",
    "preg_scan",
    "preg_heart",
    "preg_transfusion",
    "preg_delivery",
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
    "consaguinity": "consanguinity",
}
_DOMAIN_AS_TOPIC = {
    "profile_children": "intro_children",
    "profile_married": "intro_married",
    "consaguinity": "fam_related",
}


def _normalize_finding(
    domain: str,
    detail: str,
    raw_topics: list,
    last_user_text: str = "",
    status: str = "",
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
    combined_text = f"{detail} {last_user_text}"
    user_no = bool(_NO_MORE_USER_RE.match((last_user_text or "").strip()))
    unmarried_now = (
        "intro_married" in topics
        and (status == "negative" or user_no or bool(_UNMARRIED_RE.search(combined_text)))
    )
    no_child_now = bool(_NO_CHILD_ONLY_RE.search(combined_text)) or (
        "intro_children" in topics and status == "negative"
    )
    never_preg_now = bool(_NEVER_PREGNANT_RE.search(combined_text))
    ever_preg_no = bool(_EVER_PREGNANT_NO_RE.search(combined_text))
    allow_group_skip = unmarried_now or no_child_now or never_preg_now
    if not allow_group_skip:
        topics = [t for t in topics if str(t).strip() not in checklist.GROUPS]
    else:
        if unmarried_now:
            for item in ("intro_children", "birth", "body", "illness", *_PREG_DELIVERY_TOPICS, "fam_related"):
                if item not in topics:
                    topics.append(item)
        if no_child_now:
            for grp in ("birth", "body", "illness"):
                if grp not in topics:
                    topics.append(grp)
            for pt in _PREG_DELIVERY_TOPICS:
                if pt not in topics:
                    topics.append(pt)
        if never_preg_now:
            for pt in _PREG_DELIVERY_TOPICS:
                if pt not in topics:
                    topics.append(pt)
            if ever_preg_no and "fam_miscarriage" not in topics:
                topics.append("fam_miscarriage")
            if domain == "pregnancy" or any(str(t).startswith("preg_") for t in topics):
                if "pregnancy" not in topics:
                    topics.append("pregnancy")
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


def _remember_user_line(state: _TurnState, text: str) -> None:
    """Keep each finished user line. A later count can still see the ages."""
    text = (text or "").strip()
    if not text:
        return
    state.last_user_text = text
    lines = state.user_lines
    if lines:
        prev = lines[-1]
        if text == prev or text.startswith(prev) or prev.startswith(text):
            if len(text) >= len(prev):
                lines[-1] = text
            return
    lines.append(text)


def _prior_user_lines(state: _TurnState | None, current: str) -> list[str]:
    if state is None:
        return []
    lines = list(state.user_lines)
    if lines and lines[-1] == (current or "").strip():
        return lines[:-1]
    return lines


def _transcript_turn_id(call_id: int) -> int:
    call = db.get_call(call_id) or {}
    transcript = call.get("transcript") or []
    if not transcript:
        return 0
    return len(transcript) - 1


def _load_live_state(call_id: int, turn_state: _TurnState | None) -> dict:
    if turn_state is not None and isinstance(turn_state.call_state, dict):
        return turn_state.call_state
    saved = db.get_call_state(call_id)
    if saved is None:
        saved = db.initialize_call_state(call_id)
    if turn_state is not None:
        turn_state.call_state = saved
    return saved


def _store_live_state(call_id: int, turn_state: _TurnState | None, live_state: dict) -> None:
    if turn_state is not None:
        turn_state.call_state = live_state
    db.save_call_state(call_id, live_state)


def _progress_label(live_state: dict) -> str:
    open_ids = [
        topic_id
        for topic_id in call_state.unresolved_topic_ids(live_state)
        if topic_id in checklist.REQUIRED_IDS
    ]
    done = len(checklist.REQUIRED_IDS) - len(open_ids)
    return f"{done}/{len(checklist.REQUIRED_IDS)}"


def _advance(live_state: dict) -> tuple[dict, dict | None]:
    """Move to the next question after a save. Returns the new state and that question."""
    question = call_state.next_question(live_state)
    if question is None and call_state.is_complete(live_state):
        live_state = call_state.request_close(live_state, "complete")
        question = call_state.next_question(live_state)
    elif question and question.get("question_kind") != "closing":
        live_state = call_state.begin_question(live_state, question)
    return live_state, question


def _apply_client_telemetry(call_id: int, telemetry: dict, payload: dict) -> None:
    client = payload.get("client") if isinstance(payload.get("client"), dict) else {}
    telemetry["client"] = client
    db.update_call_telemetry(call_id, telemetry)
    db.append_call_event(call_id, "client_telemetry", "", {"client": client})


def _apply_browser_hangup(call_id: int, telemetry: dict, turn_state: _TurnState | None) -> None:
    live_state = turn_state.call_state if turn_state is not None and isinstance(turn_state.call_state, dict) else None
    if live_state is None:
        live_state = db.get_call_state(call_id)
    if live_state is None:
        if not telemetry.get("close_reason"):
            telemetry["close_reason"] = "browser_hangup"
        return
    if not live_state.get("close_reason"):
        live_state = call_state.request_close(live_state, "browser_hangup")
        _store_live_state(call_id, turn_state, live_state)
    telemetry["close_reason"] = live_state.get("close_reason") or telemetry.get("close_reason") or "browser_hangup"


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
    last_bot_text: str = "",
    turn_state: _TurnState | None = None,
) -> None:
    responses: list[types.FunctionResponse] = []
    logged: list[str] = []
    db_records: list[tuple[str, str, str, bool]] = []
    ws_messages: list[dict] = []
    live_state = _load_live_state(call_id, turn_state)
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
            end_reason = str(_call_args(fc).get("reason") or "").strip().lower()
            if end_reason not in ("emergency", "caller_stop"):
                body = {
                    "result": "rejected",
                    "note": (
                        "Use this tool only for emergency or caller_stop. "
                        "For a completed screening, speak the closing. "
                        "The server ends the call after your audio finishes."
                    ),
                }
                logged.append("end_call_rejected")
            else:
                live_state = call_state.request_close(live_state, end_reason)
                _store_live_state(call_id, turn_state, live_state)
                telemetry["close_reason"] = end_reason
                if end_reason == "emergency" and not telemetry.get("emergency_detail"):
                    telemetry["emergency_detail"] = (last_user_text or "")[:300]
                question = call_state.next_question(live_state)
                body = {"result": "ok", "note": answer_tool.build_turn_context(live_state, question)}
                logged.append(f"end_call_{end_reason}")
        elif name == "record_answers":
            args = _call_args(fc)
            updates = args.get("updates") or []
            if isinstance(updates, dict):
                updates = [updates]
            try:
                live_state = call_state.apply_updates(
                    live_state,
                    list(updates),
                    _transcript_turn_id(call_id),
                    utterance=last_user_text or "",
                    prior_utterances=_prior_user_lines(turn_state, last_user_text or ""),
                )
            except ValueError as exc:
                recovery_context = answer_tool.build_turn_context(
                    live_state, call_state.next_question(live_state)
                )
                body = {
                    "result": "rejected",
                    "note": f"Error: {exc}. Follow this active requirement: {recovery_context}",
                }
                logged.append("record_answers_rejected")
                telemetry.setdefault("tool_errors", []).append({
                    "turn": _transcript_turn_id(call_id),
                    "error": str(exc),
                    "args": args,
                })
            else:
                live_state, question = _advance(live_state)
                _store_live_state(call_id, turn_state, live_state)
                body = {"result": "ok", "note": answer_tool.build_turn_context(live_state, question)}
                label = _progress_label(live_state)
                telemetry["progress_group"] = label
                ws_messages.append({"type": "progress", "group": label})
                logged.append("record_answers")
        else:
            body = {"result": "ignored"}
            logged.append(name or "ignored")
        if not fc_id:
            continue
        responses.append(
            types.FunctionResponse(
                id=fc_id,
                name=name or "record_answers",
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


async def _run_answer_backstop(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    telemetry: dict,
    state: _TurnState,
) -> bool:
    """Save her answer when the model ended its turn without record_answers. True if a cue was sent."""
    live_state = _load_live_state(call_id, state)
    asked = call_state.next_question(live_state)
    if not asked or asked.get("question_kind") not in ("topic", "followup"):
        return False
    label = f"{asked['topic_id']}/{asked.get('followup_id') or '-'}"
    try:
        result = await asyncio.wait_for(
            answer_backstop.extract(asked, live_state, state.last_bot_text, state.last_user_text),
            answer_backstop.TIMEOUT_SEC,
        )
        updates = answer_backstop.to_updates(asked, live_state, result)
        if updates:
            live_state = call_state.apply_updates(
                live_state,
                updates,
                _transcript_turn_id(call_id),
                utterance=state.last_user_text,
                prior_utterances=_prior_user_lines(state, state.last_user_text),
            )
    except Exception as exc:
        await _emit(ws, ws_lock, call_id, telemetry, "answer_backstop", f"{label} error: {exc}"[:240])
        return False
    if not updates:
        await _emit(ws, ws_lock, call_id, telemetry, "answer_backstop", f"{label} {result.get('kind')}")
        return False
    live_state, question = _advance(live_state)
    _store_live_state(call_id, state, live_state)
    telemetry["backstop_saves"] = telemetry.get("backstop_saves", 0) + 1
    progress = _progress_label(live_state)
    telemetry["progress_group"] = progress
    await _send_json(ws, ws_lock, {"type": "progress", "group": progress})
    await _emit(ws, ws_lock, call_id, telemetry, "answer_backstop", f"{label} saved", {"updates": updates})
    # A model that already spoke its next line is resynced by its next record_answers response.
    if state.bot_text_since_user and (question or {}).get("question_kind") != "closing":
        return False
    cue = (
        "SYSTEM CUE (from the app, not the mother): the app saved her last answer. "
        + answer_tool.build_turn_context(live_state, question)
    )
    try:
        async with send_lock:
            await session.send_realtime_input(text=cue)
    except Exception:
        return False
    _arm_waiting_reply(state, speech_end_waiting=False)
    state.quiet_from = 0.0
    return True


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
            _remember_user_line(state, user_text)
            if question_flow.is_emergency_text(user_text):
                telemetry["emergency_seen"] = True
                telemetry["emergency_detail"] = user_text[:300]
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
        if turn_state := state:
            turn_state.tool_called_since_user = True
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
            last_bot_text=state.last_bot_text,
            turn_state=state,
        )
        if not telemetry.get("end_requested"):
            _arm_waiting_reply(state, speech_end_waiting=False)
            state.quiet_from = 0.0

    if server_content is not None and server_content.turn_complete:
        state.bot_speaking = False
        live_state = state.call_state if isinstance(state.call_state, dict) else None
        if (
            live_state
            and live_state.get("phase") == "asking"
            and not state.tool_called_since_user
            and state.last_user_text
            and not state.backstop_checked
            and not telemetry.get("end_requested")
        ):
            state.backstop_checked = True
            if await _run_answer_backstop(session, send_lock, ws, ws_lock, call_id, telemetry, state):
                return True
            live_state = state.call_state
        if (
            live_state
            and live_state.get("phase") == "asking"
            and state.awaiting_bot_reply
            and not telemetry.get("end_requested")
        ):
            state.awaiting_bot_reply = False
            state.speech_end_waiting = False
        if not state.awaiting_bot_reply:
            state.quiet_from = max(time.time(), state.playback_ends_at)
        if live_state and live_state.get("phase") == "closing" and not telemetry.get("ending_sent"):
            telemetry["ending_sent"] = True
            telemetry["end_requested"] = True
            telemetry["end_requested_at"] = time.time()
            if not telemetry.get("close_reason"):
                telemetry["close_reason"] = live_state.get("close_reason") or ""
            await _send_json(ws, ws_lock, {"type": "ending"})
            ended = dict(live_state)
            ended["phase"] = "ended"
            state.call_state = ended
            try:
                db.save_call_state(call_id, ended)
            except Exception:
                pass
        elif live_state and live_state.get("phase") == "asking":
            waited = call_state.wait_for_answer(live_state)
            state.call_state = waited
            try:
                db.save_call_state(call_id, waited)
            except Exception:
                pass
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
            phase = state.call_state.get("phase") if isinstance(state.call_state, dict) else ""
            if (
                now - state.call_started_ts >= checklist.SOFT_LIMIT_SEC
                and not state.time_cue_sent
                and phase != "closing"
            ):
                closing_state = _load_live_state(call_id, state)
                closing_state = call_state.request_close(closing_state, "time_limit")
                _store_live_state(call_id, state, closing_state)
                telemetry["close_reason"] = "time_limit"
                cue, reason = _TIME_CLOSE_INSTRUCTION, "time_limit"
                state.time_cue_sent = True
            elif (
                phase != "closing"
                and state.quiet_from > 0
                and not state.bot_speaking
                and not state.awaiting_bot_reply
            ):
                quiet = now - state.quiet_from
                if quiet >= checklist.SILENCE_CLOSE_SEC:
                    closing_state = _load_live_state(call_id, state)
                    closing_state = call_state.mark_current_no_response(closing_state)
                    closing_state = call_state.request_close(closing_state, "silence")
                    _store_live_state(call_id, state, closing_state)
                    telemetry["close_reason"] = "silence"
                    cue, reason = _SILENCE_CLOSE_INSTRUCTION, "silence"
                elif quiet >= checklist.CHECK_IN_SEC and not state.checked_in:
                    cue = checklist.CHECK_IN_CUE
            if cue:
                if reason:
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
                _remember_user_line(state, text)
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
            elif kind == "client_telemetry":
                _apply_client_telemetry(call_id, telemetry, payload)
            elif kind == "hangup":
                _apply_browser_hangup(call_id, telemetry, state)
                stop.set()
                return
    except asyncio.CancelledError:
        raise


def _spawn_doctor_note(
    call_id: int,
    transcript: list,
    findings: dict,
    red_flags: list,
    telemetry: dict | None,
    call_state_obj: dict | None = None,
) -> None:
    task = asyncio.create_task(
        _write_doctor_note(call_id, transcript, findings, red_flags, telemetry, call_state_obj)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _write_doctor_note(
    call_id: int,
    transcript: list,
    findings: dict,
    red_flags: list,
    telemetry: dict | None,
    call_state_obj: dict | None = None,
) -> None:
    try:
        result = await generate_call_summaries(
            transcript, findings, red_flags, state=call_state_obj
        )
    except Exception:
        return
    if not isinstance(result, dict):
        return
    summary_doctor = str(result.get("summary_doctor") or "")
    summary_mother = str(result.get("summary_mother") or "").strip() or None
    merged_findings = result.get("findings") if isinstance(result.get("findings"), dict) else findings
    merged_flags = result.get("red_flags") if isinstance(result.get("red_flags"), list) else red_flags
    reconciled_state = result.get("state") if isinstance(result.get("state"), dict) else call_state_obj
    if isinstance(reconciled_state, dict):
        try:
            db.save_call_state(call_id, reconciled_state)
        except Exception:
            pass
    if telemetry is not None:
        if reconciled_state:
            open_ids = set(call_state.unresolved_topic_ids(reconciled_state))
            covered = [topic_id for topic_id in checklist.REQUIRED_IDS if topic_id not in open_ids]
            close_reason = str(reconciled_state.get("close_reason") or telemetry.get("close_reason") or "")
        else:
            covered = telemetry.get("coverage") or []
            close_reason = str(telemetry.get("close_reason") or "")
        doctor_note, mother_note = checklist.incomplete_notes(
            covered,
            close_reason,
            str(telemetry.get("emergency_detail") or ""),
        )
        if doctor_note:
            summary_doctor = doctor_note + "\n\n" + summary_doctor
        if mother_note and summary_mother:
            if close_reason == "emergency":
                summary_mother = (mother_note + "\n\n" + summary_mother).strip()
            else:
                summary_mother = (summary_mother + "\n\n" + mother_note).strip()
    try:
        db.save_doctor_summary(
            call_id,
            summary_doctor,
            merged_findings,
            list(merged_flags),
            summary_mother=summary_mother if reconciled_state else None,
        )
    except Exception:
        return


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
    transcript = list(call.get("transcript") or [])
    findings = dict(call.get("findings") or {})
    red_flags = list(call.get("red_flags") or [])
    saved_state = call.get("state") if isinstance(call.get("state"), dict) else None
    if saved_state:
        import summarizer

        findings, red_flags = summarizer.state_to_findings_and_flags(saved_state, findings, red_flags)
    try:
        extracted = await answer_extractor.cached_answers(call)
    except Exception:
        extracted = answer_extractor.fallback_from_call(call)
        await _report_error(
            ws,
            ws_lock,
            call_id,
            telemetry,
            "Mother report extract failed",
        )
    summary_mother = answer_extractor.mother_summary(extracted, findings, red_flags)
    if telemetry is not None:
        if saved_state:
            open_ids = set(call_state.unresolved_topic_ids(saved_state))
            covered = [topic_id for topic_id in checklist.REQUIRED_IDS if topic_id not in open_ids]
            close_reason = str(saved_state.get("close_reason") or telemetry.get("close_reason") or "")
        else:
            covered = telemetry.get("coverage") or []
            close_reason = str(telemetry.get("close_reason") or "")
        _doctor_note, mother_note = checklist.incomplete_notes(
            covered, close_reason, str(telemetry.get("emergency_detail") or "")
        )
        if mother_note:
            if close_reason == "emergency":
                summary_mother = (mother_note + "\n\n" + summary_mother).strip()
            else:
                summary_mother = (summary_mother + "\n\n" + mother_note).strip()
    db.complete_call(
        call_id,
        int(duration),
        filename,
        "",
        summary_mother,
        findings=findings,
        red_flags=red_flags,
    )
    _spawn_doctor_note(call_id, transcript, findings, red_flags, telemetry, saved_state)
    await _send_json(
        ws,
        ws_lock,
        {
            "type": "call_completed",
            "call_id": call_id,
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
        state.call_id = call_id
        state.call_state = db.initialize_call_state(call_id)
        opening = call_state.next_question(state.call_state)
        if opening:
            state.call_state = call_state.begin_question(state.call_state, opening)
            db.save_call_state(call_id, state.call_state)
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
            greeting_prompt = _GREETING
            if opening:
                opening_context = answer_tool.build_turn_context(state.call_state, opening)
                greeting_prompt = f"{_GREETING}\n\nSYSTEM INSTRUCTION:\n{opening_context}"
            async with send_lock:
                await session.send_realtime_input(text=greeting_prompt)
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
