"""Standalone Gemini Live pipeline for one SHISHU screening call (/ws/gemini)."""

import asyncio
import base64
import json
import os

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

import audio_recorder
import db
from prompts import GEMINI_LIVE_SYSTEM_PROMPT, HINDI_MEDICAL_VOCAB
from summarizer import generate_call_summaries

_ENV_PATH = "/Users/jigar/projects/0n-going/SHISHU-healthathon-koita/.env"
load_dotenv(_ENV_PATH)

_ENGINE = "gemini-3.8-live"
_FALLBACK_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
_INPUT_RATE = 16000
_OUTPUT_RATE = 24000
_GREETING = (
    "Call connected. Greet the mother warmly in Hindi as SHISHU and ask her name."
)

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


def _live_config() -> dict:
    base = {
        "response_modalities": ["AUDIO"],
        "system_instruction": GEMINI_LIVE_SYSTEM_PROMPT,
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
        },
        "output_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {
                "disabled": False,
                "prefix_padding_ms": 40,
                "silence_duration_ms": 700,
            }
        },
        "tools": [{"function_declarations": [_RECORD_FINDING_DECL]}],
    }
    with_vocab = {
        **base,
        "input_audio_transcription": {
            "custom_vocabulary": list(HINDI_MEDICAL_VOCAB),
        },
    }
    try:
        types.LiveConnectConfig(**with_vocab)
    except Exception:
        return {**base, "input_audio_transcription": {}}
    return with_vocab


def _primary_model() -> str:
    model = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
    return model or "gemini-3.8-live"


async def _connect_live(client: genai.Client, config: dict):
    primary = _primary_model()
    models = [primary]
    if primary != _FALLBACK_MODEL:
        models.append(_FALLBACK_MODEL)
    last_error: Exception | None = None
    for model in models:
        ctx = client.aio.live.connect(model=model, config=config)
        try:
            session = await ctx.__aenter__()
        except Exception as exc:
            last_error = exc
            continue
        if getattr(session, "setup_complete", None) is None:
            last_error = RuntimeError("Gemini Live setup did not complete")
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
            continue
        return ctx, session
    if last_error is None:
        last_error = RuntimeError("Gemini Live connection failed")
    raise last_error


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


async def _forward_transcript(
    ws: WebSocket,
    lock: asyncio.Lock,
    call_id: int,
    role: str,
    transcription: types.Transcription | None,
) -> None:
    if transcription is None:
        return
    text = transcription.text or ""
    if not text:
        return
    db.append_transcript(call_id, role, text)
    await _send_json(
        ws, lock, {"type": "transcript", "role": role, "text": text}
    )


async def _answer_function_calls(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    calls: list,
    seen: set[str],
    cancelled: set[str],
) -> None:
    responses: list[types.FunctionResponse] = []
    for fc in calls:
        fc_id = getattr(fc, "id", None) or ""
        if fc_id and (fc_id in cancelled or fc_id in seen):
            continue
        if getattr(fc, "will_continue", False):
            continue
        if fc_id:
            seen.add(fc_id)
        name = getattr(fc, "name", None) or ""
        if name == "record_finding":
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
            body = {"result": "logged"}
        else:
            body = {"result": "ignored"}
        if not fc_id:
            continue
        responses.append(
            types.FunctionResponse(
                id=fc_id,
                name=name or "record_finding",
                response=body,
                scheduling=types.FunctionResponseScheduling.SILENT,
            )
        )
    if not responses:
        return
    async with send_lock:
        await session.send_tool_response(function_responses=responses)


async def _handle_server_message(
    response,
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
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
        await _send_json(ws, ws_lock, {"type": "clear"})

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
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            part_calls.append(function_call)

    if server_content is not None:
        await _forward_transcript(
            ws,
            ws_lock,
            call_id,
            "user",
            server_content.input_transcription,
        )
        await _forward_transcript(
            ws,
            ws_lock,
            call_id,
            "assistant",
            server_content.output_transcription,
        )

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
            calls,
            seen,
            cancelled,
        )

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
        await _send_json(
            ws, ws_lock, {"type": "error", "message": "Gemini Live session ended"}
        )


async def _browser_loop(
    session,
    send_lock: asyncio.Lock,
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    stop: asyncio.Event,
    call_id: int,
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
                await _send_json(
                    ws, ws_lock, {"type": "error", "message": "Invalid JSON"}
                )
                continue
            if not isinstance(payload, dict):
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "error", "message": "Expected a JSON object"},
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
                    await _send_json(
                        ws,
                        ws_lock,
                        {
                            "type": "error",
                            "message": "Could not forward audio to Gemini",
                        },
                    )
                    return
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
                try:
                    async with send_lock:
                        await session.send_client_content(
                            turns={
                                "role": "user",
                                "parts": [{"text": text}],
                            },
                            turn_complete=True,
                        )
                except Exception:
                    stop.set()
                    await _send_json(
                        ws,
                        ws_lock,
                        {
                            "type": "error",
                            "message": "Could not forward text to Gemini",
                        },
                    )
                    return
            elif kind == "hangup":
                stop.set()
                return
    except asyncio.CancelledError:
        raise


async def _finalize(
    ws: WebSocket,
    ws_lock: asyncio.Lock,
    call_id: int,
    recorder: audio_recorder.CallAudioRecorder,
) -> None:
    filename = ""
    duration = 0
    try:
        filename, duration = recorder.save()
    except Exception:
        filename, duration = "", 0

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
        await _send_json(
            ws,
            ws_lock,
            {"type": "error", "message": "Post-call summary failed"},
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
    if tasks:
        await asyncio.gather(
            *[task for task in tasks if task is not None],
            return_exceptions=True,
        )


async def handle_gemini_ws(ws: WebSocket) -> None:
    load_dotenv(_ENV_PATH)
    if ws.client_state == WebSocketState.CONNECTING:
        await ws.accept()

    ws_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    stop = asyncio.Event()
    call_id: int | None = None
    recorder: audio_recorder.CallAudioRecorder | None = None
    session_ctx = None
    gemini_task: asyncio.Task | None = None
    browser_task: asyncio.Task | None = None
    finalized = False

    try:
        if not os.environ.get("GEMINI_API_KEY"):
            await _send_json(
                ws,
                ws_lock,
                {"type": "error", "message": "GEMINI_API_KEY is not set"},
            )
            return

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"api_version": "v1beta"},
        )
        try:
            session_ctx, session = await _connect_live(client, _live_config())
        except Exception:
            await _send_json(
                ws,
                ws_lock,
                {"type": "error", "message": "Gemini Live connection failed"},
            )
            return

        call_id = db.create_call(_ENGINE)
        recorder = audio_recorder.CallAudioRecorder(call_id)
        await _send_json(
            ws,
            ws_lock,
            {
                "type": "session_started",
                "call_id": call_id,
                "engine": _ENGINE,
                "out_sample_rate": _OUTPUT_RATE,
            },
        )

        gemini_task = asyncio.create_task(
            _gemini_loop(
                session, send_lock, ws, ws_lock, stop, call_id, recorder
            )
        )
        try:
            async with send_lock:
                await session.send_client_content(
                    turns={
                        "role": "user",
                        "parts": [{"text": _GREETING}],
                    },
                    turn_complete=True,
                )
        except Exception:
            stop.set()
            await _send_json(
                ws,
                ws_lock,
                {"type": "error", "message": "Gemini Live greeting failed"},
            )
            return

        browser_task = asyncio.create_task(
            _browser_loop(
                session, send_lock, ws, ws_lock, stop, call_id, recorder
            )
        )
        done, pending = await asyncio.wait(
            {gemini_task, browser_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
    finally:
        if gemini_task is not None or browser_task is not None:
            stop.set()
            await _stop_tasks(gemini_task, browser_task)
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if call_id is not None and recorder is not None and not finalized:
            finalized = True
            try:
                await _finalize(ws, ws_lock, call_id, recorder)
            except Exception:
                await _send_json(
                    ws,
                    ws_lock,
                    {"type": "error", "message": "Could not finish the call"},
                )
        await _close_ws(ws)
