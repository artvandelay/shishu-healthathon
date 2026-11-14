"""Phase 3 empirical probe of Gemini Live API behavior for Decision API migration.

Standalone non-test script. Synthesizes 5 spoken Hindi utterances via a helper
Gemini Live session (24kHz PCM -> 16kHz mono PCM), then streams those identical
16kHz speech turns + silence into:
  1. Tools-enabled Live session (production config, automatic VAD, 5 turns)
  2. Tool-less speaker-only silent-wait session (automatic VAD, 5 turns)
  3. Tool-less speaker-only silent-wait session (manual VAD disabled=True,
     testing no-activity_end vs with-activity_end + SYSTEM CUE playback)

Outputs structured JSON findings to code/shishu-v2/storage/live_probe_phase3.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
APP_DIR = ROOT / "code" / "shishu-v2"
load_dotenv(ROOT / ".env")
load_dotenv()

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import audio_recorder  # noqa: E402
import gemini_live  # noqa: E402
from prompts import HINDI_MEDICAL_VOCAB  # noqa: E402

MODEL = "gemini-3.1-flash-live-preview"
OUTPUT_JSON = APP_DIR / "storage" / "live_probe_phase3.json"
AUDIO_DIR = APP_DIR / "storage" / "probe_audio"

UTTERANCES = [
    ("u1_name", "मेरा नाम रेखा है"),
    ("u2_married", "हाँ, शादी हो गई है"),
    ("u3_no", "नहीं"),
    ("u4_jaundice", "हाँ, तीन दिन नीली लाइट में रखा था"),
    ("u5_emergency", "बच्चा अभी साँस नहीं ले पा रहा है"),
]

SPEAKER_ONLY_PROMPT = (
    "You are SHISHU's Hindi voice speaker. When the caller speaks, STAY COMPLETELY "
    "SILENT and produce NO audio until you receive a message starting with SYSTEM CUE. "
    "Only speak when given a SYSTEM CUE."
)


def _save_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _load_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _trim_pcm_silence(pcm_16k: bytes, rms_thresh: float = 0.012, pad_ms: int = 150) -> bytes:
    frame_bytes = 1600 * 2  # 100ms at 16kHz mono int16
    frames = [pcm_16k[i : i + frame_bytes] for i in range(0, len(pcm_16k), frame_bytes)]
    active = [idx for idx, frame in enumerate(frames) if gemini_live._pcm_rms(frame) >= rms_thresh]
    if not active:
        return pcm_16k
    pad_frames = max(1, pad_ms // 100)
    start_idx = max(0, active[0] - pad_frames)
    end_idx = min(len(frames), active[-1] + 1 + pad_frames)
    return b"".join(frames[start_idx:end_idx])


async def _synthesize_utterance(
    client: genai.Client, slug: str, phrase: str
) -> dict[str, Any]:
    wav_path = AUDIO_DIR / f"{slug}_16k.wav"
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        temperature=0.2,
        speech_config=types.SpeechConfig(
            language_code="hi-IN",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            ),
        ),
        system_instruction=types.Content(
            parts=[
                types.Part(
                    text=(
                        "You are a neutral Hindi speaker for testing. Repeat ONLY the exact "
                        "Hindi sentence given to you, clearly in one natural female voice, "
                        "with no greeting, no extra words, and no explanation."
                    )
                )
            ]
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["hi-IN"]
        ),
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )

    pcm_16k = b""
    out_text = ""
    for _attempt in range(2):
        pcm_24k = bytearray()
        out_text_chunks: list[str] = []
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await session.send_realtime_input(
                text=f"Say ONLY these exact Hindi words clearly in Hindi, nothing else: {phrase}"
            )
            deadline = time.perf_counter() + 20.0
            async for resp in session.receive():
                sc = getattr(resp, "server_content", None)
                if sc is not None:
                    if sc.output_transcription and sc.output_transcription.text:
                        out_text_chunks.append(sc.output_transcription.text)
                    if sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if getattr(part, "thought", False):
                                continue
                            audio = gemini_live._audio_from_part(part)
                            if audio is not None:
                                pcm_24k.extend(audio[0])
                    if sc.turn_complete:
                        break
                if time.perf_counter() > deadline:
                    break
        raw_16k = audio_recorder._resample_int16_mono(bytes(pcm_24k), 24000, 16000)
        pcm_16k = _trim_pcm_silence(raw_16k)
        out_text = "".join(out_text_chunks).strip()
        if len(phrase.split()) <= 1 or len(pcm_16k) >= int(0.75 * 16000 * 2):
            break

    _save_wav(wav_path, pcm_16k, 16000)
    duration_ms = int(round(len(pcm_16k) / (2 * 16)))
    return {
        "slug": slug,
        "target_phrase": phrase,
        "wav_path": str(wav_path),
        "pcm_16k_bytes": len(pcm_16k),
        "duration_ms": duration_ms,
        "rms": round(gemini_live._pcm_rms(pcm_16k), 4),
        "synth_output_transcription": out_text,
    }


async def _ensure_synthesized_utterances(
    client: genai.Client,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    meta_list: list[dict[str, Any]] = []
    pcm_by_slug: dict[str, bytes] = {}
    for slug, phrase in UTTERANCES:
        info = await _synthesize_utterance(client, slug, phrase)
        pcm, _rate = _load_wav(Path(info["wav_path"]))
        meta_list.append(info)
        pcm_by_slug[slug] = pcm
        print(
            f"[synth] {slug}: {info['duration_ms']}ms rms={info['rms']} "
            f"out={info['synth_output_transcription']!r}",
            flush=True,
        )
    return meta_list, pcm_by_slug


def _classify_chunk_mode(chunks: list[dict[str, Any]]) -> str:
    texts = [str(c.get("text") or "") for c in chunks if str(c.get("text") or "")]
    if len(texts) <= 1:
        return "single_chunk" if len(texts) == 1 else "none"
    cumulative_steps = 0
    for prev, curr in zip(texts, texts[1:]):
        p = prev.strip()
        c = curr.strip()
        if p and c.startswith(p) and len(c) > len(p):
            cumulative_steps += 1
    if cumulative_steps == len(texts) - 1:
        return "cumulative_replacement"
    if cumulative_steps > 0:
        return "mixed"
    return "incremental_delta"


async def _stream_audio_turn(
    session,
    pcm_16k: bytes,
    *,
    leading_silence_sec: float = 0.4,
    trailing_silence_sec: float = 3.5,
    chunk_ms: int = 100,
    send_activity_markers: bool = False,
    send_activity_end: bool = True,
) -> dict[str, float]:
    bytes_per_chunk = int(16000 * (chunk_ms / 1000.0) * 2)
    silence_chunk = b"\x00" * bytes_per_chunk

    if send_activity_markers:
        await session.send_realtime_input(activity_start=types.ActivityStart())

    lead_chunks = int(round(leading_silence_sec / (chunk_ms / 1000.0)))
    for _ in range(lead_chunks):
        await session.send_realtime_input(
            audio=types.Blob(data=silence_chunk, mime_type="audio/pcm;rate=16000")
        )
        await asyncio.sleep(chunk_ms / 1000.0)

    speech_start = time.perf_counter()
    for offset in range(0, len(pcm_16k), bytes_per_chunk):
        piece = pcm_16k[offset : offset + bytes_per_chunk]
        await session.send_realtime_input(
            audio=types.Blob(data=piece, mime_type="audio/pcm;rate=16000")
        )
        await asyncio.sleep(chunk_ms / 1000.0)

    speech_end = time.perf_counter()

    if send_activity_markers and send_activity_end:
        await session.send_realtime_input(activity_end=types.ActivityEnd())

    silence_chunks = int(round(trailing_silence_sec / (chunk_ms / 1000.0)))
    for _ in range(silence_chunks):
        await session.send_realtime_input(
            audio=types.Blob(data=silence_chunk, mime_type="audio/pcm;rate=16000")
        )
        await asyncio.sleep(chunk_ms / 1000.0)

    stream_done = time.perf_counter()
    return {
        "speech_start": speech_start,
        "speech_end": speech_end,
        "stream_done": stream_done,
    }


def _build_turn_record(
    *,
    turn_index: int,
    slug: str,
    target_phrase: str,
    prompt_context: str,
    speech_end_ts: float,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    in_chunks = [e for e in events if e["kind"] == "input_transcription"]
    first_audio = next((e for e in events if e["kind"] == "model_audio"), None)
    tool_calls = [e for e in events if e["kind"] == "tool_call"]
    out_chunks = [e for e in events if e["kind"] == "output_transcription"]
    turn_completes = [e for e in events if e["kind"] == "turn_complete"]
    finished_true = [e for e in in_chunks if e.get("finished") is True]

    first_in_ms = in_chunks[0]["ms_from_speech_end"] if in_chunks else None
    last_in_ms = in_chunks[-1]["ms_from_speech_end"] if in_chunks else None
    finished_ms = finished_true[0]["ms_from_speech_end"] if finished_true else None
    first_audio_ms = first_audio["ms_from_speech_end"] if first_audio else None
    first_tool_ms = tool_calls[0]["ms_from_speech_end"] if tool_calls else None
    turn_complete_ms = turn_completes[-1]["ms_from_speech_end"] if turn_completes else None

    finished_vs_audio = None
    if finished_ms is not None and first_audio_ms is not None:
        finished_vs_audio = (
            "before_model_audio" if finished_ms < first_audio_ms else "after_or_same_model_audio"
        )
    finished_vs_tool = None
    if finished_ms is not None and first_tool_ms is not None:
        finished_vs_tool = (
            "before_tool_call" if finished_ms < first_tool_ms else "after_or_same_tool_call"
        )

    chunk_mode = _classify_chunk_mode(in_chunks)
    if chunk_mode == "cumulative_replacement":
        assembled = in_chunks[-1]["text"] if in_chunks else ""
    else:
        assembled = "".join(c["text"] for c in in_chunks)

    return {
        "turn_index": turn_index,
        "slug": slug,
        "target_phrase": target_phrase,
        "prompt_context": prompt_context,
        "input_transcription_count": len(in_chunks),
        "input_chunk_mode": chunk_mode,
        "input_assembled_text": assembled,
        "input_finished_values": [c.get("finished") for c in in_chunks],
        "input_finished_ever_true": bool(finished_true),
        "first_input_transcription_ms": first_in_ms,
        "last_input_transcription_ms": last_in_ms,
        "input_finished_true_ms": finished_ms,
        "first_model_audio_ms": first_audio_ms,
        "first_tool_call_ms": first_tool_ms,
        "turn_complete_ms": turn_complete_ms,
        "finished_vs_first_model_audio": finished_vs_audio,
        "finished_vs_first_tool_call": finished_vs_tool,
        "autonomous_model_audio_emitted": first_audio is not None,
        "model_audio_chunk_count": sum(1 for e in events if e["kind"] == "model_audio"),
        "tool_calls": [{"name": e["name"], "args": e["args"], "ms": e["ms_from_speech_end"]} for e in tool_calls],
        "output_transcription": "".join(e["text"] for e in out_chunks),
        "events": events,
    }


async def _probe_tools_enabled(
    client: genai.Client, pcm_by_slug: dict[str, bytes]
) -> dict[str, Any]:
    """Session 1: production config (tools enabled, automatic VAD, 5 turns)."""
    config = gemini_live._live_config({"gemini_model": MODEL})
    turn_prompts = [
        ("u1_name", "मेरा नाम रेखा है", "नमस्ते! मैं शिशु हूँ। आपका नाम क्या है?"),
        ("u2_married", "हाँ, शादी हो गई है", "रेखा जी, क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?"),
        ("u3_no", "नहीं", "क्या जन्म के बाद बच्चे को काँच के बक्से या आईसीयू में रखना पड़ा था?"),
        ("u4_jaundice", "हाँ, तीन दिन नीली लाइट में रखा था", "क्या जन्म के बाद पीलिया के लिए नीली लाइट में रखना पड़ा था?"),
        ("u5_emergency", "बच्चा अभी साँस नहीं ले पा रहा है", "क्या बच्चे की साँस में कभी तेज़ चलना या गले-पसली धँसना देखा गया है?"),
    ]

    raw_events: list[dict[str, Any]] = []
    turn_done_event = asyncio.Event()

    async def receiver(session) -> None:
        try:
            while True:
                async for resp in session.receive():
                    ts = time.perf_counter()
                    sc = getattr(resp, "server_content", None)
                    if sc is not None:
                        if sc.interrupted:
                            raw_events.append({"kind": "interrupted", "ts": ts})
                        tr_in = sc.input_transcription
                        if tr_in is not None and (tr_in.text or getattr(tr_in, "finished", None) is not None):
                            raw_events.append(
                                {
                                    "kind": "input_transcription",
                                    "text": tr_in.text or "",
                                    "finished": getattr(tr_in, "finished", None),
                                    "start_offset": getattr(tr_in, "start_offset", None),
                                    "end_offset": getattr(tr_in, "end_offset", None),
                                    "ts": ts,
                                }
                            )
                        tr_out = sc.output_transcription
                        if tr_out is not None and (tr_out.text or getattr(tr_out, "finished", None) is not None):
                            raw_events.append(
                                {
                                    "kind": "output_transcription",
                                    "text": tr_out.text or "",
                                    "finished": getattr(tr_out, "finished", None),
                                    "ts": ts,
                                }
                            )
                        part_fcs = []
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if getattr(part, "thought", False):
                                    continue
                                audio = gemini_live._audio_from_part(part)
                                if audio is not None:
                                    raw_events.append(
                                        {
                                            "kind": "model_audio",
                                            "bytes": len(audio[0]),
                                            "rate": audio[1],
                                            "ts": ts,
                                        }
                                    )
                                fc = getattr(part, "function_call", None)
                                if fc is not None:
                                    part_fcs.append(fc)
                        if sc.turn_complete:
                            raw_events.append({"kind": "turn_complete", "ts": ts})
                            turn_done_event.set()
                    tc = getattr(resp, "tool_call", None)
                    all_fcs = list(part_fcs) if sc is not None else []
                    if tc is not None and tc.function_calls:
                        all_fcs.extend(tc.function_calls)
                    if all_fcs:
                        responses = []
                        for fc in all_fcs:
                            name = getattr(fc, "name", "")
                            args = gemini_live._call_args(fc)
                            raw_events.append(
                                {
                                    "kind": "tool_call",
                                    "id": getattr(fc, "id", ""),
                                    "name": name,
                                    "args": args,
                                    "ts": ts,
                                }
                            )
                            if getattr(fc, "id", None):
                                responses.append(
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=name,
                                        response={"result": "ok", "say": "जी, समझ गई।"},
                                    )
                                )
                        if responses:
                            await session.send_tool_response(function_responses=responses)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raw_events.append({"kind": "recv_error", "error": repr(exc), "ts": time.perf_counter()})
            turn_done_event.set()

    turns_out: list[dict[str, Any]] = []
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        recv_task = asyncio.create_task(receiver(session))
        try:
            turn_done_event.clear()
            raw_events.clear()
            await session.send_realtime_input(
                text=gemini_live._GREETING_CUE + turn_prompts[0][2]
            )
            await asyncio.wait_for(turn_done_event.wait(), timeout=15.0)
            await asyncio.sleep(0.3)

            for idx, (slug, phrase, prompt_hi) in enumerate(turn_prompts, start=1):
                if idx > 1:
                    turn_done_event.clear()
                    raw_events.clear()
                    await session.send_realtime_input(
                        text=(
                            "SYSTEM CUE (from the app, not the mother): speak only this Hindi now, "
                            "then wait. Do not call a tool.\n" + prompt_hi
                        )
                    )
                    try:
                        await asyncio.wait_for(turn_done_event.wait(), timeout=12.0)
                    except asyncio.TimeoutError:
                        pass
                    await asyncio.sleep(0.3)

                raw_events.clear()
                turn_done_event.clear()
                timings = await _stream_audio_turn(
                    session, pcm_by_slug[slug], trailing_silence_sec=3.5
                )
                try:
                    await asyncio.wait_for(turn_done_event.wait(), timeout=6.0)
                except asyncio.TimeoutError:
                    pass
                await asyncio.sleep(0.4)

                normalized: list[dict[str, Any]] = []
                for ev in raw_events:
                    copy_ev = dict(ev)
                    ts = copy_ev.pop("ts")
                    copy_ev["ms_from_speech_end"] = round((ts - timings["speech_end"]) * 1000, 1)
                    normalized.append(copy_ev)

                rec = _build_turn_record(
                    turn_index=idx,
                    slug=slug,
                    target_phrase=phrase,
                    prompt_context=prompt_hi,
                    speech_end_ts=timings["speech_end"],
                    events=normalized,
                )
                turns_out.append(rec)
                print(
                    f"[tools_enabled t{idx}] chunks={rec['input_transcription_count']} "
                    f"mode={rec['input_chunk_mode']} finished={rec['input_finished_ever_true']} "
                    f"first_in={rec['first_input_transcription_ms']}ms "
                    f"first_audio={rec['first_model_audio_ms']}ms "
                    f"first_tool={rec['first_tool_call_ms']}ms "
                    f"text={rec['input_assembled_text']!r}",
                    flush=True,
                )
        finally:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)

    return {
        "session_name": "tools_enabled_auto_vad",
        "tools_declared": ["save_profile", "record_turn"],
        "vad_mode": "automatic (1000ms, LOW/LOW)",
        "turns": turns_out,
    }


async def _run_silent_wait_probe(
    client: genai.Client,
    pcm_by_slug: dict[str, bytes],
    *,
    session_name: str,
    manual_vad: bool = False,
    send_activity_end: bool = True,
    utterance_subset: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Tool-less speaker-only silent-wait probe with auto or manual VAD."""
    if manual_vad:
        rt_cfg = types.RealtimeInputConfig(
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            ),
        )
    else:
        rt_cfg = types.RealtimeInputConfig(
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=300,
                silence_duration_ms=1000,
            ),
        )

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        temperature=0.5,
        enable_affective_dialog=False,
        speech_config=types.SpeechConfig(
            language_code="hi-IN",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            ),
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=SPEAKER_ONLY_PROMPT)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["hi-IN"],
            custom_vocabulary=list(HINDI_MEDICAL_VOCAB),
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["hi-IN"],
        ),
        realtime_input_config=rt_cfg,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )

    items = utterance_subset or UTTERANCES
    raw_events: list[dict[str, Any]] = []
    turn_done_event = asyncio.Event()

    async def receiver(session) -> None:
        try:
            while True:
                async for resp in session.receive():
                    ts = time.perf_counter()
                    sc = getattr(resp, "server_content", None)
                    if sc is not None:
                        if sc.interrupted:
                            raw_events.append({"kind": "interrupted", "ts": ts})
                        tr_in = sc.input_transcription
                        if tr_in is not None and (tr_in.text or getattr(tr_in, "finished", None) is not None):
                            raw_events.append(
                                {
                                    "kind": "input_transcription",
                                    "text": tr_in.text or "",
                                    "finished": getattr(tr_in, "finished", None),
                                    "start_offset": getattr(tr_in, "start_offset", None),
                                    "end_offset": getattr(tr_in, "end_offset", None),
                                    "ts": ts,
                                }
                            )
                        tr_out = sc.output_transcription
                        if tr_out is not None and (tr_out.text or getattr(tr_out, "finished", None) is not None):
                            raw_events.append(
                                {
                                    "kind": "output_transcription",
                                    "text": tr_out.text or "",
                                    "finished": getattr(tr_out, "finished", None),
                                    "ts": ts,
                                }
                            )
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if getattr(part, "thought", False):
                                    continue
                                audio = gemini_live._audio_from_part(part)
                                if audio is not None:
                                    raw_events.append(
                                        {
                                            "kind": "model_audio",
                                            "bytes": len(audio[0]),
                                            "rate": audio[1],
                                            "ts": ts,
                                        }
                                    )
                                fc = getattr(part, "function_call", None)
                                if fc is not None:
                                    raw_events.append(
                                        {
                                            "kind": "tool_call",
                                            "name": getattr(fc, "name", ""),
                                            "args": gemini_live._call_args(fc),
                                            "ts": ts,
                                        }
                                    )
                        if sc.turn_complete:
                            raw_events.append({"kind": "turn_complete", "ts": ts})
                            turn_done_event.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raw_events.append({"kind": "recv_error", "error": repr(exc), "ts": time.perf_counter()})
            turn_done_event.set()

    turns_out: list[dict[str, Any]] = []
    cue_speak_check: dict[str, Any] = {}

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        recv_task = asyncio.create_task(receiver(session))
        try:
            raw_events.clear()
            turn_done_event.clear()
            cue_t0 = time.perf_counter()
            await session.send_realtime_input(
                text=(
                    "SYSTEM CUE (from the app, not the mother): speak only this Hindi now, "
                    "then wait.\nनमस्ते! मैं शिशु हूँ। आपका नाम क्या है?"
                )
            )
            try:
                await asyncio.wait_for(turn_done_event.wait(), timeout=12.0)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.2)
            cue_speak_check["initial_cue_audio_chunks"] = sum(
                1 for e in raw_events if e["kind"] == "model_audio"
            )
            cue_speak_check["initial_cue_output_text"] = "".join(
                e["text"] for e in raw_events if e["kind"] == "output_transcription"
            )
            cue_speak_check["initial_cue_ms"] = round((time.perf_counter() - cue_t0) * 1000, 1)

            for idx, (slug, phrase) in enumerate(items, start=1):
                raw_events.clear()
                turn_done_event.clear()
                timings = await _stream_audio_turn(
                    session,
                    pcm_by_slug[slug],
                    trailing_silence_sec=3.5,
                    send_activity_markers=manual_vad,
                    send_activity_end=send_activity_end,
                )
                try:
                    await asyncio.wait_for(turn_done_event.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass
                await asyncio.sleep(0.3)

                normalized: list[dict[str, Any]] = []
                for ev in raw_events:
                    copy_ev = dict(ev)
                    ts = copy_ev.pop("ts")
                    copy_ev["ms_from_speech_end"] = round((ts - timings["speech_end"]) * 1000, 1)
                    normalized.append(copy_ev)

                rec = _build_turn_record(
                    turn_index=idx,
                    slug=slug,
                    target_phrase=phrase,
                    prompt_context="silent-wait after caller audio",
                    speech_end_ts=timings["speech_end"],
                    events=normalized,
                )
                turns_out.append(rec)
                print(
                    f"[{session_name} t{idx}] chunks={rec['input_transcription_count']} "
                    f"mode={rec['input_chunk_mode']} finished={rec['input_finished_ever_true']} "
                    f"auto_audio={rec['autonomous_model_audio_emitted']} "
                    f"first_in={rec['first_input_transcription_ms']}ms "
                    f"first_audio={rec['first_model_audio_ms']}ms "
                    f"text={rec['input_assembled_text']!r}",
                    flush=True,
                )

            if manual_vad and not send_activity_end:
                raw_events.clear()
                turn_done_event.clear()
                cue_t1 = time.perf_counter()
                await session.send_realtime_input(
                    text=(
                        "SYSTEM CUE (from the app, not the mother): speak only this Hindi now, "
                        "then wait.\nक्या आपकी शादी हो गई है?"
                    )
                )
                try:
                    await asyncio.wait_for(turn_done_event.wait(), timeout=8.0)
                except asyncio.TimeoutError:
                    pass
                cue_speak_check["post_no_activity_end_cue_audio_chunks"] = sum(
                    1 for e in raw_events if e["kind"] == "model_audio"
                )
                cue_speak_check["post_no_activity_end_cue_output_text"] = "".join(
                    e["text"] for e in raw_events if e["kind"] == "output_transcription"
                )
                cue_speak_check["post_no_activity_end_cue_ms"] = round((time.perf_counter() - cue_t1) * 1000, 1)
        finally:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)

    return {
        "session_name": session_name,
        "tools_declared": [],
        "vad_mode": (
            f"manual (disabled=True, send_activity_end={send_activity_end})"
            if manual_vad
            else "automatic (1000ms, LOW/LOW)"
        ),
        "cue_speak_check": cue_speak_check,
        "turns": turns_out,
    }


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    synth_meta, pcm_by_slug = await _ensure_synthesized_utterances(client)

    s1 = await _probe_tools_enabled(client, pcm_by_slug)
    s2 = await _run_silent_wait_probe(
        client,
        pcm_by_slug,
        session_name="tools_empty_silent_prompt_auto_vad",
        manual_vad=False,
    )

    sessions = [s1, s2]
    any_auto_audio = any(t["autonomous_model_audio_emitted"] for t in s2["turns"])
    if any_auto_audio:
        s3a = await _run_silent_wait_probe(
            client,
            pcm_by_slug,
            session_name="tools_empty_manual_vad_no_activity_end",
            manual_vad=True,
            send_activity_end=False,
            utterance_subset=UTTERANCES[:3],
        )
        s3b = await _run_silent_wait_probe(
            client,
            pcm_by_slug,
            session_name="tools_empty_manual_vad_with_activity_end",
            manual_vad=True,
            send_activity_end=True,
            utterance_subset=UTTERANCES[:2],
        )
        sessions.extend([s3a, s3b])

    report = {
        "model": MODEL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "synthesized_utterances": synth_meta,
        "sessions": sessions,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote probe findings to {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
