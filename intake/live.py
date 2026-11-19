"""One live intake call: Gemini Live voice puppet, Gemini's own activity detection, one Decision pass per turn."""

from __future__ import annotations

import asyncio, base64, functools, inspect, json, logging, os, time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

from intake import audio, db, decisions, engine, extractor, policy, reports
from intake.models import AnswerStatus, CallState, CloseReason, FollowupSlot, PackConfig, Phase, SubStep, TurnOutcome
from intake.turns import CursorAction, DecisionReason, Disposition, EvidenceSource, PolicyDecision, TurnEvidence, TurnInterpretation

logger = logging.getLogger(__name__)
_IN, _OUT, _TICK, _MUTE_SEC, _CUE_DELAY, _STT_WAIT, _MAX_RECONNECTS, _MAX_TURN = 16000, 24000, 0.5, 0.1, 2.2, 0.5, 5, 960_000
_GRACE, _MODEL, _BROKE = 12.0, "gemini-3.1-flash-live-preview", object()
_NOISE_MIN, _NOISE_MAX, _ONSET_MIN = 0.003, 0.035, 0.012  # the adaptive room floor and the quietest onset it will accept
_RING_MAX, _PLAYBACK_TAIL_SEC = 48, 0.35  # ~3.0 s of 64 ms frames, and the last 350 ms of a turn the model already closed
_TAIL_CHUNKS, _SILENCE_100MS = 8, b"\x00\x00" * 1600  # 100 ms digital silence keeps Gemini AAD warm in half-duplex
_PROMPT_MUTE_MAX_SEC = 5.0  # the most a step may hold the mic shut waiting for audio that never comes
_MAX_CONSECUTIVE_TOOL_ERRORS = 3  # a Decision pass that dies this many times in a row is a technical failure, not a glitch
_COUNTERS = ("user_turns", "bot_turns", "tool_calls", "tool_errors", "audio_chunks_in", "audio_chunks_out", "interruptions", "reconnects")
_RESUME_GRACE_SEC = 90.0  # how long a call whose socket died waits for the browser to rejoin before it is finalized
_ACTIVE_CALLS: dict[int, "LiveCallSession"] = {}  # call id -> the session that currently owns it
_PENDING_FINALIZE: dict[int, asyncio.Task] = {}  # call id -> the grace timer that will finalize it if nobody rejoins
_NEXT_QUESTION = {"name": "next_question", "parameters": {"type": "OBJECT", "properties": {}}, "description":
                  "Call this on every caller utterance with no arguments to fetch the exact line to speak next. Never speak to the caller before calling next_question."}


class TurnPhase(StrEnum):
    """Where the server thinks the turn is, so one predicate can answer "may the mic be open?"."""
    PROMPTING = "prompting"  # a step was just published; her echo owns the channel until audio starts
    LISTENING = "listening"  # the caller owns the turn
    DECIDING = "deciding"  # a Decision pass is in flight: neither her echo nor a new turn is wanted
    CLOSING = "closing"  # the call is ending; only the closing line is owed


def build_live_config(pack: PackConfig, resume_handle: str = "") -> types.LiveConnectConfig:
    lang, voice, vad = str(pack.meta.get("language_code") or "hi-IN"), str(pack.meta.get("gemini_voice") or "Kore"), pack.meta.get("vad") or {}
    loud_s, loud_e = str(vad.get("start_sensitivity") or "LOW").upper() != "LOW", str(vad.get("end_sensitivity") or "LOW").upper() != "LOW"
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO], temperature=0.5, enable_affective_dialog=False, tools=[{"function_declarations": [_NEXT_QUESTION]}],
        speech_config=types.SpeechConfig(language_code=lang, voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        system_instruction=types.Content(parts=[types.Part(text=str(pack.prompts["system_prompt"]))]), output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[lang], custom_vocabulary=list(pack.prompts.get("vocab") or ())),
        realtime_input_config=types.RealtimeInputConfig(
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False, prefix_padding_ms=int(vad.get("prefix_padding_ms") or 200), silence_duration_ms=int(vad.get("silence_duration_ms") or 1600),
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH if loud_s else types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH if loud_e else types.EndSensitivity.END_SENSITIVITY_LOW)),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=4096,
            sliding_window=types.SlidingWindow(target_tokens=2048),
        ),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle or None), thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"))
def _reask_say(pack: PackConfig, state: CallState) -> str:
    """The line to repeat after a glitched pass: the pack's own reask for this question, else the question itself."""
    if state.phase is Phase.ASKING and state.cursor.substep is SubStep.MAIN and not state.cursor.hold_slot:
        spec = pack.by_id.get(state.cursor.qid)
        if spec is not None and spec.reask:
            return spec.reask
    return engine.say_text(pack, state)
def speak_cue(pack: PackConfig, state: CallState, kind: str = "speak") -> str:
    key = "closing" if state.phase is not Phase.ASKING else kind
    prompt = _reask_say(pack, state) if key == "retry" else engine.say_text(pack, state)
    return str((pack.prompts.get("speak_cues") or {}).get(key) or "{prompt}").replace("{prompt}", prompt)
def tool_body(pack: PackConfig, state: CallState, *, clarify: bool = False, noted_observation: str = "") -> dict[str, Any]:
    spec, body = pack.by_id.get(state.cursor.qid), {"result": "ok", "say": engine.say_text(pack, state)}
    if state.phase is Phase.CLOSING: body["stop"] = True
    if clarify or (spec is not None and spec.shares_recorded_concerns): body["recorded_concerns"] = engine.recorded_concerns_summary(pack, state)
    if noted := " ".join(noted_observation.split()): body["noted_observation"] = noted
    return body | {"clarify": True} if clarify else body
def question_state_payload(pack: PackConfig, state: CallState, epoch: int = 0) -> dict[str, Any]:
    spec, (is_yn, replies) = pack.by_id.get(state.cursor.qid), engine.quick_replies(pack, state)
    return {"type": "question_state", "step": state.step_token, "qid": state.cursor.qid, "group": spec.group if spec else "", "is_yes_no": is_yn,
            "quick_replies": replies, "prompt_chars": len(engine.say_text(pack, state)), "epoch": int(epoch)}
def _pcm_from_part(part: types.Part) -> tuple[bytes, int] | None:
    blob, mime = part.inline_data, str(getattr(part.inline_data, "mime_type", "") or "")
    pcm = audio.pcm_bytes(blob.data) if (blob is not None and blob.data is not None) else None
    return (pcm, audio.rate_from_mime(mime, _OUT)) if (pcm and mime.lower().partition("/")[0] in ("", "audio")) else None
def consume_task_error(task: asyncio.Task) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None: logger.warning("background live task failed", exc_info=exc)
async def _ignore(coro: Awaitable[Any]) -> None:
    try: await coro
    except Exception: return  # noqa: BLE001
async def _try_call(fn: Any, *args: Any, default: Any = None) -> Any:
    try:
        out = fn(*args)
        return await out if inspect.isawaitable(out) else out
    except Exception: return default  # noqa: BLE001
class Turn(NamedTuple):
    result: decisions.DecisionCallResult
    intent: decisions.TurnIntent
    extraction: extractor.ProfileExtractionResult | None
    snapshot: CallState
    spoken: str
    pcm_len: int
    wav: bytes | None
Reduced = tuple[str, SubStep, int, bool]  # (answered qid, substep, followup index, redacted): the turn just committed
class LiveCallSession:
    def __init__(self, *, ws: WebSocket, pack: PackConfig, session: Any, call_id: int | None = None, state: CallState | None = None,
                 recorder: audio.CallAudioRecorder | None = None, connect: Callable[[str], Awaitable[Any]] | None = None,
                 on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None, resumed: bool = False,
                 telemetry: Mapping[str, Any] | None = None) -> None:
        timers = pack.meta.get("timers") or {}
        now = time.time()
        self.__dict__.update(
            ws=ws, pack=pack, session=session, call_id=call_id, state=state or engine.new_state(pack), recorder=recorder, connect=connect, on_call_completed=on_call_completed,
            resumed=bool(resumed), client_hung_up=False, superseded=False,
            started_at=now, last_voice_ts=now, closing_at=0.0, playback_ends_at=0.0, speech_ended_ts=0.0, resume_handle="", reconnects=0,
            hard_limit_sec=float(timers.get("hard_limit_sec") or 1200.0), soft_limit_sec=float(timers.get("soft_limit_sec") or 1140.0),
            silence_close_sec=float(timers.get("silence_close_sec") or 30.0), check_in_sec=float(timers.get("check_in_sec") or 15.0),
            stop=asyncio.Event(), ws_lock=asyncio.Lock(), send_lock=asyncio.Lock(), resume_requested=False, reconnecting=False,
            ending_sent=False, bot_text="", cue_kind="speak", checked_in=False, seen=set(), cancelled=set(),
            pcm=b"", text="", snapshot=None, preroll=deque(maxlen=4), tail_chunks=0,
            noise_rms=0.005, noise_seeded=False, listening_ring=deque(maxlen=_RING_MAX),
            stt_ready=asyncio.Event(), tool_task=None, fallback=None, last_reduced=None, last_evidence_step="", last_tap=None, deciding_count=0,
            epoch=0, turn_phase=TurnPhase.LISTENING, prompted_at=0.0, bot_generating=False, awaiting_abort_end=False,
            closing_spoken=False, consecutive_tool_errors=0, prompt_recued=False)
        self.last_turn_meta: dict[str, Any] = {}  # the meta row for the turn just reduced, waiting for its `tool_call` event
        # A resumed call re-reads the numbers it already earned: a rejoin that starts every counter at 0 makes the
        # finished call look like it never happened (Call #32).
        seed = dict(telemetry or {})
        samples = [int(x) for x in (seed.get("turn_latencies_ms") or ()) if isinstance(x, (int, float)) and not isinstance(x, bool)]
        self.telemetry: dict[str, Any] = {k: int(seed.get(k) or 0) for k in _COUNTERS} | {
            "build": reports.build_info()["tag"], "model": str(pack.meta.get("gemini_model") or _MODEL), "voice": str(pack.meta.get("gemini_voice") or "Kore"),
            "turn_latencies_ms": samples, "avg_latency_ms": int(round(sum(samples) / len(samples))) if samples else 0,
            "max_latency_ms": max(samples) if samples else 0, "errors": [str(e) for e in (seed.get("errors") or ())][-20:]}
    async def _send_json(self, payload: dict[str, Any]) -> None:
        async def _w() -> None:
            async with self.ws_lock: await self.ws.send_json(payload)
        await _ignore(_w())
    def _lost(self) -> bool:
        self.resume_requested = bool(self.resume_handle) and not self.stop.is_set()
        return False
    async def _send_realtime(self, **kwargs: Any) -> bool:
        if len(kwargs) != 1: raise ValueError("Gemini Live takes exactly one keyword argument")
        try:
            async with self.send_lock: await self.session.send_realtime_input(**kwargs)
            return True
        except Exception: return self._lost()  # noqa: BLE001
    async def _speak(self, kind: str = "speak") -> bool: return await self._send_realtime(text=speak_cue(self.pack, self.state, kind))

    async def _fail(self, detail: str = "") -> bool:
        if detail: await self._emit("error", detail)
        self.stop.set()
        return False
    def _cue_kind(self, clarify: bool = False) -> str: return "closing" if self.state.phase is not Phase.ASKING else ("clarify" if clarify else "speak")
    def _enter_prompting(self) -> None:
        """A new step is on the wire: bump the epoch, hold the mic, and forget the turn we just left."""
        self.epoch += 1
        self.turn_phase = TurnPhase.CLOSING if self.state.phase is not Phase.ASKING else TurnPhase.PROMPTING
        self.prompted_at = time.time()
        self.bot_generating = False
        self.bot_text = ""
        self.prompt_recued = False
    def _mic_muted(self, now: float) -> bool:
        """The one answer to "could this frame be her own echo?": every reason to shut the mic lives here."""
        if self.turn_phase in (TurnPhase.DECIDING, TurnPhase.CLOSING):
            return True
        if self.bot_generating or self.awaiting_abort_end:
            return True
        # The browser holds the whole line once the model closed the turn: the last 350 ms of it is her voice.
        tail = _PLAYBACK_TAIL_SEC if self.turn_phase is TurnPhase.LISTENING else 0.0
        if now < self.playback_ends_at - tail:
            return True
        if self.turn_phase is TurnPhase.PROMPTING and (now - self.prompted_at) < _PROMPT_MUTE_MAX_SEC:
            return True
        return False
    async def _publish_and_cue(self, kind: str = "speak", *, immediate: bool = True, text: str | None = None) -> bool:
        self._enter_prompting()
        self.cue_kind = kind
        await self._send_json(question_state_payload(self.pack, self.state, self.epoch))
        self._cancel("fallback")
        # One owner per cue: a cue that is sent IS the prompt, so nothing ticks behind it to speak it again.
        if immediate: return await (self._send_realtime(text=text) if text is not None else self._speak(kind))
        if self.awaiting_abort_end: return True  # on_turn_complete arms this once Gemini has finished killing Turn A
        self.fallback = asyncio.ensure_future(self._cue_if_silent(self.state.step_token, self.epoch, kind))
        self.fallback.add_done_callback(consume_task_error)
        return True
    async def _db(self, fn: Any, *args: Any) -> Any: return await _try_call(asyncio.to_thread, fn, self.call_id, *args) if self.call_id is not None else None
    async def _emit(self, kind: str, detail: str = "", meta: dict | None = None) -> None:
        if kind == "error": self.telemetry["errors"] = [*self.telemetry["errors"], detail][-20:]
        evt = await self._db(db.append_call_event, kind, detail, meta) or {"t": round(time.time(), 3), "type": kind, "detail": detail, "meta": meta or {}}
        self.telemetry["last_event"] = kind
        await self._send_json({"type": "telemetry", "last_event": evt, "telemetry": dict(self.telemetry)})
        if kind == "error": await self._send_json({"type": "error", "message": detail})
    async def _persist(self, state: CallState, *, skip_cursor: bool = False) -> None:
        self.state = state
        if state.phase is not Phase.ASKING and not self.closing_at: self.closing_at = time.time()
        if not skip_cursor: await self._db(db.save_cursor_state, self.pack, state)
        await self._db(db.update_call_telemetry, dict(self.telemetry))
    def _ingress_evidence(self, snapshot_step: str, *, spoken: str, pcm: bytes, selected_value: Any = None,
                          source: EvidenceSource | None = None) -> TurnEvidence:
        if source is None:
            source = EvidenceSource.TAP if selected_value is not None else (EvidenceSource.VOICE if pcm else EvidenceSource.TEXT)
        return TurnEvidence(source=source, snapshot_step=snapshot_step, transcript=spoken,
                            received_at=self.speech_ended_ts or time.time(), selected_value=selected_value)
    async def _commit_turn(self, state: CallState, evidence: TurnEvidence) -> None:
        if self.call_id is None:
            self.state = state
            if state.phase is not Phase.ASKING and not self.closing_at: self.closing_at = time.time()
            return
        await asyncio.to_thread(db.commit_turn, self.call_id, self.pack, state, evidence)
        await self._persist(state, skip_cursor=True)
    async def _commit_semantic_turn(
        self,
        state: CallState,
        evidence: TurnEvidence,
        interpretation: TurnInterpretation,
        policy_decisions: tuple[PolicyDecision, ...],
    ) -> None:
        if self.call_id is None:
            self.state = state
            if state.phase is not Phase.ASKING and not self.closing_at:
                self.closing_at = time.time()
            return
        await asyncio.to_thread(
            db.commit_semantic_turn, self.call_id, self.pack, state, evidence, interpretation, policy_decisions
        )
        self.state = state
        if state.phase is not Phase.ASKING and not self.closing_at:
            self.closing_at = time.time()
        await self._persist(state, skip_cursor=True)
    async def _commit_displaced_bank(self) -> None:
        if self.turn_phase is TurnPhase.DECIDING or not (self.pcm or self.text):
            return
        step = self.snapshot.step_token if self.snapshot is not None else self.state.step_token
        state, evidence = self.state, self._ingress_evidence(step, spoken=self.text, pcm=self.pcm)
        self._reset_turn()
        await self._commit_turn(state, evidence)
    def _cancel(self, name: str) -> None:
        if (task := getattr(self, name)) is not None and not task.done():
            task.cancel()
            task.add_done_callback(consume_task_error)
        setattr(self, name, None)
    def _wants_extractor(self, snapshot: CallState) -> bool:
        """A profile step, a step that writes the profile, or a caller whose roster is still unknown all need it."""
        spec = self.pack.by_id.get(snapshot.cursor.qid)
        return (engine.step_kind(self.pack, snapshot) == "profile" or bool(spec and spec.saves_to_profile)
                or not engine.roster_resolved(self.pack, snapshot))
    def _drop_other_step_tap(self, step: str) -> None:
        in_flight = self.deciding_count or self.turn_phase is TurnPhase.DECIDING or (self.tool_task is not None and not self.tool_task.done())
        if not in_flight and self.last_tap is not None and self.last_tap[0] != step: self.last_tap = None
    def _reset_turn(self) -> None:
        self.pcm, self.text, self.snapshot, self.tail_chunks = b"", "", None, 0
        self.preroll.clear()  # a reset abandons the banked onset, so no stale chunk can seed the next turn
        self.listening_ring.clear()
        self.stt_ready.clear()
    async def _evaluate_turn_task(self, snapshot: CallState, spoken: str, pcm: bytes) -> Turn:
        wav = audio.pcm16_to_wav_bytes(pcm, _IN) if pcm else None
        # One patient pass owns the turn: the full request is the caller's only chance, and the slim slice rides
        # beside it purely as the retry body `evaluate_turn` sends after a call that really failed. Call #16 paid
        # for a 1.2k-token gate pass and then threw the whole turn away when the 4.0 s deadline hit, so the gate
        # bought nothing but the same timeout twice.
        req = decisions.build_turn_request(self.pack, snapshot, utterance_text=spoken, audio_wav_bytes=wav)
        slim = decisions.build_turn_request(self.pack, snapshot, utterance_text=spoken, audio_wav_bytes=wav, slim=True)
        jobs: list[Awaitable[Any]] = [decisions.evaluate_turn(req, fallback_request=slim)]
        if self._wants_extractor(snapshot): jobs.append(extractor.extract_profile_entities(self.pack, snapshot, utterance_text=spoken, audio_wav_bytes=wav))
        done = await asyncio.gather(*jobs)
        return Turn(done[0], decisions.interpret_turn_decision(self.pack, snapshot, done[0].raw_answers, utterance_text=spoken),
                    done[1] if len(done) > 1 else None, snapshot, spoken, len(pcm), wav)
    def _learn_noise(self, rms: float) -> None:
        """Follow the room, never the caller: a fan must become the floor, and a soft turn must never become it.

        A frame under `_ONSET_MIN` can never cross `onset_rms` — which is floored at `_ONSET_MIN` itself — so
        learning from it buys nothing and costs everything: creeping the floor toward 0.008 lifts `tail_rms` over
        the 0.007 speech that follows a single vowel, and the turn's last 12 frames die in the 8-chunk tail budget.
        """
        if not self.noise_seeded:
            self.noise_seeded = True
            if rms <= 0.03: self.noise_rms = _NOISE_MIN if rms < _ONSET_MIN else min(_NOISE_MAX, rms)
        elif rms <= self.noise_rms:  # quiet: track it down, so the floor falls back when the fan stops
            self.noise_rms = max(_NOISE_MIN, 0.85 * self.noise_rms + 0.15 * rms)
        elif rms >= _ONSET_MIN and not (self.pcm and self.text):  # loud enough to gate: the room really changed
            self.noise_rms = min(_NOISE_MAX, 0.94 * self.noise_rms + 0.06 * min(rms, _NOISE_MAX))
    def _soft_lead(self, tail_rms: float) -> bytes:
        """The contiguous soft speech before an onset, plus the quiet consonants right in front of it."""
        ring = list(self.listening_ring)
        i = len(ring)
        while i > 0 and ring[i - 1][1] >= tail_rms: i -= 1
        return b"".join(c for c, _ in ring[max(0, i - 4):]) if i < len(ring) else b"".join(self.preroll)
    async def on_audio(self, payload: dict[str, Any]) -> bool:
        if not (pcm := audio.pcm_bytes(payload.get("data") or "")) or self.state.phase is not Phase.ASKING: return True
        raw = str(payload.get("sample_rate") or "")
        rate = int(raw) if (raw.isdigit() and int(raw) > 0) else _IN
        now = time.time()
        if self._mic_muted(now):
            self.preroll.clear()  # she can tap over SHISHU but never talk over her: her own voice is not a caller turn
            self.listening_ring.clear()
            return True
        if self.turn_phase is TurnPhase.PROMPTING: self.turn_phase = TurnPhase.LISTENING  # the 5 s valve elapsed with no audio
        self.telemetry["audio_chunks_in"] += 1
        if self.recorder is not None: self.recorder.add_user_pcm(pcm, rate)
        pcm_16k = pcm if rate == _IN else audio.resample_int16_mono(pcm, rate, _IN)
        rms = audio.pcm_rms(pcm_16k)
        self._learn_noise(rms)
        onset_rms, tail_rms = max(_ONSET_MIN, self.noise_rms * 2.5), max(0.005, self.noise_rms * 1.4)
        if rms >= onset_rms:
            self.last_voice_ts, self.checked_in = time.time(), False
            if not self.pcm:
                self._cancel("fallback")  # caller started speaking; never talk over her with a delayed cue
                if self.snapshot is None: self.snapshot = self.state.model_copy(deep=True)
                self._drop_other_step_tap(self.snapshot.step_token)
                self.last_reduced = None
                self.pcm = (self._soft_lead(tail_rms) + pcm_16k)[-_MAX_TURN:]  # the consonants before the vowel are hers too
                self.preroll.clear()
                self.telemetry["user_turns"] += 1
            else:
                lead = b"".join(self.preroll) if self.tail_chunks == 0 else b""
                self.preroll.clear()
                self.pcm = (self.pcm + lead + pcm_16k)[-_MAX_TURN:]
            self.tail_chunks = _TAIL_CHUNKS
        elif self.pcm and rms >= tail_rms:
            self.pcm, self.tail_chunks = (self.pcm + pcm_16k)[-_MAX_TURN:], _TAIL_CHUNKS
        elif self.pcm and self.tail_chunks > 0:
            self.pcm, self.tail_chunks = (self.pcm + pcm_16k)[-_MAX_TURN:], self.tail_chunks - 1
        else: self.preroll.append(pcm_16k)  # a frame at the room's own floor never moves `last_voice_ts`
        self.listening_ring.append((pcm_16k, rms))
        if self.reconnecting or self.resume_requested:
            return True  # a socket swap is in flight: the frame is banked, but the session being replaced is not fed
        if await self._send_realtime(audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={rate}")): return True
        if self.reconnecting or self.resume_requested:
            # `_lost()` marks the failed socket resumable, so this blip is a dropped frame, never a dead call:
            # returning `False` here tore down `_browser_loop` in the middle of the reconnect that would have healed it.
            return True
        await self._emit("error", "Could not forward audio to Gemini")
        return False
    async def _hear(self, role: str, text: str) -> None:
        late = (role == "user" and not self.pcm and not self.text and self.turn_phase is not TurnPhase.DECIDING
                and self.last_reduced is not None)
        if role == "user" and not late:
            self.last_voice_ts, self.checked_in = time.time(), False
            if not self.pcm and not self.text:  # soft speech under the RMS gate still counts as her turn
                self._cancel("fallback")
                if self.snapshot is None: self.snapshot = self.state.model_copy(deep=True)
                self._drop_other_step_tap(self.snapshot.step_token)
                self.last_reduced, self.telemetry["user_turns"] = None, self.telemetry["user_turns"] + 1
            self.text = engine.merge_transcript_chunk(self.text, text)
            self.stt_ready.set()
        if role == "assistant":
            if self.awaiting_abort_end or self.text: return  # a caption from an aborted or un-tooled turn must not linger
            self.bot_text = engine.merge_transcript_chunk(self.bot_text, text)
        await self._send_json({"type": "transcript", "role": role, "text": text})
        await self._db(db.append_transcript, role, text, self.last_reduced[0] if late else self.state.cursor.qid)
        if late:
            attached = False
            st = self.state
            if self.last_evidence_step:
                note_st = self._state_with_late_note(text)
                if note_st is not None:
                    attached, st = True, note_st
                await self._commit_turn(st, TurnEvidence(source=EvidenceSource.LATE_STT, snapshot_step=self.last_evidence_step,
                                                         transcript=text, received_at=time.time()))
            await self._emit("late_stt", self.last_reduced[0], {"attached": attached, "chars": len(text)})
    def _state_with_late_note(self, text: str) -> CallState | None:
        """A transcription chunk that trailed the reduction: keep it on the note of the question it answered."""
        qid, substep, idx, redacted = self.last_reduced
        rec, spec = self.state.answers.get(qid), self.pack.by_id.get(qid)
        if rec is None or rec.state is not AnswerStatus.CONCERN: return None
        st = self.state.model_copy(deep=True)
        if substep is SubStep.FOLLOWUP and spec is not None and idx < len(spec.followups):
            fid, slot = spec.followups[idx], st.answers[qid].followups.get(spec.followups[idx]) or FollowupSlot(state=AnswerStatus.CONCERN)
            st.answers[qid].followups[fid] = slot.model_copy(update={"note": slot.note.append(text, redacted=redacted)})
        else: st.answers[qid].note = st.answers[qid].note.append(text, redacted=redacted)
        return st
    async def on_text(self, payload: dict[str, Any]) -> bool:
        if not (text := " ".join(str(payload.get("text") or "").split())) or self.state.phase is not Phase.ASKING: return True
        await self._commit_displaced_bank()
        self._reset_turn()
        self.snapshot, self.text, self.speech_ended_ts, self.last_reduced = self.state.model_copy(deep=True), text, time.time(), None
        self._drop_other_step_tap(self.snapshot.step_token)
        self.last_voice_ts, self.checked_in, self.bot_generating = self.speech_ended_ts, False, False
        if self.turn_phase is TurnPhase.PROMPTING: self.turn_phase = TurnPhase.LISTENING  # the caller typed, so the prompt is over
        self.telemetry["user_turns"] += 1
        await self._hear("user", text)
        return await self._send_realtime(text=text)
    async def on_quick_reply(self, payload: dict[str, Any]) -> bool:
        val = str(payload.get("value") or payload.get("outcome") or "").strip()
        if payload.get("epoch") is not None:  # a tap stamped with an older step is a double-tap, not an answer
            try: stale = int(payload["epoch"]) != self.epoch
            except (TypeError, ValueError):
                await self._emit("error", f"Invalid epoch {payload.get('epoch')!r}")
                return True
            if stale: return True
        if not val or self.state.phase is not Phase.ASKING or str(payload.get("step") or "") != self.state.step_token: return True
        try: cmd = engine.quick_reply_command(self.pack, self.state, val)
        except ValueError:
            await self._emit("error", f"Invalid quick reply {val!r}")
            return True
        self.awaiting_abort_end = self.bot_generating  # the tap flushed the browser queue; Gemini may still be mid-Turn-A
        self._cancel("fallback")
        if self.tool_task is not None and not self.tool_task.done(): self.stt_ready.set()
        await self._commit_displaced_bank()
        self._reset_turn()
        self.checked_in, self.bot_generating, self.last_voice_ts, self.speech_ended_ts = False, False, time.time(), time.time()
        self.playback_ends_at, self.last_reduced = 0.0, None
        _yn, btns = engine.quick_replies(self.pack, self.state)
        spoken = next((b.get("hi") or b.get("en") or val for b in btns if b.get("value") == val), val)
        self.text = spoken  # a tap is already a counted turn: it must not re-arm the soft-speech count in `_hear`
        step_in = self.state.step_token
        self.last_tap = (step_in, self.state.model_copy(deep=True), cmd)
        evidence = self._ingress_evidence(step_in, spoken=spoken, pcm=b"", selected_value=val)
        await self._hear("user", spoken)
        reduced = engine.reduce_turn(self.pack, self.state, cmd)
        self.last_evidence_step = step_in
        await self._commit_turn(reduced, evidence)
        self._reset_turn()  # and its label must never ride into the next Decision pass
        await self._emit("user_transcript", f"quick_reply_{val}")
        # `clear` flushes the epoch we are leaving; `_publish_and_cue` then stamps the new step with the next one.
        await self._send_json({"type": "clear", "epoch": self.epoch})
        return await self._publish_and_cue(self._cue_kind())
    async def on_nudge(self, _payload: dict[str, Any]) -> bool:
        if self.turn_phase in (TurnPhase.DECIDING, TurnPhase.CLOSING) or self.state.phase is not Phase.ASKING:
            return True  # a pass is already in flight, or the call is ending: a nudge would race the answer
        self.bot_generating = False
        # A nudge over a banked caller turn is Call #65: Gemini sat silent, so the words she already spoke were never
        # judged. Re-speaking the old question threw her answer away, so the nudge runs the pass Gemini skipped.
        if self.pcm or self.text:
            body, clarify = await self._next_question()
            await self._emit("manual_nudge", "evaluated", self.last_turn_meta or None)
            kind = "retry" if body.get("result") == "retry" else self._cue_kind(clarify)
            if not await self._publish_and_cue(kind): return await self._fail()
            return True
        # Pure silence: the published step is re-owed a cue, so its epoch, phase, and stale fallback all move together.
        if not await self._publish_and_cue("speak"): return await self._fail()
        await self._emit("manual_nudge")
        return True
    def _turn_meta(self, turn: Turn, step_in: str, cmd: Any, ext: Any) -> dict[str, Any]:
        """One compact row of scalars per turn: each surviving field answers a different post-hoc question."""
        val, d = cmd.value if cmd is not None else None, turn.intent.diagnostics
        meta = {"step_in": step_in, "step_out": self.state.step_token, "advanced": step_in != self.state.step_token,
                "outcome": cmd.outcome.value if cmd is not None else "", "top2": d.get("top2") or "", "noul": d.get("noul_hot") or "", "stated": d.get("stated") or None,
                "flags": d.get("semantic_flags") or None, "decision_ms": turn.result.decision_ms, "audio_ms": round(turn.pcm_len / 32), "tokens": turn.result.total_input_tokens,
                "value": json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else (str(val) if val is not None else ""),
                "attempts": turn.result.attempts if turn.result.attempts > 1 else None, "stt": turn.spoken[:120],
                "ext_ms": ext.extractor_ms if ext is not None else None, "slim": True if turn.result.used_slim_fallback else None}
        return {k: v for k, v in meta.items() if v is not None and v != "" and v != {} and v != []}
    def _register_failure(self, turn: Turn | None, step_in: str, error_reason: str) -> bool:
        """Count one dead pass and answer whether it is terminal: no key at all, or the strike budget spent."""
        self.telemetry["tool_errors"] += 1
        self.consecutive_tool_errors += 1
        meta = {"step_in": step_in, "error": error_reason}
        if turn is not None:
            meta |= {"attempts": turn.result.attempts if turn.result.attempts > 1 else None, "decision_ms": turn.result.decision_ms}
        terminal = error_reason == "missing_api_key" or self.consecutive_tool_errors >= _MAX_CONSECUTIVE_TOOL_ERRORS
        if terminal:
            self.turn_phase = TurnPhase.CLOSING
            self.last_turn_meta = {k: v for k, v in meta.items() if v is not None}
            return True
        self.last_turn_meta = {k: v for k, v in (meta | {"recovered": "reask", "strike": self.consecutive_tool_errors}).items() if v is not None}
        return False
    async def _recover_from_extraction(self, turn: Turn, evidence: TurnEvidence) -> tuple[dict[str, Any], bool] | None:
        """The extractor's own read of a turn Decisions could not finish: `None` when it saved nothing.

        `extractor.py` and `decisions` are two independent reads of the same utterance, and on Call #16 the
        extractor won twice (अमन 6yo male, मीरा 2yo female in ~1.5 s) while Decisions timed out. `_next_question`
        then re-asked the question the mother had already answered, in full, through the microphone.
        """
        ext = turn.extraction
        if (turn.result.error_reason == "missing_api_key" or turn.snapshot.step_token != self.state.step_token
                or ext is None or not ext.ok or (ext.patch.is_empty and ext.patch.roster_complete is None)):
            return None  # nothing was extracted, or the call has already moved on without it
        on_profile = engine.step_kind(self.pack, turn.snapshot) == "profile"
        patched = engine.apply_profile_patch(self.pack, self.state, ext.patch, attempted_on_profile=on_profile)
        if patched.step_token != self.state.step_token:
            # The patch resolved or advanced the active step: the caller's answer is in, so no re-ask is owed.
            self.consecutive_tool_errors = 0
            self.last_evidence_step = evidence.snapshot_step
            await self._commit_turn(patched, evidence)
            self.last_turn_meta = {k: v for k, v in {"step_in": turn.snapshot.step_token, "step_out": self.state.step_token,
                                                     "advanced": True, "recovered": "extractor", "decision_error": turn.result.error_reason,
                                                     "decision_ms": turn.result.decision_ms, "ext_ms": ext.extractor_ms,
                                                     "stt": turn.spoken[:120]}.items() if v is not None and v != ""}
            return tool_body(self.pack, self.state), False
        if patched != self.state:
            self.state = patched  # a mid-call profile write still belongs in memory before the evidence commit
        return None

    async def _failed_turn(self, turn: Turn | None, step_in: str, error_reason: str, *, evidence: TurnEvidence) -> tuple[dict[str, Any], bool]:
        """A pass that produced nothing: the same question is re-asked, or the call is closed for real."""
        self.last_evidence_step = evidence.snapshot_step
        if self._register_failure(turn, step_in, error_reason):
            closed = engine.close_technical_failure(self.state)
            await self._commit_turn(closed, evidence)
            # The call is being closed for real: a `next_question_ok` here told every dumpster dive the turn that
            # ended the call had answered the question (Call #16 event `[51]`), while `stop` said otherwise.
            return {**tool_body(self.pack, self.state), "result": "failed"}, False
        await self._commit_turn(self.state, evidence)
        return {"result": "retry", "say": _reask_say(self.pack, self.state)}, False
    async def _next_question(self) -> tuple[dict[str, Any], bool]:
        self.last_turn_meta = {}
        if self.state.phase is not Phase.ASKING or not (self.pcm or self.text): return tool_body(self.pack, self.state), False
        if not self.pcm and self.text and self.listening_ring:
            # Soft speakerphone speech never crossed any threshold, so its only trace is the ring: the words arrived,
            # the audio did not, and a pass handed `pcm_len == 0` is a pass that cannot hear her (Call #30).
            self.pcm = b"".join(chunk for chunk, _rms in self.listening_ring)[-_MAX_TURN:]
        snapshot = self.snapshot or self.state.model_copy(deep=True)
        spoken, pcm = self.text, self.pcm
        evidence = self._ingress_evidence(snapshot.step_token, spoken=spoken, pcm=pcm)
        self._reset_turn()
        if not self.speech_ended_ts: self.speech_ended_ts = time.time()
        self.turn_phase = TurnPhase.DECIDING
        await self._send_json({"type": "thinking", "epoch": self.epoch, "step": self.state.step_token})
        try:
            try:
                self.deciding_count += 1
                try:
                    turn = await self._evaluate_turn_task(snapshot, spoken, pcm)
                finally:
                    self.deciding_count -= 1
            except Exception:  # noqa: BLE001 — a pass that never returned is a glitch to recover from, never a dead call
                logger.exception("decision turn failed")
                self.last_turn_meta = {}
                return await self._failed_turn(None, snapshot.step_token, "exception", evidence=evidence)
            if not turn.result.ok:
                # A dead Decision pass is not a dead turn: when the extractor read this same utterance cleanly, its
                # profile/roster patch is the caller's answer and must not be thrown away to re-ask what she just
                # said. Call #16 did exactly that twice, losing अमन and मीरा both times.
                recovered = await self._recover_from_extraction(turn, evidence)
                if recovered is not None: return recovered
                return await self._failed_turn(turn, turn.snapshot.step_token, turn.result.error_reason, evidence=evidence)
            self.consecutive_tool_errors = 0
            if self.state.phase is not Phase.ASKING:  # a close won the race; only emergency speech may change why
                closed_interp = turn.intent.interpretation
                self.last_evidence_step = evidence.snapshot_step
                if closed_interp is not None and closed_interp.emergency:
                    note = " ".join(turn.spoken.split()) or self.state.emergency_note
                    upgraded = self.state.model_copy(update={"close_reason": CloseReason.EMERGENCY, "emergency_note": note})
                    await self._commit_semantic_turn(upgraded, evidence, closed_interp, (PolicyDecision(disposition=Disposition.IGNORE, reason=DecisionReason.EMERGENCY, cursor_action=CursorAction.CLOSE),))
                    return tool_body(self.pack, self.state), False
                await self._commit_turn(self.state, evidence)
                return {**tool_body(self.pack, self.state), "result": "held", "reason": "stale_turn"}, False
            if turn.snapshot.step_token != self.state.step_token:
                interp = turn.intent.interpretation
                if self.last_tap is not None and self.last_tap[0] == turn.snapshot.step_token and interp is not None:
                    _, pre_tap, tap_cmd = self.last_tap
                    self.last_tap = None
                    ext = turn.extraction
                    if turn.intent.needs_profile_extraction and ext is None:
                        ext = await extractor.extract_profile_entities(self.pack, turn.snapshot, utterance_text=turn.spoken, audio_wav_bytes=turn.wav)
                    if ext is not None and ext.ok and not (interp.emergency or interp.caller_stop):
                        interp = interp.model_copy(update={"profile_patch": ext.patch, "profile_turn_attempted": True})
                    nxt, pol, reprompt = policy.arbitrate_tap_turn(self.pack, pre_tap, tap_cmd, self.state, evidence, interp)
                    step_in = turn.snapshot.step_token
                    self.last_evidence_step = evidence.snapshot_step
                    await self._commit_semantic_turn(nxt, evidence, interp, pol.decisions)
                    asked = pre_tap.cursor
                    self.last_reduced = ((asked.qid, asked.substep, asked.followup_idx, engine._redact_note(pol.command, asked.qid))
                                         if asked.qid in self.state.answers else None)
                    if not reprompt:
                        self.last_turn_meta = {"step_in": step_in, "step_out": self.state.step_token, "reason": "tap_agreed"}
                        return {**tool_body(self.pack, self.state), "result": "held", "reason": "tap_agreed"}, False
                    await self._send_json({"type": "clear", "epoch": self.epoch})
                    clarify = self.state.phase is Phase.ASKING and any(d.disposition in (Disposition.CONFLICT, Disposition.CLARIFY) for d in pol.decisions)
                    self.last_turn_meta = {"step_in": step_in, "step_out": self.state.step_token, "reason": "tap_conflict" if clarify else "tap_enriched"}
                    return tool_body(self.pack, self.state, clarify=clarify), clarify
                self.last_evidence_step = evidence.snapshot_step
                await self._commit_turn(self.state, evidence)
                self.last_turn_meta = {"step_in": turn.snapshot.step_token, "step_out": self.state.step_token, "reason": "stale_turn"}
                return {**tool_body(self.pack, self.state), "result": "held", "reason": "stale_turn"}, False
            cmd, ext, step_in, pol = turn.intent.command, turn.extraction, turn.snapshot.step_token, None
            if turn.intent.needs_profile_extraction and ext is None:
                ext = await extractor.extract_profile_entities(self.pack, turn.snapshot, utterance_text=turn.spoken, audio_wav_bytes=turn.wav)
            interp = turn.intent.interpretation
            if interp is not None:
                if ext is not None and ext.ok and not (interp.emergency or interp.caller_stop):
                    interp = interp.model_copy(update={"profile_patch": ext.patch, "profile_turn_attempted": True})
                next_state, pol = policy.apply_policy(self.pack, self.state, evidence, interp)
                cmd = pol.command
                self.last_evidence_step = evidence.snapshot_step
                await self._commit_semantic_turn(next_state, evidence, interp, pol.decisions)
            else:
                if ext is not None and ext.ok and not (cmd.emergency or cmd.caller_stop):
                    cmd = cmd.model_copy(update={"profile_patch": ext.patch, "profile_turn_attempted": True})
                self.last_evidence_step = evidence.snapshot_step
                await self._commit_turn(engine.reduce_turn(self.pack, self.state, cmd), evidence)
            answered = turn.snapshot.cursor
            self.last_reduced = (
                answered.qid,
                answered.substep,
                answered.followup_idx,
                engine._redact_note(cmd, answered.qid),
            )
            self.last_turn_meta = self._turn_meta(turn, step_in, cmd, ext)
            stayed = (self.state.phase is Phase.ASKING and self.state.cursor.qid == answered.qid
                      and self.state.cursor.substep is answered.substep and self.state.cursor.followup_idx == answered.followup_idx)
            # A backtrack or a side note that did not answer the question still has to be spoken back. Call #67
            # Event 42 kept the uncle's diabetic blindness and then re-asked fam_cancer with a bare speak cue.
            ack = bool(pol is not None and stayed and any(
                d.disposition is Disposition.RETAIN_OBSERVATION or d.reason is DecisionReason.OTHER_FORM for d in pol.decisions))
            noted = " ".join((turn.spoken or cmd.utterance or "").split()) if ack else ""
            clarify = not (cmd.emergency or cmd.caller_stop) and (cmd.outcome is TurnOutcome.CLARIFY or ack)
            return tool_body(self.pack, self.state, clarify=clarify, noted_observation=noted), clarify
        finally:
            if self.turn_phase is TurnPhase.DECIDING:
                # A standalone caller (`_next_question` then `_hear`) must never stay wedged in DECIDING; the mic
                # stays shut on its own until `_publish_and_cue` opens the step that replaced this one.
                self.turn_phase = TurnPhase.CLOSING if self.state.phase is not Phase.ASKING else TurnPhase.PROMPTING
                self.prompted_at = time.time()
    async def _answer_tools(self, calls: list[Any], *, wait_stt: bool = False) -> None:
        step_in = self.state.step_token
        if wait_stt and self.pcm and not self.text: await _ignore(asyncio.wait_for(self.stt_ready.wait(), timeout=_STT_WAIT))
        stale = self.state.step_token != step_in and not (self.last_tap is not None and self.last_tap[0] == step_in)
        responses, details, clarify, held, retry = [], [], False, stale, False
        for c in calls:
            cid, name = str(getattr(c, "id", "") or ""), str(getattr(c, "name", "") or "") or "next_question"
            if cid and (cid in self.seen or cid in self.cancelled): continue
            self.seen.add(cid)
            body, asked = ({**tool_body(self.pack, self.state), "result": "held", "reason": "stale_turn"}, False) if (stale and name == "next_question") else (
                (await self._next_question()) if name == "next_question" else ({"result": "ignored", "say": engine.say_text(self.pack, self.state)}, False))
            clarify, held = clarify or asked, held or body.get("result") == "held"
            if body.get("stop") and self.state.close_reason is CloseReason.EMERGENCY: held = False  # replace the time-limit cue
            retry = retry or body.get("result") == "retry"
            details.append(f"{name}_{body.get('result') or 'ok'}")
            responses.append(types.FunctionResponse(id=cid, name=name, response=body))
        if not responses: return  # every call was already seen or cancelled: nothing is owed to Gemini
        sent = await _try_call(self._send_tools, responses, default=_BROKE) is not _BROKE
        self.telemetry["tool_calls" if sent else "tool_errors"] += 1
        if not sent: await self._emit("error", "Could not send the tool response to Gemini")
        await self._emit("tool_call", ",".join(details), self.last_turn_meta or None)
        if not held: await self._publish_and_cue("retry" if retry else self._cue_kind(clarify), immediate=False)
    async def _send_tools(self, responses: list[Any]) -> None:
        async with self.send_lock: await self.session.send_tool_response(function_responses=responses)
    async def _cue_if_silent(self, step: str, epoch: int, kind: str) -> None:
        try: await asyncio.wait_for(self.stop.wait(), timeout=_CUE_DELAY)
        except asyncio.TimeoutError:
            # The step and the epoch are the cue's identity: a cue owed to a step we have left must never speak.
            if (not self.bot_generating and not self.awaiting_abort_end and self.state.step_token == step
                    and self.epoch == epoch and not self.stop.is_set()):
                await self._speak(kind)
    async def _bot_audio(self, pcm: bytes, rate: int) -> None:
        if self.awaiting_abort_end or self.text: return  # an aborted tail or un-tooled reply cannot hide an open caller turn
        if self.state.phase is not Phase.ASKING: self.closing_spoken = True
        self.telemetry["audio_chunks_out"] += 1
        if self.recorder is not None: self.recorder.add_bot_pcm(pcm, rate)
        now = time.time()
        self.playback_ends_at = max(now, self.playback_ends_at) + len(pcm) / (2 * rate)
        if not self.bot_generating:
            self.bot_generating, self.telemetry["bot_turns"] = True, self.telemetry["bot_turns"] + 1
            self._cancel("fallback")
            if self.speech_ended_ts:
                samples = [*self.telemetry["turn_latencies_ms"], int((now - self.speech_ended_ts) * 1000)]
                self.telemetry |= {"turn_latencies_ms": samples, "avg_latency_ms": int(round(sum(samples) / len(samples))), "max_latency_ms": max(samples)}
                self.speech_ended_ts = 0.0
                await self._emit("bot_first_audio", f"{samples[-1]}ms")
        await self._send_json({"type": "audio", "sample_rate": rate, "data": base64.b64encode(pcm).decode("ascii"), "epoch": self.epoch})
    async def on_message(self, response: Any) -> None:
        sc, calls = getattr(response, "server_content", None), []
        if sc is not None:
            # The caller's words can ride in the SAME message as the interrupt (soft speech sends no banked PCM),
            # so they are read up front: the recovery test below would otherwise see `text == ""` and mistake a real
            # caller utterance for her own echo being cancelled.
            user_txt = " ".join(str(getattr(getattr(sc, "input_transcription", None), "text", "") or "").split())
            if sc.interrupted:  # Gemini aborted her turn: its tail and captions belong to a step we have left
                self.playback_ends_at, self.bot_generating, self.bot_text = 0.0, False, ""
                if not self.awaiting_abort_end:
                    await self._send_json({"type": "clear", "epoch": self.epoch})  # flush any partial prompt audio
                    if (self.turn_phase is TurnPhase.PROMPTING and self.state.phase is Phase.ASKING
                            and not self.pcm and not self.text and not user_txt):
                        # Gemini cancelling her own prompt echo (Call #21: `interrupted` with no caller turn and no
                        # `turn_complete` after it). Nobody barged in, so nothing else re-owed the step: rescue it.
                        await self._emit("prompt_recue", self.state.cursor.qid)
                        if not self.prompt_recued:
                            self.prompt_recued = True  # one recovery per epoch: a repeated interrupt speaks nothing
                            self._cancel("fallback")
                            self.prompted_at = time.time()
                            await self._speak(self.cue_kind)
                    else:
                        self.telemetry["interruptions"] += 1
                        await self._emit("barge_in", self.state.cursor.qid)
            for r, p in (("user", getattr(sc, "input_transcription", None)), ("assistant", getattr(sc, "output_transcription", None))):
                if txt := " ".join(str(getattr(p, "text", "") or "").split()): await self._hear(r, txt)
            for p in (sc.model_turn.parts if sc.model_turn else None) or []:
                if not getattr(p, "thought", False) and (sp := _pcm_from_part(p)) is not None: await self._bot_audio(*sp)
                if (fc := getattr(p, "function_call", None)) is not None: calls.append(fc)
        if (tc := getattr(response, "tool_call", None)) is not None and tc.function_calls: calls.extend(tc.function_calls)
        if (canc := getattr(response, "tool_call_cancellation", None)) is not None and canc.ids: self.cancelled.update(c for c in canc.ids if c)
        if calls:
            if self.tool_task is not None and not self.tool_task.done(): await self.tool_task
            if self.pcm and not self.text:
                self.stt_ready.clear()
                self.tool_task = asyncio.ensure_future(self._answer_tools(calls, wait_stt=True))
                self.tool_task.add_done_callback(consume_task_error)
            else: await self._answer_tools(calls)
        if sc is not None and sc.turn_complete: await self.on_turn_complete()
        if (up := getattr(response, "session_resumption_update", None)) is not None and getattr(up, "resumable", False): self.resume_handle = str(getattr(up, "new_handle", "") or self.resume_handle)
        if getattr(response, "go_away", None) is not None: self.resume_requested = bool(self.resume_handle)
    async def on_turn_complete(self) -> None:
        if self.tool_task is not None and not self.tool_task.done():
            if self.awaiting_abort_end: self.awaiting_abort_end, self.bot_text = False, ""  # Turn A died; the cue may arm when the pass returns
            self.stt_ready.set()
            return
        if self.awaiting_abort_end:  # the aborted turn's death: it may never unlock or cancel the step that replaced it
            self.awaiting_abort_end, self.bot_text = False, ""
            # The tap's cue can be swallowed by the abort itself, so the silence rescue may only start ticking once
            # Turn A is really dead: no timer may run against a turn Gemini has not finished killing.
            if ((self.fallback is None or self.fallback.done()) and self.turn_phase is TurnPhase.PROMPTING
                    and not self.bot_generating and self.state.phase is Phase.ASKING):
                self.fallback = asyncio.ensure_future(self._cue_if_silent(self.state.step_token, self.epoch, self.cue_kind))
                self.fallback.add_done_callback(consume_task_error)
            await self._emit("turn_complete")
            return
        # `bot_generating` is the honest answer to "did we hear a bot turn this `turn_complete` can close?": a tap on
        # A's quick reply and `sc.interrupted` both clear it, so A's death never stamps B's token into B's pills.
        heard_bot_turn = self.bot_generating
        # Gemini may skip the `next_question` call for a spoken turn; the caller's words are still owed a Decision pass.
        recovered = not heard_bot_turn and (self.pcm or self.text) and self.state.phase is Phase.ASKING
        if recovered:
            body, clarify = await self._next_question()
            await self._emit("tool_call", "missed_tool", self.last_turn_meta or None)
            kind = "retry" if body.get("result") == "retry" else self._cue_kind(clarify)
            if not await self._publish_and_cue(kind): return await self._fail()
        if heard_bot_turn:
            # The voice puppet's line is known verbatim, yet Gemini drops its transcription tail. On a `retry` cue
            # that line is the pack's reask, not `say_text`: comparing the reask against the question left Call #16
            # transcript `[2]` stuck at "जी, आराम".
            if self.cue_kind != "clarify":
                raw_expected = _reask_say(self.pack, self.state) if self.cue_kind == "retry" else engine.say_text(self.pack, self.state)
                expected = " ".join(raw_expected.split())
                if expected and (not self.bot_text or (expected.startswith(self.bot_text.rstrip("।?!., ")) and len(self.bot_text) < len(expected))):
                    await self._hear("assistant", expected)
            self.bot_generating, self.last_reduced = False, None  # the turn is closed: a repeat must not re-merge or re-unlock
            self.turn_phase = TurnPhase.CLOSING if self.state.phase is not Phase.ASKING else TurnPhase.LISTENING
            self._reset_turn()
            self._cancel("fallback")
        if heard_bot_turn or recovered:
            await self._send_json({"type": "bot_turn_complete", "step": self.state.step_token, "epoch": self.epoch})
        self.playback_ends_at = max(self.playback_ends_at, time.time())
        await self._emit("turn_complete")
    async def _close(self, reason: CloseReason) -> None:
        if self.state.phase is not Phase.ASKING: return
        self.last_tap = None
        st = self.state.model_copy(deep=True)
        st.phase, st.close_reason = Phase.CLOSING, reason
        self._reset_turn()
        self.bot_generating = False
        await self._persist(st)
        await self._send_json({"type": "clear", "epoch": self.epoch})
        await self._publish_and_cue("closing")
        await self._emit("auto_close", reason.value)
    async def tick(self, now: float) -> bool:
        if self.resume_requested and not self.reconnecting and not await self._reconnect(): return False
        if self.state.phase is not Phase.ASKING:
            if (self.ending_sent or self.bot_generating or now < self.playback_ends_at
                    or (not self.closing_spoken and now - self.closing_at < _GRACE)):
                return True
            self.ending_sent = True
            await self._send_json({"type": "ending"})
            self.stop.set()
            return False
        elapsed = now - self.started_at
        if elapsed >= self.soft_limit_sec:
            await self._close(CloseReason.TIME_LIMIT)
            return elapsed < self.hard_limit_sec
        if self.turn_phase is TurnPhase.PROMPTING and (now - self.prompted_at) >= _PROMPT_MUTE_MAX_SEC and now >= self.playback_ends_at:
            self.turn_phase = TurnPhase.LISTENING  # a silent model: the step is hers now, mic and all
        # The silence timers count only while she owns an open turn: never over her own voice, never during a pass.
        if self.turn_phase is not TurnPhase.LISTENING or self._mic_muted(now): return True
        quiet = now - max(self.last_voice_ts, self.playback_ends_at)
        if quiet >= self.silence_close_sec: await self._close(CloseReason.SILENCE)
        elif quiet >= self.check_in_sec and not self.checked_in:
            self.checked_in = True
            await self._send_realtime(text=str(self.pack.prompts.get("silence_check_in_cue") or ""))
            await self._emit("silence_check_in", f"{int(quiet)}s")
        return True
    async def _reconnect(self) -> bool:
        self.resume_requested, self.reconnecting, self.reconnects = False, True, self.reconnects + 1
        await self._send_json({"type": "reconnecting", "attempt": self.reconnects})
        sess = await _try_call(self.connect, self.resume_handle) if (self.connect is not None and self.reconnects <= _MAX_RECONNECTS) else None
        self.reconnecting = False
        if sess is None: return await self._fail("Gemini Live could not reconnect")
        self.session, self.telemetry["reconnects"] = sess, self.reconnects
        self.seen.clear()
        self.cancelled.clear()
        await self._send_json({"type": "reconnected", "attempt": self.reconnects})
        await self._emit("reconnected", str(self.reconnects))
        self._enter_prompting()
        return await self._speak()
    async def _pump(self) -> None:
        async for resp in self.session.receive():
            if self.stop.is_set(): return
            await self.on_message(resp)
            if self.resume_requested: return
    async def _gemini_loop(self) -> None:
        while not self.stop.is_set():
            if await _try_call(self._pump, default=_BROKE) is _BROKE:
                if self.stop.is_set(): return
                self.resume_requested = bool(self.resume_handle)
                if not self.resume_requested: return await self._fail("Gemini Live session ended")
            if self.resume_requested and not await self._reconnect(): return
    async def _watchdog(self) -> None:
        last_tick = time.time()
        while not self.stop.is_set():
            now = time.time()
            speaking = self.state.phase is Phase.ASKING and self._mic_muted(now) and self.turn_phase is not TurnPhase.DECIDING
            if speaking and not self.reconnecting:
                await self._send_realtime(audio=types.Blob(data=_SILENCE_100MS, mime_type=f"audio/pcm;rate={_IN}"))
            if now - last_tick >= _TICK - 0.02:
                last_tick = now
                if not await self.tick(now): return
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=_MUTE_SEC if speaking else _TICK)
                return
            except asyncio.TimeoutError:
                continue
    async def _browser_loop(self) -> None:
        handlers = {"audio": self.on_audio, "text": self.on_text, "quick_reply": self.on_quick_reply, "nudge": self.on_nudge}
        while not self.stop.is_set():
            try: payload = json.loads(await self.ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                self.stop.set()
                return
            except json.JSONDecodeError:
                await self._emit("error", "Invalid JSON")
                continue
            if not isinstance(payload, dict):
                await self._emit("error", "Expected a JSON object")
                continue
            if payload.get("type") == "hangup":
                self.client_hung_up = True
                self.stop.set()
                return
            if (h := handlers.get(str(payload.get("type") or ""))) is not None and not await h(payload): return
    async def finalize(self) -> None:
        if self.state.phase is Phase.ASKING:  # a hangup mid-question is a partial call, never a complete one
            st = self.state.model_copy(deep=True)
            st.phase, st.close_reason = Phase.CLOSING, CloseReason.CALLER_STOP
            await self._persist(st)
        saved = await _try_call(asyncio.to_thread, self.recorder.save) if self.recorder is not None else None
        filename, duration = saved or ("", 0)
        if not self.ending_sent:
            self.ending_sent = True
            await self._send_json({"type": "ending"})
        await self._db(db.update_call_telemetry, dict(self.telemetry))
        if self.call_id is None: return
        await asyncio.to_thread(db.complete_call, self.call_id, int(duration), filename)
        ready = bool(await _try_call(self.on_call_completed, self.call_id, default=False)) if self.on_call_completed is not None else False
        call = await asyncio.to_thread(db.get_call, self.call_id) or {}
        await self._send_json({"type": "call_completed" if ready else "report_failed", "call_id": self.call_id, "share_token": str(call.get("share_token") or "")})
    async def run(self) -> None:
        tasks = [asyncio.ensure_future(self._gemini_loop()), asyncio.ensure_future(self._watchdog())]
        try:
            await self._persist(self.state)
            if self.resumed:
                # A rejoin asks for the question the call paused on: the greeting belongs to a brand-new call.
                await self._emit("session_resumed", str(self.call_id or ""), {"pack": self.pack.id, "qid": self.state.cursor.qid})
                # No `speech_ended_ts` here: a resume prompt is the server's, not a caller turn, so its cue-to-audio
                # gap is not a turn latency and must never be appended to the samples the call already earned.
                await self._publish_and_cue("speak")
            else:
                await self._emit("session_started", str(self.call_id or ""), {"pack": self.pack.id})
                cue, said = str(self.pack.prompts.get("greeting_cue") or "").rstrip("\n"), engine.say_text(self.pack, self.state)
                self.speech_ended_ts = time.time()
                await self._publish_and_cue("speak", text="\n".join(b for b in (cue, said) if b).strip())
            tasks.append(asyncio.ensure_future(self._browser_loop()))
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if not self.stop.is_set():  # a loop that dies on its own is a bug, never a quiet hangup
                for t in done:
                    if (t.cancelled() or (exc := t.exception()) is not None):
                        logger.error("live loop exited unexpectedly", exc_info=None if t.cancelled() else exc)
                        await self._emit("error", "Live loop exited unexpectedly")
                        break
        finally:
            self.stop.set()
            self._cancel("fallback")
            self._cancel("tool_task")
            for t in tasks:
                if not t.done(): t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
async def _reject(ws: WebSocket, msg: str) -> None:
    if ws.client_state == WebSocketState.CONNECTED: await _ignore(ws.send_json({"type": "error", "message": msg}))
async def _open_live(client: genai.Client, pack: PackConfig, handle: str) -> tuple[Any, Any]:
    model = str(pack.meta.get("gemini_model") or _MODEL)
    for attempt in range(3):
        try:
            ctx = client.aio.live.connect(model=model, config=build_live_config(pack, handle))
            return ctx, await ctx.__aenter__()
        except Exception:
            if attempt == 2: raise
            await asyncio.sleep(0.4 * (attempt + 1))
    raise RuntimeError("Gemini Live setup did not complete")
def _schedule_delayed_finalize(cid: int, session_obj: LiveCallSession) -> None:
    """Hold a dropped call open for the resume grace, then finalize it if no browser rejoins."""
    if (prior := _PENDING_FINALIZE.pop(cid, None)) is not None:
        prior.cancel()

    async def _finalize_after_grace() -> None:
        await asyncio.sleep(_RESUME_GRACE_SEC)
        if _PENDING_FINALIZE.get(cid) is not asyncio.current_task():
            return  # a rejoin or a newer grace timer already took this call over
        if cid in _ACTIVE_CALLS:
            return  # somebody is driving the call again
        _PENDING_FINALIZE.pop(cid, None)
        await _try_call(session_obj.finalize)

    task = asyncio.ensure_future(_finalize_after_grace())
    task.add_done_callback(consume_task_error)
    _PENDING_FINALIZE[cid] = task


async def handle_live_ws(ws: WebSocket, *, pack: PackConfig, caller_name: str = "", user_role: str = "user", clinic_code: str = "",
                         resume_token: str = "", on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None) -> None:
    if ws.client_state == WebSocketState.CONNECTING: await ws.accept()
    key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
    client, ctx, live_sess = genai.Client(api_key=key, http_options={"api_version": "v1beta"}) if key else None, None, None
    cid: int | None = None
    try:
        if not key: return await _reject(ws, "GEMINI_API_KEY is not set")
        try: ctx, session = await _open_live(client, pack, "")
        except Exception: return await _reject(ws, "Gemini Live connection failed")  # noqa: BLE001
        tok = (resume_token or "").strip()
        row = await asyncio.to_thread(db.get_call_by_token, tok) if tok else None
        saved_state = db.call_state_from_row(row) if row is not None else CallState()
        still_open = saved_state.close_reason not in (CloseReason.COMPLETE, CloseReason.EMERGENCY)
        if row is not None and row.get("pack_id") == pack.id and still_open:
            # The browser came back on a call we still hold: the row, the cursor, and the audio file are reused.
            cid = int(row["id"])
            if (pending := _PENDING_FINALIZE.pop(cid, None)) is not None: pending.cancel()
            old = _ACTIVE_CALLS.pop(cid, None)
            if old is not None:
                # A half-open session whose socket died unnoticed: it must never finalize the call it lost.
                old.superseded = True
                old.stop.set()
                if old.recorder is not None: await _try_call(asyncio.to_thread, old.recorder.save)
            restored = saved_state.model_copy(deep=True)
            restored.phase, restored.close_reason = Phase.ASKING, None
            # The step this rejoin will re-speak is the one whose caption the dropped socket already wrote down.
            await asyncio.to_thread(db.reopen_call, cid, restored.cursor.qid)
            # The row and the session that lost the call each hold part of what this call has already earned: take
            # the higher count, the longer latency list, and every error either of them saw (Call #32 lost all of it).
            row_tel, old_tel = row.get("telemetry") or {}, old.telemetry if old is not None else {}
            prior_tel = {k: max(int(row_tel.get(k) or 0), int(old_tel.get(k) or 0)) for k in _COUNTERS} | {
                "turn_latencies_ms": max((old_tel.get("turn_latencies_ms") or []), (row_tel.get("turn_latencies_ms") or []), key=len),
                "errors": list(dict.fromkeys([*(row_tel.get("errors") or ()), *(old_tel.get("errors") or ())]))[-20:]}
            live_sess = LiveCallSession(ws=ws, pack=pack, session=session, call_id=cid, state=restored,
                                        recorder=audio.CallAudioRecorder(cid, Path(db.DATA_DIR) / "audio", append_existing=True),
                                        on_call_completed=on_call_completed, resumed=True, telemetry=prior_tel)
        else:
            cid = await asyncio.to_thread(db.create_call, pack_id=pack.id, caller_name=caller_name or "", user_role=user_role or "user",
                                          clinic_code=clinic_code or "", model_used=str(pack.meta.get("gemini_model") or ""),
                                          voice_used=str(pack.meta.get("gemini_voice") or ""))
            live_sess = LiveCallSession(ws=ws, pack=pack, session=session, call_id=cid, recorder=audio.CallAudioRecorder(cid, Path(db.DATA_DIR) / "audio"),
                                        on_call_completed=on_call_completed)
        _ACTIVE_CALLS[cid] = live_sess

        async def reconnect(handle: str) -> Any:
            nonlocal ctx, session
            new_ctx, session = await _open_live(client, pack, handle)
            old, ctx = ctx, new_ctx
            await _ignore(old.__aexit__(None, None, None))
            return session
        live_sess.connect = reconnect
        call = await asyncio.to_thread(db.get_call, cid) or {}
        await live_sess._send_json({"type": "session_started", "call_id": cid, "caller_name": str(call.get("caller_name") or caller_name or ""),
                                    "model": str(pack.meta.get("gemini_model") or ""), "user_role": user_role or "user",
                                    "voice": str(pack.meta.get("gemini_voice") or ""), "out_sample_rate": _OUT,
                                    "share_token": str(call.get("share_token") or "")})
        await live_sess.run()
    finally:
        if ctx is not None: await _ignore(ctx.__aexit__(None, None, None))
        if live_sess is not None and cid is not None:
            # A dropped socket is very likely to be resumed FROM THE ROW, so the numbers this session earned must be
            # on that row before the call is let go. `_bot_audio` appends a latency sample and nothing persists after
            # it, so flushing only on the way into grace raced the rejoin and lost the sample (Call #47). An evicted
            # session is excluded: it must never overwrite the telemetry of the session that now owns the call.
            dropped = (not live_sess.superseded and not live_sess.client_hung_up
                       and live_sess.state.phase is Phase.ASKING and _RESUME_GRACE_SEC > 0)
            if dropped: await live_sess._db(db.update_call_telemetry, dict(live_sess.telemetry))
            if _ACTIVE_CALLS.get(cid) is live_sess:
                _ACTIVE_CALLS.pop(cid, None)
            if live_sess.superseded:
                if live_sess.recorder is not None: await _try_call(asyncio.to_thread, live_sess.recorder.save)
            elif dropped:
                # An unexpected socket death is a dropped line, never a hangup: hold the call, then resume or finalize.
                if live_sess.recorder is not None:
                    saved = await _try_call(asyncio.to_thread, live_sess.recorder.save)
                    if saved and saved[0]:
                        # The frames so far must survive the gap: the resumed recorder appends to this same file.
                        await _try_call(asyncio.to_thread, functools.partial(db._write, cid, audio_path=saved[0], duration_sec=int(saved[1])))
                _schedule_delayed_finalize(cid, live_sess)
            else:
                if await _try_call(live_sess.finalize, default=_BROKE) is _BROKE: await _reject(ws, "Could not finish the call")
        if ws.client_state == WebSocketState.CONNECTED: await _ignore(ws.close())
