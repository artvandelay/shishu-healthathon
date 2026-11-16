"""One live intake call: Gemini Live voice puppet, Thinking VAD, and speculative Decision turns."""

from __future__ import annotations

import asyncio, base64, inspect, json, logging, os, time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from starlette.websockets import WebSocketState

from intake import audio, db, decisions, engine, extractor, reports
from intake.models import AnswerStatus, CallState, CloseReason, FollowupSlot, PackConfig, Phase, SubStep, TurnOutcome
from intake.vad import SileroDetector, ThinkingVad, VadEvent

logger = logging.getLogger(__name__)
_IN, _OUT, _TICK, _CUE_DELAY, _MAX_RECONNECTS, _MAX_TURN = 16000, 24000, 0.5, 2.2, 5, 960_000
_BARGE_IN, _GRACE, _MODEL, _BROKE = 0.65, 12.0, "gemini-3.1-flash-live-preview", object()
_COUNTERS = ("user_turns", "bot_turns", "tool_calls", "tool_errors", "audio_chunks_in", "audio_chunks_out", "interruptions", "reconnects")
_NEXT_QUESTION = {"name": "next_question", "parameters": {"type": "OBJECT", "properties": {}}, "description":
                  "Call this on every caller utterance with no arguments to fetch the exact line to speak next. Never speak to the caller before calling next_question."}
def build_live_config(pack: PackConfig, resume_handle: str = "") -> types.LiveConnectConfig:
    lang, voice = str(pack.meta.get("language_code") or "hi-IN"), str(pack.meta.get("gemini_voice") or "Kore")
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO], temperature=0.5, enable_affective_dialog=False, tools=[{"function_declarations": [_NEXT_QUESTION]}],
        speech_config=types.SpeechConfig(language_code=lang, voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        system_instruction=types.Content(parts=[types.Part(text=str(pack.prompts["system_prompt"]))]), output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[lang], custom_vocabulary=list(pack.prompts.get("vocab") or ())),
        realtime_input_config=types.RealtimeInputConfig(turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY, automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)),
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle or None), thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"))
def speak_cue(pack: PackConfig, state: CallState, kind: str = "speak") -> str:
    key = "closing" if state.phase is not Phase.ASKING else kind
    return str((pack.prompts.get("speak_cues") or {}).get(key) or "{prompt}").replace("{prompt}", engine.say_text(pack, state))
def tool_body(pack: PackConfig, state: CallState, *, clarify: bool = False) -> dict[str, Any]:
    spec, body = pack.by_id.get(state.cursor.qid), {"result": "ok", "say": engine.say_text(pack, state)}
    if state.phase is Phase.CLOSING: body["stop"] = True
    if clarify or (spec is not None and spec.shares_recorded_concerns): body["recorded_concerns"] = engine.recorded_concerns_summary(pack, state)
    return body | {"clarify": True} if clarify else body
def question_state_payload(pack: PackConfig, state: CallState) -> dict[str, Any]:
    spec, (is_yn, replies) = pack.by_id.get(state.cursor.qid), engine.quick_replies(pack, state)
    return {"type": "question_state", "step": state.step_token, "qid": state.cursor.qid, "group": spec.group if spec else "", "is_yes_no": is_yn,
            "quick_replies": replies, "prompt_chars": len(engine.say_text(pack, state))}
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
    eval_text: str = ""  # what the Decisions pass itself saw (differs from `spoken` when the STT trailed it)
    speculated: bool = False
    reheard: bool = False
Key = tuple[str, int, str]  # (step_token, voiced bytes, cleaned STT text): a turn is only cacheable on all three
Reduced = tuple[str, SubStep, int, bool]  # (answered qid, substep, followup index, redacted): the turn just committed
class LiveCallSession:
    def __init__(self, *, ws: WebSocket, pack: PackConfig, session: Any, call_id: int | None = None, state: CallState | None = None,
                 recorder: audio.CallAudioRecorder | None = None, connect: Callable[[str], Awaitable[Any]] | None = None,
                 on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None) -> None:
        timers = pack.meta.get("timers") or {}
        now = time.time()
        self.__dict__.update(
            ws=ws, pack=pack, session=session, call_id=call_id, state=state or engine.new_state(pack), recorder=recorder, connect=connect, on_call_completed=on_call_completed,
            started_at=now, last_voice_ts=now, closing_at=0.0, playback_ends_at=0.0, speech_ended_ts=0.0, resume_handle="", reconnects=0,
            hard_limit_sec=float(timers.get("hard_limit_sec") or 1200.0), soft_limit_sec=float(timers.get("soft_limit_sec") or 1140.0),
            silence_close_sec=float(timers.get("silence_close_sec") or 30.0), check_in_sec=float(timers.get("check_in_sec") or 15.0), detector=SileroDetector(),
            vad=ThinkingVad(pack.meta.get("vad")), stop=asyncio.Event(), ws_lock=asyncio.Lock(), send_lock=asyncio.Lock(), resume_requested=False, reconnecting=False,
            ending_sent=False, bot_started=False, bot_text="", clarify_step=False, tool_pending=False, checked_in=False, seen=set(), cancelled=set(),
            pcm=b"", voiced=0, text="", snapshot=None,
            task=None, task_key=None, done=None, fallback=None, last_reduced=None)
        self.last_turn_meta: dict[str, Any] = {}  # the meta row for the turn just reduced, waiting for its `tool_call` event
        self.telemetry: dict[str, Any] = dict.fromkeys(_COUNTERS, 0) | {
            "build": reports.build_info()["tag"], "model": str(pack.meta.get("gemini_model") or _MODEL), "voice": str(pack.meta.get("gemini_voice") or "Kore"), "pace_factor": 1.0,
            "turn_latencies_ms": [], "avg_latency_ms": 0, "max_latency_ms": 0, "errors": []}
    async def _send_json(self, payload: dict[str, Any]) -> None:
        async def _w() -> None:
            async with self.ws_lock: await self.ws.send_json(payload)
        await _ignore(_w())
    def _lost(self) -> bool:
        self.resume_requested = bool(self.resume_handle) and not self.stop.is_set()
        return False
    async def _send_realtime(self, **kwargs: Any) -> bool:
        if len(kwargs) != 1: raise ValueError("Gemini Live manual VAD takes exactly one keyword argument")
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
    async def _publish_and_cue(self, kind: str = "speak", *, immediate: bool = True) -> bool:
        self.bot_text, self.clarify_step = "", kind == "clarify"
        await self._send_json(question_state_payload(self.pack, self.state))
        self._cancel("fallback")
        if immediate: return await self._speak(kind)
        self.fallback = asyncio.ensure_future(self._cue_if_silent(self.state.step_token, kind))
        self.fallback.add_done_callback(consume_task_error)
        return True
    async def _db(self, fn: Any, *args: Any) -> Any: return await _try_call(asyncio.to_thread, fn, self.call_id, *args) if self.call_id is not None else None
    async def _emit(self, kind: str, detail: str = "", meta: dict | None = None) -> None:
        if kind == "error": self.telemetry["errors"] = [*self.telemetry["errors"], detail][-20:]
        evt = await self._db(db.append_call_event, kind, detail, meta) or {"t": round(time.time(), 3), "type": kind, "detail": detail, "meta": meta or {}}
        self.telemetry["last_event"] = kind
        await self._send_json({"type": "telemetry", "last_event": evt, "telemetry": dict(self.telemetry)})
        if kind == "error": await self._send_json({"type": "error", "message": detail})
    async def _persist(self, state: CallState) -> None:
        self.state = state
        if state.phase is not Phase.ASKING and not self.closing_at: self.closing_at = time.time()
        await self._db(db.save_cursor_state, self.pack, state)
        await self._db(db.update_call_telemetry, dict(self.telemetry))
    def _cancel(self, name: str) -> None:
        if (task := getattr(self, name)) is not None and not task.done():
            task.cancel()
            task.add_done_callback(consume_task_error)
        setattr(self, name, None)
    def _key(self) -> Key:
        if self.snapshot is None: self.snapshot = self.state.model_copy(deep=True)
        return self.snapshot.step_token, self.voiced, " ".join(self.text.split())
    def _reset_turn(self) -> None:
        self._cancel("task")
        self.pcm, self.voiced, self.text, self.snapshot, self.done, self.tool_pending = b"", 0, "", None, None, False
    def _start(self, *, speculate: bool = False) -> asyncio.Task:
        key = self._key()
        if self.task is not None and not self.task.done() and self.task_key[:2] == key[:2]: return self.task  # the STT may still be growing; `_banked` folds it in
        self._cancel("task")
        self.task_key, self.task = key, asyncio.ensure_future(self._evaluate_turn_task(self.snapshot, self.text, self.pcm, speculated=speculate))
        self.task.add_done_callback(lambda t: self._settle(key, speculate, t))
        return self.task
    def _settle(self, key: Key, speculate: bool, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None: return consume_task_error(task)
        self.done = (key, turn := task.result())
        if speculate and self.vad.in_turn and turn.intent.command.turn_incomplete: self.vad.extend_for_incomplete()
    def _rehear(self, turn: Turn, spoken: str) -> Turn:
        """The banked pass ran before the transcription finished; re-interpret its raw answers against the fuller text."""
        return turn._replace(intent=decisions.interpret_turn_decision(self.pack, turn.snapshot, turn.result.raw_answers, utterance_text=spoken), spoken=spoken, reheard=True)
    def _banked(self) -> Turn | None:
        """The pass for this turn, re-judged against the fuller transcription if the STT has caught up since it ran."""
        if self.done is None: return None
        key, (settled, turn) = self._key(), self.done
        if settled[:2] != key[:2]: return None
        if settled == key or not key[2] or not turn.result.ok: return turn
        self.done = (key, reheard := self._rehear(turn, key[2]))
        return reheard
    async def _turn(self) -> Turn:
        if (banked := self._banked()) is not None: return banked
        turn = await self._start()
        return self._banked() or turn
    async def _evaluate_turn_task(self, snapshot: CallState, spoken: str, pcm: bytes, *, speculated: bool = False) -> Turn:
        wav = audio.pcm16_to_wav_bytes(pcm, _IN) if pcm else None
        jobs: list[Awaitable[Any]] = [decisions.evaluate_turn(decisions.build_turn_request(self.pack, snapshot, utterance_text=spoken, audio_wav_bytes=wav, include_turn_incomplete=True))]
        if engine.step_kind(self.pack, snapshot) == "profile": jobs.append(extractor.extract_profile_entities(self.pack, snapshot, utterance_text=spoken, audio_wav_bytes=wav))
        done = await asyncio.gather(*jobs)
        return Turn(done[0], decisions.interpret_turn_decision(self.pack, snapshot, done[0].raw_answers, utterance_text=spoken), done[1] if len(done) > 1 else None, snapshot, spoken, len(pcm), wav,
                    eval_text=" ".join(spoken.split()), speculated=speculated)
    async def on_audio(self, payload: dict[str, Any]) -> bool:
        if not (pcm := audio.pcm_bytes(payload.get("data") or "")) or self.state.phase is not Phase.ASKING: return True
        raw, spec = str(payload.get("sample_rate") or ""), self.pack.by_id.get(self.state.cursor.qid)
        rate = int(raw) if (raw.isdigit() and int(raw) > 0) else _IN
        self.telemetry["audio_chunks_in"] += 1
        if self.recorder is not None: self.recorder.add_user_pcm(pcm, rate)
        pcm_16k, events, voiced = (pcm if rate == _IN else audio.resample_int16_mono(pcm, rate, _IN)), [], False
        echo, was_open = self.playback_ends_at > time.time(), self.vad.in_turn
        refl = bool(spec and spec.is_reflective_vad) or self.state.cursor.substep is SubStep.FOLLOWUP or bool(self.state.cursor.hold_slot)
        for prob in self.detector.feed_pcm16(pcm_16k):
            if echo and not self.vad.in_turn and prob < _BARGE_IN: continue
            voiced = voiced or prob >= self.vad.speech_prob_threshold
            events.extend(self.vad.on_frame(prob, is_reflective=refl))
        self.last_voice_ts = time.time()
        if was_open or VadEvent.SPEECH_START in events:
            self.pcm = (self.pcm + pcm_16k)[-_MAX_TURN:]
            self.voiced = len(self.pcm) if voiced else self.voiced
        for ev in events:
            if ev is VadEvent.SPECULATE: self._start(speculate=True)
            elif ev is VadEvent.SPEECH_START:
                if self.snapshot is not None: self._reset_turn()
                self.snapshot, self.text, self.pcm, self.voiced, self.last_reduced = self.state.model_copy(deep=True), "", pcm_16k, len(pcm_16k), None
                self.bot_started, self.tool_pending, self.checked_in, self.speech_ended_ts = False, True, False, 0.0
                self.telemetry["user_turns"] += 1
                await self._send_realtime(activity_start=types.ActivityStart())
            elif ev is VadEvent.SPEECH_END:
                self.speech_ended_ts = self.last_voice_ts = time.time()
                self.checked_in, self.tool_pending = False, True
                await self._send_realtime(activity_end=types.ActivityEnd())
                self._start()
        if not self.vad.in_turn or await self._send_realtime(audio=types.Blob(data=pcm, mime_type="audio/pcm")): return True
        await self._emit("error", "Could not forward audio to Gemini")
        return False
    async def _hear(self, role: str, text: str) -> None:
        late = role == "user" and not self.vad.in_turn and not self.tool_pending and self.last_reduced is not None
        if role == "user" and not late: self.text = engine.merge_transcript_chunk(self.text, text)
        if role == "assistant": self.bot_text = engine.merge_transcript_chunk(self.bot_text, text)
        await self._send_json({"type": "transcript", "role": role, "text": text})
        await self._db(db.append_transcript, role, text, self.last_reduced[0] if late else self.state.cursor.qid)
        if late: await self._emit("late_stt", self.last_reduced[0], {"attached": await self._attach_late_text(text), "chars": len(text)})
    async def _attach_late_text(self, text: str) -> bool:
        """A transcription chunk that trailed the reduction: keep it on the note of the question it answered."""
        qid, substep, idx, redacted = self.last_reduced
        rec, spec = self.state.answers.get(qid), self.pack.by_id.get(qid)
        if rec is None or rec.state is not AnswerStatus.CONCERN: return False
        st = self.state.model_copy(deep=True)
        if substep is SubStep.FOLLOWUP and spec is not None and idx < len(spec.followups):
            fid, slot = spec.followups[idx], st.answers[qid].followups.get(spec.followups[idx]) or FollowupSlot(state=AnswerStatus.CONCERN)
            st.answers[qid].followups[fid] = slot.model_copy(update={"note": slot.note.append(text, redacted=redacted)})
        else: st.answers[qid].note = st.answers[qid].note.append(text, redacted=redacted)
        await self._persist(st)
        return True
    async def on_text(self, payload: dict[str, Any]) -> bool:
        if not (text := " ".join(str(payload.get("text") or "").split())) or self.state.phase is not Phase.ASKING: return True
        self._reset_turn()
        self.snapshot, self.text, self.speech_ended_ts, self.last_reduced = self.state.model_copy(deep=True), text, time.time(), None
        self.last_voice_ts = self.speech_ended_ts
        self.tool_pending, self.checked_in, self.bot_started = True, False, False
        self.telemetry["user_turns"] += 1
        await self._hear("user", text)
        self._start()
        return await self._send_realtime(text=text)
    async def on_quick_reply(self, payload: dict[str, Any]) -> bool:
        val = str(payload.get("value") or payload.get("outcome") or "").strip()
        if not val or self.state.phase is not Phase.ASKING or str(payload.get("step") or "") != self.state.step_token: return True
        try: cmd = engine.quick_reply_command(self.pack, self.state, val)
        except ValueError:
            await self._emit("error", f"Invalid quick reply {val!r}")
            return True
        self._cancel("fallback")
        self._reset_turn()
        self.checked_in, self.bot_started, self.last_voice_ts, self.speech_ended_ts = False, False, time.time(), time.time()
        self.last_reduced = None
        _yn, btns = engine.quick_replies(self.pack, self.state)
        spoken = next((b.get("hi") or b.get("en") or val for b in btns if b.get("value") == val), val)
        await self._hear("user", spoken)
        await self._persist(engine.reduce_turn(self.pack, self.state, cmd))
        await self._emit("user_transcript", f"quick_reply_{val}")
        await self._send_json({"type": "clear"})
        return await self._publish_and_cue()
    async def on_nudge(self, _payload: dict[str, Any]) -> bool:
        self.bot_started = False
        if not await self._speak(): return await self._fail()
        await self._emit("manual_nudge")
        return True
    def _turn_meta(self, turn: Turn, step_in: str, cmd: Any, ext: Any) -> dict[str, Any]:
        """One compact row of scalars per turn: each surviving field answers a different post-hoc question."""
        val, d = cmd.value if cmd is not None else None, turn.intent.diagnostics
        meta = {"step_in": step_in, "step_out": self.state.step_token, "advanced": step_in != self.state.step_token, "pace": round(self.vad.pace_factor, 2),
                "outcome": cmd.outcome.value if cmd is not None else "", "top2": d.get("top2") or "", "noul": d.get("noul_hot") or "", "stated": d.get("stated") or None,
                "flags": d.get("semantic_flags") or None, "decision_ms": turn.result.decision_ms, "audio_ms": round(turn.pcm_len / 32), "tokens": turn.result.total_input_tokens,
                "value": json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else (str(val) if val is not None else ""),
                "speculated": True if turn.speculated else None, "reheard": True if turn.reheard else None, "attempts": turn.result.attempts if turn.result.attempts > 1 else None,
                "stt_eval": turn.eval_text[:120], "stt_final": turn.spoken[:120] if turn.spoken != turn.eval_text else "", "ext_ms": ext.extractor_ms if ext is not None else None}
        return {k: v for k, v in meta.items() if v is not None and v != "" and v != {} and v != []}
    async def _next_question(self) -> tuple[dict[str, Any], bool]:
        self.last_turn_meta = {}
        if self.state.phase is not Phase.ASKING or not (self.tool_pending or self.pcm or self.text): return tool_body(self.pack, self.state), False
        turn = await self._turn()
        if not turn.result.ok:
            self.telemetry["tool_errors"] += 1
            self.last_turn_meta = {"step_in": turn.snapshot.step_token, "error": turn.result.error_reason, "decision_ms": turn.result.decision_ms, "attempts": turn.result.attempts}
            self._reset_turn()
            await self._persist(engine.close_technical_failure(self.state))
            return tool_body(self.pack, self.state), False
        if turn.snapshot.step_token != self.state.step_token:
            self.last_turn_meta = {"step_in": turn.snapshot.step_token, "step_out": self.state.step_token, "reason": "stale_turn"}
            self._reset_turn()
            return {**tool_body(self.pack, self.state), "result": "held", "reason": "stale_turn"}, False
        cmd, ext, step_in = turn.intent.command, turn.extraction, turn.snapshot.step_token
        if turn.intent.needs_profile_extraction and ext is None: ext = await extractor.extract_profile_entities(self.pack, turn.snapshot, utterance_text=turn.spoken, audio_wav_bytes=turn.wav)
        if ext is not None and ext.ok and not (cmd.emergency or cmd.caller_stop): cmd = cmd.model_copy(update={"profile_patch": ext.patch, "profile_turn_attempted": True})
        self.last_reduced = (self.state.cursor.qid, self.state.cursor.substep, self.state.cursor.followup_idx, cmd.redacted)
        await self._persist(engine.reduce_turn(self.pack, self.state, cmd))
        self.telemetry["pace_factor"] = round(self.vad.pace_factor, 2)
        self.last_turn_meta = self._turn_meta(turn, step_in, cmd, ext)
        self._reset_turn()
        clarify = not (cmd.emergency or cmd.caller_stop) and cmd.outcome is TurnOutcome.CLARIFY
        return tool_body(self.pack, self.state, clarify=clarify), clarify
    async def _answer_tools(self, calls: list[Any]) -> None:
        responses, details, clarify = [], [], False
        for c in calls:
            cid, name = str(getattr(c, "id", "") or ""), str(getattr(c, "name", "") or "") or "next_question"
            if cid and (cid in self.seen or cid in self.cancelled): continue
            self.seen.add(cid)
            body, asked = (await self._next_question()) if name == "next_question" else ({"result": "ignored", "say": engine.say_text(self.pack, self.state)}, False)
            clarify = clarify or asked
            details.append(f"{name}_{body.get('result') or 'ok'}")
            responses.append(types.FunctionResponse(id=cid, name=name, response=body))
        sent = await _try_call(self._send_tools, responses, default=_BROKE) is not _BROKE
        self.telemetry["tool_calls" if sent else "tool_errors"] += 1
        if not sent: await self._emit("error", "Could not send the tool response to Gemini")
        await self._emit("tool_call", ",".join(details), self.last_turn_meta or None)
        await self._publish_and_cue(self._cue_kind(clarify), immediate=False)
    async def _send_tools(self, responses: list[Any]) -> None:
        async with self.send_lock: await self.session.send_tool_response(function_responses=responses)
    async def _cue_if_silent(self, step: str, kind: str) -> None:
        try: await asyncio.wait_for(self.stop.wait(), timeout=_CUE_DELAY)
        except asyncio.TimeoutError:
            if not self.bot_started and self.state.step_token == step and not self.stop.is_set(): await self._speak(kind)
    async def _bot_audio(self, pcm: bytes, rate: int) -> None:
        self.telemetry["audio_chunks_out"] += 1
        if self.recorder is not None: self.recorder.add_bot_pcm(pcm, rate)
        now = time.time()
        self.playback_ends_at = max(now, self.playback_ends_at) + len(pcm) / (2 * rate)
        if not self.bot_started:
            self.bot_started, self.telemetry["bot_turns"] = True, self.telemetry["bot_turns"] + 1
            self._cancel("fallback")
            self.vad.on_bot_speaking()
            if self.speech_ended_ts:
                samples = [*self.telemetry["turn_latencies_ms"], int((now - self.speech_ended_ts) * 1000)]
                self.telemetry |= {"turn_latencies_ms": samples, "avg_latency_ms": int(round(sum(samples) / len(samples))), "max_latency_ms": max(samples)}
                self.speech_ended_ts = 0.0
                await self._emit("bot_first_audio", f"{samples[-1]}ms")
        await self._send_json({"type": "audio", "sample_rate": rate, "data": base64.b64encode(pcm).decode("ascii")})
    async def on_message(self, response: Any) -> None:
        sc, calls = getattr(response, "server_content", None), []
        if sc is not None:
            if sc.interrupted:
                self.telemetry["interruptions"] += 1
                self.playback_ends_at, self.bot_started, self.bot_text = 0.0, False, ""
                self.vad.on_bot_speaking()
                await self._emit("barge_in", self.state.cursor.qid)
                await self._send_json({"type": "clear"})
            for r, p in (("user", getattr(sc, "input_transcription", None)), ("assistant", getattr(sc, "output_transcription", None))):
                if txt := " ".join(str(getattr(p, "text", "") or "").split()): await self._hear(r, txt)
            for p in (sc.model_turn.parts if sc.model_turn else None) or []:
                if not getattr(p, "thought", False) and (sp := _pcm_from_part(p)) is not None: await self._bot_audio(*sp)
                if (fc := getattr(p, "function_call", None)) is not None: calls.append(fc)
        if (tc := getattr(response, "tool_call", None)) is not None and tc.function_calls: calls.extend(tc.function_calls)
        if calls: await self._answer_tools(calls)
        if sc is not None and sc.turn_complete: await self.on_turn_complete()
        if (up := getattr(response, "session_resumption_update", None)) is not None and getattr(up, "resumable", False): self.resume_handle = str(getattr(up, "new_handle", "") or self.resume_handle)
        if getattr(response, "go_away", None) is not None: self.resume_requested = bool(self.resume_handle)
    async def on_turn_complete(self) -> None:
        if self.tool_pending and not self.bot_started:
            _body, clarify = await self._next_question()
            await self._emit("tool_call", "missed_tool", self.last_turn_meta or None)
            self.bot_started = False
            if not await self._publish_and_cue(self._cue_kind(clarify)): return await self._fail()
        if self.bot_started and not self.clarify_step:  # the voice puppet's line is known verbatim, yet Gemini drops its transcription tail
            expected = " ".join(engine.say_text(self.pack, self.state).split())
            if expected and (not self.bot_text or (expected.startswith(self.bot_text.rstrip("।?!., ")) and len(self.bot_text) < len(expected))):
                await self._hear("assistant", expected)
        self._cancel("fallback")
        self.playback_ends_at = max(self.playback_ends_at, time.time())
        await self._send_json({"type": "bot_turn_complete", "step": self.state.step_token})
        await self._emit("turn_complete")
    async def _close(self, reason: CloseReason) -> None:
        if self.state.phase is not Phase.ASKING: return
        st = self.state.model_copy(deep=True)
        st.phase, st.close_reason = Phase.CLOSING, reason
        self._reset_turn()
        self.bot_started = False
        await self._persist(st)
        await self._send_json({"type": "clear"})
        await self._publish_and_cue("closing")
        await self._emit("auto_close", reason.value)
    async def tick(self, now: float) -> bool:
        if self.resume_requested and not self.reconnecting and not await self._reconnect(): return False
        if self.state.phase is not Phase.ASKING:
            if self.ending_sent or now < self.playback_ends_at or not (self.bot_started or now - self.closing_at >= _GRACE): return True
            self.ending_sent = True
            await self._send_json({"type": "ending"})
            self.stop.set()
            return False
        elapsed = now - self.started_at
        if elapsed >= self.soft_limit_sec:
            await self._close(CloseReason.TIME_LIMIT)
            return elapsed < self.hard_limit_sec
        if self.vad.in_turn or self.tool_pending or now < self.playback_ends_at: return True
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
        self.bot_started = False
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
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=_TICK)
                return
            except asyncio.TimeoutError:
                if not await self.tick(time.time()): return
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
            await self._emit("session_started", str(self.call_id or ""), {"pack": self.pack.id})
            cue, said = str(self.pack.prompts.get("greeting_cue") or "").rstrip("\n"), engine.say_text(self.pack, self.state)
            self.bot_started, self.speech_ended_ts = False, time.time()
            await self._send_realtime(text="\n".join(b for b in (cue, said) if b).strip())
            await self._publish_and_cue(immediate=False)
            tasks.append(asyncio.ensure_future(self._browser_loop()))
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.stop.set()
            self._cancel("fallback")
            self._cancel("task")
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
async def handle_live_ws(ws: WebSocket, *, pack: PackConfig, caller_name: str = "", user_role: str = "user", clinic_code: str = "",
                         on_call_completed: Callable[[int], Awaitable[bool] | bool] | None = None) -> None:
    if ws.client_state == WebSocketState.CONNECTING: await ws.accept()
    key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
    client, ctx, live = genai.Client(api_key=key, http_options={"api_version": "v1beta"}) if key else None, None, None
    try:
        if not key: return await _reject(ws, "GEMINI_API_KEY is not set")
        try: ctx, session = await _open_live(client, pack, "")
        except Exception: return await _reject(ws, "Gemini Live connection failed")  # noqa: BLE001
        cid = await asyncio.to_thread(db.create_call, pack_id=pack.id, caller_name=caller_name or "", user_role=user_role or "user", clinic_code=clinic_code or "",
                                      model_used=str(pack.meta.get("gemini_model") or ""), voice_used=str(pack.meta.get("gemini_voice") or ""))
        live = LiveCallSession(ws=ws, pack=pack, session=session, call_id=cid, recorder=audio.CallAudioRecorder(cid, Path(db.DATA_DIR) / "audio"), on_call_completed=on_call_completed)
        async def reconnect(handle: str) -> Any:
            nonlocal ctx, session
            new_ctx, session = await _open_live(client, pack, handle)
            old, ctx = ctx, new_ctx
            await _ignore(old.__aexit__(None, None, None))
            return session
        live.connect = reconnect
        call = await asyncio.to_thread(db.get_call, cid) or {}
        await live._send_json({"type": "session_started", "call_id": cid, "caller_name": caller_name or "", "model": str(pack.meta.get("gemini_model") or ""),
                               "user_role": user_role or "user", "voice": str(pack.meta.get("gemini_voice") or ""), "out_sample_rate": _OUT,
                               "share_token": str(call.get("share_token") or "")})
        await live.run()
    finally:
        if ctx is not None: await _ignore(ctx.__aexit__(None, None, None))
        if live is not None and await _try_call(live.finalize, default=_BROKE) is _BROKE: await _reject(ws, "Could not finish the call")
        if ws.client_state == WebSocketState.CONNECTED: await _ignore(ws.close())
