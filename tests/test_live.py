"""Behavioral offline tests for `intake.live`: fake Gemini Live session, fake WebSocket, no network."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import unittest
import unittest.mock
from typing import Any

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`
from intake import db, live
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CloseReason, CursorStep, Phase,
    ProfilePatch, RosterItem, RosterInput, SubStep, TurnCommand, TurnOutcome,
)
from intake.pack import load_pack
from starlette.websockets import WebSocketState

PACK = load_pack("shishu-hi")
VOICED = (1200).to_bytes(2, "little") * 640  # rms 0.037, safely over the 0.015 voice gate


def pcm(blob: bytes = VOICED) -> str:
    """One mic frame as the browser would send it."""
    return base64.b64encode(blob).decode()


def answers(**heads: Any) -> dict[str, Any]:
    """A Decisions response body: `noul` heads as floats, `choice` heads as distributions."""
    body: dict[str, Any] = {}
    for name, value in heads.items():
        body[name] = ({"probabilities": value} if isinstance(value, dict)
                      else {"noul": value})
    return {"answers": body, "usage": {"total_input_tokens": 12}}


class FakeSession:
    """A Gemini Live session: records every realtime send and tool response it receives."""

    def __init__(self, messages: list | None = None, *, fail_realtime: bool = False) -> None:
        self.sent: list[tuple[str, Any]] = []
        self.messages = list(messages or [])
        self.fail_realtime = fail_realtime
        self.setup_complete = True

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent.append(("realtime", kwargs))
        if self.fail_realtime:
            raise RuntimeError("live send failed")

    async def send_tool_response(self, function_responses: list) -> None:
        self.sent.append(("tool", function_responses))

    async def receive(self):
        for message in self.messages:
            yield message
        await asyncio.Event().wait()  # an exhausted script simply goes quiet

    def kinds(self) -> list[str]:
        return [next(iter(payload)) if kind == "realtime" else kind for kind, payload in self.sent]

    def realtime(self, key: str) -> list[Any]:
        return [payload[key] for kind, payload in self.sent if kind == "realtime" and key in payload]


class Feed(FakeSession):
    """A Gemini session whose `receive` is fed live, so `_gemini_loop` can be driven mid-conversation."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: asyncio.Queue = asyncio.Queue()

    async def push(self, message: Any) -> None:
        await self.queue.put(message)

    async def receive(self):
        while (message := await self.queue.get()) is not None:
            yield message


class FakeWebSocket:
    """A FastAPI-shaped socket fed from a queue, so the browser loop runs in a real task."""

    def __init__(self) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[dict[str, Any]] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def accept(self) -> None:
        self.client_state = WebSocketState.CONNECTED

    async def close(self) -> None:
        self.closed = True
        self.client_state = WebSocketState.DISCONNECTED

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def receive_text(self) -> str:
        item = await self.queue.get()
        if item is None:
            raise live.WebSocketDisconnect()
        return item if isinstance(item, str) else json.dumps(item)

    async def push(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [payload for payload in self.sent if payload.get("type") == kind]


class FakeRecorder:
    def __init__(self) -> None:
        self.user: list[tuple[bytes, int]] = []
        self.bot: list[tuple[bytes, int]] = []
        self.saves = 0

    def add_user_pcm(self, pcm: bytes, rate: int) -> None:
        self.user.append((pcm, rate))

    def add_bot_pcm(self, pcm: bytes, rate: int) -> None:
        self.bot.append((pcm, rate))

    def save(self) -> tuple[str, int]:
        self.saves += 1
        return "call_1.wav", 3


def build(*, messages: list | None = None, fail_realtime: bool = False, call_id: int | None = 1,
          state: CallState | None = None) -> tuple[live.LiveCallSession, FakeWebSocket, FakeSession]:
    """One session wired to fakes."""
    ws, session = FakeWebSocket(), FakeSession(messages, fail_realtime=fail_realtime)
    built = live.LiveCallSession(
        ws=ws, pack=PACK, session=session, call_id=call_id, state=state,
        recorder=FakeRecorder(), on_call_completed=lambda _call_id: True)
    return built, ws, session


def turn_with(state: CallState, outcome: TurnOutcome, spoken: str) -> live.Turn:
    """A turn as decisions would have produced it: raw heads that re-interpret to `outcome` for `spoken`."""
    rivals = {o.value: 0.03 / 5 for o in (TurnOutcome.CLEAR, TurnOutcome.CONCERN, TurnOutcome.UNKNOWN, TurnOutcome.CLARIFY, TurnOutcome.RETRACT) if o is not outcome}
    body = answers(danger_now=0.01, active_outcome={outcome.value: 0.97, **rivals})
    intent = live.decisions.interpret_turn_decision(PACK, state, body["answers"], utterance_text=spoken)
    return live.Turn(TurnTests.result(body), intent, None, state.model_copy(deep=True), spoken, 640, None)


class OutputTranscriptionTests(unittest.TestCase):
    """Gemini drops the trailing `output_transcription` chunks; the puppet's known line must still land."""

    LINE = "क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?"

    def setUp(self) -> None:
        self.calls: list[int] = []

    def tearDown(self) -> None:
        with db._LOCK:  # the real call rows are this suite's own; leave the shared database as it was
            for call_id in self.calls:
                db.conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
            db.conn.commit()

    @staticmethod
    def said(text: str) -> live.types.LiveServerMessage:
        return live.types.LiveServerMessage(server_content=live.types.LiveServerContent(
            output_transcription=live.types.Transcription(text=text)))

    @staticmethod
    def audio() -> live.types.LiveServerMessage:
        blob = live.types.Blob(data=b"\x00\x00" * 480, mime_type="audio/pcm;rate=24000")
        return live.types.LiveServerMessage(server_content=live.types.LiveServerContent(
            model_turn=live.types.Content(parts=[live.types.Part(inline_data=blob)])))

    @staticmethod
    def complete() -> live.types.LiveServerMessage:
        return live.types.LiveServerMessage(server_content=live.types.LiveServerContent(turn_complete=True))

    @staticmethod
    def interrupted() -> live.types.LiveServerMessage:
        return live.types.LiveServerMessage(server_content=live.types.LiveServerContent(interrupted=True))

    async def session_at(self, qid: str) -> tuple[live.LiveCallSession, FakeWebSocket, int]:
        """A live session holding on `qid`, with a real call row so the transcript really persists."""
        call_id = await asyncio.to_thread(db.create_call, pack_id=PACK.id)
        self.calls.append(call_id)
        state = live.engine.new_state(PACK)
        state.cursor = state.cursor.model_copy(update={"qid": qid})
        built, ws, _session = build(call_id=call_id, state=state)
        return built, ws, call_id

    @staticmethod
    async def stored(call_id: int, role: str) -> list[str]:
        call = await asyncio.to_thread(db.get_call, call_id)
        return [t["text"] for t in (call or {}).get("transcript") or [] if t.get("role") == role]

    def test_a_two_word_prefix_is_completed_to_the_full_line_on_turn_complete(self):
        async def body() -> None:
            built, ws, call_id = await self.session_at("intro_married")
            self.assertEqual(live.engine.say_text(PACK, built.state), self.LINE)
            await built._publish_and_cue("speak", immediate=False)
            await built.on_message(self.said("क्या आपकी"))
            await built.on_message(self.audio())
            await built.on_message(self.complete())
            self.assertEqual(ws.of("bot_turn_complete"),
                             [{"type": "bot_turn_complete", "step": built.state.step_token, "epoch": built.epoch}],
                             "the heard bot turn closes once, unlocking this step's quick replies")
            self.assertEqual(built.bot_text, self.LINE, "the session now tracks the whole spoken line")
            captions = [p["text"] for p in ws.of("transcript") if p["role"] == "assistant"]
            self.assertEqual(captions, ["क्या आपकी", self.LINE], "the browser gets the complete sentence too")
            self.assertEqual(await self.stored(call_id, "assistant"), [self.LINE], "and SQLite holds one merged turn")
            await built.on_message(self.complete())
            self.assertEqual(await self.stored(call_id, "assistant"), [self.LINE], "a repeated turn_complete never doubles the line")
            self.assertEqual(len(ws.of("bot_turn_complete")), 1, "nor does it unlock the step a second time")

        asyncio.run(body())


class ConfigTests(unittest.TestCase):
    """The one Live config: Gemini's own activity detection, one tool, one Hindi voice, minimal thinking."""

    def test_resume_handle_is_optional_and_forwarded(self):
        self.assertIsNone(live.build_live_config(PACK).session_resumption.handle)
        self.assertEqual(live.build_live_config(PACK, "abc").session_resumption.handle, "abc")

    def test_live_config_bounds_context_window_sliding_target(self):
        # A 56-question call re-sends a `SYSTEM CUE` prompt on every step. With no trigger and no target the KV
        # cache grows all call, so the late turns pay for every cue ever spoken in latency and drift. The window
        # must be bounded: compress once the context passes 4096 tokens, keeping 2048 tokens of it.
        cfg = live.build_live_config(PACK)
        self.assertEqual(cfg.context_window_compression.trigger_tokens, 4096,
                         "an unbounded trigger never compresses, which is the bug")
        self.assertEqual(cfg.context_window_compression.sliding_window.target_tokens, 2048,
                         "the sliding window needs an explicit retained-token target")

    def test_geminis_automatic_activity_detection_carries_the_packs_vad_defaults(self):
        # A 1000 ms end-of-speech window cut Hindi mid-sentence: Call #44's 1.4–1.5 s pauses closed the turn
        # early, so the model answered half a sentence and the `DECIDING` mute swallowed the rest of it. The same
        # 100 ms prefix dropped the consonants in front of a first syllable. Both packs must carry the widened
        # window, so this sweeps every real pack instead of trusting the one the suite happens to import.
        for pack_id in ("shishu-hi", "fever-opd-hi"):
            with self.subTest(pack=pack_id):
                aad = live.build_live_config(load_pack(pack_id)).realtime_input_config.automatic_activity_detection
                self.assertIs(aad.disabled, False, "the model, not a local ONNX graph, owns the turn boundary")
                self.assertEqual(aad.silence_duration_ms, 1600,
                                 f"{pack_id}: 1.4 s–1.5 s Hindi mid-sentence pauses must finish inside the window "
                                 "instead of buying a premature DECIDING mute (Call #44)")
                self.assertEqual(aad.prefix_padding_ms, 200,
                                 f"{pack_id}: the onset padding keeps the consonants before the first vowel "
                                 "(300 ms dropped brisk 400 ms \"नहीं\" replies, `u3_no_16k.wav`)")

    def test_tool_body_carries_the_line_and_clarify_flag(self):
        state = live.engine.new_state(PACK)
        self.assertEqual(live.tool_body(PACK, state)["say"],
                         live.engine.say_text(PACK, state))
        clear = live.tool_body(PACK, state, clarify=True)
        self.assertTrue(clear["clarify"])
        self.assertEqual(clear["recorded_concerns"], [])

    def test_asr_vocabulary_carries_the_twins_and_cousin_words(self):
        # Gemini's Hindi ASR writes "twins" and "cousin" as the English loans the caller actually says.
        vocab = list(live.build_live_config(PACK).input_audio_transcription.custom_vocabulary or ())
        self.assertIn("twins", vocab)
        self.assertIn("cousin", vocab)


class PrerollTests(unittest.TestCase):
    """A caller's first frame above the RMS gate is rarely her first frame: the silent onset must survive."""

    @staticmethod
    def chunk(sample: int) -> bytes:
        return (sample.to_bytes(2, "little")) * 640  # 640 samples of 16 kHz int16 mono

    def test_silent_frames_before_the_first_voiced_one_open_the_turn(self):
        async def body() -> None:
            built, _ws, session = build()
            silent, voiced = self.chunk(0), self.chunk(1200)
            for pcm in (silent, silent, silent, voiced):
                await built.on_audio({"data": base64.b64encode(pcm).decode(), "sample_rate": 16000})
            self.assertEqual(built.pcm, silent * 3 + voiced, "the consonants before the vowel stay in the turn's audio")
            self.assertEqual(built.telemetry["user_turns"], 1, "one voiced onset is one caller turn")
            self.assertEqual(session.kinds(), ["audio"] * 4, "every frame still reaches Gemini, in the order spoken")
            self.assertEqual([b.data for b in session.realtime("audio")], [silent, silent, silent, voiced])

        asyncio.run(body())

    def test_the_turn_keeps_eight_frames_of_tail_then_lets_go(self):
        async def body() -> None:
            built, _ws, _session = build()
            silent, voiced = self.chunk(0), self.chunk(1200)
            await built.on_audio({"data": base64.b64encode(voiced).decode(), "sample_rate": 16000})
            for _ in range(12):
                await built.on_audio({"data": base64.b64encode(silent).decode(), "sample_rate": 16000})
            self.assertEqual(built.pcm, voiced + silent * 8, "the trailing silence after a word is kept, and only that much")
            self.assertEqual(built.tail_chunks, 0, "and the tail is spent")

        asyncio.run(body())


class SpeakingStepTests(unittest.TestCase):
    """An aborted bot turn's `turn_complete` belongs to the step it aborted, never the one that superseded it."""

    class Fallback:
        """The pending silent-model safety cue: records whether anything cancels it."""

        def __init__(self) -> None:
            self.cancelled_ = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled_ = True

        def add_done_callback(self, _callback: Any) -> None:
            return

    def test_an_aborted_turn_complete_never_unlocks_the_superseded_steps_quick_replies(self):
        # A tap on question A's pill advances to question B; Gemini then closes A with `interrupted` and
        # `turn_complete`. Stamping B's token into that `turn_complete` unlocked B's pills before B had
        # spoken a frame, and cancelled B's safety cue on the way out.
        # The same trap must not swing the other way: B's own real `turn_complete` still has to land.
        async def body() -> None:
            built, ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_term"})
            await built._publish_and_cue("speak", immediate=False)
            await built.on_message(OutputTranscriptionTests.audio())
            aborted = built.state.step_token
            self.assertTrue(built.bot_generating, "SHISHU is mid-sentence on the aborted question")

            await built.on_quick_reply({"value": "clear", "step": aborted})
            fresh = built.state.step_token
            self.assertNotEqual(fresh, aborted, "the tap moved the call to the next question")
            fallback = self.Fallback()
            built.fallback = fallback

            await built.on_message(OutputTranscriptionTests.interrupted())
            await built.on_message(OutputTranscriptionTests.complete())

            self.assertEqual(ws.of("bot_turn_complete"), [],
                             "the aborted turn's completion must not unlock the new step's quick replies")
            self.assertFalse(fallback.cancelled_, "nor may it cancel the new step's safety cue")
            self.assertEqual(ws.of("telemetry")[-1]["last_event"]["type"], "turn_complete",
                             "the turn is still reported for telemetry")

            # SHISHU now speaks the new question: its own turn_complete must still unlock its quick replies.
            await built.on_message(OutputTranscriptionTests.audio())
            await built.on_message(OutputTranscriptionTests.complete())
            self.assertEqual(ws.of("bot_turn_complete"),
                             [{"type": "bot_turn_complete", "step": fresh, "epoch": built.epoch}],
                             "the new step's own turn_complete still arrives")

        asyncio.run(body())


class HalfDuplexTests(unittest.TestCase):
    """She can tap over SHISHU but never talk over her: her echo is neither forwarded nor banked."""

    def test_audio_is_dropped_while_shishu_speaks_and_forwarded_once_her_turn_is_closed(self):
        async def body() -> None:
            built, _ws, session = build()
            built.bot_generating, built.playback_ends_at = True, live.time.time() + 5.0
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertEqual(session.kinds(), [], "her own voice never reaches Gemini as a caller turn")
            self.assertEqual(built.pcm, b"", "and it is never banked for a Decision pass")

            await built.on_message(OutputTranscriptionTests.complete())  # her aborted turn is closed
            built.playback_ends_at = live.time.time() - 0.1  # and the browser has finished playing it
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertEqual(len(session.realtime("audio")), 1, "the caller is heard again the moment her turn ends")
            self.assertTrue(built.pcm, "and her words are banked for the pass `next_question` will run")

        asyncio.run(body())

    def test_no_decision_runs_until_gemini_asks_for_the_next_question(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            passes: list[tuple[str, int]] = []

            async def fake_pass(snapshot: CallState, spoken: str, audio: bytes) -> live.Turn:
                passes.append((spoken, len(audio)))
                return turn_with(snapshot, TurnOutcome.CLEAR, spoken)._replace(pcm_len=len(audio))

            built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
            for _ in range(3):
                await built.on_audio({"data": pcm(), "sample_rate": 16000})
            for _ in range(20):  # her pause: a speculative VAD would have fired here
                await built.on_audio({"data": pcm(), "sample_rate": 16000})
                await asyncio.sleep(0)
            self.assertEqual(passes, [], "audio alone never buys a Decision call")

            await built._hear("user", "डोरी सूखी नहीं")
            await built._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})])
            self.assertEqual(len(passes), 1, "one pass per turn, and only when the model asks for the next line")
            self.assertEqual(passes[0][0], "डोरी सूखी नहीं", "the pass saw the whole transcription")
            self.assertGreater(passes[0][1], 0, "and the caller's audio alongside it")

        asyncio.run(body())

    def test_a_tap_over_shishu_silences_the_aborted_turn_without_an_interrupted_ack(self):
        # The browser flushes its queue on the tap, but Gemini still streams Turn A's tail afterwards. Without
        # `sc.interrupted` that tail must still be ignored: it may land before Question B's own `question_state`.
        async def body() -> None:
            built, ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_term"})
            await built._publish_and_cue("speak", immediate=False)
            await built.on_message(OutputTranscriptionTests.audio())  # Turn A starts speaking
            aborted = built.state.step_token
            self.assertTrue(built.bot_generating, "SHISHU is mid-sentence on the aborted question")

            await built.on_quick_reply({"value": "clear", "step": aborted})
            fresh = built.state.step_token
            self.assertNotEqual(fresh, aborted, "the tap moved the call to the next question")
            fallback = SpeakingStepTests.Fallback()
            built._cancel("fallback")  # the real cue the tap scheduled is replaced by a spy on Turn B's
            built.fallback = fallback
            mark = len(ws.sent)

            await built.on_message(OutputTranscriptionTests.audio())  # stale Turn A chunk
            await built.on_message(OutputTranscriptionTests.said("क्या आपकी"))
            await built.on_message(OutputTranscriptionTests.complete())  # Turn A closes, no `interrupted` at all
            reached = [p for p in ws.sent[mark:] if p.get("type") in ("audio", "bot_turn_complete")]
            self.assertEqual(reached, [], "Turn A's tail never reaches the browser")
            self.assertEqual([p for p in ws.of("transcript") if p["role"] == "assistant"], [],
                             "nor does its stale caption")
            self.assertFalse(fallback.cancelled_, "Turn B's safety cue survives Turn A's death")

            await built.on_message(OutputTranscriptionTests.audio())  # Turn B speaks
            await built.on_message(OutputTranscriptionTests.complete())
            self.assertEqual(ws.of("bot_turn_complete"),
                             [{"type": "bot_turn_complete", "step": fresh, "epoch": built.epoch}],
                             "Turn B's own turn_complete still unlocks it")

        asyncio.run(body())

    def test_tool_call_waits_for_trailing_input_transcription_before_deciding(self):
        # Gemini Live occasionally emits `tool_call(next_question)` a frame before `input_transcription`.
        # `_pump` must stay free to receive that transcription so the Decision pass sees the words.
        async def body() -> None:
            built, _ws, _session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_term"})
            seen: list[str] = []

            async def spy_pass(snapshot: CallState, spoken: str, audio_bytes: bytes) -> live.Turn:
                seen.append(spoken)
                return turn_with(snapshot, TurnOutcome.CONCERN, spoken)._replace(pcm_len=len(audio_bytes))

            built._evaluate_turn_task = spy_pass  # type: ignore[method-assign]
            feed = Feed()
            built.session = feed
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            gem = asyncio.ensure_future(built._gemini_loop())
            try:
                await feed.push(live.types.LiveServerMessage(tool_call=live.types.LiveServerToolCall(
                    function_calls=[live.types.FunctionCall(id="t1", name="next_question", args={})])))
                await asyncio.sleep(0.02)
                await feed.push(live.types.LiveServerMessage(server_content=live.types.LiveServerContent(
                    input_transcription=live.types.Transcription(text="तीन दिन नीली लाइट में रखा था।"))))
                await asyncio.sleep(0.08)
                self.assertEqual(seen, ["तीन दिन नीली लाइट में रखा था।"],
                                 "Decisions must wait for the trailing transcription instead of running blind")
            finally:
                await BoundaryEventTests.shutdown(built, gem)

        asyncio.run(body())

    def test_watchdog_keeps_aad_warm_with_digital_silence_while_shishu_speaks(self):
        async def body() -> None:
            built, _ws, session = build(call_id=None)
            built.bot_generating, built.playback_ends_at = True, live.time.time() + 1.0
            wd = asyncio.ensure_future(built._watchdog())
            try:
                await asyncio.sleep(0.15)
            finally:
                built.stop.set()
                await wd
            auds = session.realtime("audio")
            self.assertTrue(auds, "silence keepalive runs while SHISHU speaks so Gemini AAD stays warm")
            self.assertTrue(all(set(b.data) <= {0} for b in auds), "keepalive never carries voice")

        asyncio.run(body())


class TurnTests(unittest.TestCase):
    """One caller turn: the commit, its persistence, and the audio the pass was handed."""

    def test_profile_steps_run_the_extractor_beside_the_decision_pass(self):
        async def body() -> None:
            built, _ws, _session = build()
            ran: list[str] = []
            original_decisions = live.decisions.evaluate_turn
            original_extractor = live.extractor.extract_profile_entities

            async def fake_decisions(*_args: Any, **_kwargs: Any) -> live.decisions.DecisionCallResult:
                ran.append("decisions")
                return TurnTests.result(answers(danger_now=0.01, mentions_profile=0.9))

            async def fake_extractor(*_args: Any, **_kwargs: Any) -> Any:
                ran.append("extractor")
                return live.extractor.ProfileExtractionResult(True)

            live.decisions.evaluate_turn = fake_decisions  # type: ignore[assignment]
            live.extractor.extract_profile_entities = fake_extractor  # type: ignore[assignment]
            try:
                self.assertEqual(live.engine.step_kind(PACK, built.state), "profile")
                await built._evaluate_turn_task(built.state, "मेरा नाम मीरा है", b"")
                self.assertEqual(sorted(ran), ["decisions", "extractor"])
            finally:
                live.decisions.evaluate_turn = original_decisions  # type: ignore[assignment]
                live.extractor.extract_profile_entities = original_extractor  # type: ignore[assignment]

        asyncio.run(body())

    @staticmethod
    def utterance() -> str:
        return "नहीं, ऐसा कुछ नहीं है"

    @staticmethod
    def result(body: dict[str, Any]) -> live.decisions.DecisionCallResult:
        return live.decisions.DecisionCallResult(
            True, "", 1, 200, 100, 4, 12, body["answers"])

    @staticmethod
    def turn(state: CallState, command: live.TurnCommand | None = None) -> live.Turn:
        cmd = command or TurnCommand()
        intent = live.decisions.TurnIntent(
            command=cmd, needs_profile_extraction=False, diagnostics={})
        return live.Turn(TurnTests.result(answers(danger_now=0.01)), intent, None,
                         state.model_copy(deep=True), "नहीं", 640, None)

    def test_state_is_committed_before_the_tool_body_is_returned(self):
        async def body() -> None:
            built, _ws, _session = build()
            order: list[str] = []

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                return self.turn(built.state, TurnCommand(
                    outcome=TurnOutcome.CLEAR, utterance=self.utterance()))

            async def fake_persist(state: CallState) -> None:
                order.append(f"commit:{state.step_token}")
                built.state = state

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            built._persist = fake_persist  # type: ignore[method-assign]
            await built.on_text({"text": self.utterance()})
            body_json, _clarify = await built._next_question()
            self.assertEqual(len(order), 1, "exactly one commit per reduced turn")
            self.assertEqual(body_json["say"], live.engine.say_text(PACK, built.state))

        asyncio.run(body())


class CallerUtteranceTests(unittest.TestCase):
    """The caller's own words survive onto the note, even when the transcription trails the turn."""

    @staticmethod
    def spoken_turn(state: CallState, spoken: str, outcome: TurnOutcome = TurnOutcome.CONCERN) -> live.Turn:
        """A turn whose Decisions pass saw `spoken` (empty means WAV-only)."""
        return turn_with(state, outcome, spoken)

    def test_a_voiced_turn_reaches_the_pass_with_both_the_words_and_the_wav(self):
        # Gemini hears the audio and writes the transcription; the Decision pass must be handed both, in one call.
        async def body() -> None:
            built, _ws, _session = build()
            seen: list[dict[str, Any]] = []
            original = live.decisions.evaluate_turn

            async def spy(request: dict[str, Any], **_kwargs: Any) -> live.decisions.DecisionCallResult:
                seen.append(request)
                return TurnTests.result(answers(danger_now=0.01, active_outcome={"concern": 0.97, "clear": 0.01, "unknown": 0.01}))

            live.decisions.evaluate_turn = spy  # type: ignore[assignment]
            try:
                await built.on_audio({"data": pcm(), "sample_rate": 16000})
                await built._hear("user", "डोरी सूखी नहीं है")
                await built._next_question()
            finally:
                live.decisions.evaluate_turn = original  # type: ignore[assignment]
            self.assertEqual(len(seen), 1, "one pass per turn")
            parts, text = seen[0]["input"], json.dumps(seen[0]["input"], ensure_ascii=False)
            self.assertEqual([p["type"] for p in parts], ["text", "audio"], "the words ride with the waveform")
            self.assertIn("डोरी सूखी नहीं है", text)
            self.assertTrue(parts[1]["data"], "and the audio is not empty")

        asyncio.run(body())

    def test_a_late_chunk_for_a_followup_lands_on_the_followup_slot(self) -> None:
        async def body() -> None:
            built, _ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_jaundice"})

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm: bytes, **_kwargs: Any) -> live.Turn:
                return self.spoken_turn(snapshot, spoken)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_text({"text": "तीन दिन से पीलिया है"})
            await built._next_question()
            self.assertIs(built.state.cursor.substep, live.SubStep.FOLLOWUP, "the followup question is now active")
            built.last_reduced = ("birth_jaundice", live.SubStep.FOLLOWUP, 0, False)
            await built._hear("user", "और तेज़ी से बढ़ रहा है")
            fid = PACK.by_id["birth_jaundice"].followups[0]
            rec = built.state.answers["birth_jaundice"]
            self.assertIn(fid, rec.followups)
            self.assertEqual(rec.followups[fid].note.verbatim, "और तेज़ी से बढ़ रहा है")

        asyncio.run(body())

    def test_voiced_audio_typed_turns_and_taps_clear_the_late_window(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.last_reduced = ("birth_jaundice", live.SubStep.MAIN, 0, False)
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertIsNone(built.last_reduced, "a new voiced turn closes the window")
            built.last_reduced = ("birth_jaundice", live.SubStep.MAIN, 0, False)
            await built.on_text({"text": "नहीं"})
            self.assertIsNone(built.last_reduced)
            built.last_reduced = ("birth_jaundice", live.SubStep.MAIN, 0, False)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "intro_bridge"})
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            self.assertIsNone(built.last_reduced)

        asyncio.run(body())


class ExtractorParallelismTests(unittest.TestCase):
    """A step that saves to the profile must never pay for the extractor after the Decisions pass."""

    def test_a_saves_to_profile_step_runs_the_extractor_beside_the_decision_pass(self):
        # Call #11's `intro_married`: the caller answered "हाँ शादी हो गई है, मेरे दो बच्चे हैं" and `mentions_profile`
        # fired, so `_next_question` ran the extractor *after* Decisions and the turn cost 3.4 s.
        async def body() -> None:
            built, _ws, _session = build(call_id=None)
            state = live.engine.new_state(PACK)
            state.cursor = state.cursor.model_copy(update={"qid": "intro_married"})
            state.profile["caller_name"] = "मीरा"  # named, but the roster is still unresolved
            built.state = state
            ran: list[str] = []
            original_decisions, original_extractor = live.decisions.evaluate_turn, live.extractor.extract_profile_entities

            async def fake_decisions(*_args: Any, **_kwargs: Any) -> live.decisions.DecisionCallResult:
                ran.append("decisions")
                await asyncio.sleep(0.12)
                return TurnTests.result(answers(danger_now=0.01, mentions_profile=0.9,
                                                choice_value={"married": 0.95, "unmarried": 0.03, "other": 0.02}))

            async def fake_extractor(*_args: Any, **_kwargs: Any) -> Any:
                ran.append("extractor")
                await asyncio.sleep(0.12)
                return live.extractor.ProfileExtractionResult(True)

            live.decisions.evaluate_turn = fake_decisions  # type: ignore[assignment]
            live.extractor.extract_profile_entities = fake_extractor  # type: ignore[assignment]
            try:
                await built.on_text({"text": "हाँ शादी हो गई है, मेरे दो बच्चे हैं"})
                started = time.perf_counter()
                await built._next_question()
                elapsed = time.perf_counter() - started
            finally:
                live.decisions.evaluate_turn = original_decisions  # type: ignore[assignment]
                live.extractor.extract_profile_entities = original_extractor  # type: ignore[assignment]
            self.assertEqual(sorted(ran), ["decisions", "extractor"], "the extractor runs exactly once, on this turn")
            self.assertLess(elapsed, 0.21,
                            f"decisions and the extractor must overlap: 2 x 120 ms run sequentially took {elapsed * 1000:.0f} ms")

        asyncio.run(body())


class SttReevaluationTests(unittest.TestCase):
    """The transcription that arrives while the pass is in flight is what the pass sees: one pass, no re-run."""

    GATE = "gate_bones_fractures"

    def test_a_child_named_in_the_transcription_is_resolved_by_the_only_pass(self):
        # `"2 साल"` exists only in the transcription: with a single pass it must be there when the model is asked.
        async def body() -> None:
            built, _ws, _session = build(call_id=None)
            state = live.engine.new_state(PACK)
            state.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male"),
                            RosterItem(id="item_2", age_years=6.0, age_label="6 साल", tag="female")]
            state.cursor = CursorStep(qid=self.GATE, substep=SubStep.WHICH_ITEM)
            state.answers[self.GATE] = AnswerRecord(state=AnswerStatus.CONCERN)
            built.state, spoken = state, []

            async def fake_pass(snapshot: CallState, text: str, pcm: bytes) -> live.Turn:
                spoken.append(text)
                head = ({"item_1": 0.99, "other": 0.01} if "2 साल" in text
                        else {"item_2": 0.45, "item_1": 0.42, "other": 0.13})
                body = answers(danger_now=0.01, active_outcome={"concern": 0.97, "clear": 0.01, "unknown": 0.01}, item_ref=head)
                return live.Turn(TurnTests.result(body), live.decisions.interpret_turn_decision(PACK, snapshot, body["answers"], utterance_text=text),
                                 None, snapshot.model_copy(deep=True), text, len(pcm), None)

            built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            await built._hear("user", "2 साल वाला बच्चा")
            await built._next_question()
            self.assertEqual(spoken, ["2 साल वाला बच्चा"], "exactly one pass, and it saw the words")
            self.assertEqual(built.state.answers[self.GATE].item_ids, ["item_1"], "the named child is resolved first try")

        asyncio.run(body())

    def test_a_transcription_that_lands_after_the_reduction_is_booked_as_late_stt(self):
        # Call #11: 12 of 22 turns lost `cmd.utterance`; the tail is now attached to the question it answered.
        async def body() -> None:
            built, ws, _session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_bcg"})
            passes: list[str] = []

            async def fake_pass(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                passes.append(spoken)
                return turn_with(snapshot, TurnOutcome.CONCERN, spoken)

            built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            await built._next_question()
            await built._hear("user", "हाँ गाठ हुई थी")
            self.assertEqual(passes, [""], "the pass runs once, on what was banked when it started")
            self.assertEqual(ws.of("telemetry")[-1]["last_event"]["type"], "late_stt")

        asyncio.run(body())


class TurnMetaTests(unittest.TestCase):
    """Every reduced turn leaves one compact meta row behind, so a finished call can be re-read cold."""

    @staticmethod
    def fake_pass(outcome: TurnOutcome = TurnOutcome.CONCERN):
        """A Decisions pass that behaves like the real one: it saw exactly `spoken`."""
        async def evaluate(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
            return turn_with(snapshot, outcome, spoken)._replace(pcm_len=len(pcm))
        return evaluate

    def test_a_reduced_turn_leaves_a_scannable_meta_row_on_the_tool_call(self):
        async def body() -> None:
            built, ws, _session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            built._evaluate_turn_task = self.fake_pass()  # type: ignore[method-assign]
            await built.on_text({"text": "डोरी सूखी नहीं"})
            await built._answer_tools([live.types.FunctionCall(id="a", name="next_question", args={})])
            evt = ws.of("telemetry")[-1]["last_event"]
            self.assertEqual(evt["type"], "tool_call")
            meta = evt["meta"]
            for key in ("step_in", "step_out", "advanced", "outcome", "top2", "decision_ms", "audio_ms", "stt"):
                self.assertIn(key, meta, f"{key} must be on the turn's meta")
            self.assertEqual(meta["outcome"], "concern")
            self.assertEqual(meta["top2"], "concern:0.97,clarify:0.006")
            self.assertNotEqual(meta["step_in"], meta["step_out"], "the turn moved the step")
            self.assertIs(meta["advanced"], True)
            self.assertEqual(meta["decision_ms"], 200)
            self.assertEqual(meta["audio_ms"], 0, "a typed turn carries no audio")
            self.assertEqual(meta["stt"], "डोरी सूखी नहीं")
            self.assertNotIn("reheard", meta, "nothing is re-judged any more, so nothing is marked reheard")
            self.assertNotIn("speculated", meta, "and nothing is speculative")

        asyncio.run(body())

    def test_a_late_chunk_after_the_turn_is_booked_against_the_question_it_answered(self):
        # The transcription can trail the reduction: its words belong on the note, not on the next question.
        async def body() -> None:
            built, ws, _session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            built._evaluate_turn_task = self.fake_pass()  # type: ignore[method-assign]
            await built.on_text({"text": "बुखार"})
            await built._next_question()
            self.assertEqual(built.last_turn_meta["stt"], "बुखार", "the one pass saw the words it was given")
            self.assertEqual(built.last_turn_meta["audio_ms"], 0, "a typed turn carries no audio")
            await built._hear("user", "तीन दिन से")
            evt = ws.of("telemetry")[-1]["last_event"]
            self.assertEqual(evt["type"], "late_stt")
            self.assertEqual(evt["detail"], "birth_cord_3weeks", "booked against the question it answered")
            self.assertEqual(evt["meta"]["chars"], len("तीन दिन से"))

        asyncio.run(body())

    def test_a_failed_pass_still_leaves_its_reason_and_attempts(self):
        async def body() -> None:
            built, _ws, _session = build(call_id=None)

            async def failing(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                turn = turn_with(snapshot, TurnOutcome.CONCERN, spoken)
                return turn._replace(result=turn.result._replace(ok=False, error_reason="timeout", attempts=2))

            built._evaluate_turn_task = failing  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            await built._next_question()
            self.assertEqual(built.last_turn_meta["error"], "timeout")
            self.assertEqual(built.last_turn_meta["attempts"], 2)

        asyncio.run(body())

    def test_a_stale_turn_records_why_nothing_advanced(self):
        async def body() -> None:
            built, _ws, _session = build(call_id=None)

            async def from_the_past(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                stale = snapshot.model_copy(deep=True)
                stale.cursor = stale.cursor.model_copy(update={"qid": "birth_jaundice"})
                return turn_with(stale, TurnOutcome.CONCERN, spoken)

            built._evaluate_turn_task = from_the_past  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            body_json, _clarify = await built._next_question()
            self.assertEqual(body_json["result"], "held")
            self.assertEqual(built.last_turn_meta["reason"], "stale_turn")
            self.assertEqual(built.last_turn_meta["step_out"], built.state.step_token)

        asyncio.run(body())

    def test_a_stale_turn_never_republishes_or_clobbers_the_step_that_superseded_it(self):
        async def body() -> None:
            built, ws, _session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_term"})
            gate = asyncio.Event()

            async def slow_stale(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                await gate.wait()
                return turn_with(snapshot, TurnOutcome.CLEAR, spoken)

            built._evaluate_turn_task = slow_stale  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            answering = asyncio.ensure_future(built._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})]))
            await asyncio.sleep(0)
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            mark = len(ws.of("question_state"))
            built.bot_text = "क्या जन्म"  # Question B has started speaking
            gate.set()
            await answering
            self.assertEqual(ws.of("question_state")[mark:], [], "a held stale turn must not republish the active step")
            self.assertEqual(built.bot_text, "क्या जन्म", "nor may it wipe Question B's partial caption")

        asyncio.run(body())


class BoundaryEventTests(unittest.TestCase):
    """Barge-ins and late STT are the two invisible boundaries a dumpster dive has to see."""

    GATE = "birth_cord_3weeks"  # a closed-short question: 15 silent frames carry SPECULATE and SPEECH_END

    @staticmethod
    def tool_results(session: FakeSession) -> dict[str, dict[str, Any]]:
        return {r.id: r.response for kind, payload in session.sent if kind == "tool" for r in payload}

    def gemini_call(self) -> tuple[live.LiveCallSession, FakeWebSocket, Feed, list[int]]:
        """A live session with a real `_gemini_loop` and a Decision pass that takes 50 ms."""
        built, ws, _session = build(call_id=None)
        built.state.cursor = built.state.cursor.model_copy(update={"qid": self.GATE})
        passes: list[int] = []

        async def slow_pass(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
            passes.append(1)
            await asyncio.sleep(0.05)
            return turn_with(snapshot, TurnOutcome.CLEAR, spoken)._replace(pcm_len=len(pcm))

        built._evaluate_turn_task = slow_pass  # type: ignore[method-assign]
        built.session = feed = Feed()
        return built, ws, feed, passes

    @staticmethod
    async def frames(built: live.LiveCallSession, count: int) -> None:
        for _ in range(count):
            await built.on_audio({"data": pcm(), "sample_rate": 16000})

    @staticmethod
    async def shutdown(built: live.LiveCallSession, gem: asyncio.Task) -> None:
        built.stop.set()
        built._cancel("fallback")
        built._cancel("tool_task")
        gem.cancel()
        await asyncio.gather(gem, return_exceptions=True)

    def test_an_interrupted_bot_turn_never_recovers_a_caller_pass(self):
        # Calls 8, 9 and 10: Gemini closed the aborted bot turn while the caller was still mid-sentence, so the
        # recovery pass it started was judged against half of what she said, and the call then went dead air.
        async def body() -> None:
            built, ws, feed, passes = self.gemini_call()
            built.bot_generating, built.playback_ends_at = True, live.time.time() + 10.0
            await self.frames(built, 1)  # a frame of her own echo, dropped while SHISHU speaks
            gem = asyncio.ensure_future(built._gemini_loop())
            try:
                await feed.push(live.types.LiveServerMessage(server_content=live.types.LiveServerContent(interrupted=True)))
                await feed.push(live.types.LiveServerMessage(server_content=live.types.LiveServerContent(turn_complete=True)))
                await asyncio.sleep(0.01)
                self.assertEqual(passes, [], "an aborted bot turn is not a caller turn to recover")
                self.assertEqual([], list(built.preroll), "and her echo left nothing banked behind it")
                await self.frames(built, 3)
                await asyncio.sleep(0.1)
                self.assertFalse(gem.done(), "the aborted bot turn must not take the loop down")
                self.assertNotIn("missed_tool", [p["last_event"]["detail"] for p in ws.of("telemetry")])
                await feed.push(live.types.LiveServerMessage(tool_call=live.types.LiveServerToolCall(
                    function_calls=[live.types.FunctionCall(id="t1", name="next_question", args={})])))
                await feed.push(live.types.LiveServerMessage(server_content=live.types.LiveServerContent(turn_complete=True)))
                await asyncio.sleep(0.1)
                self.assertEqual(self.tool_results(feed)["t1"]["result"], "ok")
                self.assertEqual(len(passes), 1, "her turn is judged exactly once")
                self.assertEqual((built.last_reduced or ("",))[0], self.GATE)
            finally:
                await self.shutdown(built, gem)

        asyncio.run(body())

    def test_a_live_loop_that_dies_on_its_own_is_reported_never_swallowed(self):
        # The calls 8-10 symptom: a loop dying silently looked exactly like a caller hangup.
        async def body() -> None:
            built, ws, _session = build(call_id=None)

            async def dying() -> None:
                raise RuntimeError("gemini loop exploded")

            built._gemini_loop = dying  # type: ignore[method-assign]
            await asyncio.wait_for(built.run(), timeout=2)
            self.assertEqual([p["message"] for p in ws.of("error")], ["Live loop exited unexpectedly"])
            self.assertEqual(ws.of("telemetry")[-1]["last_event"]["type"], "error")

        asyncio.run(body())

    def test_geminis_interrupt_ack_closes_the_bots_turn_and_never_emits_a_caller_pass(self):
        # Calls 8 and 9: she cut in over SHISHU, Gemini acked with `interrupted`, and the call went dead air.
        async def body() -> None:
            built, ws, session = build(call_id=None)
            passes: list[int] = []

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                passes.append(1)
                return TurnTests.turn(built.state)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_message(OutputTranscriptionTests.audio())  # SHISHU is mid-line
            await built.on_message(live.types.LiveServerMessage(
                server_content=live.types.LiveServerContent(interrupted=True)))
            self.assertFalse(built.bot_generating, "the ack alone closes her turn")
            self.assertEqual(built.playback_ends_at, 0.0, "and the browser's queue is declared empty")
            self.assertEqual(built.telemetry["interruptions"], 1)
            self.assertEqual(ws.of("clear"), [{"type": "clear", "epoch": built.epoch}], "the browser queue is flushed")
            self.assertEqual(ws.of("telemetry")[-1]["last_event"]["type"], "barge_in")
            self.assertEqual(passes, [], "an ack is not a caller turn")
            self.assertEqual(session.kinds(), [], "and it talks to nobody")

        asyncio.run(body())

    def test_a_late_chunk_that_lands_on_a_concern_reports_attached(self):
        async def body() -> None:
            for outcome, attached in ((TurnOutcome.CONCERN, True), (TurnOutcome.CLEAR, False)):
                with self.subTest(outcome=outcome.value):
                    built, ws, _session = build(call_id=None)
                    built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
                    built._evaluate_turn_task = TurnMetaTests.fake_pass(outcome)  # type: ignore[method-assign]
                    await built.on_text({"text": "डोरी सूखी नहीं"})
                    await built._next_question()
                    await built._hear("user", "तीन हफ़्ते से")
                    evt = ws.of("telemetry")[-1]["last_event"]
                    self.assertEqual(evt["type"], "late_stt")
                    self.assertEqual(evt["detail"], "birth_cord_3weeks", "booked against the question it answered")
                    self.assertEqual(evt["meta"], {"attached": attached, "chars": len("तीन हफ़्ते से")})

        asyncio.run(body())


class FinalizeTests(unittest.TestCase):
    """A hangup mid-question must be reported as a partial call, never as a complete one."""

    def test_hangup_while_asking_is_stamped_caller_stop(self):
        async def body() -> None:
            built, _ws, _session = build()
            persisted: list[CallState] = []

            async def spy(state: CallState) -> None:
                persisted.append(state)
                built.state = state

            built._persist = spy  # type: ignore[method-assign]
            await built.finalize()
            self.assertIs(built.state.phase, Phase.CLOSING)
            self.assertIs(built.state.close_reason, CloseReason.CALLER_STOP, "no more पूर्ण कॉल on a partial call")
            self.assertEqual([s.close_reason for s in persisted], [CloseReason.CALLER_STOP])

        asyncio.run(body())

    def test_a_closed_call_keeps_the_reason_it_already_had(self):
        async def body() -> None:
            built, _ws, _session = build()
            st = built.state.model_copy(deep=True)
            st.phase, st.close_reason = Phase.CLOSING, CloseReason.COMPLETE
            built.state = st
            writes: list[CallState] = []

            async def spy(state: CallState) -> None:
                writes.append(state)
                built.state = state

            built._persist = spy  # type: ignore[method-assign]
            await built.finalize()
            self.assertEqual(writes, [], "a finished call is not rewritten")
            self.assertIs(built.state.close_reason, CloseReason.COMPLETE)

        asyncio.run(body())


class QuickReplyTests(unittest.TestCase):
    """A tap is a whole turn: validated, reduced once, persisted, then spoken."""

    def test_quick_reply_uses_value_and_requires_the_current_step(self):
        async def body() -> None:
            built, _ws, session = build()
            built.state = live.engine.new_state(PACK)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "intro_bridge"})
            await built.on_quick_reply({"value": "clear", "step": "stale_step"})
            self.assertEqual(session.kinds(), [], "a stale step is ignored")
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            self.assertEqual(session.kinds(), ["text"], "the cue is the only send")
            self.assertTrue(built.state.answers, "the tap answered the step")
            self.assertEqual((built.text, built.pcm), ("", b""), "and its label never rides into the next Decision pass")

        asyncio.run(body())


class TextAndToolTests(unittest.TestCase):
    """Typed turns, tool answers, the missed-tool fallback, and the silent-model cue."""

    def test_text_turn_is_spoken_as_a_cue_and_can_be_judged(self):
        async def body() -> None:
            built, ws, session = build()
            await built.on_text({"text": "  मेरा   नाम मीरा है "})
            self.assertEqual(session.realtime("text"), ["मेरा नाम मीरा है"])
            self.assertEqual(ws.of("transcript")[0]["text"], "मेरा नाम मीरा है")
            self.assertEqual(built.text, "मेरा नाम मीरा है", "the words are banked for the pass next_question will run")

        asyncio.run(body())

    def test_duplicate_tool_calls_are_answered_once(self):
        async def body() -> None:
            built, _ws, session = build()
            call = live.types.FunctionCall(id="a", name="next_question", args={})
            await built._answer_tools([call, call])
            self.assertEqual(len([kind for kind in session.kinds() if kind == "tool"]), 1)

        asyncio.run(body())

    def test_missed_tool_on_turn_complete_speaks_the_line(self):
        async def body() -> None:
            built, ws, session = build()

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                return TurnTests.turn(built.state, TurnCommand(
                    outcome=TurnOutcome.CLEAR, utterance="नहीं"))

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            await built.on_turn_complete()
            self.assertTrue(session.realtime("text"), "the fallback cue went out")
            self.assertEqual(ws.of("bot_turn_complete")[-1]["step"], built.state.step_token)

        asyncio.run(body())

    def test_untooled_bot_reply_recovers_via_missed_tool_without_bleeding_into_next_turn(self):
        # Live E2E P7: caller asked "इसका मतलब क्या है?" and Gemini replied aloud WITHOUT calling `next_question`.
        # The un-tooled reply bypassed `missed_tool` and left `built.text`/`built.pcm` dirty, so the next turn merged
        # the clarify question into the caller's real "नहीं" answer (`outcome=not_about_active`).
        async def body() -> None:
            built, ws, session = build()
            st = built.state.model_copy(deep=True)
            st.profile = {"caller_name": "कविता", "marital_status": "married"}
            st.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male")]
            for q in PACK.questions:
                if q.id == "birth_term": break
                st.answers[q.id] = AnswerRecord(state=AnswerStatus.CLEAR)
            st.cursor = CursorStep(qid="birth_term")
            built.state = st
            seen: list[str] = []

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                seen.append(spoken)
                out = TurnOutcome.CLARIFY if "मतलब" in spoken else TurnOutcome.CLEAR
                return turn_with(snapshot, out, spoken)._replace(pcm_len=len(pcm_bytes))

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            await built._hear("user", "इसका मतलब क्या है?")
            # Gemini skips `next_question` and starts speaking directly:
            await built.on_message(OutputTranscriptionTests.said("मतलब, क्या बच्चा समय से पहले पैदा हुआ?"))
            await built.on_message(OutputTranscriptionTests.audio())
            await built.on_message(OutputTranscriptionTests.complete())

            self.assertEqual(ws.of("audio"), [], "un-tooled audio cannot play over an unevaluated caller turn")
            self.assertEqual([c for c in ws.of("transcript") if c["role"] == "assistant"], [],
                             "nor may its un-tooled caption leak")
            self.assertEqual(seen, ["इसका मतलब क्या है?"], "turn_complete must recover the clarify turn via missed_tool")
            self.assertTrue(session.realtime("text"), "the authoritative clarify cue went out")
            self.assertEqual((built.text, built.pcm), ("", b""), "recovered turn clears both text and pcm")

            # SHISHU now speaks the real clarification, finishes, and the caller answers clearly:
            await built.on_message(OutputTranscriptionTests.audio())
            await built.on_message(OutputTranscriptionTests.complete())
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            await built._hear("user", "नहीं, नौ महीने में हुआ था")
            await built._answer_tools([live.types.FunctionCall(id="t2", name="next_question", args={})])
            self.assertEqual(seen, ["इसका मतलब क्या है?", "नहीं, नौ महीने में हुआ था"],
                             "the next turn must never inherit the prior clarify text")
            self.assertEqual(built.state.cursor.qid, "birth_nicu")

        asyncio.run(body())

    def test_stale_untranscribed_pcm_before_bot_turn_never_bleeds_into_next_turn(self):
        async def body() -> None:
            built, _ws, _session = build()
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertTrue(built.pcm)
            await built.on_message(OutputTranscriptionTests.audio())
            await built.on_message(OutputTranscriptionTests.complete())
            self.assertEqual(built.pcm, b"", "closing a heard bot turn wipes any pre-turn audio burst")

        asyncio.run(body())


class WatchdogTests(unittest.TestCase):
    """Timers, the closing handshake, and the resumable reconnect."""

    def test_silence_check_in_then_silence_close(self):
        async def body() -> None:
            built, ws, session = build()
            built.started_at = built.last_voice_ts = 0.0
            await built.tick(built.check_in_sec + 1)
            self.assertTrue(session.realtime("text"), "the 15 s check-in cue was sent")
            self.assertEqual(ws.of("telemetry")[-1]["last_event"]["type"], "silence_check_in")
            built.last_voice_ts = 0.0
            await built.tick(built.silence_close_sec + 1)
            self.assertIs(built.state.phase, Phase.CLOSING)
            self.assertIs(built.state.close_reason, CloseReason.SILENCE)

        asyncio.run(body())

    def test_soft_limit_closes_and_hard_limit_ends(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.started_at = 0.0
            self.assertTrue(await built.tick(built.soft_limit_sec + 1))
            self.assertIs(built.state.close_reason, CloseReason.TIME_LIMIT)
            built.started_at, built.closing_at = 0.0, 0.0
            self.assertFalse(await built.tick(built.hard_limit_sec + 1))

        asyncio.run(body())

    def test_closing_waits_for_playback_then_emits_ending(self):
        async def body() -> None:
            built, ws, _session = build()
            built.state.phase = Phase.CLOSING
            built.state.close_reason = CloseReason.COMPLETE
            built.closing_at = 0.0
            built.playback_ends_at = 100.0
            self.assertTrue(await built.tick(50.0))
            self.assertEqual(ws.of("ending"), [])
            self.assertFalse(await built.tick(101.0))
            self.assertEqual(len(ws.of("ending")), 1)

        asyncio.run(body())

    def test_reconnect_swaps_the_session_and_re_owes_the_cue(self):
        async def body() -> None:
            built, ws, _session = build()
            replacement = FakeSession()

            async def connect(handle: str) -> FakeSession:
                self.assertEqual(handle, "handle-1")
                return replacement

            built.connect, built.resume_handle = connect, "handle-1"
            built.resume_requested = True
            self.assertTrue(await built.tick(1.0))
            self.assertIs(built.session, replacement)
            self.assertEqual([p["type"] for p in ws.sent][-2:], ["reconnected", "telemetry"])
            self.assertTrue(replacement.realtime("text"), "the pending cue was re-owed")

        asyncio.run(body())

    def test_reconnect_stops_the_call_after_the_cap(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.reconnects = live._MAX_RECONNECTS

            async def never(_handle: str) -> Any:
                raise AssertionError("the cap must short-circuit before connecting")

            built.connect = never
            built.resume_requested = True
            self.assertFalse(await built.tick(1.0))
            self.assertTrue(built.stop.is_set())

        asyncio.run(body())


class BrowserLoopTests(unittest.TestCase):
    """The socket loop: bad payloads are reported, hangup stops the call."""

    def test_invalid_json_and_non_object_payloads_are_reported(self):
        async def body() -> None:
            built, ws, _session = build()
            await ws.push("{not json")
            await ws.push([1, 2, 3])
            await ws.push(None)
            await asyncio.wait_for(built._browser_loop(), timeout=2)
            self.assertEqual(len(ws.of("error")), 2)
            self.assertTrue(built.stop.is_set())

        asyncio.run(body())

    def test_hangup_stops_without_speaking(self):
        async def body() -> None:
            built, ws, session = build()
            await ws.push({"type": "hangup"})
            await asyncio.wait_for(built._browser_loop(), timeout=2)
            self.assertTrue(built.stop.is_set())
            self.assertEqual(session.kinds(), [])

        asyncio.run(body())

    def test_audio_from_a_closed_call_is_ignored(self):
        async def body() -> None:
            built, _ws, session = build()
            built.state.phase = Phase.CLOSING
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertEqual(session.kinds(), [])

        asyncio.run(body())


class EntrypointTests(unittest.TestCase):
    """`handle_live_ws` validates before it connects, and always shuts the socket down."""

    def test_a_missing_api_key_is_rejected_without_connecting(self):
        async def body() -> None:
            ws, opened = FakeWebSocket(), []

            async def never(client: Any, pack: Any, handle: str) -> Any:
                opened.append(handle)
                raise AssertionError("no Live context may be opened without a key")

            original = live._open_live
            live._open_live = never  # type: ignore[assignment]
            try:
                with unittest.mock.patch.dict(live.os.environ, {}, clear=False):
                    live.os.environ.pop("GEMINI_API_KEY", None)
                    await live.handle_live_ws(ws, pack=PACK)
            finally:
                live._open_live = original  # type: ignore[assignment]
            self.assertEqual(opened, [])
            self.assertEqual(ws.of("error")[0]["message"], "GEMINI_API_KEY is not set")
            self.assertTrue(ws.closed)

        asyncio.run(body())

    def test_a_failed_live_connect_rejects_and_closes(self):
        async def body() -> None:
            ws = FakeWebSocket()

            async def broken(client: Any, pack: Any, handle: str) -> Any:
                raise RuntimeError("live down")

            original = live._open_live
            live._open_live = broken  # type: ignore[assignment]
            try:
                with unittest.mock.patch.dict(live.os.environ, {"GEMINI_API_KEY": "k"}):
                    await live.handle_live_ws(ws, pack=PACK)
            finally:
                live._open_live = original  # type: ignore[assignment]
            self.assertEqual(ws.of("error")[0]["message"], "Gemini Live connection failed")
            self.assertTrue(ws.closed)

        asyncio.run(body())


class ResumeSessionTests(unittest.TestCase):
    """A socket that dies is a dropped line, not a hangup: the call waits in grace, then resumes where it stopped."""

    @staticmethod
    async def wait_until(predicate: Any, *, timeout: float = 3.0) -> bool:
        """Poll for a wire event: every step of `handle_live_ws` is a real task, so nothing is synchronous."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.005)
        return predicate()

    def test_unexpected_ws_disconnect_keeps_call_resumable_and_resume_token_restores_cursor(self):
        # Call #13: the phone lost its socket mid-question, the server read that as a caller hangup, stamped
        # CALLER_STOP on a live call, and the browser's rejoin opened a brand-new empty call instead.
        async def body() -> None:
            sessions: list[FakeSession] = []
            opened = {"n": 0}

            class DummyAsyncCtx:
                async def __aenter__(self) -> Any:
                    return FakeSession()

                async def __aexit__(self, *_exc: Any) -> bool:
                    return False

            async def fake_open(_client: Any, _pack: Any, _handle: str) -> Any:
                opened["n"] += 1
                session = FakeSession()
                sessions.append(session)
                return DummyAsyncCtx(), session

            ws1, ws2 = FakeWebSocket(), FakeWebSocket()
            first = second = None
            created: list[int] = []
            original = live._open_live
            live._open_live = fake_open  # type: ignore[assignment]
            try:
                with unittest.mock.patch.dict(live.os.environ, {"GEMINI_API_KEY": "k"}):
                    first = asyncio.ensure_future(live.handle_live_ws(ws1, pack=PACK, caller_name="Tester"))
                    self.assertTrue(await self.wait_until(lambda: bool(ws1.of("session_started"))),
                                    "the first socket must be told its call started")
                    started = ws1.of("session_started")[0]
                    orig_cid, share_token = int(started["call_id"]), str(started["share_token"])
                    created.append(orig_cid)
                    self.assertTrue(await self.wait_until(lambda: orig_cid in live._ACTIVE_CALLS),
                                    "an in-flight call is registered so a rejoin can find it")
                    self.assertTrue(share_token, "the call carries the token the browser resumes with")

                    active = live._ACTIVE_CALLS[orig_cid]
                    active.state.profile["caller_name"] = "तारा"
                    active.state.cursor = active.state.cursor.model_copy(update={"qid": "intro_married"})
                    await active._persist(active.state)

                    await ws1.queue.put(None)  # the socket simply dies: no `hangup` message is ever sent
                    await asyncio.wait_for(first, timeout=3)
                    first = None

                    self.assertIn(orig_cid, live._PENDING_FINALIZE, "a dropped socket starts the resume grace")
                    row = await asyncio.to_thread(db.get_call, orig_cid)
                    self.assertEqual(row["status"], "active", "the dropped call is still a live call")
                    saved = db.call_state_from_row(row)
                    self.assertEqual(saved.cursor.qid, "intro_married",
                                     "and its cursor is not overwritten as a caller stop")

                    # Call #13's other half: a half-open session whose socket died without anyone noticing
                    # still sits in the registry. The rejoin must evict it, never let it finalize over the call.
                    linger = live.LiveCallSession(ws=FakeWebSocket(), pack=PACK, session=FakeSession(), call_id=orig_cid,
                                                  state=saved.model_copy(deep=True), recorder=FakeRecorder())
                    finalized: list[int] = []
                    original_finalize = linger.finalize

                    async def spy_finalize() -> None:
                        finalized.append(1)
                        await original_finalize()

                    linger.finalize = spy_finalize  # type: ignore[method-assign]
                    live._ACTIVE_CALLS[orig_cid] = linger

                    second = asyncio.ensure_future(live.handle_live_ws(ws2, pack=PACK, resume_token=share_token))
                    self.assertTrue(await self.wait_until(
                        lambda: bool(ws2.of("session_started")) and bool(ws2.of("question_state"))),
                        "the rejoin must be given its call and its next step")
                    self.assertEqual(ws2.of("session_started")[0]["call_id"], orig_cid,
                                     "no second call row is minted for a resumed call")
                    self.assertEqual(len(created), 1)
                    self.assertNotIn(orig_cid, live._PENDING_FINALIZE, "resuming cancels the delayed finalize")

                    self.assertTrue(linger.superseded, "the lingering half-open session is told it lost the call")
                    self.assertTrue(linger.stop.is_set(), "and its loops are stopped")
                    self.assertEqual(finalized, [], "it may never finalize the call it no longer owns")
                    self.assertEqual(linger.recorder.saves, 1, "its audio is still written to disk")  # type: ignore[attr-defined]
                    still = db.call_state_from_row(await asyncio.to_thread(db.get_call, orig_cid))
                    self.assertIsNot(still.close_reason, CloseReason.CALLER_STOP,
                                      "the evicted session never stamps CALLER_STOP on the live call")

                    resumed = live._ACTIVE_CALLS[orig_cid]
                    self.assertEqual(resumed.state.profile["caller_name"], "तारा", "the caller's name survives")
                    self.assertEqual(resumed.state.cursor.qid, "intro_married", "and so does her place in the call")
                    self.assertEqual(ws2.of("question_state")[0]["qid"], "intro_married",
                                     "the browser is handed the step the call paused on")
                    events = [p["last_event"]["type"] for p in ws2.of("telemetry")]
                    self.assertIn("session_resumed", events, "the rejoin is recorded as a resume, not a new call")

                    handed = sessions[1].realtime("text")
                    self.assertEqual(handed, [live.speak_cue(PACK, resumed.state, "speak")],
                                     "Gemini is asked for the paused question's line")
                    greeting = str(PACK.prompts.get("greeting_cue") or "").rstrip("\n")
                    self.assertFalse(any(greeting in str(text) for text in handed),
                                     "the resumed call never re-greets the caller")

                    await ws2.push({"type": "hangup"})
                    await asyncio.wait_for(second, timeout=3)
                    second = None
                    finished = await asyncio.to_thread(db.get_call, orig_cid)
                    self.assertEqual(finished["status"], "completed",
                                     "an explicit hangup still finalizes immediately")
            finally:
                live._open_live = original  # type: ignore[assignment]
                for task in (first, second):
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                for call_id in created:
                    with db._LOCK:
                        db.conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
                        db.conn.commit()
                    live._ACTIVE_CALLS.pop(call_id, None)
                    if (pending := live._PENDING_FINALIZE.pop(call_id, None)) is not None:
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)

        asyncio.run(body())


class DecisionRecoveryTests(unittest.TestCase):
    """A Decision pass that dies is a glitch, not a hangup: the same question is re-asked before any call is ended."""

    QID = "body_private_parts"

    @staticmethod
    def failed(snapshot: CallState, spoken: str, *, reason: str = "timeout", attempts: int = 2) -> live.Turn:
        """A pass that never produced answers: `ok=False` with the reason the transport gave."""
        turn = turn_with(snapshot, TurnOutcome.UNKNOWN, spoken)
        return turn._replace(result=turn.result._replace(ok=False, error_reason=reason, attempts=attempts))

    @staticmethod
    def answered(snapshot: CallState, spoken: str) -> live.Turn:
        return turn_with(snapshot, TurnOutcome.UNKNOWN, spoken)

    @staticmethod
    def tool_body(session: FakeSession, call_id: str) -> dict[str, Any]:
        """The body Gemini was handed for one `next_question` call."""
        return {r.id: r.response for kind, payload in session.sent if kind == "tool" for r in payload}[call_id]

    async def speak_turn(self, built: live.LiveCallSession, call_id: str) -> None:
        """One caller turn, driven the way Gemini drives it: the words arrive, then `next_question` asks for the line."""
        await built.on_text({"text": "नहीं"})
        await built._answer_tools([live.types.FunctionCall(id=call_id, name="next_question", args={})])

    def test_single_decision_timeout_reasks_current_step_instead_of_closing_call(self):
        # Call #15: one 4.0 s tail spike on the 64-head request was the whole distance between a caller and a hangup.
        async def body() -> None:
            built, _ws, session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": self.QID})
            turns = [self.failed, self.answered]

            async def fake_pass(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                return turns.pop(0)(snapshot, spoken)

            built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
            await self.speak_turn(built, "t1")
            retry = self.tool_body(session, "t1")
            self.assertIs(built.state.phase, Phase.ASKING, "one failed pass never ends the call")
            self.assertEqual(built.state.cursor.qid, self.QID, "the caller is asked the same question again")
            self.assertEqual(built.consecutive_tool_errors, 1, "the strike is counted")
            self.assertEqual(built.telemetry["tool_errors"], 1, "and reported in telemetry")
            self.assertEqual(retry["result"], "retry", "the model is told to say the line again")
            self.assertNotIn("stop", retry, "nothing tells Gemini the call is over")
            self.assertEqual(built.last_turn_meta["recovered"], "reask")
            self.assertEqual(built.last_turn_meta["strike"], 1)

            await self.speak_turn(built, "t2")
            self.assertEqual(built.consecutive_tool_errors, 0, "a turn that lands clears the strikes")
            self.assertNotEqual(built.state.cursor.qid, self.QID, "and the call really moved on")

            # The gate slice is the whole turn on an ordinary "नहीं": the 56 stated heads and the 53-option router
            # are what blew the 4.0 s deadline on Call #15, so nothing may ask for them again.
            fresh, _fresh_ws, _fresh_session = build(call_id=None)
            fresh.state.cursor = fresh.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            seen: list[dict[str, Any]] = []
            original = live.decisions.evaluate_turn

            async def spy(request: dict[str, Any], **kwargs: Any) -> live.decisions.DecisionCallResult:
                seen.append({"request": request, **kwargs})
                return TurnTests.result(answers(danger_now=0.01,
                                                active_outcome={"clear": 0.95, "other": 0.05}))

            live.decisions.evaluate_turn = spy  # type: ignore[assignment]
            try:
                turn = await fresh._evaluate_turn_task(fresh.state, "नहीं", b"")
            finally:
                live.decisions.evaluate_turn = original  # type: ignore[assignment]
            self.assertEqual(len(seen), 1, "a turn is one patient call, never a gate pass plus a fan-out")
            self.assertIn("next_qid", seen[0]["request"]["questions"],
                          "the one pass carries the router, so an ordinary नहीं never pays for a second one")
            self.assertNotIn("next_qid", seen[0]["fallback_request"]["questions"],
                             "and the slim retry body is the narrow slice")
            meta = fresh._turn_meta(turn, fresh.state.step_token, TurnCommand(), None)
            self.assertNotIn("slim", meta, "one clean pass is not a degraded retry")
            self.assertNotIn("fanout", meta, "and there is no fan-out to stamp")

        asyncio.run(body())

    def test_three_consecutive_decision_failures_close_call_as_technical_failure(self):
        async def body() -> None:
            built, _ws, session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": self.QID})

            async def always_timeout(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                return self.failed(snapshot, spoken)

            built._evaluate_turn_task = always_timeout  # type: ignore[method-assign]
            for attempt, strikes in ((1, 1), (2, 2)):
                await self.speak_turn(built, f"t{attempt}")
                self.assertIs(built.state.phase, Phase.ASKING, f"strike {strikes} still re-asks")
                self.assertEqual(built.consecutive_tool_errors, strikes)
                self.assertEqual(self.tool_body(session, f"t{attempt}")["result"], "retry")

            await self.speak_turn(built, "t3")
            self.assertIs(built.state.phase, Phase.CLOSING, "the third strike is a technical failure, not silence")
            self.assertIs(built.state.close_reason, CloseReason.TECHNICAL_FAILURE)
            self.assertIs(self.tool_body(session, "t3")["stop"], True, "and the closing line is owed to the caller")

        asyncio.run(body())

    def test_missing_api_key_closes_immediately_on_first_failure(self):
        # No key is not a timeout: retrying twice more only holds a caller on a dead line.
        async def body() -> None:
            built, _ws, session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": self.QID})

            async def no_key(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                return self.failed(snapshot, spoken, reason="missing_api_key", attempts=1)

            built._evaluate_turn_task = no_key  # type: ignore[method-assign]
            await self.speak_turn(built, "t1")
            self.assertIs(built.state.phase, Phase.CLOSING, "one missing key ends the call on the first turn")
            self.assertIs(built.state.close_reason, CloseReason.TECHNICAL_FAILURE)
            self.assertEqual(built.consecutive_tool_errors, 1, "no strike budget is spent waiting for a key")
            self.assertIs(self.tool_body(session, "t1")["stop"], True)

        asyncio.run(body())


class OnePassDecisionTests(unittest.TestCase):
    """Staging Call #16: one patient pass per turn, a slim retry inside it, and an extractor read that still lands."""

    QID = "body_private_parts"  # a routable gate with 50+ open questions behind it

    # Call #16, event `[51]`: the mother answered the roster hold with both children named.
    ROSTER_UTTERANCE = "एक बच्चे की उम्र अमन है जो 6 साल का है और छोटी बेटी मीरा जो 2 साल की है।"

    @staticmethod
    def state() -> CallState:
        """A call held on `body_private_parts` with the roster resolved, so no extractor call is owed."""
        state = live.engine.new_state(PACK)
        state.cursor = CursorStep(qid=OnePassDecisionTests.QID)
        state.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male")]
        state.roster_stated = True
        return state

    @staticmethod
    def roster_hold_state() -> CallState:
        """Call #16 turn `[11]`: married and named, held on the ages reask with the roster still unknown."""
        state = live.engine.new_state(PACK)
        state.profile["caller_name"] = "कविता"
        state.answers["intro_married"] = AnswerRecord(state=AnswerStatus.ANSWERED, value="married")
        state.cursor = CursorStep(qid="intro_children", hold_slot="roster")
        return state

    @staticmethod
    def aman_and_meera() -> ProfilePatch:
        """The patch `extractor.py` really returned in Call #16, in ~1.5 s, while Decisions timed out."""
        return ProfilePatch(roster_complete=True, roster=(RosterInput(age_years=6.0, tag="male", name="अमन"),
                                                          RosterInput(age_years=2.0, tag="female", name="मीरा")))

    def test_evaluate_turn_task_sends_full_primary_with_slim_fallback_in_single_call(self):
        # Call #16: the two-pass gate paid 1.2k tokens for Pass 1 and then threw the whole turn away when the
        # 4.0 s deadline hit. One pass now owns the full request, and the slim slice is only its retry body.
        async def body() -> None:
            fresh, _ws, _session = build(call_id=None, state=self.state())
            calls: list[dict[str, Any]] = []
            original = live.decisions.evaluate_turn

            async def spy(request: dict[str, Any], **kwargs: Any) -> live.decisions.DecisionCallResult:
                calls.append({"request": request, **kwargs})
                return TurnTests.result(answers(danger_now=0.01,
                                                active_outcome={"clear": 0.95, "other": 0.05}))

            live.decisions.evaluate_turn = spy  # type: ignore[assignment]
            try:
                turn = await fresh._evaluate_turn_task(fresh.state, "नहीं", b"")
            finally:
                live.decisions.evaluate_turn = original  # type: ignore[assignment]

            self.assertEqual(len(calls), 1, "one turn is one call, not a gate plus a fan-out")
            request, fallback = calls[0]["request"], calls[0].get("fallback_request")
            self.assertIsNotNone(fallback, "the retry body must ride beside the primary in the same call")
            self.assertEqual(request, live.decisions.build_turn_request(PACK, fresh.state, utterance_text="नहीं", slim=False),
                             "the primary is the full request, revision heads and router included")
            self.assertEqual(fallback, live.decisions.build_turn_request(PACK, fresh.state, utterance_text="नहीं", slim=True),
                             "and the fallback is exactly the slim slice")
            self.assertIn("next_qid", request["questions"], "a full pass may still route the call")
            self.assertIn("stated_fam_cancer", request["questions"], "and revise an earlier answer")
            self.assertNotIn("next_qid", fallback["questions"], "the retry slice carries no router")
            self.assertIn("active_outcome", fallback["questions"], "but keeps the heads a turn cannot be read without")
            self.assertLess(len(fallback["questions"]), len(request["questions"]) * 0.25,
                            "the retry is a fraction of the primary's heads, that is the point of it")
            self.assertIs(turn.result.ok, True)

        asyncio.run(body())

    def test_call16_profile_extractor_advances_step_when_decision_times_out(self):
        # Call #16 turns `[11]` and `[13]`: the extractor returned अमन 6yo male and मीरा 2yo female in 1.5 s while
        # Decisions timed out, and `_next_question` threw the extraction away to re-ask the same question — twice.
        async def body() -> None:
            built, _ws, _session = build(call_id=None, state=self.roster_hold_state())
            self.assertEqual(built.state.step_token, "intro_children:hold:roster")
            original_decisions, original_extractor = live.decisions.evaluate_turn, live.extractor.extract_profile_entities

            async def timed_out(*_args: Any, **_kwargs: Any) -> live.decisions.DecisionCallResult:
                return live.decisions.DecisionCallResult(False, "timeout", 2, 8_100, 92_000, 64, None, {})

            async def extracted(*_args: Any, **_kwargs: Any) -> live.extractor.ProfileExtractionResult:
                return live.extractor.ProfileExtractionResult(True, self.aman_and_meera(), extractor_ms=1_500)

            live.decisions.evaluate_turn = timed_out  # type: ignore[assignment]
            live.extractor.extract_profile_entities = extracted  # type: ignore[assignment]
            try:
                await built.on_text({"text": self.ROSTER_UTTERANCE})
                payload, _clarify = await built._next_question()
            finally:
                live.decisions.evaluate_turn = original_decisions  # type: ignore[assignment]
                live.extractor.extract_profile_entities = original_extractor  # type: ignore[assignment]

            self.assertEqual(payload["result"], "ok",
                             "a dead Decision pass must not re-ask a question the extractor just answered")
            self.assertEqual(built.state.cursor.qid, "intro_bridge", "the resolved roster carried the call forward")
            self.assertEqual([(c.name, c.age_years, c.tag) for c in built.state.roster],
                             [("अमन", 6.0, "male"), ("मीरा", 2.0, "female")], "both children survive, named and tagged")
            self.assertEqual(built.consecutive_tool_errors, 0, "a recovered turn is not a strike")
            self.assertEqual(built.last_turn_meta["recovered"], "extractor", "and the recovery says who saved it")
            self.assertEqual(built.last_turn_meta["decision_error"], "timeout")
            self.assertEqual(built.last_turn_meta["ext_ms"], 1_500)
            self.assertEqual(payload["say"], live.engine.say_text(PACK, built.state),
                             "she speaks the step the extraction landed on, not the one that timed out")

        asyncio.run(body())

    def test_a_mid_call_profile_patch_is_preserved_even_when_the_pass_still_fails(self):
        # Not every extraction read moves the step: "दो बच्चे हैं" gives the count but no ages, so the roster hold
        # stays put and a re-ask is still owed. The entities she just gave must not be dropped with the dead pass.
        async def body() -> None:
            built, _ws, _session = build(call_id=None, state=self.roster_hold_state())
            step_in = built.state.step_token
            patch = ProfilePatch(slots={"num_children": 2})

            async def failed_with_patch(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                turn = turn_with(snapshot, TurnOutcome.UNKNOWN, spoken)
                return turn._replace(result=turn.result._replace(ok=False, error_reason="timeout"),
                                     extraction=live.extractor.ProfileExtractionResult(True, patch, extractor_ms=600))

            built._evaluate_turn_task = failed_with_patch  # type: ignore[method-assign]
            await built.on_text({"text": "दो बच्चे हैं"})
            payload, _clarify = await built._next_question()
            self.assertEqual(payload["result"], "retry", "a patch that moves no step still leaves the turn unanswered")
            self.assertEqual(built.state.step_token, step_in, "the ages reask is still owed")
            self.assertEqual(built.state.profile.get("num_children"), 2,
                             "but the count she just gave is on the record, not thrown away with the dead pass")
            self.assertEqual(built.consecutive_tool_errors, 1, "an unanswered turn is still a strike")
            self.assertEqual(built.last_turn_meta.get("recovered"), "reask",
                             "the dead pass is recovered by a re-ask, not by the extraction")

        asyncio.run(body())

    def test_no_extraction_recovery_when_the_key_is_missing(self):
        # `missing_api_key` means nothing was read at all: an offline extractor returning `ok` must not fake a
        # recovery here, or a keyless server would advance call state it never actually consulted a model for.
        async def body() -> None:
            built, _ws, _session = build(call_id=None, state=self.roster_hold_state())

            async def no_key(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                turn = turn_with(snapshot, TurnOutcome.UNKNOWN, spoken)
                return turn._replace(result=turn.result._replace(ok=False, error_reason="missing_api_key", attempts=1),
                                     extraction=live.extractor.ProfileExtractionResult(True, self.aman_and_meera()))

            built._evaluate_turn_task = no_key  # type: ignore[method-assign]
            await built.on_text({"text": self.ROSTER_UTTERANCE})
            payload, _clarify = await built._next_question()
            self.assertEqual(payload["result"], "failed", "a keyless server closes the call, it does not advance it")
            self.assertIs(built.state.close_reason, CloseReason.TECHNICAL_FAILURE)
            self.assertEqual(built.state.cursor.qid, "intro_children", "and the roster hold is untouched")

        asyncio.run(body())

    def test_call16_retry_turn_completes_truncated_reask_transcript_and_strike3_emits_failed(self):
        # Call #16 transcript `[2]`: Gemini spoke the reask, wrote only "जी, आराम", and the completion check
        # compared against `say_text` — so the truncated caption stuck. And event `[51]`: the third strike's
        # `tool_call` still read `next_question_ok` while the call was being closed for real.
        async def body() -> None:
            retry_session, retry_ws, _retry_sock = build(call_id=None)
            state = retry_session.state
            reask = live._reask_say(PACK, state)
            self.assertTrue(reask.startswith("जी, आराम से"), f"this step's reask is the line Gemini speaks: {reask!r}")
            self.assertNotEqual(reask, live.engine.say_text(PACK, state), "the retry cue says something else entirely")

            await retry_session._publish_and_cue("retry", immediate=False)
            await retry_session.on_message(OutputTranscriptionTests.said("जी, आराम"))
            await retry_session.on_message(OutputTranscriptionTests.audio())
            await retry_session.on_message(OutputTranscriptionTests.complete())

            self.assertEqual(retry_session.bot_text, reask, "the truncated reask caption is completed to the real line")
            self.assertEqual([p["text"] for p in retry_ws.of("transcript") if p["role"] == "assistant"],
                             ["जी, आराम", reask], "and the browser is handed the whole sentence")

            # Strike 3 on a step that has nothing to extract: the call closes, and the tool_call must say so.
            failed_session, failed_ws, failed_sock = build(call_id=None)
            failed_state = live.engine.new_state(PACK)
            failed_state.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male"),
                                   RosterItem(id="item_2", age_years=6.0, age_label="6 साल", tag="female")]
            failed_state.roster_stated = True
            failed_state.cursor = CursorStep(qid="birth_nicu", substep=SubStep.WHICH_ITEM)
            failed_session.state = failed_state

            async def always_timeout(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                turn = turn_with(snapshot, TurnOutcome.UNKNOWN, spoken)
                return turn._replace(result=turn.result._replace(ok=False, error_reason="timeout", attempts=2))

            failed_session._evaluate_turn_task = always_timeout  # type: ignore[method-assign]
            for attempt in (1, 2, 3):
                await failed_session.on_text({"text": "नहीं"})
                await failed_session._answer_tools(
                    [live.types.FunctionCall(id=f"t{attempt}", name="next_question", args={})])

            details = [p["last_event"]["detail"] for p in failed_ws.of("telemetry")
                       if p["last_event"]["type"] == "tool_call"]
            self.assertEqual(details[-3:], ["next_question_retry", "next_question_retry", "next_question_failed"],
                             "the closing turn is reported as a failure, never as a healthy answer")
            closing = {r.id: r.response for kind, payload in failed_sock.sent if kind == "tool"
                       for r in payload}["t3"]
            self.assertEqual(closing["result"], "failed")
            self.assertIs(closing["stop"], True, "the caller is still owed the closing line")
            self.assertIs(failed_session.state.close_reason, CloseReason.TECHNICAL_FAILURE)
            self.assertTrue(closing["say"], "and that closing line is not empty")

        asyncio.run(body())


def bridged_state() -> CallState:
    """A call held on `intro_bridge`, with the profile and roster already resolved so a tap advances for real."""
    state = live.engine.new_state(PACK)
    state.profile.update({"caller_name": "कविता", "num_children": 1})
    state.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male")]
    state.roster_stated = True
    state.answers["intro_married"] = AnswerRecord(state=AnswerStatus.CLEAR)
    state.cursor = CursorStep(qid="intro_bridge")
    return state


class PromptMuteWindowTests(unittest.TestCase):
    """The tap-to-first-audio gap and the dead air after an aborted turn are the same hole: her echo must not open a turn."""

    def test_caller_speech_in_post_tap_pre_audio_gap_is_muted_until_next_question_plays(self):
        async def body() -> None:
            built, _ws, session = build(call_id=None, state=bridged_state())
            await built._publish_and_cue("speak", immediate=False)
            await built.on_message(OutputTranscriptionTests.audio())
            self.assertTrue(built.bot_generating, "SHISHU is mid-sentence on the bridge")

            await built.on_quick_reply({"value": "concern", "step": built.state.step_token})
            self.assertEqual(built.state.cursor.qid, "birth_mode_detail", "the tap moved the call on")
            self.assertIs(built.turn_phase, live.TurnPhase.PROMPTING)

            forwarded = len(session.realtime("audio"))
            for _ in range(3):  # the browser's flushed queue still holds her voice
                await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertEqual(built.pcm, b"", "a post-tap echo burst is never banked as a caller turn")
            self.assertEqual(len(session.realtime("audio")), forwarded, "and never reaches Gemini as one")
            self.assertEqual(list(built.preroll), [], "nor may its onset seed the turn after it")

            # The aborted turn dies here. Its death must not unlock the mic: no audio has played for the new question.
            await built.on_message(OutputTranscriptionTests.interrupted())
            await built.on_message(OutputTranscriptionTests.complete())
            for _ in range(3):
                await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertEqual(built.pcm, b"", "the mute outlives the abort it was waiting on")
            self.assertEqual(len(session.realtime("audio")), forwarded)

            built.prompted_at = time.time() - 5.1  # a silent model: the 5.0 s safety valve has elapsed
            self.assertTrue(await built.tick(time.time()))
            self.assertIs(built.turn_phase, live.TurnPhase.LISTENING, "tick promotes the prompt to a listening turn")
            await built.on_audio({"data": pcm(), "sample_rate": 16000})
            self.assertTrue(built.pcm, "and the caller is heard again")
            self.assertEqual(len(session.realtime("audio")), forwarded + 1)
            built._cancel("fallback")

        asyncio.run(body())


class DecidingPhaseTests(unittest.TestCase):
    """While the Decision pass is in flight the call is neither listening nor prompting: it is deciding."""

    def test_deciding_phase_emits_thinking_and_ignores_manual_nudge(self):
        async def body() -> None:
            built, ws, session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            gate = asyncio.Event()

            async def slow_pass(snapshot: CallState, spoken: str, pcm: bytes) -> live.Turn:
                await gate.wait()
                return turn_with(snapshot, TurnOutcome.CLEAR, spoken)

            built._evaluate_turn_task = slow_pass  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            answering = asyncio.ensure_future(
                built._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})]))
            await asyncio.sleep(0)

            self.assertIs(built.turn_phase, live.TurnPhase.DECIDING)
            self.assertEqual(ws.of("thinking"),
                             [{"type": "thinking", "epoch": built.epoch, "step": built.state.step_token}],
                             "the browser is told a Decision pass is running for this exact step")

            spoken = len(session.realtime("text"))
            self.assertTrue(await built.on_nudge({}), "a nudge in flight is simply dropped")
            self.assertEqual(session.realtime("text")[spoken:], [],
                             "a manual nudge must never queue a second cue over the pass")
            self.assertNotIn("manual_nudge", [p["last_event"]["type"] for p in ws.of("telemetry")])

            gate.set()
            await answering

        asyncio.run(body())


class TurnEpochTests(unittest.TestCase):
    """Every server-side step carries an epoch, so a stale tap or a late chunk can be recognised as stale."""

    def test_every_step_transition_increments_epoch_on_wire_payloads(self):
        async def body() -> None:
            built, ws, session = build(call_id=None)
            self.assertEqual(built.epoch, 0, "a freshly constructed session has published nothing yet")

            await built._publish_and_cue("speak", immediate=False)  # the greeting, exactly as `run()` sends it
            self.assertEqual(ws.of("question_state")[-1]["epoch"], 1)
            await built.on_message(OutputTranscriptionTests.audio())
            self.assertEqual(ws.of("audio")[-1]["epoch"], 1, "her audio belongs to the step it speaks")
            await built.on_message(OutputTranscriptionTests.complete())
            self.assertEqual(ws.of("bot_turn_complete")[-1]["epoch"], 1)

            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            built._evaluate_turn_task = TurnMetaTests.fake_pass(TurnOutcome.CLEAR)  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            await built._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})])
            self.assertEqual(ws.of("thinking")[-1]["epoch"], 1, "the pass ran against the step it started on")
            self.assertEqual(ws.of("question_state")[-1]["epoch"], 2, "the next question is a new epoch")
            self.assertEqual(built.epoch, 2)

            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            self.assertEqual([p["epoch"] for p in ws.of("clear")], [2], "the tap flushes the epoch it left")
            self.assertEqual(ws.of("question_state")[-1]["epoch"], 3)
            self.assertEqual(built.epoch, 3)

            live_clears, live_states = len(ws.of("clear")), len(ws.of("question_state"))
            stale_step = built.state.step_token
            await built.on_quick_reply({"value": "clear", "step": stale_step, "epoch": 1})
            self.assertEqual(len(ws.of("clear")), live_clears, "a tap from an old epoch is ignored outright")
            self.assertEqual(len(ws.of("question_state")), live_states)
            self.assertEqual(built.state.step_token, stale_step, "and it never moves the call")
            self.assertEqual(built.epoch, 3)

            for kind in ("question_state", "audio", "bot_turn_complete", "thinking", "clear"):
                for payload in ws.of(kind):
                    self.assertIs(type(payload["epoch"]), int, f"{kind} carries an integer epoch")

        asyncio.run(body())


class Wave2CueLifecycleAndResumeTelemetryTests(unittest.TestCase):
    """Wave 2 invariants: one owner per prompt cue, one recovery per epoch, and a resume that keeps its numbers.

    Call #21 spoke `ill_fasting` twice 2.2 s apart (the greeting's own cue plus a `_cue_if_silent` collision) and
    Call #32 spoke it twice again on resume, while that resume also zeroed every counter the call had earned. Each
    bug lived in one cell of an entrypoint x latency matrix, so these tests sweep the matrix instead of one path.
    """

    CUE = 0.04  # 40 ms: the real `_CUE_DELAY` is 2.2 s, and the whole sweep has to stay under half a second
    FAST, COLLISION, SLOW = 0.005, 0.045, 0.065  # audio strictly before / just after / well after the cue delay

    @staticmethod
    def hold(qid: str = "fam_cancer") -> CallState:
        """A resolved, bridged call parked on `qid`, so every entrypoint publishes a real step."""
        state = live.engine.new_state(PACK)
        state.profile.update({"caller_name": "मीरा", "num_children": 1, "marital_status": "married"})
        state.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="female")]
        state.roster_stated = True
        state.answers["intro_married"] = AnswerRecord(state=AnswerStatus.CLEAR)
        state.answers["intro_bridge"] = AnswerRecord(state=AnswerStatus.CLEAR)
        state.cursor = CursorStep(qid=qid)
        return state

    @staticmethod
    def audio() -> live.types.LiveServerMessage:
        return OutputTranscriptionTests.audio()

    @staticmethod
    def interrupted() -> live.types.LiveServerMessage:
        return OutputTranscriptionTests.interrupted()

    @staticmethod
    def complete() -> live.types.LiveServerMessage:
        return OutputTranscriptionTests.complete()

    @staticmethod
    def heard(session: FakeSession) -> list[str]:
        """Every text cue that reached the model, in wire order."""
        return [str(text) for text in session.realtime("text")]

    @staticmethod
    async def run_cells(cells: list[Any]) -> list[Any]:
        """Run every matrix cell on its own task: the sweep costs its slowest cell, not the sum of all of them."""
        return await asyncio.gather(*(cell() for cell in cells), return_exceptions=True)

    def replay(self, labels: list[dict[str, Any]], results: list[Any]) -> None:
        """Re-raise each cell's failure under its own subTest, so every broken cell is named on its own."""
        for label, result in zip(labels, results):
            with self.subTest(**label):
                if isinstance(result, BaseException):
                    raise result

    @staticmethod
    async def opened(
        built: live.LiveCallSession, session: FakeSession, *, timeout: float = 2.0
    ) -> asyncio.Task:
        """Drive the REAL `run()` and return once its opening cue is on the wire, so the code under test is the code."""
        task = asyncio.ensure_future(built.run())
        deadline = time.monotonic() + timeout
        while not session.realtime("text") and time.monotonic() < deadline:
            await asyncio.sleep(0.002)
        return task

    @staticmethod
    async def stop_run(built: live.LiveCallSession, task: asyncio.Task) -> None:
        built.stop.set()
        built._cancel("fallback")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def test_single_owner_prompt_cues_across_entrypoints_and_response_latencies(self):
        # Call #21 `[5]`->`[7]`: the greeting's own send went out at t=0 and the fallback timer fired a second
        # `speak_cue` 2.2 s later, so Gemini said `ill_fasting` twice into a live line; Call #32 did it again on a
        # resume. One step owns exactly one cue, and only a step whose cue nobody heard may rescue itself, once.
        async def body() -> None:
            with unittest.mock.patch.object(live, "_CUE_DELAY", self.CUE):
                # --- immediate-cue entrypoints: fresh greeting, resumed call, idle quick reply -----------------
                labels: list[dict[str, Any]] = []
                cells = []

                def immediate_cell(entry: str, resumed: bool, delay: float) -> Any:
                    async def cell() -> None:
                        built, _ws, session = build(call_id=None, state=self.hold())
                        built.resumed = resumed
                        run_task = None
                        try:
                            if entry == "quick_reply_idle":
                                self.assertFalse(built.bot_generating, "no bot turn is open when the caller taps")
                                await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
                            else:
                                run_task = await self.opened(built, session)  # the real `run()`, not a copy of it

                            self.assertEqual(len(self.heard(session)), 1,
                                             f"{entry}: opening the step owns exactly one cue; no second `_speak` is owed")

                            await asyncio.sleep(delay)  # the model's first audio, or its silence, arrives about here
                            if delay < self.CUE:
                                await built.on_message(self.audio())
                            await asyncio.sleep(self.CUE + 0.02)  # well past `_CUE_DELAY`: every fallback has fired
                            built._cancel("fallback")
                            self.assertEqual(len(self.heard(session)), 1,
                                             f"{entry} at {round(delay * 1000)} ms: an already-sent cue must never be "
                                             f"spoken a second time ({self.heard(session)})")
                        finally:
                            if run_task is not None:
                                await self.stop_run(built, run_task)
                    return cell

                for entry, resumed in (("fresh_greeting", False), ("resumed_call", True), ("quick_reply_idle", False)):
                    for delay in (self.FAST, self.COLLISION, self.SLOW):
                        labels.append({"entry": entry, "latency_ms": round(delay * 1000)})
                        cells.append(immediate_cell(entry, resumed, delay))
                self.replay(labels, await self.run_cells(cells))

                # --- aborting quick-reply entrypoint: the tap lands while Turn A is still on the wire -----------
                labels, cells = [], []

                def aborting_cell(delay: float) -> Any:
                    async def cell() -> None:
                        built, _ws, session = build(call_id=None, state=self.hold())
                        await built._publish_and_cue("speak", immediate=False)  # the step is published, no cue yet
                        await built.on_message(self.audio())  # Turn A is speaking it: `bot_generating`
                        self.assertTrue(built.bot_generating)
                        self.assertIsNone(built.fallback, "her own audio cancelled the pre-turn fallback")

                        await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
                        self.assertTrue(built.awaiting_abort_end, "the tap knows Turn A may still be dying")
                        self.assertEqual(len(self.heard(session)), 1, "the tap's cue is the step's one cue")
                        self.assertIsNone(built.fallback, "no timer may tick while Turn A is still dying")

                        await asyncio.sleep(self.CUE + 0.015)  # > `_CUE_DELAY`, with Turn A still alive
                        self.assertEqual(len(self.heard(session)), 1,
                                         "not even `_CUE_DELAY` may speak over a turn that has not died yet")

                        await built.on_message(self.interrupted())  # and now Turn A's death finally arrives
                        await built.on_message(self.complete())
                        self.assertFalse(built.awaiting_abort_end, "the abort is acknowledged")
                        if delay < self.CUE:  # subcase A: Turn B's audio lands inside the rescue window
                            await asyncio.sleep(delay)
                            await built.on_message(self.audio())
                        await asyncio.sleep(self.CUE + 0.02)
                        built._cancel("fallback")
                        heard = self.heard(session)
                        if delay < self.CUE:
                            self.assertEqual(len(heard), 1,
                                             f"Turn B answered in {round(delay * 1000)} ms, so the cue was heard "
                                             f"and the post-abort rescue must stand down ({heard})")
                        else:  # subcase B: Gemini swallowed the cue with the abort and stayed silent
                            self.assertEqual(len(heard), 2,
                                             f"a model still silent {round(delay * 1000)} ms after the abort owes "
                                             f"exactly one rescue cue, never zero and never two ({heard})")
                    return cell

                for delay in (self.FAST, self.SLOW):
                    labels.append({"entry": "quick_reply_aborting_bot_turn", "latency_ms": round(delay * 1000)})
                    cells.append(aborting_cell(delay))
                self.replay(labels, await self.run_cells(cells))

                # --- tool-response entrypoint: `immediate=False`, so only real silence may earn a cue -----------
                labels, cells = [], []

                def tool_cell(delay: float) -> Any:
                    async def cell() -> None:
                        built, _ws, session = build(call_id=None, state=self.hold("birth_cord_3weeks"))

                        async def fake_pass(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                            return turn_with(snapshot, TurnOutcome.CLEAR, spoken)._replace(pcm_len=len(pcm_bytes))

                        built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
                        await built.on_text({"text": "नहीं"})  # her typed answer rides to Gemini as text
                        tapped = len(self.heard(session))
                        await built._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})])
                        self.assertEqual(len([kind for kind in session.kinds() if kind == "tool"]), 1,
                                         "the FunctionResponse itself is what Gemini speaks the line from")
                        self.assertEqual(self.heard(session)[tapped:], [],
                                         "a tool turn never pre-empts its own FunctionResponse with a cue")

                        if delay < self.CUE:
                            await asyncio.sleep(delay)
                            await built.on_message(self.audio())
                        await asyncio.sleep(self.CUE + 0.02)
                        built._cancel("fallback")
                        heard = self.heard(session)[tapped:]
                        if delay < self.CUE:
                            self.assertEqual(heard, [],
                                             f"audio in {round(delay * 1000)} ms spoke the turn: no cue is owed ({heard})")
                        else:
                            self.assertEqual(len(heard), 1,
                                             f"a tool turn silent past `_CUE_DELAY` owes exactly one rescue cue ({heard})")
                    return cell

                for delay in (self.FAST, self.SLOW):
                    labels.append({"entry": "tool_call_response", "latency_ms": round(delay * 1000)})
                    cells.append(tool_cell(delay))
                self.replay(labels, await self.run_cells(cells))

        asyncio.run(body())

    def test_self_interrupted_prompting_epoch_recues_once_without_stranding_in_dead_air(self):
        # Call #21's staging wire: `sc.interrupted` arrived with NO caller turn open and NO `turn_complete` after
        # it, because Gemini cancelled her own prompt echo. Nothing re-owed the step and the caller sat in dead air.
        # The recovery is bounded: one cue per epoch, however many interrupts Gemini repeats in that epoch.
        async def body() -> None:
            with unittest.mock.patch.object(live, "_CUE_DELAY", self.CUE):
                labels: list[dict[str, Any]] = []
                cells = []

                def epoch_cell(wire: str, entry: str) -> Any:
                    async def cell() -> None:
                        built, ws, session = build(call_id=None, state=self.hold())
                        await built._publish_and_cue("speak", immediate=False)
                        built._cancel("fallback")

                        if entry == "quick_reply":
                            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
                        else:
                            await built._publish_and_cue("speak", immediate=False)
                        self.assertIs(built.turn_phase, live.TurnPhase.PROMPTING)
                        sent, before, epoch = len(self.heard(session)), len(ws.of("bot_turn_complete")), built.epoch
                        kind, cue = built.cue_kind, live.speak_cue(PACK, built.state, built.cue_kind)

                        await built.on_message(self.interrupted())  # Gemini cancels her own prompt echo
                        if wire == "interrupted_then_complete":
                            await built.on_message(self.complete())
                        await asyncio.sleep(0.01)

                        self.assertEqual(self.heard(session)[sent:], [cue],
                                         f"{wire}/{entry}: the aborted prompt owes ONE recovery cue, verbatim "
                                         f"`speak_cue(PACK, state, {kind!r})`")
                        # Her own echo dying is not the caller barging in: counting it inflates Call #32's
                        # interruption numbers, and the event log must say what actually happened.
                        self.assertEqual(built.telemetry["interruptions"], 0,
                                         f"{wire}/{entry}: a self-interrupt is not a caller barge-in")
                        self.assertEqual([e["type"] for e in (p["last_event"] for p in ws.of("telemetry"))
                                          if e["type"] in ("barge_in", "prompt_recue")],
                                         ["prompt_recue"],
                                         f"{wire}/{entry}: the recovery is logged as `prompt_recue`, never `barge_in`")
                        self.assertIs(built.turn_phase, live.TurnPhase.PROMPTING,
                                      "the step is still owed: the mic stays shut until her echo starts")
                        self.assertEqual(ws.of("bot_turn_complete")[before:], [],
                                         "an aborted prompt is not a completed bot turn, even in wire order 2")
                        self.assertEqual(built.epoch, epoch, "the recovery never bumps the epoch it rescues")

                        await built.on_message(self.interrupted())  # Gemini repeats itself in the same epoch
                        await asyncio.sleep(0.01)
                        self.assertEqual(self.heard(session)[sent:], [cue],
                                         "a second interrupt in the same epoch must not speak a second cue")

                        # The next epoch resets the guard, so the new step can also rescue itself once.
                        if entry == "quick_reply":
                            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
                        else:
                            await built._publish_and_cue("speak", immediate=False)
                        self.assertNotEqual(built.epoch, epoch, "the next step is a new epoch")
                        sent2, epoch2 = len(self.heard(session)), built.epoch
                        await built.on_message(self.interrupted())
                        await asyncio.sleep(0.01)
                        self.assertEqual(len(self.heard(session)), sent2 + 1,
                                         "a new epoch earns its own single recovery")
                        await built.on_message(self.interrupted())
                        await asyncio.sleep(0.01)
                        self.assertEqual(len(self.heard(session)), sent2 + 1,
                                         "and only one, however many interrupts follow it")
                        self.assertEqual(built.epoch, epoch2)

                    return cell

                for wire in ("interrupted_only", "interrupted_then_complete"):
                    for entry in ("quick_reply", "_publish_and_cue"):
                        labels.append({"wire": wire, "entry": entry})
                        cells.append(epoch_cell(wire, entry))
                self.replay(labels, await self.run_cells(cells))

                # Control: an interrupt over a live caller turn is a barge-in, never a recovery.
                labels, cells = [], []

                def control_cell(control: str) -> Any:
                    async def cell() -> None:
                        built, _ws, session = build(call_id=None, state=self.hold())
                        await built._publish_and_cue("speak", immediate=False)
                        built._cancel("fallback")
                        if control == "listening_phase":
                            built.turn_phase = live.TurnPhase.LISTENING
                        elif control == "caller_pcm":
                            built.turn_phase = live.TurnPhase.LISTENING  # the caller owns the turn: the mic is open
                            await built.on_audio({"data": pcm(), "sample_rate": 16000})
                            self.assertTrue(built.pcm, "the caller's audio is banked")
                            built.turn_phase = live.TurnPhase.PROMPTING
                        elif control == "caller_text":
                            await built._hear("user", "हाँ, मेरी बुआ को कैंसर हुआ था")
                            self.assertTrue(built.text, "the caller's words are banked")
                            built.turn_phase = live.TurnPhase.PROMPTING
                        elif control == "bundled_input_transcription":
                            # Soft speech sends no banked PCM, so `pcm` is empty and only the transcription in this
                            # very message proves the caller spoke: the interrupt is hers, not her own echo's.
                            self.assertEqual((built.pcm, built.text), (b"", ""), "nothing is banked yet")
                            self.assertIs(built.turn_phase, live.TurnPhase.PROMPTING)
                        else:
                            built.awaiting_abort_end = True
                        sent = len(self.heard(session))
                        if control == "bundled_input_transcription":
                            await built.on_message(live.types.LiveServerMessage(server_content=live.types.LiveServerContent(
                                interrupted=True, input_transcription=live.types.Transcription(text="हाँ"))))
                        else:
                            await built.on_message(self.interrupted())
                        await asyncio.sleep(0.01)
                        self.assertEqual(len(self.heard(session)), sent,
                                         f"{control}: she interrupted over an open caller turn, so no cue is owed")
                        if control == "bundled_input_transcription":
                            self.assertEqual(built.text, "हाँ",
                                             "the caller's words in the same message are banked, never overwritten")

                    return cell

                for control in ("listening_phase", "caller_pcm", "caller_text", "bundled_input_transcription",
                                "awaiting_abort_end"):
                    labels.append({"control": control})
                    cells.append(control_cell(control))
                self.replay(labels, await self.run_cells(cells))

        asyncio.run(body())

    def test_ws_resume_preserves_telemetry_across_dropped_and_lingering_sockets_and_redacts_multi_topic_late_stt(self):
        # Call #32: the browser rejoined mid-call and every counter the call had earned came back as 0 / [] / 0, so
        # the finished call read like a call that never happened. The caller's audio was still on disk; only the
        # numbers were lost. And the trailing chunk for a second topic came back as patient-safe words.
        EARNED = {"user_turns": 3, "bot_turns": 4, "tool_calls": 3, "audio_chunks_in": 42, "audio_chunks_out": 18,
                  "turn_latencies_ms": [680, 920], "avg_latency_ms": 800, "max_latency_ms": 920, "errors": ["minor"]}

        async def body() -> None:
            async def resume_case(lingering: bool) -> None:
                label = "lingering half-open socket" if lingering else "dropped socket in grace"

                class DummyAsyncCtx:
                    async def __aenter__(self) -> Any:
                        return FakeSession()

                    async def __aexit__(self, *_exc: Any) -> bool:
                        return False

                async def fake_open(_client: Any, _pack: Any, _handle: str) -> Any:
                    return DummyAsyncCtx(), FakeSession()

                ws1, ws2 = FakeWebSocket(), FakeWebSocket()
                first = second = None
                created: list[int] = []
                original = live._open_live
                live._open_live = fake_open  # type: ignore[assignment]
                try:
                    with unittest.mock.patch.dict(live.os.environ, {"GEMINI_API_KEY": "k"}):
                        first = asyncio.ensure_future(live.handle_live_ws(ws1, pack=PACK, caller_name="Tester"))
                        self.assertTrue(await ResumeSessionTests.wait_until(lambda: bool(ws1.of("session_started"))),
                                        f"{label}: the call must start")
                        started = ws1.of("session_started")[0]
                        cid, share_token = int(started["call_id"]), str(started["share_token"])
                        created.append(cid)
                        self.assertTrue(await ResumeSessionTests.wait_until(lambda: cid in live._ACTIVE_CALLS))

                        active = live._ACTIVE_CALLS[cid]
                        active.telemetry |= dict(EARNED)  # a real call's worth of turns, audio and latency
                        active.state.cursor = active.state.cursor.model_copy(update={"qid": "ill_fasting"})
                        await active._persist(active.state)
                        await active._db(db.update_call_telemetry, dict(active.telemetry))

                        # The socket dies without any `hangup` message: the line dropped, the browser did not leave.
                        await ws1.queue.put(None)
                        await asyncio.wait_for(first, timeout=3)
                        first = None
                        if lingering:
                            # A session whose socket died unnoticed is still holding the call in the registry.
                            if (pending := live._PENDING_FINALIZE.pop(cid, None)) is not None:
                                pending.cancel()
                                await asyncio.gather(pending, return_exceptions=True)
                            live._ACTIVE_CALLS[cid] = live.LiveCallSession(
                                ws=FakeWebSocket(), pack=PACK, session=FakeSession(), call_id=cid,
                                state=active.state.model_copy(deep=True), recorder=FakeRecorder())
                        else:
                            self.assertIn(cid, live._PENDING_FINALIZE, f"{label}: the call waits in resume grace")

                        second = asyncio.ensure_future(live.handle_live_ws(ws2, pack=PACK, resume_token=share_token))
                        self.assertTrue(await ResumeSessionTests.wait_until(
                            lambda: bool(ws2.of("session_started")) and bool(ws2.of("question_state"))),
                            f"{label}: the rejoin must be handed its call and its next step")
                        self.assertNotIn(cid, live._PENDING_FINALIZE, f"{label}: resuming cancels the grace timer")
                        resumed = live._ACTIVE_CALLS[cid]
                        self.assertEqual(ws2.of("session_started")[0]["call_id"], cid, "no second call row is minted")

                        tele = resumed.telemetry
                        self.assertEqual({k: tele[k] for k in ("user_turns", "bot_turns", "tool_calls",
                                                               "audio_chunks_in", "audio_chunks_out")},
                                         {k: EARNED[k] for k in ("user_turns", "bot_turns", "tool_calls",
                                                                 "audio_chunks_in", "audio_chunks_out")},
                                         f"{label}: a resume restores the counters the call already earned")
                        self.assertEqual(tele["turn_latencies_ms"], [680, 920], f"{label}: the latency samples survive")
                        self.assertEqual(tele["avg_latency_ms"], 800, f"{label}: and so does their average")
                        self.assertEqual(tele["max_latency_ms"], 920, f"{label}: and the worst turn")
                        self.assertEqual(tele["errors"], ["minor"], f"{label}: an error is never quietly forgiven")

                        await resumed.on_message(self.audio())  # one more bot chunk, now on the new socket
                        self.assertEqual(resumed.telemetry["audio_chunks_out"], 19,
                                         f"{label}: new audio counts on top of the restored total")

                        await ws2.push({"type": "hangup"})
                        await asyncio.wait_for(second, timeout=3)
                        second = None
                        row = await asyncio.to_thread(db.get_call, cid)
                        self.assertEqual(row["telemetry"]["audio_chunks_out"], 19,
                                         f"{label}: the resumed total is what the finished call reports")
                        self.assertEqual(row["telemetry"]["turn_latencies_ms"], [680, 920],
                                         f"{label}: and the latency samples are on the finished row")
                finally:
                    live._open_live = original  # type: ignore[assignment]
                    for task in (first, second):
                        if task is not None and not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    for call_id in created:
                        with db._LOCK:
                            db.conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
                            db.conn.commit()
                        live._ACTIVE_CALLS.pop(call_id, None)
                        if (pending := live._PENDING_FINALIZE.pop(call_id, None)) is not None:
                            pending.cancel()
                            await asyncio.gather(pending, return_exceptions=True)

            async def late_stt_case() -> None:
                built, ws, _session = build(call_id=None, state=self.hold())
                utterance = "हां, बुआ को कैंसर हुआ था और मीरा को जन्म पर पीलिया हुआ था"

                async def fake_pass(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                    intent = live.decisions.TurnIntent(
                        command=TurnCommand(outcome=TurnOutcome.CONCERN, stated={"birth_jaundice": TurnOutcome.CONCERN},
                                            followup_answered=True, utterance=utterance, redacted=False),
                        needs_profile_extraction=False, diagnostics={})
                    return live.Turn(TurnTests.result(answers(danger_now=0.01)), intent, None,
                                     snapshot.model_copy(deep=True), spoken, len(pcm_bytes), None)

                built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
                await built.on_audio({"data": pcm(), "sample_rate": 16000})
                await built._hear("user", utterance)
                await built._next_question()
                fam = built.state.answers["fam_cancer"]
                self.assertIs(fam.state, AnswerStatus.CONCERN, "the family cancer question is answered")
                self.assertIn(utterance, fam.verbatim_text, "and the doctor's record keeps her exact words")
                self.assertEqual(fam.patient_safe_text, "", "a multi-topic sentence is nobody's patient-safe quote")
                self.assertEqual(built.last_reduced[0], "fam_cancer", "the turn is booked against the question it answered")

                await built._hear("user", "नीली लाइट में")  # the trailing `late_stt` chunk, about the jaundice topic
                self.assertEqual(ws.of("telemetry")[-1]["last_event"]["type"], "late_stt",
                                 "the trailing chunk is booked as late STT against the question it answered")
                fam = built.state.answers["fam_cancer"]
                self.assertIn("नीली लाइट में", fam.verbatim_text,
                              "the late chunk must reach the doctor's record")
                self.assertEqual(fam.patient_safe_text, "",
                                 "but a chunk about another topic may never reach the mother's PDF as a quote")

            with self.subTest(case="A: dropped socket finalized into grace before rejoin"):
                await resume_case(lingering=False)
            with self.subTest(case="B: lingering half-open socket still in _ACTIVE_CALLS on rejoin"):
                await resume_case(lingering=True)

            # --- Subtest C: a late chunk about another topic may never become patient-safe words -----------------
            with self.subTest(case="C: multi-topic late_stt hygiene on fam_cancer"):
                await late_stt_case()

        asyncio.run(body())


class Wave3AcousticAndPlaybackTailTests(unittest.TestCase):
    """Wave 3 invariants: the voice gate must follow the room, and her playback tail must end where her queue ends.

    Two shapes of the same failure. In Call #30 a soft speakerphone (`rms` 0.006–0.014) never crossed the fixed
    0.015 gate, so a whole spoken turn reached the pass as text with `pcm_len == 0`, and the frames before and
    after its one vowel were never banked at all. In Call #36 a steady clinic fan (`rms` 0.022) crossed that same
    gate on every frame, so 36 s of silence counted as one endless caller turn: no check-in, no silence close, and
    a `pcm` full of fan. And in Calls #33/#34/#41 an eager `"नहीं"` landed in the last 300 ms of her line, in the
    one window where the browser already knows the turn is over, yet the server still called it her voice.

    Each regime is a real recorded envelope, and each cell of the matrix is its own session, so one broken regime
    names itself instead of hiding behind the first failure.
    """

    TAIL_MS = 350  # the last 350 ms of a completed bot turn: the browser has the whole line, she may be eager

    class Clock:
        """A simulated `time.time`: 40 s of acoustic regime costs microseconds, never 40 real seconds."""

        def __init__(self, now: float = 100.0) -> None:
            self.now = now

        def __call__(self) -> float:
            return self.now

    @staticmethod
    def rms_frame(target_rms: float, samples: int = 1024) -> bytes:
        """One 64 ms frame of 16 kHz int16 mono whose `audio.pcm_rms` IS `target_rms`, to within 0.0005."""
        frame = round(target_rms * 32768).to_bytes(2, "little", signed=True) * samples
        assert abs(live.audio.pcm_rms(frame) - target_rms) <= 0.0005, (target_rms, live.audio.pcm_rms(frame))
        assert len(frame) == samples * 2, "16-bit mono: two bytes per sample"
        return frame

    @staticmethod
    async def stream(built: live.LiveCallSession, frame: bytes, count: int = 1) -> None:
        """Hand `built` `count` identical mic frames, exactly as the browser sends them."""
        payload = base64.b64encode(frame).decode()
        for _ in range(count):
            await built.on_audio({"data": payload, "sample_rate": 16000})

    @staticmethod
    def spy_decision_pass(built: live.LiveCallSession) -> list[int]:
        """Replace the one Decision pass with a recorder, so the test measures the audio it was really handed."""
        captured: list[int] = []

        async def fake_pass(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
            captured.append(len(pcm_bytes))
            intent = live.decisions.TurnIntent(command=TurnCommand(outcome=TurnOutcome.CLEAR, utterance=spoken),
                                               needs_profile_extraction=False, diagnostics={})
            return live.Turn(TurnTests.result(answers(danger_now=0.01)), intent, None,
                             snapshot.model_copy(deep=True), spoken, len(pcm_bytes), None)

        built._evaluate_turn_task = fake_pass  # type: ignore[method-assign]
        return captured

    class Timeline:
        """Every telemetry event the socket reported, stamped with the second of the simulated regime it landed on."""

        def __init__(self) -> None:
            self.stamped: list[tuple[float, str]] = []
            self.seen = 0

        def capture(self, ws: FakeWebSocket, now: float) -> None:
            reported = [p["last_event"] for p in ws.of("telemetry") if p.get("last_event")]
            self.stamped.extend((now, str(e.get("type") or "")) for e in reported[self.seen:])
            self.seen = len(reported)

        def kinds(self) -> list[str]:
            return [kind for _t, kind in self.stamped]

        def at(self, kind: str) -> float:
            return next(t for t, seen in self.stamped if seen == kind)

    async def regime_a_soft_speakerphone(self) -> None:
        # Call #30, `soft_speakerphone_006`: 1.6 s of real speech at `rms` 0.006, a fifth of the fixed gate.
        built, _ws, _session = build(call_id=None)
        clock = self.Clock(100.0)
        captured = self.spy_decision_pass(built)
        frame = self.rms_frame(0.006)
        self.assertLess(live.audio.pcm_rms(frame), live._ONSET_MIN, "this regime IS under the onset floor: that is the bug")
        with unittest.mock.patch.object(live.time, "time", clock):
            await self.stream(built, frame, 25)  # 25 x 64 ms = 1600 ms
            await built._hear("user", "मेरा नाम कविता है")
            await built._next_question()
        self.assertEqual(built.telemetry["user_turns"], 1, "one soft utterance is still one caller turn")
        self.assertEqual(captured, [25 * len(frame)],
                         "every one of the 25 contiguous frames must reach the pass; a text-only turn has no audio "
                         "for the model to hear and `audio_ms` reads 0 in the meta row")
        self.assertEqual(built.last_turn_meta.get("audio_ms"), 1600,
                         "and the turn is booked as the 1600 ms she actually spoke")

    async def regime_b_soft_speakerphone_single_peak(self) -> None:
        # Call #30 again: one vowel peaking at 0.014 is the ONLY landmark in an otherwise sub-onset phrase, and the
        # 20 frames of 0.007 speech that follow it are quieter than any tail the floor had learned to expect. A
        # floor that creeps up on frames it can never gate against raises `tail_rms` above 0.007, so those frames
        # fall to the 8-chunk tail budget and the last 12 are chopped — and `bool(self.pcm)` then blocks the
        # `_next_question` ring fallback, so the loss is silent.
        built, _ws, _session = build(call_id=None)
        clock = self.Clock(100.0)
        captured = self.spy_decision_pass(built)
        with unittest.mock.patch.object(live.time, "time", clock):
            await self.stream(built, self.rms_frame(0.002), 4)    # quiet room, before anything is said
            await self.stream(built, self.rms_frame(0.008), 10)   # soft speech before the peak
            await self.stream(built, self.rms_frame(0.014), 1)    # the one vowel: the turn's only onset
            await self.stream(built, self.rms_frame(0.007), 20)   # soft speech after it, quieter than the lead-in
            self.assertGreaterEqual(len(built.pcm), 31 * 2048,
                                    "the peak must open `pcm` through `_soft_lead`, and every one of the 20 trailing "
                                    "0.007 frames must survive: 31 speech frames, before any ring fallback can help")
            await built._hear("user", "छोटी बेटी मीरा")
            await built._next_question()
        self.assertEqual(len(captured), 1, "the turn really reached a Decision pass")
        self.assertGreaterEqual(captured[0], 31 * 2048,
                                "neither the soft frames before the peak nor the 20 after it may be chopped off: "
                                "the vowel alone is not the word")

    async def regime_c_continuous_fan(self) -> None:
        # Call #36, `continuous_clinic_fan_022`: the fan never stops and nobody speaks for 36 s.
        built, ws, _session = build(call_id=None)
        clock = self.Clock(100.0)
        built.started_at = built.last_voice_ts = 100.0
        fan = self.rms_frame(0.022)
        timeline = self.Timeline()
        with unittest.mock.patch.object(live.time, "time", clock):
            now = 100.0
            while now <= 136.0 + 1e-9:
                clock.now = now
                await self.stream(built, fan, 8)  # 512 ms of steady 64 ms frames per half-second tick
                await built.tick(now)
                timeline.capture(ws, now)
                now = round(now + 0.5, 3)
        self.assertEqual(built.telemetry["user_turns"], 0,
                         "a steady fan is the room, not a caller: 36 s of it is never one endless turn")
        self.assertEqual(built.pcm, b"", "and none of it may be banked as her turn audio")
        self.assertIn("silence_check_in", timeline.kinds(), "the 15 s check-in still owes this call a line")
        check_in = timeline.at("silence_check_in")
        self.assertGreaterEqual(check_in, 114.5, "the check-in fires on the 15 s of real silence, not before")
        self.assertLessEqual(check_in, 116.0, "and not late: the fan was never mistaken for her voice")
        self.assertIs(built.state.phase, Phase.CLOSING, "30 s of a quiet room closes the call by 130 s")
        self.assertIs(built.state.close_reason, CloseReason.SILENCE, "and it says why: silence, not a technical failure")

    async def regime_d_fan_onset(self) -> None:
        # `mid_call_fan_onset_022`: the room is quiet, then somebody switches the clinic fan on mid-call.
        built, ws, _session = build(call_id=None)
        clock = self.Clock(100.0)
        built.started_at = built.last_voice_ts = 100.0
        timeline = self.Timeline()
        with unittest.mock.patch.object(live.time, "time", clock):
            clock.now = 100.0
            await self.stream(built, self.rms_frame(0.002), 6)  # t = 100.0 .. 100.4, nothing said
            clock.now = 100.4
            await built.tick(100.4)
            timeline.capture(ws, 100.4)
            now = 100.9
            while now <= 140.0 + 1e-9:
                clock.now = now
                await self.stream(built, self.rms_frame(0.022), 8)  # the fan, and nothing else, from here on
                await built.tick(now)
                timeline.capture(ws, now)
                now = round(now + 0.5, 3)
        # The onset is allowed to buy a turn for as long as the floor takes to climb over it, but no longer: an
        # onset that never settles banks 39 s of fan, starves every timer below of a quiet second, and closes the
        # call on the wrong reason entirely.
        self.assertLessEqual(len(built.pcm), 33 * 2048,
                             "the floor settles over the fan within ~1 s, so the onset can never bank 39 s of it")
        self.assertLessEqual(built.telemetry["user_turns"], 1,
                             "the onset is at most one caller turn; a fan that re-arms it every frame is a room")
        kinds = timeline.kinds()
        self.assertIn("silence_check_in", kinds,
                      "the floor must climb above the new fan within ~1 s, or the check-in is never owed")
        self.assertLessEqual(timeline.at("silence_check_in"), 117.5,
                             "a floor that settles ~1 s after the onset still checks in by ~116.5 s")
        self.assertIn("auto_close", kinds, "and the 30 s silence close still fires")
        self.assertLessEqual(timeline.at("auto_close"), 140.0,
                             "both the check-in AND the silence close land before the sweep ends")
        self.assertIs(built.state.close_reason, CloseReason.SILENCE, "the call is closed for silence it can prove")

    async def regime_e_speech_over_noise(self) -> None:
        # Calls #29/#31, `speech_over_clinic_noise_022`: 1.28 s of speech buried in a fan at `rms` 0.022.
        built, _ws, _session = build(call_id=None)
        clock = self.Clock(100.0)
        with unittest.mock.patch.object(live.time, "time", clock):
            await self.stream(built, self.rms_frame(0.022), 15)   # the room: 0.96 s of fan before she starts
            await self.stream(built, self.rms_frame(0.085), 20)   # her voice, 1.28 s
            await self.stream(built, self.rms_frame(0.022), 40)   # 2.56 s of fan after she stops
        self.assertGreaterEqual(len(built.pcm), 20 * 2048,
                                "her speech must reach the pass, even though the room around it is loud")
        self.assertLessEqual(len(built.pcm), 33 * 2048,
                             "and the trailing 2 s of room noise is released, not banked: speech plus a bounded "
                             "preroll and tail, never the 60+ frames of fan this regime really sends")

    def test_adaptive_noise_floor_and_contiguous_turn_audio_across_acoustic_regimes(self):
        async def body() -> None:
            with self.subTest(regime="A: soft_speakerphone_006 (Call #30)"):
                await self.regime_a_soft_speakerphone()
            with self.subTest(regime="B: soft_speakerphone_single_peak (Call #30)"):
                await self.regime_b_soft_speakerphone_single_peak()
            with self.subTest(regime="C: continuous_clinic_fan_022 (Call #36)"):
                await self.regime_c_continuous_fan()
            with self.subTest(regime="D: mid_call_fan_onset_022"):
                await self.regime_d_fan_onset()
            with self.subTest(regime="E: speech_over_clinic_noise_022 (Calls #29, #31)"):
                await self.regime_e_speech_over_noise()

        asyncio.run(body())

    def test_server_playback_tail_unmutes_only_in_final_350ms_after_turn_complete(self):
        # Calls #33/#34/#41: she answered "नहीं" 300 ms before her line finished playing. The browser had the whole
        # line and the model had already closed its turn, yet the server still called that frame her echo and threw
        # it away. The tail is not a guess: it is the last 350 ms of a turn the model itself declared complete.
        tail = self.TAIL_MS
        cases = [
            (True, live.TurnPhase.PROMPTING, 200, False),            # mid-generation: `turn_complete` has not arrived
            (False, live.TurnPhase.LISTENING, 2000, False),           # turn closed, but 2 s of her line is still queued
            (False, live.TurnPhase.LISTENING, tail + 150, False),     # 500 ms left is still more than the tail
            (False, live.TurnPhase.LISTENING, tail - 50, True),       # inside the tail: the eager "नहीं" is her voice
            (False, live.TurnPhase.LISTENING, tail - 250, True),      # and so is the last 100 ms of it
        ]

        async def body() -> None:
            for bot_generating, phase, remaining_ms, want in cases:
                with self.subTest(bot_generating=bot_generating, turn_phase=str(phase), remaining_ms=remaining_ms,
                                  want_accepted=want):
                    built, _ws, session = build(call_id=None)
                    clock = self.Clock(100.0)
                    with unittest.mock.patch.object(live.time, "time", clock):
                        built.bot_generating, built.turn_phase = bot_generating, phase
                        built.playback_ends_at = clock.now + remaining_ms / 1000
                        await built.on_audio({"data": pcm(), "sample_rate": 16000})
                    self.assertEqual(len(session.realtime("audio")), 1 if want else 0,
                                     f"{remaining_ms} ms of playback left with bot_generating={bot_generating}: "
                                     f"the mic is {'open' if want else 'shut'} here")
                    self.assertIs(bool(built.pcm), want,
                                  "and the frame is banked for the pass exactly when it was forwarded")

        asyncio.run(body())


if __name__ == "__main__":
    unittest.main()
