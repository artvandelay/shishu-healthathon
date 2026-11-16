"""Behavioral offline tests for `intake.live`: fake Gemini Live session, fake WebSocket, no network."""

from __future__ import annotations

import asyncio
import json
import unittest
import unittest.mock
from typing import Any

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`
from intake import db, live
from intake.models import CallState, CloseReason, Phase, TurnCommand, TurnOutcome
from intake.pack import load_pack
from starlette.websockets import WebSocketState

PACK = load_pack("shishu-hi")
SILENCE = b"\x00\x00" * 2000


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


class FakeDetector:
    """Silero's stand-in: replays a scripted probability per fed chunk, last one on repeat."""

    def __init__(self, probs: list[float]) -> None:
        self.probs, self.fed = list(probs), 0

    def feed_pcm16(self, pcm: bytes) -> list[float]:
        index = min(self.fed, len(self.probs) - 1)
        self.fed += 1
        return [self.probs[index]]


class FakeRecorder:
    def __init__(self) -> None:
        self.user: list[tuple[bytes, int]] = []
        self.bot: list[tuple[bytes, int]] = []

    def add_user_pcm(self, pcm: bytes, rate: int) -> None:
        self.user.append((pcm, rate))

    def add_bot_pcm(self, pcm: bytes, rate: int) -> None:
        self.bot.append((pcm, rate))

    def save(self) -> tuple[str, int]:
        return "call_1.wav", 3


def build(*, probs: list[float] | None = None, messages: list | None = None,
          fail_realtime: bool = False, call_id: int | None = 1,
          state: CallState | None = None) -> tuple[live.LiveCallSession, FakeWebSocket, FakeSession]:
    """One session wired to fakes, with the detector scripted to the given probabilities."""
    ws, session = FakeWebSocket(), FakeSession(messages, fail_realtime=fail_realtime)
    built = live.LiveCallSession(
        ws=ws, pack=PACK, session=session, call_id=call_id, state=state,
        recorder=FakeRecorder(), on_call_completed=lambda _call_id: True)
    built.detector = FakeDetector(probs or [0.05])
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
            self.assertTrue(built.bot_started)
            self.assertEqual(built.bot_text, self.LINE, "the session now tracks the whole spoken line")
            captions = [p["text"] for p in ws.of("transcript") if p["role"] == "assistant"]
            self.assertEqual(captions, ["क्या आपकी", self.LINE], "the browser gets the complete sentence too")
            self.assertEqual(await self.stored(call_id, "assistant"), [self.LINE], "and SQLite holds one merged turn")
            await built.on_message(self.complete())
            self.assertEqual(await self.stored(call_id, "assistant"), [self.LINE], "a repeated turn_complete never doubles the line")

        asyncio.run(body())

    def test_a_clarify_step_keeps_geminis_own_explanation(self):
        async def body() -> None:
            custom = "मतलब, आपकी शादी हो चुकी है या होने वाली है?"
            built, ws, call_id = await self.session_at("intro_married")
            await built._publish_and_cue("clarify", immediate=False)
            self.assertTrue(built.clarify_step)
            await built.on_message(self.said(custom))
            await built.on_message(self.audio())
            await built.on_message(self.complete())
            captions = [p["text"] for p in ws.of("transcript") if p["role"] == "assistant"]
            self.assertEqual(captions, [custom], "the custom clarification is never overwritten")
            self.assertEqual(await self.stored(call_id, "assistant"), [custom])

        asyncio.run(body())

    def test_a_new_step_and_an_interruption_both_clear_the_tracked_line(self):
        async def body() -> None:
            built, _ws, _call_id = await self.session_at("intro_married")
            await built._publish_and_cue("speak", immediate=False)
            await built.on_message(self.said("अब मैं"))
            self.assertEqual(built.bot_text, "अब मैं")
            await built._publish_and_cue("speak", immediate=False)
            self.assertEqual(built.bot_text, "", "a new step owns a fresh line")
            await built.on_message(self.said("अब मैं"))
            await built.on_message(live.types.LiveServerMessage(server_content=live.types.LiveServerContent(interrupted=True)))
            self.assertEqual(built.bot_text, "", "an interruption drops the half-spoken line")

        asyncio.run(body())


class ConfigTests(unittest.TestCase):
    """The one Live config: manual VAD, one tool, one Hindi voice, minimal thinking."""

    def test_thinking_vad_and_single_tool_are_pinned(self):
        config = live.build_live_config(PACK)
        realtime = config.realtime_input_config
        self.assertTrue(realtime.automatic_activity_detection.disabled)
        self.assertIs(realtime.turn_coverage, live.types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY)
        declared = config.tools[0].function_declarations
        self.assertEqual([f.name for f in declared], ["next_question"])
        self.assertEqual(config.response_modalities, [live.types.Modality.AUDIO])
        self.assertEqual(config.temperature, 0.5)
        self.assertFalse(config.enable_affective_dialog)
        self.assertEqual(config.thinking_config.thinking_level, "MINIMAL")
        self.assertEqual(config.speech_config.voice_config.prebuilt_voice_config.voice_name, "Kore")
        self.assertEqual(config.speech_config.language_code, "hi-IN")
        self.assertIn("SHISHU", config.input_audio_transcription.custom_vocabulary[0])

    def test_resume_handle_is_optional_and_forwarded(self):
        self.assertIsNone(live.build_live_config(PACK).session_resumption.handle)
        self.assertEqual(live.build_live_config(PACK, "abc").session_resumption.handle, "abc")

    def test_output_transcription_is_unpinned_while_input_stays_hindi_with_vocab(self):
        config = live.build_live_config(PACK)
        said = config.output_audio_transcription
        self.assertIsNone(said.language_codes, "pinning the output language flushed 2-word Devanagari fragments")
        self.assertIsNone(said.custom_vocabulary)
        self.assertEqual(config.input_audio_transcription.language_codes, ["hi-IN"], "the caller's audio is still Hindi-only")
        self.assertIn("SHISHU", config.input_audio_transcription.custom_vocabulary[0])
        self.assertEqual(config.speech_config.language_code, "hi-IN", "the spoken voice is still Hindi")

    def test_question_state_payload_matches_call_js(self):
        state = live.engine.new_state(PACK)
        payload = live.question_state_payload(PACK, state)
        self.assertEqual(payload["type"], "question_state")
        self.assertEqual(payload["step"], state.step_token)
        self.assertEqual(payload["qid"], state.cursor.qid)
        self.assertEqual(payload["group"], PACK.by_id[state.cursor.qid].group)
        self.assertFalse(payload["is_yes_no"])
        self.assertEqual(payload["quick_replies"], [])
        self.assertEqual(payload["prompt_chars"],
                         len(live.engine.say_text(PACK, state)))

    def test_tool_body_carries_the_line_and_clarify_flag(self):
        state = live.engine.new_state(PACK)
        self.assertEqual(live.tool_body(PACK, state)["say"],
                         live.engine.say_text(PACK, state))
        clear = live.tool_body(PACK, state, clarify=True)
        self.assertTrue(clear["clarify"])
        self.assertEqual(clear["recorded_concerns"], [])


class TurnTests(unittest.TestCase):
    """One caller turn: onset, the speculative hold, the cached answer, and the commit."""

    def test_speech_start_sends_activity_start_before_any_audio(self):
        async def body() -> None:
            built, _ws, session = build(probs=[0.9])
            await built.on_audio({"data": self.pcm(), "sample_rate": 16000})
            self.assertIsInstance(session.realtime("activity_start")[0],
                                  live.types.ActivityStart)
            self.assertEqual(len(session.realtime("audio")), 1, "the voiced chunk is forwarded")

        asyncio.run(body())

    def test_speech_end_sends_activity_end_and_judges_the_turn_once(self):
        async def body() -> None:
            built, _ws, session = build(probs=[0.9])
            passes: list[int] = []

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                passes.append(1)
                return TurnTests.turn(built.state)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            built.detector = FakeDetector([0.0])  # she has stopped
            for _ in range(40):  # 40 x 32 ms outlasts even the 850 ms reflective wait
                await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            self.assertIsInstance(session.realtime("activity_end")[0], live.types.ActivityEnd)
            await asyncio.sleep(0)
            self.assertEqual(len(passes), 1, "the endpoint starts exactly one pass")

        asyncio.run(body())

    def test_speculation_runs_while_the_mic_is_open(self):
        async def body() -> None:
            built, _ws, _session = build(probs=[0.9])
            seen: list[str] = []

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                seen.append("pass")
                return TurnTests.turn(built.state)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            built.detector = FakeDetector([0.0])
            for _ in range(14):  # 14 x 32 ms = 448 ms of silence, past the 400 ms point
                await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
                await asyncio.sleep(0)
            self.assertTrue(built.vad.in_turn, "the speculative pass does not close the turn")
            self.assertEqual(seen, ["pass"])
            self.assertIsNotNone(built.done, "the answer is banked for the endpoint")

        asyncio.run(body())

    def test_turn_incomplete_holds_the_endpoint_for_the_incomplete_cap(self):
        async def body() -> None:
            built, _ws, _session = build(probs=[0.9])

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                return TurnTests.turn(built.state, TurnCommand(turn_incomplete=True))

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            built.detector = FakeDetector([0.0])
            for _ in range(14):
                await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
                await asyncio.sleep(0)
            self.assertTrue(built.vad.holding_incomplete)
            self.assertEqual(built.vad.target_wait_ms(), 2000.0)

        asyncio.run(body())

    def test_cached_speculation_answers_the_tool_without_a_second_call(self):
        async def body() -> None:
            built, _ws, _session = build(probs=[0.9])
            calls: list[int] = []

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                calls.append(1)
                return TurnTests.turn(built.state, TurnCommand(
                    outcome=TurnOutcome.CLEAR, utterance=TurnTests.utterance()))

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            built.detector = FakeDetector([0.0])
            for _ in range(14):
                await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
                await asyncio.sleep(0)
            before = len(calls)
            body_json, _clarify = await built._next_question()
            self.assertEqual(before, 1, "only the speculative pass ran so far")
            self.assertEqual(len(calls), before, "the banked pass is reused, not repeated")
            self.assertEqual(body_json["result"], "ok")

        asyncio.run(body())

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

    def test_pace_factor_reaches_the_telemetry_on_every_reduced_turn(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.vad.pauses_ms = [170.0] * 8

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                return self.turn(built.state, TurnCommand(
                    outcome=TurnOutcome.CLEAR, utterance=self.utterance()))

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_text({"text": self.utterance()})
            await built._next_question()
            self.assertEqual(built.telemetry["pace_factor"], 0.8)

        asyncio.run(body())
    @staticmethod
    def pcm(bytes_: int = 640) -> str:
        import base64
        return base64.b64encode(SILENCE[:bytes_]).decode()

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


class CallerUtteranceTests(unittest.TestCase):
    """The caller's own words survive onto the note, even when the transcription trails the turn."""

    @staticmethod
    def spoken_turn(state: CallState, spoken: str, outcome: TurnOutcome = TurnOutcome.CONCERN) -> live.Turn:
        """A turn whose Decisions pass saw `spoken` (empty means WAV-only)."""
        return turn_with(state, outcome, spoken)

    def test_speech_end_puts_activity_end_on_the_wire_before_the_endpoint_pass_starts(self):
        async def body() -> None:
            built, _ws, session = build(probs=[0.9])
            sent_before: list[bool] = []
            real = built._start

            def spy(*, speculate: bool = False) -> asyncio.Task:
                sent_before.append(bool(session.realtime("activity_end")))
                return real(speculate=speculate)

            built._start = spy  # type: ignore[method-assign]
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            built.detector = FakeDetector([0.0])
            for _ in range(40):  # 40 x 32 ms outlasts even the 850 ms reflective wait
                await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
                await asyncio.sleep(0)
            self.assertIsInstance(session.realtime("activity_end")[0], live.types.ActivityEnd)
            self.assertEqual(sent_before, [False, True], "speculation may precede it, but the endpoint pass never does")

        asyncio.run(body())

    def test_key_includes_the_transcription_so_late_stt_reruns_the_interpretation(self):
        async def body() -> None:
            built, _ws, _session = build()
            passes: list[str] = []

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm: bytes, **_kwargs: Any) -> live.Turn:
                passes.append(spoken)
                return self.spoken_turn(snapshot, spoken)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            built.snapshot, built.voiced, built.pcm, built.text = built.state.model_copy(deep=True), 640, SILENCE[:640], ""
            await built._start()
            await asyncio.sleep(0)
            self.assertEqual(passes, [""], "the speculative pass ran on WAV alone")
            built.text = "  बुखार   है "
            turn = built._banked()
            self.assertIsNotNone(turn, "the arrived STT is folded in without a second network call")
            self.assertEqual(passes, [""], "no extra Decisions call")
            self.assertEqual(turn.intent.command.utterance, "बुखार है")
            self.assertEqual(turn.spoken, "बुखार है")
            self.assertEqual(built._key()[2], "बुखार है")
            self.assertIs(built._banked(), turn, "the reheard turn is now the cache hit")

        asyncio.run(body())

    def test_a_wav_only_pass_is_not_rerun_when_no_text_ever_arrives(self):
        async def body() -> None:
            built, _ws, _session = build()

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm: bytes, **_kwargs: Any) -> live.Turn:
                return self.spoken_turn(snapshot, spoken)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            built.snapshot, built.voiced, built.pcm = built.state.model_copy(deep=True), 640, SILENCE[:640]
            await built._start()
            await asyncio.sleep(0)
            self.assertEqual(built._banked(), built.done[1], "an exact key hit reuses the banked pass")
            self.assertEqual(built.done[1].spoken, "", "the cache is untouched")

        asyncio.run(body())

    def test_voice_turn_keeps_the_caller_words_on_the_note(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm: bytes, **_kwargs: Any) -> live.Turn:
                return self.spoken_turn(snapshot, spoken)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            built.snapshot, built.voiced, built.pcm, built.text = built.state.model_copy(deep=True), 640, SILENCE[:640], ""
            await built._start()
            await asyncio.sleep(0)
            built.text = "तीन हफ़्ते से डोरी सूखी नहीं"
            await built._next_question()
            rec = built.state.answers["birth_cord_3weeks"]
            self.assertIs(rec.state, live.AnswerStatus.CONCERN)
            self.assertEqual(rec.note.verbatim, "तीन हफ़्ते से डोरी सूखी नहीं", "the note is no longer empty on voice turns")

        asyncio.run(body())

    def test_late_transcription_after_the_reduction_is_appended_to_the_answered_question(self):
        async def body() -> None:
            built, ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm: bytes, **_kwargs: Any) -> live.Turn:
                return self.spoken_turn(snapshot, spoken)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            before = built.state.step_token
            await built.on_text({"text": "डोरी सूखी नहीं"})
            await built._next_question()
            self.assertNotEqual(before, built.state.step_token, "the call moved on")
            self.assertEqual(built.state.answers["birth_cord_3weeks"].note.verbatim, "डोरी सूखी नहीं")
            built.tool_pending = False
            await built._hear("user", "तीन हफ़्ते से")
            rec = built.state.answers["birth_cord_3weeks"]
            self.assertEqual(rec.note.verbatim, "डोरी सूखी नहीं; तीन हफ़्ते से")
            self.assertEqual(built.last_reduced[0], "birth_cord_3weeks", "the late chunk is booked against the answered question")
            self.assertEqual(ws.of("transcript")[-1]["text"], "तीन हफ़्ते से")

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
            built.tool_pending = False
            built.last_reduced = ("birth_jaundice", live.SubStep.FOLLOWUP, 0, False)
            await built._hear("user", "और तेज़ी से बढ़ रहा है")
            fid = PACK.by_id["birth_jaundice"].followups[0]
            rec = built.state.answers["birth_jaundice"]
            self.assertIn(fid, rec.followups)
            self.assertEqual(rec.followups[fid].note.verbatim, "और तेज़ी से बढ़ रहा है")

        asyncio.run(body())

    def test_late_transcription_is_ignored_when_the_answered_question_was_clear(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})

            async def fake_evaluate(snapshot: CallState, spoken: str, pcm: bytes, **_kwargs: Any) -> live.Turn:
                return self.spoken_turn(snapshot, spoken, TurnOutcome.CLEAR)

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            await built._next_question()
            built.tool_pending = False
            before = built.state.answers["birth_cord_3weeks"].note.verbatim
            await built._hear("user", "कुछ और")
            self.assertEqual(built.state.answers["birth_cord_3weeks"].note.verbatim, before)

        asyncio.run(body())

    def test_speech_start_typed_turns_and_taps_clear_the_late_window(self):
        async def body() -> None:
            built, _ws, _session = build(probs=[0.9])
            built.last_reduced = ("birth_jaundice", live.SubStep.MAIN, 0, False)
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
            self.assertIsNone(built.last_reduced, "a new turn closes the window")
            built.last_reduced = ("birth_jaundice", live.SubStep.MAIN, 0, False)
            await built.on_text({"text": "नहीं"})
            self.assertIsNone(built.last_reduced)
            built.last_reduced = ("birth_jaundice", live.SubStep.MAIN, 0, False)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "intro_bridge"})
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            self.assertIsNone(built.last_reduced)

        asyncio.run(body())


class TurnMetaTests(unittest.TestCase):
    """Every reduced turn leaves one compact meta row behind, so a finished call can be re-read cold."""

    @staticmethod
    def fake_pass(outcome: TurnOutcome = TurnOutcome.CONCERN):
        """A Decisions pass that behaves like the real one: it saw exactly `spoken`, and knows whether it was speculative."""
        async def evaluate(snapshot: CallState, spoken: str, pcm: bytes, *, speculated: bool = False) -> live.Turn:
            return turn_with(snapshot, outcome, spoken)._replace(eval_text=" ".join(spoken.split()), speculated=speculated, pcm_len=len(pcm))
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
            for key in ("step_in", "step_out", "advanced", "outcome", "top2", "decision_ms", "audio_ms"):
                self.assertIn(key, meta, f"{key} must be on the turn's meta")
            self.assertEqual(meta["outcome"], "concern")
            self.assertEqual(meta["top2"], "concern:0.97,clarify:0.006")
            self.assertNotEqual(meta["step_in"], meta["step_out"], "the turn moved the step")
            self.assertIs(meta["advanced"], True)
            self.assertEqual(meta["decision_ms"], 200)
            self.assertEqual(meta["audio_ms"], 0, "a typed turn carries no audio")
            self.assertEqual(meta["stt_eval"], "डोरी सूखी नहीं")
            self.assertNotIn("stt_final", meta, "no divergence between what was judged and what was finally heard")
            self.assertNotIn("reheard", meta, "an unhurried turn is not marked reheard")
            self.assertNotIn("speculated", meta)

        asyncio.run(body())

    def test_late_stt_marks_the_turn_reheard_and_keeps_both_transcripts(self):
        async def body() -> None:
            built, _ws, _session = build(call_id=None)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "birth_cord_3weeks"})
            built._evaluate_turn_task = self.fake_pass()  # type: ignore[method-assign]
            built.snapshot, built.voiced, built.pcm, built.text = built.state.model_copy(deep=True), 640, SILENCE[:640], "बुखार"
            await built._start(speculate=True)
            await asyncio.sleep(0)
            built.text = "बुखार तीन दिन से"
            await built._next_question()
            meta = built.last_turn_meta
            self.assertIs(meta["reheard"], True, "the pass was re-interpreted once the STT caught up")
            self.assertIs(meta["speculated"], True, "it began as a speculative pass")
            self.assertEqual(meta["stt_eval"], "बुखार", "what the Decisions call itself saw")
            self.assertEqual(meta["stt_final"], "बुखार तीन दिन से", "what the caller was finally heard to say")
            self.assertEqual(meta["audio_ms"], 20, "640 bytes of 16 kHz int16 mono is 20 ms")

        asyncio.run(body())

    def test_a_failed_pass_still_leaves_its_reason_and_attempts(self):
        async def body() -> None:
            built, _ws, _session = build(call_id=None)

            async def failing(snapshot: CallState, spoken: str, pcm: bytes, *, speculated: bool = False) -> live.Turn:
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

            async def from_the_past(snapshot: CallState, spoken: str, pcm: bytes, *, speculated: bool = False) -> live.Turn:
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


class BoundaryEventTests(unittest.TestCase):
    """Barge-ins and late STT are the two invisible boundaries a dumpster dive has to see."""

    def test_an_interruption_emits_a_barge_in_event_with_the_live_qid(self):
        async def body() -> None:
            built, ws, _session = build(call_id=None)
            qid = built.state.cursor.qid
            await built.on_message(live.types.LiveServerMessage(
                server_content=live.types.LiveServerContent(interrupted=True)))
            evt = ws.of("telemetry")[-1]["last_event"]
            self.assertEqual(evt["type"], "barge_in")
            self.assertEqual(evt["detail"], qid, "the question being answered when the caller cut in")
            self.assertEqual(built.telemetry["interruptions"], 1)

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
                    built.tool_pending = False
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
            state = built.state.model_copy(deep=True)
            state.cursor = state.cursor.model_copy(update={"qid": "intro_bridge"})
            built.state = live.engine.new_state(PACK)
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "intro_bridge"})
            await built.on_quick_reply({"value": "clear", "step": "stale_step"})
            self.assertEqual(session.kinds(), [], "a stale step is ignored")
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            self.assertEqual(session.kinds(), ["text"], "the cue is the only send")
            self.assertTrue(built.state.answers, "the tap answered the step")

        asyncio.run(body())

    def test_quick_reply_accepts_the_legacy_outcome_key(self):
        async def body() -> None:
            built, _ws, _session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "intro_bridge"})
            await built.on_quick_reply({"outcome": "clear", "step": built.state.step_token})
            self.assertTrue(built.state.answers)
            self.assertNotEqual(built.state.step_token, "intro_bridge")

        asyncio.run(body())

    def test_quick_reply_emits_clear_then_question_state_before_the_cue(self):
        async def body() -> None:
            built, ws, session = build()
            built.state.cursor = built.state.cursor.model_copy(update={"qid": "intro_bridge"})
            await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
            sent = [payload["type"] for payload in ws.sent]
            self.assertLess(sent.index("clear"), sent.index("question_state"))
            self.assertLess(sent.index("question_state"), len(ws.sent))
            self.assertEqual(len(session.realtime("text")), 1)

        asyncio.run(body())


class TextAndToolTests(unittest.TestCase):
    """Typed turns, tool answers, the missed-tool fallback, and the silent-model cue."""

    def test_text_turn_is_spoken_as_a_cue_and_can_be_judged(self):
        async def body() -> None:
            built, ws, session = build()
            await built.on_text({"text": "  मेरा   नाम मीरा है "})
            self.assertEqual(session.realtime("text"), ["मेरा नाम मीरा है"])
            self.assertEqual(ws.of("transcript")[0]["text"], "मेरा नाम मीरा है")
            self.assertTrue(built.tool_pending)

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
            built.tool_pending = True

            async def fake_evaluate(*_args: Any, **_kwargs: Any) -> live.Turn:
                return TurnTests.turn(built.state, TurnCommand(
                    outcome=TurnOutcome.CLEAR, utterance="नहीं"))

            built._evaluate_turn_task = fake_evaluate  # type: ignore[method-assign]
            await built.on_text({"text": "नहीं"})
            await built.on_turn_complete()
            self.assertTrue(session.realtime("text"), "the fallback cue went out")
            self.assertEqual(ws.of("bot_turn_complete")[-1]["step"], built.state.step_token)

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
            built, _ws, session = build(probs=[0.9])
            built.state.phase = Phase.CLOSING
            await built.on_audio({"data": TurnTests.pcm(), "sample_rate": 16000})
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


if __name__ == "__main__":
    unittest.main()
