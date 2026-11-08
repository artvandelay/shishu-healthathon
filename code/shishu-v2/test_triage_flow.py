"""Checks for the checklist-driven triage flow. No network."""

import asyncio
import os
import tempfile
import time
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-triage-")
os.environ.setdefault("DATA_DIR", _TMP)

import checklist  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import prompts  # noqa: E402
from answer_tool import INTERNAL_NOTE_TAG, build_turn_context  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_prompt_lists_every_topic_and_the_outro(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        for tid in checklist.REQUIRED_IDS:
            self.assertIn(tid, text)
        self.assertIn("UNMISTAKABLY IN HINDI", text)
        self.assertIn("Aapko iske alawa doctor ke liye kuch aur batana hai?", text)
        self.assertIn("Aapki report taiyaar ho rahi hai, kripya is page par bane rahiye", text)
        self.assertIn("Ab tak ki report is page par taiyaar ho rahi hai", text)
        self.assertIn("STYLE EXAMPLES FROM REAL CALLS", text)
        for other in (
            text,
            prompts.OPENAI_LIVE_VOICE_PROMPT,
            prompts.OPENAI_RESPONSES_BACKEND_PROMPT,
        ):
            self.assertNotIn("set_progress", other)

    def test_turn_context_is_marked_internal_and_the_prompt_forbids_speaking_it(self):
        text = build_turn_context(
            {"profile": {}},
            {"topic_id": "birth_term", "followup_id": None, "question_kind": "topic",
             "prompt_note": "Ask it."},
        )
        self.assertTrue(text.startswith(INTERNAL_NOTE_TAG))
        prompt = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn(INTERNAL_NOTE_TAG, prompt)
        self.assertIn("NEVER speak it", prompt)
        self.assertIn("rejected", prompt)
        self.assertIn("silently call", prompt)


class _FakeWs:
    def __init__(self):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _FakeSession:
    def __init__(self):
        self.texts = []

    async def send_realtime_input(self, text=None, **kwargs):
        self.texts.append(text)


def _run_watchdog(state, telemetry, seconds=1.2):
    async def run():
        ws, session = _FakeWs(), _FakeSession()
        stop = asyncio.Event()
        call_id = db.create_call("gemini")
        task = asyncio.create_task(
            gemini_live._watchdog_loop(
                session, asyncio.Lock(), ws, asyncio.Lock(), stop, call_id, telemetry, state, 4.0
            )
        )
        await asyncio.sleep(seconds)
        stop.set()
        await task
        return session, ws, stop

    return asyncio.run(run())


class CallTimerTests(unittest.TestCase):
    def _telemetry(self):
        return gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 650, 300, 0)

    def test_silence_triggers_incomplete_close(self):
        state = gemini_live._TurnState()
        state.quiet_from = time.time() - (checklist.SILENCE_CLOSE_SEC + 1)
        state.checked_in = True
        telemetry = self._telemetry()
        session, _, _ = _run_watchdog(state, telemetry)
        self.assertIn(gemini_live._SILENCE_CLOSE_INSTRUCTION, session.texts)
        self.assertEqual(telemetry["close_reason"], "silence")
        self.assertFalse(telemetry.get("ending_sent"))

    def test_hard_limit_stops_the_call(self):
        state = gemini_live._TurnState()
        state.call_started_ts = time.time() - (checklist.HARD_LIMIT_SEC + 1)
        telemetry = self._telemetry()
        _, _, stop = _run_watchdog(state, telemetry)
        self.assertTrue(stop.is_set())
        self.assertEqual(telemetry["close_reason"], "time_limit")

    def test_emergency_end_is_accepted_without_an_ending_message(self):
        telemetry = self._telemetry()

        class _Fc:
            id = "fc1"
            name = "end_call"
            args = {"reason": "emergency"}
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        async def run():
            sess = _Sess()
            ws = _FakeWs()
            await gemini_live._answer_function_calls(
                sess, asyncio.Lock(), ws, asyncio.Lock(), db.create_call("gemini"),
                telemetry, [_Fc()], set(), set(),
            )
            return sess, ws

        sess, ws = asyncio.run(run())
        self.assertEqual(sess.responses[0].response["result"], "ok")
        self.assertEqual(telemetry["close_reason"], "emergency")
        self.assertFalse(telemetry.get("end_requested"))
        self.assertFalse(any(message.get("type") == "ending" for message in ws.sent))


class EndCallReasonTests(unittest.TestCase):
    def _run_end_call(self, reason):
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 750, 300, 0)
        call_id = db.create_call("gemini")

        class _FcEnd:
            id = "fc_end"
            name = "end_call"
            args = {"reason": reason}
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        async def run():
            sess = _Sess()
            ws = _FakeWs()
            await gemini_live._answer_function_calls(
                sess, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                [_FcEnd()], set(), set(),
            )
            return sess, ws

        return asyncio.run(run()), telemetry

    def test_caller_stop_is_accepted_without_an_ending_message(self):
        (sess, ws), telemetry = self._run_end_call("caller_stop")
        self.assertEqual(sess.responses[0].response["result"], "ok")
        self.assertEqual(telemetry["close_reason"], "caller_stop")
        self.assertFalse(telemetry.get("end_requested"))
        self.assertFalse(any(message.get("type") == "ending" for message in ws.sent))

    def test_complete_end_call_is_rejected(self):
        (sess, ws), telemetry = self._run_end_call("complete")
        self.assertEqual(sess.responses[0].response["result"], "rejected")
        self.assertFalse(any(message.get("type") == "ending" for message in ws.sent))


if __name__ == "__main__":
    unittest.main()
