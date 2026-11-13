"""Stage 2: pause-safe VAD, non-interrupting fallback, and quick replies."""

import asyncio
import json
import os
import tempfile
import time
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-v4-live-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import call_cursor  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import test_v4_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_profile, apply_turn  # noqa: E402


class _FakeSocket:
    def __init__(self, messages=None):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []
        self._messages = list(messages or [])

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive_text(self):
        if not self._messages:
            raise gemini_live.WebSocketDisconnect()
        return json.dumps(self._messages.pop(0), ensure_ascii=False)


class _FakeSession:
    def __init__(self, fail_tool=False):
        self.sent = []
        self.fail_tool = fail_tool

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses))
        if self.fail_tool:
            raise RuntimeError("tool response failed")

    async def send_realtime_input(self, **kwargs):
        self.sent.append(("realtime", kwargs))


class _Call:
    def __init__(self, name, args, fid):
        self.id = fid
        self.name = name
        self.args = args
        self.will_continue = False


class _Recorder:
    def add_bot_pcm(self, pcm, rate):
        return None


def _kinds(session):
    return [item[0] for item in session.sent]


async def _drop_fallback(turn):
    task = turn.speak_fallback
    turn.speak_fallback = None
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


class ChildPhraseTests(unittest.TestCase):
    def _state(self):
        return cursor_tests._profile(children=[
            cursor_tests._child(2, "male", 1),
            cursor_tests._child(8, "female", 2, name="रिया"),
            cursor_tests._child(5, "male", 3, name="अर्जुन"),
        ])

    def test_multi_word_child_refs_match_age_ordinal_and_sex(self):
        state = self._state()
        found = {
            "बीच वाले का": ["child_3"],
            "5 साल वाले": ["child_3"],
            "बड़े वाले को": ["child_2"],
            "middle child": ["child_3"],
            "oldest child": ["child_2"],
            "पाँच साल": ["child_3"],
            "पांच साल": ["child_3"],
            "दो साल": ["child_1"],
            "छोटी बेटी": ["child_2"],
        }
        for phrase, ids in found.items():
            self.assertEqual(call_cursor._resolve_child_ids(state, phrase), ids, phrase)
        half = cursor_tests._profile(children=[
            cursor_tests._child(2.5, "male", 1),
            cursor_tests._child(1.5, "female", 2),
        ])
        self.assertEqual(call_cursor._age_from_ref("ढाई साल वाले"), 2.5)
        self.assertEqual(call_cursor._age_from_ref("डेढ़ साल"), 1.5)
        self.assertEqual(call_cursor._age_from_ref("आधा साल"), 0.5)
        self.assertEqual(call_cursor._resolve_child_ids(half, "ढाई साल वाले"), ["child_1"])
        self.assertEqual(call_cursor._resolve_child_ids(half, "डेढ़ साल"), ["child_2"])


class ProfileAndCloseTests(unittest.TestCase):
    def test_nahi_maps_to_unmarried_unless_she_is_marrying(self):
        self.assertEqual(gemini_live._marital("नहीं"), "unmarried")
        self.assertEqual(gemini_live._marital("नही"), "unmarried")
        self.assertEqual(gemini_live._marital("अभी नहीं"), "unmarried")
        self.assertEqual(gemini_live._marital("अविवाहित"), "unmarried")
        self.assertEqual(gemini_live._marital("कुंवारी"), "unmarried")
        self.assertEqual(gemini_live._marital("no"), "unmarried")
        self.assertEqual(gemini_live._marital("nahin"), "unmarried")
        self.assertEqual(gemini_live._marital(False), "unmarried")
        self.assertEqual(gemini_live._marital("मेरी शादी नहीं हुई है"), "unmarried")
        self.assertEqual(gemini_live._marital("शादी हो गई"), "married")
        self.assertEqual(gemini_live._marital("होने वाली"), "about_to_marry")

    def test_demographic_skip_overwrites_a_misrouted_answer(self):
        baby = cursor_tests._profile(children=[cursor_tests._child(2, "male")])
        baby["answers"]["mod_tooth_bleed"] = call_cursor._blank("concern", note="खून")
        call_cursor._recompute(baby)
        self.assertEqual(baby["answers"]["mod_tooth_bleed"]["state"], "concern")

        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )
        state = apply_turn(state, "concern", note="तीन दिन", followup_answered=True)
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "concern")
        state, hold = apply_profile(state, marital_status="unmarried")
        self.assertIsNone(hold)
        self.assertEqual(state["profile"]["marital_status"], "unmarried")
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "not_applicable")
        self.assertNotEqual(state["active_qid"], "birth_nicu")

    def test_close_keeps_a_concern_note_on_retract_and_on_empty_clear(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        state = apply_turn(state, "concern", note="पेट दर्द")
        state = apply_turn(state, "retract")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["answers"]["close_anything_else"]["state"], "concern")
        self.assertIn("पेट दर्द", state["answers"]["close_anything_else"]["notes"])

        detail = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        detail = apply_turn(detail, "concern", note="नींद कम")
        detail["pending_followup_idx"] = 0
        detail = apply_turn(detail, "clear")
        self.assertEqual(detail["phase"], "closing")
        self.assertEqual(detail["answers"]["close_anything_else"]["state"], "concern")
        self.assertIn("नींद कम", detail["answers"]["close_anything_else"]["notes"])


class VadTests(unittest.TestCase):
    def test_legacy_defaults_move_to_a_longer_low_sensitivity_pause(self):
        self.assertEqual(db.DEFAULT_SETTINGS["vad_silence_ms"], "1000")
        self.assertEqual(db.DEFAULT_SETTINGS["vad_end_sensitivity"], "LOW")
        unset = gemini_live._session_settings({})
        self.assertEqual(unset["silence_duration_ms"], 1000)
        self.assertEqual(
            unset["end_sens"],
            gemini_live.types.EndSensitivity.END_SENSITIVITY_LOW,
        )
        custom = gemini_live._session_settings({
            "vad_silence_ms": "1200",
            "vad_end_sensitivity": "HIGH",
        })
        self.assertEqual(custom["silence_duration_ms"], 1200)
        self.assertEqual(
            custom["end_sens"],
            gemini_live.types.EndSensitivity.END_SENSITIVITY_HIGH,
        )
        db.set_setting("vad_silence_ms", "750")
        db.set_setting("vad_end_sensitivity", "HIGH")
        db._init_db()
        settings = db.get_all_settings()
        self.assertEqual(settings["vad_silence_ms"], "1000")
        self.assertEqual(settings["vad_end_sensitivity"], "LOW")
        db.set_setting("vad_silence_ms", "900")
        db._init_db()
        self.assertEqual(db.get_all_settings()["vad_silence_ms"], "900")


class LiveTurnTests(unittest.TestCase):
    def _telemetry(self):
        return {
            "tool_calls": 0,
            "tool_errors": 0,
            "end_requested": False,
            "audio_chunks_out": 0,
            "bot_turns": 0,
            "turn_latencies_ms": [],
        }

    async def _turn(self, session, cursor, turn, calls, ws=None):
        call_id = db.create_call("gemini")
        await gemini_live._answer_function_calls(
            session,
            asyncio.Lock(),
            ws or _FakeSocket(),
            asyncio.Lock(),
            call_id,
            self._telemetry(),
            calls,
            set(),
            set(),
            cursor,
            turn,
        )
        return call_id

    def test_greeting_emits_question_state(self):
        cursor = gemini_live._Cursor()
        session = _FakeSession()
        ws = _FakeSocket()

        async def run():
            await gemini_live._send_greeting(
                session, asyncio.Lock(), ws, asyncio.Lock(), cursor
            )

        asyncio.run(run())
        self.assertEqual(_kinds(session), ["realtime"])
        self.assertIn("नाम", session.sent[0][1]["text"])
        states = [item for item in ws.sent if item.get("type") == "question_state"]
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["qid"], "intro_name")
        self.assertEqual(states[0]["group"], "intro")
        self.assertFalse(states[0]["expect_yes_no"])
        self.assertEqual(states[0]["step"], "intro_name")

    def test_empty_profile_save_holds_and_a_later_correction_is_applied(self):
        async def run():
            cursor = gemini_live._Cursor()
            turn = gemini_live._TurnState()
            await self._turn(
                _FakeSession(),
                cursor,
                turn,
                [_Call("save_profile", {}, "fc-hold")],
            )
            self.assertEqual(cursor.state["hold_reason"], "name_fragment")
            self.assertIn("hold:name_fragment", gemini_live._current_step(cursor.state))
            await _drop_fallback(turn)

            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            await self._turn(
                _FakeSession(),
                cursor,
                turn,
                [_Call(
                    "save_profile",
                    {"marital_status": "मेरी शादी नहीं हुई है"},
                    "fc-fix",
                )],
            )
            self.assertEqual(cursor.state["profile"]["marital_status"], "unmarried")
            self.assertEqual(cursor.state["profile"]["children"], [])
            self.assertEqual(
                cursor.state["answers"]["birth_nicu"]["state"],
                "not_applicable",
            )
            self.assertNotEqual(cursor.state["active_qid"], "birth_nicu")
            await _drop_fallback(turn)

        asyncio.run(run())

    def test_fallback_speaks_only_when_gemini_stays_silent(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        step = gemini_live._current_step(state)

        async def silent():
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            cursor = gemini_live._Cursor()
            cursor.state = state
            session = _FakeSession()
            ws = _FakeSocket()
            await self._turn(
                session,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear"}, "fc-go")],
                ws,
            )
            self.assertEqual(_kinds(session), ["tool"])
            self.assertNotIn("wait", session.sent[0][1][0].response)
            self.assertFalse(turn.bot_replied_since_user)
            states = [item for item in ws.sent if item.get("type") == "question_state"]
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0]["step"], gemini_live._current_step(cursor.state))
            self.assertNotEqual(states[0]["qid"], "birth_jaundice")
            self.assertTrue(states[0]["expect_yes_no"])
            await turn.speak_fallback
            self.assertEqual(_kinds(session), ["tool", "realtime"])
            return cursor.state["active_qid"]

        async def interrupted():
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            cursor = gemini_live._Cursor()
            cursor.state = dict(state)
            cursor.state["answers"] = dict(state["answers"])
            session = _FakeSession()
            ws = _FakeSocket()
            call_id = await self._turn(
                session,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear"}, "fc-audio")],
                ws,
            )
            self.assertEqual(_kinds(session), ["tool"])
            await gemini_live._on_bot_audio(
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                b"\x00\x00" * 24,
                24000,
                cursor,
            )
            self.assertTrue(turn.bot_replied_since_user)
            self.assertTrue(turn.bot_speaking)
            await turn.speak_fallback
            self.assertEqual(_kinds(session), ["tool"])

        async def failed_tool():
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            cursor = gemini_live._Cursor()
            cursor.state = dict(state)
            cursor.state["answers"] = dict(state["answers"])
            session = _FakeSession(fail_tool=True)
            await self._turn(
                session,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear"}, "fc-fail")],
            )
            self.assertIsNone(turn.speak_fallback)
            self.assertEqual(_kinds(session), ["tool", "realtime"])

        self.assertEqual(asyncio.run(silent()), "birth_sudden")
        asyncio.run(interrupted())
        asyncio.run(failed_tool())

    def test_speech_end_refreshes_quiet_from_after_the_question_was_spoken(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )
        turn = gemini_live._TurnState()
        turn.spoken_qid = gemini_live._current_step(state)
        turn.bot_speaking = False
        turn.playback_ends_at = time.time() - 1
        turn.quiet_from = 0.0
        cursor = gemini_live._Cursor()
        cursor.state = state
        ws = _FakeSocket([{"type": "speech_end"}])

        async def run():
            await gemini_live._browser_loop(
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                asyncio.Event(),
                db.create_call("gemini"),
                self._telemetry(),
                turn,
                cursor,
                _Recorder(),
            )

        asyncio.run(run())
        self.assertGreater(turn.quiet_from, 0)
        self.assertGreaterEqual(turn.quiet_from, turn.playback_ends_at)
        self.assertFalse(turn.awaiting_bot_reply)
        self.assertFalse(turn.speech_end_waiting)

    def test_trailing_clause_within_2_5s_stays_on_the_concern(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )

        async def run():
            turn = gemini_live._TurnState()
            turn.spoken_qid = gemini_live._current_step(state)
            cursor = gemini_live._Cursor()
            cursor.state = state
            session = _FakeSession()
            await self._turn(
                session,
                cursor,
                turn,
                [_Call(
                    "record_turn",
                    {
                        "outcome": "concern",
                        "note": "तीन दिन",
                        "followup_answered": True,
                    },
                    "fc-1",
                )],
            )
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            self.assertEqual(turn.last_advance_qid, "birth_nicu")
            turn.last_user_text = "सांस भी थी"
            await self._turn(
                session,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "concern", "note": ""}, "fc-2")],
            )
            notes = cursor.state["answers"]["birth_nicu"]["notes"]
            self.assertIn("तीन दिन", notes)
            self.assertIn("सांस भी थी", notes)
            turn.last_advance_ts = time.time() - 3
            turn.last_user_text = "बहुत पुराना"
            await self._turn(
                session,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "concern"}, "fc-3")],
            )
            self.assertNotIn("बहुत पुराना", cursor.state["answers"]["birth_nicu"]["notes"])
            await _drop_fallback(turn)

        asyncio.run(run())

    def test_quick_reply_is_idempotent_and_closes_on_the_last_question(self):
        async def clinical():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            session = _FakeSession()
            ws = _FakeSocket([
                {"type": "quick_reply", "outcome": "clear", "step": "stale"},
                {"type": "quick_reply", "outcome": "clear", "step": step},
                {"type": "quick_reply", "outcome": "concern", "step": step},
            ])
            await gemini_live._browser_loop(
                session,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                asyncio.Event(),
                db.create_call("gemini"),
                self._telemetry(),
                turn,
                cursor,
                _Recorder(),
            )
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            self.assertEqual(cursor.state["answers"]["birth_nicu"]["state"], "clear")
            self.assertEqual(_kinds(session), ["realtime"])
            texts = [
                item["text"] for item in ws.sent
                if item.get("type") == "transcript"
            ]
            self.assertEqual(texts, ["नहीं।"])
            details = [
                item["last_event"]["detail"]
                for item in ws.sent
                if item.get("type") == "telemetry"
            ]
            self.assertIn("quick_reply_clear", details)
            states = [item for item in ws.sent if item.get("type") == "question_state"]
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0]["qid"], "birth_jaundice")
            self.assertIn({"type": "clear"}, ws.sent)

        async def close():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "close_anything_else",
            )
            state = apply_turn(state, "concern", note="पेट दर्द")
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            session = _FakeSession()
            ws = _FakeSocket([
                {"type": "quick_reply", "outcome": "clear", "step": step},
                {"type": "quick_reply", "outcome": "clear", "step": step},
            ])
            await gemini_live._browser_loop(
                session,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                asyncio.Event(),
                db.create_call("gemini"),
                self._telemetry(),
                turn,
                cursor,
                _Recorder(),
            )
            self.assertEqual(cursor.state["phase"], "closing")
            self.assertEqual(cursor.state["close_reason"], "complete")
            self.assertEqual(cursor.state["answers"]["close_anything_else"]["state"], "concern")
            self.assertIn("पेट दर्द", cursor.state["answers"]["close_anything_else"]["notes"])
            self.assertEqual(_kinds(session), ["realtime"])
            self.assertIn("धन्यवाद", session.sent[0][1]["text"])

        asyncio.run(clinical())
        asyncio.run(close())


if __name__ == "__main__":
    unittest.main()
