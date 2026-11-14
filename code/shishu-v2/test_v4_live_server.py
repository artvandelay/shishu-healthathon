"""Stage 2: pause-safe VAD, non-interrupting fallback, and quick replies."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

_TMP = tempfile.mkdtemp(prefix="shishu-v4-live-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import call_cursor  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import prompts  # noqa: E402
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
    def __init__(self, fail_tool=False, fail_realtime=False):
        self.sent = []
        self.fail_tool = fail_tool
        self.fail_realtime = fail_realtime

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses))
        if self.fail_tool:
            raise RuntimeError("tool response failed")

    async def send_realtime_input(self, **kwargs):
        self.sent.append(("realtime", kwargs))
        if self.fail_realtime:
            raise RuntimeError("realtime send failed")


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


def _server_message(*, turn_complete=False, user_text="", bot_text="", tool_calls=None, go_away=False):
    tool_call = None
    if tool_calls:
        tool_call = SimpleNamespace(function_calls=tool_calls)
    return SimpleNamespace(
        tool_call_cancellation=None,
        server_content=SimpleNamespace(
            interrupted=False,
            input_transcription=SimpleNamespace(text=user_text) if user_text else None,
            output_transcription=SimpleNamespace(text=bot_text) if bot_text else None,
            model_turn=None,
            turn_complete=turn_complete,
        ),
        tool_call=tool_call,
        session_resumption_update=None,
        go_away=SimpleNamespace(time_left="50s") if go_away else None,
    )


class RosterTests(unittest.TestCase):
    def test_tool_body_sends_exact_child_ids_for_two_or_more_children(self):
        one = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        self.assertNotIn("children", gemini_live._tool_body(one))
        two = cursor_tests._profile(children=[
            cursor_tests._child(2, "male", 1),
            cursor_tests._child(8, "female", 2, name="रिया"),
        ])
        body = gemini_live._tool_body(two)
        self.assertEqual(body["children"], call_cursor.children_roster(two))
        self.assertEqual([row["id"] for row in body["children"]], ["child_2", "child_1"])
        self.assertEqual([row["rank"] for row in body["children"]], ["oldest", "youngest"])
        desc = gemini_live._RECORD_TURN_DECL["parameters"]["properties"]["child_ref"]["description"]
        self.assertIn("child_1", desc)
        self.assertIn("child_n", desc)
        self.assertIn("all", desc)
        prompt = gemini_live.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("child_1", prompt)
        self.assertIn("child_n", prompt)
        self.assertIn("all", prompt)
        self.assertNotIn("5 साल", prompt)
        self.assertFalse(hasattr(gemini_live, "_TRAILING_NOTE_SEC"))
        self.assertFalse(hasattr(gemini_live, "_keep_trailing_note"))
        self.assertEqual(gemini_live._TOOL_SPEAK_FALLBACK_SEC, 2.5)


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
            "user_turns": 0,
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
        self.assertGreater(states[0]["prompt_chars"], 0)
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
            self.assertGreater(states[0]["prompt_chars"], 0)
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

    def test_silent_turn_complete_speaks_the_next_line_immediately(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )

        async def run():
            turn = gemini_live._TurnState()
            turn.spoken_qid = gemini_live._current_step(state)
            turn.user_turn_step = turn.spoken_qid
            cursor = gemini_live._Cursor()
            cursor.state = state
            session = _FakeSession()
            ws = _FakeSocket()
            call_id = db.create_call("gemini")
            await gemini_live._handle_server_message(
                _server_message(
                    turn_complete=True,
                    tool_calls=[_Call("record_turn", {"outcome": "clear"}, "fc-go")],
                ),
                session,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            await asyncio.sleep(0)
            self.assertEqual(_kinds(session), ["tool", "realtime"])
            self.assertIn("speak only this Hindi now", session.sent[1][1]["text"])
            self.assertIn("इमरजेंसी", session.sent[1][1]["text"])
            self.assertEqual(turn.pending_speak_step, "")
            self.assertIsNone(turn.speak_fallback)
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")
            self.assertFalse(turn.bot_replied_since_user)

            broken_turn = gemini_live._TurnState()
            broken_turn.pending_speak_step = gemini_live._current_step(cursor.state)
            broken_turn.pending_speak_kind = "speak"
            broken_ws = _FakeSocket()
            broken_telemetry = self._telemetry()
            stop = asyncio.Event()
            keep_going = await gemini_live._handle_server_message(
                _server_message(turn_complete=True),
                _FakeSession(fail_realtime=True),
                asyncio.Lock(),
                broken_ws,
                asyncio.Lock(),
                call_id,
                broken_telemetry,
                broken_turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=stop,
            )
            self.assertFalse(keep_going)
            self.assertTrue(stop.is_set())
            self.assertEqual(broken_telemetry["tool_errors"], 1)
            errors = [item["message"] for item in broken_ws.sent if item.get("type") == "error"]
            self.assertEqual(errors, ["Could not prompt Gemini to speak"])

        asyncio.run(run())

    def test_quick_reply_before_the_question_is_spoken_is_ignored(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            session = _FakeSession()
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
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
            self.assertEqual(cursor.state["active_qid"], "birth_nicu")
            self.assertNotIn("birth_nicu", cursor.state["answers"])
            self.assertEqual(_kinds(session), [])
            self.assertFalse(any(item.get("type") == "clear" for item in ws.sent))

        asyncio.run(run())

    def test_quick_reply_drops_audio_that_was_already_in_progress(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.bot_speaking = True
            turn.playback_ends_at = time.time() + 5
            turn.spoken_qid = step
            turn.pending_speak_step = step
            turn.user_turn_step = step
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
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
            self.assertTrue(turn.ignore_bot_audio)
            self.assertFalse(turn.bot_speaking)
            self.assertEqual(turn.playback_ends_at, 0.0)
            self.assertEqual(turn.user_turn_step, "")
            self.assertEqual(turn.pending_speak_step, "")
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            sent = list(ws.sent)
            await gemini_live._on_bot_audio(
                ws,
                asyncio.Lock(),
                db.create_call("gemini"),
                self._telemetry(),
                turn,
                _Recorder(),
                b"\x00\x00" * 24,
                24000,
                cursor,
            )
            self.assertEqual(ws.sent, sent)
            self.assertEqual(turn.spoken_qid, step)
            self.assertFalse(turn.bot_replied_since_user)

        asyncio.run(run())

    def test_quick_reply_drops_stale_bot_captions_until_turn_complete(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.bot_speaking = True
            turn.playback_ends_at = time.time() + 5
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
            call_id = db.create_call("gemini")
            await gemini_live._browser_loop(
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                asyncio.Event(),
                call_id,
                self._telemetry(),
                turn,
                cursor,
                _Recorder(),
            )
            self.assertTrue(turn.ignore_bot_audio)
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")

            await gemini_live._handle_server_message(
                _server_message(bot_text="पुराना सवाल जो बीच में कटा"),
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            assistant_ws = [
                item for item in ws.sent
                if item.get("type") == "transcript" and item.get("role") == "assistant"
            ]
            self.assertEqual(assistant_ws, [])
            stored = (db.get_call(call_id) or {})["transcript"]
            self.assertEqual([row for row in stored if row["role"] == "assistant"], [])

            await gemini_live._handle_server_message(
                _server_message(turn_complete=True),
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            self.assertFalse(turn.ignore_bot_audio)

            await gemini_live._handle_server_message(
                _server_message(bot_text="क्या जन्म के बाद पीलिया के लिए नीली लाइट में रखना पड़ा था?"),
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            assistant_ws = [
                item for item in ws.sent
                if item.get("type") == "transcript" and item.get("role") == "assistant"
            ]
            self.assertEqual(len(assistant_ws), 1)
            self.assertIn("नीली लाइट", assistant_ws[0]["text"])
            stored = (db.get_call(call_id) or {})["transcript"]
            assistant_db = [row for row in stored if row["role"] == "assistant"]
            self.assertEqual(len(assistant_db), 1)
            self.assertEqual(assistant_db[0]["qid"], "birth_jaundice")

        asyncio.run(run())

    def test_spoken_and_tapped_no_is_held_as_stale_turn(self):
        async def run():
            kids = [cursor_tests._child(6, "male", 1), cursor_tests._child(2, "female", 2)]
            state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "birth_nicu")
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.user_turn_step = step
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
            call_id = db.create_call("gemini")
            await gemini_live._browser_loop(
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                asyncio.Event(),
                call_id,
                self._telemetry(),
                turn,
                cursor,
                _Recorder(),
            )
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            self.assertEqual(cursor.state["answers"]["birth_nicu"]["state"], "clear")
            self.assertEqual(turn.user_turn_step, "")
            self.assertEqual(turn.last_quick_reply_step, step)
            self.assertGreater(turn.last_quick_reply_ts, 0.0)
            late = _FakeSession()
            await self._turn(
                late,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear", "note": "नहीं"}, "fc-late")],
                ws,
            )
            body = late.sent[0][1][0].response
            self.assertEqual(body["result"], "held")
            self.assertEqual(body["reason"], "stale_turn")
            self.assertEqual(body["children"], call_cursor.children_roster(cursor.state))
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            await _drop_fallback(turn)

            # Delayed ASR from Q1 before Q2's audio is heard: still pinned to Q1.
            await gemini_live._handle_server_message(
                _server_message(user_text="नहीं"),
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            self.assertEqual(turn.user_turn_step, step)
            self.assertEqual(turn.last_quick_reply_step, step)
            before_spoken = _FakeSession()
            await self._turn(
                before_spoken,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear"}, "fc-asr-early")],
            )
            early_body = before_spoken.sent[0][1][0].response
            self.assertEqual(early_body["result"], "held")
            self.assertEqual(early_body["reason"], "stale_turn")
            self.assertEqual(early_body["children"], call_cursor.children_roster(cursor.state))
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            await _drop_fallback(turn)

            # Even after Q2's first audio stamps spoken_qid, ASR within <1.2s stays on Q1.
            next_step = gemini_live._current_step(cursor.state)
            turn.spoken_qid = next_step
            turn.ignore_bot_audio = False
            turn.last_quick_reply_ts = time.time() - 0.5
            await gemini_live._handle_server_message(
                _server_message(user_text="नहीं"),
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            self.assertEqual(turn.user_turn_step, step)
            self.assertEqual(turn.last_quick_reply_step, step)
            within_window = _FakeSession()
            await self._turn(
                within_window,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear"}, "fc-asr-window")],
            )
            window_body = within_window.sent[0][1][0].response
            self.assertEqual(window_body["result"], "held")
            self.assertEqual(window_body["reason"], "stale_turn")
            self.assertEqual(window_body["children"], call_cursor.children_roster(cursor.state))
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            await _drop_fallback(turn)

            # Fresh answer >= 1.2s after tap on spoken Q2 advances Q2 -> Q3 cleanly.
            turn.last_quick_reply_ts = time.time() - 1.5
            fresh = _FakeSession()
            await gemini_live._handle_server_message(
                _server_message(user_text="नहीं"),
                fresh,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            self.assertEqual(turn.last_quick_reply_step, "")
            self.assertEqual(turn.user_turn_step, next_step)
            await self._turn(
                fresh,
                cursor,
                turn,
                [_Call("record_turn", {"outcome": "clear"}, "fc-next")],
            )
            self.assertEqual(cursor.state["answers"]["birth_jaundice"]["state"], "clear")
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")
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
            turn.spoken_qid = step
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
            self.assertGreater(states[0]["prompt_chars"], 0)
            self.assertTrue(states[0]["expect_yes_no"])
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
            turn.spoken_qid = step
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


class Stage5ATests(unittest.TestCase):
    def _telemetry(self):
        return {
            "tool_calls": 0,
            "tool_errors": 0,
            "end_requested": False,
            "audio_chunks_out": 0,
            "bot_turns": 0,
            "user_turns": 0,
            "turn_latencies_ms": [],
            "errors": [],
            "reconnects": 0,
        }

    def test_output_transcription_is_hindi_and_closing_text_is_not_invented(self):
        live = gemini_live._live_config({"gemini_model": "gemini-3.1-flash-live-preview"})
        self.assertEqual(list(live.output_audio_transcription.language_codes), ["hi-IN"])
        self.assertEqual(list(live.input_audio_transcription.language_codes), ["hi-IN"])
        older = gemini_live._live_config({"gemini_model": "gemini-2.0-flash-live"})
        self.assertFalse(older.output_audio_transcription.language_codes)
        self.assertFalse(older.input_audio_transcription.language_codes)
        with open(gemini_live.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("def _complete_closing_transcript", source)
        self.assertFalse(hasattr(gemini_live, "_complete_closing_transcript"))
        prompt = gemini_live.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("थैलेसीमिया माइनर या ट्रेट", prompt)
        self.assertIn("recorded_concerns", prompt)
        self.assertIn("none were recorded yet", prompt)

    def test_same_step_clarify_and_held_reemit_question_state(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.user_turn_step = step
            ws = _FakeSocket()
            session = _FakeSession()
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                db.create_call("gemini"),
                self._telemetry(),
                [_Call("record_turn", {"outcome": "clarify"}, "fc-clarify")],
                set(),
                set(),
                cursor,
                turn,
            )
            states = [item for item in ws.sent if item.get("type") == "question_state"]
            self.assertEqual([item["step"] for item in states], [step])
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            self.assertTrue(session.sent[0][1][0].response["clarify"])
            await _drop_fallback(turn)

            held = gemini_live._TurnState()
            held.user_turn_step = step
            held_ws = _FakeSocket()
            held_session = _FakeSession()
            await gemini_live._answer_function_calls(
                held_session,
                asyncio.Lock(),
                held_ws,
                asyncio.Lock(),
                db.create_call("gemini"),
                self._telemetry(),
                [_Call("record_turn", {"outcome": "clear"}, "fc-held")],
                set(),
                set(),
                cursor,
                held,
            )
            held_states = [item for item in held_ws.sent if item.get("type") == "question_state"]
            self.assertEqual([item["step"] for item in held_states], [step])
            body = held_session.sent[0][1][0].response
            self.assertEqual(body["result"], "held")
            self.assertEqual(body["reason"], "not_spoken")
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            await _drop_fallback(held)

        asyncio.run(run())

    def test_trailing_quick_reply_asr_does_not_record_a_3ms_latency(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            tapped = "birth_nicu"
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.awaiting_bot_reply = True
            turn.speech_end_waiting = False
            turn.last_quick_reply_step = tapped
            turn.last_quick_reply_ts = time.time()
            started = time.time() - 0.35
            turn.last_user_turn_ts = started
            telemetry = self._telemetry()
            telemetry["user_turns"] = 2
            telemetry["turn_latencies_ms"] = [640]
            call_id = db.create_call("gemini")
            ws = _FakeSocket()
            await gemini_live._handle_server_message(
                _server_message(user_text="नहीं"),
                _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                telemetry,
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            self.assertEqual(turn.user_turn_step, tapped)
            self.assertEqual(telemetry["user_turns"], 2)
            self.assertEqual(telemetry["turn_latencies_ms"], [640])
            self.assertAlmostEqual(turn.last_user_turn_ts, started, delta=0.02)
            await gemini_live._on_bot_audio(
                ws,
                asyncio.Lock(),
                call_id,
                telemetry,
                turn,
                _Recorder(),
                b"\x01\x00" * 240,
                24000,
                cursor,
            )
            self.assertEqual(telemetry["user_turns"], 2)
            self.assertEqual(len(telemetry["turn_latencies_ms"]), 2)
            self.assertGreater(telemetry["turn_latencies_ms"][-1], 200)

            fresh = gemini_live._TurnState()
            fresh.spoken_qid = step
            fresh.awaiting_bot_reply = False
            fresh.last_quick_reply_step = tapped
            fresh.last_quick_reply_ts = time.time()
            quiet = self._telemetry()
            quiet["user_turns"] = 2
            quiet["turn_latencies_ms"] = [640]
            await gemini_live._handle_server_message(
                _server_message(user_text="नहीं"),
                _FakeSession(),
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                quiet,
                fresh,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            self.assertEqual(quiet["user_turns"], 2)
            self.assertFalse(fresh.awaiting_bot_reply)
            await gemini_live._on_bot_audio(
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                quiet,
                fresh,
                _Recorder(),
                b"\x01\x00" * 240,
                24000,
                cursor,
            )
            self.assertEqual(quiet["user_turns"], 2)
            self.assertEqual(quiet["turn_latencies_ms"], [640])

        asyncio.run(run())

    def test_vad_start_sensitivity_defaults_and_migrates_to_low(self):
        low = gemini_live.types.StartSensitivity.START_SENSITIVITY_LOW
        high = gemini_live.types.StartSensitivity.START_SENSITIVITY_HIGH
        self.assertEqual(db.DEFAULT_SETTINGS["vad_start_sensitivity"], "LOW")
        self.assertEqual(gemini_live._session_settings({})["start_sens"], low)
        self.assertEqual(
            gemini_live._session_settings({"vad_start_sensitivity": ""})["start_sens"],
            low,
        )
        self.assertEqual(
            gemini_live._session_settings({"vad_start_sensitivity": "LOW"})["start_sens"],
            low,
        )
        self.assertEqual(
            gemini_live._session_settings({"vad_start_sensitivity": "HIGH"})["start_sens"],
            high,
        )
        db.set_setting("vad_start_sensitivity", "HIGH")
        db._init_db()
        self.assertEqual(db.get_all_settings()["vad_start_sensitivity"], "LOW")
        db.set_setting("vad_start_sensitivity", "LOW")
        db._init_db()
        self.assertEqual(db.get_all_settings()["vad_start_sensitivity"], "LOW")

    def test_go_away_waits_and_a_dropped_turn_is_spoken_again(self):
        async def handle(turn, message, session=None):
            cursor = gemini_live._Cursor()
            cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            ws = _FakeSocket()
            keep = await gemini_live._handle_server_message(
                message,
                session or _FakeSession(),
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                db.create_call("gemini"),
                self._telemetry(),
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            return keep, cursor, ws

        async def drive(session, turn, cursor, reconnect):
            stop = getattr(session, "stop", None) or asyncio.Event()
            ws = _FakeSocket()
            telemetry = self._telemetry()
            await asyncio.wait_for(
                gemini_live._gemini_loop(
                    session,
                    asyncio.Lock(),
                    ws,
                    asyncio.Lock(),
                    stop,
                    db.create_call("gemini"),
                    telemetry,
                    turn,
                    _Recorder(),
                    cursor,
                    reconnect,
                    None,
                ),
                timeout=3,
            )
            return ws

        async def run():
            missing = gemini_live._TurnState()
            missing.bot_speaking = True
            keep, _cursor, _ws = await handle(missing, _server_message(go_away=True))
            self.assertFalse(keep)
            self.assertFalse(missing.reconnect_requested)

            speaking = gemini_live._TurnState()
            speaking.bot_speaking = True
            speaking.resume_handle = "resume-1"
            session = _FakeSession()
            keep, cursor, _ws = await handle(speaking, _server_message(go_away=True), session)
            self.assertTrue(keep)
            self.assertTrue(speaking.go_away_pending)
            self.assertFalse(speaking.reconnect_requested)
            self.assertEqual(session.sent, [])
            step = gemini_live._current_step(cursor.state)
            speaking.pending_speak_step = step
            speaking.pending_speak_kind = "clarify"
            speaking.active_speak_kind = "clarify"
            speaking.bot_replied_since_user = False
            keep, _cursor, _ws = await handle(
                speaking,
                _server_message(turn_complete=True),
                session,
            )
            self.assertTrue(keep)
            self.assertFalse(speaking.go_away_pending)
            self.assertTrue(speaking.reconnect_requested)
            self.assertEqual(speaking.pending_speak_step, step)
            self.assertEqual(session.sent, [])

            idle = gemini_live._TurnState()
            idle.resume_handle = "resume-idle"
            keep, _cursor, _ws = await handle(idle, _server_message(go_away=True))
            self.assertTrue(keep)
            self.assertTrue(idle.reconnect_requested)
            self.assertFalse(idle.go_away_pending)

            class _Live:
                def __init__(self, messages, error=None):
                    self.messages = list(messages)
                    self.error = error
                    self.stop = asyncio.Event()
                    self.sent = []
                    self.delivered = 0
                    self.started = False

                async def send_realtime_input(self, **kwargs):
                    self.sent.append(kwargs)
                    self.stop.set()

                async def receive(self):
                    if self.started:
                        await asyncio.Future()
                        return
                    self.started = True
                    if self.error is not None:
                        raise self.error
                    for item in self.messages:
                        self.delivered += 1
                        yield item

            waiting = gemini_live._Cursor()
            waiting.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            wait_step = gemini_live._current_step(waiting.state)
            wait_turn = gemini_live._TurnState()
            wait_turn.resume_handle = "resume-wait"
            wait_turn.awaiting_bot_reply = True
            wait_turn.pending_speak_step = wait_step
            wait_turn.pending_speak_kind = "clarify"
            wait_turn.active_speak_kind = "clarify"
            live = _Live([
                _server_message(go_away=True),
                _server_message(turn_complete=True),
            ])
            seen = []

            async def reconnect(handle):
                seen.append(handle)
                return True

            ws = await drive(live, wait_turn, waiting, reconnect)
            self.assertEqual(live.delivered, 2)
            self.assertEqual(seen, ["resume-wait"])
            self.assertEqual(len(live.sent), 1)
            cue = live.sent[0]["text"]
            self.assertIn("she did not understand", cue)
            self.assertIn("नीली लाइट", cue)
            kinds = [item.get("type") for item in ws.sent]
            self.assertIn("reconnected", kinds)
            self.assertIn("question_state", kinds)

            dropped = gemini_live._Cursor()
            dropped.state = waiting.state
            drop_turn = gemini_live._TurnState()
            drop_turn.resume_handle = "resume-drop"
            drop_turn.bot_speaking = True
            drop_turn.active_speak_kind = "speak"
            drop = _Live([], error=ConnectionError("dropped mid-turn"))
            seen.clear()

            async def reconnect_drop(handle):
                seen.append(handle)
                return True

            ws = await drive(drop, drop_turn, dropped, reconnect_drop)
            self.assertEqual(seen, ["resume-drop"])
            self.assertEqual(len(drop.sent), 1)
            again = drop.sent[0]["text"]
            self.assertIn("speak only this Hindi now", again)
            self.assertIn("नीली लाइट", again)
            self.assertNotIn("she did not understand", again)
            self.assertIn("question_state", [item.get("type") for item in ws.sent])

        asyncio.run(run())

    def test_recorded_concerns_follow_the_cursor_on_clarify_and_family(self):
        async def answer(cursor, turn, call):
            session = _FakeSession()
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                db.create_call("gemini"),
                self._telemetry(),
                [call],
                set(),
                set(),
                cursor,
                turn,
            )
            await _drop_fallback(turn)
            return session.sent[0][1][0].response

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            state["answers"]["birth_nicu"] = {
                "state": "concern",
                "child_ids": ["child_1"],
                "notes": "साँस की तकलीफ",
                "followups": {"nicu_days": {"state": "concern", "note": "चार दिन"}},
            }
            self.assertNotIn("recorded_concerns", gemini_live._tool_body(state))
            lines = gemini_live._recorded_concerns(state)
            self.assertEqual(len(lines), 1)
            self.assertIn("काँच के बक्से", lines[0])
            self.assertIn("4 साल", lines[0])
            self.assertIn("साँस की तकलीफ", lines[0])
            self.assertIn("चार दिन", lines[0])
            self.assertNotIn("ऑपरेशन", lines[0])

            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.user_turn_step = step
            body = await answer(
                cursor,
                turn,
                _Call("record_turn", {"outcome": "clarify"}, "fc-clarify"),
            )
            self.assertTrue(body["clarify"])
            self.assertEqual(body["recorded_concerns"], lines)
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")

            family = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "fam_transfusion",
            )
            family = apply_turn(family, "concern", note="मामा के बेटे को बार बार खून")
            self.assertEqual(family["pending_followup_idx"], 0)
            family = apply_turn(family, "concern", note="सगे मामा")
            family = cursor_tests._clear_until(family, "fam_same_problem")
            preview = gemini_live._tool_body(family)
            self.assertTrue(any("सगे मामा" in line for line in preview["recorded_concerns"]))
            self.assertTrue(any("मामा के बेटे" in line for line in preview["recorded_concerns"]))
            close_state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "close_anything_else",
            )
            self.assertEqual(gemini_live._tool_body(close_state)["recorded_concerns"], [])

            fam = gemini_live._Cursor()
            fam.state = family
            fam_step = gemini_live._current_step(family)
            fam_turn = gemini_live._TurnState()
            fam_turn.spoken_qid = fam_step
            fam_turn.user_turn_step = fam_step
            fam_body = await answer(
                fam,
                fam_turn,
                _Call("record_turn", {"outcome": "concern", "note": ""}, "fc-same"),
            )
            self.assertIn("recorded_concerns", fam_body)
            self.assertTrue(any("सगे मामा" in line for line in fam_body["recorded_concerns"]))
            self.assertEqual(fam.state["active_qid"], "fam_same_problem")

        asyncio.run(run())


class ClosedWorldTargetTests(unittest.TestCase):
    def _telemetry(self):
        return {"tool_calls": 0, "tool_errors": 0, "end_requested": False}

    def _cursor(self, state):
        cursor = gemini_live._Cursor()
        cursor.state = state
        turn = gemini_live._TurnState()
        step = gemini_live._current_step(state)
        turn.spoken_qid = step
        turn.user_turn_step = step
        return cursor, turn

    async def _record(self, state, args):
        cursor, turn = self._cursor(state)
        session = _FakeSession()
        await gemini_live._answer_function_calls(
            session,
            asyncio.Lock(),
            _FakeSocket(),
            asyncio.Lock(),
            db.create_call("gemini"),
            self._telemetry(),
            [_Call("record_turn", args, "fc-target")],
            set(),
            set(),
            cursor,
            turn,
        )
        await _drop_fallback(turn)
        body = session.sent[0][1][0].response
        return cursor.state, body

    def test_tool_result_names_the_current_and_earlier_questions(self):
        decl = gemini_live._RECORD_TURN_DECL["parameters"]["properties"]["target_qid"]
        self.assertEqual(decl["type"], "STRING")
        self.assertIn("Leave empty", decl["description"])
        self.assertIn("last_answered_qid", decl["description"])
        prompt = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("target_qid", prompt)
        self.assertIn("last_answered_qid", prompt)
        self.assertIn("answered_qids", prompt)
        self.assertIn("Leave target_qid empty", prompt)
        for token in ("target_qid", "last_answered_qid", "answered_qids", "recorded_concerns"):
            self.assertTrue(prompts.leaked_caption(token, token))
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        body = gemini_live._tool_body(state)
        self.assertEqual(body["active_qid"], "birth_jaundice")
        self.assertEqual(body["last_answered_qid"], "birth_nicu")
        self.assertIn("birth_mode_detail", body["answered_qids"])
        self.assertNotIn("intro_bridge", body["answered_qids"])
        self.assertNotIn("birth_jaundice", body["answered_qids"])

    def test_record_turn_can_revise_or_reask_an_earlier_question(self):
        async def run():
            heart = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "fam_cancer",
            )
            heart = apply_turn(heart, "concern", note="मामा को ब्लड कैंसर")
            heart = apply_turn(heart, "concern", note="मामा")
            heart = cursor_tests._clear_until(heart, "fam_heart_kidney")
            revised, body = await self._record(heart, {
                "outcome": "concern",
                "target_qid": "fam_cancer",
                "note": "चाची को भी ब्लड कैंसर",
            })
            self.assertEqual(revised["active_qid"], "fam_heart_kidney")
            self.assertEqual(body["active_qid"], "fam_heart_kidney")
            self.assertIn("दिल या गुर्दे", body["say"])
            self.assertNotEqual(revised["answers"].get("fam_heart_kidney", {}).get("state"), "concern")
            self.assertIn("चाची को भी ब्लड कैंसर", revised["answers"]["fam_cancer"]["notes"])
            self.assertIn("fam_cancer", body["answered_qids"])

            thal = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "preg_thalassemia",
            )
            thal = apply_turn(thal, "unknown")
            self.assertEqual(thal["active_qid"], "fam_related")
            rewound, clarify_body = await self._record(thal, {
                "outcome": "clarify",
                "target_qid": "preg_thalassemia",
            })
            self.assertEqual(rewound["active_qid"], "preg_thalassemia")
            self.assertTrue(clarify_body["clarify"])
            self.assertIn("थैलेसीमिया", clarify_body["say"])
            self.assertNotIn("preg_thalassemia", rewound["answers"])
            landed, _clear_body = await self._record(rewound, {"outcome": "clear"})
            self.assertEqual(landed["answers"]["preg_thalassemia"]["state"], "clear")
            self.assertEqual(landed["active_qid"], "fam_related")
            self.assertEqual(landed["answers"]["birth_mode_detail"]["state"], "clear")

            follow = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "fam_maternal_bleeding",
            )
            follow = apply_turn(follow, "concern")
            self.assertEqual(follow["pending_followup_idx"], 0)
            held, _body = await self._record(follow, {"outcome": "clarify"})
            self.assertEqual(held["active_qid"], "fam_maternal_bleeding")
            self.assertEqual(held["pending_followup_idx"], 0)
            self.assertEqual(held["step_attempts"], 0)

        asyncio.run(run())


class Stage6BTelemetryTests(unittest.TestCase):
    def _telemetry(self):
        return {
            "tool_calls": 0,
            "tool_errors": 0,
            "end_requested": False,
            "audio_chunks_out": 0,
            "bot_turns": 0,
            "user_turns": 0,
            "turn_latencies_ms": [],
            "errors": [],
            "reconnects": 0,
        }

    def test_quick_reply_counts_a_user_turn_and_stale_turn_does_not_repeat_the_cue(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            session = _FakeSession()
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
            telemetry = self._telemetry()
            call_id = db.create_call("gemini")
            await gemini_live._browser_loop(
                session,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                asyncio.Event(),
                call_id,
                telemetry,
                turn,
                cursor,
                _Recorder(),
            )
            self.assertEqual(telemetry["user_turns"], 1)
            cues = [
                item for item in session.sent
                if item[0] == "realtime" and "SYSTEM CUE" in item[1].get("text", "")
            ]
            self.assertEqual(len(cues), 1)
            self.assertEqual(
                len([item for item in ws.sent if item.get("type") == "question_state"]),
                1,
            )

            await gemini_live._handle_server_message(
                _server_message(
                    turn_complete=True,
                    tool_calls=[_Call("record_turn", {"outcome": "clear"}, "fc-stale")],
                ),
                session,
                asyncio.Lock(),
                ws,
                asyncio.Lock(),
                call_id,
                telemetry,
                turn,
                _Recorder(),
                set(),
                set(),
                cursor,
                stop=asyncio.Event(),
            )
            cues = [
                item for item in session.sent
                if item[0] == "realtime" and "SYSTEM CUE" in item[1].get("text", "")
            ]
            self.assertEqual(len(cues), 1)
            self.assertEqual(turn.pending_speak_step, "")
            self.assertIsNone(turn.speak_fallback)
            tool = [item for item in session.sent if item[0] == "tool"]
            self.assertEqual(tool[-1][1][0].response["result"], "held")
            self.assertEqual(tool[-1][1][0].response["reason"], "stale_turn")
            self.assertEqual(
                len([item for item in ws.sent if item.get("type") == "question_state"]),
                1,
            )
            details = [
                item["last_event"]["detail"]
                for item in ws.sent
                if item.get("type") == "telemetry"
            ]
            self.assertIn("record_turn_held", details)
            self.assertEqual(telemetry["user_turns"], 1)

            closing = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "close_anything_else",
            )
            closing = apply_turn(closing, "concern", note="पेट दर्द")
            close_cursor = gemini_live._Cursor()
            close_cursor.state = closing
            close_step = gemini_live._current_step(closing)
            close_turn = gemini_live._TurnState()
            close_turn.spoken_qid = close_step
            close_ws = _FakeSocket([
                {"type": "quick_reply", "outcome": "clear", "step": close_step},
            ])
            close_telemetry = self._telemetry()
            await gemini_live._browser_loop(
                _FakeSession(),
                asyncio.Lock(),
                close_ws,
                asyncio.Lock(),
                asyncio.Event(),
                db.create_call("gemini"),
                close_telemetry,
                close_turn,
                close_cursor,
                _Recorder(),
            )
            self.assertEqual(close_cursor.state["phase"], "closing")
            self.assertEqual(close_telemetry["user_turns"], 1)

        asyncio.run(run())

    def test_dropoff_qid_follows_an_incomplete_call_and_clears_when_complete(self):
        class _Saver:
            def save(self):
                return "", 0

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            asking = json.loads(json.dumps(state))
            cursor = gemini_live._Cursor()
            cursor.state = state
            call_id = db.create_call("gemini")
            telemetry = gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)
            await gemini_live._mark_closing(
                cursor,
                "silence",
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                telemetry,
                None,
            )
            self.assertEqual(telemetry["dropoff_qid"], "birth_jaundice")

            finished = gemini_live._Cursor()
            finished.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "close_anything_else",
            )
            finished.state["active_qid"] = "close_anything_else"
            done_telemetry = gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)
            await gemini_live._mark_closing(
                finished,
                "complete",
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                done_telemetry,
                None,
            )
            self.assertEqual(finished.state["close_reason"], "complete")
            self.assertEqual(done_telemetry["dropoff_qid"], "")

            abrupt_id = db.create_call("gemini")
            abrupt = gemini_live._Cursor()
            abrupt.state = asking
            abrupt_telemetry = gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)
            await gemini_live._finalize(
                _FakeSocket(),
                asyncio.Lock(),
                abrupt_id,
                abrupt_telemetry,
                _Saver(),
                None,
                abrupt,
            )
            saved = db.get_call(abrupt_id)["telemetry"]
            self.assertEqual(saved["dropoff_qid"], "birth_jaundice")

            complete_id = db.create_call("gemini")
            complete_telemetry = gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)
            await gemini_live._finalize(
                _FakeSocket(),
                asyncio.Lock(),
                complete_id,
                complete_telemetry,
                _Saver(),
                None,
                finished,
            )
            saved_complete = db.get_call(complete_id)["telemetry"]
            self.assertEqual(saved_complete["dropoff_qid"], "")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
