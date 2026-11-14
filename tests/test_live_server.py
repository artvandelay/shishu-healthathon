"""Stage 2: pause-safe VAD, non-interrupting fallback, and quick replies."""

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import time
import unittest
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import tests  # noqa: E402, F401

_TMP = tempfile.mkdtemp(prefix="shishu-v4-live-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import call_cursor  # noqa: E402
import db  # noqa: E402
import decisions  # noqa: E402
import gemini_live  # noqa: E402
import live_decision_bridge  # noqa: E402
import profile_extractor  # noqa: E402
import prompts  # noqa: E402
import test_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_profile, apply_turn, close_technical_failure  # noqa: E402


class _FakeSocket:
    def __init__(self, messages=None):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []
        self._messages = list(messages or [])
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True
        self.client_state = gemini_live.WebSocketState.CONNECTED

    async def close(self):
        self.closed = True
        self.client_state = gemini_live.WebSocketState.DISCONNECTED

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
    def add_user_pcm(self, pcm, rate):
        return None

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

    def test_followup_spoken_detail_is_kept_when_tool_omits_note(self):
        async def run():
            call_id = db.create_call("gemini")
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(5, "female")]),
                "close_anything_else",
            )
            state = apply_turn(state, "concern")
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.user_turn_step = step
            turn.last_user_text = "हाँ।"
            session = _FakeSession()
            telemetry = gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                telemetry,
                [_Call("record_turn", {"outcome": "concern"}, "f1")],
                set(),
                set(),
                cursor,
                turn,
            )
            self.assertEqual(cursor.state["pending_followup_idx"], 0)
            self.assertEqual(cursor.state["step_attempts"], 1)

            retry_step = gemini_live._current_step(cursor.state)
            turn.spoken_qid = retry_step
            turn.user_turn_step = retry_step
            turn.last_user_text = "बच्ची को रात में नींद में डर लगता है।"
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                telemetry,
                [_Call("record_turn", {"outcome": "concern"}, "f2")],
                set(),
                set(),
                cursor,
                turn,
            )
            self.assertIsNone(cursor.state["pending_followup_idx"])
            self.assertEqual(cursor.state["answers"]["close_anything_else"]["state"], "concern")
            self.assertIn("नींद में डर", cursor.state["answers"]["close_anything_else"]["notes"])

        asyncio.run(run())

    def test_technical_failure_closing_hindi_is_calm_incomplete_without_108(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
            "birth_jaundice",
        )
        closed = close_technical_failure(state)
        line = gemini_live._closing_hi(closed)
        self.assertIn("पूजा जी", line)
        self.assertIn("अधूरी", line)
        self.assertNotIn("108", line)
        self.assertNotIn("अस्पताल", line)

        nameless = json.loads(json.dumps(closed))
        nameless["profile"]["caller_name"] = None
        nameless_line = gemini_live._closing_hi(nameless)
        self.assertIn("अधूरी", nameless_line)
        self.assertNotIn("108", nameless_line)
        self.assertNotIn("अस्पताल", nameless_line)

        telemetry = gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)
        gemini_live._stamp_dropoff(telemetry, closed)
        self.assertEqual(telemetry["dropoff_qid"], "birth_jaundice")


class Phase3ShadowModeTests(unittest.TestCase):
    def _telemetry(self):
        return gemini_live._new_telemetry("gemini", "Kore", 1000, 300, 1)

    def _msg_with_finished(
        self,
        *,
        user_text="",
        finished=None,
        has_finished_attr=True,
        turn_complete=False,
        tool_calls=None,
    ):
        tool_call = SimpleNamespace(function_calls=tool_calls) if tool_calls else None
        if user_text or has_finished_attr and finished is not None:
            if has_finished_attr:
                tr_in = SimpleNamespace(text=user_text, finished=finished)
            else:
                tr_in = SimpleNamespace(text=user_text)
        else:
            tr_in = None
        return SimpleNamespace(
            tool_call_cancellation=None,
            server_content=SimpleNamespace(
                interrupted=False,
                input_transcription=tr_in,
                output_transcription=None,
                model_turn=None,
                turn_complete=turn_complete,
            ),
            tool_call=tool_call,
            session_resumption_update=None,
            go_away=None,
        )

    def _fake_raw_answers(self, *, active_question="", utterance=""):
        has_side = "कैंसर" in utterance
        side_intent = (
            {"choice": "add_concern", "probabilities": {"add_concern": 0.89, "other": 0.06, "clarify_other": 0.05}}
            if has_side
            else {"choice": "other", "probabilities": {"other": 0.95, "add_concern": 0.03}}
        )
        side_qid = (
            {"choice": "fam_cancer", "probabilities": {"fam_cancer": 0.90, "other": 0.06, "fam_related": 0.04}}
            if has_side
            else {"choice": "other", "probabilities": {"other": 0.94, "fam_cancer": 0.03}}
        )
        outcome_choice = "concern" if ("हाँ" in utterance or "तीन दिन" in utterance or "रेखा" in utterance) else "clear"
        outcome_probs = (
            {"concern": 0.91, "other": 0.05, "clear": 0.04}
            if outcome_choice == "concern"
            else {"clear": 0.93, "other": 0.04, "concern": 0.03}
        )
        is_intro = any(tok in active_question for tok in ("नाम क्या", "शादी हो गई", "कितने बच्चे"))
        return {
            "danger_now": {"noul": 0.02},
            "caller_stop": {"noul": 0.01},
            "mentions_profile": {"noul": 0.92 if is_intro else 0.04},
            "current_outcome": {"choice": outcome_choice, "probabilities": outcome_probs},
            "side_intent": side_intent,
            "side_qid": side_qid,
            "detail_present": {"noul": 0.88 if "तीन दिन" in utterance else 0.12},
        }

    def test_decision_mode_helper_and_default_settings_unchanged(self):
        self.assertNotIn("decision_mode", db.DEFAULT_SETTINGS)
        self.assertNotIn("decision_shadow", gemini_live._TELEMETRY_EVENTS)
        self.assertFalse(hasattr(gemini_live, "_active_authority_thresholds"))
        self.assertFalse(hasattr(gemini_live, "_resolve_decision_mode"))
        self.assertFalse(hasattr(gemini_live, "_resolve_authority_thresholds"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DECISION_MODE", None)
            self.assertEqual(gemini_live._decision_mode(), "off")
            self.assertEqual(gemini_live._decision_mode({}), "off")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": "authority"}), "authority")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": "AUTHORITY"}), "authority")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": "invalid"}), "off")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": "SHADOW"}), "shadow")
        with mock.patch.dict(os.environ, {"DECISION_MODE": "shadow"}):
            self.assertEqual(gemini_live._decision_mode(), "shadow")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": ""}), "shadow")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": "off"}), "off")
            self.assertEqual(gemini_live._decision_mode({"decision_mode": "authority"}), "authority")

    def test_off_mode_never_calls_decisions_evaluate_turn(self):
        async def run():
            cursor = gemini_live._Cursor()
            turn = gemini_live._TurnState()
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DECISION_MODE", None)
                with mock.patch.object(decisions, "evaluate_turn") as eval_mock:
                    await gemini_live._handle_server_message(
                        _server_message(user_text="मेरा नाम रेखा है"),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("save_profile", {"caller_name": "रेखा"}, "fc-off")],
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
                    await _drop_fallback(turn)
                    eval_mock.assert_not_called()
            events = db.get_call(call_id)["events"]
            self.assertFalse(any(e["type"] == "decision_shadow" for e in events))

        asyncio.run(run())

    def test_multi_chunk_accumulation_and_finished_flag_handling(self):
        async def run():
            # 1) Multi-chunk deltas where _Transcription fakes have only .text (no .finished attr)
            cursor = gemini_live._Cursor()
            cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            step = gemini_live._current_step(cursor.state)
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini")

            seen_evals = []

            async def fake_eval(request, **_kwargs):
                seen_evals.append(request)
                ctx = json.loads(request["input"])
                return decisions.DecisionCallResult(
                    ok=True,
                    error_reason="",
                    attempts=1,
                    decision_ms=42,
                    request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
                    question_count=len(request["questions"]),
                    catalog_option_count=len(((request["questions"].get("side_qid") or {}).get("criteria")) or {}),
                    usage_present=True,
                    total_input_tokens=321,
                    raw_answers=self._fake_raw_answers(
                        active_question=ctx.get("active_question", ""),
                        utterance=ctx.get("caller_utterance", ""),
                    ),
                )

            with mock.patch.dict(os.environ, {"DECISION_MODE": "shadow"}):
                with mock.patch.object(decisions, "evaluate_turn", side_effect=fake_eval):
                    await gemini_live._handle_server_message(
                        _server_message(user_text="हाँ, बच्चे को"),
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
                    self.assertEqual(turn.capture.user_turn_buffer, "हाँ, बच्चे को")
                    self.assertEqual(turn.last_user_text, "हाँ, बच्चे को")
                    self.assertEqual(len(seen_evals), 0)

                    await gemini_live._handle_server_message(
                        _server_message(user_text="तीन दिन नीली लाइट में रखा था।"),
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
                    expected = "हाँ, बच्चे को तीन दिन नीली लाइट में रखा था।"
                    self.assertEqual(turn.capture.user_turn_buffer, expected)
                    self.assertEqual(turn.last_user_text, expected)

                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("record_turn", {"outcome": "concern", "note": "तीन दिन", "followup_answered": True}, "fc-multi")],
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
                    await _drop_fallback(turn)
                    await asyncio.gather(*list(turn.capture.shadow_tasks))

            self.assertEqual(turn.capture.user_turn_buffer, "")
            self.assertEqual(turn.last_user_text, expected)
            self.assertEqual(len(seen_evals), 1)
            ctx0 = json.loads(seen_evals[0]["input"])
            self.assertIn("पीलिया", ctx0["active_question"])
            self.assertEqual(ctx0["caller_utterance"], expected)

            # 2) Explicit finished=True chunk followed by empty-text finished=True / tool_call
            cursor2 = gemini_live._Cursor()
            cursor2.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            turn2 = gemini_live._TurnState()
            turn2.spoken_qid = gemini_live._current_step(cursor2.state)
            call_id2 = db.create_call("gemini")
            seen_evals.clear()
            with mock.patch.dict(os.environ, {"DECISION_MODE": "shadow"}):
                with mock.patch.object(decisions, "evaluate_turn", side_effect=fake_eval):
                    await gemini_live._handle_server_message(
                        self._msg_with_finished(user_text="हाँ, बच्चे को", finished=False),
                        session,
                        asyncio.Lock(),
                        _FakeSocket(),
                        asyncio.Lock(),
                        call_id2,
                        self._telemetry(),
                        turn2,
                        _Recorder(),
                        set(),
                        set(),
                        cursor2,
                        stop=asyncio.Event(),
                    )
                    await gemini_live._handle_server_message(
                        self._msg_with_finished(user_text="तीन दिन रखा था।", finished=True),
                        session,
                        asyncio.Lock(),
                        _FakeSocket(),
                        asyncio.Lock(),
                        call_id2,
                        self._telemetry(),
                        turn2,
                        _Recorder(),
                        set(),
                        set(),
                        cursor2,
                        stop=asyncio.Event(),
                    )
                    # Tool runs after finished=True -> must NOT dispatch a second shadow
                    await gemini_live._handle_server_message(
                        self._msg_with_finished(
                            turn_complete=True,
                            tool_calls=[_Call("record_turn", {"outcome": "concern", "note": "तीन दिन"}, "fc-fin")],
                        ),
                        session,
                        asyncio.Lock(),
                        _FakeSocket(),
                        asyncio.Lock(),
                        call_id2,
                        self._telemetry(),
                        turn2,
                        _Recorder(),
                        set(),
                        set(),
                        cursor2,
                        stop=asyncio.Event(),
                    )
                    await _drop_fallback(turn2)
                    await asyncio.gather(*list(turn2.capture.shadow_tasks))
            self.assertEqual(len(seen_evals), 1)
            self.assertIn("हाँ, बच्चे को तीन दिन रखा था।", seen_evals[0]["input"])
            self.assertEqual(turn2.capture.user_turn_buffer, "")

        asyncio.run(run())

    def test_shadow_session_matches_off_state_byte_for_byte_and_logs_future_qid(self):
        class _Saver:
            def save(self):
                return "", 0

        async def run_script(mode: str):
            cursor = gemini_live._Cursor()
            turn = gemini_live._TurnState()
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini")
            seen_requests = []

            async def fake_eval(request, **_kwargs):
                seen_requests.append(request)
                ctx = json.loads(request["input"])
                return decisions.DecisionCallResult(
                    ok=True,
                    error_reason="",
                    attempts=1,
                    decision_ms=57,
                    request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
                    question_count=len(request["questions"]),
                    catalog_option_count=len(((request["questions"].get("side_qid") or {}).get("criteria")) or {}),
                    usage_present=True,
                    total_input_tokens=450,
                    raw_answers=self._fake_raw_answers(
                        active_question=ctx.get("active_question", ""),
                        utterance=ctx.get("caller_utterance", ""),
                    ),
                )

            with mock.patch.dict(os.environ, {"DECISION_MODE": mode}):
                patch_ctx = (
                    mock.patch.object(decisions, "evaluate_turn", side_effect=fake_eval)
                    if mode == "shadow"
                    else mock.patch.object(decisions, "evaluate_turn")
                )
                with patch_ctx as mocked_eval:
                    # Turn 1: intro_name (save_profile)
                    await gemini_live._handle_server_message(
                        _server_message(user_text="मेरा नाम रेखा है"),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("save_profile", {"caller_name": "रेखा"}, "fc1")],
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
                    await _drop_fallback(turn)

                    # Turn 2: intro_married (save_profile)
                    await gemini_live._handle_server_message(
                        _server_message(user_text="हाँ, शादी हो गई है"),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("save_profile", {"marital_status": "married"}, "fc2")],
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
                    await _drop_fallback(turn)

                    # Turn 3: intro_children (save_profile)
                    await gemini_live._handle_server_message(
                        _server_message(user_text="एक बेटा है 4 साल का"),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[
                                _Call(
                                    "save_profile",
                                    {"child_count": 1, "children": [{"age_years": 4, "sex": "male"}]},
                                    "fc3",
                                )
                            ],
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
                    await _drop_fallback(turn)

                    # Turn 4: quick_reply on intro_bridge (MUST NOT trigger shadow!)
                    turn.spoken_qid = gemini_live._current_step(cursor.state)
                    quick_ws = _FakeSocket([
                        {
                            "type": "quick_reply",
                            "outcome": "clear",
                            "step": gemini_live._current_step(cursor.state),
                        }
                    ])
                    await gemini_live._browser_loop(
                        session,
                        asyncio.Lock(),
                        quick_ws,
                        asyncio.Lock(),
                        asyncio.Event(),
                        call_id,
                        telemetry,
                        turn,
                        cursor,
                        _Recorder(),
                    )
                    # Trailing ASR within quick_reply window -> also ignored for shadow!
                    await gemini_live._handle_server_message(
                        _server_message(user_text="हाँ", turn_complete=True),
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

                    # Turn 5: spoken multi-chunk on birth_mode_detail mentioning a future qid (fam_cancer)
                    turn.last_quick_reply_step = ""
                    turn.last_quick_reply_ts = 0.0
                    turn.spoken_qid = gemini_live._current_step(cursor.state)
                    await gemini_live._handle_server_message(
                        _server_message(user_text="नहीं नॉर्मल डिलीवरी थी"),
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
                    await gemini_live._handle_server_message(
                        _server_message(user_text="पर परिवार में चाचा को कैंसर हुआ था।"),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("record_turn", {"outcome": "clear"}, "fc5")],
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
                    await _drop_fallback(turn)

                    await gemini_live._finalize(
                        ws,
                        asyncio.Lock(),
                        call_id,
                        telemetry,
                        _Saver(),
                        None,
                        cursor,
                        turn=turn,
                    )
                    if mode == "off":
                        mocked_eval.assert_not_called()

            return cursor.state, db.get_call(call_id), telemetry, seen_requests, turn

        async def compare():
            off_state, off_call, off_tel, off_reqs, off_turn = await run_script("off")
            sh_state, sh_call, sh_tel, sh_reqs, sh_turn = await run_script("shadow")

            self.assertEqual(
                json.dumps(off_state, ensure_ascii=False, sort_keys=True),
                json.dumps(sh_state, ensure_ascii=False, sort_keys=True),
            )
            self.assertEqual(
                json.dumps(off_call["cursor_state"], ensure_ascii=False, sort_keys=True),
                json.dumps(sh_call["cursor_state"], ensure_ascii=False, sort_keys=True),
            )
            self.assertEqual(off_tel["last_interaction_status"], sh_tel["last_interaction_status"])
            self.assertEqual(off_reqs, [])
            # 4 spoken turns (Turns 1, 2, 3, 5); quick_reply + trailing ASR on Turn 4 skipped
            self.assertEqual(len(sh_reqs), 4)
            self.assertEqual(len(sh_turn.capture.shadow_tasks), 0)

            off_shadow_events = [e for e in off_call["events"] if e["type"] == "decision_shadow"]
            self.assertEqual(off_shadow_events, [])

            sh_events = [e for e in sh_call["events"] if e["type"] == "decision_shadow"]
            self.assertEqual(len(sh_events), 4)
            # Pre-tool cursor active_qid preserved (Turn 5 evaluated on birth_mode_detail, not birth_nicu)
            active_qids = [e["meta"]["diagnostics"]["active_qid"] for e in sh_events]
            self.assertEqual(
                active_qids,
                ["intro_name", "intro_married", "intro_children", "birth_mode_detail"],
            )
            for idx, evt in enumerate(sh_events):
                meta = evt["meta"]
                self.assertTrue(meta["ok"])
                self.assertEqual(meta["error_reason"], "")
                self.assertEqual(meta["decision_ms"], 57)
                self.assertEqual(meta["attempts"], 1)
                self.assertGreater(meta["request_bytes"], 100)
                if idx < 3:
                    self.assertEqual(meta["question_count"], 4)
                    self.assertEqual(meta["catalog_option_count"], 0)
                else:
                    self.assertGreaterEqual(meta["question_count"], 6)
                    self.assertEqual(meta["catalog_option_count"], 59)
                self.assertTrue(meta["usage_present"])
                self.assertEqual(meta["total_input_tokens"], 450)
                self.assertIn("diagnostics", meta)
            diag_last = sh_events[-1]["meta"]["diagnostics"]
            self.assertIsNone(diag_last["future_qid_match"])
            self.assertEqual(
                diag_last["raw_side_candidate"]["qid"],
                "fam_cancer",
            )
            self.assertAlmostEqual(
                diag_last["raw_side_candidate"]["confidence"],
                0.90,
            )
            self.assertEqual(
                diag_last["raw_side_candidate"]["side_intent"],
                "add_concern",
            )

        asyncio.run(compare())

    def test_shadow_evaluation_is_non_blocking_and_cleans_up_on_finalize(self):
        class _Saver:
            def save(self):
                return "", 0

        async def run():
            gate = asyncio.Event()
            started = asyncio.Event()

            async def slow_eval(request, **_kwargs):
                started.set()
                await gate.wait()
                return decisions.DecisionCallResult(
                    ok=False,
                    error_reason="timeout",
                    attempts=1,
                    decision_ms=2000,
                    request_bytes=123,
                    question_count=8,
                    catalog_option_count=59,
                    usage_present=False,
                    total_input_tokens=None,
                    raw_answers={},
                )

            cursor = gemini_live._Cursor()
            cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            turn = gemini_live._TurnState()
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            session = _FakeSession()
            ws = _FakeSocket()
            call_id = db.create_call("gemini")
            telemetry = self._telemetry()

            with mock.patch.dict(os.environ, {"DECISION_MODE": "shadow"}):
                with mock.patch.object(decisions, "evaluate_turn", side_effect=slow_eval):
                    await gemini_live._handle_server_message(
                        _server_message(user_text="नहीं"),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("record_turn", {"outcome": "clear"}, "fc-nonblock")],
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
                    # Tool response & speak cue already sent BEFORE shadow eval finished!
                    self.assertEqual(_kinds(session), ["tool", "realtime"])
                    await asyncio.wait_for(started.wait(), timeout=1.0)
                    self.assertEqual(len(turn.capture.shadow_tasks), 1)
                    # Finalize cancels any still-pending shadow task after timeout without leaking
                    await gemini_live._finalize(
                        ws,
                        asyncio.Lock(),
                        call_id,
                        telemetry,
                        _Saver(),
                        None,
                        cursor,
                        turn=turn,
                    )
                    self.assertEqual(len(turn.capture.shadow_tasks), 0)

        asyncio.run(run())


def _calibrated_test_thresholds() -> decisions.DecisionThresholds:
    return decisions.DecisionThresholds(
        calibrated=True,
        danger_now_min_yes=0.80,
        caller_stop_min_yes=0.80,
        mentions_profile_min_yes=0.65,
        detail_present_min_yes=0.65,
        outcome_min_confidence=0.65,
        outcome_min_margin=0.15,
        side_intent_min_confidence=0.65,
        side_intent_min_margin=0.15,
        side_qid_min_confidence=0.70,
        side_qid_min_margin=0.20,
        child_ref_min_confidence=0.65,
        child_ref_min_margin=0.15,
    )


def _authority_audio_msg(pcm: bytes = b"\x01\x00" * 240, *, turn_complete: bool = False):
    part = SimpleNamespace(
        thought=False,
        function_call=None,
        inline_data=SimpleNamespace(
            data=pcm,
            mime_type="audio/pcm;rate=24000",
        ),
    )
    return SimpleNamespace(
        tool_call_cancellation=None,
        server_content=SimpleNamespace(
            interrupted=False,
            input_transcription=None,
            output_transcription=None,
            model_turn=SimpleNamespace(parts=[part]),
            turn_complete=turn_complete,
        ),
        tool_call=None,
        session_resumption_update=None,
        go_away=None,
    )


def _noul_ans(prob: float) -> dict:
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1.0 - prob)}


def _choice_ans(winner: str, probs: dict[str, float]) -> dict:
    return {
        "type": "choice",
        "choice": winner,
        "probabilities": probs,
        "confidence": probs[winner],
    }


def _make_decision_result(request: dict, raw_answers: dict, *, ok: bool = True, error_reason: str = "", attempts: int = 1):
    questions = request.get("questions") or {}
    side_criteria = (questions.get("side_qid") or {}).get("criteria") or {}
    return decisions.DecisionCallResult(
        ok=ok,
        error_reason=error_reason,
        attempts=attempts,
        decision_ms=48,
        request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
        question_count=len(questions),
        catalog_option_count=len(side_criteria),
        usage_present=True,
        total_input_tokens=380,
        raw_answers=raw_answers if ok else {},
    )


class Phase4AuthorityModeTests(unittest.TestCase):
    def _telemetry(self):
        return gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1000, 300, 1)

    def test_authority_prompt_leak_guards_and_live_config_tool_isolation(self):
        self.assertIn("save_profile", prompts.GEMINI_LIVE_SYSTEM_PROMPT)
        self.assertIn("record_turn", prompts.GEMINI_LIVE_SYSTEM_PROMPT)

        auth_prompt = prompts.GEMINI_LIVE_AUTHORITY_SYSTEM_PROMPT
        self.assertIn("Kore", auth_prompt)
        self.assertIn("next_question", auth_prompt)
        self.assertIn("say", auth_prompt)
        self.assertIn("clarify", auth_prompt)
        self.assertIn("BCG", auth_prompt)
        self.assertIn("थैलेसीमिया माइनर या ट्रेट", auth_prompt)
        self.assertIn("recorded_concerns", auth_prompt)
        self.assertIn("stop", auth_prompt)
        self.assertIn("SYSTEM CUE", auth_prompt)
        for disease in (
            "Duchenne",
            "Hemophilia",
            "Pompe",
            "Gaucher",
            "SCID",
            "SMA",
            "MPS",
            "Wiskott-Aldrich",
            "ड्यूशेन",
            "हीमोफीलिया",
            "पॉम्पे",
            "गॉशर",
            "इम्यूनोडेफिशिएंसी",
            "म्यूकोपॉलीसेकेराइडोसिस",
        ):
            self.assertIn(disease, auth_prompt)
        for forbidden in (
            "save_profile",
            "record_turn",
            "target_qid",
            "child_ref",
            "outcome",
            "marital_status",
            "caller_stop",
            "ever_pregnant",
            "emergency",
        ):
            self.assertNotIn(forbidden, auth_prompt)

        leaked = "next_question कहो क्या नाल गिरी?"
        cleaned = prompts.clean_assistant_caption(leaked)
        self.assertNotIn("next_question", cleaned)
        self.assertEqual(cleaned, "कहो क्या नाल गिरी?")
        self.assertTrue(prompts.leaked_caption(leaked, cleaned))

        auth_cfg = gemini_live._live_config(
            {"gemini_model": "gemini-3.1-flash-live-preview", "decision_mode": "authority"}
        )
        self.assertEqual(
            auth_cfg.system_instruction.parts[0].text,
            prompts.GEMINI_LIVE_AUTHORITY_SYSTEM_PROMPT,
        )
        auth_decls = auth_cfg.tools[0].function_declarations
        self.assertEqual([d.name for d in auth_decls], ["next_question"])
        self.assertEqual(dict(auth_decls[0].parameters.properties or {}), {})

        for mode in ("off", "shadow"):
            cfg = gemini_live._live_config(
                {"gemini_model": "gemini-3.1-flash-live-preview", "decision_mode": mode}
            )
            self.assertEqual(
                cfg.system_instruction.parts[0].text,
                prompts.GEMINI_LIVE_SYSTEM_PROMPT,
            )
            decls = cfg.tools[0].function_declarations
            self.assertEqual([d.name for d in decls], ["save_profile", "record_turn"])

    def test_authority_mode_fails_closed_without_calibrated_thresholds(self):
        self.assertNotIn("decision_authority", gemini_live._TELEMETRY_EVENTS)
        thresholds = _calibrated_test_thresholds()

        with tempfile.TemporaryDirectory() as tmp:
            uncal_path = Path(tmp) / "uncal.json"
            uncal_path.write_text(
                json.dumps(asdict(decisions.UNCALIBRATED_THRESHOLDS)),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                decisions.load_thresholds(uncal_path, require_calibrated=True)
            with self.assertRaises(ValueError):
                gemini_live._load_authority_thresholds(
                    {"decision_thresholds_path": str(uncal_path)}
                )

            cal_path = Path(tmp) / "cal.json"
            cal_path.write_text(json.dumps(asdict(thresholds)), encoding="utf-8")
            loaded_setting = gemini_live._load_authority_thresholds(
                {"decision_thresholds_path": str(cal_path)}
            )
            self.assertEqual(loaded_setting, thresholds)
            with mock.patch.dict(
                os.environ, {"DECISION_THRESHOLDS_PATH": str(cal_path)}
            ):
                loaded_env = gemini_live._load_authority_thresholds({})
                self.assertEqual(loaded_env, thresholds)

        async def refuse_missing():
            ws = _FakeSocket()
            with mock.patch.dict(
                os.environ,
                {
                    "DECISION_MODE": "authority",
                    "DECISION_THRESHOLDS_PATH": "/tmp/does-not-exist-shishu-thresholds.json",
                    "GEMINI_API_KEY": "test-key",
                },
                clear=False,
            ):
                with mock.patch.object(
                    gemini_live, "_connect_live"
                ) as connect_mock, mock.patch.object(
                    db, "create_call"
                ) as create_mock:
                    await gemini_live.handle_gemini_ws(ws)
                    connect_mock.assert_not_called()
                    create_mock.assert_not_called()
            self.assertTrue(ws.closed)
            errors = [m for m in ws.sent if m.get("type") == "error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(
                errors[0]["message"],
                "Decision authority mode requires calibrated thresholds",
            )

        asyncio.run(refuse_missing())

    def test_authority_happy_path_dual_intent_smoke_a_and_profile_hook(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "fam_cancer",
            )
            state = apply_turn(state, "concern", note="मामा को ब्लड कैंसर")
            state = apply_turn(state, "concern", note="मामा")
            state = cursor_tests._clear_until(state, "fam_heart_kidney")

            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            seen_requests = []

            async def fake_eval(request, **_kwargs):
                seen_requests.append(request)
                raw = {
                    "danger_now": _noul_ans(0.02),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.03),
                    "detail_present": _noul_ans(0.05),
                    "current_outcome": _choice_ans(
                        "clear",
                        {"clear": 0.92, "other": 0.04, "concern": 0.03, "unknown": 0.01},
                    ),
                    "side_intent": _choice_ans(
                        "add_concern",
                        {"add_concern": 0.93, "other": 0.05, "clarify_other": 0.02},
                    ),
                    "side_qid": _choice_ans(
                        "fam_cancer",
                        {"fam_cancer": 0.91, "other": 0.04, "fam_lumps": 0.03, "fam_heart_kidney": 0.02},
                    ),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                with mock.patch.object(
                    profile_extractor,
                    "extract_profile_entities",
                    return_value=profile_extractor.ProfileExtractionResult(
                        ok=True,
                        updates={"caller_name": "रेखा"},
                        error_reason="",
                        extractor_ms=12,
                    ),
                ) as ext_mock:
                    ext_res = await live_decision_bridge.extract_profile_updates(
                        cursor.state, "रेखा", None
                    )
                    self.assertTrue(ext_res.ok)
                    self.assertEqual(ext_res.updates, {"caller_name": "रेखा"})
                    ext_mock.assert_awaited_once()
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं, पर चाची को भी कैंसर हुआ था"),
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
                self.assertFalse(turn.capture.authority_speech_allowed)

                # Duplicate next_question in same batch must only evaluate Decision once
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[
                            _Call("next_question", {}, "fc-auth-1"),
                            _Call("next_question", {}, "fc-auth-dup"),
                        ],
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
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(len(seen_requests), 1)
            self.assertIsInstance(seen_requests[0]["input"], str)
            self.assertEqual(cursor.state["answers"]["fam_heart_kidney"]["state"], "clear")
            self.assertIn("चाची को भी कैंसर हुआ था", cursor.state["answers"]["fam_cancer"]["notes"])
            self.assertIn("मामा को ब्लड कैंसर", cursor.state["answers"]["fam_cancer"]["notes"])
            self.assertEqual(cursor.state["active_qid"], "fam_bp_stroke")
            self.assertTrue(turn.capture.authority_speech_allowed)
            self.assertEqual(turn.capture.user_turn_buffer, "")
            self.assertEqual(len(turn.capture.user_turn_pcm), 0)

            tool_resps = [item[1] for item in session.sent if item[0] == "tool"][0]
            self.assertEqual(tool_resps[0].name, "next_question")
            self.assertEqual(tool_resps[0].response["result"], "ok")
            self.assertEqual(tool_resps[0].response["active_qid"], "fam_bp_stroke")
            self.assertEqual(tool_resps[1].response["result"], "held")
            self.assertEqual(tool_resps[1].response["reason"], "duplicate_turn")

            cues = [item[1]["text"] for item in session.sent if item[0] == "realtime"]
            self.assertEqual(len(cues), 1)
            self.assertIn("40-50 साल", cues[0])

            qstates = [m for m in ws.sent if m.get("type") == "question_state"]
            self.assertEqual(len(qstates), 1)
            self.assertEqual(qstates[0]["qid"], "fam_bp_stroke")

            auth_events = [
                e for e in db.get_call(call_id)["events"] if e["type"] == "decision_authority"
            ]
            self.assertEqual(len(auth_events), 1)
            meta = auth_events[0]["meta"]
            for key in (
                "ok",
                "error_reason",
                "decision_ms",
                "attempts",
                "request_bytes",
                "question_count",
                "catalog_option_count",
                "usage_present",
                "total_input_tokens",
            ):
                self.assertIn(key, meta)
            self.assertTrue(meta["ok"])
            self.assertEqual(meta["total_input_tokens"], 380)

        asyncio.run(run())

    def test_authority_clarify_and_future_qid_match_logged(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            thal = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "preg_thalassemia",
            )
            thal = apply_turn(thal, "unknown")
            cursor = gemini_live._Cursor()
            cursor.state = thal
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            async def fake_eval(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.05),
                    "current_outcome": _choice_ans(
                        "clear", {"clear": 0.91, "other": 0.05, "concern": 0.04}
                    ),
                    "side_intent": _choice_ans(
                        "clarify_other",
                        {"clarify_other": 0.91, "add_concern": 0.05, "other": 0.04},
                    ),
                    "side_qid": _choice_ans(
                        "preg_thalassemia",
                        {"preg_thalassemia": 0.90, "other": 0.05, "fam_related": 0.05},
                    ),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                await gemini_live._handle_server_message(
                    _server_message(user_text="रिश्तेदार तो नहीं, पर थैलेसीमिया क्या होता है?"),
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
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-clar")],
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
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(cursor.state["active_qid"], "preg_thalassemia")
            tool_resp = [item[1] for item in session.sent if item[0] == "tool"][0][0]
            self.assertTrue(tool_resp.response.get("clarify"))
            cues = [item[1]["text"] for item in session.sent if item[0] == "realtime"]
            self.assertIn("she did not understand", cues[0])

            # Future qid match logged on next turn
            cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            turn.spoken_qid = gemini_live._current_step(cursor.state)

            async def fake_eval_future(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.05),
                    "current_outcome": _choice_ans(
                        "clear", {"clear": 0.90, "other": 0.06, "concern": 0.04}
                    ),
                    "side_intent": _choice_ans(
                        "add_concern",
                        {"add_concern": 0.91, "other": 0.06, "clarify_other": 0.03},
                    ),
                    "side_qid": _choice_ans(
                        "fam_cancer",
                        {"fam_cancer": 0.93, "other": 0.04, "fam_lumps": 0.03},
                    ),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval_future
            ):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं, हाँ चाची को कैंसर था"),
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
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-fut")],
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
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

            auth_events = [
                e for e in db.get_call(call_id)["events"] if e["type"] == "decision_authority"
            ]
            self.assertEqual(
                auth_events[-1]["meta"]["future_qid_match"],
                {"qid": "fam_cancer", "confidence": 0.93, "act": "add_or_correct"},
            )

        asyncio.run(run())

    def test_authority_missed_tool_safety_net_drops_unprompted_audio_and_advances(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step_before = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = step_before
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            eval_calls = []

            async def fake_eval(request, **_kwargs):
                eval_calls.append(request)
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.05),
                    "current_outcome": _choice_ans(
                        "clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}
                    ),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"),
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
                self.assertFalse(turn.capture.authority_speech_allowed)

                # Unprompted bot audio + caption before turn_complete MUST be dropped
                await gemini_live._handle_server_message(
                    _server_message(bot_text="मैं अपने मन से बोल रही हूँ"),
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
                await gemini_live._handle_server_message(
                    _authority_audio_msg(turn_complete=False),
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
                self.assertEqual(telemetry["audio_chunks_out"], 0)
                self.assertEqual(telemetry["bot_turns"], 0)
                self.assertFalse(turn.bot_replied_since_user)
                self.assertFalse(any(m.get("type") == "audio" for m in ws.sent))
                self.assertFalse(
                    any(m.get("type") == "transcript" and m.get("role") == "assistant" for m in ws.sent)
                )

                # turn_complete arrives without next_question -> safety net runs Decision & cues real next question
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True),
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
                    authority_thresholds=thresholds,
                )

                self.assertEqual(len(eval_calls), 1)
                self.assertEqual(cursor.state["answers"]["birth_jaundice"]["state"], "clear")
                self.assertEqual(cursor.state["active_qid"], "birth_sudden")
                self.assertTrue(turn.capture.authority_speech_allowed)
                self.assertEqual(turn.capture.user_turn_buffer, "")
                self.assertEqual(len(turn.capture.user_turn_pcm), 0)

                qstates = [m for m in ws.sent if m.get("type") == "question_state"]
                self.assertEqual([q["qid"] for q in qstates], ["birth_sudden"])
                cues = [item[1]["text"] for item in session.sent if item[0] == "realtime"]
                self.assertEqual(len(cues), 1)
                self.assertIn("इमरजेंसी", cues[0])

                # Authorized bot audio for birth_sudden now plays normally
                await gemini_live._handle_server_message(
                    _authority_audio_msg(turn_complete=True),
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
                self.assertEqual(telemetry["audio_chunks_out"], 1)
                self.assertEqual(turn.spoken_qid, "birth_sudden")
                self.assertTrue(any(m.get("type") == "audio" for m in ws.sent))

        asyncio.run(run())

    def test_authority_pcm_buffering_cap_and_wav_fallback_when_asr_empty(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            silent_chunk = base64.b64encode(b"\x00\x00" * 1600).decode("ascii")  # 100ms silence
            speech_chunk = base64.b64encode(b"\x00\x10" * 1600).decode("ascii")  # rms ~ 0.125 >= 0.02

            # 1) Pre-speech silence is NOT buffered; speech + tail IS buffered and clears authority_speech_allowed
            ws_audio = _FakeSocket([
                {"type": "audio", "data": silent_chunk, "sample_rate": 16000},
                {"type": "audio", "data": speech_chunk, "sample_rate": 16000},
                {"type": "audio", "data": silent_chunk, "sample_rate": 16000},
            ])
            await gemini_live._browser_loop(
                session,
                asyncio.Lock(),
                ws_audio,
                asyncio.Lock(),
                asyncio.Event(),
                call_id,
                telemetry,
                turn,
                cursor,
                _Recorder(),
            )
            self.assertFalse(turn.capture.authority_speech_allowed)
            self.assertEqual(len(turn.capture.user_turn_pcm), 2 * 3200)  # speech chunk + 1 trailing chunk

            # 2) Empty ASR -> next_question sends audio/wav to Decision and clears user_turn_pcm
            seen_requests = []
            expected_pcm = bytes(turn.capture.user_turn_pcm)

            async def fake_eval(request, **_kwargs):
                seen_requests.append(request)
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.02),
                    "current_outcome": _choice_ans(
                        "clear", {"clear": 0.93, "other": 0.04, "concern": 0.03}
                    ),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                await gemini_live._answer_function_calls(
                    session,
                    asyncio.Lock(),
                    _FakeSocket(),
                    asyncio.Lock(),
                    call_id,
                    telemetry,
                    [_Call("next_question", {}, "fc-pcm")],
                    set(),
                    set(),
                    cursor,
                    turn,
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(len(seen_requests), 1)
            req_input = seen_requests[0]["input"]
            self.assertIsInstance(req_input, list)
            self.assertEqual(req_input[1]["type"], "audio")
            self.assertEqual(req_input[1]["mime_type"], "audio/wav")
            wav_raw = base64.b64decode(req_input[1]["data"])
            with wave.open(io.BytesIO(wav_raw), "rb") as wf:
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.readframes(wf.getnframes()), expected_pcm)
            self.assertEqual(cursor.state["answers"]["birth_jaundice"]["state"], "clear")
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")
            self.assertEqual(len(turn.capture.user_turn_pcm), 0)
            self.assertTrue(turn.capture.authority_speech_allowed)

            # 3) Empty text AND empty PCM -> held(empty_turn) and re-speaks without closing
            turn.capture.authority_handled_turn = False
            turn.capture.authority_last_key = None
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            turn.user_turn_step = turn.spoken_qid
            empty_session = _FakeSession()
            with mock.patch.object(decisions, "evaluate_turn") as eval_mock:
                await gemini_live._answer_function_calls(
                    empty_session,
                    asyncio.Lock(),
                    _FakeSocket(),
                    asyncio.Lock(),
                    call_id,
                    telemetry,
                    [_Call("next_question", {}, "fc-empty")],
                    set(),
                    set(),
                    cursor,
                    turn,
                    authority_thresholds=thresholds,
                )
                eval_mock.assert_not_called()
                await turn.speak_fallback
            tool_resp = [item[1] for item in empty_session.sent if item[0] == "tool"][0][0]
            self.assertEqual(tool_resp.response["result"], "held")
            self.assertEqual(tool_resp.response["reason"], "empty_turn")
            self.assertEqual(cursor.state["phase"], "asking")
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")
            self.assertEqual(_kinds(empty_session), ["tool", "realtime"])

            # 4) Cap test: > 960_000 bytes stays bounded at 960_000
            cap_turn = gemini_live._TurnState()
            cap_turn.capture.decision_mode = "authority"
            big_loud = base64.b64encode(b"\x00\x10" * 160_000).decode("ascii")  # 320_000 bytes each
            ws_cap = _FakeSocket([
                {"type": "audio", "data": big_loud, "sample_rate": 16000}
                for _ in range(4)  # 1_280_000 bytes > 960_000
            ])
            await gemini_live._browser_loop(
                _FakeSession(),
                asyncio.Lock(),
                ws_cap,
                asyncio.Lock(),
                asyncio.Event(),
                call_id,
                telemetry,
                cap_turn,
                cursor,
                _Recorder(),
            )
            self.assertEqual(len(cap_turn.capture.user_turn_pcm), 960_000)

        asyncio.run(run())

    def test_authority_emergency_vs_past_history(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            # 1) Historical past blue lips ("कभी नीली पड़ती हैं") -> records concern WITHOUT emergency close
            hist_state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "body_blue",
            )
            hist_cursor = gemini_live._Cursor()
            hist_cursor.state = hist_state
            hist_turn = gemini_live._TurnState()
            hist_turn.capture.decision_mode = "authority"
            hist_turn.capture.authority_thresholds = thresholds
            hist_turn.spoken_qid = gemini_live._current_step(hist_cursor.state)
            hist_session = _FakeSession()
            hist_tel = self._telemetry()
            hist_call = db.create_call("gemini-3.1-flash-live-preview")

            async def eval_hist(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.08),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.10),
                    "current_outcome": _choice_ans(
                        "concern", {"concern": 0.92, "other": 0.05, "clear": 0.03}
                    ),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "body_fits": 0.04}),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=eval_hist
            ):
                await gemini_live._handle_server_message(
                    _server_message(user_text="हाँ, रोते समय कभी नीली पड़ती हैं"),
                    hist_session,
                    asyncio.Lock(),
                    _FakeSocket(),
                    asyncio.Lock(),
                    hist_call,
                    hist_tel,
                    hist_turn,
                    _Recorder(),
                    set(),
                    set(),
                    hist_cursor,
                    stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-hist")],
                    ),
                    hist_session,
                    asyncio.Lock(),
                    _FakeSocket(),
                    asyncio.Lock(),
                    hist_call,
                    hist_tel,
                    hist_turn,
                    _Recorder(),
                    set(),
                    set(),
                    hist_cursor,
                    stop=asyncio.Event(),
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(hist_turn)

            self.assertEqual(hist_cursor.state["phase"], "asking")
            self.assertEqual(hist_cursor.state["answers"]["body_blue"]["state"], "concern")
            self.assertIn("कभी नीली पड़ती हैं", hist_cursor.state["answers"]["body_blue"]["notes"])
            self.assertEqual(hist_cursor.state["active_qid"], "preg_heart")

            # 2) Active danger_now -> closes immediately with emergency & 108 closing
            emg_cursor = gemini_live._Cursor()
            emg_cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
                "body_blue",
            )
            emg_turn = gemini_live._TurnState()
            emg_turn.capture.decision_mode = "authority"
            emg_turn.capture.authority_thresholds = thresholds
            emg_turn.spoken_qid = gemini_live._current_step(emg_cursor.state)
            emg_session = _FakeSession()
            emg_tel = self._telemetry()
            emg_call = db.create_call("gemini-3.1-flash-live-preview")
            ended_callbacks = []

            async def on_end(cid, snap):
                ended_callbacks.append((cid, snap["cursor_state"]["close_reason"]))

            async def eval_emg(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.96),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.85),
                    "current_outcome": _choice_ans(
                        "concern", {"concern": 0.90, "other": 0.06, "clear": 0.04}
                    ),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "body_fits": 0.04}),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=eval_emg
            ):
                await gemini_live._handle_server_message(
                    _server_message(user_text="बच्चे को अभी साँस नहीं आ रही और होंठ नीले हैं"),
                    emg_session,
                    asyncio.Lock(),
                    _FakeSocket(),
                    asyncio.Lock(),
                    emg_call,
                    emg_tel,
                    emg_turn,
                    _Recorder(),
                    set(),
                    set(),
                    emg_cursor,
                    on_end_accepted=on_end,
                    stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-emg")],
                    ),
                    emg_session,
                    asyncio.Lock(),
                    _FakeSocket(),
                    asyncio.Lock(),
                    emg_call,
                    emg_tel,
                    emg_turn,
                    _Recorder(),
                    set(),
                    set(),
                    emg_cursor,
                    on_end_accepted=on_end,
                    stop=asyncio.Event(),
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(emg_turn)

            self.assertEqual(emg_cursor.state["phase"], "closing")
            self.assertEqual(emg_cursor.state["close_reason"], "emergency")
            tool_resp = [item[1] for item in emg_session.sent if item[0] == "tool"][0][0]
            self.assertTrue(tool_resp.response["stop"])
            self.assertIn("108", tool_resp.response["say"])
            self.assertEqual(ended_callbacks, [(emg_call, "emergency")])
            cues = [item[1]["text"] for item in emg_session.sent if item[0] == "realtime"]
            self.assertIn("108", cues[0])

        asyncio.run(run())

    def test_authority_decision_failure_closes_calmly_without_108_or_guessing(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
                "birth_nicu",
            )
            state = apply_turn(state, "concern", note="सांस की दिक्कत", followup_answered=True)
            state = apply_turn(state, "concern", note="तीन दिन")
            self.assertEqual(state["active_qid"], "birth_jaundice")

            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")
            db.append_transcript(call_id, "assistant", "क्या जन्म के बाद पीलिया के लिए नीली लाइट में रखना पड़ा था?", qid="birth_jaundice")

            async def fail_eval(request, **_kwargs):
                return _make_decision_result(request, {}, ok=False, error_reason="timeout", attempts=2)

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fail_eval
            ):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"),
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
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-fail")],
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
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(cursor.state["phase"], "closing")
            self.assertEqual(cursor.state["close_reason"], "technical_failure")
            self.assertNotIn("birth_jaundice", cursor.state["answers"])
            self.assertEqual(cursor.state["answers"]["birth_nicu"]["state"], "concern")
            self.assertEqual(telemetry["dropoff_qid"], "birth_jaundice")
            self.assertTrue(turn.capture.authority_speech_allowed)

            tool_resp = [item[1] for item in session.sent if item[0] == "tool"][0][0]
            self.assertTrue(tool_resp.response["stop"])
            self.assertIn("अधूरी", tool_resp.response["say"])
            self.assertIn("पूजा जी", tool_resp.response["say"])
            self.assertNotIn("108", tool_resp.response["say"])
            self.assertNotIn("अस्पताल", tool_resp.response["say"])

            cues = [item[1]["text"] for item in session.sent if item[0] == "realtime"]
            self.assertEqual(len(cues), 1)
            self.assertIn("अधूरी", cues[0])
            self.assertNotIn("108", cues[0])

            saved_call = db.get_call(call_id)
            self.assertEqual(saved_call["cursor_state"]["close_reason"], "technical_failure")
            self.assertNotIn("birth_jaundice", saved_call["cursor_state"]["answers"])
            self.assertGreaterEqual(len(saved_call["transcript"]), 2)
            auth_ev = next(
                e for e in saved_call["events"] if e["type"] == "decision_authority"
            )
            self.assertFalse(auth_ev["meta"]["ok"])
            self.assertEqual(auth_ev["meta"]["error_reason"], "timeout")
            self.assertEqual(auth_ev["meta"]["attempts"], 2)

        asyncio.run(run())

    def test_authority_quick_reply_is_model_free_and_holds_late_next_question(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            step = gemini_live._current_step(state)
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = step
            turn.user_turn_step = step
            turn.capture.user_turn_buffer = "नह"
            turn.capture.user_turn_pcm = bytearray(b"\x00\x10" * 800)
            turn.capture.authority_speech_allowed = False
            session = _FakeSession()
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            with mock.patch.object(
                decisions, "evaluate_turn"
            ) as eval_mock:
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
                eval_mock.assert_not_called()
                self.assertEqual(cursor.state["answers"]["birth_nicu"]["state"], "clear")
                self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
                self.assertEqual(turn.capture.user_turn_buffer, "")
                self.assertEqual(len(turn.capture.user_turn_pcm), 0)
                self.assertTrue(turn.capture.authority_speech_allowed)
                self.assertEqual(_kinds(session), ["realtime"])

                # Trailing ASR from before tap -> stays stale, does NOT revoke authority_speech_allowed
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"),
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
                self.assertTrue(turn.capture.authority_speech_allowed)
                self.assertEqual(turn.user_turn_step, step)

                # Late next_question() held as stale_turn without calling Decision or repeating cue
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-late-auth")],
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
                    authority_thresholds=thresholds,
                )
                eval_mock.assert_not_called()

            tool_resp = [item[1] for item in session.sent if item[0] == "tool"][0][0]
            self.assertEqual(tool_resp.response["result"], "held")
            self.assertEqual(tool_resp.response["reason"], "stale_turn")
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            cues = [item for item in session.sent if item[0] == "realtime"]
            self.assertEqual(len(cues), 1)
            details = [
                item["last_event"]["detail"]
                for item in ws.sent
                if item.get("type") == "telemetry"
            ]
            self.assertIn("quick_reply_clear", details)
            self.assertIn("next_question_held", details)

        asyncio.run(run())

    def test_authority_handle_gemini_ws_wires_config_thresholds_and_turn_end_to_end(self):
        thresholds = _calibrated_test_thresholds()

        class _WsSession:
            def __init__(self, msgs):
                self.setup_complete = True
                self.sent = []
                self._msgs = list(msgs)

            async def send_realtime_input(self, **kwargs):
                self.sent.append(("realtime", kwargs))

            async def send_tool_response(self, function_responses):
                self.sent.append(("tool", function_responses))

            async def receive(self):
                for item in self._msgs:
                    await asyncio.sleep(0.01)
                    yield item
                await asyncio.Future()

        class _WsCtx:
            def __init__(self, sess):
                self._sess = sess

            async def __aenter__(self):
                return self._sess

            async def __aexit__(self, *_args):
                return False

        class _LiveWs(_FakeSocket):
            def __init__(self, queue_msgs):
                super().__init__(queue_msgs)
                self._q: asyncio.Queue = asyncio.Queue()
                for m in queue_msgs:
                    self._q.put_nowait(m)

            async def receive_text(self):
                item = await self._q.get()
                if item is None:
                    raise gemini_live.WebSocketDisconnect()
                return json.dumps(item, ensure_ascii=False)

        async def run():
            seen_configs = []
            live_sess = _WsSession([
                _authority_audio_msg(turn_complete=True),  # Greeting spoken -> spoken_qid="intro_name"
                _server_message(user_text="रेखा"),
                _server_message(
                    turn_complete=True,
                    tool_calls=[_Call("next_question", {}, "fc-ws-1")],
                ),
                _authority_audio_msg(turn_complete=True),
            ])

            async def fake_connect(client, model, config):
                seen_configs.append(config)
                return _WsCtx(live_sess), live_sess, 5

            ws = _LiveWs([])

            async def fake_eval(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.90),
                    "current_outcome": _choice_ans("other", {"other": 0.90, "clear": 0.10}),
                }
                return _make_decision_result(request, raw)

            async def hook_profile(_state, _utt, _wav):
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates={"caller_name": "रेखा"}, error_reason="", extractor_ms=1
                )

            async def hangup_soon():
                await asyncio.sleep(0.15)
                await ws._q.put({"type": "hangup"})

            with mock.patch.dict(
                os.environ, {"DECISION_MODE": "authority", "GEMINI_API_KEY": "test-key"}
            ), mock.patch.object(
                gemini_live, "_connect_live", side_effect=fake_connect
            ), mock.patch.object(
                live_decision_bridge, "extract_profile_updates", side_effect=hook_profile
            ), mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                task = asyncio.create_task(hangup_soon())
                await gemini_live.handle_gemini_ws(ws, authority_thresholds=thresholds)
                await task

            self.assertEqual(len(seen_configs), 1)
            decls = seen_configs[0].tools[0].function_declarations
            self.assertEqual([d.name for d in decls], ["next_question"])
            started = next(m for m in ws.sent if m.get("type") == "session_started")
            saved = db.get_call(started["call_id"])
            self.assertEqual(saved["cursor_state"]["profile"]["caller_name"], "रेखा")
            self.assertEqual(saved["cursor_state"]["active_qid"], "intro_married")
            auth_ev = [e for e in saved["events"] if e["type"] == "decision_authority"]
            self.assertEqual(len(auth_ev), 1)

        asyncio.run(run())

    def test_authority_midflight_buffer_not_wiped_after_evaluate_turn_await(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            turn.user_turn_step = turn.spoken_qid
            turn.capture.user_turn_buffer = "नहीं"
            turn.capture.user_turn_pcm = bytearray(b"\x00\x10" * 400)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            eval_started = asyncio.Event()
            release_eval = asyncio.Event()

            async def slow_eval(request, **_kwargs):
                eval_started.set()
                await release_eval.wait()
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.01),
                    "detail_present": _noul_ans(0.05),
                    "current_outcome": _choice_ans(
                        "clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}
                    ),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=slow_eval):
                turn_task = asyncio.create_task(
                    gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("next_question", {}, "fc-race")],
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
                        authority_thresholds=thresholds,
                    )
                )
                await asyncio.wait_for(eval_started.wait(), timeout=1.0)
                # Buffer was consumed and cleared BEFORE await
                self.assertEqual(turn.capture.user_turn_buffer, "")
                self.assertEqual(len(turn.capture.user_turn_pcm), 0)

                # New user speech arrives mid-flight while evaluate_turn is awaiting
                turn.capture.user_turn_buffer = "अगले सवाल का जवाब"
                turn.capture.user_turn_pcm = bytearray(b"\x01\x00" * 320)
                release_eval.set()
                await turn_task
                await _drop_fallback(turn)

            self.assertEqual(cursor.state["answers"]["birth_jaundice"]["state"], "clear")
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")
            # Mid-flight buffer MUST remain intact after evaluate_turn + apply_interpreted_turn
            self.assertEqual(turn.capture.user_turn_buffer, "अगले सवाल का जवाब")
            self.assertEqual(bytes(turn.capture.user_turn_pcm), b"\x01\x00" * 320)

        asyncio.run(run())

    def test_authority_full_intro_flow_name_married_three_children_bridge_to_birth(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            cursor = gemini_live._Cursor()
            cursor.state = call_cursor.new_cursor_state()
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            async def fake_eval(request, **_kwargs):
                active_q = json.loads(request["input"])["active_question"]
                is_bridge = "शुरू करें" in active_q
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.02 if is_bridge else 0.92),
                    "current_outcome": _choice_ans(
                        "concern" if is_bridge else "other",
                        {"concern": 0.93, "other": 0.05, "clear": 0.02}
                        if is_bridge
                        else {"other": 0.90, "clear": 0.06, "concern": 0.04},
                    ),
                }
                return _make_decision_result(request, raw)

            async def fake_extract(state, utterance_text="", audio_wav_bytes=None, **_kwargs):
                qid = state.get("active_qid")
                if qid == "intro_name":
                    updates = {"caller_name": "पूजा"}
                elif qid == "intro_married":
                    updates = {"marital_status": "married"}
                elif qid == "intro_children":
                    updates = {
                        "child_count": 3,
                        "children": [
                            {"age_years": 11.0, "sex": "female", "name": "रिया"},
                            {"age_years": 6.0, "sex": "male", "name": None},
                            {"age_years": 0.5, "sex": "male", "name": None},
                        ],
                    }
                else:
                    updates = {}
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates=updates, error_reason="", extractor_ms=24
                )

            turns = [
                ("मेरा नाम पूजा है", "fc-intro-1"),
                ("हाँ, शादी हो गई है", "fc-intro-2"),
                ("तीन बच्चे हैं — 11 साल की बड़ी बेटी रिया, 6 साल का बेटा, और 6 महीने का छोटा बेटा", "fc-intro-3"),
                ("हाँ, शुरू करें", "fc-intro-4"),
            ]
            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fake_extract
            ) as ext_mock:
                for user_text, call_sid in turns:
                    turn.spoken_qid = gemini_live._current_step(cursor.state)
                    await gemini_live._handle_server_message(
                        _server_message(user_text=user_text),
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
                    await gemini_live._handle_server_message(
                        _server_message(
                            turn_complete=True,
                            tool_calls=[_Call("next_question", {}, call_sid)],
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
                        authority_thresholds=thresholds,
                    )
                    await _drop_fallback(turn)

                self.assertEqual(ext_mock.await_count, 3)

            self.assertEqual(cursor.state["phase"], "asking")
            self.assertEqual(cursor.state["active_qid"], "birth_mode_detail")
            self.assertEqual(cursor.state["profile"]["caller_name"], "पूजा")
            self.assertEqual(cursor.state["profile"]["marital_status"], "married")
            self.assertEqual(len(cursor.state["profile"]["children"]), 3)
            self.assertEqual(cursor.state["profile"]["children"][2]["age_label_hi"], "6 महीने")
            self.assertEqual(cursor.state["answers"]["intro_bridge"]["state"], "clear")

            tool_resps = [item[1][0] for item in session.sent if item[0] == "tool"]
            self.assertEqual(tool_resps[-1].response["active_qid"], "birth_mode_detail")
            self.assertEqual(len(tool_resps[-1].response["children"]), 3)

            auth_events = [
                e for e in db.get_call(call_id)["events"] if e["type"] == "decision_authority"
            ]
            self.assertEqual(len(auth_events), 4)
            self.assertTrue(auth_events[0]["meta"]["extractor_ok"])
            self.assertEqual(auth_events[0]["meta"]["extractor_ms"], 24)
            self.assertEqual(auth_events[0]["meta"]["extractor_error"], "")
            self.assertNotIn("extractor_ok", auth_events[3]["meta"])

        asyncio.run(run())

    def test_authority_extractor_first_failure_reasks_second_closes_technical_failure(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            cursor = gemini_live._Cursor()
            cursor.state = call_cursor.new_cursor_state()
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            async def fake_eval(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.92),
                    "current_outcome": _choice_ans(
                        "other", {"other": 0.91, "clear": 0.09}
                    ),
                }
                return _make_decision_result(request, raw)

            async def fail_extract(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(
                    ok=False, updates={}, error_reason="timeout", extractor_ms=2501
                )

            with mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fail_extract
            ):
                # 1st extractor failure on intro_name -> keeps cursor on intro_name and re-asks
                turn.spoken_qid = gemini_live._current_step(cursor.state)
                await gemini_live._handle_server_message(
                    _server_message(user_text="मेरा नाम रेखा है"),
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
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-ext-1")],
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
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

                self.assertEqual(cursor.state["phase"], "asking")
                self.assertEqual(cursor.state["active_qid"], "intro_name")
                self.assertIsNone(cursor.state["profile"]["caller_name"])
                self.assertEqual(turn.capture.extractor_failures, 1)

                # 2nd consecutive failure on intro_name -> closes with technical_failure without 108
                turn.spoken_qid = gemini_live._current_step(cursor.state)
                await gemini_live._handle_server_message(
                    _server_message(user_text="रेखा"),
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
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-ext-2")],
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
                    authority_thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(cursor.state["phase"], "closing")
            self.assertEqual(cursor.state["close_reason"], "technical_failure")
            self.assertIsNone(cursor.state["profile"]["caller_name"])
            tool_resps = [item[1][0] for item in session.sent if item[0] == "tool"]
            self.assertFalse(tool_resps[0].response.get("stop", False))
            self.assertTrue(tool_resps[1].response["stop"])
            self.assertIn("अधूरी", tool_resps[1].response["say"])
            self.assertNotIn("108", tool_resps[1].response["say"])
            auth_events = [
                e for e in db.get_call(call_id)["events"] if e["type"] == "decision_authority"
            ]
            self.assertEqual(len(auth_events), 2)
            self.assertFalse(auth_events[0]["meta"]["extractor_ok"])
            self.assertEqual(auth_events[0]["meta"]["extractor_error"], "timeout")

        asyncio.run(run())

    def test_authority_ever_pregnant_sync_and_clinical_gate_first_failure(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            # 1) ever_pregnant active step: zero-extractor behavior (Decision clear resolves cleanly without calling extractor)
            ev_state = cursor_tests._clear_until(
                cursor_tests._profile(children=[]),
                "ever_pregnant",
            )
            cursor = gemini_live._Cursor()
            cursor.state = ev_state
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            turn.spoken_qid = gemini_live._current_step(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            async def eval_clear(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.05),
                    "current_outcome": _choice_ans("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "fam_cancer": 0.04}),
                }
                return _make_decision_result(request, raw)

            async def empty_ext(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates={}, error_reason="", extractor_ms=10
                )

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_clear), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=empty_ext
            ) as ext_mock:
                turn.capture.user_turn_buffer = "नहीं"
                body, _clar, _kind = await live_decision_bridge.run_authority_turn(
                    cursor,
                    turn.capture,
                    call_id,
                    telemetry,
                    tool_body_fn=gemini_live._tool_body,
                    mark_closing_fn=lambda _r: asyncio.sleep(0),
                    authority_thresholds=thresholds,
                )
                turn.capture.authority_last_key = None
                ext_mock.assert_not_awaited()

            self.assertEqual(body["result"], "ok")
            self.assertFalse(cursor.state["profile"]["ever_pregnant"])
            self.assertEqual(cursor.state["answers"]["ever_pregnant"]["state"], "clear")
            ev_auth = [
                e for e in db.get_call(call_id)["events"] if e["type"] == "decision_authority"
            ]
            self.assertEqual(len(ev_auth), 1)
            self.assertNotIn("extractor_ok", ev_auth[0]["meta"])

            # 2) Disagreement between profile_updates["ever_pregnant"] and side_qid="ever_pregnant" drops both and re-asks
            disagree_state = cursor_tests._clear_until(
                cursor_tests._profile(children=[]),
                "preg_thalassemia",
            )
            cursor.state = disagree_state
            turn.capture.extractor_failures = 0
            turn.capture.extractor_failure_step = ""

            async def eval_side_ep(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.88),
                    "current_outcome": _choice_ans("clear", {"clear": 0.90, "other": 0.06, "concern": 0.04}),
                    "side_intent": _choice_ans("clear_or_retract", {"clear_or_retract": 0.91, "other": 0.05, "clarify_other": 0.04}),
                    "side_qid": _choice_ans("ever_pregnant", {"ever_pregnant": 0.90, "other": 0.06, "fam_cancer": 0.04}),
                }
                return _make_decision_result(request, raw)

            async def ep_true_ext(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates={"ever_pregnant": True}, error_reason="", extractor_ms=11
                )

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_side_ep), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=ep_true_ext
            ):
                turn.capture.user_turn_buffer = "अस्पष्ट बात"
                await live_decision_bridge.run_authority_turn(
                    cursor,
                    turn.capture,
                    call_id,
                    telemetry,
                    tool_body_fn=gemini_live._tool_body,
                    mark_closing_fn=lambda _r: asyncio.sleep(0),
                    authority_thresholds=thresholds,
                )
                turn.capture.authority_last_key = None

            self.assertFalse(cursor.state["profile"]["ever_pregnant"])
            self.assertEqual(cursor.state["active_qid"], "preg_thalassemia")

            # 3) Clinical gate 1st extractor failure applies clinical payload and resets counter on step advance
            gate_state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            cursor.state = gate_state

            async def eval_gate_profile(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.85),
                    "detail_present": _noul_ans(0.05),
                    "current_outcome": _choice_ans("clear", {"clear": 0.93, "other": 0.04, "concern": 0.03}),
                    "side_intent": _choice_ans("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice_ans("other", {"other": 0.96, "birth_jaundice": 0.04}),
                }
                return _make_decision_result(request, raw)

            async def fail_ext(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(
                    ok=False, updates={}, error_reason="invalid_schema", extractor_ms=15
                )

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_gate_profile), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fail_ext
            ):
                turn.capture.user_turn_buffer = "नहीं"
                await live_decision_bridge.run_authority_turn(
                    cursor,
                    turn.capture,
                    call_id,
                    telemetry,
                    tool_body_fn=gemini_live._tool_body,
                    mark_closing_fn=lambda _r: asyncio.sleep(0),
                    authority_thresholds=thresholds,
                )
                turn.capture.authority_last_key = None

            self.assertEqual(cursor.state["phase"], "asking")
            self.assertEqual(cursor.state["answers"]["birth_nicu"]["state"], "clear")
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            self.assertEqual(turn.capture.extractor_failures, 0)

        asyncio.run(run())

    def test_authority_intro_clarify_skips_extractor_and_empty_updates_use_native_hold(self):
        thresholds = _calibrated_test_thresholds()

        async def run():
            # 1) Clarify on intro_name must NOT call profile_extractor or set name_fragment hold
            cursor = gemini_live._Cursor()
            cursor.state = call_cursor.new_cursor_state()
            turn = gemini_live._TurnState()
            turn.capture.decision_mode = "authority"
            turn.capture.authority_thresholds = thresholds
            telemetry = self._telemetry()
            call_id = db.create_call("gemini-3.1-flash-live-preview")

            async def eval_clarify(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.05),
                    "current_outcome": _choice_ans("clarify", {"clarify": 0.92, "other": 0.05, "clear": 0.03}),
                }
                return _make_decision_result(request, raw)

            async def empty_ext(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates={}, error_reason="", extractor_ms=1
                )

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_clarify), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=empty_ext
            ) as ext_mock:
                turn.capture.user_turn_buffer = "क्या कहा आपने?"
                body, clarify_applied, _kind = await live_decision_bridge.run_authority_turn(
                    cursor,
                    turn.capture,
                    call_id,
                    telemetry,
                    tool_body_fn=gemini_live._tool_body,
                    mark_closing_fn=lambda _r: asyncio.sleep(0),
                    authority_thresholds=thresholds,
                )
                turn.capture.authority_last_key = None
                ext_mock.assert_not_awaited()

            self.assertTrue(clarify_applied)
            self.assertTrue(body.get("clarify"))
            self.assertEqual(cursor.state["active_qid"], "intro_name")
            self.assertIsNone(cursor.state["hold_reason"])

            # 2) Empty updates on intro_married when not clarify -> uniform apply_profile hold ("marital_status")
            cursor.state = call_cursor.apply_profile(call_cursor.new_cursor_state(), caller_name="रेखा")[0]
            self.assertEqual(cursor.state["active_qid"], "intro_married")

            async def eval_not_about(request, **_kwargs):
                raw = {
                    "danger_now": _noul_ans(0.01),
                    "caller_stop": _noul_ans(0.01),
                    "mentions_profile": _noul_ans(0.10),
                    "current_outcome": _choice_ans("other", {"other": 0.90, "clarify": 0.05, "clear": 0.05}),
                }
                return _make_decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_not_about), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=empty_ext
            ) as ext_mock:
                turn.capture.user_turn_buffer = "हाँ जी"
                body2, clarify_applied2, _kind2 = await live_decision_bridge.run_authority_turn(
                    cursor,
                    turn.capture,
                    call_id,
                    telemetry,
                    tool_body_fn=gemini_live._tool_body,
                    mark_closing_fn=lambda _r: asyncio.sleep(0),
                    authority_thresholds=thresholds,
                )
                ext_mock.assert_awaited_once()

            self.assertFalse(clarify_applied2)
            self.assertEqual(body2["active_qid"], "intro_married")
            self.assertEqual(cursor.state["hold_reason"], "marital_status")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
