"""Stage 5: live-call dead air, child scope, clarify, close notes, PDF fit."""

import asyncio
import io
import os
import re
import tempfile
import unittest
from urllib.parse import unquote

_TMP = tempfile.mkdtemp(prefix="shishu-v3-stage5-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import app  # noqa: E402
import call_cursor  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import mother_report  # noqa: E402
import post_call  # noqa: E402
import prompts  # noqa: E402
import test_v4_cursor as cursor_tests  # noqa: E402
import test_v3_reports as report_tests  # noqa: E402
from call_cursor import apply_turn, prompt_directive  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from triage_catalog import FOLLOWUP_HI, active_specs, get_spec  # noqa: E402

_BACHCHA_LATER = re.compile(r"बच्चा(?!े|ों)")


def _collision(text: str) -> bool:
    start = 0
    while True:
        at = text.find("बच्चे", start)
        if at < 0:
            return False
        if _BACHCHA_LATER.search(text[at + len("बच्चे"):]):
            return True
        start = at + len("बच्चे")


class _FakeSocket:
    def __init__(self):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _FakeSession:
    def __init__(self):
        self.sent = []

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses))

    async def send_realtime_input(self, **kwargs):
        self.sent.append(("realtime", kwargs))


class _Call:
    def __init__(self, name, args, fid):
        self.id = fid
        self.name = name
        self.args = args
        self.will_continue = False


class ScopeTests(unittest.TestCase):
    def test_every_child_question_is_grammatical_for_all_children(self):
        kids = [cursor_tests._child(6, "male", 1), cursor_tests._child(2, "female", 2)]
        state = cursor_tests._profile(children=kids)
        seen = 0
        for spec in active_specs():
            if not spec.question_multi_hi:
                continue
            seen += 1
            text = call_cursor._question_hi(state, spec)
            self.assertEqual(text, spec.question_multi_hi, spec.id)
            self.assertNotIn("—", text, spec.id)
            self.assertFalse(text.startswith("आपके किसी भी बच्चे —"), spec.id)
            self.assertFalse(_collision(text), f"{spec.id}: {text}")
        self.assertEqual(seen, 36)
        term = call_cursor._question_hi(state, get_spec("birth_term"))
        sudden = call_cursor._question_hi(state, get_spec("birth_sudden"))
        bleed = call_cursor._question_hi(state, get_spec("body_bleeding_bruise"))
        self.assertIn("आपके किसी भी बच्चे का जन्म", term)
        self.assertIn("आपका कोई भी बच्चा", sudden)
        self.assertIn("आपके किसी भी बच्चे को मामूली चोट", bleed)
        fasting = call_cursor._question_hi(state, get_spec("ill_fasting"))
        self.assertIn("आपका कोई भी बच्चा", fasting)
        self.assertIn("सुस्त", fasting)


class CatalogTests(unittest.TestCase):
    def test_atomic_gates_and_bridge_offer_one_question_at_a_time(self):
        gates = [spec.id for spec in active_specs() if spec.kind == "gate"]
        self.assertEqual(len(gates), 56)
        self.assertIn("fam_maternal_bleeding", gates)
        self.assertIn("fam_maternal_infections", gates)
        self.assertIn("fam_maternal_eczema", gates)
        self.assertNotIn("fam_maternal_side", gates)
        self.assertIn("birth_nicu", gates)
        self.assertIn("body_bleeding_bruise", gates)
        self.assertIn("fam_infant_death", gates)
        bridge = get_spec("intro_bridge").question_hi
        self.assertIn("आसान सवाल", bridge)
        self.assertIn("हाँ", bridge)
        self.assertIn("नहीं", bridge)
        self.assertIn("एक-एक", bridge)

    def test_module_followups_ask_the_detail_before_moving_on(self):
        expected = {
            "birth_nicu": ("nicu_reason", "nicu_days"),
            "birth_term": ("term_how_early",),
            "ill_admissions": ("admissions_count", "admissions_reason"),
            "body_transfusion": ("transfusion_count",),
            "fam_miscarriage": ("miscarriage_count", "miscarriage_month"),
            "fam_infant_death": ("infant_death_age", "infant_death_cause"),
        }
        for qid, follow in expected.items():
            self.assertEqual(get_spec(qid).followups, follow, qid)
            for fid in follow:
                self.assertTrue(FOLLOWUP_HI[fid].strip(), fid)
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "birth_nicu")
        state = apply_turn(state, "concern", note="सांस की वजह से")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertIn("किस वजह से", prompt_directive(state))
        state = apply_turn(state, "concern", note="सांस")
        self.assertEqual(state["pending_followup_idx"], 1)
        self.assertIn("कितने दिन", prompt_directive(state))
        self.assertEqual(
            state["answers"]["birth_nicu"]["followups"]["nicu_reason"]["note"],
            "सांस",
        )
        state = apply_turn(state, "concern", note="तीन दिन")
        self.assertIsNone(state["pending_followup_idx"])
        self.assertEqual(
            state["answers"]["birth_nicu"]["followups"]["nicu_days"]["note"],
            "तीन दिन",
        )
        self.assertNotEqual(state["active_qid"], "birth_nicu")


class CloseTests(unittest.TestCase):
    def test_concern_keeps_notes_until_she_has_nothing_else(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "close_anything_else")
        state = apply_turn(state, "concern", note="पेट में दर्द")
        self.assertEqual(state["phase"], "asking")
        self.assertEqual(state["active_qid"], "close_anything_else")
        self.assertEqual(state["answers"]["close_anything_else"]["notes"], "पेट में दर्द")
        self.assertIn("मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?", prompt_directive(state))
        state = apply_turn(state, "concern", note="नींद कम है")
        self.assertEqual(
            state["answers"]["close_anything_else"]["notes"],
            "पेट में दर्द; नींद कम है",
        )
        self.assertEqual(state["phase"], "asking")
        state = apply_turn(state, "clear")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "complete")
        self.assertEqual(state["answers"]["close_anything_else"]["state"], "concern")
        self.assertIn("नींद कम है", state["answers"]["close_anything_else"]["notes"])

        fresh = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        closed = apply_turn(fresh, "unknown")
        self.assertEqual(closed["phase"], "closing")
        self.assertEqual(closed["answers"]["close_anything_else"]["state"], "unknown")


class ClarifyAndDeadAirTests(unittest.TestCase):
    def test_prompt_asks_for_one_simpler_sentence_before_the_question(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("outcome clarify", text)
        self.assertIn("one short simpler Hindi sentence", text)
        self.assertIn("Do not repeat the question with the same words", text)
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "birth_nicu")
        cue = gemini_live._speak_cue_text(state, "clarify")
        self.assertIn("one short simpler Hindi sentence", cue)
        self.assertIn("Do not repeat it with the same words", cue)
        self.assertIn("काँच के बक्से", cue)

    def test_unheard_turn_is_held_and_the_model_is_prompted_immediately(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "birth_cord_3weeks")
        turn = gemini_live._TurnState()
        turn.spoken_qid = "birth_nicu"
        self.assertFalse(gemini_live._question_already_spoken(turn, state))
        cursor = gemini_live._Cursor()
        cursor.state = state

        async def run():
            call_id = db.create_call("gemini")
            session = _FakeSession()
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                {"tool_calls": 0, "tool_errors": 0},
                [_Call("record_turn", {"outcome": "clear", "note": ""}, "fc-1")],
                set(),
                set(),
                cursor,
                turn,
            )
            self.assertEqual([item[0] for item in session.sent], ["tool"])
            body = session.sent[0][1][0].response
            self.assertNotIn("wait", body)
            self.assertFalse(turn.bot_replied_since_user)
            self.assertEqual(gemini_live._TOOL_SPEAK_FALLBACK_SEC, 2.5)
            await turn.speak_fallback
            return session

        session = asyncio.run(run())
        self.assertEqual(cursor.state["active_qid"], "birth_cord_3weeks")
        self.assertEqual(session.sent[0][1][0].response["result"], "held")
        self.assertEqual(session.sent[0][1][0].response["reason"], "not_spoken")
        self.assertEqual([item[0] for item in session.sent], ["tool", "realtime"])
        self.assertIn("नाल", session.sent[1][1]["text"])
        self.assertIn("speak only this Hindi now", session.sent[1][1]["text"])

    def test_spoken_answer_advances_and_prompts_the_next_line_without_waiting(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "body_private_parts")
        turn = gemini_live._TurnState()
        turn.spoken_qid = gemini_live._current_step(state)
        self.assertTrue(gemini_live._question_already_spoken(turn, state))
        cursor = gemini_live._Cursor()
        cursor.state = state

        async def run():
            call_id = db.create_call("gemini")
            session = _FakeSession()
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                {"tool_calls": 0, "tool_errors": 0, "end_requested": False},
                [
                    _Call("record_turn", {"outcome": "clear"}, "fc-a"),
                    _Call("record_turn", {"outcome": "clear"}, "fc-b"),
                ],
                set(),
                set(),
                cursor,
                turn,
            )
            self.assertEqual([item[0] for item in session.sent], ["tool"])
            for response in session.sent[0][1]:
                self.assertNotIn("wait", response.response)
            self.assertFalse(turn.bot_replied_since_user)
            await turn.speak_fallback
            return session

        session = asyncio.run(run())
        self.assertEqual(cursor.state["active_qid"], "ill_fasting")
        self.assertEqual(session.sent[0][1][0].response["result"], "ok")
        self.assertEqual(session.sent[0][1][1].response["result"], "held")
        self.assertIn("सुस्त", session.sent[1][1]["text"])
        self.assertEqual([item[0] for item in session.sent], ["tool", "realtime"])

    def test_clarify_sets_the_flag_and_does_not_advance(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "birth_nicu")
        turn = gemini_live._TurnState()
        turn.spoken_qid = gemini_live._current_step(state)
        cursor = gemini_live._Cursor()
        cursor.state = state

        async def run():
            call_id = db.create_call("gemini")
            session = _FakeSession()
            await gemini_live._answer_function_calls(
                session,
                asyncio.Lock(),
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                {"tool_calls": 0, "tool_errors": 0},
                [_Call("record_turn", {"outcome": "clarify"}, "fc-c")],
                set(),
                set(),
                cursor,
                turn,
            )
            self.assertEqual([item[0] for item in session.sent], ["tool"])
            body = session.sent[0][1][0].response
            self.assertNotIn("wait", body)
            self.assertFalse(turn.bot_replied_since_user)
            await turn.speak_fallback
            return session

        session = asyncio.run(run())
        self.assertEqual(cursor.state["active_qid"], "birth_nicu")
        body = session.sent[0][1][0].response
        self.assertEqual(body["result"], "ok")
        self.assertTrue(body["clarify"])
        self.assertIn("one short simpler Hindi sentence", session.sent[1][1]["text"])
        self.assertIn("काँच के बक्से", session.sent[1][1]["text"])

    def test_speech_end_does_not_rearm_after_the_question_was_spoken(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        state = cursor_tests._clear_until(state, "birth_nicu")
        turn = gemini_live._TurnState()
        turn.spoken_qid = gemini_live._current_step(state)
        self.assertTrue(gemini_live._question_already_spoken(turn, state))
        advanced = apply_turn(state, "clear")
        self.assertFalse(gemini_live._question_already_spoken(turn, advanced))
        state = cursor_tests._clear_until(state, "close_anything_else")
        turn.spoken_qid = gemini_live._current_step(state)
        noted = apply_turn(state, "concern", note="पेट दर्द")
        self.assertNotEqual(
            gemini_live._current_step(state),
            gemini_live._current_step(noted),
        )
        self.assertFalse(gemini_live._question_already_spoken(turn, noted))

    def test_first_bot_audio_records_the_spoken_step(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        cursor = gemini_live._Cursor()
        cursor.state = state
        turn = gemini_live._TurnState()
        turn.awaiting_bot_reply = True
        turn.last_user_turn_ts = 1
        telemetry = {
            "audio_chunks_out": 0,
            "bot_turns": 0,
            "turn_latencies_ms": [],
            "end_requested": False,
        }

        class _Recorder:
            def add_bot_pcm(self, pcm, rate):
                return None

        async def run():
            call_id = db.create_call("gemini")
            await gemini_live._on_bot_audio(
                _FakeSocket(),
                asyncio.Lock(),
                call_id,
                telemetry,
                turn,
                _Recorder(),
                b"\x00\x00" * 24,
                24000,
                cursor,
            )

        asyncio.run(run())
        self.assertEqual(turn.spoken_qid, gemini_live._current_step(cursor.state))
        self.assertEqual(turn.spoken_qid, "intro_bridge")
        self.assertEqual(telemetry["bot_turns"], 1)


class ReportAndLoginTests(unittest.TestCase):
    def test_completed_one_child_call_fits_on_two_pages(self):
        state = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        guard = 0
        while state["phase"] == "asking" and guard < 90:
            state = apply_turn(state, "clear")
            guard += 1
        self.assertEqual(state["phase"], "closing")
        call = report_tests._call_for(state, 11)
        synthesis = post_call.deterministic_fallback(call)
        pdf = mother_report.render_pdf(mother_report.build_v3(call, synthesis))
        pages = len(PdfReader(io.BytesIO(pdf)).pages)
        self.assertGreaterEqual(pages, 2)
        self.assertLessEqual(pages, 3)

    def test_login_redirect_keeps_the_query(self):
        client = TestClient(app.app)
        home = client.get("/?clinic=kem", follow_redirects=False)
        self.assertEqual(home.status_code, 302)
        self.assertEqual(unquote(home.headers["location"]), "/login?next=/?clinic=kem")
        admin = client.get("/admin?clinic=kem", follow_redirects=False)
        self.assertEqual(admin.status_code, 302)
        self.assertEqual(unquote(admin.headers["location"]), "/login?next=/admin?clinic=kem")


if __name__ == "__main__":
    unittest.main()
