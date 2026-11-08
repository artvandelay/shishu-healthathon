"""Checks for the answer backstop. No network."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="shishu-backstop-"))

import answer_backstop  # noqa: E402
import call_state  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402


def _up(field, value):
    return {"field": field, "state": "answered", "value": value, "scope": "none", "child_ids": [], "details": {}}


def _with_children(*ages):
    children = [{"age": {"value": age, "unit": "years"}} for age in ages]
    return call_state.apply_updates(
        call_state.new_call_state(),
        [_up("profile.caller_name", "कविता"), _up("profile.marital_status", "married"),
         _up("profile.children", children), _up("topic.intro_bridge", True)],
        1,
    )


def _asked(state, topic_id, followup_id=None):
    question = {"topic_id": topic_id, "followup_id": followup_id,
                "question_kind": "followup" if followup_id else "topic"}
    return call_state.begin_question(state, question), question


def _save(state, question, result):
    return call_state.apply_updates(state, answer_backstop.to_updates(question, state, result), 2)


class ToUpdatesTests(unittest.TestCase):
    def test_yes_no_answer_is_saved_in_the_question_direction(self):
        state, question = _asked(_with_children(3), "body_growth")
        saved = _save(state, question, {"kind": "answer", "value": False})
        self.assertEqual(saved["topics"]["body_growth"]["state"], "answered")
        self.assertIs(saved["topics"]["body_growth"]["value"], False)

    def test_non_answers_save_nothing(self):
        state, question = _asked(_with_children(3), "body_growth")
        for kind in ("clarification", "not_an_answer", None):
            self.assertEqual(answer_backstop.to_updates(question, state, {"kind": kind, "value": True}), [])
        self.assertEqual(answer_backstop.to_updates(question, state, {"kind": "answer", "value": None}), [])

    def test_unknown_is_saved_as_unknown(self):
        state, question = _asked(_with_children(3), "birth_nicu")
        saved = _save(state, question, {"kind": "unknown"})
        self.assertEqual(saved["topics"]["birth_nicu"]["state"], "unknown")

    def test_birth_mode_keeps_typed_fields(self):
        state, question = _asked(_with_children(3), "birth_mode")
        saved = _save(state, question, {"kind": "answer", "delivery_type": "operation", "place": "hospital",
                                        "operation_reason": None})
        topic = saved["topics"]["birth_mode"]
        self.assertEqual(topic["details"]["delivery_type"], "operation")
        self.assertEqual(topic["details"]["place"], "hospital")
        self.assertTrue(call_state._is_positive(topic))

    def test_children_become_profile_children(self):
        state, question = _asked(call_state.new_call_state(), "intro_children")
        saved = _save(state, question, {"kind": "answer", "count": 2, "children": [
            {"age_value": 4, "age_unit": "years", "sex": "female"},
            {"age_value": 8, "age_unit": "months", "sex": "unknown"},
        ]})
        children = saved["profile"]["children"]
        self.assertEqual([c["age"] for c in children],
                         [{"value": 4, "unit": "years"}, {"value": 8, "unit": "months"}])
        self.assertEqual(children[0]["sex"], "female")
        none = _save(state, question, {"kind": "answer", "count": 0, "children": []})
        self.assertEqual(none["profile"]["children"], [])

    def test_followup_detail_and_child_ids(self):
        state, _ = _asked(_with_children(4, 1), "birth_nicu")
        state = call_state.apply_updates(state, [_up("topic.birth_nicu", True)], 2)
        self.assertIn("affected_child", [f["id"] for f in call_state._pending_followups(state, "birth_nicu")])
        state, question = _asked(state, "birth_nicu", "affected_child")
        schema = answer_backstop.response_schema(question, state)
        self.assertEqual(schema.properties["child_ids"].items.enum, ["child_1", "child_2"])
        updates = answer_backstop.to_updates(question, state, {"kind": "answer", "detail": None,
                                                               "child_ids": ["child_2", "child_9"]})
        self.assertEqual(updates[0]["child_ids"], ["child_2"])
        self.assertEqual(updates[0]["details"], {"affected_child": "child_2"})


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


def _turn_complete():
    content = SimpleNamespace(interrupted=False, input_transcription=None, output_transcription=None,
                              model_turn=None, turn_complete=True)
    return SimpleNamespace(tool_call_cancellation=None, server_content=content, tool_call=None)


class TurnCompleteTests(unittest.TestCase):
    def _run(self, bot_text_since_user, result):
        call_id = db.create_call("gemini")
        state = _with_children(3)
        self.assertEqual(call_state.next_question(state)["topic_id"], "birth_mode")
        state, _ = _asked(state, "birth_mode")
        turn = gemini_live._TurnState()
        turn.call_state = state
        turn.tool_called_since_user = False
        turn.last_bot_text = "बच्चे का जन्म नॉर्मल डिलीवरी से हुआ या ऑपरेशन से?"
        turn.last_user_text = "ऑपरेशन से, अस्पताल में।"
        turn.bot_text_since_user = bot_text_since_user
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1300, 300, 0)
        session = _FakeSession()

        async def fake_extract(question, live_state, bot_text, user_text):
            self.assertEqual(question["topic_id"], "birth_mode")
            self.assertEqual(user_text, turn.last_user_text)
            return result

        original = answer_backstop.extract
        answer_backstop.extract = fake_extract
        try:
            asyncio.run(gemini_live._handle_server_message(
                _turn_complete(), session, asyncio.Lock(), _FakeWs(), asyncio.Lock(), call_id,
                telemetry, turn, None, set(), set(),
            ))
        finally:
            answer_backstop.extract = original
        return turn, session, telemetry

    def test_saves_and_stays_quiet_when_the_model_already_moved_on(self):
        turn, session, telemetry = self._run("ठीक है। क्या बच्चा समय से बहुत पहले पैदा हुआ था?",
                                             {"kind": "answer", "delivery_type": "operation", "place": "hospital"})
        self.assertEqual(turn.call_state["topics"]["birth_mode"]["details"]["delivery_type"], "operation")
        self.assertEqual(turn.call_state["current_question"]["followup_id"], "operation_reason")
        self.assertEqual(session.texts, [])
        self.assertEqual(telemetry["backstop_saves"], 1)
        self.assertTrue(turn.backstop_checked)

    def test_model_that_stays_out_of_sync_gets_one_resync_cue(self):
        turn, session, telemetry = self._run("क्या आपकी एक बेटी है, जो तीन साल की है?",
                                             {"kind": "answer", "delivery_type": "normal", "place": "home"})
        self.assertEqual(session.texts, [])
        self.assertTrue(turn.backstop_resync)

        async def not_an_answer(*args):
            return {"kind": "not_an_answer"}

        original = answer_backstop.extract
        answer_backstop.extract = not_an_answer
        try:
            for _ in range(2):
                turn.last_user_text = "ठीक है।"
                turn.bot_text_since_user = "क्या आपकी एक बेटी है, जो तीन साल की है?"
                turn.tool_called_since_user = False
                turn.backstop_checked = False
                asyncio.run(gemini_live._handle_server_message(
                    _turn_complete(), session, asyncio.Lock(), _FakeWs(), asyncio.Lock(),
                    turn.call_id or db.create_call("gemini"), telemetry, turn, None, set(), set(),
                ))
        finally:
            answer_backstop.extract = original
        self.assertEqual(len(session.texts), 1)
        self.assertIn("already saved", session.texts[0])
        self.assertFalse(turn.backstop_resync)

    def test_cues_the_next_question_when_the_model_said_nothing(self):
        turn, session, _ = self._run("", {"kind": "answer", "delivery_type": "normal", "place": "home"})
        self.assertEqual(len(session.texts), 1)
        self.assertIn("saved her last answer", session.texts[0])
        self.assertTrue(turn.awaiting_bot_reply)

    def test_an_empty_bot_turn_is_not_counted_as_her_silence(self):
        call_id = db.create_call("gemini")
        state, _ = _asked(_with_children(3), "birth_mode")
        turn = gemini_live._TurnState()
        turn.call_state = state
        turn.tool_called_since_user = True
        turn.awaiting_bot_reply = True
        turn.bot_replied_since_user = False
        turn.last_user_text = "ऑपरेशन से।"
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1300, 300, 0)
        asyncio.run(gemini_live._handle_server_message(
            _turn_complete(), _FakeSession(), asyncio.Lock(), _FakeWs(), asyncio.Lock(),
            call_id, telemetry, turn, None, set(), set(),
        ))
        self.assertTrue(turn.awaiting_bot_reply)
        self.assertEqual(turn.quiet_from, 0.0)

    def test_non_answer_changes_nothing(self):
        turn, session, telemetry = self._run("", {"kind": "clarification"})
        self.assertNotIn("delivery_type", turn.call_state["topics"]["birth_mode"].get("details") or {})
        self.assertEqual(session.texts, [])
        self.assertNotIn("backstop_saves", telemetry)


if __name__ == "__main__":
    unittest.main()
