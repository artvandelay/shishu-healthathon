"""Deterministic call-state routing. No network."""

import unittest

import checklist
import call_state


def _update(field, value, state="answered", scope="none", child_ids=None, details=None):
    return {
        "field": field,
        "state": state,
        "value": value,
        "scope": scope,
        "child_ids": list(child_ids or []),
        "details": dict(details or {}),
    }


def _ask(state, topic_id, followup_id=None):
    return call_state.begin_question(state, {"topic_id": topic_id, "followup_id": followup_id})


class CallStateTests(unittest.TestCase):
    def test_no_children_marks_birth_body_and_illness_not_applicable(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_update("profile.children", [])], 1)
        for topic in checklist.TOPICS:
            if topic["group"] in ("birth", "body", "illness"):
                self.assertEqual(state["topics"][topic["id"]]["state"], "not_applicable", topic["id"])
        self.assertNotEqual(state["topics"]["preg_thalassemia"]["state"], "not_applicable")
        self.assertNotEqual(state["topics"]["intro_children"]["state"], "not_applicable")

    def test_ever_pregnant_no_removes_the_exact_six_topics(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_update("profile.ever_pregnant", False)], 1)
        removed = (
            "preg_conditions",
            "preg_scan",
            "preg_heart",
            "preg_transfusion",
            "preg_delivery",
            "fam_miscarriage",
        )
        for topic_id in removed:
            self.assertEqual(state["topics"][topic_id]["state"], "not_applicable", topic_id)
        self.assertEqual(
            [topic_id for topic_id, topic in state["topics"].items() if topic["reason"] == "ever_pregnant_no"],
            list(removed),
        )

    def test_no_cuts_or_bruises_still_asks_teeth_and_periods(self):
        state = call_state.new_call_state()
        children = [{"name": "A", "age": {"value": 11, "unit": "years"}, "sex": "female"}]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        state = _ask(state, "body_bleeding_bruise")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False, scope="all")],
            2,
        )

        self.assertEqual(call_state.next_question(state)["followup_id"], "teeth")
        state = _ask(state, "body_bleeding_bruise", "teeth")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False, scope="all", details={"teeth": False})],
            3,
        )
        self.assertEqual(call_state.next_question(state)["followup_id"], "periods")

        state = _ask(state, "body_bleeding_bruise", "periods")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", True, scope="all", details={"periods": True})],
            4,
        )
        topic = state["topics"]["body_bleeding_bruise"]
        self.assertIs(topic["value"], True)
        self.assertIs(topic["details"]["bleeding_or_bruises"], False)
        self.assertIs(topic["details"]["teeth"], False)
        self.assertIs(topic["details"]["periods"], True)
        self.assertNotEqual(
            (call_state.next_question(state) or {}).get("topic_id"),
            "body_bleeding_bruise",
        )

    def test_positive_nicu_blocks_until_reason_and_days_resolve(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="all")],
            1,
        )
        self.assertEqual(call_state.next_question(state)["followup_id"], "reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="all", details={"reason": "jaundice"})],
            2,
        )
        self.assertEqual(call_state.next_question(state)["followup_id"], "days")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="all", details={"days": 4})],
            3,
        )
        nxt = call_state.next_question(state)
        self.assertFalse(
            nxt and nxt.get("topic_id") == "birth_nicu" and nxt.get("question_kind") == "followup"
        )

    def test_aggregate_no_stores_no_child_instances(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"name": "A", "age": {"value": 4, "unit": "years"}, "sex": "male"}])],
            1,
        )
        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", False, scope="all")],
            2,
        )
        topic = state["topics"]["birth_nicu"]
        self.assertEqual(topic["instances"], [])
        self.assertEqual(topic["value"], False)
        self.assertEqual(topic["state"], "answered")

    def test_correction_replaces_the_value_and_adds_history(self):
        state = _ask(call_state.new_call_state(), "intro_name")
        state = call_state.apply_updates(state, [_update("topic.intro_name", "A")], 1)
        state = call_state.apply_updates(state, [_update("topic.intro_name", "B")], 2)
        self.assertEqual(state["topics"]["intro_name"]["value"], "B")
        self.assertTrue(any(event.get("previous_value") == "A" for event in state["history"]))
        self.assertIn(2, state["topics"]["intro_name"]["source_turn_ids"])

    def test_operation_reason_keeps_her_words_not_the_parent_token(self):
        state = _ask(call_state.new_call_state(), "birth_mode")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 4}]),
             _update("topic.birth_mode", "operation", details={"place": "hospital"})],
            1,
        )
        self.assertEqual(state["topics"]["birth_mode"]["details"].get("place"), "hospital")
        state = _ask(state, "birth_mode", "operation_reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_mode", "operation", details={"operation_reason": "डॉक्टर ने पानी कम बताया था।"})],
            2,
        )
        self.assertEqual(
            state["topics"]["birth_mode"]["details"].get("operation_reason"),
            "डॉक्टर ने पानी कम बताया था।",
        )

    def test_birth_mode_value_follows_delivery_type(self):
        state = _ask(call_state.new_call_state(), "birth_mode")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_mode", True, details={"delivery_type": "normal"})],
            1,
        )
        self.assertIs(state["topics"]["birth_mode"]["value"], False)
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_mode", False, details={"delivery_type": "operation"})],
            2,
        )
        self.assertIs(state["topics"]["birth_mode"]["value"], True)

    def test_clarification_does_not_close_the_followup(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        state = call_state.apply_updates(state, [_update("topic.birth_nicu", True)], 1)
        state = _ask(state, "birth_nicu", "reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True)],
            2,
            utterance="पीलिया हुआ था।",
        )
        self.assertNotIn("reason", state["topics"]["birth_nicu"]["details"])
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["followup_id"], "reason")

        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, details={"reason": "पीलिया"})],
            3,
        )
        self.assertEqual(state["topics"]["birth_nicu"]["details"].get("reason"), "पीलिया")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["followup_id"], "days")

        # A bare boolean does close a pending yes/no follow-up: the pending question
        # is the follow-up itself, so the boolean is her typed answer to it.
        state = _ask(call_state.new_call_state(), "body_bleeding_bruise")
        state = call_state.apply_updates(state, [_update("topic.body_bleeding_bruise", False)], 1)
        state = _ask(state, "body_bleeding_bruise", "teeth")
        state = call_state.apply_updates(state, [_update("topic.body_bleeding_bruise", False)], 2)
        self.assertIs(state["topics"]["body_bleeding_bruise"]["details"].get("teeth"), False)
        self.assertNotEqual((call_state.next_question(state) or {}).get("followup_id"), "teeth")

    def test_a_no_to_a_count_followup_is_not_saved_as_the_count(self):
        state = _ask(call_state.new_call_state(), "ill_admissions")
        state = call_state.apply_updates(state, [_update("topic.ill_admissions", True)], 1)
        state = _ask(state, "ill_admissions", "count")
        state = call_state.apply_updates(
            state,
            [_update("topic.ill_admissions", True)],
            2,
            utterance="नहीं।",
        )
        topic = state["topics"]["ill_admissions"]
        self.assertNotIn("count", topic["details"])
        self.assertIs(topic["value"], True)
        self.assertEqual(call_state.next_question(state)["followup_id"], "count")

    def test_yes_no_questions_keep_haan_as_the_concern_direction(self):
        for topic in checklist.TOPICS:
            if not checklist.is_yes_no(topic["id"]):
                continue
            self.assertIn("क्या", topic["question"], topic["id"])
        for topic in checklist.TOPICS:
            if not checklist.is_yes_no(topic["id"]):
                continue
            note = call_state._topic_question(topic["id"])["prompt_note"]
            self.assertIn("same direction", note, topic["id"])
            if topic["group"] in checklist.CLINICAL_GROUPS:
                self.assertIn("हाँ", note, topic["id"])
                self.assertIn("concern", note, topic["id"])
        for spec in call_state.TOPIC_RULES["birth_hearing_vision"]["followups"]:
            if spec["id"] == "test_done":
                note = call_state._followup_question("birth_hearing_vision", spec)["prompt_note"]
                self.assertIn("हाँ", note)
                self.assertIn("concern", note)

    def test_pata_nahi_resolves_the_followup_as_unknown(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        state = call_state.apply_updates(state, [_update("topic.birth_nicu", True)], 1)
        state = _ask(state, "birth_nicu", "reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, state="unknown")],
            2,
        )
        self.assertEqual(state["topics"]["birth_nicu"]["details"].get("reason"), {"state": "unknown"})
        nxt = call_state.next_question(state)
        self.assertNotEqual((nxt or {}).get("followup_id"), "reason")


if __name__ == "__main__":
    unittest.main()
