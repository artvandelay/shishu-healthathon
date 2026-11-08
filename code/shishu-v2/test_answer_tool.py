"""Gemini answer-tool contract. No network."""

import unittest

import call_state
from answer_tool import RECORD_ANSWERS_DECL, build_turn_context


class AnswerToolTests(unittest.TestCase):
    def test_declarations_expose_updates_and_required_child_ids(self):
        params = RECORD_ANSWERS_DECL["parameters"]
        self.assertIn("updates", params["properties"])
        self.assertIn("updates", params["required"])
        item = params["properties"]["updates"]["items"]
        self.assertIn("child_ids", item["properties"])
        self.assertEqual(item["properties"]["state"]["enum"], ["answered", "unknown", "no_response"])
        self.assertEqual(item["properties"]["scope"]["enum"], ["all", "specific", "none"])
        details = item["properties"]["details"]["properties"]
        self.assertEqual(details["delivery_type"]["enum"], ["normal", "operation"])
        self.assertEqual(details["place"]["enum"], ["home", "hospital"])
        self.assertIn("child_ids", item["required"])
        self.assertEqual(item["properties"]["child_ids"]["items"]["type"], "STRING")

    def test_turn_context_contains_child_ids_and_active_topic(self):
        state = call_state.new_call_state()
        state["profile"]["children"] = [
            {"id": "child_1", "name": "A", "age": {"value": 3, "unit": "years"}, "sex": "female"}
        ]
        question = {
            "topic_id": "birth_nicu",
            "followup_id": "reason",
            "question_kind": "followup",
            "prompt_note": "Ask the reason.",
        }
        text = build_turn_context(state, question)
        self.assertIn("child_1", text)
        self.assertIn("birth_nicu", text)
        self.assertIn("reason", text)


if __name__ == "__main__":
    unittest.main()
