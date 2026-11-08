"""Gemini answer-tool contract. No network."""

import json
import unittest

import call_state
import prompts
from answer_tool import END_CALL_DECL, RECORD_ANSWERS_DECL, build_turn_context


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

    def test_value_is_a_direct_answer_union(self):
        value = RECORD_ANSWERS_DECL["parameters"]["properties"]["updates"]["items"]["properties"]["value"]
        self.assertNotIn("type", value)
        self.assertEqual(
            [item["type"] for item in value["anyOf"]],
            ["BOOLEAN", "STRING", "NUMBER", "ARRAY", "NULL"],
        )
        self.assertEqual(value["anyOf"][3]["items"]["type"], "OBJECT")

    def test_live_config_keeps_the_direct_value_union(self):
        from google.genai import types

        config = types.LiveConnectConfig(
            tools=[{"function_declarations": [RECORD_ANSWERS_DECL]}]
        )
        declared = config.model_dump(mode="json", by_alias=True, exclude_none=True)
        value = declared["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["updates"]["items"]["properties"]["value"]
        self.assertEqual(
            [item["type"] for item in value["anyOf"]],
            ["BOOLEAN", "STRING", "NUMBER", "ARRAY", "NULL"],
        )

    def test_end_call_allows_emergency_and_caller_stop_only(self):
        blob = json.dumps(END_CALL_DECL)
        self.assertIn("emergency", blob)
        self.assertIn("caller_stop", blob)
        self.assertNotIn("complete", blob)

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

    def test_gemini_prompt_uses_record_answers_and_server_completion(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("record_answers", text)
        self.assertIn("The server ends the call after your audio finishes.", text)
        self.assertNotIn('pass ["birth", "body", "illness"]', text)


if __name__ == "__main__":
    unittest.main()
