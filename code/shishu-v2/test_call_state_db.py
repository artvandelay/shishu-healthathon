"""SQLite call-state persistence. Uses a temporary DATA_DIR."""

import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-call-state-")
os.environ["DATA_DIR"] = _TMP

import db  # noqa: E402


class CallStateDbTests(unittest.TestCase):
    def test_initialize_call_state_persists_54_blank_topics(self):
        call_id = db.create_call("gemini")
        state = db.initialize_call_state(call_id)
        self.assertEqual(len(state["topics"]), 54)
        self.assertTrue(all(topic["state"] == "not_asked" for topic in state["topics"].values()))
        loaded = db.get_call_state(call_id)
        self.assertEqual(len(loaded["topics"]), 54)

    def test_save_and_get_preserve_children_and_topic_state(self):
        call_id = db.create_call("gemini")
        state = db.initialize_call_state(call_id)
        state["profile"]["children"] = [
            {"id": "child_1", "name": "A", "age": {"value": 3, "unit": "years"}, "sex": "female"}
        ]
        state["topics"]["birth_nicu"]["state"] = "answered"
        state["topics"]["birth_nicu"]["value"] = True
        state["current_question"] = {"topic_id": "birth_nicu", "followup_id": "reason", "attempts": 1}
        db.save_call_state(call_id, state)
        loaded = db.get_call_state(call_id)
        self.assertEqual(loaded["profile"]["children"][0]["id"], "child_1")
        self.assertEqual(loaded["profile"]["children"][0]["age"]["value"], 3)
        self.assertEqual(loaded["topics"]["birth_nicu"]["state"], "answered")
        self.assertIs(loaded["topics"]["birth_nicu"]["value"], True)
        self.assertEqual(loaded["current_question"]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
