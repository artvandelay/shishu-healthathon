"""SQLite call-state persistence. Uses a temporary DATA_DIR."""

import os
import sqlite3
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-call-state-")
os.environ["DATA_DIR"] = _TMP

import db  # noqa: E402


class CallStateDbTests(unittest.TestCase):
    def test_migration_adds_state_json(self):
        with sqlite3.connect(db.DB_PATH) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()}
        self.assertIn("state_json", columns)

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
        db.save_call_state(call_id, state)
        loaded = db.get_call_state(call_id)
        self.assertEqual(loaded["profile"]["children"][0]["id"], "child_1")
        self.assertEqual(loaded["profile"]["children"][0]["age"]["value"], 3)
        self.assertEqual(loaded["topics"]["birth_nicu"]["state"], "answered")
        self.assertIs(loaded["topics"]["birth_nicu"]["value"], True)

    def test_get_call_state_returns_none_for_a_legacy_empty_row(self):
        call_id = db.create_call("gemini")
        self.assertIsNone(db.get_call_state(call_id))

    def test_get_call_returns_state_none_for_a_legacy_empty_row(self):
        call_id = db.create_call("gemini")
        self.assertIsNone(db.get_call(call_id)["state"])


if __name__ == "__main__":
    unittest.main()
