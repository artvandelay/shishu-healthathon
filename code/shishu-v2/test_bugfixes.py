"""Checks for the 6 Nov 2026 bug report. No network. No secrets printed."""

import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-bugfixes-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("ADMIN_PASSCODE", "unit-test-admin")

import app  # noqa: E402
import prompts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class SessionTests(unittest.TestCase):
    def test_signed_token_round_trip_and_rejects_a_tampered_signature(self):
        token = app._session_token("admin", "Ada")
        self.assertNotIn(token, {"admin", "user"})
        self.assertEqual(app._read_session(token), ("admin", "Ada"))
        forged = token[:-8] + ("0" * 8)
        self.assertEqual(app._read_session(forged), ("", ""))

    def test_login_cookie_is_opaque_and_bugs_stay_open_to_a_user(self):
        user = TestClient(app.app)
        res = user.post(
            "/api/login",
            json={"username": "Tester", "passcode": app._passcode()},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["role"], "user")
        cookie = res.cookies.get("shishu_auth")
        self.assertTrue(cookie)
        self.assertNotIn(cookie, {"user", "admin"})
        self.assertIsNone(res.cookies.get("shishu_role"))
        self.assertEqual(user.get("/api/calls").status_code, 403)
        bug = user.post("/api/bugs", json={"category": "test", "note": "check", "call_id": 0})
        self.assertEqual(bug.status_code, 200)
        self.assertNotIn("snapshot", bug.json()["bug"])

    def test_bug_report_does_not_copy_another_callers_transcript(self):
        import db

        owner = db.create_call("gemini", caller_name="Owner")
        db.append_transcript(owner, "user", "Secret line from Owner.")
        db.complete_call(owner, 10, "", "note", "note")
        user = TestClient(app.app)
        logged = user.post("/api/login", json={"username": "Other", "passcode": app._passcode()})
        self.assertEqual(logged.status_code, 200)
        filed = user.post("/api/bugs", json={"category": "test", "note": "peek", "call_id": owner})
        self.assertEqual(filed.status_code, 200)
        self.assertNotIn("snapshot", filed.json()["bug"])
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()})
        listed = admin.get("/api/bugs")
        self.assertEqual(listed.status_code, 200)
        match = next(item for item in listed.json() if item["call_id"] == owner)
        self.assertEqual(match["snapshot"]["last_transcript"], [])

        forged = TestClient(app.app)
        denied = forged.get("/api/calls", headers={"Cookie": "shishu_auth=admin"})
        self.assertEqual(denied.status_code, 401)
        escalated = forged.get(
            "/api/config",
            headers={"Cookie": "shishu_auth=user; shishu_role=admin"},
        )
        self.assertEqual(escalated.status_code, 401)

        admin = TestClient(app.app)
        admin_login = admin.post(
            "/api/login",
            json={"username": "admin", "passcode": app._admin_passcode()},
        )
        self.assertEqual(admin_login.status_code, 200)
        self.assertEqual(admin.get("/api/calls").status_code, 200)


class PromptAndCopyTests(unittest.TestCase):
    def test_caption_filter_drops_planner_notes_and_other_scripts(self):
        leaked = (
            "BCG ki jagah par sujan hui? Wait for the user's audio input. "
            "Current domain: Child physical anomalies."
        )
        self.assertEqual(
            prompts.clean_assistant_caption(leaked),
            "BCG ki jagah par sujan hui?",
        )
        mixed = "Aasani se neel پڑ jaate hain?"
        self.assertNotIn("پ", prompts.clean_assistant_caption(mixed))
        bengali = "Bachcha normal delivery se hua ya operation se? বাবাদে"
        cleaned = prompts.clean_assistant_caption(bengali)
        self.assertNotIn("ব", cleaned)
        self.assertTrue(cleaned.startswith("Bachcha normal"))
        delta = " haan"
        self.assertEqual(prompts.clean_assistant_caption(delta, exact=True), " haan")
        note = "*Note: Audio is in English, but requested reply language is Hindi.*"
        self.assertEqual(prompts.clean_assistant_caption(note), "")


if __name__ == "__main__":
    unittest.main()
