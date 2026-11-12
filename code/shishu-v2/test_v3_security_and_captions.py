"""Auth cookies, public copy, and caption leaks. No network."""

import os
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="shishu-v3-security-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import app  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import prompts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_HERE = Path(__file__).resolve().parent


class SessionTests(unittest.TestCase):
    def test_literal_role_and_passcode_are_not_sessions(self):
        self.assertEqual(app._read_session("admin"), ("", ""))
        self.assertEqual(app._read_session("user"), ("", ""))
        self.assertEqual(app._read_session(app._passcode()), ("", ""))
        self.assertEqual(app._read_session(app._admin_passcode()), ("", ""))

    def test_signed_token_round_trip_and_rejects_a_tampered_signature(self):
        token = app._session_token("admin", "Ada")
        self.assertNotIn(token, {"admin", "user"})
        self.assertEqual(app._read_session(token), ("admin", "Ada"))
        forged = token[:-8] + ("0" * 8)
        self.assertEqual(app._read_session(forged), ("", ""))

    def test_secure_cookie_follows_forwarded_https(self):
        class Req:
            def __init__(self, headers, scheme):
                self.headers = headers
                self.url = type("U", (), {"scheme": scheme})()

        self.assertTrue(app._cookie_secure(Req({"x-forwarded-proto": "https"}, "http")))
        self.assertFalse(app._cookie_secure(Req({}, "http")))

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


class CaptionAndCopyTests(unittest.TestCase):
    def test_login_page_hides_the_passcode_and_screening_word(self):
        html = (_HERE / "static" / "login.html").read_text(encoding="utf-8")
        self.assertNotIn("shishu2026", html)
        self.assertNotIn("screening", html.lower())
        self.assertIn("आपका नाम", html)
        self.assertIn("Your name", html)

    def test_call_status_matches_the_start_button(self):
        html = (_HERE / "static" / "call.html").read_text(encoding="utf-8")
        self.assertNotIn("Tap Mic", html)
        self.assertIn("कॉल शुरू करने के लिए नीचे दबाएँ", html)
        self.assertIn(">शुरू<", html)

    def test_logo_png_is_served(self):
        html = (_HERE / "static" / "call.html").read_text(encoding="utf-8")
        self.assertIn('src="/static/shishu-logo.png"', html)
        client = TestClient(app.app)
        res = client.get("/static/shishu-logo.png")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.headers["content-type"].startswith("image/png"))
        self.assertGreater(len(res.content), 1000)

    def test_caption_filter_drops_planner_notes_and_tool_names(self):
        leaked = (
            "BCG ki jagah par sujan hui? Wait for the user's audio input. "
            "Current domain: Child physical anomalies."
        )
        self.assertEqual(
            prompts.clean_assistant_caption(leaked),
            "BCG ki jagah par sujan hui?",
        )
        self.assertTrue(prompts.leaked_caption(leaked, prompts.clean_assistant_caption(leaked)))
        mixed = "Aasani se neel پڑ jaate hain?"
        self.assertNotIn("پ", prompts.clean_assistant_caption(mixed))
        tool = "save_profile record_turn कहो क्या नाल गिरी?"
        cleaned = prompts.clean_assistant_caption(tool)
        self.assertNotIn("save_profile", cleaned)
        self.assertNotIn("record_turn", cleaned)
        self.assertTrue(prompts.leaked_caption(tool, cleaned))
        delta = " haan"
        self.assertEqual(prompts.clean_assistant_caption(delta, exact=True), " haan")
        note = "*Note: Audio is in English, but requested reply language is Hindi.*"
        self.assertEqual(prompts.clean_assistant_caption(note), "")

    def test_live_prompt_uses_only_the_cursor_tools(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("save_profile", text)
        self.assertIn("record_turn", text)
        self.assertNotIn("record_finding", text)
        self.assertNotIn("end_call", text)
        closing = gemini_live._closing_hi({
            "phase": "closing",
            "close_reason": "emergency",
            "profile": {"caller_name": "आशा"},
            "answers": {},
        })
        self.assertIn("108", closing)
        self.assertEqual(gemini_live._session_settings({})["voice"], "Kore")
        self.assertEqual(db.DEFAULT_SETTINGS["gemini_model"], "gemini-3.1-flash-live-preview")

    def test_persisted_openai_engine_migrates_to_gemini(self):
        db.set_setting("active_engine", "openai")
        db.set_setting("openai_model", "gpt-live-1")
        db.set_setting("openai_voice", "marin")
        db.set_setting("openai_backend_model", "gpt-5.6-luna")
        db._init_db()
        settings = db.get_all_settings()
        self.assertEqual(settings["active_engine"], "gemini")
        self.assertNotIn("openai_model", settings)
        self.assertNotIn("openai_voice", settings)
        self.assertNotIn("openai_backend_model", settings)
