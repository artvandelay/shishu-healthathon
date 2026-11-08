"""Checks for the 6 Nov 2026 bug report. No network. No secrets printed."""

import os
import tempfile
import time
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="shishu-bugfixes-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("ADMIN_PASSCODE", "unit-test-admin")

import app  # noqa: E402
import audio_recorder  # noqa: E402
import prompts  # noqa: E402
import summarizer  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _doctor_shell() -> str:
    parts = []
    for heading in summarizer._REQUIRED_HEADINGS:
        parts.append(heading + "\nNot reported.\n")
    return "".join(parts)


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

    def test_intro_and_cord_questions_are_split(self):
        text = prompts.CLINICAL_CLUSTERS_REFERENCE
        self.assertNotIn("naam, aur bachche ka naam aur umar", text)
        self.assertNotIn("naal se zyada khoon aaya, ya", text)
        self.assertIn("Aapka naam bataiyega please?", text)
        self.assertIn("Naal girne par zyada khoon aaya tha?", text)
        self.assertIn("Naal girne mein 3 hafte se zyada lage the?", text)
        self.assertIn("kaanch ka box", text)
        self.assertNotIn("intro, story, birth", prompts.GEMINI_LIVE_SYSTEM_PROMPT)
        self.assertIn("NEVER speak Spanish", prompts.GEMINI_LIVE_SYSTEM_PROMPT)
        self.assertIn("NEVER ANSWER YOUR OWN QUESTIONS", prompts.GEMINI_LIVE_SYSTEM_PROMPT)
        self.assertIn("CLARIFICATION POLICY", prompts.GEMINI_LIVE_SYSTEM_PROMPT)
        self.assertIn("SPOKEN OUTPUT", prompts.GEMINI_LIVE_SYSTEM_PROMPT)
        self.assertIn("SPOKEN OUTPUT", prompts.OPENAI_LIVE_VOICE_PROMPT)
        self.assertNotIn("The same recap in Roman Hindi", prompts.DOCTOR_MOTHER_SUMMARY_PROMPT)

    def test_default_gemini_model_is_31_flash_live(self):
        import db
        self.assertEqual(db.DEFAULT_SETTINGS["gemini_model"], "gemini-3.1-flash-live-preview")

    def test_intro_permits_short_or_long_and_clarifications_and_hardcoded_voices(self):
        import gemini_live, openai_live
        intro = prompts.CLINICAL_CLUSTERS_REFERENCE
        self.assertIn("chhota ya lamba", intro)
        self.assertIn("bejhijhak poochh sakti hain", intro)
        self.assertEqual(gemini_live._session_settings({})["voice"], "Kore")
        self.assertEqual(openai_live.LIVE_VOICE, "marin")

    def test_logo_png_is_served(self):
        html = (_HERE / "static" / "call.html").read_text(encoding="utf-8")
        self.assertIn('src="/static/shishu-logo.png"', html)
        self.assertIn("background: #ffffff", html)
        client = TestClient(app.app)
        res = client.get("/static/shishu-logo.png")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.headers["content-type"].startswith("image/png"))
        self.assertGreater(len(res.content), 1000)

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


class SummaryAndDurationTests(unittest.TestCase):
    def test_duration_is_wall_clock(self):
        folder = Path(_TMP) / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        rec = audio_recorder.CallAudioRecorder(7, output_dir=str(folder))
        rec._started = time.monotonic() - 5.2
        rec.add_bot_pcm(b"\x00\x00" * 100, 24000)
        _name, duration = rec.save()
        self.assertGreaterEqual(duration, 5)
        self.assertLess(duration, 30)

    def test_unknown_slot_does_not_keep_a_mismatched_flag(self):
        payload = {
            "doctor_markdown": _doctor_shell(),
            "mother_hindi": "बच्चे को बार बार बुखार रहता है।",
            "extracted_findings": {
                "bleeding_joints": {
                    "status": "info",
                    "detail": "History of easy bruising is unknown",
                    "is_red_flag": False,
                },
                "birth_nicu": {
                    "status": "positive",
                    "detail": "Blister at the BCG injection site",
                    "is_red_flag": True,
                },
            },
            "red_flags": [
                "bleeding_joints: Child requires blood transfusion and has abdominal distension",
                "birth_nicu: Blister at the BCG injection site",
                "birth_nicu: Blister at the BCG injection site again",
            ],
        }
        transcript = [
            {
                "role": "user",
                "text": (
                    "Do bade bachche teen mahine mein guzar gaye. "
                    "Six hospital admissions in three months and IV antibiotics."
                ),
            }
        ]
        result = summarizer._assemble(payload, {}, [], transcript)
        flags = result["red_flags"]
        self.assertFalse(any(flag.startswith("bleeding_joints:") for flag in flags))
        self.assertEqual(sum(flag.startswith("birth_nicu:") for flag in flags), 1)
        self.assertFalse(any(flag.startswith("infections:") for flag in flags))


if __name__ == "__main__":
    unittest.main()
