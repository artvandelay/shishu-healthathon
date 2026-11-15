"""Auth cookies, public copy, and caption leaks. No network."""

import asyncio
import os
import sys
import unittest
import wave
from pathlib import Path
from urllib.parse import unquote

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests import APP_DIR  # noqa: E402

import app  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import prompts  # noqa: E402
import report_service  # noqa: E402
import seed  # noqa: E402
from call_cursor import apply_profile, new_cursor_state  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_HERE = APP_DIR


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

    def test_client_ip_uses_forwarded_header_and_falls_back_to_client_host(self):
        class Req:
            def __init__(self, headers, host):
                self.headers = headers
                self.client = None if host is None else type("C", (), {"host": host})()

        self.assertEqual(
            app._client_ip(Req({"x-forwarded-for": "203.0.113.10, 10.0.0.1"}, "10.0.0.2")),
            "203.0.113.10",
        )
        self.assertEqual(
            app._client_ip(Req({"x-forwarded-for": "   "}, "198.51.100.4")),
            "198.51.100.4",
        )
        self.assertEqual(app._client_ip(Req({}, "198.51.100.4")), "198.51.100.4")
        self.assertEqual(app._client_ip(Req({}, None)), "")

    def test_config_is_minimal_and_dead_settings_endpoints_are_removed(self):
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "Ada", "passcode": app._admin_passcode()})
        cfg = admin.get("/api/config")
        self.assertEqual(cfg.status_code, 200)
        self.assertEqual(cfg.json(), {"role": "admin", "username": "Ada"})
        self.assertEqual(admin.post("/api/config", json={"active_engine": "gemini"}).status_code, 405)
        self.assertEqual(admin.get("/api/settings").status_code, 404)
        self.assertEqual(admin.post("/api/settings", json={"gemini_voice": "Kore"}).status_code, 404)

    def test_bug_report_does_not_copy_another_callers_transcript(self):
        owner = db.create_call(caller_name="Owner")
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
        self.assertNotIn("settings", match["snapshot"])

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

    def test_login_redirect_keeps_the_query(self):
        client = TestClient(app.app)
        home = client.get("/?clinic=kem", follow_redirects=False)
        self.assertEqual(home.status_code, 302)
        self.assertEqual(unquote(home.headers["location"]), "/login?next=/?clinic=kem")
        admin = client.get("/admin?clinic=kem", follow_redirects=False)
        self.assertEqual(admin.status_code, 302)
        self.assertEqual(unquote(admin.headers["location"]), "/login?next=/admin?clinic=kem")


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

    def test_live_prompt_uses_only_the_cursor_tools(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("next_question", text)
        self.assertNotIn("save_profile", text)
        self.assertNotIn("record_turn", text)
        self.assertNotIn("record_finding", text)
        self.assertNotIn("end_call", text)
        closing = prompts.closing_hi({
            "phase": "closing",
            "close_reason": "emergency",
            "profile": {"caller_name": "आशा"},
            "answers": {},
        })
        self.assertIn("108", closing)
        self.assertEqual(gemini_live._session_settings({})["voice"], "Kore")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["gemini_model"], "gemini-3.1-flash-live-preview")
        self.assertFalse(hasattr(db, "DEFAULT_SETTINGS"))


class Stage6BAccessTests(unittest.TestCase):
    def test_user_cannot_fetch_a_pdf_by_call_id(self):
        call_id = db.create_call(caller_name="Pooja")
        db.complete_call(call_id, 30, "", "", "")
        token = db.get_call(call_id)["share_token"]
        user = TestClient(app.app)
        logged = user.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()})
        self.assertEqual(logged.status_code, 200)
        denied = user.get(f"/api/calls/{call_id}/mother.pdf")
        self.assertEqual(denied.status_code, 403)
        denied_post = user.post(f"/api/calls/{call_id}/mother.pdf")
        self.assertEqual(denied_post.status_code, 403)
        by_token = user.get(f"/api/calls/by-token/{token}/mother.pdf")
        self.assertNotEqual(by_token.status_code, 403)
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()})
        admin_pdf = admin.get(f"/api/calls/{call_id}/mother.pdf")
        self.assertNotEqual(admin_pdf.status_code, 403)

    def test_bug_body_is_capped_and_ownership_uses_the_share_token(self):
        user = TestClient(app.app)
        user.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()})
        huge = user.post("/api/bugs", json={"category": "test", "note": "x" * (512 * 1024), "call_id": 0})
        self.assertEqual(huge.status_code, 413)
        note = "n" * 2500
        filed = user.post("/api/bugs", json={"category": "test", "note": note, "call_id": 0})
        self.assertEqual(filed.status_code, 200)
        self.assertEqual(len(filed.json()["bug"]["note"]), 2000)

        call_id = db.create_call(caller_name="Pooja")
        db.append_transcript(call_id, "user", "Secret line from this call.")
        db.complete_call(call_id, 10, "", "", "")
        token = db.get_call(call_id)["share_token"]
        owned = user.post(
            "/api/bugs",
            json={"category": "test", "note": "mine", "call_id": call_id, "share_token": token},
        )
        self.assertEqual(owned.status_code, 200)
        stranger = TestClient(app.app)
        stranger.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()})
        peeked = stranger.post(
            "/api/bugs",
            json={"category": "test", "note": "guess", "call_id": call_id},
        )
        self.assertEqual(peeked.status_code, 200)
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()})
        listed = admin.get("/api/bugs").json()
        match = next(item for item in listed if item["id"] == owned.json()["bug"]["id"])
        texts = [row.get("text") for row in match["snapshot"]["last_transcript"]]
        self.assertIn("Secret line from this call.", texts)
        other = next(item for item in listed if item["id"] == peeked.json()["bug"]["id"])
        self.assertEqual(other["snapshot"]["last_transcript"], [])

    def test_failed_logins_and_token_pdf_rebuilds_are_throttled(self):
        app._reset_rate_limits()
        try:
            client = TestClient(app.app)
            for _ in range(app._LOGIN_FAILS):
                failed = client.post("/api/login", json={"username": "Pooja", "passcode": "wrong"})
                self.assertEqual(failed.status_code, 401)
            blocked = client.post("/api/login", json={"username": "Pooja", "passcode": "wrong"})
            self.assertEqual(blocked.status_code, 429)

            call_id = db.create_call(caller_name="Pooja")
            db.complete_call(call_id, 12, "", "", "")
            token = db.get_call(call_id)["share_token"]
            for _ in range(app._PDF_REBUILDS):
                app._rate_hit(app._pdf_rebuilds, token, app._PDF_WINDOW_SEC)
            rebuild = client.post(f"/api/calls/by-token/{token}/mother.pdf")
            self.assertEqual(rebuild.status_code, 429)
        finally:
            app._reset_rate_limits()

    def test_report_inflight_clears_and_seed_audio_is_stereo_16k(self):
        state, hold = apply_profile(
            new_cursor_state(),
            caller_name="Pooja",
            marital_status="unmarried",
        )
        self.assertIsNone(hold)
        state["phase"] = "closing"
        state["close_reason"] = "complete"
        call_id = db.create_call(caller_name="Pooja")
        db.save_cursor_state(call_id, state)
        db.complete_call(call_id, 20, "", "", "")

        async def render():
            return await report_service.mother_pdf(call_id)

        first = asyncio.run(render())
        self.assertEqual(report_service._inflight, {})
        second = asyncio.run(render())
        self.assertEqual(report_service._inflight, {})
        self.assertTrue(first[0].startswith(b"%PDF"))
        self.assertTrue(second[0].startswith(b"%PDF"))

        with db._db_lock:
            db.conn.execute("DELETE FROM calls")
            db.conn.commit()
        seed.ensure_seeded()
        calls = db.list_calls()
        self.assertEqual(len(calls), 2)
        by_name = {}
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()})
        for row in calls:
            self.assertNotIn("findings", row)
            full = db.get_call(row["id"])
            self.assertNotIn("findings", full)
            self.assertEqual(full["cursor_state"]["phase"], "closing")
            self.assertEqual(full["cursor_state"]["close_reason"], "complete")
            self.assertTrue(full["summary_mother"])
            self.assertTrue(full["summary_doctor"])
            path = os.path.join(db.DATA_DIR, "audio", full["audio_path"])
            with wave.open(path) as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getframerate(), 16000)
                self.assertEqual(handle.getsampwidth(), 2)

            get_pdf = admin.get(f"/api/calls/{row['id']}/mother.pdf")
            self.assertEqual(get_pdf.status_code, 200)
            self.assertTrue(get_pdf.content.startswith(b"%PDF"))

            by_token = admin.get(f"/api/calls/by-token/{full['share_token']}/mother.pdf")
            self.assertEqual(by_token.status_code, 200)
            self.assertEqual(by_token.content, get_pdf.content)

            rebuilt = admin.post(f"/api/calls/{row['id']}/mother.pdf")
            self.assertEqual(rebuilt.status_code, 200)
            self.assertTrue(rebuilt.content.startswith(b"%PDF"))

            after_rebuild = admin.get(f"/api/calls/{row['id']}/mother.pdf")
            self.assertEqual(after_rebuild.status_code, 200)
            self.assertEqual(after_rebuild.content, rebuilt.content)

            by_name[full["caller_name"]] = full

        self.assertEqual(set(by_name), {"शिल्पा", "गीता"})
        shilpa = by_name["शिल्पा"]
        self.assertTrue(shilpa["red_flags"])
        shilpa_answers = shilpa["cursor_state"]["answers"]
        for qid in ("birth_jaundice", "body_joint_swelling", "preg_conditions", "fam_infant_death"):
            self.assertEqual(shilpa_answers[qid]["state"], "concern")
        self.assertIn(
            "तीन दिन नीली लाइट में रखा",
            shilpa_answers["birth_jaundice"]["followups"]["jaundice_days"]["note"],
        )
        self.assertIn(
            "5 महीने का बेटा",
            shilpa_answers["fam_infant_death"]["followups"]["infant_death_age"]["note"],
        )
        self.assertIn(
            "झटके आने से",
            shilpa_answers["fam_infant_death"]["followups"]["infant_death_cause"]["note"],
        )

        geeta = by_name["गीता"]
        self.assertEqual(geeta["red_flags"], [])
        geeta_answers = geeta["cursor_state"]["answers"]
        self.assertTrue(
            all(
                entry.get("state") in ("clear", "not_applicable")
                for entry in geeta_answers.values()
            )
        )
        clarified = [
            turn["text"]
            for turn in geeta["transcript"]
            if "थैलेसीमिया माइनर या ट्रेट" in turn["text"] and turn["role"] == "assistant"
        ]
        self.assertEqual(len(clarified), 1)
        self.assertNotIn("transfusion", clarified[0].lower())
        self.assertNotIn("baar-baar", clarified[0].lower())
