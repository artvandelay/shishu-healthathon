"""Auth cookies, public copy, and caption leaks. No network."""

import asyncio
import os
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

from tests import APP_DIR

import app  # noqa: E402
import copy_hi  # noqa: E402
import db  # noqa: E402
import engine  # noqa: E402
import gemini_live  # noqa: E402
import report_service  # noqa: E402
import seed  # noqa: E402
from models import (  # noqa: E402
    AnswerStatus,
    CallState,
    CloseReason,
    MaritalStatus,
    Phase,
    ProfilePatch,
    TurnCommand,
)
from fastapi.testclient import TestClient  # noqa: E402


class SessionTests(unittest.TestCase):
    def test_literal_role_and_passcode_are_not_sessions(self):
        for raw in ("admin", "user", app._passcode(), app._admin_passcode()):
            self.assertEqual(app._read_session(raw), ("", ""))

    def test_signed_token_round_trip_and_rejects_a_tampered_signature(self):
        token = app._session_token("admin", "Ada")
        self.assertNotIn(token, {"admin", "user"})
        self.assertEqual(app._read_session(token), ("admin", "Ada"))
        self.assertEqual(app._read_session(token[:-8] + ("0" * 8)), ("", ""))

    def test_secure_cookie_and_client_ip_headers(self):
        req_https = SimpleNamespace(headers={"x-forwarded-proto": "https"}, url=SimpleNamespace(scheme="http"))
        req_http = SimpleNamespace(headers={}, url=SimpleNamespace(scheme="http"))
        self.assertTrue(app._cookie_secure(req_https))
        self.assertFalse(app._cookie_secure(req_http))

        self.assertEqual(
            app._client_ip(SimpleNamespace(headers={"x-forwarded-for": "203.0.113.10, 10.0.0.1"}, client=SimpleNamespace(host="10.0.0.2"))),
            "203.0.113.10",
        )
        self.assertEqual(
            app._client_ip(SimpleNamespace(headers={"x-forwarded-for": "   "}, client=SimpleNamespace(host="198.51.100.4"))),
            "198.51.100.4",
        )
        self.assertEqual(app._client_ip(SimpleNamespace(headers={}, client=SimpleNamespace(host="198.51.100.4"))), "198.51.100.4")
        self.assertEqual(app._client_ip(SimpleNamespace(headers={}, client=None)), "")

    def test_login_cookie_is_opaque_and_config_is_minimal(self):
        user = TestClient(app.app)
        res = user.post("/api/login", json={"username": "Tester", "passcode": app._passcode()})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["role"], "user")
        cookie = res.cookies.get("shishu_auth")
        self.assertTrue(cookie and cookie not in {"user", "admin"})
        self.assertIsNone(res.cookies.get("shishu_role"))
        self.assertEqual(user.get("/api/calls").status_code, 403)
        bug = user.post("/api/bugs", json={"category": "test", "note": "check", "call_id": 0})
        self.assertEqual(bug.status_code, 200)
        self.assertNotIn("snapshot", bug.json()["bug"])

        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "Ada", "passcode": app._admin_passcode()})
        cfg = admin.get("/api/config")
        self.assertEqual((cfg.status_code, cfg.json()), (200, {"role": "admin", "username": "Ada"}))
        self.assertEqual(admin.post("/api/config", json={}).status_code, 405)

    def test_bug_report_does_not_copy_another_callers_transcript(self):
        owner = db.create_call(caller_name="Owner")
        db.append_transcript(owner, "user", "Secret line from Owner.")
        db.complete_call(owner, 10)
        db.save_report_text(owner, "note", "note")
        user = TestClient(app.app)
        self.assertEqual(user.post("/api/login", json={"username": "Other", "passcode": app._passcode()}).status_code, 200)
        filed = user.post("/api/bugs", json={"category": "test", "note": "peek", "call_id": owner})
        self.assertEqual(filed.status_code, 200)
        self.assertNotIn("snapshot", filed.json()["bug"])

        admin = TestClient(app.app)
        self.assertEqual(admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()}).status_code, 200)
        self.assertEqual(admin.get("/api/calls").status_code, 200)
        match = next(item for item in admin.get("/api/bugs").json() if item["call_id"] == owner)
        self.assertEqual(match["snapshot"]["last_transcript"], [])

        forged = TestClient(app.app)
        self.assertEqual(forged.get("/api/calls", headers={"Cookie": "shishu_auth=admin"}).status_code, 401)
        self.assertEqual(forged.get("/api/config", headers={"Cookie": "shishu_auth=user; shishu_role=admin"}).status_code, 401)

    def test_login_redirect_keeps_the_query_and_audio_stays_under_audio_dir(self):
        client = TestClient(app.app)
        for route in ("/?clinic=kem", "/admin?clinic=kem"):
            res = client.get(route, follow_redirects=False)
            self.assertEqual(res.status_code, 302)
            self.assertEqual(unquote(res.headers["location"]), f"/login?next={route}")

        audio_dir = app._audio_dir()
        audio_dir.mkdir(parents=True, exist_ok=True)
        outside = Path(db.DATA_DIR) / "outside_secret.wav"
        outside.write_bytes(b"secret")
        inside = audio_dir / "safe_call.wav"
        inside.write_bytes(b"RIFF")
        try:
            self.assertEqual(app._resolve_audio("safe_call.wav"), inside.resolve())
            self.assertEqual(app._resolve_audio(f"/legacy/dir/{inside.name}"), inside.resolve())
            for bad in (str(outside), "../outside_secret.wav", "", ".."):
                self.assertIsNone(app._resolve_audio(bad))
        finally:
            outside.unlink(missing_ok=True)
            inside.unlink(missing_ok=True)

    def test_static_html_protection_admin_query_redirect_and_passcode_strip(self):
        client = TestClient(app.app)
        for forbidden in ("/static/admin.html", "/static/call.html", "/static/ADMIN.HTML"):
            self.assertEqual(client.get(forbidden).status_code, 404)
        self.assertEqual(client.get("/static/login.html").status_code, 200)

        user = TestClient(app.app)
        self.assertEqual(user.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()}).status_code, 200)
        denied_admin = user.get("/admin?clinic=kem", follow_redirects=False)
        self.assertEqual(denied_admin.status_code, 302)
        self.assertEqual(unquote(denied_admin.headers["location"]), "/login?next=/admin?clinic=kem")

        prev = os.environ.get("ADMIN_PASSCODE")
        try:
            os.environ["ADMIN_PASSCODE"] = "  admin-trimmed  "
            self.assertEqual(app._admin_passcode(), "admin-trimmed")
            admin = TestClient(app.app)
            res = admin.post("/api/login", json={"username": "Ada", "passcode": " admin-trimmed "})
            self.assertEqual((res.status_code, res.json()["role"]), (200, "admin"))
        finally:
            if prev is None:
                os.environ.pop("ADMIN_PASSCODE", None)
            else:
                os.environ["ADMIN_PASSCODE"] = prev


class CaptionAndCopyTests(unittest.TestCase):
    def _emergency_state(self) -> CallState:
        """An emergency close with a name, built only through the public reducer."""
        state = engine.reduce_turn(
            engine.apply_profile_patch(engine.new_state(), ProfilePatch(caller_name="आशा")),
            TurnCommand(emergency=True, utterance="साँस नहीं आ रही"),
        )
        self.assertIs(state.close_reason, CloseReason.EMERGENCY)
        return state
    def test_login_page_hides_passcode_call_page_and_logo(self):
        login_html = (APP_DIR / "static" / "login.html").read_text(encoding="utf-8")
        if app._passcode():
            self.assertNotIn(app._passcode(), login_html)
        if app._admin_passcode():
            self.assertNotIn(app._admin_passcode(), login_html)
        self.assertNotIn("screening", login_html.lower())
        self.assertIn("आपका नाम", login_html)
        self.assertIn("Your name", login_html)

        call_html = (APP_DIR / "static" / "call.html").read_text(encoding="utf-8")
        self.assertIn("कॉल शुरू करने के लिए नीचे दबाएँ", call_html)
        self.assertIn(">शुरू<", call_html)
        self.assertIn('src="/static/shishu-logo.png"', call_html)
        logo = TestClient(app.app).get("/static/shishu-logo.png")
        self.assertEqual(logo.status_code, 200)
        self.assertTrue(logo.headers["content-type"].startswith("image/png"))
        self.assertGreater(len(logo.content), 1000)

    def test_live_prompt_uses_only_the_cursor_tools(self):
        text = copy_hi.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("next_question", text)
        for old_tool in ("save_profile", "record_turn", "record_finding", "end_call"):
            self.assertNotIn(old_tool, text)
        closing = copy_hi.closing_hi(self._emergency_state(), has_clinical_concern=False)
        self.assertIn("108", closing)
        self.assertEqual(gemini_live._session_settings({})["voice"], "Kore")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["gemini_model"],
                         "gemini-3.1-flash-live-preview")


class Stage6BAccessTests(unittest.TestCase):
    def test_user_cannot_fetch_a_pdf_by_call_id(self):
        call_id = db.create_call(caller_name="Pooja")
        db.complete_call(call_id, 30)
        token = db.get_call(call_id)["share_token"]
        user = TestClient(app.app)
        self.assertEqual(user.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()}).status_code, 200)
        self.assertEqual(user.get(f"/api/calls/{call_id}/mother.pdf").status_code, 403)
        self.assertEqual(user.post(f"/api/calls/{call_id}/mother.pdf").status_code, 403)
        self.assertNotEqual(user.get(f"/api/calls/by-token/{token}/mother.pdf").status_code, 403)
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()})
        self.assertNotEqual(admin.get(f"/api/calls/{call_id}/mother.pdf").status_code, 403)

    def test_bug_body_is_capped_and_ownership_uses_the_share_token(self):
        user = TestClient(app.app)
        user.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()})
        self.assertEqual(user.post("/api/bugs", json={"category": "test", "note": "x" * (512 * 1024), "call_id": 0}).status_code, 413)
        filed = user.post("/api/bugs", json={"category": "test", "note": "n" * 2500, "call_id": 0})
        self.assertEqual((filed.status_code, len(filed.json()["bug"]["note"])), (200, 2000))

        call_id = db.create_call(caller_name="Pooja")
        db.append_transcript(call_id, "user", "Secret line from this call.")
        db.complete_call(call_id, 10)
        token = db.get_call(call_id)["share_token"]
        owned = user.post("/api/bugs", json={"category": "test", "note": "mine", "call_id": call_id, "share_token": token})
        stranger = TestClient(app.app)
        stranger.post("/api/login", json={"username": "Pooja", "passcode": app._passcode()})
        peeked = stranger.post("/api/bugs", json={"category": "test", "note": "guess", "call_id": call_id})
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "admin", "passcode": app._admin_passcode()})
        listed = admin.get("/api/bugs").json()
        match = next(item for item in listed if item["id"] == owned.json()["bug"]["id"])
        self.assertIn("Secret line from this call.", [row.get("text") for row in match["snapshot"]["last_transcript"]])
        other = next(item for item in listed if item["id"] == peeked.json()["bug"]["id"])
        self.assertEqual(other["snapshot"]["last_transcript"], [])

    def test_failed_logins_and_token_pdf_rebuilds_are_throttled(self):
        app._reset_rate_limits()
        try:
            client = TestClient(app.app)
            for _ in range(app._LOGIN_FAILS):
                self.assertEqual(client.post("/api/login", json={"username": "Pooja", "passcode": "wrong"}).status_code, 401)
            self.assertEqual(client.post("/api/login", json={"username": "Pooja", "passcode": "wrong"}).status_code, 429)

            call_id = db.create_call(caller_name="Pooja")
            db.complete_call(call_id, 12)
            token = db.get_call(call_id)["share_token"]
            for _ in range(app._PDF_REBUILDS):
                app._rate_hit(app._pdf_rebuilds, token, app._PDF_WINDOW_SEC)
            self.assertEqual(client.post(f"/api/calls/by-token/{token}/mother.pdf").status_code, 429)
            self.assertEqual(client.post(f"/api/calls/by-token/{token}%20/mother.pdf").status_code, 429)

            for _ in range(app._SHARE_EVENT_LIMIT):
                self.assertEqual(client.post(f"/api/calls/by-token/{token}/events", json={"event": "share_clicked"}).status_code, 200)
            self.assertEqual(client.post(f"/api/calls/by-token/{token}%20/events", json={"event": "share_clicked"}).status_code, 429)

            bucket = {"expired": [1.0, 2.0], "active": [95.0]}
            app._prune_hits(bucket, 100.0, 60.0)
            self.assertEqual(bucket, {"active": [95.0]})
            app._prune_hits(bucket, 200.0, 60.0)
            self.assertEqual(bucket, {})
            self.assertFalse(app._rate_limited(bucket, "never-seen-ip", 5, 60.0))
            self.assertEqual(bucket, {})
        finally:
            app._reset_rate_limits()

    def test_report_inflight_clears_and_seed_audio_is_stereo_16k(self):
        state = engine.reduce_turn(
            engine.apply_profile_patch(
                engine.new_state(),
                ProfilePatch(caller_name="Pooja", marital_status=MaritalStatus.UNMARRIED),
            ),
            TurnCommand(caller_stop=True),
        )
        self.assertIs(state.close_reason, CloseReason.CALLER_STOP)
        state = state.model_copy(update={"phase": Phase.CLOSING, "close_reason": CloseReason.COMPLETE})
        call_id = db.create_call(caller_name="Pooja")
        db.save_cursor_state(call_id, state)
        db.complete_call(call_id, 20)

        for _ in range(2):
            pdf_bytes, _ = asyncio.run(report_service.mother_pdf(call_id))
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))
            self.assertEqual(report_service._inflight, {})

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
            self.assertEqual((full["cursor_state"]["phase"], full["cursor_state"]["close_reason"]), ("closing", "complete"))
            self.assertTrue(full["summary_mother"] and full["summary_doctor"])
            with wave.open(os.path.join(db.DATA_DIR, "audio", full["audio_path"])) as handle:
                self.assertEqual((handle.getnchannels(), handle.getframerate(), handle.getsampwidth()), (2, 16000, 2))
            get_pdf = admin.get(f"/api/calls/{row['id']}/mother.pdf")
            by_token = admin.get(f"/api/calls/by-token/{full['share_token']}/mother.pdf")
            rebuilt = admin.post(f"/api/calls/{row['id']}/mother.pdf")
            after_rebuild = admin.get(f"/api/calls/{row['id']}/mother.pdf")
            self.assertEqual(get_pdf.status_code, 200)
            self.assertTrue(get_pdf.content.startswith(b"%PDF"))
            self.assertEqual(by_token.content, get_pdf.content)
            self.assertTrue(rebuilt.content.startswith(b"%PDF"))
            self.assertEqual(after_rebuild.content, rebuilt.content)
            by_name[full["caller_name"]] = full

        self.assertEqual(set(by_name), {"शिल्पा", "गीता"})
        shilpa = db.call_state_from_row(by_name["शिल्पा"])
        self.assertTrue(by_name["शिल्पा"]["red_flags"])
        for qid in ("birth_jaundice", "body_joint_swelling", "preg_conditions", "fam_infant_death"):
            self.assertIs(shilpa.answers[qid].state, AnswerStatus.CONCERN)
        jaundice = shilpa.answers["birth_jaundice"]
        self.assertIn("तीन दिन नीली लाइट में रखा", jaundice.followups["jaundice_days"].note.verbatim)
        infant = shilpa.answers["fam_infant_death"]
        self.assertIn("5 महीने का बेटा", infant.followups["infant_death_age"].note.verbatim)
        self.assertIn("झटके आने से", infant.followups["infant_death_cause"].note.verbatim)

        geeta_call = by_name["गीता"]
        geeta = db.call_state_from_row(geeta_call)
        self.assertEqual(geeta_call["red_flags"], [])
        self.assertTrue(all(
            record.state in (AnswerStatus.CLEAR, AnswerStatus.NOT_APPLICABLE)
            for record in geeta.answers.values()
        ))
        # The seeded walk asks the plain thalassemia question once and never explains it
        # with a disease word, so the transcript carries no caption leak.
        asked = [t["text"] for t in geeta_call["transcript"] if t["role"] == "assistant"]
        self.assertEqual(len([t for t in asked if "थैलेसीमिया" in t]), 1)
        for leak in ("transfusion", "baar-baar", "Duchenne", "Hemophilia", "ड्यूशेन", "हीमोफीलिया"):
            self.assertFalse(any(leak.lower() in t.lower() for t in asked), leak)


if __name__ == "__main__":
    unittest.main()
