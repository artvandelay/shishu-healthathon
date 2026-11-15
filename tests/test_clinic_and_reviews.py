"""Stage 4: static assets, clinic tag, share event, and clinician review."""

import asyncio
import unittest

from tests import APP_DIR
from tests.helpers import FakeWebSocket

import app  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from live_fsm import CLOSING_AUDIO_GRACE_SEC, TurnPhase, WatchdogAction  # noqa: E402

_STATIC = APP_DIR / "static"


class Stage4AssetTests(unittest.TestCase):
    def test_consent_pdf_and_page_scripts_are_served(self):
        call_html = (_STATIC / "call.html").read_text(encoding="utf-8")
        admin_html = (_STATIC / "admin.html").read_text(encoding="utf-8")
        self.assertIn('href="/static/SHISHU-consent.pdf"', call_html)
        self.assertIn('src="/static/call.js"', call_html)
        self.assertIn('src="/static/admin.js"', admin_html)
        self.assertIn('id="clinic-filter"', admin_html)
        self.assertIn('data-rating="missed_flag"', admin_html)
        self.assertIn('id="save-review-note"', admin_html)
        self.assertNotIn("<script>", call_html)
        self.assertNotIn("<script>", admin_html)

        client = TestClient(app.app)
        pdf = client.get("/static/SHISHU-consent.pdf")
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.headers["content-type"].startswith("application/pdf"))
        self.assertTrue(pdf.content.startswith(b"%PDF-"))

        for asset in ("/static/call.js", "/static/admin.js"):
            res = client.get(asset)
            self.assertEqual(res.status_code, 200)
            self.assertIn("javascript", res.headers["content-type"])
        self.assertEqual(client.get("/static/missing-asset.js").status_code, 404)

        login_html = (_STATIC / "login.html").read_text(encoding="utf-8")
        self.assertIn("Too many attempts. Wait a minute and try again.", login_html)
        self.assertIn("new URL(", login_html)

    def test_ending_fires_once_when_closing_audio_finishes(self):
        live = gemini_live.LiveCallSession(ws=FakeWebSocket())
        now = 1000.0
        live.fsm.mark_closing(now)
        live.fsm.closing_heard = True
        live.fsm.playback_ends_at = now
        live.fsm.phase = TurnPhase.BOT_SPEAKING
        poll, act = live.fsm.poll_watchdog, WatchdogAction
        self.assertIs(poll(live.state, now, 8.0), act.NONE)
        self.assertIs(poll(live.state, now + 1.0, 8.0), act.NONE)
        live.fsm.phase = TurnPhase.CLOSING
        grace = now + CLOSING_AUDIO_GRACE_SEC
        self.assertIs(poll(live.state, grace - 0.05, 8.0), act.NONE)
        done = poll(live.state, grace, 8.0)
        self.assertIs(done, act.CLOSING_COMPLETE)

        async def once():
            ws = FakeWebSocket()
            live2 = gemini_live.LiveCallSession(ws=ws)
            self.assertFalse(await live2._watchdog_step(done))
            self.assertFalse(await live2._watchdog_step(done))
            await live2._finalize()
            self.assertTrue(live2.fsm.ending_notified)
            return [m for m in ws.sent if m.get("type") == "ending"]

        self.assertEqual(asyncio.run(once()), [{"type": "ending"}])


class Stage4LoopTests(unittest.TestCase):
    def test_clinic_slug_and_list_round_trip(self):
        self.assertEqual(app._clinic_slug("KEM"), "kem")
        self.assertEqual(app._clinic_slug("kem"), "kem")
        self.assertEqual(app._clinic_slug("../kem"), "")
        self.assertEqual(app._clinic_slug(""), "")
        call_id = db.create_call(caller_name="पूजा", clinic_code="kem")
        row = next(item for item in db.list_calls() if item["id"] == call_id)
        self.assertEqual(row["clinic_code"], "kem")
        self.assertEqual(row["clinician_review"], {})

    def test_share_clicked_is_logged_by_token(self):
        call_id = db.create_call(caller_name="पूजा", clinic_code="kem")
        token = db.get_call(call_id)["share_token"]
        client = TestClient(app.app)
        self.assertEqual(client.post("/api/calls/by-token/not-a-real-token/events", json={"event": "share_clicked"}).status_code, 404)
        self.assertEqual(client.post(f"/api/calls/by-token/{token}/events", json={"event": "other"}).status_code, 400)
        logged = client.post(f"/api/calls/by-token/{token}/events", json={"event": "share_clicked"})
        self.assertEqual(logged.status_code, 200)
        self.assertEqual(logged.json()["ok"], True)
        self.assertEqual([item["type"] for item in db.get_call(call_id)["events"]], ["share_clicked"])

    def test_clinician_review_is_admin_only_and_round_trips(self):
        call_id = db.create_call(caller_name="पूजा", clinic_code="kem")
        anon = TestClient(app.app)
        self.assertEqual(anon.post(f"/api/calls/{call_id}/review", json={"rating": "useful", "note": "no"}).status_code, 401)
        user = TestClient(app.app)
        user.post("/api/login", json={"username": "Mother", "passcode": app._passcode()})
        self.assertEqual(user.post(f"/api/calls/{call_id}/review", json={"rating": "useful", "note": "no"}).status_code, 403)
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "Ada", "passcode": app._admin_passcode()})
        self.assertEqual(admin.post(f"/api/calls/{call_id}/review", json={"rating": "great"}).status_code, 400)
        saved = admin.post(f"/api/calls/{call_id}/review", json={"rating": "missed_flag", "note": "joint swelling"})
        self.assertEqual(saved.status_code, 200)
        body = saved.json()["clinician_review"]
        self.assertEqual((body["rating"], body["note"]), ("missed_flag", "joint swelling"))
        listed = next(item for item in admin.get("/api/calls").json() if item["id"] == call_id)
        self.assertEqual(listed["clinic_code"], "kem")
        self.assertEqual(listed["clinician_review"]["rating"], "missed_flag")
        self.assertEqual(admin.get(f"/api/calls/{call_id}").json()["clinician_review"]["note"], "joint swelling")
        self.assertEqual(admin.post("/api/calls/999999/review", json={"rating": "useful"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
