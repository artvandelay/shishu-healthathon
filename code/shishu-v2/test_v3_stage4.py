"""Stage 4: static assets, clinic tag, share event, and clinician review."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="shishu-v3-stage4-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import app  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import prompts  # noqa: E402
import test_v3_cursor as cursor_tests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"


class _FakeSocket:
    def __init__(self):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class Stage4AssetTests(unittest.TestCase):
    def test_consent_pdf_and_page_scripts_are_served(self):
        call_html = (_STATIC / "call.html").read_text(encoding="utf-8")
        admin_html = (_STATIC / "admin.html").read_text(encoding="utf-8")
        self.assertIn('href="/static/SHISHU-consent.pdf"', call_html)
        self.assertIn('src="/static/call.js"', call_html)
        self.assertIn('src="/static/admin.js"', admin_html)
        self.assertIn('id="clinic-filter"', admin_html)
        self.assertIn('data-rating="missed_flag"', admin_html)
        self.assertNotIn("<script>", call_html)
        self.assertNotIn("<script>", admin_html)

        client = TestClient(app.app)
        pdf = client.get("/static/SHISHU-consent.pdf")
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.headers["content-type"].startswith("application/pdf"))
        self.assertTrue(pdf.content.startswith(b"%PDF-"))

        call_js = client.get("/static/call.js")
        self.assertEqual(call_js.status_code, 200)
        self.assertIn("javascript", call_js.headers["content-type"])
        script = call_js.text
        self.assertIn("sessionStorage", script)
        self.assertIn('get("clinic")', script)
        self.assertIn("?clinic=", script)
        self.assertIn('wakeLock.request("screen")', script)
        self.assertIn("share_clicked", script)
        self.assertIn("/api/calls/by-token/", script)
        self.assertNotIn("muteFrame", script)
        self.assertNotIn("0.075", script)

        admin_js = client.get("/static/admin.js")
        self.assertEqual(admin_js.status_code, 200)
        self.assertIn("javascript", admin_js.headers["content-type"])
        dashboard = admin_js.text
        self.assertIn("clinic-filter", dashboard)
        self.assertIn("/review", dashboard)
        self.assertIn('getAttribute("data-rating")', dashboard)
        self.assertNotIn("function applySettings", dashboard)
        self.assertNotIn("function saveSettings", dashboard)
        self.assertNotIn("function toggleSettings", dashboard)
        self.assertFalse(hasattr(prompts, "DOCTOR_MOTHER_SUMMARY_PROMPT"))
        self.assertEqual(client.get("/static/missing-asset.js").status_code, 404)

    def test_multi_child_hindi_is_inside_the_question(self):
        kids = [cursor_tests._child(6, "male", 1), cursor_tests._child(2, "female", 2)]
        state = cursor_tests._profile(children=kids)
        state = cursor_tests._clear_until(state, "gate_birth_neonatal")
        spoken = gemini_live._say_hindi(state)
        self.assertIn("आपके किसी भी बच्चे", spoken)
        self.assertNotIn(" Say ", spoken)
        state = cursor_tests._clear_until(state, "gate_bleeding_joints")
        bleed = gemini_live._say_hindi(state)
        self.assertTrue(bleed.startswith("आपके किसी भी बच्चे"))
        one = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        one = cursor_tests._clear_until(one, "gate_birth_neonatal")
        alone = gemini_live._say_hindi(one)
        self.assertNotIn("आपके किसी भी बच्चे", alone)
        self.assertIn("किसी बच्चे", alone)

    def test_ending_fires_once_when_closing_audio_finishes(self):
        state = gemini_live._TurnState()
        now = 1000.0
        telemetry = {"closing_heard": True}
        state.bot_speaking = True
        state.playback_ends_at = now
        self.assertFalse(gemini_live._closing_audio_finished(telemetry, state, now))
        state.bot_speaking = False
        self.assertFalse(gemini_live._closing_audio_finished(telemetry, state, now))
        self.assertTrue(
            gemini_live._closing_audio_finished(
                telemetry,
                state,
                now + gemini_live._CLOSING_AUDIO_GRACE_SEC,
            )
        )

        async def once():
            ws = _FakeSocket()
            box = {}
            await gemini_live._notify_ending(ws, asyncio.Lock(), box)
            await gemini_live._notify_ending(ws, asyncio.Lock(), box)
            return ws.sent

        self.assertEqual(asyncio.run(once()), [{"type": "ending"}])


class Stage4LoopTests(unittest.TestCase):
    def test_clinic_slug_and_list_round_trip(self):
        self.assertEqual(app._clinic_slug("KEM"), "kem")
        self.assertEqual(app._clinic_slug("kem"), "kem")
        self.assertEqual(app._clinic_slug("../kem"), "")
        self.assertEqual(app._clinic_slug(""), "")
        call_id = db.create_call("gemini", caller_name="पूजा", clinic_code="kem")
        row = next(item for item in db.list_calls() if item["id"] == call_id)
        self.assertEqual(row["clinic_code"], "kem")
        self.assertEqual(row["clinician_review"], {})

    def test_share_clicked_is_logged_by_token(self):
        call_id = db.create_call("gemini", caller_name="पूजा", clinic_code="kem")
        token = db.get_call(call_id)["share_token"]
        client = TestClient(app.app)
        missing = client.post(
            "/api/calls/by-token/not-a-real-token/events",
            json={"event": "share_clicked"},
        )
        self.assertEqual(missing.status_code, 404)
        rejected = client.post(
            f"/api/calls/by-token/{token}/events",
            json={"event": "other"},
        )
        self.assertEqual(rejected.status_code, 400)
        logged = client.post(
            f"/api/calls/by-token/{token}/events",
            json={"event": "share_clicked"},
        )
        self.assertEqual(logged.status_code, 200)
        self.assertEqual(logged.json()["ok"], True)
        events = db.get_call(call_id)["events"]
        self.assertEqual([item["type"] for item in events], ["share_clicked"])

    def test_clinician_review_is_admin_only_and_round_trips(self):
        call_id = db.create_call("gemini", caller_name="पूजा", clinic_code="kem")
        anon = TestClient(app.app)
        self.assertEqual(
            anon.post(
                f"/api/calls/{call_id}/review",
                json={"rating": "useful", "note": "no"},
            ).status_code,
            401,
        )
        user = TestClient(app.app)
        user.post("/api/login", json={"username": "Mother", "passcode": app._passcode()})
        self.assertEqual(
            user.post(
                f"/api/calls/{call_id}/review",
                json={"rating": "useful", "note": "no"},
            ).status_code,
            403,
        )
        admin = TestClient(app.app)
        admin.post("/api/login", json={"username": "Ada", "passcode": app._admin_passcode()})
        bad = admin.post(f"/api/calls/{call_id}/review", json={"rating": "great"})
        self.assertEqual(bad.status_code, 400)
        saved = admin.post(
            f"/api/calls/{call_id}/review",
            json={"rating": "missed_flag", "note": "joint swelling"},
        )
        self.assertEqual(saved.status_code, 200)
        body = saved.json()["clinician_review"]
        self.assertEqual(body["rating"], "missed_flag")
        self.assertEqual(body["note"], "joint swelling")
        listed = next(item for item in admin.get("/api/calls").json() if item["id"] == call_id)
        self.assertEqual(listed["clinic_code"], "kem")
        self.assertEqual(listed["clinician_review"]["rating"], "missed_flag")
        detail = admin.get(f"/api/calls/{call_id}").json()
        self.assertEqual(detail["clinician_review"]["note"], "joint swelling")
        self.assertEqual(admin.post("/api/calls/999999/review", json={"rating": "useful"}).status_code, 404)
