"""End-to-end HTTP and socket checks for the V3 intake app. No network calls."""

import os
import unittest
from unittest import mock

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`

from fastapi.testclient import TestClient  # noqa: E402

from intake import app as intake_app  # noqa: E402
from intake import db, live, reports  # noqa: E402

USER_PASS = os.environ["APP_PASSCODE"]
ADMIN_PASS = os.environ["ADMIN_PASSCODE"]


class AppTestCase(unittest.TestCase):
    """One client and empty rate-limit windows per test, over the shared seeded database."""

    def setUp(self) -> None:
        for bucket in intake_app.HITS.values():
            bucket.clear()
        self.client = TestClient(intake_app.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def sign_in(self, passcode: str = USER_PASS, username: str = "Tester") -> None:
        res = self.client.post("/api/login", json={"username": username, "passcode": passcode})
        self.assertEqual(res.status_code, 200, res.text)

    def cookie(self) -> str:
        return self.client.cookies[intake_app.COOKIE]

    def seeded_call(self, pack_id: str = "shishu-hi") -> dict:
        """One seeded, finished call with its share token already minted."""
        return db.get_call(int(db.list_calls(pack_id)[0]["id"]))


class HealthAndPageTests(AppTestCase):
    def test_health_is_public_and_reports_the_build_stamp(self):
        payload = self.client.get("/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["build"], reports.build_info()["tag"])
        self.assertTrue(payload["build"].startswith(reports.build_info()["env"]))

    def test_pages_redirect_to_login_and_keep_the_query(self):
        login = self.client.get("/login")
        self.assertEqual(login.status_code, 200)
        self.assertIn("SHISHU", login.text)

        call_page = self.client.get("/?pack=fever-opd-hi&clinic=opd", follow_redirects=False)
        self.assertEqual(call_page.status_code, 302)
        location = call_page.headers["location"]
        self.assertTrue(location.startswith("/login?next="))
        self.assertIn("pack%3Dfever-opd-hi", location)
        self.assertIn("clinic%3Dopd", location)

        self.assertEqual(self.client.get("/admin", follow_redirects=False).status_code, 302)

    def test_signed_in_user_reaches_the_call_page(self):
        self.sign_in()
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/admin", follow_redirects=False).status_code, 302)

    def test_admin_reaches_the_dashboard(self):
        self.sign_in(ADMIN_PASS, "Ada")
        res = self.client.get("/admin")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Clinical Dashboard", res.text)

    def test_static_serves_only_css_and_js(self):
        for asset in ("styles.css", "call.js", "admin.js"):
            self.assertEqual(self.client.get(f"/static/{asset}").status_code, 200, asset)
        self.assertEqual(self.client.get("/static/call.html").status_code, 404)
        self.assertEqual(self.client.get("/static/missing.js").status_code, 404)


class AuthCookieTests(AppTestCase):
    def test_admin_passcode_sets_a_signed_httponly_cookie(self):
        res = self.client.post("/api/login", json={"username": "Ada", "passcode": ADMIN_PASS})
        self.assertEqual(res.json(), {"ok": True, "role": "admin", "username": "Ada"})
        cookie = self.cookie()
        self.assertTrue(cookie.startswith("v1.admin."), cookie)
        self.assertNotIn(ADMIN_PASS, cookie)
        for flag in ("HttpOnly", "SameSite=lax", "Max-Age=1209600", "Path=/"):
            self.assertIn(flag, res.headers["set-cookie"], flag)

    def test_user_passcode_gets_the_user_role(self):
        self.sign_in(USER_PASS, "Mira")
        self.assertEqual(self.client.get("/api/pack").json()["role"], "user")
        self.assertEqual(self.client.get("/api/calls").status_code, 403)

    def test_tampered_and_forged_cookies_are_rejected(self):
        self.sign_in()
        good = self.cookie()
        for bad in (good[:-4] + "0000", "user", "v1.user.9999999999.QQ.zz", "v1.admin.1.QQ.zz"):
            self.client.cookies.pop(intake_app.COOKIE, None)
            self.client.cookies.set(intake_app.COOKIE, bad)
            self.assertEqual(self.client.get("/api/calls").status_code, 401, bad)
        self.client.cookies.set(intake_app.COOKIE, good)
        self.assertEqual(self.client.get("/api/pack").json()["username"], "Tester")

    def test_an_expired_cookie_is_rejected(self):
        self.client.cookies.set(intake_app.COOKIE, "v1.admin.1.Ada.deadbeef")
        self.assertEqual(self.client.get("/api/calls").status_code, 401)

    def test_logout_clears_the_cookie(self):
        self.sign_in()
        self.assertEqual(self.client.post("/api/logout").json(), {"ok": True})
        self.assertEqual(self.client.get("/api/calls").status_code, 401)

    def test_bad_and_blank_passcodes_never_sign_in(self):
        for body in ({"username": "x", "passcode": "nope"}, {"passcode": ""}, {}):
            self.assertEqual(self.client.post("/api/login", json=body).status_code, 401, body)
        self.assertEqual(self.client.post("/api/login", content=b"not json").status_code, 401)


class RateLimitTests(AppTestCase):
    def test_login_failures_are_capped_per_ip(self):
        limit = intake_app.LIMITS["login"][0]
        codes = [self.client.post("/api/login", json={"passcode": "wrong"}).status_code
                 for _ in range(limit + 1)]
        self.assertEqual(codes[:limit], [401] * limit)
        self.assertEqual(codes[-1], 429)

    def test_a_good_passcode_still_works_under_the_limit(self):
        self.sign_in()
        self.assertEqual(self.client.get("/api/pack").json()["role"], "user")

    def test_pdf_rebuilds_are_capped_per_token(self):
        call = self.seeded_call("fever-opd-hi")
        url = f"/api/calls/by-token/{call['share_token']}/patient.pdf"
        limit = intake_app.LIMITS["pdf"][0]
        codes = [self.client.post(url).status_code for _ in range(limit + 1)]
        self.assertEqual(codes[:limit], [200] * limit)
        self.assertEqual(codes[-1], 429)

    def test_share_events_are_capped_per_token(self):
        call = self.seeded_call()
        url = f"/api/calls/by-token/{call['share_token']}/events"
        limit = intake_app.LIMITS["share"][0]
        codes = [self.client.post(url, json={"event": "share_clicked"}).status_code
                 for _ in range(limit + 1)]
        self.assertEqual(codes[:limit], [200] * limit)
        self.assertEqual(codes[-1], 429)


class PackTests(AppTestCase):
    def test_shishu_pack_metadata(self):
        payload = self.client.get("/api/pack").json()
        self.assertEqual(payload["id"], "shishu-hi")
        self.assertEqual(payload["title"], "SHISHU")
        self.assertEqual(payload["subtitle_native"], "शिशु")
        self.assertEqual(payload["role"], "")
        self.assertEqual(payload["username"], "")
        self.assertEqual(payload["assets_url"], {
            "logo": "/api/pack-asset/logo?pack=shishu-hi",
            "consent_pdf": "/api/pack-asset/consent_pdf?pack=shishu-hi",
        })
        self.assertIn("login", payload["ui"])

    def test_fever_pack_metadata(self):
        payload = self.client.get("/api/pack", params={"pack": "fever-opd-hi"}).json()
        self.assertEqual((payload["id"], payload["title"]), ("fever-opd-hi", "SHISHU OPD"))
        self.assertEqual(payload["assets_url"]["logo"],
                         "/api/pack-asset/logo?pack=fever-opd-hi")

    def test_available_packs_lists_both(self):
        packs = self.client.get("/api/pack").json()["available_packs"]
        self.assertEqual([p["id"] for p in packs], ["fever-opd-hi", "shishu-hi"])
        self.assertEqual({p["id"]: p["title"] for p in packs},
                         {"fever-opd-hi": "SHISHU OPD", "shishu-hi": "SHISHU"})

    def test_pack_reports_the_signed_in_identity(self):
        self.sign_in(ADMIN_PASS, "Ada")
        payload = self.client.get("/api/pack").json()
        self.assertEqual((payload["role"], payload["username"]), ("admin", "Ada"))

    def test_unknown_pack_is_404(self):
        self.assertEqual(self.client.get("/api/pack", params={"pack": "nope"}).status_code, 404)

    def test_logo_and_consent_pdf_for_both_packs(self):
        for pack_id in ("shishu-hi", "fever-opd-hi"):
            logo = self.client.get("/api/pack-asset/logo", params={"pack": pack_id})
            self.assertEqual(logo.status_code, 200, pack_id)
            self.assertEqual(logo.headers["content-type"], "image/png")
            self.assertTrue(logo.content.startswith(b"\x89PNG"))

            consent = self.client.get("/api/pack-asset/consent_pdf", params={"pack": pack_id})
            self.assertEqual(consent.status_code, 200, pack_id)
            self.assertEqual(consent.headers["content-type"], "application/pdf")
            self.assertTrue(consent.content.startswith(b"%PDF"))

    def test_unknown_asset_kinds_and_packs_are_404(self):
        self.assertEqual(self.client.get("/api/pack-asset/font").status_code, 404)
        self.assertEqual(self.client.get("/api/pack-asset/logo?pack=nope").status_code, 404)
        self.assertEqual(self.client.get("/api/pack-asset/logo?pack=../packs").status_code, 404)


class CallAdminTests(AppTestCase):
    def test_calls_need_an_admin(self):
        self.assertEqual(self.client.get("/api/calls").status_code, 401)
        self.sign_in(USER_PASS)
        self.assertEqual(self.client.get("/api/calls").status_code, 403)

    def test_admin_list_filters_by_pack(self):
        self.sign_in(ADMIN_PASS, "Ada")
        everything = self.client.get("/api/calls").json()
        self.assertGreaterEqual(len(everything), 6)
        fever = self.client.get("/api/calls", params={"pack": "fever-opd-hi"}).json()
        self.assertTrue(fever)
        self.assertEqual({c["pack_id"] for c in fever}, {"fever-opd-hi"})
        self.assertEqual(len(fever), 3)

    def test_one_call_by_id(self):
        self.sign_in(ADMIN_PASS, "Ada")
        call = self.seeded_call()
        res = self.client.get(f"/api/calls/{call['id']}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], call["id"])
        self.assertEqual(self.client.get("/api/calls/999999").status_code, 404)

    def test_patient_pdf_by_id_reads_and_rebuilds(self):
        self.sign_in(ADMIN_PASS, "Ada")
        call = self.seeded_call()
        for method in (self.client.post, self.client.get):
            res = method(f"/api/calls/{call['id']}/patient.pdf")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers["content-type"], "application/pdf")
            self.assertIn("attachment", res.headers["content-disposition"])
            self.assertTrue(res.content.startswith(b"%PDF"))

    def test_audio_is_404_until_a_call_records_one(self):
        self.sign_in(ADMIN_PASS, "Ada")
        call = self.seeded_call()
        self.assertEqual(self.client.get(f"/api/audio/{call['id']}").status_code, 404)

    def test_audio_needs_an_admin(self):
        call = self.seeded_call()
        self.assertEqual(self.client.get(f"/api/audio/{call['id']}").status_code, 401)


class TokenPdfTests(AppTestCase):
    def test_share_token_serves_the_pdf_without_a_login(self):
        call = self.seeded_call("fever-opd-hi")
        res = self.client.get(f"/api/calls/by-token/{call['share_token']}/patient.pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertTrue(res.content.startswith(b"%PDF"))
        self.assertIn("SHISHU", res.headers["content-disposition"])

    def test_rebuild_then_read(self):
        call = self.seeded_call()
        url = f"/api/calls/by-token/{call['share_token']}/patient.pdf"
        self.assertEqual(self.client.post(url).status_code, 200)
        self.assertEqual(self.client.get(url).status_code, 200)

    def test_unknown_tokens_are_404(self):
        for token in ("short", "nope-nope-nope", "x" * 40):
            res = self.client.get(f"/api/calls/by-token/{token}/patient.pdf")
            self.assertEqual(res.status_code, 404, token)

    def test_share_event_accepts_only_share_clicked(self):
        call = self.seeded_call()
        url = f"/api/calls/by-token/{call['share_token']}/events"
        self.assertEqual(self.client.post(url, json={"event": "share_clicked"}).json(), {"ok": True})
        self.assertEqual(self.client.post(url, json={"event": "download"}).status_code, 400)
        self.assertEqual(self.client.post(url, json={}).status_code, 400)

    def test_share_event_on_an_unknown_token_is_404(self):
        self.assertEqual(self.client.post("/api/calls/by-token/nope-nope/events",
                                          json={"event": "share_clicked"}).status_code, 404)


class BugReportTests(AppTestCase):
    def test_only_an_admin_reads_the_list(self):
        self.sign_in(USER_PASS)
        self.assertEqual(self.client.get("/api/bugs").status_code, 403)
        self.sign_in(ADMIN_PASS, "Ada")
        self.assertIsInstance(self.client.get("/api/bugs").json(), list)

    def test_the_owner_gets_the_call_snapshot(self):
        call = self.seeded_call()
        self.sign_in(USER_PASS, "Mira")
        res = self.client.post("/api/bugs", json={
            "call_id": call["id"], "share_token": call["share_token"],
            "category": "Slow response", "note": "  धीमा  ", "call_elapsed_sec": 42,
        })
        self.assertEqual(res.status_code, 200)
        bug = res.json()["bug"]
        self.assertEqual((bug["reporter"], bug["note"]), ("Mira", "  धीमा  "))
        self.assertTrue(bug["snapshot"]["last_transcript"])
        self.assertTrue(bug["snapshot"]["recent_events"])

    def test_a_stranger_gets_no_call_body(self):
        call = self.seeded_call()
        self.sign_in(USER_PASS, "Mira")
        bug = self.client.post("/api/bugs", json={
            "call_id": call["id"], "share_token": "not-the-real-token",
            "category": "Other", "note": "peek",
        }).json()["bug"]
        self.assertEqual(bug["snapshot"]["last_transcript"], [])
        self.assertEqual(bug["snapshot"]["recent_events"], [])

    def test_admin_owns_every_call(self):
        call = self.seeded_call()
        self.sign_in(ADMIN_PASS, "Ada")
        bug = self.client.post("/api/bugs", json={
            "call_id": call["id"], "category": "Other", "note": "cli",
        }).json()["bug"]
        self.assertTrue(bug["snapshot"]["last_transcript"])

    def test_a_call_less_report_is_recorded(self):
        self.sign_in()
        res = self.client.post("/api/bugs", json={"category": "Other", "note": "no call"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["bug"]["call_id"], 0)

    def test_reports_need_a_json_object_and_stay_under_the_cap(self):
        self.sign_in()
        self.assertEqual(self.client.post("/api/bugs", content=b"[]").status_code, 400)
        self.assertEqual(self.client.post("/api/bugs", content=b"nonsense").status_code, 400)
        self.assertEqual(self.client.post("/api/bugs", content=b"").status_code, 400)
        oversized = b'{"note":"' + b"x" * (intake_app.BUG_CAP + 1) + b'"}'
        self.assertEqual(self.client.post("/api/bugs", content=oversized).status_code, 413)

    def test_bugs_need_a_session(self):
        call = self.seeded_call()
        self.assertEqual(self.client.post("/api/bugs", json={"call_id": call["id"]}).status_code, 401)


class LiveSocketTests(AppTestCase):
    def close_code(self, query: str) -> int:
        with self.client.websocket_connect(f"/ws/live?{query}") as ws:
            return ws.receive().get("code")

    def test_unauthenticated_socket_closes_with_1008(self):
        self.assertEqual(self.close_code("pack=shishu-hi"), 1008)

    def test_unknown_pack_closes_with_1008(self):
        self.sign_in()
        self.assertEqual(self.close_code("pack=nope"), 1008)

    def test_clinic_code_keeps_a_slug_and_falls_back_to_the_default(self):
        with mock.patch.dict(os.environ, {"DEFAULT_CLINIC_CODE": "clinic-a"}):
            self.assertEqual(intake_app._clinic_code("opd_2"), "opd_2")
            self.assertEqual(intake_app._clinic_code("OPD-2"), "opd-2")
            for bad in ("", "   ", "../etc/passwd", "opd 2", "a" * 41, "opd/2"):
                self.assertEqual(intake_app._clinic_code(bad), "clinic-a", bad)
        with mock.patch.dict(os.environ, {"DEFAULT_CLINIC_CODE": "../bad"}):
            self.assertEqual(intake_app._clinic_code("../also-bad"), "", "an unsafe default is dropped too")

    def test_the_socket_is_handed_the_validated_clinic_code(self):
        self.sign_in()
        seen = {}

        async def capture(ws, **kwargs):
            seen.update(kwargs)

        with mock.patch.object(live, "handle_live_ws", capture), \
                mock.patch.dict(os.environ, {"DEFAULT_CLINIC_CODE": "clinic-a"}):
            with self.client.websocket_connect("/ws/live?pack=shishu-hi&clinic=../evil"):
                pass
            self.assertEqual(seen["clinic_code"], "clinic-a")
            with self.client.websocket_connect("/ws/live?pack=shishu-hi&clinic=opd_2"):
                pass
            self.assertEqual(seen["clinic_code"], "opd_2")


if __name__ == "__main__":
    unittest.main()
