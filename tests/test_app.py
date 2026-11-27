"""End-to-end HTTP and socket checks for the V3 intake app. No network calls."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`

from fastapi.testclient import TestClient  # noqa: E402

from intake import app as intake_app  # noqa: E402
from intake import db, live, reports  # noqa: E402
from intake.pack import load_pack  # noqa: E402

USER_PASS = os.environ["APP_PASSCODE"]
ADMIN_PASS = os.environ["ADMIN_PASSCODE"]


_SEEDED = False


def seed_once() -> None:
    """Seed the demo calls exactly once for the whole module.

    Seeding is no longer a side effect of the app lifespan (it is gated behind `SEED_DEMO_CALLS`), so the
    tests that need finished calls ask for them here instead of hoping the boot path did it.

    `reports.ensure_seeded` only fills a pack whose table is *empty*, so a module that runs first and leaves a
    single call behind would silently give this module a partial set (4 instead of 6 seeded calls). The demo rows
    are fixtures of this module alone, so they are re-created from scratch once.
    """
    global _SEEDED
    if _SEEDED:
        return
    with db._LOCK:
        db.conn.execute("DELETE FROM calls")
        db.conn.commit()
    reports.ensure_seeded()
    _SEEDED = bool(db.list_calls())


class AppTestCase(unittest.TestCase):
    """One client and empty rate-limit windows per test, over the shared seeded database."""

    def setUp(self) -> None:
        seed_once()
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

    def test_static_serves_only_css_and_js(self):
        for asset in ("styles.css", "call.js", "admin.js"):
            self.assertEqual(self.client.get(f"/static/{asset}").status_code, 200, asset)
        self.assertEqual(self.client.get("/static/call.html").status_code, 404)
        self.assertEqual(self.client.get("/static/missing.js").status_code, 404)


class AuthCookieTests(AppTestCase):

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

    def test_a_spoofed_leftmost_x_forwarded_for_cannot_dodge_the_login_cap(self):
        # Railway's edge appends the real client IP on the right of X-Forwarded-For; anything the browser
        # sends is on the left. Reading the leftmost entry let an attacker mint a fresh rate-limit bucket
        # per request just by varying the header, so the cap below must key off the rightmost entry.
        limit = intake_app.LIMITS["login"][0]
        codes = [self.client.post(
            "/api/login", json={"passcode": "wrong"},
            headers={"x-forwarded-for": f"10.0.{i}.{i}, 203.0.113.9"}).status_code
            for i in range(1, limit + 2)]
        self.assertEqual(codes[:limit], [401] * limit)
        self.assertEqual(codes[-1], 429)

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

    def test_patient_pdf_serves_hindi_and_english_and_rejects_any_other_lang(self):
        # `?lang=` picks one of the two sheets the caller was promised; a third language is a bad request,
        # never a silent fallback to Hindi that would hand an English reader a Hindi PDF.
        self.sign_in(ADMIN_PASS, "Ada")
        call = self.seeded_call()
        url = f"/api/calls/{call['id']}/patient.pdf"
        sheets = {}
        for lang in ("hi", "en"):
            with self.subTest(lang=lang):
                res = self.client.post(url, params={"lang": lang})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertEqual(res.headers["content-type"], "application/pdf")
                self.assertTrue(res.content.startswith(b"%PDF"))
                self.assertIn(f'filename="{reports.pdf_filename(load_pack("shishu-hi"), call, lang)}"',
                              res.headers["content-disposition"])
                self.assertTrue(res.headers["content-disposition"].endswith(f'-{lang}.pdf"'),
                                res.headers["content-disposition"])
                sheets[lang] = res.content
                # A GET reads back exactly the sheet that language's build wrote.
                again = self.client.get(url, params={"lang": lang})
                self.assertEqual(again.status_code, 200, again.text)
                self.assertEqual(again.content, res.content)
                self.assertIn(f"-{lang}.pdf", again.headers["content-disposition"])
        self.assertNotEqual(sheets["hi"], sheets["en"], "the two sheets are genuinely different documents")

        # `?lang=` is normalised before it is validated, so shouting or padding the same code still works…
        for loose in ("HI", " hi ", "EN"):
            with self.subTest(normalised=loose):
                res = self.client.get(url, params={"lang": loose})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertTrue(res.headers["content-disposition"].endswith(f'-{loose.strip().lower()}.pdf"'),
                                res.headers["content-disposition"])
        # An absent or blank `?lang=` is the caller's Hindi sheet, which is what the call page has always fetched.
        for default in (None, ""):
            with self.subTest(default=default):
                query = {} if default is None else {"lang": default}
                res = self.client.get(url, params=query)
                self.assertEqual(res.status_code, 200, res.text)
                self.assertEqual(res.content, sheets["hi"], "a missing lang is the Hindi sheet, byte for byte")
                self.assertTrue(res.headers["content-disposition"].endswith('-hi.pdf"'),
                                res.headers["content-disposition"])
        # …but a language the app never built is a bad request on either method, never a quiet fallback.
        for lang in ("fr", "de", "hi-en", "english"):
            with self.subTest(rejected=lang):
                self.assertEqual(self.client.get(url, params={"lang": lang}).status_code, 400)
                self.assertEqual(self.client.post(url, params={"lang": lang}).status_code, 400)

    def test_audio_is_404_until_a_call_records_one(self):
        self.sign_in(ADMIN_PASS, "Ada")
        call = self.seeded_call()
        self.assertEqual(self.client.get(f"/api/audio/{call['id']}").status_code, 404)


class TokenPdfTests(AppTestCase):
    def test_share_token_serves_the_pdf_without_a_login(self):
        call = self.seeded_call("fever-opd-hi")
        res = self.client.get(f"/api/calls/by-token/{call['share_token']}/patient.pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/pdf")
        self.assertTrue(res.content.startswith(b"%PDF"))
        self.assertIn("SHISHU", res.headers["content-disposition"])

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

    def test_a_stranger_gets_no_call_body(self):
        call = self.seeded_call()
        self.sign_in(USER_PASS, "Mira")
        bug = self.client.post("/api/bugs", json={
            "call_id": call["id"], "share_token": "not-the-real-token",
            "category": "Other", "note": "peek",
        }).json()["bug"]
        self.assertEqual(bug["snapshot"]["last_transcript"], [])
        self.assertEqual(bug["snapshot"]["recent_events"], [])

    def test_reports_need_a_json_object_and_stay_under_the_cap(self):
        self.sign_in()
        self.assertEqual(self.client.post("/api/bugs", content=b"[]").status_code, 400)
        self.assertEqual(self.client.post("/api/bugs", content=b"nonsense").status_code, 400)
        self.assertEqual(self.client.post("/api/bugs", content=b"").status_code, 400)
        oversized = b'{"note":"' + b"x" * (intake_app.BUG_CAP + 1) + b'"}'
        self.assertEqual(self.client.post("/api/bugs", content=oversized).status_code, 413)


class LiveSocketTests(AppTestCase):
    def close_code(self, query: str) -> int:
        with self.client.websocket_connect(f"/ws/live?{query}") as ws:
            return ws.receive().get("code")

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

    def test_ws_live_forwards_resume_query_param(self):
        # A dropped call is picked up again by opening the socket with `?resume=<share_token>`: the token
        # must reach the server-side session intact, and a plain socket must say plainly that it resumes nothing.
        self.sign_in()
        seen = {}

        async def capture(ws, **kwargs):
            seen.clear()
            seen.update(kwargs)

        with mock.patch.object(live, "handle_live_ws", capture):
            with self.client.websocket_connect("/ws/live?pack=shishu-hi&resume=tok123"):
                pass
            self.assertEqual(seen["resume_token"], "tok123")

            with self.client.websocket_connect("/ws/live?pack=shishu-hi"):
                pass
            self.assertEqual(seen["resume_token"], "")


class SeedGateTests(unittest.TestCase):
    """A fresh production database must stay empty unless demo seeding is asked for explicitly.

    `ensure_seeded()` used to run on every boot, so a brand-new prod deploy silently invented six completed
    patients. It now runs only when `SEED_DEMO_CALLS` says so. Each case boots the real lifespan in a child
    process against its own empty `DATA_DIR`, because `intake.db` binds its database path at import time.
    """

    SCRIPT = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "from intake import app as intake_app, db\n"
        "with TestClient(intake_app.app):\n"
        "    pass\n"
        "print(json.dumps(len(db.list_calls())))\n"
    )

    def boot_with(self, seed_flag: str | None) -> int:
        root = Path(__file__).resolve().parents[1]
        data_dir = tempfile.mkdtemp(prefix="intake-seed-gate-")
        env = {k: v for k, v in os.environ.items() if k != "SEED_DEMO_CALLS"}
        env.update({"DATA_DIR": data_dir, "PYTHONPATH": str(root)})
        if seed_flag is not None:
            env["SEED_DEMO_CALLS"] = seed_flag
        try:
            done = subprocess.run([sys.executable, "-c", self.SCRIPT], cwd=str(root), env=env,
                                  capture_output=True, text=True, timeout=300)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return int(json.loads(done.stdout.strip().splitlines()[-1]))

    def test_an_empty_database_stays_empty_without_the_seed_flag(self):
        self.assertEqual(self.boot_with(None), 0)

    def test_a_negative_seed_flag_leaves_the_database_empty_too(self):
        for flag in ("", "0", "false", "no", "off"):
            self.assertEqual(self.boot_with(flag), 0, flag)

    def test_the_seed_flag_fills_an_empty_database(self):
        for flag in ("1", "true", "yes", "TRUE", " Yes "):
            self.assertEqual(self.boot_with(flag), 6, flag)


if __name__ == "__main__":
    unittest.main()
