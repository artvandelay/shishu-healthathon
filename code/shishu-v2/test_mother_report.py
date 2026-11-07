"""Mother PDF report. No live call. LLM is mocked except where a test says otherwise."""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

_TMP = tempfile.mkdtemp(prefix="shishu-report-")
os.environ["DATA_DIR"] = _TMP
os.environ["APP_PASSCODE"] = "user-test"
os.environ["ADMIN_PASSCODE"] = "admin-test"

import answer_extractor  # noqa: E402
import checklist  # noqa: E402
import db  # noqa: E402
import mother_report  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as shishu_app  # noqa: E402


def _call(**extra):
    base = {
        "id": 1,
        "mother_name": "Shilpa",
        "child_name": "Aarav",
        "child_age": "7 years",
        "child_gender": "male",
        "started_at": "2026-11-06 08:40:31",
        "duration_sec": 621,
        "summary_mother": "1. जन्म के बाद नीली रोशनी के बारे में पूछें?\n2. घुटने की सूजन के बारे में पूछें?",
        "findings": {},
        "telemetry": {"coverage": [], "close_reason": ""},
        "transcript": [],
        "status": "completed",
    }
    base.update(extra)
    return base


class HindiQuestions(unittest.TestCase):
    def test_every_topic_has_a_hindi_question(self):
        for topic in checklist.TOPICS:
            self.assertIn(topic["id"], checklist.HI_QUESTIONS)
            self.assertTrue(checklist.HI_QUESTIONS[topic["id"]].endswith("?"))
            self.assertIn(topic["group"], checklist.GROUP_HI)


class ExtractorTests(unittest.TestCase):
    def test_parse_drops_unknown_ids_and_scrubs_disease_names(self):
        raw = (
            '{"answers": {"birth_nicu": {"answer": "no", "evidence": "nahi"}, '
            '"not_a_topic": {"answer": "yes", "evidence": "x"}, '
            '"body_blue": {"answer": "maybe", "evidence": ""}}, '
            '"yes_hindi": ["बच्चे को हीमोफीलिया हो सकता है"]}'
        )
        out = answer_extractor.parse_payload(raw)
        self.assertEqual(set(out["answers"]), {"birth_nicu"})
        self.assertNotIn("हीमोफीलिया", out["yes_hindi"][0])

    def test_fallback_uses_findings_and_coverage(self):
        call = _call(
            findings={
                "birth_nicu": {"status": "positive", "detail": "Three days of blue light.", "is_red_flag": True},
                "pregnancy": {"status": "negative", "detail": "Mother reports this area as normal.", "is_red_flag": False},
            },
            telemetry={"coverage": ["birth_nicu", "preg_conditions", "birth_cleft"], "close_reason": ""},
        )
        out = answer_extractor.fallback_from_call(call)
        self.assertEqual(out["answers"]["birth_nicu"]["answer"], "yes")
        self.assertEqual(out["answers"]["preg_conditions"]["answer"], "no")
        self.assertEqual(out["answers"]["birth_cleft"]["answer"], "unknown")
        self.assertEqual(out["answers"]["fam_cancer"]["answer"], "not_asked")
        self.assertTrue(out["yes_hindi"])


class ReportTests(unittest.TestCase):
    def test_red_flags_add_thalassemia_once_and_group_the_nos(self):
        answers = {
            "birth_nicu": {"answer": "yes", "evidence": ""},
            "body_joint_swelling": {"answer": "yes", "evidence": ""},
            "preg_conditions": {"answer": "yes", "evidence": ""},
            "fam_infant_death": {"answer": "yes", "evidence": ""},
            "preg_thalassemia": {"answer": "no", "evidence": ""},
            "birth_cleft": {"answer": "no", "evidence": ""},
            "birth_hearing_vision": {"answer": "unknown", "evidence": ""},
        }
        report = mother_report.build(
            _call(telemetry={"coverage": list(checklist.REQUIRED_IDS), "close_reason": ""}),
            {"answers": answers, "yes_hindi": ["जन्म के बाद तीन दिन नीली रोशनी में रहा।"]},
        )
        self.assertEqual(report["status"], "पूर्ण कॉल")
        self.assertEqual(report["yes"], ["जन्म के बाद तीन दिन नीली रोशनी में रहा।"])
        self.assertEqual(sum(1 for q in report["questions"] if "थैलेसीमिया" in q), 1)
        titles = [group["title"] for group in report["groups"]]
        self.assertIn("जन्म", titles)
        self.assertTrue(report["unknown"])
        self.assertNotIn("हीमोफीलिया", mother_report.render_html(report))

    def test_no_yes_uses_the_calm_line(self):
        report = mother_report.build(
            _call(summary_mother="", telemetry={"coverage": list(checklist.REQUIRED_IDS), "close_reason": ""}),
            {"answers": {}, "yes_hindi": []},
        )
        self.assertEqual(report["yes"], [])
        self.assertIn("कोई बात हाँ में नहीं आई", mother_report.render_html(report))
        self.assertGreaterEqual(len(report["questions"]), 3)

    def test_incomplete_lists_unasked_groups_and_not_as_no(self):
        report = mother_report.build(
            _call(telemetry={"coverage": ["intro_name"], "close_reason": "silence"}),
            {
                "answers": {"birth_nicu": {"answer": "no", "evidence": ""}, "ill_admissions": {"answer": "not_asked", "evidence": ""}},
                "yes_hindi": [],
            },
        )
        self.assertTrue(report["incomplete"])
        self.assertIn("बीमारी", report["not_asked"])
        html = mother_report.render_html(report)
        self.assertIn("यह कॉल अधूरी रही", html)
        self.assertIn("अभी नहीं पूछा गया", html)
        self.assertNotIn(checklist.question_hi("ill_admissions") + "</span><span class=\"ans\">नहीं", html)


class PdfTests(unittest.TestCase):
    def test_pdf_bytes_have_two_pages(self):
        report = mother_report.build(
            _call(),
            {"answers": {"birth_cleft": {"answer": "no", "evidence": ""}}, "yes_hindi": ["एक बात हाँ में थी।"]},
        )
        pdf = mother_report.render_pdf(report)
        self.assertTrue(pdf.startswith(b"%PDF"))
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf))
        self.assertGreaterEqual(len(reader.pages), 2)
        self.assertIn("कटा", reader.pages[1].extract_text() or "")


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(shishu_app.app)
        self.call_id = db.create_call("gemini", caller_name="owner", user_role="user")
        db.complete_call(self.call_id, 30, "", "doctor", "1. क्या जाँच करवाएँ?")

    def _login(self, username, passcode):
        res = self.client.post("/api/login", json={"username": username, "passcode": passcode})
        self.assertEqual(res.status_code, 200)

    def test_access(self):
        with patch.object(answer_extractor, "_from_llm", new=AsyncMock(return_value={"answers": {}, "yes_hindi": []})), \
             patch.object(mother_report, "render_pdf", return_value=b"%PDF-1.4\n"):
            self._login("owner", "user-test")
            own = self.client.get(f"/api/calls/{self.call_id}/mother.pdf")
            self.assertEqual(own.status_code, 200)
            self.assertTrue(own.content.startswith(b"%PDF"))
            again = self.client.get(f"/api/calls/{self.call_id}/mother.pdf")
            self.assertEqual(again.status_code, 200)
            self.assertEqual(answer_extractor._from_llm.await_count, 1)

            self._login("other", "user-test")
            denied = self.client.get(f"/api/calls/{self.call_id}/mother.pdf")
            self.assertEqual(denied.status_code, 403)

            self._login("boss", "admin-test")
            allowed = self.client.get(f"/api/calls/{self.call_id}/mother.pdf")
            self.assertEqual(allowed.status_code, 200)

            missing = self.client.get("/api/calls/99999/mother.pdf")
            self.assertEqual(missing.status_code, 404)

            active_id = db.create_call("gemini", caller_name="boss", user_role="admin")
            running = self.client.get(f"/api/calls/{active_id}/mother.pdf")
            self.assertEqual(running.status_code, 409)


if __name__ == "__main__":
    unittest.main()
