"""Mother PDF report. No live call. LLM is mocked except where a test says otherwise."""

import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_TMP = tempfile.mkdtemp(prefix="shishu-report-")
os.environ["DATA_DIR"] = _TMP
os.environ["APP_PASSCODE"] = "user-test"
os.environ["ADMIN_PASSCODE"] = "admin-test"

import answer_extractor  # noqa: E402
import checklist  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import mother_report  # noqa: E402
import report_service  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as shishu_app  # noqa: E402


class ReportReadyFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_ending_then_one_generation_then_share(self):
        events = []
        recorder = MagicMock()
        recorder.save.return_value = ("call.wav", 12)

        async def record_send(_ws, _lock, payload):
            events.append(payload["type"])

        def complete_call(*_args, **_kwargs):
            events.append("db_completed")

        async def prepare(call_id):
            events.append(f"generate:{call_id}")
            return True

        def doctor_note(*_args, **_kwargs):
            events.append("doctor_note")

        with patch.object(gemini_live.db, "get_call", return_value={"transcript": []}), \
             patch.object(gemini_live.db, "complete_call", new=complete_call), \
             patch.object(gemini_live, "_spawn_doctor_note", new=doctor_note), \
             patch.object(gemini_live, "_send_json", new=record_send):
            await gemini_live._finalize(
                MagicMock(), MagicMock(), 42, None, recorder, prepare
            )

        self.assertEqual(
            events,
            ["ending", "db_completed", "generate:42", "doctor_note", "call_completed"],
        )

    async def test_generation_failure_keeps_the_call_completed_and_asks_retry(self):
        events = []
        recorder = MagicMock()
        recorder.save.return_value = ("call.wav", 12)

        async def record_send(_ws, _lock, payload):
            events.append(payload["type"])

        completed = MagicMock()
        doctor_note = MagicMock()

        async def prepare(_call_id):
            events.append("generate")
            return False

        with patch.object(gemini_live.db, "get_call", return_value={"transcript": []}), \
             patch.object(gemini_live.db, "complete_call", new=completed), \
             patch.object(gemini_live, "_spawn_doctor_note", new=doctor_note), \
             patch.object(gemini_live, "_send_json", new=record_send):
            await gemini_live._finalize(
                MagicMock(), MagicMock(), 42, None, recorder, prepare
            )

        completed.assert_called_once()
        doctor_note.assert_not_called()
        self.assertEqual(events, ["ending", "generate", "report_failed"])
        self.assertNotIn("call_completed", events)

    async def test_prepare_and_retry_use_the_same_report_operation(self):
        operation = AsyncMock(return_value=(b"%PDF-1.4\n", "SHISHU.pdf"))
        with patch.object(report_service, "mother_pdf", new=operation), \
             patch.object(report_service, "ready_pdf", return_value=(b"%PDF-1.4\n", "SHISHU.pdf")):
            ready = await shishu_app._prepare_mother_pdf(42)
        operation.assert_awaited_once_with(42)
        self.assertTrue(ready)

    def test_voice_ui_shows_preparing_before_share(self):
        page = Path(__file__).with_name("static").joinpath("call.html").read_text(encoding="utf-8")
        ending = page.index('msg.type === "ending"')
        preparing = page.index("beginPreparing()")
        completed = page.index('msg.type === "call_completed"')
        share = page.index("revealReadyReport")
        failed = page.index('msg.type === "report_failed"')
        self.assertLess(ending, preparing)
        self.assertLess(preparing, completed)
        self.assertLess(completed, share)
        self.assertIn("Preparing report…", page)
        self.assertIn("This can take a little while.", page)
        self.assertIn('method: "POST"', page)
        self.assertNotIn("FALLBACK_MOTHER_QUESTIONS", page)
        self.assertLess(failed, page.index("showReportRetry"))
        preparing_fn = page[page.index("function beginPreparing()"):page.index("function hangupWhenQuiet()")]
        self.assertNotIn("60000", preparing_fn)
        self.assertNotIn("showReportRetry", preparing_fn)
        self.assertIn('id="share-pdf"', page)
        self.assertLess(page.index('id="share-pdf"'), page.index("revealReadyReport"))
        self.assertIn("hidden = !pdfFile", page)


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
    def test_parse_drops_unknown_ids_and_freeform_lines(self):
        raw = (
            '{"answers": {"birth_nicu": {"answer": "no", "evidence": "nahi"}, '
            '"not_a_topic": {"answer": "yes", "evidence": "x"}, '
            '"body_blue": {"answer": "maybe", "evidence": ""}}, '
            '"yes_hindi": ["बच्चे को हीमोफीलिया हो सकता है"], '
            '"questions_hindi": ["क्या हीमोफीलिया की जाँच?"]}'
        )
        out = answer_extractor.parse_payload(raw)
        self.assertEqual(set(out["answers"]), {"birth_nicu"})
        self.assertNotIn("yes_hindi", out)
        self.assertNotIn("questions_hindi", out)
        self.assertNotIn("हीमोफीलिया", json.dumps(out, ensure_ascii=False))

    def test_fallback_uses_findings_but_not_routing_coverage_as_an_answer(self):
        call = _call(
            findings={
                "birth_nicu": {"status": "positive", "detail": "Three days of blue light.", "is_red_flag": True},
                "pregnancy": {"status": "negative", "detail": "Mother reports this area as normal.", "is_red_flag": False},
            },
            telemetry={"coverage": ["birth_nicu", "preg_conditions", "birth_cleft"], "close_reason": ""},
        )
        out = answer_extractor.fallback_from_call(call)
        self.assertEqual(out["answers"]["birth_nicu"]["answer"], "not_asked")
        self.assertEqual(out["answers"]["preg_conditions"]["answer"], "not_asked")
        self.assertEqual(out["answers"]["birth_cleft"]["answer"], "not_asked")
        self.assertEqual(out["answers"]["fam_cancer"]["answer"], "not_asked")
        self.assertNotIn("yes_hindi", out)
        self.assertEqual(answer_extractor.report_content(out)["questions"], [])

    def test_validation_keeps_a_short_verbatim_no_on_its_question(self):
        call = _call(transcript=[
            {"role": "assistant", "text": checklist.question_hi("birth_cleft")},
            {"role": "user", "text": "नहीं।"},
        ])
        out = answer_extractor.validate_answers(
            call,
            {"answers": {"birth_cleft": {"answer": "no", "evidence": "नहीं"}}},
        )
        self.assertEqual(out["answers"]["birth_cleft"], {"answer": "no", "evidence": "नहीं"})

    def test_two_identical_nos_do_not_cross_authorize(self):
        call = _call(transcript=[
            {"role": "assistant", "text": checklist.question_hi("birth_cleft")},
            {"role": "user", "text": "नहीं।"},
            {"role": "assistant", "text": checklist.question_hi("fam_cancer")},
            {"role": "user", "text": "नहीं।"},
        ])
        out = answer_extractor.validate_answers(
            call,
            {
                "answers": {
                    "birth_cleft": {"answer": "no", "evidence": "नहीं"},
                    "fam_cancer": {"answer": "no", "evidence": "नहीं"},
                    "fam_heart": {"answer": "no", "evidence": "नहीं"},
                }
            },
        )
        self.assertEqual(out["answers"]["birth_cleft"]["answer"], "no")
        self.assertEqual(out["answers"]["fam_cancer"]["answer"], "no")
        self.assertEqual(out["answers"]["fam_heart"]["answer"], "not_asked")
        only_cleft = answer_extractor.validate_answers(
            _call(transcript=call["transcript"][:2]),
            {"answers": {"fam_cancer": {"answer": "no", "evidence": "नहीं"}}},
        )
        self.assertEqual(only_cleft["answers"]["fam_cancer"]["answer"], "not_asked")

    def test_validation_drops_an_unquoted_verdict_and_exposes_findings_as_hints(self):
        call = _call(
            transcript=[{"role": "user", "text": "नहीं।"}],
            findings={"birth_nicu": {"status": "negative", "detail": "No NICU stay."}},
        )
        out = answer_extractor.validate_answers(
            call,
            {"answers": {"birth_nicu": {"answer": "no", "evidence": "No NICU stay"}}},
        )
        self.assertEqual(out["answers"]["birth_nicu"]["answer"], "not_asked")
        content = answer_extractor._user_content(call)
        self.assertIn("Saved live-call findings", content)
        self.assertIn("birth_nicu: status=negative; detail=No NICU stay.", content)

    def test_grounding_never_spans_user_turns(self):
        call = _call(
            transcript=[
                {"role": "user", "text": "चार बार"},
                {"role": "assistant", "text": "कब?"},
                {"role": "user", "text": "हो चुका है।"},
            ]
        )
        out = answer_extractor.validate_answers(
            call,
            {"answers": {"fam_miscarriage": {"answer": "yes", "evidence": "चार बार हो चुका है"}}},
        )
        self.assertEqual(out["answers"]["fam_miscarriage"]["answer"], "not_asked")


class MotherPathTests(unittest.TestCase):
    def test_positive_lines_and_questions_are_deterministic(self):
        evidence = "नीली रोशनी में रखा था"
        call = _call(
            summary_mother=(
                "1. क्या वज़न ठीक है?\n"
                "2. बच्चे के सभी ज़रूरी टीके बाकी हैं?\n"
                "3. रोज़ के खान-पान के लिए क्या ध्यान रखूँ?"
            ),
            transcript=[
                {"role": "assistant", "text": checklist.question_hi("birth_nicu")},
                {"role": "user", "text": evidence + "।"},
            ],
        )
        extracted = answer_extractor.validate_answers(
            call,
            {
                "answers": {"birth_nicu": {"answer": "yes", "evidence": evidence}},
                "yes_hindi": ["बच्चे को हीमोफीलिया हो सकता है"],
                "questions_hindi": ["क्या एक सामान्य सवाल?"],
            },
        )
        content = answer_extractor.report_content(extracted)
        self.assertEqual(len(content["yes"]), 1)
        self.assertEqual(len(content["questions"]), 1)
        self.assertIn(checklist.question_hi("birth_nicu"), content["yes"][0])
        self.assertIn(evidence, content["yes"][0])
        self.assertIn(evidence, content["questions"][0])
        self.assertTrue(content["questions"][0].rstrip().endswith("?"))
        self.assertNotIn("हीमोफीलिया", content["questions"][0])
        self.assertNotIn("वज़न", content["questions"][0])
        self.assertNotIn("टीके", content["questions"][0])
        report = mother_report.build(call, extracted)
        self.assertEqual(len(report["questions"]), 1)
        html = mother_report.render_html(report)
        self.assertIn(evidence, html)
        self.assertNotIn("सभी ज़रूरी टीके", html)
        self.assertNotIn("खान-पान", html)
        self.assertNotIn("हीमोफीलिया", html)

    def test_no_positive_answer_adds_no_generic_question(self):
        content = answer_extractor.report_content(
            {"questions_hindi": ["क्या एक ही सवाल?"], "yes_hindi": ["एक बात हाँ में थी।"]}
        )
        self.assertEqual(content["yes"], [])
        self.assertEqual(content["questions"], [])

    def test_doctor_save_leaves_the_mother_text(self):
        call_id = db.create_call("gemini")
        db.complete_call(call_id, 10, "", "", "1. क्या जाँच करवाएँ?")
        db.save_doctor_summary(
            call_id,
            "### Patient Profile & Chief Complaint\nNoted.",
            {"birth_nicu": {"status": "positive", "detail": "NICU", "is_red_flag": True}},
            ["birth_nicu: NICU"],
        )
        call = db.get_call(call_id)
        self.assertIn("जाँच", call["summary_mother"])
        self.assertIn("Patient Profile", call["summary_doctor"])
        self.assertEqual(call["findings"]["birth_nicu"]["detail"], "NICU")


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
            _call(
                transcript=[
                    {
                        "role": "user",
                        "text": "तीन दिन नीली रोशनी में रहा। घुटने में सूजन है। बीपी बढ़ा था। बच्चा मर गया। नहीं। पता नहीं।",
                    }
                ],
                telemetry={"coverage": list(checklist.REQUIRED_IDS), "close_reason": ""},
            ),
            {
                "answers": {
                    **answers,
                    "birth_nicu": {"answer": "yes", "evidence": "तीन दिन नीली रोशनी"},
                    "body_joint_swelling": {"answer": "yes", "evidence": "घुटने में सूजन"},
                    "preg_conditions": {"answer": "yes", "evidence": "बीपी बढ़ा था"},
                    "fam_infant_death": {"answer": "yes", "evidence": "बच्चा मर गया"},
                    "preg_thalassemia": {"answer": "no", "evidence": "नहीं"},
                    "birth_cleft": {"answer": "no", "evidence": "नहीं"},
                    "birth_hearing_vision": {"answer": "unknown", "evidence": "पता नहीं"},
                },
                "yes_hindi": ["जन्म के बाद तीन दिन नीली रोशनी में रहा।"],
            },
        )
        self.assertEqual(mother_report.transliterate("Shilpa"), "शिल्पा")
        self.assertEqual(mother_report.transliterate("Aarav"), "आरव")
        self.assertEqual(report["status"], "पूर्ण कॉल")
        self.assertIn("शिल्पा", report["who"])
        self.assertIn("आरव", report["who"])
        self.assertNotIn("Shilpa", mother_report.render_html(report))
        self.assertEqual(len(report["yes"]), 4)
        self.assertIn(checklist.question_hi("birth_nicu"), report["yes"][0])
        self.assertIn("तीन दिन नीली रोशनी", report["yes"][0])
        self.assertFalse(any("थैलेसीमिया" in q for q in report["questions"]))
        html = mother_report.render_html(report)
        self.assertEqual(html.count(mother_report.THALASSEMIA_LINE), 1)
        self.assertLess(html.index('class="universal"'), html.index('class="page2"'))
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
        html = mother_report.render_html(report)
        self.assertIn("कोई जवाब दर्ज नहीं हुआ", html)
        self.assertEqual(report["questions"], [])
        self.assertNotIn("डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से ये पूछें", html)
        self.assertNotIn("वज़न, लंबाई और सीखना", html)
        self.assertNotIn("सभी ज़रूरी टीके", html)
        self.assertNotIn("खान-पान", html)

    def test_short_verbatim_no_is_printed_without_coverage(self):
        question = checklist.question_hi("birth_cleft")
        report = mother_report.build(
            _call(
                transcript=[
                    {"role": "assistant", "text": question},
                    {"role": "user", "text": "नहीं।"},
                ],
                telemetry={"coverage": [], "close_reason": ""},
            ),
            {"answers": {"birth_cleft": {"answer": "no", "evidence": "नहीं"}}, "yes_hindi": []},
        )
        html = mother_report.render_html(report)
        self.assertIn(question + '</span><span class="ans">नहीं', html)

    def test_coverage_alone_does_not_print_an_unquoted_no(self):
        question = checklist.question_hi("birth_cleft")
        call = _call(
            transcript=[{"role": "user", "text": "नमस्ते।"}],
            telemetry={"coverage": ["birth_cleft"], "close_reason": ""},
        )
        extracted = answer_extractor.validate_answers(
            call,
            {"answers": {"birth_cleft": {"answer": "no", "evidence": ""}}, "yes_hindi": []},
        )
        report = mother_report.build(call, extracted)
        self.assertNotIn(question + '</span><span class="ans">नहीं', mother_report.render_html(report))

    def test_incomplete_lists_unasked_groups_and_not_as_no(self):
        call = _call(telemetry={"coverage": ["intro_name"], "close_reason": "silence"})
        extracted = answer_extractor.validate_answers(
            call,
            {
                "answers": {"birth_nicu": {"answer": "no", "evidence": ""}, "ill_admissions": {"answer": "not_asked", "evidence": ""}},
                "yes_hindi": [],
            },
        )
        report = mother_report.build(call, extracted)
        self.assertTrue(report["incomplete"])
        self.assertIn("बीमारी", report["not_asked"])
        html = mother_report.render_html(report)
        self.assertIn("यह कॉल अधूरी रही", html)
        self.assertIn("अभी नहीं पूछा गया", html)
        self.assertIn(checklist.question_hi("ill_admissions") + "</span><span class=\"ans\">पूछा नहीं गया", html)
        self.assertNotIn(checklist.question_hi("ill_admissions") + "</span><span class=\"ans\">नहीं", html)
        self.assertNotIn(checklist.question_hi("birth_nicu") + "</span><span class=\"ans\">नहीं", html)


class QuestionPaddingTests(unittest.TestCase):
    def test_summary_filler_never_reaches_the_pdf(self):
        import summarizer

        summary = "\n".join(
            f"{idx}. {question}"
            for idx, question in enumerate(summarizer._DEFAULT_MOTHER_QUESTIONS, start=1)
        )
        report = mother_report.build(
            _call(summary_mother=summary),
            {
                "answers": {},
                "questions_hindi": list(summarizer._DEFAULT_MOTHER_QUESTIONS),
            },
        )
        self.assertEqual(report["questions"], [])
        html = mother_report.render_html(report)
        self.assertNotIn("सभी ज़रूरी टीके", html)
        self.assertNotIn("खान-पान", html)
        self.assertNotIn("एनीमिया", html)
        self.assertNotIn("<li>" + mother_report.THALASSEMIA_LINE, html)


class UniversalThalassemiaTests(unittest.TestCase):
    def test_thalassemia_box_is_fixed_at_the_end_of_page_1(self):
        covered = _call(telemetry={"coverage": list(checklist.REQUIRED_IDS), "close_reason": ""})
        report = mother_report.build(
            covered,
            {"answers": {"preg_thalassemia": {"answer": "no", "evidence": ""}}, "yes_hindi": []},
        )
        self.assertEqual(report["universal"], mother_report.THALASSEMIA_LINE)
        self.assertNotIn(mother_report.THALASSEMIA_LINE, report["questions"])
        html = mother_report.render_html(report)
        self.assertIn("सभी के लिए एक जैसी सलाह", html)
        self.assertIn("यह इस माँ या बच्चे के बारे में नहीं है।", html)
        self.assertEqual(html.count(mother_report.THALASSEMIA_LINE), 1)
        self.assertLess(html.index("इस कॉल में जो हाँ था"), html.index('class="universal"'))
        self.assertLess(html.index('class="universal"'), html.index('class="page2"'))
        self.assertNotIn(f"<li>{mother_report.THALASSEMIA_LINE}</li>", html)
        yes_html = mother_report.render_html(
            mother_report.build(
                covered,
                {"answers": {"preg_thalassemia": {"answer": "yes", "evidence": ""}}, "yes_hindi": []},
            )
        )
        self.assertIn(mother_report.THALASSEMIA_LINE, yes_html)
        self.assertIn("सभी के लिए एक जैसी सलाह", yes_html)


class UngroundedNoTests(unittest.TestCase):
    def test_a_greeting_only_call_does_not_print_no(self):
        answers = {
            topic["id"]: {"answer": "no", "evidence": "माँ ने बताया कि ऐसा नहीं हुआ।"}
            for topic in checklist.TOPICS
        }
        call = _call(
            transcript=[{"role": "assistant", "text": "Namaste, aapka naam bataiyega?"}],
            telemetry={"coverage": [], "close_reason": ""},
            summary_mother="",
        )
        extracted = answer_extractor.validate_answers(
            call, {"answers": answers, "yes_hindi": []}
        )
        report = mother_report.build(call, extracted)
        html = mother_report.render_html(report)
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["groups"], [])
        self.assertIn("पूछा नहीं गया", html)
        self.assertNotIn('class="ans">नहीं', html)
        self.assertIn("कोई जवाब दर्ज नहीं हुआ", html)
        self.assertIn("अधूरी कॉल", html)

    def test_an_unquoted_yes_does_not_print_its_free_text_summary(self):
        call = _call(transcript=[{"role": "user", "text": "नमस्ते।"}], summary_mother="")
        extracted = answer_extractor.validate_answers(
            call,
            {
                "answers": {"birth_nicu": {"answer": "yes", "evidence": "blue light"}},
                "yes_hindi": ["बच्चे को जन्म के बाद नीली रोशनी में रखा गया था।"],
            },
        )
        report = mother_report.build(
            call,
            extracted,
        )
        self.assertEqual(report["yes"], [])

    def test_each_yes_line_is_tied_to_its_validated_topic(self):
        call = _call(
            transcript=[
                {"role": "assistant", "text": checklist.question_hi("birth_nicu")},
                {"role": "user", "text": "नीली रोशनी में रखा था।"},
                {"role": "assistant", "text": checklist.question_hi("fam_miscarriage")},
                {"role": "user", "text": "चार बार हुआ।"},
                {"role": "assistant", "text": checklist.question_hi("fam_cancer")},
                {"role": "user", "text": "नहीं।"},
            ]
        )
        extracted = answer_extractor.validate_answers(
            call,
            {
                "answers": {
                    "birth_nicu": {"answer": "yes", "evidence": "नीली रोशनी में रखा था"},
                    "fam_miscarriage": {"answer": "yes", "evidence": "चार बार हुआ"},
                    "fam_cancer": {"answer": "no", "evidence": "नहीं"},
                },
                "yes_hindi": ["कैंसर के सवाल पर हाँ कहा।"],
            },
        )
        yes = mother_report.build(call, extracted)["yes"]
        self.assertEqual(len(yes), 2)
        self.assertTrue(any(checklist.question_hi("birth_nicu") in line for line in yes))
        self.assertTrue(any(checklist.question_hi("fam_miscarriage") in line for line in yes))
        self.assertFalse(any("कैंसर के सवाल" in line for line in yes))
        self.assertEqual(extracted["answers"]["fam_cancer"]["answer"], "no")

    def test_renderer_does_not_reclassify_transcript_evidence(self):
        report = mother_report.build(
            _call(transcript=[{"role": "user", "text": "हाँ जी।"}]),
            {"answers": {"birth_cleft": {"answer": "no", "evidence": "नहीं"}}},
        )
        self.assertIn('class="ans">नहीं', mother_report.render_html(report))


class ProductionReplayTests(unittest.TestCase):
    def test_call_286_short_grounded_nos_render_as_no(self):
        replay_path = Path("/tmp/shishu-production-replay/call-286.json")
        if not replay_path.is_file():
            self.skipTest("saved Call 286 replay is unavailable")
        call = json.loads(replay_path.read_text(encoding="utf-8"))
        no_topics = (
            "preg_thalassemia",
            "fam_related",
            "fam_infant_death",
            "fam_fits_delay",
            "fam_blind",
            "fam_deaf",
            "fam_transfusion",
            "fam_bleeding",
            "fam_cancer",
            "fam_heart",
            "fam_kidney",
            "fam_bp_stroke",
            "fam_maternal_side",
            "fam_lumps",
        )
        extracted = answer_extractor.validate_answers(
            call,
            {
                "answers": {
                    topic_id: {"answer": "no", "evidence": "नहीं"}
                    for topic_id in no_topics
                }
            },
        )
        report = mother_report.build(call, extracted)
        rendered_nos = sum(len(group["items"]) for group in report["groups"])
        self.assertEqual(rendered_nos, len(no_topics))
        self.assertTrue(
            all(extracted["answers"][topic_id]["answer"] == "no" for topic_id in no_topics)
        )


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

class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_pdf_per_call_overwrites_atomically(self):
        call_id = db.create_call("gemini")
        db.complete_call(call_id, 1, "", "", "")
        calls = {"n": 0}

        async def extract(_call):
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return {"version": answer_extractor._VERSION, "answers": {}}

        def render(_report):
            return b"%PDF-" + str(calls["n"]).encode()

        with patch.object(answer_extractor, "cached_answers", new=extract), \
             patch.object(mother_report, "render_pdf", new=render):
            first, second = await asyncio.gather(
                report_service.mother_pdf(call_id),
                report_service.mother_pdf(call_id),
            )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(first[0], second[0])
        path = report_service.pdf_path(call_id)
        self.assertEqual(path.name, f"call-{call_id}.pdf")
        self.assertTrue(str(path).startswith(str(Path(db.DATA_DIR))))
        self.assertEqual(path.read_bytes(), first[0])
        self.assertEqual(
            [item.name for item in path.parent.iterdir() if item.name.startswith(".")],
            [],
        )

        def render_again(_report):
            return b"%PDF-overwrite"

        with patch.object(answer_extractor, "cached_answers", new=extract), \
             patch.object(mother_report, "render_pdf", new=render_again):
            await report_service.mother_pdf(call_id)
        self.assertEqual(path.read_bytes(), b"%PDF-overwrite")
        self.assertFalse(any(item.name.startswith(".") for item in path.parent.iterdir()))

    def test_ready_pdf_rejects_a_partial_file(self):
        call_id = db.create_call("gemini")
        path = report_service.pdf_path(call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-pdf")
        self.assertIsNone(report_service.ready_pdf(call_id))


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(shishu_app.app)
        self.call_id = db.create_call("gemini", caller_name="owner", user_role="user")
        db.complete_call(self.call_id, 30, "", "doctor", "1. क्या जाँच करवाएँ?")

    def _login(self, username, passcode):
        res = self.client.post("/api/login", json={"username": username, "passcode": passcode})
        self.assertEqual(res.status_code, 200)

    def test_get_serves_only_a_ready_pdf_and_retry_rebuilds(self):
        operation = AsyncMock(return_value=(b"%PDF-1.4\nretry", "SHISHU.pdf"))
        self._login("owner", "user-test")
        waiting = self.client.get(f"/api/calls/{self.call_id}/mother.pdf")
        self.assertEqual(waiting.status_code, 409)
        self.assertEqual(waiting.json()["detail"], "Report is not ready")

        path = report_service.pdf_path(self.call_id)
        report_service._atomic_write(path, b"%PDF-1.4\nready")
        with patch.object(report_service, "mother_pdf", new=operation):
            own = self.client.get(f"/api/calls/{self.call_id}/mother.pdf")
            self.assertEqual(own.status_code, 200)
            self.assertEqual(own.content, b"%PDF-1.4\nready")
            self.assertEqual(operation.await_count, 0)

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
            self.assertEqual(running.json()["detail"], "Call is still running")

            self._login("owner", "user-test")
            retried = self.client.post(f"/api/calls/{self.call_id}/mother.pdf")
            self.assertEqual(retried.status_code, 200)
            self.assertEqual(retried.content, b"%PDF-1.4\nretry")
        operation.assert_awaited_once_with(self.call_id)


def _grounded(topic_id, user, answer, slots=None, literal=True, findings=None):
    entry = {"answer": answer, "evidence": user}
    if literal:
        entry["literal"] = True
    if slots:
        entry["slots"] = slots
    call = _call(
        transcript=[
            {"role": "assistant", "text": checklist.question_hi(topic_id)},
            {"role": "user", "text": user},
        ],
        findings=findings or {},
    )
    extracted = answer_extractor.validate_answers(call, {"answers": {topic_id: entry}})
    return extracted


class ConcernSemanticsTests(unittest.TestCase):
    def test_prompt_asks_for_literal_slots_not_a_concern_verdict(self):
        prompt = answer_extractor._PROMPT
        self.assertIn("delivery_mode", prompt)
        self.assertIn("delivery_place", prompt)
        self.assertIn("mother_result", prompt)
        self.assertNotIn("are no, not yes", prompt)
        self.assertIn("do not return a second clinical yes or no", prompt)
        self.assertNotIn("is_concern", prompt)

    def test_token_bound_covers_every_topic_and_is_below_the_old_cap(self):
        self.assertLess(answer_extractor._MAX_OUTPUT_TOKENS, 8192)
        slot_chars = 0
        for rule in checklist.REPORT_RULES.values():
            slots = rule.get("slots") or {}
            if slots:
                slot_chars += 32 + 20 * len(slots)
        chars = 64 + len(checklist.TOPICS) * (64 + answer_extractor._EVIDENCE_CHARS) + slot_chars
        self.assertGreaterEqual(answer_extractor._MAX_OUTPUT_TOKENS, chars // 2)

    def test_gemini_flash_disables_thinking(self):
        config = answer_extractor._gemini_config()
        self.assertEqual(answer_extractor._GEMINI_MODEL, "gemini-3.8-flash")
        self.assertIn("LOW", str(config.thinking_config.thinking_level).upper())
        self.assertIsNone(config.thinking_config.thinking_budget)
        self.assertEqual(config.max_output_tokens, answer_extractor._MAX_OUTPUT_TOKENS)
        self.assertEqual(config.temperature, 0.2)

    def test_concern_matrix(self):
        cases = [
            ("birth_mode", "अस्पताल में सामान्य प्रसव हुआ", "yes",
             {"delivery_mode": "vaginal", "delivery_place": "hospital"}, False),
            ("birth_mode", "ऑपरेशन से हुआ", "yes",
             {"delivery_mode": "operation", "delivery_place": "hospital"}, True),
            ("birth_term", "नहीं", "no", None, False),
            ("birth_hearing_vision", "जाँच हुई और सामान्य थी", "yes",
             {"result": "normal"}, False),
            ("birth_hearing_vision", "जाँच में कमी बताई", "yes",
             {"result": "abnormal"}, True),
            ("preg_scan", "स्कैन हुआ और सामान्य था", "yes", {"result": "normal"}, False),
            ("preg_thalassemia", "माँ की रिपोर्ट सामान्य है", "yes",
             {"mother_result": "normal", "father_result": "normal"}, False),
            ("preg_thalassemia", "माँ वाहक हैं", "yes",
             {"mother_result": "carrier", "father_result": "normal"}, True),
            ("body_growth", "हाँ बाकी बच्चों जैसा बढ़ रहा है", "yes", None, False),
            ("body_growth", "नहीं बढ़ नहीं रहा", "no", None, True),
            ("preg_delivery", "नहीं कोई दिक्कत नहीं आई", "no", None, False),
            ("birth_nicu", "नीली रोशनी में रखा था", "yes", None, True),
            ("body_transfusion", "दो बार खून चढ़ा", "yes", None, True),
            ("body_puberty", "बाकी बच्चों के साथ शुरू हुए", "yes", {"timing": "typical"}, False),
            ("body_puberty", "बहुत पहले शुरू हो गए", "yes", {"timing": "early"}, True),
            ("intro_married", "हाँ", "yes", None, False),
        ]
        for topic_id, user, answer, slots, concern in cases:
            extracted = _grounded(topic_id, user, answer, slots)
            entry = extracted["answers"][topic_id]
            self.assertEqual(checklist.is_concern(topic_id, entry), concern, topic_id)
            content = answer_extractor.report_content(extracted)
            self.assertEqual(bool(content["questions"]), concern, topic_id)
            if concern:
                self.assertIn(user, content["questions"][0])

    def test_normal_delivery_fact_is_on_the_detail_page_not_a_doctor_line(self):
        user = "अस्पताल में सामान्य प्रसव हुआ"
        extracted = _grounded(
            "birth_mode",
            user,
            "yes",
            {"delivery_mode": "vaginal", "delivery_place": "hospital"},
        )
        report = mother_report.build(_call(), extracted)
        self.assertEqual(report["questions"], [])
        html = mother_report.render_html(report)
        self.assertIn("सामान्य प्रसव, अस्पताल", html)
        self.assertNotIn("डॉक्टर या स्वास्थ्य कार्यकर्ता", html)

    def test_intro_literal_yes_stays_hidden(self):
        extracted = _grounded("intro_married", "हाँ शादी हो गई है", "yes")
        content = answer_extractor.report_content(extracted)
        self.assertEqual(content["yes"], [])
        self.assertEqual(content["questions"], [])

    def test_slots_from_another_turn_are_dropped(self):
        call = _call(transcript=[
            {"role": "assistant", "text": checklist.question_hi("birth_mode")},
            {"role": "user", "text": "अस्पताल में सामान्य प्रसव हुआ"},
            {"role": "assistant", "text": checklist.question_hi("fam_cancer")},
            {"role": "user", "text": "नहीं"},
        ])
        extracted = answer_extractor.validate_answers(call, {"answers": {
            "birth_mode": {
                "answer": "yes",
                "evidence": "नहीं",
                "literal": True,
                "slots": {"delivery_mode": "operation", "delivery_place": "hospital"},
            }
        }})
        entry = extracted["answers"]["birth_mode"]
        self.assertEqual(entry["answer"], "not_asked")
        self.assertNotIn("slots", entry)
        self.assertFalse(checklist.is_concern("birth_mode", entry))

    def test_old_slotless_choice_and_test_do_not_become_concerns(self):
        for topic_id, user in (
            ("birth_mode", "अस्पताल में सामान्य प्रसव हुआ"),
            ("birth_hearing_vision", "जाँच हुई थी"),
            ("preg_thalassemia", "जाँच हुई थी"),
            ("body_growth", "हाँ बढ़ रहा है"),
        ):
            extracted = _grounded(topic_id, user, "yes", literal=False)
            entry = extracted["answers"][topic_id]
            self.assertNotIn("literal", entry)
            self.assertFalse(checklist.is_concern(topic_id, entry), topic_id)
            self.assertEqual(answer_extractor.report_content(extracted)["questions"], [])

    def test_old_concern_yes_still_reaches_the_doctor_line(self):
        extracted = _grounded("birth_nicu", "नीली रोशनी में रखा था", "yes", literal=False)
        self.assertTrue(checklist.is_concern("birth_nicu", extracted["answers"]["birth_nicu"]))
        self.assertEqual(len(answer_extractor.report_content(extracted)["questions"]), 1)

    def test_live_finding_does_not_override_a_typed_normal_delivery(self):
        user = "अस्पताल में सामान्य प्रसव हुआ"
        extracted = _grounded(
            "birth_mode",
            user,
            "yes",
            {"delivery_mode": "vaginal", "delivery_place": "hospital"},
            findings={"birth_mode": {"status": "positive", "detail": user, "is_red_flag": True}},
        )
        self.assertFalse(checklist.is_concern("birth_mode", extracted["answers"]["birth_mode"]))
        self.assertEqual(answer_extractor.report_content(extracted)["questions"], [])

    def test_fallback_positive_choice_stays_uncertain(self):
        call = _call(
            transcript=[
                {"role": "assistant", "text": checklist.question_hi("birth_mode")},
                {"role": "user", "text": "अस्पताल में हुआ"},
            ],
            findings={"birth_mode": {"status": "positive", "detail": "अस्पताल में हुआ", "is_red_flag": True}},
        )
        out = answer_extractor.fallback_from_call(call)
        self.assertNotEqual(out["answers"]["birth_mode"]["answer"], "yes")
        self.assertFalse(checklist.is_concern("birth_mode", out["answers"]["birth_mode"]))
        self.assertEqual(answer_extractor.report_content(out)["questions"], [])


class ExtractorCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_bump_does_not_call_the_model(self):
        call_id = db.create_call("gemini")
        quote = "नीली रोशनी में रखा था"
        db.append_transcript(call_id, "assistant", checklist.question_hi("birth_nicu"))
        db.append_transcript(call_id, "user", quote)
        db.save_answers(call_id, {
            "version": 6,
            "answers": {"birth_nicu": {"answer": "yes", "evidence": quote}},
        })

        async def paid(_call):
            raise AssertionError("paid extraction")

        with patch.object(answer_extractor, "_from_llm", new=paid):
            result = await answer_extractor.cached_answers(db.get_call(call_id))
        self.assertEqual(result["answers"]["birth_nicu"]["answer"], "yes")
        self.assertTrue(checklist.is_concern("birth_nicu", result["answers"]["birth_nicu"]))
        self.assertEqual(db.get_answers(call_id)["version"], answer_extractor._VERSION)

    async def test_fallback_log_keeps_the_exception_class_and_drops_the_message(self):
        call_id = db.create_call("gemini")

        async def gemini_fails(_content):
            raise RuntimeError("SECRET transcript line")

        async def openai_ok(_content):
            return answer_extractor.parse_payload('{"answers": {}}')

        with patch.object(answer_extractor, "_from_gemini", new=gemini_fails), \
             patch.object(answer_extractor, "_from_openai", new=openai_ok):
            await answer_extractor.cached_answers(db.get_call(call_id))
        raw = json.dumps(db.get_call(call_id)["events"], ensure_ascii=False)
        self.assertNotIn("SECRET", raw)
        classes = [
            event.get("meta", {}).get("error_class")
            for event in db.get_call(call_id)["events"]
            if event.get("type") == "extraction_fallback"
        ]
        self.assertIn("RuntimeError", classes)


class EndLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_end_call_schedules_once_and_ignores_later_input(self):
        call_id = db.create_call("gemini")
        db.append_transcript(call_id, "user", "marker-not-for-logs")
        scheduled = []

        async def on_end(seen_id, snapshot):
            scheduled.append((seen_id, sorted(snapshot)))

        telemetry = {"coverage": [], "force_close": True, "tool_calls": 0, "errors": []}

        class _Fc:
            id = "fc-end"
            name = "end_call"
            args = {"reason": "complete"}
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        ws = MagicMock()
        ws.client_state = gemini_live.WebSocketState.CONNECTED

        async def send_json(_payload):
            return None

        ws.send_json = send_json
        session = _Sess()
        await gemini_live._answer_function_calls(
            session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
            [_Fc()], set(), set(), on_end_accepted=on_end,
        )
        await gemini_live._answer_function_calls(
            session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
            [_Fc()], set(), set(), on_end_accepted=on_end,
        )
        self.assertEqual(scheduled, [(call_id, sorted({
            "id", "transcript", "findings", "telemetry", "status", "started_at", "duration_sec",
        }))])
        self.assertTrue(telemetry["end_requested"])
        self.assertTrue(telemetry["closing"])
        self.assertNotIn("marker-not-for-logs", json.dumps(
            [event for event in db.get_call(call_id)["events"] if event.get("type") == "end_accepted"],
            ensure_ascii=False,
        ))

        class _Feed:
            def __init__(self, lines):
                self.lines = lines
                self.client_state = gemini_live.WebSocketState.CONNECTED

            async def receive_text(self):
                if not self.lines:
                    raise gemini_live.WebSocketDisconnect()
                return self.lines.pop(0)

            async def send_json(self, _payload):
                return None

        class _Live:
            def __init__(self):
                self.sent = []

            async def send_realtime_input(self, **kwargs):
                self.sent.append(kwargs)

        pcm = base64.b64encode(b"\x10\x00" * 80).decode("ascii")
        locked = _Live()
        transcripts = []

        def record_transcript(*args, **kwargs):
            transcripts.append(args)

        with patch.object(gemini_live.db, "append_transcript", new=record_transcript):
            await gemini_live._browser_loop(
                locked, asyncio.Lock(), _Feed([
                    json.dumps({"type": "audio", "data": pcm, "sample_rate": 16000}),
                    json.dumps({"type": "text", "text": "ऑपरेशन से हुआ"}),
                    json.dumps({"type": "hangup"}),
                ]), asyncio.Lock(), asyncio.Event(), call_id, telemetry,
                gemini_live._TurnState(), MagicMock(),
            )
        self.assertEqual(locked.sent, [])
        self.assertEqual(transcripts, [])

        open_session = _Live()
        open_telemetry = {
            "coverage": [], "tool_calls": 0, "errors": [], "audio_chunks_in": 0,
            "user_turns": 0, "turn_latencies_ms": [],
        }
        with patch.object(gemini_live.db, "append_transcript", new=record_transcript):
            await gemini_live._browser_loop(
                open_session, asyncio.Lock(), _Feed([
                    json.dumps({"type": "text", "text": "नमस्ते"}),
                ]), asyncio.Lock(), asyncio.Event(), call_id, open_telemetry,
                gemini_live._TurnState(), MagicMock(),
            )
        self.assertEqual(len(open_session.sent), 1)
        self.assertEqual(transcripts[0][1], "user")

    async def test_user_transcript_after_the_lock_is_not_saved(self):
        call_id = db.create_call("gemini")
        saved = []

        class _Text:
            def __init__(self, text):
                self.text = text

        class _Content:
            interrupted = False
            input_transcription = _Text("late caller words")
            output_transcription = _Text("अलविदा")
            turn_complete = False
            model_turn = None

        class _Response:
            server_content = _Content()
            tool_call = None
            tool_call_cancellation = None
            session_resumption_update = None
            go_away = None

        def record(*args, **_kwargs):
            saved.append(args)

        ws = MagicMock()
        ws.client_state = gemini_live.WebSocketState.CONNECTED
        ws.send_json = AsyncMock()
        telemetry = {
            "end_requested": True, "closing": True, "coverage": [], "errors": [],
            "audio_chunks_out": 0, "turn_latencies_ms": [],
        }
        with patch.object(gemini_live.db, "append_transcript", new=record):
            await gemini_live._handle_server_message(
                _Response(), MagicMock(), asyncio.Lock(), ws, asyncio.Lock(),
                call_id, telemetry, gemini_live._TurnState(), MagicMock(), set(), set(),
            )
        roles = [args[1] for args in saved]
        self.assertNotIn("user", roles)
        self.assertIn("assistant", roles)

    async def test_schedule_failure_does_not_raise_or_log_the_exception_text(self):
        call_id = db.create_call("gemini")
        telemetry = {"coverage": [], "force_close": True, "tool_calls": 0, "errors": []}

        class _Fc:
            id = "fc-end-2"
            name = "end_call"
            args = {}
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                return None

        async def boom(_call_id, _snapshot):
            raise RuntimeError("SECRET mother text")

        ws = MagicMock()
        ws.client_state = gemini_live.WebSocketState.CONNECTED
        ws.send_json = AsyncMock()
        await gemini_live._answer_function_calls(
            _Sess(), asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
            [_Fc()], set(), set(), on_end_accepted=boom,
        )
        self.assertTrue(telemetry["extraction_schedule_failed"])
        raw = json.dumps(db.get_call(call_id)["events"], ensure_ascii=False)
        self.assertNotIn("SECRET", raw)
        self.assertTrue(any(event.get("type") == "end_accepted" for event in db.get_call(call_id)["events"]))


class OverlapReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_extraction_overlaps_drain_and_render_uses_the_completed_call(self):
        call_id = db.create_call("gemini")
        db.append_transcript(call_id, "user", "UNIQUEPII-should-not-be-logged")
        started = asyncio.Event()
        release = asyncio.Event()
        calls = {"n": 0}
        seen = {}

        async def extract(_call):
            calls["n"] += 1
            started.set()
            await release.wait()
            return {"version": answer_extractor._VERSION, "answers": {}}

        def build(call, _extracted):
            seen["duration"] = call.get("duration_sec")
            seen["status"] = call.get("status")
            return {
                "who": "माँ", "date": "", "time": "", "duration": "", "status": "",
                "incomplete": False, "emergency": False, "questions": [], "universal": "",
                "yes": [], "groups": [], "unknown": [], "unasked": [], "not_asked": [],
                "started": None,
            }

        order = []
        reached = asyncio.Event()

        async def prepare(seen_id):
            order.append("pdf")
            reached.set()
            await report_service.mother_pdf(seen_id)
            order.append("pdf_done")
            return report_service.ready_pdf(seen_id) is not None

        def doctor(*_args, **_kwargs):
            order.append("doctor")

        recorder = MagicMock()
        recorder.save.return_value = ("call.wav", 321)
        with patch.object(answer_extractor, "cached_answers", new=extract), \
             patch.object(mother_report, "build", new=build), \
             patch.object(mother_report, "render_pdf", return_value=b"%PDF-1.4\nready"), \
             patch.object(gemini_live, "_spawn_doctor_note", new=doctor), \
             patch.object(gemini_live, "_send_json", new=AsyncMock()):
            await report_service.begin_answers(call_id, {
                "id": call_id,
                "transcript": [{"role": "user", "text": "UNIQUEPII-should-not-be-logged"}],
                "findings": {},
            })
            await started.wait()
            finishing = asyncio.create_task(gemini_live._finalize(
                MagicMock(), MagicMock(), call_id, {}, recorder, prepare,
            ))
            await reached.wait()
            self.assertEqual(order, ["pdf"])
            self.assertNotIn("duration", seen)
            release.set()
            await finishing
        self.assertEqual(calls["n"], 1)
        self.assertEqual(seen["duration"], 321)
        self.assertEqual(seen["status"], "completed")
        self.assertEqual(order, ["pdf", "pdf_done", "doctor"])
        events = [event.get("type") for event in db.get_call(call_id)["events"]]
        for name in ("render_start", "render_end", "pdf_write", "report_ready"):
            self.assertIn(name, events)
        raw = json.dumps(db.get_call(call_id)["events"], ensure_ascii=False)
        self.assertNotIn("UNIQUEPII", raw)

        with patch.object(answer_extractor, "cached_answers", new=extract), \
             patch.object(mother_report, "build", new=build), \
             patch.object(mother_report, "render_pdf", return_value=b"%PDF-1.4\nretry"):
            await report_service.mother_pdf(call_id)
        self.assertEqual(calls["n"], 2)


class ProductionReplayTestsMore(unittest.TestCase):
    def test_call_287_normal_delivery_is_not_a_concern(self):
        path = next((
            candidate for candidate in (
                Path("/tmp/shishu-production-replay/call-287.json"),
                Path("/tmp/call-287.json"),
                Path("/tmp/call_287.json"),
            ) if candidate.is_file()
        ), None)
        if path is None:
            self.skipTest("saved Call 287 replay is unavailable")
        call = json.loads(path.read_text(encoding="utf-8"))
        saved = call.get("answers") if isinstance(call.get("answers"), dict) else {}
        if saved and "answers" not in saved:
            saved = {"answers": saved}
        extracted = answer_extractor.validate_answers(call, saved)
        entry = (extracted.get("answers") or {}).get("birth_mode") or {}
        self.assertFalse(checklist.is_concern("birth_mode", entry))
        questions = answer_extractor.report_content(extracted)["questions"]
        self.assertFalse(any("प्रसव सामान्य" in line for line in questions))


if __name__ == "__main__":
    unittest.main()
