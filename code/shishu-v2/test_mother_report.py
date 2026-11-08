"""Mother PDF report. No live call. LLM is mocked except where a test says otherwise."""

import asyncio
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

        with patch.object(gemini_live.db, "get_call", return_value={"transcript": []}), \
             patch.object(gemini_live.db, "complete_call", new=complete_call), \
             patch.object(gemini_live, "_spawn_doctor_note"), \
             patch.object(gemini_live, "_send_json", new=record_send):
            await gemini_live._finalize(
                MagicMock(), MagicMock(), 42, None, recorder, prepare
            )

        self.assertEqual(events, ["ending", "db_completed", "generate:42", "call_completed"])

    async def test_generation_failure_keeps_the_call_completed_and_asks_retry(self):
        events = []
        recorder = MagicMock()
        recorder.save.return_value = ("call.wav", 12)

        async def record_send(_ws, _lock, payload):
            events.append(payload["type"])

        completed = MagicMock()

        async def prepare(_call_id):
            events.append("generate")
            return False

        with patch.object(gemini_live.db, "get_call", return_value={"transcript": []}), \
             patch.object(gemini_live.db, "complete_call", new=completed), \
             patch.object(gemini_live, "_spawn_doctor_note"), \
             patch.object(gemini_live, "_send_json", new=record_send):
            await gemini_live._finalize(
                MagicMock(), MagicMock(), 42, None, recorder, prepare
            )

        completed.assert_called_once()
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
        self.assertIn('method: "POST"', page)
        self.assertNotIn("FALLBACK_MOTHER_QUESTIONS", page)
        self.assertLess(failed, page.index("showReportRetry"))


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


if __name__ == "__main__":
    unittest.main()
