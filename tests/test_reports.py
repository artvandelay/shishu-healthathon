"""Stage 4: one typed projection, one WeasyPrint PDF, and no diagnosis in Hindi."""

import asyncio
import io
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import tests  # noqa: F401

import app as shishu_app  # noqa: E402
import copy_hi  # noqa: E402
import db  # noqa: E402
import engine  # noqa: E402
import mother_report  # noqa: E402
import post_call  # noqa: E402
import report_service  # noqa: E402
import seed  # noqa: E402
import summary  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from models import (  # noqa: E402
    CallState,
    ChildInput,
    ChildSex,
    MaritalStatus,
    Phase,
    ProfilePatch,
    TurnCommand,
    TurnOutcome,
)
from pypdf import PdfReader  # noqa: E402
from triage_catalog import CATALOG, get_spec  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLES = _REPO_ROOT / "docs" / "samples"
_CALL = {"id": 1, "started_at": "2026-11-12 04:00:00", "duration_sec": 180}
_CHILD_QUESTIONS = tuple(spec.question_hi for spec in CATALOG if "no_children" in spec.skip_if)
_FORBIDDEN = (
    "Duchenne", "Hemophilia", "Haemophilia", "Pompe", "Gaucher", "SCID",
    "Wiskott", "SMA", "MPS", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे", "गॉशर",
)
_DIAGNOSTIC_NOTE = "डॉक्टर ने Hemophilia और ड्यूशेन और Pompe बताया"
_SAFE_NOTE = "चोट पर खून देर तक बहता है"


def _kids(*specs):
    return tuple(
        ChildInput(age_years=age, sex=ChildSex(sex), name=name) for age, sex, name in specs
    )


def _state(children=(), *, name="पूजा", marital=MaritalStatus.MARRIED):
    return engine.apply_profile_patch(
        CallState(),
        ProfilePatch(caller_name=name, marital_status=marital, children=tuple(children)),
    )


def _to_end(state, limit: int = 200):
    for _ in range(limit):
        if state.phase is not Phase.ASKING:
            break
        state = engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
    return state


def _clear_until(state, qid, limit: int = 90):
    for _ in range(limit):
        if state.phase is not Phase.ASKING or state.cursor.qid == qid:
            break
        state = engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
    return state


def _complete_call():
    state = _state(_kids((7.0, "male", "आरव"), (3.0, "female", "मीरा")))
    state = _clear_until(state, "body_bleeding_bruise")
    state = engine.reduce_turn(
        state,
        TurnCommand(outcome=TurnOutcome.CONCERN, utterance=_SAFE_NOTE, child_ref="child_2"),
    )
    state = _clear_until(state, "preg_thalassemia")
    state = engine.reduce_turn(
        state, TurnCommand(outcome=TurnOutcome.CONCERN, utterance="थैलेसीमिया जाँच में माइनर बताया")
    )
    state = _clear_until(state, "close_anything_else")
    state = engine.reduce_turn(
        state,
        TurnCommand(outcome=TurnOutcome.CONCERN, utterance=_DIAGNOSTIC_NOTE, diagnostic=True),
    )
    return _to_end(state)


def _childless_call():
    state = _state((), name="गीता")
    return _to_end(state)


def _emergency_call():
    state = _state(_kids((3.0, "female", "मीरा")))
    state = _clear_until(state, "body_bleeding_bruise")
    state = engine.reduce_turn(
        state, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=_SAFE_NOTE, child_ref="child_1")
    )
    return engine.reduce_turn(state, TurnCommand(emergency=True, utterance="अभी बेहोश है"))


def _stopped_call():
    state = _state(_kids((4.0, "female", None)), name="सुनीता")
    state = _clear_until(state, "ill_fasting")
    return engine.reduce_turn(state, TurnCommand(caller_stop=True))


def _pages(pdf: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf)).pages)


def _pdf_text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)


def _page2(state: CallState) -> str:
    call = {**_CALL, "cursor_state": state.to_dict()}
    return mother_report.render_html(mother_report.build_report(call)).split('class="page2"', 1)[1]


def _store(state: CallState, mother_name: str = "पूजा") -> int:
    call_id = db.create_call(caller_name=mother_name)
    db.save_cursor_state(call_id, state)
    db.complete_call(call_id, 180)
    return call_id


class ProjectionTests(unittest.TestCase):
    def test_completed_call_projects_one_summary_for_every_builder(self):
        state = _complete_call()
        projected = summary.project_summary(state)
        self.assertEqual((projected.close_reason, projected.incomplete), ("complete", False))
        self.assertTrue(projected.red_flags)

        built = post_call.build_reports({"id": 1, "cursor_state": state.to_dict()})
        self.assertEqual(built["doctor_markdown"], projected.doctor_markdown)
        self.assertEqual(built["red_flags"], list(projected.red_flags))
        self.assertEqual(built["mother_name_hi"], "पूजा")
        self.assertEqual(built["children_label_hi"], "आरव 7 साल बेटा · मीरा 3 साल बेटी")
        self.assertEqual(built["summary"], projected)

        report = mother_report.build_report({**_CALL, "cursor_state": state.to_dict()}, projected)
        self.assertEqual(report["questions"], list(projected.mother_questions_hi))
        self.assertEqual(report["page2_groups"], [
            {"title": group.title, "items": list(group.items)}
            for group in projected.page2_clear_groups
        ])
        self.assertEqual(report["page2_unasked"], [])
        self.assertEqual(report["date_hi"], "12 नवम्बर 2026")

    def test_typed_state_and_raw_mapping_rows_resolve_identically(self):
        state = _complete_call()
        as_model = db.call_state_from_row({"cursor_state": state})
        as_mapping = db.call_state_from_row({"cursor_state": state.to_dict()})
        self.assertIs(as_model, state)
        self.assertEqual(as_mapping.to_dict(), state.to_dict())
        blank = db.call_state_from_row(None)
        self.assertEqual((blank.phase, blank.cursor.qid), (Phase.ASKING, "intro_name"))
        self.assertEqual(db.call_state_from_row({"cursor_state": {}}).to_dict(), blank.to_dict())

    def test_mother_summary_text_matches_the_projection_from_a_model_or_a_dict(self):
        projected = summary.project_summary(_complete_call())
        built = post_call.build_reports(_complete_call())
        self.assertEqual(post_call.mother_summary_text(projected), projected.mother_summary_text)
        self.assertEqual(post_call.mother_summary_text(built), projected.mother_summary_text)
        self.assertEqual(post_call.mother_summary_text({}), copy_hi.mother_summary_text([], []))

    def test_diagnostic_words_stay_in_the_doctor_note_and_never_reach_the_pdf(self):
        state = _complete_call()
        built = post_call.build_reports({"id": 43, "cursor_state": state.to_dict()})
        for heading in summary._HEADINGS:
            self.assertIn(heading, built["doctor_markdown"])
        self.assertIn(_DIAGNOSTIC_NOTE, built["doctor_markdown"])
        self.assertIn("child_2 (female, age 3.0, मीरा)", built["doctor_markdown"])
        self.assertTrue(any(_DIAGNOSTIC_NOTE in flag for flag in built["red_flags"]))
        self.assertIn(_SAFE_NOTE, built["yes_summaries_hi"][0])
        self.assertIn("मीरा 3 साल बेटी", built["yes_summaries_hi"][0])

        report = mother_report.build_report({"id": 43, "cursor_state": state.to_dict()})
        pdf_text = _pdf_text(mother_report.render_pdf(report))
        mother_blob = "\n".join([
            *built["yes_summaries_hi"],
            *built["mother_questions_hi"],
            mother_report.render_html(report),
            pdf_text,
        ]).lower()
        for name in _FORBIDDEN:
            self.assertNotIn(name.lower(), mother_blob, name)
        for latin in ("Hemophilia", "Pompe"):
            self.assertNotIn(latin.lower(), pdf_text.lower(), latin)

    def test_childless_pdf_omits_every_child_question_and_stays_two_to_three_pages(self):
        state = _childless_call()
        projected = summary.project_summary(state)
        self.assertEqual((projected.children_label_hi, projected.yes_summaries_hi), ("", ()))
        self.assertNotIn("मेरे बच्चे", " ".join(projected.mother_questions_hi))

        report = mother_report.build_report({**_CALL, "cursor_state": state.to_dict()})
        page2 = _page2(state)
        for question in _CHILD_QUESTIONS:
            self.assertNotIn(question, page2, question[:24])
        self.assertIn(get_spec("fam_related").question_hi, page2)

        self.assertIn("यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है।", mother_report.render_html(report))
        pdf = mother_report.render_pdf(report)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertTrue(2 <= _pages(pdf) <= 3)
        self.assertIn("SHISHU-1", _pdf_text(pdf))

    def test_multi_child_pdf_names_the_child_and_carries_the_concern(self):
        report = mother_report.build_report({**_CALL, "cursor_state": _complete_call().to_dict()})
        document = mother_report.render_html(report)
        self.assertIn("मीरा 3 साल बेटी", document)
        self.assertIn(_SAFE_NOTE, document)
        self.assertIn("पूर्ण कॉल", document)
        self.assertTrue(2 <= _pages(mother_report.render_pdf(report)) <= 3)

    def test_emergency_pdf_keeps_the_danger_banner_and_drops_routine_filler(self):
        state = _emergency_call()
        projected = summary.project_summary(state)
        emergency_flag = "emergency: active danger sign now; hospital or 108"
        self.assertEqual(projected.red_flags[0], f"{emergency_flag}. अभी बेहोश है")
        self.assertEqual(list(projected.mother_questions_hi[1:]), list(copy_hi.EMERGENCY_HI[:2]))
        self.assertTrue(all(q not in projected.mother_questions_hi for q in copy_hi.ROUTINE_HI))

        report = mother_report.build_report({**_CALL, "cursor_state": state.to_dict()})
        document = mother_report.render_html(report)
        self.assertTrue("108" in document and "नज़दीकी अस्पताल" in document)
        self.assertNotIn("यह कॉल अधूरी रही।", document)
        self.assertIn("108", _pdf_text(mother_report.render_pdf(report)))

    def test_incomplete_pdf_shows_the_unasked_section(self):
        state = _stopped_call()
        projected = summary.project_summary(state)
        self.assertEqual((projected.close_reason, projected.incomplete), ("caller_stop", True))
        self.assertTrue(projected.page2_unasked_groups)

        report = mother_report.build_report({**_CALL, "cursor_state": state.to_dict()})
        document = mother_report.render_html(report)
        self.assertIn("यह कॉल अधूरी रही।", document)
        self.assertIn("अभी नहीं पूछा गया", document)
        self.assertIn("पूछा नहीं गया", document)
        self.assertNotIn("108", document)
        self.assertTrue(2 <= _pages(mother_report.render_pdf(report)) <= 3)


class PersistenceTests(unittest.TestCase):
    def test_report_service_saves_every_text_field_and_stores_one_pdf(self):
        bleed, calm = _store(_complete_call(), "Pooja"), _store(_childless_call(), "Pooja")
        for call_id in (bleed, calm):
            pdf, name = asyncio.run(report_service.mother_pdf(call_id))
            self.assertTrue(pdf.startswith(b"%PDF") and name.endswith(".pdf"))
            self.assertEqual(report_service._inflight, {})

        stored = db.get_call(bleed)
        self.assertTrue(stored["summary_doctor"].strip() and stored["summary_mother"].strip())
        self.assertTrue(stored["red_flags"])
        self.assertEqual(stored["mother_name"], "पूजा")
        self.assertIn("साल", stored["child_age"])
        self.assertEqual(db.get_call(calm)["red_flags"], [])
        stored_pdf, _name = report_service.ready_pdf(bleed)
        self.assertTrue(stored_pdf.startswith(b"%PDF"))
        self.assertEqual(_pages(stored_pdf), 3)

    def test_save_cursor_state_accepts_a_model_or_a_mapping_and_clears_labels(self):
        call_id = db.create_call(caller_name="पूजा")
        db.save_cursor_state(call_id, _state(_kids((5.0, "male", None))))
        self.assertEqual(db.get_call(call_id)["mother_name"], "पूजा")
        self.assertEqual(db.get_call(call_id)["child_age"], "5 साल बेटा")

        blank = {"profile": {"caller_name": "", "children": []}, "answers": {}}
        db.save_cursor_state(call_id, blank)
        self.assertEqual((db.get_call(call_id)["mother_name"], db.get_call(call_id)["child_age"]), ("", ""))

        db.save_cursor_state(call_id, None)
        self.assertEqual(db.get_call(call_id)["cursor_state"]["profile"]["children"], [])

    def test_append_transcript_merges_a_chunked_turn_and_skips_blank_text(self):
        call_id = db.create_call(caller_name="पूजा")
        db.append_transcript(call_id, "user", "   \n\t  ")
        self.assertEqual(db.get_call(call_id)["transcript"], [])
        db.append_transcript(call_id, "user", "चोट पर खून", qid="body_bleeding_bruise")
        db.append_transcript(call_id, "user", "देर तक बहता है", qid="body_bleeding_bruise")
        transcript = db.get_call(call_id)["transcript"]
        self.assertEqual(len(transcript), 1)
        self.assertIn("चोट पर खून", transcript[0]["text"])
        self.assertIn("देर तक बहता है", transcript[0]["text"])

    def test_events_are_capped_and_timing_meta_is_filtered(self):
        call_id = db.create_call(caller_name="पूजा")
        for index in range(300):
            db.append_call_event(call_id, f"evt_{index}")
        self.assertEqual(len(db.get_call(call_id)["events"]), 300)

        report_service._event(call_id, "render_end", {"ms": 42, "bytes": 1024, "model": "old"})
        self.assertEqual(db.get_call(call_id)["events"][-1]["meta"], {"ms": 42, "bytes": 1024})

    def test_bug_report_snapshot_round_trips_without_the_raw_column(self):
        call_id = _store(_childless_call(), "पूजा")
        bug = db.create_bug_report(
            call_id, "Tester", "Other", "shot", 5,
            extra_snapshot={"screenshot": "data:image/png;base64,AAAA"},
        )
        listed = next(item for item in db.list_bug_reports() if item["id"] == bug["id"])
        self.assertNotIn("snapshot_json", listed)
        self.assertEqual(listed["snapshot"]["client"]["screenshot"], "data:image/png;base64,AAAA")


class EndpointTests(unittest.TestCase):
    def test_token_endpoint_and_admin_endpoint_serve_the_same_saved_pdf(self):
        calm_id, bleed_id = _store(_childless_call(), "Pooja"), _store(_complete_call(), "Pooja")
        calm, bleed = db.get_call(calm_id), db.get_call(bleed_id)
        self.assertNotEqual(calm["share_token"], bleed["share_token"])

        with TestClient(shishu_app.app) as client:
            anonymous = client.get(f"/api/calls/by-token/{calm['share_token']}/mother.pdf")
            self.assertEqual(anonymous.status_code, 409)

            calm_pdf = client.post(f"/api/calls/by-token/{calm['share_token']}/mother.pdf")
            bleed_pdf = client.post(f"/api/calls/by-token/{bleed['share_token']}/mother.pdf")
            self.assertEqual((calm_pdf.status_code, bleed_pdf.status_code), (200, 200))
            self.assertTrue(calm_pdf.content.startswith(b"%PDF"))
            self.assertTrue(bleed_pdf.content.startswith(b"%PDF"))
            self.assertNotEqual(calm_pdf.content, bleed_pdf.content)
            self.assertTrue(all(
                2 <= _pages(rendered.content) <= 3 for rendered in (calm_pdf, bleed_pdf)
            ))

            fetched = client.get(f"/api/calls/by-token/{calm['share_token']}/mother.pdf")
            self.assertEqual(fetched.content, calm_pdf.content)
            self.assertEqual(client.get("/api/calls/by-token/not-a-real-token/mother.pdf").status_code, 404)

            self.assertEqual(client.get(f"/api/calls/{calm_id}/mother.pdf").status_code, 401)
            client.post("/api/login", json={"username": "admin", "passcode": "admin-test"})
            admin_pdf = client.get(f"/api/calls/{calm_id}/mother.pdf")
            self.assertEqual(admin_pdf.status_code, 200)
            self.assertEqual(admin_pdf.content, calm_pdf.content)

        self.assertIn("खून", db.get_call(bleed_id)["summary_mother"])
        self.assertNotIn("खून", db.get_call(calm_id)["summary_mother"])

    def test_overlapping_builds_share_one_render_and_survive_cancellation(self):
        call_id = _store(_childless_call(), "पूजा")
        entered, release, seen = threading.Event(), threading.Event(), []
        real_render = mother_report.render_pdf

        def slow_render(report):
            seen.append(report["call_id"])
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("timeout")
            return real_render(report)

        async def run():
            with patch.object(mother_report, "render_pdf", side_effect=slow_render):
                first = asyncio.create_task(report_service.mother_pdf(call_id))
                second = asyncio.create_task(report_service.mother_pdf(call_id))
                while not entered.is_set():
                    await asyncio.sleep(0.01)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                release.set()
                return await second

        pdf, name = asyncio.run(run())
        self.assertTrue(pdf.startswith(b"%PDF") and name.endswith(".pdf"))
        self.assertEqual(seen, [call_id])
        self.assertEqual(report_service._inflight, {})


class SamplePdfTests(unittest.TestCase):
    def test_all_three_sample_pdfs_are_valid_devnagari_documents(self):
        for name, report in seed.sample_reports().items():
            with self.subTest(name):
                path = _SAMPLES / name
                self.assertTrue(path.is_file(), name)
                pdf = path.read_bytes()
                self.assertTrue(pdf.startswith(b"%PDF"))
                self.assertTrue(2 <= _pages(pdf) <= 3, name)
                self.assertIn("SHISHU", _pdf_text(pdf))

                document = mother_report.render_html(report).lower()
                self.assertIn("यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है।", document)
                for banned in _FORBIDDEN:
                    self.assertNotIn(banned.lower(), document, f"{banned} in {name}")
                    self.assertNotIn(banned.lower(), _pdf_text(pdf).lower(), f"{banned} in {name}")

    def test_sample_bodies_match_their_intent(self):
        reports = seed.sample_reports()
        red, calm, partial = (
            mother_report.render_html(reports[name])
            for name in (
                "SHISHU-sample-red-flags.pdf",
                "SHISHU-sample-no-flags.pdf",
                "SHISHU-sample-incomplete.pdf",
            )
        )
        self.assertNotIn("108", red)
        self.assertIn("इस कॉल में कोई बात हाँ में नहीं आई।", calm)
        self.assertNotIn("अभी नहीं पूछा गया", calm)
        self.assertIn("यह कॉल अधूरी रही।", partial)
        self.assertIn("अभी नहीं पूछा गया", partial)

    def test_samples_are_reproducible_from_the_seed_scenarios(self):
        written = seed.write_sample_pdfs(_SAMPLES)
        self.assertEqual(
            [path.name for path in written],
            [
                "SHISHU-sample-red-flags.pdf",
                "SHISHU-sample-no-flags.pdf",
                "SHISHU-sample-incomplete.pdf",
            ],
        )
        for path in written:
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            self.assertTrue(2 <= _pages(path.read_bytes()) <= 3, path.name)


if __name__ == "__main__":
    unittest.main()
