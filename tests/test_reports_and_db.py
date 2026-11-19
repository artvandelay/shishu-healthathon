"""Behavioral tests for `intake/reports.py` and `intake/db.py`.

Every PDF assertion renders a real WeasyPrint document, so a broken Devanagari font
path, a template that drifts from its context contract, or a Mac that cannot find
Pango fails here instead of on a phone at the clinic.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`
from intake import db, engine, reports
from intake.models import AnswerRecord, AnswerStatus, CallState, Phase, ProfilePatch, TurnCommand
from intake.pack import load_pack

_SHISHU = "shishu-hi"
_FEVER = "fever-opd-hi"
_SCENARIOS = {
    _SHISHU: ("red_flags", "no_flags", "incomplete"),
    _FEVER: ("urgent_fever", "clean_fever", "stopped_early"),
}
_SAMPLE_CALL = {"id": 7, "started_at": "2026-11-12 04:00:00", "duration_sec": 420}


def _scenario(pack, scenario_id: str) -> dict:
    return next(s for s in pack.seed_scenarios if s["id"] == scenario_id)


def _replay(pack, scenario_id: str) -> CallState:
    return reports.replay_seed_scenario(pack, _scenario(pack, scenario_id))[0]


class ProjectionTests(unittest.TestCase):

    def test_emergency_band_replaces_routine_filler_with_emergency_questions(self):
        pack = load_pack(_FEVER)
        state = engine.reduce_turn(
            pack,
            engine.reduce_turn(
                pack,
                engine.new_state(pack),
                TurnCommand(profile_patch=ProfilePatch(
                    slots={"patient_name": "रीना", "patient_age": 30})),
            ),
            TurnCommand(emergency=True, utterance="अभी बेहोश है"),
        )
        proj = reports.project_summary(pack, state)
        self.assertTrue(proj.emergency)
        self.assertEqual(proj.close_reason, "emergency")
        self.assertEqual(list(proj.patient_questions), list(pack.report_cfg["emergency_questions"]))
        self.assertEqual(len(proj.red_flags), 1)
        self.assertTrue(proj.red_flags[0].startswith("emergency: active danger sign now"))
        self.assertIn("hospital or 108", proj.red_flags[0])
        self.assertTrue(proj.red_flags[0].endswith("अभी बेहोश है"))
        self.assertIn("Active emergency", proj.doctor_markdown)


class Page2DetailTests(unittest.TestCase):
    """Page two must print what she actually answered, not a blanket `नहीं` on every detail row.

    Regression: `intro_married="married"` and `fever_duration_bucket="days_3_to_7"` are
    `AnswerStatus.ANSWERED`, so they landed in `page2_groups` as bare question stems and the
    template painted `<span class="ans">नहीं</span>` next to them — a married mother read
    "आपकी शादी हो गई है… नहीं" and a 3-7 day fever read "बुखार कब से है… नहीं".
    """

    MARRIED_Q = "आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई"
    DURATION_Q = "बुखार कब से है — 3 दिन से कम, 3 से 7 दिन, या 7 दिन से ज़्यादा"
    NATURE_Q = "बुखार लगातार रहता है, या बीच-बीच में आता-जाता है"
    SHISHU_CLEAR_Q = "बच्चे का जन्म ऑपरेशन (सिजेरियन) से हुआ था"
    FEVER_CLEAR_Q = "साँस लेने में तकलीफ़ है, या छाती में दर्द या जकड़न है"

    def html(self, pack_id: str, scenario_id: str) -> str:
        pack = load_pack(pack_id)
        state = _replay(pack, scenario_id)
        call = {"id": 7, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": 360}
        return reports.render_patient_html(pack, call, reports.project_summary(pack, state))

    def ans_for(self, html: str, question: str) -> str:
        row = re.search(r'<div class="no-row"><span>' + re.escape(question) + r'</span>'
                        r'<span class="ans">([^<]*)</span></div>', html)
        self.assertIsNotNone(row, f"no page-two row for {question!r}")
        return row.group(1)

    def test_answered_choice_rows_print_the_chosen_label_not_nahi(self):
        shishu, fever = self.html(_SHISHU, "no_flags"), self.html(_FEVER, "clean_fever")
        cases = (("shishu married", shishu, self.MARRIED_Q, "शादी हो गई"),
                 ("fever duration", fever, self.DURATION_Q, "3 दिन से कम"),
                 ("fever nature", fever, self.NATURE_Q, "लगातार रहता है"),
                 ("shishu clear gate", shishu, self.SHISHU_CLEAR_Q, "नहीं"),
                 ("fever clear gate", fever, self.FEVER_CLEAR_Q, "नहीं"))
        for name, html, question, expected in cases:
            with self.subTest(row=name):
                self.assertEqual(self.ans_for(html, question), expected)

    def test_projection_keeps_answered_detail_as_question_and_answer_pair(self):
        pack = load_pack(_SHISHU)
        groups = reports.project_summary(pack, _replay(pack, "no_flags")).page2_groups
        items = [item for group in groups for item in group["items"]]
        self.assertIn({"question": self.MARRIED_Q, "answer": "शादी हो गई"}, items)
        self.assertIn(self.SHISHU_CLEAR_Q, items)


class ResponseStateGlossaryTests(unittest.TestCase):
    """UNKNOWN is what she said (पता नहीं). UNANSWERED is a question the call never got an answer for.

    Call #67 wrote UNKNOWN after off-topic misses, so the report told the doctor she had said she did not know.
    """

    _CASES = {
        _SHISHU: {
            "unknown": ("birth_jaundice",
                        "जन्म के बाद पीलिया के लिए बच्चे को नीली लाइट में रखना पड़ा था",
                        "jaundice treated under blue light"),
            "unanswered": ("birth_bcg",
                           "कंधे के BCG टीके की जगह पर बड़ा फोड़ा, मवाद, या बगल में गाँठ हुई थी",
                           "big boil, pus, or armpit lump at the BCG site"),
        },
        _FEVER: {
            "unknown": ("jaundice_or_bleeding",
                        "आँखों या शरीर में पीलापन दिख रहा है, या नाक, मसूड़े या पेशाब से खून आ रहा है",
                        "jaundice, or bleeding from the nose, gums, or in the urine"),
            "unanswered": ("breathing_or_chest",
                           "साँस लेने में तकलीफ़ है, या छाती में दर्द या जकड़न है",
                           "difficulty breathing, or chest pain or tightness"),
        },
    }

    def ans_for(self, html: str, question: str) -> str:
        row = re.search(r'<div class="no-row"><span>' + re.escape(question) + r'</span>'
                        r'<span class="ans">([^<]*)</span></div>', html)
        self.assertIsNotNone(row, f"no page-two row for {question!r}")
        return row.group(1)

    def test_glossary_is_fixed_on_clinician_note_and_patient_pdf(self):
        for pack_id in (_SHISHU, _FEVER):
            pack = load_pack(pack_id)
            proj = reports.project_summary(pack, CallState())
            html = reports.render_patient_html(
                pack, {"id": 3, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": 60}, proj)
            with self.subTest(pack=pack_id, surface="clinician"):
                md = proj.doctor_markdown
                self.assertIn("### Response State Glossary", md)
                self.assertIn("Concern / हाँ — caller reported yes.", md)
                self.assertIn("Clear / नहीं — caller reported no.", md)
                self.assertIn("Unknown / पता नहीं — caller said don't know / not tested.", md)
                self.assertIn(
                    "Unanswered / जवाब नहीं हो पाया — question was not answered on the call — doctor should confirm.",
                    md)
            with self.subTest(pack=pack_id, surface="patient"):
                self.assertIn("जवाब का मतलब", html)
                self.assertIn("पता नहीं", html)
                self.assertIn("खुद कहा — पता नहीं, या जाँच नहीं हुई", html)
                self.assertIn("जवाब नहीं हो पाया", html)
                self.assertIn("कॉल पर जवाब नहीं मिला — डॉक्टर से बात करें", html)

    def test_unanswered_renders_apart_from_unknown(self):
        for pack_id, case in self._CASES.items():
            pack = load_pack(pack_id)
            unk_id, unk_stem, unk_need = case["unknown"]
            miss_id, miss_stem, miss_need = case["unanswered"]
            state = CallState(answers={
                unk_id: AnswerRecord(state=AnswerStatus.UNKNOWN),
                miss_id: AnswerRecord(state=AnswerStatus.UNANSWERED),
            })
            proj = reports.project_summary(pack, state)
            html = reports.render_patient_html(
                pack, {"id": 7, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": 90}, proj)
            with self.subTest(pack=pack_id):
                self.assertIn(f"Unknown. {unk_need}.", proj.doctor_markdown)
                self.assertIn(f"Not answered on call (confirm with patient). {miss_need}.", proj.doctor_markdown)
                self.assertNotIn(f"Unknown. {miss_need}.", proj.doctor_markdown)
                self.assertNotIn(f"Not answered on call (confirm with patient). {unk_need}.", proj.doctor_markdown)
                self.assertIn(unk_stem, proj.page2_unknown)
                self.assertNotIn(miss_stem, proj.page2_unknown)
                self.assertIn(miss_stem, proj.page2_unanswered)
                self.assertEqual(self.ans_for(html, unk_stem), "पता नहीं")
                self.assertEqual(self.ans_for(html, miss_stem), "जवाब नहीं हो पाया — डॉक्टर से बात करें")
                negatives = proj.doctor_markdown.split("### Pertinent Negatives", 1)[1]
                self.assertNotIn(miss_need, negatives)


class BuildInfoTests(unittest.TestCase):
    """The deploy stamp resolves from platform env first, then a plain `.git` read, and never raises."""

    def stamp(self, **env: str) -> dict[str, str]:
        reports.build_info.cache_clear()
        self.addCleanup(reports.build_info.cache_clear)
        with mock.patch.dict(os.environ, env, clear=True):
            return reports.build_info()

    def test_app_env_wins_and_a_bare_local_stamp_is_consistent(self):
        self.assertEqual(self.stamp(APP_ENV="prod", GIT_COMMIT="aaaaaaa", RAILWAY_PROJECT_NAME="staging")
                         ["tag"], "prod@aaaaaaa")
        local = self.stamp()
        self.assertEqual(local["env"], "local")
        self.assertEqual(local["tag"], f"local@{local['commit']}" if local["commit"] else "local")

    def test_a_missing_or_corrupt_git_directory_never_raises(self):
        reports.build_info.cache_clear()
        self.addCleanup(reports.build_info.cache_clear)
        with mock.patch.object(reports, "__file__", "/nonexistent/intake/reports.py"), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(reports.build_info(), {"env": "local", "commit": "", "tag": "local"})

    def test_the_real_ref_and_a_packed_ref_both_resolve_to_seven_hex(self):
        reports.build_info.cache_clear()
        self.addCleanup(reports.build_info.cache_clear)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            git = Path(tmp, ".git")
            (git / "refs" / "heads").mkdir(parents=True)
            (git / "refs/heads/v3").write_text("0123456789abcdef\n", encoding="utf-8")
            (git / "HEAD").write_text("ref: refs/heads/v3\n", encoding="utf-8")
            with mock.patch.object(reports, "__file__", str(Path(tmp, "intake", "reports.py"))):
                self.assertEqual(reports.build_info()["commit"], "0123456")
                reports.build_info.cache_clear()
                (git / "refs/heads/v3").unlink()
                (git / "packed-refs").write_text("fedcba9876543210 refs/heads/v3\n", encoding="utf-8")
                self.assertEqual(reports.build_info()["commit"], "fedcba9")
                reports.build_info.cache_clear()
                (git / "HEAD").write_text("not-a-sha\n", encoding="utf-8")
                self.assertEqual(reports.build_info()["commit"], "")


class PdfFilenameTests(unittest.TestCase):

    def test_a_call_without_a_start_or_an_id_still_gets_a_name(self):
        pack = load_pack(_FEVER)
        self.assertEqual(reports.pdf_filename(pack, {}), "SHISHU.pdf")
        self.assertEqual(reports.pdf_filename(pack, {"id": 12}), "SHISHU-c12.pdf")


class PatientPdfTests(unittest.TestCase):
    """Real WeasyPrint renders: these are the tests that catch a Mac with no Pango."""

    def test_render_html_carries_ist_date_time_duration_and_the_font(self):
        pack = load_pack(_FEVER)
        html = reports.render_patient_html(pack, _SAMPLE_CALL)
        self.assertIn("12 नवम्बर 2026", html)
        self.assertIn("9:30 पूर्वाह्न", html)
        self.assertIn("7 मिनट", html)
        self.assertIn("फ़ॉन्ट".replace("फ़ॉन्ट", "font-face"), html)
        self.assertIn("NotoSansDevanagari.ttf", html)
        self.assertIn("shishu-logo.png", html)
        self.assertNotIn("__FONT__", html)
        self.assertNotIn("__FOOTER__", html)

    def test_write_sample_pdfs_names_every_scenario_file(self):
        for pack_id, names in _SCENARIOS.items():
            pack = load_pack(pack_id)
            with tempfile.TemporaryDirectory() as tmp:
                written = reports.write_sample_pdfs(pack, Path(tmp))
                with self.subTest(pack=pack_id):
                    self.assertEqual([p.name for p in written],
                                     [_scenario(pack, name)["pdf"] for name in names])
                    self.assertEqual([p.parent for p in written], [Path(tmp)] * len(written))
                    for path in written:
                        self.assertTrue(path.read_bytes().startswith(b"%PDF"))


class _TempDataDir(unittest.TestCase):
    """Point `db.DATA_DIR` at a throwaway directory and empty the tables between tests."""

    def setUp(self):
        self._previous = db.DATA_DIR
        self._folder = tempfile.TemporaryDirectory()
        db.DATA_DIR = self._folder.name
        Path(db.DATA_DIR, "pdfs").mkdir(parents=True, exist_ok=True)
        with db._LOCK:
            for table in ("calls", "bug_reports", "server_secrets"):
                db.conn.execute(f"DELETE FROM {table}")
            db.conn.commit()

    def tearDown(self):
        with db._LOCK:
            for table in ("calls", "bug_reports", "server_secrets"):
                db.conn.execute(f"DELETE FROM {table}")
            db.conn.commit()
        db.DATA_DIR = self._previous
        self._folder.cleanup()


class StorageTests(_TempDataDir):

    def test_cursor_state_round_trips_through_save_and_load(self):
        pack = load_pack(_SHISHU)
        call_id = db.create_call()
        state = engine.reduce_turn(
            pack, engine.new_state(pack),
            TurnCommand(profile_patch=ProfilePatch(slots={"caller_name": "कविता"})))
        db.save_cursor_state(call_id, pack, state)
        call = db.get_call(call_id)
        self.assertEqual(call["patient_name"], "कविता")
        loaded = db.call_state_from_row(call)
        self.assertEqual(loaded.to_dict(), state.to_dict())
        self.assertEqual(loaded.cursor.qid, "intro_married")
        blank = db.call_state_from_row(None)
        self.assertEqual((blank.phase, blank.answers), (Phase.ASKING, {}))
        broken = db.call_state_from_row({"cursor_json": "{not json"})
        self.assertEqual(broken.to_dict(), blank.to_dict())
        self.assertEqual(db.call_state_from_row({"cursor_json": ""}).to_dict(), blank.to_dict())

    def test_transcript_merges_chunks_of_one_turn_and_keeps_other_turns_apart(self):
        call_id = db.create_call()
        db.append_transcript(call_id, "user", "   ")
        self.assertEqual(db.get_call(call_id)["transcript"], [])
        db.append_transcript(call_id, "user", "क्या हाँ है", qid="birth_nicu")
        db.append_transcript(call_id, "user", "दो दिन", qid="birth_nicu")
        db.append_transcript(call_id, "assistant", "समझ गई", qid="birth_nicu")
        db.append_transcript(call_id, "user", "अब आगे", qid="body_fits")
        turns = db.get_call(call_id)["transcript"]
        self.assertEqual([t["role"] for t in turns], ["user", "assistant", "user"])
        self.assertEqual(turns[0]["text"], "क्या हाँ है दो दिन")
        self.assertEqual([t["qid"] for t in turns], ["birth_nicu", "birth_nicu", "body_fits"])

    def test_bug_reports_store_the_client_snapshot_without_the_raw_column(self):
        call_id = db.create_call()
        db.append_transcript(call_id, "user", "कुछ सुनाई नहीं दिया", qid="intro_name")
        db.update_call_telemetry(call_id, {"latency_ms": 90})
        shot = "data:image/png" + "\u003bbase64,AAAA"
        bug = db.create_bug_report(call_id, "Tester", "Other", "कॉल अटक गई", 12,
                                   {"screenshot": shot}, include_call_body=True)
        self.assertEqual((bug["call_id"], bug["category"], bug["call_elapsed_sec"]),
                         (call_id, "Other", 12))
        listed = next(row for row in db.list_bug_reports() if row["id"] == bug["id"])
        self.assertNotIn("snapshot_json", listed)
        self.assertEqual(listed["snapshot"]["client"]["screenshot"], shot)
        self.assertEqual(listed["snapshot"]["telemetry"], {"latency_ms": 90})
        self.assertEqual(len(listed["snapshot"]["last_transcript"]), 1)
        self.assertEqual(db.get_call(call_id)["events"][-1]["type"], "bug_reported")

    def test_reopen_call_marks_completed_call_active_and_clears_ended_at(self):
        """A caller who comes back on the same call id must get a live call row, kept audio path and all."""
        call_id = db.create_call(pack_id=_SHISHU)
        db.complete_call(call_id, 12, "call_1.wav")
        finished = db.get_call(call_id)
        self.assertEqual(finished["status"], "completed")
        self.assertIsNotNone(finished["ended_at"])

        db.reopen_call(call_id)

        row = db.get_call(call_id)
        self.assertEqual(row["status"], "active")
        self.assertIsNone(row["ended_at"])
        self.assertEqual(row["audio_path"], "call_1.wav")

    def test_ready_pdf_ignores_a_truncated_file(self):
        call_id = db.create_call()
        path = reports.patient_pdf_path(call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")
        self.assertIsNone(reports.ready_pdf(call_id))
        path.write_bytes(b"")
        self.assertIsNone(reports.ready_pdf(call_id))

    def test_concurrent_asks_for_one_pdf_share_a_single_build(self):
        pack = load_pack(_SHISHU)
        call_id = db.create_call()
        db.save_cursor_state(call_id, pack, _replay(pack, "no_flags"))
        db.complete_call(call_id, 360)
        builds = []
        real = reports.build_patient_pdf_sync

        def counting(call_id_arg):
            builds.append(call_id_arg)
            return real(call_id_arg)

        async def race():
            original = reports.build_patient_pdf_sync
            reports.build_patient_pdf_sync = counting
            try:
                return await asyncio.gather(
                    *(reports.ensure_patient_pdf(call_id) for _ in range(3)))
            finally:
                reports.build_patient_pdf_sync = original

        results = asyncio.run(race())
        self.assertEqual(len(builds), 1)
        self.assertTrue(all(pdf.startswith(b"%PDF") for pdf, _name in results))
        self.assertEqual({name for _pdf, name in results}.__len__(), 1)
        self.assertEqual(reports._inflight, {})

    def test_ensure_patient_pdf_raises_for_a_missing_call(self):
        with self.assertRaises(LookupError):
            asyncio.run(reports.ensure_patient_pdf(987654))
        self.assertEqual(reports._inflight, {})


if __name__ == "__main__":
    unittest.main()
