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
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`
from intake import db, engine, reports
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CursorStep, NoteText, Phase, ProfilePatch, RosterItem, TurnCommand,
)
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
        # The old "questions to ask the doctor" filler is gone: an emergency adds no advice list, routine or otherwise.
        self.assertEqual(proj.patient_questions, ())
        self.assertEqual(len(proj.red_flags), 1)
        self.assertTrue(proj.red_flags[0].startswith("emergency: active danger sign now"))
        self.assertIn("hospital or 108", proj.red_flags[0])
        self.assertTrue(proj.red_flags[0].endswith("अभी बेहोश है"))
        self.assertIn("Active emergency", proj.doctor_markdown)
        # The sheet still says an emergency was reported, and still prints no question list at all.
        html = reports.render_patient_html(pack, _SAMPLE_CALL, proj)
        self.assertIn("इस कॉल में इमरजेंसी की बात दर्ज हुई।", html)
        self.assertNotIn("डॉक्टर से पूछने के सवाल", html)


class EmergencyReportTests(unittest.TestCase):
    """The two surfaces the Aanchal demo grades an emergency on: the mother's band and the clinician's first line.

    Both regressed on a real demo call (`scripts/run_demo_aanchal.py`): the emergency PDF printed only
    "इस कॉल में इमरजेंसी की बात दर्ज हुई।" with no 108 or nearest-hospital line, and the clinician note opened
    "Chief complaint: NICU or glass-box stay after birth." — the first recorded concern, not the emergency that
    ended the call.
    """

    CALL = {"id": 7, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": 420}
    EMERGENCY = "नहीं, लेकिन मेरी छोटी बच्ची अभी बहुत तेज़ साँस ले रही है"
    NICU = "NICU or glass-box stay after birth"

    def emergency_state(self, pack, concern_qid: str, concern_note: str) -> CallState:
        """A concern already on record, then an active emergency on the next turn — exactly the demo's order."""
        st = CallState(
            profile={("caller_name" if pack.roster_cfg else "patient_name"): "आँचल", "marital_status": "married"},
            roster=[RosterItem(id="item_1", age_years=8, age_label="8 साल", tag="male"),
                    RosterItem(id="item_2", age_years=5, age_label="5 साल", tag="female")] if pack.roster_cfg else [],
            roster_stated=bool(pack.roster_cfg),
            answers={concern_qid: AnswerRecord(state=AnswerStatus.CONCERN, note=NoteText(verbatim=concern_note))},
            cursor=CursorStep(qid="close_anything_else"),
        )
        return engine.reduce_turn(pack, st, TurnCommand(emergency=True, utterance=self.EMERGENCY))

    @staticmethod
    def band(html: str) -> str:
        """Only the emergency banner, so a `108` in the page-two rows cannot make this pass."""
        match = re.search(r'<div class="band">(.*?)</div>', html, re.S)
        return match.group(1) if match else ""

    def test_emergency_band_carries_108_and_the_nearest_hospital_in_both_languages(self):
        for pack_id, concern_qid, concern_note in ((_SHISHU, "birth_nicu", "हाँ भर्ती हुई थी मेरी छोटी बेटी"),
                                                   (_FEVER, "fever_severity", "बहुत तेज़ बुखार है")):
            pack = load_pack(pack_id)
            proj = reports.project_summary(pack, self.emergency_state(pack, concern_qid, concern_note))
            self.assertTrue(proj.emergency, pack_id)
            for lang, hospital in (("hi", "अस्पताल"), ("en", "hospital")):
                with self.subTest(pack=pack_id, lang=lang):
                    band = self.band(reports.render_patient_html(pack, self.CALL, proj, lang=lang))
                    self.assertTrue(band, f"{pack_id}/{lang} rendered no emergency band")
                    self.assertIn("108", band)
                    self.assertIn(hospital, band.lower() if lang == "en" else band)
                    # The bare statement stays: the band still says an emergency was reported on this call.
                    self.assertIn("emergency was reported" if lang == "en" else "इमरजेंसी", band)

    def test_the_written_pdf_itself_carries_the_108_line(self):
        """`scripts/run_demo_aanchal.py` grades the downloaded PDF's text, not this test's HTML."""
        from pypdf import PdfReader
        pack = load_pack(_SHISHU)
        proj = reports.project_summary(pack, self.emergency_state(pack, "birth_nicu", "हाँ भर्ती हुई थी मेरी छोटी बेटी"))
        for lang in ("hi", "en"):
            with self.subTest(lang=lang):
                blob = reports.render_patient_pdf(pack, self.CALL, proj, lang=lang)
                self.assertTrue(blob.startswith(b"%PDF"))
                text = " ".join(" ".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(blob)).pages).split())
                self.assertIn("108", text)
                # The band is page one, before the "yes" list: a `108` anywhere lower would not be the banner.
                self.assertLess(text.index("108"), text.index('Reported "Yes"' if lang == "en" else "इस कॉल में जो"))

    def test_chief_complaint_leads_with_the_active_emergency_not_the_first_concern(self):
        for pack_id, concern_qid, concern_note, concern_need in (
            (_SHISHU, "birth_nicu", "हाँ भर्ती हुई थी मेरी छोटी बेटी", self.NICU),
            (_FEVER, "fever_severity", "बहुत तेज़ बुखार है, बिस्तर पर रहना पड़ता है",
             "how severe the fever is, from mild warmth to being unable to stand or eat"),
        ):
            pack = load_pack(pack_id)
            md = reports.project_summary(pack, self.emergency_state(pack, concern_qid, concern_note)).doctor_markdown
            chief = next((line for line in md.splitlines() if line.startswith("- Chief complaint:")), "")
            with self.subTest(pack=pack_id):
                self.assertTrue(chief, "the clinician note has no chief complaint line")
                self.assertTrue(chief.startswith("- Chief complaint: Active emergency"), chief)
                self.assertIn("108", chief)
                self.assertIn(self.EMERGENCY, chief, "her own words must lead the chief complaint")
                # The earlier concern is still on record for the doctor, but never as the chief complaint.
                self.assertNotIn(concern_need, chief)
                self.assertIn(concern_need, md.split("### Positive Red Flags", 1)[1])


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
                # Factual legend: "no answer recorded on call", never "talk to the doctor".
                self.assertIn("कॉल पर जवाब नहीं मिला", html)
                self.assertNotIn("डॉक्टर से बात करें", html)

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
                # The unanswered row states the fact (no answer on the call); it no longer tells her to see a doctor.
                self.assertEqual(self.ans_for(html, miss_stem), "जवाब नहीं हो पाया")
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
        # Every sheet is language-tagged: the Hindi file is not allowed to look like the English one.
        self.assertEqual(reports.pdf_filename(pack, {}), "SHISHU-hi.pdf")
        self.assertEqual(reports.pdf_filename(pack, {}, "en"), "SHISHU-en.pdf")
        self.assertEqual(reports.pdf_filename(pack, {"id": 12}), "SHISHU-c12-hi.pdf")
        self.assertEqual(reports.pdf_filename(pack, {"id": 12}, "en"), "SHISHU-c12-en.pdf")


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


class SeedScenarioReportTests(unittest.TestCase):
    """Each seeded scenario must keep the clinical content it exists to demonstrate.

    A seed that silently degrades to "0 red flags" or "complete" still writes a PDF, so file-existence smoke tests
    pass on a pack whose danger signs stopped being reported. These assert the actual flag text, the actual
    all-clear, and the actual stopped-early inventory, on the clinician note, the projection and the rendered page.
    """

    CALL = {"id": 7, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": 420}

    def report(self, pack_id: str, scenario_id: str) -> tuple[reports.ProjectedReport, str]:
        pack = load_pack(pack_id)
        proj = reports.project_summary(pack, _replay(pack, scenario_id))
        return proj, reports.render_patient_html(pack, self.CALL, proj)

    @staticmethod
    def alerts(proj: reports.ProjectedReport) -> str:
        """The clinician note's red-flag block, so an assertion cannot pass on text from another section."""
        return proj.doctor_markdown.split("### Positive Red Flags", 1)[1].split("\n### ", 1)[0]

    def test_shishu_red_flags_reports_every_seeded_gate_with_the_mothers_words(self):
        proj, html = self.report(_SHISHU, "red_flags")
        flags = self.alerts(proj)
        self.assertEqual(len(proj.red_flags), 4)
        # Each seeded answer is a different clinical concern and each must reach the doctor.
        for tag_id, need, said in (
            ("Q03 birth_jaundice", "jaundice treated under blue light", "हाँ, तीन दिन नीली लाइट में रखा था"),
            ("Q04 body_joint_swelling", "knee or joint swelling after a fall", "हाँ, गिरने पर घुटने में सूजन 5 से 7 दिन रहती है"),
            ("Antenatal preg_conditions", "high BP, sugar, or thyroid in a pregnancy",
             "हाँ, हाई बीपी और थायरॉइड की दवा डिलीवरी के बाद भी चल रही है"),
            ("Q12 fam_infant_death", "a baby in the family died before 1 year", "झटके आने से उसकी मौत हुई थी"),
        ):
            with self.subTest(flag=tag_id):
                line = next((f for f in proj.red_flags if f.startswith(f"{tag_id}:")), None)
                self.assertIsNotNone(line, f"no red flag for {tag_id} in {proj.red_flags}")
                self.assertIn(need, line)
                self.assertIn(said, line)
                self.assertIn(need, flags)
                self.assertIn(said, flags)
        # The two-page patient sheet repeats them as the "हाँ" list she asked the doctor about.
        yes_list = re.search(r'<ul class="yes">(.*?)</ul>', html, re.S)
        self.assertIsNotNone(yes_list)
        for said in ("नीली लाइट में रखा था", "घुटने में सूजन", "हाई बीपी और थायरॉइड", "झटके आने से उसकी मौत"):
            with self.subTest(yes_item=said):
                self.assertIn(said, yes_list.group(1))
        self.assertFalse(proj.incomplete)
        self.assertNotIn("यह कॉल अधूरी रही", html)

    def test_shishu_no_flags_is_a_clean_call_on_every_surface(self):
        proj, html = self.report(_SHISHU, "no_flags")
        self.assertEqual(proj.red_flags, ())
        self.assertEqual(proj.yes_items, ())
        self.assertEqual(self.alerts(proj).strip().splitlines(), ["& NRROID Gate Alerts", "- No positive gate alerts."])
        self.assertEqual(proj.page2_unknown, ("इसके अलावा डॉक्टर के लिए कुछ और बताना है जो आपके मन में हो",))
        self.assertEqual(proj.page2_unanswered, ())
        self.assertIn("इस कॉल में कोई बात हाँ में दर्ज नहीं हुई।", html)
        self.assertNotIn('<ul class="yes">', html)
        self.assertFalse(proj.incomplete)
        self.assertNotIn("यह कॉल अधूरी रही", html)

    def test_shishu_incomplete_marks_the_call_partial_and_lists_what_was_never_asked(self):
        proj, html = self.report(_SHISHU, "incomplete")
        self.assertEqual(proj.close_reason, "caller_stop")
        self.assertTrue(proj.incomplete)
        # The one answered gate is still reported; nothing answered must be invented for the rest.
        self.assertEqual(len(proj.red_flags), 1)
        self.assertIn("Q03 birth_nicu: NICU or glass-box stay after birth", proj.red_flags[0])
        self.assertIn("दो दिन काँच के बक्से में रखा पड़ा था", proj.red_flags[0])
        unasked = [item for group in proj.page2_unasked for item in group["items"]]
        self.assertIn("जन्म के बाद पीलिया के लिए बच्चे को नीली लाइट में रखना पड़ा था", unasked)
        self.assertIn("परिवार में किसी छोटे बच्चे की 1 साल से कम उम्र में मौत हुई है", unasked)
        self.assertEqual(len(unasked), 50)
        self.assertIn("Call ended before every applicable question was asked.", proj.doctor_markdown)
        self.assertIn("यह कॉल अधूरी रही।", html)
        self.assertIn("अभी नहीं पूछा गया", html)
        self.assertNotIn("पूर्ण कॉल", html)

    def test_fever_urgent_fever_reports_the_red_flag_gate_and_the_fever_gates(self):
        proj, html = self.report(_FEVER, "urgent_fever")
        flags = self.alerts(proj)
        self.assertEqual(len(proj.red_flags), 7)
        # The seeded danger sign: breathlessness on stairs, with its own follow-up, under the RED_FLAG gate.
        breathing = next(f for f in proj.red_flags if f.startswith("RED_FLAG breathing_or_chest:"))
        self.assertIn("difficulty breathing, or chest pain or tightness", breathing)
        self.assertIn("हाँ, सीढ़ियाँ चढ़ने पर साँस फूलती है", breathing)
        self.assertIn("- RED_FLAG breathing_or_chest: difficulty breathing, or chest pain or tightness. "
                      "Mother's words: हाँ, सीढ़ियाँ चढ़ने पर साँस फूलती है.", flags)
        # Duration (9 days -> 7+ bucket) and severity are scored gates, so they flag without a danger-sign tag.
        self.assertIn("- FEVER fever_duration_bucket: how long the fever has lasted. Mother's words: 7 दिन से ज़्यादा.", flags)
        self.assertIn("- FEVER fever_severity: how severe the fever is, from mild warmth to being unable to stand or eat. "
                      "Mother's words: बहुत तेज़ बुखार है — खड़े नहीं हो पाते, या कुछ खा-पी नहीं पाते.", flags)
        self.assertIn("- FEVER associated_symptoms: which symptoms accompany the fever. "
                      "Mother's words: कंपकंपी और ठंड लगना; बदन दर्द.", flags)
        self.assertIn("हाँ, सीढ़ियाँ चढ़ने पर साँस फूलती है", html)
        self.assertFalse(proj.incomplete)
        self.assertNotIn("यह कॉल अधूरी रही", html)

    def test_fever_clean_fever_is_a_clean_call_on_every_surface(self):
        proj, html = self.report(_FEVER, "clean_fever")
        self.assertEqual(proj.red_flags, ())
        self.assertEqual(proj.yes_items, ())
        self.assertEqual(self.alerts(proj).strip().splitlines(), ["& Gate Alerts", "- No positive gate alerts."])
        self.assertEqual((proj.page2_unknown, proj.page2_unanswered), ((), ()))
        # Every gate was answered "नहीं", so page two prints them as negatives rather than dropping them.
        negatives = proj.doctor_markdown.split("### Pertinent Negatives", 1)[1]
        self.assertIn("Clear: difficulty breathing, or chest pain or tightness.", negatives)
        self.assertIn("Clear: jaundice, or bleeding from the nose, gums, or in the urine.", negatives)
        self.assertIn("इस कॉल में कोई बात हाँ में दर्ज नहीं हुई।", html)
        self.assertNotIn('<ul class="yes">', html)
        self.assertFalse(proj.incomplete)

    def test_fever_stopped_early_keeps_the_flags_it_got_and_the_questions_it_never_asked(self):
        proj, html = self.report(_FEVER, "stopped_early")
        self.assertEqual(proj.close_reason, "caller_stop")
        self.assertTrue(proj.incomplete)
        self.assertEqual(len(proj.red_flags), 3)
        self.assertIn("RED_FLAG breathing_or_chest: difficulty breathing, or chest pain or tightness", proj.red_flags[2])
        self.assertIn("हल्की साँस फूलती है, पर छाती में दर्द नहीं", proj.red_flags[2])
        self.assertIn("तेज़ बुखार है — बिस्तर पर रहना पड़ता है", proj.red_flags[0])
        # Stopped after the second warning gate: the remaining danger-sign and treatment questions are unasked,
        # never recorded as "no".
        groups = {group["title"]: list(group["items"]) for group in proj.page2_unasked}
        self.assertEqual(set(groups), {"खतरे की निशानी", "इलाज और संपर्क", "अंत"})
        self.assertEqual(groups["खतरे की निशानी"], ["बार-बार उल्टी हो रही है, या पतले दस्त आ रहे हैं"])
        self.assertIn("इस बुखार के लिए अब तक कोई दवा ली है, या डॉक्टर से दिखाया है", groups["इलाज और संपर्क"])
        negatives = proj.doctor_markdown.split("### Pertinent Negatives", 1)[1]
        self.assertIn("Call ended before every applicable question was asked.", negatives)
        self.assertNotIn("Clear: repeated vomiting or loose stools.", negatives)
        self.assertIn("यह कॉल अधूरी रही।", html)
        self.assertIn("अभी नहीं पूछा गया", html)
        self.assertIn("बार-बार उल्टी हो रही है, या पतले दस्त आ रहे हैं", html)
        self.assertNotIn("पूर्ण कॉल", html)


def _run_together(worker, threads: int) -> None:
    """Start `threads` copies of `worker(i)` that all leave the gate at the same instant."""
    gate, errors = threading.Barrier(threads), []

    def run(i: int) -> None:
        try:
            gate.wait()
            worker(i)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the test thread so a crash is a failure
            errors.append(exc)

    running = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
    for t in running: t.start()
    for t in running: t.join()
    if errors: raise errors[0]


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


class ConcurrentWriteTests(_TempDataDir):
    """Every read-modify-write helper must be one transaction, or a second caller silently overwrites the first.

    `append_call_event` and `append_transcript` read the JSON column, mutate it and write it back. With the lock
    released in between, two threads on the same call interleave: both read the same list, both write base+own,
    and everything the slower thread read is gone. On the clinic floor that is a lost transcript turn or a lost
    red-flag event on a call two people are watching at once.
    """

    THREADS, APPENDS = 4, 25

    def setUp(self):
        super().setUp()
        # Force frequent GIL handovers so the read/modify/write window is actually hit on every machine.
        self._previous_switch = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, self._previous_switch)

    def test_concurrent_append_call_event_keeps_every_event(self):
        call_id = db.create_call()
        _run_together(lambda i: [db.append_call_event(call_id, "step", f"{i}-{n}") for n in range(self.APPENDS)],
                      self.THREADS)
        events = db.get_call(call_id)["events"]
        self.assertEqual(len(events), self.THREADS * self.APPENDS)
        expected = {f"{i}-{n}" for i in range(self.THREADS) for n in range(self.APPENDS)}
        self.assertEqual({e["detail"] for e in events}, expected)

    def test_concurrent_append_transcript_keeps_every_turn(self):
        call_id = db.create_call()
        # A qid unique per append keeps the neighbour-merge rule out of the way, so every append must survive
        # as its own turn instead of being folded into the turn before it.
        _run_together(lambda i: [db.append_transcript(call_id, "user", f"बात {i}-{n}", qid=f"q{i}-{n}")
                                 for n in range(self.APPENDS)], self.THREADS)
        turns = db.get_call(call_id)["transcript"]
        self.assertEqual(len(turns), self.THREADS * self.APPENDS)
        self.assertEqual({t["text"] for t in turns},
                         {f"बात {i}-{n}" for i in range(self.THREADS) for n in range(self.APPENDS)})
        self.assertEqual({t["qid"] for t in turns},
                         {f"q{i}-{n}" for i in range(self.THREADS) for n in range(self.APPENDS)})

    def test_concurrent_telemetry_patches_all_land_in_the_same_object(self):
        call_id = db.create_call()
        db.update_call_telemetry(call_id, {"seeded": True})
        _run_together(lambda i: [db.update_call_telemetry(call_id, {f"k{i}_{n}": n}) for n in range(self.APPENDS)],
                      self.THREADS)
        telemetry = db.get_call(call_id)["telemetry"]
        self.assertTrue(telemetry.pop("seeded"))
        self.assertEqual(telemetry, {f"k{i}_{n}": n for i in range(self.THREADS) for n in range(self.APPENDS)})


class BilingualFactualReportTests(unittest.TestCase):
    """Patient summary PDFs must be strictly factual (zero advice/directives) and available in both hi and en."""

    BANNED_PHRASES = (
        "डॉक्टर से पूछें",
        "डॉक्टर या स्वास्थ्य कार्यकर्ता",
        "डॉक्टर से बात करें",
        "सलाह",
        # No advice on a routine sheet. The emergency band is the one deliberate exception: it names 108 and the
        # nearest hospital, because a call that closed on a danger sign must say so. Every scenario here is a
        # non-emergency close, so an emergency seed would have to assert its band separately (see EmergencyReportTests).
        "108",
        "दोबारा कॉल",
        "अस्पताल ले जाइए",
        "अस्पताल जाइए",
        "ask the doctor",
        "consult a doctor",
        "talk to the doctor",
        "thalassemia screening",
        "call again",
        "never blue light",
        "only if a child is",
        "only if a girl is",
        "item_1",
    )

    def test_all_reportable_questions_have_clean_stem_en(self):
        for pack_id in (_SHISHU, _FEVER):
            pack = load_pack(pack_id)
            for q in pack.questions:
                if not q.is_routable:
                    continue
                with self.subTest(pack=pack_id, qid=q.id):
                    stem_en = getattr(q, "stem_en", "")
                    self.assertTrue(stem_en.strip(), f"{pack_id}:{q.id} missing stem_en")
                    self.assertNotIn("never blue light", stem_en.lower())
                    self.assertNotIn("only if", stem_en.lower())

    def test_hindi_and_english_html_contain_zero_advice_or_prompt_leakage(self):
        for pack_id, scenarios in _SCENARIOS.items():
            pack = load_pack(pack_id)
            for sc in scenarios:
                state = _replay(pack, sc)
                proj = reports.project_summary(pack, state)
                self.assertEqual(proj.patient_questions, (), f"{pack_id}:{sc} still emits patient_questions")
                for lang in ("hi", "en"):
                    with self.subTest(pack=pack_id, scenario=sc, lang=lang):
                        html = reports.render_patient_html(pack, _SAMPLE_CALL, proj, lang=lang)
                        low = html.lower()
                        for banned in self.BANNED_PHRASES:
                            self.assertNotIn(banned.lower(), low, f"Found banned phrase {banned!r} in {pack_id}/{sc}/{lang}")
                        if lang == "en":
                            self.assertIn('lang="en"', html)
                            self.assertIn("Call Record", html)
                            self.assertIn("Details", html)
                        else:
                            self.assertIn('lang="hi"', html)
                            self.assertIn("विवरण", html)

    def test_build_and_ready_pdf_support_both_hi_and_en(self):
        pack = load_pack(_SHISHU)
        state, _ = reports.replay_seed_scenario(pack, _scenario(pack, "red_flags"))
        cid = db.create_call(pack_id=_SHISHU, caller_name="सुनीता")
        db.save_cursor_state(cid, pack, state)
        db.complete_call(cid, 300)
        hi_bytes, hi_name = reports.build_patient_pdf_sync(cid, lang="hi")
        en_bytes, en_name = reports.build_patient_pdf_sync(cid, lang="en")
        self.assertTrue(hi_bytes.startswith(b"%PDF"))
        self.assertTrue(en_bytes.startswith(b"%PDF"))
        self.assertTrue(hi_name.endswith("-hi.pdf"), hi_name)
        self.assertTrue(en_name.endswith("-en.pdf"), en_name)
        self.assertNotEqual(hi_bytes, en_bytes)
        self.assertIsNotNone(reports.ready_pdf(cid, lang="hi"))
        self.assertIsNotNone(reports.ready_pdf(cid, lang="en"))


if __name__ == "__main__":
    unittest.main()
