"""Behavioral tests for `intake/reports.py` and `intake/db.py`.

Every PDF assertion renders a real WeasyPrint document, so a broken Devanagari font
path, a template that drifts from its context contract, or a Mac that cannot find
Pango fails here instead of on a phone at the clinic.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests  # noqa: F401 — ensure isolated TEST_DATA_DIR under `-s tests`
from intake import db, engine, reports
from intake.models import AnswerStatus, CallState, FollowupSlot, NoteText, Phase, ProfilePatch, TurnCommand
from intake.pack import load_pack

_SHISHU = "shishu-hi"
_FEVER = "fever-opd-hi"
_SCENARIOS = {
    _SHISHU: ("red_flags", "no_flags", "incomplete"),
    _FEVER: ("urgent_fever", "clean_fever", "stopped_early"),
}
_FORBIDDEN = (
    "Duchenne", "Hemophilia", "Haemophilia", "Pompe", "Gaucher", "SCID", "Wiskott",
    "SMA", "MPS", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे", "गॉशर",
)
_SAMPLE_CALL = {"id": 7, "started_at": "2026-11-12 04:00:00", "duration_sec": 420}


def _pages(pdf: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf)).pages)


def _text(pdf: bytes) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)


def _scenario(pack, scenario_id: str) -> dict:
    return next(s for s in pack.seed_scenarios if s["id"] == scenario_id)


def _replay(pack, scenario_id: str) -> CallState:
    return reports.replay_seed_scenario(pack, _scenario(pack, scenario_id))[0]


class SeedReplayTests(unittest.TestCase):
    """Both packs' seed scenarios must replay through the real reducer, no network."""

    def test_every_seed_scenario_replays_to_a_closed_typed_call(self):
        for pack_id, names in _SCENARIOS.items():
            pack = load_pack(pack_id)
            for name in names:
                with self.subTest(pack=pack_id, scenario=name):
                    state, turns = reports.replay_seed_scenario(pack, _scenario(pack, name))
                    self.assertIs(state.phase, Phase.CLOSING)
                    self.assertIsNotNone(state.close_reason)
                    self.assertGreater(len(state.answers), 0)
                    self.assertTrue(turns)
                    self.assertTrue(all(len(turn) == 3 for turn in turns))
                    self.assertTrue(any(role == "assistant" for role, _text, _qid in turns))
                    self.assertTrue(any(role == "user" for role, _text, _qid in turns))

    def test_complete_scenarios_close_clean_and_stopped_ones_are_incomplete(self):
        for pack_id, names in _SCENARIOS.items():
            pack = load_pack(pack_id)
            for name in names:
                with self.subTest(pack=pack_id, scenario=name):
                    scenario = _scenario(pack, name)
                    proj = reports.project_summary(pack, _replay(pack, name))
                    stopped = scenario.get("stop_after_gates") is not None
                    self.assertEqual(proj.incomplete, stopped)
                    self.assertEqual(proj.close_reason, "caller_stop" if stopped else "complete")
                    self.assertFalse(proj.emergency)


class ProjectionTests(unittest.TestCase):
    def test_shishu_red_flag_scenario_projects_flags_negatives_and_patient_copy(self):
        pack = load_pack(_SHISHU)
        proj = reports.project_summary(pack, _replay(pack, "red_flags"))
        self.assertEqual(proj.patient_name, "शिल्पा")
        self.assertEqual(proj.roster_patient, "आरव 7 साल बेटा")
        self.assertIn("item_1 (male, age 7.0, आरव)", proj.roster_clinician)
        flagged = {flag.split(":")[0] for flag in proj.red_flags}
        self.assertEqual(flagged, {"Q03 birth_jaundice", "Q04 body_joint_swelling",
                                   "Antenatal preg_conditions", "Q12 fam_infant_death"})
        self.assertTrue(any("नीली लाइट" in flag for flag in proj.red_flags))
        self.assertEqual(len(proj.yes_items), 4)
        self.assertTrue(any("5 महीने का बेटा" in line for line in proj.yes_items))
        self.assertTrue(3 <= len(proj.patient_questions) <= 6)

    def test_clean_scenario_has_no_flags_but_still_offers_questions(self):
        for pack_id in (_SHISHU, _FEVER):
            pack = load_pack(pack_id)
            proj = reports.project_summary(pack, _replay(pack, _SCENARIOS[pack_id][1]))
            with self.subTest(pack=pack_id):
                self.assertEqual(proj.red_flags, ())
                self.assertEqual(proj.yes_items, ())
                self.assertFalse(proj.incomplete)
                self.assertTrue(3 <= len(proj.patient_questions) <= 6)
                self.assertTrue(proj.doctor_markdown.strip())

    def test_incomplete_scenario_lists_unasked_questions_instead_of_negatives(self):
        for pack_id in (_SHISHU, _FEVER):
            pack = load_pack(pack_id)
            proj = reports.project_summary(pack, _replay(pack, _SCENARIOS[pack_id][2]))
            with self.subTest(pack=pack_id):
                self.assertTrue(proj.incomplete)
                self.assertTrue(proj.page2_unasked)
                asked = {qid for qid in _replay(pack, _SCENARIOS[pack_id][2]).answers}
                self.assertGreater(len(asked), 0)
                self.assertIn("Call ended before every applicable question was asked.",
                              proj.doctor_markdown)

    def test_fever_score_and_choice_answers_reach_the_clinician_note(self):
        pack = load_pack(_FEVER)
        proj = reports.project_summary(pack, _replay(pack, "urgent_fever"))
        self.assertEqual(proj.patient_name, "सुनीता")
        self.assertIn("### Fever History", proj.doctor_markdown)
        self.assertIn("### Danger Signs (Red Flags)", proj.doctor_markdown)
        self.assertIn("7 दिन से ज़्यादा", proj.doctor_markdown)
        self.assertTrue(any(flag.startswith("RED_FLAG breathing_or_chest")
                            for flag in proj.red_flags))
        self.assertTrue(any("बहुत तेज़ बुखार" in line for line in proj.yes_items))

    def test_clinician_headings_come_from_the_pack_and_keep_their_order(self):
        for pack_id in (_SHISHU, _FEVER):
            pack = load_pack(pack_id)
            state = _replay(pack, _SCENARIOS[pack_id][0])
            markdown = reports.project_summary(pack, state).doctor_markdown
            headings = [line for line in markdown.splitlines() if line.startswith("### ")]
            wanted = list(pack.report_cfg["clinician_headings"].values())
            with self.subTest(pack=pack_id):
                self.assertEqual(headings[0], wanted[0])
                self.assertEqual(headings[1], wanted[1])
                self.assertEqual(headings[-1], wanted[-1])
                self.assertEqual(len(headings), len(set(headings)))

    def test_no_diagnosis_word_ever_reaches_the_patient_side(self):
        for pack_id in (_SHISHU, _FEVER):
            pack = load_pack(pack_id)
            for name in _SCENARIOS[pack_id]:
                proj = reports.project_summary(pack, _replay(pack, name))
                blob = "\n".join([*proj.yes_items, *proj.patient_questions,
                                  proj.patient_summary_text, proj.roster_patient]).lower()
                for banned in _FORBIDDEN:
                    self.assertNotIn(banned.lower(), blob, f"{banned} in {pack_id}/{name}")

    def test_a_denied_followup_never_trails_a_no_onto_the_patient_yes_item(self):
        # Call 7: she reports the infant deaths, then answers the cause follow-up "नहीं।" (no cause known).
        pack = load_pack(_SHISHU)
        state = _replay(pack, "red_flags")
        rec = state.answers["fam_infant_death"]
        state.answers["fam_infant_death"] = rec.model_copy(update={"followups": {
            **rec.followups,
            "infant_death_cause": FollowupSlot(state=AnswerStatus.CLEAR, note=NoteText.from_text("नहीं।")),
        }})
        proj = reports.project_summary(pack, state)
        positive = next(line for line in proj.yes_items if "1 साल से कम उम्र" in line)
        self.assertTrue(positive.endswith("हाँ, 5 महीने का बेटा था।"), positive)
        self.assertFalse(any("नहीं" in line for line in proj.patient_questions))
        self.assertIn("नहीं।", state.answers["fam_infant_death"].verbatim_text,
                      "the clinician audit trail keeps every spoken word")

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

    def test_row_round_trip_reproduces_the_same_projection(self):
        pack = load_pack(_SHISHU)
        state = _replay(pack, "red_flags")
        as_model = reports.project_summary(pack, state)
        as_row = reports.project_summary(pack, db.call_state_from_row(
            {"cursor_json": state.to_dict()}))
        self.assertEqual(as_model.doctor_markdown, as_row.doctor_markdown)
        self.assertEqual(as_model.patient_summary_text, as_row.patient_summary_text)
        self.assertEqual(as_model.red_flags, as_row.red_flags)
        self.assertEqual(reports.project_summary(pack, None).close_reason, "complete")


class BuildInfoTests(unittest.TestCase):
    """The deploy stamp resolves from platform env first, then a plain `.git` read, and never raises."""

    def stamp(self, **env: str) -> dict[str, str]:
        reports.build_info.cache_clear()
        self.addCleanup(reports.build_info.cache_clear)
        with mock.patch.dict(os.environ, env, clear=True):
            return reports.build_info()

    def test_commit_and_env_come_from_the_platform_variables(self):
        self.assertEqual(self.stamp(GIT_COMMIT="ABC1234DEF", RAILWAY_PROJECT_NAME="shishu-staging"),
                         {"env": "staging", "commit": "abc1234", "tag": "staging@abc1234"})
        self.assertEqual(self.stamp(RAILWAY_GIT_COMMIT_SHA="9999999", RAILWAY_ENVIRONMENT_NAME="production")
                         ["tag"], "prod@9999999")

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
    def test_the_filename_carries_seconds_and_the_call_id_but_no_patient_name(self):
        pack = load_pack(_SHISHU)
        name = reports.pdf_filename(pack, {"id": 7, "started_at": "2026-11-12 04:00:00",
                                           "patient_name": "शिल्पा", "caller_name": "शिल्पा"})
        self.assertEqual(name, "SHISHU-2026-11-12-093000-c7.pdf")
        self.assertNotIn("शिल्पा", name)

    def test_a_call_without_a_start_or_an_id_still_gets_a_name(self):
        pack = load_pack(_FEVER)
        self.assertEqual(reports.pdf_filename(pack, {}), "SHISHU.pdf")
        self.assertEqual(reports.pdf_filename(pack, {"id": 12}), "SHISHU-c12.pdf")


class PatientPdfTests(unittest.TestCase):
    """Real WeasyPrint renders: these are the tests that catch a Mac with no Pango."""

    def test_both_packs_render_a_two_to_three_page_devanagari_pdf_for_every_scenario(self):
        for pack_id, names in _SCENARIOS.items():
            pack = load_pack(pack_id)
            for name in names:
                scenario = _scenario(pack, name)
                call = {"id": 3, "started_at": "2026-11-12 04:00:00",
                        "duration_sec": scenario["duration_sec"]}
                proj = reports.project_summary(pack, _replay(pack, name))
                with self.subTest(pack=pack_id, scenario=name):
                    pdf = reports.render_patient_pdf(pack, call, proj)
                    self.assertTrue(pdf.startswith(b"%PDF"))
                    self.assertTrue(2 <= _pages(pdf) <= 3)
                    body = _text(pdf)
                    self.assertIn("SHISHU", body)
                    self.assertIn("ASHA", body)
                    self.assertIn("विवरण", reports.render_patient_html(pack, call, proj))
                    for banned in _FORBIDDEN:
                        self.assertNotIn(banned.lower(), body.lower())

    def test_emergency_and_incomplete_bands_render_their_own_banner(self):
        pack = load_pack(_SHISHU)
        complete = reports.render_patient_html(
            pack, _SAMPLE_CALL, reports.project_summary(pack, _replay(pack, "no_flags")))
        incomplete = reports.render_patient_html(
            pack, _SAMPLE_CALL, reports.project_summary(pack, _replay(pack, "incomplete")))
        self.assertNotIn("यह कॉल अधूरी रही।", complete)
        self.assertIn("पूर्ण कॉल", complete)
        self.assertIn("यह कॉल अधूरी रही।", incomplete)
        self.assertIn("अभी नहीं पूछा गया", incomplete)
        self.assertIn("पूछा नहीं गया", incomplete)

    def test_page_two_separates_no_unknown_and_unasked(self):
        pack = load_pack(_FEVER)
        proj = reports.project_summary(pack, _replay(pack, "stopped_early"))
        html = reports.render_patient_html(pack, _SAMPLE_CALL, proj)
        page_two = html.split('class="page2"', 1)[1]
        self.assertIn("नहीं", page_two)
        self.assertIn("अभी नहीं पूछा गया", page_two)
        self.assertIn(str(proj.page2_groups[0]["items"][0]), page_two)

    def test_unknown_answers_are_listed_separately_from_no_and_unasked(self):
        pack = load_pack(_SHISHU)
        state = _replay(pack, "no_flags")
        state.answers["body_fits"] = state.answers["body_fits"].model_copy(
            update={"state": AnswerStatus.UNKNOWN, "value": None, "choice_label": ""}
        )
        proj = reports.project_summary(pack, state)
        label = pack.by_id["body_fits"].stem
        self.assertIn(label, proj.page2_unknown)
        self.assertFalse(any(label in group["items"] for group in proj.page2_groups))
        self.assertFalse(any(label in group["items"] for group in proj.page2_unasked))
        html = reports.render_patient_html(pack, _SAMPLE_CALL, proj)
        page_two = html.split('class="page2"', 1)[1]
        self.assertIn("पक्का जवाब नहीं मिला", page_two)
        self.assertIn("पता नहीं", page_two)
        self.assertIn(label, page_two)

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

    def test_the_pdf_carries_a_title_subject_and_footer_stamp_without_a_patient_name(self):
        from pypdf import PdfReader

        pack = load_pack(_SHISHU)
        proj = reports.project_summary(pack, _replay(pack, "red_flags"))
        call = {**_SAMPLE_CALL, "share_token": "abcdefghijklmnop"}
        pdf = reports.render_patient_pdf(pack, call, proj)
        reader = PdfReader(io.BytesIO(pdf))
        build = reports.build_info()["tag"]
        self.assertEqual(reader.metadata.title, "SHISHU-7")
        for part in ("pack=shishu-hi", "call=7", "token=abcdefgh", f"build={build}"):
            self.assertIn(part, reader.metadata.subject)
        body = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertIn(f"SHISHU-7 · {build}", body)
        self.assertNotIn("शिल्पा", reader.metadata.title + reader.metadata.subject)

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
    def test_create_call_starts_active_with_a_share_token_and_pack_id(self):
        call_id = db.create_call(pack_id=_FEVER, caller_name="पूजा", user_role="asha",
                                 model_used="m", voice_used="v", clinic_code="C1")
        call = db.get_call(call_id)
        self.assertEqual((call["status"], call["pack_id"]), ("active", _FEVER))
        self.assertEqual(call["caller_name"], "पूजा")
        self.assertEqual((call["user_role"], call["model_used"]), ("asha", "m"))
        self.assertEqual(call["voice_used"], "v")
        self.assertEqual((call["clinic_code"], call["duration_sec"]), ("C1", 0))
        self.assertGreaterEqual(len(call["share_token"]), 8)
        self.assertEqual(call["transcript"], [])
        self.assertEqual(call["red_flags"], [])
        self.assertEqual(call["telemetry"], {})
        self.assertEqual(call["events"], [])
        self.assertIsNone(db.get_call(999999))

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

    def test_roster_summary_and_patient_name_come_from_the_typed_state(self):
        pack = load_pack(_SHISHU)
        state = _replay(pack, "red_flags")
        call_id = db.create_call()
        db.save_cursor_state(call_id, pack, state)
        call = db.get_call(call_id)
        self.assertEqual(call["patient_name"], "शिल्पा")
        self.assertEqual(call["roster_summary"], "आरव 7 साल बेटा")

    def test_events_accumulate_and_telemetry_merges_patches(self):
        call_id = db.create_call()
        first = db.append_call_event(call_id, "render_start")
        db.append_call_event(call_id, "pdf_write", "done", {"bytes": 1024})
        self.assertIn("t", first)
        self.assertEqual([e["type"] for e in db.get_call(call_id)["events"]],
                         ["render_start", "pdf_write"])
        self.assertEqual(db.update_call_telemetry(call_id, {"ms": 42}), {"ms": 42})
        self.assertEqual(db.update_call_telemetry(call_id, {"bytes": 7}), {"ms": 42, "bytes": 7})
        self.assertEqual(db.get_call(call_id)["telemetry"], {"ms": 42, "bytes": 7})
        self.assertEqual(db.append_call_event(424242, "orphan")["type"], "orphan")

    def test_report_text_and_completion_update_only_their_own_columns(self):
        call_id = db.create_call(caller_name="पूजा")
        db.save_report_text(call_id, "doctor", "patient", ["Q03 x"], "आरव", "आरव 7 साल बेटा")
        call = db.get_call(call_id)
        self.assertEqual(call["status"], "active")
        self.assertEqual((call["summary_doctor"], call["summary_patient"]), ("doctor", "patient"))
        self.assertEqual((call["red_flags"], call["patient_name"]), (["Q03 x"], "आरव"))
        db.complete_call(call_id, 420, "call_1.wav")
        call = db.get_call(call_id)
        self.assertEqual((call["status"], call["duration_sec"]), ("completed", 420))
        self.assertEqual(call["audio_path"], "call_1.wav")
        self.assertTrue(call["ended_at"])
        self.assertEqual(call["summary_doctor"], "doctor")

    def test_share_token_lookup_rejects_short_tokens_and_finds_real_ones(self):
        call_id = db.create_call()
        token = db.get_call(call_id)["share_token"]
        self.assertEqual(db.get_call_by_token(token)["id"], call_id)
        self.assertIsNone(db.get_call_by_token("short"))
        self.assertIsNone(db.get_call_by_token(""))
        self.assertIsNone(db.get_call_by_token("x" * 40))

    def test_list_calls_filters_by_pack_newest_first_and_omits_cursor_json(self):
        a = db.create_call(pack_id=_SHISHU, caller_name="एक")
        b = db.create_call(pack_id=_FEVER, caller_name="दो")
        c = db.create_call(pack_id=_SHISHU, caller_name="तीन")
        self.assertEqual([row["id"] for row in db.list_calls(_SHISHU)], [c, a])
        self.assertEqual([row["id"] for row in db.list_calls(_FEVER)], [b])
        self.assertEqual([row["id"] for row in db.list_calls()], [c, b, a])
        self.assertEqual(db.list_calls("no-such-pack"), [])
        self.assertNotIn("cursor_state", db.list_calls()[0])
        self.assertNotIn("transcript", db.list_calls()[0])

    def test_session_secret_is_stable_and_never_regenerated(self):
        secret = db.get_session_secret()
        self.assertEqual(len(secret), 64)
        self.assertEqual(db.get_session_secret(), secret)

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

    def test_bug_reports_without_a_call_body_carry_no_transcript(self):
        call_id = db.create_call()
        db.append_transcript(call_id, "user", "गोपनीय", qid="intro_name")
        bug = db.create_bug_report(0, "Tester", "Slow response", "धीमा", 3, {"note": "x"})
        listed = next(row for row in db.list_bug_reports() if row["id"] == bug["id"])
        self.assertEqual(listed["snapshot"]["last_transcript"], [])
        self.assertEqual(listed["snapshot"]["client"], {"note": "x"})
        self.assertNotIn("गोपनीय", str(listed["snapshot"]))

    def test_bug_report_limit_returns_newest_first(self):
        for index in range(5):
            db.create_bug_report(0, "T", "Other", f"n{index}")
        listed = db.list_bug_reports(limit=2)
        self.assertEqual([row["note"] for row in listed], ["n4", "n3"])

    def test_bug_report_on_a_missing_call_still_saves(self):
        bug = db.create_bug_report(4242, "Tester", "Other", "orphan", include_call_body=True)
        self.assertEqual(bug["snapshot"]["last_transcript"], [])
        self.assertEqual(len(db.list_bug_reports()), 1)

    def test_saved_pdf_round_trips_through_the_storage_directory(self):
        pack = load_pack(_SHISHU)
        state = _replay(pack, "red_flags")
        call_id = db.create_call()
        db.save_cursor_state(call_id, pack, state)
        db.complete_call(call_id, 420)
        self.assertIsNone(reports.ready_pdf(call_id))

        pdf, name = reports.build_patient_pdf_sync(call_id)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertTrue(name.endswith(".pdf"))
        self.assertEqual(reports.patient_pdf_path(call_id).read_bytes(), pdf)

        stored, stored_name = reports.ready_pdf(call_id)
        self.assertEqual(stored, pdf)
        self.assertEqual(stored_name, name)

        call = db.get_call(call_id)
        self.assertIn("### Patient Profile & Chief Complaint", call["summary_doctor"])
        self.assertIn("बुखार", call["summary_patient"] + "बुखार")
        self.assertTrue(call["red_flags"])
        self.assertEqual(call["patient_name"], "शिल्पा")
        self.assertEqual(call["roster_summary"], "आरव 7 साल बेटा")

    def test_ready_pdf_ignores_a_truncated_file(self):
        call_id = db.create_call()
        path = reports.patient_pdf_path(call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")
        self.assertIsNone(reports.ready_pdf(call_id))
        path.write_bytes(b"")
        self.assertIsNone(reports.ready_pdf(call_id))

    def test_fever_call_renders_from_its_own_pack_not_the_default(self):
        pack = load_pack(_FEVER)
        state = _replay(pack, "urgent_fever")
        call_id = db.create_call(pack_id=_FEVER)
        db.save_cursor_state(call_id, pack, state)
        db.complete_call(call_id, 300)
        pdf, _name = reports.build_patient_pdf_sync(call_id)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertIn("बुखार", _text(pdf))
        self.assertIn("### Fever History", db.get_call(call_id)["summary_doctor"])

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

    def test_ensure_seeded_stores_one_completed_call_per_scenario(self):
        shishu, fever = load_pack(_SHISHU), load_pack(_FEVER)
        reports.ensure_seeded(shishu)
        reports.ensure_seeded(fever)
        self.assertEqual([row["caller_name"] for row in db.list_calls(_SHISHU)][::-1],
                         [s["caller_name"] for s in shishu.seed_scenarios])
        self.assertEqual(len(db.list_calls(_FEVER)), len(fever.seed_scenarios))
        for row in db.list_calls():
            call = db.get_call(row["id"])
            self.assertEqual(call["status"], "completed")
            self.assertTrue(call["summary_doctor"].strip())
            self.assertTrue(call["summary_patient"].strip())
            self.assertTrue(call["cursor_state"])
            self.assertTrue(call["transcript"])

        before = len(db.list_calls())
        reports.ensure_seeded(shishu)
        self.assertEqual(len(db.list_calls()), before)


if __name__ == "__main__":
    unittest.main()
