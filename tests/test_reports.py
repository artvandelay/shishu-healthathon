"""Stage 2 reports: deterministic cursor PDF, disease-name scrub, and token privacy."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import tests  # noqa: E402, F401

_TMP = tempfile.mkdtemp(prefix="shishu-v3-reports-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")

import app as shishu_app  # noqa: E402
import call_cursor  # noqa: E402
import db  # noqa: E402
import mother_report  # noqa: E402
import post_call  # noqa: E402
import report_service  # noqa: E402
import test_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_profile, apply_turn, close_technical_failure  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from triage_catalog import active_specs, get_spec  # noqa: E402

_CHILD_QUESTIONS = tuple(
    spec.question_hi for spec in active_specs() if "no_children" in spec.skip_if
)
_FORBIDDEN = ("Duchenne", "Hemophilia", "Haemophilia", "Pompe", "Gaucher", "SCID", "Wiskott")


def _childless_cursor():
    state = cursor_tests._profile(name="पूजा", children=[])
    state = apply_turn(state, "clear")
    state, hold = apply_profile(state, ever_pregnant=False)
    assert hold is None, hold
    guard = 0
    while state["phase"] == "asking" and guard < 90:
        state = apply_turn(state, "clear")
        guard += 1
    return state


def _bleed_cursor():
    kids = [cursor_tests._child(7, "male", 1), cursor_tests._child(2, "female", 2)]
    state = cursor_tests._profile(name="रेखा", children=kids)
    concerns = {
        "body_bleeding_bruise": "बड़े वाले को चोट पर खून देर तक बहता है",
        "body_joint_swelling": "गिरने पर घुटना सूजता है और कुछ दिनों में ठीक हो जाता है",
        "fam_infant_death": "मामा का बेटा छह महीने में गुजर गया",
    }
    guard = 0
    while state["phase"] == "asking" and guard < 90:
        guard += 1
        qid = state["active_qid"]
        if state["pending_followup_idx"] == -1:
            state = apply_turn(state, "concern", child_ref="child_1")
            continue
        if isinstance(state["pending_followup_idx"], int) and state["pending_followup_idx"] >= 0:
            state = apply_turn(state, "clear")
            continue
        if qid in concerns:
            state = apply_turn(state, "concern", note=concerns[qid], child_ref="child_1")
            continue
        state = apply_turn(state, "clear")
    return state


def _call_for(state, call_id, started="2026-11-12 04:00:00"):
    return {
        "id": call_id,
        "started_at": started,
        "duration_sec": 180,
        "cursor_state": state,
        "cursor_json": json.dumps(state, ensure_ascii=False),
        "transcript": _transcript(state),
    }


def _transcript(state):
    turns = []
    for spec, entry, status in post_call._rows(state):
        turns.append({"role": "assistant", "qid": spec.id, "text": spec.question_hi})
        if status == "concern":
            said = entry.get("notes") or "हाँ"
            turns.append({"role": "user", "qid": spec.id, "text": f"हाँ। {said}"})
        elif status == "unknown":
            turns.append({"role": "user", "qid": spec.id, "text": "पता नहीं"})
        else:
            turns.append({"role": "user", "qid": spec.id, "text": "नहीं"})
    return turns


def _pages(pdf: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf)).pages)


def _page2(call: dict) -> str:
    synthesis = post_call.build_reports(call)
    document = mother_report.render_html(mother_report.build_v3(call, synthesis))
    return document.split('class="page2"', 1)[1]


class CursorReportTests(unittest.TestCase):
    def test_childless_cursor_omits_child_questions_and_renders_two_pages(self):
        state = _childless_cursor()
        self.assertEqual(state["close_reason"], "complete")
        call = _call_for(state, 7)
        synthesis = post_call.build_reports(call)
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])
        self.assertGreaterEqual(len(synthesis["mother_questions_hi"]), 3)
        self.assertLessEqual(len(synthesis["mother_questions_hi"]), 6)
        self.assertNotIn("मेरे बच्चे", " ".join(synthesis["mother_questions_hi"]))
        self.assertEqual(synthesis["yes_summaries_hi"], [])
        self.assertEqual(synthesis["children_label_hi"], "")
        report = mother_report.build_v3(call, synthesis)
        document = mother_report.render_html(report)
        page2 = document.split('class="page2"', 1)[1]
        for question in _CHILD_QUESTIONS:
            self.assertNotIn(question, page2, question[:24])
        self.assertIn(get_spec("fam_related").question_hi, page2)
        self.assertIn("SHISHU-7", document)
        self.assertIn(mother_report.THALASSEMIA_LINE, document.split('class="page2"', 1)[0])
        self.assertIn("यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है।", document)
        pdf = mother_report.render_pdf(report)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreaterEqual(_pages(pdf), 2)
        extracted = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)
        self.assertNotIn("काँच के बक्से", extracted)
        self.assertIn("SHISHU-7", extracted)

    def test_scrubber_strips_disease_names_from_mother_hindi(self):
        dirty_questions = [
            "क्या मेरे बच्चे को Duchenne है?",
            "Hemophilia की जाँच करानी चाहिए?",
        ]
        fitted = post_call._fit_questions(dirty_questions)
        scrubbed_yes = post_call._scrub_diagnostic_terms("डॉक्टर ने Hemophilia और Pompe बताया।")
        blob = " ".join([*fitted, scrubbed_yes])
        for name in _FORBIDDEN:
            self.assertNotIn(name, blob)
            self.assertNotIn(name.lower(), blob.lower())
        self.assertGreaterEqual(len(fitted), 3)
        self.assertLessEqual(len(fitted), 6)

        call = _call_for(_bleed_cursor(), 3)
        synthesis = post_call.build_reports(call)
        mother = " ".join(synthesis["mother_questions_hi"] + synthesis["yes_summaries_hi"])
        for name in _FORBIDDEN:
            self.assertNotIn(name.lower(), mother.lower())
        self.assertIn("खून", mother)
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])

    def test_concerns_unknowns_negatives_and_closing_disclosure_reach_every_report(self):
        note = "पति को गुप्त रोग; बच्चे पति के नहीं हैं"
        qids = [
            "birth_nicu",
            "body_bleeding_bruise",
            "body_transfusion",
            "fam_fits_delay",
            "fam_cancer",
            "fam_heart_kidney",
        ]
        notes = {
            "birth_nicu": "तीन दिन काँच का बक्सा",
            "body_bleeding_bruise": "चोट पर खून देर तक बहता है",
            "body_transfusion": "दो बार खून चढ़ा",
            "fam_fits_delay": "भाभी के बेटे को दौरे",
            "fam_cancer": "मामा को कैंसर",
            "fam_heart_kidney": "दादा को दिल की बीमारी",
        }
        state = cursor_tests._profile(name="मीना", children=[cursor_tests._child(4, "male")])
        for qid in qids:
            state["answers"][qid] = {
                "state": "concern",
                "child_ids": ["child_1"] if not qid.startswith("fam_") else [],
                "notes": notes[qid],
                "followups": {},
            }
        state["answers"]["birth_jaundice"] = {
            "state": "clear",
            "child_ids": [],
            "notes": "",
            "followups": {},
        }
        state["answers"]["ill_fasting"] = {
            "state": "unknown",
            "child_ids": [],
            "notes": "",
            "followups": {},
        }
        state["answers"]["close_anything_else"] = {
            "state": "concern",
            "child_ids": [],
            "notes": note,
            "followups": {"open_concern_detail": {"state": "concern", "note": note}},
        }
        state["phase"] = "closing"
        state["close_reason"] = "complete"
        call = _call_for(state, 9)
        synthesis = post_call.build_reports(call)
        self.assertEqual(
            list(synthesis.keys()),
            [
                "mother_name_hi",
                "children_label_hi",
                "mother_questions_hi",
                "yes_summaries_hi",
                "doctor_markdown",
                "red_flags",
            ],
        )
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])
        self.assertEqual(len(synthesis["yes_summaries_hi"]), 7)
        for qid in qids:
            self.assertTrue(any(notes[qid] in line for line in synthesis["yes_summaries_hi"]), qid)
            self.assertIn(qid, synthesis["doctor_markdown"])
            self.assertTrue(any(qid in flag for flag in synthesis["red_flags"]), qid)
        self.assertIn(note, synthesis["yes_summaries_hi"][-1])
        self.assertEqual(len(synthesis["mother_questions_hi"]), 6)

        profile = synthesis["doctor_markdown"].split("### Positive", 1)[0]
        self.assertIn(note, profile)
        positive = synthesis["doctor_markdown"].split("### Positive", 1)[1].split("\n### ", 1)[0]
        self.assertIn("close_anything_else", positive)
        self.assertIn(note, positive)
        self.assertIn("Unknown. " + get_spec("ill_fasting").need_en, synthesis["doctor_markdown"])
        negatives = synthesis["doctor_markdown"].split("### Pertinent Negatives", 1)[1]
        self.assertIn("Clear: " + get_spec("birth_jaundice").need_en, negatives)
        self.assertIn(note, "\n".join(synthesis["red_flags"]))
        self.assertTrue(any("close_anything_else" in flag for flag in synthesis["red_flags"]))

        document = mother_report.render_html(mother_report.build_v3(call, synthesis))
        page1, page2 = document.split('class="page2"', 1)
        for qid in qids:
            self.assertIn(notes[qid], page1)
        self.assertIn(note, page1)
        self.assertIn("पता नहीं", page2)
        self.assertIn(get_spec("ill_fasting").question_hi, page2)
        self.assertIn(get_spec("birth_jaundice").question_hi, page2)

    def test_devanagari_disease_names_and_followups_reach_reports(self):
        scrubbed = post_call._scrub_diagnostic_terms("बच्चे को ड्यूशेन और हीमोफीलिया बताया")
        self.assertNotIn("ड्यूशेन", scrubbed)
        self.assertNotIn("हीमोफीलिया", scrubbed)
        self.assertIn("बताया", scrubbed)
        self.assertNotIn("पॉम्पे", post_call._scrub_diagnostic_terms("पॉम्पे"))
        self.assertNotIn("थैलेसीमिया मेजर", post_call._scrub_diagnostic_terms("थैलेसीमिया मेजर की बात"))
        self.assertIn("थैलेसीमिया", post_call._scrub_diagnostic_terms("थैलेसीमिया जाँच में माइनर"))
        state = _bleed_cursor()
        state["answers"]["gate_bones_fractures"] = {
            "state": "concern",
            "child_ids": ["child_1"],
            "notes": "गिरने पर हड्डी टूटी",
            "followups": {"fracture_count_or_bowing": {"state": "concern", "note": "दो बार"}},
        }
        state["answers"]["birth_nicu"] = {
            "state": "concern",
            "child_ids": ["child_1"],
            "notes": "NICU",
            "followups": {
                "nicu_reason": {"state": "concern", "note": "सांस"},
                "nicu_days": {"state": "concern", "note": "तीन दिन"},
            },
        }
        sentence = post_call._concern_sentence(
            get_spec("gate_bones_fractures"),
            state["answers"]["gate_bones_fractures"],
        )
        self.assertIn("दो बार", sentence)
        synthesis = post_call.build_reports(_call_for(state, 19))
        for expected in ("दो बार", "सांस", "तीन दिन"):
            self.assertIn(expected, "\n".join(synthesis["yes_summaries_hi"]))
            self.assertIn(expected, synthesis["doctor_markdown"])
            self.assertIn(expected, "\n".join(synthesis["red_flags"]))
        self.assertNotIn("fracture_count_or_bowing", synthesis["doctor_markdown"])
        self.assertNotIn("nicu_reason", synthesis["doctor_markdown"])

    def test_v3_build_saves_flags_and_admin_labels(self):
        bleed_id = _store(_bleed_cursor(), "Pooja")
        calm_id = _store(_childless_cursor(), "Pooja")
        asyncio.run(report_service.mother_pdf(bleed_id))
        asyncio.run(report_service.mother_pdf(calm_id))
        bleed = db.get_call(bleed_id)
        calm = db.get_call(calm_id)
        self.assertTrue(bleed["red_flags"])
        self.assertTrue(bleed["summary_doctor"].strip())
        self.assertTrue(bleed["summary_mother"].strip())
        self.assertTrue(bleed["mother_name"])
        self.assertIn("साल", bleed["child_age"])
        self.assertEqual(calm["red_flags"], [])
        self.assertTrue(calm["mother_name"])
        self.assertNotEqual(calm["mother_name"], "Mother not named")

    def test_emergency_keeps_the_danger_and_drops_routine_filler(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(3, "female")]),
            "body_bleeding_bruise",
        )
        state = apply_turn(state, "concern", note="चोट पर खून देर तक बहता है")
        answers_before = json.loads(json.dumps(state["answers"]))
        state = apply_turn(state, "emergency", note="  अभी   बेहोश है  ")
        self.assertEqual(state["emergency_note"], "अभी बेहोश है")
        self.assertEqual(state["answers"], answers_before)
        call = _call_for(state, 21)
        synthesis = post_call.build_reports(call)
        flags = "\n".join(synthesis["red_flags"])
        self.assertEqual(flags.count("emergency: active danger sign now"), 1)
        self.assertIn("emergency: active danger sign now; hospital or 108. अभी बेहोश है", flags)
        positive = synthesis["doctor_markdown"].split(post_call._POSITIVE_HEADING, 1)[1].split("\n### ", 1)[0]
        self.assertIn(
            "- Active emergency. Caller was told to go to the nearest hospital or call 108. Mother's words: अभी बेहोश है.",
            positive,
        )
        self.assertEqual(positive.count("Active emergency."), 1)
        self.assertNotIn("No positive gate alerts.", positive)
        for routine in post_call._ROUTINE_HI:
            self.assertNotIn(routine, synthesis["mother_questions_hi"])
        self.assertIn("खून", synthesis["mother_questions_hi"][0])
        self.assertEqual(synthesis["mother_questions_hi"][1:], list(post_call._EMERGENCY_HI[:2]))
        doc = mother_report.render_html(mother_report.build_v3(call, synthesis))
        self.assertIn("108", doc)
        self.assertIn("नज़दीकी अस्पताल", doc)

        calm = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(3, "female")]),
            "ill_fasting",
        )
        calm = apply_turn(calm, "emergency", note="सांस रुक रही है")
        calm_out = post_call.build_reports(_call_for(calm, 22))
        self.assertEqual(calm_out["mother_questions_hi"], list(post_call._EMERGENCY_HI))
        self.assertEqual(calm_out["yes_summaries_hi"], [])
        self.assertIn("emergency: active danger sign now; hospital or 108. सांस रुक रही है", calm_out["red_flags"])
        calm_positive = calm_out["doctor_markdown"].split(post_call._POSITIVE_HEADING, 1)[1].split("\n### ", 1)[0]
        self.assertNotIn("No positive gate alerts.", calm_positive)
        self.assertIn("Mother's words: सांस रुक रही है.", calm_positive)

    def test_two_step_followup_notes_reach_reports_without_internal_keys(self):
        entry = {
            "state": "concern",
            "child_ids": ["child_1"],
            "notes": "",
            "followups": {
                "nicu_reason": {"state": "concern", "note": "साँस की तकलीफ"},
                "nicu_days": {"state": "concern", "note": "चार दिन"},
            },
        }
        self.assertEqual(post_call._human_notes(entry), "साँस की तकलीफ; चार दिन")
        state = cursor_tests._profile(name="मीना", children=[cursor_tests._child(4, "male")])
        state["answers"]["birth_nicu"] = entry
        state["answers"]["close_anything_else"] = {
            "state": "concern",
            "child_ids": [],
            "notes": "",
            "followups": {"open_concern_detail": {"state": "concern", "note": "रात को जागना"}},
        }
        state["phase"] = "closing"
        state["close_reason"] = "complete"
        call = _call_for(state, 23)
        synthesis = post_call.build_reports(call)
        report = mother_report.build_v3(call, synthesis)
        document = mother_report.render_html(report)
        page1 = document.split('class="page2"', 1)[0]
        for surface in (
            page1,
            synthesis["doctor_markdown"],
            "\n".join(synthesis["red_flags"]),
            "\n".join(synthesis["yes_summaries_hi"]),
        ):
            self.assertIn("साँस की तकलीफ", surface)
            self.assertIn("चार दिन", surface)
            self.assertNotIn("nicu_reason", surface)
            self.assertNotIn("open_concern_detail", surface)
        self.assertEqual(len(synthesis["yes_summaries_hi"]), 2)
        self.assertIn("रात को जागना", synthesis["yes_summaries_hi"][1])
        self.assertIn("close_anything_else", synthesis["doctor_markdown"])
        self.assertIn("birth_nicu", synthesis["doctor_markdown"])
        self.assertTrue(any("close_anything_else" in flag for flag in synthesis["red_flags"]))
        self.assertTrue(any("birth_nicu" in flag for flag in synthesis["red_flags"]))
        self.assertEqual(len(synthesis["mother_questions_hi"]), 3)
        self.assertIn(post_call._ROUTINE_HI[0], synthesis["mother_questions_hi"])
        self.assertNotIn("concern", page1.lower())
        self.assertIn("रात को जागना", page1)
        self.assertIn("साँस की तकलीफ", "\n".join(report["yes"]))

    def test_page2_uses_marital_and_multi_child_wording(self):
        unmarried = apply_turn(cursor_tests._profile(marital="unmarried", name="पूजा"), "clear")
        self.assertEqual(unmarried["active_qid"], "preg_thalassemia")
        unmarried = apply_turn(unmarried, "clear")
        unmarried_page = _page2(_call_for(unmarried, 31))
        self.assertIn(call_cursor.question_hi_for(unmarried, get_spec("preg_thalassemia")), unmarried_page)
        self.assertNotIn("पति", unmarried_page)
        self.assertNotIn(get_spec("preg_thalassemia").question_hi, unmarried_page)

        about = apply_turn(cursor_tests._profile(marital="about_to_marry", name="पूजा"), "clear")
        about = apply_turn(about, "clear")
        about_page = _page2(_call_for(about, 32))
        self.assertIn("होने वाले पति", about_page)
        self.assertNotIn(get_spec("preg_thalassemia").question_hi, about_page)
        self.assertNotIn(get_spec("fam_related").question_hi, about_page)

        kids = [cursor_tests._child(6, "male", 1), cursor_tests._child(2, "female", 2)]
        two = cursor_tests._clear_until(cursor_tests._profile(children=kids, name="पूजा"), "birth_jaundice")
        nicu = get_spec("birth_nicu")
        two_page = _page2(_call_for(two, 33))
        self.assertIn(nicu.question_multi_hi, two_page)
        self.assertNotIn(nicu.question_hi, two_page)

    def test_disease_questions_and_summaries_stay_grammatical(self):
        samples = (
            "Sickle Cell",
            "sickle-cell",
            "सिकल सेल",
            "Cystic Fibrosis",
            "सिस्टिक फाइब्रोसिस",
            "Fanconi",
            "फैंकोनी",
            "PKU",
        )
        for term in samples:
            cleaned = post_call._scrub_diagnostic_terms(f"उसे {term} बताया")
            self.assertNotIn(term.casefold(), cleaned.casefold())
            self.assertIn("बताया", cleaned)
        self.assertIn("थैलेसीमिया", post_call._scrub_diagnostic_terms("थैलेसीमिया जाँच में माइनर"))
        self.assertNotIn("थैलेसीमिया मेजर", post_call._scrub_diagnostic_terms("थैलेसीमिया मेजर की बात"))

        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
            "body_bleeding_bruise",
        )
        state = apply_turn(state, "concern", note="चोट पर खून देर तक बहता है")
        state["phase"] = "closing"
        state["close_reason"] = "complete"
        call = _call_for(state, 34)
        synthesis = post_call.build_reports(call)
        mother = " ".join(synthesis["mother_questions_hi"] + synthesis["yes_summaries_hi"])
        self.assertNotIn("Sickle", mother)
        self.assertNotIn("सिकल", mother)
        self.assertIn("खून", synthesis["yes_summaries_hi"][0])
        self.assertIn("खून", synthesis["mother_questions_hi"][0])
        self.assertTrue(all(question.endswith("?") for question in synthesis["mother_questions_hi"]))
        self.assertGreaterEqual(len(synthesis["mother_questions_hi"]), 3)

        fitted = post_call._fit_questions(
            ["क्या मेरे बच्चे को Sickle Cell है?", "क्या सिकल सेल है?"],
            concern_questions=synthesis["mother_questions_hi"],
        )
        scrubbed_mother = " ".join([*fitted, post_call._scrub_diagnostic_terms("बच्चे को सिकल सेल है।")])
        self.assertNotIn("Sickle", scrubbed_mother)
        self.assertNotIn("सिकल", scrubbed_mother)
        self.assertGreaterEqual(len(fitted), 3)

    def test_technical_failure_renders_incomplete_non_emergency_mother_pdf(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
            "birth_jaundice",
        )
        state = close_technical_failure(state)
        call = _call_for(state, 99)
        synthesis = post_call.build_reports(call)
        self.assertFalse(any("emergency" in flag for flag in synthesis["red_flags"]))
        self.assertNotIn("Active emergency.", synthesis["doctor_markdown"])
        self.assertIn("Close reason: technical_failure.", synthesis["doctor_markdown"])
        for emergency_q in post_call._EMERGENCY_HI:
            self.assertNotIn(emergency_q, synthesis["mother_questions_hi"])

        report = mother_report.build_v3(call, synthesis)
        self.assertTrue(report["incomplete"])
        self.assertFalse(report["emergency"])
        self.assertEqual(report["status"], "अधूरी कॉल")
        self.assertTrue(report["unasked"])

        doc = mother_report.render_html(report)
        self.assertIn("अधूरी कॉल", doc)
        self.assertIn("यह कॉल अधूरी रही।", doc)
        self.assertIn("पूछा नहीं गया", doc)
        self.assertNotIn("108", doc)
        self.assertNotIn("नज़दीकी अस्पताल", doc)

    def test_build_reports_is_deterministic(self):
        state = _bleed_cursor()
        close_note = "शाम को हल्का बुखार रहता है"
        state["answers"]["close_anything_else"] = {
            "state": "concern",
            "child_ids": [],
            "notes": close_note,
            "followups": {"open_concern_detail": {"state": "concern", "note": close_note}},
        }
        call = _call_for(state, 41)
        synthesis = post_call.build_reports(call)
        self.assertEqual(synthesis, post_call.build_reports(call))
        self.assertIn(close_note, "\n".join(synthesis["yes_summaries_hi"]))
        self.assertIn(close_note, synthesis["doctor_markdown"])
        self.assertIn(close_note, "\n".join(synthesis["red_flags"]))
        self.assertTrue(synthesis["red_flags"])
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])
        pdf = mother_report.render_pdf(mother_report.build_v3(call, synthesis))
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreaterEqual(_pages(pdf), 2)

    def test_scrubber_strips_disease_names_from_doctor_markdown_and_mother_notes(self):
        state = _bleed_cursor()
        state["profile"]["caller_name"] = "रेखा Hemophilia"
        state["profile"]["children"][0]["name"] = "आरव Pompe"
        state["answers"]["body_bleeding_bruise"]["notes"] = "Suspected Hemophilia and Pompe"
        state["answers"]["close_anything_else"] = {
            "state": "concern",
            "child_ids": [],
            "notes": "डॉक्टर ने Duchenne और ड्यूशेन का नाम लिया",
            "followups": {},
        }
        call = _call_for(state, 43)
        result = post_call.build_reports(call)
        for heading in post_call._HEADINGS:
            prefix = f"{heading}\n" if heading == post_call._HEADINGS[0] else f"\n\n{heading}\n"
            self.assertIn(prefix, result["doctor_markdown"])
        combined = "\n".join([
            result["mother_name_hi"],
            result["children_label_hi"],
            result["doctor_markdown"],
            *result["yes_summaries_hi"],
            *result["mother_questions_hi"],
            *result["red_flags"],
        ])
        for name in _FORBIDDEN:
            self.assertNotIn(name.lower(), combined.lower())
        self.assertNotIn("ड्यूशेन", combined)
        self.assertIn("नाम लिया", result["doctor_markdown"])
        self.assertIn("रेखा", result["mother_name_hi"])
        self.assertIn("आरव", result["children_label_hi"])


class TokenPrivacyTests(unittest.TestCase):
    def test_share_token_qid_and_review_round_trip(self):
        call_id = db.create_call("gemini", caller_name="पूजा", clinic_code="kem")
        call = db.get_call(call_id)
        self.assertTrue(call["share_token"])
        self.assertGreaterEqual(len(call["share_token"]), 8)
        self.assertEqual(call["clinic_code"], "kem")
        with db._db_lock:
            db.conn.execute("UPDATE calls SET share_token = '' WHERE id = ?", (call_id,))
            db.conn.commit()
        refilled = db.get_call(call_id)
        self.assertTrue(refilled["share_token"])
        self.assertEqual(db.get_call_by_token(refilled["share_token"])["id"], call_id)
        self.assertIsNone(db.get_call_by_token("not-a-real-token"))
        db.append_transcript(call_id, "user", "हाँ खून बहता है", qid="body_bleeding_bruise")
        db.append_transcript(call_id, "user", "और सूजन", qid="body_joint_swelling")
        turns = db.get_call(call_id)["transcript"]
        self.assertEqual([turn.get("qid") for turn in turns], ["body_bleeding_bruise", "body_joint_swelling"])
        saved = db.save_clinician_review(call_id, "missed_flag", "joint swelling")
        self.assertEqual(saved["rating"], "missed_flag")
        self.assertEqual(db.get_call(call_id)["clinician_review"]["note"], "joint swelling")
        with self.assertRaises(ValueError):
            db.save_clinician_review(call_id, "great")

    def test_token_pdf_separates_two_poojas_and_saves_summary(self):
        calm_id = _store(_childless_cursor(), "Pooja")
        bleed_id = _store(_bleed_cursor(), "Pooja")
        calm = db.get_call(calm_id)
        bleed = db.get_call(bleed_id)
        self.assertEqual(calm["mother_name"], "Pooja")
        self.assertEqual(bleed["mother_name"], "Pooja")
        self.assertNotEqual(calm["share_token"], bleed["share_token"])

        with TestClient(shishu_app.app) as client:
            calm_pdf = client.post(f"/api/calls/by-token/{calm['share_token']}/mother.pdf")
            bleed_pdf = client.post(f"/api/calls/by-token/{bleed['share_token']}/mother.pdf")
            self.assertEqual(calm_pdf.status_code, 200, calm_pdf.text)
            self.assertEqual(bleed_pdf.status_code, 200, bleed_pdf.text)
            self.assertTrue(calm_pdf.content.startswith(b"%PDF"))
            self.assertTrue(bleed_pdf.content.startswith(b"%PDF"))
            self.assertNotEqual(calm_pdf.content, bleed_pdf.content)
            self.assertGreaterEqual(_pages(calm_pdf.content), 2)
            self.assertGreaterEqual(_pages(bleed_pdf.content), 2)
            again = client.get(f"/api/calls/by-token/{calm['share_token']}/mother.pdf")
            self.assertEqual(again.status_code, 200)
            self.assertEqual(again.content, calm_pdf.content)
            missing = client.get("/api/calls/by-token/not-a-real-token/mother.pdf")
            self.assertEqual(missing.status_code, 404)
            other = client.get(f"/api/calls/by-token/{bleed['share_token']}/mother.pdf")
            self.assertEqual(other.status_code, 200)
            self.assertNotEqual(other.content, calm_pdf.content)

        calm_saved = db.get_call(calm_id)["summary_mother"]
        bleed_saved = db.get_call(bleed_id)["summary_mother"]
        self.assertTrue(calm_saved.strip())
        self.assertTrue(bleed_saved.strip())
        self.assertIn("खून", bleed_saved)
        self.assertNotIn("खून", calm_saved)
        self.assertNotEqual(calm_saved, bleed_saved)
        calm_text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(calm_pdf.content)).pages)
        for question in _CHILD_QUESTIONS:
            self.assertNotIn(question[:40], calm_text)


def _store(state, mother_name: str) -> int:
    call_id = db.create_call("gemini", caller_name=mother_name)
    db.save_cursor_state(call_id, state)
    with db._db_lock:
        db.conn.execute(
            "UPDATE calls SET mother_name = ? WHERE id = ?",
            (mother_name, call_id),
        )
        db.conn.commit()
    db.complete_call(call_id, 180, "", "", "")
    return call_id


if __name__ == "__main__":
    unittest.main()
