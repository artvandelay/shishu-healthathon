"""Stage 2 reports: cursor PDF, disease-name scrub, and token privacy. No live call."""

import asyncio
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
import test_v4_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_profile, apply_turn  # noqa: E402
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
            state = apply_turn(state, "concern", child_ref="oldest")
            continue
        if isinstance(state["pending_followup_idx"], int) and state["pending_followup_idx"] >= 0:
            state = apply_turn(state, "clear")
            continue
        if qid in concerns:
            state = apply_turn(state, "concern", note=concerns[qid], child_ref="oldest")
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


def _headed(positive_lines: list[str]) -> str:
    parts = []
    for heading in post_call._HEADINGS:
        if heading == post_call._POSITIVE_HEADING:
            body = "\n".join(positive_lines)
        else:
            body = "- Noted."
        parts.append(f"{heading}\n{body}")
    return "\n\n".join(parts)


def _page2(call: dict) -> str:
    synthesis = post_call.deterministic_fallback(call)
    document = mother_report.render_html(mother_report.build_v3(call, synthesis))
    return document.split('class="page2"', 1)[1]


class CursorReportTests(unittest.TestCase):
    def test_childless_cursor_omits_child_questions_and_renders_two_pages(self):
        state = _childless_cursor()
        self.assertEqual(state["close_reason"], "complete")
        call = _call_for(state, 7)
        synthesis = post_call.deterministic_fallback(call)
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])
        self.assertGreaterEqual(len(synthesis["mother_questions_hi"]), 3)
        self.assertLessEqual(len(synthesis["mother_questions_hi"]), 6)
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
        doctor = "\n\n".join(f"{heading}\n- Noted." for heading in post_call._HEADINGS)
        dirty = {
            "mother_name_hi": "पूजा",
            "children_label_hi": "7 साल बेटा",
            "mother_questions_hi": [
                "क्या मेरे बच्चे को Duchenne है?",
                "Hemophilia की जाँच करानी चाहिए?",
            ],
            "yes_summaries_hi": ["डॉक्टर ने Hemophilia और Pompe बताया।"],
            "doctor_markdown": doctor,
            "red_flags": ["Q04 bleeding"],
        }
        clean = post_call._finish(dirty)
        blob = " ".join(clean["mother_questions_hi"] + clean["yes_summaries_hi"] + [clean["mother_name_hi"]])
        for name in _FORBIDDEN:
            self.assertNotIn(name, blob)
            self.assertNotIn(name.lower(), blob.lower())
        self.assertGreaterEqual(len(clean["mother_questions_hi"]), 3)
        self.assertLessEqual(len(clean["mother_questions_hi"]), 6)

        async def explode(_call):
            raise ConnectionError("offline")

        call = _call_for(_bleed_cursor(), 3)
        with patch("post_call._from_gemini", new=explode):
            synthesis = asyncio.run(post_call.synthesize_call(call))
        mother = " ".join(synthesis["mother_questions_hi"] + synthesis["yes_summaries_hi"])
        for name in _FORBIDDEN:
            self.assertNotIn(name.lower(), mother.lower())
        self.assertIn("खून", mother)
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])

    def test_omitted_closing_concern_is_backfilled_into_every_report(self):
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
        state["answers"]["close_anything_else"] = {
            "state": "concern",
            "child_ids": [],
            "notes": note,
            "followups": {"open_concern_detail": {"state": "concern", "note": note}},
        }
        state["phase"] = "closing"
        state["close_reason"] = "complete"
        call = _call_for(state, 9)
        sections = []
        for heading in post_call._HEADINGS:
            if heading.startswith("### Positive"):
                lines = [f"- {qid}: noted." for qid in qids]
            else:
                lines = ["- Noted."]
            sections.append(heading + "\n" + "\n".join(lines))
        raw = {
            "mother_name_hi": "मीना",
            "children_label_hi": "4 साल बेटा",
            "mother_questions_hi": [
                f"डॉक्टर, {notes[qid]} की जाँच क्या होनी चाहिए?" for qid in reversed(qids)
            ],
            "yes_items_hi": [
                {"qid": qid, "summary_hi": f"{notes[qid]} की बात है।"}
                for qid in reversed(qids)
            ],
            "doctor_markdown": "\n\n".join(sections),
            "red_flags": [f"{qid}: noted" for qid in qids],
        }
        synthesis = post_call._finish(post_call._coerce(raw, call))
        self.assertEqual(
            [key for key in synthesis],
            [
                "mother_name_hi",
                "children_label_hi",
                "mother_questions_hi",
                "yes_summaries_hi",
                "doctor_markdown",
                "red_flags",
            ],
        )
        self.assertEqual(len(synthesis["yes_summaries_hi"]), 7)
        expected = [f"{notes[qid]} की बात है।" for qid in qids]
        self.assertEqual(synthesis["yes_summaries_hi"][:6], expected)
        self.assertIn(note, synthesis["yes_summaries_hi"][6])
        self.assertEqual(synthesis["mother_questions_hi"], raw["mother_questions_hi"])
        self.assertEqual(len(synthesis["mother_questions_hi"]), 6)
        asked = "\n".join(synthesis["mother_questions_hi"])
        self.assertNotIn("मैंने बताया कि", asked)
        self.assertNotIn("मैंने हाँ कहा कि", asked)
        self.assertNotIn(note, asked)
        document = mother_report.render_html(mother_report.build_v3(call, synthesis))
        page1 = document.split('class="page2"', 1)[0]
        for qid in qids:
            self.assertIn(notes[qid], page1)
            self.assertIn(qid, synthesis["doctor_markdown"])
        self.assertIn(note, page1)
        profile = synthesis["doctor_markdown"].split("### Positive", 1)[0]
        self.assertIn("Additional mother disclosure (close_anything_else)", profile)
        self.assertIn(note, profile)
        self.assertIn(note, "\n".join(synthesis["red_flags"]))
        self.assertTrue(any("close_anything_else" in flag for flag in synthesis["red_flags"]))
        for qid in qids:
            self.assertTrue(
                any(qid in flag for flag in synthesis["red_flags"]),
                qid,
            )
        fallback = post_call.deterministic_fallback(call)
        self.assertIn(note, fallback["yes_summaries_hi"][-1])
        self.assertIn(note, fallback["doctor_markdown"].split("### Positive", 1)[0])
        self.assertTrue(any("close_anything_else" in flag for flag in fallback["red_flags"]))

    def test_devanagari_disease_names_and_followups_reach_the_brief(self):
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
        brief = post_call._answered_brief(state)
        self.assertIn("followups=", brief)
        self.assertIn("fracture_count_or_bowing", brief)
        self.assertIn("दो बार", brief)
        self.assertIn("nicu_reason", brief)
        self.assertIn("nicu_days", brief)
        self.assertIn("सांस", brief)
        self.assertIn("तीन दिन", brief)
        sentence = post_call._concern_sentence(
            get_spec("gate_bones_fractures"),
            state["answers"]["gate_bones_fractures"],
        )
        self.assertIn("दो बार", sentence)

    def test_v3_build_saves_flags_and_admin_labels(self):
        bleed_id = _store(_bleed_cursor(), "Pooja")
        calm_id = _store(_childless_cursor(), "Pooja")

        async def explode(_call):
            raise ConnectionError("offline")

        with patch("post_call._from_gemini", new=explode):
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
        content = post_call._user_content(call)
        self.assertLess(content.index("Close reason: emergency"), content.index("Cursor answers"))
        self.assertLess(content.index("Emergency note: अभी बेहोश है"), content.index("Cursor answers"))

        raw = {
            "mother_name_hi": "पूजा",
            "children_label_hi": "3 साल बेटी",
            "mother_questions_hi": list(post_call._ROUTINE_HI),
            "yes_items_hi": [],
            "doctor_markdown": _headed([
                "- None documented in this call.",
                "- No positive gate alerts.",
                "- Active emergency. Caller was told to go to the nearest hospital or call 108.",
            ]),
            "red_flags": [
                "body_bleeding_bruise: bruising noted",
                "emergency: active danger sign now; hospital or 108",
            ],
        }
        synthesis = post_call._finish(post_call._coerce(raw, call))
        flags = "\n".join(synthesis["red_flags"])
        self.assertEqual(flags.count("emergency: active danger sign now"), 1)
        self.assertIn("emergency: active danger sign now; hospital or 108. अभी बेहोश है", flags)
        positive = synthesis["doctor_markdown"].split(post_call._POSITIVE_HEADING, 1)[1].split("\n### ", 1)[0]
        self.assertIn(
            "- Active emergency. Caller was told to go to the nearest hospital or call 108. Mother's words: अभी बेहोश है.",
            positive,
        )
        self.assertEqual(positive.count("Active emergency."), 1)
        self.assertNotIn("None documented", positive)
        self.assertNotIn("No positive", positive)
        self.assertNotIn("None reported", positive)
        for routine in post_call._ROUTINE_HI:
            self.assertNotIn(routine, synthesis["mother_questions_hi"])
        self.assertIn("खून", synthesis["mother_questions_hi"][0])
        self.assertEqual(synthesis["mother_questions_hi"][1:], list(post_call._EMERGENCY_HI[:2]))
        fallback = post_call.deterministic_fallback(call)
        self.assertIn("Mother's words: अभी बेहोश है.", fallback["doctor_markdown"])
        self.assertIn("emergency: active danger sign now; hospital or 108. अभी बेहोश है", fallback["red_flags"])
        for routine in post_call._ROUTINE_HI:
            self.assertNotIn(routine, fallback["mother_questions_hi"])

        calm = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(3, "female")]),
            "ill_fasting",
        )
        calm = apply_turn(calm, "emergency", note="सांस रुक रही है")
        calm_raw = {
            "mother_name_hi": "पूजा",
            "children_label_hi": "3 साल बेटी",
            "mother_questions_hi": list(post_call._ROUTINE_HI),
            "yes_items_hi": [],
            "doctor_markdown": _headed(["- None reported in this call."]),
            "red_flags": ["nothing urgent"],
        }
        calm_out = post_call._finish(post_call._coerce(calm_raw, _call_for(calm, 22)))
        self.assertEqual(calm_out["mother_questions_hi"], list(post_call._EMERGENCY_HI))
        self.assertEqual(calm_out["yes_summaries_hi"], [])
        self.assertIn("emergency: active danger sign now; hospital or 108. सांस रुक रही है", calm_out["red_flags"])
        calm_positive = calm_out["doctor_markdown"].split(post_call._POSITIVE_HEADING, 1)[1].split("\n### ", 1)[0]
        self.assertNotIn("None reported", calm_positive)
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
        raw = {
            "mother_name_hi": "मीना",
            "children_label_hi": "4 साल बेटा",
            "mother_questions_hi": ["क्या टीके पूरे हैं?"],
            "yes_items_hi": [{"qid": "birth_nicu", "summary_hi": "NICU stay"}],
            "doctor_markdown": _headed(["- Noted."]),
            "red_flags": ["screening"],
        }
        synthesis = post_call._finish(post_call._coerce(raw, call))
        fallback = post_call.deterministic_fallback(call)
        report = mother_report.build_v3(call, synthesis)
        document = mother_report.render_html(report)
        page1 = document.split('class="page2"', 1)[0]
        for surface in (
            page1,
            synthesis["doctor_markdown"],
            "\n".join(synthesis["red_flags"]),
            fallback["doctor_markdown"],
            "\n".join(fallback["red_flags"]),
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
        self.assertEqual(
            synthesis["mother_questions_hi"],
            ["क्या टीके पूरे हैं?", *post_call._ROUTINE_HI[:2]],
        )
        asked = "\n".join(synthesis["mother_questions_hi"])
        self.assertNotIn("मैंने बताया कि", asked)
        self.assertNotIn("मैंने हाँ कहा कि", asked)
        self.assertNotIn("साँस की तकलीफ", asked)
        self.assertNotIn("चार दिन", asked)
        self.assertNotIn("concern", page1.lower())
        self.assertIn("रात को जागना", page1)
        self.assertIn("साँस की तकलीफ", "\n".join(report["yes"]))
        self.assertNotIn("साँस की तकलीफ", "\n".join(report["questions"]))

    def test_grouped_gemini_questions_are_not_padded_with_templates(self):
        def planted(name, children, concerns):
            state = cursor_tests._profile(name=name, children=children)
            for qid, entry in concerns.items():
                state["answers"][qid] = entry
            state["phase"] = "closing"
            state["close_reason"] = "complete"
            return state

        def concern(note, followups=None, child_ids=None):
            return {
                "state": "concern",
                "child_ids": child_ids or [],
                "notes": note,
                "followups": followups or {},
            }

        gemini_64 = [
            "मेरे 6 साल के बेटे को जन्म के बाद पीलिया के लिए 3 दिन नीली लाइट में रखा गया था — क्या आगे कोई दिक्कत हो सकती है?",
            "मेरे सगे मामा के बेटे को बार-बार खून चढ़ाना पड़ता है — क्या मेरे बच्चों की भी खून की जाँच करानी चाहिए?",
            "मेरे पिताजी को 42 साल की उम्र में बीपी हुआ था — बच्चों के खान-पान में कोई खास ध्यान रखना चाहिए?",
        ]
        call_64 = _call_for(planted(
            "कविता",
            [
                cursor_tests._child(11, "female", 1),
                cursor_tests._child(6, "male", 2),
                cursor_tests._child(2, "male", 3),
            ],
            {
                "birth_jaundice": concern(
                    "3 दिन नीली लाइट",
                    {"jaundice_days": {"state": "concern", "note": "3 दिन"}},
                    ["child_2"],
                ),
                "fam_transfusion": concern(
                    "सगे मामा के बेटे को",
                    {"fam_who": {"state": "concern", "note": "सगे मामा के बेटे को"}},
                ),
                "fam_bp_stroke": concern(
                    "",
                    {
                        "bp_who": {"state": "concern", "note": "पिता"},
                        "bp_age": {"state": "concern", "note": "42 साल"},
                    },
                ),
            },
        ), 64)
        yes_64 = [
            {"qid": "birth_jaundice", "summary_hi": "बीच वाले बेटे को 3 दिन नीली लाइट में रखा गया।"},
            {"qid": "fam_transfusion", "summary_hi": "सगे मामा के बेटे को बार-बार खून चढ़ता है।"},
            {"qid": "fam_bp_stroke", "summary_hi": "पिता को 42 साल की उम्र में बीपी हुआ।"},
        ]
        raw_64 = {
            "mother_name_hi": "कविता",
            "children_label_hi": "11 साल बेटी · 6 साल बेटा · 2 साल बेटा",
            "mother_questions_hi": gemini_64,
            "yes_items_hi": yes_64,
            "doctor_markdown": _headed([
                "- Q03 birth_jaundice: blue light.",
                "- Q11 fam_transfusion: family transfusions.",
                "- Q11 fam_bp_stroke: father at 42.",
            ]),
            "red_flags": [f"{item['qid']}: noted" for item in yes_64],
        }
        out_64 = post_call._finish(post_call._coerce(raw_64, call_64))
        self.assertEqual(out_64["mother_questions_hi"], gemini_64)
        asked_64 = "\n".join(out_64["mother_questions_hi"])
        self.assertNotIn("मैंने बताया कि", asked_64)
        self.assertNotIn("मैंने हाँ कहा कि", asked_64)
        self.assertEqual(
            out_64["yes_summaries_hi"],
            [item["summary_hi"] for item in yes_64],
        )
        for item in yes_64:
            self.assertIn(item["qid"], out_64["doctor_markdown"])
            self.assertTrue(any(item["qid"] in flag for flag in out_64["red_flags"]))

        notes_10 = {
            "birth_term": "तीन हफ्ते पहले",
            "birth_nicu": "दो हफ्ते",
            "body_joint_swelling": "गिरने पर घुटना सूजता है",
            "preg_thalassemia": "ट्रेट बताया गया",
            "fam_related": "चचेरे मामा के बेटे",
            "fam_miscarriage": "3 महीने में एक बार",
            "fam_infant_death": "6 महीने में गुजर गई",
            "fam_heart_kidney": "बूढ़ों को दिल की बीमारी",
            "fam_bp_stroke": "सगे रिश्तेदार को 35 साल में",
            "fam_same_problem": "बहन के बेटे को भी वही दिक्कत",
            "close_anything_else": "सर दर्द",
        }
        gemini_10 = [
            "मेरा 2 साल का बच्चा 3 हफ्ते पहले पैदा हुआ और 2 हफ्ते NICU में रहा — आगे कोई खास ध्यान रखना है?",
            "मेरे तीनों बच्चों के घुटनों में गिरने पर सूजन आ जाती है — क्या कोई खास जाँच करानी चाहिए?",
            "मेरी और पति की खून जाँच में थैलेसीमिया ट्रेट बताया गया — बच्चों पर इसका क्या असर हो सकता है?",
            "हमारी शादी रिश्ते में हुई है — बच्चों के लिए कोई खास सलाह लेनी चाहिए?",
            "मेरी बहन के बेटे को भी बच्चों जैसी दिक्कत है और परिवार में एक बच्चा अचानक गुजर गया — क्या यह परिवार की बात हो सकती है?",
            "मुझे अक्सर सिरदर्द रहता है — राहत के लिए क्या करना चाहिए?",
        ]
        call_10 = _call_for(planted(
            "रीटा",
            [
                cursor_tests._child(3, "male", 1),
                cursor_tests._child(2, "male", 2),
                cursor_tests._child(1, "female", 3),
            ],
            {
                qid: concern(
                    note,
                    {"open_concern_detail": {"state": "concern", "note": note}}
                    if qid == "close_anything_else"
                    else {},
                    ["child_2"] if qid in {"birth_term", "birth_nicu"} else [],
                )
                for qid, note in notes_10.items()
            },
        ), 10)
        yes_10 = [
            {"qid": qid, "summary_hi": f"{note}।"}
            for qid, note in notes_10.items()
        ]
        raw_10 = {
            "mother_name_hi": "रीटा",
            "children_label_hi": "3 साल बेटा · 2 साल बेटा · 1 साल बेटी",
            "mother_questions_hi": gemini_10,
            "yes_items_hi": list(reversed(yes_10)),
            "doctor_markdown": _headed([f"- {qid}: noted." for qid in notes_10]),
            "red_flags": [f"{qid}: noted" for qid in notes_10],
        }
        out_10 = post_call._finish(post_call._coerce(raw_10, call_10))
        self.assertEqual(out_10["mother_questions_hi"], gemini_10)
        asked_10 = "\n".join(out_10["mother_questions_hi"])
        self.assertNotIn("मैंने बताया कि", asked_10)
        self.assertNotIn("मैंने हाँ कहा कि", asked_10)
        self.assertNotIn("सर दर्द", asked_10)
        self.assertIn("सिरदर्द", asked_10)
        self.assertEqual(
            out_10["yes_summaries_hi"],
            [item["summary_hi"] for item in yes_10],
        )
        page1 = mother_report.render_html(mother_report.build_v3(call_10, out_10)).split('class="page2"', 1)[0]
        for qid, note in notes_10.items():
            self.assertIn(note, page1)
            self.assertIn(qid, out_10["doctor_markdown"])
            self.assertTrue(any(qid in flag for flag in out_10["red_flags"]), qid)
        self.assertNotIn("मैंने बताया कि", page1)

        emergency = cursor_tests._profile(
            name="पूजा",
            children=[cursor_tests._child(3, "female")],
        )
        emergency["phase"] = "closing"
        emergency["close_reason"] = "emergency"
        emergency["emergency_note"] = "अभी सांस नहीं आ रही"
        paraphrased = [
            "क्या बच्चे का वज़न और कद उम्र के हिसाब से ठीक बढ़ रहा है?",
            "क्या सारे टीके लग चुके हैं या टीका कार्ड में कुछ बाकी है?",
            "घर पर खाने-पीने में क्या ध्यान रखूँ?",
        ]
        for line in paraphrased:
            self.assertNotIn(line, post_call._ROUTINE_HI)
        calm = post_call._finish(post_call._coerce({
            "mother_name_hi": "पूजा",
            "children_label_hi": "3 साल बेटी",
            "mother_questions_hi": paraphrased,
            "yes_items_hi": [],
            "doctor_markdown": _headed(["- None reported in this call."]),
            "red_flags": [],
        }, _call_for(emergency, 60)))
        self.assertEqual(calm["mother_questions_hi"], list(post_call._EMERGENCY_HI))
        self.assertEqual(calm["yes_summaries_hi"], [])
        for line in paraphrased:
            self.assertNotIn(line, calm["mother_questions_hi"])

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
        raw = {
            "mother_name_hi": "पूजा",
            "children_label_hi": "4 साल बेटा",
            "mother_questions_hi": [
                "क्या मेरे बच्चे को Sickle Cell है?",
                "क्या सिकल सेल है?",
            ],
            "yes_items_hi": [
                {"qid": "body_bleeding_bruise", "summary_hi": "बच्चे को सिकल सेल है।"},
            ],
            "doctor_markdown": _headed(["- Noted."]),
            "red_flags": ["Q04 bleeding"],
        }
        synthesis = post_call._finish(post_call._coerce(raw, call))
        mother = " ".join(synthesis["mother_questions_hi"] + synthesis["yes_summaries_hi"])
        self.assertNotIn("Sickle", mother)
        self.assertNotIn("सिकल", mother)
        self.assertNotIn("को है?", mother)
        self.assertIn("खून", synthesis["yes_summaries_hi"][0])
        self.assertIn("खून", synthesis["mother_questions_hi"][0])
        self.assertTrue(all(question.endswith("?") for question in synthesis["mother_questions_hi"]))
        self.assertGreaterEqual(len(synthesis["mother_questions_hi"]), 3)


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

        async def explode(_call):
            raise ConnectionError("offline")

        with TestClient(shishu_app.app) as client, patch("post_call._from_gemini", new=explode):
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


def _assert_live(raw: dict, call: dict, label: str) -> dict:
    if not isinstance(raw, dict):
        raise AssertionError(f"{label} synthesis was not a JSON object")
    for key in (
        "mother_name_hi",
        "children_label_hi",
        "mother_questions_hi",
        "doctor_markdown",
        "red_flags",
    ):
        if key not in raw:
            raise AssertionError(f"{label} JSON missing {key}")
    if "yes_items_hi" not in raw and "yes_summaries_hi" not in raw:
        raise AssertionError(f"{label} JSON missing yes_items_hi")
    synthesis = post_call._finish(post_call._coerce(raw, call))
    mother = " ".join(
        [synthesis["mother_name_hi"], synthesis["children_label_hi"]]
        + synthesis["mother_questions_hi"]
        + synthesis["yes_summaries_hi"]
    )
    for name in _FORBIDDEN:
        if name.lower() in mother.lower():
            raise AssertionError(f"{label} mother text still contains {name}")
    for heading in post_call._HEADINGS:
        if heading not in synthesis["doctor_markdown"]:
            raise AssertionError(f"{label} doctor note missing {heading}")
    if not (3 <= len(synthesis["mother_questions_hi"]) <= 6):
        raise AssertionError(f"{label} question count {len(synthesis['mother_questions_hi'])}")
    report = mother_report.build_v3(call, synthesis)
    pdf = mother_report.render_pdf(report)
    if not pdf.startswith(b"%PDF") or _pages(pdf) < 2:
        raise AssertionError(f"{label} PDF was not a 2-page document")
    folder = Path(_TMP) / "smoke"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}.pdf").write_bytes(pdf)
    print(
        f"{label}: questions={len(synthesis['mother_questions_hi'])} "
        f"yes={len(synthesis['yes_summaries_hi'])} pdf_bytes={len(pdf)} pages={_pages(pdf)}"
    )
    return synthesis


async def _live_smoke() -> None:
    bleed = _call_for(_bleed_cursor(), 1)
    calm = _call_for(_childless_cursor(), 2)
    bleed_raw = await post_call._from_gemini(bleed)
    calm_raw = await post_call._from_gemini(calm)
    bleed_out = _assert_live(bleed_raw, bleed, "bleed-infant")
    calm_out = _assert_live(calm_raw, calm, "childless")
    if not bleed_out["yes_summaries_hi"]:
        raise AssertionError("bleed call produced no हाँ lines")
    if calm_out["yes_summaries_hi"]:
        raise AssertionError("childless call invented a हाँ line")
    if calm_out["children_label_hi"].strip():
        raise AssertionError("childless call invented a child label")
    document = mother_report.render_html(mother_report.build_v3(calm, calm_out))
    page2 = document.split('class="page2"', 1)[1]
    for question in _CHILD_QUESTIONS:
        if question in page2:
            raise AssertionError("childless page 2 included a child question")
    print(f"smoke pdfs: {Path(_TMP) / 'smoke'}")


def run_live_smoke() -> None:
    asyncio.run(_live_smoke())


if __name__ == "__main__":
    run_live_smoke()
