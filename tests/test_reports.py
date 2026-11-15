"""Stage 2 reports: deterministic cursor PDF, disease-name scrub, and token privacy."""

import asyncio
import io
import json
import threading
import unittest
from unittest.mock import patch

import tests  # noqa: F401
from tests.helpers import clear_until, make_child, make_profile_state

import app as shishu_app  # noqa: E402
import call_cursor  # noqa: E402
import db  # noqa: E402
import mother_report  # noqa: E402
import post_call  # noqa: E402
import prompts  # noqa: E402
import report_service  # noqa: E402
from call_cursor import apply_interpreted_turn, apply_profile, apply_turn, close_technical_failure  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from triage_catalog import EVER_PREGNANT_HI, active_specs, get_spec  # noqa: E402

_CHILD_QUESTIONS = tuple(spec.question_hi for spec in active_specs() if "no_children" in spec.skip_if)
_FORBIDDEN = ("Duchenne", "Hemophilia", "Haemophilia", "Pompe", "Gaucher", "SCID", "Wiskott", "ड्यूशेन", "हीमोफीलिया")


def _childless_cursor():
    state = apply_turn(make_profile_state(name="पूजा", children=[]), "clear")
    state, hold = apply_profile(state, ever_pregnant=False)
    assert hold is None, hold
    guard = 0
    while state["phase"] == "asking" and guard < 90:
        state = apply_turn(state, "clear")
        guard += 1
    return state


def _bleed_cursor():
    kids = [make_child(7, "male", 1), make_child(2, "female", 2)]
    state = make_profile_state(name="रेखा", children=kids)
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
        elif isinstance(state["pending_followup_idx"], int) and state["pending_followup_idx"] >= 0:
            state = apply_turn(state, "clear")
        elif qid in concerns:
            state = apply_turn(state, "concern", note=concerns[qid], child_ref="child_1")
        else:
            state = apply_turn(state, "clear")
    return state


def _call_for(state, call_id):
    turns = []
    for spec, entry, status in post_call._rows(state):
        turns.append({"role": "assistant", "qid": spec.id, "text": spec.question_hi})
        turns.append({"role": "user", "qid": spec.id, "text": f"हाँ। {entry.get('notes') or 'हाँ'}" if status == "concern" else ("पता नहीं" if status == "unknown" else "नहीं")})
    return {"id": call_id, "started_at": "2026-11-12 04:00:00", "duration_sec": 180, "cursor_state": state, "transcript": turns}


def _pages(pdf: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf)).pages)


def _pdf_text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)


def _page2(call: dict) -> str:
    return mother_report.render_html(mother_report.build_report(call, post_call.build_reports(call))).split('class="page2"', 1)[1]


def _store(state, mother_name: str) -> int:
    call_id = db.create_call(caller_name=mother_name)
    db.save_cursor_state(call_id, state)
    with db._db_lock:
        db.conn.execute("UPDATE calls SET mother_name = ? WHERE id = ?", (mother_name, call_id))
        db.conn.commit()
    db.complete_call(call_id, 180)
    return call_id


class CursorReportTests(unittest.TestCase):
    def test_childless_cursor_omits_child_questions_and_renders_two_pages(self):
        state = _childless_cursor()
        self.assertEqual(state["close_reason"], "complete")
        call = _call_for(state, 7)
        synthesis = post_call.build_reports(call)
        for heading in post_call._HEADINGS:
            self.assertIn(heading, synthesis["doctor_markdown"])
        self.assertTrue(3 <= len(synthesis["mother_questions_hi"]) <= 6)
        self.assertEqual((synthesis["yes_summaries_hi"], synthesis["children_label_hi"]), ([], ""))
        self.assertNotIn("मेरे बच्चे", " ".join(synthesis["mother_questions_hi"]))
        report = mother_report.build_report(call, synthesis)
        document = mother_report.render_html(report)
        page1, page2 = document.split('class="page2"', 1)
        for question in _CHILD_QUESTIONS:
            self.assertNotIn(question, page2, question[:24])
        self.assertIn(get_spec("fam_related").question_hi, page2)
        self.assertIn(mother_report.THALASSEMIA_LINE, page1)
        self.assertIn("यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है।", document)
        pdf = mother_report.render_pdf(report)
        self.assertTrue(pdf.startswith(b"%PDF") and _pages(pdf) >= 2)
        extracted = _pdf_text(pdf)
        self.assertTrue("SHISHU-7" in extracted and "काँच के बक्से" not in extracted)

    def test_diagnostic_turns_stay_in_doctor_note_and_are_omitted_from_mother_pdf(self):
        kids = [make_child(7, "male", 1, name="आरव"), make_child(3, "female", 2, name="मीरा")]
        state = clear_until(make_profile_state(children=kids, name="रेखा"), "body_bleeding_bruise")
        state = apply_turn(state, "concern", child_ref="child_2", note="चोट पर खून देर तक बहता है", diagnostic=False)
        state = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active", "diagnostic_note": True,
            "side_qid": "body_bleeding_bruise", "side_action": "add_or_correct", "side_outcome": "concern",
            "side_note": "डॉक्टर ने Hemophilia और ड्यूशेन और Pompe बताया",
        })
        state = clear_until(state, "preg_thalassemia")
        state = apply_turn(state, "concern", note="थैलेसीमिया जाँच में माइनर बताया", diagnostic=False)
        state = clear_until(state, "close_anything_else")
        state = apply_turn(state, "concern", note="डॉक्टर ने Duchenne और Sickle Cell का नाम लिया", diagnostic=True)
        state = apply_turn(state, "clear")

        call = _call_for(state, 43)
        result = post_call.build_reports(call)
        for heading in post_call._HEADINGS:
            self.assertIn(heading, result["doctor_markdown"])
        for term in ("Hemophilia", "ड्यूशेन", "Pompe", "Duchenne", "Sickle Cell", "child_2 (female, age 3.0, मीरा)"):
            self.assertIn(term, result["doctor_markdown"])
            self.assertTrue(any(term in flag for flag in result["red_flags"]), term)

        report = mother_report.build_report(call, result)
        html_doc = mother_report.render_html(report)
        pdf_text = _pdf_text(mother_report.render_pdf(report))
        mother_blob = "\n".join([*result["yes_summaries_hi"], *result["mother_questions_hi"], html_doc, pdf_text])
        for name in _FORBIDDEN:
            self.assertNotIn(name.lower(), mother_blob.lower())
        self.assertIn("चोट पर खून देर तक बहता है", result["yes_summaries_hi"][0])
        self.assertIn("मीरा 3 साल बेटी", result["yes_summaries_hi"][0])
        self.assertIn("मीरा 3 साल बेटी के लिए:", result["mother_questions_hi"][0])
        self.assertIn("थैलेसीमिया जाँच में माइनर बताया", mother_blob)
        self.assertTrue(all(q.endswith("?") for q in result["mother_questions_hi"]) and 3 <= len(result["mother_questions_hi"]) <= 6)

    def test_concerns_unknowns_negatives_and_closing_disclosure_reach_every_report(self):
        note = "पति को गुप्त रोग; बच्चे पति के नहीं हैं"
        notes = {
            "birth_nicu": "तीन दिन काँच का बक्सा", "body_bleeding_bruise": "चोट पर खून देर तक बहता है",
            "body_transfusion": "दो बार खून चढ़ा", "fam_fits_delay": "भाभी के बेटे को दौरे",
            "fam_cancer": "मामा को कैंसर", "fam_heart_kidney": "दादा को दिल की बीमारी",
        }
        state = make_profile_state(name="मीना", children=[make_child(4, "male")])
        for qid, text in notes.items():
            state["answers"][qid] = {"state": "concern", "child_ids": [] if qid.startswith("fam_") else ["child_1"], "notes": text, "followups": {}}
        state["answers"]["birth_jaundice"] = {"state": "clear", "child_ids": [], "notes": "", "followups": {}}
        state["answers"]["ill_fasting"] = {"state": "unknown", "child_ids": [], "notes": "", "followups": {}}
        state["answers"]["close_anything_else"] = {"state": "concern", "child_ids": [], "notes": note, "followups": {"open_concern_detail": {"state": "concern", "note": note}}}
        state["phase"], state["close_reason"] = "closing", "complete"

        call = _call_for(state, 9)
        synthesis = post_call.build_reports(call)
        self.assertEqual(list(synthesis.keys()), ["mother_name_hi", "children_label_hi", "mother_questions_hi", "yes_summaries_hi", "doctor_markdown", "red_flags"])
        self.assertEqual((len(synthesis["yes_summaries_hi"]), len(synthesis["mother_questions_hi"])), (7, 6))
        for qid, text in notes.items():
            self.assertTrue(any(text in line for line in synthesis["yes_summaries_hi"]), qid)
            self.assertIn(qid, synthesis["doctor_markdown"])
            self.assertTrue(any(qid in flag for flag in synthesis["red_flags"]), qid)
        self.assertIn(note, synthesis["yes_summaries_hi"][-1])
        self.assertIn(note, synthesis["doctor_markdown"].split("### Positive", 1)[0])
        self.assertIn("Unknown. " + get_spec("ill_fasting").need_en, synthesis["doctor_markdown"])
        self.assertIn("Clear: " + get_spec("birth_jaundice").need_en, synthesis["doctor_markdown"].split("### Pertinent Negatives", 1)[1])
        self.assertTrue(any("close_anything_else" in flag for flag in synthesis["red_flags"]))

        page1, page2 = mother_report.render_html(mother_report.build_report(call, synthesis)).split('class="page2"', 1)
        self.assertIn(note, page1)
        self.assertTrue("पता नहीं" in page2 and get_spec("ill_fasting").question_hi in page2 and get_spec("birth_jaundice").question_hi in page2)

    def test_followups_multichild_labels_and_db_saving(self):
        state = _bleed_cursor()
        state["answers"]["gate_bones_fractures"] = {"state": "concern", "child_ids": ["child_2"], "notes": "गिरने पर हड्डी टूटी", "followups": {"fracture_count_or_bowing": {"state": "concern", "note": "दो बार"}}}
        state["answers"]["birth_nicu"] = {"state": "concern", "child_ids": ["child_1"], "notes": "NICU", "followups": {"nicu_reason": {"state": "concern", "note": "सांस"}, "nicu_days": {"state": "concern", "note": "तीन दिन"}}}
        synthesis = post_call.build_reports(_call_for(state, 19))
        for expected in ("दो बार", "सांस", "तीन दिन"):
            self.assertIn(expected, "\n".join(synthesis["yes_summaries_hi"]))
            self.assertIn(expected, synthesis["doctor_markdown"])
            self.assertIn(expected, "\n".join(synthesis["red_flags"]))
        self.assertIn("child_2 (female, age 2", synthesis["doctor_markdown"])
        self.assertTrue(any("2 साल बेटी" in line for line in synthesis["yes_summaries_hi"]))
        self.assertTrue(any("2 साल बेटी के लिए:" in q for q in synthesis["mother_questions_hi"]))
        self.assertNotIn("fracture_count_or_bowing", synthesis["doctor_markdown"])
        self.assertNotIn("nicu_reason", synthesis["doctor_markdown"])

        bleed_id, calm_id = _store(_bleed_cursor(), "Pooja"), _store(_childless_cursor(), "Pooja")
        asyncio.run(report_service.mother_pdf(bleed_id))
        asyncio.run(report_service.mother_pdf(calm_id))
        bleed, calm = db.get_call(bleed_id), db.get_call(calm_id)
        self.assertTrue(bleed["red_flags"] and bleed["summary_doctor"].strip() and bleed["summary_mother"].strip() and "साल" in bleed["child_age"])
        self.assertEqual(calm["red_flags"], [])
        self.assertTrue(calm["mother_name"] and calm["mother_name"] != "Mother not named")

    def test_emergency_keeps_the_danger_and_drops_routine_filler(self):
        state = apply_turn(clear_until(make_profile_state(children=[make_child(3, "female")]), "body_bleeding_bruise"), "concern", note="चोट पर खून देर तक बहता है")
        state = apply_turn(state, "emergency", note="  अभी   बेहोश है  ")
        call = _call_for(state, 21)
        synthesis = post_call.build_reports(call)
        flags = "\n".join(synthesis["red_flags"])
        self.assertEqual(flags.count("emergency: active danger sign now"), 1)
        self.assertIn("अभी बेहोश है", flags)
        positive = synthesis["doctor_markdown"].split(post_call._POSITIVE_HEADING, 1)[1].split("\n### ", 1)[0]
        self.assertIn("- Active emergency. Caller was told to go to the nearest hospital or call 108. Mother's words: अभी बेहोश है.", positive)
        self.assertEqual(synthesis["mother_questions_hi"][1:], list(post_call._EMERGENCY_HI[:2]))
        doc = mother_report.render_html(mother_report.build_report(call, synthesis))
        self.assertTrue("108" in doc and "नज़दीकी अस्पताल" in doc)

        calm = apply_turn(clear_until(make_profile_state(children=[make_child(3, "female")]), "ill_fasting"), "emergency", note="सांस रुक रही है")
        calm_out = post_call.build_reports(_call_for(calm, 22))
        self.assertEqual(calm_out["mother_questions_hi"], list(post_call._EMERGENCY_HI))
        self.assertEqual(calm_out["yes_summaries_hi"], [])

    def test_two_step_followup_notes_page2_wording_and_technical_failure(self):
        state = make_profile_state(name="मीना", children=[make_child(4, "male")])
        state["answers"]["birth_nicu"] = {"state": "concern", "child_ids": ["child_1"], "notes": "", "followups": {"nicu_reason": {"state": "concern", "note": "साँस की तकलीफ"}, "nicu_days": {"state": "concern", "note": "चार दिन"}}}
        state["answers"]["close_anything_else"] = {"state": "concern", "child_ids": [], "notes": "", "followups": {"open_concern_detail": {"state": "concern", "note": "रात को जागना"}}}
        state["phase"], state["close_reason"] = "closing", "complete"
        synthesis = post_call.build_reports(_call_for(state, 23))
        self.assertEqual(synthesis, post_call.build_reports(_call_for(state, 23)))
        doc_html = mother_report.render_html(mother_report.build_report(_call_for(state, 23), synthesis))
        page1 = doc_html.split('class="page2"', 1)[0]
        for surface in (page1, synthesis["doctor_markdown"], "\n".join(synthesis["red_flags"])):
            self.assertTrue("साँस की तकलीफ" in surface and "चार दिन" in surface and "nicu_reason" not in surface and "open_concern_detail" not in surface)
        self.assertIn("रात को जागना", synthesis["yes_summaries_hi"][1])

        unmarried = apply_turn(apply_turn(make_profile_state(marital="unmarried", name="पूजा"), "clear"), "clear")
        unmarried_p2 = _page2(_call_for(unmarried, 31))
        self.assertTrue("पति" not in unmarried_p2 and get_spec("preg_thalassemia").question_hi not in unmarried_p2)
        about = apply_turn(apply_turn(make_profile_state(marital="about_to_marry", name="पूजा"), "clear"), "clear")
        self.assertIn("होने वाले पति", _page2(_call_for(about, 32)))
        two = clear_until(make_profile_state(children=[make_child(6, "male", 1), make_child(2, "female", 2)], name="पूजा"), "birth_jaundice")
        self.assertIn(get_spec("birth_nicu").question_multi_hi, _page2(_call_for(two, 33)))

        tf = close_technical_failure(clear_until(make_profile_state(children=[make_child(4, "male")], name="पूजा"), "birth_jaundice"))
        tf_synth = post_call.build_reports(_call_for(tf, 99))
        self.assertIn("Close reason: technical_failure.", tf_synth["doctor_markdown"])
        tf_rep = mother_report.build_report(_call_for(tf, 99), tf_synth)
        self.assertEqual((tf_rep["incomplete"], tf_rep["emergency"], tf_rep["status"]), (True, False, "अधूरी कॉल"))
        tf_html = mother_report.render_html(tf_rep)
        self.assertTrue("यह कॉल अधूरी रही।" in tf_html and "108" not in tf_html)

    def test_ever_pregnant_appears_on_page2_and_not_clinical_concern(self):
        clear_s = apply_turn(apply_turn(make_profile_state(name="पूजा", children=[]), "clear"), "clear")
        while clear_s["phase"] == "asking":
            clear_s = apply_turn(clear_s, "clear")
        self.assertIn(EVER_PREGNANT_HI, _page2(_call_for(clear_s, 51)))

        unk_s = apply_turn(apply_turn(make_profile_state(name="पूजा", children=[]), "clear"), "unknown")
        while unk_s["phase"] == "asking":
            unk_s = apply_turn(unk_s, "clear")
        unk_rep = mother_report.build_report(_call_for(unk_s, 52), post_call.build_reports(_call_for(unk_s, 52)))
        self.assertIn(EVER_PREGNANT_HI, unk_rep["unknown"])

        inc_s = close_technical_failure(apply_turn(make_profile_state(name="पूजा", children=[]), "clear"))
        inc_rep = mother_report.build_report(_call_for(inc_s, 53), post_call.build_reports(_call_for(inc_s, 53)))
        unasked_preg = next(g for g in inc_rep["unasked"] if g["title"] == mother_report._GROUP_HI["pregnancy"])
        self.assertIn(EVER_PREGNANT_HI, unasked_preg["items"])

        conc_s = apply_turn(apply_turn(make_profile_state(name="पूजा", children=[]), "clear"), "concern")
        while conc_s["phase"] == "asking":
            conc_s = apply_turn(conc_s, "clear")
        self.assertEqual(prompts.recorded_concerns_hi(conc_s), [])
        self.assertIn("अगर आगे इनमें से कुछ भी दिखे", prompts.closing_hi(conc_s))
        conc_synth = post_call.build_reports(_call_for(conc_s, 54))
        self.assertEqual((conc_synth["red_flags"], conc_synth["yes_summaries_hi"]), ([], []))
        self.assertIn("Ever pregnant: True.", conc_synth["doctor_markdown"])
        self.assertNotIn("ever_pregnant", conc_synth["doctor_markdown"].split("### Positive", 1)[1])

        prof_clear = make_profile_state(name="पूजा", children=[], ever_pregnant=False)
        while prof_clear["phase"] == "asking":
            prof_clear = apply_turn(prof_clear, "clear")
        prof_rep = mother_report.build_report(_call_for(prof_clear, 55), post_call.build_reports(_call_for(prof_clear, 55)))
        self.assertIn(EVER_PREGNANT_HI, next(g for g in prof_rep["groups"] if g["title"] == mother_report._GROUP_HI["pregnancy"])["items"])

    def test_post_call_edge_cases_and_incomplete_unknown_copy(self):
        state = call_cursor.new_cursor_state()
        state["answers"]["close_anything_else"] = {"state": "concern", "child_ids": [], "notes": "रात को रोना", "followups": {"open_concern_detail": {"state": "concern", "note": "रात को रोना"}}}
        state["phase"], state["close_reason"], state["emergency_note"] = "closing", "emergency", "सांस लेने में तकलीफ"
        synth = post_call.build_reports(_call_for(state, 61))
        self.assertEqual(synth["mother_name_hi"], "")
        self.assertTrue("- Caller: not reported." in synth["doctor_markdown"] and "Ever pregnant: not reported." in synth["doctor_markdown"] and "close_anything_else close_anything_else:" not in synth["doctor_markdown"])
        self.assertEqual(mother_report.build_report(_call_for(state, 61), synth)["who"], "माँ")

        calm_emg = dict(call_cursor.new_cursor_state(), phase="closing", close_reason="emergency", emergency_note="दौरा पड़ रहा है")
        self.assertIn("- Chief complaint: Active emergency reported during call.", post_call.build_reports(_call_for(calm_emg, 62))["doctor_markdown"])

        unk_only = clear_until(make_profile_state(children=[make_child(4, "male")], name="पूजा"), "birth_jaundice")
        unk_only["answers"] = {"birth_jaundice": {"state": "unknown", "child_ids": [], "notes": "", "followups": {}}}
        unk_only = close_technical_failure(unk_only)
        rep = mother_report.build_report(_call_for(unk_only, 63), post_call.build_reports(_call_for(unk_only, 63)))
        html_doc = mother_report.render_html(rep)
        self.assertNotIn("इस कॉल में कोई जवाब दर्ज नहीं हुआ।", html_doc)
        self.assertIn("इस कॉल में कोई बात हाँ में नहीं आई।", html_doc)

    def test_db_label_reset_event_retention_and_inflight_cancel_shield(self):
        call_id = db.create_call(caller_name="पूजा")
        state = make_profile_state(name="पूजा", children=[make_child(5, "male")])
        db.save_cursor_state(call_id, state)
        self.assertEqual((db.get_call(call_id)["mother_name"], db.get_call(call_id)["child_age"]), ("पूजा", "5 साल बेटा"))
        db.save_cursor_state(call_id, {"profile": {"caller_name": "", "children": []}, "answers": {}})
        self.assertEqual((db.get_call(call_id)["mother_name"], db.get_call(call_id)["child_age"]), ("", ""))
        db.save_cursor_state(call_id, state)
        db.save_report_text(call_id, "doc", "mom", [], "", "")
        self.assertEqual((db.get_call(call_id)["mother_name"], db.get_call(call_id)["child_age"]), ("", ""))
        db.append_transcript(call_id, "user", "   \n\t  ")
        self.assertEqual(db.get_call(call_id)["transcript"], [])
        for i in range(300):
            db.append_call_event(call_id, f"evt_{i}")
        self.assertEqual(len(db.get_call(call_id)["events"]), 300)

        bug = db.create_bug_report(call_id, "Tester", "Other", "shot", 5, extra_snapshot={"screenshot": "data:image/png;base64,AAAA"}, include_call_body=False)
        listed = next(item for item in db.list_bug_reports() if item["id"] == bug["id"])
        self.assertNotIn("snapshot_json", listed)
        self.assertEqual(listed["snapshot"]["client"]["screenshot"], "data:image/png;base64,AAAA")

        cid2 = _store(_childless_cursor(), "पूजा")
        entered, release, calls_seen = threading.Event(), threading.Event(), []
        real_render = mother_report.render_pdf

        def slow_render(report):
            calls_seen.append(1)
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("timeout")
            return real_render(report)

        async def run():
            with patch.object(mother_report, "render_pdf", side_effect=slow_render):
                t1 = asyncio.create_task(report_service.mother_pdf(cid2))
                t2 = asyncio.create_task(report_service.mother_pdf(cid2))
                while not entered.is_set():
                    await asyncio.sleep(0.01)
                t1.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await t1
                release.set()
                return await t2

        pdf, name = asyncio.run(run())
        self.assertTrue(pdf.startswith(b"%PDF") and name.endswith(".pdf") and len(calls_seen) == 1)
        self.assertEqual(report_service._inflight, {})

        ev_cid = db.create_call(caller_name="पूजा")
        report_service._event(ev_cid, "render_end", {"ms": 42, "bytes": 1024, "model": "old", "error_class": "Err", "reason": "legacy"})
        self.assertEqual(db.get_call(ev_cid)["events"][-1]["meta"], {"ms": 42, "bytes": 1024})


class TokenPrivacyTests(unittest.TestCase):
    def test_share_token_qid_and_review_round_trip(self):
        call_id = db.create_call(caller_name="पूजा", clinic_code="kem")
        token = db.get_call(call_id)["share_token"]
        self.assertTrue(len(token) >= 8 and db.get_call(call_id)["clinic_code"] == "kem")
        with db._db_lock:
            db.conn.execute("UPDATE calls SET share_token = '' WHERE id = ?", (call_id,))
            db.conn.commit()
        refilled = db.get_call(call_id)
        self.assertEqual(db.get_call_by_token(refilled["share_token"])["id"], call_id)
        self.assertIsNone(db.get_call_by_token("not-a-real-token"))
        db.append_transcript(call_id, "user", "हाँ खून बहता है", qid="body_bleeding_bruise")
        db.append_transcript(call_id, "user", "और सूजन", qid="body_joint_swelling")
        self.assertEqual([t.get("qid") for t in db.get_call(call_id)["transcript"]], ["body_bleeding_bruise", "body_joint_swelling"])
        self.assertEqual(db.save_clinician_review(call_id, "missed_flag", "joint swelling")["rating"], "missed_flag")
        with self.assertRaises(ValueError):
            db.save_clinician_review(call_id, "great")

    def test_token_pdf_separates_two_poojas_and_saves_summary(self):
        calm_id, bleed_id = _store(_childless_cursor(), "Pooja"), _store(_bleed_cursor(), "Pooja")
        calm, bleed = db.get_call(calm_id), db.get_call(bleed_id)
        self.assertNotEqual(calm["share_token"], bleed["share_token"])

        with TestClient(shishu_app.app) as client:
            calm_pdf = client.post(f"/api/calls/by-token/{calm['share_token']}/mother.pdf")
            bleed_pdf = client.post(f"/api/calls/by-token/{bleed['share_token']}/mother.pdf")
            self.assertEqual((calm_pdf.status_code, bleed_pdf.status_code), (200, 200))
            self.assertTrue(calm_pdf.content.startswith(b"%PDF") and bleed_pdf.content.startswith(b"%PDF"))
            self.assertNotEqual(calm_pdf.content, bleed_pdf.content)
            self.assertTrue(_pages(calm_pdf.content) >= 2 and _pages(bleed_pdf.content) >= 2)
            self.assertEqual(client.get(f"/api/calls/by-token/{calm['share_token']}/mother.pdf").content, calm_pdf.content)
            self.assertEqual(client.get("/api/calls/by-token/not-a-real-token/mother.pdf").status_code, 404)
            self.assertNotEqual(client.get(f"/api/calls/by-token/{bleed['share_token']}/mother.pdf").content, calm_pdf.content)

        calm_saved, bleed_saved = db.get_call(calm_id)["summary_mother"], db.get_call(bleed_id)["summary_mother"]
        self.assertTrue("खून" in bleed_saved and "खून" not in calm_saved)
        calm_text = _pdf_text(calm_pdf.content)
        for question in _CHILD_QUESTIONS:
            self.assertNotIn(question[:40], calm_text)


if __name__ == "__main__":
    unittest.main()
