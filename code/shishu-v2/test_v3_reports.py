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
import db  # noqa: E402
import mother_report  # noqa: E402
import post_call  # noqa: E402
import test_v3_cursor as cursor_tests  # noqa: E402
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
    while state["phase"] == "asking" and guard < 40:
        state = apply_turn(state, "clear")
        guard += 1
    return state


def _bleed_cursor():
    kids = [cursor_tests._child(7, "male", 1), cursor_tests._child(2, "female", 2)]
    state = cursor_tests._profile(name="रेखा", children=kids)
    concerns = {
        "gate_bleeding_joints": "बड़े वाले को चोट पर खून देर तक बहता है",
        "body_joint_swelling": "गिरने पर घुटना सूजता है और कुछ दिनों में ठीक हो जाता है",
        "fam_loss_gate": "परिवार में एक बच्चे की मौत हुई",
        "fam_infant_death": "मामा का बेटा छह महीने में गुजर गया",
    }
    guard = 0
    while state["phase"] == "asking" and guard < 80:
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
        db.append_transcript(call_id, "user", "हाँ खून बहता है", qid="gate_bleeding_joints")
        db.append_transcript(call_id, "user", "और सूजन", qid="body_joint_swelling")
        turns = db.get_call(call_id)["transcript"]
        self.assertEqual([turn.get("qid") for turn in turns], ["gate_bleeding_joints", "body_joint_swelling"])
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
    db.record_finding_db(call_id, "profile_mother_name", "info", mother_name, False)
    db.complete_call(call_id, 180, "", "", "")
    return call_id


def _assert_live(raw: dict, call: dict, label: str) -> dict:
    if not isinstance(raw, dict):
        raise AssertionError(f"{label} synthesis was not a JSON object")
    for key in (
        "mother_name_hi",
        "children_label_hi",
        "mother_questions_hi",
        "yes_summaries_hi",
        "doctor_markdown",
        "red_flags",
    ):
        if key not in raw:
            raise AssertionError(f"{label} JSON missing {key}")
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
