"""Mother PDF uses structured state and does not print an unasked topic as no."""

import io
import unittest

import checklist
import mother_report
import call_state_contract as contract


def _call(state):
    return {
        "id": 1,
        "mother_name": "Test",
        "started_at": "2026-11-08 00:00:00",
        "duration_sec": 60,
        "summary_mother": "",
        "transcript": [],
        "telemetry": {"coverage": [], "close_reason": ""},
        "findings": {},
        "state": state,
    }


def _joined(groups):
    return " ".join(item for group in groups for item in group["items"])


class StructuredMotherReportTests(unittest.TestCase):
    def _state(self):
        return contract.new_call_state()

    def test_explicit_false_is_printed_as_no(self):
        state = self._state()
        state["topics"]["birth_cord_bleed"]["state"] = "answered"
        state["topics"]["birth_cord_bleed"]["value"] = False
        report = mother_report.build(_call(state))
        question = checklist.question_hi("birth_cord_bleed")
        rendered = mother_report.render_html(report)
        self.assertIn(question, _joined(report["groups"]))
        self.assertIn(question + '</span><span class="ans">नहीं</span>', rendered)

    def test_unknown_is_printed_under_unknown_not_unasked(self):
        state = self._state()
        state["topics"]["birth_cleft"]["state"] = "unknown"
        report = mother_report.build(_call(state))
        question = checklist.question_hi("birth_cleft")
        rendered = mother_report.render_html(report)
        self.assertIn(question, report["unknown"])
        self.assertNotIn(question, _joined(report["groups"]))
        self.assertNotIn(question, _joined(report["unasked"]))
        self.assertIn(question + '</span><span class="ans">पता नहीं</span>', rendered)
        self.assertNotIn(question + '</span><span class="ans">नहीं</span>', rendered)

    def test_not_applicable_is_omitted_from_both_detail_sections(self):
        state = self._state()
        state["topics"]["body_puberty"]["state"] = "not_applicable"
        report = mother_report.build(_call(state))
        question = checklist.question_hi("body_puberty")
        self.assertNotIn(question, _joined(report["groups"]))
        self.assertNotIn(question, _joined(report["unasked"]))

    def test_structured_thalassemia_no_adds_the_screening_line(self):
        state = self._state()
        state["topics"]["preg_thalassemia"]["state"] = "answered"
        state["topics"]["preg_thalassemia"]["value"] = False
        report = mother_report.build(_call(state))
        self.assertTrue(any("थैलेसीमिया" in question for question in report["questions"]))

        state = self._state()
        state["topics"]["preg_thalassemia"]["state"] = "not_asked"
        report = mother_report.build(_call(state))
        self.assertFalse(any("थैलेसीमिया" in question for question in report["questions"]))

    def test_yes_line_uses_a_name_or_a_hindi_ordinal_not_a_field_key(self):
        state = self._state()
        state["profile"]["children"] = [
            {"id": "child_1", "name": "राम", "age": {"value": 6, "unit": "years"}, "sex": None},
            {"id": "child_2", "name": None, "age": {"value": 2, "unit": "years"}, "sex": None},
        ]
        state["topics"]["body_fits_milestones"]["state"] = "answered"
        state["topics"]["body_fits_milestones"]["value"] = True
        state["topics"]["body_fits_milestones"]["instances"] = [
            {"person_id": "child_1", "value": True, "details": {}},
        ]
        state["topics"]["fam_bp_stroke"]["state"] = "answered"
        state["topics"]["fam_bp_stroke"]["value"] = True
        state["topics"]["fam_bp_stroke"]["details"] = {
            "condition": "बीपी",
            "who": "मामा",
            "onset_age": "35",
        }
        state["topics"]["birth_nicu"]["state"] = "answered"
        state["topics"]["birth_nicu"]["value"] = True
        state["topics"]["birth_nicu"]["instances"] = [
            {"person_id": "child_2", "value": True, "details": {}},
        ]
        text = " ".join(mother_report.build(_call(state))["yes"])
        self.assertIn("राम", text)
        self.assertIn("छोटा बच्चा", text)
        self.assertIn("बीपी", text)
        self.assertNotIn("child_", text)
        self.assertNotIn("condition", text)
        self.assertNotIn("चिल्द", text)

    def test_operation_is_printed_as_yes_and_normal_as_no(self):
        state = self._state()
        state["topics"]["birth_mode"]["state"] = "answered"
        state["topics"]["birth_mode"]["value"] = "operation"
        state["topics"]["birth_mode"]["details"] = {"delivery_type": "operation"}
        report = mother_report.build(_call(state))
        self.assertTrue(len(report["yes"]) >= 1)
        self.assertTrue(any("ऑपरेशन" in y or "जन्म" in y for y in report["yes"]))

        state_normal = self._state()
        state_normal["topics"]["birth_mode"]["state"] = "answered"
        state_normal["topics"]["birth_mode"]["value"] = "normal"
        state_normal["topics"]["birth_mode"]["details"] = {"delivery_type": "normal"}
        report_normal = mother_report.build(_call(state_normal))
        question = checklist.question_hi("birth_mode")
        rendered = mother_report.render_html(report_normal)
        self.assertIn(question, _joined(report_normal["groups"]))

        # A contradictory true value alongside a normal delivery classifies as no.
        state_contra = self._state()
        state_contra["topics"]["birth_mode"]["state"] = "answered"
        state_contra["topics"]["birth_mode"]["value"] = True
        state_contra["topics"]["birth_mode"]["details"] = {"delivery_type": "normal", "place": "hospital"}
        report_contra = mother_report.build(_call(state_contra))
        self.assertEqual(report_contra["yes"], [])
        self.assertIn(question + '</span><span class="ans">नहीं</span>', rendered)

    def test_english_clinical_details_and_child_ages_render_in_clean_hindi(self):
        state = self._state()
        state["profile"]["caller_name"] = "Shilpa"
        state["profile"]["children"] = [
            {"id": "child_1", "name": "Aarav", "age": {"value": 6, "unit": "years"}, "sex": "male"},
            {"id": "child_2", "name": "Isha", "age": {"value": 6, "unit": "months"}, "sex": "female"},
        ]
        state["topics"]["birth_nicu"]["state"] = "answered"
        state["topics"]["birth_nicu"]["value"] = True
        state["topics"]["birth_nicu"]["instances"] = [
            {"person_id": "child_2", "value": True, "details": {"reason": "jaundice", "days": "4 days"}},
        ]
        state["topics"]["body_fits_milestones"]["state"] = "answered"
        state["topics"]["body_fits_milestones"]["value"] = True
        state["topics"]["body_fits_milestones"]["instances"] = [
            {"person_id": "child_1", "value": True, "details": {"type": "fits", "onset": "late walking"}},
        ]
        state["topics"]["fam_related"]["state"] = "answered"
        state["topics"]["fam_related"]["value"] = True
        state["topics"]["fam_related"]["details"] = {"relationship": "first cousin"}

        report = mother_report.build(_call(state))
        self.assertIn("6 साल", report["who"])
        self.assertIn("6 महीने", report["who"])
        self.assertNotIn("years", report["who"])
        self.assertNotIn("months", report["who"])

        yes_blob = " ".join(report["yes"])
        self.assertIn("पीलिया", yes_blob)
        self.assertIn("झटके", yes_blob)
        self.assertIn("चलने में देरी", yes_blob)
        self.assertIn("सगे कज़िन", yes_blob)
        self.assertNotIn("जौन्दिcए", yes_blob)
        self.assertNotIn("jaundice", yes_blob.lower())

    def test_pdf_bytes_render_the_report(self):
        state = self._state()
        state["topics"]["birth_cleft"]["state"] = "answered"
        state["topics"]["birth_cleft"]["value"] = True
        report = mother_report.build(_call(state))
        pdf = mother_report.render_pdf(report)
        self.assertTrue(pdf.startswith(b"%PDF"))
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf))
        self.assertGreaterEqual(len(reader.pages), 2)
        # pypdf mangles Devanagari matras on extraction; the header line survives verbatim.
        text = "".join(page.extract_text() or "" for page in reader.pages)
        self.assertIn("SHISHU कॉल रिपोर्ट", text)


if __name__ == "__main__":
    unittest.main()
