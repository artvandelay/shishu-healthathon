"""Mother PDF uses structured state and does not print an unasked topic as no."""

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

    def test_unknown_is_not_printed_as_no(self):
        state = self._state()
        state["topics"]["birth_cleft"]["state"] = "unknown"
        report = mother_report.build(_call(state))
        question = checklist.question_hi("birth_cleft")
        rendered = mother_report.render_html(report)
        self.assertNotIn(question, _joined(report["groups"]))
        self.assertNotIn(question + '</span><span class="ans">नहीं</span>', rendered)

    def test_no_response_and_not_asked_are_not_printed_as_no(self):
        state = self._state()
        state["topics"]["body_blue"]["state"] = "no_response"
        state["topics"]["body_tummy"]["state"] = "not_asked"
        report = mother_report.build(_call(state))
        rendered = mother_report.render_html(report)
        nos = _joined(report["groups"])
        for topic_id in ("body_blue", "body_tummy"):
            question = checklist.question_hi(topic_id)
            self.assertNotIn(question, nos)
            self.assertNotIn(question + '</span><span class="ans">नहीं</span>', rendered)

    def test_not_applicable_is_omitted_from_both_detail_sections(self):
        state = self._state()
        state["topics"]["body_puberty"]["state"] = "not_applicable"
        report = mother_report.build(_call(state))
        question = checklist.question_hi("body_puberty")
        self.assertNotIn(question, _joined(report["groups"]))
        self.assertNotIn(question, _joined(report["unasked"]))

    def test_emergency_structured_state_is_incomplete_and_shows_the_notice(self):
        state = self._state()
        state["close_reason"] = "emergency"
        state["phase"] = "closing"
        report = mother_report.build(_call(state))
        rendered = mother_report.render_html(report)
        self.assertTrue(report["incomplete"])
        self.assertTrue(report["emergency"])
        self.assertIn("अधूरी कॉल", rendered)
        self.assertIn("तुरंत नज़दीकी अस्पताल", rendered)

    def test_structured_thalassemia_no_adds_the_screening_line(self):
        state = self._state()
        state["topics"]["preg_thalassemia"]["state"] = "answered"
        state["topics"]["preg_thalassemia"]["value"] = False
        report = mother_report.build(_call(state))
        self.assertTrue(any("थैलेसीमिया" in question for question in report["questions"]))

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
        state["profile"]["children"][1]["sex"] = "female"
        state["profile"]["children"][0]["sex"] = "male"
        text = " ".join(mother_report.build(_call(state))["yes"])
        self.assertIn("राम", text)
        self.assertIn("छोटी बेटी", text)
        self.assertNotIn("child_", text)
        self.assertIn("बीपी", text)
        self.assertNotIn("child_", text)
        self.assertNotIn("condition", text)
        self.assertNotIn("चिल्द", text)

    def test_not_asked_thalassemia_does_not_add_the_screening_line(self):
        state = self._state()
        state["topics"]["preg_thalassemia"]["state"] = "not_asked"
        report = mother_report.build(_call(state))
        self.assertFalse(any("थैलेसीमिया" in question for question in report["questions"]))

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
        self.assertIn(question + '</span><span class="ans">नहीं</span>', rendered)
        self.assertEqual(mother_report.hindi_text("heavy"), "ज़्यादा")
        self.assertNotIn("हेअव्य", mother_report.hindi_text("heavy"))

    def test_unknown_is_printed_under_unknown_not_unasked(self):
        state = self._state()
        state["topics"]["birth_cleft"]["state"] = "unknown"
        report = mother_report.build(_call(state))
        question = checklist.question_hi("birth_cleft")
        rendered = mother_report.render_html(report)
        self.assertIn(question, report["unknown"])
        self.assertNotIn(question, _joined(report["unasked"]))
        self.assertIn(question + '</span><span class="ans">पता नहीं</span>', rendered)

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

    def test_structured_positive_findings_generate_tailored_mother_questions(self):
        state = self._state()
        state["topics"]["body_joint_swelling"]["state"] = "answered"
        state["topics"]["body_joint_swelling"]["value"] = True
        state["topics"]["body_joint_swelling"]["details"] = {"joint": "ghutna", "duration": "4 din"}
        state["topics"]["fam_infant_death"]["state"] = "answered"
        state["topics"]["fam_infant_death"]["value"] = True
        state["topics"]["fam_infant_death"]["details"] = {"age_at_death": "3 mahine", "cause": "fits"}

        report = mother_report.build(_call(state))
        q_blob = " ".join(report["questions"])
        self.assertIn("घुटने", q_blob)
        self.assertIn("गुजर", q_blob)

    def test_reassuring_growth_and_puberty_are_not_haan_lines(self):
        state = self._state()
        state["profile"]["children"] = [
            {"id": "child_1", "age": {"value": 11, "unit": "years"}, "sex": "female"}
        ]
        state["topics"]["body_growth"]["state"] = "answered"
        state["topics"]["body_growth"]["value"] = True
        state["topics"]["body_growth"]["details"] = {"result": "normal"}
        state["topics"]["body_puberty"]["state"] = "answered"
        state["topics"]["body_puberty"]["value"] = True
        state["topics"]["body_puberty"]["details"] = {"timing": "same", "result": "normal"}
        state["topics"]["preg_thalassemia"]["state"] = "answered"
        state["topics"]["preg_thalassemia"]["value"] = True
        state["topics"]["preg_thalassemia"]["details"] = {"result": "normal"}
        report = mother_report.build(_call(state))
        self.assertEqual(report["yes"], [])
        self.assertIn("बेटी", report["who"])

    def test_normal_delivery_with_true_value_is_classified_as_no(self):
        state = self._state()
        state["topics"]["birth_mode"]["state"] = "answered"
        state["topics"]["birth_mode"]["value"] = True
        state["topics"]["birth_mode"]["details"] = {"delivery_type": "normal", "place": "hospital"}
        report = mother_report.build(_call(state))
        self.assertEqual(report["yes"], [])
        question = checklist.question_hi("birth_mode")
        self.assertIn(question, _joined(report["groups"]))


if __name__ == "__main__":
    unittest.main()
