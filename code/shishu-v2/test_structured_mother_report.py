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
        self.assertIn("बीपी", text)
        self.assertNotIn("child_", text)
        self.assertNotIn("condition", text)
        self.assertNotIn("चिल्द", text)

    def test_not_asked_thalassemia_does_not_add_the_screening_line(self):
        state = self._state()
        state["topics"]["preg_thalassemia"]["state"] = "not_asked"
        report = mother_report.build(_call(state))
        self.assertFalse(any("थैलेसीमिया" in question for question in report["questions"]))


if __name__ == "__main__":
    unittest.main()
