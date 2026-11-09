"""Structured summaries read call state and do not call a provider."""

import asyncio
import copy
import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-structured-summary-")
os.environ.setdefault("DATA_DIR", _TMP)

import answer_extractor  # noqa: E402
import call_state_contract as contract  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import models  # noqa: E402
import mother_report  # noqa: E402
import summarizer  # noqa: E402


def _state():
    return contract.new_call_state()


class StructuredSummaryTests(unittest.TestCase):
    def test_structured_call_returns_without_provider_calls(self):
        async def boom(*_args, **_kwargs):
            raise AssertionError("provider called")

        answer_extractor._from_llm = boom
        state = _state()
        state["topics"]["birth_nicu"]["state"] = "unknown"
        result = asyncio.run(answer_extractor.cached_answers({"id": 1, "state": state, "transcript": []}))
        self.assertEqual(result["version"], 5)
        self.assertEqual(result["answers"]["birth_nicu"]["answer"], "unknown")

    def test_state_to_findings_and_flags_populates_domains_and_mother_questions(self):
        state = _state()
        state["profile"]["caller_name"] = "कविता"
        state["profile"]["children"] = [
            {"id": "child_1", "name": "आरव", "age": {"value": 3, "unit": "years"}, "sex": "male"}
        ]
        state["topics"]["birth_nicu"]["state"] = "answered"
        state["topics"]["birth_nicu"]["value"] = True
        state["topics"]["birth_nicu"]["details"] = {"reason": "jaundice", "days": 4}
        state["topics"]["ill_admissions"]["state"] = "answered"
        state["topics"]["ill_admissions"]["value"] = True
        state["topics"]["ill_admissions"]["details"] = {"count": 3, "reasons": "fever"}

        findings, flags = summarizer.state_to_findings_and_flags(state)
        self.assertIn("birth_nicu", findings)
        self.assertTrue(findings["birth_nicu"]["is_red_flag"])
        self.assertIn("infections", findings)
        self.assertTrue(findings["infections"]["is_red_flag"])
        self.assertTrue(any(f.startswith("birth_nicu:") for f in flags))
        self.assertTrue(any(f.startswith("ill_admissions:") for f in flags))

        extracted = answer_extractor.structured_answers({"state": state})
        self.assertTrue(len(extracted["yes_hindi"]) >= 2)
        self.assertTrue(any("पीलिया" in y for y in extracted["yes_hindi"]))
        self.assertTrue(len(extracted["questions_hindi"]) >= 3)
        mother_text = answer_extractor.mother_summary(extracted, findings, flags)
        self.assertIn("NICU", mother_text)
        self.assertIn("भर्ती", mother_text)

    def test_post_call_corrections_cannot_change_structured_state(self):
        call_id = db.create_call("gemini")
        state = db.initialize_call_state(call_id)
        state["profile"]["caller_name"] = "कविता"
        state["profile"]["children"] = [
            {"id": "child_1", "age": {"value": 2, "unit": "years"}, "sex": "male"}
        ]
        state["topics"]["birth_mode"]["state"] = "answered"
        state["topics"]["birth_mode"]["value"] = True
        state["topics"]["birth_mode"]["details"] = {"delivery_type": "operation"}
        state["topics"]["birth_mode"]["instances"] = [
            {"person_id": "child_1", "value": True, "details": {"delivery_type": "operation"}}
        ]
        db.save_call_state(call_id, state)
        live_findings = {
            "birth_nicu": {"status": "positive", "detail": "captured operation", "is_red_flag": True}
        }
        live_flags = ["birth_mode: operation"]
        db.save_doctor_summary(call_id, "seed", live_findings, live_flags, summary_mother="seed")
        before_state = copy.deepcopy(state)
        before_saved = db.get_call_state(call_id)
        transcript = [
            {"role": "user", "text": "डिलीवरी नॉर्मल हुई, अस्पताल में। मेरा नाम सुनीता है।"},
        ]
        titles = [heading.removeprefix("### ").strip() for heading in summarizer._REQUIRED_HEADINGS]
        captured = []

        async def fake_gemini(user_content, system_prompt="", **_kwargs):
            captured.append((user_content, system_prompt))
            return {
                "doctor_sections": {title: "Not reported." for title in titles},
                "mother_questions": [
                    {"topic_id": "birth_mode", "question": "बच्चे का वज़न डॉक्टर से पूछें?"}
                ],
                "extracted_findings": {
                    "birth_nicu": {
                        "status": "positive",
                        "detail": "3 days NICU for jaundice",
                        "is_red_flag": True,
                    }
                },
                "red_flags": ["birth_nicu: 3 days NICU for jaundice"],
                "profile_corrections": {
                    "caller_name": "सुनीता",
                    "evidence": "मेरा नाम सुनीता है",
                },
                "topic_corrections": [
                    {
                        "topic_id": "birth_mode",
                        "state": "answered",
                        "value": False,
                        "evidence": "डिलीवरी नॉर्मल हुई, अस्पताल में।",
                        "details": {"delivery_type": "normal", "place": "hospital"},
                    },
                    {
                        "topic_id": "birth_nicu",
                        "state": "answered",
                        "value": True,
                        "evidence": "डिलीवरी नॉर्मल हुई, अस्पताल में।",
                        "details": {"days": 3},
                    },
                ],
            }

        orig = summarizer._from_gemini
        summarizer._from_gemini = fake_gemini
        try:
            asyncio.run(
                gemini_live._write_doctor_note(
                    call_id, transcript, live_findings, live_flags, None, state
                )
            )
        finally:
            summarizer._from_gemini = orig

        self.assertEqual(len(captured), 1)
        self.assertNotIn("topic_corrections", captured[0][1])
        self.assertNotIn("profile_corrections", captured[0][1])
        schema_props = summarizer._summary_schema().properties
        self.assertNotIn("topic_corrections", schema_props)
        self.assertNotIn("profile_corrections", schema_props)
        self.assertIn("doctor_sections", schema_props)
        self.assertIn("mother_questions", schema_props)
        self.assertIn("extracted_findings", schema_props)
        self.assertIn("red_flags", schema_props)
        self.assertEqual(state, before_state)
        self.assertEqual(db.get_call_state(call_id), before_saved)
        saved = db.get_call(call_id)
        self.assertEqual(saved["findings"], live_findings)
        self.assertEqual(saved["red_flags"], live_flags)
        self.assertIn("3 days NICU for jaundice", saved["summary_doctor"])
        self.assertIn("बच्चे का वज़न डॉक्टर से पूछें?", saved["summary_mother"])
        report = mother_report.build(
            {"state": saved["state"], "summary_mother": saved["summary_mother"]}
        )
        self.assertTrue(any("ऑपरेशन" in line for line in report["yes"]))
        self.assertFalse(any("काँच" in line or "ICU" in line for line in report["yes"]))
        self.assertIn("कविता", report["who"])
        self.assertNotIn("सुनीता", report["who"])

    def test_text_calls_use_the_configured_flash_model(self):
        self.assertEqual(models.TEXT_MODEL, os.environ.get("SHISHU_TEXT_MODEL", "gemini-3.8-flash"))
        self.assertEqual(db.DEFAULT_SETTINGS["gemini_model"], models.LIVE_MODEL)
        self.assertNotIn("summary_model", db.DEFAULT_SETTINGS)

if __name__ == "__main__":
    unittest.main()
