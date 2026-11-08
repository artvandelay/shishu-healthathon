"""Structured summaries read call state and do not call a provider."""

import asyncio
import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-structured-summary-")
os.environ.setdefault("DATA_DIR", _TMP)

import answer_extractor  # noqa: E402
import call_state_contract as contract  # noqa: E402
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

    def test_unknown_does_not_become_no(self):
        state = _state()
        state["topics"]["birth_cleft"]["state"] = "unknown"
        answers = answer_extractor.structured_answers({"state": state})["answers"]
        self.assertEqual(answers["birth_cleft"]["answer"], "unknown")
        self.assertNotEqual(answers["birth_cleft"]["answer"], "no")

    def test_not_applicable_does_not_become_no_or_not_asked(self):
        state = _state()
        state["topics"]["body_puberty"]["state"] = "not_applicable"
        answer = answer_extractor.structured_answers({"state": state})["answers"]["body_puberty"]["answer"]
        self.assertEqual(answer, "not_applicable")
        self.assertNotIn(answer, {"no", "not_asked"})

    def test_child_instance_appears_in_summary_context(self):
        state = _state()
        state["profile"]["children"] = [
            {"id": "child_1", "name": "A", "age": {"value": 3, "unit": "years"}, "sex": "female"}
        ]
        state["topics"]["birth_nicu"]["state"] = "answered"
        state["topics"]["birth_nicu"]["value"] = True
        state["topics"]["birth_nicu"]["instances"] = [
            {"person_id": "child_1", "value": True, "details": {"reason": "jaundice", "days": 4}}
        ]
        text = summarizer.state_to_summary_context(state)
        self.assertIn("child_1", text)
        self.assertIn("jaundice", text)
        self.assertIn("days", text)

    def test_fallback_doctor_note_distinguishes_unresolved_states(self):
        state = _state()
        state["topics"]["birth_nicu"]["state"] = "no_response"
        state["topics"]["body_blue"]["state"] = "not_asked"
        state["topics"]["body_puberty"]["state"] = "not_applicable"
        note = summarizer.build_state_fallback(state)["summary_doctor"]
        self.assertIn("birth_nicu: no_response", note)
        self.assertIn("body_blue: not_asked", note)
        self.assertIn("body_puberty: not_applicable", note)

    def test_legacy_call_still_uses_the_old_fallback(self):
        out = summarizer.build_fallback_summaries(
            [{"role": "user", "text": "नमस्ते"}],
            {},
            [],
        )
        self.assertIn("Not reported.", out["summary_doctor"])
        self.assertIn("1. ", out["summary_mother"])
        self.assertNotIn("not_applicable", out["summary_doctor"])

    def test_positive_operation_recorded_as_yes_in_answers_and_flags(self):
        state = _state()
        state["topics"]["birth_mode"]["state"] = "answered"
        state["topics"]["birth_mode"]["value"] = "operation"
        answers = answer_extractor.structured_answers({"state": state})["answers"]
        self.assertEqual(answers["birth_mode"]["answer"], "yes")

        fallback = summarizer.build_state_fallback(state)
        self.assertIn("birth_mode: yes", fallback["summary_doctor"])
        self.assertTrue(any(f.startswith("birth_mode:") for f in fallback["red_flags"]))


if __name__ == "__main__":
    unittest.main()
