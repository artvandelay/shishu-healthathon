"""Structured summaries read call state and do not call a provider."""

import asyncio
import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-structured-summary-")
os.environ.setdefault("DATA_DIR", _TMP)

import answer_extractor  # noqa: E402
import call_state  # noqa: E402
import call_state_contract as contract  # noqa: E402
import db  # noqa: E402
import models  # noqa: E402
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

    def test_frontier_gemini_reconciliation_pass_fixes_missed_topic_and_preserves_unasked(self):
        state = _state()
        state["topics"]["body_puberty"]["state"] = "not_applicable"
        state["topics"]["body_puberty"]["source"] = "server_rule"
        state["topics"]["body_puberty"]["reason"] = "all_under_10"
        state["profile"]["children"] = [
            {"id": "child_1", "name": "Aarav", "age": {"value": 2, "unit": "years"}, "sex": "male"}
        ]
        transcript = [
            {"role": "assistant", "text": "आपका नाम क्या है?"},
            {"role": "user", "text": "मेरा नाम सुनीता है। बच्चे को काँच के बक्से में 3 दिन पीलिया के लिए रखा था।"},
        ]
        shell = "\n".join(f"{h}\nNot reported." for h in summarizer._REQUIRED_HEADINGS)
        captured_prompts = []

        async def fake_gemini(user_content, system_prompt="", **_kwargs):
            captured_prompts.append((user_content, system_prompt))
            return {
                "doctor_markdown": shell.replace(
                    "### Birth & Neonatal History\nNot reported.",
                    "### Birth & Neonatal History\n- NICU stay for 3 days due to neonatal jaundice.",
                ),
                "mother_hindi": "1. जन्म के समय पीलिया और काँच के बक्से (NICU) के बारे में डॉक्टर से क्या जाँच पूछें?",
                "extracted_findings": {},
                "red_flags": ["birth_nicu: 3 days NICU for jaundice"],
                "profile_corrections": {"caller_name": "सुनीता"},
                "topic_corrections": {
                    "birth_nicu": {
                        "state": "answered",
                        "value": True,
                        "evidence": "काँच के बक्से में 3 दिन पीलिया के लिए रखा था",
                        "details": {"reason": "पीलिया", "days": "3 दिन"},
                    },
                    "fam_cancer": {
                        "state": "answered",
                        "value": False,
                        "evidence": "hallucinated quote not in transcript",
                    },
                },
            }

        orig_gemini = summarizer._from_gemini
        summarizer._from_gemini = fake_gemini
        try:
            res = asyncio.run(
                summarizer.generate_call_summaries(transcript, {}, [], state=state)
            )
        finally:
            summarizer._from_gemini = orig_gemini

        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("Call Transcript", captured_prompts[0][0])
        self.assertIn("सुनीता", res["state"]["profile"]["caller_name"])
        self.assertEqual(res["state"]["topics"]["birth_nicu"]["state"], "answered")
        self.assertTrue(res["state"]["topics"]["birth_nicu"]["value"])
        # Unasked topic without grounded transcript quote must NOT become answered/no
        self.assertEqual(res["state"]["topics"]["fam_cancer"]["state"], "not_asked")
        # Server skip rule must remain enforced
        self.assertEqual(res["state"]["topics"]["body_puberty"]["state"], "not_applicable")
        self.assertIn("NICU", res["summary_mother"])
        self.assertTrue(any(f.startswith("birth_nicu:") for f in res["red_flags"]))

    def test_frontier_model_can_override_a_false_positive(self):
        state = _state()
        state["profile"]["children"] = [
            {"id": "child_1", "age": {"value": 2, "unit": "years"}, "sex": "male"}
        ]
        state["topics"]["birth_mode"]["state"] = "answered"
        state["topics"]["birth_mode"]["value"] = True
        state["topics"]["birth_mode"]["details"] = {"delivery_type": "operation"}
        state["topics"]["birth_mode"]["instances"] = [
            {"person_id": "child_1", "value": True, "details": {"delivery_type": "operation"}}
        ]
        transcript = [
            {"role": "user", "text": "डिलीवरी नॉर्मल हुई, अस्पताल में। ऑपरेशन नहीं हुआ।"},
        ]
        shell = "\n".join(f"{h}\nNot reported." for h in summarizer._REQUIRED_HEADINGS)
        doctor = shell.replace(
            "### Birth & Neonatal History\nNot reported.",
            "### Birth & Neonatal History\n- Normal hospital delivery.",
        ).replace(
            "### Positive Red Flags & NRROID Gate Alerts\nNot reported.",
            "### Positive Red Flags & NRROID Gate Alerts\nNone.",
        )

        async def fake_gemini(_user_content, system_prompt="", **_kwargs):
            return {
                "doctor_markdown": doctor,
                "mother_hindi": "1. बच्चे का वज़न और टीकाकरण डॉक्टर से पूछें?",
                "extracted_findings": {},
                "red_flags": [],
                "topic_corrections": {
                    "birth_mode": {
                        "state": "answered",
                        "value": False,
                        "evidence": "डिलीवरी नॉर्मल हुई, अस्पताल में।",
                        "details": {"delivery_type": "normal", "place": "hospital"},
                    }
                },
            }

        orig = summarizer._from_gemini
        summarizer._from_gemini = fake_gemini
        try:
            res = asyncio.run(
                summarizer.generate_call_summaries(
                    transcript,
                    {"birth_nicu": {"status": "positive", "detail": "operation", "is_red_flag": True}},
                    ["birth_mode: operation"],
                    state=state,
                )
            )
        finally:
            summarizer._from_gemini = orig

        topic = res["state"]["topics"]["birth_mode"]
        self.assertEqual(topic["value"], False)
        self.assertEqual(topic["details"].get("delivery_type"), "normal")
        self.assertFalse(call_state._is_positive(topic))
        self.assertEqual(call_state.topic_answer_kind(topic), "no")
        self.assertFalse(any(str(flag).startswith("birth_mode:") for flag in res["red_flags"]))
        self.assertNotIn("birth_nicu", res["findings"])
        self.assertNotIn("operation", res["summary_doctor"].lower())
        import mother_report

        report = mother_report.build({"state": res["state"], "summary_mother": res["summary_mother"]})
        self.assertFalse(report["yes"])

    def test_text_calls_use_the_configured_flash_model(self):
        self.assertEqual(models.TEXT_MODEL, os.environ.get("SHISHU_TEXT_MODEL", "gemini-3.8-flash"))
        self.assertEqual(db.DEFAULT_SETTINGS["gemini_model"], models.LIVE_MODEL)
        self.assertNotIn("summary_model", db.DEFAULT_SETTINGS)

    def test_birth_mode_enums_drop_anything_else(self):
        state = _state()
        state["topics"]["birth_mode"]["state"] = "answered"
        state["topics"]["birth_mode"]["value"] = "operation"
        state["topics"]["birth_mode"]["details"] = {
            "delivery_type": "operation",
            "place": "hospital",
            "count": 3,
            "last_date": "jan",
        }
        state["topics"]["birth_mode"]["instances"] = [
            {
                "person_id": "child_1",
                "value": True,
                "details": {"count": 3, "last_date": "jan", "delivery_type": "operation"},
            }
        ]
        quote = "डिलीवरी नॉर्मल हुई, अस्पताल में।"
        transcript = [{"role": "user", "text": quote}]
        kept = summarizer.reconcile_state_with_llm_payload(
            state,
            {"topic_corrections": [{
                "topic_id": "birth_mode",
                "state": "answered",
                "value": "operation",
                "details": {"delivery_type": "c_section", "place": "hospital"},
                "instances": [{
                    "person_id": "child_1",
                    "value": True,
                    "details": {"delivery_type": "c_section", "place": "hospital"},
                }],
                "evidence": quote,
            }]},
            transcript,
        )
        details = kept["topics"]["birth_mode"]["details"]
        self.assertNotIn("delivery_type", details)
        self.assertEqual(details.get("place"), "hospital")
        inst = kept["topics"]["birth_mode"]["instances"][0]["details"]
        self.assertEqual(
            (details.get("count"), details.get("last_date"), inst.get("count"), inst.get("last_date")),
            (3, "jan", 3, "jan"),
        )
        unchanged = summarizer.reconcile_state_with_llm_payload(
            state,
            {"topic_corrections": [{
                "topic_id": "birth_mode",
                "state": "answered",
                "value": "c_section",
                "evidence": quote,
            }]},
            transcript,
        )
        self.assertEqual(unchanged["topics"]["birth_mode"]["value"], "operation")
        # A judge payload cannot reintroduce value=True alongside a normal delivery.
        repaired = summarizer.reconcile_state_with_llm_payload(
            state,
            {"topic_corrections": [{
                "topic_id": "birth_mode",
                "state": "answered",
                "value": True,
                "details": {"delivery_type": "normal"},
                "evidence": quote,
            }]},
            transcript,
        )
        self.assertIs(repaired["topics"]["birth_mode"]["value"], False)


if __name__ == "__main__":
    unittest.main()
