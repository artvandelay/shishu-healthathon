"""100% offline unit tests for Phase 2 Decisions API client, schema, and interpreter."""

from __future__ import annotations

import ast
import asyncio
import base64
import http.client
import io
import json
import os
import re
import tempfile
import time
import unittest
import urllib.error
import wave
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import call_cursor
import decisions
import test_v4_cursor as cursor_tests
from call_cursor import apply_interpreted_turn, apply_profile, apply_turn, new_cursor_state, routable_qids
from triage_catalog import FOLLOWUP_HI, get_spec

_SMOKE_TEST_FLOATS = (
    "0.8295",
    "0.9991",
    "0.9978",
    "0.9872",
    "0.9818",
    "0.9975",
)


def _test_thresholds(*, calibrated: bool = True) -> decisions.DecisionThresholds:
    return decisions.DecisionThresholds(
        calibrated=calibrated,
        danger_now_min_yes=0.80,
        caller_stop_min_yes=0.80,
        mentions_profile_min_yes=0.65,
        detail_present_min_yes=0.65,
        mentions_other_min_yes=0.70,
        outcome_min_confidence=0.65,
        outcome_min_catchall_margin=0.20,
        outcome_min_runnerup_margin=0.10,
        side_qid_min_confidence=0.70,
        side_qid_min_catchall_margin=0.25,
        side_qid_min_runnerup_margin=0.15,
        side_act_min_confidence=0.65,
        side_polarity_min_confidence=0.60,
        child_ref_min_confidence=0.65,
        child_ref_min_catchall_margin=0.20,
        child_ref_min_runnerup_margin=0.10,
    )


def _noul(prob: float) -> dict:
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1.0 - prob)}


def _choice(winner: str, probs: dict[str, float]) -> dict:
    return {
        "type": "choice",
        "choice": winner,
        "probabilities": probs,
        "confidence": probs[winner],
    }


class SchemaBuilderTests(unittest.TestCase):
    def test_referenced_qid_has_59_options_and_standalone_none_description(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        req = decisions.build_turn_request(state, utterance_text="नहीं")
        self.assertEqual(set(req.keys()), {"model", "input", "questions"})
        self.assertEqual(req["model"], "gemini-decision-1")
        self.assertIsInstance(req["input"], str)
        self.assertIn("active_qid: birth_jaundice", req["input"])
        self.assertIn("step_kind: main", req["input"])
        self.assertIn(FOLLOWUP_HI["jaundice_days"], req["input"])

        questions = req["questions"]
        for qkey in questions:
            self.assertRegex(qkey, r"^[A-Za-z0-9_-]{1,64}$")

        ref_criteria = questions["referenced_qid"]["criteria"]
        self.assertEqual(len(ref_criteria), 59)
        expected_keys = set(routable_qids()) | {"none_or_unclear"}
        self.assertEqual(set(ref_criteria.keys()), expected_keys)
        none_desc = ref_criteria["none_or_unclear"]
        self.assertNotIn("above", none_desc.lower())
        self.assertNotIn("last", none_desc.lower())
        self.assertIn("outside the active prompt", none_desc)
        self.assertIn("ever_pregnant", ref_criteria)
        self.assertIn(call_cursor._EVER_PREGNANT_HI, ref_criteria["ever_pregnant"])
        for qid in routable_qids():
            if qid != "ever_pregnant":
                spec = get_spec(qid)
                self.assertIn(call_cursor.question_hi_for(state, spec), ref_criteria[qid])
                self.assertIn(spec.need_en, ref_criteria[qid])

    def test_hold_ever_pregnant_and_close_more_prompts(self):
        held_name, hold = apply_profile(new_cursor_state(), caller_name="मेरा नाम")
        self.assertEqual(hold, "name_fragment")
        req_hold = decisions.build_turn_request(held_name, utterance_text="पूजा शर्मा")
        self.assertIn("step_kind: profile", req_hold["input"])
        self.assertIn(call_cursor._HOLD_HI["name_fragment"], req_hold["input"])

        childless = cursor_tests._profile(children=[])
        ever_state = apply_turn(childless, "clear")
        self.assertEqual(ever_state["active_qid"], "ever_pregnant")
        req_ever = decisions.build_turn_request(ever_state, utterance_text="नहीं")
        self.assertIn("step_kind: main", req_ever["input"])
        self.assertIn(call_cursor._EVER_PREGNANT_HI, req_ever["input"])
        self.assertNotIn("detail_present", req_ever["questions"])

        close = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        close_more = apply_turn(close, "concern", note="पेट दर्द")
        req_more = decisions.build_turn_request(close_more, utterance_text="नहीं")
        self.assertIn("step_kind: close", req_more["input"])
        self.assertIn(call_cursor._CLOSE_MORE_HI, req_more["input"])
        self.assertIn("detail_present", req_more["questions"])

    def test_conditional_child_ref_and_detail_present_across_step_kinds(self):
        # 0 children: no child_ref, birth_cord_3weeks skipped so go to preg_thalassemia (no followups)
        no_kids = cursor_tests._clear_until(
            cursor_tests._profile(marital="unmarried"),
            "preg_thalassemia",
        )
        req_no_kids = decisions.build_turn_request(no_kids, utterance_text="नहीं")
        self.assertNotIn("child_ref", req_no_kids["questions"])
        self.assertNotIn("detail_present", req_no_kids["questions"])
        self.assertIn("थैलेसीमिया", req_no_kids["input"])
        self.assertNotIn("पति", req_no_kids["input"])
        self.assertIn(
            "थैलेसीमिया",
            req_no_kids["questions"]["referenced_qid"]["criteria"]["preg_thalassemia"],
        )
        self.assertNotIn(
            "पति",
            req_no_kids["questions"]["referenced_qid"]["criteria"]["preg_thalassemia"],
        )

        # 1 child on gate without followups: no child_ref, no detail_present
        one_kid_no_fup = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_cord_3weeks",
        )
        req_one = decisions.build_turn_request(one_kid_no_fup, utterance_text="नहीं")
        self.assertNotIn("child_ref", req_one["questions"])
        self.assertNotIn("detail_present", req_one["questions"])

        # 1 child on gate WITH followups: no child_ref, HAS detail_present
        one_kid_fup = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        req_one_fup = decisions.build_turn_request(one_kid_fup, utterance_text="हाँ दो दिन")
        self.assertNotIn("child_ref", req_one_fup["questions"])
        self.assertIn("detail_present", req_one_fup["questions"])

        # 2+ children on main gate: HAS child_ref (oldest-first roster) and detail_present
        three_kids = [
            cursor_tests._child(2, "male", 1),
            cursor_tests._child(8, "female", 2, name="रिया"),
            cursor_tests._child(5, "male", 3, name="अर्जुन"),
        ]
        multi = cursor_tests._clear_until(
            cursor_tests._profile(children=three_kids),
            "birth_jaundice",
        )
        req_multi = decisions.build_turn_request(multi, utterance_text="बीच वाले को")
        self.assertIn("child_ref", req_multi["questions"])
        self.assertIn("detail_present", req_multi["questions"])
        child_crit = req_multi["questions"]["child_ref"]["criteria"]
        self.assertEqual(
            set(child_crit.keys()),
            {"child_1", "child_2", "child_3", "all", "none_or_unclear"},
        )
        self.assertNotIn("above", child_crit["none_or_unclear"].lower())
        self.assertNotIn("last", child_crit["none_or_unclear"].lower())
        self.assertIn("children_roster:", req_multi["input"])
        self.assertLess(
            req_multi["input"].index("child_2 | 8 साल"),
            req_multi["input"].index("child_3 | 5 साल"),
        )
        self.assertLess(
            req_multi["input"].index("child_3 | 5 साल"),
            req_multi["input"].index("child_1 | 2 साल"),
        )

        # which_child sub-step: HAS child_ref, NO detail_present
        which = apply_turn(multi, "concern")
        self.assertEqual(which["pending_followup_idx"], -1)
        req_which = decisions.build_turn_request(which, utterance_text="अर्जुन")
        self.assertIn("step_kind: which_child", req_which["input"])
        self.assertIn("child_ref", req_which["questions"])
        self.assertNotIn("detail_present", req_which["questions"])
        self.assertIn("यह किस बच्चे की बात है", req_which["input"])

        # followup sub-step: HAS detail_present
        fup = apply_turn(which, "concern", child_ref="child_3")
        self.assertEqual(fup["pending_followup_idx"], 0)
        req_fup = decisions.build_turn_request(fup, utterance_text="चार दिन")
        self.assertIn("step_kind: followup", req_fup["input"])
        self.assertIn("detail_present", req_fup["questions"])
        self.assertIn(FOLLOWUP_HI["jaundice_days"], req_fup["input"])

        # close_anything_else: HAS detail_present
        close = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        req_close = decisions.build_turn_request(close, utterance_text="हाँ")
        self.assertIn("step_kind: close", req_close["input"])
        self.assertIn("detail_present", req_close["questions"])

        # profile & bridge: NO detail_present
        req_profile = decisions.build_turn_request(new_cursor_state(), utterance_text="मेरा नाम पूजा")
        self.assertIn("step_kind: profile", req_profile["input"])
        self.assertNotIn("detail_present", req_profile["questions"])

        bridge = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        req_bridge = decisions.build_turn_request(bridge, utterance_text="हाँ शुरू करें")
        self.assertIn("step_kind: bridge", req_bridge["input"])
        self.assertNotIn("detail_present", req_bridge["questions"])

    def test_pcm16_to_wav_bytes_and_audio_input_blocks(self):
        pcm = (b"\x01\x00\xff\x7f" * 80) + b"\x00"  # odd trailing byte trimmed cleanly
        wav_bytes = decisions.pcm16_to_wav_bytes(pcm, sample_rate=16000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.readframes(wf.getnframes()), pcm[:-1])

        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        req = decisions.build_turn_request(
            state,
            utterance_text="नहीं",
            audio_wav_bytes=wav_bytes,
            audio_mime_type="audio/wav",
        )
        self.assertIsInstance(req["input"], list)
        self.assertEqual(len(req["input"]), 2)
        self.assertEqual(req["input"][0]["type"], "text")
        self.assertTrue(req["input"][0]["text"].endswith("\n"))
        self.assertEqual(req["input"][1]["type"], "audio")
        self.assertEqual(req["input"][1]["mime_type"], "audio/wav")
        self.assertEqual(base64.b64decode(req["input"][1]["data"]), wav_bytes)

    def test_static_instruction_and_criteria_have_no_devanagari_or_persona_examples(self):
        devanagari_re = re.compile(r"[\u0900-\u097F]")
        kids = [
            cursor_tests._child(2, "male", 1),
            cursor_tests._child(8, "female", 2, name="रिया"),
        ]
        main_state = cursor_tests._clear_until(
            cursor_tests._profile(children=kids),
            "birth_jaundice",
        )
        fup_state = apply_turn(main_state, "concern", child_ref="child_1")
        close_state = cursor_tests._clear_until(
            cursor_tests._profile(children=kids),
            "close_anything_else",
        )
        for state in (main_state, fup_state, close_state):
            req = decisions.build_turn_request(state, utterance_text="नहीं")
            for qkey, qobj in req["questions"].items():
                instr = str(qobj.get("instructions") or "")
                self.assertIsNone(devanagari_re.search(instr), (qkey, instr))
                self.assertNotIn("चाची", instr, qkey)
                criteria = qobj.get("criteria") or {}
                for opt_key, opt_desc in criteria.items():
                    if qkey == "referenced_qid" and opt_key != "none_or_unclear":
                        continue
                    if qkey == "child_ref" and opt_key not in ("all", "none_or_unclear"):
                        continue
                    desc = str(opt_desc or "")
                    self.assertIsNone(devanagari_re.search(desc), (qkey, opt_key, desc))
                    self.assertNotIn("चाची", desc, (qkey, opt_key))


class ThresholdAndGuardTests(unittest.TestCase):
    def test_no_smoke_test_float_constants_in_decisions_source(self):
        source = Path(decisions.__file__).read_text(encoding="utf-8")
        for forbidden in _SMOKE_TEST_FLOATS:
            self.assertNotIn(forbidden, source, forbidden)
        tree = ast.parse(source)
        found_floats = {
            repr(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        }
        for forbidden in _SMOKE_TEST_FLOATS:
            self.assertNotIn(forbidden, found_floats, forbidden)

    def test_uncalibrated_refused_for_authority_and_noul_bounds_validated(self):
        self.assertFalse(decisions.UNCALIBRATED_THRESHOLDS.calibrated)
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        answers = {
            "danger_now": _noul(0.02),
            "caller_stop": _noul(0.02),
            "mentions_profile": _noul(0.02),
            "mentions_other_item": _noul(0.02),
            "detail_present": _noul(0.02),
            "current_outcome": _choice("clear", {"clear": 0.95, "not_about_active": 0.03, "concern": 0.02}),
            "referenced_qid": _choice("none_or_unclear", {"none_or_unclear": 0.96, "fam_cancer": 0.02}),
            "referenced_act": _choice("not_a_side_action", {"not_a_side_action": 0.96, "add_or_correct": 0.02}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.96, "concern": 0.02}),
        }

        with self.assertRaises(ValueError):
            decisions.validate_thresholds(decisions.UNCALIBRATED_THRESHOLDS)
        with self.assertRaises(ValueError):
            decisions.interpret_for_authority(state, answers, utterance_text="नहीं")
        with self.assertRaises(ValueError):
            decisions.interpret_turn_decision(
                state,
                answers,
                utterance_text="नहीं",
                thresholds=decisions.UNCALIBRATED_THRESHOLDS,
                require_calibrated=True,
            )

        # Out-of-bounds noul threshold when calibrated=True must fail validation
        bad_low = decisions.DecisionThresholds(calibrated=True, danger_now_min_yes=0.01)
        bad_high = decisions.DecisionThresholds(calibrated=True, danger_now_min_yes=0.99)
        for bad in (bad_low, bad_high):
            with self.assertRaises(ValueError):
                decisions.validate_thresholds(bad)

        # Uncalibrated thresholds must use explicit 1.0 sentinels (never guess floats)
        self.assertFalse(decisions.UNCALIBRATED_THRESHOLDS.calibrated)
        for field in decisions._ALL_NUMERIC_THRESHOLD_FIELDS:
            self.assertEqual(getattr(decisions.UNCALIBRATED_THRESHOLDS, field), 1.0, field)

        shadow_answers = {
            "danger_now": _noul(0.02),
            "caller_stop": _noul(0.02),
            "mentions_profile": _noul(0.03),
            "mentions_other_item": _noul(0.95),
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.92, "not_about_active": 0.05, "concern": 0.03}),
            "referenced_qid": _choice("fam_cancer", {"fam_cancer": 0.91, "none_or_unclear": 0.05, "fam_lumps": 0.04}),
            "referenced_act": _choice("add_or_correct", {"add_or_correct": 0.93, "not_a_side_action": 0.05, "clarify_meaning": 0.02}),
            "referenced_polarity": _choice("concern", {"concern": 0.94, "not_applicable_or_none": 0.06}),
        }
        shadow_interp = decisions.interpret_turn_decision(
            state,
            shadow_answers,
            utterance_text="नहीं",
            thresholds=decisions.UNCALIBRATED_THRESHOLDS,
            require_calibrated=False,
        )
        self.assertEqual(shadow_interp.cursor_payload.get("side_qid", ""), "")
        self.assertIsNone(shadow_interp.future_qid_match)
        self.assertEqual(shadow_interp.diagnostics["current_outcome"]["winner"], "")
        self.assertEqual(shadow_interp.diagnostics["current_outcome"]["top_option"], "clear")
        self.assertAlmostEqual(shadow_interp.diagnostics["current_outcome"]["top_prob"], 0.92)
        self.assertAlmostEqual(shadow_interp.diagnostics["current_outcome"]["catchall_margin"], 0.87)
        self.assertAlmostEqual(shadow_interp.diagnostics["current_outcome"]["runnerup_margin"], 0.89)
        self.assertEqual(shadow_interp.diagnostics["mentions_other_item"], 0.95)
        self.assertEqual(
            shadow_interp.diagnostics["raw_referenced_qid_candidate"],
            {
                "qid": "fam_cancer",
                "confidence": 0.91,
                "catchall_margin": 0.86,
                "runnerup_margin": 0.87,
                "mentions_other_prob": 0.95,
                "act": "add_or_correct",
                "polarity": "concern",
            },
        )

        # load_thresholds from JSON artifact works and validates
        valid = _test_thresholds(calibrated=True)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "thresholds.json"
            artifact.write_text(json.dumps(asdict(valid)), encoding="utf-8")
            loaded = decisions.load_thresholds(artifact)
            self.assertEqual(loaded, valid)
            res = decisions.interpret_for_authority(
                state, answers, utterance_text="नहीं", thresholds=loaded
            )
            self.assertEqual(res.cursor_payload["current_outcome"], "clear")

            bad_artifact = Path(tmp) / "bad.json"
            bad_data = asdict(valid)
            bad_data["mentions_other_min_yes"] = 0.995
            bad_artifact.write_text(json.dumps(bad_data), encoding="utf-8")
            with self.assertRaises(ValueError):
                decisions.load_thresholds(bad_artifact)


class InterpreterEndToEndTests(unittest.TestCase):
    def test_smoke_a_mixed_clear_and_earlier_fam_cancer_addition(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "fam_cancer",
        )
        state = apply_turn(state, "concern", note="मामा को ब्लड कैंसर")
        state = apply_turn(state, "concern", note="मामा")
        state = cursor_tests._clear_until(state, "fam_heart_kidney")

        answers = {
            "danger_now": _noul(0.02),
            "caller_stop": _noul(0.02),
            "mentions_profile": _noul(0.03),
            "mentions_other_item": _noul(0.95),
            "detail_present": _noul(0.05),
            "current_outcome": _choice(
                "clear",
                {"clear": 0.92, "not_about_active": 0.04, "concern": 0.03, "unknown": 0.01},
            ),
            "referenced_qid": _choice(
                "fam_cancer",
                {"fam_cancer": 0.91, "none_or_unclear": 0.04, "fam_lumps": 0.03, "fam_heart_kidney": 0.02},
            ),
            "referenced_act": _choice(
                "add_or_correct",
                {"add_or_correct": 0.93, "not_a_side_action": 0.05, "clarify_meaning": 0.02},
            ),
            "referenced_polarity": _choice(
                "concern",
                {"concern": 0.94, "clear": 0.02, "retract": 0.02, "not_applicable_or_none": 0.02},
            ),
        }
        utterance = "नहीं, पर चाची को भी कैंसर हुआ था"
        interp = decisions.interpret_turn_decision(
            state, answers, utterance_text=utterance, thresholds=thresholds
        )
        self.assertIsNone(interp.future_qid_match)
        self.assertFalse(interp.needs_profile_extraction)
        self.assertEqual(interp.cursor_payload["current_outcome"], "clear")
        self.assertEqual(interp.cursor_payload["side_qid"], "fam_cancer")
        self.assertEqual(interp.cursor_payload["side_action"], "add_or_correct")
        self.assertEqual(interp.cursor_payload["side_outcome"], "concern")

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["answers"]["fam_heart_kidney"]["state"], "clear")
        self.assertEqual(updated["answers"]["fam_heart_kidney"]["notes"], "")
        self.assertIn("चाची को भी कैंसर हुआ था", updated["answers"]["fam_cancer"]["notes"])
        self.assertIn("मामा को ब्लड कैंसर", updated["answers"]["fam_cancer"]["notes"])
        self.assertEqual(updated["active_qid"], "fam_bp_stroke")

    def test_side_retract_clears_earlier_concern(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "fam_cancer",
        )
        state = apply_turn(state, "concern", note="शायद कैंसर")
        state = apply_turn(state, "concern", note="मामा")
        state = cursor_tests._clear_until(state, "fam_heart_kidney")

        answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.02),
            "mentions_other_item": _noul(0.93),
            "detail_present": _noul(0.04),
            "current_outcome": _choice(
                "not_about_active",
                {"not_about_active": 0.88, "clear": 0.08, "concern": 0.04},
            ),
            "referenced_qid": _choice(
                "fam_cancer",
                {"fam_cancer": 0.92, "none_or_unclear": 0.05, "fam_lumps": 0.03},
            ),
            "referenced_act": _choice(
                "add_or_correct",
                {"add_or_correct": 0.95, "not_a_side_action": 0.03, "clarify_meaning": 0.02},
            ),
            "referenced_polarity": _choice(
                "retract",
                {"retract": 0.86, "clear": 0.10, "concern": 0.02, "not_applicable_or_none": 0.02},
            ),
        }
        interp = decisions.interpret_turn_decision(
            state,
            answers,
            utterance_text="पहले वाले सवाल में गलती से हाँ बोल दिया था, कैंसर किसी को नहीं है",
            thresholds=thresholds,
        )
        self.assertEqual(interp.cursor_payload["side_qid"], "fam_cancer")
        self.assertEqual(interp.cursor_payload["side_outcome"], "retract")
        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["answers"]["fam_cancer"]["state"], "clear")
        self.assertEqual(updated["active_qid"], "fam_heart_kidney")

        # Real-API regression: side retract must not retract an unanswered active gate
        leaked_retract = dict(
            answers,
            current_outcome=_choice("retract", {"retract": 0.82, "clear": 0.12, "not_about_active": 0.06}),
        )
        interp_leaked = decisions.interpret_turn_decision(
            state,
            leaked_retract,
            utterance_text="पहले वाले सवाल में गलती से हाँ बोल दिया था, कैंसर किसी को नहीं है",
            thresholds=thresholds,
        )
        self.assertEqual(interp_leaked.cursor_payload["current_outcome"], "not_about_active")
        updated_leaked = apply_interpreted_turn(state, interp_leaked.cursor_payload)
        self.assertEqual(updated_leaked["answers"]["fam_cancer"]["state"], "clear")
        self.assertEqual(updated_leaked["active_qid"], "fam_heart_kidney")
        self.assertNotIn("fam_heart_kidney", updated_leaked["answers"])

    def test_future_qid_match_is_not_applied_as_side_qid(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        answers = {
            "danger_now": _noul(0.02),
            "caller_stop": _noul(0.02),
            "mentions_profile": _noul(0.02),
            "mentions_other_item": _noul(0.92),
            "detail_present": _noul(0.05),
            "current_outcome": _choice(
                "clear",
                {"clear": 0.90, "not_about_active": 0.06, "concern": 0.04},
            ),
            "referenced_qid": _choice(
                "fam_cancer",
                {"fam_cancer": 0.93, "none_or_unclear": 0.04, "fam_lumps": 0.03},
            ),
            "referenced_act": _choice(
                "add_or_correct",
                {"add_or_correct": 0.91, "not_a_side_action": 0.06, "clarify_meaning": 0.03},
            ),
            "referenced_polarity": _choice(
                "concern",
                {"concern": 0.92, "clear": 0.04, "not_applicable_or_none": 0.04},
            ),
        }
        interp = decisions.interpret_turn_decision(
            state,
            answers,
            utterance_text="पीलिया तो नहीं हुआ, हाँ चाची को कैंसर हुआ था",
            thresholds=thresholds,
        )
        self.assertEqual(interp.cursor_payload.get("side_qid", ""), "")
        self.assertEqual(interp.cursor_payload["current_outcome"], "clear")
        self.assertIsNotNone(interp.future_qid_match)
        self.assertEqual(interp.future_qid_match["qid"], "fam_cancer")
        self.assertEqual(interp.future_qid_match["act"], "add_or_correct")
        self.assertAlmostEqual(interp.future_qid_match["confidence"], 0.93)

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["answers"]["birth_jaundice"]["state"], "clear")
        self.assertNotIn("fam_cancer", updated["answers"])
        self.assertEqual(updated["active_qid"], "birth_sudden")

    def test_multichild_middle_son_selection_and_detail_present(self):
        thresholds = _test_thresholds()
        kids = [
            cursor_tests._child(9, "female", 1, name="रिया"),
            cursor_tests._child(5, "male", 2, name="अर्जुन"),
            cursor_tests._child(2, "female", 3, name="मीरा"),
        ]
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=kids),
            "birth_jaundice",
        )
        utterance = "हाँ बीच वाले बेटे को तीन दिन नीली लाइट में रखा था"
        answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.03),
            "mentions_other_item": _noul(0.03),
            "detail_present": _noul(0.94),
            "current_outcome": _choice(
                "concern",
                {"concern": 0.96, "not_about_active": 0.02, "clear": 0.02},
            ),
            "child_ref": _choice(
                "child_2",
                {"child_2": 0.91, "none_or_unclear": 0.04, "child_1": 0.03, "child_3": 0.01, "all": 0.01},
            ),
            "referenced_qid": _choice("none_or_unclear", {"none_or_unclear": 0.95, "birth_nicu": 0.05}),
            "referenced_act": _choice("not_a_side_action", {"not_a_side_action": 0.96, "add_or_correct": 0.04}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.96, "concern": 0.04}),
        }
        interp = decisions.interpret_turn_decision(
            state, answers, utterance_text=utterance, thresholds=thresholds
        )
        self.assertEqual(interp.cursor_payload["child_ref"], "child_2")
        self.assertTrue(interp.cursor_payload["followup_answered"])
        self.assertEqual(interp.cursor_payload["note"], utterance)

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["answers"]["birth_jaundice"]["state"], "concern")
        self.assertEqual(updated["answers"]["birth_jaundice"]["child_ids"], ["child_2"])
        self.assertEqual(
            updated["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"],
            utterance,
        )
        self.assertIsNone(updated["pending_followup_idx"])
        self.assertEqual(updated["active_qid"], "birth_sudden")

    def test_bare_yes_detail_present_false_asks_followup(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.02),
            "mentions_other_item": _noul(0.02),
            "detail_present": _noul(0.08),
            "current_outcome": _choice(
                "concern",
                {"concern": 0.97, "not_about_active": 0.02, "clear": 0.01},
            ),
            "referenced_qid": _choice("none_or_unclear", {"none_or_unclear": 0.96, "birth_nicu": 0.04}),
            "referenced_act": _choice("not_a_side_action", {"not_a_side_action": 0.97, "add_or_correct": 0.03}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.97, "concern": 0.03}),
        }
        interp = decisions.interpret_turn_decision(
            state, answers, utterance_text="हाँ", thresholds=thresholds
        )
        self.assertEqual(interp.cursor_payload["current_outcome"], "concern")
        self.assertFalse(interp.cursor_payload["followup_answered"])
        self.assertEqual(interp.cursor_payload["note"], "")

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["active_qid"], "birth_jaundice")
        self.assertEqual(updated["pending_followup_idx"], 0)

    def test_active_emergency_overrides_everything(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "fam_cancer",
        )
        state = apply_turn(state, "concern", note="मामा को कैंसर")
        state = apply_turn(state, "concern", note="मामा")
        state = cursor_tests._clear_until(state, "fam_heart_kidney")
        before_answers = json.loads(json.dumps(state["answers"]))

        answers = {
            "danger_now": _noul(0.95),
            "caller_stop": _noul(0.91),
            "mentions_profile": _noul(0.85),
            "mentions_other_item": _noul(0.95),
            "detail_present": _noul(0.90),
            "current_outcome": _choice("clear", {"clear": 0.95, "not_about_active": 0.03, "concern": 0.02}),
            "referenced_qid": _choice("fam_cancer", {"fam_cancer": 0.95, "none_or_unclear": 0.03}),
            "referenced_act": _choice("add_or_correct", {"add_or_correct": 0.95, "not_a_side_action": 0.05}),
            "referenced_polarity": _choice("clear", {"clear": 0.95, "not_applicable_or_none": 0.05}),
        }
        interp = decisions.interpret_turn_decision(
            state,
            answers,
            utterance_text="अभी बच्चे को सांस नहीं आ रही है",
            thresholds=thresholds,
        )
        self.assertTrue(interp.cursor_payload["emergency"])
        self.assertEqual(interp.cursor_payload.get("side_qid", ""), "")
        self.assertIsNone(interp.future_qid_match)

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["phase"], "closing")
        self.assertEqual(updated["close_reason"], "emergency")
        self.assertEqual(updated["emergency_note"], "अभी बच्चे को सांस नहीं आ रही है")
        self.assertEqual(updated["answers"], before_answers)

    def test_side_qid_rejected_on_low_margin_close_runnerup_or_failed_scope(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "fam_cancer",
        )
        state = apply_turn(state, "clear")
        self.assertEqual(state["active_qid"], "fam_heart_kidney")

        base = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.01),
            "mentions_other_item": _noul(0.92),
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.94, "not_about_active": 0.04, "concern": 0.02}),
            "referenced_qid": _choice(
                "fam_cancer",
                {"fam_cancer": 0.85, "none_or_unclear": 0.10, "fam_lumps": 0.05},
            ),
            "referenced_act": _choice("add_or_correct", {"add_or_correct": 0.90, "not_a_side_action": 0.10}),
            "referenced_polarity": _choice("concern", {"concern": 0.90, "not_applicable_or_none": 0.10}),
        }

        # 1. Failed mentions_other_item scope check -> rejects side_qid
        failed_scope = dict(base, mentions_other_item=_noul(0.60))
        res_scope = decisions.interpret_turn_decision(
            state, failed_scope, utterance_text="नहीं", thresholds=thresholds
        )
        self.assertEqual(res_scope.cursor_payload.get("side_qid", ""), "")
        self.assertIsNone(res_scope.future_qid_match)

        # 2. Close runner-up margin (< 0.15) -> rejects side_qid
        close_runnerup = dict(
            base,
            referenced_qid=_choice(
                "fam_cancer",
                {"fam_cancer": 0.72, "fam_lumps": 0.62, "none_or_unclear": 0.04},
            ),
        )
        res_runner = decisions.interpret_turn_decision(
            state, close_runnerup, utterance_text="नहीं", thresholds=thresholds
        )
        self.assertEqual(res_runner.cursor_payload.get("side_qid", ""), "")
        self.assertEqual(res_runner.diagnostics["referenced_qid"]["reject_reason"], "below_runnerup_margin")

        # 3. Low catchall margin (< 0.25) -> rejects side_qid
        low_catchall = dict(
            base,
            referenced_qid=_choice(
                "fam_cancer",
                {"fam_cancer": 0.72, "none_or_unclear": 0.55, "fam_lumps": 0.05},
            ),
        )
        res_catchall = decisions.interpret_turn_decision(
            state, low_catchall, utterance_text="नहीं", thresholds=thresholds
        )
        self.assertEqual(res_catchall.cursor_payload.get("side_qid", ""), "")
        self.assertEqual(res_catchall.diagnostics["referenced_qid"]["reject_reason"], "below_catchall_margin")

    def test_needs_profile_extraction_rules(self):
        thresholds = _test_thresholds()
        intro = new_cursor_state()
        self.assertEqual(decisions.turn_view(intro)[0], "intro_name")
        plain_answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.05),
            "current_outcome": _choice("not_about_active", {"not_about_active": 0.90, "clear": 0.10}),
            "referenced_qid": _choice("none_or_unclear", {"none_or_unclear": 0.95, "fam_cancer": 0.05}),
            "referenced_act": _choice("not_a_side_action", {"not_a_side_action": 0.95, "add_or_correct": 0.05}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.95, "concern": 0.05}),
        }
        # True on profile step when not clarify even if mentions_profile score is low
        self.assertTrue(
            decisions.interpret_turn_decision(intro, plain_answers, thresholds=thresholds).needs_profile_extraction
        )

        # False on profile step when resolved_outcome == clarify and mentions_profile is low
        clarify_answers = dict(
            plain_answers,
            current_outcome=_choice("clarify", {"clarify": 0.92, "not_about_active": 0.05, "clear": 0.03}),
        )
        self.assertFalse(
            decisions.interpret_turn_decision(intro, clarify_answers, thresholds=thresholds).needs_profile_extraction
        )
        married_state = apply_profile(intro, caller_name="पूजा")[0]
        self.assertEqual(married_state["active_qid"], "intro_married")
        self.assertFalse(
            decisions.interpret_turn_decision(married_state, clarify_answers, thresholds=thresholds).needs_profile_extraction
        )
        # True on clarify if caller explicitly mentions profile entities
        clarify_with_profile = dict(clarify_answers, mentions_profile=_noul(0.90))
        self.assertTrue(
            decisions.interpret_turn_decision(intro, clarify_with_profile, thresholds=thresholds).needs_profile_extraction
        )

        # False on ever_pregnant when mentions_profile score is low; True only when mentions_profile >= threshold
        childless = cursor_tests._profile(children=[])
        ever_state = apply_turn(childless, "clear")
        self.assertEqual(ever_state["active_qid"], "ever_pregnant")
        self.assertFalse(
            decisions.interpret_turn_decision(ever_state, plain_answers, thresholds=thresholds).needs_profile_extraction
        )
        ever_with_profile = dict(plain_answers, mentions_profile=_noul(0.90))
        self.assertTrue(
            decisions.interpret_turn_decision(ever_state, ever_with_profile, thresholds=thresholds).needs_profile_extraction
        )

        # True on mid-call gate when mentions_profile >= threshold, False when below
        gate_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        self.assertFalse(
            decisions.interpret_turn_decision(gate_state, plain_answers, thresholds=thresholds).needs_profile_extraction
        )
        with_profile = dict(plain_answers, mentions_profile=_noul(0.90))
        self.assertTrue(
            decisions.interpret_turn_decision(gate_state, with_profile, thresholds=thresholds).needs_profile_extraction
        )


class TransportRetryTests(unittest.TestCase):
    def _sample_request(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        return decisions.build_turn_request(state, utterance_text="नहीं")

    def test_retries_once_on_timeout_503_and_429_then_succeeds(self):
        req = self._sample_request()
        ok_body = json.dumps({
            "answers": {"danger_now": _noul(0.02)},
            "usage": {"total_input_tokens": 310, "total_output_tokens": 0},
        }).encode("utf-8")

        for first_outcome in (
            TimeoutError("read timed out"),
            (503, b'{"error":{"message":"unavailable"}}'),
            (429, b'{"error":{"message":"rate limit"}}'),
        ):
            calls = []

            def fake_transport(url, body_bytes, headers, timeout_sec, first=first_outcome):
                calls.append((url, len(body_bytes), headers, timeout_sec))
                if len(calls) == 1:
                    if isinstance(first, BaseException):
                        raise first
                    return first
                return 200, ok_body

            res = asyncio.run(
                decisions.evaluate_turn(req, api_key="test-key", transport=fake_transport)
            )
            self.assertTrue(res.ok, first_outcome)
            self.assertEqual(res.error_reason, "")
            self.assertEqual(res.attempts, 2)
            self.assertEqual(res.question_count, len(req["questions"]))
            self.assertEqual(res.catalog_option_count, 59)
            self.assertGreater(res.request_bytes, 0)
            self.assertTrue(res.usage_present)
            self.assertEqual(res.total_input_tokens, 310)
            self.assertEqual(res.raw_answers["danger_now"]["noul"], 0.02)
            self.assertEqual(calls[0][2]["x-goog-api-key"], "test-key")

    def test_fails_after_two_timeouts(self):
        req = self._sample_request()
        calls = 0

        def always_timeout(url, body_bytes, headers, timeout_sec):
            nonlocal calls
            calls += 1
            raise TimeoutError("timed out")

        res = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=always_timeout)
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(calls, 2)
        self.assertEqual(res.error_reason, "timeout")
        self.assertEqual(res.raw_answers, {})

    def test_no_retry_on_http_400_or_safety_block_or_empty_answers(self):
        req = self._sample_request()

        # 1. HTTP 400
        calls_400 = 0

        def bad_req(url, body_bytes, headers, timeout_sec):
            nonlocal calls_400
            calls_400 += 1
            return 400, b'{"error":{"code":"invalid_request","message":"bad field"}}'

        res_400 = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=bad_req)
        )
        self.assertFalse(res_400.ok)
        self.assertEqual(res_400.attempts, 1)
        self.assertEqual(calls_400, 1)
        self.assertEqual(res_400.error_reason, "http_400")

        # 2. HTTP 200 safety block (snake_case & camelCase)
        for payload in (
            {"prompt_feedback": {"block_reason": "safety"}, "answers": {}},
            {"promptFeedback": {"blockReason": "SAFETY"}},
        ):
            calls_sb = 0

            def safety_fn(url, body_bytes, headers, timeout_sec, body=payload):
                nonlocal calls_sb
                calls_sb += 1
                return 200, json.dumps(body).encode("utf-8")

            res_sb = asyncio.run(
                decisions.evaluate_turn(req, api_key="test-key", transport=safety_fn)
            )
            self.assertFalse(res_sb.ok)
            self.assertEqual(res_sb.attempts, 1)
            self.assertEqual(calls_sb, 1)
            self.assertEqual(res_sb.error_reason, "safety_block")

        # 3. Missing / empty answers without safety block
        calls_empty = 0

        def empty_fn(url, body_bytes, headers, timeout_sec):
            nonlocal calls_empty
            calls_empty += 1
            return 200, b'{"answers": {}, "usage": null}'

        res_empty = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=empty_fn)
        )
        self.assertFalse(res_empty.ok)
        self.assertEqual(res_empty.attempts, 1)
        self.assertEqual(res_empty.error_reason, "empty_answers")
        self.assertFalse(res_empty.usage_present)
        self.assertIsNone(res_empty.total_input_tokens)

        # 4. Malformed JSON response
        calls_bad = 0

        def malformed_fn(url, body_bytes, headers, timeout_sec):
            nonlocal calls_bad
            calls_bad += 1
            return 200, b"{not json"

        res_bad = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=malformed_fn)
        )
        self.assertFalse(res_bad.ok)
        self.assertEqual(res_bad.attempts, 1)
        self.assertEqual(res_bad.error_reason, "malformed_response")

    def test_usage_null_and_usage_metadata_variants(self):
        req = self._sample_request()

        def null_usage(url, body_bytes, headers, timeout_sec):
            return 200, json.dumps({
                "answers": {"danger_now": _noul(0.01)},
                "usage": None,
            }).encode("utf-8")

        res_null = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=null_usage)
        )
        self.assertTrue(res_null.ok)
        self.assertFalse(res_null.usage_present)
        self.assertIsNone(res_null.total_input_tokens)

        def camel_usage(url, body_bytes, headers, timeout_sec):
            return 200, json.dumps({
                "answers": {"danger_now": _noul(0.01)},
                "usageMetadata": {"promptTokenCount": 425, "totalTokenCount": 425},
            }).encode("utf-8")

        res_camel = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=camel_usage)
        )
        self.assertTrue(res_camel.ok)
        self.assertTrue(res_camel.usage_present)
        self.assertEqual(res_camel.total_input_tokens, 425)

    def test_missing_api_key_returns_immediately_without_socket_call(self):
        req = self._sample_request()
        with mock.patch.dict(os.environ, {}, clear=True):
            res = asyncio.run(decisions.evaluate_turn(req))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_reason, "missing_api_key")
        self.assertEqual(res.attempts, 0)
        self.assertEqual(res.raw_answers, {})

    def test_urllib_http_error_exception_from_transport_handled_cleanly(self):
        req = self._sample_request()
        calls = 0

        def raise_http_503(url, body_bytes, headers, timeout_sec):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.HTTPError(
                    url, 503, "Service Unavailable", {}, io.BytesIO(b"busy")
                )
            return 200, json.dumps({"answers": {"danger_now": _noul(0.01)}}).encode("utf-8")

        res = asyncio.run(
            decisions.evaluate_turn(req, api_key="test-key", transport=raise_http_503)
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.attempts, 2)

    def test_retries_on_http_408_connection_error_urlerror_and_respects_budget(self):
        req = self._sample_request()
        ok_body = json.dumps({"answers": {"danger_now": _noul(0.01)}}).encode("utf-8")

        for first_exc in (
            (408, b"request timeout"),
            ConnectionResetError("peer reset"),
            urllib.error.URLError("dns error"),
            http.client.RemoteDisconnected("closed"),
        ):
            calls = 0

            def retry_fn(url, body_bytes, headers, timeout_sec, err=first_exc):
                nonlocal calls
                calls += 1
                if calls == 1:
                    if isinstance(err, BaseException):
                        raise err
                    return err
                return 200, ok_body

            res = asyncio.run(
                decisions.evaluate_turn(req, api_key="test-key", transport=retry_fn)
            )
            self.assertTrue(res.ok, first_exc)
            self.assertEqual(res.attempts, 2)

        def slow_fn(url, body_bytes, headers, timeout_sec):
            time.sleep(0.15)
            return 200, ok_body

        started = time.perf_counter()
        res_budget = asyncio.run(
            decisions.evaluate_turn(
                req,
                api_key="test-key",
                attempt_timeout_sec=0.04,
                total_budget_sec=0.09,
                transport=slow_fn,
            )
        )
        elapsed = time.perf_counter() - started
        self.assertFalse(res_budget.ok)
        self.assertEqual(res_budget.error_reason, "timeout")
        self.assertLessEqual(res_budget.attempts, 2)
        self.assertLess(elapsed, 0.5)

    def test_default_urllib_transport_mocked_urlopen(self):
        req = self._sample_request()
        captured = {}

        class _FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "answers": {"danger_now": _noul(0.01)},
                    "usage": None,
                }).encode("utf-8")

        def fake_urlopen(request_obj, timeout=None):
            captured["url"] = request_obj.full_url
            captured["method"] = request_obj.get_method()
            captured["headers"] = dict(request_obj.header_items())
            captured["body"] = json.loads(request_obj.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeResp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = asyncio.run(decisions.evaluate_turn(req, api_key="live-test-key"))

        self.assertTrue(res.ok)
        self.assertEqual(res.attempts, 1)
        self.assertEqual(captured["url"], decisions.DECISIONS_ENDPOINT)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["headers"]["X-goog-api-key"], "live-test-key")
        self.assertEqual(captured["body"]["model"], decisions.DECISIONS_MODEL)
        self.assertLessEqual(captured["timeout"], 2.0)

    def test_clarify_meaning_caller_stop_and_close_bare_vs_noted(self):
        thresholds = _test_thresholds()
        thal = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "preg_thalassemia",
        )
        thal = apply_turn(thal, "unknown")
        self.assertEqual(thal["active_qid"], "fam_related")

        clarify_answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.01),
            "mentions_other_item": _noul(0.94),
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.91, "not_about_active": 0.05, "concern": 0.04}),
            "referenced_qid": _choice("preg_thalassemia", {"preg_thalassemia": 0.90, "none_or_unclear": 0.05, "fam_related": 0.05}),
            "referenced_act": _choice("clarify_meaning", {"clarify_meaning": 0.91, "add_or_correct": 0.05, "not_a_side_action": 0.04}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.92, "concern": 0.08}),
        }
        res_clar = decisions.interpret_turn_decision(
            thal,
            clarify_answers,
            utterance_text="रिश्तेदार तो नहीं हैं, पर थैलेसीमिया क्या होता है?",
            thresholds=thresholds,
        )
        self.assertEqual(res_clar.cursor_payload["side_qid"], "preg_thalassemia")
        self.assertEqual(res_clar.cursor_payload["side_action"], "clarify_meaning")
        rewound = apply_interpreted_turn(thal, res_clar.cursor_payload)
        self.assertEqual(rewound["active_qid"], "preg_thalassemia")
        self.assertEqual(rewound["answers"]["fam_related"]["state"], "clear")

        stop_answers = dict(
            clarify_answers,
            caller_stop=_noul(0.92),
            mentions_other_item=_noul(0.02),
        )
        res_stop = decisions.interpret_turn_decision(
            thal,
            stop_answers,
            utterance_text="अभी फोन बंद कर दो, बाद में बात करेंगे",
            thresholds=thresholds,
        )
        self.assertTrue(res_stop.cursor_payload["caller_stop"])
        stopped = apply_interpreted_turn(thal, res_stop.cursor_payload)
        self.assertEqual(stopped["phase"], "closing")
        self.assertEqual(stopped["close_reason"], "caller_stop")

        close_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        bare_close = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.01),
            "mentions_other_item": _noul(0.01),
            "detail_present": _noul(0.05),
            "current_outcome": _choice("concern", {"concern": 0.95, "not_about_active": 0.03, "clear": 0.02}),
            "referenced_qid": _choice("none_or_unclear", {"none_or_unclear": 0.96, "fam_cancer": 0.04}),
            "referenced_act": _choice("not_a_side_action", {"not_a_side_action": 0.96, "add_or_correct": 0.04}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.96, "concern": 0.04}),
        }
        res_bare_close = decisions.interpret_turn_decision(
            close_state, bare_close, utterance_text="हाँ", thresholds=thresholds
        )
        asked_detail = apply_interpreted_turn(close_state, res_bare_close.cursor_payload)
        self.assertEqual(asked_detail["pending_followup_idx"], 0)
        self.assertEqual(asked_detail["phase"], "asking")

        noted_close = dict(bare_close, detail_present=_noul(0.91))
        res_noted_close = decisions.interpret_turn_decision(
            asked_detail,
            noted_close,
            utterance_text="रात में नींद कम आती है",
            thresholds=thresholds,
        )
        saved_close = apply_interpreted_turn(asked_detail, res_noted_close.cursor_payload)
        self.assertIsNone(saved_close["pending_followup_idx"])
        self.assertEqual(saved_close["answers"]["close_anything_else"]["notes"], "रात में नींद कम आती है")

    def test_followup_step_saves_spoken_answer_even_when_detail_present_is_low(self):
        thresholds = _test_thresholds()
        nicu_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )
        nicu_state = apply_turn(nicu_state, "concern", note="सांस", followup_answered=True)
        self.assertEqual(nicu_state["active_qid"], "birth_nicu")
        self.assertEqual(nicu_state["pending_followup_idx"], 1)  # nicu_days

        low_detail_answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.01),
            "mentions_other_item": _noul(0.02),
            "detail_present": _noul(0.20),
            "current_outcome": _choice("concern", {"concern": 0.92, "not_about_active": 0.05, "clear": 0.03}),
            "referenced_qid": _choice("none_or_unclear", {"none_or_unclear": 0.95, "birth_jaundice": 0.05}),
            "referenced_act": _choice("not_a_side_action", {"not_a_side_action": 0.96, "add_or_correct": 0.04}),
            "referenced_polarity": _choice("not_applicable_or_none", {"not_applicable_or_none": 0.96, "concern": 0.04}),
        }
        interp_nicu = decisions.interpret_turn_decision(
            nicu_state,
            low_detail_answers,
            utterance_text="चार दिन",
            thresholds=thresholds,
        )
        self.assertEqual(interp_nicu.cursor_payload["note"], "चार दिन")
        updated_nicu = apply_interpreted_turn(nicu_state, interp_nicu.cursor_payload)
        self.assertEqual(
            updated_nicu["answers"]["birth_nicu"]["followups"]["nicu_days"],
            {"state": "concern", "note": "चार दिन"},
        )
        self.assertIsNone(updated_nicu["pending_followup_idx"])
        self.assertEqual(updated_nicu["active_qid"], "birth_jaundice")

        close_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        open_detail_state = apply_turn(close_state, "concern")
        self.assertEqual(open_detail_state["pending_followup_idx"], 0)
        interp_open = decisions.interpret_turn_decision(
            open_detail_state,
            low_detail_answers,
            utterance_text="नींद में डर लगता है",
            thresholds=thresholds,
        )
        self.assertEqual(interp_open.cursor_payload["note"], "नींद में डर लगता है")
        updated_open = apply_interpreted_turn(open_detail_state, interp_open.cursor_payload)
        self.assertIsNone(updated_open["pending_followup_idx"])
        self.assertEqual(
            updated_open["answers"]["close_anything_else"]["followups"]["open_concern_detail"]["note"],
            "नींद में डर लगता है",
        )
        self.assertIn("नींद में डर लगता है", updated_open["answers"]["close_anything_else"]["notes"])

    def test_provisional_staging_thresholds_load_and_default_authority_is_fail_closed(self):
        repo_root = Path(__file__).resolve().parents[2]
        provisional_path = repo_root / "scripts" / "provisional_staging_thresholds.json"
        default_prod_path = Path(__file__).resolve().parent / "decision_thresholds.json"
        self.assertFalse(default_prod_path.exists())
        loaded = decisions.load_thresholds(provisional_path, require_calibrated=True)
        self.assertTrue(loaded.calibrated)
        self.assertTrue(0.01 < loaded.danger_now_min_yes < 0.99)


if __name__ == "__main__":
    unittest.main()
