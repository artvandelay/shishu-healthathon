"""100% offline unit tests for Decisions API client, declarative schema, and interpreter."""

from __future__ import annotations

import ast
import asyncio
import base64
import http.client
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
import urllib.error
import wave
from dataclasses import asdict
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests import APP_DIR, REPO_ROOT  # noqa: E402

import audio_recorder  # noqa: E402
import call_cursor  # noqa: E402
import decisions  # noqa: E402
import test_cursor as cursor_tests  # noqa: E402
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
        outcome_min_confidence=0.65,
        outcome_min_margin=0.15,
        side_intent_min_confidence=0.65,
        side_intent_min_margin=0.15,
        side_qid_min_confidence=0.70,
        side_qid_min_margin=0.20,
        child_ref_min_confidence=0.65,
        child_ref_min_margin=0.15,
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


def _parsed_input(req: dict) -> dict:
    if isinstance(req["input"], list):
        return json.loads(req["input"][0]["text"])
    return json.loads(req["input"])


class SchemaBuilderTests(unittest.TestCase):
    def test_side_qid_has_59_options_and_json_input_fields(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        req = decisions.build_turn_request(state, utterance_text="नहीं")
        self.assertEqual(set(req.keys()), {"model", "input", "questions"})
        self.assertEqual(req["model"], "gemini-decision-1")
        self.assertIsInstance(req["input"], str)

        ctx = _parsed_input(req)
        self.assertEqual(
            set(ctx.keys()),
            {"active_question", "followup_question", "last_answered_question", "caller_utterance"},
        )
        self.assertEqual(ctx["active_question"], call_cursor.active_prompt_hi(state))
        self.assertEqual(ctx["followup_question"], FOLLOWUP_HI["jaundice_days"])
        self.assertEqual(ctx["last_answered_question"], call_cursor.last_answered_prompt_hi(state)[1])
        self.assertEqual(ctx["caller_utterance"], "नहीं")

        questions = req["questions"]
        for qkey in questions:
            self.assertRegex(qkey, r"^[A-Za-z0-9_-]{1,64}$")

        side_criteria = questions["side_qid"]["criteria"]
        self.assertEqual(len(side_criteria), 59)
        expected_keys = set(routable_qids()) | {"other"}
        self.assertEqual(set(side_criteria.keys()), expected_keys)
        other_desc = side_criteria["other"]
        self.assertNotIn("above", other_desc.lower())
        self.assertNotIn("last", other_desc.lower())
        self.assertIn("outside active_question", other_desc)
        self.assertIn("ever_pregnant", side_criteria)
        self.assertIn(call_cursor._EVER_PREGNANT_HI, side_criteria["ever_pregnant"])
        for qid in routable_qids():
            if qid != "ever_pregnant":
                spec = get_spec(qid)
                self.assertIn(call_cursor.question_hi_for(state, spec), side_criteria[qid])
                self.assertIn(spec.need_en, side_criteria[qid])

    def test_intro_omits_side_questions_and_optional_context_keys(self):
        # Intro profile & bridge turns without answered gates omit side_intent, side_qid, followup_question, last_answered_question
        req_profile = decisions.build_turn_request(new_cursor_state(), utterance_text="मेरा नाम पूजा")
        ctx_profile = _parsed_input(req_profile)
        self.assertEqual(set(ctx_profile.keys()), {"active_question", "caller_utterance"})
        self.assertEqual(
            set(req_profile["questions"].keys()),
            {"danger_now", "caller_stop", "mentions_profile", "current_outcome"},
        )

        bridge = cursor_tests._profile(children=[cursor_tests._child(4, "male")])
        req_bridge = decisions.build_turn_request(bridge, utterance_text="हाँ शुरू करें")
        ctx_bridge = _parsed_input(req_bridge)
        self.assertEqual(
            set(ctx_bridge.keys()),
            {"active_question", "last_answered_question", "caller_utterance"},
        )
        self.assertEqual(
            ctx_bridge["last_answered_question"],
            call_cursor.last_answered_prompt_hi(bridge)[1],
        )
        self.assertEqual(
            set(req_bridge["questions"].keys()),
            {"danger_now", "caller_stop", "mentions_profile", "current_outcome"},
        )

    def test_hold_ever_pregnant_and_close_more_prompts(self):
        held_name, hold = apply_profile(new_cursor_state(), caller_name="मेरा नाम")
        self.assertEqual(hold, "name_fragment")
        req_hold = decisions.build_turn_request(held_name, utterance_text="पूजा शर्मा")
        self.assertEqual(_parsed_input(req_hold)["active_question"], call_cursor._HOLD_HI["name_fragment"])

        childless = cursor_tests._profile(children=[])
        ever_state = apply_turn(childless, "clear")
        self.assertEqual(ever_state["active_qid"], "ever_pregnant")
        req_ever = decisions.build_turn_request(ever_state, utterance_text="नहीं")
        self.assertEqual(_parsed_input(req_ever)["active_question"], call_cursor._EVER_PREGNANT_HI)
        self.assertNotIn("detail_present", req_ever["questions"])
        self.assertIn("side_intent", req_ever["questions"])
        self.assertIn("side_qid", req_ever["questions"])

        close = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        close_more = apply_turn(close, "concern", note="पेट दर्द")
        req_more = decisions.build_turn_request(close_more, utterance_text="नहीं")
        self.assertEqual(_parsed_input(req_more)["active_question"], call_cursor._CLOSE_MORE_HI)
        self.assertIn("detail_present", req_more["questions"])

    def test_conditional_child_ref_and_detail_present_across_step_kinds(self):
        # 0 children on gate without followups: no child_ref, no detail_present
        no_kids = cursor_tests._clear_until(
            cursor_tests._profile(marital="unmarried"),
            "preg_thalassemia",
        )
        req_no_kids = decisions.build_turn_request(no_kids, utterance_text="नहीं")
        self.assertNotIn("child_ref", req_no_kids["questions"])
        self.assertNotIn("detail_present", req_no_kids["questions"])
        ctx_no_kids = _parsed_input(req_no_kids)
        self.assertNotIn("children_roster", ctx_no_kids)
        self.assertIn("थैलेसीमिया", ctx_no_kids["active_question"])
        self.assertNotIn("पति", ctx_no_kids["active_question"])
        self.assertIn("थैलेसीमिया", req_no_kids["questions"]["side_qid"]["criteria"]["preg_thalassemia"])
        self.assertNotIn("पति", req_no_kids["questions"]["side_qid"]["criteria"]["preg_thalassemia"])

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
        self.assertEqual(set(child_crit.keys()), {"child_1", "child_2", "child_3", "all", "other"})
        self.assertNotIn("above", child_crit["other"].lower())
        self.assertNotIn("last", child_crit["other"].lower())
        roster_out = _parsed_input(req_multi)["children_roster"]
        self.assertEqual([r["id"] for r in roster_out], ["child_2", "child_3", "child_1"])
        self.assertEqual([r["rank"] for r in roster_out], ["oldest", "middle", "youngest"])
        self.assertEqual(roster_out[0]["name"], "रिया")
        self.assertNotIn("name", roster_out[2])

        # which_child sub-step: HAS child_ref and parent_question, NO detail_present
        which = apply_turn(multi, "concern")
        self.assertEqual(which["pending_followup_idx"], -1)
        req_which = decisions.build_turn_request(which, utterance_text="अर्जुन")
        self.assertIn("child_ref", req_which["questions"])
        self.assertNotIn("detail_present", req_which["questions"])
        which_ctx = _parsed_input(req_which)
        self.assertIn("यह किस बच्चे की बात है", which_ctx["active_question"])
        self.assertIn("नीली लाइट", which_ctx["parent_question"])

        # followup sub-step: HAS parent_question, NO detail_present
        fup = apply_turn(which, "concern", child_ref="child_3")
        self.assertEqual(fup["pending_followup_idx"], 0)
        req_fup = decisions.build_turn_request(fup, utterance_text="चार दिन")
        self.assertNotIn("detail_present", req_fup["questions"])
        fup_ctx = _parsed_input(req_fup)
        self.assertEqual(fup_ctx["followup_question"], FOLLOWUP_HI["jaundice_days"])
        self.assertIn("नीली लाइट", fup_ctx["parent_question"])

        # close_anything_else: HAS detail_present
        close = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        req_close = decisions.build_turn_request(close, utterance_text="हाँ")
        self.assertIn("detail_present", req_close["questions"])

    def test_audio_input_blocks_and_no_reexport_of_pcm16_to_wav_bytes(self):
        self.assertFalse(hasattr(decisions, "pcm16_to_wav_bytes"))
        pcm = (b"\x01\x00\xff\x7f" * 80) + b"\x00"
        wav_bytes = audio_recorder.pcm16_to_wav_bytes(pcm, sample_rate=16000)
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
            utterance_text="",
            audio_wav_bytes=wav_bytes,
            audio_mime_type="audio/wav",
        )
        self.assertIsInstance(req["input"], list)
        self.assertEqual(len(req["input"]), 2)
        self.assertEqual(req["input"][0]["type"], "text")
        self.assertTrue(req["input"][0]["text"].endswith("\n"))
        self.assertEqual(_parsed_input(req)["caller_utterance"], "[spoken audio attached]")
        self.assertEqual(req["input"][1]["type"], "audio")
        self.assertEqual(req["input"][1]["mime_type"], "audio/wav")
        self.assertEqual(base64.b64decode(req["input"][1]["data"]), wav_bytes)

    def test_no_devanagari_persona_or_internal_enums_in_schema_and_decisions(self):
        devanagari_re = re.compile(r"[\u0900-\u097F]")
        forbidden_enums = ("step_kind", "active_qid", "which_child", "open_capture")
        schema_path = Path(decisions.__file__).with_name("decision_schema.json")
        schema_text = schema_path.read_text(encoding="utf-8")
        decisions_text = Path(decisions.__file__).read_text(encoding="utf-8")
        self.assertIsNone(devanagari_re.search(schema_text))
        self.assertIsNone(devanagari_re.search(decisions_text))
        self.assertNotIn("चाची", schema_text)
        self.assertNotIn("चाची", decisions_text)
        for token in (*forbidden_enums, "main", "followup"):
            self.assertIsNone(re.search(rf"\b{re.escape(token)}\b", schema_text), token)

        kids = [cursor_tests._child(2, "male", 1), cursor_tests._child(8, "female", 2, name="रिया")]
        main_state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "birth_jaundice")
        fup_state = apply_turn(main_state, "concern", child_ref="child_1")
        close_state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "close_anything_else")
        for state in (main_state, fup_state, close_state):
            req = decisions.build_turn_request(state, utterance_text="नहीं")
            ctx = _parsed_input(req)
            for key in ctx:
                for token in (*forbidden_enums, "main", "followup"):
                    self.assertNotEqual(key, token)
            for qkey, qobj in req["questions"].items():
                instr = str(qobj.get("instructions") or "")
                self.assertIsNone(devanagari_re.search(instr), (qkey, instr))
                for token in (*forbidden_enums, "main", "followup"):
                    self.assertIsNone(re.search(rf"\b{re.escape(token)}\b", instr), (qkey, token))
                for opt_key, opt_desc in (qobj.get("criteria") or {}).items():
                    if (qkey == "side_qid" and opt_key != "other") or (qkey == "child_ref" and opt_key not in ("all", "other")):
                        continue
                    desc = str(opt_desc or "")
                    self.assertIsNone(devanagari_re.search(desc), (qkey, opt_key, desc))
                    for token in (*forbidden_enums, "main", "followup"):
                        self.assertIsNone(re.search(rf"\b{re.escape(token)}\b", desc), (qkey, opt_key, token))


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

    def test_uncalibrated_refused_and_probabilities_required(self):
        self.assertFalse(decisions.UNCALIBRATED_THRESHOLDS.calibrated)
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        answers = {
            "danger_now": _noul(0.02),
            "caller_stop": _noul(0.02),
            "mentions_profile": _noul(0.02),
            "detail_present": _noul(0.02),
            "current_outcome": _choice("clear", {"clear": 0.95, "other": 0.03, "concern": 0.02}),
            "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
            "side_qid": _choice("other", {"other": 0.96, "fam_cancer": 0.04}),
        }

        with self.assertRaises(ValueError):
            decisions.validate_thresholds(decisions.UNCALIBRATED_THRESHOLDS)
        with self.assertRaises(ValueError):
            decisions.interpret_for_authority(state, answers, utterance_text="नहीं")
        with self.assertRaises(ValueError):
            decisions.interpret_turn_decision(
                state, answers, utterance_text="नहीं",
                thresholds=decisions.UNCALIBRATED_THRESHOLDS, require_calibrated=True,
            )

        for bad in (
            decisions.DecisionThresholds(calibrated=True, danger_now_min_yes=0.01),
            decisions.DecisionThresholds(calibrated=True, danger_now_min_yes=0.99),
        ):
            with self.assertRaises(ValueError):
                decisions.validate_thresholds(bad)

        for field in decisions._ALL_NUMERIC_THRESHOLD_FIELDS:
            self.assertEqual(getattr(decisions.UNCALIBRATED_THRESHOLDS, field), 1.0, field)

        # evaluate_choice strictly requires probabilities dict
        no_probs = {"current_outcome": {"type": "choice", "choice": "clear", "confidence": 0.99}}
        ev_missing = decisions.evaluate_choice(
            no_probs, "current_outcome", catchall_key="other", min_confidence=0.5, min_margin=0.1,
        )
        self.assertFalse(ev_missing.accepted)
        self.assertEqual(ev_missing.reject_reason, "missing_answer")

        shadow_answers = {
            "danger_now": _noul(0.02),
            "caller_stop": _noul(0.02),
            "mentions_profile": _noul(0.03),
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.92, "other": 0.05, "concern": 0.03}),
            "side_intent": _choice("add_concern", {"add_concern": 0.93, "other": 0.05, "clarify_other": 0.02}),
            "side_qid": _choice("fam_cancer", {"fam_cancer": 0.91, "other": 0.05, "fam_lumps": 0.04}),
        }
        shadow_interp = decisions.interpret_turn_decision(
            state, shadow_answers, utterance_text="नहीं",
            thresholds=decisions.UNCALIBRATED_THRESHOLDS, require_calibrated=False,
        )
        self.assertEqual(shadow_interp.cursor_payload.get("side_qid", ""), "")
        self.assertIsNone(shadow_interp.future_qid_match)
        self.assertEqual(shadow_interp.diagnostics["current_outcome"]["winner"], "")
        self.assertEqual(shadow_interp.diagnostics["current_outcome"]["top_option"], "clear")
        self.assertAlmostEqual(shadow_interp.diagnostics["current_outcome"]["top_prob"], 0.92)
        self.assertAlmostEqual(shadow_interp.diagnostics["current_outcome"]["margin"], 0.87)
        self.assertEqual(
            shadow_interp.diagnostics["raw_side_candidate"],
            {
                "qid": "fam_cancer",
                "confidence": 0.91,
                "margin": 0.86,
                "side_intent": "add_concern",
                "side_intent_prob": 0.93,
            },
        )

        valid = _test_thresholds(calibrated=True)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "thresholds.json"
            artifact.write_text(json.dumps(asdict(valid)), encoding="utf-8")
            loaded = decisions.load_thresholds(artifact)
            self.assertEqual(loaded, valid)
            res = decisions.interpret_for_authority(state, answers, utterance_text="नहीं", thresholds=loaded)
            self.assertEqual(res.cursor_payload["current_outcome"], "clear")

            bad_artifact = Path(tmp) / "bad.json"
            bad_data = asdict(valid)
            bad_data["detail_present_min_yes"] = 0.995
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
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.92, "other": 0.04, "concern": 0.03, "unknown": 0.01}),
            "side_intent": _choice("add_concern", {"add_concern": 0.93, "other": 0.05, "clarify_other": 0.02}),
            "side_qid": _choice("fam_cancer", {"fam_cancer": 0.91, "other": 0.04, "fam_lumps": 0.03, "fam_heart_kidney": 0.02}),
        }
        utterance = "नहीं, पर चाची को भी कैंसर हुआ था"
        interp = decisions.interpret_turn_decision(state, answers, utterance_text=utterance, thresholds=thresholds)
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

    def test_side_retract_clears_earlier_concern_and_guards_active_retract(self):
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
            "detail_present": _noul(0.04),
            "current_outcome": _choice("other", {"other": 0.88, "clear": 0.08, "concern": 0.04}),
            "side_intent": _choice("clear_or_retract", {"clear_or_retract": 0.95, "other": 0.03, "add_concern": 0.02}),
            "side_qid": _choice("fam_cancer", {"fam_cancer": 0.92, "other": 0.05, "fam_lumps": 0.03}),
        }
        interp = decisions.interpret_turn_decision(
            state, answers,
            utterance_text="पहले वाले सवाल में गलती से हाँ बोल दिया था, कैंसर किसी को नहीं है",
            thresholds=thresholds,
        )
        self.assertEqual(interp.cursor_payload["side_qid"], "fam_cancer")
        self.assertEqual(interp.cursor_payload["side_outcome"], "retract")
        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["answers"]["fam_cancer"]["state"], "clear")
        self.assertEqual(updated["active_qid"], "fam_heart_kidney")

        leaked_retract = dict(
            answers,
            current_outcome=_choice("retract", {"retract": 0.82, "clear": 0.12, "other": 0.06}),
        )
        interp_leaked = decisions.interpret_turn_decision(
            state, leaked_retract,
            utterance_text="पहले वाले सवाल में गलती से हाँ बोल दिया था, कैंसर किसी को नहीं है",
            thresholds=thresholds,
        )
        self.assertEqual(interp_leaked.cursor_payload["current_outcome"], "not_about_active")
        updated_leaked = apply_interpreted_turn(state, interp_leaked.cursor_payload)
        self.assertEqual(updated_leaked["answers"]["fam_cancer"]["state"], "clear")
        self.assertEqual(updated_leaked["active_qid"], "fam_heart_kidney")
        self.assertNotIn("fam_heart_kidney", updated_leaked["answers"])

    def test_clarify_other_suppresses_current_clarify_and_rewinds_cleanly(self):
        thresholds = _test_thresholds()
        thal = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "preg_thalassemia",
        )
        thal = apply_turn(thal, "unknown")
        self.assertEqual(thal["active_qid"], "fam_related")

        # Both side_intent=clarify_other AND current_outcome=clarify fired -> suppress current clarify!
        answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.01),
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clarify", {"clarify": 0.85, "other": 0.10, "clear": 0.05}),
            "side_intent": _choice("clarify_other", {"clarify_other": 0.92, "other": 0.05, "add_concern": 0.03}),
            "side_qid": _choice("preg_thalassemia", {"preg_thalassemia": 0.91, "other": 0.05, "fam_related": 0.04}),
        }
        interp = decisions.interpret_turn_decision(
            thal, answers, utterance_text="थैलेसीमिया का क्या मतलब होता है?", thresholds=thresholds,
        )
        self.assertEqual(interp.cursor_payload["current_outcome"], "not_about_active")
        self.assertEqual(interp.cursor_payload["side_qid"], "preg_thalassemia")
        self.assertEqual(interp.cursor_payload["side_action"], "clarify_meaning")
        rewound = apply_interpreted_turn(thal, interp.cursor_payload)
        self.assertEqual(rewound["active_qid"], "preg_thalassemia")
        self.assertNotIn("fam_related", rewound["answers"])

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
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.90, "other": 0.06, "concern": 0.04}),
            "side_intent": _choice("add_concern", {"add_concern": 0.91, "other": 0.06, "clarify_other": 0.03}),
            "side_qid": _choice("fam_cancer", {"fam_cancer": 0.93, "other": 0.04, "fam_lumps": 0.03}),
        }
        interp = decisions.interpret_turn_decision(
            state, answers, utterance_text="पीलिया तो नहीं हुआ, हाँ चाची को कैंसर हुआ था", thresholds=thresholds,
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
        state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "birth_jaundice")
        utterance = "हाँ बीच वाले बेटे को तीन दिन नीली लाइट में रखा था"
        answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.03),
            "detail_present": _noul(0.94),
            "current_outcome": _choice("concern", {"concern": 0.96, "other": 0.02, "clear": 0.02}),
            "child_ref": _choice("child_2", {"child_2": 0.91, "other": 0.04, "child_1": 0.03, "child_3": 0.01, "all": 0.01}),
            "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
            "side_qid": _choice("other", {"other": 0.95, "birth_nicu": 0.05}),
        }
        interp = decisions.interpret_turn_decision(state, answers, utterance_text=utterance, thresholds=thresholds)
        self.assertEqual(interp.cursor_payload["child_ref"], "child_2")
        self.assertTrue(interp.cursor_payload["followup_answered"])
        self.assertEqual(interp.cursor_payload["note"], utterance)

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["answers"]["birth_jaundice"]["state"], "concern")
        self.assertEqual(updated["answers"]["birth_jaundice"]["child_ids"], ["child_2"])
        self.assertEqual(updated["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"], utterance)
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
            "detail_present": _noul(0.08),
            "current_outcome": _choice("concern", {"concern": 0.97, "other": 0.02, "clear": 0.01}),
            "side_intent": _choice("other", {"other": 0.97, "add_concern": 0.03}),
            "side_qid": _choice("other", {"other": 0.96, "birth_nicu": 0.04}),
        }
        interp = decisions.interpret_turn_decision(state, answers, utterance_text="हाँ", thresholds=thresholds)
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
            "detail_present": _noul(0.90),
            "current_outcome": _choice("clear", {"clear": 0.95, "other": 0.03, "concern": 0.02}),
            "side_intent": _choice("add_concern", {"add_concern": 0.95, "other": 0.05}),
            "side_qid": _choice("fam_cancer", {"fam_cancer": 0.95, "other": 0.03}),
        }
        interp = decisions.interpret_turn_decision(
            state, answers, utterance_text="अभी बच्चे को सांस नहीं आ रही है", thresholds=thresholds,
        )
        self.assertTrue(interp.cursor_payload["emergency"])
        self.assertEqual(interp.cursor_payload.get("side_qid", ""), "")
        self.assertIsNone(interp.future_qid_match)

        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual(updated["phase"], "closing")
        self.assertEqual(updated["close_reason"], "emergency")
        self.assertEqual(updated["emergency_note"], "अभी बच्चे को सांस नहीं आ रही है")
        self.assertEqual(updated["answers"], before_answers)

    def test_side_qid_rejected_on_low_margin_or_failed_side_intent(self):
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
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
            "side_intent": _choice("add_concern", {"add_concern": 0.90, "other": 0.10}),
            "side_qid": _choice("fam_cancer", {"fam_cancer": 0.85, "other": 0.10, "fam_lumps": 0.05}),
        }

        # 1. Failed side_intent -> rejects side_qid
        failed_intent = dict(base, side_intent=_choice("other", {"other": 0.70, "add_concern": 0.30}))
        res_intent = decisions.interpret_turn_decision(state, failed_intent, utterance_text="नहीं", thresholds=thresholds)
        self.assertEqual(res_intent.cursor_payload.get("side_qid", ""), "")
        self.assertIsNone(res_intent.future_qid_match)

        # 2. Close runner-up real option margin (< 0.20) -> rejects side_qid
        close_runnerup = dict(
            base,
            side_qid=_choice("fam_cancer", {"fam_cancer": 0.72, "fam_lumps": 0.62, "other": 0.04}),
        )
        res_runner = decisions.interpret_turn_decision(state, close_runnerup, utterance_text="नहीं", thresholds=thresholds)
        self.assertEqual(res_runner.cursor_payload.get("side_qid", ""), "")
        self.assertEqual(res_runner.diagnostics["side_qid"]["reject_reason"], "below_margin")

        # 3. Low margin vs catch-all "other" (< 0.20) -> rejects side_qid
        low_catchall = dict(
            base,
            side_qid=_choice("fam_cancer", {"fam_cancer": 0.72, "other": 0.55, "fam_lumps": 0.05}),
        )
        res_catchall = decisions.interpret_turn_decision(state, low_catchall, utterance_text="नहीं", thresholds=thresholds)
        self.assertEqual(res_catchall.cursor_payload.get("side_qid", ""), "")
        self.assertEqual(res_catchall.diagnostics["side_qid"]["reject_reason"], "below_margin")

    def test_needs_profile_extraction_rules(self):
        thresholds = _test_thresholds()
        intro = new_cursor_state()
        plain_answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.05),
            "current_outcome": _choice("other", {"other": 0.90, "clear": 0.10}),
        }
        self.assertTrue(decisions.interpret_turn_decision(intro, plain_answers, thresholds=thresholds).needs_profile_extraction)

        clarify_answers = dict(plain_answers, current_outcome=_choice("clarify", {"clarify": 0.92, "other": 0.05, "clear": 0.03}))
        self.assertFalse(decisions.interpret_turn_decision(intro, clarify_answers, thresholds=thresholds).needs_profile_extraction)
        married_state = apply_profile(intro, caller_name="पूजा")[0]
        self.assertFalse(decisions.interpret_turn_decision(married_state, clarify_answers, thresholds=thresholds).needs_profile_extraction)

        clarify_with_profile = dict(clarify_answers, mentions_profile=_noul(0.90))
        self.assertTrue(decisions.interpret_turn_decision(intro, clarify_with_profile, thresholds=thresholds).needs_profile_extraction)

        ever_state = apply_turn(cursor_tests._profile(children=[]), "clear")
        self.assertEqual(ever_state["active_qid"], "ever_pregnant")
        self.assertFalse(decisions.interpret_turn_decision(ever_state, plain_answers, thresholds=thresholds).needs_profile_extraction)
        ever_with_profile = dict(plain_answers, mentions_profile=_noul(0.90))
        self.assertTrue(decisions.interpret_turn_decision(ever_state, ever_with_profile, thresholds=thresholds).needs_profile_extraction)

        gate_state = cursor_tests._clear_until(cursor_tests._profile(children=[cursor_tests._child(4, "male")]), "birth_jaundice")
        self.assertFalse(decisions.interpret_turn_decision(gate_state, plain_answers, thresholds=thresholds).needs_profile_extraction)
        with_profile = dict(plain_answers, mentions_profile=_noul(0.90))
        self.assertTrue(decisions.interpret_turn_decision(gate_state, with_profile, thresholds=thresholds).needs_profile_extraction)


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

            res = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=fake_transport))
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

        res = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=always_timeout))
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(calls, 2)
        self.assertEqual(res.error_reason, "timeout")
        self.assertEqual(res.raw_answers, {})

    def test_no_retry_on_http_400_or_safety_block_or_empty_answers(self):
        req = self._sample_request()
        calls_400 = 0

        def bad_req(url, body_bytes, headers, timeout_sec):
            nonlocal calls_400
            calls_400 += 1
            return 400, b'{"error":{"code":"invalid_request","message":"bad field"}}'

        res_400 = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=bad_req))
        self.assertFalse(res_400.ok)
        self.assertEqual(res_400.attempts, 1)
        self.assertEqual(calls_400, 1)
        self.assertEqual(res_400.error_reason, "http_400")

        for code in ("safety", "prohibited_content"):
            calls_sb = 0
            body = {"error": {"code": code, "message": "Request blocked due to safety violations."}}

            def safety_fn(url, body_bytes, headers, timeout_sec, payload=body):
                nonlocal calls_sb
                calls_sb += 1
                return 400, json.dumps(payload).encode("utf-8")

            res_sb = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=safety_fn))
            self.assertFalse(res_sb.ok)
            self.assertEqual(res_sb.attempts, 1)
            self.assertEqual(calls_sb, 1)
            self.assertEqual(res_sb.error_reason, "safety_block")

        calls_empty = 0

        def empty_fn(url, body_bytes, headers, timeout_sec):
            nonlocal calls_empty
            calls_empty += 1
            return 200, b'{"answers": {}, "usage": null}'

        res_empty = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=empty_fn))
        self.assertFalse(res_empty.ok)
        self.assertEqual(res_empty.attempts, 1)
        self.assertEqual(res_empty.error_reason, "empty_answers")
        self.assertFalse(res_empty.usage_present)
        self.assertIsNone(res_empty.total_input_tokens)

        calls_bad = 0

        def malformed_fn(url, body_bytes, headers, timeout_sec):
            nonlocal calls_bad
            calls_bad += 1
            return 200, b"{not json"

        res_bad = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=malformed_fn))
        self.assertFalse(res_bad.ok)
        self.assertEqual(res_bad.attempts, 1)
        self.assertEqual(res_bad.error_reason, "malformed_response")

    def test_usage_null_and_documented_usage_schema(self):
        req = self._sample_request()

        def null_usage(url, body_bytes, headers, timeout_sec):
            return 200, json.dumps({"answers": {"danger_now": _noul(0.01)}, "usage": None}).encode("utf-8")

        res_null = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=null_usage))
        self.assertTrue(res_null.ok)
        self.assertFalse(res_null.usage_present)
        self.assertIsNone(res_null.total_input_tokens)

        def documented_usage(url, body_bytes, headers, timeout_sec):
            return 200, json.dumps({
                "answers": {"danger_now": _noul(0.01)},
                "usage": {
                    "total_input_tokens": 425,
                    "input_tokens_by_modality": [{"modality": "text", "tokens": 425}],
                    "total_output_tokens": 0,
                    "total_tokens": 425,
                },
            }).encode("utf-8")

        res_doc = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=documented_usage))
        self.assertTrue(res_doc.ok)
        self.assertTrue(res_doc.usage_present)
        self.assertEqual(res_doc.total_input_tokens, 425)

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
                raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, io.BytesIO(b"busy"))
            return 200, json.dumps({"answers": {"danger_now": _noul(0.01)}}).encode("utf-8")

        res = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=raise_http_503))
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

            res = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=retry_fn))
            self.assertTrue(res.ok, first_exc)
            self.assertEqual(res.attempts, 2)

        def slow_fn(url, body_bytes, headers, timeout_sec):
            time.sleep(0.15)
            return 200, ok_body

        started = time.perf_counter()
        res_budget = asyncio.run(
            decisions.evaluate_turn(
                req, api_key="test-key", attempt_timeout_sec=0.04, total_budget_sec=0.09, transport=slow_fn,
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
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"answers": {"danger_now": _noul(0.01)}, "usage": None}).encode("utf-8")

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
            "detail_present": _noul(0.05),
            "current_outcome": _choice("clear", {"clear": 0.91, "other": 0.05, "concern": 0.04}),
            "side_intent": _choice("clarify_other", {"clarify_other": 0.91, "add_concern": 0.05, "other": 0.04}),
            "side_qid": _choice("preg_thalassemia", {"preg_thalassemia": 0.90, "other": 0.05, "fam_related": 0.05}),
        }
        res_clar = decisions.interpret_turn_decision(
            thal, clarify_answers,
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
            side_intent=_choice("other", {"other": 0.96, "clarify_other": 0.04}),
        )
        res_stop = decisions.interpret_turn_decision(
            thal, stop_answers, utterance_text="अभी फोन बंद कर दो, बाद में बात करेंगे", thresholds=thresholds,
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
            "detail_present": _noul(0.05),
            "current_outcome": _choice("concern", {"concern": 0.95, "other": 0.03, "clear": 0.02}),
            "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
            "side_qid": _choice("other", {"other": 0.96, "fam_cancer": 0.04}),
        }
        res_bare_close = decisions.interpret_turn_decision(close_state, bare_close, utterance_text="हाँ", thresholds=thresholds)
        asked_detail = apply_interpreted_turn(close_state, res_bare_close.cursor_payload)
        self.assertEqual(asked_detail["pending_followup_idx"], 0)
        self.assertEqual(asked_detail["phase"], "asking")

        noted_close = dict(bare_close, detail_present=_noul(0.91))
        res_noted_close = decisions.interpret_turn_decision(
            asked_detail, noted_close, utterance_text="रात में नींद कम आती है", thresholds=thresholds,
        )
        saved_close = apply_interpreted_turn(asked_detail, res_noted_close.cursor_payload)
        self.assertIsNone(saved_close["pending_followup_idx"])
        self.assertEqual(saved_close["answers"]["close_anything_else"]["notes"], "रात में नींद कम आती है")

    def test_followup_step_saves_spoken_answer_without_detail_present(self):
        thresholds = _test_thresholds()
        nicu_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )
        nicu_state = apply_turn(nicu_state, "concern", note="सांस", followup_answered=True)
        self.assertEqual(nicu_state["active_qid"], "birth_nicu")
        self.assertEqual(nicu_state["pending_followup_idx"], 1)

        followup_answers = {
            "danger_now": _noul(0.01),
            "caller_stop": _noul(0.01),
            "mentions_profile": _noul(0.01),
            "current_outcome": _choice("concern", {"concern": 0.92, "other": 0.05, "clear": 0.03}),
            "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
            "side_qid": _choice("other", {"other": 0.95, "birth_jaundice": 0.05}),
        }
        interp_nicu = decisions.interpret_turn_decision(
            nicu_state, followup_answers, utterance_text="चार दिन", thresholds=thresholds,
        )
        self.assertIsNone(interp_nicu.diagnostics["detail_present"])
        self.assertEqual(interp_nicu.cursor_payload["note"], "चार दिन")
        updated_nicu = apply_interpreted_turn(nicu_state, interp_nicu.cursor_payload)
        self.assertEqual(updated_nicu["answers"]["birth_nicu"]["followups"]["nicu_days"], {"state": "concern", "note": "चार दिन"})
        self.assertIsNone(updated_nicu["pending_followup_idx"])
        self.assertEqual(updated_nicu["active_qid"], "birth_jaundice")

        close_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        open_detail_state = apply_turn(close_state, "concern")
        self.assertEqual(open_detail_state["pending_followup_idx"], 0)
        interp_open = decisions.interpret_turn_decision(
            open_detail_state, followup_answers, utterance_text="नींद में डर लगता है", thresholds=thresholds,
        )
        self.assertEqual(interp_open.cursor_payload["note"], "नींद में डर लगता है")
        updated_open = apply_interpreted_turn(open_detail_state, interp_open.cursor_payload)
        self.assertIsNone(updated_open["pending_followup_idx"])
        self.assertEqual(updated_open["answers"]["close_anything_else"]["followups"]["open_concern_detail"]["note"], "नींद में डर लगता है")
        self.assertIn("नींद में डर लगता है", updated_open["answers"]["close_anything_else"]["notes"])

    def test_provisional_staging_thresholds_load_and_default_authority_is_fail_closed(self):
        provisional_path = REPO_ROOT / "scripts" / "provisional_staging_thresholds.json"
        default_prod_path = APP_DIR / "decision_thresholds.json"
        self.assertFalse(default_prod_path.exists())
        loaded = decisions.load_thresholds(provisional_path, require_calibrated=True)
        self.assertTrue(loaded.calibrated)
        self.assertTrue(0.01 < loaded.danger_now_min_yes < 0.99)


if __name__ == "__main__":
    unittest.main()
