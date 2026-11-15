"""100% offline unit tests for Decisions API client, declarative schema, and interpreter."""

from __future__ import annotations

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

from tests import APP_DIR
from tests.helpers import choice, clear_until, make_child, make_profile_state, noul, parsed_input, test_thresholds

import audio_recorder  # noqa: E402
import call_cursor  # noqa: E402
import decisions  # noqa: E402
from call_cursor import apply_interpreted_turn, apply_profile, apply_turn, new_cursor_state, routable_qids  # noqa: E402
from triage_catalog import EVER_PREGNANT_HI, FOLLOWUP_HI, get_spec  # noqa: E402


def _answers(**overrides) -> dict:
    base = {
        "danger_now": noul(0.01),
        "caller_stop": noul(0.01),
        "mentions_profile": noul(0.01),
        "detail_present": noul(0.05),
        "current_outcome": choice("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
        "side_intent": choice("other", {"other": 0.96, "add_concern": 0.04}),
        "side_qid": choice("other", {"other": 0.96, "birth_nicu": 0.04}),
    }
    base.update(overrides)
    return base


class SchemaBuilderTests(unittest.TestCase):
    def test_side_qid_has_59_options_and_json_input_fields(self):
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        req = decisions.build_turn_request(state, utterance_text="नहीं")
        self.assertEqual((set(req.keys()), req["model"]), ({"model", "input", "questions"}, "gemini-decision-1"))
        self.assertIsInstance(req["input"], str)

        ctx = parsed_input(req)
        self.assertEqual(set(ctx.keys()), {"active_question", "followup_question", "last_answered_question", "caller_utterance"})
        self.assertEqual(
            (ctx["active_question"], ctx["followup_question"], ctx["last_answered_question"], ctx["caller_utterance"]),
            (call_cursor.active_prompt_hi(state), FOLLOWUP_HI["jaundice_days"], call_cursor.last_answered_prompt_hi(state)[1], "नहीं"),
        )
        questions = req["questions"]
        for qkey in questions:
            self.assertRegex(qkey, r"^[A-Za-z0-9_-]{1,64}$")
        self.assertEqual(questions["mentions_diagnosis"]["type"], "noul")

        side_criteria = questions["side_qid"]["criteria"]
        self.assertEqual(len(side_criteria), 59)
        self.assertEqual(set(side_criteria.keys()), set(routable_qids()) | {"other"})
        other_desc = side_criteria["other"]
        self.assertTrue("above" not in other_desc.lower() and "last" not in other_desc.lower() and "outside active_question" in other_desc)
        self.assertIn(EVER_PREGNANT_HI, side_criteria["ever_pregnant"])
        for qid in routable_qids():
            if qid != "ever_pregnant":
                spec = get_spec(qid)
                self.assertIn(call_cursor.question_hi_for(state, spec), side_criteria[qid])
                self.assertIn(spec.need_en, side_criteria[qid])

    def test_intro_hold_ever_pregnant_and_close_prompts(self):
        req_profile = decisions.build_turn_request(new_cursor_state(), utterance_text="मेरा नाम पूजा")
        self.assertEqual(set(parsed_input(req_profile).keys()), {"active_question", "caller_utterance"})
        self.assertEqual(set(req_profile["questions"].keys()), {"danger_now", "caller_stop", "mentions_profile", "current_outcome"})

        bridge = make_profile_state(children=[make_child(4, "male")])
        req_bridge = decisions.build_turn_request(bridge, utterance_text="हाँ शुरू करें")
        self.assertEqual(set(parsed_input(req_bridge).keys()), {"active_question", "last_answered_question", "caller_utterance"})
        self.assertEqual(set(req_bridge["questions"].keys()), {"danger_now", "caller_stop", "mentions_profile", "current_outcome"})

        held_name, hold = apply_profile(new_cursor_state())
        self.assertEqual(hold, "name_fragment")
        self.assertEqual(parsed_input(decisions.build_turn_request(held_name, utterance_text="पूजा शर्मा"))["active_question"], call_cursor._HOLD_HI["name_fragment"])

        ever_state = apply_turn(make_profile_state(children=[]), "clear")
        req_ever = decisions.build_turn_request(ever_state, utterance_text="नहीं")
        self.assertEqual(parsed_input(req_ever)["active_question"], EVER_PREGNANT_HI)
        self.assertNotIn("detail_present", req_ever["questions"])
        for k in ("mentions_diagnosis", "side_intent", "side_qid"):
            self.assertIn(k, req_ever["questions"])

        close_more = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else"), "concern", note="पेट दर्द")
        req_more = decisions.build_turn_request(close_more, utterance_text="नहीं")
        self.assertEqual(parsed_input(req_more)["active_question"], call_cursor._CLOSE_MORE_HI)
        self.assertIn("detail_present", req_more["questions"])

    def test_conditional_child_ref_and_detail_present_across_step_kinds(self):
        no_kids = clear_until(make_profile_state(marital="unmarried"), "preg_thalassemia")
        req_no = decisions.build_turn_request(no_kids, utterance_text="नहीं")
        self.assertTrue("child_ref" not in req_no["questions"] and "detail_present" not in req_no["questions"])
        self.assertNotIn("पति", parsed_input(req_no)["active_question"])
        self.assertNotIn("पति", req_no["questions"]["side_qid"]["criteria"]["preg_thalassemia"])

        req_one_no_fup = decisions.build_turn_request(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_cord_3weeks"), utterance_text="नहीं")
        self.assertTrue("child_ref" not in req_one_no_fup["questions"] and "detail_present" not in req_one_no_fup["questions"])

        req_one_fup = decisions.build_turn_request(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice"), utterance_text="हाँ दो दिन")
        self.assertTrue("child_ref" not in req_one_fup["questions"] and "detail_present" in req_one_fup["questions"])

        three_kids = [make_child(2, "male", 1), make_child(8, "female", 2, name="रिया"), make_child(5, "male", 3, name="अर्जुन")]
        multi = clear_until(make_profile_state(children=three_kids), "birth_jaundice")
        req_multi = decisions.build_turn_request(multi, utterance_text="बीच वाले को")
        self.assertTrue("child_ref" in req_multi["questions"] and "detail_present" in req_multi["questions"])
        self.assertEqual(set(req_multi["questions"]["child_ref"]["criteria"].keys()), {"child_1", "child_2", "child_3", "all", "other"})
        roster_out = parsed_input(req_multi)["children_roster"]
        self.assertEqual([r["id"] for r in roster_out], ["child_2", "child_3", "child_1"])
        self.assertEqual(([r["rank"] for r in roster_out], roster_out[0]["name"]), (["oldest", "middle", "youngest"], "रिया"))
        self.assertNotIn("name", roster_out[2])

        which = apply_turn(multi, "concern")
        req_which = decisions.build_turn_request(which, utterance_text="अर्जुन")
        self.assertTrue("child_ref" in req_which["questions"] and "detail_present" not in req_which["questions"])
        self.assertIn("यह किस बच्चे की बात है", parsed_input(req_which)["active_question"])
        self.assertIn("नीली लाइट", parsed_input(req_which)["parent_question"])

        fup = apply_turn(which, "concern", child_ref="child_3")
        req_fup = decisions.build_turn_request(fup, utterance_text="चार दिन")
        self.assertNotIn("detail_present", req_fup["questions"])
        self.assertEqual(parsed_input(req_fup)["followup_question"], FOLLOWUP_HI["jaundice_days"])

        close = clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else")
        self.assertIn("detail_present", decisions.build_turn_request(close, utterance_text="हाँ")["questions"])

    def test_audio_input_blocks_and_wav_encoding(self):
        pcm = (b"\x01\x00\xff\x7f" * 80) + b"\x00"
        wav_bytes = audio_recorder.pcm16_to_wav_bytes(pcm, sample_rate=16000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual((wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())),
                             (1, 2, 16000, pcm[:-1]))

        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        req = decisions.build_turn_request(state, utterance_text="", audio_wav_bytes=wav_bytes, audio_mime_type="audio/wav")
        self.assertEqual((len(req["input"]), req["input"][0]["type"], req["input"][1]["type"], req["input"][1]["mime_type"]),
                         (2, "text", "audio", "audio/wav"))
        self.assertTrue(req["input"][0]["text"].endswith("\n"))
        self.assertEqual(parsed_input(req)["caller_utterance"], "[spoken audio attached]")
        self.assertEqual(base64.b64decode(req["input"][1]["data"]), wav_bytes)

    def test_no_devanagari_persona_or_internal_enums_in_schema_and_decisions(self):
        devanagari_re = re.compile(r"[\u0900-\u097F]")
        forbidden = ("step_kind", "active_qid", "which_child", "open_capture", "main", "followup")
        schema_text = Path(decisions.__file__).with_name("decision_schema.json").read_text(encoding="utf-8")
        decisions_text = Path(decisions.__file__).read_text(encoding="utf-8")
        for text in (schema_text, decisions_text):
            self.assertIsNone(devanagari_re.search(text))
            self.assertNotIn("चाची", text)
        for token in forbidden:
            self.assertIsNone(re.search(rf"\b{re.escape(token)}\b", schema_text), token)

        kids = [make_child(2, "male", 1), make_child(8, "female", 2, name="रिया")]
        main_s = clear_until(make_profile_state(children=kids), "birth_jaundice")
        for st in (main_s, apply_turn(main_s, "concern", child_ref="child_1"), clear_until(make_profile_state(children=kids), "close_anything_else")):
            req = decisions.build_turn_request(st, utterance_text="नहीं")
            for k in parsed_input(req):
                self.assertNotIn(k, forbidden)
            for qkey, qobj in req["questions"].items():
                instr = str(qobj.get("instructions") or "")
                self.assertIsNone(devanagari_re.search(instr), (qkey, instr))
                for tok in forbidden:
                    self.assertIsNone(re.search(rf"\b{re.escape(tok)}\b", instr), (qkey, tok))
                for opt_k, opt_d in (qobj.get("criteria") or {}).items():
                    if (qkey == "side_qid" and opt_k != "other") or (qkey == "child_ref" and opt_k not in ("all", "other")):
                        continue
                    self.assertIsNone(devanagari_re.search(str(opt_d or "")), (qkey, opt_k))


class ThresholdAndGuardTests(unittest.TestCase):
    def test_thresholds_validation_artifact_loading_and_probabilities_required(self):
        self.assertEqual(len(decisions._ALL_NUMERIC_THRESHOLD_FIELDS), 13)
        self.assertNotIn("calibrated", decisions.DecisionThresholds.__dataclass_fields__)

        with self.assertRaises(TypeError):
            decisions.validate_thresholds({"danger_now_min_yes": 0.5})  # type: ignore[arg-type]

        valid = test_thresholds()
        base = asdict(valid)
        for bad in (
            {**base, "danger_now_min_yes": 0.01}, {**base, "danger_now_min_yes": 0.99},
            {**base, "outcome_min_confidence": 0.0}, {**base, "outcome_min_confidence": 1.0},
            {**base, "outcome_min_margin": -0.01}, {**base, "outcome_min_margin": 1.0},
            {**base, "child_ref_min_confidence": float("nan")},
        ):
            with self.assertRaises(ValueError):
                decisions.validate_thresholds(decisions.DecisionThresholds(**bad))

        ev_missing = decisions.evaluate_choice(
            {"current_outcome": {"type": "choice", "choice": "clear", "confidence": 0.99}},
            "current_outcome", catchall_key="other", min_confidence=0.5, min_margin=0.1,
        )
        self.assertEqual((ev_missing.accepted, ev_missing.reject_reason), (False, "missing_answer"))

        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        with tempfile.TemporaryDirectory() as tmp:
            p_flat = Path(tmp) / "t.json"
            p_flat.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(decisions.load_thresholds(p_flat), valid)

            p_wrap = Path(tmp) / "w.json"
            p_wrap.write_text(json.dumps({"thresholds": base}), encoding="utf-8")
            loaded = decisions.load_thresholds(p_wrap)
            self.assertEqual(loaded, valid)
            self.assertEqual(decisions.interpret_turn_decision(state, _answers(), utterance_text="नहीं", thresholds=loaded).cursor_payload["current_outcome"], "clear")

            for bad_obj in (
                {k: v for k, v in base.items() if k != "child_ref_min_margin"},
                [1, 2, 3],
                {**base, "detail_present_min_yes": 0.995},
                {**base, "unknown_typo": 0.5},
                {"thresholds": {**base, "unknown_typo": 0.5}},
            ):
                p_bad = Path(tmp) / "bad.json"
                p_bad.write_text(json.dumps(bad_obj), encoding="utf-8")
                with self.assertRaises(ValueError):
                    decisions.load_thresholds(p_bad)

    def test_shipped_decision_thresholds_load_and_missing_raises(self):
        loaded = decisions.load_thresholds(APP_DIR / "decision_thresholds.json")
        self.assertEqual(decisions.validate_thresholds(loaded), loaded)
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        self.assertEqual(decisions.interpret_turn_decision(state, _answers(), utterance_text="नहीं", thresholds=loaded).cursor_payload["current_outcome"], "clear")
        with self.assertRaises(FileNotFoundError):
            decisions.load_thresholds(APP_DIR / "nonexistent_thresholds.json")


class InterpreterEndToEndTests(unittest.TestCase):
    def test_smoke_a_mixed_clear_and_earlier_fam_cancer_addition(self):
        state = clear_until(apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "concern", note="मामा को ब्लड कैंसर"), "concern", note="मामा"), "fam_heart_kidney")
        ans = _answers(
            current_outcome=choice("clear", {"clear": 0.92, "other": 0.04, "concern": 0.03}),
            side_intent=choice("add_concern", {"add_concern": 0.93, "other": 0.05, "clarify_other": 0.02}),
            side_qid=choice("fam_cancer", {"fam_cancer": 0.91, "other": 0.04, "fam_lumps": 0.03}),
        )
        interp = decisions.interpret_turn_decision(state, ans, utterance_text="नहीं, पर चाची को भी कैंसर हुआ था", thresholds=test_thresholds())
        self.assertFalse(interp.needs_profile_extraction or bool(interp.future_qid_match))
        self.assertEqual((interp.cursor_payload["current_outcome"], interp.cursor_payload["side_qid"], interp.cursor_payload["side_action"], interp.cursor_payload["side_outcome"]),
                         ("clear", "fam_cancer", "add_or_correct", "concern"))
        updated = apply_interpreted_turn(state, interp.cursor_payload)
        self.assertEqual((updated["answers"]["fam_heart_kidney"]["state"], updated["active_qid"]), ("clear", "fam_bp_stroke"))
        self.assertIn("चाची को भी कैंसर हुआ था", updated["answers"]["fam_cancer"]["notes"])

    def test_side_retract_and_clarify_other_suppress_active_leaks(self):
        thresholds = test_thresholds()
        state = clear_until(apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "concern", note="शायद कैंसर"), "concern", note="मामा"), "fam_heart_kidney")
        retract_ans = _answers(
            current_outcome=choice("other", {"other": 0.88, "clear": 0.08, "concern": 0.04}),
            side_intent=choice("clear_or_retract", {"clear_or_retract": 0.95, "other": 0.03}),
            side_qid=choice("fam_cancer", {"fam_cancer": 0.92, "other": 0.05}),
        )
        interp = decisions.interpret_turn_decision(state, retract_ans, utterance_text="कैंसर किसी को नहीं है", thresholds=thresholds)
        self.assertEqual((interp.cursor_payload["side_qid"], interp.cursor_payload["side_outcome"]), ("fam_cancer", "retract"))
        leaked = dict(retract_ans, current_outcome=choice("retract", {"retract": 0.82, "clear": 0.12, "other": 0.06}))
        interp_leaked = decisions.interpret_turn_decision(state, leaked, utterance_text="कैंसर किसी को नहीं है", thresholds=thresholds)
        self.assertEqual(interp_leaked.cursor_payload["current_outcome"], "not_about_active")
        updated = apply_interpreted_turn(state, interp_leaked.cursor_payload)
        self.assertEqual((updated["answers"]["fam_cancer"]["state"], updated["active_qid"]), ("clear", "fam_heart_kidney"))
        self.assertNotIn("fam_heart_kidney", updated["answers"])

        thal = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "preg_thalassemia"), "unknown")
        clar_ans = _answers(
            current_outcome=choice("clarify", {"clarify": 0.85, "other": 0.10, "clear": 0.05}),
            side_intent=choice("clarify_other", {"clarify_other": 0.92, "other": 0.05}),
            side_qid=choice("preg_thalassemia", {"preg_thalassemia": 0.91, "other": 0.05}),
        )
        clar_interp = decisions.interpret_turn_decision(thal, clar_ans, utterance_text="थैलेसीमिया क्या है?", thresholds=thresholds)
        self.assertEqual((clar_interp.cursor_payload["current_outcome"], clar_interp.cursor_payload["side_qid"], clar_interp.cursor_payload["side_action"]),
                         ("not_about_active", "preg_thalassemia", "clarify_meaning"))
        self.assertEqual(apply_interpreted_turn(thal, clar_interp.cursor_payload)["active_qid"], "preg_thalassemia")

    def test_future_qid_match_multichild_detail_bare_yes_and_emergency(self):
        thresholds = test_thresholds()
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        fut = decisions.interpret_turn_decision(
            state,
            _answers(side_intent=choice("add_concern", {"add_concern": 0.91, "other": 0.06}), side_qid=choice("fam_cancer", {"fam_cancer": 0.93, "other": 0.04})),
            utterance_text="पीलिया तो नहीं हुआ, हाँ चाची को कैंसर हुआ था",
            thresholds=thresholds,
        )
        self.assertEqual(fut.cursor_payload.get("side_qid", ""), "")
        self.assertEqual(fut.future_qid_match, {"qid": "fam_cancer", "confidence": 0.93, "act": "add_or_correct"})
        self.assertEqual(apply_interpreted_turn(state, fut.cursor_payload)["active_qid"], "birth_sudden")

        kids = [make_child(9, "female", 1, name="रिया"), make_child(5, "male", 2, name="अर्जुन"), make_child(2, "female", 3, name="मीरा")]
        multi = clear_until(make_profile_state(children=kids), "birth_jaundice")
        utt = "हाँ बीच वाले बेटे को तीन दिन नीली लाइट में रखा था"
        m_ans = _answers(
            detail_present=noul(0.94),
            current_outcome=choice("concern", {"concern": 0.96, "other": 0.02, "clear": 0.02}),
            child_ref=choice("child_2", {"child_2": 0.91, "other": 0.04, "child_1": 0.03}),
        )
        m_interp = decisions.interpret_turn_decision(multi, m_ans, utterance_text=utt, thresholds=thresholds)
        self.assertEqual((m_interp.cursor_payload["child_ref"], m_interp.cursor_payload["followup_answered"], m_interp.cursor_payload["current_note"]), ("child_2", True, utt))
        m_upd = apply_interpreted_turn(multi, m_interp.cursor_payload)
        self.assertEqual((m_upd["answers"]["birth_jaundice"]["child_ids"], m_upd["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"], m_upd["active_qid"]),
                         (["child_2"], utt, "birth_sudden"))

        bare = decisions.interpret_turn_decision(state, _answers(detail_present=noul(0.08), current_outcome=choice("concern", {"concern": 0.97, "other": 0.03})), utterance_text="हाँ", thresholds=thresholds)
        self.assertEqual((bare.cursor_payload["followup_answered"], bare.cursor_payload["current_note"]), (False, ""))
        self.assertEqual(apply_interpreted_turn(state, bare.cursor_payload)["pending_followup_idx"], 0)

        before = json.loads(json.dumps(state["answers"]))
        emg = decisions.interpret_turn_decision(
            state,
            _answers(danger_now=noul(0.95), caller_stop=noul(0.91), side_intent=choice("add_concern", {"add_concern": 0.95, "other": 0.05}), side_qid=choice("fam_cancer", {"fam_cancer": 0.95, "other": 0.05})),
            utterance_text="अभी बच्चे को सांस नहीं आ रही है",
            thresholds=thresholds,
        )
        self.assertTrue(emg.cursor_payload["emergency"] and not emg.cursor_payload.get("side_qid") and emg.future_qid_match is None)
        emg_upd = apply_interpreted_turn(state, emg.cursor_payload)
        self.assertEqual((emg_upd["phase"], emg_upd["close_reason"], emg_upd["answers"]), ("closing", "emergency", before))

    def test_side_qid_rejected_on_low_margin_or_failed_side_intent(self):
        thresholds = test_thresholds()
        state = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "clear")
        base = _answers(side_intent=choice("add_concern", {"add_concern": 0.90, "other": 0.10}), side_qid=choice("fam_cancer", {"fam_cancer": 0.85, "other": 0.10}))
        for bad_ans, exp_reason in (
            ({**base, "side_intent": choice("other", {"other": 0.70, "add_concern": 0.30})}, None),
            ({**base, "side_qid": choice("fam_cancer", {"fam_cancer": 0.72, "fam_lumps": 0.62, "other": 0.04})}, "below_margin"),
            ({**base, "side_qid": choice("fam_cancer", {"fam_cancer": 0.72, "other": 0.55, "fam_lumps": 0.05})}, "below_margin"),
        ):
            res = decisions.interpret_turn_decision(state, bad_ans, utterance_text="नहीं", thresholds=thresholds)
            self.assertEqual(res.cursor_payload.get("side_qid", ""), "")
            if exp_reason:
                self.assertEqual(res.diagnostics["side_qid"]["reject_reason"], exp_reason)

    def test_needs_profile_extraction_rules(self):
        thresholds = test_thresholds()
        intro = new_cursor_state()
        plain = _answers(current_outcome=choice("other", {"other": 0.90, "clear": 0.10}))
        clar = _answers(current_outcome=choice("clarify", {"clarify": 0.92, "other": 0.08}))
        self.assertTrue(decisions.interpret_turn_decision(intro, plain, thresholds=thresholds).needs_profile_extraction)
        self.assertFalse(decisions.interpret_turn_decision(intro, clar, thresholds=thresholds).needs_profile_extraction)
        self.assertFalse(decisions.interpret_turn_decision(apply_profile(intro, caller_name="पूजा")[0], clar, thresholds=thresholds).needs_profile_extraction)
        self.assertTrue(decisions.interpret_turn_decision(intro, {**clar, "mentions_profile": noul(0.90)}, thresholds=thresholds).needs_profile_extraction)

        ever = apply_turn(make_profile_state(children=[]), "clear")
        gate = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        for st in (ever, gate):
            self.assertFalse(decisions.interpret_turn_decision(st, plain, thresholds=thresholds).needs_profile_extraction)
            self.assertTrue(decisions.interpret_turn_decision(st, {**plain, "mentions_profile": noul(0.90)}, thresholds=thresholds).needs_profile_extraction)

    def test_clarify_stop_close_followup_and_mentions_diagnosis_isolation(self):
        thresholds = test_thresholds()
        thal = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "preg_thalassemia"), "unknown")
        stop = decisions.interpret_turn_decision(thal, _answers(caller_stop=noul(0.92)), utterance_text="अभी फोन बंद कर दो", thresholds=thresholds)
        self.assertEqual(apply_interpreted_turn(thal, stop.cursor_payload)["close_reason"], "caller_stop")

        close_s = clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else")
        asked = apply_interpreted_turn(close_s, decisions.interpret_turn_decision(close_s, _answers(detail_present=noul(0.05), current_outcome=choice("concern", {"concern": 0.95, "other": 0.05})), utterance_text="हाँ", thresholds=thresholds).cursor_payload)
        self.assertEqual(asked["pending_followup_idx"], 0)
        saved = apply_interpreted_turn(asked, decisions.interpret_turn_decision(asked, _answers(current_outcome=choice("concern", {"concern": 0.95, "other": 0.05})), utterance_text="रात में नींद कम आती है", thresholds=thresholds).cursor_payload)
        self.assertEqual((saved["pending_followup_idx"], saved["answers"]["close_anything_else"]["followups"]["open_concern_detail"]["note"]), (None, "रात में नींद कम आती है"))

        nicu = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu"), "concern", note="सांस", followup_answered=True)
        fup_interp = decisions.interpret_turn_decision(nicu, _answers(current_outcome=choice("concern", {"concern": 0.92, "other": 0.08})), utterance_text="चार दिन", thresholds=thresholds)
        self.assertIsNone(fup_interp.diagnostics["detail_present"])
        self.assertEqual(apply_interpreted_turn(nicu, fup_interp.cursor_payload)["answers"]["birth_nicu"]["followups"]["nicu_days"]["note"], "चार दिन")

        bleed = clear_until(make_profile_state(children=[make_child(4, "male")]), "body_bleeding_bruise")
        diag = decisions.interpret_turn_decision(bleed, _answers(mentions_diagnosis=noul(0.88), current_outcome=choice("concern", {"concern": 0.94, "other": 0.06})), utterance_text="हाँ डॉक्टर ने Hemophilia बताया है", thresholds=thresholds)
        diag_entry = apply_interpreted_turn(bleed, diag.cursor_payload)["answers"]["body_bleeding_bruise"]
        self.assertTrue(diag.cursor_payload["diagnostic_note"])
        self.assertEqual((call_cursor.answer_notes_text(diag_entry), call_cursor.mother_notes_text(diag_entry)), ("हाँ डॉक्टर ने Hemophilia बताया है", ""))

        thal_s = clear_until(make_profile_state(children=[make_child(4, "male")]), "preg_thalassemia")
        routine = decisions.interpret_turn_decision(thal_s, _answers(mentions_diagnosis=noul(0.12), current_outcome=choice("concern", {"concern": 0.93, "other": 0.07})), utterance_text="हाँ थैलेसीमिया माइनर निकला था", thresholds=thresholds)
        self.assertFalse(routine.cursor_payload["diagnostic_note"])
        self.assertEqual(call_cursor.mother_notes_text(apply_interpreted_turn(thal_s, routine.cursor_payload)["answers"]["preg_thalassemia"]), "हाँ थैलेसीमिया माइनर निकला था")


class TransportRetryTests(unittest.TestCase):
    def _req(self):
        return decisions.build_turn_request(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice"), utterance_text="नहीं")

    def test_retries_once_on_transient_errors_and_handles_failures(self):
        req = self._req()
        ok_body = json.dumps({"answers": {"danger_now": noul(0.02)}, "usage": {"total_input_tokens": 310, "total_output_tokens": 0}}).encode()

        for first in (
            TimeoutError("timed out"), (503, b'{"error":{"message":"busy"}}'), (429, b'{"error":{"message":"rate"}}'),
            (408, b"timeout"), ConnectionResetError("reset"), urllib.error.URLError("dns"),
            http.client.RemoteDisconnected("closed"),
            urllib.error.HTTPError(decisions.DECISIONS_ENDPOINT, 503, "Busy", {}, io.BytesIO(b"busy")),
        ):
            calls = []

            def retry_fn(url, body_bytes, headers, timeout_sec, err=first):
                calls.append((url, headers))
                if len(calls) == 1:
                    if isinstance(err, BaseException):
                        raise err
                    return err
                return 200, ok_body

            res = asyncio.run(decisions.evaluate_turn(req, api_key="test-key", transport=retry_fn))
            self.assertEqual((res.ok, res.attempts, res.question_count, res.catalog_option_count, res.total_input_tokens),
                             (True, 2, len(req["questions"]), 59, 310), first)
            self.assertEqual(calls[0][1]["x-goog-api-key"], "test-key")

        # Two timeouts fail; 400 / safety_block / empty_answers / malformed_response do not retry
        res_to = asyncio.run(decisions.evaluate_turn(req, api_key="k", transport=lambda *_: (_ for _ in ()).throw(TimeoutError())))
        self.assertEqual((res_to.ok, res_to.attempts, res_to.error_reason), (False, 2, "timeout"))

        for resp, expected_err in (
            ((400, b'{"error":{"code":"invalid_request"}}'), "http_400"),
            ((400, b'{"error":{"code":"safety"}}'), "safety_block"),
            ((400, b'{"error":{"code":"prohibited_content"}}'), "safety_block"),
            ((200, b'{"answers": {}, "usage": null}'), "empty_answers"),
            ((200, b"{not json"), "malformed_response"),
        ):
            hits = 0

            def one_shot(*_args, r=resp):
                nonlocal hits
                hits += 1
                return r

            res = asyncio.run(decisions.evaluate_turn(req, api_key="k", transport=one_shot))
            self.assertEqual((res.ok, res.attempts, hits, res.error_reason), (False, 1, 1, expected_err))

        # Usage schema checks
        for body, exp_present, exp_tok in (
            ({"answers": {"danger_now": noul(0.01)}, "usage": None}, False, None),
            ({"answers": {"danger_now": noul(0.01)}, "usage": {"total_input_tokens": 425}}, True, 425),
            ({"answers": {"danger_now": noul(0.01)}, "usage": {"total_tokens": 425}}, False, None),
        ):
            r_u = asyncio.run(decisions.evaluate_turn(req, api_key="k", transport=lambda *_, b=body: (200, json.dumps(b).encode())))
            self.assertEqual((r_u.ok, r_u.usage_present, r_u.total_input_tokens), (True, exp_present, exp_tok))

    def test_cancelled_error_missing_key_budget_and_mocked_urlopen(self):
        req = self._req()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(decisions.evaluate_turn(req, api_key="k", transport=lambda *_: (_ for _ in ()).throw(asyncio.CancelledError())))
        with mock.patch.dict(os.environ, {}, clear=True):
            res_mk = asyncio.run(decisions.evaluate_turn(req))
            self.assertEqual((res_mk.ok, res_mk.error_reason, res_mk.attempts), (False, "missing_api_key", 0))

        started = time.perf_counter()
        res_budget = asyncio.run(decisions.evaluate_turn(req, api_key="k", attempt_timeout_sec=0.04, total_budget_sec=0.09, transport=lambda *_: (time.sleep(0.15), (200, b"{}"))[1]))
        self.assertEqual((res_budget.ok, res_budget.error_reason), (False, "timeout"))
        self.assertLess(time.perf_counter() - started, 0.5)

        captured = {}

        class _FakeResp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_a): return False
            def read(self): return json.dumps({"answers": {"danger_now": noul(0.01)}, "usage": None}).encode()

        def fake_urlopen(request_obj, timeout=None):
            captured.update(url=request_obj.full_url, method=request_obj.get_method(), headers=dict(request_obj.header_items()), body=json.loads(request_obj.data.decode()), timeout=timeout)
            return _FakeResp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res_u = asyncio.run(decisions.evaluate_turn(req, api_key="live-test-key"))
        self.assertEqual((res_u.ok, captured["url"], captured["method"], captured["headers"]["X-goog-api-key"], captured["body"]["model"]),
                         (True, decisions.DECISIONS_ENDPOINT, "POST", "live-test-key", decisions.DECISIONS_MODEL))


if __name__ == "__main__":
    unittest.main()
