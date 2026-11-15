"""Offline unit tests for the Decisions API client, schema builder, and interpreter."""

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
from pathlib import Path
from unittest import mock

from tests import APP_DIR

import audio_recorder
import copy_hi
import decisions
import engine
from models import (
    AnswerRecord,
    AnswerStatus,
    CallState,
    ChildInput,
    ChildSex,
    MaritalStatus,
    NoteText,
    Phase,
    ProfilePatch,
    SideAction,
    SubStep,
    TurnCommand,
    TurnOutcome,
)
from triage_catalog import FOLLOWUP_HI, get_spec, routable_qids

import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import calibrate_decisions  # noqa: E402
from calibrate_decisions import build_state  # noqa: E402

CASES_PATH = SCRIPTS / "decision_calibration_cases.json"
_FORBIDDEN_CONTEXT_KEYS = ("step_kind", "active_qid", "which_child", "open_capture", "main")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def noul(prob: float) -> dict:
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1.0 - prob)}


def choice(winner: str, probs: dict[str, float]) -> dict:
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": probs[winner]}


def answers(**overrides) -> dict:
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


def thresholds() -> decisions.DecisionThresholds:
    return decisions.DecisionThresholds(
        danger_now_min_yes=0.80,
        caller_stop_min_yes=0.80,
        mentions_profile_min_yes=0.65,
        mentions_diagnosis_min_yes=0.60,
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


def parsed_input(request: dict) -> dict:
    raw = request["input"][0]["text"] if isinstance(request["input"], list) else request["input"]
    return json.loads(raw)


def start(children=None, *, marital: MaritalStatus = MaritalStatus.MARRIED, name: str = "पूजा"):
    """Walk the intro with typed profile patches, exactly as a call would."""
    patches = [ProfilePatch(caller_name=name), ProfilePatch(marital_status=marital)]
    if children is not None:
        patches.append(ProfilePatch(children=children))
    state = engine.new_state()
    for patch in patches:
        state = engine.reduce_turn(state, TurnCommand(profile_patch=patch))
    return state


def kids(*rows) -> tuple[ChildInput, ...]:
    return tuple(ChildInput(age_years=age, sex=ChildSex(sex), name=name) for age, sex, name in rows)


def clear_to(state: CallState, qid: str, limit: int = 100) -> CallState:
    """Answer everything clear until `qid` is the active main question."""
    for _ in range(limit):
        if state.phase is not Phase.ASKING:
            return state
        if (state.cursor.qid, state.cursor.substep, state.cursor.hold) == (qid, SubStep.MAIN, None):
            return state
        before = state.step_token
        state = engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
        if state.step_token == before:
            raise AssertionError(f"stuck at {before} while seeking {qid}")
    raise AssertionError(f"never reached {qid}, at {state.cursor.qid}")


def step(state: CallState, outcome: str, **kwargs) -> CallState:
    return engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome(outcome), **kwargs))


class SchemaBuilderTests(unittest.TestCase):
    def test_side_qid_criteria_cover_every_routable_item(self):
        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        request = decisions.build_turn_request(state, utterance_text="नहीं")
        self.assertEqual(set(request), {"model", "input", "questions"})
        self.assertEqual(request["model"], decisions.DECISIONS_MODEL)
        self.assertIsInstance(request["input"], str)

        side = request["questions"]["side_qid"]["criteria"]
        self.assertEqual(len(side), len(routable_qids()) + 1)
        self.assertEqual(set(side), set(routable_qids()) | {"other"})
        for qid in routable_qids():
            spec = get_spec(qid)
            self.assertIn(copy_hi.question_hi_for(state.profile, spec), side[qid])
            self.assertIn(spec.need_en, side[qid])

    def test_context_always_carries_active_question_and_utterance(self):
        state = start()
        ctx = parsed_input(decisions.build_turn_request(state, utterance_text="मेरा नाम पूजा"))
        self.assertLessEqual(set(ctx), {"active_question", "last_answered_question", "caller_utterance"})
        self.assertEqual(ctx["active_question"], engine.active_prompt_hi(state))
        self.assertEqual(ctx["caller_utterance"], "मेरा नाम पूजा")
        self.assertNotIn("children_roster", ctx)

    def test_parent_question_only_on_which_child_and_followup(self):
        multi = clear_to(start(kids((8, "female", "रिया"), (5, "male", "अर्जुन"))), "birth_jaundice")
        asked = step(multi, "concern")
        self.assertIs(asked.cursor.substep, SubStep.WHICH_CHILD)
        ctx = parsed_input(decisions.build_turn_request(asked, utterance_text="बड़ी बेटी"))
        self.assertIn("यह किस बच्चे की बात है", ctx["active_question"])
        self.assertEqual(ctx["parent_question"], copy_hi.question_hi_for(multi.profile, get_spec("birth_jaundice")))

        picked = step(asked, "concern", child_ref="child_1")
        followup_ctx = parsed_input(decisions.build_turn_request(picked, utterance_text="दो दिन"))
        self.assertEqual(followup_ctx["parent_question"], get_spec("birth_jaundice").question_multi_hi)
        self.assertEqual(followup_ctx["followup_question"], FOLLOWUP_HI["jaundice_days"])

    def test_conditional_roster_child_ref_and_detail_heads(self):
        no_kids = clear_to(start(marital=MaritalStatus.UNMARRIED), "preg_thalassemia")
        request = decisions.build_turn_request(no_kids, utterance_text="नहीं")
        self.assertNotIn("child_ref", request["questions"])
        self.assertNotIn("detail_present", request["questions"])
        self.assertNotIn("पति", parsed_input(request)["active_question"])

        one = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        one_request = decisions.build_turn_request(one, utterance_text="हाँ दो दिन")
        self.assertNotIn("child_ref", one_request["questions"])
        self.assertIn("detail_present", one_request["questions"])

        three = kids((2, "male", None), (8, "female", "रिया"), (5, "male", "अर्जुन"))
        multi = clear_to(start(three), "birth_jaundice")
        multi_request = decisions.build_turn_request(multi, utterance_text="बीच वाले को")
        self.assertEqual(
            set(multi_request["questions"]["child_ref"]["criteria"]),
            {"child_1", "child_2", "child_3", "all", "other"},
        )
        roster = parsed_input(multi_request)["children_roster"]
        self.assertEqual([r["id"] for r in roster], ["child_2", "child_3", "child_1"])
        self.assertEqual([r["rank"] for r in roster], ["oldest", "middle", "youngest"])

    def test_audio_turns_send_multimodal_blocks(self):
        pcm = (b"\x01\x00\xff\x7f" * 80) + b"\x00"
        wav = audio_recorder.pcm16_to_wav_bytes(pcm, sample_rate=16000)
        self.assertTrue(wav.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav), "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.readframes(handle.getnframes()), pcm[:-1])

        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        request = decisions.build_turn_request(state, audio_wav_bytes=wav)
        self.assertEqual([block["type"] for block in request["input"]], ["text", "audio"])
        self.assertEqual(request["input"][1]["mime_type"], "audio/wav")
        self.assertEqual(base64.b64decode(request["input"][1]["data"]), wav)
        self.assertEqual(parsed_input(request)["caller_utterance"], decisions._AUDIO_ONLY_TEXT)

    def test_no_devanagari_or_internal_enums_leak_into_the_schema(self):
        schema_text = (APP_DIR / "decision_schema.json").read_text(encoding="utf-8")
        decisions_text = (APP_DIR / "decisions.py").read_text(encoding="utf-8")
        for text in (schema_text, decisions_text):
            self.assertIsNone(_DEVANAGARI.search(text))
        for token in _FORBIDDEN_CONTEXT_KEYS:
            self.assertIsNone(re.search(rf"\b{re.escape(token)}\b", schema_text), token)


class ThresholdTests(unittest.TestCase):
    def test_strict_frozen_model_rejects_out_of_range_and_nan(self):
        self.assertEqual(len(decisions.DecisionThresholds.model_fields), 13)
        for field_name, bad in (
            ("danger_now_min_yes", 0.01),
            ("danger_now_min_yes", 0.99),
            ("outcome_min_confidence", 0.0),
            ("outcome_min_confidence", 1.0),
            ("outcome_min_margin", -0.01),
            ("outcome_min_margin", 1.0),
            ("child_ref_min_confidence", float("nan")),
            ("child_ref_min_confidence", float("inf")),
        ):
            with self.subTest(field=field_name, bad=bad):
                with self.assertRaises(ValueError):
                    decisions.DecisionThresholds(**(thresholds().model_dump() | {field_name: bad}))

        with self.assertRaises(ValueError):
            decisions.DecisionThresholds(**thresholds().model_dump(), unknown_typo=0.5)
        with self.assertRaises(ValueError):
            thresholds().danger_now_min_yes = 0.5

    def test_validate_thresholds_type_guard(self):
        good = thresholds()
        self.assertIs(decisions.validate_thresholds(good), good)
        with self.assertRaises(TypeError):
            decisions.validate_thresholds({"danger_now_min_yes": 0.5})

    def test_load_thresholds_flat_wrapped_cached_and_invalid(self):
        good = thresholds()
        base = good.model_dump()
        with tempfile.TemporaryDirectory() as tmp:
            flat = Path(tmp) / "flat.json"
            flat.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(decisions.load_thresholds(flat), good)

            wrapped = Path(tmp) / "wrapped.json"
            wrapped.write_text(json.dumps({"thresholds": base}), encoding="utf-8")
            self.assertEqual(decisions.load_thresholds(wrapped), good)

            before = decisions.load_thresholds(wrapped)
            self.assertIs(decisions.load_thresholds(wrapped), before, "cache must hit by mtime")
            wrapped.write_text(json.dumps({"thresholds": base | {"side_qid_min_margin": 0.3}}), "utf-8")
            self.assertEqual(decisions.load_thresholds(wrapped).side_qid_min_margin, 0.3)

            for bad in (
                {k: v for k, v in base.items() if k != "child_ref_min_margin"},
                [1, 2, 3],
                base | {"detail_present_min_yes": 0.995},
                base | {"unknown_typo": 0.5},
                {"thresholds": base | {"unknown_typo": 0.5}},
            ):
                broken = Path(tmp) / "broken.json"
                broken.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    decisions.load_thresholds(broken)

    def test_env_var_and_default_threshold_resolution(self):
        good = thresholds()
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.json"
            custom.write_text(json.dumps(good.model_dump()), encoding="utf-8")
            with mock.patch.dict(os.environ, {"DECISION_THRESHOLDS_PATH": str(custom)}):
                self.assertEqual(decisions.load_thresholds(), good)
        self.assertEqual(decisions.load_thresholds(), decisions.load_thresholds(APP_DIR / "decision_thresholds.json"))
        with self.assertRaises(FileNotFoundError):
            decisions.load_thresholds(APP_DIR / "nonexistent_thresholds.json")


class ChoiceEvaluationTests(unittest.TestCase):
    def test_choice_rejection_reasons(self):
        missing = decisions.evaluate_choice(
            {"current_outcome": {"type": "choice", "choice": "clear", "confidence": 0.99}},
            "current_outcome",
            min_confidence=0.5,
            min_margin=0.1,
        )
        self.assertEqual((missing.accepted, missing.reject_reason), (False, "missing_answer"))

        catchall = decisions.evaluate_choice(
            {"k": choice("other", {"other": 0.9, "clear": 0.05})}, "k", min_confidence=0.5, min_margin=0.1
        )
        self.assertEqual(catchall.reject_reason, "catchall_top")

        disallowed = decisions.evaluate_choice(
            {"k": choice("odd", {"odd": 0.9, "clear": 0.05})},
            "k",
            min_confidence=0.5,
            min_margin=0.1,
            allowed_winners=frozenset({"clear"}),
        )
        self.assertEqual(disallowed.reject_reason, "disallowed_option")

        low = decisions.evaluate_choice(
            {"k": choice("clear", {"clear": 0.5, "other": 0.4})}, "k", min_confidence=0.6, min_margin=0.05
        )
        self.assertEqual(low.reject_reason, "below_min_confidence")

        tight = decisions.evaluate_choice(
            {"k": choice("clear", {"clear": 0.6, "other": 0.55})}, "k", min_confidence=0.5, min_margin=0.2
        )
        self.assertEqual(tight.reject_reason, "below_margin")

        accepted = decisions.evaluate_choice(
            {"k": choice("clear", {"clear": 0.9, "other": 0.05})}, "k", min_confidence=0.5, min_margin=0.1
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.winner, "clear")


class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.limits = thresholds()

    def interpret(self, state: CallState, raw: dict, utterance: str = "") -> decisions.TurnIntent:
        return decisions.interpret_turn_decision(
            state, raw, thresholds=self.limits, utterance_text=utterance
        )

    def test_emergency_and_stop_dominates_every_other_head(self):
        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        intent = self.interpret(
            state,
            answers(
                danger_now=noul(0.95),
                caller_stop=noul(0.91),
                side_intent=choice("add_concern", {"add_concern": 0.95, "other": 0.05}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.95, "other": 0.05}),
            ),
            "अभी बच्चे को सांस नहीं आ रही है",
        )
        self.assertTrue(intent.command.emergency)
        self.assertFalse(intent.command.caller_stop)
        self.assertEqual(intent.command.side_qid, "")
        self.assertIsNone(intent.future_qid_match)
        closed = engine.reduce_turn(state, intent.command)
        self.assertEqual((closed.phase, closed.close_reason), (Phase.CLOSING, "emergency"))
        self.assertEqual(closed.answers, state.answers)

    def test_clear_with_earlier_family_concern_adds_side_answer(self):
        base = step(step(clear_to(start(kids((4, "male", None))), "fam_cancer"), "concern",
                         utterance="मामा को ब्लड कैंसर"), "concern", utterance="मामा")
        state = clear_to(base, "fam_heart_kidney")
        intent = self.interpret(
            state,
            answers(
                current_outcome=choice("clear", {"clear": 0.92, "other": 0.04, "concern": 0.03}),
                side_intent=choice("add_concern", {"add_concern": 0.93, "other": 0.05}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.91, "other": 0.04}),
            ),
            "नहीं, पर चाची को भी कैंसर हुआ था",
        )
        self.assertEqual(intent.command.outcome, TurnOutcome.CLEAR)
        self.assertEqual(intent.command.side_qid, "fam_cancer")
        self.assertIs(intent.command.side_action, SideAction.ADD_CONCERN)
        updated = engine.reduce_turn(state, intent.command)
        self.assertEqual(updated.answers["fam_heart_kidney"].state, AnswerStatus.CLEAR)
        self.assertIn("चाची को भी कैंसर हुआ था", updated.answers["fam_cancer"].verbatim_text)

    def test_side_retract_conflict_never_leaks_onto_active_question(self):
        state = clear_to(start(kids((4, "male", None))), "fam_heart_kidney")
        intent = self.interpret(
            state,
            answers(
                current_outcome=choice("retract", {"retract": 0.82, "clear": 0.12, "other": 0.06}),
                side_intent=choice("clear_or_retract", {"clear_or_retract": 0.95, "other": 0.03}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.92, "other": 0.05}),
            ),
            "कैंसर किसी को नहीं है",
        )
        self.assertEqual(intent.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertNotIn("fam_heart_kidney", engine.reduce_turn(state, intent.command).answers)

    def test_clarify_on_side_suppresses_active_clarify(self):
        state = step(clear_to(start(kids((4, "male", None))), "preg_thalassemia"), "unknown")
        intent = self.interpret(
            state,
            answers(
                current_outcome=choice("clarify", {"clarify": 0.85, "other": 0.10}),
                side_intent=choice("clarify_other", {"clarify_other": 0.92, "other": 0.05}),
                side_qid=choice("preg_thalassemia", {"preg_thalassemia": 0.91, "other": 0.05}),
            ),
            "थैलेसीमिया क्या है?",
        )
        self.assertEqual(intent.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertIs(intent.command.side_action, SideAction.CLARIFY_MEANING)
        self.assertEqual(engine.reduce_turn(state, intent.command).cursor.qid, "preg_thalassemia")

    def test_unanswered_future_question_becomes_a_future_match(self):
        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        intent = self.interpret(
            state,
            answers(
                side_intent=choice("add_concern", {"add_concern": 0.91, "other": 0.06}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.93, "other": 0.04}),
            ),
            "पीलिया तो नहीं हुआ, चाची को कैंसर हुआ था",
        )
        self.assertEqual(intent.command.side_qid, "")
        self.assertEqual(
            intent.future_qid_match,
            {"qid": "fam_cancer", "confidence": 0.93, "act": "add_or_correct"},
        )
        self.assertEqual(intent.diagnostics["future_qid_match"], intent.future_qid_match)

    def test_low_margin_side_qid_never_writes(self):
        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        intent = self.interpret(
            state,
            answers(
                side_intent=choice("add_concern", {"add_concern": 0.90, "other": 0.10}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.72, "fam_lumps": 0.62}),
            ),
            "नहीं",
        )
        self.assertEqual(intent.command.side_qid, "")
        self.assertEqual(intent.diagnostics["side_qid"]["reject_reason"], "below_margin")

    def test_profile_extraction_is_needed_only_on_profile_or_explicit_mention(self):
        intro = engine.new_state()
        plain = answers(current_outcome=choice("other", {"other": 0.90, "clear": 0.10}))
        clarify = answers(current_outcome=choice("clarify", {"clarify": 0.92, "other": 0.08}))
        self.assertTrue(self.interpret(intro, plain).needs_profile_extraction)
        self.assertFalse(self.interpret(intro, clarify).needs_profile_extraction)
        self.assertTrue(
            self.interpret(intro, {**clarify, "mentions_profile": noul(0.90)}).needs_profile_extraction
        )
        gate = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        self.assertFalse(self.interpret(gate, plain).needs_profile_extraction)
        self.assertTrue(
            self.interpret(gate, {**plain, "mentions_profile": noul(0.90)}).needs_profile_extraction
        )

    def test_detail_present_drives_followup_answered_and_note(self):
        multi = clear_to(start(kids((9, "female", "रिया"), (5, "male", "अर्जुन"), (2, "female", "मीरा"))), "birth_jaundice")
        utterance = "हाँ बीच वाले बेटे को तीन दिन नीली लाइट में रखा था"
        intent = self.interpret(
            multi,
            answers(
                detail_present=noul(0.94),
                current_outcome=choice("concern", {"concern": 0.96, "other": 0.02}),
                child_ref=choice("child_2", {"child_2": 0.91, "other": 0.04}),
            ),
            utterance,
        )
        self.assertEqual(intent.command.child_ref, "child_2")
        self.assertTrue(intent.command.followup_answered)
        self.assertEqual(intent.command.utterance, utterance)

        bare = self.interpret(
            multi,
            answers(detail_present=noul(0.08), current_outcome=choice("concern", {"concern": 0.97})),
            "हाँ",
        )
        self.assertFalse(bare.command.followup_answered)

    def test_diagnostic_note_is_flagged_and_routine_conditions_are_not(self):
        state = clear_to(start(kids((4, "male", None))), "body_bleeding_bruise")
        flagged = self.interpret(
            state,
            answers(
                mentions_diagnosis=noul(0.88),
                current_outcome=choice("concern", {"concern": 0.94, "other": 0.06}),
            ),
            "हाँ डॉक्टर ने Hemophilia बताया है",
        )
        self.assertTrue(flagged.command.diagnostic)
        record = engine.reduce_turn(state, flagged.command).answers["body_bleeding_bruise"]
        self.assertEqual(record.verbatim_text, "हाँ डॉक्टर ने Hemophilia बताया है")
        self.assertEqual(record.mother_safe_text, "")

        routine = self.interpret(
            state,
            answers(
                mentions_diagnosis=noul(0.12),
                current_outcome=choice("concern", {"concern": 0.93, "other": 0.07}),
            ),
            "हाँ थैलेसीमिया माइनर निकला था",
        )
        self.assertFalse(routine.command.diagnostic)

    def test_diagnostics_carry_every_head_probability(self):
        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        intent = self.interpret(state, answers(), "नहीं")
        for key in ("danger_now", "caller_stop", "mentions_profile", "mentions_diagnosis", "detail_present"):
            self.assertIn(key, intent.diagnostics)
        self.assertEqual(intent.diagnostics["active_qid"], "birth_jaundice")
        self.assertEqual(intent.diagnostics["step_kind"], "main")
        self.assertIsInstance(intent.diagnostics["current_outcome"], dict)

    def test_closing_call_is_not_interpreted_into_movement(self):
        closed = CallState(phase=Phase.CLOSING, close_reason="complete")
        intent = self.interpret(closed, answers(danger_now=noul(0.99)), "अभी बच्चे को सांस नहीं आ रही")
        self.assertEqual(engine.reduce_turn(closed, intent.command).phase, Phase.CLOSING)


class TransportTests(unittest.TestCase):
    def request(self) -> dict:
        state = clear_to(start(kids((4, "male", None))), "birth_jaundice")
        return decisions.build_turn_request(state, utterance_text="नहीं")

    def test_retries_transient_failures_then_succeeds(self):
        request = self.request()
        ok_body = json.dumps(
            {"answers": {"danger_now": noul(0.02)}, "usage": {"total_input_tokens": 310}}
        ).encode()
        transients = (
            TimeoutError("slow"),
            (503, b'{"error":{"message":"busy"}}'),
            (429, b'{"error":{"message":"rate"}}'),
            (408, b"timeout"),
            ConnectionResetError("reset"),
            urllib.error.URLError("dns"),
            http.client.RemoteDisconnected("closed"),
            urllib.error.HTTPError(decisions.DECISIONS_ENDPOINT, 503, "Busy", {}, io.BytesIO(b"busy")),
        )
        for transient in transients:
            calls = []

            def transport(url, body, headers, timeout, first=transient):
                calls.append(headers)
                if len(calls) == 1:
                    if isinstance(first, BaseException):
                        raise first
                    return first
                return 200, ok_body

            result = asyncio.run(decisions.evaluate_turn(request, api_key="test-key", transport=transport))
            self.assertTrue(result.ok, transient)
            self.assertEqual(result.attempts, 2)
            self.assertEqual(result.total_input_tokens, 310)
            self.assertEqual(result.question_count, len(request["questions"]))
            self.assertEqual(calls[0]["x-goog-api-key"], "test-key")

    def test_terminal_failures_never_retry(self):
        request = self.request()
        cases = (
            (lambda *_: (400, b'{"error":{"code":"invalid_request"}}'), "http_400"),
            (lambda *_: (400, b'{"error":{"code":"safety"}}'), "safety_block"),
            (lambda *_: (200, b'{"answers": {}, "usage": null}'), "empty_answers"),
            (lambda *_: (200, b"{not json"), "malformed_response"),
        )
        for transport, expected in cases:
            hits = []
            result = asyncio.run(
                decisions.evaluate_turn(
                    request, api_key="k", transport=lambda *a, t=transport: (hits.append(1), t(*a))[1]
                )
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.error_reason, expected)
            self.assertEqual(len(hits), 1, expected)

    def test_timeout_cancel_missing_key_and_budget(self):
        request = self.request()
        timed_out = asyncio.run(
            decisions.evaluate_turn(
                request, api_key="k", transport=lambda *_: (_ for _ in ()).throw(TimeoutError())
            )
        )
        self.assertEqual((timed_out.ok, timed_out.attempts, timed_out.error_reason), (False, 2, "timeout"))

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(
                decisions.evaluate_turn(
                    request, api_key="k", transport=lambda *_: (_ for _ in ()).throw(asyncio.CancelledError())
                )
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            missing = asyncio.run(decisions.evaluate_turn(request))
        self.assertEqual((missing.ok, missing.error_reason), (False, "missing_api_key"))

        started = time.perf_counter()
        budget = asyncio.run(
            decisions.evaluate_turn(
                request,
                api_key="k",
                attempt_timeout_sec=0.04,
                total_budget_sec=0.09,
                transport=lambda *_: (time.sleep(0.15), (200, b"{}"))[1],
            )
        )
        self.assertEqual((budget.ok, budget.error_reason), (False, "timeout"))
        self.assertLess(time.perf_counter() - started, 0.6)

    def test_usage_shapes_and_mocked_urlopen_capture(self):
        request = self.request()
        for body, present, tokens in (
            ({"answers": {"danger_now": noul(0.01)}, "usage": None}, False, None),
            ({"answers": {"danger_now": noul(0.01)}, "usage": {"total_input_tokens": 425}}, True, 425),
            ({"answers": {"danger_now": noul(0.01)}, "usage": {"total_tokens": 425}}, False, None),
        ):
            result = asyncio.run(
                decisions.evaluate_turn(request, api_key="k", transport=lambda *_, b=body: (200, json.dumps(b).encode()))
            )
            self.assertTrue(result.ok)
            self.assertEqual((result.usage_present, result.total_input_tokens), (present, tokens))

        captured = {}

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"answers": {"danger_now": noul(0.01)}, "usage": None}).encode()

        def fake_urlopen(request_obj, timeout=None):
            captured.update(
                url=request_obj.full_url,
                method=request_obj.get_method(),
                headers=dict(request_obj.header_items()),
                body=json.loads(request_obj.data.decode()),
                timeout=timeout,
            )
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            live = asyncio.run(decisions.evaluate_turn(request, api_key="live-test-key"))
        self.assertTrue(live.ok)
        self.assertEqual(captured["url"], decisions.DECISIONS_ENDPOINT)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["headers"]["X-goog-api-key"], "live-test-key")
        self.assertEqual(captured["body"]["model"], decisions.DECISIONS_MODEL)


class CalibrationCaseTests(unittest.TestCase):
    def test_all_cases_build_valid_states_and_split_cleanly(self):
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 50)
        splits = [case["split"] for case in cases]
        self.assertEqual((splits.count("dev"), splits.count("holdout")), (40, 10))
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for case in cases:
            with self.subTest(case=case["id"]):
                state = build_state(case["cursor_setup"])
                self.assertIsInstance(state, CallState)
                self.assertIn(case["gold"]["current_outcome"], decisions.VALID_OUTCOMES | {"not_about_active"})

    def test_dev_cases_are_the_only_fitting_split(self):
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        dev = [c for c in cases if c["split"] == "dev"]
        self.assertEqual(len(dev), 40)
        for case in cases:
            self.assertIn(case["split"], {"dev", "holdout"})


if __name__ == "__main__":
    unittest.main()
