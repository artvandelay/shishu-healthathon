"""Behavioral tests for `intake.decisions` and `intake.extractor`: zero network calls."""

from __future__ import annotations

import asyncio
import json
import unittest

from intake import decisions, engine, extractor
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CursorStep, PackConfig, QuestionKind,
    RosterItem, SubStep, TurnOutcome,
)
from intake.pack import load_pack

OK_BODY = {"answers": {"danger_now": {"noul": 0.02}}, "usage": {"total_input_tokens": 312}}


def state_at(pack: PackConfig, qid: str, **kw) -> CallState:
    st = engine.new_state(pack)
    st.cursor = CursorStep(qid=qid, **kw)
    return st


def noul(value: float) -> dict:
    return {"noul": value}


def choice(**probs: float) -> dict:
    return {"probabilities": dict(probs)}


def transport_returning(status: int, body: dict, calls: list | None = None):
    def send(url: str, payload: bytes, headers: dict, timeout: float):
        if calls is not None:
            calls.append((url, payload, headers, timeout))
        return status, json.dumps(body).encode()
    return send


class EvaluateTurnTests(unittest.TestCase):
    """Transport, retry, safety, and telemetry — all through an injected fake."""

    def run_turn(self, transport, **kw) -> decisions.DecisionCallResult:
        request = {"model": "gemini-decision-1", "input": "x", "questions": {"a": {}, "b": {}}}
        return asyncio.run(decisions.evaluate_turn(request, api_key="k", transport=transport, **kw))

    def test_success_records_telemetry_and_reuses_answers(self):
        calls: list = []
        result = self.run_turn(transport_returning(200, OK_BODY, calls))
        self.assertTrue(result.ok)
        self.assertEqual(result.error_reason, "")
        self.assertEqual((result.attempts, result.question_count), (1, 2))
        self.assertEqual(result.total_input_tokens, 312)
        self.assertEqual(result.raw_answers, OK_BODY["answers"])
        self.assertGreater(result.request_bytes, 0)
        self.assertGreaterEqual(result.decision_ms, 0)
        url, body, headers, timeout = calls[0]
        self.assertEqual(url, decisions.DECISIONS_ENDPOINT)
        self.assertEqual(headers["x-goog-api-key"], "k")
        self.assertGreater(timeout, 0)
        self.assertEqual(json.loads(body)["questions"], {"a": {}, "b": {}})

    def test_retries_only_retryable_statuses_and_stops_at_the_cap(self):
        calls: list = []
        result = self.run_turn(transport_returning(503, {"error": {"code": "internal"}}, calls))
        self.assertFalse(result.ok)
        self.assertEqual((result.error_reason, result.attempts), ("http_503", 2))
        self.assertEqual(len(calls), 2)

    def test_rate_limit_retries_then_succeeds(self):
        seen: list = []

        def send(url, payload, headers, timeout):
            seen.append(url)
            return (429, b"{}") if len(seen) == 1 else (200, json.dumps(OK_BODY).encode())

        result = self.run_turn(send)
        self.assertTrue(result.ok)
        self.assertEqual((result.attempts, len(seen)), (2, 2))

    def test_safety_block_is_terminal_and_unretried(self):
        calls: list = []
        body = {"error": {"code": "prohibited_content", "message": "blocked"}}
        result = self.run_turn(transport_returning(400, body, calls))
        self.assertFalse(result.ok)
        self.assertEqual((result.error_reason, result.attempts, len(calls)), ("safety_block", 1, 1))

    def test_client_errors_do_not_retry(self):
        result = self.run_turn(transport_returning(404, {"error": {"code": "not_found"}}))
        self.assertEqual((result.error_reason, result.attempts), ("http_404", 1))

    def test_timeout_and_connection_faults_are_retryable(self):
        def boom(url, payload, headers, timeout):
            raise TimeoutError("slow")

        result = self.run_turn(boom, max_attempts=2)
        self.assertEqual((result.error_reason, result.attempts), ("timeout", 2))

    def test_async_transport_is_awaited(self):
        async def send(url, payload, headers, timeout):
            return 200, json.dumps(OK_BODY).encode()

        result = self.run_turn(send)
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 1)

    def test_missing_api_key_fails_without_any_call(self):
        calls: list = []
        request = {"model": "m", "input": "x", "questions": {}}
        result = asyncio.run(decisions.evaluate_turn(
            request, api_key="  ", transport=transport_returning(200, OK_BODY, calls)))
        self.assertEqual((result.ok, result.error_reason, result.attempts), (False, "missing_api_key", 0))
        self.assertEqual(calls, [])

    def test_empty_answers_payload_is_a_failure(self):
        result = self.run_turn(transport_returning(200, {"answers": {}}))
        self.assertEqual((result.ok, result.error_reason), (False, "empty_answers"))


class BuildRequestTests(unittest.TestCase):
    """The four head groups, the multimodal input, and the compact context."""

    @classmethod
    def setUpClass(cls):
        cls.shishu = load_pack("shishu-hi")
        cls.fever = load_pack("fever-opd-hi")

    def test_profile_step_carries_core_heads_and_a_flag(self):
        request = decisions.build_turn_request(self.shishu, engine.new_state(self.shishu))
        heads = request["questions"]
        self.assertEqual(request["model"], "gemini-decision-1")
        for head in ("danger_now", "caller_stop", "mentions_profile", "redact_patient_note"):
            self.assertIn(head, heads)
        self.assertIn("active_outcome", heads)
        self.assertIn("flag_confirmed_hemoglobinopathy", heads)
        self.assertNotIn("turn_incomplete", heads)
        self.assertEqual(heads["flag_confirmed_hemoglobinopathy"]["type"], "noul")

    def test_turn_incomplete_is_opt_in(self):
        state = engine.new_state(self.shishu)
        request = decisions.build_turn_request(self.shishu, state, include_turn_incomplete=True)
        self.assertIn("turn_incomplete", request["questions"])

    def test_single_choice_renders_option_criteria_plus_other(self):
        state = state_at(self.fever, "fever_duration_bucket")
        heads = decisions.build_turn_request(self.fever, state)["questions"]
        value = heads["choice_value"]
        self.assertEqual(value["type"], "choice")
        self.assertEqual(
            set(value["criteria"]),
            {"lt_3_days", "days_3_to_7", "gt_7_days", "other"},
        )
        self.assertIn("gt_7_days", value["criteria"])
        self.assertIn("7 दिन", self.fever.by_id["fever_duration_bucket"].options["gt_7_days"].label)

    def test_multi_choice_emits_one_noul_head_per_option(self):
        state = state_at(self.fever, "associated_symptoms")
        heads = decisions.build_turn_request(self.fever, state)["questions"]
        options = self.fever.by_id["associated_symptoms"].options
        self.assertEqual({f"opt_{k}" for k in options}, {k for k in heads if k.startswith("opt_")})
        self.assertEqual(heads["opt_none"]["type"], "noul")
        self.assertIn("in response to active_question", heads["opt_chills"]["instructions"])

    def test_score_head_uses_the_pack_rubric(self):
        state = state_at(self.fever, "fever_severity")
        heads = decisions.build_turn_request(self.fever, state)["questions"]
        self.assertEqual(heads["score_value"]["type"], "score")
        self.assertEqual(heads["score_value"]["criteria"], list(self.fever.by_id["fever_severity"].score_rubric))

    def test_detail_present_follows_whether_a_followup_is_pending(self):
        without = state_at(self.fever, "fever_nature")
        self.assertNotIn("detail_present", decisions.build_turn_request(self.fever, without)["questions"])
        with_followup = state_at(self.fever, "breathing_or_chest")
        heads = decisions.build_turn_request(self.fever, with_followup)["questions"]
        self.assertIn("detail_present", heads)
        ctx = json.loads(decisions.build_turn_request(self.fever, with_followup)["input"])
        self.assertEqual(ctx["followup_question"], self.fever.followups["breathing_detail"])

    def test_substep_carries_the_parent_question(self):
        st = state_at(self.fever, "breathing_or_chest", substep=SubStep.FOLLOWUP, followup_idx=0)
        ctx = json.loads(decisions.build_turn_request(self.fever, st)["input"])
        question = self.fever.by_id["breathing_or_chest"]
        self.assertEqual(ctx["parent_question"], engine.question_say(self.fever, st, question))

    def test_audio_attaches_text_and_audio_even_with_a_transcript(self):
        state = engine.new_state(self.shishu)
        request = decisions.build_turn_request(
            self.shishu, state, utterance_text="हाँ जी", audio_wav_bytes=b"RIFFdata")
        parts = request["input"]
        self.assertEqual([p["type"] for p in parts], ["text", "audio"])
        self.assertEqual(parts[1]["mime_type"], "audio/wav")
        ctx = json.loads(parts[0]["text"])
        self.assertEqual(ctx["caller_utterance"], "हाँ जी")
        self.assertEqual(ctx["active_question"], engine.active_prompt(self.shishu, state))

    def test_text_only_and_audio_only_contexts(self):
        state = engine.new_state(self.shishu)
        self.assertIsInstance(decisions.build_turn_request(self.shishu, state, utterance_text="x")["input"], str)
        audio_only = decisions.build_turn_request(self.shishu, state, audio_wav_bytes=b"aa")
        self.assertEqual(json.loads(audio_only["input"][0]["text"])["caller_utterance"],
                         "[spoken audio attached]")

    def test_roster_heads_appear_only_for_two_or_more_children(self):
        st = engine.new_state(self.shishu)
        st.roster = [RosterItem(id="item_1", age_years=6, age_label="6 साल", tag="female", name="Riya")]
        single = decisions.build_turn_request(self.shishu, st)["questions"]
        self.assertNotIn("item_ref", single)
        st.roster.append(RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="male"))
        heads = decisions.build_turn_request(self.shishu, st)["questions"]
        self.assertIn("item_ref", heads)
        self.assertIn("roster", json.loads(decisions.build_turn_request(self.shishu, st)["input"]))
        self.assertEqual(set(heads["item_ref"]["criteria"]), {"item_1", "item_2", "all", "other"})

    def test_router_heads_need_two_open_questions_and_a_routable_step(self):
        st = state_at(self.fever, "fever_duration_bucket")
        heads = decisions.build_turn_request(self.fever, st)["questions"]
        self.assertIn("next_qid", heads)
        self.assertIn("jump_topic", heads)
        self.assertIn("other", heads["next_qid"]["criteria"])
        self.assertNotIn("fever_duration_bucket", heads["next_qid"]["criteria"])

    def test_stated_heads_cover_every_other_routable_gate(self):
        st = state_at(self.fever, "fever_nature")
        heads = decisions.build_turn_request(self.fever, st)["questions"]
        stated = {k for k in heads if k.startswith("stated_")}
        expected = {f"stated_{q.id}" for q in self.fever.questions
                    if q.is_routable and q.id != "fever_nature"
                    and q.kind in (QuestionKind.GATE, QuestionKind.ROUTING_GATE, QuestionKind.OPEN_CAPTURE)}
        self.assertEqual(stated, expected)
        self.assertIn("unmentioned", heads["stated_breathing_or_chest"]["criteria"])


class InterpretTests(unittest.TestCase):
    """Threshold gating in both directions, per question kind."""

    @classmethod
    def setUpClass(cls):
        cls.fever = load_pack("fever-opd-hi")
        cls.shishu = load_pack("shishu-hi")

    def interpret(self, pack, state, answers, **kw) -> decisions.TurnIntent:
        return decisions.interpret_turn_decision(pack, state, answers, **kw)

    def test_emergency_wins_over_every_other_head(self):
        st = engine.new_state(self.shishu)
        intent = self.interpret(self.shishu, st, {
            "danger_now": noul(0.95), "caller_stop": noul(0.99),
            "active_outcome": choice(concern=0.95, other=0.01),
        }, utterance_text="बच्चा साँस नहीं ले रहा")
        self.assertTrue(intent.command.emergency)
        self.assertFalse(intent.command.caller_stop)
        self.assertIs(intent.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual(intent.command.utterance, "बच्चा साँस नहीं ले रहा")

    def test_single_choice_above_thresholds_answers_with_the_winner(self):
        st = state_at(self.fever, "fever_duration_bucket")
        intent = self.interpret(self.fever, st, {
            "choice_value": choice(gt_7_days=0.82, days_3_to_7=0.1, lt_3_days=0.05, other=0.03)})
        self.assertIs(intent.command.outcome, TurnOutcome.ANSWERED)
        self.assertEqual(intent.command.value, "gt_7_days")

    def test_single_choice_below_margin_falls_back_to_active_outcome(self):
        st = state_at(self.fever, "fever_duration_bucket")
        intent = self.interpret(self.fever, st, {
            "choice_value": choice(gt_7_days=0.4, lt_3_days=0.38, other=0.2),
            "active_outcome": choice(clarify=0.9, other=0.02)})
        self.assertIs(intent.command.outcome, TurnOutcome.CLARIFY)
        self.assertIsNone(intent.command.value)

    def test_catchall_winner_is_never_accepted(self):
        st = state_at(self.fever, "fever_duration_bucket")
        intent = self.interpret(self.fever, st, {"choice_value": choice(other=0.97, gt_7_days=0.01)})
        self.assertIs(intent.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE)

    def test_multi_choice_collects_every_option_above_the_yes_gate(self):
        st = state_at(self.fever, "associated_symptoms")
        intent = self.interpret(self.fever, st, {
            "opt_chills": noul(0.91), "opt_body_ache": noul(0.77),
            "opt_rash": noul(0.2), "opt_none": noul(0.01)})
        self.assertIs(intent.command.outcome, TurnOutcome.ANSWERED)
        self.assertEqual(intent.command.value, ["chills", "body_ache"])

    def test_multi_choice_falls_back_when_nothing_clears(self):
        st = state_at(self.fever, "associated_symptoms")
        intent = self.interpret(self.fever, st, {
            "opt_chills": noul(0.3), "active_outcome": choice(clear=0.93, other=0.02)})
        self.assertIs(intent.command.outcome, TurnOutcome.CLEAR)

    def test_score_reads_value_and_level_with_confidence(self):
        st = state_at(self.fever, "fever_severity")
        intent = self.interpret(self.fever, st, {"score_value": {
            "score": 2.5, "confidence": 0.8, "probabilities": {"0": 0.05, "1": 0.1, "2": 0.6, "3": 0.25}}})
        self.assertIs(intent.command.outcome, TurnOutcome.ANSWERED)
        self.assertEqual(intent.command.value, {"score": 2.5, "level": 2})

    def test_low_confidence_score_falls_back(self):
        st = state_at(self.fever, "fever_severity")
        intent = self.interpret(self.fever, st, {"score_value": {"score": 3.0, "confidence": 0.1}})
        self.assertIs(intent.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertIsNone(intent.command.value)

    def test_stated_heads_merge_only_above_their_stricter_gate(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {
            "stated_recent_travel": choice(clear=0.93, unmentioned=0.03, concern=0.02, clarify=0.02),
            "stated_vomiting_or_diarrhea": choice(concern=0.6, unmentioned=0.3, clear=0.08, clarify=0.02),
        })
        self.assertEqual(intent.command.stated, {"recent_travel": TurnOutcome.CLEAR})

    def test_open_capture_is_not_double_written_outside_a_closing_or_clarifying_turn(self):
        st = state_at(self.fever, "fever_nature")
        answers = {"stated_close_anything_else": choice(concern=0.95, unmentioned=0.02)}
        self.assertNotIn("close_anything_else", self.interpret(self.fever, st, answers).command.stated)
        clarifying = dict(answers, active_outcome=choice(clarify=0.9, other=0.02))
        intent = self.interpret(self.fever, st, clarifying)
        self.assertEqual(intent.command.stated["close_anything_else"], TurnOutcome.CONCERN)

    def test_semantic_flags_use_the_pack_min_yes(self):
        st = engine.new_state(self.shishu)
        raised = self.interpret(self.shishu, st, {"flag_confirmed_hemoglobinopathy": noul(0.9)})
        self.assertEqual(raised.command.semantic_flags, ("confirmed_hemoglobinopathy",))
        borderline = self.interpret(self.shishu, st, {"flag_confirmed_hemoglobinopathy": noul(0.5)})
        self.assertEqual(borderline.command.semantic_flags, ())

    def test_already_set_flags_are_not_re_raised(self):
        st = engine.new_state(self.shishu)
        st.semantic_flags = ["confirmed_hemoglobinopathy"]
        intent = self.interpret(self.shishu, st, {"flag_confirmed_hemoglobinopathy": noul(0.99)})
        self.assertEqual(intent.command.semantic_flags, ())

    def test_next_qid_probabilities_drop_the_catchall_and_non_finite_values(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {"next_qid": {"probabilities": {
            "recent_travel": 0.6, "other": 0.3, "received_treatment": float("nan")}}})
        self.assertEqual(intent.command.next_qid_probs, {"recent_travel": 0.6})

    def test_jump_topic_and_detail_present_are_gated_by_step_kind(self):
        main = state_at(self.fever, "breathing_or_chest")
        head = {"jump_topic": noul(0.95), "active_outcome": choice(concern=0.95, other=0.01)}
        intent = self.interpret(self.fever, main, head)
        self.assertTrue(intent.command.jump_topic)
        self.assertFalse(intent.command.followup_answered)
        follow = state_at(self.fever, "breathing_or_chest", substep=SubStep.FOLLOWUP)
        answered = self.interpret(self.fever, follow, {
            "detail_present": noul(0.95), "active_outcome": choice(concern=0.95, other=0.01)})
        self.assertFalse(answered.command.followup_answered)
        self.assertTrue(answered.command.redacted is False)

    def test_redact_and_turn_incomplete_heads_map_onto_the_command(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {
            "redact_patient_note": noul(0.9), "turn_incomplete": noul(0.8)})
        self.assertTrue(intent.command.redacted)
        self.assertTrue(intent.command.turn_incomplete)

    def test_profile_step_needs_extraction_unless_clarified(self):
        st = engine.new_state(self.shishu)
        asked = self.interpret(self.shishu, st, {"active_outcome": choice(unknown=0.95, other=0.01)})
        self.assertTrue(asked.needs_profile_extraction)
        clarified = self.interpret(self.shishu, st, {"active_outcome": choice(clarify=0.95, other=0.01)})
        self.assertFalse(clarified.needs_profile_extraction)

    def test_mentions_profile_forces_extraction_on_any_step(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {"mentions_profile": noul(0.9)})
        self.assertTrue(intent.needs_profile_extraction)

    def test_diagnostics_keep_the_raw_head_probabilities(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {"danger_now": noul(0.31), "jump_topic": noul(0.44)})
        self.assertEqual(intent.diagnostics["danger_now"], 0.31)
        self.assertEqual(intent.diagnostics["jump_topic"], 0.44)
        self.assertEqual(intent.diagnostics["step_kind"], "main")
        self.assertEqual(intent.diagnostics["active_qid"], "fever_nature")

    def test_diagnostics_carry_a_compact_top2_and_the_hot_nouls(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {
            "danger_now": noul(0.31), "caller_stop": noul(0.05), "jump_topic": noul(0.44),
            "active_outcome": choice(concern=0.62, clear=0.2137, unknown=0.1, clarify=0.05, other=0.02)})
        self.assertEqual(intent.diagnostics["top2"], "concern:0.62,clear:0.214")
        self.assertEqual(intent.diagnostics["noul_hot"], "danger_now:0.31,jump_topic:0.44",
                         "only heads at or above 0.25, in _NOUL order, rounded to two places")
        for key in ("active_qid", "step_kind", "outcome", "stated", "next_qid_probs", "semantic_flags", "heads", "danger_now", "jump_topic"):
            self.assertIn(key, intent.diagnostics, f"{key} must survive the enrichment")

    def test_an_empty_turn_still_carries_both_scalars_as_empty_strings(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {})
        self.assertEqual(intent.diagnostics["top2"], "")
        self.assertEqual(intent.diagnostics["noul_hot"], "")

    def test_caller_stop_is_suppressed_only_by_an_emergency(self):
        st = engine.new_state(self.shishu)
        intent = self.interpret(self.shishu, st, {"caller_stop": noul(0.93)})
        self.assertTrue(intent.command.caller_stop)
        self.assertFalse(intent.command.emergency)

    def test_weak_out_of_turn_concern_cannot_flip_an_already_clear_answer(self):
        st = state_at(self.shishu, "body_bruise_bleed")
        st.answers["body_blue"] = AnswerRecord(state=AnswerStatus.CLEAR)
        st.answers["birth_cord_3weeks"] = AnswerRecord(state=AnswerStatus.CLEAR)
        noisy = {
            "stated_body_blue": choice(concern=0.82, unmentioned=0.1, clear=0.05, clarify=0.03),
            "stated_birth_cord_3weeks": choice(concern=0.84, unmentioned=0.08, clear=0.05, clarify=0.03),
        }
        self.assertEqual(self.interpret(self.shishu, st, noisy).command.stated, {})

    def test_strong_out_of_turn_concern_still_revises_a_clear_answer(self):
        st = state_at(self.shishu, "body_bruise_bleed")
        st.answers["body_blue"] = AnswerRecord(state=AnswerStatus.CLEAR)
        intent = self.interpret(self.shishu, st, {
            "stated_body_blue": choice(concern=0.95, unmentioned=0.02, clear=0.02, clarify=0.01)})
        self.assertEqual(intent.command.stated, {"body_blue": TurnOutcome.CONCERN})

    def test_clear_answer_still_takes_a_retraction_and_a_clarification_at_the_normal_gate(self):
        st = state_at(self.shishu, "body_bruise_bleed")
        st.answers["birth_cord_3weeks"] = AnswerRecord(state=AnswerStatus.CLEAR)
        retracted = self.interpret(self.shishu, st, {
            "stated_birth_cord_3weeks": choice(clear=0.85, unmentioned=0.1, concern=0.03, clarify=0.02)})
        self.assertEqual(retracted.command.stated["birth_cord_3weeks"], TurnOutcome.CLEAR)
        clarified = self.interpret(self.shishu, st, {
            "stated_birth_cord_3weeks": choice(clarify=0.86, unmentioned=0.08, concern=0.04, clear=0.02)})
        self.assertEqual(clarified.command.stated["birth_cord_3weeks"], TurnOutcome.CLARIFY)

    def test_unresolved_and_unknown_priors_keep_the_pack_stated_gate(self):
        for prior in (None, AnswerRecord(state=AnswerStatus.UNKNOWN), AnswerRecord(state=AnswerStatus.CONCERN)):
            with self.subTest(prior=None if prior is None else prior.state.value):
                st = state_at(self.shishu, "body_bruise_bleed")
                if prior is not None:
                    st.answers["body_blue"] = prior
                intent = self.interpret(self.shishu, st, {
                    "stated_body_blue": choice(concern=0.85, unmentioned=0.09, clear=0.03, clarify=0.03)})
                self.assertEqual(intent.command.stated, {"body_blue": TurnOutcome.CONCERN})

    def test_engine_applies_the_guarded_flip_end_to_end(self):
        st = state_at(self.shishu, "body_bruise_bleed")
        st.answers["body_blue"] = AnswerRecord(state=AnswerStatus.CLEAR)
        weak = self.interpret(self.shishu, st, {
            "active_outcome": choice(clear=0.95, other=0.02),
            "stated_body_blue": choice(concern=0.82, unmentioned=0.1, clear=0.05, clarify=0.03),
        }, utterance_text="नील पड़ जाते हैं")
        kept = engine.reduce_turn(self.shishu, st, weak.command)
        self.assertIs(kept.answers["body_blue"].state, AnswerStatus.CLEAR)


class ExtractorTests(unittest.TestCase):
    """Schema construction, strict validation, and failure mapping — no live client."""

    @classmethod
    def setUpClass(cls):
        cls.shishu = load_pack("shishu-hi")
        cls.fever = load_pack("fever-opd-hi")

    def run_extract(self, pack, payload, **kw) -> extractor.ProfileExtractionResult:
        async def gen(**_kwargs):
            return payload

        return asyncio.run(extractor.extract_profile_entities(
            pack, utterance_text="मेरा नाम सुनीता है", generator=gen, **kw))

    def test_shishu_schema_has_slots_marital_enum_and_roster(self):
        schema = extractor.build_response_schema(self.shishu)
        self.assertEqual(set(schema.properties), {"caller_name", "marital_status", "roster_count", "roster"})
        self.assertEqual(list(schema.properties["marital_status"].enum),
                         list(self.shishu.by_id["intro_married"].options))
        roster = schema.properties["roster"]
        self.assertIs(roster.type, extractor.types.Type.ARRAY)
        self.assertEqual(set(roster.items.properties), {"age_years", "tag", "name"})
        self.assertEqual(sorted(schema.required), ["caller_name", "marital_status", "roster", "roster_count"])

    def test_fever_schema_has_no_roster_or_marital_field(self):
        schema = extractor.build_response_schema(self.fever)
        self.assertEqual(set(schema.properties), {"patient_name", "patient_age"})

    def test_valid_payload_becomes_a_typed_patch(self):
        result = self.run_extract(self.shishu, {
            "caller_name": "  सुनीता  ", "marital_status": "married",
            "roster_count": 2, "roster": [
                {"age_years": 6, "tag": "female", "name": "Riya"},
                {"age_years": 0.5, "tag": "male", "name": None}]})
        self.assertTrue(result.ok)
        self.assertEqual(result.error_reason, "")
        self.assertEqual(result.patch.slots["caller_name"], "सुनीता")
        self.assertEqual(result.patch.roster_count, 2)
        self.assertEqual([r.age_years for r in result.patch.roster], [6.0, 0.5])
        self.assertEqual(result.patch.roster[0].name, "Riya")

    def test_null_fields_are_left_unstated(self):
        result = self.run_extract(self.fever, {"patient_name": None, "patient_age": None})
        self.assertTrue(result.ok)
        self.assertTrue(result.patch.is_empty)

    def test_unexpected_keys_are_rejected(self):
        result = self.run_extract(self.fever, {"patient_name": "x", "diagnosis": "thal"})
        self.assertEqual((result.ok, result.error_reason), (False, "invalid_schema"))

    def test_invalid_age_and_count_are_rejected(self):
        bad_age = self.run_extract(self.shishu, {"roster": [{"age_years": -1, "tag": "male"}]})
        self.assertEqual(bad_age.error_reason, "invalid_schema")
        bad_count = self.run_extract(self.shishu, {"roster_count": -2})
        self.assertEqual(bad_count.error_reason, "invalid_schema")

    def test_json_text_output_is_parsed(self):
        async def gen(**_kwargs):
            return type("Out", (), {"text": '{"patient_age": 4}'})()

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="chaar saal", generator=gen))
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.slots, {"patient_age": "4"})

    def test_malformed_json_maps_to_invalid_json(self):
        async def gen(**_kwargs):
            return "not json"

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=gen))
        self.assertEqual((result.ok, result.error_reason), (False, "invalid_json"))

    def test_timeout_and_provider_errors_are_reported(self):
        async def slow(**_kwargs):
            await asyncio.sleep(0.2)

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=slow, attempt_timeout_sec=0.01))
        self.assertEqual(result.error_reason, "timeout")

        async def broken(**_kwargs):
            raise RuntimeError("upstream 500")

        failed = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=broken))
        self.assertEqual((failed.ok, failed.error_reason), (False, "RuntimeError"))

    def test_empty_input_and_missing_key_short_circuit(self):
        empty = asyncio.run(extractor.extract_profile_entities(self.fever))
        self.assertEqual(empty.error_reason, "empty_input")
        no_key = asyncio.run(extractor.extract_profile_entities(self.fever, utterance_text="x", api_key=" "))
        self.assertEqual(no_key.error_reason, "missing_api_key")

    def test_audio_only_turn_still_reaches_the_generator(self):
        seen: dict = {}

        async def gen(**kwargs):
            seen.update(kwargs)
            return {"patient_name": "Aman"}

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, audio_wav_bytes=b"RIFFdata", generator=gen))
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.slots, {"patient_name": "Aman"})
        self.assertEqual(seen["config"].temperature, 0.0)
        self.assertEqual(seen["config"].response_mime_type, "application/json")
        self.assertEqual(seen["model"], extractor.DEFAULT_MODEL)
        self.assertEqual(len(seen["contents"]), 2)

    def test_model_override_is_used(self):
        seen: dict = {}

        async def gen(**kwargs):
            seen.update(kwargs)
            return {}

        asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=gen, model="gemini-3.5-flash-lite"))
        self.assertEqual(seen["model"], "gemini-3.5-flash-lite")

    def test_blocking_generator_runs_off_the_event_loop(self):
        def gen(**_kwargs):
            return {"patient_name": "Sync"}

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=gen))
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.slots, {"patient_name": "Sync"})

    def test_system_instruction_comes_from_the_pack(self):
        seen: dict = {}

        async def gen(**kwargs):
            seen.update(kwargs)
            return {}

        asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=gen))
        self.assertEqual(seen["config"].system_instruction, self.fever.extractor_prompt)


if __name__ == "__main__":
    unittest.main()
