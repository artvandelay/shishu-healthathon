"""Behavioral tests for `intake.decisions` and `intake.extractor`: zero network calls."""

from __future__ import annotations

import asyncio
import json
import unittest

from google.genai import types

from intake import decisions, engine, extractor
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CursorStep, PackConfig,
    RosterItem, TurnOutcome,
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

    def test_timeout_and_connection_faults_are_retryable(self):
        def boom(url, payload, headers, timeout):
            raise TimeoutError("slow")

        result = self.run_turn(boom, max_attempts=2)
        self.assertEqual((result.error_reason, result.attempts), ("timeout", 2))


class BuildRequestTests(unittest.TestCase):
    """The four head groups, the multimodal input, and the compact context."""

    @classmethod
    def setUpClass(cls):
        cls.shishu = load_pack("shishu-hi")
        cls.fever = load_pack("fever-opd-hi")

    def test_multi_choice_emits_one_noul_head_per_option(self):
        state = state_at(self.fever, "associated_symptoms")
        heads = decisions.build_turn_request(self.fever, state)["questions"]
        options = self.fever.by_id["associated_symptoms"].options
        self.assertEqual({f"opt_{k}" for k in options}, {k for k in heads if k.startswith("opt_")})
        self.assertEqual(heads["opt_none"]["type"], "noul")
        self.assertIn("in response to active_question", heads["opt_chills"]["instructions"])

    def test_detail_present_follows_whether_a_followup_is_pending(self):
        without = state_at(self.fever, "fever_nature")
        self.assertNotIn("detail_present", decisions.build_turn_request(self.fever, without)["questions"])
        with_followup = state_at(self.fever, "breathing_or_chest")
        heads = decisions.build_turn_request(self.fever, with_followup)["questions"]
        self.assertIn("detail_present", heads)
        ctx = json.loads(decisions.build_turn_request(self.fever, with_followup)["input"])
        self.assertEqual(ctx["followup_question"], self.fever.followups["breathing_detail"])

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

    def test_a_bridge_step_still_offers_the_stated_heads_and_the_router(self):
        # Bug: `router = kind not in ("profile", "bridge")` muted every routing and out-of-turn head on a bridge
        # step, whose whole script invites the caller to speak freely ("बेझिझक खुलकर बोलकर बताइए"). In fever-opd-hi
        # patient_name/patient_age are profile questions (is_routable False), so `answered` was False too and the
        # caller's volunteered breathing complaint had neither a stated_ head nor a next_qid head to land in.
        st = state_at(self.fever, "intro_bridge")
        st.profile = {"patient_name": "रीटा", "patient_age": 4}
        heads = decisions.build_turn_request(self.fever, st)["questions"]
        self.assertIn("stated_breathing_or_chest", heads)
        self.assertNotIn("stated_close_anything_else", heads)
        self.assertIn("next_qid", heads)
        self.assertIn("jump_topic", heads)

    def test_the_context_carries_the_profile_and_a_lone_child_without_item_ref(self):
        # Bug: build_turn_request never sent state.profile, and sent state.roster only for 2+ children, so
        # mentions_profile could not see what was already stored when a caller corrected a detail or said
        # "मेरे ट्विंस बच्चे हैं, मुझे लगता नहीं आप वो सुन पाए".
        st = state_at(self.shishu, "intro_children")
        st.profile = {"caller_name": "रीटा"}
        st.roster = [RosterItem(id="item_1", age_years=3, age_label="3 साल", tag="female", name="मीरा")]
        req = decisions.build_turn_request(self.shishu, st)
        self.assertEqual(json.loads(req["input"])["profile"], {"caller_name": "रीटा"})
        self.assertEqual(json.loads(req["input"])["roster"], [
            {"id": "item_1", "rank": "only", "age_label": "3 साल", "tag": "female", "name": "मीरा"}])
        self.assertNotIn("item_ref", req["questions"], "one child has nobody to choose between")

        blank = state_at(self.shishu, "intro_children")
        blank.profile = {"caller_name": "", "marital_status": None}
        self.assertNotIn("profile", json.loads(decisions.build_turn_request(self.shishu, blank)["input"]),
                         "an all-blank profile is not a stored fact")

    def test_the_pack_names_the_roster_key_the_context_actually_sends(self):
        # Bug: decisions.yaml told the model "children_roster" while ctx sent "roster", and its mentions_profile
        # instructions only counted a stated or corrected detail, so "मेरे ट्विंस बच्चे हैं, आप वो सुन नहीं पाए"
        # (twins already on record, never restated) could not be read as a profile mention at all.
        text = (self.shishu.root / "decisions.yaml").read_text(encoding="utf-8")
        self.assertNotIn("children_roster", text)
        self.assertIn("roster", text)
        self.assertIn("twins", text)
        self.assertIn("multiples", text)
        self.assertEqual(self.shishu.decisions["heads"]["mentions_profile"]["type"], "noul")


class InterpretTests(unittest.TestCase):
    """Threshold gating in both directions, per question kind."""

    @classmethod
    def setUpClass(cls):
        cls.fever = load_pack("fever-opd-hi")
        cls.shishu = load_pack("shishu-hi")

    def interpret(self, pack, state, answers, **kw) -> decisions.TurnIntent:
        return decisions.interpret_turn_decision(pack, state, answers, **kw)

    def test_a_quiet_answer_split_between_clear_and_unknown_is_read_as_the_pair(self):
        st = state_at(self.fever, "fever_nature")
        pair = self.interpret(self.fever, st, {"active_outcome": choice(clear=0.50, unknown=0.45, concern=0.03, other=0.02)})
        self.assertIs(pair.command.outcome, TurnOutcome.CLEAR, "0.95 of the quiet pair beats the 0.2297 margin")
        flipped = self.interpret(self.fever, st, {"active_outcome": choice(unknown=0.51, clear=0.44, concern=0.03, other=0.02)})
        self.assertIs(flipped.command.outcome, TurnOutcome.UNKNOWN, "the pair's own winner is returned")
        noisy = self.interpret(self.fever, st, {"active_outcome": choice(concern=0.50, clear=0.45, other=0.05)})
        self.assertIs(noisy.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE, "only clear+unknown pool, never clear+concern")
        thin = self.interpret(self.fever, st, {"active_outcome": choice(clear=0.24, unknown=0.22, concern=0.20, other=0.20)})
        self.assertIs(thin.command.outcome, TurnOutcome.NOT_ABOUT_ACTIVE, "a pair under 0.5196 is still too weak to read")

    def test_multi_choice_collects_every_option_above_the_yes_gate(self):
        st = state_at(self.fever, "associated_symptoms")
        intent = self.interpret(self.fever, st, {
            "opt_chills": noul(0.91), "opt_body_ache": noul(0.77),
            "opt_rash": noul(0.2), "opt_none": noul(0.01)})
        self.assertIs(intent.command.outcome, TurnOutcome.ANSWERED)
        self.assertEqual(intent.command.value, ["chills", "body_ache"])

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

    def test_jump_topic_is_honoured_on_a_bridge_step(self):
        # Bug: interpret_turn_decision gated jump_topic on `kind not in ("profile", "bridge")`, so even when the
        # router head fired on the bridge the command dropped it.
        st = state_at(self.fever, "intro_bridge")
        intent = self.interpret(self.fever, st, {"jump_topic": noul(0.93)})
        self.assertTrue(intent.command.jump_topic)
        profile_step = state_at(self.fever, "patient_name")
        self.assertFalse(self.interpret(self.fever, profile_step, {"jump_topic": noul(0.93)}).command.jump_topic,
                         "a profile slot still owns the turn")

    def test_an_open_capture_stated_head_is_never_sent_and_never_written(self):
        # Bug: OPEN_CAPTURE sat in _STATED_KINDS, so every turn paid for a stated_close_anything_else head that
        # _stated_writes then threw away unless the turn was closing or clarifying.
        st = state_at(self.fever, "fever_nature")
        answers = {"stated_close_anything_else": choice(concern=0.95, unmentioned=0.02)}
        self.assertNotIn("close_anything_else", self.interpret(self.fever, st, answers).command.stated)
        clarifying = dict(answers, active_outcome=choice(clarify=0.9, other=0.02))
        self.assertNotIn("close_anything_else", self.interpret(self.fever, st, clarifying).command.stated)
        self.assertNotIn("stated_close_anything_else", decisions.build_turn_request(self.fever, st)["questions"])

    def test_next_qid_probabilities_drop_the_catchall_and_non_finite_values(self):
        st = state_at(self.fever, "fever_nature")
        intent = self.interpret(self.fever, st, {"next_qid": {"probabilities": {
            "recent_travel": 0.6, "other": 0.3, "received_treatment": float("nan")}}})
        self.assertEqual(intent.command.next_qid_probs, {"recent_travel": 0.6})

    def test_strong_out_of_turn_concern_still_revises_a_clear_answer(self):
        st = state_at(self.shishu, "body_bruise_bleed")
        st.answers["body_blue"] = AnswerRecord(state=AnswerStatus.CLEAR)
        intent = self.interpret(self.shishu, st, {
            "stated_body_blue": choice(concern=0.95, unmentioned=0.02, clear=0.02, clarify=0.01)})
        self.assertEqual(intent.command.stated, {"body_blue": TurnOutcome.CONCERN})


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

    def test_valid_payload_becomes_a_typed_patch(self):
        result = self.run_extract(self.shishu, {
            "caller_name": "  सुनीता  ", "marital_status": "married",
            "roster_complete": True, "roster": [
                {"id": None, "age_years": 6, "tag": "female", "name": "Riya"},
                {"id": "item_1", "age_years": 0.5, "tag": "male", "name": None}]})
        self.assertTrue(result.ok)
        self.assertEqual(result.error_reason, "")
        self.assertEqual(result.patch.slots["caller_name"], "सुनीता")
        self.assertIs(result.patch.roster_complete, True)
        self.assertEqual([r.age_years for r in result.patch.roster], [6.0, 0.5])
        self.assertEqual([r.id for r in result.patch.roster], [None, "item_1"])
        self.assertEqual(result.patch.roster[0].name, "Riya")

    def test_count_without_ages_is_an_incomplete_roster_not_a_complete_one(self):
        result = self.run_extract(self.shishu, {"roster_complete": False, "roster": []})
        self.assertTrue(result.ok)
        self.assertIs(result.patch.roster_complete, False)
        self.assertEqual(result.patch.roster, ())

    def test_unexpected_keys_are_rejected(self):
        result = self.run_extract(self.fever, {"patient_name": "x", "diagnosis": "thal"})
        self.assertEqual((result.ok, result.error_reason), (False, "invalid_schema"))

    def test_invalid_age_and_completeness_are_rejected(self):
        bad_age = self.run_extract(self.shishu, {"roster": [{"age_years": -1, "tag": "male"}]})
        self.assertEqual(bad_age.error_reason, "invalid_schema")
        bad_complete = self.run_extract(self.shishu, {"roster_complete": "yes"})
        self.assertEqual(bad_complete.error_reason, "invalid_schema")
        bad_id = self.run_extract(self.shishu, {"roster": [{"id": 7, "age_years": 2, "tag": "male"}]})
        self.assertEqual(bad_id.error_reason, "invalid_schema")

    def test_json_text_output_is_parsed_to_a_number(self):
        async def gen(**_kwargs):
            return type("Out", (), {"text": '{"patient_age": 4}'})()

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="chaar saal", generator=gen))
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.slots, {"patient_age": 4})
        self.assertNotIsInstance(result.patch.slots["patient_age"], str)

    def test_a_number_slot_builds_a_number_schema_and_rejects_a_word(self):
        # Bug: every profile slot was types.Type.STRING, so the model's 4 came back as "4" and nothing could
        # ever compare an age numerically — and "four" sailed through as a valid slot value.
        schema = extractor.build_response_schema(self.fever)
        self.assertIs(schema.properties["patient_age"].type, types.Type.NUMBER)
        self.assertIs(schema.properties["patient_name"].type, types.Type.STRING)
        self.assertEqual(extractor.validate_payload(self.fever, {"patient_age": 4}).slots, {"patient_age": 4})
        self.assertIsInstance(extractor.validate_payload(self.fever, {"patient_age": 4}).slots["patient_age"], int)
        self.assertEqual(extractor.validate_payload(self.fever, {"patient_age": 0.5}).slots, {"patient_age": 0.5})
        with self.assertRaises(ValueError):
            extractor.validate_payload(self.fever, {"patient_age": "four"})
        self.assertEqual(self.run_extract(self.fever, {"patient_age": "four"}).error_reason, "invalid_schema")

    def test_roster_schema_carries_ids_and_a_completeness_flag(self):
        schema = extractor.build_response_schema(self.shishu)
        self.assertIs(schema.properties["roster_complete"].type, types.Type.BOOLEAN)
        items = schema.properties["roster"].items
        self.assertEqual(set(items.properties), {"id", "age_years", "tag", "name"})
        self.assertEqual(tuple(items.required), ("age_years",))
        self.assertNotIn("roster_count", schema.properties)

    def test_children_already_on_record_reach_the_prompt_with_their_ids(self):
        # Bug: only current_profile was sent, so a mid-call "she has a twin too" could not name the existing
        # children and the extractor re-invented item_1..item_n from scratch.
        st = state_at(self.shishu, "intro_children")
        st.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male"),
                     RosterItem(id="item_2", age_years=4.0, age_label="4 साल", tag="female", name="रीना")]
        text = self.prompt_sent(st)
        self.assertIn("current_roster:", text)
        self.assertIn('{"id": "item_1", "age_years": 2.0, "tag": "male", "name": null}', text)
        self.assertIn('{"id": "item_2", "age_years": 4.0, "tag": "female", "name": "रीना"}', text)
        self.assertIn("roster_complete: true", text)

    def test_an_incomplete_roster_is_announced_as_incomplete(self):
        st = state_at(self.shishu, "intro_children")
        st.roster_incomplete = True
        text = self.prompt_sent(st)
        self.assertIn("current_roster: []", text)
        self.assertIn("roster_complete: false", text)
        self.assertEqual(self.prompt_sent(state_at(self.shishu, "birth_nicu")).count("roster_complete"), 0)

    def prompt_sent(self, state) -> str:
        seen: dict = {}

        async def gen(**kwargs):
            seen["contents"] = kwargs["contents"]
            return {"caller_name": None}

        asyncio.run(extractor.extract_profile_entities(
            self.shishu, state, utterance_text="उसकी उम्र चार साल है", generator=gen))
        return str(seen["contents"])

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

    def test_blocking_generator_runs_off_the_event_loop(self):
        def gen(**_kwargs):
            return {"patient_name": "Sync"}

        result = asyncio.run(extractor.extract_profile_entities(
            self.fever, utterance_text="x", generator=gen))
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.slots, {"patient_name": "Sync"})


if __name__ == "__main__":
    unittest.main()
