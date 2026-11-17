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

    def test_evaluate_turn_switches_to_fallback_request_after_primary_timeout(self):
        # Call #15: turn 29 on body_private_parts hit two 4.0s tail spikes against the full 64-head request and
        # the call hung up. Attempt 1 must carry the full primary request; the retry must carry only the slim
        # fallback (Group A + semantic flags) so the turn survives inside the same 8.0s budget.
        shishu = load_pack("shishu-hi")
        state = state_at(shishu, "body_private_parts")
        request = decisions.build_turn_request(shishu, state, utterance_text="पता नहीं।")
        fallback_request = decisions.build_turn_request(shishu, state, utterance_text="पता नहीं।", slim=True)
        seen_bodies: list[bytes] = []

        def send(url, payload, headers, timeout):
            seen_bodies.append(payload)
            if len(seen_bodies) == 1:
                raise TimeoutError("primary timed out")
            body = {"answers": {"active_outcome": {"probabilities": {"unknown": 0.95, "clear": 0.05}}},
                    "usage": {"total_input_tokens": 1180}}
            return 200, json.dumps(body).encode("utf-8")

        result = asyncio.run(decisions.evaluate_turn(
            request, api_key="k", transport=send, fallback_request=fallback_request))

        self.assertEqual(len(seen_bodies), 2)
        self.assertGreater(len(json.loads(seen_bodies[0])["questions"]), 40,
                           "attempt 1 is the full primary request")
        self.assertLessEqual(len(json.loads(seen_bodies[1])["questions"]), 10,
                             "attempt 2 is the slim fallback request")
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertIs(result.used_slim_fallback, True)
        self.assertEqual(result.question_count, len(fallback_request["questions"]))
        self.assertEqual(result.total_input_tokens, 1180)

        direct_bodies: list[bytes] = []

        def direct(url, payload, headers, timeout):
            direct_bodies.append(payload)
            return transport_returning(200, OK_BODY)(url, payload, headers, timeout)

        direct_result = asyncio.run(decisions.evaluate_turn(
            request, api_key="k", transport=direct, fallback_request=fallback_request))
        self.assertEqual(len(direct_bodies), 1)
        self.assertTrue(direct_result.ok)
        self.assertIs(direct_result.used_slim_fallback, False,
                      "a primary request that succeeds never reports the slim fallback")

    def hedged(self, coro) -> tuple[decisions.DecisionCallResult, list]:
        """Run one `evaluate_turn` and report every `asyncio.Task` still alive the moment it returned."""
        async def wrapper():
            result = await coro
            return result, [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return asyncio.run(wrapper())

    def test_hedged_retry_races_second_attempt_without_killing_in_flight_first_attempt(self):
        # Real `gemini-decision-1` 50-case bench: even 7-head `slim=True` calls sometimes sat 5.4s-6.5s in Google's
        # per-request queue while a concurrent identical call returned 200 in 0.7s-1.1s. Aborting Attempt 1 at
        # 3.5s/4.5s threw away healthy 5.5s `200 OK`s, so Attempt 2 must start *beside* Attempt 1 (hedge) and the
        # first clean read wins — whichever attempt it comes from.
        request = {"model": "gemini-decision-1", "input": "x", "questions": {"a": {}, "b": {}}}

        # Subcase 1: Attempt 2 rescues a stalled Attempt 1 long before Attempt 1's own timeout.
        seen: list[int] = []

        async def racing(url, payload, headers, timeout):
            seen.append(len(seen) + 1)
            await asyncio.sleep(0.30 if seen[-1] == 1 else 0.03)
            return 200, json.dumps(OK_BODY).encode()

        result, leaked = self.hedged(decisions.evaluate_turn(
            request, api_key="k", transport=racing,
            hedge_after_sec=0.05, attempt_timeout_sec=0.50, total_budget_sec=0.60))
        self.assertEqual(len(seen), 2, "the hedge launched Attempt 2 while Attempt 1 was still in flight")
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertLess(result.decision_ms, 200,
                        "Attempt 2 answered in 30ms: nothing may wait out Attempt 1's 500ms timeout")
        self.assertEqual(leaked, [], "every hedged task is cancelled and awaited before returning")

        # Subcase 2: the hedge must NOT kill Attempt 1 — it can still be the attempt that lands.
        order: list[int] = []

        async def slow_primary(url, payload, headers, timeout):
            order.append(len(order) + 1)
            if order[-1] == 1:
                await asyncio.sleep(0.12)
                return 200, json.dumps(OK_BODY).encode()
            return 503, b"{}"

        result2, leaked2 = self.hedged(decisions.evaluate_turn(
            request, api_key="k", transport=slow_primary, max_attempts=2,
            hedge_after_sec=0.04, attempt_timeout_sec=0.50, total_budget_sec=0.60))
        self.assertEqual(len(order), 2, "the 503 hedge fires 40ms in, while Attempt 1 is still pending")
        self.assertTrue(result2.ok, "Attempt 1 still finished cleanly at 120ms after the hedge fired")
        self.assertEqual(result2.attempts, 2)
        self.assertIs(result2.used_slim_fallback, False, "the full primary read, not the hedge, won")
        self.assertEqual(leaked2, [])


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

    def test_slim_turn_request_keeps_group_a_and_flags_while_dropping_stated_and_router(self):
        # Call #15: the 56 `stated_<qid>` heads plus the 53-option `next_qid` head are what pushed every spoken
        # turn to ~21.7k tokens. A slim request keeps only the heads a turn cannot be read without: the always-on
        # core, the active outcome, the active typed value, the semantic flags, and the one narrow gate
        # (`mentions_other_topic`) that decides whether Pass 2's routing heads are worth a second call at all.
        state = state_at(self.shishu, "body_private_parts")
        req = decisions.build_turn_request(self.shishu, state, utterance_text="पता नहीं।", slim=True)
        heads = req["questions"]
        always_on = (
            "active_outcome",
            "danger_now",
            "caller_stop",
            "mentions_profile",
            "redact_patient_note",
            "mentions_other_topic",
            "flag_confirmed_hemoglobinopathy",
        )
        for head in always_on:
            self.assertIn(head, heads)
        self.assertEqual([k for k in heads if k.startswith("stated_")], [])
        self.assertNotIn("jump_topic", heads)
        self.assertNotIn("next_qid", heads)
        self.assertLessEqual(len(heads), 10)

        full = decisions.build_turn_request(self.shishu, state, utterance_text="पता नहीं।")
        self.assertTrue([k for k in full["questions"] if k.startswith("stated_")],
                        "the default request still pays for the stated heads")
        self.assertIn("next_qid", full["questions"])

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

    def test_the_other_topic_gate_is_pinned_to_the_calibrated_yes_floor(self):
        # Real `gemini-decision-1` 50-case bench (DEV + HOLDOUT): `mentions_other_topic` scored 0.7467-0.9760 on
        # all 5 positive multi-topic cases (5/5 recall) and stayed under 0.50 on 34/36 calm single-topic cases
        # (0.06-0.44). The loose 0.35 gate bought nothing but fan-out on near-miss quiet turns, so the floor is
        # the empirical midpoint, in every pack, not just the one it was measured on.
        for pack in (self.shishu, self.fever):
            with self.subTest(pack=pack.id):
                self.assertEqual(pack.decisions["thresholds"]["mentions_other_topic_min_yes"], 0.5)
                text = (pack.root / "decisions.yaml").read_text(encoding="utf-8")
                self.assertIn("mentions_other_topic_min_yes: 0.5", text)


class GatedFanOutTests(unittest.TestCase):
    """Gate-first fan-out: Pass 1 is the narrow gate slice, Pass 2 runs only when another topic fires."""

    @classmethod
    def setUpClass(cls):
        cls.shishu = load_pack("shishu-hi")

    def transport(self, first: dict, second: dict | Exception | None, calls: list[dict]):
        """Two-pass fake: answers the gate slice first, then the routing slice (or an exception)."""

        def send(url, payload, headers, timeout):
            calls.append(json.loads(payload)["questions"])
            body = first if len(calls) == 1 else second
            if isinstance(body, Exception):
                raise body
            return 200, json.dumps({"answers": body, "usage": {"total_input_tokens": 1000}}).encode()

        return send

    def test_slim_gate_request_carries_mentions_other_topic_when_routing_is_eligible(self):
        # Pass 1 is the whole turn on an ordinary "पता नहीं।": the slim slice plus the one narrow semantic logic
        # gate that decides whether the 56-head routing fan-out is worth paying for. Before this, slim=True sent
        # no gate at all, so a multi-topic answer had nothing to trigger on.
        profile = decisions.build_turn_request(self.shishu, state_at(self.shishu, "intro_name"), slim=True)["questions"]
        self.assertNotIn("mentions_other_topic", profile, "a profile step resolves no routable answer")
        self.assertEqual([k for k in profile if k.startswith("stated_")], [])
        self.assertNotIn("next_qid", profile)

        gate = decisions.build_turn_request(self.shishu, state_at(self.shishu, "body_private_parts"),
                                            utterance_text="पता नहीं।", slim=True)["questions"]
        self.assertEqual(gate["mentions_other_topic"]["type"], "noul")
        for head in ("active_outcome", "danger_now", "caller_stop", "mentions_profile",
                     "redact_patient_note", "flag_confirmed_hemoglobinopathy"):
            self.assertIn(head, gate)
        self.assertEqual([k for k in gate if k.startswith("stated_")], [])
        self.assertNotIn("jump_topic", gate)
        self.assertNotIn("next_qid", gate)
        self.assertLessEqual(len(gate), 10)

    def test_build_routing_request_emits_only_stated_and_router_heads(self):
        # Pass 2 pays for Group B and Group D only. Re-sending Group A would ask the model to re-read the active
        # answer it already gave and let a second, thinner read overwrite the gate's verdict.
        state = state_at(self.shishu, "body_private_parts")
        req = decisions.build_routing_request(
            self.shishu, state, utterance_text="नहीं, पर चाची को भी कैंसर हुआ था")
        self.assertIn("stated_fam_cancer", req["questions"])
        self.assertIn("jump_topic", req["questions"])
        self.assertIn("next_qid", req["questions"])
        for head in ("active_outcome", "danger_now", "caller_stop", "mentions_profile",
                     "redact_patient_note", "mentions_other_topic"):
            self.assertNotIn(head, req["questions"])
        self.assertEqual(decisions.build_routing_request(self.shishu, state_at(self.shishu, "intro_name"))["questions"], {},
                         "nothing to route when no routable answer exists and the step is a profile slot")

    def test_evaluate_turn_gated_short_circuits_ordinary_turns_and_fans_out_only_when_hot(self):
        quiet = {"danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_other_topic": noul(0.02),
                 "active_outcome": choice(unknown=0.94, clear=0.04, other=0.02)}
        hot = {"danger_now": noul(0.02), "caller_stop": noul(0.01), "mentions_other_topic": noul(0.92),
               "active_outcome": choice(clear=0.93, other=0.07)}
        routing = {"stated_fam_cancer": choice(concern=0.95, unmentioned=0.05), "jump_topic": noul(0.85),
                   "next_qid": choice(fam_cancer=0.91, other=0.09)}
        utterance = "नहीं, पर चाची को भी कैंसर हुआ था"

        calls: list[dict] = []
        ordinary = asyncio.run(decisions.evaluate_turn_gated(
            self.shishu, state_at(self.shishu, "body_private_parts"), utterance_text="पता नहीं।",
            api_key="k", transport=self.transport(quiet, None, calls)))
        self.assertEqual(len(calls), 1, "an ordinary नहीं/पता नहीं turn stops after the gate")
        self.assertLessEqual(len(calls[0]), 10)
        self.assertIs(ordinary.ok, True)
        self.assertIs(ordinary.fanout_ran, False)
        self.assertIs(ordinary.used_slim_fallback, False)

        calls = []
        state = state_at(self.shishu, "body_private_parts")
        fanned = asyncio.run(decisions.evaluate_turn_gated(
            self.shishu, state, utterance_text=utterance, api_key="k",
            transport=self.transport(hot, routing, calls)))
        self.assertEqual(len(calls), 2)
        self.assertIn("active_outcome", calls[0])
        self.assertIn("mentions_other_topic", calls[0])
        self.assertIn("stated_fam_cancer", calls[1])
        self.assertIn("next_qid", calls[1])
        self.assertNotIn("active_outcome", calls[1], "Pass 2 never re-asks what Pass 1 already read")
        self.assertIs(fanned.ok, True)
        self.assertIs(fanned.fanout_ran, True)
        self.assertIs(fanned.used_slim_fallback, False)
        intent = decisions.interpret_turn_decision(self.shishu, state, fanned.raw_answers, utterance_text=utterance)
        self.assertIs(intent.command.outcome, TurnOutcome.CLEAR)
        self.assertEqual(intent.command.stated, {"fam_cancer": TurnOutcome.CONCERN})
        self.assertIs(intent.command.jump_topic, True)

        calls = []
        timed_out = asyncio.run(decisions.evaluate_turn_gated(
            self.shishu, state_at(self.shishu, "body_private_parts"), utterance_text=utterance, api_key="k",
            transport=self.transport(hot, TimeoutError("slow"), calls)))
        self.assertEqual(len(calls), 2)
        self.assertIs(timed_out.ok, True, "the gate's own read still stands when the fan-out dies")
        self.assertIs(timed_out.fanout_ran, True)
        self.assertIs(timed_out.used_slim_fallback, True)
        self.assertEqual(timed_out.raw_answers["active_outcome"], hot["active_outcome"])

        emergency = {"danger_now": noul(0.95), "caller_stop": noul(0.01), "mentions_other_topic": noul(0.95),
                     "active_outcome": choice(clear=0.5, other=0.4)}
        calls = []
        urgent = asyncio.run(decisions.evaluate_turn_gated(
            self.shishu, state_at(self.shishu, "body_private_parts"), utterance_text=utterance, api_key="k",
            transport=self.transport(emergency, None, calls)))
        self.assertEqual(len(calls), 1, "an active emergency never spends latency on 56 routing heads")
        self.assertIs(urgent.fanout_ran, False)


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

    def test_calibration_sweep_picks_widest_gap_cut_instead_of_grid_floor(self):
        from scripts import calibrate_decisions as cal

        pairs = [(0.08, False), (0.19, False), (0.43, False), (0.75, True), (0.97, True)]
        cut = cal._sweep(pairs, 200.0, 10.0)
        self.assertGreater(cut, 0.43, "a zero-loss valley between 0.43 and 0.75 must never collapse to 0.025")
        self.assertLess(cut, 0.75)


if __name__ == "__main__":
    unittest.main()
