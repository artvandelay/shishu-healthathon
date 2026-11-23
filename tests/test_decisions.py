"""Behavioral tests for `intake.decisions` and `intake.extractor`: zero network calls."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import unittest
from unittest import mock

import httpx
from google.genai import types

from intake import decisions, engine, extractor, policy
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CursorStep, PackConfig,
    RosterItem, SubStep, TurnOutcome,
)
from intake.pack import load_pack
from intake.turns import (
    DomainFit, EvidenceSource, SpeechAct, TopicRelation, TurnEvidence,
)

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
            # The slim fallback also asks for `danger_now`, so this reply must answer it: a 200 that omits a
            # requested safety head is now a retryable bad reply, never a confident "not an emergency".
            body = {"answers": {"danger_now": {"noul": 0.02},
                                "active_outcome": {"probabilities": {"unknown": 0.95, "clear": 0.05}}},
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

    def test_evaluate_turn_waits_patiently_for_slow_single_attempt_without_hedging(self):
        # Call #16 (`5fce012`): `hedge_after_sec=2.5` fired a second 92 KB WAV request beside a healthy first one,
        # and `min(6.5, total_budget_sec)` killed slow-but-alive turns at 6.5s. A slow primary call is not a dead
        # one: this turn must wait it out on one sequential call with the patient 12.0s / 18.0s defaults.
        params = inspect.signature(decisions.evaluate_turn).parameters
        self.assertNotIn("hedge_after_sec", params, "a hedged retry duplicates the WAV and doubles the bill")
        self.assertEqual(params["attempt_timeout_sec"].default, 12.0)
        self.assertEqual(params["total_budget_sec"].default, 18.0)

        request = {"model": "gemini-decision-1", "input": "x", "questions": {"a": {}, "b": {}}}
        in_flight, peak = 0, 0

        async def slow(url, payload, headers, timeout):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.08)
            in_flight -= 1
            return 200, json.dumps(OK_BODY).encode()

        result = asyncio.run(decisions.evaluate_turn(
            request, api_key="k", transport=slow, attempt_timeout_sec=0.50, total_budget_sec=0.60))
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 1, "an answered turn pays exactly one call")
        self.assertEqual(peak, 1, "never a second request racing the one already in flight")
        self.assertIs(result.used_slim_fallback, False)


def danger_request() -> dict:
    """A request that ASKS for `danger_now`, so the reply is required to answer it."""
    return {"model": "gemini-decision-1", "input": "x",
            "questions": {"danger_now": {"type": "noul", "instructions": "emergency?"},
                          "active_outcome": {"type": "choice"}}}


def plain_request() -> dict:
    """A request that never asks for `danger_now` (the shape a caller-less fixture uses)."""
    return {"model": "gemini-decision-1", "input": "x", "questions": {"a": {}, "b": {}}}


def reply(status: int, body: object) -> tuple[int, bytes]:
    return status, json.dumps(body).encode()


class DangerNowFailClosedTests(unittest.TestCase):
    """A 200 that does not answer the safety head is a BAD REPLY, never a confident "not an emergency".

    `_verdict` treated `answers` as a bag of optional heads, so a 200 whose `answers` omitted `danger_now`
    (or carried `null` / `NaN` / `Inf` / a string / a bool under `noul`) fell through `_yes` -> `_num` -> None
    and read as `emergency=False`. That is a fail-OPEN on the one head that closes a call for an emergency:
    the turn was reported `ok=True`, `interpret_turn_decision` saw no emergency, and the caller was re-asked
    the screening question while a life-threatening complaint sat unanswered in the reply.
    """

    def run_danger(self, transport, **kw) -> decisions.DecisionCallResult:
        return asyncio.run(decisions.evaluate_turn(danger_request(), api_key="k", transport=transport, **kw))

    def assert_fail_closed(self, result: decisions.DecisionCallResult, label: str) -> None:
        self.assertFalse(result.ok, f"{label}: a bad 200 must not be reported as a clean decision")
        self.assertIn(result.error_reason, {"missing_danger_now", "invalid_danger_now", "response_invalid"},
                      f"{label}: reason={result.error_reason!r} must name the unanswered safety head")
        self.assertEqual(result.raw_answers, {}, f"{label}: nothing from a rejected reply may be interpreted")
        self.assertEqual(result.attempts, 2, f"{label}: the slim fallback must still get its one attempt")

    def test_a_200_whose_answers_omit_danger_now_is_retried_and_then_fails_closed(self):
        # The head WAS requested; the model just did not return it. Treating that as `emergency=False` is the bug.
        bodies = [{"answers": {"active_outcome": choice(concern=0.97, clear=0.01, other=0.02)}},
                  {"answers": {"active_outcome": choice(clear=0.9, unknown=0.05, other=0.05)}}]
        seen: list = []

        def send(url, payload, headers, timeout):
            seen.append(payload)
            return reply(200, bodies[len(seen) - 1])

        result = self.run_danger(send)
        self.assertEqual(len(seen), 2, "a missing safety head is retryable, so Attempt 2 still runs")
        self.assert_fail_closed(result, "missing danger_now")
        self.assertEqual(result.error_reason, "missing_danger_now")

    def test_a_200_with_an_unreadable_danger_now_value_is_retried_and_then_fails_closed(self):
        cases = {
            "None": {"noul": None},
            "NaN": {"noul": float("nan")},
            "Inf": {"noul": math.inf},
            "-Inf": {"noul": -math.inf},
            "a string": {"noul": "bad"},
            "a bool": {"noul": True},
            "a dict without noul": {"probability": 0.9},
            "not a dict at all": 0.02,
        }
        for label, head in cases.items():
            with self.subTest(danger_now=label):
                result = self.run_danger(transport_returning(200, {"answers": {"danger_now": head}}))
                self.assert_fail_closed(result, label)
                self.assertEqual(result.error_reason, "invalid_danger_now")

    def test_a_readable_danger_now_still_passes_on_the_first_attempt(self):
        result = self.run_danger(transport_returning(200, OK_BODY))
        self.assertTrue(result.ok)
        self.assertEqual((result.error_reason, result.attempts), ("", 1))
        self.assertEqual(result.raw_answers, OK_BODY["answers"])

    def test_danger_now_is_only_required_when_the_request_actually_asked_for_it(self):
        # Every fixture that never sends the head must keep working: the rule is "answer what you were asked",
        # not "every reply must carry this head".
        body = {"answers": {"active_outcome": choice(clear=0.9, unknown=0.05, other=0.05)}}
        result = asyncio.run(decisions.evaluate_turn(
            plain_request(), api_key="k", transport=transport_returning(200, body)))
        self.assertTrue(result.ok)
        self.assertEqual((result.error_reason, result.attempts), ("", 1))
        self.assertEqual(result.raw_answers, body["answers"])

    def test_the_slim_fallback_is_judged_by_the_heads_IT_requested(self):
        # Attempt 2 carries the slim body, which also asks for `danger_now`: a slim reply that omits it is
        # equally invalid, so a primary timeout plus a headless slim reply must not pass.
        seen: list = []

        def send(url, payload, headers, timeout):
            seen.append(json.loads(payload))
            if len(seen) == 1:
                raise TimeoutError("primary timed out")
            return reply(200, {"answers": {"active_outcome": choice(clear=0.9, unknown=0.05, other=0.05)}})

        result = asyncio.run(decisions.evaluate_turn(
            danger_request(), api_key="k", transport=send, fallback_request=danger_request()))
        self.assertEqual(len(seen), 2)
        self.assert_fail_closed(result, "slim fallback missing danger_now")


class VerdictFailurePathTests(unittest.TestCase):
    """The transport/verdict branches the suite had no coverage for: non-JSON, empty answers, 5xx, 4xx, connect."""

    def run_plain(self, transport, **kw) -> decisions.DecisionCallResult:
        return asyncio.run(decisions.evaluate_turn(plain_request(), api_key="k", transport=transport, **kw))

    def test_a_200_that_is_not_json_is_a_terminal_bad_reply(self):
        calls: list = []

        def send(url, payload, headers, timeout):
            calls.append(payload)
            return 200, b"<html>not json</html>"

        result = self.run_plain(send)
        self.assertFalse(result.ok)
        self.assertEqual((result.error_reason, result.attempts, len(calls)), ("response_invalid", 1, 1))

    def test_a_200_with_empty_answers_is_terminal_and_never_retried(self):
        calls: list = []
        result = self.run_plain(transport_returning(200, {"answers": {}}, calls))
        self.assertFalse(result.ok)
        self.assertEqual((result.error_reason, result.attempts, len(calls)), ("empty_answers", 1, 1))
        self.assertEqual(result.raw_answers, {})

    def test_a_500_is_retried_and_a_following_200_recovers_the_turn(self):
        seen: list = []

        def send(url, payload, headers, timeout):
            seen.append(payload)
            return reply(500, {}) if len(seen) == 1 else reply(200, OK_BODY)

        result = self.run_plain(send)
        self.assertTrue(result.ok)
        self.assertEqual((result.error_reason, result.attempts, len(seen)), ("", 2, 2))
        self.assertEqual(result.total_input_tokens, 312)

    def test_a_400_is_terminal_and_never_retried(self):
        calls: list = []
        result = self.run_plain(transport_returning(400, {"error": {"code": "invalid_argument"}}, calls))
        self.assertFalse(result.ok)
        self.assertEqual((result.error_reason, result.attempts, len(calls)), ("http_400", 1, 1))

    def test_a_connection_error_is_retryable(self):
        seen: list = []

        def send(url, payload, headers, timeout):
            seen.append(payload)
            raise httpx.ConnectError("no route to host")

        result = self.run_plain(send)
        self.assertFalse(result.ok)
        self.assertEqual((result.error_reason, result.attempts, len(seen)), ("connection_error", 2, 2))

    def test_a_connection_error_then_a_200_recovers_the_turn(self):
        seen: list = []

        def send(url, payload, headers, timeout):
            seen.append(payload)
            if len(seen) == 1:
                raise httpx.ConnectError("no route to host")
            return reply(200, OK_BODY)

        result = self.run_plain(send)
        self.assertTrue(result.ok)
        self.assertEqual((result.error_reason, result.attempts), ("", 2))


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
        # core, the active outcome, the active typed value, and the semantic flags.
        state = state_at(self.shishu, "body_private_parts")
        req = decisions.build_turn_request(self.shishu, state, utterance_text="पता नहीं।", slim=True)
        heads = req["questions"]
        always_on = (
            "active_outcome",
            "danger_now",
            "caller_stop",
            "mentions_profile",
            "redact_patient_note",
            "active_topic",
            "residual_clinical",
            "asks_clarify",
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

    def test_shishu_item_ref_template_covers_volunteered_child_concerns_when_active_is_denied(self):
        # Call #16, Sunita step 4: "नहीं नीली लाइट में तो नहीं रखा था, पर मेरे बड़े बेटे अमन को तीन बार खून चढ़ाना
        # पड़ा है और उसका पेट भी धीरे-धीरे बड़ा हो रहा है।" denies active_question and then volunteers two concerns
        # for a named child in the same breath. The old template scoped item_ref to "the health symptom or event
        # asked about in active_question or parent_question", so the model answered `other` and body_transfusion
        # plus body_tummy landed unattributed (`item_ids=[]`) even though Aman was named out loud.
        text = " ".join(str(self.shishu.decisions["heads"]["item_ref_template"]["instructions"]).split())
        self.assertIn("volunteering another child symptom", text,
                      "a concern volunteered while denying active_question is still that child's concern")
        self.assertIn("whether answering active_question/parent_question", text)
        self.assertIn("choose other ONLY if no child concern is affirmed", text)
        self.assertNotIn("asked about in active_question or parent_question", text,
                         "the old active_question-only scope is what dropped both symptoms")

        # `mentions_other_topic` (`5fce012`) scored 0.29 and 0.068 on exactly this volunteered-NICU-and-नीली-लाइट
        # shape, so the gate went away entirely rather than being re-tuned: no head, no threshold, no request slot.
        for pack in (self.shishu, self.fever):
            with self.subTest(pack=pack.id):
                self.assertNotIn("mentions_other_topic", pack.decisions["heads"])
                self.assertNotIn("mentions_other_topic_min_yes", pack.decisions["thresholds"])
        state = state_at(self.shishu, "body_private_parts")
        for slim in (False, True):
            with self.subTest(slim=slim):
                self.assertNotIn(
                    "mentions_other_topic",
                    decisions.build_turn_request(self.shishu, state, utterance_text="पता नहीं।", slim=slim)["questions"])
        self.assertNotIn("mentions_other_topic", decisions._NOUL)
        self.assertNotIn("fanout_ran", decisions.DecisionCallResult._fields)
        self.assertFalse(hasattr(decisions, "build_routing_request"), "Pass 2's fan-out request is gone")
        self.assertFalse(hasattr(decisions, "evaluate_turn_gated"), "the gate-first fan-out is gone")
        # Sunita Step 7b: revising "नीली लाइट" (phototherapy) on fam_cancer flipped `stated_body_blue` to 0.94
        # when `body_blue.need` said only "lips or fingers turn blue". The pack's own `need` must disambiguate
        # cyanosis from phototherapy blue light.
        self.assertIn("नीली लाइट", self.shishu.by_id["body_blue"].need)

    def test_a_stated_need_widens_the_out_of_turn_head_without_widening_the_asked_question(self):
        # `birth_jaundice` on the `intro_bridge` turn: the caller said "हाँ शुरू करें, जन्म के बाद बच्चे को पीलिया
        # हुआ था।" and the `stated_birth_jaundice` head was in the request, yet the turn came back
        # `clinical_aside` — because the head told the model to choose `unmentioned` unless she named its `need`
        # ("jaundice treated under blue light") and she never said phototherapy. The stated head must read the
        # topic she can actually answer, while `active_question` keeps the narrower wording the pack asks aloud.
        st = state_at(self.shishu, "intro_bridge")
        heads = decisions.build_turn_request(self.shishu, st)["questions"]
        jaundice = heads["stated_birth_jaundice"]
        text = " ".join(str(jaundice["instructions"]).split())

        self.assertEqual(self.shishu.by_id["birth_jaundice"].need, "jaundice treated under blue light",
                         "the asked question keeps its narrow phototherapy wording")
        self.assertIn("jaundice (पीलिया) in the newborn after birth, with or without blue-light phototherapy", text,
                      "the stated head names the topic she can actually volunteer")
        self.assertNotIn('"jaundice treated under blue light"', text,
                         "the narrow need is exactly what made the model answer `unmentioned`")
        self.assertTrue(all("{" not in instruction for instruction in jaundice["criteria"].values()),
                        f"the .format() holes must be filled, got {jaundice['criteria']}")
        for criterion in jaundice["criteria"].values():
            self.assertIn("jaundice (पीलिया)", criterion, "every criterion reads the same widened topic")
        self.assertIn("क्या जन्म के बाद पीलिया के लिए बच्चे को नीली लाइट में रखना पड़ा था?", text,
                      "the head still quotes the question it revises")

    def test_a_question_without_a_stated_need_still_uses_its_need(self):
        # The override is opt-in per question: every gate that says nothing keeps its own `need` in the head.
        st = state_at(self.shishu, "intro_bridge")
        heads = decisions.build_turn_request(self.shishu, st)["questions"]
        needed = [q for q in self.shishu.questions
                  if q.is_routable and q.kind.value in ("gate", "routing_gate") and q.id != st.cursor.qid]
        self.assertTrue(needed)
        for q in needed:
            if q.stated_need:
                continue
            text = " ".join(str(heads[f"stated_{q.id}"]["instructions"]).split())
            self.assertIn(q.need, text, f"{q.id}: the head must still name the question's own need")


class InterpretTests(unittest.TestCase):
    """Threshold gating in both directions, per question kind."""

    @classmethod
    def setUpClass(cls):
        cls.fever = load_pack("fever-opd-hi")
        cls.shishu = load_pack("shishu-hi")

    def interpret(self, pack, state, answers, **kw) -> decisions.TurnIntent:
        return decisions.interpret_turn_decision(pack, state, answers, **kw)

    def test_a_quiet_answer_split_between_clear_and_unknown_is_read_as_the_pair(self):
        st = state_at(self.fever, "breathing_or_chest")
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
            "stated_body_blue": choice(concern=0.95, unmentioned=0.02, clear=0.02)})
        self.assertEqual(intent.command.stated, {"body_blue": TurnOutcome.CONCERN})


class Wave4StagingGapTests(unittest.TestCase):
    """The three live-voice gaps Calls #47 and #57 exposed on deployed staging, as invariants.

    1. a `stated_<qid>` head that reads `clarify` while the ACTIVE question is already being clarified is not a
       revision of anything: it is the same clarification spilling onto every sibling question, and acting on it
       moves the cursor off the question the caller actually asked about;
    2. an out-of-turn CONCERN on a question already answered CLEAR must be read on the pack's own stated gate, not
       on a second, undocumented 0.92 floor that even a 0.8559 / 0.7318-margin multimodal read cannot clear.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.shishu = load_pack("shishu-hi")

    def two_children(self) -> CallState:
        """Sunita's call: married, two children on record, so roster-scoped questions are all live."""
        st = engine.new_state(self.shishu)
        st.profile.update({"caller_name": "सुनीता", "marital_status": "married"})
        st.roster = [RosterItem(id="item_1", age_years=6.0, age_label="6 साल", tag="male", name="अमन"),
                     RosterItem(id="item_2", age_years=2.0, age_label="2 साल", tag="female", name="मीरा")]
        st.roster_stated = True
        for qid in ("intro_name", "intro_married", "intro_children", "intro_bridge"):
            st.answers[qid] = AnswerRecord(state=AnswerStatus.CLEAR)
        return st

    def at(self, qid: str) -> CallState:
        st = self.two_children()
        st.cursor = CursorStep(qid=qid)
        return st

    def test_active_clarify_suppresses_spurious_stated_clarify_and_multimodal_revision_honors_pack_stated_gate(self):
        # --- Subtest A: Call #57 `preg_thalassemia` replay ------------------------------------------------
        # "थैलेसीमिया का मतलब क्या है?" is a clarify of the ACTIVE question. The same audio also won `clarify` on
        # `stated_body_regression` (0.88) and `stated_fam_bleeding` (0.91) — questions she had already answered
        # CLEAR — and `_apply_stated` popped both of those answers and moved the cursor onto one of them.
        st = self.at("preg_thalassemia")
        st.answers["body_regression"] = AnswerRecord(state=AnswerStatus.CLEAR)
        st.answers["fam_bleeding"] = AnswerRecord(state=AnswerStatus.CLEAR)
        intent = decisions.interpret_turn_decision(self.shishu, st, {
            "active_outcome": choice(clarify=0.793, other=0.164, unknown=0.043),
            "asks_clarify": noul(0.85),
            "stated_body_regression": choice(concern=0.12, clear=0.11, unmentioned=0.77),
            "stated_fam_bleeding": choice(concern=0.10, clear=0.09, unmentioned=0.81),
            "stated_body_transfusion": choice(concern=0.93, unmentioned=0.04),
        }, utterance_text="मतलब क्या है थैलेसीमिया")
        observed_a = {
            "the active question is the one being clarified": intent.command.outcome is TurnOutcome.CLARIFY,
            "a stated clarify on an already-CLEAR question is spurious": "body_regression" not in intent.command.stated,
            "and so is the sibling family question": "fam_bleeding" not in intent.command.stated,
            "a real out-of-turn CONCERN still lands": intent.command.stated == {"body_transfusion": TurnOutcome.CONCERN},
        }
        for label, ok in observed_a.items():
            with self.subTest(subtest="A", invariant=label):
                self.assertTrue(ok, f"{label}: outcome={intent.command.outcome.value} "
                                    f"stated={ {k: v.value for k, v in intent.command.stated.items()} } "
                                    f"active_outcome={intent.diagnostics.get('top2')}")

        # Control: with NO active clarify the same head is a genuine clarify, and must keep working.
        control = self.at("body_transfusion")
        control.last_answered_qid = "preg_thalassemia"
        control.answers["preg_thalassemia"] = AnswerRecord(state=AnswerStatus.UNKNOWN)
        control.answers["body_regression"] = AnswerRecord(state=AnswerStatus.CLEAR)
        live_read = decisions.interpret_turn_decision(self.shishu, control, {
            "active_outcome": choice(clear=0.92, other=0.05),
            "asks_clarify": noul(0.91),
            "clarify_target": choice(preg_thalassemia=0.88, active_or_none=0.08),
        }, utterance_text="नहीं, खून नहीं चढ़ाया। थैलेसीमिया का मतलब क्या है?")
        with self.subTest(subtest="A control", invariant="a lone stated clarify still lands and still moves the call"):
            self.assertEqual(live_read.command.stated, {"preg_thalassemia": TurnOutcome.CLARIFY},
                             f"a clarify of another question is a real revision: "
                             f"outcome={live_read.command.outcome.value} "
                             f"stated={ {k: v.value for k, v in live_read.command.stated.items()} }")

        # --- Subtest B: Call #57 `fam_cancer` multimodal `birth_jaundice` revision ------------------------
        # The caller corrected herself out loud ("पहले नीली लाइट वाले सवाल में गलती हो गई थी"). The multimodal read
        # came back concern=0.8559 on `stated_birth_jaundice` over clear=0.1241: past the pack's 0.80 gate and
        # past its 0.35 margin by a mile, yet `_REVISED_CONCERN_MIN = 0.92` silently threw the correction away and
        # `birth_jaundice` stayed CLEAR for the rest of the call.
        correction = self.at("fam_cancer")
        correction.answers["birth_jaundice"] = AnswerRecord(state=AnswerStatus.CLEAR)
        heads = {"active_outcome": choice(concern=0.936, other=0.031), "detail_present": noul(0.55),
                 "jump_topic": noul(0.78),
                 "item_ref": choice(item_2=0.9975, other=0.0018, all=0.0004, item_1=0.0003)}
        intent_b = decisions.interpret_turn_decision(self.shishu, correction, heads | {
            "stated_birth_jaundice": choice(concern=0.8559, clear=0.1241, unmentioned=0.0199)})
        observed_b = {
            "0.8559 over the pack's 0.80 gate revises CLEAR -> CONCERN":
                intent_b.command.stated == {"birth_jaundice": TurnOutcome.CONCERN},
            "no second, higher concern floor survives beside the pack's own gate":
                not hasattr(decisions, "_REVISED_CONCERN_MIN"),
        }
        for label, ok in observed_b.items():
            with self.subTest(subtest="B", invariant=label):
                self.assertTrue(ok, f"{label}: stated={ {k: v.value for k, v in intent_b.command.stated.items()} } "
                                    f"floor={getattr(decisions, '_REVISED_CONCERN_MIN', None)!r} "
                                    f"stated_min_confidence={self.shishu.decisions['thresholds']['stated_min_confidence']}")

        # ...and a sub-threshold read is still refused: the pack's gate is the only gate, not no gate at all.
        weak = decisions.interpret_turn_decision(self.shishu, correction, heads | {
            "stated_birth_jaundice": choice(concern=0.78, clear=0.12, unmentioned=0.10)})
        with self.subTest(subtest="B control", invariant="a 0.78 concern is still under the pack's 0.80 stated gate"):
            self.assertEqual(weak.command.stated, {},
                             f"a sub-threshold revision is refused: "
                             f"stated={ {k: v.value for k, v in weak.command.stated.items()} }")

        # --- Subtest C: a clarify can only name the question she JUST LEFT ---------------------------------
        # Suppressing every `clarify` under an active clarify is too blunt: `route_clarify_preg_thalassemia` is the
        # legitimate shape — she said she did not know what thalassemia is (UNKNOWN), moved on to `fam_related`, and
        # then asked what the term means. That ONE head is the question she just left, named again. Everything else
        # is the ASR garble of Call #47/#57: an unasked future question (`prior is None`), a settled CLEAR/CONCERN
        # answer she already closed, or two-or-more heads scatter-firing `clarify` at once with nobody named.
        def clarify_read(**targets: float):
            """One active clarify on `fam_related`, with `clarify_target` softmax over named screening items."""
            base = self.at("fam_related")
            base.last_answered_qid = "preg_thalassemia"
            base.answers["preg_thalassemia"] = AnswerRecord(state=AnswerStatus.UNKNOWN)
            base.answers["body_regression"] = AnswerRecord(state=AnswerStatus.CLEAR)
            base.answers["fam_lumps"] = AnswerRecord(state=AnswerStatus.CONCERN)
            probs = dict(targets)
            probs["active_or_none"] = round(max(0.0, 1.0 - sum(targets.values())), 4)
            return decisions.interpret_turn_decision(
                self.shishu,
                base,
                {
                    "active_outcome": choice(clarify=0.929, other=0.048, unknown=0.013),
                    "asks_clarify": noul(0.93),
                    "clarify_target": choice(**probs),
                },
                utterance_text="थैलेसीमिया का क्या मतलब होता है?",
            )

        legit = clarify_read(preg_thalassemia=0.911)
        with self.subTest(subtest="C", invariant="the question she just left may still be named by one clarify head"):
            self.assertEqual(legit.command.stated, {"preg_thalassemia": TurnOutcome.CLARIFY},
                             f"a lone clarify on the UNKNOWN question she just left is a real request: "
                             f"stated={ {k: v.value for k, v in legit.command.stated.items()} } "
                             f"last_answered_qid=preq_thalassemia prior=unknown")

        cases = {
            "an unasked future question (prior is None) is never a clarify": {"fam_bleeding": 0.91},
            "a settled CLEAR answer is never re-clarified": {"body_regression": 0.93},
            "a settled CONCERN answer is never re-clarified": {"fam_lumps": 0.93},
            "two-or-more clarify heads scatter-firing name nobody": {"preg_thalassemia": 0.911, "fam_bleeding": 0.90,
                                                                    "fam_transfusion": 0.88},
        }
        for label, stated in cases.items():
            with self.subTest(subtest="C", invariant=label):
                intent_c = clarify_read(**stated)
                self.assertEqual(intent_c.command.stated, {},
                                 f"{label}: stated={ {k: v.value for k, v in intent_c.command.stated.items()} } "
                                 f"heads={stated}")


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


class Wave1DetailPresentAndExtractorTimeoutTests(unittest.TestCase):
    """Wave 1: the detail the caller already spoke must be readable on every step that carries a followup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.shishu = load_pack("shishu-hi")

    def test_detail_present_travels_and_interprets_on_which_item_when_followup_exists(self):
        # `detail_present` was offered only on "main"/"close", so at SubStep.WHICH_ITEM the model had no head to
        # say "she already answered the followup in the same breath" and Call #32 re-asked a settled question.
        steps = (
            ("gate_bones_fractures", SubStep.MAIN, 0, "main"),
            ("gate_bones_fractures", SubStep.WHICH_ITEM, 0, "which_item"),
            ("gate_bones_fractures", SubStep.FOLLOWUP, 0, "followup"),
            ("birth_nicu", SubStep.WHICH_ITEM, 0, "which_item"),
            ("birth_cord_3weeks", SubStep.MAIN, 0, "main"),
            ("birth_cord_3weeks", SubStep.WHICH_ITEM, 0, "which_item"),
            ("intro_children", SubStep.MAIN, 0, "profile"),
            ("intro_bridge", SubStep.MAIN, 0, "bridge"),
            ("close_anything_else", SubStep.MAIN, 0, "close"),
        )
        for qid, substep, fidx, kind in steps:
            with self.subTest(qid=qid, substep=substep.value, followup_idx=fidx):
                st = state_at(self.shishu, qid, substep=substep, followup_idx=fidx)
                self.assertEqual(engine.step_kind(self.shishu, st), kind)
                pending = bool(engine.followup_prompt(self.shishu, st))
                heads = decisions.build_turn_request(self.shishu, st)["questions"]
                self.assertEqual("detail_present" in heads, pending and kind in ("main", "which_item", "close"),
                                 f"detail_present follows the pending followup on a {kind} step (pending={pending})")

        # A which-item turn whose own question has no followup must not pay for a detail_present head.
        with self.subTest(qid="birth_cord_3weeks", case="followup-less which_item"):
            bare = state_at(self.shishu, "birth_cord_3weeks", substep=SubStep.WHICH_ITEM)
            self.assertEqual(engine.step_kind(self.shishu, bare), "which_item")
            self.assertNotIn("detail_present", decisions.build_turn_request(self.shishu, bare)["questions"])

        which_item = state_at(self.shishu, "gate_bones_fractures", substep=SubStep.WHICH_ITEM)
        self.assertEqual(engine.step_kind(self.shishu, which_item), "which_item")
        outcomes = {
            "detail + concern wins": (0.85, dict(concern=0.86, clear=0.06, other=0.05), True),
            "no detail + concern wins": (0.10, dict(concern=0.86, clear=0.06, other=0.05), False),
            "detail + clear wins": (0.85, dict(clear=0.86, concern=0.06, other=0.05), False),
        }
        for label, (detail, outcome, want) in outcomes.items():
            with self.subTest(case=label):
                intent = decisions.interpret_turn_decision(
                    self.shishu, which_item, {"detail_present": noul(detail), "active_outcome": choice(**outcome)})
                self.assertEqual(intent.diagnostics["step_kind"], "which_item")
                self.assertIs(intent.command.followup_answered, want,
                              f"followup_answered is detail_present AND a concern on the active question: {label} "
                              f"(detail_present={detail}, outcome={intent.command.outcome.value})")

    def test_extractor_default_timeout_is_patient_8s_for_spoken_rosters(self):
        # Call #25: the extractor cliffed at 3.001s on a spoken roster and reported "timeout", so "मेरे ट्विंस
        # बच्चे हैं" reached the reducer as nothing at all.
        seen: list[float] = []
        real_wait_for = asyncio.wait_for

        async def recording_wait_for(fut, timeout):  # noqa: ANN001 - mirrors asyncio.wait_for's signature
            seen.append(float(timeout))
            return await real_wait_for(fut, timeout)

        async def gen(**_kwargs):
            return {"roster_complete": True, "roster": [{"age_years": 5.0, "tag": "male", "name": "आरव"}]}

        async def run() -> extractor.ProfileExtractionResult:
            with mock.patch.object(extractor.asyncio, "wait_for", recording_wait_for):
                return await extractor.extract_profile_entities(
                    self.shishu, state_at(self.shishu, "intro_children"),
                    utterance_text="एक बेटा है 5 साल का आरव", generator=gen)

        result = asyncio.run(run())
        default = inspect.signature(extractor.extract_profile_entities).parameters["attempt_timeout_sec"].default
        self.assertTrue(result.ok)
        self.assertEqual([r.age_years for r in result.patch.roster], [5.0])
        for label, ok in {
            "the declared default is a patient 8.0s, not a 3.0s cliff": float(default) == 8.0,
            "the default budget is what asyncio.wait_for actually gets": seen == [8.0],
        }.items():
            with self.subTest(invariant=label):
                self.assertTrue(ok, f"{label}: declared default={default!r} recorded wait_for budgets={seen!r}")


class Call67DecisionTests(unittest.TestCase):
    """Call #67: a concern/clarify split must be heard as a question, and a note already stored must not be filed twice."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.shishu = load_pack("shishu-hi")
        cls.fever = load_pack("fever-opd-hi")

    def interpret(self, state, answers, **kw) -> decisions.TurnIntent:
        return decisions.interpret_turn_decision(self.shishu, state, answers, **kw)

    def test_concern_clarify_split_answers_the_meta_question_instead_of_burning_the_attempt(self):
        # Event 24, fam_fits_delay: "बूढ़े को है। आप बच्चे की पूछ रहे कि बूढ़े की पूछ रहे?"
        # active_outcome split concern 0.493 / clarify 0.329, asks_clarify 0.41 under the 0.55 gate.
        st = state_at(self.shishu, "fam_fits_delay")
        said = "बूढ़े को है। आप बच्चे की पूछ रहे कि बूढ़े की पूछ रहे?"
        cases = {
            "concern/clarify": {
                "active_outcome": choice(concern=0.493, clarify=0.329, clear=0.10, unknown=0.05, retract=0.028),
                "asks_clarify": noul(0.41),
                "active_topic": noul(0.48),
                "residual_clinical": noul(0.22),
            },
            "clear/clarify": {
                "active_outcome": choice(clear=0.48, clarify=0.36, concern=0.10, unknown=0.04, other=0.02),
                "asks_clarify": noul(0.40),
            },
        }
        for label, answers in cases.items():
            with self.subTest(split=label):
                intent = self.interpret(st, answers, utterance_text=said)
                self.assertIs(
                    intent.command.outcome, TurnOutcome.CLARIFY,
                    f"{label} split must answer her question, got {intent.command.outcome.value} "
                    f"top2={intent.diagnostics.get('top2')}")

    def test_asks_clarify_covers_meta_questions_about_the_call_and_the_notes(self):
        for pack in (self.shishu, self.fever):
            head = pack.decisions["heads"]["asks_clarify"]
            text = "\n".join((
                str(head.get("instructions") or ""),
                str((head.get("criteria") or {}).get("true_description") or ""),
            ))
            with self.subTest(pack=pack.id):
                self.assertIn("did you note that", text)
                self.assertIn("kids or adults", text)

    def test_followup_concern_is_not_also_filed_as_a_clinical_aside(self):
        # Event 30, fam_fits_delay:followup:0: the Alzheimer's note was accepted as the followup AND emitted CLINICAL_ASIDE.
        st = state_at(self.shishu, "fam_fits_delay", substep=SubStep.FOLLOWUP, followup_idx=0)
        st.answers["fam_fits_delay"] = AnswerRecord(state=AnswerStatus.CONCERN)
        said = "मेरे नाना-नानी को अल्जाइमर था"
        intent = self.interpret(st, {
            "active_topic": noul(0.92),
            "residual_clinical": noul(0.91),
            "active_outcome": choice(concern=0.93, clear=0.04, other=0.03),
            "asks_clarify": noul(0.05),
        }, utterance_text=said)
        rels = [c.relation for c in intent.interpretation.claims]
        self.assertNotIn(
            TopicRelation.CLINICAL_ASIDE, rels,
            f"followup already stores the note, got {rels}")
        self.assertIn(TopicRelation.ACTIVE, rels)
        self.assertIs(intent.command.outcome, TurnOutcome.CONCERN)

    def test_which_item_and_stored_concern_notes_are_not_duplicated_as_asides(self):
        cases = {
            "which_item": (
                state_at(self.shishu, "birth_jaundice", substep=SubStep.WHICH_ITEM),
                {"active_topic": noul(0.9), "residual_clinical": noul(0.9),
                 "active_outcome": choice(concern=0.91, clear=0.05, other=0.04)},
                "बड़े वाले को नीली लाइट में रखा था",
                False,
            ),
            "concern with no followup": (
                state_at(self.shishu, "birth_bcg"),
                {"active_topic": noul(0.93), "residual_clinical": noul(0.9),
                 "active_outcome": choice(concern=0.92, clear=0.04, other=0.04)},
                "हाँ, कंधे पर फोड़ा हुआ था",
                False,
            ),
            "concern with followup already answered": (
                state_at(self.shishu, "fam_fits_delay"),
                {"active_topic": noul(0.93), "detail_present": noul(0.9), "residual_clinical": noul(0.88),
                 "active_outcome": choice(concern=0.94, other=0.04, clear=0.02)},
                "नाना को अल्जाइमर था",
                True,
            ),
        }
        for label, (st, answers, said, want_followup) in cases.items():
            if st.cursor.substep is SubStep.WHICH_ITEM:
                st.answers[st.cursor.qid] = AnswerRecord(state=AnswerStatus.CONCERN)
            with self.subTest(step=label):
                intent = self.interpret(st, answers, utterance_text=said)
                rels = [c.relation for c in intent.interpretation.claims]
                if want_followup:
                    self.assertTrue(intent.command.followup_answered, label)
                self.assertNotIn(TopicRelation.CLINICAL_ASIDE, rels, f"{label} already stores the note, got {rels}")


class Call7FollowupAnswerTests(unittest.TestCase):
    """Call #7 `seq=77`: a free-text answer to an open **followup** must settle that followup.

    `interpret_turn_decision` folded `kind == "followup"` into the binary yes/no gate branch, whose `w` requires
    `active_outcome` to clear `outcome_min_confidence`. "मेरे नाना जी को बुढ़ापे में भूलने की अल्जाइमर बीमारी थी।"
    came back as a soft concern 0.499 (< 0.5196), so `w == ""`, no ACTIVE claim existed, `_utterance_already_stored`
    said False, `CLINICAL_ASIDE` took the turn, and `fam_who` closed UNANSWERED with an empty note.
    """

    FOLLOWUP_STEP = "fam_fits_delay:followup:0"
    SAID = "मेरे नाना जी को बुढ़ापे में भूलने की अल्जाइमर बीमारी थी।"

    @classmethod
    def setUpClass(cls) -> None:
        cls.shishu = load_pack("shishu-hi")

    def at_fam_who(self) -> CallState:
        st = state_at(self.shishu, "fam_fits_delay", substep=SubStep.FOLLOWUP, followup_idx=0)
        st.answers["fam_fits_delay"] = AnswerRecord(state=AnswerStatus.CONCERN)
        self.assertEqual(st.step_token, self.FOLLOWUP_STEP, "the replay must run on the live followup step")
        return st

    def turn(self, st: CallState, answers: dict, said: str | None = None, **kw):
        """One full turn: the interpreter, then the pure policy + reducer, exactly as `live.py` runs them."""
        text = self.SAID if said is None else said
        intent = decisions.interpret_turn_decision(self.shishu, st, answers, utterance_text=text, **kw)
        ev = TurnEvidence(source=EvidenceSource.VOICE, snapshot_step=st.step_token, transcript=text, received_at=1.0)
        nxt, out = policy.apply_policy(self.shishu, st, ev, intent.interpretation)
        return intent, nxt, out

    def test_a_free_text_followup_answer_settles_the_slot_and_suppresses_the_duplicate_aside(self):
        # seq=77: active_topic 0.92, residual_clinical 0.81, active_outcome concern 0.499 / clear 0.204 / other 0.15.
        st = self.at_fam_who()
        answers = {
            "active_topic": noul(0.92),
            "residual_clinical": noul(0.81),
            "active_outcome": choice(concern=0.499, clear=0.204, other=0.15, unknown=0.05, clarify=0.05, retract=0.047),
            "asks_clarify": noul(0.03),
        }
        intent, nxt, _out = self.turn(st, answers)
        claims = intent.interpretation.claims
        active = [c for c in claims if c.relation is TopicRelation.ACTIVE]
        self.assertEqual(intent.diagnostics["outcome"], "answered",
                         f"the outcome head never cleared its 0.5196 gate, so w is the empty read: "
                         f"top2={intent.diagnostics.get('top2')} outcome={intent.command.outcome.value}")
        self.assertTrue(active, f"a followup answer must carry an ACTIVE claim, got {[c.relation for c in claims]}")
        self.assertIs(active[0].status, TurnOutcome.ANSWERED, f"got status={active[0].status!r}")
        self.assertIs(active[0].domain_fit, DomainFit.IN_DOMAIN)
        self.assertNotIn(TopicRelation.CLINICAL_ASIDE, [c.relation for c in claims],
                         "the followup note is already stored, so a second aside would file it twice")

        rec = nxt.answers["fam_fits_delay"]
        self.assertIs(rec.state, AnswerStatus.CONCERN, "the parent gate answer is untouched")
        self.assertIs(rec.followups["fam_who"].state, AnswerStatus.ANSWERED,
                      f"fam_who must settle, got {rec.followups['fam_who'].state.value!r}")
        self.assertIn("अल्जाइमर", rec.verbatim_text, "her words are the note: they must survive verbatim")
        self.assertIn("अल्जाइमर", rec.patient_safe_text, "an ANSWERED slot is not skipped by skip_clear")
        self.assertNotEqual(nxt.cursor.qid, "fam_fits_delay", "the call advances past the answered family question")
        self.assertNotEqual(nxt.step_token, self.FOLLOWUP_STEP)

    def test_an_off_topic_aside_on_a_followup_still_retains_the_observation_and_burns_the_attempt(self):
        # Control: active_topic 0.10 — she is not talking about the question at all.
        st = self.at_fam_who()
        intent, nxt, out = self.turn(st, {
            "active_topic": noul(0.10),
            "residual_clinical": noul(0.85),
            "active_outcome": choice(other=0.44, concern=0.21, clear=0.18, unknown=0.10, clarify=0.05, retract=0.02),
            "asks_clarify": noul(0.02),
        }, said="गैस बंद कर दो, दूध उबल रहा है")
        rels = [c.relation for c in intent.interpretation.claims]
        self.assertNotIn(TopicRelation.ACTIVE, rels, "an off-topic remark answers nothing")
        self.assertIn(TopicRelation.CLINICAL_ASIDE, rels, "the residual clinical mention is kept as an observation")
        self.assertEqual([d.disposition.value for d in out.decisions], ["retain_observation"])
        self.assertEqual(nxt.cursor.qid, "fam_fits_delay")
        self.assertEqual(nxt.step_token, f"{self.FOLLOWUP_STEP}:try:1",
                         f"an unanswered followup burns one attempt, got {nxt.step_token}")
        self.assertNotIn("fam_who", nxt.answers["fam_fits_delay"].followups, "no slot is settled by an aside")

    def test_a_clarification_on_a_followup_holds_the_step_without_settling_or_burning_an_attempt(self):
        # Control: asks_clarify 0.80 — she is asking what the question means, not answering it.
        st = self.at_fam_who()
        intent, nxt, _out = self.turn(st, {
            "active_topic": noul(0.85),
            "asks_clarify": noul(0.80),
            "residual_clinical": noul(0.20),
            "active_outcome": choice(clarify=0.40, concern=0.24, other=0.20, clear=0.10, unknown=0.05, retract=0.01),
        }, said="मतलब क्या है? किसकी बात कर रहे हैं?")
        self.assertIs(intent.interpretation.act, SpeechAct.CLARIFY)
        self.assertNotIn(TopicRelation.ACTIVE, [c.relation for c in intent.interpretation.claims],
                         "a clarification is not an answer to the followup")
        self.assertEqual(nxt.cursor.qid, "fam_fits_delay")
        self.assertEqual(nxt.step_token, self.FOLLOWUP_STEP,
                         f"a clarify holds the same step and spends nothing, got {nxt.step_token}")
        self.assertNotIn("fam_who", nxt.answers["fam_fits_delay"].followups, "nothing is settled by a clarify")

    def test_explicit_unknown_and_clear_answers_keep_their_own_outcome(self):
        cases = {
            "unknown": (
                {"active_topic": noul(0.90), "active_outcome": choice(unknown=0.90, other=0.04, clear=0.03, concern=0.03),
                 "asks_clarify": noul(0.02)},
                TurnOutcome.UNKNOWN, AnswerStatus.UNKNOWN, "पता नहीं, मुझे याद नहीं",
            ),
            "clear": (
                {"active_topic": noul(0.90), "active_outcome": choice(clear=0.90, unknown=0.04, other=0.03, concern=0.03),
                 "asks_clarify": noul(0.02)},
                TurnOutcome.CLEAR, AnswerStatus.CLEAR, "नहीं, परिवार में किसी को नहीं है",
            ),
        }
        for label, (answers, want_outcome, want_status, said) in cases.items():
            with self.subTest(outcome=label):
                st = self.at_fam_who()
                intent, nxt, _out = self.turn(st, answers, said=said)
                self.assertIs(intent.command.outcome, want_outcome,
                              f"an explicit {label} keeps its outcome, got {intent.command.outcome.value}")
                self.assertIs(nxt.answers["fam_fits_delay"].followups["fam_who"].state, want_status)
                self.assertNotEqual(nxt.cursor.qid, "fam_fits_delay")

    def test_the_pack_boundary_sentences_for_pure_backtracking_load_cleanly(self):
        # Task 2: a pure correction preamble ("अरे रुकिए, मैंने पहले गलत बोल दिया — ...") that describes only the
        # other item is `other` on active_outcome and `false` on active_topic. The pack must still load, and the
        # two sentences must live in the YAML (no Python heuristic), without any threshold moving.
        pack = load_pack("shishu-hi")
        heads = pack.decisions["heads"]
        outcome_other = " ".join(str(heads["active_outcome"]["criteria"]["other"]).split())
        topic_false = " ".join(str(heads["active_topic"]["criteria"]["false_description"]).split())
        for label, text in (("active_outcome.criteria.other", outcome_other),
                            ("active_topic.criteria.false_description", topic_false)):
            with self.subTest(head=label):
                self.assertIn("correct", text.lower(),
                              f"{label} must name the interrupting correction preamble: {text!r}")
                self.assertIn("active_question", text)
        self.assertIn("wrong earlier", outcome_other.lower(),
                      "the worked example ('wait, I spoke wrongly earlier') is what the model needs to see")
        self.assertEqual(pack.decisions["thresholds"]["outcome_min_confidence"], 0.5196,
                         "thresholds are untouched by a prompt-boundary fix")
        self.assertEqual(pack.decisions["thresholds"]["active_topic_min_yes"], 0.55)


class Call8RetractSplitTests(unittest.TestCase):
    """Call #8 `birth_sudden:followup:0`: a flat denial was filed onto the red flag it retracts.

    "नहीं, ऐसा कुछ नहीं हुआ था।" came back `clear` 0.563 / `retract` 0.418 — one negation split across two labels,
    so neither cleared the 0.2297 margin, `w == ""`, and the `followup` branch read `active_topic` 0.82 as
    ANSWERED. The denial landed INSIDE `birth_sudden`'s CONCERN instead of clearing it.
    """

    CALL8_STEP = "birth_sudden:followup:0"
    SAID = "नहीं, ऐसा कुछ नहीं हुआ था।"
    CALL8_OUTCOME = dict(clear=0.563, retract=0.418, concern=0.009, other=0.005, unknown=0.003, clarify=0.002)

    @classmethod
    def setUpClass(cls) -> None:
        cls.shishu = load_pack("shishu-hi")

    def at_concern_followup(self) -> CallState:
        st = state_at(self.shishu, "birth_sudden", substep=SubStep.FOLLOWUP, followup_idx=0)
        st.answers["birth_sudden"] = AnswerRecord(state=AnswerStatus.CONCERN)
        self.assertEqual(st.step_token, self.CALL8_STEP, "the replay must run on the live followup step")
        return st

    def turn(self, st: CallState, answers: dict, said: str | None = None):
        """One full turn: the interpreter, then the pure policy + reducer, exactly as `live.py` runs them."""
        text = self.SAID if said is None else said
        intent = decisions.interpret_turn_decision(self.shishu, st, answers, utterance_text=text)
        ev = TurnEvidence(source=EvidenceSource.VOICE, snapshot_step=st.step_token, transcript=text, received_at=1.0)
        nxt, out = policy.apply_policy(self.shishu, st, ev, intent.interpretation)
        return intent, nxt, out

    def test_the_call8_denial_retracts_the_parent_concern_instead_of_answering_it(self):
        st = self.at_concern_followup()
        intent, nxt, _out = self.turn(st, {
            "active_topic": noul(0.82),
            "active_outcome": choice(**self.CALL8_OUTCOME),
        })
        self.assertIs(intent.interpretation.act, SpeechAct.RETRACT,
                      f"a denial of the parent gate is a retraction, got act={intent.interpretation.act} "
                      f"outcome={intent.command.outcome.value} top2={intent.diagnostics.get('top2')}")
        self.assertIs(intent.command.outcome, TurnOutcome.RETRACT,
                      f"clear 0.563 + retract 0.418 is one negation, not an answer, got "
                      f"{intent.command.outcome.value} top2={intent.diagnostics.get('top2')}")
        rec = nxt.answers["birth_sudden"]
        self.assertIs(rec.state, AnswerStatus.CLEAR,
                      f"taking the gate back clears it, got {rec.state.value!r}")
        self.assertEqual(rec.verbatim_text, "", "a denial is not a finding: it must never become the note")
        self.assertEqual(rec.patient_safe_text, "")
        self.assertNotEqual(nxt.cursor.qid, "birth_sudden", "the cleared question is done with")
        self.assertNotEqual(nxt.step_token, self.CALL8_STEP)

    def test_a_clear_retract_split_retracts_on_which_item_too(self):
        st = state_at(self.shishu, "birth_sudden", substep=SubStep.WHICH_ITEM)
        st.answers["birth_sudden"] = AnswerRecord(state=AnswerStatus.CONCERN)
        intent, nxt, _out = self.turn(st, {
            "active_outcome": choice(clear=0.55, retract=0.42, concern=0.02, other=0.01)})
        self.assertIs(intent.command.outcome, TurnOutcome.RETRACT,
                      f"a substep retract clears the parent, got {intent.command.outcome.value} "
                      f"top2={intent.diagnostics.get('top2')}")
        self.assertIs(nxt.answers["birth_sudden"].state, AnswerStatus.CLEAR,
                      f"the parent concern is cleared, got {nxt.answers['birth_sudden'].state.value!r}")

    def test_a_weak_split_with_a_concern_majority_never_falsely_retracts(self):
        st = state_at(self.shishu, "birth_sudden", substep=SubStep.WHICH_ITEM)
        st.answers["birth_sudden"] = AnswerRecord(state=AnswerStatus.CONCERN)
        intent, nxt, _out = self.turn(st, {
            "active_outcome": choice(clear=0.30, retract=0.20, concern=0.40, other=0.10)})
        self.assertIsNot(intent.command.outcome, TurnOutcome.RETRACT,
                         f"a 0.20 retract beside a 0.40 concern retracts nothing, got {intent.command.outcome.value}")
        self.assertIs(nxt.answers["birth_sudden"].state, AnswerStatus.CONCERN,
                      "the red flag survives a split too weak to read")

    def test_a_clear_retract_pair_under_the_outcome_gate_never_falsely_retracts(self):
        # The pair gate is the whole reason clear+retract can be summed: 0.28 + 0.23 is 0.51, under the pack's
        # 0.5196 `outcome_min_confidence`, so it is a coin toss, not a denial, and the red flag must survive it.
        st = state_at(self.shishu, "birth_sudden", substep=SubStep.WHICH_ITEM)
        st.answers["birth_sudden"] = AnswerRecord(state=AnswerStatus.CONCERN)
        intent, nxt, _out = self.turn(st, {
            "active_outcome": choice(clear=0.28, retract=0.23, concern=0.20, other=0.20, unknown=0.09)})
        self.assertIsNot(intent.command.outcome, TurnOutcome.RETRACT,
                         f"a 0.51 pair is under the 0.5196 gate, got {intent.command.outcome.value} "
                         f"top2={intent.diagnostics.get('top2')}")
        self.assertIs(nxt.answers["birth_sudden"].state, AnswerStatus.CONCERN,
                      f"the red flag survives an unreadable pair, got {nxt.answers['birth_sudden'].state.value!r}")

    def test_a_clear_retract_split_on_the_main_gate_resolves_cleanly(self):
        # On the main gate the two labels are the same "no": the pair's own winner is the read, never a burnt retry.
        st = state_at(self.shishu, "birth_sudden")
        intent, nxt, _out = self.turn(st, {"active_outcome": choice(clear=0.55, retract=0.40, concern=0.05)})
        self.assertIs(intent.command.outcome, TurnOutcome.CLEAR,
                      f"the quiet label of the pair wins on main, got {intent.command.outcome.value} "
                      f"top2={intent.diagnostics.get('top2')}")
        self.assertIs(nxt.answers["birth_sudden"].state, AnswerStatus.CLEAR)
        self.assertNotEqual(nxt.step_token, st.step_token, "a read answer never burns a retry")


class PooledClientLimitTests(unittest.TestCase):
    """The Decision pass runs once per caller turn, and every live call runs one: 100 sockets is not 50 calls' worth.

    httpx's default pool is 100 connections with 20 kept alive. Under the concurrency this box admits, the 21st
    caller's POST either opens a cold TLS connection per turn or waits on the pool's 3 s timeout — the decision
    latency cliff, bought for nothing. The pool is sized once, at construction, so this pins the constructor's limits.
    """

    def test_the_pooled_client_is_sized_for_the_concurrency_the_box_admits(self):
        built: dict[str, Any] = {}
        original = decisions.httpx.AsyncClient

        class RecordingClient(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                built.update(kwargs)
                super().__init__(*args, **kwargs)

            async def post(self, *args: Any, **kwargs: Any) -> Any:
                return httpx.Response(200, json=OK_BODY, request=httpx.Request("POST", decisions.DECISIONS_ENDPOINT))

        decisions.httpx.AsyncClient = RecordingClient  # type: ignore[assignment]
        try:
            status, _body = asyncio.run(decisions._post(decisions.DECISIONS_ENDPOINT, b"{}", {}, 1.0))
        finally:
            decisions.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(status, 200, "the fake reply must travel the pooled path")

        defaults = httpx.Limits()
        limits = built.get("limits")
        self.assertIsNotNone(limits, "the pooled client sets no explicit limits, so httpcore's default pool of "
                                     "100 connections / 20 keep-alive still applies")
        self.assertGreaterEqual(limits.max_connections, 256,
                                f"max_connections is {limits.max_connections}, under 256: the pool cannot carry the "
                                f"concurrent calls this box admits (httpcore's default is {defaults.max_connections or 100})")
        self.assertGreaterEqual(limits.max_keepalive_connections, 128,
                                f"max_keepalive_connections is {limits.max_keepalive_connections}, under 128: turns pay "
                                f"a cold TLS handshake (httpcore's default is {defaults.max_keepalive_connections or 20})")


if __name__ == "__main__":
    unittest.main()
