"""Automated test harness for deterministic call state.

Includes:
1. Fast offline replay test (Call #13 scenario + multi-child & name disambiguation + PDF generation)
2. Live WebSocket text injection runner against a live SHISHU server (no microphone needed).
"""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

import call_state
import checklist
import db
import mother_report


def _up(field, value, state="answered", scope="none", child_ids=None, details=None):
    return {
        "field": field,
        "state": state,
        "value": value,
        "scope": scope,
        "child_ids": list(child_ids or []),
        "details": dict(details or {}),
    }


def _ask(state, topic_id, followup_id=None):
    return call_state.begin_question(state, {"topic_id": topic_id, "followup_id": followup_id})


class Call13DeterministicReplayTests(unittest.TestCase):
    def test_call_13_flow_with_disambiguation_and_pdf(self):
        # 1. Start call
        state = call_state.new_call_state()
        self.assertEqual(call_state.next_question(state)["topic_id"], "intro_name")

        # Turn 1: "मेरा नाम जिगर है।"
        # Model updates profile.caller_name -> automatically marks intro_name answered!
        state = call_state.apply_updates(
            state,
            [_up("profile.caller_name", "जिगर")],
            turn_id=1,
        )
        self.assertEqual(state["profile"]["caller_name"], "जिगर")
        self.assertEqual(state["topics"]["intro_name"]["state"], "answered")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_married")

        # Turn 3: "हो गई है।"
        state = _ask(state, nxt["topic_id"])
        state = call_state.apply_updates(
            state,
            [_up("profile.marital_status", "married")],
            turn_id=2,
        )
        self.assertEqual(state["profile"]["marital_status"], "married")
        self.assertEqual(state["topics"]["intro_married"]["state"], "answered")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_children")

        # Turn 5-11: "दो बच्चे हैं... एक 2 साल का है और एक 1 साल का है।"
        state = _ask(state, nxt["topic_id"])
        children = [
            {"age": {"value": 2, "unit": "years"}},
            {"age": {"value": 1, "unit": "years"}},
        ]
        state = call_state.apply_updates(
            state,
            [_up("profile.children", children)],
            turn_id=3,
        )
        self.assertEqual(len(state["profile"]["children"]), 2)
        self.assertEqual(state["topics"]["intro_children"]["state"], "answered")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_bridge")

        # Turn 13: "ठीक है।"
        state = _ask(state, nxt["topic_id"])
        state = call_state.apply_updates(
            state,
            [_up("topic.intro_bridge", True)],
            turn_id=4,
        )
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "birth_mode")

        # Turn 15: "एक नॉर्मल एक ऑपरेशन" -> positive birth_mode
        state = _ask(state, nxt["topic_id"])
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_mode", True, details={"delivery_type": "operation"})],
            turn_id=5,
        )
        # Follow-ups should include delivery_type/place, operation_reason, AND affected_child (since 2 kids exist)
        pending_fids = [f["id"] for f in call_state.TOPIC_RULES["birth_mode"]["followups"]]
        self.assertIn("affected_child", pending_fids)
        self.assertIn("operation_reason", pending_fids)

        # Resolve birth_mode followups
        state = call_state.apply_updates(
            state,
            [_up(
                "topic.birth_mode",
                True,
                scope="specific",
                child_ids=["child_1"],
                details={
                    "delivery_type": "operation",
                    "place": "hospital",
                    "operation_reason": "doctor ne bola",
                    "affected_child": "child_1",
                },
            )],
            turn_id=6,
        )
        # birth_mode is now resolved
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "birth_mode")

        # Skip forward to body_joint_swelling to test UNMAPPED NAME "Ram"
        state = _ask(state, "body_joint_swelling")
        state = call_state.apply_updates(
            state,
            [_up("topic.body_joint_swelling", True, scope="none", details={
                "joint": "ghutna", "trigger": "khelna", "duration": "kuch din",
            })],
            turn_id=7,
        )
        # Caller says "राम को" -> name not yet in profile.children!
        state = call_state.apply_updates(
            state,
            [_up("topic.body_joint_swelling", True, details={"affected_child": "Ram"})],
            turn_id=8,
        )
        # affected_child should still be UNRESOLVED because Ram is not mapped to an age!
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "body_joint_swelling")
        self.assertEqual(nxt["followup_id"], "affected_child")
        # Prompt note explicitly instructs Gemini to ask which age child Ram is
        self.assertIn("Ram", nxt["prompt_note"])
        self.assertIn("2 saal", nxt["prompt_note"])

        # Caller clarifies Ram is the 2-year-old child -> update profile.children name + bind child_ids
        updated_children = [
            {"id": "child_1", "name": "Ram", "age": {"value": 2, "unit": "years"}},
            {"id": "child_2", "name": None, "age": {"value": 1, "unit": "years"}},
        ]
        state = call_state.apply_updates(
            state,
            [
                _up("profile.children", updated_children),
                _up(
                    "topic.body_joint_swelling",
                    True,
                    scope="specific",
                    child_ids=["child_1"],
                    details={
                        "joint": "ghutna",
                        "duration": "kuch din",
                        "trigger": "khelne ke baad",
                        "affected_child": "Ram",
                    },
                ),
            ],
            turn_id=9,
        )
        # Now child_1 has name Ram, and body_joint_swelling is resolved
        self.assertEqual(state["profile"]["children"][0]["name"], "Ram")
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "body_joint_swelling")

        # Test Family Kinship rule on fam_bp_stroke
        state = _ask(state, "fam_bp_stroke")
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, scope="none", details={"condition": "BP", "onset_age": "38"})],
            turn_id=10,
        )
        # Caller says "Ramesh ko" (personal name only without kinship)
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, details={"who": "Ramesh"})],
            turn_id=11,
        )
        # who is UNRESOLVED because bare name is not kinship
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "fam_bp_stroke")
        self.assertEqual(nxt["followup_id"], "who")
        self.assertIn("kinship", nxt["prompt_note"])

        # Caller provides kinship "mama"
        state = call_state.apply_updates(
            state,
            [_up(
                "topic.fam_bp_stroke",
                True,
                details={
                    "who": "Ramesh",
                    "relationship": "mama (mother's brother)",
                    "condition": "BP",
                    "onset_age": "38",
                },
            )],
            turn_id=12,
        )
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "fam_bp_stroke")

        # Caller asks to stop -> request_close("caller_stop")
        state = call_state.request_close(state, "caller_stop")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "caller_stop")

        # Build Mother PDF from this state
        call_dict = {
            "id": 999,
            "started_at": "2026-11-08 02:36:00",
            "duration_sec": 300,
            "caller_name": "jigar",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "caller_stop"},
        }
        report = mother_report.build(call_dict)

        # CRITICAL ASSERTIONS:
        # 1. Mother name is NOT "माँ" -> it must be "जिगर"
        self.assertTrue(report["who"].startswith("जिगर"), f"Expected जिगर in who, got: {report['who']}")
        # 2. Status is incomplete
        self.assertEqual(report["status"], "अधूरी कॉल")
        # 3. "इस कॉल में जो हाँ था" contains positive findings with transliterated details
        self.assertEqual(len(report["yes"]), 3, f"Expected 3 yes items, got: {report['yes']}")
        yes_text = " ".join(report["yes"])
        self.assertIn("ऑपरेशन", yes_text)
        self.assertNotIn("ओपेरतिओन", yes_text)


class AdversarialFlowTests(unittest.TestCase):
    """5 adversarial, non-compliant, and edge-case flows to stress-test deterministic call state."""

    def test_flow_1_child_count_and_age_volatility_puberty_toggle(self):
        """Caller contradicts child count and ages; verifies puberty rule dynamically toggles."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.caller_name", "सुनीता")], 1)
        state = _ask(state, "intro_married")
        state = call_state.apply_updates(state, [_up("profile.marital_status", "married")], 2)

        # First claim: 1 child aged 12 -> puberty should be applicable
        state = _ask(state, "intro_children")
        state = call_state.apply_updates(
            state,
            [_up("profile.children", [{"age": {"value": 12, "unit": "years"}}])],
            3,
        )
        self.assertNotEqual(state["topics"]["body_puberty"]["state"], "not_applicable")

        # Caller corrects: "Wait, actually 2 children aged 3 and 8, the 12-year-old was a cousin"
        state = call_state.apply_updates(
            state,
            [_up("profile.children", [
                {"age": {"value": 3, "unit": "years"}},
                {"age": {"value": 8, "unit": "years"}},
            ])],
            4,
        )
        # Now every child is under 10 -> puberty MUST become not_applicable!
        self.assertEqual(state["topics"]["body_puberty"]["state"], "not_applicable")
        self.assertEqual(state["topics"]["body_puberty"]["reason"], "all_under_10")

        # Caller corrects age of child 2 to 10 years old -> puberty MUST become applicable again!
        state = call_state.apply_updates(
            state,
            [_up("profile.children", [
                {"age": {"value": 3, "unit": "years"}},
                {"age": {"value": 10, "unit": "years"}},
            ])],
            5,
        )
        self.assertNotEqual(state["topics"]["body_puberty"]["state"], "not_applicable")
        self.assertEqual(state["topics"]["body_puberty"]["state"], "not_asked")

    def test_flow_2_non_kinship_and_vague_family_rejection(self):
        """Caller names neighbors/friends or vague names without kinship; verifies server blocks."""
        state = call_state.new_call_state()
        state = _ask(state, "fam_bp_stroke")
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, scope="none", details={"condition": "stroke", "onset_age": "45"})],
            1,
        )

        # Adversarial answer: neighbor ("Sharma ji padosi ko")
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, details={"who": "Sharma ji", "relationship": "padosi"})],
            2,
        )
        # Server must NOT consider relationship resolved because neighbor is not family kinship!
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "fam_bp_stroke")
        self.assertEqual(nxt["followup_id"], "relationship")

        # Caller provides genuine kinship ("chacha")
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, details={"who": "chacha", "relationship": "chacha (father's brother)"})],
            3,
        )
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "fam_bp_stroke")

    def test_flow_3_compound_multi_condition_and_multi_child(self):
        """2 kids have different symptoms on the same question; verifies instances store both."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.caller_name", "पूजा")], 1)
        state = call_state.apply_updates(state, [_up("profile.marital_status", "married")], 2)
        state = call_state.apply_updates(
            state,
            [_up("profile.children", [
                {"id": "child_1", "name": "Aarav", "age": {"value": 4, "unit": "years"}},
                {"id": "child_2", "name": "Isha", "age": {"value": 2, "unit": "years"}},
            ])],
            3,
        )
        state = _ask(state, "body_fits_milestones")

        # Child 1 had fits
        state = call_state.apply_updates(
            state,
            [_up(
                "topic.body_fits_milestones",
                True,
                scope="specific",
                child_ids=["child_1"],
                details={"type": "fits", "onset": "2 saal pehle", "affected_child": "Aarav"},
            )],
            4,
        )
        # Child 2 had delayed walking
        state = call_state.apply_updates(
            state,
            [_up(
                "topic.body_fits_milestones",
                True,
                scope="specific",
                child_ids=["child_2"],
                details={"type": "late walking", "onset": "18 mahine", "affected_child": "Isha"},
            )],
            5,
        )

        topic = state["topics"]["body_fits_milestones"]
        self.assertEqual(len(topic["instances"]), 2)
        inst_ids = {inst["person_id"] for inst in topic["instances"]}
        self.assertEqual(inst_ids, {"child_1", "child_2"})

        # PDF should format both instances
        call_dict = {
            "id": 888,
            "started_at": "2026-11-08 03:00:00",
            "duration_sec": 200,
            "caller_name": "pooja",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "caller_stop"},
        }
        report = mother_report.build(call_dict)
        self.assertTrue(len(report["yes"]) >= 1)
        yes_line = report["yes"][0]
        self.assertTrue("Aarav" in yes_line or "आरव" in yes_line or "child_1" in yes_line)
        self.assertTrue("Isha" in yes_line or "इशा" in yes_line or "ईशा" in yes_line or "child_2" in yes_line)

    def test_flow_4_mid_call_acute_emergency_interrupt(self):
        """Caller interrupts mid-call with an acute emergency; verifies immediate shutdown and safety banner."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.caller_name", "सीमा")], 1)
        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_nicu", True, scope="all", details={"reason": "jaundice", "days": 3})],
            2,
        )

        # Emergency happens!
        state = call_state.request_close(state, "emergency")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "emergency")

        # Next question MUST be closing (not asking checklist topics)
        closing = call_state.next_question(state)
        self.assertEqual(closing["question_kind"], "closing")
        self.assertIn("emergency", closing["prompt_note"])

        # Mother PDF must prominently display emergency 108 banner
        call_dict = {
            "id": 777,
            "started_at": "2026-11-08 03:10:00",
            "duration_sec": 120,
            "caller_name": "seema",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "emergency", "emergency_detail": "bachche ko saans nahi aa rahi"},
        }
        report = mother_report.build(call_dict)
        self.assertTrue(report["emergency"])
        self.assertTrue(report["incomplete"])
        html_out = mother_report.render_html(report)
        self.assertIn("108", html_out)
        self.assertIn("अस्पताल ले जाइए", html_out)
        # Past answered topic (NICU) is still preserved
        self.assertEqual(len(report["yes"]), 1)

    def test_flow_5_unmarried_prior_pregnancy_miscarriage_and_thalassemia(self):
        """Caller is unmarried, has 0 children, but had a prior pregnancy miscarriage; verifies routing."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.caller_name", "रीना")], 1)
        state = _ask(state, "intro_married")
        state = call_state.apply_updates(state, [_up("profile.marital_status", "unmarried")], 2)

        # 0 living children
        state = _ask(state, "intro_children")
        state = call_state.apply_updates(state, [_up("profile.children", [])], 3)

        # All child groups (birth, body, ill) MUST be not_applicable
        for tid, top in state["topics"].items():
            if checklist._BY_ID[tid]["group"] in ("birth", "body", "illness"):
                self.assertEqual(top["state"], "not_applicable", tid)

        # Spouse question fam_related MUST be not_applicable
        self.assertEqual(state["topics"]["fam_related"]["state"], "not_applicable")

        # Complete intro_bridge
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_bridge")
        state = _ask(state, "intro_bridge")
        state = call_state.apply_updates(state, [_up("topic.intro_bridge", True)], 4)

        # All child groups (birth, body, ill) were skipped -> router goes straight to pregnancy!
        # Pregnancy history is unknown -> router asks profile.ever_pregnant first
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "profile.ever_pregnant")

        # Caller says YES she was pregnant before
        state = call_state.apply_updates(state, [_up("profile.ever_pregnant", True)], 5)

        # Miscarriage MUST remain applicable
        self.assertNotEqual(state["topics"]["fam_miscarriage"]["state"], "not_applicable")
        # Thalassemia MUST remain applicable
        self.assertNotEqual(state["topics"]["preg_thalassemia"]["state"], "not_applicable")

        # Record miscarriage at 3 months
        state = _ask(state, "fam_miscarriage")
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_miscarriage", True, details={"count": 1, "month": 3})],
            5,
        )

        # Record thalassemia test = NO
        state = _ask(state, "preg_thalassemia")
        state = call_state.apply_updates(
            state,
            [_up("topic.preg_thalassemia", False, details={"caller_test": False})],
            6,
        )

        # PDF must flag miscarriage in YES, and must recommend Thalassemia test before next pregnancy
        call_dict = {
            "id": 666,
            "started_at": "2026-11-08 03:20:00",
            "duration_sec": 240,
            "caller_name": "reena",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "complete"},
        }
        report = mother_report.build(call_dict)
        self.assertTrue(any("थैलेसीमिया" in q for q in report["questions"]))
        self.assertTrue(len(report["yes"]) >= 1)
        self.assertTrue(any("गर्भ" in y or "miscarriage" in y.lower() for y in report["yes"]))

    def test_flow_6_operation_reason_only_when_cesarean(self):
        """Normal delivery skips operation_reason; C-section requires operation_reason."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 2, "unit": "years"}}])], 1)
        state = _ask(state, "birth_mode")

        # Normal delivery
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_mode", True, details={"delivery_type": "normal", "place": "hospital"})],
            2,
        )
        # operation_reason should NOT be pending because delivery was normal
        pending_ids = [f["id"] for f in call_state._pending_followups(state, "birth_mode")]
        self.assertNotIn("operation_reason", pending_ids)

        # Correction to C-section
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_mode", True, details={"delivery_type": "operation", "place": "hospital"})],
            3,
        )
        pending_ids = [f["id"] for f in call_state._pending_followups(state, "birth_mode")]
        self.assertIn("operation_reason", pending_ids)

    def test_flow_7_hearing_vision_abnormal_triggers_type(self):
        """Normal newborn hearing/vision skips hearing_or_vision; abnormal requires it."""
        state = call_state.new_call_state()
        state = _ask(state, "birth_hearing_vision")

        # Test done, normal result
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_hearing_vision", True, details={"test_done": True, "result": "normal"})],
            1,
        )
        self.assertNotIn("hearing_or_vision", [f["id"] for f in call_state._pending_followups(state, "birth_hearing_vision")])

        # Abnormal result
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_hearing_vision", True, details={"test_done": True, "result": "abnormal"})],
            2,
        )
        self.assertIn("hearing_or_vision", [f["id"] for f in call_state._pending_followups(state, "birth_hearing_vision")])

    def test_flow_8_cleft_lip_palate_multi_child_attribution(self):
        """Cleft lip/palate requires child attribution when multiple children exist."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_up("profile.children", [
                {"age": {"value": 3, "unit": "years"}},
                {"age": {"value": 1, "unit": "years"}},
            ])],
            1,
        )
        state = _ask(state, "birth_cleft")
        state = call_state.apply_updates(state, [_up("topic.birth_cleft", True)], 2)
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "birth_cleft")
        self.assertEqual(nxt["followup_id"], "affected_child")

    def test_flow_9_cord_bleeding_and_delay_multi_child(self):
        """Cord delay > 3 weeks requires child attribution when multiple children exist."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_up("profile.children", [
                {"age": {"value": 4, "unit": "years"}},
                {"age": {"value": 2, "unit": "years"}},
            ])],
            1,
        )
        state = _ask(state, "birth_cord_3weeks")
        state = call_state.apply_updates(state, [_up("topic.birth_cord_3weeks", True)], 2)
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "birth_cord_3weeks")
        self.assertEqual(nxt["followup_id"], "affected_child")

    def test_flow_10_blood_count_abnormal_levels_and_timing(self):
        """Abnormal blood count follow-up requires low/high and timing."""
        state = call_state.new_call_state()
        state = _ask(state, "body_blood_count")
        state = call_state.apply_updates(state, [_up("topic.body_blood_count", True)], 1)
        pending = [f["id"] for f in call_state._pending_followups(state, "body_blood_count")]
        self.assertIn("low_or_high", pending)
        self.assertIn("timing", pending)

    def test_flow_11_blue_lips_or_fingers_location_and_timing(self):
        """Cyanosis episodes require location and timing."""
        state = call_state.new_call_state()
        state = _ask(state, "body_blue")
        state = call_state.apply_updates(state, [_up("topic.body_blue", True)], 1)
        pending = [f["id"] for f in call_state._pending_followups(state, "body_blue")]
        self.assertIn("location", pending)
        self.assertIn("timing", pending)

    def test_flow_12_limb_pain_hospital_visits_resolution(self):
        """Limb pain requires location, frequency, and hospital visits."""
        state = call_state.new_call_state()
        state = _ask(state, "body_limb_pain")
        state = call_state.apply_updates(state, [_up("topic.body_limb_pain", True)], 1)
        pending = [f["id"] for f in call_state._pending_followups(state, "body_limb_pain")]
        self.assertIn("location", pending)
        self.assertIn("frequency", pending)
        self.assertIn("hospital_visits", pending)

    def test_flow_13_metabolic_fasting_spell_multiple_triggers(self):
        """Metabolic fasting spells require symptom, trigger, and count."""
        state = call_state.new_call_state()
        state = _ask(state, "ill_fasting")
        state = call_state.apply_updates(state, [_up("topic.ill_fasting", True)], 1)
        pending = [f["id"] for f in call_state._pending_followups(state, "ill_fasting")]
        self.assertIn("symptom", pending)
        self.assertIn("trigger", pending)
        self.assertIn("count", pending)

    def test_flow_14_recurrent_infections_iv_antibiotics(self):
        """Hospital admissions and IV antibiotics require admission count and reasons."""
        state = call_state.new_call_state()
        state = _ask(state, "ill_admissions")
        state = call_state.apply_updates(state, [_up("topic.ill_admissions", True)], 1)
        pending = [f["id"] for f in call_state._pending_followups(state, "ill_admissions")]
        self.assertIn("count", pending)
        self.assertIn("reasons", pending)

    def test_flow_15_pregnancy_conditions_postpartum_persistence(self):
        """Pregnancy conditions require conditions (BP/sugar/thyroid) and postpartum persistence."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.ever_pregnant", True)], 1)
        state = _ask(state, "preg_conditions")
        state = call_state.apply_updates(state, [_up("topic.preg_conditions", True)], 2)
        pending = [f["id"] for f in call_state._pending_followups(state, "preg_conditions")]
        self.assertIn("conditions", pending)
        self.assertIn("continued_after_delivery", pending)

    def test_flow_16_currently_pregnant_first_time_delivery_not_applicable(self):
        """First pregnancy in progress (never delivered) skips preg_delivery but retains preg_scan."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [
                _up("profile.ever_pregnant", True),
                _up("profile.currently_pregnant", True),
                _up("profile.has_delivered", False),
            ],
            1,
        )
        self.assertEqual(state["topics"]["preg_delivery"]["state"], "not_applicable")
        self.assertEqual(state["topics"]["preg_delivery"]["reason"], "has_delivered_no")
        self.assertNotEqual(state["topics"]["preg_scan"]["state"], "not_applicable")
        self.assertNotEqual(state["topics"]["preg_conditions"]["state"], "not_applicable")

    def test_flow_17_partner_thalassemia_skipped_when_unmarried(self):
        """Unmarried caller skips partner thalassemia testing follow-ups."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.marital_status", "unmarried")], 1)
        state = _ask(state, "preg_thalassemia")
        state = call_state.apply_updates(state, [_up("topic.preg_thalassemia", True)], 2)
        pending = [f["id"] for f in call_state._pending_followups(state, "preg_thalassemia")]
        self.assertIn("caller_test", pending)
        self.assertIn("caller_result", pending)
        self.assertNotIn("partner_test", pending)
        self.assertNotIn("partner_result", pending)

    def test_flow_18_consanguinity_fam_related_skipped_for_unmarried(self):
        """Unmarried caller skips consanguinity (fam_related) completely."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.marital_status", "unmarried")], 1)
        self.assertEqual(state["topics"]["fam_related"]["state"], "not_applicable")
        self.assertEqual(state["topics"]["fam_related"]["reason"], "no_spouse")

    def test_flow_19_fam_same_problem_dynamic_activation(self):
        """fam_same_problem is not_applicable until a family condition is positive AND child exists."""
        state = call_state.new_call_state()
        self.assertEqual(state["topics"]["fam_same_problem"]["state"], "not_applicable")

        # Add child -> still not applicable because no family YES yet
        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 3, "unit": "years"}}])], 1)
        self.assertEqual(state["topics"]["fam_same_problem"]["state"], "not_applicable")

        # Family cancer = YES -> fam_same_problem MUST activate!
        state = _ask(state, "fam_cancer")
        state = call_state.apply_updates(state, [_up("topic.fam_cancer", True, details={"who": "dadi", "relationship": "dadi"})], 2)
        self.assertNotEqual(state["topics"]["fam_same_problem"]["state"], "not_applicable")
        self.assertEqual(state["topics"]["fam_same_problem"]["state"], "not_asked")

    def test_flow_20_fam_infant_death_age_and_cause(self):
        """Family infant death requires relationship, age at death, and cause."""
        state = call_state.new_call_state()
        state = _ask(state, "fam_infant_death")
        state = call_state.apply_updates(state, [_up("topic.fam_infant_death", True)], 1)
        pending = [f["id"] for f in call_state._pending_followups(state, "fam_infant_death")]
        self.assertIn("relationship", pending)
        self.assertIn("age_at_death", pending)
        self.assertIn("cause", pending)

    def test_flow_21_fam_cancer_kinship_validation(self):
        """Family cancer rejects non-family / friends and requires genuine kinship."""
        state = call_state.new_call_state()
        state = _ask(state, "fam_cancer")
        state = call_state.apply_updates(state, [_up("topic.fam_cancer", True, details={"who": "dost", "relationship": "friend"})], 1)
        # friend is not kinship -> relationship must not be resolved
        self.assertIn("relationship", [f["id"] for f in call_state._pending_followups(state, "fam_cancer")])

        # Update to genuine kinship
        state = call_state.apply_updates(state, [_up("topic.fam_cancer", True, details={"who": "nani", "relationship": "nani (mother's mother)"})], 2)
        self.assertNotIn("relationship", [f["id"] for f in call_state._pending_followups(state, "fam_cancer")])

    def test_flow_22_teeth_age_boundary_4_vs_5(self):
        """Child age 4 does not trigger tooth-fall bleeding; child age 5 triggers it."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 4, "unit": "years"}}])], 1)
        self.assertFalse(call_state._when_teeth(state, state["topics"]["body_bleeding_bruise"]))

        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 5, "unit": "years"}}])], 2)
        self.assertTrue(call_state._when_teeth(state, state["topics"]["body_bleeding_bruise"]))

    def test_flow_23_periods_daughter_age_boundary_10_vs_11(self):
        """Periods only activate for a female child aged >= 11; male aged 15 never triggers periods."""
        state = call_state.new_call_state()
        # 15 year old male
        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 15, "unit": "years"}, "sex": "male"}])], 1)
        self.assertFalse(call_state._when_periods(state, state["topics"]["body_bleeding_bruise"]))

        # 10 year old female
        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 10, "unit": "years"}, "sex": "female"}])], 2)
        self.assertFalse(call_state._when_periods(state, state["topics"]["body_bleeding_bruise"]))

        # 11 year old female
        state = call_state.apply_updates(state, [_up("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": "female"}])], 3)
        self.assertTrue(call_state._when_periods(state, state["topics"]["body_bleeding_bruise"]))

    def test_flow_24_silence_30s_marks_current_no_response(self):
        """30s silence marks active topic/follow-up as no_response and shifts to closing."""
        state = call_state.new_call_state()
        state = _ask(state, "birth_nicu", "reason")
        state = call_state.mark_current_no_response(state)
        # Reason detail should record no_response
        self.assertEqual(state["topics"]["birth_nicu"]["details"]["reason"], {"state": "no_response"})

        # Close for silence
        state = call_state.request_close(state, "silence")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "silence")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["question_kind"], "closing")

    def test_flow_25_complete_screening_pdf(self):
        """A complete call with all topics cleanly resolved renders 'पूर्ण कॉल' in Mother PDF."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.caller_name", "अंजली")], 1)
        state = call_state.apply_updates(state, [_up("profile.marital_status", "married")], 2)
        state = call_state.apply_updates(state, [_up("profile.children", [])], 3)
        state = call_state.apply_updates(state, [_up("topic.intro_bridge", True)], 4)
        state = call_state.apply_updates(state, [_up("profile.ever_pregnant", False)], 5)

        # Answer remaining applicable family topics with clean NO
        for tid in checklist.REQUIRED_IDS:
            if state["topics"][tid]["state"] in ("not_asked", "asking"):
                state = call_state.apply_updates(state, [_up(f"topic.{tid}", False)], 6)

        self.assertTrue(call_state.is_complete(state))
        state = call_state.request_close(state, "complete")

        call_dict = {
            "id": 555,
            "started_at": "2026-11-08 03:30:00",
            "duration_sec": 450,
            "caller_name": "anjali",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "complete"},
        }
        report = mother_report.build(call_dict)
        self.assertFalse(report["incomplete"])
        self.assertEqual(report["status"], "पूर्ण कॉल")
        self.assertTrue(report["who"].startswith("अंजली"))


_FULL_CALL_REPLIES = {
    ("intro_name", None): "मेरा नाम कविता है।",
    ("intro_married", None): "शादी हो गई है।",
    ("intro_children", None): "एक बच्चा है, तीन साल का।",
    ("intro_bridge", None): "ठीक है।",
    ("birth_mode", None): "अस्पताल में ऑपरेशन से हुआ था।",
    ("birth_mode", "delivery_type"): "ऑपरेशन से हुआ था।",
    ("birth_mode", "place"): "अस्पताल में।",
    ("birth_mode", "operation_reason"): "डॉक्टर ने पानी कम बताया था।",
    ("birth_nicu", None): "हाँ, पीलिया की वजह से 4 दिन काँच के बॉक्स में रखा था।",
    ("birth_nicu", "reason"): "पीलिया की वजह से।",
    ("birth_nicu", "days"): "चार दिन।",
    ("body_fits_milestones", None): "हाँ, एक साल पहले बुखार में झटके आए थे।",
    ("body_fits_milestones", "type"): "झटके आए थे।",
    ("body_fits_milestones", "onset"): "एक साल पहले।",
    ("fam_bp_stroke", None): "हाँ, मेरे सगे मामा को 35 साल की उम्र में बीपी हुआ था।",
    ("fam_bp_stroke", "condition"): "बीपी हुआ था।",
    ("fam_bp_stroke", "who"): "मेरे सगे मामा को।",
    ("fam_bp_stroke", "relationship"): "मेरे सगे मामा हैं।",
    ("fam_bp_stroke", "onset_age"): "35 साल की उम्र में।",
    ("profile.ever_pregnant", None): "हाँ, एक बार प्रेग्नेंट हुई थी।",
    ("profile.has_delivered", None): "हाँ, डिलीवरी हुई थी।",
    ("close_anything_else", None): "नहीं, बस इतना ही।",
}


def _reply_for_state(state: dict) -> str | None:
    if not state or state.get("phase") in ("closing", "ended"):
        return None
    cur = state.get("current_question") or {}
    tid = cur.get("topic_id")
    fid = cur.get("followup_id")
    if (tid, fid) in _FULL_CALL_REPLIES:
        return _FULL_CALL_REPLIES[(tid, fid)]
    if fid is not None:
        return "पता नहीं।"
    return "नहीं।"


async def run_live_text_call(base_url: str = "ws://127.0.0.1:8023/ws/gemini") -> dict:
    """Connect to a live SHISHU WebSocket and conduct a full state-driven screening to completion."""
    import websockets
    import app

    events = []
    current_step = 0
    call_id = None
    token = app._session_token("user", "jigar")
    from inspect import signature
    sig = signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        kwargs = {"additional_headers": [("Cookie", f"{app.COOKIE_NAME}={token}")]}
    else:
        kwargs = {"extra_headers": {"Cookie": f"{app.COOKIE_NAME}={token}"}}

    try:
        async with websockets.connect(base_url, **kwargs) as ws:
            print("[live_test] Connected to SHISHU WebSocket", flush=True)
            assistant_spoke_since_turn = False

            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=90.0)
                payload = json.loads(msg)
                mtype = payload.get("type")
                events.append(payload)

                if mtype == "session_started":
                    call_id = payload.get("call_id")
                    print(f"[live_test] Session started, call_id: {call_id}", flush=True)

                elif mtype == "audio":
                    assistant_spoke_since_turn = True

                elif mtype == "transcript" and payload.get("role") == "assistant":
                    bot_text = payload.get("text", "")
                    print(f"[live_test] Assistant: {bot_text}", flush=True)
                    assistant_spoke_since_turn = True

                elif mtype == "telemetry":
                    last_evt = (payload.get("last_event") or {}).get("type")
                    if last_evt == "turn_complete" and assistant_spoke_since_turn and call_id:
                        assistant_spoke_since_turn = False
                        saved = db.get_call(call_id) or {}
                        state = saved.get("state") or {}
                        reply = _reply_for_state(state)
                        if reply and current_step < 85:
                            current_step += 1
                            cur = state.get("current_question") or {}
                            print(
                                f"[live_test] Turn {current_step} ({cur.get('topic_id')}/{cur.get('followup_id')}): {reply}",
                                flush=True,
                            )
                            await ws.send(json.dumps({"type": "text", "text": reply}))

                elif mtype == "ending":
                    print("[live_test] Ending signal received", flush=True)

                elif mtype == "call_completed":
                    print(f"[live_test] Call completed! ID: {payload.get('call_id')}", flush=True)
                    return {
                        "call_id": payload.get("call_id"),
                        "events": events,
                        "success": True,
                    }
    except Exception as exc:
        print(f"[live_test] Connection ended: {exc}")
        return {"success": False, "call_id": call_id, "error": str(exc), "events": events}


if __name__ == "__main__":
    if "--live" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8023"
        url = f"ws://127.0.0.1:{port}/ws/gemini"
        print(f"Running live text-injection test against {url}...")
        result = asyncio.run(run_live_text_call(url))
        print("Live Result:", result.get("success"), "Call ID:", result.get("call_id"))
    else:
        unittest.main()
