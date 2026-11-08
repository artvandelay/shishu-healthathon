"""Deterministic call-state routing. No network."""

import unittest

import checklist
import call_state


def _update(field, value, state="answered", scope="none", child_ids=None, details=None):
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


class CallStateTests(unittest.TestCase):
    def test_no_children_marks_birth_body_and_illness_not_applicable(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_update("profile.children", [])], 1)
        for topic in checklist.TOPICS:
            if topic["group"] in ("birth", "body", "illness"):
                self.assertEqual(state["topics"][topic["id"]]["state"], "not_applicable", topic["id"])
        self.assertNotEqual(state["topics"]["preg_thalassemia"]["state"], "not_applicable")
        self.assertNotEqual(state["topics"]["intro_children"]["state"], "not_applicable")

    def test_unmarried_does_not_remove_pregnancy_history(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_update("profile.marital_status", "unmarried")], 1)
        for topic_id in (
            "preg_conditions",
            "preg_scan",
            "preg_heart",
            "preg_transfusion",
            "preg_delivery",
            "fam_miscarriage",
            "preg_thalassemia",
        ):
            self.assertNotEqual(state["topics"][topic_id]["state"], "not_applicable", topic_id)

    def test_ever_pregnant_no_removes_the_exact_six_topics(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_update("profile.ever_pregnant", False)], 1)
        removed = (
            "preg_conditions",
            "preg_scan",
            "preg_heart",
            "preg_transfusion",
            "preg_delivery",
            "fam_miscarriage",
        )
        for topic_id in removed:
            self.assertEqual(state["topics"][topic_id]["state"], "not_applicable", topic_id)
        self.assertEqual(
            [topic_id for topic_id, topic in state["topics"].items() if topic["reason"] == "ever_pregnant_no"],
            list(removed),
        )

    def test_thalassemia_remains_applicable_when_ever_pregnant_is_no(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_update("profile.ever_pregnant", False)], 1)
        self.assertNotEqual(state["topics"]["preg_thalassemia"]["state"], "not_applicable")

    def test_under_10_children_remove_puberty(self):
        state = call_state.new_call_state()
        children = [
            {"name": "A", "age": {"value": 3, "unit": "years"}, "sex": "male"},
            {"name": "B", "age": {"value": 9, "unit": "years"}, "sex": "female"},
        ]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        self.assertEqual(state["topics"]["body_puberty"]["state"], "not_applicable")
        self.assertEqual(state["topics"]["body_puberty"]["reason"], "all_under_10")

    def test_girl_age_11_activates_periods(self):
        state = call_state.new_call_state()
        children = [{"name": "A", "age": {"value": 11, "unit": "years"}, "sex": "female"}]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        state = _ask(state, "body_bleeding_bruise")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", True, scope="all")],
            2,
        )
        seen = []
        for turn in range(6):
            nxt = call_state.next_question(state)
            if not nxt or nxt.get("topic_id") != "body_bleeding_bruise" or nxt.get("question_kind") != "followup":
                break
            seen.append(nxt["followup_id"])
            state = _ask(state, "body_bleeding_bruise", nxt["followup_id"])
            state = call_state.apply_updates(
                state,
                [_update(
                    "topic.body_bleeding_bruise",
                    True,
                    scope="all",
                    details={nxt["followup_id"]: "noted"},
                )],
                3 + turn,
            )
        self.assertIn("periods", seen)

    def test_no_cuts_or_bruises_still_asks_teeth_and_periods(self):
        state = call_state.new_call_state()
        children = [{"name": "A", "age": {"value": 11, "unit": "years"}, "sex": "female"}]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        state = _ask(state, "body_bleeding_bruise")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False, scope="all")],
            2,
        )

        self.assertEqual(call_state.next_question(state)["followup_id"], "teeth")
        state = _ask(state, "body_bleeding_bruise", "teeth")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False, scope="all", details={"teeth": False})],
            3,
        )
        self.assertEqual(call_state.next_question(state)["followup_id"], "periods")

        state = _ask(state, "body_bleeding_bruise", "periods")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", True, scope="all", details={"periods": True})],
            4,
        )
        topic = state["topics"]["body_bleeding_bruise"]
        self.assertIs(topic["value"], True)
        self.assertIs(topic["details"]["bleeding_or_bruises"], False)
        self.assertIs(topic["details"]["teeth"], False)
        self.assertIs(topic["details"]["periods"], True)
        self.assertNotEqual(
            (call_state.next_question(state) or {}).get("topic_id"),
            "body_bleeding_bruise",
        )

    def test_positive_nicu_blocks_until_reason_and_days_resolve(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="all")],
            1,
        )
        self.assertEqual(call_state.next_question(state)["followup_id"], "reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="all", details={"reason": "jaundice"})],
            2,
        )
        self.assertEqual(call_state.next_question(state)["followup_id"], "days")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="all", details={"days": 4})],
            3,
        )
        nxt = call_state.next_question(state)
        self.assertFalse(
            nxt and nxt.get("topic_id") == "birth_nicu" and nxt.get("question_kind") == "followup"
        )

    def test_family_bp_blocks_until_all_four_followups_resolve(self):
        state = _ask(call_state.new_call_state(), "fam_bp_stroke")
        state = call_state.apply_updates(
            state,
            [_update("topic.fam_bp_stroke", True, scope="none")],
            1,
        )
        seen = []
        for turn in range(6):
            nxt = call_state.next_question(state)
            if not nxt or nxt.get("topic_id") != "fam_bp_stroke" or nxt.get("question_kind") != "followup":
                break
            seen.append(nxt["followup_id"])
            state = _ask(state, "fam_bp_stroke", nxt["followup_id"])
            state = call_state.apply_updates(
                state,
                [_update("topic.fam_bp_stroke", True, scope="none", details={nxt["followup_id"]: "noted"})],
                2 + turn,
            )
        self.assertEqual(len(seen), 4)
        nxt = call_state.next_question(state)
        self.assertFalse(
            nxt and nxt.get("topic_id") == "fam_bp_stroke" and nxt.get("question_kind") == "followup"
        )

    def test_aggregate_no_stores_no_child_instances(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"name": "A", "age": {"value": 4, "unit": "years"}, "sex": "male"}])],
            1,
        )
        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", False, scope="all")],
            2,
        )
        topic = state["topics"]["birth_nicu"]
        self.assertEqual(topic["instances"], [])
        self.assertEqual(topic["value"], False)
        self.assertEqual(topic["state"], "answered")

    def test_specific_child_positive_stores_the_child_id(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"name": "A", "age": {"value": 4, "unit": "years"}, "sex": "female"}])],
            1,
        )
        child_id = state["profile"]["children"][0]["id"]
        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, scope="specific", child_ids=[child_id], details={"reason": "jaundice"})],
            2,
        )
        self.assertEqual(state["topics"]["birth_nicu"]["instances"][0]["person_id"], child_id)
        self.assertEqual(child_id, "child_1")

    def test_correction_replaces_the_value_and_adds_history(self):
        state = _ask(call_state.new_call_state(), "intro_name")
        state = call_state.apply_updates(state, [_update("topic.intro_name", "A")], 1)
        state = call_state.apply_updates(state, [_update("topic.intro_name", "B")], 2)
        self.assertEqual(state["topics"]["intro_name"]["value"], "B")
        self.assertTrue(any(event.get("previous_value") == "A" for event in state["history"]))
        self.assertIn(2, state["topics"]["intro_name"]["source_turn_ids"])

    def test_is_complete_is_false_for_unresolved_required_topics(self):
        self.assertFalse(call_state.is_complete(call_state.new_call_state()))

    def test_topic_update_before_children_are_known_is_not_stored(self):
        state = _ask(call_state.new_call_state(), "intro_name")
        with self.assertRaises(ValueError):
            call_state.apply_updates(state, [_update("topic.birth_nicu", True)], 1)
        self.assertEqual(state["topics"]["birth_nicu"]["state"], "not_asked")

    def test_child_count_without_ages_does_not_mean_no_children(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(state, [_update("profile.children", 2)], 1)
        self.assertEqual(len(state["profile"]["children"]), 2)
        self.assertTrue(all(child.get("age") is None for child in state["profile"]["children"]))
        self.assertNotEqual(state["topics"]["intro_children"]["state"], "answered")
        self.assertNotEqual(state["topics"]["birth_mode"]["state"], "not_applicable")
        state = call_state.apply_updates(state, [_update("topic.birth_mode", "normal")], 2)
        self.assertEqual(state["topics"]["birth_mode"]["value"], "normal")

    def test_late_count_keeps_ages_already_said(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", 1)],
            2,
            utterance="ठीक है।",
            prior_utterances=["एक बच्चा है, तीन साल का।"],
        )
        self.assertEqual(len(state["profile"]["children"]), 1)
        self.assertAlmostEqual(call_state._child_age_years(state["profile"]["children"][0]["age"]), 3)
        self.assertEqual(state["topics"]["intro_children"]["state"], "answered")

    def test_late_count_keeps_two_ages_and_ignores_a_nearer_different_count(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", 2)],
            2,
            utterance="ठीक है।",
            prior_utterances=["दो बच्चे हैं। एक चार साल का है और एक एक साल का है।"],
        )
        ages = [call_state._child_age_years(child["age"]) for child in state["profile"]["children"]]
        self.assertEqual(ages, [4, 1])

        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", 2)],
            3,
            utterance="ठीक है।",
            prior_utterances=[
                "दो बच्चे हैं। एक चार साल का है और एक एक साल का है।",
                "मेरे मामा की उम्र 35 साल है।",
            ],
        )
        self.assertTrue(all(child.get("age") is None for child in state["profile"]["children"]))
        self.assertNotEqual(state["topics"]["intro_children"]["state"], "answered")

    def test_empty_child_list_takes_ages_from_the_same_words(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"name": None}])],
            2,
            utterance="ठीक है।",
            prior_utterances=["एक बच्चा है, छह महीने का।"],
        )
        self.assertAlmostEqual(call_state._child_age_years(state["profile"]["children"][0]["age"]), 0.5)
        self.assertEqual(state["topics"]["intro_children"]["state"], "answered")

    def test_topic_logged_next_turn_keeps_the_child_already_named(self):
        state = _ask(call_state.new_call_state(), "intro_bridge")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_fits_milestones", True, scope="specific", child_ids=["child_1"])],
            2,
            utterance="ठीक है।",
            prior_utterances=["एक बच्चा है, दो साल का, और उसे झटके आते हैं।"],
        )
        self.assertEqual(len(state["profile"]["children"]), 1)
        self.assertAlmostEqual(call_state._child_age_years(state["profile"]["children"][0]["age"]), 2)
        self.assertIs(state["topics"]["body_fits_milestones"]["value"], True)
        self.assertEqual(state["topics"]["intro_children"]["state"], "answered")

    def test_a_relatives_age_does_not_open_a_child_slot(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        with self.assertRaises(ValueError):
            call_state.apply_updates(
                state,
                [_update("topic.body_fits_milestones", True, scope="specific", child_ids=["child_1"])],
                2,
                utterance="ठीक है।",
                prior_utterances=["मेरे मामा की उम्र 35 साल है।"],
            )
        self.assertEqual(state["profile"]["children"], [])

    def test_late_bare_number_keeps_the_month_unit_already_said(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 6}])],
            2,
            utterance="ठीक है।",
            prior_utterances=["एक बच्चा है, छह महीने का।"],
        )
        self.assertAlmostEqual(call_state._child_age_years(state["profile"]["children"][0]["age"]), 0.5)

    def test_dotted_topic_name_is_the_same_topic(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(state, [_update("profile.children", [{"age": 2}])], 1)
        state = call_state.apply_updates(
            state,
            [_update("body.fits_milestones", False, scope="specific", child_ids=["child_1"])],
            2,
        )
        self.assertIs(state["topics"]["body_fits_milestones"]["value"], False)
        self.assertEqual(state["topics"]["body_fits_milestones"]["state"], "answered")

    def test_a_duration_in_months_does_not_rewrite_a_year_age(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 2}])],
            2,
            utterance="झटके छह महीने से आ रहे हैं।",
            prior_utterances=["एक बच्चा है, दो साल का, और उसे झटके आते हैं।"],
        )
        age = state["profile"]["children"][0]["age"]
        self.assertEqual(age, 2)
        self.assertAlmostEqual(call_state._child_age_years(age), 2)
        self.assertEqual(call_state._spoken_ages("झटके छह महीने से आ रहे हैं।"), [])
        self.assertAlmostEqual(call_state._spoken_ages("एक बच्चा है, छह महीने का।")[0], 0.5)

    def test_spoken_months_fill_a_bare_count(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", 1)],
            1,
            utterance="एक बच्चा है, छह महीने का।",
        )
        self.assertEqual(len(state["profile"]["children"]), 1)
        age = state["profile"]["children"][0]["age"]
        self.assertEqual(age, {"value": 6, "unit": "months"})
        self.assertAlmostEqual(call_state._child_age_years(age), 0.5)
        self.assertEqual(state["topics"]["intro_children"]["state"], "answered")

    def test_month_only_list_is_not_stored_as_years(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 6}])],
            1,
            utterance="एक बच्चा है, छह महीने का।",
        )
        self.assertAlmostEqual(call_state._child_age_years(state["profile"]["children"][0]["age"]), 0.5)

    def test_hindi_and_not_married_mean_unmarried(self):
        for value in ("नहीं हुई", "not_married", "false"):
            state = call_state.apply_updates(
                call_state.new_call_state(),
                [_update("profile.marital_status", value)],
                1,
            )
            self.assertEqual(state["profile"]["marital_status"], "unmarried", value)
            self.assertEqual(state["topics"]["fam_related"]["state"], "not_applicable", value)

    def test_bare_topic_id_is_kept_beside_a_real_field(self):
        state = _ask(call_state.new_call_state(), "intro_children")
        state = call_state.apply_updates(
            state,
            [
                _update("profile.children", [{"age": 2}]),
                {"field": "fam_related", "state": "answered", "value": True, "scope": "none", "child_ids": [], "details": {}},
                {"field": "not_a_real_field", "state": "answered", "value": True, "scope": "none", "child_ids": [], "details": {}},
            ],
            1,
        )
        self.assertEqual(len(state["profile"]["children"]), 1)
        self.assertTrue(state["topics"]["fam_related"]["value"])

    def test_correction_is_kept_after_the_question_moves_on(self):
        state = _ask(call_state.new_call_state(), "birth_mode")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 3}]), _update("topic.birth_mode", "normal")],
            1,
        )
        state = _ask(state, "intro_bridge")
        state = call_state.apply_updates(state, [_update("topic.birth_mode", "operation")], 2)
        self.assertEqual(state["topics"]["birth_mode"]["value"], "operation")
        self.assertEqual(state["topics"]["birth_mode"]["details"].get("delivery_type"), "operation")

    def test_normal_hearing_is_not_a_positive_finding(self):
        state = _ask(call_state.new_call_state(), "birth_hearing_vision")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 3}]), _update("topic.birth_hearing_vision", True)],
            1,
            utterance="सुनने की जाँच हुई थी और नॉर्मल थी।",
        )
        self.assertEqual(state["topics"]["birth_hearing_vision"]["details"].get("result"), "normal")
        self.assertFalse(call_state._is_positive(state["topics"]["birth_hearing_vision"]))

    def test_fact_from_the_next_question_is_kept(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": {"value": 4, "unit": "years"}}, {"age": {"value": 1, "unit": "years"}}])],
            1,
        )
        state = _ask(state, "intro_bridge")
        state = call_state.apply_updates(
            state,
            [
                _update("topic.body_fits_milestones", True, scope="specific", child_ids=["child_1"]),
                _update("topic.birth_mode", "normal", scope="all"),
            ],
            2,
        )
        self.assertTrue(state["topics"]["body_fits_milestones"]["value"])
        self.assertEqual(state["topics"]["birth_mode"]["value"], "normal")

    def test_marital_false_means_unmarried(self):
        state = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.marital_status", "false")],
            1,
        )
        self.assertEqual(state["profile"]["marital_status"], "unmarried")
        self.assertEqual(state["topics"]["fam_related"]["state"], "not_applicable")

    def test_unknown_child_id_is_rejected(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        with self.assertRaises(ValueError):
            call_state.apply_updates(
                state,
                [_update("topic.birth_nicu", True, scope="specific", child_ids=["child_9"])],
                1,
            )

    def test_operation_reason_keeps_her_words_not_the_parent_token(self):
        state = _ask(call_state.new_call_state(), "birth_mode")
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 4}]), _update("topic.birth_mode", "operation")],
            1,
            utterance="अस्पताल में ऑपरेशन से हुआ था।",
        )
        self.assertEqual(state["topics"]["birth_mode"]["details"].get("place"), "hospital")
        state = _ask(state, "birth_mode", "operation_reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_mode", "operation")],
            2,
            utterance="डॉक्टर ने पानी कम बताया था।",
        )
        self.assertEqual(
            state["topics"]["birth_mode"]["details"].get("operation_reason"),
            "डॉक्टर ने पानी कम बताया था।",
        )

    def test_a_reason_and_a_stay_stated_with_the_answer_are_not_asked_again(self):
        state = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [{"age": {"value": 2, "unit": "years"}, "sex": "male"}])],
            1,
        )
        state = _ask(state, "birth_mode")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_mode", "जी हाँ")],
            2,
            utterance="जी हाँ, ऑपरेशन हुआ, पानी कम होने की वजह से, अस्पताल में।",
        )
        details = state["topics"]["birth_mode"]["details"]
        self.assertEqual(details.get("delivery_type"), "operation")
        self.assertEqual(details.get("place"), "hospital")
        self.assertIn("पानी कम", details.get("operation_reason") or "")
        pending = [item["id"] for item in call_state._pending_followups(state, "birth_mode")]
        self.assertNotIn("operation_reason", pending)
        self.assertNotIn("place", pending)

        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True)],
            3,
            utterance="पीलिया की वजह से चार दिन काँच के बक्से में रखा।",
        )
        nicu = state["topics"]["birth_nicu"]["details"]
        self.assertEqual(nicu.get("reason"), "jaundice")
        self.assertEqual(nicu.get("days"), 4)
        pending = [item["id"] for item in call_state._pending_followups(state, "birth_nicu")]
        self.assertNotIn("reason", pending)
        self.assertNotIn("days", pending)

        mixed = _ask(call_state.new_call_state(), "birth_nicu")
        mixed = call_state.apply_updates(
            mixed,
            [_update("topic.birth_nicu", True)],
            1,
            utterance="पीलिया और झटके दोनों की वजह से रखा।",
        )
        self.assertNotIn("reason", mixed["topics"]["birth_nicu"]["details"])

    def test_clarification_does_not_close_the_followup(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        state = call_state.apply_updates(state, [_update("topic.birth_nicu", True)], 1)
        state = _ask(state, "birth_nicu", "reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True)],
            2,
            utterance="क्या मतलब?",
        )
        self.assertNotIn("reason", state["topics"]["birth_nicu"]["details"])
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "birth_nicu")
        self.assertEqual(nxt["followup_id"], "reason")

        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, details={"reason": "clarified"})],
            3,
            utterance="क्या मतलब?",
        )
        self.assertNotIn("reason", state["topics"]["birth_nicu"]["details"])
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "birth_nicu")
        self.assertEqual(nxt["followup_id"], "reason")

    def test_pata_nahi_resolves_the_followup_as_unknown(self):
        state = _ask(call_state.new_call_state(), "birth_nicu")
        state = call_state.apply_updates(state, [_update("topic.birth_nicu", True)], 1)
        state = _ask(state, "birth_nicu", "reason")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True)],
            2,
            utterance="पता नहीं।",
        )
        self.assertEqual(state["topics"]["birth_nicu"]["details"].get("reason"), {"state": "unknown"})
        nxt = call_state.next_question(state)
        self.assertNotEqual((nxt or {}).get("followup_id"), "reason")

        for utterance in ("वजह पता नहीं।", "मुझे पता नहीं है।", "याद नहीं है।"):
            wrapped = _ask(call_state.new_call_state(), "birth_nicu")
            wrapped = call_state.apply_updates(wrapped, [_update("topic.birth_nicu", True)], 1)
            wrapped = _ask(wrapped, "birth_nicu", "reason")
            wrapped = call_state.apply_updates(
                wrapped,
                [_update("topic.birth_nicu", True)],
                2,
                utterance=utterance,
            )
            self.assertEqual(
                wrapped["topics"]["birth_nicu"]["details"].get("reason"),
                {"state": "unknown"},
                utterance,
            )

        numbered = _ask(call_state.new_call_state(), "birth_nicu")
        numbered = call_state.apply_updates(numbered, [_update("topic.birth_nicu", True)], 1)
        numbered = _ask(numbered, "birth_nicu", "reason")
        numbered = call_state.apply_updates(
            numbered,
            [_update("topic.birth_nicu", True)],
            2,
            utterance="दो दिन पता नहीं।",
        )
        self.assertNotEqual(
            numbered["topics"]["birth_nicu"]["details"].get("reason"),
            {"state": "unknown"},
        )

    def test_conditions_and_kinship_stated_with_the_answer_are_not_asked_again(self):
        state = _ask(call_state.new_call_state(), "preg_conditions")
        state = call_state.apply_updates(
            state,
            [_update("topic.preg_conditions", True)],
            1,
            utterance="प्रेग्नेंसी में बीपी और शुगर था, डिलीवरी के बाद भी है।",
        )
        details = state["topics"]["preg_conditions"]["details"]
        self.assertEqual(details.get("conditions"), ["bp", "sugar"])
        self.assertIs(details.get("continued_after_delivery"), True)
        pending = [item["id"] for item in call_state._pending_followups(state, "preg_conditions")]
        self.assertEqual(pending, [])

        state = _ask(call_state.new_call_state(), "fam_infant_death")
        state = call_state.apply_updates(
            state,
            [_update("topic.fam_infant_death", True)],
            1,
            utterance="मामा का बच्चा चार महीने में गुजर गया, झटके की वजह से।",
        )
        details = state["topics"]["fam_infant_death"]["details"]
        self.assertIn("मामा", details.get("relationship") or "")
        pending = [item["id"] for item in call_state._pending_followups(state, "fam_infant_death")]
        self.assertNotIn("relationship", pending)

    def test_devanagari_kinship_resolves_the_family_followup(self):
        state = _ask(call_state.new_call_state(), "fam_bp_stroke")
        state = call_state.apply_updates(
            state,
            [_update("topic.fam_bp_stroke", True, details={"relationship": "मामा"})],
            1,
        )
        pending = [item["id"] for item in call_state._pending_followups(state, "fam_bp_stroke")]
        self.assertNotIn("relationship", pending)

    def test_bade_wale_binds_the_older_child(self):
        state = call_state.new_call_state()
        state = call_state.apply_updates(
            state,
            [_update("profile.children", [{"age": 2}, {"age": 6}])],
            1,
        )
        state = _ask(state, "body_fits_milestones")
        state = call_state.apply_updates(
            state,
            [_update(
                "topic.body_fits_milestones",
                True,
                details={"type": "fits", "onset": "एक साल", "affected_child": "बड़े वाले"},
            )],
            2,
        )
        call_state.next_question(state)
        ids = [item.get("person_id") for item in state["topics"]["body_fits_milestones"]["instances"]]
        self.assertIn("child_2", ids)

    def test_child_disambiguation_by_months_and_hindi_words(self):
        state = call_state.new_call_state()
        children = [
            {"id": "child_1", "name": None, "age": {"value": 3, "unit": "years"}},
            {"id": "child_2", "name": None, "age": {"value": 6, "unit": "months"}},
        ]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        self.assertEqual(call_state._find_child_id(state, "6 mahine"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "6 महीने"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "छह महीने वाला"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "3 saal"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "3 साल"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "तीन साल वाला"), "child_1")

    def test_child_disambiguation_ordinal_and_sex(self):
        state = call_state.new_call_state()
        children = [
            {"id": "child_1", "name": None, "age": {"value": 4, "unit": "years"}, "sex": "female"},
            {"id": "child_2", "name": None, "age": {"value": 2, "unit": "years"}, "sex": "male"},
        ]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        self.assertEqual(call_state._find_child_id(state, "pehle wala"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "पहले वाला"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "dusre wala"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "दूसरा बच्चा"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "beti ko"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "बेटी को"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "beta"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "बेटे को"), "child_2")
        self.assertIsNone(call_state._find_child_id(state, "बच्चा"))
        self.assertIsNone(call_state._find_child_id(state, "बच्चा को"))

        three = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [
                {"age": {"value": 6, "unit": "years"}, "sex": "male"},
                {"age": {"value": 4, "unit": "years"}, "sex": "female"},
                {"age": {"value": 1, "unit": "years"}, "sex": "male"},
            ])],
            1,
        )
        self.assertEqual(call_state._find_child_id(three, "दूसरे वाले"), "child_2")
        self.assertEqual(call_state._find_child_id(three, "बीच वाला"), "child_2")
        self.assertEqual(call_state._find_child_id(three, "तीसरे वाले"), "child_3")
        self.assertEqual(call_state._find_child_id(three, "बड़े वाले"), "child_1")
        self.assertEqual(call_state._find_child_id(three, "छोटे वाले"), "child_3")

    def test_child_name_with_case_markers(self):
        state = call_state.new_call_state()
        children = [
            {"id": "child_1", "name": "Ram", "age": {"value": 3, "unit": "years"}},
            {"id": "child_2", "name": "Aarav", "age": {"value": 1, "unit": "years"}},
        ]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        self.assertEqual(call_state._find_child_id(state, "Ram ko"), "child_1")
        self.assertEqual(call_state._find_child_id(state, "Aarav ko hua tha"), "child_2")

    def test_affected_child_all_resolves_and_populates_instances(self):
        state = call_state.new_call_state()
        children = [
            {"id": "child_1", "name": "A", "age": {"value": 3, "unit": "years"}},
            {"id": "child_2", "name": "B", "age": {"value": 1, "unit": "years"}},
        ]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_nicu", True, details={"reason": "jaundice", "days": 3, "affected_child": "दोनों बच्चों को"})],
            2,
        )
        nxt = call_state.next_question(state)
        self.assertFalse(nxt and nxt.get("topic_id") == "birth_nicu" and nxt.get("followup_id") == "affected_child")
        instances = state["topics"]["birth_nicu"]["instances"]
        self.assertEqual({inst["person_id"] for inst in instances}, {"child_1", "child_2"})

    def test_children_payload_string_counts_and_none(self):
        self.assertEqual(call_state._children_payload("दो बच्चे"), ("count", 2))
        self.assertEqual(call_state._children_payload("2 kids"), ("count", 2))
        self.assertEqual(call_state._children_payload("कोई नहीं"), ("none", []))
        self.assertEqual(call_state._children_payload("none"), ("none", []))
        self.assertEqual(call_state._children_payload("कोई बच्चा नहीं है"), ("none", []))
        self.assertEqual(call_state._children_payload("बच्चे नहीं हैं"), ("none", []))
        self.assertEqual(call_state._children_payload("no children"), ("none", []))
        self.assertEqual(call_state._children_payload("दो बच्चे"), ("count", 2))
        state = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", "कोई बच्चा नहीं है")],
            1,
            utterance="कोई बच्चा नहीं है।",
        )
        self.assertEqual(state["profile"]["children"], [])
        self.assertEqual(state["topics"]["intro_children"]["value"], False)
        self.assertEqual(state["topics"]["birth_mode"]["state"], "not_applicable")

    def test_a_no_phrase_means_never_pregnant_and_a_yes_phrase_is_a_family_yes(self):
        never = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.ever_pregnant", "नहीं जी")],
            1,
            utterance="नहीं जी, कभी गर्भ नहीं रहा।",
        )
        self.assertIs(never["profile"]["ever_pregnant"], False)
        self.assertEqual(never["topics"]["fam_miscarriage"]["state"], "not_applicable")

        pregnant = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.ever_pregnant", "जी हाँ")],
            1,
            utterance="जी हाँ।",
        )
        self.assertIs(pregnant["profile"]["ever_pregnant"], True)

        family = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [{"age": {"value": 4, "unit": "years"}, "sex": "female"}])],
            1,
        )
        family = _ask(family, "fam_related")
        family = call_state.apply_updates(
            family,
            [_update("topic.fam_related", "जी हाँ")],
            2,
            utterance="जी हाँ, कज़िन हैं।",
        )
        self.assertIsNone(call_state._skip_reason(family, "fam_same_problem"))

    def test_marital_status_hindi_words(self):
        self.assertEqual(call_state._canonical_marital("विवाहित"), "married")
        self.assertEqual(call_state._canonical_marital("शादीशुदा"), "married")
        self.assertEqual(call_state._canonical_marital("shadishuda"), "married")

    def test_half_a_year_stays_years_and_six_months_stay_months(self):
        months = call_state._spoken_age_records("बड़ी बेटी छह महीने की है।")
        half = call_state._spoken_age_records("एक आधा साल का है।")
        self.assertEqual(months, [{"value": 6, "unit": "months"}])
        self.assertEqual(half, [{"value": 0.5, "unit": "years"}])
        self.assertAlmostEqual(call_state._child_age_years(months[0]), 0.5)
        self.assertAlmostEqual(call_state._child_age_years(half[0]), 0.5)

    def test_hindi_half_step_ages_keep_their_year_value(self):
        samples = {
            "डेढ़ साल का है।": 1.5,
            "ढाई साल का है।": 2.5,
            "साढ़े तीन साल का है।": 3.5,
            "सवा दो साल का है।": 2.25,
            "पौने दो साल का है।": 1.75,
        }
        for utterance, years in samples.items():
            records = call_state._spoken_age_records(utterance)
            self.assertEqual(len(records), 1, utterance)
            self.assertEqual(records[0]["unit"], "years", utterance)
            self.assertAlmostEqual(records[0]["value"], years, msg=utterance)

    def test_sex_words_bind_only_when_each_child_is_named(self):
        both = _ask(call_state.new_call_state(), "intro_children")
        both = call_state.apply_updates(
            both,
            [_update("profile.children", 2)],
            1,
            utterance="बड़ी बेटी छह महीने की है और छोटा बेटा दो साल का है।",
        )
        self.assertEqual([c["sex"] for c in both["profile"]["children"]], ["female", "male"])
        self.assertEqual(both["profile"]["children"][0]["age"], {"value": 6, "unit": "months"})

        one_named = _ask(call_state.new_call_state(), "intro_children")
        one_named = call_state.apply_updates(
            one_named,
            [_update("profile.children", 3)],
            1,
            utterance="तीन बच्चे हैं। एक आधा साल का है, एक चार साल का है, और बेटी ग्यारह साल की है।",
        )
        self.assertEqual([c["sex"] for c in one_named["profile"]["children"]], [None, None, None])

    def test_a_full_term_answer_is_not_a_finding(self):
        full = _ask(call_state.new_call_state(), "birth_term")
        full = call_state.apply_updates(
            full,
            [_update("topic.birth_term", True)],
            1,
            utterance="पूरे नौ महीने में हुआ, समय पर।",
        )
        self.assertEqual(call_state.topic_answer_kind(full["topics"]["birth_term"], "birth_term"), "no")
        early = _ask(call_state.new_call_state(), "birth_term")
        early = call_state.apply_updates(
            early,
            [_update("topic.birth_term", True)],
            1,
            utterance="समय से पहले सात महीने में हुआ।",
        )
        self.assertEqual(call_state.topic_answer_kind(early["topics"]["birth_term"], "birth_term"), "yes")
        late = _ask(call_state.new_call_state(), "birth_term")
        late = call_state.apply_updates(
            late,
            [_update("topic.birth_term", True)],
            1,
            utterance="दस महीने में हुआ, देर से।",
        )
        self.assertEqual(call_state.topic_answer_kind(late["topics"]["birth_term"], "birth_term"), "yes")

    def test_a_count_of_times_stated_with_the_answer_is_not_asked_again(self):
        state = _ask(call_state.new_call_state(), "body_transfusion")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_transfusion", True)],
            1,
            utterance="दो बार खून चढ़ाना पड़ा, पिछली बार पिछले साल।",
        )
        self.assertEqual(state["topics"]["body_transfusion"]["details"].get("count"), 2)
        pending = [item["id"] for item in call_state._pending_followups(state, "body_transfusion")]
        self.assertNotIn("count", pending)

    def test_a_kinship_word_is_a_family_yes_and_a_denial_is_not(self):
        cousin = {"state": "answered", "value": "कज़िन", "details": {}}
        self.assertEqual(call_state.topic_answer_kind(cousin, "fam_related"), "yes")
        uncle = {"state": "answered", "value": "मामा", "details": {"condition": "बीपी"}}
        self.assertEqual(call_state.topic_answer_kind(uncle, "fam_bp_stroke"), "yes")
        denied = {"state": "answered", "value": "नहीं, मामा को नहीं", "details": {}}
        self.assertEqual(call_state.topic_answer_kind(denied, "fam_bp_stroke"), "no")

    def test_hindi_sex_words_count_as_female_or_male(self):
        for word in ("लड़की", "बेटी", "बच्ची"):
            state = call_state.apply_updates(
                call_state.new_call_state(),
                [_update("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": word}])],
                1,
            )
            self.assertEqual(state["profile"]["children"][0]["sex"], "female", word)
            state = _ask(state, "body_bleeding_bruise")
            state = call_state.apply_updates(
                state,
                [_update("topic.body_bleeding_bruise", False)],
                2,
                utterance="नहीं।",
            )
            pending = [item["id"] for item in call_state._pending_followups(state, "body_bleeding_bruise")]
            self.assertIn("periods", pending, word)
        state = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": "लड़का"}])],
            1,
        )
        self.assertEqual(state["profile"]["children"][0]["sex"], "male")
        state = _ask(state, "body_bleeding_bruise")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False)],
            2,
            utterance="नहीं।",
        )
        pending = [item["id"] for item in call_state._pending_followups(state, "body_bleeding_bruise")]
        self.assertNotIn("periods", pending)

    def test_a_described_period_problem_is_a_finding_after_a_no(self):
        state = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": "लड़की"}])],
            1,
        )
        state = _ask(state, "body_bleeding_bruise")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False)],
            2,
            utterance="नहीं।",
        )
        state = _ask(state, "body_bleeding_bruise", "periods")
        state = call_state.apply_updates(
            state,
            [_update("topic.body_bleeding_bruise", False, details={"periods": "बहुत ज़्यादा"})],
            3,
            utterance="माहवारी में बहुत ज़्यादा खून आता है।",
        )
        saved = state["topics"]["body_bleeding_bruise"]
        self.assertEqual(saved["details"].get("periods"), "बहुत ज़्यादा")
        self.assertEqual(call_state.topic_answer_kind(saved, "body_bleeding_bruise"), "yes")
        import mother_report
        state["close_reason"] = "complete"
        yes = " ".join(mother_report.build({"state": state, "started_at": "2026-11-08 14:10:00"})["yes"])
        self.assertIn("बहुत ज़्यादा", yes)
        self.assertNotIn("नहीं", yes)

        quiet = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": "लड़की"}])],
            1,
        )
        quiet = _ask(quiet, "body_bleeding_bruise")
        quiet = call_state.apply_updates(
            quiet,
            [_update("topic.body_bleeding_bruise", False)],
            2,
            utterance="नहीं।",
        )
        quiet = _ask(quiet, "body_bleeding_bruise", "periods")
        quiet = call_state.apply_updates(
            quiet,
            [_update("topic.body_bleeding_bruise", False)],
            3,
            utterance="नहीं।",
        )
        self.assertEqual(
            call_state.topic_answer_kind(quiet["topics"]["body_bleeding_bruise"], "body_bleeding_bruise"),
            "no",
        )

    def test_yes_and_no_phrases_keep_their_polarity(self):
        for value in ("जी हाँ", "हाँ जी", "haan ji", "yes"):
            topic = {"state": "answered", "value": value, "details": {}}
            self.assertEqual(call_state.topic_answer_kind(topic, "birth_nicu"), "yes", value)
        for value in ("नहीं जी", "जी नहीं", "nahi ji"):
            topic = {"state": "answered", "value": value, "details": {}}
            self.assertEqual(call_state.topic_answer_kind(topic, "birth_nicu"), "no", value)
        mixed = {"state": "answered", "value": "हाँ, नहीं पता", "details": {}}
        self.assertNotEqual(call_state.topic_answer_kind(mixed, "birth_nicu"), "yes")
        for value in ("ठीक है", "theek hai", "कोई दिक्कत नहीं"):
            topic = {"state": "answered", "value": value, "details": {}}
            self.assertEqual(call_state.topic_answer_kind(topic, "birth_nicu"), "no", value)
        girl = call_state.apply_updates(
            call_state.new_call_state(),
            [_update("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": "female"}])],
            1,
        )
        girl = _ask(girl, "body_bleeding_bruise")
        girl = call_state.apply_updates(
            girl,
            [_update("topic.body_bleeding_bruise", "ठीक है")],
            2,
            utterance="ठीक है।",
        )
        pending = [item["id"] for item in call_state._pending_followups(girl, "body_bleeding_bruise")]
        self.assertIn("periods", pending)

        growth = _ask(call_state.new_call_state(), "body_growth")
        growth = call_state.apply_updates(
            growth,
            [_update("topic.body_growth", "जी हाँ")],
            1,
            utterance="जी हाँ, बाकी बच्चों जैसा बढ़ रहा है।",
        )
        self.assertEqual(call_state.topic_answer_kind(growth["topics"]["body_growth"], "body_growth"), "no")

        bleeding = call_state.new_call_state()
        bleeding = call_state.apply_updates(
            bleeding,
            [_update("profile.children", [{"age": {"value": 11, "unit": "years"}, "sex": "female"}])],
            1,
        )
        bleeding = _ask(bleeding, "body_bleeding_bruise")
        bleeding = call_state.apply_updates(
            bleeding,
            [_update("topic.body_bleeding_bruise", "नहीं जी")],
            2,
            utterance="नहीं जी।",
        )
        pending = [item["id"] for item in call_state._pending_followups(bleeding, "body_bleeding_bruise")]
        self.assertIn("periods", pending)

    def test_normal_framed_answers_are_not_findings(self):
        growth_ok = _ask(call_state.new_call_state(), "body_growth")
        growth_ok = call_state.apply_updates(
            growth_ok,
            [_update("topic.body_growth", True)],
            1,
            utterance="हाँ, बाकी बच्चों जैसा बढ़ रहा है।",
        )
        self.assertEqual(call_state.topic_answer_kind(growth_ok["topics"]["body_growth"], "body_growth"), "no")
        self.assertFalse(call_state._is_positive(growth_ok["topics"]["body_growth"], "body_growth"))

        growth_bad = _ask(call_state.new_call_state(), "body_growth")
        growth_bad = call_state.apply_updates(
            growth_bad,
            [_update("topic.body_growth", True)],
            1,
            utterance="नहीं, बाकी बच्चों जैसा नहीं बढ़ रहा।",
        )
        self.assertEqual(call_state.topic_answer_kind(growth_bad["topics"]["body_growth"], "body_growth"), "yes")

        puberty = _ask(call_state.new_call_state(), "body_puberty")
        puberty = call_state.apply_updates(
            puberty,
            [_update("topic.body_puberty", True)],
            1,
            utterance="हाँ, दूसरे बच्चों के साथ ही शुरू हुआ।",
        )
        self.assertEqual(call_state.topic_answer_kind(puberty["topics"]["body_puberty"], "body_puberty"), "no")

        early = _ask(call_state.new_call_state(), "body_puberty")
        early = call_state.apply_updates(
            early,
            [_update("topic.body_puberty", True)],
            1,
            utterance="हाँ, समय से पहले शुरू हुआ।",
        )
        self.assertEqual(call_state.topic_answer_kind(early["topics"]["body_puberty"], "body_puberty"), "yes")

        thal = _ask(call_state.new_call_state(), "preg_thalassemia")
        thal = call_state.apply_updates(
            thal,
            [_update("topic.preg_thalassemia", True)],
            1,
            utterance="हाँ, दोनों की जाँच हुई और रिपोर्ट नॉर्मल थी।",
        )
        self.assertEqual(call_state.topic_answer_kind(thal["topics"]["preg_thalassemia"], "preg_thalassemia"), "no")


if __name__ == "__main__":
    unittest.main()
