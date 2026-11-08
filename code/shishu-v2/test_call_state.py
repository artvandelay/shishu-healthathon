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
        self.assertAlmostEqual(call_state._child_age_years(state["profile"]["children"][0]["age"]), 0.5)
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

    def test_test_topics_ask_whether_the_test_was_done_only_after_no(self):
        for topic_id, done_id in (
            ("birth_hearing_vision", "test_done"),
            ("preg_scan", "scan_done"),
            ("preg_thalassemia", "caller_test"),
        ):
            base = call_state.apply_updates(
                call_state.new_call_state(), [_update("profile.children", [{"age": 3}])], 1
            )
            no = call_state.apply_updates(_ask(base, topic_id), [_update(f"topic.{topic_id}", False)], 2)
            self.assertFalse(call_state._is_positive(no["topics"][topic_id]))
            self.assertIn(done_id, [f["id"] for f in call_state._pending_followups(no, topic_id)])
            yes = call_state.apply_updates(_ask(base, topic_id), [_update(f"topic.{topic_id}", True)], 2)
            self.assertTrue(call_state._is_positive(yes["topics"][topic_id]))
            self.assertNotIn(done_id, [f["id"] for f in call_state._pending_followups(yes, topic_id)])

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

    def test_marital_status_hindi_words(self):
        self.assertEqual(call_state._canonical_marital("विवाहित"), "married")
        self.assertEqual(call_state._canonical_marital("शादीशुदा"), "married")
        self.assertEqual(call_state._canonical_marital("shadishuda"), "married")

    def test_three_children_ordinal_and_whole_word_boundaries(self):
        state = call_state.new_call_state()
        children = [
            {"id": "child_1", "name": None, "age": {"value": 8, "unit": "years"}},
            {"id": "child_2", "name": None, "age": {"value": 5, "unit": "years"}},
            {"id": "child_3", "name": None, "age": {"value": 2, "unit": "years"}},
        ]
        state = call_state.apply_updates(state, [_update("profile.children", children)], 1)
        self.assertEqual(call_state._find_child_id(state, "दूसरा बच्चा"), "child_2")
        self.assertEqual(call_state._find_child_id(state, "तीसरा बच्चा"), "child_3")
        self.assertEqual(call_state._find_child_id(state, "सबसे बड़े"), "child_1")

        state = _ask(state, "birth_mode")
        state = call_state.apply_updates(
            state,
            [_update("topic.birth_mode", True)],
            2,
            utterance="अस्पताल में सिजेरियन हुआ, घरेलू नहीं।",
        )
        self.assertEqual(state["topics"]["birth_mode"]["details"].get("delivery_type"), "operation")
        self.assertEqual(state["topics"]["birth_mode"]["details"].get("place"), "hospital")


if __name__ == "__main__":
    unittest.main()
