"""Adversarial invariants for the v4 FormBharo engine.

Deterministic FormBharo edge cases first, then a 500-call seeded fuzzer asserting
hard system invariants on every step.
"""

import random
import unittest

from models import (
    AnswerStatus, CallState, ChildInput, ChildSex, CloseReason, HoldReason, MaritalStatus, Phase,
    ProfilePatch, SideAction, SubStep, TurnCommand, TurnOutcome,
)

import copy_hi
import engine
import triage_catalog
from triage_catalog import FOLLOWUP_HI, PROFILE_QIDS, spec_or_none

MARKER = "MARKER_DIAG_SECRET"
SEED = 20261115

def kids(*rows):
    """ChildInput tuples from compact (age, sex, name) rows."""
    return tuple(ChildInput(age_years=age, sex=ChildSex(sex), name=name) for age, sex, name in rows)

def start(name="पूजा", marital=MaritalStatus.MARRIED, children=None, *, count=None):
    """Walk the intro with one demographic patch per turn, exactly as a call would."""
    patches = [ProfilePatch(caller_name=name), ProfilePatch(marital_status=marital)]
    if children is not None or count is not None:
        patches.append(ProfilePatch(children=children, child_count=count))
    st = engine.new_state()
    for patch in patches:
        st = engine.reduce_turn(st, TurnCommand(profile_patch=patch))
    return st

def step(st, outcome, **kw):
    return engine.reduce_turn(st, TurnCommand(outcome=TurnOutcome(outcome), **kw))

def clear_to(st, qid, limit=100):
    """Answer everything clear until `qid` is the active main question."""
    for _ in range(limit):
        if st.phase is not Phase.ASKING:
            return st
        if (st.cursor.qid, st.cursor.substep, st.cursor.hold) == (qid, SubStep.MAIN, None):
            return st
        if engine.step_kind(st) == "profile":
            raise AssertionError(f"profile hold at {st.cursor.qid} while seeking {qid}")
        before = st.step_token
        st = step(st, "clear")
        if st.step_token == before:
            raise AssertionError(f"stuck at {before}")
    raise AssertionError(f"never reached {qid}, at {st.cursor.qid}")

def skipped(st, qid):
    """True when that item is currently stamped NOT_APPLICABLE."""
    return qid in st.answers and st.answers[qid].state is AnswerStatus.NOT_APPLICABLE

def fingerprint(st: CallState) -> tuple:
    """Cheap structural identity: catches any in-place edit the engine might make."""
    cur, prof = st.cursor, st.profile
    answers = tuple((q, r.state, tuple(r.child_ids), r.note.verbatim, r.note.mother_safe,
                     tuple(sorted(r.followups.items()))) for q, r in st.answers.items())
    return (st.phase, st.close_reason, st.last_answered_qid, st.emergency_note,
            cur.qid, cur.substep, cur.followup_idx, cur.hold, cur.attempts, cur.close_rounds,
            prof.caller_name, prof.marital_status, prof.ever_pregnant, prof.children_stated,
            prof.pending_child_count, tuple(c.id for c in prof.children),
            tuple(c.age_years for c in prof.children), answers)

class DemographicMatrixTests(unittest.TestCase):
    def test_unmarried_skips_child_and_pregnancy_branches(self):
        st = start(marital=MaritalStatus.UNMARRIED)
        self.assertEqual(st.cursor.qid, "intro_bridge")
        self.assertIs(st.profile.ever_pregnant, False)
        child_scoped = [s.id for s in triage_catalog.CATALOG if s.is_child_scoped]
        self.assertEqual(len(child_scoped), 36)
        historical = ("intro_children", "ever_pregnant", "preg_conditions", "preg_scan",
                      "preg_transfusion", "preg_delivery", "fam_miscarriage", "fam_related")
        off = [qid for qid in (*child_scoped, *historical) if not skipped(st, qid)]
        self.assertEqual(off, [], "these must be NOT_APPLICABLE for an unmarried caller")
        self.assertFalse(skipped(st, "preg_thalassemia"), "thalassemia is still asked")

    def test_about_to_marry_asks_fam_related_with_engaged_wording(self):
        st = clear_to(start(marital=MaritalStatus.ABOUT_TO_MARRY), "preg_thalassemia")
        self.assertIn("होने वाले पति", engine.active_prompt_hi(st))
        self.assertTrue(skipped(st, "ever_pregnant"))
        st = step(st, "clear")
        self.assertEqual(st.cursor.qid, "fam_related")
        self.assertEqual(engine.active_prompt_hi(st), triage_catalog.get_spec("fam_related").question_about_hi)

    def test_married_no_kids_routing_gate_clear_vs_concern(self):
        childless = clear_to(start(children=kids()), "ever_pregnant")
        self.assertEqual(childless.cursor.qid, "ever_pregnant")
        self.assertTrue(engine.expect_yes_no(childless))
        cleared = step(childless, "clear")
        self.assertIs(cleared.profile.ever_pregnant, False)
        self.assertEqual(cleared.cursor.qid, "preg_thalassemia")
        history = ("preg_conditions", "preg_scan", "preg_transfusion", "preg_delivery", "fam_miscarriage")
        self.assertEqual([q for q in history if not skipped(cleared, q)], [])
        yes = step(childless, "concern")
        self.assertIs(yes.profile.ever_pregnant, True)
        self.assertEqual(yes.cursor.qid, "preg_conditions")
        self.assertFalse(skipped(yes, "preg_conditions"))
        self.assertTrue(engine.expect_yes_no(yes))

    def test_child_scoped_side_uses_first_unfilled_slot(self):
        two = kids((6.0, "male", None), (2.0, "female", None))
        st = step(clear_to(start(children=two), "birth_nicu"), "concern", child_ref="child_2")
        self.assertEqual(st.answers["birth_nicu"].child_ids, ["child_2"])
        out = engine.reduce_turn(st, TurnCommand(
            outcome=TurnOutcome.CONCERN, child_ref="child_2", side_qid="birth_nicu",
            side_action=SideAction.ADD_CONCERN, utterance="तीन दिन"))
        note = out.answers["birth_nicu"]
        self.assertEqual((note.child_ids, note.followups["nicu_reason"].note.verbatim), (["child_2"], "तीन दिन"))

class ProfileMutationTests(unittest.TestCase):
    def test_midcall_unmarried_to_married_reopens_children_and_keeps_answers(self):
        st = clear_to(start(marital=MaritalStatus.UNMARRIED), "fam_infant_death")
        one = kids((4.0, "male", None))
        moved = engine.reduce_turn(st, TurnCommand(profile_patch=ProfilePatch(
            marital_status=MaritalStatus.MARRIED, child_count=1, children=one)))
        self.assertEqual((moved.cursor.qid, moved.profile.ever_pregnant), ("birth_mode_detail", True))
        self.assertEqual(moved.answers["preg_thalassemia"].state, AnswerStatus.CLEAR)
        self.assertFalse(skipped(moved, "intro_children"))

        childless = engine.reduce_turn(st, TurnCommand(profile_patch=ProfilePatch(
            marital_status=MaritalStatus.MARRIED, child_count=0, children=())))
        self.assertIsNone(childless.profile.ever_pregnant)
        self.assertEqual(childless.cursor.qid, "ever_pregnant")
        self.assertNotIn("ever_pregnant", childless.answers)

    def test_child_count_without_ages_holds_then_resumes_the_same_gate(self):
        st = clear_to(start(children=kids((4.0, "male", None))), "birth_nicu")
        partial = kids((4.0, "male", None))
        held = engine.reduce_turn(st, TurnCommand(profile_patch=ProfilePatch(child_count=2, children=partial)))
        self.assertIs(held.cursor.hold, HoldReason.CHILDREN_AGES)
        self.assertEqual(engine.step_kind(held), "profile")
        self.assertFalse(engine.expect_yes_no(held))
        self.assertIn("उम्र", engine.say_hindi(held))
        still = step(held, "clear")
        self.assertIs(still.cursor.hold, HoldReason.CHILDREN_AGES)
        self.assertNotIn("birth_nicu", still.answers, "no gate answer while ages are missing")
        resumed = engine.reduce_turn(still, TurnCommand(profile_patch=ProfilePatch(
            child_count=2, children=kids((4.0, "male", None), (1.0, "female", None)))))
        self.assertIsNone(resumed.cursor.hold)
        self.assertEqual((resumed.cursor.qid, len(resumed.profile.children)), ("birth_nicu", 2))

    def test_three_children_middle_child_preabsorption_and_retraction(self):
        three = kids((2.0, "male", None), (8.0, "female", "रिया"), (5.0, "male", "अर्जुन"))
        roster_src = start(children=three)
        self.assertEqual([r["rank"] for r in engine.children_roster(roster_src)],
                         ["oldest", "middle", "youngest"])
        asked = step(clear_to(roster_src, "birth_jaundice"), "concern")
        self.assertIs(asked.cursor.substep, SubStep.WHICH_CHILD)
        self.assertIn("बीच", engine.active_prompt_hi(asked))
        self.assertFalse(engine.expect_yes_no(asked), "a which_child step is not yes/no")
        picked = step(asked, "concern", child_ref="child_3")
        self.assertEqual((picked.answers["birth_jaundice"].child_ids, picked.cursor.substep),
                         (["child_3"], SubStep.FOLLOWUP))
        self.assertEqual(engine.active_prompt_hi(picked), FOLLOWUP_HI["jaundice_days"])
        pre = engine.reduce_turn(clear_to(roster_src, "birth_nicu"), TurnCommand(
            outcome=TurnOutcome.CONCERN, followup_answered=True, utterance="साँस नहीं ले रहा"))
        self.assertIs(pre.cursor.substep, SubStep.WHICH_CHILD)
        self.assertEqual(pre.answers["birth_nicu"].followups["nicu_reason"].note.verbatim, "साँस नहीं ले रहा")
        absorbed = step(pre, "concern", child_ref="child_2")
        self.assertEqual((absorbed.answers["birth_nicu"].child_ids, absorbed.cursor.substep),
                         (["child_2"], SubStep.FOLLOWUP))
        self.assertEqual(engine.active_prompt_hi(absorbed), FOLLOWUP_HI["nicu_days"])

        plain = start(children=kids((2.0, "unknown", None), (8.0, "unknown", None), (5.0, "unknown", None)))
        retracted = step(step(clear_to(plain, "body_bleeding_bruise"), "concern"), "clear")
        self.assertEqual(retracted.answers["body_bleeding_bruise"].state, AnswerStatus.CLEAR)
        self.assertIs(retracted.cursor.substep, SubStep.MAIN)
        self.assertNotEqual(retracted.cursor.qid, "body_bleeding_bruise", "a retraction advances")

class SideTurnTests(unittest.TestCase):
    def _fam(self):
        """A family concern already recorded, with the cursor on a later gate."""
        st = step(step(clear_to(start(children=kids((4.0, "male", None))), "fam_cancer"), "concern",
                       utterance="मामा को कैंसर"), "concern", utterance="मामा")
        return clear_to(st, "fam_heart_kidney")

    def test_add_concern_keeps_cursor_and_fills_followup(self):
        st = self._fam()
        out = engine.reduce_turn(st, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, side_qid="fam_cancer",
                                                side_action=SideAction.ADD_CONCERN, diagnostic=True,
                                                utterance="चाची को भी कैंसर"))
        self.assertEqual(out.cursor.qid, "fam_heart_kidney")
        self.assertIn("चाची को भी कैंसर", out.answers["fam_cancer"].verbatim_text)
        self.assertNotIn(MARKER, out.answers["fam_cancer"].mother_safe_text)

    def test_clear_or_retract_revises_without_moving(self):
        st = self._fam()
        out = engine.reduce_turn(st, TurnCommand(
            outcome=TurnOutcome.NOT_ABOUT_ACTIVE, side_qid="fam_cancer",
            side_action=SideAction.CLEAR_OR_RETRACT))
        self.assertEqual(out.cursor.qid, "fam_heart_kidney")
        self.assertEqual(out.answers["fam_cancer"].state, AnswerStatus.CLEAR)

    def test_clarify_stashes_clean_answer_then_rewinds(self):
        st = step(clear_to(start(children=kids((4.0, "male", None))), "preg_thalassemia"), "unknown")
        self.assertEqual(st.cursor.qid, "fam_related")
        out = engine.reduce_turn(st, TurnCommand(outcome=TurnOutcome.CLEAR, side_qid="preg_thalassemia",
                                                side_action=SideAction.CLARIFY_MEANING))
        self.assertEqual(out.cursor.qid, "preg_thalassemia")
        self.assertEqual(out.answers["fam_related"].state, AnswerStatus.CLEAR)
        self.assertNotIn("preg_thalassemia", out.answers)
        self.assertIn("थैलेसीमिया", engine.active_prompt_hi(out))

    def test_child_scoped_side_isolates_both_child_concerns(self):
        st = clear_to(start(children=kids((6.0, "male", None), (2.0, "female", None))), "birth_jaundice")
        out = engine.reduce_turn(st, TurnCommand(outcome=TurnOutcome.CONCERN, child_ref="child_2",
                                                side_qid="birth_nicu", side_action=SideAction.ADD_CONCERN,
                                                utterance="काँच के बक्से में"))
        side = out.answers["birth_nicu"]
        self.assertEqual(side.child_ids, [], "both-child concern must not borrow child_ref")
        self.assertIn("काँच के बक्से में", side.verbatim_text)
        self.assertEqual(out.answers["birth_jaundice"].child_ids, ["child_2"])

class OpenCaptureTests(unittest.TestCase):
    def test_bare_concern_asks_detail_then_close_more_then_finishes(self):
        st = clear_to(start(children=kids((4.0, "male", None))), "close_anything_else")
        self.assertEqual(engine.step_kind(st), "close")
        bare = step(st, "concern")
        self.assertEqual((bare.phase, bare.cursor.substep), (Phase.ASKING, SubStep.FOLLOWUP))
        self.assertEqual(engine.active_prompt_hi(bare), FOLLOWUP_HI["open_concern_detail"])
        noted = step(bare, "concern", utterance="पेट में दर्द")
        self.assertEqual(engine.active_prompt_hi(noted), copy_hi.CLOSE_MORE_HI)
        self.assertIn("पेट में दर्द", noted.answers["close_anything_else"].verbatim_text)
        done = step(noted, "clear")
        self.assertEqual((done.phase, done.close_reason), (Phase.CLOSING, CloseReason.COMPLETE))
        self.assertEqual(done.answers["close_anything_else"].state, AnswerStatus.CONCERN)
        self.assertIn("पेट में दर्द", done.answers["close_anything_else"].mother_safe_text)
        self.assertTrue(engine.say_hindi(done).strip())

class RetryCapTests(unittest.TestCase):
    def test_retry_counter_always_resets_at_the_cap(self):
        st = clear_to(start(children=kids((4.0, "male", None))), "birth_nicu")
        st = step(st, "concern")
        self.assertIs(st.cursor.substep, SubStep.FOLLOWUP)
        self.assertEqual(st.cursor.followup_idx, 0)
        seen = []
        for _ in range(2):
            st = step(st, "not_about_active")
            seen.append(st.cursor.attempts)
        self.assertEqual((seen, max(seen) < engine.MAX_STEP_ATTEMPTS), ([1, 0], True), "cap resets, never persists")
        self.assertEqual(st.cursor.followup_idx, 1, "two failed attempts must move on")
        self.assertEqual(st.answers["birth_nicu"].followups["nicu_reason"].state, AnswerStatus.UNKNOWN)

    def test_which_child_cap_marks_unknown_and_moves_to_the_followup(self):
        two = start(children=kids((6.0, "male", None), (2.0, "female", None)))
        st = step(clear_to(two, "birth_nicu"), "concern")
        self.assertIs(st.cursor.substep, SubStep.WHICH_CHILD)
        once = step(st, "not_about_active")
        self.assertEqual((once.cursor.substep, once.cursor.attempts), (SubStep.WHICH_CHILD, 1))
        twice = step(once, "not_about_active")
        self.assertEqual((twice.cursor.attempts, twice.cursor.substep), (0, SubStep.FOLLOWUP))
        self.assertEqual(twice.answers["birth_nicu"].child_ids, [])

def profile_patch(rng, st):
    """Random demographic mutation, including marital flips and partial counts."""
    if rng.random() > 0.30:
        return ProfilePatch(), rng.random() < 0.15
    kind = rng.choice(["name", "marital", "kids", "count_only"])
    if kind == "name":
        return ProfilePatch(caller_name=rng.choice(["कविता", "   ", "सुनीता"])), True
    if kind == "marital":
        return ProfilePatch(marital_status=rng.choice(list(MaritalStatus))), True
    n = rng.choice([0, 1, 2, 3])
    ages, sexes = [0.5, 2.0, 5.0, 8.0, 12.0], list(ChildSex)
    rows = tuple(ChildInput(age_years=rng.choice(ages), sex=rng.choice(sexes), name=rng.choice([None, "रीना"]))
                 for _ in range(n if kind == "kids" else max(0, n - 1)))
    return ProfilePatch(child_count=n, children=None if kind == "count_only" else rows), True

def random_command(rng, st):
    """One hostile turn: random outcome, side intent, child ref, and poisoned note."""
    roll = rng.random()
    if roll < 0.015:
        return TurnCommand(emergency=True)
    if roll < 0.03:
        return TurnCommand(caller_stop=True)
    patch, attempted = profile_patch(rng, st)
    outcome = rng.choice([TurnOutcome.CLEAR, TurnOutcome.CONCERN, TurnOutcome.UNKNOWN,
                          TurnOutcome.CLARIFY, TurnOutcome.RETRACT, TurnOutcome.NOT_ABOUT_ACTIVE])
    qids, marker = list(st.answers) or ["intro_name"], rng.random() < 0.25
    return TurnCommand(
        outcome=outcome, profile_patch=patch, profile_turn_attempted=attempted,
        child_ref=rng.choice(["", "all", "child_1", "child_2", "child_3", "नहीं"]),
        followup_answered=rng.random() < 0.30,
        utterance=f"घर पर ठीक है {MARKER}" if marker else "कुछ बात",
        side_qid=rng.choice(["", *rng.sample(qids, min(3, len(qids)))]),
        side_action=rng.choice([SideAction.NONE, SideAction.ADD_CONCERN,
                                SideAction.CLEAR_OR_RETRACT, SideAction.CLARIFY_MEANING]),
        diagnostic=marker or rng.random() < 0.20,
    )

class FuzzerInvariantTests(unittest.TestCase):
    CALLS, STEPS = 500, 160

    def test_500_seeded_calls_hold_every_invariant(self):
        rng, closed = random.Random(SEED), 0
        for call in range(self.CALLS):
            straight = call < 50          # a monotonic subset that must terminate
            st = self.straight_start(rng) if straight else engine.new_state()
            poisoned = False
            for index in range(self.STEPS):
                cmd = self.straighten(rng, random_command(rng, st)) if straight else random_command(rng, st)
                poisoned = poisoned or MARKER in cmd.utterance
                before = fingerprint(st)
                nxt = engine.reduce_turn(st, cmd)
                if fingerprint(st) != before:
                    self.fail("engine mutated its input CallState")
                if call < 40 or index % 5 == 0:
                    snap = nxt.to_dict()
                    if CallState.from_dict(snap) != nxt:
                        self.fail("state is not JSON-stable")
                if not 0 <= nxt.cursor.attempts < engine.MAX_STEP_ATTEMPTS:
                    self.fail(f"retry counter escaped its cap: {nxt.cursor.attempts}")
                self.check_na(nxt)
                self.check_yes_no(nxt)
                spoken = engine.say_hindi(nxt)
                if not (isinstance(spoken, str) and spoken.strip()):
                    self.fail(f"engine spoke nothing at {nxt.step_token}")
                if poisoned and any(MARKER in rec.mother_safe_text for rec in nxt.answers.values()):
                    self.fail(f"diagnostic marker reached Mother copy at {nxt.step_token}")
                if poisoned and any(MARKER in line for line in copy_hi.recorded_concerns_hi(nxt)):
                    self.fail(f"diagnostic marker reached recorded_concerns at {nxt.step_token}")
                st = nxt
                if st.phase is Phase.CLOSING:
                    break
            if straight:
                self.assertIs(st.phase, Phase.CLOSING, f"straight call {call} never closed")
                self.assertLess(index, 130, f"straight call {call} took {index} steps")
                closed += 1
        self.assertEqual(closed, 50)

    @staticmethod
    def straighten(rng, cmd):
        """No clarify, no side turns, no profile re-flips: monotone yes/no progress."""
        stalled = (TurnOutcome.CLARIFY, TurnOutcome.RETRACT, TurnOutcome.NOT_ABOUT_ACTIVE)
        outcome = TurnOutcome.CONCERN if rng.random() < 0.35 else (
            TurnOutcome.CLEAR if cmd.outcome in stalled else cmd.outcome)
        return TurnCommand(outcome=outcome, child_ref=rng.choice(["", "all", "child_1", "child_2"]),
                           followup_answered=rng.random() < 0.4, utterance="कुछ बात",
                           diagnostic=rng.random() < 0.3)

    def straight_start(self, rng):
        """A fully specified profile, so monotone turns can walk the whole spine."""
        rows = ((4.0, rng.choice(["male", "female"]), None), (9.0, "female", None))
        return start(children=kids(*rows) if rng.random() < 0.5 else ())

    def check_na(self, st):
        """No NOT_APPLICABLE may survive on an item whose skip rule no longer fires."""
        stale = [q for q, r in st.answers.items() if r.state is AnswerStatus.NOT_APPLICABLE
                 and not engine._skip_hits(spec_or_none(q), st)]
        self.assertEqual(stale, [], "stale NOT_APPLICABLE on an applicable item")

    def check_yes_no(self, st):
        """Buttons may only be live on a plain yes/no gate she can actually answer."""
        allowed, cur, spec = engine.expect_yes_no(st), st.cursor, spec_or_none(st.cursor.qid)
        gate = (st.phase is Phase.ASKING and cur.hold is None and cur.substep is SubStep.MAIN
                and cur.qid not in PROFILE_QIDS and spec is not None and spec.allows_yes_no)
        self.assertEqual(allowed, gate, f"yes/no mis-offered at {st.step_token}")

if __name__ == "__main__":
    unittest.main()
