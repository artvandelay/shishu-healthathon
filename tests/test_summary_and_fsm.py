"""Stage 3A: single-pass ScreeningSummary projection and the pure live turn machine."""

import unittest

import tests  # noqa: F401

import copy_hi  # noqa: E402
import mother_report  # noqa: E402
import post_call  # noqa: E402
import summary  # noqa: E402
from engine import apply_profile_patch, reduce_turn  # noqa: E402
from live_fsm import (  # noqa: E402
    CHECK_IN_SEC, CLOSING_AUDIO_GRACE_SEC, CLOSING_MAX_WAIT_SEC, HARD_LIMIT_SEC,
    MAX_WATCHDOG_NUDGES, SILENCE_CLOSE_SEC, SOFT_LIMIT_SEC, LiveTurnMachine,
    SpeakKind, TurnPhase, WatchdogAction,
)
from models import (  # noqa: E402
    CallState, ChildInput, ChildSex, MaritalStatus, Phase, ProfilePatch, TurnCommand, TurnOutcome,
)
from triage_catalog import get_spec  # noqa: E402

_CONCERN_QID = "body_bleeding_bruise"
_SAFE_NOTE = "चोट पर खून देर तक बहता है"
_DIAGNOSTIC_NOTE = "डॉक्टर ने हीमोफीलिया बताया था"
_FORBIDDEN_HI = "हीमोफीलिया"
_PCM_CHUNK = b"\x01\x00" * 1_600


def _kids(*specs):
    return tuple(
        ChildInput(age_years=age, sex=ChildSex(sex), name=name) for age, sex, name in specs
    )


def _two_kids():
    return _kids((7.0, "male", None), (2.0, "female", "मीरा"))


def _state(children=(), *, name="पूजा"):
    patch = ProfilePatch(
        caller_name=name, marital_status=MaritalStatus.MARRIED, children=tuple(children)
    )
    return apply_profile_patch(CallState(), patch)


def _answer(state, qid, outcome, *, utterance="", child_ref=""):
    """Clear every gate up to `qid`, then answer that one."""
    for _ in range(90):
        if state.phase is not Phase.ASKING or state.cursor.qid == qid:
            break
        state = reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
    command = TurnCommand(outcome=outcome, utterance=utterance, child_ref=child_ref)
    return reduce_turn(state, command)


def _run_to_end(state):
    for _ in range(200):
        if state.phase is not Phase.ASKING:
            break
        state = reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
    return state


def _concern_state(*, children=None, child_ref="child_2", diagnostic=False):
    state = _state(_two_kids() if children is None else children)
    command = dict(utterance=_SAFE_NOTE, child_ref=child_ref)
    state = _answer(state, _CONCERN_QID, TurnOutcome.CONCERN, **command)
    if diagnostic:
        record = state.answers[_CONCERN_QID]
        record.note = record.note.append(_DIAGNOSTIC_NOTE, diagnostic=True)
    return _run_to_end(state)


def _section(markdown, heading):
    return markdown.partition(f"{heading}\n")[2].partition("\n\n### ")[0]


def _legacy_cursor(state):
    """Render a typed CallState the way `db.py` stores `cursor_state`."""
    profile, marital = state.profile, state.profile.marital_status
    return {
        "phase": state.phase.value,
        "close_reason": state.close_reason.value if state.close_reason else None,
        "emergency_note": state.emergency_note,
        "profile": {
            "caller_name": profile.caller_name,
            "marital_status": marital.value if marital else None,
            "children": [child.model_dump(mode="json") for child in profile.children],
            "ever_pregnant": profile.ever_pregnant,
        },
        "active_qid": state.cursor.qid,
        "answers": {qid: _legacy_answer(record) for qid, record in state.answers.items()},
    }


def _legacy_answer(record):
    """One `answers[qid]` entry: verbatim notes for the doctor, safe notes for Mother."""
    return {
        "state": record.state.value,
        "child_ids": list(record.child_ids),
        "notes": record.note.verbatim,
        "mother_notes": record.note.mother_safe,
        "followups": {fid: _legacy_answer_slot(slot) for fid, slot in record.followups.items()},
    }


def _legacy_answer_slot(slot):
    state, note = slot.state.value, slot.note
    return {"state": state, "note": note.verbatim, "mother_note": note.mother_safe}


def _as_dicts(groups):
    return [{"title": group.title, "items": list(group.items)} for group in groups]


class SummaryTests(unittest.TestCase):
    def test_matches_post_call_build_reports_byte_for_byte(self):
        for label, state in {
            "complete": _run_to_end(_state(_two_kids())),
            "concern": _concern_state(),
            "diagnostic": _concern_state(diagnostic=True),
            "caller_stop": reduce_turn(_state(_two_kids()), TurnCommand(caller_stop=True)),
            "emergency": reduce_turn(_state(_two_kids()), TurnCommand(emergency=True)),
            "childless": _run_to_end(_state()),
        }.items():
            with self.subTest(label):
                projected = summary.project_summary(state)
                built = post_call.build_reports({"id": 1, "cursor_state": _legacy_cursor(state)})
                self.assertEqual(projected.doctor_markdown, built["doctor_markdown"])
                self.assertEqual(projected.mother_name, built["mother_name_hi"])
                self.assertEqual(projected.children_label_hi, built["children_label_hi"])
                for attr in ("red_flags", "yes_summaries_hi", "mother_questions_hi"):
                    self.assertEqual(list(getattr(projected, attr)), built[attr])

    def test_matches_mother_report_page2_rows(self):
        for state in (
            _run_to_end(_state(_two_kids())),
            reduce_turn(_state(_two_kids()), TurnCommand(caller_stop=True)),
        ):
            projected = summary.project_summary(state)
            groups, unknown, unasked = mother_report._page2_from_cursor(
                _legacy_cursor(state), projected.incomplete
            )
            self.assertEqual(_as_dicts(projected.page2_clear_groups), groups)
            self.assertEqual(list(projected.page2_unknown), unknown)
            self.assertEqual(_as_dicts(projected.page2_unasked_groups), unasked)

    def test_diagnostic_firewall_keeps_disease_words_out_of_hindi(self):
        projected = summary.project_summary(_concern_state(diagnostic=True))
        self.assertIn("Mother's words:", projected.doctor_markdown)
        self.assertIn(_DIAGNOSTIC_NOTE, projected.doctor_markdown)
        self.assertIn(_DIAGNOSTIC_NOTE, projected.red_flags[0])
        hindi = "\n".join((
            *projected.yes_summaries_hi,
            *projected.mother_questions_hi,
            projected.mother_summary_text,
        ))
        self.assertNotIn(_DIAGNOSTIC_NOTE, hindi)
        self.assertNotIn(_FORBIDDEN_HI, hindi)
        self.assertIn(_SAFE_NOTE, projected.yes_summaries_hi[0])
        text = projected.mother_summary_text
        self.assertTrue(text.startswith("जो बातें हाँ में दर्ज हुईं:\n- "))
        self.assertIn("\n\nडॉक्टर से पूछने के सवाल:\n1. ", text)

    def test_multi_child_tag_on_note_and_hindi_scope_on_mother_lines(self):
        projected = summary.project_summary(_concern_state())
        self.assertIn("[child_2 (female, age 2.0, मीरा)]", projected.doctor_markdown)
        self.assertIn("(मीरा 2 साल बेटी)", projected.yes_summaries_hi[0])
        self.assertIn("child_1 (male, age 7.0), child_2", projected.children_label_en)

    def test_single_child_profile_omits_every_scope_tag(self):
        projected = summary.project_summary(_concern_state(children=_kids((7.0, "male", None))))
        self.assertNotIn("[child_1", projected.doctor_markdown)
        self.assertNotIn("(child_1", projected.yes_summaries_hi[0])
        stem = get_spec(_CONCERN_QID).stem_hi
        self.assertEqual(projected.yes_summaries_hi[0], f"{stem} — {_SAFE_NOTE}।")

    def test_page2_unasked_groups_only_when_incomplete(self):
        complete = summary.project_summary(_run_to_end(_state(_two_kids())))
        self.assertEqual(complete.close_reason, "complete")
        self.assertFalse(complete.incomplete)
        self.assertEqual(complete.page2_unasked_groups, ())
        self.assertTrue(complete.page2_clear_groups)
        stopped = reduce_turn(_state(_two_kids()), TurnCommand(caller_stop=True))
        partial = summary.project_summary(stopped)
        self.assertEqual((partial.close_reason, partial.incomplete), ("caller_stop", True))
        self.assertEqual(partial.page2_clear_groups, ())
        self.assertIn(copy_hi.GROUP_HI["family"], [g.title for g in partial.page2_unasked_groups])
        asked = [i for group in partial.page2_unasked_groups for i in group.items]
        profile = _state(_two_kids()).profile
        self.assertIn(copy_hi.question_hi_for(profile, get_spec("body_blue")), asked)
        self.assertNotIn(copy_hi.question_hi_for(profile, get_spec("body_puberty")), asked)

    def test_emergency_is_incomplete_and_leads_red_flags(self):
        hit = TurnCommand(emergency=True, utterance="सांस नहीं ले रहा")
        projected = summary.project_summary(reduce_turn(_state(_two_kids()), hit))
        self.assertTrue(projected.emergency and projected.incomplete)
        self.assertTrue(projected.red_flags[0].startswith("emergency: active danger sign"))
        self.assertIn("सांस नहीं ले रहा", projected.red_flags[0])
        self.assertIn("Mother's words: सांस नहीं ले रहा", projected.doctor_markdown)
        self.assertTrue(projected.page2_unasked_groups)

    def test_hematology_item_routes_to_its_own_section(self):
        markdown = summary.project_summary(_concern_state()).doctor_markdown
        need_en = get_spec(_CONCERN_QID).need_en
        self.assertEqual(markdown.count("\n### "), 7)
        system = "### Systemic, Infections & Hematology"
        growth = "### Development, Growth & Physical Features"
        self.assertIn(need_en, _section(markdown, system))
        self.assertNotIn(need_en, _section(markdown, "### Birth & Neonatal History"))
        self.assertNotIn(need_en, _section(markdown, growth))

    def test_quiet_complete_call_has_no_yes_lines_and_three_questions(self):
        projected = summary.project_summary(_run_to_end(_state(_kids((7.0, "male", None)))))
        self.assertEqual(projected.yes_summaries_hi, ())
        self.assertEqual(projected.red_flags, ())
        self.assertEqual(len(projected.mother_questions_hi), 3)
        self.assertEqual(
            projected.mother_summary_text.splitlines()[0],
            "इस बातचीत में कोई हाँ वाला बिंदु दर्ज नहीं हुआ।",
        )


class LiveTurnMachineTests(unittest.TestCase):
    def _spoken_machine(self, state, now=100.0):
        machine = LiveTurnMachine(call_started_ts=0.0)
        machine.on_speak_cue_sent(state.step_token, SpeakKind.SPEAK)
        machine.on_bot_audio(state, 4_800, 24_000, now)
        return machine

    def test_echo_shield_ignores_mic_audio_until_playback_ends(self):
        state = _state(_two_kids())
        machine = self._spoken_machine(state)
        self.assertIs(machine.phase, TurnPhase.BOT_SPEAKING)
        self.assertFalse(machine.on_browser_audio_energy(state, _PCM_CHUNK, 0.9, 100.05))
        self.assertTrue(machine.speech_allowed)
        self.assertFalse(machine.pcm_16k)
        machine.on_bot_audio(state, 4_800, 24_000, 100.2)
        machine.on_turn_complete_cleanup(state, 100.25, decision_ran=True)
        self.assertIs(machine.phase, TurnPhase.LISTENING)
        self.assertFalse(machine.on_browser_audio_energy(state, _PCM_CHUNK, 0.9, 100.28))
        self.assertTrue(machine.speech_allowed)
        self.assertTrue(machine.on_browser_audio_energy(state, _PCM_CHUNK, 0.9, 100.4))
        self.assertIs(machine.phase, TurnPhase.CAPTURING_USER)
        self.assertFalse(machine.speech_allowed)
        self.assertEqual(machine.pcm_tail_chunks, 6)
        for step in range(6):
            machine.on_browser_audio_energy(state, _PCM_CHUNK, 0.0, 100.5 + step * 0.01)
        self.assertEqual(machine.pcm_tail_chunks, 0)
        self.assertEqual(len(machine.pcm_16k), 7 * len(_PCM_CHUNK))
        machine.on_browser_audio_energy(state, _PCM_CHUNK, 0.0, 100.7)
        self.assertEqual(len(machine.pcm_16k), 7 * len(_PCM_CHUNK))

    def test_quick_reply_mid_sentence_mutes_leftover_audio_and_stales_late_speech(self):
        state = _state(_two_kids())
        machine = self._spoken_machine(state)
        self.assertTrue(machine.can_accept_quick_reply(state, state.step_token, "concern"))
        machine.on_quick_reply_accepted(state.step_token, 100.2)
        self.assertTrue(machine.ignore_bot_audio)
        self.assertEqual(machine.playback_ends_at, 0.0)
        self.assertFalse(machine.can_accept_bot_output())
        self.assertEqual(machine.on_bot_audio(state, 4_800, 24_000, 100.3), (False, False, None))
        self.assertFalse(machine.on_user_transcript(state, "हाँ", 100.4))
        self.assertTrue(machine.marked_quick_reply_stale)
        self.assertEqual(machine.hold_reason_for_tool(state, 100.5), "stale_turn")
        machine.on_turn_complete_cleanup(state, 101.0, decision_ran=False)
        self.assertFalse(machine.ignore_bot_audio or machine.marked_quick_reply_stale)
        self.assertTrue(machine.can_accept_bot_output())
        self.assertEqual(machine.hold_reason_for_tool(state, 101.0), "")
        self.assertTrue(machine.on_user_transcript(state, "हाँ", 102.0))
        self.assertEqual(machine.text_buffer, "हाँ")

    def test_hold_reasons_and_missed_tool_detection(self):
        state = _state(_two_kids())
        machine = LiveTurnMachine(call_started_ts=0.0)
        self.assertEqual(machine.hold_reason_for_tool(state, 1.0), "not_spoken")
        machine.on_bot_audio(state, 4_800, 24_000, 1.0)
        self.assertFalse(machine.should_run_missed_tool_on_turn_complete(state))
        machine.on_browser_text(state, "हाँ", 2.0)
        self.assertEqual(machine.hold_reason_for_tool(state, 2.0), "")
        self.assertTrue(machine.should_run_missed_tool_on_turn_complete(state))
        machine.begin_evaluation(state)
        self.assertEqual(machine.hold_reason_for_tool(state, 3.0), "duplicate_turn")

    def test_begin_evaluation_locks_the_turn_and_blocks_replays(self):
        state = _state(_two_kids())
        machine = LiveTurnMachine(call_started_ts=0.0)
        self.assertEqual(machine.begin_evaluation(state)[0], "empty_turn")
        self.assertFalse(machine.handled_current_turn)
        machine.on_browser_text(state, "हाँ", 10.0)
        reason, utterance, pcm, snapshot, key = machine.begin_evaluation(state)
        self.assertIsNone(reason)
        self.assertEqual((utterance, pcm), ("हाँ", b""))
        self.assertEqual((snapshot.step_token, key[0]), (state.step_token,) * 2)
        self.assertIs(machine.phase, TurnPhase.EVALUATING)
        self.assertTrue(machine.handled_current_turn)
        machine.finish_evaluation(step_changed=True)
        self.assertIs(machine.phase, TurnPhase.AWAITING_BOT)
        self.assertTrue(machine.speech_allowed)
        machine.text_buffer = "हाँ"
        self.assertEqual(machine.begin_evaluation(state)[0], "duplicate_turn")

    def test_silence_check_in_and_silence_close(self):
        state = _state(_two_kids())
        machine = LiveTurnMachine(call_started_ts=0.0)
        machine.phase = TurnPhase.LISTENING
        machine.quiet_from = 100.0
        poll, act = machine.poll_watchdog, WatchdogAction
        self.assertIs(poll(state, 100.0 + CHECK_IN_SEC - 0.1, 8.0), act.NONE)
        self.assertIs(poll(state, 100.0 + CHECK_IN_SEC, 8.0), act.SILENCE_CHECK_IN)
        self.assertIs(poll(state, 100.0 + SILENCE_CLOSE_SEC, 8.0), act.AUTO_CLOSE_SILENCE)
        self.assertIs(poll(state, SOFT_LIMIT_SEC, 8.0), act.AUTO_CLOSE_TIME)
        self.assertIs(poll(state, HARD_LIMIT_SEC, 8.0), act.HARD_LIMIT_STOP)

    def test_closing_grace_and_closing_max_wait(self):
        state = _state(_two_kids())
        machine = LiveTurnMachine(call_started_ts=1_000.0)
        self.assertTrue(machine.mark_closing(1_000.0))
        self.assertFalse(machine.mark_closing(1_000.4))
        self.assertEqual(machine.end_requested_at, 1_000.0)
        self.assertIs(machine.phase, TurnPhase.CLOSING)
        machine.closing_heard, machine.playback_ends_at = True, 1_001.0
        grace = 1_001.0 + CLOSING_AUDIO_GRACE_SEC
        poll, act = machine.poll_watchdog, WatchdogAction
        self.assertIs(poll(state, grace - 0.05, 8.0), act.NONE)
        self.assertIs(poll(state, grace, 8.0), act.CLOSING_COMPLETE)
        late = 1_000.0 + CLOSING_MAX_WAIT_SEC + 0.1
        self.assertIs(poll(state, late, 8.0), act.CLOSING_COMPLETE)

    def test_speech_end_before_question_arms_six_second_nudge(self):
        state = _state(_two_kids())
        machine = LiveTurnMachine(call_started_ts=500.0)
        self.assertFalse(machine.on_speech_end(state, 510.0))
        self.assertTrue(machine.speech_end_waiting and machine.awaiting_bot_reply)
        self.assertEqual(machine.quiet_from, 510.0)
        poll, act, now = machine.poll_watchdog, WatchdogAction, 516.0
        self.assertIs(poll(state, now - 0.1, 8.0), act.NONE)
        self.assertIs(poll(state, now, 8.0), act.WATCHDOG_NUDGE)
        self.assertEqual(machine.nudges_for_turn, 1)
        for _ in range(MAX_WATCHDOG_NUDGES - 1):
            now += 6.0
            self.assertIs(poll(state, now, 8.0), act.WATCHDOG_NUDGE)
        self.assertEqual(machine.nudges_for_turn, MAX_WATCHDOG_NUDGES)
        self.assertIs(poll(state, now + 6.0, 8.0), act.NONE)

    def test_extractor_failures_escalate_once_per_step(self):
        machine = LiveTurnMachine(call_started_ts=0.0)
        fail = machine.record_extractor_failure
        self.assertEqual([fail("step_a"), fail("step_a"), fail("step_a")], [False, True, False])
        machine.finish_evaluation(step_changed=True)
        self.assertEqual([fail("step_a"), fail("step_a")], [False, True])
        self.assertEqual(machine.extractor_failures, 0)


if __name__ == "__main__":
    unittest.main()
