"""RED-first tests for Stage 3: same-step tap vs speech source arbitration."""

from __future__ import annotations

import asyncio
import unittest

import tests  # noqa: F401 — isolated storage bootstrap

from intake import db, engine, live
from intake.models import AnswerRecord, AnswerStatus, CallState, CloseReason, CursorStep, ProfilePatch, RosterItem, SubStep, TurnCommand, TurnOutcome
from intake.pack import load_pack
from intake.turns import CursorAction, DecisionReason, Disposition, EvidenceSource
from tests.test_live import FakeRecorder, FakeSession, FakeWebSocket, pcm

SHISHU = load_pack("shishu-hi")
FEVER = load_pack("fever-opd-hi")


def _two_child_shishu(qid: str) -> CallState:
    st = engine.new_state(SHISHU)
    st.profile.update({"caller_name": "सुनीता", "marital_status": "married"})
    st.roster = [
        RosterItem(id="item_1", age_years=6.0, age_label="6 साल", tag="male", name="अमन"),
        RosterItem(id="item_2", age_years=2.0, age_label="2 साल", tag="female", name="मीरा"),
    ]
    st.roster_stated = True
    for done in ("intro_name", "intro_married", "intro_children", "intro_bridge"):
        st.answers[done] = AnswerRecord(state=AnswerStatus.CLEAR)
    st.cursor = CursorStep(qid=qid)
    return st


def _fever_at_frequency() -> CallState:
    st = engine.reduce_turn(
        FEVER,
        engine.new_state(FEVER),
        TurnCommand(profile_patch=ProfilePatch(slots={"patient_name": "रवि", "patient_age": 28})),
    )
    st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="days_3_to_7"))
    st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="intermittent"))
    return st


def _make_session(pack, state: CallState, call_id: int) -> tuple[live.LiveCallSession, FakeWebSocket]:
    ws = FakeWebSocket()
    sess = live.LiveCallSession(
        ws=ws,
        pack=pack,
        session=FakeSession(),
        call_id=call_id,
        state=state,
        recorder=FakeRecorder(),
        on_call_completed=lambda _cid: True,
    )
    return sess, ws


class TapSpeechConflictTests(unittest.TestCase):

    def setUp(self) -> None:
        with db._LOCK:
            for table in ("policy_decisions", "clinical_claims", "turn_interpretations", "turn_evidence", "calls"):
                db.conn.execute(f"DELETE FROM {table}")
            db.conn.commit()

    def tearDown(self) -> None:
        with db._LOCK:
            for table in ("policy_decisions", "clinical_claims", "turn_interpretations", "turn_evidence", "calls"):
                db.conn.execute(f"DELETE FROM {table}")
            db.conn.commit()

    def test_tap_yes_and_spoken_no_on_same_step_rolls_back_which_item_and_asks_confirmation(self):
        async def run() -> None:
            call_id = db.create_call(pack_id=SHISHU.id)
            sess, ws = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_id)
            gate = asyncio.Event()

            async def spoken_no(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.96},
                    "active_outcome": {"probabilities": {"clear": 0.95, "concern": 0.02, "other": 0.03}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 180, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess._evaluate_turn_task = spoken_no  # type: ignore[method-assign]
            await sess.on_audio({"data": pcm(), "sample_rate": 16000})
            await sess._hear("user", "नहीं, कोई गाँठ नहीं हुई थी")
            answering = asyncio.ensure_future(
                sess._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            # Accidental "हाँ" tap on the same step while spoken "नहीं" is being decided:
            await sess.on_quick_reply({"value": "concern", "step": "birth_bcg"})
            gate.set()
            await answering

            stored = db.call_state_from_row(db.get_call(call_id))
            self.assertNotIn("birth_bcg", stored.answers, "contradiction must leave birth_bcg unresolved")
            self.assertEqual((stored.cursor.qid, stored.cursor.substep), ("birth_bcg", SubStep.MAIN))
            self.assertEqual(ws.of("question_state")[-1]["qid"], "birth_bcg")
            self.assertEqual(
                [e.source for e in db.list_turn_evidence(call_id)],
                [EvidenceSource.TAP, EvidenceSource.VOICE],
            )
            self.assertEqual(
                [(d.disposition, d.reason, d.cursor_action) for d in db.list_policy_decisions(call_id)],
                [(Disposition.CONFLICT, DecisionReason.SOURCE_CONFLICT, CursorAction.STAY)],
            )

        asyncio.run(run())

    def test_choice_tap_vs_contradictory_or_unlisted_speech_leaves_choice_unresolved(self):
        for label, raw, spoken in (
            (
                "different_listed_option",
                {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.95},
                    "choice_value": {"probabilities": {"gt_7_days": 0.92, "lt_3_days": 0.04, "other": 0.04}},
                },
                "दस दिन से बुखार है",
            ),
            (
                "unlisted_fourth_option",
                {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.94},
                    "choice_value": {"probabilities": {"other": 0.93, "few_hours": 0.04, "alternate_days": 0.03}},
                    "active_outcome": {"probabilities": {"concern": 0.88, "other": 0.08}},
                },
                "रोज़ शाम को एक बार आता है",
            ),
        ):
            with self.subTest(case=label):
                async def run() -> None:
                    call_id = db.create_call(pack_id=FEVER.id)
                    st0 = _fever_at_frequency().model_copy(update={"cursor": CursorStep(qid="fever_duration_bucket")})
                    st0.answers.pop("fever_duration_bucket", None)
                    sess, _ws = _make_session(FEVER, st0, call_id)
                    gate = asyncio.Event()

                    async def eval_choice(snapshot: CallState, text: str, pcm_bytes: bytes) -> live.Turn:
                        await gate.wait()
                        res = live.decisions.DecisionCallResult(True, "", 1, 160, 120, 6, 40, raw)
                        intent = live.decisions.interpret_turn_decision(FEVER, snapshot, raw, utterance_text=text)
                        return live.Turn(res, intent, None, snapshot, text, len(pcm_bytes), None)

                    sess._evaluate_turn_task = eval_choice  # type: ignore[method-assign]
                    await sess.on_text({"text": spoken})
                    answering = asyncio.ensure_future(
                        sess._answer_tools([live.types.FunctionCall(id="c1", name="next_question", args={})])
                    )
                    await asyncio.sleep(0)
                    await sess.on_quick_reply({"value": "lt_3_days", "step": "fever_duration_bucket"})
                    gate.set()
                    await answering

                    stored = db.call_state_from_row(db.get_call(call_id))
                    self.assertNotIn("fever_duration_bucket", stored.answers)
                    self.assertEqual(stored.cursor.qid, "fever_duration_bucket")
                    self.assertEqual(
                        [d.disposition for d in db.list_policy_decisions(call_id)],
                        [Disposition.CONFLICT],
                    )

                asyncio.run(run())

    def test_agreeing_speech_with_aside_or_tv_noise_keeps_tap_advance_without_republishing(self):
        async def run() -> None:
            # 1. Tap "clear" + speech "नहीं, बस हल्का बुखार आया था" -> agrees on CLEAR AND retains fever observation!
            call_agree = db.create_call(pack_id=SHISHU.id)
            sess, ws = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_agree)
            gate = asyncio.Event()

            async def clear_plus_fever(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.95},
                    "residual_clinical": {"noul": 0.92},
                    "active_outcome": {"probabilities": {"clear": 0.93, "other": 0.05, "concern": 0.02}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 180, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess._evaluate_turn_task = clear_plus_fever  # type: ignore[method-assign]
            await sess.on_text({"text": "नहीं, बस हल्का बुखार आया था"})
            answering = asyncio.ensure_future(
                sess._answer_tools([live.types.FunctionCall(id="a1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess.on_quick_reply({"value": "clear", "step": "birth_bcg"})
            mark = len(ws.of("question_state"))
            sess.bot_text = "क्या जन्म से"
            gate.set()
            await answering

            stored = db.call_state_from_row(db.get_call(call_agree))
            self.assertIs(stored.answers["birth_bcg"].state, AnswerStatus.CLEAR)
            self.assertEqual(stored.cursor.qid, "birth_mode_detail")
            self.assertEqual(ws.of("question_state")[mark:], [], "agreement never republishes Question B")
            self.assertEqual(sess.bot_text, "क्या जन्म से")
            self.assertEqual(
                [d.disposition for d in db.list_policy_decisions(call_agree)],
                [Disposition.ACCEPT_FORM, Disposition.RETAIN_OBSERVATION],
            )

            # 2. Tap "clear" + background TV noise ("गैस बंद कर दो") -> tap stands, noise is IGNORE, no conflict!
            call_tv = db.create_call(pack_id=SHISHU.id)
            sess_tv, _ws_tv = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_tv)
            gate_tv = asyncio.Event()

            async def tv_noise(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate_tv.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.03},
                    "residual_clinical": {"noul": 0.02},
                    "active_outcome": {"probabilities": {"other": 0.94, "clear": 0.04}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 150, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess_tv._evaluate_turn_task = tv_noise  # type: ignore[method-assign]
            await sess_tv.on_text({"text": "गैस बंद कर दो"})
            answering_tv = asyncio.ensure_future(
                sess_tv._answer_tools([live.types.FunctionCall(id="tv1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess_tv.on_quick_reply({"value": "clear", "step": "birth_bcg"})
            gate_tv.set()
            await answering_tv

            stored_tv = db.call_state_from_row(db.get_call(call_tv))
            self.assertIs(stored_tv.answers["birth_bcg"].state, AnswerStatus.CLEAR)
            self.assertEqual(stored_tv.cursor.qid, "birth_mode_detail")
            self.assertEqual([d.disposition for d in db.list_policy_decisions(call_tv)], [Disposition.IGNORE])

        asyncio.run(run())

    def test_emergency_speech_overrides_same_step_tap_and_closes_call(self):
        async def run() -> None:
            call_id = db.create_call(pack_id=SHISHU.id)
            sess, _ws = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_id)
            gate = asyncio.Event()

            async def emergency_turn(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate.wait()
                raw = {"danger_now": {"noul": 0.98}, "active_outcome": {"probabilities": {"other": 0.95}}}
                res = live.decisions.DecisionCallResult(True, "", 1, 150, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess._evaluate_turn_task = emergency_turn  # type: ignore[method-assign]
            await sess.on_text({"text": "अभी बच्चे को साँस नहीं आ रही है!"})
            answering = asyncio.ensure_future(
                sess._answer_tools([live.types.FunctionCall(id="e1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess.on_quick_reply({"value": "clear", "step": "birth_bcg"})
            gate.set()
            await answering

            stored = db.call_state_from_row(db.get_call(call_id))
            self.assertIs(stored.close_reason, CloseReason.EMERGENCY)

        asyncio.run(run())

    def test_agreeing_concern_speech_enriches_bare_tap_with_child_note_and_followup(self):
        async def run() -> None:
            call_id = db.create_call(pack_id=SHISHU.id)
            sess, ws = _make_session(SHISHU, _two_child_shishu("birth_nicu"), call_id)
            gate = asyncio.Event()

            async def detailed_yes(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.97},
                    "detail_present": {"noul": 0.92},
                    "item_ref": {"probabilities": {"item_1": 0.96, "item_2": 0.02, "all": 0.01, "other": 0.01}},
                    "active_outcome": {"probabilities": {"concern": 0.96, "clear": 0.02, "other": 0.02}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 180, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess._evaluate_turn_task = detailed_yes  # type: ignore[method-assign]
            await sess.on_text({"text": "हाँ, अमन को साँस की तकलीफ के लिए रखा था"})
            answering = asyncio.ensure_future(
                sess._answer_tools([live.types.FunctionCall(id="n1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess.on_quick_reply({"value": "concern", "step": "birth_nicu"})
            gate.set()
            await answering

            stored = db.call_state_from_row(db.get_call(call_id))
            rec = stored.answers["birth_nicu"]
            self.assertEqual(rec.item_ids, ["item_1"])
            self.assertIn("अमन", rec.verbatim_text)
            self.assertEqual((stored.cursor.qid, stored.cursor.substep, stored.cursor.followup_idx), ("birth_nicu", SubStep.FOLLOWUP, 1))
            self.assertEqual(ws.of("question_state")[-1]["step"], stored.step_token)
            self.assertEqual(sess.cue_kind, "speak", "agreeing enrichment speaks the next followup normally, not clarify")

        asyncio.run(run())

    def test_trailing_stt_abort_and_mid_race_close_do_not_corrupt_conflict_handling(self):
        async def run() -> None:
            # 1. Trailing _hear chunk during in-flight tool_task + awaiting_abort_end from bot_generating:
            call_race = db.create_call(pack_id=SHISHU.id)
            sess, _ws = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_race)
            sess.bot_generating = True
            gate = asyncio.Event()

            async def spoken_no(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.96},
                    "active_outcome": {"probabilities": {"clear": 0.95, "concern": 0.02, "other": 0.03}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 160, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess._evaluate_turn_task = spoken_no  # type: ignore[method-assign]
            await sess.on_audio({"data": pcm(), "sample_rate": 16000})
            await sess._hear("user", "नहीं, कोई गाँठ नहीं")
            sess.tool_task = asyncio.ensure_future(
                sess._answer_tools([live.types.FunctionCall(id="r1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess.on_quick_reply({"value": "concern", "step": "birth_bcg"})
            await sess._hear("user", "नहीं वाकई")
            gate.set()
            await sess.tool_task

            stored = db.call_state_from_row(db.get_call(call_race))
            self.assertNotIn("birth_bcg", stored.answers, "trailing STT chunk must not erase last_tap")
            self.assertEqual([d.disposition for d in db.list_policy_decisions(call_race)], [Disposition.CONFLICT])
            self.assertIsNone(sess.fallback, "no fallback timer may run while awaiting_abort_end is set")
            await sess.on_turn_complete()
            self.assertIsNotNone(sess.fallback, "abort completion must arm the clarify rescue cue")
            self.assertFalse(sess.fallback.done())
            sess._cancel("fallback")

            # 2. Call closed for TIME_LIMIT during race must stay closed, never reopen to ASKING:
            call_closed = db.create_call(pack_id=SHISHU.id)
            sess_c, _ws_c = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_closed)
            gate_c = asyncio.Event()

            async def slow_no(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate_c.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.96},
                    "active_outcome": {"probabilities": {"clear": 0.95, "concern": 0.02, "other": 0.03}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 160, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess_c._evaluate_turn_task = slow_no  # type: ignore[method-assign]
            await sess_c.on_text({"text": "नहीं"})
            answering_c = asyncio.ensure_future(
                sess_c._answer_tools([live.types.FunctionCall(id="cl1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess_c.on_quick_reply({"value": "concern", "step": "birth_bcg"})
            await sess_c._close(CloseReason.TIME_LIMIT)
            gate_c.set()
            await answering_c

            stored_c = db.call_state_from_row(db.get_call(call_closed))
            self.assertIs(stored_c.close_reason, CloseReason.TIME_LIMIT)
            self.assertIs(stored_c.phase, live.Phase.CLOSING)

            # 3. turn_complete arriving DURING tool_task clears awaiting_abort_end so conflict rollback arms fallback:
            call_tc = db.create_call(pack_id=SHISHU.id)
            sess_tc, _ws_tc = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_tc)
            sess_tc.bot_generating = True
            gate_tc = asyncio.Event()

            async def slow_conflict(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate_tc.wait()
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.96},
                    "active_outcome": {"probabilities": {"clear": 0.95, "concern": 0.02, "other": 0.03}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 160, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess_tc._evaluate_turn_task = slow_conflict  # type: ignore[method-assign]
            await sess_tc.on_audio({"data": pcm(), "sample_rate": 16000})
            await sess_tc._hear("user", "नहीं")
            sess_tc.tool_task = asyncio.ensure_future(
                sess_tc._answer_tools([live.types.FunctionCall(id="tc1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess_tc.on_quick_reply({"value": "concern", "step": "birth_bcg"})
            await sess_tc.on_turn_complete()  # Turn A dies BEFORE tool_task finishes!
            gate_tc.set()
            await sess_tc.tool_task
            self.assertFalse(sess_tc.awaiting_abort_end)
            self.assertIsNotNone(sess_tc.fallback, "conflict rollback must arm fallback once Turn A already ended")
            sess_tc._cancel("fallback")

            # 4. Emergency speech arriving during TIME_LIMIT close upgrades close_reason and keeps emergency_note:
            call_emg = db.create_call(pack_id=SHISHU.id)
            sess_e, _ws_e = _make_session(SHISHU, _two_child_shishu("birth_bcg"), call_emg)
            gate_e = asyncio.Event()

            async def slow_emg(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
                await gate_e.wait()
                raw = {"danger_now": {"noul": 0.98}, "active_outcome": {"probabilities": {"other": 0.95}}}
                res = live.decisions.DecisionCallResult(True, "", 1, 160, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess_e._evaluate_turn_task = slow_emg  # type: ignore[method-assign]
            await sess_e.on_text({"text": "अभी साँस नहीं आ रही है!"})
            answering_e = asyncio.ensure_future(
                sess_e._answer_tools([live.types.FunctionCall(id="em1", name="next_question", args={})])
            )
            await asyncio.sleep(0)
            await sess_e.on_quick_reply({"value": "concern", "step": "birth_bcg"})
            await sess_e._close(CloseReason.TIME_LIMIT)
            gate_e.set()
            await answering_e

            stored_e = db.call_state_from_row(db.get_call(call_emg))
            self.assertIs(stored_e.close_reason, CloseReason.EMERGENCY)
            self.assertIn("साँस", stored_e.emergency_note)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
