"""RED-first contract tests for Stage 2A: pure semantic policy and append-only claim ledger."""

from __future__ import annotations

import sqlite3
import unittest
from html import unescape

import tests  # noqa: F401 — isolated storage bootstrap before importing intake

from intake import db, engine, reports
from intake.models import AnswerRecord, AnswerStatus, CursorStep, RosterItem, TurnOutcome
from intake.pack import load_pack
from intake.policy import apply_policy
from intake.turns import (
    ClinicalClaim,
    CursorAction,
    DecisionReason,
    Disposition,
    DomainFit,
    EvidenceSource,
    SpeechAct,
    TopicRelation,
    TurnEvidence,
    TurnInterpretation,
)

SHISHU = load_pack("shishu-hi")
FEVER = load_pack("fever-opd-hi")


def _shishu_at(qid: str):
    st = engine.new_state(SHISHU)
    st.profile.update({"caller_name": "सुनीता", "marital_status": "married"})
    st.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male", name="अमन")]
    st.roster_stated = True
    for done in ("intro_name", "intro_married", "intro_children", "intro_bridge"):
        st.answers[done] = AnswerRecord(state=AnswerStatus.CLEAR)
    st.cursor = CursorStep(qid=qid)
    return st


def _fever_at_frequency():
    from intake.models import ProfilePatch, TurnCommand

    st = engine.reduce_turn(
        FEVER,
        engine.new_state(FEVER),
        TurnCommand(profile_patch=ProfilePatch(slots={"patient_name": "रवि", "patient_age": 28})),
    )
    st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="days_3_to_7"))
    st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="intermittent"))
    assert st.cursor.qid == "fever_frequency", f"expected fever_frequency, landed on {st.cursor.qid}"
    return st


def _voice(step: str, text: str) -> TurnEvidence:
    return TurnEvidence(
        source=EvidenceSource.VOICE,
        snapshot_step=step,
        transcript=text,
        received_at=1_795_000_100.0,
    )


class SemanticPolicyTests(unittest.TestCase):

    def setUp(self) -> None:
        with db._LOCK:
            for table in ("policy_decisions", "clinical_claims", "turn_interpretations", "turn_evidence", "calls"):
                db.conn.execute(f"DELETE FROM {table}")
            db.conn.commit()

    def tearDown(self) -> None:
        with db._LOCK:
            db.conn.execute("DROP TRIGGER IF EXISTS reject_policy_decisions")
            for table in ("policy_decisions", "clinical_claims", "turn_interpretations", "turn_evidence", "calls"):
                db.conn.execute(f"DELETE FROM {table}")
            db.conn.commit()

    def test_clinical_aside_is_retained_without_answering_any_gate_while_noise_is_ignored(self):
        for qid, text in (("birth_bcg", "बुखार आया था"), ("birth_hearing", "खांसी थी")):
            with self.subTest(qid=qid):
                st = _shishu_at(qid)
                interp = TurnInterpretation(
                    act=SpeechAct.ANSWER,
                    active_relation=TopicRelation.CLINICAL_ASIDE,
                    active_domain=DomainFit.NONE,
                    claims=(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE),),
                )
                nxt, out = apply_policy(SHISHU, st, _voice(st.step_token, text), interp)
                self.assertNotIn(qid, nxt.answers)
                self.assertEqual((nxt.cursor.qid, nxt.cursor.attempts), (qid, 1))
                self.assertEqual(
                    [(d.disposition, d.reason, d.cursor_action) for d in out.decisions],
                    [(Disposition.RETAIN_OBSERVATION, DecisionReason.RESIDUAL_CLINICAL, CursorAction.STAY)],
                )

        st_noise = _shishu_at("birth_bcg")
        noise = TurnInterpretation(
            act=SpeechAct.NONE,
            active_relation=TopicRelation.UNRELATED,
            active_domain=DomainFit.NONE,
        )
        nxt_noise, out_noise = apply_policy(
            SHISHU, st_noise, _voice(st_noise.step_token, "गैस बंद कर दो, दूध उबल रहा है"), noise
        )
        self.assertNotIn("birth_bcg", nxt_noise.answers)
        self.assertEqual((nxt_noise.cursor.qid, nxt_noise.cursor.attempts), ("birth_bcg", 1))
        self.assertEqual(
            [(d.disposition, d.reason, d.cursor_action) for d in out_noise.decisions],
            [(Disposition.IGNORE, DecisionReason.UNRELATED, CursorAction.STAY)],
        )

    def test_co_reported_clear_plus_aside_clears_gate_and_retains_observation(self):
        st = _shishu_at("birth_bcg")
        text = "नहीं, कोई फोड़ा या गाँठ नहीं हुई थी, बस हल्का बुखार आया था"
        interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.IN_DOMAIN,
            claims=(
                ClinicalClaim(
                    target_qid="birth_bcg",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CLEAR,
                ),
                ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE),
            ),
        )
        nxt, out = apply_policy(SHISHU, st, _voice(st.step_token, text), interp)
        self.assertIs(nxt.answers["birth_bcg"].state, AnswerStatus.CLEAR)
        self.assertNotEqual(nxt.cursor.qid, "birth_bcg")
        self.assertEqual(
            [(d.disposition, d.reason) for d in out.decisions],
            [
                (Disposition.ACCEPT_FORM, DecisionReason.IN_DOMAIN),
                (Disposition.RETAIN_OBSERVATION, DecisionReason.RESIDUAL_CLINICAL),
            ],
        )

    def test_unlisted_on_topic_choice_records_caller_words_without_fabricating_an_option(self):
        for qid, setup, text in (
            ("fever_frequency", _fever_at_frequency, "रोज़ शाम को एक बार आता है"),
            (
                "fever_duration_bucket",
                lambda: _fever_at_frequency().model_copy(
                    update={"cursor": CursorStep(qid="fever_duration_bucket"), "answers": {"intro_bridge": AnswerRecord(state=AnswerStatus.CLEAR)}}
                ),
                "महीने भर से कभी-कभी हल्का रहता है",
            ),
        ):
            with self.subTest(qid=qid):
                st = setup()
                interp = TurnInterpretation(
                    act=SpeechAct.ANSWER,
                    active_relation=TopicRelation.ACTIVE,
                    active_domain=DomainFit.UNLISTED,
                    claims=(
                        ClinicalClaim(
                            target_qid=qid,
                            relation=TopicRelation.ACTIVE,
                            domain_fit=DomainFit.UNLISTED,
                        ),
                    ),
                )
                nxt, out = apply_policy(FEVER, st, _voice(st.step_token, text), interp)
                rec = nxt.answers[qid]
                self.assertIs(rec.state, AnswerStatus.ANSWERED)
                self.assertIsNone(rec.value)
                self.assertEqual(rec.choice_label, "")
                self.assertEqual(rec.patient_safe_text, text)
                self.assertNotEqual(nxt.cursor.qid, qid)
                self.assertEqual(nxt.cursor.attempts, 0)
                self.assertEqual(out.decisions[0].disposition, Disposition.ACCEPT_UNLISTED)
                proj = reports.project_summary(FEVER, nxt)
                # Her unlisted wording reaches the report's answered block — no fabricated option stands in for it.
                self.assertIn(text, proj.report_markdown)
                self.assertIn(text, str(proj.page2_groups))

        # Redacted unlisted answer: her words never reach a rendered surface, only her own record keeps them.
        st_red = _fever_at_frequency()
        secret = "डॉक्टर ने मलेरिया कन्फर्म बताया, हर शाम आता है"
        red_interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.UNLISTED,
            redacted=True,
            claims=(
                ClinicalClaim(
                    target_qid="fever_frequency",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.UNLISTED,
                ),
            ),
        )
        nxt_red, _ = apply_policy(FEVER, st_red, _voice(st_red.step_token, secret), red_interp)
        proj_red = reports.project_summary(FEVER, nxt_red)
        html_red = reports.render_patient_html(FEVER, {"id": 9, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": 120}, proj_red)
        # The clinician's own record of the turn keeps her exact words; no report surface ever prints them.
        self.assertEqual(nxt_red.answers["fever_frequency"].verbatim_text, secret)
        self.assertEqual(nxt_red.answers["fever_frequency"].patient_safe_text, "")
        self.assertNotIn(secret, proj_red.report_markdown)
        self.assertNotIn(secret, str(proj_red.page2_groups))
        self.assertNotIn(secret, html_red)

        # Multi-topic unlisted + later OTHER_FORM claim must ALSO redact patient_safe even when active claim is index 0.
        multi_txt = "हर शाम आता है, और उल्टी भी हो रही है"
        multi_interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.UNLISTED,
            claims=(
                ClinicalClaim(target_qid="fever_frequency", relation=TopicRelation.ACTIVE, domain_fit=DomainFit.UNLISTED),
                ClinicalClaim(
                    target_qid="vomiting_or_diarrhea",
                    relation=TopicRelation.OTHER_FORM,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CONCERN,
                ),
            ),
        )
        nxt_multi, _ = apply_policy(FEVER, _fever_at_frequency(), _voice("fever_frequency", multi_txt), multi_interp)
        proj_multi = reports.project_summary(FEVER, nxt_multi)
        # Redacted multi-topic: her words are still the record on the reported concern, but never on the PDF's
        # structured rows — the same line the single-topic case above pins.
        self.assertIn(multi_txt, proj_multi.report_markdown)
        self.assertNotIn(multi_txt, str(proj_multi.page2_groups))

        # Control 1: an off-topic symptom on the same choice question stays an aside and re-asks.
        st_aside = _fever_at_frequency()
        aside_interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.CLINICAL_ASIDE,
            active_domain=DomainFit.NONE,
            claims=(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE),),
        )
        nxt_aside, out_aside = apply_policy(
            FEVER, st_aside, _voice(st_aside.step_token, "खांसी भी आती है"), aside_interp
        )
        self.assertNotIn("fever_frequency", nxt_aside.answers)
        self.assertEqual((nxt_aside.cursor.qid, nxt_aside.cursor.attempts), ("fever_frequency", 1))
        self.assertEqual(out_aside.decisions[0].disposition, Disposition.RETAIN_OBSERVATION)

        # Control 2: a profile-routing choice (saves_to_profile) never accepts value=None as resolved.
        st_married = engine.new_state(SHISHU)
        st_married.profile["caller_name"] = "सुनीता"
        st_married.cursor = CursorStep(qid="intro_married")
        unlisted_married = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.UNLISTED,
            claims=(
                ClinicalClaim(
                    target_qid="intro_married",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.UNLISTED,
                ),
            ),
        )
        nxt_married, out_married = apply_policy(
            SHISHU, st_married, _voice(st_married.step_token, "अलग रहते हैं"), unlisted_married
        )
        self.assertEqual((nxt_married.cursor.qid, nxt_married.cursor.attempts), ("intro_married", 1))
        self.assertEqual(len(out_married.decisions), 1)
        self.assertIs(out_married.decisions[0].disposition, Disposition.IGNORE)

    def test_co_reported_answer_plus_named_clarify_preserves_both_actions(self):
        st = _shishu_at("body_transfusion")
        st.last_answered_qid = "preg_thalassemia"
        st.answers["preg_thalassemia"] = AnswerRecord(state=AnswerStatus.UNKNOWN)
        interp = TurnInterpretation(
            act=SpeechAct.CLARIFY,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.IN_DOMAIN,
            clarify_target_qid="preg_thalassemia",
            claims=(
                ClinicalClaim(
                    target_qid="body_transfusion",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CLEAR,
                ),
            ),
        )
        nxt, out = apply_policy(
            SHISHU,
            st,
            _voice(st.step_token, "नहीं, खून नहीं चढ़ाया। थैलेसीमिया का मतलब क्या है?"),
            interp,
        )
        self.assertIs(nxt.answers["body_transfusion"].state, AnswerStatus.CLEAR)
        self.assertEqual(nxt.cursor.qid, "preg_thalassemia")
        self.assertEqual(
            [(d.disposition, d.reason) for d in out.decisions],
            [
                (Disposition.ACCEPT_FORM, DecisionReason.IN_DOMAIN),
                (Disposition.CLARIFY, DecisionReason.NAMED_CLARIFY),
            ],
        )

    def test_other_form_claim_and_unasked_clarify_never_hijack_the_active_question(self):
        st = _shishu_at("birth_bcg")
        other_interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.OTHER_FORM,
            active_domain=DomainFit.NONE,
            claims=(
                ClinicalClaim(
                    target_qid="fam_cancer",
                    relation=TopicRelation.OTHER_FORM,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CONCERN,
                ),
            ),
        )
        nxt, out = apply_policy(
            SHISHU, st, _voice(st.step_token, "हाँ, बच्चे की बुआ को कैंसर हुआ था"), other_interp
        )
        self.assertIs(nxt.answers["fam_cancer"].state, AnswerStatus.CONCERN)
        self.assertEqual(nxt.answers["fam_cancer"].patient_safe_text, "")
        self.assertNotIn("birth_bcg", nxt.answers)
        self.assertEqual((nxt.cursor.qid, nxt.cursor.attempts), ("birth_bcg", 0))
        self.assertEqual(out.decisions[0].disposition, Disposition.ACCEPT_FORM)

        # Asking about an unasked future term ("थैलेसीमिया") while on `birth_bcg` must clarify in place, never jump.
        clarify_unasked = TurnInterpretation(
            act=SpeechAct.CLARIFY,
            active_relation=TopicRelation.OTHER_FORM,
            active_domain=DomainFit.NONE,
            clarify_target_qid="preg_thalassemia",
        )
        nxt_clarify, out_clarify = apply_policy(
            SHISHU, st, _voice(st.step_token, "थैलेसीमिया का मतलब क्या होता है?"), clarify_unasked
        )
        self.assertEqual((nxt_clarify.cursor.qid, nxt_clarify.cursor.attempts), ("birth_bcg", 0))
        self.assertEqual(
            [(d.disposition, d.reason, d.cursor_action) for d in out_clarify.decisions],
            [(Disposition.CLARIFY, DecisionReason.ACTIVE_CLARIFY, CursorAction.STAY)],
        )

    def test_retraction_updates_projection_while_append_only_ledger_keeps_prior_concern_atomically(self):
        call_id = db.create_call(pack_id=SHISHU.id)
        st0 = _shishu_at("birth_bcg")
        ev1 = _voice(st0.step_token, "हाँ, गाँठ हुई थी")
        concern_interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.IN_DOMAIN,
            claims=(
                ClinicalClaim(
                    target_qid="birth_bcg",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CONCERN,
                ),
            ),
        )
        st1, out1 = apply_policy(SHISHU, st0, ev1, concern_interp)
        db.commit_semantic_turn(call_id, SHISHU, st1, ev1, concern_interp, out1.decisions)
        self.assertIs(db.call_state_from_row(db.get_call(call_id)).answers["birth_bcg"].state, AnswerStatus.CONCERN)

        ev2 = _voice(st1.step_token, "नहीं नहीं, गलती से हाँ बोल दिया, कोई गाँठ नहीं हुई थी")
        retract_interp = TurnInterpretation(
            act=SpeechAct.RETRACT,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.IN_DOMAIN,
            claims=(
                ClinicalClaim(
                    target_qid="birth_bcg",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.RETRACT,
                ),
            ),
        )
        st2, out2 = apply_policy(SHISHU, st0, ev2, retract_interp)
        db.commit_semantic_turn(call_id, SHISHU, st2, ev2, retract_interp, out2.decisions)

        self.assertIs(db.call_state_from_row(db.get_call(call_id)).answers["birth_bcg"].state, AnswerStatus.CLEAR)
        claims = db.list_clinical_claims(call_id)
        decisions_rows = db.list_policy_decisions(call_id)
        self.assertEqual([c.status for c in claims], [TurnOutcome.CONCERN, TurnOutcome.RETRACT])
        self.assertEqual([d.reason for d in decisions_rows], [DecisionReason.IN_DOMAIN, DecisionReason.RETRACT])

        # Atomicity check: if policy_decisions insert aborts, neither cursor_json nor evidence/claims advance.
        with db._LOCK:
            db.conn.execute(
                """
                CREATE TRIGGER reject_policy_decisions
                BEFORE INSERT ON policy_decisions
                BEGIN
                    SELECT RAISE(ABORT, 'forced policy decision failure');
                END
                """
            )
            db.conn.commit()
        ev3 = _voice(st2.step_token, "बुखार आया था")
        aside_interp = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.CLINICAL_ASIDE,
            active_domain=DomainFit.NONE,
            claims=(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE),),
        )
        st3, out3 = apply_policy(SHISHU, st2, ev3, aside_interp)
        with self.assertRaises(sqlite3.IntegrityError):
            db.commit_semantic_turn(call_id, SHISHU, st3, ev3, aside_interp, out3.decisions)
        self.assertEqual(len(db.list_turn_evidence(call_id)), 2)
        self.assertEqual(len(db.list_clinical_claims(call_id)), 2)
        self.assertEqual(db.call_state_from_row(db.get_call(call_id)).to_dict(), st2.to_dict())


class SemanticReaderAndLiveSeamTests(unittest.TestCase):

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

    def test_request_strips_clarify_from_stated_heads_and_scopes_clarify_target_to_eligible_priors(self):
        from intake import decisions

        for pack, st in ((SHISHU, _shishu_at("birth_bcg")), (FEVER, _fever_at_frequency())):
            with self.subTest(pack=pack.id):
                qs = decisions.build_turn_request(pack, st, utterance_text="पता नहीं")["questions"]
                for head in ("active_topic", "residual_clinical", "asks_clarify"):
                    self.assertIn(head, qs)
                self.assertNotIn("clarify_target", qs, "no eligible prior question means zero extra choice head")
                stated = [v for k, v in qs.items() if k.startswith("stated_")]
                self.assertTrue(stated)
                for item in stated:
                    self.assertEqual(set(item["criteria"]), {"concern", "clear", "unmentioned"})

        with_prior = _shishu_at("fam_related")
        with_prior.last_answered_qid = "preg_thalassemia"
        with_prior.answers["preg_thalassemia"] = AnswerRecord(state=AnswerStatus.UNKNOWN)
        qs_prior = decisions.build_turn_request(SHISHU, with_prior, utterance_text="मतलब?")["questions"]
        self.assertEqual(set(qs_prior["clarify_target"]["criteria"]), {"preg_thalassemia", "active_or_none"})

        # A closed choice never accepts bare status=CONCERN/CLEAR with value=None as in-domain.
        st_freq = _fever_at_frequency()
        bad_choice = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.IN_DOMAIN,
            claims=(
                ClinicalClaim(
                    target_qid="fever_frequency",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CONCERN,
                    value=None,
                ),
            ),
        )
        nxt_bad, out_bad = apply_policy(FEVER, st_freq, _voice(st_freq.step_token, "हाँ"), bad_choice)
        self.assertNotIn("fever_frequency", nxt_bad.answers)
        self.assertIs(out_bad.decisions[0].disposition, Disposition.IGNORE)

    def test_reader_maps_unlisted_choice_aside_and_co_report_into_turn_interpretation(self):
        from intake import decisions

        st_freq = _fever_at_frequency()
        intent_unlisted = decisions.interpret_turn_decision(
            FEVER,
            st_freq,
            {
                "active_topic": {"noul": 0.93},
                "choice_value": {"probabilities": {"other": 0.93, "few_hours": 0.05, "alternate_days": 0.02}},
                "active_outcome": {"probabilities": {"concern": 0.88, "other": 0.08, "clear": 0.04}},
            },
            utterance_text="रोज़ शाम को एक बार आता है",
        )
        self.assertIsNotNone(intent_unlisted.interpretation)
        self.assertIs(intent_unlisted.interpretation.active_relation, TopicRelation.ACTIVE)
        self.assertIs(intent_unlisted.interpretation.active_domain, DomainFit.UNLISTED)

        st_bcg = _shishu_at("birth_bcg")
        intent_aside = decisions.interpret_turn_decision(
            SHISHU,
            st_bcg,
            {
                "active_topic": {"noul": 0.08},
                "residual_clinical": {"noul": 0.94},
                "active_outcome": {"probabilities": {"other": 0.91, "concern": 0.05, "clear": 0.04}},
            },
            utterance_text="बुखार आया था",
        )
        self.assertIs(intent_aside.interpretation.active_relation, TopicRelation.CLINICAL_ASIDE)
        self.assertEqual([c.relation for c in intent_aside.interpretation.claims], [TopicRelation.CLINICAL_ASIDE])

        intent_co = decisions.interpret_turn_decision(
            SHISHU,
            st_bcg,
            {
                "active_topic": {"noul": 0.95},
                "residual_clinical": {"noul": 0.91},
                "active_outcome": {"probabilities": {"clear": 0.92, "other": 0.05, "concern": 0.03}},
            },
            utterance_text="नहीं, कोई फोड़ा या गाँठ नहीं हुई थी, बस हल्का बुखार आया था",
        )
        self.assertEqual(
            [c.relation for c in intent_co.interpretation.claims],
            [TopicRelation.ACTIVE, TopicRelation.CLINICAL_ASIDE],
        )

        # Gate safety: even if active_outcome leans concern, active_topic=False blocks a false gate CONCERN.
        intent_false_concern = decisions.interpret_turn_decision(
            SHISHU,
            st_bcg,
            {
                "active_topic": {"noul": 0.08},
                "residual_clinical": {"noul": 0.93},
                "active_outcome": {"probabilities": {"concern": 0.89, "other": 0.08, "clear": 0.03}},
            },
            utterance_text="बुखार आया था",
        )
        self.assertIs(intent_false_concern.interpretation.active_relation, TopicRelation.CLINICAL_ASIDE)
        self.assertEqual([c.relation for c in intent_false_concern.interpretation.claims], [TopicRelation.CLINICAL_ASIDE])

        # Retract + co-reported other-form concern preserves both in one turn.
        retract_plus_other = TurnInterpretation(
            act=SpeechAct.RETRACT,
            active_relation=TopicRelation.ACTIVE,
            active_domain=DomainFit.IN_DOMAIN,
            claims=(
                ClinicalClaim(
                    target_qid="birth_bcg",
                    relation=TopicRelation.ACTIVE,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.RETRACT,
                ),
                ClinicalClaim(
                    target_qid="fam_cancer",
                    relation=TopicRelation.OTHER_FORM,
                    domain_fit=DomainFit.IN_DOMAIN,
                    status=TurnOutcome.CONCERN,
                ),
            ),
        )
        nxt_ro, out_ro = apply_policy(
            SHISHU, st_bcg, _voice(st_bcg.step_token, "नहीं गाँठ नहीं हुई थी, पर बुआ को कैंसर था"), retract_plus_other
        )
        self.assertIs(nxt_ro.answers["birth_bcg"].state, AnswerStatus.CLEAR)
        self.assertIs(nxt_ro.answers["fam_cancer"].state, AnswerStatus.CONCERN)
        self.assertEqual(len(out_ro.decisions), 2)

    def test_live_session_commits_semantic_ledger_and_accepts_unlisted_choice(self):
        import asyncio
        from intake import live
        from tests.test_live import FakeRecorder, FakeSession, FakeWebSocket

        async def run() -> None:
            call_freq = db.create_call(pack_id=FEVER.id)
            st_freq = _fever_at_frequency()
            sess_freq = live.LiveCallSession(
                ws=FakeWebSocket(),
                pack=FEVER,
                session=FakeSession(),
                call_id=call_freq,
                state=st_freq,
                recorder=FakeRecorder(),
                on_call_completed=lambda _cid: True,
            )

            async def eval_unlisted(snapshot, spoken, pcm_bytes):
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.94},
                    "choice_value": {"probabilities": {"other": 0.93, "few_hours": 0.05}},
                    "active_outcome": {"probabilities": {"concern": 0.88, "other": 0.08}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 180, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(FEVER, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess_freq._evaluate_turn_task = eval_unlisted  # type: ignore[method-assign]
            await sess_freq.on_text({"text": "रोज़ शाम को एक बार आता है"})
            await sess_freq._next_question()

            stored_freq = db.call_state_from_row(db.get_call(call_freq))
            self.assertEqual(stored_freq.answers["fever_frequency"].choice_label, "")
            self.assertEqual(stored_freq.answers["fever_frequency"].patient_safe_text, "रोज़ शाम को एक बार आता है")
            self.assertIsNone(stored_freq.answers["fever_frequency"].value)
            self.assertNotEqual(stored_freq.cursor.qid, "fever_frequency")
            self.assertEqual(
                [d.disposition for d in db.list_policy_decisions(call_freq)],
                [Disposition.ACCEPT_UNLISTED],
            )

            call_bcg = db.create_call(pack_id=SHISHU.id)
            st_bcg = _shishu_at("birth_bcg")
            sess_bcg = live.LiveCallSession(
                ws=FakeWebSocket(),
                pack=SHISHU,
                session=FakeSession(),
                call_id=call_bcg,
                state=st_bcg,
                recorder=FakeRecorder(),
                on_call_completed=lambda _cid: True,
            )

            async def eval_aside(snapshot, spoken, pcm_bytes):
                raw = {
                    "danger_now": {"noul": 0.01},
                    "active_topic": {"noul": 0.06},
                    "residual_clinical": {"noul": 0.92},
                    "active_outcome": {"probabilities": {"other": 0.91, "concern": 0.05}},
                }
                res = live.decisions.DecisionCallResult(True, "", 1, 180, 120, 6, 40, raw)
                intent = live.decisions.interpret_turn_decision(SHISHU, snapshot, raw, utterance_text=spoken)
                return live.Turn(res, intent, None, snapshot, spoken, len(pcm_bytes), None)

            sess_bcg._evaluate_turn_task = eval_aside  # type: ignore[method-assign]
            await sess_bcg.on_text({"text": "बुखार आया था"})
            await sess_bcg._next_question()

            stored_bcg = db.call_state_from_row(db.get_call(call_bcg))
            self.assertNotIn("birth_bcg", stored_bcg.answers)
            self.assertEqual((stored_bcg.cursor.qid, stored_bcg.cursor.attempts), ("birth_bcg", 1))
            self.assertEqual(
                [d.disposition for d in db.list_policy_decisions(call_bcg)],
                [Disposition.RETAIN_OBSERVATION],
            )

        asyncio.run(run())

    def test_retained_observations_reach_the_unified_report_and_the_pdf_as_attributed_asides(self):
        """A retained clinical aside is quoted the same way on every surface now: the pack's own label, her words.

        The old split kept asides on a clinician-only note and off her sheet. One unified report means the aside
        is on both — but still attributed, still never a red flag, and unrelated household noise is still dropped.
        """
        call_id = db.create_call(pack_id=SHISHU.id)
        st0 = _shishu_at("birth_bcg")
        aside_txt = "टीके के बाद दो दिन हल्का बुखार आया था"
        ev1 = _voice(st0.step_token, aside_txt)
        interp1 = TurnInterpretation(
            act=SpeechAct.ANSWER,
            active_relation=TopicRelation.CLINICAL_ASIDE,
            active_domain=DomainFit.NONE,
            claims=(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE),),
        )
        st1, out1 = apply_policy(SHISHU, st0, ev1, interp1)
        db.commit_semantic_turn(call_id, SHISHU, st1, ev1, interp1, out1.decisions)

        # Unrelated household noise on the next turn must NOT appear in observations:
        noise_txt = "गैस बंद कर दो, दूध उबल रहा है"
        ev2 = _voice(st1.step_token, noise_txt)
        interp2 = TurnInterpretation(
            act=SpeechAct.NONE,
            active_relation=TopicRelation.UNRELATED,
            active_domain=DomainFit.NONE,
        )
        st2, out2 = apply_policy(SHISHU, st1, ev2, interp2)
        db.commit_semantic_turn(call_id, SHISHU, st2, ev2, interp2, out2.decisions)
        db.complete_call(call_id, 180)

        pdf_bytes, _name = reports.build_patient_pdf_sync(call_id)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        saved = db.get_call(call_id)
        report = saved["summary_report"]
        html = unescape(reports.render_patient_html(SHISHU, saved))
        label = str(SHISHU.report_cfg["caller_words_hi"])
        heading = str(SHISHU.report_cfg["notes_header_hi"])

        self.assertIn(f"### {heading}", report)  # the aside has its own block, not a diagnostic one
        self.assertIn(f"{label}: “{aside_txt}”", report)
        self.assertIn(f"{label}: “{aside_txt}”", html)
        self.assertNotIn(noise_txt, report)
        self.assertNotIn(noise_txt, html)
        # An unconfirmed aside never becomes a clinical finding or a red flag: it stays out of the reported-yes block.
        self.assertNotIn(aside_txt, report.split('### Reported "Yes"', 1)[1].split("\n### ", 1)[0])
        for gone in ("Red Flags", "Gate Alerts", "Chief complaint", "Additional Clinical Observations"):
            self.assertNotIn(gone, report)

        # Audio-only RETAIN_OBSERVATION (empty transcript at commit) + trailing LATE_STT must still reach the report!
        call_late = db.create_call(pack_id=SHISHU.id)
        ev_audio = _voice(st0.step_token, "")
        st_a, out_a = apply_policy(SHISHU, st0, ev_audio, interp1)
        db.commit_semantic_turn(call_late, SHISHU, st_a, ev_audio, interp1, out_a.decisions)
        db.commit_turn(
            call_late,
            SHISHU,
            st_a,
            TurnEvidence(source=EvidenceSource.LATE_STT, snapshot_step=st0.step_token, transcript="टीके के बाद उल्टी हुई थी", received_at=1_795_000_101.0),
        )
        db.complete_call(call_late, 90)
        reports.build_patient_pdf_sync(call_late)
        self.assertIn(f"{label}: “टीके के बाद उल्टी हुई थी”", db.get_call(call_late)["summary_report"])


if __name__ == "__main__":
    unittest.main()
