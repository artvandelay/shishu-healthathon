"""Model-based invariants for apply_policy and arbitrate_tap_turn."""

from __future__ import annotations

import unittest

import tests  # noqa: F401 — isolated storage before intake

from intake import db, engine, reports
from intake.models import AnswerRecord, AnswerStatus, CloseReason, CursorStep, Phase, QuestionKind, RosterItem, SubStep, TurnCommand, TurnOutcome
from intake.pack import load_pack
from intake.policy import apply_policy, arbitrate_tap_turn
from intake.turns import (
    ClinicalClaim, CursorAction, DecisionReason, Disposition, DomainFit, EvidenceSource, SpeechAct, TopicRelation, TurnEvidence, TurnInterpretation,
)

PACKS = {pid: load_pack(pid) for pid in ("shishu-hi", "fever-opd-hi")}
_BASE = {pid: engine.new_state(pack) for pid, pack in PACKS.items()}
UTT, _RESOLVED = "INVTOKEN9f3a-कच्चा", (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.ANSWERED)
_ACCEPT = (Disposition.ACCEPT_FORM, Disposition.ACCEPT_UNLISTED)
_CHOICE = (QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE, QuestionKind.SCORE)
_YES = (QuestionKind.GATE, QuestionKind.ROUTING_GATE, QuestionKind.BRIDGE, QuestionKind.OPEN_CAPTURE)
_IDS = {
    "shishu-hi": ("intro_name", "intro_married", "intro_bridge", "fam_infant_death", "birth_mode_detail", "ever_pregnant", "close_anything_else"),
    "fever-opd-hi": ("patient_name", "intro_bridge", "fever_duration_bucket", "fever_severity", "associated_symptoms", "breathing_or_chest", "close_anything_else"),
}
# act, relation, domain, co-reported other form, residual aside, redacted, jump_topic, contradictory tap
_PATTERNS = (
    (SpeechAct.ANSWER, TopicRelation.ACTIVE, DomainFit.IN_DOMAIN, 0, 0, 0, 0, 1),
    (SpeechAct.ANSWER, TopicRelation.ACTIVE, DomainFit.IN_DOMAIN, 1, 1, 0, 1, 0),
    (SpeechAct.ANSWER, TopicRelation.ACTIVE, DomainFit.UNLISTED, 0, 0, 1, 0, 1),
    (SpeechAct.ANSWER, TopicRelation.CLINICAL_ASIDE, DomainFit.NONE, 0, 0, 0, 0, 0),
    (SpeechAct.NONE, TopicRelation.UNRELATED, DomainFit.NONE, 1, 0, 0, 0, 0),
    (SpeechAct.CLARIFY, TopicRelation.ACTIVE, DomainFit.NONE, 0, 1, 0, 0, 0),
    (SpeechAct.RETRACT, TopicRelation.ACTIVE, DomainFit.IN_DOMAIN, 0, 0, 0, 0, 0),
    (SpeechAct.CLOSE, TopicRelation.UNRELATED, DomainFit.NONE, 0, 0, 0, 0, 0),
)
def _roster(n):
    kids = (RosterItem(id="item_1", age_years=6, age_label="6 साल", tag="male", name="अमन"), RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="female", name="मीरा"))
    return list(kids[:n])
def _anchors():
    for pid, ids in _IDS.items():
        pack = PACKS[pid]
        for qid in ids:
            spec, subs = pack.by_id[qid], [SubStep.MAIN]
            if spec.followups: subs.append(SubStep.FOLLOWUP)
            if spec.roster_scoped: subs.append(SubStep.WHICH_ITEM)
            rosters = (2,) if spec.roster_scoped else ((0,) if spec.kind is QuestionKind.ROUTING_GATE else (0, 2))
            for sub in subs:
                for n in rosters: yield pack, spec, sub, n
def _state(pack, spec, n, sub):
    st = _BASE[pack.id].model_copy(deep=True)
    slot = "caller_name" if pack.roster_cfg else "patient_name"
    if spec.requires_slot != slot: st.profile[slot] = "सुनीता" if pack.roster_cfg else "रवि"
    if pack.roster_cfg: st.profile["marital_status"] = "married"
    else: st.profile["patient_age"] = 28
    n = 0 if spec.kind is QuestionKind.ROUTING_GATE else n
    st.roster, st.roster_stated = _roster(n), True if pack.roster_cfg else bool(n)
    if spec.id != "intro_bridge": st.answers["intro_bridge"] = AnswerRecord(state=AnswerStatus.CLEAR)
    st.answers.pop(spec.id, None)
    # WHICH_ITEM and FOLLOWUP exist only after the parent question recorded CONCERN.
    if sub in (SubStep.WHICH_ITEM, SubStep.FOLLOWUP):
        st.answers[spec.id] = AnswerRecord(state=AnswerStatus.CONCERN)
    st.cursor = CursorStep(qid=spec.id, substep=sub)
    return st
def _other(pack, st, active):
    for q in pack.questions:
        if q.id == active or not q.is_routable or q.kind not in (QuestionKind.GATE, QuestionKind.ROUTING_GATE): continue
        blocked = [r for r in q.skip_if if "group_no_concern" not in (pack.rules.get(r) or {})]
        if not any(engine.eval_rule(pack, st, r) for r in blocked): return q.id
def _payload(spec, domain):
    if domain is not DomainFit.IN_DOMAIN: return None, None
    if spec.kind is QuestionKind.SINGLE_CHOICE: return TurnOutcome.ANSWERED, next(iter(spec.options))
    if spec.kind is QuestionKind.MULTI_CHOICE: return TurnOutcome.ANSWERED, [next(k for k, o in spec.options.items() if not o.exclusive)]
    if spec.kind is QuestionKind.SCORE: return TurnOutcome.CONCERN, {"score": 3, "level": 3}
    return TurnOutcome.CONCERN, None
def _listed_tap(spec, domain, value):
    if spec.kind is QuestionKind.PROFILE: return None
    if domain is DomainFit.UNLISTED: return None if spec.kind not in _CHOICE or spec.saves_to_profile else ("clear" if spec.kind is QuestionKind.SCORE else next(iter(spec.options)))
    if domain is not DomainFit.IN_DOMAIN: return None
    if spec.kind in _YES: return "clear"
    if spec.kind is QuestionKind.SINGLE_CHOICE: return next(k for k in spec.options if k != value)
    if spec.kind is QuestionKind.MULTI_CHOICE: return "none" if "none" in spec.options else None
    return "clear" if spec.kind is QuestionKind.SCORE else None
def _interp(pack, st, spec, act, rel, dom, co, aside, red, jump):
    status, value = _payload(spec, dom)
    claims = []
    if rel is TopicRelation.ACTIVE and act is SpeechAct.RETRACT:
        claims.append(ClinicalClaim(target_qid=spec.id, relation=rel, domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.RETRACT))
    elif rel is TopicRelation.ACTIVE and act is not SpeechAct.CLARIFY and dom is DomainFit.UNLISTED:
        claims.append(ClinicalClaim(target_qid=spec.id, relation=rel, domain_fit=DomainFit.UNLISTED))
    elif rel is TopicRelation.ACTIVE and act is not SpeechAct.CLARIFY and dom is DomainFit.IN_DOMAIN:
        claims.append(ClinicalClaim(target_qid=spec.id, relation=rel, domain_fit=dom, status=status, value=value))
    elif rel is TopicRelation.CLINICAL_ASIDE:
        claims.append(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE))
    if co and (other := _other(pack, st, spec.id)):
        claims.append(ClinicalClaim(target_qid=other, relation=TopicRelation.OTHER_FORM, domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.CONCERN))
    if aside and rel is not TopicRelation.CLINICAL_ASIDE: claims.append(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE))
    probs = {nxt: 0.99} if jump and (nxt := _other(pack, st, spec.id)) else {}
    return TurnInterpretation(act=act, active_relation=rel, active_domain=dom, claims=tuple(claims), caller_stop=act is SpeechAct.CLOSE, jump_topic=bool(jump), next_qid_probs=probs, redacted=bool(red)), value
def _cases():
    for pack, spec, sub, n in _anchors():
        st = _state(pack, spec, n, sub)
        for pi, row in enumerate(_PATTERNS):
            act, rel, dom, co, aside, red, jump, tap = row
            interp, value = _interp(pack, st, spec, act, rel, dom, co, aside, red, jump)
            ev = TurnEvidence(source=EvidenceSource.VOICE, snapshot_step=st.step_token, transcript=UTT, received_at=1.0)
            listed = _listed_tap(spec, dom, value) if tap and sub is SubStep.MAIN else None
            yield pack, spec, sub, n, pi, st, ev, interp, listed
def _tag(pack, spec, sub, n, pi):
    return f"pack={pack.id} q={spec.id} kind={spec.kind.value} sub={sub.value} roster={n} pattern={pi}"
_EMERGENCY_UTT = "अभी बच्चे को साँस नहीं आ रही है, तुरंत डॉक्टर को दिखाना है!"
def _emergency_cases():
    """The same anchors, but every turn also carries an active emergency plus a full in-domain answer to the question
    being asked. The claim is deliberate bait: a closed call must read neither it nor any later question."""
    for pack, spec, sub, n in _anchors():
        st = _state(pack, spec, n, sub)
        claim = ClinicalClaim(target_qid=spec.id, relation=TopicRelation.ACTIVE, domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.CONCERN)
        interp = TurnInterpretation(act=SpeechAct.ANSWER, active_relation=TopicRelation.ACTIVE, active_domain=DomainFit.IN_DOMAIN, claims=(claim,), emergency=True)
        ev = TurnEvidence(source=EvidenceSource.VOICE, snapshot_step=st.step_token, transcript=_EMERGENCY_UTT, received_at=1.0)
        yield pack, spec, sub, n, st, ev, interp
class PolicyInvariantTests(unittest.TestCase):
    def test_universal_laws_hold_on_generated_matrix(self):
        bad, seen = [], 0
        for pack, spec, sub, n, pi, st, ev, interp, listed in _cases():
            seen += 1
            tag, qid = _tag(pack, spec, sub, n, pi), st.cursor.qid
            nxt, out = apply_policy(pack, st, ev, interp)
            prior = st.answers.get(qid)
            aside = interp.active_relation in (TopicRelation.CLINICAL_ASIDE, TopicRelation.UNRELATED)
            if aside and st.cursor.attempts == 0 and not (prior is not None and prior.state in _RESOLVED):
                got = nxt.answers.get(qid)
                if got is not None and got.state in _RESOLVED: bad.append(f"law1 phantom {tag} wrote {got.state.value}")
            accepted = {d.target_qid for d in out.decisions if d.disposition in _ACCEPT and d.target_qid}
            for aq, rec in nxt.answers.items():
                old = st.answers.get(aq)
                if rec.state is AnswerStatus.NOT_APPLICABLE or (old is not None and old.model_dump() == rec.model_dump()) or aq in accepted: continue
                src = next((pack.by_id[s] for s in accepted if s in pack.by_id and pack.by_id[s].saves_to_profile), None)
                if src is not None and aq in pack.by_id and pack.by_id[aq].saves_to_profile == src.saves_to_profile: continue
                bad.append(f"law2 untethered {tag} {aq}={rec.state.value} accepts={sorted(accepted)}")
            cmd = out.command
            if cmd.redacted or cmd.jump_topic or cmd.stated:
                proj = reports.project_summary(pack, nxt)
                if UTT in proj.patient_summary_text or UTT in str(proj.page2_groups):
                    bad.append(f"law3 utterance leaked {tag} redacted={cmd.redacted} jump={cmd.jump_topic} stated={list(cmd.stated)}")
            if nxt.last_answered_qid not in (st.last_answered_qid, qid):
                bad.append(f"law4 cursor {tag} last {st.last_answered_qid!r} -> {nxt.last_answered_qid!r}")
            if listed:
                tap_cmd = engine.quick_reply_command(pack, st, listed)
                current = engine.reduce_turn(pack, st, tap_cmd)
                restored, arb, _moved = arbitrate_tap_turn(pack, st, tap_cmd, current, ev, interp)
                if not any(d.disposition is Disposition.CONFLICT and d.reason is DecisionReason.SOURCE_CONFLICT for d in arb.decisions):
                    bad.append(f"law5 no CONFLICT {tag} tap={listed} speech={arb.command.outcome.value}")
                held = restored.answers.get(qid)
                if held is not None and held.state in _RESOLVED: bad.append(f"law5 resolved through contradiction {tag} tap={listed} state={held.state.value}")
        self.assertGreaterEqual(seen, 250, f"matrix only produced {seen} cases")
        for pack, qid in ((PACKS["shishu-hi"], "fam_infant_death"), (PACKS["fever-opd-hi"], "breathing_or_chest")):
            st = _state(pack, pack.by_id[qid], 0, SubStep.MAIN)
            concern = TurnInterpretation(act=SpeechAct.ANSWER, active_relation=TopicRelation.ACTIVE, active_domain=DomainFit.IN_DOMAIN, claims=(ClinicalClaim(target_qid=qid, relation=TopicRelation.ACTIVE, domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.CONCERN),))
            ev1 = TurnEvidence(source=EvidenceSource.VOICE, snapshot_step=st.step_token, transcript="हाँ गाँठ", received_at=2.0)
            st1, out1 = apply_policy(pack, st, ev1, concern)
            cid = db.create_call(pack_id=pack.id)
            db.commit_semantic_turn(cid, pack, st1, ev1, concern, out1.decisions)
            retract = TurnInterpretation(act=SpeechAct.RETRACT, active_relation=TopicRelation.ACTIVE, active_domain=DomainFit.IN_DOMAIN, claims=(ClinicalClaim(target_qid=qid, relation=TopicRelation.ACTIVE, domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.RETRACT),))
            ev2 = TurnEvidence(source=EvidenceSource.VOICE, snapshot_step=st1.step_token, transcript="गलती से हाँ", received_at=3.0)
            st2, out2 = apply_policy(pack, st1, ev2, retract)
            db.commit_semantic_turn(cid, pack, st2, ev2, retract, out2.decisions)
            saved = db.call_state_from_row(db.get_call(cid))
            n_claims, n_dec = len(db.list_clinical_claims(cid)), len(db.list_policy_decisions(cid))
            if saved.answers[qid].state is not AnswerStatus.CLEAR or n_claims < 2 or n_dec < 2:
                bad.append(f"law6 {pack.id} {qid} state={saved.answers[qid].state.value} claims={n_claims} decisions={n_dec}")
        self.assertFalse(bad, f"{len(bad)} violations / {seen} cases\n" + "\n".join(bad[:12]))
    def test_emergency_closes_the_call_and_keeps_the_invariants(self):
        """An active danger sign ends the call on that turn: no question is read, asked, or advanced past.

        The anchor turn deliberately co-reports a full in-domain CONCERN for the question on the cursor, so an
        implementation that kept walking the form would record the answer and move on. Nothing about that claim may
        land, and `reduce_turn` must be closed for every later turn too — the emergency is not a phase to leave.
        """
        bad, packs, seen = [], set(), 0
        for pack, spec, sub, n, st, ev, interp in _emergency_cases():
            seen, tag = seen + 1, f"pack={pack.id} q={spec.id} kind={spec.kind.value} sub={sub.value} roster={n}"
            nxt, out = apply_policy(pack, st, ev, interp)
            packs.add(pack.id)
            if nxt.phase is Phase.ASKING or nxt.close_reason is not CloseReason.EMERGENCY:
                bad.append(f"emg1 not closed {tag} phase={nxt.phase.value} reason={nxt.close_reason}")
            if nxt.step_token != f"closing:{CloseReason.EMERGENCY}":
                bad.append(f"emg2 still askable {tag} token={nxt.step_token}")
            if out.command.emergency is not True or any(d.cursor_action is not CursorAction.CLOSE for d in out.decisions):
                bad.append(f"emg3 command not a close {tag} emergency={out.command.emergency} actions={[d.cursor_action.value for d in out.decisions]}")
            if not any(d.reason is DecisionReason.EMERGENCY for d in out.decisions):
                bad.append(f"emg4 no EMERGENCY decision {tag} reasons={[d.reason.value for d in out.decisions]}")
            # The co-reported answer is bait: a closed call reads nothing from the question it was asking.
            prior, after = st.answers.get(st.cursor.qid), nxt.answers.get(st.cursor.qid)
            if (prior.model_dump() if prior else None) != (after.model_dump() if after else None):
                bad.append(f"emg5 answer read through the emergency {tag} {st.cursor.qid} -> {after.state.value if after else None}")
            for aq, rec in nxt.answers.items():
                old = st.answers.get(aq)
                if rec.state is not AnswerStatus.NOT_APPLICABLE and (old is None or old.model_dump() != rec.model_dump()):
                    bad.append(f"emg6 untethered write {tag} {aq}={rec.state.value}")
            if nxt.cursor.qid != st.cursor.qid or nxt.cursor.attempts or nxt.cursor.followup_idx:
                bad.append(f"emg7 advanced {tag} {st.cursor.qid}/{sub.value} -> {nxt.cursor.qid}/{nxt.cursor.substep.value} attempts={nxt.cursor.attempts}")
            if nxt.last_answered_qid != st.last_answered_qid or list(nxt.semantic_flags) != list(st.semantic_flags):
                bad.append(f"emg8 ledger moved {tag} last={nxt.last_answered_qid!r}")
            if not nxt.emergency_note or nxt.emergency_note != ev.transcript:
                bad.append(f"emg9 caller's words lost {tag} note={nxt.emergency_note!r}")
            # A later turn can never resume the form, and cannot overwrite why the call ended.
            later = engine.reduce_turn(pack, nxt, TurnCommand(outcome=TurnOutcome.CONCERN, value=next(iter(spec.options), None) if spec.options else None, utterance=UTT))
            if later.phase is Phase.ASKING or later.close_reason is not CloseReason.EMERGENCY or later.answers != nxt.answers:
                bad.append(f"emg10 call resumed {tag} phase={later.phase.value} reason={later.close_reason}")
            # Report invariants: the emergency band owns page one, and no routine question is put to the mother.
            cfg, proj = pack.report_cfg, reports.project_summary(pack, nxt)
            if not (proj.emergency and proj.close_reason == str(CloseReason.EMERGENCY) and proj.incomplete):
                bad.append(f"emg11 report not an emergency {tag} emergency={proj.emergency} reason={proj.close_reason!r} incomplete={proj.incomplete}")
            fillers, routine = set(cfg.get("emergency_questions") or ()), set(cfg.get("routine_questions") or ())
            # A closed call adds no question of its own: page one may only carry copy for concerns already on record
            # (the bait claim must not become one) plus the emergency fillers, never a routine question.
            from_record = [q for q in proj.patient_questions if q not in fillers]
            on_record = [qid for qid, rec in nxt.answers.items() if rec.state is AnswerStatus.CONCERN]
            if routine & set(proj.patient_questions) or len(from_record) > len(on_record):
                bad.append(f"emg12 questions invented after emergency {tag} from_record={len(from_record)} concerns={len(on_record)} questions={list(proj.patient_questions)}")
            if not any(str(cfg.get("emergency_flag_en") or "") in f for f in proj.red_flags) or "Active emergency" not in proj.doctor_markdown:
                bad.append(f"emg13 emergency band missing {tag} flags={list(proj.red_flags)}")
            if _EMERGENCY_UTT.rstrip("।. ") not in proj.doctor_markdown:
                bad.append(f"emg14 mother's words missing from the clinician note {tag}")
        self.assertEqual(packs, {"shishu-hi", "fever-opd-hi"}, f"emergency pattern only covered {sorted(packs)}")
        self.assertEqual(seen, len(list(_anchors())), "every anchor must be exercised under an emergency")
        self.assertFalse(bad, f"{len(bad)} emergency violations / {seen} cases\n" + "\n".join(bad[:12]))
if __name__ == "__main__":
    unittest.main()
