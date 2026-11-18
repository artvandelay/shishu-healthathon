"""Pure semantic policy: map interpretation + evidence to ``TurnCommand`` and ``reduce_turn``."""

from __future__ import annotations

from typing import Any

from intake import engine
from intake.models import AnswerRecord, AnswerStatus, CallState, NoteText, PackConfig, Phase, QuestionKind, QuestionSpec, SubStep, TurnCommand, TurnOutcome
from intake.turns import ClinicalClaim, CursorAction, DecisionReason, Disposition, DomainFit, PolicyDecision, PolicyResult, SpeechAct, TopicRelation, TurnEvidence, TurnInterpretation

CHOICE_KINDS = (QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE, QuestionKind.SCORE)
GATES = (QuestionKind.GATE, QuestionKind.ROUTING_GATE)
OPENISH = (QuestionKind.OPEN_CAPTURE, QuestionKind.BRIDGE)
_CHOICE_KINDS, _GATES, _OPENISH = CHOICE_KINDS, GATES, OPENISH
_GATE_OUTCOMES = (TurnOutcome.CONCERN, TurnOutcome.CLEAR, TurnOutcome.UNKNOWN)
_ACTIVE_OUTCOMES = (TurnOutcome.CONCERN, TurnOutcome.CLEAR, TurnOutcome.UNKNOWN, TurnOutcome.ANSWERED)
_STATED_OUTCOMES = (TurnOutcome.CONCERN, TurnOutcome.CLEAR)
def _skip_blocked(pack: PackConfig, state: CallState, spec: QuestionSpec) -> bool:
    return any(engine.eval_rule(pack, state, r) for r in spec.skip_if if "group_no_concern" not in (pack.rules.get(r) or {}))
def _domain_membership(spec: QuestionSpec, claim: ClinicalClaim) -> bool:
    kind, status, value = spec.kind, claim.status, claim.value
    if kind in _GATES: return status in _GATE_OUTCOMES and value is None
    if kind in _CHOICE_KINDS:
        if status is TurnOutcome.UNKNOWN and value is None: return True
        if kind is QuestionKind.SINGLE_CHOICE: return isinstance(value, str) and value in spec.options
        if kind is QuestionKind.MULTI_CHOICE:
            raw = value if isinstance(value, (list, tuple, set)) else ([value] if isinstance(value, str) and value else [])
            return any(isinstance(k, str) and k in spec.options for k in raw)
        raw_score = value.get("score") if isinstance(value, dict) else value
        if isinstance(raw_score, bool): return False
        try: float(raw_score); return True
        except (TypeError, ValueError): return False
    if kind in _OPENISH: return status in _ACTIVE_OUTCOMES
    return False
def _rejected_target(pack: PackConfig, state: CallState, target: str, active_qid: str) -> bool:
    spec = pack.by_id.get(target)
    return spec is None or not spec.is_routable or target == active_qid or _skip_blocked(pack, state, spec)
def named_clarify_ok(pack: PackConfig, state: CallState, target: str) -> bool:
    spec = pack.by_id.get(target)
    if spec is None or not spec.is_routable or _skip_blocked(pack, state, spec): return False
    rec = state.answers.get(target)
    return rec is not None and rec.state is AnswerStatus.UNKNOWN
def _base_command(interpretation: TurnInterpretation, utt: str) -> TurnCommand:
    return TurnCommand(emergency=interpretation.emergency, caller_stop=interpretation.caller_stop, profile_patch=interpretation.profile_patch,
                       profile_turn_attempted=interpretation.profile_turn_attempted, item_ref=interpretation.item_ref, followup_answered=interpretation.followup_answered,
                       semantic_flags=interpretation.semantic_flags, jump_topic=interpretation.jump_topic, next_qid_probs=dict(interpretation.next_qid_probs),
                       redacted=interpretation.redacted, utterance=utt)
def evaluate_policy(pack: PackConfig, state: CallState, evidence: TurnEvidence, interpretation: TurnInterpretation) -> PolicyResult:
    utt, active_qid = " ".join((evidence.transcript or "").split()), state.cursor.qid
    active_spec, cmd = (pack.by_id.get(active_qid) if active_qid else None), _base_command(interpretation, utt)
    D = PolicyDecision
    if interpretation.emergency:
        return PolicyResult(decisions=(D(disposition=Disposition.IGNORE, reason=DecisionReason.EMERGENCY, cursor_action=CursorAction.CLOSE),), command=cmd.model_copy(update={"emergency": True}))
    if interpretation.caller_stop:
        return PolicyResult(decisions=(D(disposition=Disposition.IGNORE, reason=DecisionReason.CALLER_STOP, cursor_action=CursorAction.CLOSE),), command=cmd.model_copy(update={"caller_stop": True}))
    decisions: list[PolicyDecision] = []
    unlisted_accepted = False
    active_accepted, stated = False, {}
    substep, allow_active_substep = state.cursor.substep, state.cursor.substep is not SubStep.MAIN
    for idx, claim in enumerate(interpretation.claims):
        rel = claim.relation
        if rel is TopicRelation.CLINICAL_ASIDE:
            decisions.append(D(claim_index=idx, disposition=Disposition.RETAIN_OBSERVATION, reason=DecisionReason.RESIDUAL_CLINICAL)); continue
        if rel is TopicRelation.UNRELATED:
            decisions.append(D(claim_index=idx, disposition=Disposition.IGNORE, reason=DecisionReason.UNRELATED)); continue
        if rel is TopicRelation.OTHER_FORM:
            target = claim.target_qid
            if not target or _rejected_target(pack, state, target, active_qid) or claim.status not in _STATED_OUTCOMES:
                decisions.append(D(claim_index=idx, disposition=Disposition.IGNORE, reason=DecisionReason.REJECTED_TARGET, target_qid=target))
            else:
                decisions.append(D(claim_index=idx, disposition=Disposition.ACCEPT_FORM, reason=DecisionReason.OTHER_FORM, target_qid=target))
                stated[target] = claim.status
            continue
        target = claim.target_qid or active_qid
        if target != active_qid or active_spec is None:
            decisions.append(D(claim_index=idx, disposition=Disposition.IGNORE, reason=DecisionReason.REJECTED_TARGET, target_qid=target or None)); continue
        if claim.domain_fit is DomainFit.UNLISTED or interpretation.active_domain is DomainFit.UNLISTED:
            if active_spec.kind in _CHOICE_KINDS and substep is SubStep.MAIN and not active_spec.saves_to_profile and interpretation.active_relation is TopicRelation.ACTIVE and utt:
                unlisted_accepted = True
                decisions.append(D(claim_index=idx, disposition=Disposition.ACCEPT_UNLISTED, reason=DecisionReason.UNLISTED_TOPIC, target_qid=active_qid))
                active_accepted, cmd = True, cmd.model_copy(update={"outcome": TurnOutcome.ANSWERED, "value": None})
            else:
                decisions.append(D(claim_index=idx, disposition=Disposition.IGNORE, reason=DecisionReason.REJECTED_TARGET, target_qid=active_qid))
            continue
        if claim.status is TurnOutcome.RETRACT:
            decisions.append(D(claim_index=idx, disposition=Disposition.ACCEPT_FORM, reason=DecisionReason.RETRACT, target_qid=active_qid))
            active_accepted, cmd = True, cmd.model_copy(update={"outcome": TurnOutcome.RETRACT}); continue
        in_domain = claim.domain_fit is DomainFit.IN_DOMAIN or interpretation.active_domain is DomainFit.IN_DOMAIN
        if (not in_domain and not allow_active_substep) or (not allow_active_substep and not _domain_membership(active_spec, claim)) or claim.status is None:
            decisions.append(D(claim_index=idx, disposition=Disposition.IGNORE, reason=DecisionReason.REJECTED_TARGET, target_qid=active_qid)); continue
        reason = DecisionReason.CALLER_UNKNOWN if claim.status is TurnOutcome.UNKNOWN else DecisionReason.IN_DOMAIN
        decisions.append(D(claim_index=idx, disposition=Disposition.ACCEPT_FORM, reason=reason, target_qid=active_qid))
        active_accepted = True
        extra: dict[str, Any] = {"outcome": claim.status}
        if claim.value is not None: extra["value"] = claim.value
        cmd = cmd.model_copy(update=extra)
    if interpretation.act is SpeechAct.CLARIFY:
        target = (interpretation.clarify_target_qid or "").strip() or None
        if target and target != active_qid and named_clarify_ok(pack, state, target):
            decisions.append(D(disposition=Disposition.CLARIFY, reason=DecisionReason.NAMED_CLARIFY, target_qid=target))
            stated[target] = TurnOutcome.CLARIFY
        else:
            decisions.append(D(disposition=Disposition.CLARIFY, reason=DecisionReason.ACTIVE_CLARIFY, target_qid=active_qid or None))
            if not active_accepted: cmd = cmd.model_copy(update={"outcome": TurnOutcome.CLARIFY})
    if not decisions: decisions.append(D(disposition=Disposition.IGNORE, reason=DecisionReason.UNRELATED))
    if stated: cmd = cmd.model_copy(update={"stated": {**cmd.stated, **stated}})
    if not active_accepted and interpretation.act is not SpeechAct.CLARIFY: cmd = cmd.model_copy(update={"outcome": TurnOutcome.NOT_ABOUT_ACTIVE})
    unlisted_record = (AnswerRecord(state=AnswerStatus.ANSWERED, value=None, choice_label="", note=NoteText.from_text(utt, redacted=engine._redact_note(cmd, active_qid)))
                       if unlisted_accepted else None)
    return PolicyResult(decisions=tuple(decisions), command=cmd, unlisted_record=unlisted_record)
def apply_policy(pack: PackConfig, state: CallState, evidence: TurnEvidence, interpretation: TurnInterpretation) -> tuple[CallState, PolicyResult]:
    result = evaluate_policy(pack, state, evidence, interpretation)
    st = state.model_copy(deep=True)
    if result.unlisted_record is not None and state.cursor.qid: st.answers[state.cursor.qid] = result.unlisted_record
    nxt = engine.reduce_turn(pack, st, result.command)
    c0, c1, action = state.cursor, nxt.cursor, CursorAction.STAY
    if nxt.phase is not Phase.ASKING: action = CursorAction.CLOSE
    elif not (c0.qid == c1.qid and c0.substep == c1.substep and c0.followup_idx == c1.followup_idx): action = CursorAction.ADVANCE
    decisions = tuple(d.model_copy(update={"cursor_action": action}) for d in result.decisions)
    return nxt, PolicyResult(decisions=decisions, command=result.command, unlisted_record=result.unlisted_record)