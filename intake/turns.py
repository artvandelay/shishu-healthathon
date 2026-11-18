"""Immutable turn evidence (Stage 1A) and semantic interpretation types (Stage 2A)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from intake.models import AnswerRecord, ProfilePatch, TurnCommand, TurnOutcome, _enum, _rec

EvidenceSource = _enum("EvidenceSource", "VOICE TEXT TAP LATE_STT")
SpeechAct = _enum("SpeechAct", "ANSWER CLARIFY RETRACT CLOSE NONE")
TopicRelation = _enum("TopicRelation", "ACTIVE OTHER_FORM CLINICAL_ASIDE UNRELATED")
DomainFit = _enum("DomainFit", "IN_DOMAIN UNLISTED NONE")
Disposition = _enum("Disposition", "ACCEPT_FORM ACCEPT_UNLISTED RETAIN_OBSERVATION CLARIFY IGNORE")
DecisionReason = _enum(
    "DecisionReason",
    "IN_DOMAIN UNLISTED_TOPIC RESIDUAL_CLINICAL OTHER_FORM ACTIVE_CLARIFY NAMED_CLARIFY RETRACT UNRELATED "
    "CALLER_UNKNOWN REJECTED_TARGET EMERGENCY CALLER_STOP",
)
CursorAction = _enum("CursorAction", "STAY ADVANCE CLOSE")
TurnEvidence = _rec("TurnEvidence", source=(EvidenceSource, ...), snapshot_step=(str, ...), transcript=(str, ""), received_at=(float, ...), selected_value=(Any | None, None))
ClinicalClaim = _rec(
    "ClinicalClaim", target_qid=(str | None, None), subject_ids=(tuple[str, ...], ()), relation=(TopicRelation, ...),
    domain_fit=(DomainFit, DomainFit.NONE), status=(TurnOutcome | None, None), value=(Any | None, None),
)
TurnInterpretation = _rec(
    "TurnInterpretation", schema_version=(str, "2"), act=(SpeechAct, ...), active_relation=(TopicRelation, ...), active_domain=(DomainFit, DomainFit.NONE),
    claims=(tuple[ClinicalClaim, ...], ()), clarify_target_qid=(str | None, None), emergency=(bool, False), caller_stop=(bool, False),
    profile_patch=(ProfilePatch | None, None), profile_turn_attempted=(bool, False), item_ref=(str, ""), followup_answered=(bool, False),
    semantic_flags=(tuple[str, ...], ()), jump_topic=(bool, False), next_qid_probs=(dict[str, float], Field(default_factory=dict)), redacted=(bool, False),
)
PolicyDecision = _rec(
    "PolicyDecision", claim_index=(int | None, None), disposition=(Disposition, ...), reason=(DecisionReason, ...),
    target_qid=(str | None, None), cursor_action=(CursorAction, CursorAction.STAY),
)
PolicyResult = _rec("PolicyResult", decisions=(tuple[PolicyDecision, ...], ...), command=(TurnCommand, ...), unlisted_record=(AnswerRecord | None, None))
StoredTurnEvidence = _rec(
    "StoredTurnEvidence", evidence_id=(str, ...), call_id=(int, ...), sequence=(int, ...), source=(EvidenceSource, ...),
    snapshot_step=(str, ...), transcript=(str, ""), received_at=(float, ...), selected_value=(Any | None, None),
)

def _persisted(cls, evidence: TurnEvidence, *, evidence_id: str, call_id: int, sequence: int) -> StoredTurnEvidence:
    return cls(
        evidence_id=evidence_id, call_id=call_id, sequence=sequence, source=evidence.source, snapshot_step=evidence.snapshot_step,
        transcript=evidence.transcript, received_at=evidence.received_at, selected_value=evidence.selected_value,
    )

StoredTurnEvidence.persisted = classmethod(_persisted)
StoredTurnInterpretation = _rec(
    "StoredTurnInterpretation", interpretation_id=(str, ...), call_id=(int, ...), sequence=(int, ...),
    evidence_id=(str, ...), interpretation=(TurnInterpretation, ...),
)
StoredClinicalClaim = _rec(
    "StoredClinicalClaim", claim_id=(str, ...), call_id=(int, ...), sequence=(int, ...), claim_index=(int, ...),
    target_qid=(str | None, ...), subject_ids=(tuple[str, ...], ...), relation=(TopicRelation, ...), domain_fit=(DomainFit, ...),
    status=(TurnOutcome | None, ...), value=(Any | None, ...),
)
StoredPolicyDecision = _rec(
    "StoredPolicyDecision", decision_id=(str, ...), call_id=(int, ...), sequence=(int, ...), claim_index=(int | None, ...),
    disposition=(Disposition, ...), reason=(DecisionReason, ...), target_qid=(str | None, ...), cursor_action=(CursorAction, ...),
)
StoredSemanticCommit = _rec(
    "StoredSemanticCommit", call_id=(int, ...), sequence=(int, ...), evidence=(StoredTurnEvidence, ...),
    interpretation=(StoredTurnInterpretation, ...), claims=(tuple[StoredClinicalClaim, ...], ...), decisions=(tuple[StoredPolicyDecision, ...], ...),
)