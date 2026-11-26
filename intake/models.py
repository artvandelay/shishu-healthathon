"""Typed state, turn commands, and pack configuration models for V3 intake."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

SEP, SEP_SP = "\N{SEMICOLON}", "\N{SEMICOLON} "  # note clauses are split and re-joined on this

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
_Frozen = create_model("_Frozen", __base__=Strict, __config__=ConfigDict(frozen=True))
_Pack = create_model("_Pack", __config__=ConfigDict(arbitrary_types_allowed=True, frozen=True))

def _rec(_name: str, _base: type[BaseModel] = _Frozen, **fields: Any) -> Any:
    return create_model(_name, __base__=_base, **fields)
def _enum(_name: str, _members: str) -> StrEnum:  # members are space-separated, valued as their lowercase
    return StrEnum(_name, {m: m.lower() for m in _members.split()})
Phase, SubStep = _enum("Phase", "ASKING CLOSING"), _enum("SubStep", "MAIN WHICH_ITEM FOLLOWUP")
QuestionKind = _enum("QuestionKind", "PROFILE BRIDGE ROUTING_GATE GATE SINGLE_CHOICE MULTI_CHOICE SCORE OPEN_CAPTURE")
TurnOutcome = _enum("TurnOutcome", "CLEAR CONCERN ANSWERED UNKNOWN CLARIFY RETRACT NOT_ABOUT_ACTIVE")
CloseReason = _enum("CloseReason", "COMPLETE SILENCE TIME_LIMIT EMERGENCY CALLER_STOP TECHNICAL_FAILURE")
_YES_NO = (QuestionKind.BRIDGE, QuestionKind.ROUTING_GATE, QuestionKind.GATE, QuestionKind.OPEN_CAPTURE)
AnswerStatus = _enum("AnswerStatus", "CONCERN CLEAR ANSWERED UNKNOWN UNANSWERED NOT_APPLICABLE")
AnswerStatus.is_resolved = property(lambda s: s in (s.CONCERN, s.CLEAR, s.ANSWERED, s.UNKNOWN, s.UNANSWERED))

def dedupe_clauses(clauses: Iterable[str]) -> str:
    seen: dict[str, str] = {}
    for text in filter(None, (" ".join(p.split()) for c in clauses for p in str(c or "").split(SEP))): seen.setdefault(text.casefold(), text)
    return SEP_SP.join(seen.values())
def _joined(rec: AnswerRecord, attr: str, *, skip_clear: bool = False) -> str:
    return dedupe_clauses([getattr(rec.note, attr), *(getattr(s.note, attr) for s in rec.followups.values() if not (skip_clear and s.state in (AnswerStatus.CLEAR, AnswerStatus.UNKNOWN, AnswerStatus.UNANSWERED)))])
class NoteText(_Frozen):
    verbatim: str = ""
    patient_safe: str = ""
    @classmethod
    def from_text(cls, text: str, *, redacted: bool = False) -> NoteText:
        v = dedupe_clauses([text])
        return cls(verbatim=v, patient_safe="" if redacted else v)
    def append(self, text: str, *, redacted: bool = False) -> NoteText:
        safe = self.patient_safe if redacted else dedupe_clauses([self.patient_safe, text])
        return self.model_copy(update={"verbatim": dedupe_clauses([self.verbatim, text]), "patient_safe": safe})
RosterInput = _rec("RosterInput", id=(str | None, None), age_years=(float, Field(ge=0.0)), tag=(str, "unknown"), name=(str | None, None))
RosterItem = _rec("RosterItem", id=(str, ...), age_years=(float, Field(ge=0.0)), age_label=(str, ...), tag=(str, "unknown"), name=(str | None, None))
FollowupSlot = _rec("FollowupSlot", Strict, state=(AnswerStatus, ...), note=(NoteText, Field(default_factory=NoteText)))
AnswerRecord = _rec("AnswerRecord", Strict, state=(AnswerStatus, ...), value=(Any, None), choice_label=(str, ""), item_ids=(list[str], Field(default_factory=list)),
                    note=(NoteText, Field(default_factory=NoteText)), followups=(dict[str, FollowupSlot], Field(default_factory=dict)))
AnswerRecord.verbatim_text = property(lambda self: _joined(self, "verbatim"))
AnswerRecord.patient_safe_text = property(lambda self: _joined(self, "patient_safe", skip_clear=True))
CursorStep = _rec("CursorStep", Strict, qid=(str, ""), substep=(SubStep, SubStep.MAIN), followup_idx=(int, Field(default=0, ge=0)),
                  hold_slot=(str | None, None), attempts=(int, Field(default=0, ge=0)), open_rounds=(int, Field(default=0, ge=0)))
ProfilePatch = _rec("ProfilePatch", slots=(dict[str, Any], Field(default_factory=dict)), roster_complete=(bool | None, None),
                    roster=(tuple[RosterInput, ...] | None, None))
ProfilePatch.is_empty = property(lambda p: not p.slots and p.roster_complete is None and p.roster is None)
VariantSpec = _rec("VariantSpec", when=(str, ...), say=(str, ...))
OptionSpec = _rec("OptionSpec", label=(str, ...), en=(str, ""), criteria=(str, ""), exclusive=(bool, False))
TurnCommand = _rec(
    "TurnCommand", emergency=(bool, False), caller_stop=(bool, False), profile_patch=(ProfilePatch | None, None),
    profile_turn_attempted=(bool, False), outcome=(TurnOutcome, TurnOutcome.NOT_ABOUT_ACTIVE), value=(Any, None), item_ref=(str, ""),
    followup_answered=(bool, False), stated=(dict[str, TurnOutcome], Field(default_factory=dict)), semantic_flags=(tuple[str, ...], ()),
    jump_topic=(bool, False), next_qid_probs=(dict[str, float], Field(default_factory=dict)), utterance=(str, ""), redacted=(bool, False))
class CallState(Strict):
    phase: Phase = Phase.ASKING
    close_reason: CloseReason | None = None
    profile: dict[str, Any] = Field(default_factory=dict)
    roster: list[RosterItem] = Field(default_factory=list)
    roster_stated: bool = False
    roster_incomplete: bool = False
    semantic_flags: list[str] = Field(default_factory=list)
    cursor: CursorStep = Field(default_factory=CursorStep)
    last_answered_qid: str | None = None
    emergency_note: str = ""
    answers: dict[str, AnswerRecord] = Field(default_factory=dict)
    ordered_roster = property(lambda s: sorted(s.roster, key=lambda x: x.age_years, reverse=True))
    def to_dict(self) -> dict[str, Any]: return self.model_dump(mode="json")
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CallState: return cls.model_validate(raw) if raw else cls()
    @property
    def step_token(self) -> str:
        c = self.cursor
        if self.phase is not Phase.ASKING: return f"closing:{self.close_reason or CloseReason.COMPLETE}"
        head = [f"{c.substep}:{c.followup_idx}"] if c.substep is not SubStep.MAIN else []
        return ":".join([c.qid, *head, *(f"{k}:{v}" for k, v in (("more", c.open_rounds), ("hold", c.hold_slot), ("try", c.attempts)) if v)])
QuestionSpec = _rec(
    "QuestionSpec", id=(str, ...), group=(str, ...), kind=(QuestionKind, ...), say=(str, ...), stem=(str, ""), stem_en=(str, ""), need=(str, ""), tag=(str, ""),
    clinician_group=(str, ""), requires_slot=(str, ""), reask=(str, ""), saves_to_profile=(str, ""), roster_scoped=(bool, False),
    skip_if=(tuple[str, ...], ()), followups=(tuple[str, ...], ()), variants=(tuple[VariantSpec, ...], ()),
    options=(dict[str, OptionSpec], Field(default_factory=dict)), quick_options=(bool, False), score_rubric=(tuple[str, ...], ()),
    concern_if=(dict[str, Any], Field(default_factory=dict)), shares_recorded_concerns=(bool, False), stated_need=(str, ""))
QuestionSpec.is_routable = property(lambda q: q.kind not in (QuestionKind.PROFILE, QuestionKind.BRIDGE))
QuestionSpec.is_reportable = property(lambda q: q.is_routable and q.kind is not QuestionKind.ROUTING_GATE)
QuestionSpec.allows_yes_no = property(lambda q: q.kind in _YES_NO)
PackConfig = _rec(
    "PackConfig", _Pack, id=(str, ...), root=(Path, ...), meta=(dict[str, Any], ...), prompts=(dict[str, Any], ...),
    profile_slots=(dict[str, dict[str, Any]], ...), roster_cfg=(dict[str, Any] | None, ...), extractor_prompt=(str, ...),
    implications=(tuple[dict[str, Any], ...], ...), semantic_flags=(dict[str, dict[str, Any]], ...), rules=(dict[str, dict[str, Any]], ...),
    groups=(dict[str, dict[str, str]], ...), followups=(dict[str, str], ...), questions=(tuple[QuestionSpec, ...], ...),
    by_id=(dict[str, QuestionSpec], ...), report_cfg=(dict[str, Any], ...), decisions=(dict[str, Any], ...),
    seed_scenarios=(tuple[dict[str, Any], ...], ...))
