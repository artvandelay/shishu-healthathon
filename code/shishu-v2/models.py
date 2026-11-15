"""Typed call state for SHISHU v4.

Every value the cursor, Decision API, and reports exchange is a validated
model here. No raw dict soup, no -1 sentinels, no defensive type-check ladders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_LIVE_VOICE = "Kore"


class Phase(StrEnum):
    ASKING = "asking"
    CLOSING = "closing"


class CloseReason(StrEnum):
    COMPLETE = "complete"
    SILENCE = "silence"
    TIME_LIMIT = "time_limit"
    EMERGENCY = "emergency"
    CALLER_STOP = "caller_stop"
    TECHNICAL_FAILURE = "technical_failure"


class MaritalStatus(StrEnum):
    MARRIED = "married"
    ABOUT_TO_MARRY = "about_to_marry"
    UNMARRIED = "unmarried"

    @property
    def is_unmarried_or_engaged(self) -> bool:
        return self in (MaritalStatus.UNMARRIED, MaritalStatus.ABOUT_TO_MARRY)


class ChildSex(StrEnum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class SubStep(StrEnum):
    MAIN = "main"
    WHICH_CHILD = "which_child"
    FOLLOWUP = "followup"


class HoldReason(StrEnum):
    NAME_FRAGMENT = "name_fragment"
    MARITAL_STATUS = "marital_status"
    CHILDREN_AGES = "children_ages"


class AnswerStatus(StrEnum):
    CONCERN = "concern"
    CLEAR = "clear"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"

    @property
    def is_answered(self) -> bool:
        return self in (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN)


class TurnOutcome(StrEnum):
    CLEAR = "clear"
    CONCERN = "concern"
    UNKNOWN = "unknown"
    CLARIFY = "clarify"
    RETRACT = "retract"
    NOT_ABOUT_ACTIVE = "not_about_active"


class SideAction(StrEnum):
    NONE = ""
    ADD_CONCERN = "add_concern"
    CLEAR_OR_RETRACT = "clear_or_retract"
    CLARIFY_MEANING = "clarify_meaning"


class Strict(BaseModel):
    """Base for every model: unknown keys and invalid assignments fail immediately."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def dedupe_clauses(clauses: Iterable[str]) -> str:
    """Split on ';', normalise whitespace, and drop case-insensitive repeats."""
    seen: set[str] = set()
    kept: list[str] = []
    for clause in clauses:
        for part in str(clause or "").split(";"):
            clean = " ".join(part.split())
            if clean and clean.casefold() not in seen:
                seen.add(clean.casefold())
                kept.append(clean)
    return "; ".join(kept)


class NoteText(Strict, frozen=True):
    """One answer's words, with the Mother-facing copy kept diagnostic-free."""

    verbatim: str = ""
    mother_safe: str = ""

    @classmethod
    def from_text(cls, text: str, *, diagnostic: bool = False) -> NoteText:
        verbatim = dedupe_clauses([text])
        return cls(verbatim=verbatim, mother_safe="" if diagnostic else verbatim)

    def append(self, text: str, *, diagnostic: bool = False) -> NoteText:
        mother = self.mother_safe if diagnostic else dedupe_clauses([self.mother_safe, text])
        return self.model_copy(update={"verbatim": dedupe_clauses([self.verbatim, text]), "mother_safe": mother})


class ChildInput(Strict, frozen=True):
    age_years: float = Field(ge=0.0)
    sex: ChildSex = ChildSex.UNKNOWN
    name: str | None = None


class Child(Strict, frozen=True):
    id: str
    age_years: float = Field(ge=0.0)
    age_label_hi: str
    sex: ChildSex = ChildSex.UNKNOWN
    name: str | None = None


class CallerProfile(Strict):
    caller_name: str | None = None
    marital_status: MaritalStatus | None = None
    children: list[Child] = Field(default_factory=list)
    ever_pregnant: bool | None = None
    children_stated: bool = False
    pending_children: list[Child] = Field(default_factory=list)
    pending_child_count: int | None = None

    @property
    def ordered_children(self) -> list[Child]:
        return sorted(self.children, key=lambda child: child.age_years, reverse=True)


class FollowupSlot(Strict):
    state: AnswerStatus
    note: NoteText = Field(default_factory=NoteText)


class AnswerRecord(Strict):
    state: AnswerStatus
    child_ids: list[str] = Field(default_factory=list)
    note: NoteText = Field(default_factory=NoteText)
    followups: dict[str, FollowupSlot] = Field(default_factory=dict)

    @property
    def verbatim_text(self) -> str:
        return dedupe_clauses([self.note.verbatim, *(slot.note.verbatim for slot in self.followups.values())])

    @property
    def mother_safe_text(self) -> str:
        return dedupe_clauses([self.note.mother_safe, *(slot.note.mother_safe for slot in self.followups.values())])


class CursorStep(Strict):
    qid: str = "intro_name"
    substep: SubStep = SubStep.MAIN
    followup_idx: int = Field(default=0, ge=0)
    hold: HoldReason | None = None
    attempts: int = Field(default=0, ge=0)
    close_rounds: int = Field(default=0, ge=0)


class CallState(Strict):
    phase: Phase = Phase.ASKING
    close_reason: CloseReason | None = None
    profile: CallerProfile = Field(default_factory=CallerProfile)
    cursor: CursorStep = Field(default_factory=CursorStep)
    last_answered_qid: str | None = None
    emergency_note: str = ""
    answers: dict[str, AnswerRecord] = Field(default_factory=dict)

    @property
    def step_token(self) -> str:
        """Opaque identity of the exact step she is hearing; changes on every distinct step."""
        if self.phase is not Phase.ASKING:
            return f"closing:{self.close_reason or CloseReason.COMPLETE}"
        step = self.cursor
        token = step.qid
        if step.substep is not SubStep.MAIN:
            token = f"{token}:{step.substep}:{step.followup_idx}"
        if step.close_rounds:
            token = f"{token}:more:{step.close_rounds}"
        if step.hold is not None:
            token = f"{token}:hold:{step.hold}"
        if step.attempts:
            token = f"{token}:try:{step.attempts}"
        return token

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CallState:
        return cls.model_validate(raw) if raw else cls()


class ProfilePatch(Strict, frozen=True):
    caller_name: str | None = None
    marital_status: MaritalStatus | None = None
    child_count: int | None = Field(default=None, ge=0)
    children: tuple[ChildInput, ...] | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (self.caller_name, self.marital_status, self.child_count, self.children)
        )


class TurnCommand(Strict, frozen=True):
    emergency: bool = False
    caller_stop: bool = False
    profile_patch: ProfilePatch | None = None
    profile_turn_attempted: bool = False
    outcome: TurnOutcome = TurnOutcome.NOT_ABOUT_ACTIVE
    child_ref: str = ""
    followup_answered: bool = False
    side_qid: str = ""
    side_action: SideAction = SideAction.NONE
    utterance: str = ""
    diagnostic: bool = False
