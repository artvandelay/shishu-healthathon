"""Post-call facade: `summary.project_summary` already holds every report field."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import copy_hi
import db
import summary
from models import CallState
from summary import ScreeningSummary


def build_reports(call: Mapping[str, Any] | CallState) -> dict[str, Any]:
    """The clinician note plus the Mother payload for one finished call."""
    state = call if isinstance(call, CallState) else db.call_state_from_row(call)
    proj = summary.project_summary(state, fallback_reason="incomplete")
    return {
        "doctor_markdown": proj.doctor_markdown,
        "red_flags": list(proj.red_flags),
        "mother_questions_hi": list(proj.mother_questions_hi),
        "yes_summaries_hi": list(proj.yes_summaries_hi),
        "mother_name_hi": proj.mother_name,
        "children_label_hi": proj.children_label_hi,
        "summary": proj,
    }


def mother_summary_text(synthesis: Mapping[str, Any] | ScreeningSummary) -> str:
    """The Hindi Mother summary text, laid out by the one canonical builder in `copy_hi`."""
    if isinstance(synthesis, ScreeningSummary):
        return synthesis.mother_summary_text
    projected = synthesis.get("summary")
    if isinstance(projected, ScreeningSummary):
        return projected.mother_summary_text
    return copy_hi.mother_summary_text(
        synthesis.get("yes_summaries_hi") or [],
        synthesis.get("mother_questions_hi") or [],
    )
