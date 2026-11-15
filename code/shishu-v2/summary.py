"""Single-pass ScreeningSummary projection for both post-call artefacts.

`project_summary` walks the catalog once and hands the clinician note and the
Mother PDF a frozen answer, so neither report builder loops over `CATALOG` or
`answers` again and no diagnostic word ever reaches Hindi.
"""

from __future__ import annotations

import copy_hi
from models import AnswerRecord, AnswerStatus, CallerProfile, CallState, Strict
from triage_catalog import CATALOG, QuestionSpec

_HEADINGS = (
    "### Patient Profile & Chief Complaint",
    "### Positive Red Flags & NRROID Gate Alerts",
    "### Birth & Neonatal History",
    "### Development, Growth & Physical Features",
    "### Systemic, Infections & Hematology",
    "### Antenatal (Pregnancy) History",
    "### Family & Pedigree History",
    "### Pertinent Negatives",
)
_ALERTS, _HEMATOLOGY, _NEGATIVES = _HEADINGS[1], _HEADINGS[4], _HEADINGS[-1]
_GROUP_SECTION = (
    ("birth", 2), ("body", 3), ("illness", 4), ("pregnancy", 5), ("family", 6), ("close", 0)
)
_HEADING_BY_GROUP = {group: _HEADINGS[index] for group, index in _GROUP_SECTION}
_HEMATOLOGY_QIDS = frozenset(
    "body_bleeding_bruise body_joint_swelling mod_tooth_bleed mod_heavy_periods "
    "body_transfusion body_blood_count body_limb_pain".split()
)
_INCOMPLETE_REASONS = frozenset(
    "silence time_limit emergency caller_stop technical_failure incomplete abandoned".split()
)

_EMERGENCY_FLAG = "emergency: active danger sign now; hospital or 108"
_EMERGENCY_LINE = "Active emergency. Caller was told to go to the nearest hospital or call 108."
_NOTHING_REPORTED = "None reported in this call."
_NO_ALERTS = "No positive gate alerts."
_NO_NEGATIVES = "No confirmed negatives were recorded."
_PARTIAL_CALL = "Call ended before every applicable question was asked."
_YES_HEADER_HI = "जो बातें हाँ में दर्ज हुईं:"
_YES_EMPTY_HI = "इस बातचीत में कोई हाँ वाला बिंदु दर्ज नहीं हुआ।"
_QUESTION_HEADER_HI = "डॉक्टर से पूछने के सवाल:"


class Page2Group(Strict, frozen=True):
    """One Mother-PDF section: a Hindi group title and its formatted questions."""

    title: str
    items: tuple[str, ...] = ()


class ScreeningSummary(Strict, frozen=True):
    """Everything either report needs, projected once from a finished call."""

    mother_name: str
    children_label_hi: str
    children_label_en: str
    has_children: bool
    close_reason: str
    emergency: bool
    incomplete: bool
    doctor_markdown: str
    mother_summary_text: str
    red_flags: tuple[str, ...] = ()
    yes_summaries_hi: tuple[str, ...] = ()
    mother_questions_hi: tuple[str, ...] = ()
    page2_clear_groups: tuple[Page2Group, ...] = ()
    page2_unknown: tuple[str, ...] = ()
    page2_unasked_groups: tuple[Page2Group, ...] = ()


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _unique(items) -> list[str]:
    return list(dict.fromkeys(clean for clean in map(_clean, items) if clean))


def _page_groups(buckets: dict[str, list[str]]) -> tuple[Page2Group, ...]:
    """Page-2 groups in first-seen catalogue order, one group per Hindi title."""
    return tuple(Page2Group(title=t, items=tuple(v)) for t, v in buckets.items())


def _child_labels(profile: CallerProfile, record: AnswerRecord) -> tuple[str, str]:
    """Hindi label and English bracket tag, both empty unless she named a child."""
    if len(profile.children) <= 1 or not record.child_ids:
        return "", ""
    en = f" [{copy_hi.children_label_en(profile, record.child_ids)}]"
    return copy_hi.children_label_hi(profile, record.child_ids), en


def _note_rows(spec: QuestionSpec, record: AnswerRecord, profile: CallerProfile) -> list[tuple]:
    """The (heading, English line, red flag) rows one answered gate contributes."""
    heading = _HEMATOLOGY if spec.id in _HEMATOLOGY_QIDS else _HEADING_BY_GROUP[spec.group]
    if record.state in (AnswerStatus.CLEAR, AnswerStatus.UNKNOWN):
        return [(heading, f"{record.state.value.title()}. {spec.need_en}.", "")]
    if record.state is not AnswerStatus.CONCERN:
        return []
    note, tag = record.verbatim_text, _child_labels(profile, record)[1]
    head = f"{spec.nrroid_gate} {spec.id}" if spec.nrroid_gate else spec.id
    base, tail = f"{head}: {spec.need_en}{tag}", f" {note}" if note else ""
    words = f" Mother's words: {note}." if note else ""
    flag = f"{base}. {note}" if note else base
    concern_row = (heading, f"Concern. {spec.need_en}{tag}.{tail}", "")
    return [(_ALERTS, f"{base}.{words}", flag), concern_row]


def _header_lines(call: CallState, concerns: list[tuple], reason: str) -> list[str]:
    profile, ever = call.profile, call.profile.ever_pregnant
    chief = concerns[0][0].need_en if concerns else "Screening call with no positive gate."
    if not concerns and reason == "emergency":
        chief = "Active emergency reported during call."
    return [
        f"Caller: {_clean(profile.caller_name or '') or 'not reported'}.",
        f"Marital status: {profile.marital_status or 'not reported'}.",
        f"Ever pregnant: {ever if ever is not None else 'not reported'}.",
        f"Children: {copy_hi.children_label_en(profile) or 'none.'}",
        f"Close reason: {reason}.",
        f"Chief complaint: {chief}",
    ]


def _doctor_note(call: CallState, rows: list[tuple], concerns: list[tuple], reason: str):
    """The 7-section clinician markdown plus its red-flag lines."""
    profile, emergency = call.profile, reason == "emergency"
    sections: dict[str, list[str]] = {heading: [] for heading in _HEADINGS}
    sections[_HEADINGS[0]].extend(_header_lines(call, concerns, reason))
    flags = [_EMERGENCY_FLAG] if emergency else []
    if emergency:
        note = _clean(call.emergency_note)
        sections[_ALERTS].append(_EMERGENCY_LINE + (f" Mother's words: {note}." if note else ""))
        flags = [f"{_EMERGENCY_FLAG}. {note}" if note else _EMERGENCY_FLAG]
    for spec, record in rows:
        for heading, line, flag in _note_rows(spec, record, profile):
            sections[heading].append(line)
            flags.extend([flag] if flag else [])
    if not concerns and not emergency:
        sections[_ALERTS].append(_NO_ALERTS)
    cleared = [f"Clear: {s.need_en}." for s, r in rows if r.state is AnswerStatus.CLEAR]
    partial = [] if reason == "complete" else [_PARTIAL_CALL]
    sections[_NEGATIVES].extend((cleared or [_NO_NEGATIVES]) + partial)
    blocks = (
        h + "\n" + "\n".join(f"- {x}" for x in sections[h] or [_NOTHING_REPORTED])
        for h in _HEADINGS
    )
    return "\n\n".join(blocks), tuple(_unique(flags))


def _mother_lines(concerns: list[tuple], profile: CallerProfile, emergency: bool):
    """The Hindi yes list, the fitted doctor questions, and the Mother summary text."""
    scopes = [_child_labels(profile, record)[0] for _spec, record in concerns]
    yes = tuple(_unique(
        copy_hi.mother_yes_summary_hi(spec, record.mother_safe_text, hi)
        for (spec, record), hi in zip(concerns, scopes)
    ))
    asked = tuple(copy_hi.fit_mother_questions_hi(
        [copy_hi.mother_concern_question_hi(spec, record.mother_safe_text, hi)
         for (spec, record), hi in zip(concerns, scopes)],
        emergency=emergency, has_children=bool(profile.children),
    ))
    body = _YES_EMPTY_HI
    if yes:
        body = "\n".join([_YES_HEADER_HI, *(f"- {line}" for line in yes)])
    numbered = [f"{index}. {q}" for index, q in enumerate(asked, 1)]
    return yes, asked, "\n".join([body, "", _QUESTION_HEADER_HI, *numbered])


def project_summary(
    state: CallState | None, *, fallback_reason: str = "incomplete"
) -> ScreeningSummary:
    """Build every report field from one catalog pass over a finished call."""
    call = state if state is not None else CallState()
    profile = call.profile
    reason = str(call.close_reason) if call.close_reason is not None else fallback_reason
    emergency, incomplete = reason == "emergency", reason in _INCOMPLETE_REASONS
    clear_rows: dict[str, list[str]] = {}
    unasked_rows: dict[str, list[str]] = {}
    unknown, rows = [], []

    for spec in CATALOG:
        record = call.answers.get(spec.id)
        if spec.is_routable:
            label = copy_hi.question_hi_for(profile, spec)
            title = copy_hi.GROUP_HI.get(spec.group, spec.group)
            if record is None and incomplete:
                unasked_rows.setdefault(title, []).append(label)
            elif record is not None and record.state is AnswerStatus.CLEAR:
                clear_rows.setdefault(title, []).append(label)
            elif record is not None and record.state is AnswerStatus.UNKNOWN:
                unknown.append(label)
        if spec.is_reportable and record is not None:
            rows.append((spec, record))

    concerns = [(spec, record) for spec, record in rows if record.state is AnswerStatus.CONCERN]
    markdown, red_flags = _doctor_note(call, rows, concerns, reason)
    yes, asked, mother_text = _mother_lines(concerns, profile, emergency)
    has_children = bool(profile.children)
    return ScreeningSummary(
        mother_name=_clean(profile.caller_name or ""),
        children_label_hi=copy_hi.children_label_hi(profile),
        children_label_en=copy_hi.children_label_en(profile),
        has_children=has_children,
        close_reason=reason,
        emergency=emergency,
        incomplete=incomplete,
        doctor_markdown=markdown,
        mother_summary_text=mother_text,
        red_flags=red_flags,
        yes_summaries_hi=yes,
        mother_questions_hi=asked,
        page2_clear_groups=_page_groups(clear_rows),
        page2_unknown=tuple(unknown),
        page2_unasked_groups=_page_groups(unasked_rows) if incomplete else (),
    )
