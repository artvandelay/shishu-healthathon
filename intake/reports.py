"""Project one finished call into a clinician note and a patient PDF via pack templates."""

from __future__ import annotations

import asyncio, functools, os, sys, tempfile, threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

if sys.platform == "darwin": os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import weasyprint
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import db, engine
from .models import AnswerRecord, AnswerStatus, CallState, CloseReason, PackConfig, QuestionKind, QuestionSpec, Strict
from .turns import StoredTurnEvidence
from .pack import DEFAULT_PACK_ID, list_pack_ids, load_pack

IST = timezone(timedelta(hours=5, minutes=30))
SAMPLE_STARTED_AT = "2026-11-12 04:00:00"
_INCOMPLETE = frozenset(r.value for r in CloseReason if r is not CloseReason.COMPLETE)
VALID_LANGS = ("hi", "en")
_MONTHS_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# One build per (call, language): the Hindi and English sheets are separate renders and must never share a task.
_inflight: WeakKeyDictionary[Any, dict[tuple[int, str], asyncio.Task]] = WeakKeyDictionary()
_guards: WeakKeyDictionary[Any, asyncio.Lock] = WeakKeyDictionary()
_guard_lock = threading.Lock()

@functools.lru_cache(maxsize=1)
def build_info() -> dict[str, str]:
    """Deploy stamp: `env@commit`, from platform variables or a plain `.git` read (never a subprocess)."""
    commit = (os.environ.get("GIT_COMMIT") or os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "").strip()[:7].lower()
    if not commit:
        git, sha = Path(__file__).resolve().parents[1] / ".git", ""
        try:
            head = (git / "HEAD").read_text(encoding="utf-8").strip(); rel = head.partition(":")[2].strip() if head.startswith("ref:") else ""
            if rel: sha = (git / rel).read_text(encoding="utf-8") if (git / rel).is_file() else next((l.split()[0] for l in (git / "packed-refs").read_text(encoding="utf-8").splitlines() if l.endswith(f" {rel}")), "")
            sha = (sha or head).strip()[:7].lower(); commit = sha if all(c in "0123456789abcdef" for c in sha) else ""
        except OSError: commit = ""
    where = " ".join(os.environ.get(k, "") for k in ("RAILWAY_PROJECT_NAME", "RAILWAY_PUBLIC_DOMAIN", "RAILWAY_ENVIRONMENT_NAME")).lower()
    env = (os.environ.get("APP_ENV") or "").strip().lower() or ("staging" if "staging" in where else "prod" if "prod" in where else "local")
    return {"env": env, "commit": commit, "tag": f"{env}@{commit}" if commit else env}

class ProjectedReport(Strict, frozen=True):
    patient_name: str = ""
    roster_patient: str = ""
    close_reason: str = ""
    emergency: bool = False
    incomplete: bool = False
    # One non-diagnostic summary, the same facts as the PDF: what she reported, in her own words, and what was left
    # open. No red-flag block, no gate alerts, no chief complaint — this report states the call, it does not diagnose.
    report_markdown: str = ""
    yes_items: tuple[str, ...] = ()
    notes_hi: tuple[str, ...] = ()
    page2_groups: tuple[dict[str, Any], ...] = ()
    page2_unknown: tuple[str, ...] = ()
    page2_unanswered: tuple[str, ...] = ()
    page2_unasked: tuple[dict[str, Any], ...] = ()
    # The same recorded facts as the Hindi fields above, projected into English from `stem_en`/`title_clinician`.
    # Deterministic strings only — no translation call, and the caller's own words stay exactly as recorded.
    notes_en: tuple[str, ...] = ()
    yes_items_en: tuple[str, ...] = ()
    roster_patient_en: str = ""
    page2_groups_en: tuple[dict[str, Any], ...] = ()
    page2_unknown_en: tuple[str, ...] = ()
    page2_unanswered_en: tuple[str, ...] = ()
    page2_unasked_en: tuple[dict[str, Any], ...] = ()
def _clean(text: Any) -> str: return " ".join(str(text or "").split())
def _note(text: Any) -> str: return _clean(text).rstrip("।. ")
def _profile_shown(value: Any) -> bool:
    """Whether a profile slot has anything to print. `False` is a real answer ("she was never pregnant") and must
    not be dropped by a truthiness test; only a blank or absent value is silent."""
    return isinstance(value, bool) or bool(_clean(value))
def _profile_value(value: Any) -> str:
    """A profile slot as a person reads it: `True`/`False` become Yes/No, and a slug like `about_to_marry` becomes
    "about to marry". The value is hers, not this module's, so nothing else about it is rewritten."""
    if isinstance(value, bool): return "Yes" if value else "No"
    return _clean(value).replace("_", " ")
def _unique(items: Sequence[Any]) -> list[str]: return list(dict.fromkeys(c for c in map(_clean, items) if c))
def _grouped(pairs: Mapping[str, list[Any]]) -> tuple[dict[str, Any], ...]:
    """Group detail rows for page two. A row is either a plain string (a clear gate, printed with the group default
    answer) or a `{question, answer}` pair (a real answer, where a blanket negative would be a lie)."""
    return tuple({"title": k, "items": tuple(_clean(i) if isinstance(i, str) else {"question": _clean(i["question"]), "answer": _clean(i["answer"])} for i in v)} for k, v in pairs.items())

def _roster_en(st: CallState, item_ids: Sequence[str] | None = None) -> str:
    """Clean patient-facing English roster (never leaks internal `item_1` IDs)."""
    chosen = [r for r in st.roster if not item_ids or r.id in item_ids]
    parts = []
    for r in chosen:
        if r.age_years is not None:
            age = (f"{mo} mo" if (mo := round(r.age_years * 12)) > 0 else "newborn") if r.age_years < 1 else f"{int(r.age_years) if r.age_years.is_integer() else r.age_years:g} yr"
        else:
            age = ""
        sex = {"male": "boy", "female": "girl"}.get(r.tag or "", "child")
        bits = [b for b in (_clean(r.name), age, sex) if b]
        parts.append(" ".join(bits))
    return ", ".join(parts)

def _twins(labels: Mapping[str, str]) -> dict[str, list[str]]:
    """Patient labels that cover more than one child, in roster order: same-age twins read out identically."""
    groups: dict[str, list[str]] = {}
    for cid, label in labels.items(): groups.setdefault(label, []).append(cid)
    return {label: ids for label, ids in groups.items() if len(ids) > 1}
def _scope(pack: PackConfig, st: CallState, rec: AnswerRecord, *, lang: str = "hi") -> str:
    """Which child the answer is about, in the caller's own words. Empty when there is only one child.

    Two twins of the same age and sex share one patient label ("2 साल बेटा"), so printing it bare would put an
    answer about the second boy on a row that reads exactly like the first. When the label repeats across the
    roster it is numbered with the pack's own `which_item.ordinals` — the same words the call speaks — so the
    report never collapses two children into one. An answer that covers both twins (`item_ids=["item_1", "item_2"]`)
    numbers each child on its own; joining the shared bare label twice printed "(2 साल बेटा · 2 साल बेटा)"."""
    if not rec.item_ids or len(st.roster) <= 1: return ""
    def label(ids: Sequence[str] | None) -> str:
        return _roster_en(st, ids) if lang == "en" else engine.roster_label_patient(pack, st, ids)
    words = [str(w) for w in ((pack.prompts.get("roster_format") or {}).get("which_item") or {}).get("ordinals") or ()]
    twins = _twins({r.id: label([r.id]) for r in st.roster})
    def numbered(cid: str) -> str:
        """One child's label, numbered by its own place in the twin group when its bare label repeats."""
        bare, group = label([cid]), twins.get(label([cid]))
        if not group: return bare
        idx = group.index(cid)
        return f"{words[idx]} {bare}" if lang != "en" and idx < len(words) else f"{bare} #{idx + 1}"
    known = {r.id for r in st.roster}
    ids = [cid for cid in dict.fromkeys(rec.item_ids) if cid in known]
    if not ids: return label(rec.item_ids)
    if len(ids) == 1: return numbered(ids[0])
    joiner = str((pack.prompts.get("roster_format") or {}).get("join", " · "))
    return joiner.join(numbered(cid) for cid in ids)
def _template(pack: PackConfig, name: str):
    return Environment(loader=FileSystemLoader(str(pack.root / "templates")), undefined=StrictUndefined, autoescape=False).get_template(name)
def _pick(cfg: dict[str, Any], *keys: str, fallback: str = "") -> str:
    return next((str(cfg[k]) for k in keys if cfg.get(k)), fallback)
def _label_en(spec: QuestionSpec) -> str:
    """The English question label: the pack's own `stem_en` when it has one, else the clinical `need`."""
    return _clean(getattr(spec, "stem_en", "") or spec.need)
def _title_en(pack: PackConfig, group: str) -> str:
    return str((pack.groups.get(group) or {}).get("title_clinician") or group)
def _attributed(cfg: dict[str, Any], said: str, *, lang: str) -> str:
    """The one way her own words are quoted on any surface: the pack's own label, her transcript verbatim.
    Both a reported concern and a retained aside go through here, so the doctor reads the same thing twice."""
    label = _pick(cfg, f"caller_words_{lang}")
    return f"{label}: “{said}”" if label else f"“{said}”"
def _concern_copy(cfg: dict[str, Any], spec: QuestionSpec, rec: AnswerRecord, scope: str, *, lang: str = "hi") -> str:
    """One reported concern as the single factual line it prints on page one: the question she said yes to, then her
    own words under the pack's label. No advice, no question, no filler, and her words appear exactly once.
    They come from `verbatim_text`, never `patient_safe_text` — a redacted concern has an empty `patient_safe`, and
    reading it would print her answer as a bare question stem with her words gone."""
    stem = _label_en(spec) if lang == "en" else _clean(spec.stem or spec.say)
    said, choice = _note(rec.verbatim_text), _note(rec.choice_label)
    head = f"{stem} ({scope})" if scope else stem
    if not said and choice and choice not in stem: head = f"{head} ({choice})"
    if not said: return f"{head}."
    if spec.kind is QuestionKind.OPEN_CAPTURE: return _attributed(cfg, said, lang=lang)
    return f"{head} — {_attributed(cfg, said, lang=lang)}"
def _patient_copy(pack: PackConfig, st: CallState, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]]) -> list[str]:
    """Only the recorded yes-lines. The old "questions to ask" list is gone: this sheet states facts, never advice."""
    cfg = pack.report_cfg
    return [_concern_copy(cfg, q, rec, _scope(pack, st, rec)) for q, rec in concerns]
def _patient_copy_en(pack: PackConfig, st: CallState, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]]) -> list[str]:
    """The same concerns in English, built from `stem_en` — deterministic, and the caller's words kept verbatim."""
    return [_concern_copy(pack.report_cfg, q, rec, _scope(pack, st, rec, lang="en"), lang="en") for q, rec in concerns]
def _md_rows(items: Sequence[Any]) -> list[str]:
    """One markdown bullet per row: a plain string is a question, a pair prints the answer she actually chose."""
    return [f"- {_clean(i['question'])} — {_clean(i['answer'])}" if isinstance(i, Mapping) else f"- {_clean(i)}" for i in items]
def _md_block(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    """One `#### group` heading plus its bullets, for whichever response state the caller passed in."""
    return [line for group in groups for line in [f"#### {_clean(group['title'])}", *_md_rows(group["items"])]]
def _only(rows: Mapping[str, list[tuple[AnswerStatus, Any]]], *states: AnswerStatus) -> dict[str, list[Any]]:
    """The detail rows of the given response states, keeping their group order. Page two wants both CLEAR and
    ANSWERED; the markdown prints each state under its own heading, so the two never share one."""
    return {title: [item for state, item in items if state in states] for title, items in rows.items()
            if any(state in states for state, _ in items)}
def _markdown(cfg: dict[str, Any], *, profile: Sequence[str], reason: str, emergency: bool, emergency_note: str, incomplete: bool,
              yes: Sequence[str], notes: Sequence[str], answered: Sequence[Mapping[str, Any]],
              clear: Sequence[Mapping[str, Any]], unknown: Sequence[str],
              unanswered: Sequence[str], unasked: Sequence[Mapping[str, Any]]) -> str:
    """One plain, non-diagnostic Markdown summary of the same facts as the PDF: who the call was about, how it
    ended, what she reported in her own words, what was never covered. No red flags, no gate alerts, no chief
    complaint — nothing here is a clinical judgement, only a record of the call. Every word of hers is quoted
    through `_attributed`, so the label and the quotes come from the pack, never from this module.

    `ANSWERED` (she picked an option: "about to marry", "3 to 7 days") is its own block. Filing it under
    "### Clear / No" printed a chosen answer directly beneath a glossary saying Clear means "did not happen".

    Each section is one block and the blocks are joined with a blank line. Building one flat list with `""`
    separators and then dropping every empty line with `if line` glued each `### heading` onto the previous
    bullet, so the dashboard rendered the report as one run-on paragraph."""
    # The close reason prints as words ("caller stop"), never the stored enum slug ("caller_stop").
    why = _clean(str(reason)).replace("_", " ")
    profile_block = ["### Profile", *(f"- {line}" for line in profile), f"- Close reason: {why}."]
    if emergency and (line := _pick(cfg, "emergency_line_en")):
        said = _note(emergency_note)
        profile_block += [f"- {line}", *([f"- {_attributed(cfg, said, lang='hi')}"] if said else [])]
    blocks = [profile_block, ['### Reported "Yes"', *([f"- {x}" for x in yes] or ["- None reported on this call."])]]
    if notes and (heading := _pick(cfg, "notes_header_hi")):
        blocks.append([f"### {heading}", *(f"- {x}" for x in notes)])
    if answered: blocks.append(["### Answered", *_md_block(answered)])
    blocks.append(["### Clear / No", *(_md_block(clear) or ["- None recorded."])])
    if unknown: blocks.append(["### Don't know", *(f"- {x}" for x in unknown)])
    if unanswered: blocks.append(["### Not answered", *(f"- {x}" for x in unanswered)])
    if incomplete:
        blocks.append(["### Not asked", f"- {_pick(cfg, 'partial_call_en', fallback='Call ended before every applicable question was asked.')}", *_md_block(unasked)])
    blocks.append(["### Response state glossary",
                   "- Reported yes — the caller said this happened or was true.",
                   "- Clear / no — the caller said this did not happen.",
                   "- Don't know — the caller said she did not know, or the child was not tested.",
                   "- Not answered — the question was asked but no answer was recorded on the call.",
                   "- Not asked — the call ended before this question was asked."])
    return "# Call report\n\n" + "\n\n".join("\n".join(block) for block in blocks) + "\n"
def project_summary(pack: PackConfig, state: CallState | None, observations: Sequence[StoredTurnEvidence] = ()) -> ProjectedReport:
    st, cfg = state or CallState(), pack.report_cfg
    reason = str(st.close_reason or CloseReason.COMPLETE)
    emergency, incomplete = reason == CloseReason.EMERGENCY, reason in _INCOMPLETE
    unasked, unknown, unanswered, concerns = {}, [], [], []
    unasked_en, unknown_en, unanswered_en = {}, [], []
    # Detail rows in question order, tagged with the state that produced them, so the markdown can keep ANSWERED
    # apart from CLEAR while page two still prints both in one list of `#### group` sections.
    rows: dict[str, list[tuple[AnswerStatus, Any]]] = {}
    rows_en: dict[str, list[tuple[AnswerStatus, Any]]] = {}
    for q in pack.questions:
        if not q.is_routable: continue
        rec, label, title = st.answers.get(q.id), _clean(q.stem or q.say), str((pack.groups.get(q.group) or {}).get("title_patient") or q.group)
        label_en, title_en = _label_en(q), _title_en(pack, q.group)
        if rec is None:
            if incomplete:
                unasked.setdefault(title, []).append(label)
                unasked_en.setdefault(title_en, []).append(label_en)
            continue
        if rec.state is AnswerStatus.NOT_APPLICABLE: continue
        if rec.state is AnswerStatus.UNKNOWN:
            unknown.append(label)
            unknown_en.append(label_en)
        elif rec.state is AnswerStatus.UNANSWERED:
            unanswered.append(label)
            unanswered_en.append(label_en)
        elif rec.state is AnswerStatus.CLEAR:
            rows.setdefault(title, []).append((rec.state, label))
            rows_en.setdefault(title_en, []).append((rec.state, label_en))
        elif rec.state is AnswerStatus.ANSWERED:
            # A chosen option ("about to marry", "3 to 7 days") is not a denial, so it is never printed under
            # "### Clear / No". It keeps its `{question, answer}` pair for page two, where the template prints her
            # own answer in the row, and the markdown gives it the separate "### Answered" heading.
            chosen = _clean(rec.choice_label) or _clean(rec.patient_safe_text)
            if chosen:
                pair = {"question": label, "answer": chosen}
                rows.setdefault(title, []).append((rec.state, pair))
                rows_en.setdefault(title_en, []).append((rec.state, {"question": label_en, "answer": chosen}))
        if q.is_reportable and rec.state is AnswerStatus.CONCERN:
            concerns.append((q, rec))
    yes, yes_en = _patient_copy(pack, st, concerns), _patient_copy_en(pack, st, concerns)
    # Retained clinical asides are quoted exactly like a reported concern: her transcript, the pack's own label.
    notes_hi = tuple(_unique(_attributed(cfg, said, lang="hi") for obs in observations if (said := _note(obs.transcript))))
    notes_en = tuple(_unique(_attributed(cfg, said, lang="en") for obs in observations if (said := _note(obs.transcript))))
    roster_patient = engine.roster_label_patient(pack, st)
    patient_name = next((_clean(st.profile[s]) for s in pack.profile_slots if st.profile.get(s)), "")
    slots = [*pack.profile_slots, *(k for k in st.profile if k not in pack.profile_slots)]
    profile = [f"{s.replace('_', ' ').capitalize()}: {_profile_value(v)}." for s in slots if _profile_shown(v := st.profile.get(s))]
    if pack.roster_cfg and roster_patient: profile.append(f"{str(pack.roster_cfg.get('key') or 'roster').capitalize()}: {roster_patient}.")
    groups, groups_en = _grouped(_only(rows, AnswerStatus.CLEAR, AnswerStatus.ANSWERED)), _grouped(_only(rows_en, AnswerStatus.CLEAR, AnswerStatus.ANSWERED))
    answered_groups = _grouped(_only(rows, AnswerStatus.ANSWERED))
    unasked_groups, unasked_groups_en = (_grouped(unasked), _grouped(unasked_en)) if incomplete else ((), ())
    unknown_items, unanswered_items = _unique(unknown), _unique(unanswered)
    md = _markdown(cfg, profile=profile, reason=reason, emergency=emergency, emergency_note=st.emergency_note,
                   incomplete=incomplete, yes=yes, notes=notes_hi, answered=answered_groups,
                   clear=_grouped(_only(rows, AnswerStatus.CLEAR)),
                   unknown=unknown_items, unanswered=unanswered_items, unasked=unasked_groups)
    return ProjectedReport(
        patient_name=patient_name, roster_patient=roster_patient, close_reason=reason,
        emergency=emergency, incomplete=incomplete, report_markdown=md, yes_items=tuple(yes), notes_hi=notes_hi,
        page2_groups=groups, page2_unknown=unknown_items, page2_unanswered=unanswered_items,
        page2_unasked=unasked_groups, notes_en=notes_en, yes_items_en=tuple(yes_en), roster_patient_en=_roster_en(st),
        page2_groups_en=groups_en, page2_unknown_en=tuple(_unique(unknown_en)), page2_unanswered_en=tuple(_unique(unanswered_en)),
        page2_unasked_en=unasked_groups_en)
def _ist(raw: Any) -> datetime | None:
    try: return datetime.strptime(str(raw)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone(IST)
    except ValueError: return None
def render_patient_html(pack: PackConfig, call: Mapping[str, Any], proj: ProjectedReport | None = None, *, lang: str = "hi") -> str:
    lang = "en" if str(lang).lower() == "en" else "hi"
    rep = proj or project_summary(pack, db.call_state_from_row(call), () if (cid := int(call.get("id") or 0)) <= 0 else db.list_retained_observations(cid))
    when, cfg = _ist(call.get("started_at")), pack.report_cfg
    months = list(cfg.get("months_hi") or ("",))
    m, s = divmod(max(0, int(call.get("duration_sec") or 0)), 60)
    mu, su = cfg.get("minute_unit", "min"), cfg.get("second_unit", "sec")
    dur = f"{m} {mu} {s} {su}" if (m and s) else (f"{m} {mu}" if m else f"{s} {su}")
    date = f"{when.day} {months[when.month]} {when.year}" if (when and 0 < when.month < len(months)) else ""
    clock = f"{when.hour % 12 or 12}:{when.minute:02d} {cfg.get('am_label', 'AM') if when.hour < 12 else cfg.get('pm_label', 'PM')}" if when else ""
    who = " · ".join(b for b in (rep.patient_name, rep.roster_patient) if b) or str((pack.prompts.get("roster_format") or {}).get("patient_default_who") or "")
    badge = str(cfg.get("incomplete_badge", "incomplete") if rep.incomplete else cfg.get("complete_badge", "complete"))
    yes_items, page2_groups, page2_unknown, page2_unanswered, page2_unasked = rep.yes_items, rep.page2_groups, rep.page2_unknown, rep.page2_unanswered, rep.page2_unasked
    if lang == "en":
        # The English sheet is projected from the same records, deterministically: `stem_en` for labels,
        # the group's clinician title for headings, and the caller's own words kept exactly as recorded.
        date = f"{when.day} {_MONTHS_EN[when.month - 1]} {when.year}" if when else ""
        clock = f"{when.hour % 12 or 12}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}" if when else ""
        dur = f"{m} min {s} sec" if (m and s) else (f"{m} min" if m else f"{s} sec")
        who = " · ".join(b for b in (rep.patient_name, rep.roster_patient_en) if b) or "Caller"
        badge = "Incomplete" if rep.incomplete else "Complete"
        yes_items, page2_groups, page2_unknown, page2_unanswered, page2_unasked = (
            rep.yes_items_en, rep.page2_groups_en, rep.page2_unknown_en, rep.page2_unanswered_en, rep.page2_unasked_en)
    pref = str(pack.meta.get("pdf_prefix") or pack.id.upper())
    build, cid, tok = build_info()["tag"], int(call.get("id") or 0), str(call.get("share_token") or "")[:8]; ref = f"{pref}-{cid}" if cid else pref
    meta = " · ".join(b for b in (ref, date, clock, dur, badge) if b)
    assets = pack.meta.get("assets") or {}
    subject = " ".join(x for x in (f"pack={pack.id}", f"call={cid}" if cid else "", f"token={tok}" if tok else "", f"build={build}", f"lang={lang}") if x)
    css = (pack.root / "templates" / "pdf.css").read_text(encoding="utf-8").replace("__FONT__", (pack.root / "assets" / str(assets.get("font") or "")).as_uri())
    css = css.replace("__FOOTER__", f"{ref} · {build}".replace("\\", "\\\\").replace('"', '\\"'))
    return _template(pack, "patient.html.j2").render(
        lang=lang, css=css, doc_title=ref, pdf_subject=subject, logo_uri=(pack.root / "assets" / str(assets.get("logo") or "")).as_uri(), call_id=cid, who=who, meta=meta,
        emergency=rep.emergency, incomplete=rep.incomplete, yes_items=list(yes_items), page2_groups=list(page2_groups),
        page2_unknown=list(page2_unknown), page2_unanswered=list(page2_unanswered), page2_unasked=list(page2_unasked),
        observations=list(rep.notes_en if lang == "en" else rep.notes_hi),
        observations_heading=str(cfg.get("notes_header_en" if lang == "en" else "notes_header_hi") or ""))
def render_patient_pdf(pack: PackConfig, call: Mapping[str, Any], proj: ProjectedReport | None = None, *, lang: str = "hi") -> bytes:
    return weasyprint.HTML(string=render_patient_html(pack, call, proj, lang=lang), base_url=str(pack.root / "assets")).write_pdf()
def _lang_of(lang: str) -> str:
    """Normalize to a stored language; anything unrecognised is the Hindi sheet the caller is reading."""
    value = str(lang).lower()
    return value if value in VALID_LANGS else "hi"
def patient_pdf_path(call_id: int, lang: str = "hi") -> Path:
    return Path(db.DATA_DIR) / "pdfs" / f"call-{int(call_id)}-{_lang_of(lang)}.pdf"

def pdf_filename(pack: PackConfig, call: Mapping[str, Any], lang: str = "hi") -> str:
    when, pref, cid = _ist(call.get("started_at")), str(pack.meta.get("pdf_prefix") or pack.id.upper()), int(call.get("id") or 0)
    suffix = f"-c{cid}" if cid > 0 else ""
    stem = f"{pref}{suffix}" if when is None else f"{pref}-{when:%Y-%m-%d-%H%M%S}{suffix}"
    return f"{stem}-{_lang_of(lang)}.pdf"
def build_patient_pdf_sync(call_id: int, lang: str = "hi") -> tuple[bytes, str]:
    lang = _lang_of(lang)
    if (call := db.get_call(call_id)) is None: raise LookupError(f"Call {call_id} not found")
    pack = load_pack(str(call.get("pack_id") or DEFAULT_PACK_ID))
    proj = project_summary(pack, db.call_state_from_row(call), db.list_retained_observations(call_id))
    db.save_report_text(call_id, proj.report_markdown, proj.patient_name, proj.roster_patient)
    if not (pdf := render_patient_pdf(pack, call, proj, lang=lang)).startswith(b"%PDF"): raise ValueError("Renderer did not return a PDF")
    target = patient_pdf_path(call_id, lang)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as h:
            h.write(pdf)
            h.flush()
            os.fsync(h.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return pdf, pdf_filename(pack, call, lang)
def ready_pdf(call_id: int, lang: str = "hi") -> tuple[bytes, str] | None:
    lang = _lang_of(lang)
    path = patient_pdf_path(call_id, lang)
    if not path.is_file() or len(data := path.read_bytes()) <= 4 or not data.startswith(b"%PDF"): return None
    call = db.get_call(call_id)
    return (data, path.name) if call is None else (data, pdf_filename(load_pack(str(call.get("pack_id") or DEFAULT_PACK_ID)), call, lang))
def is_pdf_inflight(call_id: int, lang: str = "hi") -> bool:
    """True when a background build for `(call_id, lang)` is currently running on this loop."""
    try: loop = asyncio.get_running_loop()
    except RuntimeError: return False
    with _guard_lock: bucket = _inflight.get(loop)
    task = bucket.get((int(call_id), _lang_of(lang))) if bucket else None
    return task is not None and not task.done()
async def ensure_patient_pdf(call_id: int, lang: str = "hi") -> tuple[bytes, str]:
    """Build (or join the in-flight build of) one language's sheet. `hi` and `en` are separate keys, so the two
    never coalesce into one task or overwrite each other's file."""
    lang, loop = _lang_of(lang), asyncio.get_running_loop()
    key = (int(call_id), lang)
    # Hindi keeps the one-argument call it has always made; English names its language.
    work = functools.partial(build_patient_pdf_sync, call_id) if lang == "hi" else functools.partial(build_patient_pdf_sync, call_id, lang)
    with _guard_lock: guard, bucket = _guards.setdefault(loop, asyncio.Lock()), _inflight.setdefault(loop, {})
    async with guard:
        if (task := bucket.get(key)) is None or task.done():
            task = bucket[key] = asyncio.create_task(asyncio.to_thread(work))
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
    try: return await asyncio.shield(task)
    finally:
        async with guard:
            if bucket.get(key) is task: bucket.pop(key, None)
def replay_seed_scenario(pack: PackConfig, scenario: Mapping[str, Any]):
    from scripts.seed_replay import replay_seed_scenario as _replay
    return _replay(pack, scenario)
def write_sample_pdfs(pack: PackConfig, folder: Path) -> list[Path]:
    from scripts.seed_replay import write_sample_pdfs as _write
    return _write(pack, folder)
def ensure_seeded(pack: PackConfig | None = None) -> None:
    for cur in [pack] if pack else [load_pack(pid) for pid in list_pack_ids()]:
        for sc in (() if (db.list_calls(cur.id) or not cur.seed_scenarios) else cur.seed_scenarios):
            st, turns = replay_seed_scenario(cur, sc)
            cid = db.create_call(pack_id=cur.id, caller_name=str(sc.get("caller_name") or ""))
            for r, txt, q in turns: db.append_transcript(cid, r, txt, qid=q)
            db.save_cursor_state(cid, cur, st)
            db.complete_call(cid, int(sc.get("duration_sec") or 0))
            proj = project_summary(cur, st)
            db.save_report_text(cid, proj.report_markdown, proj.patient_name, proj.roster_patient)
