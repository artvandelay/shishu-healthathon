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
SAMPLE_STARTED_AT, _SPECIAL = "2026-11-12 04:00:00", ("patient_profile", "alerts", "negatives")
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
    roster_clinician: str = ""
    close_reason: str = ""
    emergency: bool = False
    incomplete: bool = False
    doctor_markdown: str = ""
    patient_summary_text: str = ""
    red_flags: tuple[str, ...] = ()
    yes_items: tuple[str, ...] = ()
    patient_questions: tuple[str, ...] = ()
    page2_groups: tuple[dict[str, Any], ...] = ()
    page2_unknown: tuple[str, ...] = ()
    page2_unanswered: tuple[str, ...] = ()
    page2_unasked: tuple[dict[str, Any], ...] = ()
    observations: tuple[str, ...] = ()
    # The same recorded facts as the Hindi fields above, projected into English from `stem_en`/`title_clinician`.
    # Deterministic strings only — no translation call, and the caller's own words stay exactly as recorded.
    yes_items_en: tuple[str, ...] = ()
    roster_patient_en: str = ""
    page2_groups_en: tuple[dict[str, Any], ...] = ()
    page2_unknown_en: tuple[str, ...] = ()
    page2_unanswered_en: tuple[str, ...] = ()
    page2_unasked_en: tuple[dict[str, Any], ...] = ()
def _clean(text: Any) -> str: return " ".join(str(text or "").split())
def _note(text: Any) -> str: return _clean(text).rstrip("।. ")
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

def _scope(pack: PackConfig, st: CallState, rec: AnswerRecord, *, clinician: bool) -> str:
    if not rec.item_ids or len(st.roster) <= 1: return ""
    return engine.roster_label_clinician(st, rec.item_ids) if clinician else engine.roster_label_patient(pack, st, rec.item_ids)
def _section_target(pack: PackConfig, spec: QuestionSpec, known: set[str]) -> str:
    """The clinician heading an answer belongs under: the spec's own override when declared, else its group's title."""
    if spec.clinician_group and f"### {spec.clinician_group}" in known: return f"### {spec.clinician_group}"
    return f"### {(pack.groups.get(spec.group) or {}).get('title_clinician') or spec.group}"
def _chief_complaint(pack: PackConfig, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]], emergency: bool, note: str = "") -> str:
    """The one line the doctor reads first. An active emergency leads it — a concern recorded earlier in the call is
    history the doctor still needs (it stays in the red-flag block), never the complaint that ended the call."""
    if emergency:
        line = _clean(pack.report_cfg.get("emergency_line_en")) or "Active emergency. Caller was told to go to the nearest hospital or call 108."
        line = line if line.endswith((".", "।")) else f"{line}."
        said = _note(note)
        return f"{line} Mother's words: {said}" if said else line
    if concerns: return _clean(concerns[0][0].need)
    return _clean(pack.meta.get("chief_complaint")) or "not reported"
def _template(pack: PackConfig, name: str):
    return Environment(loader=FileSystemLoader(str(pack.root / "templates")), undefined=StrictUndefined, autoescape=False).get_template(name)
def _pick(cfg: dict[str, Any], *keys: str, fallback: str = "") -> str:
    return next((str(cfg[k]) for k in keys if cfg.get(k)), fallback)
def _label_en(spec: QuestionSpec) -> str:
    """The English question label: the pack's own `stem_en` when it has one, else the clinical `need`."""
    return _clean(getattr(spec, "stem_en", "") or spec.need)
def _title_en(pack: PackConfig, group: str) -> str:
    return str((pack.groups.get(group) or {}).get("title_clinician") or group)
def _concern_copy(cfg: dict[str, Any], spec: QuestionSpec, rec: AnswerRecord, scope: str) -> str:
    """One reported concern as the single factual line it prints on page one. No advice, no question, no filler:
    the sheet only restates what the mother said."""
    stem, choice, note = _clean(spec.stem or spec.say), _note(rec.choice_label), _note(rec.patient_safe_text)
    head = f"{stem} ({scope})" if scope else stem
    if not note and choice and choice not in stem: head = f"{head} ({choice})"
    if spec.kind is QuestionKind.OPEN_CAPTURE:
        return _pick(cfg, "yes_summary_open_hi" if note else "yes_summary_open_empty_hi", fallback="{note}").format(note=note)
    return _pick(cfg, "yes_summary_stem_hi" if note else "yes_summary_stem_no_note_hi", fallback="{head}").format(head=head, note=note)
def _patient_copy(pack: PackConfig, st: CallState, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]]) -> list[str]:
    """Only the recorded yes-lines. The old "questions to ask" list is gone: this sheet states facts, never advice."""
    cfg = pack.report_cfg
    return [_concern_copy(cfg, q, rec, _scope(pack, st, rec, clinician=False)) for q, rec in concerns]
def _patient_copy_en(pack: PackConfig, st: CallState, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]]) -> list[str]:
    """The same concerns in English, built from `stem_en` — deterministic, and the caller's words kept verbatim."""
    lines = []
    for q, rec in concerns:
        label, scope, note = _label_en(q), (_roster_en(st, rec.item_ids) if (rec.item_ids and len(st.roster) > 1) else ""), _note(rec.patient_safe_text)
        head = f"{label} ({scope})" if (label and scope) else label
        lines.append(f"{head} — {note}." if note else f"{head}.")
    return lines
def project_summary(pack: PackConfig, state: CallState | None, observations: Sequence[StoredTurnEvidence] = ()) -> ProjectedReport:
    st, cfg = state or CallState(), pack.report_cfg
    reason = str(st.close_reason or CloseReason.COMPLETE)
    emergency, incomplete = reason == CloseReason.EMERGENCY, reason in _INCOMPLETE
    headings = {str(k): str(v) for k, v in (cfg.get("clinician_headings") or {}).items()}
    known, sections = set(headings.values()), {h: [] for k, h in headings.items() if k not in _SPECIAL}
    alerts, negatives, flags, profile_extra, clear, unasked, unknown, unanswered, concerns = [], [], [], [], {}, {}, [], [], []
    clear_en, unasked_en, unknown_en, unanswered_en = {}, {}, [], []
    specials = {headings.get("alerts"): alerts, headings.get("negatives"): negatives, headings.get("patient_profile"): profile_extra}
    if emergency:
        said, line, flag = _note(st.emergency_note), str(cfg.get("emergency_line_en") or ""), str(cfg.get("emergency_flag_en") or "")
        alerts.append(f"{line} Mother's words: {said}." if said else line)
        flags.append(f"{flag}. {said}" if said else flag)
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
            clear.setdefault(title, []).append(label)
            clear_en.setdefault(title_en, []).append(label_en)
        elif rec.state is AnswerStatus.ANSWERED:
            chosen = _clean(rec.choice_label) or _clean(rec.patient_safe_text)
            if chosen:
                clear.setdefault(title, []).append({"question": label, "answer": chosen})
                clear_en.setdefault(title_en, []).append({"question": label_en, "answer": chosen})
        if not q.is_reportable: continue
        child, said = (f" [{sc}]" if (sc := _scope(pack, st, rec, clinician=True)) else ""), (_note(rec.verbatim_text) or _note(rec.choice_label))
        target = _section_target(pack, q, known)
        lines = specials[target] if target in specials else sections.setdefault(target, [])
        if rec.state is AnswerStatus.CONCERN:
            concerns.append((q, rec))
            head = f"{q.tag} {q.id}: {q.need}{child}".strip()
            alerts.append(f"{head}. Mother's words: {said}." if said else head)
            flags.append(f"{head}. {said}" if said else head)
            lines.append(f"Concern. {q.need}{child}. {said or 'reported'}.")
        elif rec.state is AnswerStatus.ANSWERED: lines.append(f"Answered{f' ({said})' if said else ''}. {q.need}.")
        elif rec.state is AnswerStatus.UNANSWERED: lines.append(f"Not answered on call (confirm with patient). {q.need}.")
        else:
            lines.append(f"{rec.state.value.title()}. {q.need}.")
            if rec.state is AnswerStatus.CLEAR: negatives.append(f"Clear: {q.need}.")
    roster = engine.roster_label_clinician(st)
    chief = _chief_complaint(pack, concerns, emergency, st.emergency_note)
    prof = [f"{s.replace('_', ' ').capitalize()}: {_clean(st.profile[s])}." for s in pack.profile_slots if st.profile.get(s)]
    prof += [f"{s.replace('_', ' ').capitalize()}: {_clean(v)}." for s, v in st.profile.items() if s not in pack.profile_slots]
    if pack.roster_cfg: prof.append(f"Children: {roster or 'none reported'}.")
    obs_lines = tuple(_unique(f"Unlisted clinical aside during {spec.id} ({spec.need}): {said}." if (spec := pack.by_id.get(obs.snapshot_step.split(":", 1)[0]))
                      else f"Unlisted clinical aside ({obs.snapshot_step}): {said}." for obs in observations if (said := _note(obs.transcript))))
    md = _template(pack, "clinician.md.j2").render(profile_lines=[*prof, f"Close reason: {reason}.", f"Chief complaint: {chief}.", *profile_extra], alerts=alerts,
                                                   sections=[{"heading": h, "lines": v} for h, v in sections.items()], negatives=negatives, incomplete=incomplete, red_flags=flags, observations=obs_lines)
    yes = _patient_copy(pack, st, concerns)
    yes_en = _patient_copy_en(pack, st, concerns)
    body = "\n".join([str(cfg.get("yes_header_hi") or ""), *(f"- {x}" for x in yes)]) if yes else str(cfg.get("yes_empty_hi") or "")
    return ProjectedReport(
        patient_name=next((_clean(st.profile[s]) for s in pack.profile_slots if st.profile.get(s)), ""), roster_patient=engine.roster_label_patient(pack, st),
        roster_clinician=roster, close_reason=reason, emergency=emergency, incomplete=incomplete, doctor_markdown=md,
        patient_summary_text=body,
        red_flags=tuple(_unique(flags)), yes_items=tuple(yes), patient_questions=(), page2_groups=_grouped(clear),
        page2_unknown=tuple(_unique(unknown)), page2_unanswered=tuple(_unique(unanswered)),
        page2_unasked=_grouped(unasked) if incomplete else (), observations=obs_lines,
        yes_items_en=tuple(yes_en), roster_patient_en=_roster_en(st), page2_groups_en=_grouped(clear_en),
        page2_unknown_en=tuple(_unique(unknown_en)), page2_unanswered_en=tuple(_unique(unanswered_en)),
        page2_unasked_en=_grouped(unasked_en) if incomplete else ())
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
        page2_unknown=list(page2_unknown), page2_unanswered=list(page2_unanswered), page2_unasked=list(page2_unasked))
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
    db.save_report_text(call_id, proj.doctor_markdown, proj.patient_summary_text, list(proj.red_flags), proj.patient_name, proj.roster_patient)
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
            db.save_report_text(cid, proj.doctor_markdown, proj.patient_summary_text, list(proj.red_flags), proj.patient_name, proj.roster_patient)
