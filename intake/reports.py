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
from .pack import DEFAULT_PACK_ID, list_pack_ids, load_pack

IST = timezone(timedelta(hours=5, minutes=30))
SAMPLE_STARTED_AT, _SPECIAL = "2026-11-12 04:00:00", ("patient_profile", "alerts", "negatives")
_INCOMPLETE = frozenset(r.value for r in CloseReason if r is not CloseReason.COMPLETE)
_inflight: WeakKeyDictionary[Any, dict[int, asyncio.Task]] = WeakKeyDictionary()
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
    page2_unasked: tuple[dict[str, Any], ...] = ()
def _clean(text: Any) -> str: return " ".join(str(text or "").split())
def _note(text: Any) -> str: return _clean(text).rstrip("।. ")
def _unique(items: Sequence[Any]) -> list[str]: return list(dict.fromkeys(c for c in map(_clean, items) if c))
def _grouped(pairs: Mapping[str, list[Any]]) -> tuple[dict[str, Any], ...]:
    """Group detail rows for page two. A row is either a plain string (a clear gate, printed with the group default
    answer) or a `{question, answer}` pair (a real answer, where a blanket negative would be a lie)."""
    return tuple({"title": k, "items": tuple(_clean(i) if isinstance(i, str) else {"question": _clean(i["question"]), "answer": _clean(i["answer"])} for i in v)} for k, v in pairs.items())

def _scope(pack: PackConfig, st: CallState, rec: AnswerRecord, *, clinician: bool) -> str:
    if not rec.item_ids or len(st.roster) <= 1: return ""
    return engine.roster_label_clinician(st, rec.item_ids) if clinician else engine.roster_label_patient(pack, st, rec.item_ids)
def _section_target(pack: PackConfig, spec: QuestionSpec, known: set[str]) -> str:
    """The clinician heading an answer belongs under: the spec's own override when declared, else its group's title."""
    if spec.clinician_group and f"### {spec.clinician_group}" in known: return f"### {spec.clinician_group}"
    return f"### {(pack.groups.get(spec.group) or {}).get('title_clinician') or spec.group}"
def _chief_complaint(pack: PackConfig, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]], emergency: bool) -> str:
    if concerns: return _clean(concerns[0][0].need)
    return _clean(pack.meta.get("chief_complaint")) or ("Active emergency reported during call." if emergency else "not reported")
def _template(pack: PackConfig, name: str):
    return Environment(loader=FileSystemLoader(str(pack.root / "templates")), undefined=StrictUndefined, autoescape=False).get_template(name)
def _pick(cfg: dict[str, Any], *keys: str, fallback: str = "") -> str:
    return next((str(cfg[k]) for k in keys if cfg.get(k)), fallback)
def _concern_copy(cfg: dict[str, Any], spec: QuestionSpec, rec: AnswerRecord, scope: str) -> tuple[str, str]:
    """One reported concern as the two patient-facing lines it produces: the yes-summary and the question to ask."""
    stem, choice, note = _clean(spec.stem or spec.say), _note(rec.choice_label), _note(rec.patient_safe_text)
    head = f"{stem} ({scope})" if scope else stem
    if not note and choice and choice not in stem: head = f"{head} ({choice})"
    prefix = str(cfg["concern_question_scope_prefix_hi"]).format(child_scope=scope) if (scope and scope not in note and cfg.get("concern_question_scope_prefix_hi")) else ""
    if spec.kind is QuestionKind.OPEN_CAPTURE:
        yes = _pick(cfg, "yes_summary_open_hi" if note else "yes_summary_open_empty_hi", fallback="{note}")
        asked = _pick(cfg, "concern_question_with_note_hi" if note else "concern_question_open_hi", fallback="{prefix}{note}")
        return yes.format(note=note), asked.format(prefix=prefix, note=note)
    yes = _pick(cfg, "yes_summary_stem_hi" if note else "yes_summary_stem_no_note_hi", fallback="{head}")
    asked = _pick(cfg, "concern_question_with_note_hi" if note else "concern_question_stem_hi", fallback="{head}")
    return yes.format(head=head, note=note), asked.format(prefix=prefix, head=head, note=note)

def _patient_copy(pack: PackConfig, st: CallState, concerns: Sequence[tuple[QuestionSpec, AnswerRecord]], emergency: bool) -> tuple[list[str], list[str]]:
    cfg = pack.report_cfg
    routine = _unique(cfg.get("routine_questions") or ())
    if pack.roster_cfg is not None and not st.roster: routine = _unique(cfg.get("routine_empty_roster_questions") or ()) or routine
    fillers = _unique(cfg.get("emergency_questions") or ()) if emergency else routine
    yes, asked = [], []
    for q, rec in concerns:
        line, question = _concern_copy(cfg, q, rec, _scope(pack, st, rec, clinician=False))
        yes.append(line)
        asked.append(question)
    kept = [q for q in _unique(asked) if q not in routine or not emergency][:6]
    return yes, (kept + [f for f in fillers if len(kept) < 3 and f not in kept])[:6]
def project_summary(pack: PackConfig, state: CallState | None) -> ProjectedReport:
    st, cfg = state or CallState(), pack.report_cfg
    reason = str(st.close_reason or CloseReason.COMPLETE)
    emergency, incomplete = reason == CloseReason.EMERGENCY, reason in _INCOMPLETE
    headings = {str(k): str(v) for k, v in (cfg.get("clinician_headings") or {}).items()}
    known, sections = set(headings.values()), {h: [] for k, h in headings.items() if k not in _SPECIAL}
    alerts, negatives, flags, profile_extra, clear, unasked, unknown, concerns = [], [], [], [], {}, {}, [], []
    specials = {headings.get("alerts"): alerts, headings.get("negatives"): negatives, headings.get("patient_profile"): profile_extra}
    if emergency:
        said, line, flag = _note(st.emergency_note), str(cfg.get("emergency_line_en") or ""), str(cfg.get("emergency_flag_en") or "")
        alerts.append(f"{line} Mother's words: {said}." if said else line)
        flags.append(f"{flag}. {said}" if said else flag)
    for q in pack.questions:
        if not q.is_routable: continue
        rec, label, title = st.answers.get(q.id), _clean(q.stem or q.say), str((pack.groups.get(q.group) or {}).get("title_patient") or q.group)
        if rec is None:
            if incomplete: unasked.setdefault(title, []).append(label)
            continue
        if rec.state is AnswerStatus.NOT_APPLICABLE: continue
        if rec.state is AnswerStatus.UNKNOWN: unknown.append(label)
        elif rec.state is AnswerStatus.CLEAR:
            clear.setdefault(title, []).append(label)
        elif rec.state is AnswerStatus.ANSWERED:
            chosen = _clean(rec.choice_label) or _clean(rec.patient_safe_text)
            if chosen:
                clear.setdefault(title, []).append({"question": label, "answer": chosen})
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
        else:
            lines.append(f"{rec.state.value.title()}. {q.need}.")
            if rec.state is AnswerStatus.CLEAR: negatives.append(f"Clear: {q.need}.")
    roster = engine.roster_label_clinician(st)
    chief = _chief_complaint(pack, concerns, emergency)
    prof = [f"{s.replace('_', ' ').capitalize()}: {_clean(st.profile[s])}." for s in pack.profile_slots if st.profile.get(s)]
    prof += [f"{s.replace('_', ' ').capitalize()}: {_clean(v)}." for s, v in st.profile.items() if s not in pack.profile_slots]
    if pack.roster_cfg: prof.append(f"Children: {roster or 'none reported'}.")
    md = _template(pack, "clinician.md.j2").render(profile_lines=[*prof, f"Close reason: {reason}.", f"Chief complaint: {chief}.", *profile_extra], alerts=alerts,
                                                   sections=[{"heading": h, "lines": v} for h, v in sections.items()], negatives=negatives, incomplete=incomplete, red_flags=flags)
    yes, asked = _patient_copy(pack, st, concerns, emergency)
    body = "\n".join([str(cfg.get("yes_header_hi") or ""), *(f"- {x}" for x in yes)]) if yes else str(cfg.get("yes_empty_hi") or "")
    pat_text = "\n".join([body, "", str(cfg.get("question_header_hi") or ""), *(f"{i}. {q}" for i, q in enumerate(asked, 1))])
    return ProjectedReport(
        patient_name=next((_clean(st.profile[s]) for s in pack.profile_slots if st.profile.get(s)), ""), roster_patient=engine.roster_label_patient(pack, st),
        roster_clinician=roster, close_reason=reason, emergency=emergency, incomplete=incomplete, doctor_markdown=md, patient_summary_text=pat_text,
        red_flags=tuple(_unique(flags)), yes_items=tuple(yes), patient_questions=tuple(asked), page2_groups=_grouped(clear),
        page2_unknown=tuple(_unique(unknown)), page2_unasked=_grouped(unasked) if incomplete else ())
def _ist(raw: Any) -> datetime | None:
    try: return datetime.strptime(str(raw)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone(IST)
    except ValueError: return None
def render_patient_html(pack: PackConfig, call: Mapping[str, Any], proj: ProjectedReport | None = None) -> str:
    rep = proj or project_summary(pack, db.call_state_from_row(call))
    when, months = _ist(call.get("started_at")), list(pack.report_cfg.get("months_hi") or ("",))
    cfg = pack.report_cfg
    m, s = divmod(max(0, int(call.get("duration_sec") or 0)), 60)
    mu, su = cfg.get("minute_unit", "min"), cfg.get("second_unit", "sec")
    dur = f"{m} {mu} {s} {su}" if (m and s) else (f"{m} {mu}" if m else f"{s} {su}")
    date_hi = f"{when.day} {months[when.month]} {when.year}" if (when and 0 < when.month < len(months)) else ""
    clock = f"{when.hour % 12 or 12}:{when.minute:02d} {cfg.get('am_label', 'AM') if when.hour < 12 else cfg.get('pm_label', 'PM')}" if when else ""
    who = " · ".join(b for b in (rep.patient_name, rep.roster_patient) if b) or str((pack.prompts.get("roster_format") or {}).get("patient_default_who") or "")
    pref = str(pack.meta.get("pdf_prefix") or pack.id.upper())
    build, cid, tok = build_info()["tag"], int(call.get("id") or 0), str(call.get("share_token") or "")[:8]; ref = f"{pref}-{cid}" if cid else pref
    meta = " · ".join(b for b in (ref, date_hi, clock, dur, str(cfg.get("incomplete_badge", "incomplete") if rep.incomplete else cfg.get("complete_badge", "complete"))) if b)
    assets = pack.meta.get("assets") or {}
    subject = " ".join(x for x in (f"pack={pack.id}", f"call={cid}" if cid else "", f"token={tok}" if tok else "", f"build={build}") if x)
    css = (pack.root / "templates" / "pdf.css").read_text(encoding="utf-8").replace("__FONT__", (pack.root / "assets" / str(assets.get("font") or "")).as_uri())
    css = css.replace("__FOOTER__", f"{ref} · {build}".replace("\\", "\\\\").replace('"', '\\"'))
    return _template(pack, "patient.html.j2").render(
        css=css, doc_title=ref, pdf_subject=subject, logo_uri=(pack.root / "assets" / str(assets.get("logo") or "")).as_uri(), call_id=cid, who=who, meta=meta,
        emergency=rep.emergency, incomplete=rep.incomplete, questions=list(rep.patient_questions), yes_items=list(rep.yes_items), page2_groups=list(rep.page2_groups),
        page2_unknown=list(rep.page2_unknown), page2_unasked=list(rep.page2_unasked), universal_line=str(pack.report_cfg.get("universal_advice_line_hi") or ""))
def render_patient_pdf(pack: PackConfig, call: Mapping[str, Any], proj: ProjectedReport | None = None) -> bytes:
    return weasyprint.HTML(string=render_patient_html(pack, call, proj), base_url=str(pack.root / "assets")).write_pdf()
def patient_pdf_path(call_id: int) -> Path: return Path(db.DATA_DIR) / "pdfs" / f"call-{int(call_id)}.pdf"

def pdf_filename(pack: PackConfig, call: Mapping[str, Any]) -> str:
    when, pref, cid = _ist(call.get("started_at")), str(pack.meta.get("pdf_prefix") or pack.id.upper()), int(call.get("id") or 0)
    suffix = f"-c{cid}" if cid > 0 else ""
    return f"{pref}{suffix}.pdf" if when is None else f"{pref}-{when:%Y-%m-%d-%H%M%S}{suffix}.pdf"
def build_patient_pdf_sync(call_id: int) -> tuple[bytes, str]:
    if (call := db.get_call(call_id)) is None: raise LookupError(f"Call {call_id} not found")
    pack = load_pack(str(call.get("pack_id") or DEFAULT_PACK_ID))
    proj = project_summary(pack, db.call_state_from_row(call))
    db.save_report_text(call_id, proj.doctor_markdown, proj.patient_summary_text, list(proj.red_flags), proj.patient_name, proj.roster_patient)
    if not (pdf := render_patient_pdf(pack, call, proj)).startswith(b"%PDF"): raise ValueError("Renderer did not return a PDF")
    target = patient_pdf_path(call_id)
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
    return pdf, pdf_filename(pack, call)
def ready_pdf(call_id: int) -> tuple[bytes, str] | None:
    path = patient_pdf_path(call_id)
    if not path.is_file() or len(data := path.read_bytes()) <= 4 or not data.startswith(b"%PDF"): return None
    call = db.get_call(call_id)
    return (data, path.name) if call is None else (data, pdf_filename(load_pack(str(call.get("pack_id") or DEFAULT_PACK_ID)), call))
async def ensure_patient_pdf(call_id: int) -> tuple[bytes, str]:
    loop = asyncio.get_running_loop()
    with _guard_lock: guard, bucket = _guards.setdefault(loop, asyncio.Lock()), _inflight.setdefault(loop, {})
    async with guard:
        if (task := bucket.get(call_id)) is None or task.done():
            task = bucket[call_id] = asyncio.create_task(asyncio.to_thread(build_patient_pdf_sync, call_id))
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
    try: return await asyncio.shield(task)
    finally:
        async with guard:
            if bucket.get(call_id) is task: bucket.pop(call_id, None)
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
