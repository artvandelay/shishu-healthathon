"""The one Mother PDF per call: project, persist, render, and store it atomically."""

import asyncio
import os
import tempfile
import threading
import time
from pathlib import Path
from weakref import WeakKeyDictionary

import db
import mother_report
import summary

_TaskBucket = dict[int, asyncio.Task]
_inflight: WeakKeyDictionary[asyncio.AbstractEventLoop, _TaskBucket] = WeakKeyDictionary()
_loop_guards: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()
_loop_guard_lock = threading.Lock()
_EVENT_KEYS = frozenset({"ms", "bytes"})


def _reports_dir() -> Path:
    return Path(db.DATA_DIR) / "mother-pdfs"


def pdf_path(call_id: int) -> Path:
    return _reports_dir() / f"call-{int(call_id)}.pdf"


def ready_pdf(call_id: int) -> tuple[bytes, str] | None:
    """Return a stored PDF, or None while it is missing or not a PDF."""
    try:
        data = pdf_path(call_id).read_bytes()
    except FileNotFoundError:
        return None
    if not data.startswith(b"%PDF") or len(data) <= len(b"%PDF"):
        return None
    return data, mother_report.filename(db.get_call(call_id) or {})


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _guard() -> asyncio.Lock:
    """One asyncio lock per running loop. A process-wide lock breaks the next loop."""
    loop = asyncio.get_running_loop()
    with _loop_guard_lock:
        return _loop_guards.setdefault(loop, asyncio.Lock())


def _event(call_id: int, event_type: str, meta: dict | None = None) -> None:
    """Timing only. Detail stays empty so a transcript cannot land here."""
    clean = {
        key: value
        for key, value in (meta or {}).items()
        if key in _EVENT_KEYS and isinstance(value, int) and not isinstance(value, bool)
    }
    try:
        db.append_call_event(int(call_id), event_type, "", clean)
    except Exception:
        return


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _inflight_bucket() -> _TaskBucket:
    loop = asyncio.get_running_loop()
    with _loop_guard_lock:
        return _inflight.setdefault(loop, {})


def _consume_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


def build_mother_pdf_sync(call_id: int, *, emit_events: bool = True) -> tuple[bytes, str]:
    """Project the call once, save every text field, then render and store the PDF."""
    started = time.perf_counter()
    if emit_events:
        _event(call_id, "render_start")
    call = db.get_call(call_id)
    if call is None:
        raise LookupError(f"Call {call_id} was not found")

    proj = summary.project_summary(db.call_state_from_row(call))
    report = mother_report.build_report(call, proj)
    db.save_report_text(
        call_id,
        proj.doctor_markdown,
        proj.mother_summary_text,
        list(proj.red_flags),
        proj.mother_name,
        proj.children_label_hi,
    )
    pdf = mother_report.render_pdf(report)
    if not pdf.startswith(b"%PDF"):
        raise ValueError("Renderer did not return a PDF")
    if emit_events:
        _event(call_id, "render_end", {"ms": _ms(started)})
    _atomic_write(pdf_path(call_id), pdf)
    if emit_events:
        _event(call_id, "pdf_write", {"bytes": len(pdf)})
        _event(call_id, "report_ready", {"ms": _ms(started)})
    return pdf, mother_report.filename(call)


async def _claim(call_id: int) -> asyncio.Task:
    """Join the build already running for this id, or start the one build."""
    async with _guard():
        bucket = _inflight_bucket()
        task = bucket.get(call_id)
        if task is None or task.done():
            task = asyncio.create_task(_build(call_id))
            task.add_done_callback(_consume_task_exception)
            bucket[call_id] = task
        return task


async def _build(call_id: int) -> tuple[bytes, str]:
    current_task = asyncio.current_task()
    try:
        return await asyncio.to_thread(build_mother_pdf_sync, call_id, emit_events=True)
    finally:
        async with _guard():
            bucket = _inflight_bucket()
            if bucket.get(call_id) is current_task:
                bucket.pop(call_id, None)
            if not bucket:
                _inflight.pop(asyncio.get_running_loop(), None)


async def mother_pdf(call_id: int) -> tuple[bytes, str]:
    """Render and atomically store this call's one PDF.

    Overlapping calls for the same id share one build. A later call, including
    Retry, builds again and overwrites call-{id}.pdf.
    """
    return await asyncio.shield(await _claim(call_id))
