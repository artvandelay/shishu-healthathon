"""One mother PDF per completed call."""

import asyncio
import os
import tempfile
import threading
import time
from pathlib import Path
from weakref import WeakKeyDictionary

import db
import mother_report
import post_call

_inflight: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Task]] = WeakKeyDictionary()
_loop_guards: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()
_loop_guard_lock = threading.Lock()


def _reports_dir() -> Path:
    return Path(db.DATA_DIR) / "mother-pdfs"


def pdf_path(call_id: int) -> Path:
    return _reports_dir() / f"call-{int(call_id)}.pdf"


def _read_pdf(path: Path) -> bytes | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    if data.startswith(b"%PDF") and len(data) > len(b"%PDF"):
        return data
    return None


def ready_pdf(call_id: int) -> tuple[bytes, str] | None:
    """Return a stored PDF, or None while it is missing or not a PDF."""
    data = _read_pdf(pdf_path(call_id))
    if data is None:
        return None
    call = db.get_call(call_id) or {}
    return data, mother_report.filename(call)


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
        lock = _loop_guards.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _loop_guards[loop] = lock
        return lock


def _event(call_id: int, event_type: str, meta: dict | None = None) -> None:
    """Timing only. Detail stays empty so a transcript cannot land here."""
    clean: dict = {}
    for key, value in (meta or {}).items():
        if key not in {"ms", "bytes"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, str) and len(value) <= 80 and " " not in value:
            clean[key] = value
    try:
        db.append_call_event(int(call_id), event_type, "", clean)
    except Exception:
        return


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _loop_inflight() -> dict[int, asyncio.Task]:
    loop = asyncio.get_running_loop()
    with _loop_guard_lock:
        bucket = _inflight.get(loop)
        if bucket is None:
            bucket = {}
            _inflight[loop] = bucket
        return bucket


def _consume_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


def build_mother_pdf_sync(call_id: int, *, emit_events: bool = True) -> tuple[bytes, str]:
    started = time.perf_counter()
    if emit_events:
        _event(call_id, "render_start")
    call = db.get_call(call_id)
    if call is None:
        raise LookupError(f"Call {call_id} was not found")
    synthesis = post_call.build_reports(call)
    report = mother_report.build_report(call, synthesis)
    mother_name = str(synthesis.get("mother_name_hi") or "").strip()
    child_age = str(synthesis.get("children_label_hi") or "").strip()
    db.save_report_text(
        call_id,
        synthesis.get("doctor_markdown") or "",
        post_call.mother_summary_text(synthesis),
        list(synthesis.get("red_flags") or []),
        mother_name,
        child_age,
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
    async with _guard():
        bucket = _loop_inflight()
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
            loop = asyncio.get_running_loop()
            with _loop_guard_lock:
                bucket = _inflight.get(loop)
                if bucket and bucket.get(call_id) is current_task:
                    bucket.pop(call_id, None)
                if bucket is not None and not bucket:
                    _inflight.pop(loop, None)


async def mother_pdf(call_id: int) -> tuple[bytes, str]:
    """Render and atomically store this call's one PDF.

    Overlapping calls for the same id share one build. A later call, including
    Retry, builds again and overwrites call-{id}.pdf.
    """
    task = await _claim(call_id)
    return await asyncio.shield(task)
