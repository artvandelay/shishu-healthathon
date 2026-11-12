"""One mother PDF per completed call.

Extraction can start from an immutable snapshot while the outro is still playing.
Rendering waits until the caller has the completed call, then reuses that extraction.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import answer_extractor
import db
import mother_report
import post_call

_REPORTS = Path(db.DATA_DIR) / "mother-pdfs"
_answers: dict[int, asyncio.Task] = {}
_inflight: dict[int, asyncio.Task] = {}
_claim_guard: asyncio.Lock | None = None


def pdf_path(call_id: int) -> Path:
    return _REPORTS / f"call-{int(call_id)}.pdf"


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
    global _claim_guard
    if _claim_guard is None:
        _claim_guard = asyncio.Lock()
    return _claim_guard


def _failed(task: asyncio.Task) -> bool:
    if not task.done():
        return False
    if task.cancelled():
        return True
    return task.exception() is not None


def _event(call_id: int, event_type: str, meta: dict | None = None) -> None:
    answer_extractor._event(call_id, event_type, meta)


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _freeze(call_id: int, snapshot: dict) -> dict:
    frozen = json.loads(json.dumps(snapshot, ensure_ascii=False))
    if not isinstance(frozen, dict):
        frozen = {}
    frozen["id"] = int(call_id)
    return frozen


async def begin_answers(call_id: int, snapshot: dict) -> None:
    """Start extraction once from a snapshot. A later render awaits this task."""
    async with _guard():
        task = _answers.get(call_id)
        if task is not None and not _failed(task):
            return
        frozen = _freeze(call_id, snapshot)
        if post_call.has_cursor(frozen):
            job = post_call.synthesize_call(frozen)
        else:
            job = answer_extractor.cached_answers(frozen)
        _answers[call_id] = asyncio.create_task(job)


async def _answers_task(call_id: int) -> asyncio.Task:
    async with _guard():
        task = _answers.get(call_id)
        if task is not None and not _failed(task):
            return task
        call = db.get_call(call_id) or {"id": int(call_id)}
        if post_call.has_cursor(call):
            job = post_call.synthesize_call(call)
        else:
            job = answer_extractor.cached_answers(call)
        task = asyncio.create_task(job)
        _answers[call_id] = task
        return task


async def _claim(call_id: int) -> asyncio.Task:
    async with _guard():
        task = _inflight.get(call_id)
        if task is None or task.done():
            task = asyncio.create_task(_build(call_id))
            _inflight[call_id] = task
        return task


async def _build(call_id: int) -> tuple[bytes, str]:
    task = await _answers_task(call_id)
    try:
        extracted = await task
    except Exception:
        async with _guard():
            if _answers.get(call_id) is task:
                _answers.pop(call_id, None)
        raise
    started = time.perf_counter()
    _event(call_id, "render_start")
    try:
        call = await asyncio.to_thread(db.get_call, call_id)
        if call is None:
            raise LookupError(f"Call {call_id} was not found")
        if post_call.is_synthesis(extracted) or post_call.has_cursor(call):
            synthesis = extracted if post_call.is_synthesis(extracted) else post_call.deterministic_fallback(call)
            report = mother_report.build_v3(call, synthesis)
            await asyncio.to_thread(
                db.save_report_text,
                call_id,
                synthesis.get("doctor_markdown") or "",
                post_call.mother_summary_text(synthesis),
            )
        else:
            report = mother_report.build(call, extracted)
        pdf = await asyncio.to_thread(mother_report.render_pdf, report)
        if not pdf.startswith(b"%PDF"):
            raise ValueError("Renderer did not return a PDF")
        _event(call_id, "render_end", {"ms": _ms(started)})
        await asyncio.to_thread(_atomic_write, pdf_path(call_id), pdf)
        _event(call_id, "pdf_write", {"bytes": len(pdf)})
        _event(call_id, "report_ready", {"ms": _ms(started)})
    finally:
        async with _guard():
            if _answers.get(call_id) is task:
                _answers.pop(call_id, None)
    return pdf, mother_report.filename(call)


async def mother_pdf(call_id: int) -> tuple[bytes, str]:
    """Render and atomically store this call's one PDF.

    Reuses an extraction already started from the end-of-call snapshot. Overlapping
    calls for the same id share one build. A later call, including Retry, extracts
    again and overwrites call-{id}.pdf.
    """
    task = await _claim(call_id)
    return await task
