"""SQLite storage for calls, transcripts, events, and bug reports.

`cursor_state` is stored as one `CallState` JSON blob. Everything that reads it
back goes through `call_state_from_row`, so no caller ever handles a raw dict.
"""

import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Mapping
from typing import Any

import copy_hi
import engine
from models import CallState

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "storage"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "shishu.db")
_db_lock = threading.Lock()

CLINICIAN_RATINGS = frozenset({"useful", "missed_flag", "false_alarm"})

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row

for _pragma in (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
):
    try:
        conn.execute(_pragma)
    except sqlite3.Error:
        pass

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mother_name TEXT DEFAULT '',
        child_age TEXT DEFAULT '',
        status TEXT DEFAULT 'active',
        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        ended_at DATETIME,
        duration_sec INTEGER DEFAULT 0,
        audio_path TEXT DEFAULT '',
        transcript_json TEXT DEFAULT '[]',
        red_flags_json TEXT DEFAULT '[]',
        summary_doctor TEXT DEFAULT '',
        summary_mother TEXT DEFAULT '',
        caller_name TEXT DEFAULT '',
        user_role TEXT DEFAULT 'user',
        model_used TEXT DEFAULT '',
        voice_used TEXT DEFAULT '',
        telemetry_json TEXT DEFAULT '{}',
        events_json TEXT DEFAULT '[]',
        cursor_json TEXT DEFAULT '',
        share_token TEXT DEFAULT '',
        clinic_code TEXT DEFAULT '',
        clinician_review_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS server_secrets (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bug_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        reporter TEXT DEFAULT '',
        category TEXT DEFAULT '',
        note TEXT DEFAULT '',
        call_elapsed_sec INTEGER DEFAULT 0,
        snapshot_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_calls_share_token
    ON calls(share_token) WHERE share_token != ''
    """,
    """
    UPDATE calls
    SET status = 'abandoned', ended_at = CURRENT_TIMESTAMP
    WHERE status = 'active' AND started_at < datetime('now', '-25 minutes')
    """,
)

_EVENT_CAP = 600
_MIN_TOKEN = 8


def _init_db() -> None:
    with _db_lock:
        cur = conn.cursor()
        for statement in _SCHEMA:
            cur.execute(statement)
        conn.commit()


_init_db()


def get_session_secret() -> str:
    """Opaque HMAC key for login cookies. Not a setting, and not returned by the API."""
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT value FROM server_secrets WHERE key = ?", ("session",))
        row = cur.fetchone()
        if row and row["value"]:
            return str(row["value"])
        value = secrets.token_hex(32)
        cur.execute(
            "INSERT INTO server_secrets (key, value) VALUES (?, ?)",
            ("session", value),
        )
        conn.commit()
        return value


def create_call(
    caller_name: str = "",
    user_role: str = "user",
    model_used: str = "",
    voice_used: str = "",
    clinic_code: str = "",
) -> int:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO calls (
                status, caller_name, user_role, model_used, voice_used,
                mother_name, share_token, clinic_code
            )
            VALUES ('active', ?, ?, ?, ?, '', ?, ?)
            """,
            (
                caller_name or "",
                user_role or "user",
                model_used or "",
                voice_used or "",
                secrets.token_urlsafe(18),
                clinic_code or "",
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def append_transcript(call_id: int, role: str, text: str, qid: str = "") -> None:
    """Append one spoken chunk, merging it into the open turn of the same qid."""
    if not (text or "").strip():
        return
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT transcript_json FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return
        transcript = _load_json(row["transcript_json"], list)
        last = transcript[-1] if transcript else None
        same_role = bool(last) and last.get("role") == role
        same_turn = same_role and (last.get("qid") or "") == (qid or "")
        if same_turn:
            last["text"] = engine.merge_transcript_chunk(last.get("text", ""), text)
            last["ts"] = time.time()
            last["qid"] = qid or last.get("qid") or ""
        else:
            entry: dict[str, Any] = {"role": role, "text": text.strip(), "ts": time.time()}
            if qid:
                entry["qid"] = qid
            transcript.append(entry)
        cur.execute(
            "UPDATE calls SET transcript_json = ? WHERE id = ?",
            (json.dumps(transcript, ensure_ascii=False), call_id),
        )
        conn.commit()


def complete_call(call_id: int, duration_sec: int, audio_path: str = "") -> None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE calls
            SET status = 'completed', ended_at = CURRENT_TIMESTAMP,
                duration_sec = ?, audio_path = ?
            WHERE id = ?
            """,
            (int(duration_sec), audio_path or "", int(call_id)),
        )
        conn.commit()


def call_state_from_row(call: Mapping[str, Any] | None) -> CallState:
    """The typed state stored in `cursor_json`, or a fresh state when absent."""
    raw = (call or {}).get("cursor_state")
    if isinstance(raw, CallState):
        return raw
    if isinstance(raw, Mapping) and raw:
        return CallState.from_dict(raw)
    return CallState()


def save_cursor_state(call_id: int, state: CallState | Mapping[str, Any] | None) -> None:
    call_state = state if isinstance(state, CallState) else CallState.from_dict(state)
    name = (call_state.profile.caller_name or "").strip()
    child_age = copy_hi.children_label_hi(call_state.profile)
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            "UPDATE calls SET cursor_json = ?, mother_name = ?, child_age = ? WHERE id = ?",
            (json.dumps(call_state.to_dict(), ensure_ascii=False), name, child_age, int(call_id)),
        )
        conn.commit()


def get_call_by_token(share_token: str) -> dict | None:
    token = (share_token or "").strip()
    if len(token) < _MIN_TOKEN:
        return None
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM calls WHERE share_token = ?", (token,))
        row = cur.fetchone()
        return _row_to_call_dict(row) if row else None


def save_clinician_review(call_id: int, rating: str, note: str = "") -> dict:
    if rating not in CLINICIAN_RATINGS:
        raise ValueError(f"unknown clinician rating {rating!r}")
    payload = {"rating": rating, "note": note or ""}
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            "UPDATE calls SET clinician_review_json = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), call_id),
        )
        conn.commit()
    return payload


def save_report_text(
    call_id: int,
    summary_doctor: str | None = None,
    summary_mother: str | None = None,
    red_flags: list | None = None,
    mother_name: str | None = None,
    child_age: str | None = None,
) -> None:
    """Write report text without changing call status or the other summary."""
    columns: list[str] = []
    values: list[Any] = []
    flags_json = None if red_flags is None else json.dumps(list(red_flags), ensure_ascii=False)
    patches = (
        ("summary_doctor", summary_doctor),
        ("summary_mother", summary_mother),
        ("red_flags_json", flags_json),
        ("mother_name", mother_name),
        ("child_age", child_age),
    )
    for column, value in patches:
        if value is not None:
            columns.append(f"{column} = ?")
            values.append(value)
    if not columns:
        return
    values.append(int(call_id))
    with _db_lock:
        cur = conn.cursor()
        cur.execute(f"UPDATE calls SET {', '.join(columns)} WHERE id = ?", values)
        conn.commit()


def append_call_event(
    call_id: int, event_type: str, detail: str = "", meta: dict | None = None
) -> dict:
    event = {
        "t": round(time.time(), 3),
        "type": event_type,
        "detail": detail or "",
        "meta": meta or {},
    }
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT events_json FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return event
        events = _load_json(row["events_json"], list)[-_EVENT_CAP:]
        events.append(event)
        cur.execute(
            "UPDATE calls SET events_json = ? WHERE id = ?",
            (json.dumps(events, ensure_ascii=False), call_id),
        )
        conn.commit()
    return event


def update_call_telemetry(call_id: int, patch: dict[str, Any]) -> dict:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT telemetry_json FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return {}
        telemetry = _load_json(row["telemetry_json"], dict)
        telemetry.update(patch if isinstance(patch, dict) else {})
        cur.execute(
            "UPDATE calls SET telemetry_json = ? WHERE id = ?",
            (json.dumps(telemetry, ensure_ascii=False), call_id),
        )
        conn.commit()
        return telemetry


def create_bug_report(
    call_id: int,
    reporter: str,
    category: str,
    note: str,
    call_elapsed_sec: int = 0,
    extra_snapshot: dict | None = None,
    include_call_body: bool = False,
) -> dict:
    owned = bool(include_call_body and call_id and call_id > 0)
    call = get_call(call_id) if owned else None
    transcript = (call or {}).get("transcript") or []
    snapshot = {
        "last_transcript": transcript[-8:] if isinstance(transcript, list) else [],
        "telemetry": (call or {}).get("telemetry") or {},
        "recent_events": ((call or {}).get("events") or [])[-20:],
        "client": extra_snapshot or {},
    }
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO bug_reports (
                call_id, reporter, category, note, call_elapsed_sec, snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(call_id or 0),
                reporter or "",
                category or "",
                note or "",
                int(call_elapsed_sec or 0),
                json.dumps(snapshot, ensure_ascii=False),
            ),
        )
        bug_id = int(cur.lastrowid)
        conn.commit()
    if owned:
        append_call_event(call_id, "bug_reported", f"{category}: {note}".strip(": "))
    return {
        "id": bug_id,
        "call_id": int(call_id or 0),
        "reporter": reporter or "",
        "category": category or "",
        "note": note or "",
        "call_elapsed_sec": int(call_elapsed_sec or 0),
    }


def list_bug_reports(limit: int = 50) -> list[dict]:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM bug_reports ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    reports: list[dict] = []
    for row in rows:
        item = dict(row)
        item["snapshot"] = _load_json(item.pop("snapshot_json", None), dict)
        reports.append(item)
    return reports


_JSON_COLUMNS: tuple[tuple[str, str, type], ...] = (
    ("red_flags_json", "red_flags", list),
    ("clinician_review_json", "clinician_review", dict),
    ("transcript_json", "transcript", list),
    ("telemetry_json", "telemetry", dict),
    ("events_json", "events", list),
    ("cursor_json", "cursor_state", dict),
)


def _load_json(raw: Any, default_cls: type) -> Any:
    """Parsed JSON, or a fresh default when the column is blank or malformed."""
    try:
        parsed = json.loads(raw) if raw and str(raw).strip() else None
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, default_cls):
        return parsed
    return default_cls()


def _row_to_call_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    call = dict(row)
    for raw_key, out_key, default_cls in _JSON_COLUMNS:
        if raw_key in call:
            call[out_key] = _load_json(call.pop(raw_key), default_cls)
    return call


def get_call(call_id: int) -> dict | None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if row is None:
            return None
        if not str(row["share_token"] or "").strip():
            cur.execute(
                "UPDATE calls SET share_token = ? WHERE id = ?",
                (secrets.token_urlsafe(18), int(call_id)),
            )
            conn.commit()
            cur.execute("SELECT * FROM calls WHERE id = ?", (call_id,))
            row = cur.fetchone()
    return _row_to_call_dict(row)


_LIST_COLUMNS = (
    "id", "mother_name", "child_age",
    "status", "started_at", "ended_at", "duration_sec", "red_flags_json",
    "caller_name", "user_role", "model_used", "voice_used",
    "clinic_code", "clinician_review_json",
)


def list_calls() -> list[dict]:
    """Every call, without transcripts or cursor JSON. The detail view loads those."""
    with _db_lock:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(_LIST_COLUMNS)} FROM calls ORDER BY id DESC")
        rows = cur.fetchall()
    return [_row_to_call_dict(row) for row in rows]
