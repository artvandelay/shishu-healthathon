import json
import os
import secrets
import sqlite3
import threading
import time
from typing import Any

import call_cursor

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "storage"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "shishu.db")
_db_lock = threading.Lock()

_CLINICIAN_RATINGS = frozenset({"useful", "missed_flag", "false_alarm"})

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
try:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
except Exception:
    pass

def _init_db() -> None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
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
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS server_secrets (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute(
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
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_calls_share_token
            ON calls(share_token)
            WHERE share_token != ''
            """
        )
        cur.execute(
            """
            UPDATE calls
            SET status = 'abandoned', ended_at = CURRENT_TIMESTAMP
            WHERE status = 'active'
              AND started_at < datetime('now', '-25 minutes')
            """
        )
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
    share_token = secrets.token_urlsafe(18)
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO calls (
                status, caller_name, user_role, model_used, voice_used,
                mother_name, share_token, clinic_code
            )
            VALUES ('active', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                caller_name or "",
                user_role or "user",
                model_used or "",
                voice_used or "",
                "",
                share_token,
                clinic_code or "",
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

def append_transcript(call_id: int, role: str, text: str, qid: str = "") -> None:
    if not (text or "").strip():
        return
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT transcript_json FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return
        try:
            transcript = json.loads(row["transcript_json"] or "[]")
        except Exception:
            transcript = []

        same_turn = (
            transcript
            and transcript[-1].get("role") == role
            and (transcript[-1].get("qid") or "") == (qid or "")
        )
        if same_turn:
            prev_text = transcript[-1].get("text", "")
            transcript[-1]["text"] = call_cursor.merge_transcript_chunk(prev_text, text)
            transcript[-1]["ts"] = time.time()
            if qid:
                transcript[-1]["qid"] = qid
        else:
            entry = {"role": role, "text": text.strip(), "ts": time.time()}
            if qid:
                entry["qid"] = qid
            transcript.append(entry)

        cur.execute(
            "UPDATE calls SET transcript_json = ? WHERE id = ?",
            (json.dumps(transcript, ensure_ascii=False), call_id),
        )
        conn.commit()

def complete_call(
    call_id: int,
    duration_sec: int,
    audio_path: str = "",
) -> None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE calls
            SET status = 'completed',
                ended_at = CURRENT_TIMESTAMP,
                duration_sec = ?,
                audio_path = ?
            WHERE id = ?
            """,
            (int(duration_sec), audio_path or "", int(call_id)),
        )
        conn.commit()

def _cursor_labels(state: dict) -> tuple[str, str]:
    """Admin list labels: mother name, and the children's ages and sexes."""
    profile = (state or {}).get("profile") if isinstance(state, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    name = str(profile.get("caller_name") or "").strip()
    return name, call_cursor.children_label_hi(profile)

def save_cursor_state(call_id: int, state: dict) -> None:
    name, child_age = _cursor_labels(state or {})
    cursor_raw = json.dumps(state or {}, ensure_ascii=False)
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            "UPDATE calls SET cursor_json = ?, mother_name = ?, child_age = ? WHERE id = ?",
            (cursor_raw, name, child_age, int(call_id)),
        )
        conn.commit()

def get_call_by_token(share_token: str) -> dict | None:
    token = (share_token or "").strip()
    if len(token) < 8:
        return None
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM calls WHERE share_token = ?", (token,))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_call_dict(row)

def save_clinician_review(call_id: int, rating: str, note: str = "") -> dict:
    if rating not in _CLINICIAN_RATINGS:
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
    sets: list[str] = []
    values: list[Any] = []
    if summary_doctor is not None:
        sets.append("summary_doctor = ?")
        values.append(summary_doctor)
    if summary_mother is not None:
        sets.append("summary_mother = ?")
        values.append(summary_mother)
    if red_flags is not None:
        sets.append("red_flags_json = ?")
        values.append(json.dumps(list(red_flags), ensure_ascii=False))
    if mother_name is not None:
        sets.append("mother_name = ?")
        values.append(mother_name)
    if child_age is not None:
        sets.append("child_age = ?")
        values.append(child_age)
    if not sets:
        return
    values.append(int(call_id))
    with _db_lock:
        cur = conn.cursor()
        cur.execute(f"UPDATE calls SET {', '.join(sets)} WHERE id = ?", values)
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
        try:
            events = json.loads(row["events_json"] or "[]")
        except Exception:
            events = []
        if not isinstance(events, list):
            events = []
        events.append(event)
        events = events[-600:]
        cur.execute(
            "UPDATE calls SET events_json = ? WHERE id = ?",
            (json.dumps(events, ensure_ascii=False), call_id),
        )
        conn.commit()
    return event

def update_call_telemetry(call_id: int, patch: dict[str, Any]) -> dict:
    if not isinstance(patch, dict):
        patch = {}
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT telemetry_json FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return {}
        try:
            telemetry = json.loads(row["telemetry_json"] or "{}")
        except Exception:
            telemetry = {}
        if not isinstance(telemetry, dict):
            telemetry = {}
        telemetry.update(patch)
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
    call = get_call(call_id) if include_call_body and call_id and call_id > 0 else None
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
            INSERT INTO bug_reports (call_id, reporter, category, note, call_elapsed_sec, snapshot_json)
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
    if include_call_body and call_id and call_id > 0:
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
    out = []
    for row in rows:
        item = dict(row)
        raw_snapshot = item.pop("snapshot_json", None)
        try:
            item["snapshot"] = json.loads(raw_snapshot or "{}")
        except Exception:
            item["snapshot"] = {}
        out.append(item)
    return out

_JSON_CALL_COLUMNS: tuple[tuple[str, str, type], ...] = (
    ("red_flags_json", "red_flags", list),
    ("clinician_review_json", "clinician_review", dict),
    ("transcript_json", "transcript", list),
    ("telemetry_json", "telemetry", dict),
    ("events_json", "events", list),
    ("cursor_json", "cursor_state", dict),
)


def _row_to_call_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for raw_key, out_key, default_cls in _JSON_CALL_COLUMNS:
        if raw_key not in d:
            continue
        raw = d.pop(raw_key)
        try:
            d[out_key] = json.loads(raw) if raw and str(raw).strip() else default_cls()
        except Exception:
            d[out_key] = default_cls()
    return d

def get_call(call_id: int) -> dict | None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return None
        if not str(row["share_token"] or "").strip():
            token = secrets.token_urlsafe(18)
            cur.execute(
                "UPDATE calls SET share_token = ? WHERE id = ?",
                (token, call_id),
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
    cols = ", ".join(_LIST_COLUMNS)
    with _db_lock:
        cur = conn.cursor()
        cur.execute(f"SELECT {cols} FROM calls ORDER BY id DESC")
        rows = cur.fetchall()
        return [_row_to_call_dict(r) for r in rows]
