import json
import os
import sqlite3
import threading
import time
from typing import Any

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "storage"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "shishu.db")
_db_lock = threading.Lock()

DEFAULT_SETTINGS: dict[str, str] = {
    "active_engine": "gemini",
    "gemini_model": "gemini-3.8-live",
    "gemini_voice": "Kore",
    "openai_model": "gpt-live-1",
    "openai_voice": "marin",
    "openai_backend_model": "gpt-5.6-luna",
    "vad_prefix_padding_ms": "300",
    "vad_silence_ms": "650",
    "vad_start_sensitivity": "HIGH",
    "vad_end_sensitivity": "HIGH",
    "watchdog_timeout_sec": "4.0",
    "summary_model": "gemini-2.5-flash",
}

_CALL_EXTRA_COLUMNS = (
    ("caller_name", "TEXT DEFAULT ''"),
    ("user_role", "TEXT DEFAULT 'user'"),
    ("model_used", "TEXT DEFAULT ''"),
    ("voice_used", "TEXT DEFAULT ''"),
    ("telemetry_json", "TEXT DEFAULT '{}'"),
    ("events_json", "TEXT DEFAULT '[]'"),
)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row


def _init_db() -> None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        for key, value in DEFAULT_SETTINGS.items():
            cur.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine TEXT,
                mother_name TEXT DEFAULT '',
                child_name TEXT DEFAULT '',
                child_age TEXT DEFAULT '',
                child_gender TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                duration_sec INTEGER DEFAULT 0,
                audio_path TEXT DEFAULT '',
                transcript_json TEXT DEFAULT '[]',
                findings_json TEXT DEFAULT '{}',
                red_flags_json TEXT DEFAULT '[]',
                summary_doctor TEXT DEFAULT '',
                summary_mother TEXT DEFAULT ''
            )
            """
        )
        existing = {
            row["name"]
            for row in cur.execute("PRAGMA table_info(calls)").fetchall()
        }
        for name, decl in _CALL_EXTRA_COLUMNS:
            if name not in existing:
                cur.execute(f"ALTER TABLE calls ADD COLUMN {name} {decl}")
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
        conn.commit()


_init_db()


def get_setting(key: str, default: str = "") -> str:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def get_all_settings() -> dict[str, str]:
    merged = dict(DEFAULT_SETTINGS)
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM settings")
        for row in cur.fetchall():
            merged[row["key"]] = row["value"] if row["value"] is not None else ""
    return merged


def update_settings(updates: dict[str, Any]) -> dict[str, str]:
    if not isinstance(updates, dict):
        return get_all_settings()
    with _db_lock:
        cur = conn.cursor()
        for key, value in updates.items():
            if key not in DEFAULT_SETTINGS:
                continue
            cur.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, "" if value is None else str(value)),
            )
        conn.commit()
    return get_all_settings()


def create_call(
    engine: str,
    caller_name: str = "",
    user_role: str = "user",
    model_used: str = "",
    voice_used: str = "",
) -> int:
    mother = caller_name if caller_name and caller_name.strip().lower() != "admin" else ""
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO calls (engine, status, caller_name, user_role, model_used, voice_used, mother_name)
            VALUES (?, 'active', ?, ?, ?, ?, ?)
            """,
            (engine, caller_name or "", user_role or "user", model_used or "", voice_used or "", mother),
        )
        conn.commit()
        return int(cur.lastrowid)


def append_transcript(call_id: int, role: str, text: str, exact: bool = False) -> None:
    if not text:
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

        if transcript and transcript[-1].get("role") == role:
            prev_text = transcript[-1].get("text", "")
            if exact:
                transcript[-1]["text"] = prev_text + text
            elif prev_text and not prev_text.endswith(" ") and not text.startswith(" "):
                transcript[-1]["text"] = prev_text + " " + text
            else:
                transcript[-1]["text"] = prev_text + text
            transcript[-1]["ts"] = time.time()
        else:
            transcript.append({"role": role, "text": text, "ts": time.time()})

        cur.execute(
            "UPDATE calls SET transcript_json = ? WHERE id = ?",
            (json.dumps(transcript, ensure_ascii=False), call_id),
        )
        conn.commit()


def record_finding_db(
    call_id: int, domain: str, status: str, detail: str, is_red_flag: bool = False
) -> dict:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT findings_json, red_flags_json FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return {}
        try:
            findings = json.loads(row["findings_json"] or "{}")
        except Exception:
            findings = {}
        try:
            red_flags = json.loads(row["red_flags_json"] or "[]")
        except Exception:
            red_flags = []

        findings[domain] = {
            "status": status,
            "detail": detail,
            "is_red_flag": bool(is_red_flag),
        }

        profile_col_map = {
            "profile_mother_name": "mother_name",
            "profile_child_name": "child_name",
            "profile_child_age": "child_age",
            "profile_child_gender": "child_gender",
        }
        if domain in profile_col_map and detail:
            col = profile_col_map[domain]
            cur.execute(f"UPDATE calls SET {col} = ? WHERE id = ?", (detail, call_id))

        if is_red_flag:
            flag_str = f"{domain}: {detail}"
            if flag_str not in red_flags:
                red_flags.append(flag_str)

        cur.execute(
            "UPDATE calls SET findings_json = ?, red_flags_json = ? WHERE id = ?",
            (
                json.dumps(findings, ensure_ascii=False),
                json.dumps(red_flags, ensure_ascii=False),
                call_id,
            ),
        )
        conn.commit()
        return findings


def complete_call(
    call_id: int,
    duration_sec: int,
    audio_filename: str,
    summary_doctor: str,
    summary_mother: str,
) -> None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE calls
            SET status = 'completed',
                ended_at = CURRENT_TIMESTAMP,
                duration_sec = ?,
                audio_path = ?,
                summary_doctor = ?,
                summary_mother = ?
            WHERE id = ?
            """,
            (duration_sec, audio_filename, summary_doctor, summary_mother, call_id),
        )
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
        events = events[-250:]
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
) -> dict:
    call = get_call(call_id) if call_id and call_id > 0 else None
    transcript = (call or {}).get("transcript") or []
    snapshot = {
        "settings": get_all_settings(),
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
    if call_id and call_id > 0:
        append_call_event(call_id, "bug_reported", f"{category}: {note}".strip(": "))
    return {
        "id": bug_id,
        "call_id": int(call_id or 0),
        "reporter": reporter or "",
        "category": category or "",
        "note": note or "",
        "call_elapsed_sec": int(call_elapsed_sec or 0),
        "snapshot": snapshot,
    }


def list_bug_reports(limit: int = 50) -> list[dict]:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM bug_reports ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["snapshot"] = json.loads(item.get("snapshot_json") or "{}")
        except Exception:
            item["snapshot"] = {}
        out.append(item)
    return out


def _row_to_call_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["transcript"] = json.loads(d.get("transcript_json") or "[]")
    except Exception:
        d["transcript"] = []
    try:
        d["findings"] = json.loads(d.get("findings_json") or "{}")
    except Exception:
        d["findings"] = {}
    try:
        d["red_flags"] = json.loads(d.get("red_flags_json") or "[]")
    except Exception:
        d["red_flags"] = []
    try:
        d["telemetry"] = json.loads(d.get("telemetry_json") or "{}")
    except Exception:
        d["telemetry"] = {}
    try:
        d["events"] = json.loads(d.get("events_json") or "[]")
    except Exception:
        d["events"] = []
    return d


def get_call(call_id: int) -> dict | None:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_call_dict(row)


def list_calls(limit: int = 50) -> list[dict]:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [_row_to_call_dict(r) for r in rows]
