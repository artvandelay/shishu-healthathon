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
        cur.execute(
            """
            INSERT OR IGNORE INTO settings (key, value)
            VALUES ('active_engine', 'gemini')
            """
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


def create_call(engine: str) -> int:
    with _db_lock:
        cur = conn.cursor()
        cur.execute("INSERT INTO calls (engine, status) VALUES (?, 'active')", (engine,))
        conn.commit()
        return int(cur.lastrowid)


def append_transcript(call_id: int, role: str, text: str) -> None:
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
            if prev_text and not prev_text.endswith(" ") and not text.startswith(" "):
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
