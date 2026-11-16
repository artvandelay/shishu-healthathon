"""SQLite storage for calls, transcripts, events, and bug reports (`intake/db.py`)."""

from __future__ import annotations

import json, os, secrets, sqlite3, threading, time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import engine
from .models import CallState, PackConfig

DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "storage"))
DB_PATH = Path(DATA_DIR) / "intake.db"
_LOCK, _EVENT_CAP, _MIN_TOKEN = threading.Lock(), 600, 8
_JSON = (("red_flags_json", "red_flags", list), ("transcript_json", "transcript", list), ("telemetry_json", "telemetry", dict),
         ("events_json", "events", list), ("cursor_json", "cursor_state", dict))
_LISTED = ("id, pack_id, patient_name, roster_summary, status, started_at, ended_at, duration_sec, audio_path, caller_name, "
           "user_role, model_used, voice_used, clinic_code, share_token, red_flags_json")
_CALLS = ("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY AUTOINCREMENT, pack_id TEXT DEFAULT 'shishu-hi', patient_name TEXT DEFAULT '', roster_summary TEXT DEFAULT '', status TEXT DEFAULT 'active', "
          "started_at DATETIME DEFAULT CURRENT_TIMESTAMP, ended_at DATETIME, duration_sec INTEGER DEFAULT 0, audio_path TEXT DEFAULT '', transcript_json TEXT DEFAULT '[]', red_flags_json TEXT DEFAULT '[]', "
          "summary_doctor TEXT DEFAULT '', summary_patient TEXT DEFAULT '', caller_name TEXT DEFAULT '', user_role TEXT DEFAULT 'user', model_used TEXT DEFAULT '', voice_used TEXT DEFAULT '', "
          "telemetry_json TEXT DEFAULT '{}', events_json TEXT DEFAULT '[]', cursor_json TEXT DEFAULT '', share_token TEXT DEFAULT '', clinic_code TEXT DEFAULT '')")
_BUGS = ("CREATE TABLE IF NOT EXISTS bug_reports (id INTEGER PRIMARY KEY AUTOINCREMENT, call_id INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
         "reporter TEXT DEFAULT '', category TEXT DEFAULT '', note TEXT DEFAULT '', call_elapsed_sec INTEGER DEFAULT 0, snapshot_json TEXT DEFAULT '{}')")
for _sub in ("", "audio", "pdfs"): (Path(DATA_DIR) / _sub).mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
conn.row_factory = sqlite3.Row
with _LOCK:
    for _sql in ("PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=5000", _CALLS, _BUGS, "CREATE TABLE IF NOT EXISTS server_secrets (key TEXT PRIMARY KEY, value TEXT)",
                 "CREATE UNIQUE INDEX IF NOT EXISTS idx_calls_token ON calls(share_token) WHERE share_token != ''"): conn.execute(_sql)
    conn.commit()
def _json(raw: Any, default: type) -> Any:
    try: parsed = json.loads(raw) if raw and str(raw).strip() else None
    except (TypeError, ValueError): parsed = None
    return parsed if isinstance(parsed, default) else default()
def _write(call_id: int, **fields: Any) -> None:
    with _LOCK:
        conn.execute(f"UPDATE calls SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", (*fields.values(), int(call_id)))
        conn.commit()
def _read(call_id: int, col: str) -> Any:
    with _LOCK:
        row = conn.execute(f"SELECT {col} FROM calls WHERE id = ?", (int(call_id),)).fetchone()
    return None if row is None else row[col]
def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None: return None
    call = dict(row)
    for rk, ok, cls in _JSON:
        if rk in call: call[ok] = _json(call.pop(rk), cls)
    return call
def get_session_secret() -> str:
    with _LOCK:
        if (row := conn.execute("SELECT value FROM server_secrets WHERE key = 'session'").fetchone()) and row["value"]: return str(row["value"])
        val = secrets.token_hex(32)
        conn.execute("INSERT INTO server_secrets VALUES ('session', ?)", (val,))
        conn.commit()
    return val
def create_call(*, pack_id: str = "shishu-hi", caller_name: str = "", user_role: str = "user", model_used: str = "", voice_used: str = "",
                clinic_code: str = "") -> int:
    with _LOCK:
        cur = conn.execute("INSERT INTO calls (pack_id, caller_name, user_role, model_used, voice_used, clinic_code, share_token) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (pack_id, caller_name, user_role, model_used, voice_used, clinic_code, secrets.token_urlsafe(18)))
        conn.commit()
    return int(cur.lastrowid)
def append_transcript(call_id: int, role: str, text: str, qid: str = "") -> None:
    if not (text or "").strip() or (raw := _read(call_id, "transcript_json")) is None: return
    turns = _json(raw, list)
    if turns and turns[-1].get("role") == role and (turns[-1].get("qid") or "") == qid: turns[-1].update(text=engine.merge_transcript_chunk(turns[-1]["text"], text), ts=time.time())
    else: turns.append({"role": role, "text": text.strip(), "ts": time.time(), **({"qid": qid} if qid else {})})
    _write(call_id, transcript_json=json.dumps(turns, ensure_ascii=False))
def save_cursor_state(call_id: int, pack: PackConfig, state: CallState) -> None:
    name = next((str(state.profile[s]).strip() for s in pack.profile_slots if state.profile.get(s)), "")
    _write(call_id, cursor_json=json.dumps(state.to_dict(), ensure_ascii=False), patient_name=name, roster_summary=engine.roster_label_patient(pack, state))
def call_state_from_row(call: Mapping[str, Any] | None) -> CallState:
    raw = (call or {}).get("cursor_state") or (call or {}).get("cursor_json")
    raw = _json(raw, dict) if isinstance(raw, str) else raw
    try: return CallState.from_dict(raw) if isinstance(raw, Mapping) and raw else CallState()
    except ValidationError: return CallState()
def append_call_event(call_id: int, event_type: str, detail: str = "", meta: dict | None = None) -> dict:
    evt = {"t": round(time.time(), 3), "type": event_type, "detail": detail or "", "meta": meta or {}}
    if (raw := _read(call_id, "events_json")) is not None: _write(call_id, events_json=json.dumps((_json(raw, list) + [evt])[-_EVENT_CAP:], ensure_ascii=False))
    return evt
def update_call_telemetry(call_id: int, patch: dict[str, Any]) -> dict:
    if (raw := _read(call_id, "telemetry_json")) is None: return {}
    merged = _json(raw, dict) | (patch if isinstance(patch, dict) else {})
    _write(call_id, telemetry_json=json.dumps(merged, ensure_ascii=False))
    return merged
def save_report_text(call_id: int, summary_doctor: str = "", summary_patient: str = "", red_flags: list[str] | None = None, patient_name: str = "",
                     roster_summary: str = "") -> None:
    _write(call_id, summary_doctor=summary_doctor, summary_patient=summary_patient, red_flags_json=json.dumps(list(red_flags or []), ensure_ascii=False),
           patient_name=patient_name, roster_summary=roster_summary)
def complete_call(call_id: int, duration_sec: int, audio_path: str = "") -> None:
    _write(call_id, status="completed", duration_sec=int(duration_sec), audio_path=audio_path, ended_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
def create_bug_report(call_id: int = 0, reporter: str = "", category: str = "", note: str = "", call_elapsed_sec: int = 0,
                      extra_snapshot: dict | None = None, include_call_body: bool = False) -> dict:
    call = (get_call(call_id) if (owned := bool(include_call_body and call_id > 0)) else None) or {}
    tr = call.get("transcript")
    snap = {"last_transcript": tr[-8:] if isinstance(tr, list) else [], "telemetry": call.get("telemetry") or {},
            "recent_events": (call.get("events") or [])[-20:], "client": extra_snapshot or {}}
    with _LOCK:
        cur = conn.execute("INSERT INTO bug_reports (call_id, reporter, category, note, call_elapsed_sec, snapshot_json) VALUES (?, ?, ?, ?, ?, ?)",
                           (int(call_id), reporter, category, note, int(call_elapsed_sec), json.dumps(snap, ensure_ascii=False)))
        conn.commit()
    if owned: append_call_event(call_id, "bug_reported", f"{category}: {note}".strip(": "))
    return {"id": int(cur.lastrowid), "call_id": int(call_id), "reporter": reporter, "category": category, "note": note,
            "call_elapsed_sec": int(call_elapsed_sec), "snapshot": snap}
def list_bug_reports(limit: int = 50) -> list[dict]:
    with _LOCK:
        rows = conn.execute("SELECT * FROM bug_reports ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [{**{k: v for k, v in dict(r).items() if k != "snapshot_json"}, "snapshot": _json(r["snapshot_json"], dict)} for r in rows]
def get_call(call_id: int) -> dict | None:
    with _LOCK:
        if (row := conn.execute("SELECT * FROM calls WHERE id = ?", (int(call_id),)).fetchone()) is not None and not str(row["share_token"] or "").strip():
            conn.execute("UPDATE calls SET share_token = ? WHERE id = ?", (secrets.token_urlsafe(18), int(call_id)))
            conn.commit()
            row = conn.execute("SELECT * FROM calls WHERE id = ?", (int(call_id),)).fetchone()
    return _row(row)
def get_call_by_token(share_token: str) -> dict | None:
    tok = (share_token or "").strip()
    if len(tok) < _MIN_TOKEN: return None
    with _LOCK:
        return _row(conn.execute("SELECT * FROM calls WHERE share_token = ?", (tok,)).fetchone())
def list_calls(pack_id: str = "") -> list[dict]:
    where, args = (" WHERE pack_id = ?", (pack_id,)) if pack_id else ("", ())
    with _LOCK:
        return [_row(r) or {} for r in conn.execute(f"SELECT {_LISTED} FROM calls{where} ORDER BY id DESC", args).fetchall()]
