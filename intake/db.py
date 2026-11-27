"""SQLite storage for calls, transcripts, events, and bug reports (`intake/db.py`)."""
from __future__ import annotations
import json, os, secrets, sqlite3, threading, time, uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from . import engine
from .models import CallState, PackConfig, TurnOutcome
from .turns import CursorAction, DecisionReason, Disposition, DomainFit, EvidenceSource, PolicyDecision, StoredClinicalClaim, StoredPolicyDecision, StoredSemanticCommit, StoredTurnEvidence, StoredTurnInterpretation, TopicRelation, TurnEvidence, TurnInterpretation
DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "storage"))
DB_PATH = Path(DATA_DIR) / "intake.db"
_LOCK, _EVENT_CAP, _MIN_TOKEN = threading.Lock(), 600, 8
_JSON = (("transcript_json", "transcript", list), ("telemetry_json", "telemetry", dict),
         ("events_json", "events", list), ("cursor_json", "cursor_state", dict))
_LISTED = ("id, pack_id, patient_name, roster_summary, status, started_at, ended_at, duration_sec, audio_path, caller_name, "
           "user_role, model_used, voice_used, clinic_code, share_token")
_CALLS = ("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY AUTOINCREMENT, pack_id TEXT DEFAULT 'shishu-hi', patient_name TEXT DEFAULT '', roster_summary TEXT DEFAULT '', status TEXT DEFAULT 'active', "
          "started_at DATETIME DEFAULT CURRENT_TIMESTAMP, ended_at DATETIME, duration_sec INTEGER DEFAULT 0, audio_path TEXT DEFAULT '', transcript_json TEXT DEFAULT '[]', "
          "summary_report TEXT DEFAULT '', caller_name TEXT DEFAULT '', user_role TEXT DEFAULT 'user', model_used TEXT DEFAULT '', voice_used TEXT DEFAULT '', "
          "telemetry_json TEXT DEFAULT '{}', events_json TEXT DEFAULT '[]', cursor_json TEXT DEFAULT '', share_token TEXT DEFAULT '', clinic_code TEXT DEFAULT '')")
_BUGS = ("CREATE TABLE IF NOT EXISTS bug_reports (id INTEGER PRIMARY KEY AUTOINCREMENT, call_id INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
         "reporter TEXT DEFAULT '', category TEXT DEFAULT '', note TEXT DEFAULT '', call_elapsed_sec INTEGER DEFAULT 0, snapshot_json TEXT DEFAULT '{}')")
_TURN_EVIDENCE = ("CREATE TABLE IF NOT EXISTS turn_evidence (id INTEGER PRIMARY KEY AUTOINCREMENT, evidence_id TEXT NOT NULL UNIQUE, call_id INTEGER NOT NULL, "
                  "sequence INTEGER NOT NULL, source TEXT NOT NULL, snapshot_step TEXT NOT NULL, transcript TEXT NOT NULL DEFAULT '', received_at REAL NOT NULL, "
                  "selected_value_json TEXT, UNIQUE(call_id, sequence))")
_TURN_INTERPRETATIONS = ("CREATE TABLE IF NOT EXISTS turn_interpretations (id INTEGER PRIMARY KEY AUTOINCREMENT, interpretation_id TEXT NOT NULL UNIQUE, "
                         "call_id INTEGER NOT NULL, sequence INTEGER NOT NULL, evidence_id TEXT NOT NULL, interpretation_json TEXT NOT NULL, "
                         "UNIQUE(call_id, sequence))")
_CLAIMS = ("CREATE TABLE IF NOT EXISTS clinical_claims (id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT NOT NULL UNIQUE, call_id INTEGER NOT NULL, "
           "sequence INTEGER NOT NULL, claim_index INTEGER NOT NULL, target_qid TEXT, subject_ids_json TEXT NOT NULL DEFAULT '[]', relation TEXT NOT NULL, "
           "domain_fit TEXT NOT NULL, status TEXT, value_json TEXT, UNIQUE(call_id, sequence, claim_index))")
_POLICY_DECISIONS = ("CREATE TABLE IF NOT EXISTS policy_decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL UNIQUE, call_id INTEGER NOT NULL, "
                     "sequence INTEGER NOT NULL, claim_index INTEGER, disposition TEXT NOT NULL, reason TEXT NOT NULL, target_qid TEXT, cursor_action TEXT NOT NULL)")
for _sub in ("", "audio", "pdfs"): (Path(DATA_DIR) / _sub).mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
conn.row_factory = sqlite3.Row
_MIGRATIONS = (("summary_report", "TEXT DEFAULT ''"),)  # columns added after a database may already exist
def _migrate() -> None:
    """Add any column `CREATE TABLE IF NOT EXISTS` cannot: on a legacy file it is a no-op, and an old `calls`
    table keeps its own shape, so `save_report_text` failed with "no such column: summary_report"."""
    have = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
    for name, decl in _MIGRATIONS:
        if name not in have: conn.execute(f"ALTER TABLE calls ADD COLUMN {name} {decl}")
def init_db() -> None:
    """Create or upgrade the schema. Runs once at import, so opening a legacy database migrates it."""
    with _LOCK:
        for _sql in ("PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=5000", _CALLS, _BUGS, _TURN_EVIDENCE, _TURN_INTERPRETATIONS, _CLAIMS, _POLICY_DECISIONS,
                     "CREATE TABLE IF NOT EXISTS server_secrets (key TEXT PRIMARY KEY, value TEXT)",
                     "CREATE UNIQUE INDEX IF NOT EXISTS idx_calls_token ON calls(share_token) WHERE share_token != ''"): conn.execute(_sql)
        _migrate()
        conn.commit()
init_db()
def _json(raw: Any, default: type) -> Any:
    try: parsed = json.loads(raw) if raw and str(raw).strip() else None
    except (TypeError, ValueError): parsed = None
    return parsed if isinstance(parsed, default) else default()
def _write(call_id: int, **fields: Any) -> None:
    with _LOCK: _write_unlocked(call_id, **fields)
def _write_unlocked(call_id: int, **fields: Any) -> None:
    """Write columns with `_LOCK` assumed held, so a caller can read, mutate and write as one transaction."""
    conn.execute(f"UPDATE calls SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", (*fields.values(), int(call_id)))
    conn.commit()
def _read(call_id: int, col: str) -> Any:
    with _LOCK: return _read_unlocked(call_id, col)
def _read_unlocked(call_id: int, col: str) -> Any:
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
    if not (text or "").strip(): return
    with _LOCK:  # read, merge and write in one transaction, or a concurrent append overwrites this turn
        if (raw := _read_unlocked(call_id, "transcript_json")) is None: return
        turns = _json(raw, list)
        if turns and turns[-1].get("role") == role and (turns[-1].get("qid") or "") == qid: turns[-1].update(text=engine.merge_transcript_chunk(turns[-1]["text"], text), ts=time.time())
        else: turns.append({"role": role, "text": text.strip(), "ts": time.time(), **({"qid": qid} if qid else {})})
        _write_unlocked(call_id, transcript_json=json.dumps(turns, ensure_ascii=False))
def save_cursor_state(call_id: int, pack: PackConfig, state: CallState) -> None:
    name = next((str(state.profile[s]).strip() for s in pack.profile_slots if state.profile.get(s)), "")
    _write(call_id, cursor_json=json.dumps(state.to_dict(), ensure_ascii=False), patient_name=name, roster_summary=engine.roster_label_patient(pack, state))
def _optional_outcome(raw: Any) -> TurnOutcome | None:
    if raw is None or raw == "": return None
    try: return TurnOutcome(str(raw))
    except ValueError: return None
_EVIDENCE_COLS = "evidence_id, call_id, sequence, source, snapshot_step, transcript, received_at, selected_value_json"
def _col_json(raw: Any) -> Any | None:
    if raw is None: return None
    try: return json.loads(raw)
    except (TypeError, ValueError): return None
def _stored_turn_evidence(row: sqlite3.Row) -> StoredTurnEvidence:
    sv = _col_json(row["selected_value_json"]) if row["selected_value_json"] is not None else None
    return StoredTurnEvidence(evidence_id=str(row["evidence_id"]), call_id=int(row["call_id"]), sequence=int(row["sequence"]), source=EvidenceSource(str(row["source"])),
                            snapshot_step=str(row["snapshot_step"]), transcript=str(row["transcript"] or ""), received_at=float(row["received_at"]),
                            selected_value=sv if row["selected_value_json"] is not None else None)
def _insert_turn_base(cid: int, pack: PackConfig, state: CallState, evidence: TurnEvidence) -> tuple[str, int]:
    name = next((str(state.profile[s]).strip() for s in pack.profile_slots if state.profile.get(s)), "")
    cursor_json, roster_summary = json.dumps(state.to_dict(), ensure_ascii=False), engine.roster_label_patient(pack, state)
    evidence_id, sv_json = str(uuid.uuid4()), None if evidence.selected_value is None else json.dumps(evidence.selected_value, ensure_ascii=False)
    updated = conn.execute("UPDATE calls SET cursor_json = ?, patient_name = ?, roster_summary = ? WHERE id = ?", (cursor_json, name, roster_summary, cid))
    if int(updated.rowcount or 0) < 1:
        raise LookupError(f"call {cid} not found")
    sequence = int(conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 AS n FROM turn_evidence WHERE call_id = ?", (cid,)).fetchone()["n"])
    conn.execute(f"INSERT INTO turn_evidence ({_EVIDENCE_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (evidence_id, cid, sequence, evidence.source.value, evidence.snapshot_step, evidence.transcript, evidence.received_at, sv_json))
    return evidence_id, sequence
def _stored_clinical_claim(row: sqlite3.Row) -> StoredClinicalClaim:
    val = _col_json(row["value_json"])
    return StoredClinicalClaim(claim_id=str(row["claim_id"]), call_id=int(row["call_id"]), sequence=int(row["sequence"]), claim_index=int(row["claim_index"]),
                               target_qid=str(row["target_qid"]) if row["target_qid"] else None, subject_ids=tuple(str(x) for x in _json(row["subject_ids_json"], list)),
                               relation=TopicRelation(str(row["relation"])), domain_fit=DomainFit(str(row["domain_fit"])),
                               status=_optional_outcome(row["status"]), value=val if row["value_json"] is not None else None)
def _stored_policy_decision(row: sqlite3.Row) -> StoredPolicyDecision:
    ci = row["claim_index"]
    return StoredPolicyDecision(decision_id=str(row["decision_id"]), call_id=int(row["call_id"]), sequence=int(row["sequence"]), claim_index=int(ci) if ci is not None else None,
                                disposition=Disposition(str(row["disposition"])), reason=DecisionReason(str(row["reason"])),
                                target_qid=str(row["target_qid"]) if row["target_qid"] else None, cursor_action=CursorAction(str(row["cursor_action"])))
def _stored_turn_interpretation(row: sqlite3.Row) -> StoredTurnInterpretation:
    return StoredTurnInterpretation(interpretation_id=str(row["interpretation_id"]), call_id=int(row["call_id"]), sequence=int(row["sequence"]),
                                    evidence_id=str(row["evidence_id"]), interpretation=TurnInterpretation.model_validate(_json(row["interpretation_json"], dict)))
def _turn_txn(cid: int, call_id: int, body):
    if cid <= 0: raise LookupError(f"call {call_id} not found")
    with _LOCK:
        try: conn.execute("BEGIN IMMEDIATE"); out = body(); conn.commit(); return out
        except Exception: conn.rollback(); raise
def commit_turn(call_id: int, pack: PackConfig, state: CallState, evidence: TurnEvidence) -> StoredTurnEvidence:
    cid = int(call_id)
    evidence_id, sequence = _turn_txn(cid, call_id, lambda: _insert_turn_base(cid, pack, state, evidence))
    return StoredTurnEvidence.persisted(evidence, evidence_id=evidence_id, call_id=cid, sequence=sequence)
_CLAIM_INS, _DEC_INS = ("INSERT INTO clinical_claims (claim_id, call_id, sequence, claim_index, target_qid, subject_ids_json, relation, domain_fit, status, value_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        "INSERT INTO policy_decisions (decision_id, call_id, sequence, claim_index, disposition, reason, target_qid, cursor_action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
def _insert_semantic_children(cid: int, sequence: int, interpretation: TurnInterpretation, decisions: tuple[PolicyDecision, ...]) -> tuple[list[StoredClinicalClaim], list[StoredPolicyDecision]]:
    stored_claims: list[StoredClinicalClaim] = []
    for idx, claim in enumerate(interpretation.claims):
        claim_id = str(uuid.uuid4())
        conn.execute(_CLAIM_INS, (claim_id, cid, sequence, idx, claim.target_qid, json.dumps(list(claim.subject_ids), ensure_ascii=False),
                                  claim.relation.value, claim.domain_fit.value, claim.status.value if claim.status is not None else None,
                                  None if claim.value is None else json.dumps(claim.value, ensure_ascii=False)))
        stored_claims.append(StoredClinicalClaim(claim_id=claim_id, call_id=cid, sequence=sequence, claim_index=idx, target_qid=claim.target_qid,
                                                 subject_ids=claim.subject_ids, relation=claim.relation, domain_fit=claim.domain_fit, status=claim.status, value=claim.value))
    stored_decisions: list[StoredPolicyDecision] = []
    for decision in decisions:
        decision_id = str(uuid.uuid4())
        conn.execute(_DEC_INS, (decision_id, cid, sequence, decision.claim_index, decision.disposition.value, decision.reason.value,
                                decision.target_qid, decision.cursor_action.value))
        stored_decisions.append(StoredPolicyDecision(decision_id=decision_id, call_id=cid, sequence=sequence, claim_index=decision.claim_index,
                                                     disposition=decision.disposition, reason=decision.reason, target_qid=decision.target_qid, cursor_action=decision.cursor_action))
    return stored_claims, stored_decisions
def commit_semantic_turn(call_id: int, pack: PackConfig, state: CallState, evidence: TurnEvidence, interpretation: TurnInterpretation, decisions: tuple[PolicyDecision, ...]) -> StoredSemanticCommit:
    cid, interpretation_id = int(call_id), str(uuid.uuid4())
    def body():
        evidence_id, sequence = _insert_turn_base(cid, pack, state, evidence)
        conn.execute("INSERT INTO turn_interpretations (interpretation_id, call_id, sequence, evidence_id, interpretation_json) VALUES (?, ?, ?, ?, ?)",
                     (interpretation_id, cid, sequence, evidence_id, json.dumps(interpretation.model_dump(mode="json"), ensure_ascii=False)))
        return evidence_id, sequence, _insert_semantic_children(cid, sequence, interpretation, decisions)
    evidence_id, sequence, (stored_claims, stored_decisions) = _turn_txn(cid, call_id, body)
    stored_evidence = StoredTurnEvidence.persisted(evidence, evidence_id=evidence_id, call_id=cid, sequence=sequence)
    stored_interp = StoredTurnInterpretation(interpretation_id=interpretation_id, call_id=cid, sequence=sequence, evidence_id=evidence_id, interpretation=interpretation)
    return StoredSemanticCommit(call_id=cid, sequence=sequence, evidence=stored_evidence, interpretation=stored_interp, claims=tuple(stored_claims), decisions=tuple(stored_decisions))
def _list_for_call(table: str, order: str, call_id: int, mapfn):
    with _LOCK:
        rows = conn.execute(f"SELECT * FROM {table} WHERE call_id = ? ORDER BY {order}", (int(call_id),)).fetchall()
    return [mapfn(r) for r in rows]
list_turn_evidence = lambda call_id: _list_for_call("turn_evidence", "sequence", call_id, _stored_turn_evidence)
list_turn_interpretations = lambda call_id: _list_for_call("turn_interpretations", "sequence", call_id, _stored_turn_interpretation)
list_clinical_claims = lambda call_id: _list_for_call("clinical_claims", "sequence, claim_index", call_id, _stored_clinical_claim)
list_policy_decisions = lambda call_id: _list_for_call("policy_decisions", "sequence, id", call_id, _stored_policy_decision)
def list_retained_observations(call_id: int) -> list[StoredTurnEvidence]:
    cid = int(call_id)
    with _LOCK:
        ev = conn.execute("SELECT * FROM turn_evidence WHERE call_id = ? ORDER BY sequence", (cid,)).fetchall()
        keep = {int(r[0]) for r in conn.execute("SELECT DISTINCT sequence FROM policy_decisions WHERE call_id = ? AND disposition = 'retain_observation'", (cid,))}
    out = []
    for i, row in enumerate(ev):
        if int(row["sequence"]) not in keep: continue
        text = str(row["transcript"] or "")
        for nxt in ev[i + 1:]:
            if nxt["source"] != "late_stt" or nxt["snapshot_step"] != row["snapshot_step"]: break
            text = engine.merge_transcript_chunk(text, str(nxt["transcript"] or ""))
        out.append(_stored_turn_evidence(row).model_copy(update={"transcript": text}))
    return out
def call_state_from_row(call: Mapping[str, Any] | None) -> CallState:
    raw = (call or {}).get("cursor_state") or (call or {}).get("cursor_json")
    raw = _json(raw, dict) if isinstance(raw, str) else raw
    try: return CallState.from_dict(raw) if isinstance(raw, Mapping) and raw else CallState()
    except ValidationError: return CallState()
def append_call_event(call_id: int, event_type: str, detail: str = "", meta: dict | None = None) -> dict:
    evt = {"t": round(time.time(), 3), "type": event_type, "detail": detail or "", "meta": meta or {}}
    with _LOCK:  # read, append and write in one transaction, or a concurrent append drops this event
        if (raw := _read_unlocked(call_id, "events_json")) is not None:
            _write_unlocked(call_id, events_json=json.dumps((_json(raw, list) + [evt])[-_EVENT_CAP:], ensure_ascii=False))
    return evt
def update_call_telemetry(call_id: int, patch: dict[str, Any]) -> dict:
    with _LOCK:  # read, merge and write in one transaction, or a concurrent patch is lost
        if (raw := _read_unlocked(call_id, "telemetry_json")) is None: return {}
        merged = _json(raw, dict) | (patch if isinstance(patch, dict) else {})
        _write_unlocked(call_id, telemetry_json=json.dumps(merged, ensure_ascii=False))
    return merged
def save_report_text(call_id: int, summary_report: str = "", patient_name: str = "", roster_patient: str = "") -> None:
    """Store the one report a finished call is read from: the same Markdown the dashboard shows and Copy hands out."""
    _write(call_id, summary_report=summary_report, patient_name=patient_name, roster_summary=roster_patient)
def complete_call(call_id: int, duration_sec: int, audio_path: str = "") -> None:
    _write(call_id, status="completed", duration_sec=int(duration_sec), audio_path=audio_path, ended_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
def reopen_call(call_id: int, active_qid: str = "") -> None:
    """Put a completed call back in progress after the caller reconnects, keeping its audio path.

    `active_qid` is the step the browser will be handed. On rejoin, the assistant turn at the transcript tail for that
    qid is dropped in the same atomic write so the resumed caption does not double (Call #57 `T[66]`, 102→205 chars).
    """
    with _LOCK:  # read the transcript and reopen in one transaction, or a concurrent append is dropped
        if active_qid and (raw := _read_unlocked(call_id, "transcript_json")) is not None:
            turns = _json(raw, list)
            if (turns and turns[-1].get("role") == "assistant" and (turns[-1].get("qid") or "") == active_qid):
                _write_unlocked(call_id, status="active", ended_at=None, transcript_json=json.dumps(turns[:-1], ensure_ascii=False))
                return
        _write_unlocked(call_id, status="active", ended_at=None)
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