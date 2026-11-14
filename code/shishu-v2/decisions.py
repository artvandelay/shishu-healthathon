"""Gemini Decisions API client, declarative turn schema builder, and interpreter."""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import call_cursor
from call_cursor import children_roster, question_hi_for, routable_qids
from triage_catalog import get_spec

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DECISIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
DECISIONS_MODEL = "gemini-decision-1"
_SCHEMA: dict[str, Any] = json.loads(
    Path(__file__).resolve().with_name("decision_schema.json").read_text(encoding="utf-8")
)

_QUESTION_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RETRYABLE_HTTP = frozenset({408, 429})
_SAFETY_CODES = frozenset(
    {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}
)
VALID_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract"})
VALID_SIDE_INTENTS = frozenset({"add_concern", "clear_or_retract", "clarify_other"})
_ROUTABLE_QID_SET = frozenset(routable_qids())
_EPS = 1e-12

_NOUL_FIELDS = (
    "danger_now_min_yes",
    "caller_stop_min_yes",
    "mentions_profile_min_yes",
    "detail_present_min_yes",
)
_CONF_FIELDS = (
    "outcome_min_confidence",
    "side_intent_min_confidence",
    "side_qid_min_confidence",
    "child_ref_min_confidence",
)
_MARGIN_FIELDS = (
    "outcome_min_margin",
    "side_intent_min_margin",
    "side_qid_min_margin",
    "child_ref_min_margin",
)
_ALL_NUMERIC_THRESHOLD_FIELDS = _NOUL_FIELDS + _CONF_FIELDS + _MARGIN_FIELDS

TransportFn = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]


@dataclass(frozen=True, slots=True)
class DecisionCallResult:
    ok: bool
    error_reason: str
    attempts: int
    decision_ms: int
    request_bytes: int
    question_count: int
    catalog_option_count: int
    usage_present: bool
    total_input_tokens: int | None
    raw_answers: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    calibrated: bool = False
    danger_now_min_yes: float = 1.0
    caller_stop_min_yes: float = 1.0
    mentions_profile_min_yes: float = 1.0
    detail_present_min_yes: float = 1.0
    outcome_min_confidence: float = 1.0
    outcome_min_margin: float = 1.0
    side_intent_min_confidence: float = 1.0
    side_intent_min_margin: float = 1.0
    side_qid_min_confidence: float = 1.0
    side_qid_min_margin: float = 1.0
    child_ref_min_confidence: float = 1.0
    child_ref_min_margin: float = 1.0


UNCALIBRATED_THRESHOLDS = DecisionThresholds(calibrated=False)


@dataclass(frozen=True, slots=True)
class ChoiceEvaluation:
    accepted: bool
    winner: str
    top_option: str
    top_prob: float | None
    runnerup_option: str
    runnerup_prob: float | None
    margin: float | None
    reject_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    cursor_payload: dict[str, Any]
    needs_profile_extraction: bool
    future_qid_match: dict[str, Any] | None
    diagnostics: dict[str, Any]


def turn_view(state: dict) -> tuple[str, str, str, str]:
    """Return (active_qid, step_kind, active_prompt, followup_prompt) for state."""
    return (
        str(state.get("active_qid") or ""),
        call_cursor.step_kind(state),
        call_cursor.active_prompt_hi(state),
        call_cursor.followup_prompt_hi(state),
    )


def _side_qid_question(state: dict) -> dict[str, Any]:
    t = _SCHEMA["side_qid_template"]
    opt = t["option_template"]
    criteria: dict[str, str] = {}
    for qid in routable_qids():
        if qid == "ever_pregnant":
            q_hi, need_en = call_cursor.ever_pregnant_prompt_hi(), t["ever_pregnant_need_en"]
        else:
            spec = get_spec(qid)
            q_hi, need_en = question_hi_for(state, spec), spec.need_en
        criteria[qid] = opt.format(question_hi=q_hi, need_en=need_en)
    criteria["other"] = t["other_description"]
    return {"type": t["type"], "instructions": t["instructions"], "criteria": criteria}


def _child_ref_question(roster: list[dict[str, Any]]) -> dict[str, Any]:
    t = _SCHEMA["child_ref_template"]
    criteria = {
        r["id"]: t["option_template"].format(
            id=r["id"],
            rank=r["rank"],
            age_label_hi=r["age_label_hi"],
            sex=r["sex"],
            name_or_unnamed=r.get("name") or t["unnamed_label"],
        )
        for r in roster
    }
    criteria["all"] = t["all_description"]
    criteria["other"] = t["other_description"]
    return {"type": t["type"], "instructions": t["instructions"], "criteria": criteria}


def build_turn_request(
    state: dict,
    *,
    utterance_text: str = "",
    audio_wav_bytes: bytes | None = None,
    audio_mime_type: str = "audio/wav",
) -> dict[str, Any]:
    """Build a single-pass Gemini Decisions API request for one caller turn."""
    active_qid, step_kind, active_prompt, followup_prompt = turn_view(state)
    parent_prompt = ""
    if step_kind in ("which_child", "followup") and active_qid and active_qid != "ever_pregnant":
        parent_prompt = question_hi_for(state, get_spec(active_qid))
    _, last_prompt_hi = call_cursor.last_answered_prompt_hi(state)
    roster = children_roster(state)
    audio_data = bytes(audio_wav_bytes) if audio_wav_bytes else b""
    clean_utt = " ".join(str(utterance_text or "").split())
    if not clean_utt and audio_data:
        clean_utt = "[spoken audio attached]"

    roster_rows: list[dict[str, Any]] | None = None
    if len(roster) >= 2:
        roster_rows = [
            {
                "id": r["id"],
                "age": r["age_label_hi"],
                "sex": r["sex"],
                "rank": r["rank"],
                **({"name": r["name"]} if r.get("name") else {}),
            }
            for r in roster
        ]

    ctx_items = (
        ("active_question", active_prompt),
        ("parent_question", parent_prompt),
        ("followup_question", followup_prompt),
        ("last_answered_question", last_prompt_hi),
        ("children_roster", roster_rows),
        ("caller_utterance", clean_utt),
    )
    ctx = {
        k: v
        for k, v in ctx_items
        if v is not None and (v != "" or k in ("active_question", "caller_utterance"))
    }
    ctx_json = json.dumps(ctx, ensure_ascii=False)
    if audio_data:
        req_input: str | list[dict[str, Any]] = [
            {"type": "text", "text": ctx_json + "\n"},
            {
                "type": "audio",
                "data": base64.b64encode(audio_data).decode("ascii"),
                "mime_type": audio_mime_type,
            },
        ]
    else:
        req_input = ctx_json

    qs = _SCHEMA["questions"]
    questions: dict[str, dict[str, Any]] = {
        "danger_now": qs["danger_now"],
        "caller_stop": qs["caller_stop"],
        "mentions_profile": qs["mentions_profile"],
        "current_outcome": qs["current_outcome"],
    }
    if step_kind not in ("profile", "bridge") or call_cursor.has_answered_gates(state):
        questions["side_intent"] = qs["side_intent"]
        questions["side_qid"] = _side_qid_question(state)
    if len(roster) >= 2:
        questions["child_ref"] = _child_ref_question(roster)
    if step_kind in ("main", "close") and bool(followup_prompt):
        questions["detail_present"] = qs["detail_present"]

    for qkey in questions:
        if not _QUESTION_KEY_RE.match(qkey):
            raise ValueError(f"invalid Decisions API question key: {qkey!r}")
    return {"model": DECISIONS_MODEL, "input": req_input, "questions": questions}


def _urllib_transport(
    url: str, body_bytes: bytes, headers: dict[str, str], timeout_sec: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return int(getattr(resp, "status", 200)), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() if hasattr(exc, "read") else b""


def _parse_json_body(raw_body: bytes) -> dict[str, Any] | None:
    if not isinstance(raw_body, (bytes, bytearray)):
        return None
    text = bytes(raw_body).decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_usage(payload: dict[str, Any]) -> tuple[bool, int | None]:
    raw = payload.get("usage")
    if not isinstance(raw, dict) or not raw:
        return False, None
    in_tok = raw.get("total_input_tokens")
    if isinstance(in_tok, int) and not isinstance(in_tok, bool):
        return True, in_tok
    tot_tok = raw.get("total_tokens")
    if isinstance(tot_tok, int) and not isinstance(tot_tok, bool):
        return True, None
    return False, None


def _is_safety_error(payload: dict[str, Any] | None) -> bool:
    err = payload.get("error") if isinstance(payload, dict) else None
    code = str(err.get("code") or "").strip().lower() if isinstance(err, dict) else ""
    return code in _SAFETY_CODES


async def evaluate_turn(
    request: dict[str, Any],
    *,
    api_key: str | None = None,
    endpoint: str = DECISIONS_ENDPOINT,
    max_attempts: int = 2,
    attempt_timeout_sec: float = 2.0,
    total_budget_sec: float = 4.0,
    transport: TransportFn | None = None,
) -> DecisionCallResult:
    """Execute one Decisions API request with a bounded retry budget."""
    qs = request.get("questions") if isinstance(request, dict) else {}
    q_count = len(qs) if isinstance(qs, dict) else 0
    side_obj = qs.get("side_qid") if isinstance(qs, dict) else None
    side_c = side_obj.get("criteria") if isinstance(side_obj, dict) else {}
    cat_count = len(side_c) if isinstance(side_c, dict) else 0
    body_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8")
    req_bytes = len(body_bytes)
    raw_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    key = str(raw_key or "").strip()

    def _res(
        ok: bool,
        err: str,
        att: int,
        ms: int,
        u_ok: bool = False,
        tok: int | None = None,
        ans: dict[str, Any] | None = None,
    ) -> DecisionCallResult:
        return DecisionCallResult(
            ok, err, att, ms, req_bytes, q_count, cat_count, u_ok, tok, ans or {}
        )

    if not key and (api_key is not None or transport is None):
        return _res(False, "missing_api_key", 0, 0)

    headers = {"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": key}
    fn = transport or _urllib_transport
    max_tries = max(1, int(max_attempts))
    per_att = max(0.01, float(attempt_timeout_sec))
    started = time.perf_counter()
    deadline = started + max(0.01, float(total_budget_sec))
    attempts, last_err, u_ok, in_tok = 0, "unknown_error", False, None

    for attempt in range(1, max_tries + 1):
        rem = deadline - time.perf_counter()
        if rem <= 0:
            if last_err == "unknown_error":
                last_err = "timeout"
            break
        attempts = attempt
        this_timeout = min(per_att, rem)
        try:
            status, raw_body = await asyncio.wait_for(
                asyncio.to_thread(fn, endpoint, body_bytes, headers, this_timeout),
                timeout=this_timeout,
            )
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw_body = exc.read() if hasattr(exc, "read") else b""
        except BaseException as exc:
            is_to = isinstance(exc, (TimeoutError, asyncio.TimeoutError, socket.timeout)) or (
                isinstance(exc, urllib.error.URLError)
                and isinstance(exc.reason, (TimeoutError, socket.timeout))
            )
            is_conn = isinstance(
                exc, (urllib.error.URLError, ConnectionError, OSError, http.client.HTTPException)
            )
            if is_to:
                last_err = "timeout"
            elif is_conn:
                last_err = "connection_error"
            else:
                last_err = "malformed_response"
            if (is_to or is_conn) and attempt < max_tries:
                continue
            break

        if status in _RETRYABLE_HTTP or 500 <= status <= 599:
            last_err = f"http_{status}"
            if attempt < max_tries:
                continue
            break
        if 400 <= status <= 499:
            last_err = "safety_block" if _is_safety_error(_parse_json_body(raw_body)) else f"http_{status}"
            break
        parsed = _parse_json_body(raw_body) if 200 <= status <= 299 else None
        if not (200 <= status <= 299):
            last_err = f"http_{status}"
            break
        if parsed is None:
            last_err = "malformed_response"
            break
        u_ok, in_tok = _extract_usage(parsed)
        answers = parsed.get("answers")
        if answers is None or (isinstance(answers, dict) and not answers):
            last_err = "empty_answers"
        elif not isinstance(answers, dict):
            last_err = "malformed_response"
        else:
            ms = max(0, int(round((time.perf_counter() - started) * 1000)))
            return _res(True, "", attempts, ms, u_ok, in_tok, answers)
        break

    ms = max(0, int(round((time.perf_counter() - started) * 1000)))
    return _res(False, last_err, attempts, ms, u_ok, in_tok)


def validate_thresholds(
    thresholds: DecisionThresholds, *, require_calibrated: bool = True
) -> DecisionThresholds:
    """Validate DecisionThresholds ranges and enforce calibration when required."""
    if not isinstance(thresholds, DecisionThresholds):
        raise TypeError("thresholds must be a DecisionThresholds instance")
    if require_calibrated and not thresholds.calibrated:
        raise ValueError(
            "Uncalibrated DecisionThresholds refused; "
            "load calibrated thresholds via load_thresholds()."
        )
    if thresholds.calibrated:
        bounds = (
            (_NOUL_FIELDS, 0.01, 0.99, False),
            (_CONF_FIELDS, 0.0, 1.0, False),
            (_MARGIN_FIELDS, 0.0, 1.0, True),
        )
        for fields, lo, hi, inclusive_lo in bounds:
            for f in fields:
                v = float(getattr(thresholds, f))
                in_range = (lo <= v < hi) if inclusive_lo else (lo < v < hi)
                if not math.isfinite(v) or not in_range:
                    raise ValueError(f"Calibrated threshold {f}={v!r} out of range")
    return thresholds


def load_thresholds(path: str | Path, require_calibrated: bool = False) -> DecisionThresholds:
    """Load and validate a DecisionThresholds JSON artifact."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("thresholds"), dict):
        data = data["thresholds"]
    if not isinstance(data, dict):
        raise ValueError("Threshold artifact must be a JSON object")
    calibrated = bool(data.get("calibrated", False))
    if calibrated:
        missing = [f for f in _ALL_NUMERIC_THRESHOLD_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Calibrated threshold artifact missing fields: {', '.join(missing)}")
    kwargs = {
        f: float(data[f]) if f in data else float(getattr(UNCALIBRATED_THRESHOLDS, f))
        for f in _ALL_NUMERIC_THRESHOLD_FIELDS
    }
    return validate_thresholds(
        DecisionThresholds(calibrated=calibrated, **kwargs),
        require_calibrated=require_calibrated,
    )


def _noul_prob(raw_answers: dict[str, Any], key: str) -> float | None:
    ans = raw_answers.get(key) if isinstance(raw_answers, dict) else None
    val = ans.get("noul") if isinstance(ans, dict) else None
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return float(val)
    return None


def evaluate_choice(
    raw_answers: dict[str, Any],
    key: str,
    *,
    catchall_key: str = "other",
    min_confidence: float,
    min_margin: float,
    allowed_winners: frozenset[str] | None = None,
) -> ChoiceEvaluation:
    """Accept a choice answer when top probability beats min_confidence and runner-up."""
    ans = raw_answers.get(key) if isinstance(raw_answers, dict) else None
    raw_probs = ans.get("probabilities") if isinstance(ans, dict) else None
    probs: dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for opt, prob in raw_probs.items():
            if (
                isinstance(prob, (int, float))
                and not isinstance(prob, bool)
                and math.isfinite(prob)
            ):
                probs[str(opt)] = float(prob)
    if not probs:
        return ChoiceEvaluation(False, "", "", None, "", None, None, "missing_answer")

    ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    top_opt, top_prob = ranked[0]
    ru_opt, ru_prob = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = top_prob - ru_prob

    if top_opt == catchall_key:
        reason = "catchall_top"
    elif allowed_winners is not None and top_opt not in allowed_winners:
        reason = "disallowed_option"
    elif top_prob + _EPS < min_confidence:
        reason = "below_min_confidence"
    elif margin + _EPS < min_margin:
        reason = "below_margin"
    else:
        reason = ""
    return ChoiceEvaluation(
        not reason, "" if reason else top_opt, top_opt, top_prob, ru_opt, ru_prob, margin, reason
    )


def interpret_turn_decision(
    state: dict,
    raw_answers: dict[str, Any],
    *,
    utterance_text: str = "",
    thresholds: DecisionThresholds = UNCALIBRATED_THRESHOLDS,
    require_calibrated: bool = False,
) -> InterpretationResult:
    """Translate raw Decisions API probabilities into a cursor payload and telemetry."""
    validate_thresholds(thresholds, require_calibrated=require_calibrated)
    answers = raw_answers if isinstance(raw_answers, dict) else {}
    clean_utt = " ".join(str(utterance_text or "").split())
    active_qid, step_kind, _, followup_prompt = turn_view(state)
    roster = children_roster(state)
    needs_detail = step_kind in ("main", "close") and bool(followup_prompt)
    has_side = step_kind not in ("profile", "bridge") or call_cursor.has_answered_gates(state)

    danger_p = _noul_prob(answers, "danger_now")
    stop_p = _noul_prob(answers, "caller_stop")
    profile_p = _noul_prob(answers, "mentions_profile")
    detail_p = _noul_prob(answers, "detail_present") if needs_detail else None
    emergency = danger_p is not None and danger_p >= thresholds.danger_now_min_yes
    caller_stop = (not emergency) and (
        stop_p is not None and stop_p >= thresholds.caller_stop_min_yes
    )
    mentions_profile = profile_p is not None and profile_p >= thresholds.mentions_profile_min_yes

    out_eval = evaluate_choice(
        answers,
        "current_outcome",
        min_confidence=thresholds.outcome_min_confidence,
        min_margin=thresholds.outcome_min_margin,
        allowed_winners=VALID_OUTCOMES,
    )
    resolved_outcome = out_eval.winner if out_eval.accepted else "not_about_active"
    needs_profile = bool(
        mentions_profile or (step_kind == "profile" and resolved_outcome != "clarify")
    )

    intent_eval = (
        evaluate_choice(
            answers,
            "side_intent",
            min_confidence=thresholds.side_intent_min_confidence,
            min_margin=thresholds.side_intent_min_margin,
            allowed_winners=VALID_SIDE_INTENTS,
        )
        if has_side
        else None
    )
    qid_eval = (
        evaluate_choice(
            answers,
            "side_qid",
            min_confidence=thresholds.side_qid_min_confidence,
            min_margin=thresholds.side_qid_min_margin,
            allowed_winners=_ROUTABLE_QID_SET,
        )
        if has_side
        else None
    )
    child_eval = (
        evaluate_choice(
            answers,
            "child_ref",
            min_confidence=thresholds.child_ref_min_confidence,
            min_margin=thresholds.child_ref_min_margin,
            allowed_winners=frozenset(r["id"] for r in roster) | {"all"},
        )
        if len(roster) >= 2
        else None
    )
    child_ref = child_eval.winner if (child_eval is not None and child_eval.accepted) else ""

    detail_present = (
        needs_detail and detail_p is not None and detail_p >= thresholds.detail_present_min_yes
    )
    followup_answered = bool(detail_present and resolved_outcome == "concern")

    side_qid, side_action, side_outcome = "", "", ""
    future_match: dict[str, Any] | None = None
    if (
        not emergency
        and not caller_stop
        and intent_eval is not None
        and qid_eval is not None
        and intent_eval.accepted
        and qid_eval.accepted
    ):
        cand_qid, intent = qid_eval.winner, intent_eval.winner
        if intent == "clarify_other":
            m_act, m_out = "clarify_meaning", ""
        else:
            m_act = "add_or_correct"
            m_out = "concern" if intent == "add_concern" else "retract"
        accepted = call_cursor.accepted_target(state, cand_qid)
        if accepted:
            side_qid, side_action, side_outcome = accepted, m_act, m_out
        elif cand_qid != active_qid and not call_cursor.is_answered_or_na(state, cand_qid):
            future_match = {"qid": cand_qid, "confidence": qid_eval.top_prob, "act": m_act}

    active_ans = (state.get("answers") or {}).get(active_qid) or {}
    active_can_retract = (
        state.get("pending_followup_idx") is not None or active_ans.get("state") == "concern"
    )
    if side_qid:
        clarify_conflict = side_action == "clarify_meaning" and resolved_outcome == "clarify"
        retract_conflict = resolved_outcome == "retract" and not active_can_retract
        if clarify_conflict or retract_conflict:
            resolved_outcome = "not_about_active"

    if needs_detail and step_kind in ("main", "close"):
        outcome_keeps_note = detail_present and resolved_outcome == "concern"
    else:
        outcome_keeps_note = resolved_outcome == "concern" or (
            step_kind == "which_child" and resolved_outcome == "unknown"
        )
    active_note = clean_utt if (emergency or outcome_keeps_note) else ""
    side_note = clean_utt if (side_qid and side_action == "add_or_correct") else ""

    payload: dict[str, Any] = {
        "emergency": emergency,
        "caller_stop": caller_stop,
        "current_outcome": resolved_outcome,
        "child_ref": child_ref,
        "note": (side_note or active_note) if side_qid else active_note,
        "followup_answered": followup_answered,
    }
    if side_qid:
        payload.update(
            {
                "current_note": active_note,
                "side_qid": side_qid,
                "side_action": side_action,
                "side_outcome": side_outcome,
                "side_note": side_note,
            }
        )

    raw_side: dict[str, Any] | None = None
    if qid_eval is not None and qid_eval.top_option not in ("", "other"):
        raw_intent = ""
        if intent_eval is not None and intent_eval.top_option not in ("", "other"):
            raw_intent = intent_eval.top_option
        raw_side = {
            "qid": qid_eval.top_option,
            "confidence": qid_eval.top_prob,
            "margin": qid_eval.margin,
            "side_intent": raw_intent,
            "side_intent_prob": intent_eval.top_prob if intent_eval is not None else None,
        }

    diagnostics = {
        "active_qid": active_qid,
        "step_kind": step_kind,
        "calibrated": thresholds.calibrated,
        "danger_now": danger_p,
        "caller_stop": stop_p,
        "mentions_profile": profile_p,
        "detail_present": detail_p,
        "current_outcome": out_eval.as_dict(),
        "side_intent": intent_eval.as_dict() if intent_eval is not None else None,
        "side_qid": qid_eval.as_dict() if qid_eval is not None else None,
        "child_ref": child_eval.as_dict() if child_eval is not None else None,
        "future_qid_match": future_match,
        "raw_side_candidate": raw_side,
    }
    return InterpretationResult(payload, needs_profile, future_match, diagnostics)


def interpret_for_authority(
    state: dict,
    raw_answers: dict[str, Any],
    *,
    utterance_text: str = "",
    thresholds: DecisionThresholds = UNCALIBRATED_THRESHOLDS,
    require_calibrated: bool = True,
) -> InterpretationResult:
    """Authority-mode wrapper requiring calibrated thresholds by default."""
    return interpret_turn_decision(
        state,
        raw_answers,
        utterance_text=utterance_text,
        thresholds=thresholds,
        require_calibrated=require_calibrated,
    )
