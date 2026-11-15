"""Gemini Decisions API client, turn-schema builder, and turn interpreter.

One caller turn in, one typed ``TurnCommand`` out. Probability gates and side
routes are decided here; ``engine.reduce_turn`` alone owns note retention.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, NamedTuple

from dotenv import load_dotenv
from pydantic import ConfigDict, Field

import copy_hi
import engine
from models import AnswerStatus, CallState, SideAction, Strict, SubStep, TurnCommand, TurnOutcome
from triage_catalog import get_spec, routable_specs

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DECISIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
DECISIONS_MODEL = "gemini-decision-1"
VALID_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract"})
VALID_SIDE_INTENTS = frozenset({"add_concern", "clear_or_retract", "clarify_other"})
TransportFn = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]
_SCHEMA = json.loads(
    Path(__file__).resolve().with_name("decision_schema.json").read_text(encoding="utf-8")
)
_THRESHOLDS_PATH = Path(__file__).resolve().with_name("decision_thresholds.json")
_CORE_QUESTIONS = ("danger_now", "caller_stop", "mentions_profile", "current_outcome")
_SIDE_QUESTIONS = ("mentions_diagnosis", "side_intent")
_GATES = {
    "danger_now": "danger_now_min_yes",
    "caller_stop": "caller_stop_min_yes",
    "mentions_profile": "mentions_profile_min_yes",
    "mentions_diagnosis": "mentions_diagnosis_min_yes",
    "detail_present": "detail_present_min_yes",
}
_KEEP_IN_CONTEXT = frozenset({"active_question", "caller_utterance"})
_AUDIO_ONLY_TEXT = "[spoken audio attached]"
_RETRYABLE = frozenset({408, 429})
_SAFETY_CODES = frozenset(
    {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}
)
_SIDE_ACTIONS = {
    "add_concern": SideAction.ADD_CONCERN,
    "clear_or_retract": SideAction.CLEAR_OR_RETRACT,
    "clarify_other": SideAction.CLARIFY_MEANING,
}
_ROUTABLE_QIDS = frozenset(spec.id for spec in routable_specs())
_THRESHOLD_CACHE: dict[tuple[str, int], DecisionThresholds] = {}
_EPS = 1e-12


class DecisionCallResult(NamedTuple):
    """One Decisions API attempt's outcome and transport telemetry."""

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


class ChoiceEvaluation(NamedTuple):
    """One choice head's verdict, with probabilities kept for calibration."""

    accepted: bool
    winner: str
    top_option: str
    top_prob: float | None
    runnerup_option: str
    runnerup_prob: float | None
    margin: float | None
    reject_reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self._asdict())


class DecisionThresholds(Strict, frozen=True):
    """Calibrated gates: strict, immutable, and NaN-free by construction."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    danger_now_min_yes: Annotated[float, Field(gt=0.01, lt=0.99)]
    caller_stop_min_yes: Annotated[float, Field(gt=0.01, lt=0.99)]
    mentions_profile_min_yes: Annotated[float, Field(gt=0.01, lt=0.99)]
    mentions_diagnosis_min_yes: Annotated[float, Field(gt=0.01, lt=0.99)]
    detail_present_min_yes: Annotated[float, Field(gt=0.01, lt=0.99)]
    outcome_min_confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    side_intent_min_confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    side_qid_min_confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    child_ref_min_confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    outcome_min_margin: Annotated[float, Field(ge=0.0, lt=1.0)]
    side_intent_min_margin: Annotated[float, Field(ge=0.0, lt=1.0)]
    side_qid_min_margin: Annotated[float, Field(ge=0.0, lt=1.0)]
    child_ref_min_margin: Annotated[float, Field(ge=0.0, lt=1.0)]


class TurnIntent(Strict, frozen=True):
    """One interpreted turn: a reducer-ready command plus its telemetry."""

    command: TurnCommand
    needs_profile_extraction: bool
    diagnostics: dict[str, Any]
    future_qid_match: dict[str, Any] | None = None


def _step(state: CallState) -> tuple[str, bool, bool]:
    """Step kind, whether the detail head applies, and whether the side heads do."""
    kind = engine.step_kind(state)
    needs_detail = kind in ("main", "close") and bool(engine.followup_prompt_hi(state))
    sideways = kind not in ("profile", "bridge") or engine.has_answered_gates(state)
    return kind, needs_detail, sideways


def _head(template_key: str, options: dict[str, str]) -> dict[str, Any]:
    """A choice head over rendered options, plus the schema's own catchall."""
    template = _SCHEMA[template_key]
    head = {"type": template["type"], "instructions": template["instructions"]}
    head["criteria"] = {**options, "other": template["other_description"]}
    return head


def _side_qid_head(state: CallState) -> dict[str, Any]:
    """Every routable item as a side option, worded for this caller."""
    template = _SCHEMA["side_qid_template"]
    options = {
        spec.id: template["option_template"].format(
            question_hi=copy_hi.question_hi_for(state.profile, spec), need_en=spec.need_en
        )
        for spec in routable_specs()
    }
    return _head("side_qid_template", options)


def _child_ref_head(roster: list[dict[str, Any]]) -> dict[str, Any]:
    """Every roster child as a pickable option, plus all-of-them."""
    template = _SCHEMA["child_ref_template"]
    options = {
        child["id"]: template["option_template"].format(
            id=child["id"],
            rank=child["rank"],
            age_label_hi=child["age_label_hi"],
            sex=child["sex"],
            name_or_unnamed=child.get("name") or template["unnamed_label"],
        )
        for child in roster
    }
    return _head("child_ref_template", options | {"all": template["all_description"]})


def _roster_rows(roster: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Compact roster context, present only when a child must be disambiguated."""
    if len(roster) < 2:
        return None
    rows = []
    for child in roster:
        row = {"id": child["id"], "age": child["age_label_hi"], "rank": child["rank"]}
        row["sex"] = child["sex"]
        if child.get("name"):
            row["name"] = child["name"]
        rows.append(row)
    return rows


def build_turn_request(
    state: CallState,
    *,
    utterance_text: str = "",
    audio_wav_bytes: bytes | None = None,
    audio_mime_type: str = "audio/wav",
) -> dict[str, Any]:
    """Build a single-pass Gemini Decisions API request for one caller turn."""
    kind, needs_detail, has_side = _step(state)
    roster = engine.children_roster(state)
    audio = bytes(audio_wav_bytes) if audio_wav_bytes else b""
    utterance = " ".join(str(utterance_text or "").split()) or (_AUDIO_ONLY_TEXT if audio else "")
    roster_rows = _roster_rows(roster)
    parent = ""
    if kind in ("which_child", "followup"):
        parent = copy_hi.question_hi_for(state.profile, get_spec(state.cursor.qid))
    ctx = {
        "active_question": engine.active_prompt_hi(state),
        "followup_question": engine.followup_prompt_hi(state),
        "last_answered_question": engine.last_answered_prompt_hi(state)[1],
        "children_roster": roster_rows,
        "caller_utterance": utterance,
    }
    if parent:
        ctx["parent_question"] = parent
    ctx = {k: v for k, v in ctx.items() if v is not None and (v != "" or k in _KEEP_IN_CONTEXT)}
    ctx_json = json.dumps(ctx, ensure_ascii=False)
    req_input: str | list[dict[str, Any]] = ctx_json
    if audio:
        encoded = base64.b64encode(audio).decode("ascii")
        req_input = [
            {"type": "text", "text": ctx_json + "\n"},
            {"type": "audio", "data": encoded, "mime_type": audio_mime_type},
        ]
    questions = {key: _SCHEMA["questions"][key] for key in _CORE_QUESTIONS}
    if has_side:
        questions |= {key: _SCHEMA["questions"][key] for key in _SIDE_QUESTIONS}
        questions["side_qid"] = _side_qid_head(state)
    if roster_rows is not None:
        questions["child_ref"] = _child_ref_head(roster)
    if needs_detail:
        questions["detail_present"] = _SCHEMA["questions"]["detail_present"]
    return {"model": DECISIONS_MODEL, "input": req_input, "questions": questions}


def _urllib_transport(
    url: str, body: bytes, headers: dict[str, str], timeout_sec: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return int(getattr(response, "status", 200)), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() if hasattr(exc, "read") else b""


def _json_object(raw_body: Any) -> dict[str, Any] | None:
    """Decode a body that must be exactly one JSON object, else None."""
    if not isinstance(raw_body, (bytes, bytearray)):
        return None
    text = bytes(raw_body).decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _verdict(status: int, raw: Any) -> tuple[dict[str, Any] | None, str, bool]:
    """Judge one HTTP response as (payload, failure reason, worth retrying)."""
    if 200 <= status <= 299:
        payload = _json_object(raw)
        if isinstance((payload or {}).get("answers"), dict) and payload["answers"]:
            return payload, "", False
        return None, "empty_answers" if payload else "malformed_response", False
    error = (_json_object(raw) or {}).get("error")
    code = str(error.get("code") or "").lower() if isinstance(error, dict) else ""
    if code in _SAFETY_CODES and status < 500:
        return None, "safety_block", False
    return None, f"http_{status}", status in _RETRYABLE or status >= 500


async def _try_once(
    send: TransportFn, url: str, body: bytes, headers: dict[str, str], timeout: float
) -> tuple[dict[str, Any] | None, str, bool]:
    """One bounded round trip, normalised so the caller only reads a verdict."""
    try:
        status, raw = await asyncio.wait_for(
            asyncio.to_thread(send, url, body, headers, timeout), timeout=timeout
        )
    except asyncio.CancelledError:
        raise
    except urllib.error.HTTPError as exc:
        return _verdict(int(exc.code), exc.read() if hasattr(exc, "read") else b"")
    except Exception as exc:  # noqa: BLE001 - transport faults are data, never crashes
        is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError, socket.timeout)) or (
            isinstance(exc, urllib.error.URLError)
            and isinstance(exc.reason, (TimeoutError, socket.timeout))
        )
        if is_timeout:
            return None, "timeout", True
        connected = isinstance(exc, (urllib.error.URLError, ConnectionError, OSError))
        return None, "connection_error" if connected else "malformed_response", connected
    return _verdict(status, raw)


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
    """Execute one Decisions API request inside a bounded retry budget."""
    started = time.perf_counter()
    body = json.dumps(request, ensure_ascii=False).encode("utf-8")
    raw_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    key = str(raw_key or "").strip()
    limit, per_try = max(1, int(max_attempts)), max(0.01, float(attempt_timeout_sec))
    deadline = started + max(0.01, float(total_budget_sec))
    headers = {"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": key}
    send = transport or _urllib_transport
    payload: dict[str, Any] | None = None
    attempts, reason = 0, "missing_api_key"

    allow_unkeyed_transport = api_key is None and transport is not None
    can_attempt = bool(key) or allow_unkeyed_transport
    while can_attempt and attempts < limit:
        timeout = min(per_try, deadline - time.perf_counter())
        if timeout <= 0:
            if attempts == 0:
                reason = "timeout"
            break
        attempts += 1
        payload, reason, retry = await _try_once(send, endpoint, body, headers, timeout)
        if payload is not None or not retry:
            break

    questions = request.get("questions") or {}
    usage = (payload or {}).get("usage")
    tokens = usage.get("total_input_tokens") if isinstance(usage, dict) else None
    usable = isinstance(tokens, int) and not isinstance(tokens, bool)
    return DecisionCallResult(
        payload is not None,
        "" if payload is not None else reason,
        attempts,
        max(0, int(round((time.perf_counter() - started) * 1000))),
        len(body),
        len(questions),
        len((questions.get("side_qid") or {}).get("criteria") or {}),
        usable,
        tokens if usable else None,
        (payload or {}).get("answers") or {},
    )


def validate_thresholds(thresholds: DecisionThresholds) -> DecisionThresholds:
    """Reject anything that is not already a validated threshold artifact."""
    if not isinstance(thresholds, DecisionThresholds):
        raise TypeError("thresholds must be a DecisionThresholds instance")
    return thresholds


def load_thresholds(path: str | Path | None = None) -> DecisionThresholds:
    """Load, cache by file mtime, and strictly validate a threshold artifact."""
    configured = str(os.environ.get("DECISION_THRESHOLDS_PATH") or "").strip()
    resolved = Path(path or configured) if (path or configured) else _THRESHOLDS_PATH
    cache_key = (str(resolved), resolved.stat().st_mtime_ns)
    if (cached := _THRESHOLD_CACHE.get(cache_key)) is not None:
        return cached
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("thresholds"), dict):
        data = data["thresholds"]
    return _THRESHOLD_CACHE.setdefault(cache_key, DecisionThresholds.model_validate(data))


def _prob(raw_answers: Mapping[str, Any], key: str) -> float | None:
    """One noul head's yes-probability, or None when the model never answered it."""
    answer = raw_answers.get(key) if isinstance(raw_answers, Mapping) else None
    value = answer.get("noul") if isinstance(answer, Mapping) else None
    usable = isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value) if usable and math.isfinite(value) else None


def _clears(probability: float | None, floor: float) -> bool:
    """True only when the head was answered and clears its calibrated yes-gate."""
    return probability is not None and probability >= floor


def evaluate_choice(
    raw_answers: Mapping[str, Any],
    key: str,
    *,
    catchall_key: str = "other",
    min_confidence: float,
    min_margin: float,
    allowed_winners: frozenset[str] | None = None,
) -> ChoiceEvaluation:
    """Accept a choice answer only when the winner clears confidence and margin."""
    answer = raw_answers.get(key) if isinstance(raw_answers, Mapping) else None
    raw = answer.get("probabilities") if isinstance(answer, Mapping) else None
    probs: dict[str, float] = {}
    for option, probability in (raw if isinstance(raw, Mapping) else {}).items():
        usable = isinstance(probability, (int, float)) and not isinstance(probability, bool)
        if usable and math.isfinite(probability):
            probs[str(option)] = float(probability)
    if not probs:
        return ChoiceEvaluation(False, "", "", None, "", None, None, "missing_answer")
    ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    top_option, top_prob = ranked[0]
    runnerup_option, runnerup_prob = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = top_prob - runnerup_prob
    if top_option == catchall_key:
        reason = "catchall_top"
    elif allowed_winners is not None and top_option not in allowed_winners:
        reason = "disallowed_option"
    elif top_prob + _EPS < min_confidence:
        reason = "below_min_confidence"
    elif margin + _EPS < min_margin:
        reason = "below_margin"
    else:
        reason = ""
    return ChoiceEvaluation(
        not reason,
        "" if reason else top_option,
        top_option,
        top_prob,
        runnerup_option,
        runnerup_prob,
        margin,
        reason,
    )


def _gated(
    answers: Mapping[str, Any], key: str, limits: DecisionThresholds, allowed: frozenset[str]
) -> ChoiceEvaluation:
    """Score one choice head against the threshold pair named after its key."""
    prefix = "outcome" if key == "current_outcome" else key
    return evaluate_choice(
        answers,
        key,
        min_confidence=getattr(limits, f"{prefix}_min_confidence"),
        min_margin=getattr(limits, f"{prefix}_min_margin"),
        allowed_winners=allowed,
    )


def _route_side(
    state: CallState,
    intent_eval: ChoiceEvaluation | None,
    qid_eval: ChoiceEvaluation | None,
    *,
    blocked: bool,
) -> tuple[str, SideAction, dict[str, Any] | None]:
    """Resolve the accepted side target, its action, and any future-question match."""
    if blocked or intent_eval is None or qid_eval is None:
        return "", SideAction.NONE, None
    if not (intent_eval.accepted and qid_eval.accepted):
        return "", SideAction.NONE, None
    action = _SIDE_ACTIONS[intent_eval.winner]
    if accepted := engine.accepted_side_target(state, qid_eval.winner):
        return accepted, action, None
    if qid_eval.winner == state.cursor.qid or engine.is_answered_or_na(state, qid_eval.winner):
        return "", SideAction.NONE, None
    act = "clarify_meaning" if action is SideAction.CLARIFY_MEANING else "add_or_correct"
    match = {"qid": qid_eval.winner, "confidence": qid_eval.top_prob, "act": act}
    return "", SideAction.NONE, match


def _can_retract(state: CallState) -> bool:
    """A retraction only makes sense on a step she could actually take back."""
    if state.cursor.substep is not SubStep.MAIN:
        return True
    active = state.answers.get(state.cursor.qid)
    return active is not None and active.state is AnswerStatus.CONCERN


def interpret_turn_decision(
    state: CallState,
    raw_answers: Mapping[str, Any],
    *,
    thresholds: DecisionThresholds,
    utterance_text: str = "",
) -> TurnIntent:
    """Translate one Decisions API response into a typed TurnCommand plus telemetry."""
    limits = validate_thresholds(thresholds)
    answers = raw_answers if isinstance(raw_answers, Mapping) else {}
    kind, needs_detail, has_side = _step(state)
    roster = engine.children_roster(state)

    optional = {"mentions_diagnosis": has_side, "detail_present": needs_detail}
    heads = {
        key: _prob(answers, key) if optional.get(key, True) else None for key in _GATES
    }

    emergency = _clears(heads["danger_now"], limits.danger_now_min_yes)
    caller_stop = not emergency and _clears(heads["caller_stop"], limits.caller_stop_min_yes)
    mentions_profile = _clears(heads["mentions_profile"], limits.mentions_profile_min_yes)
    diagnostic = _clears(heads["mentions_diagnosis"], limits.mentions_diagnosis_min_yes)
    detail_present = _clears(heads["detail_present"], limits.detail_present_min_yes)

    outcome_eval = _gated(answers, "current_outcome", limits, VALID_OUTCOMES)
    outcome = (
        TurnOutcome(outcome_eval.winner) if outcome_eval.accepted else TurnOutcome.NOT_ABOUT_ACTIVE
    )
    needs_profile = bool(
        mentions_profile or (kind == "profile" and outcome is not TurnOutcome.CLARIFY)
    )
    intent_eval = qid_eval = None
    if has_side:
        intent_eval = _gated(answers, "side_intent", limits, VALID_SIDE_INTENTS)
        qid_eval = _gated(answers, "side_qid", limits, _ROUTABLE_QIDS)
    side_qid, side_action, future_match = _route_side(
        state, intent_eval, qid_eval, blocked=emergency or caller_stop
    )
    if side_qid and (
        (side_action is SideAction.CLARIFY_MEANING and outcome is TurnOutcome.CLARIFY)
        or (outcome is TurnOutcome.RETRACT and not _can_retract(state))
    ):
        outcome = TurnOutcome.NOT_ABOUT_ACTIVE

    allowed = frozenset(row["id"] for row in roster) | {"all"}
    child_eval = _gated(answers, "child_ref", limits, allowed) if len(roster) >= 2 else None
    command = TurnCommand(
        emergency=emergency,
        caller_stop=caller_stop,
        outcome=outcome,
        child_ref=child_eval.winner if child_eval is not None and child_eval.accepted else "",
        followup_answered=bool(detail_present and outcome is TurnOutcome.CONCERN),
        side_qid=side_qid,
        side_action=side_action,
        utterance=" ".join(str(utterance_text or "").split()),
        diagnostic=diagnostic,
    )
    diagnostics = {
        "active_qid": state.cursor.qid,
        "step_kind": kind,
        **heads,
        "current_outcome": outcome_eval.as_dict(),
        "side_intent": None if intent_eval is None else intent_eval.as_dict(),
        "side_qid": None if qid_eval is None else qid_eval.as_dict(),
        "child_ref": None if child_eval is None else child_eval.as_dict(),
        "future_qid_match": future_match,
    }
    return TurnIntent(
        command=command,
        needs_profile_extraction=needs_profile,
        diagnostics=diagnostics,
        future_qid_match=future_match,
    )
