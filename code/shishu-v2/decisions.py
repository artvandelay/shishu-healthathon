"""Gemini Decisions API client, single-pass turn schema builder, and policy interpreter.

Phase 2 standalone module: evaluated against deterministic call_cursor states
without altering live voice wiring.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import inspect
import io
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import call_cursor
from call_cursor import children_roster, question_hi_for, routable_qids
from triage_catalog import FOLLOWUP_HI, get_spec

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

DECISIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
DECISIONS_MODEL = "gemini-decision-1"

_QUESTION_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RETRYABLE_HTTP_CODES = frozenset({408, 429})
_SAFETY_ERROR_CODES = frozenset({
    "safety",
    "prohibited_content",
    "blocklist",
    "image_safety",
    "content_blocked",
})
_VALID_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract"})
_VALID_SIDE_ACTS = frozenset({"add_or_correct", "clarify_meaning"})
_VALID_SIDE_POLARITIES = frozenset({"concern", "clear", "unknown", "retract"})
_ROUTABLE_QID_SET = frozenset(routable_qids())
_EVER_PREGNANT_NEED_EN = "whether the caller has ever been pregnant"
_FLOAT_EPS = 1e-12

_NOUL_THRESHOLD_FIELDS = (
    "danger_now_min_yes",
    "caller_stop_min_yes",
    "mentions_profile_min_yes",
    "detail_present_min_yes",
    "mentions_other_min_yes",
)
_CONFIDENCE_THRESHOLD_FIELDS = (
    "outcome_min_confidence",
    "side_qid_min_confidence",
    "side_act_min_confidence",
    "side_polarity_min_confidence",
    "child_ref_min_confidence",
)
_MARGIN_THRESHOLD_FIELDS = (
    "outcome_min_catchall_margin",
    "outcome_min_runnerup_margin",
    "side_qid_min_catchall_margin",
    "side_qid_min_runnerup_margin",
    "child_ref_min_catchall_margin",
    "child_ref_min_runnerup_margin",
)
_ALL_NUMERIC_THRESHOLD_FIELDS = (
    _NOUL_THRESHOLD_FIELDS + _CONFIDENCE_THRESHOLD_FIELDS + _MARGIN_THRESHOLD_FIELDS
)


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
    mentions_other_min_yes: float = 1.0
    outcome_min_confidence: float = 1.0
    outcome_min_catchall_margin: float = 1.0
    outcome_min_runnerup_margin: float = 1.0
    side_qid_min_confidence: float = 1.0
    side_qid_min_catchall_margin: float = 1.0
    side_qid_min_runnerup_margin: float = 1.0
    side_act_min_confidence: float = 1.0
    side_polarity_min_confidence: float = 1.0
    child_ref_min_confidence: float = 1.0
    child_ref_min_catchall_margin: float = 1.0
    child_ref_min_runnerup_margin: float = 1.0


UNCALIBRATED_THRESHOLDS = DecisionThresholds(calibrated=False)


@dataclass(frozen=True, slots=True)
class ChoiceEvaluation:
    accepted: bool
    winner: str
    top_option: str
    top_prob: float | None
    catchall_prob: float | None
    catchall_margin: float | None
    runnerup_option: str
    runnerup_prob: float | None
    runnerup_margin: float | None
    reject_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "winner": self.winner,
            "top_option": self.top_option,
            "top_prob": self.top_prob,
            "catchall_prob": self.catchall_prob,
            "catchall_margin": self.catchall_margin,
            "runnerup_option": self.runnerup_option,
            "runnerup_prob": self.runnerup_prob,
            "runnerup_margin": self.runnerup_margin,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    cursor_payload: dict[str, Any]
    needs_profile_extraction: bool
    future_qid_match: dict[str, Any] | None
    diagnostics: dict[str, Any]


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw signed 16-bit mono little-endian PCM bytes in a WAV container."""
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    data = bytes(pcm)
    if len(data) % 2 == 1:
        data = data[:-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)
    return buf.getvalue()


def _spec_or_none(qid: str) -> Any:
    if not qid or qid == "ever_pregnant":
        return None
    try:
        return get_spec(qid)
    except KeyError:
        return None


def turn_view(state: dict) -> tuple[str, str, str, str]:
    """Return (active_qid, step_kind, active_prompt, followup_prompt) for state."""
    active_qid = str(state.get("active_qid") or "")
    hold = state.get("hold_reason")
    pending_idx = state.get("pending_followup_idx")
    spec = _spec_or_none(active_qid)

    if hold in call_cursor._HOLD_HI:
        active_prompt = call_cursor._HOLD_HI[hold]
    elif active_qid == "ever_pregnant":
        active_prompt = call_cursor._EVER_PREGNANT_HI
    elif pending_idx == -1:
        children = (state.get("profile") or {}).get("children") or []
        active_prompt = call_cursor._which_child_hi(children)
    elif isinstance(pending_idx, int) and pending_idx >= 0 and spec is not None and pending_idx < len(spec.followups):
        active_prompt = FOLLOWUP_HI[spec.followups[pending_idx]]
    elif (
        active_qid == "close_anything_else"
        and ((state.get("answers") or {}).get("close_anything_else") or {}).get("state") == "concern"
    ):
        active_prompt = call_cursor._CLOSE_MORE_HI
    elif spec is not None:
        active_prompt = question_hi_for(state, spec)
    else:
        active_prompt = ""

    if hold in ("name_fragment", "marital_status", "children_ages"):
        step_kind = "profile"
    elif active_qid == "ever_pregnant":
        step_kind = "main"
    elif pending_idx == -1:
        step_kind = "which_child"
    elif isinstance(pending_idx, int) and pending_idx >= 0 and spec is not None and pending_idx < len(spec.followups):
        step_kind = "followup"
    elif active_qid == "close_anything_else":
        step_kind = "close"
    elif spec is not None and spec.kind == "profile":
        step_kind = "profile"
    elif spec is not None and spec.kind == "bridge":
        step_kind = "bridge"
    else:
        step_kind = "main"

    if step_kind == "followup" and spec is not None and isinstance(pending_idx, int) and pending_idx < len(spec.followups):
        followup_prompt = FOLLOWUP_HI[spec.followups[pending_idx]]
    elif step_kind == "main" and spec is not None and spec.followups:
        followup_prompt = FOLLOWUP_HI[spec.followups[0]]
    elif step_kind == "close":
        followup_prompt = FOLLOWUP_HI["open_concern_detail"]
    else:
        followup_prompt = ""

    return active_qid, step_kind, active_prompt, followup_prompt


def _needs_detail_present(active_qid: str, step_kind: str, followup_prompt: str) -> bool:
    return (step_kind in ("main", "followup") and bool(followup_prompt)) or (
        active_qid == "close_anything_else"
    )


def _referenced_qid_criteria(state: dict) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for qid in routable_qids():
        if qid == "ever_pregnant":
            criteria[qid] = f"{call_cursor._EVER_PREGNANT_HI} ({_EVER_PREGNANT_NEED_EN})"
        else:
            spec = get_spec(qid)
            criteria[qid] = f"{question_hi_for(state, spec)} ({spec.need_en})"
    criteria["none_or_unclear"] = (
        "The caller is not referring to any screening item outside the active prompt, "
        "or there is not enough evidence to identify another item."
    )
    return criteria


def _build_context_text(
    *,
    active_qid: str,
    step_kind: str,
    active_prompt: str,
    followup_prompt: str,
    roster: list[dict[str, Any]],
    utterance_text: str,
    has_audio: bool,
) -> str:
    lines = [
        f"active_qid: {active_qid}",
        f"step_kind: {step_kind}",
        f"active_prompt: {active_prompt}",
        f"followup_prompt: {followup_prompt}",
    ]
    if roster:
        lines.append("children_roster:")
        for row in roster:
            name = str(row.get("name") or "")
            lines.append(
                f"{row['id']} | {row['age_label_hi']} | {row['sex']} | {row['rank']} | {name}"
            )
    clean_utterance = " ".join(str(utterance_text or "").split())
    if has_audio and clean_utterance:
        lines.append(f"caller_utterance: {clean_utterance} [audio block attached]")
    elif has_audio:
        lines.append("caller_utterance: [spoken reply in attached audio block]")
    else:
        lines.append(f"caller_utterance: {clean_utterance}")
    return "\n".join(lines)


def build_turn_request(
    state: dict,
    *,
    utterance_text: str = "",
    audio_wav_bytes: bytes | None = None,
    audio_mime_type: str = "audio/wav",
) -> dict[str, Any]:
    """Build a single-pass Gemini Decisions API request for one caller turn."""
    active_qid, step_kind, active_prompt, followup_prompt = turn_view(state)
    roster = children_roster(state)
    audio_data = bytes(audio_wav_bytes) if audio_wav_bytes else b""
    has_audio = bool(audio_data)

    context_text = _build_context_text(
        active_qid=active_qid,
        step_kind=step_kind,
        active_prompt=active_prompt,
        followup_prompt=followup_prompt,
        roster=roster,
        utterance_text=utterance_text,
        has_audio=has_audio,
    )

    if has_audio:
        b64_audio = base64.b64encode(audio_data).decode("ascii")
        request_input: str | list[dict[str, Any]] = [
            {"type": "text", "text": context_text + "\n"},
            {"type": "audio", "data": b64_audio, "mime_type": audio_mime_type},
        ]
    else:
        request_input = context_text

    questions: dict[str, dict[str, Any]] = {
        "danger_now": {
            "type": "noul",
            "instructions": (
                "Is the caller describing an active life-threatening medical emergency happening "
                "RIGHT NOW (such as not breathing, severe fast breathing, blue lips now, "
                "unconscious, unstoppable fits or seizures, unstoppable bleeding, or explicitly "
                "saying it is an emergency right now)?"
            ),
            "criteria": {
                "true_description": (
                    "Active life-threatening danger sign happening right now, or explicit statement "
                    "of a medical emergency happening right now."
                ),
                "false_description": (
                    "Past medical history, a resolved event from earlier in life, a routine "
                    "yes/no or symptom answer, or no emergency happening right now."
                ),
            },
        },
        "caller_stop": {
            "type": "noul",
            "instructions": (
                "Does the caller explicitly ask to end, stop, or hang up the call right now?"
            ),
            "criteria": {
                "true_description": (
                    "Caller explicitly asks to end, stop, or disconnect the call now or continue another time."
                ),
                "false_description": (
                    "Caller asks only to pause or wait briefly, asks a clarification question, "
                    "or gives a normal response."
                ),
            },
        },
        "mentions_profile": {
            "type": "noul",
            "instructions": (
                "Does the caller state or correct her name, marital status (married, about to "
                "marry, or unmarried), number of children, children's ages/sex/names, or whether "
                "she has ever been pregnant?"
            ),
            "criteria": {
                "true_description": (
                    "Caller provides or updates her name, marital status, child count, children's "
                    "ages or details, or ever-pregnant status."
                ),
                "false_description": (
                    "Caller only answers a clinical screening item, asks for clarification, or "
                    "makes no statement or correction about her demographic profile."
                ),
            },
        },
        "current_outcome": {
            "type": "choice",
            "instructions": (
                "Considering active_qid, step_kind, and active_prompt, how does caller_utterance "
                "respond to active_prompt?"
            ),
            "criteria": {
                "clear": (
                    "The caller answers no, negative, normal, or nothing further regarding active_prompt."
                ),
                "concern": (
                    "The caller answers yes, agrees to proceed on a bridge step, confirms the "
                    "event or symptom in active_prompt, names or selects a child when step_kind "
                    "is which_child, or gives clinical detail answering active_prompt when "
                    "step_kind is followup or close."
                ),
                "unknown": (
                    "The caller says she does not know, does not remember, or is unsure about active_prompt."
                ),
                "clarify": (
                    "The caller did not understand active_prompt or asks what active_prompt or a "
                    "term inside active_prompt means."
                ),
                "retract": (
                    "The caller explicitly retracts or reverses an earlier affirmative answer on "
                    "active_qid itself (such as while answering a followup or which_child prompt "
                    "for active_qid)."
                ),
                "not_about_active": (
                    "The caller's utterance does not answer active_prompt at all—for example, she "
                    "only speaks about another screening item, only gives profile demographics, "
                    "or says something unrelated to active_prompt."
                ),
            },
        },
        "mentions_other_item": {
            "type": "noul",
            "instructions": (
                "Does the caller add information to, correct, retract, or ask the meaning of a "
                "screening item OTHER than active_prompt?"
            ),
            "criteria": {
                "true_description": (
                    "Caller adds a detail, corrects or retracts a prior answer, or asks the "
                    "meaning of a different screening question outside active_prompt."
                ),
                "false_description": (
                    "Caller only addresses active_prompt, only updates demographic profile info, "
                    "or makes no reference to any screening item outside active_prompt."
                ),
            },
        },
        "referenced_qid": {
            "type": "choice",
            "instructions": (
                "Which screening item OTHER than active_prompt (active_qid) is the caller adding "
                "to, correcting, retracting, or asking about? Select none_or_unclear when no "
                "other screening item is referenced or identifiable."
            ),
            "criteria": _referenced_qid_criteria(state),
        },
        "referenced_act": {
            "type": "choice",
            "instructions": (
                "If the caller refers to a screening item other than active_prompt, what action "
                "is she taking on that other item?"
            ),
            "criteria": {
                "add_or_correct": (
                    "The caller is adding details to, confirming, correcting, or retracting an "
                    "answer for a screening item outside active_prompt."
                ),
                "clarify_meaning": (
                    "The caller is asking the meaning of or expressing confusion about a "
                    "screening item or term outside active_prompt."
                ),
                "not_a_side_action": (
                    "The caller is not acting on any screening item outside active_prompt."
                ),
            },
        },
        "referenced_polarity": {
            "type": "choice",
            "instructions": (
                "If the caller adds to, corrects, or retracts a screening item other than "
                "active_prompt, what is the clinical outcome for that other item?"
            ),
            "criteria": {
                "concern": (
                    "The caller reports a positive finding, symptom, event, or an additional "
                    "affected relative or child for that other item."
                ),
                "clear": (
                    "The caller states that the other item is negative, normal, or never happened."
                ),
                "unknown": (
                    "The caller says she does not know or does not remember regarding that other item."
                ),
                "retract": (
                    "The caller explicitly takes back or corrects a mistaken earlier affirmative "
                    "answer on that other item."
                ),
                "not_applicable_or_none": (
                    "The caller is not adding to, correcting, or retracting another screening "
                    "item, or is only asking what another item means."
                ),
            },
        },
    }

    if len(roster) >= 2:
        child_criteria = {
            row["id"]: (
                f"Refers to child {row['id']} (rank={row['rank']}, age={row['age_label_hi']}, "
                f"sex={row['sex']}, name={row.get('name') or 'unnamed'})."
            )
            for row in roster
        }
        child_criteria["all"] = (
            "Refers to all or every child in children_roster (such as both or all of them)."
        )
        child_criteria["none_or_unclear"] = (
            "No specific child is mentioned or identifiable from the utterance, or the utterance "
            "does not specify which child."
        )
        questions["child_ref"] = {
            "type": "choice",
            "instructions": "Which child from children_roster is the caller referring to?",
            "criteria": child_criteria,
        }

    if _needs_detail_present(active_qid, step_kind, followup_prompt):
        questions["detail_present"] = {
            "type": "noul",
            "instructions": (
                "Does the caller's utterance state concrete clinical or factual detail answering "
                "followup_prompt (or describing a concern when step_kind is close), rather than "
                "only a bare yes/no, child selection, or acknowledgement?"
            ),
            "criteria": {
                "true_description": (
                    "The caller gives concrete clinical or factual detail (such as duration, "
                    "count, cause, symptom, age, or affected person) answering followup_prompt or "
                    "describing a concern."
                ),
                "false_description": (
                    "Only a bare yes/no, bare acknowledgement, statement of not knowing, "
                    "clarification question, or no concrete detail."
                ),
            },
        }

    for qkey in questions:
        if not _QUESTION_KEY_RE.match(qkey):
            raise ValueError(f"invalid Decisions API question key: {qkey!r}")

    return {
        "model": DECISIONS_MODEL,
        "input": request_input,
        "questions": questions,
    }


def _urllib_transport(
    url: str,
    body_bytes: bytes,
    headers: dict[str, str],
    timeout_sec: float,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = int(getattr(resp, "status", 200))
            return status, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        except Exception:
            payload = b""
        return int(exc.code), payload


def _is_timeout_exc(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (TimeoutError, socket.timeout))
    return False


def _is_connection_exc(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (urllib.error.URLError, ConnectionError, OSError, http.client.HTTPException),
    )


def _parse_json_body(raw_body: Any) -> dict[str, Any] | None:
    if isinstance(raw_body, dict):
        return raw_body
    if isinstance(raw_body, (bytes, bytearray)):
        text = bytes(raw_body).decode("utf-8", errors="replace")
    elif isinstance(raw_body, str):
        text = raw_body
    else:
        return None
    if not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_usage(payload: dict[str, Any] | None) -> tuple[bool, int | None]:
    if not isinstance(payload, dict):
        return False, None
    for key in ("usage", "usageMetadata", "usage_metadata"):
        raw = payload.get(key)
        if not isinstance(raw, dict) or not raw:
            continue
        input_tokens: int | None = None
        has_token_count = False
        for field in (
            "total_input_tokens",
            "totalInputTokens",
            "prompt_token_count",
            "promptTokenCount",
        ):
            val = raw.get(field)
            if isinstance(val, int) and not isinstance(val, bool):
                input_tokens = val
                has_token_count = True
                break
        if not has_token_count:
            for field in ("total_tokens", "totalTokenCount", "total_token_count"):
                val = raw.get(field)
                if isinstance(val, int) and not isinstance(val, bool):
                    has_token_count = True
                    break
        if has_token_count:
            return True, input_tokens
    return False, None


def _safety_reason_from_response(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("prompt_feedback", "promptFeedback"):
        pf = payload.get(key)
        if isinstance(pf, dict):
            reason = str(pf.get("block_reason") or pf.get("blockReason") or "").strip()
            if reason:
                return reason
    err = payload.get("error")
    if isinstance(err, dict):
        code = str(err.get("code") or "").strip().lower()
        if code in _SAFETY_ERROR_CODES:
            return code
    return ""


async def _invoke_transport(
    fn: Callable[..., Any],
    url: str,
    body_bytes: bytes,
    headers: dict[str, str],
    timeout_sec: float,
) -> tuple[int, Any]:
    if inspect.iscoroutinefunction(fn):
        raw = await asyncio.wait_for(
            fn(url, body_bytes, headers, timeout_sec),
            timeout=timeout_sec,
        )
    else:
        raw = await asyncio.wait_for(
            asyncio.to_thread(fn, url, body_bytes, headers, timeout_sec),
            timeout=timeout_sec,
        )
    if inspect.isawaitable(raw):
        raw = await asyncio.wait_for(raw, timeout=timeout_sec)

    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], int):
        return int(raw[0]), raw[1]
    if isinstance(raw, (dict, bytes, bytearray, str)):
        return 200, raw
    raise ValueError(f"unexpected transport return value: {type(raw).__name__}")


async def evaluate_turn(
    request: dict[str, Any],
    *,
    api_key: str | None = None,
    endpoint: str = DECISIONS_ENDPOINT,
    max_attempts: int = 2,
    attempt_timeout_sec: float = 2.0,
    total_budget_sec: float = 4.0,
    transport: Callable[..., Any] | None = None,
) -> DecisionCallResult:
    """Execute one Decisions API request with a bounded retry budget."""
    questions = request.get("questions") if isinstance(request, dict) else {}
    question_count = len(questions) if isinstance(questions, dict) else 0
    ref_criteria = (
        (questions.get("referenced_qid") or {}).get("criteria")
        if isinstance(questions, dict) and isinstance(questions.get("referenced_qid"), dict)
        else {}
    )
    catalog_option_count = len(ref_criteria) if isinstance(ref_criteria, dict) else 0

    body_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8")
    request_bytes = len(body_bytes)

    if api_key is not None:
        resolved_key = str(api_key).strip()
    else:
        resolved_key = str(os.environ.get("GEMINI_API_KEY") or "").strip()

    if not resolved_key and (api_key is not None or transport is None):
        return DecisionCallResult(
            ok=False,
            error_reason="missing_api_key",
            attempts=0,
            decision_ms=0,
            request_bytes=request_bytes,
            question_count=question_count,
            catalog_option_count=catalog_option_count,
            usage_present=False,
            total_input_tokens=None,
            raw_answers={},
        )

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "x-goog-api-key": resolved_key,
    }
    fn = transport or _urllib_transport
    max_tries = max(1, int(max_attempts))
    per_attempt = max(0.01, float(attempt_timeout_sec))
    budget = max(0.01, float(total_budget_sec))

    started = time.perf_counter()
    deadline = started + budget
    attempts = 0
    last_error = "unknown_error"
    usage_present = False
    total_input_tokens: int | None = None

    for attempt in range(1, max_tries + 1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if last_error == "unknown_error":
                last_error = "timeout"
            break
        attempts = attempt
        this_timeout = min(per_attempt, remaining)

        try:
            status, raw_body = await _invoke_transport(
                fn, endpoint, body_bytes, headers, this_timeout
            )
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            try:
                raw_body = exc.read()
            except Exception:
                raw_body = b""
        except BaseException as exc:
            if _is_timeout_exc(exc):
                last_error = "timeout"
                if attempt < max_tries:
                    continue
                break
            if _is_connection_exc(exc):
                last_error = "connection_error"
                if attempt < max_tries:
                    continue
                break
            last_error = "malformed_response"
            break

        if status in _RETRYABLE_HTTP_CODES or 500 <= status <= 599:
            last_error = f"http_{status}"
            if attempt < max_tries:
                continue
            break

        if 400 <= status <= 499:
            parsed_err = _parse_json_body(raw_body)
            if _safety_reason_from_response(parsed_err):
                last_error = "safety_block"
            else:
                last_error = f"http_{status}"
            break

        if not (200 <= status <= 299):
            last_error = f"http_{status}"
            break

        parsed = _parse_json_body(raw_body)
        if parsed is None:
            last_error = "malformed_response"
            break

        usage_present, total_input_tokens = _extract_usage(parsed)

        if _safety_reason_from_response(parsed):
            last_error = "safety_block"
            break

        if "answers" not in parsed or parsed["answers"] is None:
            last_error = "empty_answers"
            break
        answers = parsed["answers"]
        if not isinstance(answers, dict):
            last_error = "malformed_response"
            break
        if not answers:
            last_error = "empty_answers"
            break

        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return DecisionCallResult(
            ok=True,
            error_reason="",
            attempts=attempts,
            decision_ms=elapsed_ms,
            request_bytes=request_bytes,
            question_count=question_count,
            catalog_option_count=catalog_option_count,
            usage_present=usage_present,
            total_input_tokens=total_input_tokens,
            raw_answers=answers,
        )

    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
    return DecisionCallResult(
        ok=False,
        error_reason=last_error,
        attempts=attempts,
        decision_ms=elapsed_ms,
        request_bytes=request_bytes,
        question_count=question_count,
        catalog_option_count=catalog_option_count,
        usage_present=usage_present,
        total_input_tokens=total_input_tokens,
        raw_answers={},
    )


def validate_thresholds(
    thresholds: DecisionThresholds,
    *,
    require_calibrated: bool = True,
) -> DecisionThresholds:
    """Validate DecisionThresholds ranges and enforce calibration when required."""
    if not isinstance(thresholds, DecisionThresholds):
        raise TypeError("thresholds must be a DecisionThresholds instance")
    if require_calibrated and not thresholds.calibrated:
        raise ValueError(
            "Uncalibrated DecisionThresholds refused; load calibrated thresholds via load_thresholds()."
        )
    if thresholds.calibrated:
        for field in _NOUL_THRESHOLD_FIELDS:
            val = float(getattr(thresholds, field))
            if not math.isfinite(val) or not (0.01 < val < 0.99):
                raise ValueError(
                    f"Calibrated threshold {field}={val!r} must lie strictly inside (0.01, 0.99)."
                )
        for field in _CONFIDENCE_THRESHOLD_FIELDS:
            val = float(getattr(thresholds, field))
            if not math.isfinite(val) or not (0.0 < val < 1.0):
                raise ValueError(
                    f"Calibrated threshold {field}={val!r} must lie strictly inside (0.0, 1.0)."
                )
        for field in _MARGIN_THRESHOLD_FIELDS:
            val = float(getattr(thresholds, field))
            if not math.isfinite(val) or not (0.0 <= val < 1.0):
                raise ValueError(
                    f"Calibrated margin {field}={val!r} must lie in [0.0, 1.0)."
                )
    return thresholds


def load_thresholds(
    path: str | Path,
    require_calibrated: bool = False,
) -> DecisionThresholds:
    """Load and validate a DecisionThresholds JSON artifact."""
    raw_text = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if isinstance(data, dict) and isinstance(data.get("thresholds"), dict):
        data = data["thresholds"]
    if not isinstance(data, dict):
        raise ValueError("Threshold artifact must be a JSON object")

    calibrated = bool(data.get("calibrated", False))
    if calibrated:
        missing = [field for field in _ALL_NUMERIC_THRESHOLD_FIELDS if field not in data]
        if missing:
            raise ValueError(
                f"Calibrated threshold artifact missing fields: {', '.join(missing)}"
            )

    kwargs: dict[str, Any] = {"calibrated": calibrated}
    for field in _ALL_NUMERIC_THRESHOLD_FIELDS:
        if field in data:
            kwargs[field] = float(data[field])
        else:
            kwargs[field] = float(getattr(UNCALIBRATED_THRESHOLDS, field))
    thresholds = DecisionThresholds(**kwargs)
    return validate_thresholds(thresholds, require_calibrated=require_calibrated)


def _noul_prob(raw_answers: dict[str, Any], key: str) -> float | None:
    ans = raw_answers.get(key) if isinstance(raw_answers, dict) else None
    if not isinstance(ans, dict):
        return None
    val = ans.get("noul")
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return float(val)
    return None


def evaluate_choice(
    raw_answers: dict[str, Any],
    key: str,
    *,
    catchall_key: str,
    min_confidence: float,
    min_catchall_margin: float,
    min_runnerup_margin: float,
    allowed_winners: frozenset[str] | None = None,
) -> ChoiceEvaluation:
    """Accept a choice answer only when it beats catchall and runner-up by required margins."""
    ans = raw_answers.get(key) if isinstance(raw_answers, dict) else None
    if not isinstance(ans, dict):
        return ChoiceEvaluation(
            accepted=False,
            winner="",
            top_option="",
            top_prob=None,
            catchall_prob=None,
            catchall_margin=None,
            runnerup_option="",
            runnerup_prob=None,
            runnerup_margin=None,
            reject_reason="missing_answer",
        )

    raw_probs = ans.get("probabilities")
    probs: dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for opt, prob in raw_probs.items():
            if (
                isinstance(prob, (int, float))
                and not isinstance(prob, bool)
                and math.isfinite(prob)
            ):
                probs[str(opt)] = float(prob)

    if probs:
        ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        top_option, top_prob = ranked[0]
        catchall_prob = float(probs.get(catchall_key, 0.0))
        non_catchall = [
            (opt, prob) for opt, prob in ranked if opt != top_option and opt != catchall_key
        ]
        if non_catchall:
            runnerup_option, runnerup_prob = non_catchall[0]
        else:
            runnerup_option, runnerup_prob = "", 0.0
    else:
        top_option = str(ans.get("choice") or "").strip()
        conf = ans.get("confidence")
        if (
            not top_option
            or not isinstance(conf, (int, float))
            or isinstance(conf, bool)
            or not math.isfinite(conf)
        ):
            return ChoiceEvaluation(
                accepted=False,
                winner="",
                top_option=top_option,
                top_prob=None,
                catchall_prob=None,
                catchall_margin=None,
                runnerup_option="",
                runnerup_prob=None,
                runnerup_margin=None,
                reject_reason="missing_answer",
            )
        top_prob = float(conf)
        catchall_prob = 0.0
        runnerup_option = ""
        runnerup_prob = 0.0

    catchall_margin = top_prob - catchall_prob
    runnerup_margin = top_prob - runnerup_prob

    if top_option == catchall_key:
        reason = "catchall_top"
    elif allowed_winners is not None and top_option not in allowed_winners:
        reason = "disallowed_option"
    elif top_prob + _FLOAT_EPS < min_confidence:
        reason = "below_min_confidence"
    elif catchall_margin + _FLOAT_EPS < min_catchall_margin:
        reason = "below_catchall_margin"
    elif runnerup_margin + _FLOAT_EPS < min_runnerup_margin:
        reason = "below_runnerup_margin"
    else:
        return ChoiceEvaluation(
            accepted=True,
            winner=top_option,
            top_option=top_option,
            top_prob=top_prob,
            catchall_prob=catchall_prob,
            catchall_margin=catchall_margin,
            runnerup_option=runnerup_option,
            runnerup_prob=runnerup_prob,
            runnerup_margin=runnerup_margin,
            reject_reason="",
        )

    return ChoiceEvaluation(
        accepted=False,
        winner="",
        top_option=top_option,
        top_prob=top_prob,
        catchall_prob=catchall_prob,
        catchall_margin=catchall_margin,
        runnerup_option=runnerup_option,
        runnerup_prob=runnerup_prob,
        runnerup_margin=runnerup_margin,
        reject_reason=reason,
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
    clean_utterance = " ".join(str(utterance_text or "").split())

    active_qid, step_kind, _active_prompt, followup_prompt = turn_view(state)
    roster = children_roster(state)
    needs_detail = _needs_detail_present(active_qid, step_kind, followup_prompt)

    danger_prob = _noul_prob(answers, "danger_now")
    stop_prob = _noul_prob(answers, "caller_stop")
    profile_prob = _noul_prob(answers, "mentions_profile")
    other_prob = _noul_prob(answers, "mentions_other_item")
    detail_prob = _noul_prob(answers, "detail_present") if needs_detail else None

    emergency = danger_prob is not None and danger_prob >= thresholds.danger_now_min_yes
    caller_stop = (not emergency) and (
        stop_prob is not None and stop_prob >= thresholds.caller_stop_min_yes
    )
    mentions_profile = (
        profile_prob is not None and profile_prob >= thresholds.mentions_profile_min_yes
    )
    mentions_other = other_prob is not None and other_prob >= thresholds.mentions_other_min_yes

    outcome_eval = evaluate_choice(
        answers,
        "current_outcome",
        catchall_key="not_about_active",
        min_confidence=thresholds.outcome_min_confidence,
        min_catchall_margin=thresholds.outcome_min_catchall_margin,
        min_runnerup_margin=thresholds.outcome_min_runnerup_margin,
        allowed_winners=_VALID_OUTCOMES,
    )
    resolved_outcome = outcome_eval.winner if outcome_eval.accepted else "not_about_active"
    needs_profile_extraction = bool(
        mentions_profile or (step_kind == "profile" and resolved_outcome != "clarify")
    )

    qid_eval = evaluate_choice(
        answers,
        "referenced_qid",
        catchall_key="none_or_unclear",
        min_confidence=thresholds.side_qid_min_confidence,
        min_catchall_margin=thresholds.side_qid_min_catchall_margin,
        min_runnerup_margin=thresholds.side_qid_min_runnerup_margin,
        allowed_winners=_ROUTABLE_QID_SET,
    )
    act_eval = evaluate_choice(
        answers,
        "referenced_act",
        catchall_key="not_a_side_action",
        min_confidence=thresholds.side_act_min_confidence,
        min_catchall_margin=0.0,
        min_runnerup_margin=0.0,
        allowed_winners=_VALID_SIDE_ACTS,
    )
    polarity_eval = evaluate_choice(
        answers,
        "referenced_polarity",
        catchall_key="not_applicable_or_none",
        min_confidence=thresholds.side_polarity_min_confidence,
        min_catchall_margin=0.0,
        min_runnerup_margin=0.0,
        allowed_winners=_VALID_SIDE_POLARITIES,
    )

    child_eval: ChoiceEvaluation | None = None
    resolved_child_ref = ""
    if len(roster) >= 2:
        allowed_children = frozenset(row["id"] for row in roster) | {"all"}
        child_eval = evaluate_choice(
            answers,
            "child_ref",
            catchall_key="none_or_unclear",
            min_confidence=thresholds.child_ref_min_confidence,
            min_catchall_margin=thresholds.child_ref_min_catchall_margin,
            min_runnerup_margin=thresholds.child_ref_min_runnerup_margin,
            allowed_winners=allowed_children,
        )
        if child_eval.accepted:
            resolved_child_ref = child_eval.winner

    detail_present = (
        needs_detail
        and detail_prob is not None
        and detail_prob >= thresholds.detail_present_min_yes
    )
    followup_answered = bool(detail_present and resolved_outcome == "concern")

    side_qid = ""
    side_action = ""
    side_outcome = ""
    future_qid_match: dict[str, Any] | None = None

    if (
        not emergency
        and not caller_stop
        and mentions_other
        and qid_eval.accepted
        and act_eval.accepted
    ):
        candidate_qid = qid_eval.winner
        candidate_act = act_eval.winner
        accepted_target = call_cursor._accepted_target(state, candidate_qid)
        if accepted_target:
            side_qid = accepted_target
            side_action = candidate_act
            if side_action == "add_or_correct":
                side_outcome = polarity_eval.winner if polarity_eval.accepted else "concern"
        elif candidate_qid != active_qid:
            prior_entry = (state.get("answers") or {}).get(candidate_qid)
            prior_state = prior_entry.get("state") if isinstance(prior_entry, dict) else ""
            if prior_state not in call_cursor._ANSWERED and prior_state != "not_applicable":
                future_qid_match = {
                    "qid": candidate_qid,
                    "confidence": qid_eval.top_prob,
                    "act": candidate_act,
                }

    active_can_retract = (
        state.get("pending_followup_idx") is not None
        or ((state.get("answers") or {}).get(active_qid) or {}).get("state") == "concern"
    )
    if side_qid and resolved_outcome == "retract" and not active_can_retract:
        resolved_outcome = "not_about_active"

    if emergency:
        active_note = clean_utterance
    elif needs_detail and step_kind in ("main", "close"):
        active_note = clean_utterance if (detail_present and resolved_outcome == "concern") else ""
    elif resolved_outcome == "concern" or (step_kind == "which_child" and resolved_outcome == "unknown"):
        active_note = clean_utterance
    else:
        active_note = ""

    side_note = clean_utterance if (side_qid and side_action == "add_or_correct") else ""

    if side_qid:
        cursor_payload: dict[str, Any] = {
            "emergency": emergency,
            "caller_stop": caller_stop,
            "current_outcome": resolved_outcome,
            "child_ref": resolved_child_ref,
            "note": side_note or active_note,
            "current_note": active_note,
            "followup_answered": followup_answered,
            "side_qid": side_qid,
            "side_action": side_action,
            "side_outcome": side_outcome,
            "side_note": side_note,
        }
    else:
        cursor_payload = {
            "emergency": emergency,
            "caller_stop": caller_stop,
            "current_outcome": resolved_outcome,
            "child_ref": resolved_child_ref,
            "note": active_note,
            "followup_answered": followup_answered,
        }

    raw_referenced_qid_candidate: dict[str, Any] | None = None
    if qid_eval.top_option and qid_eval.top_option != "none_or_unclear":
        raw_referenced_qid_candidate = {
            "qid": qid_eval.top_option,
            "confidence": qid_eval.top_prob,
            "catchall_margin": qid_eval.catchall_margin,
            "runnerup_margin": qid_eval.runnerup_margin,
            "mentions_other_prob": other_prob,
            "act": (
                act_eval.top_option
                if act_eval.top_option and act_eval.top_option != "not_a_side_action"
                else ""
            ),
            "polarity": (
                polarity_eval.top_option
                if polarity_eval.top_option
                and polarity_eval.top_option != "not_applicable_or_none"
                else ""
            ),
        }

    diagnostics = {
        "active_qid": active_qid,
        "step_kind": step_kind,
        "calibrated": thresholds.calibrated,
        "danger_now": danger_prob,
        "caller_stop": stop_prob,
        "mentions_profile": profile_prob,
        "mentions_other_item": other_prob,
        "detail_present": detail_prob,
        "current_outcome": outcome_eval.as_dict(),
        "referenced_qid": qid_eval.as_dict(),
        "referenced_act": act_eval.as_dict(),
        "referenced_polarity": polarity_eval.as_dict(),
        "child_ref": child_eval.as_dict() if child_eval is not None else None,
        "future_qid_match": future_qid_match,
        "raw_referenced_qid_candidate": raw_referenced_qid_candidate,
    }

    return InterpretationResult(
        cursor_payload=cursor_payload,
        needs_profile_extraction=needs_profile_extraction,
        future_qid_match=future_qid_match,
        diagnostics=diagnostics,
    )


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
