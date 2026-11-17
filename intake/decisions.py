"""`gemini-decision-1`: pooled keep-alive transport, one-pass request builder, and interpreter."""

from __future__ import annotations

import asyncio, base64, inspect, json, math, os, time
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any, NamedTuple
from weakref import WeakKeyDictionary

import httpx

from intake import engine
from intake.models import AnswerStatus, CallState, PackConfig, QuestionKind, QuestionSpec, Strict, TurnCommand, TurnOutcome

DECISIONS_ENDPOINT, DEFAULT_MODEL = "https://generativelanguage.googleapis.com/v1beta/decisions", "gemini-decision-1"
Transport = Callable[..., tuple[int, bytes] | Awaitable[tuple[int, bytes]]]
_CLIENTS: WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = WeakKeyDictionary()
_CORE = ("danger_now", "caller_stop", "mentions_profile", "redact_patient_note")
_NOUL = (*_CORE, "detail_present", "jump_topic")
_STATED_KINDS = (QuestionKind.GATE, QuestionKind.ROUTING_GATE)
_OUTCOMES, _STATED_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract"}), frozenset({"concern", "clear", "clarify"})
# Out-of-turn CONCERN on an already-CLEAR question is a revision of a closed answer, not a first read: it must clear
# this floor on top of the pack's stated gate (the bench's 0.82 audio-only flip must never be enough).
_REVISED_CONCERN_MIN = 0.92
_SAFETY, _RETRY_CODES = frozenset({"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}), {408, 429}

class DecisionCallResult(NamedTuple):
    ok: bool
    error_reason: str
    attempts: int
    decision_ms: int
    request_bytes: int
    question_count: int
    total_input_tokens: int | None
    raw_answers: dict[str, Any]
    used_slim_fallback: bool = False
class TurnIntent(Strict, frozen=True):
    command: TurnCommand
    needs_profile_extraction: bool
    diagnostics: dict[str, Any]
async def _post(url: str, body: bytes, headers: dict[str, str], budget: float) -> tuple[int, bytes]:
    """One pooled POST whose read budget belongs to THIS attempt, never to the client's lifetime.

    Call #15's 4.0s cliff was made worse by the first `httpx.AsyncClient(..., timeout=budget)` baking that
    attempt's budget onto every later call on the same loop; the timeout now travels with the request.
    """
    loop = asyncio.get_running_loop()
    client = _CLIENTS.get(loop)
    if client is None or client.is_closed:
        client = _CLIENTS[loop] = httpx.AsyncClient(http2=False)
    timeout = httpx.Timeout(connect=3.0, read=budget, write=3.0, pool=3.0)
    reply = await client.post(url, content=body, headers=headers, timeout=timeout)
    return reply.status_code, reply.content
def _verdict(status: int, raw: bytes) -> tuple[dict[str, Any] | None, str, bool]:
    """One reply's read: its payload, or a reason plus whether another call could still rescue the turn."""
    try:
        obj = json.loads(bytes(raw).decode("utf-8"))
    except Exception:  # noqa: BLE001 - a body that is not JSON is a bad reply, not a bad day
        return None, "response_invalid", False
    err = obj.get("error") if isinstance(obj, dict) else None
    code = str(err.get("code") or "").lower() if isinstance(err, dict) else ""
    if 200 <= int(status) <= 299:
        return ((obj, "", False) if isinstance(obj, dict) and obj.get("answers")
                else (None, "empty_answers", False))
    if code in _SAFETY and int(status) < 500:
        return None, "safety_block", False
    return None, f"http_{int(status)}", int(status) in _RETRY_CODES or int(status) >= 500


async def _send(send: Transport, endpoint: str, body: bytes, headers: dict[str, str],
                budget: float) -> tuple[dict[str, Any] | None, str, bool]:
    """One POST on its own read budget, classified where it landed."""
    try:
        sent = send(endpoint, body, headers, budget)
        status, raw = (await asyncio.wait_for(sent, budget + 0.25) if inspect.isawaitable(sent) else sent)
    except (TimeoutError, httpx.TimeoutException):
        return None, "timeout", True
    except httpx.TransportError:
        return None, "connection_error", True
    except Exception:  # noqa: BLE001
        return None, "response_invalid", False
    return _verdict(int(status), raw)


async def evaluate_turn(request: dict[str, Any], *, api_key: str | None = None, endpoint: str = DECISIONS_ENDPOINT,
                        max_attempts: int = 2, attempt_timeout_sec: float = 12.0, total_budget_sec: float = 18.0,
                        transport: Transport | None = None,
                        fallback_request: dict[str, Any] | None = None) -> DecisionCallResult:
    """One sequential call, waited out patiently, with a single retry only when the call itself failed.

    Call #16 (`5fce012`) paid for this twice: `hedge_after_sec = 2.5` launched a second 92 KB audio request beside
    a healthy first one, and `min(6.5, total_budget_sec)` killed slow-but-alive turns at 6.5s. A 5-7s reply from
    `gemini-decision-1` is Google's queue, not a dead attempt, so Attempt 1 now owns the whole 12.0s of its own
    read budget inside an 18.0s turn and nothing races it. Attempt 2 exists only for a real failure — timeout,
    connection error, 408/429, 5xx — and then carries the slim `fallback_request` so a retry is cheaper than the
    request that just failed. A safety block, an empty `answers`, an unparseable body, or any other 4xx is final:
    repeating it only spends the caller's patience on the same answer.
    """
    started = time.perf_counter()
    primary_body = json.dumps(request, ensure_ascii=False).encode("utf-8")
    fallback_body = (json.dumps(fallback_request, ensure_ascii=False).encode("utf-8")
                     if fallback_request is not None else None)
    key = str(api_key if api_key is not None else os.environ.get("GEMINI_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    send, deadline, limit = transport or _post, started + float(total_budget_sec), max(1, int(max_attempts))
    attempts, payload, reason = 0, None, "missing_api_key"
    body, sent_request = primary_body, request
    while payload is None and (key or (transport and api_key is None)) and attempts < limit:
        budget = min(float(attempt_timeout_sec), deadline - time.perf_counter())
        if budget <= 0:
            reason = "timeout"
            break
        attempts += 1
        payload, reason, retry = await _send(send, endpoint, body, headers, budget)
        if payload is not None or not retry:
            break
        body, sent_request = ((fallback_body, fallback_request) if fallback_body is not None
                             else (primary_body, request))
    toks = ((payload or {}).get("usage") or {}).get("total_input_tokens")
    return DecisionCallResult(payload is not None, "" if payload is not None else reason, attempts,
                              max(0, int((time.perf_counter() - started) * 1000)), len(body),
                              len(sent_request.get("questions") or {}), toks if type(toks) is int else None,
                              (payload or {}).get("answers") or {},
                              payload is not None and fallback_body is not None and body is fallback_body)


def _build_input(pack: PackConfig, state: CallState, *, utterance_text: str, audio_wav_bytes: bytes | None,
                 audio_mime_type: str) -> Any:
    """The context document plus the optional spoken audio that every Decision request carries.

    It is built exactly once per request because the model reads every head against this one document: an input
    assembled twice could describe the same turn two different ways, and the answers would no longer line up.
    """
    kind = engine.step_kind(pack, state)
    spec, rows, audio = pack.by_id.get(state.cursor.qid), engine.roster_rows(state), bytes(audio_wav_bytes or b"")
    said, follow = partial(engine.question_say, pack, state), engine.followup_prompt(pack, state)
    utt = " ".join(utterance_text.split()) or ("[spoken audio attached]" if audio else "")
    ctx: dict[str, Any] = {"active_question": engine.active_prompt(pack, state), "caller_utterance": utt, "followup_question": follow,
                           "last_answered_question": engine.last_answered_prompt(pack, state)[1]}
    if kind in ("which_item", "followup") and spec is not None: ctx["parent_question"] = said(spec)
    # A mid-call "मेरे ट्विंस बच्चे हैं, मुझे लगता नहीं आप वो सुन पाए" can only be read against what is already
    # stored, so the caller's filled profile slots and the children on record both travel with every turn.
    if known := {k: v for k, v in state.profile.items() if v not in (None, "")}: ctx["profile"] = known
    if state.ordered_roster:
        # One child still travels: a single child is what she is talking about, and "rank: only" carries no
        # choice set. item_ref stays gated on two or more rows above — there is nobody to choose between.
        ctx["roster"] = rows or [{"id": c.id, "rank": "only", "age_label": c.age_label, "tag": c.tag, "name": c.name}
                                 for c in state.ordered_roster]
    text = json.dumps(ctx, ensure_ascii=False)
    return ([{"type": "text", "text": text + "\n"}, {"type": "audio", "data": base64.b64encode(audio).decode(), "mime_type": audio_mime_type}]
            if audio else text)


def _stated_heads(pack: PackConfig, state: CallState, heads: Mapping[str, Any]) -> dict[str, Any]:
    """Group B: one `stated_<qid>` revision head per routable question other than the active one."""
    tpl = heads.get("stated_template")
    if tpl is None: return {}
    said = partial(engine.question_say, pack, state)
    return {f"stated_{q.id}": {"type": "choice",
                               "instructions": tpl["instructions"].format(question_say=said(q), need=q.need, group=q.group),
                               "criteria": tpl["criteria"]}
            for q in pack.questions if q.is_routable and q.id != state.cursor.qid and q.kind in _STATED_KINDS}


def _router_heads(pack: PackConfig, state: CallState, heads: Mapping[str, Any]) -> dict[str, Any]:
    """Group D: the jump flag and the open-question router, present only with two or more open questions."""
    tpl = heads.get("next_qid_template")
    open_q = [q for q in pack.questions if q.is_routable and q.kind is not QuestionKind.OPEN_CAPTURE
              and q.id != state.cursor.qid and state.answers.get(q.id) is None]
    if tpl is None or len(open_q) < 2: return {}
    said = partial(engine.question_say, pack, state)
    qs: dict[str, Any] = {"jump_topic": heads["jump_topic"]} if "jump_topic" in heads else {}
    qs["next_qid"] = {"type": "choice", "instructions": tpl["instructions"],
                      "criteria": {q.id: tpl["option_template"].format(question_say=said(q), need=q.need) for q in open_q}
                      | {"other": tpl["other_description"]}}
    return qs


def build_turn_request(pack: PackConfig, state: CallState, *, utterance_text: str = "", audio_wav_bytes: bytes | None = None,
                       audio_mime_type: str = "audio/wav", slim: bool = False) -> dict[str, Any]:
    """One request for one turn. `slim` drops the two groups a retry can no longer afford.

    The default request carries everything: the `stated_<qid>` revisions and the `next_qid` router are what make
    an out-of-turn "मेरे बच्चे के सीने में दर्द है" land without a re-ask. They are also 56 heads and a 53-option
    choice, which is exactly what blew the 4.0s deadline on `Call #15`, so the slim request keeps only the heads
    a turn cannot be read without: Group A, the active typed value, and the semantic flags.
    """
    heads, kind = pack.decisions.get("heads") or {}, engine.step_kind(pack, state)
    spec, rows = pack.by_id.get(state.cursor.qid), engine.roster_rows(state)
    router, follow = kind != "profile", engine.followup_prompt(pack, state)
    answered = any(r.state.is_resolved for q, r in state.answers.items() if pack.by_id[q].is_routable)
    qs: dict[str, Any] = {h: heads[h] for h in (*_CORE, "active_outcome") if h in heads}
    if kind == "main" and spec is not None:
        if spec.kind is QuestionKind.SINGLE_CHOICE:
            qs["choice_value"] = {"type": "choice", "instructions": "Which option in active_question did the caller choose?", "criteria": {k: o.criteria or o.label for k, o in spec.options.items()} | {"other": "Did not pick any listed option."}}
        elif spec.kind is QuestionKind.MULTI_CHOICE:
            qs |= {f"opt_{k}": {"type": "noul", "instructions": f"Did the caller select or report '{o.label}' ({o.criteria or o.en or k}) in response to active_question?"} for k, o in spec.options.items()}
        elif spec.kind is QuestionKind.SCORE: qs["score_value"] = {"type": "score", "instructions": f"Rate caller's answer to active_question ({spec.say}):", "criteria": list(spec.score_rubric)}
    if follow and kind in ("main", "close") and "detail_present" in heads: qs["detail_present"] = heads["detail_present"]
    if len(rows) >= 2 and (tpl := heads.get("item_ref_template")):
        named = {r["id"]: tpl["option_template"].format(**{**r, "name_or_unnamed": r["name"] or tpl["unnamed_label"]}) for r in rows}
        qs["item_ref"] = {"type": "choice", "instructions": tpl["instructions"], "criteria": named | {"all": tpl["all_description"], "other": tpl["other_description"]}}
    if not slim and (router or answered):
        qs |= _stated_heads(pack, state, heads)
    qs |= {f"flag_{f}": {"type": "noul", "instructions": fs["instructions"], "criteria": {k: fs[k] for k in ("true_description", "false_description") if k in fs}} for f, fs in pack.semantic_flags.items() if f not in state.semantic_flags}
    if not slim and router:
        qs |= _router_heads(pack, state, heads)
    return {"model": str(pack.decisions.get("model") or DEFAULT_MODEL),
            "input": _build_input(pack, state, utterance_text=utterance_text, audio_wav_bytes=audio_wav_bytes, audio_mime_type=audio_mime_type),
            "questions": qs}


def _num(v: Any) -> float | None: return float(v) if type(v) in (int, float) and math.isfinite(v) else None
def _yes(a: Mapping[str, Any], key: str, floor: Any) -> bool:
    p = _num(a[key].get("noul")) if isinstance(a.get(key), Mapping) else None
    return p is not None and p >= float(floor)
def _rank(a: Mapping[str, Any], key: str) -> list[tuple[str, float]]:
    """A choice head's options, best first, with non-finite and non-numeric probabilities dropped."""
    raw = a[key].get("probabilities") if isinstance(a.get(key), Mapping) else None
    return sorted(((str(k), p) for k, v in (raw or {}).items() if (p := _num(v)) is not None), key=lambda x: (-x[1], x[0]))
def _pick(a: Mapping[str, Any], key: str, t: Mapping[str, float], pair: str = "", catchall: str = "other",
          allowed: frozenset[str] | None = None, floors: Mapping[str, float] | None = None) -> str:
    """The head's winner, accepted only when it clears its own confidence gate (optionally raised per option) and margin."""
    ranked = _rank(a, key)
    if not ranked or ranked[0][0] == catchall or (allowed and ranked[0][0] not in allowed): return ""
    (top, best), sec, name = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0), pair or key
    need = max(t[f"{name}_min_confidence"], (floors or {}).get(top, 0.0))
    return top if best >= need and best - sec >= t[f"{name}_min_margin"] else ""
def _quiet_split(a: Mapping[str, Any], t: Mapping[str, float]) -> str:
    """`clear`/`unknown` are one quiet answer split across two labels, so neither can win alone: their PAIR is the read."""
    ranked = _rank(a, "active_outcome")
    if len(ranked) < 2 or {ranked[0][0], ranked[1][0]} != {"clear", "unknown"}: return ""
    pair, third = ranked[0][1] + ranked[1][1], (ranked[2][1] if len(ranked) > 2 else 0.0)
    return ranked[0][0] if pair >= t["outcome_min_confidence"] and pair - third >= t["outcome_min_margin"] else ""
def _score_value(a: Mapping[str, Any], t: Mapping[str, float]) -> dict[str, Any] | None:
    """A rubric score plus its nearest level, accepted only when the head's own confidence clears the pack gate."""
    head = a.get("score_value")
    if not isinstance(head, Mapping): return None
    score, confidence = _num(head.get("score")), _num(head.get("confidence"))
    if score is None or confidence is None or confidence < t["score_min_confidence"]: return None
    levels = {k: p for k, v in (head.get("probabilities") or {}).items() if (p := _num(v))}
    return {"score": score, "level": int(max(levels, key=levels.get)) if levels else int(round(score))}
def _active_value(spec: QuestionSpec | None, a: Mapping[str, Any], t: Mapping[str, float]) -> Any:
    """The typed answer the active question asks for, or None when its head did not produce one."""
    if spec is None: return None
    if spec.kind is QuestionKind.SINGLE_CHOICE: return _pick(a, "choice_value", t, "choice", allowed=frozenset(spec.options)) or None
    if spec.kind is QuestionKind.MULTI_CHOICE: return [k for k in spec.options if _yes(a, f"opt_{k}", t["multi_choice_min_yes"])] or None
    return _score_value(a, t) if spec.kind is QuestionKind.SCORE else None
def _stated_writes(pack: PackConfig, state: CallState, a: Mapping[str, Any], t: Mapping[str, float]) -> dict[str, TurnOutcome]:
    """Out-of-turn revisions of the other routable questions, each gated by its own stated thresholds."""
    out: dict[str, TurnOutcome] = {}
    for key in (k for k in a if str(k).startswith("stated_")):
        spec = pack.by_id.get(str(key).removeprefix("stated_"))
        if spec is None or not spec.is_routable or spec.kind not in _STATED_KINDS or spec.id == state.cursor.qid: continue
        # An out-of-turn concern on a question she already denied needs near-certain evidence: the pack's own stated
        # gate passes 0.80, which a noisy audio turn can clear while it is really talking about another symptom.
        # Revising UNKNOWN, retracting CONCERN to CLEAR, and asking what a term means keep the normal stated gate.
        prior = state.answers.get(spec.id)
        raised = {"concern": _REVISED_CONCERN_MIN} if prior is not None and prior.state is AnswerStatus.CLEAR else None
        if w := _pick(a, key, t, "stated", "unmentioned", _STATED_OUTCOMES, raised): out[spec.id] = TurnOutcome(w)
    return out
def _next_qid_probs(a: Mapping[str, Any]) -> dict[str, float]:
    """The router head's candidate questions, minus the catchall and any non-finite probability."""
    head = a.get("next_qid")
    probs = (head.get("probabilities") if isinstance(head, Mapping) else None) or {}
    return {str(k): p for k, v in probs.items() if str(k) != "other" and (p := _num(v)) is not None}
def interpret_turn_decision(pack: PackConfig, state: CallState, raw_answers: Mapping[str, Any], *, utterance_text: str = "") -> TurnIntent:
    t = {k: float(v) for k, v in (pack.decisions.get("thresholds") or {}).items()}
    a = raw_answers if isinstance(raw_answers, Mapping) else {}
    kind, spec = engine.step_kind(pack, state), pack.by_id.get(state.cursor.qid)
    on = {h: _yes(a, h, t[f"{h}_min_yes"]) for h in _NOUL}
    closing = on["danger_now"] or on["caller_stop"]
    val = None if closing or kind != "main" else _active_value(spec, a, t)
    outcome = TurnOutcome.ANSWERED if val is not None else TurnOutcome.NOT_ABOUT_ACTIVE
    if not closing and val is None and (w := _pick(a, "active_outcome", t, "outcome", allowed=_OUTCOMES) or _quiet_split(a, t)): outcome = TurnOutcome(w)
    stated = {} if closing else _stated_writes(pack, state, a, t)
    next_probs = _next_qid_probs(a)
    flags = tuple(f for f, fs in pack.semantic_flags.items() if f not in state.semantic_flags and _yes(a, f"flag_{f}", fs.get("min_yes", 0.8)))
    cmd = TurnCommand(
        emergency=on["danger_now"], caller_stop=not on["danger_now"] and on["caller_stop"], outcome=outcome, value=val,
        item_ref=_pick(a, "item_ref", t, "item_ref", allowed=frozenset({i.id for i in state.roster}) | {"all"}) if len(state.roster) >= 2 else "",
        followup_answered=bool(on["detail_present"] and kind in ("main", "close") and outcome is TurnOutcome.CONCERN),
        stated=stated, semantic_flags=flags, jump_topic=bool(not closing and kind != "profile" and on["jump_topic"]),
        next_qid_probs=next_probs, utterance=" ".join(utterance_text.split()), redacted=on["redact_patient_note"])
    diag = {"active_qid": state.cursor.qid, "step_kind": kind, "outcome": outcome.value, "stated": {q: o.value for q, o in stated.items()}, "next_qid_probs": next_probs,
            "semantic_flags": list(flags), "heads": on, **{h: _num(a[h].get("noul")) for h in _NOUL if isinstance(a.get(h), Mapping)},
            "top2": ",".join(f"{k}:{round(p, 3)}" for k, p in _rank(a, "active_outcome")[:2]),
            "noul_hot": ",".join(f"{h}:{round(v, 2)}" for h in _NOUL if isinstance(a.get(h), Mapping) and (v := _num(a[h].get("noul"))) is not None and v >= 0.25)}
    return TurnIntent(command=cmd, needs_profile_extraction=on["mentions_profile"] or (kind == "profile" and outcome is not TurnOutcome.CLARIFY), diagnostics=diag)
