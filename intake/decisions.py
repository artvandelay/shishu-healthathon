"""`gemini-decision-1`: pooled keep-alive transport, one-pass request builder, and interpreter."""

from __future__ import annotations

import asyncio, base64, inspect, json, math, os, time
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any, NamedTuple
from weakref import WeakKeyDictionary

import httpx

from intake import engine
from intake.models import CallState, PackConfig, QuestionKind, Strict, TurnCommand, TurnOutcome

DECISIONS_ENDPOINT, DEFAULT_MODEL = "https://generativelanguage.googleapis.com/v1beta/decisions", "gemini-decision-1"
Transport = Callable[..., tuple[int, bytes] | Awaitable[tuple[int, bytes]]]
_CLIENTS: WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = WeakKeyDictionary()
_CORE = ("danger_now", "caller_stop", "mentions_profile", "redact_patient_note")
_NOUL = (*_CORE, "turn_incomplete", "detail_present", "jump_topic")
_STATED_KINDS = (QuestionKind.GATE, QuestionKind.ROUTING_GATE, QuestionKind.OPEN_CAPTURE)
_OUTCOMES, _STATED_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract"}), frozenset({"concern", "clear", "clarify"})
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
class TurnIntent(Strict, frozen=True):
    command: TurnCommand
    needs_profile_extraction: bool
    diagnostics: dict[str, Any]
async def _post(url: str, body: bytes, headers: dict[str, str], budget: float) -> tuple[int, bytes]:
    loop = asyncio.get_running_loop()
    if (client := _CLIENTS.get(loop)) is None or client.is_closed: client = _CLIENTS[loop] = httpx.AsyncClient(http2=False, timeout=budget)
    reply = await client.post(url, content=body, headers=headers, timeout=budget)
    return reply.status_code, reply.content
async def evaluate_turn(request: dict[str, Any], *, api_key: str | None = None, endpoint: str = DECISIONS_ENDPOINT, max_attempts: int = 2,
                        attempt_timeout_sec: float = 4.0, total_budget_sec: float = 8.0, transport: Transport | None = None) -> DecisionCallResult:
    started, body = time.perf_counter(), json.dumps(request, ensure_ascii=False).encode("utf-8")
    key = str(api_key if api_key is not None else os.environ.get("GEMINI_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    send, deadline, payload, attempts, reason = transport or _post, started + float(total_budget_sec), None, 0, "missing_api_key"
    while (key or (transport and api_key is None)) and attempts < max(1, int(max_attempts)):
        if (budget := min(float(attempt_timeout_sec), deadline - time.perf_counter())) <= 0:
            reason = "timeout"
            break
        attempts += 1
        try:
            sent = send(endpoint, body, headers, budget)
            st, raw = await asyncio.wait_for(sent, budget) if inspect.isawaitable(sent) else sent
            obj = json.loads(bytes(raw).decode("utf-8"))
            err = obj.get("error") if isinstance(obj, dict) else None
            code = str(err.get("code") or "").lower() if isinstance(err, dict) else ""
            if 200 <= int(st) <= 299: payload, reason, retry = (obj, "", False) if isinstance(obj, dict) and obj.get("answers") else (None, "empty_answers", False)
            elif code in _SAFETY and int(st) < 500: payload, reason, retry = None, "safety_block", False
            else: payload, reason, retry = None, f"http_{int(st)}", int(st) in _RETRY_CODES or int(st) >= 500
        except (TimeoutError, httpx.TimeoutException): payload, reason, retry = None, "timeout", True
        except httpx.TransportError: payload, reason, retry = None, "connection_error", True
        except Exception: payload, reason, retry = None, "response_invalid", False  # noqa: BLE001
        if payload is not None or not retry: break
    toks = ((payload or {}).get("usage") or {}).get("total_input_tokens")
    return DecisionCallResult(payload is not None, "" if payload is not None else reason, attempts, max(0, int((time.perf_counter() - started) * 1000)),
                              len(body), len(request.get("questions") or {}), toks if type(toks) is int else None, (payload or {}).get("answers") or {})
def build_turn_request(pack: PackConfig, state: CallState, *, utterance_text: str = "", audio_wav_bytes: bytes | None = None,
                       audio_mime_type: str = "audio/wav", include_turn_incomplete: bool = False) -> dict[str, Any]:
    heads, kind = pack.decisions.get("heads") or {}, engine.step_kind(pack, state)
    spec, rows, audio = pack.by_id.get(state.cursor.qid), engine.roster_rows(state), bytes(audio_wav_bytes or b"")
    router, said, follow = kind not in ("profile", "bridge"), partial(engine.question_say, pack, state), engine.followup_prompt(pack, state)
    answered = any(r.state.is_resolved for q, r in state.answers.items() if pack.by_id[q].is_routable)
    qs: dict[str, Any] = {h: heads[h] for h in (*_CORE, "active_outcome", *(("turn_incomplete",) if include_turn_incomplete else ())) if h in heads}
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
    if (router or answered) and (tpl := heads.get("stated_template")):
        qs |= {f"stated_{q.id}": {"type": "choice", "instructions": tpl["instructions"].format(question_say=said(q), need=q.need, group=q.group), "criteria": tpl["criteria"]} for q in pack.questions if q.is_routable and q.id != state.cursor.qid and q.kind in _STATED_KINDS}
    qs |= {f"flag_{f}": {"type": "noul", "instructions": fs["instructions"], "criteria": {k: fs[k] for k in ("true_description", "false_description") if k in fs}} for f, fs in pack.semantic_flags.items() if f not in state.semantic_flags}
    open_q = [q for q in pack.questions if q.is_routable and q.kind is not QuestionKind.OPEN_CAPTURE and q.id != state.cursor.qid and state.answers.get(q.id) is None]
    if router and len(open_q) >= 2 and (tpl := heads.get("next_qid_template")):
        if "jump_topic" in heads: qs["jump_topic"] = heads["jump_topic"]
        qs["next_qid"] = {"type": "choice", "instructions": tpl["instructions"], "criteria": {q.id: tpl["option_template"].format(question_say=said(q), need=q.need) for q in open_q} | {"other": tpl["other_description"]}}
    utt = " ".join(utterance_text.split()) or ("[spoken audio attached]" if audio else "")
    ctx: dict[str, Any] = {"active_question": engine.active_prompt(pack, state), "caller_utterance": utt, "followup_question": follow,
                           "last_answered_question": engine.last_answered_prompt(pack, state)[1]}
    if kind in ("which_item", "followup") and spec is not None: ctx["parent_question"] = said(spec)
    if len(rows) >= 2: ctx["roster"] = rows
    text = json.dumps(ctx, ensure_ascii=False)
    inp: Any = [{"type": "text", "text": text + "\n"}, {"type": "audio", "data": base64.b64encode(audio).decode(), "mime_type": audio_mime_type}] if audio else text
    return {"model": str(pack.decisions.get("model") or DEFAULT_MODEL), "input": inp, "questions": qs}
def _num(v: Any) -> float | None: return float(v) if type(v) in (int, float) and math.isfinite(v) else None

def _yes(a: Mapping[str, Any], key: str, floor: Any) -> bool:
    p = _num(a[key].get("noul")) if isinstance(a.get(key), Mapping) else None
    return p is not None and p >= float(floor)
def _pick(a: Mapping[str, Any], key: str, t: Mapping[str, float], pair: str = "", catchall: str = "other", allowed: frozenset[str] | None = None) -> str:
    raw = a[key].get("probabilities") if isinstance(a.get(key), Mapping) else None
    ranked = sorted([(str(k), p) for k, v in (raw or {}).items() if (p := _num(v)) is not None], key=lambda x: (-x[1], x[0]))
    if not ranked or ranked[0][0] == catchall or (allowed and ranked[0][0] not in allowed): return ""
    (top, best), sec, name = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0), pair or key
    return top if best >= t[f"{name}_min_confidence"] and best - sec >= t[f"{name}_min_margin"] else ""
def interpret_turn_decision(pack: PackConfig, state: CallState, raw_answers: Mapping[str, Any], *, utterance_text: str = "") -> TurnIntent:
    t = {k: float(v) for k, v in (pack.decisions.get("thresholds") or {}).items()}
    a = raw_answers if isinstance(raw_answers, Mapping) else {}
    kind, spec = engine.step_kind(pack, state), pack.by_id.get(state.cursor.qid)
    on = {h: _yes(a, h, t[f"{h}_min_yes"]) for h in _NOUL}
    closing, val = on["danger_now"] or on["caller_stop"], None
    if not closing and kind == "main" and spec is not None:
        if spec.kind is QuestionKind.SINGLE_CHOICE: val = _pick(a, "choice_value", t, "choice", allowed=frozenset(spec.options)) or None
        elif spec.kind is QuestionKind.MULTI_CHOICE: val = [k for k in spec.options if _yes(a, f"opt_{k}", t["multi_choice_min_yes"])] or None
        elif spec.kind is QuestionKind.SCORE and isinstance(sh := a.get("score_value"), Mapping):
            sc, cf = _num(sh.get("score")), _num(sh.get("confidence"))
            probs = {k: p for k, v in (sh.get("probabilities") or {}).items() if (p := _num(v))}
            if sc is not None and cf is not None and cf >= t["score_min_confidence"]: val = {"score": sc, "level": int(max(probs, key=probs.get)) if probs else int(round(sc))}
    outcome = TurnOutcome.ANSWERED if val is not None else TurnOutcome.NOT_ABOUT_ACTIVE
    if not closing and val is None and (w := _pick(a, "active_outcome", t, "outcome", allowed=_OUTCOMES)): outcome = TurnOutcome(w)
    stated: dict[str, TurnOutcome] = {}
    for k in ([] if closing else [k for k in a if str(k).startswith("stated_")]):
        tgt = pack.by_id.get(str(k).removeprefix("stated_"))
        if tgt is None or not tgt.is_routable or tgt.id == state.cursor.qid: continue
        if tgt.kind is QuestionKind.OPEN_CAPTURE and kind != "close" and outcome is not TurnOutcome.CLARIFY: continue
        if w := _pick(a, k, t, "stated", "unmentioned", _STATED_OUTCOMES): stated[tgt.id] = TurnOutcome(w)
    nh = a.get("next_qid")
    next_probs = {str(k): p for k, v in ((nh.get("probabilities") if isinstance(nh, Mapping) else None) or {}).items() if str(k) != "other" and (p := _num(v))}
    flags = tuple(f for f, fs in pack.semantic_flags.items() if f not in state.semantic_flags and _yes(a, f"flag_{f}", fs.get("min_yes", 0.8)))
    cmd = TurnCommand(
        emergency=on["danger_now"], caller_stop=not on["danger_now"] and on["caller_stop"], turn_incomplete=on["turn_incomplete"], outcome=outcome, value=val,
        item_ref=_pick(a, "item_ref", t, "item_ref", allowed=frozenset({i.id for i in state.roster}) | {"all"}) if len(state.roster) >= 2 else "",
        followup_answered=bool(on["detail_present"] and kind in ("main", "close") and outcome is TurnOutcome.CONCERN),
        stated=stated, semantic_flags=flags, jump_topic=bool(not closing and kind not in ("profile", "bridge") and on["jump_topic"]),
        next_qid_probs=next_probs, utterance=" ".join(utterance_text.split()), redacted=on["redact_patient_note"])
    diag = {"active_qid": state.cursor.qid, "step_kind": kind, "outcome": outcome.value, "stated": {q: o.value for q, o in stated.items()}, "next_qid_probs": next_probs,
            "semantic_flags": list(flags), "heads": on, **{h: _num(a[h].get("noul")) for h in _NOUL if isinstance(a.get(h), Mapping)}}
    return TurnIntent(command=cmd, needs_profile_extraction=on["mentions_profile"] or (kind == "profile" and outcome is not TurnOutcome.CLARIFY), diagnostics=diag)
