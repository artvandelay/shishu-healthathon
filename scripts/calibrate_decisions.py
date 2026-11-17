"""Pack-aware offline calibration of the eleven Decision API gates.

Rebuilds each `cursor_setup` of `scripts/decision_calibration_cases.json` into a real `CallState` through the pure
reducer, calls `intake.decisions` once per case (raw answers cached in `intake/storage/calibration_raw_<pack>.json`),
fits on `dev` only, and prints `DEV` / `HOLDOUT` tables so overfit stays visible. `--write` rewrites just the
`thresholds:` block of `packs/<pack_id>/decisions.yaml`; `--dry-check` builds all 50 cases with zero network calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from intake import decisions, engine  # noqa: E402
from intake.models import (  # noqa: E402
    CallState, PackConfig, Phase, ProfilePatch, RosterInput, SubStep, TurnCommand, TurnOutcome,
)
from intake.pack import DEFAULT_PACK_ID, load_pack  # noqa: E402

CASES = ROOT / "scripts" / "decision_calibration_cases.json"
RAW_DIR, MAX_STEPS, DEV, HOLDOUT = ROOT / "intake" / "storage", 160, "dev", "holdout"
OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract"})
STATED = frozenset({"concern", "clear", "clarify"})
# (false-negative, false-positive) cost: missing an emergency is the most expensive mistake this pack can make.
NOUL_COST = {"danger_now": (1000.0, 100.0), "caller_stop": (100.0, 1000.0), "mentions_profile": (10.0, 15.0),
             "redact_patient_note": (200.0, 20.0), "detail_present": (10.0, 15.0),
             # The fan-out gate: a missed other-topic strands a volunteered complaint for the whole call, while a
             # false yes only buys one extra routing call, so this is deliberately the loosest gate in the pack.
             "mentions_other_topic": (200.0, 10.0)}
CHOICE_COST = {"active_outcome": (12.0, 10.0, "outcome"), "stated_": (50.0, 10.0, "stated"),
               "item_ref": (100.0, 10.0, "item_ref")}
FITTED = tuple(f"{k}_min_yes" for k in NOUL_COST) + tuple(
    f"{n}_min_{s}" for _, (_, _, n) in CHOICE_COST.items() for s in ("confidence", "margin"))
GRID = tuple(round(i / 40, 4) for i in range(1, 40))          # yes-gates 0.025 .. 0.975
CONFIDENCE = tuple(round(i / 20, 4) for i in range(9, 17))    # 0.45 .. 0.80
MARGIN = tuple(round(i / 20, 4) for i in range(7))            # 0.00 .. 0.30
RATES = ("emergency_recall", "caller_stop_recall", "outcome_accuracy", "detail_accuracy", "profile_accuracy",
         "redact_accuracy", "item_ref_accuracy", "stated_accuracy", "other_topic_gate_recall")
COUNTS = ("calm_false_stop", "stated_false_writes")
Row, Ev = Mapping[str, Any], dict[str, Any]


def _answer(pack: PackConfig, state: CallState, item: Row | None, stop: int | None = None) -> CallState:
    """Answer one recorded step, then feed its notes and follow-up notes in order, pausing at `stop`."""
    item, notes = item or {}, [str(n) for n in (item or {}).get("followup_notes") or ()]
    state = engine.reduce_turn(pack, state, TurnCommand(
        outcome=TurnOutcome(str(item.get("outcome") or "clear")), item_ref=str(item.get("item_ref") or ""),
        utterance=str(item.get("note") or "")))
    qid = state.cursor.qid
    while state.cursor.qid == qid and state.cursor.substep is SubStep.FOLLOWUP:
        at = state.cursor.followup_idx
        if stop is not None and at >= stop:
            break
        said = notes[at] if at < len(notes) else ""
        state = engine.reduce_turn(pack, state, TurnCommand(
            outcome=TurnOutcome.CONCERN if said else TurnOutcome.UNKNOWN, utterance=said))
    return state


def build_state(pack: PackConfig, setup: Row) -> CallState:
    """Rebuild the exact cursor a case was authored against; a bad landing is a loud error, never a guess."""
    prof, kids = setup.get("profile") or {}, tuple((setup.get("profile") or {}).get("children") or ())
    patches = [ProfilePatch(slots={k: str(prof[k])}) for k in ("caller_name", "marital_status") if prof.get(k)]
    if "children" in prof:
        patches.append(ProfilePatch(roster_complete=True, roster=tuple(
            RosterInput(age_years=float(c["age_years"]), name=c.get("name"),
                        tag=str(c.get("tag") or c.get("sex") or "unknown")) for c in kids)))
    state, priors = engine.new_state(pack), {str(p["qid"]): p for p in setup.get("prior_answers") or ()}
    for patch in patches:
        state = engine.reduce_turn(pack, state, TurnCommand(profile_patch=patch))
    target, stop = str(setup["active_qid"]), setup.get("pending_followup_idx")
    for _ in range(MAX_STEPS):
        if state.phase is not Phase.ASKING or state.cursor.qid == target:
            break
        state = _answer(pack, state, priors.get(state.cursor.qid))
    if stop == -1:  # -1 parks on the which_item substep: a bare yes, no child named yet.
        state = _answer(pack, state, {**priors.get(target, {}), "outcome": "concern", "item_ref": ""})
    elif isinstance(stop, int) and stop >= 0:
        state = _answer(pack, state, {**priors.get(target, {}), "outcome": "concern"}, stop)
    if state.cursor.qid != target or state.phase is not Phase.ASKING:
        raise AssertionError(f"case did not land on {target}, at {state.cursor.qid}/{state.phase}")
    if isinstance(stop, int) and stop >= 0 and state.cursor.followup_idx != stop:
        raise AssertionError(f"expected follow-up {stop}, at {state.cursor.followup_idx}")
    return state


def _request(pack: PackConfig, case: Row) -> dict[str, Any]:
    """The one Decision request a case's rebuilt cursor and utterance produce."""
    setup, rel = case["cursor_setup"], case.get("audio_wav") or case["cursor_setup"].get("audio_wav")
    return decisions.build_turn_request(
        pack, build_state(pack, setup), utterance_text=str(setup.get("utterance_text") or ""),
        audio_wav_bytes=(ROOT / "scripts" / str(rel)).read_bytes() if rel else None)


async def score_cases(pack: PackConfig, cases: Sequence[Row]) -> list[dict[str, Any]]:
    """One live Decision call per case, capturing every raw head answer."""
    rows = []
    for index, case in enumerate(cases, start=1):
        result = await decisions.evaluate_turn(_request(pack, case), max_attempts=2,
                                               attempt_timeout_sec=5.0, total_budget_sec=10.0)
        if not result.ok:
            raise RuntimeError(f"case {case['id']} failed: {result.error_reason}")
        rows.append({k: case[k] for k in ("id", "split", "category", "gold", "cursor_setup")} | {
            "decision_ms": result.decision_ms, "attempts": result.attempts, "raw_answers": result.raw_answers,
            "request_bytes": result.request_bytes, "question_count": result.question_count,
            "total_input_tokens": result.total_input_tokens})
        print(f"[{index}/{len(cases)}] {case['id']}: {result.decision_ms}ms", flush=True)
    return rows


_ROSTER_PROFILE_KEYS = frozenset({"children", "roster"})


def _profile_keys(pack: PackConfig) -> set[str]:
    """Every profile field this pack can actually store: declared slots plus the fields its rules read."""
    return (set(pack.profile_slots) | {q.saves_to_profile for q in pack.questions if q.saves_to_profile}
            | {str(rule["field"]) for rule in pack.rules.values() if rule.get("field")})


def _owns(pack: PackConfig, case: Row) -> bool:
    """Whether a calibration case was authored against this pack's form and profile fields.

    `decision_calibration_cases.json` is one shared file with no pack column, so ownership is read off the two
    things a case cannot fake: the question it lands on and the profile fields it fills. Replaying another pack's
    cases proves nothing about this pack, and a green tick from them would be worse than a loud skip.
    """
    setup = case.get("cursor_setup") or {}
    if str(setup.get("active_qid") or "") not in pack.by_id:
        return False
    return set(setup.get("profile") or {}) - _ROSTER_PROFILE_KEYS <= _profile_keys(pack)


def _form_probe(pack: PackConfig) -> list[str]:
    """Every question in the pack must build a real request at its own cursor, with no case file involved.

    A pack with no cases of its own still has a Decision contract to hold: the gate head must travel on every step
    that could route, and the full request must carry its Group A heads wherever it lands.
    """
    bad = []
    for spec in pack.questions:
        state = engine.new_state(pack)
        state.cursor = state.cursor.model_copy(update={"qid": spec.id})
        try:
            request = decisions.build_turn_request(pack, state, utterance_text="नहीं")
            assert request["model"] and request["input"], "empty request"
            assert "active_outcome" in request["questions"], "missing active_outcome head"
            routable = engine.step_kind(pack, state) != "profile"
            assert not routable or "mentions_other_topic" in request["questions"], "gate head missing on a routing step"
        except Exception as exc:  # noqa: BLE001 - every failure is data for this report
            bad.append(f"{spec.id}: {type(exc).__name__}: {exc}")
    print(f"{len(pack.questions) - len(bad)}/{len(pack.questions)} form questions built successfully (0 network calls)")
    return bad


def dry_check(pack: PackConfig, cases: Sequence[Row]) -> int:
    """Every case this pack owns must build to its expected cursor and a full request, with zero network calls."""
    missing = [] if "mentions_other_topic" in (pack.decisions.get("heads") or {}) else [
        "decisions.yaml has no mentions_other_topic head, so every turn would short-circuit the fan-out"]
    owned = [case for case in cases if _owns(pack, case)]
    bad = []
    for case in owned:
        try:
            request = _request(pack, case)
            assert request["model"] and request["input"], "empty request"
            assert "active_outcome" in request["questions"], "missing active_outcome head"
        except Exception as exc:  # noqa: BLE001 - every failure is data for this report
            bad.append(f"{case['id']}: {type(exc).__name__}: {exc}")
    print(f"{len(owned) - len(bad)}/{len(owned)} cursor_setup cases built successfully (0 network calls)")
    if skipped := len(cases) - len(owned):
        print(f"{skipped} case(s) in {CASES.name} belong to another pack's form and were not replayed here")
    if not owned:
        bad += _form_probe(pack)
    print("".join(f"\n  FAIL {line}" for line in bad + missing))
    return 0 if not bad and not missing else 1


def _prob(row: Row, key: str) -> float | None:
    """A noul head's probability, or None when that head was not sent for this row."""
    head = (row.get("raw_answers") or {}).get(key)
    value = head.get("noul") if isinstance(head, Mapping) else None
    return float(value) if isinstance(value, (int, float)) else None


def _top(row: Row, key: str, allowed: Iterable[str]) -> tuple[str, float, float]:
    """The head's best allowed option, its confidence, and its margin over the runner-up."""
    head = (row.get("raw_answers") or {}).get(key)
    probs = (head.get("probabilities") if isinstance(head, Mapping) else None) or {}
    ranked = sorted(((str(k), float(v)) for k, v in probs.items() if isinstance(v, (int, float))),
                    key=lambda kv: (-kv[1], kv[0]))
    if not ranked or ranked[0][0] not in set(allowed):
        return "", 0.0, 0.0
    return ranked[0][0], ranked[0][1], ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)


def evidence(pack: PackConfig, rows: Sequence[Row]) -> list[Ev]:
    """One pass per row: its cursor, the gold each choice head should reach, and every head it actually sent."""
    out = []
    for row in rows:
        state, gold = build_state(pack, row["cursor_setup"]), row["gold"]
        kids = (row["cursor_setup"].get("profile") or {}).get("children") or ()
        ids = {str(k["id"]): i.id for k, i in zip(kids, state.roster) if k.get("id")}
        ref, side = str(gold.get("child_ref") or ""), str(gold.get("side_qid") or "")
        pol = str(gold.get("side_polarity") or "")
        want = {"active_outcome": str(gold.get("current_outcome") or ""),
                "item_ref": ref if ref == "all" else ids.get(ref, ref)}
        if side:
            want[f"stated_{side}"] = ("clarify" if str(gold.get("side_action") or "") == "clarify_meaning"
                                     else ("clear" if pol == "retract" else pol))
        raw = row.get("raw_answers") or {}
        allowed = {"active_outcome": OUTCOMES, "item_ref": {"all", *ids.values()}} | {
            k: STATED for k in raw if str(k).startswith("stated_")}
        out.append({"row": row, "state": state, "kind": engine.step_kind(pack, state), "raw": raw, "want": want,
                    "heads": {h: _top(row, h, o) for h, o in allowed.items() if h in raw}})
    return out


def _plateau(scored: list[tuple[float, float, float]]) -> tuple[float, float]:
    """The cheapest operating point, taken at the middle of its plateau so one lucky cut never wins."""
    best = min(sc[0] for sc in scored)
    tied = [sc for sc in scored if sc[0] == best and sc[1] > 0.0]
    chosen = tied[len(tied) // 2] if tied else max(scored, key=lambda sc: (sc[1], sc[2]))
    return round(chosen[1], 4), round(chosen[2], 4)


def _sweep(pairs: list[tuple[float, bool]], fn_cost: float, fp_cost: float) -> float:
    """The yes-gate on GRID that minimises weighted misses, ties broken by the widest gap."""
    scored = [
        (
            sum(fn_cost for p, gold in pairs if gold and p < cut)
            + sum(fp_cost for p, gold in pairs if not gold and p >= cut),
            min(
                cut - max([p for p, _ in pairs if p < cut], default=0.0),
                min([p for p, _ in pairs if p >= cut], default=1.0) - cut,
            ),
            cut,
        )
        for cut in GRID
    ]
    best = min(sc[0] for sc in scored)
    best_gap = max(sc[1] for sc in scored if sc[0] == best)
    tied = [sc[2] for sc in scored if sc[0] == best and round(sc[1], 6) == round(best_gap, 6)]
    return round(tied[len(tied) // 2], 4)


def _fit(items: list[tuple[str, float, float, str]], fp_cost: float, fn_cost: float) -> tuple[float, float]:
    """Search confidence x margin for one choice head family's cheapest stable operating point."""
    if not items:
        return 0.5, 0.0
    return _plateau([(sum(0.0 if (pred := o if o and c >= conf and m >= margin else "") == w
                         else (fp_cost if pred else (fn_cost if w else 0.0)) for o, c, m, w in items), conf, margin)
                     for conf in CONFIDENCE for margin in MARGIN])


def fit_thresholds(pack: PackConfig, ev: list[Ev]) -> dict[str, float]:
    """Fit every gate on the dev evidence only, leaving the pack's other thresholds untouched."""
    t = {k: float(v) for k, v in (pack.decisions.get("thresholds") or {}).items()}
    calm = [e for e in ev if not e["row"]["gold"]["danger_now"] and not e["row"]["gold"]["caller_stop"]]
    detail = [e for e in calm if e["kind"] in ("main", "close") and e["row"]["gold"].get("detail_present") is not None]
    for key, (fn_cost, fp_cost) in NOUL_COST.items():
        gold = "mentions_diagnosis" if key == "redact_patient_note" else key
        # A profile mention only exists off the calm path, and a follow-up step answers its own detail.
        scope = {"mentions_profile": calm, "detail_present": detail}.get(key, ev)
        if key == "mentions_other_topic":
            # The gate's own gold is "this turn also touched another topic": any named side question or any
            # volunteered future question. Only calm turns matter, because a hot turn short-circuits anyway.
            # A raw cache recorded before this head existed has no pair to fit on, and an absent gate must keep the
            # pack's authored floor rather than be silently swept down to the grid's lowest cut.
            pairs = [(p, bool(e["row"]["gold"].get("side_qid") or e["row"]["gold"].get("future_qid")))
                     for e in calm if (p := _prob(e["row"], key)) is not None]
            if pairs:
                t[f"{key}_min_yes"] = _sweep(pairs, fn_cost, fp_cost)
            continue
        t[f"{key}_min_yes"] = _sweep([(p, bool(e["row"]["gold"].get(gold)))
                                      for e in scope if (p := _prob(e["row"], key)) is not None], fn_cost, fp_cost)
    for family, (fp_cost, fn_cost, name) in CHOICE_COST.items():
        t[f"{name}_min_confidence"], t[f"{name}_min_margin"] = _fit(
            [(o, c, m, e["want"].get(h, "")) for e in ev
             for h, (o, c, m) in e["heads"].items() if h.startswith(family)], fp_cost, fn_cost)
    return {k: round(float(t[k]), 4) for k in FITTED}


def _checks(e: Ev, intent: Any) -> Iterable[tuple[str, bool]]:
    """Every (metric, ok) this row can be scored on; a metric it cannot exercise is simply absent."""
    gold, raw, kind, want = e["row"]["gold"], e["raw"], e["kind"], e["want"]
    command, side, calm = intent.command, str(gold.get("side_qid") or ""), not gold["danger_now"] and not gold["caller_stop"]
    if gold["danger_now"]:
        yield "emergency_recall", bool(command.emergency)
    if gold["caller_stop"]:
        yield "caller_stop_recall", bool(command.caller_stop)
    if calm:
        yield "calm_false_stop", bool(command.emergency or command.caller_stop)
        yield "outcome_accuracy", command.outcome.value == gold["current_outcome"]
        yield "profile_accuracy", intent.needs_profile_extraction == bool(gold["mentions_profile"])
    if kind in ("main", "close") and "detail_present" in raw and gold.get("detail_present") is not None:
        yield "detail_accuracy", command.followup_answered == bool(gold["detail_present"])
    if "redact_patient_note" in raw:
        yield "redact_accuracy", bool(command.redacted) == bool(gold.get("mentions_diagnosis"))
    if "item_ref" in raw:
        yield "item_ref_accuracy", command.item_ref == want["item_ref"]
    if side:
        yield "stated_accuracy", command.stated.get(side) == want.get(f"stated_{side}", "")
    if (gold.get("side_qid") or gold.get("future_qid")) and "mentions_other_topic" in raw:
        # Pass 2 only ever runs because this gate fired, so its recall is the ceiling on every routing gain.
        yield "other_topic_gate_recall", bool(intent.diagnostics.get("heads", {}).get("mentions_other_topic"))
    yield "stated_false_writes", bool(command.stated) and not side


def score(pack: PackConfig, ev: list[Ev], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Interpret one split with one threshold set, reporting a rate per metric and a count per leak."""
    base = {k: float(v) for k, v in (pack.decisions.get("thresholds") or {}).items()}
    scoped = pack.model_copy(update={"decisions": {**pack.decisions, "thresholds": base | dict(thresholds)}})
    checks = [c for e in ev for c in _checks(e, decisions.interpret_turn_decision(
        scoped, e["state"], e["raw"], utterance_text=str(e["row"]["cursor_setup"].get("utterance_text") or "")))]
    metrics: dict[str, Any] = {"n": len(ev)}
    for name in RATES:
        flags = [ok for metric, ok in checks if metric == name]
        metrics[name] = round(sum(flags) / len(flags), 4) if flags else 1.0
    metrics |= {name: sum(1 for metric, ok in checks if metric == name and ok) for name in COUNTS}
    return metrics


def _table(title: str, metrics: Mapping[str, Any]) -> None:
    """One aligned column per table, so DEV and HOLDOUT are read, never confused."""
    width = max(len(k) for k in metrics)
    print(f"\n{title}\n" + "\n".join(f"  {n.ljust(width)}  {v}" for n, v in metrics.items()))


def write_thresholds(pack: PackConfig, fitted: Mapping[str, float]) -> Path:
    """Rewrite only the trailing `thresholds:` block, leaving every head and comment intact."""
    path = pack.root / "decisions.yaml"
    head, _, tail = path.read_text(encoding="utf-8").partition("\nthresholds:")
    merged = (dict(yaml.safe_load(tail) or {}) if tail else {}) | dict(fitted)
    path.write_text(head.rstrip("\n") + "\n\n" + yaml.safe_dump(
        {"thresholds": merged}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=DEFAULT_PACK_ID, help="pack id under packs/ (default shishu-hi)")
    parser.add_argument("--from-raw", action="store_true", help="re-score the cached raw answers, no network")
    parser.add_argument("--write", action="store_true", help="fold the fitted thresholds into the pack")
    parser.add_argument("--dry-check", action="store_true", help="only verify every cursor_setup builds, offline")
    args = parser.parse_args()
    pack, cases = load_pack(args.pack), json.loads(CASES.read_text(encoding="utf-8"))
    if args.dry_check:
        return dry_check(pack, cases)
    raw = RAW_DIR / f"calibration_raw_{pack.id}.json"
    cached = args.from_raw and raw.is_file()
    rows = json.loads(raw.read_text(encoding="utf-8")) if cached else asyncio.run(score_cases(pack, cases))
    if not cached:
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dev, holdout = ([r for r in rows if r.get("split") == s] for s in (DEV, HOLDOUT))
    dev_ev = evidence(pack, dev)
    fitted = fit_thresholds(pack, dev_ev)
    _table(f"DEV (N={len(dev)})", score(pack, dev_ev, fitted))
    if holdout:
        _table(f"HOLDOUT (N={len(holdout)})", score(pack, evidence(pack, holdout), fitted))
    print("\nTHRESHOLDS\n" + json.dumps(fitted, indent=2))
    if args.write:
        print(f"\nwrote {write_thresholds(pack, fitted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
