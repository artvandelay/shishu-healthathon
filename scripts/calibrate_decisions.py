"""Offline calibration of the Decision API threshold artifact.

Fits ``DecisionThresholds`` strictly on the dev split, then reports dev and
holdout metrics separately so an overfit is visible instead of hidden.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "code" / "shishu-v2"
CASES_PATH = ROOT / "scripts" / "decision_calibration_cases.json"
RAW_PATH = APP_DIR / "storage" / "calibration_raw_results.json"
OUT_PATH = APP_DIR / "decision_thresholds.json"

load_dotenv(ROOT / ".env")
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import decisions  # noqa: E402
import engine  # noqa: E402
from models import (  # noqa: E402
    AnswerStatus,
    CallState,
    ChildInput,
    ChildSex,
    FollowupSlot,
    MaritalStatus,
    NoteText,
    ProfilePatch,
    SideAction,
    SubStep,
    TurnCommand,
    TurnOutcome,
)
from triage_catalog import get_spec  # noqa: E402

DEV, HOLDOUT = "dev", "holdout"
_MAX_STEPS = 120


def _child_rows(raw: list[dict[str, Any]]) -> tuple[ChildInput, ...]:
    return tuple(
        ChildInput(
            age_years=float(item["age_years"]),
            sex=ChildSex(str(item.get("sex") or "unknown")),
            name=item.get("name"),
        )
        for item in raw
    )


def _apply_profile(state: CallState, profile: dict[str, Any]) -> CallState:
    """Write every stated demographic field, one reviewed patch per field."""
    patches: list[ProfilePatch] = []
    if profile.get("caller_name"):
        patches.append(ProfilePatch(caller_name=profile["caller_name"]))
    if profile.get("marital_status"):
        patches.append(ProfilePatch(marital_status=MaritalStatus(profile["marital_status"])))
    if "children" in profile:
        patches.append(ProfilePatch(children=_child_rows(profile["children"])))
    for patch in patches:
        state = engine.reduce_turn(state, TurnCommand(profile_patch=patch))
    return state


def _replay_prior(state: CallState, qid: str, item: dict[str, Any]) -> CallState:
    """Answer one already-recorded item, keeping its note and follow-up notes."""
    spec = get_spec(qid)
    command = TurnCommand(
        outcome=TurnOutcome(item.get("outcome") or "clear"),
        child_ref=str(item.get("child_ref") or ""),
        utterance=str(item.get("note") or ""),
    )
    state = engine.reduce_turn(state, command)
    notes = [str(note) for note in item.get("followup_notes") or []]
    for _ in range(len(spec.followups) + 2):
        if state.cursor.qid != qid or state.cursor.substep is not SubStep.FOLLOWUP:
            break
        index = state.cursor.followup_idx
        note = notes[index] if index < len(notes) else ""
        state = engine.reduce_turn(
            state, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=note or "हाँ")
        )
    return state


def build_state(setup: dict[str, Any]) -> CallState:
    """Rebuild the exact cursor the calibration case was authored against."""
    state = _apply_profile(engine.new_state(), setup.get("profile") or {})
    priors = {item["qid"]: item for item in setup.get("prior_answers") or []}
    target_qid = setup["active_qid"]
    target_idx = setup.get("pending_followup_idx")

    for _ in range(_MAX_STEPS):
        if state.cursor.qid == target_qid or state.phase.value != "asking":
            break
        prior = priors.get(state.cursor.qid)
        state = (
            _replay_prior(state, state.cursor.qid, prior)
            if prior is not None
            else engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
        )

    prior = priors.get(target_qid) or {}
    if target_idx == -1:
        state = engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome.CONCERN))
    elif isinstance(target_idx, int) and target_idx >= 0:
        state = engine.reduce_turn(
            state,
            TurnCommand(
                outcome=TurnOutcome.CONCERN,
                child_ref=str(prior.get("child_ref") or ""),
                utterance=str(prior.get("note") or "हाँ"),
            ),
        )
        state = _stop_at_followup(state, target_qid, target_idx, prior)
    if state.cursor.qid != target_qid:
        raise AssertionError(f"case did not land on {target_qid}, at {state.cursor.qid}")
    if target_idx is not None and state.cursor.substep is SubStep.FOLLOWUP:
        if state.cursor.followup_idx != target_idx:
            raise AssertionError(f"expected follow-up {target_idx}, at {state.cursor.followup_idx}")
    return state


def _stop_at_followup(
    state: CallState, qid: str, stop_idx: int, prior: dict[str, Any]
) -> CallState:
    """Advance follow-ups until the requested one is the active sub-question."""
    notes = [str(note) for note in prior.get("followup_notes") or []]
    for _ in range(len(notes) + 2):
        if state.cursor.substep is not SubStep.FOLLOWUP or state.cursor.followup_idx >= stop_idx:
            return state
        index = state.cursor.followup_idx
        note = notes[index] if index < len(notes) else ""
        state = engine.reduce_turn(
            state, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=note or "हाँ")
        )
    return state


async def score_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call the live Decision API once per case and capture its raw answers."""
    rows: list[dict[str, Any]] = []
    base_dir = Path(__file__).resolve().parent
    for index, case in enumerate(cases, start=1):
        setup = case["cursor_setup"]
        audio_rel = case.get("audio_wav") or setup.get("audio_wav")
        audio = (base_dir / str(audio_rel)).read_bytes() if audio_rel else None
        request = decisions.build_turn_request(
            build_state(setup),
            utterance_text=setup.get("utterance_text", ""),
            audio_wav_bytes=audio,
        )
        result = await decisions.evaluate_turn(
            request, max_attempts=2, attempt_timeout_sec=5.0, total_budget_sec=10.0
        )
        if not result.ok:
            raise RuntimeError(f"Case {case['id']} failed: {result.error_reason}")
        rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "category": case["category"],
                "audio_wav": audio_rel or "",
                "cursor_setup": setup,
                "gold": case["gold"],
                "decision_ms": result.decision_ms,
                "attempts": result.attempts,
                "request_bytes": result.request_bytes,
                "question_count": result.question_count,
                "catalog_option_count": result.catalog_option_count,
                "usage_present": result.usage_present,
                "total_input_tokens": result.total_input_tokens,
                "raw_answers": result.raw_answers,
            }
        )
        print(f"[{index}/{len(cases)}] {case['id']}: {result.decision_ms}ms", flush=True)
    return rows


def _grid(values: list[float], low: float, high: float, base: list[float]) -> list[float]:
    """Candidate cut points: the base grid plus every midpoint between scores."""
    scored = sorted({round(v, 4) for v in values if low < v < high})
    mids = (round((a + b) / 2.0, 4) for a, b in zip(scored, scored[1:]))
    return sorted({round(v, 4) for v in (*base, *mids) if low < v < high})


def _sweep_noul(pairs: list[tuple[float, bool]], fn_cost: float, fp_cost: float) -> float:
    """Pick the yes-gate that minimises weighted false negatives and positives."""
    best = (float("inf"), 0.0, 0.5)
    for cut in _grid([p for p, _ in pairs], 0.01, 0.99, [x / 20 for x in range(1, 20)]):
        loss = sum(fn_cost for p, gold in pairs if gold and p < cut)
        loss += sum(fp_cost for p, gold in pairs if not gold and p >= cut)
        below = [p for p, _ in pairs if p < cut]
        above = [p for p, _ in pairs if p >= cut]
        margin = min(cut - max(below, default=0.01), min(above, default=0.99) - cut)
        if (loss, -margin) < best[:2]:
            best = (loss, -margin, cut)
    return round(best[2], 4)


def _top_two(
    raw: dict[str, Any], key: str, catchall: str, allowed: frozenset[str]
) -> tuple[str, float, float]:
    """The head's best allowed option with its confidence and margin."""
    evaluation = decisions.evaluate_choice(
        raw, key, catchall_key=catchall, min_confidence=0.0, min_margin=0.0,
        allowed_winners=allowed,
    )
    usable = evaluation.top_option in allowed and evaluation.top_option != catchall
    return (
        evaluation.top_option if usable else "",
        evaluation.top_prob or 0.0,
        evaluation.margin or 0.0,
    )


def _plateau_midpoint(candidates: list[tuple[float, float, float]], floor: float) -> tuple[float, float]:
    best = min(loss for loss, _, _ in candidates)
    tied = [c for c in candidates if c[0] == best and c[1] > floor]
    chosen = tied[len(tied) // 2] if tied else max(candidates, key=lambda c: c[1])
    return round(chosen[1], 4), round(chosen[2], 4)


def _fit_choice(
    rows: list[dict[str, Any]],
    key: str,
    catchall: str,
    allowed: frozenset[str],
    gold: Callable[[dict[str, Any]], Any],
    *,
    fp_cost: float,
    fn_cost: float,
) -> tuple[float, float]:
    """Search confidence x margin for one choice head's cheapest operating point."""
    items = []
    for row in rows:
        option, confidence, margin = _top_two(row["raw_answers"], key, catchall, allowed)
        items.append((option, confidence, margin, str(gold(row) or "")))
    candidates = []
    for min_confidence in _grid([i[1] for i in items if i[0]], 0.01, 0.99, [0.45, 0.55, 0.65, 0.75]):
        for min_margin in _grid([i[2] for i in items if i[0]], 0.0, 0.99, [0.0, 0.05, 0.15, 0.3]):
            loss = 0.0
            for option, confidence, margin, want in items:
                predicted = option if option and confidence >= min_confidence and margin >= min_margin else ""
                if not want or want == catchall:
                    loss += fp_cost if predicted else 0.0
                elif predicted != want:
                    loss += fp_cost if predicted else fn_cost
            candidates.append((loss, min_confidence, min_margin))
    return _plateau_midpoint(candidates, 0.01)


def _gold_side_intent(row: dict[str, Any]) -> str:
    action = row["gold"]["side_action"]
    if action == "clarify_meaning":
        return "clarify_other"
    if action == "add_or_correct":
        return "clear_or_retract" if row["gold"].get("side_polarity") in {"clear", "retract"} else "add_concern"
    return "add_concern" if row["gold"].get("future_qid") else ""


def _non_stop(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r["gold"]["danger_now"] and not r["gold"]["caller_stop"]]


def fit_thresholds(rows: list[dict[str, Any]]) -> decisions.DecisionThresholds:
    """Fit every gate on the dev rows only."""
    calm = _non_stop(rows)
    danger = _sweep_noul(
        [(r["raw_answers"]["danger_now"]["noul"], r["gold"]["danger_now"]) for r in rows], 1000.0, 100.0
    )
    stop = _sweep_noul(
        [(r["raw_answers"]["caller_stop"]["noul"], r["gold"]["caller_stop"]) for r in rows], 100.0, 1000.0
    )
    profile = _sweep_noul(
        [(r["raw_answers"]["mentions_profile"]["noul"], r["gold"]["mentions_profile"]) for r in calm],
        10.0,
        15.0,
    )
    diagnosis = _sweep_noul(
        [
            (r["raw_answers"]["mentions_diagnosis"]["noul"], bool(r["gold"].get("mentions_diagnosis")))
            for r in calm
            if "mentions_diagnosis" in r["raw_answers"]
        ],
        200.0,
        20.0,
    )
    detail = _sweep_noul(
        [
            (r["raw_answers"]["detail_present"]["noul"], r["gold"]["detail_present"])
            for r in calm
            if "detail_present" in r["raw_answers"] and r["gold"]["detail_present"] is not None
        ],
        10.0,
        15.0,
    )
    outcome_conf, outcome_margin = _fit_choice(
        calm, "current_outcome", "other", frozenset(decisions.VALID_OUTCOMES),
        lambda r: r["gold"]["current_outcome"], fp_cost=12.0, fn_cost=10.0,
    )
    side_intent = _fit_choice(
        [r for r in calm if "side_intent" in r["raw_answers"]],
        "side_intent", "other", frozenset(decisions.VALID_SIDE_INTENTS),
        _gold_side_intent, fp_cost=50.0, fn_cost=10.0,
    )
    side_qid = _fit_choice(
        [r for r in calm if "side_qid" in r["raw_answers"]],
        "side_qid", "other", frozenset(spec.id for spec in decisions.routable_specs()),
        lambda r: r["gold"]["side_qid"], fp_cost=100.0, fn_cost=20.0,
    )
    child_rows = [r for r in calm if "child_ref" in r["raw_answers"]]
    child = _fit_choice(
        child_rows, "child_ref", "other", frozenset(_child_ids(child_rows)),
        lambda r: r["gold"]["child_ref"], fp_cost=100.0, fn_cost=10.0,
    )
    return decisions.DecisionThresholds(
        danger_now_min_yes=danger,
        caller_stop_min_yes=stop,
        mentions_profile_min_yes=profile,
        mentions_diagnosis_min_yes=diagnosis,
        detail_present_min_yes=detail,
        outcome_min_confidence=outcome_conf,
        outcome_min_margin=outcome_margin,
        side_intent_min_confidence=side_intent[0],
        side_intent_min_margin=side_intent[1],
        side_qid_min_confidence=side_qid[0],
        side_qid_min_margin=side_qid[1],
        child_ref_min_confidence=child[0],
        child_ref_min_margin=child[1],
    )


def _child_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = ["all"]
    for row in rows:
        ids.extend(child["id"] for child in engine.children_roster(build_state(row["cursor_setup"])))
    return list(dict.fromkeys(ids))


def _ratio(hits: int, total: int, default: float) -> float:
    return round(hits / total, 4) if total else default


def score(
    rows: list[dict[str, Any]], limits: decisions.DecisionThresholds
) -> dict[str, Any]:
    """Interpret the whole split with one threshold set and summarise it."""
    intents = [decisions.interpret_turn_decision(build_state(r["cursor_setup"]), r["raw_answers"],
                                                thresholds=limits,
                                                utterance_text=r["cursor_setup"].get("utterance_text", ""))
               for r in rows]
    paired = list(zip(rows, intents))
    calm = [(r, i) for r, i in paired if not r["gold"]["danger_now"] and not r["gold"]["caller_stop"]]
    emergencies = [(r, i) for r, i in paired if r["gold"]["danger_now"]]
    stops = [(r, i) for r, i in paired if r["gold"]["caller_stop"]]

    side_pairs = [(r["gold"]["side_qid"], i.command.side_qid) for r, i in paired]
    side_hits = sum(1 for want, got in side_pairs if want and got == want)
    side_wrong = sum(1 for want, got in side_pairs if got and got != want)
    wanted_sides = sum(1 for want, _ in side_pairs if want)
    future_rows = [(r, i) for r, i in paired if r["gold"].get("future_qid")]
    child_rows = [(r, i) for r, i in paired if "child_ref" in r["raw_answers"]]
    detail_rows = [
        (r, i) for r, i in calm
        if r["gold"]["current_outcome"] == "concern"
        and r["gold"]["detail_present"] is not None
        and "detail_present" in r["raw_answers"]
    ]
    diagnosis_rows = [(r, i) for r, i in calm if "mentions_diagnosis" in r["raw_answers"]]

    metrics = {
        "n": len(rows),
        "emergency_recall": _ratio(sum(1 for _, i in emergencies if i.command.emergency), len(emergencies), 1.0),
        "caller_stop_recall": _ratio(sum(1 for _, i in stops if i.command.caller_stop), len(stops), 1.0),
        "outcome_exact_accuracy": _ratio(
            sum(1 for r, i in calm if i.command.outcome.value == r["gold"]["current_outcome"]),
            len(calm), 1.0,
        ),
        "side_qid_precision": _ratio(side_hits, side_hits + side_wrong, 1.0),
        "side_qid_recall": _ratio(side_hits, wanted_sides, 1.0),
        "side_qid_false_writes": side_wrong,
        "future_qid_recall": _ratio(
            sum(1 for r, i in future_rows if (i.future_qid_match or {}).get("qid") == r["gold"]["future_qid"]),
            len(future_rows), 1.0,
        ),
        "child_ref_accuracy": _ratio(
            sum(1 for r, i in child_rows if i.command.child_ref == r["gold"]["child_ref"]),
            len(child_rows), 1.0,
        ),
        "detail_present_accuracy": _ratio(
            sum(1 for r, i in detail_rows if i.command.followup_answered == r["gold"]["detail_present"]),
            len(detail_rows), 1.0,
        ),
        "mentions_diagnosis_accuracy": _ratio(
            sum(1 for r, i in diagnosis_rows if i.command.diagnostic == bool(r["gold"].get("mentions_diagnosis"))),
            len(diagnosis_rows), 1.0,
        ),
        "mentions_profile_accuracy": _ratio(
            sum(1 for r, i in calm if i.needs_profile_extraction == r["gold"]["mentions_profile"]),
            len(calm), 1.0,
        ),
    }
    return metrics


def _print_table(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    width = max(len(name) for name in metrics)
    for name, value in metrics.items():
        print(f"  {name.ljust(width)}  {value}")


def report(rows: list[dict[str, Any]]) -> tuple[decisions.DecisionThresholds, dict[str, Any]]:
    """Fit on dev, then score dev and holdout separately with the same artifact."""
    dev = [r for r in rows if r.get("split") == DEV]
    holdout = [r for r in rows if r.get("split") == HOLDOUT]
    limits = fit_thresholds(dev)
    dev_metrics = score(dev, limits)
    holdout_metrics = score(holdout, limits) if holdout else {}
    _print_table(f"DEV (N={len(dev)})", dev_metrics)
    if holdout:
        _print_table(f"HOLDOUT (N={len(holdout)})", holdout_metrics)
    return limits, {"dev": dev_metrics, HOLDOUT: holdout_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", action="store_true", help="re-score saved raw answers")
    parser.add_argument("--write", action="store_true", help="overwrite the threshold artifact")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.from_raw and RAW_PATH.exists():
        rows = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    else:
        rows = asyncio.run(score_cases(cases))
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    limits, metrics = report(rows)
    print("\nTHRESHOLDS")
    print(json.dumps(asdict(limits), indent=2))
    if args.write:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        artifact = {
            "provisional": True,
            "note": f"Fitted on the dev split (N={len(rows)} fixtures); holdout reported separately.",
            "n_cases": len(rows),
            "class_counts": counts,
            "metrics": metrics,
            "thresholds": asdict(limits),
        }
        OUT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        decisions.load_thresholds(OUT_PATH)
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
