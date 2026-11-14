"""Phase 7 Decisions API calibration sweep against real gemini-decision-1 scores."""

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
OUT_PATH = ROOT / "scripts" / "provisional_staging_thresholds.json"

load_dotenv(ROOT / ".env")
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import call_cursor  # noqa: E402
import decisions  # noqa: E402


def _advance_followups(
    state: dict[str, Any],
    cur: str,
    item: dict[str, Any],
    stop_idx: int | None = None,
) -> dict[str, Any]:
    fnotes = item.get("followup_notes") or []
    while state.get("active_qid") == cur and state.get("pending_followup_idx") not in (
        None,
        stop_idx,
    ):
        pidx = state["pending_followup_idx"]
        if pidx == -1:
            kw = {"child_ref": item.get("child_ref") or "child_1"}
        else:
            kw = {"note": fnotes[pidx] if pidx < len(fnotes) else "हाँ"}
        state = call_cursor.apply_turn(state, "concern", **kw)
    return state


def build_state(setup: dict[str, Any]) -> dict[str, Any]:
    prof = setup.get("profile") or {}
    state = call_cursor.new_cursor_state()
    for field in ("caller_name", "marital_status"):
        if prof.get(field):
            state, _ = call_cursor.apply_profile(state, **{field: prof[field]})
    if "children" in prof:
        state, _ = call_cursor.apply_profile(state, children=prof["children"])
    if prof.get("ever_pregnant") is not None and state["profile"]["ever_pregnant"] is None:
        state, _ = call_cursor.apply_profile(state, ever_pregnant=prof["ever_pregnant"])

    priors = {item["qid"]: item for item in setup.get("prior_answers") or []}
    target_qid = setup["active_qid"]
    target_idx = setup.get("pending_followup_idx")
    for _ in range(90):
        cur = state.get("active_qid")
        if cur == target_qid or state.get("phase") != "asking":
            break
        if cur in priors:
            it = priors[cur]
            state = call_cursor.apply_turn(
                state,
                it["outcome"],
                child_ref=it.get("child_ref", ""),
                note=it.get("note", ""),
            )
            state = _advance_followups(state, cur, it)
        else:
            state = call_cursor.apply_turn(state, "clear")

    if target_idx == -1:
        state = call_cursor.apply_turn(state, "concern")
    elif isinstance(target_idx, int) and target_idx >= 0:
        it = priors.get(target_qid) or {}
        state = call_cursor.apply_turn(
            state,
            "concern",
            child_ref=it.get("child_ref", ""),
            note=it.get("note", "हाँ"),
        )
        state = _advance_followups(state, target_qid, it, stop_idx=target_idx)
    assert (
        state.get("active_qid") == target_qid
        and state.get("pending_followup_idx") == target_idx
    )
    return state


async def score_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        setup = case["cursor_setup"]
        req = decisions.build_turn_request(
            build_state(setup),
            utterance_text=setup["utterance_text"],
        )
        res = await decisions.evaluate_turn(
            req,
            max_attempts=2,
            attempt_timeout_sec=5.0,
            total_budget_sec=10.0,
        )
        if not res.ok:
            raise RuntimeError(f"Case {case['id']} failed: {res.error_reason}")
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "cursor_setup": setup,
                "gold": case["gold"],
                "decision_ms": res.decision_ms,
                "attempts": res.attempts,
                "request_bytes": res.request_bytes,
                "question_count": res.question_count,
                "catalog_option_count": res.catalog_option_count,
                "usage_present": res.usage_present,
                "total_input_tokens": res.total_input_tokens,
                "raw_answers": res.raw_answers,
            }
        )
        print(f"[{idx}/{len(cases)}] {case['id']}: {res.decision_ms}ms", flush=True)
    return rows


def _midpoints(vals: list[float], lo: float, hi: float, base: list[float]) -> list[float]:
    pts = sorted({round(v, 4) for v in vals if lo < v < hi})
    mids = (round((a + b) / 2.0, 4) for a, b in zip(pts, pts[1:]))
    return sorted({round(v, 4) for v in (*base, *mids) if lo < v < hi})


def _sweep_noul(pairs: list[tuple[float, bool]], fn_cost: float, fp_cost: float) -> float:
    best = (float("inf"), 1.0, 0.5)
    grid = _midpoints([p for p, _ in pairs], 0.01, 0.99, [x / 20 for x in range(1, 20)])
    for t in grid:
        loss = sum(fn_cost for p, g in pairs if g and p < t) + sum(
            fp_cost for p, g in pairs if not g and p >= t
        )
        below = [p for p, _ in pairs if p < t]
        above = [p for p, _ in pairs if p >= t]
        margin = min(
            t - (max(below) if below else 0.01),
            (min(above) if above else 0.99) - t,
        )
        if (loss, -margin) < best[:2]:
            best = (loss, -margin, t)
    return round(best[2], 4)


def _extract_choice(
    rows: list[dict[str, Any]],
    qkey: str,
    catchall: str,
    allowed: frozenset[str],
    gold_fn: Callable[[dict[str, Any]], Any],
) -> list[tuple[str, float, float, str]]:
    out: list[tuple[str, float, float, str]] = []
    for r in rows:
        ev = decisions.evaluate_choice(
            r["raw_answers"],
            qkey,
            catchall_key=catchall,
            min_confidence=0.0,
            min_margin=0.0,
            allowed_winners=allowed,
        )
        ok = bool(
            ev.top_option
            and ev.top_option != catchall
            and ev.top_option in allowed
            and ev.top_prob is not None
        )
        out.append(
            (
                ev.top_option if ok else "",
                ev.top_prob or 0.0,
                ev.margin or 0.0,
                str(gold_fn(r) or ""),
            )
        )
    return out


def _sweep_choice(
    rows: list[dict[str, Any]],
    qkey: str,
    catchall: str,
    allowed: frozenset[str],
    gold_fn: Callable[[dict[str, Any]], Any],
    fp_cost: float,
    fn_cost: float,
) -> tuple[float, float]:
    items = _extract_choice(rows, qkey, catchall, allowed, gold_fn)
    confs = _midpoints(
        [x[1] for x in items if x[0]],
        0.01,
        0.99,
        [0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    )
    margs = _midpoints(
        [x[2] for x in items if x[0]],
        0.0,
        0.99,
        [0.0, 0.05, 0.1, 0.2, 0.3],
    )
    scored: list[tuple[float, float, float]] = []
    for mc in confs:
        for mm in margs:
            loss = 0.0
            for opt, tp, mg, gold in items:
                pred = opt if (opt and tp >= mc and mg >= mm) else ""
                if gold in ("", catchall):
                    if pred:
                        loss += fp_cost
                elif pred != gold:
                    loss += fp_cost if pred else fn_cost
            scored.append((loss, mc, mm))
    min_loss = min(y[0] for y in scored)
    plat = [x for x in scored if x[0] == min_loss]
    mid = plat[len(plat) // 2]
    return round(mid[1], 4), round(mid[2], 4)


def _macro_pr(golds: list[str], preds: list[str]) -> tuple[float, float]:
    ps, rs = [], []
    for lbl in sorted(set(golds)):
        tp = sum(1 for g, p in zip(golds, preds) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(golds, preds) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(golds, preds) if g == lbl and p != lbl)
        ps.append(tp / (tp + fp) if (tp + fp) else 1.0)
        rs.append(tp / (tp + fn) if (tp + fn) else 1.0)
    return round(sum(ps) / len(ps), 4), round(sum(rs) / len(rs), 4)


def _gold_side_intent(r: dict[str, Any]) -> str:
    act = r["gold"]["side_action"]
    pol = r["gold"].get("side_polarity") or ""
    if act == "clarify_meaning":
        return "clarify_other"
    if act == "add_or_correct":
        return "clear_or_retract" if pol in ("clear", "retract") else "add_concern"
    if r["gold"].get("future_qid"):
        return "add_concern"
    return ""


def calibrate(rows: list[dict[str, Any]]) -> tuple[decisions.DecisionThresholds, dict[str, Any]]:
    non_stop = [
        r for r in rows if not r["gold"]["danger_now"] and not r["gold"]["caller_stop"]
    ]
    danger_t = _sweep_noul(
        [(r["raw_answers"]["danger_now"]["noul"], r["gold"]["danger_now"]) for r in rows],
        1000.0,
        100.0,
    )
    stop_t = _sweep_noul(
        [(r["raw_answers"]["caller_stop"]["noul"], r["gold"]["caller_stop"]) for r in rows],
        100.0,
        1000.0,
    )
    prof_t = _sweep_noul(
        [
            (r["raw_answers"]["mentions_profile"]["noul"], r["gold"]["mentions_profile"])
            for r in non_stop
        ],
        10.0,
        15.0,
    )
    det_rows = [
        (r["raw_answers"]["detail_present"]["noul"], r["gold"]["detail_present"])
        for r in non_stop
        if "detail_present" in r["raw_answers"] and r["gold"]["detail_present"] is not None
    ]
    detail_t = _sweep_noul(det_rows, 10.0, 15.0)

    out_c, out_m = _sweep_choice(
        non_stop,
        "current_outcome",
        "other",
        decisions.VALID_OUTCOMES,
        lambda r: r["gold"]["current_outcome"],
        12.0,
        10.0,
    )
    child_rows = [r for r in non_stop if "child_ref" in r["raw_answers"]]
    ch_c, ch_m = _sweep_choice(
        child_rows,
        "child_ref",
        "other",
        frozenset({"child_1", "child_2", "child_3", "all"}),
        lambda r: r["gold"]["child_ref"],
        100.0,
        10.0,
    )
    si_c, si_m = _sweep_choice(
        non_stop,
        "side_intent",
        "other",
        decisions.VALID_SIDE_INTENTS,
        _gold_side_intent,
        50.0,
        10.0,
    )

    states = {r["id"]: build_state(r["cursor_setup"]) for r in rows}
    routable_qids = frozenset(call_cursor.routable_qids())
    q_items = _extract_choice(
        non_stop,
        "side_qid",
        "other",
        routable_qids,
        lambda r: r["gold"]["side_qid"],
    )
    i_items = _extract_choice(
        non_stop,
        "side_intent",
        "other",
        decisions.VALID_SIDE_INTENTS,
        _gold_side_intent,
    )
    sq_confs = _midpoints(
        [x[1] for x in q_items if x[0]],
        0.01,
        0.99,
        [0.45, 0.55, 0.65, 0.75, 0.85],
    )
    sq_ms = _midpoints(
        [x[2] for x in q_items if x[0]],
        0.0,
        0.99,
        [0.05, 0.15, 0.25, 0.35],
    )

    sq_scored: list[tuple[float, float, float]] = []
    for sc in sq_confs:
        for sm in sq_ms:
            loss = 0.0
            for r, (qopt, qtp, qm, gold_side), (iopt, itp, im, _) in zip(
                non_stop, q_items, i_items
            ):
                ok_q = bool(qopt and qtp >= sc and qm >= sm)
                ok_i = bool(iopt and itp >= si_c and im >= si_m)
                ok_o = ok_q and ok_i
                targ = call_cursor.accepted_target(states[r["id"]], qopt) if ok_o else ""
                if not gold_side and targ:
                    loss += 10000.0
                elif gold_side and targ != gold_side:
                    loss += 10000.0 if targ else 100.0
                fut = r["gold"].get("future_qid")
                if fut and not (ok_o and not targ and qopt == fut):
                    loss += 50.0
            sq_scored.append((loss, sc, sm))
    sq_min = min(y[0] for y in sq_scored)
    sq_plat = [x for x in sq_scored if x[0] == sq_min]
    sq_mid = sq_plat[len(sq_plat) // 2]

    th = decisions.validate_thresholds(
        decisions.DecisionThresholds(
            calibrated=True,
            danger_now_min_yes=danger_t,
            caller_stop_min_yes=stop_t,
            mentions_profile_min_yes=prof_t,
            detail_present_min_yes=detail_t,
            outcome_min_confidence=out_c,
            outcome_min_margin=out_m,
            side_intent_min_confidence=si_c,
            side_intent_min_margin=si_m,
            side_qid_min_confidence=round(sq_mid[1], 4),
            side_qid_min_margin=round(sq_mid[2], 4),
            child_ref_min_confidence=ch_c,
            child_ref_min_margin=ch_m,
        ),
        require_calibrated=True,
    )

    interps = [
        decisions.interpret_turn_decision(
            states[r["id"]],
            r["raw_answers"],
            utterance_text=r["cursor_setup"]["utterance_text"],
            thresholds=th,
            require_calibrated=True,
        )
        for r in rows
    ]
    emg_pos = [r for r in rows if r["gold"]["danger_now"]]
    hist_neg = [r for r in rows if r["category"] == "historical_lookalike_negative"]
    ns_pairs = [
        (r, it)
        for r, it in zip(rows, interps)
        if not r["gold"]["danger_now"] and not r["gold"]["caller_stop"]
    ]
    out_p, out_r = _macro_pr(
        [r["gold"]["current_outcome"] for r, _ in ns_pairs],
        [it.cursor_payload["current_outcome"] for _, it in ns_pairs],
    )
    side_preds = [
        (r["gold"]["side_qid"], it.cursor_payload.get("side_qid", ""))
        for r, it in zip(rows, interps)
    ]
    side_tp = sum(1 for g, p in side_preds if g and p == g)
    side_fp = sum(1 for g, p in side_preds if p and p != g)
    stop_pos = sum(1 for r in rows if r["gold"]["caller_stop"])
    stop_neg = sum(1 for r in rows if not r["gold"]["caller_stop"])
    fut_pos = sum(1 for r in rows if r["gold"].get("future_qid"))

    metrics = {
        "emergency_recall": round(
            sum(
                1
                for r, it in zip(rows, interps)
                if r["gold"]["danger_now"] and it.cursor_payload["emergency"]
            )
            / len(emg_pos),
            4,
        ),
        "history_false_emergency_rate": round(
            sum(
                1
                for r, it in zip(rows, interps)
                if r["category"] == "historical_lookalike_negative"
                and it.cursor_payload["emergency"]
            )
            / len(hist_neg),
            4,
        ),
        "danger_now_separation_gap": [
            round(
                max(
                    r["raw_answers"]["danger_now"]["noul"]
                    for r in rows
                    if not r["gold"]["danger_now"]
                ),
                4,
            ),
            round(min(r["raw_answers"]["danger_now"]["noul"] for r in emg_pos), 4),
        ],
        "caller_stop_recall": round(
            sum(
                1
                for r, it in zip(rows, interps)
                if r["gold"]["caller_stop"] and it.cursor_payload["caller_stop"]
            )
            / stop_pos,
            4,
        ),
        "caller_stop_false_positive_rate": round(
            sum(
                1
                for r, it in zip(rows, interps)
                if not r["gold"]["caller_stop"] and it.cursor_payload["caller_stop"]
            )
            / stop_neg,
            4,
        ),
        "outcome_exact_accuracy": round(
            sum(
                1
                for r, it in ns_pairs
                if it.cursor_payload["current_outcome"] == r["gold"]["current_outcome"]
            )
            / len(ns_pairs),
            4,
        ),
        "outcome_macro_precision": out_p,
        "outcome_macro_recall": out_r,
        "side_qid_precision": (
            round(side_tp / (side_tp + side_fp), 4) if (side_tp + side_fp) else 1.0
        ),
        "side_qid_recall": round(side_tp / sum(1 for g, _ in side_preds if g), 4),
        "side_qid_false_writes": side_fp,
        "future_qid_recall": round(
            sum(
                1
                for r, it in zip(rows, interps)
                if r["gold"].get("future_qid")
                and (it.future_qid_match or {}).get("qid") == r["gold"]["future_qid"]
            )
            / fut_pos,
            4,
        ),
        "child_ref_accuracy": round(
            sum(
                1
                for r, it in zip(rows, interps)
                if "child_ref" in r["raw_answers"]
                and it.cursor_payload["child_ref"] == r["gold"]["child_ref"]
            )
            / len(child_rows),
            4,
        ),
        "detail_present_accuracy": round(
            sum(
                1
                for r, it in ns_pairs
                if "detail_present" in r["raw_answers"]
                and r["gold"]["detail_present"] is not None
                and (it.diagnostics["detail_present"] >= th.detail_present_min_yes)
                == r["gold"]["detail_present"]
            )
            / len(det_rows),
            4,
        ),
        "mentions_profile_accuracy": round(
            sum(
                1
                for r, it in ns_pairs
                if (it.diagnostics["mentions_profile"] >= th.mentions_profile_min_yes)
                == r["gold"]["mentions_profile"]
            )
            / len(ns_pairs),
            4,
        ),
        "avg_decision_ms": round(sum(r["decision_ms"] for r in rows) / len(rows), 1),
    }
    return th, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-raw", action="store_true")
    args = parser.parse_args()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.from_raw and RAW_PATH.exists():
        rows = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    else:
        rows = asyncio.run(score_cases(cases))
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    th, metrics = calibrate(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    artifact = {
        "provisional": True,
        "note": (
            f"Calibrated on authored Hindi fixtures (N={len(rows)}). "
            "Replace with real OPD staging transcript sweep before enabling "
            "DECISION_MODE=authority in production."
        ),
        "n_cases": len(rows),
        "class_counts": counts,
        "metrics": metrics,
        "thresholds": asdict(th),
    }
    OUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    decisions.load_thresholds(OUT_PATH, require_calibrated=True)
    print(json.dumps({"metrics": metrics, "thresholds": asdict(th)}, indent=2))


if __name__ == "__main__":
    main()
