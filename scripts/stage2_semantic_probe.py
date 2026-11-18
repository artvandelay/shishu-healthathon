#!/usr/bin/env python3
"""Manager-run Stage 2 semantic probe — offline request boundary + optional live Decision calls.

Replays seven canonical Hindi stress utterances through real ``CallState`` cursors and the
``decisions.build_turn_request`` / ``interpret_turn_decision`` / ``engine.reduce_turn`` boundary.
Does not edit packs or intake logic; observation only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from intake import decisions, engine, policy  # noqa: E402
from intake.models import CallState, PackConfig, Phase, ProfilePatch, RosterInput, TurnCommand, TurnOutcome  # noqa: E402
from intake.turns import Disposition, EvidenceSource, TurnEvidence  # noqa: E402
from intake.pack import load_pack  # noqa: E402

from scripts.calibrate_decisions import build_state  # noqa: E402

SHISHU_CHILD_PROFILE: dict[str, Any] = {
    "caller_name": "प्रिया",
    "marital_status": "married",
    "children": [{"id": "child_1", "age_years": 2, "sex": "male"}],
}


def _shishu_setup(
    active_qid: str,
    *,
    prior: Sequence[dict[str, Any]] = (),
    pending: int | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "profile": dict(profile or SHISHU_CHILD_PROFILE),
        "active_qid": active_qid,
        "prior_answers": list(prior),
        "pending_followup_idx": pending,
        "utterance_text": "",
    }


PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "bcg_fever_aside",
        "pack_id": "shishu-hi",
        "utterance": "बुखार आया था",
        "setup": _shishu_setup("birth_bcg"),
        "watch_qids": ("birth_bcg",),
    },
    {
        "id": "bcg_clear_plus_fever",
        "pack_id": "shishu-hi",
        "utterance": "नहीं, कोई फोड़ा या गाँठ नहीं हुई थी, बस हल्का बुखार आया था",
        "setup": _shishu_setup("birth_bcg"),
        "watch_qids": ("birth_bcg",),
    },
    {
        "id": "fever_fourth_option",
        "pack_id": "fever-opd-hi",
        "utterance": "रोज़ शाम को एक बार आता है",
        "setup": None,
        "watch_qids": ("fever_frequency", "fever_nature"),
    },
    {
        "id": "other_form_cancer",
        "pack_id": "shishu-hi",
        "utterance": "हाँ, बच्चे की बुआ को कैंसर हुआ था",
        "setup": _shishu_setup("birth_bcg"),
        "watch_qids": ("birth_bcg", "fam_cancer"),
    },
    {
        "id": "tv_unrelated_noise",
        "pack_id": "shishu-hi",
        "utterance": "गैस बंद कर दो, दूध उबल रहा है",
        "setup": _shishu_setup("birth_bcg"),
        "watch_qids": ("birth_bcg",),
    },
    {
        "id": "clarify_scatter_term",
        "pack_id": "shishu-hi",
        "utterance": "थैलेसीमिया का मतलब क्या होता है?",
        "setup": _shishu_setup("birth_bcg"),
        "watch_qids": ("birth_bcg", "preg_thalassemia"),
    },
    {
        "id": "retract_prior_concern",
        "pack_id": "shishu-hi",
        "utterance": "नहीं नहीं, गलती से हाँ बोल दिया, कोई गाँठ नहीं हुई थी",
        "setup": _shishu_setup(
            "birth_bcg",
            pending=-1,
            profile={
                **SHISHU_CHILD_PROFILE,
                "children": [
                    {"id": "child_1", "age_years": 2, "sex": "male"},
                    {"id": "child_2", "age_years": 4, "sex": "female"},
                ],
            },
        ),
        "watch_qids": ("birth_bcg",),
    },
)


def _num(v: Any) -> float | None:
    return float(v) if type(v) in (int, float) and math.isfinite(v) else None


def _to_jsonable(val: Any) -> Any:
    """Recursively coerce probe rows for ``json.dumps`` (Pydantic models, tuples, enums)."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if hasattr(val, "model_dump"):
        return val.model_dump(mode="json")
    if isinstance(val, Mapping):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]
    if hasattr(val, "value"):
        return val.value
    return str(val)


def _top_probabilities(raw: Mapping[str, Any], key: str, *, limit: int = 3) -> list[tuple[str, float]]:
    head = raw.get(key)
    probs = (head.get("probabilities") if isinstance(head, Mapping) else None) or {}
    ranked = sorted(
        ((str(k), p) for k, v in probs.items() if (p := _num(v)) is not None),
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:limit]


def _noul(raw: Mapping[str, Any], key: str) -> float | None:
    head = raw.get(key)
    if not isinstance(head, Mapping):
        return None
    return _num(head.get("noul"))


def _build_fever_frequency_state(pack: PackConfig) -> CallState:
    """Land on ``fever_frequency`` with ``fever_nature=intermittent`` already recorded."""
    st = engine.new_state(pack)
    st = engine.reduce_turn(
        pack,
        st,
        TurnCommand(profile_patch=ProfilePatch(slots={"patient_name": "सुनीता", "patient_age": 34})),
    )
    if pack.by_id.get(st.cursor.qid) and pack.by_id[st.cursor.qid].kind.value == "bridge":
        st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    st = engine.reduce_turn(
        pack,
        st,
        TurnCommand(outcome=TurnOutcome.ANSWERED, value="lt_3_days", utterance="तीन दिन से कम"),
    )
    st = engine.reduce_turn(
        pack,
        st,
        TurnCommand(outcome=TurnOutcome.ANSWERED, value="intermittent", utterance="बीच-बीच में आता है"),
    )
    if st.cursor.qid != "fever_frequency" or st.phase is not Phase.ASKING:
        raise AssertionError(f"fever probe did not land on fever_frequency (at {st.cursor.qid}/{st.phase})")
    return st


def build_probe_state(pack: PackConfig, probe: dict[str, Any]) -> CallState:
    if probe["setup"] is not None:
        return build_state(pack, probe["setup"])
    return _build_fever_frequency_state(pack)


def _cursor_row(state: CallState) -> dict[str, Any]:
    c = state.cursor
    return {
        "phase": state.phase.value,
        "qid": c.qid,
        "substep": c.substep.value,
        "followup_idx": c.followup_idx,
        "step_kind": None,
    }


def _answer_snapshot(pack: PackConfig, state: CallState, qids: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid in qids:
        rec = state.answers.get(qid)
        if rec is None:
            out[qid] = None
            continue
        out[qid] = {
            "state": rec.state.value,
            "value": rec.value,
            "verbatim": (rec.verbatim_text or "")[:120],
            "choice_label": rec.choice_label or "",
        }
    return out


def _request_diag(pack: PackConfig, state: CallState, utterance: str) -> dict[str, Any]:
    request = decisions.build_turn_request(pack, state, utterance_text=utterance)
    qs = request.get("questions") or {}
    choice_keys: list[str] = []
    if isinstance(qs.get("choice_value"), Mapping):
        crit = qs["choice_value"].get("criteria") or {}
        choice_keys = list(crit.keys())
    stated = sorted(k for k in qs if str(k).startswith("stated_"))
    flags = sorted(k for k in qs if str(k).startswith("flag_"))
    row = _cursor_row(state)
    row["step_kind"] = engine.step_kind(pack, state)
    return {
        "model": request.get("model"),
        "question_count": len(qs),
        "heads": sorted(qs.keys()),
        "stated_head_count": len(stated),
        "semantic_flag_head_count": len(flags),
        "active_choice_criteria": choice_keys,
        "active_prompt": (engine.active_prompt(pack, state) or "")[:100],
        "followup_prompt": (engine.followup_prompt(pack, state) or "")[:80],
        "cursor": row,
        "request_bytes_est": len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
    }


def _command_row(cmd: TurnCommand) -> dict[str, Any]:
    return {
        "outcome": cmd.outcome.value,
        "value": cmd.value,
        "item_ref": cmd.item_ref,
        "followup_answered": cmd.followup_answered,
        "stated": {q: o.value for q, o in cmd.stated.items()},
        "semantic_flags": list(cmd.semantic_flags),
        "jump_topic": cmd.jump_topic,
        "redacted": cmd.redacted,
    }


def _interpret_extra(intent: Any) -> dict[str, Any]:
    """Surface Stage 2 fields if ``decisions`` grows them later."""
    extras: dict[str, Any] = {}
    for name in ("interpretation", "semantic", "stage2", "intent_label", "routing"):
        if hasattr(decisions, name):
            extras[f"decisions.{name}"] = getattr(decisions, name)
    for name in ("interpretation", "semantic", "stage2", "intent_label"):
        if hasattr(intent, name):
            val = getattr(intent, name)
            if val is not None and val != {} and val != ():
                extras[f"intent.{name}"] = _to_jsonable(val)
    return extras


def _reduce_diff(
    pack: PackConfig,
    before: CallState,
    after: CallState,
    watch_qids: Sequence[str],
) -> dict[str, Any]:
    return {
        "cursor_before": _cursor_row(before),
        "cursor_after": _cursor_row(after),
        "answers_before": _answer_snapshot(pack, before, watch_qids),
        "answers_after": _answer_snapshot(pack, after, watch_qids),
        "semantic_flags_after": list(after.semantic_flags),
    }


def probe_offline(pack: PackConfig, probe: dict[str, Any]) -> dict[str, Any]:
    state = build_probe_state(pack, probe)
    utterance = str(probe["utterance"])
    return {
        "id": probe["id"],
        "pack_id": pack.id,
        "utterance": utterance,
        "mode": "offline",
        "request": _request_diag(pack, state, utterance),
        "answers_at_cursor": _answer_snapshot(pack, state, probe["watch_qids"]),
    }


async def probe_live(pack: PackConfig, probe: dict[str, Any]) -> dict[str, Any]:
    state = build_probe_state(pack, probe)
    utterance = str(probe["utterance"])
    before = state.model_copy(deep=True)
    request = decisions.build_turn_request(pack, state, utterance_text=utterance)
    result = await decisions.evaluate_turn(request, max_attempts=2, attempt_timeout_sec=12.0, total_budget_sec=18.0)
    row: dict[str, Any] = {
        "id": probe["id"],
        "pack_id": pack.id,
        "utterance": utterance,
        "mode": "live-decisions",
        "decision_ms": result.decision_ms,
        "attempts": result.attempts,
        "ok": result.ok,
        "error_reason": result.error_reason,
        "question_count": result.question_count,
        "total_input_tokens": result.total_input_tokens,
        "request": _request_diag(pack, before, utterance),
    }
    if not result.ok:
        return row
    raw = result.raw_answers
    heads_summary = {
        "active_outcome_top": _top_probabilities(raw, "active_outcome"),
        "choice_value_top": _top_probabilities(raw, "choice_value"),
        "noul_hot": {
            k: _noul(raw, k)
            for k in (
                "danger_now",
                "caller_stop",
                "detail_present",
                "jump_topic",
                "mentions_profile",
                "redact_patient_note",
                "active_topic",
                "residual_clinical",
                "asks_clarify",
            )
            if _noul(raw, k) is not None
        },
    }
    for key in sorted(raw):
        if str(key).startswith("stated_"):
            top = _top_probabilities(raw, key, limit=2)
            if top:
                heads_summary.setdefault("stated_tops", {})[key] = top
    intent = decisions.interpret_turn_decision(pack, before, raw, utterance_text=utterance)
    interp = intent.interpretation
    if interp is not None:
        evidence = TurnEvidence(
            source=EvidenceSource.TEXT,
            snapshot_step=before.step_token,
            transcript=utterance,
            received_at=0.0,
        )
        after, pol = policy.apply_policy(pack, before, evidence, interp)
        row |= {
            "heads_summary": heads_summary,
            "diagnostics": intent.diagnostics,
            "command": _command_row(intent.command),
            "interpretation": {
                "act": interp.act.value,
                "relation": interp.active_relation.value,
                "domain": interp.active_domain.value,
                "clarify_target_qid": interp.clarify_target_qid,
            },
            "dispositions": [d.disposition.value for d in pol.decisions],
            "interpretation_extras": _interpret_extra(intent),
            "state_diff": _reduce_diff(pack, before, after, probe["watch_qids"]),
        }
    else:
        after = engine.reduce_turn(pack, before, intent.command)
        row |= {
            "heads_summary": heads_summary,
            "diagnostics": intent.diagnostics,
            "command": _command_row(intent.command),
            "interpretation_extras": _interpret_extra(intent),
            "state_diff": _reduce_diff(pack, before, after, probe["watch_qids"]),
        }
    if after.cursor.qid != before.cursor.qid or after.phase != before.phase:
        row["state_diff"]["cursor_after"]["step_kind"] = engine.step_kind(pack, after)
        row["state_diff"]["cursor_before"]["step_kind"] = engine.step_kind(pack, before)
    return row


def _print_table(rows: Sequence[dict[str, Any]]) -> None:
    cols = ("id", "pack", "ms", "act", "relation", "domain", "disp", "cursorΔ")
    widths = {c: len(c) for c in cols}

    def cell(row: dict[str, Any], name: str) -> str:
        if name == "id":
            return str(row.get("id", ""))[:28]
        if name == "pack":
            return str(row.get("pack_id", ""))[:12]
        if name == "ms":
            return "" if row.get("decision_ms") is None else str(row["decision_ms"])
        interp = row.get("interpretation") or {}
        if name == "act":
            return str(interp.get("act") or "")
        if name == "relation":
            return str(interp.get("relation") or "")[:10]
        if name == "domain":
            return str(interp.get("domain") or "")[:10]
        if name == "disp":
            disps = row.get("dispositions") or []
            return ",".join(d[:4] for d in disps[:2]) if disps else ""
        if name == "cursorΔ":
            diff = row.get("state_diff") or {}
            b, a = diff.get("cursor_before") or {}, diff.get("cursor_after") or {}
            if not b and not a:
                c = (row.get("request") or {}).get("cursor") or {}
                return f"{c.get('qid','')}:{c.get('substep','')}"
            if (b.get("qid"), b.get("substep")) == (a.get("qid"), a.get("substep")):
                return "="
            return f"{b.get('qid')}→{a.get('qid')}"
        return ""

    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(cell(row, c)))

    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for row in rows:
        print("  ".join(cell(row, c).ljust(widths[c]) for c in cols))


async def _run_live_all(probes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    packs: dict[str, PackConfig] = {}
    for probe in probes:
        pack = packs.setdefault(probe["pack_id"], load_pack(probe["pack_id"]))
        print(f"[live] {probe['id']} …", flush=True)
        out.append(await probe_live(pack, probe))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="offline",
        choices=("offline", "live-decisions"),
        help="offline = zero-network request boundary; live-decisions = real gemini-decision-1",
    )
    parser.add_argument("--json", action="store_true", help="emit full JSON array after the summary table")
    parser.add_argument("--probe", action="append", help="run only these probe ids (repeatable)")
    args = parser.parse_args(argv)

    selected = [p for p in PROBES if not args.probe or p["id"] in args.probe]
    if args.probe and len(selected) != len(set(args.probe)):
        missing = set(args.probe) - {p["id"] for p in selected}
        print(f"unknown probe id(s): {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    packs: dict[str, PackConfig] = {}
    rows: list[dict[str, Any]] = []
    if args.mode == "offline":
        for probe in selected:
            pack = packs.setdefault(probe["pack_id"], load_pack(probe["pack_id"]))
            rows.append(probe_offline(pack, probe))
    else:
        rows = asyncio.run(_run_live_all(selected))

    _print_table(rows)
    if args.json:
        print()
        print(json.dumps(_to_jsonable(rows), ensure_ascii=False, indent=2))
    failed = [r["id"] for r in rows if r.get("mode") == "live-decisions" and not r.get("ok", True)]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())