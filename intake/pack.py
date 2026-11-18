"""Pack loader and startup schema validator for `packs/<pack_id>/`."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from intake.models import SEP_SP, OptionSpec, PackConfig, QuestionKind, QuestionSpec, VariantSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKS_DIR, DEFAULT_PACK_ID = REPO_ROOT / "packs", "shishu-hi"
_CHOICE, _SEQS = (QuestionKind.SINGLE_CHOICE, QuestionKind.MULTI_CHOICE), ("skip_if", "followups", "score_rubric")
_TEMPLATES, _DOCS, _MAPS = ("patient.html.j2", "clinician.md.j2", "pdf.css"), ("pack", "prompts", "decisions", "seed"), ("semantic_flags", "rules", "groups")
_SHAPES = ((QuestionKind.SCORE, "score_rubric", "score needs >=2 rubric levels"), (QuestionKind.PROFILE, "requires_slot", "profile step needs requires_slot"))
_THRESHOLDS = frozenset("""danger_now_min_yes caller_stop_min_yes
    mentions_profile_min_yes redact_patient_note_min_yes detail_present_min_yes jump_topic_min_yes
    active_topic_min_yes residual_clinical_min_yes asks_clarify_min_yes
    outcome_min_confidence outcome_min_margin choice_min_confidence choice_min_margin multi_choice_min_yes
    score_min_confidence stated_min_confidence stated_min_margin
    clarify_target_min_confidence clarify_target_min_margin
    item_ref_min_confidence item_ref_min_margin""".split())
def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    if data is not None and not isinstance(data, dict): raise ValueError(f"{path} must decode to a YAML mapping")
    return data or {}
def _parse_question(raw: dict[str, Any], prefixes: list[str]) -> QuestionSpec:
    say = str(raw["say"]).strip()
    bare = say.rstrip("?।!.")
    stem = str(raw.get("stem") or "").strip() or next((bare[len(p):].strip() for p in prefixes if bare.startswith(p)), bare)
    opts = {str(k): OptionSpec.model_validate(v if isinstance(v, dict) else {"label": str(v)}) for k, v in (raw.get("options") or {}).items()}
    shaped = {"say": say, "stem": stem, "options": opts, "variants": tuple(VariantSpec.model_validate(v) for v in raw.get("variants") or ())}
    return QuestionSpec.model_validate(raw | shaped | {k: tuple(raw.get(k) or ()) for k in _SEQS})
def _validate_pack(pack: PackConfig) -> PackConfig:
    ids = [q.id for q in pack.questions]
    errs = ["form.yaml has no questions"] * (not ids) + ["duplicate question ids"] * (len(set(ids)) < len(ids))
    for q in pack.questions:
        refs = (("group", pack.groups, (q.group,)), ("rule", pack.rules, q.skip_if), ("variant rule", pack.rules, [v.when for v in q.variants]), ("followup", pack.followups, q.followups))
        bad = ((q.kind in _CHOICE and len(q.options) < 2, f"{q.kind} needs >=2 options"), *((q.kind is k and not getattr(q, f), msg) for k, f, msg in _SHAPES))
        errs += [f"{q.id}: unknown {n} {i!r}" for n, known, items in refs for i in items if i not in known] + [f"{q.id}: {msg}" for flag, msg in bad if flag]
    missing = sorted(_THRESHOLDS - set(pack.decisions.get("thresholds") or {}))
    errs += [f"missing thresholds: {missing}"] * bool(missing) + [f"missing template {t}" for t in _TEMPLATES if not (pack.root / "templates" / t).is_file()]
    if errs: raise ValueError(f"Invalid pack {pack.id!r}: " + SEP_SP.join(errs))
    return pack
@lru_cache(maxsize=16)
def load_pack(pack_id: str | None = None, packs_dir: Path | None = None) -> PackConfig:
    chosen, base = (pack_id or os.environ.get("INTAKE_PACK") or DEFAULT_PACK_ID).strip(), (packs_dir or PACKS_DIR).resolve()
    root = (base / chosen).resolve()
    if root.parent != base or not root.is_dir(): raise LookupError(f"Pack {chosen!r} not found under {base}")
    form, docs = _read_yaml(root / "form.yaml"), {n: _read_yaml(root / f"{n}.yaml") for n in _DOCS}
    specs = tuple(_parse_question(q, form.get("stem_prefixes") or ()) for q in form.get("questions") or ())
    return _validate_pack(PackConfig(
        id=chosen, root=root, questions=specs, by_id={q.id: q for q in specs}, meta=docs["pack"], prompts=docs["prompts"],
        decisions=docs["decisions"], roster_cfg=form.get("roster"), report_cfg=dict(form.get("report") or ()),
        seed_scenarios=tuple(docs["seed"].get("scenarios") or ()), implications=tuple(form.get("implications") or ()),
        extractor_prompt=form.get("extractor_prompt") or "", followups={str(a): str(b) for a, b in (form.get("followups") or {}).items()},
        profile_slots={str(s["id"]): dict(s) for s in form.get("profile_slots") or ()},
        **{k: {str(a): dict(b) for a, b in (form.get(k) or {}).items()} for k in _MAPS}))
def list_pack_ids(packs_dir: Path | None = None) -> list[str]:
    return sorted(p.parent.name for p in (packs_dir or PACKS_DIR).resolve().glob("*/pack.yaml"))
