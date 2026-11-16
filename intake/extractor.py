"""Narrow generative extractor for the open demographic slots a pack declares."""

from __future__ import annotations

import asyncio, inspect, json, math, os, time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from intake import engine
from intake.models import CallState, PackConfig, ProfilePatch, QuestionKind, RosterInput

DEFAULT_MODEL = "gemini-3.5-flash-lite"
_CLIENTS: dict[str, genai.Client] = {}
_ROSTER_FIELDS = ("roster_complete", "roster")
_SLOT_TYPES = {"number": types.Type.NUMBER, "integer": types.Type.INTEGER, "boolean": types.Type.BOOLEAN}
_UNSET = object()

@dataclass(frozen=True, slots=True)
class ProfileExtractionResult:
    ok: bool
    patch: ProfilePatch = field(default_factory=ProfilePatch)
    error_reason: str = ""
    extractor_ms: int = 0
def build_response_schema(pack: PackConfig) -> types.Schema:
    fields = {sid: types.Schema(type=_SLOT_TYPES.get(str(cfg.get("type") or "").lower(), types.Type.STRING), nullable=True)
              for sid, cfg in pack.profile_slots.items()}
    if m := next((q for q in pack.questions if q.saves_to_profile == "marital_status" and q.kind is QuestionKind.SINGLE_CHOICE), None):
        fields["marital_status"] = types.Schema(type=types.Type.STRING, enum=list(m.options), nullable=True)
    if pack.roster_cfg is not None:
        tags = [str(t) for t in pack.roster_cfg.get("tags") or ["unknown"]]
        entry = {"id": types.Schema(type=types.Type.STRING, nullable=True), "age_years": types.Schema(type=types.Type.NUMBER),
                 "tag": types.Schema(type=types.Type.STRING, enum=tags), "name": types.Schema(type=types.Type.STRING, nullable=True)}
        fields["roster_complete"] = types.Schema(type=types.Type.BOOLEAN, nullable=True)
        fields["roster"] = types.Schema(type=types.Type.ARRAY, nullable=True, items=types.Schema(type=types.Type.OBJECT, required=["age_years"], properties=entry))
    return types.Schema(type=types.Type.OBJECT, properties=fields, required=sorted(fields))
def _roster_input(i: int, entry: Any) -> RosterInput:
    age = entry.get("age_years") if isinstance(entry, dict) else None
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0: raise ValueError(f"roster[{i}].age_years invalid")
    name, cid = entry.get("name"), entry.get("id")
    if cid is not None and (not isinstance(cid, str) or not cid.strip()): raise ValueError(f"roster[{i}].id invalid")
    if name is not None and not isinstance(name, str): raise ValueError(f"roster[{i}].name invalid")
    return RosterInput(id=(" ".join(cid.split()) if cid else None), age_years=float(age), tag=str(entry.get("tag") or "unknown"),
                       name=(" ".join(name.split()) or None) if name else None)
def _slot_value(k: str, cfg: dict[str, Any], v: Any) -> Any:
    """A typed slot's value, `_UNSET` when unmentioned, or `ValueError` when it cannot be that type at all."""
    if v is None: return _UNSET
    kind = str(cfg.get("type") or "").lower()
    if kind == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v): raise ValueError(f"invalid {k}")
        return int(v) if float(v).is_integer() else float(v)
    if kind == "integer":
        if isinstance(v, bool) or not isinstance(v, int): raise ValueError(f"invalid {k}")
        return v
    if kind == "boolean":
        if not isinstance(v, bool): raise ValueError(f"invalid {k}")
        return v
    if isinstance(v, bool) or not isinstance(v, (str, int, float)): return _UNSET
    return " ".join(str(v).split()) or _UNSET
def validate_payload(pack: PackConfig, raw: Any) -> ProfilePatch:
    if not isinstance(raw, dict) or (set(raw) - {*pack.profile_slots, "marital_status", *_ROSTER_FIELDS}): raise ValueError("unexpected profile payload")
    values = {k: _slot_value(k, pack.profile_slots.get(k) or {}, raw.get(k)) for k in ({*pack.profile_slots, "marital_status"})}
    slots = {k: v for k, v in values.items() if v is not _UNSET}
    complete, entries = raw.get("roster_complete"), raw.get("roster")
    if complete is not None and not isinstance(complete, bool): raise ValueError("invalid roster_complete")
    if entries is not None and not isinstance(entries, list): raise ValueError("invalid roster")
    return ProfilePatch(slots=slots, roster_complete=complete, roster=tuple(_roster_input(i, e) for i, e in enumerate(entries)) if entries is not None else None)
async def extract_profile_entities(pack: PackConfig, state: CallState | None = None, utterance_text: str = "", audio_wav_bytes: bytes | None = None,
                                   *, model: str | None = None, api_key: str | None = None, attempt_timeout_sec: float = 3.0,
                                   generator: Callable[..., Any] | None = None) -> ProfileExtractionResult:
    started = time.perf_counter()
    fail = lambda r: ProfileExtractionResult(False, error_reason=r, extractor_ms=max(0, int((time.perf_counter() - started) * 1000)))
    spoken, audio = " ".join(str(utterance_text or "").split()), bytes(audio_wav_bytes or b"")
    if not spoken and not audio: return fail("empty_input")
    key = str(api_key if api_key is not None else os.environ.get("GEMINI_API_KEY") or "").strip()
    if generator is None and not key: return fail("missing_api_key")
    known = {k: v for k, v in (state.profile if state else {}).items() if v not in (None, "")}
    roster: list[str] = []
    if state is not None and pack.roster_cfg is not None and (state.roster or state.roster_stated or state.roster_incomplete):
        roster = [f"current_roster: {json.dumps([{'id': c.id, 'age_years': c.age_years, 'tag': c.tag, 'name': c.name} for c in state.roster], ensure_ascii=False)}",
                  f"roster_complete: {json.dumps(not state.roster_incomplete and bool(state.roster or state.roster_stated))}"]
    text = "\n".join([f"active_qid: {state.cursor.qid if state else ''}", f"active_prompt: {engine.active_prompt(pack, state) if state else ''}",
                      *([f"current_profile: {json.dumps(known, ensure_ascii=False)}"] if known else []), *roster,
                      f"utterance_text: {spoken or '[audio reply]'}"])
    contents: Any = [types.Part.from_text(text=text), types.Part.from_bytes(data=audio, mime_type="audio/wav")] if audio else text
    cfg = types.GenerateContentConfig(system_instruction=pack.extractor_prompt, response_mime_type="application/json", response_schema=build_response_schema(pack), temperature=0.0)
    chosen = str(model or os.environ.get("PROFILE_EXTRACTOR_MODEL") or "").strip() or DEFAULT_MODEL
    if generator is None: job = (_CLIENTS.get(key) or _CLIENTS.setdefault(key, genai.Client(api_key=key))).aio.models.generate_content(model=chosen, contents=contents, config=cfg)
    else: job = generator(model=chosen, contents=contents, config=cfg) if inspect.iscoroutinefunction(generator) else asyncio.to_thread(generator, model=chosen, contents=contents, config=cfg)
    try: out = await asyncio.wait_for(job, timeout=max(0.01, float(attempt_timeout_sec)))
    except (TimeoutError, asyncio.TimeoutError): return fail("timeout")
    except Exception as exc: return fail(type(exc).__name__ or "api_error")  # noqa: BLE001
    try:
        obj = out if isinstance(out, dict) else json.loads(str(out if isinstance(out, str) else getattr(out, "text", None) or "").strip() or "null")
        if not isinstance(obj, dict): raise ValueError("not dict")
        return ProfileExtractionResult(True, validate_payload(pack, obj), extractor_ms=max(0, int((time.perf_counter() - started) * 1000)))
    except json.JSONDecodeError: return fail("invalid_json")
    except ValueError: return fail("invalid_schema")
