"""Narrowly scoped generative entity extractor for caller demographic profiles."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import engine
from models import CallState, ChildInput, ChildSex, MaritalStatus, ProfilePatch

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ATTEMPT_TIMEOUT_SEC = 3.0

_TOP_KEYS = frozenset({"caller_name", "marital_status", "child_count", "children"})
_CHILD_KEYS = frozenset({"age_years", "sex", "name"})
_SEX_VALUES = ["male", "female", "unknown"]
_MARITAL_VALUES = ["married", "about_to_marry", "unmarried"]

_SYSTEM_INSTRUCTION = (
    "Extract ONLY demographic profile entities stated or directly answered by the caller's "
    "utterance given active_qid, active_prompt, and optional current_profile. Return strictly "
    "matching JSON with only the four schema fields, and leave every unmentioned field null.\n"
    "- caller_name: her given/family name only when actually stated; null for a lead-in "
    "fragment with no name. Never copy a diagnosis or disease name into caller_name.\n"
    "- marital_status: 'married', 'about_to_marry', or 'unmarried' when stated or directly "
    "answered (on intro_married a bare negative means 'unmarried' and a bare affirmative means "
    "'married'); otherwise null.\n"
    "- child_count and children: when she mentions, adds, completes, or updates children, set "
    "child_count to the total (integer >= 0) and children to one item per child whose age was "
    "stated (age_years as float years >= 0, months as fractional years such as 0.5 for 6 "
    "months; sex 'male', 'female', or 'unknown'; optional personal name or null, never a "
    "disease name). When current_profile is present and she adds another child, supplies "
    "remaining ages in an ages follow-up, or corrects a child's details, output the COMPLETE "
    "updated child_count and children, preserving known children and pending_child_count "
    "unless she explicitly replaces them. On intro_children a bare negative or zero count means "
    "child_count=0 and children=[]. If she states a count without all ages, keep the count and "
    "list only the children whose ages were given. Set both fields null when children are not "
    "mentioned or updated."
)

_CHILD_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "age_years": types.Schema(type=types.Type.NUMBER),
        "sex": types.Schema(type=types.Type.STRING, enum=_SEX_VALUES),
        "name": types.Schema(type=types.Type.STRING, nullable=True),
    },
    required=["age_years", "sex"],
)
_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "caller_name": types.Schema(type=types.Type.STRING, nullable=True),
        "marital_status": types.Schema(
            type=types.Type.STRING, enum=_MARITAL_VALUES, nullable=True
        ),
        "child_count": types.Schema(type=types.Type.INTEGER, nullable=True),
        "children": types.Schema(type=types.Type.ARRAY, items=_CHILD_SCHEMA, nullable=True),
    },
    required=sorted(_TOP_KEYS),
)
_GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=_SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=_RESPONSE_SCHEMA,
    temperature=0.0,
    thinking_config=None,
)

_clients: dict[str, genai.Client] = {}


@dataclass(frozen=True, slots=True)
class ProfileExtractionResult:
    """One extractor attempt: the typed patch it proposed, or why it failed."""

    ok: bool
    patch: ProfilePatch = field(default_factory=ProfilePatch)
    error_reason: str = ""
    extractor_ms: int = 0


def resolve_model(model: str | None = None) -> str:
    chosen = model if model is not None else os.environ.get("PROFILE_EXTRACTOR_MODEL")
    return str(chosen or "").strip() or DEFAULT_MODEL


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))


def _failed(reason: str, started: float) -> ProfileExtractionResult:
    return ProfileExtractionResult(
        ok=False, error_reason=reason, extractor_ms=_elapsed_ms(started)
    )


def _clean_name(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    return " ".join(value.split()) or None


def _child(item: Any, index: int) -> ChildInput:
    """One child object, with every field pinned to the schema's contract."""
    if not isinstance(item, dict):
        raise ValueError(f"children[{index}] must be an object")
    if extra := set(item) - _CHILD_KEYS:
        raise ValueError(f"unexpected keys in children[{index}]: {', '.join(sorted(extra))}")
    age = item.get("age_years")
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        raise ValueError(f"children[{index}].age_years must be numeric")
    if not math.isfinite(age) or age < 0:
        raise ValueError(f"children[{index}].age_years must be finite and >= 0")
    try:
        sex = ChildSex(str(item.get("sex", "unknown")))
    except ValueError:
        raise ValueError(f"invalid children[{index}].sex: {item.get('sex')!r}") from None
    return ChildInput(age_years=float(age), sex=sex, name=_clean_name(item.get("name"), "name"))


def validate_profile_payload(raw: Any) -> ProfilePatch:
    """Strictly validate raw extractor output into one typed profile patch."""
    if not isinstance(raw, dict):
        raise ValueError("profile payload must be a JSON object")
    if extra := set(raw) - _TOP_KEYS:
        raise ValueError(f"unexpected profile keys: {', '.join(sorted(extra))}")
    count, raw_kids = raw.get("child_count"), raw.get("children")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise ValueError(f"invalid child_count: {count!r}")
    if raw_kids is not None and not isinstance(raw_kids, list):
        raise ValueError("children must be a list or null")
    kids = None if raw_kids is None else tuple(_child(k, i) for i, k in enumerate(raw_kids))
    marital = raw.get("marital_status")
    if marital is not None and marital not in _MARITAL_VALUES:
        raise ValueError(f"invalid marital_status: {marital!r}")

    if count is None and kids == ():
        count = 0
    if count == 0 and kids:
        raise ValueError("child_count is 0 with non-empty children")
    if count is not None and kids is not None and len(kids) > count:
        raise ValueError("children length exceeds child_count")
    return ProfilePatch(
        caller_name=_clean_name(raw.get("caller_name"), "caller_name"),
        marital_status=MaritalStatus(marital) if marital else None,
        child_count=count,
        children=() if count == 0 else kids,
    )


def _known_children(kids: list[Any]) -> list[dict[str, Any]]:
    return [{"age_years": k.age_years, "sex": k.sex.value, "name": k.name} for k in kids]


def _known_profile(state: CallState | None) -> dict[str, Any]:
    """Only the demographics already known, so the extractor can preserve them."""
    if state is None:
        return {}
    profile = state.profile
    known: dict[str, Any] = {}
    if profile.caller_name:
        known["caller_name"] = profile.caller_name
    if profile.marital_status is not None:
        known["marital_status"] = profile.marital_status.value
    if profile.children:
        known["children"] = _known_children(profile.children)
    if profile.pending_children:
        known["pending_children"] = _known_children(profile.pending_children)
    if profile.pending_child_count:
        known["pending_child_count"] = profile.pending_child_count
    return known


def _context(state: CallState | None, qid: str, prompt: str, utterance: str) -> str:
    """The whole extractor prompt: active step, known demographics, and her words."""
    lines = [f"active_qid: {qid}", f"active_prompt: {prompt}"]
    if known := _known_profile(state):
        lines.append(f"current_profile: {json.dumps(known, ensure_ascii=False)}")
    return "\n".join([*lines, f"utterance_text: {utterance}"])


def _decode(raw_output: Any) -> dict[str, Any]:
    """The extractor's JSON object, whether it answered with text or a dict."""
    if isinstance(raw_output, dict):
        return raw_output
    text = raw_output if isinstance(raw_output, str) else getattr(raw_output, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise json.JSONDecodeError("empty JSON output", "", 0)
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("extractor response must decode to a JSON object")
    return parsed


async def _generate(generator: Callable[..., Any], model: str, contents: Any) -> Any:
    """Run one generator call, accepting async and blocking callables alike."""
    kwargs = {"model": model, "contents": contents, "config": _GENERATE_CONFIG}
    if inspect.iscoroutinefunction(generator):
        return await generator(**kwargs)
    output = await asyncio.to_thread(generator, **kwargs)
    return await output if inspect.isawaitable(output) else output


def _dispatch(
    contents: Any, model: str, api_key: str, generator: Callable[..., Any] | None
) -> Any:
    """Route the call to the injected generator or to one cached live client."""
    if generator is not None:
        return _generate(generator, model, contents)
    if api_key not in _clients:
        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key].aio.models.generate_content(
        model=model, contents=contents, config=_GENERATE_CONFIG
    )


async def extract_profile_entities(
    state: CallState | None = None,
    utterance_text: str = "",
    audio_wav_bytes: bytes | None = None,
    *,
    active_qid: str | None = None,
    active_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    attempt_timeout_sec: float = DEFAULT_ATTEMPT_TIMEOUT_SEC,
    generator: Callable[..., Any] | None = None,
) -> ProfileExtractionResult:
    """Extract a schema-validated profile patch from one caller turn."""
    started = time.perf_counter()
    qid = active_qid if active_qid is not None else (state.cursor.qid if state else "")
    prompt = active_prompt
    if prompt is None:
        prompt = engine.active_prompt_hi(state) if state else ""
    clean_utt = " ".join(str(utterance_text or "").split())
    audio = bytes(audio_wav_bytes) if audio_wav_bytes else b""
    if not clean_utt and not audio:
        return _failed("empty_input", started)

    text = _context(state, qid or "", prompt or "", clean_utt or "[audio reply]")
    contents: Any = text
    if audio and not clean_utt:
        parts = [types.Part.from_text(text=text)]
        parts.append(types.Part.from_bytes(data=audio, mime_type="audio/wav"))
        contents = parts

    raw_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    key = str(raw_key or "").strip()
    if generator is None and not key:
        return _failed("missing_api_key", started)

    try:
        pending = _dispatch(contents, resolve_model(model), key, generator)
        output = await asyncio.wait_for(pending, timeout=max(0.01, float(attempt_timeout_sec)))
        patch = validate_profile_payload(_decode(output))
    except (TimeoutError, asyncio.TimeoutError):
        return _failed("timeout", started)
    except json.JSONDecodeError:
        return _failed("invalid_json", started)
    except ValueError:
        return _failed("invalid_schema", started)
    except Exception as exc:  # noqa: BLE001 - provider faults are data, not crashes
        return _failed(type(exc).__name__ or "api_error", started)
    return ProfileExtractionResult(ok=True, patch=patch, extractor_ms=_elapsed_ms(started))
