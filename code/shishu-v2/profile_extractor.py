"""Narrowly scoped generative entity extractor for caller demographic profiles."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

import decisions

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ATTEMPT_TIMEOUT_SEC = 3.0

_ALLOWED_TOP_KEYS = frozenset({
    "caller_name", "marital_status", "child_count", "children", "ever_pregnant",
})
_ALLOWED_CHILD_KEYS = frozenset({"age_years", "sex", "name"})
_VALID_MARITAL_STATUSES = frozenset({"married", "about_to_marry", "unmarried"})
_VALID_CHILD_SEX = frozenset({"male", "female", "unknown"})

_SYSTEM_INSTRUCTION = (
    "Extract ONLY demographic profile entities stated or directly answered by the caller's "
    "utterance given active_qid and active_prompt. Return strictly matching JSON with only the "
    "five schema fields and leave any unmentioned field as null.\n"
    "- caller_name: the person's actual name only if stated; null if she only uttered a "
    "lead-in fragment without an actual name or did not mention a name.\n"
    "- marital_status: 'married', 'about_to_marry', or 'unmarried' when stated or directly "
    "answered (when active_qid is intro_married, a bare negative means 'unmarried' and a bare "
    "affirmative means 'married'); otherwise null.\n"
    "- child_count and children: when she mentions children, set child_count to total count "
    "(integer >= 0) and children to one item per child whose age was stated (age_years as float "
    "years >= 0, converting months to fractional years such as 0.5 for 6 months; sex as "
    "'male', 'female', or 'unknown'; optional name or null). When active_qid is intro_children "
    "and she gives a bare negative or zero count, set child_count=0 and children=[]. If she "
    "states count > 0 without all ages, keep child_count and list only children with stated "
    "ages. Set both child_count and children to null when children are not mentioned.\n"
    "- ever_pregnant: boolean when stated or when active_qid is ever_pregnant and she gives "
    "a direct affirmative (true) or negative (false) answer; otherwise null."
)

_CHILD_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "age_years": types.Schema(type=types.Type.NUMBER),
        "sex": types.Schema(type=types.Type.STRING, enum=["male", "female", "unknown"]),
        "name": types.Schema(type=types.Type.STRING, nullable=True),
    },
    required=["age_years", "sex"],
)

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "caller_name": types.Schema(type=types.Type.STRING, nullable=True),
        "marital_status": types.Schema(
            type=types.Type.STRING,
            enum=["married", "about_to_marry", "unmarried"],
            nullable=True,
        ),
        "child_count": types.Schema(type=types.Type.INTEGER, nullable=True),
        "children": types.Schema(type=types.Type.ARRAY, items=_CHILD_SCHEMA, nullable=True),
        "ever_pregnant": types.Schema(type=types.Type.BOOLEAN, nullable=True),
    },
    required=["caller_name", "marital_status", "child_count", "children", "ever_pregnant"],
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
    ok: bool
    updates: dict[str, Any]
    error_reason: str
    extractor_ms: int


def _get_client(api_key: str) -> genai.Client:
    client = _clients.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _clients[api_key] = client
    return client


def resolve_model(model: str | None = None) -> str:
    chosen = model if model is not None else os.environ.get("PROFILE_EXTRACTOR_MODEL")
    text = str(chosen or "").strip()
    return text or DEFAULT_MODEL


def _clean_optional_text(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    return " ".join(value.strip().split()) or None


def validate_profile_payload(raw: Any) -> dict[str, Any]:
    """Strictly validate and normalize raw extractor output into cursor profile updates."""
    if not isinstance(raw, dict):
        raise ValueError("profile payload must be a JSON object")
    extra_keys = set(raw.keys()) - _ALLOWED_TOP_KEYS
    if extra_keys:
        raise ValueError(f"unexpected profile keys: {', '.join(sorted(extra_keys))}")

    updates: dict[str, Any] = {}
    if "caller_name" in raw:
        clean_name = _clean_optional_text(raw.get("caller_name"), field_name="caller_name")
        if clean_name is not None:
            updates["caller_name"] = clean_name

    if "marital_status" in raw and raw.get("marital_status") is not None:
        marital = raw.get("marital_status")
        if not isinstance(marital, str) or marital not in _VALID_MARITAL_STATUSES:
            raise ValueError(f"invalid marital_status: {marital!r}")
        updates["marital_status"] = marital

    child_count: int | None = None
    if "child_count" in raw and raw.get("child_count") is not None:
        raw_count = raw.get("child_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"invalid child_count: {raw_count!r}")
        child_count = raw_count
        updates["child_count"] = child_count

    validated_children: list[dict[str, Any]] | None = None
    if "children" in raw and raw.get("children") is not None:
        raw_children = raw.get("children")
        if not isinstance(raw_children, list):
            raise ValueError("children must be a list or null")
        validated_children = []
        for idx, item in enumerate(raw_children):
            if not isinstance(item, dict):
                raise ValueError(f"children[{idx}] must be an object")
            extra_child = set(item.keys()) - _ALLOWED_CHILD_KEYS
            if extra_child:
                raise ValueError(f"unexpected keys in children[{idx}]: {', '.join(sorted(extra_child))}")
            raw_age = item.get("age_years")
            if isinstance(raw_age, bool) or not isinstance(raw_age, (int, float)):
                raise ValueError(f"children[{idx}].age_years must be numeric")
            age = float(raw_age)
            if not math.isfinite(age) or age < 0.0:
                raise ValueError(f"children[{idx}].age_years must be finite and >= 0")
            raw_sex = item.get("sex", "unknown") if "sex" in item else "unknown"
            if not isinstance(raw_sex, str) or raw_sex not in _VALID_CHILD_SEX:
                raise ValueError(f"invalid children[{idx}].sex: {raw_sex!r}")
            child_name = _clean_optional_text(item.get("name"), field_name=f"children[{idx}].name")
            validated_children.append({"age_years": age, "sex": raw_sex, "name": child_name})

    if child_count == 0 and validated_children:
        raise ValueError("child_count is 0 with non-empty children")
    if child_count is not None and validated_children is not None and len(validated_children) > child_count:
        raise ValueError("children length exceeds child_count")
    if child_count == 0:
        updates["children"] = []
    elif validated_children:
        updates["children"] = validated_children

    if "ever_pregnant" in raw and raw.get("ever_pregnant") is not None:
        raw_ever = raw.get("ever_pregnant")
        if not isinstance(raw_ever, bool):
            raise ValueError(f"ever_pregnant must be a boolean, got {raw_ever!r}")
        updates["ever_pregnant"] = raw_ever

    return updates


def _decode_raw_response(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        return raw_output
    text = raw_output if isinstance(raw_output, str) else getattr(raw_output, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise json.JSONDecodeError("empty JSON output", "", 0)
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("extractor response must decode to a JSON object")
    return parsed


async def extract_profile_entities(
    state: dict | None = None,
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
    """Extract schema-validated profile entities from one caller turn."""
    started = time.perf_counter()
    if (active_qid is None or active_prompt is None) and isinstance(state, dict):
        view_qid, _step_kind, view_prompt, _followup = decisions.turn_view(state)
        active_qid = view_qid if active_qid is None else active_qid
        active_prompt = view_prompt if active_prompt is None else active_prompt

    clean_utt = " ".join(str(utterance_text or "").split())
    audio_data = bytes(audio_wav_bytes) if audio_wav_bytes else b""
    if not clean_utt and not audio_data:
        return ProfileExtractionResult(ok=False, updates={}, error_reason="empty_input", extractor_ms=0)

    has_audio = bool(not clean_utt and audio_data)
    utt_line = clean_utt or "[spoken reply in attached audio]"
    context_text = f"active_qid: {active_qid or ''}\nactive_prompt: {active_prompt or ''}\nutterance_text: {utt_line}"
    contents: Any = (
        [types.Part.from_text(text=context_text), types.Part.from_bytes(data=audio_data, mime_type="audio/wav")]
        if has_audio
        else context_text
    )
    resolved_model = resolve_model(model)
    timeout = max(0.01, float(attempt_timeout_sec))

    if generator is None:
        resolved_key = str(api_key if api_key is not None else (os.environ.get("GEMINI_API_KEY") or "")).strip()
        if not resolved_key:
            return ProfileExtractionResult(ok=False, updates={}, error_reason="missing_api_key", extractor_ms=0)

    try:
        if generator is not None:
            if inspect.iscoroutinefunction(generator):
                raw_coro = generator(model=resolved_model, contents=contents, config=_GENERATE_CONFIG)
            else:
                async def _run_sync() -> Any:
                    res = await asyncio.to_thread(generator, model=resolved_model, contents=contents, config=_GENERATE_CONFIG)
                    return await res if inspect.isawaitable(res) else res
                raw_coro = _run_sync()
        else:
            client = _get_client(resolved_key)
            raw_coro = client.aio.models.generate_content(model=resolved_model, contents=contents, config=_GENERATE_CONFIG)

        raw_output = await asyncio.wait_for(raw_coro, timeout=timeout)
        updates = validate_profile_payload(_decode_raw_response(raw_output))
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return ProfileExtractionResult(ok=True, updates=updates, error_reason="", extractor_ms=elapsed_ms)
    except (TimeoutError, asyncio.TimeoutError):
        error_reason = "timeout"
    except json.JSONDecodeError:
        error_reason = "invalid_json"
    except ValueError:
        error_reason = "invalid_schema"
    except Exception as exc:
        error_reason = type(exc).__name__ or "api_error"

    elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
    return ProfileExtractionResult(ok=False, updates={}, error_reason=error_reason, extractor_ms=elapsed_ms)
