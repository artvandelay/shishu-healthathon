"""The two models SHISHU calls.

The live call uses LIVE_MODEL. Every other Gemini call (the answer backstop,
the post-call summary, the old-transcript reader) uses TEXT_MODEL. Override
either with SHISHU_LIVE_MODEL or SHISHU_TEXT_MODEL.
"""

import os

from google.genai import types

LIVE_MODEL = os.environ.get("SHISHU_LIVE_MODEL", "gemini-3.1-flash-live-preview")
TEXT_MODEL = os.environ.get("SHISHU_TEXT_MODEL", "gemini-3.8-flash")

_client = None


def text_client():
    """One v1beta client for every TEXT_MODEL call."""
    global _client
    if _client is None:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is unset")
        _client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    return _client


def text_config(
    *,
    system: str = "",
    schema: types.Schema | None = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system or None,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
