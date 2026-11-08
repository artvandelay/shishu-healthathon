"""The two models SHISHU calls.

The live call uses LIVE_MODEL. Every other Gemini call (the answer backstop,
the post-call judge, the old-transcript reader) uses TEXT_MODEL. Override
either with SHISHU_LIVE_MODEL or SHISHU_TEXT_MODEL.
"""

import os

from google.genai import types

LIVE_MODEL = os.environ.get("SHISHU_LIVE_MODEL", "gemini-3.1-flash-live-preview")
TEXT_MODEL = os.environ.get("SHISHU_TEXT_MODEL", "gemini-3.8-flash")


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
