"""Saves her answer when the live model spoke without calling record_answers.

A small model reads the server's current question and her reply, and returns
typed fields for that one question. `to_updates` turns them into the same
updates record_answers takes, so call_state applies them the same way.
"""

import json
import os

from google import genai
from google.genai import types

import checklist

MODEL = os.environ.get("SHISHU_BACKSTOP_MODEL", "gemini-3.8-flash")
TIMEOUT_SEC = 6.0

_KINDS = ["answer", "unknown", "clarification", "not_an_answer"]

_PROMPT = """
You check one turn of a Hindi screening call between an assistant and a mother.
The app is waiting for her answer to one question. Decide whether her reply answers that question.

kind:
- answer: her reply answers this question. Fill the fields from what she said.
- unknown: she says she does not know or does not remember.
- clarification: she asks what the question means, or asks to repeat it.
- not_an_answer: anything else, including an answer to a different question.

For `value` true/false: true means she said yes to the question exactly as worded.
Leave a field null when she did not say it. Never guess.
Return JSON only.
""".strip()


def _s(kind: str, **kwargs) -> types.Schema:
    return types.Schema(type=kind, **kwargs)


def _children_ids(state: dict) -> list[str]:
    children = (state.get("profile") or {}).get("children") or []
    return [str(child["id"]) for child in children if isinstance(child, dict) and child.get("id")]


def _field(question: dict) -> str:
    topic_id = question.get("topic_id") or ""
    if question.get("followup_id"):
        return "followup"
    if topic_id.startswith("profile."):
        return "profile"
    return (checklist._BY_ID.get(topic_id) or {}).get("answer") or "yes_no"


def response_schema(question: dict, state: dict) -> types.Schema:
    props: dict[str, types.Schema] = {"kind": _s("STRING", enum=_KINDS)}
    field = _field(question)
    if field == "followup":
        props["detail"] = _s("STRING", nullable=True)
        ids = _children_ids(state)
        if question.get("followup_id") == "affected_child" and ids:
            props["child_ids"] = _s("ARRAY", items=_s("STRING", enum=ids))
    elif field == "name":
        props["value"] = _s("STRING", nullable=True)
    elif field == "marital":
        props["value"] = _s("STRING", enum=["married", "about_to_marry", "unmarried"], nullable=True)
    elif field == "children":
        props["count"] = _s("INTEGER", nullable=True)
        props["children"] = _s(
            "ARRAY",
            items=_s(
                "OBJECT",
                properties={
                    "age_value": _s("NUMBER", nullable=True),
                    "age_unit": _s("STRING", enum=["years", "months"], nullable=True),
                    "sex": _s("STRING", enum=["male", "female", "unknown"]),
                },
                required=["sex"],
            ),
        )
    elif field == "birth_mode":
        props["delivery_type"] = _s("STRING", enum=["normal", "operation"], nullable=True)
        props["place"] = _s("STRING", enum=["home", "hospital"], nullable=True)
        props["operation_reason"] = _s("STRING", nullable=True)
    else:
        props["value"] = _s("BOOLEAN", nullable=True)
    return _s("OBJECT", properties=props, required=["kind"])


def _question_text(question: dict) -> str:
    topic_id = question.get("topic_id") or ""
    if question.get("followup_id") or topic_id not in checklist._BY_ID:
        return str(question.get("prompt_note") or "")
    topic = checklist._BY_ID[topic_id]
    return f"{topic['question']} ({topic['need']})"


def user_content(question: dict, bot_text: str, user_text: str) -> str:
    return (
        f"Question the app is waiting on: {_question_text(question)}\n"
        f"Assistant's last line before her reply: {bot_text or '(none)'}\n"
        f"Her reply: {user_text}"
    )


async def extract(question: dict, state: dict, bot_text: str, user_text: str) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1beta"})
    try:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=user_content(question, bot_text, user_text),
            config=types.GenerateContentConfig(
                system_instruction=_PROMPT,
                response_mime_type="application/json",
                response_schema=response_schema(question, state),
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        result = json.loads(response.text or "{}")
        return result if isinstance(result, dict) else {}
    finally:
        await client.aio.aclose()
        client.close()


def _update(field: str, status: str, value=None, details: dict | None = None, child_ids: list | None = None) -> dict:
    return {
        "field": field,
        "state": status,
        "value": value,
        "scope": "specific" if child_ids else "none",
        "child_ids": list(child_ids or []),
        "details": details or {},
    }


def _children(result: dict) -> list | None:
    items = [item for item in result.get("children") or [] if isinstance(item, dict)]
    if not items:
        return [] if result.get("count") == 0 else None
    children = []
    for item in items:
        child: dict = {}
        if isinstance(item.get("age_value"), (int, float)):
            child["age"] = {"value": item["age_value"], "unit": item.get("age_unit") or "years"}
        if item.get("sex") in ("male", "female"):
            child["sex"] = item["sex"]
        children.append(child)
    return children


def to_updates(question: dict, state: dict, result: dict) -> list[dict]:
    """record_answers updates for the backstop result, or [] when nothing should be saved."""
    kind = result.get("kind")
    if kind not in ("answer", "unknown"):
        return []
    topic_id = str(question.get("topic_id") or "")
    followup_id = question.get("followup_id")
    field = _field(question)
    profile_field = {"name": "caller_name", "marital": "marital_status", "children": "children"}.get(field)
    if topic_id.startswith("profile."):
        target = topic_id
    elif profile_field:
        target = f"profile.{profile_field}"
    else:
        target = f"topic.{topic_id}"
    if kind == "unknown":
        if profile_field:
            return []
        saved = (state.get("topics") or {}).get(topic_id) or {}
        return [_update(target, "unknown", saved.get("value") if followup_id else None)]

    if field == "followup":
        saved = (state.get("topics") or {}).get(topic_id) or {}
        ids = [cid for cid in result.get("child_ids") or [] if cid in _children_ids(state)]
        detail = str(result.get("detail") or "").strip() or ", ".join(ids)
        if not detail:
            return []
        return [_update(target, "answered", saved.get("value"), {followup_id: detail}, ids)]
    if field == "children":
        children = _children(result)
        return [] if children is None else [_update(target, "answered", children)]
    if field == "birth_mode":
        details = {
            key: result[key]
            for key in ("delivery_type", "place", "operation_reason")
            if isinstance(result.get(key), str) and result[key].strip()
        }
        if not details:
            return []
        return [_update(target, "answered", details.get("delivery_type"), details)]
    value = result.get("value")
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    return [_update(target, "answered", value)]
