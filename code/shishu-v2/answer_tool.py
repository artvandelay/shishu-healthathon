"""Fixed Gemini tools for recording answers and the two caller-driven endings."""

import json

import call_state_contract as contract

_PROFILE_LIST = ", ".join(f"profile.{field}" for field in contract.PROFILE_FIELDS)

RECORD_ANSWERS_DECL = {
    "name": "record_answers",
    "description": (
        "MANDATORY ON EVERY CALLER TURN: Record all facts the caller stated in this turn. "
        "Include the current question PLUS any other topic.<topic_id> or profile.<field> facts she volunteered. "
        "Log initial yes/no answers immediately BEFORE or WHILE asking a follow-up. "
        "States are answered, unknown, or no_response. "
        "Use all for every child, specific for named child IDs, and none when no child applies. "
        "Never submit not_applicable. "
        "Do not copy transcript text. "
        "Send value directly as true, false, a short string, a number, a child list, or null. "
        "Do not wrap value in another object."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "updates": {
                "type": "ARRAY",
                "description": "One object per fact stated in this turn.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "field": {
                            "type": "STRING",
                            "description": "profile.<field> or topic.<topic_id>",
                        },
                        "state": {
                            "type": "STRING",
                            "description": "answered, unknown, or no_response",
                        },
                        "value": {
                            "description": (
                                "The answer itself: true, false, a short string, a number, "
                                "a list of child objects, or null. Do not wrap it."
                            ),
                            "anyOf": [
                                {"type": "BOOLEAN"},
                                {"type": "STRING"},
                                {"type": "NUMBER"},
                                {"type": "ARRAY", "items": {"type": "OBJECT"}},
                                {"type": "NULL"},
                            ],
                        },
                        "scope": {
                            "type": "STRING",
                            "description": "all, specific, or none",
                        },
                        "child_ids": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Child IDs from the server profile. Empty when scope is not specific.",
                        },
                        "details": {"type": "OBJECT"},
                    },
                    "required": ["field", "state", "value", "scope", "child_ids", "details"],
                },
            }
        },
        "required": ["updates"],
    },
}

END_CALL_DECL = {
    "name": "end_call",
    "description": (
        "End the call for an emergency happening now or because the caller asked to stop. "
        "Speak the matching closing before this call."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "reason": {
                "type": "STRING",
                "enum": ["emergency", "caller_stop"],
                "description": "emergency or caller_stop",
            }
        },
        "required": ["reason"],
    },
}


def _format_children_context(children: list) -> str:
    if not children:
        return "none"
    parsed = []
    for c in children:
        if not isinstance(c, dict):
            continue
        cid = c.get("id") or "child_?"
        name = c.get("name")
        age = c.get("age")
        sex = c.get("sex")
        yrs = None
        if isinstance(age, (int, float)):
            yrs = float(age)
        elif isinstance(age, dict) and isinstance(age.get("value"), (int, float)):
            val = float(age["value"])
            unit = str(age.get("unit") or "years").lower()
            yrs = val / 12.0 if "month" in unit or "mahine" in unit else val
        parsed.append({"id": cid, "name": name, "age": age, "sex": sex, "yrs": yrs})

    with_yrs = [p for p in parsed if p["yrs"] is not None]
    with_yrs.sort(key=lambda p: p["yrs"], reverse=True)
    rank_map = {}
    if len(with_yrs) >= 2 and with_yrs[0]["yrs"] != with_yrs[-1]["yrs"]:
        rank_map[with_yrs[0]["id"]] = "older / bada"
        rank_map[with_yrs[-1]["id"]] = "younger / chhota"

    child_bits = []
    for p in parsed:
        cid = p["id"]
        rank = f" ({rank_map[cid]})" if cid in rank_map else ""
        name_str = f", name={p['name']}" if p.get("name") else ""
        sex_str = f", sex={p['sex']}" if p.get("sex") else ""
        age_str = json.dumps(p["age"], ensure_ascii=False) if p.get("age") is not None else "unknown"
        child_bits.append(f"{cid}{rank}: age={age_str}{name_str}{sex_str}")
    desc = "; ".join(child_bits)
    if len(parsed) >= 2:
        desc += ". Rule: 'bade wale' -> older child; 'chhote wale' -> younger child. If caller gives an unknown name or ambiguous 'pehle wale', ask which age child."
    return desc


def build_turn_context(state: dict, question: dict | None) -> str:
    profile = (state or {}).get("profile") or {}
    children = profile.get("children") or []
    current = question or {}
    topic_id = current.get("topic_id")
    followup_id = current.get("followup_id")
    lines = [
        "The server selects routing. Ask ONLY the current question below. Do not jump ahead.",
        f"Current question: topic={topic_id}; follow-up={followup_id}; kind={current.get('question_kind')}.",
        current.get("prompt_note") or "",
        "Children: " + _format_children_context(children),
        f"Permitted profile fields: {_PROFILE_LIST}.",
        "Next tool: you MUST call record_answers on the caller's next reply for the current question AND any other topic.<topic_id> or profile.<field> facts she states.",
        "Call end_call only for emergency or caller_stop. The server accepts either reason.",
    ]
    return "\n".join(line for line in lines if line)
