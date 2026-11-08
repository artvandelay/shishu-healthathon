"""Frozen CallState shape. No routing."""

import checklist

TOPIC_STATES = ("not_asked", "answered", "unknown", "no_response", "not_applicable")
MODEL_TOPIC_STATES = ("answered", "unknown", "no_response")
CALL_PHASES = ("asking", "closing", "ended")
DELIVERY_TYPES = ("normal", "operation")
BIRTH_PLACES = ("home", "hospital")
END_REASONS = ("emergency", "caller_stop", "silence", "time_limit", "browser_hangup")
INTERNAL_CLOSE_REASONS = END_REASONS + ("complete",)
CLOSE_REASON_NOTES = {
    "silence": "the mother was silent for 30 seconds",
    "time_limit": "the 20-minute limit was reached",
    "caller_stop": "the mother asked to stop the call",
}
PROFILE_FIELDS = (
    "caller_name",
    "marital_status",
    "children",
    "ever_pregnant",
    "currently_pregnant",
    "has_delivered",
)

TOPIC_IDS = tuple(topic["id"] for topic in checklist.TOPICS)
REQUIRED_TOPIC_IDS = tuple(checklist.REQUIRED_IDS)

_HISTORY_LIMIT = 300


def blank_topic() -> dict:
    return {
        "state": "not_asked",
        "value": None,
        "details": {},
        "instances": [],
        "source_turn_ids": [],
        "source": None,
        "reason": None,
    }


def new_call_state() -> dict:
    return {
        "version": 1,
        "phase": "asking",
        "profile": {
            "caller_name": None,
            "marital_status": None,
            "ever_pregnant": None,
            "currently_pregnant": None,
            "has_delivered": None,
            "children": [],
        },
        "topics": {topic_id: blank_topic() for topic_id in TOPIC_IDS},
        "current_question": None,
        "close_reason": None,
        "history": [],
    }


def _copy_topic(saved: dict) -> dict:
    topic = blank_topic()
    if saved.get("state") in TOPIC_STATES:
        topic["state"] = saved["state"]
    if "value" in saved:
        topic["value"] = saved["value"]
    if isinstance(saved.get("details"), dict):
        topic["details"] = dict(saved["details"])
    if isinstance(saved.get("instances"), list):
        topic["instances"] = list(saved["instances"])
    if isinstance(saved.get("source_turn_ids"), list):
        topic["source_turn_ids"] = list(saved["source_turn_ids"])
    if "source" in saved:
        topic["source"] = saved["source"]
    if "reason" in saved:
        topic["reason"] = saved["reason"]
    return topic


def normalize_state(raw: object) -> dict:
    state = new_call_state()
    if not isinstance(raw, dict):
        return state
    phase = raw.get("phase")
    if phase == "waiting_for_answer":
        phase = "asking"
    if phase in CALL_PHASES:
        state["phase"] = phase
    profile = raw.get("profile")
    if isinstance(profile, dict):
        for field in PROFILE_FIELDS:
            if field not in profile:
                continue
            if field == "children":
                children = profile.get("children")
                state["profile"]["children"] = list(children) if isinstance(children, list) else []
            else:
                state["profile"][field] = profile.get(field)
    topics = raw.get("topics")
    if isinstance(topics, dict):
        for topic_id in TOPIC_IDS:
            saved = topics.get(topic_id)
            if isinstance(saved, dict):
                state["topics"][topic_id] = _copy_topic(saved)
    current = raw.get("current_question")
    if isinstance(current, dict):
        question = {
            "topic_id": current.get("topic_id"),
            "followup_id": current.get("followup_id"),
        }
        # Replies that did not answer the pending follow-up. Routing owns the bound.
        attempts = current.get("attempts")
        if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts > 0:
            question["attempts"] = attempts
        state["current_question"] = question
    close_reason = raw.get("close_reason")
    if close_reason in INTERNAL_CLOSE_REASONS or close_reason is None:
        state["close_reason"] = close_reason
    history = raw.get("history")
    if isinstance(history, list):
        state["history"] = [item for item in history if isinstance(item, dict)][-_HISTORY_LIMIT:]
    return state


def append_history(state: dict, event: dict) -> dict:
    if not isinstance(event, dict):
        return state
    history = state.get("history")
    if not isinstance(history, list):
        history = []
        state["history"] = history
    history.append(event)
    if len(history) > _HISTORY_LIMIT:
        del history[:-_HISTORY_LIMIT]
    return state
