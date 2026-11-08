"""Six state-driven typed calls. Run with --live against a server on port 8023."""

import asyncio
import json
import sys
from inspect import signature

import app
import call_state
import checklist
import db


def _intro(name, married, children):
    return {
        ("intro_name", None): f"मेरा नाम {name} है।",
        ("intro_married", None): married,
        ("intro_children", None): children,
        ("intro_bridge", None): "ठीक है।",
    }


def reply_for(persona, state, memory):
    if not state or state.get("phase") in ("closing", "ended"):
        return None
    cur = state.get("current_question") or {}
    tid, fid = cur.get("topic_id"), cur.get("followup_id")
    key = (tid, fid)
    if persona == "two":
        table = _intro("सीमा", "शादी हो गई है।", "दो बच्चे हैं। बड़ा छह साल का है और छोटा दो साल का है।")
        table[("birth_mode", None)] = "बड़े वाले का ऑपरेशन हुआ, छोटे वाले का नॉर्मल।"
        table[("body_fits_milestones", None)] = "हाँ, बड़े वाले को झटके आते हैं।"
        if key in table:
            return table[key]
        if tid == "birth_mode":
            return "बड़े वाले का ऑपरेशन, अस्पताल में। छोटे वाले का नॉर्मल।"
        if tid == "body_fits_milestones":
            return "बड़े वाले को, एक साल पहले।"
        return "नहीं।"
    if persona == "unmarried":
        table = _intro("रीना", "शादी नहीं हुई।", "कोई बच्चा नहीं है।")
        table[("profile.ever_pregnant", None)] = "हाँ, पहले गर्भ रहा था।"
        table[("fam_miscarriage", None)] = "हाँ, तीन महीने में गर्भ गिर गया।"
        table[("close_anything_else", None)] = "नहीं, बस इतना ही।"
        if key in table:
            return table[key]
        if fid is not None:
            return "पता नहीं।"
        return "नहीं।"
    if persona == "none":
        table = _intro("लता", "शादी हो गई है।", "कोई बच्चा नहीं है।")
        table[("profile.ever_pregnant", None)] = "नहीं, कभी प्रेग्नेंट नहीं हुई।"
        table[("close_anything_else", None)] = "नहीं, बस इतना ही।"
        return table.get(key, "नहीं।")
    if persona == "emergency":
        table = _intro("सुनीता", "शादी हो गई है।", "एक बच्चा है, तीन साल का।")
        if key in table:
            return table[key]
        if str(tid).startswith("birth") and memory["birth"] < 2:
            memory["birth"] += 1
            return "नॉर्मल डिलीवरी हुई, अस्पताल में।" if memory["birth"] == 1 else "पूरे नौ महीने में हुआ।"
        return "अभी बच्चे को साँस नहीं आ रही। इमरजेंसी है।"
    if persona == "stop":
        table = _intro("पूजा", "शादी हो गई है।", "एक बच्चा है, दो साल का।")
        if key in table and key[0] != "intro_bridge":
            return table[key]
        if not memory["extra"]:
            memory["extra"] = True
            return "ठीक है।"
        return "नहीं, कॉल एंड करिए प्लीज।"
    if persona == "confused":
        table = _intro("कमला", "शादी हो गई है।", "एक बच्चा है, चार साल का।")
        if key in table:
            return table[key]
        if fid is not None and not memory["clarified"]:
            memory["clarified"] = True
            memory["stuck"] = key
            return "क्या मतलब?"
        if memory.get("stuck") and key == memory["stuck"]:
            memory["restuck"] = True
        if fid is not None:
            return "पता नहीं।"
        if tid == "close_anything_else":
            return "नहीं, बस इतना ही।"
        return "नहीं।"
    return "नहीं।"


async def one_call(persona, url, headers):
    import websockets

    memory = {"birth": 0, "extra": False, "clarified": False, "restuck": False, "stuck": None}
    call_id = None
    step = 0
    spoke = False
    try:
        async with websockets.connect(url, **headers) as ws:
            while step < 90:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                payload = json.loads(raw)
                kind = payload.get("type")
                if kind == "session_started":
                    call_id = payload.get("call_id")
                    print(f"[{persona}] call {call_id}", flush=True)
                elif kind in ("audio",) or (kind == "transcript" and payload.get("role") == "assistant"):
                    spoke = True
                elif kind == "telemetry" and (payload.get("last_event") or {}).get("type") == "turn_complete" and spoke and call_id:
                    spoke = False
                    state = (db.get_call(call_id) or {}).get("state") or {}
                    text = reply_for(persona, state, memory)
                    if not text:
                        continue
                    step += 1
                    cur = state.get("current_question") or {}
                    print(f"[{persona}] {step} {cur.get('topic_id')}/{cur.get('followup_id')}: {text}", flush=True)
                    await ws.send(json.dumps({"type": "text", "text": text}))
                elif kind == "call_completed":
                    return {"persona": persona, "call_id": call_id, "memory": memory, "status": "completed"}
    except Exception as exc:
        return {"persona": persona, "call_id": call_id, "memory": memory, "status": f"error: {exc}"}
    return {"persona": persona, "call_id": call_id, "memory": memory, "status": "cap"}


def judge(persona, row):
    call_id = row.get("call_id")
    call = db.get_call(call_id) or {} if call_id else {}
    state = call.get("state") or {}
    tel = call.get("telemetry") or {}
    reason = state.get("close_reason")
    errors = tel.get("tool_errors") or []
    unresolved = call_state.unresolved_topic_ids(state) if state else ["missing"]
    topics = state.get("topics") or {}
    profile = state.get("profile") or {}
    ok = False
    note = ""
    if persona == "two":
        fits = (topics.get("body_fits_milestones") or {}).get("value") is True
        ok = reason == "complete" and not unresolved and fits
        note = f"fits={fits}"
    elif persona == "unmarried":
        misc = (topics.get("fam_miscarriage") or {}).get("value") is True
        related = (topics.get("fam_related") or {}).get("state") == "not_applicable"
        ok = reason == "complete" and not unresolved and misc and related
        note = f"miscarriage={misc} related={related}"
    elif persona == "none":
        child_topics = [
            tid for tid, topic in checklist._BY_ID.items()
            if topic["group"] in ("birth", "body", "illness")
        ]
        skipped = all((topics.get(tid) or {}).get("state") == "not_applicable" for tid in child_topics)
        ok = reason == "complete" and not unresolved and profile.get("ever_pregnant") is False and skipped
        note = f"ever_pregnant={profile.get('ever_pregnant')} skipped={skipped}"
    elif persona == "emergency":
        ok = reason == "emergency" and not call_state.is_complete(state)
        note = "emergency"
    elif persona == "stop":
        ok = reason == "caller_stop" and not call_state.is_complete(state)
        note = "caller_stop"
    elif persona == "confused":
        ok = reason == "complete" and not unresolved and row["memory"].get("restuck")
        note = f"reasked={row['memory'].get('restuck')}"
    return {
        "persona": persona,
        "call_id": call_id,
        "close_reason": reason,
        "tool_errors": len(errors),
        "unresolved": len(unresolved),
        "pass": ok,
        "note": note,
        "status": row.get("status"),
    }


async def main():
    url = "ws://127.0.0.1:8023/ws/gemini"
    token = app._session_token("user", "jigar")
    sig = signature(__import__("websockets").connect)
    if "additional_headers" in sig.parameters:
        headers = {"additional_headers": [("Cookie", f"{app.COOKIE_NAME}={token}")]}
    else:
        headers = {"extra_headers": {"Cookie": f"{app.COOKIE_NAME}={token}"}}
    order = ["two", "unmarried", "none", "emergency", "stop", "confused"]
    rows = []
    for start in (0, 3):
        batch = order[start:start + 3]
        done = await asyncio.gather(*(one_call(name, url, headers) for name in batch))
        rows.extend(judge(item["persona"], item) for item in done)
        for row in rows[-3:]:
            print("RESULT", row, flush=True)
    lines = [
        "# Six personas",
        "",
        "| Persona | Call | Close | Tool errors | Unresolved | Pass | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['persona']} | {row['call_id']} | {row['close_reason']} | {row['tool_errors']} | {row['unresolved']} | {row['pass']} | {row['note']} |"
        )
    path = "reports/call-state/2026-11-08-six-personas.md"
    open(path, "w").write("\n".join(lines) + "\n")
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
