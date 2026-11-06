"""Text-only smoke call: real SHISHU prompt, config and checklist against Gemini Live.

Run from code/shishu-v2: PYTHONPATH=. /Users/jigar/pyenv/shishu/bin/python live_smoke.py [v1beta|v1alpha]
"""

import asyncio
import os
import sys
import time

from google import genai
from google.genai import types

import checklist
import gemini_live

MOTHER_TURNS = [
    "Mera naam Sunita hai.",
    "Haan, shaadi ho gayi hai.",
    "Ek beti hai, chaar saal ki.",
    "Theek hai.",
    "Normal delivery, hospital mein.",
    "Matlab? Samajh nahi aaya.",
    "Poore nau mahine.",
]
FILLER = "Nahi."
MAX_TURNS = 80
TURN_TIMEOUT_SEC = 25
RESUME_AT_TURN = 10


async def _collect(session, coverage: list, log: list, out: dict) -> None:
    while True:
        tool_after_text = False
        async for response in session.receive():
            update = response.session_resumption_update
            if update is not None and update.resumable and update.new_handle:
                out["handle"][0] = update.new_handle
            if response.tool_call:
                replies = []
                for fc in response.tool_call.function_calls:
                    args = dict(fc.args or {})
                    if fc.name == "record_finding":
                        checklist.mark(coverage, args.get("topic_ids") or [])
                        body = {"result": "ok", "note": checklist.tool_note(coverage)}
                        log.append(f"  [tool] record_finding {args.get('domain')} {args.get('topic_ids')}")
                    elif fc.name == "end_call":
                        ok, note = checklist.can_end(coverage)
                        body = {"result": "ok"} if ok else {"result": "not_yet", "note": note}
                        out["ended"] = ok
                        log.append(f"  [tool] end_call -> {body['result']}")
                    else:
                        body = {"result": "ignored"}
                    replies.append(types.FunctionResponse(id=fc.id, name=fc.name, response=body))
                await session.send_tool_response(function_responses=replies)
                tool_after_text = True
            sc = response.server_content
            if sc is None:
                continue
            if sc.output_transcription and sc.output_transcription.text:
                if out.get("first_ms") is None:
                    out["first_ms"] = int((time.perf_counter() - out["t0"]) * 1000)
                out["parts"].append(sc.output_transcription.text)
                tool_after_text = False
            if sc.turn_complete:
                break
        if not tool_after_text or out.get("ended"):
            return


async def main(api_version: str) -> int:
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"api_version": api_version},
    )
    chosen = gemini_live._session_settings({})
    config = gemini_live._live_config({})
    coverage: list = []
    log: list = []
    latencies: list[int] = []
    ended = False
    resumed = False
    handle = [""]
    started = time.perf_counter()
    ctx = client.aio.live.connect(model=chosen["model"], config=config)
    session = await ctx.__aenter__()
    try:
        await session.send_realtime_input(text=gemini_live._GREETING)
        for i in range(MAX_TURNS):
            out = {"parts": [], "t0": time.perf_counter(), "first_ms": None, "ended": False, "handle": handle}
            try:
                await asyncio.wait_for(_collect(session, coverage, log, out), TURN_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                log.append("  [timeout]")
            if out["first_ms"] is not None:
                latencies.append(out["first_ms"])
            print(f"SHISHU ({out['first_ms']} ms): {''.join(out['parts']).strip()}")
            for line in log:
                print(line)
            log.clear()
            if out["ended"]:
                ended = True
                break
            if i == RESUME_AT_TURN:
                if handle[0]:
                    await ctx.__aexit__(None, None, None)
                    ctx = client.aio.live.connect(
                        model=chosen["model"],
                        config=gemini_live._live_config({}, resume_handle=handle[0]),
                    )
                    session = await ctx.__aenter__()
                    resumed = True
                    print("[resumed with handle]")
                else:
                    print("[resume skipped: no handle received]")
            mother = MOTHER_TURNS[i] if i < len(MOTHER_TURNS) else FILLER
            print(f"MOTHER: {mother}")
            await session.send_realtime_input(text=mother)
    finally:
        await ctx.__aexit__(None, None, None)
    done = sum(1 for tid in checklist.REQUIRED_IDS if tid in coverage)
    median = sorted(latencies)[len(latencies) // 2] if latencies else -1
    print(
        f"RESULT covered={done}/{len(checklist.REQUIRED_IDS)} ended={ended} resumed={resumed} "
        f"median_first_text_ms={median} seconds={int(time.perf_counter() - started)}"
    )
    return 0 if ended and resumed and done == len(checklist.REQUIRED_IDS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "v1beta")))
