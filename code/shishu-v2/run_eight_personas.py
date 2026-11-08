"""Run 8 real clinical personas (from Case scenarios.docx) over live /ws/gemini."""

import asyncio
import json
import sys
from inspect import signature
from pathlib import Path

from dotenv import load_dotenv

import app
import call_state
import checklist
import db
import models
import mother_report

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _intro(name: str, married: str, children: str) -> dict:
    return {
        ("intro_name", None): f"मेरा नाम {name} है।",
        ("intro_married", None): married,
        ("intro_children", None): children,
        ("intro_bridge", None): "ठीक है।",
    }


PERSONAS = {
    "p1_scid_bcg": {
        "desc": "1yo boy, consanguineous (mama ki beti), BCG ulceration, 6 admissions in 3m for pneumonia + IV antibiotics",
        "table": {
            **_intro("सलमा", "शादी हो गई है।", "एक बेटा है, एक साल का।"),
            ("birth_mode", None): "नॉर्मल डिलीवरी हुई, अस्पताल में।",
            ("birth_bcg", None): "हाँ, BCG के टीके की जगह पर घाव और फोड़ा हो गया था।",
            ("birth_bcg", "reaction"): "टीके की जगह पर फोड़ा और घाव बन गया था।",
            ("ill_admissions", None): "हाँ, निमोनिया की वजह से तीन महीने में 6 बार अस्पताल में भर्ती करना पड़ा।",
            ("ill_admissions", "count"): "छह बार भर्ती हुआ।",
            ("ill_admissions", "reasons"): "बार-बार निमोनिया और साँस में तकलीफ़ की वजह से।",
            ("ill_iv_antibiotics", None): "हाँ, हर बार नस में सुई से एंटीबायोटिक दवा चली थी।",
            ("ill_iv_antibiotics", "count"): "छह बार।",
            ("ill_iv_antibiotics", "infection"): "निमोनिया के लिए।",
            ("fam_related", None): "हाँ, मेरे पति मेरे सगे मामा के बेटे हैं।",
            ("fam_related", "relationship"): "मेरे सगे मामा के बेटे हैं।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": ("birth_bcg", "ill_admissions", "ill_iv_antibiotics", "fam_related"),
    },
    "p2_hemophilia": {
        "desc": "7yo boy, knee swelling after falling (5-7 days), infant brother died at 5m after seizures; asks क्या मतलब once",
        "table": {
            **_intro("कविता", "शादी हो गई है।", "अभी एक बेटा है, सात साल का।"),
            ("birth_mode", None): "नॉर्मल डिलीवरी हुई, अस्पताल में।",
            ("body_joint_swelling", None): "हाँ, खेलते समय गिरने पर इसके घुटनों में सूजन आ जाती है।",
            ("body_joint_swelling", "joint"): "दोनों घुटनों में सूजन आती है।",
            ("body_joint_swelling", "trigger"): "खेलते वक्त गिरने या चोट लगने से।",
            ("body_joint_swelling", "duration"): "सूजन पाँच से सात दिन तक रहती है।",
            ("fam_infant_death", None): "हाँ, मेरा एक पहला बेटा पाँच महीने की उम्र में झटके आने के बाद गुज़र गया था।",
            ("fam_infant_death", "relationship"): "मेरा अपना बड़ा बेटा था।",
            ("fam_infant_death", "age_at_death"): "पाँच महीने की उम्र में।",
            ("fam_infant_death", "cause"): "तेज़ झटके आए थे और फिर गुज़र गया।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": ("body_joint_swelling", "fam_infant_death"),
    },
    "p3_thal_trait_3kids": {
        "desc": "3 kids (6y, 4y, 3m girl with low Hb), transfusion in 1st pregnancy, mother thalassemia trait, consanguineous",
        "table": {
            **_intro(
                "शबाना",
                "शादी हो गई है।",
                "तीन बच्चे हैं। बड़ा बेटा छह साल का, दूसरा बेटा चार साल का, और छोटी बेटी तीन महीने की है।",
            ),
            ("birth_mode", None): "तीनों नॉर्मल डिलीवरी से अस्पताल में हुए।",
            ("body_blood_count", None): "हाँ, तीन महीने वाली छोटी बेटी का हीमोग्लोबिन कम बताया है।",
            ("body_blood_count", "low_or_high"): "खून कम बताया है।",
            ("body_blood_count", "timing"): "अभी पिछले महीने जाँच में पता चला।",
            ("body_blood_count", "affected_child"): "तीन महीने वाली छोटी बेटी को।",
            ("preg_thalassemia", None): "हाँ, मेरी जाँच में थैलेसीमिया माइनर आया था, और पति की रिपोर्ट नॉर्मल थी।",
            ("preg_thalassemia", "caller_result"): "मुझे थैलेसीमिया माइनर बताया था।",
            ("preg_thalassemia", "partner_result"): "पति की रिपोर्ट नॉर्मल थी।",
            ("preg_transfusion", None): "हाँ, पहली प्रेग्नेंसी के सातवें महीने में मुझे खून चढ़ा था।",
            ("preg_transfusion", "pregnancy_number"): "पहली प्रेग्नेंसी में।",
            ("preg_transfusion", "month"): "सातवें महीने में।",
            ("fam_related", None): "हाँ, हम शादी से पहले रिश्तेदार थे, मेरे बुआ के बेटे हैं।",
            ("fam_related", "relationship"): "मेरे सगे बुआ के बेटे हैं।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": ("body_blood_count", "preg_thalassemia", "preg_transfusion", "fam_related"),
    },
    "p4_storage_spleen": {
        "desc": "3yo boy, progressive tummy distension (liver/spleen big), 1 transfusion, poor growth, bruises",
        "table": {
            **_intro("सुनीता", "शादी हो गई है।", "एक बेटा है, तीन साल का।"),
            ("birth_mode", None): "नॉर्मल डिलीवरी हुई, अस्पताल में।",
            ("body_tummy", None): "हाँ, बच्चे का पेट धीरे-धीरे फूल रहा है और डॉक्टर ने कहा तिल्ली और लिवर बड़ा है।",
            ("body_tummy", "onset"): "करीब एक साल से पेट बढ़ रहा है।",
            ("body_tummy", "clinician_comment"): "डॉक्टर ने अल्ट्रासाउंड के बाद कहा कि लिवर और तिल्ली बढ़े हुए हैं।",
            ("body_transfusion", None): "हाँ, खून की कमी की वजह से एक बार खून चढ़ाना पड़ा था।",
            ("body_transfusion", "count"): "एक बार।",
            ("body_transfusion", "last_date"): "छह महीने पहले चढ़ा था।",
            ("body_blood_count", None): "हाँ, डॉक्टर ने बताया खून की गिनती कम रहती है।",
            ("body_blood_count", "low_or_high"): "कम रहती है।",
            ("body_blood_count", "timing"): "छह महीने पहले से।",
            ("body_bleeding_bruise", None): "हाँ, शरीर पर अपने आप नील पड़ जाते हैं।",
            ("body_bleeding_bruise", "bleeding_or_bruises"): "नील पड़ जाते हैं।",
            ("body_growth", None): "हाँ, हमउम्र बच्चों के हिसाब से इसका वज़न और कद नहीं बढ़ रहा है।",
            ("body_growth", "reported_by"): "डॉक्टर ने भी कहा और हमें भी दिखता है।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": ("body_tummy", "body_transfusion", "body_blood_count", "body_bleeding_bruise", "body_growth"),
    },
    "p5_blind_delayed_8m": {
        "desc": "8m infant, cannot see and still cannot sit",
        "table": {
            **_intro("मीना", "शादी हो गई है।", "एक बच्चा है, आठ महीने का।"),
            ("birth_mode", None): "नॉर्मल डिलीवरी हुई, अस्पताल में।",
            ("birth_hearing_vision", None): "हाँ, मुझे लगता है बच्चे को दिखाई नहीं देता, आँख से कुछ देखता नहीं है।",
            ("birth_hearing_vision", "hearing_or_vision"): "देखने में दिक्कत है।",
            ("body_fits_milestones", None): "हाँ, आठ महीने का हो गया है पर अभी तक बैठ नहीं पाता और सिर्फ़ रोता रहता है।",
            ("body_fits_milestones", "type"): "बैठना अभी तक शुरू नहीं किया।",
            ("body_fits_milestones", "onset"): "छठे महीने से ध्यान दिया।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": ("birth_hearing_vision", "body_fits_milestones"),
    },
    "p6_thal_major_10yo": {
        "desc": "10yo girl, monthly blood transfusions, elder sister died of bleeding, mother has heavy periods",
        "table": {
            **_intro("रेखा", "शादी हो गई है।", "एक बेटी है, दस साल की।"),
            ("birth_mode", None): "नॉर्मल डिलीवरी हुई, अस्पताल में।",
            ("body_transfusion", None): "हाँ, बेटी को हर महीने खून चढ़ाना पड़ता है।",
            ("body_transfusion", "count"): "अब तक पचास से ज़्यादा बार चढ़ चुका है, हर महीने लगता है।",
            ("body_transfusion", "last_date"): "पिछले महीने ही चढ़ाया है।",
            ("fam_bleeding", None): "हाँ, मेरी एक बड़ी बेटी को बहुत ज़्यादा खून बहने की बीमारी थी और मुझे भी माहवारी में बहुत खून आता है।",
            ("fam_bleeding", "who"): "मेरी बड़ी बेटी और मुझे खुद।",
            ("fam_bleeding", "relationship"): "मेरी अपनी बड़ी बेटी थी।",
            ("fam_bleeding", "context"): "चोट लगने और माहवारी में बहुत ज़्यादा खून बहता था।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": ("body_transfusion", "fam_bleeding"),
        "clarify_steps": (8, 20, 35),
    },
    "p7_all_okay": {
        "desc": "Healthy 5yo child, no red flags anywhere",
        "table": {
            **_intro("अंजली", "शादी हो गई है।", "एक बच्चा है, पाँच साल का।"),
            ("birth_mode", None): "नॉर्मल डिलीवरी हुई, अस्पताल में।",
            ("birth_mode", "delivery_type"): "नॉर्मल डिलीवरी।",
            ("birth_mode", "place"): "अस्पताल में।",
            ("close_anything_else", None): "नहीं, बस इतना ही।",
        },
        "expect_close": "complete",
        "expect_yes": (),
    },
    "p8_emergency_interrupt": {
        "desc": "Mother interrupts mid-call because baby is turning blue and not breathing",
        "table": {
            **_intro("पूजा", "शादी हो गई है।", "एक बच्चा है, एक साल का।"),
            ("birth_mode", None): "अस्पताल में ऑपरेशन से हुआ था क्योंकि पानी कम था।",
            ("birth_mode", "operation_reason"): "डॉक्टर ने कहा पानी कम हो गया था।",
        },
        "emergency_on": "birth_term",
        "expect_close": "emergency",
        "expect_yes": ("birth_mode", "body_blue"),
    },
}


def _facts(spec: dict) -> str:
    lines = [spec.get("desc") or ""]
    for (topic_id, followup_id), answer in spec["table"].items():
        which = topic_id if not followup_id else f"{topic_id} / {followup_id}"
        lines.append(f"{which}: {answer}")
    return "\n".join(lines)


async def ask_mother(spec: dict, bot_line: str, step: int) -> str:
    """One spoken reply to the line the bot actually said."""
    if step in spec.get("clarify_steps", ()):
        return "क्या मतलब?"
    behaviour = ""
    if spec.get("emergency_on"):
        behaviour = (
            "If she asks whether the baby was born much too early or much too late, "
            "do not answer it. Say exactly: रुकिए, अभी मेरे बच्चे को साँस नहीं आ रही है "
            "और होंठ नीले पड़ रहे हैं, इमरजेंसी है!"
        )
    from google.genai import types

    client = models.text_client()
    response = await client.aio.models.generate_content(
            model=models.TEXT_MODEL,
            contents=(
                f"Step {step}.\n{behaviour}\nFacts:\n{_facts(spec)}\n\n"
                f"The assistant just said: {bot_line or '(nothing)'}"
            ),
            config=models.text_config(
                system=(
                    "You are the mother on a phone call. Reply with one short spoken Hindi sentence "
                    "to the assistant's last line, using only the facts. If the facts do not mention "
                    "a problem, say नहीं. Never answer a question she did not just ask."
                ),
                schema=types.Schema(
                    type="OBJECT",
                    properties={"reply": types.Schema(type="STRING")},
                    required=["reply"],
                ),
            ),
        )
    return str((json.loads(response.text or "{}")).get("reply") or "").strip() or "नहीं।"


def _state_snapshot(state: dict) -> tuple:
    topics = {
        key: (row.get("state"), row.get("value"), row.get("details"))
        for key, row in (state.get("topics") or {}).items()
    }
    return (topics, dict(state.get("profile") or {}))


async def one_call(name: str, url: str, headers: dict) -> dict:
    import websockets

    memory = {"clarify_mutated": False, "desyncs": 0}
    call_id = None
    step = 0
    spoke = False
    last_bot = ""
    clarify_snapshot = None
    try:
        async with websockets.connect(url, **headers) as ws:
            while step < 95:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                payload = json.loads(raw)
                kind = payload.get("type")
                if kind == "error":
                    raise RuntimeError(str(payload.get("message") or "server error"))
                if kind == "session_started":
                    call_id = payload.get("call_id")
                    print(f"[{name}] started call_id={call_id}", flush=True)
                elif kind == "audio":
                    spoke = True
                elif kind == "transcript" and payload.get("role") == "assistant":
                    spoke = True
                    last_bot = (last_bot + " " + str(payload.get("text") or "")).strip()[-400:]
                elif (
                    kind == "telemetry"
                    and (payload.get("last_event") or {}).get("type") == "turn_complete"
                    and spoke
                    and call_id
                ):
                    spoke = False
                    state = (db.get_call(call_id) or {}).get("state") or {}
                    if clarify_snapshot is not None:
                        now = _state_snapshot(state)
                        if now != clarify_snapshot:
                            memory["clarify_mutated"] = True
                        clarify_snapshot = None
                    if state.get("phase") in ("closing", "ended"):
                        last_bot = ""
                        continue
                    cur = state.get("current_question") or {}
                    heard = last_bot
                    if not heard.strip():
                        memory["desyncs"] += 1
                        print(
                            f"[{name}] DESYNC step {step + 1} server={cur.get('topic_id')} "
                            f"asked nothing",
                            flush=True,
                        )
                    clarifying = (step + 1) in PERSONAS[name].get("clarify_steps", ())
                    text = await ask_mother(PERSONAS[name], heard, step + 1)
                    last_bot = ""
                    if not text:
                        continue
                    step += 1
                    if clarifying:
                        clarify_snapshot = _state_snapshot(state)
                    asked = checklist.question_hi(str(cur.get("topic_id") or ""))
                    print(
                        f"[{name}] step {step} server={cur.get('topic_id')}/{cur.get('followup_id')} "
                        f"q={asked!r} bot={heard!r} -> {text}",
                        flush=True,
                    )
                    await ws.send(json.dumps({"type": "text", "text": text}))
                elif kind == "call_completed":
                    return {"persona": name, "call_id": call_id, "steps": step, "memory": memory, "status": "completed"}
    except Exception as exc:
        return {"persona": name, "call_id": call_id, "steps": step, "memory": memory, "status": f"error: {exc}"}
    return {"persona": name, "call_id": call_id, "steps": step, "memory": memory, "status": "step_cap"}


async def evaluate_call(row: dict) -> dict:
    name = row["persona"]
    spec = PERSONAS[name]
    call_id = row.get("call_id")
    # Wait briefly for async _write_doctor_note (frontier Gemini reconciler) to finish
    for _ in range(90):
        call = db.get_call(call_id) if call_id else None
        if call and (call.get("summary_doctor") or "").strip():
            break
        await asyncio.sleep(1.0)
    call = db.get_call(call_id) or {} if call_id else {}
    state = call.get("state") or {}
    tel = call.get("telemetry") or {}
    reason = state.get("close_reason")
    tool_errors = tel.get("tool_errors") or []
    unresolved = call_state.unresolved_topic_ids(state) if state else ["missing"]
    topics = state.get("topics") or {}

    actual_yes = tuple(
        tid for tid in checklist.REQUIRED_IDS
        if not tid.startswith("intro_")
        and tid != "close_anything_else"
        and tid in topics
        and call_state.topic_answer_kind(topics[tid]) == "yes"
    )
    missing_yes = [tid for tid in spec["expect_yes"] if tid not in actual_yes]
    extra_yes = [tid for tid in actual_yes if tid not in spec["expect_yes"]]

    report = mother_report.build(call) if call else {}
    yes_lines = report.get("yes") or []
    questions = report.get("questions") or []
    has_doctor_note = bool((call.get("summary_doctor") or "").strip())

    close_ok = reason == spec["expect_close"]
    complete_ok = (not unresolved) if spec["expect_close"] == "complete" else True

    passed = (
        row.get("status") == "completed"
        and close_ok
        and complete_ok
        and not missing_yes
        and not extra_yes
        and not row["memory"].get("clarify_mutated")
        and has_doctor_note
    )
    return {
        "persona": name,
        "call_id": call_id,
        "status": row.get("status"),
        "steps": row.get("steps"),
        "close_reason": reason,
        "tool_errors": len(tool_errors),
        "unresolved": len(unresolved),
        "expected_yes": list(spec["expect_yes"]),
        "actual_yes": list(actual_yes),
        "missing_yes": missing_yes,
        "extra_yes": extra_yes,
        "yes_hindi_count": len(yes_lines),
        "questions_count": len(questions),
        "has_doctor_note": has_doctor_note,
        "clarify_mutated": row["memory"].get("clarify_mutated"),
        "desyncs": row["memory"].get("desyncs"),
        "pass": passed,
    }


async def main():
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8023"
    url = f"ws://127.0.0.1:{port}/ws/gemini"
    token = app._session_token("user", "jigar")
    sig = signature(__import__("websockets").connect)
    if "additional_headers" in sig.parameters:
        headers = {"additional_headers": [("Cookie", f"{app.COOKIE_NAME}={token}")]}
    else:
        headers = {"extra_headers": {"Cookie": f"{app.COOKIE_NAME}={token}"}}

    names = list(PERSONAS.keys())
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
        names = [n for n in names if n in only]

    results = []
    # Run in batches of 2 to avoid Gemini Live rate limits
    for i in range(0, len(names), 2):
        batch = names[i : i + 2]
        raw_rows = await asyncio.gather(*(one_call(n, url, headers) for n in batch))
        eval_rows = await asyncio.gather(*(evaluate_call(r) for r in raw_rows))
        for r in eval_rows:
            print("EVAL:", json.dumps(r, ensure_ascii=False), flush=True)
            results.append(r)

    print("\n=== SUMMARY ===")
    passed = sum(1 for r in results if r["pass"])
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status_str = "PASS" if r["pass"] else "FAIL"
        print(
            f"- {r['persona']}: {status_str} (call={r['call_id']}, steps={r['steps']}, "
            f"close={r['close_reason']}, tool_err={r['tool_errors']}, unresolved={r['unresolved']}, "
            f"missing_yes={r['missing_yes']}, extra_yes={r['extra_yes']}, "
            f"clarify_mutated={r.get('clarify_mutated')}, desyncs={r.get('desyncs')}, "
            f"pdf_yes={r['yes_hindi_count']}, "
            f"pdf_q={r['questions_count']}, doc_note={r['has_doctor_note']})"
        )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
