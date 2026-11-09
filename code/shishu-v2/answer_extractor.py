"""Post-call read of a saved transcript. The live call does not use this."""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

import checklist
import db
import question_flow
import summarizer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

_ANSWERS = ("yes", "no", "unknown", "not_asked")
_VERSION = 7
_GEMINI_MODEL = "gemini-3.8-flash"
_OPENAI_MODEL = "gpt-4o-mini"
# Evidence in the prompt is capped so this bound stays under the old 8192 cap.
# 2 characters per token is the floor used for Hindi JSON. The envelope covers
# the wrapper and a few quotes that run past the asked 80 characters.
_EVIDENCE_CHARS = 80

_DOMAIN_TOPIC = {
    "birth_nicu": "birth_nicu",
    "milestones": "body_fits_milestones",
    "bleeding_joints": "body_joint_swelling",
    "metabolic_fasting": "ill_fasting",
    "infections": "ill_admissions",
    "pregnancy": "preg_conditions",
    "consanguinity": "fam_related",
    "family_infant_death": "fam_infant_death",
    "family_history": "fam_fits_delay",
}

def _max_output_tokens() -> int:
    slot_chars = 0
    for rule in checklist.REPORT_RULES.values():
        slots = rule.get("slots") or {}
        if slots:
            slot_chars += 32 + 20 * len(slots)
    chars = 64 + len(checklist.TOPICS) * (64 + _EVIDENCE_CHARS) + slot_chars
    return chars // 2 + 512


_MAX_OUTPUT_TOKENS = _max_output_tokens()

_PROMPT = f"""
You read a finished SHISHU screening call. You extract what the mother said. You do not decide whether it is a doctor concern, and you do not return a second clinical yes or no.

For every topic id return answer, evidence, and slots.
answer is the literal reply to the question that was asked, one of: yes, no, unknown, not_asked.
- not_asked: the bot never asked it, and she did not answer it in her own story.
- unknown: she said she does not know, or does not remember.
- yes: she agreed with the question, or answered it in the affirmative.
- no: she said no, or said that thing did not happen.
evidence is a verbatim contiguous quote from the user turn that answers that topic, at most {_EVIDENCE_CHARS} characters. Do not translate, paraphrase, combine separate turns, or reuse a reply from a different question. A one-word answer such as "नहीं" is valid only for the topic asked just before that reply. For yes, no, or unknown, if that question's answer has no exact user quote, use not_asked and an empty evidence string.
slots is an object only for the topics listed below. Use only the allowed values. For every other topic, slots must be {{}}.
{checklist.slot_prompt_lines()}

A later clear answer replaces an earlier one. Do not invent. Intro topics and close_anything_else are ordinary facts: record the literal answer.

You may receive saved live-call findings after the transcript. They are fallible, broad notes. They never override a quoted answer or a typed slot. If the quote and a finding disagree, keep the quote and the slots. Use unknown only when the quote itself does not say.

Return JSON only: {{"answers": {{"topic_id": {{"answer": "no", "evidence": "quote", "slots": {{}}}}}}}}
""".strip()


def _topics_brief() -> str:
    lines = []
    for topic in checklist.TOPICS:
        q_hi = checklist.question_hi(topic["id"])
        lines.append(f"{topic['id']} ({q_hi}): {topic['need']}")
    return "\n".join(lines)


def _transcript_text(transcript: list) -> str:
    lines = []
    for turn in transcript or []:
        role = str(turn.get("role") or "")
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _live_findings_text(call: dict) -> str:
    lines = []
    for domain, finding in (call.get("findings") or {}).items():
        if not isinstance(finding, dict):
            continue
        status = str(finding.get("status") or "").strip()
        detail = str(finding.get("detail") or "").strip()
        if status or detail:
            lines.append(f"{domain}: status={status}; detail={detail}")
    return "\n".join(lines) or "(none saved)"


# Grammar and acknowledgements. These are not question identity.
_STOP = {
    "theek", "thik", "hai", "hain", "ji", "achha", "accha", "kya", "mein", "main",
    "aur", "ya", "se", "par", "ko", "ka", "ke", "ki", "tha", "thi", "hua", "hui",
    "hue", "aap", "aapke", "aapki", "aapka", "kisi", "koi", "nahin", "nahi",
    "yeh", "woh", "toh", "bhi", "hota", "hoti", "hote", "liye", "tak", "bahut",
    "abhi", "bas", "gayi", "gaya", "gaye", "sakti", "sakta", "karte", "karti",
    "ठीक", "जी", "है", "हैं", "क्या", "में", "की", "के", "का", "को", "से", "या",
    "और", "था", "थी", "हुआ", "हुई", "किसी", "कोई", "यह", "तो", "भी", "नहीं",
    "बहुत", "अभी", "गई", "गया", "सकती", "सकता",
}


def _fold(text: str) -> str:
    """Casefold and drop virama so loose spellings of the same word still match."""
    return (text or "").replace("\u094d", "").casefold()


def _tokens(text: str) -> set[str]:
    found = set(re.findall(r"[a-z]{4,}|[\u0900-\u097f]{2,}", _fold(text)))
    return {token for token in found if token not in _STOP}


def _topic_blob(topic_id: str) -> str:
    topic = checklist._BY_ID[topic_id]
    return " ".join(
        (
            str(topic.get("words") or ""),
            question_flow.WIDE_WORDS.get(topic_id, ""),
            checklist.question_hi(topic_id),
        )
    )


_TOPIC_TOKENS = {topic["id"]: _tokens(_topic_blob(topic["id"])) for topic in checklist.TOPICS}
_TOKEN_OWNERS: dict[str, set[str]] = {}
for _topic_id, _topic_tokens in _TOPIC_TOKENS.items():
    for _token in _topic_tokens:
        _TOKEN_OWNERS.setdefault(_token, set()).add(_topic_id)


def _exact_question(assistant: str, topic_id: str) -> bool:
    text = _fold(assistant)
    topic = checklist._BY_ID[topic_id]
    phrases = (
        checklist.question_hi(topic_id).rstrip(" ?？"),
        str(topic.get("words") or ""),
        question_flow.WIDE_WORDS.get(topic_id, ""),
    )
    for phrase in phrases:
        folded = _fold(phrase).strip(" ?？")
        if len(folded) >= 12 and folded in text:
            return True
    return False


def _score_topic(assistant_tokens: set[str], topic_id: str) -> tuple[int, int]:
    found = assistant_tokens & _TOPIC_TOKENS[topic_id]
    distinctive = sum(1 for token in found if _TOKEN_OWNERS.get(token) == {topic_id})
    return distinctive, len(found)


def _match_topic(assistant: str) -> str | None:
    """Pick the one checklist question this assistant turn asked.

    Uses the question text that already belongs to the topic. A turn that
    matches two topics equally is left unmatched.
    """
    exact = [
        topic["id"]
        for topic in checklist.TOPICS
        if _exact_question(assistant, topic["id"])
    ]
    if len(exact) == 1:
        return exact[0]
    tokens = _tokens(assistant)
    ranked = sorted(
        ((_score_topic(tokens, topic["id"]), topic["id"]) for topic in checklist.TOPICS),
        reverse=True,
    )
    (best_distinct, best_total), best_id = ranked[0]
    (next_distinct, next_total), _next_id = ranked[1]
    if best_total < 2 or (best_distinct, best_total) == (next_distinct, next_total):
        return None
    if best_distinct == 0 and best_total < 3:
        return None
    return best_id


def _answer_turns(transcript: list) -> dict[str, list[str]]:
    """Map each topic to the user turns that answer its question."""
    paired: dict[str, list[str]] = {}
    assistant: str | None = None
    users: list[str] = []

    def flush() -> None:
        nonlocal assistant, users
        if assistant:
            topic_id = _match_topic(assistant)
            if topic_id:
                bucket = paired.setdefault(topic_id, [])
                for text in users:
                    if text.strip() and text not in bucket:
                        bucket.append(text)
        assistant = None
        users = []

    for turn in transcript or []:
        role = str(turn.get("role") or "")
        text = str(turn.get("text") or "")
        if role == "assistant":
            flush()
            assistant = text
            users = []
        elif role == "user" and assistant is not None:
            users.append(text)
    flush()
    return paired


def _quote_in_text(quote: str, text: str) -> bool:
    normalized_quote = " ".join(_fold(quote).split())
    if not normalized_quote:
        return False
    return normalized_quote in " ".join(_fold(text).split())


def _stored_answer(topic_id: str, answer: str, evidence: str, entry: dict) -> dict | None:
    if answer not in _ANSWERS:
        return None
    stored = {"answer": answer, "evidence": evidence[:240]}
    slots = checklist.accepted_slots(topic_id, entry.get("slots"))
    # literal is set only by a new extraction. Old saved rows stay without it.
    if entry.get("literal") is True and answer in {"yes", "no", "unknown"} and evidence:
        stored["literal"] = True
        if slots:
            stored["slots"] = slots
    return stored


def validate_answers(call: dict, extracted: dict) -> dict:
    """Keep a value only when its quote is in the user turn for that topic."""
    transcript = list((call or {}).get("transcript") or [])
    turns = _answer_turns(transcript)
    answers: dict[str, dict] = {}
    for topic_id, entry in ((extracted or {}).get("answers") or {}).items():
        if topic_id not in checklist._BY_ID or not isinstance(entry, dict):
            continue
        answer = str(entry.get("answer") or "")
        evidence = str(entry.get("evidence") or "").strip()
        matched = any(_quote_in_text(evidence, text) for text in turns.get(topic_id, []))
        if answer in {"yes", "no", "unknown"} and not matched:
            answer, evidence = "not_asked", ""
            entry = {"answer": answer, "evidence": evidence}
        stored = _stored_answer(topic_id, answer, evidence, entry)
        if stored is not None:
            answers[topic_id] = stored
    return {"version": _VERSION, "answers": answers}


def report_content(extracted: dict) -> dict:
    """Doctor lines only where the typed answer is a concern. The model does not decide that."""
    yes: list[str] = []
    questions: list[str] = []
    answers = (extracted or {}).get("answers") or {}
    for topic in checklist.TOPICS:
        topic_id = topic["id"]
        entry = answers.get(topic_id) or {}
        if not checklist.is_concern(topic_id, entry if isinstance(entry, dict) else None):
            continue
        evidence = str(entry.get("evidence") or "").strip()
        question = checklist.question_hi(topic_id).strip()
        yes.append(f'{question} — माँ का जवाब: “{evidence}”')
        questions.append(
            f'{question} माँ ने कहा “{evidence}”. क्या डॉक्टर या ASHA/ANM दीदी से इसकी जाँच पूछें?'
        )
    return {"yes": yes, "questions": questions}


def parse_payload(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("extractor response was not an object")
    answers: dict[str, dict] = {}
    incoming = data.get("answers") or {}
    if not isinstance(incoming, dict):
        raise ValueError("answers was not an object")
    for key, value in incoming.items():
        topic_id = str(key)
        if topic_id not in checklist._BY_ID:
            continue
        if isinstance(value, str):
            answer, evidence, slots = value, "", {}
        elif isinstance(value, dict):
            answer = str(value.get("answer") or "")
            evidence = str(value.get("evidence") or "")
            slots = value.get("slots") if isinstance(value.get("slots"), dict) else {}
        else:
            continue
        if answer not in _ANSWERS:
            continue
        # parse_payload is the fresh model contract, so the answer is literal.
        item = {"answer": answer, "evidence": evidence[:240], "literal": True}
        clean = checklist.accepted_slots(topic_id, slots)
        if clean:
            item["slots"] = clean
        answers[topic_id] = item
    return {"version": _VERSION, "answers": answers}


def fallback_from_call(call: dict) -> dict:
    answers: dict[str, dict] = {}
    findings = call.get("findings") or {}
    for domain, value in findings.items():
        if not isinstance(value, dict):
            continue
        topic_id = _DOMAIN_TOPIC.get(str(domain))
        if topic_id is None and str(domain) in checklist._BY_ID:
            topic_id = str(domain)
        if topic_id is None:
            continue
        status = str(value.get("status") or "").lower()
        detail = str(value.get("detail") or "")
        if summarizer._unknown_detail(detail):
            kind = "unknown"
        elif status == "positive" or value.get("is_red_flag"):
            kind = "yes"
        elif status == "negative":
            kind = "no"
        else:
            continue
        polarity = checklist.report_rule(topic_id)["polarity"]
        # A live finding has no typed slots. Choice, test, reassuring, and fact
        # rows stay uncertain instead of becoming a doctor concern.
        if polarity != "concern" and kind == "yes":
            kind = "unknown"
        answers[topic_id] = {"answer": kind, "evidence": detail[:240]}
    for topic in checklist.TOPICS:
        if topic["id"] not in answers:
            answers[topic["id"]] = {"answer": "not_asked", "evidence": ""}
    return validate_answers(call, {"version": _VERSION, "answers": answers})


def _user_content(call: dict) -> str:
    return (
        "Topics:\n"
        + _topics_brief()
        + "\n\nTranscript:\n"
        + _transcript_text(call.get("transcript") or [])
        + "\n\nSaved live-call findings (hints only; transcript evidence wins):\n"
        + _live_findings_text(call)
    )


def _gemini_config():
    """Gemini 3.8 rejects MINIMAL. LOW is the smallest level it accepts."""
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )


def _event(call_id: int, event_type: str, meta: dict | None = None) -> None:
    """Timing only. Detail stays empty so a transcript or error string cannot land here."""
    clean: dict = {}
    for key, value in (meta or {}).items():
        if key not in {"model", "error_class", "ms", "bytes", "reason"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, str) and len(value) <= 80 and " " not in value:
            clean[key] = value
    try:
        db.append_call_event(int(call_id), event_type, "", clean)
    except Exception:
        return


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


async def _from_gemini(user_content: str) -> dict:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        response = await client.aio.models.generate_content(
            model=_GEMINI_MODEL,
            contents=user_content,
            config=_gemini_config(),
        )
        return parse_payload(response.text or "")
    finally:
        await client.aio.aclose()
        client.close()


async def _from_openai(user_content: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = await client.chat.completions.create(
            model=_OPENAI_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return parse_payload(response.choices[0].message.content or "")
    finally:
        await client.close()


async def _from_llm(call: dict) -> tuple[dict, dict]:
    user_content = _user_content(call)
    info = {"model": _GEMINI_MODEL, "error_class": ""}
    try:
        return await _from_gemini(user_content), info
    except Exception as exc:
        error_class = type(exc).__name__
        _event(call.get("id") or 0, "extraction_fallback", {"model": _GEMINI_MODEL, "error_class": error_class})
        info = {"model": _OPENAI_MODEL, "error_class": error_class}
        return await _from_openai(user_content), info


async def cached_answers(call: dict) -> dict:
    call_id = int(call["id"])
    started = time.perf_counter()
    saved = db.get_answers(call_id)
    if isinstance(saved, dict) and isinstance(saved.get("answers"), dict):
        # A version change revalidates the saved JSON. It does not call a model.
        _event(call_id, "extraction_start", {"model": "saved"})
        result = validate_answers(call, saved)
        if (
            saved.get("version") != _VERSION
            or "yes_hindi" in saved
            or "questions_hindi" in saved
        ):
            db.save_answers(call_id, result)
        _event(call_id, "extraction_end", {"model": "saved", "error_class": "", "ms": _ms(started)})
        return result
    _event(call_id, "extraction_start", {"model": _GEMINI_MODEL})
    info = {"model": "findings", "error_class": ""}
    try:
        result, info = await _from_llm(call)
    except Exception as exc:
        info = {"model": "findings", "error_class": type(exc).__name__}
        _event(call_id, "extraction_fallback", {"model": _OPENAI_MODEL, "error_class": info["error_class"]})
        result = fallback_from_call(call)
    else:
        result = validate_answers(call, result)
    result["version"] = _VERSION
    db.save_answers(call_id, result)
    _event(
        call_id,
        "extraction_end",
        {"model": info.get("model") or "", "error_class": info.get("error_class") or "", "ms": _ms(started)},
    )
    return result
