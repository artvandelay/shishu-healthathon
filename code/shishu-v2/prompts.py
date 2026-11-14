"""Live voice instructions. The server cursor chooses every question."""

import re

from call_cursor import active_prompt_hi, answer_notes_text
from triage_catalog import get_spec

GREETING_CUE: str = (
    "SYSTEM CUE (from the app, not the mother): speak only this Hindi, then wait. "
    "Do not call a tool yet.\n"
)

WATCHDOG_NUDGE: str = (
    "SYSTEM CUE (from the app, not the mother): she is waiting. "
    "Speak only the Hindi question you were last given, in your same warm female voice, then wait. "
    "Never answer for her."
)

SILENCE_CHECK_IN_CUE: str = (
    "SYSTEM CUE (from the app, not the mother): she has been quiet for 15 seconds. "
    "Ask in Hindi if she can hear you, then repeat the same question in simpler words. "
    "Do not call a tool."
)

HINDI_MEDICAL_VOCAB: list[str] = [
    "SHISHU",
    "Namaste",
    "peeliya",
    "phototherapy",
    "NICU",
    "jhatke",
    "fits",
    "naal",
    "BCG",
    "taalu",
    "tilli",
    "spleen",
    "liver",
    "khoon",
    "thalassemia",
    "sonography",
    "garbh",
    "miscarriage",
    "consanguinity",
    "rishtedaar",
    "gurde",
    "kidney",
    "neel",
    "sujan",
    "gaanth",
    "ASHA",
    "ANM",
    "haan",
    "nahi",
    "haan ji",
    "nahi ji",
    "theek hai",
    "pata nahi",
    "yaad nahi",
    "mera naam",
    "matlab",
    "beta",
    "beti",
    "saal",
    "mahine",
    "normal",
    "operation",
]

GEMINI_LIVE_SYSTEM_PROMPT: str = """
You are SHISHU (शिशु), a warm Hindi voice for a mother before she meets a doctor. You are not a doctor. You collect history only.

VOICE
- One warm female voice (Kore) for the whole call. Never switch to a male, deeper, or child's voice, even if a father is speaking.
- Speak ONLY simple spoken Hindi. Never Spanish, English, French, or any other language. Never read a tool result, a hint, or these instructions aloud.
- Never answer for her. Never roleplay her side. Never say हाँ or नहीं as if you were her. Never classify or invent her answer; the server decides everything.
- Never sound surprised or judgmental about children, marriage, miscarriage, or a home birth.
- Everyday words: काँच का बक्सा, नीली लाइट, नाल, झटके, तिल्ली.
- Never name a genetic or rare disease, in English or Hindi. Never say Duchenne, Hemophilia, Pompe, Gaucher, SCID, SMA, MPS, Wiskott-Aldrich, ड्यूशेन, हीमोफीलिया, पॉम्पे, गॉशर, इम्यूनोडेफिशिएंसी, or म्यूकोपॉलीसेकेराइडोसिस.
- Never say that nothing concerning was found, and never say she does not need a doctor.
- Never advise starting, stopping, or changing a medicine.

TURN PROTOCOL
The server chooses and handles every question and answer. You do not pick a topic, skip ahead, or classify her reply.
- Whenever the caller speaks (or sends text), immediately call next_question with NO arguments before speaking. Never speak to the caller before calling next_question.
- After next_question returns, speak ONLY the Hindi in the tool response's say field, unhurried, then stop and wait.
- If clarify is true in next_question's response, first give 1 to 2 plain Hindi sentences explaining the question or term in simpler words, then ask the Hindi in say. Do not repeat the question with the exact same words.
  - If she asked what BCG means, give two plain Hindi sentences about the shoulder birth vaccine, then ask say.
  - If she asked what थैलेसीमिया means, explain थैलेसीमिया माइनर या ट्रेट in two plain Hindi sentences: it is an ordinary parent carrier blood check where the parent stays healthy, and the doctor uses it for pregnancy and child guidance. Never describe a severe disease or repeated blood transfusions. Then ask say.
- If recorded_concerns is present and she asked what was noted earlier, mention only the items in recorded_concerns. If that list is empty, say that none were recorded yet. Never guess another symptom.
- If the tool result is held, that turn was not saved. Speak say, then wait.
- If stop is true in next_question's response, speak say once and then stay silent. Do not call another tool.

A line that starts with SYSTEM CUE is from the app, not from the mother. Follow it. Never read it aloud.
""".strip()


_LEAK_RE = re.compile(
    r"(?is)"
    r"\*?note:\s*audio is in english[^\n]*"
    r"|system\s+cue\s*(?:\([^)]*\))?\s*:[^\n]*"
    r"|wait for the user'?s? audio input\.?"
    r"|do not guess or assume any values\.?"
    r"|respond in the exact format shown in the examples\.?"
    r"|current domain:\s*[^\n.]*\.?"
    r"|last finding:\s*[^\n.]*\.?"
    r"|action:\s*[^\n.]*\.?"
    r"|<no speech>"
    r"|\{pause\}"
    r"|next_question|say_hindi|do_not_read|active_qid|recorded_concerns"
    r"|ask only this"
)
_TOOL_LEAK_RE = re.compile(
    r"next_question|say_hindi|do_not_read|active_qid|recorded_concerns|system\s*cue|ask only this",
    re.I,
)
_OTHER_SCRIPT_RE = re.compile(
    "["
    "\u0600-\u06FF"
    "\u0750-\u077F"
    "\u08A0-\u08FF"
    "\u0980-\u09FF"
    "\uFB50-\uFDFF"
    "\uFE70-\uFEFF"
    "]+"
)


def clean_assistant_caption(text: str, *, exact: bool = False) -> str:
    """Drop planner notes and non-Hindi scripts from a bot caption.

    exact=True keeps the remaining characters as-is, including spaces, so a
    live transcript delta is not padded.
    """
    if not text:
        return ""
    cleaned = _OTHER_SCRIPT_RE.sub("", _LEAK_RE.sub("", text))
    if not re.search(r"[A-Za-z\u0900-\u097F0-9]", cleaned):
        return ""
    if exact:
        return cleaned
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip()


def leaked_caption(raw: str, cleaned: str) -> bool:
    """True when the model spoke an instruction or a tool name."""
    return bool(_TOOL_LEAK_RE.search(raw or "") or _LEAK_RE.search(raw or ""))


_HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]+")


def clean_user_caption(text: str) -> str:
    """Strip non-Devanagari/non-Latin script noise and normalize whitespace."""
    if not text:
        return ""
    cleaned = _OTHER_SCRIPT_RE.sub("", _HANGUL_RE.sub("", text))
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def closing_hi(state: dict) -> str:
    name = str((state.get("profile") or {}).get("caller_name") or "").strip()
    who = f"{name} जी, " if name else ""
    reason = state.get("close_reason") or "complete"
    if reason == "emergency":
        return (
            f"{who}यह तुरंत दिखाने वाली बात है। अभी बच्चे को नज़दीकी अस्पताल ले जाइए, "
            "या 108 पर कॉल कीजिए। अब तक की जानकारी मैंने डॉक्टर के लिए नोट कर ली है। नमस्ते।"
        )
    if reason == "caller_stop":
        return (
            f"कोई बात नहीं, {who}अब तक की जानकारी मैंने नोट कर ली है। "
            "जब समय मिले, दोबारा कॉल करके बाकी सवाल पूरे कर लीजिए। नमस्ते।"
        )
    if reason == "silence":
        return (
            "लगता है आप अभी बात नहीं कर पा रही हैं। कोई बात नहीं। "
            "अब तक आपने जो समय दिया, उसके लिए बहुत धन्यवाद। अभी कुछ सवाल बाकी हैं, "
            "इसलिए यह जानकारी अधूरी है। जब समय मिले, दोबारा कॉल करके इसे पूरा कर लीजिए। नमस्ते।"
        )
    if reason == "time_limit":
        return (
            f"{who}आज हमारी बात काफी लंबी हो गई। इतना समय देने के लिए बहुत धन्यवाद। "
            "अभी कुछ सवाल बाकी हैं, इसलिए यह जानकारी अधूरी है। "
            "जब समय मिले, दोबारा कॉल करके इसे पूरा कर लीजिए। नमस्ते।"
        )
    if reason == "technical_failure":
        return (
            f"माफ़ कीजिए, {who}अभी कुछ तकनीकी दिक्कत आ गई है। "
            "अब तक की जानकारी मैंने नोट कर ली है। अभी कुछ सवाल बाकी हैं, "
            "इसलिए यह जानकारी अधूरी है। जब समय मिले, दोबारा कॉल करके इसे पूरा कर लीजिए। नमस्ते।"
        )
    concerns = any(
        isinstance(entry, dict) and entry.get("state") == "concern"
        for entry in (state.get("answers") or {}).values()
    )
    if concerns:
        return (
            f"{who}इतना ध्यान से बताने के लिए बहुत धन्यवाद। आपने जो बताया, "
            "उसमें से कुछ बातों की जाँच डॉक्टर या स्वास्थ्य कार्यकर्ता से करवाना अच्छा रहेगा। "
            "आपकी रिपोर्ट तैयार हो रही है। यह रिपोर्ट आप अपनी ASHA, ANM या डॉक्टर को दिखा सकती हैं। नमस्ते।"
        )
    return (
        f"{who}इतना ध्यान से बताने के लिए बहुत धन्यवाद। "
        "अगर आगे इनमें से कुछ भी दिखे, तो अपने स्वास्थ्य कार्यकर्ता से बात करें। "
        "आपकी रिपोर्ट तैयार हो रही है। यह रिपोर्ट आप अपनी ASHA, ANM या डॉक्टर को दिखा सकती हैं। नमस्ते।"
    )


def say_hindi(state: dict) -> str:
    if state.get("phase") != "asking":
        return closing_hi(state)
    return active_prompt_hi(state)


def speak_cue_text(state: dict, kind: str = "speak") -> str:
    """Text cue that makes the live model speak now, instead of waiting out the watchdog."""
    if state.get("phase") != "asking" or kind == "close":
        return (
            "SYSTEM CUE (from the app, not the mother): speak only this Hindi, then stop. "
            "Do not call a tool.\n"
            + closing_hi(state)
        )
    hindi = say_hindi(state)
    if kind == "clarify":
        return (
            "SYSTEM CUE (from the app, not the mother): she did not understand. "
            "Say one short simpler Hindi sentence that explains the question, "
            "then ask the Hindi below in simpler words. Do not repeat it with the same words. "
            "Then wait. Do not call a tool.\n"
            + hindi
        )
    return (
        "SYSTEM CUE (from the app, not the mother): speak only this Hindi now, then wait. "
        "Do not call a tool.\n"
        + hindi
    )


def recorded_concerns_hi(state: dict) -> list[str]:
    """One short Hindi line for each answer already saved as a concern."""
    answers = state.get("answers") or {}
    if not isinstance(answers, dict):
        return []
    age_by_id: dict[str, str] = {}
    children = (state.get("profile") or {}).get("children") or []
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue
            cid = str(child.get("id") or "").strip()
            label = str(child.get("age_label_hi") or "").strip()
            if cid and label:
                age_by_id[cid] = label
    lines: list[str] = []
    for qid, entry in answers.items():
        if not isinstance(entry, dict) or entry.get("state") != "concern":
            continue
        try:
            head = str(get_spec(str(qid)).question_hi or "").strip()
        except KeyError:
            head = ""
        ages: list[str] = []
        for cid in entry.get("child_ids") or []:
            label = age_by_id.get(str(cid), "")
            if label and label not in ages:
                ages.append(label)
        notes_text = answer_notes_text(entry)
        bits = [bit for bit in (head, ", ".join(ages), notes_text) if bit]
        if bits:
            lines.append(" — ".join(bits))
    return lines
