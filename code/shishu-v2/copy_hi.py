"""Every Hindi string SHISHU speaks, prints, or renders that is not catalog question text.

Live voice cues, closing lines, child labels, and the deterministic Mother report
builders live here once, so app code never re-spells a Hindi sentence.
"""

from collections.abc import Sequence

from models import AnswerStatus, CallerProfile, CallState, Child, ChildSex, CloseReason, HoldReason, MaritalStatus
from triage_catalog import QuestionSpec, spec_or_none

HOLD_HI: dict[HoldReason, str] = {
    HoldReason.NAME_FRAGMENT: "जी, आराम से बताइए — आपका पूरा नाम क्या है?",
    HoldReason.MARITAL_STATUS: "जी, क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?",
    HoldReason.CHILDREN_AGES: "जी, उन सभी बच्चों की उम्र क्या-क्या है?",
}
CLOSE_MORE_HI: str = "मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?"

GREETING_CUE: str = (
    "SYSTEM CUE (from the app, not the mother): speak only this Hindi, then wait. "
    "Do not call a tool yet.\n"
)
SILENCE_CHECK_IN_CUE: str = (
    "SYSTEM CUE (from the app, not the mother): she has been quiet for 15 seconds. "
    "Ask in Hindi if she can hear you, then repeat the same question in simpler words. "
    "Do not call a tool."
)
HINDI_MEDICAL_VOCAB: list[str] = [
    "SHISHU", "Namaste", "peeliya", "phototherapy", "NICU", "jhatke", "fits", "naal", "BCG",
    "taalu", "tilli", "spleen", "liver", "khoon", "thalassemia", "sonography", "garbh",
    "miscarriage", "consanguinity", "rishtedaar", "gurde", "kidney", "neel", "sujan", "gaanth",
    "ASHA", "ANM", "haan", "nahi", "haan ji", "nahi ji", "theek hai", "pata nahi", "yaad nahi",
    "mera naam", "matlab", "beta", "beti", "saal", "mahine", "normal", "operation",
]

GEMINI_LIVE_SYSTEM_PROMPT: str = """
You are SHISHU (शिशु), a warm Hindi voice for a mother before she meets a doctor. You are not a doctor. You collect history only.

VOICE
- One warm female voice for the whole call. Never switch to a male, deeper, or child's voice, even if a father is speaking.
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

GROUP_HI: dict[str, str] = {
    "birth": "जन्म और नवजात",
    "body": "बढ़त और शरीर",
    "illness": "बीमारी",
    "pregnancy": "प्रेग्नेंसी",
    "family": "परिवार",
    "close": "अंत",
}
MONTHS_HI: tuple[str, ...] = (
    "", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितम्बर", "अक्टूबर", "नवम्बर", "दिसम्बर",
)
ROUTINE_HI: tuple[str, ...] = (
    "क्या मेरे बच्चे का वज़न, कद और बैठना-चलना-बोलना उम्र के हिसाब से ठीक है?",
    "क्या टीके के कार्ड के सारे टीके पूरे हैं?",
    "घर पर खान-पान और किन चेतावनी की बातों पर ध्यान रखूँ?",
)
ROUTINE_CHILDLESS_HI: tuple[str, ...] = (
    "आगे प्रेग्नेंसी या स्वास्थ्य योजना से पहले मुझे कौन सी ज़रूरी जाँच करवानी चाहिए?",
    "परिवार की सेहत को देखते हुए डॉक्टर से किन बातों पर सलाह लेनी चाहिए?",
    "खान-पान, आयरन-फोलिक एसिड और किन बातों का अभी से ध्यान रखूँ?",
)
EMERGENCY_HI: tuple[str, ...] = (
    "इमरजेंसी डॉक्टर से अभी पूछूँ कि यह खतरे की निशानी कितनी गंभीर है "
    "और मुझे अस्पताल या 108 पर क्या करना है?",
    "अभी साँस और हालत को स्थिर करने के लिए डॉक्टर कौन सी जाँच तुरंत करेंगे?",
    "अस्पताल पहुँचकर डॉक्टर को कौन सी चेतावनी की बातों और पहले की बीमारी बतानी चाहिए?",
)
_SEX_HI: dict[ChildSex, str] = {ChildSex.MALE: "बेटा", ChildSex.FEMALE: "बेटी"}
_INCOMPLETE_HI = "अभी कुछ सवाल बाकी हैं, इसलिए यह जानकारी अधूरी है।"
_RESUME_HI = "जब समय मिले, दोबारा कॉल करके इसे पूरा कर लीजिए। नमस्ते।"


def age_label_hi(age_years: float) -> str:
    """'9 महीने' under one year, '5 साल' for whole years, otherwise the exact number."""
    if age_years < 1:
        return f"{int(round(age_years * 12))} महीने"
    return f"{int(age_years)} साल" if age_years == int(age_years) else f"{age_years} साल"


def _selected(profile: CallerProfile, child_ids: Sequence[str] | None) -> list[Child]:
    """All children in profile order, or the named ones oldest-first when ids are given."""
    if child_ids is None:
        return list(profile.children)
    wanted = {cid.strip() for cid in child_ids if cid.strip()}
    return [child for child in profile.ordered_children if child.id in wanted] if wanted else []


def children_label_hi(profile: CallerProfile, child_ids: Sequence[str] | None = None) -> str:
    """'आरव 5 साल बेटा · मीरा 2 साल बेटी', oldest first once child_ids are named."""
    labels = (
        " ".join(part for part in (child.name or "", child.age_label_hi, _SEX_HI.get(child.sex, "")) if part)
        for child in _selected(profile, child_ids)
    )
    return " · ".join(label for label in labels if label)


def children_label_en(profile: CallerProfile, child_ids: Sequence[str] | None = None) -> str:
    """Clinician-facing child roster: 'child_1 (male, age 5.0, Aarav)'."""
    return ", ".join(
        f"{child.id} ({child.sex}, age {child.age_years}" + (f", {child.name})" if child.name else ")")
        for child in _selected(profile, child_ids)
    )


def which_child_hi(children: Sequence[Child]) -> str:
    """Ask which child she means, oldest first, marking the middle child of three."""
    ordered = sorted(children, key=lambda child: child.age_years, reverse=True)
    middle = len(ordered) // 2 if len(ordered) == 3 else -1
    parts: list[str] = []
    for index, child in enumerate(ordered):
        label, name, female = child.age_label_hi, (child.name or "").strip(), child.sex is ChildSex.FEMALE
        if name:
            phrase = f"{label} {'वाली' if female else 'वाले'} {name}"
        else:
            phrase = label + (" वाली बेटी" if female else (" वाले बेटे" if child.sex is ChildSex.MALE else " वाले"))
        parts.append(phrase + (" (बीच वाली)" if female else " (बीच वाले)") if index == middle else phrase)
    listed = ", ".join(parts[:-1]) + ", या " + parts[-1] if len(parts) >= 2 else (parts[0] if parts else "")
    return f"यह किस बच्चे की बात है — {listed}?"


def question_hi_for(profile: CallerProfile, spec: QuestionSpec) -> str:
    """The one Hindi wording for this caller: marital variant first, then multi-child."""
    if profile.marital_status is MaritalStatus.ABOUT_TO_MARRY and spec.question_about_hi:
        return spec.question_about_hi
    if profile.marital_status is MaritalStatus.UNMARRIED and spec.question_unmarried_hi:
        return spec.question_unmarried_hi
    if len(profile.children) > 1 and spec.question_multi_hi:
        return spec.question_multi_hi
    return spec.question_hi


def closing_hi(state: CallState, *, has_clinical_concern: bool) -> str:
    """The exact farewell for this close reason, never claiming a diagnosis or all-clear."""
    name = (state.profile.caller_name or "").strip()
    who = f"{name} जी, " if name else ""
    reason = state.close_reason or CloseReason.COMPLETE
    if reason is CloseReason.EMERGENCY:
        return (
            f"{who}यह तुरंत दिखाने वाली बात है। अभी बच्चे को नज़दीकी अस्पताल ले जाइए, "
            "या 108 पर कॉल कीजिए। अब तक की जानकारी मैंने डॉक्टर के लिए नोट कर ली है। नमस्ते।"
        )
    if reason is CloseReason.CALLER_STOP:
        return (
            f"कोई बात नहीं, {who}अब तक की जानकारी मैंने नोट कर ली है। "
            "जब समय मिले, दोबारा कॉल करके बाकी सवाल पूरे कर लीजिए। नमस्ते।"
        )
    if reason is CloseReason.SILENCE:
        return (
            "लगता है आप अभी बात नहीं कर पा रही हैं। कोई बात नहीं। "
            f"अब तक आपने जो समय दिया, उसके लिए बहुत धन्यवाद। {_INCOMPLETE_HI} {_RESUME_HI}"
        )
    if reason is CloseReason.TIME_LIMIT:
        return (
            f"{who}आज हमारी बात काफी लंबी हो गई। इतना समय देने के लिए बहुत धन्यवाद। "
            f"{_INCOMPLETE_HI} {_RESUME_HI}"
        )
    if reason is CloseReason.TECHNICAL_FAILURE:
        return (
            f"माफ़ कीजिए, {who}अभी कुछ तकनीकी दिक्कत आ गई है। "
            f"अब तक की जानकारी मैंने नोट कर ली है। {_INCOMPLETE_HI} {_RESUME_HI}"
        )
    if has_clinical_concern:
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


def speak_cue_text(prompt_hi: str, *, closing: bool = False, clarify: bool = False) -> str:
    """The SYSTEM CUE that makes the live model speak now instead of waiting out a watchdog."""
    if closing:
        return (
            "SYSTEM CUE (from the app, not the mother): speak only this Hindi, then stop. "
            "Do not call a tool.\n" + prompt_hi
        )
    if clarify:
        return (
            "SYSTEM CUE (from the app, not the mother): she did not understand. "
            "Say one short simpler Hindi sentence that explains the question, "
            "then ask the Hindi below in simpler words. Do not repeat it with the same words. "
            "Then wait. Do not call a tool.\n" + prompt_hi
        )
    return (
        "SYSTEM CUE (from the app, not the mother): speak only this Hindi now, then wait. "
        "Do not call a tool.\n" + prompt_hi
    )


def mother_concern_question_hi(spec: QuestionSpec, mother_note: str, child_scope_hi: str) -> str:
    """One Mother-PDF question for a concern, scoped to the child she named."""
    prefix = f"{child_scope_hi} के लिए: " if (child_scope_hi and child_scope_hi not in mother_note) else ""
    if mother_note:
        return f"{prefix}मैंने बताया कि {mother_note}। डॉक्टर, इसकी जाँच क्या होनी चाहिए?"
    if spec.is_open_capture:
        return "मैंने कॉल के अंत में एक और बात दर्ज करवाई थी। डॉक्टर, उस पर क्या सलाह है?"
    head = f"{spec.stem_hi} ({child_scope_hi})" if child_scope_hi else spec.stem_hi
    return f"मैंने हाँ कहा कि {head}। डॉक्टर, अब क्या जाँच करनी चाहिए?"


def mother_yes_summary_hi(spec: QuestionSpec, mother_note: str, child_scope_hi: str) -> str:
    """One line for the Mother PDF's 'what was a हाँ' list, with no diagnosis."""
    if spec.is_open_capture:
        return f"अन्य बात — {mother_note}।" if mother_note else "डॉक्टर को बताने के लिए एक अन्य बात दर्ज की गई।"
    head = f"{spec.stem_hi} ({child_scope_hi})" if child_scope_hi else spec.stem_hi
    return f"{head} — {mother_note}।" if mother_note else f"{head}।"


def fit_mother_questions_hi(
    questions: Sequence[str], *, emergency: bool, has_children: bool
) -> list[str]:
    """Keep 3 to 6 unique questions, topping up with emergency or routine Hindi fillers."""
    routine = ROUTINE_HI if has_children else ROUTINE_CHILDLESS_HI
    if emergency and not questions:
        return list(EMERGENCY_HI)
    seen: set[str] = set()
    kept: list[str] = []
    for question in questions:
        clean = " ".join(question.split())
        if clean and clean not in seen and not (emergency and clean in (*ROUTINE_HI, *ROUTINE_CHILDLESS_HI)):
            seen.add(clean)
            kept.append(clean)
    kept = kept[:6]
    for filler in EMERGENCY_HI if emergency else routine:
        if len(kept) >= 3:
            break
        if filler not in seen:
            seen.add(filler)
            kept.append(filler)
    return kept[:6]


def recorded_concerns_hi(state: CallState) -> list[str]:
    """Non-diagnostic Hindi lines for saved clinical concerns, safe for voice recall."""
    lines: list[str] = []
    for qid, record in state.answers.items():
        spec = spec_or_none(qid)
        if spec is None or not spec.is_reportable or record.state is not AnswerStatus.CONCERN:
            continue
        bits = [
            bit
            for bit in (
                spec.question_hi.strip(),
                children_label_hi(state.profile, record.child_ids),
                record.mother_safe_text,
            )
            if bit
        ]
        if bits:
            lines.append(" — ".join(bits))
    return lines

