"""Wide-question-first wording, drill-down follow-ups, and reply classifiers for SHISHU calls."""

import re

WIDE_WORDS: dict[str, str] = {
    "birth_mode": "bachche ka janm normal delivery se hua tha ya operation se",
    "birth_nicu": "bachche ko kabhi kaanch ke box ya bachchon ke ICU (NICU) mein rakhna pada tha",
    "birth_bcg": "BCG ke teeke ki jagah par kabhi kuch alag hua tha",
    "birth_hearing_vision": "janm ke baad bachche ke sunne aur dekhne ki jaanch hui thi",
    "body_fits_milestones": "bachche ko kabhi jhatke aaye, ya baithna, chalna, bolna doosre bachchon se der se seekha",
    "body_transfusion": "bachche ko kabhi khoon chadhana pada",
    "body_bleeding_bruise": "chot lagne par bachche ko zyada khoon behta hai, ya aasani se neel pad jaate hain",
    "body_puberty": "agar koi bachcha 10 saal ya usse bada hai, toh kya shareer mein badhti umar ke badlaav shuru ho gaye hain",
    "ill_admissions": "bachche ko baar baar hospital mein bharti karna pada",
    "ill_iv_antibiotics": "baar baar infection hua jiske liye drip ya nas mein dawai deni padi",
    "ill_fasting": "khaana chhootne par ya halki bimaari mein bachcha bahut sust ho jaata hai",
    "preg_conditions": "pregnancy ke dauraan aapko BP, sugar ya thyroid ki koi dikkat thi",
    "preg_thalassemia": "aapki ya aapke pati ki kabhi thalassemia ki khoon ki jaanch hui",
    "preg_scan": "pregnancy mein sonography ya scan hua tha",
    "preg_transfusion": "pregnancy mein kabhi khoon chadhana pada",
    "preg_delivery": "delivery ke samay koi dikkat aayi thi",
    "fam_related": "aap aur aapke pati shaadi se pehle rishtedaar the, jaise cousin ya ek hi khaandaan se",
    "fam_infant_death": "parivaar mein kisi chhote bachche ka ek saal se kam umar mein guzar jaana hua",
    "fam_bp_stroke": "parivaar mein 40-50 saal se kam umar ke kisi bade ya bachche ko BP, sugar, cholesterol ya lakwa hua",
    "fam_maternal_side": "aapke maike mein kisi bachche ya bade ko zyada khoon behna, baar baar infection, ya khujli wale sookhe dhabbe",
}

_WHO = "yeh kisko hua — aapka unse kya rishta hai?"

DRILLS: dict[str, list[str]] = {
    "birth_nicu": ["kis wajah se rakha tha — peeliya ki neeli light, infection, jhatke, ya kuch aur?", "kitne din rakha tha?"],
    "birth_bcg": ["kya hua tha — zyada sujan, phoda, ya bagal mein gaanth?"],
    "birth_hearing_vision": ["report normal aayi thi, ya koi dikkat batayi gayi?", "dikkat sunne mein thi ya dekhne mein?"],
    "body_fits_milestones": ["kaun si baat — jhatke, ya der se seekhna?", "yeh kab se hai?"],
    "body_tummy": ["yeh kab se hai, aur kisne bataya?"],
    "body_transfusion": ["kitni baar khoon chadhana pada?", "aakhri baar kab chadha tha?"],
    "body_blood_count": ["kam bataya tha ya zyada, aur kab?"],
    "body_joint_swelling": ["kaun se jod mein, aur kitne din mein theek hota hai?"],
    "body_bleeding_bruise": ["kya hota hai — khoon der tak behta hai, ya neel padte hain?", "kitni baar hua hai?"],
    "body_limb_pain": ["kitni baar hospital jaana pada?"],
    "body_puberty": ["yeh kaun se bachche mein shuru hua, jo 10 saal ya usse bada hai?"],
    "ill_admissions": ["kitni baar bharti karna pada, aur kis wajah se?"],
    "ill_iv_antibiotics": ["kitni baar, aur kaunsa infection tha, jaise pneumonia?"],
    "ill_sudden": ["kab hua tha, aur kya hua tha?"],
    "ill_fasting": ["kya hota hai — bahut neend, baar baar ulti, ya jhatke?", "kitni baar hua hai?"],
    "preg_conditions": ["kaun si dikkat thi — BP, sugar, ya thyroid?", "kya delivery ke baad bhi hai, ya theek ho gayi?"],
    "preg_thalassemia": ["kiski jaanch hui thi — aapki, pati ki, ya dono ki?", "report normal aayi thi, ya kuch bataya gaya?"],
    "preg_scan": ["scan normal tha, ya koi dikkat batayi gayi?"],
    "preg_heart": ["kya bataya gaya tha, aur kab?"],
    "preg_transfusion": ["kaun si pregnancy mein, aur kaun sa mahina tha?"],
    "preg_delivery": ["kya dikkat aayi thi?"],
    "fam_related": ["kya rishta tha, jaise cousin?"],
    "fam_miscarriage": ["kitni baar hua, aur kaun se mahine mein?"],
    "fam_infant_death": ["aapka us bachche se kya rishta tha?", "kis wajah se hua, agar pata ho?"],
    "fam_fits_delay": [_WHO, "kya hota hai — jhatke, der se seekhna, ya seekha hua bhoolna?"],
    "fam_blind": [_WHO],
    "fam_deaf": [_WHO],
    "fam_transfusion": [_WHO],
    "fam_bleeding": [_WHO],
    "fam_cancer": [_WHO],
    "fam_heart": [_WHO],
    "fam_kidney": [_WHO],
    "fam_bp_stroke": ["kaun si dikkat — BP, sugar, cholesterol, ya lakwa?", "yeh kisko hua — maata-pita, bhai-behen, ya door ke rishtedaar?", "unhe kitni umar mein hua tha?"],
    "fam_maternal_side": ["kaun si baat thi, aur kisko?"],
    "fam_lumps": [_WHO],
}

GROUP_BRIDGE_GROUPS: tuple[str, ...] = ("birth", "body", "illness", "pregnancy", "family")

SECTION_FIRST: dict[str, tuple[str, re.Pattern]] = {
    "birth_mode": ("birth", re.compile(r"operation|delivery|normal|ऑपरेशन|डिलीवरी|नॉर्मल", re.I)),
    "body_fits_milestones": ("body", re.compile(r"jhatke|baithna|chalna|bolna|झटके|बैठना|चलना|बोलना", re.I)),
    "ill_admissions": ("illness", re.compile(r"bharti|hospital|भर्ती|हॉस्पिटल|अस्पताल", re.I)),
    "preg_conditions": ("pregnancy", re.compile(r"\bBP\b|sugar|thyroid|बीपी|शुगर|थायरॉइड|थायराइड|ब्लड प्रेशर", re.I)),
    "fam_related": ("family", re.compile(r"rishte|cousin|khaandaan|khandan|shaadi se pehle|रिश्ते|कज़िन|कजिन|खानदान|शादी से पहले", re.I)),
}


def bot_asked(topic_id: str, bot_text: str) -> bool:
    entry = SECTION_FIRST.get(topic_id)
    if not entry:
        return True
    return bool(entry[1].search(bot_text or ""))


def section_note(topic_id: str) -> str:
    group = SECTION_FIRST[topic_id][0] if topic_id in SECTION_FIRST else ""
    words = WIDE_WORDS.get(topic_id, "")
    ask = f" ({words})" if words else ""
    return (
        "Your last turn did not actually ask this question (it was cut off or skipped), so her last reply "
        f"does NOT answer it. Say the {group.capitalize()} GROUP BRIDGE from your instructions in one short "
        f"sentence, then ask ONLY the {topic_id} question{ask} as one short Hindi question, then STOP and wait. "
        "Do not read this note aloud."
    )


def _norm(text: str) -> str:
    cleaned = re.sub(r"[?.!,।…\"'“”]+", " ", (text or "").strip().lower())
    return " ".join(cleaned.split())


_BARE_YES = {
    "haan", "han", "haa", "ha", "haan ji", "ji haan", "ji", "jee", "yes", "yes ji",
    "haan hua tha", "haan hui thi", "haan tha", "haan thi",
    "हाँ", "हां", "हा", "जी", "जी हाँ", "जी हां", "हाँ जी", "हां जी", "यस",
    "हाँ हुआ था", "हां हुआ था", "हाँ हुई थी", "हां हुई थी", "हाँ था", "हां था", "हाँ थी", "हां थी",
}


def is_bare_yes(text: str) -> bool:
    return _norm(text) in _BARE_YES


_CONFUSED = {
    "kya", "kya bola", "kya kaha", "kya bol rahe ho", "aap kya bol rahe ho", "matlab",
    "samajh nahi aaya", "samajh mein nahi aaya", "samjhi nahi", "samjha nahi",
    "phir se boliye", "phir se bolo", "dobara boliye", "dobara bolo", "what", "sorry", "pardon",
    "क्या", "क्या बोला", "क्या कहा", "क्या बोल रहे हो", "आप क्या बोल रहे हो", "आप क्या बोल रहे हैं",
    "मतलब", "समझ नहीं आया", "समझ में नहीं आया", "समझी नहीं", "समझा नहीं",
    "फिर से बोलिए", "फिर से बोलो", "दोबारा बोलिए", "दोबारा बोलो", "सॉरी", "व्हाट",
}


def is_confused_reply(text: str) -> bool:
    return _norm(text) in _CONFUSED


_EMERGENCY_RE = re.compile(
    r"saans\s+nahi\s+(?:le|aa)|saans\s+bahut\s+tez|not\s+breathing|can'?t\s+breathe|"
    r"behosh\s+(?:hai|ho\s+gaya|ho\s+gayi)|abhi\s+behosh|unconscious|not\s+responding|"
    r"jhatke\s+ruk\s+nahi|khoon\s+ruk\s+nahi|emergency\s+hai|"
    r"सांस\s+नहीं\s+(?:ले|आ)|साँस\s+नहीं\s+(?:ले|आ)|सांस\s+बहुत\s+तेज|साँस\s+बहुत\s+तेज|"
    r"बेहोश\s+(?:है|हो\s+गया|हो\s+गई)|होश\s+नहीं\s+आ|झटके\s+रुक\s+नहीं|खून\s+रुक\s+नहीं|इमरजेंसी\s+है",
    re.I,
)


def is_emergency_text(text: str) -> bool:
    return bool(_EMERGENCY_RE.search(text or ""))


def drill_note(topic_id: str) -> str:
    steps = DRILLS.get(topic_id) or []
    first = steps[0] if steps else "thoda aur bataiye"
    return (
        "She said only 'haan', so this topic is not finished. Do NOT move to the next topic. "
        f"Ask ONLY this one short follow-up in simple Hindi: '{first}'. Then STOP speaking and wait for her answer. "
        "Keep your same warm female voice (Kore). Do not read this note aloud."
    )


CONFUSED_NOTE = (
    "She did not understand your last question (she said something like 'kya?' or 'samajh nahi aaya'). "
    "Nothing was saved. Do NOT move on and do NOT answer for her. Say 'Koi baat nahi' and ask the SAME "
    "question again in simpler, shorter Hindi words, then STOP and wait. Do not read this note aloud."
)
