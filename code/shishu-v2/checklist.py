"""Triage topic checklist and per-call coverage helpers for SHISHU calls."""

import re

GROUPS: tuple[str, ...] = ("intro", "birth", "body", "illness", "pregnancy", "family", "close")

TOPICS: tuple[dict, ...] = (
    {"id": "intro_name", "group": "intro", "need": "mother's name", "words": "aapka naam"},
    {"id": "intro_married", "group": "intro", "need": "married, or about to marry", "words": "shaadi ho gayi hai, ya hone wali hai"},
    {"id": "intro_children", "group": "intro", "need": "how many children and their ages", "words": "kitne bachche hain, umar kitni hai"},
    {"id": "intro_bridge", "group": "intro", "need": "explain the call; she may answer short or long and ask anything", "words": "chhota ya lamba jawab, bejhijhak poochh sakti hain"},
    {"id": "birth_mode", "group": "birth", "need": "normal delivery or operation; home or hospital; why the operation", "words": "normal delivery ya operation, ghar pe ya hospital"},
    {"id": "birth_term", "group": "birth", "need": "full 9 months, or early or late, and by how much", "words": "poore 9 mahine, samay se pehle ya baad"},
    {"id": "birth_nicu", "group": "birth", "need": "NICU stay and the reason: jaundice light, infection, fits, other", "words": "kaanch ka box, bachchon ka ICU, peeliya ki neeli light, jhatke"},
    {"id": "birth_cord_bleed", "group": "birth", "need": "heavy bleeding when the umbilical cord fell", "words": "naal girne par zyada khoon"},
    {"id": "birth_cord_3weeks", "group": "birth", "need": "cord took more than 3 weeks to fall", "words": "naal girne mein 3 hafte se zyada"},
    {"id": "birth_bcg", "group": "birth", "need": "unusual BCG site reaction: big swelling, boil, or armpit lump", "words": "BCG teeka, kandhe pe nishaan, sujan, phoda, bagal mein gaanth"},
    {"id": "birth_hearing_vision", "group": "birth", "need": "newborn hearing and vision test, and was it normal", "words": "sunne aur dekhne ki jaanch"},
    {"id": "birth_cleft", "group": "birth", "need": "cleft lip or palate", "words": "hoth ya taalu kata hua, munh ki chhat"},
    {"id": "body_fits_milestones", "group": "body", "need": "fits; sitting, walking, talking later than other children", "words": "jhatke, baithna chalna bolna der se"},
    {"id": "body_regression", "group": "body", "need": "lost a skill the child already had", "words": "pehle kar leta tha, ab nahi kar paata"},
    {"id": "body_blue", "group": "body", "need": "blue lips, fingers or toes", "words": "hoth ya ungliyan neeli"},
    {"id": "body_tummy", "group": "body", "need": "tummy slowly getting bigger, or doctor said liver or spleen is enlarged", "words": "pet dheere dheere bada, liver ya tilli badhna"},
    {"id": "body_transfusion", "group": "body", "need": "child ever needed blood, and how many times", "words": "khoon chadhana pada, kitni baar"},
    {"id": "body_blood_count", "group": "body", "need": "doctor said the blood count is low or high", "words": "blood kam ya zyada"},
    {"id": "body_joint_swelling", "group": "body", "need": "painful knee or joint swelling after walking or play that settles in days or weeks", "words": "ghutno ya jodon mein sujan, dard, apne aap theek"},
    {"id": "body_bleeding_bruise", "group": "body", "need": "heavy bleeding after minor cuts or injury, or easy bruising; if girl 11+: heavy periods; if child 5+: tooth fall bleeding", "words": "chot lagne par zyada khoon behna, ya aasani se neel padna"},
    {"id": "body_private_parts", "group": "body", "need": "private parts do not look as they should", "words": "gupt ang waise nahi dikhte"},
    {"id": "body_lumps_birth", "group": "body", "need": "swellings or lumps on the body since birth", "words": "janm se sharir par sujan ya gaanthein"},
    {"id": "body_proportion", "group": "body", "need": "body parts out of proportion, arms or legs very short", "words": "haath pair sharir ke mukable bahut chhote"},
    {"id": "body_thumb", "group": "body", "need": "small or missing thumb or finger", "words": "angootha ya ungli chhoti ya nahi hai"},
    {"id": "body_growth", "group": "body", "need": "growing like other children", "words": "baaki bachchon jaisa badh raha hai"},
    {"id": "body_limb_pain", "group": "body", "need": "repeated joint or limb pain that needs hospital", "words": "haath pair mein baar baar dard, hospital jaana"},
    {"id": "body_silver_hair", "group": "body", "need": "silvery hair", "words": "baal chaandi jaise"},
    {"id": "body_kidney", "group": "body", "need": "any kidney disease", "words": "kidney ya gurde ki bimaari"},
    {"id": "body_puberty_early", "group": "body", "need": "only if a girl over 12 or a boy over 14: puberty changes came earlier than usual", "words": "badhti umar ke badlaav jaldi", "optional": True},
    {"id": "body_puberty_late", "group": "body", "need": "only if a girl over 12 or a boy over 14: puberty changes have not started", "words": "badhti umar ke badlaav abhi tak nahi", "optional": True},
    {"id": "ill_admissions", "group": "illness", "need": "admitted to hospital again and again", "words": "baar baar hospital mein bharti"},
    {"id": "ill_iv_antibiotics", "group": "illness", "need": "repeated infections that needed drip or IV antibiotics", "words": "baar baar infection, drip ya nas mein dawai"},
    {"id": "ill_odor", "group": "illness", "need": "urine or stool smells strange, unlike other children", "words": "peshaab ya potty mein ajeeb badboo"},
    {"id": "ill_sudden", "group": "illness", "need": "fine at birth, then suddenly very ill and taken to emergency", "words": "achaanak bahut beemaar, emergency"},
    {"id": "ill_fasting", "group": "illness", "need": "very sleepy, repeated vomiting or fits when ill, after a missed meal, or after certain foods", "words": "khaana chhootne par sust, neend, ulti, jhatke"},
    {"id": "preg_conditions", "group": "pregnancy", "need": "high BP, sugar or thyroid in pregnancy, and still present after delivery", "words": "pregnancy mein BP, sugar, thyroid, delivery ke baad bhi"},
    {"id": "preg_thalassemia", "group": "pregnancy", "need": "thalassemia test for mother and father, and the result", "words": "thalassemia ki jaanch, maa aur pita dono, result"},
    {"id": "preg_scan", "group": "pregnancy", "need": "pregnancy scan done, and was it normal", "words": "sonography, scan normal tha"},
    {"id": "preg_heart", "group": "pregnancy", "need": "doctor found a heart problem on the scan or at birth", "words": "scan mein ya janm par dil mein kharaabi"},
    {"id": "preg_transfusion", "group": "pregnancy", "need": "blood transfusion in pregnancy, which pregnancy and which month", "words": "pregnancy mein khoon chadhana, kaunsa mahina"},
    {"id": "preg_delivery", "group": "pregnancy", "need": "delivery went without problems", "words": "delivery bina problem ke"},
    {"id": "fam_related", "group": "family", "need": "she and her husband were related before marriage", "words": "shaadi se pehle rishtedaar, cousin, ek hi khaandaan"},
    {"id": "fam_miscarriage", "group": "family", "need": "miscarriage or abortion", "words": "aapka kabhi garbh gira, ya garbh girwaana pada"},
    {"id": "fam_infant_death", "group": "family", "need": "a baby/child in the family died before 1 year, and why", "words": "parivaar mein kisi chhote bachche ka ek saal se kam umar mein guzar jaana"},
    {"id": "fam_fits_delay", "group": "family", "need": "any child or adult in the family has fits, late milestones, or lost skills", "words": "parivaar mein kisi bachche ya bade ko jhatke, der se seekhna, ya seekha hua bhool jaana"},
    {"id": "fam_blind", "group": "family", "need": "blindness in any child or adult in the family (not from old age)", "words": "parivaar mein kisi bachche ya bade ko bachpan ya kam umar se andhapan"},
    {"id": "fam_deaf", "group": "family", "need": "poor or no hearing in any child or adult in the family (not from old age)", "words": "parivaar mein kisi bachche ya bade ko kam umar se kam sunaai dena ya bilkul na sunaai dena"},
    {"id": "fam_transfusion", "group": "family", "need": "any child or adult in the family needs blood transfusions", "words": "parivaar mein kisi bachche ya bade ko baar baar khoon chadhana"},
    {"id": "fam_bleeding", "group": "family", "need": "any child or adult in the family bleeds too much", "words": "parivaar mein kisi bachche ya bade ko zyada khoon behne ki problem"},
    {"id": "fam_cancer", "group": "family", "need": "cancer in any child or adult in the family", "words": "parivaar mein kisi bachche ya bade ko cancer"},
    {"id": "fam_heart", "group": "family", "need": "heart disease in any child or adult in the family", "words": "parivaar mein kisi bachche ya bade ko dil ki bimaari"},
    {"id": "fam_kidney", "group": "family", "need": "kidney disease in any child or adult in the family", "words": "parivaar mein kisi bachche ya bade ko kidney ya gurde ki bimaari"},
    {"id": "fam_bp_stroke", "group": "family", "need": "high BP, sugar, high cholesterol or stroke in any family member under 40-50 years of age; if yes, MUST ask WHO in the family (immediate relative like mother/father/brother/sister vs distant relative) and at what age", "words": "parivaar mein 40-50 saal se kam umar ke kisi bade ya bachche ko BP, sugar, cholesterol, ya lakwa (agar haan, toh zaroor poochhein kisko — apना/सगा रिश्तेदार जैसे माता-पिता/भाई-बहन या दूर के रिश्तेदार)"},
    {"id": "fam_maternal_side", "group": "family", "need": "on the mother's side (maike mein kisi bachche ya bade ko): heavy bleeding, repeated infections, or dry itchy skin patches", "words": "maa ki taraf (maike mein) kisi bachche ya bade ko zyada khoon, baar baar infection, ya khujli wale sookhe dhabbe"},
    {"id": "fam_lumps", "group": "family", "need": "any child or adult in the family has swellings or lumps on the body", "words": "parivaar mein kisi bachche ya bade ke badan pe sujan ya gaanthein"},
    {"id": "fam_same_problem", "group": "family", "need": "only if any family answer was yes: someone has the same problem as her child", "words": "parivaar mein kisi bachche ya bade ko aapke bachche jaisi same problem", "optional": True},
    {"id": "close_anything_else", "group": "close", "need": "anything else she wants to tell the doctor", "words": "iske alawa kuch aur batana hai"},
)

_BY_ID: dict[str, dict] = {t["id"]: t for t in TOPICS}
REQUIRED_IDS: tuple[str, ...] = tuple(t["id"] for t in TOPICS if not t.get("optional"))


def mark(coverage: list, topic_ids) -> list[str]:
    if isinstance(topic_ids, str):
        topic_ids = [topic_ids]
    added: list[str] = []
    for raw in topic_ids or []:
        key = str(raw).strip()
        if key in GROUPS:
            ids = [t["id"] for t in TOPICS if t["group"] == key]
        elif key in _BY_ID:
            ids = [key]
        else:
            ids = []
        for tid in ids:
            if tid not in coverage:
                coverage.append(tid)
                added.append(tid)
    return added


def open_topics(coverage: list, include_optional: bool = False) -> list[dict]:
    return [
        t
        for t in TOPICS
        if t["id"] not in coverage and (include_optional or not t.get("optional"))
    ]


def current_group(coverage: list) -> str:
    left = open_topics(coverage)
    return left[0]["group"] if left else "done"


def progress_label(coverage: list) -> str:
    done = sum(1 for tid in REQUIRED_IDS if tid in coverage)
    return f"{current_group(coverage)} {done}/{len(REQUIRED_IDS)}"


def tool_note(coverage: list) -> str:
    left = open_topics(coverage)
    if not left:
        return (
            "All topics are covered. Speak the full closing now in your same warm female voice (Kore) in Hindi, "
            "then call end_call. Do not read this note aloud."
        )
    t = left[0]
    nxt = f"{t['id']} (Hindi: {t['words']} — {t['need']})"
    if t["id"] == "body_bleeding_bruise":
        nxt += " [NOTE: Ask about cuts/scratches and easy bruising. NEVER ask about periods unless she specifically stated she has an older daughter aged 11+]"
    return (
        f"Saved. Next open topic: {nxt}. "
        "CRITICAL: If her last answer was 'Haan' or positive, or if you ALREADY started asking a follow-up question in this turn "
        "(such as asking the children's ages, how many weeks early, NICU stay, cause of death, or who in the family had cancer/BP), "
        "STOP SPEAKING IMMEDIATELY and WAIT for her answer! Do NOT say 'Theek hai' and move on to the next open topic in the same turn. "
        "Otherwise, if no follow-up is pending, ask ONLY this single open topic as ONE short Hindi question (never combine two questions in one turn, and never speak English words from this note). "
        "NEVER say 'Nahi' or 'Haan' at the end of your question. "
        "Keep your exact same warm female voice (Kore) and pitch — never shift to a male voice. "
        "If she has more than one child, say 'aapke kisi bhi bachche ko'; in family topics, explicitly say 'kisi bachche ya bade ko'. "
        "Do not read this note aloud."
    )


def can_end(coverage: list) -> tuple[bool, str]:
    left = open_topics(coverage)
    if not left:
        return True, ""
    names = ", ".join(t["id"] for t in left[:5])
    return False, (
        f"Do not end the call yet. Still open: {names}. Say 'Bas ek-do sawal aur.' "
        "and ask the next open topic in Hindi. Do not read this note aloud."
    )


def render_topic_list() -> str:
    lines: list[str] = []
    for group in GROUPS:
        lines.append(group.upper())
        for t in TOPICS:
            if t["group"] != group:
                continue
            extra = " (optional)" if t.get("optional") else ""
            lines.append(f"- {t['id']}{extra}: {t['need']}. Her words: {t['words']}.")
    return "\n".join(lines)


CHECK_IN_SEC = 15
SILENCE_CLOSE_SEC = 30
SOFT_LIMIT_SEC = 19 * 60
HARD_LIMIT_SEC = 20 * 60
FORCE_END_GRACE_SEC = 25

CHECK_IN_CUE = (
    "SYSTEM CUE (from the app, not the mother): she has been quiet for 15 seconds. "
    "Gently ask in Hindi if she can hear you, then repeat your last question in simpler words."
)
SILENCE_CLOSE_CUE = (
    "SYSTEM CUE (from the app, not the mother): she has been silent for 30 seconds. "
    "Speak the INCOMPLETE CLOSING for silence now, in Hindi, then call end_call."
)
TIME_CLOSE_CUE = (
    "SYSTEM CUE (from the app, not the mother): the call has reached its time limit. "
    "Speak the INCOMPLETE CLOSING for time now, in Hindi, then call end_call."
)

_CLOSE_REASONS = {
    "silence": "the mother was silent for 30 seconds",
    "time_limit": "the 20-minute limit was reached",
}


def incomplete_notes(coverage: list, reason: str) -> tuple[str, str]:
    left = open_topics(coverage)
    if not left:
        return "", ""
    why = _CLOSE_REASONS.get(reason, "the call ended early")
    ids = ", ".join(t["id"] for t in left)
    doctor = (
        f"> **Incomplete call:** {why}. {len(left)} of {len(REQUIRED_IDS)} topics were not asked: {ids}."
    )
    mother = "यह कॉल अधूरी रही। कृपया जब समय मिले, दोबारा कॉल (dobara call) करके बाकी सवाल पूरे कर लीजिए।"
    return doctor, mother


NAME_MISSING_NOTE = (
    "Mother's name was not heard yet (she only said an incomplete fragment like 'Mera naam' or paused). "
    "Do NOT move on. Gently ask in Hindi: 'Ji, aaram se bataiye, aapka poora naam kya hai?' "
    "Do not read this note aloud."
)
CHILDREN_AGES_MISSING_NOTE = (
    "She stated how many children she has, but has NOT given their ages yet. "
    "Do NOT say the intro bridge and do NOT move on yet! "
    "If you already asked 'Un sabhi bachchon ki umar kya-kya hai?' in this turn, STOP SPEAKING NOW and wait for her answer. "
    "Otherwise ask ONLY: 'Ji, bahut achha. Un sabhi bachchon ki umar kya-kya hai?' and wait for her reply. "
    "Do not read this note aloud."
)
PRETERM_NICU_FOLLOWUP_NOTE = (
    "She reported a preterm or post-term birth, so we still need to know by how many weeks/months AND whether any child needed NICU ('kaanch ka box') or blue light for jaundice. "
    "If you already asked that follow-up in this turn, STOP SPEAKING NOW and wait for her answer. "
    "Otherwise ask how early/late the baby was born and whether the baby needed 'kaanch ka box' (NICU), then wait for her reply. "
    "Do not read this note aloud."
)

_CHILD_AGE_PRESENT_RE = re.compile(
    r"\b(year|years|yr|yrs|month|months|mo|mos|day|days|week|weeks|saal|mahin\w*|haft\w*|din|"
    r"no\s+child|no\s+children|childless|0\s+children|zero\s+children|none)\b"
    r"|साल|महीने|महीना|हफ्ते|हफ्ता|दिन|वरस|बच्चे\s+नहीं|कोई\s+बच्चा\s+नहीं",
    re.I,
)


def has_child_age_detail(detail: str, last_user_text: str = "") -> bool:
    blob = f"{detail or ''} {last_user_text or ''}"
    return bool(_CHILD_AGE_PRESENT_RE.search(blob))

_MISSING_NAME_DETAIL_RE = re.compile(
    r"\b(not provided|not given|not stated|not mentioned|unknown|unspecified|"
    r"did not|didn't|no name|incomplete|fragment|only said|unclear|pending)\b",
    re.I,
)
_BARE_NAME_PHRASES = {
    "मेरा नाम",
    "मेरा नाम है",
    "नाम",
    "जी मेरा नाम",
    "हाँ मेरा नाम",
    "हां मेरा नाम",
    "mera naam",
    "mera naam hai",
    "naam",
    "my name",
    "my name is",
}


def is_valid_name_detail(detail: str, last_user_text: str = "") -> bool:
    d = (detail or "").strip()
    if not d or _MISSING_NAME_DETAIL_RE.search(d):
        return False
    cleaned_user = re.sub(r"[.,!?;:।\-…]+", " ", (last_user_text or "").strip().lower())
    cleaned_user = " ".join(cleaned_user.split())
    if cleaned_user in _BARE_NAME_PHRASES:
        return False
    if d.lower() in _BARE_NAME_PHRASES:
        return False
    return True


GROUP_HI: dict[str, str] = {
    "intro": "परिचय",
    "birth": "जन्म",
    "body": "शरीर और बढ़ना",
    "illness": "बीमारी",
    "pregnancy": "गर्भावस्था",
    "family": "परिवार",
    "close": "अंत",
}

DETAIL_GROUPS: tuple[str, ...] = ("birth", "body", "illness", "pregnancy", "family")

HI_QUESTIONS: dict[str, str] = {
    "intro_name": "आपका नाम क्या है?",
    "intro_married": "क्या आपकी शादी हो गई है, या होने वाली है?",
    "intro_children": "कितने बच्चे हैं, और उनकी उम्र क्या है?",
    "intro_bridge": "क्या कॉल के बारे में समझ आ गया?",
    "birth_mode": "क्या प्रसव ऑपरेशन से हुआ, या घर पर?",
    "birth_term": "क्या प्रसव समय से बहुत पहले या बहुत बाद हुआ?",
    "birth_nicu": "क्या बच्चे को काँच के बक्से या बच्चों के ICU में रखना पड़ा?",
    "birth_cord_bleed": "क्या नाल गिरने पर ज़्यादा खून बहा?",
    "birth_cord_3weeks": "क्या नाल गिरने में तीन हफ्ते से ज़्यादा लगे?",
    "birth_bcg": "क्या BCG के टीके की जगह पर बड़ी सूजन, फोड़ा, या बगल में गाँठ हुई?",
    "birth_hearing_vision": "क्या जन्म के समय सुनने और देखने की जाँच हुई?",
    "birth_cleft": "क्या होंठ या तालू कटा हुआ था?",
    "body_fits_milestones": "क्या झटके हुए, या बैठना, चलना, बोलना दूसरे बच्चों से देर से हुआ?",
    "body_regression": "क्या बच्चा कोई सीखी हुई बात अब नहीं कर पाता?",
    "body_blue": "क्या होंठ या उंगलियाँ नीली पड़ती हैं?",
    "body_tummy": "क्या पेट धीरे-धीरे बड़ा हो रहा है?",
    "body_transfusion": "क्या बच्चे को कभी खून चढ़ाना पड़ा?",
    "body_blood_count": "क्या डॉक्टर ने कहा कि खून कम या ज़्यादा है?",
    "body_joint_swelling": "क्या गिरने पर घुटने या जोड़ में सूजन आती है, जो कुछ दिनों में अपने आप ठीक हो जाती है?",
    "body_bleeding_bruise": "क्या चोट पर ज़्यादा खून बहता है, या आसानी से नील पड़ जाते हैं?",
    "body_private_parts": "क्या गुप्त अंग जैसे होने चाहिए वैसे नहीं दिखते?",
    "body_lumps_birth": "क्या जन्म से शरीर पर सूजन या गाँठें हैं?",
    "body_proportion": "क्या हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं?",
    "body_thumb": "क्या अंगूठा या उंगली छोटी है, या नहीं है?",
    "body_growth": "क्या बच्चा बाकी बच्चों जैसा बढ़ रहा है?",
    "body_limb_pain": "क्या हाथ-पैर में बार-बार दर्द होता है, जिसके लिए अस्पताल जाना पड़ता है?",
    "body_silver_hair": "क्या बाल चाँदी जैसे हैं?",
    "body_kidney": "क्या गुर्दे की कोई बीमारी बताई गई है?",
    "body_puberty_early": "क्या बढ़ती उम्र के बदलाव जल्दी आए?",
    "body_puberty_late": "क्या बढ़ती उम्र के बदलाव अभी तक नहीं आए?",
    "ill_admissions": "क्या बार-बार अस्पताल में भर्ती होना पड़ा?",
    "ill_iv_antibiotics": "क्या बार-बार इंफेक्शन के लिए ड्रिप या नस में दवा लगी?",
    "ill_odor": "क्या पेशाब या पाखाना में अजीब बदबू है?",
    "ill_sudden": "क्या जन्म के बाद अचानक बहुत बीमार होकर इमरजेंसी में ले जाना पड़ा?",
    "ill_fasting": "क्या खाना छूटने पर बहुत सुस्ती, उल्टी या झटके होते हैं?",
    "preg_conditions": "क्या प्रेग्नेंसी में ब्लड प्रेशर, शुगर या थायरॉइड रहा?",
    "preg_thalassemia": "क्या माँ और पिता दोनों की थैलेसीमिया जाँच हुई थी?",
    "preg_scan": "क्या प्रेग्नेंसी का स्कैन असामान्य था?",
    "preg_heart": "क्या स्कैन में या जन्म पर दिल में कोई खराबी बताई गई?",
    "preg_transfusion": "क्या प्रेग्नेंसी में खून चढ़ाना पड़ा?",
    "preg_delivery": "क्या प्रसव में कोई दिक्कत आई?",
    "fam_related": "क्या शादी से पहले पति रिश्तेदार थे?",
    "fam_miscarriage": "क्या कभी गर्भ गिरा, या गिरवाना पड़ा?",
    "fam_infant_death": "क्या परिवार में कोई बच्चा एक साल से कम उम्र में गुजर गया?",
    "fam_fits_delay": "क्या परिवार में किसी बच्चे या बड़े को झटके, देर से सीखना, या सीखी बात भूलना है?",
    "fam_blind": "क्या परिवार में किसी बच्चे या बड़े को कम उम्र से अंधापन है?",
    "fam_deaf": "क्या परिवार में किसी बच्चे या बड़े को कम उम्र से कम सुनाई देता है?",
    "fam_transfusion": "क्या परिवार में किसी को बार-बार खून चढ़ाना पड़ता है?",
    "fam_bleeding": "क्या परिवार में किसी को खून ज़्यादा बहने की समस्या है?",
    "fam_cancer": "क्या परिवार में किसी को कैंसर है?",
    "fam_heart": "क्या परिवार में किसी को दिल की बीमारी है?",
    "fam_kidney": "क्या परिवार में किसी को गुर्दे की बीमारी है?",
    "fam_bp_stroke": "क्या परिवार में 40-50 साल से कम उम्र के किसी को बीपी, शुगर, या लकवा है?",
    "fam_maternal_side": "क्या माँ की तरफ (माइके में) किसी को ज़्यादा खून, बार-बार इंफेक्शन, या खुजली वाले सूखे धब्बे हैं?",
    "fam_lumps": "क्या परिवार में किसी के शरीर पर सूजन या गाँठें हैं?",
    "fam_same_problem": "क्या परिवार में किसी को आपके बच्चे जैसी ही समस्या है?",
    "close_anything_else": "क्या इसके अलावा कुछ और बताना है?",
}


def question_hi(topic_id: str) -> str:
    return HI_QUESTIONS.get(topic_id) or topic_id
