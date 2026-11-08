"""Triage topics: one Hindi question per topic, and per-call coverage helpers.

Every clinical yes/no `question` is worded so that हाँ means a concern and
नहीं means no concern (see QUESTION_DIRECTION). `answer` marks the few
topics that are not yes/no.
"""

import question_flow

# Normative phrasing rule for every clinical screening question below: each one
# is worded so that answering 'नहीं' (no) means NO red flag / no concern, and
# 'हाँ' (yes) means a concern. The bot must ask the question in exactly this
# direction and must not invert it (e.g. never turn "does a tooth bleed a lot
# when it falls" into "does the bleeding stop quickly").
QUESTION_DIRECTION = (
    "Every screening question is phrased so that answering 'नहीं' (no) means "
    "NO red flag / no concern, and 'हाँ' (yes) means a concern. The bot must "
    "ask the question in exactly this direction and must not invert it."
)

GROUPS: tuple[str, ...] = ("intro", "birth", "body", "illness", "pregnancy", "family", "close")

# Groups whose yes/no questions are clinical red-flag screens and so follow
# QUESTION_DIRECTION. The intro and close groups are conversational, not screens.
CLINICAL_GROUPS: tuple[str, ...] = ("birth", "body", "illness", "pregnancy", "family")

TOPICS: tuple[dict, ...] = (
    {"id": "intro_name", "group": "intro", "answer": "name", "need": "mother's name",
     "question": "आपका नाम क्या है?"},
    {"id": "intro_married", "group": "intro", "answer": "marital", "need": "married, about to marry, or unmarried",
     "question": "क्या आपकी शादी हो गई है, या होने वाली है?"},
    {"id": "intro_children", "group": "intro", "answer": "children", "need": "how many children and their ages",
     "question": "आपके कितने बच्चे हैं, और उनकी उम्र क्या है?"},
    {"id": "intro_bridge", "group": "intro", "need": "she understood how the call works",
     "question": "क्या कॉल के बारे में समझ आ गया?"},
    {"id": "birth_mode", "group": "birth", "answer": "birth_mode", "need": "normal delivery or operation, at home or in hospital",
     "question": "बच्चे का जन्म नॉर्मल डिलीवरी से हुआ या ऑपरेशन से, और घर पर या अस्पताल में?"},
    {"id": "birth_term", "group": "birth", "need": "born much earlier or much later than 9 months",
     "question": "क्या बच्चा समय से बहुत पहले या बहुत बाद में पैदा हुआ था?"},
    {"id": "birth_nicu", "group": "birth", "need": "NICU (kaanch ka box) stay",
     "question": "क्या बच्चे को काँच के बक्से या बच्चों के ICU में रखना पड़ा?"},
    {"id": "birth_cord_bleed", "group": "birth", "need": "heavy bleeding when the umbilical cord fell",
     "question": "क्या नाल गिरने पर ज़्यादा खून बहा?"},
    {"id": "birth_cord_3weeks", "group": "birth", "need": "cord took more than 3 weeks to fall",
     "question": "क्या नाल गिरने में तीन हफ्ते से ज़्यादा लगे?"},
    {"id": "birth_bcg", "group": "birth", "need": "unusual BCG site reaction: big swelling, boil, or armpit lump",
     "question": "क्या BCG के टीके की जगह पर बड़ी सूजन, फोड़ा, या बगल में गाँठ हुई?"},
    {"id": "birth_hearing_vision", "group": "birth", "need": "a hearing or vision problem found on a test or noticed by the mother",
     "question": "क्या सुनने या देखने की जाँच में कोई दिक्कत बताई गई, या बच्चा ठीक से सुन या देख नहीं पाता?"},
    {"id": "birth_cleft", "group": "birth", "need": "cleft lip or palate",
     "question": "क्या होंठ या तालू कटा हुआ था?"},
    {"id": "body_fits_milestones", "group": "body", "need": "fits, or sitting, walking, talking later than other children",
     "question": "क्या झटके आए, या बैठना, चलना, बोलना दूसरे बच्चों से देर से हुआ?"},
    {"id": "body_regression", "group": "body", "need": "lost a skill the child already had",
     "question": "क्या बच्चा कोई सीखी हुई बात अब नहीं कर पाता?"},
    {"id": "body_blue", "group": "body", "need": "blue lips, fingers or toes",
     "question": "क्या होंठ या उंगलियाँ नीली पड़ती हैं?"},
    {"id": "body_tummy", "group": "body", "need": "tummy slowly getting bigger, or liver or spleen enlarged",
     "question": "क्या पेट धीरे-धीरे बड़ा हो रहा है, या डॉक्टर ने लिवर या तिल्ली बड़ी बताई?"},
    {"id": "body_transfusion", "group": "body", "need": "child ever needed a blood transfusion",
     "question": "क्या बच्चे को कभी खून चढ़ाना पड़ा?"},
    {"id": "body_blood_count", "group": "body", "need": "doctor said the blood count is low or high",
     "question": "क्या डॉक्टर ने कहा कि खून कम या ज़्यादा है?"},
    {"id": "body_joint_swelling", "group": "body", "need": "knee or joint swelling after a fall or play that settles in days",
     "question": "क्या गिरने पर घुटने या जोड़ में सूजन आती है, जो कुछ दिनों में अपने आप ठीक हो जाती है?"},
    {"id": "body_bleeding_bruise", "group": "body", "need": "heavy bleeding after minor cuts or injury, or easy bruising",
     "question": "क्या चोट पर ज़्यादा खून बहता है, या आसानी से नील पड़ जाते हैं?"},
    {"id": "body_private_parts", "group": "body", "need": "private parts do not look as they should",
     "question": "क्या गुप्त अंग जैसे होने चाहिए वैसे नहीं दिखते?"},
    {"id": "body_lumps_birth", "group": "body", "need": "swellings or lumps on the body since birth",
     "question": "क्या जन्म से शरीर पर सूजन या गाँठें हैं?"},
    {"id": "body_proportion", "group": "body", "need": "arms or legs very short for the body",
     "question": "क्या हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं?"},
    {"id": "body_thumb", "group": "body", "need": "small or missing thumb or finger",
     "question": "क्या अंगूठा या उंगली छोटी है, या नहीं है?"},
    {"id": "body_growth", "group": "body", "need": "weight or height growing less than other children",
     "question": "क्या बच्चे का वज़न या कद बाकी बच्चों से कम बढ़ रहा है?"},
    {"id": "body_limb_pain", "group": "body", "need": "repeated arm or leg pain that needs hospital",
     "question": "क्या हाथ-पैर में बार-बार दर्द होता है, जिसके लिए अस्पताल जाना पड़ता है?"},
    {"id": "body_puberty", "group": "body", "need": "puberty changes came much earlier or much later than other children",
     "question": "क्या शरीर में बढ़ती उम्र के बदलाव बाकी बच्चों से बहुत पहले या बहुत बाद में आए?"},
    {"id": "ill_admissions", "group": "illness", "need": "admitted to hospital again and again",
     "question": "क्या बार-बार अस्पताल में भर्ती होना पड़ा?"},
    {"id": "ill_iv_antibiotics", "group": "illness", "need": "repeated infections that needed drip or IV antibiotics",
     "question": "क्या बार-बार इंफेक्शन के लिए ड्रिप या नस में दवा लगी?"},
    {"id": "ill_odor", "group": "illness", "need": "urine or stool smells strange",
     "question": "क्या पेशाब या पाखाना में अजीब बदबू है?"},
    {"id": "ill_sudden", "group": "illness", "need": "fine at birth, then suddenly very ill and taken to emergency",
     "question": "क्या जन्म के बाद अचानक बहुत बीमार होकर इमरजेंसी में ले जाना पड़ा?"},
    {"id": "ill_fasting", "group": "illness", "need": "very sleepy, vomiting or fits after a missed meal or during a minor illness",
     "question": "क्या खाना छूटने पर या हल्की बीमारी में बहुत सुस्ती, उल्टी या झटके होते हैं?"},
    {"id": "preg_conditions", "group": "pregnancy", "need": "high BP, sugar or thyroid in pregnancy",
     "question": "क्या प्रेग्नेंसी में ब्लड प्रेशर, शुगर या थायरॉइड की दिक्कत रही?"},
    {"id": "preg_thalassemia", "group": "pregnancy", "need": "a thalassemia test of the mother or father showed trait, minor, or another problem",
     "question": "क्या आपकी या पति की थैलेसीमिया जाँच में कोई दिक्कत, जैसे माइनर या ट्रेट, बताई गई?"},
    {"id": "preg_scan", "group": "pregnancy", "need": "a pregnancy scan showed a problem",
     "question": "क्या प्रेग्नेंसी के किसी स्कैन में कोई दिक्कत बताई गई?"},
    {"id": "preg_heart", "group": "pregnancy", "need": "a heart problem found on a scan or at birth",
     "question": "क्या स्कैन में या जन्म पर दिल में कोई खराबी बताई गई?"},
    {"id": "preg_transfusion", "group": "pregnancy", "need": "blood transfusion during pregnancy",
     "question": "क्या प्रेग्नेंसी में खून चढ़ाना पड़ा?"},
    {"id": "preg_delivery", "group": "pregnancy", "need": "a problem during delivery",
     "question": "क्या प्रसव में कोई दिक्कत आई?"},
    {"id": "fam_related", "group": "family", "need": "she and her husband were related before marriage",
     "question": "क्या शादी से पहले आप और आपके पति रिश्तेदार थे?"},
    {"id": "fam_miscarriage", "group": "family", "need": "miscarriage or abortion",
     "question": "क्या कभी गर्भ गिरा, या गिरवाना पड़ा?"},
    {"id": "fam_infant_death", "group": "family", "need": "a baby in the family died before 1 year",
     "question": "क्या परिवार में किसी बच्चे की एक साल से कम उम्र में मौत हुई?"},
    {"id": "fam_fits_delay", "group": "family", "need": "fits, late milestones, or lost skills in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े को झटके, देर से सीखना, या सीखी बात भूलना है?"},
    {"id": "fam_blind", "group": "family", "need": "blindness from a young age in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े को कम उम्र से अंधापन है?"},
    {"id": "fam_deaf", "group": "family", "need": "poor hearing from a young age in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े को कम उम्र से कम सुनाई देता है?"},
    {"id": "fam_transfusion", "group": "family", "need": "any child or adult in the family needs blood transfusions",
     "question": "क्या परिवार में किसी बच्चे या बड़े को बार-बार खून चढ़ाना पड़ता है?"},
    {"id": "fam_bleeding", "group": "family", "need": "any child or adult in the family bleeds too much",
     "question": "क्या परिवार में किसी बच्चे या बड़े को खून ज़्यादा बहने की समस्या है?"},
    {"id": "fam_cancer", "group": "family", "need": "cancer in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े को कैंसर है?"},
    {"id": "fam_heart", "group": "family", "need": "heart disease in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े को दिल की बीमारी है?"},
    {"id": "fam_kidney", "group": "family", "need": "kidney disease in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े को गुर्दे की बीमारी है?"},
    {"id": "fam_bp_stroke", "group": "family", "need": "high BP, sugar, cholesterol or stroke in a family member under 40-50",
     "question": "क्या परिवार में 40-50 साल से कम उम्र के किसी को बीपी, शुगर, कोलेस्ट्रॉल या लकवा है?"},
    {"id": "fam_maternal_side", "group": "family", "need": "on the mother's side: heavy bleeding, repeated infections, or dry itchy skin patches",
     "question": "क्या माँ की तरफ (मायके में) किसी को ज़्यादा खून, बार-बार इंफेक्शन, या खुजली वाले सूखे धब्बे हैं?"},
    {"id": "fam_lumps", "group": "family", "need": "swellings or lumps on the body in any child or adult in the family",
     "question": "क्या परिवार में किसी बच्चे या बड़े के शरीर पर सूजन या गाँठें हैं?"},
    {"id": "fam_same_problem", "group": "family", "need": "someone in the family has the same problem as her child",
     "question": "क्या परिवार में किसी को आपके बच्चे जैसी ही समस्या है?", "optional": True},
    {"id": "close_anything_else", "group": "close", "need": "anything else she wants to tell the doctor",
     "question": "क्या इसके अलावा डॉक्टर को कुछ और बताना है?"},
)

# Old saved findings used these domain names. This is not the reverse of the
# doctor-note grouping: several topics share a domain, and the old reader
# picks one topic per domain.
OLD_DOMAIN_TO_TOPIC = {
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

_BY_ID: dict[str, dict] = {t["id"]: t for t in TOPICS}
REQUIRED_IDS: tuple[str, ...] = tuple(t["id"] for t in TOPICS if not t.get("optional"))


def question_hi(topic_id: str) -> str:
    topic = _BY_ID.get(topic_id)
    return topic["question"] if topic else topic_id


def is_yes_no(topic_id: str) -> bool:
    topic = _BY_ID.get(topic_id)
    return bool(topic) and "answer" not in topic


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


def tool_note(coverage: list, bridged: list | None = None) -> str:
    left = open_topics(coverage)
    if not left:
        return (
            "All topics are covered. Speak the full closing now in your same warm female voice (Kore) in Hindi, "
            "then call end_call. Do not read this note aloud."
        )
    t = left[0]
    nxt = f"{t['id']} (ask: {t['question']})"
    bridge = ""
    if bridged is not None and t["group"] in question_flow.GROUP_BRIDGE_GROUPS and t["group"] not in bridged:
        bridged.append(t["group"])
        bridge = (
            f"NEW SECTION ({t['group']}): first say the {t['group'].capitalize()} GROUP BRIDGE from your "
            "instructions in one short sentence, then ask the next open topic. "
        )
    return (
        f"Saved. {bridge}Next open topic: {nxt}. "
        "If her last answer was 'Haan' and you already asked a follow-up in this turn, stop and wait for her answer. "
        "Otherwise ask ONLY this single open topic as ONE short Hindi question, in the same yes/no direction as written "
        "('haan' flags a concern, 'nahi' means no concern — never invert the question). "
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
        f"Do not end the call yet. Still open: {names}. "
        "Do NOT say anything about ending the call or about questions being left. "
        "If she reported an emergency danger sign happening now, speak the EMERGENCY line and call end_call with reason 'emergency'. "
        "If she clearly asked to stop the call, speak the CALLER STOP line and call end_call with reason 'caller_stop'. "
        "Otherwise ask the next open topic as one short Hindi question. Do not read this note aloud."
    )


def render_topic_list() -> str:
    lines: list[str] = []
    for group in GROUPS:
        lines.append(group.upper())
        for t in TOPICS:
            if t["group"] != group:
                continue
            extra = " (optional)" if t.get("optional") else ""
            lines.append(f"- {t['id']}{extra}: {t['question']}")
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

EMERGENCY_MOTHER_NOTE = "बच्चे को अभी तुरंत नज़दीकी अस्पताल ले जाइए, या 108 पर कॉल कीजिए।"


def incomplete_notes(coverage: list, reason: str, detail: str = "") -> tuple[str, str]:
    import call_state_contract as contract

    left = open_topics(coverage)
    if reason == "emergency":
        said = (detail or "").strip()[:200]
        quote = f' Caller said: "{said}".' if said else ""
        doctor = (
            f"> **EMERGENCY reported during call.**{quote} Caller was told to go to the nearest hospital now or call 108. "
            f"Screening was stopped; {len(left)} of {len(REQUIRED_IDS)} topics were not asked."
        )
        return doctor, EMERGENCY_MOTHER_NOTE
    if not left:
        return "", ""
    why = contract.CLOSE_REASON_NOTES.get(reason, "the call ended early")
    ids = ", ".join(t["id"] for t in left)
    doctor = (
        f"> **Incomplete call:** {why}. {len(left)} of {len(REQUIRED_IDS)} topics were not asked: {ids}."
    )
    mother = "यह कॉल अधूरी रही। कृपया जब समय मिले, दोबारा कॉल (dobara call) करके बाकी सवाल पूरे कर लीजिए।"
    return doctor, mother


GROUP_HI: dict[str, str] = {
    "intro": "परिचय",
    "birth": "जन्म",
    "body": "शरीर और बढ़ना",
    "illness": "बीमारी",
    "pregnancy": "गर्भावस्था",
    "family": "परिवार",
    "close": "अंत",
}

DETAIL_GROUPS: tuple[str, ...] = CLINICAL_GROUPS
