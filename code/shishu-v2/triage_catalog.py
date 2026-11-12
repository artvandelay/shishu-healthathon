"""Declarative gate-first clinical catalog for SHISHU v3.

Question text comes from docs/requirements-v3.md sections 4-5 and the Nov 4
topic wording already used in clinic. हाँ on a gate is a concern and opens
its modules. नहीं moves on. enabled=False drops an item from the live list
without deleting it.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    id: str
    group: str
    kind: str
    question_hi: str
    need_en: str
    nrroid_gate: str
    skip_if: tuple[str, ...] = ()
    opens: tuple[str, ...] = ()
    followups: tuple[str, ...] = ()
    enabled: bool = True


def _q(
    qid: str,
    group: str,
    kind: str,
    question_hi: str,
    need_en: str,
    nrroid: str = "",
    skip: tuple[str, ...] = (),
    opens: tuple[str, ...] = (),
    follow: tuple[str, ...] = (),
    enabled: bool = True,
) -> QuestionSpec:
    return QuestionSpec(
        id=qid,
        group=group,
        kind=kind,
        question_hi=question_hi,
        need_en=need_en,
        nrroid_gate=nrroid,
        skip_if=skip,
        opens=opens,
        followups=follow,
        enabled=enabled,
    )


# Follow-up prompts are not separate catalog rows. The cursor speaks one at a time.
FOLLOWUP_HI: dict[str, str] = {
    "fracture_count_or_bowing": "यह कितनी बार हुआ — हड्डी टूटी, या पैर टेढ़े या मुड़े हुए हैं?",
    "urine_or_stone_detail": "बहुत ज़्यादा पेशाब और पानी की बात है, या पथरी या गुर्दे की दिक्कत? थोड़ा और बताइए।",
    "bp_who": "यह नज़दीकी रिश्तेदार को हुआ, जैसे माँ-पिता या भाई-बहन, या दूर के रिश्तेदार को?",
    "bp_age": "उन्हें यह किस उम्र में हुआ?",
}

_CHILD = ("no_children",)
_PREG_HISTORY = ("unmarried", "ever_pregnant_no")

CATALOG: tuple[QuestionSpec, ...] = (
    # --- intro (section 5.1) ---
    _q(
        "intro_name", "intro", "profile",
        "नमस्ते, मैं शिशु हूँ। आपका नाम बताइएगा प्लीज़?",
        "caller name; ask again if only a fragment such as मेरा नाम",
    ),
    _q(
        "intro_married", "intro", "profile",
        "क्या आपकी शादी हो गई है, या होने वाली है?",
        "married, about to marry, or unmarried",
    ),
    _q(
        "intro_children", "intro", "profile",
        "आपके कितने बच्चे हैं, और उनकी उम्र क्या-क्या है?",
        "how many children and each child's age; hold if count has no ages",
        skip=("unmarried",),
    ),
    _q(
        "intro_bridge", "intro", "bridge",
        "मैं आपके बच्चे, प्रेग्नेंसी और परिवार के बारे में कुछ आसान सवाल पूछूँगी। आप छोटा या लंबा जैसे चाहें जवाब दे सकती हैं, और कुछ समझ न आए तो बेझिझक पूछ सकती हैं। ठीक है?",
        "explain the call; she may answer short or long and ask anything",
    ),
    # --- 14 wide gates + 3 family sub-gates (section 5.2) ---
    _q(
        "gate_birth_neonatal", "birth", "gate",
        "क्या किसी बच्चे का जन्म ऑपरेशन से या समय से बहुत पहले हुआ था, या जन्म के बाद काँच के बक्से (NICU) या पीलिया की नीली लाइट में रखना पड़ा था?",
        "operation, very preterm, NICU, or jaundice blue light",
        "Q03",
        skip=_CHILD,
        opens=("birth_mode_detail", "birth_term", "birth_nicu", "birth_sudden"),
    ),
    _q(
        "gate_cord_bcg", "birth", "gate",
        "क्या जन्म के बाद नाल गिरने में 3 हफ्ते से ज़्यादा लगे, नाल से ज़्यादा खून आया, या BCG के टीके की जगह पर बड़ा फोड़ा या बगल में गाँठ हुई?",
        "cord delay, cord bleeding, or unusual BCG site",
        "Q03",
        skip=_CHILD,
        opens=("birth_cord_bleed", "birth_cord_3weeks", "birth_bcg"),
    ),
    _q(
        "gate_metabolic_crisis", "illness", "gate",
        "क्या खाना छूटने पर, हल्की बीमारी में, या कुछ खास खाने पर बच्चा बहुत सुस्त हो जाता है, बार-बार उल्टी करता है, या पेशाब-पॉटी में अजीब बदबू आती है?",
        "sleepiness, vomiting, or unusual urine or stool smell with fasting, illness, or foods",
        "Q01",
        skip=_CHILD,
        opens=("ill_fasting", "ill_odor"),
    ),
    _q(
        "gate_milestones_neuro", "body", "gate",
        "क्या बच्चे को कभी झटके (फ़िट्स) आए हैं, या बैठना-चलना-बोलना देर से सीखा, या कोई सीखी हुई बात अब भूल गया है?",
        "fits, late milestones, or lost a learned skill",
        "Q02",
        skip=_CHILD,
        opens=("body_fits_milestones", "body_regression"),
    ),
    _q(
        "gate_sensory_cleft_heart", "body", "gate",
        "क्या बच्चे को सुनने या देखने में कोई दिक्कत है, होंठ या तालू कटा था, या होंठ-उंगलियाँ कभी नीली पड़ती हैं?",
        "hearing or vision difficulty, cleft lip or palate, or blue lips or fingers",
        "Q02",
        skip=_CHILD,
        opens=("birth_hearing_vision", "birth_cleft", "body_blue", "preg_heart"),
    ),
    _q(
        "gate_infections", "illness", "gate",
        "क्या बच्चे को बार-बार अस्पताल में भर्ती करना पड़ा है, या बार-बार इंफेक्शन के लिए नस में (IV) एंटीबायोटिक दवा लगी है?",
        "repeated hospital admissions or IV antibiotics",
        "Q03",
        skip=_CHILD,
        opens=("ill_admissions", "ill_iv_antibiotics"),
    ),
    _q(
        "gate_bleeding_joints", "body", "gate",
        "क्या चोट लगने पर खून देर तक बहता है, आसानी से नील पड़ते हैं, या गिरने पर घुटने-जोड़ में सूजन आती है जो कुछ दिनों में ठीक हो जाती है?",
        "prolonged bleeding, easy bruising, or joint swelling after a fall",
        "Q04",
        skip=_CHILD,
        opens=("body_bleeding_bruise", "body_joint_swelling", "mod_tooth_bleed", "mod_heavy_periods"),
    ),
    _q(
        "gate_transfusion_anemia", "body", "gate",
        "क्या बच्चे को कभी खून चढ़ाना पड़ा है, डॉक्टर ने खून कम-ज़्यादा बताया है, या हाथ-पैर में बार-बार तेज़ दर्द के लिए अस्पताल जाना पड़ता है?",
        "transfusion, abnormal blood count, or repeated limb pain needing hospital",
        "Q05",
        skip=_CHILD,
        opens=("body_transfusion", "body_blood_count", "body_limb_pain"),
    ),
    _q(
        "gate_tummy_lumps", "body", "gate",
        "क्या बच्चे का पेट धीरे-धीरे बड़ा हो रहा है (लिवर या तिल्ली बढ़ना), या जन्म से शरीर पर सूजन या गाँठें हैं?",
        "enlarging tummy or swellings or lumps since birth",
        "Q06",
        skip=_CHILD,
        opens=("body_tummy", "body_lumps_birth"),
    ),
    _q(
        "gate_growth_shape", "body", "gate",
        "क्या बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है, हाथ-पैर शरीर के मुकाबले छोटे हैं, या कोई उंगली-अंगूठा छोटा या कम है?",
        "poor growth, short limbs, or a small or missing digit",
        "Q07",
        skip=_CHILD,
        opens=("body_growth", "body_proportion", "body_thumb"),
    ),
    _q(
        "gate_bones_fractures", "body", "gate",
        "क्या मामूली गिरने पर कभी बच्चे की हड्डी टूटी है, या विटामिन-डी और कैल्शियम देने के बाद भी पैर टेढ़े या मुड़े हुए हैं?",
        "fracture after a minor fall, or bowed legs after vitamin D and calcium",
        "Q08",
        skip=_CHILD,
        follow=("fracture_count_or_bowing",),
    ),
    _q(
        "gate_urine_kidney", "body", "gate",
        "क्या बच्चा बार-बार बहुत ज़्यादा पेशाब करता और बहुत पानी पीता है, या कभी पेशाब में पथरी या गुर्दे की दिक्कत बताई गई है?",
        "excess urine and thirst, or stones or a kidney problem",
        "Q09",
        skip=_CHILD,
        follow=("urine_or_stone_detail",),
    ),
    _q(
        "gate_endocrine_private", "body", "gate",
        "क्या बच्चे के गुप्त अंग वैसे नहीं दिखते जैसे होने चाहिए, या (10 साल से बड़े बच्चे में) बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं?",
        "atypical genitalia, or puberty much earlier or later in a child 10 or older",
        "Q10",
        skip=_CHILD,
        opens=("body_private_parts", "body_puberty"),
    ),
    _q(
        "gate_pregnancy_thal", "pregnancy", "gate",
        "क्या किसी प्रेग्नेंसी में बीपी-शुगर-थायरॉइड, स्कैन में खराबी, प्रेग्नेंसी में खून चढ़ना, डिलीवरी में दिक्कत, या आपकी/पति की थैलेसीमिया जाँच में माइनर/ट्रेट बताया गया?",
        "pregnancy BP, sugar, thyroid, scan, transfusion, delivery trouble, or thalassemia trait",
        "Antenatal",
        skip=_PREG_HISTORY,
        opens=("preg_conditions", "preg_scan", "preg_transfusion", "preg_delivery", "preg_thalassemia"),
    ),
    _q(
        "fam_related", "family", "gate",
        "क्या आप और पति शादी से पहले रिश्तेदार थे?",
        "she and her husband were related before marriage",
        "Q11",
        skip=("unmarried",),
    ),
    _q(
        "fam_loss_gate", "family", "gate",
        "क्या कभी गर्भ गिरा, या परिवार में किसी छोटे बच्चे की 1 साल से कम उम्र में मौत हुई?",
        "miscarriage or a family infant death before 1 year",
        "Q12",
        opens=("fam_miscarriage", "fam_infant_death"),
    ),
    _q(
        "fam_conditions_gate", "family", "gate",
        "क्या परिवार में किसी बच्चे या बड़े को खून चढ़ने, झटके, अंधेपन, बहरेपन, कैंसर, दिल-गुर्दे या कम उम्र में बीपी-लकवे की बीमारी है?",
        "family bleeding, fits, vision, hearing, cancer, heart, kidney, or early BP or stroke",
        "Q11",
        opens=(
            "fam_fits_delay", "fam_transfusion", "fam_bleeding", "fam_blind", "fam_deaf",
            "fam_cancer", "fam_heart", "fam_kidney", "fam_bp_stroke", "fam_maternal_side",
            "fam_lumps", "fam_same_problem",
        ),
    ),
    _q(
        "close_anything_else", "close", "open_capture",
        "क्या इसके अलावा डॉक्टर के लिए कुछ और बताना है जो आपके मन में हो?",
        "anything else she wants the doctor to know",
    ),
    # --- modules opened only after a gate concern ---
    _q(
        "birth_mode_detail", "birth", "module",
        "क्या बच्चे का जन्म ऑपरेशन से हुआ था?",
        "birth by operation",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_term", "birth", "module",
        "क्या प्रसव समय से बहुत पहले या बहुत बाद हुआ?",
        "birth much before or after 9 months",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_nicu", "birth", "module",
        "क्या बच्चे को काँच के बक्से या बच्चों के ICU में रखना पड़ा?",
        "NICU or glass-box stay",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_sudden", "birth", "module",
        "क्या जन्म के बाद अचानक बहुत बीमार होकर इमरजेंसी में ले जाना पड़ा?",
        "well at birth, then suddenly very ill and taken to emergency",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_cord_bleed", "birth", "module",
        "क्या नाल गिरने पर ज़्यादा खून बहा?",
        "heavy bleeding when the cord fell",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_cord_3weeks", "birth", "module",
        "क्या नाल गिरने में तीन हफ्ते से ज़्यादा लगे?",
        "cord took more than 3 weeks to fall",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_bcg", "birth", "module",
        "क्या BCG के टीके की जगह पर बड़ी सूजन, फोड़ा, या बगल में गाँठ हुई?",
        "big BCG swelling, boil, or armpit lump",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "birth_hearing_vision", "birth", "module",
        "क्या सुनने या देखने में दिक्कत बताई गई?",
        "hearing or vision difficulty",
        "Q02",
        skip=_CHILD,
    ),
    _q(
        "birth_cleft", "birth", "module",
        "क्या होंठ या तालू कटा हुआ था?",
        "cleft lip or palate",
        "Q02",
        skip=_CHILD,
    ),
    _q(
        "body_blue", "body", "module",
        "क्या होंठ या उंगलियाँ नीली पड़ती हैं?",
        "blue lips or fingers",
        "Q02",
        skip=_CHILD,
    ),
    _q(
        "body_fits_milestones", "body", "module",
        "क्या झटके हुए, या बैठना, चलना, बोलना दूसरे बच्चों से देर से हुआ?",
        "fits or late sitting, walking, or talking",
        "Q02",
        skip=_CHILD,
    ),
    _q(
        "body_regression", "body", "module",
        "क्या बच्चा कोई सीखी हुई बात अब नहीं कर पाता?",
        "lost a skill the child already had",
        "Q02",
        skip=_CHILD,
    ),
    _q(
        "ill_fasting", "illness", "module",
        "क्या खाना छूटने पर बहुत सुस्ती, उल्टी या झटके होते हैं?",
        "very sleepy, repeated vomiting, or fits when a meal is missed",
        "Q01",
        skip=_CHILD,
    ),
    _q(
        "ill_odor", "illness", "module",
        "क्या पेशाब या पाखाना में अजीब बदबू है?",
        "urine or stool smells unusual",
        "Q01",
        skip=_CHILD,
    ),
    _q(
        "ill_admissions", "illness", "module",
        "क्या बार-बार अस्पताल में भर्ती होना पड़ा?",
        "admitted to hospital again and again",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "ill_iv_antibiotics", "illness", "module",
        "क्या बार-बार इंफेक्शन के लिए ड्रिप या नस में दवा लगी?",
        "repeated infections that needed IV antibiotics",
        "Q03",
        skip=_CHILD,
    ),
    _q(
        "body_bleeding_bruise", "body", "module",
        "क्या चोट पर ज़्यादा खून बहता है, या आसानी से नील पड़ जाते हैं?",
        "prolonged bleeding after injury, or easy bruising",
        "Q04",
        skip=_CHILD,
    ),
    _q(
        "body_joint_swelling", "body", "module",
        "क्या गिरने पर घुटने या जोड़ में सूजन आती है, जो कुछ दिनों में अपने आप ठीक हो जाती है?",
        "knee or joint swelling after a fall that settles in days",
        "Q04",
        skip=_CHILD,
    ),
    _q(
        "mod_tooth_bleed", "body", "module",
        "क्या दूध का दाँत गिरने पर खून देर तक बहा?",
        "bleeding when a milk tooth fell; only if a child is 5 or older",
        "Q04",
        skip=("no_children", "no_child_5_plus"),
    ),
    _q(
        "mod_heavy_periods", "body", "module",
        "क्या पीरियड्स में खून बहुत ज़्यादा बहता है?",
        "heavy periods; only if a girl is 11 or older",
        "Q04",
        skip=("no_children", "no_daughter_11_plus"),
    ),
    _q(
        "body_transfusion", "body", "module",
        "क्या बच्चे को कभी खून चढ़ाना पड़ा?",
        "child ever needed blood",
        "Q05",
        skip=_CHILD,
    ),
    _q(
        "body_blood_count", "body", "module",
        "क्या डॉक्टर ने कहा कि खून कम या ज़्यादा है?",
        "doctor said the blood count is low or high",
        "Q05",
        skip=_CHILD,
    ),
    _q(
        "body_limb_pain", "body", "module",
        "क्या हाथ-पैर में बार-बार दर्द होता है, जिसके लिए अस्पताल जाना पड़ता है?",
        "repeated limb pain that needs hospital",
        "Q05",
        skip=_CHILD,
    ),
    _q(
        "body_tummy", "body", "module",
        "क्या पेट धीरे-धीरे बड़ा हो रहा है?",
        "tummy slowly getting bigger",
        "Q06",
        skip=_CHILD,
    ),
    _q(
        "body_lumps_birth", "body", "module",
        "क्या जन्म से शरीर पर सूजन या गाँठें हैं?",
        "swellings or lumps on the body since birth",
        "Q06",
        skip=_CHILD,
    ),
    _q(
        "body_growth", "body", "module",
        "क्या बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है?",
        "height or weight growing much less than other children",
        "Q07",
        skip=_CHILD,
    ),
    _q(
        "body_proportion", "body", "module",
        "क्या हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं?",
        "arms or legs very short compared with the body",
        "Q07",
        skip=_CHILD,
    ),
    _q(
        "body_thumb", "body", "module",
        "क्या अंगूठा या उंगली छोटी है, या नहीं है?",
        "small or missing thumb or finger",
        "Q07",
        skip=_CHILD,
    ),
    _q(
        "body_private_parts", "body", "module",
        "क्या गुप्त अंग जैसे होने चाहिए वैसे नहीं दिखते?",
        "private parts do not look as they should",
        "Q10",
        skip=_CHILD,
    ),
    _q(
        "body_puberty", "body", "module",
        "क्या बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं?",
        "puberty changes much earlier or later; only if a child is 10 or older",
        "Q10",
        skip=("no_children", "all_under_10"),
    ),
    _q(
        "preg_conditions", "pregnancy", "module",
        "क्या प्रेग्नेंसी में ब्लड प्रेशर, शुगर या थायरॉइड रहा?",
        "high BP, sugar, or thyroid in pregnancy",
        "Antenatal",
        skip=_PREG_HISTORY,
    ),
    _q(
        "preg_scan", "pregnancy", "module",
        "क्या प्रेग्नेंसी का स्कैन असामान्य था?",
        "pregnancy scan was abnormal",
        "Antenatal",
        skip=_PREG_HISTORY,
    ),
    _q(
        "preg_transfusion", "pregnancy", "module",
        "क्या प्रेग्नेंसी में खून चढ़ाना पड़ा?",
        "blood transfusion in pregnancy",
        "Antenatal",
        skip=_PREG_HISTORY,
    ),
    _q(
        "preg_delivery", "pregnancy", "module",
        "क्या प्रसव में कोई दिक्कत आई?",
        "trouble at delivery",
        "Antenatal",
        skip=_PREG_HISTORY,
    ),
    _q(
        "preg_thalassemia", "pregnancy", "module",
        "क्या आपकी या पति की थैलेसीमिया जाँच में माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?",
        "thalassemia trait, minor, or an abnormal result was reported; asked alone when there is no pregnancy history",
        "Antenatal",
    ),
    _q(
        "preg_heart", "pregnancy", "module",
        "क्या स्कैन में या जन्म पर दिल में कोई खराबी बताई गई?",
        "heart problem on the scan or at birth",
        "Q02",
        skip=_CHILD,
    ),
    _q(
        "fam_miscarriage", "family", "module",
        "क्या कभी गर्भ गिरा, या गिरवाना पड़ा?",
        "miscarriage or abortion",
        "Q12",
        skip=("ever_pregnant_no", "unmarried_not_pregnant"),
    ),
    _q(
        "fam_infant_death", "family", "module",
        "क्या परिवार में कोई बच्चा एक साल से कम उम्र में गुजर गया?",
        "a baby in the family died before 1 year",
        "Q12",
    ),
    _q(
        "fam_fits_delay", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को झटके, देर से सीखना, या सीखी बात भूलना है?",
        "fits, late milestones, or lost skills in any child or adult in the family",
        "Q11",
    ),
    _q(
        "fam_transfusion", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को बार-बार खून चढ़ाना पड़ता है?",
        "any child or adult in the family needs blood transfusions",
        "Q11",
    ),
    _q(
        "fam_bleeding", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को खून ज़्यादा बहने की समस्या है?",
        "any child or adult in the family bleeds too much",
        "Q11",
    ),
    _q(
        "fam_blind", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को कम उम्र से अंधापन है?",
        "blindness in any child or adult in the family, not from old age",
        "Q11",
    ),
    _q(
        "fam_deaf", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को कम उम्र से कम सुनाई देता है?",
        "poor hearing in any child or adult in the family, not from old age",
        "Q11",
    ),
    _q(
        "fam_cancer", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को कैंसर है?",
        "cancer in any child or adult in the family",
        "Q11",
    ),
    _q(
        "fam_heart", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को दिल की बीमारी है?",
        "heart disease in any child or adult in the family",
        "Q11",
    ),
    _q(
        "fam_kidney", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को गुर्दे की बीमारी है?",
        "kidney disease in any child or adult in the family",
        "Q11",
    ),
    _q(
        "fam_bp_stroke", "family", "module",
        "क्या परिवार में 40-50 साल से कम उम्र के किसी बच्चे या बड़े को बीपी, शुगर, या लकवा है?",
        "BP, sugar, or stroke in a family member under 40-50; then who and at what age",
        "Q11",
        follow=("bp_who", "bp_age"),
    ),
    _q(
        "fam_maternal_side", "family", "module",
        "क्या माँ की तरफ (माइके में) किसी बच्चे या बड़े को ज़्यादा खून, बार-बार इंफेक्शन, या खुजली वाले सूखे धब्बे हैं?",
        "on the mother's side: heavy bleeding, repeated infections, or dry itchy patches",
        "Q11",
    ),
    _q(
        "fam_lumps", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े के शरीर पर सूजन या गाँठें हैं?",
        "swellings or lumps on any child or adult in the family",
        "Q11",
    ),
    _q(
        "fam_same_problem", "family", "module",
        "क्या परिवार में किसी बच्चे या बड़े को आपके बच्चे जैसी ही समस्या है?",
        "someone in the family has the same problem as her child; only after a family concern",
        "Q11",
        skip=("no_children", "no_family_concern"),
    ),
)

_BY_ID: dict[str, QuestionSpec] = {spec.id: spec for spec in CATALOG}

_SPINE_KINDS = frozenset({"profile", "bridge", "gate", "open_capture"})


def _validate() -> None:
    ids = [spec.id for spec in CATALOG]
    dupes = {qid for qid in ids if ids.count(qid) > 1}
    if dupes:
        raise RuntimeError(f"duplicate catalog ids: {sorted(dupes)}")
    known = set(ids)
    for spec in CATALOG:
        for qid in spec.opens:
            if qid not in known:
                raise RuntimeError(f"{spec.id} opens unknown {qid}")
        for fid in spec.followups:
            if fid not in FOLLOWUP_HI:
                raise RuntimeError(f"{spec.id} followup {fid} has no Hindi")
        if spec.kind not in ("profile", "bridge", "gate", "module", "open_capture"):
            raise RuntimeError(f"{spec.id} has kind {spec.kind}")


_validate()


def get_spec(qid: str) -> QuestionSpec:
    try:
        return _BY_ID[qid]
    except KeyError:
        raise KeyError(qid) from None


def active_specs() -> list[QuestionSpec]:
    return [spec for spec in CATALOG if spec.enabled]


def spine_ids() -> list[str]:
    """Ask order for items that are spoken on every applicable call.

    preg_thalassemia sits just after the pregnancy gate so a caller with no
    pregnancy history still gets the trait question (हाँ = minor/trait reported),
    and a caller who already heard it inside the gate does not hear it twice.
    """
    ids = [spec.id for spec in active_specs() if spec.kind in _SPINE_KINDS]
    anchor = "gate_pregnancy_thal"
    if anchor in ids and "preg_thalassemia" not in ids:
        ids.insert(ids.index(anchor) + 1, "preg_thalassemia")
    return ids
