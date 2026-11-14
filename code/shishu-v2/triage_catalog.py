"""Atomic yes/no catalog for SHISHU v4.

Question text is the Devanagari in docs/requirements-v4.md section 4.
हाँ on a clinical item is a concern. नहीं is clear. Nothing is hidden behind
another question: opens stays empty. question_multi_hi is the 2+ child
wording; an empty string means the general question_hi is used as written.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    id: str
    group: str
    kind: str
    question_hi: str
    question_multi_hi: str
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
    follow: tuple[str, ...] = (),
    multi: str = "",
    enabled: bool = True,
) -> QuestionSpec:
    return QuestionSpec(
        id=qid,
        group=group,
        kind=kind,
        question_hi=question_hi,
        question_multi_hi=multi,
        need_en=need_en,
        nrroid_gate=nrroid,
        skip_if=skip,
        opens=(),
        followups=follow,
        enabled=enabled,
    )


FOLLOWUP_HI: dict[str, str] = {
    'operation_reason': 'ऑपरेशन किस वजह से करना पड़ा था?',
    'term_how_early': 'बच्चा पूरे समय से कितने हफ्ते या महीने पहले पैदा हुआ था?',
    'nicu_reason': 'काँच के बक्से (NICU) में किस वजह से रखना पड़ा था?',
    'nicu_days': 'वहाँ कितने दिन रखना पड़ा था?',
    'jaundice_days': 'नीली लाइट में कितने दिन रखना पड़ा था?',
    'sudden_detail': 'यह कब हुआ था और डॉक्टर ने क्या बताया था?',
    'fits_onset': 'झटके पहली बार किस उम्र में आए थे?',
    'lost_skill': 'बच्चा पहले क्या कर लेता था जो अब नहीं कर पाता?',
    'transfusion_count': 'खून कितनी बार चढ़ाना पड़ा है?',
    'fracture_count_or_bowing': 'हड्डी कितनी बार टूटी, या पैर टेढ़े होने की बात है?',
    'urine_or_stone_detail': 'बार-बार पेशाब-प्यास की बात है, या पथरी या गुर्दे की दिक्कत?',
    'admissions_count': 'बच्चे को कितनी बार अस्पताल में भर्ती करना पड़ा है?',
    'admissions_reason': 'किस बीमारी या वजह से भर्ती करना पड़ा था?',
    'preg_cond_detail': 'बीपी, शुगर या थायरॉइड में से क्या था, और क्या डिलीवरी के बाद भी रहा?',
    'scan_detail': 'स्कैन में क्या दिक्कत बताई गई थी?',
    'delivery_detail': 'डिलीवरी के समय क्या दिक्कत आई थी?',
    'related_how': 'आप दोनों का शादी से पहले क्या रिश्ता था?',
    'miscarriage_count': 'गर्भ कितनी बार गिरा था?',
    'miscarriage_month': 'यह कितने महीने की प्रेग्नेंसी में हुआ था?',
    'infant_death_age': 'उस बच्चे की उम्र कितनी थी, और आपसे क्या रिश्ता था?',
    'infant_death_cause': 'क्या मौत की कोई वजह पता चली थी?',
    'fam_who': 'परिवार में यह दिक्कत किसको है — आपसे उनका क्या रिश्ता है?',
    'bp_who': 'यह सगे रिश्तेदार (जैसे माता-पिता, भाई-बहन) को हुआ, या दूर के रिश्तेदार को?',
    'bp_age': 'उन्हें यह किस उम्र में हुआ था?',
    'open_concern_detail': 'जी बताइए, डॉक्टर के लिए क्या बात नोट करूँ?',
}


CATALOG: tuple[QuestionSpec, ...] = (
    _q('intro_name', 'intro', 'profile', 'नमस्ते, मैं शिशु हूँ। आपका नाम बताइएगा प्लीज़?', 'caller name; ask again if only a fragment such as मेरा नाम', skip=(), follow=()),
    _q('intro_married', 'intro', 'profile', 'क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?', 'married, about to marry, or unmarried', skip=(), follow=()),
    _q('intro_children', 'intro', 'profile', 'आपके कितने बच्चे हैं, और उनकी उम्र क्या-क्या है?', "how many children and each child's age", skip=('unmarried', 'about_to_marry'), follow=()),
    _q('intro_bridge', 'intro', 'bridge', 'अब मैं एक-एक करके छोटे और आसान सवाल पूछूँगी। आप बोलकर या नीचे हाँ या नहीं का बटन दबाकर जवाब दे सकती हैं, और कुछ समझ न आए तो पूछ सकती हैं। शुरू करें?', 'short yes or no questions; she can speak or tap हाँ or नहीं', skip=(), follow=()),
    _q('birth_mode_detail', 'birth', "gate", 'क्या बच्चे का जन्म ऑपरेशन (सिजेरियन) से हुआ था?', 'birth by operation', 'Q03', skip=('no_children',), follow=('operation_reason',), multi='क्या आपके किसी भी बच्चे का जन्म ऑपरेशन (सिजेरियन) से हुआ था?'),
    _q('birth_term', 'birth', "gate", 'क्या बच्चे का जन्म समय से बहुत पहले (9 महीने से काफी पहले) हुआ था?', 'birth much before 9 months', 'Q03', skip=('no_children',), follow=('term_how_early',), multi='क्या आपके किसी भी बच्चे का जन्म समय से बहुत पहले (9 महीने से काफी पहले) हुआ था?'),
    _q('birth_nicu', 'birth', "gate", 'क्या जन्म के बाद बच्चे को काँच के बक्से या बच्चों के ICU (NICU) में रखना पड़ा था?', 'NICU or glass-box stay after birth', 'Q03', skip=('no_children',), follow=('nicu_reason', 'nicu_days'), multi='क्या जन्म के बाद आपके किसी भी बच्चे को काँच के बक्से या बच्चों के ICU (NICU) में रखना पड़ा था?'),
    _q('birth_jaundice', 'birth', "gate", 'क्या जन्म के बाद पीलिया के लिए बच्चे को नीली लाइट में रखना पड़ा था?', 'jaundice treated under blue light', 'Q03', skip=('no_children',), follow=('jaundice_days',), multi='क्या जन्म के बाद पीलिया के लिए आपके किसी भी बच्चे को नीली लाइट में रखना पड़ा था?'),
    _q('birth_sudden', 'birth', "gate", 'क्या जन्म के बाद बच्चा अचानक बहुत बीमार पड़ा और इमरजेंसी में ले जाना पड़ा?', 'suddenly very ill after birth and taken to emergency', 'Q03', skip=('no_children',), follow=('sudden_detail',), multi='क्या जन्म के बाद आपका कोई भी बच्चा अचानक बहुत बीमार पड़ा और इमरजेंसी में ले जाना पड़ा?'),
    _q('birth_cord_3weeks', 'birth', "gate", 'क्या जन्म के बाद बच्चे की नाल गिरने में 3 हफ्ते से ज़्यादा लगे थे?', 'cord took more than 3 weeks to fall', 'Q03', skip=('no_children',), follow=(), multi='क्या जन्म के बाद आपके किसी भी बच्चे की नाल गिरने में 3 हफ्ते से ज़्यादा लगे थे?'),
    _q('birth_cord_bleed', 'birth', "gate", 'क्या नाल गिरने पर बहुत ज़्यादा खून आया था?', 'heavy bleeding when the cord fell', 'Q03', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे की नाल गिरने पर बहुत ज़्यादा खून आया था?'),
    _q('birth_bcg', 'birth', "gate", 'क्या कंधे के BCG टीके की जगह पर बड़ा फोड़ा, मवाद, या बगल में गाँठ हुई थी?', 'big boil, pus, or armpit lump at the BCG site', 'Q03', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे को कंधे के BCG टीके की जगह पर बड़ा फोड़ा, मवाद, या बगल में गाँठ हुई थी?'),
    _q('birth_cleft', 'birth', "gate", 'क्या जन्म से बच्चे का होंठ या तालू कटा हुआ था?', 'cleft lip or palate from birth', 'Q02', skip=('no_children',), follow=(), multi='क्या जन्म से आपके किसी भी बच्चे का होंठ या तालू कटा हुआ था?'),
    _q('birth_hearing', 'birth', "gate", 'क्या बच्चे को सुनने में कोई दिक्कत है?', 'hearing difficulty', 'Q02', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे को सुनने में कोई दिक्कत है?'),
    _q('birth_vision', 'birth', "gate", 'क्या बच्चे को देखने में कोई दिक्कत है?', 'vision difficulty', 'Q02', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे को देखने में कोई दिक्कत है?'),
    _q('body_fits', 'body', "gate", 'क्या बच्चे को कभी झटके या दौरे (फ़िट्स) आए हैं?', 'fits or seizures', 'Q02', skip=('no_children',), follow=('fits_onset',), multi='क्या आपके किसी भी बच्चे को कभी झटके या दौरे (फ़िट्स) आए हैं?'),
    _q('body_milestones', 'body', "gate", 'क्या बच्चे ने बैठना, चलना या बोलना बाकी बच्चों से काफी देर से सीखा?', 'late sitting, walking, or talking', 'Q02', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे ने बैठना, चलना या बोलना बाकी बच्चों से काफी देर से सीखा?'),
    _q('body_regression', 'body', "gate", 'क्या बच्चा कोई पहले से सीखी हुई बात (जैसे चलना या बोलना) अब भूल गया है?', 'lost a skill the child had already learned', 'Q02', skip=('no_children',), follow=('lost_skill',), multi='क्या आपका कोई भी बच्चा पहले से सीखी हुई कोई बात (जैसे चलना या बोलना) अब भूल गया है?'),
    _q('body_blue', 'body', "gate", 'क्या बच्चे के होंठ या उंगलियाँ कभी नीली पड़ती हैं?', 'lips or fingers turn blue', 'Q02', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे के होंठ या उंगलियाँ कभी नीली पड़ती हैं?'),
    _q('preg_heart', 'body', "gate", 'क्या स्कैन में या जन्म पर डॉक्टर ने बच्चे के दिल में कोई खराबी बताई थी?', 'heart problem on the scan or at birth', 'Q02', skip=('no_children',), follow=(), multi='क्या स्कैन में या जन्म पर डॉक्टर ने आपके किसी भी बच्चे के दिल में कोई खराबी बताई थी?'),
    _q('body_bleeding_bruise', 'body', "gate", 'क्या मामूली चोट पर खून देर तक बहता है, या आसानी से नील पड़ जाते हैं?', 'prolonged bleeding after a small injury, or easy bruising', 'Q04', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे को मामूली चोट पर खून देर तक बहता है, या आसानी से नील पड़ जाते हैं?'),
    _q('body_joint_swelling', 'body', "gate", 'क्या गिरने पर घुटने या जोड़ में सूजन आती है जो कुछ दिनों में अपने आप ठीक हो जाती है?', 'knee or joint swelling after a fall that settles in a few days', 'Q04', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे के घुटने या जोड़ में गिरने पर सूजन आती है जो कुछ दिनों में ठीक हो जाती है?'),
    _q('mod_tooth_bleed', 'body', "gate", 'क्या दूध का दाँत टूटने पर खून बहुत देर तक बहता रहा था?', 'prolonged bleeding when a milk tooth broke; only if a child is 5 or older', 'Q04', skip=('no_children', 'no_child_5_plus'), follow=(), multi='क्या आपके किसी भी बच्चे का दूध का दाँत टूटने पर खून बहुत देर तक बहता रहा था?'),
    _q('mod_heavy_periods', 'body', "gate", 'क्या आपकी बेटी को पीरियड्स में बहुत ज़्यादा खून आता है?', 'very heavy periods; only if a girl is 11 or older', 'Q04', skip=('no_children', 'no_daughter_11_plus'), follow=(), multi='क्या आपकी किसी बेटी को पीरियड्स में बहुत ज़्यादा खून आता है?'),
    _q('body_transfusion', 'body', "gate", 'क्या बच्चे को कभी खून चढ़ाना पड़ा है?', 'the child ever needed a blood transfusion', 'Q05', skip=('no_children',), follow=('transfusion_count',), multi='क्या आपके किसी भी बच्चे को कभी खून चढ़ाना पड़ा है?'),
    _q('body_blood_count', 'body', "gate", 'क्या डॉक्टर ने कभी बच्चे का खून बहुत कम या असामान्य बताया है?', 'a doctor said the blood count was very low or abnormal', 'Q05', skip=('no_children',), follow=(), multi='क्या डॉक्टर ने कभी आपके किसी भी बच्चे का खून बहुत कम या असामान्य बताया है?'),
    _q('body_limb_pain', 'body', "gate", 'क्या बच्चे के हाथ-पैर में बार-बार इतना तेज़ दर्द होता है कि अस्पताल जाना पड़े?', 'repeated severe limb pain that needs hospital', 'Q05', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे के हाथ-पैर में बार-बार इतना तेज़ दर्द होता है कि अस्पताल जाना पड़े?'),
    _q('body_tummy', 'body', "gate", 'क्या बच्चे का पेट धीरे-धीरे बड़ा हो रहा है (लिवर या तिल्ली बढ़ना)?', 'tummy slowly getting bigger', 'Q06', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे का पेट धीरे-धीरे बड़ा हो रहा है (लिवर या तिल्ली बढ़ना)?'),
    _q('body_lumps_birth', 'body', "gate", 'क्या जन्म से बच्चे के शरीर पर कोई सूजन या गाँठें हैं?', 'swellings or lumps on the body since birth', 'Q06', skip=('no_children',), follow=(), multi='क्या जन्म से आपके किसी भी बच्चे के शरीर पर कोई सूजन या गाँठें हैं?'),
    _q('body_growth', 'body', "gate", 'क्या बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है?', 'height or weight growing much less than other children', 'Q07', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है?'),
    _q('body_proportion', 'body', "gate", 'क्या बच्चे के हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं?', 'arms or legs very short compared with the body', 'Q07', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे के हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं?'),
    _q('body_thumb', 'body', "gate", 'क्या बच्चे की कोई उंगली या अंगूठा छोटा है, या कम है?', 'a small or missing finger or thumb', 'Q07', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे की कोई उंगली या अंगूठा छोटा है, या कम है?'),
    _q('gate_bones_fractures', 'body', "gate", 'क्या मामूली गिरने पर बच्चे की हड्डी टूटी है, या दवा के बाद भी पैर टेढ़े हैं?', 'fracture after a minor fall, or bowed legs after treatment', 'Q08', skip=('no_children',), follow=('fracture_count_or_bowing',), multi='क्या मामूली गिरने पर आपके किसी भी बच्चे की हड्डी टूटी है, या दवा के बाद भी पैर टेढ़े हैं?'),
    _q('gate_urine_kidney', 'body', "gate", 'क्या बच्चा बार-बार बहुत ज़्यादा पेशाब करता और पानी पीता है, या गुर्दे/पथरी की दिक्कत है?', 'excess urine and thirst, or a kidney or stone problem', 'Q09', skip=('no_children',), follow=('urine_or_stone_detail',), multi='क्या आपका कोई भी बच्चा बार-बार बहुत ज़्यादा पेशाब करता और पानी पीता है, या गुर्दे/पथरी की दिक्कत है?'),
    _q('body_private_parts', 'body', "gate", 'क्या बच्चे के गुप्त अंग वैसे नहीं दिखते जैसे होने चाहिए?', 'private parts do not look as they should', 'Q10', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे के गुप्त अंग वैसे नहीं दिखते जैसे होने चाहिए?'),
    _q('body_puberty', 'body', "gate", 'क्या 10 साल से बड़े बच्चे में बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं?', 'puberty much earlier or later; only if a child is 10 or older', 'Q10', skip=('no_children', 'all_under_10'), follow=(), multi='क्या 10 साल से बड़े आपके किसी बच्चे में बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं?'),
    _q('ill_fasting', 'illness', "gate", 'क्या खाना छूटने या हल्की बीमारी में बच्चा बहुत सुस्त पड़ जाता है या बार-बार उल्टी करता है?', 'very sleepy or repeated vomiting when a meal is missed or during a mild illness', 'Q01', skip=('no_children',), follow=(), multi='क्या खाना छूटने या हल्की बीमारी में आपका कोई भी बच्चा बहुत सुस्त पड़ जाता है या बार-बार उल्टी करता है?'),
    _q('ill_odor', 'illness', "gate", 'क्या बच्चे के पेशाब या पॉटी में कभी अजीब बदबू आती है?', 'urine or stool smells unusual', 'Q01', skip=('no_children',), follow=(), multi='क्या आपके किसी भी बच्चे के पेशाब या पॉटी में कभी अजीब बदबू आती है?'),
    _q('ill_admissions', 'illness', "gate", 'क्या बच्चे को बार-बार अस्पताल में भर्ती करना पड़ा है?', 'admitted to hospital again and again', 'Q03', skip=('no_children',), follow=('admissions_count', 'admissions_reason'), multi='क्या आपके किसी भी बच्चे को बार-बार अस्पताल में भर्ती करना पड़ा है?'),
    _q('ill_iv_antibiotics', 'illness', "gate", 'क्या बार-बार इंफेक्शन के लिए बच्चे को नस में (IV ड्रिप से) दवा लगी है?', 'repeated infections that needed medicine through a vein', 'Q03', skip=('no_children',), follow=(), multi='क्या बार-बार इंफेक्शन के लिए आपके किसी भी बच्चे को नस में (IV ड्रिप से) दवा लगी है?'),
    _q('preg_conditions', 'pregnancy', "gate", 'क्या किसी प्रेग्नेंसी में आपको हाई बीपी, शुगर या थायरॉइड रहा था?', 'high BP, sugar, or thyroid in a pregnancy', 'Antenatal', skip=('unmarried', 'about_to_marry', 'ever_pregnant_no'), follow=('preg_cond_detail',), multi=''),
    _q('preg_scan', 'pregnancy', "gate", 'क्या किसी प्रेग्नेंसी के सोनोग्राफी स्कैन में कोई खराबी बताई गई थी?', 'a pregnancy scan showed a problem', 'Antenatal', skip=('unmarried', 'about_to_marry', 'ever_pregnant_no'), follow=('scan_detail',), multi=''),
    _q('preg_transfusion', 'pregnancy', "gate", 'क्या किसी प्रेग्नेंसी के दौरान आपको खून चढ़ाना पड़ा था?', 'blood transfusion during a pregnancy', 'Antenatal', skip=('unmarried', 'about_to_marry', 'ever_pregnant_no'), follow=(), multi=''),
    _q('preg_delivery', 'pregnancy', "gate", 'क्या किसी डिलीवरी के समय कोई बड़ी दिक्कत आई थी?', 'major trouble at a delivery', 'Antenatal', skip=('unmarried', 'about_to_marry', 'ever_pregnant_no'), follow=('delivery_detail',), multi=''),
    _q('preg_thalassemia', 'pregnancy', "gate", 'क्या आपकी या आपके पति की थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?', 'thalassemia minor or trait on a blood test', 'Antenatal', skip=(), follow=(), multi=''),
    _q('fam_related', 'family', "gate", 'क्या आप और आपके पति शादी से पहले आपस में रिश्तेदार थे, जैसे कज़िन या एक ही खानदान से?', 'she and her husband are related, such as cousins', 'Q11', skip=('unmarried',), follow=('related_how',), multi=''),
    _q('fam_miscarriage', 'family', "gate", 'क्या आपका कभी गर्भ गिरा है (मिसकैरेज), या गिरवाना पड़ा था?', 'miscarriage or an ended pregnancy', 'Q12', skip=('unmarried', 'about_to_marry', 'ever_pregnant_no'), follow=('miscarriage_count', 'miscarriage_month'), multi=''),
    _q('fam_infant_death', 'family', "gate", 'क्या परिवार में किसी छोटे बच्चे की 1 साल से कम उम्र में मौत हुई है?', 'a baby in the family died before 1 year', 'Q12', skip=(), follow=('infant_death_age', 'infant_death_cause'), multi=''),
    _q('fam_fits_delay', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े को झटके (फ़िट्स), देर से सीखने, या सीखी बात भूलने की समस्या है?', 'fits, late learning, or lost skills in the family', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_transfusion', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े को बार-बार खून चढ़ाना पड़ता है?', 'someone in the family needs blood again and again', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_bleeding', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की समस्या है?', 'someone in the family bleeds too much after injury', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_blind_deaf', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े को बचपन या कम उम्र से अंधेपन या बहरेपन की समस्या है?', 'childhood blindness or deafness in the family', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_cancer', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े को कैंसर हुआ है?', 'cancer in the family', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_heart_kidney', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े को दिल या गुर्दे (किडनी) की गंभीर बीमारी है?', 'serious heart or kidney disease in the family', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_bp_stroke', 'family', "gate", 'क्या परिवार में 40-50 साल से कम उम्र के किसी व्यक्ति को बीपी, शुगर या लकवा हुआ है?', 'BP, sugar, or paralysis before age 40-50 in the family', 'Q11', skip=(), follow=('bp_who', 'bp_age'), multi=''),
    _q('fam_maternal_bleeding', 'family', "gate", 'क्या आपके मायके की तरफ किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की दिक्कत है?', "on her parents' side: heavy bleeding", 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_maternal_infections', 'family', "gate", 'क्या आपके मायके की तरफ किसी बच्चे या बड़े को बार-बार गंभीर इंफेक्शन की दिक्कत है?', "on her parents' side: repeated serious infections", 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_maternal_eczema', 'family', "gate", 'क्या आपके मायके की तरफ किसी बच्चे या बड़े को त्वचा पर सूखे खुजली वाले धब्बों की दिक्कत है?', "on her parents' side: dry itchy skin patches", 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_lumps', 'family', "gate", 'क्या परिवार में किसी बच्चे या बड़े के शरीर पर सूजन या गाँठें हैं?', 'swellings or lumps on someone in the family', 'Q11', skip=(), follow=('fam_who',), multi=''),
    _q('fam_same_problem', 'family', "gate", 'क्या परिवार में किसी और को भी आपके बच्चे जैसी ही दिक्कत है?', 'someone else in the family has the same problem as her child', 'Q11', skip=('no_children', 'no_family_concern'), follow=('fam_who',), multi=''),
    _q('close_anything_else', 'close', 'open_capture', 'क्या इसके अलावा डॉक्टर के लिए कुछ और बताना है जो आपके मन में हो?', 'anything else she wants the doctor to know', skip=(), follow=("open_concern_detail",)),
)


_BY_ID: dict[str, QuestionSpec] = {spec.id: spec for spec in CATALOG}
_SPINE_KINDS = frozenset({"profile", "bridge", "gate", "open_capture"})
_CHILD_GROUPS = frozenset({"birth", "body", "illness"})


def _validate() -> None:
    ids = [spec.id for spec in CATALOG]
    dupes = sorted({qid for qid in ids if ids.count(qid) > 1})
    if dupes:
        raise RuntimeError(f"duplicate catalog ids: {dupes}")
    if len(FOLLOWUP_HI) != 25:
        raise RuntimeError(f"expected 25 follow-ups, found {len(FOLLOWUP_HI)}")
    for spec in CATALOG:
        if spec.kind not in _SPINE_KINDS:
            raise RuntimeError(f"{spec.id} has kind {spec.kind}")
        if spec.opens:
            raise RuntimeError(f"{spec.id} still opens {spec.opens}")
        for fid in spec.followups:
            if fid not in FOLLOWUP_HI:
                raise RuntimeError(f"{spec.id} followup {fid} has no Hindi")
        # fam_same_problem skips when there is no child, but section 4.3 leaves
        # its question_multi_hi empty: the sentence is already about the family.
        if spec.group in _CHILD_GROUPS and "no_children" in spec.skip_if and not spec.question_multi_hi.strip():
            raise RuntimeError(f"{spec.id} needs question_multi_hi")


_validate()


def get_spec(qid: str) -> QuestionSpec:
    try:
        return _BY_ID[qid]
    except KeyError:
        raise KeyError(qid) from None


def active_specs() -> list[QuestionSpec]:
    return [spec for spec in CATALOG if spec.enabled]


def spine_ids() -> list[str]:
    """Ask order. ever_pregnant is not a row; the cursor inserts it."""
    return [spec.id for spec in active_specs() if spec.kind in _SPINE_KINDS]

