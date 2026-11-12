# SHISHU v3 — Requirements & Clinical Baseline

Status: Locked baseline for the `next` branch (Mumbai clinical pilot toward PMF).
Production freeze: `main` (`https://shishu-production.up.railway.app`) stays untouched during `next` development. `next` deploys to an isolated Railway staging project/service.

---

## 1. Purpose & Design Objective

SHISHU is a mobile-web vernacular Hindi voice pre-consultation screening app for rare and inherited childhood disorders (built on ICMR NRROID Sections 6-7 & Annexure 1 and Dr. Poulami's Nov 4 clinical triage protocol).

- **Clinical setting**: A doctor's office or clinic waiting room (or between visits) suggests SHISHU to a mother. She opens the browser link on Android Chrome, gives informed consent, speaks in everyday Hindi, and shares the resulting Hindi PDF with her doctor or ASHA/ANM health worker via WhatsApp.
- **Engineering objective**: Maintainable, simple code (~1,600 LOC total core stack; ~520 LOC clinical brain) that supports rapid, isolated edits in single files as real clinicians and mothers give feedback for 10s of pilot users toward PMF.

---

## 2. Hard Safety & Clinical Guardrails (Never Violate)

1. **Non-Diagnostic Boundary**:
   - Never utter, print, or suggest any genetic/rare disease name or differential (e.g., Duchenne, Pompe, MPS, Wiskott-Aldrich, SCID, Hemophilia, Gaucher, SMA) to the mother in voice or PDF.
   - Never say "nothing concerning was found" or "you do not need to see a doctor."
   - Never advise starting, stopping, or changing any medicine.
   - State upfront on Page 1 of the Mother PDF that this is not a diagnosis (`यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है। केवल डॉक्टर जाँच करके सही सलाह दे सकते हैं।`).
   - Plain-language 2-sentence explanations are allowed only if the mother asks what *BCG* or *Thalassemia* means.
2. **Active Emergency Halt**:
   - If the caller describes an active danger sign happening **right now** (not breathing, severe fast breathing, blue lips now, unconscious/unresponsive, unstoppable fits, unstoppable bleeding, or says it is an emergency now): immediately stop all screening questions, instruct her calmly in Hindi to go to the nearest hospital or call `108`, mark close reason `emergency`, and generate an emergency-banner partial PDF.
   - Past historical events (e.g., "he fainted once last year") are recorded as history, not treated as an active emergency.
3. **Honest Epistemic States (Unanswered != No)**:
   - Every clinical question has one of five explicit states:
     - `concern`: positive / affirmative clinical concern (`हाँ`).
     - `clear`: negative / reassuring answer (`नहीं`).
     - `unknown`: caller does not know or does not remember (`पता नहीं`).
     - `not_asked`: call ended before the question was reached, or caller never answered.
     - `not_applicable`: skipped deterministically by demographic rule (never printed on the Mother PDF).
   - Never default an unasked or unanswered question to `clear` (`नहीं`).
4. **Uniform Question Polarity**:
   - Every clinical yes/no question is phrased in Hindi so that **`हाँ` = screened clinical concern** and **`नहीं` = no concern**.
   - Never ask compound two-directional questions (never ask *"पूरे 9 महीने पर हुआ या समय से पहले?"* where *"हाँ"* is ambiguous).
5. **Voice & Tone**:
   - Single warm female voice (`Kore`) in simple spoken Hindi throughout the entire call; never shift to a male/deeper register even if a father/male caller speaks.
   - Strictly one short question per turn, then stop and wait. Never answer on behalf of the caller.
   - Non-judgmental, respectful tone regardless of parity (5+ children), consanguinity, miscarriage/abortion, unmarried status, or home delivery.
   - Vernacular phrasing over clinical jargon: *kaanch ka box* (NICU), *neeli light* (phototherapy), *naal* (umbilical cord), *bagal mein gaanth* (axillary lymphadenopathy), *jhatke* (seizures), *seekhi baat bhool jaana* (neuroregression), *tilli* (spleen), *neel padna* (bruising).

---

## 3. Call Lifecycle & End Conditions

1. **Consent Gate**: Call button stays disabled until the user ticks the bilingual consent checkbox linking to `/static/SHISHU-consent.pdf` (Hindi first, English second; paid Gemini API tier only so audio/text are never used for model training).
2. **Silence & Time Watchdogs**:
   - `15s` caller silence -> polite Hindi check-in + simpler repeat of current question.
   - `30s` caller silence -> polite incomplete-call closing + end call (`close_reason = "silence"`).
   - `19m` soft limit / `20m` hard ceiling -> polite time-limit closing + end call (`close_reason = "time_limit"`).
   - Caller asks to stop (*"अभी बंद कर दो"*, *"बाद में बात करेंगे"*) -> polite stop closing + end call (`close_reason = "caller_stop"`). Brief holds (*"रुको"*, *"एक मिनट"*) wait patiently.
3. **Server-Owned Call Completion**:
   - When the cursor reaches `closing`, the server starts report synthesis from an immutable snapshot during the spoken outro (`report_service.py` pattern) and closes the session automatically once closing audio finishes playing—never requiring a second tool call from the live model.

---

## 4. Demographic Gating & Multi-Child Rules

Captured during the 4-step `intro` before clinical gates:
- `caller_name`: string (never invented; re-ask gently if caller only says *"मेरा नाम..."*).
- `marital_status`: `married | about_to_marry | unmarried`.
- `children`: list of `{id: "child_1", age_years: float, age_label_hi: str, sex: "male"|"female"|"unknown", name: str|None}`. If caller gives count without ages, ask ages before advancing.
- `ever_pregnant`: `bool` (inferred `True` if `len(children) > 0`; asked explicitly before pregnancy/miscarriage gates if `len(children) == 0` and married).

Deterministic `skip_if` rules (`state = "not_applicable"`):
- `no_children` (`len(children) == 0`): skips all child gates and child modules (`birth`, `body`, `illness`), plus `fam_same_problem`.
- `unmarried` (`marital_status == "unmarried"`): skips `intro_children` (if not volunteered), all child gates/modules, pregnancy history (`preg_*` except optional carrier awareness `gate_thalassemia`), `fam_related`, and `fam_miscarriage` (unless `ever_pregnant` is `True`).
- `ever_pregnant_no` (`ever_pregnant is False`): skips `gate_pregnancy`, `preg_*` delivery/scan/transfusion items, and `fam_miscarriage`. Keeps `gate_thalassemia` and family pedigree gates.
- `all_under_10` (every child `< 10` years): skips puberty items (`mod_puberty`).
- `no_child_5_plus` (every child `< 5` years): skips milk-tooth fall bleeding (`mod_tooth_bleed`).
- `no_daughter_11_plus` (no female child `>= 11` years): strictly forbids asking menstruation/heavy-period questions (`mod_heavy_periods`).
- **Multi-child scope**: When `len(children) > 1`, phrase child questions for all children (*"आपके किसी भी बच्चे को..."*). When a gate or module item is answered `concern`, record which child (`child_1`, `child_2`, etc., mapping *"बड़े वाले"* -> oldest, *"छोटे वाले"* -> youngest, or asking age if an unmapped name is given).
- **Family clarity**: Always specify *"परिवार में किसी बच्चे या बड़े को..."* (or *"एक साल से कम उम्र के बच्चे"* for infant death) so the caller never guesses whether family questions mean kids or adults.

---

## 5. Declarative Gate-First Clinical Catalog (15 Gates + Triggered Modules)

All items live in one declarative file (`code/shishu-v2/triage_catalog.py`) with `enabled: bool = True` so medical experts can toggle, reorder, or flatten items without code changes elsewhere.

### 5.1 Intro & Open Capture
| ID | Kind | Hindi Question | Notes |
|---|---|---|---|
| `intro_name` | `profile` | नमस्ते, मैं शिशु हूँ। आपका नाम बताइएगा प्लीज़? | Captures `caller_name` |
| `intro_married` | `profile` | क्या आपकी शादी हो गई है, या होने वाली है? | Captures `marital_status` |
| `intro_children` | `profile` | आपके कितने बच्चे हैं, और उनकी उम्र क्या-क्या है? | `skip_if=("unmarried",)`; captures `children` |
| `intro_bridge` | `bridge` | मैं आपके बच्चे, प्रेग्नेंसी और परिवार के बारे में कुछ आसान सवाल पूछूँगी। आप छोटा या लंबा जैसे चाहें जवाब दे सकती हैं, और कुछ समझ न आए तो बेझिझक पूछ सकती हैं। ठीक है? | Explains rules |
| `close_anything_else` | `open_capture` | क्या इसके अलावा डॉक्टर के लिए कुछ और बताना है जो आपके मन में हो? | Asked once before closing |

### 5.2 The 15 Wide Screening Gates (Asked of Applicable Callers; ~3-4 min Normal Call)

Every gate is phrased so `हाँ` = concern (opens its `opens` sub-questions + `followups`), `नहीं` = clear (moves immediately to next gate).

| Gate ID | NRROID | `skip_if` | Uniform-Polarity Hindi Question (`हाँ` = Concern) | `opens` / `followups` on `हाँ` |
|---|---|---|---|---|
| `gate_birth_neonatal` | Q03 / Neonatal | `no_children` | क्या किसी बच्चे का जन्म ऑपरेशन से या समय से बहुत पहले हुआ था, या जन्म के बाद काँच के बक्से (NICU) या पीलिया की नीली लाइट में रखना पड़ा था? | Opens: `birth_mode_detail`, `birth_term`, `birth_nicu`, `birth_sudden` |
| `gate_cord_bcg` | Q03 (M2) | `no_children` | क्या जन्म के बाद नाल गिरने में 3 हफ्ते से ज़्यादा लगे, नाल से ज़्यादा खून आया, या BCG के टीके की जगह पर बड़ा फोड़ा या बगल में गाँठ हुई? | Opens: `birth_cord_bleed`, `birth_cord_3weeks`, `birth_bcg` |
| `gate_metabolic_crisis` | Q01 (M1) | `no_children` | क्या खाना छूटने पर, हल्की बीमारी में, या कुछ खास खाने पर बच्चा बहुत सुस्त हो जाता है, बार-बार उल्टी करता है, या पेशाब-पॉटी में अजीब बदबू आती है? | Opens: `ill_fasting`, `ill_odor` |
| `gate_milestones_neuro` | Q02 (M4/M1/M5) | `no_children` | क्या बच्चे को कभी झटके (फ़िट्स) आए हैं, या बैठना-चलना-बोलना देर से सीखा, या कोई सीखी हुई बात अब भूल गया है? | Opens: `body_fits_milestones`, `body_regression` |
| `gate_sensory_cleft_heart` | Q02 / Neonatal | `no_children` | क्या बच्चे को सुनने या देखने में कोई दिक्कत है, होंठ या तालू कटा था, या होंठ-उंगलियाँ कभी नीली पड़ती हैं? | Opens: `birth_hearing_vision`, `birth_cleft`, `body_blue`, `preg_heart` |
| `gate_infections` | Q03 (M2) | `no_children` | क्या बच्चे को बार-बार अस्पताल में भर्ती करना पड़ा है, या बार-बार इंफेक्शन के लिए नस में (IV) एंटीबायोटिक दवा लगी है? | Opens: `ill_admissions`, `ill_iv_antibiotics` |
| `gate_bleeding_joints` | Q04 (M3) | `no_children` | क्या चोट लगने पर खून देर तक बहता है, आसानी से नील पड़ते हैं, या गिरने पर घुटने-जोड़ में सूजन आती है जो कुछ दिनों में ठीक हो जाती है? | Opens: `body_bleeding_bruise`, `body_joint_swelling`, `mod_tooth_bleed` (`>=5y`), `mod_heavy_periods` (`girl >=11y`) |
| `gate_transfusion_anemia` | Q05 (M3) | `no_children` | क्या बच्चे को कभी खून चढ़ाना पड़ा है, डॉक्टर ने खून कम-ज़्यादा बताया है, या हाथ-पैर में बार-बार तेज़ दर्द के लिए अस्पताल जाना पड़ता है? | Opens: `body_transfusion`, `body_blood_count`, `body_limb_pain` |
| `gate_tummy_lumps` | Q06 (M5/M1/M3) | `no_children` | क्या बच्चे का पेट धीरे-धीरे बड़ा हो रहा है (लिवर या तिल्ली बढ़ना), या जन्म से शरीर पर सूजन या गाँठें हैं? | Opens: `body_tummy`, `body_lumps_birth` |
| `gate_growth_shape` | Q07 (M5/M7) | `no_children` | क्या बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है, हाथ-पैर शरीर के मुकाबले छोटे हैं, या कोई उंगली-अंगूठा छोटा या कम है? | Opens: `body_growth`, `body_proportion`, `body_thumb` |
| `gate_bones_fractures` | Q08 (M5/M6) | `no_children` | क्या मामूली गिरने पर कभी बच्चे की हड्डी टूटी है, या विटामिन-डी और कैल्शियम देने के बाद भी पैर टेढ़े या मुड़े हुए हैं? | Followups: `fracture_count_or_bowing` |
| `gate_urine_kidney` | Q09 (M6) | `no_children` | क्या बच्चा बार-बार बहुत ज़्यादा पेशाब करता और बहुत पानी पीता है, या कभी पेशाब में पथरी या गुर्दे की दिक्कत बताई गई है? | Followups: `urine_or_stone_detail` |
| `gate_endocrine_private` | Q10 (M7) | `no_children` | क्या बच्चे के गुप्त अंग वैसे नहीं दिखते जैसे होने चाहिए, या (10 साल से बड़े बच्चे में) बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं? | Opens: `body_private_parts`, `body_puberty` (`>=10y`) |
| `gate_pregnancy_thal` | Antenatal | `ever_pregnant_no` (except Thal) | क्या किसी प्रेग्नेंसी में बीपी-शुगर-थायरॉइड, स्कैन में खराबी, प्रेग्नेंसी में खून चढ़ना, डिलीवरी में दिक्कत, या आपकी/पति की थैलेसीमिया जाँच में माइनर/ट्रेट बताया गया? | Opens: `preg_conditions`, `preg_scan`, `preg_transfusion`, `preg_delivery`, `preg_thalassemia` (Note: if `ever_pregnant_no`, ask standalone `preg_thalassemia` only) |
| `gate_family_pedigree` | Q11 & Q12 + Family | `()` | क्या आप और पति शादी से पहले रिश्तेदार थे, कभी गर्भ गिरा, परिवार में किसी छोटे बच्चे की 1 साल से कम उम्र में मौत हुई, या परिवार में किसी बच्चे या बड़े को खून चढ़ने, झटके, अंधेपन, बहरेपन, कैंसर, दिल-गुर्दे या कम उम्र में बीपी-लकवे की बीमारी है? | Split into 3 clean family sub-gates so cognitive load stays low: (1) `fam_related` (`skip_if="unmarried"`), (2) `fam_loss_gate` (`fam_miscarriage` + `fam_infant_death`), (3) `fam_conditions_gate` (opens `fam_fits_delay`, `fam_transfusion`, `fam_bleeding`, `fam_blind`, `fam_deaf`, `fam_cancer`, `fam_heart`, `fam_kidney`, `fam_bp_stroke` [with immediate vs distant relative + age], `fam_maternal_side`, `fam_lumps`, `fam_same_problem`) |

*(Note: Splitting Gate 15 into 3 short spoken questions brings a normal healthy mother's call to **17 short questions (~3.5–4.5 minutes)** while preserving 100% of both Oct 29 NRROID gates and Nov 4 Dr. Poulami questions.)*

---

## 6. Two-Sided Output & Admin Contracts

### 6.1 Output 1: Mother Hindi PDF (`mother_report.py` + `report_service.py`)
- **Format**: A4 WeasyPrint PDF using embedded `NotoSansDevanagari.ttf`, navy/gold/grey palette, SHISHU logo.
- **Page 1**:
  - Compact header: Mother's Devanagari name + children's ages/sexes (`बेटा`/`बेटी`), Call Ref (`SHISHU-<id>`), IST date, clock time, duration, complete/incomplete status.
  - Boxed non-diagnosis notice right under header.
  - Emergency (`108`) or incomplete-call banner if applicable.
  - **Actionable Questions (`3–6` numbered items in Hindi)**: First-person questions she can ask her doctor or ASHA/ANM based on her `concern` answers (plus routine questions if fewer than 3 concerns). Saved to `calls.summary_mother` so Admin shows the exact same text.
  - **"इस कॉल में जो हाँ था"**: Clear Devanagari sentences summarizing only confirmed `concern` items with her specific details (zero English character-by-character transliteration).
  - Universal Thalassemia screening box at bottom of Page 1.
- **Page 2 (`विवरण`)**:
  - Grouped list of questions answered `clear` (`नहीं`), `unknown` (`पता नहीं`), and—on incomplete calls—`not_asked` (`पूछा नहीं गया`).
  - **`not_applicable` items are omitted completely** (never shown as `पता नहीं` or `पूछा नहीं गया`).
- **Delivery & Privacy**:
  - Exposed on `call.html` Thanks screen only after atomic PDF write succeeds; shared via `navigator.share({files: [pdf]})` on Android or downloaded as fallback.
  - Protected via unguessable UUID4 `/api/calls/by-token/{share_token}/mother.pdf` (plus Admin session access), preventing same-name patient collisions.

### 6.2 Output 2: Clinician English Note & PMF Dashboard (`/admin`)
- **8 Standardized Markdown Headings**:
  1. `### Patient Profile & Chief Complaint`
  2. `### Positive Red Flags & NRROID Gate Alerts`
  3. `### Birth & Neonatal History`
  4. `### Development, Growth & Physical Features`
  5. `### Systemic, Infections & Hematology`
  6. `### Antenatal (Pregnancy) History`
  7. `### Family & Pedigree History`
  8. `### Pertinent Negatives`
- **Audio Verification**: Time-aligned 2-channel WAV (`Ch1` Mother, `Ch2` SHISHU) built off the asyncio event loop thread.
- **Pilot & PMF Loop**:
  - URL tag `/?clinic=<slug>` recorded on each call and filterable in `/admin`.
  - Telemetry tracks `share_clicked` and `dropoff_qid`.
  - 1-click Clinician Review on each call in `/admin`: rating (`useful | missed_flag | false_alarm`) + free-text clinician feedback note stored in SQLite.

---

## 7. Codebase Modularity Rules

1. **Single-file question edits**: Adding, editing, disabling, or changing follow-ups for a clinical question touches **only** `code/shishu-v2/triage_catalog.py`.
2. **Thin live voice loop**: `gemini_live.py` handles WebSocket, audio streaming, session resumption, and watchdogs—zero clinical regexes, zero mid-call secondary LLM calls.
3. **Single post-call synthesis**: One schema-typed `gemini-3.8-flash` pass (`post_call.py`) over the `qid`-tagged transcript produces both Mother Hindi takeaways and Clinician English sections so the two reports never contradict each other.
4. **Environment & Ops**:
   - Secrets stay exclusively in root `.env`.
   - Python virtualenv lives in `~/pyenv/shishu` managed with `uv`.
   - Long-running servers and persona test suites run inside `tmux`.
