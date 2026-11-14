# SHISHU v4 — Requirements, FormBharo State Engine & UI Quick-Reply Spec

Status: Locked baseline for branch `next_v2` (branched from `next`, keeping working stereo audio, post-call PDF/Doctor synthesis, UUID token auth, Wake Lock, and Admin Clinician Review untouched).
Production freeze: `main` (`https://shishu-production.up.railway.app`) stays completely untouched. `next_v2` deploys only to `https://shishu-staging.up.railway.app` (`project f196b873-8b92-4877-a764-d435ce52af39`).

---

## 1. Why v4 Exists (Root Causes from Staging Calls #3–#8 + FormBharo Principles)

Real staging calls exposed 5 structural flaws in v3 (`next`) that v4 eliminates from first principles, applying the proven hybrid architecture from **FormBharo** (*Dalmia, Midha, Doshi 2026*):

1. **No more clubbed cognitive overload or hidden clinical modules**:
   - v3 crammed 4–9 conditions into single gates (`gate_pregnancy_thal`, `gate_cord_bcg`, `fam_conditions_gate`) and hid critical questions (`preg_heart`, `mod_heavy_periods`, `mod_tooth_bleed`, `birth_jaundice`, `fam_maternal_side`) behind unrelated gates.
   - **v4 Rule**: Every screening item is **one short, atomic yes/no question** (`हाँ` = concern, `नहीं` = clear). Zero `opens` sub-trees, zero hidden questions.
2. **Same-size `नहीं (No)` and `हाँ (Yes)` quick-tap buttons flanking `बंद (End)`**:
   - Spoken-only `"नहीं"` across ~48 atomic questions takes too long and risks noisy-OPD mishearing.
   - **v4 Rule**: During yes/no questions, two circular buttons of the **exact same size (`84×84px`)** as the red `बंद (End)` button appear on its **left (`नहीं / No`)** and **right (`हाँ / Yes`)**. The mother can either speak naturally OR tap `नहीं` / `हाँ` to advance in `<100ms` (immediately flushing bot playback and speaking the next question).
3. **FormBharo non-monotonic state & bounded retries (zero infinite loops)**:
   - v3 locked into an infinite 5-turn loop on `pending_followup_idx = -1` when a caller said *"हाँ"* by mistake and then said *"नहीं, कुछ नहीं हुआ"* 4 times, and threw away demographic corrections after `intro_married`.
   - **v4 Rule**:
     - **Bounded Retry Cap (`MAX_STEP_ATTEMPTS = 2`)**: Any pending sub-step (`which_child` or `followup`) that receives 2 consecutive non-resolving turns automatically marks that sub-slot `"unknown"` and advances. Infinite loops are impossible.
     - **Retraction Transition**: Saying *"नहीं / कुछ नहीं हुआ / नॉर्मल था"* (`outcome = "retract"`, or `outcome = "clear"` with empty `child_ref` while on `which_child`) reverts `active_qid` from `"concern"` to `"clear"` and advances.
     - **Anytime Profile Correction**: `save_profile` works on **any** turn in the call, immediately re-evaluating all `skip_if` branches (`_recompute`) and advancing past newly skipped questions.
     - **Pre-Answered Detail Absorption**: If the mother states the child and/or follow-up detail in her initial `"हाँ"` sentence, `apply_turn` absorbs `child_ref` and `followup_answered=True` without re-asking what she already said.
4. **Full N-child support (including middle children — `बीच वाले`)**:
   - v3 hardcoded 2 children (*"बड़े वाले या छोटे वाले?"*) and broke on 3 children (`2y, 5y, 8y`).
   - **v4 Rule**: `triage_catalog.py` stores explicit `question_hi` (1 child) and `question_multi_hi` (2+ children) text with zero Hindi regex assembly. `_which_child_hi(children)` lists all children by age, sex, or name from oldest to youngest. When a profile has two or more children, the tool result includes `children_roster`, and Gemini passes an exact `id` (`child_1`..`child_n`) or `"all"`. `_resolve_child_ids` matches only that exact `id`, a stored name, a closed-world ordinal token, a unique sex token, or `"all"`.
5. **Clean Live Voice Turn-Taking (zero self-interruptions, pause-safe VAD)**:
   - v3 put `"wait": True` in `FunctionResponse` (stalling audio in Call #3) and then fired `send_realtime_input` 1ms after `send_tool_response` (causing 26 self-interruptions in Call #8).
   - **v4 Rule**: `FunctionResponse` omits `"wait": True` and carries a direct speak directive. When a tool turn finishes without bot audio, `gemini_live.py` sends the speak cue on `server_content.turn_complete`. A `2.5s` fallback timer acts only as a safety net when `turn_complete` never arrives. An accepted `quick_reply` drops leftover audio and captions from the interrupted sentence, and holds any late `record_turn` as `stale_turn`. `vad_silence_ms = 1000` and `vad_end_sensitivity = "LOW"` preserve mid-sentence pauses, and `speech_end` refreshes `quiet_from` for the 15s check-in and 30s close.

---

## 2. Hard Clinical & Safety Guardrails (Never Violate)

1. **Non-Diagnostic Boundary**:
   - Never utter, print, or suggest any genetic/rare disease name (Duchenne, Pompe, MPS, Wiskott-Aldrich, SCID, Hemophilia, Gaucher, SMA, ड्यूशेन, हीमोफीलिया, पॉम्पे, गॉशर, इम्यूनोडेफिशिएंसी, म्यूकोपॉलीसेकेराइडोसिस) in voice or PDF.
   - Never say "nothing concerning was found" or "you do not need a doctor." Never advise starting/stopping/changing medicine.
   - Plain 2-sentence Hindi explanations are permitted if she asks what *BCG*, *NICU*, *प्रसव*, or *थैलेसीमिया* means.
2. **Active Emergency Halt**:
   - Active danger sign **right now** (not breathing, severe fast breathing, blue lips now, unconscious, unstoppable fits, unstoppable bleeding, or caller says emergency now) -> immediately set `phase = "closing"`, `close_reason = "emergency"`, speak `108` / nearest hospital Hindi instruction, and build emergency-banner partial PDF. Historical past events are recorded as history, not emergencies.
3. **Honest Epistemic States**:
   - `concern` (`हाँ`), `clear` (`नहीं`), `unknown` (`पता नहीं`), `not_asked` (unanswered when call ended early), `not_applicable` (skipped by demographic rule — omitted from Mother PDF). Never default unanswered questions to `clear`.
4. **Uniform Polarity & Voice**:
   - Every clinical question is phrased so **`हाँ` = concern** and **`नहीं` = clear**. (`intro_bridge` is `kind="bridge"` — any affirmative `"हाँ"` / `"ठीक है"` records `clear` and advances).
   - Single warm female voice (`Kore`) in simple everyday Hindi throughout the entire call.

---

## 3. Complete Demographic Branching Matrix (`married` vs `about_to_marry` vs `unmarried`)

Profile fields in `state["profile"]`:
- `caller_name`: `str | None` (reject bare `"मेरा नाम"` fragments -> hold `"name_fragment"`).
- `marital_status`: `"married" | "about_to_marry" | "unmarried" | None` (plain `"नहीं" / "no" / "nahi" / False` on `intro_married` maps to `"unmarried"`; `"होने वाली"` -> `"about_to_marry"`; `"हाँ" / "हो गई" / True` -> `"married"`).
- `children`: `list[{"id": "child_1", "age_years": float, "age_label_hi": str, "sex": "male"|"female"|"unknown", "name": str|None}]`. If `child_count > 0` without all ages, hold `"children_ages"`.
- `ever_pregnant`: `bool | None`:
  - Automatically `True` if `len(children) > 0`.
  - Automatically `False` (never asked!) if `marital_status in ("unmarried", "about_to_marry")` unless explicitly stated `True`.
  - Asked via synthetic yes/no step `ever_pregnant` (`"क्या आप कभी गर्भवती हुई हैं?"`, `expect_yes_no = True`) **only** when `marital_status == "married"`, `len(children) == 0`, and `ever_pregnant is None`.

### Deterministic `skip_if` Rules (`state = "not_applicable"`):

| Rule Token | Condition When Skipped (`True`) |
|---|---|
| `unmarried` | `marital_status == "unmarried"` |
| `about_to_marry` | `marital_status == "about_to_marry"` |
| `no_children` | Children resolved (`marital_status in ("unmarried", "about_to_marry")` or past `intro_children`) AND `len(children) == 0` |
| `ever_pregnant_no` | `ever_pregnant is False` OR `marital_status in ("unmarried", "about_to_marry") and ever_pregnant is not True` |
| `no_child_5_plus` | `no_children` OR no child has `age_years >= 5` |
| `all_under_10` | `no_children` OR no child has `age_years >= 10` |
| `no_daughter_11_plus` | `no_children` OR no child has `sex == "female" and age_years >= 11` |
| `no_family_concern` | No `fam_*` item (excluding `fam_same_problem`) has `state == "concern"` |

### Dynamic Marital Wording Overrides in `call_cursor._question_hi`:
- **`preg_thalassemia`**:
  - `married`: `"क्या आपकी या आपके पति की थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?"`
  - `about_to_marry`: `"क्या आपकी या आपके होने वाले पति की थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?"`
  - `unmarried`: `"क्या आपकी थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?"`
- **`fam_related`**:
  - `married`: `"क्या आप और आपके पति शादी से पहले आपस में रिश्तेदार थे, जैसे कज़िन या एक ही खानदान से?"`
  - `about_to_marry`: `"क्या आप और आपके होने वाले पति पहले से आपस में रिश्तेदार हैं, जैसे कज़िन या एक ही खानदान से?"`
  - `unmarried`: skipped (`not_applicable`).

---

## 4. Complete Atomic Triage Catalog (`code/shishu-v2/triage_catalog.py`)

`QuestionSpec` dataclass fields:
`id`, `group`, `kind` (`"profile" | "bridge" | "gate" | "open_capture"` — keep `"gate"` for all clinical spine items so `mother_report.py` marks unasked items `"पूछा नहीं गया"` on incomplete calls), `question_hi` (1-child / general), `question_multi_hi` (explicit 2+ children phrasing; `""` if non-child item), `need_en`, `nrroid_gate`, `skip_if`, `opens` (default `()`), `followups` (default `()`), `enabled` (`True`).

### 4.1 Follow-Up Prompts (`FOLLOWUP_HI`)
| Follow-up ID | Hindi Prompt (`expect_yes_no = False`) |
|---|---|
| `operation_reason` | ऑपरेशन किस वजह से करना पड़ा था? |
| `term_how_early` | बच्चा पूरे समय से कितने हफ्ते या महीने पहले पैदा हुआ था? |
| `nicu_reason` | काँच के बक्से (NICU) में किस वजह से रखना पड़ा था? |
| `nicu_days` | वहाँ कितने दिन रखना पड़ा था? |
| `jaundice_days` | नीली लाइट में कितने दिन रखना पड़ा था? |
| `sudden_detail` | यह कब हुआ था और डॉक्टर ने क्या बताया था? |
| `fits_onset` | झटके पहली बार किस उम्र में आए थे? |
| `lost_skill` | बच्चा पहले क्या कर लेता था जो अब नहीं कर पाता? |
| `transfusion_count` | खून कितनी बार चढ़ाना पड़ा है? |
| `fracture_count_or_bowing` | हड्डी कितनी बार टूटी, या पैर टेढ़े होने की बात है? |
| `urine_or_stone_detail` | बार-बार पेशाब-प्यास की बात है, या पथरी या गुर्दे की दिक्कत? |
| `admissions_count` | बच्चे को कितनी बार अस्पताल में भर्ती करना पड़ा है? |
| `admissions_reason` | किस बीमारी या वजह से भर्ती करना पड़ा था? |
| `preg_cond_detail` | बीपी, शुगर या थायरॉइड में से क्या था, और क्या डिलीवरी के बाद भी रहा? |
| `scan_detail` | स्कैन में क्या दिक्कत बताई गई थी? |
| `delivery_detail` | डिलीवरी के समय क्या दिक्कत आई थी? |
| `related_how` | आप दोनों का शादी से पहले क्या रिश्ता था? |
| `miscarriage_count` | गर्भ कितनी बार गिरा था? |
| `miscarriage_month` | यह कितने महीने की प्रेग्नेंसी में हुआ था? |
| `infant_death_age` | उस बच्चे की उम्र कितनी थी, और आपसे क्या रिश्ता था? |
| `infant_death_cause` | क्या मौत की कोई वजह पता चली थी? |
| `fam_who` | परिवार में यह दिक्कत किसको है — आपसे उनका क्या रिश्ता है? |
| `bp_who` | यह सगे रिश्तेदार (जैसे माता-पिता, भाई-बहन) को हुआ, या दूर के रिश्तेदार को? |
| `bp_age` | उन्हें यह किस उम्र में हुआ था? |
| `open_concern_detail` | जी बताइए, डॉक्टर के लिए क्या बात नोट करूँ? |

### 4.2 Intro & Open Capture Items
| ID | Group | Kind | `skip_if` | `question_hi` |
|---|---|---|---|---|
| `intro_name` | `intro` | `profile` | `()` | नमस्ते, मैं शिशु हूँ। आपका नाम बताइएगा प्लीज़? |
| `intro_married` | `intro` | `profile` | `()` | क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई? |
| `intro_children` | `intro` | `profile` | `("unmarried", "about_to_marry")` | आपके कितने बच्चे हैं, और उनकी उम्र क्या-क्या है? |
| `intro_bridge` | `intro` | `bridge` | `()` | अब मैं एक-एक करके छोटे और आसान सवाल पूछूँगी। आप बोलकर या नीचे हाँ या नहीं का बटन दबाकर जवाब दे सकती हैं, और कुछ समझ न आए तो पूछ सकती हैं। शुरू करें? |
| `close_anything_else` | `close` | `open_capture` | `()` | क्या इसके अलावा डॉक्टर के लिए कुछ और बताना है जो आपके मन में हो? |

### 4.3 Atomic Clinical Questions (56 Items — All `kind="gate"`, `opens=()`)

| # | ID | Group | NRROID | `skip_if` | `question_hi` (1 Child / General) | `question_multi_hi` (2+ Children) | `followups` |
|---|---|---|---|---|---|---|---|
| 1 | `birth_mode_detail` | `birth` | `Q03` | `no_children` | क्या बच्चे का जन्म ऑपरेशन (सिजेरियन) से हुआ था? | क्या आपके किसी भी बच्चे का जन्म ऑपरेशन (सिजेरियन) से हुआ था? | `operation_reason` |
| 2 | `birth_term` | `birth` | `Q03` | `no_children` | क्या बच्चे का जन्म समय से बहुत पहले (9 महीने से काफी पहले) हुआ था? | क्या आपके किसी भी बच्चे का जन्म समय से बहुत पहले (9 महीने से काफी पहले) हुआ था? | `term_how_early` |
| 3 | `birth_nicu` | `birth` | `Q03` | `no_children` | क्या जन्म के बाद बच्चे को काँच के बक्से या बच्चों के ICU (NICU) में रखना पड़ा था? | क्या जन्म के बाद आपके किसी भी बच्चे को काँच के बक्से या बच्चों के ICU (NICU) में रखना पड़ा था? | `nicu_reason`, `nicu_days` |
| 4 | `birth_jaundice` | `birth` | `Q03` | `no_children` | क्या जन्म के बाद पीलिया के लिए बच्चे को नीली लाइट में रखना पड़ा था? | क्या जन्म के बाद पीलिया के लिए आपके किसी भी बच्चे को नीली लाइट में रखना पड़ा था? | `jaundice_days` |
| 5 | `birth_sudden` | `birth` | `Q03` | `no_children` | क्या जन्म के बाद बच्चा अचानक बहुत बीमार पड़ा और इमरजेंसी में ले जाना पड़ा? | क्या जन्म के बाद आपका कोई भी बच्चा अचानक बहुत बीमार पड़ा और इमरजेंसी में ले जाना पड़ा? | `sudden_detail` |
| 6 | `birth_cord_3weeks` | `birth` | `Q03` | `no_children` | क्या जन्म के बाद बच्चे की नाल गिरने में 3 हफ्ते से ज़्यादा लगे थे? | क्या जन्म के बाद आपके किसी भी बच्चे की नाल गिरने में 3 हफ्ते से ज़्यादा लगे थे? | `()` |
| 7 | `birth_cord_bleed` | `birth` | `Q03` | `no_children` | क्या नाल गिरने पर बहुत ज़्यादा खून आया था? | क्या आपके किसी भी बच्चे की नाल गिरने पर बहुत ज़्यादा खून आया था? | `()` |
| 8 | `birth_bcg` | `birth` | `Q03` | `no_children` | क्या कंधे के BCG टीके की जगह पर बड़ा फोड़ा, मवाद, या बगल में गाँठ हुई थी? | क्या आपके किसी भी बच्चे को कंधे के BCG टीके की जगह पर बड़ा फोड़ा, मवाद, या बगल में गाँठ हुई थी? | `()` |
| 9 | `birth_cleft` | `birth` | `Q02` | `no_children` | क्या जन्म से बच्चे का होंठ या तालू कटा हुआ था? | क्या जन्म से आपके किसी भी बच्चे का होंठ या तालू कटा हुआ था? | `()` |
| 10 | `birth_hearing` | `birth` | `Q02` | `no_children` | क्या बच्चे को सुनने में कोई दिक्कत है? | क्या आपके किसी भी बच्चे को सुनने में कोई दिक्कत है? | `()` |
| 11 | `birth_vision` | `birth` | `Q02` | `no_children` | क्या बच्चे को देखने में कोई दिक्कत है? | क्या आपके किसी भी बच्चे को देखने में कोई दिक्कत है? | `()` |
| 12 | `body_fits` | `body` | `Q02` | `no_children` | क्या बच्चे को कभी झटके या दौरे (फ़िट्स) आए हैं? | क्या आपके किसी भी बच्चे को कभी झटके या दौरे (फ़िट्स) आए हैं? | `fits_onset` |
| 13 | `body_milestones` | `body` | `Q02` | `no_children` | क्या बच्चे ने बैठना, चलना या बोलना बाकी बच्चों से काफी देर से सीखा? | क्या आपके किसी भी बच्चे ने बैठना, चलना या बोलना बाकी बच्चों से काफी देर से सीखा? | `()` |
| 14 | `body_regression` | `body` | `Q02` | `no_children` | क्या बच्चा कोई पहले से सीखी हुई बात (जैसे चलना या बोलना) अब भूल गया है? | क्या आपका कोई भी बच्चा पहले से सीखी हुई कोई बात (जैसे चलना या बोलना) अब भूल गया है? | `lost_skill` |
| 15 | `body_blue` | `body` | `Q02` | `no_children` | क्या बच्चे के होंठ या उंगलियाँ कभी नीली पड़ती हैं? | क्या आपके किसी भी बच्चे के होंठ या उंगलियाँ कभी नीली पड़ती हैं? | `()` |
| 16 | `preg_heart` | `body` | `Q02` | `no_children` | क्या स्कैन में या जन्म पर डॉक्टर ने बच्चे के दिल में कोई खराबी बताई थी? | क्या स्कैन में या जन्म पर डॉक्टर ने आपके किसी भी बच्चे के दिल में कोई खराबी बताई थी? | `()` |
| 17 | `body_bleeding_bruise` | `body` | `Q04` | `no_children` | क्या मामूली चोट पर खून देर तक बहता है, या आसानी से नील पड़ जाते हैं? | क्या आपके किसी भी बच्चे को मामूली चोट पर खून देर तक बहता है, या आसानी से नील पड़ जाते हैं? | `()` |
| 18 | `body_joint_swelling` | `body` | `Q04` | `no_children` | क्या गिरने पर घुटने या जोड़ में सूजन आती है जो कुछ दिनों में अपने आप ठीक हो जाती है? | क्या आपके किसी भी बच्चे के घुटने या जोड़ में गिरने पर सूजन आती है जो कुछ दिनों में ठीक हो जाती है? | `()` |
| 19 | `mod_tooth_bleed` | `body` | `Q04` | `no_children`, `no_child_5_plus` | क्या दूध का दाँत टूटने पर खून बहुत देर तक बहता रहा था? | क्या आपके किसी भी बच्चे का दूध का दाँत टूटने पर खून बहुत देर तक बहता रहा था? | `()` |
| 20 | `mod_heavy_periods` | `body` | `Q04` | `no_children`, `no_daughter_11_plus` | क्या आपकी बेटी को पीरियड्स में बहुत ज़्यादा खून आता है? | क्या आपकी किसी बेटी को पीरियड्स में बहुत ज़्यादा खून आता है? | `()` |
| 21 | `body_transfusion` | `body` | `Q05` | `no_children` | क्या बच्चे को कभी खून चढ़ाना पड़ा है? | क्या आपके किसी भी बच्चे को कभी खून चढ़ाना पड़ा है? | `transfusion_count` |
| 22 | `body_blood_count` | `body` | `Q05` | `no_children` | क्या डॉक्टर ने कभी बच्चे का खून बहुत कम या असामान्य बताया है? | क्या डॉक्टर ने कभी आपके किसी भी बच्चे का खून बहुत कम या असामान्य बताया है? | `()` |
| 23 | `body_limb_pain` | `body` | `Q05` | `no_children` | क्या बच्चे के हाथ-पैर में बार-बार इतना तेज़ दर्द होता है कि अस्पताल जाना पड़े? | क्या आपके किसी भी बच्चे के हाथ-पैर में बार-बार इतना तेज़ दर्द होता है कि अस्पताल जाना पड़े? | `()` |
| 24 | `body_tummy` | `body` | `Q06` | `no_children` | क्या बच्चे का पेट धीरे-धीरे बड़ा हो रहा है (लिवर या तिल्ली बढ़ना)? | क्या आपके किसी भी बच्चे का पेट धीरे-धीरे बड़ा हो रहा है (लिवर या तिल्ली बढ़ना)? | `()` |
| 25 | `body_lumps_birth` | `body` | `Q06` | `no_children` | क्या जन्म से बच्चे के शरीर पर कोई सूजन या गाँठें हैं? | क्या जन्म से आपके किसी भी बच्चे के शरीर पर कोई सूजन या गाँठें हैं? | `()` |
| 26 | `body_growth` | `body` | `Q07` | `no_children` | क्या बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है? | क्या आपके किसी भी बच्चे का कद या वज़न बाकी बच्चों से काफी कम बढ़ रहा है? | `()` |
| 27 | `body_proportion` | `body` | `Q07` | `no_children` | क्या बच्चे के हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं? | क्या आपके किसी भी बच्चे के हाथ-पैर शरीर के मुकाबले बहुत छोटे हैं? | `()` |
| 28 | `body_thumb` | `body` | `Q07` | `no_children` | क्या बच्चे की कोई उंगली या अंगूठा छोटा है, या कम है? | क्या आपके किसी भी बच्चे की कोई उंगली या अंगूठा छोटा है, या कम है? | `()` |
| 29 | `gate_bones_fractures` | `body` | `Q08` | `no_children` | क्या मामूली गिरने पर बच्चे की हड्डी टूटी है, या दवा के बाद भी पैर टेढ़े हैं? | क्या मामूली गिरने पर आपके किसी भी बच्चे की हड्डी टूटी है, या दवा के बाद भी पैर टेढ़े हैं? | `fracture_count_or_bowing` |
| 30 | `gate_urine_kidney` | `body` | `Q09` | `no_children` | क्या बच्चा बार-बार बहुत ज़्यादा पेशाब करता और पानी पीता है, या गुर्दे/पथरी की दिक्कत है? | क्या आपका कोई भी बच्चा बार-बार बहुत ज़्यादा पेशाब करता और पानी पीता है, या गुर्दे/पथरी की दिक्कत है? | `urine_or_stone_detail` |
| 31 | `body_private_parts` | `body` | `Q10` | `no_children` | क्या बच्चे के गुप्त अंग वैसे नहीं दिखते जैसे होने चाहिए? | क्या आपके किसी भी बच्चे के गुप्त अंग वैसे नहीं दिखते जैसे होने चाहिए? | `()` |
| 32 | `body_puberty` | `body` | `Q10` | `no_children`, `all_under_10` | क्या 10 साल से बड़े बच्चे में बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं? | क्या 10 साल से बड़े आपके किसी बच्चे में बढ़ती उम्र के बदलाव बहुत जल्दी या बहुत देर से आए हैं? | `()` |
| 33 | `ill_fasting` | `illness` | `Q01` | `no_children` | क्या खाना छूटने या हल्की बीमारी में बच्चा बहुत सुस्त पड़ जाता है या बार-बार उल्टी करता है? | क्या खाना छूटने या हल्की बीमारी में आपका कोई भी बच्चा बहुत सुस्त पड़ जाता है या बार-बार उल्टी करता है? | `()` |
| 34 | `ill_odor` | `illness` | `Q01` | `no_children` | क्या बच्चे के पेशाब या पॉटी में कभी अजीब बदबू आती है? | क्या आपके किसी भी बच्चे के पेशाब या पॉटी में कभी अजीब बदबू आती है? | `()` |
| 35 | `ill_admissions` | `illness` | `Q03` | `no_children` | क्या बच्चे को बार-बार अस्पताल में भर्ती करना पड़ा है? | क्या आपके किसी भी बच्चे को बार-बार अस्पताल में भर्ती करना पड़ा है? | `admissions_count`, `admissions_reason` |
| 36 | `ill_iv_antibiotics` | `illness` | `Q03` | `no_children` | क्या बार-बार इंफेक्शन के लिए बच्चे को नस में (IV ड्रिप से) दवा लगी है? | क्या बार-बार इंफेक्शन के लिए आपके किसी भी बच्चे को नस में (IV ड्रिप से) दवा लगी है? | `()` |
| 37 | `preg_conditions` | `pregnancy` | `Antenatal` | `unmarried`, `about_to_marry`, `ever_pregnant_no` | क्या किसी प्रेग्नेंसी में आपको हाई बीपी, शुगर या थायरॉइड रहा था? | *(empty)* | `preg_cond_detail` |
| 38 | `preg_scan` | `pregnancy` | `Antenatal` | `unmarried`, `about_to_marry`, `ever_pregnant_no` | क्या किसी प्रेग्नेंसी के सोनोग्राफी स्कैन में कोई खराबी बताई गई थी? | *(empty)* | `scan_detail` |
| 39 | `preg_transfusion` | `pregnancy` | `Antenatal` | `unmarried`, `about_to_marry`, `ever_pregnant_no` | क्या किसी प्रेग्नेंसी के दौरान आपको खून चढ़ाना पड़ा था? | *(empty)* | `()` |
| 40 | `preg_delivery` | `pregnancy` | `Antenatal` | `unmarried`, `about_to_marry`, `ever_pregnant_no` | क्या किसी डिलीवरी के समय कोई बड़ी दिक्कत आई थी? | *(empty)* | `delivery_detail` |
| 41 | `preg_thalassemia` | `pregnancy` | `Antenatal` | `()` | क्या आपकी या आपके पति की थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई? | *(empty — dynamic per Sec 3)* | `()` |
| 42 | `fam_related` | `family` | `Q11` | `unmarried` | क्या आप और आपके पति शादी से पहले आपस में रिश्तेदार थे, जैसे कज़िन या एक ही खानदान से? | *(empty — dynamic per Sec 3)* | `related_how` |
| 43 | `fam_miscarriage` | `family` | `Q12` | `unmarried`, `about_to_marry`, `ever_pregnant_no` | क्या आपका कभी गर्भ गिरा है (मिसकैरेज), या गिरवाना पड़ा था? | *(empty)* | `miscarriage_count`, `miscarriage_month` |
| 44 | `fam_infant_death` | `family` | `Q12` | `()` | क्या परिवार में किसी छोटे बच्चे की 1 साल से कम उम्र में मौत हुई है? | *(empty)* | `infant_death_age`, `infant_death_cause` |
| 45 | `fam_fits_delay` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े को झटके (फ़िट्स), देर से सीखने, या सीखी बात भूलने की समस्या है? | *(empty)* | `fam_who` |
| 46 | `fam_transfusion` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े को बार-बार खून चढ़ाना पड़ता है? | *(empty)* | `fam_who` |
| 47 | `fam_bleeding` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की समस्या है? | *(empty)* | `fam_who` |
| 48 | `fam_blind_deaf` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े को बचपन या कम उम्र से अंधेपन या बहरेपन की समस्या है? | *(empty)* | `fam_who` |
| 49 | `fam_cancer` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े को कैंसर हुआ है? | *(empty)* | `fam_who` |
| 50 | `fam_heart_kidney` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े को दिल या गुर्दे (किडनी) की गंभीर बीमारी है? | *(empty)* | `fam_who` |
| 51 | `fam_bp_stroke` | `family` | `Q11` | `()` | क्या परिवार में 40-50 साल से कम उम्र के किसी व्यक्ति को बीपी, शुगर या लकवा हुआ है? | *(empty)* | `bp_who`, `bp_age` |
| 52 | `fam_maternal_bleeding` | `family` | `Q11` | `()` | क्या आपके मायके की तरफ किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की दिक्कत है? | *(empty)* | `fam_who` |
| 53 | `fam_maternal_infections` | `family` | `Q11` | `()` | क्या आपके मायके की तरफ किसी बच्चे या बड़े को बार-बार गंभीर इंफेक्शन की दिक्कत है? | *(empty)* | `fam_who` |
| 54 | `fam_maternal_eczema` | `family` | `Q11` | `()` | क्या आपके मायके की तरफ किसी बच्चे या बड़े को त्वचा पर सूखे खुजली वाले धब्बों की दिक्कत है? | *(empty)* | `fam_who` |
| 55 | `fam_lumps` | `family` | `Q11` | `()` | क्या परिवार में किसी बच्चे या बड़े के शरीर पर सूजन या गाँठें हैं? | *(empty)* | `fam_who` |
| 56 | `fam_same_problem` | `family` | `Q11` | `no_children`, `no_family_concern` | क्या परिवार में किसी और को भी आपके बच्चे जैसी ही दिक्कत है? | *(empty)* | `fam_who` |

---

## 5. FormBharo State Engine Contract (`code/shishu-v2/call_cursor.py`)

### 5.1 State Shape (Backwards-Compatible with `db.py`, `post_call.py`, `mother_report.py`)
```python
{
  "version": 2,
  "phase": "asking" | "closing",
  "close_reason": None | "complete" | "silence" | "time_limit" | "emergency" | "caller_stop",
  "profile": {
    "caller_name": None | str,
    "marital_status": None | "married" | "about_to_marry" | "unmarried",
    "children": [],
    "ever_pregnant": None | bool,
  },
  "active_qid": "intro_name",
  "pending_followup_idx": None | -1 | int,  # -1 = which_child, 0+ = followup index
  "step_attempts": 0,
  "hold_reason": None | str,
  "last_answered_qid": None | str,
  "answers": {},
}
```

### 5.2 Core Cursor Functions & Transitions
1. **`apply_profile(state, *, caller_name=None, marital_status=None, children=None, ever_pregnant=None, child_count=None) -> tuple[dict, str | None]`**:
   - Works at **any** point in the call (not restricted to `intro_*`).
   - If `marital_status in ("unmarried", "about_to_marry")`: if `children is None` and not explicitly provided, clears `state["profile"]["children"] = []`; if `ever_pregnant is None` and not `True`, sets `state["profile"]["ever_pregnant"] = False`.
   - Re-runs `_recompute(state)`. If `state["active_qid"]` is now skipped (e.g. caller was on `intro_children` and said *"मेरी शादी नहीं हुई है"*), immediately calls `_advance(state)`!
   - When a profile step is incomplete (`hold` in `{"name_fragment", "marital_status", "children_ages", "ever_pregnant"}`), stores `state["hold_reason"] = hold` so `prompt_directive(state)` speaks a **specific, natural Hindi prompt** instead of repeating the full intro question:
     - `"name_fragment"` -> `"जी, आराम से बताइए — आपका पूरा नाम क्या है?"`
     - `"marital_status"` -> `"जी, क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?"`
     - `"children_ages"` -> `"जी, उन सभी बच्चों की उम्र क्या-क्या है?"`
     - `"ever_pregnant"` -> `"क्या आप कभी गर्भवती हुई हैं?"`
2. **`apply_turn(state, outcome, *, child_ref="", note="", followup_answered=False) -> dict`**:
   - Allowed `outcome`: `"clear" | "concern" | "unknown" | "clarify" | "retract" | "emergency" | "caller_stop"`.
   - **`intro_bridge` (`spec.kind == "bridge"`)**:
     - `"concern"` or `"clear"` or `"unknown"` -> writes `"clear"` and advances (never creates a `"concern"`!).
   - **Active `ever_pregnant` step**:
     - Allows `apply_turn` (including UI quick-tap `हाँ`/`नहीं`!) to resolve `ever_pregnant`: `"concern"` -> `ever_pregnant = True`, `"clear"` -> `ever_pregnant = False`, `"unknown"` -> `ever_pregnant = False` + advances cleanly.
   - **Retraction (`outcome == "retract"`, OR `pending_followup_idx == -1` with `outcome == "clear"` and empty `child_ref`)**:
     - Overwrites `state["answers"][active_qid]` to `_blank("clear")`, clears `pending_followup_idx = None` and `step_attempts = 0`, runs `_recompute(state)`, and `_advance(state)`.
   - **Multi-child resolution (`_resolve_child_ids(state, child_ref)`)**:
     - `_resolve_child_ids` uses closed-world exact matching without regular expressions. It normalizes whitespace on `child_ref` and sorts children from oldest to youngest (`ordered = sorted(children, key=lambda c: float(c["age_years"]), reverse=True)`).
       - `"all" / "both" / "दोनों" / "सभी" / "सबको" / "तीनों" / "चारों"` returns all child IDs in `ordered`.
       - Exact `child_id` (`child_1`..`child_n`) or exact casefolded `child["name"]` returns that child ID.
       - Exact ordinal tokens map to `ordered`: `"oldest" / "बड़े" / "बड़ा" / "बड़ी" / "बड़े वाले" / "पहले" / "पहला"` -> `ordered[0]`, `"middle" / "बीच" / "बीच वाले" / "बीच वाला" / "बीच वाली" / "मंझले" / "मंझला" / "मंझली" / "मंझल"` -> `ordered[len(ordered)//2]`, `"दूसरे" / "दूसरा"` -> `ordered[1]` (for 2+ children), `"तीसरे" / "तीसरा"` -> `ordered[2]` (for 3+ children), and `"youngest" / "छोटे" / "छोटा" / "छोटी" / "छोटे वाले" / "आखिरी" / "सबसे छोटा"` -> `ordered[-1]`.
       - Exact sex tokens (`"बेटा" / "लड़का" / "boy"` or `"बेटी" / "लड़की" / "girl"`) match when only one child has that sex.
     - When the profile has two or more children, `children_roster(state)` returns `[{"id", "rank", "age_label_hi", "sex", "name"}]` from oldest to youngest (`rank` is `oldest`, `middle`, or `youngest`). `_tool_body(state)` sends this list as `"children"`, and Gemini passes an exact `id` or `"all"`.
     - `_which_child_hi(children)` lists each child in `ordered` by `age_label_hi` (plus `बेटा`/`बेटी` or `name`) and adds `(बीच वाले)` on the middle child of three, for example `"यह किस बच्चे की बात है — 8 साल वाले, 5 साल वाले (बीच वाले), या 2 साल वाले?"`.
   - **FormBharo Bounded Retry Cap (`MAX_STEP_ATTEMPTS = 2`)**:
     - If `pending_followup_idx == -1` (`which_child`) receives an unresolvable reply (not retracted, not `unknown`, no matched child), increments `step_attempts`. On `step_attempts >= 2`, leaves `child_ids = []`, records `note` if any, and moves to `followups[0]` or `_advance(state)`.
     - If `pending_followup_idx >= 0` receives an empty/clarify non-answer twice (`step_attempts >= 2`), records `{"state": "unknown", "note": ""}` for that follow-up and advances to the next follow-up or question.
   - **Pre-Answered Follow-up Skip**:
     - When a main question has `outcome == "concern"`, `followup_answered` is `True`, `note.strip()` is non-empty, and `spec.followups` is non-empty, `_resolve_main` stores `note` in `followups[spec.followups[0]]`.
     - If `len(spec.followups) == 1` and which-child is not pending, the cursor advances to the next question.
     - If `len(spec.followups) > 1` and which-child is not pending, `_resolve_main` sets `pending_followup_idx = 1` so the cursor asks the second follow-up.
     - If which-child is pending (`pending_followup_idx == -1`), `_after_child_known` skips the filled first follow-up and asks the next unfilled follow-up.
   - **`close_anything_else` (`open_capture`)**:
     - If `outcome == "concern"` and `not note.strip()` (e.g. she tapped the `हाँ` button or said bare `"हाँ"`): sets `pending_followup_idx = 0` using follow-up `"open_concern_detail"` (`"जी बताइए, डॉक्टर के लिए क्या बात नोट करूँ?"`). When she speaks the concern note on that follow-up, stores it on `close_anything_else["notes"]` with `state = "concern"`, resets `pending_followup_idx = None`, increments `close_rounds`, and asks `"मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?"`.
     - If `outcome == "concern"` WITH `note.strip()`: appends `note` to `close_anything_else["notes"]`, increments `close_rounds`, and asks `"मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?"`.
     - If `outcome in ("clear", "unknown")`: preserves any prior `"concern"` note (or writes `"clear"`/`"unknown"` if none) and transitions `phase = "closing"`, `close_reason = "complete"`.
3. **`expect_yes_no(state) -> bool`**:
   - Returns `True` **only** when `state["phase"] == "asking"`, `state["pending_followup_idx"] is None`, and `state["active_qid"]` is NOT in `{"intro_name", "intro_married", "intro_children"}` (i.e. `True` on `intro_bridge`, `ever_pregnant`, all 56 clinical `gate` questions, and `close_anything_else` when `pending_followup_idx is None`; `False` on `intro_name`, `intro_married`, `intro_children`, `which_child`, and all detail `followups`).

---

## 6. Live Voice & Quick-Reply WebSocket Contract (`gemini_live.py` + `static/call.html` + `static/call.js`)

### 6.1 Clean Turn-Taking in `gemini_live.py`
1. **No `"wait": True` and causal speak on `turn_complete`**:
   - `_tool_body(state)` returns `{"result": "ok", "say": _say_hindi(state)}` and never sets `"wait": True`. It adds `"stop": True` on closing, `"clarify": True` on clarify, and `"children": children_roster(state)` when two or more children exist.
   - When `_answer_function_calls` handles a tool call, it stores `pending_speak_step` and `pending_speak_kind` and waits for `server_content.turn_complete`.
   - When `server_content.turn_complete` arrives on that same step before any bot audio, `_handle_server_message` cancels the fallback task, clears `pending_speak_step`, and sends `_speak_cue_text`.
   - `_TOOL_SPEAK_FALLBACK_SEC = 2.5` acts only as a dead-socket safety net when `turn_complete` never arrives. Bot audio and an accepted `quick_reply` both clear `pending_speak_step`.
2. **Pause-Safe VAD and Reliable Silence Check-In**:
   - `db.py` and `gemini_live.py` set `vad_silence_ms = "1000"` and `vad_end_sensitivity = "LOW"` by default.
   - When browser `{"type": "speech_end"}` arrives and the bot is not speaking, `_browser_loop` sets `state.quiet_from = max(now_ts, state.playback_ends_at)` so the 15s check-in still runs.
   - `CallAudioRecorder._remember` records `(elapsed, pcm, rate)` under `self._lock` in O(1) time. `CallAudioRecorder.save()` performs PCM resampling off the event loop inside `asyncio.to_thread`.
3. **Anytime `save_profile`**:
   - `_profile_kwargs(qid, args)` parses **any** non-empty field (`caller_name`, `marital_status`, `child_count`, `children`, `ever_pregnant`) regardless of `qid`, and `_marital()` maps `"no" / "nahi" / "नहीं" / "अभी नहीं" / "अविवाहित" / "कुंवारी"` -> `"unmarried"`.

### 6.2 WebSocket Quick-Reply Protocol
- **Server -> Browser (`question_state` + `bot_turn_complete`)**:
  - `gemini_live.py` sends `question_state` after the greeting cue and after every handled tool call or `quick_reply`:
    `{"type": "question_state", "step": _current_step(cursor.state), "qid": active_qid, "group": group, "expect_yes_no": expect_yes_no(cursor.state), "prompt_chars": len(_say_hindi(cursor.state))}`
  - On `server_content.turn_complete` when the bot is done streaming that question, `gemini_live.py` sends `{"type": "bot_turn_complete", "step": _current_step(cursor.state)}`.
- **Browser -> Server (`quick_reply`)**:
  - Payload: `{"type": "quick_reply", "outcome": "clear" | "concern", "step": currentStep}`
  - Server handling:
    1. If `telemetry.get("end_requested")` is true or `cursor.state.get("phase") != "asking"`, ignore the message.
    2. Make sure that `payload.get("step") == _current_step(cursor.state)` and `expect_yes_no(cursor.state)` is `True`, or ignore the duplicate tap.
    3. Record the transcript (`"नहीं।"` or `"हाँ।"`), and emit `user_transcript` and `tool_call` (`quick_reply_clear` or `quick_reply_concern`).
    4. If `state.bot_speaking` is `True`, set `state.ignore_bot_audio = True` so `_on_bot_audio` and `output_transcription` drop leftover bot output until `interrupted` or `turn_complete`.
    5. Clear `user_turn_step` and `pending_speak_step`, record `last_quick_reply_step` and `last_quick_reply_ts`, apply the turn, and send `{"type": "clear"}` plus `question_state`.
    6. If `cursor.state["phase"] == "closing"`, call `_mark_closing(...)`. Then send `_speak_cue_text` (`"close"` or `"speak"`) to Gemini Live.
    7. If `user_text` arrives before the new step is spoken, during `ignore_bot_audio`, or within `1.2s` of `last_quick_reply_ts`, keep `user_turn_step = last_quick_reply_step`. When `user_turn_step` does not match `_current_step(cursor.state)` (or is empty after a tap), hold `record_turn` with `reason = "stale_turn"`.

### 6.3 Call Screen Flanking Buttons (`static/call.html` + `static/call.js`)
- Inside `#stage`, wrap the main call button in `<div id="call-dock" class="call-dock">`:
  - **Left**: `<button id="no-btn" class="call-btn quick-btn no-btn" type="button" hidden><span class="hi">नहीं</span><span class="en">No</span></button>`
  - **Center**: `<button id="call-btn" class="call-btn" type="button" disabled><span class="hi">शुरू</span><span class="en">Start</span></button>`
  - **Right**: `<button id="yes-btn" class="call-btn quick-btn yes-btn" type="button" hidden><span class="hi">हाँ</span><span class="en">Yes</span></button>`
- **Sizing & Styling**:
  - All three buttons share `.call-btn` dimensions (`clamp(68px, 21vw, 84px)` circle so all 3 stay on one horizontal row down to `320px` width with `clamp(12px, 4vw, 20px)` gap).
  - `#call-btn` (`शुरू / Start`): warm amber (`var(--yellow)` `#e89d2d`) with high-contrast `var(--navy)` text (`9.68:1`).
  - `#no-btn`: white background (`#ffffff`), `2px solid var(--navy)` border, `var(--navy)` text (`--en` muted).
  - `#call-btn.end`: red (`var(--red)` `#dc2626`), white text (`बंद / End`).
  - `#yes-btn`: navy (`var(--navy)` `#1b2a4a`), white text (`हाँ / Yes`).
  - `#no-btn` and `#yes-btn` stay mounted during `phase === "live"` (`hidden = !live`) so `#call-dock` never jumps, and fade in via `.quick-btn.is-visible` **only** when `question_state.expect_yes_no === true` AND speaker-clock playback (`audioCtx.currentTime - stepAudioStart`) reaches halfway through the spoken question (or `bot_turn_complete` finishes). Hidden immediately during `idle`, `connecting`, `preparing`, `done`, `error`, and non-yes/no steps.
  - On tap: immediately disable `#no-btn` & `#yes-btn` until the next `question_state` arrives, call `flushPlayback()`, display `"You: नहीं।"` or `"You: हाँ।"` in `#caption`, call `startThinkingWatch()`, and send `quick_reply` over WebSocket.

---

## 7. Staged Implementation & Verification Rules

1. **Environment**:
   - Secrets stay in root `.env`.
   - Python env: `~/pyenv/shishu/bin/python` (packages via `uv`).
   - Long-running servers and live persona suites run inside `tmux`.
2. **Stages on `next_v2`**:
   - **Stage 1**: Pure offline `triage_catalog.py` + `call_cursor.py` + `test_v4_cursor.py` (including replays of Staging Calls #3, #6, #8).
   - **Stage 2**: `gemini_live.py` + `prompts.py` + `db.py` (VAD defaults) + server `question_state`/`quick_reply` + `test_v4_live_server.py`.
   - **Stage 3**: `static/call.html` + `static/call.js` flanking `नहीं` / `हाँ` buttons + end-to-end WebSocket `TestClient` tests (`test_v4_quick_reply_ui.py`).
   - **Stages 4–6**: Speaker-clock halfway button reveal, atomic `fam_maternal_*` split, closed-world `target_qid` revisions/clarifications, emergency & report synthesis hardening, mobile clinic WCAG contrast/accessibility polish, **102 offline unit/socket tests + 6-Persona live Gemini verification in `tmux`** (`run_eight_personas.py`: red-flags, all-clear, midway emergency, 3-kids middle-child + retraction, `about_to_marry` mid-call correction + clarify, and `target_qid` late addition + `close_anything_else` disclosure) + deploy `next_v2` to `shishu-staging`.
