# SHISHU V3 — The Moldability Boundary

**Status:** design note for branch `v3`. **Answers one question:** *what can a pack author
change without touching Python or JavaScript, and where does the clay stop being clay?*

Companion documents: [`v3-design-and-thinking-vad.md`](v3-design-and-thinking-vad.md) (the
design of record), [`../README.md`](../README.md) (how to run it).

---

## 0. The one-sentence boundary

> A pack can rebuild **any ordered medical-history conversation** — its wording, its
> questions, its branching, its answer vocabulary, its reports — on this same engine.
> A pack cannot build a **different kind of application**.

Everything below is measured against the code as it stands on `v3`, not against intent.

---

## 1. What a pack *is*

A pack is `packs/<pack_id>/`, loaded by `intake/pack.py::load_pack()` and validated at
startup by `_validate_pack()`. Nothing in `intake/*.py` names `shishu-hi` except the
`DEFAULT_PACK_ID` constant.

```
packs/<pack_id>/
  pack.yaml       branding, model, voice, VAD numbers, timers, assets, every UI string
  prompts.yaml    system prompt, speak cues, closings, vocabulary, roster phrasing
  form.yaml       profile slots, roster, rules, implications, flags, questions, report config
  decisions.yaml  Decision API head instructions + the 17 required thresholds
  seed.yaml       offline replay scenarios (zero network)
  templates/      patient.html.j2, clinician.md.j2, pdf.css
  assets/         logo, font, consent PDF
```

Two complete packs exist and both pass startup validation: `shishu-hi` (62 questions, a
children roster, rare-disease screening) and `fever-opd-hi` (14 questions, single patient,
acute fever triage). They share zero Python.

A pack is selected per request by `?pack=<id>` on the URL, or globally by the `INTAKE_PACK`
environment variable, or by `DEFAULT_PACK_ID`.

---

## 2. Inside the clay — changeable with config only

### 2.1 Conversation content

| Thing | Where |
|---|---|
| Every spoken question, its `stem` (the short form used in reports), its `need` (the clinical intent the Decision API reads) | `form.yaml → questions[].say / stem / need` |
| Re-ask wording when the answer did not land | `form.yaml → questions[].reask` |
| Alternative phrasings chosen by a rule | `form.yaml → questions[].variants[].when / say` |
| Follow-up questions asked after a concern | `form.yaml → followups{}` + `questions[].followups[]` |
| The order of the entire call | the order of `form.yaml → questions[]` |
| The intro/bridge priming line | a `bridge` question, or `pack.yaml → ui.yes_no_prime` |
| The "anything else?" second round | `prompts.yaml → open_capture_more` |
| Every closing line by reason (complete-clear, complete-concern, caller stop, silence, time limit, emergency, technical failure) | `prompts.yaml → closings{reason}` |
| The honorific appended to a name | `prompts.yaml → honorific` |
| The live model's whole system prompt | `prompts.yaml → system_prompt` |
| Cue wrappers around spoken lines | `prompts.yaml → speak_cues.{speak,clarify,retry,closing}`, `greeting_cue`, `silence_check_in_cue` |
| STT vocabulary hints | `prompts.yaml → vocab[]` |
| Report group headings, patient-side and clinician-side | `form.yaml → groups{}` |

### 2.2 Conversation shape

| Capability | Where |
|---|---|
| Eight question kinds: `profile`, `bridge`, `routing_gate`, `gate`, `single_choice`, `multi_choice`, `score`, `open_capture` | `form.yaml → questions[].kind` |
| Choice options, their display labels, English labels and Decision-API criteria | `form.yaml → questions[].options{}` |
| An ordered score rubric (`0..3`, or any length ≥ 2) | `form.yaml → questions[].score_rubric[]` |
| Which answers count as a *concern* rather than an *answer* | `form.yaml → questions[].concern_if.in` / `min_score` |
| Quick-tap buttons on a yes/no step, or choice pills on a `single_choice` | implicit from `kind`, plus `form.yaml → questions[].quick_options: true` |
| Which questions are skipped, and why | `form.yaml → questions[].skip_if[]` → `rules{}` |
| Conditional rewordings | `form.yaml → questions[].variants[]` → `rules{}` |
| Automatic profile implications | `form.yaml → implications[].when / set_profile / default_profile / clear_roster_if_unprovided` |
| A semantic shortcut (confirmed diagnosis prunes redundant probes) | `form.yaml → semantic_flags{}` + `rules → semantic_flag` |
| Redaction criteria for the patient-facing copy | `decisions.yaml → heads.redact_patient_note.instructions` |

### 2.3 The rule language

`form.yaml → rules{}` is a small declarative DSL evaluated by
`intake/engine.py::eval_rule`. Available today:

| Rule form | Meaning |
|---|---|
| `{field, op, value}` | `eq`, `ne`, `in`, `not_in`, `gte`, `lte`, `missing`, `contains`, `not_contains` over a profile value or a recorded answer |
| `{any_of: [...]}` / `{all_of: [...]}` | boolean composition, nestable |
| `{semantic_flag: <id>}` | a Decision API flag fired earlier in this call |
| `{group_no_concern: <group>}` | true when no recorded concern in that group would be contradicted |
| `{roster_op: <op>, tag, min_age, max_age, value}` | `non_empty`, `is_empty`, `count_gt`, `any_match`, `none_match` over the roster |

`field:` targets accept either a profile slot name or a question id.

### 2.4 Identity, voice, branding

| Thing | Where |
|---|---|
| Live model and voice | `pack.yaml → gemini_model`, `gemini_voice` |
| Language code (drives both the voice and STT) | `pack.yaml → language_code` |
| Title, native subtitle, PDF filename prefix | `pack.yaml → title`, `subtitle_native`, `pdf_prefix` |
| Logo, Devanagari font, consent PDF | `pack.yaml → assets{}` + `packs/<id>/assets/` |
| Every string on the login page and the call page (headings, status copy, buttons, thanks screen, consent labels, bug-report dialog and its category chips) | `pack.yaml → ui.{login,call,yes_no_prime,yes_no_buttons,bug_categories}` |
| How the patient is named in the PDF header | `prompts.yaml → roster_format` |
| How each child is described on screen, and how the "which child?" prompt reads | `prompts.yaml → roster_format.{tag_labels,join,which_item}` |
| Month/year words used for ages | `prompts.yaml → roster_format.{month_unit,year_unit}` |

### 2.5 Voice-activity tuning

`pack.yaml → vad{}` sets the Gemini Live `AutomaticActivityDetection` configuration consumed
by `intake/live.py::build_live_config` (`start_sensitivity`, `end_sensitivity`,
`prefix_padding_ms`, `silence_duration_ms`). `pack.yaml → timers{}` sets the check-in,
silence-close, soft limit, and hard limit timers.

### 2.6 Decision layer

| Thing | Where |
|---|---|
| Which model scores the turn | `decisions.yaml → model` |
| The wording of every `noul` head's question (`danger_now`, `caller_stop`, `mentions_profile`, `redact_patient_note`, `detail_present`, `jump_topic`) | `decisions.yaml → heads.<name>.instructions / criteria` |
| The `active_outcome` class definitions (what counts as clear / concern / unknown / clarify / retract) | `decisions.yaml → heads.active_outcome.criteria` |
| How out-of-turn answers are phrased to the model | `decisions.yaml → heads.stated_template` |
| How the topic router is phrased | `decisions.yaml → heads.next_qid_template` |
| How per-child attribution is phrased | `decisions.yaml → heads.item_ref_template` |
| Every confidence gate | `decisions.yaml → thresholds{}` (all 16 keys are mandatory; a pack that omits one fails to load) |

### 2.7 Reports

| Thing | Where |
|---|---|
| The full patient PDF layout | `templates/patient.html.j2` |
| The full clinician note layout | `templates/clinician.md.j2` |
| The patient PDF styling | `templates/pdf.css` |
| Patient-PDF copy: yes/no headers, the "questions to ask the doctor" boilerplate, per-concern sentence templates, emergency variants, the universal advice line, month names, minute/second units, AM/PM labels, complete/incomplete badges | `form.yaml → report{}` |
| Clinician section titles | `form.yaml → groups{}` + `form.yaml → report.clinician_headings{}` |
| Which section a question lands in | `form.yaml → questions[].clinician_group` |
| The default chief complaint when nothing was flagged | `pack.yaml → chief_complaint` |
| Which questions are reportable at all | question `kind` (`routing_gate` is excluded) |
| Routine and emergency question lists | `form.yaml → report.routine_questions`, `emergency_questions`, `routine_empty_roster_questions` |

### 2.8 Test and replay fixtures

`seed.yaml → scenarios[]` replays scripted answers through the real reducer with zero
network, and is what produces the sample PDFs. A new pack with scenarios seeds its own
non-empty dashboard on first boot.

---

## 3. Outside the clay — needs code

These are not gaps in the configs; they are the engine's actual shape. Asking for them is
asking for a different product.

### 3.1 The conversation is one ordered pass, not a program

- There is exactly **one** question list, walked once, forward. No sub-forms, no nested
  sections, no reusable blocks, no loops, no returning to a completed section.
- `jump_topic` can only *reorder* within the current eligible set. It can never reach a
  question that is already answered, and never one outside the pack's own list.
- A "repeat this block for every child" form (one set of questions per roster member) is
  **not** expressible. `roster_scoped` attributes one concern to one child; it does not
  multiply the form.
- Re-ask depth is fixed at `MAX_STEP_ATTEMPTS = 2` in `intake/engine.py`. It is not a
  config value.
- There is no "on answer X, stop and ask an entirely different questionnaire."

### 3.2 The answer vocabulary is fixed

A pack cannot introduce an answer type the engine does not already understand:

- no date or date-range input
- no numeric measurement with units and range validation (`temperature 97 F`, `weight 12 kg`,
  `height 84 cm`)
- no computed values (BMI, gestational age, dose, risk score)
- no structured objects (address, medication list with dose/frequency/duration)
- no ranked or matrix answers
- no file or photo upload inside the flow
- no signature

The `profile_slots[].type` field exists but is **not honoured**: `intake/extractor.py`
builds every slot as a nullable string in the extraction schema, so `type: number` yields a
string. A numeric slot works only because the string is parsed nowhere it matters.

### 3.3 The roster is one flat list

One optional roster per pack, keyed by `roster.key`, with items auto-identified as
`item_1..n`, each holding `age_years`, `tag`, `name`. There is no person-level sub-record,
no second roster (e.g. "mother" *and* "children" as separate lists), and no relationship
graph.

### 3.4 The rules compute nothing

`eval_rule` compares stored values. It cannot do arithmetic, date math, or external lookup.
Anything that needs to *derive* a number from several answers has to become an engine
function.

### 3.5 The Decision layer's structure is code

You can rewrite the instructions of the heads that exist. You cannot add a head, remove a
head, change a head's *type*, or change what its result does:

- The head inventory is fixed in `intake/decisions.py` (`_CORE`, `_NOUL`, plus
  `active_outcome`, `choice_value` / `opt_<k>` / `score_value`, `item_ref`, `stated_<qid>`,
  `flag_<id>`, `next_qid`).
- The instructions for `choice_value`, `opt_<k>` and `score_value` are **generated in
  Python**; only the per-option `criteria` text is config.
- Out-of-turn auto-fill (`stated_<qid>` heads) exists **only** for `gate` and `routing_gate`
  questions. A `single_choice`, `multi_choice` or `score` question cannot be answered out of
  turn.
- `danger_now` above threshold always means `CloseReason.EMERGENCY` and an immediate
  hangup with the `emergency` closing line. A pack can redefine what the model *looks for*
  but not what happens next.
- The two-tier audio-plus-text request shape (full 4-group primary pass + slim Group-A
  fallback on timeout), the retry budget, the 4.5 s primary timeout and the 8 s total budget
  are constants.

### 3.6 The UI is skinned, not composed

- `call.html`, `call.js`, `login.html`, `admin.html`, `admin.js` and `styles.css` are engine
  files, not pack files. A pack changes their **words**, not their **structure**.
- Layout changes (a different screen flow, a progress list, a summary sidebar, a tablet
  kiosk mode) require editing those files, which then apply to every pack.
- Quick taps exist only for yes/no steps and `single_choice` with `quick_options: true`.
  `multi_choice` and `score` are voice-only; there is no checkbox grid or slider.
- The admin dashboard's structure, filters and fields are fixed.
- `/api/pack`, `/api/pack-asset/{logo,consent_pdf}` and the rest of the HTTP surface are
  fixed. `ASSETS` in `intake/app.py` only serves a **logo** and a **consent PDF**; a new
  asset kind is a code change.

### 3.7 The report projection is fixed, even though the templates are not

The templates can be rewritten completely, but they receive a fixed data model:
`profile_lines`, `alerts`, `sections`, `negatives`, `incomplete`, `red_flags` for the
clinician note; `who`, `meta`, `emergency`, `incomplete`, `questions`, `yes_items`,
`page2_groups`, `page2_unknown`, `page2_unasked`, `universal_line`, `logo_uri`, `css` for
the patient page.

So a *prescription*, a *referral letter*, a *longitudinal chart*, a *FHIR bundle* or a
*scored risk card* is out of scope: the projection in `intake/reports.py::project_summary`
only knows how to describe this call's answers.

### 3.8 Environment and operations

Not pack-defined at all: passcodes and roles (`APP_PASSCODE`, `ADMIN_PASSCODE`), the SQLite
schema, `DATA_DIR` and the file lifecycle, audio retention, share tokens, the bug-report
table, WhatsApp/email delivery, EHR integration, appointment booking, analytics, the
STT/TTS provider (Gemini Live only), and the transport (browser WebSocket only — no
telephony/IVR, no messaging channel).

---

## 4. Declared but not wired

Config keys that exist in a pack and are read by **nothing**. They will silently do nothing
if changed. Listed so a future pack author does not lose an afternoon.

| Key | Reality |
|---|---|
| `pack.yaml → version` | never read |
| `pack.yaml → domain`, `setting`, `patient_scope` | human-readable comments only; no engine module reads them |
| `pack.yaml → ui.choice_buttons` (present in `fever-opd-hi`) | never read — single-choice pills come from `questions[].options` |
| `pack.yaml → timers.watchdog_timeout_sec` | never read — `live.py` drives a fixed `0.5 s` tick (`0.1 s` silence keepalive while speaking) |

Anything else in the packs is live. If you add a key and want it honoured, add it to a
loader *and* to a test.

---

## 5. Adding a third pack — the honest checklist

Feasible with config only, if the new intake is:

1. a single ordered pass of questions,
2. over one patient or one flat list of people,
3. answered by voice (with optional yes/no and single-choice tapping),
4. judged by yes/no, one-of-N, several-of-N, an ordered severity scale, or free speech,
5. reported as "what was positive, what was negative, what is unknown, what was not asked".

If all five hold, a new pack is `packs/<new-id>/` and nothing else. Then:

```
PYTHONPATH=. ~/pyenv/shishu/bin/python -c "from intake.pack import load_pack; load_pack('new-id')"
PYTHONPATH=. ~/pyenv/shishu/bin/python scripts/calibrate_decisions.py --pack new-id --dry-check
PYTHONPATH=. ~/pyenv/shishu/bin/python -m unittest discover -s tests
```

The startup validator rejects duplicate ids, unknown groups/rules/variants/follow-ups, choice
kinds with fewer than two options, `score` with fewer than two rubric levels, a `profile`
step without `requires_slot`, any of the 17 missing thresholds, and any missing template — so
a broken pack fails at load with the whole error list rather than at the first bad question.

If any of the five does not hold, the work is in `intake/`, and it will affect every pack.
That is the line. Configs make many versions of *this* intake; code makes a different intake.
