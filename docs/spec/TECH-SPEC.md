# SHISHU — Technical Specification

Normative contract; code deviating from this is a defect. IDs (`R-`, `S-`, `N-`) map to tests.
Companion: [`PRD-user-flows.md`](../product/PRD-user-flows.md).
**Measured:** `intake/*.py` 3,193 LOC · browser JS+HTML 1,377 LOC (`styles.css` +1,221 LOC) · pack YAML 2,586 LOC · 286 tests (~18 s, offline).

---

## 1. Architecture & Module Boundaries

`browser ──WS──> live.py ──> decisions.py (+extractor.py) ──> TurnInterpretation ──> policy.py (pure) ──> engine.reduce_turn (pure) ──> CallState ──> reports.py (pure, offline) ──> db.py (atomic)`

- **R-1.1** Only `decisions.py` and `extractor.py` call an LLM.
- **R-1.2** `engine.py`, `policy.py`, `reports.py` import no HTTP client and perform no network I/O (post-call purity is verifiable via import inspection).
- **R-1.3** Zero clinical strings, questions, thresholds, or disease names in `intake/*.py`.

- **Packs shipped:** `shishu-hi` (62 Qs, 25 followups, 12 rules, semantic flag `confirmed_hemoglobinopathy`, roster `True`) and `fever-opd-hi` (14 Qs, 3 followups, 2 rules, `confirmed_malaria_or_dengue`, roster `False`).

| Module | Owns | Forbidden |
|---|---|---|
| `models.py` | Frozen/strict Pydantic models, enums | Behaviour |
| `pack.py` | `load_pack`, `list_pack_ids`, fail-fast validator | Pack-specific logic |
| `engine.py` | Pure reducer, eligibility, skip matrix, roster, cursor | Network / I/O |
| `policy.py` | `evaluate_policy`, `apply_policy`, `arbitrate_tap_turn` | Moving cursor |
| `turns.py` | Immutable `TurnEvidence`, `TurnInterpretation`, `PolicyDecision` | Policy |
| `decisions.py` | Single-pass multimodal request + reply parser | Moving cursor |
| `extractor.py` | Schema-validated demographic/roster patch | Mutating `CallState` |
| `live.py` | Turn-epoch FSM, mic gate, cue ownership, resume, telemetry | Clinical content |
| `audio.py` | 16 kHz resampling, RMS, wall-clock resumable stereo WAV | Speech parsing |
| `db.py` | Atomic SQLite + append-only semantic ledger | Policy |
| `reports.py` | Pure projection → Jinja2/WeasyPrint PDF & clinician MD | LLM / network calls |
| `app.py` | FastAPI gate, HMAC auth, rate limits, `/ws/live` | Business logic |

---

## 2. Data Model

```28:29:intake/models.py
AnswerStatus = _enum("AnswerStatus", "CONCERN CLEAR ANSWERED UNKNOWN UNANSWERED NOT_APPLICABLE")
AnswerStatus.is_resolved = property(lambda s: s in (s.CONCERN, s.CLEAR, s.ANSWERED, s.UNKNOWN, s.UNANSWERED))
```

- **S-2.1** `UNKNOWN` only when caller stated don't know / untested; never to break loops.
- **S-2.2** `UNANSWERED` when step retry budget is exhausted without a usable answer.
- **S-2.3** `NOT_APPLICABLE` (from `skip_if`) is unresolved, never printed, never negative.
- **S-2.4** `UNKNOWN` may be revised later to `CONCERN` or `CLEAR`.
- **S-2.5** `CallState` (`phase`, `close_reason`, `profile`, `roster[{id, age_years, age_label, tag, name}]`, `roster_stated`, `roster_incomplete`, `semantic_flags`, `cursor{qid, substep, followup_idx, hold_slot, attempts, open_rounds}`, `last_answered_qid`, `emergency_note`, `answers{qid→AnswerRecord}`) is deep-copied on every reduction; `AnswerRecord` holds `state`, `value`, `choice_label`, `item_ids[]`, `note{verbatim, patient_safe}`, `followups{fid→{state, note}}`.
- **S-2.6** Entire `CallState` serializes to `calls.cursor_json` every turn; resume restores solely from it.
- **S-2.7** `note.verbatim` keeps all caller words for that step, including unslotted extras.
- **S-2.8** `note.patient_safe` drops text whenever the turn is multi-topic, out-of-turn (`stated`), `jump_topic`, or `redacted`; never deletes from `verbatim` (**S-2.9**).
- **S-2.10** `step_token` encodes cursor `(qid, substep, followup_idx, open_rounds, hold_slot, attempts)`; turns admit only if current, resume re-speaks the active token.

---

## 3. Pure Reducer (`engine.reduce_turn`)

Five sequential phases, **at most one cursor advance**:
`1 semantic_flags + _resync` → `2 profile_patch (no cursor move)` → `3 _reduce_active → CursorStep|None` → `4 _apply_stated` → `5 single cursor resolution / _advance`.

- **S-3.1** Only Phase 5 advances; Phase 2 never moves cursor; Phase 3 holds via returned `CursorStep`.
- **S-3.2** Post-turn eligible set recomputes after Phases 1–4; answered/skipped items cannot become next.
- **S-3.3** `MAX_STEP_ATTEMPTS = 2`; 2nd miss writes `UNANSWERED` and releases (including profile steps: `intro_name` miss 1 → `attempts=1`, miss 2 → `intro_name=unanswered`, advance to `intro_married`).
- **S-3.4** `PROFILE` step satisfaction depends strictly on `requires_slot` / `roster_resolved`, never stale `answers` entries.
- **S-3.5** A recorded `CONCERN` with `verbatim_text` is never overwritten to `UNANSWERED`/`CLEAR`/`UNKNOWN` by off-topic turns.
- **S-3.6** Profile/roster changes rewind to earlier unsatisfied steps (profile-only while active step stays live).

**Rules & Roster:**
- Rule DSL: `{field, op, value}` (`eq ne in not_in gte lte missing contains not_contains`), `{any_of|all_of}`, `{semantic_flag}`, `{group_no_concern}`, `{roster_op, tag, min_age, max_age, value}` (`non_empty is_empty count_gt any_match none_match`). Value comparison only.
- **S-3.7** *(Requirement; body validator open)* Unknown rule operators must fail pack load rather than silently returning `False` in `implications`/`variants`.
- **S-3.8–S-3.9** While `roster_incomplete`, `roster_op` `skip_if` rules stay frozen so age-gated questions don't prematurely mark `NOT_APPLICABLE`. Re-sync strips `NOT_APPLICABLE` when rule shape (`field`, `roster_op`, `any_of`, `all_of`, `semantic_flag`) changes.
- **S-3.10** Roster IDs `item_1..n` never renumber; echoed IDs persist, new children take next free ID.
- **S-3.11–S-3.14** Count without ages sets `roster_incomplete` (holds on ages re-ask); explicit zero clears roster (count-only preserves it); display orders oldest-first with name/ordinal disambiguation on equal-age ties (no false `oldest`/`youngest` rank); volunteering a previously `UNANSWERED` profile slot revives it.

---

## 4. Pack Contract (`packs/<pack_id>/`)

Required files: `pack.yaml` (branding, model, voice, VAD, timers, assets, UI strings), `prompts.yaml` (system prompt, cues, 7 closings, vocab, roster format), `form.yaml` (slots, roster, rules, implications, `semantic_flags`, groups, followups, questions, report config), `decisions.yaml` (head instructions + 21 thresholds), `seed.yaml`, `templates/{patient.html.j2, clinician.md.j2, pdf.css}`, `assets/{logo, font, consent_pdf}`.

- **S-4.1** `_validate_pack` fails fast with **all** errors: empty/duplicate IDs, unknown groups/rules/variants/followups, choice `<2` options, score `<2` rubric levels, `profile` without `requires_slot`, missing thresholds or templates.
- **S-4.2–S-4.3** Fixed 8 `QuestionKind`s (`profile`, `bridge`, `routing_gate`, `gate`, `single_choice`, `multi_choice`, `score`, `open_capture`). Out-of-turn auto-fill (`stated_*`) exists only for `gate` and `routing_gate`.
- Question keys: `say`, `stem`, `need`, `reask`, `variants`, `followups`, `skip_if`, `options{label,en,criteria,exclusive}`, `score_rubric`, `concern_if{in|min_score}`, `roster_scoped`, `requires_slot`, `saves_to_profile`, `quick_options`, `clinician_group`, `shares_recorded_concerns`.
- **S-4.4** `semantic_flags` prune only questions explicitly referencing them in `skip_if`.
- **S-4.5–S-4.6** All 21 thresholds mandatory (no defaults, per-pack calibrated, never copied blindly): `*_min_yes` (`danger_now`, `caller_stop`, `mentions_profile`, `redact_patient_note`, `detail_present`, `jump_topic`, `active_topic`, `residual_clinical`, `asks_clarify`, `multi_choice`), `*_min_confidence` + `*_min_margin` (`outcome`, `choice`, `stated`, `item_ref`, `clarify_target`), plus `score_min_confidence`.

---

## 5. Decision Layer (`decisions.py` & `extractor.py`)

- **S-5.1–S-5.3** One multimodal forward pass per spoken turn via `gemini-decision-1` (`/v1beta/decisions`; no speculative pre-passes, no parallel hedging). Sends raw 16 kHz turn WAV + STT hint + current state slice (`active_question`, `caller_utterance`, `followup_question`, `last_answered_question`, optional `parent_question`, `profile`, `roster`).
- **S-5.4–S-5.6** Fixed 4 head groups (pack rewrites prompts/thresholds only):
  - **A (Core + typed value):** `danger_now`, `caller_stop`, `mentions_profile`, `redact_patient_note`, `active_topic`, `residual_clinical`, `asks_clarify`, `active_outcome`, (`choice_value`|`opt_<k>`|`score_value`), `item_ref` (`≥2` roster items), `detail_present` (if followup pending), `clarify_target` (if eligible `UNKNOWN` priors).
  - **B (Independent out-of-turn):** `stated_<qid>` per routable gate (`≠ active`), avoiding winner-take-all mass splitting across multi-fact answers.
  - **C (Shortcut):** `flag_<id>` for unfired pack flags.
  - **D (Advisory router):** `jump_topic`, `next_qid` over open non-`open_capture` questions (only honoured above `choice_min_confidence` over post-turn `E`).
- **S-5.7–S-5.10** Strict JSON output; empty `answers` terminal. **Safety heads (`danger_now`) fail closed**: omitted / non-finite / `null` → retryable failure (not "safe"); after Attempt 2 still bad → failed turn (`0.97` → `True`). Pair-split recovery (`_SPLIT_READS`) combines split halves (`clear`+`retract` on substeps → `retract`; `concern`/`clear`+`clarify` → `clarify`; `clear`+`unknown` → top) when sum ≥ `outcome_min_confidence` and sum − 3rd ≥ `outcome_min_margin`.
- **S-5.11–S-5.15** Per-request `httpx` timeout (`attempt_timeout_sec=12.0`, `total_budget_sec=18.0`, `max_attempts=2`). Retry sequentially **only** on timeout, transport error, `408`, `429`, `5xx`, switching to `slim=True` (drops Groups B & D + `clarify_target`); safety blocks / empty / invalid JSON / other `4xx` never retry. On dead Decision pass, a valid parallel `extractor` patch still commits and advances; otherwise re-ask up to `_MAX_CONSECUTIVE_TOOL_ERRORS=3`, then `TECHNICAL_FAILURE`.
- **S-5.16–S-5.19** `extractor.py` (`gemini-3.5-flash-lite` default, `8.0s` timeout) runs **in parallel** with Decisions whenever `profile`, `saves_to_profile`, or unresolved roster. Returns typed `ProfilePatch` (`_SLOT_TYPES`: `string`, `number`, `integer`, `boolean`, `marital_status` enum, `roster_complete`, `roster`); never invents `age_years: 0`, counts spoken twin/triplet multiplicities.

---

## 6. Policy (`policy.py`)

`evaluate_policy` maps `TurnInterpretation` claims → `PolicyDecision` (`ACCEPT_FORM`, `ACCEPT_UNLISTED`, `RETAIN_OBSERVATION`, `CLARIFY`, `CONFLICT`, `IGNORE`) → `TurnCommand`:
- **S-6.1–S-6.5** Orthogonal `relation` (`ACTIVE`/`OTHER_FORM`/`CLINICAL_ASIDE`/`UNRELATED`) and `domain_fit` (`IN_DOMAIN`/`UNLISTED`/`NONE`). Skipped questions (`skip_if` excluding `group_no_concern`) rejected. Named clarify valid only on `UNKNOWN` priors; stated clarify ignored if >1 stated clarify head fires (scatter protection). `emergency` / `caller_stop` short-circuit all else.
- **S-6.6–S-6.8** `arbitrate_tap_turn`: tap vs contradicting/unlisted speech rolls back to `pre_tap_state`, holds step, stores `SOURCE_CONFLICT`, prompts confirmation; agreement enriches tap with child/note/followup in 1 advance; speech ignoring active step applies side writes cleanly with no double move.

---

## 7. Live Session (`live.py` & `audio.py`)

- **TurnPhase FSM & Epoch:** `PROMPTING` (mic shut; `5.0s` `_PROMPT_MUTE_MAX_SEC` valve) → `LISTENING` (mic open) → `DECIDING` (mic shut) → `CLOSING` (mic shut). Every step publish increments `epoch`; stale wire messages drop by integer compare (**S-7.1**). `_mic_muted()` is the single gate (**S-7.2**); server streams 100 ms zero-PCM while muted so Gemini AAD stays warm (**S-7.3**).
- **Half-duplex + Tail:** Caller taps over SHISHU anytime; voice during SHISHU speech is ignored to prevent echo-interrupt storms (**S-7.4**), except the `350 ms` post-`bot_turn_complete` playback tail (**S-7.5**).
  - **S-7.6** Muted-window audio frames still reach the stereo audit recording, so speech made over SHISHU is never lost from the WAV: `on_audio` writes the frame to `recorder.add_user_pcm` before the mute gate returns.
- **Endpointing & Audio Ring:** Gemini Live (`gemini-3.1-flash-live-preview`, voice `Kore`) AAD (`prefix_padding_ms=200`, `silence_duration_ms=1600`, `LOW` sensitivity). Server tracks adaptive ambient floor (`_NOISE_MIN=0.003`..`_NOISE_MAX=0.035`, `_ONSET_MIN=0.012`), keeps a 48-frame (~3 s) `listening_ring` + 4-frame preroll so STT arriving before onset yields a contiguous WAV (max 30 s / 960 KB per turn). Watchdog `8.0 s`, silence check-in `15.0 s`, silence close `30.0 s` (ticks only in unmuted `LISTENING`), soft/hard call limits `1140 s / 1200 s` (**S-7.7–S-7.9**).
- **Cue Ownership:** Single owner per cue; immediate cues arm no duplicate `2.2s` fallback; aborted Turn A's completion never unlocks/cancels Turn B; `0.5s` `_STT_WAIT` settles STT before deciding; unprompted `sc.interrupted` during `PROMPTING` with no caller speech rescues once per epoch (`prompt_recued`); skipped tool calls recover on `turn_complete` if speech is banked (**S-7.10–S-7.13**).
- **Commit Ordering, Resume & Nudge:** DB commit precedes tool response/cue (**S-7.14**); semantic ledger + cursor commit atomically (**S-7.15**); telemetry flushes before release and merges via `max` counters + longest latency list on `?resume=<share_token>` (**S-7.16–S-7.19**; 90 s grace `_RESUME_GRACE_SEC`, max 5 Gemini live reconnects, `thinking_level=MINIMAL`, context compression `4096→2048`). Nudge (`3s` silence) hidden/ignored in `DECIDING`/`CLOSING`, evaluates banked speech if present (**S-7.20–S-7.21**).

---

## 8. HTTP, Auth & WebSocket (`app.py`, `call.js`)

| Surface | Auth / Contract |
|---|---|
| Public | `/health` (build stamp), `/login`, `/static/{asset}` (`.css`/`.js`), `/api/login`, `/api/logout`, `/api/pack`, `/api/pack-asset/{logo,consent_pdf}` |
| Caller/Admin | `/`, `/ws/live?pack=&clinic=&resume=`, `POST /api/bugs` (`512 KiB` body, `2000` char note; non-admin needs matching `share_token`) |
| Token-scoped | `GET/POST /api/calls/by-token/{t}/patient.pdf` (`t ≥ 8` chars), `POST /api/calls/by-token/{t}/events` (`share_clicked`) |
| Admin-only | `/admin`, `GET /api/calls`, `/api/calls/{id}`, `/api/bugs`, `/api/audio/{id}`, `GET/POST /api/calls/{id}/patient.pdf` |

- **S-8.1–S-8.6** Single `_gate` middleware (`302` pages, `401`/`403` API); path containment `_inside`; constant-time passcodes + HMAC-SHA256 `shishu_auth` cookie (`v1.role.exp.user.sig`, 14 d, `httpOnly`, `samesite=lax`, `secure` on HTTPS, secret persisted in `server_secrets`). Rate limits: `login` 15/min failures (rightmost proxy hop IP), `pdf` 5/min/token, `share` 20/min/token.
- **Wire S→C:** `session_started`, `question_state`, `audio`, `bot_turn_complete`, `thinking`, `clear`, `transcript`, `telemetry`, `reconnecting`/`reconnected`, `ending`, `call_completed`/`report_failed`, `error`.
- **Wire C→S:** `audio` (16 kHz mono PCM), `text`, `quick_reply` (`value, step, epoch`), `nudge`, `hangup`.
- **S-8.7–S-8.12** Server validates `quick_reply` against active `step` & `epoch`. Browser drops stale epochs, unlocks quick replies at ≥50% played / drained `bot_turn_complete` (fallback `5000 ms`), shows nudge at `3000 ms` silence, enables mic only in unmuted `LISTENING` or `350 ms` tail, reconnects `[600, 1500, 3000]` ms, downloads PDF via plain `<a download>` (no `navigator.share`), stores consent in `sessionStorage` (cleared on logout).

---

## 9. Persistence & Reports (`db.py`, `reports.py`)

- **Tables:** `calls`, `bug_reports`, `server_secrets`, plus append-only ledger `turn_evidence`, `turn_interpretations`, `clinical_claims`, `policy_decisions`. WAL SQLite, `BEGIN IMMEDIATE` commits cursor + ledger atomically (**S-9.1–S-9.2**); read-modify-write JSON columns (`transcript`, `events` capped at 600, `telemetry`) lock single transactions (**S-9.3–S-9.4**).
- **Files:** Wall-clock aligned 16 kHz stereo WAV `audio/call_<id>.wav` (caller L, bot R; `append_existing` on resume, **S-9.5**); atomic temp-file-swap PDF `pdfs/call-<id>.pdf`, non-PII filename `<prefix>-<IST_ts>-c<id>.pdf` (**S-9.6–S-9.7**).
- **Reports (`S-10.1–S-10.6`):** Pure projection of `CallState` + `list_retained_observations(call_id)` (zero network/LLM). Separates `CONCERN`/`CLEAR`/`ANSWERED`/`UNKNOWN`/`UNANSWERED`, omits `NOT_APPLICABLE`, prints fixed bilingual/English state glossaries on both artefacts, shows page-1 emergency/incomplete bands, withholds redacted/multi-topic quotes from `patient_safe`, caps patient doctor-questions at 6 (padded to ≥3 with fillers).

---

## 10. Non-Functional & Deployment

- **N-1..N-9:** Pure post-call (verified by imports); 100% offline testability via fakes; fail-fast full-error pack load; fail-closed safety heads; bounded loops everywhere (`MAX_STEP_ATTEMPTS=2`, 3 strikes, 5 reconnects, 600 events, soft/hard call limits); single-pass turn latency; single-process SQLite; env-only secrets; Dockerized Debian/WeasyPrint (`pango`/`harfbuzz`).
- **Env:** `GEMINI_API_KEY`, `APP_PASSCODE`, `ADMIN_PASSCODE`, `DATA_DIR`, `DEFAULT_CLINIC_CODE`, `INTAKE_PACK`, `PROFILE_EXTRACTOR_MODEL`, `SEED_DEMO_CALLS` (opt-in; replays `seed.yaml` through real reducer, **S-11.1**), `APP_ENV`/`GIT_COMMIT`/`RAILWAY_*`.

---

## 11. Invariants & Verification Evidence

| Invariant / Claim | Enforcement & Measured Evidence |
|---|---|
| **Offline regression suite** | **286 tests, ~14 s**, zero network/keys. Tests follow red-first discipline (**S-12.1**). |
| **Policy & cursor laws** | **36 anchors × 8 patterns = 288 cases** across both packs (`seen ≥ 250` non-collapse guard, **S-12.2**) + per-anchor emergency sweep: zero resolved answers without accepted decision, no cursor drift, no dropped answers. |
| **Fail-closed safety (`danger_now`)** | Missing / `null` / `NaN` → retried then turn failed; `0.97` → `EMERGENCY`. |
| **Transport & recovery** | Per-attempt `12 s` / `18 s` sequential slim retry, 4xx terminal, extractor recovery on Decision timeout, 3-strike close. |
| **Notes & tap arbitration** | Multi-topic/stated withheld from `patient_safe`, retained in `verbatim`; `SOURCE_CONFLICT` rollback vs agreement enrichment. |
| **Live protocol & resume** | Epoch/step-token gating, single-owner cues, `PROMPTING` self-interrupt rescue, adaptive noise floor, real `handle_live_ws` + SQLite drop/resume appending WAV & merging telemetry, atomic DB rollback on trigger failure. |
| **Live persona gates** | 3 scripted calls (red-flag + multi-fact + redaction; mid-call emergency; 3-child middle attribution + retraction): valid stereo WAV, expected red-flag band, zero tool/watchdog errors, zero spoken leaks (system cues / forbidden rare disease names). |
| **Measured reference latency** | 8-turn live benchmark: **2.62 s** median perceived turn latency, **3.68 s** p95, **0** interruptions, **30 ms** median quick-reply reveal. |

---

## 12. Known Boundaries & Open Gaps

| Item | Status |
|---|---|
| Ordered pass (no loops/per-child repeat), fixed `MAX_STEP_ATTEMPTS=2`, 8 kinds, 1 flat roster, fixed head inventory, fixed report model, browser WS only, single process | Design boundary (requires code to change) |
| Speech during bot turn discarded before WAV recorder | **Fixed** against S-7.6 (`live.py` `on_audio` records before the mute gate) |
| Rule operator validation (`_OPS`/`roster_op`) checks names only, not bodies | **Open gap against S-3.7** (`pack.py`) |
| `live.py` (~992 LOC, >40 state attrs) hosts FSM + gate + cues + resume | Deferred refactor (high risk) |
