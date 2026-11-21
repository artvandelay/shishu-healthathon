# SHISHU — Product Requirements & User Flows

Hindi voice intake before a clinic visit. Asks a fixed history, preserves the caller's words verbatim, and produces two deterministic artefacts with **zero post-call LLM calls**: a Hindi Patient PDF and an English clinician note. Companion: [`TECH-SPEC.md`](../spec/TECH-SPEC.md).

## 1. Scope & Context

- **Users & setting:** Hindi caller on a browser in a noisy clinic (ambient fan, unstable mobile data) ahead of a visit with a doctor or health worker (ASHA/ANM). No "press 1" IVR; no LLM improvising questions.
- **In scope:** One ordered pass over one patient or one flat child roster; voice answers with optional quick-tap; two pure-projection artefacts.
- **Out by design:** Diagnosis, medical advice, prescriptions, telephony/IVR, messaging (WhatsApp/SMS), EHR, booking, second voice/language provider.
- **Packs (all clinical content; zero in engine):** `shishu-hi` (mother + children, 62 Qs, multi-child roster) and `fever-opd-hi` (OPD fever, 1 patient, 14 Qs).

## 2. Principles

| # | Principle | Forbids |
|---|---|---|
| **P1** | Post-call is pure. | Model/network calls after hangup; reports disagreeing with transcript. |
| **P2** | Silence is never an answer. | Missing answer printed as "नहीं"; infinite retries; skipped Q shown as negative. |
| **P3** | Model proposes, form disposes. | Spoken model picking questions, classifying answers, or advancing cursor. |
| **P4** | Commit before speaking. | Confirmation, cue, or "noted" before DB commit. |
| **P5** | Short allowed, longer welcome. | Punishing elaboration. |
| **P6** | No diagnosis, no all-clear, no advice. | Disease names to caller; "nothing to worry about"; medicine changes. |

## 3. Answer States

| State | Meaning | Patient PDF | Clinician note |
|---|---|---|---|
| `CONCERN` | Affirmed / reported | **हाँ — चिंता की बात** | Concern + verbatim words |
| `CLEAR` | Denied | **नहीं — साफ़ इनकार** | Pertinent negative |
| `ANSWERED` | Open / choice answer | Chosen label / answer | Answered (value) |
| `UNKNOWN` | **Caller** said पता नहीं / untested | **पता नहीं** | Unknown — caller stated |
| `UNANSWERED` | Retry budget spent, no answer | **जवाब नहीं हो पाया — डॉक्टर से बात करें** | Confirm with patient |
| `NOT_APPLICABLE` | Skipped by rules | *omitted* | *omitted* |

First five resolve the step; `NOT_APPLICABLE` never counts as an answer.

## 4. User Flows

- **F1 Sign in:** Pack logo/title/subtitle, name + shared passcode (`user` or `admin`); 14-day signed cookie; wrong passcode never names which field failed; per-IP failure cap.
  *AC-1:* Name travels with call; logout clears cookie & consent; no caller account/OTP.
- **F2 Consent → start:** Checkbox ("मैं सहमत हूँ") + Hindi consent PDF; **Start disabled until checked** (`sessionStorage`, cleared on logout). Requests mic (denial recoverable cleanly), opens WS, greets and asks Q1 in one breath.
  *AC-2:* Reload mid-call keeps consent and call.
- **F3 Profile opening:** Name, marital status (3 choices, voice/tap), child roster (count + ages). Count without ages asks missing ages; twins/triplets expand ("दो जुड़वाँ, 2 साल के" = two 2y items); unmarried / about-to-marry skips children; late correction ("मेरे ट्विंस बच्चे हैं, आप वो सुन नहीं पाए") reopens ages.
  *AC-3:* Never invent `age=0`; never renumber `item_1..n`; max 1 re-ask/step → `UNANSWERED` & advance.
- **F4 Bridge priming:** Explains tap vs free speech, clarifies welcome, asks "शुरू करें?" (tappable/spoken). One-line visual prime (`ui.yes_no_prime`) renders **once** on the first yes/no step only.
  *AC-4:* Works for both screen-watching and speakerphone callers without repeating later.
- **F5 Closed question (tap):** Flanking नहीं / हाँ (or choice pills if `quick_options: true`) unlock once ≥50% of step audio plays or turn drains. Tapping interrupts playback, sends answer, advances; hidden in `DECIDING`.
  *AC-5:* Never answers an unheard question or double-fires; tap-over-SHISHU is the supported barge-in.
- **F6 Closed question (voice):** हाँ, नहीं, or full sentence. Out-of-turn facts auto-fill future eligible gates in the **same turn**; multi-fact turns populate all matching gates; unlisted remarks persist as clinician observations (verbatim, withheld from patient page).
  *AC-6:* Out-of-turn speech never overrides `skip_if` / pruned gates.
- **F7 Follow-up:** Triggered by `CONCERN`; skipped if detail was already stated upfront; free speech kept verbatim.
  *AC-7:* Max 1 re-ask → `UNANSWERED`, preserving the parent concern.
- **F8 Which child:** For `roster_scoped` concerns with ≥2 children: prompts oldest-first with age + kinship word; equal-age twins disambiguated by name/ordinal, never false oldest/youngest rank. Accepts one, multiple, "दोनों", or don't know.
  *AC-8:* Caller words stay on the note even if attribution fails.
- **F9 Clarify:** Explains in 1–2 plain Hindi sentences (never verbatim repeat or disease names), then re-asks. Naming an earlier `UNKNOWN` item clarifies it without answering the active one.
  *AC-9:* Never burns a retry attempt or closes the call (including at `open_capture`).
- **F10 Off-topic & silence:** Clinical aside → saved as observation, acknowledged once, question repeated. Non-clinical → ignored, re-asked (after budget → `UNANSWERED`). 15 s silence → check-in; 30 s → close. Nudge button appears after 3 s silence; if pending speech exists, nudge **evaluates it** instead of re-asking.
  *AC-10:* No infinite loops; off-topic never fabricates an answer.
- **F11 Retraction:** Reopens an open/`UNKNOWN` item or updates a settled one. Recorded spoken `CONCERN` notes are never wiped to bare "no".
  *AC-11:* Correction acknowledged before next question.
- **F12 Tap vs speech conflict:** Neither source wins silently. Cursor rolls back to pre-tap, holds step, stores both evidences + `SOURCE_CONFLICT`, prompts one confirmation. Spoken emergency or stop always overrides tap.
  *AC-12:* Conflict logged; cursor stays put until resolved.
- **F13 Mid-call drop:** Shows "लाइन जुड़ रही है…", keeps timer/mic, attempts 3 reconnects (`?resume=<token>`) to the **same call**. Rejoin re-speaks active step, appends same WAV, deduplicates caption tail. 90 s unresumed → finalise incomplete.
  *AC-13:* Never re-asks answered Qs, resets timers/counters, or re-shows consent.
- **F14 Call end:** One closing line, one `CloseReason`: `COMPLETE` (clear vs concern wording — never "you are fine"), `EMERGENCY` (nearest hospital / 108, history noted), `CALLER_STOP`, `SILENCE` (30 s), `TIME_LIMIT` (19 min soft / 20 min hard), `TECHNICAL_FAILURE` (3 consecutive dead turns). Non-complete closings state the report is incomplete and invite calling back.
  *AC-14:* Spoken once; emergency preempts any queued prompt.
- **F15 Post-call PDF:** Spinner ("रिपोर्ट तैयार हो रही है…") → download button (arrow icon + server filename; anchor download only, no `navigator.share` / copy link). Build failure shows "रिपोर्ट दोबारा बनाएँ"; "नई कॉल" resets UI without reload.
  *AC-15:* Addressed by unguessable `share_token`; bearer sees only their own PDF.
- **F16 In-call bug report:** Modal over live call (category chip, optional note, optional screenshot); saves client-state snapshot + copies JSON (copy failure still saves).
  *AC-16:* Never interrupts call; strangers cannot attach another call's snapshot.
- **F17 Clinician review (`/admin`):** Pack-filterable call list (flags, caller, roster, model, pack, clinic, duration, time); detail view with stereo WAV player, red-flag banner, English note, Hindi summary, answers (`concern → unknown → unanswered → answered → clear`), transcript, KPIs + event trace, copy/print/PDF/debug-bundle; bug tab with screenshots + call jump.
  *AC-17:* Red flags at top; `NOT_APPLICABLE` hidden.

## 5. Copy & Voice Rules

- Warm female voice, simple Hindi only (even if a father speaks).
- Zero rare/genetic disease names in HI or EN; everyday words (`काँच का बक्सा`, `नीली लाइट`, `नाल`, `झटके`, `तिल्ली`).
- Never say nothing was found, doctor not needed, or advise changing medicines.
- Re-ask with fresh phrasing (not louder); append "जी" to known caller names; non-judgemental on children/marriage/miscarriage/home birth; never speak in caller's voice or read system cues aloud.

## 6. Artefacts & Instrumentation

- **Patient PDF (Hindi):** *Page 1:* logo, ref, IST date/time/duration, complete/incomplete badge, caller/roster, non-diagnosis notice, emergency/incomplete band, 3–6 questions for doctor (actual concerns + routine/emergency fillers), "yes" bullets, universal general-advice line. *Page 2:* 4-line state key (`हाँ / नहीं / पता नहीं / जवाब नहीं हो पाया`), grouped `CLEAR`/`ANSWERED`, separate `UNKNOWN` and `UNANSWERED` lists, unasked groups if incomplete.
- **Clinician note (English):** Profile + chief complaint, red flags/gate alerts, grouped sections (intent + child + verbatim words), pertinent negatives, unlisted observations, close reason, fixed response-state glossary (unanswered ≠ denied).
- **Telemetry:** Caller/bot turns, tool calls/errors, audio chunks in/out, interruptions, reconnects, latency samples (avg/max), build stamp, model/voice, capped event trace.

## 7. Operational Limits & New-Pack DoD

- **Limits:** Single forward pass (no loops/sub-forms/per-child block repeats); 8 answer kinds (no dates, unit-validated vitals, computed scores, in-flow uploads); 1 flat roster; fixed report projection model (templates only); browser WS only; single server process (for resume); speech over SHISHU is not a caller turn (half-duplex tap-over avoids echo storms); classifier is a hard runtime dependency (3 failures end call).
- **New-pack DoD (zero Python):** Valid when intake is 1 ordered pass, 1 patient or flat roster, voice + optional yes/no or single-choice taps, 8 supported kinds, and positive / negative / unknown / unasked reporting. Contract: [`TECH-SPEC.md`](../spec/TECH-SPEC.md) §4.
