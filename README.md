# SHISHU V3 — Moldable Voice Patient Intake

SHISHU is an offline-renderable Hindi voice intake call. A caller speaks with the app
before meeting a doctor or health worker; the app asks a pack-defined set of structured
questions, records what was said verbatim, and then projects one finished call into two
artifacts with **zero post-call LLM calls**: a Hindi Patient PDF and an English clinician
note.

V3 is an **engine**, not a rare-disease app. The clinical content — every question, every
Hindi string, every template — lives in a declarative pack under `packs/`. Adding a second
use case costs zero lines of Python.

Design of record: [`docs/v3-design-and-thinking-vad.md`](docs/v3-design-and-thinking-vad.md).
What a pack can and cannot change without touching code:
[`docs/v3-moldability-boundary.md`](docs/v3-moldability-boundary.md).

## The loop

```
passcode -> consent + first-time open-speech priming -> structured voice + quick-tap
         -> deterministic Patient PDF + Clinician Summary
```

Three invariants hold for every pack:

- **Post-call is pure.** Both artifacts are functions of the stored `CallState`.
- **Silence is never an answer.** `AnswerStatus.UNKNOWN` is a real, resolved state and the
  caller's words survive verbatim.
- **One forward pass per turn.** No sidecar pass, no uncached giant prompt.

## Layout

### `intake/` — the engine

| Module | Role |
|---|---|
| `models.py` | The typed vocabulary: `CallState`, `TurnCommand`, `AnswerRecord`, `QuestionSpec`, `PackConfig` |
| `pack.py` | `load_pack`, `list_pack_ids`, and the startup schema validator |
| `engine.py` | Pure FormBharo reducer: eligibility, skip matrix, roster, re-ask policy |
| `audio.py` | 16 kHz PCM resampling and resumable stereo 16 kHz call WAV recorder |
| `decisions.py` | Single-pass `gemini-decision-1` multimodal turn classifier with slim retry fallback |
| `extractor.py` | Schema-validated profile extractor (typed slot values and roster out of free speech) |
| `live.py` | Gemini Live AAD voice puppet, `TurnPhase` + `epoch` state machine, `?resume=<share_token>` continuity |
| `reports.py` | `project_summary` plus the Jinja → WeasyPrint Patient PDF and clinician markdown |
| `db.py` | Atomic SQLite: calls, transcripts, events, bug reports, share tokens |
| `app.py` | FastAPI routes, HMAC cookie auth, sliding-window rate limits, `/ws/live` |
| `static/` | `login.html`, `call.html`, `admin.html`, `styles.css`, `call.js`, `admin.js` |

`app.py` is deliberately thin: it resolves a pack, checks the signed cookie, and hands the
work to `db`, `reports`, and `live`. One middleware, `_gate`, decides access for every
request.

### `packs/` — the clinical content

Two complete packs ship today, and both run on the same engine, reducer, and templates:

- **`packs/shishu-hi/`** — `SHISHU (शिशु)`: the mother-and-children screening call. 62
  questions across child and family groups, a multi-child roster, and a red-flag band.
- **`packs/fever-opd-hi/`** — `SHISHU OPD (बुखार जाँच)`: fever and acute-symptom triage
  for one OPD patient. 14 questions covering duration, nature, severity, and danger signs.

Each pack folder holds the whole surface:

```
packs/<pack_id>/
  pack.yaml        title, subtitle, Gemini model + voice, VAD timings, every UI string
  prompts.yaml     spoken copy and the live session's system prompt
  form.yaml        questions, groups, followups, rules, implications, report config
  decisions.yaml   calibrated thresholds for the eleven Decision API gates
  seed.yaml        scripted regression calls, replayed through the real reducer
  assets/          logo, Devanagari font, consent PDF
  templates/       patient.html.j2, clinician.md.j2, pdf.css
```

`form.yaml` is the contract. `intake/pack.py` validates it at startup: every question's
group, rule, variant, and followup must exist, every choice question needs two options,
and every required threshold must be present. A malformed pack fails fast with the whole
error list rather than at the first bad question.

## Requirements

- Python 3.12 and `uv`
- A Gemini API key
- WeasyPrint's system libraries (`pango`, `harfbuzz`). The `Dockerfile` lists the Linux
  packages; on macOS install them with Homebrew.
- Node.js, only for the frontend VM tests in `tests/test_ui.py`
- The Railway CLI, for deployment

## Configure

Put a `.env` in the repo root. The app reads it at startup.

```
GEMINI_API_KEY=
APP_PASSCODE=
ADMIN_PASSCODE=
DEFAULT_CLINIC_CODE=
DATA_DIR=./intake/storage
```

- `APP_PASSCODE` signs a caller in as `user`; `ADMIN_PASSCODE` signs in as `admin`.
- `DEFAULT_CLINIC_CODE` tags a call when the URL carries no `?clinic=`.
- `DATA_DIR` holds the SQLite database, call WAVs, and generated PDFs.

Do not commit `.env`.

## Run locally

Create the virtual environment in `~/pyenv/shishu` and install with `uv`:

```
uv venv ~/pyenv/shishu --python 3.12
uv pip install --python ~/pyenv/shishu/bin/python -r requirements-dev.txt
```

Start the server from the repo root:

```
PYTHONPATH=. ~/pyenv/shishu/bin/uvicorn intake.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, or go straight to a pack:

- `/?pack=shishu-hi` — the mother-and-children call
- `/?pack=fever-opd-hi` — the OPD fever call
- `/admin` — the clinician dashboard, which needs `ADMIN_PASSCODE`

At startup the lifespan seeds both packs from their `seed.yaml` scenarios, so a fresh
database is never empty. Seeding replays each scripted call through the real reducer and
skips a pack that already has calls.

## Tests

Every test is offline: no network, no API keys.

```
~/pyenv/shishu/bin/python -m unittest tests/test_app.py
```

The full suite, from the repo root:

```
~/pyenv/shishu/bin/python -m unittest discover -s tests
```

`tests/test_app.py` drives the real app through FastAPI's `TestClient`: the auth cookie and
its tamper cases, each of the three rate limiters, `/api/pack` for both packs,
`/api/pack-asset/{kind}` media types and traversal guards, the admin call list with its
pack filter, the patient PDF by both id and share token, `/api/audio`, and `/api/bugs`
ownership. `live.py` is imported lazily inside `/ws/live`, so the HTTP surface stays
testable while the live session is still being built.

## Calibration and persona scripts

Both live in `scripts/` and are pack-aware.

- `scripts/calibrate_decisions.py` rebuilds every case in
  `scripts/decision_calibration_cases.json` into a real `CallState` through the pure
  reducer, calls `intake.decisions` once per case, fits on the `dev` split only, and prints
  `DEV` and `HOLDOUT` tables so overfitting stays visible. `--write` rewrites just the
  `thresholds:` block of a pack's `decisions.yaml`; `--dry-check` builds all cases with no
  network calls.
- `scripts/run_personas.py` boots the app, logs in over `/api/login`, and drives one live
  persona per call over `ws://.../ws/live?pack=<pack_id>`. A persona passes only when the
  call ends with both summaries, the expected red-flag band, a valid Patient PDF, and no
  spoken leak, tool error, or watchdog nudge. Run it inside `tmux`.

```
PYTHONPATH=. ~/pyenv/shishu/bin/python scripts/calibrate_decisions.py --dry-check
PYTHONPATH=. ~/pyenv/shishu/bin/python scripts/calibrate_decisions.py --pack shishu-hi
tmux new -s personas 'PYTHONPATH=. ~/pyenv/shishu/bin/python scripts/run_personas.py'
```

## Deploy (Railway / Docker)

`Dockerfile` installs the WeasyPrint system libraries, copies `intake/` and `packs/`, sets
`PYTHONPATH=/app`, and starts `uvicorn intake.app:app` on `${PORT:-8000}`.
`railway.toml` builds that Dockerfile and restarts on failure.

Before the first deploy, on the Railway service:

1. Set `GEMINI_API_KEY`, `APP_PASSCODE`, and `ADMIN_PASSCODE`.
2. Set `DATA_DIR` to `/data`.
3. Attach a volume at `/data`, so calls, WAVs, and PDFs survive a redeploy.

Then, from the repo root:

```
railway up --detach -m "short summary"
railway deployment list --limit 1 --json
```

`SUCCESS` means the new version is live. If the directory is not linked to the service,
pass `--project`, `--environment`, and `--service`.

## Files that stay off GitHub

`.gitignore` keeps private notes, call transcripts, medical source material, and runtime
storage out of the repo. The packs' own assets and templates are tracked, because a pack
is not runnable without them.
