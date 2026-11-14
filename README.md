# SHISHU

SHISHU is a Hindi voice app for a pre-consultation call.
A mother speaks with the app before she meets a doctor or a health worker.
The live call uses Gemini Live alongside the Gemini Decisions API and a schema-validated profile extractor.
After the call, the mother gets a 2-page Hindi PDF (`mother_report.py`), and the doctor gets an English intake note (`post_call.py`).

## Architecture (`code/shishu-v2/`)

- `triage_catalog.py` — 56 atomic yes/no clinical questions + 25 follow-up prompts
- `call_cursor.py` — deterministic FormBharo state engine, demographic skip matrix, multi-child roster, and dual-intent turn composer
- `decisions.py`, `decision_schema.json`, `decision_thresholds.json` — single-pass `gemini-decision-1` probability turn classifier and calibrated thresholds
- `profile_extractor.py` — schema-validated 4-field demographic extractor (`gemini-3.5-flash-lite`)
- `gemini_live.py`, `live_decision_bridge.py`, `prompts.py` — Gemini Live (`gemini-3.1-flash-live-preview`, `Kore` Hindi voice) WebSocket session, speaker-clock `नहीं`/`हाँ` quick-reply protocol, and silence/reconnect watchdogs
- `post_call.py`, `mother_report.py`, `report_service.py` — pure deterministic English clinician intake note and 2-page Hindi WeasyPrint Mother PDF
- `audio_recorder.py` — off-loop 16 kHz stereo wall-clock WAV recorder
- `app.py`, `db.py`, `seed.py` — FastAPI routes, HMAC cookie auth, UUID `share_token` PDF access, SQLite persistence, and v4 clinical seed calls

The mother PDF stays in Hindi, and the doctor intake note stays in English.

## Sample reports

The sample files are in `docs/samples/`:

- `docs/samples/SHISHU-sample-red-flags.pdf` — Hindi Mother PDF with red flags
- `docs/samples/SHISHU-sample-red-flags-en.pdf` — English version for readers who do not read Hindi
- `docs/samples/SHISHU-sample-no-flags.pdf` — Hindi sample with no yes answers
- `docs/samples/SHISHU-sample-incomplete.pdf` — short Hindi sample from a call that ended early

The call consent is `docs/SHISHU-consent.pdf`.

## Requirements

- Python 3.12 and `uv`
- A Gemini API key
- System libraries required by WeasyPrint (`pango`, `harfbuzz`) and `ffmpeg` (see `Dockerfile` for Linux packages; on macOS via Homebrew)
- The Railway CLI, for deployment

## Configure the app

Put a file named `.env` in the root of the repo. The app reads this file at startup.

```
GEMINI_API_KEY=
APP_PASSCODE=
ADMIN_PASSCODE=
DATA_DIR=./storage
```

`DATA_DIR` is the folder for the SQLite database, call audio WAVs, and generated Mother PDFs.

CAUTION: Do not commit `.env`. It holds the API key and passcodes.

## Run the app locally

From the root of the repo, create the virtual environment in `~/pyenv/shishu` and install dependencies with `uv`:

```
uv venv ~/pyenv/shishu --python 3.12 && uv pip install --python ~/pyenv/shishu/bin/python -r code/shishu-v2/requirements.txt
```

Start the server:

```
PYTHONPATH=code/shishu-v2 ~/pyenv/shishu/bin/uvicorn app:app --app-dir code/shishu-v2 --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and enter your configured `APP_PASSCODE` on the login page.
The `/admin` page uses `ADMIN_PASSCODE`.

## Unit tests & scripts

Run the full offline test suite from the root of the repo (zero network calls, does not require live API keys):

```
~/pyenv/shishu/bin/python -m unittest discover -s tests
```

Calibration and live persona verification scripts live in `scripts/`:

- `scripts/calibrate_decisions.py` — sweep and calibrate `decision_thresholds.json` against `scripts/decision_calibration_cases.json` using `gemini-decision-1`
- `scripts/run_eight_personas.py` — end-to-end live Gemini persona suite (run inside `tmux`)

## Deploy (Railway / Docker)

The deploy uses `Dockerfile` and `railway.toml`. The container writes persistent data to `/data`.

On the Railway service, complete these steps before the first deploy:

1. Set `GEMINI_API_KEY`, `APP_PASSCODE`, and `ADMIN_PASSCODE`.
2. Set `DATA_DIR` to `/data`.
3. Attach a volume at `/data`.

From the root of the repo, deploy:

```
railway up --detach -m "short summary"
```

If this directory is not linked to the service, add `--project`, `--environment`, and `--service`.
Check deployment status with:

```
railway deployment list --limit 1 --json
```

A status of `SUCCESS` means the new version is live.

## Files that stay off GitHub

This repo does not track private notes, call transcripts, medical source files, or runtime storage.
`.gitignore` blocks accidental commits of those files. The sample reports in `docs/samples/` and `docs/SHISHU-consent.pdf` remain in the repo.
