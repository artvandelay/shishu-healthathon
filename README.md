# SHISHU

SHISHU is a Hindi voice app for a pre-consultation call.
A mother speaks with the app before she meets a doctor or a health worker.
The live call uses Gemini.
After the call, the doctor gets an English report.
The mother gets Hindi questions for her next visit.

The current service is https://shishu-production.up.railway.app.
Open that address.
Enter any name on the login page.
For a call, enter the passcode `shishu2026`.

## Requirements

You need these items:

- Python 3.12
- A Gemini API key
- The Railway CLI, for a deploy

## Configure the app

Put a file named `.env` in the root of the repo.
The app reads this file at start.

```
GEMINI_API_KEY=
APP_PASSCODE=
ADMIN_PASSCODE=
OPENAI_API_KEY=
DATA_DIR=./storage
```

`OPENAI_API_KEY` is optional.
The live call uses Gemini.
`DATA_DIR` is the folder for the database and the call audio.

CAUTION: Do not commit `.env`.
The file holds the API keys and the passcodes.

## Run the app on this computer

Go to the root of the repo.
Create a virtual environment.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r code/shishu-v2/requirements.txt
```

Start the server.

```
PYTHONPATH=code/shishu-v2 uvicorn app:app --app-dir code/shishu-v2 --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.
Enter `APP_PASSCODE` on the login page.
The admin page uses `ADMIN_PASSCODE`.

## Unit tests

Go to `code/shishu-v2`.
Run the unit tests.

```
python -m unittest test_bugfixes test_triage_flow
```

The tests do not call the network.
The tests do not read `.env`.

## Deploy

The deploy uses `Dockerfile` and `railway.toml`.
The image writes data to `/data`.

On the Railway service, complete these steps before the first deploy:

1. Set `GEMINI_API_KEY`, `APP_PASSCODE`, and `ADMIN_PASSCODE`.
2. Set `DATA_DIR` to `/data`.
3. Attach a volume at `/data`.

The volume keeps the database and the call audio after a new deploy.

From the root of the repo, upload this directory.

```
railway up --detach -m "short summary"
```

If this directory is not linked to the service, add `--project`, `--environment`, and `--service`.

`railway up --detach` returns after the upload.
The build is not complete at that moment.
Read the deployment status first.

```
railway deployment list --limit 1 --json
```

A status of `SUCCESS` means that the new version is in service.

## Files that stay off GitHub

This repo does not track private notes, call transcripts, medical source files, or old app versions.
Those files can stay on this computer.
`.gitignore` blocks a new commit of those files.
