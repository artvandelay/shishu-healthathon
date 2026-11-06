"""
Phone call -> Gemini Live -> summary by email + WhatsApp.

Flow:
  Caller -> Twilio number -> POST /incoming-call (returns TwiML)
         -> Twilio opens WebSocket /media (8 kHz mu-law audio)
         -> this server <-> Gemini Live (16 kHz PCM in, 24 kHz PCM out)
  On hangup -> transcript -> Gemini text model summary -> email + WhatsApp
"""
import asyncio
import audioop  # on Python 3.13+, provided by the audioop-lts package
import base64
import json
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from xml.sax.saxutils import quoteattr

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from google import genai
from google.genai import types
from twilio.rest import Client as TwilioClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gemini-phone")

# ---------- config ----------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# Model names change often. Check the current Live and text model names in the
# Gemini API docs and override these in .env if needed.
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025")
SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-2.5-flash")
VOICE_NAME = os.getenv("GEMINI_VOICE", "Puck")

PUBLIC_HOST = os.getenv("PUBLIC_HOST", "")  # e.g. abc123.ngrok-free.app (no scheme)

TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # Twilio sandbox number
WHATSAPP_TO = os.environ["WHATSAPP_TO"]  # e.g. whatsapp:+91XXXXXXXXXX

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.environ["SMTP_USER"]
SMTP_PASS = os.environ["SMTP_PASS"]  # Gmail: use an App Password
EMAIL_TO = os.environ["EMAIL_TO"]

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a friendly, concise voice assistant on a phone call. "
    "Keep answers short and conversational. Ask a clarifying question if needed.",
)

gemini = genai.Client(api_key=GEMINI_API_KEY)
twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
        )
    ),
)

app = FastAPI()


# ---------- helpers ----------
class Transcript:
    """Merges streamed transcription fragments into speaker turns."""

    def __init__(self):
        self.turns: list[list[str]] = []

    def add(self, speaker: str, text: str | None):
        if not text:
            return
        if self.turns and self.turns[-1][0] == speaker:
            self.turns[-1][1] += text
        else:
            self.turns.append([speaker, text])

    def render(self) -> str:
        return "\n".join(f"{s}: {t.strip()}" for s, t in self.turns)


def send_email(subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


def send_whatsapp(body: str):
    twilio.messages.create(from_=WHATSAPP_FROM, to=WHATSAPP_TO, body=body[:1500])


async def summarize(caller: str, transcript: str) -> str:
    prompt = (
        "Summarize this phone call in 3-5 short bullet points, then list any action "
        "items. Plain text only, no markdown, under 900 characters.\n\n"
        f"Caller: {caller}\n\nTranscript:\n{transcript}"
    )
    resp = await gemini.aio.models.generate_content(model=SUMMARY_MODEL, contents=prompt)
    return (resp.text or "").strip()


async def finish_call(caller: str, call_sid: str | None, transcript: Transcript):
    text = transcript.render()
    if not text.strip():
        log.info("Call %s ended with no transcript; skipping summary", call_sid)
        return
    try:
        summary = await summarize(caller, text)
    except Exception:
        log.exception("Summary generation failed; falling back to raw transcript")
        summary = text[:900]

    message = f"Call summary ({caller})\n\n{summary}"
    results = await asyncio.gather(
        asyncio.to_thread(send_email, f"Call summary from {caller}", f"{message}\n\n--- Transcript ---\n{text}"),
        asyncio.to_thread(send_whatsapp, message),
        return_exceptions=True,
    )
    for name, r in zip(("email", "whatsapp"), results):
        if isinstance(r, Exception):
            log.error("Sending %s failed: %r", name, r)
        else:
            log.info("Sent %s summary for call %s", name, call_sid)


# ---------- Twilio webhook ----------
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    caller = str(form.get("From", "unknown"))
    host = PUBLIC_HOST or request.headers.get("x-forwarded-host") or request.headers.get("host")
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="wss://{host}/media">'
        f"<Parameter name=\"caller\" value={quoteattr(caller)}/>"
        "</Stream></Connect></Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.get("/")
async def health():
    return {"status": "ok"}


# ---------- audio bridge ----------
@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    state = {"stream_sid": None, "call_sid": None, "caller": "unknown"}
    transcript = Transcript()
    started = asyncio.Event()

    try:
        async with gemini.aio.live.connect(model=LIVE_MODEL, config=LIVE_CONFIG) as session:

            async def twilio_to_gemini():
                resample_state = None
                while True:
                    msg = json.loads(await ws.receive_text())
                    event = msg.get("event")
                    if event == "start":
                        s = msg["start"]
                        state["stream_sid"] = s["streamSid"]
                        state["call_sid"] = s.get("callSid")
                        state["caller"] = s.get("customParameters", {}).get("caller", "unknown")
                        started.set()
                        log.info("Call started: %s from %s", state["call_sid"], state["caller"])
                        try:
                            await session.send_realtime_input(
                                text="The call just connected. Greet the caller briefly."
                            )
                        except Exception:
                            log.warning("Could not send greeting prompt", exc_info=True)
                    elif event == "media":
                        ulaw = base64.b64decode(msg["media"]["payload"])
                        pcm8k = audioop.ulaw2lin(ulaw, 2)
                        pcm16k, resample_state = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, resample_state)
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm16k, mime_type="audio/pcm;rate=16000")
                        )
                    elif event == "stop":
                        log.info("Twilio stop event")
                        return

            async def gemini_to_twilio():
                await started.wait()
                resample_state = None
                while True:
                    async for resp in session.receive():
                        sc = resp.server_content
                        if not sc:
                            continue
                        # Caller talked over the assistant: flush Twilio's audio buffer.
                        if sc.interrupted:
                            await ws.send_text(json.dumps({"event": "clear", "streamSid": state["stream_sid"]}))
                        if sc.input_transcription and sc.input_transcription.text:
                            transcript.add("Caller", sc.input_transcription.text)
                        if sc.output_transcription and sc.output_transcription.text:
                            transcript.add("Assistant", sc.output_transcription.text)
                        if sc.model_turn:
                            for part in sc.model_turn.parts or []:
                                if part.inline_data and part.inline_data.data:
                                    pcm8k, resample_state = audioop.ratecv(
                                        part.inline_data.data, 2, 1, 24000, 8000, resample_state
                                    )
                                    payload = base64.b64encode(audioop.lin2ulaw(pcm8k, 2)).decode()
                                    await ws.send_text(
                                        json.dumps(
                                            {
                                                "event": "media",
                                                "streamSid": state["stream_sid"],
                                                "media": {"payload": payload},
                                            }
                                        )
                                    )

            t1 = asyncio.create_task(twilio_to_gemini())
            t2 = asyncio.create_task(gemini_to_twilio())
            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    log.error("Bridge task error: %r", exc)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Call bridge failed")
    finally:
        await finish_call(state["caller"], state["call_sid"], transcript)
