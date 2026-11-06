# Gemini Live phone line with email + WhatsApp call summary

## 1. One-time setup
1. Twilio: buy a voice-capable number. On a trial account, add your mobile under
   Verified Caller IDs so you can call it.
2. WhatsApp sandbox: Twilio Console > Messaging > Try it out > Send a WhatsApp message.
   From your WhatsApp, send the shown `join <code>` message to the sandbox number.
3. Gmail: enable 2-step verification, then create an App Password and use it as `SMTP_PASS`.
4. Gemini: create an API key in Google AI Studio.

## 2. Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the values
uvicorn server:app --port 8000
```

In a second terminal:
```bash
ngrok http 8000
```
Copy the `https://....ngrok-free.app` URL.

## 3. Connect Twilio
Twilio Console > Phone Numbers > your number > Voice Configuration:
- "A call comes in": Webhook, `POST`, `https://<your-ngrok-host>/incoming-call`

## 4. Test
Call the number from your verified phone, talk, interrupt it once, then hang up.
Within a few seconds you should get the summary by email and WhatsApp.
Watch the uvicorn logs if either is missing.

## 5. Deploy (when ready)
Deploy to Cloud Run or Fly.io (needs a public HTTPS URL that supports WebSockets),
then replace the ngrok URL in the Twilio webhook.

## Notes
- WhatsApp sandbox sessions expire after a while; re-send the `join` message if messages stop arriving.
- Before sharing the number publicly, validate Twilio's request signature on `/incoming-call`
  and add call-length limits, since every call uses Gemini and Twilio minutes.
- SMS to Indian numbers needs DLT registration; that is why this version uses WhatsApp and email.
