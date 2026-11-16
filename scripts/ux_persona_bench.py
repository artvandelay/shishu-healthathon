#!/usr/bin/env python3
"""Meena Devi (मीना देवी) spoken-Hindi UX benchmark for one SHISHU target.

Real 16 kHz Hindi (`say -v Lekha` -> ffmpeg, sha1-cached) down the same WebSocket the browser uses; every
millisecond between her mouth closing and SHISHU's first audio chunk is recorded.

    scripts/ux_persona_bench.py --target prod | v3 [--out-dir DIR]

A voice caller hears the *audio*, not the caption, so a short caption is booked as the UI defect
`caption_truncated` instead of triggering a "please repeat" deadlock; on V3 the responder also gets the
pack's true wording for the current step. Artifacts: `{target}_ux_log.json`,
`{target}_call_detail.json`, `{target}.pdf`, `tts_cache/`.
"""
from __future__ import annotations

import argparse, asyncio, base64, hashlib, io, json, os, re, subprocess, sys, tempfile, time
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx, numpy as np
from dotenv import load_dotenv

RATE, PAUSE, OUT_RATE, GAP = 16_000, 0.55, 24_000, 8_800  # mic, `[PAUSE]` secs, Live out, gap samples
FRAME = {"prod": 2048, "v3": 1024}  # call.html ScriptProcessor vs call.js worklet
WS = {"prod": "/ws/gemini", "v3": "/ws/live?pack={pack}"}
BASE = {"prod": "https://shishu-production.up.railway.app", "v3": "http://127.0.0.1:8093"}
PDF = {"prod": "mother.pdf", "v3": "patient.pdf"}
GRACE, SETTLE, SETTLE_MAX, QUIET = 0.35, 0.75, 2.0, 0.85  # her gap, caption settling, bot-quiet fallback
OUTRO, OUTRO_MAX, POST_HANGUP, NUDGE, NUDGE_MS = 2.2, 30.0, 60.0, 14.0, 3_000.0
TRUNC_CHARS, TRUNC_MS, LEAK = 22, 1_800, "हीमोफीलिया"
MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite-preview")
ROOT, T0 = Path(__file__).resolve().parents[1], time.perf_counter()
log = lambda text="": print(f"[{time.perf_counter() - T0:7.2f}s] {text}", flush=True)
def snippet(text: Any, limit: int = 78) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
def stat(values: list[float], fn: Any) -> float:
    return round(float(fn(values)), 1) if values else 0.0

# --------------------------------------------------------------- Meena Devi's clinical sheet
NICU = "हाँ डॉक्टर [PAUSE] जन्म के बाद आरव को चार दिन काँच के बॉक्स में रखना पड़ा था, उसे साँस लेने में थोड़ी तकलीफ़ थी।"
NICU_MORE = "चार दिन रखा था, साँस लेने में थोड़ी दिक्कत थी, फिर बिल्कुल ठीक हो गया था।"
BOTH = "हाँ डॉक्टर, कभी-कभी खेलते समय आरव के दाएँ घुटने में सूजन आ जाती है, और छोटी चोट लगने पर खून दस-पंद्रह मिनट तक बहता रहता है और नील भी पड़ जाते हैं।"
BOTH_AGAIN = "जी डॉक्टर, जैसा मैंने बताया, उसके दाएँ घुटने में सूजन आ जाती है और चोट लगने पर खून दस-पंद्रह मिनट तक बहता रहता है।"
FAM = "हाँ डॉक्टर [PAUSE] मेरे सगे मामा के बेटे को हीमोफीलिया की बीमारी बताई थी, उसे चोट लगने पर खून बहुत देर तक बंद नहीं होता था।"
FAM_REL = "वह मेरे सगे मामा का बेटा है, यानी मेरा ममेरा भाई है।"
MATERNAL = "जी हाँ डॉक्टर, मायके तरफ़ मेरे सगे मामा के बेटे को ही खून देर तक बहने की समस्या है, और किसी को नहीं।"
CLOSING = "बस डॉक्टर, यही पूछना था कि आरव के घुटने की सूजन और खून देर से रुकने के लिए किस डॉक्टर को दिखाना चाहिए।"
CLOSING_NOTE = "अरे हाँ, आरव के घुटने में सूजन आ जाती है और चोट पर खून देर तक बहता है, यही डॉक्टर को बताना है।"
CLOSING_DONE, NEUTRAL = "नहीं जी, बस इतना ही, बहुत धन्यवाद।", "नहीं जी डॉक्टर, ऐसा कुछ नहीं हुआ।"
LINES = dict(NICU=NICU, NICU_MORE=NICU_MORE, BOTH=BOTH, BOTH_AGAIN=BOTH_AGAIN, FAM=FAM, FAM_REL=FAM_REL,
             MATERNAL=MATERNAL, CLOSING=CLOSING, CLOSING_NOTE=CLOSING_NOTE, CLOSING_DONE=CLOSING_DONE)
SHEET = """You are Meena Devi (मीना देवी), an ordinary Hindi-speaking mother on a call with SHISHU, a Hindi health-intake assistant asking about her only child. Speak Devanagari Hindi only, 1-2 short sentences, everyday rural Hindi. SHISHU may ask in Roman letters ("kaanch ke box"): read it as Hindi, still answer in Devanagari. No stage directions, quotes, brackets or bullets: the only non-Hindi token you may ever emit is [PAUSE], and only inside triggers 1 and 3. Match the polarity asked: a false "क्या ... है?" gets "नहीं जी ...", a true one gets "हाँ जी ...". If the wording arrives short or garbled, answer what was clearly asked from this sheet. Never invent a symptom, child, pregnancy, hospital stay or relative, never ask her to repeat, and state every fact identically every time.
SHEET
- मीना देवी. शादीशुदा -> "हाँ जी, मैं शादीशुदा हूँ।" रिश्तेदारी में नहीं -> "नहीं डॉक्टर, हमारी शादी रिश्तेदारी या परिवार में नहीं हुई है।" एक ही बेटा आरव, छह साल -> "जी मेरा एक ही बेटा है, उसका नाम आरव है और उसकी उम्र छह साल है।" शुरू करने को तैयार -> "हाँ जी डॉक्टर, पूछिए।"
- जन्म: नॉर्मल डिलीवरी, ऑपरेशन नहीं -> "नहीं जी, ऑपरेशन नहीं हुआ था, नॉर्मल डिलीवरी हुई थी।" पूरे नौ महीने, समय पर (समय से पहले नहीं! अगर पूछे "समय से बहुत पहले हुआ था?" -> "नहीं जी, पूरे नौ महीने पर समय पर हुआ था।"; अगर पूछे "पूरे नौ महीने पर हुआ था?" -> "हाँ जी, पूरे नौ महीने पर हुआ था।") NICU हाँ -> {NICU} क्यों या कितने दिन -> "{NICU_MORE}"
- हर दूसरा जन्म सवाल सामान्य/नहीं: पीलिया या नीली लाइट, जन्म पर इमरजेंसी, नाल तीन हफ्ते बाद या खून, BCG पर फोड़ा या गांठ, होंठ या तालु कटा, कोई जन्म दोष, सुनना या दिखना। सुनना-दिखना ठीक -> "हाँ जी, बिल्कुल ठीक सुनाई और दिखाई देता है।"
- खून देर तक बहना या घुटने में सूजन: दोनों हाँ, पहली बार में दोनों एक साथ -> {BOTH} वही दोबारा पूछे जाएँ -> "{BOTH_AGAIN}"
- हर दूसरा शरीर सवाल नहीं: दौरे, देर से विकास, सीखा हुआ भूलना, होंठ या उंगली नीली, दिल में मर्मर या छेद, खून चढ़ाना या कमी, तेज़ दर्द, पेट या लिवर या तिल्ली, जन्म से गांठ, कद या वजन, छोटे अंग या अंगूठा, हड्डी टूटना या टेढ़े पैर, पेशाब या प्यास या पथरी, गुप्त अंग, जल्दी या देर से यौवन। दाँत या मसूड़े से खून -> "नहीं डॉक्टर, दाँत या मसूड़े से कभी ज़्यादा खून नहीं आया।"
- हर बीमारी सवाल नहीं: सुस्ती या उल्टी, पेशाब या मल की बदबू, नवजात के बाद भर्ती, बार-बार इंफेक्शन पर ड्रिप -> "नहीं डॉक्टर, जन्म के बाद कभी अस्पताल में भर्ती नहीं करना पड़ा, ऐसी कोई परेशानी नहीं है।"
- प्रेग्नेंसी में थायरॉइड और दवाई, डिलीवरी के बाद ठीक, बीपी या शुगर नहीं -> "हाँ डॉक्टर, प्रेग्नेंसी के दौरान मुझे थायरॉइड बताया था और दवाई चली थी, लेकिन डिलीवरी के बाद ठीक हो गया। बीपी या शुगर नहीं था।" थायरॉइड अब भी चल रहा है -> "नहीं जी, डिलीवरी के बाद थायरॉइड ठीक हो गया था, अब दवाई बंद है।" गर्भवती हुई -> "हाँ जी, मैं गर्भवती हुई हूँ।"
- थैलेसीमिया की जाँच कभी नहीं हुई -> "नहीं डॉक्टर, प्रेग्नेंसी में थैलेसीमिया की जाँच नहीं हुई थी।" ("कोई परेशानी नहीं थी" कभी न जोड़ें।) बाकी हर प्रेग्नेंसी सवाल सामान्य: सोनोग्राफी, भ्रूण का दिल, उसे खून चढ़ाना, प्रसव में खून।
- परिवार में खून देर तक बहना: हाँ, बीमारी का नाम भी -> {FAM} वह कौन है -> "{FAM_REL}" मायके की तरफ पूछे तो सिर्फ़ -> {MATERNAL}
- हर दूसरा परिवार सवाल नहीं: बार-बार गर्भपात या मृत बच्चा, एक साल से पहले मौत, दौरे या सीखने में देरी, खून चढ़ाना, अंधापन या बहरापन, कैंसर, पचास से पहले दिल का दौरा, किडनी फेल या डायलिसिस, चालीस से पहले बीपी या लकवा, बार-बार गंभीर इंफेक्शन, खुजली व एक्ज़िमा, गांठ, किसी और को वही समस्या -> "नहीं डॉक्टर, परिवार में ऐसी कोई समस्या किसी को नहीं है।"
- कुछ और बताना या पूछना चाहेंगी, पहली बार -> {CLOSING} क्या नोट करूँ -> "{CLOSING_NOTE}" फिर कुछ और -> "{CLOSING_DONE}"
TRIGGERS (state of this call): {TRIGGERS}
Output only Meena's next spoken line."""

def sheet_for(t: dict[str, bool]) -> str:
    said = lambda k: "ALREADY GIVEN — never again" if t[k] else "not given yet"  # noqa: E731
    return SHEET.format(**LINES, TRIGGERS="; ".join((f"1 NICU (uses [PAUSE]): {said('nicu')}",
        f"2 bleeding + knee swelling: {said('compound')}", f"3 family bleeding (uses [PAUSE]): {said('fam')}",
        f"4 the closing question: {said('closing')}", "5 Never emit [PAUSE] outside triggers 1 and 3, never twice.")))

def clean_reply(raw: Any) -> str:
    """Keep only the spoken Devanagari line, whatever shape the model returned it in."""
    text = " ".join(str(raw or "").replace("[PAUSE]", "\x00").split())
    text = re.sub(r"[A-Za-z]+", " ", (text.split(":", 1)[1] if ":" in text[:12] else text)
                  .translate({ord(c): " " for c in '"\'“”‘’*`•—#'}))
    text = " ".join(text.replace("\x00", " [PAUSE] ").split()).strip().rstrip(".। ")
    return text if not text or text.endswith(("?", "!")) else f"{text}।"

FALLBACKS = (  # (probe needles, first answer, its repeat, trigger key)
    (("मायके", "maayke", "mayke"), MATERNAL, MATERNAL, None),
    (("कुछ और", "अलावा", "kuch aur", "alawa"), CLOSING, CLOSING_DONE, "closing"),
    (("काँच", "कांच", "kaanch", "kanch", "nicu"), NICU, NICU_MORE, "nicu"),
    (("घुटन", "जोड़", "ghutn", "jod", "नील", "neel"), BOTH, BOTH_AGAIN, "compound"),
    (("परिवार", "parivaar", "family", "मामा"), FAM, FAM_REL, "fam"))

def fallback_reply(question: str, triggers: dict[str, bool]) -> str:
    """Reduced sheet answer when the responder is down; the four red-flag triggers still fire."""
    text = " ".join(str(question or "").lower().split())
    for needles, first, again, key in FALLBACKS:
        if any(n in text for n in needles):
            return again if key and triggers[key] else first
    return NEUTRAL

def looks_truncated(caption: str, audio_ms: float) -> bool:
    """A short caption behind long audio is a UI defect, never a reason to ask her to repeat.

    Production speaks complete Romanized Hinglish captions ending in an ASCII full stop, so the old "must
    end with ? or ।" rule called them fragments and deadlocked the call 16 turns running. The caller heard
    the audio; long audio always wins.
    """
    flat = " ".join(str(caption or "").split())
    return len(flat) < TRUNC_CHARS and not flat.endswith(("?", "।", ".", "!")) and audio_ms > TRUNC_MS

def pack_question(pack_id: str, step: str) -> str:
    """The wording SHISHU actually speaks for a V3 `step` token, rebuilt from the pack itself."""
    qid, _, rest = str(step or "").partition(":")
    head, _, arg = rest.partition(":")
    if not qid or head not in ("", "main", "followup"):
        return ""  # which_item / hold_slot / open-capture rounds need live call state
    sys.path.insert(0, str(ROOT))
    from intake.engine import active_prompt
    from intake.models import CallState, CursorStep, SubStep
    from intake.pack import load_pack
    cursor = CursorStep(qid=qid)
    if head == "followup":
        cursor = cursor.model_copy(update={"substep": SubStep.FOLLOWUP, "followup_idx": int(arg or 0)})
    try:
        return " ".join(active_prompt(load_pack(pack_id), CallState(cursor=cursor)).split())
    except Exception:  # noqa: BLE001 - a pack hiccup must not stop the benchmark
        return ""

# ----------------------------------------------------------------- real 16 kHz spoken Hindi
def tts_segment(text: str, cache: Path) -> np.ndarray:
    """One sentence as 16 kHz mono int16, sha1-cached so a rerun only pays for new sentences."""
    path = cache / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}.pcm"
    if path.is_file() and path.stat().st_size >= 2:
        return np.frombuffer(path.read_bytes(), dtype=np.int16)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shishu-tts-") as tmp:
        aiff, raw = Path(tmp) / "say.aiff", Path(tmp) / "say.pcm"
        subprocess.run(["say", "-v", "Lekha", "-o", str(aiff), text], check=True, capture_output=True, timeout=120)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(RATE), "-ac", "1",
                        "-f", "s16le", str(raw)], check=True, capture_output=True, timeout=120)
        pcm = np.frombuffer(raw.read_bytes(), dtype=np.int16).copy()
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) >= 250)  # trim silence, keep 20 ms of plosive pad
    trimmed = pcm[max(0, int(loud[0]) - 320): int(loud[-1]) + 321] if loud.size else pcm
    path.write_bytes(trimmed.astype(np.int16).tobytes())
    return trimmed

def synthesize_utterance(text: str, cache_dir: Path) -> tuple[bytes, bool]:
    """16 kHz mono PCM16LE for `text`; a literal `[PAUSE]` becomes exactly 550 ms of silence."""
    parts = [part.strip() for part in str(text or "").split("[PAUSE]")]
    spoken = [part for part in parts if part]
    if not spoken:
        return b"", len(parts) > 1
    out = tts_segment(spoken[0], Path(cache_dir))
    for piece in spoken[1:]:  # splice the exact 8 800-sample gap between spoken segments
        out = np.concatenate([out, np.zeros(GAP, np.int16), tts_segment(piece, Path(cache_dir))])
    return out.astype(np.int16).tobytes(), len(parts) > 1

def merge_caption(prev: str, chunk: Any) -> str:
    """Append one caption fragment, tolerating the cumulative re-sends a live caption does."""
    chunk = " ".join(str(chunk or "").split())
    if not chunk or prev.endswith(chunk):
        return prev
    if not prev or chunk.startswith(prev):
        return chunk
    return prev + ("" if prev[-1] in " ।,.;:!?" or chunk[0] in " ।,.;:!?" else " ") + chunk

def server_summary(detail: dict[str, Any]) -> dict[str, Any]:
    """What the server recorded; V3 keeps a typed `cursor_state`, prod stores flat text columns."""
    if not detail:
        return {}
    load = lambda raw: (json.loads(raw) if isinstance(raw, str) else raw) or {}  # noqa: E731
    cursor = detail.get("cursor_state") or {}
    answers = cursor.get("answers") or load(detail.get("answers_json")).get("answers") or {}
    verdict = lambda rec: str(rec.get("state") or rec.get("answer") or "").lower() if isinstance(rec, dict) else ""  # noqa: E731
    telemetry = load(detail.get("telemetry"))
    latencies = [float(v) for v in telemetry.get("turn_latencies_ms") or () if isinstance(v, (int, float))]
    return {"schema": "v3-cursor_state" if cursor else "prod-flat", "status": detail.get("status"),
            "close_reason": cursor.get("close_reason"), "duration_sec": detail.get("duration_sec"),
            "patient_name": detail.get("patient_name") or detail.get("mother_name"), "answers": len(answers),
            "concerns": sorted(q for q, r in answers.items() if verdict(r) in ("concern", "yes", "positive")),
            "red_flags": len(detail.get("red_flags") or ()), "server_turns": telemetry.get("user_turns"),
            "server_latency_median_ms": stat(latencies, np.median),
            "has_doctor_summary": bool(str(detail.get("summary_doctor") or "").strip()),
            "has_patient_summary": bool(str(detail.get("summary_patient") or detail.get("summary_mother") or "").strip())}

def pdf_text(payload: bytes) -> str:
    """Extract report text, so the log can prove the diagnosis never leaked into the PDF."""
    try:
        import pymupdf

        with pymupdf.open(stream=payload, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:  # noqa: BLE001 - extraction is a nice-to-have on top of the saved bytes
        try:
            from pypdf import PdfReader

            return "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(payload)).pages)
        except Exception:  # noqa: BLE001
            return ""
# ------------------------------------------------------------------------------- the bench


class Bench:
    """One Meena Devi call against one target, measured end to end."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg, self.target = cfg, cfg.target
        self.base = (cfg.base_url or BASE[cfg.target]).rstrip("/")
        self.out, self.tts = Path(cfg.out_dir).resolve(), Path(cfg.out_dir).resolve() / "tts_cache"
        self.fb, self.fs = FRAME[self.target] * 2, FRAME[self.target] / RATE  # frame bytes, frame seconds
        self.phase, self.t0, self.started = "init", time.perf_counter(), ""
        self.http = self.ws = self.client = None
        self.cookie, self.call_id, self.token, self.model, self.limit = "", 0, "", "", ""
        self.lock = asyncio.Lock()
        self.silence = base64.b64encode(bytes(self.fb)).decode("ascii")
        self.utter: deque[bytes] = deque()  # her mouth; `waiting` means closed and owed a reply
        self.played, self.waiting, self.speech_end_at, self.mouth_at = 0.0, False, None, 0.0
        self.turn, self.reply_task, self.thinking = None, None, False
        self.turns, self.questions, self.history, self.closed = [], [], [], []
        self.nudges = self.cutoffs = self.clears = self.fallbacks = 0
        self.bot, self.orphan, self.orphan_at = None, "", 0.0
        self.primed, self.step, self.ui = False, "", {}
        self.triggers = dict(nicu=False, compound=False, fam=False, closing=False)
        self.http_rtt = self.ws_rtt = self.ending_at = self.hangup_at = 0.0
        self.done = self.completed = self.failed = False

    async def ping(self) -> float:
        """One round trip, on our own clock when the library hands us nothing better."""
        started = time.perf_counter()
        try:
            if isinstance(m := await asyncio.wait_for(await self.ws.ping(), timeout=10.0), (int, float)) and m > 0:
                return float(m) * 1000.0
        except Exception:  # noqa: BLE001 - fall back to our own clock
            pass
        return (time.perf_counter() - started) * 1000.0

    async def connect(self) -> None:
        """Health RTT baseline, admin login, the V3 pack UI copy, then the live socket."""
        import websockets
        self.http = httpx.AsyncClient(base_url=self.base, timeout=60.0, follow_redirects=True)
        await self.http.get("/health")  # warm-up: the first request pays connection setup
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            (await self.http.get("/health")).raise_for_status()
            samples.append((time.perf_counter() - started) * 1000.0)
        self.http_rtt = stat(samples, np.median)
        log(f"[{self.target}] HTTP /health median {self.http_rtt} ms ({' '.join(f'{s:.0f}' for s in samples)})")
        passcode = (os.environ.get("ADMIN_PASSCODE") or "").strip()
        resp = await self.http.post("/api/login", json={"username": "MeenaUXBench", "passcode": passcode})
        resp.raise_for_status()
        self.cookie = self.http.cookies.get("shishu_auth") or resp.cookies.get("shishu_auth") or ""
        if not self.cookie:
            raise RuntimeError(f"login returned no shishu_auth cookie ({resp.text[:200]})")
        if self.target == "v3":
            data = (await self.http.get("/api/pack", params={"pack": self.cfg.pack})).json() or {}
            prime, ui = (data.get("ui") or {}).get("yes_no_prime") or {}, data.get("ui") or {}
            self.ui = {"title": data.get("title", ""), "yes_no_buttons": ui.get("yes_no_buttons") or [],
                       "yes_no_prime": (prime.get("hi") if isinstance(prime, dict) else str(prime)) or ""}
        url = (f"{'wss' if self.base.startswith('https') else 'ws'}://{urlparse(self.base).netloc}"
               f"{WS[self.target].format(pack=self.cfg.pack)}")
        log(f"[{self.target}] connecting {url}")
        self.ws = await websockets.connect(url, additional_headers={"Cookie": f"shishu_auth={self.cookie}"},
                                          max_size=8 << 20, ping_interval=20, ping_timeout=30)
        await self.ping()  # warm-up absorbs post-upgrade TLS/HTTP lag
        self.ws_rtt = stat([await self.ping() for _ in range(3)], np.median)
        log(f"[{self.target}] WS ping median {self.ws_rtt} ms | frame {FRAME[self.target]} samples ({self.fs * 1000:.0f} ms)")

    async def send(self, payload: dict[str, Any]) -> None:
        try:
            async with self.lock:
                await self.ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - the receive loop notices a closed socket
            pass

    async def mic_loop(self) -> None:
        """Stream like an open microphone: her speech, otherwise exact-silence frames."""
        next_at = time.perf_counter()
        while self.phase == "live":
            now, delay = time.perf_counter(), next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(min(delay, 0.25))
                continue
            if self.utter:
                await self.send({"type": "audio", "data": base64.b64encode(self.utter.popleft()).decode(),
                                 "sample_rate": RATE})
                if self.turn is not None and self.turn["t_speech_start"] is None:
                    self.turn["t_speech_start"] = round(now - self.t0, 3)
                if not self.utter:  # her mouth just closed: this instant is the latency zero point
                    self.mouth_at = now
                    if self.turn is not None:
                        self.turn["t_mouth_closed"] = round(now - self.t0, 3)
                    self.waiting = True
                    self.speech_end_at = now + PAUSE if self.target == "prod" else None
            else:
                await self.send({"type": "audio", "data": self.silence, "sample_rate": RATE})
                if self.speech_end_at is not None and now >= self.speech_end_at:
                    self.speech_end_at = None
                    await self.send({"type": "speech_end"})
            next_at += self.fs
            if next_at < now - 1.0:  # fell far behind: resync instead of bursting frames
                next_at = time.perf_counter()

    def bot_audio(self, msg: dict[str, Any], now: float) -> None:
        """Queue one audio chunk, then book the two UI facts only real audio can reveal."""
        try:
            raw = base64.b64decode(msg.get("data") or "")
        except Exception:  # noqa: BLE001 - a malformed chunk must not kill the call
            return
        if not raw:
            return
        duration = len(raw) / (2.0 * (int(msg.get("sample_rate") or 0) or OUT_RATE))
        self.played = max(now, self.played) + duration
        if self.utter and self.turn is not None:  # SHISHU talked over her mid-sentence
            self.turn |= {"premature_cutoff": True, "unspoken_ms": round(len(self.utter) * self.fs * 1000.0, 1)}
            self.cutoffs += 1
            self.utter.clear()
            log(f"    ⚡ premature cutoff: {self.turn['unspoken_ms']:.0f} ms of her sentence was never heard")
        elif self.waiting and self.turn is not None and self.mouth_at:
            latency = round((now - self.mouth_at) * 1000.0, 1)
            self.turn |= {"t_first_bot_audio": round(now - self.t0, 3), "perceived_latency_ms": latency,
                          "rtt_adjusted_latency_ms": round(max(0.0, latency - self.ws_rtt), 1),
                          "nudge_btn_would_appear": latency > NUDGE_MS}
            log(f"    ⏱ perceived {latency:.0f} ms (−{self.ws_rtt:.0f} ms RTT = "
                f"{self.turn['rtt_adjusted_latency_ms']:.0f} ms){' · #nudge-btn would show' if latency > NUDGE_MS else ''}")
        if self.bot is None:
            self.bot, self.orphan, self.orphan_at = {"text": self.orphan, "last_caption": self.orphan_at, "chunks": 0,
                                                     "audio_ms": 0.0, "answered": False, "done": False}, "", 0.0
        self.bot |= {"chunks": self.bot["chunks"] + 1, "audio_ms": round(self.bot["audio_ms"] + duration * 1000.0, 1),
                     "last_audio": now, "ready_at": self.played}
        self.waiting, self.speech_end_at = False, None

    async def handle(self, msg: dict[str, Any], now: float) -> None:
        """Route one server frame; every counter here exists to make the two runs comparable."""
        kind = str(msg.get("type") or "")
        if kind == "audio":
            self.bot_audio(msg, now)
        elif kind == "bot_turn_complete" or (kind == "telemetry" and (msg.get("last_event") or {}).get("type") == "turn_complete"):
            if (kind == "telemetry" or str(msg.get("step") or "") in ("", self.step)) and self.bot is not None and not self.bot["done"]:
                self.closed.append(self.bot | {"done": True, "ready_at": self.played})
                self.bot = None
        elif kind == "transcript" and msg.get("role") == "assistant":
            text = msg.get("text")  # prod also echoes her own audio back as a user turn
            if self.bot is not None:
                self.bot |= {"text": merge_caption(self.bot["text"], text), "last_caption": now}
            else:
                self.orphan, self.orphan_at = merge_caption(self.orphan, text), now
        elif kind == "question_state":
            self.step = str(msg.get("step") or "")
            entry = {"step": self.step, "qid": str(msg.get("qid") or ""), "group": str(msg.get("group") or ""),
                     "is_yes_no": bool(msg.get("is_yes_no")), "prompt_chars": msg.get("prompt_chars"),
                     "yes_no_prime_shown": False}
            if entry["is_yes_no"] and not self.primed:  # the prime renders once, on the first yes/no step
                self.primed = entry["yes_no_prime_shown"] = True
            self.questions.append(entry)
        elif kind == "session_started":
            self.call_id = int(msg.get("call_id") or 0) or self.call_id
            self.token = str(msg.get("share_token") or "") or self.token
            log(f"[{self.target}] session started: call_id={self.call_id} engine={msg.get('model') or msg.get('engine') or '?'}")
        elif kind == "clear":  # she was interrupted: whatever SHISHU was saying has stopped
            self.clears += 1
            self.played, self.bot, self.orphan, self.waiting, self.mouth_at = 0.0, None, "", False, 0.0
        elif kind == "ending" and not self.ending_at:
            self.ending_at = now
            log(f"[{self.target}] ending received — letting the outro play out")
        elif kind == "call_completed":
            self.call_id = int(msg.get("call_id") or 0) or self.call_id
            self.completed = self.done = True
            log(f"[{self.target}] call_completed (call_id={self.call_id})")
        elif kind == "report_failed":
            self.failed = self.done = True
            log(f"[{self.target}] report_failed")
        elif kind == "error":
            log(f"    server error: {msg.get('message')}")

    async def respond(self, question: str) -> str:
        """Ask the fastest Flash responder for her next line, with a verbatim sheet fallback."""
        from google import genai
        from google.genai import types
        if self.client is None:
            key = (os.environ.get("GEMINI_API_KEY") or "").strip()
            if not key:
                raise RuntimeError("GEMINI_API_KEY is missing — put it in the repo-root .env")
            self.client = genai.Client(api_key=key)  # built once, then reused for every turn
        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=256,
                                             system_instruction=sheet_for(self.triggers))
        recent = "\n".join(f"{row['who']}: {row['text']}" for row in self.history[-8:]) or "(पहली बात है)"
        prompt = (f"पिछली बातचीत:\n{recent}\n\nशिशु ने अभी यह पूछा (इसी का जवाब दो):\n\"{question}\"\n\n"
                  "मीना की अगली बोली जाने वाली पंक्ति (सिर्फ़ देवनागरी, 1-2 छोटे वाक्य, कोई निर्देश नहीं):")
        last: Exception | None = None
        for name in ([self.model] if self.model else []) + [m for m in MODELS if m != self.model]:
            try:
                reply = clean_reply(getattr(await self.client.aio.models.generate_content(
                    model=name, contents=prompt, config=config), "text", ""))
                if not reply:
                    raise RuntimeError(f"{name} returned an empty candidate")
                if name != self.model:
                    log(f"    responder pinned to {name}")
                self.model = name
                return reply
            except Exception as exc:  # noqa: BLE001 - a retired model must not stall the call
                last = exc
                self.model = "" if self.model == name else self.model
        self.fallbacks += 1
        log(f"    ⚠ responder failed ({type(last).__name__}: {snippet(last, 100)}); sheet fallback")
        return fallback_reply(question, self.triggers)

    async def answer(self, bot: dict[str, Any]) -> None:
        """Answer one SHISHU turn and put Meena's line on the wire as real audio."""
        started = time.perf_counter()
        try:
            caption = " ".join(str(bot["text"]).split())
            truncated = looks_truncated(caption, bot["audio_ms"])
            question = (pack_question(self.cfg.pack, self.step) if truncated and self.target == "v3" else "") or caption
            if question != caption:
                log(f"    ✂ caption cut at {snippet(caption, 24)!r}; answering the pack's wording for {self.step!r}")
            reply = await self.respond(question)
            pcm, had_pause = await asyncio.to_thread(synthesize_utterance, reply, self.tts)
            if not pcm:
                raise RuntimeError("TTS produced no audio")
            if self.ending_at:
                return log("    (dropping her line — the call is closing)")
            latest = self.questions[-1] if self.questions else {}
            self.turn = {"turn": len(self.turns) + 1, "qid": latest.get("qid", ""), "group": latest.get("group", ""),
                         "step": self.step, "bot_text": caption, "bot_audio_ms": bot["audio_ms"],
                         "caption_truncated": truncated, "question_used": question,
                         "caption_repaired_from_pack": question != caption, "meena_reply": reply,
                         "had_pause": had_pause, "think_ms": round((time.perf_counter() - started) * 1000.0, 1),
                         "speech_ms": round(len(pcm) / 2 / RATE * 1000.0, 1), "t_speech_start": None,
                         "t_mouth_closed": None, "t_first_bot_audio": None, "perceived_latency_ms": None,
                         "rtt_adjusted_latency_ms": None, "nudge_btn_would_appear": None,
                         "premature_cutoff": False, "unspoken_ms": None, "nudged": False}
            self.turns.append(self.turn)
            for key, needles in (("nicu", ("काँच के बॉक्स", "चार दिन")), ("compound", ("घुटने",)),
                                 ("fam", ("मामा",)), ("closing", ("यही पूछना था",))):
                self.triggers[key] |= any(needle in reply for needle in needles)
            self.history += [{"who": "शिशु", "text": caption}, {"who": "मीना", "text": reply}]
            self.utter = deque(pcm[i:i + self.fb].ljust(self.fb, b"\x00") for i in range(0, len(pcm), self.fb))
            self.waiting, self.speech_end_at, self.mouth_at = False, None, 0.0
            log(f"[{self.target}] [Turn {len(self.turns):02d}] Q({snippet(caption, 58)}) -> "
                f"A({snippet(reply, 70)}){' [+PAUSE]' if had_pause else ''} · {len(self.utter)} frames "
                f"({self.turn['speech_ms'] / 1000:.1f}s, {self.turn['think_ms']:.0f} ms to think)")
        except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the call
            log(f"  reply failed: {type(exc).__name__}: {exc}")
        finally:
            self.thinking, self.reply_task = False, None

    async def advance(self, now: float) -> None:
        """Close quiet turns, speak her next turn, nudge on dead air, and hang up after the outro."""
        bot = self.bot  # fallback end-of-turn: audio, then 850 ms of silence and drained playback
        if bot and bot["chunks"] and now - bot["last_audio"] >= QUIET and now >= self.played:
            self.closed.append(bot | {"done": True, "ready_at": self.played})
            self.bot = None
        if not self.ending_at and now - self.t0 >= self.cfg.max_seconds:
            self.limit, self.ending_at = f"max-seconds {self.cfg.max_seconds:.0f}", now
            log(f"[{self.target}] stopping: {self.limit} — closing the call cleanly")
        if self.ending_at and not self.hangup_at:
            if now >= self.played + OUTRO or now - self.ending_at >= OUTRO_MAX:
                self.phase, self.hangup_at = "closing", now
                await self.send({"type": "hangup"})
                log(f"[{self.target}] → microphone closed, sending hangup")
            return
        if self.hangup_at:
            return
        # Speak only once SHISHU's whole turn has landed and played out into her ear.
        ready = [b for b in self.closed if not b["answered"]] if not (
            self.thinking or self.utter or self.waiting) else []
        if ready and now >= ready[0]["ready_at"] + GRACE and not (
                ready[0]["text"] and now - ready[0]["last_caption"] < SETTLE
                and now - ready[0]["ready_at"] < SETTLE_MAX):  # a caption that trails the audio
            bot = ready[0]
            if len(self.turns) >= self.cfg.max_turns:
                self.limit, self.ending_at = f"max-turns {self.cfg.max_turns}", now
                return log(f"[{self.target}] stopping: {self.limit} — closing the call cleanly")
            bot["answered"], self.thinking = True, True
            self.reply_task = asyncio.ensure_future(self.answer(bot))
        turn = self.turn
        if (self.waiting and turn and not turn["nudged"] and self.mouth_at
                and now - self.mouth_at >= NUDGE and self.bot is None
                and not (self.closed and not self.closed[-1]["answered"])):  # a real caller taps #nudge-btn
            turn["nudged"], self.nudges = True, self.nudges + 1
            log(f"  ⚠ {NUDGE:.0f}s of silence — tapping #nudge-btn (nudge #{self.nudges})")
            await self.send({"type": "nudge"})

    async def pump(self) -> None:
        """40 ms heartbeat: drain the socket, then advance the turn coordinator."""
        recv = asyncio.ensure_future(self.ws.recv())
        try:
            while not self.done:
                ready, _ = await asyncio.wait({recv}, timeout=0.04)
                now = time.perf_counter()
                if ready:
                    try:
                        raw = recv.result()
                    except Exception as exc:  # noqa: BLE001 - any close ends the call cleanly
                        return log(f"  socket closed ({type(exc).__name__})")
                    recv = asyncio.ensure_future(self.ws.recv())
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(msg, dict):
                        await self.handle(msg, now)
                elif self.hangup_at and now - self.hangup_at > POST_HANGUP:
                    return log(f"  no call_completed within {POST_HANGUP:.0f}s of hangup — moving on")
                elif not self.hangup_at and now - self.t0 > self.cfg.max_seconds + 300:
                    return log("  receive deadline reached")
                else:
                    await self.advance(now)
        finally:
            if not recv.done():
                recv.cancel()

    async def fetch(self) -> dict[str, Any]:
        """Pull the server's own call record and the finished PDF (a POST builds one that is missing)."""
        art = {"detail": {}, "pdf_bytes": 0, "pdf_text": "", "detail_saved": False, "pdf_saved": False}
        if not self.call_id:
            return log("  no call_id — cannot download the call detail or the PDF") or art
        try:
            resp = await self.http.get(f"/api/calls/{self.call_id}")
            if resp.status_code == 200:
                art["detail"] = resp.json() or {}
                path = self.out / f"{self.target}_call_detail.json"
                path.write_text(json.dumps(art["detail"], ensure_ascii=False, indent=2), encoding="utf-8")
                art["detail_saved"] = True
                log(f"[{self.target}] saved {path.name} ({path.stat().st_size} bytes)")
            else:
                log(f"  call detail returned {resp.status_code}: {resp.text[:160]}")
        except Exception as exc:  # noqa: BLE001
            log(f"  call detail download failed: {type(exc).__name__}: {exc}")
        for method, wait in (("GET", 0.0), ("POST", 0.0), ("POST", 5.0)):
            await asyncio.sleep(wait)
            try:
                resp = await self.http.request(method, f"/api/calls/{self.call_id}/{PDF[self.target]}", timeout=180.0)
            except Exception as exc:  # noqa: BLE001
                log(f"  PDF {method} failed: {type(exc).__name__}: {exc}")
                continue
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                (self.out / f"{self.target}.pdf").write_bytes(resp.content)
                log(f"[{self.target}] saved {self.target}.pdf ({len(resp.content)} bytes)")
                return art | {"pdf_bytes": len(resp.content), "pdf_text": pdf_text(resp.content), "pdf_saved": True}
            log(f"  PDF {method} -> {resp.status_code}")
        return art

    def build_log(self, art: dict[str, Any]) -> dict[str, Any]:
        """The one JSON a report reads: baselines, latency, UI defects, and the per-turn ledger."""
        latencies = [t["perceived_latency_ms"] for t in self.turns if isinstance(t["perceived_latency_ms"], (int, float))]
        adjusted = [t["rtt_adjusted_latency_ms"] for t in self.turns if isinstance(t["rtt_adjusted_latency_ms"], (int, float))]
        asks = [q["qid"] for q in self.questions if q["qid"]]
        repeats = {qid: n for qid, n in Counter(asks).items() if n > 1}
        yes_no = [q for q in self.questions if q["is_yes_no"]]
        pdf_flat = "".join(art["pdf_text"].split())
        ui = {"caption_truncated_turns": sum(1 for t in self.turns if t["caption_truncated"]),
              "captions_repaired_from_pack": sum(1 for t in self.turns if t["caption_repaired_from_pack"]),
              "yes_no_turns": len(yes_no), "yes_no_prime_shown_first_turn": bool(yes_no and yes_no[0]["yes_no_prime_shown"]),
              "premature_cutoffs": self.cutoffs, "clear_events": self.clears, "watchdog_nudges": self.nudges,
              "nudge_btn_triggers_gt_3s": sum(1 for t in self.turns if t["nudge_btn_would_appear"]),
              "repeats": {"questions": len(asks), "distinct": len(set(asks)), "repeated": repeats}}
        if self.target == "v3":
            ui |= {"pack_ui": self.ui, "questions": self.questions}
        return {
            "target": self.target, "base_url": self.base, "ws_path": WS[self.target], "pack": self.cfg.pack,
            "call_id": self.call_id, "share_token": self.token, "started_at": self.started,
            "duration_s": round(time.perf_counter() - self.t0, 1), "turns": len(self.turns),
            "http_rtt_ms": self.http_rtt, "ws_rtt_ms": self.ws_rtt, "ui": ui,
            "latency_ms": {"avg": stat(latencies, np.mean), "median": stat(latencies, np.median),
                           "p95": stat(latencies, lambda v: np.percentile(v, 95)), "clean_turns": len(latencies),
                           "min": stat(latencies, min), "max": stat(latencies, max),
                           "rtt_adjusted_avg": stat(adjusted, np.mean)},
            "responder": {"pinned_model": self.model or "(none — sheet fallback)", "sheet_fallbacks": self.fallbacks,
                          "avg_think_ms": stat([t["think_ms"] for t in self.turns], np.mean)},
            "trigger_state": self.triggers,
            "limits": {"max_turns": self.cfg.max_turns, "max_seconds": self.cfg.max_seconds, "reached": self.limit},
            "lifecycle": {"ending_seen": bool(self.ending_at), "call_completed": self.completed, "report_failed": self.failed},
            "server": server_summary(art["detail"]),
            "pdf": {"saved": art["pdf_saved"], "bytes": art["pdf_bytes"], "file": f"{self.target}.pdf",
                    "text_chars": len(art["pdf_text"]), "redaction": {"checked_term": LEAK, "leaked": LEAK in pdf_flat}},
            "call_detail_saved": art["detail_saved"], "turn_log": self.turns}

    async def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(lambda: [synthesize_utterance(line, self.tts) for line in LINES.values()])
        await self.connect()
        self.phase, self.started, self.t0 = "live", time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.perf_counter()
        log(f"[{self.target}] live · max {self.cfg.max_turns} turns / {self.cfg.max_seconds:.0f}s")
        mic = asyncio.ensure_future(self.mic_loop())
        try:
            await self.pump()
        finally:
            self.phase = "done"
            pending = [task for task in (mic, self.reply_task) if task is not None and not task.done()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self.ws.close()
        record = self.build_log(await self.fetch())
        path = self.out / f"{self.target}_ux_log.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        lat, ui, server = record["latency_ms"], record["ui"], record["server"]
        log(f"turns {record['turns']} · latency median {lat['median']} ms · p95 {lat['p95']} ms · "
            f"RTT-adjusted {lat['rtt_adjusted_avg']} ms (http {record['http_rtt_ms']} / ws {record['ws_rtt_ms']})")
        log(f"premature cutoffs {ui['premature_cutoffs']} · #nudge-btn>3s {ui['nudge_btn_triggers_gt_3s']} · "
            f"nudges {ui['watchdog_nudges']} · truncated captions {ui['caption_truncated_turns']} · PDF "
            f"redaction leaked {record['pdf']['redaction']['leaked']}")
        if server:
            log(f"server: answers {server['answers']} · concerns {len(server['concerns'])} · close "
                f"{server.get('close_reason') or '(still asking)'} · log {path}")
        await self.http.aclose()
        return 0


# ---------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ux_persona_bench",
        description="Drive Meena Devi, a spoken-Hindi mother, through one live SHISHU call and record every "
                    "millisecond of perceived turn latency.")
    parser.add_argument("--target", choices=("prod", "v3"), required=True, help="deployment to measure")
    parser.add_argument("--base-url", default=None, help="http(s) base URL (default: the target's deployment)")
    parser.add_argument("--pack", default="shishu-hi", help="intake pack id (default: shishu-hi)")
    parser.add_argument("--out-dir", default="reports/ux-bench", help="artifact directory (default: reports/ux-bench)")
    parser.add_argument("--max-turns", type=int, default=95, help="stop after this many spoken turns")
    parser.add_argument("--max-seconds", type=float, default=900.0, help="stop after this much live call")
    cfg = parser.parse_args(argv)
    cfg.base_url = (cfg.base_url or BASE[cfg.target]).rstrip("/")
    for parent in Path(__file__).resolve().parents:
        if (parent / ".env").is_file():
            load_dotenv(parent / ".env")
            print(f"[env ] {parent / '.env'}", flush=True)
            break
    missing = [k for k in ("GEMINI_API_KEY", "ADMIN_PASSCODE") if not (os.environ.get(k) or "").strip()]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} missing — put it in the repo-root .env")
    try:
        return asyncio.run(Bench(cfg).run())
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
