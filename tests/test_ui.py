"""Node VM checks for the V3 call page: one-time yes/no prime, quick replies, bug dialog."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "intake" / "static"
NODE = shutil.which("node")

HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.env.HTML_PATH, "utf8");
const source = fs.readFileSync(process.env.JS_PATH, "utf8");
const nodes = {};
const uiNodes = [];

function node(id) {
  const attrs = {}, listeners = {}, set = new Set(), kids = [];
  const el = {
    id, hidden: false, disabled: false, checked: false, value: "", src: "", href: "", textContent: "", className: "",
    dataset: {}, style: {}, files: [], children: kids, open: false,
    classList: {
      add: (c) => set.add(c), remove: (c) => set.delete(c), contains: (c) => set.has(c),
      toggle: (c, on) => { const v = on === undefined ? !set.has(c) : !!on; if (v) set.add(c); else set.delete(c); return v; }
    },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return k in attrs ? attrs[k] : null; },
    removeAttribute(k) { delete attrs[k]; },
    addEventListener(t, fn) { (listeners[t] = listeners[t] || []).push(fn); },
    fire(t, ev) { (listeners[t] || []).forEach((fn) => fn(ev || { target: el })); },
    appendChild(c) { kids.push(c); return c; },
    replaceChildren() { kids.length = 0; },
    querySelectorAll(sel) { return sel === ".chip" ? kids.filter((k) => k.className === "chip") : []; },
    querySelector() { return null; }, remove() {}, focus() {},
    click() { el.fire("click"); },
    showModal() { el.open = true; }, close() { el.open = false; }
  };
  return el;
}

/* Build the DOM from the served page: ids, hidden flags, and data-ui hooks on real elements. */
function parseDom() {
  for (const m of html.matchAll(/<([a-z]+)\s([^>]*?)\/?>/g)) {
    const attrs = m[2] || "";
    const id = /id="([^"]+)"/.exec(attrs);
    const ui = /data-ui="([^"]+)"/.exec(attrs);
    if (!id && !ui) continue;
    const el = node(id ? id[1] : "ui:" + ui[1]);
    el.hidden = /\shidden(\s|$)/.test(attrs.split("data-ui")[0].trim() + " ");
    if (ui) { el._ui = ui[1]; uiNodes.push(el); }
    if (id) nodes[id[1]] = el;
  }
}

function resetDom() {
  Object.keys(nodes).forEach((k) => delete nodes[k]);
  uiNodes.length = 0;
  parseDom();
}

resetDom();

const sent = [], pdfCalls = [], intervals = [], timeouts = [], worklets = [], wsUrls = [], opens = [];
let timerId = 0;
let bugBody = null, ctxRef = null, current = null, frames = [];
let PACK_JSON = {}, NOW = 1_700_000_000_000;
let pdfHeaders = "", pdfNoHeaders = false, canShare = true;
const fileNames = [], anchors = [];

/* One tab's sessionStorage. It outlives a `Cmd+R`, exactly like the real thing. */
const storageMap = new Map();
const storage = {
  getItem: (k) => (storageMap.has(String(k)) ? storageMap.get(String(k)) : null),
  setItem: (k, v) => { storageMap.set(String(k), String(v)); },
  removeItem: (k) => { storageMap.delete(String(k)); },
  clear: () => { storageMap.clear(); }
};

function Ctx() {
  this.currentTime = 0; this.sampleRate = 48000; this.destination = {}; this.sources = [];
  ctxRef = this;
}
Ctx.prototype.resume = () => Promise.resolve();
Ctx.prototype.close = () => Promise.resolve();
Ctx.prototype.createBuffer = (ch, len, rate) => ({ duration: len / rate, getChannelData: () => new Float32Array(len) });
Ctx.prototype.createBufferSource = function () {
  const src = { connect() {}, start() {}, stop() {}, disconnect() {}, onended: null };
  this.sources.push(src);
  return src;
};
Ctx.prototype.createMediaStreamSource = () => ({ connect() {} });
Object.defineProperty(Ctx.prototype, "audioWorklet", { get: () => ({ addModule: () => Promise.resolve() }) });

function MockWS(url) {
  this.url = url; this.readyState = 1; this.bufferedAmount = 0;
  wsUrls.push(url); current = this;
  opens.push(() => this.onopen && this.onopen());
}
MockWS.OPEN = 1;
MockWS.prototype.send = (raw) => sent.push(JSON.parse(raw));
MockWS.prototype.close = function () { this.readyState = 3; };

function MockWorkletNode() {
  this.port = { onmessage: null };
  this.connect = () => {};
  this.disconnect = () => {};
  worklets.push(this);
}

/* Deterministic clocks: `advance(ms)` is the only thing that moves time forward. */
const fakeSetTimeout = (fn, ms) => {
  const t = { id: ++timerId, fn, due: NOW + Math.max(0, Number(ms) || 0) };
  timeouts.push(t);
  return t.id;
};
const fakeClearTimeout = (id) => { const at = timeouts.findIndex((t) => t.id === id); if (at >= 0) timeouts.splice(at, 1); };
const fakeSetInterval = (fn, ms) => { intervals.push({ id: ++timerId, fn, ms }); return timerId; };
const fakeClearInterval = (id) => { const at = intervals.findIndex((t) => t.id === id); if (at >= 0) intervals.splice(at, 1); };

function makeSandbox() {
  const box = {
    console,
    setTimeout: fakeSetTimeout, clearTimeout: fakeClearTimeout,
    setInterval: fakeSetInterval, clearInterval: fakeClearInterval,
    sessionStorage: storage,
    AudioWorkletNode: MockWorkletNode,
    requestAnimationFrame: (fn) => frames.push(fn), cancelAnimationFrame() {},
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
    URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
    Blob: function (parts) { this.parts = parts; },
    File: function (parts, name, cfg) { this.parts = parts; this.name = name; this.type = cfg && cfg.type; fileNames.push(name); },
    Image: function () { const self = this; setTimeout(() => self.onerror && self.onerror(), 0); },
    URLSearchParams, Uint8Array, Int16Array, Float32Array, DataView, ArrayBuffer, JSON, Promise,
    encodeURIComponent, decodeURIComponent, Object, Number, String, Boolean, Math, RegExp, Error,
    Date: Object.assign(class extends Date { constructor(...args) { super(...(args.length ? args : [NOW])); } }, { now: () => NOW }),
    document: {
      getElementById: (id) => nodes[id] || (nodes[id] = node(id)),
      createElement: (tag) => { const made = node(tag); if (tag === "a") anchors.push(made); return made; },
      querySelectorAll: (sel) => (sel === "[data-ui]" ? uiNodes : []),
      body: { appendChild() {} }
    },
    location: { protocol: "http:", host: "t", search: "", href: "http://t/", pathname: "/" },
    navigator: {
      mediaDevices: { getUserMedia: () => Promise.resolve({ getTracks: () => [{ stop() {} }] }) },
      canShare: () => canShare, share: () => Promise.resolve(),
      clipboard: { writeText: () => Promise.resolve() }
    },
    WebSocket: MockWS,
    window: { AudioContext: Ctx, sessionStorage: storage }
  };
  box.fetch = (url, init) => {
    if (url.startsWith("/api/pack")) {
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(PACK_JSON) });
    }
    if (url === "/api/bugs") {
      bugBody = JSON.parse((init && init.body) || "{}");
      return Promise.resolve({ ok: true, status: 200 });
    }
    if (url.includes("patient.pdf")) {
      pdfCalls.push(init && init.method);
      const headers = pdfNoHeaders ? undefined : { get: (k) => (k.toLowerCase() === "content-disposition" ? pdfHeaders : null) };
      return Promise.resolve({ ok: true, status: 200, headers, blob: () => Promise.resolve(new Blob([])) });
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) });
  };
  return box;
}

let sandbox = makeSandbox();

const api = {
  nodes, sent, pdfCalls,
  async settle() { for (let i = 0; i < 40; i += 1) await Promise.resolve(); },
  flushRaf() { const q = frames.splice(0); q.forEach((fn) => fn()); },
  ctx: () => ctxRef,
  ws: () => current,
  wsUrls: () => wsUrls.slice(),
  openPending() { opens.splice(0).forEach((fn) => fn()); },
  storage() { return Object.fromEntries(storageMap); },
  push(msg) { current.onmessage({ data: JSON.stringify(msg) }); },
  pcm(bytes) { return Buffer.alloc(bytes, 8).toString("base64"); },
  visible(id) { return !nodes[id].hidden && nodes[id].classList.contains("is-visible"); },
  /* Time travel: 250 ms steps, so due timers (reconnect backoff, the 5 s unlock) and the tick interleave. */
  advance(ms) {
    const total = Math.max(0, Number(ms) || 0);
    const steps = Math.max(1, Math.ceil(total / 250));
    for (let i = 0; i < steps; i += 1) {
      NOW += total / steps;
      for (let guard = 0; guard < 500; guard += 1) {
        let at = -1;
        timeouts.forEach((t, k) => { if (t.due <= NOW && (at < 0 || t.due < timeouts[at].due)) at = k; });
        if (at < 0) break;
        const t = timeouts.splice(at, 1)[0];
        t.fn();
      }
      intervals.filter((t) => t.ms === 250).forEach((t) => t.fn());
    }
  },
  now() { return NOW; },
  /* A mic frame straight from the capture worklet: amplitude 0 is silence, anything above 0.015 is voice. */
  mic(amplitude) {
    const frame = new Float32Array(1024).fill(amplitude);
    worklets.forEach((w) => w.port.onmessage && w.port.onmessage({ data: frame }));
  },
  micReady() { return worklets.length > 0; },
  boot(pack) {
    PACK_JSON = pack;
    resetDom();
    intervals.length = 0; timeouts.length = 0; frames.length = 0; opens.length = 0; worklets.length = 0;
    current = null; ctxRef = null;
    sandbox = makeSandbox();
    vm.createContext(sandbox);
    vm.runInContext(source, sandbox, { filename: "call.js" });
  },
  /* A browser reload in the same tab: new DOM, new page globals, the SAME sessionStorage. */
  reboot(pack) { api.boot(pack); },
  async start() {
    nodes["consent-check"].checked = true;
    nodes["call-btn"].fire("click");
    await api.settle();
    opens.splice(0).forEach((fn) => fn());
    await api.settle();
  },
  async reply(value) {
    const chip = nodes["bug-chips"].children.find((c) => c.dataset.value === value);
    nodes["bug-chips"].fire("click", { target: { closest: () => chip } });
  },
  bug() { return bugBody; },
  pdfName() { return fileNames.slice(); },
  downloads() { return anchors.map((a) => a.download); },
  setPdf(disposition, noHeaders, shareable) { pdfHeaders = disposition || ""; pdfNoHeaders = !!noHeaders; canShare = shareable !== false; },
  /* `playing()` is a `let` binding inside the page's own context, so it is read there, never from this scope. */
  playing() { return vm.runInContext("playing()", sandbox); }
};
globalThis.api = api;
"""

PACK = {
    "ui": {
        "yes_no_prime": {"hi": "प्राइम हिंदी", "en": "Prime English"},
        "bug_categories": {"chips": [{"value": "Slow response", "hi": "धीमा", "en": "Slow"}]},
        "call": {
            "listening": {"hi": "सुन रही हूँ…", "en": "Listening"},
            "thinking": {"hi": "सोच रही हूँ…", "en": "Thinking"},
            "reconnecting": {"hi": "लाइन जुड़ रही है…", "en": "Reconnecting"},
        },
    },
    "assets_url": {"logo": "/static/pack-logo.png", "consent_pdf": "/static/consent.pdf"},
    "role": "admin",
}


@unittest.skipIf(NODE is None, "node is not installed")
class CallPageTests(unittest.TestCase):
    def run_js(self, body: str) -> dict:
        script = HARNESS + "\n(async () => {\n" + body + "\n})()" \
            ".then((out) => process.stdout.write(JSON.stringify(out)))" \
            ".catch((err) => { console.error(err); process.exit(1); });"
        done = subprocess.run(
            [NODE, "-e", script], capture_output=True, text=True, check=False,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "JS_PATH": str(STATIC / "call.js"),
                 "HTML_PATH": str(STATIC / "call.html"), "PACK": json.dumps(PACK)},
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def test_first_yes_no_primes_once_then_pills_and_bug_dialog(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
const beforeAny = api.nodes["yes-no-prime"].hidden;
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();

api.push({ type: "question_state", step: "fever_now", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
api.ctx().currentTime = 0.4;
api.flushRaf();
const ynShown = api.visible("no-btn") && api.visible("yes-btn");
const primeShown = !api.nodes["yes-no-prime"].hidden;
const primeText = api.nodes["yes-no-prime"].children.map((c) => c.textContent).join("|");
api.nodes["yes-btn"].fire("click");
await api.settle();
const first = api.sent.filter((m) => m.type === "quick_reply");

api.push({ type: "question_state", step: "fever_days", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
api.ctx().currentTime = 1.2;
api.flushRaf();
const primeAgain = !api.nodes["yes-no-prime"].hidden;
const primeGone = api.nodes["yes-no-prime"].hidden;

api.push({ type: "question_state", step: "fever_grade", is_yes_no: false,
  quick_replies: [{ value: "first", hi: "पहला", en: "First" }, { value: "second", hi: "दूसरा", en: "Second" }] });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
api.ctx().currentTime = 3.4;
api.ctx().sources.forEach((s) => s.onended && s.onended());
api.push({ type: "bot_turn_complete", step: "fever_grade" });
api.flushRaf();
const pills = api.nodes["choice-pills"].children.map((c) => c.dataset.value);
const pillsShown = !api.nodes["choice-pills"].hidden;
api.nodes["choice-pills"].children[1].fire("click");
await api.settle();
const pillReply = api.sent.filter((m) => m.type === "quick_reply").pop();

api.nodes["bug-btn"].fire("click");
const dialogOpen = !!api.nodes["bug-dialog"].open;
await api.reply("Slow response");
api.nodes["bug-note"].value = "  ऑपरेशन के बाद अटक गई  ";
api.nodes["bug-save"].fire("click");
await api.settle();

return {
  beforeAny, ynShown, primeShown, primeText, first, primeAgain, primeGone,
  pills, pillsShown, pillReply, dialogOpen, confirmed: !api.nodes["bug-confirm"].hidden,
  bugBody: api.bug(), chips: api.nodes["bug-chips"].children.length,
  adminHidden: api.nodes["admin-link"].hidden, logo: api.nodes["brand-logo"].src,
  consent: api.nodes["consent-link"].href
};
""")

        self.assertTrue(out["beforeAny"], "the prime stays hidden until the first yes/no question")
        self.assertTrue(out["ynShown"], "the first yes/no question reveals the No/Yes pair")
        self.assertTrue(out["primeShown"], "the first yes/no question shows the open-speech prime")
        self.assertIn("प्राइम हिंदी", out["primeText"])
        self.assertEqual(out["first"], [{"type": "quick_reply", "value": "concern", "step": "fever_now", "epoch": 1}])
        self.assertFalse(out["primeAgain"], "the second yes/no question must not re-prime")
        self.assertTrue(out["primeGone"], "the prime hides once answered and the next question starts")
        self.assertEqual(out["pills"], ["first", "second"])
        self.assertTrue(out["pillsShown"])
        self.assertEqual(out["pillReply"], {"type": "quick_reply", "value": "second", "step": "fever_grade", "epoch": 3})
        self.assertTrue(out["dialogOpen"], "the native dialog opens through showModal")
        self.assertTrue(out["confirmed"], "a saved bug confirms to the caller")
        self.assertEqual(out["bugBody"]["category"], "Slow response")
        self.assertEqual(out["bugBody"]["note"], "ऑपरेशन के बाद अटक गई")
        self.assertEqual(out["bugBody"]["share_token"], "tok")
        self.assertEqual(out["chips"], 1)
        self.assertFalse(out["adminHidden"], "admin role reveals the admin link")
        self.assertEqual(out["logo"], "/static/pack-logo.png")
        self.assertEqual(out["consent"], "/static/consent.pdf")

    def test_the_nudge_button_surfaces_after_three_seconds_of_silence_only(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
const micReady = api.micReady();

/* Silence with no caller speech never reveals the nudge. */
api.advance(4000);
const quietIdle = api.nodes["nudge-btn"].hidden;

/* The caller speaks, then stops: the 3 s clock starts at the silence, not the speech. */
api.mic(0.2);
const duringSpeech = api.nodes["nudge-btn"].hidden;
api.advance(4000);
const stillSpeaking = api.nodes["nudge-btn"].hidden;
api.mic(0);
api.advance(2000);
const twoSeconds = api.nodes["nudge-btn"].hidden;
api.advance(1500);
const afterThree = !api.nodes["nudge-btn"].hidden;

/* Talking again hides it and re-arms the clock. */
api.mic(0.2);
const talksAgain = api.nodes["nudge-btn"].hidden;
api.mic(0);
api.advance(3500);
const revealedAgain = !api.nodes["nudge-btn"].hidden;

/* A tap on the nudge clears it and sends one nudge. */
api.nodes["nudge-btn"].fire("click");
await api.settle();
const afterTap = api.nodes["nudge-btn"].hidden;
const nudges = api.sent.filter((m) => m.type === "nudge").length;

/* A bot turn arriving hides it; a quick-reply tap starts a fresh wait. */
api.push({ type: "question_state", step: "fever_now", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
api.ctx().currentTime = 0.4;
api.flushRaf();
api.nodes["yes-btn"].fire("click");
await api.settle();
/* The answer opens a 5 s prompting window: no premature 3 s nudge while she starts talking. */
api.advance(4000);
const nudgeDuringPrompting = api.nodes["nudge-btn"].hidden;
api.advance(1500);
const afterReply = !api.nodes["nudge-btn"].hidden;
api.push({ type: "question_state", step: "fever_days", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
await api.settle();
const afterBotAudio = api.nodes["nudge-btn"].hidden;
api.push({ type: "clear" });
api.advance(4000);
const afterClear = api.nodes["nudge-btn"].hidden;

return { micReady, quietIdle, duringSpeech, stillSpeaking, twoSeconds, afterThree, talksAgain, revealedAgain,
         afterTap, nudges, nudgeDuringPrompting, afterReply, afterBotAudio, afterClear };
""")

        self.assertTrue(out["micReady"], "the capture worklet is wired up by startCall")
        self.assertTrue(out["quietIdle"], "silence with no caller speech never nags")
        self.assertTrue(out["duringSpeech"], "the nudge never appears while the caller is mid-sentence")
        self.assertTrue(out["stillSpeaking"], "a long utterance keeps it hidden")
        self.assertTrue(out["twoSeconds"], "2 s of silence is too early")
        self.assertTrue(out["afterThree"], "past 3 s it must surface")
        self.assertTrue(out["talksAgain"], "speaking again hides it")
        self.assertTrue(out["revealedAgain"], "and the clock restarts on silence")
        self.assertTrue(out["afterTap"], "the tap hides it")
        self.assertEqual(out["nudges"], 1, "the tap sends exactly one nudge")
        self.assertTrue(out["nudgeDuringPrompting"], "no nudge in the 5 s prompting window after a tap")
        self.assertTrue(out["afterReply"], "a quick-reply tap arms the wait even with no mic frames")
        self.assertTrue(out["afterBotAudio"], "bot audio clears the wait")
        self.assertTrue(out["afterClear"], "a clear clears the wait")

    def test_the_mic_is_muted_while_shishu_speaks_so_no_echo_is_sent(self):
        # "She can tap over SHISHU but not talk over her": while any of her audio is still queued, the capture
        # worklet keeps running (the meter moves) but nothing is sent and the 3 s nudge timer never arms.
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
api.push({ type: "question_state", step: "fever_now", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
await api.settle();
const speaking = api.playing();

api.mic(0.2);
await api.settle();
api.advance(4000);
const echoed = api.sent.filter((m) => m.type === "audio").length;
const nudgeWhileSpeaking = api.nodes["nudge-btn"].hidden;

/* Her line ends: the caller is heard again at once, and the silence clock starts from that frame. */
api.ctx().currentTime = 0.4;
api.ctx().sources.forEach((s) => s.onended && s.onended());
await api.settle();
const quietAgain = api.playing();
api.mic(0.2);
await api.settle();
const afterPlayback = api.sent.filter((m) => m.type === "audio").length;
api.mic(0);
api.advance(3500);
const nudgeAfterPlayback = !api.nodes["nudge-btn"].hidden;

return { speaking, echoed, nudgeWhileSpeaking, quietAgain, afterPlayback, nudgeAfterPlayback };
""")

        self.assertTrue(out["speaking"], "her audio really was queued")
        self.assertEqual(out["echoed"], 0, "not one echo frame reaches the socket while she speaks")
        self.assertTrue(out["nudgeWhileSpeaking"], "and a muted mic never arms the 3 s nudge")
        self.assertFalse(out["quietAgain"], "her queue is empty once the last frame ends")
        self.assertEqual(out["afterPlayback"], 1, "the moment her audio ends the caller is heard again")
        self.assertTrue(out["nudgeAfterPlayback"], "and the silence clock runs off her real listening window")

    def test_a_tap_over_shishu_stops_her_at_once_and_ignores_the_aborted_turns_tail(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
api.push({ type: "question_state", step: "fever_now", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
api.ctx().currentTime = 0.4;
api.flushRaf();
const wasPlaying = api.playing();

api.nodes["yes-btn"].fire("click");
await api.settle();
const stillPlaying = api.playing();
const tapped = api.sent.filter((m) => m.type === "quick_reply").length;

/* Question A's in-flight tail lands before Question B's question_state: it must stay silent. */
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
await api.settle();
const tailAudible = api.playing();

api.push({ type: "question_state", step: "fever_days", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
await api.settle();
const secondAudible = api.playing();

return { wasPlaying, stillPlaying, tapped, tailAudible, secondAudible };
""")

        self.assertTrue(out["wasPlaying"], "the first question really was playing")
        self.assertFalse(out["stillPlaying"], "the tap cuts her off in the same frame, before any server reply")
        self.assertEqual(out["tapped"], 1, "and sends exactly one quick reply")
        self.assertFalse(out["tailAudible"], "Question A's aborted tail never restarts playback over the new step")
        self.assertTrue(out["secondAudible"], "Question B plays normally once its own state arrives")

    def test_the_pdf_adopts_the_served_filename_and_falls_back_without_headers(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();

api.setPdf('attachment; filename="SHISHU-2026-11-12-093000-c4.pdf"', false, false);
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();
const adopted = api.pdfName().pop();
api.nodes["share-pdf"].fire("click");
await api.settle();
const adoptedDownload = api.downloads().pop();

api.setPdf("", true, false);
api.nodes["retry-pdf"].fire("click");
await api.settle();
const fallback = api.pdfName().pop();
api.nodes["share-pdf"].fire("click");
await api.settle();
const fallbackDownload = api.downloads().pop();

return { adopted, adoptedDownload, fallback, fallbackDownload };
""")

        self.assertEqual(out["adopted"], "SHISHU-2026-11-12-093000-c4.pdf")
        self.assertEqual(out["adoptedDownload"], "SHISHU-2026-11-12-093000-c4.pdf")
        self.assertEqual(out["fallback"], "SHISHU.pdf", "no Content-Disposition still yields a name")
        self.assertEqual(out["fallbackDownload"], "SHISHU.pdf")

    def test_ws_drop_mid_call_auto_reconnects_with_resume_token_and_never_shows_consent(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 13, share_token: "tok13", out_sample_rate: 24000 });
await api.settle();
api.push({ type: "question_state", step: "intro_married", epoch: 1, is_yes_no: false, quick_replies: [] });
await api.settle();
const urlsBefore = api.wsUrls();

/* The carrier drops the socket in the middle of the call. */
api.ws().onclose();
await api.settle();
const consentHidden = api.nodes["consent-row"].hidden;
const statusText = api.nodes["status"].children.map((c) => c.textContent).join("|");
const urlsAfterClose = api.wsUrls();

api.advance(700);
await api.settle();
const urlsAfterBackoff = api.wsUrls();

api.openPending();
await api.settle();
api.push({ type: "session_started", call_id: 13, share_token: "tok13", out_sample_rate: 24000 });
api.push({ type: "question_state", step: "intro_married", epoch: 1, is_yes_no: false, quick_replies: [] });
await api.settle();

return {
  urlsBefore, consentHidden, statusText, urlsAfterClose, urlsAfterBackoff,
  consentAfterResume: api.nodes["consent-row"].hidden,
  liveAfterResume: api.nodes["avatar-wrap"].classList.contains("live"),
  errorHidden: api.nodes["error"].hidden
};
""")

        self.assertEqual(len(out["urlsBefore"]), 1, "the first socket is opened without a resume token")
        self.assertNotIn("resume=", out["urlsBefore"][0], "a fresh call never resumes anything")
        self.assertTrue(out["consentHidden"], "a mid-call drop must never flash the consent row back")
        self.assertIn("लाइन जुड़ रही है…", out["statusText"])
        self.assertIn("Reconnecting", out["statusText"])
        self.assertEqual(len(out["urlsAfterClose"]), 1, "no socket is opened before the backoff elapses")
        self.assertEqual(len(out["urlsAfterBackoff"]), 2, "the 600 ms backoff opens exactly one new socket")
        self.assertIn("resume=tok13", out["urlsAfterBackoff"][1], "the reconnect resumes the same call")
        self.assertTrue(out["consentAfterResume"], "the resumed call is live, not a fresh consent gate")
        self.assertTrue(out["liveAfterResume"], "the avatar stays in its live ring across the drop")
        self.assertTrue(out["errorHidden"], "a recovered drop never surfaces the cut error")

    def test_consent_persists_in_session_storage_across_reload_and_clears_on_logout(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
const initialUnchecked = api.nodes["consent-check"].checked !== true;
const initialDisabled = api.nodes["call-btn"].disabled;

api.nodes["consent-check"].checked = true;
api.nodes["consent-check"].fire("change");
const saved = api.storage()["shishu_consent"] || "";
const enabled = api.nodes["call-btn"].disabled === false;

await api.start();
api.push({ type: "session_started", call_id: 21, share_token: "tok_old", out_sample_rate: 24000 });
await api.settle();

/* Cmd+R in the same tab: same sessionStorage, brand new page globals. */
await api.reboot(JSON.parse(process.env.PACK));
await api.settle();
const afterReloadChecked = api.nodes["consent-check"].checked === true;
const afterReloadEnabled = api.nodes["call-btn"].disabled === false;

await api.start();
const freshUrl = api.wsUrls().pop();
const resumedAfterReload = freshUrl.indexOf("resume=") >= 0;

api.nodes["logout-btn"].fire("click");
await api.settle();
const cleared = api.storage();

return { initialUnchecked, initialDisabled, saved, enabled, afterReloadChecked, afterReloadEnabled, freshUrl, resumedAfterReload, cleared };
""")

        self.assertTrue(out["initialUnchecked"], "consent starts unchecked in a fresh tab")
        self.assertTrue(out["initialDisabled"], "and the call button is gated on it")
        self.assertEqual(out["saved"], "1", "checking consent is remembered for this tab")
        self.assertTrue(out["enabled"], "and it unlocks the call button")
        self.assertTrue(out["afterReloadChecked"], "a reload keeps the consent already given")
        self.assertTrue(out["afterReloadEnabled"], "so the call button is ready without re-ticking")
        self.assertFalse(out["resumedAfterReload"], "a manual reload starts a fresh call, never a resume")
        self.assertNotIn("shishu_consent", out["cleared"], "logging out forgets the consent")

    def test_client_gates_mic_in_post_tap_prompting_gap_and_hides_nudge_while_thinking(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 31, share_token: "tok31", out_sample_rate: 24000 });
api.push({ type: "question_state", step: "intro_bridge", epoch: 1, is_yes_no: true, quick_replies: [] });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000, epoch: 1 });
api.ctx().currentTime = 0.4;
api.flushRaf();
const yesShown = api.visible("yes-btn");
api.nodes["yes-btn"].fire("click");
await api.settle();

/* Question B's state lands before its first audio frame: the client sits in `prompting`. */
api.push({ type: "question_state", step: "birth_mode_detail", epoch: 2, is_yes_no: true, quick_replies: [] });
await api.settle();
const beforeGap = api.sent.filter((m) => m.type === "audio").length;
api.mic(0.2);
await api.settle();
const duringGap = api.sent.filter((m) => m.type === "audio").length;

/* No audio ever arrives for B: the 5 s safety unlock re-opens the mic. */
api.advance(5100);
await api.settle();
api.mic(0.2);
await api.settle();
const afterUnlock = api.sent.filter((m) => m.type === "audio").length;

api.push({ type: "thinking", epoch: 2, step: "birth_mode_detail" });
await api.settle();
const thinkingText = api.nodes["status"].children.map((c) => c.textContent).join("|");
api.advance(4000);
await api.settle();
const nudgeHidden = api.nodes["nudge-btn"].hidden;
const afterThinking = api.sent.filter((m) => m.type === "audio").length;

return {
  yesShown, gapFrames: duringGap - beforeGap, unlockFrames: afterUnlock - duringGap,
  thinkingText, nudgeHidden, thinkingFrames: afterThinking - afterUnlock
};
""")

        self.assertTrue(out["yesShown"], "Question A really is answerable")
        self.assertEqual(out["gapFrames"], 0, "no mic frame is sent in the post-tap prompting gap")
        self.assertEqual(out["unlockFrames"], 1, "the 5 s safety unlock re-opens the mic when no audio arrives")
        self.assertIn("सोच रही हूँ…", out["thinkingText"])
        self.assertIn("Thinking", out["thinkingText"])
        self.assertTrue(out["nudgeHidden"], "the nudge never surfaces while the server is deciding")
        self.assertEqual(out["thinkingFrames"], 0, "and the mic stays shut while she thinks")


if __name__ == "__main__":
    unittest.main()
