"""Node VM checks for the V3 call page: one-time yes/no prime, quick replies, bug dialog."""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from intake.pack import load_pack

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

/* Build the DOM from the served page: ids, hidden flags, data-ui hooks, and the real child tree — so a test
   can prove an inline SVG still sits inside the button after `applyUi` rewrites the label span. */
const VOID = new Set(["meta", "link", "img", "input", "br", "hr", "source"]);
const htmlIds = new Set();

function parseDom() {
  const stack = [];
  /* Tag tokens and text tokens in one pass, so an element's own copy is captured even when nothing rewrites it. */
  for (const m of html.matchAll(/<(\/?)([a-z]+)\s*([^>]*?)(\/?)>|([^<>]+)/g)) {
    if (m[5] != null) {
      const parent = stack.length ? stack[stack.length - 1] : null;
      const text = m[5].replace(/&amp;/g, "&").replace(/&quot;/g, '"').trim();
      if (parent && text) parent.textContent = (parent.textContent || "") + text;
      continue;
    }
    const closing = m[1], tag = m[2], attrs = m[3] || "", selfClose = m[4];
    if (closing) { if (stack.length) stack.pop(); continue; }
    const id = /id="([^"]+)"/.exec(attrs);
    const ui = /data-ui="([^"]+)"/.exec(attrs);
    const made = node(id ? id[1] : (ui ? "ui:" + ui[1] : "tag:" + tag));
    made.tagName = tag.toUpperCase();
    /* Mirror every served attribute into the mock's attribute map: `getAttribute("data-ui")` is how
       applyUi() finds its hooks, so a missing map would make the whole pack copy a no-op here. */
    for (const a of attrs.matchAll(/([a-zA-Z-]+)="([^"]*)"/g)) made.setAttribute(a[1], a[2]);
    const cls = /class="([^"]*)"/.exec(attrs);
    made.className = cls ? cls[1] : "";
    made.className.split(/\s+/).filter(Boolean).forEach((c) => made.classList.add(c));
    made.hidden = /(^|\s)hidden(\s|$)/.test(attrs.split("data-ui")[0].trim() + " ");
    if (ui) { made._ui = ui[1]; uiNodes.push(made); }
    const parent = stack.length ? stack[stack.length - 1] : null;
    if (parent) parent.appendChild(made);
    if (id) { nodes[id[1]] = made; htmlIds.add(id[1]); }
    if (!VOID.has(tag) && !selfClose) stack.push(made);
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
let pdfHeaders = "", pdfNoHeaders = false, pdfFails = false, pdfGate = false, pdfRelease = null;
let canShare = true, shareCalls = 0, revoked = [];
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
    URL: { createObjectURL: () => "blob:x", revokeObjectURL: (u) => revoked.push(u) },
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
      canShare: () => canShare,
      share: () => { shareCalls += 1; return Promise.resolve(); },
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
      const reply = () => (pdfFails
        ? { ok: false, status: 500, headers, blob: () => Promise.resolve(new Blob([])) }
        : { ok: true, status: 200, headers, blob: () => Promise.resolve(new Blob([])) });
      if (pdfGate) return new Promise((resolve) => { pdfRelease = () => resolve(reply()); });
      return Promise.resolve(reply());
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
  shares() { return shareCalls; },
  revoked() { return revoked.slice(); },
  /* `has` is the only honest "this id exists on the served page" check: a missing id would otherwise
     be conjured by getElementById with `hidden === false` and read as visible. */
  has(id) { return htmlIds.has(id); },
  setPdf(disposition, noHeaders, shareable) { pdfHeaders = disposition || ""; pdfNoHeaders = !!noHeaders; canShare = shareable !== false; },
  failPdf(fails) { pdfFails = !!fails; },
  /* Hold the PDF response open, so "preparing" can be observed mid-flight instead of guessed at. */
  gatePdf() { pdfGate = true; pdfRelease = null; },
  releasePdf() { pdfGate = false; const done = pdfRelease; pdfRelease = null; if (done) done(); },
  /* A missing id is a real defect (the element was never served), so say so — never let getElementById's
     lazy stand-in answer as if it were on the page. */
  need(id) { if (!nodes[id]) throw new Error("#" + id + " is not on the served call page"); return nodes[id]; },
  shown(id) { return !api.need(id).hidden; },
  tree(id) {
    const out = [];
    const textOf = (n) => (n.textContent || "") + (n.children || []).map(textOf).join("");
    const walk = (n) => {
      out.push({ tag: n.tagName || "", cls: n.className || "", aria: n.getAttribute("aria-hidden"),
                 d: n.getAttribute("d"), text: textOf(n), id: n.id || "" });
      (n.children || []).forEach(walk);
    };
    walk(api.need(id));
    return out;
  },
  /* `playing()` is a `let` binding inside the page's own context, so it is read there, never from this scope. */
  playing() { return vm.runInContext("playing()", sandbox); }
};
globalThis.api = api;
"""

ADMIN_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.env.HTML_PATH, "utf8");
const source = fs.readFileSync(process.env.JS_PATH, "utf8");
const nodes = {}, intervals = [];

function node(id) {
  const attrs = {}, listeners = {}, set = new Set(), kids = [];
  const el = {
    id, hidden: false, disabled: false, checked: false, value: "", src: "", href: "", textContent: "", innerHTML: "", className: "",
    dataset: {}, style: {}, children: kids,
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
    querySelectorAll() { return []; }, querySelector() { return null; }, remove() {}, focus() {}, click() { el.fire("click"); }
  };
  return el;
}

/* Ids and hidden flags, straight from the served dashboard page. */
for (const m of html.matchAll(/<([a-z]+)\s([^>]*?)\/?>/g)) {
  const attrs = m[2] || "", id = /id="([^"]+)"/.exec(attrs);
  if (!id) continue;
  const el = node(id[1]);
  el.hidden = /(^|\s)hidden(\s|$)/.test(attrs);
  nodes[id[1]] = el;
}

const PLAN = {
  pack: JSON.parse(process.env.ADMIN_PACK),
  calls: JSON.parse(process.env.ADMIN_CALLS),
  detail: JSON.parse(process.env.ADMIN_DETAIL)
};
const ok = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });

const box = {
  console,
  setInterval: (fn, ms) => { intervals.push({ fn, ms }); return intervals.length; },
  clearInterval() {},
  setTimeout: () => 0, clearTimeout() {},
  document: {
    getElementById: (id) => nodes[id] || (nodes[id] = node(id)),
    createElement: (tag) => node(tag),
    body: { classList: { add() {}, remove() {} } },
    title: ""
  },
  location: { search: "", pathname: "/admin", href: "http://t/admin" },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
  URLSearchParams,
  window: { addEventListener() {}, print() {} },
  fetch: (url) => {
    if (url === "/api/pack") return ok(PLAN.pack);
    if (url === "/api/bugs") return ok([]);
    if (url.indexOf("/api/calls/") === 0) return ok(PLAN.detail);
    if (url === "/api/calls") return ok({ calls: PLAN.calls });
    return ok({});
  }
};

vm.createContext(box);
vm.runInContext(source, box, { filename: "admin.js" });
globalThis.api = {
  nodes,
  async settle() { for (let i = 0; i < 120; i += 1) await Promise.resolve(); }
};
"""

PACK = {
    "ui": {
        "yes_no_prime": {"hi": "प्राइम हिंदी", "en": "Prime English"},
        "bug_categories": {"chips": [{"value": "Slow response", "hi": "धीमा", "en": "Slow"}]},
        "call": {
            "listening": {"hi": "सुन रही हूँ…", "en": "Listening"},
            "thinking": {"hi": "सोच रही हूँ…", "en": "Thinking"},
            "reconnecting": {"hi": "लाइन जुड़ रही है…", "en": "Reconnecting"},
            "preparing": {"hi": "रिपोर्ट तैयार हो रही है…", "en": "Preparing report…"},
            "download_btn": {"hi": "डाउनलोड करें", "en": "Download"},
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

    def run_admin_js(self, body: str, calls=None, detail=None) -> dict:
        script = ADMIN_HARNESS + "\n(async () => {\n" + body + "\n})()" \
            ".then((out) => process.stdout.write(JSON.stringify(out)))" \
            ".catch((err) => { console.error(err); process.exit(1); });"
        done = subprocess.run(
            [NODE, "-e", script], capture_output=True, text=True, check=False,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "JS_PATH": str(STATIC / "admin.js"),
                 "HTML_PATH": str(STATIC / "admin.html"), "ADMIN_PACK": json.dumps({"role": "admin"}),
                 "ADMIN_CALLS": json.dumps(calls or [{"id": 1}]), "ADMIN_DETAIL": json.dumps(detail or {"id": 1})},
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def test_structured_answers_sort_concern_unknown_unanswered_answered_clear(self):
        # A chart reads worst-first: an `unanswered` question is a gap the clinician must close, so it belongs
        # beside `unknown` — above `answered` and `clear`, never below them.
        answers = {
            "q_a": {"state": "clear", "choice_label": "clear"},
            "q_b": {"state": "answered", "choice_label": "answered"},
            "q_c": {"state": "unanswered"},
            "q_d": {"state": "unknown"},
            "q_e": {"state": "concern", "choice_label": "concern"},
            "q_f": {"state": "not_applicable"},
        }
        out = self.run_admin_js(r"""
await api.settle();
const html = api.nodes["answers"].innerHTML;
const states = [];
for (const m of html.matchAll(/<tr[^>]*><td>([^<]*)<\/td><td>([^<]*)<\/td>/g)) states.push(m[2]);
return { states, html };
""", calls=[{"id": 7, "pack_id": "p", "status": "completed"}],
            detail={"id": 7, "pack_id": "p", "status": "completed", "cursor_state": {"answers": answers},
                    "events": [], "transcript": [], "telemetry": {}})

        self.assertEqual(out["states"], ["concern", "unknown", "unanswered", "answered", "clear"])
        self.assertNotIn("not_applicable", out["html"], "not_applicable questions stay out of the chart")

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

/* The revealed button is the saffron one (`styles.css .nudge-btn.hot`), not an invisible white pill. */
const hotOnReveal = api.nodes["nudge-btn"].classList.contains("hot");

/* Talking again hides it and re-arms the clock. */
api.mic(0.2);
const talksAgain = api.nodes["nudge-btn"].hidden;
const hotWhileQuietAgain = api.nodes["nudge-btn"].classList.contains("hot");
api.mic(0);
api.advance(3500);
const revealedAgain = !api.nodes["nudge-btn"].hidden;

/* A tap on the nudge clears it and sends one nudge. */
api.nodes["nudge-btn"].fire("click");
await api.settle();
const afterTap = api.nodes["nudge-btn"].hidden;
const hotAfterTap = api.nodes["nudge-btn"].classList.contains("hot");
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
const hotAfterReply = api.nodes["nudge-btn"].classList.contains("hot");
api.push({ type: "question_state", step: "fever_days", is_yes_no: true, quick_replies: [], prompt_chars: 60 });
api.push({ type: "audio", data: api.pcm(4800), sample_rate: 24000 });
await api.settle();
const afterBotAudio = api.nodes["nudge-btn"].hidden;
const hotAfterBotAudio = api.nodes["nudge-btn"].classList.contains("hot");
api.push({ type: "clear" });
api.advance(4000);
const afterClear = api.nodes["nudge-btn"].hidden;
const hotAfterClear = api.nodes["nudge-btn"].classList.contains("hot");

return { micReady, quietIdle, duringSpeech, stillSpeaking, twoSeconds, afterThree, hotOnReveal, talksAgain,
         hotWhileQuietAgain, revealedAgain, afterTap, hotAfterTap, nudges, nudgeDuringPrompting, afterReply,
         hotAfterReply, afterBotAudio, hotAfterBotAudio, afterClear, hotAfterClear };
""")

        self.assertTrue(out["micReady"], "the capture worklet is wired up by startCall")
        self.assertTrue(out["quietIdle"], "silence with no caller speech never nags")
        self.assertTrue(out["duringSpeech"], "the nudge never appears while the caller is mid-sentence")
        self.assertTrue(out["stillSpeaking"], "a long utterance keeps it hidden")
        self.assertTrue(out["twoSeconds"], "2 s of silence is too early")
        self.assertTrue(out["afterThree"], "past 3 s it must surface")
        self.assertTrue(out["hotOnReveal"], "the revealed nudge carries .hot so it paints saffron")
        self.assertTrue(out["talksAgain"], "speaking again hides it")
        self.assertFalse(out["hotWhileQuietAgain"], "and the .hot highlight goes with it")
        self.assertTrue(out["revealedAgain"], "and the clock restarts on silence")
        self.assertTrue(out["afterTap"], "the tap hides it")
        self.assertFalse(out["hotAfterTap"], "the tap clears the .hot highlight")
        self.assertEqual(out["nudges"], 1, "the tap sends exactly one nudge")
        self.assertTrue(out["nudgeDuringPrompting"], "no nudge in the 5 s prompting window after a tap")
        self.assertTrue(out["afterReply"], "a quick-reply tap arms the wait even with no mic frames")
        self.assertTrue(out["hotAfterReply"], "the re-revealed nudge is highlighted too")
        self.assertTrue(out["afterBotAudio"], "bot audio clears the wait")
        self.assertFalse(out["hotAfterBotAudio"], "and it drops the .hot highlight")
        self.assertTrue(out["afterClear"], "a clear clears the wait")
        self.assertFalse(out["hotAfterClear"], "and drops the .hot highlight")

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
api.need("download-pdf").fire("click");
await api.settle();
const adoptedDownload = api.downloads().pop();

api.setPdf("", true, false);
api.nodes["retry-pdf"].fire("click");
await api.settle();
const fallback = api.pdfName().pop();
api.need("download-pdf").fire("click");
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

    def test_playback_tail_allows_eager_reply_only_after_bot_turn_complete_inside_350ms(self):
        # Calls #33/#34/#41: she said "नहीं" 300 ms before SHISHU's line finished playing. `playing()` stayed true
        # until the very last sample, so her first word was never sent, and the model heard silence where the
        # answer was. By `bot_turn_complete` the browser holds the whole line and the server has closed the turn,
        # so the last 350 ms of the queue is a window where a voice is hers, never an echo.
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 44, share_token: "tok44", out_sample_rate: 24000 });
await api.settle();
api.push({ type: "question_state", step: "intro_married", epoch: 1, is_yes_no: false, quick_replies: [] });
/* 96 000 PCM bytes at 24 kHz = 48 000 samples = 2.0 s of her line, all of it still queued. */
api.push({ type: "audio", data: api.pcm(96000), sample_rate: 24000, epoch: 1 });
await api.settle();
const sent = () => api.sent.filter((m) => m.type === "audio").length;

/* 1) 200 ms left, but the model has not closed the turn: this is still her own voice. */
api.ctx().currentTime = 1.80;
api.mic(0.2);
await api.settle();
const beforeComplete = sent();

api.push({ type: "bot_turn_complete", step: "intro_married", epoch: 1 });
await api.settle();

/* 2) 1500 ms left: the turn is closed, but most of the line is still unheard. */
api.ctx().currentTime = 0.50;
api.mic(0.2);
await api.settle();
const midLine = sent();

/* 3) 500 ms left: outside the 350 ms tail, still muted. */
api.ctx().currentTime = 1.50;
api.mic(0.2);
await api.settle();
const outsideTail = sent();

/* 4) 300 ms left: inside the tail, so the eager "नहीं" is sent and the page shows it is listening. */
api.ctx().currentTime = 1.70;
api.mic(0.2);
await api.settle();
const insideTail = sent();
const turnPhase = vm.runInContext("turnPhase", sandbox);
const statusText = api.nodes["status"].children.map((c) => c.textContent).join("|");

return { beforeComplete, midLine, outsideTail, insideTail, turnPhase, statusText };
""")

        self.assertEqual(out["beforeComplete"], 0,
                         "no frame is sent while the model is still generating: turn_complete has not arrived")
        self.assertEqual(out["midLine"], 0, "1500 ms of her line still queued is not the tail")
        self.assertEqual(out["outsideTail"], 0, "500 ms left is still more than the 350 ms tail")
        self.assertEqual(out["insideTail"], 1,
                         "300 ms left, after turn_complete: the eager \"नहीं\" is her voice, not an echo")
        self.assertEqual(out["turnPhase"], "listening", "inside the tail the client is already listening")
        self.assertIn("सुन रही हूँ…", out["statusText"], "and it says so on screen")


    def test_preparing_state_animates_on_the_end_screen_until_the_pdf_lands(self):
        # The end screen used to say "Preparing report…" and then simply do nothing visible: `#wait` belongs to
        # the live stage, and `call_completed` flashed an empty thank-you card. The preparing indicator has to
        # live on #thanks, spin while the PDF is in flight, and be gone the moment the file is ready.
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
const idleHidden = !api.shown("thanks-preparing");

api.gatePdf();
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();
const waiting = api.shown("thanks-preparing");
const waitingThanks = api.shown("thanks");
const downloadHiddenWhileWaiting = !api.shown("download-pdf");
const spinner = api.tree("thanks-preparing").some((n) => n.cls.split(/\s+/).includes("prep-spinner"));
const hint = api.tree("thanks-preparing").map((n) => n.text).join("|");

api.releasePdf();
await api.settle();
const afterReady = api.shown("thanks-preparing");
const downloadAfterReady = api.shown("download-pdf");

return { idleHidden, waiting, waitingThanks, downloadHiddenWhileWaiting, spinner, hint, afterReady, downloadAfterReady };
""")

        self.assertTrue(out["idleHidden"], "the preparing indicator is hidden before the call ends")
        self.assertTrue(out["waiting"], "while the PDF is being built the end screen shows the preparing indicator")
        self.assertTrue(out["waitingThanks"], "and it does so on the #thanks screen, not the live stage")
        self.assertTrue(out["downloadHiddenWhileWaiting"], "no download button before there is a file to download")
        self.assertTrue(out["spinner"], "the indicator carries the .prep-spinner that styles.css animates")
        self.assertIn("रिपोर्ट तैयार हो रही है…", out["hint"], "and the pack's preparing copy")
        self.assertFalse(out["afterReady"], "once the PDF lands the preparing indicator is gone")
        self.assertTrue(out["downloadAfterReady"], "and the download button takes its place")

    def test_preparing_indicator_clears_when_the_pdf_build_fails(self):
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 9, share_token: "tok9", out_sample_rate: 24000 });
await api.settle();
api.failPdf(true);
api.push({ type: "call_completed", call_id: 9, share_token: "tok9" });
await api.settle();
const preparing = api.shown("thanks-preparing");
const retry = api.shown("retry-pdf");
const download = api.shown("download-pdf");
return { preparing, retry, download };
""")

        self.assertFalse(out["preparing"], "a failed build must not leave the spinner running forever")
        self.assertTrue(out["retry"], "a failed build offers the rebuild button")
        self.assertFalse(out["download"], "and never a download button with no file behind it")

    def test_download_button_has_an_svg_arrow_and_never_calls_navigator_share(self):
        # Windows Safari/Chrome and the iPhone home-screen browser all behaved differently through
        # navigator.share. A plain <a download> is the one thing every target does the same way.
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();

/* Both share APIs present and happy: the page must ignore them anyway. */
api.setPdf('attachment; filename="SHISHU-report.pdf"', false, true);
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();

const hasButton = api.has("download-pdf");
const tree = api.tree("download-pdf");
const svg = tree.find((n) => n.tag === "SVG");
const arrow = tree.find((n) => n.tag === "PATH");
const label = tree.find((n) => n.cls.split(/\s+/).includes("btn-label"));
const revokedBefore = api.revoked().length;
api.need("download-pdf").fire("click");
await api.settle();
const shareCalls = api.shares();
const downloads = api.downloads();
/* The blob must outlive the click: revoking it in the same tick can cancel the download on Safari. */
const revokedAtClick = api.revoked().length;
api.advance(1500);
const revokedLater = api.revoked().length;

return { hasButton, svg: svg || null, arrow: arrow || null, shareCalls, downloads,
         revokedAtClick: revokedAtClick - revokedBefore, revokedLater: revokedLater - revokedBefore,
         label: label ? label.text : "", button: tree.map((n) => n.text).join("|") };
""")

        self.assertTrue(out["hasButton"], "the end screen ships a #download-pdf button")
        self.assertIsNotNone(out["svg"], "the button carries an inline SVG download icon")
        self.assertEqual(out["svg"]["aria"], "true", "the icon is decorative: aria-hidden on the SVG")
        self.assertIsNotNone(out["arrow"], "the icon is drawn with a <path>")
        self.assertIn("M12 3", out["arrow"]["d"] or "",
                      "the icon is the standard down-arrow: the arrow shaft starts at the top of the viewBox")
        self.assertIn("h12.4", out["arrow"]["d"] or "", "and ends in the tray/bar the arrow drops into")
        self.assertIn("डाउनलोड करें", out["label"], "the pack's Hindi download copy lands on the button")
        self.assertIn("Download", out["label"], "beside the English copy")
        self.assertEqual(out["shareCalls"], 0,
                         "navigator.share must never be touched, even when it exists and canShare says yes")
        self.assertEqual(out["downloads"], ["SHISHU-report.pdf"], "the click is a direct <a download>")
        self.assertEqual(out["revokedAtClick"], 0, "the blob URL outlives the click itself")
        self.assertEqual(out["revokedLater"], 1, "and is released a moment later, so memory is not leaked")

    def test_download_icon_survives_the_pack_copy_that_rewrites_the_button_label(self):
        # applyUi() calls replaceChildren() on every [data-ui] node, which would shred an inline <svg> that sat
        # directly inside the button. The icon lives beside a .btn-label span, so the pack only repaints the text.
        out = self.run_js(r"""
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
api.setPdf('attachment; filename="SHISHU-report.pdf"', false, true);
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();
const tree = api.tree("download-pdf");
return { svg: tree.filter((n) => n.tag === "SVG").length,
         label: tree.find((n) => n.cls.split(/\s+/).includes("btn-label")) || null,
         text: api.need("download-pdf").textContent };
""")

        self.assertEqual(out["svg"], 1, "the SVG down-arrow is still in the button after the pack copy applies")
        self.assertIsNotNone(out["label"], "the label rides in its own .btn-label span")
        self.assertIn("डाउनलोड करें", out["label"]["text"] if out["label"] else "",
                      "and the pack's Hindi copy is what the span shows")

    def test_both_packs_ship_the_download_copy_and_no_share_instruction(self):
        # The end screen has no share step any more, so no pack may still tell her to tap 'शेयर करें'.
        for pack_id in ("shishu-hi", "fever-opd-hi"):
            with self.subTest(pack=pack_id):
                ui = load_pack(pack_id).meta["ui"]["call"]
                self.assertIn("download_btn", ui, f"{pack_id} ships the download button copy")
                self.assertNotIn("share_btn", ui, f"{pack_id} no longer ships share button copy")
                hi, en = ui["thanks_summary"]["hi"], ui["thanks_summary"]["en"]
                self.assertNotIn("शेयर करें", hi, f"{pack_id} stops pointing at a Share button in Hindi")
                self.assertNotIn("Share", en, f"{pack_id} stops pointing at a Share button in English")
                self.assertIn("डाउनलोड करें", hi, f"{pack_id} points at Download in Hindi")
                self.assertIn("Download", en, f"{pack_id} points at Download in English")
                self.assertEqual(ui["download_btn"]["hi"], "डाउनलोड करें")
                self.assertEqual(ui["download_btn"]["en"], "Download")


if __name__ == "__main__":
    unittest.main()
