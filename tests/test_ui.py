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
let pdfHeaders = "", pdfNoHeaders = false, pdfFails = false, pdfGate = false;
const pdfPending = [];
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
    File: function (parts, name, cfg) { this.parts = parts; this.name = name; this.type = cfg && cfg.type; fileNames.push({ name, lang: parts[0] && parts[0].lang }); },
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
      /* Two sheets, two requests: tag every call with its language so a test can gate, fail or name them apart. */
      const lang = (/[?&]lang=([a-z]+)/.exec(url) || [null, "hi"])[1];
      pdfCalls.push({ lang, method: (init && init.method) || "GET", url });
      const disposition = pdfHeaders.replace(/\{lang\}/g, lang);
      const headers = pdfNoHeaders ? undefined : { get: (k) => (k.toLowerCase() === "content-disposition" ? disposition : null) };
      const fails = pdfFails === true || (Array.isArray(pdfFails) && pdfFails.includes(lang));
      const reply = () => {
        if (fails) return { ok: false, status: 500, headers, blob: () => Promise.resolve(new Blob([])) };
        const blob = new Blob([]);
        blob.lang = lang;
        return { ok: true, status: 200, headers, blob: () => Promise.resolve(blob) };
      };
      if (pdfGate === true || (Array.isArray(pdfGate) && pdfGate.includes(lang))) {
        return new Promise((resolve) => { pdfPending.push({ lang, resolve: () => resolve(reply()) }); });
      }
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
  pdfName() { return fileNames.map((f) => f.name); },
  /* Which language each built File belongs to, so a test can prove the two sheets never swap names or files. */
  pdfFiles() { return fileNames.slice(); },
  downloads() { return anchors.map((a) => a.download); },
  shares() { return shareCalls; },
  revoked() { return revoked.slice(); },
  /* `has` is the only honest "this id exists on the served page" check: a missing id would otherwise
     be conjured by getElementById with `hidden === false` and read as visible. */
  has(id) { return htmlIds.has(id); },
  setPdf(disposition, noHeaders, shareable) { pdfHeaders = disposition || ""; pdfNoHeaders = !!noHeaders; canShare = shareable !== false; },
  /* `true` fails both sheets; an array fails only the named languages, so one sheet can fail alone. */
  failPdf(fails) { pdfFails = fails === true ? true : (Array.isArray(fails) ? fails.slice() : false); },
  /* Hold the PDF responses open, so "preparing" can be observed mid-flight instead of guessed at.
     `langs` names the sheets to hold; with no argument both are held, as before. */
  gatePdf(langs) { pdfGate = langs == null ? true : (Array.isArray(langs) ? langs.slice() : [langs]); },
  /* Release the held sheets — only the named languages when given, else everything that is waiting. */
  releasePdf(langs) {
    const wanted = langs == null ? null : (Array.isArray(langs) ? langs : [langs]);
    for (let i = pdfPending.length - 1; i >= 0; i -= 1) {
      if (wanted && wanted.indexOf(pdfPending[i].lang) < 0) continue;
      pdfPending.splice(i, 1)[0].resolve();
    }
    if (!pdfPending.length) pdfGate = false;
  },
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
        # Two sheets, two responses: each button must adopt its OWN served filename, never the other language's.
        out = self.run_js(r"""
const byLang = (rows) => Object.fromEntries(rows.map((r) => [r.lang, r.name]));
const endCall = async () => {
  api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
  await api.settle();
};

api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();

api.setPdf('attachment; filename="SHISHU-2026-11-12-093000-c4-{lang}.pdf"', false, false);
await endCall();
const adopted = byLang(api.pdfFiles());
api.need("download-pdf-hi").fire("click");
api.need("download-pdf-en").fire("click");
await api.settle();
const adoptedDownloads = api.downloads();

/* A fresh page whose server sends no Content-Disposition at all: each sheet still lands on a
   language-tagged name of its own instead of one shared "SHISHU.pdf". */
const seenFiles = api.pdfFiles().length, seenDownloads = api.downloads().length;
api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
api.setPdf("", true, false);
await endCall();
const fallback = byLang(api.pdfFiles().slice(seenFiles));
api.need("download-pdf-hi").fire("click");
api.need("download-pdf-en").fire("click");
await api.settle();
const fallbackDownloads = api.downloads().slice(seenDownloads);

return { adopted, fallback, adoptedDownloads, fallbackDownloads };
""")

        self.assertEqual(out["adopted"], {"hi": "SHISHU-2026-11-12-093000-c4-hi.pdf",
                                          "en": "SHISHU-2026-11-12-093000-c4-en.pdf"})
        self.assertEqual(out["adoptedDownloads"], ["SHISHU-2026-11-12-093000-c4-hi.pdf",
                                                   "SHISHU-2026-11-12-093000-c4-en.pdf"])
        self.assertEqual(out["fallback"], {"hi": "SHISHU-hi.pdf", "en": "SHISHU-en.pdf"},
                         "no Content-Disposition still yields a name, and it names its own language")
        self.assertEqual(out["fallbackDownloads"], ["SHISHU-hi.pdf", "SHISHU-en.pdf"])

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
const labels = () => ({
  hi: api.need("download-pdf-hi-label").children.map((c) => c.textContent).join("|"),
  en: api.need("download-pdf-en-label").children.map((c) => c.textContent).join("|")
});

api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
const idleHidden = !api.shown("thanks-preparing");
const idleButtonsHidden = [!api.shown("download-pdf-hi"), !api.shown("download-pdf-en")];

/* Both sheets are held open, so the mid-flight state is observed instead of guessed at. */
api.gatePdf();
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();
const waiting = api.shown("thanks-preparing");
const waitingThanks = api.shown("thanks");
/* Each button is on screen with its own preparing copy, but there is no file behind either one yet. */
const shownWhileWaiting = [api.shown("download-pdf-hi"), api.shown("download-pdf-en")];
const disabledWhileWaiting = [api.need("download-pdf-hi").disabled, api.need("download-pdf-en").disabled];
const downloadsWhileWaiting = api.downloads().length;
api.need("download-pdf-hi").fire("click");
api.need("download-pdf-en").fire("click");
await api.settle();
const clicksWhileWaiting = { builds: api.pdfCalls.length, downloads: api.downloads().length };
const bothLoading = [api.need("download-pdf-hi").classList.contains("sheet-loading"),
                    api.need("download-pdf-en").classList.contains("sheet-loading")];
const spinner = api.tree("thanks-preparing").some((n) => n.cls.split(/\s+/).includes("prep-spinner"));
const hint = api.tree("thanks-preparing").map((n) => n.text).join("|");
const waitingLabels = labels();

/* Hindi lands first: its own button unlocks while English is still building, and the hint stays up. */
api.releasePdf("hi");
await api.settle();
const hindiOnly = {
  preparing: api.shown("thanks-preparing"),
  hi: api.shown("download-pdf-hi"), en: api.shown("download-pdf-en"),
  hiDisabled: api.need("download-pdf-hi").disabled, enDisabled: api.need("download-pdf-en").disabled,
  hiLoading: api.need("download-pdf-hi").classList.contains("sheet-loading"),
  enLoading: api.need("download-pdf-en").classList.contains("sheet-loading"),
  hiIcon: !api.need("download-pdf-hi-icon").hidden, enIcon: !api.need("download-pdf-en-icon").hidden,
  labels: labels()
};

api.releasePdf();
await api.settle();
const afterReady = api.shown("thanks-preparing");
const bothReady = [api.shown("download-pdf-hi"), api.shown("download-pdf-en")];
const readyLabels = labels();

return { idleHidden, idleButtonsHidden, waiting, waitingThanks, shownWhileWaiting, disabledWhileWaiting,
         downloadsWhileWaiting, clicksWhileWaiting, bothLoading, spinner, hint,
         waitingLabels, hindiOnly, afterReady, bothReady, readyLabels };
""")

        self.assertTrue(out["idleHidden"], "the preparing indicator is hidden before the call ends")
        self.assertEqual(out["idleButtonsHidden"], [True, True], "and so is every download button")
        self.assertTrue(out["waiting"], "while the PDF is being built the end screen shows the preparing indicator")
        self.assertTrue(out["waitingThanks"], "and it does so on the #thanks screen, not the live stage")
        self.assertEqual(out["shownWhileWaiting"], [True, True],
                         "each language shows its own button while its file is being built")
        self.assertEqual(out["disabledWhileWaiting"], [True, True], "but neither is clickable with no file behind it")
        self.assertEqual(out["downloadsWhileWaiting"], 0, "and no download happens before a PDF exists")
        self.assertEqual(out["clicksWhileWaiting"]["downloads"], 0, "a click while preparing saves nothing")
        self.assertEqual(out["bothLoading"], [True, True], "both buttons say they are preparing")
        self.assertTrue(out["spinner"], "the indicator carries the .prep-spinner that styles.css animates")
        self.assertIn("रिपोर्ट तैयार हो रही है…", out["hint"], "and the pack's preparing copy")
        self.assertIn("हिन्दी PDF तैयार हो रही है…", out["waitingLabels"]["hi"])
        self.assertIn("Preparing English PDF…", out["waitingLabels"]["en"])

        # The two sheets settle independently: Hindi ready, English still preparing.
        self.assertTrue(out["hindiOnly"]["preparing"], "one language still building keeps the shared hint up")
        self.assertTrue(out["hindiOnly"]["hi"], "the Hindi button is on screen the moment the Hindi file lands")
        self.assertFalse(out["hindiOnly"]["hiDisabled"], "and it is clickable, because its file is here")
        self.assertTrue(out["hindiOnly"]["enDisabled"], "the English button stays unclickable while English builds")
        self.assertFalse(out["hindiOnly"]["hiLoading"], "and Hindi no longer claims to be preparing")
        self.assertTrue(out["hindiOnly"]["enLoading"], "while English still does")
        self.assertEqual([out["hindiOnly"]["hiIcon"], out["hindiOnly"]["enIcon"]], [True, False],
                         "the down-arrow appears only on the sheet that actually has a file")
        self.assertIn("हिन्दी PDF डाउनलोड करें", out["hindiOnly"]["labels"]["hi"])
        self.assertIn("Preparing English PDF…", out["hindiOnly"]["labels"]["en"])

        self.assertFalse(out["afterReady"], "once both PDFs land the preparing indicator is gone")
        self.assertEqual(out["bothReady"], [True, True], "and both download buttons take its place")
        self.assertIn("हिन्दी PDF डाउनलोड करें", out["readyLabels"]["hi"])
        self.assertIn("Download English PDF", out["readyLabels"]["en"])

    def test_preparing_indicator_clears_when_the_pdf_build_fails(self):
        out = self.run_js(r"""
const labels = () => ({
  hi: api.need("download-pdf-hi-label").children.map((c) => c.textContent).join("|"),
  en: api.need("download-pdf-en-label").children.map((c) => c.textContent).join("|")
});

api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 9, share_token: "tok9", out_sample_rate: 24000 });
await api.settle();
api.failPdf(true);
api.push({ type: "call_completed", call_id: 9, share_token: "tok9" });
await api.settle();
const failed = {
  preparing: api.shown("thanks-preparing"),
  hi: api.shown("download-pdf-hi"), en: api.shown("download-pdf-en"),
  hiDisabled: api.need("download-pdf-hi").disabled, enDisabled: api.need("download-pdf-en").disabled,
  hiFailed: api.need("download-pdf-hi").classList.contains("sheet-failed"),
  enFailed: api.need("download-pdf-en").classList.contains("sheet-failed"),
  hiIcon: !api.need("download-pdf-hi-icon").hidden, enIcon: !api.need("download-pdf-en-icon").hidden,
  labels: labels()
};
const downloadsBefore = api.downloads().length;
api.need("download-pdf-hi").fire("click");
await api.settle();
const afterRetryClick = { downloads: api.downloads().length - downloadsBefore, builds: api.pdfCalls.length };

return { failed, afterRetryClick, downloadsBefore };
""")

        f = out["failed"]
        self.assertFalse(f["preparing"], "a failed build must not leave the spinner running forever")
        self.assertEqual([f["hi"], f["en"]], [True, True],
                         "each failed language keeps its own button, offering a rebuild")
        self.assertEqual([f["hiDisabled"], f["enDisabled"]], [False, False], "and both are clickable again")
        self.assertEqual([f["hiFailed"], f["enFailed"]], [True, True], "both carry the failed, not the ready, styling")
        self.assertEqual([f["hiIcon"], f["enIcon"]], [False, False],
                         "a failed sheet never shows the download arrow it has no file for")
        self.assertIn("हिन्दी PDF दोबारा बनाएँ", f["labels"]["hi"], "the Hindi button says it can rebuild")
        self.assertIn("Retry English PDF", f["labels"]["en"], "and so does the English one")
        self.assertEqual(out["afterRetryClick"]["downloads"], 0,
                         "clicking a failed sheet rebuilds it; it never downloads a file that does not exist")
        self.assertEqual(out["afterRetryClick"]["builds"], 3,
                         "the retry click sent one new build, on top of the two that failed")

    def test_download_button_has_an_svg_arrow_and_never_calls_navigator_share(self):
        # Windows Safari/Chrome and the iPhone home-screen browser all behaved differently through
        # navigator.share. A plain <a download> is the one thing every target does the same way.
        out = self.run_js(r"""
const icon = (lang) => {
  const tree = api.tree("download-pdf-" + lang);
  const svg = tree.find((n) => n.tag === "SVG");
  const path = tree.find((n) => n.tag === "PATH");
  const label = tree.find((n) => n.cls.split(/\s+/).includes("btn-label"));
  return { svg: svg || null, path: path || null, label: label ? label.text : "",
           text: tree.map((n) => n.text).join("|") };
};

api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();

/* Both share APIs present and happy: the page must ignore them anyway. */
api.setPdf('attachment; filename="SHISHU-report-{lang}.pdf"', false, true);
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();

const hasButtons = [api.has("download-pdf-hi"), api.has("download-pdf-en")];
const hi = icon("hi"), en = icon("en");
const revokedBefore = api.revoked().length;
api.need("download-pdf-hi").fire("click");
await api.settle();
const revokedAtHiClick = api.revoked().length;
const sharesAfterHi = api.shares();
api.need("download-pdf-en").fire("click");
await api.settle();
const shareCalls = api.shares();
const downloads = api.downloads();
/* The blob must outlive the click: revoking it in the same tick can cancel the download on Safari. */
const revokedAtClick = revokedAtHiClick - revokedBefore;
api.advance(1500);
const revokedLater = api.revoked().length - revokedBefore;

return { hasButtons, hi, en, sharesAfterHi, shareCalls, downloads,
         revokedAtClick, revokedLater, shared: sharesAfterHi > 0 || shareCalls > 0 };
""")

        self.assertEqual(out["hasButtons"], [True, True],
                         "the end screen ships one download button per language")
        for lang in ("hi", "en"):
            with self.subTest(lang=lang):
                icon = out[lang]
                self.assertIsNotNone(icon["svg"], "the button carries an inline SVG download icon")
                self.assertEqual(icon["svg"]["aria"], "true", "the icon is decorative: aria-hidden on the SVG")
                self.assertIsNotNone(icon["path"], "the icon is drawn with a <path>")
                self.assertIn("M12 3", icon["path"]["d"] or "",
                              "the icon is the standard down-arrow: the arrow shaft starts at the top of the viewBox")
                self.assertIn("h12.4", icon["path"]["d"] or "",
                              "and ends in the tray/bar the arrow drops into")
                self.assertTrue(icon["label"], "the label rides in its own span, beside the icon")
        self.assertIn("डाउनलोड करें", out["hi"]["label"], "the Hindi button says download in Hindi")
        self.assertIn("Download Hindi PDF", out["hi"]["label"], "and names its own language in English")
        self.assertIn("डाउनलोड करें", out["en"]["label"], "the English button says download in Hindi too")
        self.assertIn("Download English PDF", out["en"]["label"], "and names its own language in English")
        self.assertEqual(out["shareCalls"], 0,
                         "navigator.share must never be touched, even when it exists and canShare says yes")
        self.assertFalse(out["shared"], "no share API was reached on either click")
        self.assertEqual(out["downloads"], ["SHISHU-report-hi.pdf", "SHISHU-report-en.pdf"],
                         "each click is a direct <a download> of that language's own file")
        self.assertEqual(out["revokedAtClick"], 0, "the blob URL outlives the click itself")
        self.assertEqual(out["revokedLater"], 2, "each blob is released a moment later, so memory is not leaked")

    def test_download_icon_survives_the_pack_copy_that_rewrites_the_button_label(self):
        # applyUi() calls replaceChildren() on every [data-ui] node, which would shred an inline <svg> that sat
        # directly inside the button. The icon lives beside a .btn-label span, so the pack only repaints the text.
        out = self.run_js(r"""
const icon = (lang) => {
  const tree = api.tree("download-pdf-" + lang);
  const label = tree.find((n) => n.cls.split(/\s+/).includes("btn-label"));
  return { svg: tree.filter((n) => n.tag === "SVG").length, label: label || null,
           text: label ? label.text : "" };
};

api.boot(JSON.parse(process.env.PACK));
await api.settle();
await api.start();
api.push({ type: "session_started", call_id: 4, share_token: "tok", out_sample_rate: 24000 });
await api.settle();
api.setPdf('attachment; filename="SHISHU-report-{lang}.pdf"', false, true);
api.push({ type: "call_completed", call_id: 4, share_token: "tok" });
await api.settle();
return { hi: icon("hi"), en: icon("en") };
""")

        for lang in ("hi", "en"):
            with self.subTest(lang=lang):
                icon = out[lang]
                self.assertEqual(icon["svg"], 1,
                                 f"the {lang} SVG down-arrow is still in the button after the pack copy applies")
                self.assertIsNotNone(icon["label"], "the label rides in its own .btn-label span")
                self.assertIn("डाउनलोड करें", icon["label"]["text"] if icon["label"] else "",
                              "and the pack's Hindi download copy is what the span shows")
                self.assertIn(lang.capitalize() if lang == "en" else "PDF", icon["text"],
                              "with each button naming its own language")

    def test_both_packs_ship_the_download_copy_and_no_share_instruction(self):
        # The end screen has no share step any more, so no pack may still tell her to tap 'शेयर करें'.
        # SHISHU's summary is now the factual, bilingual record line; the OPD pack still names its download button.
        expected = {"shishu-hi": {"hi": ("डाउनलोड", "कॉल का रिकॉर्ड"), "en": ("record", "Hindi and English")},
                    "fever-opd-hi": {"hi": ("डाउनलोड करें",), "en": ("Download",)}}
        for pack_id, wanted in expected.items():
            with self.subTest(pack=pack_id):
                ui = load_pack(pack_id).meta["ui"]["call"]
                self.assertIn("download_btn", ui, f"{pack_id} ships the download button copy")
                self.assertNotIn("share_btn", ui, f"{pack_id} no longer ships share button copy")
                hi, en = ui["thanks_summary"]["hi"], ui["thanks_summary"]["en"]
                self.assertNotIn("शेयर करें", hi, f"{pack_id} stops pointing at a Share button in Hindi")
                self.assertNotIn("Share", en, f"{pack_id} stops pointing at a Share button in English")
                for phrase in wanted["hi"]:
                    self.assertIn(phrase, hi, f"{pack_id} Hindi summary keeps {phrase!r}")
                for phrase in wanted["en"]:
                    self.assertIn(phrase, en, f"{pack_id} English summary keeps {phrase!r}")
                self.assertEqual(ui["download_btn"]["hi"], "डाउनलोड करें")
                self.assertEqual(ui["download_btn"]["en"], "Download")

    def test_shishu_thanks_summary_is_factual_and_names_both_languages(self):
        # The old line promised a list of questions to ask the doctor. The record is now a factual call record,
        # so the copy says what it is (the call record) and where it comes from (both languages), never advice.
        ui = load_pack("shishu-hi").meta["ui"]["call"]
        hi, en = ui["thanks_summary"]["hi"], ui["thanks_summary"]["en"]
        self.assertEqual(hi, "इस कॉल का रिकॉर्ड नीचे हिन्दी और English दोनों में डाउनलोड के लिए उपलब्ध है।")
        self.assertEqual(en, "Your call record is available below in both Hindi and English.")
        for text in (hi, en):
            for banned in ("सवाल", "डॉक्टर", "questions", "doctor", "अगली मुलाक़ात", "next visit"):
                self.assertNotIn(banned, text, f"the thanks summary still reads as advice: {banned!r}")


if __name__ == "__main__":
    unittest.main()
