"""Stage 3: same-size flanking replies, admin cursor answers, and the live tap path."""

import asyncio
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests import APP_DIR  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="shishu-v4-ui-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import app  # noqa: E402
import db  # noqa: E402
import decisions  # noqa: E402
import gemini_live  # noqa: E402
import profile_extractor  # noqa: E402
import report_service  # noqa: E402
import test_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_profile, apply_turn, new_cursor_state  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_STATIC = APP_DIR / "static"
_NODE = shutil.which("node")


class _ScriptedSession:
    """Gemini stand-in. The test pushes server messages; the call loop reads them."""

    def __init__(self):
        self.setup_complete = True
        self.sent = []
        self._lock = threading.Lock()
        self._pending = []
        self._loop = None
        self._queue = None

    def push(self, item):
        with self._lock:
            loop = self._loop
            mailbox = self._queue
            if loop is None or mailbox is None:
                self._pending.append(item)
                return
        loop.call_soon_threadsafe(mailbox.put_nowait, item)

    async def send_realtime_input(self, **kwargs):
        self.sent.append(("realtime", kwargs))

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses))

    async def receive(self):
        with self._lock:
            if self._queue is None:
                self._loop = asyncio.get_running_loop()
                self._queue = asyncio.Queue()
                pending = list(self._pending)
                self._pending.clear()
            else:
                pending = []
        for item in pending:
            self._queue.put_nowait(item)
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


class _ScriptCtx:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        self.session.push(None)
        return False


def _server_message(*, tool_calls=None, parts=None, user_text="", turn_complete=False):
    tool_call = SimpleNamespace(function_calls=tool_calls) if tool_calls else None
    model_turn = SimpleNamespace(parts=parts) if parts else None
    return SimpleNamespace(
        tool_call_cancellation=None,
        server_content=SimpleNamespace(
            interrupted=False,
            input_transcription=SimpleNamespace(text=user_text) if user_text else None,
            output_transcription=None,
            model_turn=model_turn,
            turn_complete=turn_complete,
        ),
        tool_call=tool_call,
        session_resumption_update=None,
        go_away=None,
    )


def _tool(fid, name, args):
    return SimpleNamespace(id=fid, name=name, args=args, will_continue=False)


def _audio_message():
    part = SimpleNamespace(
        thought=False,
        function_call=None,
        inline_data=SimpleNamespace(
            data=b"\x01\x00" * 240,
            mime_type="audio/pcm;rate=24000",
        ),
    )
    return _server_message(parts=[part], turn_complete=True)


class QuickReplyScreenTests(unittest.TestCase):
    def test_call_page_serves_a_same_size_three_button_dock(self):
        client = TestClient(app.app)
        client.post("/api/login", json={"username": "Mother", "passcode": app._passcode()})
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        served = page.text

        dock_at = served.index('id="call-dock"')
        no_at = served.index('id="no-btn"')
        call_at = served.index('id="call-btn"')
        yes_at = served.index('id="yes-btn"')
        self.assertLess(dock_at, no_at)
        self.assertLess(no_at, call_at)
        self.assertLess(call_at, yes_at)
        self.assertEqual(served.count('id="call-btn"'), 1)
        self.assertIn('aria-label="नहीं, आगे बढ़ें (No)"', served)
        self.assertIn('aria-label="हाँ (Yes)"', served)
        self.assertIn("clamp(68px, 21vw, 84px)", served)
        self.assertIn(".stage.working #call-dock", served)
        self.assertIn(".stage.working #call-btn", served)
        self.assertIn("prefers-reduced-motion: reduce", served)
        self.assertNotIn("engine-note", served)

    def test_speaker_clock_and_quick_reply_click_behavior(self):
        if _NODE is None:
            self.skipTest("node is not installed")
        result = _run_call_js_behavior()

        idle = result["idle"]
        self.assertTrue(idle["noHidden"])
        self.assertTrue(idle["yesHidden"])
        self.assertFalse(idle["noVisible"])
        self.assertFalse(idle["yesVisible"])

        self.assertEqual(result["hudModel"], "gemini-3.1-flash-live-preview")

        # Non-yes/no question: mounted in live phase without jumping layout, stays hidden/disabled
        non_yn_mounted = result["nonYnMounted"]
        self.assertFalse(non_yn_mounted["noHidden"])
        self.assertFalse(non_yn_mounted["yesHidden"])
        self.assertFalse(non_yn_mounted["noVisible"])
        self.assertFalse(non_yn_mounted["yesVisible"])
        self.assertTrue(non_yn_mounted["noDisabled"])
        self.assertTrue(non_yn_mounted["yesDisabled"])

        non_yn_after_turn = result["nonYnAfterTurn"]
        self.assertFalse(non_yn_after_turn["noVisible"])
        self.assertFalse(non_yn_after_turn["yesVisible"])
        self.assertTrue(non_yn_after_turn["noDisabled"])
        self.assertTrue(non_yn_after_turn["yesDisabled"])

        # Yes/no question: invisible before 50% playback, visible & enabled at 50%
        before = result["beforeHalf"]
        self.assertFalse(before["noVisible"])
        self.assertFalse(before["yesVisible"])
        self.assertTrue(before["noDisabled"])
        self.assertTrue(before["yesDisabled"])

        halfway = result["halfway"]
        self.assertTrue(halfway["noVisible"])
        self.assertTrue(halfway["yesVisible"])
        self.assertFalse(halfway["noDisabled"])
        self.assertFalse(halfway["yesDisabled"])
        self.assertFalse(halfway["noHidden"])
        self.assertFalse(halfway["yesHidden"])

        # Mismatched bot_turn_complete is ignored; matching step reveals buttons
        self.assertFalse(result["otherStep"]["noVisible"])
        self.assertFalse(result["otherStep"]["yesVisible"])
        finished = result["turnComplete"]
        self.assertTrue(finished["noVisible"])
        self.assertTrue(finished["yesVisible"])
        self.assertFalse(finished["noDisabled"])
        self.assertFalse(finished["yesDisabled"])

        # Clicking #no-btn stops playback, disables buttons, sets caption, sends quick_reply once
        self.assertTrue(result["sourceStoppedOnClick"])
        self.assertTrue(result["afterNoClick"]["noDisabled"])
        self.assertTrue(result["afterNoClick"]["yesDisabled"])
        self.assertEqual(result["captionAfterNo"], "You: नहीं।")
        self.assertEqual(
            result["quickRepliesSent"],
            [{"type": "quick_reply", "outcome": "clear", "step": "birth_nicu"}],
        )

        # Clicking #yes-btn on a new yes/no question sends concern
        self.assertEqual(result["captionAfterYes"], "You: हाँ।")
        self.assertEqual(
            result["yesReplySent"],
            [{"type": "quick_reply", "outcome": "concern", "step": "birth_jaundice"}],
        )

        # ending message transitions to preparing and hides/resets quick-reply buttons
        ending = result["afterEnding"]
        self.assertTrue(ending["stageWorking"])
        self.assertFalse(ending["waitHidden"])
        self.assertTrue(ending["noHidden"])
        self.assertTrue(ending["yesHidden"])
        self.assertFalse(ending["noVisible"])
        self.assertFalse(ending["yesVisible"])

    def test_admin_dashboard_boots_and_renders_cursor_answers(self):
        if _NODE is None:
            self.skipTest("node is not installed")
        rendered = _run_admin_js_behavior()

        self.assertNotIn("/api/config", rendered["fetchedUrls"])
        self.assertEqual(rendered["headers"], ["Question ID", "State", "Detail", "Red flag"])
        self.assertEqual(rendered["modelBadge"], "Gemini 3.1 Flash Live")
        self.assertIn("पूजा · 2 साल", rendered["callMeta"])
        self.assertNotIn("अमन", rendered["callMeta"])
        self.assertNotIn("female", rendered["callMeta"])
        self.assertIn("5 tool calls / 1 nudges", rendered["kpis"])

        rows = rendered["rows"]
        self.assertEqual(
            [row["state"] for row in rows],
            ["concern", "concern", "unknown", "clear"],
        )
        concern = rows[0]
        self.assertEqual(concern["qid"], "birth_mode_detail")
        self.assertEqual(concern["flag"], "Yes")
        self.assertEqual(concern["rowClass"], "flag")
        self.assertIn("child_1", concern["detail"])
        self.assertIn("ऑपरेशन", concern["detail"])
        self.assertIn("operation_reason: सांस की तकलीफ", concern["detail"])

        bridge = next(row for row in rows if row["qid"] == "intro_bridge")
        self.assertEqual(bridge["state"], "concern")
        self.assertEqual(bridge["flag"], "")
        self.assertEqual(bridge["rowClass"], "")
        self.assertNotIn("preg_scan", [row["qid"] for row in rows])

        self.assertEqual(rendered["emptyText"], "No structured findings were recorded.")
        self.assertEqual(rendered["copiedReport"], "# Doctor Summary\nAll good.")
        self.assertEqual(rendered["copyNotice"], "Report copied.")


def _run_call_js_behavior() -> dict:
    import subprocess

    harness = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.env.CALL_JS, "utf8");

function el() {
  const set = new Set();
  const attrs = {};
  const listeners = {};
  return {
    hidden: false,
    disabled: false,
    textContent: "",
    className: "",
    innerHTML: "",
    value: "",
    checked: false,
    style: {},
    children: [],
    classList: {
      add(name) { set.add(name); },
      remove(name) { set.delete(name); },
      contains(name) { return set.has(name); },
      toggle(name, force) {
        const on = force === undefined ? !set.has(name) : !!force;
        if (on) set.add(name); else set.delete(name);
        return on;
      }
    },
    addEventListener(type, fn) {
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push(fn);
    },
    trigger(type, event) {
      const list = listeners[type] || [];
      for (const fn of list) fn(event || { target: this, preventDefault() {} });
    },
    click() { this.trigger("click"); },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k] || ""; },
    removeAttribute(k) { delete attrs[k]; },
    appendChild(child) { this.children.push(child); return child; },
    remove() {},
    querySelectorAll() { return []; },
    closest() { return null; },
    replaceChildren() { this.children = []; }
  };
}

function pcm(frames) {
  const bytes = Buffer.alloc(frames * 2);
  for (let i = 0; i < frames; i++) bytes.writeInt16LE(800, i * 2);
  return bytes.toString("base64");
}

function makeCtx(now) {
  const ctx = {
    currentTime: now,
    state: "running",
    destination: {},
    sampleRate: 48000,
    sources: [],
    resume() { return Promise.resolve(); },
    close() { return Promise.resolve(); },
    createMediaStreamSource() {
      return { connect() {}, disconnect() {} };
    },
    createScriptProcessor() {
      return { onaudioprocess: null, connect() {}, disconnect() {} };
    },
    createGain() {
      return { gain: { value: 0 }, connect() {}, disconnect() {} };
    },
    createBuffer(_channels, length, rate) {
      return {
        duration: length / rate,
        getChannelData() { return new Float32Array(length); }
      };
    },
    createBufferSource() {
      const src = {
        buffer: null,
        stopped: false,
        connect() {},
        start() {},
        stop() { this.stopped = true; },
        disconnect() {},
        onended: null
      };
      ctx.sources.push(src);
      return src;
    }
  };
  return ctx;
}

async function flushMicrotasks() {
  for (let i = 0; i < 32; i++) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

function boot() {
  const frames = [];
  let nextId = 1;
  const ids = {};
  const sockets = [];
  let activeCtx = null;

  const initiallyHidden = new Set([
    "no-btn", "yes-btn", "wait", "nudge-btn", "thanks", "hud",
    "hud-toggle", "admin-link", "admin-view", "error", "bug-overlay"
  ]);

  const document = {
    getElementById(id) {
      if (!ids[id]) {
        ids[id] = el();
        if (initiallyHidden.has(id)) ids[id].hidden = true;
      }
      return ids[id];
    },
    addEventListener() {},
    createElement() { return el(); },
    querySelector() { return null; },
    body: el(),
    visibilityState: "visible"
  };

  function MockWebSocket() {
    this.readyState = 1;
    this.bufferedAmount = 0;
    this.sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    sockets.push(this);
    queueMicrotask(() => { if (this.onopen) this.onopen(); });
  }
  MockWebSocket.OPEN = 1;
  MockWebSocket.prototype.send = function (raw) {
    this.sent.push(JSON.parse(raw));
  };
  MockWebSocket.prototype.close = function () {};

  const store = {};
  const sandbox = {
    console,
    setTimeout() { return 1; },
    clearTimeout() {},
    setInterval() { return 1; },
    clearInterval() {},
    requestAnimationFrame(fn) {
      const id = nextId++;
      frames.push({ id, fn });
      return id;
    },
    cancelAnimationFrame(id) {
      const index = frames.findIndex((item) => item.id === id);
      if (index >= 0) frames.splice(index, 1);
    },
    fetch(url) {
      if (url === "/api/config") {
        return Promise.resolve({
          ok: true,
          status: 200,
          json() { return Promise.resolve({ role: "admin", username: "Mother" }); }
        });
      }
      return Promise.resolve({ ok: false, status: 404, json() { return Promise.resolve(null); } });
    },
    WebSocket: MockWebSocket,
    URL,
    URLSearchParams,
    Blob,
    btoa,
    atob,
    encodeURIComponent,
    decodeURIComponent,
    document,
    location: { protocol: "http:", host: "127.0.0.1", pathname: "/", search: "" },
    navigator: {
      mediaDevices: {
        getUserMedia() {
          return Promise.resolve({
            getTracks() { return [{ stop() {} }]; },
            getAudioTracks() { return []; }
          });
        }
      }
    },
    sessionStorage: {
      getItem(key) { return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null; },
      setItem(key, value) { store[key] = String(value); }
    }
  };
  sandbox.window = {
    AudioContext: function () {
      activeCtx = makeCtx(10);
      return activeCtx;
    },
    addEventListener() {},
    removeEventListener() {}
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "call.js" });

  return {
    document,
    async startLiveCall() {
      const consent = document.getElementById("consent-check");
      consent.checked = true;
      consent.trigger("change");
      document.getElementById("call-btn").click();
      await flushMicrotasks();
      return { ctx: activeCtx, ws: sockets[sockets.length - 1] };
    },
    flushRaf(limit) {
      let count = 0;
      while (frames.length && count < limit) {
        frames.shift().fn();
        count += 1;
      }
    }
  };
}

function snap(doc) {
  const noBtn = doc.getElementById("no-btn");
  const yesBtn = doc.getElementById("yes-btn");
  return {
    noHidden: !!noBtn.hidden,
    yesHidden: !!yesBtn.hidden,
    noVisible: noBtn.classList.contains("is-visible"),
    yesVisible: yesBtn.classList.contains("is-visible"),
    noDisabled: !!noBtn.disabled,
    yesDisabled: !!yesBtn.disabled
  };
}

function pushMsg(ws, payload) {
  ws.onmessage({ data: JSON.stringify(payload) });
}

(async function main() {
  // 1. Idle + non-yes/no live question
  const p1 = boot();
  const idle = snap(p1.document);
  const live1 = await p1.startLiveCall();
  pushMsg(live1.ws, {
    type: "session_started",
    call_id: 7,
    model: "gemini-3.1-flash-live-preview",
    voice: "Kore"
  });
  const hudModel = p1.document.getElementById("hud-model").textContent;
  pushMsg(live1.ws, {
    type: "question_state",
    step: "intro_name",
    qid: "intro_name",
    group: "intro",
    expect_yes_no: false,
    prompt_chars: 40
  });
  const nonYnMounted = snap(p1.document);
  pushMsg(live1.ws, { type: "audio", data: pcm(24000), sample_rate: 24000 });
  live1.ctx.currentTime = 12;
  live1.ctx.sources[0].onended();
  p1.flushRaf(4);
  pushMsg(live1.ws, { type: "bot_turn_complete", step: "intro_name" });
  const nonYnAfterTurn = snap(p1.document);

  // 2. Halfway reveal + #no-btn click + duplicate click ignored + #yes-btn + ending
  const p2 = boot();
  const live2 = await p2.startLiveCall();
  pushMsg(live2.ws, {
    type: "question_state",
    step: "birth_nicu",
    qid: "birth_nicu",
    group: "birth",
    expect_yes_no: true,
    prompt_chars: 120
  });
  pushMsg(live2.ws, { type: "audio", data: pcm(48000), sample_rate: 24000 });
  live2.ctx.currentTime = 14;
  p2.flushRaf(4);
  const beforeHalf = snap(p2.document);
  live2.ctx.currentTime = 15;
  p2.flushRaf(4);
  const halfway = snap(p2.document);

  // Click #no-btn while audio is still playing
  p2.document.getElementById("no-btn").click();
  // Second click before next question_state must be ignored
  p2.document.getElementById("no-btn").click();
  p2.document.getElementById("yes-btn").click();
  const sourceStoppedOnClick = live2.ctx.sources[0].stopped;
  const afterNoClick = snap(p2.document);
  const captionAfterNo = p2.document.getElementById("caption").textContent;
  const quickRepliesSent = live2.ws.sent.filter((m) => m.type === "quick_reply");

  // 3. Turn-complete reveal (mismatched step ignored, matching step reveals) + #yes-btn + ending
  const p3 = boot();
  const live3 = await p3.startLiveCall();
  pushMsg(live3.ws, {
    type: "question_state",
    step: "birth_jaundice",
    qid: "birth_jaundice",
    group: "birth",
    expect_yes_no: true,
    prompt_chars: 120
  });
  pushMsg(live3.ws, { type: "audio", data: pcm(4800), sample_rate: 24000 });
  live3.ctx.currentTime = 10.2;
  live3.ctx.sources[0].onended();
  p3.flushRaf(4);
  pushMsg(live3.ws, { type: "bot_turn_complete", step: "birth_term" });
  const otherStep = snap(p3.document);
  pushMsg(live3.ws, { type: "bot_turn_complete", step: "birth_jaundice" });
  const turnComplete = snap(p3.document);
  p3.document.getElementById("yes-btn").click();
  const captionAfterYes = p3.document.getElementById("caption").textContent;
  const yesReplySent = live3.ws.sent.filter((m) => m.type === "quick_reply");

  pushMsg(live3.ws, { type: "ending" });
  const afterEnding = Object.assign(snap(p3.document), {
    stageWorking: p3.document.getElementById("stage").classList.contains("working"),
    waitHidden: !!p3.document.getElementById("wait").hidden
  });

  process.stdout.write(JSON.stringify({
    idle,
    hudModel,
    nonYnMounted,
    nonYnAfterTurn,
    beforeHalf,
    halfway,
    otherStep,
    turnComplete,
    sourceStoppedOnClick,
    afterNoClick,
    captionAfterNo,
    quickRepliesSent,
    captionAfterYes,
    yesReplySent,
    afterEnding
  }));
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    env = {"PATH": os.environ.get("PATH", ""), "CALL_JS": str(_STATIC / "call.js")}
    done = subprocess.run(
        [_NODE, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if done.returncode != 0:
        raise AssertionError(done.stderr or done.stdout)
    return json.loads(done.stdout)


def _run_admin_js_behavior() -> dict:
    import subprocess

    harness = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.env.ADMIN_JS, "utf8");

function makeNode(tag) {
  const set = new Set();
  const attrs = {};
  const listeners = {};
  const node = {
    tagName: String(tag || "div").toUpperCase(),
    hidden: false,
    disabled: false,
    value: "",
    href: "",
    src: "",
    title: "",
    type: "",
    innerHTML: "",
    _text: "",
    children: [],
    get childNodes() { return this.children; },
    get className() { return Array.from(set).join(" "); },
    set className(val) {
      set.clear();
      String(val || "").split(/\s+/).filter(Boolean).forEach((c) => set.add(c));
    },
    get textContent() { return this._text; },
    set textContent(val) {
      this._text = val == null ? "" : String(val);
      if (this._text === "") this.children = [];
    },
    classList: {
      add(c) { set.add(c); },
      remove(c) { set.delete(c); },
      contains(c) { return set.has(c); },
      toggle(c, force) {
        const on = force === undefined ? !set.has(c) : !!force;
        if (on) set.add(c); else set.delete(c);
        return on;
      }
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k] || ""; },
    addEventListener(type, fn) {
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push(fn);
    },
    trigger(type, event) {
      const list = listeners[type] || [];
      for (const fn of list) fn(event || { target: this, preventDefault() {} });
    },
    click() { this.trigger("click"); },
    querySelectorAll() { return []; },
    closest() { return null; },
    contains() { return false; },
    remove() {}
  };
  return node;
}

const ids = {};
const document = {
  getElementById(id) {
    if (!ids[id]) ids[id] = makeNode("div");
    return ids[id];
  },
  createElement(tag) { return makeNode(tag); },
  querySelectorAll() { return []; },
  body: makeNode("body")
};

const call1 = {
  id: 1,
  caller_name: "पूजा",
  user_role: "user",
  mother_name: "पूजा",
  child_name: "अमन",
  child_age: "2 साल",
  child_gender: "female",
  clinic_code: "kem",
  model_used: "gemini-3.1-flash-live-preview",
  voice_used: "Kore",
  duration_sec: 95,
  started_at: "2026-11-15T04:00:00Z",
  status: "completed",
  red_flags: ["birth_mode_detail"],
  summary_doctor: "# Doctor Summary\nAll good.",
  summary_mother: "माँ के लिए सवाल",
  telemetry: {
    connect_ms: 120,
    avg_latency_ms: 400,
    max_latency_ms: 600,
    turn_latencies_ms: [400],
    user_turns: 4,
    bot_turns: 5,
    tool_calls: 5,
    watchdog_nudges: 1,
    audio_chunks_in: 10,
    audio_chunks_out: 12
  },
  events: [],
  transcript: [],
  cursor_state: {
    answers: {
      preg_scan: { state: "not_applicable", notes: "", child_ids: [], followups: {} },
      intro_name: { state: "clear", notes: "पूजा", child_ids: [], followups: {} },
      intro_bridge: { state: "concern", notes: "", child_ids: [], followups: {} },
      birth_term: { state: "unknown", notes: "", child_ids: [], followups: {} },
      birth_mode_detail: {
        state: "concern",
        child_ids: ["child_1"],
        notes: "ऑपरेशन",
        followups: { operation_reason: { state: "concern", note: "सांस की तकलीफ" } }
      }
    }
  }
};

const call2 = {
  id: 2,
  caller_name: "रेखा",
  user_role: "user",
  mother_name: "रेखा",
  child_age: "4 साल",
  model_used: "gemini-3.1-flash-live-preview",
  voice_used: "Kore",
  duration_sec: 45,
  status: "completed",
  red_flags: [],
  summary_doctor: "",
  summary_mother: "",
  telemetry: {},
  events: [],
  transcript: [],
  cursor_state: {
    answers: {
      preg_scan: { state: "not_applicable", notes: "", child_ids: [], followups: {} }
    }
  }
};

const fetchedUrls = [];
let copiedReport = "";

const sandbox = {
  console,
  setTimeout,
  clearTimeout,
  setInterval() { return 1; },
  clearInterval() {},
  URLSearchParams,
  encodeURIComponent,
  document,
  location: { pathname: "/admin", search: "", href: "http://127.0.0.1/admin" },
  navigator: {
    clipboard: {
      writeText(text) {
        copiedReport = text;
        return Promise.resolve();
      }
    }
  },
  window: { addEventListener() {}, removeEventListener() {}, print() {} },
  fetch(path) {
    fetchedUrls.push(path);
    if (path === "/api/calls") {
      return Promise.resolve({ ok: true, status: 200, json() { return Promise.resolve([call1, call2]); } });
    }
    if (path === "/api/calls/1") {
      return Promise.resolve({ ok: true, status: 200, json() { return Promise.resolve(call1); } });
    }
    if (path === "/api/calls/2") {
      return Promise.resolve({ ok: true, status: 200, json() { return Promise.resolve(call2); } });
    }
    if (path === "/api/bugs") {
      return Promise.resolve({ ok: true, status: 200, json() { return Promise.resolve([]); } });
    }
    return Promise.resolve({ ok: false, status: 404, json() { return Promise.resolve({}); } });
  }
};

async function flushMicrotasks() {
  for (let i = 0; i < 16; i++) await Promise.resolve();
}

(async function main() {
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "admin.js" });
  await flushMicrotasks();

  const findingsTable = document.getElementById("findings").children[0];
  const headers = findingsTable.children[0].children[0].children.map((th) => th.textContent);
  const rows = findingsTable.children[1].children.map((tr) => ({
    rowClass: tr.className,
    qid: tr.children[0].textContent,
    state: tr.children[1].textContent,
    detail: tr.children[2].textContent,
    flag: tr.children[3].textContent
  }));

  const firstListItem = document.getElementById("call-list").children[0];
  const modelBadge = firstListItem.children[3].children[0].textContent;
  const callMeta = document.getElementById("call-meta").textContent;
  const kpis = document.getElementById("telemetry-grid").children
    .map((box) => box.children.map((n) => n.textContent).join(": "))
    .join(" | ");

  document.getElementById("copy-report").click();
  await flushMicrotasks();
  const copyNotice = document.getElementById("notice").textContent;

  // Click second call to verify empty findings message
  document.getElementById("call-list").children[1].click();
  await flushMicrotasks();
  const emptyText = document.getElementById("findings").children[0].textContent;

  process.stdout.write(JSON.stringify({
    fetchedUrls,
    headers,
    rows,
    modelBadge,
    callMeta,
    kpis,
    copiedReport,
    copyNotice,
    emptyText
  }));
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    env = {"PATH": os.environ.get("PATH", ""), "ADMIN_JS": str(_STATIC / "admin.js")}
    done = subprocess.run(
        [_NODE, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if done.returncode != 0:
        raise AssertionError(done.stderr or done.stdout)
    return json.loads(done.stdout)


class QuickReplySocketTests(unittest.TestCase):
    def test_quick_reply_walks_bridge_gate_and_concern_followup(self):
        session = _ScriptedSession()

        async def fake_connect(client, model, config):
            return _ScriptCtx(session), session, 4

        async def fast_pdf(call_id):
            return b"%PDF-1.4\n", "SHISHU.pdf"

        profile = {
            "caller_name": "पूजा",
            "marital_status": "married",
            "child_count": 2,
            "children": [
                {"age_years": 6, "sex": "male", "name": "अमन"},
                {"age_years": 2, "sex": "female", "name": "रिया"},
            ],
        }
        inbox = queue.Queue()
        messages = []

        def pump(ws):
            try:
                while True:
                    inbox.put(ws.receive_json())
            except BaseException as exc:
                inbox.put(exc)

        def wait_for(predicate, timeout=12):
            import time

            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    item = inbox.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(item, BaseException):
                    raise item
                messages.append(item)
                if item.get("type") == "error":
                    self.fail(item.get("message") or item)
                if predicate(item):
                    return item
            tail = json.dumps(messages[-12:], ensure_ascii=False, default=str)
            self.fail("timed out waiting for a call message: " + tail)

        async def fake_eval(request, **_kwargs):
            ctx = json.loads(request["input"])
            active_q = ctx.get("active_question", "")
            is_which_child = "किस बच्चे" in active_q
            if is_which_child:
                raw = {
                    "danger_now": {"noul": 0.01},
                    "caller_stop": {"noul": 0.01},
                    "mentions_profile": {"noul": 0.01},
                    "current_outcome": {
                        "choice": "concern",
                        "probabilities": {"concern": 0.94, "other": 0.04, "clear": 0.02},
                    },
                    "child_ref": {
                        "choice": "child_1",
                        "probabilities": {"child_1": 0.92, "other": 0.04, "child_2": 0.03, "all": 0.01},
                    },
                    "side_intent": {"choice": "other", "probabilities": {"other": 0.96, "add_concern": 0.04}},
                    "side_qid": {"choice": "other", "probabilities": {"other": 0.96, "birth_nicu": 0.04}},
                }
            else:
                raw = {
                    "danger_now": {"noul": 0.01},
                    "caller_stop": {"noul": 0.01},
                    "mentions_profile": {"noul": 0.94},
                    "current_outcome": {
                        "choice": "other",
                        "probabilities": {"other": 0.92, "clear": 0.05, "concern": 0.03},
                    },
                }
            questions = request.get("questions") or {}
            side_criteria = (questions.get("side_qid") or {}).get("criteria") or {}
            return decisions.DecisionCallResult(
                ok=True,
                error_reason="",
                attempts=1,
                decision_ms=22,
                request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
                question_count=len(questions),
                catalog_option_count=len(side_criteria),
                usage_present=True,
                total_input_tokens=240,
                raw_answers=raw,
            )

        async def fake_extract(state, utterance_text="", audio_wav_bytes=None, **_kwargs):
            qid = state.get("active_qid")
            if qid == "intro_name":
                updates = {"caller_name": profile["caller_name"]}
            elif qid == "intro_married":
                updates = {"marital_status": profile["marital_status"]}
            elif qid == "intro_children":
                updates = {
                    "child_count": profile["child_count"],
                    "children": profile["children"],
                }
            else:
                updates = {}
            return profile_extractor.ProfileExtractionResult(
                ok=True,
                updates=updates,
                error_reason="",
                extractor_ms=15,
            )

        with patch.object(gemini_live, "_connect_live", fake_connect), patch.object(
            report_service, "mother_pdf", fast_pdf
        ), patch.object(
            decisions, "evaluate_turn", side_effect=fake_eval
        ) as eval_mock, patch.object(
            profile_extractor, "extract_profile_entities", side_effect=fake_extract
        ) as ext_mock:
            client = TestClient(app.app)
            logged = client.post(
                "/api/login",
                json={"username": "पूजा", "passcode": app._passcode()},
            )
            self.assertEqual(logged.status_code, 200)
            with client.websocket_connect("/ws/gemini") as ws:
                worker = threading.Thread(target=pump, args=(ws,), daemon=True)
                worker.start()
                try:
                    intro = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("qid") == "intro_name"
                    )
                    self.assertFalse(intro["expect_yes_no"])
                    self.assertGreater(intro["prompt_chars"], 0)
                    self.assertEqual(intro["step"], "intro_name")
                    session.push(_audio_message())
                    wait_for(lambda msg: msg.get("type") == "audio")
                    ws.send_json({
                        "type": "quick_reply",
                        "outcome": "concern",
                        "step": intro["step"],
                    })
                    ws.send_json({"type": "text", "text": "मेरा नाम पूजा है"})
                    wait_for(
                        lambda msg: msg.get("type") == "transcript"
                        and msg.get("role") == "user"
                        and "पूजा" in str(msg.get("text") or "")
                    )
                    session.push(_server_message(
                        tool_calls=[_tool("nq-1", "next_question", {})],
                        turn_complete=True,
                    ))
                    married = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("qid") == "intro_married"
                    )
                    self.assertFalse(married["expect_yes_no"])
                    session.push(_audio_message())
                    wait_for(lambda msg: msg.get("type") == "audio")
                    ws.send_json({"type": "text", "text": "हाँ, शादी हो गई है"})
                    wait_for(
                        lambda msg: msg.get("type") == "transcript"
                        and msg.get("role") == "user"
                        and "शादी" in str(msg.get("text") or "")
                    )
                    session.push(_server_message(
                        tool_calls=[_tool("nq-2", "next_question", {})],
                        turn_complete=True,
                    ))
                    children_qs = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("qid") == "intro_children"
                    )
                    self.assertFalse(children_qs["expect_yes_no"])
                    session.push(_audio_message())
                    wait_for(lambda msg: msg.get("type") == "audio")
                    ws.send_json({"type": "text", "text": "दो बच्चे हैं, 6 साल का अमन और 2 साल की रिया"})
                    wait_for(
                        lambda msg: msg.get("type") == "transcript"
                        and msg.get("role") == "user"
                        and "अमन" in str(msg.get("text") or "")
                    )
                    session.push(_server_message(
                        tool_calls=[_tool("nq-3", "next_question", {})],
                        turn_complete=True,
                    ))
                    bridge = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("qid") == "intro_bridge"
                    )
                    self.assertTrue(bridge["expect_yes_no"])
                    self.assertGreater(bridge["prompt_chars"], 0)
                    self.assertEqual(bridge["group"], "intro")
                    session.push(_audio_message())
                    wait_for(lambda msg: msg.get("type") == "audio")
                    ws.send_json({
                        "type": "quick_reply",
                        "outcome": "clear",
                        "step": bridge["step"],
                    })
                    gate = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("qid") == "birth_mode_detail"
                        and msg.get("expect_yes_no") is True
                    )
                    self.assertEqual(gate["step"], "birth_mode_detail")
                    self.assertEqual(gate["group"], "birth")
                    self.assertGreater(gate["prompt_chars"], 0)
                    session.push(_audio_message())
                    wait_for(lambda msg: msg.get("type") == "audio")
                    ws.send_json({
                        "type": "quick_reply",
                        "outcome": "concern",
                        "step": gate["step"],
                    })
                    which = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("step") == "birth_mode_detail:-1"
                    )
                    self.assertFalse(which["expect_yes_no"])
                    self.assertGreater(which["prompt_chars"], 0)
                    self.assertEqual(which["qid"], "birth_mode_detail")
                    session.push(_audio_message())
                    wait_for(lambda msg: msg.get("type") == "audio")
                    ws.send_json({"type": "text", "text": "बड़ा वाला"})
                    wait_for(
                        lambda msg: msg.get("type") == "transcript"
                        and msg.get("role") == "user"
                        and "बड़ा" in str(msg.get("text") or "")
                    )
                    session.push(_server_message(
                        tool_calls=[_tool("nq-child", "next_question", {})],
                        turn_complete=True,
                    ))
                    follow = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("step") == "birth_mode_detail:0"
                    )
                    self.assertFalse(follow["expect_yes_no"])
                    self.assertGreater(follow["prompt_chars"], 0)
                    self.assertEqual(follow["qid"], "birth_mode_detail")
                finally:
                    ws.close()

            self.assertEqual(eval_mock.await_count, 4)
            self.assertEqual(ext_mock.await_count, 3)

        states = [item for item in messages if item.get("type") == "question_state"]
        self.assertEqual(
            [(item["qid"], item["step"], item["expect_yes_no"]) for item in states],
            [
                ("intro_name", "intro_name", False),
                ("intro_married", "intro_married", False),
                ("intro_children", "intro_children", False),
                ("intro_bridge", "intro_bridge", True),
                ("birth_mode_detail", "birth_mode_detail", True),
                ("birth_mode_detail", "birth_mode_detail:-1", False),
                ("birth_mode_detail", "birth_mode_detail:0", False),
            ],
        )
        said = [
            item.get("text")
            for item in messages
            if item.get("type") == "transcript" and item.get("role") == "user"
        ]
        self.assertIn("नहीं।", said)
        self.assertIn("हाँ।", said)
        self.assertIn({"type": "clear"}, messages)
        started = next(item for item in messages if item.get("type") == "session_started")
        self.assertIn("model", started)
        self.assertNotIn("engine", started)
        saved = db.get_call(started["call_id"])["cursor_state"]
        self.assertEqual(saved["answers"]["intro_bridge"]["state"], "clear")
        self.assertEqual(saved["answers"]["birth_mode_detail"]["state"], "concern")
        self.assertEqual(saved["answers"]["birth_mode_detail"]["child_ids"], ["child_1"])
        self.assertEqual(saved["pending_followup_idx"], 0)
        self.assertEqual(saved["active_qid"], "birth_mode_detail")
        for item in states:
            self.assertGreater(item["prompt_chars"], 0)

    def test_prompt_chars_and_yes_no_follow_the_step_kind(self):
        def payload(state):
            body = gemini_live._question_state_payload(state)
            self.assertGreater(body["prompt_chars"], 0)
            return body

        state = new_cursor_state()
        intro = payload(state)
        self.assertEqual(intro["qid"], "intro_name")
        self.assertFalse(intro["expect_yes_no"])

        state, hold = apply_profile(state, caller_name="रेखा")
        self.assertIsNone(hold)
        married = payload(state)
        self.assertEqual(married["qid"], "intro_married")
        self.assertFalse(married["expect_yes_no"])

        state, hold = apply_profile(state, marital_status="married")
        self.assertIsNone(hold)
        children = payload(state)
        self.assertEqual(children["qid"], "intro_children")
        self.assertFalse(children["expect_yes_no"])

        kids = [cursor_tests._child(8, "male", 1), cursor_tests._child(2, "female", 2)]
        state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "birth_mode_detail")
        gate = payload(state)
        self.assertTrue(gate["expect_yes_no"])
        state = apply_turn(state, "concern")
        which = payload(state)
        self.assertEqual(which["step"], "birth_mode_detail:-1")
        self.assertFalse(which["expect_yes_no"])
        state = apply_turn(state, "concern", child_ref="child_1")
        follow = payload(state)
        self.assertEqual(follow["step"], "birth_mode_detail:0")
        self.assertFalse(follow["expect_yes_no"])
        self.assertEqual(follow["qid"], "birth_mode_detail")


if __name__ == "__main__":
    unittest.main()
