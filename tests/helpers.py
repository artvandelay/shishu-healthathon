"""Shared test builders, fake live transports, and Node VM runners for SHISHU tests."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from tests import APP_DIR

import call_cursor
import decisions
import gemini_live
from call_cursor import apply_profile, apply_turn, new_cursor_state

_STATIC = APP_DIR / "static"
_NODE = shutil.which("node")


def make_child(age: float, sex: str = "unknown", n: int = 1, name: str | None = None) -> dict:
    return {
        "id": f"child_{n}",
        "age_years": age,
        "age_label_hi": f"{int(round(age * 12))} महीने" if age < 1 else f"{age:g} साल",
        "sex": sex,
        "name": name,
    }


def make_profile_state(
    marital: str = "married",
    children: list | None = None,
    ever_pregnant: bool | None = None,
    name: str = "पूजा",
) -> dict:
    state, hold = apply_profile(new_cursor_state(), caller_name=name)
    assert hold is None, hold
    state, hold = apply_profile(state, marital_status=marital)
    assert hold is None, hold
    if marital == "married" or children:
        state, hold = apply_profile(state, children=list(children or []))
        assert hold is None, hold
    if ever_pregnant is not None and state["profile"]["ever_pregnant"] is not True:
        state, hold = apply_profile(state, ever_pregnant=ever_pregnant)
        assert hold is None, hold
    return state


def clear_until(state: dict, qid: str, limit: int = 90) -> dict:
    for _ in range(limit):
        if state["phase"] != "asking":
            return state
        if state["active_qid"] == qid and state["pending_followup_idx"] is None:
            return state
        if state["active_qid"] == "ever_pregnant":
            state = apply_turn(state, "clear")
            continue
        prev = (state["active_qid"], state["pending_followup_idx"])
        state = apply_turn(state, "clear")
        if (state["active_qid"], state["pending_followup_idx"]) == prev and state["phase"] == "asking":
            raise AssertionError(f"cursor stuck at {prev}")
    raise AssertionError(f"never reached {qid}; still at {state['active_qid']}")


def noul(prob: float) -> dict:
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1.0 - prob)}


def choice(winner: str, probs: dict[str, float]) -> dict:
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": probs[winner]}


def test_thresholds() -> decisions.DecisionThresholds:
    return decisions.DecisionThresholds(
        danger_now_min_yes=0.80,
        caller_stop_min_yes=0.80,
        mentions_profile_min_yes=0.65,
        mentions_diagnosis_min_yes=0.60,
        detail_present_min_yes=0.65,
        outcome_min_confidence=0.65,
        outcome_min_margin=0.15,
        side_intent_min_confidence=0.65,
        side_intent_min_margin=0.15,
        side_qid_min_confidence=0.70,
        side_qid_min_margin=0.20,
        child_ref_min_confidence=0.65,
        child_ref_min_margin=0.15,
    )


def shipped_thresholds() -> decisions.DecisionThresholds:
    return decisions.load_thresholds(APP_DIR / "decision_thresholds.json")


def parsed_input(req: dict) -> dict:
    raw = req["input"][0]["text"] if isinstance(req["input"], list) else req["input"]
    return json.loads(raw)


def decision_result(
    request: dict,
    raw_answers: dict,
    *,
    ok: bool = True,
    error_reason: str = "",
    attempts: int = 1,
    decision_ms: int = 42,
    total_input_tokens: int = 380,
) -> decisions.DecisionCallResult:
    questions = request.get("questions") or {}
    side_criteria = (questions.get("side_qid") or {}).get("criteria") or {}
    return decisions.DecisionCallResult(
        ok=ok,
        error_reason=error_reason,
        attempts=attempts,
        decision_ms=decision_ms,
        request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
        question_count=len(questions),
        catalog_option_count=len(side_criteria),
        usage_present=ok,
        total_input_tokens=total_input_tokens if ok else None,
        raw_answers=raw_answers if ok else {},
    )


class FakeWebSocket:
    def __init__(self, messages: list | None = None):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent: list[dict] = []
        self._messages = list(messages or [])
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True
        self.client_state = gemini_live.WebSocketState.CONNECTED

    async def close(self):
        self.closed = True
        self.client_state = gemini_live.WebSocketState.DISCONNECTED

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive_text(self):
        if not self._messages:
            raise gemini_live.WebSocketDisconnect()
        return json.dumps(self._messages.pop(0), ensure_ascii=False)


class QueueWebSocket(FakeWebSocket):
    def __init__(self):
        super().__init__([])
        self.queue: asyncio.Queue = asyncio.Queue()

    async def push(self, item):
        await self.queue.put(item)

    async def receive_text(self):
        item = await self.queue.get()
        if item is None:
            raise gemini_live.WebSocketDisconnect()
        return json.dumps(item, ensure_ascii=False)


class FakeLiveSession:
    """Flexible Gemini Live session mock supporting scripted turns, queued pushes, and failure modes."""

    def __init__(
        self,
        messages: list | None = None,
        *,
        fail_tool: bool = False,
        fail_realtime: bool = False,
        error: BaseException | None = None,
        wait_forever: bool = False,
        sleep_per_msg: float = 0.0,
        on_realtime=None,
    ):
        self.setup_complete = True
        self.sent: list[tuple[str, object]] = []
        self.fail_tool = fail_tool
        self.fail_realtime = fail_realtime
        self.error = error
        self.wait_forever = wait_forever
        self.sleep_per_msg = sleep_per_msg
        self.on_realtime = on_realtime
        self._lock = threading.Lock()
        self._pending = list(messages or [])
        self._loop = None
        self._queue: asyncio.Queue | None = None

    def push(self, item):
        with self._lock:
            loop = self._loop
            mailbox = self._queue
            if loop is None or mailbox is None:
                self._pending.append(item)
                return
        loop.call_soon_threadsafe(mailbox.put_nowait, item)

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses))
        if self.fail_tool:
            raise RuntimeError("tool response failed")

    async def send_realtime_input(self, **kwargs):
        if self.on_realtime is not None:
            self.on_realtime(self, kwargs)
        if self.fail_realtime:
            raise RuntimeError("realtime send failed")
        self.sent.append(("realtime", kwargs))

    async def receive(self):
        if self.error is not None:
            raise self.error
        with self._lock:
            if self._queue is None:
                self._loop = asyncio.get_running_loop()
                self._queue = asyncio.Queue()
                pending = list(self._pending)
                self._pending.clear()
            else:
                pending = []
        for item in pending:
            if self.sleep_per_msg:
                await asyncio.sleep(self.sleep_per_msg)
            if item is None:
                return
            yield item
        if not self.wait_forever:
            return
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


class SessionCtx:
    def __init__(self, session, *, close_on_exit: bool = False):
        self.session = session
        self.close_on_exit = close_on_exit

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_args):
        if self.close_on_exit and hasattr(self.session, "push"):
            self.session.push(None)
        return False


class FakeToolCall:
    def __init__(self, name: str = "next_question", args: dict | None = None, fid: str = "fc-1", *, will_continue: bool = False):
        self.id = fid
        self.name = name
        self.args = args or {}
        self.will_continue = will_continue


class DummyRecorder:
    def add_user_pcm(self, _pcm, _rate):
        return None

    def add_bot_pcm(self, _pcm, _rate):
        return None


async def drop_fallback(turn) -> None:
    task = turn.speak_fallback
    turn.speak_fallback = None
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def server_message(
    *,
    turn_complete: bool = False,
    interrupted: bool = False,
    user_text: str = "",
    user_finished: bool = False,
    bot_text: str = "",
    parts: list | None = None,
    tool_calls: list | None = None,
    go_away: bool = False,
    resume_handle: str = "",
):
    tool_call = SimpleNamespace(function_calls=tool_calls) if tool_calls else None
    input_tr = SimpleNamespace(text=user_text, finished=user_finished) if user_text or user_finished else None
    output_tr = SimpleNamespace(text=bot_text) if bot_text else None
    model_turn = SimpleNamespace(parts=parts) if parts else None
    resumption = SimpleNamespace(resumable=True, new_handle=resume_handle) if resume_handle else None
    return SimpleNamespace(
        tool_call_cancellation=None,
        server_content=SimpleNamespace(
            interrupted=interrupted,
            input_transcription=input_tr,
            output_transcription=output_tr,
            model_turn=model_turn,
            turn_complete=turn_complete,
        ),
        tool_call=tool_call,
        session_resumption_update=resumption,
        go_away=SimpleNamespace(time_left="50s") if go_away else None,
    )


def bot_audio_message(pcm: bytes = b"\x01\x00" * 240, *, turn_complete: bool = False):
    part = SimpleNamespace(
        thought=False,
        function_call=None,
        inline_data=SimpleNamespace(data=pcm, mime_type="audio/pcm;rate=24000"),
    )
    return server_message(parts=[part], turn_complete=turn_complete)


_NODE_DOM_PREAMBLE = r"""
const fs = require("fs");
const vm = require("vm");

function makeNode(tag) {
  const set = new Set();
  const attrs = {};
  const listeners = {};
  return {
    tagName: String(tag || "div").toUpperCase(),
    hidden: false,
    disabled: false,
    value: "",
    checked: false,
    href: "",
    src: "",
    title: "",
    type: "",
    innerHTML: "",
    style: {},
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
    querySelectorAll() { return []; },
    closest() { return null; },
    contains() { return false; },
    replaceChildren() { this.children = []; },
    remove() {}
  };
}

async function flushMicrotasks(n = 32) {
  for (let i = 0; i < n; i++) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}
"""

_CALL_VM_HARNESS = _NODE_DOM_PREAMBLE + r"""
const callSource = fs.readFileSync(process.env.JS_PATH, "utf8");

function makeCtx(now) {
  const ctx = {
    currentTime: now,
    state: "running",
    closed: false,
    destination: {},
    sampleRate: 48000,
    sources: [],
    resume() { return Promise.resolve(); },
    close() { this.closed = true; this.state = "closed"; return Promise.resolve(); },
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; },
    createScriptProcessor() { return { onaudioprocess: null, connect() {}, disconnect() {} }; },
    createGain() { return { gain: { value: 0 }, connect() {}, disconnect() {} }; },
    createBuffer(_ch, len, rate) { return { duration: len / rate, getChannelData() { return new Float32Array(len); } }; },
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

function pcm(frames) {
  const bytes = Buffer.alloc(frames * 2);
  for (let i = 0; i < frames; i++) bytes.writeInt16LE(800, i * 2);
  return bytes.toString("base64");
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

function boot(options) {
  const opts = options || {};
  const frames = [];
  const ids = {};
  const sockets = [];
  const ctxs = [];
  const timeouts = [];
  const wakeLockKinds = [];
  const store = {};
  let nextId = 1, nextTimer = 1, clearedIntervals = 0, micCalls = 0;
  let activeCtx = null, retryPostOk = false, customFetch = null;
  let bugPostedBody = null, pendingImage = null, clinicRemoved = false;
  const initiallyHidden = new Set([
    "no-btn", "yes-btn", "wait", "nudge-btn", "thanks", "hud",
    "hud-toggle", "admin-link", "admin-view", "error", "bug-overlay",
    "share-pdf", "retry-pdf"
  ]);
  const document = {
    getElementById(id) {
      if (!ids[id]) {
        ids[id] = makeNode("div");
        if (initiallyHidden.has(id)) ids[id].hidden = true;
      }
      return ids[id];
    },
    addEventListener() {},
    createElement(tag) {
      const node = makeNode(tag);
      if (String(tag || "").toLowerCase() === "canvas") {
        node.getContext = () => ({ drawImage() {} });
        node.toDataURL = () => "data:image/jpeg;base64,SHRUNK";
      }
      return node;
    },
    querySelector() { return null; },
    body: makeNode("body"),
    visibilityState: "visible"
  };

  function MockWebSocket(url) {
    this.url = url;
    this.readyState = 1;
    this.bufferedAmount = 0;
    this.sent = [];
    this.closed = false;
    this.onopen = this.onmessage = this.onerror = this.onclose = null;
    sockets.push(this);
    queueMicrotask(() => { if (this.onopen) this.onopen(); });
  }
  MockWebSocket.OPEN = 1;
  MockWebSocket.prototype.send = function (raw) {
    if (opts.throwOnHangup && raw.indexOf("hangup") !== -1) throw new Error("send failed");
    this.sent.push(JSON.parse(raw));
  };
  MockWebSocket.prototype.close = function () { this.closed = true; };

  const loc = { protocol: "http:", host: "127.0.0.1", pathname: "/", search: opts.search || "", href: "http://127.0.0.1/" };
  function pdfOkResponse() {
    return {
      ok: true, status: 200,
      headers: { get() { return 'attachment; filename="SHISHU.pdf"'; } },
      blob() { return Promise.resolve(new Blob(["%PDF-1.4"])); }
    };
  }
  function MockImage() {
    this.width = 1800; this.height = 900; this.onload = this.onerror = null;
    pendingImage = this;
  }

  const sandbox = {
    console,
    setTimeout(fn, ms) { const id = nextTimer++; timeouts.push({ id, fn, ms, cleared: false }); return id; },
    clearTimeout(id) { const t = timeouts.find((x) => x.id === id); if (t) t.cleared = true; },
    setInterval() { return nextTimer++; },
    clearInterval() { clearedIntervals++; },
    requestAnimationFrame(fn) { const id = nextId++; frames.push({ id, fn }); return id; },
    cancelAnimationFrame(id) { const idx = frames.findIndex((x) => x.id === id); if (idx >= 0) frames.splice(idx, 1); },
    fetch(url, init) {
      if (customFetch) {
        const res = customFetch(url, init);
        if (res) return res;
      }
      if (url === "/api/config") {
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ role: "admin", username: "Ada" }) });
      }
      if (url === "/api/bugs" && init && init.method === "POST") {
        bugPostedBody = JSON.parse(init.body);
        return Promise.resolve({ ok: false, status: 413, json: () => Promise.resolve({ detail: "Bug report is too large" }) });
      }
      if (url === "/api/logout") return Promise.reject(new Error("offline"));
      if (url.indexOf("/mother.pdf") !== -1 && init && init.method === "POST") {
        return retryPostOk ? Promise.resolve(pdfOkResponse()) : Promise.resolve({ ok: false, status: 500 });
      }
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve(null) });
    },
    WebSocket: MockWebSocket,
    Image: MockImage,
    File: function FileImpl(parts, name, cfg) { this.parts = parts; this.name = name; this.type = cfg && cfg.type; },
    URL: { createObjectURL() { return "blob:mock"; }, revokeObjectURL() {} },
    URLSearchParams, Blob, btoa, atob, encodeURIComponent, decodeURIComponent,
    document,
    location: loc,
    navigator: {
      wakeLock: {
        request(kind) {
          wakeLockKinds.push(kind);
          return Promise.resolve({ addEventListener() {}, release() { return Promise.resolve(); } });
        }
      },
      mediaDevices: {
        getUserMedia() {
          micCalls++;
          if (opts.customMic) return opts.customMic(micCalls);
          return Promise.resolve({ getTracks: () => [{ stop() {} }], getAudioTracks: () => [] });
        }
      }
    },
    sessionStorage: {
      getItem(k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
      setItem(k, v) { store[k] = String(v); },
      removeItem(k) { clinicRemoved = k === "shishu_clinic"; delete store[k]; }
    }
  };
  sandbox.window = {
    AudioContext: function () { activeCtx = makeCtx(10); ctxs.push(activeCtx); return activeCtx; },
    addEventListener() {},
    removeEventListener() {}
  };
  vm.createContext(sandbox);
  vm.runInContext(callSource, sandbox, { filename: "call.js" });

  return {
    document, sockets, ctxs, timeouts, location: loc, pdfOkResponse, wakeLockKinds, store,
    getClearedIntervals: () => clearedIntervals,
    setRetryPostOk: (v) => { retryPostOk = v; },
    setCustomFetch: (fn) => { customFetch = fn; },
    getBugPostedBody: () => bugPostedBody,
    getPendingImage: () => pendingImage,
    wasClinicRemoved: () => clinicRemoved,
    start() {
      const consent = document.getElementById("consent-check");
      consent.checked = true;
      consent.trigger("change");
      document.getElementById("call-btn").click();
    },
    async startLiveCall() {
      this.start();
      await flushMicrotasks();
      return { ctx: activeCtx, ws: sockets[sockets.length - 1] };
    },
    flushRaf(limit) {
      let count = 0;
      while (frames.length && count < limit) { frames.shift().fn(); count += 1; }
    }
  };
}
"""

_ADMIN_VM_HARNESS = _NODE_DOM_PREAMBLE + r"""
const adminSource = fs.readFileSync(process.env.JS_PATH, "utf8");

function bootAdmin(options) {
  const opts = options || {};
  const ids = {};
  const fetchedUrls = [];
  let copiedText = "";
  const document = {
    activeElement: null,
    getElementById(id) {
      if (!ids[id]) ids[id] = makeNode("div");
      return ids[id];
    },
    createElement(tag) { return makeNode(tag); },
    querySelectorAll(sel) { return opts.querySelectorAll ? opts.querySelectorAll(sel, document) : []; },
    body: makeNode("body")
  };
  if (opts.setupDoc) opts.setupDoc(document);

  const loc = { pathname: "/admin", search: "", href: "http://127.0.0.1/admin" };
  const sandbox = {
    console, setTimeout, clearTimeout,
    setInterval() { return 1; },
    clearInterval() {},
    URLSearchParams,
    URL: { createObjectURL() { return "blob:1"; }, revokeObjectURL() {} },
    encodeURIComponent,
    document,
    location: loc,
    navigator: {
      clipboard: {
        writeText(text) { copiedText = text; return Promise.resolve(); }
      }
    },
    window: { addEventListener() {}, removeEventListener() {}, print() {} },
    fetch(path, init) {
      fetchedUrls.push(path);
      return opts.fetch(path, init);
    }
  };
  vm.createContext(sandbox);
  vm.runInContext(adminSource, sandbox, { filename: "admin.js" });
  return {
    document,
    fetchedUrls,
    location: loc,
    getCopiedText: () => copiedText
  };
}
"""


def _run_node_vm(preamble: str, js_path: Path, script_body: str) -> dict:
    if _NODE is None:
        raise RuntimeError("node is not installed")
    full = preamble + "\n(async function main() {\n" + script_body + "\n})().catch((err) => { console.error(err); process.exit(1); });\n"
    done = subprocess.run(
        [_NODE, "-e", full],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "JS_PATH": str(js_path)},
    )
    if done.returncode != 0:
        raise AssertionError(done.stderr or done.stdout)
    return json.loads(done.stdout)


def run_call_vm(script_body: str) -> dict:
    return _run_node_vm(_CALL_VM_HARNESS, _STATIC / "call.js", script_body)


def run_admin_vm(script_body: str) -> dict:
    return _run_node_vm(_ADMIN_VM_HARNESS, _STATIC / "admin.js", script_body)
