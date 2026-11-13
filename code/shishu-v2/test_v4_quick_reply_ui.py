"""Stage 3: same-size flanking replies, admin cursor answers, and the live tap path."""

import asyncio
import json
import os
import queue
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="shishu-v4-ui-")
os.environ["DATA_DIR"] = _TMP
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import app  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import report_service  # noqa: E402
import test_v4_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_profile, apply_turn, new_cursor_state  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_NODE = shutil.which("node")


def _function_body(source: str, name: str) -> str:
    start = source.index("function " + name)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unclosed function {name}")


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
        html = (_STATIC / "call.html").read_text(encoding="utf-8")
        dock_at = html.index('id="call-dock"')
        no_at = html.index('id="no-btn"')
        call_at = html.index('id="call-btn"')
        yes_at = html.index('id="yes-btn"')
        self.assertLess(dock_at, no_at)
        self.assertLess(no_at, call_at)
        self.assertLess(call_at, yes_at)
        self.assertEqual(html.count('id="call-btn"'), 1)
        self.assertIn('aria-label="नहीं, आगे बढ़ें (No)"', html)
        self.assertIn('aria-label="हाँ (Yes)"', html)
        self.assertIn("prefers-reduced-motion: reduce", html)
        motion = html.split("@media (prefers-reduced-motion: reduce)", 1)[1].split("}", 1)[0]
        self.assertIn(".ring", motion)
        self.assertIn("#wait .dots i", motion)
        self.assertIn("animation: none", motion)
        self.assertIn(".stage.working #call-dock", html)
        self.assertIn(".stage.working #call-btn", html)

        size = html.split(".call-btn,\n    #no-btn,\n    #call-btn,\n    #yes-btn {", 1)[1].split("}", 1)[0]
        self.assertEqual(size.count("clamp(68px, 21vw, 84px)"), 2)
        self.assertIn("flex: 0 0 auto", size)
        dock = html.split(".call-dock {", 1)[1].split("}", 1)[0]
        self.assertIn("display: flex", dock)
        self.assertIn("justify-content: center", dock)
        self.assertIn("gap: clamp(12px, 4vw, 20px)", dock)
        self.assertIn("margin-top: 14px", dock)
        button = html.split(".call-btn {", 1)[1].split("}", 1)[0]
        self.assertNotIn("margin-top", button)
        self.assertIn("background: #ffffff", html)
        self.assertIn("border: 2px solid var(--navy)", html)
        self.assertIn(".no-btn .en { color: var(--muted); }", html)
        self.assertIn(".no-btn:hover:not(:disabled) { background: var(--navy-soft); }", html)
        self.assertIn(".yes-btn .en { color: rgba(255, 255, 255, 0.88); }", html)
        self.assertIn(".yes-btn:hover:not(:disabled) { background: var(--navy-dark); }", html)
        self.assertIn("opacity: 0;", html)
        self.assertIn("transform: scale(0.9);", html)
        self.assertIn("visibility: hidden;", html)
        self.assertIn("pointer-events: none;", html)
        self.assertIn(
            "transition: opacity 240ms ease-out, transform 240ms ease-out, visibility 0s linear 240ms;",
            html,
        )
        self.assertIn(".quick-btn.is-visible {", html)
        self.assertIn("transform: scale(1);", html)
        self.assertIn("visibility: visible;", html)
        self.assertIn("pointer-events: auto;", html)
        self.assertIn(
            "transition: opacity 240ms ease-out, transform 240ms ease-out, visibility 0s;",
            html,
        )
        self.assertIn(".quick-btn.is-visible:disabled { opacity: 0.45; cursor: default; }", html)
        self.assertIn(".quick-btn", motion)
        self.assertIn("transition: none", motion)

        client = TestClient(app.app)
        client.post("/api/login", json={"username": "Mother", "passcode": app._passcode()})
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        served = page.text
        self.assertLess(served.index('id="no-btn"'), served.index('id="call-btn"'))
        self.assertLess(served.index('id="call-btn"'), served.index('id="yes-btn"'))
        self.assertIn("clamp(68px, 21vw, 84px)", served)
        self.assertIn(".stage.working #call-dock", served)
        self.assertIn("prefers-reduced-motion", served)

    def test_call_js_sends_quick_reply_and_admin_reads_cursor_answers(self):
        client = TestClient(app.app)
        call_js = client.get("/static/call.js")
        admin_js = client.get("/static/admin.js")
        self.assertEqual(call_js.status_code, 200)
        self.assertEqual(admin_js.status_code, 200)
        script = call_js.text
        dashboard = admin_js.text

        sync = _function_body(script, "syncQuickButtons")
        self.assertIn('const live = phase === "live";', sync)
        self.assertIn("noBtn.hidden = !live;", sync)
        self.assertIn("yesBtn.hidden = !live;", sync)
        self.assertIn("const show = live && expectYesNo && quickReplyReady;", sync)
        self.assertIn('noBtn.classList.toggle("is-visible", show);', sync)
        self.assertIn('yesBtn.classList.toggle("is-visible", show);', sync)
        self.assertIn("noBtn.disabled = !show || quickReplyPending;", sync)
        self.assertIn("yesBtn.disabled = !show || quickReplyPending;", sync)

        state = script.split('msg.type === "question_state"', 1)[1].split('msg.type === "ending"', 1)[0]
        self.assertIn('questionStep = String(msg.step || "");', state)
        self.assertIn("expectYesNo = Boolean(msg.expect_yes_no);", state)
        self.assertIn("promptChars = Number(msg.prompt_chars) || 0;", state)
        self.assertIn("quickReplyReady = false;", state)
        self.assertIn("quickReplyPending = false;", state)
        self.assertIn("stepAudioStart = 0;", state)
        self.assertIn("stepQueuedSec = 0;", state)
        self.assertIn("stepStreamDone = false;", state)
        self.assertIn("cancelQuickRevealRaf();", state)
        self.assertIn("flowGroup = String(msg.group);", state)
        self.assertIn("renderHud();", state)
        self.assertIn("syncQuickButtons();", state)
        self.assertNotIn("noBtn.hidden = false", state)
        self.assertNotIn("yesBtn.hidden = false", state)

        reset = _function_body(script, "resetQuickState")
        self.assertIn('questionStep = "";', reset)
        self.assertIn("expectYesNo = false;", reset)
        self.assertIn("quickReplyReady = false;", reset)
        self.assertIn("quickReplyPending = false;", reset)
        self.assertIn("stepAudioStart = 0;", reset)
        self.assertIn("stepQueuedSec = 0;", reset)
        self.assertIn("stepStreamDone = false;", reset)
        self.assertIn("cancelQuickRevealRaf();", reset)
        self.assertIn("syncQuickButtons();", reset)
        stripped = [line.strip() for line in script.splitlines()]
        self.assertEqual(stripped.count('questionStep = "";'), 1)
        self.assertEqual(stripped.count("expectYesNo = false;"), 1)
        self.assertEqual(stripped.count("quickReplyPending = false;"), 2)
        self.assertEqual(stripped.count("stepAudioStart = 0;"), 2)
        self.assertEqual(stripped.count("stepQueuedSec = 0;"), 2)
        self.assertEqual(stripped.count("stepStreamDone = false;"), 2)
        self.assertNotIn("stepFirstPlayTime", script)
        self.assertNotIn("quickRevealTimer", script)
        self.assertNotIn("clearQuickRevealTimer", script)
        self.assertIn('msg.type === "bot_turn_complete"', script)

        play = _function_body(script, "playPcm")
        self.assertIn("const startAt = Math.max(now, nextPlayTime);", play)
        self.assertIn("stepQueuedSec += audioBuffer.duration;", play)
        self.assertIn("stepAudioStart === 0 && stepQueuedSec >= 0.05", play)
        self.assertIn("scheduleQuickRevealCheck();", play)
        self.assertIn("checkQuickReveal();", play)
        self.assertNotIn("halfwayMs", play)
        self.assertNotIn("quickReplyReady = true", play)
        self.assertLess(play.index("stepQueuedSec += audioBuffer.duration;"), play.index("src.start(startAt)"))

        reveal = _function_body(script, "checkQuickReveal")
        self.assertIn("playedSec >= expectedSec * 0.5", reveal)
        self.assertIn("playedSec >= 0.65", reveal)
        self.assertIn("stepStreamDone && !isBotPlaying()", reveal)
        self.assertIn("promptChars / 12", reveal)
        self.assertIn("quickReplyReady = true", reveal)
        self.assertIn("cancelQuickRevealRaf();", reveal)

        reply = _function_body(script, "sendQuickReply")
        send_at = reply.index("ws.send")
        self.assertLess(reply.index("flushPlayback()"), send_at)
        self.assertLess(reply.index("startThinkingWatch()"), send_at)
        self.assertIn('showCaption("user", outcome === "clear" ? "नहीं।" : "हाँ।")', reply)
        self.assertIn('{ type: "quick_reply", outcome: outcome, step: questionStep }', reply)
        self.assertIn('phase !== "live"', reply)
        self.assertIn("!expectYesNo", reply)
        self.assertIn("!quickReplyReady", reply)
        self.assertIn("quickReplyPending", reply)
        self.assertIn("!questionStep", reply)
        self.assertIn("WebSocket.OPEN", reply)
        self.assertIn('sendQuickReply("clear")', script)
        self.assertIn('sendQuickReply("concern")', script)

        for name in (
            "leaveLiveChrome",
            "onSocketClose",
            "beginPreparing",
            "endCall",
            "showThanks",
            "resetForNewCall",
            "startCall",
        ):
            body = _function_body(script, name)
            self.assertIn("resetQuickState();", body, name)
            self.assertNotIn('questionStep = "";', body, name)
            self.assertNotIn("expectYesNo = false;", body, name)

        self.assertIn("renderFindings(call.findings, call.cursor_state)", dashboard)
        self.assertIn("function renderFindings(findings, cursorState)", dashboard)
        self.assertIn("const answers = asObject(asObject(cursorState).answers);", dashboard)
        self.assertIn("not_applicable", dashboard)
        self.assertIn("Question ID", dashboard)
        self.assertIn('qid !== "intro_bridge"', dashboard)
        self.assertIn('"Red flag"', dashboard)
        self.assertIn("function renderCursorAnswers", dashboard)

        if _NODE is None:
            self.skipTest("node is not installed")
        rendered = _render_findings(dashboard)
        concern = rendered["cursor"][0]
        self.assertEqual(concern["qid"], "birth_mode_detail")
        self.assertEqual(concern["state"], "concern")
        self.assertEqual(concern["flag"], "Yes")
        self.assertIn("child_1", concern["detail"])
        self.assertIn("ऑपरेशन", concern["detail"])
        self.assertIn("operation_reason: सांस की तकलीफ", concern["detail"])
        self.assertEqual([row["state"] for row in rendered["cursor"]], ["concern", "concern", "unknown", "clear"])
        bridge = next(row for row in rendered["cursor"] if row["qid"] == "intro_bridge")
        self.assertEqual(bridge["state"], "concern")
        self.assertEqual(bridge["flag"], "")
        self.assertNotIn("preg_scan", [row["qid"] for row in rendered["cursor"]])
        self.assertEqual(rendered["legacy"][0]["domain"], "Chief Complaint")
        self.assertEqual(rendered["legacy"][0]["flag"], "Yes")
        self.assertNotIn("birth_nicu", [row["domain"] for row in rendered["legacy"]])
        self.assertEqual(rendered["empty"], "No structured findings were recorded.")

    def test_speaker_clock_hides_buttons_until_halfway_or_turn_complete(self):
        if _NODE is None:
            self.skipTest("node is not installed")
        clock = _speaker_clock()
        idle = clock["idle"]
        self.assertTrue(idle["noHidden"])
        self.assertTrue(idle["yesHidden"])
        self.assertFalse(idle["noVisible"])
        self.assertFalse(idle["yesVisible"])

        mounted = clock["mounted"]
        self.assertFalse(mounted["noHidden"])
        self.assertFalse(mounted["yesHidden"])
        self.assertFalse(mounted["noVisible"])
        self.assertFalse(mounted["yesVisible"])
        self.assertTrue(mounted["noDisabled"])
        self.assertTrue(mounted["yesDisabled"])

        header = clock["header"]
        self.assertFalse(header["noHidden"])
        self.assertFalse(header["yesHidden"])
        self.assertFalse(header["noVisible"])
        self.assertFalse(header["yesVisible"])

        before = clock["beforeHalf"]
        self.assertFalse(before["noVisible"])
        self.assertFalse(before["yesVisible"])
        halfway = clock["halfway"]
        self.assertTrue(halfway["noVisible"])
        self.assertTrue(halfway["yesVisible"])
        self.assertFalse(halfway["noDisabled"])
        self.assertFalse(halfway["yesDisabled"])
        self.assertFalse(halfway["noHidden"])
        self.assertFalse(halfway["yesHidden"])

        ended = clock["ended"]
        self.assertFalse(ended["noVisible"])
        self.assertFalse(ended["yesVisible"])
        other = clock["otherStep"]
        self.assertFalse(other["noVisible"])
        self.assertFalse(other["yesVisible"])
        finished = clock["turnComplete"]
        self.assertTrue(finished["noVisible"])
        self.assertTrue(finished["yesVisible"])
        self.assertFalse(finished["noDisabled"])
        self.assertFalse(finished["yesDisabled"])


def _speaker_clock() -> dict:
    import subprocess

    harness = r"""
const fs = require("fs");
const vm = require("vm");
const raw = fs.readFileSync(process.env.CALL_JS, "utf8");
const marker = "    })();";
const at = raw.lastIndexOf(marker);
if (at < 0) throw new Error("call.js wrapper missing");
const source = raw.slice(0, at) + `
      globalThis.__shishuCall = {
        playPcm: playPcm,
        onSocketMessage: onSocketMessage,
        arm: function (ctx) {
          phase = "live";
          audioCtx = ctx;
          syncQuickButtons();
        }
      };
` + raw.slice(at);

function el() {
  const set = new Set();
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
    addEventListener() {},
    setAttribute() {},
    getAttribute() { return ""; },
    removeAttribute() {},
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
    createBuffer(_channels, length, rate) {
      return {
        duration: length / rate,
        getChannelData() { return { set() {} }; }
      };
    },
    createBufferSource() {
      const src = {
        buffer: null,
        connect() {},
        start() {},
        stop() {},
        disconnect() {},
        onended: null
      };
      ctx.sources.push(src);
      return src;
    }
  };
  return ctx;
}

function boot() {
  const frames = [];
  let nextId = 1;
  const ids = {};
  const document = {
    getElementById(id) {
      if (!ids[id]) ids[id] = el();
      return ids[id];
    },
    addEventListener() {},
    createElement() { return el(); },
    body: el(),
    visibilityState: "visible"
  };
  const store = {};
  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    requestAnimationFrame(fn) {
      const id = nextId++;
      frames.push({ id, fn });
      return id;
    },
    cancelAnimationFrame(id) {
      const index = frames.findIndex((item) => item.id === id);
      if (index >= 0) frames.splice(index, 1);
    },
    fetch() {
      return Promise.resolve({
        ok: false,
        status: 503,
        json() { return Promise.resolve(null); }
      });
    },
    WebSocket: function () {},
    URL,
    URLSearchParams,
    Blob,
    btoa,
    atob,
    encodeURIComponent,
    decodeURIComponent,
    Math,
    JSON,
    Date,
    Object,
    Array,
    Number,
    String,
    Boolean,
    Error,
    RegExp,
    Promise,
    Map,
    Set,
    isNaN,
    isFinite,
    parseInt,
    parseFloat,
    document,
    location: { protocol: "http:", host: "127.0.0.1", pathname: "/", search: "" },
    navigator: {},
    sessionStorage: {
      getItem(key) { return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null; },
      setItem(key, value) { store[key] = String(value); }
    }
  };
  sandbox.WebSocket.OPEN = 1;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "call.js" });
  return {
    api: sandbox.__shishuCall,
    document,
    flush(limit) {
      let count = 0;
      while (frames.length && count < limit) {
        frames.shift().fn();
        count += 1;
      }
      return count;
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

function question(page) {
  page.api.onSocketMessage({
    data: JSON.stringify({
      type: "question_state",
      step: "birth_nicu",
      qid: "birth_nicu",
      group: "birth",
      expect_yes_no: true,
      prompt_chars: 120
    })
  });
}

const headerPage = boot();
question(headerPage);
const idle = snap(headerPage.document);
const headerCtx = makeCtx(10);
headerPage.api.arm(headerCtx);
question(headerPage);
const mounted = snap(headerPage.document);
headerPage.api.playPcm(pcm(1), 24000);
headerCtx.sources[0].onended();
headerPage.flush(8);
const header = snap(headerPage.document);

const halfPage = boot();
const halfCtx = makeCtx(10);
halfPage.api.arm(halfCtx);
question(halfPage);
halfPage.api.playPcm(pcm(48000), 24000);
halfCtx.currentTime = 14;
halfPage.flush(4);
const beforeHalf = snap(halfPage.document);
halfCtx.currentTime = 15;
halfPage.flush(4);
const halfway = snap(halfPage.document);

const endPage = boot();
const endCtx = makeCtx(10);
endPage.api.arm(endCtx);
question(endPage);
endPage.api.playPcm(pcm(4800), 24000);
endCtx.currentTime = 10.2;
endCtx.sources[0].onended();
endPage.flush(4);
const ended = snap(endPage.document);
endPage.api.onSocketMessage({
  data: JSON.stringify({ type: "bot_turn_complete", step: "birth_term" })
});
const otherStep = snap(endPage.document);
endPage.api.onSocketMessage({
  data: JSON.stringify({ type: "bot_turn_complete", step: "birth_nicu" })
});
const turnComplete = snap(endPage.document);

process.stdout.write(JSON.stringify({
  idle, mounted, header, beforeHalf, halfway, ended, otherStep, turnComplete
}));
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


def _render_findings(dashboard: str) -> dict:
    start = dashboard.index("function cursorDetail")
    end = dashboard.index("function renderTranscript")
    body = dashboard[start:end]
    harness = r"""
const findingsEl = { children: [], _text: "" };
Object.defineProperty(findingsEl, "textContent", {
  set(value) { this._text = value; if (value === "") this.children = []; },
  get() { return this._text; }
});
findingsEl.appendChild = function (child) { this.children.push(child); };
function el(tag, className, text) {
  const node = { tag: tag, className: className || "", text: text == null ? "" : String(text), children: [] };
  node.appendChild = function (child) { this.children.push(child); };
  return node;
}
function asList(value) { return Array.isArray(value) ? value : []; }
function asObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  return {};
}
function labelize(key) {
  return String(key || "").replace(/^profile_/, "").replace(/_/g, " ").replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
}
""" + body + r"""
function dump(table) {
  const rows = table.children.find(function (node) { return node.tag === "tbody"; }).children;
  return rows.map(function (row) {
    const cells = row.children.map(function (cell) { return cell.text; });
    return { qid: cells[0], domain: cells[0], state: cells[1], detail: cells[2], flag: cells[3] };
  });
}
renderFindings({}, {
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
});
const cursor = dump(findingsEl.children[0]);
renderFindings(
  { chief_complaint: { status: "present", detail: "खांसी", is_red_flag: true } },
  { answers: { birth_nicu: { state: "concern", notes: "hidden", child_ids: [], followups: {} } } }
);
const legacy = dump(findingsEl.children[0]);
renderFindings({}, { answers: { preg_scan: { state: "not_applicable" } } });
const empty = findingsEl.children[0].text;
process.stdout.write(JSON.stringify({ cursor: cursor, legacy: legacy, empty: empty }));
"""
    import subprocess

    done = subprocess.run(
        [_NODE, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
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

        with patch.object(gemini_live, "_connect_live", fake_connect), patch.object(
            report_service, "mother_pdf", fast_pdf
        ):
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
                    ws.send_json({
                        "type": "quick_reply",
                        "outcome": "concern",
                        "step": intro["step"],
                    })
                    session.push(_server_message(tool_calls=[
                        _tool("sp-1", "save_profile", profile),
                        _tool("sp-2", "save_profile", profile),
                        _tool("sp-3", "save_profile", profile),
                    ]))
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
                    session.push(_server_message(tool_calls=[
                        _tool("fc-child", "record_turn", {
                            "outcome": "concern",
                            "child_ref": "child_1",
                        }),
                    ]))
                    follow = wait_for(
                        lambda msg: msg.get("type") == "question_state" and msg.get("step") == "birth_mode_detail:0"
                    )
                    self.assertFalse(follow["expect_yes_no"])
                    self.assertGreater(follow["prompt_chars"], 0)
                    self.assertEqual(follow["qid"], "birth_mode_detail")
                finally:
                    ws.close()

        states = [item for item in messages if item.get("type") == "question_state"]
        self.assertEqual(
            [(item["qid"], item["step"], item["expect_yes_no"]) for item in states],
            [
                ("intro_name", "intro_name", False),
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
