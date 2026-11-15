"""Stage 3: same-size flanking replies, admin cursor answers, and the live tap path."""

import asyncio
import json
import queue
import shutil
import threading
import time
import unittest
from unittest.mock import patch

from tests import APP_DIR
from tests.helpers import (
    DummyRecorder,
    FakeLiveSession,
    FakeToolCall,
    FakeWebSocket,
    SessionCtx,
    bot_audio_message,
    choice,
    clear_until,
    decision_result,
    make_child,
    make_profile_state,
    noul,
    run_admin_vm,
    run_call_vm,
    server_message,
    shipped_thresholds,
)

import app  # noqa: E402
import db  # noqa: E402
import decisions  # noqa: E402
import gemini_live  # noqa: E402
import profile_extractor  # noqa: E402
import report_service  # noqa: E402
from call_cursor import apply_profile, apply_turn, new_cursor_state, step_id  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_STATIC = APP_DIR / "static"
_NODE = shutil.which("node")


class QuickReplyScreenTests(unittest.TestCase):
    def test_call_page_dock_and_accessibility_contrast_in_html(self):
        client = TestClient(app.app)
        client.post("/api/login", json={"username": "Mother", "passcode": app._passcode()})
        served = client.get("/").text
        dock_at, no_at, call_at, yes_at = (
            served.index('id="call-dock"'),
            served.index('id="no-btn"'),
            served.index('id="call-btn"'),
            served.index('id="yes-btn"'),
        )
        self.assertTrue(dock_at < no_at < call_at < yes_at)
        self.assertEqual(served.count('id="call-btn"'), 1)
        for snippet in (
            'aria-label="नहीं, आगे बढ़ें (No)"',
            'aria-label="हाँ (Yes)"',
            "clamp(68px, 21vw, 84px)",
            ".stage.working #call-dock",
            ".stage.working #call-btn",
            "prefers-reduced-motion: reduce",
            'id="hud-text"',
            'aria-label="Type message to SHISHU"',
            'id="bug-note"',
            'aria-label="समस्या का विवरण (Bug note)"',
        ):
            self.assertIn(snippet, served)

        admin_html = (_STATIC / "admin.html").read_text(encoding="utf-8")
        for snippet in (
            'id="review-note"',
            'aria-label="Clinician review note"',
            'id="player"',
            'aria-label="Call recording playback"',
            "--red-dark: #b91c1c;",
            "--amber-dark: #92400e;",
            "color: var(--navy);",
        ):
            self.assertIn(snippet, admin_html)

    def test_speaker_clock_quick_reply_clinic_and_share_click_behavior(self):
        if _NODE is None:
            self.skipTest("node is not installed")
        result = run_call_vm(r"""
const sharePosts = [];
const p1 = boot({ search: "?clinic=KEM" });
const idle = snap(p1.document);
const live1 = await p1.startLiveCall();
const clinicWsUrl = live1.ws.url;
const storedClinic = p1.store.shishu_clinic;
const wakeLockKinds = p1.wakeLockKinds.slice();
pushMsg(live1.ws, { type: "session_started", call_id: 7, model: "gemini-3.1-flash-live-preview", voice: "Kore" });
const hudModel = p1.document.getElementById("hud-model").textContent;
pushMsg(live1.ws, { type: "question_state", step: "intro_name", qid: "intro_name", group: "intro", expect_yes_no: false, prompt_chars: 40 });
const nonYnMounted = snap(p1.document);
pushMsg(live1.ws, { type: "audio", data: pcm(24000), sample_rate: 24000 });
live1.ctx.currentTime = 12;
live1.ctx.sources[0].onended();
p1.flushRaf(4);
pushMsg(live1.ws, { type: "bot_turn_complete", step: "intro_name" });
const nonYnAfterTurn = snap(p1.document);

const p2 = boot();
const live2 = await p2.startLiveCall();
pushMsg(live2.ws, { type: "question_state", step: "birth_nicu", qid: "birth_nicu", group: "birth", expect_yes_no: true, prompt_chars: 120 });
pushMsg(live2.ws, { type: "audio", data: pcm(48000), sample_rate: 24000 });
live2.ctx.currentTime = 14; p2.flushRaf(4);
const beforeHalf = snap(p2.document);
live2.ctx.currentTime = 15; p2.flushRaf(4);
const halfway = snap(p2.document);
p2.document.getElementById("no-btn").click();
p2.document.getElementById("no-btn").click();
p2.document.getElementById("yes-btn").click();
const sourceStoppedOnClick = live2.ctx.sources[0].stopped;
const afterNoClick = snap(p2.document);
const captionAfterNo = p2.document.getElementById("caption").textContent;
const quickRepliesSent = live2.ws.sent.filter((m) => m.type === "quick_reply");

const p3 = boot();
p3.setCustomFetch((url, init) => {
  if (url.indexOf("/events") !== -1) {
    sharePosts.push({ url, body: JSON.parse(init.body) });
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ ok: true }) });
  }
  if (url.indexOf("/mother.pdf") !== -1) return Promise.resolve(p3.pdfOkResponse());
  return null;
});
const live3 = await p3.startLiveCall();
pushMsg(live3.ws, { type: "session_started", call_id: 8, share_token: "tok-share" });
pushMsg(live3.ws, { type: "question_state", step: "birth_jaundice", qid: "birth_jaundice", group: "birth", expect_yes_no: true, prompt_chars: 120 });
pushMsg(live3.ws, { type: "audio", data: pcm(4800), sample_rate: 24000 });
live3.ctx.currentTime = 10.2; live3.ctx.sources[0].onended(); p3.flushRaf(4);
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
pushMsg(live3.ws, { type: "call_completed", call_id: 8 });
await flushMicrotasks();
p3.document.getElementById("share-pdf").click();
await flushMicrotasks();

process.stdout.write(JSON.stringify({
  idle, clinicWsUrl, storedClinic, wakeLockKinds, hudModel, nonYnMounted, nonYnAfterTurn,
  beforeHalf, halfway, otherStep, turnComplete, sourceStoppedOnClick, afterNoClick,
  captionAfterNo, quickRepliesSent, captionAfterYes, yesReplySent, afterEnding, sharePosts
}));
""")
        self.assertTrue(result["idle"]["noHidden"] and result["idle"]["yesHidden"])
        self.assertFalse(result["idle"]["noVisible"] or result["idle"]["yesVisible"])
        self.assertIn("?clinic=kem", result["clinicWsUrl"])
        self.assertEqual(result["storedClinic"], "kem")
        self.assertIn("screen", result["wakeLockKinds"])
        self.assertEqual(result["hudModel"], "gemini-3.1-flash-live-preview")
        self.assertFalse(result["nonYnMounted"]["noHidden"] or result["nonYnMounted"]["noVisible"])
        self.assertTrue(result["nonYnMounted"]["noDisabled"] and result["nonYnAfterTurn"]["noDisabled"])
        self.assertFalse(result["beforeHalf"]["noVisible"] or result["beforeHalf"]["yesVisible"])
        self.assertTrue(result["halfway"]["noVisible"] and result["halfway"]["yesVisible"] and not result["halfway"]["noDisabled"])
        self.assertFalse(result["otherStep"]["noVisible"] or result["otherStep"]["yesVisible"])
        self.assertTrue(result["turnComplete"]["noVisible"] and result["turnComplete"]["yesVisible"] and not result["turnComplete"]["yesDisabled"])
        self.assertTrue(result["sourceStoppedOnClick"] and result["afterNoClick"]["noDisabled"] and result["afterNoClick"]["yesDisabled"])
        self.assertEqual(result["captionAfterNo"], "You: नहीं।")
        self.assertEqual(result["quickRepliesSent"], [{"type": "quick_reply", "outcome": "clear", "step": "birth_nicu"}])
        self.assertEqual(result["captionAfterYes"], "You: हाँ।")
        self.assertEqual(result["yesReplySent"], [{"type": "quick_reply", "outcome": "concern", "step": "birth_jaundice"}])
        self.assertTrue(result["afterEnding"]["stageWorking"] and not result["afterEnding"]["waitHidden"] and result["afterEnding"]["noHidden"])
        self.assertEqual(result["sharePosts"], [{"url": "/api/calls/by-token/tok-share/events", "body": {"event": "share_clicked"}}])

    def test_call_js_lifecycle_timers_disconnect_and_pdf_race_guards(self):
        if _NODE is None:
            self.skipTest("node is not installed")
        result = run_call_vm(r"""
const s1 = boot();
await flushMicrotasks();
s1.document.getElementById("hud-toggle").click();
const hudBeforeStart = s1.document.getElementById("hud").hidden;
s1.start();
await flushMicrotasks();
const hudStayedClosed = hudBeforeStart === true && s1.document.getElementById("hud").hidden === true;
const ws1 = s1.sockets[0];
pushMsg(ws1, { type: "session_started", call_id: 9, model: "m", voice: "v" });
pushMsg(ws1, { type: "question_state", step: "birth_nicu", qid: "birth_nicu", group: "birth", expect_yes_no: true, prompt_chars: 100 });
pushMsg(ws1, { type: "bot_turn_complete", step: "birth_nicu" });

s1.document.getElementById("no-btn").click();
const beforeClearCount = s1.timeouts.filter((t) => t.ms === 900 && !t.cleared).length;
pushMsg(ws1, { type: "clear" });
const thinkingSetAfterClear = beforeClearCount >= 1 && s1.timeouts.some((t) => t.ms === 900 && !t.cleared);

s1.document.getElementById("timer").textContent = "01:23";
const clearedBeforeClose = s1.getClearedIntervals();
ws1.onclose();
const timersClearedOnClose = s1.getClearedIntervals() - clearedBeforeClose;
const audioClosedOnClose = s1.ctxs[0].closed;
const statusAfterClose = s1.document.getElementById("status").innerHTML.indexOf("Tap below to start the call") !== -1;
const errorHiddenAfterClose = !!s1.document.getElementById("error").hidden;

s1.start();
const timerAfterRestart = s1.document.getElementById("timer").textContent;
const hudStepAfterRestart = s1.document.getElementById("hud-step").textContent;
await flushMicrotasks();
const ws2 = s1.sockets[1];
pushMsg(ws2, { type: "session_started", call_id: 10, share_token: "tok-12345678" });
pushMsg(ws2, { type: "report_failed", call_id: 10 });
const summaryHiddenOnFail = !!s1.document.getElementById("summary").hidden;
const shareHiddenOnFail = !!s1.document.getElementById("share-pdf").hidden;
const retryHiddenOnFail = !!s1.document.getElementById("retry-pdf").hidden;
s1.setRetryPostOk(true);
s1.document.getElementById("retry-pdf").click();
await flushMicrotasks();
const summaryHiddenOnRetryOk = !!s1.document.getElementById("summary").hidden;

let resolveMic1; let staleTrackStopped = false;
const s2 = boot({
  customMic: (n) => n === 1
    ? new Promise((r) => { resolveMic1 = r; })
    : Promise.resolve({ getTracks: () => [{ stop() {} }], getAudioTracks: () => [] })
});
await flushMicrotasks();
s2.start(); s2.document.getElementById("call-btn").click(); s2.start();
await flushMicrotasks();
resolveMic1({ getTracks: () => [{ stop() { staleTrackStopped = true; } }], getAudioTracks: () => [] });
await flushMicrotasks();
const activeWsCountAfterRace = s2.sockets.filter((s) => !s.closed).length;
const secondCtxClosed = s2.ctxs[s2.ctxs.length - 1].closed;

const s3 = boot({ throwOnHangup: true });
await flushMicrotasks(); s3.start(); await flushMicrotasks();
s3.document.getElementById("call-btn").click();
const workingAfterEndSendThrow = s3.document.getElementById("stage").classList.contains("working");
const errorHiddenAfterThrow = !!s3.document.getElementById("error").hidden;

let resolveGetPdf, resolvePostPdf, rejectPostPdf;
const s4 = boot({ search: "?clinic=kem" });
s4.setCustomFetch((url, init) => {
  if (url.indexOf("/mother.pdf") !== -1 && (!init || !init.method || init.method === "GET")) {
    return new Promise((r) => { resolveGetPdf = r; });
  }
  if (url.indexOf("/mother.pdf") !== -1 && init && init.method === "POST") {
    return new Promise((r, j) => { resolvePostPdf = r; rejectPostPdf = j; });
  }
  return null;
});
await flushMicrotasks();
s4.start(); await flushMicrotasks();
pushMsg(s4.sockets[0], { type: "session_started", call_id: 11, share_token: "tok-11" });
pushMsg(s4.sockets[0], { type: "ending" });
pushMsg(s4.sockets[0], { type: "call_completed", call_id: 11 });
s4.sockets[0].onclose();
resolveGetPdf(s4.pdfOkResponse());
await flushMicrotasks();
const thanksHiddenAfterNormalWsClose = !!s4.document.getElementById("thanks").hidden;
const shareHiddenAfterNormalWsClose = !!s4.document.getElementById("share-pdf").hidden;

s4.document.getElementById("new-call").click();
s4.start(); await flushMicrotasks();
pushMsg(s4.sockets[1], { type: "session_started", call_id: 12, share_token: "tok-12" });
pushMsg(s4.sockets[1], { type: "ending" });
pushMsg(s4.sockets[1], { type: "call_completed", call_id: 12 });
s4.document.getElementById("new-call").click();
resolveGetPdf(s4.pdfOkResponse());
await flushMicrotasks();
const thanksHiddenAfterNewCallGet = !!s4.document.getElementById("thanks").hidden;
const workingAfterNewCallGet = s4.document.getElementById("stage").classList.contains("working");

s4.start(); await flushMicrotasks();
pushMsg(s4.sockets[2], { type: "session_started", call_id: 13, share_token: "tok-13" });
pushMsg(s4.sockets[2], { type: "report_failed", call_id: 13 });
s4.document.getElementById("retry-pdf").click();
const retryPreparingOk = s4.document.getElementById("stage").classList.contains("working")
  && !s4.document.getElementById("wait").hidden
  && !!s4.document.getElementById("call-btn").disabled
  && s4.document.getElementById("status").textContent === "";
s4.document.getElementById("new-call").click();
resolvePostPdf(s4.pdfOkResponse());
await flushMicrotasks();
const thanksHiddenAfterNewCallPost = !!s4.document.getElementById("thanks").hidden;
const waitHiddenAfterNewCallPost = !!s4.document.getElementById("wait").hidden;

s4.start(); await flushMicrotasks();
pushMsg(s4.sockets[3], { type: "session_started", call_id: 14, share_token: "tok-14" });
pushMsg(s4.sockets[3], { type: "report_failed", call_id: 14 });
s4.document.getElementById("retry-pdf").click();
s4.document.getElementById("new-call").click();
s4.start(); await flushMicrotasks();
rejectPostPdf(new Error("late fail"));
await flushMicrotasks();
const thanksHiddenAfterLateRejectNewCall = !!s4.document.getElementById("thanks").hidden;

const catChip = { getAttribute: (k) => k === "data-cat" ? "Other" : "", setAttribute() {}, classList: { toggle() {} }, closest() { return this; } };
const bugCats = s4.document.getElementById("bug-cats");
bugCats.querySelectorAll = () => [catChip]; bugCats.contains = (n) => n === catChip;
bugCats.trigger("click", { target: catChip });
const bugShot = s4.document.getElementById("bug-shot");
bugShot.files = [{ name: "shot.png" }]; bugShot.trigger("change");
s4.document.getElementById("bug-save").click();
if (s4.getPendingImage() && s4.getPendingImage().onload) s4.getPendingImage().onload();
await flushMicrotasks();
const bugPostedShot = s4.getBugPostedBody() && s4.getBugPostedBody().client_state && s4.getBugPostedBody().client_state.screenshot;
const bugErrorTextOn413 = s4.document.getElementById("bug-error").textContent;
s4.document.getElementById("logout-btn").click();
await flushMicrotasks();

process.stdout.write(JSON.stringify({
  hudStayedClosed, thinkingSetAfterClear, timersClearedOnClose, audioClosedOnClose, statusAfterClose,
  errorHiddenAfterClose, timerAfterRestart, hudStepAfterRestart, summaryHiddenOnFail, shareHiddenOnFail,
  retryHiddenOnFail, summaryHiddenOnRetryOk, staleTrackStopped, activeWsCountAfterRace, secondCtxClosed,
  workingAfterEndSendThrow, errorHiddenAfterThrow, thanksHiddenAfterNormalWsClose, shareHiddenAfterNormalWsClose,
  thanksHiddenAfterNewCallGet, workingAfterNewCallGet, retryPreparingOk, thanksHiddenAfterNewCallPost,
  waitHiddenAfterNewCallPost, thanksHiddenAfterLateRejectNewCall, bugPostedShot, bugErrorTextOn413,
  clinicRemovedOnLogout: s4.wasClinicRemoved(), logoutHrefOnCall: s4.location.href
}));
""")
        for flag in (
            "hudStayedClosed", "thinkingSetAfterClear", "audioClosedOnClose", "statusAfterClose",
            "summaryHiddenOnFail", "shareHiddenOnFail", "staleTrackStopped",
            "thanksHiddenAfterNewCallGet", "retryPreparingOk",
            "thanksHiddenAfterNewCallPost", "waitHiddenAfterNewCallPost", "thanksHiddenAfterLateRejectNewCall",
            "clinicRemovedOnLogout",
        ):
            self.assertTrue(result[flag], flag)
        for flag in (
            "errorHiddenAfterClose", "retryHiddenOnFail", "summaryHiddenOnRetryOk", "secondCtxClosed",
            "workingAfterEndSendThrow", "errorHiddenAfterThrow", "thanksHiddenAfterNormalWsClose",
            "shareHiddenAfterNormalWsClose", "workingAfterNewCallGet",
        ):
            self.assertFalse(result[flag], flag)
        self.assertGreater(result["timersClearedOnClose"], 0)
        self.assertEqual((result["timerAfterRestart"], result["hudStepAfterRestart"], result["activeWsCountAfterRace"]), ("00:00", "—", 1))
        self.assertEqual((result["bugPostedShot"], result["logoutHrefOnCall"]), ("data:image/jpeg;base64,SHRUNK", "/login"))
        self.assertIn("too large", result["bugErrorTextOn413"])

    def test_admin_dashboard_answers_and_race_guards(self):
        if _NODE is None:
            self.skipTest("node is not installed")
        rendered = run_admin_vm(r"""
const call1 = {
  id: 1, caller_name: "पूजा", user_role: "user", mother_name: "पूजा", child_age: "2 साल",
  clinic_code: "kem", model_used: "gemini-3.1-flash-live-preview", voice_used: "Kore", duration_sec: 95,
  started_at: "2026-11-15T04:00:00Z", status: "completed", red_flags: ["birth_mode_detail"],
  summary_doctor: "# Doctor Summary\nAll good.", summary_mother: "माँ के लिए सवाल",
  telemetry: { connect_ms: 120, avg_latency_ms: 400, max_latency_ms: 600, turn_latencies_ms: [400], user_turns: 4, bot_turns: 5, tool_calls: 5, watchdog_nudges: 1, audio_chunks_in: 10, audio_chunks_out: 12 },
  events: [], transcript: [],
  cursor_state: { answers: {
    preg_scan: { state: "not_applicable", notes: "", child_ids: [], followups: {} },
    ever_pregnant: { state: "concern", notes: "", child_ids: [], followups: {} },
    intro_name: { state: "clear", notes: "पूजा", child_ids: [], followups: {} },
    intro_bridge: { state: "clear", notes: "", child_ids: [], followups: {} },
    birth_term: { state: "unknown", notes: "", child_ids: [], followups: {} },
    birth_mode_detail: { state: "concern", child_ids: ["child_1"], notes: "ऑपरेशन", followups: { operation_reason: { state: "concern", note: "सांस की तकलीफ" } } }
  } }
};
const call2 = {
  id: 2, caller_name: "रेखा", user_role: "user", mother_name: "रेखा", child_age: "4 साल",
  model_used: "gemini-custom-live", voice_used: "Kore", duration_sec: 45, status: "completed",
  red_flags: [], summary_doctor: "", summary_mother: "", telemetry: {}, events: [], transcript: [],
  cursor_state: { answers: { preg_scan: { state: "not_applicable", notes: "", child_ids: [], followups: {} } } }
};
const env = bootAdmin({
  fetch(path) {
    if (path === "/api/calls") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([call1, call2]) });
    if (path === "/api/calls/1") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(call1) });
    if (path === "/api/calls/2") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(call2) });
    if (path === "/api/bugs") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([]) });
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
  }
});
await flushMicrotasks();
const rows = env.document.getElementById("findings").children[0].children[1].children.map((tr) => ({
  rowClass: tr.className, qid: tr.children[0].textContent, state: tr.children[1].textContent,
  detail: tr.children[2].textContent, flag: tr.children[3].textContent
}));
const headers = env.document.getElementById("findings").children[0].children[0].children[0].children.map((th) => th.textContent);
const callList = env.document.getElementById("call-list").children;
const modelBadge = callList[0].children[3].children[0].textContent;
const customModelBadge = callList[1].children[3].children[0].textContent;
const callMeta = env.document.getElementById("call-meta").textContent;
const kpis = env.document.getElementById("telemetry-grid").children.map((b) => b.children.map((n) => n.textContent).join(": ")).join(" | ");
env.document.getElementById("copy-report").click();
await flushMicrotasks();
const copyNotice = env.document.getElementById("notice").textContent;
callList[1].click();
await flushMicrotasks();
process.stdout.write(JSON.stringify({
  fetchedUrls: env.fetchedUrls, headers, rows, modelBadge, customModelBadge, callMeta, kpis,
  copiedReport: env.getCopiedText(), copyNotice, emptyText: env.document.getElementById("findings").children[0].textContent
}));
""")
        self.assertNotIn("/api/config", rendered["fetchedUrls"])
        self.assertEqual(rendered["headers"], ["Question ID", "State", "Detail", "Red flag"])
        self.assertEqual((rendered["modelBadge"], rendered["customModelBadge"]), ("Gemini 3.1 Flash Live", "gemini-custom-live"))
        self.assertIn("पूजा · 2 साल", rendered["callMeta"])
        self.assertIn("5 tool calls / 1 nudges", rendered["kpis"])
        rows = rendered["rows"]
        self.assertEqual([r["state"] for r in rows], ["concern", "concern", "unknown", "clear", "clear"])
        self.assertEqual((rows[0]["qid"], rows[0]["flag"], rows[0]["rowClass"]), ("birth_mode_detail", "Yes", "flag"))
        self.assertTrue("child_1" in rows[0]["detail"] and "ऑपरेशन" in rows[0]["detail"] and "operation_reason: सांस की तकलीफ" in rows[0]["detail"])
        self.assertEqual((rows[1]["qid"], rows[1]["flag"], rows[1]["rowClass"]), ("ever_pregnant", "", ""))
        self.assertEqual((rendered["emptyText"], rendered["copiedReport"], rendered["copyNotice"]),
                         ("No structured findings were recorded.", "# Doctor Summary\nAll good.", "Report copied."))

        races = run_admin_vm(r"""
let releaseCall1, call1Delay = false, failCall1 = false, call1Status = "active", savedReviewPayload = null;
const call1 = { id: 1, caller_name: "पूजा", clinic_code: "kem", status: "active", audio_path: "", duration_sec: 10, red_flags: [], telemetry: {}, events: [], transcript: [], cursor_state: { answers: {} }, clinician_review: { rating: "useful", note: "" } };
const call2 = {
  id: 2, caller_name: "रेखा", clinic_code: "sion", status: "completed", audio_path: "", duration_sec: 20, red_flags: [],
  telemetry: { turn_latencies_ms: [], avg_latency_ms: 0, max_latency_ms: 0 },
  events: [
    { t: 100, type: "decision_turn", detail: "birth_nicu:clear", meta: { decision_ms: 42, error_reason: "", diagnostics: { nested: 1 } } },
    { t: 101, type: "pdf_write", detail: "", meta: { bytes: 2048 } },
    { t: 102, type: "decision_turn", detail: "error:timeout", meta: { error_reason: "timeout", decision_ms: 3000 } }
  ],
  transcript: [], cursor_state: { answers: {} }, clinician_review: {}
};
const bugs = [{ id: 5, call_id: 1, category: "Other", note: "ui shot", call_elapsed_sec: 5, snapshot: { client: { screenshot: "data:image/png;base64,BASE64DATA" } } }];
const env = bootAdmin({
  setupDoc(doc) {
    doc.getElementById("review-useful").setAttribute("data-rating", "useful");
    doc.getElementById("review-bar").contains = (n) => n === doc.getElementById("review-useful");
  },
  querySelectorAll(sel, doc) { return sel === "#review-bar [data-rating]" ? [doc.getElementById("review-useful")] : []; },
  fetch(path, init) {
    if (path === "/api/calls") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([Object.assign({}, call1, { status: call1Status }), call2]) });
    if (path === "/api/calls/1") {
      if (failCall1) return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ detail: "Server boom" }) });
      if (call1Delay) return new Promise((res) => { releaseCall1 = () => res({ ok: true, status: 200, json: () => Promise.resolve(Object.assign({}, call1, { status: call1Status })) }); });
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(Object.assign({}, call1, { status: call1Status })) });
    }
    if (path === "/api/calls/2") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(call2) });
    if (path === "/api/logout") return Promise.reject(new Error("offline"));
    if (path === "/api/calls/2/mother.pdf") return Promise.resolve({ ok: false, status: 409, json: () => Promise.resolve({ detail: "Report is not ready" }) });
    if (path === "/api/calls/1/review" && init && init.method === "POST") {
      savedReviewPayload = JSON.parse(init.body);
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ ok: true, clinician_review: savedReviewPayload }) });
    }
    if (path === "/api/bugs") return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(bugs) });
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
  }
});
await flushMicrotasks();
const bugItem = env.document.getElementById("bug-list").children[0];
const bugPreText = bugItem.children.find((c) => c.tagName === "DETAILS").children.find((c) => c.tagName === "PRE").textContent;
bugItem.children.find((c) => c.textContent === "Copy Bug JSON").click();
await flushMicrotasks();
const copiedBugJson = env.getCopiedText();

const reviewNote = env.document.getElementById("review-note");
reviewNote.value = "draft clinical observation"; reviewNote.trigger("input");
env.document.activeElement = reviewNote; call1Status = "completed";
env.document.getElementById("refresh").click(); await flushMicrotasks();
const noteAfterBackgroundRefresh = reviewNote.value;
env.document.getElementById("save-review-note").click(); await flushMicrotasks();
env.document.activeElement = null;

call1Delay = true;
const btns = env.document.getElementById("call-list").children;
btns[0].click(); btns[1].click(); await flushMicrotasks();
releaseCall1(); await flushMicrotasks();
const titleAfterRace = env.document.getElementById("call-title").textContent;
const noAudioState = !!env.document.getElementById("player").hidden && !!env.document.getElementById("download").hidden && !env.document.getElementById("audio-note").hidden;
env.document.getElementById("mother-pdf").click(); await flushMicrotasks();
const pdf409Notice = env.document.getElementById("notice").textContent;
const call2Kpis = env.document.getElementById("telemetry-grid").children.map((b) => b.children.map((n) => n.textContent).join(": ")).join(" | ");
const call2EventDetails = env.document.getElementById("event-table").children[0].children[1].children.map((tr) => tr.children[2].textContent).join(" || ");

failCall1 = true; env.document.getElementById("call-list").children[0].click(); await flushMicrotasks();
const titleAfterFailedDetail = env.document.getElementById("call-title").textContent;
const noticeAfterFailedDetail = env.document.getElementById("notice").textContent;
const btnsFail = env.document.getElementById("call-list").children;
const row0Sel = btnsFail[0].classList.contains("selected"), row1Sel = btnsFail[1].classList.contains("selected");
failCall1 = false; call1Delay = false;

const filterSelect = env.document.getElementById("clinic-filter");
filterSelect.value = "sion"; filterSelect.trigger("change", { target: filterSelect });
bugItem.children[0].children.find((c) => c.textContent === "Call #1").click(); await flushMicrotasks();
const clinicFilterAfterBugJump = filterSelect.value, titleAfterBugJump = env.document.getElementById("call-title").textContent;
env.document.getElementById("logout-btn").click(); await flushMicrotasks();

process.stdout.write(JSON.stringify({
  titleAfterRace, noAudioState, pdf409Notice, noteAfterBackgroundRefresh, savedReviewPayload,
  bugPreText, copiedBugJson, clinicFilterAfterBugJump, titleAfterBugJump, call2Kpis, call2EventDetails,
  titleAfterFailedDetail, noticeAfterFailedDetail, row0Sel, row1Sel, logoutHrefOnAdmin: env.location.href
}));
""")
        self.assertTrue(races["noAudioState"] and races["row1Sel"] and not races["row0Sel"])
        self.assertEqual((races["titleAfterRace"], races["titleAfterFailedDetail"], races["titleAfterBugJump"], races["clinicFilterAfterBugJump"]),
                         ("Call #2", "Call #2", "Call #1", ""))
        self.assertIn("Report is not ready", races["pdf409Notice"])
        self.assertEqual((races["noteAfterBackgroundRefresh"], races["savedReviewPayload"]),
                         ("draft clinical observation", {"rating": "useful", "note": "draft clinical observation"}))
        self.assertTrue("[screenshot attached]" in races["bugPreText"] and "BASE64DATA" not in races["bugPreText"])
        self.assertIn("data:image/png;base64,BASE64DATA", races["copiedBugJson"])
        self.assertIn("Server boom", races["noticeAfterFailedDetail"])
        self.assertTrue("Turn Latency (Avg / Max): — / —" in races["call2Kpis"] and "0 ms" not in races["call2Kpis"])
        self.assertTrue("birth_nicu:clear · decision_ms=42" in races["call2EventDetails"] and "bytes=2048" in races["call2EventDetails"])
        self.assertTrue("error:timeout · error_reason=timeout decision_ms=3000" in races["call2EventDetails"])
        self.assertTrue("diagnostics" not in races["call2EventDetails"] and "[object Object]" not in races["call2EventDetails"])
        self.assertEqual(races["logoutHrefOnAdmin"], "/login")


class QuickReplySocketTests(unittest.TestCase):
    def test_quick_reply_walks_bridge_gate_and_concern_followup(self):
        session = FakeLiveSession(wait_forever=True)
        inbox, messages = queue.Queue(), []

        def pump(ws):
            try:
                while True:
                    inbox.put(ws.receive_json())
            except BaseException as exc:
                inbox.put(exc)

        def wait_for(predicate, timeout=12):
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    item = inbox.get(timeout=max(0.01, deadline - time.time()))
                except queue.Empty:
                    break
                if isinstance(item, BaseException):
                    raise item
                messages.append(item)
                if item.get("type") == "error":
                    self.fail(item.get("message") or item)
                if predicate(item):
                    return item
            self.fail("timed out waiting for message: " + json.dumps(messages[-8:], ensure_ascii=False, default=str))

        async def fake_eval(request, **_kwargs):
            ctx, qs = json.loads(request["input"]), request.get("questions") or {}
            if "child_ref" in qs and "detail_present" not in qs and "parent_question" in ctx:
                raw = {
                    "danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_profile": noul(0.01),
                    "current_outcome": choice("concern", {"concern": 0.94, "other": 0.04, "clear": 0.02}),
                    "child_ref": choice("child_1", {"child_1": 0.92, "other": 0.04, "child_2": 0.03}),
                    "side_intent": choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": choice("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
            else:
                raw = {"danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_profile": noul(0.94),
                       "current_outcome": choice("other", {"other": 0.92, "clear": 0.05, "concern": 0.03})}
            return decision_result(request, raw, decision_ms=22, total_input_tokens=240)

        async def fake_extract(state, **_kwargs):
            qid = state.get("active_qid")
            updates = ({"caller_name": "पूजा"} if qid == "intro_name"
                       else {"marital_status": "married"} if qid == "intro_married"
                       else {"child_count": 2, "children": [{"age_years": 6, "sex": "male", "name": "अमन"}, {"age_years": 2, "sex": "female", "name": "रिया"}]} if qid == "intro_children"
                       else {})
            return profile_extractor.ProfileExtractionResult(ok=True, updates=updates, error_reason="", extractor_ms=15)

        async def fake_connect(_client, _model, _config):
            return SessionCtx(session, close_on_exit=True), session, 4

        async def fast_pdf(_cid):
            return b"%PDF-1.4\n", "SHISHU.pdf"

        with patch.object(gemini_live, "_connect_live", fake_connect), patch.object(
            report_service, "mother_pdf", fast_pdf
        ), patch.object(decisions, "evaluate_turn", side_effect=fake_eval) as eval_mock, patch.object(
            profile_extractor, "extract_profile_entities", side_effect=fake_extract
        ) as ext_mock:
            client = TestClient(app.app)
            self.assertEqual(client.post("/api/login", json={"username": "पूजा", "passcode": app._passcode()}).status_code, 200)
            with client.websocket_connect("/ws/gemini") as ws:
                threading.Thread(target=pump, args=(ws,), daemon=True).start()
                try:
                    intro = wait_for(lambda m: m.get("type") == "question_state" and m.get("qid") == "intro_name")
                    session.push(bot_audio_message(turn_complete=True))
                    wait_for(lambda m: m.get("type") == "audio")
                    ws.send_json({"type": "quick_reply", "outcome": "concern", "step": intro["step"]})
                    for utt, needle, tid, next_q in (
                        ("मेरा नाम पूजा है", "पूजा", "nq-1", "intro_married"),
                        ("हाँ, शादी हो गई है", "शादी", "nq-2", "intro_children"),
                        ("दो बच्चे हैं, 6 साल का अमन और 2 साल की रिया", "अमन", "nq-3", "intro_bridge"),
                    ):
                        ws.send_json({"type": "text", "text": utt})
                        wait_for(lambda m, n=needle: m.get("type") == "transcript" and m.get("role") == "user" and n in str(m.get("text") or ""))
                        session.push(server_message(tool_calls=[FakeToolCall(fid=tid)], turn_complete=True))
                        nxt = wait_for(lambda m, q=next_q: m.get("type") == "question_state" and m.get("qid") == q)
                        session.push(bot_audio_message(turn_complete=True))
                        wait_for(lambda m: m.get("type") == "audio")

                    ws.send_json({"type": "quick_reply", "outcome": "clear", "step": nxt["step"]})
                    gate = wait_for(lambda m: m.get("type") == "question_state" and m.get("qid") == "birth_mode_detail" and m.get("expect_yes_no") is True)
                    session.push(bot_audio_message(turn_complete=True))
                    wait_for(lambda m: m.get("type") == "audio")
                    ws.send_json({"type": "quick_reply", "outcome": "concern", "step": gate["step"]})
                    wait_for(lambda m: m.get("type") == "question_state" and m.get("step") == "birth_mode_detail:-1")
                    session.push(bot_audio_message(turn_complete=True))
                    wait_for(lambda m: m.get("type") == "audio")
                    ws.send_json({"type": "text", "text": "बड़ा वाला"})
                    wait_for(lambda m: m.get("type") == "transcript" and m.get("role") == "user" and "बड़ा" in str(m.get("text") or ""))
                    session.push(server_message(tool_calls=[FakeToolCall(fid="nq-child")], turn_complete=True))
                    wait_for(lambda m: m.get("type") == "question_state" and m.get("step") == "birth_mode_detail:0")
                finally:
                    ws.close()
            self.assertEqual((eval_mock.await_count, ext_mock.await_count), (4, 3))

        states = [m for m in messages if m.get("type") == "question_state"]
        self.assertEqual(
            [(m["qid"], m["step"], m["expect_yes_no"]) for m in states],
            [("intro_name", "intro_name", False), ("intro_married", "intro_married", False),
             ("intro_children", "intro_children", False), ("intro_bridge", "intro_bridge", True),
             ("birth_mode_detail", "birth_mode_detail", True), ("birth_mode_detail", "birth_mode_detail:-1", False),
             ("birth_mode_detail", "birth_mode_detail:0", False)],
        )
        self.assertTrue(all(m["prompt_chars"] > 0 for m in states))
        said = [m.get("text") for m in messages if m.get("type") == "transcript" and m.get("role") == "user"]
        self.assertTrue("नहीं।" in said and "हाँ।" in said and {"type": "clear"} in messages)
        saved = db.get_call(next(m["call_id"] for m in messages if m.get("type") == "session_started"))["cursor_state"]
        self.assertEqual((saved["answers"]["birth_mode_detail"]["child_ids"], saved["pending_followup_idx"]), (["child_1"], 0))

    def test_prompt_chars_and_yes_no_follow_the_step_kind(self):
        state = new_cursor_state()
        for expected_qid, expected_yn, prof_kw in (
            ("intro_name", False, None),
            ("intro_married", False, {"caller_name": "रेखा"}),
            ("intro_children", False, {"marital_status": "married"}),
        ):
            if prof_kw:
                state, _ = apply_profile(state, **prof_kw)
            body = gemini_live._question_state_payload(state)
            self.assertEqual((body["qid"], body["expect_yes_no"]), (expected_qid, expected_yn))
            self.assertGreater(body["prompt_chars"], 0)

        kids = [make_child(8, "male", 1), make_child(2, "female", 2)]
        state = clear_until(make_profile_state(children=kids), "birth_mode_detail")
        self.assertTrue(gemini_live._question_state_payload(state)["expect_yes_no"])
        state = apply_turn(state, "concern")
        self.assertEqual((gemini_live._question_state_payload(state)["step"], gemini_live._question_state_payload(state)["expect_yes_no"]), ("birth_mode_detail:-1", False))
        state = apply_turn(state, "concern", child_ref="child_1")
        self.assertEqual((gemini_live._question_state_payload(state)["step"], gemini_live._question_state_payload(state)["expect_yes_no"]), ("birth_mode_detail:0", False))

    def test_quick_reply_superseding_inflight_decision_holds_without_duplicate_cue(self):
        async def run():
            thresholds = shipped_thresholds()
            state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu")
            step1 = step_id(state)
            eval_started, release_eval = asyncio.Event(), asyncio.Event()

            async def slow_eval(request, **_kwargs):
                eval_started.set()
                await release_eval.wait()
                return decision_result(request, {
                    "danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_profile": noul(0.01),
                    "detail_present": noul(0.02),
                    "current_outcome": choice("concern", {"concern": 0.94, "other": 0.04, "clear": 0.02}),
                    "side_intent": choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": choice("other", {"other": 0.96, "birth_jaundice": 0.04}),
                }, decision_ms=12, total_input_tokens=180)

            for bot_already_spoke, tc in ((True, True), (False, False)):
                eval_started.clear()
                release_eval.clear()
                ws = FakeWebSocket([{"type": "quick_reply", "outcome": "clear", "step": step1}])
                session = FakeLiveSession()
                live = gemini_live.LiveCallSession(
                    ws=ws, session=session, call_id=db.create_call(), recorder=DummyRecorder(),
                    cursor_state=json.loads(json.dumps(state)), thresholds=thresholds,
                )
                live.turn.spoken_qid = live.turn.user_turn_step = step1
                live.capture.user_turn_buffer = "नहीं"

                with patch.object(decisions, "evaluate_turn", side_effect=slow_eval):
                    tool_task = asyncio.create_task(live._handle_server_message(server_message(tool_calls=[FakeToolCall(fid="fc-in")], turn_complete=tc)))
                    await asyncio.wait_for(eval_started.wait(), timeout=1.0)
                    await live._browser_loop()
                    qr_ts = live.turn.last_user_turn_ts
                    if bot_already_spoke:
                        await live._on_bot_audio(b"\x01\x00" * 24, 24000)
                        live.turn.bot_speaking = False
                        pinned_quiet = time.time() - 0.2
                        live.turn.quiet_from = pinned_quiet
                    release_eval.set()
                    self.assertTrue(await tool_task)

                self.assertEqual((live.cursor_state["answers"]["birth_nicu"]["state"], live.cursor_state["active_qid"]), ("clear", "birth_jaundice"))
                tool_resp = [item[1] for item in session.sent if item[0] == "tool"][0][0]
                self.assertEqual((tool_resp.response["result"], tool_resp.response["reason"]), ("held", "stale_turn"))
                self.assertEqual(len([m for m in ws.sent if m.get("type") == "question_state" and m.get("step") == "birth_jaundice"]), 1)
                self.assertEqual(len([item for item in session.sent if item[0] == "realtime"]), 1)
                self.assertIsNone(live.turn.speak_fallback)
                self.assertEqual(live.turn.last_user_turn_ts, qr_ts)
                self.assertEqual(live.turn.awaiting_bot_reply, not bot_already_spoke)
                if bot_already_spoke:
                    self.assertGreaterEqual(live.turn.quiet_from, pinned_quiet)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
