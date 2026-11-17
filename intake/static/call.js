/* SHISHU V3 call page: one pack, one socket, one call. */
"use strict";

const el = (id) => document.getElementById(id);
const on = (id, type, fn) => el(id).addEventListener(type, fn);
const query = new URLSearchParams(location.search);
const PACK = (query.get("pack") || "").trim();
const CLINIC = (query.get("clinic") || "").trim();
const IN_RATE = 16000;
const OUT_RATE = 24000;
const CONSENT_KEY = "shishu_consent";
const RECONNECT_DELAYS = [600, 1500, 3000];
const NUDGE_AFTER_MS = 3000;
const PROMPT_UNLOCK_MS = 5000;
const PLAYBACK_TAIL_SEC = 0.35; /* the last 350 ms of a completed turn: she may answer over it, it is not an echo */

const NAMES = (
  "status timer caption wait stage avatar-wrap thanks error thanks-sub call-btn no-btn yes-btn nudge-btn " +
  "consent-row consent-check yes-no-prime choice-pills share-pdf retry-pdf bug-dialog bug-chips bug-note bug-shot " +
  "bug-save bug-confirm bug-error brand-logo avatar-logo admin-link consent-link"
).split(" ");

const [
  statusEl, timerEl, captionEl, waitEl, stageEl, avatarEl, thanksEl, errorEl, subEl, callBtn, noBtn, yesBtn,
  nudgeBtn, consentRow, consentCheck, primeEl, pillsEl, shareBtn, retryBtn, bugDialog, bugChips, bugNote,
  bugShot, bugSave, bugOk, bugErr, brandLogo, avatarLogo, adminLink, consentLink
] = NAMES.map(el);

const micMeter = el("mic-meter");
const micLevel = el("mic-level");
let micSmooth = 0;

const bi = (hi, en) => ({ hi, en });
const T = {
  start: bi("कॉल शुरू करने के लिए नीचे दबाएँ", "Tap below to start the call"),
  listen: bi("सुन रही हूँ…", "Listening"),
  speak: bi("SHISHU बोल रही है…", "SHISHU is speaking"),
  think: bi("सोच रही हूँ…", "Thinking"),
  reconnect: bi("लाइन जुड़ रही है…", "Reconnecting"),
  cut: bi("कॉल कट गई। नीचे दबाकर फिर शुरू करें।", "The call disconnected. Start again."),
  failed: bi("कॉल में गड़बड़ी आ गई। फिर कोशिश करें।", "The call hit an error."),
  denied: bi("कॉल के लिए माइक्रोफ़ोन की अनुमति दें।", "Allow the microphone."),
  noStart: bi("कॉल शुरू नहीं हो पाई। फिर कोशिश करें।", "The call could not start."),
  needChip: "श्रेणी चुनें। Pick a category.",
  saveFail: "सेव नहीं हो पाया। कॉल खुली है। Could not save.",
  saved: "सेव हो गया। स्नैपशॉट कॉपी हो गया। Saved. Snapshot copied.",
  savedPartly: "सेव हो गया। कॉपी नहीं हो पाया। Saved. Could not copy the snapshot."
};

/* Page state. `epoch` is the turn the client is on; anything older is a stale tail. */
let phase = "idle";
let ws = null;
let ctx = null;
let mic = null;
let node = null;
let tick = 0;
let raf = 0;
let shotJob = null;
let queueFrom = 0;
let stepFrom = null;
let stepQueued = 0;
let stepDone = false;
let primed = false;
let ready = false;

let epoch = 0;
let minEpoch = 0;
let turnPhase = "listening";
let promptedAt = 0;

let hungUp = false;
let reconnectAttempt = 0;
let reconnectTimer = 0;

let callId = null;
let token = "";
let role = "user";
let pdf = null;
let startedAt = 0;
let shot = "";
let step = "";
let yesNo = false;
let options = [];
let capWho = "";
let capText = "";
let chip = "";
let waitingSince = 0;
let userSpoke = false;

const live = new Set();
const show = (n, v) => {
  n.hidden = !v;
};
const hide = (ns) => ns.forEach((n) => show(n, false));
const pick = (obj, path) => (path == null ? undefined : path.split(".").reduce((n, k) => (n == null ? undefined : n[k]), obj));
const open = () => ws != null && ws.readyState === WebSocket.OPEN;
const send = (m) => {
  if (open()) ws.send(JSON.stringify(m));
};
const playing = () => live.size > 0 || (ctx != null && ctx.currentTime < queueFrom);
/* The server closed this turn, so the browser already holds the whole line: its last 350 ms is hers to answer over. */
const tailOpen = () => stepDone && ctx != null && queueFrom - ctx.currentTime <= PLAYBACK_TAIL_SEC;
const unraf = () => {
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
};

function stopWaiting() {
  waitingSince = 0;
  userSpoke = false;
  show(nudgeBtn, false);
}

const clock = (sec) => `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
const say = (copy) => setLabels(statusEl, copy);

function warn(copy) {
  setLabels(errorEl, copy);
  show(errorEl, copy != null);
}

function setLabels(n, copy) {
  if (!n || copy == null) return;
  const v = typeof copy === "string" ? { hi: copy } : copy;
  n.replaceChildren();
  for (const [cls, text] of [["hi", v.hi], ["en", v.en]]) {
    if (text) {
      const span = document.createElement("span");
      span.className = cls;
      span.textContent = text;
      n.appendChild(span);
    }
  }
}

/* Consent is remembered for the tab, so a reload does not re-gate a caller mid-flow. */
function readConsent() {
  try {
    return Boolean(window.sessionStorage && window.sessionStorage.getItem(CONSENT_KEY) === "1");
  } catch (e) {
    return false;
  }
}

function writeConsent(on) {
  try {
    if (!window.sessionStorage) return;
    if (on) window.sessionStorage.setItem(CONSENT_KEY, "1");
    else window.sessionStorage.removeItem(CONSENT_KEY);
  } catch (e) {
    /* private mode or a blocked store: the tick still works for this page */
  }
}

function paint(view) {
  show(stageEl, view !== "thanks");
  show(thanksEl, view === "thanks");
  show(waitEl, view === "wait");
  stageEl.classList.toggle("working", view === "wait");
  avatarEl.classList.toggle("working", view === "wait");
  if (view !== "live") avatarEl.classList.remove("live", "speaking", "reconnecting");
  show(consentRow, view === "idle" || (view === "error" && !consentCheck.checked));
}

function runTimer(active) {
  const read = () => clock(Math.floor((Date.now() - startedAt) / 1000));
  if (!active) {
    if (tick) clearInterval(tick);
    tick = 0;
    if (startedAt) timerEl.textContent = read();
    return;
  }
  if (tick) return;
  startedAt = Date.now();
  tick = setInterval(() => {
    timerEl.textContent = read();
    if (phase !== "live" || playing()) return;
    /* A tap opens a short prompting window: her line may still be in flight. */
    if (turnPhase === "prompting" && Date.now() - promptedAt >= PROMPT_UNLOCK_MS) {
      turnPhase = "listening";
      say(T.listen);
    }
    if (turnPhase === "listening" && waitingSince && Date.now() - waitingSince > NUDGE_AFTER_MS) {
      show(nudgeBtn, true);
    }
  }, 250);
}

function flushAudio() {
  live.forEach((src) => {
    try {
      src.onended = null;
      src.stop(0);
      src.disconnect();
    } catch (e) {
      /* already gone */
    }
  });
  live.clear();
  queueFrom = 0;
  avatarEl.classList.remove("speaking");
}

function playPcm(b64, rate) {
  if (phase !== "live" || !ctx || !b64) return;
  stopWaiting();
  const pcm = new Int16Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer);
  if (!pcm.length) return;
  const buffer = ctx.createBuffer(1, pcm.length, rate);
  const at = Math.max(ctx.currentTime, queueFrom);
  buffer.getChannelData(0).set(Float32Array.from(pcm, (v) => v / 32768));
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start(at);
  queueFrom = at + buffer.duration;
  stepQueued += buffer.duration;
  if (stepFrom == null) stepFrom = at;
  live.add(src);
  avatarEl.classList.add("speaking");
  say(T.speak);
  scheduleReveal();
  src.onended = () => {
    live.delete(src);
    if (live.size) return;
    avatarEl.classList.remove("speaking");
    if (phase === "live") {
      if (turnPhase === "prompting") turnPhase = "listening";
      if (turnPhase === "listening") say(T.listen);
    }
    checkReveal();
  };
}

const WORKLET = `class Pcm16 extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(1024);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (const s of ch) {
        this.buf[this.n++] = s;
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm16-capture", Pcm16);`;

function rateFor(samples, from) {
  if (from === IN_RATE) return samples;
  const ratio = from / IN_RATE;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const at = Math.floor(i * ratio);
    out[i] = (samples[at] + samples[at + 1] + samples[at + 2]) / 3;
  }
  return out;
}

function onMicFrame(samples) {
  const rms = samples && samples.length ? Math.hypot(...samples) / Math.sqrt(samples.length) : 0;
  /* The turn is closed and only its tail is left: that window is hers, so the mic must be open in it. */
  if (phase === "live" && turnPhase === "prompting" && tailOpen()) {
    turnPhase = "listening";
    say(T.listen);
  }
  /* Half-duplex: she can be tapped over, never talked over. A muted frame never arms the nudge clock. */
  const micAllowed = phase === "live" && turnPhase === "listening" && (!playing() || tailOpen());
  micSmooth = micAllowed && rms ? micSmooth * 0.5 + rms * 0.5 : 0;
  const pct = Math.min(100, Math.round((micSmooth / 0.12) * 100));
  micLevel.style.width = `${pct}%`;
  if (micMeter) micMeter.setAttribute("aria-valuenow", String(pct));
  if (micAllowed && rms > 0.015) {
    userSpoke = true;
    waitingSince = 0;
    show(nudgeBtn, false);
  } else if (micAllowed && userSpoke) {
    userSpoke = false;
    waitingSince = Date.now();
  }
  if (!micAllowed || !open() || ws.bufferedAmount > 262144) return;
  const pcm = rateFor(samples, (ctx && ctx.sampleRate) || IN_RATE);
  if (!pcm.length) return;
  const bytes = new Uint8Array(pcm.length * 2);
  const view = new DataView(bytes.buffer);
  pcm.forEach((v, i) => view.setInt16(i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true));
  send({ type: "audio", data: btoa(String.fromCharCode(...bytes)), sample_rate: IN_RATE });
}

async function startMic() {
  const audio = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  mic = await navigator.mediaDevices.getUserMedia({ audio, video: false });
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
  node = new AudioWorkletNode(ctx, "pcm16-capture");
  node.port.onmessage = (e) => onMicFrame(e.data);
  ctx.createMediaStreamSource(mic).connect(node);
}

function stopMic() {
  if (node) {
    node.port.onmessage = null;
    try {
      node.disconnect();
    } catch (e) {
      /* already gone */
    }
  }
  if (mic) mic.getTracks().forEach((t) => t.stop());
  node = mic = null;
}

function hideQuick() {
  hide([noBtn, yesBtn, primeEl, pillsEl]);
  [noBtn, yesBtn].forEach((b) => {
    b.disabled = true;
    b.classList.remove("is-visible");
  });
  pillsEl.replaceChildren();
}

function revealQuick() {
  ready = true;
  unraf();
  if (!yesNo) {
    options.forEach((opt) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "choice-pill";
      pill.dataset.value = String(opt.value);
      setLabels(pill, { hi: opt.hi, en: opt.en });
      pill.addEventListener("click", () => answer(String(opt.value)));
      pillsEl.appendChild(pill);
    });
    show(pillsEl, options.length > 0);
    return;
  }
  [noBtn, yesBtn].forEach((b) => {
    show(b, true);
    b.disabled = false;
    b.classList.add("is-visible");
  });
  if (primed) return;
  primed = true;   // the first yes/no of this call, and only then
  show(primeEl, true);
}

function checkReveal() {
  if (phase !== "live" || ready || (!yesNo && !options.length)) return;
  const played = ctx && stepFrom != null ? Math.max(0, ctx.currentTime - stepFrom) : 0;
  if ((stepQueued > 0 && played >= stepQueued * 0.5) || (stepDone && !playing())) revealQuick();
  else scheduleReveal();
}

function scheduleReveal() {
  if (raf || ready) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    checkReveal();
  });
}

function resetQuick() {
  step = "";
  yesNo = false;
  options = [];
  ready = false;
  stepDone = false;
  stepFrom = null;
  stepQueued = 0;
  unraf();
  hideQuick();
}

/* A message from an older turn than the one on screen is a tail, never a prompt. */
function staleEpoch(m) {
  const msgEpoch = m && m.epoch != null ? Number(m.epoch) : epoch;
  return msgEpoch < minEpoch;
}

function questionState(m) {
  epoch = m.epoch != null ? Number(m.epoch) : epoch + 1;
  minEpoch = epoch;
  turnPhase = "prompting";
  promptedAt = Date.now();
  step = String(m.step || "");
  yesNo = Boolean(m.is_yes_no);
  options = Array.isArray(m.quick_replies) ? m.quick_replies : [];
  ready = false;
  stepDone = false;
  stepFrom = null;
  stepQueued = 0;
  unraf();
  stopWaiting();
  hideQuick();
  scheduleReveal();
}

function onAudio(m) {
  if (staleEpoch(m)) return;
  playPcm(m.data, Number(m.sample_rate) || OUT_RATE);
}

function botTurnComplete(m) {
  if (staleEpoch(m)) return;
  if (m.step && m.step !== step) return;
  stepDone = true;
  if ((!playing() || tailOpen()) && phase === "live") {
    turnPhase = "listening";
    say(T.listen);
  }
  checkReveal();
}

function onClear(m) {
  if (m && m.epoch != null && Number(m.epoch) < epoch) return;
  flushAudio();
  stopWaiting();
  if (phase === "live" && turnPhase === "listening") say(T.listen);
}

function onThinking(m) {
  if (staleEpoch(m)) return;
  turnPhase = "deciding";
  stopWaiting();
  if (phase === "live") say(T.think);
}

function answer(value) {
  if (phase !== "live" || !open() || !ready || epoch < minEpoch) return;
  ready = false;
  minEpoch = epoch + 1;
  turnPhase = "prompting";
  promptedAt = Date.now();
  waitingSince = Date.now();
  userSpoke = false;
  const chosen = options.find((o) => String(o.value) === value);
  hideQuick();
  flushAudio();
  show(nudgeBtn, false);
  say(T.listen);
  showCaption("user", chosen ? chosen.hi : value);
  send({ type: "quick_reply", value, step, epoch });
}

function showCaption(who, text) {
  const chunk = String(text || "");
  if (who !== capWho) {
    capWho = who;
    capText = "";
  }
  if (!capText || chunk.indexOf(capText) === 0) capText = chunk;
  else if (!capText.endsWith(chunk)) {
    const glue = /[\s।,.;:!?]$/.test(capText) || /^[\s।,.;:!?]/.test(chunk) ? "" : " ";
    capText += glue + chunk;
  }
  captionEl.textContent = capText ? `${who === "assistant" ? "SHISHU" : "You"}: ${capText}` : "";
}

function wsUrl(resumeToken) {
  const pairs = new URLSearchParams();
  if (PACK) pairs.set("pack", PACK);
  if (CLINIC) pairs.set("clinic", CLINIC);
  if (resumeToken) pairs.set("resume", resumeToken);
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/live${pairs.toString() ? `?${pairs}` : ""}`;
}

function adopt(msg) {
  if (msg.share_token) token = String(msg.share_token);
  if (msg.call_id != null) callId = msg.call_id;
}

const handlers = {
  session_started(m) {
    adopt(m);
    reconnectAttempt = 0;
    epoch = 0;
    minEpoch = 0;
    avatarEl.classList.remove("reconnecting");
    runTimer(true);
    if (phase === "live" && turnPhase === "listening" && !playing()) say(T.listen);
  },
  audio(m) { onAudio(m); },
  clear(m) { onClear(m); },
  transcript(m) { showCaption(m.role, m.text); },
  question_state(m) { questionState(m); },
  bot_turn_complete(m) { botTurnComplete(m); },
  thinking(m) { onThinking(m); },
  reconnecting() {
    avatarEl.classList.add("reconnecting");
    say(T.reconnect);
  },
  reconnected() {
    avatarEl.classList.remove("reconnecting");
    if (phase === "live" && !playing()) say(T.listen);
  },
  ending() { prepare(false); },
  call_completed(m) {
    adopt(m);
    showThanks();
    loadPdf("GET");
  },
  report_failed() {
    showThanks();
    show(retryBtn, true);
  },
  error() { warn(T.failed); }
};

/* A dropped socket mid-call is a reconnect, not a hang-up: keep the mic and the clock. */
function onClose(sock) {
  if (ws !== sock) return;
  ws = null;
  if (phase === "preparing") {
    showThanks();
    show(retryBtn, true);
    return;
  }
  if (phase === "live" && !hungUp && token && reconnectAttempt < RECONNECT_DELAYS.length) {
    flushAudio();
    stopWaiting();
    turnPhase = "prompting";
    promptedAt = Date.now();
    avatarEl.classList.add("reconnecting");
    say(T.reconnect);
    const delay = RECONNECT_DELAYS[reconnectAttempt];
    reconnectAttempt += 1;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = 0;
      if (phase === "live" && !hungUp) connectSocket(token);
    }, delay);
    return;
  }
  if (phase !== "live" && phase !== "connecting") return;
  phase = "error";
  stopMic();
  runTimer(false);
  resetQuick();
  hide([shareBtn, retryBtn]);
  paint("error");
  setCallButton(false);
  say(T.start);
  warn(T.cut);
}

function setCallButton(isLive) {
  callBtn.classList.toggle("end", isLive);
  setLabels(callBtn, isLive ? { hi: "बंद", en: "End" } : { hi: "शुरू", en: "Start" });
  callBtn.disabled = isLive ? false : !consentCheck.checked;
}

function connectSocket(resumeToken) {
  const sock = new WebSocket(wsUrl(resumeToken));
  ws = sock;
  sock.onopen = () => {
    if (ws !== sock) return;
    phase = "live";
    avatarEl.classList.add("live");
    if (!resumeToken) say(T.listen);
    runTimer(true);
  };
  sock.onmessage = (event) => {
    if (ws !== sock) return;
    let msg = null;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      return;
    }
    const fn = msg && Object.prototype.hasOwnProperty.call(handlers, msg.type) ? handlers[msg.type] : null;
    if (fn) fn(msg);
  };
  sock.onclose = () => onClose(sock);
  return sock;
}

async function startCall() {
  warn(null);
  phase = "connecting";
  hungUp = false;
  reconnectAttempt = 0;
  epoch = 0;
  minEpoch = 0;
  turnPhase = "listening";
  promptedAt = 0;
  setCallButton(true);
  say(T.listen);
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    ctx = new Ctx({ latencyHint: "interactive" });
    if (ctx.resume) ctx.resume().catch(() => {});
    await startMic();
    if (phase !== "connecting") return;
    connectSocket("");
  } catch (err) {
    phase = "error";
    stopMic();
    setCallButton(false);
    paint("error");
    say(T.start);
    warn(err && err.name === "NotAllowedError" ? T.denied : T.noStart);
  }
}

function prepare(wasHungUp) {
  if (phase === "preparing" || phase === "done") return;
  phase = "preparing";
  hungUp = true;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = 0;
  stopMic();
  runTimer(false);
  stopWaiting();
  resetQuick();
  avatarEl.classList.remove("speaking");
  paint("wait");
  if (wasHungUp) return;
  const beganAt = Date.now();
  let quietSince = 0;
  const poll = setInterval(() => {
    if (phase !== "preparing") return clearInterval(poll);
    if (playing()) quietSince = 0;
    else if (!quietSince) quietSince = Date.now();
    if ((quietSince && Date.now() - quietSince >= 2000) || Date.now() - beganAt >= 30000) {
      clearInterval(poll);
      send({ type: "hangup" });
    }
  }, 250);
}

function hangUp() {
  if (phase !== "live" && phase !== "connecting") return;
  hungUp = true;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = 0;
  send({ type: "hangup" });
  prepare(true);
}

function closeAudio() {
  hungUp = true;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = 0;
  stopMic();
  flushAudio();
  if (ctx) {
    try {
      ctx.close();
    } catch (e) {
      /* already closed */
    }
    ctx = null;
  }
  const sock = ws;
  ws = null;
  if (!sock) return;
  sock.onclose = sock.onmessage = null;
  try {
    sock.close();
  } catch (e) {
    /* already closed */
  }
}

async function loadPdf(method) {
  hide([shareBtn, retryBtn]);
  try {
    const url = `/api/calls/by-token/${encodeURIComponent(token)}/patient.pdf`;
    const res = await fetch(url, { method, credentials: "same-origin" });
    const cd = (res.headers && res.headers.get && res.headers.get("content-disposition")) || "";
    if (!res.ok) throw new Error("report");
    const m = /filename="([^"]+)"/.exec(cd);
    pdf = new File([await res.blob()], (m && m[1]) || "SHISHU.pdf", { type: "application/pdf" });
    show(shareBtn, true);
  } catch (e) {
    show(retryBtn, true);
  }
}

function savePdf() {
  if (!pdf) return;
  const url = URL.createObjectURL(pdf);
  const link = Object.assign(document.createElement("a"), { href: url, download: pdf.name || "SHISHU.pdf" });
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function sharePdf() {
  if (!pdf) return;
  if (token) {
    const url = `/api/calls/by-token/${encodeURIComponent(token)}/events`;
    const body = JSON.stringify({ event: "share_clicked" });
    fetch(url, { method: "POST", credentials: "same-origin", body }).catch(() => {});
  }
  if (!navigator.canShare || !navigator.canShare({ files: [pdf] })) return savePdf();
  navigator.share({ files: [pdf], title: "SHISHU" }).catch(savePdf);
}

function showThanks() {
  phase = "done";
  runTimer(false);
  closeAudio();
  paint("thanks");
  hide([shareBtn, retryBtn]);
  subEl.textContent = timerEl.textContent;
}

function resetForNewCall() {
  phase = "idle";
  hungUp = true;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = 0;
  reconnectAttempt = 0;
  callId = null;
  token = "";
  pdf = null;
  primed = false;
  startedAt = 0;
  tick = 0;
  epoch = 0;
  minEpoch = 0;
  turnPhase = "listening";
  promptedAt = 0;
  capWho = capText = "";
  captionEl.textContent = "";
  timerEl.textContent = "00:00";
  resetQuick();
  stopWaiting();
  consentCheck.checked = readConsent();
  paint("idle");
  hide([shareBtn, retryBtn]);
  warn(null);
  say(T.start);
  setCallButton(false);
}

function applyPack(pack) {
  const copy = pack.ui || {};
  const call = copy.call || {};
  const assets = pack.assets_url || {};
  if (call.start_prompt) T.start = call.start_prompt;
  if (call.listening) T.listen = call.listening;
  if (call.speaking) T.speak = call.speaking;
  if (call.thinking) T.think = call.thinking;
  if (call.reconnecting) T.reconnect = call.reconnecting;
  if (assets.logo) {
    brandLogo.src = avatarLogo.src = assets.logo;
    brandLogo.style.visibility = avatarLogo.style.visibility = "visible";
  }
  if (assets.consent_pdf) consentLink.href = assets.consent_pdf;
  role = pack.role || role;
  show(adminLink, role === "admin");
  setLabels(primeEl, copy.yes_no_prime);
  document.querySelectorAll("[data-ui]").forEach((n) => setLabels(n, pick(copy, n.getAttribute("data-ui"))));
  renderChips(pick(copy, "bug_categories.chips"));
}

function renderChips(chips) {
  bugChips.replaceChildren();
  (chips || []).forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.dataset.value = String(c.value);
    btn.setAttribute("aria-pressed", "false");
    setLabels(btn, { hi: c.hi, en: c.en });
    bugChips.appendChild(btn);
  });
}

async function hydrate() {
  let res = null;
  try {
    const url = `/api/pack${PACK ? `?pack=${encodeURIComponent(PACK)}` : ""}`;
    res = await fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" } });
  } catch (e) {
    return;
  }
  if (res.status === 401) return void (location.href = "/login");
  if (!res.ok) return;
  try {
    applyPack(await res.json());
  } catch (e) {
    /* keep the shipped copy */
  }
}

function setChip(value) {
  chip = value || "";
  bugChips.querySelectorAll(".chip").forEach((c) => {
    const lit = Boolean(chip) && c.dataset.value === chip;
    c.classList.toggle("on", lit);
    c.setAttribute("aria-pressed", lit ? "true" : "false");
  });
}

function shrink(file) {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const scale = Math.min(1, 900 / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(img.width * scale));
      canvas.height = Math.max(1, Math.round(img.height * scale));
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.55));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve("");
    };
    img.src = url;
  });
}

async function saveBug() {
  hide([bugOk, bugErr]);
  if (!chip) {
    bugErr.textContent = T.needChip;
    show(bugErr, true);
    return;
  }
  bugSave.disabled = true;
  try {
    if (shotJob) await shotJob;
    const payload = {
      call_id: callId != null ? Number(callId) : 0,
      share_token: token,
      category: chip,
      note: bugNote.value.trim(),
      call_elapsed_sec: startedAt ? Math.floor((Date.now() - startedAt) / 1000) : 0,
      client_state: { phase, step, is_yes_no: yesNo, epoch, turn_phase: turnPhase, caption: capText, screenshot: shot }
    };
    const res = await fetch("/api/bugs", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("save failed");
    let copied = true;
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    } catch (e) {
      copied = false;
    }
    bugNote.value = "";
    bugShot.value = "";
    shot = shotJob = null;
    setChip("");
    bugOk.textContent = copied ? T.saved : T.savedPartly;
    show(bugOk, true);
  } catch (err) {
    bugErr.textContent = T.saveFail;
    show(bugErr, true);
  } finally {
    bugSave.disabled = false;
  }
}

function pressCall() {
  if (phase === "live" || phase === "connecting") return hangUp();
  if (!consentCheck.checked) return;
  show(consentRow, false);
  primed = false;
  startCall();
}

function nudge() {
  if (phase !== "live" || turnPhase !== "listening") return;
  stopWaiting();
  send({ type: "nudge" });
}

on("consent-check", "change", () => {
  writeConsent(consentCheck.checked);
  if (phase === "idle" || phase === "error") setCallButton(false);
});
on("bug-btn", "click", () => {
  hide([bugOk, bugErr]);
  bugDialog.showModal();
});
on("bug-close", "click", () => bugDialog.close());
on("call-btn", "click", pressCall);
on("no-btn", "click", () => answer("clear"));
on("yes-btn", "click", () => answer("concern"));
on("nudge-btn", "click", nudge);
on("bug-save", "click", saveBug);
on("share-pdf", "click", sharePdf);
on("retry-pdf", "click", () => {
  show(retryBtn, false);
  loadPdf("POST");
});
on("new-call", "click", resetForNewCall);
on("logout-btn", "click", async () => {
  writeConsent(false);
  try {
    await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
  } catch (e) {
    /* go anyway */
  }
  location.href = "/login";
});
bugChips.addEventListener("click", (event) => {
  const target = event.target.closest("button[data-value]");
  if (target) setChip(target.dataset.value);
});
bugShot.addEventListener("change", () => {
  const file = bugShot.files && bugShot.files[0];
  shot = "";
  if (!file) {
    shotJob = null;
    return;
  }
  shotJob = shrink(file).then((data) => {
    shot = data;
    return data;
  });
});

resetForNewCall();
hydrate();
