(function () {
  const $ = function (id) { return document.getElementById(id); };
  const statusEl = $("status");
  const timerEl = $("timer");
  const captionEl = $("caption");
  const waitEl = $("wait");
  const callBtn = $("call-btn");
  const noBtn = $("no-btn");
  const yesBtn = $("yes-btn");
  const consentRow = $("consent-row");
  const consentCheck = $("consent-check");
  const stageEl = $("stage");
  const thanksEl = $("thanks");
  const summaryEl = $("summary");
  const sharePdfBtn = $("share-pdf");
  const retryPdfBtn = $("retry-pdf");
  const errorEl = $("error");
  const avatarWrap = $("avatar-wrap");
  const micMeter = $("mic-meter");
  const micLevel = $("mic-level");
  const nudgeBtn = $("nudge-btn");
  const bugOverlay = $("bug-overlay");
  const bugCats = $("bug-cats");
  const bugNote = $("bug-note");
  const bugShot = $("bug-shot");
  const bugSave = $("bug-save");
  const bugConfirm = $("bug-confirm");
  const bugError = $("bug-error");
  const hudToggle = $("hud-toggle");
  const hudEl = $("hud");

  const START_PROMPT = { hi: "कॉल शुरू करने के लिए नीचे दबाएँ", en: "Tap below to start the call" };
  const LISTENING_STATUS = { hi: "सुन रही हूँ…", en: "Listening" };
  const BUG_SAVE_ERR = "सेव नहीं हो पाया। कॉल खुली है। Could not save the bug. The call is still open.";
  const CLINIC_KEY = "shishu_clinic";
  const WORKLET_SOURCE = [
    "class PcmCapture extends AudioWorkletProcessor {",
    "  constructor() { super(); this.buf = new Float32Array(2048); this.used = 0; }",
    "  process(inputs) {",
    "    const ch = inputs[0] && inputs[0][0];",
    "    if (!ch) return true;",
    "    let offset = 0;",
    "    while (offset < ch.length) {",
    "      const n = Math.min(this.buf.length - this.used, ch.length - offset);",
    "      this.buf.set(ch.subarray(offset, offset + n), this.used);",
    "      this.used += n; offset += n;",
    "      if (this.used === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.used = 0; }",
    "    }",
    "    return true;",
    "  }",
    "}",
    "registerProcessor('pcm-capture', PcmCapture);"
  ].join("\n");

  let phase = "idle", flowGroup = "", questionStep = "";
  let expectYesNo = false, quickReplyReady = false, quickReplyPending = false;
  let promptChars = 0, stepAudioStart = 0, stepQueuedSec = 0, stepStreamDone = false, quickRevealRaf = 0;
  let captureRate = 16000, ws = null, audioCtx = null, micStream = null;
  let micSource = null, processor = null, workletNode = null, silentGain = null;
  let nextPlayTime = 0;
  const activeSources = new Set();
  let timerHandle = 0, startedAt = 0, frozenDuration = "00:00";
  let callId = null, shareToken = "", reportPending = false, hangupArmed = false;
  let wakeLock = null, pdfFile = null;
  let captionRole = "", captionText = "";
  let socketOpened = false, connectReject = null;
  let userRole = "user", callerName = "", modelName = "", voiceName = "";
  let lastTelemetry = null;
  const recentEvents = [];
  let userSpeaking = false, lastVoiceTs = 0, speechEndAt = 0;
  let nudgeWatch = 0, micSmooth = 0;
  let bugCategory = "", bugShotData = "", bugShotPending = null;
  let hudNoteTimer = 0, thinkingTimer = 0, callAttemptId = 0, hudInitialized = false;

  function setBilingual(node, value) {
    if (value && typeof value === "object") {
      node.innerHTML = '<span class="hi">' + value.hi + '</span>' +
        (value.en ? '<span class="en">' + value.en + '</span>' : "");
    } else {
      node.innerHTML = "";
      node.textContent = value || "";
    }
  }

  function setStatus(text, mode) {
    statusEl.hidden = false;
    setBilingual(statusEl, text);
    statusEl.className = mode || "";
  }

  function showError(message) {
    errorEl.hidden = false;
    setBilingual(errorEl, message);
  }

  function clearError() {
    errorEl.hidden = true;
    errorEl.textContent = "";
  }

  function clearNudge() {
    nudgeBtn.classList.remove("hot");
    nudgeBtn.hidden = true;
  }

  function leaveWorkingChrome() {
    stageEl.classList.remove("working");
    avatarWrap.classList.remove("working");
    waitEl.hidden = true;
  }

  function showPreparingChrome() {
    thanksEl.hidden = true;
    stageEl.hidden = false;
    setStatus("");
    stageEl.classList.add("working");
    avatarWrap.classList.add("working");
    waitEl.hidden = false;
    callBtn.disabled = true;
  }

  function clearThinkingWatch() {
    if (thinkingTimer) clearTimeout(thinkingTimer);
    thinkingTimer = 0;
  }

  function startThinkingWatch() {
    clearThinkingWatch();
    thinkingTimer = setTimeout(function () {
      if (phase === "live" && !isBotPlaying() && !avatarWrap.classList.contains("reconnecting")) {
        avatarWrap.classList.remove("speaking");
        avatarWrap.classList.add("thinking");
        setStatus({ hi: "सोच रही हूँ…", en: "Thinking" }, "thinking");
      }
    }, 900);
  }

  function clinicSlug() {
    let fromUrl = "";
    try {
      fromUrl = (new URLSearchParams(location.search).get("clinic") || "").trim().toLowerCase();
    } catch (err) {
      fromUrl = "";
    }
    if (/^[a-z0-9][a-z0-9_-]{0,39}$/.test(fromUrl)) {
      try { sessionStorage.setItem(CLINIC_KEY, fromUrl); } catch (err) { /* private mode */ }
      return fromUrl;
    }
    try {
      return (sessionStorage.getItem(CLINIC_KEY) || "").trim().toLowerCase();
    } catch (err) {
      return "";
    }
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const clinic = clinicSlug();
    return proto + "//" + location.host + "/ws/gemini" + (clinic ? ("?clinic=" + encodeURIComponent(clinic)) : "");
  }

  function wsOpen() {
    return ws && ws.readyState === WebSocket.OPEN;
  }

  function sendJson(payload) {
    if (wsOpen()) ws.send(JSON.stringify(payload));
  }

  async function acquireWakeLock() {
    if (!navigator.wakeLock || wakeLock) return;
    try {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", function () { wakeLock = null; });
    } catch (err) { /* call continues without screen lock */ }
  }

  function releaseWakeLock() {
    const lock = wakeLock;
    wakeLock = null;
    if (lock) lock.release().catch(function () {});
  }

  function loginRedirect() {
    location.href = "/login?next=" + encodeURIComponent((location.pathname + location.search) || "/");
  }

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function formatElapsed(ms) {
    const sec = Math.max(0, Math.floor(ms / 1000));
    return pad(Math.floor(sec / 60)) + ":" + pad(sec % 60);
  }

  function syncTimer() {
    frozenDuration = formatElapsed(Date.now() - startedAt);
    timerEl.textContent = frozenDuration;
  }

  function startTimer() {
    if (timerHandle) return;
    startedAt = Date.now();
    timerHandle = setInterval(syncTimer, 250);
  }

  function stopTimer() {
    if (!timerHandle) return;
    clearInterval(timerHandle);
    timerHandle = 0;
    if (startedAt) syncTimer();
  }

  function setCallButton(isEnd) {
    callBtn.innerHTML = isEnd
      ? '<span class="hi">बंद</span><span class="en">End</span>'
      : '<span class="hi">शुरू</span><span class="en">Start</span>';
    callBtn.classList.toggle("end", isEnd);
    callBtn.removeAttribute("aria-label");
    callBtn.disabled = isEnd ? false : !consentCheck.checked;
  }

  function showConsentRow(clear) {
    consentRow.hidden = false;
    if (clear) consentCheck.checked = false;
    if (phase === "idle" || phase === "error") setCallButton(false);
  }

  function bytesToBase64(bytes) {
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function base64ToBytes(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  function floatToPcm16LE(float32) {
    const buffer = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      const val = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
      view.setInt16(i * 2, Math.max(-32768, Math.min(32767, val)), true);
    }
    return new Uint8Array(buffer);
  }

  function downsampleBuffer(buffer, sampleRate, outSampleRate) {
    if (!buffer || !buffer.length) return new Float32Array(0);
    if (outSampleRate === sampleRate) return buffer;
    const ratio = sampleRate / outSampleRate;
    const result = new Float32Array(Math.round(buffer.length / ratio));
    let offsetResult = 0, offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult++] = count > 0 ? (accum / count) : 0;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  function rmsOf(float32) {
    if (!float32 || !float32.length) return 0;
    let sum = 0;
    for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
    return Math.sqrt(sum / float32.length);
  }

  function paintMic(level) {
    const pct = Math.max(0, Math.min(100, Math.round((level / 0.12) * 100)));
    micLevel.style.width = pct + "%";
    micMeter.setAttribute("aria-valuenow", String(pct));
  }

  function refreshNudgeHighlight() {
    const stuck = phase === "live" && speechEndAt > 0 && activeSources.size === 0 && (Date.now() - speechEndAt) > 3000;
    if (stuck) {
      nudgeBtn.hidden = false;
      nudgeBtn.classList.add("hot");
    } else {
      clearNudge();
    }
  }

  function syncQuickButtons() {
    const live = phase === "live";
    const show = live && expectYesNo && quickReplyReady;
    [noBtn, yesBtn].forEach(function (btn) {
      btn.hidden = !live;
      btn.classList.toggle("is-visible", show);
      btn.disabled = !show || quickReplyPending;
    });
  }

  function cancelQuickRevealRaf() {
    if (quickRevealRaf) cancelAnimationFrame(quickRevealRaf);
    quickRevealRaf = 0;
  }

  function scheduleQuickRevealCheck() {
    if (quickRevealRaf) return;
    quickRevealRaf = requestAnimationFrame(function () {
      quickRevealRaf = 0;
      checkQuickReveal();
    });
  }

  function checkQuickReveal() {
    if (!(phase === "live" && expectYesNo && !quickReplyReady && (stepAudioStart > 0 || stepStreamDone))) return;
    const playedSec = (audioCtx && stepAudioStart > 0) ? Math.max(0, audioCtx.currentTime - stepAudioStart) : 0;
    const expectedSec = stepStreamDone ? stepQueuedSec : Math.max(stepQueuedSec, promptChars / 12);
    if (
      (expectedSec > 0 && playedSec >= expectedSec * 0.5 && (stepStreamDone || playedSec >= 0.65)) ||
      (stepStreamDone && !isBotPlaying())
    ) {
      quickReplyReady = true;
      cancelQuickRevealRaf();
      syncQuickButtons();
      return;
    }
    if (isBotPlaying() || !stepStreamDone) scheduleQuickRevealCheck();
  }

  function resetQuickState() {
    questionStep = "";
    expectYesNo = false;
    quickReplyReady = false;
    quickReplyPending = false;
    stepAudioStart = 0;
    stepQueuedSec = 0;
    stepStreamDone = false;
    cancelQuickRevealRaf();
    syncQuickButtons();
  }

  function leaveLiveChrome() {
    if (nudgeWatch) clearInterval(nudgeWatch);
    nudgeWatch = 0;
    clearThinkingWatch();
    avatarWrap.classList.remove("live", "speaking", "thinking", "reconnecting");
    clearNudge();
    userSpeaking = false;
    speechEndAt = 0;
    micSmooth = 0;
    paintMic(0);
    resetQuickState();
  }

  function sendQuickReply(outcome) {
    if (phase !== "live" || !expectYesNo || !quickReplyReady || quickReplyPending || !questionStep || !wsOpen()) return;
    quickReplyPending = true;
    syncQuickButtons();
    flushPlayback();
    speechEndAt = 0;
    clearNudge();
    showCaption("user", outcome === "clear" ? "नहीं।" : "हाँ।");
    startThinkingWatch();
    sendJson({ type: "quick_reply", outcome: outcome, step: questionStep });
  }

  function isBotPlaying() {
    return activeSources.size > 0 || (audioCtx && audioCtx.currentTime < nextPlayTime);
  }

  function trackSpeech(rms) {
    if (phase !== "live") return;
    const now = Date.now();
    if (rms > 0.011) {
      userSpeaking = true;
      lastVoiceTs = now;
      speechEndAt = 0;
      clearNudge();
      return;
    }
    if (!isBotPlaying() && userSpeaking && (now - lastVoiceTs) > 550) {
      userSpeaking = false;
      speechEndAt = now;
      refreshNudgeHighlight();
      if (wsOpen()) {
        sendJson({ type: "speech_end" });
        startThinkingWatch();
      }
    }
  }

  function onMicFrame(float32) {
    const rms = rmsOf(float32);
    micSmooth = micSmooth * 0.5 + rms * 0.5;
    paintMic(micSmooth);
    trackSpeech(rms);
    if (phase !== "live" || !wsOpen() || ws.bufferedAmount > 262144) return;
    const down = downsampleBuffer(float32, audioCtx ? audioCtx.sampleRate : captureRate, captureRate);
    if (!down.length) return;
    sendJson({ type: "audio", data: bytesToBase64(floatToPcm16LE(down)), sample_rate: captureRate });
  }

  function flushPlayback() {
    clearThinkingWatch();
    activeSources.forEach(function (src) {
      try { src.onended = null; } catch (err) { /* already gone */ }
      try { src.stop(0); } catch (err) { /* not started yet */ }
      try { src.disconnect(); } catch (err) { /* already disconnected */ }
    });
    activeSources.clear();
    nextPlayTime = 0;
    avatarWrap.classList.remove("speaking", "thinking");
  }

  function playPcm(b64, sampleRate) {
    if (!b64 || !audioCtx || phase !== "live") return;
    const bytes = base64ToBytes(b64);
    const usable = bytes.byteLength - (bytes.byteLength % 2);
    if (usable < 2) return;
    const view = new DataView(bytes.buffer, bytes.byteOffset, usable);
    const floats = new Float32Array(usable / 2);
    for (let i = 0; i < floats.length; i++) floats[i] = view.getInt16(i * 2, true) / 32768;
    let rateOut = Number(sampleRate);
    if (!rateOut || rateOut < 8000 || rateOut > 96000) rateOut = 24000;
    if (audioCtx.state === "suspended") audioCtx.resume().catch(function () {});
    const audioBuffer = audioCtx.createBuffer(1, floats.length, rateOut);
    audioBuffer.getChannelData(0).set(floats);
    const src = audioCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(audioCtx.destination);
    if (!isBotPlaying()) userSpeaking = false;
    clearThinkingWatch();
    avatarWrap.classList.remove("thinking", "reconnecting");
    speechEndAt = 0;
    clearNudge();
    const startAt = Math.max(audioCtx.currentTime, nextPlayTime);
    stepQueuedSec += audioBuffer.duration;
    if (stepAudioStart === 0 && stepQueuedSec >= 0.05) stepAudioStart = startAt;
    src.start(startAt);
    nextPlayTime = startAt + audioBuffer.duration;
    activeSources.add(src);
    avatarWrap.classList.add("speaking");
    setStatus({ hi: "SHISHU बोल रही है…", en: "SHISHU is speaking" });
    if (expectYesNo && !quickReplyReady) scheduleQuickRevealCheck();
    src.onended = function () {
      activeSources.delete(src);
      if (activeSources.size === 0 && phase === "live") {
        avatarWrap.classList.remove("speaking", "thinking");
        if (!avatarWrap.classList.contains("reconnecting")) setStatus(LISTENING_STATUS);
      }
      checkQuickReveal();
    };
  }

  function mergeChunk(prev, chunk) {
    chunk = String(chunk || "");
    if (!prev || chunk.indexOf(prev) === 0) return chunk;
    if (!chunk || prev.slice(-chunk.length) === chunk) return prev;
    const needsSpace = !/\s$/.test(prev) && !/^[\s,.;:!?।]/.test(chunk);
    return prev + (needsSpace ? " " : "") + chunk;
  }

  function showCaption(role, text) {
    if (role !== captionRole) {
      captionRole = role || "";
      captionText = "";
    }
    captionText = mergeChunk(captionText, text);
    captionEl.textContent = captionText ? ((role === "assistant" ? "SHISHU" : "You") + ": " + captionText) : "";
  }

  function dash(value) {
    return value === 0 || value ? String(value) : "—";
  }

  function msText(value) {
    return (value === 0 || (typeof value === "number" && isFinite(value))) ? (String(Math.round(value)) + " ms") : "—";
  }

  function lastLatency(telemetry) {
    const samples = telemetry && telemetry.turn_latencies_ms;
    return (Array.isArray(samples) && samples.length) ? msText(samples[samples.length - 1]) : "—";
  }

  function eventClock(t) {
    if (typeof t !== "number" || !isFinite(t)) return "";
    const when = new Date(t > 1e12 ? t : t * 1000);
    if (isNaN(when.getTime())) return String(t);
    return pad(when.getHours()) + ":" + pad(when.getMinutes()) + ":" + pad(when.getSeconds());
  }

  function pushEvent(evt) {
    if (!evt || typeof evt !== "object") return;
    const prev = recentEvents[recentEvents.length - 1];
    if (prev && prev.t === evt.t && prev.type === evt.type && prev.detail === evt.detail) return;
    recentEvents.push(evt);
    if (recentEvents.length > 10) recentEvents.splice(0, recentEvents.length - 10);
  }

  function renderHud() {
    const t = lastTelemetry || {};
    const fields = {
      "hud-call": callId != null ? ("#" + callId) : "—",
      "hud-caller": callerName || "—",
      "hud-role": userRole || "—",
      "hud-model": modelName || "—",
      "hud-voice": voiceName || "—",
      "hud-connect": msText(t.connect_ms),
      "hud-last-lat": lastLatency(lastTelemetry),
      "hud-avg-lat": msText(t.avg_latency_ms),
      "hud-turns": dash(t.user_turns) + " / " + dash(t.bot_turns),
      "hud-tools": dash(t.tool_calls),
      "hud-nudges": dash(t.watchdog_nudges),
      "hud-audio": dash(t.audio_chunks_in) + " / " + dash(t.audio_chunks_out),
      "hud-step": flowGroup || "—"
    };
    Object.keys(fields).forEach(function (id) {
      const node = $(id);
      if (node) node.textContent = fields[id];
    });
    const eventsEl = $("hud-events");
    eventsEl.replaceChildren();
    recentEvents.forEach(function (evt) {
      const li = document.createElement("li");
      const bits = [eventClock(evt.t), evt.type, evt.detail].filter(Boolean);
      li.textContent = bits.join(" · ") || "event";
      eventsEl.appendChild(li);
    });
  }

  function applyConfig(cfg) {
    if (!cfg) return;
    if (cfg.role) userRole = cfg.role;
    if (cfg.username) callerName = cfg.username;
    const isAdmin = userRole === "admin";
    hudToggle.hidden = !isAdmin;
    $("admin-link").hidden = !isAdmin;
    $("admin-view").hidden = !isAdmin;
    if (isAdmin && !hudInitialized) {
      hudEl.hidden = false;
      hudInitialized = true;
    }
    renderHud();
  }

  function showHudNote(text) {
    const note = $("hud-send-note");
    note.textContent = text;
    note.hidden = false;
    if (hudNoteTimer) clearTimeout(hudNoteTimer);
    hudNoteTimer = setTimeout(function () { note.hidden = true; }, 2200);
  }

  function disconnectNode(node) {
    if (!node) return;
    try { node.disconnect(); } catch (err) { /* ignore */ }
  }

  function stopCapture() {
    if (processor) processor.onaudioprocess = null;
    if (workletNode) workletNode.port.onmessage = null;
    [processor, workletNode, micSource, silentGain].forEach(disconnectNode);
    processor = workletNode = micSource = silentGain = null;
    if (micStream) {
      micStream.getTracks().forEach(function (track) { track.stop(); });
      micStream = null;
    }
  }

  function closeAudioContext() {
    stopCapture();
    flushPlayback();
    if (audioCtx) {
      try { audioCtx.close(); } catch (err) { /* ignore */ }
      audioCtx = null;
    }
  }

  function closeSocket() {
    const reject = connectReject;
    connectReject = null;
    if (ws) {
      const sock = ws;
      ws = null;
      sock.onopen = sock.onmessage = sock.onerror = sock.onclose = null;
      try { sock.close(); } catch (err) { /* ignore */ }
    }
    if (reject) reject(new Error("cancelled"));
  }

  const socketHandlers = {
    session_started: function (msg) {
      callId = msg.call_id;
      rememberShareToken(msg);
      if (msg.caller_name) callerName = msg.caller_name;
      if (msg.user_role) userRole = msg.user_role;
      if (msg.model) modelName = msg.model;
      if (msg.voice) voiceName = msg.voice;
      renderHud();
      if (phase === "live") setStatus(LISTENING_STATUS);
      startTimer();
    },
    audio: function (msg) {
      try { playPcm(msg.data, msg.sample_rate || 24000); } catch (err) { /* skip a bad chunk */ }
    },
    clear: function () {
      flushPlayback();
      if (quickReplyPending) startThinkingWatch();
      speechEndAt = 0;
      clearNudge();
      if (phase === "live" && !avatarWrap.classList.contains("reconnecting")) setStatus(LISTENING_STATUS);
    },
    reconnecting: function () {
      clearThinkingWatch();
      avatarWrap.classList.remove("speaking", "thinking");
      avatarWrap.classList.add("reconnecting");
      setStatus({ hi: "लाइन जुड़ रही है…", en: "Reconnecting" }, "reconnecting");
    },
    reconnected: function () {
      avatarWrap.classList.remove("reconnecting");
      if (phase === "live" && !isBotPlaying()) setStatus(LISTENING_STATUS);
    },
    transcript: function (msg) {
      showCaption(msg.role, msg.text);
    },
    telemetry: function (msg) {
      lastTelemetry = msg.telemetry || {};
      pushEvent(msg.last_event);
      renderHud();
    },
    bot_turn_complete: function (msg) {
      if (!msg.step || msg.step === questionStep) {
        stepStreamDone = true;
        if (stepAudioStart === 0 && stepQueuedSec > 0 && audioCtx) stepAudioStart = audioCtx.currentTime;
        checkQuickReveal();
      }
    },
    question_state: function (msg) {
      questionStep = String(msg.step || "");
      expectYesNo = Boolean(msg.expect_yes_no);
      promptChars = Number(msg.prompt_chars) || 0;
      quickReplyReady = quickReplyPending = stepStreamDone = false;
      stepAudioStart = stepQueuedSec = 0;
      cancelQuickRevealRaf();
      if (msg.group) {
        flowGroup = String(msg.group);
        renderHud();
      }
      syncQuickButtons();
    },
    ending: function () {
      beginPreparing();
      hangupWhenQuiet();
    },
    call_completed: function (msg) {
      rememberShareToken(msg);
      revealReadyReport(msg.call_id);
    },
    report_failed: function (msg) {
      showReportRetry(msg.call_id);
    },
    error: function () {
      showError({
        hi: "कॉल में गड़बड़ी आ गई। नीचे दबाकर फिर कोशिश करें।",
        en: "The call hit an error. Tap below to try again."
      });
    }
  };

  function onSocketMessage(event) {
    let msg;
    try { msg = JSON.parse(event.data); } catch (err) { return; }
    if (!msg || typeof msg.type !== "string") return;
    const handler = Object.prototype.hasOwnProperty.call(socketHandlers, msg.type) && socketHandlers[msg.type];
    if (handler) handler(msg);
  }

  function onSocketClose() {
    const reject = connectReject;
    connectReject = null;
    ws = null;
    if (reject) reject(new Error("closed"));
    if (phase === "done") return;
    if (phase === "preparing") {
      if (!reportPending) showReportRetry(callId);
      return;
    }
    if (phase === "live" || phase === "connecting") {
      callAttemptId++;
      returnToStartError({
        hi: "कॉल कट गई। नीचे दबाकर फिर शुरू करें।",
        en: "The call disconnected. Tap below to start again."
      });
    }
  }

  function returnToStartError(errMsg) {
    phase = "error";
    stopTimer();
    closeAudioContext();
    closeSocket();
    leaveLiveChrome();
    leaveWorkingChrome();
    releaseWakeLock();
    setStatus(START_PROMPT);
    showConsentRow();
    showError(errMsg);
  }

  function openSocket() {
    socketOpened = false;
    return new Promise(function (resolve, reject) {
      connectReject = reject;
      const sock = new WebSocket(wsUrl());
      ws = sock;
      sock.onopen = function () {
        if (ws !== sock) return;
        socketOpened = true;
        connectReject = null;
        resolve();
      };
      sock.onmessage = function (event) { if (ws === sock) onSocketMessage(event); };
      sock.onerror = function () {
        if (ws !== sock || socketOpened) return;
        const r = connectReject;
        connectReject = null;
        if (r) r(new Error("Could not connect to the voice service."));
      };
      sock.onclose = function () { if (ws === sock) onSocketClose(); };
    });
  }

  function requestMic() {
    return navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false
    });
  }

  async function wireMic() {
    if (!audioCtx || audioCtx.state === "closed") throw new Error("Microphone capture was interrupted.");
    if (audioCtx.state === "suspended") await audioCtx.resume().catch(function () {});
    audioCtx.onstatechange = function () {
      if (phase === "live" && audioCtx && audioCtx.state === "suspended") audioCtx.resume().catch(function () {});
    };
    micSource = audioCtx.createMediaStreamSource(micStream);
    const primaryTrack = micStream.getAudioTracks()[0];
    if (primaryTrack) {
      primaryTrack.addEventListener("ended", function () {
        if (phase !== "live") return;
        requestMic().then(function (newStream) {
          if (phase !== "live" || !audioCtx) {
            newStream.getTracks().forEach(function (t) { t.stop(); });
            return;
          }
          disconnectNode(micSource);
          micStream = newStream;
          micSource = audioCtx.createMediaStreamSource(newStream);
          if (workletNode) micSource.connect(workletNode);
          else if (processor) micSource.connect(processor);
        }).catch(function () {
          showError({ hi: "माइक्रोफ़ोन कट गया। फिर जोड़ें, या कॉल बंद करें।", en: "The microphone disconnected. Reconnect it, or end the call." });
        });
      });
    }
    if (audioCtx.audioWorklet) {
      const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
      try {
        await audioCtx.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");
      workletNode.port.onmessage = function (event) { onMicFrame(event.data); };
      micSource.connect(workletNode);
    } else if (typeof audioCtx.createScriptProcessor === "function") {
      processor = audioCtx.createScriptProcessor(2048, 1, 1);
      processor.onaudioprocess = function (event) { onMicFrame(event.inputBuffer.getChannelData(0)); };
      micSource.connect(processor);
      silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(audioCtx.destination);
    } else {
      throw new Error("This browser cannot capture microphone audio.");
    }
  }

  function clearSessionData() {
    stopTimer();
    startedAt = 0;
    frozenDuration = "00:00";
    timerEl.textContent = "00:00";
    flowGroup = shareToken = modelName = voiceName = captionRole = captionText = "";
    captionEl.textContent = "";
    hangupArmed = reportPending = userSpeaking = false;
    callId = lastTelemetry = pdfFile = null;
    recentEvents.length = 0;
    lastVoiceTs = speechEndAt = 0;
    resetQuickState();
    clearError();
    renderHud();
  }

  async function startCall(attemptId) {
    function isCurrent() {
      return attemptId === callAttemptId && phase === "connecting";
    }
    function stale() {
      if (isCurrent()) return false;
      if (attemptId === callAttemptId) {
        closeAudioContext();
        closeSocket();
      }
      return true;
    }
    clearSessionData();
    setStatus(LISTENING_STATUS);
    acquireWakeLock();
    try {
      if (audioCtx && audioCtx.state !== "closed") {
        try { audioCtx.close(); } catch (e) { /* ignore */ }
      }
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error("AudioContext unavailable");
      audioCtx = new AudioContextClass({ latencyHint: "interactive" });
      audioCtx.resume().catch(function () {});
      const micPromise = requestMic();
      const configPromise = fetch("/api/config", {
        credentials: "same-origin",
        headers: { "Accept": "application/json" }
      });
      const stream = await micPromise;
      if (!isCurrent()) {
        stream.getTracks().forEach(function (track) { track.stop(); });
        if (attemptId === callAttemptId) closeAudioContext();
        return;
      }
      micStream = stream;
      const res = await configPromise;
      if (stale()) return;
      if (res.status === 401) {
        closeAudioContext();
        loginRedirect();
        return;
      }
      if (!res.ok) throw new Error("Could not read the call configuration.");
      const cfg = await res.json();
      if (stale()) return;
      applyConfig(cfg);
      await wireMic();
      if (stale()) return;
      await openSocket();
      if (stale()) return;
      phase = "live";
      syncQuickButtons();
      avatarWrap.classList.add("live");
      if (!nudgeWatch) nudgeWatch = setInterval(refreshNudgeHighlight, 250);
      setStatus(LISTENING_STATUS);
      startTimer();
      setCallButton(true);
    } catch (err) {
      if (!isCurrent()) return;
      const denied = err && (err.name === "NotAllowedError" || err.name === "NotFoundError");
      returnToStartError(denied
        ? { hi: "कॉल के लिए माइक्रोफ़ोन की अनुमति दें।", en: "Allow the microphone to start the call." }
        : { hi: "कॉल शुरू नहीं हो पाई। फिर कोशिश करें।", en: "The call could not start. Try again." });
    }
  }

  function beginPreparing() {
    if (phase === "preparing" || phase === "done") return;
    phase = "preparing";
    stopCapture();
    stopTimer();
    leaveLiveChrome();
    showPreparingChrome();
  }

  function hangupWhenQuiet() {
    if (hangupArmed || phase !== "preparing") return;
    hangupArmed = true;
    const waitStartedAt = Date.now();
    let quietSince = 0;
    const timer = setInterval(function () {
      if (phase !== "preparing") { clearInterval(timer); return; }
      if (isBotPlaying()) quietSince = 0;
      else if (!quietSince) quietSince = Date.now();
      if ((quietSince && Date.now() - quietSince >= 2500) || Date.now() - waitStartedAt >= 30000) {
        clearInterval(timer);
        try { sendJson({ type: "hangup" }); } catch (err) { /* server also ends */ }
      }
    }, 250);
  }

  function endCall() {
    if (phase !== "live" && phase !== "connecting") return;
    callAttemptId++;
    const open = wsOpen();
    closeAudioContext();
    stopTimer();
    if (!open) {
      phase = "idle";
      leaveLiveChrome();
      releaseWakeLock();
      closeSocket();
      setStatus(START_PROMPT);
      leaveWorkingChrome();
      showConsentRow();
      return;
    }
    hangupArmed = true;
    beginPreparing();
    try {
      ws.send(JSON.stringify({ type: "hangup" }));
    } catch (err) {
      returnToStartError({
        hi: "कॉल बंद नहीं हो पाई। नीचे दबाकर फिर कोशिश करें।",
        en: "The call could not end cleanly. Tap below to try again."
      });
    }
  }

  function rememberShareToken(msg) {
    if (msg && msg.share_token) shareToken = String(msg.share_token);
  }

  function motherPdfUrl() {
    if (shareToken) return "/api/calls/by-token/" + encodeURIComponent(shareToken) + "/mother.pdf";
    if (userRole === "admin" && callId != null) return "/api/calls/" + encodeURIComponent(callId) + "/mother.pdf";
    return "";
  }

  function pdfName(res) {
    const match = /filename="?([^";]+)"?/.exec(res.headers.get("Content-Disposition") || "");
    return (match && match[1]) || "SHISHU.pdf";
  }

  function loadPdf(method) {
    const attemptId = callAttemptId;
    const url = motherPdfUrl();
    if (!url) return Promise.reject(new Error("pdf"));
    return fetch(url, { method: method, credentials: "same-origin" }).then(function (res) {
      if (attemptId !== callAttemptId) throw new Error("cancelled");
      if (!res.ok) throw new Error("pdf");
      const name = pdfName(res);
      return res.blob().then(function (blob) {
        if (attemptId !== callAttemptId) throw new Error("cancelled");
        return new File([blob], name, { type: "application/pdf" });
      });
    });
  }

  function fetchMotherPdf() {
    return loadPdf("GET");
  }

  function requestReport() {
    const attemptId = callAttemptId;
    return loadPdf("POST").then(function (file) {
      if (attemptId !== callAttemptId) throw new Error("cancelled");
      return file;
    });
  }

  function showReportRetry(id, expectedAttemptId) {
    const attemptId = expectedAttemptId !== undefined ? expectedAttemptId : callAttemptId;
    if (attemptId !== callAttemptId) return;
    if (id != null) callId = id;
    showThanks({ call_id: callId, pdf_failed: true });
  }

  function awaitPdfPromise(attemptId, loader) {
    reportPending = true;
    loader().then(function (file) {
      if (attemptId !== callAttemptId) return;
      reportPending = false;
      showThanks({ call_id: callId, pdfFile: file });
    }).catch(function () {
      if (attemptId !== callAttemptId) return;
      reportPending = false;
      showReportRetry(callId, attemptId);
    });
  }

  function revealReadyReport(id) {
    const attemptId = callAttemptId;
    if (id != null) callId = id;
    if (callId == null) {
      showReportRetry(null, attemptId);
      return;
    }
    if (phase !== "preparing" && phase !== "done") beginPreparing();
    awaitPdfPromise(attemptId, fetchMotherPdf);
  }

  function logShareClick() {
    if (!shareToken) return;
    fetch("/api/calls/by-token/" + encodeURIComponent(shareToken) + "/events", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ event: "share_clicked" })
    }).catch(function () {});
  }

  function downloadPdf() {
    if (!pdfFile) return;
    const url = URL.createObjectURL(pdfFile);
    const link = document.createElement("a");
    link.href = url;
    link.download = pdfFile.name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function sharePdf() {
    if (!pdfFile) return;
    logShareClick();
    if (navigator.canShare && navigator.canShare({ files: [pdfFile] })) {
      navigator.share({ files: [pdfFile], title: "SHISHU" }).catch(function (err) {
        if (!err || err.name === "AbortError") return;
        showError({ hi: "इस फ़ोन पर शेयर नहीं खुल पाया। फ़ाइल डाउनलोड हो रही है।", en: "Share did not open. The file is downloading." });
        downloadPdf();
      });
      return;
    }
    downloadPdf();
  }

  function showThanks(msg) {
    phase = "done";
    releaseWakeLock();
    stopCapture();
    flushPlayback();
    stopTimer();
    leaveLiveChrome();
    leaveWorkingChrome();
    callId = msg.call_id != null ? msg.call_id : callId;
    stageEl.hidden = true;
    const bits = [];
    if (callId != null && userRole === "admin") bits.push("Call #" + callId);
    bits.push(frozenDuration);
    $("thanks-sub").textContent = bits.join(" · ");
    $("admin-view").href = callId != null ? ("/admin?call=" + encodeURIComponent(callId)) : "/admin";
    pdfFile = msg.pdfFile || null;
    if ($("summary")) $("summary").hidden = !pdfFile;
    sharePdfBtn.hidden = !pdfFile;
    retryPdfBtn.hidden = !msg.pdf_failed;
    thanksEl.hidden = false;
    renderHud();
    clearError();
  }

  function resetForNewCall() {
    callAttemptId++;
    phase = "idle";
    closeSocket();
    closeAudioContext();
    leaveLiveChrome();
    leaveWorkingChrome();
    releaseWakeLock();
    clearSessionData();
    if ($("summary")) $("summary").hidden = false;
    sharePdfBtn.hidden = true;
    retryPdfBtn.hidden = true;
    thanksEl.hidden = true;
    stageEl.hidden = false;
    setStatus(START_PROMPT);
    showConsentRow(true);
  }

  function setPageInert(on) {
    if (!document.querySelector) return;
    ["header", "main"].forEach(function (sel) {
      const node = document.querySelector(sel);
      if (node) node.inert = !!on;
    });
  }

  function focusNode(node) {
    if (!node || typeof node.focus !== "function") return;
    try { node.focus(); } catch (err) { /* a test DOM may mock focus */ }
  }

  function closeBug() {
    bugOverlay.hidden = true;
    setPageInert(false);
    focusNode($("bug-btn"));
  }

  function setBugCategory(cat) {
    bugCategory = cat || "";
    bugCats.querySelectorAll(".chip").forEach(function (btn) {
      const on = Boolean(bugCategory && btn.getAttribute("data-cat") === bugCategory);
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function resetBugDraft() {
    bugNote.value = "";
    bugShotPending = null;
    if (bugShot) bugShot.value = "";
    bugShotData = "";
    setBugCategory("");
  }

  function openBug() {
    bugConfirm.hidden = true;
    bugError.hidden = true;
    bugOverlay.hidden = false;
    setPageInert(true);
    const chip = bugCats && typeof bugCats.querySelector === "function" ? bugCats.querySelector(".chip") : null;
    focusNode(chip || $("bug-close"));
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    if (!ok) throw new Error("copy failed");
  }

  function showBugError(msg) {
    bugError.textContent = msg;
    bugError.hidden = false;
  }

  async function saveBug() {
    bugConfirm.hidden = true;
    bugError.hidden = true;
    if (!bugCategory) {
      showBugError("श्रेणी चुनें। Pick a category.");
      return;
    }
    bugSave.disabled = true;
    try {
      if (bugShotPending) await bugShotPending;
      const payload = {
        call_id: callId != null ? Number(callId) : 0,
        share_token: shareToken,
        category: bugCategory,
        note: bugNote.value.trim(),
        call_elapsed_sec: startedAt ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000)) : 0,
        client_state: {
          phase: phase,
          flowGroup: flowGroup,
          captureRate: captureRate,
          captionText: captionText,
          lastTelemetry: lastTelemetry,
          recentEvents: recentEvents.slice(),
          screenshot: bugShotData || ""
        }
      };
      const snapshot = JSON.stringify(payload, null, 2);
      const res = await fetch("/api/bugs", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        if (res.status === 401) {
          showBugError("फिर साइन-इन करें। कॉल खुली है। Sign in again to save this bug. The call is still open.");
        } else if (res.status === 413) {
          showBugError("रिपोर्ट बहुत बड़ी है। छोटी फोटो चुनें या फोटो हटाएँ। Bug report is too large. Choose a smaller screenshot or clear it.");
        } else {
          showBugError(BUG_SAVE_ERR);
        }
        return;
      }
      let copied = true;
      try { await copyText(snapshot); } catch (err) { copied = false; }
      resetBugDraft();
      bugConfirm.textContent = copied
        ? "सेव हो गया। स्नैपशॉट कॉपी हो गया। Saved. Snapshot copied."
        : "सेव हो गया। स्नैपशॉट कॉपी नहीं हो पाया। Saved. Could not copy the snapshot.";
      bugConfirm.hidden = false;
    } catch (err) {
      showBugError(BUG_SAVE_ERR);
    } finally {
      bugSave.disabled = false;
    }
  }

  function shrinkShot(file) {
    return new Promise(function (resolve) {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = function () {
        const scale = Math.min(1, 900 / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        resolve(canvas.toDataURL("image/jpeg", 0.55));
      };
      img.onerror = function () {
        URL.revokeObjectURL(url);
        resolve("");
      };
      img.src = url;
    });
  }

  consentCheck.addEventListener("change", function () {
    if (phase === "idle" || phase === "error") setCallButton(false);
  });

  callBtn.addEventListener("click", function () {
    if (phase === "idle" || phase === "error") {
      if (!consentCheck.checked) return;
      consentRow.hidden = true;
      const attemptId = ++callAttemptId;
      phase = "connecting";
      setCallButton(true);
      startCall(attemptId);
      return;
    }
    if (phase === "connecting" || phase === "live") endCall();
  });

  noBtn.addEventListener("click", function () { sendQuickReply("clear"); });
  yesBtn.addEventListener("click", function () { sendQuickReply("concern"); });

  nudgeBtn.addEventListener("click", function () {
    if (phase !== "live" || !wsOpen()) return;
    sendJson({ type: "nudge" });
    speechEndAt = 0;
    clearNudge();
    setStatus({ hi: "फिर से पूछ रही हूँ…", en: "Asking again" });
  });

  $("bug-btn").addEventListener("click", openBug);
  $("bug-close").addEventListener("click", closeBug);
  bugSave.addEventListener("click", saveBug);
  bugOverlay.addEventListener("click", function (event) { if (event.target === bugOverlay) closeBug(); });
  bugCats.addEventListener("click", function (event) {
    const chip = event.target.closest("button[data-cat]");
    if (chip && bugCats.contains(chip)) setBugCategory(chip.getAttribute("data-cat") || "");
  });
  if (bugShot) {
    bugShot.addEventListener("change", function () {
      const file = bugShot.files && bugShot.files[0];
      if (!file) {
        bugShotPending = null;
        bugShotData = "";
        return;
      }
      const job = shrinkShot(file).then(function (data) {
        if (bugShotPending === job) {
          bugShotData = data;
          bugShotPending = null;
        }
        return data;
      });
      bugShotPending = job;
    });
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !bugOverlay.hidden) closeBug();
  });

  $("logout-btn").addEventListener("click", async function () {
    try { sessionStorage.removeItem(CLINIC_KEY); } catch (err) { /* private mode */ }
    try {
      await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
    } catch (err) {
      /* redirect anyway */
    } finally {
      location.href = "/login";
    }
  });

  hudToggle.addEventListener("click", function () {
    hudEl.hidden = !hudEl.hidden;
    if (!hudEl.hidden) renderHud();
  });

  $("hud-form").addEventListener("submit", function (event) {
    event.preventDefault();
    const hudInput = $("hud-text");
    const text = hudInput.value.trim();
    if (!text) return;
    if (phase !== "live" || !wsOpen()) {
      showHudNote("Start the call before sending text.");
      return;
    }
    sendJson({ type: "text", text: text });
    hudInput.value = "";
    showHudNote("Sent.");
  });

  $("new-call").addEventListener("click", resetForNewCall);
  sharePdfBtn.addEventListener("click", sharePdf);
  retryPdfBtn.addEventListener("click", function () {
    if (callId == null) return;
    const attemptId = callAttemptId;
    phase = "preparing";
    showPreparingChrome();
    sharePdfBtn.hidden = true;
    retryPdfBtn.hidden = true;
    awaitPdfPromise(attemptId, requestReport);
  });

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible" && (phase === "connecting" || phase === "live" || phase === "preparing")) {
      acquireWakeLock();
    }
  });

  clinicSlug();

  fetch("/api/config", { credentials: "same-origin", headers: { "Accept": "application/json" } })
    .then(function (res) {
      if (res.status === 401) {
        loginRedirect();
        return null;
      }
      return res.ok ? res.json() : null;
    })
    .then(function (cfg) { if (cfg) applyConfig(cfg); })
    .catch(function () { /* the start button reports a real failure */ });
})();
