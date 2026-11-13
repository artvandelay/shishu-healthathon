    (function () {
      const statusEl = document.getElementById("status");
      const timerEl = document.getElementById("timer");
      const captionEl = document.getElementById("caption");
      const waitEl = document.getElementById("wait");
      const callBtn = document.getElementById("call-btn");
      const noBtn = document.getElementById("no-btn");
      const yesBtn = document.getElementById("yes-btn");
      const consentRow = document.getElementById("consent-row");
      const consentCheck = document.getElementById("consent-check");
      const stageEl = document.getElementById("stage");
      const thanksEl = document.getElementById("thanks");
      const summaryEl = document.getElementById("summary");
      const thanksSubEl = document.getElementById("thanks-sub");
      const adminView = document.getElementById("admin-view");
      const adminLink = document.getElementById("admin-link");
      const errorEl = document.getElementById("error");
      const engineNote = document.getElementById("engine-note");
      const avatarWrap = document.getElementById("avatar-wrap");
      const newCallBtn = document.getElementById("new-call");
      const micMeter = document.getElementById("mic-meter");
      const micLevel = document.getElementById("mic-level");
      const nudgeBtn = document.getElementById("nudge-btn");
      const bugBtn = document.getElementById("bug-btn");
      const bugOverlay = document.getElementById("bug-overlay");
      const bugCats = document.getElementById("bug-cats");
      const bugNote = document.getElementById("bug-note");
      const bugShot = document.getElementById("bug-shot");
      const hudStep = document.getElementById("hud-step");
      const logoutBtn = document.getElementById("logout-btn");
      let flowGroup = "";
      let summaryTimer = 0;
      const bugSave = document.getElementById("bug-save");
      const bugClose = document.getElementById("bug-close");
      const bugConfirm = document.getElementById("bug-confirm");
      const bugError = document.getElementById("bug-error");
      const hudToggle = document.getElementById("hud-toggle");
      const hudEl = document.getElementById("hud");
      const hudCall = document.getElementById("hud-call");
      const hudCaller = document.getElementById("hud-caller");
      const hudRole = document.getElementById("hud-role");
      const hudModel = document.getElementById("hud-model");
      const hudVoice = document.getElementById("hud-voice");
      const hudConnect = document.getElementById("hud-connect");
      const hudLastLat = document.getElementById("hud-last-lat");
      const hudAvgLat = document.getElementById("hud-avg-lat");
      const hudTurns = document.getElementById("hud-turns");
      const hudTools = document.getElementById("hud-tools");
      const hudNudges = document.getElementById("hud-nudges");
      const hudAudio = document.getElementById("hud-audio");
      const hudEvents = document.getElementById("hud-events");
      const hudForm = document.getElementById("hud-form");
      const hudText = document.getElementById("hud-text");
      const hudSendNote = document.getElementById("hud-send-note");

      const WORKLET_SOURCE = [
        "class PcmCapture extends AudioWorkletProcessor {",
        "  constructor() {",
        "    super();",
        "    this.buf = new Float32Array(2048);",
        "    this.used = 0;",
        "  }",
        "  process(inputs) {",
        "    const ch = inputs[0] && inputs[0][0];",
        "    if (!ch) return true;",
        "    let offset = 0;",
        "    while (offset < ch.length) {",
        "      const n = Math.min(this.buf.length - this.used, ch.length - offset);",
        "      this.buf.set(ch.subarray(offset, offset + n), this.used);",
        "      this.used += n;",
        "      offset += n;",
        "      if (this.used === this.buf.length) {",
        "        this.port.postMessage(this.buf.slice(0));",
        "        this.used = 0;",
        "      }",
        "    }",
        "    return true;",
        "  }",
        "}",
        "registerProcessor('pcm-capture', PcmCapture);"
      ].join("\n");

      let phase = "idle";
      let questionStep = "";
      let expectYesNo = false;
      let quickReplyReady = false;
      let quickReplyPending = false;
      let promptChars = 0;
      let stepFirstPlayTime = 0;
      let quickRevealTimer = 0;
      let engine = "gemini";
      let captureRate = 16000;
      let ws = null;
      let audioCtx = null;
      let micStream = null;
      let micSource = null;
      let processor = null;
      let workletNode = null;
      let silentGain = null;
      let nextPlayTime = 0;
      const activeSources = new Set();
      let timerHandle = 0;
      let startedAt = 0;
      let frozenDuration = "00:00";
      let callId = null;
      let shareToken = "";
      let wakeLock = null;
      const CLINIC_KEY = "shishu_clinic";
      let captionRole = "";
      let captionText = "";
      let socketOpened = false;
      let connectReject = null;
      let userRole = "user";
      let callerName = "";
      let modelName = "";
      let voiceName = "";
      let lastTelemetry = null;
      const recentEvents = [];
      let userSpeaking = false;
      let lastVoiceTs = 0;
      let speechEndAt = 0;
      let nudgeWatch = 0;
      let micSmooth = 0;
      let bugCategory = "";
      let hudNoteTimer = 0;
      let thinkingTimer = 0;

      function setStatus(text, mode) {
        statusEl.hidden = false;
        if (text && typeof text === "object") {
          statusEl.innerHTML = '<span class="hi">' + text.hi + '</span>' +
            (text.en ? '<span class="en">' + text.en + '</span>' : '');
        } else {
          statusEl.textContent = text || "";
        }
        statusEl.className = mode || "";
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

      function showError(message) {
        errorEl.hidden = false;
        if (message && typeof message === "object") {
          errorEl.innerHTML = '<span class="hi">' + message.hi + '</span><span class="en">' + message.en + '</span>';
        } else {
          errorEl.textContent = message;
        }
      }

      function clearError() {
        errorEl.hidden = true;
        errorEl.textContent = "";
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
        const query = clinic ? ("?clinic=" + encodeURIComponent(clinic)) : "";
        return proto + "//" + location.host + "/ws/gemini" + query;
      }

      async function acquireWakeLock() {
        if (!navigator.wakeLock || wakeLock) return;
        try {
          wakeLock = await navigator.wakeLock.request("screen");
          wakeLock.addEventListener("release", function () { wakeLock = null; });
        } catch (err) { /* the call still runs if the screen lock is denied */ }
      }

      function releaseWakeLock() {
        const lock = wakeLock;
        wakeLock = null;
        if (lock) lock.release().catch(function () {});
      }

      function callIsOpen() {
        return phase === "connecting" || phase === "live" || phase === "preparing";
      }

      function loginRedirect() {
        const next = location.pathname + location.search;
        location.href = "/login?next=" + encodeURIComponent(next || "/");
      }

      function showEngine() {
        engine = "gemini";
        captureRate = 16000;
        engineNote.textContent = "";
      }

      function pad(n) {
        return String(n).padStart(2, "0");
      }

      function formatElapsed(ms) {
        const sec = Math.max(0, Math.floor(ms / 1000));
        return pad(Math.floor(sec / 60)) + ":" + pad(sec % 60);
      }

      function startTimer() {
        if (timerHandle) return;
        startedAt = Date.now();
        timerHandle = setInterval(function () {
          frozenDuration = formatElapsed(Date.now() - startedAt);
          timerEl.textContent = frozenDuration;
        }, 250);
      }

      function stopTimer() {
        if (!timerHandle) return;
        clearInterval(timerHandle);
        timerHandle = 0;
        if (startedAt) {
          frozenDuration = formatElapsed(Date.now() - startedAt);
          timerEl.textContent = frozenDuration;
        }
      }

      function armEndButton() {
        callBtn.innerHTML = '<span class="hi">बंद</span><span class="en">End</span>';
        callBtn.classList.add("end");
        callBtn.setAttribute("aria-label", "कॉल बंद करें");
        callBtn.disabled = false;
      }

      function armStartButton() {
        callBtn.innerHTML = '<span class="hi">शुरू</span><span class="en">Start</span>';
        callBtn.classList.remove("end");
        callBtn.setAttribute("aria-label", "कॉल शुरू करें");
        callBtn.disabled = !consentCheck.checked;
      }

      function showConsentRow(clear) {
        consentRow.hidden = false;
        if (clear) consentCheck.checked = false;
        if (phase === "idle" || phase === "error") armStartButton();
      }

      function hideConsentRow() {
        consentRow.hidden = true;
      }

      function bytesToBase64(bytes) {
        let binary = "";
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        return btoa(binary);
      }

      function floatToPcm16LE(float32) {
        const buffer = new ArrayBuffer(float32.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < float32.length; i++) {
          let sample = float32[i];
          if (sample > 1) sample = 1;
          else if (sample < -1) sample = -1;
          let val = sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767);
          if (val > 32767) val = 32767;
          if (val < -32768) val = -32768;
          view.setInt16(i * 2, val, true);
        }
        return new Uint8Array(buffer);
      }

      function downsampleBuffer(buffer, sampleRate, outSampleRate) {
        if (!buffer || !buffer.length) return new Float32Array(0);
        if (outSampleRate === sampleRate) return buffer;
        const ratio = sampleRate / outSampleRate;
        const newLength = Math.round(buffer.length / ratio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < result.length) {
          const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
          let accum = 0, count = 0;
          for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
          }
          result[offsetResult] = count > 0 ? (accum / count) : 0;
          offsetResult++;
          offsetBuffer = nextOffsetBuffer;
        }
        return result;
      }

      function rmsOf(float32) {
        if (!float32 || !float32.length) return 0;
        let sum = 0;
        for (let i = 0; i < float32.length; i++) {
          const sample = float32[i];
          sum += sample * sample;
        }
        return Math.sqrt(sum / float32.length);
      }

      function paintMic(level) {
        const pct = Math.max(0, Math.min(100, Math.round((level / 0.12) * 100)));
        micLevel.style.width = pct + "%";
        micMeter.setAttribute("aria-valuenow", String(pct));
      }

      function updateMicLevel(rms) {
        micSmooth = micSmooth * 0.5 + rms * 0.5;
        paintMic(micSmooth);
      }

      function resetMicLevel() {
        micSmooth = 0;
        paintMic(0);
      }

      function refreshNudgeHighlight() {
        const stuck = phase === "live"
          && speechEndAt > 0
          && activeSources.size === 0
          && (Date.now() - speechEndAt) > 3000;
        nudgeBtn.hidden = !stuck;
        nudgeBtn.classList.toggle("hot", stuck);
      }

      function startNudgeWatch() {
        if (nudgeWatch) return;
        nudgeWatch = setInterval(refreshNudgeHighlight, 250);
      }

      function stopNudgeWatch() {
        if (!nudgeWatch) return;
        clearInterval(nudgeWatch);
        nudgeWatch = 0;
      }

      function syncQuickButtons() {
        const show = phase === "live" && expectYesNo && quickReplyReady;
        noBtn.hidden = !show;
        yesBtn.hidden = !show;
        noBtn.disabled = !show || quickReplyPending;
        yesBtn.disabled = !show || quickReplyPending;
      }

      function clearQuickRevealTimer() {
        if (quickRevealTimer) clearTimeout(quickRevealTimer);
        quickRevealTimer = 0;
      }

      function resetQuickState() {
        questionStep = "";
        expectYesNo = false;
        quickReplyReady = false;
        quickReplyPending = false;
        clearQuickRevealTimer();
        syncQuickButtons();
      }

      function leaveLiveChrome() {
        stopNudgeWatch();
        clearThinkingWatch();
        avatarWrap.classList.remove("speaking", "thinking", "reconnecting");
        nudgeBtn.classList.remove("hot");
        nudgeBtn.hidden = true;
        userSpeaking = false;
        speechEndAt = 0;
        resetMicLevel();
        resetQuickState();
      }

      function sendQuickReply(outcome) {
        if (phase !== "live" || !expectYesNo || !quickReplyReady || quickReplyPending || !questionStep || !ws || ws.readyState !== WebSocket.OPEN) return;
        quickReplyPending = true;
        syncQuickButtons();
        flushPlayback();
        speechEndAt = 0;
        nudgeBtn.classList.remove("hot");
        nudgeBtn.hidden = true;
        showCaption("user", outcome === "clear" ? "नहीं।" : "हाँ।");
        startThinkingWatch();
        ws.send(JSON.stringify({ type: "quick_reply", outcome: outcome, step: questionStep }));
      }

      function isBotPlaying() {
        return activeSources.size > 0 || (audioCtx && audioCtx.currentTime < nextPlayTime);
      }

      function emitSpeechEnd() {
        speechEndAt = Date.now();
        refreshNudgeHighlight();
        if (phase !== "live" || !ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ type: "speech_end" }));
        startThinkingWatch();
      }

      function trackSpeech(rms) {
        if (phase !== "live") return;
        const now = Date.now();
        if (rms > 0.011) {
          userSpeaking = true;
          lastVoiceTs = now;
          speechEndAt = 0;
          nudgeBtn.classList.remove("hot"); nudgeBtn.hidden = true;
          return;
        }
        if (!isBotPlaying() && userSpeaking && (now - lastVoiceTs) > 550) {
          userSpeaking = false;
          emitSpeechEnd();
        }
      }

      function onMicFrame(float32) {
        const rms = rmsOf(float32);
        updateMicLevel(rms);
        trackSpeech(rms);
        sendPcm(float32);
      }

      function sendPcm(float32) {
        if (phase !== "live" || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (ws.bufferedAmount > 262144) return;
        const inRate = audioCtx ? audioCtx.sampleRate : captureRate;
        const down = downsampleBuffer(float32, inRate, captureRate);
        if (!down.length) return;
        ws.send(JSON.stringify({
          type: "audio",
          data: bytesToBase64(floatToPcm16LE(down)),
          sample_rate: captureRate
        }));
      }

      function base64ToBytes(b64) {
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
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
        const frames = usable / 2;
        const floats = new Float32Array(frames);
        for (let i = 0; i < frames; i++) floats[i] = view.getInt16(i * 2, true) / 32768;
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
        nudgeBtn.classList.remove("hot"); nudgeBtn.hidden = true;
        const now = audioCtx.currentTime;
        nextPlayTime = Math.max(now, nextPlayTime);
        src.start(nextPlayTime);
        nextPlayTime += audioBuffer.duration;
        activeSources.add(src);
        avatarWrap.classList.add("speaking");
        setStatus({ hi: "SHISHU बोल रही है…", en: "SHISHU is speaking" });
        if (expectYesNo && !quickReplyReady && !quickRevealTimer) {
          if (!stepFirstPlayTime) stepFirstPlayTime = Date.now();
          const estMs = Math.max(1200, Math.min(4500, Math.round((promptChars / 14) * 1000)));
          const halfwayMs = Math.round(estMs * 0.5);
          quickRevealTimer = setTimeout(function () {
            quickRevealTimer = 0;
            if (phase === "live" && expectYesNo) {
              quickReplyReady = true;
              syncQuickButtons();
            }
          }, halfwayMs);
        }
        src.onended = function () {
          activeSources.delete(src);
          if (activeSources.size === 0 && phase === "live") {
            avatarWrap.classList.remove("speaking", "thinking");
            if (!avatarWrap.classList.contains("reconnecting")) {
              setStatus({ hi: "सुन रही हूँ…", en: "Listening" });
            }
          }
          if (activeSources.size === 0 && phase === "live" && expectYesNo && !quickReplyReady) {
            clearQuickRevealTimer();
            quickReplyReady = true;
            syncQuickButtons();
          }
        };
      }

      function mergeChunk(prev, chunk) {
        chunk = String(chunk || "");
        if (!prev) return chunk;
        if (!chunk) return prev;
        if (chunk.indexOf(prev) === 0) return chunk;
        if (prev.slice(-chunk.length) === chunk) return prev;
        const needsSpace = !/\s$/.test(prev) && !/^[\s,.;:!?।]/.test(chunk);
        return prev + (needsSpace ? " " : "") + chunk;
      }

      function showCaption(role, text) {
        if (role !== captionRole) {
          captionRole = role || "";
          captionText = "";
        }
        captionText = mergeChunk(captionText, text);
        const who = role === "assistant" ? "SHISHU" : "You";
        captionEl.textContent = captionText ? (who + ": " + captionText) : "";
      }

      function dash(value) {
        return value === 0 || value ? String(value) : "—";
      }

      function msText(value) {
        if (value === 0 || (typeof value === "number" && isFinite(value))) return String(Math.round(value)) + " ms";
        return "—";
      }

      function lastLatency(telemetry) {
        if (!telemetry) return "—";
        const samples = telemetry.turn_latencies_ms;
        if (Array.isArray(samples) && samples.length) return msText(samples[samples.length - 1]);
        if (telemetry.last_latency_ms != null) return msText(Number(telemetry.last_latency_ms));
        return "—";
      }

      function eventClock(t) {
        if (typeof t !== "number" || !isFinite(t)) return "";
        const ms = t > 1e12 ? t : t * 1000;
        const when = new Date(ms);
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
        const telemetry = lastTelemetry || {};
        hudCall.textContent = callId != null ? ("#" + callId) : "—";
        hudCaller.textContent = callerName || "—";
        hudRole.textContent = userRole || "—";
        hudModel.textContent = modelName || "—";
        hudVoice.textContent = voiceName || "—";
        hudConnect.textContent = msText(telemetry.connect_ms);
        hudLastLat.textContent = lastLatency(lastTelemetry);
        hudAvgLat.textContent = msText(telemetry.avg_latency_ms);
        hudTurns.textContent = dash(telemetry.user_turns) + " / " + dash(telemetry.bot_turns);
        hudTools.textContent = dash(telemetry.tool_calls);
        hudNudges.textContent = dash(telemetry.watchdog_nudges);
        hudAudio.textContent = dash(telemetry.audio_chunks_in) + " / " + dash(telemetry.audio_chunks_out);
        if (hudStep) hudStep.textContent = flowGroup || telemetry.progress_group || "—";
        hudEvents.replaceChildren();
        recentEvents.forEach(function (evt) {
          const li = document.createElement("li");
          const bits = [];
          const clock = eventClock(evt && evt.t);
          if (clock) bits.push(clock);
          if (evt && evt.type) bits.push(evt.type);
          if (evt && evt.detail) bits.push(evt.detail);
          li.textContent = bits.join(" · ") || "event";
          hudEvents.appendChild(li);
        });
      }

      function applyConfig(cfg) {
        if (!cfg) return;
        if (cfg.role) userRole = cfg.role;
        if (cfg.username) callerName = cfg.username;
        const isAdmin = userRole === "admin";
        hudToggle.hidden = !isAdmin;
        adminLink.hidden = !isAdmin;
        adminView.hidden = !isAdmin;
        if (isAdmin) hudEl.hidden = false;
        renderHud();
      }

      function showHudNote(text) {
        hudSendNote.textContent = text;
        hudSendNote.hidden = false;
        if (hudNoteTimer) clearTimeout(hudNoteTimer);
        hudNoteTimer = setTimeout(function () {
          hudSendNote.hidden = true;
        }, 2200);
      }

      function stopCapture() {
        if (processor) {
          processor.onaudioprocess = null;
          try { processor.disconnect(); } catch (err) { /* ignore */ }
          processor = null;
        }
        if (workletNode) {
          workletNode.port.onmessage = null;
          try { workletNode.disconnect(); } catch (err) { /* ignore */ }
          workletNode = null;
        }
        if (micSource) {
          try { micSource.disconnect(); } catch (err) { /* ignore */ }
          micSource = null;
        }
        if (silentGain) {
          try { silentGain.disconnect(); } catch (err) { /* ignore */ }
          silentGain = null;
        }
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
          sock.onopen = null;
          sock.onmessage = null;
          sock.onerror = null;
          sock.onclose = null;
          try { sock.close(); } catch (err) { /* ignore */ }
        }
        if (reject) reject(new Error("cancelled"));
      }

      function onSocketMessage(event) {
        let msg;
        try { msg = JSON.parse(event.data); } catch (err) { return; }
        if (!msg || typeof msg.type !== "string") return;
        if (msg.type === "session_started") {
          callId = msg.call_id;
          rememberShareToken(msg);
          if (msg.caller_name) callerName = msg.caller_name;
          if (msg.user_role) userRole = msg.user_role;
          if (msg.engine) modelName = msg.engine;
          if (msg.voice) voiceName = msg.voice;
          renderHud();
          if (phase === "live") setStatus({ hi: "सुन रही हूँ…", en: "Listening" });
          startTimer();
        } else if (msg.type === "audio") {
          try { playPcm(msg.data, msg.sample_rate || 24000); } catch (err) { /* skip a bad chunk */ }
        } else if (msg.type === "clear") {
          flushPlayback();
          speechEndAt = 0;
          nudgeBtn.classList.remove("hot"); nudgeBtn.hidden = true;
          if (phase === "live" && !avatarWrap.classList.contains("reconnecting")) setStatus({ hi: "सुन रही हूँ…", en: "Listening" });
        } else if (msg.type === "reconnecting") {
          clearThinkingWatch();
          avatarWrap.classList.remove("speaking", "thinking");
          avatarWrap.classList.add("reconnecting");
          setStatus({ hi: "लाइन जुड़ रही है…", en: "Reconnecting" }, "reconnecting");
        } else if (msg.type === "reconnected") {
          avatarWrap.classList.remove("reconnecting");
          if (phase === "live" && !isBotPlaying()) {
            setStatus({ hi: "सुन रही हूँ…", en: "Listening" });
          }
        } else if (msg.type === "transcript") {
          showCaption(msg.role, msg.text);
        } else if (msg.type === "telemetry") {
          lastTelemetry = msg.telemetry || {};
          pushEvent(msg.last_event);
          renderHud();
        } else if (msg.type === "progress") {
          flowGroup = msg.group || "";
          renderHud();
        } else if (msg.type === "question_state") {
          questionStep = String(msg.step || "");
          expectYesNo = Boolean(msg.expect_yes_no);
          promptChars = Number(msg.prompt_chars) || 0;
          quickReplyReady = false;
          quickReplyPending = false;
          stepFirstPlayTime = 0;
          clearQuickRevealTimer();
          if (msg.group) {
            flowGroup = String(msg.group);
            renderHud();
          }
          syncQuickButtons();
        } else if (msg.type === "ending") {
          beginPreparing();
          hangupWhenQuiet();
        } else if (msg.type === "call_completed") {
          rememberShareToken(msg);
          revealReadyReport(msg.call_id);
        } else if (msg.type === "report_failed") {
          showReportRetry(msg.call_id);
        } else if (msg.type === "error") {
          showError(msg.message ? { hi: msg.message, en: "The call hit an error." } : { hi: "कॉल में गड़बड़ी आ गई।", en: "The call hit an error." });
        }
      }

      let reportPending = false;
      let hangupArmed = false;

      function onSocketClose() {
        if (phase === "done") return;
        if (phase === "preparing") {
          if (!reportPending) showReportRetry(callId);
          return;
        }
        if (phase === "live" || phase === "connecting") {
          stopCapture();
          flushPlayback();
          leaveLiveChrome();
          avatarWrap.classList.remove("live", "speaking");
          phase = "error";
          resetQuickState();
          releaseWakeLock();
          setStatus({ hi: "कॉल शुरू करने के लिए नीचे दबाएँ", en: "Tap below to start the call" });
          showConsentRow();
          showError({ hi: "कॉल कट गई। नीचे दबाकर फिर शुरू करें।", en: "The call disconnected. Tap below to start again." });
        }
      }

      function openSocket() {
        socketOpened = false;
        return new Promise(function (resolve, reject) {
          connectReject = reject;
          const sock = new WebSocket(wsUrl());
          ws = sock;
          sock.onopen = function () {
            socketOpened = true;
            connectReject = null;
            resolve();
          };
          sock.onmessage = onSocketMessage;
          sock.onerror = function () {
            if (!socketOpened) reject(new Error("Could not connect to the voice service."));
          };
          sock.onclose = onSocketClose;
        });
      }

      function requestMic() {
        return navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          },
          video: false
        });
      }

      async function wireMic() {
        if (!audioCtx || audioCtx.state === "closed") {
          throw new Error("Microphone capture was interrupted.");
        }
        if (audioCtx.state === "suspended") await audioCtx.resume().catch(function () {});
        audioCtx.onstatechange = function () {
          if (phase === "live" && audioCtx && audioCtx.state === "suspended") {
            audioCtx.resume().catch(function () {});
          }
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
              try { if (micSource) micSource.disconnect(); } catch (e) {}
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
          const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
          const url = URL.createObjectURL(blob);
          try {
            await audioCtx.audioWorklet.addModule(url);
          } finally {
            URL.revokeObjectURL(url);
          }
          workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");
          workletNode.port.onmessage = function (event) {
            onMicFrame(event.data);
          };
          micSource.connect(workletNode);
        } else if (typeof audioCtx.createScriptProcessor === "function") {
          processor = audioCtx.createScriptProcessor(2048, 1, 1);
          processor.onaudioprocess = function (event) {
            onMicFrame(event.inputBuffer.getChannelData(0));
          };
          micSource.connect(processor);
          silentGain = audioCtx.createGain();
          silentGain.gain.value = 0;
          processor.connect(silentGain);
          silentGain.connect(audioCtx.destination);
        } else {
          throw new Error("This browser cannot capture microphone audio.");
        }
      }

      async function startCall(micPromise) {
        resetQuickState();
        clearError();
        captionRole = "";
        captionText = "";
        captionEl.textContent = "";
        callId = null;
        shareToken = "";
        modelName = "";
        voiceName = "";
        lastTelemetry = null;
        recentEvents.length = 0;
        userSpeaking = false;
        lastVoiceTs = 0;
        speechEndAt = 0;
        renderHud();
        setStatus({ hi: "सुन रही हूँ…", en: "Listening" });
        acquireWakeLock();
        try {
          const configPromise = fetch("/api/config", {
            credentials: "same-origin",
            headers: { "Accept": "application/json" }
          });
          const stream = await micPromise;
          if (phase !== "connecting") {
            stream.getTracks().forEach(function (track) { track.stop(); });
            closeAudioContext();
            return;
          }
          micStream = stream;
          const res = await configPromise;
          if (phase !== "connecting") return;
          if (res.status === 401) {
            closeAudioContext();
            loginRedirect();
            return;
          }
          if (!res.ok) throw new Error("Could not read the voice engine setting.");
          const cfg = await res.json();
          if (phase !== "connecting") return;
          showEngine();
          applyConfig(cfg);
          await wireMic();
          if (phase !== "connecting") {
            closeAudioContext();
            return;
          }
          await openSocket();
          if (phase !== "connecting") return;
          phase = "live";
          syncQuickButtons();
          avatarWrap.classList.add("live");
          startNudgeWatch();
          setStatus({ hi: "सुन रही हूँ…", en: "Listening" });
          startTimer();
          armEndButton();
        } catch (err) {
          if (phase !== "connecting") return;
          closeAudioContext();
          closeSocket();
          leaveLiveChrome();
          avatarWrap.classList.remove("live", "speaking");
          phase = "error";
          resetQuickState();
          releaseWakeLock();
          setStatus({ hi: "कॉल शुरू करने के लिए नीचे दबाएँ", en: "Tap below to start the call" });
          showConsentRow();
          const denied = err && (err.name === "NotAllowedError" || err.name === "NotFoundError");
          showError(denied
            ? { hi: "कॉल के लिए माइक्रोफ़ोन की अनुमति दें।", en: "Allow the microphone to start the call." }
            : { hi: "कॉल शुरू नहीं हो पाई। फिर कोशिश करें।", en: "The call could not start. Try again." });
        }
      }

      function beginPreparing() {
        if (phase === "preparing" || phase === "done") return;
        stopCapture();
        stopTimer();
        leaveLiveChrome();
        avatarWrap.classList.remove("live", "speaking");
        phase = "preparing";
        resetQuickState();
        setStatus("");
        stageEl.classList.add("working");
        avatarWrap.classList.add("working");
        waitEl.hidden = false;
        callBtn.disabled = true;
      }

      function hangupWhenQuiet() {
        if (hangupArmed || phase !== "preparing") return;
        hangupArmed = true;
        const startedAt = Date.now();
        let quietSince = 0;
        const timer = setInterval(function () {
          if (phase !== "preparing") {
            clearInterval(timer);
            return;
          }
          const playing = activeSources.size > 0 || (audioCtx && audioCtx.currentTime < nextPlayTime);
          if (playing) {
            quietSince = 0;
          } else if (!quietSince) {
            quietSince = Date.now();
          }
          const quietLongEnough = quietSince && Date.now() - quietSince >= 2500;
          if (quietLongEnough || Date.now() - startedAt >= 30000) {
            clearInterval(timer);
            const sock = ws;
            if (sock && sock.readyState === WebSocket.OPEN) {
              try { sock.send(JSON.stringify({ type: "hangup" })); } catch (err) { /* the server also ends */ }
            }
          }
        }, 250);
      }

      function endCall() {
        if (phase !== "live" && phase !== "connecting") return;
        resetQuickState();
        const sock = ws;
        const open = sock && sock.readyState === WebSocket.OPEN;
        closeAudioContext();
        stopTimer();
        leaveLiveChrome();
        avatarWrap.classList.remove("live", "speaking");
        if (!open) {
          phase = "idle";
          releaseWakeLock();
          closeSocket();
          setStatus({ hi: "कॉल शुरू करने के लिए नीचे दबाएँ", en: "Tap below to start the call" });
          stageEl.classList.remove("working");
          avatarWrap.classList.remove("working");
          waitEl.hidden = true;
          showConsentRow();
          return;
        }
        hangupArmed = true;
        beginPreparing();
        try {
          sock.send(JSON.stringify({ type: "hangup" }));
        } catch (err) {
          showError({ hi: "कॉल बंद नहीं हो पाई।", en: "The call could not end cleanly." });
        }
      }

      let pdfFile = null;

      function rememberShareToken(msg) {
        if (msg && msg.share_token) shareToken = String(msg.share_token);
      }

      function motherPdfUrl() {
        if (shareToken) {
          return "/api/calls/by-token/" + encodeURIComponent(shareToken) + "/mother.pdf";
        }
        return "/api/calls/" + encodeURIComponent(callId) + "/mother.pdf";
      }

      function pdfName(res) {
        const header = res.headers.get("Content-Disposition") || "";
        const match = /filename="?([^";]+)"?/.exec(header);
        return (match && match[1]) || "SHISHU.pdf";
      }

      function fetchMotherPdf() {
        return fetch(motherPdfUrl(), { credentials: "same-origin" })
          .then(function (res) {
            if (!res.ok) throw new Error("pdf");
            const name = pdfName(res);
            return res.blob().then(function (blob) {
              return new File([blob], name, { type: "application/pdf" });
            });
          });
      }

      function revealReadyReport(id) {
        if (id != null) callId = id;
        if (callId == null) {
          showReportRetry(null);
          return;
        }
        if (phase !== "preparing" && phase !== "done") beginPreparing();
        reportPending = true;
        fetchMotherPdf().then(function (file) {
          reportPending = false;
          showThanks({ call_id: callId, pdfFile: file });
        }).catch(function () {
          reportPending = false;
          showReportRetry(callId);
        });
      }

      function showReportRetry(id) {
        if (id != null) callId = id;
        showThanks({ call_id: callId, pdf_failed: true });
      }

      function requestReport() {
        return fetch(motherPdfUrl(), {
          method: "POST",
          credentials: "same-origin"
        }).then(function (res) {
          if (!res.ok) throw new Error("pdf");
          const name = pdfName(res);
          return res.blob().then(function (blob) {
            return new File([blob], name, { type: "application/pdf" });
          });
        });
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

      function sharePdf() {
        if (!pdfFile) return;
        logShareClick();
        const shareable = navigator.canShare && navigator.canShare({ files: [pdfFile] });
        if (shareable) {
          navigator.share({ files: [pdfFile], title: "SHISHU" }).catch(function (err) {
            if (!err || err.name === "AbortError") return;
            showError({ hi: "इस फ़ोन पर शेयर नहीं खुल पाया। फ़ाइल डाउनलोड हो रही है।", en: "Share did not open. The file is downloading." });
            downloadPdf();
          });
          return;
        }
        downloadPdf();
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

      function showThanks(msg) {
        if (summaryTimer) clearTimeout(summaryTimer);
        summaryTimer = 0;
        phase = "done";
        resetQuickState();
        releaseWakeLock();
        stopCapture();
        flushPlayback();
        stopTimer();
        leaveLiveChrome();
        callId = msg.call_id != null ? msg.call_id : callId;
        stageEl.classList.remove("working");
        avatarWrap.classList.remove("working");
        waitEl.hidden = true;
        stageEl.hidden = true;
        summaryEl.textContent = "इस रिपोर्ट (PDF) में वे सवाल हैं जो आप अगली मुलाकात में डॉक्टर या ASHA / ANM दीदी से पूछ सकती हैं। इसे व्हाट्सएप पर भेजने या सेव करने के लिए नीचे 'शेयर करें' दबाएँ।";
        const bits = [];
        if (callId != null && userRole === "admin") bits.push("Call #" + callId);
        bits.push(frozenDuration);
        thanksSubEl.textContent = bits.join(" · ");
        adminView.href = callId != null ? ("/admin?call=" + encodeURIComponent(callId)) : "/admin";
        pdfFile = msg.pdfFile || null;
        document.getElementById("share-pdf").hidden = !pdfFile;
        document.getElementById("retry-pdf").hidden = !msg.pdf_failed;
        thanksEl.hidden = false;
        renderHud();
        clearError();
      }

      function resetForNewCall() {
        closeSocket();
        closeAudioContext();
        stopTimer();
        leaveLiveChrome();
        phase = "idle";
        resetQuickState();
        releaseWakeLock();
        hangupArmed = false;
        reportPending = false;
        callId = null;
        shareToken = "";
        modelName = "";
        voiceName = "";
        lastTelemetry = null;
        recentEvents.length = 0;
        startedAt = 0;
        frozenDuration = "00:00";
        timerEl.textContent = "00:00";
        captionRole = "";
        captionText = "";
        captionEl.textContent = "";
        summaryEl.textContent = "इस रिपोर्ट (PDF) में वे सवाल हैं जो आप अगली मुलाकात में डॉक्टर या ASHA / ANM दीदी से पूछ सकती हैं। इसे व्हाट्सएप पर भेजने या सेव करने के लिए नीचे 'शेयर करें' दबाएँ।";
        pdfFile = null;
        document.getElementById("share-pdf").hidden = true;
        document.getElementById("retry-pdf").hidden = true;
        thanksEl.hidden = true;
        stageEl.hidden = false;
        stageEl.classList.remove("working");
        avatarWrap.classList.remove("working", "live", "speaking");
        waitEl.hidden = true;
        setStatus({ hi: "कॉल शुरू करने के लिए नीचे दबाएँ", en: "Tap below to start the call" });
        showConsentRow(true);
        renderHud();
        clearError();
      }

      function closeBug() {
        bugOverlay.hidden = true;
      }

      function resetBugDraft() {
        bugNote.value = "";
        bugCategory = "";
        if (bugShot) bugShot.value = "";
        window.__bugShot = "";
        bugCats.querySelectorAll(".chip").forEach(function (btn) {
          btn.classList.remove("on");
          btn.setAttribute("aria-pressed", "false");
        });
      }

      function openBug() {
        bugConfirm.hidden = true;
        bugError.hidden = true;
        bugOverlay.hidden = false;
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

      async function saveBug() {
        bugConfirm.hidden = true;
        bugError.hidden = true;
        if (!bugCategory) {
          bugError.textContent = "Pick a category.";
          bugError.hidden = false;
          return;
        }
        const payload = {
          call_id: callId != null ? Number(callId) : 0,
          category: bugCategory,
          note: bugNote.value.trim(),
          call_elapsed_sec: startedAt ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000)) : 0,
          client_state: {
            phase: phase,
            engine: engine,
            flowGroup: flowGroup,
            captureRate: captureRate,
            captionText: captionText,
            lastTelemetry: lastTelemetry,
            recentEvents: recentEvents.slice(),
            screenshot: window.__bugShot || ""
          }
        };
        const snapshot = JSON.stringify(payload, null, 2);
        bugSave.disabled = true;
        try {
          const res = await fetch("/api/bugs", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify(payload)
          });
          if (!res.ok) {
            bugError.textContent = res.status === 401
              ? "Sign in again to save this bug. The call is still open."
              : "Could not save the bug. The call is still open.";
            bugError.hidden = false;
            return;
          }
          let copied = true;
          try {
            await copyText(snapshot);
          } catch (err) {
            copied = false;
          }
          resetBugDraft();
          bugConfirm.textContent = copied ? "Saved. Snapshot copied." : "Saved. Could not copy the snapshot.";
          bugConfirm.hidden = false;
        } catch (err) {
          bugError.textContent = "Could not save the bug. The call is still open.";
          bugError.hidden = false;
        } finally {
          bugSave.disabled = false;
        }
      }

      consentCheck.addEventListener("change", function () {
        if (phase === "idle" || phase === "error") armStartButton();
      });

      callBtn.addEventListener("click", function () {
        if (phase === "idle" || phase === "error") {
          if (!consentCheck.checked) return;
          hideConsentRow();
          phase = "connecting";
          armEndButton();
          if (audioCtx && audioCtx.state !== "closed") {
            try { audioCtx.close(); } catch (e) {}
          }
          const AudioContextClass = window.AudioContext || window.webkitAudioContext;
          audioCtx = new AudioContextClass({ latencyHint: "interactive" });
          audioCtx.resume().catch(function () {});
          startCall(requestMic());
          return;
        }
        if (phase === "connecting" || phase === "live") endCall();
      });

      noBtn.addEventListener("click", function () { sendQuickReply("clear"); });
      yesBtn.addEventListener("click", function () { sendQuickReply("concern"); });

      nudgeBtn.addEventListener("click", function () {
        if (phase !== "live" || !ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ type: "nudge" }));
        speechEndAt = 0;
        nudgeBtn.classList.remove("hot"); nudgeBtn.hidden = true;
        setStatus({ hi: "फिर से पूछ रही हूँ…", en: "Asking again" });
      });

      bugBtn.addEventListener("click", openBug);
      bugClose.addEventListener("click", closeBug);
      function shrinkShot(file) {
        return new Promise(function (resolve) {
          const img = new Image();
          const url = URL.createObjectURL(file);
          img.onload = function () {
            const max = 900;
            const scale = Math.min(1, max / Math.max(img.width, img.height));
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

      if (bugShot) {
        bugShot.addEventListener("change", function () {
          const file = bugShot.files && bugShot.files[0];
          if (!file) {
            window.__bugShot = "";
            return;
          }
          shrinkShot(file).then(function (data) { window.__bugShot = data; });
        });
      }

      logoutBtn.addEventListener("click", async function () {
        await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
        location.href = "/login";
      });

      bugSave.addEventListener("click", saveBug);
      bugOverlay.addEventListener("click", function (event) {
        if (event.target === bugOverlay) closeBug();
      });
      bugCats.addEventListener("click", function (event) {
        const chip = event.target.closest("button[data-cat]");
        if (!chip || !bugCats.contains(chip)) return;
        bugCategory = chip.getAttribute("data-cat") || "";
        bugCats.querySelectorAll(".chip").forEach(function (btn) {
          const on = btn === chip;
          btn.classList.toggle("on", on);
          btn.setAttribute("aria-pressed", on ? "true" : "false");
        });
      });
      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && !bugOverlay.hidden) closeBug();
      });

      hudToggle.addEventListener("click", function () {
        hudEl.hidden = !hudEl.hidden;
        if (!hudEl.hidden) renderHud();
      });

      hudForm.addEventListener("submit", function (event) {
        event.preventDefault();
        const text = hudText.value.trim();
        if (!text) return;
        if (phase !== "live" || !ws || ws.readyState !== WebSocket.OPEN) {
          showHudNote("Start the call before sending text.");
          return;
        }
        ws.send(JSON.stringify({ type: "text", text: text }));
        hudText.value = "";
        showHudNote("Sent.");
      });

      newCallBtn.addEventListener("click", resetForNewCall);
      document.getElementById("share-pdf").addEventListener("click", sharePdf);
      document.getElementById("retry-pdf").addEventListener("click", function () {
        if (callId == null) return;
        thanksEl.hidden = true;
        stageEl.hidden = false;
        stageEl.classList.add("working");
        avatarWrap.classList.add("working");
        waitEl.hidden = false;
        phase = "preparing";
        document.getElementById("share-pdf").hidden = true;
        document.getElementById("retry-pdf").hidden = true;
        requestReport().then(function (file) {
          showThanks({ call_id: callId, pdfFile: file });
        }).catch(function () {
          showReportRetry(callId);
        });
      });

      document.addEventListener("visibilitychange", function () {
        if (document.visibilityState === "visible" && callIsOpen()) acquireWakeLock();
      });

      clinicSlug();

      fetch("/api/config", { credentials: "same-origin", headers: { "Accept": "application/json" } })
        .then(function (res) {
          if (res.status === 401) {
            loginRedirect();
            return null;
          }
          if (!res.ok) return null;
          return res.json();
        })
        .then(function (cfg) {
          if (!cfg) return;
          showEngine();
          applyConfig(cfg);
        })
        .catch(function () { /* the start button reports a real failure */ });
    })();
