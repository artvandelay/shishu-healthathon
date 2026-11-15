(function () {
  const $ = function (id) { return document.getElementById(id); };
  const listEl = $("call-list");
  const emptyEl = $("empty");
  const reportEl = $("report");
  const noticeEl = $("notice");
  const updatedEl = $("updated");
  const titleEl = $("call-title");
  const metaEl = $("call-meta");
  const player = $("player");
  const download = $("download");
  const audioNote = $("audio-note");
  const flagBanner = $("flag-banner");
  const flagList = $("flag-list");
  const noFlags = $("no-flags");
  const doctorEl = $("doctor");
  const motherEl = $("mother");
  const findingsEl = $("findings");
  const transcriptEl = $("transcript");

  let calls = [];
  let bugs = [];
  let selectedId = null;
  let selectedStatus = "";
  let current = null;
  let refreshTimer = 0;
  let loadingList = false;
  let refreshQueued = false;
  let loadingBugs = false;
  let clinicFilter = "";
  let detailSeq = 0;
  let reviewPaintedCallId = null;
  let reviewDirtyCallId = null;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function showNotice(message) {
    noticeEl.textContent = message;
    noticeEl.hidden = !message;
  }

  function loginRedirect() {
    location.href = "/login?next=" + encodeURIComponent(location.pathname + location.search);
  }

  function stampUpdated(node) {
    node.textContent = "Updated " + new Date().toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  async function readErrorMessage(res, fallback) {
    let detail = fallback;
    try {
      const body = await res.json();
      if (body && (body.detail || body.message)) detail = body.detail || body.message;
    } catch (err) { /* keep fallback text */ }
    return typeof detail === "string" ? detail : fallback;
  }

  async function api(path, options) {
    const opts = options ? Object.assign({}, options) : {};
    opts.credentials = "same-origin";
    opts.headers = Object.assign({ "Accept": "application/json" }, opts.headers || {});
    const res = await fetch(path, opts);
    if (res.status === 401) {
      loginRedirect();
      return null;
    }
    if (!res.ok) throw new Error(await readErrorMessage(res, "Request failed (" + res.status + ")"));
    return res;
  }

  function asList(value) {
    if (Array.isArray(value)) return value;
    if (typeof value === "string") {
      try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed : [];
      } catch (err) { return []; }
    }
    return [];
  }

  function asObject(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) return value;
    if (typeof value === "string") {
      try {
        const parsed = JSON.parse(value);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
      } catch (err) { return {}; }
    }
    return {};
  }

  function asCalls(data) {
    if (Array.isArray(data)) return data;
    return (data && Array.isArray(data.calls)) ? data.calls : [];
  }

  function modelBadge(call) {
    const model = String((call && call.model_used) || "").trim();
    if (!model) return "—";
    if (model.toLowerCase() === "gemini-3.1-flash-live-preview") return "Gemini 3.1 Flash Live";
    return model;
  }

  function flagCount(call) {
    return asList(call && call.red_flags).length;
  }

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function fmtDuration(sec) {
    const n = Math.max(0, Number(sec) || 0);
    return pad(Math.floor(n / 60)) + ":" + pad(Math.floor(n % 60));
  }

  function fmtTime(value) {
    if (!value) return "";
    const raw = String(value).trim();
    const iso = raw.indexOf("T") === -1 ? raw.replace(" ", "T") : raw;
    const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
    if (Number.isNaN(d.getTime())) return raw;
    return d.toLocaleString(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  function callerLine(call) {
    const name = String(call.caller_name || "").trim();
    return name ? "Logged in as " + name : "Login name not recorded";
  }

  function people(call) {
    const mother = (call && call.mother_name) || "Mother not named";
    const age = String((call && call.child_age) || "").trim();
    return age ? (mother + " · " + age) : mother;
  }

  function clinicCode(call) {
    return String((call && call.clinic_code) || "").trim();
  }

  function fillClinicFilter() {
    const select = $("clinic-filter");
    if (!select) return;
    const codes = [];
    calls.forEach(function (call) {
      const code = clinicCode(call);
      if (code && codes.indexOf(code) === -1) codes.push(code);
    });
    codes.sort();
    const previous = clinicFilter;
    select.textContent = "";
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All clinics";
    select.appendChild(all);
    codes.forEach(function (code) {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = code;
      select.appendChild(opt);
    });
    clinicFilter = (previous && codes.indexOf(previous) !== -1) ? previous : "";
    select.value = clinicFilter;
  }

  function visibleCalls() {
    if (!clinicFilter) return calls;
    return calls.filter(function (call) { return clinicCode(call) === clinicFilter; });
  }

  function renderList() {
    fillClinicFilter();
    const shown = visibleCalls();
    $("tab-calls").textContent = "Calls (" + shown.length + ")";
    listEl.textContent = "";
    if (!calls.length) {
      listEl.appendChild(el("p", "empty", "No calls yet."));
      return;
    }
    if (!shown.length) {
      listEl.appendChild(el("p", "empty", "No calls for this clinic."));
      return;
    }
    shown.forEach(function (call) {
      const btn = el("button", "call-item");
      btn.type = "button";
      if (call.id === selectedId) btn.classList.add("selected");
      const top = el("div", "row");
      const left = el("div", "row-left");
      left.appendChild(el("strong", "", "Call #" + call.id));
      if (call.status === "active") left.appendChild(el("span", "live-dot", "Live"));
      top.appendChild(left);
      const count = flagCount(call);
      const pill = el("span", count > 0 ? "pill hot" : "pill", count === 1 ? "1 flag" : count + " flags");
      pill.setAttribute("aria-label", count + " red flags");
      top.appendChild(pill);
      btn.appendChild(top);
      btn.appendChild(el("div", "caller", callerLine(call)));
      btn.appendChild(el("div", "people", people(call)));
      const meta = el("div", "meta");
      meta.appendChild(el("span", "badge", modelBadge(call)));
      const clinic = clinicCode(call);
      if (clinic) meta.appendChild(el("span", "", clinic));
      meta.appendChild(el("span", "", fmtDuration(call.duration_sec)));
      const when = fmtTime(call.started_at);
      if (when) meta.appendChild(el("span", "", when));
      btn.appendChild(meta);
      btn.addEventListener("click", function () {
        loadDetail(call.id).catch(function (err) {
          showNotice((err && err.message) || "Could not load call.");
        });
      });
      listEl.appendChild(btn);
    });
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function inlineMd(value) {
    return escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
  }

  function renderMarkdown(md) {
    const lines = String(md || "").split(/\r?\n/);
    let html = "";
    let inList = false;
    function closeList() {
      if (!inList) return;
      html += "</ul>";
      inList = false;
    }
    lines.forEach(function (line) {
      const t = line.trim();
      if (!t) { closeList(); return; }
      if (t === "---") { closeList(); html += "<hr>"; return; }
      const heading = /^(#{1,3})\s+(.*)$/.exec(t);
      if (heading) {
        closeList();
        const level = heading[1].length;
        html += "<h" + level + ">" + inlineMd(heading[2]) + "</h" + level + ">";
        return;
      }
      const item = /^[-*]\s+(.*)$/.exec(t);
      if (item) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += "<li>" + inlineMd(item[1]) + "</li>";
        return;
      }
      closeList();
      html += "<p>" + inlineMd(t) + "</p>";
    });
    closeList();
    return html || "<p>The report is not ready yet.</p>";
  }

  function noteText(val) {
    if (typeof val === "string") return val.trim();
    const obj = asObject(val);
    return String(obj.verbatim || obj.note || obj.notes || "").trim();
  }

  function cursorDetail(item) {
    const bits = [];
    const ids = asList(item && item.child_ids).map(function (id) { return String(id || "").trim(); }).filter(Boolean);
    if (ids.length) bits.push(ids.join(", "));
    const mainNote = noteText(item && (item.note || item.notes));
    if (mainNote) bits.push(mainNote);
    const followups = asObject(item && item.followups);
    Object.keys(followups).sort().forEach(function (fid) {
      const row = followups[fid];
      const fNote = typeof row === "string" ? row.trim() : noteText(asObject(row).note || row);
      if (fNote && fNote !== mainNote) bits.push(fid + ": " + fNote);
    });
    return bits.join(" · ");
  }

  function renderFindings(cursorState) {
    findingsEl.textContent = "";
    const answers = asObject(asObject(cursorState).answers);
    const rank = { concern: 0, unknown: 1, clear: 2 };
    const keys = Object.keys(answers).filter(function (qid) {
      return String(asObject(answers[qid]).state || "") !== "not_applicable";
    });
    keys.sort(function (a, b) {
      const sa = String(asObject(answers[a]).state || "");
      const sb = String(asObject(answers[b]).state || "");
      const ra = Object.prototype.hasOwnProperty.call(rank, sa) ? rank[sa] : 3;
      const rb = Object.prototype.hasOwnProperty.call(rank, sb) ? rank[sb] : 3;
      return ra - rb || String(a).localeCompare(String(b));
    });
    if (!keys.length) {
      findingsEl.appendChild(el("p", "calm", "No structured findings were recorded."));
      return;
    }
    const table = el("table");
    const head = el("tr");
    ["Question ID", "State", "Detail", "Red flag"].forEach(function (label) {
      head.appendChild(el("th", "", label));
    });
    const thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    const tbody = el("tbody");
    keys.forEach(function (qid) {
      const item = asObject(answers[qid]);
      const state = String(item.state || "");
      const flagged = state === "concern" && qid !== "ever_pregnant";
      const row = el("tr", flagged ? "flag" : "");
      row.appendChild(el("td", "", qid));
      row.appendChild(el("td", "", state));
      row.appendChild(el("td", "", cursorDetail(item)));
      row.appendChild(el("td", "", flagged ? "Yes" : ""));
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    findingsEl.appendChild(table);
  }

  function renderTranscript(turns) {
    transcriptEl.textContent = "";
    const list = asList(turns);
    if (!list.length) {
      transcriptEl.appendChild(el("p", "calm", "No transcript was recorded."));
      return;
    }
    list.forEach(function (turn) {
      const role = turn.role || turn.speaker || "";
      const who = role === "assistant" ? "SHISHU" : (role === "user" ? "Mother" : (role || "Note"));
      const block = el("div", role === "assistant" ? "turn shishu" : "turn");
      block.appendChild(el("strong", "", who));
      block.appendChild(el("p", "", turn.text || ""));
      transcriptEl.appendChild(block);
    });
  }

  function renderFlags(flags) {
    flagList.textContent = "";
    const items = asList(flags);
    if (!items.length) {
      flagBanner.hidden = true;
      noFlags.hidden = false;
      return;
    }
    noFlags.hidden = true;
    flagBanner.hidden = false;
    items.forEach(function (item) {
      flagList.appendChild(el("li", "", typeof item === "string" ? item : JSON.stringify(item)));
    });
  }

  function showDetail(call) {
    current = call;
    selectedId = call.id;
    selectedStatus = call.status || "";
    emptyEl.hidden = true;
    reportEl.hidden = false;
    titleEl.textContent = "Call #" + call.id;
    metaEl.textContent = [
      callerLine(call),
      people(call),
      clinicCode(call),
      modelBadge(call),
      fmtDuration(call.duration_sec),
      fmtTime(call.started_at),
      call.status || ""
    ].filter(Boolean).join(" · ");
    const hasAudio = Boolean(String(call.audio_path || "").trim());
    const src = "/api/audio/" + encodeURIComponent(call.id);
    if (hasAudio) {
      player.hidden = false;
      player.src = src;
      download.hidden = false;
      download.href = src;
      download.setAttribute("download", "call_" + call.id + ".wav");
      audioNote.hidden = true;
    } else {
      if (typeof player.removeAttribute === "function") player.removeAttribute("src");
      else player.src = "";
      if (typeof player.load === "function") player.load();
      player.hidden = true;
      download.hidden = true;
      if (typeof download.removeAttribute === "function") download.removeAttribute("href");
      else download.href = "";
      audioNote.hidden = false;
    }
    const motherPdfEl = $("mother-pdf");
    motherPdfEl.href = "/api/calls/" + encodeURIComponent(call.id) + "/mother.pdf";
    const isActive = call.status === "active";
    motherPdfEl.classList.toggle("disabled", isActive);
    motherPdfEl.setAttribute("aria-disabled", isActive ? "true" : "false");
    renderFlags(call.red_flags);
    doctorEl.innerHTML = renderMarkdown(call.summary_doctor || "");
    motherEl.textContent = call.summary_mother || "The mother summary is not ready yet.";
    renderFindings(call.cursor_state);
    renderTranscript(call.transcript);
    renderTelemetry(call);
    paintReview(call);
    renderList();
  }

  function paintReview(call) {
    const review = (call && call.clinician_review) || {};
    const note = $("review-note");
    const status = $("review-status");
    const sameCall = Boolean(call && reviewPaintedCallId === call.id);
    const focused = Boolean(note && typeof document !== "undefined" && document.activeElement === note);
    if (note && !(sameCall && (reviewDirtyCallId === call.id || focused))) {
      note.value = review.note || "";
      reviewDirtyCallId = null;
    }
    reviewPaintedCallId = call ? call.id : null;
    document.querySelectorAll("#review-bar [data-rating]").forEach(function (btn) {
      btn.classList.toggle("on", btn.getAttribute("data-rating") === review.rating);
    });
    if (!status) return;
    status.hidden = !review.rating;
    status.textContent = review.rating ? ("Saved: " + review.rating) : "";
  }

  async function saveReview(rating) {
    if (!current) {
      showNotice("Select a call before saving a review.");
      return;
    }
    const noteEl = $("review-note");
    const note = noteEl ? noteEl.value.trim() : "";
    const res = await api("/api/calls/" + encodeURIComponent(current.id) + "/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating: rating, note: note })
    });
    if (!res) return;
    const data = await res.json();
    current.clinician_review = data.clinician_review || { rating: rating, note: note };
    const row = calls.find(function (call) { return call.id === current.id; });
    if (row) row.clinician_review = current.clinician_review;
    reviewDirtyCallId = null;
    paintReview(current);
    showNotice("Review saved.");
  }

  function dashNum(value, suffix) {
    if (value == null || value === "") return "—";
    const n = Number(value);
    if (Number.isNaN(n)) return String(value);
    const shown = Number.isInteger(n) ? String(n) : String(Math.round(n * 10) / 10);
    return suffix ? shown + suffix : shown;
  }

  function kpi(label, value) {
    const box = el("div", "kpi");
    box.appendChild(el("span", "", label));
    box.appendChild(el("strong", "", value));
    return box;
  }

  function fmtOffset(sec) {
    const totalTenths = Math.round(Math.max(0, Number(sec) || 0) * 10);
    const totalSec = Math.floor(totalTenths / 10);
    return "+" + pad(Math.floor(totalSec / 60)) + ":" + pad(totalSec % 60) + "." + (totalTenths % 10);
  }

  function eventBadgeClass(type) {
    if (type === "error" || type === "bug_reported") return "pill hot";
    if (type === "watchdog_nudge" || type === "manual_nudge") return "pill warn";
    if (type === "session_started" || type === "tool_call" || type === "bot_first_audio" || type === "decision_turn" || type === "user_transcript") return "badge";
    return "pill";
  }

  function formatEventMeta(meta) {
    const obj = asObject(meta);
    const parts = [];
    Object.keys(obj).forEach(function (key) {
      const val = obj[key];
      if (val == null || val === "" || typeof val === "object") return;
      parts.push(key + "=" + String(val));
    });
    return parts.join(" ");
  }

  function renderTelemetry(call) {
    const grid = $("telemetry-grid");
    grid.textContent = "";
    const t = asObject(call.telemetry);
    const latencies = asList(t.turn_latencies_ms);
    const avgText = latencies.length ? dashNum(t.avg_latency_ms, " ms") : "—";
    const maxText = latencies.length ? dashNum(t.max_latency_ms, " ms") : "—";
    const latency = kpi("Turn Latency (Avg / Max)", avgText + " / " + maxText);
    const pills = el("div", "latency-pills");
    latencies.forEach(function (ms) { pills.appendChild(el("span", "pill", dashNum(ms, " ms"))); });
    if (pills.childNodes.length) latency.appendChild(pills);
    grid.appendChild(kpi("Caller & Role", (call.caller_name || "—") + " (" + (call.user_role || "—") + ")"));
    grid.appendChild(kpi("Model / Voice", (call.model_used || "—") + " · " + (call.voice_used || "default")));
    grid.appendChild(kpi("Connect Latency", dashNum(t.connect_ms, " ms")));
    grid.appendChild(latency);
    grid.appendChild(kpi("Turns (User / Bot)", dashNum(t.user_turns) + " / " + dashNum(t.bot_turns)));
    grid.appendChild(kpi("Tool Calls & Nudges", dashNum(t.tool_calls) + " tool calls / " + dashNum(t.watchdog_nudges) + " nudges"));
    grid.appendChild(kpi("Audio Packets (In / Out)", dashNum(t.audio_chunks_in) + " / " + dashNum(t.audio_chunks_out)));
    const stopped = String(t.dropoff_qid || "").trim();
    if (stopped) grid.appendChild(kpi("Stopped at", stopped));
    renderEvents(asList(call.events));
  }

  function renderEvents(events) {
    const summary = $("event-summary");
    const host = $("event-table");
    summary.textContent = "Under-the-Hood Event Trace (" + events.length + ")";
    host.textContent = "";
    if (!events.length) {
      host.appendChild(el("p", "calm", "No events were recorded."));
      return;
    }
    let origin = null;
    events.forEach(function (evt) {
      const t = Number(evt && evt.t);
      if (Number.isFinite(t) && (origin == null || t < origin)) origin = t;
    });
    const table = el("table");
    const head = el("tr");
    ["Time", "Type", "Detail"].forEach(function (label) { head.appendChild(el("th", "", label)); });
    const thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    const tbody = el("tbody");
    events.forEach(function (evt) {
      const row = el("tr");
      const t = Number(evt && evt.t);
      const timeCell = el("td", "", Number.isFinite(t) && origin != null ? fmtOffset(t - origin) : "—");
      if (Number.isFinite(t) && t > 1e9) timeCell.title = new Date(t * 1000).toLocaleString();
      row.appendChild(timeCell);
      const type = String((evt && (evt.type || evt.event)) || "event");
      const typeCell = el("td");
      typeCell.appendChild(el("span", eventBadgeClass(type), type));
      row.appendChild(typeCell);
      const detail = evt && evt.detail != null ? String(evt.detail) : "";
      const metaText = formatEventMeta(evt && evt.meta);
      row.appendChild(el("td", "", [detail, metaText].filter(Boolean).join(" · ")));
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    host.appendChild(table);
  }

  async function copyText(text, okMessage) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    showNotice(okMessage);
  }

  function copyDebug() {
    if (!current) return Promise.resolve();
    const bundle = {
      call_id: current.id,
      telemetry: asObject(current.telemetry),
      events: asList(current.events),
      transcript: asList(current.transcript)
    };
    return copyText(JSON.stringify(bundle, null, 2), "Debug bundle copied.");
  }

  function asBugs(data) {
    if (Array.isArray(data)) return data;
    return (data && Array.isArray(data.bugs)) ? data.bugs : [];
  }

  function showTab(name) {
    const callsOn = name !== "bugs";
    $("tab-calls").setAttribute("aria-selected", callsOn ? "true" : "false");
    $("tab-bugs").setAttribute("aria-selected", callsOn ? "false" : "true");
    $("calls-pane").hidden = !callsOn;
    $("bugs-pane").hidden = callsOn;
  }

  function sanitizePreview(value) {
    if (typeof value === "string") return value.indexOf("data:image/") === 0 ? "[screenshot attached]" : value;
    if (Array.isArray(value)) return value.map(sanitizePreview);
    if (value && typeof value === "object") {
      const out = {};
      Object.keys(value).forEach(function (k) { out[k] = sanitizePreview(value[k]); });
      return out;
    }
    return value;
  }

  function renderBugs() {
    const list = $("bug-list");
    list.textContent = "";
    $("tab-bugs").textContent = "Bug Reports (" + bugs.length + ")";
    if (!bugs.length) {
      list.appendChild(el("p", "empty", "No bug reports yet."));
      return;
    }
    bugs.forEach(function (bug) {
      const item = el("article", "bug-item");
      const top = el("div", "row");
      top.appendChild(el("span", "badge", bug.category || "Uncategorized"));
      const callId = Number(bug.call_id) || 0;
      if (callId > 0) {
        const link = el("button", "bug-link", "Call #" + callId);
        link.type = "button";
        link.addEventListener("click", function () {
          if (clinicFilter) {
            const match = calls.find(function (c) { return c.id === callId; });
            if (!match || clinicCode(match) !== clinicFilter) {
              clinicFilter = "";
              const select = $("clinic-filter");
              if (select) select.value = "";
            }
          }
          showTab("calls");
          loadDetail(callId).catch(function (err) { showNotice(err.message || "Could not load call."); });
        });
        top.appendChild(link);
      } else {
        top.appendChild(el("span", "meta", "No call"));
      }
      item.appendChild(top);
      const sec = Math.max(0, Math.round(Number(bug.call_elapsed_sec) || 0));
      const bits = [fmtDuration(sec) + " (" + sec + "s)"];
      const when = fmtTime(bug.created_at);
      if (when) bits.push(when);
      item.appendChild(el("p", "meta", bits.join(" · ")));
      item.appendChild(el("p", "bug-note", bug.note || "No note."));
      if (bug.snapshot && typeof bug.snapshot === "object") {
        const details = document.createElement("details");
        details.className = "snapshot";
        const summary = document.createElement("summary");
        summary.textContent = "Snapshot";
        details.appendChild(summary);
        const shot = bug.snapshot.client && typeof bug.snapshot.client.screenshot === "string"
          ? bug.snapshot.client.screenshot
          : "";
        if (shot.indexOf("data:image/") === 0) {
          const img = document.createElement("img");
          img.className = "bug-shot-preview";
          img.alt = "Bug screenshot";
          img.src = shot;
          details.appendChild(img);
        }
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(sanitizePreview(bug.snapshot), null, 2);
        details.appendChild(pre);
        item.appendChild(details);
      }
      const copy = el("button", "ghost", "Copy Bug JSON");
      copy.type = "button";
      copy.addEventListener("click", function () {
        copyText(JSON.stringify(bug, null, 2), "Bug JSON copied.").catch(function (err) {
          showNotice(err.message || "Could not copy the bug.");
        });
      });
      item.appendChild(copy);
      list.appendChild(item);
    });
  }

  async function loadBugs() {
    if (loadingBugs) return;
    loadingBugs = true;
    try {
      const res = await api("/api/bugs");
      if (!res) return;
      bugs = asBugs(await res.json());
      stampUpdated($("bugs-updated"));
      renderBugs();
    } finally {
      loadingBugs = false;
    }
  }

  async function logBug(event) {
    event.preventDefault();
    if (!current) {
      showNotice("Select a call before logging a bug.");
      return;
    }
    const btn = $("log-bug");
    btn.disabled = true;
    const callId = current.id;
    try {
      const res = await api("/api/bugs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          call_id: callId,
          category: $("bug-category").value,
          note: $("bug-note").value.trim(),
          call_elapsed_sec: Math.max(0, Math.round(Number(current.duration_sec) || 0)),
          client_state: { source: "admin" }
        })
      });
      if (!res) return;
      const data = await res.json();
      $("bug-note").value = "";
      showNotice("Bug logged on call #" + callId + ".");
      if (data && data.bug) {
        bugs = [data.bug].concat(bugs.filter(function (bug) { return bug.id !== data.bug.id; }));
        renderBugs();
      } else {
        await loadBugs();
      }
      await loadDetail(callId);
    } catch (err) {
      showNotice(err.message || "Could not log the bug.");
    } finally {
      btn.disabled = false;
    }
  }

  async function loadDetail(id) {
    const seq = ++detailSeq;
    selectedId = id;
    renderList();
    let loaded = false;
    try {
      const res = await api("/api/calls/" + encodeURIComponent(id));
      if (!res || seq !== detailSeq) return;
      const call = await res.json();
      if (seq !== detailSeq) return;
      loaded = true;
      showDetail(call);
    } finally {
      if (!loaded && seq === detailSeq) {
        selectedId = current ? current.id : null;
        renderList();
      }
    }
  }

  async function refreshList() {
    if (loadingList) {
      refreshQueued = true;
      return;
    }
    loadingList = true;
    const btn = $("refresh");
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Refreshing...";
    try {
      do {
        refreshQueued = false;
        const res = await api("/api/calls", { cache: "no-store" });
        if (!res) return;
        calls = asCalls(await res.json());
        stampUpdated(updatedEl);
        renderList();
        const currentRow = calls.find(function (call) { return call.id === selectedId; });
        if (currentRow && currentRow.status !== selectedStatus) await loadDetail(selectedId);
      } while (refreshQueued);
    } finally {
      loadingList = false;
      btn.disabled = false;
      btn.textContent = label;
    }
  }

  function printReport() {
    if (!current) return;
    document.body.classList.add("printing");
    function cleanup() {
      document.body.classList.remove("printing");
      window.removeEventListener("afterprint", cleanup);
    }
    window.addEventListener("afterprint", cleanup);
    window.print();
  }

  player.addEventListener("error", function () {
    if (player.getAttribute("src")) audioNote.hidden = false;
  });

  $("refresh").addEventListener("click", function () {
    refreshList().catch(function (err) { showNotice(err.message); });
  });
  $("copy-report").addEventListener("click", function () {
    const text = current && current.summary_doctor ? current.summary_doctor : "";
    copyText(text, "Report copied.").catch(function (err) {
      showNotice(err.message || "Could not copy the report.");
    });
  });
  $("print-report").addEventListener("click", printReport);
  $("logout-btn").addEventListener("click", async function () {
    try {
      await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
    } catch (err) {
      /* redirect anyway */
    } finally {
      location.href = "/login";
    }
  });
  $("tab-calls").addEventListener("click", function () { showTab("calls"); });
  $("tab-bugs").addEventListener("click", function () {
    showTab("bugs");
    loadBugs().catch(function (err) { showNotice(err.message || "Could not load bug reports."); });
  });
  $("refresh-bugs").addEventListener("click", function () {
    loadBugs().catch(function (err) { showNotice(err.message || "Could not load bug reports."); });
  });
  $("copy-debug").addEventListener("click", function () {
    copyDebug().catch(function (err) { showNotice(err.message || "Could not copy the debug bundle."); });
  });
  $("bug-form").addEventListener("submit", function (event) {
    logBug(event).catch(function (err) { showNotice(err.message || "Could not log the bug."); });
  });
  $("clinic-filter").addEventListener("change", function (event) {
    clinicFilter = event.target.value || "";
    renderList();
  });
  $("review-bar").addEventListener("click", function (event) {
    const bar = $("review-bar");
    const btn = event.target.closest("[data-rating]");
    if (!btn || !bar.contains(btn)) return;
    saveReview(btn.getAttribute("data-rating")).catch(function (err) {
      showNotice(err.message || "Could not save the review.");
    });
  });
  const reviewNoteEl = $("review-note");
  if (reviewNoteEl) {
    reviewNoteEl.addEventListener("input", function () {
      if (current) reviewDirtyCallId = current.id;
    });
  }
  const saveReviewNoteBtn = $("save-review-note");
  if (saveReviewNoteBtn) {
    saveReviewNoteBtn.addEventListener("click", function () {
      if (!current) {
        showNotice("Select a call before saving a note.");
        return;
      }
      const rating = current.clinician_review && current.clinician_review.rating;
      if (!rating) {
        showNotice("Select Useful, Missed Flag, or False Alarm before saving a note.");
        return;
      }
      saveReview(rating).catch(function (err) {
        showNotice((err && err.message) || "Could not save the review.");
      });
    });
  }
  $("mother-pdf").addEventListener("click", async function (event) {
    event.preventDefault();
    if (!current) return;
    if (current.status === "active") {
      showNotice("Mother PDF is not ready while the call is active.");
      return;
    }
    const url = "/api/calls/" + encodeURIComponent(current.id) + "/mother.pdf";
    try {
      const res = await fetch(url, { credentials: "same-origin" });
      if (res.status === 401) {
        loginRedirect();
        return;
      }
      if (!res.ok) {
        showNotice(await readErrorMessage(res, "Mother PDF is not ready yet (" + res.status + ")."));
        return;
      }
      const match = /filename="?([^";]+)"?/.exec(res.headers.get("Content-Disposition") || "");
      const name = (match && match[1]) || ("SHISHU-" + current.id + ".pdf");
      const objUrl = URL.createObjectURL(await res.blob());
      const link = document.createElement("a");
      link.href = objUrl;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(function () { URL.revokeObjectURL(objUrl); }, 1000);
    } catch (err) {
      showNotice((err && err.message) || "Could not download the Mother PDF.");
    }
  });

  async function boot() {
    const callsRes = await api("/api/calls", { cache: "no-store" });
    if (!callsRes) return;
    calls = asCalls(await callsRes.json());
    stampUpdated(updatedEl);
    const wanted = Number(new URLSearchParams(location.search).get("call"));
    const pick = calls.find(function (call) { return call.id === wanted; }) || calls[0];
    if (pick) await loadDetail(pick.id);
    else renderList();
    loadBugs().catch(function (err) { showNotice(err.message || "Could not load bug reports."); });
    refreshTimer = setInterval(function () {
      refreshList().catch(function (err) { showNotice(err.message); });
    }, 15000);
  }

  boot().catch(function (err) { showNotice(err.message || "Could not load the dashboard."); });

  window.addEventListener("beforeunload", function () {
    if (refreshTimer) clearInterval(refreshTimer);
  });
})();
