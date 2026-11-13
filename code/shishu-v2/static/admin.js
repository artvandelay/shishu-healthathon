    (function () {
      const listEl = document.getElementById("call-list");
      const emptyEl = document.getElementById("empty");
      const reportEl = document.getElementById("report");
      const noticeEl = document.getElementById("notice");
      const updatedEl = document.getElementById("updated");
      const titleEl = document.getElementById("call-title");
      const metaEl = document.getElementById("call-meta");
      const player = document.getElementById("player");
      const download = document.getElementById("download");
      const audioNote = document.getElementById("audio-note");
      const flagBanner = document.getElementById("flag-banner");
      const flagList = document.getElementById("flag-list");
      const noFlags = document.getElementById("no-flags");
      const doctorEl = document.getElementById("doctor");
      const motherEl = document.getElementById("mother");
      const findingsEl = document.getElementById("findings");
      const transcriptEl = document.getElementById("transcript");

      let calls = [];
      let bugs = [];
      let selectedId = null;
      let selectedStatus = "";
      let current = null;
      let refreshTimer = 0;
      let loadingList = false;
      let refreshQueued = false;
      let loadingBugs = false;
      let settingsCache = {};
      let clinicFilter = "";

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

      async function api(path, options) {
        const opts = options ? Object.assign({}, options) : {};
        opts.credentials = "same-origin";
        const headers = Object.assign({ "Accept": "application/json" }, opts.headers || {});
        opts.headers = headers;
        const res = await fetch(path, opts);
        if (res.status === 401) {
          loginRedirect();
          return null;
        }
        if (!res.ok) {
          let detail = "Request failed (" + res.status + ")";
          try {
            const body = await res.json();
            if (body && (body.detail || body.message)) detail = body.detail || body.message;
          } catch (err) { /* keep the status text */ }
          throw new Error(typeof detail === "string" ? detail : "Request failed");
        }
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
        if (data && Array.isArray(data.calls)) return data.calls;
        return [];
      }

      function engineBadge(call) {
        const model = String((call && call.model_used) || "").trim();
        const engine = String((call && call.engine) || call || "").toLowerCase();
        const v = (model || engine).toLowerCase();
        if (!v) return "—";
        if (v.indexOf("openai") !== -1 || v.indexOf("gpt") !== -1) return "gpt-live-1";
        if (v.indexOf("gemini-3.1") !== -1 || v.indexOf("gemini") !== -1) return "Gemini 3.1 Flash Live";
        return model || String(call.engine || call);
      }

      function flagCount(call) {
        if (Array.isArray(call.red_flags)) return call.red_flags.length;
        if (typeof call.red_flag_count === "number") return call.red_flag_count;
        return asList(call.red_flags).length;
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
        const mother = call.mother_name || "Mother not named";
        const childBits = [call.child_name, call.child_age].filter(Boolean).join(", ");
        return childBits ? mother + " · " + childBits : mother;
      }

      function labelize(key) {
        return String(key || "")
          .replace(/^profile_/, "")
          .replace(/_/g, " ")
          .replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
      }

      function clinicCode(call) {
        return String((call && call.clinic_code) || "").trim();
      }

      function fillClinicFilter() {
        const select = document.getElementById("clinic-filter");
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
        if (previous && codes.indexOf(previous) !== -1) {
          clinicFilter = previous;
          select.value = previous;
        } else {
          clinicFilter = "";
          select.value = "";
        }
      }

      function visibleCalls() {
        if (!clinicFilter) return calls;
        return calls.filter(function (call) { return clinicCode(call) === clinicFilter; });
      }

      function renderList() {
        fillClinicFilter();
        const shown = visibleCalls();
        document.getElementById("tab-calls").textContent = "Calls (" + shown.length + ")";
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
          const badgeName = engineBadge(call);
          const badge = el("span", badgeName === "gpt-live-1" ? "badge openai" : "badge", badgeName);
          meta.appendChild(badge);
          const clinic = clinicCode(call);
          if (clinic) meta.appendChild(el("span", "", clinic));
          meta.appendChild(el("span", "", fmtDuration(call.duration_sec)));
          const when = fmtTime(call.started_at);
          if (when) meta.appendChild(el("span", "", when));
          btn.appendChild(meta);
          btn.addEventListener("click", function () { loadDetail(call.id); });
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
          if (inList) {
            html += "</ul>";
            inList = false;
          }
        }
        lines.forEach(function (line) {
          const t = line.trim();
          if (!t) {
            closeList();
            return;
          }
          if (t === "---") {
            closeList();
            html += "<hr>";
            return;
          }
          const heading = /^(#{1,3})\s+(.*)$/.exec(t);
          if (heading) {
            closeList();
            const level = heading[1].length;
            html += "<h" + level + ">" + inlineMd(heading[2]) + "</h" + level + ">";
            return;
          }
          const item = /^[-*]\s+(.*)$/.exec(t);
          if (item) {
            if (!inList) {
              html += "<ul>";
              inList = true;
            }
            html += "<li>" + inlineMd(item[1]) + "</li>";
            return;
          }
          closeList();
          html += "<p>" + inlineMd(t) + "</p>";
        });
        closeList();
        return html || "<p>The report is not ready yet.</p>";
      }

      function cursorDetail(item) {
        const bits = [];
        const ids = asList(item && item.child_ids).map(function (id) {
          return String(id || "").trim();
        }).filter(Boolean);
        if (ids.length) bits.push(ids.join(", "));
        const notes = String((item && item.notes) || "").trim();
        if (notes) bits.push(notes);
        const followups = asObject(item && item.followups);
        Object.keys(followups).sort().forEach(function (fid) {
          const row = followups[fid];
          const note = typeof row === "string"
            ? row.trim()
            : String((asObject(row).note || asObject(row).notes || "")).trim();
          if (!note || note === notes) return;
          bits.push(fid + ": " + note);
        });
        return bits.join(" · ");
      }

      function renderCursorAnswers(cursorState) {
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
        ["Question ID", "State", "Detail", "Flag"].forEach(function (label) {
          head.appendChild(el("th", "", label));
        });
        const thead = el("thead");
        thead.appendChild(head);
        table.appendChild(thead);
        const tbody = el("tbody");
        keys.forEach(function (qid) {
          const item = asObject(answers[qid]);
          const state = String(item.state || "");
          const flagged = state === "concern" && qid !== "intro_bridge";
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

      function renderFindings(findings, cursorState) {
        findingsEl.textContent = "";
        const data = asObject(findings);
        if (Object.keys(data).length === 0) {
          renderCursorAnswers(cursorState);
          return;
        }
        const keys = Object.keys(data).sort(function (a, b) {
          function rank(key) {
            if (String(key).indexOf("profile_") === 0) return 0;
            if (key === "chief_complaint") return 1;
            return 2;
          }
          return rank(a) - rank(b) || String(a).localeCompare(String(b));
        });
        if (!keys.length) {
          findingsEl.appendChild(el("p", "calm", "No structured findings were recorded."));
          return;
        }
        const table = el("table");
        const head = el("tr");
        ["Domain", "Status", "Detail", "Red flag"].forEach(function (label) {
          head.appendChild(el("th", "", label));
        });
        const thead = el("thead");
        thead.appendChild(head);
        table.appendChild(thead);
        const tbody = el("tbody");
        keys.forEach(function (key) {
          const item = data[key] || {};
          const flagged = item.is_red_flag === true || item.is_red_flag === 1 || item.is_red_flag === "true";
          const row = el("tr", flagged ? "flag" : "");
          row.appendChild(el("td", "", labelize(key)));
          row.appendChild(el("td", "", item.status || ""));
          const detail = item.detail == null ? "" : (typeof item.detail === "string" ? item.detail : JSON.stringify(item.detail));
          row.appendChild(el("td", "", detail));
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
          const text = typeof item === "string" ? item : JSON.stringify(item);
          flagList.appendChild(el("li", "", text));
        });
      }

      function showDetail(call) {
        current = call;
        selectedId = call.id;
        selectedStatus = call.status || "";
        emptyEl.hidden = true;
        reportEl.hidden = false;
        titleEl.textContent = "Call #" + call.id;
        const meta = [
          callerLine(call),
          people(call),
          clinicCode(call),
          call.child_gender || "",
          engineBadge(call),
          fmtDuration(call.duration_sec),
          fmtTime(call.started_at),
          call.status || ""
        ].filter(Boolean);
        metaEl.textContent = meta.join(" · ");
        const src = "/api/audio/" + encodeURIComponent(call.id);
        player.src = src;
        download.href = src;
        download.setAttribute("download", "call_" + call.id + ".wav");
        document.getElementById("mother-pdf").href = "/api/calls/" + encodeURIComponent(call.id) + "/mother.pdf";
        audioNote.hidden = true;
        renderFlags(call.red_flags);
        doctorEl.innerHTML = renderMarkdown(call.summary_doctor || "");
        motherEl.textContent = call.summary_mother || "The mother summary is not ready yet.";
        renderFindings(call.findings, call.cursor_state);
        renderTranscript(call.transcript);
        renderTelemetry(call);
        paintReview(call);
        renderList();
      }

      function paintReview(call) {
        const review = (call && call.clinician_review) || {};
        const note = document.getElementById("review-note");
        const status = document.getElementById("review-status");
        if (note) note.value = review.note || "";
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
        const noteEl = document.getElementById("review-note");
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
        const tenths = totalTenths % 10;
        const totalSec = Math.floor(totalTenths / 10);
        return "+" + pad(Math.floor(totalSec / 60)) + ":" + pad(totalSec % 60) + "." + tenths;
      }

      function eventBadgeClass(type) {
        if (type === "error" || type === "bug_reported") return "pill hot";
        if (type === "watchdog_nudge" || type === "manual_nudge") return "pill warn";
        if (type === "session_started" || type === "tool_call" || type === "bot_first_audio") return "badge";
        if (type === "user_transcript") return "badge openai";
        return "pill";
      }

      function renderTelemetry(call) {
        const grid = document.getElementById("telemetry-grid");
        grid.textContent = "";
        const t = asObject(call.telemetry);
        const caller = (call.caller_name || "—") + " (" + (call.user_role || "—") + ")";
        const model = (call.model_used || call.engine || "—") + " · " + (call.voice_used || "default");
        const latency = kpi(
          "Turn Latency (Avg / Max)",
          dashNum(t.avg_latency_ms, " ms") + " / " + dashNum(t.max_latency_ms, " ms")
        );
        const pills = el("div", "latency-pills");
        asList(t.turn_latencies_ms).forEach(function (ms) {
          pills.appendChild(el("span", "pill", dashNum(ms, " ms")));
        });
        if (pills.childNodes.length) latency.appendChild(pills);
        grid.appendChild(kpi("Caller & Role", caller));
        grid.appendChild(kpi("Model / Voice", model));
        grid.appendChild(kpi("Connect Latency", dashNum(t.connect_ms, " ms")));
        grid.appendChild(latency);
        grid.appendChild(kpi("Turns (User / Bot)", dashNum(t.user_turns) + " / " + dashNum(t.bot_turns)));
        grid.appendChild(kpi(
          "Tool Calls & Nudges",
          dashNum(t.tool_calls) + " findings / " + dashNum(t.watchdog_nudges) + " nudges"
        ));
        grid.appendChild(kpi(
          "Audio Packets (In / Out)",
          dashNum(t.audio_chunks_in) + " / " + dashNum(t.audio_chunks_out)
        ));
        renderEvents(asList(call.events));
      }

      function renderEvents(events) {
        const summary = document.getElementById("event-summary");
        const host = document.getElementById("event-table");
        summary.textContent = "Under-the-Hood Event Trace (" + events.length + ")";
        host.textContent = "";
        if (!events.length) {
          host.appendChild(el("p", "calm", "No events were recorded."));
          return;
        }
        let origin = null;
        events.forEach(function (evt) {
          const t = Number(evt && evt.t);
          if (!Number.isFinite(t)) return;
          if (origin == null || t < origin) origin = t;
        });
        const table = el("table");
        const head = el("tr");
        ["Time", "Type", "Detail"].forEach(function (label) {
          head.appendChild(el("th", "", label));
        });
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
          row.appendChild(el("td", "", evt && evt.detail != null ? String(evt.detail) : ""));
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
          transcript: asList(current.transcript),
          settings: settingsCache
        };
        return copyText(JSON.stringify(bundle, null, 2), "Debug bundle copied.");
      }

      function asBugs(data) {
        if (Array.isArray(data)) return data;
        if (data && Array.isArray(data.bugs)) return data.bugs;
        return [];
      }

      function paintBugsTab() {
        document.getElementById("tab-bugs").textContent = "Bug Reports (" + bugs.length + ")";
      }

      function showTab(name) {
        const callsOn = name !== "bugs";
        document.getElementById("tab-calls").setAttribute("aria-selected", callsOn ? "true" : "false");
        document.getElementById("tab-bugs").setAttribute("aria-selected", callsOn ? "false" : "true");
        document.getElementById("calls-pane").hidden = !callsOn;
        document.getElementById("bugs-pane").hidden = callsOn;
      }

      function renderBugs() {
        const list = document.getElementById("bug-list");
        list.textContent = "";
        paintBugsTab();
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
              showTab("calls");
              loadDetail(callId).catch(function (err) { showNotice(err.message); });
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
            const pre = document.createElement("pre");
            pre.textContent = JSON.stringify(bug.snapshot, null, 2);
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
          document.getElementById("bugs-updated").textContent = "Updated " + new Date().toLocaleTimeString(undefined, {
            hour: "2-digit",
            minute: "2-digit"
          });
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
        const btn = document.getElementById("log-bug");
        btn.disabled = true;
        const callId = current.id;
        try {
          const res = await api("/api/bugs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              call_id: callId,
              category: document.getElementById("bug-category").value,
              note: document.getElementById("bug-note").value.trim(),
              call_elapsed_sec: Math.max(0, Math.round(Number(current.duration_sec) || 0)),
              client_state: { source: "admin" }
            })
          });
          if (!res) return;
          const data = await res.json();
          document.getElementById("bug-note").value = "";
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
        selectedId = id;
        renderList();
        const res = await api("/api/calls/" + encodeURIComponent(id));
        if (!res) return;
        const call = await res.json();
        showDetail(call);
      }

      async function refreshList() {
        if (loadingList) {
          refreshQueued = true;
          return;
        }
        loadingList = true;
        const btn = document.getElementById("refresh");
        const label = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Refreshing...";
        try {
          do {
            refreshQueued = false;
            const res = await api("/api/calls", { cache: "no-store" });
            if (!res) return;
            calls = asCalls(await res.json());
            updatedEl.textContent = "Updated " + new Date().toLocaleTimeString(undefined, {
              hour: "2-digit",
              minute: "2-digit"
            });
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

      async function copyReport() {
        const text = current && current.summary_doctor ? current.summary_doctor : "";
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
        showNotice("Report copied.");
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

      document.getElementById("refresh").addEventListener("click", function () {
        refreshList().catch(function (err) { showNotice(err.message); });
      });
      document.getElementById("copy-report").addEventListener("click", function () {
        copyReport().catch(function (err) { showNotice(err.message || "Could not copy the report."); });
      });
      document.getElementById("print-report").addEventListener("click", printReport);
      document.getElementById("logout-btn").addEventListener("click", async function () {
        await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
        location.href = "/login";
      });
      document.getElementById("tab-calls").addEventListener("click", function () { showTab("calls"); });
      document.getElementById("tab-bugs").addEventListener("click", function () {
        showTab("bugs");
        loadBugs().catch(function (err) { showNotice(err.message || "Could not load bug reports."); });
      });
      document.getElementById("refresh-bugs").addEventListener("click", function () {
        loadBugs().catch(function (err) { showNotice(err.message || "Could not load bug reports."); });
      });
      document.getElementById("copy-debug").addEventListener("click", function () {
        copyDebug().catch(function (err) { showNotice(err.message || "Could not copy the debug bundle."); });
      });
      document.getElementById("bug-form").addEventListener("submit", function (event) {
        logBug(event).catch(function (err) { showNotice(err.message || "Could not log the bug."); });
      });
      document.getElementById("clinic-filter").addEventListener("change", function (event) {
        clinicFilter = event.target.value || "";
        renderList();
      });
      document.getElementById("review-bar").addEventListener("click", function (event) {
        const btn = event.target.closest("[data-rating]");
        if (!btn || !document.getElementById("review-bar").contains(btn)) return;
        saveReview(btn.getAttribute("data-rating")).catch(function (err) {
          showNotice(err.message || "Could not save the review.");
        });
      });

      async function boot() {
        const cfgRes = await api("/api/config");
        const callsRes = await api("/api/calls", { cache: "no-store" });
        if (!cfgRes || !callsRes) return;
        await cfgRes.json();
        calls = asCalls(await callsRes.json());
        updatedEl.textContent = "Updated " + new Date().toLocaleTimeString(undefined, {
          hour: "2-digit",
          minute: "2-digit"
        });
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
