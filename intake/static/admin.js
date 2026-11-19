(function () {
  "use strict";
  // SHISHU clinician dashboard: pack filter, call list, one call's full record, bug reports.
  const $ = (id) => document.getElementById(id);
  const S = { calls: [], bugs: [], packs: [], labels: {}, pack: "", sel: null, cur: null, seq: 0 };
  const el = (t, c, x) => { const n = document.createElement(t); if (c) n.className = c; if (x != null) n.textContent = x; return n; }, set = (id, html) => { $(id).innerHTML = html; };
  const notice = (m) => { $("notice").textContent = m || ""; $("notice").hidden = !m; }, stamp = (id) => { $(id).textContent = "Updated " + new Date().toLocaleTimeString(); };
  const esc = (v) => String(v == null ? "" : v).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  const inline = (v) => esc(v).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");
  const txt = (v) => (v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v));
  const parse = (v) => { if (typeof v !== "string") return v; try { return JSON.parse(v); } catch (e) { return v; } };
  const asList = (v) => { const p = parse(v); return Array.isArray(p) ? p : []; }, asObj = (v) => { const p = parse(v); return p && typeof p === "object" && !Array.isArray(p) ? p : {}; };
  const unpack = (data, key) => (Array.isArray(data) ? data : data[key] || []);
  const dur = (s) => { const n = Math.max(0, Number(s) || 0), p = (x) => String(x).padStart(2, "0"); return p(n / 60 | 0) + ":" + p(n % 60 | 0); };
  const num = (v, s) => (v == null || v === "" ? "—" : Number.isNaN(Number(v)) ? String(v) : String(Math.round(Number(v) * 10) / 10) + (s || ""));
  const model = (c) => { const m = String((c && c.model_used) || "").trim(); return !m ? "—" : /gemini-3\.1-flash-live/i.test(m) ? "Gemini 3.1 Flash Live" : m; };
  const when = (v) => { if (!v) return ""; const raw = String(v).trim(), d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(raw) ? raw : raw.replace(" ", "T") + "Z"); return Number.isNaN(d.getTime()) ? raw : d.toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }); };
  // A table from header labels plus cell arrays; `cls` tints a whole row (e.g. a red-flagged concern).
  const grid = (labels, rows) => "<table><thead><tr>" + labels.map((l) => "<th>" + l + "</th>").join("") + "</tr></thead><tbody>" + rows.map((r) => "<tr" + (r.cls ? ' class="' + r.cls + '"' : "") + ">" + r.cells.map((c) => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</tbody></table>";
  const kpi = (label, value, extra) => '<div class="kpi"><span>' + esc(label) + "</span><strong>" + esc(value) + "</strong>" + (extra || "") + "</div>";
  // Every 401 anywhere sends the clinician back through the login card with a return path.
  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ credentials: "same-origin", headers: { Accept: "application/json" } }, opts || {}));
    if (res.status === 401) { location.href = "/login?next=" + encodeURIComponent(location.pathname + location.search); return null; }
    if (res.ok) return res;
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.message || "Request failed (" + res.status + ")");
  }
  // Copies text, falling back to a hidden textarea when the clipboard API is blocked.
  async function copy(text, ok) {
    try { await navigator.clipboard.writeText(text); } catch (e) { const a = el("textarea"); a.value = text; document.body.appendChild(a); a.select(); document.execCommand("copy"); a.remove(); }
    notice(ok);
  }
  // Markdown subset: headings, bullets, rules, paragraphs; runs of items collapse into one <ul>. Escaped first.
  function md(src) {
    const html = String(src || "").split(/\r?\n/).map((raw) => {
      const t = raw.trim(), h = /^(#{1,3})\s+(.*)$/.exec(t), li = /^[-*]\s+(.*)$/.exec(t);
      if (!t) return ""; if (t === "---") return "<hr>";
      return h ? "<h" + h[1].length + ">" + inline(h[2]) + "</h" + h[1].length + ">" : li ? "<li>" + inline(li[1]) + "</li>" : "<p>" + inline(t) + "</p>";
    }).join("").replace(/(?:<li>.*?<\/li>)+/g, (list) => "<ul>" + list + "</ul>");
    return html || "<p>The report is not ready yet.</p>";
  }
  // Packs come from /api/pack, widened by whatever pack_id the loaded calls report.
  function fillPacks() {
    const sel = $("pack-filter"), ids = [], keep = S.pack;
    S.packs.concat(S.calls.map((c) => c.pack_id)).forEach((p) => { const id = String((p && p.id) || p || ""); if (id && ids.indexOf(id) === -1) ids.push(id); });
    sel.textContent = "";
    const all = el("option", "", "All packs"); all.value = ""; sel.appendChild(all);
    ids.sort().forEach((id) => { const o = el("option", "", S.labels[id] ? S.labels[id] + " · " + id : id); o.value = id; sel.appendChild(o); });
    sel.value = S.pack = ids.indexOf(keep) !== -1 ? keep : "";
  }
  function renderList() {
    const rows = S.calls.filter((c) => !S.pack || String(c.pack_id || "") === S.pack), count = "Calls (" + rows.length + ")";
    $("tab-calls").textContent = count;
    if (!rows.length) return set("call-list", '<p class="empty">' + (S.calls.length ? "No calls for this pack." : "No calls yet.") + "</p>");
    set("call-list", rows.map((c) => {
      const n = asList(c.red_flags).length, meta = [model(c)].concat([c.pack_id, c.clinic_code, dur(c.duration_sec), when(c.started_at)].filter(Boolean));
      const head = '<div class="row"><div class="row-left">Call #' + esc(c.id) + (c.status === "active" ? '<span class="live-dot">Live</span>' : "") + '</div><span class="' + (n ? "pill hot" : "pill") + '">' + (n === 1 ? "1 flag" : n + " flags") + "</span></div>";
      const body = '<div class="caller">' + esc(c.caller_name || "Login name not recorded") + '</div><div class="people">' + esc(c.roster_summary || c.patient_name || "Patient not named") + "</div>";
      const tags = meta.map((m, i) => '<span class="' + (i ? "" : "badge") + '">' + esc(m) + "</span>").join("");
      return '<button type="button" class="call-item' + (c.id === S.sel ? " selected" : "") + '" data-id="' + esc(c.id) + '">' + head + body + '<div class="meta">' + tags + "</div></button>";
    }).join(""));
  }
  // An answer reads as: who it was about, the chosen label, then the verbatim note and follow-up notes.
  function answerText(r) {
    const bits = [], note = asObj(r.note), fut = asObj(r.followups), said = note.verbatim || note.patient_safe || "", value = r.choice_label || txt(r.value);
    if (asList(r.item_ids).length) bits.push(asList(r.item_ids).join(", "));
    if (value) bits.push(value); if (said && said !== value) bits.push(said);
    Object.keys(fut).sort().forEach((k) => { const nt = asObj(asObj(fut[k]).note), str = nt.verbatim || nt.patient_safe; if (str) bits.push(k + ": " + str); });
    return bits.join(" · ");
  }
  // Concerns first, then unknown/answered/clear — the order a clinician reads a chart in.
  function paintAnswers(answers) {
    const rank = { concern: 0, unknown: 1, unanswered: 2, answered: 3, clear: 4 };
    const keys = Object.keys(answers).filter((q) => String(asObj(answers[q]).state || "") !== "not_applicable");
    keys.sort((a, b) => (rank[asObj(answers[a]).state] ?? 5) - (rank[asObj(answers[b]).state] ?? 5) || a.localeCompare(b));
    if (!keys.length) return set("answers", '<p class="calm">No structured findings were recorded.</p>');
    set("answers", grid(["Question ID", "State", "Value / Note"], keys.map((q) => ({ cls: asObj(answers[q]).state === "concern" ? "flag" : "", cells: [esc(q), esc(asObj(answers[q]).state), esc(answerText(asObj(answers[q])))] }))));
  }
  // Event kinds drive badge tone: errors and bug reports red, watchdog nudges amber, the rest neutral.
  function paintEvents(events) {
    $("event-summary").textContent = "Under-the-Hood Event Trace (" + events.length + ")";
    if (!events.length) return set("event-table", '<p class="calm">No events were recorded.</p>');
    set("event-table", grid(["Time", "Type", "Detail"], events.map((evt) => {
      const kind = String(evt.type || evt.event || "event"), meta = asObj(evt.meta);
      const badge = /error|bug/i.test(kind) ? "pill hot" : /nudge/i.test(kind) ? "pill warn" : "badge";
      const kv = Object.keys(meta).filter((k) => meta[k] != null && meta[k] !== "" && typeof meta[k] !== "object").map((k) => k + "=" + meta[k]);
      return { cells: [esc(evt.t == null ? "—" : evt.t), '<span class="' + badge + '">' + esc(kind) + "</span>", esc([txt(evt.detail), kv.join(" ")].filter(Boolean).join(" · "))] };
    })));
  }
  function paintKpis(call, t) {
    const lats = asList(t.turn_latencies_ms), pills = lats.length ? '<div class="latency-pills">' + lats.map((ms) => '<span class="pill">' + esc(num(ms, " ms")) + "</span>").join("") + "</div>" : "";
    const rows = [["Model / Voice", (call.model_used || "—") + " · " + (call.voice_used || "default")], ["Turns (Patient / Bot)", num(t.user_turns) + " / " + num(t.bot_turns)]];
    rows.push(["Avg Latency", lats.length ? num(t.avg_latency_ms, " ms") : "—", pills], ["Connect", num(t.connect_ms, " ms")], ["Tool Calls / Nudges", num(t.tool_calls) + " / " + num(t.watchdog_nudges)]);
    rows.push(["Audio Chunks In / Out", num(t.audio_chunks_in) + " / " + num(t.audio_chunks_out)], ["Max Latency", t.max_latency_ms == null ? "—" : num(t.max_latency_ms, " ms")]);
    if (t.dropoff_qid) rows.push(["Stopped At", String(t.dropoff_qid)]);
    set("kpis", rows.map((r) => kpi(r[0], r[1], r[2])).join(""));
  }
  function renderDetail(call) {
    const url = "/api/calls/" + encodeURIComponent(call.id), audio = Boolean(String(call.audio_path || "").trim()), flags = asList(call.red_flags), turns = asList(call.transcript);
    S.cur = call; S.sel = call.id; $("empty").hidden = true; $("report").hidden = false;
    $("call-title").textContent = "Call #" + call.id + (call.patient_name ? " · " + call.patient_name : "");
    $("call-meta").textContent = [call.pack_id ? "pack " + call.pack_id : "", call.roster_summary || "", "duration " + dur(call.duration_sec), "model " + model(call), call.clinic_code || "", call.status || ""].filter(Boolean).join(" · ");
    $("player").hidden = $("download").hidden = !audio; $("audio-note").hidden = audio;
    if (audio) { $("player").src = "/api/audio/" + call.id; $("download").href = "/api/audio/" + call.id; $("download").download = "call_" + call.id + ".wav"; }
    else { $("player").removeAttribute("src"); $("download").removeAttribute("href"); }
    $("patient-pdf").href = url + "/patient.pdf"; $("patient-pdf").classList.toggle("disabled", call.status === "active");
    $("flag-banner").hidden = !flags.length; $("no-flags").hidden = Boolean(flags.length);
    set("flag-list", flags.map((item) => "<li>" + esc(txt(item)) + "</li>").join(""));
    $("doctor").innerHTML = md(call.summary_doctor);
    $("patient").textContent = call.summary_patient || "The patient summary is not ready yet.";
    set("transcript", turns.map((turn) => {
      const role = turn.role || turn.speaker || "", who = role === "assistant" ? "SHISHU" : role === "user" ? "Patient" : role || "Note";
      return '<div class="turn' + (role === "assistant" ? " shishu" : "") + '"><strong>' + esc(who) + "</strong><p>" + esc(turn.text || "") + "</p></div>";
    }).join("") || '<p class="calm">No transcript was recorded.</p>');
    paintAnswers(asObj(asObj(call.cursor_state).answers));
    paintEvents(asList(call.events));
    paintKpis(call, asObj(call.telemetry));
    renderList();
  }
  // A sequence number stops a slow detail response from overwriting a newer click.
  async function loadDetail(id) {
    const seq = ++S.seq;
    S.sel = id; renderList();
    const res = await api("/api/calls/" + encodeURIComponent(id));
    if (res && seq === S.seq) renderDetail(await res.json());
  }
  function showTab(name) {
    const calls = name !== "bugs";
    $("tab-calls").setAttribute("aria-selected", calls ? "true" : "false"); $("tab-bugs").setAttribute("aria-selected", calls ? "false" : "true");
    $("calls-pane").hidden = !calls; $("bugs-pane").hidden = calls;
    if (!calls) loadBugs().catch((e) => notice(e.message));
  }
  // Bug cards carry their own Copy button; the screenshot previews only when it is a data: URL.
  function renderBugs() {
    $("tab-bugs").textContent = "Bug Reports (" + S.bugs.length + ")";
    if (!S.bugs.length) return set("bug-list", '<p class="empty">No bug reports yet.</p>');
    set("bug-list", S.bugs.map((bug, i) => {
      const cid = Number(bug.call_id) || 0, sec = Math.max(0, Math.round(Number(bug.call_elapsed_sec) || 0));
      const shot = asObj(asObj(bug.snapshot).client).screenshot, img = typeof shot === "string" && shot.indexOf("data:image/") === 0;
      const link = cid ? '<button type="button" class="bug-link" data-call="' + esc(cid) + '">Call #' + esc(cid) + "</button>" : '<span class="meta">No call</span>';
      const meta = esc([dur(sec) + " (" + sec + "s)", when(bug.created_at)].filter(Boolean).join(" · "));
      return '<article class="bug-item"><div class="row"><span class="badge">' + esc(bug.category || "Uncategorized") + "</span>" + link + '</div><p class="meta">' + meta +
        '</p><p class="bug-note">' + esc(bug.note || "No note.") + "</p>" + (img ? '<img class="bug-shot-preview" src="' + esc(shot) + '" alt="Bug screenshot">' : "") +
        '<button type="button" class="ghost" data-bug="' + i + '">Copy Bug JSON</button></article>';
    }).join(""));
  }
  // One loader: fetch, unwrap the list, hand it to a renderer, stamp the aside.
  async function pull(path, key, render, stampId) {
    const res = await api(path, { cache: "no-store" });
    if (res) { render(unpack(await res.json(), key)); stamp(stampId); }
  }
  const loadBugs = () => pull("/api/bugs", "bugs", (list) => { S.bugs = list; renderBugs(); }, "bugs-updated"), refresh = () => pull("/api/calls", "calls", (list) => { S.calls = list; fillPacks(); renderList(); }, "calls-updated");
  // Public pack metadata: brand the dashboard and learn every pack id for the filter.  // Public pack metadata: brand the dashboard and learn every pack id for the filter.
  async function hydratePack() {
    const res = await fetch("/api/pack", { headers: { Accept: "application/json" } });
    if (!res.ok) return;
    const pack = await res.json(), url = asObj(pack.assets_url), title = pack.title || "";
    S.packs = asList(pack.available_packs);
    S.packs.forEach((p) => { if (p && p.id) S.labels[p.id] = p.title || p.id; });
    if (url.logo) $("logo").src = url.logo;
    if (title) { $("brand-title").textContent = title; document.title = title + " — Clinical Dashboard"; }
    if (pack.subtitle_native) $("brand-sub").textContent = "Clinical dashboard · " + pack.subtitle_native;
  }
  const printing = () => { document.body.classList.add("printing"); window.addEventListener("afterprint", () => document.body.classList.remove("printing"), { once: true }); window.print(); };
  const actions = {
    refresh: refresh, "refresh-bugs": loadBugs, "print-report": printing,
    "copy-report": () => copy((S.cur && S.cur.summary_doctor) || "", "Report copied."),
    "copy-debug": () => !S.cur ? Promise.resolve() : copy(JSON.stringify({ call: S.cur.id, pack_id: S.cur.pack_id, cursor_state: asObj(S.cur.cursor_state), telemetry: asObj(S.cur.telemetry), events: asList(S.cur.events), transcript: asList(S.cur.transcript) }, null, 2), "Debug bundle copied."),
    logout: async () => { try { await fetch("/api/logout", { method: "POST", credentials: "same-origin" }); } catch (e) { /* redirect anyway */ } location.href = "/login"; }
  };
  Object.keys(actions).forEach((id) => $(id).addEventListener("click", () => Promise.resolve(actions[id]()).catch((e) => notice(e.message))));
  $("tab-calls").addEventListener("click", () => showTab("calls")); $("tab-bugs").addEventListener("click", () => showTab("bugs"));
  $("pack-filter").addEventListener("change", (e) => { S.pack = e.target.value || ""; renderList(); });
  // Delegated clicks: both lists re-render wholesale, so bind once on their hosts.
  $("call-list").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-id]");
    if (btn) loadDetail(Number(btn.getAttribute("data-id"))).catch((err) => notice(err.message));
  });
  $("bug-list").addEventListener("click", (e) => {
    const call = e.target.closest("[data-call]"), bug = e.target.closest("[data-bug]");
    if (call) { showTab("calls"); $("pack-filter").value = S.pack = ""; loadDetail(Number(call.getAttribute("data-call"))).catch((err) => notice(err.message)); }
    if (bug) copy(JSON.stringify(S.bugs[Number(bug.getAttribute("data-bug"))], null, 2), "Bug JSON copied.").catch((err) => notice(err.message));
  });  (async function boot() {
    await hydratePack().catch(() => {}); await refresh();
    const wanted = Number(new URLSearchParams(location.search).get("call")), pick = S.calls.find((c) => c.id === wanted) || S.calls[0];
    if (pick) await loadDetail(pick.id);
    loadBugs().catch(() => {}); setInterval(() => { refresh().catch(() => {}); }, 15000);
  })().catch((e) => notice(e.message || "Could not load the dashboard."));
})();