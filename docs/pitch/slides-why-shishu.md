---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    font-size: 19px;
    padding: 38px 50px;
    background: #ffffff;
    color: #0f172a;
  }
  h1 { font-size: 1.52em; margin: 0 0 6px 0; color: #0f172a; letter-spacing: -0.02em; }
  h2 { font-size: 0.92em; font-weight: 500; color: #475569; margin: 0 0 18px 0; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .grid2x2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 15px 18px;
  }
  .card.blue { background: #f0f9ff; border-color: #bae6fd; }
  .card.green { background: #f0fdf4; border-color: #bbf7d0; }
  .card h3 { margin: 0 0 8px 0; font-size: 0.95em; color: #0f172a; }
  .card ul { margin: 0; padding-left: 18px; }
  .card li { margin-bottom: 7px; line-height: 1.33; font-size: 0.86em; }
  .card li:last-child { margin-bottom: 0; }
  .tag {
    display: inline-block;
    font-size: 0.68em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #0369a1;
    margin-bottom: 4px;
  }
  .takeaway {
    margin-top: 16px;
    padding: 11px 16px;
    background: #0f172a;
    color: #f8fafc;
    border-radius: 10px;
    font-size: 0.82em;
    text-align: center;
  }
  .states {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 10px;
  }
  .state-pill {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 6px 8px;
    font-size: 0.75em;
    line-height: 1.25;
  }
  .state-pill strong { display: block; color: #0f172a; }
---

# SHISHU: Structured Hindi Voice Intake
## Browser voice intake that prepares a clear patient history ahead of a clinic or ASHA/ANM visit

<div class="cols">
<div class="card blue">
<div class="tag">What It Is</div>
<h3>Guided Hindi Voice + Tap Intake</h3>
<ul>
  <li><strong>Browser-based:</strong> Runs directly in a phone browser after sign-in and Hindi consent (<em>"मैं सहमत हूँ"</em>)</li>
  <li><strong>Voice + tap hybrid:</strong> Warm female Hindi voice guides a fixed history; callers speak freely or tap <strong>नहीं / हाँ</strong></li>
  <li><strong>Two artefacts per call:</strong> Produces a <strong>2-page Hindi Patient PDF</strong> (takeaways + 3–6 doctor questions) and a <strong>1-page English Doctor Note</strong></li>
  <li><strong>Configurable per clinic:</strong> Ships with maternal/child screening (62 Qs) and OPD fever triage (14 Qs) on one engine</li>
</ul>
</div>

<div class="card green">
<div class="tag">Why It Matters</div>
<h3>Structured History Ahead of the Visit</h3>
<ul>
  <li><strong>Prepares for the consult:</strong> Organises scattered oral stories into grouped findings, top red-flag alerts, and exact verbatim Hindi quotes</li>
  <li><strong>Everyday Hindi:</strong> Uses familiar terms (<em>काँच का बक्सा, नीली लाइट, नाल, झटके</em>) and directs families toward doctor consultation and emergency care</li>
  <li><strong>Deterministic reporting:</strong> Patient and doctor documents render directly from locked call answers via fixed templates</li>
</ul>

<div class="states">
  <div class="state-pill"><strong>Concern / Answer</strong>Captured + verbatim</div>
  <div class="state-pill"><strong>Clear / Skipped</strong>Denied or rule-skipped</div>
  <div class="state-pill"><strong>Don't Know / Unanswered</strong>Flagged explicitly</div>
</div>
</div>
</div>

<div class="takeaway">
<strong>Core design:</strong> Conversational Hindi voice handles delivery while a deterministic form engine holds clinical state, branching rules, and final reports.
</div>

---

# UX &amp; Product Details
## Built for noisy clinics, speakerphone audio, and unstable mobile data connections

<div class="grid2x2">
<div class="card">
<div class="tag">1 · Conversation &amp; Family Context</div>
<ul>
  <li><strong>Multi-fact capture in one turn:</strong> Extra symptoms shared early populate matching future questions and skip redundant follow-ups</li>
  <li><strong>Twin &amp; multi-child support:</strong> Expands <em>"दो जुड़वाँ, 2 साल के"</em> into distinct children, asks <em>which child</em> oldest-first (by name/order for twins), and updates ages on corrections</li>
  <li><strong>Clarify &amp; revise anytime:</strong> Plain 1–2 sentence explanations preserve retry budgets, and earlier answers update while keeping prior spoken concerns intact</li>
</ul>
</div>

<div class="card">
<div class="tag">2 · Audio &amp; Quick-Tap Controls</div>
<ul>
  <li><strong>Noise adaptation &amp; speech buffer:</strong> Tracks ambient fan levels and keeps a ~3&nbsp;s pre-speech audio ring so soft Hindi openings remain intact</li>
  <li><strong>Timed button reveal:</strong> <strong>नहीं / हाँ</strong> and choice pills unlock once ≥50% of a question plays, enabling immediate tap-over-voice progression</li>
  <li><strong>Tap vs speech check:</strong> When a tap and spoken answer conflict, SHISHU holds the step and asks one confirmation; spoken emergency/stop applies immediately</li>
</ul>
</div>

<div class="card">
<div class="tag">3 · Drops, Silence &amp; Safety Routing</div>
<ul>
  <li><strong>Call resumption:</strong> Network drops show <em>"लाइन जुड़ रही है…"</em>, attempt 3 auto-reconnects (90&nbsp;s window) to the active question, and append the same audio file</li>
  <li><strong>Check-in &amp; Nudge:</strong> Prompts gently at 15&nbsp;s of silence and closes cleanly at 30&nbsp;s; after 3&nbsp;s of silence, a <strong>Nudge</strong> button evaluates any buffered speech</li>
  <li><strong>Emergency &amp; partial endings:</strong> Danger signs switch directly to nearest hospital / <strong>108</strong> guidance; early endings mark the report partial and invite calling back</li>
</ul>
</div>

<div class="card">
<div class="tag">4 · Delivery, Clinician Review &amp; QA</div>
<ul>
  <li><strong>Token-scoped Patient PDF:</strong> Shows <em>"रिपोर्ट तैयार हो रही है…"</em>, then one-tap download (with retry and <em>"नई कॉल"</em> reset); multi-topic asides stay clinician-only</li>
  <li><strong>Clinician cockpit (<code>/admin</code>):</strong> Pack-filtered calls, red-flag banners, dual-channel audio playback (caller vs agent), priority-sorted answers, and event trace</li>
  <li><strong>In-call bug capture:</strong> Bug modal saves category, optional note, screenshot, and UI state alongside the active call</li>
</ul>
</div>
</div>

---

# How It Works: Swappable Pack + Deterministic Core
## Plug a clinic's configuration folder into the top layer — questions, rules, and templates stay modular

<svg viewBox="0 0 1160 486" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; display:block; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif;">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284c7"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed"/>
    </marker>
  </defs>

  <!-- TOP PLUG-IN: Swappable Intake Pack -->
  <rect x="276" y="4" width="876" height="92" rx="14" fill="#f5f3ff" stroke="#7c3aed" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="298" y="34" font-size="17" font-weight="700" fill="#4c1d95">Swappable Intake Pack</text>
  <text x="298" y="56" font-size="13.5" fill="#6d28d9">1 folder (questions, branching rules, Hindi prompts, PDF layout)</text>
  <rect x="298" y="64" width="260" height="24" rx="6" fill="#ede9fe" stroke="#c4b5fd"/>
  <text x="428" y="80" text-anchor="middle" font-size="12.5" font-weight="600" fill="#4c1d95">Maternal &amp; Child · 62 Qs</text>
  <rect x="574" y="64" width="260" height="24" rx="6" fill="#ede9fe" stroke="#c4b5fd"/>
  <text x="704" y="80" text-anchor="middle" font-size="12.5" font-weight="600" fill="#4c1d95">OPD Fever · 14 Qs</text>
  <rect x="850" y="64" width="280" height="24" rx="6" fill="#ffffff" stroke="#7c3aed" stroke-dasharray="4 3"/>
  <text x="990" y="80" text-anchor="middle" font-size="12.5" font-weight="600" fill="#6d28d9">+ Add next clinic pack</text>

  <!-- Plug-in arrows down -->
  <line x1="411" y1="96" x2="411" y2="136" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arrowPurple)"/>
  <text x="419" y="121" font-size="11.5" font-weight="600" fill="#6d28d9">Scoring rules</text>

  <line x1="746" y1="96" x2="746" y2="136" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arrowPurple)"/>
  <text x="754" y="121" font-size="11.5" font-weight="600" fill="#6d28d9">Questions &amp; Hindi script</text>

  <line x1="1049" y1="96" x2="1049" y2="136" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arrowPurple)"/>
  <text x="1057" y="121" font-size="11.5" font-weight="600" fill="#6d28d9">Templates</text>

  <!-- 1. Mother's Phone -->
  <rect x="8" y="138" width="210" height="264" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
  <text x="113" y="168" text-anchor="middle" font-size="17" font-weight="700" fill="#0f172a">Mother's Phone</text>
  <text x="113" y="189" text-anchor="middle" font-size="12.5" fill="#64748b">Mobile browser · WebSocket</text>

  <rect x="22" y="206" width="182" height="52" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="113" y="229" text-anchor="middle" font-size="13.5" font-weight="600" fill="#0f172a">Hindi Voice (Mic)</text>
  <text x="113" y="247" text-anchor="middle" font-size="12" fill="#64748b">Active during listening</text>

  <rect x="22" y="268" width="182" height="52" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="113" y="291" text-anchor="middle" font-size="13.5" font-weight="600" fill="#0f172a">Quick Taps (नहीं / हाँ)</text>
  <text x="113" y="309" text-anchor="middle" font-size="12" fill="#64748b">Unlocks after ≥50% audio</text>

  <rect x="22" y="330" width="182" height="52" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="113" y="353" text-anchor="middle" font-size="13.5" font-weight="600" fill="#0f172a">Drop Recovery</text>
  <text x="113" y="371" text-anchor="middle" font-size="12" fill="#64748b">Resumes active step</text>

  <!-- 2. Turn Understanding (Top Center) -->
  <rect x="276" y="138" width="270" height="126" rx="14" fill="#fefce8" stroke="#fde047" stroke-width="2"/>
  <text x="411" y="166" text-anchor="middle" font-size="16.5" font-weight="700" fill="#713f12">Turn Understanding</text>
  <text x="411" y="190" text-anchor="middle" font-size="13" fill="#854d0e">Single pass on raw 16 kHz audio</text>
  <text x="411" y="211" text-anchor="middle" font-size="13" fill="#854d0e">+ parallel child/profile extractor</text>
  <rect x="294" y="222" width="234" height="30" rx="7" fill="#ffffff" stroke="#facc15"/>
  <text x="411" y="242" text-anchor="middle" font-size="12.5" font-weight="600" fill="#713f12">Strict safety &amp; retry checks</text>

  <!-- 3. Voice AI (Bottom Center) -->
  <rect x="276" y="286" width="270" height="116" rx="14" fill="#eff6ff" stroke="#93c5fd" stroke-width="2"/>
  <text x="411" y="319" text-anchor="middle" font-size="16.5" font-weight="700" fill="#1e3a8a">Voice AI (Gemini Live)</text>
  <text x="411" y="344" text-anchor="middle" font-size="13.5" fill="#1e40af">Speaks &amp; clarifies in warm Hindi</text>
  <text x="411" y="368" text-anchor="middle" font-size="13" font-weight="600" fill="#0369a1">Stateless turn delivery</text>

  <!-- 4. Deterministic Engine -->
  <rect x="606" y="138" width="280" height="264" rx="14" fill="#f0fdf4" stroke="#86efac" stroke-width="2"/>
  <text x="746" y="168" text-anchor="middle" font-size="17" font-weight="700" fill="#14532d">Deterministic Engine</text>
  <text x="746" y="189" text-anchor="middle" font-size="12.5" fill="#166534">Shared core · rule-driven state</text>

  <rect x="624" y="206" width="244" height="52" rx="8" fill="#ffffff" stroke="#bbf7d0"/>
  <text x="746" y="229" text-anchor="middle" font-size="13.5" font-weight="700" fill="#14532d">Skip &amp; Roster Rules</text>
  <text x="746" y="247" text-anchor="middle" font-size="12" fill="#166534">Auto-fills multi-fact answers</text>

  <rect x="624" y="268" width="244" height="52" rx="8" fill="#ffffff" stroke="#bbf7d0"/>
  <text x="746" y="291" text-anchor="middle" font-size="13.5" font-weight="700" fill="#14532d">Safety &amp; Conflict Check</text>
  <text x="746" y="309" text-anchor="middle" font-size="12" fill="#166534">Tap vs speech · Retry caps</text>

  <rect x="624" y="330" width="244" height="54" rx="8" fill="#dcfce7" stroke="#4ade80"/>
  <text x="746" y="353" text-anchor="middle" font-size="13.5" font-weight="700" fill="#14532d">Saved State + Database</text>
  <text x="746" y="371" text-anchor="middle" font-size="12" fill="#166534">Commits before next prompt</text>

  <!-- 5. Outputs -->
  <rect x="946" y="138" width="206" height="264" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
  <text x="1049" y="168" text-anchor="middle" font-size="17" font-weight="700" fill="#0f172a">Outputs</text>
  <text x="1049" y="189" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0369a1">Rendered from saved state</text>

  <rect x="962" y="206" width="174" height="52" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="1049" y="229" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">Hindi Patient PDF</text>
  <text x="1049" y="247" text-anchor="middle" font-size="12" fill="#64748b">2 pages + doctor Qs</text>

  <rect x="962" y="268" width="174" height="52" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="1049" y="291" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">Doctor Note (EN)</text>
  <text x="1049" y="309" text-anchor="middle" font-size="12" fill="#64748b">Flags + verbatim Hindi</text>

  <rect x="962" y="330" width="174" height="52" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="1049" y="353" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">Audit &amp; Stereo WAV</text>
  <text x="1049" y="371" text-anchor="middle" font-size="12" fill="#64748b">Full decision trace</text>

  <!-- Runtime loop arrows (zero crossings) -->
  <line x1="218" y1="201" x2="274" y2="201" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>
  <text x="246" y="190" text-anchor="middle" font-size="11.5" font-weight="600" fill="#475569">Turn</text>

  <line x1="546" y1="201" x2="604" y2="201" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>
  <text x="575" y="190" text-anchor="middle" font-size="11.5" font-weight="600" fill="#475569">Claims</text>

  <line x1="606" y1="344" x2="548" y2="344" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <text x="577" y="333" text-anchor="middle" font-size="11.5" font-weight="600" fill="#0284c7">Prompt</text>

  <line x1="276" y1="344" x2="220" y2="344" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <text x="248" y="333" text-anchor="middle" font-size="11.5" font-weight="600" fill="#0284c7">Hindi</text>

  <line x1="886" y1="270" x2="944" y2="270" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>
  <text x="915" y="259" text-anchor="middle" font-size="11.5" font-weight="600" fill="#475569">Locked</text>

  <!-- Bottom Principle Strip -->
  <rect x="8" y="422" width="1144" height="56" rx="12" fill="#0f172a"/>
  <text x="198" y="455" text-anchor="middle" font-size="15.5" font-weight="600" fill="#f8fafc">1. Voice AI stays stateless</text>
  <line x1="395" y1="434" x2="395" y2="466" stroke="#334155" stroke-width="1.5"/>
  <text x="580" y="455" text-anchor="middle" font-size="15.5" font-weight="600" fill="#f8fafc">2. Six explicit answer states</text>
  <line x1="765" y1="434" x2="765" y2="466" stroke="#334155" stroke-width="1.5"/>
  <text x="960" y="455" text-anchor="middle" font-size="15.5" font-weight="600" fill="#f8fafc">3. Reports render strictly from saved state</text>
</svg>
