# Architecture diagrams

Static HTML diagram for engineering reviews. **Not** the normative contract — see [`../spec/TECH-SPEC.md`](../spec/TECH-SPEC.md) and [`../pitch/slides-why-shishu.md`](../pitch/slides-why-shishu.md) for current V3 architecture and messaging.

| Artifact | Status | Use |
|----------|--------|-----|
| [`turn-evidence-architecture.html`](turn-evidence-architecture.html) | **V3 engineering** — evidence → interpretation → policy → reducer. Slice 1 (`turns.py`, `policy.py`, atomic ledger in `db.py`) is implemented on `v3`; `TurnController` & single structured reader remain deferred. | Internal reviews, onboarding engineers |

Open HTML files in a browser locally (`file://` or `python -m http.server` from repo root).
