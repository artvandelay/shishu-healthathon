# SHISHU documentation map

Read this index before spelunking `docs/`. Paths below are relative to the repo root.

## Who should read what

| Audience | Start here | Then |
|----------|------------|------|
| **Pitch / stakeholders** | [`pitch/slides-why-shishu.md`](pitch/slides-why-shishu.md) (Marp → PDF in same folder) | [`product/PRD-user-flows.md`](product/PRD-user-flows.md) for flows |
| **Engineers shipping behavior** | [`spec/TECH-SPEC.md`](spec/TECH-SPEC.md) (normative; `R-` / `S-` / `N-` ↔ tests) | [`design/v3-design-and-thinking-vad.md`](design/v3-design-and-thinking-vad.md), [`design/v3-moldability-boundary.md`](design/v3-moldability-boundary.md) |
| **Pack authors** | [`design/v3-moldability-boundary.md`](design/v3-moldability-boundary.md) | `packs/<pack_id>/` + validator errors from `intake/pack.py` |
| **Release / regression review** | [`reports/`](reports/) | Staging health + commit pin on Railway |
| **Diagrams / deep architecture** | [`architecture/README.md`](architecture/README.md) | Turn & evidence lifecycle diagram |

## Layout

```
docs/
  README.md           ← you are here
  product/            PRD, consent copy source
  spec/               TECH-SPEC (contract with code)
  design/             V3 design of record + moldability boundary
  pitch/              Why-SHISHU slide deck (md + pdf)
  architecture/       Turn & evidence engineering diagram (HTML)
  reports/            Dated review memos (e.g. v3 vs main)
  samples/            Example patient PDFs (no PHI)
  i18n/               Translator exports (`v3_branch_pack.csv`)
```

**Normative stack:** `spec/TECH-SPEC.md` wins over design notes, slides, and HTML diagrams when they disagree. Refresh measured LOC / test counts in TECH-SPEC when you change the suite; design docs may lag until someone edits them.

**Consent:** [`product/SHISHU-consent.md`](product/SHISHU-consent.md) is the wording source; shipped PDFs live under each pack's `assets/`.

## Related (outside `docs/`)

- Runbook: [`../README.md`](../README.md)
- Code: `intake/`, `packs/`, `tests/`
- Private notes: `jd-raw-words.md` (gitignored), root `/reports/` (gitignored local artifacts)