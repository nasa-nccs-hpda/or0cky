# Status slide deck: filesystem record

**Current deck (refreshed 2026-10-08):** `status_deck.pdf` (19 slides, 16:9): chronology of the whole project and
its findings, as of HEAD `ec36779` and ledger entry D198 (there is no D199). Owner: G. Tamkin. Drafted by a
Claude Code agent on 2026-10-08 from the repository documents only; every slide footer names its sources
(`FULL_FIDELITY_DELTAS.md` ledger ids, `Reports/*.md`, `git log`). Companion text table:
`../Reports/PROJECT_CHRONOLOGY.md` (generated from the same data). Review by: when the day criterion status
changes, a GPU run exists, or the owner records a relaxation (ACCEPTANCE_CRITERIA section 9).

Slides: 1 at a glance; 2-3 chronology figures; 4 phases; 5 categories A/B/C/D and C1/C2; 6 gate verdicts per
date; 7 what is JAX and what is not; 8-9 step timings and time shares; 10 the 54-step day (NOT MET); 11
diagnoses D193-D197; 12 noise floor; 13 math-library hazards; 14 GPU; 15 tests; 16 what remains; 17 decisions
taken and pending; 18 where documents disagree; 19 appendix (Track A history, sources, rebuild).

**Rebuild** (no network needed; core 11 and scratch outside the repository are the owner's convention):
```
cd status_slides
# 1. figures (matplotlib; the conda env python)
/home/gtamkin/.conda/envs/graphcast-env/bin/python make_figures.py images
# 2. PDF (reportlab is installed for the system python3 only, not in the conda env)
python3 build_pdf.py status_deck.pdf [--png PREVIEW_DIR]
# 3. text chronology
python3 build_chronology_md.py ../Reports/PROJECT_CHRONOLOGY.md
```
`build_pdf.py` exits non-zero and prints `OVERFLOW:` lines if any text block does not fit its box;
`--png` writes one PNG per slide (PIL, Liberation Sans / Nimbus Mono, metric-compatible with the PDF's
Helvetica / Courier) so legibility can be checked without a PDF renderer (none is installed here).
Facts live in `chronology_data.py` (dates, commits, events) and in the slide functions of `build_pdf.py`; the
figure numbers of `make_figures.py` are quoted from D191, D195 and the regression records cited on the slides.

**Files**
- `build_pdf.py`, `deck_layout.py`: deck content and layout layer (reportlab, same navy/green/orange style as before).
- `make_figures.py`, `images/*.png`: matplotlib figures (chronology, zoom, timings, day score, tests). `images/diff_grid.png`, `images/render_diff.py`: the 2026-09 difference-map grid (unchanged).
- `chronology_data.py`, `build_chronology_md.py`: single source of the chronology.
- `status_deck_2026-10-01.pdf`, `build_pdf_2026-10-01.py`: the previous PDF and its builder, kept for history (Track A era figures: 14 of 17 modules, GPU kernels 2.3-6.3x, stop-the-port recommendation; superseded by the full-fidelity direction). `build_pdf_2026-10-01.py` writes to the path you give it: do not point it at `status_deck.pdf`.
- `deck.json`, `slides/*.html`: the 2026-09-24 live Claude Artifact content (https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA). NOT refreshed by this work and stale relative to the current status.

**Limits**: not a pixel-identical export of any live artifact; numbers are quoted from the ledger and were not re-measured; where two documents disagree the newer ledger entry is used and slide 18 says so; one start state (nov26) for every day-level result.

---
Earlier README (kept for history, 2026-09-24):

# Status slide deck — filesystem record 

Filesystem copy of the "ROCKE-3D → JAX: Status" slide deck (8 slides, last
synced 2026-09-24: HQ-audience refresh with the five-stage Fortran→JAX progression, Phase 1 vs Phase 2 profile, and the stop-the-port recommendation), kept alongside `STATUS.md` in this directory.

**Live, viewable/presentable version**: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA

**What's here**:
- `status_deck.pdf` — a real, standalone-viewable PDF of all 4 slides. Open
  this if you just want to look at or share the deck from the filesystem.
- `deck.json` (the slide order/outline) and `slides/*.html` — the raw
  content of each slide, in the closed HTML/CSS subset the Claude Artifact
  "Slides" type uses. Each slide file is a bare `<section id="...">`
  fragment (no `<html>`/`<head>`/`<body>`, no linked stylesheet) — **opening
  one directly in a browser will not render it correctly**. Kept for
  diffing/reusing the text and numbers.
- `images/diff_grid.png` — the JAX-minus-Fortran difference-map grid shown
  on slide 4 (drycnv.T, pbl.u, pbl.dpsih, pbl.dpsiq; CPU and GPU rows).
  `images/render_diff.py` regenerates it directly from `compare_data/`
  (matplotlib, not Plotly/kaleido — kaleido needs a headless-Chromium
  download this system has no network path to). Same underlying values as
  `outputs/p2saom40_kernel_*_diff_*.html` from
  `visualize_p2saom40_kernel_maps.ipynb`; verified matching max\|diff\| on
  generation.
- `build_pdf.py` — generates `status_deck.pdf` from the content above using
  `reportlab`, not a browser/CSS renderer. **Why**: this system's `libpango`
  (1.42) is too old for any available `weasyprint` release (all of them call
  a Pango function added around 1.48+), and there's no network path here to
  a newer libpango or a headless browser (Playwright/Chromium aren't
  installed, no route to install them). `build_pdf.py` hand-recreates each
  slide's layout with reportlab's built-in Helvetica/Courier fonts (no
  Google Fonts dependency) in the same navy/green/orange color scheme, and
  embeds `images/diff_grid.png` directly on slide 4 — it's a faithful
  content recreation, not a pixel-identical export of the live Artifact.
  Re-run with `python3 build_pdf.py status_deck.pdf` if the content changes;
  it has the slide text hardcoded, so edits go in that script (and ideally
  in `slides/*.html`/the live deck too, to keep them in sync).

**Live, editable version**: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA
— if it's edited there later, nothing in this directory updates
automatically. Everything here is a snapshot as of 2026-09-22.

**STALE: `status_deck.pdf` / `build_pdf.py` still render the 2026-09-22
4-slide deck (they predate the 2026-09-24 refresh).** `slides/*.html` and
`deck.json` here match the live deck; regenerate the PDF (edit `build_pdf.py`
to the new content) before sharing it from the filesystem.
