# Status slide deck — filesystem record

Filesystem copy of the "ROCKE-3D → JAX: Status" slide deck (4 slides, last
synced 2026-09-22), kept alongside `STATUS.md` in this directory.

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
