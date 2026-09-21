"""
Build a PDF rendition of the ROCKE-3D -> JAX status slide deck using
reportlab directly (weasyprint is unusable here: this system's libpango
1.42 predates a Pango function every available weasyprint release calls,
and there's no network path to a newer libpango or a headless browser).

Not a pixel-identical export of the live Claude Artifact -- a faithful,
hand-built recreation of the same content/structure/numbers, using
reportlab's built-in Helvetica/Courier fonts (no Google Fonts network
dependency) in the same navy/green/orange color scheme.
"""
from reportlab.lib.pagesizes import landscape
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

PAGE = landscape((7.5 * inch, 13.333 * inch))  # 540 x 960 pt, 16:9
W, H = PAGE

NAVY = HexColor("#0F172A")
CARD = HexColor("#1E293B")
CARD2 = HexColor("#14212C")
GREEN = HexColor("#4ADE80")
ORANGE = HexColor("#F97316")
BLUE = HexColor("#38BDF8")
INK = HexColor("#F8FAFC")
INK2 = HexColor("#E2E8F0")
MUTED = HexColor("#94A3B8")
FOOTER = HexColor("#64748B")

MARGIN = 36

def para(text, size=11, color=INK2, bold=False, leading=None, font="Helvetica"):
    fname = font + ("-Bold" if bold else "")
    style = ParagraphStyle(
        "s", fontName=fname, fontSize=size, textColor=color,
        leading=leading or size * 1.3, alignment=TA_LEFT,
    )
    return Paragraph(text, style)

def draw_para(c, text, x, y, width, height, **kw):
    p = para(text, **kw)
    w, h = p.wrap(width, height)
    p.drawOn(c, x, y - h)
    return h

def rounded(c, x, y, w, h, r, fill, stroke=None):
    c.saveState()
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(1)
    c.roundRect(x, y, w, h, r, stroke=1 if stroke else 0, fill=1)
    c.restoreState()

def left_bar_card(c, x, y, w, h, bar_color, r=8):
    rounded(c, x, y, w, h, r, CARD2)
    c.saveState()
    c.setFillColor(bar_color)
    c.rect(x, y, 4, h, stroke=0, fill=1)
    c.restoreState()

def footer(c, text):
    draw_para(c, text, MARGIN, 22, W - 2 * MARGIN, 20, size=8, color=FOOTER)

def eyebrow_title(c, eyebrow, title, subtitle=None, title_size=26):
    y = H - MARGIN - 14
    draw_para(c, eyebrow, MARGIN, y, W - 2 * MARGIN, 20,
              size=10, color=GREEN, bold=True, font="Courier")
    y -= 26
    draw_para(c, title, MARGIN, y, W - 2 * MARGIN, 60,
              size=title_size, color=INK, bold=True, leading=title_size * 1.1)
    y -= (title_size * 1.15 + 10)
    if subtitle:
        h = draw_para(c, subtitle, MARGIN, y, W - 2 * MARGIN, 60, size=11, color=MUTED)
        y -= (h + 10)
    return y


def slide_bottom_line(c):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    y = eyebrow_title(
        c,
        "ROCKE-3D &rarr; JAX &middot; Status as of 2026-09-21",
        "Where Things Stand",
        "Use an AI coding agent to autonomously convert ROCKE-3D (NASA GISS's Fortran GCM) to "
        "Python/JAX, and measure accuracy, CPU/GPU performance, conversion cost, and gotchas "
        "&mdash; validated against real production restart data (P2SAoM40).",
        title_size=30,
    )

    cards = [
        (GREEN, "14 / 17", "Modules faithfully ported",
         "Validated directly against real Fortran. 3 remain placeholders &mdash; see slide 3."),
        (GREEN, "1e-9&ndash;1e-3", "Accuracy vs. real Fortran",
         "PBL + DRYCNV, same inputs, CPU and GPU &mdash; floating-point-level agreement."),
        (GREEN, "2.3&ndash;6.3&times;", "GPU speedup vs. real Fortran",
         "Measured, not estimated &mdash; first real GPU run, 2026-09-20 (kernel-level)."),
        (ORANGE, "Mixed on CPU", "JAX is not a blanket CPU win",
         "Fortran beats JAX-CPU on DRYCNV. The case for JAX here is GPU, not CPU."),
    ]
    card_w = (W - 2 * MARGIN - 3 * 10) / 4
    card_h = 150
    card_y = y - card_h
    x = MARGIN
    for bar, num, label, note in cards:
        left_bar_card(c, x, card_y, card_w, card_h, bar)
        draw_para(c, num, x + 14, card_y + card_h - 14, card_w - 24, 40,
                  size=20, color=INK, bold=True, font="Courier")
        draw_para(c, label, x + 14, card_y + card_h - 46, card_w - 24, 30,
                  size=10.5, color=INK2, bold=True)
        draw_para(c, note, x + 14, card_y + card_h - 66, card_w - 24, 60, size=8.5, color=MUTED)
        x += card_w + 10

    band_y = card_y - 20 - 46
    rounded(c, MARGIN, band_y, W - 2 * MARGIN, 46, 8, CARD2)
    draw_para(c, "The comparison that matters:", MARGIN + 16, band_y + 32, 150, 20,
              size=9.5, color=MUTED, bold=True)
    draw_para(c,
              "Every number here is <b><font color='#4ADE80'>JAX vs. the real production Fortran "
              "model</font></b> &mdash; not JAX vs. a second Python reimplementation. NumPy-relative "
              "comparisons exist elsewhere in this project's history but are dropped from this "
              "summary as answering the wrong question.",
              MARGIN + 170, band_y + 34, W - 2 * MARGIN - 190, 40, size=9.5, color=INK2)

    footer(c, "Full detail: STATUS.md (this repo, projects/imvi/rocke3d_jax) &middot; "
              "historical record kept as-is in FINDINGS.md / PORTING_STATUS.md / EXECUTIVE_SUMMARY.md")


def bar_row(c, x, y, label, bar_w, value_text, bar_color, label_w=90, max_bar=220, row_h=20):
    draw_para(c, label, x, y + 12, label_w, 16, size=9, color=MUTED)
    c.saveState()
    c.setFillColor(bar_color)
    c.roundRect(x + label_w, y, max(bar_w, 3), 14, 3, stroke=0, fill=1)
    c.restoreState()
    draw_para(c, value_text, x + label_w + max_bar + 8, y + 12, 260, 16, size=9, color=INK2, font="Courier")


def slide_performance(c):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    y = eyebrow_title(c, "PERFORMANCE", "Real Fortran vs. JAX &mdash; CPU and GPU", title_size=26)

    col_w = (W - 2 * MARGIN - 20) / 2
    left_x = MARGIN
    right_x = MARGIN + col_w + 20
    panel_top = y
    panel_h = 330

    rounded(c, left_x, panel_top - panel_h, col_w, panel_h, 12, CARD)
    px, py = left_x + 18, panel_top - 28
    draw_para(c, "Kernel-level, measured (mean of 100 calls)", px, py, col_w - 36, 20,
              size=12, color=INK, bold=True)
    py -= 30
    draw_para(c, "DRYCNV", px, py, col_w - 36, 16, size=11, color=INK, bold=True)
    py -= 30
    max_bar = 220
    bar_row(c, px, py, "Fortran-CPU", 220 * (1.27 / 17.1), "1.27 ms", BLUE); py -= 24
    bar_row(c, px, py, "JAX-CPU", 220 * (17.1 / 17.1), "17.1 ms — 13.4x slower", ORANGE); py -= 24
    bar_row(c, px, py, "JAX-GPU", max(220 * (0.56 / 17.1), 3), "0.56 ms — 2.3x faster than Fortran", GREEN); py -= 40

    draw_para(c, "PBL similarity", px, py, col_w - 36, 16, size=11, color=INK, bold=True)
    py -= 30
    bar_row(c, px, py, "Fortran-CPU", 220 * (0.37 / 0.37), "0.37 ms", BLUE); py -= 24
    bar_row(c, px, py, "JAX-CPU", 220 * (0.20 / 0.37), "0.20 ms — 1.9x faster", ORANGE); py -= 24
    bar_row(c, px, py, "JAX-GPU", 220 * (0.06 / 0.37), "0.06 ms — 6.3x faster than Fortran", GREEN); py -= 30

    draw_para(c, "Same shared inputs on every leg. Grid: P2SAoM40's real 72x46x40.",
              px, py, col_w - 36, 20, size=8.5, color=FOOTER)

    top2_h = 160
    rounded(c, right_x, panel_top - top2_h, col_w, top2_h, 12, CARD)
    rx, ry = right_x + 18, panel_top - 26
    draw_para(c, "Full physics chain (CPU only)", rx, ry, col_w - 36, 18, size=12, color=INK, bold=True)
    ry -= 22
    ry -= draw_para(c, "PBL + radiation + surface + ground, driven by P2SAoM40's real restart "
              "state, chained in the real per-timestep order.", rx, ry, col_w - 36, 40, size=9, color=MUTED)
    ry -= 6
    draw_para(c, "~5.5x faster", rx, ry, col_w - 36, 26, size=22, color=GREEN, bold=True, font="Courier")
    ry -= 30
    ry -= draw_para(c, "~48 ms/step (JAX) vs. ~264 ms/step (real Fortran). Radiation excluded "
              "(simplified graybody stand-in, not real spectral transfer).", rx, ry, col_w - 36, 40, size=8.5, color=MUTED)
    draw_para(c, "GPU: not yet measured for this full chain.", rx, ry - 4, col_w - 36, 16,
              size=9.5, color=ORANGE, bold=True)

    bot2_h = panel_h - top2_h - 20
    rounded(c, right_x, panel_top - panel_h, col_w, bot2_h, 12, CARD)
    rx, ry = right_x + 18, panel_top - top2_h - 20 - 22
    draw_para(c, "Reading this honestly", rx, ry, col_w - 36, 18, size=12, color=INK, bold=True)
    ry -= 26
    ry -= draw_para(c, "JAX-CPU is <b>not</b> a blanket win &mdash; it loses DRYCNV outright to "
              "Fortran on CPU (small-array dispatch overhead dominates at this grid size).",
              rx, ry, col_w - 36, 50, size=9, color=INK2)
    ry -= 4
    draw_para(c, "GPU is where JAX wins decisively on both kernels tested. That's the real case "
              "for this port.", rx, ry, col_w - 36, 40, size=9, color=GREEN, bold=True)

    footer(c, "Source: compare_fortran.f90 / compare_jax.py / compare_run_gpu_interactive.py "
              "(kernel-level) &middot; p2saom40_driver.py (full chain) &middot; STATUS.md")


def slide_next_steps(c):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    y = eyebrow_title(c, "WHAT'S NEXT", "Recommendation &amp; Open Items", title_size=26)

    col_w = (W - 2 * MARGIN - 20) / 2
    left_x = MARGIN
    right_x = MARGIN + col_w + 20
    panel_top = y
    panel_h = 330

    rounded(c, left_x, panel_top - panel_h, col_w, panel_h, 12, CARD)
    px, py = left_x + 18, panel_top - 26
    draw_para(c, "The remaining 3 modules &mdash; port or defer?", px, py, col_w - 36, 20,
              size=12, color=INK, bold=True)
    py -= 30

    items = [
        (GREEN, "SEAICE (core thermodynamics) — port next",
         "Physically active in every P2SAoM40-class run, not optional. Dominates polar surface "
         "energy balance — exactly where the current 0.987 global correlation check is least "
         "able to see a local error."),
        (GREEN, "ATURB (PBL-top-finding) — port next",
         "Feeds surface-flux accuracy globally, not just at the poles. Also tests whether the "
         "vmap-over-grid-cells approach holds up on harder physics."),
        (ORANGE, "LAKES (lkmix) — defer",
         "Small fraction of Earth's surface — lower accuracy payoff for the effort. Revisit "
         "only if a lake-focused science need requires it."),
    ]
    for bar, title, note in items:
        ih = 88
        left_bar_card(c, px, py - ih, col_w - 36, ih, bar, r=6)
        draw_para(c, title, px + 14, py - 18, col_w - 64, 20, size=10.5, color=INK, bold=True)
        draw_para(c, note, px + 14, py - 38, col_w - 64, 50, size=8.5, color=MUTED)
        py -= ih + 12

    top2_h = 110
    rounded(c, right_x, panel_top - top2_h, col_w, top2_h, 12, CARD)
    rx, ry = right_x + 18, panel_top - 26
    draw_para(c, "Reporting cleanup", rx, ry, col_w - 36, 18, size=12, color=INK, bold=True)
    ry -= 24
    draw_para(c, "Drop NumPy-relative speedups from headline reporting. The target has always "
              "been <b><font color='#4ADE80'>JAX vs. real Fortran</font></b> — NumPy was a "
              "control group for a narrower question, and the PBL one "
              "(<font face='Courier'>simil_numpy</font>) was also found to be incomplete.",
              rx, ry, col_w - 36, 70, size=9, color=INK2)

    bot2_h = panel_h - top2_h - 20
    rounded(c, right_x, panel_top - panel_h, col_w, bot2_h, 12, CARD)
    rx, ry = right_x + 18, panel_top - top2_h - 20 - 24
    draw_para(c, "Open items", rx, ry, col_w - 36, 18, size=12, color=INK, bold=True)
    ry -= 24
    opens = [
        "Measure the full physics-chain GPU run (radiation+surface+ground together) — only "
        "the DRYCNV/PBL kernel subset has real GPU numbers today.",
        "Fix the wind-speed convention bug in the shared FLUXES/SURFACE module files themselves "
        "— currently only worked around locally.",
        "Port SEAICE + ATURB, then re-run the 0.987 correlation check.",
    ]
    for i, item in enumerate(opens, 1):
        draw_para(c, f"<b><font color='#4ADE80'>{i}</font></b>  {item}", rx, ry, col_w - 36, 44, size=9, color=INK2)
        ry -= 42

    footer(c, "Full detail and reasoning: STATUS.md (this repo, projects/imvi/rocke3d_jax)")


def main(out_path):
    c = canvas.Canvas(out_path, pagesize=PAGE)
    for slide_fn in (slide_bottom_line, slide_performance, slide_next_steps):
        slide_fn(c)
        c.showPage()
    c.save()
    print("wrote", out_path)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "deck.pdf")
