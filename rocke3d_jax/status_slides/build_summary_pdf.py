"""
Build summary_deck.pdf: a concise summary of the whole ROCKE-3D -> JAX
exercise (approach, five-stage results, lessons), written as the starting
point for repeating the exercise with a full-fidelity port in a separate branch.

reportlab only (same reasons as build_pdf.py: no browser/weasyprint here).
Every card is measured before drawing; a text overflow raises instead of
silently clipping, since there is no renderer available to eyeball the output.

Numbers: STATUS.md ("Full Physics Chain", "Round 2 optimization", "Accuracy",
"What's simplified"). Stage 1 (~60 ms) was timed on a different GPU-less node.

Usage: python3 build_summary_pdf.py [out.pdf]   (default: summary_deck.pdf)
"""
import math
import sys

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

PAGE = landscape((7.5 * inch, 13.333 * inch))
W, H = PAGE
M = 40

NAVY = HexColor("#0F172A")
CARD = HexColor("#1E293B")
CARD2 = HexColor("#14212C")
GREEN = HexColor("#4ADE80")
LGREEN = HexColor("#86EFAC")
ORANGE = HexColor("#F97316")
AMBER = HexColor("#FBBF24")
BLUE = HexColor("#38BDF8")
INK = HexColor("#F8FAFC")
INK2 = HexColor("#E2E8F0")
MUTED = HexColor("#94A3B8")
FOOT = HexColor("#64748B")

# (label, plain name, ms/step, colour, speed-up vs Fortran, note)
STAGES = [
    ("Original Fortran", 264.0, BLUE),
    ("1  JAX on CPU", 59.56, ORANGE),
    ("2  JAX on GPU", 32.96, AMBER),
    ("3  Phase 1 optimization", 4.26, LGREEN),
    ("4  Phase 2 optimization", 0.74, GREEN),
]
FORTRAN_MS = 264.0


def style(size, color=INK2, bold=False, font="Helvetica", leading=None):
    return ParagraphStyle("s", fontName=font + ("-Bold" if bold else ""), fontSize=size,
                          textColor=color, leading=leading or size * 1.28, alignment=TA_LEFT)


def text(c, s, x, ytop, width, size=13, color=INK2, bold=False, font="Helvetica", leading=None, maxh=None):
    p = Paragraph(s, style(size, color, bold, font, leading))
    _, h = p.wrap(width, 10000)
    if maxh is not None and h > maxh + 0.5:
        raise ValueError(f"text overflow ({h:.0f} > {maxh:.0f}pt): {s[:60]!r}")
    p.drawOn(c, x, ytop - h)
    return h


def measure(s, width, size=13, bold=False, font="Helvetica", leading=None):
    p = Paragraph(s, style(size, bold=bold, font=font, leading=leading))
    return p.wrap(width, 10000)[1]


def box(c, x, ytop, w, h, fill=CARD, bar=None):
    c.saveState()
    c.setFillColor(fill)
    c.roundRect(x, ytop - h, w, h, 8, stroke=0, fill=1)
    if bar is not None:
        c.setFillColor(bar)
        c.rect(x, ytop - h, 4, h, stroke=0, fill=1)
    c.restoreState()


def page(c, n, eyebrow, title, foot=None):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    y = H - M
    text(c, eyebrow.upper(), M, y, W - 2 * M, size=11, color=GREEN, bold=True, font="Courier")
    y -= 20
    h = text(c, title, M, y, W - 2 * M, size=27, color=INK, bold=True, leading=30)
    text(c, foot or "", M, 46, W - 2 * M - 60, size=9.5, color=FOOT, maxh=26)
    text(c, str(n), W - M - 20, 46, 20, size=9.5, color=FOOT)
    return y - h - 16


def card(c, x, ytop, w, h, head, body, bar=GREEN, head_size=14, body_size=12, head_color=INK):
    box(c, x, ytop, w, h, CARD2, bar)
    pad = 14
    yy = ytop - 12
    if head:
        yy -= text(c, head, x + pad + 4, yy, w - 2 * pad - 4, size=head_size, color=head_color, bold=True) + 5
    if body:
        text(c, body, x + pad + 4, yy, w - 2 * pad - 4, size=body_size, color=MUTED,
             maxh=(yy - (ytop - h)) - 10)


# --------------------------------------------------------------------------- slides
def s_title(c):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    text(c, "ROCKE-3D ON PYTHON/JAX", M, H - 90, W - 2 * M, size=13, color=GREEN, bold=True, font="Courier")
    text(c, "Approach and Results of a Fortran-to-JAX Port", M, H - 115, W - 2 * M, size=40, color=INK,
         bold=True, leading=44)
    text(c, "What an AI coding agent could do with part of NASA GISS's ROCKE-3D climate model: translate it, "
            "check it against the real model, and make it fast. Written as the starting point for repeating "
            "the exercise with a full-fidelity port in a separate branch.",
         M, H - 205, 820, size=16, color=MUTED, leading=22, maxh=95)
    stats = [("356×", "faster than Fortran on a GPU, for the surface-physics slice that was ported", GREEN),
             ("5", "stages measured on the same real P2SAoM40 restart data", BLUE),
             ("0.965", "spatial correlation with real model temperatures; kernels match Fortran to rounding", GREEN),
             ("~9%", "of Fortran's runtime is covered by that slice: scope is the caveat", AMBER)]
    bw = (W - 2 * M - 3 * 16) / 4
    for i, (big, lab, col) in enumerate(stats):
        x = M + i * (bw + 16)
        box(c, x, 215, bw, 150, CARD, col)
        text(c, big, x + 18, 200, bw - 28, size=34, color=col, bold=True, font="Courier")
        text(c, lab, x + 18, 158, bw - 30, size=12, color=INK2, maxh=85)
    text(c, "2026-09-28 · Source of truth: projects/imvi/rocke3d_jax/STATUS.md", M, 60, W - 2 * M, size=11, color=FOOT)


def s_approach(c):
    y = page(c, 2, "Approach", "How the Exercise Was Run",
             "Detail: STATUS.md “The goal”, “Conversion cost & gotchas”, README_GPU.md")
    steps = [
        ("1  Fix the reference", "Real Fortran ROCKE-3D, configuration P2SAoM40 (72×46 grid, 40 layers), a real restart file, and per-routine costs read from the real run log."),
        ("2  Port module by module", "An AI agent translated the column-independent physics (PBL, dry convection, surface, radiation stand-in) to Python/JAX."),
        ("3  Validate against Fortran", "Compiled Fortran test drivers run on identical inputs; JAX and Fortran outputs compared (CPU and GPU)."),
        ("4  Chain them into a driver", "PBL + surface + ground + simplified radiation, in the real per-step call order, started from the real restart."),
        ("5  Benchmark five stages", "Fortran, JAX/CPU, JAX/GPU, then two optimization rounds, all timed on the same step of the same data."),
        ("6  Optimize by profiling", "Profile, remove the biggest cost, re-check accuracy, re-measure. Repeat. Verify any dramatic result before reporting it."),
    ]
    cw = (W - 2 * M - 2 * 16) / 3
    ch = (y - 70) / 2 - 8
    for i, (h, b) in enumerate(steps):
        r, k = divmod(i, 3)
        card(c, M + k * (cw + 16), y - r * (ch + 16), cw, ch, h, b, GREEN if i < 4 else AMBER, 15, 13)


def s_results(c):
    y = page(c, 3, "Results", "Five Stages, from Fortran to 356× Faster",
             "Log-scale bars (shorter = faster); one 30-minute model step, real P2SAoM40 data. Stage 1 timed on a different GPU-less node; stages 3–4 on an A100 node.")
    top = y - 40
    label_w, bar_x = 175, M + 175 + 8
    maxbar = 380
    lo, hi = math.log10(0.74), math.log10(264.0)
    rowh = 46
    for i, (lab, ms, col) in enumerate(STAGES):
        yy = top - i * rowh
        text(c, lab, M, yy - 4, label_w, size=13, color=INK, bold=True)
        w = 30 + (maxbar - 30) * (math.log10(ms) - lo) / (hi - lo)
        c.setFillColor(col)
        c.roundRect(bar_x, yy - 26, w, 24, 4, stroke=0, fill=1)
        val = f"{ms:.2f} ms" if ms < 1 else f"{ms:.1f} ms"
        text(c, val, bar_x + w + 8, yy - 9, 100, size=12.5, color=INK, bold=True, font="Courier")
    # table on right
    tx = bar_x + maxbar + 120
    tw = W - M - tx
    text(c, "Speed-up vs. Fortran", tx, y - 4, 88, size=10.5, color=MUTED, bold=True)
    text(c, "vs. previous stage", tx + 96, y - 4, 88, size=10.5, color=MUTED, bold=True)
    prev = [None, None, 59.56, 32.96, 4.26]
    for i, (lab, ms, col) in enumerate(STAGES):
        yy = top - i * rowh - 2
        vf = FORTRAN_MS / ms
        text(c, "1×" if i == 0 else f"{vf:,.0f}×" if vf >= 10 else f"{vf:.1f}×", tx, yy - 2, 90, size=15, color=col, bold=True, font="Courier")
        if prev[i]:
            text(c, f"{prev[i] / ms:.1f}×", tx + 96, yy - 3, 80, size=13, color=INK2, font="Courier")
    # bottom callouts
    cy = top - 5 * rowh - 6
    ch = cy - 58
    cw = (W - 2 * M - 2 * 16) / 3
    card(c, M, cy, cw, ch, "CPU-only tells the same story",
         "Phase 1 → Phase 2 on CPU: 17.9 ms → 1.9 ms (9.5×). CPU stage 1 (≈60 ms) is an older, cross-node measurement.", BLUE, 13, 11.5)
    card(c, M + cw + 16, cy, cw, ch, "Two numbers for Phase 2",
         "0.74 ms only when the state stays on the GPU across steps. One step per call: 3.6 ms (1.2× over Phase 1).", AMBER, 13, 11.5)
    card(c, M + 2 * (cw + 16), cy, cw, ch, "Why 8× is not 356×",
         "The big wins came from software structure (job count, data movement). The first GPU port alone gained under 2× (cross-node).", GREEN, 13, 11.5)


def s_stages(c):
    y = page(c, 4, "The five stages", "What Each Stage Is, in Plain Terms",
             "Technical detail: STATUS.md “Full Physics Chain”, “GPU optimization: Phase 1”, “Round 2 optimization”")
    desc = [
        ("Original Fortran", "264 ms", BLUE, "The real NASA model code, run on an ordinary processor. It is the reference for both speed and answers."),
        ("1  JAX on CPU", "≈60 ms", ORANGE, "An AI agent rewrote the physics in Python/JAX, a language built for fast array math. Same processor, first draft, not tuned."),
        ("2  JAX on GPU", "33 ms", AMBER, "The same rewrite on a graphics processor made for thousands of parallel calculations. It gains only modestly because the work is split into many tiny GPU jobs."),
        ("3  Phase 1", "4.3 ms", LGREEN, "About 33 tiny GPU jobs per step were merged into one, so the GPU spends its time computing instead of starting jobs."),
        ("4  Phase 2", "0.74 ms", GREEN, "The model's data now stays on the GPU across many steps instead of being copied back and forth each step, constants are computed once, and the slowest routines were rewritten."),
    ]
    rh = (y - 62) / 5 - 8
    for i, (n, t, col, d) in enumerate(desc):
        yy = y - i * (rh + 8)
        box(c, M, yy, W - 2 * M, rh, CARD2, col)
        text(c, n, M + 20, yy - 12, 230, size=15, color=INK, bold=True)
        text(c, t, M + 20, yy - 34, 230, size=17, color=col, bold=True, font="Courier")
        text(c, d, M + 270, yy - 14, W - 2 * M - 290, size=14, color=INK2, leading=19, maxh=rh - 18)


def s_profile(c):
    y = page(c, 5, "Profiling", "Where the Time Went",
             "CPU, per step. Right panel: same busy shared node, back to back, so absolute times are ~2× the A100-node figures; proportions are the point.")
    pw = (W - 2 * M - 24) / 2
    # left panel: pre-fusion breakdown
    box(c, M, y, pw, y - 60, CARD)
    text(c, "Before Phase 1: one step ≈ 55 ms", M + 16, y - 14, pw - 32, size=14, color=INK, bold=True)
    pre = [("Python glue between calls", 18.00, GREEN), ("Dry convection (1 call)", 13.68, BLUE),
           ("Surface-layer similarity (24 calls)", 13.35, BLUE), ("Pressure prep (Exner)", 4.56, BLUE),
           ("Wind fluxes (not JIT-compiled)", 2.58, ORANGE), ("Pressure profile (NumPy loop)", 2.34, ORANGE),
           ("Everything else", 0.95, BLUE)]
    for i, (lab, ms, col) in enumerate(pre):
        yy = y - 52 - i * 40
        text(c, lab, M + 16, yy + 2, 190, size=11, color=MUTED)
        bw = 150 * ms / 18.0
        c.setFillColor(col)
        c.roundRect(M + 214, yy - 16, max(bw, 3), 16, 3, stroke=0, fill=1)
        text(c, f"{ms:.1f} ms · {100 * ms / 55.46:.0f}%", M + 214 + bw + 8, yy - 3, 100, size=10.5, color=INK2, font="Courier")
    text(c, "Every function was compiled but called separately from Python: 33 GPU/CPU jobs per step plus glue. Phase 1 fused these into one.",
         M + 16, y - 52 - 7 * 40 + 6, pw - 32, size=11.5, color=INK2, maxh=(y - 52 - 7 * 40 + 6) - 64)
    # right panel: stacked
    rx = M + pw + 24
    box(c, rx, y, pw, y - 60, CARD)
    text(c, "Phase 1 vs Phase 2 (stacked)", rx + 16, y - 14, pw - 32, size=14, color=INK, bold=True)
    k = (pw - 64) / 34.0
    bars = [("Phase 1", [(12.3, BLUE), (14.7, GREEN), (7.0, ORANGE)], "34.0 ms"),
            ("Phase 2, one call per step", [(1.06, BLUE), (3.05, GREEN), (2.82, ORANGE)], "6.9 ms · 4.9×"),
            ("Phase 2, chained on device", [(1.06, BLUE), (3.05, GREEN)], "4.1 ms · 8.3×")]
    for i, (lab, segs, val) in enumerate(bars):
        yy = y - 56 - i * 54
        text(c, lab, rx + 16, yy + 12, pw - 32, size=11.5, color=INK, bold=True)
        xx = rx + 16
        for ms, col in segs:
            c.setFillColor(col)
            c.rect(xx, yy - 30, max(ms * k, 2), 22, stroke=0, fill=1)
            xx += ms * k
        text(c, val, xx + 8, yy - 15, 110, size=10.5, color=INK2, font="Courier")
    ly = y - 56 - 3 * 54 - 4
    for i, (lab, col) in enumerate([("Dry convection", BLUE), ("All other computing", GREEN), ("Copying data to/from device", ORANGE)]):
        c.setFillColor(col)
        c.rect(rx + 16, ly - 14 - i * 20, 10, 10, stroke=0, fill=1)
        text(c, lab, rx + 32, ly - 11 - i * 20, 300, size=11, color=MUTED)
    text(c, "Dry convection 12.3 → 1.1 ms (11×) · other computing 14.7 → 3.1 ms (4.8×) · data copying 7.0 → 2.8 ms, now 41% of a single call, which is why keeping data on the device matters.",
         rx + 16, ly - 78, pw - 32, size=11.5, color=INK2, maxh=(ly - 78) - 64)


def s_accuracy(c):
    y = page(c, 6, "Accuracy", "Did the Answers Stay the Same?",
             "Detail: STATUS.md “Accuracy”, “Round 2 optimization”; tests in rocke3d_jax/tests (115 pass)")
    cw = (W - 2 * M - 16) / 2
    ch = (y - 62) / 2 - 8
    items = [
        ("JAX vs. real Fortran (kernels)", "PBL and dry convection, same inputs, CPU and GPU: agreement at floating-point level (1e-9 to 1e-3 across fields).", GREEN),
        ("Full step vs. real model data", "Layer-1 air temperature vs. the real run's period-mean surface temperature: correlation 0.965, bias −1.9 °C. Unchanged by both optimizations.", GREEN),
        ("Optimized vs. original JAX", "Dry convection vs. an independent float64 reference: < 5e-4 K. Chained steps differ from Phase 1 at rounding level; a few marginal cells flip a branch in an iterative solver, exactly as they do when Phase 1 itself is nudged by one rounding step, so this is noise, not regression.", AMBER),
        ("How this was checked", "Independent float64 references over every stability branch; deliberately breaking the code to confirm the tests fail (mutation testing); a noise-floor test; comparison to real Fortran outputs. Two measurement bugs were found and fixed on the way.", GREEN),
    ]
    for i, (h, b, col) in enumerate(items):
        r, k = divmod(i, 2)
        card(c, M + k * (cw + 16), y - r * (ch + 16), cw, ch, h, b, col, 15, 13)


def s_scope(c):
    y = page(c, 7, "Scope", "What the Numbers Do Not Say",
             "Detail: STATUS.md “What's simplified in the full-chain driver”, “A validation-methodology gotcha”")
    cw = (W - 2 * M - 16) / 2
    ch = y - 62
    card(c, M, y, cw, ch, "Coverage of the real model (Fortran run log)",
         "<b>Ported and timed:</b> surface and ground physics, about 9% of Fortran's runtime (SURFACE 8.9%).<br/><br/>"
         "<b>Radiation, 65.6% of runtime:</b> a simplified stand-in here, excluded from the comparison.<br/><br/>"
         "<b>Moist convection (9.9%) and atmospheric dynamics (8.5%):</b> not ported. They need values from neighbouring grid cells, which does not vectorize as easily.<br/><br/>"
         "So <b>no whole-model speed-up is claimed</b>.", AMBER, 15, 13.5)
    card(c, M + cw + 16, y, cw, ch, "The ported slice does less physics than Fortran",
         "One dominant surface type per cell instead of area-weighted sub-tiling; simplified turbulence; land hydrology (GHY), sea ice and lakes are placeholder code.<br/><br/>"
         "An earlier claim of “14 of 17 modules faithfully ported” was wrong: the GHY test compared a stub with itself and hardcoded outputs to zero.<br/><br/>"
         "A full-fidelity port would do more work and would likely show a smaller multiple. Whether a real speed-up survives is the question a full-fidelity branch answers.", ORANGE, 15, 13.5)


def s_lessons(c):
    y = page(c, 8, "Repeating it", "Lessons for the Full-Fidelity Port",
             "Full-fidelity sizes (Fortran lines): GHY 4,939 · SEAICE 5,394 · ATURB 1,405. Effort for these, or for radiation/dynamics/moist convection, has not been estimated.")
    cw = (W - 2 * M - 16) / 2
    ch = y - 62
    card(c, M, y, cw, ch, "Do",
         "• <b>Write the pass criteria first</b>: per module, which Fortran routine and what tolerance.<br/>"
         "• <b>Derive every test driver from the real Fortran source</b>; never a hand-written stub.<br/>"
         "• <b>Audit for “placeholder / simplified” before calling a module done.</b><br/>"
         "• <b>Time all stages on the same node, commit and method</b>; record the Fortran and first-JAX baselines first.<br/>"
         "• <b>Force CPU in its own process</b>; synchronize before reading clocks; report single-call and chained.<br/>"
         "• <b>Design for the GPU from day one</b>: layer-first arrays, state resident on the device, constants computed once, one compiled job per step.<br/>"
         "• <b>Profile before optimizing</b>; verify any dramatic result (reproduce, sanity-check, state scope).", GREEN, 16, 13)
    card(c, M + cw + 16, y, cw, ch, "Avoid (each one happened here)",
         "• <b>Self-referential tests:</b> a stub “reference” shared the port's simplifications and passed (GHY, SEAICE, LAKES, ATURB).<br/>"
         "• <b>Incomplete reference code:</b> a NumPy stand-in omitted branches and showed diffs up to 1.26e4 until rebuilt from the real formulas.<br/>"
         "• <b>A “CPU” baseline that silently ran on the GPU</b>, producing a false “no GPU benefit”.<br/>"
         "• <b>Mixing nodes:</b> the ≈60 ms JAX/CPU figure came from another machine.<br/>"
         "• <b>Array-layout footguns:</b> Fortran vs C ordering corrupted shared test inputs until an input round-trip check caught it.<br/>"
         "• <b>Claiming whole-model results from a slice:</b> radiation alone is 66% of Fortran's time.<br/>"
         "• Unit tests pass on synthetic values; only real restart data exposed a wind-speed bug that zeroed land fluxes.", ORANGE, 16, 13)


def s_refs(c):
    y = page(c, 9, "Reference", "Where Everything Lives",
             "All under projects/imvi/rocke3d_jax/ (a project-local record; nothing here has been promoted to shared knowledge/).")
    rows = [
        ("STATUS.md", "Single source of truth: goal, accuracy, five-stage results, Round 2 optimization, simplifications, recommendation, gotchas."),
        ("README_GPU.md", "How to run the benchmarks on the GPU cluster and which driver path to use."),
        ("RESTART_SESSION.md", "Session-to-session hand-off notes and history."),
        ("p2saom40_driver.py · p2saom40_compare.py", "Full-chain driver (single-call and chained device-resident) and the five-stage benchmark harness."),
        ("drycnv.py · pbl.py", "The two kernels validated against Fortran and optimized in Round 2."),
        ("compare_fortran.f90 · compare_jax.py", "Kernel-level Fortran-vs-JAX comparison; data in compare_data/."),
        ("tests/ · tests/test_round2_optimization*.py", "115 regression tests including float64 references."),
        ("status_slides/", "Status deck sources (deck.json, slides/*.html) and this deck's build script."),
    ]
    rh = (y - 62) / len(rows) - 6
    for i, (a, b) in enumerate(rows):
        yy = y - i * (rh + 6)
        box(c, M, yy, W - 2 * M, rh, CARD2, BLUE)
        text(c, a, M + 18, yy - rh / 2 + 8, 330, size=12.5, color=INK, bold=True, font="Courier")
        text(c, b, M + 370, yy - rh / 2 + 9, W - 2 * M - 390, size=12.5, color=INK2, leading=15)


def main(out):
    c = canvas.Canvas(out, pagesize=PAGE)
    c.setTitle("ROCKE-3D to JAX: Summary")
    for fn in (s_title, s_approach, s_results, s_stages, s_profile, s_accuracy, s_scope, s_lessons, s_refs):
        fn(c)
        c.showPage()
    c.save()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "summary_deck.pdf")
