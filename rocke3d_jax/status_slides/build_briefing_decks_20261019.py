"""Build the three ROCKE-3D / JAX briefing decks of 2026-10-19: management, science, developer.

usage (no network needed; reportlab and Pillow are the only requirements):
    python3 make_figures.py images            # the three figures
    python3 build_briefing_decks_20261019.py  # three PDFs next to this file, PNG previews in preview_*/ (not tracked)

Every number is typed in from a document named in the slide footer (STATUS_20261016.md of this branch, Reports/ in this branch, or the
documents of the archive tag archive/full-fidelity-port-20261016). Nothing here is read from run files, so the decks do not change when the data does.
Layout layer: deck_layout.py (reportlab; text blocks that do not fit raise at build time).
"""
import os
import sys

from reportlab.pdfgen import canvas

import deck_layout as L
from deck_layout import (AMBER, BLUE, CARD2, GREEN, INK, INK2, MUTED, ORANGE, W, H, MARGIN, Page, banner, card, footer, header)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
DATE = "2026-10-19"
BRANCH = "minimal-port-20261016"


def _page(title, subtitle, eyebrow, size=24):
    p = Page(title)
    y = header(p, eyebrow, title, subtitle, size)
    return p, y


def _metric(p, x, y, w, h, value, label, detail, color=GREEN, value_size=28):
    card(p, x, y, w, h, color)
    p.text(x + 15, y + 10, w - 30, value, value_size, INK, True, maxh=40)
    p.text(x + 15, y + 10 + value_size * 1.5, w - 30, label, 10.5, INK2, True, maxh=28)
    p.text(x + 15, y + 10 + value_size * 1.5 + 30, w - 30, detail, 9, MUTED, maxh=h - (value_size * 1.5 + 44))


def _metric_c(p, x, y, w, h, value, label, detail, color=GREEN):
    """Compact metric card: value, label and a two-line detail in about 76 pt."""
    card(p, x, y, w, h, color)
    p.text(x + 15, y + 6, w - 30, value, 20, INK, True, maxh=26)
    p.text(x + 15, y + 31, w - 30, label, 9.8, INK2, True, maxh=14)
    p.text(x + 15, y + 46, w - 30, detail, 8.6, MUTED, maxh=h - 48)


def _cards(p, x, y, w, h, title, items, accent=BLUE, size=10.2, gap=8):
    card(p, x, y, w, h, accent)
    p.text(x + 15, y + 11, w - 30, title, 12.5, INK, True, maxh=22)
    yy = y + 38
    for it in items:
        p.text(x + 16, yy, 12, "+", size, accent, True)
        used = p.text(x + 30, yy, w - 46, it, size, INK2)
        yy += used + gap
    if yy - gap > y + h:
        L.OVERFLOWS.append("%s: card '%s' needs %.0f pt, box %.0f pt" % (p.name, title, yy - gap - y, h))


def _save(path, title, slides, preview):
    L.OVERFLOWS.clear()
    c = canvas.Canvas(path, pagesize=(W, H))
    c.setTitle(title)
    c.setAuthor("ROCKE-3D JAX project (G. Tamkin)")
    os.makedirs(preview, exist_ok=True)
    for i, make in enumerate(slides, 1):
        p = make(i)
        p.to_pdf(c)
        p.to_png(os.path.join(preview, "slide_%02d.png" % i))
    c.save()
    for problem in L.OVERFLOWS:
        print("OVERFLOW:", problem)
    if L.OVERFLOWS:
        raise RuntimeError("%d layout overflow(s) in %s" % (len(L.OVERFLOWS), path))
    print("wrote", path, "%d slides" % len(slides))


# =====================================================================================================================
# MANAGEMENT
# =====================================================================================================================
def management():
    def m1(n):
        p, y = _page("ROCKE-3D on JAX: where the project stands",
                     "Goal: run the ROCKE-3D P2SAoM40 climate model on GPUs through a JAX rewrite that is shown to behave like the original Fortran model.",
                     "MANAGEMENT BRIEFING | %s" % DATE, 24)
        p.rect(MARGIN, y + 6, 470, 244, L.CARD, 8)
        p.image(MARGIN + 8, y + 10, 454, 196, os.path.join(IMG, "fig_step_time.png"))
        p.text(MARGIN + 14, y + 210, 444, "Range over the runs on exclusive V100 nodes (Fortran on one CPU core). Steady state only: start-up (about 16 + 2 minutes of compilation) is excluded.", 8.4, MUTED, maxh=28)
        mx, mw = MARGIN + 486, W - 2 * MARGIN - 486
        _metric_c(p, mx, y + 6, mw, 76, "2.7-4.4x", "Fast configuration, steady state", "0.51-0.84 s per step against about 2.3 s for Fortran; the spread is unexplained.", GREEN)
        _metric_c(p, mx, y + 90, mw, 76, "0.75-0.97", "Release configuration / Fortran", "Near parity, not a speed win (1.7-2.2 s per step).", BLUE)
        _metric_c(p, mx, y + 174, mw, 76, "NOT MET", "Strict fidelity rule", "Cloud ice at step 3 is 2.37x (release), 2.47x (fast) the reference spread; limit 2x.", ORANGE)
        banner(p, MARGIN, y + 262, W - 2 * MARGIN, 66,
               "Bottom line: the JAX model runs a full day with radiation in JAX and is faster than Fortran in steady state, but it does not yet meet the agreed accuracy rule, the fast speed is not stable between jobs, and only the author has run the handoff. A research result, not a production claim.",
               AMBER, 11.2)
        p.text(MARGIN, y + 338, W - 2 * MARGIN, "Three labels to keep apart", 12, INK, True)
        p.text(MARGIN, y + 357, W - 2 * MARGIN,
               "RELEASE = all speed flags off, near parity.   FAST = nine speed flags on, 2.7-4.4x steady state, unqualified.   QUALIFICATION = repeat, fidelity, long-run and second-person checks still open.", 10.2, INK2, maxh=40)
        footer(p, n, "STATUS_20261016 s5 (runs c-e and the 2026-10-18 step-time runs); RELEASE_NATIVE_RADIATION_20261012; ACCEPTANCE_CRITERIA s9-10")
        return p

    def m2(n):
        p, y = _page("How the project got here", "Seven weeks, five stages: build a trustworthy reference, port piece by piece, couple, replace the radiation callback, then speed it up and hand it over.",
                     "01 | JOURNEY", 23)
        cw, gap = 166, 14
        stages = [
            ("1  REDUCED PORT", "28 Aug - 23 Sep", "Reduced driver with GPU kernels. It recommended against a full port; the owner reversed that.", "Kernels 2.3-6.3x faster than CPU (reduced scope, since superseded).", MUTED),
            ("2  REFERENCE + PIECES", "24 Sep - 6 Oct", "Real Fortran rebuilt as the oracle. Ocean, ice, dynamics, clouds and land ported and checked against it call by call.", "Every ported routine matched the real model to float64 precision; 680 tests passing by 30 Sep.", BLUE),
            ("3  COUPLE + SCORE A DAY", "6 - 8 Oct", "Pieces chained into one JAX-driven step; one model day scored against five real one-ulp runs.", "First coupled step 559 of 559 fields bitwise equal to the NumPy chain on 3 dates; day rule NOT MET.", BLUE),
            ("4  RADIATION + V100", "10 - 13 Oct", "SOCRATES radiation ported to JAX; first V100 benchmark; fast configuration built.", "Release near parity (0.95-0.97 of Fortran); fast 0.524 s/step in the archived run.", GREEN),
            ("5  HANDOFF", "16 - 19 Oct", "Minimal branch, one-command runner, shared environment, group-readable data, three briefings.", "Full runner: 7 jobs exit 0; 3,288 tests pass; only the author has run it.", GREEN),
        ]
        yy = y + 14
        for i, (t_, d, txt, res, col) in enumerate(stages):
            x = MARGIN + i * (cw + gap)
            card(p, x, yy, cw, 232, col)
            p.text(x + 13, yy + 12, cw - 24, t_, 10.5, col, True, maxh=30)
            p.text(x + 13, yy + 40, cw - 24, d, 9, MUTED, True, maxh=14)
            p.text(x + 13, yy + 60, cw - 24, txt, 11, INK2, maxh=100)
            p.rect(x + 13, yy + 160, cw - 26, 1, L.RULE)
            p.text(x + 13, yy + 168, cw - 24, res, 9.8, col, True, maxh=60)
        banner(p, MARGIN, yy + 246, W - 2 * MARGIN, 64,
               "Owner decisions that shaped it: end-to-end JAX first (7 Oct); a Fortran radiation callback was acceptable but must be labelled (7 Oct); the day must never exceed 2x the reference spread (8 Oct); a relaxed 3x rule may be used only for experiments and shown beside the strict result (11 Oct); amended criteria signed off (17 Oct).",
               AMBER, 10.4)
        footer(p, n, "PROJECT_CHRONOLOGY (archive); Project_Summary_and_Conclusions; GOAL; ACCEPTANCE_CRITERIA s9-10, sign-off 2026-10-17; RELEASE_NATIVE_RADIATION_20261012")
        return p

    def m3(n):
        p, y = _page("What the evidence says, and does not say", "The speed number is not the only gate: fidelity, repeatability and run length decide whether it is useful.",
                     "02 | EVIDENCE AND LIMITS", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 262, "SUPPORTED BY MEASUREMENTS", [
            "The full runner works end to end on this branch: seven SLURM jobs, all exit 0, summary table written (18 Oct).",
            "Regression: 3,288 tests passed, 0 failed, 40 skipped, in both the release and the fast configuration.",
            "Fast configuration: 0.51-0.53 s per step in four exclusive-node runs, which reproduces the archived 0.524 s.",
            "Release configuration: near parity with Fortran on two nodes; native JAX radiation cut the median radiation step from 10.0 s to 1.9 s.",
            "Day 2 of the scored window stays within 2x the reference spread (worst 1.22 release, 1.11 fast).",
        ], GREEN, 11.6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 262, "NOT ESTABLISHED", [
            "Strict fidelity rule NOT MET on day 1: cloud ice at step 3 is 2.37x / 2.47x the spread (limit 2x), and steps 0-2 are further beyond. The relaxed experiment rule (3x from step 3) is met.",
            "Fast speed is unstable: 0.51-0.53 s in four runs, 0.80-0.84 s in four others; the node, code, environment and Fortran runs were ruled out.",
            "Runs are not bitwise repeatable (native radiation, same code).",
            "One start date, one scored day, 108 timed steps; the 30-day run is invalid after step 1288.",
            "Only the author has run the handoff, the shared environment and the shared data.",
        ], ORANGE, 11.6)
        banner(p, MARGIN, y + 282, W - 2 * MARGIN, 56,
               "Scope: nov26 start, V100, float64, inputs from one recorded Fortran run. Do not generalize to other seasons, GPUs or long production runs.", AMBER, 10.5)
        footer(p, n, "STATUS_20261016 s2, s4, s5; ACCEPTANCE_CRITERIA s9-10; RELEASE_NATIVE_RADIATION_20261012; SPEED_STEP_TIME_LOG_20261012 (archive)")
        return p

    def m4(n):
        p, y = _page("Decisions needed from management", "The next work should answer what the speedup is for and what 'good enough' means.", "03 | DECISIONS", 23)
        cw, gap = (W - 2 * MARGIN - 15) / 2, 15
        _cards(p, MARGIN, y + 6, cw, 148, "1  TARGET USE", [
            "What run length and restart cadence matter? Start-up costs about 17 minutes of compilation; payback against Fortran is about 12-15 model days (arithmetic, not measured).",
            "Is the V100 the deployment target, or should another GPU be qualified?",
        ], BLUE, 11.6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 148, "2  SCIENCE BAR", [
            "Keep the strict 2x rule for release, or approve a separate experimental criterion?",
            "Who owns the disposition of the cloud-ice excess at step 3 (2.37 -> 2.47 with the fast flags), and which further dates and run lengths are needed?",
        ], ORANGE, 11.6)
        _cards(p, MARGIN, y + 168, cw, 148, "3  QUALIFICATION RESOURCES", [
            "Exclusive V100 time for repeats and for the step-time question.",
            "A second developer to reproduce the handoff (nobody but the author has).",
        ], GREEN, 11.6)
        _cards(p, MARGIN + cw + gap, y + 168, cw, 148, "4  RELEASE CONTROL", [
            "The minimal branch is now the repository default; the full history is kept under a tag.",
            "Turn the fast flags on by default only after the gates on the next slide pass. Support the shared Python environment and the Numba worker it needs.",
        ], BLUE, 11.6)
        banner(p, MARGIN, y + 332, W - 2 * MARGIN, 52, "Recommended posture: fund a time-boxed qualification, not a production claim. Keep the release as the stable baseline.", AMBER, 11)
        footer(p, n, "ACCEPTANCE_CRITERIA s9-10; STATUS_20261016 s4-6; SPEED_NEXT_STEPS_20261013 (archive); DATA_AND_PATHS")
        return p

    def m5(n):
        p, y = _page("Proposed gates: from a promising result to a decision", "Small, ordered checks; stop or re-scope if the evidence does not support deployment.", "04 | NEXT STEPS", 23)
        rows = [
            ("A", "REPRODUCE", "A second person runs gpu/run_variants.sh from a fresh clone; explain why the fast step time is 0.52 s in some jobs and 0.80-0.84 s in others.", BLUE),
            ("B", "QUALIFY THE BUILD", "Rebuild the environment from its file (untested); package the Numba worker; trace the 17 untraced test skips.", BLUE),
            ("C", "CHECK FIDELITY", "Run the CPU day with Intel libimf; localize the repeat differences of native radiation; score more dates; keep strict and relaxed results side by side.", ORANGE),
            ("D", "LONGER RUNS", "Fix the 30-day failure (Fortran branch the port does not execute) and run the fast flags beyond two days.", ORANGE),
            ("E", "PROMOTE OR HOLD", "Compare the evidence with the target workload and the science bar, then decide flag defaults and release.", GREEN),
        ]
        yy = y + 12
        for key, t, d, col in rows:
            card(p, MARGIN, yy, W - 2 * MARGIN, 58, col)
            p.text(MARGIN + 14, yy + 11, 30, key, 20, col, True, maxh=28)
            p.text(MARGIN + 54, yy + 12, 150, t, 12, INK, True, maxh=30)
            p.text(MARGIN + 214, yy + 9, W - 2 * MARGIN - 230, d, 12, INK2, maxh=46)
            yy += 66
        banner(p, MARGIN, yy + 2, W - 2 * MARGIN, 44, "No schedule is implied: the owner sets dates, run counts and acceptance limits once the target workload is chosen.", AMBER, 10.5)
        footer(p, n, "STATUS_20261016 s4-6; REPEATABILITY_AND_FLAGS_REGRESSION_20261014; QA_HANDOFF_20261016 (archive)")
        return p

    return [m1, m2, m3, m4, m5]


# =====================================================================================================================
# SCIENCE
# =====================================================================================================================
def science():
    def s1(n):
        p, y = _page("What is being ported, and what is still not JAX", "The configuration is the published ROCKE-3D 2.0 P2SAoM40 setup; the port is a hybrid, and every result says which parts are hybrid.",
                     "SCIENCE BRIEFING | %s" % DATE, 23)
        cw, gap = 282, 21
        _cards(p, MARGIN, y + 6, cw, 252, "THE MODEL", [
            "ModelE2 planet 2.0, rundeck P2SAoM40: Sun-like star, dynamic ocean (13 layers), 40-layer atmosphere on a 72 x 46 grid, 30-minute step.",
            "Intended use: reproduce the published climatology (Tsigaridis et al. 2025) at far lower cost so that parameter sweeps of rocky-planet climates become practical.",
            "Scored here: one model day (54 steps) from the nov26 restart.",
        ], BLUE, 11.2)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 252, "RUNS IN JAX (V100, float64)", [
            "Atmosphere dynamics, clouds and convection (CONDSE), surface tiles, land (GHY, Ent), ocean, sea ice, lakes.",
            "Radiation: SOCRATES ported to JAX (--rad native), or the unmodified Fortran callback (--rad server), always labelled.",
            "All behaviour flags default off; the fast configuration adds nine opt-in flags.",
        ], GREEN, 11.2)
        _cards(p, MARGIN + 2 * (cw + gap), y + 6, cw, 252, "STILL HOST OR RECORDED", [
            "Five step-0 Fortran dump files (about 278 MB) and the nov26 restart; irrigation year fixed at 1850, no fossil groundwater.",
            "Day-boundary update, pole columns and Intel libimf math run on the host (libimf is needed: without it cloud thresholds flip in 3-5% of columns).",
            "Fast configuration: the cloud-base stage runs in a separate Numba process.",
        ], ORANGE, 11.2)
        banner(p, MARGIN, y + 272, W - 2 * MARGIN, 56,
               "Hybrid boundary: the reference and the JAX inputs come from the same recorded Fortran run, so a start from a bare restart, other seasons and long free runs are not yet demonstrated.", AMBER, 10.5)
        footer(p, n, "GOAL (archive); RELEASE_NATIVE_RADIATION_20261012; ACCEPTANCE_CRITERIA s1; STANDALONE_INPUT_AUDIT (archive); SPEED_RUN_RECIPE_20261013")
        return p

    def s2(n):
        p, y = _page("How fidelity is measured", "Plain-language wording drafted from the acceptance criteria; the wording has not been reviewed by the owner or a domain expert.", "01 | METHOD", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 200, "THE IDEA", [
            "The climate model is chaotic: nudging the starting point by one rounding unit gives a slightly different answer later (about 1-3% of the field variability within 5 days in the measured case).",
            "So the yardstick is the original model's own variation: five Fortran runs started one rounding-unit apart ('members').",
            "Ratio = how far JAX is from the real Fortran run, divided by the largest member distance from it.",
        ], BLUE, 11.2)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 200, "THE RULES (OWNER-SET, ACCEPTANCE s9-10)", [
            "Strict: never beyond 2x at every step and every scored field (8 Oct).",
            "Relaxed, experiments only: at most 3x from step 3, always shown beside the strict result (11 Oct).",
            "A ratio of 2.37 means cloud ice is 2.37 times the reference spread away from the reference, not that there is 2.37 times more ice.",
        ], ORANGE, 11.2)
        _cards(p, MARGIN, y + 220, cw, 130, "TWO DIFFERENT COMPARISONS", [
            "C2 (fidelity): JAX against the real Fortran members. Only C2 supports a statement about ROCKE-3D.",
            "C1 (port consistency): JAX against our NumPy chain; says nothing about Fortran.",
        ], GREEN, 11)
        _cards(p, MARGIN + cw + gap, y + 220, cw, 130, "A SEPARATE, SECONDARY CHECK", [
            "Agreement to many decimals (categories A to D) is a precision diagnostic, not the fidelity test; ulp-level differences grow in a chaotic model without meaning physics is missing.",
        ], AMBER, 11)
        footer(p, n, "ACCEPTANCE_CRITERIA s2-4, s9-10; FIDELITY_EXPLAINER_NOTES (archive status_slides, draft); PROJECT_CHRONOLOGY phase B (noise floor D3)")
        return p

    def s3(n):
        p, y = _page("Result: one model day from the nov26 start", "Release and fast configurations, native radiation, scored against five one-ulp Fortran members (run of 2026-10-18).", "02 | FIDELITY RESULT", 23)
        p.rect(MARGIN, y + 6, 520, 238, L.CARD, 8)
        p.image(MARGIN + 8, y + 12, 504, 226, os.path.join(IMG, "fig_fidelity.png"))
        cx, cwid = MARGIN + 536, W - 2 * MARGIN - 536
        _cards(p, cx, y + 6, cwid, 238, "READING THE CHART", [
            "Worst ratio at steps 3 to 53. Only cloud ice (QCI) at step 3 exceeds 2x: 2.37 release, 2.47 fast.",
            "Steps 0-2 are not shown and are worse: P 20.7 / 24.2 at step 0, QCL 7.9 at step 1, U 3.4 / 2.6 at step 1.",
            "Day 2: never beyond 2x (worst 1.22 release, 1.11 fast).",
        ], BLUE, 10.6)
        p.table(MARGIN, y + 256, [330, 150, 150, 258], [
            ["Rule", "Release", "Fast", "Source"],
            ["Strict, all steps (approved)", ("NOT MET", ORANGE), ("NOT MET", ORANGE), "ACCEPTANCE s9"],
            ["Strict, steps >= 3 only (secondary)", ("NOT MET", ORANGE), ("NOT MET", ORANGE), "ACCEPTANCE s9"],
            ["Relaxed 3x, steps >= 3 (experiment)", ("MET", GREEN), ("MET", GREEN), "ACCEPTANCE s10"],
        ], size=9.2, maxh=92, pad=5, bold_cols=(0,))
        p.text(MARGIN, y + 352, W - 2 * MARGIN, "Release ratios repeated to two decimals in three runs; fast ratios vary between repeats by up to 0.15 (P) while QCI at step 3 stays 2.47. The fast flags shift QCI by about 0.1 and are not bitwise equal to the release (Ent batching, host cloud stage).", 9, MUTED, maxh=28)
        footer(p, n, "STATUS_20261016 s5 (runs c, d, e; d207_score.py against members p1-p5); ACCEPTANCE_CRITERIA s9-10; SPEED_STEP_TIME_LOG_20261012 s13 (archive)")
        return p

    def s4(n):
        p, y = _page("Radiation: from a Fortran callback to native JAX", "The port started with the original radiation code served from a helper process; a JAX port of the SOCRATES call followed.", "03 | RADIATION", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 220, "STEP 1: THE ORIGINAL CODE AS A SERVER (6-7 OCT)", [
            "A helper built from the unmodified model runs its radiation step on inputs we supply; 21 of 22 outputs are bitwise equal to the recorded model.",
            "Cost 11-14 s per call; every result carried the sentence 'radiation computed by the original Fortran (hybrid component)'.",
            "A free-running day through the server kept temperature and winds inside the member spread and global-mean radiative flux within 1 W/m2.",
        ], BLUE, 11)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 220, "STEP 2: SOCRATES IN JAX (11-12 OCT)", [
            "Ten consecutive nov26 radiation steps reproduce the recorded server outputs at rounding level (21 fields, AIJ excluded), CPU and V100; one V100 does the whole call in 0.174 s.",
            "Day-varying products bitwise against the instrumented server on 48 dates.",
            "In the coupled run the median radiation step fell from 10.0 s to 1.9 s.",
        ], GREEN, 11)
        banner(p, MARGIN, y + 240, W - 2 * MARGIN, 66,
               "Limits: only the nov26 window has been run through the full model; the 750 AIJ diagnostics are not ported; the cadence of the native radiation call is unconfirmed; same-code native-radiation repeats are not bitwise and the cause is not localized.", ORANGE, 10.2)
        p.text(MARGIN, y + 322, W - 2 * MARGIN, "Where the Fortran time went (profile of the unmodified call): aerosol-table interpolation 37%, two-stream flux solvers 26%, optical properties 12%, band driver 8%, math library and I/O 10%, other ModelE side 7%.", 11, INK2, maxh=40)
        footer(p, n, "SOCRATES_SPIKE_RESULT s2; ACCEPTANCE_CRITERIA s12a-12d; RELEASE_NATIVE_RADIATION_20261012; PROJECT_CHRONOLOGY D152-D157 (archive)")
        return p

    def s5(n):
        p, y = _page("What this evidence cannot say", "Each statement has a scope; a passing score does not imply broader validity.", "04 | LIMITS", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 262, "ABOUT THE TEST", [
            "It does not prove the physics is complete. Whether an explicit physics-coverage check exists has not been established.",
            "Trajectories are never compared; only distributions against the member spread.",
            "One start date (nov26), one scored day, V100 float64.",
            "The reference and the JAX inputs come from one recorded Fortran run.",
        ], BLUE, 11.2)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 262, "ABOUT REPEATABILITY AND LONG RUNS", [
            "Native-radiation repeats of the same code differ: raw surface radiative flux by up to 543 W/m2 pairwise in 54-step runs; zenith bitwise equal; cause not localized.",
            "GPU exp/pow differ from CPU libm by ulps; the owner-approved route is an Intel libimf host callback for the headline comparison.",
            "The 30-day shadow run is invalid after step 1288: a Fortran branch (qdynam_do_z_extra) that the port does not execute.",
            "The fast flags were run for 108 steps only, and they change arithmetic (Ent batching is not bitwise equal; QCI at step 3 moves 2.37 -> 2.47).",
        ], ORANGE, 11.2)
        banner(p, MARGIN, y + 282, W - 2 * MARGIN, 52, "Describe this as a well-documented experimental result, not as a validated or production-ready port.", AMBER, 11)
        footer(p, n, "REPEATABILITY_AND_FLAGS_REGRESSION_20261014; STATUS_20261016 s4; ACCEPTANCE_CRITERIA s8, s10; SPEED_STEP_TIME_LOG_20261012 s13-14 (archive)")
        return p

    def s6(n):
        p, y = _page("Open science questions and the check for each", "Ordered by how much they would change the conclusions.", "05 | NEXT", 23)
        rows = [
            ["Question", "Proposed check", "State"],
            ["Why does cloud ice exceed 2x at step 3 (2.37, 2.47)?", "CPU day with Intel libimf (cpu_day_run_imf.py). Caution: the fast run already uses libimf for the cloud-base stage and still gives 2.47.", "dry-run only"],
            ["Why are same-code native-radiation repeats not bitwise?", "Localize the staged-input path or flag responsible; a flags-off short window repeated bitwise.", "cause not found"],
            ["Is the physics complete?", "An explicit physics-coverage check, separate from the spread test.", "not established"],
            ["Other seasons and longer runs?", "More start dates (restart, step-0 dumps and members exist for nov26 only); fix the 30-day failure.", "not done"],
            ["Does host NumPy on AVX-512 nodes move the ratios?", "Repeat a day with the AVX-512 paths disabled (they change 65 bitwise tests).", "untested"],
            ["Which published target?", "100-year annual mean or a shorter run; to be confirmed with the project lead.", "open"],
        ]
        p.table(MARGIN, y + 8, [270, 470, 148], rows, size=11.4, maxh=370, pad=9, bold_cols=(0,))
        footer(p, n, "STATUS_20261016 s4; QA_HANDOFF_20261016 and GOAL (archive); fullfidelity/qa_20261016/ENVIRONMENT.md; DATA_AND_PATHS")
        return p

    return [s1, s2, s3, s4, s5, s6]


# =====================================================================================================================
# DEVELOPER
# =====================================================================================================================
def developer():
    def d1(n):
        p, y = _page("Orientation: the minimal branch", "One orphan branch holds the source, tests and recipes for the release and fast configurations; everything else is in the archive tag.",
                     "DEVELOPER BRIEFING | %s" % DATE, 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 172, "WHERE THINGS ARE", [
            "Repo nasa-nccs-hpda/or0cky, default branch minimal-port-20261016. Archive: tag archive/full-fidelity-port-20261016 (scoping notes, drafts, run records).",
            "Read in order: README.md, rocke3d_jax/README.md, STATUS_20261016.md, DATA_AND_PATHS.md.",
            "Data (ff_data, the ModelE tree, binaries) lives outside the repo.",
        ], BLUE, 10.8)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 172, "SIZE", [
            "fullfidelity/: 317 Python modules (about 76,600 lines) plus 135 test files; socrates_spike/: 16 modules.",
            "gpu/: run_variants.sh, summarize_variants.py, benchmark and gate scripts, environment file.",
            "Reports/: acceptance criteria, release and speed recipes, repeatability report.",
        ], GREEN, 10.8)
        p.table(MARGIN, y + 190, [300, 588], [
            ["Entry point", "What it does"],
            ["gpu_day_run.py -> d212_day.py", "GPU day run (54 or 108 steps); --rad native or server"],
            ["gpu_step_run.py / d193_day.py", "6-step GPU gate / the day loop under it"],
            ["cpu_day_run_imf.py", "CPU day with Intel libimf (dry-run only so far)"],
            ["d207_score.py, multiday_score.py, atm_day_report.py", "Scoring against the five one-ulp members; defines within / near / beyond"],
            ["run_all_tests_sharded.sh, run_all_tests_flags_on.sh", "Regression subset, flags off / nine flags on"],
        ], size=9.8, maxh=136, pad=4, bold_cols=(0,))
        footer(p, n, "README.md; rocke3d_jax/README.md; STATUS_20261016 s1; NEW_BRANCH_BRIEF_20261016 s3 (archive)")
        return p

    def d2(n):
        p, y = _page("Anatomy of a coupled step", "A mixed host/device program: the release path is all flags off; the fast path swaps stages through opt-in flags.", "01 | ARCHITECTURE", 23)
        stages = [("STATE + SETUP", "restart, five step-0\ndump files, providers", BLUE), ("ATMOSPHERE", "dyn_step_jax2.py\njax_p1_condse.py (CONDSE)", GREEN),
                  ("SURFACE + LAND", "surface tiles, PBL, GHY\nEnt, daily update", GREEN), ("OCEAN + ICE", "ocean_chain_io.py\nsea ice, lakes, pole", GREEN),
                  ("RADIATION", "socrates_spike/\nradia_native.py", GREEN)]
        gap, ww = 13, 166
        yy = y + 12
        for i, (t, d, col) in enumerate(stages):
            x = MARGIN + i * (ww + gap)
            card(p, x, yy, ww, 98, col)
            p.text(x + 12, yy + 11, ww - 24, t, 10, col, True, maxh=16)
            p.text(x + 12, yy + 34, ww - 24, d, 10.4, INK2, mono=False, maxh=58)
            if i < 4:
                p.text(x + ww + 1, yy + 38, gap, ">", 12, MUTED, True)
        cw = (W - 2 * MARGIN - 16) / 2
        _cards(p, MARGIN, yy + 112, cw, 188, "RELEASE CONFIGURATION", [
            "jax_coupled.py Coupled.step drives the stages on device; atm_step.py is the NumPy chain used for C1 consistency.",
            "Native radiation, standalone flags (ROCKE_TPL_OWN etc.) set by the runner; no speed flags.",
            "Needs XLA flags --xla_disable_hlo_passes=algsimp,reshape-mover on CPU (cold compile 30x faster, bitwise identical).",
        ], BLUE, 10.8)
        _cards(p, MARGIN + cw + 16, yy + 112, cw, 188, "FAST CONFIGURATION (nine flags)", [
            "MSTCNV on the host in a persistent Numba worker (ROCKE_MSTCNV_PY), libimf exp/pow, batched host I/O.",
            "QDYN vmap, pole columns fast + async, sequential-sum blocks of 32, glue jit, Ent batching.",
            "Flags default off; ENT_BATCH is not bitwise equal to the level-by-level version.",
        ], GREEN, 10.8)
        banner(p, MARGIN, yy + 312, W - 2 * MARGIN, 42, "Input boundary: nov26 restart plus five step-0 Fortran dumps; no per-step records are read in the standalone-capable path.", AMBER, 10)
        footer(p, n, "JAX_COVERAGE_MATRIX (archive); SPEED_RUN_RECIPE_20261013; RELEASE_NATIVE_RADIATION_20261012; ENVIRONMENT.md; LESSONS: XLA flag fix D182")
        return p

    def d3(n):
        p, y = _page("Running it: one command, seven SLURM jobs", "gpu/run_variants.sh submits everything and a dependent summary job; it prints one table, gpu_runs/<name>/SUMMARY.md.", "02 | RUN", 23)
        rows = [
            ["Job", "Where", "What"],
            ["pair_release, pair_fast", "exclusive V100, 8 CPUs, 64 GB", "Fortran x3 on one core, then JAX, 108 steps, same node (timing pair)"],
            ["fid_release, fid_fast", "V100, shared node ok", "54-step day, scored against the five members by d207_score.py"],
            ["tests_off, tests_on", "CPU", "Sharded regression with speed flags off / nine flags on; 2 isolated files"],
            ["summary", "CPU, runs after the others end", "summarize_variants.py writes SUMMARY.md and score_<kind>.md"],
        ]
        p.table(MARGIN, y + 6, [210, 250, 428], rows, size=10.2, maxh=134, pad=6, bold_cols=(0,))
        cw = (W - 2 * MARGIN - 16) / 2
        _cards(p, MARGIN, y + 142, cw, 196, "SETUP (DATA_AND_PATHS.md s4)", [
            "Export FF_DATA, MODELE_PROD_INPUT (the complete ModelE_Support/prod_input_files), ROCKE_SOCRATES_DIR, FORTRAN_HUGE_SPACE, FORTRAN_RUNS, ROCKE_MSTCNV_PY, CONDA_BIN.",
            "Optional: ROCKE_ENV (name or absolute path), SBATCH_HOST=gpulogin1, ONLY='fid_fast tests_off', DRY_RUN=1.",
            "Pre-flight checks exit with code 2 and name what is missing.",
        ], BLUE, 10.8)
        _cards(p, MARGIN + cw + 16, y + 142, cw, 196, "EXPECTED RESULT (RUN E, 18 OCT)", [
            "All seven jobs exit 0 in 38 minutes from submission (queue time extra).",
            "Fidelity worst ratio (steps >= 3): release QCI 2.37, fast QCI 2.47.",
            "Tests: 3,288 passed, 0 failed, 40 skipped in both configurations.",
            "Fortran 251-254 s per run; JAX step 0 about 940-1,090 s, step 1 96-104 s.",
        ], GREEN, 10.8)
        banner(p, MARGIN, y + 350, W - 2 * MARGIN, 40, "forest204 has no SLURM client: set SBATCH_HOST=gpulogin1. Set PYTHONNOUSERSITE=1 so a personal ~/.local cannot change results.", AMBER, 10)
        footer(p, n, "gpu/run_variants.sh; gpu/summarize_variants.py; DATA_AND_PATHS s4; STATUS_20261016 s5 (run e)")
        return p

    def d4(n):
        p, y = _page("Environment, data and cluster facts", "Everything outside the repository, and the traps that cost the most time.", "03 | ENVIRONMENT", 23)
        cw = (W - 2 * MARGIN - 16) / 2
        _cards(p, MARGIN, y + 6, cw, 200, "ENVIRONMENTS (READ-ONLY, GROUP ilab)", [
            "JAX: /explore/nobackup/projects/ilab/conda/envs/rocke3d-jax (Python 3.10, jax 0.5.3, NumPy 2.2.4, mpmath 1.3.0), made by conda-pack; passed the 6-step V100 gate bitwise.",
            "Numba worker (ROCKE_MSTCNV_PY): the ilab pytorch-kernel-copy_20260408 Python (3.12, numba 0.60, NumPy 2.0.2).",
            "Rebuild from gpu/environment-graphcast-env.yml is untested (package server rate limit).",
        ], BLUE, 10.8)
        _cards(p, MARGIN + cw + 16, y + 6, cw, 200, "TEST HOSTS (XEON GOLD 6248, AVX-512)", [
            "JAX_PLATFORMS=cpu, or a job landing on a GPU node fails to start CUDA.",
            "XLA_FLAGS='--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover'.",
            "NPY_DISABLE_CPU_FEATURES for the AVX-512 set: NumPy pow differs in the last bit otherwise (probe: 16 of 83 failed).",
            "MODELE_SRC must be set (3 tests skip without it).",
        ], ORANGE, 10.8)
        _cards(p, MARGIN, y + 218, W - 2 * MARGIN, 128, "TRAPS", [
            "MODELE_PROD_INPUT must be the complete prod_input_files; the in-repo copy is partial and a day run then fails early behind a misleading 'NoneType ... calls' error.",
            "Two tests (test_daily_lake_hook, test_d187_imf_p1) must run in their own process; test_libimf_fused is order-dependent. test_jax_state collects no tests (rc=5 is expected).",
            "The data is group-readable under the author's directories (never o+r); do not copy it into home directories without asking.",
        ], AMBER, 10.8)
        footer(p, n, "DATA_AND_PATHS; fullfidelity/qa_20261016/ENVIRONMENT.md; STATUS_20261016 s5; run_variants.sh")
        return p

    def d5(n):
        p, y = _page("Benchmark arithmetic and what the speed claim covers", "Separate steady-state throughput from start-up; compare only equal windows.", "04 | MEASUREMENT", 23)
        cw = (W - 2 * MARGIN - 16) / 2
        _cards(p, MARGIN, y + 6, cw, 200, "THE MATCHED PAIR", [
            "One exclusive V100 node: Fortran on one core (x3), then JAX, same nov26 restart, 108 steps (2 model days).",
            "Archived fast pair (job 37941875): Fortran main loop 246.7 / 246.6 s (2.285 s/step); JAX steps 2-107: 55.6 s (0.524 s/step).",
            "Branch runs: Fortran whole process 251-254 s (about 2.35 s/step, includes start-up).",
        ], BLUE, 10.8)
        _cards(p, MARGIN + cw + 16, y + 6, cw, 200, "WHAT IT DOES NOT INCLUDE", [
            "Step 0: 940-1,090 s; step 1: 96-104 s (compile); no persistent XLA cache.",
            "Crossover with Fortran: (940 + 100) / (2.285 - step time) = about 590 steps at 0.524 s, about 700 at 0.80 s, i.e. 12-15 days at 48 steps/day. Arithmetic, not measured.",
            "Release: 1.7-2.2 s/step, parity at best.",
        ], ORANGE, 10.8)
        banner(p, MARGIN, y + 220, W - 2 * MARGIN, 52, "Say '2.7-4.4x steady state on one fast configuration (V100, nov26)'. Do not say 'the model runs 4.4x faster': start-up dominates a 108-step run and the long-run workload is untested.", AMBER, 10.4)
        p.table(MARGIN, y + 286, [250, 160, 160, 318], [
            ["Run", "Mean s/step", "Median", "Note"],
            ["Fast, archived matched (37941875)", "0.524", "0.456", "steps 2-107"],
            ["Fast, best / worst of 9 runs", "0.508 / 0.841", "0.453 / 0.789", "tag and branch code, exclusive nodes"],
            ["Release, branch (run e)", "1.729", "1.689", "gpu011"],
        ], size=10.2, maxh=100, pad=5, bold_cols=(0,))
        footer(p, n, "SPEED_STEP_TIME_LOG_20261012 s13 (archive); STATUS_20261016 s5; RELEASE_NATIVE_RADIATION_20261012; gpu/summarize_variants.py")
        return p

    def d6(n):
        p, y = _page("Why the fast configuration is faster", "A bundle of changes aimed at measured bottlenecks, not one kernel optimization.", "05 | OPTIMIZATION", 23)
        rows = [
            ["Stage / change", "What changed", "Evidence / caveat"],
            ["Convection (MSTCNV)", "Host Numba worker, Intel libimf exp/pow, batched host I/O", "57 of 57 fields equal to the NumPy-libimf port on captured columns; needs a second Python environment"],
            ["Dynamics, ocean, pole", "QDYN vmap, pole columns fast + async, sequential-sum blocks of 32", "Targeted checks; scan unrolling tried and rejected (slower)"],
            ["Land / Ent", "Batch leaf evaluations across qsimp levels and sun/shade", "Ent call 5.39 -> 2.37 ms; not bitwise equal (1e-15 relative)"],
            ["Host glue", "Template-state glue as one jit; one batched device_get/put", "MSTCNV stage 0.087 -> 0.071 s"],
            ["Radiation", "SOCRATES JAX inside the coupled step", "About 0.17 s per call on a V100; median radiation step 10.0 -> 1.9 s"],
        ]
        p.table(MARGIN, y + 8, [170, 340, 378], rows, size=11.2, maxh=290, pad=8, bold_cols=(0,))
        banner(p, MARGIN, y + 226, W - 2 * MARGIN, 70,
               "Held step 1.71 s (baseline) -> 0.385-0.402 s in the archived profile runs; remaining held-step stages: surface 0.086 s, MSTCNV 0.071 s, dynamics 0.065 s, ocean_b 0.048 s, plus about 0.16 s on radiation steps. XLA command buffers and larger scan unrolls were slower and were not adopted.", BLUE, 10)
        footer(p, n, "SPEED_STEP_TIME_LOG_20261012 s11-13 (archive); SPEED_RUN_RECIPE_20261013; SOCRATES_R2_DUMPS s12 (archive)")
        return p

    def d7(n):
        p, y = _page("The open puzzle: the same fast run takes 0.52 s or 0.84 s", "Nine exclusive-node runs of one configuration; colour = Fortran runs before JAX on that node.", "06 | STEP-TIME VARIATION", 23)
        p.rect(MARGIN, y + 6, 520, 240, L.CARD, 8)
        p.image(MARGIN + 8, y + 10, 504, 232, os.path.join(IMG, "fig_fast_runs.png"))
        _cards(p, MARGIN + 536, y + 6, W - 2 * MARGIN - 536, 240, "WHAT WAS RULED OUT", [
            "The code: tag and branch both give 0.51-0.53 s, and the held step is 0.365 s on both under the profile wrapper.",
            "The node: gpu011 gave fast, fast and slow; gpu012 slow x3, fast x1.",
            "Fortran before JAX: an early 0/1/3 pattern was reversed by two simultaneous jobs.",
            "Memory, NUMA, GPU clocks, throttling, other processes: no difference sampled.",
        ], GREEN, 9)
        banner(p, MARGIN, y + 258, W - 2 * MARGIN, 52, "Cause not identified. In slow runs the whole step-time distribution shifts up (held-step median 0.78 s against 0.46 s); the fastest step is 0.43 s in every run. Quote the range.", ORANGE, 10.4)
        p.text(MARGIN, y + 322, W - 2 * MARGIN, "Suggested next step: instrument a slow and a fast job with per-stage timers (ROCKE_DYN_TIMERS, ROCKE_SURF_TIMERS, ROCKE_MSTCNV_HOST_TIMING) on exclusive nodes started together, and compare which stage moves. The profile wrapper itself adds 0.09-0.34 s of file I/O per step; do not compare its step means.", 9.4, INK2, maxh=48)
        footer(p, n, "STATUS_20261016 s5 (jobs 37943471-37943488); SPEED_RUN_RECIPE_20261013 (diagnostic flags)")
        return p

    def d8(n):
        p, y = _page("Tests, gates and open work", "What was checked on this branch, what was changed on purpose, and what is left.", "07 | QUALITY", 23)
        cw = (W - 2 * MARGIN - 16) / 2
        _cards(p, MARGIN, y + 6, cw, 232, "GATES (16-18 OCT)", [
            "Kept files identical to the archive except about 110 path-scrubbed files and a few changed on purpose (listed in STATUS s2).",
            "6-step V100 run bitwise identical to jobs 37942390, 37942591, 37943287; also from the shared environment (job 37943470).",
            "Scoring reproduces the review-package table (QCI 2.37 at step 3).",
            "Regression: 3,288 passed, 0 failed, 40 skipped (flags off and on); 23 of 40 skips traced to declared causes, 17 untraced.",
        ], GREEN, 10.8)
        _cards(p, MARGIN + cw + 16, y + 6, cw, 232, "OPEN WORK (PRIORITY ORDER)", [
            "Second-person reproduction from a fresh clone (nobody but the author has run it).",
            "Explain the fast step-time variation (previous slide).",
            "CPU libimf day; localize the native-radiation repeat differences; trace the 17 skips.",
            "Rebuild the environment from its file; package the Numba worker.",
            "Fix the 30-day failure (qdynam_do_z_extra) and run beyond two days; other start dates.",
        ], ORANGE, 10.8)
        banner(p, MARGIN, y + 252, W - 2 * MARGIN, 60, "Report both strict and relaxed fidelity results every time; label the radiation actually used (--rad native or server); never state a result the SUMMARY.md of a run does not show.", AMBER, 10.4)
        p.text(MARGIN, y + 324, W - 2 * MARGIN, "Records: STATUS_20261016.md is the record of what was verified; Reports/ACCEPTANCE_CRITERIA.md is the owner-approved rule set (amendments signed off 2026-10-17); the archive tag holds the full ledger (FULL_FIDELITY_DELTAS.md, D1-D198).", 9.4, INK2, maxh=40)
        footer(p, n, "STATUS_20261016 s2, s4-6; ACCEPTANCE_CRITERIA; run_all_tests_sharded.sh; QA_HANDOFF_20261016 (archive)")
        return p

    return [d1, d2, d3, d4, d5, d6, d7, d8]


def main():
    decks = [("management_briefing_20261019", "ROCKE-3D to JAX: management briefing, 2026-10-19", management),
             ("science_briefing_20261019", "ROCKE-3D to JAX: science briefing, 2026-10-19", science),
             ("developer_briefing_20261019", "ROCKE-3D to JAX: developer briefing, 2026-10-19", developer)]
    only = sys.argv[1:]
    for stem, title, fn in decks:
        if only and not any(o in stem for o in only):
            continue
        _save(os.path.join(HERE, stem + ".pdf"), title, fn(), os.path.join(HERE, "preview_" + stem.split("_")[0]))


if __name__ == "__main__":
    main()
