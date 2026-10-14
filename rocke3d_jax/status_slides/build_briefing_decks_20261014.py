"""Build concise management and technical-peer ROCKE-3D/JAX briefing decks (2026-10-14 refresh of the 2026-10-13 decks)."""
import os

from reportlab.pdfgen import canvas

from deck_layout import (
    AMBER, BLUE, CARD, CARD2, GREEN, INK, INK2, MUTED, NAVY, ORANGE,
    W, H, MARGIN, Page, banner, card, footer as _legacy_footer, header,
)
import deck_layout as L


HERE = os.path.dirname(os.path.abspath(__file__))


def footer(p, n, src):
    _legacy_footer(p, n, src)
    p.ops = [
        op[:4] + (op[4].replace("2026-10-08", "2026-10-14"),) + op[5:]
        if op[0] == "text" and "2026-10-08" in op[4] else op
        for op in p.ops
    ]


def _page(title, subtitle, eyebrow, size=24):
    p = Page(title)
    y = header(p, eyebrow, title, subtitle, size)
    return p, y


def _metric(p, x, y, w, h, value, label, detail, color=GREEN, value_size=30):
    card(p, x, y, w, h, color)
    p.text(x + 15, y + 11, w - 30, value, value_size, INK, True, maxh=42)
    p.text(x + 15, y + 54, w - 30, label, 11, INK2, True, maxh=30)
    p.text(x + 15, y + 84, w - 30, detail, 9.5, MUTED, maxh=h - 92)


def _section_card(p, x, y, w, h, title, items, accent=BLUE, body_size=11):
    card(p, x, y, w, h, accent)
    p.text(x + 15, y + 12, w - 30, title, 14, INK, True, maxh=24)
    yy = y + 45
    for item in items:
        p.text(x + 16, yy, 12, "+", 11, accent, True)
        used = p.text(x + 31, yy, w - 47, item, body_size, INK2, maxh=66)
        yy += used + 12


def _save_deck(path, title, slides, preview=None):
    L.OVERFLOWS.clear()
    c = canvas.Canvas(path, pagesize=(W, H))
    c.setTitle(title)
    c.setAuthor("ROCKE-3D JAX project")
    for i, make_slide in enumerate(slides, 1):
        p = make_slide(i)
        p.to_pdf(c)
        if preview:
            os.makedirs(preview, exist_ok=True)
            p.to_png(os.path.join(preview, "slide_%02d.png" % i))
    c.save()
    for problem in L.OVERFLOWS:
        print("OVERFLOW:", problem)
    if L.OVERFLOWS:
        raise RuntimeError("%d slide layout overflow(s)" % len(L.OVERFLOWS))
    print("wrote", path, "%d slides" % len(slides))


# ---------------------------------------------------------------------------
# Management deck

def management_slides():
    def m1(n):
        p, y = _page(
            "ROCKE-3D to JAX: progress, proof, decisions",
            "Goal: a JAX implementation of the P2SAoM40 model that is demonstrably faithful and faster than the original Fortran.",
            "MANAGEMENT BRIEFING | 14 OCT 2026",
            25,
        )
        gap = 14
        cw = (W - 2 * MARGIN - 2 * gap) / 3
        _metric(p, MARGIN, y + 8, cw, 151, "2.13-2.21 s", "Tagged release: JAX / step",
                "Fortran: 2.24-2.28 s (1 CPU core) vs JAX on one V100 GPU, two exclusive nodes: near parity, not a decisive speed win.",
                BLUE, 25)
        _metric(p, MARGIN + cw + gap, y + 8, cw, 151, "0.524 s", "Optimized branch: JAX / step",
                "Fortran: 2.285 s on 1 CPU core; JAX on one V100 GPU. One matched run, steps 2-107: 4.4x steady-state. Not a cost or energy comparison.",
                GREEN, 30)
        _metric(p, MARGIN + 2 * (cw + gap), y + 8, cw, 151, "933 s", "Fast-run cold step 0",
                "Plus 96 s at step 1. Estimated payback vs Fortran: about 580 steps / 12 days.",
                ORANGE, 30)
        banner(p, MARGIN, y + 175, W - 2 * MARGIN, 60,
               "Bottom line: a credible speed result exists, but only with a specially flagged configuration. The code is merged (full-fidelity-port, 1996ef1) with every speed flag off by default. Short runs lose to compilation; the flagged path still needs qualification before release.",
               AMBER, 12)
        p.text(MARGIN, y + 252, W - 2 * MARGIN, "Keep these three labels distinct", 13, INK, True)
        p.text(MARGIN, y + 277, W - 2 * MARGIN,
               "RELEASE = native-radiation JAX, near steady-state parity.  FAST CANDIDATE = 4.4x steady-state; code merged, flags off, configuration unqualified.  QUALIFICATION = fast-path tests, repeat benchmark, fidelity and longer runs still open.",
               11, INK2, maxh=54)
        footer(p, n, "RELEASE_NATIVE_RADIATION_20261012; SPEED_STEP_TIME_LOG_20261012 s13-14; SPEED_NEXT_STEPS_20261013")
        return p

    def m2(n):
        p, y = _page(
            "How the project got here",
            "A deliberate progression: establish a real Fortran reference, couple the physics, then attack the measured bottleneck.",
            "01 | JOURNEY",
            24,
        )
        xs = [MARGIN, MARGIN + 222, MARGIN + 444, MARGIN + 666]
        widths = [204, 204, 204, 222]
        stages = [
            ("1  TRUST THE REFERENCE", "Rebuilt and replayed real Fortran runs; ported components against recorded calls.", BLUE),
            ("2  COUPLE THE MODEL", "Assembled a multi-day JAX-driven hybrid; Fortran RADIA/SOCRATES remained live.", BLUE),
            ("3  PORT RADIATION", "JAX SOCRATES/RADIA matched same-packet Fortran outputs at rounding level; coupled native run completed.", GREEN),
            ("4  OPTIMIZE", "Measured and removed host/dispatch bottlenecks; one configuration reached 0.524 s/step.", GREEN),
        ]
        yy = y + 24
        for i, (title, desc, color) in enumerate(stages):
            x = xs[i]
            w = widths[i]
            card(p, x, yy, w, 185, color)
            p.text(x + 13, yy + 15, w - 26, title, 11, color, True, maxh=34)
            p.text(x + 13, yy + 61, w - 26, desc, 10.5, INK2, maxh=110)
            if i < 3:
                p.text(x + w + 1, yy + 82, 18, ">", 17, MUTED, True)
        banner(p, MARGIN, yy + 204, W - 2 * MARGIN, 60,
               "Progress is real, but maturity differs: the released native-radiation build is near parity; the larger speedup belongs to a flagged configuration that is merged but off by default.",
               AMBER, 11.5)
        footer(p, n, "PROJECT_CHRONOLOGY; RELEASE_NATIVE_RADIATION_20261012; SOCRATES_R2_DUMPS s12; SPEED_STEP_TIME_LOG_20261012")
        return p

    def m3(n):
        p, y = _page(
            "What the evidence says—and does not say",
            "The speed number is not the only gate: fidelity, reproducibility and run length determine whether it is useful.",
            "02 | EVIDENCE & LIMITS",
            23,
        )
        gap = 16
        cw = (W - 2 * MARGIN - gap) / 2
        _section_card(p, MARGIN, y + 8, cw, 296, "SUPPORTED BY MEASUREMENTS", [
            "Native JAX radiation is wired into the coupled run; it reduced median radiation-step wall time from 10.0 s to 1.9 s.",
            "Optimized matched benchmark: same exclusive V100 node, Fortran x2 then JAX, same 108-step window; 2.285 vs 0.524 s/step after startup.",
            "Two-day fast run completed, finite at each step: day-1 worst ratio 2.47 (baseline 2.37), day-2 worst 1.11.",
            "Narrow reproducibility check: two V100 runs of the recorded-radiation hybrid replay step at one code version, on different nodes, gave bitwise-identical end states.",
            "Release native configuration independently measured on two V100 nodes: near parity with Fortran.",
        ], GREEN, 10.3)
        _section_card(p, MARGIN + cw + gap, y + 8, cw, 296, "NOT YET ESTABLISHED", [
            "The strict accuracy test is still NOT MET: at step 3 the cloud-ice field (QCI) differs from the reference by 2.47x the spread between real ensemble members (limit 2x; unmodified branch 2.37x). This is a ratio of difference to spread, not 2.47x more ice.",
            "Regression with the nine speed flags set (CPU, 14 Oct): 3,329 passed, 1 known order-dependent failure, 16 skipped, the same counts as flags off. No test names the flags, so flagged code ran only where tests reach those stages.",
            "Repeatability: three runs of the identical native 54-step command at one commit (two nodes) differ pairwise in raw radiation fluxes by up to 543 W/m2; a fourth with NATIVE_NRAD=60 differs alike. Cause not yet localized.",
            "No repeat of the exact matched fast benchmark; 0.434 s is a separate profile-wrapper run, not a second matched run.",
            "No long free run on the fast flags. The 30-day shadow run used the non-fast configuration and had invalid model states after step 1288.",
        ], ORANGE, 10.3)
        banner(p, MARGIN, y + 320, W - 2 * MARGIN, 64,
               "Scope: nov26 start only; five recorded step-0 inputs; V100 / float64. The fast configuration needs an extra Python environment (Numba) for one stage. Do not generalize to other seasons, GPUs or production runs yet.",
               AMBER, 10.7)
        footer(p, n, "SPEED_STEP_TIME_LOG_20261012 s13-14; ACCEPTANCE_CRITERIA s9-10; SOCRATES_TRIAL_RUN_RECIPE s10; SPEED_RUN_RECIPE_20261013")
        return p

    def m4(n):
        p, y = _page(
            "Management input needed to proceed usefully",
            "The next work should answer the decisions that determine whether this speedup matters and what “good enough” means.",
            "03 | DECISIONS",
            22,
        )
        gap = 15
        cw = (W - 2 * MARGIN - gap) / 2
        _section_card(p, MARGIN, y + 8, cw, 165, "1  DEFINE THE TARGET USE", [
            "What is a representative production run length and restart cadence?",
            "Is a ~12-day estimated payback useful for the intended workload, or is cold-start latency a blocker?",
            "Is V100 the deployment target, or should qualification include another device?",
        ], BLUE, 10.5)
        _section_card(p, MARGIN + cw + gap, y + 8, cw, 165, "2  SET THE SCIENCE BAR", [
            "Keep the original strict 2x rule for release, or explicitly approve a separate experimental criterion?",
            "Who owns the scientific disposition of the QCI step-3 outlier and the branch’s 2.37 -> 2.47 shift?",
            "Which additional dates / run duration are necessary for a useful go/no-go?",
        ], ORANGE, 10.5)
        _section_card(p, MARGIN, y + 185, cw, 148, "3  AUTHORIZE A BOUNDED QUALIFICATION", [
            "Reserve exclusive V100 runs for repeat benchmark and fast-path regression.",
            "Decide whether to support the Numba worker dependency in a maintained environment.",
        ], GREEN, 10.5)
        _section_card(p, MARGIN + cw + gap, y + 185, cw, 148, "4  CONTROL THE RELEASE DECISION", [
            "Keep the tagged near-parity release as the stable baseline.",
            "The code is already merged with flags off; turn the fast flags on by default only after agreed test, repeatability, fidelity and long-run gates pass.",
        ], BLUE, 10.5)
        banner(p, MARGIN, y + 343, W - 2 * MARGIN, 50,
               "Recommended posture: fund a time-boxed qualification, not a production claim. Preserve the release; promote the fast path only against explicit gates.",
               AMBER, 11)
        footer(p, n, "Owner input requested; SPEED_NEXT_STEPS_20261013; RELEASE_NATIVE_RADIATION_20261012")
        return p

    def m5(n):
        p, y = _page(
            "Proposed gates: turn a promising result into a decision",
            "Small, ordered checks; stop or re-scope if the evidence does not support deployment.",
            "04 | NEXT STEPS",
            23,
        )
        rows = [
            ("A", "REPRODUCE", "Repeat the locked matched benchmark on exclusive V100 nodes; publish run-to-run spread and amortized time for the agreed workload.", BLUE),
            ("B", "QUALIFY THE CONFIG", "Regression with the speed flags set is done (baseline counts); next: flag-specific tests in the suite, and package the Numba worker dependency.", BLUE),
            ("C", "CHECK FIDELITY", "Score the fast path against the accepted rule on agreed starts; preserve strict and any separately authorized experimental status.", ORANGE),
            ("D", "TEST LONGER RUNS", "Run the fast configuration beyond two days; investigate `qdynam_do_z_extra` before claiming the 30-day trajectory is valid.", ORANGE),
            ("E", "PROMOTE OR HOLD", "Compare evidence to the target workload and science bar; only then decide flag defaults, release and operational support.", GREEN),
        ]
        yy = y + 12
        for key, title, desc, color in rows:
            card(p, MARGIN, yy, W - 2 * MARGIN, 57, color)
            p.text(MARGIN + 14, yy + 10, 34, key, 20, color, True, maxh=30)
            p.text(MARGIN + 56, yy + 8, 156, title, 10.5, INK, True, maxh=30)
            p.text(MARGIN + 214, yy + 8, W - 2 * MARGIN - 232, desc, 10, INK2, maxh=44)
            yy += 67
        banner(p, MARGIN, yy + 1, W - 2 * MARGIN, 49,
               "No schedule is implied here: the owner should set the run count, dates, and acceptance limits after choosing the target workload.",
               AMBER, 10.5)
        footer(p, n, "SPEED_NEXT_STEPS_20261013; SPEED_STEP_TIME_LOG_20261012 s13; SOCRATES_TRIAL_RUN_RECIPE s10")
        return p

    return [m1, m2, m3, m4, m5]


# ---------------------------------------------------------------------------
# Technical peer deck

def technical_slides():
    def t1(n):
        p, y = _page(
            "ROCKE-3D / JAX: benchmark state at a glance",
            "Steady-state speed is now compelling on the fast candidate; qualification, startup and fidelity remain the constraints.",
            "TECHNICAL PEERS | 14 OCT 2026",
            23,
        )
        # Proportional bars are normalized to the Fortran value.
        left, top, chart_w = MARGIN + 175, y + 34, 600
        p.text(MARGIN, top + 5, 150, "Fortran, 1 core", 11, INK2, True)
        p.rect(left, top, chart_w, 31, CARD, 6)
        p.rect(left, top, chart_w, 31, BLUE, 6)
        p.text(left + chart_w + 12, top + 7, 110, "2.285 s/step", 11, INK, True)
        p.text(MARGIN, top + 64, 150, "JAX release", 11, INK2, True)
        p.rect(left, top + 59, chart_w, 31, CARD, 6)
        p.rect(left, top + 59, chart_w * 0.95, 31, MUTED, 6)
        p.text(left + chart_w * 0.95 + 12, top + 66, 110, "2.13-2.21 s", 11, INK, True)
        p.text(MARGIN, top + 123, 150, "JAX fast configuration", 11, INK2, True)
        p.rect(left, top + 118, chart_w, 31, CARD, 6)
        p.rect(left, top + 118, chart_w * (0.524 / 2.285), 31, GREEN, 6)
        p.text(left + chart_w * (0.524 / 2.285) + 12, top + 125, 110, "0.524 s", 11, INK, True)
        p.text(left, top + 160, chart_w, "Bars show steady-state mean; release range is two V100 nodes; fast configuration is one matched run.", 8.8, MUTED)
        y2 = top + 198
        _metric(p, MARGIN, y2, 274, 111, "4.4x", "Fortran / fast JAX", "2.285 / 0.524; steps 2-107.", GREEN, 26)
        _metric(p, MARGIN + 291, y2, 274, 111, "933 + 96 s", "Fast JAX startup", "Step 0 then step 1; compile-heavy.", ORANGE, 22)
        _metric(p, MARGIN + 582, y2, 274, 111, "~580 steps", "Estimated payback", "About 12 days at 48 steps/day; arithmetic, not measured.", BLUE, 22)
        banner(p, MARGIN, y2 + 127, W - 2 * MARGIN, 53,
               "Claim boundary: 4.4x applies only to the optimized fast-configuration steady steps, not to short-run wall time, the release configuration, other dates or a validated long free run.",
               AMBER, 10.7)
        footer(p, n, "RELEASE_NATIVE_RADIATION_20261012; SPEED_STEP_TIME_LOG_20261012 s13-14")
        return p

    def t2(n):
        p, y = _page(
            "How the current coupled step is assembled",
            "The production path is a mixed host/device program; fast-path flags change selected stages without changing defaults.",
            "01 | COMPUTATION PATH",
            22,
        )
        stages = [
            ("STATE + SETUP", "Restart / carried state\nHost providers", BLUE),
            ("ATMOSPHERE", "JAX dynamics\nCONDSE / convection", GREEN),
            ("SURFACE + LAND", "JAX tiles, PBL, GHY\nEnt and daily updates", GREEN),
            ("OCEAN + ICE", "JAX ocean / sea ice\nhost pole / callbacks", GREEN),
            ("RADIATION", "native JAX SOCRATES\ninside coupled step", GREEN),
        ]
        gap, x0, ww = 13, MARGIN, 166
        yy = y + 42
        for i, (title, desc, color) in enumerate(stages):
            x = x0 + i * (ww + gap)
            card(p, x, yy, ww, 144, color)
            p.text(x + 12, yy + 14, ww - 24, title, 10, color, True, maxh=28)
            p.text(x + 12, yy + 50, ww - 24, desc, 10.5, INK2, maxh=70)
            if i < len(stages) - 1:
                p.text(x + ww + 1, yy + 62, gap, ">", 12, MUTED, True)
        _section_card(p, MARGIN, yy + 164, 429, 159, "RELEASE CONFIGURATION", [
            "All behavior flags off by default.",
            "Native JAX radiation integrated; standalone/input-removal flags are opt-in.",
            "Measured steady-state parity on two V100 nodes (2.13-2.21 vs 2.24-2.28 s/step).",
        ], BLUE, 10.2)
        _section_card(p, MARGIN + 445, yy + 164, 443, 159, "FAST CONFIGURATION (MERGED, FLAGS OFF)", [
            "Multiple opt-in flags; cloud-base/MSTCNV runs in persistent Numba worker; ENT_BATCH, pole, dynamics, host I/O tuned.",
            "Exact matched run: Fortran x2 then JAX on the same exclusive V100; 108-step window.",
            "Worker uses a separate Numba-capable Python environment; deployment packaging is unresolved.",
        ], GREEN, 10.2)
        banner(p, MARGIN, yy + 336, W - 2 * MARGIN, 50,
               "Input boundary: same nov26 initial state plus five step-0 Fortran dumps (~278 MB); no per-step records in the five-day standalone-capable run. This is not a restart-only cold launch.",
               AMBER, 10.4)
        footer(p, n, "JAX_COVERAGE_MATRIX; RELEASE_NATIVE_RADIATION_20261012; STANDALONE_INPUT_AUDIT s9-10; SPEED_RUN_RECIPE_20261013")
        return p

    def t3(n):
        p, y = _page(
            "Benchmark arithmetic and comparison scope",
            "Separate steady-state throughput from startup. Compare only equivalent windows and report the hybrid boundary.",
            "02 | MEASUREMENT",
            22,
        )
        _section_card(p, MARGIN, y + 12, 430, 191, "MATCHED FAST BENCHMARK (JOB 37941875)", [
            "Exclusive V100 gpu012, Xeon Gold 6248; Fortran twice, 1 core; then JAX `--rad native`.",
            "Same nov26 initial state and 108-step window; JAX driver includes per-step state dump.",
            "Fortran main loop: 246.7 / 246.6 s; mean 2.285 s/step.",
            "JAX steps 2-107: 55.6 s; mean 0.524 s/step, median 0.456 s.",
        ], BLUE, 10.2)
        _section_card(p, MARGIN + 446, y + 12, 442, 191, "WHAT THE 4.4x DOES NOT INCLUDE", [
            "Step 0: 933 s; step 1: 96 s. JAX total over 108 steps ~1,085 s vs Fortran ~247 s.",
            "Steady savings: 2.285 - 0.524 = 1.761 s/step. Estimated crossover ~580 steps (~12 days); assumes constant steady cost and one cold compile.",
            "Per-step mean includes radiation cadence and boundary work from steps 2-107; median is not the mean.",
            "No persistent XLA cache in these runs (D175). Radiation only: a cached repeat of the same 30-day segment compiled in 30 s vs 273 s (job 37941836); not an end-to-end cold-start figure.",
        ], ORANGE, 10.2)
        banner(p, MARGIN, y + 220, W - 2 * MARGIN, 63,
               "Replication caveat: one exact matched fast run. The separate 0.434 s mean / profile-wrapper run is corroborating but not a second matched benchmark. Release timing was repeated on two nodes and shows near parity.",
               AMBER, 10.6)
        p.text(MARGIN, y + 301, W - 2 * MARGIN, "Terminology to use", 12, INK, True)
        p.text(MARGIN, y + 326, W - 2 * MARGIN,
               "“4.4x steady-state on one optimized V100 configuration” is supported. “4.4x faster model run” is not: startup dominates this 108-step run, and the fast-path long-run workload is untested.",
               10.8, INK2, maxh=52)
        footer(p, n, "SPEED_STEP_TIME_LOG_20261012 s13; SPEED_NEXT_STEPS_20261013; RELEASE_NATIVE_RADIATION_20261012")
        return p

    def t4(n):
        p, y = _page(
            "Why it got faster: remove the measured bottlenecks",
            "The fast configuration is a bundle of changes, not one JAX kernel optimization.",
            "03 | OPTIMIZATION",
            22,
        )
        rows = [
            ["Stage / change", "What changed", "Evidence / caveat"],
            ["Convection", "MSTCNV moved to host Numba worker; Intel libimf math; batched host/device fields.", "57/57 fields match NumPy-IMF on captured columns; worker env and startup cost."],
            ["Dynamics + ocean", "Q-dynamics vmap, pole buffers/async, sequential sum blocks; avoid slower scan unroll.", "Targeted checks; flags are opt-in."],
            ["Land / Ent", "Batch leaf evaluations across qsimp levels and sun/shade cohorts.", "Ent call 5.39 -> 2.37 ms; GHY piece 0.0534 -> 0.0296 s; tiny FP-rounding differences."],
            ["Host glue / transfers", "JIT template-state glue; one batched device_get/put for required MSTCNV outputs.", "MSTCNV stage 0.087 -> 0.071 s."],
            ["Native radiation", "SOCRATES in coupled step; about 0.16-0.17 s per call in isolated V100 test.", "Matched native integration released near parity; no longer dominant cost."],
        ]
        p.table(MARGIN, y + 12, [160, 379, 313], rows, size=9.2, maxh=293, pad=6, bold_cols=(0,))
        banner(p, MARGIN, y + 322, W - 2 * MARGIN, 59,
               "Residual fast-step stages (profile run): surface 0.086 s, MSTCNV 0.071 s, dynamics 0.065 s, ocean_b 0.048 s; plus ~0.16 s on radiation steps. These are measured profiles, not a full attribution of every benchmark second.",
               BLUE, 10.4)
        footer(p, n, "SPEED_STEP_TIME_LOG_20261012 s11-13; SOCRATES_R2_DUMPS s12")
        return p

    def t5(n):
        p, y = _page(
            "Fidelity, standalone behavior and test status",
            "Each statement has a scope; do not let a relaxed criterion or successful run imply broader validity.",
            "04 | VALIDATION",
            22,
        )
        _section_card(p, MARGIN, y + 7, 430, 190, "FIDELITY AGAINST REAL-MEMBER SPREAD", [
            "Fast run: day-1 QCI (cloud ice) at step 3 = 2.47x the reference member spread (difference relative to spread, not 2.47x more ice); baseline = 2.37x. Strict 2x day criterion: NOT MET.",
            "Day 2 worst ratio = 1.11 in profile run; relaxed rule (3x for steps >=3) met.",
            "Owner accepted the ~0.1 ratio shift for the speed experiment; this is not approval of strict fidelity.",
        ], ORANGE, 10.1)
        _section_card(p, MARGIN + 446, y + 7, 442, 190, "LONG-RUN EVIDENCE IS NOT FAST-PATH EVIDENCE", [
            "Release/native non-fast configuration completed a 5-day run without per-step records.",
            "30-day shadow run: QDYNAM extra-column branch made states invalid after step 1288 (~26.8 days); 30 days not demonstrated.",
            "The fast flags have only been exercised for 108 steps (2 days).",
        ], BLUE, 10.1)
        _section_card(p, MARGIN, y + 213, 430, 162, "TESTS", [
            "Full CPU regression, flags off and with the nine speed flags set: 3,329 passed, 1 known order-dependent failure, 16 skipped in both. No test in tests/ names the flags; flagged code runs only where tests reach those stages.",
            "Native 54-step V100 runs, flags off (HEAD 061486d): three identical-command repeats (two on one node, one on another) differ pairwise in raw FSF by 506-543 W/m2; a fourth run with NATIVE_NRAD=60 differs alike (467-521); zenith bitwise equal. Cause not localized.",
        ], BLUE, 10.1)
        _section_card(p, MARGIN + 446, y + 213, 442, 162, "BOUNDARY / REPRODUCIBILITY", [
            "nov26 only; five step-0 recorded dumps; irrigation assumptions as documented.",
            "No other start date, fast-path multi-week run, second exact benchmark, or packaged worker environment yet.",
        ], AMBER, 10.1)
        footer(p, n, "SPEED_STEP_TIME_LOG_20261012 s13-14; ACCEPTANCE_CRITERIA s9-10; STANDALONE_INPUT_AUDIT s10; SOCRATES_TRIAL_RUN_RECIPE s10")
        return p

    def t6(n):
        p, y = _page(
            "Technical next steps and go / no-go gates",
            "Preserve the release baseline; qualify the fast candidate with evidence matched to the claim.",
            "05 | PLAN",
            22,
        )
        steps = [
            ("1", "REPEAT PERFORMANCE", "Same exclusive-node recipe, exact fast flags, same workload. Publish spread, total wall and amortization.", BLUE),
            ("2", "RUN THE FAST TEST MATRIX", "Regression with flags set passed at baseline counts; add flag-specific tests; capture environment and worker version; localize the native-radiation run-to-run difference.", BLUE),
            ("3", "QUALIFY SCIENCE", "Score against the agreed strict rule; keep any experimental relaxation separate and dated; assess the QCI step-3 shift.", ORANGE),
            ("4", "EXTEND THE RUN", "Run fast flags beyond 2 days; debug qdynam_do_z_extra and assess states after that branch before claiming longer free runs.", ORANGE),
            ("5", "PROMOTE ONLY IF GATES PASS", "Then decide flag defaults and release; otherwise keep the flags off (code already merged as 1996ef1) and document the fast path as experimental.", GREEN),
        ]
        yy = y + 9
        for num, title, desc, col in steps:
            card(p, MARGIN, yy, W - 2 * MARGIN, 57, col)
            p.text(MARGIN + 14, yy + 12, 28, num, 18, col, True, maxh=24)
            p.text(MARGIN + 53, yy + 10, 206, title, 10.2, INK, True, maxh=31)
            p.text(MARGIN + 262, yy + 8, W - 2 * MARGIN - 278, desc, 9.7, INK2, maxh=43)
            yy += 66
        banner(p, MARGIN, yy + 5, W - 2 * MARGIN, 50,
               "Management dependencies: target run length / cold-start tolerance; owner-approved fidelity bar and extra dates; V100 allocation; Numba worker packaging; merge authority.",
               AMBER, 10.5)
        footer(p, n, "SPEED_NEXT_STEPS_20261013; SPEED_RUN_RECIPE_20261013; ACCEPTANCE_CRITERIA")
        return p

    return [t1, t2, t3, t4, t5, t6]


def main():
    _save_deck(
        os.path.join(HERE, "management_briefing_20261014.pdf"),
        "ROCKE-3D to JAX: management briefing, 2026-10-14",
        management_slides(),
        os.path.join(HERE, "preview_management_20261014"),
    )
    _save_deck(
        os.path.join(HERE, "technical_briefing_20261014.pdf"),
        "ROCKE-3D to JAX: technical briefing, 2026-10-14",
        technical_slides(),
        os.path.join(HERE, "preview_technical_20261014"),
    )


if __name__ == "__main__":
    main()
