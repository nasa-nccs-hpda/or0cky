"""
Build status_deck.pdf (2026-10-08 refresh: chronology and findings of the
whole project). Same pipeline as build_pdf_2026-10-01.py (reportlab, built-in
Helvetica/Courier, navy/green/orange palette); layout goes through
deck_layout.py so overflow is checked at build time and PNG previews exist.

Usage (system python3 has reportlab; the conda env has matplotlib):
  conda-python make_figures.py images_out     # figures -> images/
  python3 build_pdf.py status_deck.pdf [--png PREVIEW_DIR]
Every number on a slide comes from the repository documents named in the
slide's footer (FULL_FIDELITY_DELTAS.md ledger ids D1..D198, Reports/*.md,
git log). Status: ledger through D198, HEAD ec36779, 2026-10-08.
"""
import os, sys
from reportlab.pdfgen import canvas
from deck_layout import *
import deck_layout as L

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
SLIDES = []


def slide(fn):
    SLIDES.append(fn)
    return fn


def P(name):
    return Page(name)


# ------------------------------------------------------------------ 1
@slide
def s_glance(n):
    p = P("glance")
    y = header(p, "ROCKE-3D -> JAX | AT A GLANCE | status 2026-10-09 (ledger through D213)",
               "Where the project stands",
               "Goal: rewrite NASA GISS's ROCKE-3D (Fortran GCM) in Python/JAX, checked against the real Fortran, for GPU use. "
               "Configuration P2SAoM40 (72x46 grid, 40 layers, dynamic ocean). Every card says what is met and what is not.", 24)
    cards = [
        (ORANGE, "NOT MET", "One-day criterion (nov26, 54 steps)",
         "ACCEPTANCE s9: never beyond 2x the largest real-member distance at every step. QCL at step 1 is 2.84x (one bistable cell, D197). Steps >= 3: worst ratio 1.69 (QCI, inside the run-to-run spread). Two days run. D197, D207, D213."),
        (AMBER, "MET / PARTLY MET", "One coupled step vs real Fortran (C2, step 0)",
         "nov26 MET with one named column (25,16); dec01 and jan01 PARTLY MET (worst EGCM 9.9e-9 and 7.6e-9 of scale). Needs the libimf callback. D191."),
        (GREEN, "18.8 s", "Steady step on CPU, was 44.3 s",
         "99 jit executions (was 262), eager dispatches 490 (was ~15,700). Only 1.2-1.4x faster than the NumPy chain on the same cores. HYBRID step. D187, D191."),
        (GREEN, "558 / 558", "C1: bitwise vs our NumPy chain",
         "Step 0 on nov26, dec01, jan01 (category A). On the day: bitwise through step 47 with the computed GHY schedule; the NumPy chain has no daily_LAKE. D191, D204, D206."),
        (GREEN, "3,056 passed", "Last batch regression (a89e117)",
         "6 skipped, 1 known failure (export artifact: test asserts a git HEAD; fixed afterwards, not re-run). README_START_HERE handoff."),
        (ORANGE, "No GPU result", "GPU runner built, validated on CPU only",
         "A100 probe: launch-bound for small sequential ops; device exp/sin/pow differ from NumPy by 1 ulp in 6-12% of values, so never bitwise on GPU. D198, gpu/DISCOVER_RUN.md."),
    ]
    cw, ch = (W - 2 * MARGIN - 2 * 12) / 3, 138
    for i, (col, big, lab, note) in enumerate(cards):
        x = MARGIN + (i % 3) * (cw + 12)
        yy = y + (i // 3) * (ch + 12)
        card(p, x, yy, cw, ch, col)
        p.text(x + 14, yy + 10, cw - 24, big, 19, INK, True, True, maxh=26)
        p.text(x + 14, yy + 38, cw - 24, lab, 9.5, INK2, True, maxh=26)
        p.text(x + 14, yy + 62, cw - 24, note, 9.2, MUTED, maxh=ch - 66)
    yb = y + 2 * (ch + 12) + 2
    banner(p, MARGIN, yb, W - 2 * MARGIN, 52,
           "This is a HYBRID port, not end-to-end JAX: radiation is the original Fortran (served) or replayed from a record; libimf math, QUS, pole columns and the OADVT2 pre-pass are host callbacks; "
           "Ent exports, land forcing and tile radiation columns are recorded inputs (D191 s4). SOCRATES is never ported or modified.", AMBER, 9)
    footer(p, n, "ACCEPTANCE_CRITERIA s1,3,4,8,9; D187, D191, D193-D198 (FULL_FIDELITY_DELTAS.md); Reports/README_START_HERE.md; gpu/DISCOVER_RUN.md")
    return p


# ------------------------------------------------------------------ 1b
RED_, VIOLET = "#EF4444", "#A78BFA"


def _dot(p, x, y, color, d=14):
    p.rect(x, y, d, d, color, r=d / 2)


@slide
def s_stoplight(n):
    p = P("stoplight")
    y = header(p, "OVERVIEW | PROJECT STATUS AT A GLANCE", "Status by area, and what is still outstanding",
               "Green = verified success, yellow = in progress or partial, red = broken or not met, grey = not started. Violet marks outstanding items and their next action.", 22)
    colw = (W - 2 * MARGIN - 16) / 2
    # left: status lights
    card(p, MARGIN, y, colw, 352, GREEN)
    p.text(MARGIN + 14, y + 8, colw - 24, "Status by area (2026-10-08)", 11, INK, True)
    rows = [
        (GREEN, "Component ports vs the Fortran (Track B)", "validated rung by rung, D4-D145"),
        (GREEN, "One coupled step vs the real Fortran (C2)", "step 0 MET on nov26, dec01, jan01 (D199); hybrid"),
        (GREEN, "JAX step equals the NumPy chain (C1)", "bitwise: 561/561 at step 0; nov26 steps 0-47 (D206)"),
        (GREEN, "Regression suite", "3,611 passed, 6 skipped; 12 day-2 test cases excluded, fixed file 219 passed (15baa26)"),
        (AMBER, "Device-resident assembled step", "99 jit executions, 18.8 s/step; hybrid, not end-to-end JAX"),
        (AMBER, "Radiation (SOCRATES)", "Fortran server callback works (D210/D212); not ported"),
        (RED_, "GPU assembled step", "runs on the A100 (step 0 OK) but NaN from step 1 (surface stage); cause unknown"),
        (GREEN, "Second model day (steps 54-107)", "real records made (D207); day 2 never beyond 2x of the members"),
        (RED_, "One model day vs 5 real members (ACCEPTANCE s9)", "NOT MET: QCL step 1 = 2.83x in every run (systematic); rest inside the spread (D213)"),
        (GREEN, "Daily lake update at the day boundary", "ported, bitwise vs the compiled Fortran (D205, D208)"),
        (MUTED, "Months-long run and its criterion", "not started; criterion not defined"),
    ]
    yy = y + 32
    for col, title, note in rows:
        _dot(p, MARGIN + 16, yy + 3, col)
        p.text(MARGIN + 38, yy, colw - 50, title, 9.6, INK, True, maxh=14)
        p.text(MARGIN + 38, yy + 12, colw - 50, note, 8.4, MUTED, maxh=14)
        yy += 31
    # right: outstanding items in the complementary colour
    x2 = MARGIN + colw + 16
    card(p, x2, y, colw, 352, VIOLET)
    p.text(x2 + 14, y + 8, colw - 24, "Outstanding items and their next action", 11, VIOLET, True)
    items = [
        ("GPU NaN at step 1 (surface stage)", "owner: JAX_DEBUG_NANS=1 run on Discover; CPU control with the simplifier on is finite"),
        ("Replayed inputs in the step", "Ent exports, land forcing, tile radiation columns, PBL templates (D212 list); tile radiation columns next"),
        ("QCL step 1 exceedance (2.84x)", "one bistable cell (D197); owner may relax s9 later (dated, both statuses)"),
        ("Free-running comparison", "compute Ent and land forcing instead of replaying the real records"),
        ("Radiation and SOCRATES", "135-295 h port estimate; only if the owner lifts the rule (D1-D6)"),
        ("Multi-month criterion", "define it; use the 8-member JAN1950 leave-one-out scoring"),
        ("NumPy reference beyond day 1", "give the NumPy chain a daily_LAKE so C1 holds across the boundary"),
        ("Housekeeping", "owner: close extra sessions; decide where ANN4099 lives; push when batches pass"),
    ]
    yy = y + 32
    for title, note in items:
        p.rect(x2 + 16, yy + 2, 4, 26, VIOLET)
        p.text(x2 + 30, yy, colw - 44, title, 9.6, INK, True, maxh=14)
        p.text(x2 + 30, yy + 12, colw - 44, note, 8.4, MUTED, maxh=24)
        yy += 39
    banner(p, MARGIN, y + 362, W - 2 * MARGIN, 36,
           "Honest status: step 0 matches the real Fortran on three dates and two model days run, but the step is a hybrid with replayed inputs and the one-day statistical test is not passed (QCL step 1).", AMBER, 9.4)
    footer(p, n, "README_START_HERE.md; ACCEPTANCE_CRITERIA.md s9; D187, D191, D199, D203-D213; SOCRATES_PORT_PLAN.md; gpu/DISCOVER_RUN.md")
    return p


# ------------------------------------------------------------------ 2
@slide
def s_chron(n):
    p = P("chronology")
    y = header(p, "CHRONOLOGY | WHOLE PROJECT", "Six weeks, 273 commits: from reduced ports to a hybrid JAX coupled step", None, 20)
    p.image(MARGIN - 6, y - 2, W - 2 * MARGIN + 12, 420, os.path.join(IMG, "chronology_full.png"))
    p.text(MARGIN, 486, W - 2 * MARGIN, "Most work happened after 2026-09-24, when the owner reversed Track A's recommendation and set the full-fidelity direction. Rows and sources: Reports/PROJECT_CHRONOLOGY.md (generated from status_slides/chronology_data.py).", 8.5, MUTED, maxh=24)
    footer(p, n, "git log (dates, commit ids); Project_Summary s1; ACCEPTANCE s8-9; Reports/PROJECT_CHRONOLOGY.md")
    return p


@slide
def s_zoom(n):
    p = P("zoom")
    y = header(p, "CHRONOLOGY | THE LAST 48 HOURS", "From the F1 gate to the one-day verdict: 2026-10-06 to 2026-10-08", None, 20)
    p.image(MARGIN - 6, y - 4, W - 2 * MARGIN + 12, 442, os.path.join(IMG, "chronology_zoom.png"))
    footer(p, n, "git log; ledger D118-D198 (FULL_FIDELITY_DELTAS.md); ACCEPTANCE s8-9. Owner decisions: 10-07 13:39 / 14:19 / 14:25 and 10-08 08:20.")
    return p


# ------------------------------------------------------------------ 4
@slide
def s_phases(n):
    p = P("phases")
    y = header(p, "CHRONOLOGY | PHASES AND WHAT EACH TAUGHT", "Seven phases, seven lessons", None, 22)
    rows = [["Phase", "Dates", "Key commits / ledger", "What was learned"],
            ["A Track A: reduced component ports, GPU kernels", "08-28 to 09-24", "c2991dd, 5cbe7f8, abd805b", "Reduced drivers ran fast (kernels 2.3-6.3x on GPU; fused chain 4.2x) but were not faithful; Track A recommended against a full port, later reversed by the owner."],
            ["B Phase 0: Fortran oracle", "09-24", "97e2457, 9cd4bc6, 4ec6056 (D3)", "Instrumented real ModelE reproduces the restart bitwise. Chaos floor: 1-ulp perturbation reaches ~1-3% of variability in 5 days, so acceptance beyond one step must be statistical."],
            ["C Track B: components vs real Fortran", "09-24 to 10-06", "D4-D145; D29, D36, D51, D55, D121-D123", "Dump-hook-and-validate found real bugs (3 in DYNSI, OMEGA constant, 2 land errors D135/D136) and dead code (OCNDYN.f legacy driver, OCNTDMIX.f, OCNQUS.f)."],
            ["D Validation rungs F1 / F2 / F3", "10-06 to 10-07", "D127-D129, D149-D157, D163, D165", "Single-step gate needs libimf (3-5% of cloud columns flip without it). One-day runs sit inside the real model's own spread except condensate tails. Radiation can be served by real RADIA (D152-D162)."],
            ["E Reconciliation + owner decisions", "10-07", "9cf08e6, 721d789, b71eb22, bd9b16c", "Fidelity work was ahead of the JAX deliverable; criteria were fixed BEFORE the coupled step was compared; C1 and C2 reported separately."],
            ["F Hybrid coupled step, stages S0-S8", "10-07 to 10-08", "D180-D197", "Assembled device-resident step bitwise to NumPy (C1), C2 at step 0 MET/PARTLY MET; the 54-step day is NOT MET; every day diagnosis found a cause, none a port defect in clouds."],
            ["G GPU tooling", "10-07 to 10-08", "5c5b7a4, c7c42c8, d320221, ec36779 (D198)", "Environment works on an A100; small sequential ops are launch-bound; no bitwise on GPU; no GPU run of the step yet."]]
    p.table(MARGIN, y, [190, 80, 160, W - 2 * MARGIN - 430], rows, 10, maxh=450 - y + 40, bold_cols=(0,))
    footer(p, n, "git log; Project_Summary s3-4; D3, D29, D36, D129, D135-D136, D151, D157, D191-D198; RECONCILIATION_PLAN s1; STATUS.md; gpu/DISCOVER_RUN.md")
    return p


# ------------------------------------------------------------------ 5
@slide
def s_categories(n):
    p = P("categories")
    y = header(p, "FINDINGS | HOW FIDELITY IS JUDGED", "Categories A / B / C / D, the gate, and two comparisons",
               "Per prognostic field, maximum absolute difference as a fraction of the field's scale (ACCEPTANCE s3; categories fixed by the project plan and not loosened).", 22)
    cw = (W - 2 * MARGIN - 36) / 4
    cats = [(GREEN, "A", "bitwise equal"), (GREEN, "B", "<= 1e-12 of scale (rounding level)"), (AMBER, "C", "<= 1e-6 of scale"), (ORANGE, "D", "worse than C")]
    for i, (c, k, d) in enumerate(cats):
        x = MARGIN + i * (cw + 12)
        card(p, x, y, cw, 62, c)
        p.text(x + 14, y + 8, 30, k, 26, INK, True, True)
        p.text(x + 50, y + 14, cw - 58, d, 9.5, INK2, True, maxh=40)
    y += 76
    h = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, y, h, 150, GREEN)
    p.text(MARGIN + 14, y + 8, h - 24, "Verdict for one coupled step", 11, INK, True)
    p.bullets(MARGIN + 14, y + 28, h - 26, [
        "MET: every prognostic field in A or B, except named exception columns (at most 10 per field, each <= 1e-9 of scale).",
        "PARTLY MET: runs end to end with named exceptions in B or C and the first failing stage identified.",
        "NOT MET: otherwise. Reported per date (nov26, dec01, jan01), never pooled.",
        "Bitwise (A) needs the Intel libimf runtime; without it results are B or C and say so."], 9.6, maxh=118)
    card(p, MARGIN + h + 12, y, h, 150, BLUE)
    p.text(MARGIN + h + 26, y + 8, h - 24, "C1 versus C2 (reported separately)", 11, INK, True)
    p.bullets(MARGIN + h + 26, y + 28, h - 26, [
        "C1, port consistency: JAX-driven step vs our NumPy chained step, same start state, cores, flags, no compile cache. A pass says JAX reproduces our NumPy port, NOT that it reproduces ROCKE-3D.",
        "C2, fidelity: JAX-driven step vs the real Fortran step-boundary dumps. Only C2 supports a statement about ROCKE-3D.",
        "Hazards D139, D145, D175: results differ across core counts and with a warm compile cache."], 9.6, maxh=118)
    y += 164
    card(p, MARGIN, y, W - 2 * MARGIN, 120, ORANGE)
    p.text(MARGIN + 14, y + 8, 400, "Multi-day rung (F2, statistical), ACCEPTANCE s4 and s9", 11, INK, True)
    p.bullets(MARGIN + 14, y + 28, W - 2 * MARGIN - 28, [
        "Reference spread: five real one-ulp members of the nov26 day (D151). Classes per field and step: within (<= largest member distance), near (<= 2x), beyond (> 2x).",
        "Owner decision 2026-10-08 (s9): the pass rule is 'never beyond 2x the largest member distance, at every step, for every scored field' over ALL steps; 'within at every step' is a secondary line.",
        "Longer windows: the D172 leave-one-out scoring with the 8 JAN1950 members (D165); the earlier '>= 95% of zonal bins within 2 sigma' rule is not used (true members fail it)."], 9.6, maxh=88)
    footer(p, n, "ACCEPTANCE_CRITERIA.md s2, s3, s4, s9; D139, D145, D151, D165, D172, D175")
    return p


# ------------------------------------------------------------------ 5b
@slide
def s_compare(n):
    p = P("compare")
    y = header(p, "FINDINGS | WHAT WE COMPARE", "Identical inputs: what 'the same run' means, and what is compared at each horizon",
               "Same restart, boundary data and configuration (P2SAoM40) for the Fortran and the JAX port; how exact the match can be depends on how far you run.", 22)
    G, A_, O = GREEN, AMBER, ORANGE
    rows = [["Horizon", "What is compared", "What 'match' means", "Status (2026-10-08)"],
            ["One step", "Full model state after the step: T, U, V, Q, P, cloud water and ice, moment arrays; ocean, ice, lake and land state, field by field",
             "Categories A bitwise, B <= 1e-12 of scale, C <= 1e-6, D worse; met if every gate field is A or B except named columns", ("nov26, dec01, jan01 MET at step 0 after D199 (nov26 with named column (25,16))", G)],
            ["One model day (54 steps)", "The same fields at every step, as distance from the real run",
             "Statistical: distance <= 2x the largest spread among five real one-ulp members at every step (ACCEPTANCE s9); bitwise is not expected (chaos)", ("nov26 NOT MET (D195, QCL step 1 = 2.84); rerun on the D199 land code in progress (D200)", O)],
            ["Months and longer", "Climate statistics: zonal means, variability, energy balance",
             "Not defined yet; 8-member JAN1950 ensemble and D172 leave-one-out scoring exist, no long run exists", ("not started", MUTED)]]
    hh = p.table(MARGIN, y, [110, 300, 330, W - 2 * MARGIN - 740], rows, 10, maxh=250, bold_cols=(0,))
    yy = y + hh + 12
    h = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, yy, h, 190, BLUE)
    p.text(MARGIN + 14, yy + 8, h - 24, "Identical inputs: what that includes today", 11, INK, True)
    p.bullets(MARGIN + 14, yy + 28, h - 26, [
        "Same restart state, boundary data, orbital and configuration settings, same physics code version: our tests start from the real model's own state at a step boundary.",
        "The model is chaotic: two correct runs drift apart, so beyond a few steps agreement can only be statistical.",
        "Bitwise matching needs the Intel libimf math library (host callback); on the GPU even exp/pow differ by 1 ulp in 6-12% of values."], 9.6, maxh=150)
    card(p, MARGIN + h + 12, yy, h, 190, ORANGE)
    p.text(MARGIN + h + 26, yy + 8, h - 24, "Not yet an independent same-parameters run", 11, INK, True)
    p.bullets(MARGIN + h + 26, yy + 28, h - 26, [
        "Recorded inputs: vegetation (Ent) exports, land forcing and some surface columns are taken from the real model's records each step, so the run is not free-running.",
        "Radiation is replayed from the record or called through the original Fortran; SOCRATES is not ported.",
        "The daily lake update at the day boundary is not ported (D196). A real blind comparison needs these computed, and a multi-month criterion defined."], 9.6, maxh=150)
    footer(p, n, "ACCEPTANCE_CRITERIA.md s2, s3, s4, s8, s9; D187, D191, D195, D196, D199; GPU probe in gpu/DISCOVER_RUN.md")
    return p


# ------------------------------------------------------------------ 6
@slide
def s_gates(n):
    p = P("gates")
    y = header(p, "FINDINGS | GATE VERDICTS BY DATE", "Single-step (step 0) verdicts over time: honest status per date",
               "dec01 and jan01 were PARTLY MET in every ported-land configuration until D199 fixed the land humidity (step 0, one run each); nothing here is claimed beyond what the ledger says.", 22)
    G, A_, O = GREEN, AMBER, ORANGE
    rows = [["Date / entry", "Configuration", "nov26", "dec01", "jan01", "Note"],
            ["10-06 D129", "libimf, RECORDED land patch", ("MET (named cols)", G), ("MET (A/B)", G), ("MET (named col)", G), "nov26 W2GCM 4 cols 1.7e-12; jan01 1 col <= 2.5e-12"],
            ["10-06 D129", "libimf, ported GHY", ("NOT MET", O), ("PARTLY MET", A_), ("PARTLY MET", A_), "first failing stage: SURFACE/GHY land cell (62,34)"],
            ["10-06 D129", "no libimf (libm)", ("NOT MET", O), ("NOT MET", O), ("NOT MET", O), "first failing stage CONDSE: cloud threshold flips in 3-5% of columns"],
            ["10-06 0ad795b", "libimf, ported GHY after D135/D136 fixes", ("MET", G), ("PARTLY MET", A_), ("PARTLY MET", A_), "all MET with recorded land"],
            ["10-07 D187", "HYBRID assembled (libimf callback, replay radiation)", ("MET, 1 col", G), ("PARTLY MET", A_), ("PARTLY MET", A_), "fails ACCEPTANCE items 1 and 3; 44.3 s/step"],
            ["10-08 D191", "assembled device-resident step", ("MET, 1 col (25,16)", G), ("PARTLY MET", A_), ("PARTLY MET", A_), "worst EGCM 9.9e-9 (dec01), 7.6e-9 (jan01); item 1 holds only under owner decision 8.3 with named exceptions"],
            ["10-08 D199", "assembled step after the land qg_aver/elhx fix", ("MET, 1 col (25,16)", G), ("MET (1/13/0/0)", G), ("MET (1/13/0/0)", G), "worst gate field 4.1e-13 (dec01), 6.3e-13 (jan01); C1 559/559; step 0 only, replayed radiation, libimf callback"],
            ["10-08 D195 / s9", "54-step nov26 day vs 5 real members", ("NOT MET", O), ("not run", MUTED), ("not run", MUTED), "QCL step 1 ratio 2.84; steps >= 3 worst 1.13"],
            ["10-08 D198", "libm-mode GPU runner, CPU check", ("n/a (not fidelity)", MUTED), ("-", MUTED), ("-", MUTED), "vs NumPy libm: 555 A + 3 B of 558; vs libimf CPU: A126 B170 C47 D215 (a property of libm, not of the port)"]]
    hh = p.table(MARGIN, y, [92, 190, 100, 82, 82, W - 2 * MARGIN - 546], rows, 10, maxh=360, bold_cols=(0,))
    yy = y + hh + 10
    banner(p, MARGIN, yy, W - 2 * MARGIN, 50,
           "dec01 / jan01 were PARTLY MET because of one land humidity term (D199, fixed at step 0 only; later steps and the day are being re-measured). Four surface-composite exports (QGAVG, USAVG, VSAVG, TGVAVG) were category D against the real end record in D187 and were not re-measured in D191.", AMBER, 8.8)
    footer(p, n, "D129, 0ad795b, D135-D136, D187, D191 s3, D195, D198 s2-3; ACCEPTANCE s3")
    return p


# ------------------------------------------------------------------ 7
@slide
def s_what_is_jax(n):
    p = P("jax")
    y = header(p, "FINDINGS | WHAT IS JAX AND WHAT IS NOT", "The assembled coupled step (D191) is a hybrid", None, 22)
    G, A_, O = GREEN, AMBER, ORANGE
    rows = [["Component", "How it runs in the assembled step", "Status", "Source"],
            ["Atmosphere phase 1 dynamics (J1)", "JAX; 75 kernel dispatches from Python; pow through libimf host callback", ("JAX + callback", A_), "D186, D191"],
            ["Clouds / convection", "LSCOND and MSTCNV on device (D189, fused libimf callbacks); QUS subsidence (176 calls, 249.5 MB) and 2 pole columns are host callbacks", ("JAX + callbacks", A_), "D189, D191 s4"],
            ["Radiation (SOCRATES)", "Original Fortran RADIA via persistent server (~11 s/call, every 5th step, 33.7 MB to host, 11.6 MB back) or REPLAYED from the real record. Never ported.", ("Fortran / replay", O), "D159-D162, D185, SOCRATES_PORT_PLAN"],
            ["libimf pow/exp", "Host callback into the original build's Intel runtime (the fidelity configuration); GPU/libm mode has none", ("host callback", O), "ACCEPTANCE s8.1, D183"],
            ["Surface tiles + land (GHY)", "JAX SURFACE stage on fixed-shape tiles (D188); Ent exports, land forcing and tile radiation columns are RECORDED each step", ("JAX + recorded", A_), "D188, D191 s1"],
            ["Post-tile surface + ocean", "Device-resident JAX (GROUND_*, RIVERF, DYNSI, OCEANS, FORM_SI, ADVSI); OADVT2 east-west pre-pass is a host callback", ("JAX + callback", A_), "D190, D191"],
            ["Phase 2 (DISSIP, FILTER)", "jnp, bitwise vs NumPy; pow of SLP/MAtoPMB/PEK through libimf callback", ("JAX + callback", A_), "D191 s1"],
            ["Host work every step", "template_build (eager, 294 dispatches), record load (200 calls, 138 MB), day boundary DAILY_ATMDYN in NumPy", ("host", O), "D191 s4, D195"],
            ["Not implemented", "new-ice-tile donor rule; daily_LAKE (D196); daily ocean/ice/land updates; transfer-guard test of the full step", ("missing", O), "D191 s1, D195 s6, D196"]]
    p.table(MARGIN, y, [150, 480, 95, W - 2 * MARGIN - 725], rows, 10, maxh=410, bold_cols=(0,))
    p.text(MARGIN, 484, W - 2 * MARGIN, "D191 verdict on ACCEPTANCE s1: items 2-5 hold as reporting items; item 1 holds under owner decision 8.3 (three jit units plus host calls) for the prognostic state, with named exceptions. 'I do not call it end-to-end JAX.'", 8.6, AMBER, True, maxh=24)
    footer(p, n, "D191 s1, s4 (non-JAX stage list); D183, D185, D188-D190; ACCEPTANCE s1, s8; SOCRATES_PORT_PLAN (planning only)")
    return p


# ------------------------------------------------------------------ 8
@slide
def s_perf(n):
    p = P("perf")
    y = header(p, "FINDINGS | STEP TIMINGS", "44.3 s to 18.8 s per step; 262 to 99 jit executions", None, 22)
    p.image(MARGIN - 4, y - 2, W - 2 * MARGIN + 8, 200, os.path.join(IMG, "perf_steps.png"))
    yy = y + 204
    rows = [["Entry", "Change", "Before -> after", "Note"],
            ["D182", "cold-compile pathology: XLA flags algsimp + reshape-mover disabled", "~3,000 s -> 390 s cold step 0 (commit 7131447: ~30x faster cold compile)", "results bitwise identical"],
            ["D188", "SURFACE stage in JAX on fixed-shape tiles", "~2x faster, zero eager dispatches inside", "category A on 3 dates"],
            ["D189", "MSTCNV as one device program, fused libimf callbacks", "phase 1 of step 0: 33.6 s -> 11.9 s", "bitwise equal"],
            ["D190", "post-tile surface + ocean as JAX", "123 -> 5 jit executions; 8.0 s -> 1.76 s", "149/149 category A"],
            ["D191", "assembly of J1 + J2 + J3", "steady 44.3 -> 18.8 s; cold first step 349 -> 490 s (worse)", "shared node, one run per number"],
            ["D191", "vs NumPy libimf chain, same cores", "only 1.2-1.4x faster (22.6-23.3 s)", "46% of step is MSTCNV + callbacks, 13% dynamics dispatch"],
            ["D198", "libm-mode runner on CPU", "cold 572 s, steady 11.0 s (load average ~20)", "indicative; GPU time unknown"]]
    p.table(MARGIN, yy, [42, 270, 300, W - 2 * MARGIN - 612], rows, 7.9, maxh=272, bold_cols=(0,))
    footer(p, n, "D182 (+commit 7131447), D188-D191 s5, D198 s3; times: CPU, no GPU, no compile cache, cores/loads as stated in each entry")
    return p


@slide
def s_stage_share(n):
    p = P("stageshare")
    y = header(p, "FINDINGS | WHERE THE STEP TIME GOES", "Callbacks and recorded-input work still dominate the assembled step",
               "D191 section 4, nov26, steady, timed run with a device block after every stage (step 17.8 s, stage sum 20.1 s incl. record load).", 21)
    items = [("condse_mstcnv [device + libimf/QUS callbacks]", 46.4, ORANGE), ("dyn [75 dispatches + libimf pow]", 13.1, AMBER),
             ("condse_post [LSCOND libimf, north-pole callback]", 12.5, ORANGE), ("record_load [recorded]", 11.3, ORANGE),
             ("surface_tiles_land [JAX + recorded Ent/forcing]", 3.7, AMBER), ("ocean_b [OADVT2 pre-pass callback]", 2.7, AMBER),
             ("condse_setup [south pole callback]", 1.8, AMBER), ("dissip_filter [libimf callbacks]", 1.2, AMBER),
             ("template_build [host]", 1.2, ORANGE), ("ocean_a [pure JAX]", 5.3, GREEN)]
    x0, bw = MARGIN + 300, 360
    for i, (lab, v, c) in enumerate(items):
        yy = y + 6 + i * 26
        p.text(MARGIN, yy, 296, lab, 8.6, INK2, maxh=22)
        p.rect(x0, yy + 1, max(bw * v / 50.0, 3), 14, c, 3)
        p.text(x0 + bw * v / 50.0 + 6, yy + 1, 60, "%.1f%%" % v, 8.8, INK, True, True)
    yy = y + 6 + len(items) * 26 + 8
    p.bullets(MARGIN, yy, W - 2 * MARGIN, [
        "Callback bodies (libimf_ops 5.0 s, fused 0.68 s, QUS 0.78 s, poles 0.39 s, OADVT2 0.26 s = 7.1 s) are included in these shares: about 40% of the step. Radiation: 0.04 s replayed; 12.7 s per call with the real server (1 call per 5 steps).",
        "Purely-JAX stages (post_a, post_b, surface_pre, template_apply, tile_acc, carry_writeback, melt_si_dev, state_assembly) are each <= 0.2%. The kind labels (JJ/NP/REC) were set by hand from code inspection; the shares are measured.",
        "Over the 54-step day (D195): steady median 15.9 s (min 15.1, max 18.7); 13 SURFACE rebuild steps of 68-74 s each; cold step 0 428 s; total wall 2,128 s."], 8.8, maxh=110)
    footer(p, n, "D191 s4-5; D195 s2.2 (day timings). Colors: green = JAX only, amber = JAX with callbacks/records, orange = callback-dominated or host/recorded")
    return p


# ------------------------------------------------------------------ 10
@slide
def s_configs(n):
    p = P("configs")
    y = header(p, "FINDINGS | CONFIGURATIONS, TWO DAYS AND THE SPREAD", "Which differences between our runs mean something (D207, D209-D213)", None, 22)
    rows = [["Run (54 steps, nov26)", "T", "U", "V", "Q", "P", "QCL", "QCI", "worst ratio steps >= 3"],
            ["D209 replayed radiation (default)", "54/0/0", "54/0/0", "54/0/0", "42/12/0", "52/2/0", "52/1/1", "37/17/0", "QCI 1.69"],
            ["D210 server radiation (Fortran callback)", "54/0/0", "54/0/0", "53/1/0", "54/0/0", "52/2/0", "52/1/1", "49/5/0", "QCI 1.08"],
            ["D212 server + own surface fields, seed, COSZ", "54/0/0", "53/1/0", "54/0/0", "54/0/0", "51/3/0", ("41/12/1", ORANGE), "42/12/0", ("QCL 1.41", ORANGE)],
            ["D213 spread (control, p1, p2, p3), within min..max", "53..54", "53..54", "54", "42..49", "52..54", "50..52", "37..52", "QCI 1.07..1.84"],
            ["Day 2 (steps 54-107), replayed radiation", "54/0/0", "54/0/0", "54/0/0", "17/37/0", "54/0/0", "53/1/0", "52/2/0", "QCI 1.10"]]
    hh = p.table(MARGIN, y, [250, 62, 62, 62, 70, 62, 64, 70, W - 2 * MARGIN - 702], rows, 8.2, maxh=170, bold_cols=(0,))
    yy = y + hh + 10
    cw = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, yy, cw, 150, GREEN)
    p.text(MARGIN + 14, yy + 8, cw - 24, "What the spread says", 10.5, INK, True)
    p.bullets(MARGIN + 14, yy + 28, cw - 26, [
        "One-ulp perturbations like the real members' move our QCI within-count from 37 to 52 and Q from 42 to 49: the D209 vs D210 differences are inside that spread.",
        "QCL at step 1 is 2.83..2.92 in EVERY run and configuration: systematic, not noise.",
        "D210/D212 Q = 54 and D212 QCL = 41/12/1 lie outside a 3-run range, which underestimates the spread: single unconfirmed results."], 9.6, maxh=120)
    card(p, MARGIN + cw + 12, yy, cw, 150, AMBER)
    p.text(MARGIN + cw + 26, yy + 8, cw - 24, "Conditions and limits", 10.5, INK, True)
    p.bullets(MARGIN + cw + 26, yy + 28, cw - 26, [
        "HYBRID. Ent exports, land forcing, tile radiation columns and PBL templates are replayed from the real records; radiation is replayed, or computed by the unmodified Fortran server (D210, D212).",
        "Day 2 reuses the same member runs (perturbed at step 0): its floor is a continuation, not a fresh perturbation. Replayed records pull both days toward the real trajectory.",
        "ACCEPTANCE s9 stays NOT MET (QCL step 1). Tables are single runs; n = 3-4 realisations."], 9.6, maxh=120)
    footer(p, n, "D207 (day 2), D209, D210, D212, D213 (spread table), ACCEPTANCE s9")
    return p


@slide
def s_day(n):
    p = P("day")
    y = header(p, "FINDINGS | THE 54-STEP DAY", "One model day (nov26) vs five real members: NOT MET under ACCEPTANCE s9", None, 22)
    p.image(MARGIN - 4, y - 4, W - 2 * MARGIN + 8, 176, os.path.join(IMG, "day_score.png"))
    yy = y + 174
    rows = [["Field", "within / near / beyond D209 (D203)", "worst ratio steps >= 3", "worst ratio all steps", "near steps", "beyond"],
            ["T", "54/0/0 (54/0/0)", "0.94 (step 15)", "0.94", "-", "-"], ["U", "54/0/0 (53/1/0)", "0.89 (21)", "0.89", "-", "-"],
            ["V", "54/0/0 (54/0/0)", "0.97 (10)", "0.97", "-", "-"], ["Q", "42/12/0 (40/14/0)", "1.03 (46)", "1.03", "40-53 (12)", "-"],
            ["P", "52/2/0 (54/0/0)", "1.04 (33)", "1.04", "2", "-"],
            ["QCL", "52/1/1 (50/3/1)", "0.95 (9)", ("2.83 (step 1)", ORANGE), "1", ("step 1", ORANGE)],
            ["QCI", "37/17/0 (49/5/0)", "1.69 (47)", "1.69", "17 steps", "-"]]
    hh = p.table(MARGIN, yy, [50, 190, 140, 140, 150, W - 2 * MARGIN - 670], rows, 7.8, maxh=140, bold_cols=(0,))
    yy += hh + 6
    banner(p, MARGIN, yy, W - 2 * MARGIN, 66,
           "Status: NOT MET (owner decision 2026-10-08, ACCEPTANCE s9). The only field beyond 2x is QCL at step 1 (ours 1.15e-6 vs largest member distance 4.06e-7). "
           "All 54 steps are finite since the D202 land fix. QCL step 1 (2.83) is systematic in every run; the other fields vary within the run-to-run spread (next slide). 'Within at every step' also fails (Q, P, QCL, QCI near at some steps). The earlier claim of a ~1e-9 floor at steps < 3 was wrong (correction 8a649f7); no exception clause is recorded.", ORANGE, 8.8)
    p.text(MARGIN, yy + 72, W - 2 * MARGIN, "Conditions: HYBRID; radiation REPLAYED (not computed); libimf host callback; Ent, land forcing, template columns recorded; ocean/ice/land daily updates not applied; nit_strict=False after step 22; one start state; five members. D203 (before the computed GHY schedule, D204) in brackets.", 7.8, MUTED, maxh=24)
    footer(p, n, "D209 (score.md/json, parent re-score), D203, D213 (spread), ACCEPTANCE s9, correction 8a649f7, D197")
    return p


@slide
def s_diag(n):
    p = P("diag")
    y = header(p, "FINDINGS | DIAGNOSES AFTER THE DAY", "D193-D197: each exceedance traced to a cause", None, 22)
    cards = [
        (GREEN, "D194 -> D195 (fixed)", "Step 7 divergence",
         "Not the tile set. GHY cells with ffnit > 11 must regenerate sub-iteration lengths from OUR precipitation; the step used the recorded one. Fix merged (nit_fix default on): C1 bitwise steps 0-22. The NumPy reference aborts at step 23 (cell (41,21): 11 vs 14 iterations). D193's 'rebuild when the tile set changes' was wrong: the cause is max_substeps."),
        (ORANGE, "D196 (open, patch not applied)", "Step-48 tile mismatches",
         "54 lake cells (FOCEAN = 0) lose their open-water tile because daily_LAKE (LAKES.f:2492) is not applied at the day boundary (FLAKE changes in 632 of 636 lake cells at 47->48). Our NumPy chain has the same omission. Level-2 fix is a ~500-line port, not started."),
        (ORANGE, "D197 (explained, still NOT MET)", "QCL step 1, ratio 2.84",
         "One bistable cell (31,12,15) holds 96.8% of the squared error. 1-ulp PK differences from the dynamics exit flip a stratiform threshold; 1 of 8 random 1-ulp draws reproduces it exactly. CLOUDS port from real inputs is category A/B; the NumPy chain has the same value. Without that cell the ratio is 0.51. Does not relax the criterion."),
        (GREEN, "D199 -> D203 (fixed)", "Land runaway, NaN at step 39",
         "D199 (land humidity uses the current substep's elhx) fixed dec01/jan01 at step 0 but sent the nov26 day to NaN (cell 41,20). D201: the elhx switch alone decides finite vs NaN. D202: GHY must receive the PBL's gusti (GHY_DRV.f:1267), not the recorded one. After that all 54 steps are finite; QCL step 1 (2.84x) is unchanged. Mechanism of the original trajectory change is not proven. Correction 8a649f7 (noise floor) stays in force."),
    ]
    cw = (W - 2 * MARGIN - 12) / 2
    hs = [150, 130]
    for i, (c, t1, t2, body) in enumerate(cards):
        x = MARGIN + (i % 2) * (cw + 12)
        yy = y + (i // 2) * 168
        h = 160
        card(p, x, yy, cw, h, c)
        p.text(x + 14, yy + 8, cw - 24, t1, 8.5, c, True, True, maxh=14)
        p.text(x + 14, yy + 24, cw - 24, t2, 12, INK, True, maxh=18)
        p.text(x + 14, yy + 46, cw - 24, body, 10.2, INK2, maxh=h - 50)
    footer(p, n, "D193, D194, D195 s1-3, D196 s1-2, D197, D199-D203, correction 8a649f7; ACCEPTANCE s9")
    return p


# ------------------------------------------------------------------ 12
@slide
def s_floor(n):
    p = P("floor")
    y = header(p, "FINDINGS | WHY THE TEST IS STATISTICAL", "The real model is chaotic; our step is judged against its own spread", None, 22)
    cw = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, y, cw, 200, BLUE)
    p.text(MARGIN + 14, y + 8, cw - 24, "Noise floors measured with the real binary", 11, INK, True)
    p.bullets(MARGIN + 14, y + 28, cw - 26, [
        "D3 (5 days, 1-ulp T perturbation): T layer-1 pointwise rms 0.048 K, global-mean 1.4e-4 K; whole column T rms 0.27 K; only 36 of 257 restart variables stay bitwise identical.",
        "D151/D149: five real members for the nov26 day; our NumPy and JAX runs diverge from each other exactly like real members.",
        "D165: 8 real JAN1950 members (ctrl + 7 one-ulp); ctrl reproduces the stored month bitwise; t_500 global-mean sd 0.043 K; 8 members = about +-25% on a std; one season."], 10, maxh=165)
    card(p, MARGIN + cw + 12, y, cw, 200, GREEN)
    p.text(MARGIN + cw + 26, y + 8, cw - 24, "Earlier day results (replayed / free radiation)", 11, INK, True)
    p.bullets(MARGIN + cw + 26, y + 28, cw - 26, [
        "D151 open loop: T, U, V within at all 54 steps; Q near 15 steps, condensates near (<= 1.11x); none beyond 2x.",
        "D155-D157 free-running radiation (real RADIA server): T/U/V within at every step; condensates up to 1.95x (QCI step 47); radiative flux differences < 1 W/m2 global mean.",
        "D171 (Ent computed): QCI worst 1.62; QCL step 1 ratio 2.90 (D192 re-score). The D192 scorer reproduces D157 and D171 exactly.",
        "None of these days is 'within at every step' for all fields; they were reported before the 2026-10-08 pass rule."], 9.8, maxh=165)
    y += 212
    card(p, MARGIN, y, W - 2 * MARGIN, 108, AMBER)
    p.text(MARGIN + 14, y + 8, 600, "Consequences", 11, INK, True)
    p.bullets(MARGIN + 14, y + 28, W - 2 * MARGIN - 28, [
        "Single steps are deterministic (floor 0) and can be bitwise only with libimf; beyond step ~1 any rounding difference grows, so acceptance is statistical (ACCEPTANCE s4).",
        "Without libimf, cloud thresholds flip in 3-5% of columns: a multi-step run drifts about 0.1-0.2 K after 6 steps (GOAL.md; D129).",
        "Limits: five members, one start state, one day; heavy-tailed condensates (QCI, QCL) are single-cell dominated, so a ratio near 2 is not a significance statement (D192)."], 10, maxh=76)
    footer(p, n, "D3, D129, D149-D151, D155-D157, D165, D171, D192; GOAL.md; ACCEPTANCE s4")
    return p


# ------------------------------------------------------------------ 13
@slide
def s_hazards(n):
    p = P("hazards")
    y = header(p, "FINDINGS | MATH-LIBRARY AND REPRODUCIBILITY HAZARDS", "What can silently change a bitwise or statistical result", None, 22)
    rows = [["Hazard", "What was found", "Source"],
            ["Intel libimf vs glibc pow/exp", "Bitwise needs the real build's libimf. With NumPy/glibc dynamics differ at ~1e-13 and cloud columns flip in 3-5%. pow/exp/log are not correctly rounded: ~0.02% of a KPP table differs by 1 ulp (D54).", "README caveats; D54; D129"],
            ["Single-precision literals", "Fortran literals without d0 (e.g. **(1./3.) in OCNKPP.f) are single precision under the build; REAL*4 DZDH1 in RIVERF.", "D54, D167"],
            ["REAL*16 sites", "Only Ti/Ti2b in SEAICE.f; a binary128 emulation makes seaice_to_atmgrid and ADDICE bitwise; not applied to existing modules.", "D173"],
            ["XLA CPU flags", "FMA fusion and x/c rewrites break bitwise; needs --xla_cpu_max_isa=AVX and algsimp (and reshape-mover, D182) disabled before importing JAX; such tests need separate processes.", "README caveats; D182"],
            ["Core count / compile cache", "Existing code gives different results on 1 and 3 cores; a warm JAX compile cache changes results. Use identical affinity and no cache for validated runs.", "D139, D145, D175"],
            ["Host callbacks on one core", "io_callback/pure_callback inside jit DEADLOCK on a single core (0.1% CPU for 23 min). Give such jobs >= 2 cores.", "D185; README"],
            ["NumPy vs jnp semantics", "x**4 is libm pow in NumPy but repeated multiplication in jnp; jnp.cumsum is not a left-to-right sum.", "SOCRATES_PORT_PLAN s3"],
            ["GPU device math", "exp, sin, pow differ from NumPy by 1 ulp in 6.14%, 11.96%, 9.07% of values (log 0.22%): category A is unattainable on GPU unless every call goes through a host callback.", "gpu/DISCOVER_RUN.md"]]
    p.table(MARGIN, y, [150, 600, W - 2 * MARGIN - 750], rows, 10, maxh=420, bold_cols=(0,))
    footer(p, n, "README_START_HERE.md caveats; D54, D129, D139, D145, D167, D173, D175, D182, D185; gpu/DISCOVER_RUN.md; SOCRATES_PORT_PLAN s3")
    return p


# ------------------------------------------------------------------ 14
@slide
def s_gpu(n):
    p = P("gpu")
    y = header(p, "FINDINGS | GPU", "The A100 probe says: fuse into few large kernels, and do not expect bitwise", None, 22)
    cw = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, y, cw, 250, GREEN)
    p.text(MARGIN + 14, y + 8, cw - 24, "Measured on Discover (NVIDIA A100-SXM4-40GB, JAX 0.6.1 CUDA)", 10.5, INK, True, maxh=30)
    p.bullets(MARGIN + 14, y + 40, cw - 26, [
        "Smoke test 2026-10-07 (job 58776063): 27 pass, 2 warn (no pytest, no libimf), 0 fail; float64 works; jit+scan on model-sized state compiled in 0.6 s.",
        "Probe (job 58778982): float64 matmul 2048x2048 1.2 ms (about 820 ms on one CPU core of the work node).",
        "100,000 sequential scan steps: 575 ms on the A100 vs 41 ms on one CPU core: launch-bound, ~5.7 us/step; the probe note calls it 14x slower. Long chains of small ops lose; only wide parallel work wins.",
        "Device exp 6.14%, sin 11.96%, pow(x,0.25) 9.07%, log 0.22% of values differ from NumPy (max 1 ulp)."], 10, maxh=200)
    card(p, MARGIN + cw + 12, y, cw, 250, ORANGE)
    p.text(MARGIN + cw + 26, y + 8, cw - 24, "What D198 built, and what the first GPU runs show", 10.5, INK, True)
    p.bullets(MARGIN + cw + 26, y + 28, cw - 26, [
        "gpu_step_run.py runs the ASSEMBLED step in libm mode (no libimf), radiation replayed, no NumPy reference needed; data subset 6b (66 files, 1.58 GB), runbook MODE=step.",
        "On CPU: libm step equals the NumPy libm chain (555 A + 3 B of 558). Against the libimf CPU result: A126 B170 C47 D215: threshold fields flip with ANY different arithmetic (another XLA version: 198 of 558 category D; no CPU flags: 208).",
        "RUN on the A100 (jobs 58786705, 58788593, 58789387): step 0 finite, within the CPU libimf-vs-libm spread (T 7e-7, Q 2e-4). Cold 726-768 s (655-689 s compile). Steps >= 1: NaN from the surface stage (land/ice/lake/ocean); atmosphere phase 1 is clean; error flags did not catch it.",
        "A CPU control with the simplifier on is finite; cause on the GPU not known yet (JAX_DEBUG_NANS run pending). The 13.1 s step was a repeat of step 0; steps 2-5 took 25 s (NaN steps), the CPU 10-19 s.",
        "99 jit executions, ~490 eager dispatches, 176 QUS callbacks per step: launch-bound; the step is not faster on the GPU so far."], 10, maxh=215)
    y += 262
    banner(p, MARGIN, y, W - 2 * MARGIN, 52,
           "Honest label for any future GPU result: libm mode, radiation replayed, recorded inputs, HYBRID, not bitwise to CPU or Fortran; judged at rounding-level categories for one step and statistically beyond. It must not be used for a C2 / fidelity claim (D198 parent check).", AMBER, 8.8)
    footer(p, n, "gpu/DISCOVER_RUN.md (smoke 58776063, probe 58778982, step jobs 58786705/58788593/58789387); D198 s1-3, s6; Track A GPU numbers (kernels 2.3-6.3x, chain 4.2x) are in STATUS.md and are reduced-scope")
    return p


# ------------------------------------------------------------------ 15
@slide
def s_tests(n):
    p = P("tests")
    y = header(p, "FINDINGS | TESTS AND REPRODUCIBILITY", "3,056 passed, 6 skipped at a89e117 (plus one known artifact failure)", None, 22)
    p.image(MARGIN, y, 470, 243, os.path.join(IMG, "tests_growth.png"))
    x2 = MARGIN + 490
    p.bullets(x2, y + 4, W - x2 - MARGIN, [
        "Batch regression of a89e117 (clean git-archive export, JOBS=4, cores 8-11, 3,667 s): 3,056 passed, 6 skipped, 1 failed.",
        "The failure, test_jax_harness::test_header_complete, asserts a 40-character git HEAD that an export lacks; passes in the real tree. The test was then changed to tolerate a missing .git (commit 1036056); the full suite was NOT re-run after it.",
        "The 6 skips are gated tests (environment switches) not run in the batch.",
        "Mutation and non-vacuity checks accompany most ports; flag-sensitive JAX tests run in separate processes (run_all_tests.sh; sharded runner 34 min vs ~83 min serial, aggregate totals equal).",
        "Rule of the owner (2026-10-07): large batches; no full-suite re-run per code change."], 8.8, maxh=260)
    y += 256
    banner(p, MARGIN, y, W - 2 * MARGIN, 70,
           "Passed counts are what the repository records at each commit (680 after D51 per Project_Summary; 2,333; 2,848; 2,925; 2,953; 2,972; 3,043; 3,056). They count tests, not validated physics: acceptance is the ledger's per-field categories and the scorer of D192. "
           "The 2,471 + 206 entry of 264283d is left out (ambiguous).", MUTED, 8.5)
    footer(p, n, "README_START_HERE handoff; commits 6cb73ed, e843ad6, bd8dd2e, 4d352a8, 13d00c8, 51acc2f, eda8789, 1036056; Project_Summary s5")
    return p


# ------------------------------------------------------------------ 16
@slide
def s_remains(n):
    p = P("remains")
    y = header(p, "WHAT REMAINS", "Open work, grouped by what it blocks", None, 22)
    cw = (W - 2 * MARGIN - 24) / 3
    cols = [
        (ORANGE, "Blocks the day criterion", [
            "Day boundary: apply daily_LAKE (D196, ~500-line level-2 port, not started); other daily updates (ocean/ice/land) not applied.",
            "New-ice-tile donor rule (D190 s6): not implemented; multi-day runs without per-step records need it.",
            "NumPy reference aborts at step 23: C1 beyond step 22 unavailable; nit_strict policy beyond step 22.",
            "Leave-one-out scoring with the 8 JAN1950 members (D172) for longer windows: not implemented for the coupled step."]),
        (AMBER, "Blocks 'end-to-end JAX'", [
            "Radiation stays Fortran (served/replayed): SOCRATES port is planning only, 34,899 reached lines, 135-295 h estimate, rule unchanged.",
            "Recorded inputs: Ent exports (stage-1 port exists, not in the assembled step), land forcing, tile radiation columns, CONDSE entry set, PBL profile columns; AG2OG/IG2OG and straits start in ocean chain.",
            "Host callbacks (libimf, QUS, poles, OADVT2) and host template build; no transfer-guard test of the full step.",
            "dec01/jan01 PARTLY MET from the ported land surface (not re-diagnosed)."]),
        (BLUE, "Blocks the project's purpose (GPU)", [
            "First GPU run of the assembled step (D198 runner ready; MODE=step).",
            "Fusing the step into few large kernels (probe: launch-bound).",
            "Speed without the libimf callback was not measured (D191 s5).",
            "Reference identity: our run is not shown to be the supplement run P2SAoM40_003 (PROVENANCE_MANIFEST); the 100-year climatology comparison is a later milestone, not tested."]),
    ]
    for i, (c, t, its) in enumerate(cols):
        x = MARGIN + i * (cw + 12)
        card(p, x, y, cw, 380, c)
        p.text(x + 14, y + 8, cw - 24, t, 11, INK, True, maxh=18)
        p.bullets(x + 14, y + 32, cw - 26, its, 10.6, maxh=340, gap=8)
    p.text(MARGIN, y + 388, W - 2 * MARGIN, "Effort: no current total is derived. GOAL.md (10-06) says ~100-170 h more to a month comparison; README_START_HERE (10-07) says that figure is too high but gives no new total; treat any total as provisional.", 8.3, MUTED, maxh=22)
    footer(p, n, "README_START_HERE (next steps, remaining effort); D190 s6, D191 s7, D195 s6, D196, D198 s6; SOCRATES_PORT_PLAN; PROVENANCE_MANIFEST s4, s6; ACCEPTANCE s9")
    return p


# ------------------------------------------------------------------ 17
@slide
def s_decisions(n):
    p = P("decisions")
    y = header(p, "DECISIONS", "Owner decisions taken (10-07, 10-08) and decisions still pending", None, 22)
    cw = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, y, cw, 392, GREEN)
    p.text(MARGIN + 14, y + 8, cw - 24, "Taken (recorded in the repository)", 11, INK, True)
    p.bullets(MARGIN + 14, y + 30, cw - 26, [
        "2026-10-07 (GOAL.md): 'port' = end-to-end JAX first, validated components second; success = one coupled step first, then multi-day; a Fortran radiation callback is acceptable but must be labelled; match P2SAoM40 first; one-month and 100-year comparisons are later milestones.",
        "2026-10-07: ACCEPTANCE_CRITERIA approved before any coupled step was compared; later changes must be dated, never silent.",
        "2026-10-07 (s8): headline C2 uses a labelled libimf host callback and is also reported in libm mode; three jit units plus host calls satisfy s1 if boundaries and non-JAX stages are reported; D176 files adopted, D178 after verification.",
        "2026-10-07: Ent exports stay recorded up to one day; Ent port started. One writer per working tree. Test/push rule: large batches.",
        "2026-10-08 (s9): day pass rule = never beyond 2x at every step; nov26 day NOT MET; no exception for steps 0-2; relaxation only later, only to advance to the GPU run, dated, with both statuses."], 10.6, maxh=350, gap=7)
    card(p, MARGIN + cw + 12, y, cw, 392, ORANGE)
    p.text(MARGIN + cw + 26, y + 8, cw - 24, "Pending for the owner (as the documents stand)", 11, INK, True)
    p.bullets(MARGIN + cw + 26, y + 30, cw - 26, [
        "Whether to relax the day criterion to advance to the GPU run (s9.3): must be a dated decision after the results, stating the unrelaxed status next to it.",
        "nit_strict policy beyond step 22 and what to do about the NumPy reference abort at step 23 (D195).",
        "Whether to implement daily_LAKE (D196 patch proposal, level 2 ~500 lines) and the new-ice-tile donor rule.",
        "Name the GPU host formally (ACCEPTANCE s7.4 / s8.5 still list it open; Discover A100 smoke/probe jobs exist) and when to run MODE=step.",
        "Whether carbon outputs are part of the F3 acceptance; whether to apply the binary128 Ti/Ti2b diff (speed vs bitwise).",
        "Whether to lift the rule 'SOCRATES is never ported' (plan: 135-295 h, decisions D1-D6).",
        "With the project lead: the final comparison target (published 100-year climatology vs a shorter run), identity of our reference with P2SAoM40_003, where ANN4099.aijP2SAoM40.nc lives."], 10.6, maxh=350, gap=7)
    footer(p, n, "GOAL.md; ACCEPTANCE_CRITERIA s7-s9; README_START_HERE (owner decisions, 6a-6c); D195 parent check; D196 s4; SOCRATES_PORT_PLAN; PROVENANCE_MANIFEST s6; DOCUMENTATION_REVIEW")
    return p


# ------------------------------------------------------------------ 18
@slide
def s_caveats(n):
    p = P("caveats")
    y = header(p, "HONESTY NOTES", "Where documents disagree, and what this deck did about it", None, 22)
    rows = [["Topic", "Disagreement found", "Used here"],
            ["Ledger range", "The request mentions D1..D199; the ledger ends at D198 (HEAD ec36779).", "D1-D198"],
            ["Step time", "GOAL.md (10-06): ~6 s per step plus clouds; D187: 44.3 s; D191: 18.8 s steady.", "D191 (newest)"],
            ["Remaining effort", "GOAL.md ~100-170 h; README says too high, no new total; older 30-65 h understated.", "no total claimed"],
            ["Noise floor at steps < 3", "D192/D193/D195 '~1e-9'; correction 8a649f7 shows 4e-7 (QCL, step 1).", "correction"],
            ["SURFACE rebuild cause", "D193: tile set changes; D195: max_substeps.", "D195"],
            ["Day table counts", "D193 Q 43/11/0, QCL 49/4/1; D195 Q 41/13/0, QCL 47/6/1.", "D195"],
            ["C1 array count", "D187 559/559; D191 558/558 (p4_ij host index array is reference-only).", "558 for D191"],
            ["Summary page age", "Project_Summary_and_Conclusions.md stops at D51 (680 tests); STATUS.md is Track A.", "ledger + README"],
            ["GPU host", "ACCEPTANCE lists the GPU host as to be named; README/DISCOVER_RUN record an A100 smoke test and probe on Discover.", "both stated"],
            ["Jit executions in the probe note", "DISCOVER_RUN says 'about 125'; D187 262, D191 99.", "D187/D191"]]
    p.table(MARGIN, y, [150, 600, W - 2 * MARGIN - 750], rows, 9.6, maxh=380, bold_cols=(0,))
    p.text(MARGIN, 470, W - 2 * MARGIN, "Not claimed anywhere in this deck: a pass of the day criterion; end-to-end JAX; any GPU speed-up of the full port; reproduction of the published 100-year climatology (ACCEPTANCE s5); any C2 statement from a C1 result.", 8.8, AMBER, True, maxh=30)
    footer(p, n, "GOAL.md; README_START_HERE; Project_Summary; STATUS.md; ACCEPTANCE s5, s7-s9; D187, D191, D193, D195, correction 8a649f7; gpu/DISCOVER_RUN.md")
    return p


# ------------------------------------------------------------------ 19
@slide
def s_appendix(n):
    p = P("appendix")
    y = header(p, "APPENDIX | BEFORE THE FULL-FIDELITY PORT, SOURCES, REBUILD", "Track A history (superseded) and where everything lives", None, 22)
    cw = (W - 2 * MARGIN - 12) / 2
    card(p, MARGIN, y, cw, 400, MUTED)
    p.text(MARGIN + 14, y + 8, cw - 24, "Track A (2026-08-28 to 09-24), reduced scope", 11, INK, True)
    p.bullets(MARGIN + 14, y + 30, cw - 26, [
        "Chained GHY/ATURB/SEAICE/LAKES/PBL/SURFACE driver, JAX-vectorized; measured CPU speedups 20-104x per component (Project_Summary s4A); kernels on GPU 2.3x (DRYCNV) and 6.3x (PBL); fused chain 4.2x GPU vs CPU on an A100 (STATUS.md).",
        "Deliberately NOT full fidelity: SEAICE, LAKES and part of ATURB were placeholders; GHY found to be one on 2026-09-23. DRYCNV is not in the real P2SAoM40 binary (ATURB is; commit b2e0487).",
        "The earlier deck (status_deck_2026-10-01.pdf, built by build_pdf_2026-10-01.py) reported 14 of 17 modules, 2.3-6.3x GPU kernels and a stop-the-port recommendation; the owner reversed that, so those figures are history, not current status.",
        "Live artifact of 2026-09-24 (deck.json, slides/*.html) is unchanged and NOT refreshed."], 10.4, maxh=360, gap=6)
    card(p, MARGIN + cw + 12, y, cw, 400, GREEN)
    p.text(MARGIN + cw + 26, y + 8, cw - 24, "Sources and rebuild", 11, INK, True)
    p.bullets(MARGIN + cw + 26, y + 30, cw - 26, [
        "Ledger: FULL_FIDELITY_DELTAS.md (D1-D198). Index: Reports/README_START_HERE.md. Goal and status: Reports/GOAL.md. Criteria: Reports/ACCEPTANCE_CRITERIA.md (s1, 3, 4, 8, 9).",
        "Design: JAX_COVERAGE_MATRIX.md (stages S0-S8). Plan/provenance: RECONCILIATION_PLAN.md, PROVENANCE_MANIFEST.md. Contingency: SOCRATES_PORT_PLAN.md. Lessons: LESSONS_LEARNED.md. Chronology table: Reports/PROJECT_CHRONOLOGY.md.",
        "Rebuild: status_slides/README.md. In short: conda python make_figures.py DIR (matplotlib); python3 build_pdf.py status_deck.pdf --png DIR (system python3, reportlab). Core 11, scratch outside the repository.",
        "Stage map S0-S8 (JAX_COVERAGE_MATRIX s6): S0 harness, S1 state pytree, S2 phase 1, S3 radiation hand-off, S4 surface tiles/land, S5 post-tile + ocean, S6 assembly, S7 one coupled step, S8 multi-day.",
        "Review by: when the day criterion status changes, a GPU run exists, or the owner records a relaxation."], 10.4, maxh=360, gap=6)
    footer(p, n, "Project_Summary s4A; STATUS.md; JAX_COVERAGE_MATRIX s6; status_slides/README.md")
    return p


def main(out, png=None):
    L.OVERFLOWS.clear()
    c = canvas.Canvas(out, pagesize=(W, H))
    c.setTitle("ROCKE-3D to JAX: chronology and findings, 2026-10-08")
    for i, fn in enumerate(SLIDES, 1):
        pg = fn(i)
        pg.to_pdf(c)
        if png:
            os.makedirs(png, exist_ok=True)
            pg.to_png(os.path.join(png, "slide_%02d.png" % i))
    c.save()
    print("wrote", out, len(SLIDES), "slides")
    for o in L.OVERFLOWS:
        print("OVERFLOW:", o)
    return len(L.OVERFLOWS)


if __name__ == "__main__":
    args = sys.argv[1:]
    png = None
    if "--png" in args:
        i = args.index("--png"); png = args[i + 1]; del args[i:i + 2]
    sys.exit(1 if main(args[0] if args else os.path.join(HERE, "status_deck.pdf"), png) else 0)
