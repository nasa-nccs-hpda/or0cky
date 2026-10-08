"""
Single source of the project chronology, used by make_figures.py (timeline
figures) and build_chronology_md.py (Reports/PROJECT_CHRONOLOGY.md).

Every row comes from `git log` (date, short commit id, subject) and from the
ledger entry (FULL_FIDELITY_DELTAS.md) or report named in `src`. Nothing here
is a new measurement. Status as of 2026-10-08, HEAD ec36779 (ledger ends at
D198; there is no D199 entry in the repository).
"""

# phase key -> (label, start, end, color)
PHASES = [
    ("A", "Track A: reduced ports + GPU kernels", "2026-08-28", "2026-09-24", "#94A3B8"),
    ("B", "Phase 0: Fortran oracle", "2026-09-24", "2026-09-24", "#38BDF8"),
    ("C", "Track B: component ports vs Fortran (D4-D145)", "2026-09-24", "2026-10-06", "#4ADE80"),
    ("D", "Validation rungs F1/F2/F3 (D127-D165)", "2026-10-06", "2026-10-07", "#FBBF24"),
    ("E", "Reconciliation + owner decisions", "2026-10-07", "2026-10-07", "#F472B6"),
    ("F", "Hybrid coupled step, JAX stages S0-S8 (D180-D197)", "2026-10-07", "2026-10-08", "#F97316"),
    ("G", "GPU tooling (Discover A100)", "2026-10-07", "2026-10-08", "#A78BFA"),
]

# (date, "HH:MM", commit, phase, ids, event, what was learned, source)
ROWS = [
    ("2026-08-28", "", "af7bdfb", "A", "", "Initial commit",
     "Start of the repository.", "git log"),
    ("2026-09-19", "", "c2991dd", "A", "", "Physics-only JAX orchestrator driven by real P2SAoM40 production data",
     "First chained reduced-scope driver (Track A).", "git log; STATUS.md"),
    ("2026-09-21", "", "5cbe7f8", "A", "", "GPU update for P2SAoM40",
     "Kernel-level GPU speedups 2.3-6.3x measured 2026-09-20 (DRYCNV, PBL); fused chain 4.2x GPU vs CPU on an A100 (2026-09-22). Reduced scope, later superseded.", "STATUS.md"),
    ("2026-09-23", "", "abd805b", "A", "", "Close out port (Track A)",
     "Track A recommended against a full-fidelity port; the owner later reversed this (twice confirmed).", "Project_Summary s1; STATUS.md"),
    ("2026-09-24", "11:43", "97e2457", "B", "Phase 0", "ModelE build environment recovered; 5-day Fortran re-run reproduces the restart bitwise",
     "A trustworthy Fortran oracle exists.", "git log"),
    ("2026-09-24", "11:52", "c787c0c", "B", "", "Delta ledger opened: Track A vs real Fortran SURFACE",
     "Controls show the Track A surface step is worse than identity (60b008b).", "git log"),
    ("2026-09-24", "12:00", "4ec6056", "B", "D3", "Phase 0 complete: chaos noise floor measured, gate passed",
     "A 1-ulp perturbation grows to 1-3% of field variability pointwise in 5 days (T rms 0.048 K layer 1); acceptance beyond one step must be statistical.", "D3"),
    ("2026-09-24", "", "9870c35", "C", "D4", "Track B starts: ATURB A-grid port matches real Fortran at rounding level",
     "Dump-hook-and-validate method works.", "D4"),
    ("2026-09-28", "", "8b2e509", "C", "D29", "DYNSI/VPICEDYN/FORM/PLAST/RELAX validated bitwise/float64-exact",
     "3 real porting bugs found by comparison, not by reading.", "D29; Project_Summary s3"),
    ("2026-09-29", "", "169659b", "C", "D36", "OSTRES2 ported; legacy OCNDYN.f driver found dead code",
     "Scope correction: roughly half of the estimated ocean dynamics code never executes.", "D36"),
    ("2026-09-30", "", "f648263", "C", "D51", "Gent-McWilliams family closed (680 tests passed after D51)",
     "OCNDYN2 core and mesoscale mixing done; OCNKPP sized as the largest remaining item.", "Project_Summary s1, s5"),
    ("2026-10-03", "", "ec5a323", "C", "D55", "OVDIFFS + TRIDIAG bitwise-exact on all 152,022 real calls",
     "KPP vertical mixing proceeds piece by piece (D55-D61).", "D55"),
    ("2026-10-04", "", "d1134bd", "C", "D70-D73", "Straits step ported to batched JAX, validated on all three dates",
     "Straits confirmed active for P2SAoM40 (NMST=12). Direction: speed-first batched JAX.", "git log; Project_Summary update"),
    ("2026-10-05", "", "867268e", "C", "D121-D123", "Chained atmosphere dynamics step (all dynamics ports, real order)",
     "Bit-for-bit with the real model on 18 steps when Intel libimf pow is used.", "README_START_HERE; GOAL.md"),
    ("2026-10-06", "", "978672e", "C", "D118-D120", "Chained whole-ocean step",
     "Matches real dumps on 3 dates x 12 steps; wrong OMEGA constant found and fixed (07c35ee).", "README_START_HERE; GOAL.md"),
    ("2026-10-06", "13:08", "d7c871a", "D", "D127-D129", "Chained atmosphere step and first F1-gate verdict",
     "With libimf and recorded land: MET with named columns; with ported GHY nov26 NOT MET; without libimf NOT MET (3-5% of cloud columns flip).", "D129"),
    ("2026-10-06", "", "f21e007", "D", "D135-D136", "Two land-model porting errors fixed (precipitation conditioning, irrigation term)",
     "Hidden by loose tolerances; found by the F1 diagnosis. Re-run (0ad795b): nov26 MET, dec01/jan01 PARTLY MET with ported GHY.", "D135, D136; 0ad795b"),
    ("2026-10-06", "18:14", "3ccf563", "D", "D149-D151", "One model day (54 steps) replayed open loop against 5 real one-ulp members",
     "T, U, V within the members' spread at all steps; Q, P, condensates near its top (<= 1.1x).", "D151"),
    ("2026-10-06", "", "eb8cdb8", "D", "D152-D154", "Radiation server: real RADIA, SOCRATES unmodified, bitwise oracle",
     "Radiation can be served by the original Fortran without porting SOCRATES.", "D152-D154"),
    ("2026-10-06", "19:55", "7bd7a14", "D", "D155-D157", "Free-running radiation day through the server",
     "T, U, V within at every step; condensates <= 1.95x; global-mean radiative flux differences < 1 W/m2.", "D157; GOAL.md"),
    ("2026-10-06", "22:50", "6e4859c", "D", "D163", "F3 AIJ-style diagnostics match the real 54-step accumulation to 5.3e-14",
     "257 AIJ + 4 AIJL columns; ~226 of 483 changed columns not accumulated.", "D163"),
    ("2026-10-07", "05:08", "02c80a8", "D", "D165", "Real JAN1950 ensemble (8 members); control reproduces the stored month bitwise",
     "Month-scale noise floor (t_500 global-mean sd 0.043 K); 8 members = about +-25% on a std.", "D165"),
    ("2026-10-07", "13:13", "9cf08e6", "E", "", "RECONCILIATION_PLAN: independent review checked against the repository",
     "Fidelity work was ahead of the JAX deliverable: coupled results were Python/NumPy orchestration.", "RECONCILIATION_PLAN s1 (O2)"),
    ("2026-10-07", "13:39", "721d789", "E", "", "Owner decisions recorded: end-to-end JAX first, one coupled step then multi-day, Fortran radiation callback acceptable (noted), match P2SAoM40",
     "Month-scale F3 work paused.", "GOAL.md; RECONCILIATION_PLAN s3"),
    ("2026-10-07", "14:19", "b71eb22", "E", "", "ACCEPTANCE_CRITERIA approved by the owner before any coupled step was compared",
     "Categories A/B/C/D unchanged; C1 (port consistency) and C2 (fidelity) must be reported separately.", "ACCEPTANCE_CRITERIA"),
    ("2026-10-07", "14:25", "bd9b16c", "E", "", "Owner answers: libimf host callback for the headline C2; three jit units count as JAX-driven",
     "Written as ACCEPTANCE section 8.", "ACCEPTANCE s8"),
    ("2026-10-07", "14:44", "5c5b7a4", "G", "", "gpu/: one-command sbatch and results collector (draft)",
     "GPU tooling starts (probe bc51149, smoke test 4afc426).", "git log"),
    ("2026-10-07", "16:58", "c7c42c8", "G", "", "GPU smoke test passed on Discover (A100-SXM4-40GB, JAX 0.6.1 CUDA): 27 pass, 2 warn, 0 fail",
     "float64 and jit+scan work on the GPU.", "gpu/DISCOVER_RUN.md"),
    ("2026-10-07", "16:26", "56e322d", "F", "D180", "JAX atmosphere step (stage 1) bitwise equal to the NumPy chain on 6 steps",
     "Cold compile about 3,000 s recorded.", "D180"),
    ("2026-10-07", "18:03", "b59481e", "F", "D182", "Cause of the ~3,000 s cold JAX step found; flag fix cuts step 0 to 390 s",
     "Disabling XLA algsimp makes XLA:CPU grow reshape chains; also disabling reshape-mover (7131447) gives ~30x faster cold compile, bitwise identical.", "D182"),
    ("2026-10-07", "18:26", "45e0d20", "F", "D183", "libimf host callback for the JAX stages",
     "JAX chain equals the NumPy libimf chain bitwise; step 0 MET.", "D183"),
    ("2026-10-07", "20:39", "8909e4e", "F", "D187", "First coupled-step report (HYBRID): C1 559/559 bitwise on 3 dates",
     "C2: nov26 MET with one named column, dec01/jan01 PARTLY MET; fails ACCEPTANCE items 1 and 3; 44.3 s per step, 262 jit executions.", "D187"),
    ("2026-10-07", "21:01", "dfe8ce1", "F", "D188", "Stage S4 step one: SURFACE stage in JAX on fixed-shape tiles",
     "Category A vs NumPy, about 2x faster, no eager dispatches inside the stage.", "D188"),
    ("2026-10-07", "22:42", "bf212f8", "F", "D189", "MSTCNV cloud-base loop as one device program with fused libimf callbacks",
     "Phase 1 of step 0 from about 33.6 s to 11.9 s, bitwise equal.", "D189"),
    ("2026-10-07", "23:26", "d320221", "G", "", "GPU probe on the A100",
     "Sequential scan 14x slower than a CPU core (launch-bound); device exp/sin/pow differ from NumPy by 1 ulp in 6-12% of values.", "gpu/DISCOVER_RUN.md"),
    ("2026-10-07", "23:38", "9602dc3", "E", "", "SOCRATES_PORT_PLAN (planning only; rule unchanged)",
     "34,899 reached lines (static count), 135-295 h estimate; SOCRATES is never ported or modified.", "SOCRATES_PORT_PLAN"),
    ("2026-10-07", "23:39", "3f421ec", "F", "D190", "Stage S5: post-tile surface and ocean as a device-resident JAX program",
     "Category A (149/149); 5 jit executions vs 123, 1.76 s vs 8.0 s.", "D190"),
    ("2026-10-08", "01:14", "cc429f8", "F", "D191", "ONE coupled step assembled device-resident (stages S6/S7)",
     "C1 558/558 bitwise on 3 dates; C2 reproduces D187; 99 jit executions vs 262; 18.8 s vs 44.3 s steady. Hybrid, not end-to-end JAX.", "D191"),
    ("2026-10-08", "04:06", "0ffa8f5", "F", "D192-D193", "S8 first attempt: 54-step nov26 day scored against 5 real members",
     "Not a pass; C1 bitwise only to step 6 (scorer D192 reproduces D157/D171 tables).", "D192, D193"),
    ("2026-10-08", "04:29", "eea8cf5", "F", "D194", "Step-7 divergence diagnosed",
     "GHY sub-iteration lengths of cells with ffnit > 11 must follow OUR precipitation; fix gives C1 bitwise steps 0-10.", "D194"),
    ("2026-10-08", "06:03", "a89e117", "F", "D195", "Nit rule merged; day re-scored",
     "C1 bitwise steps 0-22 (NumPy reference aborts at step 23); SURFACE rebuilds are caused by max_substeps, not the tile set (D193 wording corrected).", "D195"),
    ("2026-10-08", "08:15", "8a649f7", "F", "", "Correction: the '~1e-9 noise floor at steps < 3' statement is wrong",
     "QCL step 1: largest member distance 4.06e-7, ours 1.15e-6 (ratio 2.84). No exception clause recorded.", "D195 correction"),
    ("2026-10-08", "08:20", "f045127", "E", "", "ACCEPTANCE section 9: owner decision on the day criterion",
     "Pass rule = never beyond 2x at every step, all scored fields; nov26 day NOT MET; relaxation only later, dated, with both statuses.", "ACCEPTANCE s9"),
    ("2026-10-08", "08:54", "4349428", "F", "D196", "Step-48 tile mismatches traced to the missing daily_LAKE step at the day boundary",
     "54 pure-ice lake cells with FLAKE growth; diagnosis only, patch proposed as text.", "D196"),
    ("2026-10-08", "08:54", "c198401", "F", "D197", "QCL step-1 exceedance is ONE bistable cell",
     "Cell (31,12,15) holds 96.8% of the squared error; flipped by 1-ulp PK differences; status unchanged (NOT MET).", "D197"),
    ("2026-10-08", "09:27", "ec36779", "G", "D198", "GPU runner for the assembled coupled step (libm mode, radiation replayed); validated on CPU only",
     "No GPU run exists yet; a GPU result will not be bitwise.", "D198"),
]

# Owner decisions drawn on the figures: (date, "HH:MM", label)
DECISIONS = [
    ("2026-10-07", "13:39", "Owner 10-07: end-to-end JAX first; one coupled step, then multi-day; Fortran radiation callback OK (noted); match P2SAoM40"),
    ("2026-10-07", "14:19", "Owner 10-07: ACCEPTANCE approved before the first coupled step"),
    ("2026-10-07", "14:25", "Owner 10-07: libimf host callback for headline C2; 3 jit units count"),
    ("2026-10-08", "08:20", "Owner 10-08: day rule = never beyond 2x at every step"),
]

# Regression results quoted in the repository (date, "HH:MM", commit, passed, note)
TESTS = [
    ("2026-09-30", "00:00", "f648263", 680, "after D51 (Project_Summary)"),
    ("2026-10-06", "12:13", "6cb73ed", 2333, "0 failed"),
    ("2026-10-07", "04:37", "e843ad6", 2848, ""),
    ("2026-10-07", "08:56", "bd8dd2e", 2925, "1 skipped (RECONCILIATION_PLAN)"),
    ("2026-10-07", "15:22", "4d352a8", 2953, "2 skipped"),
    ("2026-10-07", "17:42", "13d00c8", 2972, "3 skipped"),
    ("2026-10-08", "02:42", "51acc2f", 3043, "1 export-artifact failure"),
    ("2026-10-08", "07:42", "eda8789", 3056, "6 skipped, 1 export-artifact failure"),
]

# Commits per calendar day (git log --format=%ad --date=short, 273 commits)
COMMITS_PER_DAY = {
    "2026-08-28": 2, "2026-09-07": 2, "2026-09-09": 2, "2026-09-18": 2, "2026-09-19": 4,
    "2026-09-20": 2, "2026-09-21": 6, "2026-09-22": 8, "2026-09-23": 2, "2026-09-24": 19,
    "2026-09-27": 14, "2026-09-28": 28, "2026-09-29": 20, "2026-09-30": 16, "2026-10-03": 5,
    "2026-10-04": 12, "2026-10-05": 24, "2026-10-06": 34, "2026-10-07": 58, "2026-10-08": 13,
}
