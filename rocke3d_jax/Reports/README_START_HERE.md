# Start here: where the work is recorded

This is the `full-fidelity-port` branch's index. It exists because the branch's own three
tracking documents (`FULL_FIDELITY_PLAN.md`, `FULL_FIDELITY_DELTAS.md`,
`fullfidelity/PHASE0_LOG.md`) are exhaustive working records, not a fast way to answer "what is
the state of the port right now" — that is this file's job. Read `Project_Summary_and_Conclusions.md`
next for the one-page version.

## Project sources (the only starting information given by the project lead)

Owner: G. Tamkin. Recorded 2026-10-07; findings and limits of the review are in `DOCUMENTATION_REVIEW.md` (section 13 of the paper was not readable; read it before relying on it).

- ROCKE-3D 2.0 paper (Tsigaridis et al. 2025, GMD 18, 5825), anchored at its section 13: <https://gmd.copernicus.org/articles/18/5825/2025/#section13>
- NCCS publication supplement, run P2SAoM40_003 (100 annual `aij`/`aijl` files, `ANN4000`-`ANN4099`): <https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/publication-supplements/Tsigaridis2025GMD-planet_2.0/P2SAoM40_003/>
- Cited by the paper for code and data (found via search, not read in the paper): Zenodo "ROCKE-3D v2", DOI 10.5281/zenodo.14721184: <https://zenodo.org/records/14721184>

## Handoff (refreshed 2026-10-07, about 09:30 EDT; the newest first-hand state of the port)

**Branch `full-fidelity-port`; the remote has everything through `4afc426`** (D158-D178, the sharded runner, the Reports documents and the `gpu/` helpers). Last full regression (sharded, 4 pinned cores, 101 jobs, 50 min under load) on a clean export of `acbdb3d`: **2,953 passed, 2 skipped, 0 failed**; the code added afterwards (D176 `33197fa`, D178 `a8c5294`) are new files only and their own test files passed separately (7 and 10). The 2 skips are data/environment-gated tests (one passes when `F3_CHAINED_NPZ` is set).
**One agent is still working:** D180 (stage 1 of the JAX-driven coupled step, the atmosphere half): `jax_atm_step.py`, `jax_atm_step_run.py`, `jax_atm_step_cmp.py` are untracked and UNVERIFIED; no ledger entry yet. Finished and committed: D175 (speed), D176 (radiation-derived columns), D177 (zenith, PBL carry, ice and land columns), D178 (driver skeleton, adopted as the host layer), D179 (`Reports/JAX_COVERAGE_MATRIX.md`). The 15-minute heartbeat (session-only cron) is running; it dies with the session.

**DIRECTION (decided by the project owner 2026-10-07):** (1) end-to-end JAX first, validated components second; (2) success = one JAX-driven coupled step compared with the real Fortran, then a multi-day run; (3) a Fortran radiation callback is acceptable but must be noted in every result that uses it; (4) match `P2SAoM40` first. Month-scale F3 work is paused. Reproducibility hazards that apply to every bitwise comparison (D175): the existing code gives different results on 1 and 3 cores (use the same core affinity and thread count on both sides); do not use a warm JAX compile cache for validated runs. See `RECONCILIATION_PLAN.md` and `PROVENANCE_MANIFEST.md`.

### What exists and how well it is validated

| Area | State | Evidence (ledger `FULL_FIDELITY_DELTAS.md`) |
|---|---|---|
| Ocean core | All live pieces ported and batched/JAX; a **chained whole-ocean step** (4.05 s/step CPU) matches the real dumps on 3 dates x 12 steps (exit error <= 1e-10 on tracers/moments, 2e-9 on velocities, pressure bitwise). ODIFF and the OPFIL2 coefficient setup are ported (D137-D138) | D33-D88, D118-D120, D137-D138 |
| Sea-ice dynamics | VPICEDYN batched (55x) and in JAX | D87-D88 |
| Atmosphere dynamics | Every live piece ported; a **chained dynamics step is bit-for-bit with the real model on 18 steps** when the Intel libimf `pow` is used; JAX version of the whole step 2.7x faster on CPU (bitwise vs numpy-pow) | D90-D123, D139-D144 |
| Clouds / convection | `get_dq_*`, helpers, MASS_FLUX, LSCOND, MSTCNV, the CONDSE column chain; batched across columns (~5 s/step instead of ~111 s); JAX versions are bit-identical to the numpy batch but not faster on CPU | D89-D132, D145-D147 |
| Land / surface | Ported earlier (D4-D32); **two real porting errors found and fixed on 2026-10-06** (precipitation conditioning D135, vegetated-tile irrigation D136; every output of 753 cells now matches the real record to 2e-13 of scale); D158 explained and fixed the last mismatches (ffg dump truncated at 11 sub-iterations; applied 2026-10-06, abetad bounded at 5e-6 in the 4 stiff-cell files only, other tolerances unchanged) | D135, D136, D158 |
| Chained atmosphere step, F1 gate | libimf, step 0: **MET on all 3 dates with the recorded land patch; with our ported land code MET on nov26, PARTLY MET on dec01/jan01** (worst field 9.9e-9 of scale). Without libimf not met (cloud threshold flips in 3-5% of columns) | D127-D129, D136b |
| One model day (nov26, 54 steps) | Open loop with recorded radiation (D149-D151) and **free-running radiation through the radiation server** (D155-D157): T, U, V within the real model's own chaos level at all 54 steps, Q/P within or near, cloud condensates near the top of the level (<= 1.95x), never beyond 2x; global-mean radiative flux differences < 1 W/m2. Surface replayed from the real run; one start state; 5 real one-ulp members | D149-D151, D155-D157 |
| Radiation | **Never ported (SOCRATES is third-party).** A "radiation server" (scratch build of the real ModelE objects running the unmodified RADIA from a packet file) reproduces the recorded radiation bit for bit on nov26 steps 0 and 5; 27-190 s per call (file exchange, re-runs the model up to the radiation step) | D148, D152-D154 |
| Persistent radiation server | Real RADIA, SOCRATES untouched. 21 of 22 output fields bitwise vs recorded real radiation (nov26 5 steps incl. a day boundary; dec01, jan01 first radiation step), ~11 s per call (was 27-190 s), call-order independent on nov26; the free-radiation nov26 day through it equals the one-shot day bitwise (1,192 arrays); radiation time 158 s vs 1,194 s. AIJ needs the real trajectory's seed | D159-D162 |
| F3 diagnostics (AIJ-style) | 257 AIJ + 4 AIJL columns match the real model's own 54-step accumulation to <= 5.3e-14 (tolerance 1e-12, not loosened); ~226 of 483 changed columns NOT accumulated (surface winds/state, cloud columns, budgets); reference `ff_data/nov26_day/real_acc54_nov26.npz` made with the real binary | D163 |
| Sea-ice advection (ADVSI) | New numpy port, **bitwise (0 differing elements) on all 150 real calls** dumped from nov26/dec01/jan01 (new instrumentation `ICEDYN_DRV_advsi.f.patch`, units 1600/1601). Carries its own binary128 `Ti2b`. Fixed-SST branch and the north-pole-box crunch are not exercised | D166 |
| River routing (RIVERF) | Original RIVERF ported; flows bitwise at step 0, <= 1e-12 of scale at later steps (6 steps, nov26); confirms RIVERF as the cause of the D164 lake errors. Includes the REAL*4 `DZDH1` finding | D167 |
| Sea-ice dynamics glue (DYNSI) | Input assembly and post-processing around VPICEDYN ported: assembled inputs bitwise, outputs 1e-10..1e-11 (residual is the existing VPICEDYN); computed odmui/odmvi/ustar change the free-loop drift by ~1e-3 of the error itself | D168 |
| Ent vegetation (stage 1) | Per-iteration exports (cnc, betadl, lai, ...) ported from the real Ent source: trans_sw/lai/ipp and the per-call exports bitwise, cnc/ci/gpp within 1e-15 relative (non-bitwise cases mostly in the first step after a restart, cause unexplained); daily LAI/albedo update bitwise across the nov26->nov27 boundary. **Wired into the chained land path (D171): closed mode on the 54-step day within 1.5e-14 of the record, iteration counts equal on all 81,324 calls.** Carbon/soil (`soil_bgc`, `Respauto_NPP_Clabile`), the first-day `set_vegetation_data` and month-scale evolution are NOT ported or validated | D169, D171 |
| Surface loop v2 (6 steps, nov26) | ADVSI + RIVERF + DYNSI all computed, coupled path rewired: free loop step-5 errors ice msi2 1.3e-8, ocean exit uo 1.0e-7 / vo 2.2e-7, lake ~1e-16, land-ice 0 (a D164 `EDIFS` defect found and fixed in v2); coupled atmosphere + closed surface: T/Q/U/V/P within or near the 5 real members' spread (max ratio 1.22, never beyond 2x). One start state, 6 steps. Step-0 ice difference in one cell (msi 1e-6 at (65,38)) unresolved | D164, D170 |
| Sea-ice REAL*16 | Only `Ti`/`Ti2b` in SEAICE.f use REAL*16; every other port uses float64. A binary128 emulation makes seaice_to_atmgrid and ADDICE fully bitwise; GROUND_SI improves but is not bitwise (732 of 4,524 rows). Not yet applied to the existing modules (performance and ownership decision) | D173 |
| Real JAN1950 ensemble | 8 real members (ctrl + 7 one-ulp perturbed), 85 min each; the ctrl run reproduces the stored real month bitwise (only `itimee`/`cputime` differ). Noise floor e.g. t_500 global-mean sd 0.043 K, grid-point sd 0.56 K; prec 0.0105 / 0.89 mm/day; srnf_toa 0.18 / 7.1 W/m2. 8 members = about +-25% on a std; one season | D165 |

### Caveats that matter when reading any result
- **Bitwise results need the Intel libimf `pow`/`exp`** (`intel_libm_ff.py`, a ctypes bridge to the real build's runtime; only on hosts that have it). With numpy/glibc the dynamics differ at ~1e-13 and cloud columns near thresholds flip (3-5% of columns), so multi-step comparison against one real trajectory cannot be bitwise beyond step 1: acceptance is statistical (the noise-floor method of D151/D157).
- **JAX tests are flag-sensitive.** XLA:CPU fuses `a*b+c` into FMAs and rewrites `x/c`; bitwise agreement needs `--xla_cpu_max_isa=AVX` and `--xla_disable_hlo_passes=algsimp` set before JAX is imported (`dyn_jax_env.py`, `clouds_jax_env.py`). Those test files cannot share a process with the rest: run the whole suite with **`fullfidelity/run_all_tests.sh`** (the three files sit behind `tests/conftest.py` and run each in a fresh process with `RUN_XLA_FLAG_TESTS=1`).
- **Recorded (not ported) inputs:** radiation (SOCRATES, except through the real-RADIA server); in the one-day runs Ent exports (an opt-in computed path exists, D171), the surface/ocean/ice/lake state (computed only in the 6-step surface loops), the AG2OG/IG2OG fluxes and the straits start state in the ocean chain; in the surface loops also the ffg land forcing columns, TRUP_in_rad, PBL profile columns, MMST, the ADVSI geometry vectors and the irrigation demand (reconstructed from the recorded flux).
- **No GPU on this node** (12 CPU cores, JAX sees only `CpuDevice`); GPU speed-ups, the project's stated purpose, cannot be measured here.
- **One start state (nov26) for every one-day and surface-loop result**; the persistent radiation server was exercised on nov26 (5 steps) and the first radiation step of dec01/jan01; the F3 diagnostics on one 54-step window.
- Open-loop or surface-replayed runs validate dynamics, clouds, land code and the day boundary, not radiation-surface feedback.

### In flight and uncommitted (check before doing anything else)
1. **Nothing is unpushed** as of 15:25 (remote `4afc426`). Run the sharded regression again after D180 lands (`fullfidelity/run_all_tests_sharded.sh`, from a clean export of HEAD, pinned cores).
2. **D180 agent (JAX atmosphere step, stage 1)** still working; its files are untracked and unverified. It must report against `ACCEPTANCE_CRITERIA.md` (which section-1 items its stage satisfies, every recorded input, every non-JAX stage) and compare with the NumPy chain in libm mode on identical cores.
2b. **GPU:** this node has no Slurm and no route to Discover. `gpu/run_gpu_job.sbatch` (one-command version of the owner's manual session), `gpu/collect_gpu_results.py`, `gpu/gpu_probe.py` and `gpu/smoke_test_env.py` are committed (sbatch and collector are drafts never run on the cluster; probe and smoke test run on CPU here). Open: which container has CUDA-enabled JAX, where `benchmark_all.py` lives (not in this repository), and where the data would sit on Discover (restart 532 MB; the minimal 6-step file list is being derived by tracing real runs).
3. **Untracked stray file** `fullfidelity/instrumentation/ICEDYN_DRV_advsi.f.patch.new` (next to the real ADVSI patch; origin unclear, not created by the parent); inspect and delete or keep. `../full-fidelity-port-new/` is an untracked sibling directory of unknown origin; not touched.
4. **Known defects left in place (documented in the ledger):** the D164 and D166-D168 loop scripts still carry the `EDIFS` land-ice bug (use `surface_loop_v2`); `seaice_core_ff.Ti/Ti2b` are float64 (proposed binary128 diff in D173, not applied); the first-step Ent ulp residual (D169/D171) is unexplained (a small dump on unused units 1470-1479 is proposed, not built).
5. **Open observation, not a validated result:** in the chained 54-step day against the 5 real members the Ent-computed run has QCI at worst ratio 1.62 (level 21: 2.14x) versus 1.06/1.11 for recorded-Ent runs; one realisation cannot separate Ent from chaos (D171).
6. **Owner decisions:** (a) DECIDED 2026-10-07: Ent exports stay recorded for runs up to one day and the Ent port starts now (done: stage 1 and its wiring); whether carbon outputs are part of the F3 acceptance is NOT decided. (b) Still pending: the 'intended science use' line of `Reports/GOAL.md`. (c) Whether to apply the binary128 `Ti/Ti2b` diff to `seaice_core_ff` (speed vs bitwise).

### Next steps, in priority order (re-ordered 2026-10-07 after the owner's decisions)
1. **DONE: JAX coverage matrix and design** (`Reports/JAX_COVERAGE_MATRIX.md`, A5): 26 rows; build stages S0-S8 each with a gate. Owner answers 2026-10-07: the headline fidelity comparison uses a labelled libimf host callback (plus a libm-mode report; speed measured without it); three jit units plus host calls count as JAX-driven; D176 files adopted as host layer, D178 after verification; the GPU host is still to be named. First structural prerequisite: a fixed-shape tile layout with a validity mask (stage S1).
2. **One JAX-driven coupled step** (A6): compose the existing validated JAX stages under as few `jit` boundaries as practical, list every recorded or Fortran-served input, compare with the NumPy chained step (C1) and with the real Fortran (C2) on the criteria in `Reports/ACCEPTANCE_CRITERIA.md` (APPROVED by the owner 2026-10-07; open: name the GPU host). Radiation through the persistent server, noted as a hybrid component.
3. **A short multi-day JAX-driven run** from the same start, with the same notes.
4. Finish and wire the closures of recorded inputs that the coupled step still takes from records (D176 radiation-derived columns; D177 wiring into the coupled path; D178 driver skeleton), because each removed record shortens the list the step must declare.
5. Speed on the intended hardware (needs a GPU host; this node has none); the D175 parallel variants are CPU-only.
6. Paused until the above: more F3 columns, month-scale runs and ensembles, the Ent carbon half, the 100-year climatology comparison.

### Remaining effort (estimates, not measurements)
- The 2026-10-06 figure of ~100-170 h to the one-month F3 comparison included items now done (persistent radiation server, real ensemble, ADVSI/RIVERF/DYNSI, Ent stage 1 and its wiring, the F3 accumulator core). It is therefore too high, but no new total has been derived.
- Agent-provided figures for what remains: Ent water/energy side about 9-14 h (first-day `set_vegetation_data` 5-8 h, month-scale validation 4-6 h); carbon half and the carbon part of the daily update bring Ent to about 30-45 h in total (optional, see decision 6a); batched/JAX Ent 15-25 h (optional).
- NOT estimated: the remaining diagnostic columns (D172 in progress), the longer-window surface loop and other dates, AG2OG/IG2OG and the straits start, the model-month driver and its first month-scale result, and the GPU measurement (needs hardware). Earlier estimates in this project were revised upward once; treat any total as provisional.

### Environment and conventions
- Python: `/home/gtamkin/.conda/envs/graphcast-env/bin/python` (has jax 0.5.3 and omegaconf; the default python lacks omegaconf). Run tests from `fullfidelity/`.
- Full regression, faster: `fullfidelity/run_all_tests_sharded.sh` (4 concurrent jobs by default; `CPUS=8-11` pins it; one job per test file, the radiation-server files serial, the three XLA-flag JAX files in fresh processes). Measured 2026-10-07: 2,878 passed, 1 skipped, 0 failed in 34 min on 4 cores vs about 83 min for the serial `run_all_tests.sh`; the aggregate totals must equal the serial counts. `pytest-xdist` is not installed.
- Real-model dumps: `/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/` (about 79 GB: dates `nov26`, `dec01`, `jan01`, the 54-step `nov26_day`, `_pristine_restarts`). The original ModelE tree is `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0` (read-only; instrumented builds are scratch copies per `fullfidelity/instrumentation/build_and_run.md`).
- Instrumentation units used so far: 1050-1440, 1500-1503 (persistent radiation server) and 1600-1601 (ADVSI dumps); 1470-1479 are proposed but unused (see the build notes per delta). `cp` is aliased to `cp -i` here: use `\cp`.
- Scratch builds (`mE_*`) live under the session scratchpad (`.nccstmp/...`), which is temporary; everything needed to rebuild is in `instrumentation/*.patch` and `build_and_run.md`.
- The `claude` binary is not on PATH; it is bundled in the VS Code extension directory (symlink it into `~/bin`), `screen` is installed, `tmux` is not.
- Standing conditions (below) still apply: SOCRATES is never ported or modified; no reduced ocean without surfacing it as a decision; record, do not hide, cut corners; do not pause for confirmation mid-port.

## Detailed log (accumulated, historical)

Everything below is the accumulated per-delta log, kept for the record. Where it differs from the Handoff above (stale statuses such as "uncommitted at handoff", "next step", older estimates and test counts), **the Handoff above is current.** Newest entries are lower in the section "Current state"; the entries before it were written first.

### Entries written before the 2026-10-05 handoff (D54-D58 and earlier notes)

**D58 (2026-10-03, ready to commit): `KVINIT` ported — exact copy on all 198 real snapshot checks.**
`KVINIT` (OCNKPP.f:1315-1359) saves the pre-source surface snapshot the KPP step reads. A pure copy
with no arithmetic, validated by exact equality against 18 real per-step dumps (3 dates x 6 steps).
19 new tests. Full suite passed (785 passed, 0 failed); the skip count rose from 6 to 23, which is
being investigated before the commit (the D58 tests themselves do not skip).

**D57 (committed `4c93716`, 2026-10-03): `REDUCE_FIG` ported — bitwise-exact on all 1,750,248 real
calls.** The significant-figure reduction applied to `OCONV`'s live main-grid moments
(OCNKPP.f:2843-2847). 524,741 of those calls actually changed the value, so the check is not vacuous.
22 new tests. Full suite: 766 passed, 0 failed.

**D56 (committed `fca7a79`, 2026-10-02): momentum `OVDIFF` ported — bitwise-exact on all 420,444
real calls.** The velocity counterpart of D55's `OVDIFFS` (UL 281,808 calls, ULD 138,636 calls).
Structural differences from `OVDIFFS` ported verbatim. 19 new tests. Full suite: 744 passed.

**D55 (committed `ec5a323`, 2026-10-02): `OVDIFFS` + plain `TRIDIAG` ported — bitwise-exact on all
152,022 real calls.** Tracer vertical-diffusion solver (G0M/S0M) with non-local transport. The DTP4
argument is structurally zero for this build (its only setter is in the dead `OCN_GISS_SM` block).
20 new tests.

**Note on instrumentation patches for D55–D58:** the scratch model tree was removed and rebuilt from
pristine source, so the patch files for these deltas were written from the inserted text, not
generated by `diff`. Their headers say so. Verify before reapplying.

**D54 (2026-09-30 ~18:00): `KPPMIX`+`z121`+`kmixinit`+`init_solar` ported — first piece of the KPP
vertical-mixing scheme, and the first delta in this project not validated bit-for-bit (with a
fully diagnosed reason).** `KPPMIX` sits inside `OCONV`'s fixed-point HBL iteration (up to 4 calls
per column per step); rather than port that outer loop, instrumented the real `CALL KPPMIX` site
directly and dumped every real call as its own record — a new per-call dump shape for this
project, 76,011 records captured across the standard sweep. Found and fixed two real
single-precision-literal bugs (`OCNKPP.f`'s `**(1./3.)` and `sqrt(0.2/...)` write their literals
without a `d0` suffix, so Fortran computes them in single precision before promoting to double —
confirmed via a standalone `ifort` test program reproducing the real dumped constants bit-for-bit
only once fixed). Even after that fix, ~0.02% of the lookup table's cells differ from a real
one-time table dump by exactly 1 ULP — confirmed (via both `pow` and `exp`/`log` recomputation) to
be an inherent IEEE 754 gap: `pow`/`exp`/`log` are not required to be correctly rounded, unlike
`+`,`-`,`*`,`/`,`sqrt`. Quantified the consequence precisely: max residual ~5e-6 across all 76,011
real calls, `KBL` (the integer output) never mismatches once. Validated at this suite's ordinary
`atol=1e-6`. 25 new tests.

**D53 (2026-09-30 ~17:45): correction — `bldepth` is dead, not live; `KPPMIX` inlines its own
boundary-layer-depth logic.** Caught while reading `KPPMIX`'s full body to prep for D54: both of
`bldepth`'s call sites are gated `#ifndef OCN_GISS_TURB -> KPPMIX / #else -> bldepth`, and
`OCN_GISS_TURB` is confirmed undefined, so `KPPMIX` is the live branch both times — D52 had missed
the surrounding `#ifdef` and called `bldepth` live. `KPPMIX`'s own body explains why: it has the
entire boundary-layer-depth and boundary-layer-mixing-coefficient algorithms inlined directly.
Revised main-grid live-scope estimate: ~2,235 lines (`bldepth`'s 225 lines removed).

**D52 (2026-09-30 ~17:20): scoping — `OCNKPP.f`'s KPP vertical-mixing scheme sized end to end.**
With Gent-McWilliams closed, read all 3,714 lines of the file the largest remaining item in Stage
2. Corrected D41's coarse subroutine boundaries and found four entirely-dead subroutines
(`get_kvtdiss`, `get_gradients0`, `wscale`, `swfrac` — 362 lines) plus a compile-time-dead `ddmix`
(`LDD=.false.`). Traced the real call graph: `OCONV` (main-grid driver, confirmed live),
`KPPMIX` (the actual diffusivity computation), `STCONV` (straits analog, live but blocked on
unscoped `OSTRAITS.f`). Revised live-scope estimate ~2,460 lines (later corrected to ~2,235 by
D53) — ~3.5x the entire Gent-McWilliams family, confirmed the single largest remaining physics
item in Stage 2.

**D51 FINISHED (2026-09-30 ~17:10): `GMFEXP`+helpers ported — the Gent-McWilliams mesoscale-
mixing family (D46-D51) is now fully closed.** The actual skew-flux application to `G0M`/`S0M`
(`GMFEXP` + `computeFluxes` + `wrapAdjustFluxes`/`addFluxes`), the largest delta of this family.
**Bitwise-exact on all 4 checked fields, both calls (G0M/S0M), all 3 dates, first try — no bugs
found.** One real ambiguity sidestepped rather than guessed: `MO` (the mass field `GMFEXP` reads)
couldn't be conclusively traced back to D45's `OADVT2` output from source alone, so it was
recorded directly as a real input instead. Confirmed from a close read (not assumed): `GMFEXP`'s
own loop never touches the pole rows, and the real Fortran's `TZM`-update code is commented out
in full (`TZM` is never actually written despite being `INTENT(INOUT)`) — both became dedicated
regression tests. 15 new tests. `OCNMESO_DRV.f`+`OCNGM.f`'s entire real per-step live path (the
"skew-GM" branch) is now fully ported and validated; `OCNTDMIX.f` (2,030 lines) and
`GET_PSI_DIAG` confirmed dead/diagnostic.

**D50 (2026-09-30 ~16:15): `GMKDIF`'s remaining coefficients ported — bitwise-exact first try,
no bugs.** Applied D49's pole-inclusive J-range finding directly rather than rediscovering it,
and reasoned through (then confirmed) why the real source's domain-decomposition-only "halo"
extension block never fires in this rundeck's serial execution. Found and recorded a new
not-yet-ported dependency, `KPL` (mixed-layer-depth index, set by `OCNKPP.f`). 12 new tests.

**D49 FINISHED (2026-09-30 ~15:00): `ISOSLOPE4` ported — first piece of the Gent-McWilliams
mesoscale-mixing scheme itself.** The isopycnal-slope-derived diffusion coefficients (24 output
arrays), embarrassingly parallel per-cell — unlike almost everything else in Stage 2. Real inputs
are exactly D47's `densgrad` outputs plus D47's constant `K3D`, so no new input instrumentation
was needed. **One real bug**: the main loop, unlike almost every other Stage-2 routine, genuinely
includes the North Pole row (J=JM) — a first draft assumed the usual pole-exclusion convention and
silently left it zero, which happened to match on 16 of 24 output fields (where `RHOX`/`RHOY`
were also zero there) while exposing itself on the other 8 via suspiciously round differences.
Fixed by extending the loop; bitwise-exact on all 24 fields, all 3 dates. 9 new tests.

**D48 (2026-09-30 ~14:30): scoping-only — read `GMKDIF`/`ISOSLOPE4`/`GET_PSI_DIAG`/`GMFEXP`
+ helpers in full, found two more real scope reductions.** `GET_PSI_DIAG` confirmed purely
diagnostic (skip). More importantly: `QCROSS` (gating roughly half of `GMKDIF`'s and `GMFEXP`'s
coefficient/flux logic) is **always false** for this rundeck — `ocnmeso_drv` calls
`gmkdif(k3d,1d0)` with `RGMI_in` hardcoded to `1d0` in the source itself, not a tunable parameter.
Revised remaining estimate down to ~700 lines.

**D47 (2026-09-30 ~14:00): `ocnstate_derived` + `densgrad` + `get_1d_mesodiff` ported —
bitwise-exact, first try, no bugs.** Cell-centered thermodynamic state and density gradients that
feed Gent-McWilliams. Corrected a D46 assumption along the way: `ocnstate_derived` is called
twice in the source, but the first call is gated by `#ifdef TRACERS_OceanBiology`, not defined for
this rundeck — confirmed via the real dump (1 record, not 2) before writing any port code.
Reused D40's already-validated `ODHORZ0` `DH3D` output directly rather than re-instrumenting it.
One cosmetic cleanup (not a correctness bug): the same North-Pole `m_active` mask used since D40,
applied here to silence a harmless-but-noisy transient divide-by-zero. 15 new tests.

**D46 (2026-09-30 ~13:45): scoping-only — corrected a backwards D41 assumption about the
mesoscale-mixing family.** `CONSTANT_MESO_DIFFUSIVITY` does **not** select a simplified path; it
only fixes the diffusivity coefficient fed into the full Redi/Gent-McWilliams skew-flux scheme,
which still runs in full. Found `OCNTDMIX.f` (2,030 lines, the largest file in this family) is
**entirely dead code** for this build (`use_tdmix=0`). Revised live-code estimate: ~1,455 lines
remaining (later cut further by D48's findings).

**D45 FINISHED (2026-09-30 ~13:40): `OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2` ported and validated —
`OCNDYN2.f`'s entire real per-step dynamical core is now ported.** The long-timestep advection of
potential enthalpy (`G0M`) and salt (`S0M`), via `OADVT2`'s Strang-splitting dispatcher (X
half-step, Y, Z, X half-step again). The largest, most intricate delta this session (~570 lines).
**Three real bugs found, each via its own debugging cycle**: (1) `OADVTX2`'s `mudt` array is a
single persistent array with genuinely stale-by-design semantics (indices 1,2,IM unconditionally
refreshed every pass, everything else only within that pass's own active segments); its segments
turned out to be linear, not circular ("wraparound is disabled" in the real segment-builder), with
a single-cell-segment skip missed on first read — fixing this (a precise segment-based rewrite)
took errors from ~1e14 down to a widespread but small (~1e-5 relative) residual. (2) `MMI` (the
mass `OADVT2` actually advects) had been re-derived as `MO0*DXYPO(J)`, which doesn't match the
real `MMI` — a persistent module array `ODHORZ0` only partially overwrites — fixed by dumping it
directly instead. (3) `OADVTZ2` needed the same "`nbyzm` restricts the North Pole row to I=1 only"
mask established in D40 — found by adding temporary debug instrumentation to bisect the mismatch
to exactly this routine, exactly the pole row, then removed before finalizing. A fourth, smaller
fix: `np.sum()`'s pairwise reduction rounds differently from ifort's sequential `SUM` for a
72-term pole-average. Bitwise-exact on all 9 checked fields, all 3 dates, after all four fixes.
22 new tests. JAX deferred (dynamic segment structure + pole masking make this a poor first
batching candidate).

**D44 FINISHED (2026-09-30 ~09:40): `ODHORZ`'s `SMU`/`SMV` accumulation completed, closing a real
gap found while scoping D45.** `OADVT2` reads `SMU`/`SMV`/`SMW` as its mass-flux inputs; `SMW` was
already ported (D43), but `SMU`/`SMV` turned out to be accumulated inside `ODHORZ` itself — real
per-step physics D42 never captured since nothing D42 validated needed it. Traced the accumulation
scheme by hand (confirmed `NOCEAN=1` for this rundeck, so it reduces to "once per `OCEANS` call")
before writing any code. Bitwise-exact, first try, both the per-call replay and the full 5-call
chained accumulation against a new ground-truth dump. 13 new tests.

**D43 FINISHED (2026-09-30 ~06:35): `OFLUXV` + `OADVUZ` ported and validated** (long-timestep
vertical mass redistribution that rescales layer-1/bottom-layer mass to restore the L13
fractional-thickness profile, plus the embedded simplest-upstream vertical advection of U/V).
**Corrects D41's scoping**: `OFLUXV` does NOT call `OPFIL2` — re-reading the routine start-to-end
found no such call; the `opfil2_coeffs` module just sits adjacent in the source file. `OPFIL2`
itself remains unported (open item below). **Two real bugs found and fixed**: a `DXYPO(J)/DTOLF`
bookkeeping error (the Fortran's mass-flux accumulation carries a factor that only partially
cancels when averaged onto U/V-points — first draft dropped it entirely), and a JAX-only `OADVUZ`
NaN bug (dense unmasked vectorization divided 0/0 at genuinely-inactive columns that the Fortran's
gated loop simply skips — fixed with an explicit active-cell mask threaded through the
`lax.scan` carry). Confirmed via honest-scoping tests that this rundeck's real bathymetry has no
single-layer (`LMM==1`) columns on any of the 3 test dates — cross-checked against a synthetic
construction instead. Both plain-Python and JAX ports float64-tolerance exact on all 3 dates. 18
new tests. Full regression: **594 passed, 0 failed.**

**D42 FINISHED (2026-09-29 ~22:15): `ODHORZ` ported and validated — first "new architecture"-
scale delta closed.** The actual horizontal momentum + mass-continuity solve `ODHORZ0` (D40)
preps pressure/EOS inputs for — comparable in scope to D29's ADI solve, the largest delta since
then. `OPFIL2`'s two per-layer outputs recorded directly (same pattern as D40's `VOLGSP`),
decoupling this from the `OPFIL2`/`AVR`-file dependency. **Two real bugs caught before any
validation run**: a missing `HOCEAN` recording for `OGEOZ`'s init (caught mid-writing), and
`OPBOT`'s cross-layer accumulation incorrectly reset each layer instead of accumulating (caught
by re-reading the source). Float64-tolerance exact on all 15 real call records, first full run
after fixes. `OMEGA` (a genuine runtime planet parameter) validated empirically at Earth-standard.
JAX vectorization deliberately deferred as its own follow-up (GHY-lesson discipline — this
routine is large/multi-physics enough to deserve dedicated batching care). 16 new tests. Full
regression: **576 passed, 0 failed.**

**D41 (2026-09-29, ~21:00): a scoping sweep, not a port — resolved essentially all remaining
Stage 2 ambiguity.** Read `ODHORZ` and `OADVTX2` (dispatcher `OADVT2` + real physics) in full and
confirmed both are genuine "new architecture"-scale items (comparable to D29's ADI solve);
`OADVTX2` specifically has dynamic Courant-number substepping that may never fire in a real test
window, deliberately deferred rather than rushed. **Confirmed `OCNQUS.f` (1,846 lines) is
entirely dead code** for this rundeck — gated by `USE_QUS`, which defaults to 0 and is not
overridden; `OCNDYN2.f`'s own simpler `OADVT2` family is the live one. Surveyed `OCNKPP.f`
(confirmed live core is `KPPMIX`+`OCONV`, ~2,800 lines, likely the single largest remaining item)
and `OCNMESO_DRV.f` (confirmed `CONSTANT_MESO_DIFFUSIVITY` is live, a simpler path than full
Gent-McWilliams; `OCNGISS_TURB.f`/`OCNGISS_SM.f`, 1,252 lines, confirmed entirely dead). Revised
the Stage 2 line-count estimate down from ~19,000+ to ~17,150+.

**D40 FINISHED (2026-09-29 ~19:50): `ODHORZ0` (pressure/equation-of-state prep) ported and
validated.** Caught a real North Pole masking bug via the dump comparison itself (`nbyzm`
hard-restricts J=JM to I=1 only, independent of `LMM`'s value at other longitudes there); fixed
and pinned with a dedicated regression test. Confirmed `USE_OPGFQ=0` for this rundeck. 17 new
tests.

**D39 FINISHED (2026-09-29 ~19:15): polar UOD/VOD relaxation block + `polevel()` ported and
validated.** Float64-op-order exact against real Fortran, both ports, all 3 dates, first try. 21
new tests.

**D38 FINISHED (2026-09-29 ~15:07): `OBDRAG2` (implicit bottom-layer current drag) ported and
validated.** Float64-op-order exact, both ports, all 3 dates, first try. `FULL_FIDELITY_DELTAS.md`'s
D38-D42 entries have the full detail.

**D37 FINISHED (2026-09-29 ~11:50): `OCOAST` ported; permanent fix for a restart-file trap that
had cost real time across D27/D36/D37.** GISS ModelE's restart reader picks whichever of
`fort.1.nc`/`fort.2.nc` has the *later* itime, so a prior run in a reused scratch directory can
leave the "wrong" one looking pristine. All three test dates' pristine restarts are now archived
permanently at `ff_data/_pristine_restarts/fort1_{nov26,dec01,jan01}_itime{33312,33552,17520}.nc`
— restoring before a rerun is a three-line copy, not a debugging cycle. See `build_and_run.md`.

**D36 FINISHED (2026-09-29 ~08:40): major scope correction — `OCNDYN.f`'s legacy ocean
dynamical core confirmed entirely dead code.** `OCNDYN.f`'s own leapfrog-dynamics driver
(`OCEANS_old`) is commented out in full; the live driver is a complete rewrite in `OCNDYN2.f`
(`SUBROUTINE OCEANS`, called from `OCN_DRV.f:41`). Every call site of `OCNDYN.f`'s
`OFLUX`/`OADVM`/`OADVV`/`OPGF`/`OVtoM`/`OMtoV`/`OSTRES`/`OBDRAG`/`OPFIL` project-wide was checked
and found dead — do not port these from `OCNDYN.f`. `OSTRES2` (the live equivalent) ported and
validated same delta. `GLMELT` (glacial meltwater) scoped and **deferred**: fires only once per
calendar day, not confirmed within the existing 6-step/3-hour test windows.

**Stage 2 (ocean core) opened 2026-09-29 (D33): `PRECIP_OC`, `OSOURC`, `GROUND_OC`'s below-
freezing sweep all validated (D33-D35)** before the D36 scope correction above. Ocean grid
confirmed same resolution as the atmosphere (IMO=72, JMO=46) — a major de-risking finding.

**Stage 1 (ice dynamics) closed 2026-09-29 (D26-D32): `PRECIP_SI`/`PRECIP_LK`/`PRECIP_LI`,
`DYNSI`/`VPICEDYN`/`FORM`/`PLAST`/`RELAX` (3 real bugs found and fixed), `CALC_APRESS`,
`seaice_to_atmgrid`, `UNDERICE`** all ported and validated bitwise/float64-exact against real
Fortran. `IRRIG_LK` deferred (external dataset dependency, not yet needed).

**Standing constraint, unconditional, from every session since it was set:** SOCRATES (the
radiation library) is third-party — **never port or modify it**; call it as a black box only if
radiation work is undertaken.

**Standing decision, twice user-confirmed:** the full faithful port of DYNSI (ice dynamics) and
the entire ocean numerical core (Stage 2, ~19,000 lines before D36's dead-code correction), no
reduced/mixed-layer approximation. This supersedes an earlier (2026-09-23, Track A) decision
recorded in `STATUS.md` *against* a full-fidelity port — that document is now superseded for
everything after D25; see "Document map" below.

### Current state (refreshed 2026-10-05, handoff)

**Committed through D77 (`db561d7`, pushed).** Ocean core numerics:
- D54-D61: bitwise-exact numpy ports (KPPMIX, OVDIFF/OVDIFFS, REDUCE_FIG, KVINIT, OCONV setup block,
  HBL scaling, mass bookkeeping, convergence, flux save).
- D62-D64: batched JAX ports, checked to 1e-9 relative (OVDIFFS, momentum OVDIFF, KPPMIX, setup block).
- D66: batched JAX OCONV HBL loop, validated on 18 real steps.
- D67-D73: straits step in batched JAX (STPGF, STADV with shared-cell copy, STCONV, STBDRA), chained,
  validated on 18 real steps. Seawater EOS from OFTABLE_NEW (`eos_jax.py`).
- D74-D75: OPFIL2 application, validated (3e-15). D76: OCONV loop computes EOS from the table.
  D77: OPFIL2 as batched 72x72 row operators (`opfil2_jax.py`).

**D78-D79 (ODHORZ batched numpy, then JAX layer body, `odhorz_jax.py`; see deltas ledger).**
Originally uncommitted at handoff: `odhorz_vec.py` (D78): batched ODHORZ,
layer loop kept sequential, grid vectorized. Matches the scalar port to ~3e-17 and the real Fortran
outputs to ~2e-7 on all three dates (`odhorz_vec_compare.py <date> <itime>`).

**D80-D81:** `oadvt_vec.py` batches OADVTX2/Y2/Z2 in numpy, bitwise identical to the scalar port and the
real Fortran dumps on all three dates (`oadvt_vec_compare.py`).

**D82:** `ocnmeso_vec.py` batches the OCNMESO inputs (bitwise identical to the scalar port, all three dates).

**D84:** `oadvt_jax.py` jits the OADVT2 sweeps (1e-16 vs numpy and real Fortran; same CPU speed, the X pre-pass
is the remaining cost).

**D83:** `gm_vec.py` batches ISOSLOPE4/GMKDIF/GMFEXP (bitwise identical to the scalar port, all three dates;
written by a subagent, re-verified here).

**D85-D86:** GM (`gm_jax.py`) and OCNMESO (`ocnmeso_jax.py`) under `jax.jit` (bitwise to 4e-16 vs numpy).

**D87-D88:** sea-ice dynamics VPICEDYN batched (`icedyn_vec.py`, bitwise vs scalar, ~55x) and under jit
(`icedyn_jax.py`, <=8e-12 vs numpy; no CPU speedup over numpy).

**D89 (first atmosphere/cloud delta):** `clouds_dq_ff.py` ports get_dq_cond/get_dq_evap, validated on 174,097
sampled real calls (10% sample, 6 steps per date), 0 branch mismatches, rounding-level agreement (99.2% bitwise).

**D90 (first atmosphere-dynamics delta):** `dyn_fltruv_ff.py` ports the end-of-DYNAM velocity filter chain
(FLTRUV, fltry2, angular-momentum fix); bitwise identical to the real model on all 18 calls, all three dates.

**D96-D98 (dynamics):** `dyn_aflux_ff.py` (AFLUX/ADVECM/MAtoP, with a transliterated FFT72 radix code),
`dyn_pgf_ff.py` (PGF), `dyn_advecv_ff.py` (ADVECV), validated on 90 real calls each (3 dates x 6 steps x 5 leapfrog
passes). AFLUX/ADVECM and ADVECV are bitwise identical; `x**KAPA` needs the Intel libimf `pow` for bitwise (via
`intel_libm_ff.py`, optional): with numpy pow, PK differs by 1 ulp in ~0.04% of cells and PGF outputs agree to
~1e-12 of field scale. Mode-0 topography patches, ADVECM exception paths and MPI-halo branches are not exercised.

**D94-D95 (dynamics):** `dyn_avrx_ff.py` (FFT/FFTI/AVRX radix port, bitwise), `dyn_isotropuv_ff.py`, `dyn_sdrag_ff.py`,
`dyn_geom_ff.py` (analytic geometry); bitwise on every recorded real call. SDRAG's wind-clamp branch is never
exercised (max wind 64-71 m/s vs 200) and is hand-tested only; the FFT72 code exists twice (D94, D96): consolidate.

**D99-D100 (dynamics):** `dyn_aadvt_ff.py`/`dyn_adv1d_ff.py` port the AADVT temperature-advection family (X/Y/Z
sweeps, per-row Courant nstep, adv1d, advection_1D_custom, limitq as importable functions); bitwise on 36 real calls
plus 21 forced multi-substep calls. In the real windows every nstep is 1 and qlimit is always false, so the
limiter branches are validated only against a standalone ifort build on 3,000 synthetic lines (2 of 2,295 differ
by 1 ulp, unphysical |fracm|>1); the y-direction `apply_limiter` is not ported.

**D91-D93 (clouds):** `clouds_helpers_ff.py` (PRECIP_MP, ANVIL_OPTICAL_THICKNESS, MC_CLOUD_FRACTION,
MC_PRECIP_PHASE, CONVECTIVE_MICROPHYSICS) and `clouds_massflux_ff.py` (MASS_FLUX, THBAR); validated on sampled real
calls (about 7 MB of dumps per date), 99.5-100% bitwise, worst relative 3.9e-14 (MASS_FLUX dqsum), zero branch flips
in MASS_FLUX. Several REAL(4) literals in CONVECTIVE_MICROPHYSICS must stay single precision. Branches never
exercised by real data (see ledger) are covered only by labelled line-by-line transcription tests.

**D101-D102 (dynamics):** `dyn_filter_ff.py` ports the SLP FILTER chain (SLP, SHAP1D, isotropslp, MAtoPMB) and the
energy functions (getTotalEnergy, conserv_PE, addEnergyAsDiffuseHeat); bitwise on all 18 calls with the Intel libimf
pow (numpy pow: 1-2 of 3,312 cells differ in 6 of 18 calls, worst 3.4e-13 mb); the SLP exp branch, the +-1.18% clip
and shap1 n>1 are never exercised (hand-tested only).

**D107-D109 (clouds):** `clouds_lscond_size_ff.py` and `clouds_lscond_ff.py` port LSCOND (size/optical-thickness
tail, main layer loop, CTEI, whole column call; VMP arm only). Bitwise on all three dates with the Intel libimf
exp/pow (1,902 columns/date for the whole call); with platform libm 99.96-99.97% bitwise, worst relative 2.8e-11 on
most fields, and up to 1.2e-7 absolute on sqrt-amplified cloud fields (CLEARA, CLDSAVL, CLDSSL, RHF). No branch
flips. Never reached by real data: negative TAUSS, the VMP stop_model arm, ER>ERMAX clip, QNEW<0, CTEI CKR>CKM and
FPMAX<=0 cycles, FMASS clipping (3 of these have synthetic tests).

**D114-D117 (dynamics coupling glue):** `dyn_glue_ff.py` ports CALC_TROP/tropwmo, COMPUTE_WSAVE, PGRAD_PBL, DISSIP,
calc_kea_3d, regrid_btoa_3d, recalc_agrid_uv, the energy-fix block and DAILY_ATMDYN (the latter validated by extending
the dump runs to 50 steps across the day boundary); bitwise on all calls, 3 dates. Never hit in real data (hand tests,
labelled): tropwmo no-limit-exit/no-valid-level/default-level paths, DAILY_ATMDYN early returns. Not hooked (dead):
the two ATM_DRV recalc_agrid_uv cold-start/relayer calls, cubed-sphere PGRAD_PBL, STRATDYN callers.

**D103-D106 (dynamics, moisture advection QDYNAM):** `dyn_aadvq_ff.py` ports AADVQ0 (flux-cycle counts), the
AADVQ/aadvqx/y/z/aadvqz_column/XSTEP/ZSTEP/checkflux sweeps and the QDYNAM glue; bitwise on all 18 real calls
(all 199 internal stages per call), on x8-scaled real-model stress runs reaching ncyc 4-6, and on a standalone ifort
build of the real QUS3D.f (7 synthetic-flux cases, labelled synthetic). Never reached by any model run: ncycxy>1
(harness only), ncycxy>ncmax, xstep too-many-steps (error flag unit-tested only). Unlimited *2 variants, halo
tables, AGC diagnostics and TrDYNAM not ported (dead/serial/diagnostic).

**D110-D113 (clouds, MSTCNV):** `clouds_mstcnv_ff.py` ports moist convection as a per-column function. With the Intel
libimf exp/pow it is bitwise on 6,528/6,540, 6,389/6,399 and 6,429/6,435 sampled column calls (nov26/dec01/jan01), zero
branch flips, all decision outputs identical. **With plain numpy exp/pow only about 80% of records are bitwise** and
39-48 records per date show threshold (cancellation) flips, with 15-26 records per date (0.2-0.4%) showing cloud and
optical-thickness deviations up to 7.8% of field scale where WCU2 is near zero at plume top: a faithful port on a
platform without libimf will therefore need statistical rather than bitwise acceptance for cloud fields. Never
exercised: the limitq limiter, IERR/LERR, SVLATL!=VLAT correction, plumes reaching L=LM, polar columns (not called);
ksub=2 subsidence in ~1% of calls.

**D121-D123 (chained dynamics step, milestone):** `dyn_step.py` chains all dynamics ports (5 leapfrog passes, AFLUX/
ADVECM/MAtoP, ADVECV, PGF, AADVT, QDYNAM, filters, SDRAG, SLP FILTER, glue, plus the DIAGA polar Q fill) in the real
order from the real state at the start of a step. **With the Intel libimf pow the whole DYNAM exit state and the
end-of-block state are bit-for-bit identical to the real model on all 18 steps (3 dates x 6)**, all 264 per-stage
series and 115 stage-boundary replays exact. Without libimf (numpy pow) most cells differ at rounding level (first
inexact quantity: PK=PMID**KAPA, 1 ulp in 916 of 2.38e6 cells; worst scale-relative 3.8e-13, PGRAD_PBL gradients up
to 2.7e-11). One chained step: 3.3 s warm (libimf), 2.4 s (numpy), CPU, not yet JAX. Gaps: the loop through
physics and atm_phase2 back to the next step, DIAGA/DIAGB bodies beyond the polar Q fill (no-feedback is an
inference), ATMSRF exports of MAtoPMB, NIdyn other than 4 and the 8-pass restart path.

**D124-D126 (clouds, CONDSE driver chain):** `clouds_condse_ff.py` runs the CONDSE column driver over all 3,170
columns per step, calling the validated MSTCNV and LSCOND ports; validated against the real exit state (68 fields)
on 3 dates x 6 steps. **With libimf: rounding level on all 57,060 columns except 26 columns (0.05%) over the 1e-12
field-scale bound from real threshold flips (worst: dec01 33555 column (67,22), T off by 3.6e-5); with platform libm
3.6% of columns exceed the bound and 32-44 of 68 fields fail per step** (statistical acceptance needed off libimf).
Python per-column speed is ~40 ms/column (~125 s per step): not usable for runs, needs the JAX/batched form.
Not ported: ISCCP/diagnostic bookkeeping, RIS/RI1/RI2 (unused), init_CLD parameter reads (recorded constants).

**D118-D120 (chained whole-ocean step, milestone):** `ocean_step.py` runs the live ocean step end to end (15 stage
boundaries) and matches the real dumps on all three dates over 12 steps each: worst full-step exit error from the real
entry state G0M 7e-12, S0M 3e-13, GX/GY/GZ 3e-11/6e-11/1e-10, SX/SY 7e-10/6e-10, MO 5e-15, UO/VO 2e-9, OPRESS
bitwise; a free-running 12-step chain stays within ~1e-8. 4.05 s per step warm (CPU). Found the OMEGA error (fixed).
Still recorded inputs: AG2OG/IG2OG fluxes, the OPFIL2 coefficient file, the straits start state, ODIFF (not ported;
recorded on the steps where it fires). Needs the OFTAB tables from the original ModelE support directory.

**D130-D132 (clouds, batched across columns, numpy):** `clouds_lscond_batch.py`, `clouds_mstcnv_batch.py`,
`clouds_condse_batch.py`. Profile of the per-column CONDSE (~111 s/step): MSTCNV 88%, LSCOND 3%. The batched CONDSE takes
~5 s per step in libm mode (7.6-8.3 s with libimf), about 22x faster, and reproduces the per-column ports bit for bit in
libm mode (0 of 62 exit fields differ, 3 dates). Against the real exit state: libimf 0 failing fields, 0 columns beyond the
bound; libm 149 columns (4.7%) beyond it (the D126 threshold flips, statistical acceptance needed). Still per-column: the
two pole columns, the `mc_new_ddrft_thetav=0` arm (asserted off), the Q-advection qlimit line loop. Diagnostic
accumulators are not reproduced. Remaining cost: ~4.3 s MSTCNV per step.

**D127-D129 (chained atmosphere step, F1 gate; milestone with caveats):** `atm_step.py` chains dynamics, radiation
hand-off (recorded SOCRATES outputs), surface/PBL/GHY, CONDSE, DISSIP and FILTER for one 30-minute step from the real step-start
state. **F1 verdict (pre-declared categories A bitwise / B <=1e-12 / C <=1e-6 / D worse, not loosened):** with the Intel
libimf and the *recorded* land patch the gate is MET with named exception columns (dec01 step 0 strictly A/B; nov26 W2GCM
4 columns at 1.7e-12; jan01 one column <=2.5e-12; one flipped CONDSE column at dec01 33555 gives PRECSS 1.7e-4). With the
*ported* GHY it is NOT MET on nov26 step 0 (errors confined to the known runoff-threshold cell (62,34) and neighbours: T 6.2e-5 K,
U 4.0e-4 m/s) and PARTLY MET on dec01/jan01 (all fields <=1e-6 of scale, worst 1.85e-7); I reproduced the dec01 verdict. Without
libimf it is not met (CONDSE threshold flips in 3-4.6% of columns; free-running 6 steps diverge chaotically, T up to 0.07-0.16 K).
Radiation (SOCRATES) and Ent vegetation exports are recorded inputs in every case. Findings: DRYCNV is not a separate stage in
this build; the TMOM/QMOM first-layer update (SURFACE.f:1068-1089) needed a port; RADIA zeroes CLDSS/CLDMC on radiation steps.
Timing with the per-column CONDSE: ~6 s per step plus 116-138 s CONDSE (the batched CONDSE of D130-D132 is not yet wired in).
119 tests (`ATM_STEP_SLOW=1`).

**D135 (land model fix):** the F1 diagnosis found a real porting error, not a threshold flip: `land_chain.run_ghy` fed the JAX
GHY `htprs=0` instead of the conditioned precipitation (pr>=0, 0<=prs<=pr, htprs=htpr/pr*prs) of ghy_ref; fixed, with a regression
test. With the fix the F1 verdict with the *ported* GHY is **PARTLY MET on all three dates** (nov26 was NOT MET; worst field Q 9.96e-7
of scale). Still open: ~97 of 753 nov26 land cells have runoff terms (aruns etc., up to 3.7e-4 of scale) that differ from the real
record in both ghy_ref and ghy_jax; undiagnosed.

**D133-D134 (fast chained step, first multi-step run):** `atm_step_fast.py` is the chained step with the batched CONDSE; it is
bitwise identical to `atm_step.py` (0 differing cells, step 0, 3 dates, libm and libimf) at ~9-15 s per step instead of 120-196 s.
Six-step run from our own end states (recorded land/radiation inputs): step 0 at rounding level (dynamics bitwise, CONDSE <=2e-12);
at step 1 CONDSE threshold flips create 157-209 columns with 1-2% errors, which then spread (597 to all 3312 columns by the
end of step 1) and grow slowly (nov26 T 1.2e-4 -> 3.6e-4, Q 1.4e-2 -> 6.4e-2, U 3e-4 -> 5.6e-3 of scale over steps 1-5); beyond step 1
a single real trajectory cannot be matched bitwise, so acceptance must be statistical (F2). Not instrumented: which CONDSE branch
flips; libm nov26 step 5 shows a worst error of 1.3 on a small-scale field (not analysed). Land was recorded, not ported, in these runs.

**D136 (second land-model error found and fixed):** the remaining runoff discrepancy (~97 of 753 nov26 cells) was the vegetated-tile
irrigation term GHY.f applies (giss_LSM/GHY.f:2230-2234; IRRIGATION_ON is defined), dropped by both ports on the false premise that
irrigation is always 0 (704 of 3012 cell-substeps in nov26/dec01 carry it). Fixed in ghy_jax, land_chain and ghy_ref; every output of
all 753 cells on 3 dates now matches the real record to 2e-13 of field scale in the JAX port (aruns was off by up to 2.1e-3).
The F1 verdict with the ported GHY has not been re-run with this second fix. ghy_ref keeps an undiagnosed residual on multi-substep cells.

**D137-D138 (ocean ODIFF and OPFIL2 coefficients ported):** `ocean_odiff.py` ports ODIFF: stage replay on the firing step of each
window gives UO/VO at <=2.3e-16 relative and VONP bitwise; the full ocean step on the ODIFF steps with no recorded ODIFF has UO/VO
exit errors <=1.9e-9, the level of the other steps; a free-running 12-step chain stays <=3.6e-9 (one nov26 non-ODIFF step reaches
1.9e-8). `ocean_opfil2_coeffs.py` computes the OPFIL2 coefficient file: integer tables and assigned SMOOTH entries bitwise, 665 of
52,861 REDUCO values differ by <=1.1e-16. The seventh step of each window is validated only through its exit snapshot; assumes
AKHFAC=1 (rundeck override not searched). Still recorded in the ocean chain: AG2OG/IG2OG fluxes and the init_STRAITS start state.

**D139-D141 (dynamics step, first JAX stages):** `dyn_step_jax.py` runs ADVECV, PGF (column recursion as lax.scan, traced FFT72),
isotropuv, SDRAG, the filter chain, calc_kea_3d and COMPUTE_WSAVE in jitted JAX; every converted stage and the 18-step chained
step are bitwise identical to the numpy-pow chain (bitwise-with-libimf is not claimed for the JAX path). Needs
`--xla_cpu_max_isa=AVX` (set by `dyn_jax_env.py`) because XLA:CPU fuses a*b+c into FMAs (PGF DUT off by 2.5e-12), and constants
passed as traced arguments; other XLA versions or a GPU may differ at 1e-16 to 1e-13. Whole step only ~1.3x faster on CPU
(2.6-2.8 s -> 2.06-2.08 s) because AADVT, QDYNAM, AFLUX/ADVECM/MAtoP (two thirds of the time) remain numpy; compile ~21 s per process.
No GPU run was possible.

**D142-D144 (dynamics JAX, remaining big stages):** `dyn_jax_aflux.py`, `dyn_jax_aadvt.py`, `dyn_jax_qdynam.py` and `dyn_step_jax2.py`:
AFLUX/ADVECM/MAtoP, AADVT (incl. 21 x4 stress calls, nstep up to 4, per-row masking) and QDYNAM (incl. 18 x8 stress calls, ncyc 4-6) run
in JAX; every stage and the 18-step chained end state are bitwise identical to the numpy-pow chain (0 unequal elements; distance from
the real dumps equals the numpy-pow chain, worst 3.8e-13 in u). Whole step 2.70-2.79 s numpy -> 1.01 s JAX on CPU (~2.7x), compile
40 s per process. Still numpy: the QDYNAM extra-column branch (reached only by 8 of the 18 stress calls), AADVT qlimit=True limiter
(dead), QDYNAM diagnostics, trop, MAtoPMB, efix/pgrad/glue. Untested (never reached): ncycxy>1 and the error-flag paths. Needs
--xla_cpu_max_isa=AVX; CPU only.

**F1 gate re-run with both land fixes (D136b, 2026-10-06, libimf mode, step 0):** with the *ported* GHY nov26 is now **MET** (worst field W2GCM
5.1e-12 of scale) and dec01/jan01 are **PARTLY MET** (worst field EGCM 9.9e-9 / 7.6e-9 of scale, previously 1.85e-7 / 2.1e-7); with the
recorded land patch all three dates are MET. Steps 1-5, libm mode and isolated stages were not re-run.

**D145-D147 (clouds in JAX):** `clouds_lscond_jax.py`, `clouds_mstcnv_jax.py`, `clouds_condse_jax.py` (+ `clouds_jax_env.py`): LSCOND, MSTCNV (host-side
compaction of convecting columns into fixed buckets, jitted masked loops) and a JAX CONDSE reproduce the numpy batch bit for bit on
the real steps run (0 of 3168 columns differ, 3 dates, steps 0-1; libm and xla modes for LSCOND). **Not faster than numpy on this CPU**
(CONDSE 5.6-6.0 s vs 4.8-5.6 s) and MSTCNV compile costs ~2.5 min per fresh process (`CLOUDS_JAX_CACHE` enables the persistent cache).
Finding: XLA's algebraic simplifier rewrites x/35.0 to x*(1/35.0), a/(b/c) to a*c/b etc. and changed 43 of 44 fields;
`--xla_disable_hlo_passes=algsimp` fixes it (probably the real cause of the closure-constant problem seen in D139). XLA:CPU exp/pow equal
glibm on this host (0 differences in 2e6 / 2e5 arguments); a GPU will probably differ by ulps. Still numpy: the QUS vertical advection
in MSTCNV subsidence (host callback), the pole columns, set-up/bookkeeping, snow ageing, recalc_agrid_uv. No libimf mode, no GPU.

**Radiation and F2/F3 plan (D148, scoping, `fullfidelity/scoping/RADIATION_AND_F2_PLAN.md`):** with GISS_RAD_OFF defined SOCRATES is the only radiation
scheme. Recommendation: recorded radiation for a ~1-day open-loop run; for anything longer a 'radiation server' (a scratch build of the real
ModelE objects running RADIA with SOCRATES untouched, exchanging state packets with Python). Hours (judgment): one-day open-loop 8-14 (central 10);
free-running day with black-box radiation 22-40 (30); one-month F3 comparison 90-150 (110, incl. ~45 for closing the surface/ocean/ice/Ent loop).
The surface, ocean, ice, lake and vegetation (Ent, recorded only) state is still replayed in every 'free-running' variant. Compute: ~11 min per model
day per member on this CPU (the real model ~1.2-1.5 h per month).

**D149-D151 (one model day, open loop; milestone):** a real 54-step model day from the nov26 restart (itime 33312, day boundary at 33360 inside the
window; dumps in `ff_data/nov26_day/`, 37 GB, 0.30 GB/step for what the chain reads, ~14.4 GB per 48-step day; every existing hook honours FFD_NSTEP) and
`atm_day_open_loop.py`, which drives the fast chain for all 54 steps feeding each step's own end state forward (radiation replayed as recorded, frozen over
each 5-step cycle with COSZ1 every step; land/ocean/ice/Ent as recorded; DAILY_ATMDYN ported; daily_ch4ox water and SNOAGE ageing recorded). 14.2 s/step
(numpy dynamics, libimf pow) or 11.1 s/step (JAX dynamics), ~14 / ~12 min per 48 steps, plus 67-100 s first-step compile. Divergence from the real run (RMS):
T 7e-14 K at step 0, 2e-4 at step 1 (first CONDSE flips), 2e-2 at step 23, 6e-2 at step 47; U 1.6e-1 m/s and Q 7.3e-5 at step 47; T drift -1e-4 to -3e-4 K.
**Noise floor (real runs perturbed by one ulp, 5 members): our whole-column RMS is within the real model's own chaos level for T, U and V at every step
and at or just above its top edge (at most 1.1x, never beyond 2x) for Q, P, QCL and QCI; zonal-mean and global-mean-difference metrics are worse (up to ~2.5x the top
member; QCL at layer 5 up to 5.3x at two steps where that layer's floors are tiny).** Our two runs (numpy- and JAX-dynamics) diverge from each other like real
members do, so the port behaves as one more member of the real model's chaotic ensemble. Limits, stated plainly: open loop (the recorded radiation nudges ours
toward the real trajectory; valid for a day or two only; no radiative feedback validated), only 5 real members, one start state, one day, steps 0-2
uninformative (floor spans 12 orders of magnitude), recorded land (ghy mode, libm mode, other dates, a second day and ocean state for the day not
exercised), and 'within the floor' is necessary evidence, not sufficient: a systematic error below ~1e-2 K per day would not show. The JAX-dynamics path is
not bitwise with the numpy path under libimf (8e-5 K at step 0), an equally valid rounding-level member. Table: `Reports/one_day_open_loop_overlay.md` (the plot is
generated by `atm_day_report.py`; PNG files are git-ignored in this repo). 11 tests (`tests/test_atm_day.py`).

**D152-D154 (radiation server, milestone for a free-running model):** a scratch build of the real ModelE objects (SOCRATES untouched, called as
part of the unmodified RADIA) that, at CALL RADIA, overwrites 52 input records from a packet file, runs RADIA and writes an output packet
(`instrumentation/ATM_DRV_radsrv.f.patch`, unit 1440; `radiation_server.py`; `scoping/RADIATION_SERVER_PLAN.md` holds the table of RADIA's inputs
with source lines). **Oracle, nov26 steps 0 and 5: the server's SRHR(0:40), TRHR(0:40), COSZ1, T and Q are bitwise equal to the recorded `ffa_step_*_r`
dumps, and RQT, KLIQ, SNOAGE, CLDSS, CLDMC, FSF, TRSURF, ALB, FSRDIR, SRVISSURF, FSRDIF, DIRVIS, DIRNIR, DIFNIR, SRDN and CFRAC are bitwise equal to
the live model outputs.** Cross-checks: CLDSS/CLDMC/SNOAGE equal the next step's `ffc_cse_in`; FSF(1)*COSZ1 equals SRHEAT in all 5418 ocean-tile
records. The TOA fluxes (taken from an AIJ delta, rounded in the reference) differ by up to 4.6e-11 relative, consistent with rounding but not proved.
Audit: of 52 packet fields, 45 move the outputs when perturbed by 1e-6; the other 7 are explained (TAUSS/TAUMC only feed the cloud mask, MA and TSAVG are
not read by RADIA in this build, FLAKE/DLAKE move outputs when changed macroscopically, SNOWI only matters for KSIALB==1, not checked). Not auditable via
the packet: PVT (Ent), BCdalbsn, RSDIST, gas/ozone/volcanic tables, random seeds (restart/time state). Cost: 27 s (step 0) to 42 s (step 5) per call,
~16 s of it model start-up (RADIA itself ~11 s); file exchange, no persistent server yet. Only nov26 was run (not dec01/jan01). 3 tests.

**D155-D157 (one model day, free-running radiation; milestone):** `atm_day_free_rad.py` replaces the recorded radiation of the D150 day by calls to the
radiation server (real RADIA, SOCRATES unmodified) with the packet assembled from OUR chained state on every radiation step (11 calls, steps 0,5,...,50);
the surface stays replayed from the real run (a free-running ATMOSPHERE over a replayed surface), nov26 only. Plumbing check: at step 0 with the real
state the assembled packet equals the live one in 52 of 52 fields and the server returns the recorded radiation bitwise. Whole 54-step day: 2250 s
(1056 s chain + 1194 s in the server calls, 31-190 s each: each call re-runs the real model from the restart up to its radiation step, cost grows with the
step; a persistent server would remove that). **Versus the real run, whole-column RMS against the real model's own chaos level (5 members): T, U, V within at all 54
steps (largest ratio 0.96); Q within at 49 steps and near at 5 (1.13); P within at 52; QCL near at 11 steps (largest 1.44) and QCI near at 10 (largest 1.95);
nothing beyond 2x; condensate excursions are dominated by single cells.** Versus the open loop (D150): the free-radiation drift is larger late in the day (T +9%,
U +15%, V +19% at the last step) and the open loop's persistent negative global-mean dT is gone; both runs behave as chaotic realizations. Radiative flux
differences free-minus-real: global means at most 0.75 W/m2 (net SW at the surface), 0.18 (downward LW at the surface), 0.70 (TOA net), signs alternating;
pointwise RMS grows with the state divergence (TOA net 3.3 to 20.6 W/m2). No flux noise floor exists (the real members record no radiation). Validates the atmosphere
with real radiation acting on our own T, Q and clouds for one day from one start state; does NOT validate radiation feedback onto the surface (replayed), multi-day
behaviour, other dates, or flux-level chaos. Findings: radiation_server._prepare ends the model window at 03:00 on nov26 (calls after step 33317 returned nothing;
fixed in the new file's run_radiation_day, merging into radiation_server.py is advisable); the AIJ column numbers were identified by global means, not an index table.
6 tests (`tests/test_atm_day_free_rad.py`; the day itself is not run in the test).

**D158 (stiff land cells, cause found, not a physics bug):** the ffg dump keeps the Ent exports and dts for at most 11 GHY sub-iterations, so for the 15 cells with ffnit >= 12
in the day-long dumps (1 per file in 14 files) both ports advanced the cell over less than the 900 s step. With the missing iterations reconstructed
(`ghy_advnc_test_nit.py`, `ghy_ref_nit.py`; the 12th iteration's dts is exactly 900 - sum of the recorded ones) all 15 cells match the real record (ashg to 5.4e-7 relative;
iterations 12+ reuse iteration 11's Ent exports, leaving alhg/aevap off by up to 3.8e-6), and all 677 cells with ffnit 8-11 still match. A second bug surfaced once the
loop is replayed: `ghy_ref.evap_limits` keeps epb/epv local where GHY.f:905,907 keeps them in module variables that gdtm reads (harmless for ffnit <= 11 where the recorded dts
are imposed). Not yet applied to the existing modules (the diffs are in the ledger): build_batch, run_cell and every ghy_jax.advnc caller (needs max_substeps); the four
nov26_day files stay marked xfail in `tests/test_ghy_jax.py`. A free-running land (no record) would need gdtm in JAX. 25 tests (`test_ghy_stiff_nit.py`, `test_ghy_jax_stiff_nit.py`).

**Test isolation (2026-10-06):** the 2026-10-06 regression showed 18 failures in `test_dyn_jax*.py` and `test_clouds_jax.py`: their bitwise checks need XLA flags set before jax is
imported and cannot work in a full run where jax is already imported; they now sit behind `tests/conftest.py` and run each in its own process via `fullfidelity/run_all_tests.sh`
(`RUN_XLA_FLAG_TESTS=1`). **Run `./run_all_tests.sh` for the full regression.**

## Standing conditions

- **SOCRATES is off-limits** (see the top-of-file note) — always.
- **Full faithful port, not a simplified ocean** — always; do not substitute a reduced/mixed-layer
  ocean without surfacing it as a decision first, the same way this scope itself was surfaced.
- **Never pause for confirmation mid-port.** The user corrected this explicitly once
  ("why did you stop and wait for me? we lost three hours") — work continues through debugging,
  deltas, and documentation without stopping to ask.
- **Timer-table trip counts are not proof of execution** — GISS ModelE checkpoints its cumulative
  CPU-timer table into the restart file, so a stale trip count can carry forward unchanged across
  reruns even when a routine never actually ran. Verify with real dump files, not the timer table.
- **`./P2SAoM40` (what actually executes) is a separate file from `P2SAoM40.bin`** (the build
  output), not a symlink — copy a fresh build to both names after every rebuild.
- **Reset BOTH `fort.1.nc` and `fort.2.nc` to the same pristine itime before every rerun** in a
  reused scratch directory — the restart reader picks whichever has the later itime, and GISS
  ModelE double-buffers checkpoint writes across both. Permanent pristine copies for all 3 test
  dates: `ff_data/_pristine_restarts/`.
- **`cp` is aliased to `cp -i`** in this environment — force-copies must use `\cp` or `rm -f` first,
  or a background task will hang forever on an interactive prompt no one can answer.

## Document map

**Start with `Project_Summary_and_Conclusions.md`**: the one-page version — goal, method,
headline results, what's open, where the evidence is.

**The full-fidelity port itself (branch `full-fidelity-port`, started 2026-09-24; Stage 1/Stage 2
work started 2026-09-28):**
- `../FULL_FIDELITY_PLAN.md`: the scope, delta-by-delta, with every subroutine's line count, what's
  real per-step physics vs. diagnostics vs. one-time init, and what's dead code (D36's correction).
- `../FULL_FIDELITY_DELTAS.md`: the full ledger, one entry per delta (D1 onward), each with the
  exact instrumentation, validation numbers, and any bugs found and fixed.
- `../fullfidelity/PHASE0_LOG.md`: the narrative working log — the debugging stories behind the
  ledger entries (e.g. the `P2SAoM40`/`P2SAoM40.bin` binary-naming bug, D29's three real numerics
  bugs, D36's dead-code discovery, the restart double-buffering trap and its permanent fix).
- `../fullfidelity/instrumentation/build_and_run.md`: the actual build/run/dump recipe, with every
  operational lesson (binary naming, restart double-buffering, `cp -i`, patch-application order)
  recorded as a standing warning, not just a one-time fix.

**Added 2026-10-05/06 (atmosphere, chains, radiation):**
- `GOAL.md` (this directory): the plain-language goal, status and realistic-path summary for management (project owner's file; the "intended science use" line is blank on purpose).
- `Reports/one_day_open_loop_overlay.md`: per-step table of our one-day drift against the real model's own noise floor (D150-D151).
- `../fullfidelity/scoping/`: `ATM_CLOUDS_SCOPE.md`, `ATM_DYNAMICS_SCOPE.md`, `ATM_DYNAMICS_CHAIN_PLAN.md`, `CLOUDS_CONDSE_PLAN.md`, `OCEAN_CHAIN_PLAN.md`,
  `ATM_STEP_PLAN.md` (the chained atmosphere step and its F1-gate protocol), `RADIATION_AND_F2_PLAN.md` (radiation options, chaos-aware F2/F3 protocol, hours) and
  `RADIATION_SERVER_PLAN.md` (the 52-field RADIA packet table and its oracle). Each states its sources and what was not read.
- `../fullfidelity/run_all_tests.sh`: the full regression (main suite in one process, the three XLA-flag-sensitive JAX test files each in a fresh process).

**Earlier work (branch history before the Stage 1/Stage 2 full-fidelity push):**
- `../STATUS.md`: **partially superseded.** Accurate for "Track A" (the earlier, deliberately-
  reduced representative driver — GHY/ATURB/SEAICE/LAKES/PBL/SURFACE chained and JAX-vectorized,
  with measured CPU/GPU speedups) and for background facts about `P2SAoM40` itself (a ROCKE-3D 2.0
  rundeck: Sun-like star, SOCRATES radiation, dynamic ocean, 40-layer atmosphere at 72x46
  horizontal resolution; Zenodo-archived reference config). **Its "Recommendation" section
  (2026-09-23) explicitly decided against a full-fidelity DYNSI/ocean port — that decision was
  later reversed** by explicit, twice-confirmed user direction; the full-fidelity work above is
  the result. Do not cite that section as current.
- `../README_GPU.md`, `../RESTART_SESSION.md`: Track A setup notes, still accurate for that track.
- `../status_slides/`: an HQ-audience slide deck snapshot (Track A only; predates the full-fidelity
  push). Live/editable version linked from `status_slides/README.md`.

## Open items (historical list from 2026-10-04; most are closed, the current list is "Next steps" in the Handoff above)

1. **`ODHORZ`'s JAX vectorization** is deliberately deferred as its own follow-up (D42's
   plain-Python port is validated; batching this large multi-physics routine deserves dedicated
   care, the GHY-lesson discipline). **`OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2`'s JAX vectorization**
   is deferred the same way (D45's plain-Python port is validated; the dynamic segment structure
   and pole-masking subtlety make it a poor first batching candidate).
2. **`GLMELT`** (glacial meltwater, `OCNDYN.f`, 72 lines) is scoped but deferred: it fires only
   once per calendar day (`daily_OCEAN`'s `end_of_day` gate), and the current 6-step/3-hour test
   windows are not confirmed to cross a day boundary. Needs either a day-boundary-crossing test
   window or a decision to skip it.
3. **`OPFIL2`** (the polar Fourier filter itself) is still unported and still needs an external
   `AVR` reduction-matrix binary file plus an FFT (`OFFT`/`OFFTI`) whenever something that does
   call it is tackled — resolved that neither `OFLUXV` (D43) nor `OADVT2`/its family (D45)
   actually depend on it, so nothing currently blocks on this; it's simply not yet needed.
4. **`OCNDYN2.f`'s entire real per-step dynamical core is now ported** (D45 closed the last
   piece, the `OADVT2` tracer-advection family) — nothing remains open in this file beyond
   `OPFIL2` itself (item 3) and JAX vectorization (item 1).
5. **Mesoscale mixing (`OCNMESO_DRV.f`+`OCNGM.f`) is CLOSED (D46-D51)** —
   `ocnstate_derived`/`densgrad`/`get_1d_mesodiff` (D47), `ISOSLOPE4` (D49), `GMKDIF`'s remaining
   coefficients (D50), and `GMFEXP`+`computeFluxes`+`wrapAdjustFluxes`+`addFluxes` (D51) are all
   **done**, bitwise-exact, every field, every date. `OCNTDMIX.f` (2,030 lines) confirmed
   **entirely dead** for this build (`use_tdmix=0`, D46); `GET_PSI_DIAG` confirmed purely
   diagnostic (D48); `QCROSS` confirmed always false, eliminating roughly half of `GMKDIF`'s and
   `GMFEXP`'s cross-term logic as dead code (D48). `OCNQUS.f` (1,846 lines)
   and `OCNGISS_TURB.f`/`OCNGISS_SM.f` (1,252 lines) confirmed entirely dead — do not port.
   JAX vectorization for this whole family deliberately deferred, same discipline as `ODHORZ`/
   `OADVT2` (item 1).
5a. **Ocean core remaining (refreshed 2026-10-04):** the batched JAX OCONV HBL loop (in progress,
   see Current state); the straits code is ACTIVE for this build (`NMST=12`), so `OSTRAITS.f` +
   `OSTRAITS_COM.f` (~1,200 lines) and `STCONV` (378 lines) must be ported. Estimate 25-40 hours.
   The earlier "OCONV driver ~1,500 lines" item is now split into the pieces D59-D64 cover.
5b. **`OCNKPP.f` (KPP vertical mixing) — scoped in full (D52, corrected D53); `KPPMIX` (D54, JAX D63),
   `OVDIFF`/`OVDIFFS` (D55-D56, JAX D62/D64), `REDUCE_FIG` (D57), `KVINIT` (D58), setup (D59, JAX D64) all done.** Four subroutines confirmed entirely dead
   (`get_kvtdiss`/`get_gradients0`/`wscale`/`swfrac`) plus `ddmix` (compile-time `LDD=.false.`)
   and `bldepth` (D53: dead, `KPPMIX` inlines its own boundary-layer-depth logic instead).
   Remaining in this file: `OCONV`'s own ~1,526-line per-column driver (including the
   fixed-point HBL iteration D54 deliberately didn't port), `KVINIT`(46), `OVDIFF`(60)/
   `OVDIFFS`(47), `REDUCE_FIG`(14) — roughly 1,700 lines — plus `STCONV`(378, still blocked on
   unscoped `OSTRAITS.f`+`OSTRAITS_COM.f`, the largest still-entirely-unread item in Stage 2).
6. **`IRRIG_LK`** (Stage 1, external prescribed-irrigation dataset dependency) deferred since D28,
   not yet needed by anything downstream.
7. **The JAX/batched `jax.lax.while_loop` version of `DYNSI`'s own outer `KKI` loop** (`VPICEDYN`)
   is still plain-Python only — the per-cell Newton/ADI solve inside it is batched and validated,
   but the outer iteration-count loop itself has not been converted.
8. **No whole-model chained Track B step exists yet** for the full-fidelity branch (Track A has
   one, `run_steps_device`, but it uses the reduced/approximated physics this branch exists to
   replace) — every piece validated so far is a per-routine, per-call-record match against real
   Fortran, not yet wired into a running chained step.

## Suggested wording for a new session

"Read `Reports/README_START_HERE.md` (the Handoff section first) and `Reports/GOAL.md`. Check the regression log named in 'In flight and uncommitted', push if it is clean, then
continue with 'Next steps': dump-hook-and-validate methodology, verify every agent's result yourself before committing, report differences honestly (no silent tolerance loosening),
SOCRATES stays untouched (radiation only through the radiation server or as recorded input), keep the docs current."
