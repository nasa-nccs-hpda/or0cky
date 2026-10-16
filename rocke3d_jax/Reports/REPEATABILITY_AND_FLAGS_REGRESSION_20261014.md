# Repeatability, flags-on regression and review package (2026-10-14)

Owner: G. Tamkin. Written by a Claude Code session working for the owner. Project-local. Status: MEASURED results of 2026-10-14; the cause of the
native-radiation difference is NOT localized. Code under test: `full-fidelity-port` at `061486d` (clean tree) unless stated. Review by: before any
decision to turn the speed flags on by default.

## 1. Flags-on regression (CPU, forest204)

* Command: `fullfidelity/run_all_tests_flags_on.sh` (wraps `run_all_tests.sh`, same `XLA_FLAGS`), speed flags exported: `ROCKE_MSTCNV_HOST`,
  `ROCKE_MSTCNV_IMF`, `ROCKE_MSTCNV_HOST_IO`, `ROCKE_QDYN_VMAP`, `ROCKE_POLE_FAST=2`, `ROCKE_POLE_ASYNC`, `ROCKE_SEQSUM_BLK=32`, `ROCKE_GLUE_JIT`,
  `ROCKE_ENT_BATCH`; `ROCKE_MSTCNV_PY` = the DEV-Kernel numba interpreter; staged flags NOT set (`STAGED=0`).
* Result: main suite 3329 passed, 1 failed, 16 skipped in 1 h 30 min; the five isolated / XLA-flag files all passed. The one failure is the known
  order-dependent `test_libimf_fused::test_dq_dev_and_conv_micro_dev_bitwise_vs_cmj_libm_and_mutation`. Counts are the same as flags off
  (`SPEED_STEP_TIME_LOG_20261012.md` s13). Log: `fullfidelity/tests_flags_on_20261014_1132.log` (gitignored, local).
* Limit: no test under `tests/` names a speed flag. The source reads the flags from the environment (for example `ROCKE_QDYN_VMAP` at call time,
  `ROCKE_ENT_BATCH` at import), so flagged code ran only where the tests reach those stages. This shows no regression, not that every flagged path is
  covered; the flag-specific checks (`speed_*_test.py`) are outside this suite.

## 2. GPU replay step: bitwise reproducible, and what moved since 2026-10-10

* `gpu_step_run.py nov26 --steps 6 --save first-last` (LIBM, radiation replayed), jobs 37942390 (`gpu013`) and 37942591 (`gpu012`), both V100, HEAD
  `061486d`: step 0 and step 5 end states are bitwise identical (561 and 572 fields, byte-identical npz). Steady mean 2.61 s per step (against 9.71 s in job
  37941449 of 2026-10-10 at `594e6ec`).
* Against the CPU libm references (`gpu_data_stage/ref_cpu_libm_nov26_step0/5.npz`) the V100 end state is in categories A/B/C/D = 126/162/55/215
  (step 0) and 58/1/27/483 (step 5); GPU is not bitwise with the CPU, as stated in the run labels.
* Against job 37941449 (older code), step 0 agrees to 2e-16; step 5 differs in 459 fields (category D). That is a code change between `594e6ec` and
  `061486d` (the D214 device ADV1D default, OADVT2 pre-pass and pole-column changes are the candidates), NOT nondeterminism, since the same-code repeat above
  is bitwise. Not bisected.

## 3. Native radiation: same-code repeats are NOT bitwise

* Command (same for all): the ten staged flags of `STANDALONE_INPUT_AUDIT.md` s9 (`ROCKE_TPL_OWN` ... `ROCKE_STAGE_ARGS`), `GPU_DAYDIR=nov26_day2
  GPU_DAY_DRIVER=d212_day python -u gpu_day_run.py <out> --rad native --surf-fields computed --seed computed --cosz computed --nsteps 54 --timed 0`,
  speed flags off, V100.
* Runs (outputs `gpu_runs/fluxdiff_*`, gitignored): A1 37942781 (`gpu011`), A2 37942782 (`gpu011`), A3 37942785 (`gpu013`), B 37942783 (`gpu011`,
  `ROCKE_NATIVE_NRAD=60`), A1x 37942831 (`gpu011`, exclusive), A2x 37942832 (`gpu009`, exclusive).
* Result: none of the 15 pairs is bitwise equal in `FSF` (`native_rad_outputs.npz`); maximum differences 443 to 732 W/m2 against `FSF` up to 1300 W/m2;
  `COSZ1` is bitwise equal in every pair. Radiation step 0 (itime 33312) is equal in all runs; the first difference appears at itime 33317 or 33322.
* Not the cause: `ROCKE_NATIVE_NRAD=60` (B differs like the others), the node (same-node and cross-node pairs alike), GPU sharing (the exclusive-node
  pair differs too), the zenith.
* Earlier files `ctrl_tag54`, `ctrl_merge54b`, `merge1996ef1_54` also differ pairwise by up to 509 W/m2; they were different commits and settings, so on their
  own they did not establish nondeterminism; the runs above do.
* Fidelity scores of the six runs (`d207_score.py`, day 1, steps >= 3): worst ratio QCI 2.366 at step 3 in all six; other fields within 0.1 across runs
  (T 1.18-1.22, U 1.23-1.26, V 1.23-1.24, Q 1.14-1.19, P 1.18-1.30, QCL 1.58); beyond-2x counts identical (U 1, P 1, QCL 2, QCI 1). Verdict identical in all
  six: unrelaxed NOT MET, relaxed MET.
* Not known: which stage first differs (the window is the model steps between radiation steps 33312 and 33317/33322), whether the source is the native
  SOCRATES port, one of the staged own-input paths, or a host thread / callback. The replay step of section 2 is bitwise, so it lies in the day-run path.
  Next experiment: toggle native radiation and the staged flags one at a time on a short window, twice each (needs the driver options checked first).
* Consequence: bitwise run-to-run comparison cannot validate a code change on this path; use the statistical score. The score itself is stable across
  repeats.

### 3.1 Localization: the staged own-input flags, not native radiation (same day, later)

2 x 2 factors, 11 steps (itime 33312-33322), `--surf-fields computed --seed computed --cosz computed`, V100 exclusive nodes, HEAD `061486d`, each config
run twice (jobs 37942866-37942873, outputs `gpu_runs/loc_*`, gitignored). Per-step state files (`ours_d193/step_<it>.npz`) compared bitwise:

| config | radiation | staged flags | repeats bitwise equal (11 steps) | first differing step |
|---|---|---|---|---|
| Non | native | on (all ten) | NO (8 of 11 steps differ, max 1.44) | 33315 (U, V) |
| Noff | native | off | YES | none |
| Ron | replayed | on, `ROCKE_REC_LAZY` off | NO (5 of 11 steps differ, max 1.32) | 33318 (T, U, V, Q) |
| Roff | replayed | off | YES | none |

* Reading: the difference follows the staged own-input flags (`ROCKE_TPL_OWN`, `ROCKE_OWN_GHY`, `ROCKE_IRRIG_OWN`, `ROCKE_PBL_CARRY`, `ROCKE_REC_LAZY`,
  `ROCKE_CSE_IN_OWN`, `ROCKE_CARRY_CSE_OUT`, `ROCKE_ENT_LIVE`, `ROCKE_RADCOLS`, `ROCKE_STAGE_ARGS`) and NOT native radiation: replayed radiation with the flags on
  also differs, native radiation with the flags off does not. The native radiation `FSF` differences of section 3 are downstream of that.
* With every staged flag off the day-run path is bitwise reproducible on this V100 window (native and replay). 11 steps only; longer windows not run.
* Ron differs from Non by one flag (`ROCKE_REC_LAZY` is off in Ron because it needs a server-radiation setup the replay mode does not enforce); the
  first differing step also moves (33318 against 33315), so the flag set matters, not only the on/off split.
* Not known: which of the ten flags (or which pair) causes it, and why (device reductions, host threads, a read of an uninitialized slot). Next: halve the
  flag set (five and five), then single flags, same windows, two repeats each.

## 4. Review package (for the science expert)

`/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/rocke3d_review_package_20261016/` (and `.tar`, 5.37 GB, with `.sha256`; `MANIFEST.sha256`
inside, 549 files after the 2026-10-16 additions below). Contents: the Fortran control and five one-ulp members and real end states for itime 33312-33365, the restart, the step-0 records, the run
deck and I-file (`inputs/rundeck/`), the instrumented binary with build patches, the 54-step JAX run (A1), the scoring tool, and a README with limits.
The scorer was run from the package alone and reproduced the table above. Outside git (data directory). The owner is copying it to the expert's directory (the expert's account is not in group ilab).

Renamed 2026-10-16: the package directory and tar are now `rocke3d_review_package_20261016` (the tar and its `.sha256` are rebuilt under that name); the earlier `rocke3d_review_package_20261014.tar` and `.tar.sha256` were deleted the same day (they had been overwritten with the amended files, so they no longer matched the original 2026-10-14 delivery).
Additions of 2026-10-16 (outside git, in the package; the `.tar` and `.tar.sha256` were rebuilt and verified against `MANIFEST.sha256`):
* `plots/`: five quick-look figures of the JAX run against the Fortran reference (maps of the lowest layer and layer 13 at step 53, zonal-mean cross sections, per-step rms difference against the one-ulp member band, zonal profiles at steps 0 and 53) and `make_plots.py`; README section 8 describes them. They are a visual sanity check, not a score. Limits: no units labelled, layers not mapped to pressure, pole rows masked, grid axes assumed 5 x 4 degrees. The same script is kept in the repo as `fullfidelity/review_package_plots.py` (run with `REVIEW_PKG=<package dir>`).
* `ACCEPTANCE_CRITERIA.md` in the package and in `Reports/` (commit `d24ed88`): section 5 now allows describing radiation as ported to JAX for `--rad native` results and requires saying which radiation a result used. Lines 36 and 44 (coupled-step coverage and window) were amended the same day to name both radiation options (Fortran callback or `--rad native`). Section 1 item 4 was amended the same day: radiation is the Fortran callback or the SOCRATES JAX port, the result says which, and each option has its own required sentence.
* Not done: no fresh run of the JAX code from the package alone.

## 5. Not done

No localization of the section 3 difference; no bisect of the section 2 step-5 shift; flag-specific tests are not in the regression suite; no fresh run of
the JAX code from the package alone.
