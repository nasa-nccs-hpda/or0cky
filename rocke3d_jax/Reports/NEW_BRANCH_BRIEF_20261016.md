# Brief for the new session: a minimal orphan branch (2026-10-16)

Owner: G. Tamkin. Written by a Claude Code session (rocke3d-jax-c8) from the owner's decisions of 2026-10-16. Project-local. Status: DRAFT for the owner; nothing in this brief has been carried out. Read `Reports/QA_HANDOFF_20261016.md` and `AGENTS.md` first.

## 1. Decisions already made by the owner

* **Orphan branch in this repository.** No history or blame needed on the kept files; the archive is enough. Archive = a tag on the current head of `full-fidelity-port` (suggested `archive/full-fidelity-port-20261016`) plus the existing branches. Do not delete or rewrite them.
* **Minimum source and documents only.** Test code, scoping notes, drafts, slides, output maps and run records stay in the archive.
* **Minimum evidence (tests that support the final summary deck):** (1) the Fortran reference runs; (2) the JAX release configuration (all speed flags off); (3) the JAX fast configuration (speed flags on); (4) the scoring tool and the reference-spread definition; (5) the regression subset that exercises the kept source. Both JAX configurations are kept.
* **"Fastest overall" = the fast configuration.** Nine speed flags (`ROCKE_MSTCNV_HOST`, `ROCKE_MSTCNV_IMF`, `ROCKE_MSTCNV_HOST_IO`, `ROCKE_QDYN_VMAP`, `ROCKE_POLE_FAST=2`, `ROCKE_POLE_ASYNC`, `ROCKE_SEQSUM_BLK=32`, `ROCKE_GLUE_JIT`, `ROCKE_ENT_BATCH`) on top of the release flags, merged as `1996ef1`, tag `speed-step-time-20261013`. Numbers: matched benchmark job 37941875 (0.524 s/step against Fortran 2.285 s); untimed profile run job 37941870 (held step 0.385 s, mean 0.434 s). Recipe: `Reports/SPEED_RUN_RECIPE_20261013.md`. Release: `Reports/RELEASE_NATIVE_RADIATION_20261012.md`.
* **Replication gaps are shipped as open, not closed.** The fast result is one matched run; the fast flags ran 108 steps only; the 30-day shadow run is invalid after step 1288; the native-radiation repeat differences are not localized; the strict 2x fidelity rule is not met (QCI step 3: 2.37 release, 2.47 fast). State these in the new branch's status document.

## 2. Owner's end goal

A single script that runs all variants at once, in parallel. Design constraints found so far (the owner has been told these; confirm them in the new session):

* **Parallel across nodes, sequential within a matched timing pair.** `gpu/benchmark_native_v100.sh` runs the Fortran timing runs (one core) and then the JAX run on the SAME exclusive node, strictly sequentially, so the hardware matches. Timing runs must not share a node with anything else. Fidelity-only runs have no such limit and can run concurrently.
* **So the script should submit SLURM jobs** (one per variant, on `compute` with `--constraint=v100`, `--exclusive` for timing) and a dependent job that scores and writes one summary table; not one process running everything. Needs at least three exclusive nodes at once (release pair, fast pair, fidelity-only extras) plus a CPU job for the regression subset.
* **Variants:** Fortran reference and timing (3 runs, one core); JAX release; JAX fast; scoring against the five one-ulp members; the regression subset (flags off, and with the nine flags set).
* **The script cannot promise bitwise repeats.** Native-radiation same-code repeats are not bitwise (`Reports/REPEATABILITY_AND_FLAGS_REGRESSION_20261014.md`); report ratios against the member spread, not equality. Timing varies by node (2.13 to 2.21 s release on two nodes).
* **Cluster facts:** forest204 has no SLURM client (use `ssh gpulogin1 sbatch`); partition `compute`, feature `v100`; `/discover` is not reachable from there.

## 3. Tasks, in order

1. Tag the archive. Create the orphan branch from a clean checkout. Do not push until the owner approves.
2. **Manifest first, pause for review.** An allowlist: path and a one-line reason per entry. Start from the entry points `gpu_day_run.py`, `d212_day.py`, `gpu_step_run.py`, `d193_day.py`, `cpu_day_run_imf.py`, the scoring tools (`d207_score.py`, `multiday_score.py`, `atm_day_report.py`) and `clouds_condse_compare.py`. A static import closure (this session's estimate) gives about 222 modules and 63,600 lines; it excludes `socrates_spike/` (156 files, needed for `--rad native`), the dynamic imports it cannot see, and tests. Trace `socrates_spike/` by file. Find the tests with coverage against the kept modules (138 test files under `fullfidelity/tests/`, 169 in the repository).
3. **Housekeeping (owner asked for it):** remove reliance on the author's environment and paths in kept files.
   * `fullfidelity/mstcnv_host.py:21`: `ROCKE_MSTCNV_PY` defaults to `/home/gtamkin/.conda/envs/DEV-Kernel/bin/python` (numba 0.60, NumPy 2.0.2). The fast configuration depends on this personal environment; document an environment spec and make the variable required or default to a neutral interpreter.
   * `fullfidelity/run_all_tests.sh:6`: `PY` defaults to the personal `graphcast-env` python.
   * `gpu/*.sh`, `gpu/*.sbatch`: `PROJ`/`ROCKE_PROJ`, `FF_DATA`, `CONDA` paths (the recipes name the author's checkout, and some still name `/explore/...`).
   * Counts in tracked scripts and sources (all tracked, including archived ones): `/home/gtamkin` 11 occurrences in 9 files; `/explore/nobackup/people/gtamkin` 16 in 12; `/panfs/ccds02/nobackup/people/gtamkin` 175 in 144 (most in code that will not be kept); `/discover/nobackup/gtamkin` 1 in 1. Fix only what the kept set contains.
   * Also see `fullfidelity/qa_20261016/ENVIRONMENT.md` and `gpu/environment-graphcast-env.yml` (rebuild untested).
4. **Gates, before the branch counts:** every kept file is byte-identical to its archive copy except files changed on purpose (list them); every kept module imports from an empty environment; the package scoring reproduces the table in the review-package README section 3; the 6-step V100 run reproduces bitwise (`gpu/v100_step6_graphcast.sbatch`); the kept test subset passes (flags off, then the nine flags set).
5. Single status document (dated) with the open replication gaps of section 1, and a short README. No scratch files, logs or analysis scripts in the repository; put them outside it.
6. Then the parallel runner (section 2), written and checked against the gates before any large submission.

## 4. Things to keep out of the new branch

`fullfidelity/scoping/` (276 files), `outputs/` (45), `status_slides/` (28), the draft reports in `Reports/`, `gpu_runs/` records, old drivers not reached from the entry points, `__pycache__`, logs. Keep the owner-reviewed acceptance criteria; the 2026-10-16 amended wording in `ACCEPTANCE_CRITERIA.md` has NOT been reviewed by the owner, so mark it as such if it is carried over.

## 5. Open items from the QA handoff that this brief does not close

Rebuilding the review-package tar; the CPU libimf day (`fullfidelity/cpu_day_run_imf.py`, dry-run only); owner sign-off of the criteria wording. One caution for that run: the fast configuration already sets `ROCKE_MSTCNV_IMF=1` (libimf for the cloud-base stage) and its day-1 QCI step-3 ratio is 2.47, not lower than the 2.37 baseline, so a libimf effect on the cloud-base stage alone does not remove the excess.
