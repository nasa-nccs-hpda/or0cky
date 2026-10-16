# QA handoff: review package, GPU baseline and the early-step cloud differences (2026-10-16)

Owner: G. Tamkin. Written by a Claude Code session (rocke3d-jax-c8) for a QA developer taking over. Project-local. Status: DRAFT; none of the interpretation below has been reviewed by the owner, the in-house climate scientist or the external reviewer. Code under test: `full-fidelity-port` at `061486d` unless stated. Review by: before the review package is sent again or any fidelity statement is made from it.

## 1. Terms used in this note

* **Owner**: G. Tamkin. Decides criteria and thresholds (`Reports/ACCEPTANCE_CRITERIA.md`).
* **SME**: the external reviewer who received the review package (not in group `ilab`). The owner's manager asked them for a quick look at whether the results "make sense"; the package inputs are those of the ROCKE-3D 2.0 paper.
* **Climateer**: the in-house climate scientist who commented on fidelity versus accuracy. A different person from the SME.
* **Fidelity** here = JAX distance from the Fortran reference compared with the spread of five one-ulp Fortran members (ACCEPTANCE s2-4, s9-10). **Accuracy / precision** = the one-step A/B/C/D categories (bitwise, <= 1e-12, <= 1e-6, worse). The A/B/C/D counts are not a fidelity test.

## 2. State at handoff

* Git: `full-fidelity-port` was level with `origin/full-fidelity-port` at `40a87da` before this note and `fullfidelity/qa_20261016/` were added (committed locally; see section 8 for whether they were pushed). Working tree clean before them.
* Review package (outside git, data directory `/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/`): `rocke3d_review_package_20261016/` (549 files in `MANIFEST.sha256`), `rocke3d_review_package_20261016.tar` (5.37 GB) and `.tar.sha256`. The 2026-10-14 tar was deleted (it had been overwritten with amended files).
* **The tar is stale relative to the directory.** The directory README has a note (added 2026-10-16, after the tar was built) that `jax_run/d193_run.json` fields `radiation` and `radiation_sentence` are fixed driver text ("replayed") and do not describe the run. The tar also contains nine `__pycache__` entries (not in the manifest) that were since removed from the directory. Rebuild with `fullfidelity/qa_20261016/rebuild_review_package_tar.sh` (about 8 minutes; verifies the manifest first; swaps the tar in only after it lists cleanly).
* **The copy on Discover** that the owner placed for the SME is the earlier tar (sha256 `fd546a6e1411f88e7af52b09e523e6d7fe94738c3c7369f0d57549133f6e2d6a`). It was NOT verified from the work node (`/discover` is not reachable from forest204 or gpulogin1). The owner has the directory; it is kept in the session memory, not in this repository.
* The reply drafted for the SME (plain text) was written to the session scratchpad only; the owner sends it. It tells the SME to start with `plots/`, states radiation is the SOCRATES JAX port, states the run is one model day and not a climate simulation, and that cloud ice at one early step is slightly outside the strict threshold.

## 3. What was established (with the evidence)

1. **V100 baseline and repeatability.** `gpu/v100_step6_graphcast.sbatch` (graphcast-env, V100, partition `compute`, feature `v100`) runs the 6-step coupled nov26 case. Two independent runs at `061486d` on different nodes (jobs 37942390, 37942591) are bitwise identical at steps 0 and 5; steady mean about 2.6 s/step (the Oct 10 run, job 37941449, was 9.7 s/step on older code). Against the CPU libm references the GPU end states are category D for most fields at step 5 (device exp/pow differ by one ulp in 6-12% of values; chaos amplifies). Not bitwise with CPU is expected. Source: `scoring` outputs in the session scratchpad (`cmp*`, `c_*`), not kept.
2. **Review package JAX run** (job 37942781, native SOCRATES radiation, flags as in the package README): strict day rule NOT MET (QCI 2.37 at step 3), relaxed rule met (3x from step 3). Same figures in `RELEASE_NATIVE_RADIATION_20261012.md`.
3. **Quick-look plots** (`plots/`, script `fullfidelity/review_package_plots.py`): fields look the same by eye; per-step rms difference sits at or just above the one-ulp member band for T, U, V, Q and above it early for QCI.
4. **Spread versus RMSE** (`qa_20261016/spread_rmse.py`). The Fortran leave-one-out RMSE/spread ratio is an algebraic identity (sqrt(1.2) = 1.095 for six runs) and carries no information; do not cite it as a check. The informative result is the JAX distance from the Fortran six-run mean divided by the six-run standard-deviation spread (about 1.08 would be member-like): steps >= 3 medians T 1.21, U 1.15, V 1.15, Q 1.12, QCL 1.03, QCI 1.42; above 2 only at steps 3-6 (QCL up to 2.98, QCI up to 5.23) and U at step 4 (2.07). These ratios use the ensemble mean and a standard-deviation spread, so they are NOT comparable with the project's ratio (largest member distance from the control, about 1.4-1.7 times larger).
5. **Localization (steps 0-10)** (`qa_20261016/localize_qc.py`). The QCL/QCI error is concentrated: the 20 worst cells carry 98-100% of the squared error at steps 0-5 (73-96% at step 10). Layers (from the surface, no pressure mapping) 15-18 for QCL and 16-20 for QCI; QCI at step 3 is 98% layer 19. Zero/non-zero cloud flips: JAX is within the member range up to step 5, above it at steps 6-10 (QCL 172 versus 55-145 at step 6; QCI 24 versus 3-12).
6. **Column follow-up** (`qa_20261016/follow_columns.py`). Columns A (lon -157.5, lat -10) and B (-162.5, -14), indices (i=4,j=20) and (3,19), are far outside the Fortran members (hundreds to tens of thousands of times the member range) from steps 0-1; columns (-162.5,-6) and (137.5,34) are member-like. At step 3 in column A the QCL loss in layer 18 (-5.9e-4) and the QCI gain in layer 19 (+6.2e-4) are nearly equal and opposite.
7. **Radiation inputs are not the cause for column A.** From the 2026-10-11 server-radiation run (`gpu_runs/twoday_server`, older code, NOT the package run) the JAX radiation input packet in column A at the first radiation step matches the recorded Fortran packet to about 1e-13 (outputs SRHR/TRHR to about 1e-8 relative). Column B differs slightly (TAUSS 7.5e-4, T 3e-3). The package native run recorded no packets, so this was not checked on it.
8. **CONDSE cloud scheme, same entry state** (`clouds_condse_compare.py`, NumPy port, nov26 steps 0-3, run in `fullfidelity/`): without libimf column A is "flipped" (a field beyond 1e-12 of its scale against the real Fortran) at steps 0 and 1 and fine at 2 and 3; with libimf (`--imf`) it is fine at all four steps. Columns B, C and D are not flipped in either mode. Flipped columns overall without libimf: 151, 149, 109, 117 of 3,170 (about 3-5%, matching the project's documented threshold-flip rate); with libimf: 6, 0, 17, 0. The package JAX run made zero libimf calls (`d193_run.json` `libimf_ops`).

**Working hypothesis (not shown):** the early QCL/QCI excess in column A comes from exp/pow rounding differences (XLA GPU device math instead of Intel libimf) tipping cloud thresholds. Column B's step-0 difference is not explained by it.

## 4. Statements corrected during the session (do not reuse the earlier wording)

* "QCI returns inside the member band" was too strong: its median for steps >= 20 is still 1.43 against about 1.08 expected.
* The spread-versus-RMSE check on the Fortran members was proposed as informative; it is an identity (item 4 above).
* The lead "radiation inputs from JAX's own state cause the step-3 excess" is not supported for column A (item 7).
* The package README title said 2026-10-14; it now says 2026-10-16 revision.

## 5. Open items

1. **libimf run on the GPU (not built).** Plan: a NEW wrapper beside `gpu_day_run.py` (leave existing files alone) that does not import `gpu_step_run` (it forces libm: `LI.set_mode('libm')`, `FX.set_mode('libm')`, `W.install` replaced, `make_ctx imf=False`) and instead uses the libimf wiring (`d187_imf_p1.install`, `libimf_ops` / `libimf_fused` mode `libimf`, context `imf=True`). Needs: provenance `require_xla_flags=False` as in `gpu_day_run.py`, no JAX compile cache, at least 2 CPU cores (host callbacks), Intel runtime reachable from the compute node (`d193_run.json` header shows it was available on `gpu011`), `GPU_DAY_DRIVER=d212_day` with the package flags (README section 2). `gpu_day_run.py` says libimf is not available on GPU nodes; the package header shows it is, so test with a short run first. Expected cost: slower than the 23 min libm day; GPU library behavior with host callbacks is untested. Success test: does QCI at step 3 drop below 2x, and are columns A and B closer to the members.
2. Rebuild the tar, re-verify, and tell the owner that the copy on Discover is stale (or send only the changed files: `README.md`, `MANIFEST.sha256`).
3. Verify the Discover copy against the `.sha256` (owner, from Discover) and that the SME can read it.
4. Owner review of the amended `ACCEPTANCE_CRITERIA.md`: section 1 item 4 (both radiation options, own required sentence), lines 36 and 44 (both options named; the "every fifth step" cadence for `--rad native` was not confirmed) and section 5 (radiation may be described as ported to JAX for `--rad native`). The wording was written by the agent; the owner has not signed it off. The older lines in the file that say "SOCRATES is not ported" outside these places were not searched for.
5. The status of the separate fidelity-report agent whose output the owner pasted is unknown to this session. Whether an explicit physics-coverage check exists in the repo was not searched for.
6. The SME has not replied to the package yet.

## 6. How to re-run what is described here

**Start with `fullfidelity/qa_20261016/ENVIRONMENT.md`**: it says which python to use (the site anaconda base python, no personal conda environment, tested from an empty environment), how to point the scripts at the package (`REVIEW_PKG`), and what the V100 jobs need (`gpu/environment-graphcast-env.yml`, rebuild untested). The commands below assume that setup.

* Package scoring: `cd <package>/scoring_tool && python d207_score.py --ours ../jax_run/ours_d193 --ff ../fortran_reference --daydir nov26_day2 --nsteps 54 --md out.md` (README section 5).
* Plots: `REVIEW_PKG=<package dir> python fullfidelity/review_package_plots.py <outdir>` (about 1.5 minutes; numpy and matplotlib, graphcast-env python).
* Items 4-6 of section 3: `fullfidelity/qa_20261016/{spread_rmse,localize_qc,follow_columns}.py` (hard-coded package path; `spread_rmse.py` and `localize_qc.py` take an output directory; graphcast-env python; 1-3 minutes each).
* Item 8: `cd fullfidelity && OMP_NUM_THREADS=1 python clouds_condse_compare.py [--imf] --dates nov26 --steps 0,1,2,3 --json out.json` (about 7 minutes per mode, one core).
* GPU recipe: `ssh gpulogin1 "cd <project dir> && sbatch gpu/v100_step6_graphcast.sbatch"`; outputs in `gpu_runs/v100_step6_graphcast_<jobid>/` (git-ignored).

## 7. Known weak points in this note

* Items 1 and 3-6 of section 3 come from session-scratchpad runs whose outputs are not in the repository; the scripts are, and reproduce them.
* Only four columns were followed, chosen because they were worst; item 7 used an older run; item 8 used the NumPy port with the CPU math library, not the GPU code.
* Layer numbers are counted from the surface and not mapped to pressure; longitude and latitude assume the 72 x 46 grid at 5 x 4 degrees and were not read from the files; field units are not labelled.

## 8. Handoff bookkeeping

* Added by this handoff: `Reports/QA_HANDOFF_20261016.md` (this file), `fullfidelity/qa_20261016/` (three analysis scripts, `rebuild_review_package_tar.sh`). Commit state: see `git log -1 -- Reports/QA_HANDOFF_20261016.md`; if `git status -sb` shows ahead of origin, the owner has not yet approved the push.
* No SLURM jobs or background processes were running at handoff. Scratchpad outputs (comparison tables, `loc_out`, `spread_rmse_out`, `cse/*.json`) are session-local and may be deleted.
