# Speed review handoff brief (for a new session)

Owner: G. Tamkin. Written 2026-10-12 by the port-driver session. Project-local. Every number below is copied from the file named; items marked NOT KNOWN were not measured.

## 1. Task
A whole-code-base look at runtime speed of the ported model, starting from the release tag `ROCKE-3D-Claude-Native-Radiation-20261012` (commit `a13087e`, branch `release/native-radiation-20261012`). Work on a new branch off the tag (suggested `speed/<topic>`); do not push to `full-fidelity-port` (the port-driver session owns pushes there and the post-release merge of `socrates-spike`).

**Target (owner decision, 2026-10-12): a steady step time below the Fortran's about 2.24-2.28 s per step** (not multi-day throughput; the cold start is out of scope unless it blocks measurement).

Note for the session: the baseline is already nominally below that line (JAX 2.13-2.21 s against Fortran 2.24-2.28 s, ratio 0.95-0.97, one JAX run per node, node noise 0.1-0.2 s). So "below the Fortran" is met only within noise. A first job for the session is to ask the owner for a margin that makes the claim meaningful; candidates from the record: (i) a gap larger than the 0.2 s node noise, e.g. at most 2.0 s per step; (ii) the SOCRATES decision-memo bar, non-radiation plus hand-off at about 0.75 s per step (`Reports/DECISION_MEMO_2026-10-11.md`, memo `c88b5b9`). The owner has not chosen between these.

## 2. The frozen baseline you measure against
`gpu/benchmark_native_v100.sh` (one exclusive V100 node, Fortran x3 on one core then JAX `--rad native`, nov26 restart, 108 steps = two model days). Results: `Reports/BENCHMARK_V100_PROTOCOL.md` sections 8 and 8.1.
| | Fortran, 1 core | JAX native, V100 |
|---|---|---|
| gpu020 (job 37941716), per step steady | 2.28 s (main loop 244.6-247.0 s / 108) | 2.21 s (steps 2-107, 234.4 s) |
| gpu022 (job 37941737), per step steady | 2.24 s (241.3-243.1 s / 108) | 2.13 s (226.2 s) |
| held / radiation / day-boundary step | not timed per step | 2.14 / 2.49 / 2.3 s (gpu020); 2.06 / 2.43 / 2.0 s (gpu022) |
| cold start (steps 0 and 1) | none | 1,208 s; 1,191 s |
Rule (owner, 2026-10-11): one frozen benchmark; rerun the same job after each change and keep the old baseline for comparison. One JAX run per node so far; node noise about 0.1-0.2 s per held step. Any claim under about 0.2 s per step needs repeats.

## 3. What is known about where time goes
* Radiation with the Fortran helper: radiation step 10.0 s; with SOCRATES native: 1.9 s (earlier jobs allfix2, native108); on the matched nodes a radiation step is only about 0.4 s above a held step. Radiation is no longer the main cost.
* A held step (no radiation) is 2.06-2.14 s on the matched nodes and 1.7-1.8 s in earlier jobs on other nodes (cause not isolated). SOCRATES session decision memo `c88b5b9` (branch `socrates-spike`, `Reports/DECISION_MEMO_2026-10-11.md`): the bar for beating the Fortran is non-radiation plus hand-off falling from about 2.1 s to 0.75 s per step.
* Cold start is about 20 minutes (XLA compile, no persistent cache by design, rule D175); native radiation added about 310 s to it.
* Day-boundary stall (step 48: 135 s) was removed by `ROCKE_STAGE_ARGS=1` (1.7-2.3 s now); that flag is off by default and not bitwise equal to flag-off on the GPU (protocol section 7).
* Per-program profile of a held step on the V100: NOT KNOWN in this brief. First task: profile one held step and one radiation step (device time against host and launch time), then decide.
* Earlier component timings on the dev host (D215) and the hybrid host scaffolding (cloud-base, pole columns, template build) are in `Reports/STATUS_2026-10-11.md` and `Reports/DECISION_MEMO_2026-10-11.md`.

## 4. Constraints
* Fidelity band must not move: `fullfidelity/d207_score.py --ours DIR/ours_d193 --nsteps 108`, five real one-ulp members; baseline day 1 worst ratio QCI 2.37 at step 3 (unrelaxed 2x NOT MET), day 2 worst 1.22 (A) / 1.10 (B). Never compare trajectories (noise floor: identical configurations diverge from step 2 at about 1e-15). Any change to compiled programs must be re-scored; rounding-level changes are amplified like any one-ulp perturbation.
* Flags default off; keep defaults unchanged unless the owner decides.
* Hybrid boundary: the run still reads five step-0 dump files, nov26 only, irrigation year 1850 (no fossil groundwater). State it with every speed result.
* GPU access: `gpu/v100_run.sh NAME MINUTES 'CMD'` from `rocke3d_jax/` (ssh gpulogin1, partition compute, constraint v100, graphcast-env jax 0.5.3). XLA behaviour differs by jax version.
* Regression: `fullfidelity/run_all_tests.sh` with `XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover"`; last full run 3329 passed, 1 failed (`test_libimf_fused`, order-dependent, passes alone), 16 skipped, 1 h 48 min. Run it in large batches, not per change.
* AGENTS.md rules: project-local docs, no invented facts, record owners and sources.

## 5. Files to read first
`Reports/BENCHMARK_V100_PROTOCOL.md` (sections 6-8.1), `Reports/RELEASE_NATIVE_RADIATION_20261012.md`, `Reports/DECISION_MEMO_2026-10-11.md`, `Reports/SOCRATES_TRIAL_RUN_RECIPE.md` (flags and command), `Reports/STANDALONE_INPUT_AUDIT.md` (what is recorded vs computed), `fullfidelity/d212_day.py`, `jax_coupled.py`, `jax_atm_phase1.py`, `jax_surface.py`, `jax_tpl_state.py`.

## 6. Suggested order (a recommendation, not a decision)
1. Get the owner's target (a or b).
2. Profile one held and one radiation step on the V100; produce a table of device time, host time and launch count.
3. Rank by size of the gap to 0.75 s (target b) or by cold-start share (target a), then propose changes each as a labelled benchmark version beside the baseline.
