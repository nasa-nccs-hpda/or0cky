# Release: ROCKE-3D-Claude-Native-Radiation-20261012

Owner: G. Tamkin (port-driver session). Written 2026-10-12. Branch `full-fidelity-port`; tag `ROCKE-3D-Claude-Native-Radiation-20261012`; stable branch `release/native-radiation-20261012`. Every number below is copied from the file named; nothing here was re-measured for this note.

## What this release is
The hybrid JAX port of ROCKE-3D (dynamics, clouds/CONDSE, surface, land/GHY/Ent, ocean, ice, lake on one V100) with two radiation options: the unmodified Fortran RADIA helper (`--rad server`) or the SOCRATES JAX port (`--rad native`, `fullfidelity/socrates_spike/`, owned by the SOCRATES session). All behaviour flags are default off.

## Benchmarked state (what the numbers apply to)
Job 37941716, node gpu020 (exclusive V100, Xeon Gold 6248), nov26 restart itime 33312, 108 steps, commit `212de04` (this tag adds only a diagnostic-script edit and the merge of `main`'s status-deck files; no model code). Source: `Reports/BENCHMARK_V100_PROTOCOL.md` section 8; run files `gpu_runs/bench_native_jax/` and `gpu_runs/bench_native_37941716.out` (not in git, large).
| | Fortran, 1 core (3 runs) | JAX native, V100 (1 run) |
|---|---|---|
| 108 steps wall | 254.4-257.5 s | 1,442 s |
| cold start | none | 1,208 s |
| steady per step | 2.28 s (main loop) | 2.21 s |
Steady state is at parity (0.97), not faster; one JAX run, so within node noise. Repeated on a second exclusive node (gpu022, job 37941737, section 8.1): Fortran 2.24 s, JAX 2.13 s per step, ratio 0.95, cold start 1,191 s, same fidelity band (day 1 QCI 2.37 at step 3, day 2 worst 1.10). Extrapolated amortization is in section 8.

## Fidelity
`d207_score.py` against five real one-ulp Fortran members: day 1 worst ratio at steps >= 3 QCI 2.37 at step 3 (unrelaxed 2x criterion NOT MET); day 2 never beyond 2x (worst QCL 1.22). Relaxed criterion (3x at steps >= 3, ACCEPTANCE s10) met. Never compare trajectories (noise floor; `Reports/STANDALONE_INPUT_AUDIT.md` section 6).

## Boundary and known limits
* Still reads the five step-0 dump files from `ff_data/nov26_day2` (about 278 MB); no per-step record read (five-day 240-step run finished, finite).
* nov26 window only. The SOCRATES day products are not the limit (`daily_products.DailyProducts` builds them for any calendar day from the rundeck input files, bitwise vs Fortran on 48 dates, report 14.4; source: SOCRATES session message 2026-10-12); the limit is that no other date has been run through the model (restart, step-0 dumps and reference members exist for nov26 only). On-demand callbacks (`day_source=`, `zen_source=`) cost 6-9 s per radiation step on the V100 and are CPU-only; `--rad native` uses the precomputed day products / zenith table.
* Irrigation uses year 1850 and no fossil groundwater; GHY iterations beyond 11 not read under `ROCKE_TPL_OWN`.
* Flags that make the run standalone (`ROCKE_TPL_OWN`, `ROCKE_OWN_GHY`, ... see the command in `Reports/SOCRATES_TRIAL_RUN_RECIPE.md` section 2) are off by default.
* `ROCKE_STAGE_ARGS=1` is not bitwise equal to the flag-off result (rounding-level, protocol section 7).
* Regression: see the result line at the end of this file; `test_libimf_fused` is order-dependent (passes alone), polluter not found.
* JAX held steps were 2.14 s here against 1.7-1.8 s on earlier nodes; cause not isolated.

## Not in this release
SOCRATES commits `eeef43c` (38-date whole-step check, report 14.3) and `bdeb975` (volcanic reader, day products and zenith on demand, report 14.4) on `origin/socrates-spike`; they postdate the benchmarked code. To be merged after this release.

## How to reproduce
`gpu/benchmark_native_v100.sh` (Fortran x3 then JAX native, one exclusive node): `V100_CPUS=8 V100_EXTRA=--exclusive gpu/v100_run.sh bench_native 100 'bash ../gpu/benchmark_native_v100.sh'` from `rocke3d_jax/`. Cold compile is about 18-20 min.

## Regression result
Full regression (`fullfidelity/run_all_tests.sh`, XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover", graphcast-env python) on merge commit `7b99198` (identical model code to the tag; the tag commit adds only the benchmark section 8.1, this note and the benchmark script), log `gpu_runs/release_regression_7b99198.log`, 2026-10-12, 1 h 48 min:
* main suite: 3329 passed, 16 skipped, 1 failed: `tests/test_libimf_fused.py::test_dq_dev_and_conv_micro_dev_bitwise_vs_cmj_libm_and_mutation` (the known order-dependent test; passes alone; polluter not found).
* separate-process files: test_dyn_jax, test_dyn_jax2, test_clouds_jax and the two isolated files (test_daily_lake_hook, test_d187_imf_p1): all passed (6, 8, 8, 2 passed lines in the log).
