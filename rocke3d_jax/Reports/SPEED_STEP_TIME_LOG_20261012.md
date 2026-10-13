# Speed track log: held-step time on the V100 (branch `speed/step-time`, off tag ROCKE-3D-Claude-Native-Radiation-20261012)

Owner: G. Tamkin. Written 2026-10-12 by the speed session. Project-local. Every number is copied from the job output named; nothing here was estimated unless labelled. Target (owner decision 2026-10-12): the SOCRATES memo bar, non-radiation work about 0.75 s per held step.
Hybrid boundary unchanged: five step-0 dump files, nov26 only, irrigation year 1850. Flags of the frozen benchmark set (`gpu/benchmark_native_v100.sh`), `--rad native`, one exclusive V100 node per job.

## 1. Profile of the baseline (jobs 37941763/64 timed, 37941765 untimed, 17 steps; trace jobs 37941761/62)
| stage (held step, median s, two nodes agree) | s | share |
|---|---|---|
| condse_mstcnv | 0.92 | 55% |
| dyn | 0.20 | 12% |
| surface_tiles_land | 0.13 | 8% |
| template_apply | 0.09 | 6% |
| condse_post + condse_setup | 0.15 | 9% |
| post_a + ocean_a + ocean_b | 0.13 | 8% |
| everything else | about 0.06 | 4% |
Held step wall 1.62 (untimed) to 1.71 (timed 1) s; radiation step 1.79-1.87 s (`radia` 0.005 s -> 0.17 s). Files: `gpu_runs/prof_stage_{u,t1,t2}/step_walls.json`.

## 2. Why MSTCNV costs 0.92 s (job 37941794 trace of one event block, job 37941797 HLO dump)
One event block (LMIN 3, 128 columns): 53 ms wall, 21,741 GPU kernels, median kernel 1.6 us, summed device time 47 ms. About 2,500 static `fori` iterations plus about 600 plume-ascent iterations per block, each a handful of tiny kernels (the per-iteration counter add alone is a kernel). The device is saturated with launches, not compute. Already ruled out by the project: round-based loop (1.78 s), full unrolling (compile > 26 min), XLA command-buffer flags incl. WHILE (`gpu/mstcnv_flag_sweep.sh`).

## 3. Changes tried (all default off)
| change | stage bench `events` steady (job) | bitwise vs original (digest) | compile of that stage |
|---|---|---|---|
| baseline | 0.894 s (37941795) | - | 200 s |
| `ROCKE_MSTCNV_WIN=1` ascent step on per-level windows | 0.833 s (37941796) | equal | 198 s |
| WIN=1 + `ROCKE_MSTCNV_UNROLL=2` | 0.776 s (37941798) | equal | 634 s |
| WIN=1 + UNROLL=4 | 0.767 s (37941799) | equal | 915 s |
| UNROLL=8 | cancelled after 67 min of compile (37941800) | - | > 4,000 s |
| PBL-carry `loadbl` as one jit + index arrays cached (no flag; arithmetic unchanged, select/where only) | `template_apply` stage 0.09 -> 0.004 s, eager ops 399 -> 3 per held step (job 37941824) | n/a (full run scored, section 4) | none |
| `ROCKE_FUSED_DYN=1` (existing fused-dynamics mode) | not usable: step 0 2,234 s, step 1 (a new DIAGA phase) 887 s; 27 distinct phases -> about 6.5 h cold start (job 37941822, cancelled) | - | - |

## 4. Matched 108-step runs (jobs 37941820 baseline, 37941821 WIN=1 UNROLL=2; profile wrapper, untimed, exclusive nodes gpu006 / gpu016)
| | baseline (37941820) | WIN=1 UNROLL=2 (37941821) | + PBL-carry jit (37941824) |
|---|---|---|---|
| held step median (n=85; min-max) | 1.71 s (1.63-1.94) | 1.51 s (1.45-1.67) | 1.44 s (1.38-1.56) |
| radiation step median (n=22) | 1.88 s | 1.67 s | 1.61 s |
| sum of steps 2-107 | 185.1 s | 163.6 s | 156.6 s |
| step 0 (cold compile) | 1,090 s | 1,512 s | 1,495 s |
| score day 1 worst ratio (steps >= 3) | 2.37 QCI step 3 | 2.37 QCI step 3 | 2.37 QCI step 3 (unchanged; unrelaxed 2x NOT MET as before) |
| score day 2 worst ratio | 1.20 QCL step 84 | 1.08 QCI step 107 | 1.19 QCL step 86 |
Files: `gpu_runs/full_base/`, `gpu_runs/full_win_u2/`, `gpu_runs/full_win_u2_pc/` (`step_walls.json`, `score.md`, `score.json`). Three different nodes (gpu006, gpu016, one more): node noise is 0.1-0.2 s, so the 0.07 s step between the last two columns is within it on its own; the template_apply stage change (0.09 -> 0.004 s, eager ops counted) is the evidence for that one.

## 5. Position against the bar
Held step 1.44 s against the 0.75 s bar: 0.69 s still to remove, from 1.71 s (-0.27 s, -16%). MSTCNV is now about 0.7 s of it (stage bench 0.77 s; not re-profiled per stage in the full step); the rest of the step is about 0.7 s. Incremental restructuring of the launch-bound loops gave -0.20 s per held step and +7 min of cold start; further unrolling is flat (0.776 -> 0.767 s on the stage bench) at +15 min.
Node note: the baseline held step is 1.62-1.71 s on these nodes against 2.06-2.14 s on gpu020/gpu022 in `BENCHMARK_V100_PROTOCOL.md` section 8; cause not isolated (node noise 0.1-0.2 s stated there is smaller than the difference). Compare only within one job set.

## 6. Not done / open
* A host-compiled MSTCNV (numba is present in `graphcast-env`; `clouds_mstcnv_ff.py` is the per-column scalar port, 1,214 lines) would remove the launch-bound structure; not started, not sized.
* Remaining eager dispatches per held step (job 37941824): `melt_si_dev` 57, `template_build` 30, `surface_tiles_land` 26; phase 1 (job 37941821): `condse_setup` 140 and `condse_post` 140 (the two pole columns run as host NumPy steps, 34 calls / 1.24 s over 17 steps), `dyn` 66 eager + 73 jit, `condse_mstcnv` 80. The pole columns are the next-largest identifiable host cost (about 0.07 s per step); `dyn` is 0.18-0.20 s.
* Per-stage times in the untimed profile runs are dispatch times, not device times; use the `--timed 1` jobs (section 1) for stage shares.
* The benchmark of section 8 (108 steps with Fortran x3 beside it) was not rerun for the changed flags; the numbers above are JAX-only on different nodes.
