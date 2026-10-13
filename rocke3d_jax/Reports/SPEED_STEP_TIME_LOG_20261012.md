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

## 7. Feasibility spike: host-compiled MSTCNV (2026-10-12; CPU only, nothing wired in)
Measured (`fullfidelity/spike_mstcnv_percol.py`, captured nov26 step-1 inputs, 32 of 3,168 columns at stride 100, 24 of them convective, interpreted per-column scalar port `clouds_mstcnv_ff.mstcnv_column`, one core, dev host):
* per column: median 19.8 ms, mean 33.8 ms, max 146.8 ms; extrapolated to 3,168 columns at the mean: 107 s per step (interpreted). That is the work a compiled port would have to shrink by about 150x to reach 0.7 s on one core, or about 40x on four cores, to match the present device time; the bar needs it near 0.1-0.2 s.
Estimate (NOT measured; labelled arithmetic): compiled scalar code of this kind typically runs 100-1000x faster than interpreted Python; at 100x the one-core time is about 1 s, at 300x about 0.35 s; with 4-8 cores and about 45 MB of column state each way over PCIe (about 10-20 ms) the stage could land at 0.1-0.4 s. Fidelity: the compiled code would be the validated per-column port (libm mode), the same reference the device path is scored against.
Blockers found:
* numba is NOT usable in `graphcast-env`: the user-site numba (`~/.local`) refuses NumPy 2.2.4 ("Numba needs NumPy 1.26 or less"). Fixing means a separate environment (not done, not authorised here).
* The port is 900 live lines in vectorised-helper style (`np.asarray`, `np.broadcast_arrays` even for scalars; dict inputs; `locals()` snapshots for checkpoints; `adv1d` QUS from `dyn_adv1d_ff`, `mass_flux`, conv microphysics from helper modules). A compiled version is a rewrite of the port and its helpers (C via gcc, which exists on the node, or Cython/numba in a new env), plus the per-column tests the NumPy port already has (`clouds_mstcnv_compare.py`, `clouds_mstcnv_batch_compare.py` give the reference). No effort estimate exists in the repo; my guess is days, not hours.
* Integration: a host callback inside the jitted step fires only after the device program finishes (D214 finding, `GPU_V100_VS_A100.md`), so the compiled MSTCNV would have to run between jit programs (as the pole columns do), splitting `condse` into set-up jit -> host MSTCNV -> post jit.

## 8. Host-compiled MSTCNV (numba, worker process) -- written 2026-10-13 by the speed session
Owner direction 2026-10-13: bar now 0.6 s per held step, "try everything including architecture changes". Hybrid boundary unchanged (five step-0 dump files, nov26 only, irrigation year 1850). One exclusive V100 node per job.

**Port.** `fullfidelity/clouds_mstcnv_nb.py`: numba nopython port of `clouds_mstcnv_ff.mstcnv_column` (same statement order, scalar helpers, explicit loops for the XYM/ZM moment lists, one-column adv1d/limitq). `speed_mstcnv_nb_compare.py` against the NumPy port on all 3,168 columns of the captured nov26 step-1 inputs (`mstcnv_inputs_nov26_step1.pkl`, DEV-Kernel python, numba 0.60): **57 of 57 output fields bitwise equal in every column**; 31.8 ms per column (NumPy port) vs 0.203 ms per column (numba, one thread, first call excluded). With 8 threads the batch is about 0.04 s.

**Integration (flag `ROCKE_MSTCNV_HOST=1`, default off).** graphcast-env (python 3.10, NumPy 2.2.4) cannot import numba (0.58.1 refuses NumPy > 1.26; no network from the dev host to install another), so `mstcnv_host.py` starts a persistent worker under a numba-capable interpreter (`ROCKE_MSTCNV_PY`, default `/home/gtamkin/.conda/envs/DEV-Kernel/bin/python`, numba 0.60 + NumPy 2.0.2). Inputs/outputs go through one memory-mapped file in /dev/shm; control is stdin/stdout. `CondseDevice.mstcnv` returns the same dict as the device `_post` (checked field by field by `speed_mstcnv_host_test.py`: no mismatch on sampled columns; `speed_mstcnv_col_test.py`: the single-column drop-in used by the pole columns, no mismatch on 33 columns). `jax_coupled.py` skips the instance-level device-loop override when the flag is set. Environment dependency: the worker interpreter path (not in the repo); numba cache files are written beside the source (`__pycache__`).

**Results on the V100** (frozen flag set, `--rad native`, 8 CPUs allocated, same stage tables as sections 1 and 4):
| | baseline (37941820) | WIN+U2+PBL-carry jit (37941824) | + host MSTCNV (37941843, untimed) |
|---|---|---|---|
| held step median | 1.71 s | 1.449 s | **0.740 s** |
| steps 2-107 total (106 steps) | 185.1 s | 156.6 s | **81.8 s** (0.771 s mean per step, radiation steps included) |
| step 0 (cold) | 1,090 s | 1,495 s | 982 s (no device MSTCNV compile) |
| day-1 worst ratio (steps >= 3) | 2.37 QCI step 3 | 2.37 QCI step 3 | **2.46 QCI step 3** |
| day-2 worst ratio | 1.20 | 1.19 | 1.105 (QCL step 97) |
Timed run 37941842 (24 steps, `--timed 1`, held steps): condse_mstcnv 0.924 -> **0.085 s**; wall median 0.729 s. Stage table of that run (median s): dyn 0.180, surface_tiles_land 0.133, condse_mstcnv 0.085, condse_post 0.074, condse_setup 0.065, post_a 0.061, ocean_b 0.044, melt_si_dev 0.017, ocean_a 0.012, rest < 0.02.

**Band.** Same failing step and field as the baseline (QCI at step 3), ratio 2.46 against 2.37; day-1 "beyond" counts U 1, P 2 (baseline 1), QCL 2, QCI 1; "within" counts per field are higher than the baseline's (e.g. T 43 vs 11 of 54). The host path is bitwise equal to the NumPy port, which is not bitwise equal to the GPU device port (XLA fusion/FMA), so this is a rounding-level change of the same class as the existing noise floor; it is NOT bitwise equal to the flag-off run and the 2x criterion remains NOT MET exactly where it was. The owner decides whether this counts as "band unchanged". Files: `gpu_runs/host_full/d207.md`, `gpu_runs/full_win_u2_pc/d207.md`.

## 9. Level-batched QUS moisture advection, pole fast path (2026-10-13)
Hybrid boundary unchanged. Per-unit timers (flags `ROCKE_DYN_TIMERS`, `ROCKE_SURF_TIMERS`, `ROCKE_MSTCNV_HOST_TIMING`, `ROCKE_POLE_PROFILE`, `ROCKE_STAGE_TRACE`, `ROCKE_SURF_TRACE`; diagnostics only, default off) located the remaining held-step time (job 37941845, host MSTCNV on, `--timed 1`, median steady steps):
* `dyn` 0.18 s, of which `qdynam` 0.105 s (the QUS moisture advection: per cycle a 41-iteration level loop with the y/x sweeps nested inside), aflux 0.021, aadvt 0.013, trop 0.009.
* `surface_tiles_land` 0.133 s = `core_ns1` 0.06 + `core_ns2` 0.06 (the same compiled program twice) + 0.01 glue.
* pole columns: each call 0.062 s = slice + device_get 0.021 + `host.run` 0.033 + upload 0.008; cProfile of the column itself: 0.004 s. The 33 ms is `np.zeros` + `_fresh_outputs` copies of about 100 full-size arrays (job 37941846).
* `post_a` 0.061 s: trace (job 37941850) 30,659 GPU kernels, 23,000 of them `loop_add_fusion*` = the left-to-right `seqsum` scans of `jax_dynsi` (sea-ice dynamics iteration, 3,404-element sums, up to 20 iterations); device busy 0.041 s inside a 0.131 s span.
* `ocean_b` 0.044 s: trace (job 37941851) 12,247 GPU kernels (scatter fusions, memcpy), 4 host callbacks.

Changes (all default off, each checked on CPU):
| flag | change | check |
|---|---|---|
| `ROCKE_QDYN_VMAP=1` | `dyn_jax_qdynam.make_cycle_vmap`: y/x sweeps and z checkflux of all 40 levels as one vmap over the level axis (a level's own sweeps and checkflux precede every interface sweep that touches it), only the interface sweep keeps its sequential carry | `speed_qdyn_vmap_test.py`: q and qmom bitwise equal to the level loop and to the Fortran `fin` record on 2 recorded calls (132,480 of 132,480 q values exact) |
| `ROCKE_POLE_FAST=1` | pole columns: entry/output arrays built as zeros plus the pole slice (no copies), the slicing of the ~100 entry arrays as one jit, `device_put` of the results | synthetic equality test of the built X dict for both poles |
| `ROCKE_SEQSUM_BLK=N` | `seqsum_fast.py`: the scan `c + x` over blocks of N written out in order (fused into one kernel), tail added in order | `speed_seqsum_test.py`: bitwise equal for 1-D/2-D, several lengths and axes |
| `ROCKE_GLUE_JIT=1` | `_tpl_own_state` as one jit | not yet measured |
Job 37941847 (`ROCKE_QDYN_VMAP=1`, `--timed 1`): `qdynam` 0.105 -> **0.0085 s**; held wall 0.7 s timed.

**Full 108 steps (job 37941849: host MSTCNV + QDYN_VMAP + POLE_FAST, untimed; NOTE the node was shared with three other jobs of this session from about step 25 on, so the log walls after step 24 are inflated by I/O and CPU contention; the step_walls.json times below measure `Coupled.step` only)**:
| | baseline (37941820) | host MSTCNV (37941843) | + QDYN_VMAP + POLE_FAST (37941849) |
|---|---|---|---|
| held step median (n=85) | 1.71 s | 0.735 s | **0.568 s** (min 0.522, max 0.701) |
| radiation step median (n=21) | 1.88 s | 0.898 s | 0.731 s |
| steps 2-107 total | 185.1 s | 81.8 s | **64.0 s** (0.604 s mean per step, radiation steps included) |
| step 0 (cold) | 1,090 s | 982 s | 930 s |
| day-1 worst ratio (steps >= 3) | 2.37 QCI step 3 | 2.46 QCI step 3 | 2.46 QCI step 3 |
| day-1 "beyond" counts (U/P/QCL/QCI) | 1/1/2/1 | 1/2/2/1 | 1/2/2/3 |
| day-2 worst ratio | 1.19 | 1.105 | 1.104 (QCI step 68) |
The failing step and field are the baseline's (QCI at step 3); QCI has two more "beyond" steps than the baseline, within-counts per field are higher than the baseline's. As in section 8 this is a rounding-level change (QDYN_VMAP changes XLA fusion, the host MSTCNV is the NumPy-port arithmetic), judged by the band score, not bitwise; the owner decides whether it counts as "band unchanged". Files: `gpu_runs/pf_full/d207.md`.

Caution for anyone repeating timings: run one job per node (`V100_EXTRA='--exclusive'`); four of this session's jobs shared gpu012 and the later steps of 37941849 show it.

## 10. Everything-on run on an exclusive node (job 37941856, 2026-10-13)
Flags: `ROCKE_MSTCNV_HOST=1 ROCKE_QDYN_VMAP=1 ROCKE_POLE_FAST=2 ROCKE_SEQSUM_BLK=32 ROCKE_GLUE_JIT=1` on top of the frozen set; `V100_EXTRA='--exclusive'`, 8 CPUs; 108 steps, untimed. Hybrid boundary unchanged (five step-0 dump files, nov26 only, irrigation year 1850).
| | baseline (37941820) | host MSTCNV (37941843) | + VMAP/POLE_FAST (37941849, shared node) | **all five flags (37941856)** |
|---|---|---|---|---|
| held step median | 1.71 s | 0.735 s | 0.568 s | **0.449 s** (min 0.406) |
| radiation step median | 1.88 s | 0.898 s | 0.731 s | 0.612 s |
| steps 2-107 total / mean per step | 185.1 s / 1.75 s | 81.8 s / 0.77 s | 64.0 s / 0.60 s | **52.5 s / 0.495 s** |
| step 0 (cold) | 1,090 s | 982 s | 930 s | 895 s |
| day-1 worst ratio (steps >= 3) | 2.37 QCI step 3 | 2.46 QCI step 3 | 2.46 QCI step 3 | 2.46 QCI step 3 |
| day-1 "beyond" counts U/P/QCL/QCI | 1/1/2/1 | 1/2/2/1 | 1/2/2/3 | 1/2/2/1 |
| day-2 worst ratio | 1.19 | 1.105 | 1.104 | 1.143 (QCI step 75) |
Held-step max 1.805 s is the day-boundary step (daily lake / Ent update). Cycle arithmetic against the Fortran (labelled arithmetic, not a matched benchmark): 0.495 s mean per step against the Fortran's 2.24-2.28 s on its node is about 4.5x; the matched Fortran benchmark has not been rerun for this branch.
Stage table of the matching timed diagnostic (job 37941854, held steps, median s, ROCKE_NEWTON_UNROLL=4 also on, no effect): surface_tiles_land 0.138, condse_mstcnv 0.085, dyn 0.063, ocean_b 0.043, condse_post 0.027, post_a 0.019, condse_setup 0.017, ocean_a 0.013, melt_si_dev 0.008; timed wall median 0.477 s.
Effects measured one at a time (timed diagnostics, held median s): SEQSUM_BLK=32 post_a 0.061 -> 0.019, dyn 0.081 -> 0.063, melt_si_dev 0.016 -> 0.009 (job 37941853); POLE_FAST=2 condse_setup 0.040 -> 0.017, condse_post 0.051 -> 0.027 (self-check on the first three calls: results equal to the plain construction); NEWTON_UNROLL=4: no change in `core` (0.060 / 0.063).
Surface `core` (0.06 s per substep) is 0.051 s GHY (`_land`): PBL 0.007, water and land-ice tiles 0.0004 each (job 37941855). The GHY loop runs up to GDTM_WIDTH = 40 sub-iterations (the slowest land cell), each with the Ent photosynthesis `rtsafe` while loops (<= 100 iterations); the core trace shows 1,677 device->host predicate copies. `ROCKE_WHILE_UNROLL` (masked K-iteration body, `while_unroll.py`) is applied to `rtsafe`; measurement pending (job 37941858).

## 11. GHY / Ent: where the surface stage goes, and the batched Ent step (2026-10-13)
Hybrid boundary unchanged. Jobs on exclusive nodes (8 CPUs).
* `surface_tiles_land` `core` (0.060 s per substep, twice per step): pieces as separate jits (jobs 37941855, 37941860): PBL batch 0.0074 s, water and land-ice tiles 0.0004 s each, **GHY `_land` 0.051-0.054 s**. GHY substep iterations per land cell (753 cells): max 10, mean 2.28, counts 1..10 = 226/301/125/42/31/15/7/2/1/3; the loop runs 10 iterations of the whole N-lane body. A profiler trace of the GHY piece (job 37941857): 10,798 kernels, 1,673 while iterations with a host predicate copy each (about 6 kernels per iteration): the nested `rtsafe` while loops of the Ent photosynthesis (12 qsimp leaf calls per Ent step, plus `_ci_cubic`, each up to 100 iterations, about 150 sequential iterations per Ent step); `ent_step` alone costs 4.5-5.4 ms per call on the V100 (`gpu_ent_bench.py`).
* Flags tried on `ent_step` alone (job 37941859): `--xla_gpu_enable_command_buffer=...,WHILE` 4.64 -> 4.04 ms; `ROCKE_WHILE_UNROLL=4` (masked 4-iteration while body) 4.51 -> 4.06 ms; other flag sets no change. Not adopted.
* `ROCKE_ENT_SKIP=1` (skip Ent when no vegetated lane is active, skip converged qsimp refinements; equal on CPU in `ent_live_check.py`): no gain on the GPU (GHY piece 0.0534 -> 0.0575 s, job 37941861): vegetated lanes stay active in every iteration. Left in, default off, not recommended.
* **`ROCKE_ENT_BATCH=1`**: the leaf evaluations of the six qsimp refinement levels do not depend on each other (only the convergence tests do), so they are done in one batched call, sunlit and shaded leaves in one concatenated array: 12 sequential rtsafe loops become 1 (+ the `_ci_cubic` one). `ent_step` alone 5.39 -> 2.37 ms (job 37941864), `cnc` sum identical to 16 digits; on the CPU 35 of 1,146 `cnc` values (before the sun/shade merge) differ by up to 5.7e-16 relative (array-shape dependent vectorised exp), i.e. NOT bitwise equal to the level-by-level version. GHY piece 0.0534 -> **0.0296 s** (job 37941863); surface units 0.138 -> 0.096 s (job 37941866).
* `ROCKE_SCAN_UNROLL=80` (full unroll of every static scan/fori <= 80 steps): host process killed by the memory limit during compilation (job 37941862, 24 GB). Not usable at that setting; flag left in, default off.
* `ROCKE_MSTCNV_HOST_IO=1` (batched async device_get / device_put, only the 17 + scalar output fields the post stage reads): written, measurement pending (job 37941867). Worker removes its shared file on EOF.

**Full 108 steps, exclusive node, flags `ROCKE_MSTCNV_HOST=1 ROCKE_QDYN_VMAP=1 ROCKE_POLE_FAST=2 ROCKE_SEQSUM_BLK=32 ROCKE_GLUE_JIT=1 ROCKE_ENT_BATCH=1` (job 37941865, untimed)**:
| | baseline (37941820) | section 10 flags (37941856) | + ENT_BATCH (37941865) |
|---|---|---|---|
| held step median | 1.71 s | 0.449 s | **0.402 s** (min 0.381) |
| radiation step median | 1.88 s | 0.612 s | 0.568 s |
| steps 2-107 total / mean per step | 185.1 s / 1.75 s | 52.5 s / 0.495 s | **47.6 s / 0.449 s** |
| step 0 (cold) | 1,090 s | 895 s | 871 s |
| day-1 worst ratio / beyond U,P,QCL,QCI | 2.37 QCI 3 / 1,1,2,1 | 2.46 QCI 3 / 1,2,2,1 | 2.46 QCI 3 / 1,2,2,1 |
| day-2 worst ratio | 1.19 | 1.143 (QCI 75) | 1.214 (QCL 93) |
Band: the failing step and field are the baseline's (QCI at step 3, 2.46 against 2.37, first appearing with the host MSTCNV in section 8); counts of "beyond" steps per field are the baseline's except P (2 against 1). Held-step max 1.71 s is the day-boundary step. Remaining held-step stages (timed, median s, job 37941866): surface_tiles_land 0.096, condse_mstcnv 0.087, dyn 0.063-0.074, ocean_b 0.043, condse_post 0.027, post_a 0.019, condse_setup 0.017, ocean_a 0.013.
