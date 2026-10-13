# Speed track: how to run the fast configuration (branch `speed/step-time`, 2026-10-13)

Owner: G. Tamkin. Written by the speed session. Project-local. Numbers are copied from `Reports/SPEED_STEP_TIME_LOG_20261012.md` (sections 8-13); nothing here is estimated unless labelled. Hybrid boundary unchanged: the run still reads the five step-0 dump files, nov26 only, irrigation year 1850. Every flag below defaults off; the frozen benchmark is unchanged unless they are set.

## Command (V100, exclusive node, graphcast-env)
```
V100_CPUS=8 V100_EXTRA='--exclusive' ROCKE_PROJ=<worktree> gpu/v100_run.sh NAME 120 \
 'ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1 \
  GPU_DAYDIR=nov26_day2 GPU_DAY_DRIVER=d212_day \
  ROCKE_MSTCNV_HOST=1 ROCKE_MSTCNV_IMF=1 ROCKE_MSTCNV_HOST_IO=1 ROCKE_QDYN_VMAP=1 ROCKE_POLE_FAST=2 ROCKE_POLE_ASYNC=1 ROCKE_SEQSUM_BLK=32 ROCKE_GLUE_JIT=1 ROCKE_ENT_BATCH=1 \
  python -u gpu_day_run.py OUTDIR --rad native --surf-fields computed --seed computed --cosz computed --nsteps 108 --timed 0'
```
or `SPEED_FLAGS='...' gpu/benchmark_native_v100.sh` for the matched Fortran-then-JAX benchmark.

## Flags (all default off), what each does, how it was checked
| flag | effect | check |
|---|---|---|
| `ROCKE_MSTCNV_HOST=1` | cloud-base stage on the host through the numba port of `clouds_mstcnv_ff` (`clouds_mstcnv_nb.py`, `mstcnv_host.py`); a persistent worker process under `ROCKE_MSTCNV_PY` (default `/home/gtamkin/.conda/envs/DEV-Kernel/bin/python`: numba 0.60, NumPy 2.0.2) because graphcast-env cannot import numba; data through one file in /dev/shm | bitwise equal to the NumPy port on all 3,168 columns of the nov26 step-1 capture (57 of 57 fields); `speed_mstcnv_host_test.py`, `speed_mstcnv_col_test.py` |
| `ROCKE_MSTCNV_IMF=1` | exp/pow through Intel libimf as `set_backend('imf')` of the NumPy port (the Fortran-faithful arithmetic); no numba cache for this variant | bitwise equal to the NumPy imf backend on 159 columns |
| `ROCKE_MSTCNV_HOST_IO=1` | one batched `device_get` / `device_put`, only the output fields the post stage reads | timed: condse_mstcnv 0.087 -> 0.071 s |
| `ROCKE_QDYN_VMAP=1` | QUS moisture advection: y/x sweeps and z checkflux of the 40 levels batched with vmap | q, qmom bitwise equal to the level loop and to the Fortran record on 2 calls (CPU) |
| `ROCKE_POLE_FAST=2` | pole columns: persistent buffers restored after each call, jitted pole slicing, `device_put` of results | self-check against the plain construction on the first 3 calls (prints `pole_fast self-check ...`) |
| `ROCKE_POLE_ASYNC=1` | south pole column in a worker thread overlapping the set-up jit and MSTCNV | held 0.402 -> 0.385 s |
| `ROCKE_SEQSUM_BLK=32` | left-to-right sums as blocks of 32 written out in order (same additions, same order) | bitwise equal to the scan (`speed_seqsum_test.py`) |
| `ROCKE_GLUE_JIT=1` | `_tpl_own_state` as one jit | selections only |
| `ROCKE_ENT_BATCH=1` | Ent: leaf evaluations of the six qsimp levels and of the sunlit/shaded leaves in one batched call | not bitwise equal to the level-by-level version (CPU: 1e-15 relative in 16-36 of 1,146 cells); `ent_step` on the V100 5.39 -> 2.37 ms |
Tried, not adopted (default off, kept for the record): `ROCKE_ENT_SKIP`, `ROCKE_NEWTON_UNROLL`, `ROCKE_WHILE_UNROLL`, `ROCKE_SCAN_UNROLL` (N=80 exhausts host memory in compilation, N=16 slower), `ROCKE_FUSED_DYN` (14 min compile per DIAGA phase), XLA command-buffer flags incl. WHILE (full step 0.371 -> 0.406 s).
Diagnostics (no effect on results): `ROCKE_DYN_TIMERS`, `ROCKE_SURF_TIMERS`, `ROCKE_SURF_PIECES`, `ROCKE_SURF_TRACE`, `ROCKE_STAGE_TRACE`, `ROCKE_RADIA_TRACE`, `ROCKE_OCEAN_TIMERS`, `ROCKE_POLE_PROFILE`, `ROCKE_MSTCNV_HOST_TIMING`.

## Result (exclusive V100 node, 108 steps; job ids in the log)
| | baseline (37941820) | all flags above (37941870) |
|---|---|---|
| held step median | 1.71 s | **0.385 s** |
| radiation step median | 1.88 s | 0.549 s |
| steps 2-107, mean per step | 1.75 s | **0.434 s** |
| cold start (step 0) | 1,090 s | 925 s |
| day-1 worst ratio (steps >= 3) | 2.37 QCI step 3 | 2.47 QCI step 3 |
| day-2 worst ratio | 1.19 | 1.11 |
The band is NOT bitwise unchanged: the failing step and field are the baseline's (QCI at step 3, 2x criterion not met), the ratio moves by about 0.1 (2.37 -> 2.46/2.47) as soon as the arithmetic of the cloud-base stage is the NumPy port's rather than the XLA GPU program's; "beyond" counts per field are the baseline's in the imf variant. Owner decision 2026-10-13: acceptable for now.

## Requirements and risks
* numba-capable interpreter for the worker (`ROCKE_MSTCNV_PY`); first call compiles for about 80 s (cached beside the source for the libm variant, not for the imf variant). The worker is a child process: a crash of the parent removes its shared file at EOF.
* `ROCKE_POLE_ASYNC` uses one extra thread; all host callbacks of the pole columns stay NumPy (LSCOND etc.).
* Regression suite: run on this branch with flags off (see the log, section 13).
* Cold start still about 15 min of XLA compile (rule D175, no persistent cache).
