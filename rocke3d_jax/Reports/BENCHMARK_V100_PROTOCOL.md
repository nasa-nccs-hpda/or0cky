# V100 benchmark protocol: JAX hybrid against the real Fortran (written 2026-10-11, owner decision: A100 no longer required, V100 is the deployment target)

Owner: G. Tamkin. Status: DRAFT protocol; results are appended in section 6 only when measured.

## 1. What is compared
* **Same system:** one V100 node of the ADAPT `compute` partition (job requests `--exclusive`; node, CPU model and load average recorded at the start of the log). Fortran and JAX run in ONE job, strictly one after another, so neither sees the other's load.
* **Same restart and window:** nov26 restart `_pristine_restarts/fort1_nov26_itime33312.nc`, itime 33312, 54 steps = one model day (Fortran: end 1950-11-27 03:00); then a second day (108 steps) for JAX only.
* **Fortran:** production hook-free binary (`P2SAoM40.bin`), one process, `OMP_NUM_THREADS=1`, pinned to one core, 3 runs (`gpu/fortran_day.sh`). Reported: real, user, sys, model-log main loop; per-step = /54.
* **JAX:** `gpu_day_run.py` with `d212_day.py --rad server --surf-fields computed --seed computed --cosz computed`, libm mode, one V100, 8+ host cores; radiation computed every NRAD step by the unmodified Fortran persistent server.

## 2. Equivalent physics: what is computed and what is still recorded (the hybrid boundary)
Computed in this benchmark: atmosphere dynamics, CONDSE/MSTCNV, surface, ocean, ice (JAX, on the device); RADIA/SOCRATES (unmodified Fortran, host server); the radiation packet's 21 surface fields, the seed chain and COSZ from OUR state (D212); the day-boundary DAILY_ATMDYN, ch4ox, daily_LAKE, land fractions and constants (host NumPy/JAX).
Still RECORDED (taken from the real run's files, so the JAX run is not yet a standalone model): Ent vegetation exports, land forcing, the SURFACE template columns, tile radiation columns, CONDSE entry set (see `recorded_inputs` in the result JSON). A fair speed statement must carry this sentence. Radiation replay is NOT used in this benchmark. Removing the remaining recorded inputs is separate work.

## 3. Compilation, reported separately
* **Cold start:** step 0 wall, with the XLA compile seconds from the run log (about 540-680 s on this stack). No persistent compile cache is used (D175 rule).
* **Amortized:** (cold start + sum of steady steps) / steps, for 54 steps (one day), 108 steps (two days) and extrapolated to 1, 10, 30 days by cold + days x steady-day (extrapolation labelled as such). Steady day = mean of steps >= 2, including radiation steps (cycle average) and the day-boundary work.
* The Fortran has no compile cost; its start-up (about 17-20 s) is reported separately from the main loop.

## 4. Fidelity beside speed
* GPU result is not bitwise with the CPU/Fortran (device exp/pow); judged by `multiday_score.py` against the five real one-ulp members (day criterion ACCEPTANCE s9: never beyond 2x the largest member distance; status stays NOT MET unless it actually passes, any relaxation is a dated owner decision).
* Compared with the CPU server-mode day (D212) and replay-mode CPU days (D206/D209/D210). Finiteness per step (atmosphere and ocean) is checked.

## 5. Not claimed
No speed claim beyond: this node, this restart, this day, these inputs. Fortran on a quiet node, other days, other months are not covered.

## 6. BASELINE RESULT (frozen 2026-10-11; job 37941503, node gpu005 Xeon Gold 6248, exclusive; commit of this file's parent; V100 baseline v1)
Boundary: hybrid as in section 2 (recorded Ent / land forcing / SURFACE template columns / tile radiation columns / CONDSE entry set; radiation computed by the unmodified Fortran server; cosz computed with libimf on the host). One JAX run, 54 steps (records exist for 54 steps only; the requested 108 stopped at step 54 with FileNotFoundError, expected).
| quantity | Fortran, 1 core (3 runs) | JAX hybrid, V100 (1 run) |
|---|---|---|
| whole day, wall | 138.0-139.1 s real (main loop 127.2-128.7 s, start-up about 10 s) | 1,071 s (cold step 0 731 s of which XLA compile 641 s; steps 1-53 340 s) |
| steady step, no radiation (steps 2-47, 49-53) | per-step not timed | 2.02 s (41 steps) |
| radiation step (11 per day) | server request 8.4 s per call | 11.45 s (10 steps, excl. step 0) |
| cycle average, excl. cold start and step 48 | 2.4 s per step (main loop / 54) | 3.91 s per step (1.6x Fortran) |
| steady JAX day incl. step 48 | | 43 x 2.02 + 11 x 11.45 + 135 = about 350 s per day after the first (2.5x Fortran) |
Fidelity (multiday_score, five real members): within/near/beyond T 34/20/0, U 25/28/1, V 30/24/0, Q 33/21/0, P 38/15/1, QCL 44/8/2, QCI 23/30/1; worst ratio at steps >= 3: QCI 2.37 (step 3), QCL 1.80 (step 4); all steps finite; day criterion NOT MET (beyond 2x at steps >= 3). Replay-radiation GPU day (job 37941501, same code, radiation replayed): worst ratio at steps >= 3 is 1.44 (QCI step 47), beyond 2x only at steps 0-1; 1.8-2.0 s per step across the day boundary, no stall. Cause of the fidelity difference between the two GPU days not isolated (candidates: computed radiation packet inputs, cosz via libimf, daily_LAKE and constants hooks, live radiation).
### Step 48 stall (137 s), investigated
Evidence in `d193_run.json`: step 48 has `stage_rebuilt = True`, 8 new XLA compilations costing 115.3 s (compile seconds cum 643.3 -> 758.6), the rest (about 20 s) is Python tracing, lowering and host rebuild. Cause: the day-boundary `daily_lake_update` (632 cells with FLAKE changed) rebuilds the static geometry (`make_static_all`), calls `_build_jits(only=('pre','pa'))` and sets `_stage_cache = None`; the next step rebuilds the SURFACE stage (`JS.make_stage(host)`, `make_apply_state`, `tile_accumulators_dev`), whose host constants are closed over by the jit programs, so they compile again. The replay day (job 37941501) did not run daily_LAKE and shows no stall. It will recur at EVERY day boundary (about 137 s per day, 39% of the steady JAX day). Same effect on the CPU (D206: 2 SURFACE rebuilds per day, 60-100 s at rebuild steps). Not yet done: confirm with a per-program compile list; fix candidates (owner decision, changes the compiled programs): pass the day-varying statics as traced arguments instead of closed-over constants; or rebuild only the programs that depend on FLAKE.
