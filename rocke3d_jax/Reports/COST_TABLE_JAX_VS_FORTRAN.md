# Cost table: hybrid JAX step vs the Fortran (what is measured, what is not)

Built 2026-10-09 from numbers already recorded in this repository; nothing was re-measured for this table. Owner of the question: G. Tamkin. Status: DRAFT for review.
Purpose: answer "does the complete, non-replay hybrid run finish faster than Fortran when all its costs are counted?" without assuming. **Answer from the existing numbers: no, not today; the hybrid is several times slower than the Fortran on the CPU, and no valid GPU result exists.** The comparison is NOT apples to apples yet (see section 5).

## 1. What each number is (conditions matter)
All JAX numbers: configuration P2SAoM40, nov26 (1950-11-26), libimf host callback (the fidelity configuration), 5 cores (0-2,6-7) on a shared node, no compile cache, one run each, jax 0.5.3 CPU unless stated. "Replay" = radiation heating and several inputs replayed from the real records; "server" = radiation computed by the real Fortran radiation server through a host callback (the hybrid component), surface inputs of the packet built from our own state (D212).

## 2. Per-step cost, JAX hybrid on the CPU
| quantity | value | source |
|---|---|---|
| cold first step (includes compilation) | 457-490 s (compile 370-396 s, 124-126 compilations) | D191 s5, D203, D206 |
| steady step, replay radiation | 14.9-18.8 s (median 14.9, D206; 18.8, D191) | D206, D191 s5 |
| steady step, radiation server computed | about 16 s on non-radiation steps; about 31 s on the 11 radiation steps | D210 |
| radiation server, per call | about 12.5 s (RADIA + IO 138 s over 11 calls); callback total 166 s per day | D210 |
| radiation transfers per day | 370 MB device to host, 128 MB host to device, 11 host sync points | D210, D212 |
| step-time shares (D191, replay, steady 18.8 s step, timed run) | condse_mstcnv 46.4% (device program + libimf callbacks + QUS callback); dyn 13.1% (75 kernel dispatches from Python, libimf pow callbacks); condse_post 12.5% (LSCOND libimf callbacks, north-pole callback); record_load 11.3% (file system); remainder surface, post and ocean | D191 s5 |
| host callbacks per step | 176 QUS calls (250 MB), 4 OADVT2 pre-pass, 2 pole columns, libimf pow/exp callbacks | D191, D198 result JSON |
| jit executions / eager dispatches per step | 96-99 / about 490 | D191 |
| SURFACE rebuilds in a day | 2 (steps 0 and 48; 60-100 s each at rebuild steps) | D206 |

## 3. A whole day (54 steps), JAX hybrid on the CPU
| run | wall | steady steps | note |
|---|---|---|---|
| replay radiation (D203/D206) | 2,107 s (sum of per-step walls, shared node) | 15-20 s | includes 457 s cold step 0 |
| radiation server, computed (D210) | 1,653 s | about 16 s, 31 s on the 11 radiation steps | includes about 465 s cold step 0 (125 compiles) |
| derived: steps 1-53 only, server run (D210) | (1,653 - 465) / 53 = about 22.4 s per step averaged over the day | | derived by me, not logged |

## 4. Fortran reference (the weak part of this table)
| quantity | value | source and caveat |
|---|---|---|
| **matched run: real Fortran, same 54-step nov26 day, same restart, 1 core** | **202-222 s wall (mean 212 s over 6 runs)**, 3.4-3.75 s per step in the main loop (183-203 s) plus about 17-20 s start-up | D215: node forest204 (12 vCPU Xeon E5-2697 v3), pinned to core 7, OMP_NUM_THREADS=1, serial binary; two binaries (ensemble hook build and production) not distinguishable (run-to-run spread about 8%); node load average 3.7-9.4 during the runs although mpstat showed the cores idle, so NOT a provably quiet node; end states bit-identical across the six runs |
| instrumented real model, 54 steps | 5 min 23 s with 6 jobs running concurrently = about 6.0 s per step per job | D150; heavy per-step dump I/O; node load unknown |
| real month, ctrl run | 85 min 42 s real, 84 min 41 s user, 20 s sys for JAN1950 (`ff_data/ens_jan1950/ctrl/run.time`) | user is 99% of real, so the run used ONE core. One model day is 54 steps (86,400 s / 1,600 s), so 31 days = 1,674 steps and the cost is **about 3.07 s per step on one core** (derived by me; assumes January has 31 days and the step is 1,600 s). I/O of a normal run included |

## 5. What the numbers say, and what they cannot say
* Day cost without compile, server (non-replay) hybrid: about 22.4 s per step; with the cold compile counted: 1,653 s per day.
* Fortran (measured, D215): 202-222 s for the 54-step day on ONE core, about 3.5 s per step in the main loop (3.9 s per step counting start-up); the January month run gives 3.07 s per step on one core in other conditions.
* So, per step against 3.5 s: the hybrid on the CPU (5 cores, non-replay, 22.4 s per step) is about **6.4x slower**; the replay step is about 3.4x (12 s) to 5.4x (19 s) slower; per day with the cold compile counted 1,653 s against 212 s is about **7.8x slower**. These now compare the same day, restart and node type; the JAX timings were taken on a shared node in other sessions, so treat the ratios as indicative.
* **GPU:** 3 valid-looking runs on the A100 (jobs 58786705, 58788593, 58789387, all replay mode): cold 716-768 s (648-689 s compile), steady 13.1 s on a repeat of step 0, 16.6 s at step 1 and about 25 s at steps 2-5. These runs are NOT valid evidence: the ocean state is already NaN at the end of step 0 (D214 findings), so the step time includes NaN-contaminated work and the result makes no fidelity claim.
* **Not measured:** Fortran time on a provably quiet node and at other times of day (D215 measured it on the same node type, one core); the device-versus-host split of the non-replay step; the radiation share of a real Fortran step (so the best case with the radiation callback kept is unknown); the effect of removing the libimf callback (speed was never measured without it); any multi-step GPU run that is finite; a compile cache.
* **Science side:** the same-scientific-work condition is not met: the one-day criterion (ACCEPTANCE s9) is NOT MET on every run (QCL step 1 = 2.83 and near steps), part of the inputs are still recorded (Ent, land forcing, template columns), and the GPU run is not valid. A speedup claim cannot be made until both sides hold.

## 6. What a clean measurement would need (cost estimate is a guess, not measured)
1. Same restart, same steps (say 54), same node and the same number of cores for the Fortran (uninstrumented) and the JAX run; record both walls. Cheap if the Fortran can be re-run (the real model reproduces itself).
2. A non-replay JAX day (server radiation, own-state inputs) timed with the cold compile reported separately, plus a per-stage blocked run to split device, callbacks, transfers and template work.
3. Radiation share of the Fortran step from its own timers, to bound the best case with the radiation kept as a callback.
4. A finite GPU multi-step run (blocked on the D214 NaN) in the same non-replay mode.
