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
| instrumented real model, 54 steps | 5 min 23 s with 6 jobs running concurrently = about 6.0 s per step per job | D150; heavy per-step dump I/O; node load unknown |
| real month, ctrl run | 85 min for JAN1950 | README ensemble row; if 31 days x 48 steps = 1,488 steps this is about 3.4 s per step; core count, MPI layout and I/O NOT recorded in the entries I read |

## 5. What the numbers say, and what they cannot say
* Day cost without compile, server (non-replay) hybrid: about 22.4 s per step; with the cold compile counted: 1,653 s per day.
* Fortran: about 3.4-6.0 s per step (two inconsistent sources) or about 184-323 s per day.
* So the hybrid on the CPU is about **3.7x slower (against 6.0 s) to 6.6x slower (against 3.4 s) per step**, and 5-9x slower per day with the compile counted. These ratios are indicative only: different cores, shared nodes, the Fortran figure comes from other conditions.
* **GPU:** 3 valid-looking runs on the A100 (jobs 58786705, 58788593, 58789387, all replay mode): cold 716-768 s (648-689 s compile), steady 13.1 s on a repeat of step 0, 16.6 s at step 1 and about 25 s at steps 2-5. These runs are NOT valid evidence: the ocean state is already NaN at the end of step 0 (D214 findings), so the step time includes NaN-contaminated work and the result makes no fidelity claim.
* **Not measured:** Fortran time on the same node and cores for the same steps; the device-versus-host split of the non-replay step; the radiation share of a real Fortran step (so the best case with the radiation callback kept is unknown); the effect of removing the libimf callback (speed was never measured without it); any multi-step GPU run that is finite; a compile cache.
* **Science side:** the same-scientific-work condition is not met: the one-day criterion (ACCEPTANCE s9) is NOT MET on every run (QCL step 1 = 2.83 and near steps), part of the inputs are still recorded (Ent, land forcing, template columns), and the GPU run is not valid. A speedup claim cannot be made until both sides hold.

## 6. What a clean measurement would need (cost estimate is a guess, not measured)
1. Same restart, same steps (say 54), same node and the same number of cores for the Fortran (uninstrumented) and the JAX run; record both walls. Cheap if the Fortran can be re-run (the real model reproduces itself).
2. A non-replay JAX day (server radiation, own-state inputs) timed with the cold compile reported separately, plus a per-stage blocked run to split device, callbacks, transfers and template work.
3. Radiation share of the Fortran step from its own timers, to bound the best case with the radiation kept as a callback.
4. A finite GPU multi-step run (blocked on the D214 NaN) in the same non-replay mode.
