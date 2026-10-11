# Budget for the owner's challenge: a non-replay steady step at or below about 3.5 s on one A100

Written 2026-10-10 from measured numbers only. Status: DRAFT, no new measurement was needed beyond reading the Fortran run log of D215.

## 1. Where the real Fortran spends its time (the model's own timers, D215 run `run_ens1.PRT`, 54 steps, one core, per-step averages)
| Fortran stage | per step | share | note |
|---|---|---|---|
| RADIA (radiation, NRAD = 5) | 2.38 s | 58.9% | 12.0 s on each radiation step (1 in 5), about 0.001 s on the other four |
| CONDSE (clouds and convection) | 0.53 s | 13.0% | |
| Atmosphere dynamics | 0.30 s | 7.4% | |
| SURFACE | 0.29 s | 7.7% | |
| OCEANS | 0.07 s | 2.0% | |
| daily, diagnostics, other | about 0.4 s | | one 1.22 s daily update per day |
| total | about 4.0 s | 100% | wall 3.9 s per step including start-up |
The Fortran spends 1.66 s per step outside the radiation.

## 2. The JAX GPU step now (job 58814135 profile, steady, replay, ocean split applied from job 58815058)
condse_mstcnv 6.20 s, ocean_b 1.46 s (split), condse_post 0.67, template_build 0.42, dyn 0.26 (before the dynamics being re-measured: 1.45 s inside ocean_b is the ocean dynamics, not atmosphere dyn), the rest about 0.3: about 8.3-8.6 s per step with the radiation REPLAYED (free).

## 3. What the radiation costs when it is NOT replayed
The real RADIA runs in a persistent Fortran server behind a host callback (D210): about 12.5 s per call, one call per 5 steps, 2.5 s per step averaged over the cycle (166 s per 54-step day measured, 3.1 s per step). This is the same code and about the same time as the Fortran's own RADIA (12.0 s). It cannot be sped up without touching RADIA/SOCRATES (standing rule: never modified), because the server runs the unmodified model's full-grid call on one core, and it cannot be overlapped with the step: the radiative heating computed at a radiation step is applied in that same step.

## 4. Consequence for the target (two metrics)
* **(b) cycle average including the real radiation call**, the like-for-like comparison with the Fortran's 3.5-4.0 s: the radiation callback alone costs 2.4-3.1 s per step, so everything else would have to fit into 0.4-1.1 s per step. The Fortran needs 1.66 s for the same non-radiation work. Beating that would mean running CONDSE, dynamics, surface and ocean on the A100 faster than one CPU core runs them (today 8.3 s against 1.66 s; the ocean alone 1.46 s against 0.07 s). **Not achievable under the current rules**; it becomes plausible only if the radiation moves to the device (the SOCRATES port, 135-295 h estimate, needs the owner to lift the standing rule), and then the rest still has to fall from 8.3 s to about 3 s.
* **(a) steps without a radiation call** (4 of every 5 steps): target 3.5 s against 8.3-8.6 s now, a factor of about 2.4. This is within the rules and is the first thing to achieve; it is also a necessary step for (b). The dominant stage is `condse_mstcnv` (6.2 s of 8.3 s); reaching 3.5 s needs it at or below about 1.5-2 s, with the rest unchanged.

## 5. Constraints on how fast this can be done
* No GPU on the development node: every GPU measurement is a job on Discover (compile about 11 min, run about 15 min), submitted and read back by the owner.
* A stage-level GPU benchmark (compile only the stage being optimised, from captured inputs; the xsweep probe took 14 s of wall) would cut a measurement cycle to a few minutes and is the first thing to build.
* Results must stay bitwise on the CPU (C1) and finite and in the ensemble spread on the GPU; no tolerance may be loosened.

## 6. Update 2026-10-11 (V100, after the D214 GPU optimization): the budget above was written for a step of about 8.3 s; it is now about 1.7 s
* **Steps without a radiation call (replay, V100):** 1.67-1.74 s steady (was 9.0 s at the start of the work, 3.98 s after the QUS port, 2.14 s after the pre-pass port). Target 3.5 s: met on the V100; A100 confirmation owed.
* **Why it fell so far:** the cost was not kernel launches but host callbacks inside jit programs, which on this software stack fire only after the whole device program has finished (cause not identified; a callback alone costs 2 ms). The three removals (QUS ADV1D with limiter and the OADVT2 pre-pass as device programs, bitwise equal to the NumPy originals; the pole columns as host steps between jits) account for 7.3 of the 7.3 s saved. What remains is launch-bound loops: the cloud-base stage 0.94 s (22 event blocks of 43 ms, 69% of a block in the plume ascent loop), dynamics 0.20 s (eager dispatches), template_build 0.20 s (host, replay scaffolding).
* **Cycle average with the REAL radiation (`gpu_server_step_run.py`, V100, job 37941496, 16 steps: radiation steps 0, 5, 10, 15):** radiation steps (5, 10, 15) 10.23-10.28 s, mean 10.25 s, of which the unmodified Fortran server request is 8.21-8.24 s; held steps (11 of them, steps 2-4, 6-9, 11-14) 1.80-2.03 s, mean **1.90 s**. Cycle average over NRAD = 5 (4 held + 1 radiation): (4 x 1.90 + 10.25) / 5 = **3.57 s per step**, i.e. 2% ABOVE the 3.5 s line. (A first 6-step job, 37941495, had 3 held steps at 1.64-1.74 s around one radiation step and gave 3.45 s: too few samples; the 16-step run replaces it.) Hybrid: radiation computed by the real Fortran in a persistent server; the 21 surface fields of the packet and a few inputs from the recorded packet; Ent exports, land forcing and the SURFACE template columns still recorded. Held steps are slower than in the replay-only run (1.9 s against 1.7 s): the server child takes two of the job's eight cores and the node and time differ. The Fortran radiation took 8.2 s per call on the GPU node's cores against 12.5 s in D210 on the shared dev host: the radiation cost is node-dependent, and the one-core Fortran model (3.5 s per step, D215) was timed on the dev host.
* **What this changes in section 4:** the cycle-average target (b) is no longer out of reach: it is the radiation call (8.2 s per call = 1.65 s per step averaged on this node) plus the held-step work (1.9 s) = 3.57 s per step, 2% above the line. To get below 3.5 s the held steps have to lose about 0.1-0.15 s (candidates: template_build 0.20 s host scaffolding, dynamics 0.20 s eager dispatches, the 22 cloud-base blocks 0.94 s). It is one V100, one node, and the A100 has not been run.

## 7. Owner decision, 2026-10-11 (G. Tamkin): the 3.5 s challenge is declared MET
Recorded as the owner's decision, made after seeing the results above. The facts it rests on, so the decision is not read as more than it is: steps without a radiation call 1.67-1.74 s and the cycle average with the real Fortran radiation server 3.33-3.57 s per step (four runs, noise 0.1-0.2 s), all on ONE V100 (ADAPT, jax 0.5.3), libm mode, hybrid (radiation computed by the unmodified Fortran server; Ent exports, land forcing, SURFACE template columns and part of the radiation packet still recorded; template prefetch is replay scaffolding), day-1 records only. NOT yet confirmed on the A100 (owner submits through Discover; commands in the 2026-10-11 hand-off). The unrelaxed day criterion (ACCEPTANCE s9) remains NOT MET; this decision concerns speed only.
