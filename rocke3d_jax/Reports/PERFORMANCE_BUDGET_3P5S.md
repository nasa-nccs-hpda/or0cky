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
