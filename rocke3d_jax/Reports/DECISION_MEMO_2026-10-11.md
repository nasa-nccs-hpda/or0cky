# Decision memo, 2026-10-11: continue the JAX port or stop here? (V100, hybrid, two model days)

Prepared by the driver session for G. Tamkin. Facts are from `BENCHMARK_V100_PROTOCOL.md` sections 6 and 7 (jobs 37941503, 37941514-16, Fortran runs b1-b3 on node gpu005, exclusive). The decision is the owner's.

## 1. What was measured
One frozen test: nov26 restart, 54 steps (one day) for Fortran x3 and JAX; JAX also 108 steps (two days). Same node type, strictly sequential, compile reported separately. JAX is a HYBRID: ported physics on the V100; radiation by the unmodified Fortran server; Ent exports, land forcing, SURFACE template columns, tile radiation columns and CONDSE entry set still read from the real run's records.

## 2. Results
| | Fortran, 1 core | JAX hybrid, V100 |
|---|---|---|
| one day, wall | 138.5 s | 1,071 s (cold) |
| two days, wall | about 277 s | 1,393 s (1,233 s with the stall fix) |
| cold start (compile, once) | none | 733 s |
| non-radiation work per step | about 0.75 s | about 2.0 s (held steps) |
| radiation, per radiation step | 8.1 s | 10.8 s (8.4 s server + about 2.4 s transfers/packet) |
| cycle average per step | 2.4 s | 3.7-3.9 s (excl. cold start and day-boundary steps) |
| day-boundary step | not separate | 134 s, 44.5 s with `ROCKE_STAGE_ARGS=1` |
| stability | n/a | 108 of 108 steps finite; error does NOT grow from day 1 to day 2 (day-2 worst ratio 1.22) |
| day criterion, unrelaxed (ACCEPTANCE s9) | n/a | NOT MET (day 1, QCI 2.37 at step 3; QCL/U/P beyond 2x at steps 0-1) |
| day criterion, relaxed (s10) | n/a | MET, both days |

## 3. What this says
* The earlier statement "3.5 s challenge met" holds as a statement about the cycle average at 3.3-3.9 s per step (noise 0.2 s). It does not mean JAX is faster than the Fortran: the Fortran cycle is 2.4 s on this node, so JAX is 1.5-1.6x slower per step and about 2.4x slower per day once the day-boundary recompile is counted (1.8x with the stall fix), and compile adds 733 s once.
* Radiation is 68% of the Fortran step (RADIA 1.62 s of 2.37 s per step) and is the SAME code in both runs. JAX cannot win on speed while radiation stays on the host: it would need the other 32% of the step (0.75 s) to fall below what the JAX non-radiation work plus the radiation hand-off costs (now about 2.1 s per step), i.e. about 3x faster, and that still gives no more than parity.
* The only lever that changes that is a device radiation (the SOCRATES port: 135-295 h estimate, standing rule says no).
* What the experiment does show: the ported physics (dynamics, clouds, surface, ocean, ice, daily updates) runs stably on a V100 for two days inside the ensemble spread at a loose margin, with a hybrid boundary that is explicit.

## 4. Options
A. **Stop here.** Deliverables stay: frozen benchmark, scripts, two-day hybrid result, relaxed/unrelaxed table, documented limits.
B. **Continue narrowly (recommended if the aim is a standalone multi-day JAX model, not speed):** finish the day-boundary fix for `pre`/`pa` (small); port the recorded inputs that block a longer run, cheapest first (sizing entry first); rerun the frozen test after each. No speed claim against Fortran.
C. **Continue for speed:** requires a device radiation (SOCRATES) or a different goal, e.g. throughput of many ensemble members on one GPU (NOT measured; no claim).

## 5. Limits of this evidence
One JAX run per configuration (three V100 nodes, noise about 0.2 s per step); Fortran timed on the same node type, 1 core, 3 runs; two days of one start date; GPU result not bitwise with CPU/Fortran; relaxed criterion chosen after seeing the results; recorded inputs mean JAX was never run independently of the real trajectory for land, vegetation and surface templates.
