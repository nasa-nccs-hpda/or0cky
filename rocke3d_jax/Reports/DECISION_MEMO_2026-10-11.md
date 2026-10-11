# Decision memo, 2026-10-11: continue the JAX port or stop here? (V100, hybrid, two model days)

Prepared by the driver session for G. Tamkin. Facts are from `BENCHMARK_V100_PROTOCOL.md` sections 6 and 7 (jobs 37941503, 37941514-16, Fortran runs b1-b3 on node gpu005, exclusive). The decision is the owner's.

## 1. What was measured
One frozen test: nov26 restart, 54 steps (one day) for Fortran x3 and JAX; JAX also 108 steps (two days). Same node type, strictly sequential, compile reported separately. JAX is a HYBRID: ported physics on the V100; radiation by the unmodified Fortran server; Ent exports, land forcing, SURFACE template columns, tile radiation columns and CONDSE entry set still read from the real run's records.

## 2. Results
| | Fortran, 1 core | JAX hybrid, V100 |
|---|---|---|
| one day, wall | 138.5 s | 1,071 s (cold) |
| two days, wall | about 277 s | 1,393 s (1,104 s with the stall fix) |
| cold start (compile, once) | none | 733 s |
| non-radiation work per step | about 0.75 s | about 2.0 s (held steps) |
| radiation, per radiation step | 8.1 s (RADIA) | 10.3-10.8 s = an ordinary step (1.7-1.9 s) + 8.4 s server + 0.2-0.45 s hand-off (packet transfers) |
| cycle average per step | 2.4 s | 3.45-3.9 s (excl. cold start and day-boundary steps; 3.45 s with the stall fix) |
| day-boundary step | not separate | 134 s; 1.7 s with `ROCKE_STAGE_ARGS=1` (GPU result then differs from the unflagged run at rounding level, scores equivalent) |
| stability | n/a | 108 of 108 steps finite; error does NOT grow from day 1 to day 2 (day-2 worst ratio 1.22) |
| day criterion, unrelaxed (ACCEPTANCE s9) | n/a | NOT MET (day 1, QCI 2.37 at step 3; QCL/U/P beyond 2x at steps 0-1) |
| day criterion, relaxed (s10) | n/a | MET, both days |

## 3. What this says
* The earlier statement "3.5 s challenge met" holds as a statement about the cycle average at 3.3-3.9 s per step (noise 0.2 s). It does not mean JAX is faster than the Fortran: the Fortran cycle is 2.4 s on this node, so JAX is 1.5-1.6x slower per step and about 2.4x slower per day once the day-boundary recompile is counted (1.36x with the stall fix: 188 s against 138 s per steady day), and compile adds 733 s once.
* Radiation is 68% of the Fortran step (RADIA 1.62 s of 2.37 s per step) and is the SAME code in both runs. JAX cannot win on speed while radiation stays on the host: it would need the other 32% of the step (0.75 s) to fall below what the JAX non-radiation work plus the radiation hand-off costs (now about 2.1 s per step), i.e. about 3x faster, and that still gives no more than parity.
* The only lever that changes that is a device radiation (the SOCRATES port: 135-295 h estimate, standing rule says no).
* What the experiment does show: the ported physics (dynamics, clouds, surface, ocean, ice, daily updates) runs stably on a V100 for two days inside the ensemble spread at a loose margin, with a hybrid boundary that is explicit.

## 4. Options
A. **Stop here.** Deliverables stay: frozen benchmark, scripts, two-day hybrid result, relaxed/unrelaxed table, documented limits.
B. **Continue narrowly (recommended if the aim is a standalone multi-day JAX model, not speed):** the day-boundary fix is done (flag-gated); port the recorded inputs that block a longer run, cheapest first (sizing entry first); rerun the frozen test after each. No speed claim against Fortran.
C. **Continue for speed:** requires a device radiation (SOCRATES) or a different goal, e.g. throughput of many ensemble members on one GPU (NOT measured; no claim).

## 5. Limits of this evidence
One JAX run per configuration (three V100 nodes, noise about 0.2 s per step); Fortran timed on the same node type, 1 core, 3 runs; two days of one start date; GPU result not bitwise with CPU/Fortran; relaxed criterion chosen after seeing the results; recorded inputs mean JAX was never run independently of the real trajectory for land, vegetation and surface templates.

## 6. What is still recorded (inventory from `fullfidelity/scoping/D212_RADIATION_PACKET_OWN_STATE_ENTRY.md` section 4; sizes and rankings there are that entry's, the cost ranking is its hypothesis, not measured)
| recorded input (every step unless noted) | size | computed form in the repo | status |
|---|---|---|---|
| SURFACE templates ffp/ffs/ffl/ffg/fft: Ent exports (LAI, albedo, vegetation), land forcing columns, PBL template columns, tile rows and sets, soil static columns | about 1.3 GB per day | Ent: NumPy only (no JAX port; D169/D171 validated); GHY batched JAX exists but takes recorded Ent exports; `PBLCarry` (D177) not wired | largest piece |
| tile radiation columns (srheat, trhr0, qsol, trup_in_rad, vis_rad, dvis, Ca, cosz1) | part of the templates | `drv_radcols*.py` (D176), bitwise over the nov26 day, not wired in | the cheapest next one (D212 hypothesis) |
| CONDSE entry non-state fields (ffc_cse_in) | about 4.6 GB per day | not ported (cloud/precip carry, ground fields, tuning) | needs a decision on what the entry set contains |
| step-0 hidden/PBL start state, MMST, ADVSI geometry, per-date constants, first SEEDS[0], RQT/KLIQ/SNOAGE of the first call | small | statics and restart-derived | do not block a longer run |
Consequence for "blocking a multi-day run": the records come from the instrumented real model, so a longer hybrid run only needs more days of records (day 2 was produced that way: `nov26_day2`, 108 steps; about 6 GB of dumps and a few minutes of Fortran per day). It does NOT need new JAX code, but it keeps the JAX run tied to the real trajectory for land, vegetation and surface templates (errors in those pieces cannot show). A standalone JAX run needs the three largest items above ported; that work is not sized in hours here (no estimate exists in the repo; the SOCRATES estimate of 135-295 h is for radiation only).

## 7. The bar for any proposal aimed at beating the Fortran (added 2026-10-11 after the second-opinion review; corrected the same day)
Correction: the first version of this section used the older 3.7 s cycle and a wrong reading of the radiation step (10.8 s is an ordinary step plus the 8.4 s server plus a 0.2-0.45 s hand-off, not "8.4 s plus 2.4 s of transfers"). Numbers below are from the 3.45 s run (job 37941519, held 1.73 s, radiation step 10.32 s).
* Fortran cycle 2.37 s per step = RADIA 1.62 s + everything else 0.75 s. JAX cycle 3.45 s = radiation server 1.68 s (8.4 s / 5) + hand-off 0.04 s + ordinary step work 1.73 s.
* Parity needs the JAX ordinary step work (including the hand-off) at about 0.75 s while the radiation stays the same Fortran code (cycle then about 2.5 s, 5% above the Fortran because the server request is 8.4 s against RADIA 8.09 s); a WIN (below 2.37 s) needs about 0.65 s. Today 1.73 s: a reduction of 2.3x to 2.7x, about 1.0 to 1.1 s per step.
* A device radiation (SOCRATES port, off the table under the standing rule) is necessary for a win only if the server cost is to shrink; it does not touch the 1.0 s above.
### Proposed allocation (HYPOTHESES, none demonstrated; stage seconds from job 37941486, 1.70 s step, V100)
| stage | now (s) | target (s) | saving (s) | how, and what is known |
|---|---|---|---|---|
| cloud-base (condse_mstcnv) | 0.94 | 0.30 | 0.64 (about 60%) | 22 launch-bound event blocks of 43 ms, 69% of a block in the plume ascent loop. Round-based loop (slower, 1.78 s), unrolling (compile over 26 min) and XLA command-buffer flags already tried: no gain. Make-or-break item; no known route. |
| dynamics | 0.20 | 0.05 | 0.15 | 770 eager dispatches per step, 294 in the template build; fuse into jit programs |
| template build (host scaffolding) | 0.20 | 0.05 | 0.15 | prefetch (already built, opt-in) measured 1.71 to 1.60 s; the cost returns as device work once Ent/land forcing are computed, so this saving is NOT additive with the standalone port |
| everything else (condse_post 0.075, setup 0.067, post_a 0.060, ocean_b 0.045, surface tiles/land 0.037, radiation apply 0.033, rest) | 0.36 | 0.25 | 0.11 | small jit fusion |
| total | 1.70 | 0.65 | 1.05 | borderline: a miss of 0.1 s on any line removes the win; parity is reachable only if cloud-base reaches 0.3 s |
### How it is measured
Rerun the frozen job (`gpu/benchmark_v100.sh`, node type, restart, 54-step day, exclusive node, Fortran x3 beside it); report the held-step mean, the cycle average (4 held + 1 radiation, excluding step 0 and day-boundary steps, with `ROCKE_STAGE_ARGS=1` stated), the stage table from a `--timed` run, and the end-state score against the five members. A stage claim counts only if its stage seconds fall in the timed run AND the cycle average falls in the frozen run; noise is 0.2 s per step, so single savings below 0.2 s need repeated runs. Success for the speed track: cycle average at or below the Fortran's measured cycle average on the same node (2.37 s).
