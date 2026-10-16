# Acceptance criteria for the first JAX-driven coupled step and the multi-day run

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07.
Status: **APPROVED by the project owner on 2026-10-07** as drafted (sections 1 to 6). It was approved BEFORE any coupled step was compared, as required (a point of the independent review: criteria are set before the run, not after). Open points of section 7 at approval: points 1 and 2 are approved as proposed; for point 3 the draft's rule stands (recorded inputs are allowed in the first coupled step if every one is listed, section 1.5) and no specific input is required to be computed first unless the owner says so later; point 4 (the GPU host) is still to be named. Any later change to these criteria must be dated and recorded here, never applied silently after a result is known.
Review by: when the GPU host is named, or when the owner changes a criterion.

Direction fixed by the owner on 2026-10-07 (see `GOAL.md`): end-to-end JAX first, validated components second; success is one coupled step first, then a multi-day run; a Fortran radiation callback is acceptable but must be noted in every result that uses it; match `P2SAoM40` first.

## 1. What "JAX-driven" means here

A step counts as JAX-driven only if ALL of these hold; each one is reported as a line in the result file:

1. **State updates:** every prognostic variable that the step advances (atmosphere T, U, V, Q, cloud water, P and the moment arrays; ocean and ice state; lake and land state) is updated by a JAX function, with the arrays living on the JAX device between stages. A Python or NumPy driver that merely calls JAX kernels and copies results back and forth does **not** qualify.
2. **Boundaries:** one `jit` is not required. The number of `jit` boundaries, the number of host-device transfers per step and their sizes are measured and reported.
3. **Non-JAX stages:** any stage that executes as NumPy, as eager per-cell Python, or as Fortran is listed by name with its share of the step time. Nothing may be omitted from that list.
4. **Radiation:** the Fortran radiation callback (the real RADIA through the persistent radiation server; SOCRATES is not ported) is an explicitly bounded exception: the result states the packet inputs sent, the outputs received, the number of calls, the bytes transferred, the time spent and the host synchronisation points. Every table, plot and statement of the result carries the sentence "radiation computed by the original Fortran (hybrid component)".
5. **Recorded inputs:** every input still taken from a real-model record instead of being computed is listed with its source file and its size (the inventory is in the D174 ledger entry). No recorded input may be used that is not in the list.

## 2. Two comparisons, reported separately

Every result states which comparison it supports:

- **C1, port consistency:** the JAX-driven step against the NumPy chained step (`atm_step` and the surface loop), same start state, same core affinity, same thread count, the documented XLA flags set before importing JAX, no persistent compile cache (all four are conditions found necessary by D139, D145 and D175).
- **C2, fidelity:** the JAX-driven step against the real Fortran model: the state after one step from the real restart, using the real model's own step-boundary dumps.

A pass of C1 says the JAX code reproduces our existing NumPy port; it does not say the port reproduces ROCKE-3D. Only C2 supports a statement about ROCKE-3D.

## 3. Gate for one coupled step (rungs F1 and "coupled")

Categories, stated in advance by the project plan (`scoping/ATM_STEP_PLAN.md` section 5) and **not loosened here**, applied per prognostic field to the maximum absolute difference as a fraction of the field's scale:

- **A** bitwise equal. **B** at most 1e-12 of scale (rounding level). **C** at most 1e-6 of scale. **D** worse.

The step is **met** only if, for the start state used, every prognostic field of the end state is in A or B, except a short list of named exception columns (at most 10 columns per field, each at most 1e-9 of scale, listed with their coordinates). **Partly met:** the step runs end to end with named exceptions in B or C and the first failing stage identified. **Not met:** otherwise.

What the step must cover for "coupled": atmosphere (dynamics, surface fluxes, boundary layer, clouds and convection), land, ocean, sea ice and lakes, through the surface loop, with radiation by the Fortran callback or, from the SOCRATES port of 2026-10-12, by the SOCRATES JAX port (`--rad native`); a result says which (amended 2026-10-16). The start states are the three existing dates (`nov26`, `dec01`, `jan01`); the gate is reported per date, not pooled.

Libm: bitwise (A) results require the Intel libimf runtime on the host (`intel_libm_ff.py`); without it the comparison is reported at B or C and says so.

## 4. Gate for the multi-day run (rung F2, statistical)

Bitwise agreement is not expected beyond the first step (chaos; threshold flips). The comparison is against the real model's own noise floor:

- **Window and start:** 54 steps (one model day) from `nov26` first, then longer windows as the cost allows; the radiation call (Fortran callback, or `--rad native`) every fifth step (amended 2026-10-16).
- **Reference spread:** the five real one-ulp members of the nov26 day (D151) for the day; for longer windows the 8 real JAN1950 members (D165), with the scoring tool of D172 (calibrated leave-one-out).
- **Criterion for the day:** for T, U, V, Q, P and the cloud condensates, the difference from the real run stays within the real members' own spread at every step, and never exceeds 2 times the largest member distance (the D151/D157 convention; "within", "near", "beyond" as defined there). Reported per field and per step, with the worst ratio.
- **Criterion for longer windows:** the per-field statistics of the D172 scoring tool must lie inside the range obtained when each real member is scored against the others (leave-one-out). The plan's earlier rule ">= 95% of zonal bins within 2 sigma" is **not used** because the leave-one-out test shows true members fail it with 7 to 8 reference members.
- **Always reported:** the number of recorded inputs, whether the surface was replayed or computed, and the radiation callback sentence of section 1.

## 5. What a result must NOT claim

- It must not claim to reproduce the published 100-year equilibrated climate of `P2SAoM40_003`: our reference run is a one-year cold start (`PROVENANCE_MANIFEST.md`), and the identity of the two runs is not established. The configuration match (source, rundeck template) is established; the climate match is not tested.
- It must say which radiation a result used: the Fortran radiation callback (RADIA through the radiation server, `--rad server`) or the SOCRATES JAX port (`--rad native`, `fullfidelity/socrates_spike/`). It may describe radiation as ported to JAX only for results run with `--rad native`, and only within the limits recorded for that port (sections 12-12f). (Amended 2026-10-16 after the SOCRATES port of 2026-10-12; the earlier wording was: "It must not describe radiation as ported to JAX.")
- It must not call a step "end-to-end JAX" if any item of section 1 fails; it says which items fail.

## 6. Performance report (separate from correctness)

Reported with every correctness result: wall time per step (first step with compilation, steady state), the core count and device, the share of time per stage, and whether the compile cache was used (it must not be, for validated runs). Speed on an accelerator needs a GPU host; this node has none (see `RECONCILIATION_PLAN.md`, A6b).

## 7. Open points for the owner

1. Approve section 3's use of the existing A/B/C/D categories and the exception rule unchanged.
2. Approve section 4's use of the D151/D157 convention for the day and the D172 scoring rule for longer windows.
3. State whether the first coupled-step gate may use recorded Ent exports and the other recorded inputs listed in the D174 inventory (they must be listed, section 1.5), or whether specific ones must be computed first.
4. Name the GPU host for the performance report (A6b).

## 8. Decisions added by the owner on 2026-10-07 (after the approval of sections 1 to 6; additions, nothing in sections 3 and 4 is loosened)

1. **libimf and the fidelity comparison (C2):** the headline C2 uses libimf semantics through a **labelled host callback** (the Intel libm `pow`/`exp` of the real build), because without libimf cloud thresholds flip in 3 to 5 percent of columns and the gate cannot reach category A or B. The callback is another hybrid item: every result using it says so, next to the radiation sentence of section 1.4 ("libimf math functions provided by a host callback to the original build's runtime"), with its call count and time. C2 is ALSO reported in libm mode (rounding level, expected category C or worse at the threshold columns), labelled as such. Speed is measured separately WITHOUT the libimf callback, and is not described as the fidelity configuration.
2. **Radiation-packet surface side:** computed where verified (D176 `drv_radpacket.py`: bitwise at step 33312 of nov26), recorded and listed (section 1.5) where it is not.
3. **"JAX-driven" with several jit units:** three jit units (atmosphere phase 1, surface and ocean, atmosphere phase 2) plus the host calls for radiation and libimf SATISFY section 1, provided the state stays device-resident between stages and the boundaries, transfers and every non-JAX stage are reported.
4. **Host layer:** the D176 files are verified and adopted as the host-side provider layer; the D178 files (`model_driver.py`, `drv_daily.py`) are adopted after the parent session has verified them.
5. **Still open:** the GPU host (section 7 point 4).

## 9. Owner decision of 2026-10-08 on the day criterion (section 4); nothing in sections 3 and 4 is loosened

1. **Pass rule for the one-day run (54 steps):** "never beyond 2 times the largest member distance, at every step, for every scored field" (the D151/D157 convention), over ALL steps. "Within the members' spread at every step" is reported as a secondary line, not the pass rule.
2. **Status of the nov26 day (D193, D195):** NOT MET on this rule. The only field beyond 2x is QCL at step 1 (ratio 2.84: ours 1.15e-6 vs largest member distance 4.06e-7). Steps >= 3: worst ratio <= 1.17 in D193 and 1.13 in D195 (reported as additional information only). The earlier statements that the floor at steps < 3 is "~1e-9" were wrong (correction in `FULL_FIDELITY_DELTAS.md`, commit 8a649f7); no exception clause or exclusion of steps 0-2 is recorded.
3. **Possible later relaxation:** the owner said (2026-10-08) the criteria may be relaxed later only to advance to the GPU run. If that happens it must be recorded here as a dated owner decision made AFTER the results were known, with the unrelaxed status stated next to it; every result report will then carry both statuses.
4. **Work queued:** a short QCL step-1 diagnosis; leave-one-out scoring with the 8 JAN1950 members (D172) for longer windows.

## 10. Owner decision of 2026-10-11 (G. Tamkin): relaxed day criterion for the V100 hybrid experiment, reported BESIDE the unrelaxed status
Context: the owner declared the work an experiment with no wrong answers, not a production migration, and asked to decide today whether to continue; relaxation of the day criterion was authorised ("even if we have to relax the threshold"). Section 9 item 3 required the relaxation to be recorded as a dated decision made AFTER the results were known, with the unrelaxed status next to it. Both hold here.
1. **Relaxed rule (experiment status, "RELAXED-MET"):** every step finite (atmosphere and ocean) AND, at steps >= 3, never beyond 3 times the largest member distance for every scored field. Steps 0-2 are reported, not gated (the member floor there is tiny and ratios are unstable). The factor 3 and the exclusion of steps 0-2 were chosen by the agent on 2026-10-11 AFTER seeing the results below (worst ratio at steps >= 3: 2.37); a MET under this rule shows the run stays inside the ensemble spread at a loose margin; it does NOT show agreement at rounding level.
2. **Unrelaxed rule (section 9 item 1)** remains the standard for any claim of fidelity; every report carries both.
3. **Statuses known on 2026-10-11 (V100, libm mode, nov26 day 1):**
| run | unrelaxed (s9) | relaxed (s10) | worst ratio steps >= 3 | worst ratio steps 0-2 |
|---|---|---|---|---|
| GPU replay-radiation day (job 37941501) | NOT MET (U 2.61, QCL 2.95 at step 1; P 25.8 at step 0) | MET | 1.44 (QCI step 47) | P 25.79 step 0 |
| GPU server-radiation day, radiation inputs from own state (baseline v1, job 37941503) | NOT MET | MET | 2.37 (QCI step 3) | P 25.48 step 0; QCL 7.34 step 1 |

## 11. Owner decision of 2026-10-11 (G. Tamkin): the SOCRATES rule is lifted for a SPIKE only
The standing rule "SOCRATES is never ported or modified" is lifted for a time-boxed spike whose scope, limits and deliverables are in `SOCRATES_SPIKE_BRIEF.md`. Everything else under the rule stays: no edit of the SOCRATES/ModelE trees (an instrumented copy needs a separate approval), no change of the production radiation path, no full port without a further dated decision. The spike's result is a go/no-go recommendation against the gate in the brief; the owner decides.

## 12. Owner decision of 2026-10-11 (G. Tamkin): D3 approved, instrumented scratch copy of SOCRATES for stages R1-R2 only
Approved after the spike result (`SOCRATES_SPIKE_RESULT.md`, recommendation GO to R1-R2). Scope: (R1) reachability and the exact option set of the rundeck; (R2) an INSTRUMENTED COPY of the SOCRATES/ModelE radiation sources in a SCRATCH tree, dumping intermediate arrays for the recorded packets, so that stage-level oracles exist for the JAX kernels and the assumed 27 % of the RADIA time is measured. Time box: the plan's own estimates, R1 6-12 h + R2 6-12 h of focused work; stop and report if R2's dumps do not reproduce the server's final outputs bitwise or if either stage exceeds twice its estimate. Conditions: the original SOCRATES and ModelE trees stay untouched (edits only in the copy); the production radiation path, the persistent server and `jax_radiation.py` are not changed; the copy and its dumps are not committed (referenced by path); no claim of a port or of a model speed-up. NOT decided: the full port (R3-R8), the `interp_aer` edit of the Fortran, funding. Those need a further dated decision after the R1-R2 report.

### 12a. Addendum, 2026-10-11 evening (owner away 6 h; recorded by the SOCRATES session)
The owner challenged the session to "finish the SOCRATES port overnight" and asked it to keep going without waiting. The session treated this as the go for R3 onward on branch `socrates-spike` (worktree `ilab-agentic-ai-socrates-imvi`). Constraints kept: the original SOCRATES/ModelE trees are not modified (all instrumentation is in the scratch copy and in `fullfidelity/socrates_spike/ffsoc/`); the `interp_aer` Fortran edit and the funding of R6-R8 are NOT decided by this addendum. Result and honest status: `SOCRATES_R2_DUMPS.md` sections 10-11 (R3-R5 validated for the rundeck options, from atmosphere to band fluxes, rounding-level agreement; the ModelE-side interface R6, the production-size batched GPU run R7 and the validation campaign R8 are NOT done).

### 12b. Addendum, 2026-10-12 (owner: "proceed to R6"; recorded by the SOCRATES session)
R6 (the ModelE side of the radiation call) was done for the rundeck on branch `socrates-spike`: RADIA/RCOMPX set-up, planet_rad set-up and read-out, COSZS, the output assembly and a drop-in `NativeHandoff`. Ten consecutive radiation steps of nov26 reproduce the recorded server outputs (21 fields, AIJ excluded) at rounding level, CPU and V100; one V100 does the whole step in 0.174 s. Constraints kept: original trees untouched; `jax_radiation.py` (other session's branch) not edited; the `interp_aer` edit, funding of the remaining work and A100 confirmation remain undecided. Not done: daily climatology products for other dates (estimate 15-30 agent-hours), the 750 AIJ diagnostics, integration into the full step, R8. Details and the list of reference-model findings (float32 literals in REAL*8 data, a threshold in the sea-ice albedo): `SOCRATES_R2_DUMPS.md` section 12.

### 12c. Addendum, 2026-10-12 (owner: "the daily-products reader only"; recorded by the SOCRATES session)
The hold on the daily-products reader was lifted for the reader alone (validation otherwise stays limited to nov26 and the oracle dates it needed). `daily_products.py` reproduces the day-varying products for other calendar days: bitwise on 38 dates against the instrumented persistent server, rounding level for the whole radiation step at those dates and at the recorded nov27 step (`SOCRATES_R2_DUMPS.md` section 14). Not decided and not done: the `interp_aer` Fortran edit, funding of the remaining work, A100/float32, the AIJ diagnostics, a year-long run; the original SOCRATES/ModelE trees and `jax_radiation.py` were not touched.

### 12d. Addendum, 2026-10-12 (owner: "do items 1-2 now"; recorded by the SOCRATES session)
Items 1-2 of the recommended next steps, inside the same scope as 12c: (1) day products and zenith grids on demand by host callbacks (`RadiaJax(day_source=)`, `NativeHandoff(zen_source=)`): works, free on CPU, but 6-9 s per radiation step on the V100, so GPU runs keep the per-segment day stack; (2) the volcanic file (RADN7 + RADN3) read by `daily_products.py` and a master year inside the file range tested (`master_yr = 1990`, `volc_yr = 0`, scratch rundeck copy and scratch run directory only). All products bitwise against the instrumented server on 48 dates; the whole step at rounding level at 9 of 10 dates of the 1990 run; the tenth differs because land cover (`PVT`) is not a product of the files read (`SOCRATES_R2_DUMPS.md` section 14.4). Not decided and not done: the `interp_aer` Fortran edit, funding, A100/float32, AIJ, a year-long run, a land-cover reader; no original SOCRATES/ModelE file and no port-tree file was touched.

### 12e. Addendum, 2026-10-12 (owner: "approve the year-long validation run"; recorded by the SOCRATES session)
Year-long check inside the same scope as 12c/12d: the recorded nov26 packet served to the Fortran server at the first radiation step of 365 consecutive days (control build, scratch run directory only), compared with `radia_jax` using `daily_products.py` (CPU float64, `SOCRATES_R2_DUMPS.md` 14.6): all 365 dates at rounding level (SRHR <= 9.8e-9, TRHR <= 1.8e-7 W/m2, FSF <= 7.0e-9, T <= 7.2e-7 K; Q, KLIQ, CLDSS, CLDMC, CFRAC identical). Limits stated there: fixed state (no free-running year), one radiation step per day, CPU only. The default path of the hand-off is unchanged from 24fc300 (14.7). Not decided and not done: the `interp_aer` Fortran edit, funding, A100/float32, AIJ, a land-cover reader; no original SOCRATES/ModelE file and no port-tree file was touched.

### 12f. Addendum, 2026-10-12 (owner: "do your recommendations in order"; recorded by the SOCRATES session)
Land-cover reader inside the same scope: `daily_products.LandCover` builds `PVT` / `frac_veg` of the master year from the `VEG` and `CROPS` files exactly as the model does at its first daily update (753 land columns, 1850 and 1990, max difference 0.0); with it the master-year-1990 whole step is at rounding level at 9 of 10 dates (the tenth is the request made before the model's first daily update). Scope of validity written in `SOCRATES_R2_DUMPS.md` 14.8. Not decided and not done: the `interp_aer` Fortran edit, funding, A100/float32, AIJ, the coupled free-running year (port session). No original SOCRATES/ModelE file and no port-tree file was touched.
