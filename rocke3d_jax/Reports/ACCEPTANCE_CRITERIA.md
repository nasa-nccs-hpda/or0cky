# Acceptance criteria for the first JAX-driven coupled step and the multi-day run

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07.
Status: **DRAFT for the owner's approval; must be approved BEFORE the first coupled step is compared** (this was a point of the independent review: criteria are set before the run, not after).
Review by: before the first Phase 2 comparison is run.

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

What the step must cover for "coupled": atmosphere (dynamics, surface fluxes, boundary layer, clouds and convection), land, ocean, sea ice and lakes, through the surface loop, with radiation by the callback. The start states are the three existing dates (`nov26`, `dec01`, `jan01`); the gate is reported per date, not pooled.

Libm: bitwise (A) results require the Intel libimf runtime on the host (`intel_libm_ff.py`); without it the comparison is reported at B or C and says so.

## 4. Gate for the multi-day run (rung F2, statistical)

Bitwise agreement is not expected beyond the first step (chaos; threshold flips). The comparison is against the real model's own noise floor:

- **Window and start:** 54 steps (one model day) from `nov26` first, then longer windows as the cost allows; the radiation callback every fifth step.
- **Reference spread:** the five real one-ulp members of the nov26 day (D151) for the day; for longer windows the 8 real JAN1950 members (D165), with the scoring tool of D172 (calibrated leave-one-out).
- **Criterion for the day:** for T, U, V, Q, P and the cloud condensates, the difference from the real run stays within the real members' own spread at every step, and never exceeds 2 times the largest member distance (the D151/D157 convention; "within", "near", "beyond" as defined there). Reported per field and per step, with the worst ratio.
- **Criterion for longer windows:** the per-field statistics of the D172 scoring tool must lie inside the range obtained when each real member is scored against the others (leave-one-out). The plan's earlier rule ">= 95% of zonal bins within 2 sigma" is **not used** because the leave-one-out test shows true members fail it with 7 to 8 reference members.
- **Always reported:** the number of recorded inputs, whether the surface was replayed or computed, and the radiation callback sentence of section 1.

## 5. What a result must NOT claim

- It must not claim to reproduce the published 100-year equilibrated climate of `P2SAoM40_003`: our reference run is a one-year cold start (`PROVENANCE_MANIFEST.md`), and the identity of the two runs is not established. The configuration match (source, rundeck template) is established; the climate match is not tested.
- It must not describe radiation as ported to JAX.
- It must not call a step "end-to-end JAX" if any item of section 1 fails; it says which items fail.

## 6. Performance report (separate from correctness)

Reported with every correctness result: wall time per step (first step with compilation, steady state), the core count and device, the share of time per stage, and whether the compile cache was used (it must not be, for validated runs). Speed on an accelerator needs a GPU host; this node has none (see `RECONCILIATION_PLAN.md`, A6b).

## 7. Open points for the owner

1. Approve section 3's use of the existing A/B/C/D categories and the exception rule unchanged.
2. Approve section 4's use of the D151/D157 convention for the day and the D172 scoring rule for longer windows.
3. State whether the first coupled-step gate may use recorded Ent exports and the other recorded inputs listed in the D174 inventory (they must be listed, section 1.5), or whether specific ones must be computed first.
4. Name the GPU host for the performance report (A6b).
