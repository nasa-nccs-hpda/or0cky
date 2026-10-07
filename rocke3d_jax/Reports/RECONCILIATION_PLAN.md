# Plan to reconcile the independent review with the current state of the port

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07 in response to an independent review written by a second (monitoring) session.
Status: DRAFT for owner approval. Nothing in section 3 has been started except where marked.
Review by: after the Phase 0 decisions are made.

## 1. The observations, checked against the repository

"Agree / partly / disagree" is this session's reading after checking the facts named in the third column.

| # | Observation of the review | Reading | Facts checked |
|---|---|---|---|
| O1 | The Fortran-oracle method (instrumented real model, full routine outputs, recorded-input list) is strong and caught real errors. | Agree | The wrong OMEGA constant, the land-surface errors (D135, D136), the `EDIFS` land-ice defect (D170) and the REAL*4 `DZDH1` literal (D167) were all found this way. |
| O2 | The fidelity work is ahead of the JAX deliverable: the coupled results are Python/NumPy orchestration, not a compiled, device-resident JAX model step. | **Agree for the full-fidelity track** | 74 of the 288 modules in `fullfidelity/` import `jax`. Among the chained drivers: `atm_step`, `ocean_step`, `land_chain` import it; `surface_loop_v2`, `land_chain_ent`, `atm_day_free_rad_persist` and `f3_diagnostics2` do not. `land_chain_ent` runs the scalar GHY because the batched JAX GHY cannot take Ent exports computed inside the loop. The validated JAX kernels exist per component (dynamics step 2.7x faster on CPU; clouds batched; ocean pieces), but they were never wired into one jitted driver (the plan says so itself, `FULL_FIDELITY_PLAN.md` line 159). **Qualification:** a different track exists: Track A, `p2saom40_driver.py`, a device-resident `jax.lax.scan` that measured 4.2x on GPU, but it is the "representative workload" that HQ moved away from. |
| O3 | The radiation server is a Fortran dependency; the result would be a hybrid; "never port SOCRATES" is a project decision, not a source requirement. | **Agree** | It is a standing rule given by the owner, not something in the two links. Radiation currently comes from the real RADIA through a server (~11 s per call) or from recorded values. A GPU deliverable would need either a CPU callback to Fortran, or a decision about radiation. |
| O4 | The reference data are not shown to be the supplied `P2SAoM40_003` run. | **Agree, with new evidence** | A real supplement file already sits in the repository root, untracked since 2026-10-01: `ANN4099.aijP2SAoM40.nc`, labelled "P2SAoM40_003 (LLF40 + updated aerosol/ozone input files for CMIP6 simulations, 1...", model years 4099 to 4100, 365 days, 1,073 variables. The local reference run (`ModelE_Support/prod_runs/P2SAoM40`) is labelled "P2SAoM40 (ROCKE-3D, based on P2SAoM40 template)", years 1949 to 1950, monthly `acc` files with 143 variables. Different label, period and file type: identity is NOT established. The two share the CMIP6 input names in the rundeck, which is suggestive, not proof. |
| O5 | The one-month F3 comparison is a project-chosen milestone, not demanded by the two links. | **Partly** | True for the month rung: it comes from `scoping/RADIATION_AND_F2_PLAN.md`, written by an earlier session. But the direction itself came from HQ: the full-fidelity plan (2026-09-24) records that "HQ wants a port whose answers can be defended against the real Fortran, not a representative workload". The acceptance target still has not been agreed with HQ. |
| O6 | The one-month ensemble is one season and eight members. | Agree | Stated in D165 (about +-25% on a standard deviation; January only). The leave-one-out calibration in D172 also shows that two criteria in the original plan are too strict for 7 to 8 members. |
| O7 | REAL*16 emulation and libm matching should not become the optimisation target. | Agree | D173 left the binary128 `Ti`/`Ti2b` opt-in. Bitwise agreement needs the Intel libimf and cannot carry to a GPU, so the transferable acceptance criterion is statistical. |
| O8 | The handoff says the D172 files are untracked and unverified. | Was true; now stale | D172 was verified and committed (`898ab42`). The README Handoff needs a refresh (action A7). |
| O9 | The supplement server returned 503, so its contents could not be verified. | Transient | A fetch by this session on 2026-10-07 returned the full listing: 100 pairs `ANN4000` to `ANN4099`, `aij` and `aijl`. |

Not re-checked by this session: the review's statements about what the paper says (neither session has read section 13; see `DOCUMENTATION_REVIEW.md`).

## 2. What the review implies, in one paragraph

The validation work is sound but the sequencing is open to challenge. Three questions have not been put to the person who set the assignment: what "port" means (components validated against Fortran, or an end-to-end JAX model?), what radiation is allowed to be, and which comparison counts as success (module fidelity, one coupled step, a month, or the 100-year climatology). Until those are answered, more month-scale fidelity work risks being the wrong investment. Meanwhile three cheap actions reduce the uncertainty: prove the identity of the reference data, measure exactly what is JAX today, and write the acceptance criteria down.

## 3. Plan

### Phase 0: decisions from the project lead (no compute; start now)
Put these four questions to the person who assigned the project (draft wording below; the answers go into `GOAL.md`):
1. **Meaning of "port":** is the deliverable (a) an end-to-end JAX model that runs a coupled step on an accelerator, (b) a Fortran-referenced set of validated JAX components, or (c) both, in what order?
2. **Radiation:** is a hybrid acceptable, with the real Fortran SOCRATES called from the JAX run (CPU callback), or must radiation also be JAX or emulated? (Our standing rule is that SOCRATES is never ported or modified.)
3. **Acceptance rung:** which of these counts: one coupled step vs the Fortran; a multi-day case with a noise-floor test; a one-month F3 comparison; the 100-year published climatology (infeasible on CPU: about 1.75 million steps)?
4. **Reference identity:** is `P2SAoM40_003` the run that all our references should correspond to, and is the repository rundeck meant to be identical to it?

### Phase 1: provenance manifest (about 0.5 to 1 day; can start now)
- **A1.** Record the existing evidence (the O4 row) in a new `Reports/PROVENANCE_MANIFEST.md`: file, label, period, variable count, checksums.
- **A2.** Compare the repository rundeck `decks/P2SAoM40.R` and the source version with the rundeck and code in the Zenodo record (10.5281/zenodo.14721184): download only those small files, not the 5 GB archive. Report differences line by line.
- **A3.** Map the 1,073 variables of `ANN4099.aijP2SAoM40.nc` to our AIJ/AIJL columns (`f3_diagnostics.py`), so annual-mean comparison is possible later.
- **A4.** Decide where `ANN4099.aijP2SAoM40.nc` lives (it is an untracked 7 MB data file in the repository root): move it under `ff_data/` or track it, with the owner's approval.
- **Limit to state up front:** a climate-level match is NOT testable with what we have, because the local reference run is a 1949-1950 start state and the supplement is an equilibrated run at year 4099. Provenance can only be settled for inputs, rundeck, source and executable.

### Phase 2: JAX coverage and a JAX-driven coupled step (the gate before more month-scale work)
- **A5.** A coverage matrix: for every stage of one coupled step, which implementation executes (Fortran-served, recorded input, NumPy, JAX), with module names and the measured cost. Built from the code, reviewed by hand. About 0.5 day.
- **A6.** A first JAX-driven path: compose the existing validated JAX stages (dynamics, clouds, ocean, ice, GHY) under as few `jit` boundaries as practical for one coupled step, listing every recorded or Fortran-served input explicitly, and compare with the NumPy chained step (bitwise with the documented XLA flags) and with the Fortran. Reuse the D178 driver (the `boundary provider` interface). Effort not yet estimated; the coverage matrix (A5) will give the first honest figure.
- **A6b.** Measure speed on the intended hardware: the repository already has GPU submission scripts (`compare_submit_gpu.sbatch`, `Dockerfile.gpu`) from the Track A work; this node has no GPU. Needs the owner to say which GPU host to use.

### Phase 3: written acceptance criteria
- **A7.** Refresh the README Handoff (stale items: D172 status, the plan criteria found too strict).
- **A8.** Draft `Reports/ACCEPTANCE_CRITERIA.md` for the lead's approval: fields and budgets for a coupled step; the statistical test for multi-day and month cases (the calibrated D172 scoring tool, with criteria that pass the leave-one-out test); what counts as a recorded or Fortran-served input; the radiation boundary decided in Phase 0.

### Phase 4: resume month-scale work only after Phase 0 and A5, A6 report
The model-month driver (D174, D176-D178) continues as the bridge between the validated components and any longer comparison, but new expansion of the month-scale diagnostics (more AIJ columns, more ensemble members, month-long runs) waits for the answers to questions 1 and 3.

## 4. What to do with the work already running

| Track | Recommendation |
|---|---|
| D175 speed (parallel and faster stage variants) | Continue: it serves both the JAX gate and the cost of any later run. |
| D176 and D177 (closing recorded inputs) | Continue: they shorten the "recorded inputs" list that the review asks to enumerate. They are bounded. |
| D178 (driver skeleton, day boundary, checkpoint) | Continue, with the boundary-provider interface: it is the natural place for A6. |
| Further F3-column and month-ensemble work | Pause until Phase 0 answers 1 and 3. |

## 5. Suggested wording for the four questions

1. "For this project, does 'ported to JAX' mean an end-to-end model I can run on a GPU, or Fortran-validated JAX components, or both (and which first)?"
2. "The real SOCRATES radiation is third-party Fortran. Is it acceptable that the JAX model calls it (through a CPU callback), or should radiation be reproduced in JAX, or replaced by recorded values?"
3. "Which result would count as success: one coupled step matching Fortran, a short multi-day run within the model's own noise, a one-month comparison against the real ensemble, or reproducing the published 100-year climatology?"
4. "Is `P2SAoM40_003` (the supplement run) the run our references must correspond to, and is the rundeck in the repository supposed to be identical to the one behind it?"

## 6. Sources
- `DOCUMENTATION_REVIEW.md` (the two supplied links and the limits of reading them).
- `FULL_FIDELITY_PLAN.md` (origin of the full-fidelity direction, Track A and Track B).
- `FULL_FIDELITY_DELTAS.md` D135, D136, D165, D167, D170-D173.
- Direct reads on 2026-10-07: the labels and attributes of `ANN4099.aijP2SAoM40.nc` and of `JAN1950.accP2SAoM40.nc`; the list of modules importing `jax`.
