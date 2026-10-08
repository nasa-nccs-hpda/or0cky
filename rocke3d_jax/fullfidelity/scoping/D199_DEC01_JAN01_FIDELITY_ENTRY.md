# D199 (2026-10-08): root cause of the dec01/jan01 step-0 C2 residual (1e-8) and its fix -- the land ground humidity `qg_ij` used the wrong `elhx`

Owner: G. Tamkin. Written by a Claude Code agent. Status: DRAFT ledger text; nothing committed. Review by: when the land/GHY hand-off changes.
Sources: D129, D136b, D177, D187, D188, D191 (ledger), `GHY_DRV.f` of `modelE2_planet_2.0` (read-only; lines 1071-1075, 1104, 1304-1312), real dumps `ff_data/dec01` (ffp/ffg/fft, `ffa_33552_c1_in/c1_out/c2_in/c2_out`).
Conditions: `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, `clouds_jax_env`, no compile cache, scratch only in the session scratchpad, libimf callback mode, radiation replayed, one run per number (`--runs 1`, cold step; the timing is not a result of this entry).

## 1. Root cause (measured)
Reproduced first (NumPy two-substep tool `chain_two_substeps.run_two_substeps(land=True)`, dec01): the ATURB exit after substep 2 vs `c2_out` has E 9.4e-8 abs (9.9e-9 of scale), PBLHT 8.9e-6, U 1.4e-8 -- the D191 C2 numbers.
Bisection of that step with `bis.py`-type variants (substep 1 always on real inputs; only the substep-2 inputs changed; E = max abs of EGCM vs the real substep-2 exit):
| substep-2 inputs | E max abs | U max abs |
|---|---|---|
| everything recorded, real ATURB entry state | 2.5e-13 | 2.2e-14 |
| recorded rows, OUR ATURB sub-1 exit (it equals the real `c1_out` to 3.8e-13 in E) | 2.6e-12 | 1.6e-12 |
| predicted ocean/ice/land-ice rows, RECORDED land rows | 3.2e-12 | 1.7e-12 |
| everything predicted (the C2 path) | 9.4e-8 | 1.4e-8 |
| recorded land rows, our carried GHY state | 3.2e-12 | 1.7e-12 |
| predicted land rows, only columns {qg_aver, qg_sat, tgv, tg, tr4} | 9.4e-8 | 1.4e-8 |
| predicted land rows, only {dbl, ug, vg, zs1, ztop, utop, vtop, qtop, tkv} or only {cm, ch, cq} or only the profiles | 3.2e-12 | 1.7e-12 |
So the whole 1e-8 comes from the predicted LAND substep-2 ground columns, and within them from column 9 (`qg_aver`): all other predicted land columns differ by <= 5e-12 abs, but `qg_aver` differs by 1.9e-6 (rel 7.6e-5 of the humidity scale) in exactly TWO of 753 land cells, (57,34) and (21,33) -- the PBLHT/PBLPTOP columns of the D191 table (the other 751 cells: <= 1e-17). Both cells have tsns within 0.05 C of zero and crossed 0 C inside substep 1 ((57,34): tg 273.2099 -> 273.1059 K, (21,33): 273.14996 -> 273.1613 K).
Mechanism: `GHY_DRV.f:1304-1312` computes `qg_sat = qsat(tg1+tf, elhx, ps)`, `qg_nsat = min(qs + evap_max/(0.001*rcdhws), qg_sat)`, `qg_ij = fr_sat*qg_sat + (1-fr_sat)*qg_nsat` AFTER the GHY advance, but `elhx` is still the value set from the ENTRY ground temperature of that substep (lhs if tg1 < 0 else lhe, `GHY_DRV.f:1071-1075`). `land_chain.next_land_pbl_columns` (and its jnp copy in `jax_surface`) used column 19 of the NEXT substep's row, i.e. the elhx of the NEW temperature. They differ exactly when tsns changes sign inside the substep; then qg_sat (and the min cap) differ by about exp(dlh/R * dT/T^2) (1e-4 to 7e-4 relative), which is the 1.5e-7 / 1.9e-6 seen. The PBL column 8 (`qg_sat`, recorded at the next substep entry) correctly uses the new elhx and was already right. The tiny error in 2 cells then goes through the substep-2 land PBL/GHY into the composite fluxes and ATURB, where it is amplified in the neighbouring-level EGCM/PBLHT (these two columns) and spreads through the U/V diffusion.
This is the defect type "recorded-input / ordering difference" asked for in the task: small, local, understood. (Earlier D129/D136b/D187 attributed the residual to "ported GHY / substep-2 input prediction, not diagnosed"; it is this one column.)

## 2. Fix (3 files of the C1 path + 3 variants, 17 insertions, 9 deletions; no tolerance or category touched)
- `land_chain.land_substep` and `jax_surface._land` return the extra key `elhx` (= column 19 of the p4 rows of the substep that ran); `land_chain_ent.land_substep_ent`, `land_ent_par.land_substep_ent_par`, `drv_land_cols.land_substep_v2` the same (one line each).
- `land_chain.next_land_pbl_columns` and `jax_surface.next_land_pbl_columns` compute `qg_sat_cur = qsat(tg, land['elhx'], ps)` and use it for the cap and the blend of `qg_aver` (column 9); column 8 (`qg_sat`) keeps the new-substep elhx. If a land result has no `elhx` (old callers) the old behaviour is used.
- Same code is used by `surface_loop.py:1081`, `surface_loop_v2.py:244` and `jax_tpl_state.py:130` (previous step's land result into the next step's substep-1 row): they now get the same correction without edits.
- Test (new file `tests/test_land_qg_elhx.py`, 2 tests, 34 s): synthetic NumPy-vs-JAX bitwise-agreement/uses-current-elhx/non-vacuity, and real dec01: substep-1 land -> substep-2 `qg_aver` vs the record, max diff < 1e-12 over 753 cells, with the two cells asserted to have changed elhx class (before the fix: 1.9e-6, 1.5e-7).
- Tests of touched files run (cores 0-2,6-7): `test_land_chain.py`, `test_land_chain_ent.py`, `test_jax_surface.py` (the gated `D188_FULL` full-stage test not set, so not run), `test_drv_state_cols.py::test_land_substep_v2_equals_recorded_path`, `test_land_qg_elhx.py`: 26 passed, 1 skipped, 161 s. `test_jax_coupled.py`, `test_land_chain_precip_conditioning.py`, `test_land_chain_irrigation.py`, `test_speed_d175.py`, the rest of `test_drv_state_cols.py` and the full suite NOT run (the assembled step was re-run end to end instead, section 3).

## 3. Before / after (assembled step, step 0, `d191_run.py DATE OUT REF --runs 1`, NEW NumPy references `d187_ref_numpy.py` made with the fix)
C2 gate (14 gate fields A/B/C/D) and 30 end fields; C1 = assembled vs NumPy chain.
| date | before: gate | before: 30 end | after: gate | after: 30 end | C1 after |
|---|---|---|---|---|---|
| nov26 | MET with named columns 1/11/2/0, worst W2GCM 3.1e-12 | 2/25/3/0 | identical (1/11/2/0, W2GCM 3.11e-12, EGCM 1.27e-12, both one column (25,16)) | 2/25/3/0 (LMONINPBL 5.4e-10, 7 cols) | 559 A |
| dec01 | PARTLY MET 1/6/7/0, worst EGCM 9.9e-9 | 2/12/16/0 | **MET 1/13/0/0**, worst W2GCM 4.1e-13, EGCM 3.4e-13, PBLHT 3.1e-13 | 2/27/1/0 (only LMONINPBL 5.7e-11, 17 cols) | 559 A |
| jan01 | PARTLY MET 1/3/10/0, worst EGCM 7.6e-9 | 2/3/25/0 | **MET 1/13/0/0**, worst W2GCM 6.3e-13, PBLHT 4.3e-13, EGCM 4.0e-13 | 2/26/2/0 (USTARPBL 3.0e-12 in 2 cols, LMONINPBL 6.7e-10 in 32 cols) | 559 A |
The verdict wording "MET" is the harness's (all 14 gate fields <= 1e-12 of scale); the C columns left are outside the gate. C1 is category A on all 559 arrays on all three dates (559 instead of 558: the new `elhx` array of the land result is stored on both sides). nov26 does not change (its step has no cell that crossed 0 C in the land substeps; every number of the table equals the D191 value). Surface-state C2 categories (ocean exit, ice, DYNSI, RIVERF, next-step entry) are identical to D191 on all dates.
Raw: scratchpad `d199/final/<date>_asm.json|npz|log`, `d199/ref/`, `d199/tab.py` prints the table; bisection scripts `d199/two.py`, `bis.py`..`bis4.py` (scratch only, not in the repo).

## 4. What was NOT done / limits
- One step per date (step 0) with replayed radiation; one run each; the real radiation server was not used; nothing is claimed about steps >= 1 or the day (chaotic divergence, D129/D183; the day criterion status of D197 is NOT re-measured and may change only through a new day run).
- nov26's named column (25,16) (EGCM 1.3e-12, W2GCM 3.1e-12) is a different, unexplained residual and was NOT investigated.
- Not fixed, noticed: column 19 (elhx) of the substep-2 land row is the RECORDED one in the assembled step; in a run without records it should be set from the new tsns (`drv_land_cols.land_elhx_from_tg`); it only matters through the PBL `qsat` of column 8 for the 2 crossing cells and is not exercised by step-0 C2 (the record equals the correct value there). Same for the substep-1 row of the next step.
- The remaining C-level fields (LMONINPBL, USTARPBL jan01) were not traced; they are outside the 14 gate fields.
- SOCRATES/RADIA not ported or modified; no commit; other sessions' untracked files not touched (`status_slides/build_pdf.py` was already modified before this work and is not mine).
