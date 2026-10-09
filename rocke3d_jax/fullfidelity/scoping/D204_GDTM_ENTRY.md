# D204 (2026-10-09): the GHY `gdtm` sub-iteration schedule is computed from the state in the production land path (JAX and NumPy-chain land copy); default = computed

Owner: G. Tamkin (decision 2026-10-09). Written by a Claude Code agent (D204). Status: DRAFT; nothing committed or pushed. Review by: before the next day run.
Conditions: `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, scratch TMPDIR/D189_SHIM_DIR, CPU, shared node, radiation REPLAYED from the records (the scorer sentence "radiation computed by the original Fortran" does NOT apply), daily updates not applied, `d193_day.main` unchanged, one run of each. HEAD 11a8b4c + the uncommitted edits below. `jax_coupled.py` shows uncommitted changes by another agent (D205: jit refactor + `daily_lake_update`, not called by my driver); not mine, not touched.

## 1. What changed (tracked files)
- `ghy_jax.py` (+194): `advnc_gdtm` (the D202 prototype `d202_fix.advnc_gdtm` moved in; same physics), `GDTM_WIDTH=40`, switch `schedule()/set_schedule()/advnc_sched(..., mode=None)`; env `ROCKE_GHY_SCHEDULE=computed|recorded`, **default computed**. One difference to the prototype: the 40-lane `lax.scan` is a `lax.while_loop` that stops when no cell has `dtr > 0` (all lanes masked as before). Demonstrated bitwise equal to the scan prototype on every validated step (`while_vs_scan_bitwise` true, 12/12). Extra outputs `nit_gdtm`, `nit_exhausted`. Ent exports of iteration i = recorded iteration min(i, W0-1) (padding with the last record; for iterations > 11 this is an ASSUMPTION, as in `ghy_ref_nit`).
- `jax_surface.py` (6 lines): `_land` calls `J.advnc_sched`; `nit_rebuild` is a no-op in computed mode (no precipitation read, no device->host transfer, no rebuild batch); `_ghy_batch` uses `AT.build_batch_recorded` in computed mode (no `build_batch_nit`, so no `nit == ffnit` assertion and no NumPy loop at template build).
- `land_chain.py` (6 lines): `run_ghy` (used by the NumPy chain `land_substep`) uses `advnc_gdtm` jitted + `build_batch_recorded` in computed mode. NOTE: the NumPy chain's land stage shares the JAX GHY code, so C1 on the land stage is not an independent check; the independent check is section 2 (NumPy `ghy_ref_nit` loop).
- `tests/test_jax_surface.py`: the D195 nit-rule test now runs with the recorded schedule (fixture); new test that `nit_rebuild` is a no-op in computed mode. New `tests/test_ghy_gdtm.py` (5 tests). New: `d204_validate.py`, `d204_day.py`, this entry, `scoping/d204_results/`.
- The flag is read at TRACE time (inside jit). Switch it before building the stage / in a fresh process; do not flip it within a process after a stage was traced.

## 2. Validation on real records (`d204_validate.py`, `d204_results/validate_nit.json`)
Steps 0, 5, 14, 17, 30, 40, both substeps (12 batches x 753 land cells), real nov26_day ffg records:
- computed nit == recorded ffnit in 753/753 cells in all 12 batches; none exhausted. Cells with ffnit >= 4 per batch 64-185 (includes the 4-substep cases, and >11 cells where present).
- outputs vs the recorded-schedule `advnc` (22 output arrays): categories A/B only, no C, no D in any batch; max scaled difference 5.7e-14 (`|d|/max(1,|ref|)`). tbcs, alhg, aevap, arunu etc. are B in most batches (dts recomputed, not bitwise), nsn/dzsn/wsn/hsn/fr_snow/fice/abetad mostly A. Per-array categories and per-cell A/B/C/D counts are in the json.
- vs the real Fortran values in the record (tbcs tsns ashg alhg aevap): computed and recorded schedule give the same max abs error (0 to the printed precision except step 17 sub 2: tsns 2e-6, ashg 0.017, alhg 0.041, identical for both schedules).
- NumPy `ghy_ref_nit.advnc_full(use_recorded_dts=False)` on 24 cells per batch (>=3-iteration cells + random): nit equal 24/24 in all 12 batches (also == ffnit); max |d ashg| 9.6e-9 (relative to ashg ~1e2-1e3), |d aevap| 1.5e-15, |d tsns| 2.8e-13.
- Stiff test (tests/test_ghy_gdtm.py): ch x 20 makes the computed schedule take more iterations than the recorded one in > 50 cells and stay finite (the recorded schedule cannot do this by construction).

## 3. Step-0 C1/C2, 3 dates (`d191_run.py DATE ... --runs 1` with new NumPy references `d187_ref_numpy.py`; `d204_results/<date>.log`, `<date>_asm.json`, `step0_d204_vs_d203.txt`)
| date | gate | end-30 A/B/C/D | C1 |
|---|---|---|---|
| nov26 | MET with named exception columns {B 11, C 2, A 1} | 2/25/3/0 | 561/561 A |
| dec01 | MET {B 13, A 1} | 2/27/1/0 | 561/561 A |
| jan01 | MET {B 13, A 1} | 2/26/2/0 | 561/561 A |
Exactly what changed vs D203: verdict lines, gate categories and end-30 category counts are IDENTICAL on all three dates. C1 count is 561 (D203: 559) because the land GHY dict now has 2 more arrays (`nit_gdtm`, `nit_exhausted`); all 561 are A, so "559/559" became "561/561"; the 559 old arrays are all A. Port step-0 arrays are NOT bitwise equal to D203 any more: of 549 float arrays compared, 392/376/386 (nov26/dec01/jan01) are bitwise equal and the rest differ at most 1.4e-11 / 4.9e-10 / 1.4e-10 (scaled; worst: ocean szmo/gzmo, land ghy ashg) - the rounding-level effect of recomputing dts (category B), not a change in any verdict. I did not check the worst-field values of the gate against D203 beyond the category counts.

## 4. 54-step nov26 day (`d204_day.py --nit-strict 0`, `d204_results/score.md|json`, `day.log`, `d204_nit.json`)
All 54 steps finite in T U V Q P QCL QCI (checked on the saved per-step files), flags 0, no `nit` mismatch at all (`d204_nit.json`: empty; nit_rebuild is a no-op), a single SURFACE rebuild (step 0; D203 had 14, because the compiled GHY width no longer follows the record), steady steps about 17-22 s on a shared node (not a benchmark). Tile-mask mismatches only at steps 48-53 as in D203 (D196 signature).
| field | D204 within/near/beyond | D203 | worst ratio steps >= 3 (step) D204 | 
|---|---|---|---|
| T | 54/0/0 | 54/0/0 | 0.94 (15) |
| U | 54/0/0 | 53/1/0 | 0.89 (21) |
| V | 54/0/0 | 54/0/0 | 0.97 (10) |
| Q | 42/12/0 | 40/14/0 | 1.03 (46) |
| P | 52/2/0 | 54/0/0 | 1.04 (33) |
| QCL | 52/1/1 | 50/3/1 | 0.95 (9); all steps 2.83 (step 1) |
| QCI | 37/17/0 | 49/5/0 | 1.69 (47) |
Section 9 stays NOT MET (QCL step 1 = 2.83 beyond, "within at every step" False; steps >= 3 never beyond; worst 1.69 QCI step 47). Mixed versus D203: T/U/V/Q/QCL better or equal, P and QCI worse (QCI near 5 -> 17, worst 1.12 -> 1.69). This is ONE run of a chaotic system with a different trajectory from step 1; I do not claim the schedule change is the cause of the QCI change, nor that it is an improvement or a degradation; not analysed.

## 5. C1 against the NumPy chain (`d193_ref_day.py` unchanged, `d204_c1.json`, `ref.log`)
Bitwise (A) at ALL 54 steps (step 0: 561/561, steps 1-53: 572/572 each; B, C, D = 0). The reference no longer aborts (D203: abort at step 20; D195: step 23) because in computed mode `run_ghy` does not use `build_batch_nit`. Caveat: the reference's land stage runs the same `advnc_gdtm` code (section 1), so this shows the assembled JAX step equals the NumPy-driven chain with the same land code, not an independent GHY check.

## 6. Tests (quick, touched units; full suite NOT run)
`tests/test_ghy*.py` + `test_jax_surface.py` + `test_land_qg_elhx.py` (one process): 261 passed, 1 skipped, 1 failed = the D195 nit-rule test, which assumed the recorded schedule; fixed with a fixture (recorded mode) and re-run: test_jax_surface.py 7 passed, 1 skipped; `tests/test_ghy_gdtm.py` 5 passed. `tests/test_land_chain.py` (own process): 13 passed. NOT re-run after the test edit: the rest of the first batch (it passed before the edit, which only touched test_jax_surface.py). test_jax_coupled, test_surface_loop*, GPU: not run.

## 7. Demonstrated / hypotheses / not done
Demonstrated: sections 2-6 as measured. Hypotheses: that the Ent-export padding for iterations > 11 matches the Fortran (not testable from the records); that the QCI/P changes are trajectory noise (not analysed). Not done: GPU run (while_loop on GPU untested), cost measurement of the GHY stage alone, bisect of the day change, full suite, any benchmark. Default flip is reversible: `ROCKE_GHY_SCHEDULE=recorded` restores the old path (old path was only re-verified by the unchanged old tests: bitwise `advnc_sched(mode='recorded') == advnc`).
Reproduce: `python d204_validate.py out.json 0 5 14 17 30 40 --np 24`; day: `python d204_day.py OUT --c1dir C1 --nit-strict 0`, `python d193_ref_day.py REF C1 54`, `python multiday_score.py --ours OUT/ours_d193 --nsteps 54 --cache members.json --md score.md --json score.json`; step 0: `d187_ref_numpy.py DATE REFN` then `d191_run.py DATE FINAL REFN --runs 1`.

## Parent-session check (2026-10-09)
Re-read and re-scored the 54 saved states: all finite; the table equals the agent's (T 54/0/0, U 54/0/0, V 54/0/0, Q 42/12/0, P 52/2/0, QCL 52/1/1, QCI 37/17/0; worst ratio steps >= 3 is QCI 1.69 at step 47; all steps 2.83 QCL step 1). Re-ran tests after the agent's fixture fix: test_ghy_gdtm 5 passed; test_jax_surface 7 passed, 1 skipped. Decision (parent, standing delegation 2026-10-09): the computed schedule stays the DEFAULT because it reproduces the recorded nit in 753/753 cells in 12 batches, gives A/B outputs vs the recorded schedule (max 5.7e-14), removes the NumPy-reference abort (C1 bitwise all 54 steps) and follows the Fortran statement order; `ROCKE_GHY_SCHEDULE=recorded` restores the old behaviour. The day score is MIXED vs D203 (P 54->52 within, QCI 49->37 within and worst 1.12 -> 1.69 at step 47, U/Q/QCL better); not analysed, not claimed as improvement or degradation. C1 over 54 steps does not independently check GHY (both chains run the same GHY code). Section 9 remains NOT MET (QCL step 1 = 2.83). Committed by the parent: the D204 files only; `jax_coupled.py` (D205) is not part of this commit.
