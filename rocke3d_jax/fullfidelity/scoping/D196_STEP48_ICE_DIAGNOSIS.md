# D196: diagnosis of the 54 tile-mask mismatches at step 48 (day boundary, it 33360)

Owner: session of 2026-10-08 (diagnosis only). Status: diagnosed from the records plus one re-run of steps 0-48; no tracked file edited, no commit, no tolerance changed. Other sessions' untracked files not touched. SOCRATES/RADIA not touched.

## 1. Answer
The 54 cells are **lake cells (FOCEAN = 0, FLAKE > 0), not ocean cells**, and the missing open-water tile is the effect of **`daily_LAKE` (LAKES.f:2492), which is not applied in our day driver (nor in the NumPy chain)**. It is a MISSING STEP, not a state difference at step 47 and not a MELT_SI difference.

Evidence (all from `ff_data/nov26_day`, `ffc_cse_in_<it>.bin` RSI/FLAKE/FEARTH/FOCEAN and `ffp_<it>.bin` template rows; scripts in `d196_results/a2.py`, `a3.py`):
1. The 54 cells (the set of type-1 template rows present at 33360 and absent at 33359; j = 32, 38-43 in 1-based rows) are exactly the lake cells with RSI == 1.0 at 33359 whose RSI is < 1 at 33360: set equality checked (54 == 54, `mask == (pure & rsi48<1)`). All 54 have FOCEAN = 0.
2. In those cells FLAKE is constant through 33358 -> 33359 (checked on all 636 lake cells: FLAKE changes in 0 cells 46->47, in **632 of 636** at 47->48, 0 at 48->49) and grows at 33360 by a relative 6.5e-7 .. 2.5e-3; RSI goes from 1 to 0.99747 .. 0.9999987. The rule in the source (LAKES.f ~2813-2815, non-crunch branch) is `RSI = PLKIC/new_flake = RSI_old*FLAKE_old/new_flake`: the new lake area is open water. Check: (1 - RSI48) vs (1 - FLAKE47/FLAKE48) agree to 2.5e-6 absolute (median relative 3.4e-4, 20 of 54 cells above 1e-3 relative); the residual is not explained (possibly the step's own surface update or the other daily_LAKE terms; not isolated).
3. The tile fraction FLAKE48*(1-RSI48) of the 54 cells is 1.2e-8 .. 8.8e-5, which is the D195 range (so those are the same tiles).
4. The 3 that persist ((54,41), (64,39), (71,39)) have the largest new-water fraction (record RSI 0.99886, 0.99879, 0.99747 at 33360) and keep RSI < 1 in the record through 33364 (0.99898, 0.99887, 0.99750); at 33361 and 33362 exactly 3 of the 54 cells have RSI < 1 in the record; (54,41) is back to 1.0 at 33365. The step 52/53 mismatches (4) were NOT mapped here.
5. Our carried state at step 47 is not different in those cells: from the re-run (section 3), our RSI at the end of step 47 in the 54 cells is exactly 1.0 (min = max = 1.0), equal to the record (RSI == 1 at 33358, 33359). Before the boundary the lake tile mask has 0 mismatches (steps 0-47) and the lake RSI end-of-step vs the record RSI of the next step is within 1.6e-3 everywhere (1 cell above 1e-3 at step 46; see 4.1).
6. So the divergence first appears **at the day boundary, before MELT_SI of step 48**: the record's RSI/FLAKE were changed by `daily_LAKE` between step 47 and step 48; ours were not. MELT_SI is not involved (our MELT_SI on RSI = 1 cannot create open water in a cell it does not melt; not tested separately).
7. `daily_OCEAN` (GLMELT) is excluded as the cause: it changes only ocean MO/G0M (OCNDYN.f:5737-5758) and only for `oGMELT>0` ocean cells; FOCEAN is 0 in all 54 cells.

## 2. Things the boundary also changes (consequences, same cause)
- FLAKE/FEARTH/FLAND change in 632 of 636 lake cells at 33360 (293 decrease, the rest increase); RSI changes in 286 lake cells (the 548 other RSI changes in the cse_in difference are ocean cells, which change at every step, 549 at 46->47, so they are not the boundary).
- Our geometry keeps FLAKE at its step-0 value for the whole run (`surface_loop.make_geo(ctx, flake)` called from the restart/cse_in of the first step; `geo['flake']` is static in `surface_loop.py` and `jax_seaice_lake.py`), so every lake cell whose FLAKE changed is wrong from step 48 on (lake mass per area, MLDLK, ice mass per cell all depend on it). Not quantified here.
- Lake-cell RSI: end of step 48 versus the record's RSI of step 49 is above 1e-3 in 49 cells, all of them cells with a FLAKE change, max 0.1608 (cell (43,36): record RSI 0.90191 -> 0.74116 with FLAKE 0.0024027 -> 0.0029238; ours stays 0.90191). End of step 47 versus record 33360: 50 lake cells above 1e-3; versus the record of 33359 at the end of 46: 1.
- Only the cells with RSI == 1 become a template-mask mismatch; cells with 0 < RSI < 1 keep their open-water tile in both, with different RSI.

## 3. The re-run (what was run)
`d196_results/run_ice.py` = `d193_day.main()` through the D195 configuration (`Coupled(nit_strict=False)`), with `d193_day.allarr_of` replaced by a function that saves only `surf/ice/*`, `surf/lake/*`, `surf/v2/rsix,rsiy` per step. No repo file edited. Command:
`taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python run_ice.py OUT --nsteps 49 --c1dir C1` (steps 0-48, about 45 min on the shared node). Log `d196_results/run_ice.log`. Result: reproduces the D195 number, `mask_mismatch [54, 0]` at step 48 and `[0, 0]` at steps 46, 47. The saved per-step arrays (scratchpad, not in git) are in `d196/c1/nov26_day_step<k>.npz`, k = 0..48. Note: the git head printed by the run header is 8a649f7 (the branch moved after the 1036056 of the task statement); the files used here are unchanged by that.

## 4. Other findings that are NOT the cause but should be known
4.1 A carried-ice difference already exists before the boundary: our end-of-step-k RSI versus the record's RSI of step k+1 differs by up to 4.1e-4 (ocean cells) at step 0 (244 ocean cells above 1e-6) and up to 4.0e-3 at steps 28-47 (cell (36,7), 295-378 ocean cells above 1e-6, 38-42 above 1e-4); lake cells up to 1.6e-3 at step 46. The phase is end-of-step-k ours to cse_in(k+1) of the record (agrees to 1e-6..1e-5 in the sampled cells); it is NOT the cause of the 54 mismatches (those cells are exactly 1.0 in both). Its cause was not investigated; the jump near step 28 (4.7e-4 -> 4.1e-3 at (36,7)) was not traced.
4.2 NumPy chain (`d193_ref_day.py`): NOT run. By code reading it has the same omission: it builds `Loop2(DATE, DAYDIR, L.FF, st, SS, ...)` with the same static `st`/geo FLAKE and applies at `it % 48 == 0` only `OL.apply_daily` (DAILY_ATMDYN), `OL.apply_ch4ox` and the SNOAGE reset; no lake step. Its surface state was bitwise equal to ours for steps 0-22 (D195 C1) and it aborts at step 23 (D194/D195 nit assertion), so it cannot reach step 48 as it stands.
4.3 The hypothesis in D195 sec. 5 ("a daily ocean/ice update at it 33360") is confirmed, with the specific update being `daily_LAKE` rather than an ocean update. The D195 statement "rsi slightly below 1 in 54 cells" is correct; "not the new-ice-tile donor rule" is confirmed (the template already has the tile).
4.4 `drv_daily.py` LAKE_STATUS (D178) already records that daily_LAKE changes FLAKE in 632 cells and RSI in 834 (548 of those are ocean cells and 286 lake cells, so "834 lake RSI changes" there mixes in the ordinary per-step ocean RSI changes of the record; the lake count at the boundary is 286).

## 5. Not done / limits
- `daily_LAKE` not implemented or tested; the proposal is in `D196_PROPOSED_PATCH.txt` (text only).
- The mapping of the step 49-53 mismatches (3, 3, 3, 4, 4) is not redone; for 49-51 the 3 persisting cells are explained as above (RSI stays below 1 in the record). The 4 at 52-53 not mapped (D195 sec. 5).
- The residual (<= 2.5e-6) between the record RSI and RSI_old*FLAKE_old/FLAKE_new in the 54 cells was not explained.
- The NumPy reference was not run; the effect of the boundary on the atmospheric score was not measured.
- Cause of 4.1 not investigated. Single run; no repeat.
