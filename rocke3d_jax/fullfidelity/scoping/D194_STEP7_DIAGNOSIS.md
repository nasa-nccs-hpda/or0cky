# D194 (2026-10-08): diagnosis of the first non-bitwise C1 step (step 7) of the D193 day, and the step-23 reference assertion

Owner: G. Tamkin. Written by a Claude Code agent (subagent). Status: DRAFT; nothing committed. Review by: before a second day is run.
New files only: `d194_nit_fix.py`, `d194_run.py`, this entry, `d194_results_c1.json`, `d194_results_nitfix_log.json`. No existing file edited; other sessions' untracked files not used; SOCRATES/RADIA not touched; no category or tolerance changed.
Conditions: `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, no compile cache, scratch in the session scratchpad, hybrid (libimf via host callback; radiation REPLAYED from the real record, not computed), shared node, one run of each chain.

## 1. Verdict
Category (a), a small local bug/limitation of the assembled step, with a declared-difference flavour: the **GHY substep schedule (dts) of cells whose recorded iteration count ffnit is > 11 is built from the RECORDED precipitation in the assembled step, but from OUR precipitation in the NumPy reference chain.** Step 7 is the first step with such a cell (ffnit 12 at cell (64,19), substep 2). It is NOT the SURFACE rebuild and NOT the new-ice-tile rule. With the fix (new files) C1 is **569/569 A (558/558 at step 0) at every step 0-10**.

## 2. Findings with numbers
1. Step-7 C1 (D193): A 509, B 32, C 28; the `dyn` and `condse` groups are all A, the first differing groups are `surface`, then `dissip`, `filter`, `land`. The only atmosphere column is (64,19) (EGCM 2.0e-9, PBLHT 1.7e-7 on 4000, PBLPTOP, Q 2.3e-13) and one land cell (index 85 of the 1-based land list = Ent/land slot 84 0-based = grid cell (64,19)); fr_sat_ij 8.8e-10, i.e. far above 1 ulp: a real input difference of the GHY of that one cell, not codegen noise.
2. The SURFACE rebuild at step 7 is **not** a tile-set change. Checked on the records: the water/land/Ent/block cell sets (wcells, lcells, ecells, bcells) are identical for steps 0-11 (Nw 2731, Nl 346, Ne 753); `mask_mismatch` was [0,0] at steps 0-10. `Coupled._same_host` also compares `max_substeps`, and `max_substeps` (= widest GHY substep list) changes: 11 (steps 0-6), **12 (step 7)**, 13 (step 8), 12 (step 9), 11 (step 10), 11 (step 11, cache host is the step-10 one, no rebuild). The D193 entry statement "rebuilt when the tile set changes" is therefore wrong for steps 7-10 (I checked steps 0-11 only; the 13 rebuilds at 17-28 are most likely the same cause, NOT checked).
3. Cell (64,19) is the first ever cell with 12 GHY iterations (substep 2: ffnit 12; substep 1: 10; no other cell >= 11 at step 7).
4. Mechanism (read from code, then confirmed by the experiment of item 6):
   - `ghy_advnc_test.build_batch` = `ghy_advnc_test_nit.build_batch_nit` (D158): the ffg record holds at most 11 Ent/dts iterations; for a cell with ffnit > 11 ALL dts are regenerated with the real loop (`ghy_ref_nit.run_cell_full`, gdtm) from the ROW (it asserts nit == ffnit).
   - Reference (`surface_loop.Loop.stage_surface`, copied in `surface_loop_v2.Loop2`): overwrites g[:,143:146] (PREC, EPREC, PRECSS) with OUR CONDSE values BEFORE `build_batch`; the reconstructed dts follow our precipitation.
   - Assembled step (`jax_coupled.Coupled.step`): `jax_surface.build_template` builds the batch once, before CONDSE, from the recorded row; `jax_tpl_state.apply` later overwrites only the pr/htpr/prs forcing on the device. dts and nsub of ffnit>11 cells stay on the recorded precipitation.
   - Sensitivity: our PREC differs from the recorded one by 6.5e-12 (|d pr| in the units of g[:,143]) at that cell, and the regenerated edts of the cell change by up to 2.8e-4 s (dts are a threshold-driven function). Other cells use recorded dts in both chains, so steps 0-6 (no ffnit>11 cell) were bitwise.
5. Declared? D191 declares "Ent exports and land forcing RECORDED"; it does not declare that the dts of ffnit>11 cells are reconstructed from the recorded precipitation. So it is not a legitimate declared difference; it is an undeclared one, hence (a).
6. Fix and result (`d194_nit_fix.py`, driver `d194_run.py`): runtime wrappers (no file edited) of `jax_surface.build_template` (keeps the recorded g1/g2 rows and the ffnit>11 list) and `jax_tpl_state.make_apply_state` (when such cells exist: ONE declared device->host read of PREC/EPREC/PRECSS, batch rebuilt from the rows with our precipitation exactly as the reference does, edts/ecnc/elai/ebet/nsub/dt of exactly those cells replaced on the device; same shapes, no extra compile). Log: step 7 cell (64,19) nsub 12, edts change 2.75e-4; step 8 cell (63,17) substep 2, nsub 13, edts change 1.9e-8; step 9 cell (63,17) substep 1, nsub 12, edts change 9.6e-8; step 10 no such cell.
   Result, port with fix vs NumPy libimf chain, harness categories (`d194_results_c1.json`):

   | steps | A | B | C | D |
   |---|---|---|---|---|
   | 0 | 558/558 | 0 | 0 | 0 |
   | 1-10 | 569/569 each | 0 | 0 | 0 |

   (The one-sided key `land/p4_ij` appears in every step, also in D193 steps 0-6; pre-existing, not a difference.) Port timings unchanged in kind (steady 15.5-18.2 s; rebuild steps 71-79 s; cold step 414 s).
7. The unfixed port (D193) is therefore bitwise only because no cell had ffnit>11 before step 7. Which of the two chains is closer to the real model: the reference's order (dts from the live precipitation) is the physically consistent one; the recorded-precip dts are only right while our precipitation equals the record. Not tested against the real end state.

## 3. The step-23 NumPy reference assertion
- Assertion: `ghy_advnc_test_nit.build_batch_nit`, `info['nit'] == ffnit[i]`, row 111 = cell (41,21), recomputed 11 vs recorded 14 (step 23, it 33335, substep 2).
- Check run (scratch `nitscan.py`, all 54 steps of nov26_day, both substeps, every ffnit>11 cell, `run_cell_full` on the RECORDED row): 15 cell-substeps, **0 mismatches** (e.g. step 23 g2 row 111 (41,21): recorded 14, recomputed 14; step 24 g1 14, g2 15 recomputed 14/15). So the reconstruction reproduces the real model's iteration count from the real row; the records are consistent with the code; this is **not** an input mismatch of the records.
- The reference aborts because it feeds the function a row with OUR precipitation (and the cell's start state is the recorded one, not our carried state, see below); by step 23 the reference chain has left the record chaotically (D193: category D at steps 8-22), so the discrete count flips from 14 to 11. It is a reference-chain limitation: the assertion is a record-consistency check for the reconstruction, applied to a free-running state where the count has no reason to equal the record.
- Same behaviour would occur in the fixed port at step 23 (the wrapper keeps the assertion, not caught). NOT run to step 23 (reasons: ~20 min cold+rebuilds per chain; the answer is about the code path). Without the fix the port would not abort but would silently use the recorded 14 iterations.
- Unresolved, owner decision: in a free-running day nit must be allowed to differ from the record (assertion removed or turned into a report), `max_substeps` is a compile-time constant of the SURFACE stage (rebuild per change, ~55 s each), and the cell's GHY start state in `build_batch_nit` is the recorded one (`dyn0`) although the carried state is used for the integration; whether the real GHY_DRV computes gdtm from the live state (it must) is not reproduced by either chain. This was not worked around.

## 4. Proposed patch for tracked files (not applied; to the owner)
- `jax_coupled.py`/`jax_surface.py`: move the dts/nsub rebuild of ffnit>11 cells after phase 1 (precip known) as in `d194_nit_fix._fix`, or compute gdtm on device; keep the assertion, but add a report mode for free-running days.
- `d193_day.py` README text: replace "tile set changes" by "tile set or max_substeps changes" as cause of SURFACE rebuilds; `_same_host` could pad dts to a fixed width (e.g. 16) to avoid the 4+ recompiles of steps 7-10 (not tested; changes the scan length of the compiled GHY; the standalone test showed padding by 1-2 columns gives bitwise identical `advnc` output for step 7 batch).

## 5. NOT run
- No 54-step run with the fix; no step beyond 10; no unfixed re-run (the D193 result is the unfixed reference); no score (multiday_score) with the fix; no test files written; no GPU, no transfer-guard run; the D2H read added by the fix (3 small arrays, only on steps with ffnit>11 cells) is not counted in any transfer statistic.
- Not diagnosed: tile-mask mismatches from step 43 (ocean/lake-water type, 1-55 cells; the new-ice-tile donor rule is still not implemented), the cause of rebuilds at steps 17-28 (hypothesis only, section 2.2), what C1 would show at 11-22 with the fix.
- The per-step arrays of the D193 day had been deleted (24 GB), so step 7 was re-run (steps 0-10) rather than inspected; the first-differing-variable analysis rests on the stored D193 C1 json plus the record/code analysis and the A-result with the fix.

## 6. Reproduce
`taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d194_run.py OUT --nsteps 11 --c1dir C1 --nitfix 1` (about 25 min; `--nitfix 0` = D193 behaviour), then `... python d193_ref_day.py REFOUT C1 11` (about 6 min, run lagging the first). Sources/owner: G. Tamkin; evidence in `d194_results_*.json`.
