# D200 (2026-10-08): the nov26 54-step day re-run on the D199-fixed land code: the port goes NaN at step 39 (NOT a valid day); C1 bitwise only for steps 0-18

Owner: G. Tamkin. Written by a Claude Code agent (subagent). Status: DRAFT; nothing committed or pushed. Review by: before the next day run.
New files only: `d200_day.py` (driver, identical to `d195_day.py` apart from names; the untracked file of the stopped previous attempt was reviewed, found to be that copy, and kept), this entry, `d200_results/`. No tracked file edited, no other session's untracked file used, SOCRATES/RADIA untouched, no tolerance or threshold changed, no process left.
Code under test: HEAD 93976f0 (D199 land `qg_aver` elhx fix), 0 modified tracked files. Conditions as D195: `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, `clouds_jax_env`, no compile cache, TMPDIR/D189_SHIM_DIR in the scratchpad, libimf host callback, jax 0.5.3, CPU, shared node (load 0.4-3), nit fix on, `--nit-strict 0`, radiation REPLAYED from the real records (the scorer's fixed sentence "radiation computed by the original Fortran" does NOT apply), same recorded inputs and day-boundary handling (`d193_day.main` unchanged). The previous attempt's scratch (`scratchpad/d200`) was deleted and everything re-run from scratch; one run of each.

## 1. Result in one paragraph
The day does NOT survive the fix. 54 of 54 steps were executed, but from step 39 (it 33351) the state is NaN (T all 132480 values, then U/V/Q/QCL/QCI almost everywhere). The pass rule of ACCEPTANCE section 9 (never beyond 2x over ALL steps) is **NOT MET**, for two reasons: QCL step 1 is still 2.84 (beyond), and steps 39-53 are not finite (invalid, cannot be classified). The scorer silently labels those 15 steps "n/a" and its line "never beyond 2x (steps >= 3 only): True" must NOT be read as a pass. The strict "within at every step" line: not met either. The D199 fix changed the day: (a) the D196 daily_LAKE gap is not what matters any more, (b) a land-cell runaway appears. See sections 3-4.

## 2. Run
54 steps executed, flags 0 every step (the stop-model flags did not catch the NaN), total wall 2658 s (44 min; D195 2128 s). Cold step 0 489 s (126 compiles); steady steps 1-39 median 15.6 s; **from step 40 every step takes 52-56 s** (median 52.9 s, vs ~16 s before: the NaN state makes the host callbacks/iterations slower; not analysed). SURFACE rebuilds 14 including step 0: steps 0, 7, 8, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 28 (same list as D195, all `max_substeps`). nit mismatches reported (non-strict): 6, cells (40,21) row 110 (12 vs 13; new, not in D195) and (41,21) row 111 (13 vs 14, 13 vs 14, 11 vs 15, 12 vs 13, 11 vs 12) (D195: 4, all row 111). Tile-mask mismatch counts are 0 until step 40, then 708-762 slots at every step (NaN-state artefact, not comparable with the 54-cell D196 case). Day-boundary daily update at step 48: `deltam` and `smass` are NaN (state already NaN). Raw: `d200_results/d200_run.json`, `d200_nit.json`, `d200_day.log`.

## 3. Score versus the 5 real members (`multiday_score.py` unchanged, `d200_results/score.md|json`); only steps 0-38 are finite
| field | within / near / beyond (finite steps 0-38; 15 steps n/a) | D195 (54 steps) | worst ratio steps 3-38 (step) | D195 | near steps |
|---|---|---|---|---|---|
| T | 38/1/0 | 54/0/0 | 1.08 (38) | 0.94 | 38 |
| U | 35/4/0 | 53/1/0 | 1.04 (5) | 1.02 | 3,4,5,9 |
| V | 36/3/0 | 54/0/0 | 1.09 (5) | 0.90 | 5,9,10 |
| Q | 29/10/0 | 41/13/0 | 1.05 (38) | 1.05 | 25,28-33,35,37,38 |
| P | 35/4/0 | 54/0/0 | 1.28 (38) | 0.98 | 9,30,33,38 |
| QCL | 36/2/1 | 47/6/1 | 1.09 (20) | 1.13 | 2,20; beyond 1 |
| QCI | 36/3/0 | 54/0/0 | 1.15 (8) | 0.98 | 7,8,9 |
QCL step 1: ratio 2.84 (ours 1.15e-6 vs largest member 4.06e-7), identical to D195 and D193 (D197: one bistable cell); the fix does not touch it. Even the finite steps are worse than D195 for P, V, T, QCI (step 38 is already damaged: the runaway cell has tsns 264 C there). The comparison with D195 for steps >= 39 is impossible.

## 4. What happens (measured; the cause is NOT established)
- First NaN: step 39 (end state of it 33351), in the surface/ATURB stage at one grid column (41,20), land-cell index 112 (GHY states `ht`, `w`, `tp`, `tsns`, `aevap` NaN), then spreading to its lake neighbour (41,21), PBLHT/U1AA, the dissipation/filter and, through the global filter/pressure terms, the whole field.
- Cell 112 departs from a smooth path before that: `tsns` 17-20 C until step 18, then irregular 22.4 (19), 23-24, 20.5 (25), ..., 24.4 (32), 30.4 (33), 17.6 (34), 26.3 (35), 31.4 (36), 49.4 (37), 263.9 (38), NaN (39); aevap 0.09 -> 0.92 at step 38. The real level-0 T of that column stays at 42.1-42.3 (real) and D195 stays within 0.04 of the real; D200 differs by 0.1 (step 30), 0.4 (33), 0.8 (34), 7.1 (step 38, cell blowing up). D200 vs D195 column difference grows from 1e-8 (step 3) to 1e-3 (step 12) to 0.1 (step 23): ordinary chaotic growth up to about step 30, then a local runaway.
- The cell is the neighbour of the nit-mismatch cells (40,21) and (41,21), whose rebuilt `edts` change by up to 68 s in the non-strict regime (the regime D195 states is not reproduced by any other chain). This is the only hint; I did NOT show that the runaway is caused by the non-strict nit regime, by the D199 fix itself, or by a land-chain defect (no bisection: it needs code changes or a variant run, outside the "new files only" scope of this task). The D199 fix acts only when tsns crosses 0 C inside a substep; cell 112 is at 17-50 C, so the fix reaches it only indirectly through the chaotic divergence of the trajectory.
- So: the D195 day (old land code) was a lucky/non-representative trajectory with respect to this cell, or the fixed code exposes an instability of the non-strict regime. Unresolved. Not a statement about the real model.

## 5. C1 versus the NumPy chain (`d193_ref_day.py` unchanged; regenerated with the fixed code; `d200_results/d200_c1.json`, `d200_ref.log`)
| steps | result |
|---|---|
| 0 | 559/559 A (bitwise) |
| 1-18 | **570/570 A at every step** (B 0, C 0, D 0) |
| 19 | NOT AVAILABLE: reference aborted at step 19 (it 33331) in `build_batch_nit`, `AssertionError(110, 12, 13)`, not caught (by design) |
D195 ran to step 22 on the old code; with the fixed code the NumPy chain meets the nit assertion 4 steps earlier (cell (40,21) row 110, recomputed 12 vs recorded 13). The port was bitwise to the reference for all 19 steps the reference can run (0-18); nothing is claimed for steps 19-53. This is port versus reference with the same recorded inputs; it says nothing about the real model, and it shows that the NaN of section 4 is not a difference between port and NumPy chain over steps 0-18 (it arises in the regime neither chain reproduces).

## 6. Answers asked
- Did D199 change the day? Yes, qualitatively: finite for steps 0-38 only, NaN afterwards (D195: finite, worst ratio 1.13 from step 3). Mechanism not isolated (section 4).
- Is the D196 missing daily_LAKE at step 48 still missing? Yes (no day-boundary ocean/ice/lake/land daily update is applied; nothing was added), but it is not observable in this run: the state is NaN at step 48 (tile-mask mismatch 762 slots at step 48, which include the NaN effect).
- Day criterion (ACCEPTANCE section 9): NOT MET (QCL step 1 = 2.84 beyond; steps 39-53 invalid). Secondary "within at every step": NOT MET. No relaxation, no exception applied.

## 7. Not run / limits
- No bisection of the runaway (D199 fix on/off in a variant, strict vs non-strict, nit-rebuild off, land cell 112 isolated) and no check whether cell 112 / (41,20) is special in the records. Suggested next step: rerun steps 28-39 from a saved state with the D195 code path (`git stash`-free variant via monkeypatch in a d201 file) to see whether the runaway follows from the fix or from the non-strict nit regime; and look at GHY cell 112 substep by substep.
- C1 only steps 0-18; steps 19-53 only scored (steps 39-53 not even finite). Radiation replayed, day boundary on host, daily ocean/ice/land updates not applied, nit non-strict, one run each; the secondary zonal/global-mean statistics of `score.json` are not interpreted (they include the NaN steps).
- Scratch: per-step C1 arrays (24 GB) deleted; kept in the scratchpad (not in git): per-step end states `scratchpad/d200/out/ours_d193/step_<it>.npz` (54 files, 407 MB; steps 39-53 are NaN).

## 8. Reproduce
`taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d200_day.py OUT --c1dir C1 --nit-strict 0` (44 min), `... python d193_ref_day.py REF C1 54` (aborts at step 19, about 8 min), `python multiday_score.py --ours OUT/ours_d193 --nsteps 54 --cache members.json --md score.md --json score.json`.

## Parent-session check (2026-10-08)
Re-scored the 54 saved states with `multiday_score.py` myself: table identical to `score.md`. Re-read the states: the first non-finite state is step 39 (it 33351; T, U, V, Q, P and two more fields). So the day is NOT valid after step 38; section 9 NOT MET (QCL step 1 = 2.84; steps 39-53 non-finite). The scorer's verdict line still says "floor ~1e-9" and "never beyond at steps >= 3: True"; both are misleading (see the correction of 8a649f7; the steps >= 3 line is not a pass). C1 is claimed for steps 0-18 only (the D195 claim of 0-22 was before the D199 land fix; reference aborts at step 19). Cause of the runaway in land cell index 112 (grid 41,20) is NOT established; no variant run. Not a pass; no tolerance changed.
