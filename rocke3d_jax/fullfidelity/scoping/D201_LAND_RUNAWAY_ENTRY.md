# D201 (2026-10-08): the nov26 day NaN at step 39 is switched by the D199 `elhx` change (demonstrated); the cell-level cause of the runaway is a hypothesis

Owner: G. Tamkin. Written by a Claude Code agent (subagent D201). Status: DRAFT; nothing committed or pushed. Review by: before the next day run.
New files only: `d201_day.py` (driver with the switch and the capture), `d201_real_trace.py`, `d201_compare.py`, `d201_cell112.py`, this entry, `d201_results/`. No tracked file edited, SOCRATES/RADIA untouched, no tolerance changed, full test suite not run, nothing committed.
Conditions as D200: HEAD 1250c45, `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, `clouds_jax_env`, no compile cache, scratch TMPDIR/D189_SHIM_DIR, `--nit-strict 0`, radiation replayed from the records, `d193_day.main` unchanged, steps 0-40 only (`--nsteps 41`), one run of each variant (OFF 1767 s, ON 2028 s). No restart is supported by the driver, so both are full runs from step 0.

## 1. The switch
`d201_day.py --fix 0|1`. `--fix 1` = HEAD. `--fix 0` monkeypatches `jax_surface._land` so that the land result has no `elhx` key; `jax_surface.next_land_pbl_columns` then takes its existing fallback (`qg_sat_cur = qg_sat`, i.e. qsat with the NEXT substep row's elhx `rec[:,19]`), which is the pre-93976f0 behaviour of the jitted path (the diff of 93976f0 in `jax_surface.py` is exactly that). The NumPy `land_chain` copies were not patched (not used by the jitted day path; the day uses `jax_surface`). Per step the driver also writes the full land result (`aux_land`: ghy, pbl, patch, rho, elhx; 753 land cells; second surface substep of the step) and the T(41,20,:) column.

## 2. Confirmation (task 2): bitwise
- OFF (`--fix 0`) vs the saved D195 states (old land code): 41 of 41 steps bitwise identical in T U V Q P QCL QCI. ON (`--fix 1`) vs the saved D200 states: 41 of 41 bitwise identical (NaN positions equal). `d201_results/repro_vs_d195_d200.txt`.
- So the switch reproduces D195 (finite) and D200 (NaN) exactly. With the fix OFF the day is finite through step 40 (checked only to 40; D195 itself was finite for 54). With the fix ON, cell 112 is finite to step 38, NaN at step 39, whole land NaN at step 40 (ON step 40 takes 56 s as in D200). Nothing else differs between the two runs: the single `elhx` switch decides finite versus NaN. This is demonstrated, not a hypothesis.

## 3. First difference between ON and OFF (task 4), measured
| step | what differs |
|---|---|
| 0 | nothing (every land field of all 753 cells bitwise equal) |
| 1 | exactly ONE land cell (index 354, grid (59,31) 0-based = record (60,32)). Fields: dyn_next/ht (max 7.1), ae0 (1.3), ashg (3.8 W/m2), alhg (5.1), aevap (2.0e-6), fr_sat, tp, w, and through the land patch the PBL fields of that cell. tsns itself is equal. The cell is at the 0 C crossing: real record tsns 0.587 C after substep 1, 2.055 C after substep 2 (nit 1 both), ts_in 273.7 K. Its `elhx` of the substep that ran is 2.5e6 (vaporisation), the next row's is the sublimation value, exactly the D199 case |
| 2 | 681 of 753 cells differ in tsns (max 2.9e-3 K), 753 in ashg/PBL; the difference has spread through the atmosphere from step 1 (all of it downstream of cell 354) |
At step 1 the ON value of cell 354 equals the real record: aevap 5.6080e-03 (real 5.6080e-03; OFF 5.6060e-03), ashg 7562.755 (real 7562.755; OFF 7566.531), alhg 14020.066 (real 14020.066; OFF 14014.961). So at the point where ON and OFF first differ, ON is the one that agrees with the real model; the fix is locally right (consistent with D199).
Land cell 112 (41,20), which never goes near 0 C in the real record (tsns 17-25 C), first differs in step 2 (49 of its fields, largest dyn_next/ht 1.6e-7, tsns equal to 4 decimals); the difference is a forced consequence of the atmosphere changing. Cell 112 is therefore not directly hit by the fix.

## 4. Trace of land cell 112 (task 3); `d201_results/cell112.md`, `on_off_by_step.json`, `real_rows.json`
The captured quantities are the end of the second surface substep of each step (the GHY internal substeps are inside one `lax.scan` and are not stored: individual GHY sub-iterations of the cell were NOT traced). Real = ffg record rows (42,21) 1-based = row 112, second substep, from the same step; real `ffnit` (GHY iteration count, taken from the record and imposed on the port via edts/nsub).
| step | tsns OFF / ON / real (C) | aevap OFF / ON / real (kg/m2/s) | ch OFF / ON | ws OFF / ON (m/s) | real nit (sub1,sub2) |
|---|---|---|---|---|---|
| 12 | 19.261 / 19.266 / 19.261 | 4.36e-2 / 4.41e-2 / 4.36e-2 | | | 3,4 |
| 17 | 18.764 / 18.897 / 18.765 | 0.1354 / 0.1203 / 0.1350 | | | 14,14 |
| 18 | 19.340 / 19.206 / 19.387 | 0.0764 / 0.1309 / 0.0750 | | | 8,13 |
| 25 | 22.748 / 20.534 / 21.501 | 0.109 / 0.152 / 0.180 | | | 6,6 |
| 30 | 20.599 / 19.982 / 20.883 | 0.0368 / 0.0218 / 0.0074 | 0.0419 / 0.0455 | 0.69 / 0.69 | 1,1 |
| 31 | 20.219 / 20.073 / 20.798 | 1.3e-3 / -0.2616 / 4.2e-3 | 0.0022 / 0.0525 | 0.72 / 1.94 | 1,1 |
| 32 | 19.816 / 24.384 / 20.154 | 5.9e-4 / 0.164 / 1.3e-3 | 0.0023 / 0.0625 | 0.73 / 1.05 | 1,1 |
| 33 | 19.090 / 30.409 / 19.405 | -2.0e-4 / 0.031 / -1.8e-4 | 0.0015 / 0.0012 | 0.71 / 1.53 | 1,1 |
| 34 | 18.449 / 17.619 / 18.741 | -7.8e-4 / -0.145 / -6.3e-4 | 0.0015 / 0.0625 | 0.87 / 4.00 | 1,1 |
| 35 | 18.136 / 26.329 / 18.395 | -9.9e-4 / 0.0079 / -8.0e-4 | 0.0016 / 0.0625 | 0.97 / 2.72 | 1,1 |
| 37 | 17.918 / 49.370 / 18.141 | -1.04e-3 / 0.241 / -8.7e-4 | 0.0018 / 0.0401 | 0.99 / 2.11 | 1,1 |
| 38 | 17.833 / 263.9 / 18.048 | -1.12e-3 / 0.918 / -9.3e-4 | 0.0020 / 0.0625 | 1.01 / 4.97 | 1,1 |
| 39 | 17.788 / NaN / 17.974 | -1.10e-3 / NaN / -9.6e-4 | | | 1,1 |
Measured facts:
- OFF tracks the real record of this cell to 1e-3 C through step 17 and to 0.3-0.6 C afterwards (step 40: 17.83 vs 18.02 real); ON is within 0.006 C to step 12 and then drifts (0.13 C at step 17, 2.2 C at step 25 vs OFF).
- Largest ON minus OFF difference over all 753 cells, tsns: 2.9e-3 (step 2), 0.26 (8), 1.6 (13), 5.8 (24, cell 127), 11.3 (33, cell 112), 246 (38, cell 112). Cell 112 is the worst cell from step 32 on; before that other cells (142, 143, 184, 175, ...) are worse, but none runs away.
- The runaway starts at step 31 with the surface layer, not with GHY alone: in both runs ch (heat exchange coefficient) is 0.042-0.046 at step 30 (convective regime, daytime); at step 31 OFF and the real model go to the stable regime (ch 0.002), ON stays at 0.0525 with ws 1.94 (OFF 0.72) and aevap -0.26 (strong condensation, real +4e-3); from step 32 ch sits at 0.0625 (it is 0.0625 in ON at steps 32, 34, 35, 38; this looks like a cap, I did not check the PBL source) and ws is 1.0-5.0 while OFF stays 0.7-1.0.
- The real GHY iteration count of this cell is 1 for steps 30-39 (up to 14 in steps 11-18); the port takes edts/nsub from the record (`nit_rebuild` only touches cells with ffnit > 11 and only rebuilds from our precipitation; cell 112 is never rebuilt). Nit mismatch reports: ON 6 (cells (40,21) 12 vs 13, (41,21) x5), OFF 4 (all (41,21)); the neighbours of cell 112 are the same two cells in both runs (OFF is stable with them), so the nit rebuild is not what separates ON from OFF.

## 5. Hypotheses (NOT demonstrated)
H1: the fix is a correct local change (step 1 evidence). It changes the trajectory from step 1 in one cell and, through the atmosphere, in every cell from step 2 (ordinary divergence). The runaway is then a property of the port's land/surface coupling in the trajectory ON reaches, not a wrong value introduced by the fix. The D195 finite day would be a trajectory that happens to avoid it. Supporting: ON equals the real record where the fix acts; OFF is finite although less faithful there.
H2: the unstable element is the day-to-night transition of the surface layer in cell 112 at step 31: the exchange coefficient stays on the unstable branch (ch up to a 0.0625 value) with a small ground-to-air contrast, GHY then sees an evaporation/condensation demand it cannot absorb with the one internal iteration imposed from the real record (real nit = 1 at those steps because the real state was stable; nsub/edts are not recomputed from our state). Supporting: timing (step 31 flips ch, the ground temperature leaves the real path the same step), nit = 1 imposed, aevap swings in sign. Not tested: I did not replay cell 112 with a different substep schedule, did not read the PBL source for the 0.0625 value, did not capture the first-layer exports (utop, tkv, dbl) of the cell, and did not run the NumPy reference past step 18 (it aborts on the nit assertion at step 19 on the D199 code).
Not excluded: a defect in the port's PBL/GHY coupling for this cell that the real model does not have (the real record never has such a state, so the port code at those inputs was never validated against the real model).

## 6. Suggested next steps (not done)
(a) Capture the cell-112 inputs of the second substep at steps 29-31 (ON) and replay GHY eagerly with nit 1 versus a state-derived schedule; (b) record `ch`, `ws`, `lmonin` of the cell for the first substep too (needs r1 captured), and check where 0.0625 comes from in the PBL code; (c) an ON run with the cell's `ch` or nit held at the real record for steps 29-33 as an isolating experiment (a change of the port, owner decision); (d) ON/OFF to step 53 for the day score of OFF (finite 54 steps in D195, QCL step 1 still 2.84 beyond: section 9 stays NOT MET).

## 7. Not run / limits
No full-day (54-step) variants, no scoring (multiday_score) of the 41-step runs, no C1 against the NumPy chain, no test suite, no GHY internal-substep trace, no first-substep (r1) land capture, no GPU. Capture is the second substep of each step only. One run per variant; the machine was shared (loads 0.4-2.1).
Reproduce: `taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d201_day.py OUT --fix 0|1 --cap OUT/cap --nsteps 41 --c1dir OUT/c1dummy --nit-strict 0`; then `python d201_real_trace.py`, `python d201_compare.py <scratch>`, `python d201_cell112.py`.

## Parent-session check (2026-10-08)
Compared the saved states myself (T U V Q P, 41 steps): ON equals the D200 states bitwise (NaN positions included); OFF equals the D195 states (result printed in the parent report). At step 39 ON is non-finite and OFF is finite. The decisive fact (the elhx switch decides finite vs NaN) is demonstrated; the mechanism (H1, H2) is a hypothesis, not demonstrated. Nothing was loosened.
