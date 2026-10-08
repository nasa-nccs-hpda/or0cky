# D197 (2026-10-08): diagnosis of the QCL step-1 exceedance of the nov26 day (ratio 2.84)

Owner: G. Tamkin. Written by a Claude Code agent (subagent). Status: DRAFT diagnosis; nothing committed, no tracked file edited. Review by: before any decision on the day criterion (ACCEPTANCE_CRITERIA section 9).
Scope: diagnosis only. No tolerance, category or threshold changed; no fix attempted. SOCRATES/RADIA not touched; other sessions' untracked files not edited (the repo code imports some of them read-only).
Conditions: `taskset -c 3-5`, `OMP_NUM_THREADS=1`, no JAX compile cache, `TMPDIR` and `D189_SHIM_DIR` in the session scratchpad (`.../scratchpad/d197/`), jax 0.5.3 CPU, `import clouds_jax_env`. Scripts (scratchpad, not in git): `d197/load.py a1.py ... a9.py b1.py ... b7.py`.
Naming: "step k" = end state of iteration 33312+k. Cell indices are 1-based (i,j,layer), j = latitude row of 46, layers 1 (surface) to 40.

## 0. Result in six lines
1. At step 1 the QCL error is ONE cell: (i,j,L) = (31,12,15) holds 96.8% of the squared error (ours 1.6495e-3, real 1.2456e-3, difference +4.04e-4; the five real members all have 1.2456e-3 there). Without that cell the whole-column rms is 2.07e-7, ratio 0.51 to the largest member distance (4.06e-7). Without the top 5 cells: 1.05e-7 (ratio 0.26).
2. Step 0 is not a comparison on the same footing: our step-0 end state is almost bitwise the real one (QCL rms 1.1e-20), so its ratio is about 0. Step 1 is the first step whose input carries our B-level (about 1e-13 relative) residuals of step 0.
3. The CLOUDS stage port is not the cause at step 1: run from the exact real step-1 inputs it is category A/B against the real exit record (QCL max difference 5.4e-20, cell value equal). The radiation hand-off/carry (CLDSS/CLDMC masking, TAUSS, QLSS, ...) of step 1 is consistent with the real record. Our assembled step is bitwise equal to the NumPy libimf chain at step 1 (C1 569/569 A, D195 section 3), so the NumPy reference chain has the same QCL value; the difference is not specific to the JAX port.
4. The trigger is the dynamics exit of step 1 (the CLOUDS entry): our P/PMID/MA/T differ from the real exit by up to 127-222 ulp, and the pressure-derived PK differs by 1 ulp in 38,046 of 132,480 cells. Feeding ONLY our PK (everything else the real CLOUDS entry) into the CLOUDS stage reproduces the +4.04e-4 jump in the cell exactly. The cell is a knife edge of the large-scale (stratiform, LSCOND) part in a convective column: CLDSS at layer 13 is 0.361 in the real exit and 0 in ours, QLSS at layer 15 differs by +4.04e-4; MSTCNV outputs (TMC, QMC, CLDMC, LMC) agree to 1e-13 or better.
5. Random +-1 ulp noise on PK (8 draws, CLOUDS entry otherwise real): 7 of 8 give a QCL rms of 1.3e-7 to 3.4e-7 (the size of the members' 2.3e-7 to 4.1e-7); 1 of 8 (seed 0) flips the SAME cell by the SAME 4.040e-4 and gives rms 1.17e-6, the value we see. So the 2.84 is the outcome of a bistable cell that ulp-level noise can hit; five members that happen not to hit it give a largest distance that does not represent it. The real model itself flips this cell at step 4 (3 of 5 members jump from 5.4e-5 to 7.0e-4).
6. This does NOT make the criterion pass: ours is 2.84 times the largest member at step 1 and outside the leave-one-out range of the members at that step (0 to 1.56). The finding is that the cause is a threshold flip amplified from ulp-level input residuals, reproducible by 1-ulp noise, and not a defect found in the CLOUDS port. Nothing is relaxed; section 9 of ACCEPTANCE_CRITERIA is unchanged by this file.

## 1. Where the step-1 error sits (task 1)
Metric as the scorer: whole-column rms over valid cells (`atm_day_report.diff_metrics`), ours = `d195/out/ours_d193/step_33313.npz`, real = `ffa_step_33313_e.bin`. Scripts `a1.py`, `a4.py`.
- QCL rms(ours - real) at step 1 = 1.1529e-6 (scorer 1.15e-6); largest member 4.063e-7 (p5); p1 2.60e-7, p2 2.40e-7, p3 0 (bitwise the control at this step), p4 2.27e-7.
- Squared-error shares (step 1): top 1 cell 96.8%, top 5 cells 99.2%, top 10 99.6%; top 1 column 97.2%. Layer 15 holds 97.0%, layer 14 1.3%, layer 16 0.8%. Latitude row j = 12 holds 97.2%, j = 21 1.1%, j = 30 0.8%. Max |difference| 4.039e-4 at (31,12,15); next cells (10,21,14) 4.2e-5, (20,30,16) 3.6e-5, (31,12,9) -2.3e-5. Cells differing at all: 11,007 (mostly roundoff size), > 1e-9: 154, > 1e-7: 79, > 1e-5: 9.
- rms with the top N cells removed: N=1 2.07e-7, N=5 1.05e-7, N=10 7.2e-8, N=50 7.4e-9.
- Convective versus large-scale: 100% of the squared error is in columns with a convective plume (real exit LMC top > 0; those are 63% of valid columns), 100% in cells with CLDMC > 0 and 99.97% in cells with CLDSS > 0 (so both cloud types co-exist where the error is). The dominant cell is in a deep plume column (LMC top 6, base 17 in both ours and real) with CLDMC 0.47 and CLDSS 0.23 at the cell. The quantity that differs is the stratiform (LSCOND) one. Per layer, ours - real, column (31,12): CLDSS L13 -0.361, L15 +0.027; QLSS L15 +4.04e-4, L14 -1.4e-5, L13 -2.9e-6, L9 -2.3e-5; CLDMC differs at most 1.1e-13, QLMC 3e-17, TMC 3.4e-13, QMC 1.2e-16, LMC identical. Real exit at L13: QLSS 2.9e-6 with CLDSS 0.36 (a marginal cloud that ours does not form).
- Threshold flip: the step is dominated by one such decision (cloud present/absent at L13 and the connected condensate redistribution at L9, L13-L15). Which comparison inside LSCOND flips (e.g. RH against RH00, or a clear-fraction branch) was NOT traced (section 6).
- Steps 2 and 3, same cell: step 2 rms 1.21e-6 vs largest member 7.6e-7 (ratio 1.59, "near"), 79% of it from (31,12,15) (ours 3.243e-3, real 2.862e-3); step 3 rms 1.25e-6 (ratio 0.49, a member is at 2.56e-6), top cell then (4,19,17) with 64% (-3.6e-4). Step 4: the real members themselves jump at (31,12,15): QCL 5.4e-5 (p1 5.39e-5, p3 and the control) versus 7.04e-4 (p2, p4, p5).

## 2. Step 0 versus step 1 (task 2)
- Step 0: ours end state T/Q/U/V rms 7e-14/1e-17/8e-15/9e-15, QCL 1.1e-20, QCI 2e-20, P 1.7e-12 against the real end state (D191 category B for the end fields). The members at step 0 are 0 (p3) to 2.9e-7 (p5) in QCL. The step-0 ratio is 0.00 ("below"), not a close call.
- The step-0 end state is identical bit for bit with the D191 `nov26` inputs and with the `nov26_day` inputs (ffp 8 bytes, ffs 1080 bytes differing, D193): compared `d191/final/nov26_asm.npz` with step 0 of the day run on 12 arrays (filter T Q U V QCL P PK PMID MA, dyn PK, condse QCL, surface T), max difference 0 for all (`b7.py`). The ffp/ffs recorded-input difference therefore does not enter the step-0 end state and cannot cause the step-1 difference through step 0. (Not tested: an effect of those files on step 1 itself; `nov26` has no step 1 record.)
- Step 1, stage by stage versus the real stage-exit records (`a8.py`, harness categories): dyn exit B 24 / C 2 of 26 keys (DPDX 1.9e-11, U 5.6e-13 relative); then condse exit D in 12 of 18 keys (QCL relative 0.19, PRECSS 0.13, PREC 0.089, QMOM 4.5e-3, Q 3.5e-3), carried through radia/surface/dissip/filter (QCL 0.19). At step 0 the dyn exit was A 26/26 and the condse exit A 14 + B 4. Step-1 dyn exit versus the real, in cells and ulp: P 2759 of 3312 cells differ (max 127 ulp), PMID 55,937 cells (124 ulp), PDSIG 66,333 (174 ulp), MA 67,986 (222 ulp), T 117,178 (109 ulp), PK 38,046 cells (34 ulp max; 1 ulp in the flip column at layers 13, 15, 17). At step 0 all of these had 0 differing cells.
- Radiation hand-off at step 1 (record replayed; radiation steps are 0, 5, 10, ...). Carry out of step 0 (ours) versus the real CLOUDS entry of step 1 (`ffc_cse_in_33313`): TAUSS, QLSS, QLMC, SVLHX, SVLAT, TTOLD, LMC, RHSAV, CLDSAV, CLDSAV1, FSS bitwise equal; TAUMC 1.1e-14, CSIZMC 3e-15. Our step-0 exit CLDSS is bitwise the real exit (CLDMC 7e-18) and differs from the step-1 entry in 70 and 110 cells, as it must: the RADIA masking (`atm_step.radia_cloud_masking`, TAUSS/TAUMC <= taulim) applied to the real step-0 exit reproduces the real step-1 entry CLDSS/CLDMC exactly (max difference 0). So the masked carry equals the real entry. The assembled step's in-chain device carry was not extracted directly; its equality to the NumPy chain at step 1 (C1 A) and the NumPy chain's explicit masking (atm_step.py lines 769-770) carry the argument. RQT/KLIQ affect only the radiation computation, which is replayed (not computed) in this result.
- MSTCNV/LSCOND at step 1: LMC identical in all columns, CLDMC/TMC/QMC differences <= 1e-13 in the dominant column; the divergence is in the stratiform part (section 1). From real inputs both are category A/B (section 4).

## 3. NumPy libimf chain reference (task 3)
`d195_results/d195_c1.json` (D195 section 3): the assembled JAX step equals the NumPy libimf chain (`d193_ref_day.py`) bitwise at steps 0 and 1 (558/558 and 569/569 category A, `not_A` empty). Therefore the NumPy chain has the same step-1 QCL, and the difference is in the inputs/chain residuals shared by both, not in a JAX-only operation. The saved C1 arrays of D195 were deleted, so the NumPy end-state QCL was not recomputed. I re-ran the first two steps of the JAX day (`d195_day.py --nsteps 2 --nit-strict 0`, cores 3-5, step 0 525 s cold, step 1 19.3 s): its `filter/QCL` at steps 0 and 1 is bitwise equal to the saved `ours_d193/step_33312.npz` and `step_33313.npz` (`a7.py`). The two-step NumPy chain was NOT re-run; the CLOUDS stage was run in NumPy (libimf) from the real step-1 inputs (section 4), which is the code the reference chain uses for that stage.

## 4. Cause (task 4, mechanism tests; scripts `b1.py`-`b6.py`)
Tests on the CLOUDS (CONDSE) stage alone (NumPy libimf `fast_condse`, `real_state_at(R,'condse')` = real dyn exit + real carry), compared with the real exit `ffc_cse_out_33313`.
| test | result |
|---|---|
| CLOUDS from the real step-0 inputs | A 14, B 4; QCL cell 3.140173e-3 = real; max QCL difference 0 |
| CLOUDS from the real step-1 inputs | A 4, B 14 (worst PREC 1.9e-15); QCL cell 1.245983e-3 = real; max QCL difference 5.4e-20 |
| CLOUDS from OUR step-1 dyn exit | reproduces our chain bit for bit (cell 1.649966e-3; max difference to our chain condse/QCL 0) |
| real dyn exit + eps*(ours - real) on all differing keys | eps 1e-4 to 0.1: cell unchanged (< 1e-17), max dQCL elsewhere 4.5e-5 to 8.1e-5; eps 0.5 and 1: cell 1.649966e-3 (+4.04e-4) |
| ONE key from ours, others real | PK alone: cell +4.040e-4 (flip). No other single key moves the cell (MWS, PDSIG, PMID, Q, T, TMOM: cell < 4e-17, elsewhere <= 4.5e-5; U, V, MA, P, PEDN, PEK and the rest: 0) |
| PK of ours in column (31,12) only, one layer at a time (L11, 12, 13, 15, 17, 18) | no flip individually (<= 4e-17); whole column of PK (31,12): flip (+4.040e-4, rms 1.137e-6) |
| random {-1,0,+1} ulp noise on PK of the real entry, 8 seeds | rms 1.17e-6 (seed 0: same cell, same +4.040e-4); 1.3e-7, 2.5e-7, 2.2e-7, 1.7e-7, 3.4e-7, 3.3e-7, 2.0e-7 (seeds 1-7, cell unchanged) |
Reading: (a) not the CLOUDS port (A/B from real inputs at steps 0 and 1); (b) not the radiation hand-off (carry equals the real entry); (c) the input residual that decides is a 1-ulp PK difference over several layers of one column, and ulp noise on PK alone can produce the same flip. The flip is bistable in these tests (1.2456e-3 or 1.6500e-3, nothing in between), so the QCL rms is either about 2e-7 or about 1.15e-6. The draws are at the CLOUDS exit, not the step end (step-end values: 1.15e-6 in the day run versus 1.14-1.17e-6 at the CLOUDS exit).
Port difference or not: the only difference to the Fortran in the chain is the ulp-level residual of the step-0/step-1 dynamics exit (categories B/C, the known state of the port, D187/D191), which the CLOUDS stage amplifies in a knife-edge cell. I found no coding difference in CLOUDS. Whether the dynamics-exit residual itself (P/MA 100-200 ulp at step 1, with an exactly bitwise dyn exit at step 0) is a port difference that could be removed is not addressed here.
Recorded-input difference: excluded for step 0 (section 2). Step 1 uses the day records only.

## 5. What 5 members can resolve: leave-one-out at step 1 and base rates (`a5.py`, `a6.py`)
Member distance = rms(member - real unperturbed end state), from `d193/members.json` (the scorer's `member_curves`). Leave-one-out (LOO) ratio of member m = rms_m / max over the other four.
| step | QCL member rms (p1 p2 p3 p4 p5) | LOO ratios (p1 p2 p3 p4 p5) | ours | ours / largest member | ours / second largest |
|---|---|---|---|---|---|
| 0 | 2.1e-16 2.8e-8 0 1.5e-7 2.9e-7 | 0.00 0.09 0.00 0.53 1.89 | 1.1e-20 | 0.00 | 0.00 |
| 1 | 2.60e-7 2.40e-7 0 2.27e-7 4.06e-7 | 0.64 0.59 0.00 0.56 1.56 | 1.15e-6 | 2.84 | 4.43 |
| 2 | 7.6e-7 5.8e-7 1.2e-23 6.6e-7 4.5e-7 | 1.16 0.77 0.00 0.86 0.59 | 1.21e-6 | 1.59 | 1.84 |
| 3 | 1.26e-6 7.3e-7 3.1e-7 2.56e-6 1.53e-6 | 0.49 0.29 0.12 1.67 0.60 | 1.25e-6 | 0.49 | 0.82 |
- p3 is bitwise the control at steps 0-2 (distance 0 to 1e-23): the effective ensemble at those steps is four members, p5 the largest.
- QCL step 1: LOO range 0.00 to 1.56 (no member exceeds 2x the others); ours is 2.84 of the largest and 4.43 of the second largest member, i.e. outside the LOO range at step 1. All 54 steps: QCL LOO range 0.00 to 2.92 (p4, step 5); LOO ratio > 1 in 20% of member-steps (one of five members is always the largest), > 2 in 1 of 270 for QCL.
- All 7 fields x 54 steps x 5 members: 9 of 1890 member-steps are > 2x the other four (QCI 5, V 2, P 1, QCL 1; T, Q, U 0). Largest: QCI 1.2e9 (step 0, p5 against four members at 3e-17: not meaningful), V 13.6 (step 0), QCI 6.5 (step 1), P 3.4 (step 1), QCL 2.9 (step 5). Early steps with a tiny spread are where max-of-five is the poorest estimate of the possible spread.
- Single-cell size at step 1: the members' largest single-cell QCL differences are 8.1e-5 (p1), 4.5e-5 (p2), 0 (p3), 4.5e-5 (p4), 7.4e-5 (p5); ours 4.04e-4 (5x the largest). The members flip the very cell (31,12,15) at step 4 (3 of 5, to 7.0e-4).
- Flip probability for ulp-level noise from section 4: 1 of 8 draws (binomial 95% interval about 0.3% to 53%; one CLOUDS entry, PK only). If it were 1/8, the chance that all four non-trivial members miss the flip at step 1 would be 0.875^4 = 0.59; the data do not separate 1/8 from a much smaller value.
- The members start from a 1-ulp perturbation of the initial state; we carry residuals of about 100 ulp in P/MA at step 1 (section 2), so the two samples are not the same kind of perturbation (no claim on how that changes the flip probability).

## 6. What was NOT done / limits
- LSCOND not traced to the exact comparison that flips (RH vs RH00 / clear-fraction branch at L13, L15 of column (31,12)); no instrumentation, no edits.
- The NumPy chain was not re-run for two steps (relied on D195 C1 569/569 A at step 1, and the NumPy CLOUDS stage from real inputs); the NumPy end-state QCL was not recomputed.
- The random-ulp test is 8 draws on PK only, at the CLOUDS exit of one step; no other field, no other step, no full-step version. Only "1 of 8" is supported.
- What the real members perturb (which field, which ulp) is not documented here; only their recorded states were used.
- No claim that the day criterion is met or relaxed. Steps >= 3 numbers of D195 unchanged. The day was not repeated.
- Not tested: whether a bitwise step-0 end state (so that the step-1 dyn exit is bitwise) removes the flip; D187/D191 report the step-0 end as B.

## 7. Reproduce
```
S=<scratchpad>/d197; cd fullfidelity
taskset -c 3-5 env OMP_NUM_THREADS=1 TMPDIR=$S/tmp D189_SHIM_DIR=$S/tmp python d195_day.py $S/run --nsteps 2 --c1dir $S/c1 --nit-strict 0   # about 9 min
python $S/a1.py   # shares, layers, columns, latitude, top cells, members (steps 0-3)
python $S/a4.py   # convective/large-scale split, top-cell table, per-step field rms
python $S/a5.py   # leave-one-out ratios (members.json cache of D193)
python $S/a8.py   # stage-by-stage categories versus the real stage records, carry
python $S/b1.py 0 1; python $S/b2.py; python $S/b3.py; python $S/b6.py   # CLOUDS from real inputs, ours dyn exit, single-key swaps, ulp noise
```
Inputs read: `ff_data/nov26_day` (ffa_step_<it>_{a,e}, ffc_cse_in/out_<it>, ffd_state_<it>_s3/s4 via `atm_step.Real`, ffpt_p1..p5), `scratchpad/d195/out/ours_d193/step_*.npz`, `scratchpad/d193/members.json`, `scratchpad/d191/final/nov26_asm.npz`. Outputs only in the scratchpad.
