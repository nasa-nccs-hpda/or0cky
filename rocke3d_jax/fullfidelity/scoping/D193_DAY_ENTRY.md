# D193 (2026-10-08): S8 first attempt -- one model day (nov26, 54 steps) with the assembled device-resident coupled step, scored with the D192 scorer

Owner: G. Tamkin. Written by a Claude Code agent (subagent). Status: DRAFT ledger text; nothing committed. Review by: before a second day or a server-radiation day is run.
New files only: `d193_day.py` (runner), `d193_ref_day.py` (C1 reference), this entry, `d193_results/` (score.md/json, d193_run.json, d193_c1.json). No existing file edited; other sessions' untracked files not used. SOCRATES/RADIA not touched. No server process was started.
Conditions: `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, `import clouds_jax_env`, no compile cache, TMPDIR and D189_SHIM_DIR in the session scratchpad, ctx imf=True (libimf through host callbacks), shared node (load 1-7 during the runs), no GPU, jax 0.5.3, one run.

**HYBRID RESULT. Radiation was REPLAYED from the real records (SRHR/TRHR held between radiation steps, COSZ1 per step); it was NOT computed and no Fortran callback was made: "radiation replayed from the real record (not computed; no Fortran callback in this result)". The D192 scorer prints the fixed sentence "radiation computed by the original Fortran (hybrid component)" under its table; for this run that sentence does NOT apply (it was not edited). libimf math functions were provided by a host callback to the original build's runtime. Not an end-to-end JAX claim (D191 section 4 list of non-JAX parts applies).**

## 1. Inputs that exist (checked before the run)
`ff_data/nov26_day` holds all 54 steps 33312..33365 of: ffa_step_<it>_{a,d,e,r}, ffc_cse_in/out_<it>, ffp/ffs/ffl/ffg/fft_<it>, ffd_state_<it>_s1..s4, plus the five members ffpt_p1..p5. Nothing was missing, so the full day was run. `ff_data/nov26` has only 6 steps and was not used for the steps. Note: ffp_<it> (8 bytes) and ffs_<it> (1080 bytes) of `nov26` and `nov26_day` differ at step 33312 (same file sizes; all other files compared byte-identical for steps 33312, 33313, 33317); this run uses `nov26_day` for every step including step 0, so step 0 is not byte-for-byte the D191 step 0 input (C1 at step 0 is still 558/558 A, below).

## 2. Recorded inputs used (registry, strict; 8 of 20 declared inputs read, 11.07 GB read in total, no undeclared read; the registry source paths are printed with `nov26/` but the files read were under `nov26_day/`)
| input | file pattern | reads | size per file |
|---|---|---|---|
| sitea (hidden/PBL start state, step 0 only) | ffa_step_<it>_a | 1 | 39.8 MB |
| siter (SRHR/TRHR/COSZ1) | ffa_step_<it>_r | 54 | 16.1 MB |
| s1 (start state, step 0 only) | ffd_state_<it>_s1 | 1 | 34.0 MB |
| ci (CONDSE entry set) | ffc_cse_in_<it> | 54 | 85.8 MB |
| co (CONDSE exit, seeds/masking) | ffc_cse_out_<it> | 54 | 77.9 MB |
| surface_records (Ent exports, land forcing, tile radiation columns, PBL profile columns) | ffp/ffs/ffl/ffg/fft_<it> | 54 | 23.9 MB |
| surface_restart_for_melt_si | _pristine_restarts/<restart>.nc + ocean geometry | 1 | 0.5 MB |
| ctx_static (not sized) | make_ctx(nov26) files | 1 | - |
Plus (outside the registry, in `d193_day.py`, at the day boundary step 48 = it 33360 only, the D157/D171 convention of `atm_day_open_loop.run_day`): MDRYA from the real `ffd_glue_daily` dump, the ch4ox water mass DM derived from the REAL end state of step 47 (`ffa_step_33359_e`) and the real start state of step 48 (`ffa_step_33360_a`), and the recorded SNOAGE of `ffc_cse_in_33360` (SNOAGE change 14.04). DAILY_ATMDYN + ch4ox are applied on the host (NumPy, 9 atmosphere arrays, ONE declared device->host->device round trip) because `Coupled.step` has no day-boundary stage. Ocean/ice/land daily updates, if the real model has any, are not applied (as in D157/D171). Statics and the restart of nov26 as in D191. Comparison-only files (members, ffa_step_e) are read by the scorer only.

## 3. Run (d193_day.py; device block after every stage, `timed`)
- 54 of 54 steps completed, no error, stop-model flags 0 at every step, `n_slp_exp_branch` 0 cells at every step. Total wall 2135 s (35.6 min) including the cold step.
- Seconds per step: cold step 0 **437 s** (126 XLA compilations); **steady median 15.7 s (min 14.6, max 19.3; 40 steps without a rebuild)**; replay radiation.
- **Recompilations: YES.** The SURFACE stage is rebuilt when the static index sets of the template change (tile set changes): 13 rebuilds after step 0, at steps 7, 8, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 28; each = 6 XLA compilations and a step of 69-73 s (steps 1: 2 compilations, 18 s). Totals: 206 compilations, 798 s compile time over the day. A day with ~13 template changes therefore costs ~12 min of compile on top of ~14 min of steady stepping. `max_substeps` stayed 11 (Nw 2731, Nl 346, Ne 753 template sizes).
- jit executions per step 96-99 (97-98 on radiation steps/others), eager primitive dispatches 490-551 per step. Counted device->host calls inside the step: not re-instrumented here (D191: 1 call of 36 B).
- Non-JAX stages (registry text of the run, share of the measured step time): condse_mstcnv [JJ/NP] 29.2%, surface_tiles_land [JJ/REC] 36.2% (this run's blocked timing includes the SURFACE rebuild compiles, 13 x ~55 s), dyn [JJ/NP] 8.8%, condse_post [JJ/NP] 6.5%, ocean_b [JJ/NP] 4.8%, record_load [REC] 4.0%, template_build [REC/NP] 1.7%, condse_setup [JJ/NP] 0.9%, dissip_filter [JJ/NP] 0.8%, radia [JJ/REC] 0.1%, flag_read [NP] 0.0%. Shares are inflated for the compile steps; the steady split is that of D191 section 4 item 3.
- Host callbacks over the day: libimf_ops 76,259 callbacks, 206 M elements, 4.0 GB in / 2.4 GB out, 222 s; fused MSTCNV libimf 252,417 callbacks, 70 M lanes, 5.9 / 3.9 GB, 32 s; QUS 9,504 calls, 14.3 GB, 41 s; pole columns 108 calls, 30 s; OADVT2 pre-pass 216 calls, 15.5 s; radiation replay 11 calls (steps k = 0, 5, ..., 50), 370 MB device->host, 128 MB host->device, 0.26 s, 11 host synchronisation points. Sentence: "libimf math functions provided by a host callback to the original build's runtime".
- Tile-mask check (our MELT_SI tile set vs the recorded template validity): 0 mismatches at steps 0-42; **ocean/lake-water type mismatching cells at steps 43 (1), 48 (55), 49 (3), 50 (3), 51 (3), 52 (4), 53 (3)**; ice type 0. The new-tile donor rule (D190 section 6) is NOT implemented; the template of each step comes from the real record, so from step 43 the end of the day contains tiles whose set differs from the recorded one by 1-55 cells. Effect not diagnosed (the score at those steps is below).

## 4. Score (multiday_score.py, 5 real members p1..p5, whole-column rms, thresholds unchanged: within <= 1, near <= 2, beyond > 2 times the largest member)
Label: one model day (54 steps), hybrid, replayed radiation, 13 template-change recompilations.
| field | within | near | beyond | worst ratio steps >= 3 (step) | worst ratio all steps (step) | near steps | beyond steps |
|---|---|---|---|---|---|---|---|
| T | 54 | 0 | 0 | 0.94 (41) | 0.94 (41) | - | - |
| U | 53 | 1 | 0 | 1.02 (4) | 1.02 (4) | 4 | - |
| V | 54 | 0 | 0 | 0.90 (8) | 0.90 (8) | - | - |
| Q | 43 | 11 | 0 | 1.03 (47) | 1.03 (47) | 43-53 | - |
| P | 53 | 1 | 0 | 1.05 (23) | 1.05 (23) | 23 | - |
| QCL | 49 | 4 | 1 | 1.17 (39) | 2.84 (1) | 2, 13, 14, 39 | 1 |
| QCI | 54 | 0 | 0 | 0.98 (8) | 0.98 (8) | - | - |
Worst ratio over fields at steps >= 3: QCL 1.17 at step 39. Criterion (a) "within at every step, all fields": **not met** (Q, P, U, QCL near at some steps). Criterion (b) "never beyond 2x of the largest member": **not met over all steps** because of QCL at step 1 (ratio 2.84, noise floor ~1e-9 at k < 3, the same kind of case as D171 QCL step 1 with 2.90); **met for steps >= 3**. Which reading of "stays within" the owner applies is the owner's decision (as in D192). The 11 near steps of Q are the last 11 steps (43-53), coinciding with the first tile-mask mismatches (steps 43+); no causal claim (not tested).
Secondary (D157 convention, zonal-mean rms): within/near counts T 41/13, U 52/2, V 51/3, Q 53/1, P 47/7, QCL 46/7 + 1 beyond, QCI 54/0; max ratio 1.30 (QCL zrms at step 13). |global mean| classes: beyond at T 2 steps (max 2.22 at step 43), Q 1 (2.33, step 43), P 3 (3.16, step 43), QCI 1 (3.12, step 24); QCL none beyond (max 1.55). For reference D157 free-radiation day had |global mean| beyond for QCL 1, QCI 2, P 2 (ratios ~2.4-2.5); the P ratio 3.16 here is larger than any published D157/D171 value quoted in D192. Full tables in `d193_results/score.json`.
For comparison with D157 (free radiation, NumPy chain): same order (all fields never beyond at steps >= 3; Q near 11 vs 5; QCL ratio 1.17 vs 1.44; QCI 0 near vs 10). One day, five members, one start state: not a significance statement.

## 5. C1 (assembled device chain vs the NumPy libimf chain with the same closed surface, `d193_ref_day.py`, same pinning, each array compared per step; harness categories A bitwise / B <= 1e-12 / C <= 1e-6 / D worse)
| steps | result |
|---|---|
| 0 | 558/558 A |
| 1-6 | 569/569 A at every step (including radiation steps 5) |
| 7 | **first non-A: A 509, B 32, C 28**; surface, dissip, filter and land groups; ONE atmosphere column (64,19) (EGCM 2.0e-9 abs, PBLHT, PBLPTOP, Q 2.3e-13) and one land cell (85) (evap_max_ij, fr_sat_ij 8.8e-10, ht/w B). Step 7 is also the first step where our SURFACE stage was rebuilt (shape change). Cause not diagnosed. |
| 8-22 | A 108 -> 51, D 210 -> 512 of 569: the 1e-9 difference of one column is amplified chaotically; from step 8 on C1 is no longer informative about port consistency |
| 23-53 | **NOT AVAILABLE**: the NumPy reference aborted at step 23 (it 33335) inside `ghy_advnc_test_nit.build_batch_nit` (assertion `info['nit'] == ffnit[i]`, cell 111: recomputed 11 vs recorded 14), because the NumPy chain rewrites the template from its own diverged state and that helper asserts equality with the recorded substep count. Not worked around. |
So C1 holds bitwise for 7 steps (0-6) only; it is **not** a 54-step C1 and the assembled day is **not shown to be a bitwise copy of the NumPy chain beyond step 6**. The reference step costs 22-23 s (36-52 s when the batch is rebuilt; 161 s cold).

## 6. What was NOT run / limits
- No real radiation server (no radiation computed); no run with a second pairing of members; no C2 against the real end record per step beyond the D192 score (the score IS a comparison to the real end states); no transfer-guard run; no run without libimf; no GPU; no timing without the stage blocking; no repeat (one run per number).
- New ice tile donor rule not implemented (section 3). Template columns, Ent, land forcing and PBL profile columns are recorded for every step, not computed.
- Day-boundary update is a host NumPy step (section 2).
- C1 only to step 22 compared, valid (A) to step 6; the NumPy reference cannot run past step 22 with the existing code.
- Scratch outputs (per-step arrays, 24 GB) were deleted after the comparison; per-step end states `ours_d193/step_<it>.npz` are in the session scratchpad `d193/out/` (not in git); the compact results are in `d193_results/`.

## 7. Reproduce
`taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d193_day.py OUT --c1dir C1DIR` (about 36 min); `... python d193_ref_day.py OUT C1DIR 54` (aborts at step 23); `python multiday_score.py --ours OUT/ours_d193 --nsteps 54 --cache members.json --md score.md --json score.json`.
