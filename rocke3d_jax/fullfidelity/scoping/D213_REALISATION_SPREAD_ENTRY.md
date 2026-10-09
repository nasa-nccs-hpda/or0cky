# D213: realisation spread of our own 54-step day score

Owner: agent D213 (session of 2026-10-09). Sources: this directory's `d213_results/`, `../d213_day.py`, `../multiday_score.py` (unchanged), `instrumentation/ATM_DRV_pert.f.patch`, `instrumentation/build_and_run.md` (D149-D151), D209/D210/D212 results. Review date: next time a configuration difference in the day score is quoted as an effect.

## 1. Question and method
D209/D210/D212 quote day scores that differ between configurations, each a single realisation. Question: how large is the spread of the score when only the initial state is perturbed by one ulp, the way the five real members are?

* Perturbation of the real members (read from `ATM_DRV_pert.f.patch` and `build_and_run.md`): `ffpt_perturb`, called at atm_phase1 entry of step 0, applies `nearest(x, s)` to atm_com T or Q: p1 `T 36 23 10 1`, p2 `T 20 12 20 -1`, p3 `Q 50 30 5 1`, p4 `T 10 30 1 1`, p5 `T 0 0 0 1` (every cell, all levels, checkerboard sign). `d213_day.py` (new, copy of `d209_day.py`; no tracked file edited) applies the SAME one-ulp change (np.nextafter, 1-based indices, array layout (IM,JM,LM)) to `state['S']['T'|'Q']` in `initial_state` through env `D213_PERT`; unset = D209 behaviour. The applied old/new values are in `d213_results/pert_*.json`.
* Runs: HEAD 15baa26 defaults (replayed radiation, computed day constants, daily lake, nit-strict 0), `taskset -c 0-2,6-7`, `OMP_NUM_THREADS=1`, identical for every run, run sequentially (`chain.sh`): p1 1569 s, ctrl 1553 s, p2 1535 s, p3 1443 s (about 25 min each, so K=3 plus control was done; p4, p5 NOT run). Each scored with unchanged `multiday_score.py` against the 5 real members (`score_<run>.md|json`).

## 2. Checks that the experiment is what it claims (demonstrated)
* The control at HEAD is bitwise identical to the D209 `run_c/ours_d193` states (54 steps, all 7 fields, 0 differing arrays): the run is reproducible under this pinning and the score is the D209 score.
* Perturbation fidelity (p1 vs our control, step 0 / 1 / 28 / 53, rms of T difference): ours 4.5e-5 / 1.4e-4 / 2.8e-2 / 7.5e-2 against the real p1-minus-ctrl 4.5e-5 / 1.9e-4 / 3.2e-2 / 7.8e-2; Q likewise (6.5e-7 equal at step 0). Our perturbation grows like the real one (same order of magnitude; not identical, as expected for a different implementation).
* p3 (Q one ulp at (50,30,5)) was absorbed: its states equal the control to <= 1e-18 (1-2 cells in Q) up to it 33345, T differs in 2 cells at 33346 and in all cells (2.5e-3) from 33347, reaching 0.39 at the last step (`p3_vs_ctrl.txt`). So p3 is a distinct realisation only for steps >= 35; for steps 0-34 it duplicates the control. The ensemble is therefore 3 distinct realisations (ctrl, p1, p2) plus p3 for the last 19 steps.

## 3. Scores (within / near / beyond of 54 steps; worst ratio steps >= 3 with step; QCL/U step-1 ratio) 
| run | T | U | V | Q | P | QCL | QCI |
|---|---|---|---|---|---|---|---|
| ctrl (=D209) | 54/0/0 | 54/0/0 | 54/0/0 | 42/12/0 | 52/2/0 | 52/1/1 | 37/17/0 |
| p1 | 53/1/0 | 53/1/0 | 54/0/0 | 49/5/0 | 54/0/0 | 50/3/1 | 41/13/0 |
| p2 | 53/1/0 | 53/1/0 | 54/0/0 | 47/7/0 | 54/0/0 | 50/3/1 | 52/2/0 |
| p3 | 54/0/0 | 54/0/0 | 54/0/0 | 42/12/0 | 52/2/0 | 52/1/1 | 37/17/0 |
| D210 server rad | 54/0/0 | 54/0/0 | 53/1/0 | 54/0/0 | 52/2/0 | 52/1/1 | 49/5/0 |
| D212 ctl54 (= D210 numbers) | same as D210 | | | | | | |
| D212 srv54 (server + own fields) | 54/0/0 | 53/1/0 | 54/0/0 | 54/0/0 | 51/3/0 | 41/12/1 | 42/12/0 |

Spread over ctrl, p1, p2, p3 (min..max) and the worst ratio at steps >= 3 (full per-run values in `spread_table.txt`):

| field | within-count | step-1 ratio | worst ratio steps >= 3 | beyond |
|---|---|---|---|---|
| T | 53..54 | 0.42..0.73 | 0.94..1.11 | 0 in all |
| U | 53..54 | 0.07..1.41 | 0.85..0.91 | 0 |
| V | 54..54 | 0.05..0.48 | 0.85..0.97 | 0 |
| Q | 42..49 | 0.28..1.03 | 1.01..1.03 | 0 |
| P | 52..54 | 0.00..0.02 | 0.99..1.04 | 0 |
| QCL | 50..52 | 2.83..2.92 | 0.95..1.07 | 1 (step 1) in all four |
| QCI | 37..52 | 0.11..0.64 | 1.07..1.84 | 0 |

Configuration scores against that spread:
* D209 (replay) is the control itself, one point of the spread.
* D210/D212-ctl (server radiation) vs the spread: QCL 52/1/1 inside (50..52), worst k>=3 0.99 inside (0.95..1.07), QCL step 1 = 2.83 inside; QCI 49 inside (37..52), worst k>=3 1.08 inside (1.07..1.84); P 52 inside. Q 54 is ABOVE the observed maximum (49) by 5; V 53 is 1 below the V range (54), which is not meaningful.
* D212 srv54: QCI 42 and worst 1.18 inside; Q 54 above the range as D210; QCL 41/12/1 is BELOW the observed range (50..52) and its worst ratio at steps >= 3 (1.41) is ABOVE the observed maximum (1.07); P 51 and worst 1.05 are just outside (52..54, 0.99..1.04).

## 4. Conclusions (honest)
Demonstrated:
1. The one-ulp realisation spread of our own day score is large for the cloud fields and Q: QCI within-count 37..52 (15 steps), Q 42..49, QCI worst ratio at steps >= 3 1.07..1.84, QCL near/within 50..52. Therefore the D209 -> D210 QCI change (37 -> 49 within, worst 1.69 -> 1.08) lies inside the realisation spread (p2 alone scores 52/2/0 and 1.07 with replayed radiation); it is NOT evidence that server radiation improved QCI.
2. QCL step 1 = 2.83..2.92 (beyond) in every one of the four realisations and every configuration (D209, D210, D212): it is systematic, independent of the one-ulp perturbation and of radiation mode, and not a realisation effect. The "beyond only at step 1" statement is stable.
3. The T/U/V/P within-counts differ by at most 2-3 steps between realisations (and 1 step in near/within), so the quoted differences in those fields between configurations (0-3 steps) are inside the spread.
4. Only one configuration difference lies outside the observed spread: D212 server+own-fields QCL 41/12/1 with worst ratio 1.41 at steps >= 3, versus 50..52 and <= 1.07 over four realisations; and the D210/D212 Q = 54/0/0 versus 42..49 (above the range). These are outside the observed range, not shown significant: the range of 3 distinct realisations underestimates the true spread, the configurations are single realisations, and the real-model members themselves are only 5 one-ulp realisations.
Hypotheses (not tested): (a) the D212 QCL degradation is a real effect of the own-state radiation fields (it is outside every observed value) - would need K >= 3 realisations of the D212 configuration to establish; (b) the Q improvement under server radiation is real - same requirement. Not run: p4, p5, any perturbed run of the server-radiation configurations (needs the Fortran server, cores 3-5), other days.

Bottom line: for QCI, P, T, U, V and QCL (D210) the configuration differences are not significant against the one-ulp realisation spread; the QCL step-1 exceedance is systematic; the D212 QCL result and the D210/D212 Q result lie outside the 4-run spread but are unconfirmed single realisations. No tolerance was changed; SOCRATES/RADIA untouched; nothing committed.

## 5. Files and reproduce
`fullfidelity/d213_day.py` (new); `scoping/d213_results/`: `score_{ctrl,p1,p2,p3}.md|json`, `spread_table.txt`, `p3_vs_ctrl.txt`, `pert_*.json`, `d193_run_*.json`, `day_*.log`, `chain.sh`, `chain.status`; this entry. Run states/logs in the scratchpad `d213/run_*` (not committed). Reproduce: `D213_PERT='T 36 23 10 1' taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<s> D189_SHIM_DIR=<s> python d213_day.py OUT --nit-strict 0 --daily-lake 1`; `python multiday_score.py --ours OUT/ours_d193 --nsteps 54 --cache members.json --md score.md --json score.json`.

## Parent-session check (2026-10-09)
Re-read and re-scored the p2 realisation myself: all 54 steps finite; differs from the control from step 0; table equals the agent's (T 53/1/0, U 53/1/0, V 54/0/0, Q 47/7/0, P 54/0/0, QCL 50/3/1, QCI 52/2/0; QCL step-1 ratio 2.92; QCI worst 1.85 at step 2 over all steps, 1.07 at step 8 for steps >= 3). I did not re-run ctrl, p1 or p3; the ctrl = D209 identity is the agent's check (0 of 54 steps x 7 fields differ). Conclusions accepted: with K = 3 distinct realisations the within-counts spread is QCI 37..52, Q 42..49, QCL 50..52; the D209 -> D210 differences in QCI and QCL are inside it; QCL step 1 (2.83..2.92) is systematic in every run; D210/D212 Q = 54 and D212 QCL 41/12/1 (worst 1.41) lie outside a 3-run range, which underestimates the true spread, so they remain single unconfirmed results. p3 (a Q perturbation) was absorbed until step ~34 and is not an independent realisation for steps 0-34; p4 and p5 were not run.
