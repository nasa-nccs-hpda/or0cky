# SOCRATES spike: brief for a separate session (written 2026-10-11 by the driver session)

## 0. Owner decision (G. Tamkin, 2026-10-11)
"Lift rule for spike." The standing rule "SOCRATES is never ported or modified" is LIFTED FOR A SPIKE ONLY, on the terms of this page. It stays in force for everything else. A full port needs a further, separate, dated owner decision (decisions R0/D1-D3 of `SOCRATES_PORT_PLAN.md` section 8 are still open and are NOT answered by this one).

### 0b. Follow-up decision (2026-10-11)
D3 is approved for stages R1-R2 only (instrumented scratch copy, dumps, time-boxed); terms in `ACCEPTANCE_CRITERIA.md` section 12. The restrictions of section 3 below remain for everything outside that decision.

## 1. Question the spike must answer
Can a JAX radiation on one V100 get the full RADIA call (3,312 columns, longwave and shortwave) down to a cost that changes the benchmark answer, and at what effort? Today: unmodified Fortran RADIA = 8.1 s per radiation step on the benchmark node (8.4 s as the persistent server), 1 radiation step in 5, i.e. 1.62 s per step, 68 % of the Fortran step; the JAX hybrid cycle is 3.45 s against 2.37 s (`STATUS_2026-10-11.md`, `DECISION_MEMO_2026-10-11.md` section 7).

## 2. Scope (what the spike may do)
* Read the SOCRATES and ModelE sources (read-only), including `planet_rad.F90` and `radiance_calc`.
* Measure the UNMODIFIED Fortran: profile the persistent radiation server (or a standalone RADIA call) with an external profiler (`perf`, or timers outside the library); no SOCRATES file is edited. Result: the share of the call per routine/stage (gas optical depth with k-eqv scaling, cloud/aerosol properties, two-stream coefficients, flux solve, driver and diagnostics) and per band.
* Write a JAX prototype of the dominant kernel(s) on realistic inputs (shapes from the real spectral file: bands, k-terms, 3,312 columns x 40 layers), run it on a V100 (`gpu/v100_run.sh`), and time it.
* Compare the JAX prototype with the unmodified Fortran ONLY where the oracle allows it without touching SOCRATES (final fluxes and heating rates of the server for the recorded packets, `ff_data/nov26_day/rsv_n26_<it>_{in,out}.bin`, 11 radiation steps).
* Project the full-call cost on the V100 with every assumption written next to the number.

## 3. Not allowed without a separate approval
* Any edit of the SOCRATES or ModelE source tree, including an instrumented copy in a scratch tree (plan section 4 / decision D3).
* Changing the production radiation path, the persistent server, or `jax_radiation.py`; the spike lives in new files (suggested `fullfidelity/socrates_spike/`).
* Any claim of a full port, of bitwise equality on the GPU (not expected: device exp/pow differ), or of a speed-up of the model.
* Committing spectral files or data (259 MB + 276 MB): reference them by path.

## 4. Deliverables
1. `Reports/SOCRATES_SPIKE_RESULT.md`: the profile of the Fortran call (table, how measured, noise), the kernel(s) ported, the V100 timing of the prototype (compile separate from steady, several repeats), the projection to the full call with assumptions, what was validated against the oracle and how closely (categories A-D), and the effort left, re-estimated from the real reachable line count (plan stage R1).
2. Code and scripts in new files only, run commands in the result page.
3. A go / no-go recommendation against section 5, with the reasons; do not make the decision.

## 5. Gate (proposed by the driver session; the owner may change it)
Per radiation call, projected on the V100 for the full call, including host-side conversion and the packet transfer that the server path already costs (the JAX hybrid hand-off is 0.2 to 0.45 s per radiation step):
* GO to R1-R2 (reachability and instrumented dumps, needs decision D3): projection at or below 4 s per call (0.8 s per step averaged over the cycle of 5), AND the prototype within category C (1e-6) of the server's outputs on the recorded packets for the kernel(s) that can be checked.
* MARGINAL (owner decides): projection between 4 s and 8 s per call (parity with the Fortran RADIA 8.1 s is 8 s).
* NO-GO: projection above 8 s per call, or no kernel that can be validated.
Reason for 4 s: the benchmark bar in `DECISION_MEMO` section 7 needs the JAX ordinary work to fall to about 0.65-0.75 s per step; radiation on the device only pays if it also leaves room for that; at 1.62 s per step it is the Fortran's own cost, at 0.8 s it frees about 0.8 s per step (a third of the gap).

## 6. Starting points
* `Reports/SOCRATES_PORT_PLAN.md` (sizes, reachability, reuse list), `Reports/BENCHMARK_V100_PROTOCOL.md`, `Reports/STATUS_2026-10-11.md`.
* Oracle and packet code: `fullfidelity/radiation_server.py`, `radiation_server_persist.py`, `jax_radiation.py`, `drv_radpacket.py`; lessons list in the plan section 3 (`x**4`, `cumsum`, single-precision literals, REAL*16 sites).
* V100: `gpu/v100_run.sh NAME MINUTES 'command'` (graphcast-env, jax 0.5.3); keep the same node-type discipline as the benchmark (exclusive node, repeats).
* House rules of the repo: no secrets; label hybrid components; do not invent results; record owner decisions with dates; pushes come from the driver session (the spike session commits locally and says what to push).
