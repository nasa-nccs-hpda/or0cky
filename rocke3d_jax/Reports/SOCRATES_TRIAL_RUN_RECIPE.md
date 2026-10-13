# Run recipe: ported model (non-radiation) with a replacement radiation component, nov26 window

Owner: G. Tamkin (driver session). Written 2026-10-12 for the SOCRATES session. Project-local; every number below was measured or read from the files named; items marked NOT VERIFIED are not.

## 1. What you get

JAX port of dynamics, clouds/CONDSE, surface, land (GHY + Ent), ocean, sea ice, lake on one V100. Radiation is called through a server interface (`radiation_server_persist.PersistentServer.request`); today it is the unmodified Fortran RADIA helper process. The trial replaces that call.

Reference result with the Fortran helper (job `allfix2`, V100, 108 steps nov26 it 33312-33419, 2026-10-12, commit `23f9870`):
* finished, no error, no flag raised, finite fields; held step median about 1.7 s; wall 1.7 s at step 107.
* score `fullfidelity/d207_score.py` against five real one-ulp members: worst ratio at steps >= 3: 2.37 (QCI, step 3), the same value as every earlier configuration; day 2 worst 1.16. Unrelaxed 2x criterion NOT MET (QCI 2.37 at step 3, QCL 1.56 at step 4); relaxed criterion (3x at steps >= 3) met.
* score file: `gpu_runs/allfix2/score.md`, `gpu_runs/allfix2/score.json`.

Compare a trial run to this band only (score bands), never trajectories: identical configurations diverge from step 2 at about 1e-15 and reach order 1 locally by day 2 (noise floor, `Reports/STANDALONE_INPUT_AUDIT.md` section 6).

## 2. Command

From `rocke3d_jax/`, on a V100 (partition compute, constraint v100, graphcast-env, jax 0.5.3):

```
gpu/v100_run.sh NAME 60 'ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1 GPU_DAYDIR=nov26_day2 GPU_DAY_DRIVER=d212_day python -u gpu_day_run.py /ABS/PATH/gpu_runs/NAME --rad server --surf-fields computed --seed computed --cosz computed --nsteps 108 --timed 0'
```

* Output: `gpu_runs/NAME_<jobid>.out` and `gpu_runs/NAME/ours_d193/step_<it>.npz` (T,U,V,Q,P,QCL,QCI).
* Score: `cd fullfidelity && XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover" python d207_score.py --ours ../gpu_runs/NAME/ours_d193 --nsteps 108 --json OUT.json --md OUT.md` (graphcast-env python).
* Cold compile is about 12-13 min; the first step of a fresh process is slow. All flags are default off; the list above is the full set used in `allfix2`.
* Probe or other scripts must import `gpu_step_run` before anything else (GPU preamble).
* Use a branch/worktree that contains `23f9870` (local on `full-fidelity-port`; check `git log` for the push state before relying on origin).

## 3. Where radiation plugs in

`fullfidelity/d212_day.py`: `RAD['mode']=='server'` builds `radiation_server_persist.PersistentServer` and wraps it in `SrvWrap` (`RAD['wrap']`); the model calls `wrap.request(state, itime, seed)` on every radiation step (`A.is_radiation_step(it)`). `state` is a dict with the `radiation_server.INPUT_FIELDS` names and Fortran shapes (T, Q, PK, PMID, PDSIG, MA, BYMA, PEDN, cloud fractions/mixing/diameters, TAUSS, TAUMC, CLDSS, CLDMC, RQT, KLIQ, SNOAGE, LTROPO, RSI, ZSI, SNOWI, POND_MELT, FLAG_DSWS, FLAKE, DLAKE, FLICE, FLAND, FEARTH, GTEMPR1-4, TSAVG, WSAVG, SNOWLI, ZSNOWI, BARESW, FRSNOW, SNOWD). The return value is a dict of `radiation_server.OUTPUT_FIELDS`:
`T, Q, SRHR, TRHR, RQT, KLIQ, SNOAGE, CLDSS, CLDMC, FSF, TRSURF, ALB, FSRDIR, SRVISSURF, FSRDIF, DIRVIS, DIRNIR, DIFNIR, SRDN, CFRAC, COSZ1, AIJ` plus `_server_s`, `_wall_s` (used for timing records).

Minimal replacement: a class with the same `request(state, itime, seed=None)` signature and attributes `binary`, `restart`, `rundir` (read by `SrvWrap.__init__`), returning that dict, selected where `d212_day.py` builds the server (lines about 110-125). Everything downstream is unchanged.

Output fields read downstream (verified by reading the code): `SRHR`, `TRHR` (heating hold of the atmosphere step, `jax_atm_phase1.py`), `T`, `Q`, `RQT`, `KLIQ`, `SNOAGE` (state/hold update; `RQT`, `KLIQ`, `SNOAGE` are the radiation's own persistent state), `FSF`, `TRSURF`, `TRHR[0]`, `SRVISSURF`, `FSRDIR`, `COSZ1` (surface columns and Ent inputs, `drv_radcols_dev.rs_from_server`, `jax_coupled.py` radcols stage). `ALB`, `FSRDIF`, `DIRVIS`, `DIRNIR`, `DIFNIR`, `SRDN`, `CFRAC`, `AIJ`: returned and saved to `server_out_<it>.npz`; NOT VERIFIED whether anything else consumes them.

Radiation steps in the window: every `A.is_radiation_step(it)` (the 11 recorded live packets of day 1 are in `ff_data/nov26_day/rsv_n26_<it>_{in,out}.bin`; day 2 has no recorded packets, so no packet comparison runs there). A replacement can be checked against those packets of day 1 with `SrvWrap` (it logs `packet_vs_recorded_packet` and `output_vs_recorded_output` in `d212_server_calls.json`).

## 4. Things that will bite

* Precision: records and the server use float64 big-endian packets; the downstream columns are float64 (`rs_from_server`). A single-precision SOCRATES output changes the band; expect to report it as such.
* `SEEDS`/random state: the model computes the cloud RNG seed itself (`--seed computed`) and passes `seed` to `request`; the Fortran helper uses it for its own McICA draws. Whether SOCRATES needs it is yours to decide.
* The radiation state `RQT`, `KLIQ`, `SNOAGE` is held by the model between calls (`hold` in `jax_atm_phase1.py`); at step 0 they come from the step-0 dumps (`lv`).
* Start state: the run still reads 5 step-0 dumps from `ff_data/nov26_day2` (`sitea, s1, ci, co, siter`, about 278 MB); no per-step record read is left (shown by the 240-step run `long5d`, `Reports/STANDALONE_INPUT_AUDIT.md` section 10).
* Known limits, unchanged: nov26 only; irrigation uses year 1850 and no fossil groundwater; speed about 3.2 s per cycle against 2.4 s for the Fortran model.
* Regression: `fullfidelity/run_all_tests.sh` with the XLA_FLAGS above exported; not re-run after `23f9870` (flags default off, default paths unchanged).

## 5. What to send back

The `.out` file, `score.md`/`score.json` of the trial, and `d212_server_calls.json` (per-call times and, for day 1, comparison against the recorded packets). Report the SOCRATES radiation time per call separately from the model step time.

Sources: `fullfidelity/radiation_server.py` (INPUT_FIELDS, OUTPUT_FIELDS), `fullfidelity/d212_day.py` (SrvWrap, RAD), `fullfidelity/drv_radcols_dev.py` (rs_from_server), `fullfidelity/jax_coupled.py`, `gpu/v100_run.sh`, `Reports/SOCRATES_SPIKE_BRIEF.md`.

## 6. RESULT of the integration, 2026-10-12 (MEASURED, one V100, jobs `allfix2` and `native108`)

Wiring (commit after `dffe148`, merge of `socrates-spike`): `--rad native` in `fullfidelity/d212_day.py` builds `socrates_spike/radia_native.NativeHandoff` from `RadiaJax` (SOCRATES port of the other session, unmodified) and installs it as the hand-off of `jax_atm_phase1` (`ph._handoff`, `ph._hstep = jax_radiation.make_handoff_step`): radiation runs inside the same jitted step, no server process. `jax_coupled.py` copies the radiation outputs the surface columns read (`FSF, TRSURF, TRHR, SRVISSURF, FSRDIR, COSZ1`) into `drv_radcols_dev.LAST` as the server wrapper does. Inputs (products of nov26, static0, spectral data `spec2`) were copied from the SOCRATES session scratch to `/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/gpu_data_stage/socrates_nov26/` (`ROCKE_SOCRATES_DIR`); products are valid for nov26 only.

Command: the one of section 2 with `--rad native` in place of `--rad server`.

| | Fortran server (`allfix2`) | JAX SOCRATES (`native108`) |
|---|---|---|
| finished, finite, no flag | yes | yes (108 of 108 steps) |
| radiation step, median wall (steps 5..105) | 10.0 s | 1.9 s |
| non-radiation step, median wall | 1.7 s | 1.7 s |
| sum of steps 2..107 | 360 s | 186 s (1.9x) |
| step 0 (compile + first step) | 779 s | 1,089 s (about 310 s more compile) |
| worst ratio, steps >= 3, both days | 2.37 (QCI, step 3) | 2.36 (QCI, step 3) |
| day 2 worst ratio | 1.16 | 1.24 (QCI step 56) |
| unrelaxed 2x criterion | NOT MET | NOT MET (QCI 2.36 step 3, QCL 1.76 step 4) |
| relaxed criterion (3x, steps >= 3) | met | met |

Score files: `gpu_runs/native108/score.md`, `score.json`; model output `gpu_runs/native108/ours_d193/`.

Not shown: other dates (products exist for nov26 only), the 750 AIJ radiation diagnostics (four accumulators are rebuilt), a run longer than the 108 recorded steps, memory use of the combined program (it ran on a 32 GB V100), a second repetition of the timing. At step 1 the native and server runs already differ locally (T by up to 0.07 K at 24 points, from a 5e-9 K difference after radiation step 0): a discrete event of the kind seen before, judged only by the score band.

## 7. Day products and a five-day run (MEASURED 2026-10-12, V100 jobs `nativedays` 37941657, `native5d` 37941659)

Configuration: `--rad native` with the per-day products of `socrates_spike/daily_products.py` (section 14 of the SOCRATES report; rundir of input files built in `gpu_data_stage/socrates_nov26/rundir`, `ROCKE_NATIVE_DAYS` defaults to the days the zenith table reaches: 7 for 60 radiation steps), the flag set of section 2.

| run | steps | day-1 worst (>= 3) | day-2 worst | whole-window worst (>= 3) | relaxed 3x | unrelaxed 2x |
|---|---|---|---|---|---|---|
| `native108` (nov26 products for every day) | 108 | 2.36 (QCI step 3) | 1.24 (QCI 56) | 2.36 | met | NOT MET |
| `nativedays` (day products) | 108 | 2.37 (QCI step 3) | 1.11 (QCI 72) | 2.37 | met | NOT MET |
| `native5d` (day products, 7-day window) | 240 (first 108 scored) | 2.37 | 1.13 (QCI 70) | 2.37 | met | NOT MET |

* `native5d`: 240 of 240 steps, finite at every sampled step (every 24th), no flag; mean P 814.45 -> 814.77 hPa, mean T 93.83 -> 93.87 (the model's T units as saved), max |U| 82-116 m/s; steps >= 2: median 1.7 s, sum 415 s; no zenith-table / products-window warning. Steps 109-239 have no real-model reference: no fidelity statement beyond day 2.
* FSF excursions (the RXSNOW 0.05 threshold the SOCRATES session asked about): NOT detected and NOT excluded at the time of this 5-day run (superseded by section 10: same-packet check on 72 packets over 30 days, none). The per-radiation-step FSF/SRVISSURF/COSZ1 of all 48 radiation steps are in `gpu_runs/native5d/native_rad_outputs.npz`; a temporal-residual statistic over lit cells has median 34 W/m2 and maximum 574 W/m2 (cloud and terminator variability at 2.5 h sampling), so it cannot resolve an isolated few W/m2 excursion. That test needs the same packet through the Fortran server and the JAX radiation (as in the SOCRATES report 13.1), not two different trajectories.
* Regression at `93933e9` (clean worktree, XLA flags exported): main suite 3329 passed, 1 failed, 16 skipped (`test_libimf_fused::test_dq_dev_and_conv_micro_dev_bitwise_vs_cmj_libm_and_mutation`), separate-process files all passed. The failing test passes alone, with `test_jax_imf`, and paired with each of the 85 earlier test files individually; the polluting combination was not found.
* Same-packet Fortran-vs-JAX check on one V100 (float64, 38 dates; SOCRATES report 14.3, `socrates-spike` eeef43c, `socrates_spike/data/r6_days_e2e_v100_result.json`; reported by the SOCRATES session, not re-run here): no albedo-threshold excursion at any date (FSF/SRDN <= 1.1e-8 W/m2, SRHR 2.5e-9, TRHR 1.1e-7, T 1.5e-8 K), 0.2 s per call; KLIQ differs by 1 in some entry at every date (relative-humidity threshold flip). The 5.4 W/m2 hazard of their 13.1 is not excluded for other states.

## 10. Shadow comparison over a free-running 30 days (D230, job 37941828, 2026-10-12/13)

Owner decision (2026-10-12): the staged shadow comparison instead of a year-long control run. Owner of the radiation port: SOCRATES session; run by the port-driver session.

* **Method.** One `--rad native` run of 1440 steps (30 model days from nov26, itime 33312 to 34751, all default-off flags on as in `gpu/benchmark_native_v100.sh`, `ROCKE_NATIVE_NRAD=288`, day products for 30 days). With `ROCKE_SHADOW_DIR` (opt-in, `d212_day.py`) the packet and the native output of every 4th radiation step were dumped (72 packets, days 694-723, 12 different local times); `fullfidelity/shadow_compare.py` then sent each packet, in time order, to one persistent unmodified Fortran radiation server and compared the 21 output fields. Same packet, two radiations; no trajectory is compared. Script: `gpu/shadow_native_v100.sh`. Result files: `gpu_runs/shadow30/shadow_compare.json` (per packet and per field), `gpu_runs/shadow30/d193_run.json`.
* **Result.** **0 of 72 packets show an albedo-threshold excursion** (threshold 1e-6 W/m2 in FSF, SRDN or SRHR). Largest differences over all 72 packets: FSF 3.3e-9, SRDN 3.3e-9, SRHR 3.8e-9, TRHR 5.4e-8 W/m2, SRVISSURF 1.7e-9, ALB 1.6e-12, T 4.1e-8 K, RQT 6.7e-8. Q, SNOAGE, CLDSS, CLDMC, CFRAC, COSZ1 identical in all 72. KLIQ differs by 1 in up to 4 of 529,920 entries in 44 of the 72 packets (the relative-humidity threshold flip of SOCRATES report 13.1).
* **Validity limit of the run itself (important).** The driver flag `qdynam_do_z_extra` fired on 24 steps: k = 1288-1306 and 1332-1336 (itime 34600-34618, 34644-34648; model days 26-27). That is the QDYNAM extra-column vertical branch, which the port does not execute (`jax_p1_dyn.py` header: "the step must then be declared invalid"). So **model states from step 1288 onward are not valid model states**; the run finished, but a free-running model of this port is demonstrated for 1288 steps (about 26.8 days), not 30. The 7 packets at itime >= 34600 are radiation-only evidence (same packet, same result: FSF 1.5e-9 at most), not evidence about the model. The 65 packets before the first flag are the clean sample (largest FSF 3.3e-9, SRHR 3.8e-9, TRHR 5.4e-8, T 1.5e-8). What triggers the branch (a dynamics instability growing in the free run, or a state the port handles differently from Fortran) was NOT investigated.
* **Other observations.** The run completed all 1440 steps (`completed` true, no error, no nit rebuild or mismatch, finite at the first and last step, T 31-451 and 30-449 in the saved units); `mask_mismatch` (first count, tile-mask check) rose from 0 to 9 by step 1439 (first at the step-240 day boundary); its meaning for a free run was not investigated. Step time was 7.5-8 s (about 3.5x the benchmark) with the speed session's job on the same node; this run is not a timing benchmark. Cold compile 1,108 s + 111 s; wall 12,810 s for the steps.
* **Not shown.** A year; any other season (the radiation steps are nov26-december states); the fidelity of the model state against Fortran over 30 days (not compared; trajectories decorrelate by the noise floor); A100/float32.
