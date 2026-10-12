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
