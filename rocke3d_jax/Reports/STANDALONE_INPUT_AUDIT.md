# Standalone model: audit of the recorded inputs (2026-10-11, step 1)

Owner: G. Tamkin (decision 2026-10-11: "continue toward a standalone model, port the remaining recorded inputs, not sized in hours"). Written by the driver session. Status words: MEASURED (a run or comparison exists), READ FROM CODE (not run), NOT DONE.
This page starts the work; it does not claim a standalone model. Sources: `STATUS_2026-10-11.md` section 2, `FULL_FIDELITY_DELTAS.md` D174 section 1, registry of `gpu_runs/entlive/d193_run.json`, records in `ff_data/nov26_day2` (108 steps).

## 1. What is read per step in the latest run (MEASURED, registry of the two-day run)
Four per-step record bundles: `ci` CONDSE entry set (85.8 MB/step), `co` CONDSE exit (77.9 MB/step, used for RADIA cloud masking and references), `siter` radiation-step state (16.1 MB/step), `surface_records` ffp/ffs/ffl/ffg/fft (23.9 MB/step). Read once: `sitea`, `s1` (start state of step 0), restart statics, `surface_restart_for_melt_si`. A standalone model keeps only the once-per-run initial state.

## 2. `ci`, the CONDSE entry set (MEASURED over 108 steps, `ffc_cse_in_*`)
90 fields; only `FLICE` and `FOCEAN` are constant. Of the 90, `condse_inputs` (`atm_step.py`) already replaces 29 by chained values (state, pressure, hidden step-start fields, UKM/VKM) and 40 by the CONDSE carry (`CARRY_KEYS`). That leaves 20 taken from the record:

| Fields | Finding | Source | Consequence |
|---|---|---|---|
| `PREC`, `EPREC`, `PRECSS`, `DDM1`, `DDML`, `DDMS`, `TDN1`, `QDN1` | the entry value of step n+1 equals the CONDSE exit of step n bitwise in 107 of 107 transitions | `ci(it+1)` against `co(it)` | pure carry; only `TDN1`, `QDN1`, `DDMS` keep the entry value when no downdraft is found (`clouds_condse_ff.py` 353-357); the other five are assigned unconditionally (lines 358, 449-452). CLOSABLE by adding them to `CARRY_KEYS`: NOT DONE |
| `RNDSS`, `SEEDS` | random chain closed in D174 (`drv_rng.py`, 63 steps, 0 mismatches); only the first seed is an initial condition | D174 section 2 | wiring in the day driver only |
| `RSI` | already replaced by our MELT_SI result (`jax_coupled.py` 447) | READ FROM CODE | closed in the coupled path |
| `FEARTH`, `FLAND` | read by the CONDSE column code; change at the day boundary | `clouds_condse_ff.py` 260 | follow the day-boundary land-fraction update (done in `daily_lake_update`); NOT verified here that the record is no longer the source |
| `FLAKE`, `FLICE`, `FOCEAN`, `QGAVG`, `TGVAVG`, `USAVG`, `VSAVG` | not read by the CONDSE column code of the numpy port (no reference in `clouds_condse_ff.py`) | READ FROM CODE, grep | not proven unused in the batched device CONDSE; check before dropping |

LIMITS: the carry equality shows the record is self-consistent, not that our chained CONDSE output reproduces it (that is the existing free-run chain, scored statistically). Checked only `ci` against `co` of the same date; no other date.

## 3. Remaining bundles (READ FROM CODE, D174 table; not re-measured)
* `co`: needed on radiation steps for cloud masking; our own CONDSE exit (`_cloud_rad`) is used in the chain, so the dependence is for references and the server path: to check.
* `siter` (SRHR/TRHR/COSZ1): server mode computes SRHR/TRHR and COSZ1 (`--cosz computed`); what else is read from it is NOT DONE.
* `surface_records`: tile row set (10 distinct sizes, ice tiles appear and vanish), PBL template columns, soil statics, irrigation demand (needs the demand file), Ent per-call exports, land forcing columns, `nit_rebuild` rows, first-call Qf. This is the bulk of the remaining work and is where a template builder is needed (D174 blocker 1).

## 4. Order proposed (not a commitment)
1. Chain the 8 CONDSE outputs through `CARRY_KEYS`; test: `ci` bundle with these fields from our own previous exit, two-day score unchanged within the relaxed criterion.
2. Verify and drop the 7 unused-looking fields; wire `FEARTH`/`FLAND`/`RSI`/seeds so the `ci` read is only step 0.
3. Re-measure what is read from `co` and `siter` per step.
4. Template builder for the SURFACE tile rows and the PBL carry (largest item; no hour estimate exists).
- Reproduce section 2: `python ci_carry_audit.py` inside fullfidelity/ (graphcast-env, reads ff_data/nov26_day2).
