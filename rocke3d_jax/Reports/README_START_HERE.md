**PORT RUNNING (2026-09-29, ~16:00): D39, polar UOD/VOD relaxation block, instrumented and
dumped on all 3 real dates; Python port not yet written.** Real Fortran records for
`ffz_polerelax_{geom,<itime>}.bin` (`OCNDYN2.f`'s `OCEANS`, the "relax UOD,VOD toward 4-pt avgs
of UO,VO" block plus its `polevel()` pole-velocity reconstruction) are in `ff_data/{nov26,dec01,
jan01}/`, validated against no-regression on every earlier delta's dumps. Next action on resume:
write `polerelax_ff.py`/`polerelax_jax.py`, validate, write tests, update the ledger, commit.

**D38 FINISHED (2026-09-29 ~15:07): `OBDRAG2` (implicit bottom-layer current drag) ported and
validated.** Float64-op-order exact against real Fortran, both plain-Python and JAX ports, all 3
dates, first try. Full regression: **522 passed, 0 failed.** `FULL_FIDELITY_DELTAS.md`'s D38 entry.

**D37 FINISHED (2026-09-29 ~11:50): `OCOAST` ported; permanent fix for a restart-file trap that
had cost real time across D27/D36/D37.** GISS ModelE's restart reader picks whichever of
`fort.1.nc`/`fort.2.nc` has the *later* itime, so a prior run in a reused scratch directory can
leave the "wrong" one looking pristine. All three test dates' pristine restarts are now archived
permanently at `ff_data/_pristine_restarts/fort1_{nov26,dec01,jan01}_itime{33312,33552,17520}.nc`
— restoring before a rerun is a three-line copy, not a debugging cycle. See `build_and_run.md`.

**D36 FINISHED (2026-09-29 ~08:40): major scope correction — `OCNDYN.f`'s legacy ocean
dynamical core confirmed entirely dead code.** `OCNDYN.f`'s own leapfrog-dynamics driver
(`OCEANS_old`) is commented out in full; the live driver is a complete rewrite in `OCNDYN2.f`
(`SUBROUTINE OCEANS`, called from `OCN_DRV.f:41`). Every call site of `OCNDYN.f`'s
`OFLUX`/`OADVM`/`OADVV`/`OPGF`/`OVtoM`/`OMtoV`/`OSTRES`/`OBDRAG`/`OPFIL` project-wide was checked
and found dead — do not port these from `OCNDYN.f`. `OSTRES2` (the live equivalent) ported and
validated same delta. `GLMELT` (glacial meltwater) scoped and **deferred**: fires only once per
calendar day, not confirmed within the existing 6-step/3-hour test windows.

**Stage 2 (ocean core) opened 2026-09-29 (D33): `PRECIP_OC`, `OSOURC`, `GROUND_OC`'s below-
freezing sweep all validated (D33-D35)** before the D36 scope correction above. Ocean grid
confirmed same resolution as the atmosphere (IMO=72, JMO=46) — a major de-risking finding.

**Stage 1 (ice dynamics) closed 2026-09-29 (D26-D32): `PRECIP_SI`/`PRECIP_LK`/`PRECIP_LI`,
`DYNSI`/`VPICEDYN`/`FORM`/`PLAST`/`RELAX` (3 real bugs found and fixed), `CALC_APRESS`,
`seaice_to_atmgrid`, `UNDERICE`** all ported and validated bitwise/float64-exact against real
Fortran. `IRRIG_LK` deferred (external dataset dependency, not yet needed).

**Standing constraint, unconditional, from every session since it was set:** SOCRATES (the
radiation library) is third-party — **never port or modify it**; call it as a black box only if
radiation work is undertaken.

**Standing decision, twice user-confirmed:** the full faithful port of DYNSI (ice dynamics) and
the entire ocean numerical core (Stage 2, ~19,000 lines before D36's dead-code correction), no
reduced/mixed-layer approximation. This supersedes an earlier (2026-09-23, Track A) decision
recorded in `STATUS.md` *against* a full-fidelity port — that document is now superseded for
everything after D25; see "Document map" below.

# Start here: where the work is recorded

This is the `full-fidelity-port` branch's index. It exists because the branch's own three
tracking documents (`FULL_FIDELITY_PLAN.md`, `FULL_FIDELITY_DELTAS.md`,
`fullfidelity/PHASE0_LOG.md`) are exhaustive working records, not a fast way to answer "what is
the state of the port right now" — that is this file's job. Read `Project_Summary_and_Conclusions.md`
next for the one-page version.

## Current state

**38 deltas complete (D1-D38), D39 mid-flight.** The branch has been under continuous, explicitly
user-directed autonomous work ("keep going all night, never stop until there is no work left in
the port") since 2026-09-28 evening. Every delta follows the same **dump-hook-and-validate**
method: read the real Fortran source fully, instrument it with a new dump-hook subroutine and
call-site edit in a scratch copy of the tree, generate a diff patch, rebuild the real `ifort`
executable, rerun all 3 real test dates (1950-11-26, 1950-12-01, and a `1951-01-01`-equivalent
"jan01" point, 6 `DTsrc` steps / 3 hours each) against real restart files, copy the new dumps into
`ff_data/`, write a plain-Python reference port and a batched JAX port, validate both against the
real dumps (bitwise or float64-rounding exact — never a synthetic/approximate substitute except
where explicitly documented), write a pytest suite with mutation and non-vacuous-branch checks,
rerun the **entire** project regression suite, update the three tracking documents, commit and
push. Nothing is marked done without a real Fortran number to check it against.

**Test count: 522 passed, 0 failed** (as of D38's commit `8057eed`; D39 will add more once its
port is written). Every regression run this session has been zero-failure — no delta has ever
broken an earlier one.

**Nothing is running or queued** outside the current turn's own background rebuild/rerun/test
cycles; the session is autonomous and self-paced, not waiting on the user for anything.

## Standing conditions

- **SOCRATES is off-limits** (see the top-of-file note) — always.
- **Full faithful port, not a simplified ocean** — always; do not substitute a reduced/mixed-layer
  ocean without surfacing it as a decision first, the same way this scope itself was surfaced.
- **Never pause for confirmation mid-port.** The user corrected this explicitly once
  ("why did you stop and wait for me? we lost three hours") — work continues through debugging,
  deltas, and documentation without stopping to ask.
- **Timer-table trip counts are not proof of execution** — GISS ModelE checkpoints its cumulative
  CPU-timer table into the restart file, so a stale trip count can carry forward unchanged across
  reruns even when a routine never actually ran. Verify with real dump files, not the timer table.
- **`./P2SAoM40` (what actually executes) is a separate file from `P2SAoM40.bin`** (the build
  output), not a symlink — copy a fresh build to both names after every rebuild.
- **Reset BOTH `fort.1.nc` and `fort.2.nc` to the same pristine itime before every rerun** in a
  reused scratch directory — the restart reader picks whichever has the later itime, and GISS
  ModelE double-buffers checkpoint writes across both. Permanent pristine copies for all 3 test
  dates: `ff_data/_pristine_restarts/`.
- **`cp` is aliased to `cp -i`** in this environment — force-copies must use `\cp` or `rm -f` first,
  or a background task will hang forever on an interactive prompt no one can answer.

## Document map

**Start with `Project_Summary_and_Conclusions.md`**: the one-page version — goal, method,
headline results, what's open, where the evidence is.

**The full-fidelity port itself (branch `full-fidelity-port`, started 2026-09-24; Stage 1/Stage 2
work started 2026-09-28):**
- `../FULL_FIDELITY_PLAN.md`: the scope, delta-by-delta, with every subroutine's line count, what's
  real per-step physics vs. diagnostics vs. one-time init, and what's dead code (D36's correction).
- `../FULL_FIDELITY_DELTAS.md`: the full ledger, one entry per delta (D1 onward), each with the
  exact instrumentation, validation numbers, and any bugs found and fixed.
- `../fullfidelity/PHASE0_LOG.md`: the narrative working log — the debugging stories behind the
  ledger entries (e.g. the `P2SAoM40`/`P2SAoM40.bin` binary-naming bug, D29's three real numerics
  bugs, D36's dead-code discovery, the restart double-buffering trap and its permanent fix).
- `../fullfidelity/instrumentation/build_and_run.md`: the actual build/run/dump recipe, with every
  operational lesson (binary naming, restart double-buffering, `cp -i`, patch-application order)
  recorded as a standing warning, not just a one-time fix.

**Earlier work (branch history before the Stage 1/Stage 2 full-fidelity push):**
- `../STATUS.md`: **partially superseded.** Accurate for "Track A" (the earlier, deliberately-
  reduced representative driver — GHY/ATURB/SEAICE/LAKES/PBL/SURFACE chained and JAX-vectorized,
  with measured CPU/GPU speedups) and for background facts about `P2SAoM40` itself (a ROCKE-3D 2.0
  rundeck: Sun-like star, SOCRATES radiation, dynamic ocean, 40-layer atmosphere at 72x46
  horizontal resolution; Zenodo-archived reference config). **Its "Recommendation" section
  (2026-09-23) explicitly decided against a full-fidelity DYNSI/ocean port — that decision was
  later reversed** by explicit, twice-confirmed user direction; the full-fidelity work above is
  the result. Do not cite that section as current.
- `../README_GPU.md`, `../RESTART_SESSION.md`: Track A setup notes, still accurate for that track.
- `../status_slides/`: an HQ-audience slide deck snapshot (Track A only; predates the full-fidelity
  push). Live/editable version linked from `status_slides/README.md`.

## Open items

1. **D39 (polar UOD/VOD relaxation block) is instrumented and dumped but not yet ported** — the
   Python/JAX port, validation, and tests are the immediate next step on resume.
2. **`GLMELT`** (glacial meltwater, `OCNDYN.f`, 72 lines) is scoped but deferred: it fires only
   once per calendar day (`daily_OCEAN`'s `end_of_day` gate), and the current 6-step/3-hour test
   windows are not confirmed to cross a day boundary. Needs either a day-boundary-crossing test
   window or a decision to skip it.
3. **`OFLUXV`+`OPFIL2`+`ODHORZ`/`ODHORZ0`** (`OCNDYN2.f`'s mass-flux/horizontal-dynamics core) is a
   "new architecture" item on the scale of D29's ADI solve — it needs polar Fourier filtering
   against an external `AVR` reduction-matrix binary file plus an FFT (`OFFT`/`OFFTI`). Deliberately
   deferred past the smaller D36-D39 deltas; not yet started.
4. **`OADVT2`/`OADVTX2`/`OADVTY2`/`OADVTZ2`/`OADVUZ`** (tracer advection family, `OCNDYN2.f`) — not
   yet read in detail.
5. **`OCNQUS.f`, `OCNKPP.f`, `OCNMESO_DRV.f`+`OCNTDMIX.f`+`OCNGM.f`, `OSTRAITS.f`+
   `OSTRAITS_COM.f`** (~11,300 lines combined: advection, vertical mixing, mesoscale/GM-Redi
   mixing, straits) are **entirely unread** — per this project's own discipline (applied
   consistently since before D33), no time estimate is given for unread code. `OCNKPP.f` and the
   mesoscale-mixing files are each individually likely to be another "new architecture"-scale item,
   by analogy with vertical-mixing schemes elsewhere in ocean GCMs, but this is not yet confirmed.
6. **`IRRIG_LK`** (Stage 1, external prescribed-irrigation dataset dependency) deferred since D28,
   not yet needed by anything downstream.
7. **The JAX/batched `jax.lax.while_loop` version of `DYNSI`'s own outer `KKI` loop** (`VPICEDYN`)
   is still plain-Python only — the per-cell Newton/ADI solve inside it is batched and validated,
   but the outer iteration-count loop itself has not been converted.
8. **No whole-model chained Track B step exists yet** for the full-fidelity branch (Track A has
   one, `run_steps_device`, but it uses the reduced/approximated physics this branch exists to
   replace) — every piece validated so far is a per-routine, per-call-record match against real
   Fortran, not yet wired into a running chained step.

## Suggested wording for a new session

"Read `Reports/README_START_HERE.md` and `Reports/Project_Summary_and_Conclusions.md`, then
continue the full-fidelity port from the open items list — dump-hook-and-validate methodology,
never pause for confirmation, SOCRATES stays untouched, full faithful ocean/ice-dynamics port
(no simplification)."
