# ROCKE-3D → JAX full-fidelity port: what was done and what was concluded (to D42, 2026-09-29)

This is the front page for the full-fidelity port (branch `full-fidelity-port`). It states the
goal, the method, the results with their numbers, what is still open, and where each piece of
code lives. It links to the detailed ledger for the evidence rather than repeating it;
`README_START_HERE.md` is the index of that ledger.

**Status of code:** every delta described here is committed and pushed to `full-fidelity-port`
on the `or0cky` GitHub repository (`nasa-nccs-hpda/or0cky`), most recently commit `54e207d` (D42).
Nothing is staged-but-uncommitted; the standing workflow commits and pushes at the end of every
delta.

## 1. Summary

- **Goal.** Port NASA GISS's ROCKE-3D 2.0 (`ModelE2_planet_2.0`, a Fortran GCM) to Python/JAX with
  full numerical fidelity — validated call-by-call against the real, unmodified Fortran executable,
  not a reduced or approximated stand-in. Config: `P2SAoM40` (Sun-like star, SOCRATES radiation,
  dynamic 13-layer ocean, 40-layer atmosphere at 72x46 horizontal resolution).
- **Two prior, now-superseded directions.** An earlier "Track A" effort built a deliberately
  reduced, representative driver (GHY/ATURB/SEAICE/LAKES/PBL/SURFACE chained, JAX-vectorized, with
  measured CPU/GPU speedups — `STATUS.md`) and in 2026-09-23 explicitly recommended *against* a
  full-fidelity port. That recommendation was **reversed by later, twice-confirmed user direction**:
  a full faithful port of ice dynamics (DYNSI) and the entire ocean numerical core, no
  simplification. This document covers only the resulting "Track B full-fidelity" work.
- **Method.** Dump-hook-and-validate: instrument the real Fortran with new dump subroutines,
  rebuild the real `ifort` executable, rerun it on 3 real dates against real restart files, port
  the dumped routine to plain Python and to batched JAX, and check both against the real numbers
  — bitwise or float64-rounding exact, with a pytest suite (including mutation and non-vacuous-
  branch checks) added per delta.
- **Progress.** 42 deltas complete (D1-D42). Ice
  dynamics (Stage 1, D26-D32) is fully closed. Ocean core (Stage 2, D33-D42 so far) is underway;
  a major scope correction (D36) found that roughly half of the previously-estimated ocean
  dynamical-core code was dead/superseded, redirecting the remaining work to the correct file.
- **Result so far.** Every ported routine matches the real Fortran to float64 precision on every
  real test-date record checked — no exceptions, no approximated substitutes, on **522 passing
  tests** as of D42 with **zero failures** at any point this session. D42 (`ODHORZ`) closed
  the first "new architecture"-scale delta -- comparable in scope to D29's ADI solve.
- **Not established yet:** the ocean dynamical core beyond D33-D42 (mass-flux/pressure-gradient
  solve, tracer advection, vertical mixing, mesoscale mixing, straits — all separately scoped,
  ~11,300+ lines still unread); a chained whole-model Track B step; GPU speed numbers for any of
  the Stage 1/Stage 2 pieces (not yet JAX-batched beyond the per-routine level; no GPU used this
  session).

## 2. What was ported

| | Stage 1: ice dynamics | Stage 2: ocean core |
|---|---|---|
| Scope | DYNSI and its full numerical core (~1,460 lines) | The live ocean dynamical core, `OCNDYN.f`'s real physics + `OCNDYN2.f` (~19,000 lines before D36's dead-code correction) |
| Status | **Closed** (D26-D32) | **Underway** (D33-D39; D36 corrected the remaining scope) |
| Real per-step physics ported | `PRECIP_SI`/`PRECIP_LK`/`PRECIP_LI`, `DYNSI`/`VPICEDYN`/`FORM`/`PLAST`/`RELAX`, `CALC_APRESS`, `seaice_to_atmgrid`, `UNDERICE` | `PRECIP_OC`, `OSOURC`, `GROUND_OC`'s freezing sweep, `OSTRES2`, `OCOAST`, `OBDRAG2`; polar UOD/VOD relax block in progress |
| Deferred | `IRRIG_LK` (external dataset dependency) | `GLMELT` (cadence mismatch with test windows), `OFLUXV`/`OPFIL2`/`ODHORZ` (external file + FFT, "new architecture" scale) |
| Validation | Bitwise/float64-exact against real Fortran, all real records, both plain-Python and JAX ports | Same standard, same result so far |

## 3. What was found: real bugs, and one major scope correction

| # | Finding | Outcome |
|---|---|---|
| 1 | A run script executes a file literally named `P2SAoM40`, a **separate file** from `P2SAoM40.bin` (the build output), not a symlink | Every rebuild must copy the new binary to both names, or a stale binary silently keeps executing. Documented as a standing warning (`build_and_run.md`), cost real debugging time before being caught (D29). |
| 2 | GISS ModelE's timer table checkpoints into the restart file — a stable trip count can be **stale**, carried over from an earlier run, not proof a routine executed | Never trust the timer table alone; verify with a real dump file. |
| 3 | Three real numerics bugs in the DYNSI ice-dynamics port (`osurf_tilt` default, a `bydts`/`dts` unit-convention slip, an `AA1+AA2` vs `AA3+AA4` transcription error) | Found and fixed via the dump-hook-and-validate method itself — each was caught because the port didn't match the real Fortran, not by inspection. |
| 4 | **`OCNDYN.f`'s entire legacy ocean-dynamics driver (`OCEANS_old`) is dead code** — commented out in full, superseded project-wide by a rewrite in `OCNDYN2.f` | Major scope correction (D36): avoided porting ~9 subroutines that would never be validatable (they never execute), and redirected the real remaining Stage 2 work to the correct file. |
| 5 | GISS ModelE's restart reader picks whichever of `fort.1.nc`/`fort.2.nc` has the **later** itime, and checkpoints double-buffer across both files | A reused scratch directory can silently make a "fresh" rerun start from the *end* of the previous test window instead of the beginning — looks like success (`Terminated normally`) but validates nothing. Cost real time across D27, D36, and D37 before a **permanent** fix (archived pristine restarts for all 3 test dates) closed it for good. |

## 4. The work, in order

**A. Track A (2026-09-17 to 09-24, before this branch).** A deliberately reduced, representative
driver chaining GHY/ATURB/SEAICE/LAKES/PBL/SURFACE, JAX-vectorized with measured CPU speedups
(20-104x per component) and a confirmed 4.2x GPU speedup on the fused driver. Explicitly and
deliberately *not* full-fidelity — see `STATUS.md`'s now-superseded "Recommendation" section.

**B. Branch opened, full-fidelity direction set (2026-09-24 onward).** The user reversed Track
A's recommendation: full faithful port of DYNSI and the entire ocean numerical core, twice
confirmed, no reduced/mixed-layer substitute. `FULL_FIDELITY_PLAN.md` scoped the work.

**C. Stage 1: ice dynamics (2026-09-28 to 09-29, D26-D32).** `PRECIP_SI`/`PRECIP_LK`/`PRECIP_LI`
(one per land/lake/sea-ice precip path), then the 1,304-line DYNSI numerical core
(`VPICEDYN`/`FORM`/`PLAST`/`RELAX`) — bitwise/float64-exact on all 18 real records after finding
and fixing three real bugs (finding 3, above) — then `CALC_APRESS`, `seaice_to_atmgrid`, and
`UNDERICE` (`iceocean_fluxes`/`icelake_fluxes`) closed out the stage.

**D. Stage 2 opened: ocean core (2026-09-29, D33-D35).** `PRECIP_OC`, `OSOURC` (its
solar-penetration-profile dependency turned out to be fully analytic, re-derived rather than
dumped), and `GROUND_OC`'s below-freezing layer sweep (one genuine external-file dependency,
`OFTAB`'s specific-heat table, handled by recording its one use point). Confirmed the ocean grid
matches the atmosphere's resolution — a major de-risking finding for the atm-ocean regrid
boundary.

**E. Scope correction and pivot to the live dynamical core (2026-09-29, D36).** Found `OCNDYN.f`'s
legacy driver dead (finding 4, above); redirected to `OCNDYN2.f`. Scoped and deferred `GLMELT`
(cadence mismatch) and the `OFLUXV`/`OPFIL2` combination ("new architecture" scale). Ported
`OSTRES2` (momentum-stress application) the same delta.

**F. Smaller `OCNDYN2.f` deltas, plus a permanent operational fix (2026-09-29, D37-D38).**
`OCOAST` (coastal tracer-gradient damping) and `OBDRAG2` (implicit bottom-layer current drag),
both bitwise/float64-exact, first try, on all 3 dates. D37 also closed finding 5 (above)
permanently, which paid for itself immediately on D38's rebuild (a three-line restore instead of
a bootstrap run).

**G. Polar velocity handling and pressure-gradient prep (2026-09-29, D39-D40).** D39: the polar
UOD/VOD relaxation block inside `OCEANS` itself, plus its `polevel()` pole-velocity
reconstruction -- float64-exact, first try. D40: `ODHORZ0` (pressure/equation-of-state prep for
the not-yet-ported horizontal pressure-gradient solve) -- caught and fixed a real North Pole
masking bug via the dump comparison itself (`nbyzm` hard-restricts the pole row to a single
longitude, independent of the ordinary depth mask there), pinned down with a dedicated
regression test. Both delivered bitwise/float64-tolerance exact against real Fortran, all 3
dates.

**H. Scoping sweep and the first "new architecture" delta closed (2026-09-29, D41-D42).** D41:
read `ODHORZ` and `OADVTX2` in full and confirmed both are genuine "new architecture"-scale
(comparable to D29's ADI solve); confirmed `OCNQUS.f` (1,846 lines) is entirely dead code for this
rundeck (`USE_QUS=0`); surveyed `OCNKPP.f` (live core `KPPMIX`+`OCONV`, ~2,800 lines, likely the
single largest remaining item) and `OCNMESO_DRV.f` family (confirmed `OCNGISS_TURB.f`/
`OCNGISS_SM.f`, 1,252 lines, entirely dead). D42: committed to and closed `ODHORZ` itself -- the
actual horizontal momentum + mass-continuity solve, decoupled from the `OPFIL2`/`AVR`-file
dependency by recording `OPFIL2`'s outputs directly. Two real bugs (a missing `HOCEAN` recording;
`OPBOT`'s cross-layer accumulation reset incorrectly) caught before any validation run; float64-
tolerance exact on all 15 real call records after fixes. JAX vectorization deliberately deferred
as its own follow-up.

## 5. Results

**Test suite, full project regression** (run after every delta, zero exceptions):

| After delta | Tests passed | Tests failed |
|---|---|---|
| D34 | 426 | 0 |
| D35 | 464 | 0 |
| D36 | 488 | 0 |
| D37 | 505 | 0 |
| D38 | 522 | 0 |
| D39 | 543 | 0 |
| D40 | 560 | 0 |
| D42 | **576** | **0** |

**Validation standard, every delta:** bitwise-exact or float64-rounding-exact (typically 1e-9 to
1e-17 absolute difference, consistent with floating-point operation-order noise, not a real
mismatch) against real Fortran dump records, on all 3 real test dates, for both the plain-Python
reference port and the batched JAX port. Where a real record never exercises a branch (checked
explicitly, never assumed), the branch is instead validated against a synthetic input constructed
to reach it, and this is documented plainly in the delta's own entry — never silently skipped.

## 6. Open items

See `README_START_HERE.md`'s "Open items" section for the full, current list (`ODHORZ`'s JAX port,
`GLMELT`, `OFLUXV`/`OPFIL2`/`ODHORZ`, the `OADVT2` tracer-advection family, `OCNQUS.f`/
`OCNKPP.f`/`OCNMESO_DRV.f`+`OCNTDMIX.f`+`OCNGM.f`/`OSTRAITS.f` — ~11,300 lines entirely unread,
`IRRIG_LK`, the batched outer DYNSI loop, and the absence of a chained whole-model Track B step).

## 7. What changed in code, and where it lives

| Change | Where | Kind |
|---|---|---|
| Dump-hook instrumentation (one pair of Fortran subroutines per delta, e.g. `ffdump_ostres2`/`ffdump_ostres2_geom`) | `fullfidelity/instrumentation/*.f.patch` (diff patches against the pristine `modelE2_planet_2.0` tree; the pristine tree itself is never modified) | Instrumentation only — never changes real model physics |
| Plain-Python reference ports (one per delta, e.g. `ostres2_ff.py`) | `fullfidelity/*_ff.py` | New code: faithful line-by-line Fortran ports |
| Batched JAX ports (one per delta, e.g. `ostres2_jax.py`) | `fullfidelity/*_jax.py` | New code: vectorized, `jax.jit`-compilable equivalents |
| Compare/validation scripts and pytest suites | `fullfidelity/*_compare.py`, `fullfidelity/*_jax_compare.py`, `fullfidelity/tests/test_*_jax.py` | New code: validates every port against real Fortran dumps |
| Permanent pristine-restart archive | `ff_data/_pristine_restarts/` | Operational fix (finding 5) |
| Real Fortran dump data | `ff_data/{nov26,dec01,jan01}/` | Data, not code — generated by rerunning the real, unmodified (except for dump hooks) Fortran executable |

Nothing in the SOCRATES radiation library has been touched, viewed for modification, or ported —
standing constraint, unconditional.

## 8. For discussion / decisions still open

1. **Whether to pursue `OFLUXV`/`OPFIL2`/`ODHORZ`** (the next "new architecture"-scale item,
   comparable in cost to D29's ADI solve) now, or continue with smaller `OCNDYN2.f` deltas first.
2. **Whether `GLMELT` is worth a dedicated day-boundary-crossing test window**, given it's a small
   (72-line) routine whose validation would need new test-date tooling.
3. **Whether/when to scope `OCNQUS.f`/`OCNKPP.f`/`OCNMESO_DRV.f` family** — these are unread and
   could individually be as large as everything done in Stage 2 so far combined; no time estimate
   exists yet, deliberately, per this project's discipline against estimating unread code.

## 9. Where the evidence is

| Question | Document |
|---|---|
| Index of everything, and current state | `README_START_HERE.md` |
| Full delta-by-delta ledger with validation numbers | `../FULL_FIDELITY_DELTAS.md` |
| Scope, line counts, real-vs-dead-code breakdown per file | `../FULL_FIDELITY_PLAN.md` |
| Narrative working log (the debugging stories) | `../fullfidelity/PHASE0_LOG.md` |
| Build/run/instrument recipe, with every operational lesson | `../fullfidelity/instrumentation/build_and_run.md` |
| Track A (earlier, superseded-for-full-fidelity work) | `../STATUS.md` |
| Non-technical summary of lessons learned | `LESSONS_LEARNED.md` |
