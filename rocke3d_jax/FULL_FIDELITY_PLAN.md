# Full-Fidelity Port: Implementation Plan

Branch: `full-fidelity-port` (from `main` @ 62f87a1 + Round 2 working tree).
Written 2026-09-24. Supersedes the "closed, not pursuing" decision in
STATUS.md (2026-09-23): the project goal changed — HQ wants a port whose
answers can be defended against the real Fortran, not a representative
workload. Everything in STATUS.md stays as the **baseline** ("representative
driver"); this branch adds a second, separately-labelled track and records the
**deltas** between them.

## Progress (updated 2026-09-24, end of first autonomous session)

| Item | Status | Evidence |
|---|---|---|
| Phase 0: build/instrument real ModelE, reproducibility, noise floor | **done** | `fullfidelity/PHASE0_LOG.md`, ledger D1–D3 |
| Phase 1.1 ATURB (A-grid + U/V) | **done, F0 rounding-level** | D4 |
| Phase 1.2 PBL `advanc` (surface layer, all 4 tile types) | **done, F0** | D5 |
| Phase 1.3 SURFACE ocean/lake + sea-ice tile fluxes, ice properties | **done, F0** (limiter branches unvalidated) | D6 |
| Phase 1.3b land-ice tile | **done, F0** (dew limit unvalidated) | D7 |
| Phase 1.4 GHY / land (EARTH + `advnc` + snow + Ent canopy conductance) | not started; scoped below | — |
| Phase 1.5 SEAICE/LAKES ground thermodynamics; tile aggregation (`avg_patches_*`) | not started | — |
| Phases 2–5 (radiation, clouds, dynamics, ocean) | not started | — |

**Scoping learned so far.** (a) The pieces ported so far are stateless column/tile functions and validated at
1e-11–1e-16 relative; the method (instrumented real model → per-call records → JAX port → tests with
mutation checks) works and took roughly an hour of effort per ~1k Fortran lines. (b) The land model is a
different animal: `giss_LSM/GHY.f` (4.6k lines) keeps its state in module-level variables (soil water/heat
in 6 layers × bare/vegetated, 5 snow layers, canopy), is driven by Ent vegetation (26.5k lines of library; only
the canopy conductance/photosynthesis path is needed), and needs the multi-layer land state from the restart.
Expect it to be the longest single item in Phase 1. (c) Moist convection + large-scale condensation
(`CLOUDS2.F90`: MSTCNV 2.7k, LSCOND 2.0k lines, heavily branching) is next in size. (d) Track B's current
pieces are unoptimized (CPU float64 ≈ 0.4 s per DTsrc step for PBL+ATURB alone, ledger D8): performance work
comes only after each rung's correctness.

## 1. Definition of "full fidelity" (decide this first, in writing)

Fidelity is a ladder, not a switch. Each rung is a claim we can test:

| Rung | Claim | Test |
|---|---|---|
| F0 | Module-level: each ported routine matches Fortran on captured inputs | Per-routine input/output dumps, tolerance table |
| F1 | Column-physics step: SURFACE→GROUND→PBL/ATURB→RADIA→CLOUDS→DRYCNV chain matches Fortran for 1 DTsrc from the real restart | Field-by-field diff of the state after one step |
| F2 | Multi-step (e.g. 1 day = 48 steps) with dynamics: statistically indistinguishable from Fortran | Global means, zonal means, RMSE vs. Fortran output; chaos-aware (ensemble spread, not bitwise) |
| F3 | Coupled (ocean + sea ice) run: monthly `acc` diagnostics match the real P2SAoM40 `*.accP2SAoM40.nc` | Compare to the 12 monthly acc files already on disk |

**Proposed target: F2 for the atmosphere, with F1 as the hard gate.** F3
(ocean GCM) is scoped but recommended as a stretch/decision point (section 6).
Bitwise equality is *not* a goal: float32/float64 and operation order differ,
and the atmosphere is chaotic (Round 2 already showed marginal-cell branch
flips). Success criterion for each rung must be stated as tolerances agreed
before coding, not fitted afterwards.

## 2. Scope: what "full" means for P2SAoM40 (measured, from the rundeck)

Line counts of the real source in `modelE2_planet_2.0/model/` (Fortran only):

| Area | Files | Lines | Current JAX state |
|---|---|---|---|
| Radiation | RADIATION.f, RAD_DRV.f, RAD_UTILS.f, RAD_COM.f | ~19,500 | graybody stand-in |
| Radiation library (SOCRATES) | `model/socrates` + `ModelE_Support/socrates` | ~90,000 (F90, external lib) | none |
| Moist convection + large-scale cloud | CLOUDS2_DRV.F90, CLOUDS2.F90 (+COM) | ≥3,800 (CLOUDS2.F90 not yet counted) | none (out of scope so far) |
| Land surface / snow | GHY_DRV.f (+giss_LSM), VEG/ENT | 4,939+ | placeholder |
| PBL / turbulence | PBL.f, PBL_DRV.f, ATURB.f | 6,762 | PBL faithful kernels; ATURB partial; driver uses shortcut |
| Sea ice | SEAICE.f, SEAICE_DRV.f | 5,394 | placeholder |
| Lakes | LAKES.f | 4,076 | placeholder |
| Land ice | LANDICE*.f | 2,073 | none |
| Surface/fluxes | SURFACE.f, FLUXES.f | 5,123 | partial (single-type dispatch) |
| Dry convection | DRYCNV.f | 551 | faithful |
| Atmospheric dynamics | ATMDYN.f, MOMEN2ND.f, QUS_DRV/QUS3D, STRATDYN | ~8,500 | none |
| Ice dynamics | ICEDYN.f, ICEDYN_DRV.f | 3,682 | none |
| Ocean GCM | OCNDYN/OCNDYN2/OCNKPP/… | ≥12,500 | none |

Excluding the SOCRATES library and the ocean, ≈ 55–60k lines of Fortran are in
play; with them, well over 150k. This is a multi-month effort for one
person; the plan below is ordered so that each phase produces a
*measurable fidelity delta* and can be the stopping point.

> **2026-09-24 finding (see `fullfidelity/PHASE0_LOG.md`):** the real
> P2SAoM40 executable contains no DRYCNV — free-atmosphere mixing is ATURB
> (`atm_diffus`). Track A's DRYCNV work benchmarks a routine the real
> configuration does not run. Phase 1 must therefore make **ATURB** the first
> port target. Also confirmed: the real model is bitwise reproducible here
> (5-day re-run == original restart, byte-identical).

## 3. Method (what makes this different from the previous port)

The earlier ports were validated against self-written "Fortran-like" test
programs, and GHY's validation turned out to be rigged. The new method
removes that failure mode:

1. **Real-Fortran oracle only.** Every routine is validated against dumps
   from the *actual* ModelE executable, never a hand-written stand-in.
   Phase 0 builds an instrumented ModelE (write inputs/outputs of each
   routine to NetCDF at chosen steps) from the existing build tree
   (`P2SAoM40.mk`, `.o` files already present).
2. **Validation cannot be vacuous.** Every test asserts that outputs are
   non-trivial (not all zeros/constants), compares *all* output fields, and
   is mutation-checked (perturb the port; the test must fail).
3. **Port structure.** Layer-first (L,J,I) arrays (= Fortran memory order);
   per-column physics as pure functions vmapped over the grid; loops with
   dependencies as `lax.scan`; branches as `jnp.where` with the dead-branch
   NaN caveat handled explicitly. Device-resident chained stepping from day
   one (Round 2 lesson), so performance work is not a rewrite later.
4. **Faithfulness first, speed second.** No fused/optimized variant until the
   faithful version passes its rung. Keep the faithful version as the
   reference for later optimization (same as this project's Phase 1/Round 2).
5. **Delta capture is built in** (section 5), not a report written at the end.

## 4. Phases

Each phase ends with: tests green, a delta record appended, STATUS-style
write-up, and a go/no-go note. Effort figures are rough single-engineer
estimates and the least reliable part of this plan.

### Phase 0 — Oracle & harness (2–3 weeks) — HARD GATE
- Confirm the ModelE build is reproducible here (compiler, NetCDF, MPI
  stubs); run the 194-step reference or a short (e.g. 48-step) segment from
  `1JAN1950.rsfP2SAoM40.nc` and confirm it reproduces `P2SAoM40.PRT`.
- Add a small dump facility (Fortran `call dump_state(name, arrays)`)
  around: SURFACE, GHY, PBL/ATURB, RADIA, CONDSE, DRYCNV, and the full
  post-step state. Store as NetCDF under a versioned oracle directory
  (not committed if large; commit a manifest with checksums).
- Build the comparison harness: field-by-field metrics (max abs, RMS, rel,
  correlation, location of max), tolerance table per field, machine-readable
  output (JSON) so deltas can be diffed run-to-run.
- Establish the **noise floor**: run Fortran twice with a 1-ulp perturbation
  to measure intrinsic divergence; port tolerances are set relative to it
  (this reuses the Round 2 method).
- **Risk:** if the model cannot be rebuilt/instrumented here, everything
  downstream validates only against restart/acc files (much weaker). The
  fallback and its consequences must be written down, not discovered late.

### Phase 1 — Land surface & PBL fidelity (5–7 weeks) — closes the known placeholders
1. **ATURB** real closure constants, `find_pbl_top`, tridiagonal, wired into
   the driver replacing the layer-1 shortcut (1.4k lines). Validate F0 on
   dumps.
2. **PBL.f/PBL_DRV.f** Newton solve replacing the fixed-point iteration
   (a documented "Simplified" row today).
3. **SURFACE/FLUXES** full area-weighted sub-tiling over ocean / sea ice /
   land ice / land fractions (removes "single dominant itype").
4. **GHY_DRV** (4.9k lines, multi-layer soil + snow, needs its own
   orchestration and the multi-layer restart state already located in
   `1JAN1950.rsfP2SAoM40.nc`). Includes evaporation, runoff, soil thermal
   properties, snow melt — all placeholders today.
5. **SEAICE(+DRV)** thermodynamics (5.4k), **LAKES** (4k), **LANDICE** (2k).
- Gate F0 per routine, then F1 for the surface/PBL sub-chain.
- Expected delta: correct polar/land fluxes; the current sensible-heat
  pattern correlation vs. period-mean (−0.014, one-step vs. mean — not a
  like-for-like check) should be re-measured properly against the oracle.

### Phase 2 — Radiation (6–10 weeks + a decision) — biggest single item
RADIA is 65.6% of Fortran runtime, and the graybody stand-in is the most
visible fidelity gap. Options, to be decided at Phase 0 from measurements:
- **2a. Call SOCRATES from Python** (ctypes/f2py wrapper of `libsocrates.a`)
  as the "real" radiation, keeping everything else JAX. Fast to reach F1;
  radiation is then *not* JAX-accelerated (honest: reported as such), and
  breaks the "pure Python" claim for that component.
- **2b. Port RAD_DRV/RADIATION/RAD_UTILS (19.5k) to JAX and use SOCRATES
  gas/aerosol tables as data.** Feasible only if the P2SAoM40 config uses the
  parts of SOCRATES reachable through ModelE's interface; needs code-path
  analysis first (most of the 90k lines are unused options).
- **2c. Port the needed SOCRATES two-stream/correlated-k kernels to JAX.**
  Largest effort, best end state (GPU-resident radiation); do only after 2a
  proves the interface and gives an oracle for 2c.
- Recommendation: **2a first** (weeks, gives F1 immediately and an oracle),
  **then 2c for the code paths actually exercised** if HQ needs an
  all-Python/GPU claim. NRAD=5 gating and the bimodal cost profile must be
  preserved.

### Phase 3 — Clouds / moist convection (4–6 weeks)
CLOUDS2/CLOUDS2_DRV (moist convection, large-scale condensation, cloud
fraction feeding radiation). Required for F1 to mean anything — radiation and
surface fluxes depend on it. Port with `lax.scan` over levels and per-column
plume logic; watch branch-heavy code (`where` evaluates all branches).

### Phase 4 — Atmospheric dynamics (8–12 weeks) — needed for F2
ATMDYN (advection, pressure gradient, coriolis, filters), MOMEN2ND, QUS
(second-order-moment advection of T/Q/tracers), STRATDYN. Global stencils on
the lat-lon grid with polar treatment and FFT filtering (FFT72). This is the
first phase with genuine communication/stencil structure; decide sharding
strategy (single-GPU is enough at 72×46×40) only after profiling.
- Gate F2: 1-day run vs. Fortran, chaos-aware metrics.

### Phase 5 — Stretch: ice dynamics and ocean (decision point)
ICEDYN (3.7k) and the ocean GCM (≥12.5k: OCNDYN, KPP, GM, straits, …) are
required for F3. Recommend **not committing now**; decide after Phase 4 based
on what HQ's skepticism actually needs. A cheaper F3-lite is prescribed-SST /
data-ocean from the restart.

## 5. Capturing deltas (the requirement that findings "complement" existing ones)

- **Tracks, not overwrites.** Existing results are labelled *Track A:
  representative driver* (tag = current state). New results are *Track B:
  full-fidelity*. STATUS.md gets a new "Full-fidelity port" section; Track A
  text is not edited except to link.
- **Delta ledger** (`FULL_FIDELITY_DELTAS.md` + machine-readable
  `deltas/*.json`), one row per phase/module, recording for Track A vs. Track
  B vs. Fortran oracle: accuracy metrics (max/RMS/correlation per field),
  step time (CPU, GPU-single-call, GPU-chained), lines of code ported, and
  the reason for any change. Generated by the Phase 0 harness so it is
  reproducible, not hand-typed.
- **Performance expectations to record honestly:** faithful physics is more
  expensive than the stand-ins; the 356× / 5.8× figures are for the simplified
  chain and **will drop**. The delta ledger shows by how much, and the headline
  for HQ is speed *at full fidelity*, measured the same way (single-call and
  chained, CPU and GPU).
- **Deck:** extend `status_slides/` with a Track A vs. Track B slide per
  phase; reuse the verification-slide pattern (reproducibility, independent
  plausibility check, scope caveats).
- Git: one branch, small commits per module with the tests; tag at each rung
  (`ff-F0-…`, `ff-F1`, …).

## 6. Risks and decisions

| Risk | Consequence | Mitigation |
|---|---|---|
| Cannot instrument/rebuild Fortran | Only restart/acc validation → weak fidelity claim | Resolve in Phase 0 before any porting; state fallback explicitly |
| SOCRATES dependency (90k lines) | Radiation dominates effort | 2a → 2c staging; measure exercised code paths |
| Chaos makes multi-step comparison ambiguous | Cannot prove "match" beyond F1 | Noise-floor + ensemble metrics agreed up front |
| Scale (>55k lines excluding radiation lib/ocean) | Schedule | Phase gates, each independently valuable; stop at F1 if needed |
| Performance regression vs. Track A | HQ sees headline speedup vanish | Set expectation now; report fidelity-vs-speed ledger |
| Placeholder/rigged validation recurring | False confidence | Rules in Method §3.1–3.2; mutation-check every test |
| JAX limitations (NaN in dead `where` branches, long unrolled chains) | Bugs / slowness | Use `scan`, safe-branch pattern, follow Round 2 findings |

**Decisions needed from HQ/user (not blocking Phase 0):**
1. Target rung: F1 hard gate + F2 (recommended), or F3.
2. Radiation: is calling SOCRATES from Python (2a) acceptable, or must radiation be pure JAX (2c)?
3. Is ocean GCM in scope (Phase 5)?

## 7. Immediate next steps (Phase 0, first two weeks)
1. Verify ModelE build/run reproducibility from `P2SAoM40.mk` and the
   restart; record the run recipe in this directory.
2. Write the dump-instrumentation patch and the oracle manifest format.
3. Implement the comparison harness + noise-floor measurement on the
   *existing* Track A driver so the first delta row (Track A vs. real Fortran
   at F1) exists before any new physics is ported.
4. Commit the Round 2 work on this branch (currently uncommitted in the
   working tree) so Track A is fully captured at the branch point.

## 8. Confidence notes
Line counts are from `wc -l` of the ModelE tree on 2026-09-24 (CLOUDS2.F90,
the SOCRATES source count and OCN files not all measured; ocean is a lower
bound). Effort estimates are judgment, not measured. The claim that F1/F2
are achievable without bitwise agreement is a methodological position to
confirm with HQ, not a result.
