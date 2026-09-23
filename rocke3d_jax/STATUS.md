# ROCKE-3D → JAX: Status

**As of**: 2026-09-22. **This is now the single source of truth** for this
project's status — `EXECUTIVE_SUMMARY.md`, `FINDINGS.md`, `PORTING_STATUS.md`,
`SESSION_SUMMARY.md`, and `SUMMARY.md` have been retired and their content
merged in here (or explicitly dropped where superseded — see inline notes).
Anything not carried forward from those files was either duplicated here
already, or a stale/unmeasured estimate this document replaces with a real
number.

**Companion slide deck**: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA
("ROCKE-3D → JAX: Status" — bottom line, performance, recommendation, output
maps; filesystem snapshot in `status_slides/`). This is also now the *only*
slide deck for this project — an earlier, narrower deck (kernel-level timing
only) has been merged into it and retired.

**Repository layout**: this directory was reorganized on 2026-09-22 —
historical/superseded scripts, notebooks, and scratch data (old benchmark
scripts, superseded compare tooling, orphaned data files, old dashboard
notebooks) were moved into `mantle/` (see `mantle/README.md` for the full
list and why each one moved), verified first against a full grep-based
dependency map so nothing live was broken. Everything referenced in this
document lives at the top level unless a path explicitly says `mantle/...`.

## Background: what P2SAoM40 is

`P2SAoM40` is one rundeck configuration of **ROCKE-3D 2.0**
(`ModelE2_planet_2.0`), NASA GISS's Fortran GCM, from the 36-configuration
template family named `P2{S|G}{A|x|N}{p|q|o}{M40|F40}` (star type × radiation
scheme × ocean type × grid). `P2SAoM40` specifically: **S**un-like star,
**A**tmosphere with **S**OCRATES spectral radiation, **o**cean = dynamic
ocean, **M40** = 40-layer grid at 72×46 horizontal resolution. The
configuration set and its reference output are archived at Zenodo
(10.5281/zenodo.14721184), alongside the paper describing ROCKE-3D 2.0
(Tsigaridis et al. 2025, *Geoscientific Model Development*). This project
does not reproduce the paper's own GISS-vs-SOCRATES comparison (e.g. cloud
fraction ~58% vs ~29%, ~100 yr equilibration, 1000+ yr dynamic-ocean spin-up)
— it uses a real, independently-run P2SAoM40 restart as **input data** for
the Fortran-vs-JAX comparison below, nothing more.

## The goal

Use an AI coding agent to autonomously convert ROCKE-3D (NASA GISS's Fortran
GCM) to Python/JAX, and collect real benchmarks — **accuracy**, **CPU and GPU
performance**, **conversion cost**, and **gotchas** — validated against **real
production restart data** from an actual completed run (`P2SAoM40`). The
target comparison has always been **JAX vs. the real Fortran model**, not JAX
vs. a second Python reimplementation.

## Bottom line

| Question | Answer |
|---|---|
| Does the JAX port match real Fortran numerically? | **Yes**, for the modules tested directly against it (PBL, DRYCNV): floating-point-level agreement (1e-9–1e-3) on both CPU and GPU. |
| Is JAX faster than real Fortran? | **Depends on device and module.** See Performance below — not a blanket yes. |
| Has this been checked on GPU? | **Yes, both scopes.** Kernels (DRYCNV, PBL) in isolation: real 2.3–6.3× speedup, run 2026-09-20. Full chained physics group, real NVIDIA A100 (discover cluster): **4.2× GPU speedup**, measured 2026-09-22 after two fixes — a dispatch-fusion optimization (Phase 1) and a real bug in how the comparison itself was measured. See Full Physics Chain below; this replaces an earlier ~1.0× figure that turned out to be invalid, not a real physical limit. |
| Is the port complete? | **14 of 17 modules** are faithful ports, validated against real Fortran. 3 (SEAICE core thermodynamics, LAKES mixing, part of ATURB) are documented placeholders. See Recommendation below. |
| Does this project still use NumPy comparisons? | It has some (`mantle/benchmark_all.py` and related) — **dropped from the headline story**. See below. |

## Full Physics Chain: the finding that matters most

This is the single most important correction in this project's history, so it
gets its own section rather than being a line in a table. It has now been
corrected *twice* — once for an unmeasured GPU estimate (below), and again
(2026-09-22) for a real bug in how the measurement itself was taken. Both
corrections are kept visible here rather than quietly overwritten.

**Current, validated result** (2026-09-22, real NVIDIA A100, discover
cluster, via `p2saom40_compare.py` post-Phase-1-fusion — see "GPU
optimization" below): the chained PBL+SURFACE+GROUND physics group, one real
`DTsrc` step (1800 s model time), from two separate runs (one with
`JAX_PLATFORMS=cpu` forced, one natural GPU detection):

| Leg | Time / step | vs. real Fortran (264.0 ms) |
|---|---|---|
| Real Fortran (CPU) | 264.0 ms | — |
| JAX (genuine CPU) | 17.94 ms | 14.7× faster |
| JAX (GPU, A100) | 4.26 ms | 62.0× faster |

**GPU vs. genuine CPU: 4.2× — clears the >2× optimization target.** This
came from two changes, not one:

1. **Dispatch fusion (Phase 1)**: the full-chain driver was restructured
   from ~33 separately-dispatched JIT calls per step (several nested in
   Python loops) into a single fused `jax.jit` computation. Same formulas,
   same iteration order — see "GPU optimization: Phase 1" below for the
   profiling and the fix.
2. **A real bug in the measurement script, found while trying to verify
   Phase 1 on GPU**: `p2saom40_compare.py`'s "Performance (CPU)" section
   never actually forced the CPU backend — it just timed `run_dtsrc_step()`
   on whatever `jax.devices()[0]` already was. Run on a GPU node without
   explicitly forcing `JAX_PLATFORMS=cpu` (the normal way to *also* get a
   real GPU number in the same run), that "CPU" section silently ran on the
   GPU too. **The original ~1.0× "no GPU benefit" figure was comparing the
   GPU to itself, not to a real CPU baseline.** It has been fixed: the
   script now spawns a genuine CPU-only subprocess for that number whenever
   a GPU is already active in the main process, so a single run produces
   two real numbers instead of one real number and one mislabeled copy of
   itself.

**What this means for the old ~1.0× conclusion**: it wasn't wrong because
the grid was too small for GPU to help (the explanation given at the time)
— it was wrong because it was never actually a CPU-vs-GPU comparison. The
kernel-level GPU results (2.3–6.3×, unaffected by this bug — those come from
genuinely separate CPU/GPU script invocations) were the more reliable signal
all along, and the corrected full-chain number is now consistent with them.

**Why this replaces every earlier ~20–30× GPU figure** (kept for the
historical record, since that correction is now itself superseded by a
better number rather than retracted): that number was an *estimate* made
before any GPU was available in this environment, and was replaced on
2026-09-22 by the (at-the-time invalid) ~1.0× measurement. It is now
replaced again by the validated 4.2× figure above. Removed throughout this
project's documentation each time, not just flagged — it previously
appeared in `EXECUTIVE_SUMMARY.md`'s Business Impact, ROI, and Timeline
sections and in `README_GPU.md`'s setup guide (that file now points here).

**Scope of what "full physics chain" actually covers** — this matters for any
future ROI or timeline claim: of the real Fortran model's per-`DTsrc`-step
cost, JAX has *some* form of ~74.6% (the columnar physics: DRYCNV, PBL,
RADIA, SURFACE, GROUND); ~18.4% is atmospheric dynamics + moist convection
(`CONDSE`), **entirely out of scope for this port, never ported, never
timed**; the remainder is sea-ice melt and diagnostics. Real Fortran `RADIA`
alone is **65.57%** of total per-step runtime (avg. 2070 ms/step, bimodal:
~0.7 ms on held-value steps between radiation calls, ~10.8 s on the
radiation-gated steps) — but JAX's radiation is a simplified graybody
stand-in, not real SOCRATES spectral transfer, so it is **excluded from both
legs** of the 264/17.94/4.26 ms comparison above to keep it a fair speed
comparison. **Practical consequence: there is currently no honest way to
project a whole-model runtime or dollar-cost from this result** — real
spectral radiation and real atmospheric dynamics would both need to be ported
and timed first. Treat the numbers above as a components-level result, not a
whole-model ROI figure.

## Accuracy

Two different validations exist; both check against **real Fortran**, not a
NumPy stand-in:

| What | Method | Result |
|---|---|---|
| PBL + DRYCNV kernels | Real inputs shared byte-for-byte between a from-scratch `ifort`-compiled Fortran reference and JAX, at P2SAoM40's real grid size (3,312 points × 40 layers) | Max diff ≤1.5e-3 (CPU), ≤2.8e-3 (GPU) against output scales of 10¹–10³ — floating-point-level, not algorithmic |
| Full physics chain (PBL+radiation+surface+ground), restart snapshot 1949-12-01 | Driven by P2SAoM40's actual restart state (`fort.1.nc`), one real timestep | **0.987** spatial correlation vs. the real run's period-mean surface temperature (+1.0°C bias) |
| Same, later restart snapshot 1950-11-26 (2026-09-22, GPU-capable run, reconfirmed after Phase 1 fusion) | Same method, ~11 simulated months further into the same P2SAoM40 run (which has since completed its full 1-year integration) | **0.965** spatial correlation (−1.86°C bias) — consistent with the earlier snapshot, i.e. fidelity holds up over a full year of integration, not just the first month, and unchanged by the dispatch-fusion optimization |

Both full-chain rows are a coarser sanity check than the kernel row above —
period-mean vs. single-step.

**Per-field max difference** (kernel comparison, CPU vs GPU — the detail
behind the ≤1.5e-3 / ≤2.8e-3 summary above), against shared inputs, seed
12345, 100 iterations/leg:

| Field | Max diff (CPU) | Max diff (GPU) |
|---|---|---|
| drycnv T | 8.79e-05 | 9.98e-05 |
| drycnv Q | 3.70e-09 | 3.81e-09 |
| pbl u | 4.41e-04 | 1.02e-03 |
| pbl t | 1.46e-03 | 2.77e-03 |
| pbl q | 1.19e-04 | 3.24e-04 |
| pbl dpsim | 1.11e-04 | 1.09e-03 |
| pbl dpsih | 8.14e-04 | 1.14e-03 |
| pbl dpsiq | 1.26e-03 | 1.09e-03 |

GPU diffs are consistently slightly larger than CPU, consistent with a
different floating-point reduction order on the GPU, not an algorithmic
difference.

**A validation-methodology gotcha worth knowing**: this project's earlier
"Fortran-like" NumPy reference for PBL (`simil_numpy`) silently omitted half
of the real algorithm's branches — before the fix, this produced apparent max
diffs as large as **~1.26×10⁴** against the real Fortran reference. Re-deriving
the Fortran side directly from the real three-branch formula (not the NumPy
stand-in) is what produced the clean 1.5e-3 agreement above. Any older
"~2.3%"/"1e-6" accuracy claims from earlier project documentation used this
flawed reference and are superseded by this result for PBL and DRYCNV
specifically — they are not carried forward into this document.

**Spatial view of the kernel result**: `visualize_p2saom40_kernel_maps.ipynb`
projects the JAX−Fortran differences above onto the real P2SAoM40 72×46 grid
(map images in `outputs/p2saom40_kernel_*_diff_*.html`, summarized in
`status_slides/images/diff_grid.png`). Confirms the same floating-point-level
agreement spatially, and shows a handful of outlier grid cells in
`pbl.u`/`dpsih`/`dpsiq` — both Fortran and JAX reproduce the identical
outliers (traced to near-zero Monin-Obukhov length at those cells), so
they're still the same floating-point-level diffs, not a bug.

## Performance (real Fortran vs. JAX)

| Comparison | Fortran (CPU) | JAX (CPU) | JAX (GPU) |
|---|---|---|---|
| DRYCNV kernel | 1.27 ms | 17.1 ms — **13.4× slower** | 0.56 ms — **2.3× faster than Fortran-CPU** |
| PBL kernel | 0.37 ms | 0.20 ms — 1.9× faster | 0.06 ms — **6.3× faster than Fortran-CPU** |
| Full physics chain (PBL+radiation+surface+ground), post-Phase-1-fusion | ~264 ms/step (radiation excluded from JAX legs — see Full Physics Chain above) | 17.94 ms/step — **14.7× faster** (genuine CPU-only measurement) | 4.26 ms/step — **4.2× faster than JAX-CPU**, 62.0× vs. Fortran-CPU |

**Reading the kernel table honestly**: JAX-CPU is *not* uniformly faster than
Fortran — it loses DRYCNV outright on CPU (small-array dispatch overhead).
GPU wins decisively on both isolated kernels *and*, since the Phase 1
dispatch fusion, on the full chained pipeline as well — see Full Physics
Chain above for what changed and why the number used to look flat.

## What's simplified in the full-chain driver (honesty section)

The kernel-level accuracy numbers above (PBL, DRYCNV) come from a rigorous,
faithful port validated line-for-line against Fortran. The **full-chain
driver** (`p2saom40_driver.py`) used for the 0.965–0.987 correlation and the
17.94/4.26 ms timing numbers is not uniformly that rigorous — several
sub-components inside it are simplified relative to the real Fortran:

- **Monin-Obukhov length solve**: the full-chain driver uses a fixed-point
  iteration; real `PBL.f` uses a Newton solve. The standalone kernel
  comparison above (which produced the clean 1.5e-3 accuracy result) uses the
  real Newton-equivalent three-branch formula — this simplification is
  specific to the full-chain driver, not the validated kernel.
- **Surface-type dispatch**: one dominant surface type per grid cell, no
  sub-tile weighting (real Fortran blends land/ice/ocean fractions within a
  cell).
- **GHY** (ground hydrology): soil-moisture functions exist in `ghy_jax.py`
  but are not wired into the full-chain driver.
- **Layer-1 turbulence**: the driver applies a direct flux-tendency
  shortcut, bypassing `aturb_jax.py` entirely.
- **Radiation**: simplified graybody stand-in (already noted above).

None of this invalidates the 0.965–0.987 correlation result — it's still a
real, meaningful check against real restart data — but it means that result
is a **coarser sanity check of the whole pipeline**, not proof that every
individual sub-component matches Fortran to the same 1e-3 standard as the
validated PBL/DRYCNV kernels.

## Recommendation: the 3 remaining modules

**Why they're still placeholders, not a gap that was missed**: the original
port deliberately left SEAICE/LAKES/ATURB as documented stand-ins (e.g.
LAKES' `lkmix` is a no-op) to reach interface-complete (17/17) first,
prioritizing full validation of the columnar physics (PBL, DRYCNV) instead.
A scoping choice, made explicit here rather than discovered later.

**SEAICE** (core thermodynamics) and **ATURB** (PBL-top-finding) — recommend
porting next. Reasoning: sea ice is a standard, physically active component
of any ROCKE-3D ocean-coupled run including P2SAoM40, not an optional
add-on, and it dominates the surface energy balance specifically in polar
regions — exactly where the current 0.987 *global* correlation check is
least able to reveal a local error. ATURB's placeholder affects
boundary-layer depth, which feeds surface flux accuracy broadly, not just at
the poles. Both are also useful precisely *because* they're harder than
PBL/DRYCNV: the stated goal includes collecting "gotchas," and the
easy columnar-physics modules have already told us most of what they can
about where the vmap-over-grid-cells approach works cleanly.

**LAKES** (`lkmix`) — recommend leaving as placeholder for now. Lakes cover a
much smaller fraction of Earth's surface than sea ice, so the accuracy payoff
is lower for the effort; revisit only if a specific science need (a
lake-focused study) requires it.

*(This SEAICE/lakes area-of-impact reasoning is domain judgment, not a number
measured in this project's own data — flagging that distinction rather than
presenting it as measured fact.)*

### Reference: all 17 modules

| Module | JAX file | Real-Fortran validation |
|---|---|---|
| DRYCNV | `drycnv.py` | Direct, kernel-level (see Accuracy) |
| PBL | `pbl.py` | Direct, kernel-level (see Accuracy) |
| GHY | `ghy_jax.py` | Direct (1e-6-level, standalone), **not wired into the full-chain driver** |
| FLUXES | `fluxes_jax.py` | Direct — see wind-speed convention gotcha below |
| SURFACE | `surface_jax.py` | Direct, but full-chain driver uses simplified single-type dispatch |
| RADIATION | `radiation_jax.py` | Simplified graybody stand-in, not real SOCRATES spectral transfer |
| ATM_COM | `atm_com_jax.py` | Supporting state module |
| CONSTANT | `constant_jax.py` | Supporting constants module |
| GEOM | `geom_jax.py` | Supporting grid-geometry module |
| PBL_COM | `pbl_com_jax.py` | Supporting state module |
| RAD_COM | `rad_com_jax.py` | Supporting state module |
| PBL_SIMPLE | `pbl_simple_jax.py` | Simplified variant, not the primary validated path |
| SEAICE_COM | `seaice_com_jax.py` | Supporting state module (interface only) |
| LAKES_COM | `lakes_com_jax.py` | Supporting state module (interface only) |
| SEAICE | `seaice_jax.py` | **Placeholder** — not physically complete, recommended next |
| LAKES | `lakes_jax.py` | **Placeholder** (`lkmix` no-op) — recommended to defer |
| ATURB | `aturb_jax.py` | **Placeholder** for PBL-top-finding; bypassed by full-chain driver's layer-1 shortcut — recommended next |

## On NumPy: dropped from the headline story

`mantle/benchmark_all.py`/`mantle/benchmark_all_cpu.py` compare JAX against
`simil_numpy`/`dry_convection_numpy` — hand-written, un-jitted Python
reimplementations of the same formulas, useful only for answering "how much
does JIT compilation alone buy you, holding the algorithm constant." That's a
real but narrow engineering question, and it's not what management or the
project goal actually needs to know, which is JAX vs. **the real Fortran
model** run in production. Given `simil_numpy` was also found to be *wrong*
(see Accuracy above), it's a weak reference on top of being the wrong
comparison. The files are kept (harmless as engineering scratch work, moved
into `mantle/` in the 2026-09-22 reorg) but NumPy-relative speedups are no
longer cited anywhere in this project's status reporting — every headline
number is JAX vs. Fortran.

## Conversion cost & gotchas

**Cost**: no independently-verified dollar/hour figure exists for this
project as a whole. The only cost log that ever existed
(`SESSION_SUMMARY.md`, generated 2026-08-27) covered **10 sessions, ~220
total turns, ~11.5 hours, ~$11.50** — self-reported and unaudited, and it
**predates** (by nearly a month) the real Fortran P2SAoM40 build, the real
GPU comparison work, and the full-chain driver work described throughout
this document. Treat it as a partial early-phase data point, not a total
project cost — no attempt has been made to extend it to cover the later,
larger effort, since a self-reported per-session estimate isn't a
meaningful way to account for a multi-day real GCM build.

**Size**: ~5,000+ lines of code across the JAX modules and their Fortran test
drivers combined (rough count, not independently audited).

Four real gotchas hit during this conversion, each with a traceable fix:

1. **Wind-speed convention bug** (`FLUXES`/`SURFACE`): computed wind speed
   from the surface-relative wind alone; with the physically correct
   `us=vs=0` no-slip value for land/ice, this silently zeroed every flux for
   those cells. Not caught by unit tests (they used arbitrary nonzero test
   values) — only surfaced when driven with real P2SAoM40 data.
2. **Incomplete reference implementation** (`simil_numpy`): missing the
   stable branch of `find_dpsih`, wrong `dpsiq` shortcut — produced apparent
   diffs up to ~1.26e4 before the fix (see Accuracy above). A reminder that a
   hand-written "reference" needs the same validation rigor as the port
   itself.
3. **A `numpy.ndarray.tofile()` ordering footgun**: it always writes C-order
   bytes regardless of the array's actual memory layout, which silently
   corrupted shared test inputs between the Fortran and JAX legs of the
   kernel comparison above until caught by an input round-trip check. Fixed
   in `compare_generate_inputs.py` via `arr.tobytes(order='F')`.
4. **Unit tests validate placeholder logic against itself**: this is why
   SEAICE/LAKES/ATURB's placeholders passed their own tests despite not
   being physically complete — a real methodology gap, not just a code gap.

**General pattern**: columnar physics (each grid cell independent — PBL,
DRYCNV, RADIATION-as-implemented) ports cleanly onto JAX's vectorized
execution model. The genuinely out-of-scope pieces (atmospheric dynamics,
moist convection, the ocean GCM) are structurally harder because they need
values from *neighboring* grid cells, which doesn't map onto the same
vectorization approach without real additional work. (No independently
audited LOC or cross-cell-communication count exists for this out-of-scope
code — an earlier version of this document cited one that didn't actually
exist anywhere in the project's files; that citation has been removed.)

## GPU setup

Docker/bare-metal/Slurm setup instructions and troubleshooting live in
`README_GPU.md` (kept — it's operational reference, not status narrative).
Its performance-expectation numbers have been corrected to point here rather
than repeat the retired ~20–30× estimate.

## Regression verification (2026-09-22, post-`mantle/` reorg)

Everything in this document was re-verified to still run and reproduce its
numbers after the `mantle/` reorg, on this machine (CPU only — no GPU node
available in this session, so the GPU figures above were **not**
re-measured; they still reflect the 2026-09-22 A100 run cited throughout):

- **104/104 unit tests pass** (`test_*_jax.py` + `tests/`).
- **PBL + DRYCNV kernel comparison** (`compare_generate_inputs.py` →
  `compare_run_fortran.py` → `compare_jax.py` → `compare_report.py`,
  freshly recompiled with `ifort -O2`): accuracy numbers reproduced
  **exactly** (max diffs identical to the per-field table above). CPU timing
  came out somewhat different this run (DRYCNV Fortran 1.56 ms / JAX 14.1 ms;
  PBL Fortran 0.42 ms / JAX 0.34 ms) — same qualitative pattern (Fortran
  wins DRYCNV, JAX wins PBL), different absolute numbers, consistent with
  this being a third physical node/run and not a discrepancy.
- **The other 6 modules' direct Fortran comparisons** (FLUXES, SURFACE,
  RADIATION, GHY, SEAICE, LAKES): recompiled each `test_*_fortran.f` fresh
  and re-ran the comparison (via `mantle/compare_fortran_jax.py`, invoked
  with `PYTHONPATH` set to the top level so its cross-directory imports
  resolve) — all 6 still pass. Caught and fixed a real mistake in the
  process: naively redirecting each binary's stdout (`./test_x_fortran >
  test_x_fortran_output.txt`) clobbers the file, because these programs
  write their real output via an internal `OPEN` statement, not stdout —
  fixed by running them without redirection.
- **`visualize_p2saom40_kernel_maps.ipynb`** re-executed end to end
  (`jupyter nbconvert --execute`, kernel `python3`, `JAX_PLATFORMS=cpu`):
  zero error cells, all 40 output maps regenerated, PBL per-field diffs match
  the table above exactly; DRYCNV's printed diffs are smaller than the table
  (e.g. 7.83e-05 vs. 8.79e-05 for T) because the notebook prints the
  surface-layer-0 slice only, while the table's number is the max over all
  40 layers — expected, not a discrepancy (the notebook's own intro cell
  already says this).
- **Full physics-chain driver** (`p2saom40_compare.py`, real restart data,
  same `fort.1.nc`/`PARTIAL.accP2SAoM40.nc` used throughout this project):
  reproduced the 0.965 correlation / −1.86°C bias, the itype counts, and the
  74.6%/9.91%/8.50% cost breakdown exactly. CPU wall time came out at 59.56
  ms/step on this node — a third data point alongside the 48 ms and 33.5 ms
  already discussed above, reinforcing (not contradicting) the point that
  this number is node-dependent and the ratios are what matter.
- Fixed one more latent bug found in the process: `status_slides/images/
  render_diff.py` had a hardcoded path into a previous session's temporary
  scratch directory as its output location — harmless while that scratch
  dir existed, but would have failed the next time anyone tried to
  regenerate the slide deck's difference-map image. Now writes next to
  itself.

## GPU optimization: Phase 1 (fused dispatch) — done, target cleared

**Goal**: raise the full physics chain's GPU-vs-CPU ratio from ~1.0× toward
>2×, without moving accuracy outside the tolerance already established above
(kernel diffs ≤1.5e-3 CPU / ≤2.8e-3 GPU) — i.e. any change must be a pure
dispatch/structure change, not an algorithm change.

**Profiling** (`run_dtsrc_step`, CPU, post-warmup, mean of 30–100 calls)
found the full step dispatching **33 separate JIT calls**, several nested in
Python loops, instead of one fused computation:

| Cost | ms/step | Cause |
|---|---|---|
| Unaccounted Python orchestration | 18.0 | dict/array copies, repeated NumPy↔JAX conversions |
| `dry_convection_mixing_jit` | 13.68 | real compute, but its own dispatch boundary |
| `getcm`+`getchq` (24 calls: 6-iter Python fixed-point loop × NIsurf=2) | 13.35 | should be one traced loop, was 24 dispatches |
| `compute_pk_pek` | 4.56 | real transcendental-`pow()` cost, standalone dispatch |
| `_surface_fluxes_relative_wind` (2 calls) | 2.58 | **not `@jit`-decorated at all** — every op inside dispatched individually |
| `build_pressure_profile` (1 call) | 2.34 | **pure NumPy, Python loop over 40 layers** — never touched JAX |
| everything else | 0.95 | radiation, surface properties, misc |

This is a different — and more fixable — diagnosis than "the grid is too
small for GPU": the isolated PBL/DRYCNV kernels (already single fused JIT
calls) get real 2.3×–6.3× GPU speedup at this exact 3,312-point grid size
(see Performance above). The full chain showed none of that because it pays
dispatch/host-sync overhead ~33 times per step instead of once, not because
the problem is inherently too small.

**Phase 1 implemented** (`p2saom40_driver.py`): fused the entire per-step
hot path — pressure profile, PK/PEK, skin temperature, surface properties,
the Monin-Obukhov fixed-point solve, surface fluxes, layer-1 tendency
update, and DRYCNV — into a single `@jax.jit` function (`_step_core`).
Specifically: `solve_surface_layer`'s 6-iteration Python loop became a
`jax.lax.fori_loop`; `build_pressure_profile`'s 40-layer Python loop became
`jnp.cumsum`; `_surface_fluxes_relative_wind` and `surface_skin_temperature`
were converted from NumPy to `jnp` so they trace inside the fused function
instead of running as separate host-side steps. Same formulas, same
iteration counts, same operation order throughout — purely a dispatch
restructuring, not a physics change. `run_dtsrc_step` is now a thin host
wrapper: NumPy→JAX once at input, one call to `_step_core`, JAX→NumPy once
at output.

**Validated**: field-by-field diff between the old and new driver on the
same real restart input — every field (tg, fluxes, t/q/u/v) differs by
1e-6 to 1e-8 **relative**, consistent with pure floating-point reordering
(different but equally valid evaluation order from `fori_loop`/`cumsum` vs.
the original Python loops), nowhere close to the ≤3e-3 tolerance already
accepted for CPU-vs-GPU noise. The 0.965 correlation / −1.86°C bias full-chain
accuracy result is unchanged. 104/104 unit tests still pass.

**CPU-only result, measured in the (GPU-less) session that implemented
Phase 1**: 59.56 ms/step → 38.17 ms/step, a 1.56× CPU speedup from fusion
alone. This was the first evidence the fusion was doing real work, but it
isn't the number the >2× target is about.

**Phase 3 (real GPU re-measurement) — done, and it surfaced a second,
independent bug.** Re-running `p2saom40_compare.py` on a real A100 (discover
cluster) initially reproduced the *same* ~1.0× ratio as before fusion
(4.04 ms "CPU" vs. 4.08 ms GPU) — which didn't make sense given the CPU-only
result above, and prompted a closer look at the timing code itself rather
than accepting the number. That's what found the measurement bug described
in "Full Physics Chain" above: the script's "CPU" section was silently
running on the GPU whenever a GPU was already active in the process, because
nothing in it ever forced `JAX_PLATFORMS=cpu`. Two things were true at once:
Phase 1's fusion was working (both the mislabeled "CPU" number and the real
GPU number dropped from the ~33/~4 ms pre-fusion range to ~4 ms — fusion
helps GPU dispatch overhead even more than CPU's, which is why they
converged), and the CPU/GPU *ratio* was meaningless because both sides were
secretly the same device.

**Fix**: `p2saom40_compare.py`'s CPU-timing section now spawns a genuine
CPU-only subprocess (with `JAX_PLATFORMS=cpu` set before that subprocess
ever imports `jax`) whenever the main process already has a GPU claimed,
instead of trusting `jax.devices()[0]` to still say "cpu" by the time
section 5 runs. A single run on a GPU node now produces two real numbers.

**Real result, from two separate runs (one `JAX_PLATFORMS=cpu`-forced, one
natural GPU detection) on the same A100 node**: **17.94 ms/step genuine CPU
→ 4.26 ms/step GPU — a 4.2× GPU speedup, clearing the >2× target.** Accuracy
(0.965 correlation, −1.86°C bias, all diagnostic field values) matched
between both runs. No further optimization iteration was needed — the
`fori_loop`/`compute_pk_pek` fallback ideas noted earlier turned out to be
unnecessary once the measurement itself was fixed.

## Open items

- ~~Re-measure the fused driver on a real GPU node~~ — **done**, 4.2× GPU
  speedup confirmed, target cleared. See "GPU optimization: Phase 1" above.
- SEAICE and ATURB placeholder physics — recommended above, not started.
- Wind-speed convention bug — fixed locally in `p2saom40_driver.py`, not yet
  fixed in the shared module files themselves.
- Real spectral radiation (SOCRATES) and real atmospheric dynamics/`CONDSE`
  remain entirely unported and untimed — required before any whole-model
  runtime or ROI figure would mean anything (see Full Physics Chain above).
- The P2SAoM40 Fortran run itself has completed its full 1-year integration
  (Dec 1949 → Nov/Dec 1950, `run_status=13`) — available as a longer-horizon
  validation point (used for the 1950-11-26 accuracy row above) if further
  checks are wanted.
