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
| Has this been checked on GPU? | **Yes, both scopes.** Kernels (DRYCNV, PBL) in isolation: real 2.3–6.3× speedup, run 2026-09-20. Full chained physics group, real NVIDIA A100 (node `warpa005`, discover cluster): **~1.0× — essentially no GPU benefit** at this grid size (3,312 points) — corrects an earlier ~20–30× *estimate* that has now been fully retired, not just flagged (see Full Physics Chain below). |
| Is the port complete? | **14 of 17 modules** are faithful ports, validated against real Fortran. 3 (SEAICE core thermodynamics, LAKES mixing, part of ATURB) are documented placeholders. See Recommendation below. |
| Does this project still use NumPy comparisons? | It has some (`benchmark_all.py` and related) — **dropped from the headline story**. See below. |

## Full Physics Chain: the finding that matters most

This is the single most important correction in this project's history, so it
gets its own section rather than being a line in a table.

**What was measured** (2026-09-22, real NVIDIA A100, node `warpa005` on the
discover cluster, via `p2saom40_compare.py`, driven by P2SAoM40's actual
restart state): the chained PBL+SURFACE+GROUND physics group, one real
`DTsrc` step (1800 s model time):

| Leg | Time / step |
|---|---|
| Real Fortran (CPU) | 264.0 ms |
| JAX (CPU) | 33.51 ms |
| JAX (GPU, A100) | 32.96 ms |

**GPU vs. JAX-CPU: ~1.0× — essentially no GPU benefit.** Still ~8.0× faster
than real Fortran (vs. ~5.5× measured earlier on a CPU-only node — see
Performance below for why those two CPU figures differ). Root cause: this
orchestrator's grid is only 3,312 points — too small for a GPU to amortize
kernel-launch/dispatch overhead, the same effect already visible in the PBL
kernel's more modest GPU gain, just more pronounced across a full multi-step
chain.

**Why this replaces every earlier ~20–30× GPU figure**: that number was an
*estimate*, extrapolated before any GPU was available in this environment,
and it was never actually measured until this run. It has been removed
throughout this project's documentation (it previously appeared in
`EXECUTIVE_SUMMARY.md`'s Business Impact, ROI, and Timeline sections, and in
`README_GPU.md`'s setup guide) — not flagged with a correction note, *removed
and replaced* with the real number, since that document is being retired.

**What this does *not* contradict**: the DRYCNV+PBL kernel-level GPU result
(2.3–6.3×, see Performance below) is a different, narrower scope — real
per-call kernel dispatch, not a multi-module chained pipeline — and it holds.
The story is not "GPU never helps here," it's "the full chained orchestrator
at this grid size doesn't benefit the way the pre-measurement estimate
assumed."

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
legs** of the 264/33.5/33.0 ms comparison above to keep it a fair speed
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
| Same, later restart snapshot 1950-11-26 (2026-09-22, GPU-capable run) | Same method, ~11 simulated months further into the same P2SAoM40 run (which has since completed its full 1-year integration) | **0.965** spatial correlation (−1.86°C bias) — consistent with the earlier snapshot, i.e. fidelity holds up over a full year of integration, not just the first month |

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
| Full physics chain (PBL+radiation+surface+ground) | ~264 ms/step (radiation excluded from JAX legs — see Full Physics Chain above) | ~48 ms/step — **5.5× faster** (measured on the original CPU-only node) | ~33.0 ms/step on a GPU-capable node — **~1.0× vs. that node's own JAX-CPU** (33.5 ms, "no GPU benefit"); ~8.0× vs. Fortran-CPU |

The two JAX-CPU full-chain figures (48 ms vs. 33.5 ms) come from **different
physical nodes** — not a contradiction, just different hardware. Neither is
"more correct"; both are real measurements.

**Reading the kernel table honestly**: JAX-CPU is *not* uniformly faster than
Fortran — it loses DRYCNV outright on CPU (small-array dispatch overhead).
GPU is where JAX wins decisively on both kernels. **The case for JAX here is
a GPU case for isolated kernels, not a CPU case — and not (yet) a case for
the full chained pipeline at this grid size.** See Full Physics Chain above.

## What's simplified in the full-chain driver (honesty section)

The kernel-level accuracy numbers above (PBL, DRYCNV) come from a rigorous,
faithful port validated line-for-line against Fortran. The **full-chain
driver** (`p2saom40_driver.py`) used for the 0.965–0.987 correlation and the
48/33.5/33.0 ms timing numbers is not uniformly that rigorous — several
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

`benchmark_all.py`/`benchmark_all_cpu.py` compare JAX against `simil_numpy`/
`dry_convection_numpy` — hand-written, un-jitted Python reimplementations of
the same formulas, useful only for answering "how much does JIT compilation
alone buy you, holding the algorithm constant." That's a real but narrow
engineering question, and it's not what management or the project goal
actually needs to know, which is JAX vs. **the real Fortran model** run in
production. Given `simil_numpy` was also found to be *wrong* (see Accuracy
above), it's a weak reference on top of being the wrong comparison. The files
are kept (harmless as engineering scratch work) but NumPy-relative speedups
are no longer cited anywhere in this project's status reporting — every
headline number is JAX vs. Fortran.

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

## Open items

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
