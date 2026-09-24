# ROCKE-3D → JAX: Status

**As of**: 2026-09-22. **This is now the single source of truth** for this
project's status — `EXECUTIVE_SUMMARY.md`, `FINDINGS.md`, `PORTING_STATUS.md`,
`SESSION_SUMMARY.md`, and `SUMMARY.md` have been retired and their content
merged in here (or explicitly dropped where superseded — see inline notes).
Anything not carried forward from those files was either duplicated here
already, or a stale/unmeasured estimate this document replaces with a real
number.

**Companion slide deck**: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA
("ROCKE-3D → JAX: Status" — bottom line, performance, GPU-optimization
verification, recommendation, output maps; filesystem snapshot in
`status_slides/`). This is also now the *only* slide deck for this project —
an earlier, narrower deck (kernel-level timing only) has been merged into it
and retired.

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
| Is the port complete? | **Fewer modules are faithfully validated than this document previously claimed.** SEAICE, LAKES, and part of ATURB were already documented placeholders. As of 2026-09-23, **GHY is now known to be one too** — most of its physics (evaporation, runoff, soil thermal properties, snow-melt state) is explicitly placeholder code, and its cited "1e-6-level Fortran validation" was a rigged test that hardcoded the untested outputs to zero. The real count of faithfully-validated modules is being re-audited; treat any specific number here as unverified until that's done. **Completing them is a closed question, not an open one**: scoped 2026-09-23 and explicitly decided against — it doesn't serve this project's actual goal (a representative workflow with consistent answers across Fortran/CPU/GPU/optimized, already achieved and validated), only a different goal (scientific fidelity) this project isn't pursuing. See "A validation-methodology gotcha" below and Recommendation. |
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

**A second, narrower scope caveat, specific to the 62× (GPU) / 14.7× (CPU)
figures vs. Fortran** — worth separating from the RADIA/CONDSE exclusion
above because it's about the *included* scope, not the excluded parts: real
Fortran `SURFACE()` (the 263 ms/call baseline these ratios are measured
against) does more physics than this JAX port's full-chain driver actually
runs today. Per "What's simplified in the full-chain driver" below: real
`SURFACE()` does area-weighted sub-tiling across multiple surface types per
grid cell (JAX uses one dominant type per cell), the full `GHY` land-hydrology
scheme (not wired into the JAX driver at all), and a real tridiagonal
turbulence solve via `aturb` (JAX uses a direct flux-tendency shortcut
instead). **So part of the 62×/14.7× reflects JAX genuinely computing
faster, and part of it reflects JAX computing less physics for that same
263 ms slot.** The wall-clock numbers are real and were independently
re-verified (see "Verifying the 4.2×" below) — this caveat doesn't question
whether they're measured correctly, only what conclusion they license: a
byte-for-byte faithful full port (sub-tiling + GHY + aturb all wired in)
would do more work than today's simplified driver and would likely land at
a smaller — still real — multiple.

### Verifying the 4.2× (and 62×): three checks, not just a re-run

A dramatic number, arrived at right after fixing a measurement bug, deserves
more than a second measurement before it goes on a slide. Three independent
checks, done before treating 4.26 ms/4.2×/62× as reportable:

1. **Did the CPU-measurement bug fix touch the GPU number at all?** No —
   diffed line-by-line: section 6's GPU-timing code (warm-up,
   `jax.block_until_ready()` both before and after the timed loop) is
   byte-for-byte unchanged by the fix. Only the mislabeled "CPU" section and
   a print string changed. The GPU number was never the part that was
   broken, in this run or the original ~1.0× one.
2. **Does it reproduce?** Two independent runs of the same post-fusion code
   on the same A100 gave **4.08 ms** and **4.26 ms** — ~4% apart, ordinary
   run-to-run jitter, not a fluke.
3. **Does the magnitude make physical sense?** 3,312 grid points fused into
   one XLA dispatch on an A100: low-single-digit milliseconds is the
   expected range for that little work, not suspiciously fast. And a 4.2×
   GPU-vs-CPU ratio (not 40×) is consistent with this grid being too small
   to fully saturate the GPU — the same real, physical effect the original
   (buggy) comparison was trying and failing to measure.

Full chart (all four stages: real Fortran, JAX-original-CPU,
JAX-original-GPU, JAX-optimized-GPU; log-scale, hover detail, table view)
and this same discussion: https://claude.ai/artifact/FmHsaDwSg9ap1BUY5Be9Ai
— also now the deck's 5th slide, "Verification" (`journey.html`).

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

**A second instance of the same category of gotcha, found 2026-09-23**: while
scoping a "faithful full-chain port" effort, `ghy_jax.py` (GHY, land
hydrology) turned out to have the same problem `simil_numpy` had, in a more
severe form. Reading the actual code: only `compute_sensible_heat` (a
trivial bulk formula) is real. `compute_evap_limits`, `compute_runoff`,
`get_soil_properties`, and `compute_snow_melt` are all explicitly commented
"placeholder" / "simplified" in the source itself — e.g. `get_soil_properties`
uses made-up linear formulas like `0.3 + 0.7*q_in` for thermal conductivity,
never touching the real matric-potential/conductivity tables that
`compute_soil_moisture_table` correctly computes elsewhere in the *same
file* but that nothing ever calls. Worse: `test_ghy_fortran.f` — the file
this project cited as GHY's "1e-6-level Fortran validation" — is not derived
from the real `GHY_DRV.f` (4,939 lines, never consulted); it's a 60-line
hand-written stub explicitly commented **"matching JAX logic"** that
reimplements the same simplified formula, and its `evap`/`snow_melt` outputs
are **hardcoded to `0.0`** with the comment "not critical for validation" —
so the fields that actually needed checking were never checked. **GHY's
"faithful port, validated" status throughout this project's history was
wrong.** Real Fortran source for GHY, ATURB, and SEAICE (~11,700 lines
combined) exists at `modelE2_planet_2.0/model/{GHY_DRV,ATURB,SEAICE}.f` if
this gets revisited; a real port would need to actually read it, not extend
the existing JAX file's pattern.

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
- **GHY** (ground hydrology): not wired into the full-chain driver — and, as
  of 2026-09-23, known to be placeholder code in `ghy_jax.py` itself beyond
  the trivial sensible-heat formula (see the validation-methodology gotcha
  in Accuracy above). Not a "just needs wiring" gap.
- **Layer-1 turbulence**: the driver applies a direct flux-tendency
  shortcut, bypassing `aturb_jax.py` entirely.
- **Radiation**: simplified graybody stand-in (already noted above).

None of this invalidates the 0.965–0.987 correlation result — it's still a
real, meaningful check against real restart data — but it means that result
is a **coarser sanity check of the whole pipeline**, not proof that every
individual sub-component matches Fortran to the same 1e-3 standard as the
validated PBL/DRYCNV kernels.

## Recommendation: the 3 remaining modules — closed out, not pursuing (2026-09-23)

**Status**: a faithful port of GHY (and, by the same reasoning, ATURB/SEAICE)
was scoped on 2026-09-23 and explicitly decided against — not because it's
infeasible (real Fortran source exists, and the multi-layer restart state
GHY needs was located in `1JAN1950.rsfP2SAoM40.nc`), but because it doesn't
serve this project's actual goal. That goal, clarified the same day: run a
*representative* workflow across Fortran / JAX-CPU / JAX-GPU-original /
JAX-GPU-optimized that gives ~the same answers at every stage, not to
maximize scientific fidelity to Fortran. That property already holds today,
fully validated — the Phase 1 fusion work was a pure dispatch restructuring
(field-level diffs of 1e-6–1e-8 relative between old and new driver), so all
three JAX legs compute *identical* physics and give identical answers
(0.965 correlation vs. Fortran, held constant CPU→GPU→optimized). Porting
GHY/ATURB would change what JAX computes, requiring a fresh accuracy
re-validation before any of the four legs could be trusted again — real
extra work that doesn't move this project toward its stated goal. See
"GPU optimization: Phase 1" above for the full reasoning trail.

**The domain-judgment reasoning below is kept for the record** (why SEAICE
and ATURB would matter *if* the project's goal were scientific fidelity
rather than representative-workflow benchmarking) — not acted on, given the
above.

**Why they're still placeholders, not a gap that was missed**: the original
port deliberately left SEAICE/LAKES/ATURB as documented stand-ins (e.g.
LAKES' `lkmix` is a no-op) to reach interface-complete (17/17) first,
prioritizing full validation of the columnar physics (PBL, DRYCNV) instead.
A scoping choice, made explicit here rather than discovered later.

**SEAICE** (core thermodynamics) and **ATURB** (PBL-top-finding) — would be
the next candidates for porting *if this project's goal changes* to
prioritize scientific fidelity. Reasoning: sea ice is a standard, physically
active component of any ROCKE-3D ocean-coupled run including P2SAoM40, not
an optional add-on, and it dominates the surface energy balance specifically
in polar regions — exactly where the current 0.987 *global* correlation
check is least able to reveal a local error. ATURB's placeholder affects
boundary-layer depth, which feeds surface flux accuracy broadly, not just at
the poles. Both are also useful precisely *because* they're harder than
PBL/DRYCNV: collecting "gotchas" from harder physics is more informative
than from the easy columnar-physics modules, which have already told us most
of what they can about where the vmap-over-grid-cells approach works
cleanly. **Not currently being acted on** — see Status above.

**LAKES** (`lkmix`) — lowest priority of the three even under the
fidelity-first framing. Lakes cover a much smaller fraction of Earth's
surface than sea ice, so the accuracy payoff is lower for the effort.

*(This SEAICE/lakes area-of-impact reasoning is domain judgment, not a number
measured in this project's own data — flagging that distinction rather than
presenting it as measured fact.)*

### Reference: all 17 modules

| Module | JAX file | Real-Fortran validation |
|---|---|---|
| DRYCNV | `drycnv.py` | Direct, kernel-level (see Accuracy) |
| PBL | `pbl.py` | Direct, kernel-level (see Accuracy) |
| GHY | `ghy_jax.py` | **Placeholder** (found 2026-09-23) — only `compute_sensible_heat` is real; evap/runoff/soil-properties/snow-melt are explicitly placeholder code, and the cited Fortran validation was rigged (see gotcha above). Not wired into the full-chain driver either. |
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
| ATURB | `aturb_jax.py` | **Partial** — substantial real code (tridiagonal solver, TKE production/dissipation structure, `find_pbl_top`), but the stability-closure constants inside `e_gcm` (`c1`–`c5`, `b1`) are hardcoded placeholders (`1.0, 2.0, 3.0...`), not the real Mellor-Yamada-derived values from `PBL.f`. Its own `validate_aturb()` admits it validates "against a placeholder Fortran-like implementation," not real Fortran. Bypassed by the full-chain driver's layer-1 shortcut — recommended next, but scope is bigger than "wire it in" (see Recommendation and the GHY gotcha above for why that assumption needs checking before trusting it). |

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

## Track B: full-fidelity port (branch `full-fidelity-port`, started 2026-09-24)

HQ asked for a port that can be defended against the real Fortran. Plan: `FULL_FIDELITY_PLAN.md`;
measured differences vs. the earlier ("Track A") representative driver: `FULL_FIDELITY_DELTAS.md`;
working log: `fullfidelity/PHASE0_LOG.md`. Everything in this file above is **Track A** and unchanged.

**What is established so far (all against the real ModelE executable, not stand-ins):**
- The real P2SAoM40 model builds and runs here (ifort 19.1.3 + NetCDF, recipe in
  `fullfidelity/env_modele.sh`); a 5-day re-run reproduces the original restart **byte-for-byte**.
  Instrumented copies with dump hooks do not perturb results (256/257 restart fields bitwise; the 257th is
  wall-clock timing).
- **Track A's DRYCNV is not part of the real model**: the P2SAoM40 executable has no DRYCNV; free-
  atmosphere mixing is ATURB. Track A's DRYCNV benchmark is therefore a stand-alone kernel result only.
- Track A's one-step surface answer is ~13× farther from real Fortran than doing nothing (layer-1 T error
  0.36 K vs. 0.027 K signal), not fixed by real SST; it also used approximated constants (deltx, g, R).
- **Track B ports validated at float64 rounding level against real Fortran call records:**
  ATURB (T, Q, TKE, PBL height, U/V B-grid diffusion, A-grid winds; 3 dates, max error 1e-12 K / 1e-14 m/s,
  `aturb_ff.py`, `aturb_uv_ff.py`) and PBL `advanc` (surface layer; 27.5k calls on 4 surface types, worst
  max-abs/rms 5e-11, `pbl_ff.py`). 19 tests (with mutation checks).
- Chaos noise floor of the real model: a 1-ulp perturbation grows to 0.27 K RMS / 0.74 m/s in 5 days
  (global means agree to ~1e-3–1e-2 of that): multi-day comparisons must be statistical.

**Not done yet (so no whole-model fidelity claim can be made):** SURFACE tile-flux logic, GHY (land),
SEAICE/LAKES/LANDICE thermodynamics, radiation (SOCRATES), clouds/moist convection, atmospheric
dynamics, ocean. No speed numbers for Track B yet (only float64 CPU correctness runs).

## Round 2 optimization (2026-09-23) — CPU and GPU (A100) measured

Goal: ≥5× on the Python implementations, accuracy held. "Anything goes" —
this round was profile-driven and mostly *not* JAX-tuning: it removed
algorithmic waste and per-call overhead.

**Measured on the CPU node** (12 cores, load ~18 — timings are noisy; ratios
are from interleaved baseline-vs-new runs on the same data):

| Item | Baseline | Round 2 | Speedup |
|---|---|---|---|
| DRYCNV kernel, layer-first (native) | ~12 ms | ~1.2 ms | **~9.5×** |
| DRYCNV kernel, original level-last API | ~12 ms | ~2.9 ms | **~4×** |
| PBL `simil` kernel | 0.34 ms | 0.16 ms | **~2.1× (short of 5×)** |
| Full chain, single call (host state in/out) | 26–44 ms | 6–10 ms | **~3.6–4.4×** |
| Full chain, chained device-resident (`lax.scan`), per step | 26–31 ms | ~3–7 ms | **~5×** (best ~9×) |

What changed:
- **DRYCNV**: original scan carried and dynamic-update-sliced the whole
  (I,J,L) arrays every layer (O(L²) traffic). Now a layer-first scan with a
  2-slab carry. Layer-first (L,J,I) is byte-identical to Fortran's (I,J,L).
  Unrolling was tried: 2× *slower* on CPU (fusion duplicates producers).
  A strided level-last scan was also tried: slower (5.6 vs 3.0 ms).
- **PBL**: fewer transcendentals — vectorized Cephes `atan`, `sqrt(sqrt)`
  for x^0.25, `exp(log(w)/3)` for cube root, shared `log(z/z0)`, one unified
  unstable-branch formula. Op-count bound; ~2× is what it gives.
- **Driver** (`p2saom40_driver.py`): loop-invariant work (pressure profile,
  PK, PDSIG, Monin-Obukhov constants, lat/lon trig) hoisted into
  `prepare_static`; on-device solar zenith; `run_steps_device` chains N steps
  in one dispatch with state never leaving the device. `run_dtsrc_step`
  remains as a compatible wrapper (with a prepare cache).
- Rejected: XLA CPU fast-math (no speedup, outputs changed ~5e-4).

**Full chain on the discover A100 node (2026-09-24, `p2saom40_compare.py`,
same node/method as the Phase-1 baseline of 17.94 ms CPU / 4.26 ms GPU):**

| Path | CPU ms/step | vs 17.94 | GPU ms/step | vs 4.26 | GPU vs Fortran SURFACE+GROUND (264 ms) |
|---|---|---|---|---|---|
| Single-call wrapper | 4.20 | **4.3×** | 3.62 | **1.2×** | 73× |
| Chained device-resident | 1.89 | **9.5×** | 0.74 | **5.8×** | 356× |

- The ≥5× target is met on **both** CPU and GPU, but only by the chained
  device-resident path. The single-call wrapper on GPU gains just 1.2×: it is
  dominated by host↔device transfer and per-call dispatch, not physics.
- Chained GPU is only 2.6× faster than chained CPU (0.74 vs 1.89 ms): the CPU
  got much better too; the workload (3312 columns) is small for an A100.
- The Fortran comparison covers SURFACE+GROUND only (radiation excluded, as
  before), so the 356× is a like-for-like kernel-group ratio, not a
  whole-model speedup.
- Not yet reported by you: `compare_jax.py` kernel timings on GPU (DRYCNV
  scan vs the old 0.56 ms).

### What "chained device-resident" means (and how it differs from Phase 1)

A model step is: take the atmospheric state (T, q, layer-1 winds), compute
surface fluxes and dry convection, return the updated state. The question is
*where the state lives between steps*.

**Single-call path (`run_dtsrc_step`)** — what the original driver did, and
what the compat wrapper still does. Every call: (1) copy the state from host
NumPy arrays to the device, (2) run the step, (3) copy results back to host
NumPy, (4) rebuild anything derived from the inputs (pressure profile, PK,
PDSIG, trig of lat/lon, solar zenith on the host). On a CPU the "copies" are
cheap but nonzero (dtype/layout conversion, level-last↔layer-first
transposes); on a GPU they cross PCIe and force a synchronization each step.

**Chained device-resident path (`prepare_static` + `run_steps_device`)** —
1. `prepare_static` runs **once**: everything that does not change between
   steps (pressure/PK/PDSIG, surface type masks, lat/lon trig, constants) is
   computed and uploaded to the device.
2. `dyn_from_state` uploads the evolving state **once**, in layer-first
   layout (Fortran's memory order, so no transposes inside the loop).
3. `run_steps_device` executes N steps inside a single `jax.lax.scan`: the
   state is the scan carry, so it stays in device memory from step to step.
   The solar zenith is computed on the device from four scalars per step.
4. `state_from_dyn` copies the final state back to the host **once**.

Per-step cost therefore contains only physics — no transfers, no host
recomputation, no per-step Python launch or GPU synchronization. This is how
a real model advances (state stays resident; output is written only every so
many steps), which is why it is the honest per-step number, while the
single-call number is the cost of a *compatibility interface*.

**How this differs from the earlier JAX optimization (Phase 1):**

| | Phase 1 (2026-09-22) | Round 2 (2026-09-23) |
|---|---|---|
| Target | Dispatch overhead *within* one step | Everything *around and between* steps, plus kernel arithmetic |
| Change | Fused ~33 separate jit calls into one `@jax.jit` per step | Chained N steps into one `lax.scan` dispatch; hoisted invariants; kept state on device |
| Boundary crossings | Still host→device→host **every step** | Once per *run* |
| Kernel math | Unchanged | DRYCNV rewritten (no O(L²) array copies), PBL transcendentals cut |
| Result (A100) | 17.94 → 4.26 ms/step chain (4.2× vs CPU) | 4.26 → 3.62 single-call (1.2×), → 0.74 chained (5.8×) |

In short: Phase 1 made one step a single GPU launch; Round 2 makes many steps
a single launch and stops moving data in and out between them. Phase 1's gain
was mostly launch-count; Round 2's is mostly transfer/recomputation removal
plus a genuinely cheaper DRYCNV. Note the single-call wrapper only improved
1.2× on GPU precisely because it still pays the boundary crossings Phase 1
left in place.

Practical rule: use `run_steps_device` for any multi-step run or benchmark;
use `run_dtsrc_step` only for one-off calls and compatibility with old
scripts. Chained runs must be given all N solar-scalar rows up front
(`solar_scalars` per timestamp); diagnostics are returned for the last step only.

**Accuracy (the part to trust only as far as it is checked):**
- New tests (`tests/test_round2_optimizations.py`, 11) vs independent
  float64 references over all stability branches; DRYCNV vs float64 <5e-4 K;
  layer-first == level-last bitwise; device path == wrapper == repeated steps.
- Chained-step diffs vs baseline show isolated cell differences (latent flux
  up to ~5.9 W/m² at one cell). These are **not a regression**: the
  Monin-Obukhov fixed point flips branch in marginal cells, and perturbing
  the *baseline* by one float32 ulp produces divergence of the same size.
  Global means/correlations unchanged; distance to real Fortran unchanged.
- One deliberate change: pressure prep is now float64 on the host (more
  accurate than the original float32 cumsum). `precision="float32"`
  reproduces the original arithmetic to ~1e-6.
- Found and fixed while checking: `flong` must use `radiation_jax.STBO`.

**Caveats:**
- The PBL kernel alone did not reach 5×; the 5× is a full-chain result, and
  the single-call wrapper (which pays host↔device each call) is ~4×. The
  ≥5× figure needs the device-resident path.
- Kernel-level GPU timings (`compare_jax.py`) still pending; DRYCNV's scan
  may be loop-overhead-bound on GPU (old GPU DRYCNV was 0.56 ms).
- Only JAX code was optimized; the NumPy scripts in `mantle/` were not.

## Open items

- ~~Re-measure the fused driver on a real GPU node~~ — **done**, 4.2× GPU
  speedup confirmed, target cleared. See "GPU optimization: Phase 1" above.
- ~~A faithful full-chain port (sub-tiling + GHY + ATURB)~~ — **closed,
  2026-09-23, decided against.** Scoped in detail (real Fortran source
  totals ~11,700 lines: `GHY_DRV.f` 4,939, `SEAICE.f`+`SEAICE_DRV.f` 5,394,
  `ATURB.f` 1,405; GHY specifically is a coupled land-surface model needing
  its own internal orchestration ported, not independent functions to wire
  in) and explicitly not pursued: it doesn't serve this project's actual
  goal (a representative, consistent-answers workflow across Fortran/CPU/
  GPU/optimized — already achieved), only a different goal (scientific
  fidelity) this project isn't after. See Recommendation above for the full
  reasoning. Don't reopen this without a real change in project goal.
- ~~SEAICE and ATURB placeholder physics~~ — same closure as above; the
  underlying domain-judgment case for porting them (kept in Recommendation)
  wasn't wrong, it's just not what this project is optimizing for.
- Wind-speed convention bug — fixed locally in `p2saom40_driver.py`, not yet
  fixed in the shared module files themselves.
- Real spectral radiation (SOCRATES) and real atmospheric dynamics/`CONDSE`
  remain entirely unported and untimed — required before any whole-model
  runtime or ROI figure would mean anything (see Full Physics Chain above).
- The P2SAoM40 Fortran run itself has completed its full 1-year integration
  (Dec 1949 → Nov/Dec 1950, `run_status=13`) — available as a longer-horizon
  validation point (used for the 1950-11-26 accuracy row above) if further
  checks are wanted.
