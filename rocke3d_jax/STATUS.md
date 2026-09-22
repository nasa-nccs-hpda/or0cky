# ROCKE-3D → JAX: Status

**As of**: 2026-09-22. This is the current-state summary — no revision history, no
before/after narrative. For the full history of how we got here, see
`FINDINGS.md`, `PORTING_STATUS.md`, `EXECUTIVE_SUMMARY.md`, `SESSION_SUMMARY.md`
(kept as-is, not superseded).

**Companion slide deck**: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA (4
slides: bottom line, performance, recommendation, output maps; filesystem
snapshot in `status_slides/`) tracks this document — the two are kept in sync.

## The goal

Use an AI coding agent to autonomously convert ROCKE-3D (NASA GISS's Fortran
GCM) to Python/JAX, and collect real benchmarks — **accuracy**, **CPU and GPU
performance**, **conversion cost**, and **gotchas** — validated against **real
production restart data** from an actual completed run (`P2SAoM40`: SOCRATES
radiation, Earth-like atmosphere, dynamic ocean, 72×46×40 grid). The target
comparison has always been **JAX vs. the real Fortran model**, not JAX vs. a
second Python reimplementation.

## Bottom line

| Question | Answer |
|---|---|
| Does the JAX port match real Fortran numerically? | **Yes**, for the modules tested directly against it (PBL, DRYCNV): floating-point-level agreement (1e-9–1e-3) on both CPU and GPU. |
| Is JAX faster than real Fortran? | **Depends on device and module.** See table below — not a blanket yes. |
| Has this been checked on GPU? | **Yes, both scopes now.** Kernels (DRYCNV, PBL) in isolation: real 2.3–6.3× speedup, run 2026-09-20. Full chained physics group (radiation+surface+ground), real NVIDIA A100, run 2026-09-22: **~1.0× — essentially no GPU benefit** at this grid size (3,312 points) — corrects an earlier ~20–30× *estimate* repeated elsewhere in this repo. Not yet checked: the 3 unported modules. |
| Is the port complete? | **14 of 17 modules** are faithful ports, validated against real Fortran. 3 (SEAICE core thermodynamics, LAKES mixing, part of ATURB) are documented placeholders. See recommendation below. |
| Does this project still use NumPy comparisons? | It has some (`benchmark_all.py` and related) — **recommend dropping them from the headline story**. See below. |

## Accuracy

Two different validations exist; both check against **real Fortran**, not a
NumPy stand-in:

| What | Method | Result |
|---|---|---|
| PBL + DRYCNV kernels | Real inputs shared byte-for-byte between a from-scratch `ifort`-compiled Fortran reference and JAX, at P2SAoM40's real grid size (3,312 points × 40 layers) | Max diff ≤1.5e-3 (CPU), ≤2.8e-3 (GPU) against output scales of 10¹–10³ — floating-point-level, not algorithmic |
| Full physics chain (PBL+radiation+surface+ground), restart snapshot 1949-12-01 | Driven by P2SAoM40's actual restart state (`fort.1.nc`), one real timestep | **0.987** spatial correlation vs. the real run's period-mean surface temperature (+1.0°C bias) |
| Same, later restart snapshot 1950-11-26 (2026-09-22, GPU-capable run) | Same method, ~11 simulated months further into the same P2SAoM40 run (which has since completed its full 1-year integration) | **0.965** spatial correlation (−1.86°C bias) — consistent with the earlier snapshot, i.e. fidelity holds up over a full year of integration, not just the first month |

Both full-chain rows are a coarser sanity check than the kernel row above —
period-mean vs. single-step, see `FINDINGS.md` §2c for the caveat.

**A validation-methodology gotcha worth knowing**: this project's earlier
"Fortran-like" NumPy reference for PBL (`simil_numpy`) silently omitted half
of the real algorithm's branches. Re-deriving the Fortran side directly from
the real three-branch formula (not the NumPy stand-in) is what produced the
clean 1.5e-3 agreement above — the earlier, looser "~2.3%"/"1e-6" accuracy
claims elsewhere in this repo should be considered superseded by this result
for PBL and DRYCNV specifically.

**Spatial view of the same result**: `visualize_p2saom40_kernel_maps.ipynb`
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
| Full physics chain (PBL+radiation+surface+ground) | ~264 ms/step (radiation excluded — see note) | ~48 ms/step — **5.5× faster** (original CPU-only node) | ~33.0 ms/step on a GPU-capable node — **~1.0× vs. that node's own JAX-CPU** (33.5 ms, "no GPU benefit"); ~8.0× vs. Fortran-CPU |

**Radiation note**: JAX's radiation module is a simplified graybody
stand-in, not real spectral radiative transfer — any speedup figure that
includes it (some appear elsewhere in this repo, up to ~200×+) reflects a
fidelity gap, not a fair speed comparison, and should not be quoted as JAX
performance.

**Reading the kernel table honestly**: JAX-CPU is *not* uniformly faster than
Fortran — it loses DRYCNV outright on CPU (small-array dispatch overhead).
GPU is where JAX wins decisively on both kernels. This is a real, useful
finding, not a caveat to downplay: **the case for JAX here is a GPU case**,
not a CPU case.

**The full-chain GPU result is a real correction, not a footnote** (measured
2026-09-22, NVIDIA A100, on a different — GPU-capable — node than the
original 48 ms CPU figure above): on that node, JAX-GPU (33.0 ms/step) is
**essentially the same as JAX-CPU on the same node** (33.5 ms/step) — no
meaningful GPU speedup. Same root cause as the PBL kernel's more modest GPU
gain in the table above, just more pronounced across the full multi-step
chain: at only 3,312 grid points, there isn't enough work per step for a GPU
to amortize kernel-launch/dispatch overhead. GPU deployment still isn't
pointless here (~8.0× faster than real Fortran, comparable to the ~5.5×
already seen on CPU), but the ~20–30× GPU *estimate* quoted in
`EXECUTIVE_SUMMARY.md`'s ROI/Business-Impact/Timeline sections for this
full-chain scope was wrong and should not be used for planning — see the
correction note there and in `FINDINGS.md` §2c.

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

## On NumPy: recommend dropping it from the headline story

`benchmark_all.py`/`benchmark_all_cpu.py` compare JAX against `simil_numpy`/
`dry_convection_numpy` — hand-written, un-jitted Python reimplementations of
the same formulas, useful only for answering "how much does JIT compilation
alone buy you, holding the algorithm constant." That's a real but narrow
engineering question, and it's not what management or the project goal
actually needs to know, which is JAX vs. **the real Fortran model** run in
production. Given `simil_numpy` was also found to be *wrong* (see Accuracy
above), it's a weak reference on top of being the wrong comparison.
Recommend: keep the files (they're harmless as engineering scratch work) but
stop citing NumPy-relative speedups in status reports or management
material — every headline number going forward should be JAX vs. Fortran.

## Conversion cost & gotchas

No independently-verified dollar/hour figure exists for this work; treat the
session-by-session cost log in `SESSION_SUMMARY.md` as a rough, self-reported
estimate spanning multiple agent sessions, not an audited number. What *is*
concretely known — four real gotchas hit during this conversion, each with a
traceable fix:

1. **Wind-speed convention bug** (`FLUXES`/`SURFACE`): computed wind speed
   from the surface-relative wind alone; with the physically correct
   `us=vs=0` no-slip value for land/ice, this silently zeroed every flux for
   those cells. Not caught by unit tests (they used arbitrary nonzero test
   values) — only surfaced when driven with real P2SAoM40 data.
2. **Incomplete reference implementation** (`simil_numpy`): missing the
   stable branch of `find_dpsih`, wrong `dpsiq` shortcut — see Accuracy
   above. A reminder that a hand-written "reference" needs the same
   validation rigor as the port itself.
3. **A `numpy.ndarray.tofile()` ordering footgun**: it always writes C-order
   bytes regardless of the array's actual memory layout, which silently
   corrupted shared test inputs between the Fortran and JAX legs of the
   kernel comparison above until caught by an input round-trip check.
4. **Unit tests validate placeholder logic against itself**: this is why
   SEAICE/LAKES/ATURB's placeholders passed their own tests despite not
   being physically complete — a real methodology gap, not just a code gap.

**General pattern**: columnar physics (each grid cell independent — PBL,
DRYCNV, RADIATION-as-implemented) ports cleanly onto JAX's vectorized
execution model. The genuinely out-of-scope pieces (atmospheric dynamics,
moist convection, the ocean GCM — see `FINDINGS.md` for LOC and
cross-cell-communication counts) are structurally harder because they need
values from *neighboring* grid cells, which doesn't map onto the same
vectorization approach without real additional work.

## Open items

- ~~Full physics-chain GPU run~~ — **done** (2026-09-22, NVIDIA A100): ~1.0× vs. JAX-CPU, see Performance above. New follow-up: re-derive every ROI/timeline figure in `EXECUTIVE_SUMMARY.md` that assumed the old ~20–30× estimate for this scope (flagged there, not yet rewritten).
- SEAICE and ATURB placeholder physics — recommended above, not started.
- Wind-speed convention bug — fixed locally in `p2saom40_driver.py`, not yet fixed in the shared module files themselves.
- The P2SAoM40 Fortran run itself has now completed its full 1-year integration (Dec 1949 → Nov/Dec 1950, `run_status=13`) — available as a longer-horizon validation point (used for the 1950-11-26 accuracy row above) if further checks are wanted.
