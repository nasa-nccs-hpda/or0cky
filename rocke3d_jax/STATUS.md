# ROCKE-3D → JAX: Status

**As of**: 2026-09-21. This is the current-state summary — no revision history, no
before/after narrative. For the full history of how we got here, see
`FINDINGS.md`, `PORTING_STATUS.md`, `EXECUTIVE_SUMMARY.md`, `SESSION_SUMMARY.md`
(kept as-is, not superseded).

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
| Has this been checked on GPU? | **Yes, for the two directly-validated kernels** (DRYCNV, PBL) — real numbers, run 2026-09-20. **Not yet** for the larger chained physics group (radiation + surface + ground) or the 3 unported modules. |
| Is the port complete? | **14 of 17 modules** are faithful ports, validated against real Fortran. 3 (SEAICE core thermodynamics, LAKES mixing, part of ATURB) are documented placeholders. See recommendation below. |
| Does this project still use NumPy comparisons? | It has some (`benchmark_all.py` and related) — **recommend dropping them from the headline story**. See below. |

## Accuracy

Two different validations exist; both check against **real Fortran**, not a
NumPy stand-in:

| What | Method | Result |
|---|---|---|
| PBL + DRYCNV kernels | Real inputs shared byte-for-byte between a from-scratch `ifort`-compiled Fortran reference and JAX, at P2SAoM40's real grid size (3,312 points × 40 layers) | Max diff ≤1.5e-3 (CPU), ≤2.8e-3 (GPU) against output scales of 10¹–10³ — floating-point-level, not algorithmic |
| Full physics chain (PBL+radiation+surface+ground) | Driven by P2SAoM40's actual restart state (`fort.1.nc`), one real timestep | 0.987 spatial correlation vs. the real run's period-mean surface temperature (a coarser sanity check — period-mean vs. single-step, see `FINDINGS.md` §2c for the caveat) |

**A validation-methodology gotcha worth knowing**: this project's earlier
"Fortran-like" NumPy reference for PBL (`simil_numpy`) silently omitted half
of the real algorithm's branches. Re-deriving the Fortran side directly from
the real three-branch formula (not the NumPy stand-in) is what produced the
clean 1.5e-3 agreement above — the earlier, looser "~2.3%"/"1e-6" accuracy
claims elsewhere in this repo should be considered superseded by this result
for PBL and DRYCNV specifically.

## Performance (real Fortran vs. JAX)

| Comparison | Fortran (CPU) | JAX (CPU) | JAX (GPU) |
|---|---|---|---|
| DRYCNV kernel | 1.27 ms | 17.1 ms — **13.4× slower** | 0.56 ms — **2.3× faster than Fortran-CPU** |
| PBL kernel | 0.37 ms | 0.20 ms — 1.9× faster | 0.06 ms — **6.3× faster than Fortran-CPU** |
| Full physics chain (PBL+radiation+surface+ground) | ~264 ms/step (radiation excluded — see note) | ~48 ms/step — **5.5× faster** | not yet measured |

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

## Recommendation: the 3 remaining modules

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

- Full physics-chain GPU run (radiation+surface+ground together) — not done, only the two-kernel subset has real GPU numbers.
- SEAICE and ATURB placeholder physics — recommended above, not started.
- Wind-speed convention bug — fixed locally in `p2saom40_driver.py`, not yet fixed in the shared module files themselves.
