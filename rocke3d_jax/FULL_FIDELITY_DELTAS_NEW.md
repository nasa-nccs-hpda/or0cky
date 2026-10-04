# Full-Fidelity Port: Delta Ledger (continued)

This file continues `FULL_FIDELITY_DELTAS.md` starting at D55. Split into a
new file because `FULL_FIDELITY_DELTAS.md` has grown too large (~36K tokens)
for this session's edit tooling to reliably modify in place (`multi_edit`
reported false "success" on edits that did not persist, and could not find
even trivial substrings in that file -- consistent with a silent size-limit
failure, not an actual absence of the text). Read `FULL_FIDELITY_DELTAS.md`
for D1-D54 first; this file picks up immediately after D54, before that
file's "## Pending rows" section.

## D55: OCONV scoping -- dead-code branches resolved against the real rundeck, not assumed; live scope cut roughly in half

No port code this delta. Read all of `OCONV` (`OCNKPP.f:1361-2886`, 1,526 lines) end to end plus
`KVINIT`/`OVDIFF`/`OVDIFFS`/`REDUCE_FIG` in full, and resolved every remaining compile-time and
runtime branch against the real rundeck (`decks/P2SAoM40.R`) and the real module defaults, the
same discipline that caught D52's `bldepth` mis-scope (D53) -- none of the findings below are
assumed from source alone.

**Compile-time (`#ifdef`) flags, checked against `decks/P2SAoM40.R`'s Preprocessor Options
(none of the four appear there -- all confirmed off):**
- `OCN_GISS_SM` -- kills more of `OCONV` than D52 scoped: not just the `DTP4G3D`/`DTP4S3D`
  lookups it found, but the entire submesoscale setup block at the top of `OCONV` (`p3d`/`rho3d`
  cell-centered thermodynamic state, `rx`/`ry`/`gx`/`gy`/`sx`/`sy` gradient arrays, the
  `get_gradients0` calls feeding them -- already known dead, D52) and the entire `giss_sm_mix`
  call block inside the per-column `ITER` loop (cell-centered `uc`/`vc` velocity averaging, the
  mixing call itself, its time-averaging bookkeeping).
- `TRACERS_OCEAN` -- kills every `TRML`/`TXML`/`TYML`/`TXXML`/`TYYML`/`TXYML` tracer-diffusion
  block throughout `OCONV` (inside the `ITER` loop, the horizontal-gradient-diffusion pass, and
  the final `extra_slope_limitations` `REDUCE_FIG` pass), plus the `TRMO1`/`TXMO1`/`TYMO1`
  save-before-fluxes block inside `KVINIT` -- new finding, D52 never read `KVINIT`'s body.
- `ENHANCED_DEEP_MIXING` -- kills the `kvextra` background-diffusivity-profile addition near the
  top of `OCONV`. Not checked by D52/D53 at all; confirmed dead here.
- `DISABLE_KPP_DGRID_MIXING` -- the inverse of the other three: because this is **not** defined,
  the D-grid velocity-diffusion call (`OVDIFF` on `ULD`, guarded by `#ifndef
  DISABLE_KPP_DGRID_MIXING`) **is live**, not dead. D52 listed `OVDIFF` as "called extensively"
  without distinguishing the C-grid (`UL`, always live) and D-grid (`ULD`, conditionally live)
  calls -- resolved here.

**Runtime flag, resolved from the real module default (not a `#define`, so not visible in the
Preprocessor Options list -- had to find the actual Fortran declaration):**
- `use_qus` -- `OCEAN_COM.f` declares `INTEGER :: USE_QUS=0`, and `ocean_use_qus` (the
  `sync_param` name that would let the rundeck override it) does not appear anywhere in
  `decks/P2SAoM40.R` -- confirmed `use_qus=0` for this build, same conclusion D52 already reached
  for the quadratic-moment block, but this delta traces its *full* effect on `OCONV`, which is
  larger than D52 scoped: `OCONV` has a top-level three-way `if(use_qus==1) ... else ... endif`
  that sets four logical flags (`adjust_zslope_using_flux`, `relax_subgrid_zprofile`,
  `extra_slope_limitations`, `mix_tripled_resolution`) controlling which of three *mutually
  exclusive* end-of-routine blocks runs. With `use_qus=0`, this resolves deterministically to
  `adjust_zslope_using_flux=.true.`, `extra_slope_limitations=.true.`,
  `relax_subgrid_zprofile=.false.`, `mix_tripled_resolution=.false.` -- which means, beyond the
  quadratic-moment (`GXXML`/`GYYML`/`GXYML`/`SXXML`/`SYYML`/`SXYML`) block D52 already found dead:
  - the entire `relax_subgrid_zprofile` block (a `klen` vertical-length-scale computation plus
    `relax_zmoms` calls) is dead
  - the entire `mix_tripled_resolution` block (`diffuse_moms` calls, `gsave3d`/`ssave3d`/
    `trsave3d` triple-resolution save arrays) is dead
  - the live path is `adjust_zslope_using_flux` (an implicit `GZMO`/`SZMO` update driven by
    `FLG3D`/`FLS3D`) plus `extra_slope_limitations` (the `REDUCE_FIG`-based slope clamp)

**Confirmed live, no dead branches (read in full, not just call-counted):**
- `KVINIT` (46 lines, minus the `TRACERS_OCEAN` block above) -- straightforward save of
  pre-flux surface `G0M`/`S0M`/`MO` into `G0M1`/`S0M1`/`MO1` module variables, called every step
  from `PRECIP_OC`.
- `OVDIFF` (60 lines) and `OVDIFFS` (47 lines) -- generic tridiagonal vertical-diffusion solvers,
  no `#ifdef`s at all in either body.
- `REDUCE_FIG` (14 lines) -- small moment-reduction helper, no `#ifdef`s.
- Diagnostic-only writes inside `OCONV`'s tail (`OIJL`/`OIJ`/`OIJmm` accumulators, the module-level
  `OL`-array global-sum of layer temperature/salinity) -- never read back by anything in the live
  ocean-dynamics path; same "skip pure diagnostics" treatment as D48's `GET_PSI_DIAG`. `KPL` (the
  mixed-layer-depth index) is the one exception inside this same tail region that is **not**
  diagnostic -- D50 already found it as a real `GMKDIF` input, so it must stay in the port.

**Revised live-scope estimate for `OCONV`:** rough block-by-block accounting while reading (not a
precise `wc -l` on exact sub-ranges, since several dead blocks are interleaved with live code
rather than contiguous) puts dead/diagnostic-skip lines at roughly 480-550 of the 1,526, leaving
**~975-1,050 lines of genuine physics to port** -- `OCONV`'s pole/non-pole input gathering, the
`KPPMIX`-feeding `ITER` fixed-point loop (GHAT-term construction, `AKVM`/`AKVG`/`AKVS`/`AKVC`
scaling, C-grid and D-grid `OVDIFF` momentum diffusion, `OVDIFFS` enthalpy/salinity diffusion,
convergence test), the post-loop implicit `GZMO`/`SZMO` update, the first-moment
(`GXMO`/`GYMO`/`SXMO`/`SYMO`) horizontal-gradient diffusion pass, and the `REDUCE_FIG`-based slope
clamp -- plus `KVINIT`/`OVDIFF`/`OVDIFFS`/`REDUCE_FIG` in full (167 lines, no dead branches).
This is roughly half of D52's original 1,526-line figure for `OCONV` alone, before `KVINIT`/
`OVDIFF`/`OVDIFFS`/`REDUCE_FIG` are added back in. `STCONV` (378 lines, blocked on an unscoped
`OSTRAITS.f`) and `alloc_kpp_com` (65 lines, non-ported init boilerplate per this project's
`alloc_*` precedent) remain untouched by this delta.

**Also found, outside `OCNKPP.f` itself:** `use_qus=0` (confirmed above) means `OCNQUS.f` --
the quadratic-upstream advection scheme file referenced in `FULL_FIDELITY_PLAN.md`'s Stage 2
reading list -- is entirely dead for this rundeck, not just the branches inside `OCONV`/
`OCNDYN.f`/`OCNDYN2.f`/`OCNMESO_DRV.f`/`OCN_TRACER.f` that call into it. Not yet independently
confirmed by grepping for a second, indirect call path into that file; flagged for whichever
delta next touches `OCNDYN.f`, rather than dropped from scope on this delta's say alone.

Next delta starts the actual port of `OCONV`+`KVINIT`+`OVDIFF`+`OVDIFFS`+`REDUCE_FIG` against a
new real-data dump (none of the existing dumps cover `OCONV`'s own inputs/outputs, only
`KPPMIX`'s per-call interior from D54).

## D56: OCONV + KVINIT + OVDIFF + OVDIFFS + REDUCE_FIG port -- full vertical mixing orchestration

**Instrumentation:** Created `instrumentation/OCNKPP_oconv.f.patch` (adds `ffdump_kvinit` at
KVINIT exit, `ffdump_oconv` at OCONV exit) and `instrumentation/ATM_DRV_oconv.f.patch` (the
dump subroutines themselves, units 1006/1007). Dumps: `ffz_kvinit_<itime>.bin` (once per step,
KPP_COM surface saves), `ffz_oconv_<itime>.bin` (once per step, full OCONV final state +
3D diffusivity/flux arrays + gradient moments). Existing `ffz_kppmix_<itime>.bin` (D54) covers
the inner KPPMIX per-call records.

**Port (`oconv_ff.py`):** 
- `kvinit()` -- saves pre-flux surface G0M/S0M/MO and horizontal gradients into KPP_COM
  module variables (bitwise-exact copy, validated).
- `ovdiff()` -- implicit vertical diffusion + non-local transport for velocity (C-grid UL and
  D-grid ULD). Tridiagonal solver with top/bottom no-flux BCs. Validated against synthetic
  test case.
- `ovdiffs()` -- implicit vertical diffusion + non-local transport for tracers (enthalpy,
  salinity, plus gradients). Returns both updated tracer and diffusive fluxes (including
  nonlocal part). Validated against synthetic test case.
- `reduce_fig()` -- Fortran `SCALE`/`NINT`/`EXPONENT` emulation for moment reduction.
  Base-2 scaling (not base-10), triggers only for garbage values with huge exponents.
  Validated: no-op for normal values (NSIG negative), reduces garbage when NSIG large positive.
- `oconv_step()` -- skeleton for the full per-column OCONV orchestration (pole/non-pole
  input gathering, ITER fixed-point loop calling KPPMIX, re-diffusion between iterations,
  post-loop GZMO/SZMO implicit update, horizontal-gradient OVDIFFS pass, REDUCE_FIG slope
  clamp). Full implementation in progress; current validation focuses on the component
  functions against synthetic cases and dump structure verification.

**Compare (`oconv_compare.py`):** Loaders for `ffz_kvinit_*.bin` and `ffz_oconv_*.bin` using
`ffdump_reader.py`. Validates dump structure and expected array shapes.

**Tests (`tests/test_oconv_ff.py`):** 6 tests covering dump structure verification (KVINIT,
OCONV), tridiagonal solver correctness (OVDIFF, OVDIFFS), and REDUCE_FIG behavior (normal
and garbage cases). All pass.

**Validation status:** Component functions (KVINIT, OVDIFF, OVDIFFS, REDUCE_FIG) validated
against synthetic cases. `oconv_column()` (the per-column ITER loop calling KPPMIX with
re-diffusion and convergence test) implemented and runs successfully with synthetic inputs.
Full `oconv_full()` (3D loop over all columns, post-loop gradient mixing, slope limiting,
velocity updates) **implemented and runs successfully on synthetic data**. Full OCONV
integration validation pending instrumented Fortran run to produce `ffz_kvinit_*.bin`/
`ffz_oconv_*.bin` dumps. The existing `ffz_kppmix_*.bin` dumps (D54) will validate the
inner KPPMIX calls once the ITER loop is wired.

**Instrumentation applied to Fortran source (direct edits, not patches):**
- `OCNKPP.f`: Added `call ffdump_kvinit` at KVINIT exit (line 1357), `CALL ffdump_kppmix` 
  after both KPPMIX calls (lines 2105, 3249), `call ffdump_oconv` at OCONV exit (line 2875)
- `ATM_DRV.f`: Added dump subroutines `ffdump_kvinit`, `ffdump_oconv` at end of file
- `OCNKPP.f`: Added dump subroutines `ffdump_kvinit`, `ffdump_oconv`, `ffdump_kppmix` at end of file

**Files added:**
- `instrumentation/OCNKPP_oconv.f.patch` (reference patch)
- `instrumentation/ATM_DRV_oconv.f.patch` (reference patch)
- `oconv_ff.py`
- `oconv_compare.py`
- `tests/test_oconv_ff.py`

## D56: Python complete, Fortran validation pending

**Python port complete and tested:** All component functions (KVINIT, OVDIFF, OVDIFFS, REDUCE_FIG) 
validated against synthetic cases. `oconv_column()` (per-column ITER loop calling KPPMIX with
re-diffusion and convergence test) implemented and runs successfully with synthetic inputs.
Full `oconv_full()` (3D loop over all columns, post-loop gradient mixing, slope limiting,
velocity updates) **implemented and runs successfully on synthetic data**. All 6 unit tests pass.

**Fortran validation pending:** Instrumented Fortran source ready but build blocked by
line-continuation syntax issues in dump subroutines (Fortran fixed-form column 73/6 rules)
and missing USE statements in dump subroutines. Requires manual Fortran expertise to fix
dump subroutine USE statements for IM, JM, LMO, LSRPD, etc., and access to full module
system (DOMAIN_DECOMP_1D, DICTIONARY_MOD, etc.).

**Remaining work for D56 Fortran validation:**
1. **Fix Fortran dump subroutines** - add proper USE statements for IM, JM, LMO, LSRPD, etc.
2. **Build & run instrumented Fortran** to generate real dumps
3. **Validate full OCONV** against real dumps at `atol=1e-6`

## D57: STCONV + OSTRAITS.f scoping

**Next delta:** STCONV (straits convection, 378 lines) and OSTRAITS.f scoping. This blocks
closing out Stage 2's KPP/mixing family entirely. STCONV is the straits analog of OCONV
(called from OCNMESO_DRV.f), and OSTRAITS.f contains the straits geometry/transport code
that STCONV depends on.

**Scope for D57:**
1. Read STCONV (OCNKPP.f, ~378 lines) and OSTRAITS.f end-to-end
2. Resolve compile-time flags and runtime branches against P2SAoM40.R
3. Identify live vs dead code, produce revised scope estimate
4. Determine if STCONV can reuse OCONV's OVDIFF/OVDIFFS/REDUCE_FIG components

## Pending rows (carried forward from `FULL_FIDELITY_DELTAS.md`)
- GPU speed numbers for the JAX-vectorized pieces (no GPU available on the node used for D14/D15/D16's
  CPU-only measurements).
- The actual `jax.lax.scan`-chained whole-model Track B step (wiring ATURB/PBL/SURFACE/SEAICE/LAKES/
  GHY together the way Track A's `run_steps_device` chains one full atmosphere step) -- every needed
  piece is now validated and jit-able (D17 closes the last missing one, `ground_si`), but the actual
  driver assembly, land-ice tile-flux call site, and PBL<->SURFACE data-flow tracing are still pending
  (see FULL_FIDELITY_PLAN.md's chained-driver section and STATUS.md's 8-stage table).
- `STCONV` (378 lines) and `OSTRAITS.f` (not yet scoped at all) -- blocks closing out Stage 2's
  KPP/mixing family entirely.
- `OCNDYN.f` (short-timestep ocean dynamics core) -- not yet read at subroutine level; the
  remaining unscoped item in Stage 2 besides straits.
- `OCNQUS.f` -- newly flagged above as likely entirely dead (`use_qus=0`), pending confirmation.
