# Restart notes — ROCKE-3D → JAX session

**Written**: 2026-09-22, end of session, as a handoff for resuming later
(same or new Claude session). Read this first, then `STATUS.md` for the
actual project content — this file is only "where things are," not "what we
found."

**2026-09-23 update — the "faithful full-chain port" is CLOSED, not
paused. Read this before reopening it.** Attempted to start this
(sub-tiling + `GHY` + `ATURB`), scoped it in real depth, and then the user
explicitly closed it out. Sequence of findings, in case the reasoning
matters later:
1. `GHY` looked like the easiest piece (its JAX file looked most complete)
   — it wasn't. Reading it in full: only `compute_sensible_heat` is real;
   evap/runoff/soil-properties/snow-melt are explicitly placeholder in the
   source, and the Fortran "validation" test was a hand-written stub
   commented "matching JAX logic" with its untested outputs hardcoded to
   zero. Corrected STATUS.md's "14/17 faithful" claim and GHY's module-table
   entry accordingly — don't trust the old claim if you see it cached
   anywhere.
2. Real Fortran source exists for all three
   (`modelE2_planet_2.0/model/{GHY_DRV,ATURB,SEAICE}.f` / `giss_LSM/GHY.f`,
   ~11,700 lines combined) — read `get_soil_properties` (25 lines) and
   `runoff` (112 lines) from the real `giss_LSM/GHY.f` in full. Confirmed the
   real multi-layer soil state GHY needs (`w_ij`/`ht_ij`, shape (46,72,3,7))
   exists in `1JAN1950.rsfP2SAoM40.nc` — a fuller restart file in the same
   P2SAoM40 directory this project had never used (the `fort.1.nc` restart
   this project reads throughout only has a bulk `wearth` field, not the
   layered state). So this was **not blocked by missing data** — genuinely
   portable if someone wanted to.
3. But `snow` (in GHY.f) just delegates to an entirely separate module
   (`snow_drvm`), and `runoff` depends on state (`xinfc`, `xku`) computed by
   *other* GHY subroutines earlier in a specific sequence — GHY is a coupled
   land-surface model, not 5 independent functions. Faithfully porting it
   means porting its internal time-stepping orchestration (`advnc`, 600+
   lines) too.
4. **The user then reframed the actual goal**: a representative workflow
   across Fortran/JAX-CPU/JAX-GPU-original/JAX-GPU-optimized that gives
   ~the same answers at each stage — not maximizing scientific fidelity to
   Fortran. That property already holds today, fully validated (Phase 1 was
   a pure dispatch restructuring, so all three JAX legs compute identical
   physics). Porting GHY/ATURB would change what JAX computes, requiring a
   fresh accuracy re-validation before any leg could be trusted again — work
   that doesn't serve the stated goal. **Decision: closed, not pursuing.**
   STATUS.md's Recommendation and Open items, and both the live slide deck
   and `status_slides/`, were updated to reflect this closure explicitly
   (not left as a dangling "recommended next").

Do not reopen this without the project's actual goal changing first — the
domain-judgment case for SEAICE/ATURB (kept in STATUS.md's Recommendation
section) was never wrong, it's just not what this project is optimizing
for. Full detail: STATUS.md's "A validation-methodology gotcha" (second
instance) in Accuracy, and the "Recommendation: the 3 remaining modules —
closed out" section.

**Post-reorg regression check (2026-09-22, same day)**: after the `mantle/`
move below, everything was re-run to confirm nothing broke — 104/104 unit
tests, all 6 per-module Fortran comparisons (recompiled fresh), the
PBL/DRYCNV kernel pipeline, the current visualization notebook (executed
end-to-end, zero errors), and the full physics-chain driver against the
real restart data. Everything reproduced. Two real (small) bugs found and
fixed in the process — see STATUS.md's "Regression verification" section
for details. Nothing here needed re-litigating as a result.

**GPU optimization, same day, later**: profiled the full-chain driver
(`run_dtsrc_step`), found it dispatching ~33 separate JIT calls per step,
and fused the whole hot path into one `@jax.jit` function
(`p2saom40_driver.py`'s `_step_core`) — see "GPU optimization: Phase 1" in
STATUS.md. Verifying this on a real GPU (discover cluster A100) surfaced a
**second, independent bug**: `p2saom40_compare.py`'s "Performance (CPU)"
section never actually forced the CPU backend, so on a GPU node it silently
timed the GPU too — meaning the project's headline "~1.0×, no GPU benefit"
full-chain finding, repeated across STATUS.md, README_GPU.md, and the slide
deck, was comparing the GPU to itself, not a real CPU baseline. Fixed (the
script now spawns a genuine CPU-only subprocess when a GPU is already
active) and re-measured: **17.94 ms genuine CPU → 4.26 ms GPU, a real 4.2×
speedup**, clearing the project's own >2× target. All of STATUS.md,
README_GPU.md, and the slide deck (both the live artifact and
`status_slides/`) have been corrected with the real numbers — not just
flagged, since the underlying files this correction depends on
(`EXECUTIVE_SUMMARY.md` etc.) no longer exist to flag.

**Verification pass, next day (2026-09-23)**: user pushed back on the 4.2×
number before accepting it — reasonably, since this project had *just* found
one measurement bug in this exact chain. Did three checks rather than just
reassuring: confirmed the CPU-bug fix never touched the GPU-timing code path
(diffed line-by-line), confirmed the GPU number reproduces across two
independent runs (4.08 ms, 4.26 ms), and confirmed the magnitude is
physically plausible for this grid size and hardware. Also surfaced and
documented a real scope caveat that wasn't previously called out explicitly:
the 62×/14.7× figures compare JAX's *simplified* full-chain physics (no
sub-tiling, no `GHY`, no real `aturb` turbulence solve) against Fortran's
*fuller* implementation of that same scope — the numbers are real, but a
faithful full port would likely see a smaller ratio. All of this is now in
STATUS.md's "Full Physics Chain" section and a new 5th slide, "Verification"
(`journey.html`) — added to the live deck and `status_slides/build_pdf.py`.
Standing lesson for next time a big number shows up right after a bug fix:
verify before presenting, the same way this session verified before
believing it.

## Where the real content lives

- **`STATUS.md`** (this directory) — the single source of truth for project
  status: goal, accuracy, performance, the Full Physics Chain finding,
  recommendation, gotchas, open items. Consolidated from 5 now-deleted legacy
  docs (`EXECUTIVE_SUMMARY.md`, `FINDINGS.md`, `PORTING_STATUS.md`,
  `SESSION_SUMMARY.md`, `SUMMARY.md`) — if any of those filenames come up
  again, they're gone on purpose; content merged into `STATUS.md`, originals
  still in git history.
- **Slide deck**: https://claude.ai/artifact/LxdYuYeQXxHDsX18KWxBLA
  ("ROCKE-3D → JAX: Status," 5 slides — added a "Verification" slide for the
  GPU-optimization result). This is now the *only* deck for this
  project — a second, narrower one was merged into it and deleted.
- **Filesystem PDF copy**: `status_slides/status_deck.pdf` (source:
  `status_slides/build_pdf.py`, reportlab — not weasyprint, see
  `status_slides/README.md` for why). Rebuild with:
  `cd status_slides && python build_pdf.py status_deck.pdf`
  (the script defaults to `deck.pdf` if you omit the argument — don't forget
  it, or you'll create a stray file).
- **`README_GPU.md`** — kept (operational Docker/Slurm setup + troubleshooting,
  not status narrative), figures patched to match `STATUS.md`, benchmark
  script paths patched to `mantle/benchmark_all.py` (see below).

## Repository layout (reorganized 2026-09-22)

`projects/imvi/rocke3d_jax/` now separates current port artifacts from
historical/superseded ones:

- **Top level** — the active JAX port (`*_jax.py`, `pbl.py`, `drycnv.py`,
  `radiation.py`), its unit tests (`test_*_jax.py`, `tests/`), the current
  Fortran-vs-JAX validation pipeline (`compare_*.py`/`.f90`, `compare_data/`,
  the per-module `test_*_fortran*` pairs for FLUXES/GHY/SURFACE/RADIATION/
  SEAICE/LAKES/ATURB — these are each module's *only* validation, not
  superseded), the full-chain driver (`p2saom40_*.py`), the current
  visualization notebook (`visualize_p2saom40_kernel_maps.ipynb`), current
  `outputs/` (the `p2saom40_kernel_*` and `p2saom40_jax_*`/`p2saom40_itype_map`
  files), `STATUS.md`, `README_GPU.md`, this file, `status_slides/`,
  `config_rocke3d2.yaml`, `Dockerfile.gpu`, and the current restart file
  `ANN4099.aijP2SAoM40.nc`.
- **`mantle/`** — everything superseded: old standalone PBL/DRYCNV Fortran
  test drivers (superseded by `compare_fortran.f90`), old timing drivers, old
  benchmark scripts (`benchmark_all.py` and related — the NumPy comparisons
  `STATUS.md` says to stop citing), old end-to-end scripts, an
  earlier-generation compare pipeline, old dashboard notebooks
  (`visualize_2d_global_maps.ipynb`, `executive_summary_dashboard*.ipynb`,
  etc.), two orphaned NetCDF files, ~26 root-level scratch `.bin`/`.npy`
  files, and the `outputs/` files those old notebooks generated. Full list
  and rationale: `mantle/README.md`.

**How this was done safely**: before moving anything, a full grep-based
dependency map checked every `.py`/`.ipynb`/`.f`/`.f90`/`.md`/`.yaml`/`.sh`/
`.sbatch` file in the directory for references to each candidate filename.
Nothing moved was imported or path-loaded by anything that stayed. A handful
of prose/markdown mentions (a notebook "see also," a code comment, a couple
of `STATUS.md`/`README_GPU.md` lines, some `FINDINGS.md`-pointer dead links
left over from the earlier doc consolidation) were found and fixed to point
at the new `mantle/...` paths or at `STATUS.md`. All four touched Python
files (`compare_generate_inputs.py`, `p2saom40_compare.py`,
`p2saom40_driver.py`, `p2saom40_io.py`) were syntax-checked after editing.

## Persistent memory (survives across sessions, not just this one)

At `/home/gtamkin/.claude/projects/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai/memory/`:

- `rocke3d_jax_refactor_pending.md` — **now done** as of 2026-09-22 (this
  session). If it still says "still open," that's stale; the reorg described
  above is what it was tracking.
- `rocke3d_jax_doc_consolidation.md` — feedback memory: this user wants
  stale/duplicate docs and artifacts actively deleted once they drift out of
  sync, not preserved "as-is" for history. Relevant if a similar situation
  comes up again in this or another project.

## Git state as of this write (branch `main`, last commit `4072dd9 "GPU update"`)

**Nothing from today's work is committed** — all staged, since committing
wasn't requested. `git status --short` for this directory shows, at a high
level:

- Staged deletions: `EXECUTIVE_SUMMARY.md`, `EXECUTIVE_SUMMARY.html`,
  `FINDINGS.md`, `PORTING_STATUS.md`, `SESSION_SUMMARY.md`, `SUMMARY.md`
- Staged additions: `STATUS.md`, `status_slides/` (build_pdf.py, deck.json,
  slides/*.html, status_deck.pdf, README.md), `mantle/` (~106 files, see
  `mantle/README.md`)
- Staged renames/moves: everything now under `mantle/` (git recorded these as
  delete-old + add-new; `git log --follow` still traces history through them)
- Staged modifications: `README_GPU.md`, `.gitignore` (added `*.mod`),
  `compare_generate_inputs.py`, `p2saom40_compare.py`, `p2saom40_driver.py`,
  `p2saom40_io.py` (all dead-link/path fixes only, verified to still compile)
- **Also staged, pre-existing and NOT part of today's doc/reorg work**: real
  working files from the P2SAoM40 kernel-map visualization effort
  (`visualize_p2saom40_kernel_maps.ipynb`, `p2saom40_compare.out`, ~35
  `outputs/p2saom40_kernel_*_2d_map.html` files) — these were untracked
  before and got swept in by a broad `git add -A` during the reorg. Nothing
  about their *content* was touched; they were just added to the index. If
  resuming with intent to commit, consider whether these belong in the same
  commit as the reorg/doc work or a separate one.

If resuming with intent to commit: review the full diff first (not just this
summary — verify nothing unexpected crept in), and ask before committing
unless already told to.

## What's actually done vs. still open

**Done this session**: real Fortran P2SAoM40 GCM build/run (completed its
full 1-year integration), real Fortran-vs-JAX kernel comparison (CPU+GPU),
real full-physics-chain GPU comparison (rolled in from a parallel session),
`visualize_p2saom40_kernel_maps.ipynb`, the slide deck + PDF, the doc
consolidation (5 docs → 1, deck merge, stale-figure cleanup throughout), the
`mantle/` file reorg (dependency-mapped first, then executed), and the GPU
optimization work (dispatch fusion + fixing the CPU/GPU measurement bug) —
full chain now genuinely 4.2× faster on GPU, target cleared.

**Still open** (see `STATUS.md` "Open items" for the technical ones):
1. SEAICE + ATURB placeholder physics — recommended next, not started
   (explicitly deprioritized this session — user said "we will not do 1-3"
   when offered this alongside the reorg, choosing the GPU optimization work
   instead).
2. Wind-speed convention bug — fixed locally in `p2saom40_driver.py`, not in
   the shared module files. (Also deprioritized this session.)
3. Real spectral radiation (SOCRATES) and atmospheric dynamics/`CONDSE`
   remain entirely unported/untimed. (Also deprioritized this session.)
4. ~~The `mantle/` file reorg~~ — **done**, see above.
5. ~~Get the full physics-chain GPU speedup above 2×~~ — **done**, see above
   and STATUS.md's "GPU optimization: Phase 1" section.

## How to resume cold

1. Read `STATUS.md` in full — it's self-contained.
2. Check the two memory files above for standing preferences (the refactor
   one should now say done — if not, this file is more current).
3. If the user references "the deck" or "the slides," it's the single
   artifact URL above — don't create a new one.
4. If you need something that used to be at the top level and isn't there,
   check `mantle/` and `mantle/README.md` before assuming it was deleted —
   almost everything from today's reorg was moved, not deleted (the doc
   consolidation deleted 6 files outright; the reorg moved ~106).
