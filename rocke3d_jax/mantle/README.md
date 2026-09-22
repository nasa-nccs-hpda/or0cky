# mantle/

Historical and superseded files from this project's earlier iterations,
physically separated (2026-09-22) from the current, active port + validation
artifacts one level up. Nothing here is deleted — it's kept because deleting
working code/data that might still have reference value is a bigger
irreversibility than a messy directory. See `STATUS.md` (one level up) for
what's actually current.

Before moving anything into or out of this directory again, grep the rest of
the project for the exact filename first — that's the mistake this
consolidation was explicitly trying to avoid repeating (see the
`rocke3d_jax-refactor-pending` memory note if you have access to it).

## What's here and why it moved

- **Old standalone Fortran test drivers for PBL/DRYCNV**
  (`test_pbl_fortran*`, `test_pbl_short*`, `test_pbl_simple_fortran*`,
  `test_drycnv_fortran*`) and **old timing drivers**
  (`timing_{drycnv,pbl}_p2saom40*`) — superseded by the unified, more rigorous
  `compare_fortran.f90` / `compare_jax.py` pipeline one level up, which is
  what STATUS.md's Accuracy/Performance numbers for PBL and DRYCNV actually
  come from. (The equivalent test drivers for FLUXES/GHY/SURFACE/RADIATION/
  SEAICE/LAKES/ATURB were **not** moved — those remain each module's only
  validation and are still current, one level up.)
- **Early benchmark scripts** (`benchmark_all.py`, `benchmark_all_cpu.py`,
  `benchmark_all_p2saom40.py`, `benchmark_p2saom40_vs_fortran.py`,
  `benchmark_single_layer.py`) — the NumPy-relative-speedup comparisons
  STATUS.md explicitly recommends dropping from the headline story. Kept as
  engineering scratch work, not cited in current status reporting.
- **Old end-to-end scripts** (`run_end_to_end*`) — superseded by
  `p2saom40_driver.py` / `p2saom40_compare.py`.
- **Earlier-generation Fortran-vs-JAX compare tooling**
  (`compare_fortran_jax.py`, `compare_fortran_jax_notebook.ipynb`,
  `compare_fortran_jax_visual.py`, `validate_fortran_jax.py`,
  `convert_fortran_to_npy.py`) — predecessor of the current `compare_*.py`
  pipeline.
- **Old visualization/dashboard notebooks and their scripts**
  (`visualize_results.py`/`.ipynb`, `visualize_single_layer_outputs.ipynb`,
  `single_layer_pixel_map.py`, `visualize_2d_global_maps.ipynb`,
  `executive_summary_dashboard*.ipynb`, `1_scientific_maps.ipynb`,
  `2_performance_bar_plots.ipynb`, `3_comparison_plots.ipynb`,
  `hybrid_workflow.py`, `plot_3d_earth.py`, `md_to_pdf.py`) — superseded by
  `visualize_p2saom40_kernel_maps.ipynb` one level up, which uses P2SAoM40's
  real grid instead of synthetic single-layer placeholders.
- **Orphaned NetCDF files** (`ANN4000.aijP2SAoM40.nc`,
  `ANN4299.aijP2SNoM40.nc`) — not referenced by any current or historical
  script; `ANN4099.aijP2SAoM40.nc` (one level up) is the one actually in use.
- **Root-level scratch `.bin`/`.npy` files** (`fortran_*`, `jax_*`,
  `*_end_to_end_output.txt`) — predecessors of the `compare_data/` pipeline,
  consumed only by `visualize_results.py` (also here).
- **`outputs/`** — the subset of the top-level `outputs/` directory generated
  by the old notebooks above (`single_layer_pixel_map.py` →
  `visualize_2d_global_maps.ipynb` / `visualize_single_layer_outputs.ipynb`).
  The current `p2saom40_kernel_*` and `p2saom40_jax_*`/`p2saom40_itype_map`
  outputs stayed in the top-level `outputs/`.

## Dependency check performed before this move (2026-09-22)

Every file above was confirmed, by grepping every `.py`/`.ipynb`/`.f`/`.f90`/
`.md`/`.yaml`/`.sh`/`.sbatch` file in the project, to have **no live
import or path-load dependency** from any file that stayed at the top level.
A few were mentioned in prose/markdown (a "see also" in a notebook cell, a
comment in `compare_generate_inputs.py`, mentions in `STATUS.md`/
`README_GPU.md`) — those mentions were updated to point at `mantle/...` paths
rather than left dangling.
