# SOCRATES spike / port workspace (branch `socrates-spike`)

Owner decisions: `Reports/ACCEPTANCE_CRITERIA.md` sections 12 and 12a. Results and honest status: `Reports/SOCRATES_R2_DUMPS.md`
(sections 8-11), `Reports/SOCRATES_R1_REACHABILITY.md`. Nothing here modifies the original SOCRATES/ModelE trees.

## Layout

| Path | What |
|---|---|
| `ffsoc/` | instrumentation of a **scratch copy** of SOCRATES: `ffsoc_mod.F90` (timers + stream dumps, env `FFSOC`, `FFSOC_STRIDE`, `FFSOC_LIGHT`), `apply_ffsoc.py` (edits the scratch copy, refuses `ModelE_Support`), `SOCRATES_ffsoc.diff`, `build.sh.example` |
| `r2_oracle.py`, `r2_parse.py` | run the persistent radiation server from the instrumented binary on recorded packets (bitwise gate on the 21 non-AIJ outputs), read `ffsoc_dump.bin` |
| `spec_data.py` | extract the spectral arrays of the real Fortran reader (tag 100 records) into `spec_{lw,sw}.npz` |
| `gas_absorption.py`, `k_eqv.py`, `kterm.py`, `grey_opt_prop.py`, `cloud_geometry.py`, `planck.py`, `kernels.py`, `triple_column.py`, `solver_triple_hogan.py` | the ported pieces (NumPy/JAX, float64) |
| `band_chain.py`, `radiance_calc_jax.py` | reference chain: model-side inputs -> band fluxes, one column per call (follows the Fortran control flow) |
| `radiance_batch.py` | batched, jit-able formulation (all columns at once, n_cloud_top = 1, per-column major absorber, padded k-terms) |
| `r2_*_check.py` | one oracle comparison per stage (see below); results in `data/` |
| `radiance_bench.py`, `solver_bench.py` | device timing (V100 via `sbatch` on `gpulogin1`) |

## Reproduce

1. Scratch build: copy the model tree and `socrates/{make,src}` to a scratch directory, run `ffsoc/apply_ffsoc.py <scratch socrates> <scratch model/socrates/Makefile>`, build with `build.sh.example` (ifort, `-fp-model strict`; about 12 min).
2. Dumps: `python r2_oracle.py <bin> <rundir> --stride 331 33312 33317 33322` (add env `R2_LIGHT=1` for tags 0/30/100 only, which allows `--stride 1`). The oracle first checks that the instrumented binary still reproduces the recorded server outputs bitwise.
3. `python spec_data.py <dump> <specdir>` once, then the checks: `r2_gas_check.py`, `r2_keqv_check.py`, `r2_kterm_check.py`, `r2_grey_check.py`, `r2_cloud_check.py`, `r2_planck_check.py`, `r2_triple_check.py`, `r2_band_check.py`, `r2_e2e_check.py`, `r2_batch_check.py` (arguments in each docstring).

## Dump tags (file order inside a call: 0, then per band 10, 26, 25, 20, then per k-term 40-49, 21, 22, then 30)

0 call inputs (1 p, 2 t, 3 density, 4 gas mix ratio, 5 mass, 6/7 p/t level, 8-9 classic aerosol, 10 w_cloud, 11 frac_cloud, 12-14 condensate, 15 zen_0,
16 solar irradiance, 17 rho_alb (profile, basis 0 diffuse / 1 direct, band), 18 t_ground, 20 flags, 21-24 cloud parametrisation, 25-30 aerosol modes,
31 counts, 32-36 tiles); 10 grey optical properties of the band; 20 k_eqv, d_mass, zen_0; 21/22 per-k-term flux and weight; 25 band inputs of the
k-eqv solver; 26 `k_esft_layer`, continuum, rescaled gas; 30 band-end fluxes (LW: cumulative over bands); 40-49 per-k-term coefficients, sources, solver
inputs/outputs; 100 spectral data; 900 stage times.

## Known limits

CPU float64 is bitwise or at rounding level against the Fortran; GPU `exp`/`pow` are not bitwise. Only the nov26 radiation steps are available as oracle.
Unported branches and the ModelE-side interface are listed in `Reports/SOCRATES_R2_DUMPS.md` section 10.
