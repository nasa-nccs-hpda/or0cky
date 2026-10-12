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
| `planet_side.py`, `radia_cols.py`, `zenith_coszs.py`, `radia_jax.py`, `radia_native.py` | R6: planet_rad set-up and read-out, RADIA/RCOMPX set-up (albedos, gases, aerosols), COSZS, the whole radiation step as one jit, and the `NativeHandoff` drop-in for the server callback (report section 12) |
| `r6_collect.py`, `r6_stream.py`, `r6_static0.py`, `r6_stageb_check.py`, `r6_e2e_check.py`, `r6_stagea_check.py`, `r6_radia_check.py`, `r6_native_check.py`, `radia_bench.py` | R6 reading of the dumps (streaming; per-column npz) and the checks; the V100 benchmark |
| `daily_products.py`, `extract_albvnd.py`, `data/albvnd.npz` | the daily products for any calendar day from the rundeck's input files (ozone `ppm`, aerosol / dust / BC `linm2m`, REPART, eps, GHG, vegetation albedo, RSDIST) and the volcanic file (`VolcanicAerosol`, `VolcanicTables`, with `hlb0=`), and the helpers to hand them to `RadiaJax(day_products=..., day0=...)` or, on demand with no window (CPU only: ~6-9 s per radiation step on the V100), `RadiaJax(day_source=DayStream(...))` + `NativeHandoff(..., zen_source=ZenithStream(...))` (report sections 14, 14.4) |
| `segments.py`, `r6_segments_check.py` | day-aligned segments (`segment_plan`, `build_segment`) for runs longer than a few weeks on a GPU: day products and zenith table per segment, one compilation each (report 14.5); `tests/test_socrates_daily_products.py` unit-tests the reader |
| `r6_days_oracle.py`, `r6_days_check.py`, `r6_days_radia_check.py`, `r6_days_e2e_check.py` | oracle for other dates (the persistent server steps its clock through the day boundaries and dumps / returns its outputs), reader-vs-dump check, nov27 check against the recorded model step, whole-step check at 38 dates |
| `ffsoc/apply_ffsoc_r6.py` | second instrumentation stage (planet_rad, RADIATION.f, RAD_DRV.f, planet_alb.F90 of the SCRATCH model copy; `FFSOC_PL=1`) |
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

## R6 records (planet_rad side and RADIA statics; written with `FFSOC_PL=1`, independent of the stride; layouts in `planet_side.py` PACK50/60/51)

50 arguments and module inputs of `run_planet_rad` (p_level 44, t_layer 43, t_level 44, ULGAS 43x13, CLDEPS 43, PRNB 6x4, PRNX 6x4, cos zenith, S0, 15 cloud arrays x 43, cld_scaling, n_aer, n_aer_tracer, rel. humidity 43, wind, 4 fractions, 4 temperatures, soil wetness, PVT 12, snow fraction/depth land 2+2, sea-ice snow depth, snow amount); 51 outputs of `get_planet_radout` (TLB, TLT, TRUFLB, TRDFLB, TRNFLB, TRFCRL, SRUFLB, SRDFLB, SRNFLB, SRFHRL, FSRNFG, 15 scalars, TRUPTOA 9, SRUPTOA 6, SRDNTOA 6, cloud cover); 52 / 53 planet aerosol set-up (component, radius, sulphate fraction, SW/LW scaling; mass/density); 54 SW albedo weights; 60 RCOMPX state (see report 12.2); 61 / 62 dust columns, TAUCON; 101 / 102 control flags LW / SW; 103-111 band limits and exclusions, blue weights; 112 / 113 cloud-parameter limits; 120-130 / 140-150 planet aerosol tables LW / SW; 170-183 RADIA statics (byaml00, plb0, shl0, RSDIST, GTAU, SALBTG, TAUGSA, TAUTGS, REDUST, RODUST, ALBVNH, EPLOW, EPMID, EPHIG); tag 30 k = 3 direct flux, k = 4 tile upward flux.

R6 reproduction: apply `ffsoc/apply_ffsoc_r6.py SCRATCH_SOCRATES SCRATCH_MODEL` after `apply_ffsoc.py`, rebuild (the model part takes 1-2 min after the first full build), `R2_LIGHT=1 R2_PL=1 python r2_oracle.py BIN RUN --stride 1000000000 33312 33317 ...`, `python r6_collect.py RUN/ffsoc_dump.bin out.npz 3170` (one npz per step), `python r6_static0.py ...`, then the `r6_*_check.py` scripts and `radia_bench.py` (arguments in each docstring). The scripts import `fullfidelity/clouds_jax_env.py` (XLA:CPU without FMA contraction, `R6_XLA_FLAGS=0` to skip): without it a threshold of the snow/ice albedo flips in about 0.1 % of the boxes (report 12.4).

## Known limits

CPU float64 is bitwise or at rounding level against the Fortran; GPU `exp`/`pow` are not bitwise. Recorded model steps exist only for nov26 and the first step of nov27; other dates are checked against the persistent server (section 14).
Unported branches are listed in `Reports/SOCRATES_R2_DUMPS.md` sections 10 and 12.6; the daily products of other dates come from `daily_products.py` (sections 14 and 14.4; verified for master_yr 1850 and, with the volcanic file, for master_yr 1990 / volc_yr 0; land cover `PVT` is not read).
| `daily_products.LandCover`, `r6_land_cover_check.py` | `PVT` / `frac_veg` of the master year from the VEG and CROPS files (exact on 753 land columns, 1850 and 1990); opt-in |
| `r6_default_path_check.py` | hashes of day products, zenith table, jaxpr and two radiation calls: 24fc300 vs 373da9d default path (report 14.7) |
