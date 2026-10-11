# SOCRATES R1: reachability under this rundeck's options (decision D3, `ACCEPTANCE_CRITERIA.md` section 12)
Owner of the decision: G. Tamkin. Written by a Claude Code session, 2026-10-11, branch `socrates-spike`. Method: read-only static analysis (`fullfidelity/socrates_spike/r1_reach.py`); no SOCRATES or ModelE file was edited. Review when R2 dumps exist.

## 1. Option set (read from `planet_rad.F90`, `lw_control.F90`, `sw_control.F90`; NOT yet dumped from the running server)
| Option | LW | SW | Source |
|---|---|---|---|
| gas overlap | k-eqv with scaling (`ip_overlap_k_eqv_scl`); bands whose scaling is SES2 are forced to `ip_overlap_mix_ses2` | same | control defaults; `planet_rad.F90` 728-742, 826-834 |
| cloud representation / inhomogeneity / overlap | CSIW / Cairns / max-random | same | control defaults |
| McICA | **not active** (`i_inhom` is Cairns; McICA is rejected with CSIW, `planet_rad.F90` 2105-2112) | same | |
| solver / cloud scheme | `ip_solver_triple_hogan`, `ip_cloud_triple` (scatter_full, max_rand) | same | `set_solver_cloud_scheme`, `planet_rad.F90` 2096-2118 |
| angular integration / two-stream | two-stream, Elsasser | two-stream, PIFM80 | control defaults |
| aerosol | on unless `FTTAER` / `FSTAER` = 0 (`RAD_DRV.f` 403-404) | same | |
| clear-sky diagnostic (`l_clear`) | off unless `cloud_rad_forc > 0` (`RAD_DRV.f` 3731-3732) | same | |
| tiles / rescale | on / on | on / on | |
Status: these are the compiled defaults plus the overrides I found by grep in `model/*.F90`, `*.f` and the rundeck; the plan's own R1 asks for a dump of `control_lw/sw` from the server, which needs the instrumented copy (R2). Assumption flagged, not verified at run time.

## 2. Reached code
| Count | Files | Lines |
|---|---|---|
| radiance_calc, all options, CALL edges only (my walk; the plan's 2026-10-07 walk gave 108 / 38,267) | 112 | 39,082 |
| same, with the blocked set of this rundeck (20 names, `r1_reach.py` `RUNDECK_BLOCK`) | **69** | **23,115** |
| same, also pulling the USEd module files (types, constants; not a porting burden) | 88 | 28,822 |
| lower bound seen by the profiler (`executed_lines.py`, sampled routines only) | about 40 | about 16,000 |
Blocked as not selected: the other band solvers (`k_eqv`, random overlap), McICA and the mix/part-corr/IPA cloud columns, spherical harmonics, `gauss_angle`, the other triple solvers, and `column_solver` with its no-scatter/pentadiagonal/homogen-direct clear-sky solvers. Largest reached: `radiance_calc` 2,587; `grey_opt_prop` 1,479; `solve_band_k_eqv_scl` 1,104; `opt_prop_ice_cloud` 991; `solve_band_ses` 824; `opt_prop_aerosol` 794; `opt_prop_water_cloud` 748; `solve_band_one_gas` 679; `solve_band_without_gas` 632.
Not included: the ModelE side (`planet_rad.F90` 3,866 lines, `RADIATION.f`: aerosol set-up, `interp1d`, albedo, diagnostics), and the spectral-file reader (static data, converted once on the host). Function references without `CALL` are matched by name only.

## 3. Limits that matter
1. The pruning is mine: a routine is removed when it is reached only through a `CALL` I decided is not selected. A wrong decision shows as a missing routine; the sampled lower bound (about 16,000 lines) lies below the pruned figure (23,115), which is consistent but not a proof.
2. `column_solver` is blocked because its callers run only when `l_clear` is true; when `cloud_rad_forc > 0` it, `solver_homogen_direct` and `solver_no_scat` come back (about 1,000 lines). Kernel B of the spike used `solver_homogen_direct`; for this rundeck's default path the triple solver is the one that runs.
3. `i_solver_clear` is left undefined (`imdi`) by `planet_rad.F90` for the CSIW path; harmless while `l_clear` is false, fatal otherwise (`column_solver` line 202-206). Not tested.
4. Line counts include comments and blank lines (the same basis as the plan).

## 4. Effect on the estimate (judgement, not a measurement)
Reached executable code is 23,115 lines against the plan's static 34,899 (66 %). Scaling only R3-R5 (the code-size-driven stages, 70-155 h) by 0.66 gives 46-103 h, so the total moves from 135-295 h to about 110-245 h (central about 175 h). The plan's caveat that estimates may be 1.5x low stays; R6 (ModelE side) is not reduced.

## 5. Spec of the R2 dumps (to be confirmed against the argument lists when the copy is built)
Dump at the interfaces already named in the spike, per band and per profile chunk, written big-endian float64 as the existing ADVSI dumps:
1. `gas_optical_properties`/`solve_band_k_eqv_scl`: k-term optical depths per layer (the gas absorption that is assumed, not measured today), with the band, k-term index and scaling.
2. `grey_opt_prop`: total extinction, single-scattering albedo, asymmetry per layer and cloud region (strat/conv), plus the aerosol and cloud parts (kernel A's oracle).
3. `two_coeff_region` / `solar_coefficient_basic` / `trans_source_coeff`: transmission, reflection, source terms per region (kernel B's real inputs and outputs).
4. `solver_triple_hogan`: band flux up/down per level.
5. Final band-summed fluxes and heating rates (to reproduce the server's recorded outputs bitwise, the R2 gate).
Packets: the 11 recorded nov26 steps `ff_data/nov26_day/rsv_n26_<it>_{in,out}.bin`; the plan's 3 dates are not all recorded (dec01 and jan01 have one radiation step each).

## 6. Reproduce
`cd fullfidelity/socrates_spike && python3 r1_reach.py --calls-only --rundeck --list`   (add `--calls-only` off to include USEd modules).
