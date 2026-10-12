# SOCRATES R2: instrumented scratch copy, intermediate dumps, stage times (decision D3, `ACCEPTANCE_CRITERIA.md` section 12)
Written by a Claude Code session on 2026-10-11, branch `socrates-spike`. Owner of the decision: G. Tamkin. The original SOCRATES and ModelE trees were not touched; all edits are in a scratch copy (`<scratchpad>/socdump`), the edit is committed as a diff.

## 1. Result against the R2 gate
**Gate: the instrumented build reproduces the server's recorded outputs bitwise. MET.** Control build (scratch ModelE copy + `ATM_DRV_radsrv_persist.f.patch` + SOCRATES taken from the scratch copy, no dump code) and instrumented build were each served recorded packets `rsv_n26_33312` and `rsv_n26_33317` (`ff_data/nov26_day`): all 21 output fields except `AIJ` are bitwise equal to the recorded output in both builds (`r2_oracle.py`). `AIJ` is compared against the recorded rounded delta, as the existing oracle does: max abs difference 6e-11 (33312) and 1.0 (33317, the rounding step of the recorded delta); not a bitwise test. The instrumentation adds only timers and stream writes, no arithmetic.

## 2. What was added (`socrates_spike/ffsoc/`)
* `ffsoc_mod.F90`: module with stage timers (`SYSTEM_CLOCK`, every call) and big-endian stream dumps to `ffsoc_dump.bin` (unit 1700) for every `FFSOC_STRIDE`-th `radiance_calc` call (env `FFSOC`, `FFSOC_STRIDE`; off when `FFSOC` is unset).
* `apply_ffsoc.py` + `SOCRATES_ffsoc.diff`: the edits of `radiance_calc.F90` and `solve_band_k_eqv_scl.F90` (131 diff lines) and one line in the scratch `model/socrates/Makefile`. Refuses to run on a path containing `ModelE_Support`.
* `r2_oracle.py`, `r2_parse.py`, `build.sh.example`. Build about 15 min serial (control), instrumented incremental 3 min; run 12 s per radiation call.

Dump tags (per `radiance_calc` call, one profile per call, 43 layers; LW 9 bands, SW 6 bands):
| Tag | Where | Content |
|---|---|---|
| 0 | start of `radiance_calc` | `atm%p`, `t`, `density`, `gas_mix_ratio` |
| 10 | after `grey_opt_prop`, per band | `k_grey_tot_clr`, `k_ext_scat_clr`, `k_grey_tot`, `k_ext_scat` (kernel A's oracle: aerosol/cloud/Rayleigh grey optical properties) |
| 20 | `solve_band_k_eqv_scl`, after the k-eqv build | `k_eqv`, `d_mass`, `zen_0` (the gas-absorption result: the 27 % that was assumed) |
| 21 / 22 | after `monochromatic_radiance`, per k-term | `flux_total_part` (2 x layers + 2), ESFT weight |
| 30 | end of each band | cumulative `flux_down`, `flux_up` of the channel |
| 900 | end of call | 6 stage times and call counts |
Added in section 7: the per-k-term two-stream coefficients, sources and solver result (tags 40-44). Still not dumped: `k_abs_layer` itself (the spectral lookup), the inputs of `grey_opt_prop` beyond tag 0, and the cloud overlap coefficients (`cloud_overlap`, `w_free`).

## 3. Stage times of one radiation step (first radiation step of nov26, one request, instrumented run; timers add overhead of two clock reads per timed region)
| Stage | LW s (3,170 calls) | SW s (1,826 calls) | Both s | Share of `radiance_calc` |
|---|---|---|---|---|
| 1 `grey_opt_prop` (aerosol, cloud, Rayleigh optical properties) | 0.515 | 0.201 | 0.72 | 11 % |
| 3 gas absorption to k-eqv, `solve_band_k_eqv_scl` only | 0.261 | 0.067 | 0.33 | 5 % |
| 4 `monochromatic_radiance` (two-stream coefficients + triple solver), `solve_band_k_eqv_scl` only | 2.721 | 0.815 | 3.54 | 55 % |
| 5 `augment_radiance` (+ tiles), same | 0.137 | 0.052 | 0.19 | 3 % |
| rest of the band loop (band set-up, rescale, bands run by `solve_band_ses` / `one_gas` / `without_gas`, diagnostics) | 1.353 | 0.245 | 1.60 | 25 % |
| outside the band loop | 0.076 | 0.043 | 0.12 | 2 % |
| **`radiance_calc` total** | 5.062 | 1.423 | **6.49** | 100 % |
The server's `RADIA` call took 12.2 s on this host (loaded shared node `forest204`, not the exclusive benchmark node), so `radiance_calc` is 53 % of the server time and the ModelE side plus I/O the other 47 % (5.7 s). The counts: 3,170 LW calls = 72 x 44 + 2 poles; 1,826 SW calls = lit profiles.

## 4. What this changes in the spike's projection (`SOCRATES_SPIKE_RESULT.md` section 5)
* The part that was ASSUMED (gas absorption, continua, cloud logic, 27 % of the call) now has a measured split: gas absorption to k-eqv is small (5 %); the weight is the two-stream coefficients and solver per k-term (55 %) and the per-band overhead (25 %, partly bands that go through the other band solvers). The two kernels prototyped in the spike (A, B) therefore cover the dominant part (A about 11 %, B part of the 55 %); the part to port that was not prototyped is mostly per-band control flow and the solvers of the ses / one-gas bands, not gas absorption.
* The ModelE side is 47 % of the server call in this measurement (the spike's profile gave a smaller share of ModelE-side code, 7.5 % for `interp1d`/`radpar`/`radia_`, plus library and I/O groups): not reconciled; a different host load and the timer set differ. It has to be ported or kept regardless.

## 5. Limits
1. Timers: wall clock (`SYSTEM_CLOCK`) on a shared loaded host; one request; two clock reads per region (several million per call: overhead not measured, but stage 6 equals the sum, and the server time was 12.0-12.2 s with and without the dumps).
2. (Resolved in section 7: no band goes through `solve_band_ses`, `one_gas` or `without_gas`; the "rest of the band loop" is code inside `radiance_calc` itself.)
3. Dumps cover every 331st call (about 30 columns) in the 3-request file; full-column dumps were not made (file size about 1 MB for stride 100000, 5.5 MB for 3 requests at stride 331, scales linearly).
4. Packets: nov26 33312, 33317 and (section 7) 33322 were served; the dec01 and jan01 packets (other seasons, one radiation step each) were not.
5. The control/instrumented ModelE binary is built from the repo's own patch and the copy; the original persistent binary was not byte-compared.

## 6. Reproduce
```
# build (scratch copy; SOCRATESPATH must point to the scratch socrates copy)
rsync -a --exclude=ModelE_Support <modelE2_planet_2.0>/ <scr>/mE2/ ; rsync -a <ModelE_Support>/socrates/{make,src} <scr>/socrates/
patch <scr>/mE2/model/ATM_DRV.f < fullfidelity/instrumentation/ATM_DRV_radsrv_persist.f.patch ; chmod -R u+w <scr>
python3 fullfidelity/socrates_spike/ffsoc/apply_ffsoc.py <scr>/socrates <scr>/mE2/model/socrates/Makefile   # instrumented only
./build.sh   # see build.sh.example
# run and score (graphcast-env)
RADSRVP_SCRATCH=<scr> python fullfidelity/socrates_spike/r2_oracle.py <bin> <rundir> --stride 331 33312 33317
python fullfidelity/socrates_spike/r2_parse.py <rundir>/ffsoc_dump.bin --arrays
```

## 7. Addendum (same day): per-k-term dumps and the split of the remaining 25 %
Added on request of the owner; the build is again bitwise: packets **33312, 33317 and 33322**, all 21 non-AIJ fields equal to the recorded outputs (instrumented binary, stride 331).

New dump tags (all inside `triple_column`, the routine that `monochromatic_radiance_tseq` calls for this rundeck's cloud scheme; written once per k-term, just before the `tag 21` record of that k-term):
| Tag | Content |
|---|---|
| 40 (k=1..9) | `ss_prop%tau_clr`, `omega_clr`, `tau`, `omega`, `tau_clr_noscal`, `tau_noscal`, `phase_fnc_clr`, `phase_fnc`, and `sec_0`: the exact inputs of `two_coeff_region` |
| 41 (k=1..5) | `trans`, `reflect`, `trans_0_noscal`, `trans_0`, `source_coeff` (layers x regions): the output of the two-stream coefficient routine, the oracle for kernel B's coefficient part |
| 42 | `s_down`, `s_up` (after the cloud-region weighting) and `source_flux_ground`: the inputs of `solver_triple_hogan` |
| 43 | `flux_total` as returned by the solver; checked equal bitwise to the tag 21 record that follows (3,634 pairs, 0 differences) |
| 44 | `n_cloud_top`, `n_region`, `n_cloud_type`, `n_layer`: needed to know which entries of tags 41/42 are valid (strat/conv entries above the cloud top are uninitialised memory in the Fortran and are garbage, up to 1e307, in the dump) |
| 46 (k=1..3) | `cloud_overlap(:, 0:n_layer, 1..18)` (v11..v33, u11..u33 of the triple solver), `flux_inc_down`, `diffuse_albedo` (added for the solver check, section 9). |
Size: 74 MB for 3 requests at stride 331 (about 30 columns per request, LW and SW, 14,988 timed calls); 80,000 array records. Sanity: on the valid entries all transmissions and reflections lie in [0, 1] (7,268 records, 0 outside).

Split of the former 25 % (3 requests, 14,988 `radiance_calc` calls, 21.9 s of timed `radiance_calc`; the per-call instrumentation overhead grew with the dumps):
| Part | Share |
|---|---|
| `solve_band_k_eqv_scl` as a whole (every band: stages 7-9 are zero, so `solve_band_ses`, `one_gas` and `without_gas` are never called for this rundeck) | 66.8 % |
| of which `monochromatic_radiance` (two-stream coefficients + triple solver) | 57.6 % of the call (86 % of `solve_band_k_eqv_scl`) |
| of which gas k-eqv build | 4.6 %; `augment_radiance` 2.6 %; other 2.0 % |
| `grey_opt_prop` | 10.0 % |
| rest of the band loop of `radiance_calc` itself (outside `grey_opt_prop` and the band solver: per-band set-up, phase-function rescaling, cloud geometry, overlap) | 21.4 % (LW 23.3 %, SW 14.9 %) |
| outside the band loop | 1.8 % |
Consequence for the port: one band solver (`solve_band_k_eqv_scl`) and the two-stream/triple-solver chain carry two thirds of the work; the band loop's own bookkeeping (a fifth of the call) has not been decomposed further and is the next thing to instrument if a finer estimate is wanted (`rescale_phase_fnc`, `set_cloud_geometry`, `overlap_coupled`, `cloud_maxcs_split`, `prsc_opt_prop`, `set_truncation`, the continuum branches).
Not done at that point: a comparison of kernel B with tags 40-41 (done in section 8) and of kernel A with tag 10 (needs the aerosol/cloud inputs of `grey_opt_prop`, not dumped).

## 8. Kernel B (coefficient part) against the real Fortran `two_coeff_region` (tags 40-45)
`socrates_spike/r2_kernel_b_check.py` rebuilds, from the dumped inputs of every k-term, the coefficients that `two_coeff_region` returns and compares them with the dumped output (tag 41). The spike's kernel was refactored first: the coefficient part of `two_stream_homogen` is now the function `kernels.two_coeff` (port of `two_coeff_basic`, `solar_coefficient_basic`, `trans_source_coeff`, including the infra-red source coefficients, which kernel B had taken as inputs before); the solver code is unchanged and `test_kernels.py` still passes (6e-14 PIFM80, 2e-15 Elsasser against my NumPy transcription). Cloudy regions are rebuilt as the Fortran does (sum over cloud types of `frac_cloud` x coefficients, divided by `frac_region`, regions from `i_region_cloud`).

Data: packets nov26 33312, 33317, 33322, **3,634 k-term groups** (668 solar, 2,966 infra-red) at about 30 columns per request; CPU float64, NumPy/JAX `exp`/`sqrt` against Intel libimf in the Fortran. Result (`data/r2_kernel_b_check_result.txt`):
| Quantity | Region | Entries | Bitwise equal | Max relative diff | Max absolute diff |
|---|---|---|---|---|---|
| LW `trans` | clear / cloudy | 127,538 / 71,848 | 99.5 % / 97.1 % | 2e-16 / 5e-8 | 1e-16 / 2e-16 |
| LW `reflect` | clear / cloudy | same | 99.7 % / 96.1 % | 5e-8 / 2e-14 | 3e-17 / 3e-17 |
| LW `ir_1d` (source coeff 1) | clear / cloudy | same | 99.7 % / 96.1 % | 7e-9 / 2e-14 | 7e-9 / 2e-14 |
| LW `ir_2d` (source coeff 2) | clear / cloudy | same | 99.7 % / 96.2 % | 1.0 (see below) / 5e-9 | 1e-8 / 5e-12 |
| SW `trans`, `reflect` | clear / cloudy | 28,724 / 13,432 | 99.6-99.7 % / 96.4-97.0 % | 7e-16 / 2e-8 (trans), 6e-11 / 8e-15 (reflect) | 7e-16, 5e-16 / 2e-15 |
| SW `trans_0` | clear / cloudy | same | 99.4 % / 97.4 % | 2e-16 / 4e-16 | 1e-16 / 2e-16 |
| SW `s_up`, `s_down` (solar source coeffs) | clear / cloudy | same | 99.2-99.5 % / 96.3-96.8 % | 8e-10 / 3e-14 (s_up), 9e-10 / 9e-14 (s_down) | 2e-16 / 9e-16, 9e-16 / 1e-14 |
Reading: every difference is at rounding level; the unequal entries are 1-ulp differences of `exp` (Intel libimf against the JAX/NumPy library) and the larger relative values are the cancellation terms: `ir_1d` and `ir_2d` contain `1 - trans + reflect` and `1 + reflect + trans + q` of order 1e-8 in thin layers, where one ulp of `exp` is amplified to 1e-8 (the same amplification is in the Fortran, with its own `exp`). The relative 1.0 of `ir_2d` is a value near zero (absolute 1e-8). No structural discrepancy (wrong branch, wrong region, wrong layer) was found: before the indexing was fixed in the check script, such errors showed as absolute differences of order 1 (the first run of the check had my cloud-type index off by one because the dumped lower bounds are always 1 for assumed-shape dummies).
`trans_0_noscal` is not compared: the Fortran computes it only when `control%l_noscal_tau` is set; the dumped entries do not match `exp(-tau_noscal*sec_0)` and look like uninitialised memory (abs differences up to 1.5e3), so I infer the flag is off in this rundeck; I did not read it from the running model.
What this validates: the coefficient routines of kernel B (the two-stream coefficient chain, both schemes, solar and infra-red sources, the cloud-region blending) on 3,634 real k-terms. What it does not: the layer solver (`solver_triple_hogan`: the spike kernel is the homogeneous direct solver, not the triple-region Hogan solver that this rundeck runs, so tag 42/43 cannot be compared with it), GPU `exp`/`pow` (CPU only), the bands not sampled (stride 331), and any other date.
Consequence for the projection: the coefficient work is validated and cheap; the solver that carries the rest of the 57 % (`solver_triple_hogan`, three regions with cloud overlap) is not what kernel B timed (the spike's "3 regions" timing repeated the homogeneous solver three times).

## 9. The layer solver: `solver_triple_hogan` ported and compared

Added after section 8, on the same terms (D3 stage R2: instrumented scratch copy, no arithmetic changed). The dump gained tag 46 (the 18 overlap-matrix elements per layer boundary, `flux_inc_down`, `diffuse_albedo`); the instrumented binary was rebuilt and is again bitwise on packets **33312, 33317, 33322** (all 21 non-AIJ fields, at stride 331 and again at stride 100000).

`fullfidelity/socrates_spike/solver_triple_hogan.py` is a line-by-line port of `solver_triple_hogan.F90` (operand order kept; Python loops over the 43 layers, vectorised over columns: a validation port, not the production scan). `r2_solver_check.py` feeds it tags 41 (`trans`, `reflect`), 42 (`s_down`, `s_up`, ground source), 44, 46 and compares the result with tag 43 (`flux_total` as the Fortran returned it). Result (`data/r2_solver_check_result.txt`), CPU float64, 3,634 k-term calls (1 column each, stride 331):

| | entries | bitwise | max abs diff | max rel diff |
|---|---|---|---|---|
| LW (2,966 calls x 88 fluxes) | 261,008 | 100.000 % | 0 | 0 |
| SW (668 calls x 88 fluxes) | 58,784 | 99.864 % | 1.8e-308 | 1.8e-8 (denormal-size values) |

The 51 SW calls with unequal entries differ only in the last bit or in denormals (the unequal entries have magnitude <= 4e-293, i.e. fluxes underflowing in the deep shadowed layers); no entry is off by a structural error. No non-finite entries on either side.

What this validates: the three-region recurrence (downward sweep of albedo/source, cloud-top handling, upward sweep with the overlap matrices v, u) on real cloudy columns (`n_cloud_top` between 18 and 43; every call has non-trivial overlap matrices). Together with section 8 this means the two kernels that make up `monochromatic_radiance` for this rundeck (coefficients + triple-region solver) are reproduced on real inputs; the inputs of the solver (overlap matrices, region fractions) are taken from the Fortran dumps, not recomputed (they come from `overlap_coupled` / `cloud_maxcs_split`, part of the unported 21 % bookkeeping).

What it does not show: (a) `n_cloud_top = 1` (cloud in the top layer) was never sampled (the `ELSE` branch of the surface flux and the first-layer logic are ported but unexercised); (b) CPU only, no GPU `exp`/`pow` or fused-multiply-add effects; (c) the first nov26 radiation steps only, about 30 columns per request; (d) only 1 column per k-term was dumped, so column-vectorised behaviour (a `n_cloud_top` that differs per column is impossible in the Fortran: it is one value per call) is not exercised beyond that; (e) the performance of the solver on a GPU is not measured here (next step: a `lax.scan` version and a V100 timing at 3,170 LW + 1,826 SW calls x 9/6 bands x ESFT terms).

## 10. Overnight port (owner's challenge of 2026-10-11): R3-R5 validated for this rundeck, from atmosphere to band fluxes

Scope and honesty first. The owner asked for the SOCRATES port to be finished overnight; the plan's estimate for the whole port is 110-245 h (R1 report). What was done in this session (about 6 h of wall time, one instrumented rebuild per new set of dumps, about 25 min each) is the **spectral/optical/radiative core for this rundeck's options** (R3, R4, R5 and the cloud-geometry part of R6): a NumPy/JAX implementation that, given the model-side inputs of one `radiance_calc` call (atmosphere, aerosol-mode and cloud-condensate fields, albedos, solar irradiance, temperatures) and the spectral data, reproduces the Fortran band fluxes at rounding level. **The port is not finished**: the ModelE side (`planet_rad.F90` / RADIA: building those inputs from the model state and turning the fluxes into the 22 output fields), the batched GPU version at production size with its timing, other dates, and the unported branches below remain. All numbers below are the files in `fullfidelity/socrates_spike/data/`; the oracle is the instrumented binary, which stayed bitwise to the recorded server outputs on packets 33312, 33317, 33322 at every rebuild (stride 331: 45 dumped calls = 29 LW + 16 SW, one column each).

| Stage | Module | Compared with (dump tag) | Result |
|---|---|---|---|
| R3 gas: P/T lookup (`inter_pt_lookup`), `k_esft_layer`, power-quad scaling, `k_abs_layer`, major absorber per call | `gas_absorption.py`, `spec_data.py` | 26/1, 25/1 | **100 % bitwise**: `k_abs_layer` 230,781 and `k_esft_layer` 223,901 entries (LW 183,309 / 183,309, SW 47,472 / 40,592), 45 calls |
| R3 k-equivalent absorption: `k_eqv`, `adjust_solar_ke`, IR `monochromatic_gas_flux` | `k_eqv.py` | 20/1, 47/5 | SW `adjust_solar_ke` 100 %, SW `k_eqv` 93.6 % (max rel 3e-16), LW `k_eqv` 81 % (max rel 2e-12): exp ulp |
| R4 grey optical properties: Rayleigh, modal aerosol, self continuum, CSIW water/ice clouds, Cairns, normalisation, phase rescale | `grey_opt_prop.py` | 10/1-4, 49, 40/7-8 | **100 % bitwise** on 119 bands (LW 5,385 + SW 1,600 cloudy entries, plus clear-layer arrays) |
| R4/R6 cloud geometry: `set_cloud_geometry`, `aggregate_cloud`, `overlap_coupled` (triple) | `cloud_geometry.py` | 44, 45/2, 46/1, 47/3 | **100 % bitwise**, 45 calls |
| R3/R5 per-k-term `tau`, `omega` (single scattering, rescale, k_eqv add) | `kterm.py` | 40/1-4 | **100 % bitwise**, 3,634 k-terms |
| R5 two-stream coefficients, sources, Hogan solver | `kernels.two_coeff`, `triple_column.py`, `solver_triple_hogan.py` | 41, 42, 43, 48 | sections 8 and 9; with the Fortran coefficients as input the sources and the solver are **100 % bitwise** (91 groups, `data/r2_triple_check_result.txt`) |
| Planck terms of a band (table interpolation, tiles) | `planck.py` | 25/5-8 | **100 % bitwise** (261 band records, including the tile surface emission) |
| R5 one band end to end, per k-term fluxes | `band_chain.py` | 21, 48 | rounding level: LW 74 % bitwise, max abs 7e-8 W/m2; SW 77 %, max abs 4e-14 |
| band flux accumulation (`augment_radiance`), IR Planck add-back, LW channel sum | `r2_band_check.py` | 30 | rounding level (LW max abs 2.5e-8, SW 3e-14) |
| **From the model-side inputs to band fluxes** (`radiance_calc_jax.py`) | all of the above | 30 | 14 LW + 8 SW calls (every second dumped call): LW max abs 1.6e-8 W/m2 (max rel 3e-6 on near-zero upper-level fluxes), SW max abs 1.1e-13 |
| Same with n_cloud_top = 1 for every column (the formulation needed for batching) | `--nct1` | 30 | same rounding level on 3 calls (2 LW, 1 SW; LW 6e-9, SW 6e-14): the per-call cloud-top bookkeeping is not needed |

What the instrumented dumps added in this pass (new tags): 25 (band inputs of the k-eqv solver), 26 (`k_esft_layer`, continuum, rescaled gas), 46-49 (overlap matrices, boundary terms, forward scatter, direct flux), 100 (the spectral arrays of the real reader, once per isolir), and tag 0 extended with the atmosphere, aerosol-mode, cloud and surface inputs.

Findings worth keeping:
* The two spectral files have no self-broadening and no generalised continua; gas absorption is the (T, p) lookup (`i_scale_fnc` 9) for all 25 LW band-absorber pairs and 10 of the 14 SW pairs, and the power-quad scaling (function 2) for the other 4 SW pairs; only the self continuum (power-quad scaling) is a grey continuum (LW bands 2-8, SW bands 3-6).
* The **major absorber of each band is chosen per call** from the last profile's column transmission (`planet_rad.F90` 2780-2842) and swapped with the first listed gas on the persistent spectrum object. Starting from the file order every time reproduced the dumped order in all 45 calls.
* `repsilon*mol_weight_air` in the continuum amount corresponds to a water molar weight of 18.015, not 18.0153 (found from a constant ratio 1.000017 in the dumped `k_grey_tot`): a constant of this build, not of the physics.
* The LW is a **single output channel**: tag 30 is cumulative over the bands of the call, and the Planck function is added back per band (`adjust_ir_radiance`).
* `l_noscal_tau`, `l_orog` are off (tag 0/20), `l_tile` varies per call (tile surface emission), `n_order_phase` = 1, `n_order_forward` = 2, aerosol is modal (14 modes), clouds are CSIW with parametrisations 5 (droplets, Pade) and 12 (ice, Fu), `i_cloud` = triple.

Not ported (listed so that nothing is implied): spherical-harmonic and solar-phase-function branches, McICA, the other cloud and aerosol parametrisations and the classic (non-modal) aerosol, the other gas scaling functions, `l_orog`/`l_noscal_tau` on, `ip_overlap_*` other than `k_eqv_scl` (the `solve_band_ses/one_gas/without_gas` solvers have zero time share in the three packets); everything on the ModelE side of the interface (RADIA, `planet_rad.F90` conversions, the aerosol Mie/humidity tables that produce `mode_absorption` etc., albedo construction, the diagnostics of the 22 output fields); dates other than the nov26 radiation steps; accuracy of the column-batched version on the GPU (section 11).

Remaining work against the plan (hours, the plan's own ranges): R6 interface and outputs 20-40, R7 batching/jit/GPU tuning 15-40 (a first batched formulation exists, section 11), R8 validation campaign 15-30. R3-R5 (plan: 70-155 h) were done here in one overnight session because the per-stage dumps of R2 existed; the plan's pace calibration (2-6 agent-hours per 1,000 lines) was about 10x too pessimistic for this part, which is the evidence to use when deciding R6-R8, not a promise for them (R6 has no stage-level oracle yet: the model-side inputs are only what tag 0 records).

## 11. Batched, device-resident formulation (R7 first cut): whole radiation step on one V100

`fullfidelity/socrates_spike/radiance_batch.py` rewrites the validated chain of section 10 for all columns at once (no per-column Python control flow, `lax.scan` over layers, k-term axis padded to the band maximum with weight 0, major absorber chosen per column with a gather). Two properties of the formulation were established by experiment, not assumed:

* **n_cloud_top = 1 for every column is equivalent to the Fortran per-call cloud top, with one exception.** The exception is the Fortran special case `n_cloud_top == n_layer` (no cloud above the lowest layer; cloud-free columns included): in `solver_triple_hogan` the layer above the lowest one then uses only the clear-region albedo of the lowest layer (`theta11 = alpha11(l, i+1)`) and ignores the overlap with cloud in the lowest layer. Without that special case 12 of 1,825 solar columns of the step were off by 0.02-17 W/m2 and the longwave by up to 3 W/m2 (the four examined had cloud only in the lowest layer; with the special case reproduced all columns agree); with it every column agrees. The special case is physically questionable (it underestimates the cloud effect for a cloud confined to the lowest layer) but it is what the reference does, and the port reproduces it; it is flagged for the owner as a reference-model quirk, not a port decision.
* The sums over k-terms and the cumulative products over layers are not in the Fortran's sequential order; this costs only rounding.

**Correctness against the Fortran (tag 30, band-end fluxes), all dumped columns of the radiation step of packet 33312 (stride 1, light dump):**

| | columns compared | max abs diff, CPU float64 (`data/r2_batch_fullstep_cpu_result.txt`) | max abs diff, V100 float64 (`data/r2_batch_fullstep_v100_result.txt`) |
|---|---|---|---|
| SW (6 bands, down and up flux at 44 levels) | 1,825 | 4.8e-10 W/m2 | 6.7e-11 W/m2 |
| LW (9 bands, cumulative channel) | 3,169 | 7.0e-8 W/m2 | 5.0e-9 W/m2 |

The count is one column short of 1,826 / 3,170 in each case: one call of each type has no band-end record in the light dump (not investigated; the other 4,994 are all within the numbers above). Bitwise equality is 9-30 % of entries (rounding-level agreement, as in section 10); `max rel` of the LW downward flux is 1.5e-5 because the top-of-atmosphere downward LW flux is about 1e-9 W/m2 (the relative error of a number that is zero up to rounding).

**Timing on one V100 (float64, `data/radiance_bench_v100_result.json`)**, columns tiled from the dumped real columns to the model's per-step counts (3,170 LW and 1,826 SW), steady state, `block_until_ready`, median of 3:

| | bands only | whole pipeline in one `jit` (grey properties, Planck, all bands) | compile + first run |
|---|---|---|---|
| LW, 3,170 columns, 9 bands | 0.107 s | **0.116 s** | 50 s |
| SW, 1,826 columns, 6 bands | 0.039 s | **0.043 s** | 32 s |
| radiation step (LW + SW) | | **0.16 s** | |

For comparison: the Fortran `radiance_calc` is 6.5 s of a 12.2 s server call (R2, single core, loaded host; section 3 above) and the server share of a radiation step is 8.2 s in the D214 hybrid measurements (`Reports/README_START_HERE.md`). The solver alone (`solver_bench.py`): 0.07-0.11 microseconds per column on the V100, bitwise equal to the Fortran on the six real columns tested (`data/solver_bench_v100_result.json`).

What this timing does not include or show: (a) the model-side work that builds the inputs (R6: it is `planet_rad.F90`/RADIA, about 6,500 + 3,900 lines of which the used part is not yet known; the inputs here are the recorded tag-0 arrays) and the post-processing into the 22 output fields; (b) host-device transfers; (c) the A100 / Discover container (jax 0.6.1 vs 0.5.3 here; the V100 rule of `GPU_V100_VS_A100.md` applies: not A100-confirmed); (d) GPU memory use (not measured; the jobs ran with 48-96 GB host memory); (e) `n_order_phase`/solver options other than the rundeck's; (f) other dates; (g) the tiled columns repeat 29 / 16 real columns, so data-dependent branch behaviour on the GPU is not representative (the arithmetic is branch-free except `where`); (h) float64: no float32 test.

Compile time was the first problem: the first formulation (Python-unrolled layer loops) needed 2-5 minutes per band; `lax.scan` brought the whole LW to 50 s and the SW to 32 s. A persistent compilation cache or ahead-of-time export is advisable for the model run.

Status of the port after this session: the SOCRATES computation proper (R3-R5) is validated and runs at 0.16 s per radiation step on one V100 for the rundeck's options. **Not done:** R6 (the model side, with its static fields: ozone, GHG, aerosol-mode tables, albedos, orbit; and the 22 output fields), R8 (more dates and steps, a free-running comparison, GPU categories, mutation tests), the integration into `jax_radiation.py` replacing the server callback, and the open owner decisions (funding, `interp_aer` edit; A100 confirmation).

## 12. R6: the ModelE side (RADIA, RCOMPX, planet_rad set-up and read-out) for this rundeck, in JAX; one calendar day validated

Owner instruction of 2026-10-12: "proceed to R6". Result in one sentence: **the 21 non-AIJ output fields of the radiation server are reproduced from the 52-field input packet by JAX code at rounding level for ten consecutive radiation steps of one model day (nov26), from the model state to the outputs, and the whole step takes 0.174 s on one V100 in float64**. What is NOT done is listed in 12.6; the largest items are (a) the production of the daily climatology products for any date other than nov26, (b) the 750 AIJ radiation diagnostics (four of them are rebuilt), and (c) any date or run other than nov26 as oracle. Nothing in the original SOCRATES/ModelE trees was modified (instrumentation only in the scratch copy; `ffsoc/apply_ffsoc_r6.py` refuses `ModelE_Support`).

### 12.1 What was built (`fullfidelity/socrates_spike/`)

| Part | Module | Fortran it replaces |
|---|---|---|
| A. column set-up, albedos, gases, aerosols | `radia_cols.py` | RAD_DRV.f RADIA column loop (vertical arrays, RH from `QSAT`, surface fractions, ice/lake/snow state), RADIATION.f `seth2o`, `getgas` (GHGMOD.f), `FPXSCALEGAS`, `SETAER`/`GETDST`/`SETVOL` (arrays handed to planet_rad, KDELIQ hysteresis), `GETEPS` (KCLDEP=4), ALBEDO.f `GETSUR` with RAD_UTILS.f `RXSNOW`/`GTSALB`/`SGPGXG` |
| B. planet_rad set-up | `planet_side.py` | `run_planet_rad`, `set_t_level`, `set_atm`, `set_gas_mix_ratio`, `set_cld`/`set_cloud_field`, `set_aer`/`calc_aer_opt_prop`/`interp_aer`, `set_bound_lw/sw`, planet_alb.F90 (`set_surf_prop_lw/sw`, `set_tiling`, `calc_alb_wgt`, `integ_planck`) |
| C. read-out | `planet_side.py` | `get_planet_radout`, `overlap_coupled` total cloud cover, `augment_tiled_radiance` for FSRNFG |
| zenith | `zenith_coszs.py` (on `fullfidelity/drv_zenith.py`) | `COSZS` (mean and sunlit-weighted cosine over the NRAD interval) |
| whole step | `radia_jax.py` | RADIA output assembly: SRHR/TRHR, RQT update, T update, Q reset, KLIQ, CLDSS/CLDMC masking, FSF, TRSURF, ALB, FSRDIR ... CFRAC; one `jax.jit` |
| drop-in for the server callback | `radia_native.py` | `NativeHandoff.call(packet, itime, seed)` has the signature and return of `jax_radiation.RadiationHandoff.call`; `make_handoff_step` is unchanged. Zenith grids come from a table precomputed with the exact orbit arithmetic |

SOCRATES itself (`radiance_batch.py`, section 11) is unchanged; section 12 only adds what is around it.

### 12.2 New instrumentation (scratch copy only; `ffsoc/apply_ffsoc_r6.py`, `README.md` tag table)

No new arithmetic. `FFSOC_PL=1` (via `R2_PL=1` in `r2_oracle.py`) writes, for every column and independent of the stride: tag 60 (RCOMPX state just before `run_planet_rad`: PLB, TLM, SHL, RHL, CLDEPS, U0GAS, FULGAS, PPMV80, FPXCO2, aerosol optical-depth columns, volcanic radius, snow/ice/lake variables, scalars), 61/62 (dust columns, TAUCON), 50 (arguments and module inputs of `run_planet_rad`: levels, ULGAS, CLDEPS, PRNB/PRNX, cos zenith, solar irradiance, 15 cloud arrays, relative humidity, surface fractions/temperatures/snow), 52/53 (planet aerosol set-up), 51 (outputs of `get_planet_radout`); once: tags 101-150 (control flags, band limits, planet aerosol tables, cloud-parameter limits), 54 (SW albedo weights), 170-183 (GTAU/SALBTG/TAUGSA tables of RXSNOW, ALBVNH of the day, RAD_COM constants, EPLOW/EPMID/EPHIG); tag 30 gains k = 3 (direct flux) and k = 4 (tile upward flux). The instrumented binary stayed bitwise to the recorded server outputs for the 11 packets served (`all_equal` true, 21 fields). One radiation step is 3,170 columns (72 x 44 + 2 poles), 1,826 of them lit at nov26 00:00.

### 12.3 Results against the Fortran (all numbers in `data/`)

| Level | Compared with | Result |
|---|---|---|
| B: atmosphere, gas mixing ratios, cloud field, aerosol mode optical properties, t_ground, tile fractions/temperatures, SW albedos, zen_0, solar irradiance (`r6_stageb_check.py`, `data/r6_stageb_check_result.txt`) | tag 0 of the real `radiance_calc`, 3,170 LW / 1,826 SW columns | **100 % bitwise** for all of them; `t_level` 66 % bitwise (max rel 5e-15, `log10`), LW diffuse albedo and tile albedos 63-80 % bitwise (max rel 3e-15, Planck-weight sums) |
| B+C+SOCRATES: from the arguments of `run_planet_rad` to `get_planet_radout` (`r6_e2e_check.py`, `data/r6_e2e_fullstep_cpu_result.txt`) | tag 51, 3,170 columns | LW `TRUFLB/TRDFLB/TRNFLB` max abs 6e-8 W/m2, `TRFCRL` 1e-7; SW fluxes max abs 5e-10 W/m2, `FSRNFG` 1.3e-10, `ALBVIS/ALBNIR` 1.5e-15 / 7.5e-15; `tot_cloud_cover` and `S0` **bitwise** |
| A: set-up of RADIA/RCOMPX (`r6_stagea_check.py`) | tags 50/60 | PLB, TLM, SHL, p_level, t_layer, 15 cloud arrays, surface state, CLDEPS, aerosol arrays **bitwise**; RHL 99.6 % bitwise (max rel 4e-16); PRNB/PRNX 99.9 % (max abs 2e-14, `exp`/`log` in RXSNOW); ULGAS gases 6, 7, 11, 12 max rel 3e-15; COSZA (COSZS) bitwise for all 11 steps; COSZ1 bitwise |
| **Whole `radia_jax` against the recorded server output, 10 radiation steps 33312-33357 (CPU float64, no-FMA XLA flags)** (`data/r6_radia_day_cpu_result.txt`) | `rsv_n26_<it>_out.bin`, all fields but AIJ, valid boxes | Q, KLIQ, SNOAGE, CLDSS, CLDMC, CFRAC, COSZ1 **bitwise** (KLIQ: 507,200 entries per step, no flip); `SRHR` max abs 9e-10 W/m2; `TRHR` 2e-7 W/m2 (rel 5e-5 on a near-zero cooling rate); `T` 4e-7 K (rel 1e-9); `RQT` 8e-6 K; `FSF`, `SRDN` 8e-10, `ALB` 2e-13, `DIRVIS/DIFNIR` 6e-10 |
| same, one V100, float64, jax 0.5.3 (`data/radia_bench_v100_result.json`, steps 33312, 33347) | recorded output | SRHR 1.6e-10, TRHR 4.3e-8, T 1.2e-8 K, RQT 1.1e-8 K, FSF 1.2e-10, SRDN 1.2e-10; KLIQ one entry of 507,200 differs by 1 (a relative-humidity threshold: rounding) |
| `NativeHandoff` inside `jax.jit`, steps 33312-33322 (`r6_native_check.py`, `data/r6_native_check_cpu_result.txt`) | recorded output and the four AIJ columns the hand-off asks for | same fields at the same level; `SRNFP0`, `TRNFP0`, `SRINCP0`, `SRNFG` rebuilt from the outputs agree with the recorded AIJ deltas to 6e-11 (the noise of the recorded delta, `r2_oracle` reports the same 5.8e-11) |

**Timing (V100, float64, the whole step in one `jit`, `radia_bench.py`)**: first call 237 s (compile of the whole function), then **0.174 s** per radiation step (median of 5, `block_until_ready`; 3,170 LW + 3,170 SW columns: the unlit columns are computed and masked, which costs about 1.7x of the lit-only SW). The server share of a radiation step in the D214 hybrid was 8.2 s; the zenith grids (exact orbit arithmetic) cost 0.01 s on the host per step and are not in the 0.174 s. Not measured: host-device transfer of the packet, GPU memory, A100, float32.

### 12.4 Findings (reference-model properties, not port decisions)

* **REAL*8 data written with single-precision literals.** `CMANO2` (NO2 profile), `PPGRAD`, `SRFOAM`, and the aerosol constants `DENAER`, `REFDRY`, `FRSULF`, `FS8OPX`, `FT8OPX` are `REAL*8` parameters initialised from default-real literals (`0.0100`, `8.66E-06`, `1.760`): their values are the float32 numbers widened. Using the decimal values gave 1e-8 relative errors (NO2, N2O, CH4, CFC columns; sea-surface albedo). The port uses the float32 values.
* **The snow/ice albedo routine has a threshold that the model's own arithmetic crosses.** `RXSNOW` returns the diffuse albedo unchanged if `rbsno < 0.05`. For sea ice, band 5 of the melt-pond blend is `0.05*(1-f) + 0.05*f` with `AOImax(5) = AMPmin(5) = 0.05`, whose floating-point value is 0.049999999999999996 or 0.05 depending on `f`. Below the threshold the albedo stays 0.05; at exactly 0.05 `GTSALB` runs on a degenerate table end and returns direct albedos of 0.15-0.5. About 1 in 1,000 columns (3 at 33322, 1 at 33337) is affected, with up to 1.4 W/m2 in the surface solar flux. The result depends on whether the compiler fuses the multiply-add: **with FMA contraction (default XLA:CPU) 3 columns of 3,170 differ from the Fortran by up to 0.45 in a direct albedo band; with `--xla_cpu_max_isa=AVX` (the flags of `clouds_jax_env.py`, imported by the scripts) they match**. On the V100 (FMA on) steps 33312 and 33347 happened not to hit it; this is a hazard for GPU runs, not a measured defect rate. `radia_cols._two_prod` gives each product a second use so that the blend cannot be contracted: it fixes the isolated case but not all graph shapes (33337 still differed in the full graph with FMA enabled). The owner may want this threshold looked at as a model issue.
* RADIA with `GISS_RAD_OFF`: the GISS radiation branches (`THERML`, `SOLARM`, `GETCLD`) are not compiled; `MADBAK = 0`, `NTRACE = 0` (no tracer aerosols), `KEEPRH = 2` (RH from RADIA's own `QSAT`), `KRHDTK = 1`, `KCLDEP = 4`, `KSIALB = 0` (Hansen), chlorophyll albedo (`ocalbedo`) not active (`LOC_CHL < 0` on ocean columns). Four deliquescence flags per layer (KLIQ) are carried for diagnostics only; the planet optics use the relative humidity directly.
* `cloud_rad_forc = 1` is set in the rundeck's parameters, but the radiation call dumps show `l_clear` off on the CSIW path (section 10); the clear-sky columns are not produced.
* The first-call-only module state `LOC_CHL` persists between boxes (a stale value in land-only boxes; unused there).

### 12.5 Daily products: what the port takes as input and what producing them for another date needs

`radia_jax.Products` takes per box: the ozone column on the radiation layering (`o3col`), six aerosol optical-depth columns, seven dust columns and the TAUCON constants, the volcanic optical-depth column with its effective radius and `Q55`, the three cloud-heterogeneity fields (EPLOW/EPMID/EPHIG, from the day's 72x46 fields), vegetation fractions (`PVT`), the `dALBsn` black-carbon field, `FPXCO2`; and per day: FULGAS/PPMV80 (greenhouse-gas trend), the solar irradiance (`S0/RSDIST`), the `ALBVNH` vegetation-albedo table, the RXSNOW tables and the reference layering (PLB0/HLB0, RAD_COM `plb0`/`shl0`, `byaml00`). All of these depend on the date and the box only, not on the model state (checked: ten steps of the day with the products of the first step reproduce the Fortran). **At 33362, 1 h after midnight, errors appear (SRHR up to 1.3 W/m2, T 5e-3 K) because the products of nov26 are used on nov27 (solar irradiance 1398.24 -> 1398.73, a new `ALBVNH`, new ozone/aerosol day).** Producing the products for an arbitrary date means porting, from the Fortran readers: the ozone time stream (`UPDO3D`, monthly-to-daily 'ppm' interpolation of `timestream_mod`, `REPART` onto the radiation layering), the six `TAero_*` and `DUSTaer` netCDF climatologies with their month interpolation and layer remapping (`READ_AERO.f`, `get_aero_column`, `get_dust_column`), the volcanic file with `Q55H2S` splines (`get_volc_column`, `SETQVA`), the GHG table (`GHGHST`, `GTREND`), the epsilon file (`UPDEPS`), `BCdalbsn`, `UPDSUR` (simple), the orbit (`DAILY_orbit`: `RSDIST`; `drv_zenith.py` has the declination) and the ENT vegetation export (`PVT`). The files are in the rundeck's input list (`P2SAoM40.R` lines 137-160). Estimate: 15-30 agent-hours; no code written. An oracle for other dates does not exist (only nov26 restarts); a Fortran run with the instrumented binary started from another restart date, dumping tags 60/61/62 and the statics at the start of each day, would give one.

### 12.6 Not done (nothing implied)

* The 750-field `AIJ` of RADIA (cloud-overlap diagnostics, clear-sky and CRF diagnostics, band-by-band TOA, aerosol-optics diagnostics); four hand-off columns are rebuilt.
* Dates other than nov26 and any free-running comparison; the owner decisions of 12a (funding, `interp_aer` edit) stand; A100 confirmation; float32; GPU memory.
* Branches of RADIA not reached by this rundeck: `kradia /= 0`, tracer aerosols (`TRACERS_ON`), `cloud_aer_o3_rad_forc` second calls, `aer_rad_forc > 0` (calls radiation 18 more times), the `dust` / `AMP` / `TOMAS` options, `ocalbedo` with chlorophyll, KSIALB = 1 (written, not validated), spectral-file options of section 10.
* Integration test of `NativeHandoff` inside the full atmosphere step (`jax_atm_step`/`atm_day_free_rad`): the class has the interface but was exercised only by `r6_native_check.py`, with the packet taken from the record. The seed chain and the carry (`SNOAGE`, `RQT`, `KLIQ`) are the caller's, as before. `jax_radiation.py` was not edited (it belongs to the other session's branch).

## 13. Trial integration, nov26 radiation steps 33312-33357 (owner request 2026-10-12)

`fullfidelity/socrates_spike/r6_trial_integration.py` plugs `NativeHandoff` into the other session's `jax_radiation.make_handoff_step` (not modified) and drives it like a free run on the device: per radiation step the recorded atmosphere/cloud state and live surface fields, SNOAGE from the model record (RADIA does not produce it), **RQT and KLIQ from the held outputs of the previous radiation call**, then the four non-radiation steps that follow (held SRHR/TRHR, recorded per-step COSZ1). CPU, float64, no server process. Result (`data/r6_trial_integration_cpu_result.json`), 10 radiation steps + 40 held steps:

* the 21 output fields against the recorded server output: Q, KLIQ, SNOAGE, CLDSS, CLDMC, CFRAC, COSZ1 bitwise in all 10 steps; SRHR 1.7e-8, TRHR 2e-7 W/m2, FSF/SRDN 8e-10, ALB 4e-12, T 4e-7 K; **RQT drifts from 4e-6 K (first call) to 4e-5 K (tenth call)** because the carry feeds back our own rounding-level differences (not a per-step error);
* T_new of the radiation steps against the recorded T: 4e-7 K; held steps against an independent NumPy evaluation: bitwise (0.0); the hold is not modified by held steps;
* 13 XLA compilations, all in the first call (none afterwards): the `lax.cond` step is compiled once.

Two traps found while doing it, both about numerics not interface: (1) **the XLA:CPU flags must be set before `import jax`** (`import clouds_jax_env` first; the project modules do this, a script that imports jax first does not): without them the first trial run had FSF errors of 5-130 W/m2 (the snow/ice albedo threshold of section 12.4, and a SNOAGE carry taken from the server output, which is wrong: the model updates SNOAGE); (2) SNOAGE must come from the model, not from the hold.

Not covered by the trial: the drv_radpacket surface fields (taken from the live packet), the dynamics (T between radiation steps is the record's), the RNG/seed chain (`seed` is ignored), a GPU run of the chain, and any date other than nov26.
