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
