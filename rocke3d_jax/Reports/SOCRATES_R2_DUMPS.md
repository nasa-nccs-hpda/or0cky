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
Not done: a comparison of the spike's JAX kernel B with tags 40-41 (the inputs and outputs are now available; the kernel takes `g` and `omega`, the dump has `phase_fnc` of order 1 which would have to be mapped), and a comparison of kernel A with tag 10 (needs the aerosol/cloud inputs of `grey_opt_prop`).
