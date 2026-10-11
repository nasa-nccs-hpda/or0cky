# SOCRATES spike: result (2026-10-11, session on branch `socrates-spike`)

Owner of the decision: G. Tamkin. Written by a Claude Code session under `SOCRATES_SPIKE_BRIEF.md` (rule lifted for a spike only). Review by: when the owner decides on R0/D1-D3 of `SOCRATES_PORT_PLAN.md` section 8.
Scope kept: no SOCRATES or ModelE file was edited (read only), no production path changed, everything new is in `fullfidelity/socrates_spike/`, no data committed beyond small profile text and a bench log. Nothing here claims a port, a bitwise result or a model speed-up.

## 1. Answer in one paragraph
Of the unmodified RADIA call, 37 % is one ModelE-side routine chain (`interp_aer` in `model/planet_rad.F90`, aerosol-table interpolation, not SOCRATES) and 26 % is the two-stream/flux-solver chain. JAX prototypes of both, in float64 on one V100, take 4 ms (aerosol, SW+LW) and 36 ms (clear-region solver chain for all columns, 75 ms for three regions) against about 3.0 s and 2.1 s of Fortran CPU time on the benchmark node. The remaining 27 % (optical properties, band driver, ModelE glue) was NOT prototyped. Projected full call on the V100: **about 0.3 s central, about 1 s pessimistic, 2.7 s if the unported 27 % got no speed-up at all, each plus the 0.2-0.45 s hand-off**; all below the 4 s gate. Recommendation (not a decision): **GO to R1-R2**, with the caveats of section 6. The projection is an extrapolation from two kernels, not a measurement of a ported RADIA.

## 2. Profile of the unmodified Fortran call (S1)
How: persistent server (`radiation_server_persist.py`, unmodified binary), recorded packets `ff_data/nov26_day/rsv_n26_{33312,33317,33322}_in.bin`, `perf record -F 999 -p <pid>` (user space) around one request each; script `socrates_spike/profile_fortran.py`; grouping `group_profile.py`; raw reports `socrates_spike/data/perf_*.txt`. Callers of the hot routine: perf's dwarf unwinder fails on this Intel binary, so 40 sampled `gdb` backtraces (`gdb_callers.py`, `data/gdb_bt.txt`) were used: 16 of 40 samples are `interp_aer_ < calc_aer_opt_prop_ < set_aer_ < run_planet_rad` (the radiation call, not the aerosol-diagnostic path, which is off).

| Stage (share of the server call, mean of 3 steps, min-max) | Share | Fortran seconds at 8.1 s |
|---|---|---|
| A aerosol-table interpolation (`interp_aer`, `calc_aer_opt_prop`, `set_aer`; ModelE `planet_rad.F90`) | 37.4 % (37.4-37.5) | 3.0 |
| B two-stream coefficients, transmission/source, flux solvers (`solver_triple_hogan` 6.0, `trans_source_coeff` 5.3, `solver_homogen_direct` 3.4, `two_coeff*`, sources) | 25.7 % (25.6-25.8) | 2.1 |
| C optical properties (gas k-terms, grey/cloud/aerosol, rescaling) | 11.7 % | 0.9 |
| D band / k-term driver (`radiance_calc` self 5.0, `solve_band_k_eqv_scl`) | 7.6 % | 0.6 |
| E math library, malloc, I/O | 9.7 % | 0.8 |
| F ModelE side other (`interp1d` 4.7, `radpar_*`, `radia_`) | 7.5 % | 0.6 |

Noise: three different radiation steps agree within 0.4 percentage points per group. LIMITS: (a) the profile was taken on the shared host forest204 (load average 18 on 12 cores) where the call took 12.2-12.9 s, not the 8.1-8.4 s of the benchmark node, so only SHARES are used, scaled to 8.1 s; shares can shift between nodes (memory-bound parts); (b) perf attributes by symbol: inlined code is counted in the caller; (c) the grouping regexes in `group_profile.py` are my classification; (d) **per band is not available** with an external profiler (the band index is not a symbol); as a proxy the k-term counts of the spectral files are used (LW 9 bands, 180 terms in total; SW 6 bands, 94-96 terms; upper bound: sum over absorbers; the rundeck names `sp_sw_ga7_dsa`, absent from the directory, so `_sun`/`_pc` were read). Both radiations run for every column: the call is per column (`n_profile = 1`), 3,312 columns, 40 layers, 2 spectral calls.

Finding outside the spike's remit: `interp_aer` interpolates the whole radius table (every radius, 3 arrays, with `TRANSPOSE` temporaries) for every layer and keeps two radii, up to 20 times per column. Restricting it to the two used radii would give the same numbers and save roughly a third of RADIA in the Fortran itself (about 0.5 s of the 1.62 s per step). That is an edit of ModelE source (not done, needs a separate decision) and it would speed the Fortran baseline and the hybrid equally; it does not close the gap by itself.

## 3. What was ported (S2) and how it was checked
`socrates_spike/kernels.py` (float64, JAX):
* Kernel A `aer_optical`: `set_aer -> calc_aer_opt_prop -> interp_aer` for 6 aerosols (4 with sulphate), humidity then radius interpolation in the Fortran order, only the two used radii gathered.
* Kernel B `two_stream_homogen`: `two_coeff_basic` (PIFM80 / Elsasser), `sqrt`, `solar_coefficient_basic`, `trans_source_coeff`, `solver_homogen_direct` as `lax.scan` over layers, solar and infra-red. NOT ported: the source terms of `single_solar_source`/`ir_source` (taken as inputs), the cloud-region logic of `solver_triple_hogan`/`triple_column`, the band/k-eqv-scl driver, gas optical depths.

Validation (`test_kernels.py`, CPU, run: `JAX_PLATFORMS=cpu python test_kernels.py` in graphcast-env): category-A-like, but against a NumPy transcription of the Fortran text that I wrote, NOT against the Fortran executable (the server returns only final fluxes and heating rates, so no kernel-level oracle exists without an instrumented copy, decision D3). Kernel A: 0 relative difference on both real tables (LW 9 bands, SW 6 bands), including the below-table radius branch and humidity above 1. Kernel B: 6e-14 (PIFM80, solar) and 2e-15 (Elsasser, IR) against the transcription; layer-splitting invariance without sources 2e-15. Not done: any comparison with the server's final fluxes (needs the unported parts), any GPU-vs-CPU category check (GPU exp/pow differ by ulps, `gpu/DISCOVER_RUN.md`).

## 4. V100 timing (S3)
`gpu/v100_run.sh socspike_bench 25 'python -u socrates_spike/bench.py 5'`, job 37941537, node gpu012, Tesla V100-SXM2-32GB, jax 0.5.3, float64; compile+first call reported separately; steady = median of 5 (log: `data/bench_v100_37941537.out`, `bench_result_37941537.json`). One job, one node, no exclusivity check (the benchmark protocol's node discipline was not applied).

| Kernel (all 3,312 columns, 40 layers) | compile + first | steady |
|---|---|---|
| A LW table, 6 aerosols | 6.0 s | 2.7 ms |
| A SW table | 3.4 s | 1.1 ms |
| B LW Elsasser, 180 terms, 1 region (12 chunks of 276 columns) | 0.34 s | 25 ms |
| B SW PIFM80, 95 terms, 1,656 lit columns, 1 region | 0.54 s | 11 ms |
| B LW, 3 regions | 0.32 s | 53 ms |
| B SW, 3 regions | 0.51 s | 22 ms |

Inputs: kernel A uses T/Q/PMID of a recorded packet, but the relative humidity I formed is not ModelE's `RHL` (mean 0.008: wrong units or formula on my side); gather cost does not depend on the values, the timing stands, a validation on real humidity would need the real `RHL`. Kernel B uses random optical depths 1e-4..10, single scattering albedo 0..1; timing only. The "3 regions" case repeats the same arithmetic three times: the real Hogan solver does more per region.

## 5. Projection of the full call on the V100 (every assumption stated)
Base: 8.1 s Fortran per call at the shares of section 2. Hand-off: 0.2-0.45 s per radiation step as the brief states (a device-resident JAX radiation would not pay it; included to follow the gate).

| Part | Fortran s | Central | Pessimistic | Worst | Assumption |
|---|---|---|---|---|---|
| A | 3.0 | 0.004 | 0.01 | 0.01 | measured; x2.5 for pessimistic |
| B | 2.1 | 0.15 | 0.5 | 0.5 | 75 ms for 3 regions measured; real triple solver, sources, cloud logic assumed 2x (central) / 6x (pessimistic) of that |
| C+D+F | 2.2 | 0.11 | 0.43 | 2.2 | not prototyped; speed-up 20x / 5x / 1x |
| E | 0.8 | 0 | 0 | 0 | assumed absorbed in the other stages (exp/sqrt/memory are inside the kernels) |
| Compute total | 8.1 | **0.27** | **0.94** | **2.7** | |
| + hand-off 0.2-0.45 | | 0.5-0.7 | 1.1-1.4 | 2.9-3.2 | |

Break-even for the 4 s gate: with B at its pessimistic value and a 0.45 s hand-off the unported 27 % could cost up to about 3 s (0.72x of its Fortran time, i.e. even a slowdown) before the gate is crossed, so the gate does not hinge on that speed-up; the weaker links are the B extrapolation and the unmeasured stages (section 6). Against the model: per radiation step (1 in 5) the central case is about 0.1-0.15 s versus 1.62 s now, which would put the JAX hybrid cycle at about 2.0 s against the Fortran 2.37 s (arithmetic on the numbers of `DECISION_MEMO` section 7; the other step costs are unchanged and unmeasured here). Not a model speed-up claim.

## 6. What this does not show (read before relying on it)
1. Two of six stages were prototyped (63 % of the time); the other 27 % is assumed. Gas absorption with k-term/pressure/temperature lookup, continua and the cloud sub-grid logic (CSIW, Cairns, McICA if active) are where an underestimate is most likely.
2. Kernel B's inputs are synthetic and its structure is the clear region; cloud overlap, `triple_column` and `overlap_coupled` add work and data-dependent branches.
3. Fortran shares come from a loaded host; the 8.1 s base is the benchmark node's, the shares are assumed transferable.
4. Validation is against my transcription of the source, not the Fortran. Intermediate oracles need D3.
5. A real port is also the 40 files / about 16,000 lines of `radiance_core`/`modules_core` that received samples (`executed_lines.py`; lower bound, branches inside them may be unused) against the static bound of 34,899 lines in the plan, plus the ModelE side of `planet_rad.F90` and `RADIATION.f` (aerosol set-up, `interp1d`, `getdst`). Re-estimate (plan R1 not executed): the executed code is about 45 % of the static bound; the plan's range of 135-295 h was built on the bound, so the lower half (about 135-200 h) looks more likely; this is a judgement on a line count, not a measurement, and the plan's "could be 1.5x more" caveat stays.

## 7. Recommendation against the gate (section 5 of the brief); the decision is the owner's
GO to R1-R2 on the evidence above: projection 0.5-1.4 s per call (worst case 3.2 s) against the 4 s gate, and kernels with a checkable CPU reference exist. Conditions that would turn it: R2 dumps show the unported stages cost more than assumed; a real-humidity and real-source check of kernel A/B fails; or the GPU category-C tolerance (1e-6) is not met by the full chain. Cheaper first step the owner may prefer: the `interp_aer` restriction in Fortran (section 2) benefits both sides, needs a ModelE edit decision, and de-risks nothing about the device port.

## 8. Reproduce
```
cd fullfidelity/socrates_spike
python3 profile_fortran.py OUT 33312 33317 33322 ; python3 group_profile.py OUT/perf_*.txt     # needs the persistent server binary
python3 gdb_callers.py 33317 40 OUT/gdb_bt.txt ; python3 executed_lines.py data/perf_*.txt
JAX_PLATFORMS=cpu python test_kernels.py                                                          # graphcast-env
gpu/v100_run.sh socspike_bench 25 'python -u socrates_spike/bench.py 5'                           # from rocke3d_jax/
```
