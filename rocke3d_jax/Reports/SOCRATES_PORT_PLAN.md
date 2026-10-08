# Plan and estimate: porting SOCRATES radiation to JAX (contingency, if asked)

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07. Project-local per `projects/imvi/AGENTS.md`.
Review by: when the owner decides whether to lift the standing rule below, or when the radiation design changes.

**STATUS: PLANNING ONLY.** The standing project rule remains in force: SOCRATES is never ported or modified. Nothing in this document has been started, and no SOCRATES file was changed to write it. This plan exists so that, if the project lead asks for the port, the scope, the cost and the decisions are already on the table. All estimates are estimates, not measurements (section 7 says how weak they are).

## 1. Why it might be asked for

Today every result of the JAX step uses the real Fortran radiation through a callback to a stateful server (D159-D162, D185). The owner accepted that as a labelled hybrid component. Costs of keeping it: about 11 s per radiation call (every fifth step; about 7% of the CPU step), 33.7 MB to the host and 11.6 MB back per call, a host synchronisation point, a Fortran executable that must exist next to the JAX code (not available in the Discover container, so GPU runs use replayed radiation), and no end-to-end claim for the radiation part of ACCEPTANCE section 1. A JAX radiation would remove all four and is also the best-suited physics for a GPU (independent columns, wide parallel work), whereas most of the rest of the step is launch-bound (probe result, `gpu/DISCOVER_RUN.md`).

## 2. What SOCRATES is here (measured in the source tree on 2026-10-07)

| Fact | Value | Source |
|---|---|---|
| Code | Met Office SOCRATES, trunk revision 409 (2017-10-05), "British Crown Copyright 1990-2017, Met Office", BSD 3-clause licence text (redistribution and modification allowed with the notices kept; the owner should have the licence checked formally) | `ModelE_Support/socrates/_version.txt`, `COPYRIGHT.txt` |
| Full library | 435 Fortran files, 118,781 lines under `src/` | line count of `ModelE_Support/socrates/src` |
| Linked into the model | **138 objects**, 2.07 MB (`model/socrates/libsocrates.a`); 137 map to source files: **48,593 lines** (radiance_core 126 files / 47,854 lines; modules_core 11 files / 739 lines) plus the ModelE-side module `def_planetstr` | `ar t` mapped to `src/` |
| Largest linked files | `read_spectrum` 3,349; `radiance_calc` 2,587; `scale_wenyi` 1,836; `grey_opt_prop` 1,479; `solve_band_k_eqv` 1,277; `def_spectrum` 1,189; `solve_band_k_eqv_scl` 1,104; `opt_prop_ice_cloud` 991; `build_sph_matrix` 948 | line counts |
| Entry point from the model | one driver, `radiance_calc`, called twice per radiation step (longwave and shortwave), after a long list of `set_*`/`allocate_*` calls | `model/planet_rad.F90` (3,866 lines, the ModelE-SOCRATES interface, not part of the library) |
| Configuration of this rundeck | spectral files `sp_sw_ga7/sp_sw_ga7_dsa` and `sp_lw_ga7/sp_lw_ga7_dsa`, aerosol tables `aer_lw_ga7.nc`, `aer_sw_ga7.nc`, `sp_diag/aer_diag_std.nc`; LW file: 9 bands, 12 gaseous absorbers; the two directories are 259 MB (SW) and 276 MB (LW) | `decks/P2SAoM40.R`, spectral file header |
| Solver options (defaults in `sw_control.F90`, `lw_control.F90`) | gas overlap: k-equivalent with scaling (`ip_overlap_k_eqv_scl`); cloud representation `ip_cloud_csiw`; sub-grid water inhomogeneity `ip_cairns`; cloud overlap `ip_max_rand`; angular integration `ip_two_stream`; two-stream scheme: SW `ip_pifm80`, LW `ip_elsasser`; scattering `ip_scatter_full` | the two control modules |

**Static reachability, measured 2026-10-07 (read-only call-graph walk of the `CALL` statements of `radiance_core`, no SOCRATES file touched):** `radiance_calc` selects among seven band solvers; this rundeck's options select `solve_band_k_eqv_scl` (default), `solve_band_ses` (forced for some longwave bands by `planet_rad.F90`), `solve_band_one_gas` and `solve_band_without_gas`; `solve_band_k_eqv`, `solve_band_random_overlap` and `solve_band_random_overlap_resort_rebin` are not selected. Reached from `radiance_calc`: **108 files / 38,267 lines with all options; 101 files / 34,899 lines without the three unselected solvers** (about 73% of the 47,854 lines of `radiance_core`). The other 18 files / 9,587 lines are not reached from `radiance_calc` (spectrum reader and definitions, `read_spectrum` 3,349, `scale_wenyi` 1,836, `def_spectrum` 1,189, the control and output types): on the JAX side these become a host-side converter of the spectral file into arrays, not ported routines. Reached families (rough, by name): optical properties 4,778 lines, two-stream/source/flux machinery 7,473, band solvers and monochromatic drivers 7,427, other (driver, helpers, cloud sampling, geometry) 15,221. `mcica_sample` (824 lines) is reachable, so cloud sub-column sampling may be in play. LIMITS of this number: it is an UPPER bound (branches inside the selected routines that the options switch off are still counted, for example spherical-geometry code such as `build_sph_matrix`, 948 lines, and `calc_radiance_ipa`, 730), and it misses functions referenced without `CALL`. Still unknown (stage R1 settles them): the truly executed lines, whether McICA is active, the number of shortwave bands, and the rundeck-level overrides of the defaults.

## 3. What we already have (reuse)

- **A bitwise oracle**: the persistent radiation server runs the real RADIA from a packet (52 input fields, 22 output fields) in ~11 s per call, bitwise reproducible (D159-D162); recorded real inputs and outputs for the nov26 radiation steps (11 calls) and the first radiation step of dec01 and jan01; `jax_radiation.py` already assembles the packet from device arrays (D185) and validates outputs field by field.
- **The packet layout and the physics inputs**: `drv_radpacket.py`, `drv_radcols.py` (D176), the harness with categories and counters (D184), the libimf host callback and its inventory of pow/exp sites (D183; SOCRATES also uses transcendental functions, so the same libm/libimf/GPU-math question applies).
- **The interface itself**: `planet_rad.F90` is a readable specification of what the model passes in and takes out.
- **Lessons**: bitwise traps found on the way (`x**4` is libm `pow` in NumPy and repeated multiplication in `jnp`; `jnp.cumsum` is not a left-to-right sum; single-precision literals under `-r8`; REAL*16 sites), compile-time traps (XLA flags), the single-core callback deadlock.

## 4. What would NOT be reusable or is new

- The spectral file reader and the compressed k-term data structures (static data: convert once on the host to arrays; this part need not be in JAX).
- The optical-property chain (gases with scaling and continua, Rayleigh, aerosols, clouds with the CSIW parametrisation and Cairns inhomogeneity) and the solvers, which are the bulk of the 48.6k lines.
- Intermediate-state oracles: the server returns only the final fluxes and heating rates. To validate stage by stage we need the intermediate arrays (optical depths per band, k-term and layer; two-stream coefficients; band fluxes). That requires an INSTRUMENTED COPY of SOCRATES built in a scratch tree (the original untouched), which is a modification of a copy and therefore needs the owner's explicit approval (decision D3 below).

## 5. Staged plan with estimates (focused hours; see section 7 for how weak they are)

| Stage | Content | Gate | Estimate (h) |
|---|---|---|---|
| R0 | Decisions, licence check, scope statement, acceptance criteria for radiation (written before any code) | owner approval | 2-4 |
| R1 | Reachability: static call graph from `radiance_calc` under the selected options, the exact option set of the rundeck (dump `control_lw/sw` from the server), list of reached routines with line counts, McICA yes/no, band counts; spec of the intermediate dumps | document with the reached-line count (the single most important number for re-estimating) | 6-12 |
| R2 | Instrumented copy in a scratch tree: dumps of the intermediate arrays for 3 dates (restart steps, same packets as the server) | dumps reproduce the server's final outputs bitwise | 6-12 |
| R3 | Spectral data to arrays (host), plus the gas optical depths with k-eqv scaling, continua, Rayleigh | per-band optical depths vs the dumps (A/B) | 20-45 |
| R4 | Aerosol and cloud optical properties (CSIW, Cairns inhomogeneity, max-random overlap, McICA if active) | per-band cloud/aerosol properties vs dumps | 25-55 |
| R5 | Two-stream coefficients (PIFM80 SW, Elsasser LW), source functions, flux solution with scattering, surface boundary, band loop and k-eqv-scl combination | band fluxes and final fluxes/heating vs dumps and vs the server | 25-55 |
| R6 | Driver and interface: `radiance_calc` logic, the ModelE side (`planet_rad.F90` conversions, albedo module, the diagnostic columns that feed the 1,660 AIJ columns) as jnp; replace the callback in `jax_radiation.py` | full packet-to-outputs equivalence with the server on all recorded steps | 20-40 |
| R7 | Vectorisation over the 3,312 columns x bands x k-terms, `jit`, memory layout, GPU run and tuning | step time of radiation on the A100; no eager loops | 15-40 |
| R8 | Validation campaign: three dates, all recorded radiation steps, a free-running day with the JAX radiation against the server day (statistical, noise floor), GPU categories, mutation tests | acceptance written in R0 | 15-30 |
| **Total** | | | **about 135-295, central about 210 (after the static count of section 2)** |

Parallelism: after R2, stages R3, R4 and R5 can run as independent tracks (the data they exchange is the intermediate dumps), so calendar time could be about 40-90 hours with four to six agents; the verification load on the supervising session grows with the number of tracks (every agent result is re-checked before it is committed, as in this project).

## 6. What it would buy and what it would not

Buys: the radiation part of the step becomes device-resident JAX (ACCEPTANCE section 1 item 4 stops being an exception), no 33.7/11.6 MB host round trips, no Fortran server, runnable inside the Discover container, and a fast radiation on the GPU (vectorised, wide work, the best case for the probe's lesson).

Does not buy: bitwise equality with the Fortran on a GPU (GPU `exp`/`pow` differ from NumPy by 1 ulp in 6-12% of values, `gpu/DISCOVER_RUN.md`), so the GPU radiation is judged at rounding-level categories per step and statistically over days; and it does not change the other non-JAX parts of the step (QUS subsidence, poles, Ent, libimf callbacks).

Cheaper alternatives (for comparison, not recommendations): (A) keep the Fortran callback (the present, accepted, labelled hybrid; zero cost); (B) a restricted port of only the SHORTWAVE or only the LONGWAVE path first (the two calls share the driver and the data structures, so the saving is less than half); (C) a learned surrogate of RADIA (a different scientific project, not a port, and it would not match the Fortran).

## 7. How uncertain these numbers are

- The base is the static upper bound of 34,899 reached lines (section 2); the truly executed number is lower (R1 measures it). If about 20,000 lines are executed the lower half of the range applies; at the static bound the upper half applies. My first guess before measuring was about 20,000 lines; the measured bound is larger, so the central estimate below is the upper half of the range.
- Pace calibration from this project: bitwise-validated ports against a Fortran oracle took the order of 2-6 agent-hours per 1,000 lines (ADVSI 756 lines, RIVERF 508, DYNSI ~550, Ent stage 1 ~1,400, the surface stage) with occasional large outliers (a compile-time blow-up of 60x, a library difference that made bitwise impossible, a single-core deadlock). These ports had dumps at the right interfaces; SOCRATES needs new intermediate dumps first (R2).
- Earlier estimates in this project were revised upward once (150-280 h before the dynamics chain was done). Treat the range as 'could be 1.5x more'.
- The range excludes: the owner's review time, a formal licence review, and any change of the rundeck options or spectral files (a different spectral file or solver option would add stages).

## 8. Decisions needed from the project lead (if the port is wanted)

1. **D1, lift the standing rule** 'SOCRATES is never ported or modified' for this scope (it is currently a standing condition in `README_START_HERE.md`).
2. **D2, licence**: confirm that a derived work under the BSD 3-clause licence is acceptable and how it is attributed (the notice has to stay with the code).
3. **D3, instrumented copy** of SOCRATES in a scratch tree for intermediate dumps (the original in `ModelE_Support/socrates` is never touched).
4. **D4, scope**: only the configuration of this rundeck (GA7 spectral files, k-eqv-scl, two-stream, CSIW), not the whole library.
5. **D5, acceptance**: per-stage categories against the dumps (A/B per stage), final fluxes against the server (A/B on CPU), GPU at rounding level plus statistics; written in R0 before code.
6. **D6, priority** against the present plan (assembled step, multi-day run, GPU fusion).

## 9. Sources

- `ModelE_Support/socrates/` (`_version.txt`, `COPYRIGHT.txt`, `src/`), `model/socrates/libsocrates.a`, `model/planet_rad.F90`, `model/sw_control.F90`, `model/lw_control.F90`, `decks/P2SAoM40.R`, the spectral file header of `sp_lw_ga7_dsa`, all in `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0` (read only).
- `FULL_FIDELITY_DELTAS.md` D159-D162, D176, D183-D187; `gpu/DISCOVER_RUN.md`; `Reports/ACCEPTANCE_CRITERIA.md`.
