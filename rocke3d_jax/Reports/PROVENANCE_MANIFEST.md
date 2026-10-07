# Provenance manifest: are our references the supplied `P2SAoM40_003` run?

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07 (Phase 1 of `RECONCILIATION_PLAN.md`, actions A1 to A3). Project-local per `projects/imvi/AGENTS.md`.
Review by: when the project lead answers Phase 0 question 4, or when any reference file changes.

## 1. Result in one paragraph

The **source code and the rundeck template** behind our Fortran oracle are the ones published with the ROCKE-3D 2.0 paper (Zenodo 10.5281/zenodo.14721184): 361 of 362 source files are byte-identical, and the one that differs differs only in the directory paths of the SOCRATES data files. The **local rundeck** is the published template with its `#include` files expanded. What is **not** established is that our *reference simulations* are the supplement run `P2SAoM40_003`: our local reference run (`ModelE_Support/prod_runs/P2SAoM40`, years 1949 to 1950) is a separate short run of the same configuration, started from observed initial conditions (not an equilibrated state); the supplement is a 100-year equilibrated run (years 4000 to 4099). Climate-level agreement between the two cannot be tested with what we have (section 5).

## 2. What was compared and how

All downloads are from the public Zenodo record "ROCKE-3D v2" (DOI 10.5281/zenodo.14721184, CC BY 4.0), fetched on 2026-10-07 into the session scratchpad (not the repository). Checksums were verified against the record's metadata.

| File | Size | md5 (matches record) |
|---|---|---|
| `modelE2_planet_2.0.tar.gz` (model source, templates) | 5,445,753 B | `3fdbaaded4c7e128111ccebd99a5fb68` |
| `P2SAoM40.tar.gz` (the configuration's output) | 8,497,870 B | `58a14ca02be9bf89d32984729b8ee875` |

The record lists 79 files (about 5.0 GB), one archive per configuration plus the source and topographies. Other configurations are not needed for this question.

## 3. Findings

### 3.1 Source code (A2)
- Local tree: `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0`. Compared the 362 source files in `model/` (`.f`, `.F`, `.f90`, `.F90`, `.h`, `.c`): **361 identical**, 1 different, none present on only one side. `aux/` has no differences. Compiled objects present only in the local tree (`*.o`, `*.mod`) were excluded from the comparison.
- The one difference is `model/planet_rad.F90`, lines 140 to 146: the SOCRATES data directories (`dir_solar_spec`, `dir_spectral`) point to site-specific paths (the local tree's `ModelE_Support/stellar_spectra` and `.../spectral_files` versus the NASA Discover paths in the release). No physics differs.
- Not compared: the SOCRATES spectral data files themselves (contents of the two directories), and the `giss_LSM` and `Ent` source subdirectories (the `Ent` directory has its own tree; it is the source our Ent port was written from).

### 3.2 Rundeck (A2)
- `templates/P2SAoM40.R` in the release and in the local tree: **identical** (0 differing lines); the whole local `templates/` directory is identical to the release.
- The local `decks/P2SAoM40.R` is the template with the `#include` lines expanded in place (module list, input-file lists, parameter blocks) and local edits: the run window `YEARE=1950,MONTHE=12,DATEE=1` (one year from `YEARI=1949,MONTHI=12`) where the template says `YEARE=1949,MONTHE=12,DATEE=2`. The start is `ISTART=2`: a cold start from observed initial conditions (the `AIC`, `GIC` and `OIC` files named in the expanded deck); the restart option (`ISTART=8`) appears only in a comment. **Mechanical check (2026-10-07, done):** every `#include` of the release's `templates/P2SAoM40.R` was expanded recursively (no missing include; 193 significant lines after removing comments and blank lines) and compared, as a set of lines, with the local `decks/P2SAoM40.R` (192 significant lines). They differ in exactly two settings: (a) the run end `YEARE=1950,MONTHE=12,DATEE=1,HOURE=0` (local) versus `YEARE=1949,MONTHE=12,DATEE=2,HOURE=0` (release); (b) the release's build option `OPTS_dd2d = NC_IO=PNETCDF`, which the local deck drops (comment in the deck: "no PNETCDFHOME - removing NC_IO=PNETCDF"; parallel NetCDF I/O, not physics). Everything else, including the input files (CMIP6 aerosol and ozone, `master_yr=1850`, `MADVOL=2`, the SOCRATES spectral files, the diagnostic and drag parameters), is line-for-line the release template. Limits: line sets, so ordering and duplicate lines were not compared; the contents of the include files themselves are the release's own.

### 3.3 The published output (A3)
`P2SAoM40.tar.gz` contains four NetCDF files:
- `ANN4000-4099.aijP2SAoM40.nc`: the 100-year mean (1,073 variables, 72 x 46 maps, `xlabel` "P2SAoM40_003 (LLF40 + updated aerosol/ozone input files for CMIP6 simulations...", model years 4000 to 4100, 36,500 days).
- `ANN4000-4099.aijlP2SAoM40.nc`: the same for the 3-D (`aijl`) fields (21 variables, 40 levels).
- `ANN.aijP2SAoM40.nc`: **a 100-year annual time series of global means**: 540 diagnostics, each one value per year (dimension `time` = 100 only, no maps).
- `ANN.aijlP2SAoM40.nc`: the corresponding series for the 3-D fields (not opened).

The loose file `ANN4099.aijP2SAoM40.nc` in the repository root (untracked, dated 2026-10-01) has the same label and the same 1,073 variables; its global means agree with the 100-year mean file to within a plausible year-to-year spread (tsurf 13.93 against 13.96 C, prec 2.908 against 2.895 mm/day, srnf_toa 235.75 against 235.74 W/m2; same units), so it is consistent with being year 4099 of this run. A direct field-by-field match against the annual series was not possible because the series holds global means only.

All the key fields of the F3 comparison exist in both the series and the 100-year mean under the same names: `prsurf, slp, t_850, t_500, tsurf, prec, evap, srnf_toa, trnf_toa, qatm, z_500, u_850, incsw_toa, pcldt, tauus`. Field names therefore map one to one onto the AIJ columns of `f3_diagnostics.py`; the numerical conventions (units, area weighting) were checked only for the fields in the next table.

### 3.4 Interannual spread of global means over the 100 years (new, from `ANN.aijP2SAoM40.nc`)

| Field | 100-year mean | Interannual sd | Fitted drift over 100 years |
|---|---|---|---|
| tsurf (C) | 13.99 | 0.064 | +0.015 |
| t_500 (C) | -16.79 | 0.062 | +0.013 |
| t_850 (C) | 7.95 | 0.058 | +0.007 |
| prec, evap (mm/day) | 2.897 | 0.0065 | -0.0004 |
| slp (mb - 1000) | 10.83 | 0.0096 | +0.0005 |
| srnf_toa (W/m2) | 235.8 | 0.19 | -0.045 |
| trnf_toa (W/m2) | -235.8 | 0.18 | +0.054 |
| pcldt (%) | 52.64 | 0.15 | +0.015 |

The drift is negligible against the spread, so the run is equilibrated. This is a 100-sample noise floor for *annual global means*. It cannot score a single month, and it does not replace the 8-member JAN1950 ensemble (D165) for monthly maps.

## 4. What this does and does not establish

Established:
- Our Fortran oracle is the published ROCKE-3D 2.0 `planet_2.0` code (except SOCRATES data paths) built from the published `P2SAoM40` template.
- The published run's full output for the configuration, including a 100-year global-mean series, is obtainable (8.5 MB).

Not established:
- That the local `prod_runs/P2SAoM40` simulation (monthly `acc` files, labelled "P2SAoM40 (ROCKE-3D, based on P2SAoM40 template)", years 1949 to 1950) was produced with the same input files and settings as `P2SAoM40_003` (its label differs: the supplement's mentions updated aerosol and ozone files for CMIP6; the local rundeck names CMIP6 aerosol and ozone files too, but the expansion was not compared line by line).
- That the SOCRATES data directories contain the same spectral files as the release.

## 5. Why climate-level provenance cannot be tested now

The supplement is an equilibrated run (years 4000 to 4099). Our reference run starts on 1 December 1949 from observed initial conditions (`ISTART=2`) and covers one year; a run that has only just left its initial conditions does not reproduce the equilibrium climate (the deep ocean in particular is far from equilibrium), so comparing annual means would measure the spin-up, not the identity of the setup. An 85-minute-per-month real run from the January restart can reproduce its own stored month bitwise (D165), which establishes that the *executable and inputs are self-consistent*, not that they match the supplement's.

## 6. Open items (for the project lead and the next session)

1. ~~Mechanically expand the release's `#include` files and compare with the local expanded rundeck~~ DONE 2026-10-07 (section 3.2): two differing settings, both explained.
2. Compare the SOCRATES spectral data directories with the release (closes 4).
3. Decide where `ANN4099.aijP2SAoM40.nc` lives (untracked 7 MB file in the repository root). Suggested: keep the small published product under `ff_data/` with this manifest; needs the owner's approval.
4. Phase 0 question 4 of `RECONCILIATION_PLAN.md` (is `P2SAoM40_003` the run our references must match?).
5. Decide whether to track the 8.5 MB published output in the repository or only record its checksum (the manifest above already records it).

## 7. Sources
- Zenodo record: <https://zenodo.org/records/14721184> (API: <https://zenodo.org/api/records/14721184>; files fetched from the `14721185` version link in the record's file list).
- NCCS supplement directory: <https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/publication-supplements/Tsigaridis2025GMD-planet_2.0/P2SAoM40_003/>
- Paper: <https://gmd.copernicus.org/articles/18/5825/2025/> (see `DOCUMENTATION_REVIEW.md` for what could not be read).
