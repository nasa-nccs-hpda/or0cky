# Atmospheric moist convection and cloud physics: porting scope (P2SAoM40)

Status: DRAFT scoping, written 2026-10-05. No code was ported. Owner: not assigned (to be set by the project lead).
Review date: before the first cloud delta begins; revisit after the first instrumented dump.
Method: as for the ocean scoping (live vs dead from the rundeck's flags and namelist, line counts per subroutine,
validation by instrumented real-Fortran dumps). Standing constraint honoured: SOCRATES radiation is third-party and
is NOT scoped here; only the cloud-to-radiation hand-off arrays are identified (section 5).

## 0. Sources and how the numbers were made

- Rundeck: `modelE2_planet_2.0/decks/P2SAoM40.R` (the prod copy `ModelE_Support/prod_decks/P2SAoM40.R` differs only in
  the run-end date and a trailer; the preprocessor block is identical).
- Compile-flag state: `modelE2_planet_2.0/model/include/rundeck_opts.h:1-16` (generated from the rundeck; also
  `decks/P2SAoM40.mk:19-30`). The ONLY defined options are: USE_PLANET_RAD, GISS_RAD_OFF, NEW_IO, IRRIGATION_ON,
  CHECK_OCEAN, CONSTANT_MESO_DIFFUSIVITY, OCN_LAYERING=L13, ODIFF_FIXES_2017, EXPEL_COASTAL_ICEXS, NEW_BCdalbsn,
  CACHED_SUBDD. Nothing else is defined (no TRACERS_*, CLD_AER_CDNC, AIE_DIAG_FIX_MET, BLK_2MOM, COSP_SIM, SCM,
  CUBED_SPHERE, WEAKER_MC_LIMITS, ALT_MC_EXITS, ALT_UVSUB, ALT_CLDMIX_UV, CFMIP3_SUBDD, CALCULATE_LIGHTNING, ...).
  The compiler adds only `-DCOMPILER_Intel8 -DCONVERT_BIGENDIAN` (`config/compiler.intel.mk:11`).
- Live line counts: I ran the C preprocessor (`cpp -P -traditional-cpp -DCOMPILER_Intel8 -Iinclude`, output to a scratch
  directory, nothing written to the repo) on the three files and counted non-blank, non-comment lines per
  subroutine ("code lines"). "Raw" = lines in the file between SUBROUTINE and END (including comments and dead
  `#ifdef` text). Caveat: `-traditional-cpp` was used for counting only; the counts are good to a few percent but are
  not compiler output.
- Grid/step: 72x46 columns, LM = 40 layers (`AtmL40p.F90:13`), DTsrc = 1800 s (`P2SAoM40.R:237`); one CONDSE per
  source step.
- NOT read in full: the 552 live lines of `ISCCP_CLOUD_TYPES` (read only its call site and interface), the details of
  most of MSTCNV's downdraft/subsidence/precip loops and of LSCOND's CTEI block (read the structure, the headings, the
  call sites and a first ~330 live lines of MSTCNV; the rest by section headings and grep of calls/branches).
  Hour estimates below therefore carry that uncertainty.

## 1. Live vs dead (every dead claim has file:line)

Dead for this rundeck (all line numbers are in the real source, `model/`):

| Dead item | Evidence |
|---|---|
| All tracer branches (TRACERS_ON, TRACERS_WATER, TRDIAG_WETDEPO, AEROSOLS_*, DUST, TOMAS, AMP, SPECIAL_O18, COSMO, Shindell, `reset_tracer_work_arrays` at CLOUDS2.F90:3182-3204 and its call sites 1102, 1976, 2936) | `rundeck_opts.h` defines none of them; gated at e.g. CLOUDS2.F90:30, 298, 733, 931, 1100, 2092, 2441, 2587; CLOUDS2_DRV.F90:119, 583, 2883, 2988. There are no water tracers and no aerosol-tracer wet deposition in this build. |
| Aerosol-dependent droplet number (CLD_AER_CDNC, AIE_DIAG_FIX_MET, BLK_2MOM, ALT_CDNC_INPUTS) | not defined; gated at CLOUDS2.F90:60, 64, 368, 775, 1406-1575, 3001-3151, 4951-5225, 5399-5537; CLOUDS2_DRV.F90:31-40, 84-97, 1365-1419, 2213-2266. Cloud droplet size therefore comes from the fixed land/ocean sizes, not from aerosols. (Rundeck also sets `od_cdncx=0`, `cc_cdncx=0`, `P2SAoM40.R:200-201`, though that is moot with CLD_AER_CDNC undefined.) |
| WEAKER_MC_LIMITS / ALT_MC_EXITS / ALT_UVSUB variants | not defined; CLOUDS2.F90:5-7, 489-498, 1043-1048, 1280-1303, 1317-1356, 1687-1719, 2136-2146, 2282-2362, 2395-2439. Only the `#else`/`#ifndef` (default) arm is live. |
| SCM, CUBED_SPHERE branches | not defined; CLOUDS2_DRV.F90:19-26, 220-225, 298-326, 701-730 (the `#ifndef SCM`/`#else` arms are live), 2977-2988. |
| COSP simulator (`COSP_SIM`) incl. `save_cosp` (CLOUDS_COM.F90:541-644, 123 raw lines) and the big COSP block CLOUDS2_DRV.F90:1629-1921 | `COSP_SIM` not defined (CLOUDS_COM.F90:157, 296, 452, 540; CLOUDS2.F90:417, 919, 2625, 5059-5144; CLOUDS2_DRV.F90:266, 491, 604, 1630, 2578, 2766). |
| CFMIP3_SUBDD, mjo_subdd, etc_subdd, CLD_SUBDD extra diagnostics, CALCULATE_LIGHTNING / AUTOTUNE_LIGHTNING | not defined; CLOUDS2_DRV.F90:277, 448, 616, 972-997, 1030-1041, 2651-2714; 107-115, 289, 630-649, 910-928. |
| `read_aeractv_info`, `aeractv_streams_mod` (CLOUDS2_DRV.F90:3617-3821, about 205 raw lines) | only call site is `RAD_DRV.f:1354`, inside `#ifdef USE_OFFLINE_AEROSOLS` (RAD_DRV.f:1345-1356); that macro is not defined anywhere under `model/include/*.h` or the rundeck; the only other user is `TRAMP_rad.f:600`, which is not in the object list (`P2SAoM40.R`/`P2SAoM40.mk`). The module compiles into CLOUDS2_DRV.o (confirmed in the symbol table) but is never executed. |
| `qmom_topo_adjustments` (CLOUDS2_DRV.F90:3171-3418, about 250 raw lines, 196 code lines) | callers: `ATMDYN2.f:166` and `QUScubed.f:1215`. `ATMDYN2` is not in this rundeck's object list (the deck uses `ATMDYN MOMEN2ND`, `P2SAoM40.R` object modules, and there is no ATMDYN2.o in `model/`); QUScubed is the cubed-sphere path. `ATMDYN.f` contains no call (grep). Dead. (It is also not a cloud routine; it only lives in the clouds file.) |
| `get_cld_overlap` (CLOUDS_COM.F90:168-244, 47 code lines) | the three RAD_DRV call sites (RAD_DRV.f:2798, 3907, 5192-5205) are not present after preprocessing with this flag set (the cpp'd RAD_DRV.f contains only the `USE` statement, line 1321 of the preprocessed text; the call at 2798 sits in the `#else` of `GISS_RAD_OFF`, RAD_DRV.f:2776-3099); `DIAG.f:2048` has no call after preprocessing. So it is dead here. SOCRATES does its own overlap (not scoped). |
| `cijh_defs` / `cijlh_defs` (CLOUDS2_DRV.F90:3421-3614) | sub-daily diagnostic name tables, called from `SUBDD.f:1608,1613`; irrelevant to model state. Diagnostic metadata only, not ported. |
| The explicit DRYCNV (not in this deck) | already recorded in `fullfidelity/PHASE0_LOG.md`; unrelated to the cloud files. |

Live but NOT prognostic (diagnostics only; do not affect model state):

- `ISCCP_CLOUD_TYPES` (CLOUDS2.F90:5812-7258, 1446 raw, 552 code lines). Live because rundeck sets `isccp_diags=1`
  (`P2SAoM40.R:262`). Called from CONDSE (CLOUDS2_DRV.F90 live text near the "isccp_diags.eq.1" block). It outputs
  only `AISCCP`, `saveCTPI/TAUI/LCLDI/MCLDI/HCLDI/SCLDI/TCLDI` (diagnostics). It uses the RANDOM module for
  sub-column generation, but CONDSE brackets it with `RFINAL(seed)` ... `RINIT(seed)` (CLOUDS2_DRV.F90, around
  the `BURN_RANDOM(... NCOL*(LM+1))` calls and the final `call RINIT(seed)`), so the ISCCP draws do not alter the
  random stream seen by the next step. It can raise `stop_model` on `jerr` (error check only).
  The only coupling is the error exit; so it is optional for F1/F2 and needed only for F3 diagnostics.
- The CONDSE `AIJ/AIJL/AJ/AREG/ADIURN/SUBDD` bookkeeping (roughly 250 of CONDSE's 848 post-header code lines, by a
  coarse grep). Needed only if the `acc` diagnostic files are to be matched (F3).

## 2. Live inventory (post-preprocessing, per subroutine)

| Piece | File : raw range | Raw lines | Code lines | Role |
|---|---|---|---|---|
| `CONDSE` | CLOUDS2_DRV.F90:3-2854 | 2852 | 958 (about 480 physics/glue, about 370 diagnostics, about 110 declarations) | driver: per-column setup, MSTCNV, LSCOND, precip/energy bookkeeping, T/Q/moment/UV update, rad hand-off arrays |
| `MSTCNV` | CLOUDS2.F90:432-3207 | 2776 | 903 | moist convection (mass flux, two plumes) |
| `LSCOND` | CLOUDS2.F90:3211-5286 | 2076 | 923 | large-scale condensation, stratiform clouds, precip, CTEI, particle size and optical depth |
| `MASS_FLUX` | CLOUDS2.F90:5683-5804 | 122 | 85 | cloud-base closure iteration (called from MSTCNV) |
| `CONVECTIVE_MICROPHYSICS` | 5396-5584 | 189 | 84 | convective condensate partition into cloud/precip |
| `MC_PRECIP_PHASE` | 5588-5654 | 67 | 40 | convective precip phase change/evaporation per layer |
| `MC_CLOUD_FRACTION` | 5332-5392 | 61 | 31 | convective cloud cover per layer |
| `ANVIL_OPTICAL_THICKNESS` | 5290-5328 | 39 | 24 | convective cloud optical thickness |
| `PRECIP_MP` (function) | 5658-5680 | 23 | 9 | Marshall-Palmer precip rate |
| `get_dq_cond` / `get_dq_evap` | 7259-7336 | 78 | 27 + 27 | 3-iteration saturation adjustment (used by both MSTCNV and LSCOND) |
| `ISCCP_CLOUD_TYPES` | 5812-7258 | 1446 | 552 | diagnostic only (see above) |
| `init_CLD` | CLOUDS2_DRV.F90:2856-3169 | 314 | 131 | parameters, LMCLD/LLOW/LMID/LHI, ISCCP table read |
| `ALLOC_CLOUDS_COM`, `def_rsf_clouds`, `new_io_clouds` | CLOUDS_COM.F90:248-538 | 291 | 89 + 17 + 28 | allocation and restart plumbing |
| external, shared | `shared/Utilities.F90` `QSAT` (:33-51), `DQSATDT` (:53-67), `THBAR` (:5-31); `shared/Random_mod.F90` (72 lines) | n/a | about 35 | pure functions; `RANDU` is a 32-bit linear congruential generator (`ix = ix*69069+1`) |

Totals (live code lines): CLOUDS2.F90 2795 of 7336 raw; CLOUDS2_DRV.F90 2128 of 3821 raw (including the dead aeractv, qmom_topo and
diagnostic-definition tables, about 600 live-counted lines that are dead by call-site analysis above); CLOUDS_COM.F90 224 of 645 raw.

Counted by role:
- Core prognostic physics (MSTCNV + LSCOND + 8 helper routines): 903 + 923 + 85 + 84 + 40 + 31 + 24 + 9 + 54 = **2,153 code lines**.
- CONDSE physics/glue (non-diagnostic): about 480. init_CLD plus restart plumbing: about 265 (mostly parameter and I/O).
- Diagnostic-only but live: ISCCP 552 + CONDSE diagnostic bookkeeping about 370 = about 920.
- **Total live for the prognostic path: about 2,900 code lines (2,153 + 480 + about 265); about 3,800 if the live diagnostics are included.**
  Dead-by-call-site code in these files (aeractv, qmom_topo, cijh/cijlh, get_cld_overlap, COSP, tracer and CDNC arms):
  about 1,600 live-counted lines plus the roughly 6,000 `#ifdef`-dead raw lines already removed by the preprocessor.

This corrects the plan's line figures (`FULL_FIDELITY_PLAN.md` section 2 and Scoping (c): "MSTCNV 2.7k, LSCOND 2.0k lines"):
those are RAW counts (2,776 and 2,076) and include the dead tracer/CDNC/COSP arms and comments; the live code is about a third of that.

## 3. Call tree from the atmosphere driver, and state

```
ATM_DRV.f:264   atm_phase1 ... CALL CONDSE            (after SURFACE/MELT_SI/UPDTYPE, before RADIA at ATM_DRV.f:271)
  CONDSE  (CLOUDS2_DRV.F90:3)
    RANDU x (3*LMCLD) per column  -> RNDSS(3,L,I,J)    (stream burned per column to be decomposition-independent)
    recalc_agrid_uv / replicate_uv_to_agrid           (ATMDYN-side helpers; momentum on A-grid)
    per column (I,J):
      set up SM, SMOM(9 moments), QM, QMOM, TL, UM/VM (KMAX points)
      MSTCNV      (CLOUDS2.F90:432)
        MASS_FLUX; get_dq_cond; get_dq_evap; CONVECTIVE_MICROPHYSICS (-> PRECIP_MP);
        MC_CLOUD_FRACTION; MC_PRECIP_PHASE; ANVIL_OPTICAL_THICKNESS; QSAT/DQSATDT/THBAR
      post-convection: apply FSS partition, DDM*, TDN1, QDN1, precip PRCPMC, energy
      SC cloud (ISC=1 over ocean only if the rundeck sets ISC; default 0, not set in rundeck -> inactive; check)
      surface Richardson numbers RIS, RI1, RI2 (inputs to LSCOND CTEI)
      LSCOND      (CLOUDS2.F90:3211)
        get_dq_cond/evap; PRECIP rates; CTEI; sizes and optical thickness
      ISCCP_CLOUD_TYPES (diagnostic)
      build rad hand-off arrays w_cloud, frac_*_water/ice, mix_ratio_*, dim_char_*, frac_area_*
      write back T, Q, moments, QCL, QCI, UKM/VKM (momentum), PREC, EPREC, PRECSS
    ISCCP seed reset, momentum back to the native grid, diagnostics
RADIA (ATM_DRV.f:271) -> consumes the hand-off arrays (SOCRATES, out of scope)
```

State.
- Carried across steps (restart: `CLOUDS_COM.F90:481-532`): `TTOLD, QTOLD, SVLHX, RHSAV, CLDSAV, AIRX, LMC`
  (the `NCL, NCI` entries are in the CLD_AER_CDNC arm and are not live). Also needed but written by other
  modules: ATM `T, Q, QCL, QCI, TMOM/QMOM, PMID/PEDN/PDSIG, MWs` (mass flux through each layer, `SDL`), `EGCM, W2GCM, DCLEV, PBLHT, PBLPTOP`
  (PBL/ATURB), the surface averages (`TSAVG, QSAVG, USAVG, VSAVG, TGVAVG, QGAVG`), `FOCEAN/FLAND/FEARTH/RSI`, `SNOAGE`.
  Per-step recomputed/stored (not in restart): `TAUSS, CLDSS, TAUMC, CLDMC, SVLAT, CSIZSS, CSIZMC, FSS, CLDSAV1, QLss/QIss/QLmc/QImc, DDM1, DDMS, TDN1, QDN1, DDML, w_cloud, frac_*`.
- The random-number state `IX` (module RANDOM) is carried in the restart parameter `IRAND` (`MODELE.f:439, 533`, `MODEL_COM.F90:158`). It advances by
  3*LMCLD draws per column per step. I did NOT verify that LSCOND results depend on anything other than the three recorded `RNDSS` values per layer.
- Module-level scratch in `module CLOUDS` (CLOUDS2.F90:14-260) is per-column working storage reset each call (not persistent state), except the namelist-driven tunables below.
- Tunable inputs from the rundeck/namelist (all via `init_CLD` `sync_param`): `U00a=0.695, U00b=0.60, WMU_multiplier=1, WMUI_multiplier=0.001, radiusl_multiplier=1.01, radiusi_multiplier=1, use_vmp=1` (`P2SAoM40.R:205-215`);
  everything else (LMCM, MAXCTOP, HRMAX, ISC, do_blU00, entrainment_cont1/2, funio_denominator, autoconv_multiplier, cld_plow/pmid) takes its code default:
  values to be confirmed from a real run's `P2SAoM40.PRT` rather than assumed.
  `use_vmp=1` activates the precipitation-in-optical-depth branches (CSIZELIP, TAUSSLIP, WMPR; CLOUDS2.F90 around 4960-5040) and a
  `stop_model('VMP: should not be here')` guard (CLOUDS2.F90:5033).
- External library dependencies: none from SOCRATES/Ent. Only `CONSTANT`, `RESOLUTION`, `MODEL_COM`, `QUSDEF` (nmom=9 moment indices `xymoms/zmoms`), the three
  pure functions above, `RANDOM`, and (for the diagnostics) `DIAG_COM/subdd`. No radiation call and no tracer hook is executed inside the physics. The coupling to SOCRATES is one-way data (section 5).

## 4. Validation boundaries (what to dump from the instrumented real model)

Same mechanism as earlier deltas (`fullfidelity/instrumentation/build_and_run.md`: copy the tree, apply patches to the copy, `ffdump` stream files read with `ffdump_reader.py`). New patches would be needed (none exist for the cloud files; I checked the patch directory listing only partially, patch files seen are for ATM_DRV/surface/seaice/ocean). Units must be re-grepped before reuse (the build notes warn about unit collisions in the whole tree).

1. **Whole-column boundary at the MSTCNV call (CLOUDS2_DRV.F90:896) and the LSCOND call (:1352).** One record per column per step (3,312 columns), dumped for a few sampled dates as in earlier deltas.
   - MSTCNV inputs: PL, PLE, PLK, AIRM, TL, TVL, SM, SMOM(9,40), QM, QMOM(9,40), QCLL, QCIL, SDL, DPDT/WTURB/GZL/ETAL/W2L, UM/VM(KMAX,LM), DCL, PEARTH, DXYPIJ, plus the namelist tunables.
     Outputs: SM, SMOM, QM, QMOM, UM, VM, MCFLX, DGDSM, DPHASE, DTOTW, TAUMCL, CLDMCL, SVLATL, SVWMXL, CSIZEL, PRCPMC, PRECNVL, LMCMIN/LMCMAX, DDMFLX, TDNL, QDNL, CNVMMRL, CLDSLWIJ, CLDDEPIJ, FSSL, WMSUM, WMCLWP, IERR/LERR.
   - LSCOND inputs: TL, QL, TH, SMOM, QMOM, AIRM, PL/PLE, QCLX/QCIX, RNDSSL(3,40), TTOLDL, CLDSAVL, RH, FSSL, SVLHXL, SVLATL, TAUMCL, CLDMCL, PRECNVL, AQ, DPDT, ZPBL, RIS/RI1/RI2, DCL, ISC/ROICE/PLAND/PEARTH, and the module scratch it reads (check the exact list at the dump-patch stage).
     Outputs: TL, QL, TH, QCLX/QCIX, CLDSSL, CLDSAVL, CLDSV1, RHSAV(RH), TAUSSL, TAUSSLIP, CSIZEL(IP), SVLHXL, LHP, WMPR, PRCPSS, HCNDSS, QLss/QIss, DCTEI/SSHR, UM/VM, IERR/WMERR.
   - Taking `RNDSS` as a RECORDED INPUT is the simplest and sufficient choice. `RANDU` is a 32-bit LCG and could later be ported exactly (it is integer arithmetic, bitwise reproducible), but the per-column BURN_RANDOM bookkeeping adds nothing to the physics validation.
2. **Per-call boundary for the small helpers**, same shape as D54's `KPPMIX`: dump every call at the real call sites (`get_dq_cond` CLOUDS2.F90:1364, 4425; `get_dq_evap` 2067, 2716, 4002, 4013; `CONVECTIVE_MICROPHYSICS` 1938; `MC_CLOUD_FRACTION` 2675; `MC_PRECIP_PHASE` 2703; `ANVIL_OPTICAL_THICKNESS` 3114; `MASS_FLUX` 1056). Expect very large record counts; subsample (for example every Nth call) and cap file size. Some helpers have many optional-looking arguments (MASS_FLUX/CONVECTIVE_MICROPHYSICS take argument lists of 10-20 items); record all INTENT(IN) and INTENT(OUT) items.
3. **Restart/state boundary:** the 7 carried arrays (`TTOLD, QTOLD, SVLHX, RHSAV, CLDSAV, AIRX, LMC`) come straight from the real restart `fort.1.nc` (`new_io_clouds`) and the previous step's outputs; they are inputs, never "ported state", at F0/F1.
4. **State that must be taken as recorded input (not computed by the port):** PBL/ATURB outputs (`EGCM, W2GCM, DCLEV, PBLHT, PBLPTOP`), the surface averages `atmsrf%*AVG`, the dynamics mass flux `MWs`, `PMIDOLD`, the A-grid winds, the already-ported surface-type fractions, and `RNDSS`. The ISCCP `tautab`/`invtau` tables are data (read from the `ISCCP` input file) and only matter for the diagnostic.
5. **CONDSE output boundary** (for the glue delta): `T, Q, TMOM, QMOM, QCL, QCI, U/V tendencies (UKM/VKM -> UA/VA), PREC, EPREC, PRECSS, P_acc/PM_acc, SNOAGE`, the 7 restart arrays, and the hand-off arrays `w_cloud ... frac_area_cnv` (that last set is the F1 interface to radiation).

Tolerance/validation cautions drawn from earlier deltas: `QSAT` uses `exp` and `LSCOND` uses `**` and `exp` extensively; D54 showed `pow/exp/log` can differ by 1 ULP from Intel's library. Expect rounding-level (1e-12 to 1e-15 relative) agreement, not bitwise. More important, the code is full of threshold tests (`RH` vs `U00`, `QSATMP` vs `QMP`, `DMSE>-1e-10`, `FPLUME<=.001`, cloud-fraction clamps, phase tests `TL<TF`): a 1-ULP perturbation near a threshold can flip a branch, as noted in the Round 2 marginal-cell flips. The validation therefore needs (a) the same mutation/non-vacuity checks as earlier deltas, (b) a statistic of "fraction of columns/layers whose branch path matches" in addition to field tolerances, and (c) a policy for a flipped branch (report, do not hide). Only columns where convection actually fires are informative for MSTCNV; I do not know the fraction of active columns from the dumps (not measured).

## 5. Interfaces to excluded or neighboring components

- SOCRATES radiation consumes, per column, `w_cloud, frac_st_water, frac_st_ice, frac_cnv_water, frac_cnv_ice, mix_ratio_st_water/ice, mix_ratio_cnv_water/ice, dim_char_st_water/ice, dim_char_cnv_water/ice, frac_area_st, frac_area_cnv` (set in CONDSE; read in `RAD_DRV.f` at about preprocessed lines 1290-1330 and 1742ff). `RAD_DRV.f:1598-1599` also zeroes CLDSS/CLDMC where `TAUSS/TAUMC <= taulim` on the radiation side. Porting the cloud physics gives F1 parity up to these arrays; the radiation side stays on the recorded Fortran or on the existing stand-in.
- Precipitation (`PREC, EPREC, PRECSS`) feeds the PRECIP_* routines already ported (D26-D28, D33 per the ledger), so the cloud delta can be validated end to end against existing ported consumers.
- The 40-layer, 9-moment (`nmom`) second-order-moment advection scheme (QUS) owns the `TMOM/QMOM` fields; MSTCNV and LSCOND update them in place (`SMOMMC/SMOMLS`). Those moment updates are a large share of the line count and must be ported with the same sign/units conventions (`xymoms` vs `zmoms` split).

## 6. Recommended porting order (small, validatable pieces)

Principle: ascending size and branching, every piece validated against real per-call or per-column dumps before the next. Composition (running MSTCNV then LSCOND then CONDSE glue) is the F1 check.

| # | Piece | Code lines | Why here |
|---|---|---|---|
| D-C1 | **`get_dq_cond` + `get_dq_evap`** (and `QSAT`, `DQSATDT`, `THBAR` as pure helpers; the repo already has qsat-type code in `pbl_ff.py`/`ghy_ref.py`, to be checked for equivalence rather than reused blindly) | 54 + about 35 | FIRST TARGET, see below |
| D-C2 | `PRECIP_MP`, `ANVIL_OPTICAL_THICKNESS`, `MC_CLOUD_FRACTION`, `MC_PRECIP_PHASE` | 9 + 24 + 31 + 40 | all stateless, per-layer, small argument lists |
| D-C3 | `CONVECTIVE_MICROPHYSICS` | 84 | stateless per layer, but 4 phase branches |
| D-C4 | `MASS_FLUX` | 85 | fixed-count iteration with convergence tests, uses QSAT/DQSATDT/THBAR |
| D-C5 | LSCOND tail: particle size and optical thickness block (the last block, "COMPUTE CLOUD PARTICLE SIZE AND OPTICAL THICKNESS", CLOUDS2.F90:4927-5286), validated by instrumenting the LSCOND call and the intermediate arrays at that block's entry | about 200 | per-layer independent given CLDSSL, QCLX/QCIX, WMPR; uses `use_vmp` arm. Natural first JAX shape: elementwise over (L,I,J) |
| D-C6 | LSCOND main layer loop (condensation, autoconversion, precip, phase, cloud fraction) + CTEI + RH update | about 700 | sequential in L (top-down precip carry: `PREBAR`, `PREICE`), branchy; the CTEI loop is a bounded iteration (`do ITER=1,9`, with `exit`) that can re-mix a layer pair |
| D-C7 | `MSTCNV` in full (cloud-base loop x 2 plume types x 2 area partitions x ascent loop, then downdraft, subsidence, precip/evap loop, optical-thickness loop) | 903 | the largest and most branchy item; validate at the column boundary and with per-column internal checkpoints |
| D-C8 | `CONDSE` glue (per-column setup/post-processing, precip and energy bookkeeping, SNOAGE, momentum back-transfer, rad hand-off arrays) | about 480 | validated at the CONDSE output boundary against ported consumers |
| D-C9 | optional: CONDSE diagnostics (AIJ/AJL/ADIURN etc.) and `ISCCP_CLOUD_TYPES` (552 lines) | about 920 | only for F3-style `acc` comparison; defer as in previous decisions |
| D-C10 | `init_CLD`, constants/tunables, restart read of the 7 arrays | about 265 | mostly configuration; take LMCLD/LLOW/LMID/LHI/XMASS/BYBR as recorded constants first |

### First target and concrete first-delta plan (D-C1)

`get_dq_cond` and `get_dq_evap` (CLOUDS2.F90:7259-7336, 27 live lines each): pure scalar functions, no module state, no random numbers, no branching beyond `QM>0`/`COND>0` and a final `max/min` clamp, a fixed 3-iteration Newton-style saturation adjustment, inputs `(sm, qm, plk, mass, lhx, pl[, cond])`, outputs `(dqsum, fcond|fevp)`. They are the shared kernel of both MSTCNV (updraft condensation at line 1364, downdraft evaporation at 2067, precip re-evaporation at 2716) and LSCOND (4002, 4013, 4425), so every later piece reuses the validated function. They are the smallest stateless live routines with real subroutine boundaries (`PRECIP_MP` is smaller but trivial and has no call-site variety).

Plan:
1. Instrumentation: a patch on a copy of the tree (never the original) that writes, at each call, the INPUT scalars and the OUTPUT scalars to a stream file, tagged with a call-site id (the six sites above), subsampled (for example every 50th call) and with a record cap, for the standard sweep of dates. No restart change is needed. Document the patch in `fullfidelity/instrumentation/` per the existing convention (note that if the scratch tree is rebuilt, patch headers must say whether they were generated by `diff`).
2. Port as a numpy function vectorized over a flat record axis (a trivial loop of 3 fixed iterations), plus `qsat`, `dqsatdt`, `thbar` with the exact Fortran operation order (`A*exp(LH*(B-C/max(130,TM)))/PR`, constants built from `CONSTANT` values as in `Utilities.F90:33-51`). Watch the single-precision-literal hazard seen in D54 (check every literal in the helpers for `d0` suffixes: `get_dq_cond` has `0d0` and `1.` constants; `DQSUM=0.` and `1.+SLH*...` are mixed-precision literals, to be checked against the compiler's treatment).
3. Tests: comparison to the real dumps per call site; non-vacuity (a meaningful fraction of records with `dqsum>0`), mutation checks (change the iteration count, the clamp, the `QM>0` guard; tests must fail), and a report of the maximum relative error and any 1-ULP `exp` effects as in D54.
4. Done criterion: agreement at rounding level on all sampled records at the agreed tolerance (to be fixed before coding, per plan section 1); the ledger entry records records-per-site and error statistics.
Estimated 2-3 hours including the patch (the instrumentation, not the 54-line port, is the bulk).

### Parts that are branch-heavy or sequential, and suggested batching/JAX shapes

- Everything is naturally per-column (3,312 independent columns per step; the physics has no horizontal coupling inside CONDSE except the pole averaging of winds before/after). Outer batching axis = columns, as for the PBL/GHY/lake ports.
- LSCOND: sequential in L (precipitation carried downward, `PREBAR`, `PREICE`) and a CTEI pass with a data-dependent bounded iteration. Shape: `vmap` over columns, `lax.scan` over L for the main loop, a masked fixed-trip (9) loop for CTEI; the size/optical-thickness block is elementwise (L,I,J). Branches as `jnp.where` with the NaN-in-dead-branch caveat already in plan section 3.
- MSTCNV: the worst case. The `CLOUD_BASE` loop runs over up to LMCM-1 base levels with `cycle` and early `exit`; for each base, 2 plume types, 2 area partitions (the second only if the first redo condition holds), an ascent `do L=LMIN+1,LM` with early exits, then a downdraft loop and subsidence over L. The control flow depends on previous plume results (`MCCONT`, `MC1`, `FMC1`, `TPSAV`, `LDRAFT`), so a faithful per-column function with data-dependent loops is needed first (numpy, as the earlier pieces). For JAX: a masked fixed-trip formulation (bases x types x partitions x ascent) under `vmap` over columns will waste work on the many columns where no convection occurs; recommend compacting the active columns (gather by "any base unstable" using the cheap `DMSE>-1e-10` pre-test) before the expensive kernel, as the earlier work did with dead-branch masking. Do not start JAX until the numpy reference matches.
- Moment arrays (NMOM=9) double the work in both routines: carry them as a trailing axis, not as separate arrays.
- The helper routines (`get_dq_*`, `PRECIP_MP`, `MC_*`, `ANVIL_*`, `CONVECTIVE_MICROPHYSICS`) become elementwise array functions; they are the natural unit tests for the JAX versions.

## 7. Effort estimate (hours; basis stated)

Observed plan rate: about 1 h per 1,000 Fortran lines for stateless pieces; more for stateful or branching code. Here the line figures are code lines. Multipliers below are my judgement, not measured for this code.

| # | Piece | Hours | Basis |
|---|---|---|---|
| D-C0 | Instrumentation harness for the cloud files (record layout, dump patches for CONDSE column boundary and per-call helpers, reader changes, non-vacuity statistics) | 5-8 | six-plus patches, big argument lists, unit-number checking; D54's per-call dump shape reused |
| D-C1 | `get_dq_cond/evap` + QSAT/DQSATDT/THBAR | 2-3 | 54 lines stateless; most cost is the call-site dumps |
| D-C2 | PRECIP_MP, ANVIL_OPTICAL_THICKNESS, MC_CLOUD_FRACTION, MC_PRECIP_PHASE | 3-4 | 104 lines, stateless, small branching; several call sites |
| D-C3 | CONVECTIVE_MICROPHYSICS | 2-3 | 84 lines, phase branches, long argument list |
| D-C4 | MASS_FLUX | 2-3 | 85 lines, iteration with convergence tests |
| D-C5 | LSCOND size/optical thickness block | 3-5 | about 200 lines, per-layer; `use_vmp` arm; needs internal checkpoint dumps |
| D-C6 | LSCOND main layer loop + CTEI | 12-20 | about 700 lines, sequential, many thresholds; 15-30x the stateless rate |
| D-C7 | MSTCNV | 20-35 | 903 lines, deeply nested data-dependent loops, momentum and 9-moment bookkeeping; the highest-risk piece |
| D-C8 | CONDSE glue | 6-10 | about 480 lines, bookkeeping but many state arrays, UV re-gridding and rad hand-off arrays |
| D-C10 | init_CLD + constants/restart arrays | 2-3 | configuration, mostly recorded constants |
| | **Subtotal faithful numpy port of prognostic path (D-C0 to D-C10, without D-C9)** | **about 57-92, central about 70** | |
| D-C11 | JAX vectorization of the validated pieces (LSCOND scan/vmap; MSTCNV masked/compacted; helpers elementwise), jit checks and speed measurement, following earlier JAX deltas (D14-D16, D83-D86) | 12-20 | earlier JAX conversions were about 1-3x the numpy effort for simpler pieces; MSTCNV is the hard case |
| D-C9 | optional: CONDSE diagnostics (AIJ etc.) | 4-8 | about 370 lines of bookkeeping, only for F3 `acc` comparison |
| D-C9b | optional: ISCCP_CLOUD_TYPES | 8-14 | 552 lines, RNG-driven sub-columns, table lookups; diagnostic, and I did not read its body |
| | **Total including JAX, excluding optional diagnostics** | **about 70-110, central about 85** | |
| | **Total including optional diagnostics and ISCCP** | **about 85-135** | |

Order-of-magnitude check against the plan: core 2,153 lines at 1 h/1,000 lines would be about 2 h; the estimate is 25-35x that because the code is branchy, nested, sequential and per-column stateful, and because the instrumentation, validation harness and threshold-flip analysis dominate. The earlier ocean estimate used the same logic (the README's "ocean 20-35 h" for a larger line count of mostly stateless array code), so these cloud hours should be read as larger per line for a reason, not as a mismatch. This should be re-baselined after D-C1 to D-C3 (measured hours per line on these routines).

**Confidence.** Line counts and the live/dead classification of the rundeck flags: high (flags read from `rundeck_opts.h`; call-site claims checked by grep and by cpp). Call tree: high for CONDSE/MSTCNV/LSCOND/helpers. State list: medium (restart arrays read from `CLOUDS_COM.F90`; per-step recomputed arrays inferred from CONDSE). Hours: low-to-medium; MSTCNV (20-35 h) and LSCOND main loop (12-20 h) are the dominant uncertainty and I read only part of them. Probable bias: underestimate if threshold-flip behavior (section 4) turns out to need a statistical rather than per-column acceptance criterion, which would add validation design time.

## 8. Risks

1. Branch flips near thresholds make "matches the real model" a statistical claim for MSTCNV/LSCOND, not a per-column bitwise one; needs an agreed acceptance rule before coding (plan section 1 already asks for this).
2. `exp`/`pow` 1-ULP differences (D54) feed threshold tests.
3. Single-precision literals in Fortran (`1.`, `.5`, `2./3.`, ...) change double arithmetic; the cloud files use many mixed literals (for example `PGRAD = 0.7`, `.16667D0`, `1.d0`, plain `.5`); each needs checking against what ifort actually does, as in D54.
4. Recorded-input dependence: PBL/ATURB turbulence (`EGCM`, `W2GCM`), dynamics mass fluxes (`MWs`), surface averages, `RNDSS` all come from other components; F1 composition requires those components to match first.
5. Dead-code claims depend on the exact flag set; any rundeck change (for example turning on `CLD_AER_CDNC`, tracers or COSP) revives about 1,600 live-counted lines plus the arms listed in section 1.
6. The restart/IRAND handling of the random stream: if an end-to-end multi-step run is attempted, `RANDU`/`BURN_RANDOM` must be ported exactly (integer wraparound, signed 32-bit) or `RNDSS` recorded every step.

## 9. What I did not read or could not determine

- I did not read the bodies of `ISCCP_CLOUD_TYPES` (552 live lines), most of `MSTCNV` after about live line 330 (downdraft, subsidence, precip/evap, optical-thickness loops were scanned by headings and grep only), `LSCOND`'s main loop and CTEI block (scanned by headings, grep of random-number, `use_vmp`, error and loop statements), `qmom_topo_adjustments`, `save_cosp`, `read_aeractv_info`, or the SUBDD tables. The per-routine descriptions there are from headings and call sites, not full reads.
- I did not determine the actual values of the code-default tunables (LMCM, MAXCTOP, ISC, HRMAX, entrainment constants, cld_plow/pmid); they should be read from a real run's `.PRT` file.
- I did not confirm whether `ISC` (stratocumulus-over-ocean branch) is 0 for this rundeck: it is not set in `P2SAoM40.R`, and the code default is `ISC=0` (CLOUDS2.F90 module header), so it is presumed inactive; the CONDSE block `if (ISC.eq.1 .and. FOCEAN>.5)` would then be dead, but this was not checked against a run.
- I did not measure the fraction of columns in which MSTCNV actually fires, the call counts per helper, the dump sizes, or the cost of running the instrumented model; these are needed to size the dumps (section 4).
- I did not verify the restart storage of the RANDOM seed beyond finding the `IRAND` parameter handling, and did not check how `MWs`, `PMIDOLD`, `UKM/VKM` are produced upstream (ATM_DRV/ATMDYN), which affects what must be recorded.
- The preprocessed counts used `cpp -traditional-cpp`, not the Intel compiler's preprocessor; a few lines may be miscounted (for example continuation lines, `//` string-concatenation). Raw ranges are from the unprocessed source and are exact.
- I did not check what ported pieces already cover the radiation-side overlap behavior or whether any existing Python module (for example `pbl_ff.py`) has a `qsat` bit-identical to `Utilities.F90`; that is a first-step check in D-C1.
- No code was executed apart from `grep`/`sed`/`awk`/`cpp` read-only inspection with output to the scratch directory.
