# Atmospheric dynamics: scoping for the full-fidelity port (P2SAoM40)

Status: DRAFT scoping document, no code written. Date: 2026-10-05. Review by: next session that starts atmosphere work.
Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session. Project-local per `projects/imvi/AGENTS.md`.
Sources: real Fortran at `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model/` (read-only) and `decks/P2SAoM40.R`, `decks/P2SAoM40.mk`,
`model/include/rundeck_opts.h`; repo docs `Reports/README_START_HERE.md`, `FULL_FIDELITY_PLAN.md`, `fullfidelity/instrumentation/build_and_run.md`.
Standing constraints honoured: SOCRATES not scoped; no secrets; nothing in this document is a measured result unless it says so.
All Fortran paths below are relative to the `model/` directory unless stated. "Range lines" = first to last line of the routine; "code" = non-blank, non-comment lines (counted with awk).

## 1. Bottom line

- Live code for this rundeck: about **6,200 range lines (about 4,700 code lines)**, against the plan table's "~8,500". The difference is dead code (section 2), mostly `STRATDYN.f`, `UNRDRAG`, the V2 `PGF`, the `QUS3D.f` non-limited variants, and files that are not compiled at all.
- Dynamics is the **start** of each 30-minute physics step (`atm_phase1`, `ATM_DRV.f:105`), not interleaved with physics inside the step. Only the sea-level-pressure filter, `DISSIP`, and the pressure-gradient/tropopause/vertical-velocity exports touch the physics boundary.
- Per 30-minute step the real model runs 5 leapfrog sub-steps of 450 s (1 forward, 1 backward, 2 even, 1 odd; 4 dynamics steps of `DT`=450 s), 2 temperature (QUS) advections, 1 moisture (QUS3D) advection with flux cycles, 2 SDRAG calls, an end-of-step velocity filter chain, and (after physics) the SLP filter.
- **Proposed first target:** the end-of-`DYNAM` velocity filter chain (`FLTRUV`, `fltry2` for U and V, `CONSERV_AMB_EXT`, `ADD_AM_AS_SOLIDBODY_ROTATION`): about 330 range lines, stateless, no FFT, no data-dependent iteration. Plan in section 7.
- **Effort (central): about 50 hours, range 35-80**, for the full live dynamics block including instrumentation, batched numpy/JAX versions and a chained one-step validation. Faithful numeric port alone about 35 h. Confidence: low-moderate. This is higher than the README's 12-20 h for dynamics; section 9 explains the gap.
- **Key risks:** data-dependent cycle counts in `AADVQ0`/`AADVTX` that the 6-step test windows may never exercise (unvalidated branches); `**KAPA` pow rounding and reduction-order differences (the D54/D45 lessons); half-polar-box special cases at J=1, 2, JM-1, JM; a leapfrog scheme whose intermediate arrays need per-call dumps.

## 2. Live versus dead, from the rundeck

Rundeck state (all cited):
- `#define` set is exactly: `USE_PLANET_RAD`, `GISS_RAD_OFF`, `NEW_IO`, `IRRIGATION_ON`, `CHECK_OCEAN`, `CONSTANT_MESO_DIFFUSIVITY`, `OCN_LAYERING L13`, `ODIFF_FIXES_2017`, `EXPEL_COASTAL_ICEXS`, `NEW_BCdalbsn`, `CACHED_SUBDD` (`decks/P2SAoM40.R:19-30`, `decks/P2SAoM40.mk:20-30`, `model/include/rundeck_opts.h:4-14`). Not defined: `USE_FVCORE`, `CUBED_SPHERE`, `SCM`, `TRACERS_ON`, `TRACERS_OCEAN`, `PLANET_PARAMS`, `NUDGE_ON`, `V2_PGF`, `V2_ATMDYN_TIMESTEPPING`, `V2_PSURF_FILTER`, `STDHYB`, `DO_CO2_CONDENSATION`, `ALT_CLDMIX_UV`. I only read the `CPP_OPTIONS` lines and `rundeck_opts.h`; I did not check the Makefile for additionally injected `-D` flags (grep of `decks/P2SAoM40.mk` and `model/Makefile` found none).
- Resolution/time: 72x46, `LM=40`, `LS1=24` (150 mb), `PSF=984` (`AtmL40p.F90:12-45`; `PLANET_PARAMS` undefined so the non-planet branch applies). `DTsrc=1800.`, `DT=450.` (`P2SAoM40.R:237-238`), so `NIdyn = 2*nint(.5*dtsrc/dt) = 4` (`ATMDYN.f:56`).
- `planet_name='Earth'` because `PLANET_PARAMS` is undefined (`shared/Constants_mod.F90:255-258`), so `aflux_topo_adjustments` and `pfilter_using_slp` stay `.true.` (`ATMDYN.f:21,38`; Earth test at `ATMDYN.f:83-87`).
- Filter params: `DT_XUfilter=DT_XVfilter=450`, `DT_YUfilter=DT_YVfilter=0` (`P2SAoM40.R:240-243`); `MFILTR=1`, `NFILTR=1`, `DO_POLEFIX=1`, `ANG_UV=1` defaults (`ATMDYN_COM.F90:51`, not overridden in the rundeck). SDRAG: `X_SDRAG=.002,.0002`, `C_SDRAG=.0002`, `P_sdrag=1`, `PP_sdrag=1`, `P_CSDRAG=1`, `Wc_JDRAG=30`, `ANG_sdrag=1` (`P2SAoM40.R:180-187`); `rtau` not set.

| Item | Status | Evidence (file:line) |
|---|---|---|
| `ATMDYN2.f` (2,138), `ATMDYN3.F90` (3,492), `ATMDYN_SCM.f` (540), `ATMDYN_SCM_EXT.f` (56), `MOMEN4TH.f` (616), `STRATDYN.f` (1,554), `STRAT_DIAG.f` (976), `QUScubed.f` (1,858), `TQUS_DRV.f` (1,018), `TQUS_XZYZX.f` (1,071), `NUDGE.f` (480), `ATM_DUM.f` (1,001) | **not compiled** | Not in `OBJ_LIST` (`decks/P2SAoM40.mk:32-97`). Only `ATMDYN MOMEN2ND QUS_DRV QUS3D STRAT_DUM ATMDYN_COM ATM_UTILS QUS_COM QUSDEF FFT72` are listed (`.mk:42-44,56-58,36`). Rundeck comment: "GISS dynamics without gravity wave drag" (`P2SAoM40.R:41`). I did not read these files beyond line counts. |
| `GWDRAG`, `VDIFF`, `INIT_GWDRAG`, `EPFLUX` calls inside `DYNAM` | **no-ops** (empty entries) | `STRAT_DUM.F90:2-15`; call sites `ATMDYN.f:264-265,280-281,297-298,317-319,339`; `EPFLUX` at `ATMDYN.f:358` |
| FV-core dynamics, `SDRAG` at `ATM_DRV.f:114-116`, FV diag blocks | dead | `#if defined(USE_FVCORE)` at `ATM_DRV.f:103,108,114,118,122,126` |
| Cubed-sphere `GWDRAG`/`SDRAG` column block, cubed `PGRAD_PBL` | dead | `ATM_DRV.f:237-241`; `ATM_UTILS.f:3-68` (live version `ATM_UTILS.f:69-174`) |
| `TrDYNAM`, tracer parts of `FILTER`, tracer diagnostics | dead | `ATM_DRV.f:146-148`; `ATMDYN.f:1441-1444,1573-1591,3104-3152` |
| `V2_PGF` first `PGF` (257 lines) | dead | `ATMDYN.f:846` `#ifdef V2_PGF`, `#else` at 1105; live `PGF` is `ATMDYN.f:1107-1324` |
| `V2_ATMDYN_TIMESTEPPING` branches | dead (live branch: `VDIFF` at 319, `PGF` with `MODD1` at 349) | `ATMDYN.f:208-210,318-320,338-350` |
| `NUDGE_ON` calls | dead | `ATMDYN.f:253-256,275-277,292-294,312-314` |
| `DO_CO2_CONDENSATION` | dead (`dmCO2cond(:,:)=0` at 262 is a live trivial reset of a `FLUXES` array) | `ATMDYN.f:259-262`; `ATMDYN_COM.F90:461-464` |
| `UNRDRAG` module + 3 routines (673 lines) | dead (`USE_UNR_DRAG=0` default, not set in rundeck) | `ATMDYN_COM.F90:70`; `ATMDYN.f:353` (`SDRAG` live), `368-371` (dead); definitions `ATMDYN.f:3155-3827` |
| `SDRAG` linear (`rtau`) branch | dead | `linear_sdrag = is_set_param('rtau')` at `ATMDYN_COM.F90:375`; branch `ATMDYN.f:2038-2058`; default branch live |
| `FILTER` temperature filter (`MFILTR>=2`) | dead | `MFILTR=1` default (`ATMDYN_COM.F90:51`); `ATMDYN.f:1468,1597-1623` |
| `FILTER` non-SLP surface-pressure branch | dead (Earth) | `ATMDYN.f:1481,1517-1545` |
| `FILTER` `V2_PSURF_FILTER` blocks | dead (else branch live) | `ATMDYN.f:1430-1432,1501-1509` |
| `STDHYB` branches | dead (`#ifndef STDHYB` branches with `MFIXs` are live) | `ATMDYN.f:495-499,726-730,1433-1435,1550-1552`; `ATMDYN_COM.F90:151,179,200,220,248,299` |
| QUS3D `*2` variants `AADVQX2/Y2/Z2`, `AADVQZ2_COLUMN` (567 lines) | dead (`QDYNAM` calls `AADVQ` with `qlimit=.TRUE.`; the only other caller `TrDYNAM` is dead) | `ATMDYN.f:3082`; branch selection `QUS3D.f:193,216-223,229,239-251,273-277`; bodies `QUS3D.f:1690-2178,2303-2382` |
| `prather_limits` branches | dead (default 0) | `QUSDEF.f:36`; use sites `QUSDEF.f:210,546,617` |
| `limitq` / `apply_limiter` on the temperature advection path | dead for `AADVT` (called with `qlimit=.false.`), live in `CLOUDS2` | `ATMDYN.f:337`; gates `QUSDEF.f:114,318`. These kernels are **shared with moist convection** (`CLOUDS2.F90:2385,2426`). |
| `MOMEN4TH.f` | not compiled; `MOMEN2ND` is the object, `moment_enq_order` returns 2 | `MOMEN2ND.f:22-28`; effect on `AVRX` `xAVRX=1` at `ATMDYN.f:1361-1365` |
| `conserv_AM` (A-grid angular momentum), `DIAGCD`, `DIAG5D`, `DIAG5F`, `COMPUTE_MASS_FLUX_DIAGS`, `COMPUTE_DYNAM_AIJ_DIAGNOSTICS`, `DIAGA/DIAGB` | diagnostics only, not ported | Accumulate `AIJ/AJL/AGC`; they do not feed the state. I did not read `DIAGCD`/`DIAG5D`/`DIAG5F` bodies, so "no state feedback" is an inference from names and call patterns. |

Live branches that need care (all confirmed live): `do_polefix=1` paths (`ATMDYN.f:606-611`, `MOMEN2ND.f:236-337,446-476`, `ATMDYN.f:1291-1303`); AFLUX topography adjustment (`ATMDYN.f:613-706`); default (non-linear) SDRAG with `Wc_JDRAG` and `ANG_SDRAG=1` (`ATMDYN.f:2060-2134`); SLP-based `FILTER` (`ATMDYN.f:1481-1515`).

## 3. Where dynamics sits in the step, and the call tree

Main loop `MODELE.f` (read lines 295-334): `atm_phase1` -> `PRECIP_SI`/`PRECIP_OC` -> `SURFACE` -> `ocean_driver` -> `atm_phase2`; then clock tick and `dailyUpdates` at day boundaries (which call `DAILY_ATMDYN`).

`atm_phase1` (`ATM_DRV.f:3-294`), live dynamics-relevant order:
1. `MAOLD=MA`, `PMIDOLD=PMID` saved (`ATM_DRV.f:88-91`); `CONSERV_SE`, `CONSERV_KE` -> `SEINIT`,`KEINIT` (93-94).
2. `CALL DYNAM` (105): the leapfrog block below, ending with `FLTRUV` and the N-S filter.
3. `COMPUTE_DYNAM_AIJ_DIAGNOSTICS` (diagnostic, 131); `COMPUTE_WSAVE` (134, vertical velocity m/s from `MWs`).
4. `QCL,QCI` rescaled by `MAOLD/MA` (136-142).
5. `CALL QDYNAM` (143): moisture advection with the integrated fluxes `MUs,MVs,MWs`.
6. Energy-conservation fix (176-205): `CONSERV_SE/KE` again, `GLOBALSUM`, uniform `dSEpKE` subtracted from `T`.
7. `CALC_TROP` (212; `PTROPO,LTROPO`), `PGRAD_PBL` (216; `dpdx_by_rho` etc., read by `PBL_DRV.f`), `calc_kea_3d` (235, `KEA` saved for `DISSIP`).
8. Then column physics: `MELT_SI`, `CONDSE` (clouds, ~264), `RADIA` (274). Existing dumps `pre_condse`/`post_condse`/`post_radia` bracket these (`ATM_DRV.f.patch:7-17`).

`SURFACE` (includes PBL and `ATM_DIFFUS` at `SURFACE.f:1172`) then runs, which changes `U,V,T,Q` on the first layer(s).

`atm_phase2` (`ATM_DRV.f:426-525`): `ATM_DIFFUS(2,LM-1)` (458, ATURB above the PBL), `DISSIP` (466: KE lost since step 7 added back as heat), then `FILTER` (482, every step because `NFILTR=1`). So the SLP filter is the last dynamics operation of the step and consumes `ATMSRF%TSAVG` from the surface code.

`DYNAM` (`ATMDYN.f:186-390`) per call, with `NS` counting 4 down to 1:
```
300: MASUM; UX=UT=U; VX=VT=V; TZ=TMOM(MZ)            [leapfrog (re)initialisation]
 initial forward (MRCH=0, DTFS=2*DT/3):  AFLUX ADVECM [GWDRAG,VDIFF dummies] ADVECV PGF  PU,PV,SD scale  isotropuv(ux,vx)
 initial backward (MRCH=-1, DT):         AFLUX ADVECM ... ADVECV PGF  isotropuv(ut,vt)
 360 even step (MRCH=2, DTLF=2*DT): AFLUX ADVECM ADVECV; MUs,MVs,MWs += PU,PV,SD;
        AADVT(DTLF, MMA=MEVEN*AXYP, T, TMOM, qlimit=.false.);  TT=.5(T+Told);  PGF(..TT,TZT);
        isotropuv(u,v); SDRAG(DTLF);  NS-=1
 340 odd step (MRCH=-2): AFLUX ADVECM ADVECV PGF isotropuv(ut,vt)  NS-=1  -> 360 again
 (NIdyn=4 gives: fwd, bwd, even, odd, even = 5 passes; 2 AADVT; 2 SDRAG; 5 isotropuv)
 after loop: MAtoPMB (355 or 366); MUs,MVs,MWs *= DTLF;
 FLTRUV; conserv_amb_ext(u); fltry2(u); fltry2(v); conserv_amb_ext(u); GLOBALSUM; ADD_AM_AS_SOLIDBODY_ROTATION   (379-388)
```
Counts of calls per physics step (derived by tracing the `NS` logic at `ATMDYN.f:300-376`; not measured): AFLUX/ADVECM/ADVECV/PGF 5 each, AADVT 2, SDRAG 2, isotropuv 5, FLTRUV 1, fltry2 2. `AVRX` (FFT truncation, 40 layers) is called inside every `AFLUX` (`ATMDYN.f:557`) and `PGF` (`:1282`), so FFT72 is on the critical path of the core, not only of the filters.

## 4. Inventory of live routines

Columns: range (lines/code); what it computes; inputs / outputs / state.

### 4.1 `ATMDYN.f` (module `ATMDYN`, plus tail routines)

| Routine | Range (lines/code) | Computes | I/O and state |
|---|---|---|---|
| `DYNAM` | 186-390 (205/150) | Leapfrog driver (above). | In: `U,V,T,TMOM(MZ),MA,MASUM`, module `MU,MV,MW,CONV,SPA,DUT,DVT`. Out: `U,V,T,TMOM,MA,MASUM`, `MUs,MVs,MWs` (ATM_COM, kg-flux integrals), `PU,PV,SD`. Local `MEVEN,MODD1,MODD3,UX,VX,UT,VT,TT,TZT`. Prognostic state is only the ATM_COM fields; no state persists in `DYNAM` itself. |
| `AFLUX` | 483-745 (263/209) | Horizontal mass fluxes `MU,MV` (B-grid, 1/4 average with polar interpolation `POLWT`), polar `MU*3`, polar-fix scaling, uphill-flux topography adjustment (loops over L with data-dependent exit), `CONV`, then `MW` (downward flux) as a top-down recursion. | In: `U,V,MA,MASUM,ME,MESUM,NS`, `ZATMO`, geometry. Out: `MU,MV,MW,CONV,SPA` (module `DYNAMICS`). Calls `AVRX`. Stateless. |
| `ADVECM` | 748-844 (97/81) | `MNEW = MOLD + DT1*(CONV + MW differences)/DXYP`, column mass sums, polar replication, column-mass sanity check (`stop_model` if far out of range); calls `MAtoP`. | In: `MOLD,CONV,MW`. Out: `MNEW,MSUM`; side effect: `MAtoP` rewrites `PEDN,PMID,PDSIG,PK,P`. Stateless. |
| `PGF` (non-V2) | 1107-1324 (218/134) | Per-column top-down integration of pressure and layer geopotential (`**KAPA` powers), then pressure-gradient force on V (N-S) and smoothed (`AVRX`) E-W gradient on U; polar fix `acor`; adds `DUT/DVT` to `UT,VT`. | In: `U,V,MAM,S0,SZ(TZ),MAFTER`. Out: `UT,VT`; side effects: `GZ,PHI` (module `ATM_COM`, read by `PGRAD_PBL`), `SPA` used as scratch (`AdM`). `GZ` is therefore state exported to physics. |
| `AVRX` | 1327-1415 (89/72) | Zonal Fourier truncation (via `FFT`/`FFTI`) of a row near the poles; init part sets `NMIN`,`DRAT`,`BYSN` once (`SAVE`). | State: `SAVE` tables built at first call (`init_ATMDYN` calls `AVRX` with no argument, `:107`). |
| `FILTER` | 1418-1624 (207/148; live about 135/100) | SLP filter: diagnose SLP (`SLP` function), 8th-order zonal Shapiro (`SHAP1D`), `isotropslp` near poles, column-mass conservation per row, bounds +-1.18%, rebuild `MA` and `PK`, rescale `T,Q,QCL,QCI,QMOM`, then `getTotalEnergy` before/after and `addEnergyAsDiffuseHeat`. | In: `PEDN(1),ATMSRF%TSAVG,ZATMO,MA,T,Q,QCL,QCI,QMOM,U,V`. Out: modified `MA,T,Q,QCL,QCI,QMOM`, `P*` via `MAtoPMB`. Stateless; needs recorded `TSAVG`. |
| `fltry2` | 1627-1700 (74/65) | 8-pass N-S 3-point filter with pole-crossing conditions on a staggered field. | In/out `U` or `V`. Stateless. |
| `FLTRUV` | 1702-1812 (111/88) | 8th-order E-W Shapiro on U and V (`XUby4toN = (DT/450)/4^8`), then angular-momentum conservation per row (`ANG_UV=1`). | In/out `U,V`; reads `MA`. Stateless. |
| `isotropslp`, `isotropuv`, `Hemisphere`, `at_pole`, `far_from_pole`, `shap1`, `SHAP1D` | 1814-1990 (177/149) | Polar isotropisation (rows with `cos<0.15`): rotate to x-y components, `shap1` with data-dependent sub-iteration count `n=int(fac)+1`, extra FFT truncation at the pole row; `SHAP1D` is the plain 8th-order zonal Shapiro. | `isotropuv` modifies `U,V` in place for rows `J=2,3,JM-1,JM` only (my inference from `cosv`; not checked numerically). Stateless. |
| `SDRAG` | 1993-2141 (149/113; live about 128) | Stratospheric drag above `LS1` with `Wc_JDRAG` factor, wind clamp `WMAX`, optional linear region, angular momentum added back below 150 mb (`ANG_SDRAG=1`). Calls `stop_model` if `T` leaves 100-373 K. | In: `U,V,T,PK,PEDN,MA`; out `U,V`. Reads `LSDRAG,LPSDRAG,CSDRAGL,X_SDRAG,VSDRAGL,WMAX,Wc_JDRAG` (set once by `INIT_SDRAG`). Stateless per call. |
| `ADD_AM_AS_SOLIDBODY_ROTATION`, `CONSERV_AMB_EXT` | 2146-2211 (66/49) | Global angular-momentum bookkeeping for the N-S filter. | `GLOBALSUM` over a J-staggered array. |
| `CONSERV_KE`, `calc_kea_3d` | 2250-2304 (55/39) | Column KE on the B grid regridded to A (`regrid_btoa_ext`/`_3d`), used by energy fix and `DISSIP`. | |
| `recalc_agrid_uv` | 2306-2415 (110/88) | B-grid to A-grid winds `UALIJ,VALIJ`. Called from `ATM_DRV.f:729` and from `ATURB.f:619`. Not part of `DYNAM`, but dynamics-adjacent; D4 took `UALIJ/VALIJ` as recorded input. | In `U,V`; out `UALIJ,VALIJ`. |
| `regrid_btoa_3d`, `regrid_btoa_ext` | 2710-2779 (70/67) | Area-weighted B to A regrid, polar averages. | |
| `QDYNAM` | 3039-3101 (63/47) | Converts `Q,QMOM` to mass units with `MB=MAOLD*KG2MB*AXYP`, `AADVQ0` (flux/cycle preparation), `AADVQ(...,.TRUE.,'q')`, converts back with updated `MMA`. | In/out `Q,QMOM`; reads `MAOLD,MUs,MVs,MWs`. Modifies `MUs,MVs,MWs` in place (`AADVQ0`). |
| `init_ATMDYN` | 42-161 (120/93) | Parameter sync, `NIdyn`, filter flags, `AVRX` init, `FCUVA/B` allocation. Init only. | Not ported except the closed-form constants it derives. |

### 4.2 `MOMEN2ND.f`

| Routine | Range | Computes | Notes |
|---|---|---|---|
| `ADVECV` | 31-498 (468/335) | Momentum advection (horizontal flux form with W-E, S-N, SW-NE, SE-NW corner fluxes, `PU,PV` from `MU,MV`), polar x-y advection (`do_polefix`), vertical advection (`SD`), Coriolis with metric term and polar override, scaling by `VMASS` before/after. | In: `U,V,MMEAN,MBEFOR,UT,VT,MAFTER`, `MU,MV,MW,SPA`. Out `UT,VT`. Uses `DUT,DVT` as scratch (module arrays). Pure stencil: nearest-neighbour shifts in I and J, plus L+-1. |

### 4.3 `QUS_DRV.f` + `QUSDEF.f` (temperature advection, second-moment QUS)

| Routine | Range | Computes | Notes |
|---|---|---|---|
| `AADVT` | 70-196 (127) | Strang-style X(half), Y, Z, X(half) advection of `T` and its 9 moments (`TMOM`) with polar replication, mass-unit conversion. | Called with `qlimit=.false.`. In/out `T,TMOM`, in `MMA`, fluxes from module `MU,MV,MW`. |
| `aadvtx` / `aadvty` / `aadvtz` | 198-316 / 318-471 / 474-573 (119/154/100) | Per-row Courant substep selection (`nstep` up to 20) then `adv1d`; Y uses `advection_1D_custom` over all rows with polar scaling; Z loops columns. | Data-dependent `do while` loops; error exits via `stop_model`. |
| `adv1d` | `QUSDEF.f:43-221` (179/134) | 1-D quadratic-upstream flux and moment update (cyclic in X, bounded in Z). | **Shared with CLOUDS2** (moist convection); porting once pays twice. |
| `advection_1D_custom` | `QUSDEF.f:224-635` (412/307; limiter parts 455-497 and `prather` blocks dead for this path) | Parallel (all-rows) variant used for Y. Internal `contains` helpers `FluxFraction`, `MassFraction`, `tracer_slopes_and_curvatures`, `update_tracer_mass`. | |
| `limitq` | `QUSDEF.f:639-758` | Positivity limiter. | Dead on the `T` path; live in CLOUDS2 and (see 4.4) not in `AADVQ`, which has its own `checkflux`. |

### 4.4 `QUS3D.f` (moisture advection, flux-form QUS with cycles)

| Routine | Range (lines/code) | Computes |
|---|---|---|
| `AADVQ0` | 292-829 (538/460) | Chooses `ncyc` (<=10) so that mass ratio limits `mrat_limh=.25`, `mrat_limy=.20` hold; per-level `ncycxy`, per-row `nstepx` (via `XSTEP`, <=60), extra vertical sub-steps (`ZSTEP`, `mw_extra`, `lminzij`, `lmaxzij`, `nstepz_extra`); divides `MUs,MVs` by cycle counts, subtracts `mw_extra` from `MWs`; tabulates flow-out-both-sides cell lists (`ni_checkfobs_*`). Module-level scratch `TRACER_ADV` is rewritten every call. |
| `AADVQ` | 53-289 (237/200) | Cycle loop; for each level `L=1..LM+1`: Y then X sweeps at level L (`AADVQY`, `AADVQX`), then the Z interface sweep between L-1 and L carrying `mwdn,fdn,fmomdn` (sequential in L); optional extra Z column advection. Upwind-halo packing is a no-op in serial. |
| `aadvqx`, `aadvqy`, `aadvqz` | 978-1234, 1236-1509, 1511-1670 (257, 274, 160) | Explicit flux/moment formulas for one sweep, with the in-line flux limitations (`fe0`, `fe_pass`, ...), polar scaling in Y. |
| `checkflux`, `XSTEP`, `ZSTEP`, `aadvqz_column` | 1672-1688, 835-893, 896-928, 2180-2301 | Moment fix-up for flow out of both sides; Courant-step counters; extra vertical advection for columns with Courant violations. |

Live total in `QUS3D.f`: about 1,700 range lines / 1,460 code.

### 4.5 Utilities and glue

| Routine | Location | Lines | Role |
|---|---|---|---|
| `MAtoP` (45), `CALC_VERT_AMP` (45) | `ATMDYN_COM.F90:147-238` | 92 | Pressure arrays from `MA`; `MFIX/MFRAC` mass partition (initial-state code path). |
| `INIT_SDRAG` live branch | `ATMDYN_COM.F90:363-428` | 66 (about 45 live) | Sets `LSDRAG=LM`... from `PEDNL00` and `P_SDRAG`; init only, port the closed form or record `CSDRAGL`. |
| `DAILY_ATMDYN` | `ATMDYN_COM.F90:438-473` | 36 | Daily global dry-mass fixer: `DELTAM=MDRYA-MDRYANOW` added to `MA` by `MFRAC`; called by `dailyUpdates`. Live at day boundaries only. |
| `MAtoPMB` | `ATM_UTILS.f:241-284` | 44 | Recompute `PEDN,PMID,PDSIG,PK,PEK,P,MASUM` from `MA`; pushes surface pressures into `ATMSRF`/`ASFLX4` (state exported to physics). |
| `PGRAD_PBL` | `ATM_UTILS.f:69-174` | 106 | Surface and layer-1 pressure gradient / density for the PBL, polar averaging. |
| `CALC_TROP` + `tropwmo` | `ATM_UTILS.f:329-551` | 223 | WMO tropopause level and pressure per column (branching scan; I only read `CALC_TROP`, not `tropwmo` line by line). |
| `getTotalEnergy`, `addEnergyAsDiffuseHeat`, `DISSIP`, `addEnergyAsLocalHeat`, `COMPUTE_WSAVE` | `ATM_UTILS.f:579-716` | 138 | Energy bookkeeping, KE-to-heat, vertical velocity export. |
| `conserv_PE`, `CONSERV_SE` | `DIAG.f:1131-1277` | 147 (58 code; most of `CONSERV_SE` is comment) | Total potential/static energy for the energy fix. |
| `SLP` | `shared/Utilities.F90:98-127` | 30 | Sea-level pressure with lapse-rate adjustments. |
| `FFT`, `FFTI` | `FFT72.f:59-250` | 192 | Real FFT (radix-specific, `KM=72`) used by `AVRX` and `isotropuv`. |
| Energy-fix and mass-scaling blocks | `ATM_DRV.f:88-95,136-142,176-205` | about 45 | Inline in `atm_phase1`. |

### 4.6 Totals

About 6,170 range lines / 4,700 code lines live: `ATMDYN.f` about 1,860; `MOMEN2ND.f` 468; `QUS_DRV.f` 504; `QUSDEF.f` about 520 live; `QUS3D.f` about 1,700; `ATMDYN_COM.F90` about 190; `ATM_UTILS.f` about 510; `DIAG.f` 147; `Utilities.F90` 30; `FFT72.f` 192; `ATM_DRV.f` inline 45. About 5,400 range lines are the per-step numerical core and about 800 are coupling/utility glue.

## 5. State, and what must be a recorded input

Prognostic state carried between steps: `U,V` (B grid), `T` (potential temperature), `Q`, `QCL`, `QCI`, `MA` (and so `P`), `TMOM` and `QMOM` (9 moments each, `SOMTQ_COM`, restart variables via `def_rsf_somtq`, `QUS_COM.f:43-77`). Everything else in the dynamics (`MU,MV,MW,CONV,SPA,PU,PV,SD,DUT,DVT`, `MEVEN,MODD1,MODD3`, `TRACER_ADV` scratch, `FCUVA/B`) is workspace rewritten each call, with the exceptions:
- `AVRX` `SAVE` tables and `FFT72` `C,S` tables (static, derivable).
- `MUs,MVs,MWs` leave `DYNAM`, are modified by `AADVQ0`, and are also read by `CLOUDS2_DRV.F90` and `SEAICE.f` (grep hits; I did not verify how). They are exported state.
- `GZ,PHI,P,PEDN,PMID,PK,PEK`, `PTROPO,LTROPO`, `WSAVE`, `DPDX_BY_RHO*`/`DPDY_BY_RHO*`, `KEA`, `ATMSRF%P1/SRFP/AM1` are exports to physics.

Must be taken as recorded input (same pattern as `UALIJ/VALIJ` in D4, `OMEGA` empirical in D42):
1. **Geometry from `GEOM_B.f`** (not read): `IMAXJ, DXYP, DXYS, DXYN, DXYV, DYP, DYV, DXP, DXV, BYDXP, BYDYP, COSV, COSP, COSIV, SINIV, COSIP, SINIP, RAPVS, RAPVN, RAVPS, RAVPN, RAPJ, IDIJ, IDJJ, KMAXJ, FCOR, POLWT, ACOR, ACOR2, FJEQ, BYAXYP, AREAG`, `ZATMO`. No atmosphere geometry dump exists yet (`ffdump_geom` records only the ice-dynamics grid, `ATM_DRV_dynsi.f.patch:8-43`). `geom_jax.py` in the repo is the old Track A module and was not checked for consistency with the real `GEOM_B`.
2. Constants and vertical grid: `GRAV, RGAS, KAPA, SHA, RADIUS, OMEGA, BMOIST`; `MFIX, MFRAC, MFIXs, MTOP, PSF, MDRYA`; `DT, NIdyn`; SDRAG parameters and `CSDRAGL, LSDRAG, LPSDRAG, VSDRAGL, WMAX, Wc_JDRAG`; `MINCOLMASS, MAXCOLMASS`; `COS_LIMIT=0.15` (`ATMDYN_COM.F90:33`). One-time dump.
3. `ATMSRF%TSAVG` for `FILTER` (surface-code output), `KEA` for `DISSIP` (computed at step 7 of section 3, can be reconstructed from dumped `U,V,MA`).
4. `Itime` and the leapfrog counter `NSTEP` (for `MODD5K`/`MODDA` diagnostics only; not needed for numerics).

## 6. Validation boundaries and dump plan

Existing dumps (`ATM_DRV.f.patch:7-92`, `ffdump_reader.py`): whole-state records `pre_condse`, `post_condse`, `post_radia`, `pre_surface`, `post_surface`, `pre_aturb`, `post_aturb` with `MA,U,V,T,Q,QCL,QCI,PK,PMID,PEDN,PDSIG,P`. Observations: (a) `pre_condse` is the exact **output boundary of the whole dynamics block** (DYNAM + QDYNAM + energy fix), already recorded for the 3 test dates; (b) there is **no dump of the step input** (state at the end of the previous step, after `FILTER`), and no `TMOM/QMOM`, `MUs/MVs/MWs`, or any intermediate. New hooks are needed, using the existing `ffdump` pattern (`FFD_START`/`FFD_NSTEP` gating, big-endian stream, `ffdump_reader.py`).

Proposed per-call records (one patch set, one rebuild, 3 dates x 6 steps, as in Stage 1; array size 72x46x40 doubles is about 1 MB, `TMOM/QMOM` about 9 MB):

| Hook | Where | Record (in, out) | Calls/step |
|---|---|---|---|
| H0 geometry/constants | once, after init | arrays and scalars of section 5 items 1-2, `FFT72` tables | 1 per run |
| H1 DYNAM entry/exit | `ATM_DRV.f:105` | in: `U,V,T,TMOM,MA,MASUM,MAOLD`; out: `U,V,T,TMOM,MA,MASUM,MUs,MVs,MWs` | 1 |
| H2 leapfrog sub-calls | inside `DYNAM` | `AFLUX` (in `U,V,MA,MASUM,ME,MESUM,NS`; out `MU,MV,MW,CONV,SPA`), `ADVECM` (out `MNEW,MSUM`), `ADVECV` (in `U,V,MMEAN,MBEFOR,UT,VT`; out `UT,VT`), `PGF` (in `U,V,MAM,S0,SZ,UT,VT`; out `UT,VT,GZ,PHI`), `isotropuv`, `SDRAG` (in/out `U,V`) | 5/5/5/5/5/2 |
| H3 AADVT | `ATMDYN.f:337` | in/out `T,TMOM`, `MMA`, `MU,MV,MW` | 2 |
| H4 velocity filter chain | `ATMDYN.f:379-388` | in `U,V,MA`; out `U,V` and `damsum` | 1 |
| H5 QDYNAM | `ATM_DRV.f:143` | in: `Q,QMOM,MAOLD,MUs,MVs,MWs`; out: `Q,QMOM`, plus `AADVQ0` outputs (`ncyc, ncycxy(:), nstepx, nstepz_extra, lminzij, lmaxzij, mw_extra`, scaled `MUs,MVs,MWs`) | 1 |
| H6 energy fix | `ATM_DRV.f:176-205` | `SEINIT,KEINIT,SEFINAL,KEFINAL`, `dSEpKE`, out `T` | 1 |
| H7 FILTER | `ATM_DRV.f:482` | in: `MA,T,Q,QCL,QCI,QMOM,U,V,TSAVG`; out same (+`P`) | 1 |
| H8 coupling exports | `CALC_TROP`, `PGRAD_PBL`, `COMPUTE_WSAVE`, `DISSIP` | in/out as in section 4.5 | 1 each |
| H9 step input | start of `atm_phase1` | full state incl. `TMOM,QMOM`, `KEA` | 1 |

Each large routine has the same validation shape as the ocean deltas: replay each recorded call from its recorded input and compare all outputs, with non-vacuity checks (e.g. confirm `ncyc`, `nstepx` and `nstep` histograms from the dump to know which branches the window actually exercised) and mutation checks.

Expected exactness: pure add/multiply stencils (`AFLUX`, `ADVECM`, `ADVECV`, the filters, `SDRAG` away from the `sqrt`) should be bitwise if the operation order is kept; **reductions** (`Sum(MA(:,I,J))`, `GLOBALSUM`, `SUM(pgfx)`) need ifort's sequential order (the D45 `np.sum` lesson); `**KAPA` pow in `PGF`, `MAtoPMB`, `CALC_AMPK` can differ by 1 ulp from numpy (the D54 lesson), so tolerance validation (about 1e-12 relative, to be measured) is the realistic target for `PGF`, `FILTER`, `SDRAG`, `CALC_TROP`. `GLOBALSUM` is zonal `sum` over I, then `sum` over J (`MPI_Support/GlobalSum_mod.F90:166-219`; I read only the grep hits, not the whole file).

## 7. Porting order, first target, and first-delta plan

Order (small, individually validatable; each ends with tests, a ledger entry, dump-replay on 3 dates):

1. **H0/H-set instrumentation** (no port): one patch set and one rebuild covering all hooks above.
2. **Velocity filter chain** (first delta, below).
3. **FFT72 `FFT/FFTI` + `AVRX`** (about 280 lines): needed by `AFLUX`/`PGF`/`isotropuv`. Either port the radix code (bitwise target) or use a real FFT and validate to tolerance; decide from one test against dumped `SPA`.
4. **`AFLUX` + `ADVECM` + `MAtoP`** (about 450): first piece with polar rules, topography adjustment, and column recursions. Validate per call (5 per step).
5. **`PGF`** (218): per-column pressure/geopotential integrals then differences.
6. **`ADVECV`** (468): the stencil-heaviest momentum routine, including polar fix; per call.
7. **`isotropuv` + `shap1` + `SDRAG`** (about 280).
8. **`AADVT` family with `adv1d` and `advection_1D_custom`** (about 1,100): largest piece of the leapfrog; reusable by clouds.
9. **`DYNAM` driver glue**: sequencing of the 5 passes, `MUs/MVs/MWs` accumulation, `MAtoPMB`; validate whole-`DYNAM` entry/exit (H1).
10. **`QDYNAM`** (`AADVQ0`, `AADVQ`, `aadvqx/y/z`, `aadvqz_column`, `XSTEP/ZSTEP`, `checkflux`; about 1,750): largest and most branchy; port `AADVQ0` first (validate cycle counts), then one sweep at a time.
11. **`FILTER` (SLP) + `SLP` + energy functions** (about 500).
12. **Coupling glue**: `PGRAD_PBL`, `CALC_TROP`/`tropwmo`, `COMPUTE_WSAVE`, `DISSIP`, `calc_kea_3d`, `regrid_btoa_*`, `recalc_agrid_uv`, `DAILY_ATMDYN`, energy-fix block (about 800).
13. Batched/JAX versions of the hot families, then a chained dynamics step validated against H9 -> `pre_condse`.

**First target: end-of-`DYNAM` velocity filter chain.** Why: smallest self-contained stateless block with a clean boundary (`U,V,MA` in, `U,V` out), no FFT, no data-dependent loops, no reduction-order trap except one `GLOBALSUM`, and it exercises the geometry-dump and reader plumbing every later piece needs.
- Routines: `FLTRUV` (`ATMDYN.f:1702-1812`), `fltry2` x2 (`1627-1700`), `CONSERV_AMB_EXT` x2 (`2181-2211`), `ADD_AM_AS_SOLIDBODY_ROTATION` (`2146-2179`), and the glue at `379-388`. About 330 range lines, about 250 code lines.
- Inputs recorded: `U,V,MA,MASUM`; geometry `DXYN,DXYS,COSV` (and `RADIUS,OMEGA`); parameters `DT=450`, `DT_XUfilter=DT_XVfilter=450`, `ANG_UV=1`. Outputs: `U,V` (and `damsum` for the global reduction).
- Hook: one record pair per physics step at `ATMDYN.f:379` (before) and `:388` (after), plus H0. 6 steps x 3 dates = 18 record pairs, the same shape as D29/D56.
- Port: numpy, loop-order-faithful for the row sums (`ANGM`, `MMUVs`) and the 8-pass stencils; `fltry2` N-S passes as 8 successive whole-field operations with the pole-crossing rows (`yjm1` built from `-yn(im/2+1:im,2,l)` etc.). The JAX shape is immediate: `jnp.roll` along I for the E-W Shapiro, a 3-point stencil along J for `fltry2`.
- Checks: non-trivial (filter changes `U,V` measurably), all output points compared including pole rows `J=2` and `J=JM`, `damsum` compared separately, mutation tests (change `NSHAP`, drop angular-momentum fix, wrong `GLOBALSUM` order), and a dedicated test that `ANG_UV` conservation holds.
- Expected result: bitwise or about 1e-16; if not bitwise, the `GLOBALSUM`/`SUM` order is the first suspect.

## 8. Shapes for batching and JAX

| Piece | Character | Suggested shape |
|---|---|---|
| `FLTRUV`, `shap1d`, `fltry2`, `SDRAG`, `ADVECM`, `COMPUTE_WSAVE`, `DISSIP`, `calc_kea_3d` | pointwise or short fixed stencils, no cross-column dependence | fully vectorized over (L,J,I); 8-pass loops as `fori_loop` or unrolled `roll` |
| `AFLUX` | stencil in I,J, cumulative sum in L, pole special cases, topography adjustment with early exit over L | vectorize over (I,J); `cumsum`/`lax.scan` over L for `MW`; topography adjustment as per-cell masked first-index search; poles as separate rows |
| `PGF` | per-column top-down integral (scan over L, vmap over columns) then differences | `lax.scan` over L, `vmap` over (I,J); gradients by shifts |
| `ADVECV` | pure neighbour stencil (I, J, L+-1) with 4 flux directions, polar rows special | whole-field shifts; polar rows handled as 2 separate small kernels; lowest risk for batching |
| `AVRX`, `isotropuv` | per-row FFT truncation / per-row Shapiro with `n=int(fac)+1` sub-iterations | `jnp.fft.rfft` along I (tolerance) or ported radix code; apply only to the few polar rows; `fori_loop` with max sub-iteration count and per-row masks |
| `AADVT` x/z | independent rows/columns, but per-row Courant `nstep` (<=20) | per-row `nstep` array, `fori_loop` to the max with `where(ns<nstep_row)` masking (a batched max-nstep run without masking would change the result) |
| `AADVT` y (`advection_1D_custom`) | parallel over J by design (all rows computed together) | direct vectorization |
| `AADVQ` | cycle loop (`ncyc` <=10), per-level loop with a vertical carry (`fdn,fmomdn`) | `lax.scan` over L with carry; X/Y sweeps vectorized over (I,J); `AADVQ0` as `while_loop` or host-side preparation recorded as an input for a first version |
| `CALC_TROP`/`tropwmo`, `FILTER` row loops | per-column scans / per-row operations | `vmap` over columns; per-row loops stay as small vmapped scans |

Naturally vectorizable, like the ocean advection family: everything except the nominally sequential parts, which are the leapfrog time sequence (5 passes, state-dependent), the vertical carry in `AADVQ`, the data-dependent cycle loops, and `GLOBALSUM` reductions. Single-GPU is sufficient at 72x46x40 (as the plan says); no sharding design is needed.

## 9. Effort estimates

Basis: the plan's observed rate of about 1 h per 1,000 Fortran lines for stateless pieces (`FULL_FIDELITY_PLAN.md`, "Scoping learned so far"), multiplied by 1.5-3 for stencil/branching/stateful code, plus the fixed per-delta cost seen in the ocean deltas (instrumentation, tests with mutation checks, ledger entry). Hours are focused effort including validation; they are estimates, not measurements.

| # | Piece | Range lines | Hours (range) | One-line basis |
|---|---|---|---|---|
| 0 | Instrumentation patch set, rebuild, dump reader, geometry dump | n/a | 3.5 (3-6) | about 10 hook sites batched in one patch as in Stage 1; one build and run per date |
| 1 | Velocity filter chain (first delta) | 330 | 1 (1-2) | stateless, no FFT; mostly harness and tests |
| 2 | FFT72 + `AVRX` | 280 | 1.5 (1.5-3) | radix-specific code; bitwise versus tolerance decision |
| 3 | `AFLUX` + `ADVECM` + `MAtoP` | 450 | 2.5 (2.5-4.5) | polar rules and topography adjustment; D40/D49 pole-row bugs are the precedent |
| 4 | `PGF` | 218 | 1.5 (1-2.5) | per-column integral; pow tolerance |
| 5 | `ADVECV` | 468 | 2.5 (2.5-4.5) | long but regular stencil; polar fix is the risk |
| 6 | `isotropuv`/`shap1`/helpers + `SDRAG` | 280 | 2 (1.5-3) | data-dependent sub-iterations; small row set |
| 7 | `AADVT` + `adv1d` + `advection_1D_custom` | 1,100 | 5 (4.5-9) | branchy, Courant loops, 9 moments; comparable to D45 per line |
| 8 | `DYNAM` glue, `MUs/MVs/MWs`, `MAtoPMB`, energy fix, `DAILY_ATMDYN` | 300 | 2.5 (2-4) | stateful sequencing of 5 leapfrog passes |
| 9 | `QDYNAM`: `AADVQ0`, `AADVQ`, X/Y/Z, `aadvqz_column`, `XSTEP/ZSTEP`, `checkflux` | 1,750 | 8 (7-14) | largest, most branches and scratch state; limiter branches may be unexercised |
| 10 | `FILTER` + `SLP` + energy functions + `regrid_btoa_ext` | 500 | 2 (2-4) | stateless but many small pieces |
| 11 | Coupling glue (`PGRAD_PBL`, `CALC_TROP`, `tropwmo`, `WSAVE`, `DISSIP`, `recalc_agrid_uv`, `regrid_btoa_3d`) | 800 | 3 (3-5) | simple per-column code; `tropwmo` branching unread |
| | **Subtotal, faithful numeric port with validation** | about 6,200 | **about 35 (26-62)** | |
| 12 | Batched numpy then `jax.jit` for the hot families (pieces 3-9) | | 10 (7-16) | the ocean D78-D86 pattern, 1-2 h per family |
| 13 | Chained dynamics step (H9 to `pre_condse`), 6 steps x 3 dates, tolerance study | | 6 (4-9) | no physics between steps, so one-step chains from recorded inputs |
| | **Total** | | **about 50 (35-80)** | |

Comparison with the README: its atmospheric-dynamics line (12-20 h) was based on "~8,500 lines from the plan table, not yet read in detail". Reading shows about 6,200 live lines, but the larger share is in branchy, stateful code, and the README figure excludes instrumentation, JAX shaping and chained validation. If only the faithful numeric port is compared, 26-62 h here versus 12-20 h there; if the observed 1 h per 1,000 lines held with no multiplier the faithful port would be about 6 h, which I consider unrealistic for the QUS family and the polar code but cannot disprove. Suggested use: take about 35 h for the faithful port and about 50 h for the full block, and re-estimate after the first three deltas give a measured rate.

Overall confidence: **low-moderate (about +-50%)**. Highest confidence in the live/dead classification (every claim above is tied to a `#ifdef`, an object list, or a default parameter). Lowest confidence in piece 9 (`AADVQ0` branch coverage in the test windows), piece 7 (unread bodies of `adv1d`/`advection_1D_custom`), and piece 11 (`tropwmo`).

## 10. Risks

1. Unexercised branches: `AADVQ0` prints a message when `ncyc>2`, `AADVQX` retries up to 20 substeps, `AADVQZ_COLUMN` only runs when a column violates the Courant condition. The 6-step windows may never trigger these; they would then be untested, as the lake TKE branch was in D13. Mitigation: histogram the recorded `ncyc/ncycxy/nstepx/nstepz_extra` first, then add a longer or stronger-wind window or synthetic stress tests.
2. Rounding: `**KAPA`/`**` pow (1-ulp gap, D54), `np.sum` pairwise versus ifort sequential (D45), `GLOBALSUM` order, possible FMA or reassociation in ifort. Plan for tolerance validation on `PGF`/`FILTER`/`CALC_TROP` and bitwise on pure stencils.
3. Leapfrog structure: five passes with different `MRCH`, `DT`, and mass arguments; a single wrong argument pairing (`MODD1` versus `MA`) is silent. Per-call dumps for every pass are the mitigation.
4. Poles: half polar box (`IMAXJ=1` at `J=1,JM`, polar replication, `MU*3`, `acor`, `POLWT`, pole-crossing rows in `fltry2`), `isotropuv` rows, and the `AVRX`/FFT truncation. Ocean deltas D40/D45/D49 each hit a pole-row bug.
5. Atmospheric chaos: leapfrog computational mode and error growth mean only per-call (not multi-step) bitwise-like validation is meaningful until F2 criteria are agreed.
6. Cross-component state: `MUs/MVs/MWs`, `GZ/PHI`, `P*`, `PTROPO/LTROPO`, `KEA`, `ATMSRF` exports are consumed by clouds, PBL, sea ice and diagnostics; the dynamics port must keep those exports identical, so those arrays need to be part of H1/H5/H8 records.
7. Shared kernels: `adv1d`/`advection_1D_custom`/`limitq` serve both dynamics and moist convection; the scope of the clouds port must be coordinated (`fullfidelity/scoping/ATM_CLOUDS_SCOPE.md` exists from another session; I did not read it).

## 11. What I did not read or could not determine

- Not read at all: `GEOM_B.f`, `DOMAIN_DECOMP`/`MPI_Support` beyond the `GLOBALSUM` grep hits, `DIAG.f` routines other than `conserv_PE/CONSERV_SE`, `DIAGCD/DIAG5D/DIAG5F/DIAGA/DIAGB` bodies, `CHECKT`, `UPDTYPE`, `CLOUDS2*`, `PBL_DRV.f`, `ATURB.f`, `IO` routines, `aic_part2`/`PERTURB_TEMPS` (start-up paths), and all non-compiled files (`ATMDYN2.f`, `ATMDYN3.F90`, `MOMEN4TH.f`, `STRATDYN.f`, `QUScubed.f`, `TQUS_*`, `NUDGE.f`, `ATM_DUM.f`; their dead status rests on `decks/P2SAoM40.mk:32-97`).
- Read in full: `DYNAM`, `AFLUX`, `ADVECM`, `PGF` (live), `AVRX`, `FILTER`, `fltry2`, `FLTRUV`, `isotropuv`, `shap1`, `SHAP1D`, `SDRAG`, `ADVECV`, `QDYNAM`, `AADVT` and its `aadvtx/y/z`, `AADVQ`, `AADVQ0`, `XSTEP`, `ZSTEP`, `conserv_*` energy routines, `MAtoPMB`, `PGRAD_PBL`, `COMPUTE_WSAVE`, `getTotalEnergy`, `DISSIP`, `ATMDYN_COM.F90`. Read only in part: `aadvqx/y/z` (first 60-80 lines each), `adv1d`/`advection_1D_custom` (the limiter and `prather` gating, not the flux algebra), `CALC_TROP` (not `tropwmo`), `recalc_agrid_uv` (read), `FFT72.f` (subroutine list only), `GlobalSum_mod.F90` (grep hits).
- Not determined: whether the real run uses a single MPI rank (assumed serial from `build_and_run.md` using `./P2SAoM40` with `OMP_NUM_THREADS=1`; halo-update code is treated as a no-op, the same assumption the ocean deltas made for the domain-decomposition "halo extension" blocks); the actual values of `ncyc`, `ncycxy`, `nstepx`, `nstep` in the test windows; which rows `isotropuv` actually touches (inferred `J=2,3,JM-1,JM` from `cosv<0.15` but not computed); whether `MUs/MVs/MWs` consumers in `CLOUDS2_DRV.F90` and `SEAICE.f` read them before or after `AADVQ0` rescales them; whether `FFT72`'s `FFT/FFTI` can be replaced by a standard FFT within the tolerance the filters need; the exact `FFD`-window timing of `DAILY_ATMDYN` (only fires at day boundaries, probably absent from 6-step windows, like `GLMELT` in D36).
- Effort numbers are judgement applied to line counts and the project's recorded rates, not measurements of this code.
