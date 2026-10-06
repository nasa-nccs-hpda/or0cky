# One 30-minute source step of P2SAoM40: live call order, ports, state, gaps (D127)

Status: plan written 2026-10-06 before `atm_step.py` was coded; a "Result" section is appended after D128/D129 ran.
Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session. Project-local per `projects/imvi/AGENTS.md`.
Sources (pristine ModelE, read-only, line numbers are of the pristine files): `model/MODELE.f` (main loop 293-370), `model/ATM_DRV.f`
(`atm_phase1` 3-294, `atm_phase1_exports` ~296-340, `atm_phase2` 426-525), `model/SURFACE.f` (21-1500), `model/ATURB.f` (`atm_diffus` 3-637),
`model/RAD_DRV.f` (`RADIA` 1940-5501), `model/PBL_DRV.f` (`get_dbl` 1294-1360), `decks/P2SAoM40.R`, plus the earlier plans
`ATM_DYNAMICS_CHAIN_PLAN.md` (D121), `CLOUDS_CONDSE_PLAN.md` (D124), `OCEAN_CHAIN_PLAN.md` (another session, ocean side) and the ledger
`FULL_FIDELITY_DELTAS.md` D1-D126. Everything about the Fortran is read from source or a dump; nothing here is from memory of ModelE.
Review by: next session that closes the step-to-step loop (the next step's start state from this step's end state).

## 0. Facts that fix the call order for this rundeck

- `NIsurf=2` (P2SAoM40.R:252): SURFACE runs two sub-steps of `dtsurf=DTsrc/2=900 s`; `NRAD=5` (:253): the radiation library is entered only when
  `MODRD=MOD(Itime-ItimeI,NRAD)==0` (ATM_DRV.f:273, RAD_DRV.f:2502); on the other four steps `RADIA` jumps to label 900 and only applies the
  stored heating rates to `T` (RAD_DRV.f:5474-5478). With `ITIMEI=16032` the nov26 and dec01 windows start on a radiation step (`33312-16032=17280`,
  `33552-16032=17520`, both multiples of 5), so their steps 0 and 5 call SOCRATES and steps 1-4 do not; the jan01 window (`17520-16032=1488`, 1488 mod 5 = 3)
  has its only radiation step at step 2 (itime 17522). To be confirmed on the new `SRHR/TRHR` dumps (they change only on those steps).
- `USE_PLANET_RAD`, `GISS_RAD_OFF` defined (P2SAoM40.R:19-20); radiation is the SOCRATES library: **third party, never ported; a recorded-input boundary**.
- `ATURB` (P2SAoM40.R:67) is linked, `DRYCNV` is not (ledger D1, `nm` of the executable). `ATM_DIFFUS` is therefore `ATURB.f:3`, and its second call in
  `atm_phase2` (`ATM_DRV.f:458`, `CALL ATM_DIFFUS(2,LM-1,dtsrc)`) returns immediately: `ATURB.f:~116 if (lbase_min.eq.2) return`. Checked on the dumps:
  `ffd_<it>_post_surface == ffd_<it>_pre_aturb == ffd_<it>_post_aturb` bit for bit (U,V,T,Q,QCL,QCI,MA,P,PEDN, nov26 33312/33313). **There is no DRYCNV port to
  validate or to chain**; the F1 text "DRYCNV" is realised by `ATM_DIFFUS(1,1,dtsurf)` inside SURFACE (two calls per step, the `ffa_<it>_c1/c2` dumps).
  The earlier JAX `drycnv.py` ("faithful") is a stand-in for a routine this model does not run (ledger D1).
- `MFILTR=1, NFILTR=1` (ATMDYN_COM.F90:51 defaults, not overridden in the rundeck): `FILTER` runs every step.
- `NIdyn=4` (leapfrog of 5 passes), `DT=450`, `ITIMEI=16032`, `NDAA=13` (as in the D121 plan).

## 1. Live order of one step (MODELE.f main loop, then the two atm phases)

`MODELE.f:316 call atm_phase1` -> `:327 PRECIP_SI` -> `:328 PRECIP_OC` -> `:332 CALL SURFACE` -> `:336 call ocean_driver` -> `:339 call atm_phase2`
-> `:355-362` clock tick, `Itime=Itime+1`; `dailyUpdates` only on day boundary (none of the windows crosses one).

Port/validation key: **E** = bitwise/exact with the stated libm mode; **R** = rounding level (<=1e-12 relative); **T** = threshold-flip caveat;
**REC** = taken from a recorded real dump (named); **NP** = not ported, no state feedback on the atmosphere (inference from names/call patterns, stated);
**GAP** = not ported and it feeds state: listed in section 3.

### 1.1 atm_phase1 (ATM_DRV.f)

| # | Line | Call | Port (this directory) | Level | Reads -> writes (atmosphere state) |
|---|---|---|---|---|---|
| 1 | 75 | `NSTEP=(Itime-ItimeI)*NIdyn` | `dyn_step.step_plan(nstep=)` | E | - |
| 2 | 89-91 | `MAOLD=MA`, `PMIDOLD=PMID` | `dyn_step` stage `save_old` | E | MA,PMID -> MAOLD,PMIDOLD |
| 3 | 93-94 | `CONSERV_SE`, `CONSERV_KE` (initial) | `dyn_glue_ff.conserv_se`, `dyn_filter_ff.conserv_ke` | E (libimf) | diagnostic energies for the fix at 8 |
| 4 | 105 | `CALL DYNAM()` (5 leapfrog passes: AFLUX/ADVECM/ADVECV/PGF/isotropuv, AADVT x2, SDRAG x2, FLTRUV, fltry2, polar-Q DIAGA fill) | `dyn_step.dyn_step` over D90-D100 ports | E with libimf `pow`; numpy `pow`: first inexact quantity `PK=PMID**KAPA` (1 ulp, 916 of 2.38e6 cells), worst scale-relative 3.8e-13 | U,V,T,TMOM,MA,PEDN,PMID,PK,P,MASUM,Q(polar fill),MUs/MVs/MWs |
| 5 | 134,136-143 | `COMPUTE_WSAVE`; `QCL,QCI*=MAOLD/MA`; `QDYNAM` | `dyn_glue_ff.compute_wsave`, `dyn_aadvq_ff.qdynam` | E | Q,QMOM,QCL,QCI |
| 6 | 183-205 | `CONSERV_SE/KE` final, energy fix `T -= dSEpKE/(PK*SHA)` | `dyn_glue_ff.energy_fix` | E (libimf) | T |
| 7 | 212,216 | `CALC_TROP`, `PGRAD_PBL` | `dyn_glue_ff.calc_trop`, `pgrad_pbl` | E | PTROPO,LTROPO; DPDX/DPDY_BY_RHO(_0) (read later by PBL) |
| 8 | 220 | `CALC_ZENITH_ANGLE(modelEclock,1,cosz1)` | **REC** (`COSZ1`, new `ffa_step_*_r`) | - | COSZ1 -> RADIA 5476 |
| 9 | 235 | `calc_kea_3d(kea)` | `dyn_glue_ff.calc_kea_3d` | E | KEA (read by DISSIP at phase 2) |
| 10 | 257-260 | `MELT_SI` (ocean, lake), `seaice_to_atmgrid`, `UPDTYPE` | ice-side ports D12/D31 (isolated) | NP for the atmosphere | change RSI/ice state before CONDSE; RSI enters CONDSE as `ROICE` and SURFACE as `POICE`: taken from the CONDSE entry dump (`ffc_cse_in` RSI) |
| 11 | 264 | `CALL CONDSE` (random draws, MSTCNV, LSCOND per column, momentum add-back) | `clouds_condse_ff.condse_step` (D124-D126) | R with libimf; 26 of 57,060 columns (0.05 %) over 1e-12 from real threshold flips; platform libm 3.6 % of columns | T,Q,QCL,QCI,TMOM,QMOM,U,V,PREC,EPREC,PRECSS,DDM1,DDMS,TDN1,QDN1,cloud arrays,SNOAGE,random seed. Non-dynamic entry fields (EGCM,W2GCM,PBLHT,DCLEV,PBLPTOP,ATMSRF averages,RSI,FEARTH,FLAND,cloud carry state,SNOAGE) are **step-start state taken from `ffc_cse_in`** |
| 12 | 274 | `CALL RADIA` | **REC**: `SRHR`,`TRHR`,`COSZ1` (SOCRATES output stored in RAD_COM; on radiation steps recomputed inside RADIA); the glue `T += (SRHR*COSZ1+TRHR)*DTsrc*bysha*byMA/PK` (RAD_DRV.f:5474-5478) is ported in `atm_step.py` | glue E by construction (checked vs `post_radia` T where dumped) | T; also surface radiation exports (FSF, TRSURF, DIRVIS.., consumed by SURFACE) = **REC** via the PBL/tile/GHY records |
| 13 | 291 | `atm_phase1_exports` | copies (NP) | - | `atmsrf%prec/eprec/cosz1/flong` |

### 1.2 Surface and ocean (MODELE.f:327-336, SURFACE.f)

| # | Line | Call | Port | Level | State |
|---|---|---|---|---|---|
| 14 | MODELE.f:327-328 | `PRECIP_SI`, `PRECIP_OC` | `precip_*` D26, D33 (isolated) | E / R | ice and ocean top-layer state from `PREC,EPREC,PRECSS`: **REC** (their effect enters through the recorded sea-ice/ocean state of the SURFACE tile records); the atmosphere does not read the result except through the tile records |
| 15 | SURFACE.f:290-325 | `PRECIP_SI`, `PRECIP_LI`, `IRRIG_LK`, `PRECIP_LK`, `seaice_to_atmgrid` | D26-D28, D31 (isolated); `IRRIG_LK` GAP (external dataset) | as above | land-ice/lake state: **REC** |
| 16 | SURFACE.f:385-1178 | `DO NS=1,NIsurf`: `loadbl`, `recalc_agrid_uv`, `atm_exports_phasesrf` (= `get_atm_layer1`), `get_dbl`, tile loops (`PBL` per ocean/ice/land-ice/land tile, `SURFACE_LANDICE`, `EARTH`/GHY), `avg_patches_*` composite, **first-layer update (1068-1089)**, `apply_fluxes_to_atm` (dummy with ATURB), then `ATM_DIFFUS(1,1,dtsurf)` (1172) | `chain_two_substeps.substep` + `substep_chain` (get_atm_layer1/get_dbl) + `surface_chain_ff`/`pbl_ff` (D5), `surface_tile_ff` (D6), `landice_tile_ff` (D7), `land_chain`/`ghy_jax` (D9, D22), `tile_aggregate_ff` (D11), `aturb_ff`+`aturb_uv_ff` (D4) | R (PBL/tiles/aggregation/ATURB at 1e-12..1e-13 on real inputs, D18-D19); land patch T (D22: RMS 2e-7 K on the ATURB T after two substeps, worst cell 6e-5 from one runoff-threshold cell on nov26; D9) | T,Q,U,V (via ATURB), EGCM,W2GCM,PBLHT,DCLEV,PBLPTOP, UALIJ,VALIJ; composite TSAVG,QSAVG,USAVG,VSAVG,TGVAVG,QGAVG,USTAR_PBL,LMONIN_PBL |
| 16a | SURFACE.f:1068-1089 | `TMOM(:,I,J,1)*=(1-FTEVAP)`, `QMOM(:,I,J,1)*=(1-FQEVAP)`, `QMOM=0` if `Q+DQ1<qmin` | **new glue in `atm_step.py`** (no earlier port) | E by construction | TMOM,QMOM layer 1 (read by the next step's dynamics and CONDSE) |
| 17 | SURFACE.f:1229-1236 | `UNDERICE`, `GROUND_SI`, `GROUND_LK`, `RIVERF`, `FORM_SI`, `SI_diags` | D32, D10, D13, D12/D14 (isolated + `chain_two_substeps` stages) | see ledger | sea-ice/lake/land-ice state; **not atmosphere state** (the atmosphere sees them only in the next step's tile records) |
| 18 | MODELE.f:336 / OCN_DRV.f | `ocean_driver`: `DYNSI`, `UNDERICE`, `GROUND_SI`, `CALC_APRESS`, `OCEANS`, `FORM_SI`, `ADVSI`, `SI_diags` | see `OCEAN_CHAIN_PLAN.md` (other session) | - | ocean + ice only; no atmosphere prognostic variable. Out of this chain (recorded at the next step start) |

### 1.3 atm_phase2 (ATM_DRV.f:426-525)

| # | Line | Call | Port | Level | State |
|---|---|---|---|---|---|
| 19 | 453-454 | `seaice_to_atmgrid`, `ADVSI_DIAG` | NP | - | ice diagnostics |
| 20 | 458 | `ATM_DIFFUS(2,LM-1,dtsrc)` | none needed: returns at `ATURB.f:~116` | exact (dump identity) | none |
| 21 | 464 | `UPDTYPE` | NP (diagnostic types) | - | - |
| 22 | 466 | `DISSIP` (`calc_kea_3d` again, `DKE=KE-KEA`, `addEnergyAsLocalHeat`) | `dyn_glue_ff.dissip` (D114-D117) | E | T (reads U,V after ATURB and `KEA` from step 9) |
| 23 | 478-482 | `FILTER` (SLP filter chain, row loop, `MAtoPMB`, scaling of T,Q,QCL,QCI,QMOM, energy fix `addEnergyAsDiffuseHeat`) | `dyn_filter_ff.filter_slp` (D101-D102) | E with libimf pow; numpy pow PEDN <=3.4e-13 mb, T <=1.4e-14 | PEDN(1),MA,PMID,PK,P,MASUM,T,Q,QCL,QCI,QMOM. Reads `TSAVG` = the SURFACE composite (step 16) |
| 24 | 504-... | `accum_ma_ia_src`, SUBDD, `DIAG4A` | NP | - | diagnostics |

Not in the live order for this rundeck: GWDRAG/V2 dynamics/DO_CO2_CONDENSATION (dead, D121), DRYCNV, TRACERS_*, the `CUBED_SPHERE`/`SCM` blocks, `DIAG5A/DIAGCA/DIAG7A` (diagnostics).

## 2. State carried across the step and where it comes from

Prognostic atmosphere state at the start of a step (= the end of the previous step; for the first step of a window the real restart): U,V,T,Q,QCL,QCI,
MA,PEDN,PMID,PK,P,MASUM,TMOM,QMOM (`ffd_state_<it>_s1`, and the new `ffa_step_<it>_a`). Hidden state read inside the step and not in the
dynamics dump: `EGCM,W2GCM,PBLHT,DCLEV,PBLPTOP` (ATURB TKE/PBL), `T1/U1/V1_AFTER_ATURB` (PBL `dtdt_gcm`, PBL_DRV.f:328), `USTAR_PBL,LMONIN_PBL`
(`get_dbl`, PBL_DRV.f:1325-1326), `DDM1` (`mdf`, PBL_DRV.f:297), the cloud carry arrays, `SRHR,TRHR` (radiation heating, stored). These are
dumped in the new `ffa_step_<it>_a`/`_e` files; the CONDSE-only ones are already in `ffc_cse_in`.

## 3. Recorded inputs, not-yet-ported pieces, and every gap

### 3.1 Must be recorded (and are named explicitly in `atm_step.py` as `REC_*`)
1. **Radiation** (SOCRATES): `SRHR(0:LM)`, `TRHR(0:LM)` per column (heating rates, stored in RAD_COM and constant over the four non-radiation steps),
   `COSZ1`, and the surface-side radiative exports used by the tile loops (FSF, TRSURF, flong=`TRHR(0)`, DIRVIS/DIFVIS/DIRNIR/DIFNIR): taken from the
   dumps (`ffa_step_*_r`; the tile exports through the `ffp`/`ffs`/`ffl`/`ffg` records). No radiation arithmetic is ported or modified.
2. **Ent vegetation exports** to GHY (canopy conductance, `betadl`, LAI, `dts`): taken from the `ffg` records (D9/D22); GHY itself is ported.
3. **Land forcing that is outside the atmosphere chain**: precipitation into GHY (`prcp, ... `), land `TRUP_in_rad` (inferred from the substep-1 land patch, D22).
4. **Sea-ice, lake, land-ice, ocean surface state at SURFACE entry** (after MELT_SI, PRECIP_SI/LK/LI/OC): from the tile records (`ffs`,`ffl`,`ffp`).
5. **CONDSE non-dynamic entry state** (section 1.1 #11) from `ffc_cse_in`; RANDU stream is ported and checked against the recorded `RNDSS`.
6. **COSZ1** (zenith angle): recorded (`calc_zenith_angle` not ported; analytic, small, not on the critical path).
7. The ocean driver's effect on the atmosphere: none on atmosphere prognostic variables (the next step reads the new ice/ocean state through the tile records).

### 3.2 Not ported, with feedback on the atmosphere state
- **GAP-A: PRECIP_SI/PRECIP_LI/PRECIP_LK/PRECIP_OC fed by our CONDSE precipitation** (ports exist, D26-D28, D33; the chain feeds the SURFACE tile
  records with the real post-PRECIP state, so the chained `PREC` is checked against the recorded `PREC` boundary but not re-fed). Effect on the atmosphere
  end state: second order (CONDSE is exact to 1e-12 except at flipped threshold columns).
- **GAP-B: GHY needs the `ffg` records for steps 1..5 of a window** (re-dumped by the new run).
- **GAP-C: `CALC_ZENITH_ANGLE`, `get_atm_layer1` exports of MAtoPMB into ATMSRF (`SRFP`, `SRFPK`, `AM1`, `P1`)**: the surface-layer scalars are rebuilt from the
  chained state (`substep_chain.layer1_exports`, checked bitwise against the PBL record in D19).
- **GAP-D: DIAGA/DIAGB bodies, `DIAG5A/DIAGCA`, AIJ accumulations, SUBDD, ISCCP**: diagnostics, assumed no feedback (D121 inference; the polar `Q` fill is the one exception found and ported).
- **GAP-E: day-boundary physics (`dailyUpdates`, `daily_ATM`, `DAILY_ATMDYN` is ported)** and the radiation-step-only effects other than `T` (e.g. `RQT`, `SNOAGE` aging inside RADIA, `ALB`): not exercised; the windows do not cross a day.
- **GAP-F: `IRRIG_LK` (SURFACE.f:320), GLMELT**: external dataset / daily; ocean/lake side only.
- **GAP-G: speed.** CONDSE in Python is ~125 s per step (40 ms/column); the other stages are seconds. No JAX/batched CONDSE exists. The "warm CPU timing" in D129 is therefore dominated by CONDSE and is reported per stage.

### 3.3 Not an input but a hazard
libimf vs platform libm: with the Intel runtime the dynamics block and FILTER are bit-for-bit, CONDSE at rounding level; with numpy/platform libm the
dynamics differ at 1e-13 and CONDSE has threshold flips in 3.6 % of columns (D113, D126). Both modes are run and reported separately.

## 4. Chain design (`atm_step.py`)

A plan (list of stages, like `dyn_step.py`) executed on one workspace that carries every Fortran variable by name:
`dyn` (existing `dyn_step.dyn_step`, 17 stages) -> `condse_in` (assemble CONDSE entry from the chained dynamics state plus recorded non-dynamic entry fields)
-> `condse` (`clouds_condse_ff.condse_step`) -> `radia` (T update from recorded SRHR/TRHR/COSZ1, `byMA=1/MA`) -> `surf_in` (build the substep-1 tile/PBL
records from the chained state: psurf, ma1, q1, thv1, `get_atm_layer1` scalars, `get_dbl`, dpdxr/dpdyr (from `PGRAD_PBL`), `mdf=DDM1`, `dtdt_gcm`; every other
column stays recorded) -> `surf_ns1` -> `first_layer` (TMOM/QMOM FTEVAP/FQEVAP, 16a) -> `aturb1` -> `surf_in2` (existing `predict_ns2`) -> `surf_ns2` ->
`first_layer` -> `aturb2` -> `dissip` -> `filter` -> end state. Each stage can be run **chained** (inputs from our previous stage) or **replayed** (inputs from
the real boundary dump), so a failing stage is isolated from the propagation of earlier errors.

Boundary dumps used for the per-stage comparison (all on disk or created by D128):

| Boundary | Dump | Steps available | Fields |
|---|---|---|---|
| step start | `ffd_state_<it>_s1`, new `ffa_step_<it>_a` | 6 per date | full prognostic state (+ hidden state in `_a`) |
| DYNAM exit / dynamics block end / exports | `ffd_state_s2,s3,s4` | 6 per date | full prognostic state, exports |
| CONDSE in/out | `ffc_cse_in/out_<it>` | 6 per date | 68 fields incl. TMOM,QMOM,PREC,.. |
| post-RADIA | `ffd_<it>_post_radia` (2 steps/date), new `ffa_step_<it>_r` | 2 / 6 | state; `SRHR,TRHR,COSZ1` new |
| ATURB in/out (SURFACE substeps) | `ffa_<it>_c1/c2_in/out`; new files for steps 2-6 | 2 / 6 (new) | U,V,T,Q,EGCM,W2GCM,PBLHT,DCLEV,UALIJ,VALIJ,.. |
| PBL/tile/land-ice/GHY/aggregate records | `ffp_`, `ffs_`, `ffl_`, `ffg_`, `fft_` | 2 / 6 (new) | per-tile in/out |
| SURFACE in/out, phase2 entry | `ffd_<it>_pre_surface`, `post_surface`, `pre_aturb`, `post_aturb` | 2 / 6 (new) | state |
| after DISSIP | new `ffa_step_<it>_d` | 6 | T,U,V,KEA,.. |
| step end (after FILTER) | new `ffa_step_<it>_e`; `ffd_state_<it+1>_s1` for steps 0-4 | 6 | full state (= next step start) |

## 5. Comparison protocol and the F1 gate

Per step and field: max abs difference, max abs / field scale (scale = max abs of the real field; for fields that vary over many orders, also
max |d|/(|ref|+1e-300) over cells with |ref| > 1e-12 scale), number of bitwise-equal cells, first differing cell (i,j,l), and the stage at which the
first inexact value appears. Categories stated in advance and not loosened afterwards: **A** bitwise; **B** <=1e-12 of scale (rounding level);
**C** <=1e-6 of scale; **D** worse. The F1 gate (column-physics step from the real restart matches Fortran for one DTsrc, "field-by-field diff of the
state after one step", `FULL_FIDELITY_PLAN.md:48`) is called **met** only if for step 0 of every date every prognostic field of the end state is in A or B
up to a short list of named threshold cells, each quantified; **partly met** if the chain runs end to end with named exceptions in B/C and the first
failing stage identified; **not met** otherwise. The multi-step run (steps 1..5 chained from our own end state) is reported separately as the start of F2.

## 6. Instrumentation added by D128 (new patch only)

`instrumentation/ATM_DRV_atmstep.f.patch` (diff -u against the pristine `ATM_DRV.f`, local hunks): helper `ffas_dump(site)` + `ffas_on` (units 1360, free
after re-grep of model/ and all patch files; 1370 is used by `CLOUDS2_mstcnv.f90.patch` as a different unit, not touched), hooks at: phase1 entry (`a`),
end of phase1 after RADIA (`r`), after DISSIP (`d`), end of phase2 before `accum_ma_ia_src` (`e`). Files `ff_data/<date>/ffa_step_<itime>_<site>.bin`
(6 steps x 4 sites x 3 dates), same record format as the CONDSE dumps (`clouds_condse_io.read_cse`). The older surface-side dumps for steps 2-6 are
regenerated with the unchanged older patches in the same run and copied with `\cp -n` (existing files are never overwritten).

## 7. Result (D128/D129, measured; details in the ledger entries and `atm_step_compare.py`)
- Gate rule implemented as stated in section 5, plus "met with named exception columns" (fields beyond B confined to <=10 columns each and <=1e-9 of scale, columns listed).
- libimf, recorded land patch: dyn bitwise, radiation glue bitwise, CONDSE <=3.4e-14 (one flipped column in dec01 33555), DISSIP+FILTER bitwise, end-of-step state B with 0-4 exception columns of 1e-12..6.5e-12 (W2GCM/EGCM/Q/PBLHT): F1 MET with named exceptions (dec01 step 0 strictly A/B).
- libimf, ported GHY: first failing stage SURFACE (land patch, D9 runoff-threshold cells): nov26 step 0 NOT MET (U 4.6e-6, Q 5.4e-5, EGCM 2.4e-4 of scale, cell (62,34)); dec01 and jan01 step 0 PARTLY MET (<=1.7e-7).
- numpy pow/platform libm: first failing stage CONDSE (3-4.6 % of columns chained, 0.03-0.8 % from the real entry state): NOT MET.
- Corrections to section 1 found while chaining: RADIA also masks CLDSS/CLDMC on radiation steps (RAD_DRV.f:2611-2615, carried to the next CONDSE entry; ported); the B-grid U,V merge after ATURB needs all longitudes valid; isolated CONDSE at step k>0 needs the previous step's LSCOND module-array carry (`ms`).
- Free-running 6 steps (own state carried, surface recorded): rounding level at step 0, then chaotic divergence from CONDSE thresholds (T max 0.07-0.16 K after 6 steps).
- Still recorded/unported: SOCRATES output, Ent, ice/ocean/lake surface state, PRECIP_* effects, USAVG/VSAVG/TGVAVG/QGAVG composites, zenith angle, diagnostics, ocean_driver.
