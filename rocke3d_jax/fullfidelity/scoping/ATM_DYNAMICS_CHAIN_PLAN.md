# Atmospheric dynamics: chained-step plan (D121), and what D122/D123 validated

Status: plan written 2026-10-05 before the chain was coded; the "Result" section at the end was added after D122/D123 ran.
Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session. Project-local per `projects/imvi/AGENTS.md`.
Sources: real Fortran `/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model/` (read-only, pristine line numbers):
`ATM_DRV.f` (`atm_phase1` 3-294, `atm_phase2` 426-525), `ATMDYN.f` (`DYNAM` 186-390), `DIAG.f` (`DIAGA` 98-856), `ATMDYN_COM.F90`, `ATM_UTILS.f`;
rundeck `decks/P2SAoM40.R` (`ndaa=13` at line 265); ports and compare scripts in this directory. Section 1-4 are read from source, not measured.
Review by: next session that starts the step-to-step chain (physics between phase1 and phase2).

## 1. Scope of "the dynamics block"

One 30-min physics step runs `atm_phase1` -> PRECIP_SI/OC -> SURFACE -> ocean -> `atm_phase2`. The dynamics block chained here is the part of
`atm_phase1` between the pre-dynamics energy bookkeeping and `calc_kea_3d` (`ATM_DRV.f:88-235`):

| ATM_DRV.f line | Call / statement | Port (this directory) |
|---|---|---|
| 75 | `NSTEP=(Itime-ItimeI)*NIdyn` | `dyn_step.dynam_plan(nstep=...)`; `ITIMEI=16032` (restart variable `itimei`), `NDAA=13` |
| 89-91 | `MAOLD=MA`, `PMIDOLD=PMID` | `dyn_step` stage `save_old` (glue) |
| 93 | `CONSERV_SE(SEINIT)` | `dyn_glue_ff.conserv_se` |
| 94 | `CONSERV_KE(KEINIT)` | `dyn_filter_ff.conserv_ke` (CONSERV_KE + regrid_btoa_ext) |
| 105 | `CALL DYNAM()` | sections 2-3 |
| 131 | `COMPUTE_DYNAM_AIJ_DIAGNOSTICS` | not ported (AIJ diagnostics only) |
| 134 | `COMPUTE_WSAVE` | `dyn_glue_ff.compute_wsave` |
| 136-142 | `QCL,QCI *= MAOLD/MA` | `dyn_step` stage `qscale` (glue) |
| 143 | `CALL QDYNAM` | `dyn_aadvq_ff.qdynam` |
| 146-174 | `TrDYNAM` etc. | dead (`TRACERS_ON` undefined) |
| 183-184 | `CONSERV_SE(SEFINAL)`, `CONSERV_KE(KEFINAL)` | same functions |
| 185-202 | energy fix, `T -= dSEpKE/(PK*SHA)` | `dyn_glue_ff.energy_fix` (includes the two `GLOBALSUM`s) |
| 212 | `CALC_TROP` | `dyn_glue_ff.calc_trop` (tropwmo) |
| 216 | `PGRAD_PBL` | `dyn_glue_ff.pgrad_pbl` |
| 235 | `calc_kea_3d(kea)` | `dyn_glue_ff.calc_kea_3d` |

Outside the block (not chained): `MELT_SI`, `CONDSE` (264), `RADIA` (274), `SURFACE`/PBL, ocean, and in `atm_phase2` `ATM_DIFFUS(2,LM-1)`
(458), `DISSIP` (466, port `dyn_glue_ff.dissip`) and `FILTER` (482, port `dyn_filter_ff.filter_slp`). The start state of the next step is the
output of `FILTER`; closing that loop needs the physics and is not part of D121-D123.

## 2. DYNAM, exact live call order (ATMDYN.f)

`DTFS=DT*2./3.=300`, `DTLF=2.*DT=900` (234-235); `NS=NSOLD=NIdyn=4` (238); `MUs=MVs=MWs=0` (239). Calls inside `#ifdef NUDGE_ON`, `GWDRAG`/`VDIFF`
(empty entries in `STRAT_DUM.F90`), `DO_CO2_CONDENSATION` and `V2_ATMDYN_TIMESTEPPING` branches are dead for this build (scoping doc section 2).

Label 300 (242-249), re-entered only if `NS>1` after the 8-pass restart test (never at NIdyn=4): `MASUM(I,J)=Sum(MA(:,I,J))` (244), `UX=UT=U`, `VX=VT=V`
(247-248), `TZ=TMOM(MZ,:,:,:)` (249).

| Pass k | Lines | Calls (actual arguments) | Port call |
|---|---|---|---|
| 1 forward, MRCH=0, NS=4 | 257-271 | `AFLUX(NS,U,V,MA,MASUM,MA,MASUM)`; `ADVECM(DTFS,MA,MODD3,MSUMODD)`; `ADVECV(DTFS,U,V,MA,MA,UX,VX,MODD3)`; `PGF(DTFS,U,V,MA,UX,VX,MODD3,T,TZ)`; `PU,PV,SD=MU,MV,MW*kg2mb`; `isotropuv(ux,vx)` | `aflux`, `advecm`, `advecv`, `pgf`, `isotropuv` |
| 2 backward, MRCH=-1, NS=4 | 278-287 | `AFLUX(NS,UX,VX,MODD3,MSUMODD,MA,MASUM)`; `ADVECM(DT,MA,MODD1,MSUMODD)`; `ADVECV(DT,UX,VX,MODD3,MA,UT,VT,MODD1)`; `PGF(DT,UX,VX,MODD3,UT,VT,MODD1,T,TZ)`; scale; `isotropuv(ut,vt)`; `GO TO 360` | same |
| 3 even, MRCH=2, NS=4 (label 360) | 309-358 | `MEVEN=MA` (311); `AFLUX(NS,UT,VT,MODD1,MSUMODD,MEVEN,MASUM)`; `ADVECM(DTLF,MEVEN,MA,MASUM)`; `ADVECV(DTLF,UT,VT,MODD1,MEVEN,U,V,MA)`; scale; accumulate `MUs+=PU, MVs+=PV, MWs(1:LM-1)+=SD` (329-331); `TT=T`, `TZT=TZ` (333-334); `MMA(:,:,L)=MEVEN(L)*AXYP` (336); `AADVT(DTLF,MMA,T,TMOM,.False.,FPEU,FPEV)` (337); `TZ=TMOM(MZ)` (342); `TT=.5*(T+TT)`, `TZT=.5*(TZ+TZT)` (343-344); `PGF(DTLF,UT,VT,MODD1,U,V,MA,TT,TZT)` (349); `isotropuv(u,v)` (352); `SDRAG(DTLF)` (353); if `MODDA<2`: `MAtoPMB, DIAGA, DIAGB, EPFLUX` (354-358); `NS=NS-1` (361) | + `aadvt`, `sdrag` (column port, full-field wrapper) |
| 4 odd, MRCH=-2, NS=3 (label 340) | 291-306 | `AFLUX(NS,U,V,MA,MASUM,MODD1,MSUMODD)`; `ADVECM(DTLF,MODD1,MODD3,MSUMODD)`; `ADVECV(DTLF,U,V,MA,MODD1,UT,VT,MODD3)`; `PGF(DTLF,U,V,MA,UT,VT,MODD3,T,TZ)`; scale; `isotropuv(ut,vt)`; `MODD1=MODD3`; `NS=NS-1` -> 2 | same |
| 5 even, MRCH=2, NS=2 | 309-358 | as pass 3 (second `AADVT`, second `SDRAG`); `NS=NS-1` -> 1 | same |

After the loop: `If (MODDA>=2) Call MAtoPMB` (366); `MUs,MVs,MWs(1:LM-1) *= DTLF` (374-376); `FLTRUV` (379); `conserv_amb_ext(u,am1)`;
`fltry2(u,1d0)`; `fltry2(v,1d0)`; `conserv_amb_ext(u,am2)`; `globalsum`; `add_am_as_solidbody_rotation` (381-387). Tests at 362-364: `NSOLD-NS<8 and NS>1`
-> odd pass; else `NSOLD=NS` and `NS>1` -> label 300. At NIdyn=4: pass sequence 1,2,3,4,5 and exit (NS ends at 1).

Counts per step (AFLUX/ADVECM/ADVECV/PGF 5 each, AADVT 2, SDRAG 2, isotropuv 5, FLTRUV 1, fltry2 2) match the scoping doc and the dumps (`p1..p5`, `c1,c2`).

### 2.1 Data carried between the passes (workspace names = Fortran names)

| After | Produces (consumed by) |
|---|---|
| reinit (300) | `MASUM` (AFLUX of passes 1, 2, 3 as `MASUM`/`MESUM`), `UX,UT,VX,VT` copies (ADVECV `UT,VT` in pass 2, 4), `TZ` (PGF passes 1, 2, 4) |
| AFLUX | module `MU,MV,MW,CONV,SPA` (read by ADVECM `CONV,MW`; ADVECV `MU,MV,MW,SPA`; AADVT `MU,MV,MW`; PU/PV/SD scaling) |
| ADVECM | `MNEW`/`MSUM` into the named variables (`MODD3,MSUMODD` pass 1; `MODD1,MSUMODD` pass 2; `MA,MASUM` passes 3, 5; `MODD3,MSUMODD` pass 4), and as a **side effect `MAtoP(MNEW,MSUM)` rewrites module `PEDN(1:LM),PMID,PDSIG,PK,P`** (these, not MAtoPMB, are what the same pass's SDRAG reads; `PEDN(LM+1)` keeps its MAtoPMB value) |
| ADVECV | `UT,VT` named variables (`UX,VX` pass 1; `UT,VT` passes 2, 4; `U,V` passes 3, 5); module `DUT,DVT` left zero (ADVECV zeroes rows 2..JM at entry, MOMEN2ND.f:111-112, and clears at exit) |
| PGF | `UT,VT` += DUT/VMASS; module `GZ`, `PHI` (exported to physics), `SPA` overwritten with AdM; `DUT,DVT` hold the PGF increments |
| PU,PV,SD scaling | only consumed by the accumulation in even passes; in passes 1, 2, 4 it is a dead store |
| AADVT (even) | `T`, `TMOM`, `MMA` (module T/TMOM are advected in place), `FPEU,FPEV` (diagnostic) |
| MODD1=MODD3 (odd) | `MODD1` seen by pass 5 AFLUX/ADVECV/PGF |

Arrays and shapes (numpy 0-based as in the ports): U,V,UX,VX,UT,VT,T,TT,TZ,TZT,PU,PV,MU,MV,CONV,SPA,GZ,PHI (IM,JM,LM); MW,SD (IM,JM,LM-1);
MA,MEVEN,MODD1,MODD3 (LM,IM,JM); MASUM,MSUMODD (IM,JM); TMOM,QMOM (9,IM,JM,LM); MUs,MVs,MWs (IM,JM,LM, layer LM never touched);
PEDN (LM+1,IM,JM); PMID,PK,PDSIG (LM,IM,JM).

### 2.2 The 5 leapfrog passes in words
Pass 1 (forward, `DTFS=2DT/3`) and 2 (backward, `DT`) start the leapfrog from the step-start state, producing the half-step state `MODD3` and the
backward-step state `UT,VT,MODD1`. Pass 3 (even, `2DT`) advances the real state `MA,U,V` from `MEVEN` (the step-start mass) with the fluxes of the
half state and accumulates the mass fluxes `MUs,MVs,MWs` and advects temperature (`AADVT`); pass 4 (odd) advances the intermediate state `MODD3,UT,VT`; pass 5
(even) advances `MA,U,V` again from `MEVEN` (= the pass-3 result) and advects `T` again. Mass fluxes are accumulated only in the two even passes, which is why
`PU,PV,SD` scaling in passes 1, 2, 4 is dead; the temperature advection happens in the even passes only (twice per step, each with `DTLF`).

## 3. Arrays and state taken and returned by each port (see the docstrings for exact layouts)

| Port | Takes | Returns |
|---|---|---|
| `aflux(ns,u,v,ma,masum,me,mesum,g,tab)` | winds, mass, column sums (current and "ME") | `mu,mv,mw,conv,spa` (+`spa0`) |
| `advecm(dt,mold,conv,mw,g,imf_pow)` | old mass, `CONV`, `MW` | `mnew,msum,pedn,pmid,pdsig,pk,p` (MAtoP included) |
| `advecv(dt,u,v,mmean,mbefor,ut,vt,mafter,pu,pv,sd,spa,g)` | winds, three mass fields, accumulating `ut,vt`, module fluxes | `(ut,vt)` |
| `pgf(dt,mam,ut,vt,mafter,s0,sz,dut,dvt,g,tab,imf_pow)` | mass before/after, accumulating `ut,vt`, `T`/`TZ` (or `TT`/`TZT`) | `ut,vt,dut,dvt,gz,phi,adm(=SPA),...` |
| `isotropuv(u,v,geo,C,S)` | winds, FFT tables | `(u,v)` |
| `aadvt(dt,mm,rm=T,rmom=TMOM,mu,mv,mw,False)` | air mass `MEVEN*AXYP`, T, TMOM, fluxes | `rm,rmom,mm,fqu,fqv` |
| `sdrag_columns` (+ `dyn_step.sdrag_field`) | per column `u,v,t,pk,PEDN(L+1)`, four `MA` columns (Ip1 cyclic), `J` | `(u,v)` |
| `matopmb(ma,g)` | `MA` | `masum,pedn(LM+1),pmid,pk,pdsig,p` |
| `filter_chain(u,v,ma,masum,geo,omega)` | winds, final `MA`, `MASUM` | `(u,v,damsum)` |
| `compute_wsave(mws,t,pk,pedn,g)` | accumulated `MWs`, `T`, `PK`, `PEDN` | `WSAVE` (IM,JM,LM-1) |
| `qdynam(q,qmom,maold,mus,mvs,mws,axyp,imaxj,kg2mb,byim,byim)` | moisture state, `MAOLD`, accumulated fluxes | `q,qmom` and `q0` (AADVQ0 rewrites `MUs,MVs,MWs`) |
| `conserv_se`, `conserv_ke`, `energy_fix` | `MA,MASUM,PK,T,Q,QCI,U,V` | `SEINIT,KEINIT,SEFINAL,KEFINAL`; `T` after the uniform `dSEpKE` fix |
| `calc_trop`, `pgrad_pbl`, `calc_kea_3d` | `T,PK,PMID,PEDN,PHI(:,:,1)`, `U,V` | `PTROPO,LTROPO`; `DPDX/DPDY_BY_RHO(_0)`; `KEA` |

Constants/geometry come from the recorded one-time dumps only: `ffd_aflux_geom.bin` (g), `ffd_avrx_consts.bin` (FFT `C,S`), `ffd_sdrag_consts.bin`,
`ffd_glue_consts.bin`, `ffd_qdyn_geom.bin`; `ITIMEI` from the restart (`itimei`).

## 4. Gaps: code in DYNAM / atm_phase1 not covered by an existing port, and how D122 treats it

| Gap | Where | Treatment in `dyn_step.py` |
|---|---|---|
| `MUs,MVs,MWs=0`; accumulation `+PU,+PV,+SD(1:LM-1)`; final `*DTLF` | ATMDYN.f:239, 329-331, 374-376 | implemented (stages `flux_zero`, `accum`, `flux_scale`) |
| `MASUM=Sum(MA(:,I,J))` at label 300 | 244 | implemented with the strictly sequential `seqsum` (ifort `Sum`, `-fp-model strict`) |
| `UX,UT,VX,VT=U,V`, `TZ=TMOM(MZ)` | 247-249 | implemented (`reinit`) |
| `MEVEN=MA`, `MODD1=MODD3`, `TT=T`, `TZT=TZ`, `TZ=TMOM(MZ)`, `TT,TZT` averaging | 305, 311, 333-334, 342-344 | implemented (`copy`, `tz`, `avg`) |
| `PU,PV,SD=MU,MV,MW*kg2mb` | 268-270 etc. | implemented (`pscale`) |
| `MMA(:,:,L)=MEVEN(L,:,:)*AXYP` | 336 | implemented (`mma`) |
| Leapfrog control flow (labels 300/340/360, `NS`, 8-pass restart test) | 242-366 | simulated in `dynam_plan` (generic in NIdyn; validated at 4 only) |
| SDRAG over the whole field | ATMDYN.f:353 | `sdrag_field`: column port applied to every B-grid column J=2..JM with Ip1-cyclic neighbour `MA` columns |
| ADVECM's `MAtoP` side effect on module `PEDN,PMID,PDSIG,PK,P` | 748-844 | `_set_matop`, used by the same pass's SDRAG |
| Final `MAtoPMB` and the mid-step `MAtoPMB` when `MODDA<2` | 355, 366 | final one implemented; the mid-step one is inserted on firing steps (result-identical: overwritten by the next ADVECM `MAtoP`/the final MAtoPMB; its `MASUM` equals ADVECM's `MSUM` bitwise) |
| **`DIAGA` polar `Q` fill** (`Q(2:IM,pole,:)=Q(1,pole,:)`), called from DYNAM when `MODDA<2` | DIAG.f:221-233 via ATMDYN.f:356 | implemented (`diaga`); found by D122: the only change of `Q` between step start and DYNAM exit in the 18 dumped steps (dec01 33555, polar `Q` was longitude-dependent at step start). `MODDA=Mod(NSTEP+4-NS+NDAA*NIDYN,NDAA*NIDYN+2)`, `NDAA=13` (rundeck), `ITIMEI=16032` |
| `MAOLD,PMIDOLD` saves; `QCL,QCI*=MAOLD/MA` | ATM_DRV.f:89-91, 136-142 | implemented (`save_old`, `qscale`) |
| Diagnostics: `DIAGA0`, `DIAGB` (and the non-Q parts of `DIAGA`), `EPFLUX` (empty), `COMPUTE_MASS_FLUX_DIAGS`, `COMPUTE_DYNAM_AIJ_DIAGNOSTICS`, `DIAG5A/DIAGCA/DIAGCD`, `MODD5K` | 309, 327, 351, 356-358, 131, 1054-1055 | not ported; "no state feedback" except the `DIAGA` polar `Q` fill is an inference from names, call patterns and the dump comparison (DIAGA/DIAGB bodies were not read in full; `DIAGA`'s `recalc_agrid_uv` writes `UALIJ,VALIJ`, which every physics consumer recomputes itself, call sites in `dyn_glue_io.py`) |
| `UALIJ,VALIJ`, `PEK`, `byMA`, `ATMSRF%AM1,P1,SRFP,SRFPK`, `ASFLX4%SRFP` side effects of `MAtoPMB` | ATM_UTILS.f:241-284 | not ported (not in the dumped state); needed when the physics is chained |
| `AVRX` tables, FFT tables, SDRAG parameters, geometry | init | taken from the recorded one-time dumps |

Module-state subtleties that the chain must (and does) honour: `MV` row `J=1` is not written by AFLUX (stale in the real run; the chain leaves 0, no consumer
reads it, checked by the per-stage input comparisons which exclude that row); `DUT,DVT` are zero at PGF entry; `SPA` is AFLUX's value at ADVECV time and
PGF's `AdM` afterwards; `PEDN(LM+1)` is constant `MTOP*KG2MB`.

## 5. State needed at step start, and the new dumps

Existing dumps held the pieces separately (U,V,MA,MASUM at AFLUX p1 in; T,TMOM at AADVT c1 in; Q,QMOM at QDYNAM in; QCI,PK at efix kind 1), but **QCL at
step start existed nowhere** and `TMOM/QMOM/MUs/MVs/MWs` were never dumped at the end of the block. D122 added `instrumentation/ATM_DRV_dynG.f.patch`
(unit 1300; `ffd_state_<itime>_s1..s4.bin`, 6 steps x 3 dates, new files only): s1 step start, s2 DYNAM exit, s3 end of the dynamics block (after QDYNAM and
energy fix), s4 physics exports (PTROPO, LTROPO, WSAVE, KEA, DPDX/DPDY_BY_RHO(_0), PHI). `ffd_<itime>_pre_condse.bin` (first two steps per date only) is an
independent second reference for s3.

## 6. Result (D122/D123, measured)

See `scratchpad/d121_entry.md` summary and `fullfidelity/dyn_step_compare.py`. With the Intel libimf `pow` bridge the chained step starting from the real
step-start state reproduces s2, s3, s4 and pre_condse **bit for bit** for all 18 steps (every field, every element), and every stage input and output of the 264
(stage, field) per-call series. Without libimf (numpy `pow`), the first non-exact quantity is `PK=PMID**KAPA` in pass 1 ADVECM (1 ulp in <0.1 % of cells),
which then propagates through the leapfrog to the worst scale-relative differences: u 3.5e-13, v 3.8e-13, T 1.1e-15, MA 9.6e-15, TMOM 5.1e-14, MUs..MWs <1e-13,
GZ 3.4e-14, exports <=5.6e-14 except the near-cancelling PGRAD_PBL terms (2.7e-11); the integer `LTROPO` is exact.
Remaining gaps: the loop through physics and `atm_phase2` (`ATM_DIFFUS`, `DISSIP`, `FILTER`) to the next step's start; DIAGA/DIAGB bodies other than the polar `Q` fill;
`MAtoPMB` exports to `ATMSRF`; NIdyn other than 4 and the 8-pass restart path (`GO TO 300`) are generated but not exercised.
