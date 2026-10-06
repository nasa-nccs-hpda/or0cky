# CONDSE column-driver glue: live call order, interfaces, state and gaps (D124)

Status: DRAFT, written 2026-10-05 from a full read of `model/CLOUDS2_DRV.F90:3-2854` (pristine ModelE tree) and the
preprocessed live text (`cpp -traditional-cpp -DCOMPILER_Intel8 -Iinclude`, scratch only). Owner: not assigned.
Review date: after the D125 dumps are validated (D126). All line numbers are in the pristine `CLOUDS2_DRV.F90`.
Scope: D-C8 (CONDSE glue) of `ATM_CLOUDS_SCOPE.md`. MSTCNV (D110-D113) and LSCOND (D107-D109) are imported, not re-ported.
Rundeck flag state: only the options of `ATM_CLOUDS_SCOPE.md` section 0 are defined (no TRACERS_*, CLD_AER_CDNC, SCM, CUBED_SPHERE,
COSP_SIM, CFMIP3_SUBDD; `USE_PLANET_RAD`, `CACHED_SUBDD` and `isccp_diags=1` are live).

## 1. Live call order for one source step (30 min, DTsrc = 1800 s)

Caller: `ATM_DRV.f:264` `CALL CONDSE` (after SURFACE/MELT_SI/UPDTYPE, before RADIA at :271). Serial run: J_0=1, J_1=JM,
I_0=1, I_1=IM, `HAVE_SOUTH_POLE = HAVE_NORTH_POLE = .true.`

| # | Lines | Action | Class |
|---|---|---|---|
| 1 | 525-540 | random draws: `BURN_RANDOM(nij_before_j0(J_0)*LMCLD*3)`; for J, for I=1..IMAXJ(J): `RNDSS(NR,L,I,J)=RANDU` with `L=LMCLD,1,-1` outer, `NR=1..3` inner; `BURN_RANDOM` of the rest. Serial: `nij_before_j0=SUM(IMAXJ(1:J0-1))`, `nij_after_i1=0`, `nij_after_j1=SUM(IMAXJ(J1+1:JM))` (ATM_UTILS.f:789-830). RANDU = 32-bit LCG `ix=ix*69069+1`, mask `ffffff00`, scale 2**-32 REAL(4) | physics (stochastic input of LSCOND) |
| 2 | 543 | `RFINAL(seed)` (seed after the draws, restored at 2604 by `RINIT(seed)` so ISCCP draws do not shift the stream) | state |
| 3 | 544-550 | zero SAVWCU etc. (unused arrays) | none |
| 4 | 565 | `recalc_agrid_uv` (UALIJ/VALIJ from U,V; already ported, `dyn_glue_ff.recalc_agrid_uv`) | glue |
| 5 | 570-572 | `kmax_nonpolar=minval(kmaxj)=4`; `replicate_uv_to_agrid(UKM,VKM,4,UKMSP,VKMSP,UKMNP,VKMNP)` (ATMDYN.f:2554-2604): for J=2..JM-1, I=1..IM: UKM(1..4,L,I,J)=u(i-1,j),u(i,j),u(i-1,j+1),u(i,j+1) (i-1 wraps to IM), same for V; poles: UKMSP=u(:,2,:), UKNP=u(:,JM,:) | glue |
| 6 | 576-580 | `TLS=T; QLS=Q; TMC=T; QMC=Q; FSS=1.` (whole atmosphere, entry values) | state |
| 7 | 663-679 | loops `J=1..JM`, `I=1..IMAXJ(J)` (IMAXJ=IM except 1 at the poles; 3170 columns); `KMAX=KMAXJ(J)` (=4, poles 72) | |
| 8 | 680-700 | column scalars: DXYPIJ=AXYP, PEARTH=FEARTH, PLAND=FLAND, PWATER=1-PLAND, ROICE=RSI*PWATER, TS,QS,US,VS,TGV,QG = `atmsrf%TSAVG,QSAVG,USAVG,VSAVG,TGVAVG,QGAVG`, TSV=TS*(1+QS*DELTX), `DCL=int(DCLEV+.5)`, ZPBL=PBLHT, PPBL=PBLPTOP | input |
| 9 | 725-737 | `RA(K)=RAVJ(K,J)` K=1..KMAX; PL=PMID, PLE=PEDN, PLK=PK, AIRM=PDSIG, BYAM=1/AIRM, `WTURB=sqrt(.6666667*EGCM)` (REAL(4) literal) | input |
| 10 | 745-749, 773-774 | SVLHXL=SVLHX, TTOLDL=TTOLD, CLDSAVL=CLDSAV, CLDSV1=CLDSAV1, RH=RHSAV, FSSL=FSS(=1), `DPDT=(PMID-PMIDOLD)*BYDTsrc` | state in |
| 11 | 775-812 | per layer: `SM=T*AIRM; SMOM=T3MOM*AIRM; SMOMMC=SMOMLS=SMOM; TL=T*PLK; QM=Q*AIRM; QMOM=Q3MOM*AIRM; QMOMMC=QMOMLS=QMOM; QCIL,QCLL=QCI,QCL; QL=Q; SDL=MWs/DTsrc*BYAXYP; TVL=TL*(1+DELTX*QL); W2L=W2GCM`; `ETAL(L+1)=.5*ENTCON*(GZ(L+2)-GZ(L))*1.d-3*BYGRAV` and `GZL(L+1)=ETAL(L+1)/ENTCON` for L<=LM-2 (ENTCON=.2d0); `ETAL(LM)=ETAL(LM-1)`, `ETAL(1)=0`, `GZL(LM)=GZL(LM-1)`, `GZL(1)=0` | input |
| 12 | 825-842 | `U_0,V_0(1:KMAX,:)` = UKMSP/VKMSP (J=1), UKMNP/VKMNP (J=JM), else UKM/VKM; `UM=U_0*AIRM`, `UM1=UM` (same for V) | input |
| 13 | 845-859 | `PRCP=0; ENRGP=0; TPRCP=T(I,J,1)*PLK(1)-TF` (PRE-MSTCNV temperature); `AIRX,DDM1,DDMS,DDML,TDN1,QDN1 = 0` | |
| 14 | 896 | `CALL MSTCNV(IERR,LERR,i,j)` (ported, D110-D113) | physics |
| 15 | 905-908 | ierr>0 message; ierr==2 increments ICKERR, stop at 2561-2564 | error exit |
| 16 | 936-1181 | `if (LMCMIN.gt.0)` (convecting column only): `PRCP=PRCPMC*100.*BYGRAV` (1119); `if TPRCP>0: EPRCP=0, ENRGP=ENRGP+EPRCP else ENRGP=ENRGP+EPRCP-PRCP*LHM` (1125-1133); for L=1..LMCMAX: `T(I,J,L)=(1.-FSSL)*SM*BYAM+FSSL*TLS(I,J,L)`, `Q` likewise with QLS, `TMC=SM*BYAM`, `QMC=QM*BYAM`, `SMOMMC=SMOM`, `QMOMMC=QMOM`, `UM1,VM1=UM,VM` (1145-1156); `CSIZMC(1:LMCMAX)=CSIZEL`; `FSS(:,I,J)=FSSL`; `AIRX=AIRXL*AXYP`; DDML = first L<=DCL with DDMFLX(L)>0 else DCL (DO-loop variable semantics); if DDML>0: `TDN1=TDNL(DDML)`, `QDN1=QDNL(DDML)`, `DDMS=-100.*DDMFLX(DDML)/(GRAV*DTsrc)`; `DDM1=DDMFLX(1)*RGAS*TSV/(GRAV*PEDN(1,I,J)*DTsrc)` (1163-1180). The AIJ/AJL/ADIURN calls inside the block (937-1065) are diagnostics | glue |
| 17 | 1254-1255 | `LMC(1,I,J)=LMCMIN; LMC(2,I,J)=LMCMAX+1` (every column) | state out |
| 18 | 1259-1284 | LSCOND set-up: `TL=TLS*PLK; TH=TLS; QL=QLS; SMOM=SMOMLS; QMOM=QMOMLS`; `QCLX=QCLL, QCIX=QCIL`, then for L<=LMCMAX: `QCLX+=SVWMXL` if `SVLATL==LHE`, `QCIX+=SVWMXL` if `==LHS`; `AQ=(QL-QTOLD)*BYDTsrc`; `RNDSSL(:,1:LMCLD)=RNDSS`; `FSSL=FSS(:,I,J)`; `UM,VM=U_0*AIRM` (reset: MSTCNV's UM changes live in UM1) | glue |
| 19 | 1288-1315 | stratocumulus (Philander) block: gated `ISC.eq.1 .and. FOCEAN>.5`; `ISC=0` by default and is not set in the rundeck (CLOUDS2.F90:115, `P2SAoM40.R` has no ISC) -> not executed (the dump records ISC to confirm) | dead (verify) |
| 20 | 1319-1343 | `if (DCL.le.1)`: RIS, RI1, RI2 (module variables) from TH, TS, UA, VA, GZ, PEK. They are read only inside `do_blU00==1` (CLOUDS2.F90:3583-3585), and `do_blU00=0`: computed but not used by any state. Not ported (listed as gap; no effect on state) | live-unused |
| 21 | 1352 | `CALL LSCOND(IERR,WMERR,LERR,i,j)` (ported, D107-D109) | physics |
| 22 | 1360 | `if IERR/=0` write to unit 99 | none |
| 23 | 1442-1464 | `PRCP=PRCP+PRCPSS*100.*BYGRAV`; `if LHP(1)/=LHS: ENRGP=ENRGP+EPRCP(=0)` else `ENRGP=ENRGP+EPRCP-PRCPSS*100.*BYGRAV*LHM` | glue |
| 24 | 1485-1489 | `if ENRGP<0`: `SNOAGE(ITYPE,I,J)*=exp(-PRCP)`, ITYPE=1..3 (libm exp) | state |
| 25 | 1932-1945 | store: TAUMC,CLDMC,SVLAT (from MSTCNV), TAUSS,CLDSS,CLDSAV,CLDSAV1,SVLHX,CSIZSS (=CSIZEL after LSCOND), QLss,QIss,QLmc,QImc | state out |
| 26 | 1949-2080 | radiation hand-off: `w_cloud=CLDMCL+CLDSSL-CLDMCL*CLDSSL`; for L: stratiform (CLDSSL>0: water if SVLHXL==LHE else ice if ==LHS else stop 255): `frac_st_*=CLDSSL/(CLDMCL+CLDSSL+TINY)`, mix ratio QCLX/QCIX, `dim_char_st_*=CSIZSS*1e-6`, `frac_area_st=CLDSAL`; else all zero. Convective (CLDMCL>0: phase from SVLATL; `frac_cnv_*=CLDMCL/(..)`, `mix_ratio_cnv=CNVMMRL`, `dim_char_cnv=CSIZMC*1e-6`, `frac_area_cnv=CLDMCL`; if CNVMMRL==0: all cnv zero, `w_cloud=CLDSSL`, renormalise frac_st_water/ice (second statement uses the updated first); else-branch CLDMCL<=0: six cnv fields zero BUT `frac_area_cnv` is NOT reset (carried from the previous step: hidden state, 2068-2078) | rad hand-off |
| 27 | 2083-2088 | `TAUSSIP=TAUSSLIP; CSIZSSIP=CSIZELIP` (use_vmp); `RHSAV=RH` | state out |
| 28 | 2115-2125 | `TTOLD=TH; QTOLD=QL; PREC=PRCP; EPREC=ENRGP; PRECSS=PRCPSS*100.*BYGRAV; P_acc+=PRCP; PM_acc+=PRCP-PRECSS` | outputs |
| 29 | 2152-2185 | per layer final merge: `T=TH*FSSL+TMC*(1-FSSL)`, `Q` likewise with QMC, `SMOM=SMOM*FSSL+SMOMMC*(1-FSSL)` (+QMOM), `T3MOM=SMOM*BYAM`, `Q3MOM=QMOM*BYAM`, `QCI=QCIX`, `QCL=QCLX`; momentum increment `UKM(K,L,I,J)=(UM*FSSL+UM1*(1-FSSL))*BYAM-UKM(K,L,I,J)` K=1..KMAX (poles: UKMSP/NP, K=1..IM) | state out |
| 30 | 2550-2554 | end of the J loop | |
| 31 | 2589-2594 | `avg_replicated_duv_to_vgrid(UKM,VKM,4,...)` (ATMDYN.f:2606-2708; adds the tendencies to U,V on the B grid: pole rows scaled by .5; J=2..JM, wraps in I) then `recalc_agrid_uv` | glue |
| 32 | 2604 | `RINIT(seed)` | state |

Diagnostics inside CONDSE (not state; not ported): all AIJ/AIJL/AJ/AREG/ADIURN/`inc_ajl` updates (937-1065, 1421-1537,
1929, 2132-2149, 2210-2262, 2598), SUBDD (2613-2764), `ISCCP_CLOUD_TYPES` (1540-1628, diagnostic; it only raises
`stop_model` on JERR, 2568-2571; its random draws are bracketed by RFINAL/RINIT).

## 2. Inputs

From the atmosphere state: T, Q, QCL, QCI (IM,JM,LM); U, V (IM,JM,LM, B grid, with halo); TMOM, QMOM (nmom=9, IM,JM,LM; `SOMTQ_COM`);
PK, PMID, PEDN, PDSIG, PEK, PMIDOLD, EGCM, W2GCM (L,I,J); GZ, MWs (I,J,L). From PBL/ATURB: DCLEV, PBLHT, PBLPTOP (I,J), EGCM, W2GCM.
From the surface/boundary layer: `atmsrf%TSAVG,QSAVG,USAVG,VSAVG,TGVAVG,QGAVG`, `si_atm%RSI`, FEARTH, FLAND (FOCEAN, FLICE, FLAKE only in
diagnostics/ISC). Geometry: AXYP, BYAXYP, IMAXJ, KMAXJ, RAVJ(K,J). Cloud state carried from the previous step: TTOLD, QTOLD, SVLHX,
RHSAV, CLDSAV, CLDSAV1 (restart: TTOLD,QTOLD,SVLHX,RHSAV,CLDSAV,AIRX,LMC per `CLOUDS_COM.F90:481-532`; CLDSAV1 is NOT in the restart
list I read -- gap, the dump records it), plus the hand-off array `frac_area_cnv` and CSIZMC (partially rewritten: layers above LMCMAX keep
last step's values), SNOAGE, P_acc, PM_acc. Random seed IX (restart parameter IRAND).
Constants: from `ffc_mc_consts.txt`/`ffc_ls_consts.txt` and one LSCOND boundary record (BYBR, RIMAX, RWMAX, RWCLDOX, RCLDLX, RCLDIX, WMUI,
CMX, U00A, U00B, RTEMP, USE_VMP, do_blU00, WCONST, SCDNCW, SCDNCI, DTsrc), XMASS/LMCM from the MSTCNV dump; PDSIGL00(L)
(reference column, ATM_COM) from the LSCOND boundary record; LMCLD=29.

## 3. Outputs

To the atmosphere: T, Q, QCL, QCI, TMOM, QMOM (merged MC/LS), U, V (via UKM tendencies), UALIJ, VALIJ. To the surface (precipitation, consumed by the
PRECIP_* routines ported in D26-D28, D33): PREC, EPREC, PRECSS (and P_acc/PM_acc accumulators, SNOAGE in RAD_COM). Cloud state for the next step and
for other consumers: TTOLD, QTOLD, SVLHX, SVLAT, RHSAV, CLDSAV, CLDSAV1, TAUSS, TAUMC, CLDSS, CLDMC, CSIZSS, CSIZMC, FSS, TAUSSIP, CSIZSSIP, QLss/QIss/QLmc/QImc, LMC,
AIRX, DDM1, DDMS, DDML, TDN1, QDN1. Radiation hand-off (`USE_PLANET_RAD`, consumed by SOCRATES, out of scope): w_cloud, frac_st_water/ice, frac_cnv_water/ice,
mix_ratio_st_water/ice, mix_ratio_cnv_water/ice, dim_char_st_water/ice, dim_char_cnv_water/ice, frac_area_st, frac_area_cnv.

## 4. State carried across steps and hidden state

- Restart: see section 2. Not in restart but read at the next step: CLDSAV1, frac_area_cnv (read-modify only in the CLDMCL<=0 branch), CSIZMC above LMCMAX.
- Module variables of module CLOUDS read after MSTCNV (SVLATL, SVLAT1, SVWMXL, VSUBL, TAUMCL, CLDMCL, FSSL, PRECNVL, U00L, CSIZEL, CNVMMRL, ...): pure
  hand-over between the two ported routines within one column.
- Seed IX: advances by 3*LMCLD draws per column (3170 columns); restored to the seed after the draws at exit (isccp_diags=1).
- Stale-value hazards recorded elsewhere (AIRXL, PRHEAT when no convection; uninitialised stack above LMCLD in LSCOND) do not reach CONDSE outputs except via
  copies noted in section 1 (CSIZSS above LMCLD takes CSIZEL as left by MSTCNV/LSCOND).

## 5. Validation plan (D125/D126)

Existing: `ffd_<itime>_pre_condse.bin` / `post_condse.bin` (ATM_DRV.f.patch): U, V, T, Q, QCL, QCI, PK, PMID, PEDN, PDSIG, MA, P (halo as allocated).
Reused for those fields. Missing and new (`ffc_cse_*`, units 1330-1359): entry state not in the old dumps (TMOM, QMOM, GZ, MWs, PEK, PMIDOLD, EGCM, W2GCM, DCLEV,
PBLHT, PBLPTOP, surface averages, RSI, FEARTH, FLAND, FOCEAN, FLICE, FLAKE, SNOAGE, P_acc, PM_acc, TTOLD, QTOLD, SVLHX, RHSAV, CLDSAV, CLDSAV1, the cloud and hand-off
arrays as they are at entry (needed because several are only partly rewritten), RNDSS and the seed, UKM/VKM/UKMSP...) and the complete exit state (all of
section 3, TLS/QLS/TMC/QMC, UKM and pole arrays after the tendency step). The port runs every column of a step from the entry state and compares each exit
field with the real one (exact / bound / failure, first failing cell reported). The RANDU stream is ported (integer arithmetic) and checked against RNDSS.

## 6. Gaps (open at the time of writing)

1. Diagnostics (AIJ etc., SUBDD) and `ISCCP_CLOUD_TYPES` are not ported (diagnostic-only, D-C9).
2. RIS/RI1/RI2 (1319-1343) are computed but unused for this rundeck; ISC=0 (1288-1315): both are not exercised; the ISC arm is not ported (would raise).
3. `init_CLD` (CLOUDS2_DRV.F90:2856-3169, D-C10): parameters are taken from the recorded constants and the existing boundary dumps; `init_CLD` itself is
   reported in the D126 ledger entry as exercised only through its recorded outputs (no separate dump).
4. The stop_model exits (SVLHXL not LHE/LHS in the hand-off, ICKERR) are never reached on real data; covered only by labelled non-real tests.
5. UKM/VKM for K>4 on the pole rows (KMAX=72) is outside what the standalone LSCOND dump recorded (K<=4); the CONDSE exit pole arrays cover it.
6. libm versus libimf: pow/exp differences can flip threshold branches (see D107-D113); both modes are reported by the compare script.
