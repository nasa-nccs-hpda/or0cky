# Whole-ocean step chain plan (D118)

Owner: full-fidelity port session (Claude, for G. Tamkin). Written 2026-10-05/06.
Sources: `modelE2_planet_2.0/model/OCNDYN2.f` (OCEANS, lines 33-699 read in full), `OCNDYN.f` (GROUND_OC 4647-4883,
PRECIP_OC 5008-5090, ODIFF 5092-5575, KVINIT in OCNKPP.f:1315), `OCNKPP.f` (OCONV 1361-2885), `OCNMESO_DRV.f`
(ocnmeso_drv 131-432, densgrad 434-577), `OSTRAITS.f` (STADV 67-169, gather/scatter 942-1021), `OCEAN_COM.f`
(defaults), `README_START_HERE.md`, `FULL_FIDELITY_DELTAS.md` D33-D88, `build_and_run.md`.
Limitation: every statement about the Fortran is from reading the source; every statement about a port is from
its module/compare script. Nothing here was validated by running until the D119/D120 results (see ledger).

## Rundeck constants that set the call order (from source, not assumed)
NOCEAN=1 (OCEAN_COM.f:112), DTS=1800, DTO default 450 so NDYNO=2*NINT(.5*1800/450)=4, DTOLF=900, DTOFS=300
(OCNDYN.f:407-410), neven=NDYNO/(2*NOCEAN)=2. OBottom_drag=1, OCoastal_drag=1, OTIDE=0 (P2SAoM40.R:173-175).
USE_QUS=0, USE_OPGFQ=0, TRACERS_OCEAN/OCN_GISS_TURB/OCN_GISS_SM undefined, NMST=12 (straits active).
ITIME at OCEANS entry equals the step start itime (Itime=Itime+1 is MODELE.f:360, after the step), so ODIFF
(`mod(itime,6)==0`) fires at the first step of each 6-step window (33312, 33552, 17520 are all multiples of 6).

## Live call order of one step (OCN_DRV.f ocean_driver wraps OCEANS; PRECIP_OC is called before it, MODELE.f:328)
| # | Call (source line) | Reads / writes (ocean state) | Existing port | Status of inputs |
|---|---|---|---|---|
| 0a | KVINIT (inside PRECIP_OC, OCNDYN.f:5031) | copies layer-1 G0M(1:LSRPD),S0M,MO,GXMO,GYMO,SXMO,SYMO,UO,VO,UOD,VOD to KPP_COM | `kvinit_ff.kvinit` D58 | exact copy |
| 0b | AG2OG_precip, PRECIP_OC (5044-5075) | MO,G0M,S0M layer 1 | `precip_oc_jax.precip_oc_cell` D33 | oPREC,oEPREC,oRSI,oRUNPSI,oERUNPSI,oSRUNPSI are the recorded atm-to-ocean regrid boundary |
| 1 | AG2OG_oceans, IG2OG_oceans (OCNDYN2.f:90-96) | ocean flux arrays | none | RECORDED (AG2OG regrid; fluxes dumped in ffo tag 1) |
| 2 | GROUND_OC (OCNDYN2.f:114 -> OCNDYN.f:4647) | MO(1),S0M(1),G0M,GZMO, OPRESS, oDMSI/oDHSI/oDSSI | `osourc_jax.osourc` D34 + `ground_oc_sweep_jax` D35 | glue (flux sums, geothermal, DXYPJ*FOCEAN accumulation along I, OPRESS, pole fill) not previously ported; SHCGS was a recorded input (D35), now read from OFTAB record 3 |
| 3 | OSTRES2 (:149) | UO,VO,UOD,VOD layer 1 | `ostres2_jax` D36 | oDMUA/oDMVA/oDMUI/oDMVI recorded |
| 4 | OCONV (:152) | G0M,S0M,GX/GY/SX/SY/GZ/SZ,UO,VO,UOD,VOD,KPL | `ocnsetup`/`setup_jax`, `kppmix_jax`, `ovdiff(s)_jax`, `ocnhbl_jax.hbl_loop` D54-D66 | GAP: per-column preamble (OCNKPP.f:1648-1973: UL/ULD stencils, RAVM, LMUV, DTBYDZ, BYDZ2, DELTAE/S/M/SR, U2rho, PO) only validated through the recorded `ffz_hblin` dump; GAP: post-loop (UKM add-back to UO/VO/UOD/VOD, GZMO/SZMO flux update, GX/GY/SX/SY vertical diffusion, slope limits, KPL) has no port; hbl_loop takes ALPHAGSP/BETAGSP/SHCGS per ITER from the setup dump (recorded) and does not return AKVG/AKVS or max KBL |
| 5 | OBDRAG2, OCOAST (:161-162) | UO,VO,UOD,VOD bottom; GXMO,SXMO,GYMO,SYMO | `obdrag2_jax` D38, `ocoast_jax` D37 | none |
| 6 | polar UOD/VOD relax + polevel (:179-228) | UO,VO pole row, UOD,VOD | `polerelax_jax` D39 | none |
| 7 | NO loop (:253), NOCEAN=1: ODHORZ0 (:264) | OPBOT, MO pole copy, UO/VO pole, MMI, GUP.. , VBAR, DH | `odhorz0_jax` D40 | VUP/VDN were recorded; `eos_jax.volgsp` can compute them |
| 8 | SMU,SMV=0; copy states to MO1/MO2.. (zero elsewhere) | | none (bookkeeping) | masks nbyz* equal lmm/lmu/lmv with the pole row special case |
| 9 | 5 x ODHORZ (:312-345): (MO,..->2,DTOFS,F), (2->1,DTO,F), loop n=1..2: (1->MO,DTOLF,T), then for n=1 (odd) (MO->1,DTOLF,F) | MO,UO,VO,UOD,VOD,OPBOT, SMU,SMV accumulate on qeven calls | `odhorz_vec`/`odhorz_jax` D42/D78/D79 | GAPS: OPFIL2 is called inside ODHORZ on USMOOTH and PGFX; the ports take the post-filter arrays as RECORDED input. `opfil2_jax` (D77) exists but needs the recorded coefficient file `ffz_opcoef.bin` (calc_opfil2_coeffs not ported). The ports do not return SMU/SMV (D44 scalar port did) |
| 10 | UO(IVNP,JM,:)=VONP (:349) | | trivial | VONP persistent state (set by ODIFF) |
| 11 | OFLUXV (:354) | MO,UO,VO, SMW | `ofluxv_jax` D43 | does not return SMW, which OADVT2 needs |
| 12 | OADVT2 G0M (qlimit F), S0M (qlimit T) (:359-368) | G0M,GX,GY,GZ,S0M,SX,SY,SZ | `oadvt_jax.oadvt2_jax` D84 | MMI from step 7 (persistent, partially overwritten array: carried in state) |
| 13 | gather_ocean_straits, STPGF, STADV, STCONV, STBDRA, scatter (:477-491) | straits state, MO,G0M..SZMO at the 24 end cells | `straits_step_jax.straits_step` D73 | RECORDED: init_STRAITS start state (carried from the previous step after the first); gather/scatter are plain cell copies (no port needed, written in ocean_step) |
| 14 | ODHORZ0 again, ocnstate_derived (:406-408) | OPBOT.., DH, VBAR, G3D/S3D/P3D | `odhorz0_jax`, `ocnmeso_jax.ocnstate_derived_jax` D47/D86 | VUP/VDN recorded in the ports |
| 15 | ODIFF every 6th step (:416-420, OCNDYN.f:5092) | UO,VO,VONP | NONE | GAP, not ported (480 lines, coefficient arrays UXA.. from setup). Recorded boundary from the ffo tag 12 -> 13 snapshots |
| 16 | ocnmeso_drv: densgrad, k3d, gmkdif, gmfexp(G0M,F), gmfexp(S0M,T) (:430) | G0M,S0M,GX,GY,GZ,SX,SY,SZ | `ocnmeso_jax.densgrad_vertical_jax`, `gm_jax` D85/D86 | GAPS: horizontal densgrad (RHOX,RHOY, OCNMESO_DRV.f:520-575) has no port (recorded in D47-D51 compares); vertical part needs VUP/VDN/VUPU/VDNU (recorded) |
| 17 | TOC2SST, OG2AG_oceans, OG2IG_uvsurf (:690-695) | atmosphere-grid exchange | none | out of scope (regrid boundary) |
Diagnostics (OIJ, OIJL, CONSERV_OCE, CHECKO, timers) are not ported and do not feed the state.

## Dead for this rundeck (do not port)
OADVT3/QUS branches, all TRACERS_OCEAN blocks, OTIDE, OCNTDMIX (use_tdmix=0), OCN_GISS_TURB/SM, USE_OPGFQ=1, ENHANCED_DEEP_MIXING (not defined; to be confirmed by the validation, not assumed).

## Recorded or unported boundaries that remain in the chain
1. AG2OG/IG2OG flux fields (regrid, step 1) - recorded each step.
2. OPFIL2 coefficient setup (`ffz_opcoef.bin`).
3. init_STRAITS start state.
4. ODIFF (unported) - recorded UO,VO,VONP before/after on ODIFF steps.
5. Per-column OCONV pieces that were recorded: the preamble (now ported here), ALPHAGSP/BETAGSP/SHCGS (now tables), HBL-loop EOS (table since D76).
6. Table lookups: OFTAB records (VGSP, TGSP, CGS, HGSP, AGSP, BGSP) are read from the original ModelE_Support file, read-only.

## Validation design
`ffo_state_<itime>.bin` snapshots (ocean_chain_io.py docstring lists the 15 tags) bracket every stage; stages are replayed (a) from the real previous snapshot to isolate local error and (b) chained from the entry snapshot through the port outputs. Entry-to-exit compares per step over 12 steps per date.
Instrumentation: `instrumentation/OCNDYN2_oceanchain.f.patch`, `OCNDYN_oceanchain.f.patch` (units 1270-1271, grepped free).
