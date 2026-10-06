# Radiation server (real RADIA called inside the real model, SOCRATES as a black box): design, READ list, oracle (D152-D154)

Status: built and run 2026-10-06. Project-local per `projects/imvi/AGENTS.md`. Owner: project owner of `rocke3d_jax` (Glenn Tamkin); drafted by a Claude Code session. Review by: the session that wires the server into a multi-step loop.
Standing constraint (user): SOCRATES is third party and is never ported or modified; here it is only called as part of the unmodified real ModelE objects (linked in a scratch build; the original tree was not touched).

Files: `instrumentation/ATM_DRV_radsrv.f.patch` (diff -u against the pristine `ATM_DRV.f`, 2 local hunks; also applies after `ATM_DRV_atmstep.f.patch`, offsets 1 and 131 lines; units 1440 only, 1440-1469 reserved, grepped: no other use), `radiation_server.py`, `radiation_server_compare.py`, `tests/test_radiation_server.py`.
Sources read for this note (line numbers refer to the pristine `model/RAD_DRV.f` unless stated): RADIA 1940-5501 in full for the effective (P2SAoM40: `USE_PLANET_RAD`, `GISS_RAD_OFF`) code path, using the preprocessed `RAD_DRV.f.pp` that already existed in the scratchpad (its provenance was not re-verified; its RADIA body agrees with every original-source line cited below that I cross-checked by grep), `RADIATION.f` 1790-1942 (`RCOMPX`), `ATM_DRV.f` 60-300.

## 1. Protocol (what exists)

File exchange (the "2.D-like" option, but without a Python restart writer): one model process per call.
`RADSRV_MODE=serve RADSRV_ITIME=<itime> RADSRV_IN=<packet> RADSRV_OUT=<packet> ./P2SAoM40 -i I` starts from the restart, runs the real steps up to `<itime>` (must be a radiation step, `MOD(Itime-ItimeI,NRAD)==0`), and at `CALL RADIA` (`ATM_DRV.f:274`) the hook overwrites every RADIA input in the table below from the packet, zeroes `AIJ_loc` (so the AIJ accumulations of that call are an exact per-call delta), calls the real `RADIA`, writes the output packet and stops.
`RADSRV_MODE=dump` leaves the run untouched (verified: the 24 `ffa_step_*` files of a 6-step run with the hook active are byte-identical to `ff_data/nov26`) and records the live input/output packets of every radiation step in the window (`rsv_<tag>_<itime>_in/out.bin`).
Unset `RADSRV_MODE` is a no-op.
Packet: big-endian records `char*16 name, int32 rank, int32 n(1:4), float64 data` in Fortran order, inputs in the fixed order of `radiation_server.INPUT_FIELDS` (the Fortran reader stops on any name/shape mismatch).
Start-up cost (measured, 12-core node, 1 thread, nothing else heavy running): serve call for step 0: 27 s wall total (model initialisation + 1 RADIA call of 3312 columns); for step 5: 42 s (5 extra full model steps first). The model's own timer table in the 6-step dump run gives RADIA = 10.9 s max per radiation step (non-radiation steps 0.0008 s), i.e. about 11 s of the 27 s is RADIA itself and about 16 s is start-up (restart read, `init_RAD`, spectral files). A persistent FIFO loop would remove the start-up part only if the model state could be advanced inside the server; not implemented (not needed for the oracle).

## 2. READ list of RADIA (effective code path) and where each input comes from

Legend: PACKET = overwritten from the packet by the hook (and perturbed in the audit); RESTART = module state restored from the restart / run set-up, not in the packet; TIME = derived from the model clock (`modelEclock`, itime), identical for the same itime; PARAM = rundeck/compile-time constant; RECORDED = which existing dump holds it (nov26 33312 checked bitwise against the live model state: 33 of 33 existing-dump inputs equal).

| Input (RADIA / RCOMPX name) | Source line | Status | Existing dump |
|---|---|---|---|
| `T(I,J,L)` -> `TLM=T*PK` | 2783, 5476 | PACKET `T` | ffc_cse_out (T), ffd/ffa_step |
| `Q(I,J,L)` -> `SHL`, `RHL`; negative Q reset to 0 in state | 2778-2782, 3199 | PACKET `Q` | ffc_cse_out |
| `PK(L,I,J)` | 2783, 5476 | PACKET `PK` | ffc_cse_in |
| `PEDN(L,I,J)` -> `PLB`, `p_level` | 3180-3181 | PACKET `PEDN` | ffc_cse_in |
| `PMID(L,I,J)` (RH via `QSAT`) | 3199 | PACKET `PMID` | ffc_cse_in |
| `PDSIG`, `MA` (heating uses `byMA`) | cloud diagnostics, 5476 (`byMA`) | PACKET `PDSIG`, `MA`, `BYMA` | ffc_cse_in (PDSIG), ffa_step_r (MA); BYMA only in dump-mode packet |
| 15 cloud hand-off arrays `w_cloud, frac_st/cnv_water/ice, mix_ratio_*, dim_char_*, frac_area_st/cnv` (reverse-copied into the 43-layer `*_l` arrays, 3 radiation-only layers set to 0) | 3124-3160 | PACKET (`W_CLOUD ... FRAC_AREA_CNV`) | ffc_cse_out |
| `TAUSS, TAUMC` (threshold `taulim` masks CLDSS/CLDMC to 0), `CLDSS, CLDMC` (`get_cld_overlap`, overlap draw, diagnostics) | 2611-2615, 3907-3925 | PACKET; CLDSS/CLDMC are also OUTPUTS (masked in state) | ffc_cse_out; post-RADIA CLDSS/CLDMC = next step's ffc_cse_in |
| `RQT(K,I,J)` (3 radiation-only layers; updated at 5464) | 3282, 5464 | PACKET `RQT` (also OUTPUT) | NOT in any dump (restart value at step 0; dump-mode packet otherwise) |
| `KLIQ(L,4,I,J)` (dry/wet memory flags, read at 3727 `kdeliq=kliq`, written back 4382) | 3727, 4382 | PACKET `KLIQ` (also OUTPUT) | not in any dump |
| `SNOAGE(3,I,J)` -> `AGESN` | 3320-3322 | PACKET `SNOAGE` | ffc_cse_out |
| `LTROPO(I,J)` (`LS1_loc`, tropopause diagnostics) | 2753 | PACKET `LTROPO` | not in the cse dumps |
| `FLAND, FLICE, FEARTH, FLAKE` -> `PLAND,PLICE,PEARTH,PLAKE`; `RSI` -> `POICE,POCEAN` | 2708-2713 | PACKET | ffc_cse_in |
| `atmocn/atmice/atmgla/atmlnd%GTEMPR` -> `TGO,TGOI,TGLI,TGE` (also `TRSURF`, `TRHR(0)`) | 3310-3313, 4441 | PACKET `GTEMPR1..4` | not in any dump as 2-D fields |
| `atmsrf%TSAVG` -> `TSL`; `atmsrf%WSAVG` -> `WMAG` | 3314, 3391 | PACKET `TSAVG`, `WSAVG` | TSAVG ffc_cse_in; WSAVG not recorded |
| `SI_ATM%SNOWI`, `atmgla%SNOW`, `atmice%ZSNOWI`, `SI_ATM%ZSI`, `SI_ATM%POND_MELT` (-> `fmp`, `zmp`), `FLAG_DSWS` | 3315-3316, 3325, 3357-3363 | PACKET | not recorded |
| `GHY SNOWD(2,I,J)`, `atmlnd%fr_snow_rad(2)`, `atmlnd%bare_soil_wetness` -> `WEARTH` | 3318-3319, 3378 | PACKET `SNOWD`, `FRSNOW`, `BARESW` | not recorded |
| `DLAKE` -> `zlake` | 3374 | PACKET `DLAKE` | not recorded |
| Zenith: `COSZA, COSZ2` from `calc_zenith_angle(modelEclock, NRAD, ...)`; `COSZ1` | 2468, 2503-2511 | TIME (model clock) | COSZ1 in ffa_step_r |
| `RSDIST` -> `solar_irrad = solar_constant/RSDIST`; `JDAYR, JYEARR` | 2512-2516 | TIME/RESTART (daily orbit update `DAILY_orbit`) | - |
| Gases (`seth2o`, `getgas`: GHG file, ozone, `H2ObyCH4`; `fpxscalegas`), volcanic aerosol (`get_volc_column`), `GETEPS` (static `cloud.epsilon4.72x46`), `GETSUR` tables, `init_planet_rad` spectral files | RADIATION.f 1824-1868 (read), 1886 | RESTART/TIME/static input files (read at model init from the original `ModelE_Support`, read-only) | - |
| Ent vegetation fractions `PVT` (`ent_get_exports`, `map_ent2giss`) | 3384-3390 | RESTART (Ent state, changes with phenology, not per radiation step) - NOT in the packet | - |
| `LOC_CHL` (`atmocn%chl`, only if `chl_from_seawifs/obio>0`) | 2741-2749 | PARAM (both 0 in this rundeck -> -1e30) | - |
| `dALBsn = dALBsnX*BCdalbsn` | 3328-3332 | PARAM/RESTART (`dalbsnX` read from the rundeck; BC dalbsn file) - NOT in the packet | - |
| `kradia` (=0), `cldx`, `RHfix`, `cloud_rad_forc`, `cloud_aer_o3_rad_forc`, `aer_rad_forc`, `moddrf`, `taulim` | 2459-2517, 3448-3730 | PARAM | - |
| random numbers `RANDU` (`RDMC`, `RDSS` draws, overlap) | 2587-2620, 3907 | RESTART (seed); only used for diagnostics here (`TOTCLD`, `CSS/CMC`), not for planet_rad fluxes | seeds in ffc_cse (SEEDS) |

Not read by RADIA in this build (confirmed by the effective code; kept out of the packet): `QCL`, `QCI`, `U`, `V`, tracers, `RCLD` (not set by RADIA here, restart value passes through), classic GISS cloud/gas/aerosol routines (`TAUGAS, GETCLD, THERML, SOLARM` are compiled out by `GISS_RAD_OFF`).

Block 2555-3160 (cloud block), read: with `GISS_RAD_OFF` it reduces to (i) the random draws and the `taulim` mask of CLDSS/CLDMC (2587-2620), (ii) `Q<0 -> 0` and `SHL/TLM` (2778-2783), (iii) the copy of the 15 CLOUDS_COM hand-off arrays into the layer-reversed `planet_rad` arrays (3124-3160), (iv) `cld_scaling = cldx`. There is no optical-depth assembly from TAUSS/TAUMC in the planet_rad build: the optical properties are rebuilt inside `planet_rad` from the mixing ratios and characteristic sizes.
Post-processing 3740-4560, read: cloud diagnostics (`w_cloud`-based AJL/AIJL, `tot_cloud_cover -> CFRAC`), the 3907-3925 overlap diagnostics (`CSS, CMC`), TOA band diagnostics (AIJ), `rad_to_chem`, `kliq` write-back, then the exports `FSF(4) <- FSRNFG(1,3,4,2)`, `SRHR(0)=SRNFLB(1)`, `TRHR(0)=STBO*sum(p*TGR^4)-TRNFLB(1)`, `TRSURF(4)`, `SRHR/TRHR(1..LM)`, `SRHRS/TRHRS -> RQT` (5464), `ALB(I,J,1:9)`, `SRDN, FSRDIR, SRVISSURF, DIRVIS, FSRDIF, DIRNIR, DIFNIR` (4437-4551), and the AIJ/AJ/AREG accumulations. `T` is advanced inside RADIA (label 900, every step) with `COSZ1`.

Outputs in the output packet: `T, Q, SRHR(0:LM), TRHR(0:LM), RQT, KLIQ, SNOAGE, CLDSS, CLDMC, FSF(4), TRSURF(4), ALB(9), FSRDIR, SRVISSURF, FSRDIF, DIRVIS, DIRNIR, DIFNIR, SRDN, CFRAC, COSZ1` and the whole `AIJ_loc` (72x46x750) of the call. TOA fluxes are not module outputs of RADIA (locals `SNFS/TNFS`), so they are taken from the exact AIJ delta (`IJ_SRNFP0 = SNFS(3)*COSZ2`, `IJ_TRNFP0 = -TNFS(3)`, `IJ_SRNFP1`, ... band-by-band `IJ_TRUPTOA_BND1..`, `IJ_SRUPTOA_BND1..`, `IJ_SRDNTOA_BND1..`). `AJ/AJL/AREG/ADIURN` accumulations of the call are not exported.
Known gaps of the packet (restored from the restart, therefore equal to the real model only at the restart date/itime): `PVT` (Ent), `BCdalbsn`, orbit (`RSDIST`), gas/ozone/volcanic tables, random seeds. A perturbation audit of these is not possible through the packet (see section 4).

## 3. Oracle result (nov26, steps 33312 = 0 and 33317 = 5)

Setup: restart `ff_data/_pristine_restarts/fort1_nov26_itime33312.nc`; server packet built from the EXISTING dumps for everything they hold (T, Q, PK, PMID, PDSIG, PEDN, MA, the 15 cloud hand-off arrays, TAUSS/TAUMC/CLDSS/CLDMC, SNOAGE, RSI/FEARTH/FLAND/FLICE/FLAKE/TSAVG: 33 of 52 fields) and from the dump-mode packet of the live model for the other 19 (RQT, KLIQ, LTROPO, BYMA, GTEMPR1-4, WSAVG, ZSI, SNOWI, POND_MELT, FLAG_DSWS, DLAKE, SNOWLI, ZSNOWI, BARESW, FRSNOW, SNOWD; these were never dumped before: they are a new recording, produced by the same bitwise-reproducing trajectory).
All 33 existing-dump fields equal the live model state bitwise (both steps).
Result (run by `python radiation_server_compare.py oracle --steps 33312,33317`, both steps identical in outcome):
- BITWISE EQUAL (max abs difference 0.0, 0 differing cells) to the recorded real-model `ffa_step_<it>_r`: `T, Q, SRHR(0:40), TRHR(0:40), COSZ1`.
- BITWISE EQUAL to the live model outputs of the same trajectory (dump-mode output packet): `RQT, KLIQ, SNOAGE, CLDSS, CLDMC, FSF(4), TRSURF(4), ALB(9), FSRDIR, SRVISSURF, FSRDIF, DIRVIS, DIRNIR, DIFNIR, SRDN, CFRAC` (and again `T Q SRHR TRHR COSZ1`).
- BITWISE EQUAL to existing recorded dumps one step later (step 0 only): `CLDSS, CLDMC, SNOAGE` after the RADIA mask = next step's `ffc_cse_in_33313`; `FSF(1)*COSZ1` = SRHEAT of all 5418 ocean-tile records in `ffs_33312`.
- NOT bitwise: the AIJ accumulations. The server's AIJ is an exact per-call delta (AIJ zeroed before RADIA); the dump-mode reference is `AIJ_after - AIJ_before`, which is rounded: 194,607 of 5,497,920 entries differ, max abs 5.8e-11, max relative (scaled by max(|x|,1)) 4.6e-11 (step 5: 194,791, 5.8e-11, 4.9e-11). This is consistent with rounding of a difference of large accumulators; I did NOT verify that bound against the accumulator magnitudes, so it is stated as "consistent with", not proved. The TOA fluxes taken from these deltas (`IJ_SRNFP0`, `IJ_TRNFP0`, band TOA fluxes) therefore have no independent real-model record to compare with; they were not compared with anything else.
Not tested: the first-step clock/orbit/ozone inputs at another date (dec01, jan01); steps other than 0 and 5 of nov26; any RADIA input that is neither in the packet nor perturbable (see section 4).
Wall time of one server call: 27 s (step 0), 42 s (step 5, five model steps first); RADIA itself about 11 s of the 27 s (model timer, 6-step dump run, max trip 10.9 s).

## 4. Audit: does every packet field move the outputs? (`python radiation_server_compare.py audit`, 52 serve runs, 10 concurrent, 210 s total; step 33312)
Each field was perturbed alone (T: +1e-6 K; other real fields: x(1+1e-6), or +1e-6 where identically zero; KLIQ and FLAG_DSWS flipped; LTROPO +1) and all 22 outputs were compared bitwise with the unperturbed server run.
Fields that moved outputs (all of the rest, 45 of 52): every atmospheric field, all 15 cloud hand-off arrays, RQT, SNOAGE, RSI, ZSI, FLICE, FLAND, FEARTH, GTEMPR1-4 (move TRHR, TRSURF, T, RQT, AIJ only: LW only, as expected), WSAVG, SNOWLI, ZSNOWI, BARESW, FRSNOW, SNOWD, POND_MELT, FLAG_DSWS (3e2 W m-2 in SRHR: ponded-snow flag switches the sea-ice albedo branch), KLIQ (changes KLIQ and AIJ), PDSIG (AIJ only), BYMA (T only), LTROPO (AIJ only), CLDSS/CLDMC (themselves, 1e-6: only the threshold mask changes them; they do not feed planet_rad), the 15 cloud arrays (SRHR, TRHR, FSF, ALB, CFRAC...).
Fields that did NOT move at the 1e-6 level, re-tested with a macroscopic change (`auditbig`):
- `TAUSS`, `TAUMC`: not moving at 1e-6 relative (threshold mask `taulim`); with TAUSS/TAUMC=0 only `CLDSS`/`CLDMC` and AIJ change (SRHR/TRHR unchanged): TAUSS/TAUMC do not enter the planet_rad radiation, only the mask (RAD_DRV.f 2611-2615).
- `FLAKE`, `DLAKE`: not moving at 1e-6; moving with FLAKE x1.5+0.1 and DLAKE=0.1 m (12 of 22 outputs): real inputs, insensitive to a 1e-6 relative change only (lake depth enters the albedo through a saturating function).
- `MA`: no change even at x1.5: not read by RADIA in this build (only `byMA`, which is in the packet and moves T through the heating line 5476; kept in the packet so the heating uses the same value).
- `SNOWI` (x2+10 kg m-2): no change. `SNOWOI=SNOWI` (RAD_DRV.f 3315) is passed to `GETSUR` but ALBEDO.f:612 uses it only in the `KSIALB==1` branch; this rundeck's value of `KSIALB` was NOT checked (the zsnowi path moves outputs). Candidate input only for `KSIALB==1`.
- `TSAVG` (+50 K): no change: `TSL` is passed to RCOMPX (RADIATION.f:146) but no use of `TSL` was found in the planet-rad path (only commented lines `!sl`); not read in this build.
No unexplained non-mover remains among the packet fields. Not auditable through the packet (restored from the restart, not perturbed): `PVT` (Ent), `BCdalbsn`, `RSDIST`, GHG/ozone/volcanic inputs, random seeds; a missing packet field of the kind "read by RADIA but absent from the table" would not show in this audit by construction. The table in section 2 was built by reading the effective RADIA/RCOMPX code; `GETSUR`, `GETEPS`, `getgas` and `planet_rad` bodies were not read line by line (grep of the call arguments only).

## 5. What remains
- Persistent server (FIFO loop) and advancing the model state inside the server: not done; every call pays the start-up (about 16 s) and, for later steps, the real model steps before it.
- Oracle at dec01/jan01 and more steps; `PVT/BCdalbsn/orbit` perturbation audit through a model-side override.
- Per-radiation-step cost 11 s single thread; no parallel decomposition tried.
- The packet contains the full live surface state; building it from a Python-side ported surface needs `GTEMPR1-4`, `WSAVG`, `ZSNOWI`, `SNOWD`, `FRSNOW`, `BARESW`, `POND_MELT`, `FLAG_DSWS` (not in the existing dumps).
