"""R6 instrumentation, second stage (after apply_ffsoc.py): planet_rad-side dumps and direct/tile fluxes.  SCRATCH copies only.
usage: python apply_ffsoc_r6.py SCRATCH_SOCRATES_DIR SCRATCH_MODEL_DIR     (never run on the original tree; refuses ModelE_Support)
  * copies the extended ffsoc_mod.F90 (ffsoc_pl1/2, env FFSOC_PL) into SCRATCH_SOCRATES_DIR/src/radiance_core
  * radiance_calc.F90: tag 30 k=3 flux_direct (solar) and k=4 flux_up_tile per band, next to the existing k=1,2
  * planet_rad.F90 (scratch model dir): tag 50 (arguments and module inputs of run_planet_rad), 52/53 (aerosol set-up),
    51 (outputs of get_planet_radout).   Layouts: README.md table (R6 tags)."""
import os, sys, shutil
sc, md = sys.argv[1], sys.argv[2]
assert "ModelE_Support" not in sc and "ModelE_Support" not in md, "refusing to touch the original tree"
rc = os.path.join(sc, "src", "radiance_core")
def edit(path, pairs):
    s = open(path).read()
    for a, b, cnt in pairs:
        assert s.count(a) == cnt, (path, a[:60], s.count(a))
        s = s.replace(a, b)
    open(path, "w").write(s)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffsoc_mod.F90"), os.path.join(rc, "ffsoc_mod.F90"))
edit(os.path.join(rc, "radiance_calc.F90"), [
 ("    CALL ffsoc_arr(30, control%isolir, i_band, 2, radout%flux_up(:,:,control%map_channel(i_band)))\n",
  "    CALL ffsoc_arr(30, control%isolir, i_band, 2, radout%flux_up(:,:,control%map_channel(i_band)))\n"
  "    IF (control%isolir == 1) CALL ffsoc_arr(30, control%isolir, i_band, 3, radout%flux_direct(:,:,control%map_channel(i_band)))\n"
  "    IF (ALLOCATED(radout%flux_up_tile)) CALL ffsoc_arr(30, control%isolir, i_band, 4, radout%flux_up_tile(:,:,control%map_channel(i_band)))\n", 1)])
edit(os.path.join(md, "planet_rad.F90"), [
 ("  USE def_out, ONLY: allocate_out\n", "  USE def_out, ONLY: allocate_out\n  USE ffsoc_mod, ONLY: ffsoc_pl1, ffsoc_pl2\n", 1),
 ("  CALL set_t_level\n#endif\n", """  CALL set_t_level
#endif

! ffsoc: inputs of the radiation calls (R6 oracle)
  CALL ffsoc_pl1(50, (/ p_level, t_layer, t_level, RESHAPE(ULGAS, (/ npd_layer*13 /)), CLDEPS, &
      RESHAPE(PRNB, (/ 24 /)), RESHAPE(PRNX, (/ 24 /)), cos_zenith, solar_irrad, &
      w_cloud_l, frac_st_water_l, frac_st_ice_l, frac_cnv_water_l, frac_cnv_ice_l, &
      mix_ratio_st_water_l, mix_ratio_st_ice_l, mix_ratio_cnv_water_l, mix_ratio_cnv_ice_l, &
      dim_char_st_water_l, dim_char_st_ice_l, dim_char_cnv_water_l, dim_char_cnv_ice_l, &
      frac_area_st_l, frac_area_cnv_l, cld_scaling, REAL(n_aer, RealK), REAL(n_aer_tracer, RealK), &
      rel_humidity(1:npd_layer), &
      wind_speed_mag, frac_sea, frac_land, frac_sea_ice, frac_land_ice, &
      t_rad_sea, t_rad_land, t_rad_sea_ice, t_rad_land_ice, soil_wetness, &
      frac_veg_tbl, frac_land_snow_tbl, snow_depth_land_tbl, snow_depth_sea_ice, snow_amnt_sea_lake /))
  IF (n_aer > 0) THEN
    CALL ffsoc_pl2(52, RESHAPE((/ REAL(aer_component(1:n_aer), RealK), aer_radius_eff_dry(1:n_aer), &
        aer_frac_sulph(1:n_aer), aer_scaling_sw(1:n_aer), aer_scaling_lw(1:n_aer) /), (/ n_aer, 5 /)))
    CALL ffsoc_pl2(53, aer_mass_div_dens(1:n_aer, 1:npd_layer))
  END IF
""", 1),
 ("#endif\n  )\n\n  USE rad_pcf\n", "#endif\n  )\n\n  USE rad_pcf\n  USE ffsoc_mod, ONLY: ffsoc_pl1\n", 1),
 ("END SUBROUTINE get_planet_radout\n", """  CALL ffsoc_pl1(51, (/ TLB, TLT, TRUFLB, TRDFLB, TRNFLB, TRFCRL, SRUFLB, SRDFLB, SRNFLB, SRFHRL, FSRNFG, &
      SRIVIS, SROVIS, PLAVIS, SRINIR, SRONIR, PLANIR, SRDVIS, SRUVIS, ALBVIS, SRDNIR, SRUNIR, ALBNIR, &
      SRXVIS, SRXNIR, S0, TRUPTOA, SRUPTOA, SRDNTOA, tot_cloud_cover /))
END SUBROUTINE get_planet_radout
""", 1)])
edit(os.path.join(md, "planet_alb.F90"), [
 ("  USE def_bound,    ONLY: StrBound\n", "  USE def_bound,    ONLY: StrBound\n  USE ffsoc_mod,    ONLY: ffsoc_pl2\n", 1),
 ("!       Calculated direct albedos in radiation bands\n\n  ALLOCATE( &\n      alb_diff_rad_band(n_rad_band), &",
  "!       Calculated direct albedos in radiation bands\n  LOGICAL, SAVE :: l_wgt_dumped = .FALSE.\n\n  ALLOCATE( &\n      alb_diff_rad_band(n_rad_band), &", 1),
 ("! Zero the albedos initially\n  bound%rho_alb(n_profile, ip_surf_alb_dir,  1:n_rad_band) = 0.0_RealK\n",
  "  IF (.NOT. l_wgt_dumped) THEN\n    l_wgt_dumped = .TRUE.\n    CALL ffsoc_pl2(54, alb_wgt_sw)\n  END IF\n\n"
  "! Zero the albedos initially\n  bound%rho_alb(n_profile, ip_surf_alb_dir,  1:n_rad_band) = 0.0_RealK\n", 1)])


# planet_rad.F90: static records (control flags, band limits, planet aerosol tables), first call only (tags 101-170)
def _flags(c):
    names = ["l_co2", "l_o3", "l_o2", "l_no2", "l_n2o", "l_ch4", "l_so2", "l_n2", "l_cfc11", "l_cfc12", "l_ar", "l_h2",
             "l_cloud", "l_aerosol", "l_microphysics"]
    return ", &\n        ".join("MERGE(1.0_RealK, 0.0_RealK, %s%%%s)" % (c, n) for n in names)
def _aer(b, t):
    L = []
    L.append("    CALL ffsoc_pl1(%d, (/ REAL(%s%%n_component, RealK), REAL(%s%%n_radius_max, RealK), &\n        REAL(%s%%n_humidity, RealK), &\n"
             "        REAL(%s%%n_band, RealK) /))\n" % (b, t, t, t, t))
    L.append("    CALL ffsoc_pl1(%d, REAL(%s%%i_component, RealK))\n" % (b + 1, t))
    L.append("    CALL ffsoc_pl1(%d, REAL(%s%%n_radius, RealK))\n" % (b + 2, t))
    L.append("    CALL ffsoc_pl1(%d, REAL(%s%%i_component_map, RealK))\n" % (b + 3, t))
    L.append("    CALL ffsoc_pl1(%d, %s%%humidity)\n" % (b + 4, t))
    L.append("    CALL ffsoc_pl1(%d, MERGE(1.0_RealK, 0.0_RealK, %s%%l_humidity))\n" % (b + 5, t))
    L.append("    CALL ffsoc_pl2(%d, %s%%radius_eff_dry)\n" % (b + 6, t))
    L.append("    CALL ffsoc_pl1(%d, %s%%density)\n" % (b + 7, t))
    for j, f in enumerate(("k_abs", "k_scat", "g_asym")):
        L.append("    CALL ffsoc_pl2(%d, RESHAPE(%s%%%s, (/ SIZE(%s%%%s)/%s%%n_component, %s%%n_component /)))\n"
                 % (b + 8 + j, t, f, t, f, t, t))
    return "".join(L)
static = """  IF (.NOT. l_ffsoc_static) THEN
    l_ffsoc_static = .TRUE.
    CALL ffsoc_pl1(101, (/ &
        %s, &
        REAL(ffsoc_i_sea_ice, RealK), REAL(i_cloud_ice_size_scheme, RealK), REAL(control_lw%%i_cloud, RealK), &
        REAL(control_lw%%i_inhom, RealK) /))
    CALL ffsoc_pl1(102, (/ &
        %s, &
        REAL(ffsoc_i_sea_ice, RealK), REAL(i_cloud_ice_size_scheme, RealK), REAL(control_sw%%i_cloud, RealK), &
        REAL(control_sw%%i_inhom, RealK) /))
    CALL ffsoc_pl1(103, spectrum_lw%%basic%%wavelength_short)
    CALL ffsoc_pl1(104, spectrum_lw%%basic%%wavelength_long)
    CALL ffsoc_pl1(105, REAL(spectrum_lw%%basic%%n_band_exclude, RealK))
    CALL ffsoc_pl2(106, REAL(spectrum_lw%%basic%%index_exclude, RealK))
    CALL ffsoc_pl1(107, spectrum_sw%%basic%%wavelength_short)
    CALL ffsoc_pl1(108, spectrum_sw%%basic%%wavelength_long)
    CALL ffsoc_pl1(109, REAL(spectrum_sw%%basic%%n_band_exclude, RealK))
    CALL ffsoc_pl2(110, REAL(spectrum_sw%%basic%%index_exclude, RealK))
    CALL ffsoc_pl1(111, spectrum_sw%%solar%%weight_blue)
    CALL ffsoc_pl1(112, (/ &
        spectrum_lw%%drop%%parm_min_dim(control_lw%%i_st_water), &
        spectrum_lw%%drop%%parm_max_dim(control_lw%%i_st_water), &
        spectrum_lw%%drop%%parm_min_dim(control_lw%%i_cnv_water), &
        spectrum_lw%%drop%%parm_max_dim(control_lw%%i_cnv_water), &
        spectrum_lw%%ice%%parm_min_dim(control_lw%%i_st_ice), &
        spectrum_lw%%ice%%parm_max_dim(control_lw%%i_st_ice), &
        spectrum_lw%%ice%%parm_min_dim(control_lw%%i_cnv_ice), &
        spectrum_lw%%ice%%parm_max_dim(control_lw%%i_cnv_ice) /))
    CALL ffsoc_pl1(113, (/ &
        spectrum_sw%%drop%%parm_min_dim(control_sw%%i_st_water), &
        spectrum_sw%%drop%%parm_max_dim(control_sw%%i_st_water), &
        spectrum_sw%%drop%%parm_min_dim(control_sw%%i_cnv_water), &
        spectrum_sw%%drop%%parm_max_dim(control_sw%%i_cnv_water), &
        spectrum_sw%%ice%%parm_min_dim(control_sw%%i_st_ice), &
        spectrum_sw%%ice%%parm_max_dim(control_sw%%i_st_ice), &
        spectrum_sw%%ice%%parm_min_dim(control_sw%%i_cnv_ice), &
        spectrum_sw%%ice%%parm_max_dim(control_sw%%i_cnv_ice) /))
%s%s  END IF

""" % (_flags("control_lw"), _flags("control_sw"), _aer(120, "aer_tbl_lw"), _aer(140, "aer_tbl_sw"))
edit(os.path.join(md, "planet_rad.F90"), [
 ("  USE ffsoc_mod, ONLY: ffsoc_pl1, ffsoc_pl2\n", "  USE ffsoc_mod, ONLY: ffsoc_pl1, ffsoc_pl2\n  USE lw_control, ONLY: ffsoc_i_sea_ice => i_sea_ice_alb\n"
  "  LOGICAL, SAVE :: l_ffsoc_static = .FALSE.\n", 1),
 ("! Calculate diagnostic aerosol optical properties\n  IF (l_aerosol_diag .and. l_aerosol_diag_onoff) &\n",
  "! ffsoc: static records\n" + static + "! Calculate diagnostic aerosol optical properties\n  IF (l_aerosol_diag .and. l_aerosol_diag_onoff) &\n", 1)])

# RADIATION.f (RCOMPX): climatology-level inputs of the planet radiation set-up (tags 60-62)
edit(os.path.join(md, "RADIATION.f"), [
 ("      USE planet_rad, ONLY: n_aer, n_aer_tracer,\n     &                      run_planet_rad, get_planet_radout,\n     &                      deallocate_planet_rad\n",
  "      USE planet_rad, ONLY: n_aer, n_aer_tracer,\n     &                      run_planet_rad, get_planet_radout,\n     &                      deallocate_planet_rad\n      USE ffsoc_mod, ONLY: ffsoc_pl1, ffsoc_pl2\n      use ghgmod, only : ffsoc_fulgas=>fulgas, ffsoc_ppmv80=>ppmv80\n      use AerParam_mod, only : ffsoc_drym2g=>DRYM2G\n", 1),
 ("      CALL set_planet_alb_param\n      CALL run_planet_rad(",
  """      CALL set_planet_alb_param
      CALL ffsoc_pl1(60, (/ REAL(ILON,8), REAL(JLAT,8), COSZ,
     &    REAL(LS1_loc,8), REAL(KFPCO2,8), PLB(1:LX+1),
     &    PLB0(1:LX+1), HLB0(1:LX+1),
     &    TLM(1:LX), SHL(1:LX), RHL(1:LX), CLDEPS(1:LX),
     &    RESHAPE(U0GAS(1:LX,1:13), (/ LX*13 /)), ffsoc_fulgas(1:13),
     &    ffsoc_ppmv80(1:13), FPXCO2(1:LX),
     &    RESHAPE(ATAULX(1:LX,1:6), (/ LX*6 /)), VTAULX(1:LX), VReff,
     &    Q55H2S, AGESN(1:3), ZOICE, FMP, ZMP, ZLAKE, SNOWLI, dalbsn,
     &    LOC_CHL, PLAKE, MERGE(1.0D0,0.0D0,flags), SNOWOI, ZSNWOI,
     &    FSTAER, FTTAER, FSAAER, FTAAER, FSDAER, FTDAER, FSVAER,
     &    FTVAER, FS8OPX(1:8), FT8OPX(1:8), DENAER(1:8), REFDRY(1:8),
     &    FRSULF(1:8), ffsoc_drym2g(1:8) /))
      IF (ALLOCATED(DTAULX)) THEN
        CALL ffsoc_pl2(61, DTAULX)
        CALL ffsoc_pl1(62, (/ TAUCON_dust /))
      END IF
      CALL run_planet_rad(""", 1)])

# RAD_DRV.f (RADIA): static / daily records of the RADIA-level port (tags 170-181), first call only
edit(os.path.join(md, "RAD_DRV.f"), [
 ("      use RADPAR, only: nintaerext,nintaersca,nintaerasy\n",
  "      use RADPAR, only: nintaerext,nintaersca,nintaerasy\n"
  "      use ffsoc_mod, only: ffsoc_pl1, ffsoc_pl2\n"
  "      use RADPAR, only: ffsoc_eplow=>EPLOW, ffsoc_epmid=>EPMID,\n"
  "     &     ffsoc_ephig=>EPHIG\n"
  "      use GTAU_state_mod, only: ffsoc_gtau=>GTAU, ffsoc_salbtg=>SALBTG,\n"
  "     &     ffsoc_taugsa=>TAUGSA, ffsoc_tautgs=>TAUTGS\n"
  "      use DustParam_mod, only: ffsoc_redust=>redust,\n"
  "     &     ffsoc_rodust=>rodust\n"
  "      use SURF_ALBEDO, only: ffsoc_albvnh=>albvnh\n"
  "", 1),
 ("      IMPLICIT NONE\n      real*8 dz,rho\n",
  "      IMPLICIT NONE\n      LOGICAL, SAVE :: ffsoc_stat = .FALSE.\n      real*8 dz,rho\n", 1),
 ("      solar_irrad = solar_constant/RSDIST\n",
  """      solar_irrad = solar_constant/RSDIST
      IF (.NOT. ffsoc_stat) THEN
        ffsoc_stat = .TRUE.
        CALL ffsoc_pl1(170, byaml00(1:LM+LM_REQ))
        CALL ffsoc_pl1(171, plb0(1:LM_REQ))
        CALL ffsoc_pl1(172, shl0(1:LM_REQ))
        CALL ffsoc_pl1(173, (/ RSDIST, solar_constant /))
        CALL ffsoc_pl2(174, RESHAPE(ffsoc_gtau, (/ 51*11, 143 /)))
        CALL ffsoc_pl2(175, ffsoc_salbtg)
        CALL ffsoc_pl2(176, ffsoc_taugsa)
        CALL ffsoc_pl1(177, ffsoc_tautgs)
        CALL ffsoc_pl1(178, ffsoc_redust)
        CALL ffsoc_pl1(179, ffsoc_rodust)
        CALL ffsoc_pl2(180, RESHAPE(ffsoc_albvnh, (/ 72, 2 /)))
        CALL ffsoc_pl2(181, ffsoc_eplow)
        CALL ffsoc_pl2(182, ffsoc_epmid)
        CALL ffsoc_pl2(183, ffsoc_ephig)
      END IF
""", 1)])
print("applied")
