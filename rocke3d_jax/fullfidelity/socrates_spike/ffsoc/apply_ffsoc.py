"""Apply the R2 instrumentation (timers + dumps, no arithmetic) to a SCRATCH copy of SOCRATES and of the model Makefile.
usage: python apply_ffsoc.py SCRATCH_SOCRATES_DIR SCRATCH_MODEL_SOCRATES_MAKEFILE    (decision D3; never run on the original tree)"""
import os, sys, shutil
sc, mk = sys.argv[1], sys.argv[2]
assert "ModelE_Support" not in sc, "refusing to touch the original tree"
rc = os.path.join(sc, "src", "radiance_core")
def edit(path, pairs):
    s = open(path).read()
    for a, b, cnt in pairs:
        assert s.count(a) == cnt, (path, a[:60], s.count(a))
        s = s.replace(a, b)
    open(path, "w").write(s)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffsoc_mod.F90"), os.path.join(rc, "ffsoc_mod.F90"))
HOOK = "  IF (lhook) CALL dr_hook(RoutineName,zhook_in,zhook_handle)\n"
edit(os.path.join(rc, "radiance_calc.F90"), [
 ("  USE def_ss_prop\n", "  USE def_ss_prop\n  USE ffsoc_mod\n", 1),
 (HOOK, HOOK + """  CALL ffsoc_begin_call
  CALL ffsoc_tic(6)
  CALL ffsoc_arr(0, control%isolir, 0, 1, atm%p)
  CALL ffsoc_arr(0, control%isolir, 0, 2, atm%t)
  CALL ffsoc_arr(0, control%isolir, 0, 3, atm%density)
  CALL ffsoc_arr(0, control%isolir, 0, 4, atm%gas_mix_ratio)
  CALL ffsoc_arr(0, control%isolir, 0, 5, atm%mass)
  CALL ffsoc_arr(0, control%isolir, 0, 6, atm%p_level)
  CALL ffsoc_arr(0, control%isolir, 0, 7, atm%t_level)
  IF (ALLOCATED(aer%mix_ratio)) CALL ffsoc_arr(0, control%isolir, 0, 8, aer%mix_ratio)
  IF (ALLOCATED(aer%mean_rel_humidity)) CALL ffsoc_arr(0, control%isolir, 0, 9, aer%mean_rel_humidity)
  IF (ALLOCATED(cld%w_cloud)) CALL ffsoc_arr(0, control%isolir, 0, 10, cld%w_cloud)
  IF (ALLOCATED(cld%frac_cloud)) CALL ffsoc_arr(0, control%isolir, 0, 11, cld%frac_cloud)
  IF (ALLOCATED(cld%condensed_mix_ratio)) CALL ffsoc_arr(0, control%isolir, 0, 12, cld%condensed_mix_ratio)
  IF (ALLOCATED(cld%condensed_dim_char)) CALL ffsoc_arr(0, control%isolir, 0, 13, cld%condensed_dim_char)
  IF (ALLOCATED(cld%condensed_rel_var_dens)) CALL ffsoc_arr(0, control%isolir, 0, 14, cld%condensed_rel_var_dens)
  CALL ffsoc_arr(0, control%isolir, 0, 15, bound%zen_0)
  CALL ffsoc_arr(0, control%isolir, 0, 16, bound%solar_irrad)
  CALL ffsoc_arr(0, control%isolir, 0, 17, bound%rho_alb)
  CALL ffsoc_arr(0, control%isolir, 0, 18, bound%t_ground)
  IF (ALLOCATED(bound%orog_corr)) CALL ffsoc_arr(0, control%isolir, 0, 19, bound%orog_corr)
  IF (ALLOCATED(cld%i_condensed_param)) CALL ffsoc_arr(0, control%isolir, 0, 21, REAL(cld%i_condensed_param, 8))
  IF (ALLOCATED(cld%condensed_n_phf)) CALL ffsoc_arr(0, control%isolir, 0, 22, REAL(cld%condensed_n_phf, 8))
  IF (ALLOCATED(cld%condensed_param_list)) CALL ffsoc_arr(0, control%isolir, 0, 23, cld%condensed_param_list)
  IF (ALLOCATED(cld%type_condensed)) CALL ffsoc_arr(0, control%isolir, 0, 24, REAL(cld%type_condensed, 8))
  IF (ALLOCATED(aer%mr_source)) CALL ffsoc_arr(0, control%isolir, 0, 25, REAL(aer%mr_source, 8))
  IF (ALLOCATED(aer%mr_type_index)) CALL ffsoc_arr(0, control%isolir, 0, 26, REAL(aer%mr_type_index, 8))
  IF (ALLOCATED(aer%mode_mix_ratio)) CALL ffsoc_arr(0, control%isolir, 0, 27, aer%mode_mix_ratio)
  IF (ALLOCATED(aer%mode_absorption)) CALL ffsoc_arr(0, control%isolir, 0, 28, aer%mode_absorption)
  IF (ALLOCATED(aer%mode_scattering)) CALL ffsoc_arr(0, control%isolir, 0, 29, aer%mode_scattering)
  IF (ALLOCATED(aer%mode_asymmetry)) CALL ffsoc_arr(0, control%isolir, 0, 30, aer%mode_asymmetry)
  CALL ffsoc_arr(0, control%isolir, 0, 31, (/ REAL(cld%n_condensed, 8), REAL(cld%n_cloud_type, 8), REAL(aer%n_mode, 8), &
    REAL(control%i_cloud, 8), REAL(control%i_cloud_representation, 8) /))
  CALL ffsoc_arr(0, control%isolir, 0, 20, (/ MERGE(1.0d0, 0.0d0, control%l_noscal_tau), MERGE(1.0d0, 0.0d0, control%l_orog), &
    MERGE(1.0d0, 0.0d0, control%l_rescale), MERGE(1.0d0, 0.0d0, control%l_cloud), MERGE(1.0d0, 0.0d0, control%l_aerosol), &
    MERGE(1.0d0, 0.0d0, control%l_gas), MERGE(1.0d0, 0.0d0, control%l_cont_gen) /))
  IF (ffsoc_on .AND. .NOT. ffsoc_spec_done(control%isolir)) THEN
    ffsoc_spec_done(control%isolir) = .TRUE.
    ffsoc_dump_now = .TRUE.
    CALL ffsoc_arr(100, control%isolir, 0, 1, spectrum%gas%k_lookup)
    IF (ALLOCATED(spectrum%gas%k_lookup_sb)) CALL ffsoc_arr(100, control%isolir, 0, 2, spectrum%gas%k_lookup_sb)
    CALL ffsoc_arr(100, control%isolir, 0, 3, spectrum%gas%p_lookup)
    CALL ffsoc_arr(100, control%isolir, 0, 4, spectrum%gas%t_lookup)
    CALL ffsoc_arr(100, control%isolir, 0, 5, spectrum%gas%gf_lookup)
    CALL ffsoc_arr(100, control%isolir, 0, 6, (/ REAL(spectrum%gas%n_gas_frac, 8) /))
    CALL ffsoc_arr(100, control%isolir, 0, 7, spectrum%gas%k)
    CALL ffsoc_arr(100, control%isolir, 0, 8, spectrum%gas%w)
    CALL ffsoc_arr(100, control%isolir, 0, 9, REAL(spectrum%gas%i_scale_fnc, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 10, REAL(spectrum%gas%i_scale_k, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 11, spectrum%gas%p_ref)
    CALL ffsoc_arr(100, control%isolir, 0, 12, spectrum%gas%t_ref)
    CALL ffsoc_arr(100, control%isolir, 0, 13, spectrum%gas%scale)
    CALL ffsoc_arr(100, control%isolir, 0, 14, MERGE(1.0d0, 0.0d0, spectrum%gas%l_doppler))
    CALL ffsoc_arr(100, control%isolir, 0, 15, spectrum%gas%doppler_cor)
    CALL ffsoc_arr(100, control%isolir, 0, 16, REAL(spectrum%gas%i_band_k, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 17, REAL(spectrum%gas%index_absorb, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 18, REAL(spectrum%gas%n_band_absorb, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 19, REAL(spectrum%gas%i_scat, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 20, REAL(spectrum%gas%type_absorb, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 21, MERGE(1.0d0, 0.0d0, spectrum%gas%l_self_broadening))
    CALL ffsoc_arr(100, control%isolir, 0, 22, REAL(spectrum%gas%index_sb, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 30, REAL(spectrum%contgen%n_band_cont, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 31, REAL(spectrum%contgen%index_cont, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 32, REAL(spectrum%contgen%i_band_k_cont, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 33, spectrum%contgen%k_lookup_cont)
    CALL ffsoc_arr(100, control%isolir, 0, 34, spectrum%contgen%t_lookup_cont)
    CALL ffsoc_arr(100, control%isolir, 0, 35, spectrum%contgen%w_cont)
    CALL ffsoc_arr(100, control%isolir, 0, 36, REAL(spectrum%contgen%index_cont_gas_1, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 37, REAL(spectrum%contgen%index_cont_gas_2, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 38, REAL(spectrum%contgen%i_cont_overlap_band, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 39, MERGE(1.0d0, 0.0d0, spectrum%contgen%l_cont_major))
    CALL ffsoc_arr(100, control%isolir, 0, 40, REAL(spectrum%contgen%i_scat_cont, 8))
    CALL ffsoc_arr(100, control%isolir, 0, 41, spectrum%planck%thermal_coeff)
    CALL ffsoc_arr(100, control%isolir, 0, 42, spectrum%solar%solar_flux_band)
    CALL ffsoc_arr(100, control%isolir, 0, 43, (/ REAL(spectrum%planck%n_deg_fit, 8), MERGE(1.0d0, 0.0d0, spectrum%planck%l_planck_tbl), &
      spectrum%planck%t_ref_planck /))
    IF (ALLOCATED(spectrum%planck%theta_planck_tbl)) CALL ffsoc_arr(100, control%isolir, 0, 44, spectrum%planck%theta_planck_tbl)
    CALL ffsoc_arr(100, control%isolir, 0, 50, (/ REAL(spectrum%rayleigh%i_rayleigh_scheme, 8), REAL(spectrum%rayleigh%n_gas_rayleigh, 8) /))
    IF (ALLOCATED(spectrum%rayleigh%index_rayleigh)) CALL ffsoc_arr(100, control%isolir, 0, 51, REAL(spectrum%rayleigh%index_rayleigh, 8))
    IF (ALLOCATED(spectrum%rayleigh%rayleigh_coeff)) CALL ffsoc_arr(100, control%isolir, 0, 52, spectrum%rayleigh%rayleigh_coeff)
    IF (ALLOCATED(spectrum%rayleigh%rayleigh_coeff_gas)) CALL ffsoc_arr(100, control%isolir, 0, 53, spectrum%rayleigh%rayleigh_coeff_gas)
    IF (ALLOCATED(spectrum%cont%n_band_continuum)) CALL ffsoc_arr(100, control%isolir, 0, 60, REAL(spectrum%cont%n_band_continuum, 8))
    IF (ALLOCATED(spectrum%cont%index_continuum)) CALL ffsoc_arr(100, control%isolir, 0, 61, REAL(spectrum%cont%index_continuum, 8))
    IF (ALLOCATED(spectrum%cont%i_scale_fnc_cont)) CALL ffsoc_arr(100, control%isolir, 0, 62, REAL(spectrum%cont%i_scale_fnc_cont, 8))
    IF (ALLOCATED(spectrum%cont%k_cont)) CALL ffsoc_arr(100, control%isolir, 0, 63, spectrum%cont%k_cont)
    IF (ALLOCATED(spectrum%cont%scale_cont)) CALL ffsoc_arr(100, control%isolir, 0, 64, spectrum%cont%scale_cont)
    IF (ALLOCATED(spectrum%cont%p_ref_cont)) CALL ffsoc_arr(100, control%isolir, 0, 65, spectrum%cont%p_ref_cont)
    IF (ALLOCATED(spectrum%cont%t_ref_cont)) CALL ffsoc_arr(100, control%isolir, 0, 66, spectrum%cont%t_ref_cont)
    CALL ffsoc_arr(100, control%isolir, 0, 70, (/ REAL(spectrum%aerosol%n_aerosol, 8), REAL(spectrum%aerosol%n_aerosol_mr, 8) /))
    IF (ALLOCATED(spectrum%aerosol%i_aerosol_parm)) CALL ffsoc_arr(100, control%isolir, 0, 71, REAL(spectrum%aerosol%i_aerosol_parm, 8))
    IF (ALLOCATED(spectrum%aerosol%type_aerosol)) CALL ffsoc_arr(100, control%isolir, 0, 72, REAL(spectrum%aerosol%type_aerosol, 8))
    IF (ALLOCATED(spectrum%aerosol%nhumidity)) CALL ffsoc_arr(100, control%isolir, 0, 73, REAL(spectrum%aerosol%nhumidity, 8))
    IF (ALLOCATED(spectrum%aerosol%humidities)) CALL ffsoc_arr(100, control%isolir, 0, 74, spectrum%aerosol%humidities)
    IF (ALLOCATED(spectrum%aerosol%abs)) CALL ffsoc_arr(100, control%isolir, 0, 75, spectrum%aerosol%abs)
    IF (ALLOCATED(spectrum%aerosol%scat)) CALL ffsoc_arr(100, control%isolir, 0, 76, spectrum%aerosol%scat)
    IF (ALLOCATED(spectrum%aerosol%phf_fnc)) CALL ffsoc_arr(100, control%isolir, 0, 77, spectrum%aerosol%phf_fnc)
    ffsoc_dump_now = .FALSE.
  END IF
""", 1),
 ("! DEPENDS ON: grey_opt_prop\n", "    CALL ffsoc_tic(1)\n! DEPENDS ON: grey_opt_prop\n", 1),
 ("      , dimen%nd_opt_level_aerosol_prsc, dimen%nd_opt_level_cloud_prsc         &\n      )\n",
  "      , dimen%nd_opt_level_aerosol_prsc, dimen%nd_opt_level_cloud_prsc         &\n      )\n"
  "    CALL ffsoc_toc(1)\n"
  "    CALL ffsoc_arr(10, control%isolir, i_band, 1, ss_prop%k_grey_tot_clr)\n"
  "    CALL ffsoc_arr(10, control%isolir, i_band, 2, ss_prop%k_ext_scat_clr)\n"
  "    CALL ffsoc_arr(10, control%isolir, i_band, 3, ss_prop%k_grey_tot)\n"
  "    CALL ffsoc_arr(10, control%isolir, i_band, 4, ss_prop%k_ext_scat)\n", 1),
 ("  DO i_band=control%first_band, control%last_band\n", "  DO i_band=control%first_band, control%last_band\n    CALL ffsoc_tic(2)\n", 1),
 ("!   Set band-by-band flux diagnostics\n",
  "    CALL ffsoc_toc(2)\n"
  "    CALL ffsoc_arr(30, control%isolir, i_band, 1, radout%flux_down(:,:,control%map_channel(i_band)))\n"
  "    CALL ffsoc_arr(30, control%isolir, i_band, 2, radout%flux_up(:,:,control%map_channel(i_band)))\n"
  "!   Set band-by-band flux diagnostics\n", 1),
 ("    IF (control%isolir == ip_infra_red) THEN\n\n!     Calculate the change in the thermal source function\n",
  "    CALL ffsoc_arr(26, control%isolir, i_band, 1, k_esft_layer(1:atm%n_profile, 1:atm%n_layer, :, :))\n"
  "    CALL ffsoc_arr(26, control%isolir, i_band, 2, k_esft_cont_layer(1:atm%n_profile, 1:atm%n_layer, :, :))\n"
  "    CALL ffsoc_arr(26, control%isolir, i_band, 3, gas_frac_rescaled(1:atm%n_profile, 1:atm%n_layer, :))\n"
  "    IF (control%isolir == ip_infra_red) THEN\n\n!     Calculate the change in the thermal source function\n", 1),
 ("  9999 CONTINUE\n", "  9999 CONTINUE\n  CALL ffsoc_toc(6)\n  CALL ffsoc_end_call(control%isolir, control%last_band)\n", 1),
])
edit(os.path.join(rc, "solve_band_k_eqv_scl.F90"), [
 ("  USE def_ss_prop\n", "  USE def_ss_prop\n  USE ffsoc_mod\n", 1),
 (HOOK, HOOK + "  CALL ffsoc_tic(3)\n", 1),
 ("! The ESFT terms for the major gas in the band are used with\n",
  "  CALL ffsoc_toc(3)\n"
  "  CALL ffsoc_arr(20, isolir, 0, 1, k_eqv(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(20, isolir, 0, 2, d_mass(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(20, isolir, 0, 3, zen_0(1:n_profile))\n"
  "! The ESFT terms for the major gas in the band are used with\n", 1),
 ("    IF (i_cloud == ip_cloud_mcica) THEN\n\n! DEPENDS ON: mcica_sample\n",
  "    CALL ffsoc_tic(4)\n    IF (i_cloud == ip_cloud_mcica) THEN\n\n! DEPENDS ON: mcica_sample\n", 1),
 ("!   Increment the fluxes within the band.\n    weight_incr=weight_band*esft_weight\n",
  "    CALL ffsoc_toc(4)\n"
  "    CALL ffsoc_arr(21, isolir, 0, iex, flux_total_part(1:n_profile, 1:2*n_layer+2))\n"
  "    CALL ffsoc_arr(22, isolir, 0, iex, (/ esft_weight /))\n"
  "!   Increment the fluxes within the band.\n    weight_incr=weight_band*esft_weight\n", 1),
 ("! DEPENDS ON: augment_radiance\n", "    CALL ffsoc_tic(5)\n! DEPENDS ON: augment_radiance\n", 1),
 ("    l_initial=.FALSE.\n", "    CALL ffsoc_toc(5)\n    l_initial=.FALSE.\n", 1),
])
ZIN = "  IF (lhook) CALL dr_hook(RoutineName,zhook_in,zhook_handle)\n"
ZOUT = "  IF (lhook) CALL dr_hook(RoutineName,zhook_out,zhook_handle)\n"
for fn, st in (("solve_band_ses", 7), ("solve_band_one_gas", 8), ("solve_band_without_gas", 9)):
    edit(os.path.join(rc, fn + ".F90"), [
     ("  USE def_ss_prop\n", "  USE def_ss_prop\n  USE ffsoc_mod\n", 1),
     (ZIN, ZIN + "  CALL ffsoc_tic(%d)\n" % st, 1),
     (ZOUT, "  CALL ffsoc_toc(%d)\n" % st + ZOUT, 1)])
edit(os.path.join(rc, "solve_band_k_eqv_scl.F90"), [
 ("  CALL ffsoc_tic(3)\n", "  CALL ffsoc_tic(3)\n  CALL ffsoc_tic(10)\n", 1),
 ("  i_abs=index_abs(1)\n",
  "  i_abs=index_abs(1)\n"
  "  CALL ffsoc_arr(25, isolir, 0, 1, k_abs_layer(1:n_profile, 1:n_layer, :, :))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 2, w_abs_esft(:, :))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 3, REAL(n_abs_esft(:), 8))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 4, (/ REAL(n_abs, 8), REAL(index_abs(1:n_abs), 8) /))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 5, planck_flux_band(1:n_profile, 0:n_layer))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 6, diff_planck_band(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 7, diff_planck_band_2(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 8, planck_flux_surface(1:n_profile))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 9, solar_irrad(1:n_profile))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 10, rho_alb(1:n_profile, :))\n"
  "  CALL ffsoc_arr(25, isolir, 0, 11, (/ weight_band /))\n", 1),
 (ZOUT, "  CALL ffsoc_toc(10)\n" + ZOUT, 1)])
# per-k-term two-stream coefficients, sources and solver result of the triple solver (tags 40-43; band/k are not visible here: file order
# links them to the surrounding tag 10 / 20 (band) and 21 (k-term, written after monochromatic_radiance returns))
edit(os.path.join(rc, "triple_column.F90"), [
 ("  USE def_ss_prop\n", "  USE def_ss_prop\n  USE ffsoc_mod\n", 1),
 ("  IF (isolir == ip_infra_red) THEN\n\n    DO k=1, n_region\n",
  "  CALL ffsoc_arr(40, isolir, 0, 1, ss_prop%tau_clr)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 2, ss_prop%omega_clr)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 3, ss_prop%tau)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 4, ss_prop%omega)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 5, ss_prop%tau_clr_noscal)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 6, ss_prop%tau_noscal)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 7, ss_prop%phase_fnc_clr)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 8, ss_prop%phase_fnc)\n"
  "  CALL ffsoc_arr(40, isolir, 0, 9, sec_0(1:n_profile))\n"
  "  CALL ffsoc_arr(44, isolir, 0, 1, (/ REAL(n_cloud_top, 8), REAL(n_region, 8), REAL(n_cloud_type, 8), REAL(n_layer, 8) /))\n"
  "  CALL ffsoc_arr(45, isolir, 0, 1, frac_cloud(1:n_profile, id_ct:n_layer, 1:n_cloud_type))\n"
  "  CALL ffsoc_arr(45, isolir, 0, 2, frac_region(1:n_profile, id_ct:n_layer, 1:n_region))\n"
  "  CALL ffsoc_arr(45, isolir, 0, 3, REAL(i_region_cloud(1:n_cloud_type), 8))\n"
  "  CALL ffsoc_arr(46, isolir, 0, 1, cloud_overlap(1:n_profile, id_ct-1:n_layer, 1:18))\n"
  "  CALL ffsoc_arr(46, isolir, 0, 2, flux_inc_down(1:n_profile))\n"
  "  CALL ffsoc_arr(46, isolir, 0, 3, diffuse_albedo(1:n_profile))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 1, diff_planck(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 2, diff_planck_2(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 3, w_free(1:n_profile, id_ct:n_layer))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 4, w_cloud(1:n_profile, id_ct:n_layer))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 5, adjust_solar_ke(1:n_profile, 1:n_layer))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 6, flux_inc_direct(1:n_profile))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 7, direct_albedo(1:n_profile))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 8, d_planck_flux_surface(1:n_profile))\n"
  "  CALL ffsoc_arr(47, isolir, 0, 9, (/ MERGE(1.0d0, 0.0d0, l_scale_solar) /))\n"
  "  CALL ffsoc_arr(49, isolir, 0, 1, ss_prop%forward_scatter_clr)\n"
  "  CALL ffsoc_arr(49, isolir, 0, 2, ss_prop%forward_scatter)\n"
  "  CALL ffsoc_arr(41, isolir, 0, 1, trans(1:n_profile, 1:n_layer, :))\n"
  "  CALL ffsoc_arr(41, isolir, 0, 2, reflect(1:n_profile, 1:n_layer, :))\n"
  "  CALL ffsoc_arr(41, isolir, 0, 3, trans_0_noscal(1:n_profile, 1:n_layer, :))\n"
  "  CALL ffsoc_arr(41, isolir, 0, 4, trans_0(1:n_profile, 1:n_layer, :))\n"
  "  CALL ffsoc_arr(41, isolir, 0, 5, source_coeff(1:n_profile, 1:n_layer, :, :))\n"
  "  IF (isolir == ip_infra_red) THEN\n\n    DO k=1, n_region\n", 1),
 ("\n\n\n  SELECT CASE (i_solver)\n",
  "\n  CALL ffsoc_arr(42, isolir, 0, 1, s_down(1:n_profile, 1:n_layer, :))\n"
  "  CALL ffsoc_arr(42, isolir, 0, 2, s_up(1:n_profile, 1:n_layer, :))\n"
  "  CALL ffsoc_arr(42, isolir, 0, 3, source_flux_ground(1:n_profile, :))\n"
  "\n\n  SELECT CASE (i_solver)\n", 1),
 ("  END SELECT\n\n\n\n  IF (l_clear) THEN\n",
  "  END SELECT\n  CALL ffsoc_arr(43, isolir, 0, 1, flux_total(1:n_profile, 1:2*n_layer+2))\n  CALL ffsoc_arr(48, isolir, 0, 1, flux_direct(1:n_profile, 0:n_layer))\n\n\n  IF (l_clear) THEN\n", 1),
])
edit(mk, [("\tdef_planetstr.F90\n", "\tdef_planetstr.F90 \\\n\tffsoc_mod.F90\n", 1)])
print("instrumentation applied")
