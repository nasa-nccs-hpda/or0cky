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
