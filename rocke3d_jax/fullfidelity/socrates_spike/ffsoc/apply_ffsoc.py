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
edit(mk, [("\tdef_planetstr.F90\n", "\tdef_planetstr.F90 \\\n\tffsoc_mod.F90\n", 1)])
print("instrumentation applied")
