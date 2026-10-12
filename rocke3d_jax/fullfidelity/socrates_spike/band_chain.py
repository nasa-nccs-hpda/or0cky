"""One band of radiance_calc for this rundeck as a chain of the validated pieces (R3-R5):
   k_abs_layer (gas_absorption) -> k_eqv / adjust_solar_ke (k_eqv) -> per k-term tau, omega (kterm) -> region two-stream coefficients
   (kernels.two_coeff + cloud-region blend, as r2_kernel_b_check.region_coeffs) -> sources and Hogan solver (triple_column).
Inputs that are NOT yet ported are passed in: grey optical properties of the band (tag 10), phase function / forward scatter of the band,
Planck terms of the band, boundary albedos, solar irradiance.  One column per call (the dumps hold one profile per radiance_calc call) but all
code is vectorised over profiles."""
import numpy as np
import jax.numpy as jnp
import kernels as K
import gas_absorption as GA
import k_eqv as KE
import kterm as KT
import triple_column as TC

TINY = np.finfo(np.float64).tiny


def region_coeffs_arrays(tau_clr, om_clr, tau, om, ph_clr, ph, sec0, nct, nty, fc, fr, irc, solar, scheme):
    """As r2_kernel_b_check.region_coeffs but from arrays.  Shapes: (n_prof, nl), (n_prof, nl, nty+1), phase (n_prof, nl, 1[, nty+1])."""
    nl = tau_clr.shape[1]
    names = ["trans", "reflect"] + (["trans_0", "s_up", "s_down"] if solar else ["ir_1d", "ir_2d"])
    n_prof = tau.shape[0]
    res = {n: np.zeros((n_prof, nl, 3)) for n in names}

    def coef(t, o, g):
        return {k: np.asarray(v) for k, v in K.two_coeff(t, t, o, g, scheme, solar, sec0 if solar else None, True).items()}
    i0 = nct - 1
    if i0 > 0:
        c1 = coef(tau_clr[:, :i0], om_clr[:, :i0], ph_clr[:, :i0, 0])
    c2 = coef(tau[:, i0:, 0], om[:, i0:, 0], ph[:, i0:, 0, 0])
    for n in names:
        if i0 > 0:
            res[n][:, :i0, 0] = c1[n]
        res[n][:, i0:, 0] = c2[n]
    for ik in range(1, nty + 1):
        r = int(irc[ik - 1]) - 1
        ck = coef(tau[:, i0:, ik], om[:, i0:, ik], ph[:, i0:, 0, ik])
        f = fc[:, i0:, ik - 1]
        use = f > TINY
        for n in names:
            res[n][:, i0:, r] += np.where(use, f * ck[n], 0.0)
    for r in range(1, 3):
        f = fr[:, i0:, r]
        use = f > TINY
        for n in names:
            res[n][:, i0:, r] = np.where(use, res[n][:, i0:, r] / np.where(use, f, 1.0), 0.0)
    return res


def run_band(spec, iso, i_band, atm, band, kt_phase, albedo_cols, l_scale_solar=True, nct=None):
    """Returns a list over k-terms of dict(flux_total, flux_direct) for one band (n_profile columns).
    atm: dict with p, t, mix, mass, w_cloud, frac_cloud, zen_0 (tag 0 arrays);  band: dict with grey properties (k_grey_tot_clr,
    k_ext_scat_clr, k_grey_tot, k_ext_scat), planck_flux_band, diff_planck_band, diff_planck_band_2, planck_flux_surface, solar_irrad,
    rho_alb;  kt_phase: list per k-term of dict(ph_clr, ph, fwd_clr, fwd) (post-rescale phase functions: from the dumps, not ported)."""
    solar = iso == 1
    p, t, mix, mass = [jnp.asarray(atm[k]) for k in ("p", "t", "mix", "mass")]
    order = GA.major_gas_order(spec, i_band, np.asarray(atm["mix"])[-1], np.asarray(atm["mass"])[-1])
    kabs0, nes0, wes0, gidx0, _ = GA.band_absorption(spec, i_band, p, t, mix)
    pos = {int(g): j for j, g in enumerate(gidx0)}
    sel = [pos[g] for g in order]
    kabs, nes, wes = kabs0[:, :, :, sel], nes0[sel], wes0[:, sel]
    n_abs = len(order)
    idx = np.arange(1, n_abs + 1)
    d_mass = mass
    zen0 = jnp.asarray(atm["zen_0"])
    wcl, fcl = atm["w_cloud"], atm["frac_cloud"]
    nct = nct or __import__("cloud_geometry").set_cloud_top(wcl)
    import cloud_geometry as CG
    fr = CG.aggregate_csiw(fcl)
    co, w_free, _ = CG.overlap_coupled_triple(wcl, fr, nct)
    # k_eqv of the minor absorbers
    if solar:
        keq, adj = KE.k_eqv_solar(kabs, nes, wes, idx, n_abs, d_mass, zen0)
    else:
        keq = KE.k_eqv_ir(kabs, nes, wes, idx, n_abs, d_mass, jnp.asarray(band["planck_flux_band"]), jnp.asarray(band["diff_planck_band"]),
                          jnp.asarray(band["planck_flux_surface"]), jnp.asarray(albedo_cols["diffuse"]))
        adj = None
    gtc = np.array(band["k_grey_tot_clr"]); gt = np.array(band["k_grey_tot"])
    gtc[:, :nct - 1] += np.asarray(keq)[:, :nct - 1]
    gt[:, nct - 1:, :] += np.asarray(keq)[:, nct - 1:, None]
    nty = fcl.shape[2]
    scheme = K.IP_PIFM80 if solar else K.IP_ELSASSER
    irc = np.array([2, 2, 3, 3])
    out = []
    w_cloud_j = jnp.asarray(wcl)
    for iex in range(int(nes[0])):
        kg = kabs[:, :, iex, 0]
        ph = kt_phase[iex]
        tc, oc, tt, oo = [np.asarray(v) for v in KT.kterm_props(gtc, band["k_ext_scat_clr"], gt, band["k_ext_scat"], kg, d_mass, nct,
                                                              ph["fwd_clr"], ph["fwd"])]
        sec0 = np.asarray(zen0)
        cf = region_coeffs_arrays(tc, oc, tt, oo, ph["ph_clr"], ph["ph"], sec0, nct, nty, np.asarray(fcl), fr, irc, solar, scheme)
        trans, refl = cf["trans"], cf["reflect"]
        if solar:
            sc = np.stack([cf["s_up"], cf["s_down"]], axis=2)
            t0 = cf["trans_0"]
            finc = np.asarray(band["solar_irrad"]) / sec0
            r = TC.triple_column(1, nct, jnp.asarray(trans), jnp.asarray(refl), jnp.asarray(t0), jnp.asarray(sc), jnp.asarray(co),
                                 jnp.asarray(finc), jnp.asarray(albedo_cols["diffuse"]), flux_inc_direct=jnp.asarray(finc),
                                 adjust_solar_ke=adj, l_scale_solar=l_scale_solar, direct_albedo=jnp.asarray(albedo_cols["direct"]))
        else:
            sc = np.stack([cf["ir_1d"], cf["ir_2d"]], axis=2)
            pfb = np.asarray(band["planck_flux_band"])
            finc = -pfb[:, 0]
            dps = np.asarray(band["planck_flux_surface"]) - pfb[:, -1]
            r = TC.triple_column(2, nct, jnp.asarray(trans), jnp.asarray(refl), jnp.zeros_like(jnp.asarray(trans)), jnp.asarray(sc),
                                 jnp.asarray(co), jnp.asarray(finc), jnp.asarray(albedo_cols["diffuse"]),
                                 diff_planck=jnp.asarray(band["diff_planck_band"]), diff_planck_2=jnp.asarray(band["diff_planck_band_2"]),
                                 w_free=jnp.asarray(np.nan_to_num(w_free)), w_cloud=w_cloud_j, frac_region=jnp.asarray(fr),
                                 d_planck_flux_surface=jnp.asarray(dps))
        out.append(dict(flux_total=np.asarray(r["flux_total"]), flux_direct=None if r["flux_direct"] is None else np.asarray(r["flux_direct"]),
                        weight=float(wes[iex, 0])))
    return out
