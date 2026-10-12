"""Grey optical properties of one band for this rundeck (radiance_core/grey_opt_prop.F90 and callees), R4:
   Rayleigh (SW: ip_rayleigh_total, one coefficient per band) + modal aerosol (opt_prop_ukca_aerosol) + grey water-vapour self continuum
   (rescale_continuum, ip_scale_power_quad) -> k_grey_tot += k_ext_scat; CSIW cloud components (opt_prop_water_cloud ip_drop_pade_2,
   opt_prop_ice_cloud ip_ice_fu_phf, branch n_order_phase = 1 with l_rescale and n_order_forward = 2) with the Cairns inhomogeneity
   correction; normalisation by k_ext_scat.   Not ported: mcica, spherical-harmonic / solar phase function branches, the other
   cloud parametrisations (the dumps show types 5 and 12 only), classic (non-modal) aerosol (n_aerosol = 0 in the rundeck).
   Layout: (n_profile, n_layer[, n_type+1]); type 0 = cloud-free part of the layer."""
import numpy as np

TINY_K = np.finfo(np.float64).tiny
MOL_WEIGHT_AIR = 28.966e-03
REPSILON = 1.0 / (28.966 / 18.015)       # the constants module of this build uses 18.015 (found from the dumped k_grey_tot: ratio 1.000017 to 18.0153)


def continuum_amount_self(p, t, density, water, scale3, p_ref, t_ref):
    """rescale_continuum for i_continuum = ip_self_continuum, i_fnc = ip_scale_power_quad."""
    pwk = np.power(p / p_ref, scale3[0])
    tt = t / t_ref - 1.0
    amount = pwk * (1.0 + scale3[1] * tt + scale3[2] * tt ** 2)
    molar_density_water = density * water / (REPSILON * MOL_WEIGHT_AIR)
    return amount * molar_density_water * water


def cloud_component(kind, lwc, r, param, rel_var, l_cairns=True, n_order_forward=2):
    """Returns k_ext_tot, k_ext_scat, phase(1), forward for one condensed component (n_profile, n_layer arrays);
    kind 'water' = ip_drop_pade_2 (16 parameters), 'ice' = ip_ice_fu_phf (14 parameters)."""
    if kind == "water":
        p = param
        k_ext = lwc * (p[0] + r * (p[1] + r * p[2])) / (1.0 + r * (p[3] + r * (p[4] + r * p[5])))
        k_sc = k_ext * (1.0 - (p[6] + r * (p[7] + r * p[8])) / (1.0 + r * (p[9] + r * p[10])))
        g = (p[11] + r * (p[12] + r * p[13])) / (1.0 + r * (p[14] + r * p[15]))
    else:
        p = param
        x = p[3] / r
        k_ext = lwc * ((p[2] * x + p[1]) * x + p[0])
        x = r / p[8]
        k_sc = k_ext * (1.0 - (p[4] + x * (p[5] + x * (p[6] + x * p[7]))))
        x = r / p[13]
        g = p[9] + x * (p[10] + x * (p[11] + x * p[12]))
    ph = k_sc * g
    fwd = ph * g
    if l_cairns:
        corr = 1.0 / (1.0 + rel_var)
        ok = k_sc > TINY_K
        with np.errstate(all="ignore"):
            asym = ph / k_sc
            omega = k_sc / k_ext
            om_corr = 1.0 / (1.0 + rel_var * (1.0 - omega))
            as_corr = (1.0 + rel_var * (1.0 - omega)) / (1.0 + rel_var * (1.0 - omega * asym))
            k_sc2 = k_sc * corr * om_corr
            ph2 = ph * corr * om_corr * as_corr
            fwd2 = fwd * corr * om_corr * as_corr ** n_order_forward
        k_sc, ph, fwd = np.where(ok, k_sc2, k_sc), np.where(ok, ph2, ph), np.where(ok, fwd2, fwd)
        k_ext = k_ext * corr
    return k_ext, k_sc, ph, fwd


def grey_props(iso, rayleigh_coeff, mix_modes, abs_modes, scat_modes, asym_modes, n_mode, k_cont, amount_cont, comps, w_cloud,
               n_type=4, n_order_forward=2):
    """comps: list of (cloud_type 1..4, kind, lwc, r, param, rel_var).  Returns dict(kg0, kes0, ph0, fwd0, kg (.., n_type+1), kes, ph, fwd)
    after normalisation (as stored in ss_prop before rescale_phase_fnc)."""
    n_prof, nl = w_cloud.shape
    kes = np.full((n_prof, nl), rayleigh_coeff if (iso == 1) else 0.0)
    kg = np.zeros((n_prof, nl)); ph = np.zeros((n_prof, nl)); fwd = np.zeros((n_prof, nl))
    for j in range(n_mode):                                   # opt_prop_ukca_aerosol
        mix, a, s, g = mix_modes[:, :, j], abs_modes[:, :, j], scat_modes[:, :, j], asym_modes[:, :, j]
        kg = kg + mix * a
        ks = mix * s
        kes = kes + ks
        ph = ph + ks * g
        ks_phf = ks * g
        fwd = fwd + (ks_phf if n_order_forward == 1 else ks_phf * g)
    for kc, am in zip(k_cont, amount_cont):                   # grey continuum
        kg = kg + kc * am
    kg = kg + kes
    types = {0: (kg, kes, ph, fwd)}
    cl = w_cloud > 0.0                                        # n_cloud_profile / i_cloud_profile
    for k in range(1, n_type + 1):
        kgk, kek, phk, fwk = [a.copy() for a in (kg, kes, ph, fwd)]
        for (ct, kind, lwc, r, param, rv) in comps:
            if ct != k:
                continue
            ext, sc, p1, f1 = cloud_component(kind, lwc, r, param, rv, n_order_forward=n_order_forward)
            kgk = np.where(cl, kgk + ext, kgk); kek = np.where(cl, kek + sc, kek)
            phk = np.where(cl, phk + p1, phk); fwk = np.where(cl, fwk + f1, fwk)
        types[k] = (kgk, kek, phk, fwk)
    out = {"kg": [], "kes": [], "ph": [], "fwd": []}
    for k in range(0, n_type + 1):
        kgk, kek, phk, fwk = types[k]
        ok = kek > TINY_K
        with np.errstate(all="ignore"):
            inv = 1.0 / kek
        phn = np.where(ok, phk * inv, phk); fwn = np.where(ok, fwk * inv, fwk)
        out["kg"].append(kgk); out["kes"].append(kek); out["ph"].append(phn); out["fwd"].append(fwn)
    return {k: np.stack(v, axis=2) for k, v in out.items()}
