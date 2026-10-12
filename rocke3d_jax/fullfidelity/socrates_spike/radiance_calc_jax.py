"""radiance_calc for this rundeck, from the model-side inputs (atmosphere, aerosol modes, cloud condensate, surface) and the spectral data to
band fluxes, assembled from the validated pieces (gas_absorption, k_eqv, grey_opt_prop, cloud_geometry, planck, band_chain).
Validation status and the parts that are still taken from the model side: see Reports/SOCRATES_R2_DUMPS.md section 10.
One column per call in the validation (the Fortran server calls radiance_calc once per profile); the functions are vectorised over profiles
except `major_gas_order` (the Fortran uses the last profile of the call) and n_cloud_top (one value per call)."""
import numpy as np
import grey_opt_prop as G
import planck as PL
import band_chain as BC


def band_inputs(spec, iso, ib, r, planck_surface_override=None):
    """Assemble the band dict of band_chain.run_band from the call inputs r (tag-0 arrays, see r2_band_check for the keys)."""
    n_mode = int(r[31][2])
    kc, am = [], []
    for j in range(int(spec["n_band_continuum"][ib - 1])):
        ic = int(spec["index_continuum"][ib - 1, j])
        assert ic == 1 and int(spec["i_scale_fnc_cont"][ib - 1, ic - 1]) == 2, "only the self continuum with power-quad scaling is ported"
        kc.append(float(spec["k_cont"][ib - 1, ic - 1]))
        am.append(G.continuum_amount_self(r[1], r[2], r[3], r[4][:, :, 0], spec["scale_cont"][:3, ib - 1, ic - 1],
                                          float(spec["p_ref_cont"][ic - 1, ib - 1]), float(spec["t_ref_cont"][ic - 1, ib - 1])))
    comps = []
    for kk, (ty, ip) in enumerate(zip(r[24].astype(int), r[21].astype(int))):
        kind = "water" if ty in (1, 3) else "ice"
        assert ip == (5 if kind == "water" else 12)
        comps.append((kk + 1, kind, r[12][:, :, kk], r[13][:, :, kk], r[23][:16 if kind == "water" else 14, kk, ib - 1], r[14][:, :, kk]))
    rc = float(spec["rayleigh_coeff"][ib - 1]) if iso == 1 else 0.0
    g = G.grey_props(iso, rc, r[27], r[28][..., ib - 1], r[29][..., ib - 1], r[30][..., ib - 1], n_mode, kc, am, comps, r[10])
    phr = (g["ph"] - g["fwd"]) / (1.0 - g["fwd"])
    band = dict(k_grey_tot_clr=g["kg"][:, :, 0], k_ext_scat_clr=g["kes"][:, :, 0], k_grey_tot=g["kg"], k_ext_scat=g["kes"])
    phase = dict(ph_clr=phr[:, :, 0:1], ph=phr[:, :, None, :], fwd_clr=g["fwd"][:, :, 0], fwd=g["fwd"])
    if iso == 1:
        band["solar_irrad"] = r[16] * float(spec["solar_flux_band"][ib - 1])
        band["planck_flux_band"] = band["diff_planck_band"] = band["diff_planck_band_2"] = band["planck_flux_surface"] = None
    else:
        pf, dp, dp2, pg = [np.asarray(v) for v in PL.band_planck(spec, ib, r[7], r[2], r[18])]
        if 32 in r and r[32][0] > 0.5:                              # control%l_tile: surface emission from the tile temperatures
            pg = PL.surface_planck(spec, ib, r[18], dict(n_tile=int(r[32][1]), list_tile=r[33], frac_tile=r[34], t_tile=r[35]))
        band.update(planck_flux_band=pf, diff_planck_band=dp, diff_planck_band_2=dp2,
                    planck_flux_surface=pg if planck_surface_override is None else planck_surface_override, solar_irrad=None)
    return band, phase


def radiance_calc(spec, iso, r, planck_surface_fn=None, l_scale_solar=True, bands=None, nct=None):
    """r: dict of the tag-0 style inputs (keys 1 p, 2 t, 3 density, 4 gas_mix_ratio, 5 mass, 7 t_level, 10 w_cloud, 11 frac_cloud, 12-14 cloud
    condensate, 15 zen_0, 16 solar_irrad, 17 rho_alb(n, basis, band), 18 t_ground, 21-24, 27-31 cloud/aerosol parameters).
    Returns per band the (flux_down, flux_up, flux_direct) of the band (weight_band * sum over k-terms; IR without the Planck add-back
    done in `longwave_total`)."""
    n_band = spec["n_band_absorb"].shape[0]
    A = dict(p=r[1], t=r[2], mix=r[4], mass=r[5], w_cloud=r[10], frac_cloud=r[11], zen_0=r[15])
    res = []
    for ib in (bands or range(1, n_band + 1)):
        band, phase = band_inputs(spec, iso, ib, r, None if planck_surface_fn is None else planck_surface_fn(ib))
        alb = dict(diffuse=r[17][:, 0, ib - 1], direct=r[17][:, 1, ib - 1])
        out = BC.run_band(spec, iso, ib, A, band, [phase] * 100, alb, l_scale_solar=l_scale_solar, nct=nct)
        wb = 1.0
        fu = fd = fdir = None
        for o in out:
            wi = wb * o["weight"]
            ft = o["flux_total"]
            nl1 = ft.shape[1] // 2
            up, dn = wi * ft[:, 0::2][:, :nl1], wi * ft[:, 1::2][:, :nl1]
            fu = up if fu is None else fu + up
            fd = dn if fd is None else fd + dn
            if o["flux_direct"] is not None:
                d = wi * o["flux_direct"]
                fdir = d if fdir is None else fdir + d
        if iso == 2:
            fu = fu + band["planck_flux_band"]; fd = fd + band["planck_flux_band"]
        res.append(dict(band=ib, flux_down=fd, flux_up=fu, flux_direct=fdir))
    return res
