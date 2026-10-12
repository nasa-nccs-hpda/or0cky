"""Planck source terms of a longwave band (diff_planck_source_tbl.F90 with interp1d_mod; l_planck_tbl = .TRUE. in the rundeck's LW file):
planck_flux_band(l, 0..n_layer), diff_planck_band, diff_planck_band_2 (l_ir_source_quad), planck_flux_surface.
Spectral data: theta_planck_tbl (500,), thermal_coeff (500, n_band), t_ref_planck from spec_data (tags 41, 43, 44 of the dump)."""
import numpy as np
import jax.numpy as jnp


def interp1d(x, y, xi):
    """interp1d_mod: piecewise linear, constant outside; x ascending (n,), xi any shape."""
    n = x.shape[0]
    xi = jnp.asarray(xi)
    i = jnp.clip(jnp.sum(x[None] < xi.reshape(-1)[:, None], axis=1), 1, n - 1)        # first index with xi <= x(i), 0-based >= 1
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    yi = (y1 - y0) / (x1 - x0) * (xi.reshape(-1) - x0) + y0
    yi = jnp.where(xi.reshape(-1) <= x[0], y[0], jnp.where(xi.reshape(-1) >= x[n - 1], y[n - 1], yi))
    return yi.reshape(xi.shape)


def band_planck(spec, i_band, t_level, t, t_ground):
    """t_level: (n_profile, n_layer+1); t: (n_profile, n_layer); t_ground: (n_profile,)."""
    theta = jnp.asarray(spec["theta_planck_tbl"])
    coeff = jnp.asarray(spec["thermal_coeff"][:, i_band - 1])
    t_ref = float(spec["planck_misc"][2])
    planck = lambda tmp: interp1d(theta, coeff, tmp / t_ref)
    pf = planck(t_level)                                       # (n_profile, n_layer+1)
    dp = pf[:, 1:] - pf[:, :-1]
    dp2 = 2.0 * (pf[:, 1:] + pf[:, :-1] - 2.0 * planck(t))
    return pf, dp, dp2, planck(t_ground)


def surface_planck(spec, i_band, t_ground, tile=None):
    """planck_flux_ground: planck(t_ground), or with tiles (control%l_tile) the fraction-weighted sum of planck(t_tile) for the
    profiles listed in list_tile (diff_planck_source_tbl.F90).  tile = dict(list_tile (n_point,) 1-based, frac_tile (n_point, n_tile),
    t_tile (n_point, n_tile), n_tile)."""
    theta = jnp.asarray(spec["theta_planck_tbl"])
    coeff = jnp.asarray(spec["thermal_coeff"][:, i_band - 1])
    t_ref = float(spec["planck_misc"][2])
    planck = lambda tmp: interp1d(theta, coeff, tmp / t_ref)
    out = np.array(planck(jnp.asarray(t_ground)))
    if tile is not None and int(tile["n_tile"]) > 0:
        nt = int(tile["n_tile"])
        pt = np.asarray(planck(jnp.asarray(tile["t_tile"][:, :nt])))
        for lp, l in enumerate(np.asarray(tile["list_tile"]).astype(int)):
            acc = tile["frac_tile"][lp, 0] * pt[lp, 0]
            for k in range(1, nt):
                acc = acc + tile["frac_tile"][lp, k] * pt[lp, k]
            out[l - 1] = acc
    return out
