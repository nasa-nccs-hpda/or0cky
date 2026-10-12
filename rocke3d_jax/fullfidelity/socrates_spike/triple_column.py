"""JAX/NumPy port of SOCRATES triple_column for this rundeck (radiance_core/triple_column.F90 and its callees ir_source and
triple_solar_source, then solver_triple_hogan.py), given the region two-stream coefficients.

Option set of the rundeck (R1 report): i_solver = triple_hogan, l_ir_source_quad = True, l_noscal_tau = False (inferred, see the R2
report section 8), l_orog = False (assumed; the check against tags 42/48 would show a difference), l_clear = False (no clear-sky diagnostics
column on the CSIW path).

Array layout as in solver_triple_hogan.py: layers 0-based columns, regions 0 = clear, 1 = stratiform, 2 = convective.
  trans, reflect, trans_0:      (n_profile, n_layer, 3)
  source_coeff:                 (n_profile, n_layer, 2, 3)   [:, :, 0] = ir_1d / solar_up, [:, :, 1] = ir_2d / solar_down
  co:                           (n_profile, n_layer+1, 18)   cloud_overlap(:, id_ct-1:n_layer, 1..18), id_ct = 1
Layers above n_cloud_top use only the clear region (index 0); entries of the other regions there are never read."""
import numpy as np
import jax.numpy as jnp
from solver_triple_hogan import solver_triple_hogan

SOLAR, IR = 1, 2


def ir_sources(sc, diff_planck, diff_planck_2, w_free, w_cloud, frac_region, n_cloud_top):
    """ir_source for the three regions followed by the weighting (clear: w_free, cloudy: w_cloud*frac_region).
    w_free, w_cloud, frac_region: (n_profile, n_layer) (id_ct = 1)."""
    n_layer = sc.shape[1]
    s_up = sc[:, :, 0, :] * diff_planck[:, :, None] + sc[:, :, 1, :] * diff_planck_2[:, :, None]
    s_down = -sc[:, :, 0, :] * diff_planck[:, :, None] + sc[:, :, 1, :] * diff_planck_2[:, :, None]
    lay = np.arange(n_layer) >= n_cloud_top - 1
    wgt = jnp.stack([w_free, w_cloud * frac_region[:, :, 1], w_cloud * frac_region[:, :, 2]], axis=2)
    keep = jnp.asarray(lay)[None, :, None]
    s_up = jnp.where(keep, wgt * s_up, s_up)
    s_down = jnp.where(keep, wgt * s_down, s_down)
    return s_down, s_up


def solar_sources(sc, trans_0, flux_inc_direct, adjust_solar_ke, l_scale_solar, v, n_cloud_top):
    """triple_solar_source (l_noscal_tau and l_orog off).  v: (n_profile, n_layer+1, 3, 3), element [:, j, a, b] = v{a+1}{b+1}(l, j).
    Returns s_down, s_up (n_profile, n_layer, 3), flux_direct (n_profile, n_layer+1), flux_direct_ground (n_profile, 3)."""
    n_prof, n_layer = trans_0.shape[0], trans_0.shape[1]
    s_up = [[None] * 3 for _ in range(n_layer)]
    s_down = [[None] * 3 for _ in range(n_layer)]
    fd = {0: flux_inc_direct}
    z = jnp.zeros_like(flux_inc_direct)
    for i in range(1, n_cloud_top):
        c = i - 1
        if l_scale_solar:
            fd[i] = fd[i - 1] * trans_0[:, c, 0] * adjust_solar_ke[:, c]
            s_up[c][0] = sc[:, c, 0, 0] * fd[i - 1]
            s_down[c][0] = (sc[:, c, 1, 0] - trans_0[:, c, 0]) * fd[i - 1] + fd[i]
        else:
            fd[i] = fd[i - 1] * trans_0[:, c, 0]
            s_up[c][0] = sc[:, c, 0, 0] * fd[i - 1]
            s_down[c][0] = sc[:, c, 1, 0] * fd[i - 1]
    base = [fd[n_cloud_top - 1], z, z]
    for i in range(n_cloud_top, n_layer + 1):
        c = i - 1
        top = [v[:, i - 1, a, 0] * base[0] + v[:, i - 1, a, 1] * base[1] + v[:, i - 1, a, 2] * base[2] for a in range(3)]
        nb = []
        for k in range(3):
            if l_scale_solar:
                b = top[k] * trans_0[:, c, k] * adjust_solar_ke[:, c]
                s_up[c][k] = sc[:, c, 0, k] * top[k]
                s_down[c][k] = (sc[:, c, 1, k] - trans_0[:, c, k]) * top[k] + b
            else:
                b = top[k] * trans_0[:, c, k]
                s_up[c][k] = sc[:, c, 0, k] * top[k]
                s_down[c][k] = sc[:, c, 1, k] * top[k]
            nb.append(b)
        base = nb
        fd[i] = base[0] + base[1] + base[2]
    ground = jnp.stack(base, axis=1)
    nan = jnp.full_like(flux_inc_direct, jnp.nan)
    fill = lambda a: jnp.stack([jnp.stack([x if x is not None else nan for x in row], axis=1) for row in a], axis=1)
    return fill(s_down), fill(s_up), jnp.stack([fd[i] for i in range(n_layer + 1)], axis=1), ground


def triple_column(isolir, n_cloud_top, trans, reflect, trans_0, source_coeff, co, flux_inc_down, diffuse_albedo, *,
                  diff_planck=None, diff_planck_2=None, w_free=None, w_cloud=None, frac_region=None, d_planck_flux_surface=None,
                  flux_inc_direct=None, adjust_solar_ke=None, l_scale_solar=False, direct_albedo=None):
    n_prof, n_layer = trans.shape[0], trans.shape[1]
    v = jnp.reshape(co[:, :, 0:9], (n_prof, n_layer + 1, 3, 3))
    u = jnp.reshape(co[:, :, 9:18], (n_prof, n_layer + 1, 3, 3))
    fdir = None
    if isolir == IR:
        s_down, s_up = ir_sources(source_coeff, diff_planck, diff_planck_2, w_free, w_cloud, frac_region, n_cloud_top)
        # the Fortran writes cloud_overlap(l, n_layer, 10), (.., 13), (.., 16): u11, u22, u33
        sg = jnp.stack([co[:, n_layer, 9] * (1.0 - diffuse_albedo) * d_planck_flux_surface,
                        co[:, n_layer, 12] * (1.0 - diffuse_albedo) * d_planck_flux_surface,
                        co[:, n_layer, 15] * (1.0 - diffuse_albedo) * d_planck_flux_surface], axis=1)
    else:
        s_down, s_up, fdir, fground = solar_sources(source_coeff, trans_0, flux_inc_direct, adjust_solar_ke, l_scale_solar, v,
                                                    n_cloud_top)
        sg = jnp.stack([(direct_albedo - diffuse_albedo) * fground[:, k] for k in range(3)], axis=1)
    ft = solver_triple_hogan(n_cloud_top, trans[:, :, 0], reflect[:, :, 0], s_down[:, :, 0], s_up[:, :, 0],
                             trans[:, :, 1], reflect[:, :, 1], s_down[:, :, 1], s_up[:, :, 1],
                             trans[:, :, 2], reflect[:, :, 2], s_down[:, :, 2], s_up[:, :, 2], v, u,
                             flux_inc_down, sg[:, 0], sg[:, 1], sg[:, 2], diffuse_albedo)
    return dict(flux_total=ft, s_down=s_down, s_up=s_up, source_flux_ground=sg, flux_direct=fdir)
