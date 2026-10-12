"""JAX port of the k-equivalent absorption of solve_band_k_eqv_scl (radiance_core/solve_band_k_eqv_scl.F90, first part: the k_eqv and
adjust_solar_ke computation of the minor absorbers j = 2..n_abs) plus monochromatic_gas_flux (IR) for this rundeck (two-stream
angular integration, ip_two_stream).  Layer index 0-based; k_abs_layer: (n_profile, n_layer, nd_esft, nd_abs).
Python loops over absorbers, k-terms and layers (validation port); exp is jnp.exp (the Fortran uses exp_v = Intel libimf)."""
import numpy as np
import jax.numpy as jnp

DIFFUSIVITY_MINOR = 1.66
EPS = np.finfo(np.float64).eps
TEMP_MAX = float(np.log(1.0 / EPS))


def k_eqv_solar(k_abs_layer, n_abs_esft, w_abs_esft, index_abs, n_abs, d_mass, zen_0):
    """Returns k_eqv (n_profile, n_layer) and adjust_solar_ke (n_profile, n_layer)."""
    n_prof, nl = d_mass.shape
    k_eqv = jnp.zeros((n_prof, nl))
    adj = jnp.ones((n_prof, nl))
    for j in range(2, n_abs + 1):
        ia = int(index_abs[j - 1])
        nesft = int(n_abs_esft[ia - 1])
        flux_gas = [jnp.ones(n_prof)] + [jnp.zeros(n_prof)] * nl
        sum_flux = jnp.zeros(n_prof)
        sum_k_flux = jnp.zeros((n_prof, nl))
        k_min = jnp.full((n_prof, nl), np.finfo(np.float64).max)
        for iex in range(nesft):
            w = float(w_abs_esft[iex, ia - 1])
            k = k_abs_layer[:, :, iex, ia - 1]
            tr = jnp.exp(-k * d_mass * zen_0[:, None])
            ft = jnp.full(n_prof, w)
            for i in range(1, nl + 1):
                ft = ft * tr[:, i - 1]
                flux_gas[i] = flux_gas[i] + ft
            sum_k_flux = sum_k_flux + k * ft[:, None]
            k_min = jnp.minimum(k_min, k)
            sum_flux = sum_flux + ft
        pos = sum_flux > 0.0
        k_eqv = k_eqv + jnp.where(pos[:, None], sum_k_flux / jnp.where(pos, sum_flux, 1.0)[:, None], k_min)
        for i in range(1, nl + 1):
            ok = flux_gas[i - 1] > 0.0
            adj = adj.at[:, i - 1].set(jnp.where(ok, adj[:, i - 1] * flux_gas[i] / jnp.where(ok, flux_gas[i - 1], 1.0), adj[:, i - 1]))
    adj = adj * jnp.exp(jnp.minimum(k_eqv * d_mass * zen_0[:, None], TEMP_MAX))
    return k_eqv, adj


def k_eqv_ir(k_abs_layer, n_abs_esft, w_abs_esft, index_abs, n_abs, d_mass, planck_flux_band, diff_planck_band, planck_flux_surface,
             diffuse_albedo):
    """IR branch.  planck_flux_band: (n_profile, n_layer+1) (levels 0..n_layer).  Returns k_eqv (n_profile, n_layer)."""
    n_prof, nl = d_mass.shape
    sq_eps = float(np.sqrt(EPS))
    flux_inc_down = -planck_flux_band[:, 0]
    d_surf = planck_flux_surface - planck_flux_band[:, nl]
    k_eqv = jnp.zeros((n_prof, nl))
    for j in range(2, n_abs + 1):
        ia = int(index_abs[j - 1])
        nesft = int(n_abs_esft[ia - 1])
        sum_k = [jnp.zeros(n_prof) for _ in range(2 * nl + 2)]      # index i-1 of the Fortran 1..2nl+2
        sum_f = [jnp.zeros(n_prof) for _ in range(2 * nl + 2)]
        for iex in range(nesft):
            w = float(w_abs_esft[iex, ia - 1])
            k = k_abs_layer[:, :, iex, ia - 1]
            tau = k * d_mass
            trans = jnp.exp(-DIFFUSIVITY_MINOR * tau)
            s_up = (1.0 - trans + sq_eps) * diff_planck_band / (DIFFUSIVITY_MINOR * tau + sq_eps)
            s_down = -s_up
            fd = {2: flux_inc_down}
            for i in range(1, nl + 1):
                fd[2 * i + 2] = trans[:, i - 1] * fd[2 * i] + s_down[:, i - 1]
            fd[2 * nl + 1] = d_surf + diffuse_albedo * fd[2 * nl + 2]
            for i in range(nl, 0, -1):
                fd[2 * i - 1] = trans[:, i - 1] * fd[2 * i + 1] + s_up[:, i - 1]
            for i in range(2, 2 * nl + 2):
                sum_k[i - 1] = sum_k[i - 1] + k[:, i // 2 - 1] * w * jnp.abs(fd[i])
                sum_f[i - 1] = sum_f[i - 1] + w * jnp.abs(fd[i])
        cols = []
        for i in range(1, nl + 1):
            lk = sum_k[2 * i - 1] + sum_k[2 * i]
            lf = sum_f[2 * i - 1] + sum_f[2 * i]
            cols.append(lk / lf)
        k_eqv = k_eqv + jnp.stack(cols, axis=1)
    return k_eqv
