"""Per k-term optical properties of the band solver (single_scattering_all + rescale_tau_omega) for this rundeck, plus the k-eqv absorption
assembly (k_gas_abs of the first absorber).   Layout: layer index = layer - 1; cloud-type index 0 = the clear part of the cloudy
layers, 1..n_cloud_type.   Fortran: radiance_core/single_scattering.F90, single_scattering_all.F90, rescale_tau_omega.F90."""
import numpy as np
import jax.numpy as jnp

EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny


def single_scattering(k_grey_tot, k_ext_scat, k_gas_abs, d_mass):
    """i_scatter_method_band = ip_scatter_full/approx (the only branch the rundeck uses)."""
    ktot = k_grey_tot + k_gas_abs
    tau = ktot * d_mass
    omega = k_ext_scat / (ktot + TINY)
    return tau, jnp.minimum(omega, 1.0 - 32.0 * EPS)


def rescale_tau_omega(tau, omega, fwd):
    return tau * (1.0 - omega * fwd), omega * (1.0 - fwd) / (1.0 - omega * fwd)


def kterm_props(k_grey_tot_clr, k_ext_scat_clr, k_grey_tot, k_ext_scat, k_gas_abs, d_mass, n_cloud_top, fwd_clr, fwd,
                l_rescale=True):
    """Inputs (n_profile, n_layer[, n_type+1]); returns tau_clr, omega_clr (n_profile, n_layer: valid for layers < n_cloud_top) and
    tau, omega (n_profile, n_layer, n_type+1: valid for layers >= n_cloud_top)."""
    tc, oc = single_scattering(k_grey_tot_clr, k_ext_scat_clr, k_gas_abs, d_mass)
    t, o = single_scattering(k_grey_tot, k_ext_scat, k_gas_abs[:, :, None], d_mass[:, :, None])
    if l_rescale:
        tc, oc = rescale_tau_omega(tc, oc, fwd_clr)
        t, o = rescale_tau_omega(t, o, fwd)
    return tc, oc, t, o
