"""Batched JAX port of ground_oc_sweep_ff.py (OCNDYN.f GROUND_OC's below-freezing layer sweep) --
Stage 2 of the DYNSI/ocean port, D35. Both branches (below freezing / not) computed for every lane
and merged with jnp.where, the established pattern."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from seaice_core_jax import Ei
from seaice_core_ff import FSSS
from osourc_jax import gfrezs, tfrezs


def ground_oc_sweep_layer(mo_l0, g0m_l0, s0m_l0, dxypj, pcorr, p0l):
    """Batched version of ground_oc_sweep_ff.ground_oc_sweep_layer."""
    g0l = g0m_l0 / (mo_l0 * dxypj)
    s0l = s0m_l0 / (mo_l0 * dxypj)
    gf00 = gfrezs(s0l)
    gf0 = gf00 - pcorr
    freezing = g0l < gf0

    tf0 = tfrezs(s0l) - 7.53e-8 * p0l
    si0 = FSSS * s0l
    ei0 = Ei(tf0, si0 * 1e3)
    dm0_f = mo_l0 * (g0l - gf0) / (ei0 - gf0)
    de0_f = ei0 * dm0_f
    ds0_f = si0 * dm0_f

    zeros = jnp.zeros_like(mo_l0)
    dm0 = jnp.where(freezing, dm0_f, zeros)
    de0 = jnp.where(freezing, de0_f, zeros)
    ds0 = jnp.where(freezing, ds0_f, zeros)

    mo = mo_l0 - dm0
    g0m = g0m_l0 - de0 * dxypj
    s0m = s0m_l0 - ds0 * dxypj
    return dict(mo=mo, g0m=g0m, s0m=s0m, dm0=dm0, de0=de0, ds0=ds0)
