"""OCNMESO inputs under jax.jit (Stage 2, D86): ocnmeso_vec's ocnstate_derived, densgrad_vertical and
get_1d_mesodiff as jitted functions. ocnstate_derived keeps its sequential 13-layer loop (interface
pressure accumulated in the scalar order); the other two have no recurrence. Arrays are 1-based
(IM+1, JM+1, LMO+1). Validated against ocnmeso_vec by ocnmeso_jax_compare.py.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from odhorz_ff import geomo_dyn_arrays
from ocnmeso_ff import IM, JM, LMO, GRAV, MESO_DIFFUSIVITY_CONST

with jax.ensure_compile_time_eval():   # imported lazily inside jit traces (jax_ocean): a plain jnp.asarray here becomes an escaped tracer (D214)
    _DXYPO = jnp.asarray(geomo_dyn_arrays()[-1])


def _active(lmm):
    lidx = jnp.arange(LMO + 1)[None, None, :]
    act = lmm[:, :, None] >= lidx
    return act.at[:, :, 0].set(False).at[0, :, :].set(False).at[:, 0, :].set(False)


def _pole_copy(a, lmm):
    """Copy longitude 1 to 2..IM at both poles for l <= lmm[1, j]."""
    lidx = jnp.arange(LMO + 1)[None, :]
    for j in (1, JM):
        keep = (lidx >= 1) & (lidx <= lmm[1, j])                # (1, LMO+1)
        a = a.at[2:IM + 1, j, :].set(jnp.where(keep, a[1, j, :][None, :], a[2:IM + 1, j, :]))
    return a


@jax.jit
def ocnstate_derived_jax(mo, g0m, gzm, s0m, szm, opress, lmm, vup, vdn):
    shp = (IM + 1, JM + 1, LMO + 1)
    g3d, s3d, p3d, vbar, rho = [jnp.zeros(shp) for _ in range(5)]
    act = _active(lmm)
    act = act.at[2:, JM, :].set(False)      # North Pole row driven by nbyzm: only I = 1 is active
    pe_prev = opress
    for l in range(1, LMO + 1):
        within = lmm >= l
        pe_l = jnp.where(within, pe_prev + mo[:, :, l] * GRAV, pe_prev)
        a = act[:, :, l]
        bym = 1.0 / (mo[:, :, l] * _DXYPO[None, :])
        vb = (vup[:, :, l] + vdn[:, :, l]) * 0.5
        g3d = g3d.at[:, :, l].set(jnp.where(a, g0m[:, :, l] * bym, 0.0))
        s3d = s3d.at[:, :, l].set(jnp.where(a, s0m[:, :, l] * bym, 0.0))
        p3d = p3d.at[:, :, l].set(jnp.where(a, 0.5 * (pe_l + pe_prev), 0.0))
        vbar = vbar.at[:, :, l].set(jnp.where(a, vb, 0.0))
        rho = rho.at[:, :, l].set(jnp.where(a, 1.0 / vb, 0.0))
        pe_prev = pe_l
    return tuple(_pole_copy(x, lmm) for x in (g3d, s3d, p3d, vbar, rho))


@jax.jit
def densgrad_vertical_jax(lmm, dh, vbar, vup, vdn, vupu, vdnu):
    shp = (IM + 1, JM + 1, LMO + 1)
    act = _active(lmm)
    bydh = jnp.where(act, 1.0 / dh, 0.0)
    up = act[:, :, 2:]
    dvbardz = 0.5 * (vup[:, :, 2:] + vdn[:, :, 2:] - vupu[:, :, 2:] - vdnu[:, :, 2:])
    dzvlm1 = 0.5 * (dh[:, :, 2:] + dh[:, :, 1:-1])
    by = 1.0 / dzvlm1
    rz = jnp.maximum(0.0, -dvbardz * by / vbar[:, :, 1:-1] ** 2)
    brz = jnp.where(rz != 0.0, 1.0 / rz, 0.0)
    out = []
    for v in (dzvlm1, by, None, rz, brz):
        if v is None:
            out.append(bydh)
        else:
            out.append(jnp.zeros(shp).at[:, :, 1:-1].set(jnp.where(up, v, 0.0)))
    dzv, bydzv, bydh, rhomz, byrhoz = out
    return tuple(_pole_copy(x, lmm) for x in (dzv, bydzv, bydh, rhomz, byrhoz))


@jax.jit
def get_1d_mesodiff_jax(lmm):
    return jnp.where(_active(lmm), MESO_DIFFUSIVITY_CONST, 0.0)
