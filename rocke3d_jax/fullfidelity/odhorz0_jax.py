"""JAX (batched, vectorized) port of OCNDYN2.f's ODHORZ0 -- Stage 2, D40.

Vectorized over (IM,JM) per layer via broadcast masks; the North Pole row's special
"only I=1 active, regardless of LMM(I>1,JM)" restriction (`m_active` in odhorz0_ff.py -- the real
design gap caught during D40's validation) is encoded as an explicit per-layer mask override, not
just LMM>=L. See odhorz0_ff.py for the reference derivation and physical documentation.

Arrays here are 0-indexed (IM,JM,LMO) -- NOT the 1-indexed convention odhorz0_ff.py uses.
odhorz0_jax_compare.py handles the translation.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from polerelax_jax import geomo_pole_arrays_jax
from ostres2_jax import geomo_arrays_jax

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665
Z12EH = 0.28867513


def _m_active_mask(lmm):
    """0-indexed (IM,JM,LMO) boolean mask matching odhorz0_ff.m_active: ordinary LMM>=L
    everywhere except row JM-1 (0-idx, Fortran J=JM), which is restricted to I=0 only."""
    l_idx = jnp.arange(1, LMO + 1, dtype=jnp.float64)
    ordinary = l_idx[None, None, :] <= lmm[:, :, None]
    pole_active = l_idx <= lmm[0, JM - 1]  # (LMO,) -- Fortran LMM(1,JM)
    pole_row = jnp.zeros((IM, LMO), dtype=bool).at[0, :].set(pole_active)
    return ordinary.at[:, JM - 1, :].set(pole_row)


@jax.jit
def odhorz0_jax(lmm, lmv, opress, g0m, gzm, s0m, szm, mo0, uo0, vo0, vup, vdn):
    """Batched ODHORZ0. 2D fields (lmm,lmv,opress) shape (IM,JM); 3D fields shape (IM,JM,LMO),
    0-indexed. Returns a dict matching odhorz0_ff.odhorz0's keys."""
    dxys, dxyn, _, _, _ = geomo_arrays_jax()
    dxypo = dxys + dxyn  # (JM,)
    cosic, sinic, cosu, sinu = geomo_pole_arrays_jax()

    mask = _m_active_mask(lmm)  # (IM,JM,LMO)

    # ---- Pressure integration (cumulative sum over L, masked) ----
    mo_masked = jnp.where(mask, mo0, 0.0)
    cum_below = jnp.cumsum(mo_masked, axis=2) * GRAV       # OPBOT contribution up to and incl. L
    cum_above = cum_below - mo_masked * GRAV                # OPBOT contribution strictly above L
    # OPBOT(I,J) after L layers = init + cum_below(...,L); init applies only where mask(...,1)
    init_mask = mask[:, :, 0]
    opbot_init = jnp.where(init_mask, opress, 0.0)
    p = opbot_init[:, :, None] + cum_above + mo0 * GRAV * 0.5
    p = jnp.where(mask, p, 0.0)
    opbot = opbot_init + jnp.sum(jnp.where(mask, mo_masked * GRAV, 0.0), axis=2)

    # ---- GUP/GDN/SUP/SDN (Linear Upstream Scheme) ----
    mmi = jnp.where(mask, mo0 * dxypo[None, :, None], 1.0)  # avoid div-by-zero off-mask
    gup = jnp.where(mask, (g0m - 2 * Z12EH * gzm) / mmi, 0.0)
    gdn = jnp.where(mask, (g0m + 2 * Z12EH * gzm) / mmi, 0.0)
    sup_raw = (s0m - 2 * Z12EH * szm) / mmi
    sdn_raw = (s0m + 2 * Z12EH * szm) / mmi
    smean = s0m / mmi
    sup = jnp.where(mask, jnp.maximum(sup_raw, 0.5 * smean), 0.0)
    sdn = jnp.where(mask, jnp.maximum(sdn_raw, 0.5 * smean), 0.0)

    dzgdp = jnp.where(mask, vup * (0.5 - Z12EH) + vdn * (0.5 + Z12EH), 0.0)
    vbar = jnp.where(mask, (vup + vdn) * 0.5, 0.0)
    dh3d = jnp.where(mask, mo0 * vbar, 0.0)

    # ---- Copy to all longitudes at North Pole (DH3D, VBAR, dZGdP, MO) ----
    l_idx = jnp.arange(1, LMO + 1, dtype=jnp.float64)
    pole_copy_mask = l_idx <= lmm[0, JM - 1]  # (LMO,)
    dh3d = dh3d.at[:, JM - 1, :].set(
        jnp.where(pole_copy_mask, dh3d[0, JM - 1, :], dh3d[:, JM - 1, :]))
    vbar = vbar.at[:, JM - 1, :].set(
        jnp.where(pole_copy_mask, vbar[0, JM - 1, :], vbar[:, JM - 1, :]))
    dzgdp = dzgdp.at[:, JM - 1, :].set(
        jnp.where(pole_copy_mask, dzgdp[0, JM - 1, :], dzgdp[:, JM - 1, :]))
    mo = mo0.at[:, JM - 1, :].set(
        jnp.where(pole_copy_mask, mo0[0, JM - 1, :], mo0[:, JM - 1, :]))

    # ---- polevel: reconstruct the pole row (0-idx JM-1) for every layer at once ----
    v_ring = vo0[:, JM - 2, :]
    lmv_ring = lmv[:, JM - 2]
    mask_ring = l_idx[None, :] <= lmv_ring[:, None]
    unp = jnp.sum(jnp.where(mask_ring, -sinic[:, None] * v_ring, 0.0), axis=0) * 2.0 / IM
    vnp = jnp.sum(jnp.where(mask_ring, cosic[:, None] * v_ring, 0.0), axis=0) * 2.0 / IM
    u_pole = unp[None, :] * cosu[:, None] + vnp[None, :] * sinu[:, None]
    v_pole = vnp[None, :] * cosic[:, None] - unp[None, :] * sinic[:, None]
    uo = uo0.at[:, JM - 1, :].set(u_pole)
    vo = vo0.at[:, JM - 1, :].set(v_pole)

    return dict(opbot=opbot, gup=gup, gdn=gdn, sup=sup, sdn=sdn, dzgdp=dzgdp, vbar=vbar,
                dh3d=dh3d, mo=mo, uo=uo, vo=vo)
