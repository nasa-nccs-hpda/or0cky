"""JAX (batched, vectorized) port of OCNDYN2.f's OBDRAG2 -- Stage 2, D38.

Fully vectorized over the (IM,JM) ocean grid: the Fortran wraparound-I neighbor pattern becomes
jnp.roll (as in D36/D37), and the "operate only at the per-column bottom layer L=LMU(I,J) (or
LMV(I,J))" selection becomes a broadcast layer-index-equality mask merged with jnp.where -- since
WSQ/bdragfac/the scale factor must be computed AT that one layer per column, gathered via
jnp.take_along_axis rather than looped. See obdrag2_ff.py for the reference derivation.

Arrays here are 0-indexed (IM,JM,LMO) -- NOT the 1-indexed convention obdrag2_ff.py uses.
obdrag2_jax_compare.py handles the translation.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

IM, JM, LMO = 72, 46, 13
BDRAGX = 1.0
DTS = 1800.0


@jax.jit
def obdrag2_jax(lmu, lmv, mo, uo0, vo0, uod0, vod0):
    """Batched OBDRAG2. `lmu`,`lmv` shape (IM,JM) integer-valued float. `mo`,`uo0`,`vo0`,`uod0`,
    `vod0` shape (IM,JM,LMO). Returns (uo,vo,uod,vod), each (IM,JM,LMO)."""
    l_idx = jnp.arange(1, LMO + 1, dtype=jnp.float64)  # Fortran L = 1..LMO

    j_mask = jnp.zeros(JM, dtype=bool).at[1:JM - 1].set(True)  # Fortran J=2..JM-1 -> 0-idx 1..JM-2

    def gather_bottom(field, l_sel):
        """field: (IM,JM,LMO). l_sel: (IM,JM) 1-indexed layer to gather (0 where invalid).
        Returns (IM,JM) gathered values (garbage where l_sel<=0, caller must mask)."""
        idx = jnp.clip(l_sel.astype(jnp.int32) - 1, 0, LMO - 1)
        return jnp.take_along_axis(field, idx[:, :, None], axis=2)[:, :, 0]

    # ---- East-edge (U/VOD) pass ----
    mo_ip1 = jnp.roll(mo, shift=-1, axis=0)
    uo0_l = gather_bottom(uo0, lmu)
    vod0_l = gather_bottom(vod0, lmu)
    mo_l = gather_bottom(mo, lmu)
    moip1_l = gather_bottom(mo_ip1, lmu)
    wsq_u = uo0_l ** 2 + vod0_l ** 2 + 1e-20
    bdragfac_u = BDRAGX * jnp.sqrt(wsq_u)
    denom_u = mo_l + moip1_l
    factor_u = denom_u / (denom_u + DTS * bdragfac_u * 2.0)
    uo0_new = uo0_l * factor_u
    vod0_new = vod0_l * factor_u

    valid_u = (lmu > 0) & j_mask[None, :]
    mask3_u = (l_idx[None, None, :] == lmu[:, :, None]) & valid_u[:, :, None]
    uo = jnp.where(mask3_u, uo0_new[:, :, None], uo0)
    vod = jnp.where(mask3_u, vod0_new[:, :, None], vod0)

    # ---- North-edge (V/UOD) pass ----
    mo_jp1 = jnp.pad(mo[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
    vo0_l = gather_bottom(vo0, lmv)
    uod0_l = gather_bottom(uod0, lmv)
    mo_l2 = gather_bottom(mo, lmv)
    mojp1_l = gather_bottom(mo_jp1, lmv)
    wsq_v = vo0_l ** 2 + uod0_l ** 2 + 1e-20
    bdragfac_v = BDRAGX * jnp.sqrt(wsq_v)
    denom_v = mo_l2 + mojp1_l
    factor_v = denom_v / (denom_v + DTS * bdragfac_v * 2.0)
    vo0_new = vo0_l * factor_v
    uod0_new = uod0_l * factor_v

    valid_v = (lmv > 0) & j_mask[None, :]
    mask3_v = (l_idx[None, None, :] == lmv[:, :, None]) & valid_v[:, :, None]
    vo = jnp.where(mask3_v, vo0_new[:, :, None], vo0)
    uod = jnp.where(mask3_v, uod0_new[:, :, None], uod0)

    return uo, vo, uod, vod
