"""JAX (batched, vectorized) port of OCNDYN2.f's OSTRES2 -- Stage 2, D36.

Fully vectorized over the (IM,JM) ocean grid using jnp.roll for the C-grid neighbor shifts
(the Fortran wraparound-I loops become periodic jnp.roll(..., axis=0, shift=...) calls; there
is no J-direction wraparound, so J-neighbor accesses use plain array slicing/padding). See
ostres2_ff.py for the reference derivation and physical documentation.

Arrays here are 0-indexed (IM,JM) with I=0..IM-1 <-> Fortran I=1..IM, J=0..JM-1 <-> Fortran
J=1..JM -- NOT the 1-indexed (IM+1,JM+1) convention ostres2_ff.py uses. ostres2_jax_compare.py
handles the index-convention translation when validating against the same dumps.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

IM, JM = 72, 46
RADIUS = 6371000.0


def geomo_arrays_jax():
    """0-indexed (length JM / IM) analytic geometry, matching ostres2_ff.geomo_arrays()
    but without the 1-indexed padding."""
    twopi = 2.0 * jnp.pi
    dlon = twopi / IM
    fjeq = 0.5 * (1 + JM)
    odlat_dg = round(180.0 / (JM - 1))
    dlat = odlat_dg * jnp.pi / 180.0

    j1 = jnp.arange(1, JM + 1, dtype=jnp.float64)  # Fortran J = 1..JM
    latn = jnp.where(j1 == JM, twopi / 4, dlat * (j1 + 0.5 - fjeq))
    lats = jnp.where(j1 == 1, -twopi / 4, dlat * (j1 - 0.5 - fjeq))
    dxyp = RADIUS * RADIUS * dlon * (jnp.sin(latn) - jnp.sin(lats))
    dxys = 0.5 * dxyp
    dxyn = 0.5 * dxyp

    # DXYVO(J) = DXYN(J) + DXYS(J+1), defined for J=1..JM-1 (0-indexed 0..JM-2)
    dxyvo = dxyn[:-1] + dxys[1:]
    dxyvo = jnp.concatenate([dxyvo, jnp.zeros(1, dtype=dxyvo.dtype)])  # pad to length JM

    i1 = jnp.arange(1, IM + 1, dtype=jnp.float64)
    cosic = jnp.cos((i1 - 0.5) * twopi / IM)
    sinic = jnp.sin((i1 - 0.5) * twopi / IM)
    return dxys, dxyn, dxyvo, cosic, sinic


@partial(jax.jit, static_argnames=())
def ostres2_jax(lmu, lmv, dmua, dmva, dmui, dmvi, mo1, uo0, vo0, uod0, vod0, ivnp):
    """Batched OSTRES2. All 2D fields shape (IM,JM), 0-indexed. `ivnp` is a scalar (0-indexed
    column, i.e. Fortran IVNP-1). Both branches always computed; jnp.where merges by mask."""
    dxyso, dxyno, dxyvo, cosic, sinic = geomo_arrays_jax()

    im_axis = 0  # I varies along axis 0
    # "previous I" (wraparound): roll +1 along I brings index I-1 into position I
    roll_prev = lambda a: jnp.roll(a, shift=1, axis=im_axis)   # a_prev[i] = a[i-1]
    roll_next = lambda a: jnp.roll(a, shift=-1, axis=im_axis)  # a_next[i] = a[i+1]

    # ---- U component: UO(I) updated from stress at I and I+1 (Fortran's I,IP1 pair) ----
    dmua_ip1 = roll_next(dmua)
    dmui_i = dmui
    mo1_ip1 = roll_next(mo1)
    u_upd = uo0 + (dmua + dmua_ip1 + 2.0 * dmui_i) / (mo1 + mo1_ip1)
    uo = jnp.where(lmu > 0, u_upd, uo0)
    # J range: Fortran J=2..JM-1 -> 0-indexed rows 1..JM-2
    j_mask_u = jnp.zeros(JM, dtype=bool).at[1:JM - 1].set(True)
    uo = jnp.where(j_mask_u[None, :], uo, uo0)

    # North Pole special case for UO (Fortran UO(IM,JM), UO(IVNP,JM) -> 0-indexed [IM-1,JM-1], [ivnp,JM-1])
    uo = uo.at[IM - 1, JM - 1].set(uo0[IM - 1, JM - 1] + dmua[0, JM - 1] / mo1[0, JM - 1])
    uo = uo.at[ivnp, JM - 1].set(uo0[ivnp, JM - 1] + dmva[0, JM - 1] / mo1[0, JM - 1])

    # ---- VOD: uses dmvi at J and J-1, both I and I+1 ----
    dmvi_jm1 = jnp.pad(dmvi[:, :-1], ((0, 0), (1, 0)))  # dmvi shifted: col j <- dmvi[:,j-1]
    dmvi_ip1 = roll_next(dmvi)
    dmvi_ip1_jm1 = jnp.pad(dmvi_ip1[:, :-1], ((0, 0), (1, 0)))
    dmva_ip1 = roll_next(dmva)
    vod_upd = vod0 + (dmva + dmva_ip1
                       + 0.5 * (dmvi_jm1 + dmvi_ip1_jm1 + dmvi + dmvi_ip1)) / (mo1 + mo1_ip1)
    vod = jnp.where(lmu > 0, vod_upd, vod0)
    vod = jnp.where(j_mask_u[None, :], vod, vod0)

    # ---- V component: VO(J) from stress at J and J+1 (Fortran J=2..JM-2) ----
    dxyno_row = dxyno[None, :]
    dmva_jp1 = jnp.pad(dmva[:, 1:], ((0, 0), (0, 1)))
    dmvi_dxyvo = dmvi * dxyvo[None, :]
    mo1_jp1 = jnp.pad(mo1[:, 1:], ((0, 0), (0, 1)))
    dxyso_jp1_full = jnp.pad(dxyso[1:], (0, 1))[None, :]
    v_upd = vo0 + (dmva * dxyno_row + dmva_jp1 * dxyso_jp1_full + dmvi_dxyvo) / (
        mo1 * dxyno_row + mo1_jp1 * dxyso_jp1_full)
    vo = jnp.where(lmv > 0, v_upd, vo0)
    j_mask_v = jnp.zeros(JM, dtype=bool).at[1:JM - 2].set(True)  # Fortran J=2..JM-2 -> 0-idx 1..JM-3
    vo = jnp.where(j_mask_v[None, :], vo, vo0)

    # North Pole V special case: Fortran J=JM-1 -> 0-indexed JM-2
    jnp_row = JM - 2
    vo_np = vo0[:, jnp_row] + (
        dmva[:, jnp_row] * dxyno[jnp_row]
        + (dmva[0, JM - 1] * cosic - dmua[0, JM - 1] * sinic) * dxyso[JM - 1]
        + dmvi[:, jnp_row] * dxyvo[jnp_row]
    ) / (mo1[:, jnp_row] * dxyno[jnp_row] + mo1[0, JM - 1] * dxyso[JM - 1])
    vo = vo.at[:, jnp_row].set(vo_np)

    # ---- UOD: Fortran J=2..JM-2 -> 0-idx 1..JM-3, uses I-1 (IM1) and I ----
    dmua_jp1 = jnp.pad(dmua[:, 1:], ((0, 0), (0, 1)))
    dmui_im1 = roll_prev(dmui)
    dmui_im1_jp1 = jnp.pad(dmui_im1[:, 1:], ((0, 0), (0, 1)))
    dmui_jp1 = jnp.pad(dmui[:, 1:], ((0, 0), (0, 1)))
    uod_upd = uod0 + (dmua + dmua_jp1
                       + 0.5 * (dmui_im1 + dmui + dmui_im1_jp1 + dmui_jp1)) / (mo1 + mo1_jp1)
    uod = jnp.where(lmv > 0, uod_upd, uod0)
    uod = jnp.where(j_mask_v[None, :], uod, uod0)

    return uo, vo, uod, vod
