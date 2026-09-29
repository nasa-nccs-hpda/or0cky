"""JAX (batched, vectorized) port of OCNDYN2.f's polar UOD/VOD relax block + polevel() --
Stage 2, D39.

Vectorized over (IM,JM) per layer L (LMO=13 layers, looped in Python since polevel's pole
reconstruction for layer L depends on VO at layer L only, and jnp.roll-based neighbor shifts are
naturally per-2D-slice; LMO is small enough that a Python-level loop over layers, each fully
vectorized over (IM,JM), is simpler and still fully jit-compilable via lax.fori_loop). See
polerelax_ff.py for the reference derivation and physical documentation.

Arrays here are 0-indexed (IM,JM,LMO) -- NOT the 1-indexed convention polerelax_ff.py uses.
polerelax_jax_compare.py handles the translation.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

IM, JM, LMO = 72, 46, 13
RELFAC = 0.005


def geomo_pole_arrays_jax():
    """0-indexed COSIC, SINIC, COSU, SINU (length IM), matching polerelax_ff.geomo_pole_arrays()."""
    twopi = 2.0 * jnp.pi
    i1 = jnp.arange(1, IM + 1, dtype=jnp.float64)
    cosic = jnp.cos((i1 - 0.5) * twopi / IM)
    sinic = jnp.sin((i1 - 0.5) * twopi / IM)
    cosu = jnp.cos(i1 * twopi / IM).at[IM - 1].set(1.0)
    sinu = jnp.sin(i1 * twopi / IM).at[IM - 1].set(0.0)
    return cosic, sinic, cosu, sinu


@jax.jit
def polerelax_jax(lmu, lmv, uo0, vo0, uod0, vod0):
    """Batched polar relax block. All fields shape (IM,JM,LMO), 0-indexed. Returns
    (uo,vo,uod,vod), each (IM,JM,LMO)."""
    cosic, sinic, cosu, sinu = geomo_pole_arrays_jax()
    l_idx = jnp.arange(1, LMO + 1, dtype=jnp.float64)  # Fortran L = 1..LMO, shape (LMO,)

    # ---- polevel: reconstruct the pole row (0-idx JM-1) for every layer at once ----
    v_ring = vo0[:, JM - 2, :]          # (IM, LMO), Fortran VO(:,JM-1,:)
    lmv_ring = lmv[:, JM - 2]           # (IM,)
    mask_ring = l_idx[None, :] <= lmv_ring[:, None]  # (IM, LMO)
    unp = jnp.sum(jnp.where(mask_ring, -sinic[:, None] * v_ring, 0.0), axis=0) * 2.0 / IM  # (LMO,)
    vnp = jnp.sum(jnp.where(mask_ring, cosic[:, None] * v_ring, 0.0), axis=0) * 2.0 / IM
    u_pole = unp[None, :] * cosu[:, None] + vnp[None, :] * sinu[:, None]  # (IM, LMO)
    v_pole = vnp[None, :] * cosic[:, None] - unp[None, :] * sinic[:, None]

    uo = uo0.at[:, JM - 1, :].set(u_pole)
    vo = vo0.at[:, JM - 1, :].set(v_pole)

    lmv3 = lmv[:, :, None]  # (IM,JM,1) broadcast against l_idx (LMO,)
    lmu3 = lmu[:, :, None]

    # ---- UOD, pole-adjacent row (0-idx JM-2, Fortran J=JM-1): doubled uo(:,JM,l) term ----
    jpole = JM - 2
    uo_im1 = jnp.roll(uo, shift=1, axis=0)   # uo[i-1] (west neighbor), wraps IM-1<->0
    upd_pole = (uod0[:, jpole, :] * (1.0 - RELFAC) + RELFAC * 0.25 *
                (uo_im1[:, jpole, :] + uo[:, jpole, :] + 2.0 * uo[:, jpole + 1, :]))
    mask_pole = l_idx[None, :] <= lmv3[:, jpole, :]
    uod = uod0.at[:, jpole, :].set(jnp.where(mask_pole, upd_pole, uod0[:, jpole, :]))

    # ---- UOD, interior rows (0-idx 1..JM-3, Fortran J=2..JM-2): non-doubled ----
    uo_jp1 = jnp.concatenate([uo[:, 1:, :], jnp.zeros((IM, 1, LMO))], axis=1)       # uo[:,j+1,:]
    uo_im1_jp1 = jnp.roll(uo_jp1, shift=1, axis=0)
    upd_int = (uod0 * (1.0 - RELFAC) + RELFAC * 0.25 *
               (uo_im1 + uo + uo_im1_jp1 + uo_jp1))
    j_mask_int = jnp.zeros(JM, dtype=bool).at[1:JM - 2].set(True)  # 0-idx 1..JM-3
    mask_int = (l_idx[None, None, :] <= lmv3) & j_mask_int[None, :, None]
    uod = jnp.where(mask_int, upd_int, uod)

    # ---- VOD, all rows (0-idx 1..JM-2, Fortran J=2..JM-1): never touches the pole row ----
    vo_ip1 = jnp.roll(vo, shift=-1, axis=0)  # vo[i+1] (east neighbor)
    vo_jm1 = jnp.concatenate([jnp.zeros((IM, 1, LMO)), vo[:, :-1, :]], axis=1)      # vo[:,j-1,:]
    vo_ip1_jm1 = jnp.roll(vo_jm1, shift=-1, axis=0)
    upd_v = (vod0 * (1.0 - RELFAC) + RELFAC * 0.25 *
             (vo_jm1 + vo_ip1_jm1 + vo + vo_ip1))
    j_mask_v = jnp.zeros(JM, dtype=bool).at[1:JM - 1].set(True)  # 0-idx 1..JM-2
    mask_v = (l_idx[None, None, :] <= lmu3) & j_mask_v[None, :, None]
    vod = jnp.where(mask_v, upd_v, vod0)

    return uo, vo, uod, vod
