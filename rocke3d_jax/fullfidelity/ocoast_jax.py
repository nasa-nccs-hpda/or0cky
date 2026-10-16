"""JAX (batched, vectorized) port of OCNDYN.f's OCOAST -- Stage 2, D37.

Fully vectorized over the (IM,JM,LMO) ocean grid: the Fortran wraparound-I neighbor pattern
becomes jnp.roll (as in D36's ostres2_jax.py), and the variable-length L=LMIN..LMM(I,J) inner
loop becomes a broadcast comparison against a layer-index vector, merged with jnp.where. See
ocoast_ff.py for the reference derivation and physical documentation.

Arrays here are 0-indexed (IM,JM,LMO) with I=0..IM-1 <-> Fortran I=1..IM, etc. -- NOT the
1-indexed convention ocoast_ff.py uses. ocoast_jax_compare.py handles the translation.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

IM, JM, LMO = 72, 46, 13
DTS = 1800.0
SECONDS_PER_DAY = 86400.0
REDUCE = 1.0 - DTS / (SECONDS_PER_DAY * 20.0)


@jax.jit
def ocoast_jax(lmm, gxmo0, sxmo0, gymo0, symo0):
    """Batched OCOAST. `lmm` shape (IM,JM) integer-valued float array. `gxmo0` etc. shape
    (IM,JM,LMO). Returns (gxmo,sxmo,gymo,symo), each (IM,JM,LMO)."""
    l_idx = jnp.arange(1, LMO + 1, dtype=jnp.float64)  # Fortran L = 1..LMO

    # ---- X pass: neighbors at I-1 (west) and I+1 (east), Fortran J=2..JM-1 -> 0-idx 1..JM-2 ----
    lmm_im1 = jnp.roll(lmm, shift=1, axis=0)   # west neighbor: lmm[i-1]
    lmm_ip1 = jnp.roll(lmm, shift=-1, axis=0)  # east neighbor: lmm[i+1]
    lmin_x = jnp.minimum(lmm_im1, lmm_ip1) + 1.0  # (IM,JM)
    lmax_x = lmm  # own depth

    mask_x = (l_idx[None, None, :] >= lmin_x[:, :, None]) & \
             (l_idx[None, None, :] <= lmax_x[:, :, None])
    j_mask = jnp.zeros(JM, dtype=bool).at[1:JM - 1].set(True)  # Fortran J=2..JM-1 -> 0-idx 1..JM-2
    mask_x = mask_x & j_mask[None, :, None]

    gxmo = jnp.where(mask_x, gxmo0 * REDUCE, gxmo0)
    sxmo = jnp.where(mask_x, sxmo0 * REDUCE, sxmo0)

    # ---- Y pass: neighbors at J-1, J+1, no wraparound needed (J-loop excludes poles) ----
    lmm_jm1 = jnp.pad(lmm[:, :-1], ((0, 0), (1, 0)))
    lmm_jp1 = jnp.pad(lmm[:, 1:], ((0, 0), (0, 1)))
    lmin_y = jnp.minimum(lmm_jm1, lmm_jp1) + 1.0
    lmax_y = lmm

    mask_y = (l_idx[None, None, :] >= lmin_y[:, :, None]) & \
             (l_idx[None, None, :] <= lmax_y[:, :, None])
    mask_y = mask_y & j_mask[None, :, None]

    gymo = jnp.where(mask_y, gymo0 * REDUCE, gymo0)
    symo = jnp.where(mask_y, symo0 * REDUCE, symo0)

    return gxmo, sxmo, gymo, symo
