"""OPFIL2 as precomputed linear operators (Stage 2, D77, speed-first JAX form).

Within one layer, OPFIL2 acts on each latitude row independently and linearly: the FFT smoother
and the basin matrix filter are both linear maps of that row. So the whole application is one
72x72 operator per (layer, row). The operators are built once from the validated scalar port
(opfil2_ff.opfil2, D74-D75) by applying it to the unit basis, then the step applies them as a
batched matrix product in JAX.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import opfil2_ff as S

IM = S.IM
JM = S.JM
LMO = 13


def build_operators(c):
    """Returns T of shape (LMO+1, JM+1, IM, IM) with T[l, j] applied as x_row' = T[l, j] @ x_row
    (1-based l and j; rows outside a layer's filtered set are identity)."""
    T = np.zeros((LMO + 1, JM + 1, IM, IM))
    for l in range(1, LMO + 1):
        for j in range(1, JM + 1):
            basis = np.zeros((IM, JM))
            # apply to each unit vector in row j, through the scalar port on the whole band
            for i in range(IM):
                basis[:] = 0.0
                basis[i, j - 1] = 1.0
                out = S.opfil2(basis, l, j, j, c)
                T[l, j, :, i] = out[:, j - 1]
    return T


def apply_operators(T, x, l, jmin, jmax):
    """x: (IM, JM). Applies the layer-l operators to rows jmin..jmax (1-based). Batched in JAX."""
    rows = jnp.arange(jmin, jmax + 1)
    Ts = jnp.asarray(T[l])[rows]                        # (nrows, IM, IM)
    xr = jnp.asarray(x)[:, rows - 1].T                  # (nrows, IM)
    yr = jnp.einsum('rij,rj->ri', Ts, xr)
    return jnp.asarray(x).at[:, rows - 1].set(yr.T)
