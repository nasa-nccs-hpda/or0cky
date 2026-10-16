"""JAX (batched) port of OVDIFFS + TRIDIAG -- Stage 2, D62 (speed-first JAX port).

Batched over N independent columns. Each column has its own active length `lmij`; rows past
lmij are set to the identity so one fixed-size Thomas solve covers every column. The forward
and backward sweeps are lax.scan over the layer index, so the whole batch is one jitted call.

Arrays are 1-indexed in the layer dimension, padded to LMO+1 (index 0 unused), matching
ovdiffs_ff.py: shape (N, LMO+1). lmij is shape (N,) integer-valued.

Accuracy: the Fortran operation order is kept where it is cheap to keep, but this port is
checked to a stated tolerance (see ovdiffs_jax_compare.py), not bitwise. Speed is the priority.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

LMO = 13


@jax.jit
def ovdiffs_jax(k, ghat, dtp4, dtbydz, bydz2, dt, lmij, u0):
    """Batched OVDIFFS. Inputs (N, LMO+1) except dt scalar and lmij (N,). Returns (u, fl),
    both (N, LMO+1), 1-indexed."""
    N = k.shape[0]
    L = jnp.arange(1, LMO + 1)                        # layer index 1..LMO, row r = L-1
    lm = lmij[:, None]                                # (N,1)
    active = L[None, :] <= lm                         # rows 1..lmij
    bottom = L[None, :] == lm                         # row lmij
    first = (L == 1)[None, :]

    k_L = k[:, 1:LMO + 1]
    k_Lm1 = jnp.concatenate([jnp.zeros((N, 1)), k[:, 1:LMO]], axis=1)   # k[L-1], 0 at L=1
    g_L = ghat[:, 1:LMO + 1]
    g_Lm1 = jnp.concatenate([jnp.zeros((N, 1)), ghat[:, 1:LMO]], axis=1)
    p_L = dtp4[:, 1:LMO + 1]
    p_Lm1 = jnp.concatenate([jnp.zeros((N, 1)), dtp4[:, 1:LMO]], axis=1)
    dz_L = dtbydz[:, 1:LMO + 1]
    dz_Lp1 = jnp.concatenate([dtbydz[:, 2:LMO + 1], jnp.zeros((N, 1))], axis=1)  # dtbydz[L+1]
    dz_Lm1 = jnp.concatenate([jnp.zeros((N, 1)), dtbydz[:, 1:LMO]], axis=1)      # dtbydz[L-1]
    by_L = bydz2[:, 1:LMO + 1]
    by_Lm1 = jnp.concatenate([jnp.zeros((N, 1)), bydz2[:, 1:LMO]], axis=1)       # bydz2[L-1]
    u_L = u0[:, 1:LMO + 1]
    u_Lm1 = jnp.concatenate([jnp.zeros((N, 1)), u0[:, 1:LMO]], axis=1)

    # interior rows (2 <= L < lmij), first row (L=1, lmij>=2), bottom row (L=lmij)
    a_int = -dz_Lm1 * by_Lm1 * k_Lm1
    b_int = 1.0 + dz_L * (by_Lm1 * k_Lm1 + by_L * k_L)
    c_int = -dz_Lp1 * by_L * k_L
    r_int = u_L + dt * (g_Lm1 - g_L) + p_L

    a_first = jnp.zeros_like(k_L)
    b_first = 1.0 + dz_L * by_L * k_L
    c_first = -dz_Lp1 * by_L * k_L
    r_first = u_L - dt * g_L + p_L

    a_bot = -dz_Lm1 * by_Lm1 * k_Lm1
    b_bot = 1.0 + dz_L * by_Lm1 * k_Lm1
    c_bot = jnp.zeros_like(k_L)
    r_bot = u_L + dt * g_Lm1 + p_Lm1

    a = jnp.where(first, a_first, a_int)
    b = jnp.where(first, b_first, b_int)
    c = jnp.where(first, c_first, c_int)
    r = jnp.where(first, r_first, r_int)
    a = jnp.where(bottom, a_bot, a)
    b = jnp.where(bottom, b_bot, b)
    c = jnp.where(bottom, c_bot, c)
    r = jnp.where(bottom, r_bot, r)
    # rows beyond lmij: identity, so they do not couple to the active rows
    a = jnp.where(active, a, 0.0)
    b = jnp.where(active, b, 1.0)
    c = jnp.where(active, c, 0.0)
    r = jnp.where(active, r, 0.0)

    u = _thomas(a, b, c, r)

    # fluxes for L = 1..lmij-1
    u_pad = jnp.concatenate([u, jnp.zeros((N, 1))], axis=1)   # u[L+1] for L=LMO is 0
    fl_inner = (k_L * (dz_Lp1 * u_pad[:, 2:LMO + 2] - dz_L * u_pad[:, 1:LMO + 1]) * by_L
                - dt * g_L)
    fl_mask = L[None, :] < lm
    fl = jnp.concatenate([jnp.zeros((N, 1)), jnp.where(fl_mask, fl_inner, 0.0)], axis=1)
    return u, fl


def _thomas(a, b, c, r):
    """Batched Thomas solve. a,b,c,r: (N, LMO) rows L=1..LMO. Returns u (N, LMO+1), 1-indexed
    (index 0 zero). Sequential over the layer axis via two lax.scan sweeps."""
    N = a.shape[0]
    xs_in = (a.T, b.T, c.T, r.T)

    def fwd(carry, x):
        cprev, dprev = carry
        ai, bi, ci, ri = x
        denom = bi - ai * cprev
        cnew = ci / denom
        dnew = (ri - ai * dprev) / denom
        return (cnew, dnew), (cnew, dnew)

    _, (cp, dp) = jax.lax.scan(fwd, (jnp.zeros(N), jnp.zeros(N)), xs_in)

    def bwd(xnext, x):
        cpi, dpi = x
        xi = dpi - cpi * xnext
        return xi, xi

    _, xs = jax.lax.scan(bwd, jnp.zeros(N), (cp, dp), reverse=True)
    return jnp.concatenate([jnp.zeros((N, 1)), xs.T], axis=1)


@jax.jit
def ovdiff_jax(k, ghat, dtp4, dtbydz, bydz2, lmij, u0):
    """Batched momentum OVDIFF (OCNKPP.f:3410-3469; ovdiff_ff.py, D56). Off-diagonals use
    DTBYDZ(L), the bottom RHS reads DTP4(LMIJ), and there is no separate DT argument.
    Inputs (N, LMO+1) 1-indexed, lmij (N,). Returns u (N, LMO+1)."""
    N = k.shape[0]
    L = jnp.arange(1, LMO + 1)
    lm = lmij[:, None]
    active = L[None, :] <= lm
    bottom = L[None, :] == lm
    first = (L == 1)[None, :]

    def sh(x, off):
        return x[:, 1 + off:1 + off + LMO]

    zero = jnp.zeros((N, 1))
    k_L, g_L, p_L, d_L, y_L, u_L = sh(k, 0), sh(ghat, 0), sh(dtp4, 0), sh(dtbydz, 0), sh(bydz2, 0), sh(u0, 0)
    k_Lm1 = jnp.concatenate([zero, k[:, 1:LMO]], axis=1)
    g_Lm1 = jnp.concatenate([zero, ghat[:, 1:LMO]], axis=1)
    y_Lm1 = jnp.concatenate([zero, bydz2[:, 1:LMO]], axis=1)

    # interior (2 <= L < lmij)
    a_int = -d_L * y_Lm1 * k_Lm1
    c_int = -d_L * y_L * k_L
    b_int = 1.0 + d_L * (y_Lm1 * k_Lm1 + y_L * k_L)
    r_int = u_L + d_L * (g_Lm1 - g_L) + p_L
    # first row (L = 1): c = -dt(1) bydz2(1) k(1), b = 1 - c, r = u0 - dt ghat(1) + dtp4(1)
    c_first = -d_L * y_L * k_L
    b_first = 1.0 - c_first
    r_first = u_L - d_L * g_L + p_L
    # bottom row (L = lmij): a = -dt(lmij) bydz2(lmij-1) k(lmij-1), b = 1 - a, c = 0
    a_bot = -d_L * y_Lm1 * k_Lm1
    b_bot = 1.0 - a_bot
    r_bot = u_L + d_L * g_Lm1 + p_L

    a = jnp.where(first, 0.0, a_int)
    b = jnp.where(first, b_first, b_int)
    c = jnp.where(first, c_first, c_int)
    r = jnp.where(first, r_first, r_int)
    a = jnp.where(bottom, a_bot, a)
    b = jnp.where(bottom, b_bot, b)
    c = jnp.where(bottom, 0.0, c)
    r = jnp.where(bottom, r_bot, r)
    a = jnp.where(active, a, 0.0)
    b = jnp.where(active, b, 1.0)
    c = jnp.where(active, c, 0.0)
    r = jnp.where(active, r, 0.0)
    return _thomas(a, b, c, r)
