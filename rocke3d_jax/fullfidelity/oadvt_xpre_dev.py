"""D214: the OADVT2 X pre-pass (oadvt_vec._x_prepass) as a fully vectorized DEVICE program (no host callback, no sequential scan).

The NumPy original walks the (l, j) passes in order with ONE mudt array whose entries that a pass does not assign keep the value of the previous pass ('stale entries',
which the flux sweep later reads).  Observation used here: within a pass every value that is READ is assigned in that same pass, so the courant number of a pass
depends only on that pass's own data; and the final contents of mudt after a pass are 'the last assignment of each cell in the passes so far', which is a forward-fill
along the pass order.  So: (1) per pass, paint the cells the pass assigns and their values (same order of overrides as the scalar code: the zero written at i1-1 of a
later segment overrides the mu*dt written at i2+1 of the previous one); (2) per pass, the courant maximum with Python max/min semantics (NaN ignored); (3) forward-fill
over the passes with a cumulative max of 'last assigning pass'.  Same arithmetic per element as the scalar code (mu*dt, the mcheck expressions, the 1/ncourant scaling).

xpre_dev(mm, mu, dt, lmu) -> snap (LMO+1, JM+1, IM+1), ncour (LMO+1, JM+1) int64, lane (LMO+1, JM+1) bool; mm, mu (IM+1, JM+1, LMO+1) 1-based, lmu (IM+1, JM+1).
"""
import clouds_jax_env  # noqa: F401  (before jax: exactness flags)
import jax
import jax.numpy as jnp
from jax import lax

from oadvt2_ff import IM, JM, LMO

jax.config.update("jax_enable_x64", True)


def _pmax(a, b):          # Python max(a, b): returns a unless b > a (NaN b ignored)
    return jnp.where(b > a, b, a)


def _pmin0(x):            # Python min(0.0, x): 0.0 unless x < 0.0
    return jnp.where(x < 0.0, x, 0.0)


def xpre_dev(mm, mu, dt, lmu):
    NL, NJ, K = LMO, JM - 2, IM + 2            # passes l = 1..LMO, j = 2..JM-1; cells 0..IM+1 (0 and IM+1 padding)
    Lr = jnp.arange(1, LMO + 1)[:, None, None]
    # (LMO, NJ, IM+1) with cell index i = 0..IM
    mu_p = jnp.transpose(mu[:, 2:JM, 1:LMO + 1], (2, 1, 0))
    mm_p = jnp.transpose(mm[:, 2:JM, 1:LMO + 1], (2, 1, 0))
    lm_p = jnp.transpose(lmu[:, 2:JM], (1, 0))[None, :, :]            # (1, NJ, IM+1)
    pad = ((0, 0), (0, 0), (0, 1))
    mudt0 = jnp.pad(mu_p * dt, pad)                                    # (LMO, NJ, K): mu * dt, 0 at cell IM+1
    mmp = jnp.pad(mm_p, pad)
    A = jnp.pad(lm_p >= Lr, pad)                                       # active cells
    A = A.at[..., 0].set(False)                                        # cell 0 unused
    A = jnp.broadcast_to(A, (NL, NJ, K))
    Ap = jnp.pad(A, ((0, 0), (0, 0), (1, 0)))[..., :K]                 # A[c-1]
    An = jnp.pad(A, ((0, 0), (0, 0), (0, 1)))[..., 1:]                 # A[c+1]
    cell = jnp.arange(K)[None, None, :]
    mu_c = mudt0
    mu_prev = jnp.pad(mudt0, ((0, 0), (0, 0), (1, 0)))[..., :K]        # mudt0[c-1]
    mu_next = jnp.pad(mudt0, ((0, 0), (0, 0), (0, 1)))[..., 1:]        # mudt0[c+1]
    mm_next = jnp.pad(mmp, ((0, 0), (0, 0), (0, 1)))[..., 1:]          # mm[c+1]
    # ---------------------------------------------------------------- courant maximum of the pass
    # first cell (i = 1)
    m1, m2, mI = mudt0[..., 1], mudt0[..., 2], mudt0[..., IM]
    mc1 = jnp.where(m1 >= 0.0, mmp[..., 1] + _pmin0(mI - m1), -mmp[..., 2] - _pmin0(m1 - m2))
    r1 = jnp.where(A[..., 1], m1 / mc1, 0.0)
    cour = _pmax(jnp.zeros_like(r1), r1)
    # cells 2..IM-1 (active): neighbours i-1 (mu*dt if active else the zero written at the segment start) and i+1 (mu*dt)
    m_prev = jnp.where(Ap, mu_prev, 0.0)
    mc = jnp.where(mu_c >= 0.0, mmp + _pmin0(m_prev - mu_c), -mm_next - _pmin0(mu_c - mu_next))
    rr = mu_c / mc
    inner = A & (cell >= 2) & (cell <= IM - 1)
    for c in range(2, IM):                      # python-unrolled max in the scalar order (order is irrelevant for max; NaN ignored)
        cour = _pmax(cour, jnp.where(inner[..., c], rr[..., c], 0.0))
    # last cell (i = IM)
    mIm1 = jnp.where(A[..., IM - 1], mudt0[..., IM - 1], 0.0)
    m1_at_last = jnp.where(A[..., 2] & ~A[..., 1], 0.0, m1)            # a segment starting at cell 2 wrote zero into cell 1
    mcI = jnp.where(mI >= 0.0, mmp[..., IM] + _pmin0(mIm1 - mI), -mmp[..., 1] - _pmin0(mI - m1_at_last))
    rI = jnp.where(A[..., IM], mI / mcI, 0.0)
    cour = _pmax(cour, rI)
    scale = cour > 1.0
    ncour = jnp.where(scale, 1 + jnp.floor(cour).astype(jnp.int64), 1)
    zc = 1.0 / ncour
    lane = jnp.any(A, axis=-1)
    # ---------------------------------------------------------------- cells assigned by the pass and their final values
    Z = An & ~A                                                         # a segment starts at c+1 -> zero written into c
    Z = Z & (cell >= 1) & (cell <= IM - 1)
    E = (A | Ap) & (cell >= 1) & (cell <= IM)
    start = (cell == 1) | (cell == 2) | (cell == IM)
    assigned = (start | Z | E) & lane[..., None] & (cell >= 1) & (cell <= IM)
    v = jnp.where(Z, 0.0, mudt0)
    F = jnp.where(A & scale[..., None], v * zc[..., None], v)
    # ---------------------------------------------------------------- forward fill over the passes (order: l outer, j inner), stale entries
    NP = NL * NJ
    Ff = F.reshape(NP, K)
    asg = assigned.reshape(NP, K)
    pidx = jnp.arange(NP)[:, None]
    last = lax.cummax(jnp.where(asg, pidx, -1), axis=0)
    full = jnp.take_along_axis(Ff, jnp.clip(last, 0, NP - 1), axis=0)
    full = jnp.where(last >= 0, full, 0.0)
    full = jnp.where(lane.reshape(NP)[:, None], full, 0.0).reshape(NL, NJ, K)[..., :IM + 1]
    snap = jnp.zeros((LMO + 1, JM + 1, IM + 1)).at[1:LMO + 1, 2:JM, :].set(full)
    ncour_o = jnp.zeros((LMO + 1, JM + 1), jnp.int64).at[1:LMO + 1, 2:JM].set(jnp.where(lane, ncour, 0))
    lane_o = jnp.zeros((LMO + 1, JM + 1), bool).at[1:LMO + 1, 2:JM].set(lane)
    return snap, ncour_o, lane_o
