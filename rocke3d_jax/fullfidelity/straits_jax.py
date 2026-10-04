"""JAX (batched) STBDRA -- straits bottom and side drag, Stage 2, D69 (speed-first; see D62).

STBDRA (OSTRAITS.f:287-328) runs once per model step after STCONV (OCNDYN2.f:488). For each strait
it applies a bottom drag to the lowest layer, then a side drag to every layer, and decays the
cross-strait tracer gradients (20-day restoring to zero). Batched over straits (N) and layers
(1-indexed, padded to LMO+1 as in the other ports).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

LMO = 13
BDRAGX = 1.0
SDRAGX = 1e-1
SECONDS_PER_DAY = 86400.0
REDUCE = 1.0 / (SECONDS_PER_DAY * 20.0)


def stbdra_jax(dts, nmst, lmst, must, mmst, wist, dist, gxmst, sxmst):
    """Inputs: must, mmst, gxmst, sxmst (NMST, LMO+1) 1-indexed; lmst, wist, dist (NMST,).
    Returns (must, gxmst, sxmst) after the step."""
    N = nmst
    L = jnp.arange(LMO + 1)[None, :]
    act = (L >= 1) & (L <= lmst[:, None])
    bottom = L == lmst[:, None]
    w = wist[:, None]
    d = dist[:, None]
    # bottom drag on layer LMST(N)
    mb = mmst
    m_new_b = must * mb ** 2 / (mb ** 2 + dts * BDRAGX * jnp.abs(must) * w * d ** 2)
    must = jnp.where(bottom, m_new_b, must)
    # side drag on every layer 1..LMST(N)
    m_new = must * mmst * w / (mmst * w + dts * SDRAGX * jnp.abs(must) * d)
    must = jnp.where(act, m_new, must)
    gxmst = jnp.where(act, gxmst * (1.0 - REDUCE * dts), gxmst)
    sxmst = jnp.where(act, sxmst * (1.0 - REDUCE * dts), sxmst)
    return must, gxmst, sxmst


def stadvt_jax(am, mm1, mm2, mmst, rmst, rxst, rzst, rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2,
               x1, y1, x2, y2, qlimit):
    """Vectorized STADVT (OSTRAITS.f:171-285) over any batch shape. am, mm1, mm2, mmst, rmst, rxst,
    rzst, and the end-1/end-2 moments (rm1, rx1, ry1, rz1; rm2, rx2, ry2, rz2) broadcast together.
    Returns (rmst, rxst, rzst, rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2, dmass) where dmass is the
    OLN increment (FM1 + FM2). qlimit is a Python bool (STADV's S call uses True, G uses False).

    Branch AM >= 0 (flux from box 1 to box 2) and AM < 0 (flux from box 2 to box 1) are both computed
    and merged with where, as in the Fortran GO TO 200 / 300 structure."""
    tiny = jnp.finfo(jnp.float64).tiny
    # ---- AM >= 0 ----
    a1p = am / mm1
    fm1p = a1p * (rm1 + (1 - a1p) * (x1 * rx1 + y1 * ry1))
    fz1p = a1p * rz1
    a2p = am / mmst
    fm2p = a2p * (rmst + (1 - a2p) * rxst)
    fz2p = a2p * rzst
    rx1p = rx1 * (1 - a1p) * (1 - a1p * x1 * x1)
    ry1p = ry1 * (1 - a1p) * (1 - a1p * y1 * y1)
    rxstp = rxst * (1 - 2 * a2p) - fm1p + fm2p
    rx2p = rx2 + x2 * (fm2p - (rm2 - x2 * rx2) * am / mm2)
    ry2p = ry2 + y2 * (fm2p - (rm2 - y2 * ry2) * am / mm2)
    # ---- AM < 0 ----
    a1n = am / mmst
    fm1n = a1n * (rmst - (1 + a1n) * rxst)
    fz1n = a1n * rzst
    a2n = am / mm2
    fm2n = a2n * (rm2 + (1 + a2n) * (x2 * rx2 + y2 * ry2))
    fz2n = a2n * rz2
    rx1n = rx1 - x1 * (fm1n - (rm1 - x1 * rx1) * am / mm1)
    ry1n = ry1 - y1 * (fm1n - (rm1 - y1 * ry1) * am / mm1)
    rxstn = rxst * (1 + 2 * a1n) + fm1n - fm2n
    rx2n = rx2 * (1 + a2n) * (1 + a2n * x2 * x2)
    ry2n = ry2 * (1 + a2n) * (1 + a2n * y2 * y2)
    pos = am >= 0.0
    fm1 = jnp.where(pos, fm1p, fm1n)
    fz1 = jnp.where(pos, fz1p, fz1n)
    fm2 = jnp.where(pos, fm2p, fm2n)
    fz2 = jnp.where(pos, fz2p, fz2n)
    rx1 = jnp.where(pos, rx1p, rx1n)
    ry1 = jnp.where(pos, ry1p, ry1n)
    rxst = jnp.where(pos, rxstp, rxstn)
    rx2 = jnp.where(pos, rx2p, rx2n)
    ry2 = jnp.where(pos, ry2p, ry2n)
    # ---- common: new tracer masses and moments (OCNKPP.f 300-) ----
    rm1 = rm1 - fm1
    rz1 = rz1 - fz1
    rm2 = rm2 + fm2
    rz2 = rz2 + fz2
    rmst = rmst + (fm1 - fm2)
    rzst = rzst + (fz1 - fz2)
    dmass = fm1 + fm2
    if qlimit:
        rxy = jnp.abs(rx1) + jnp.abs(ry1)
        f1 = jnp.where(rxy > rm1, rm1 / (rxy + tiny), 1.0)
        rx1 = rx1 * f1
        ry1 = ry1 * f1
        rz1 = jnp.where(jnp.abs(rz1) > rm1, jnp.sign(rz1) * rm1, rz1)
        rxy = jnp.abs(rx2) + jnp.abs(ry2)
        f2 = jnp.where(rxy > rm2, rm2 / (rxy + tiny), 1.0)
        rx2 = rx2 * f2
        ry2 = ry2 * f2
        rz2 = jnp.where(jnp.abs(rz2) > rm2, jnp.sign(rz2) * rm2, rz2)
        rxst = jnp.where(jnp.abs(rxst) > rmst, jnp.sign(rxst) * rmst, rxst)
        rzst = jnp.where(jnp.abs(rzst) > rmst, jnp.sign(rzst) * rmst, rzst)
    return rmst, rxst, rzst, rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2, dmass
