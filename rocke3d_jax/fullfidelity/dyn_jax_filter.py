"""D139: end-of-DYNAM filter chain (FLTRUV, fltry2, CONSERV_AMB_EXT, ADD_AM_AS_SOLIDBODY_ROTATION), calc_kea_3d and
COMPUTE_WSAVE in JAX; jnp ports of dyn_fltruv_ff / dyn_glue_ff with the same operation order.

Sequential reductions (Fortran SUM / explicit loops, strictly left to right) are lax.scan: `seqsum_jax` (the numpy
`seqsum`), the I-loop of FLTRUV's angular-momentum fix (order I=IM,1,...,IM-1) and the L sums.  Whole-field
elementwise parts are plain jnp expressions.
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from dyn_fltruv_ff import IM, JM, LM, NSHAP, BY4TON, DT, DT_XUFILTER, DT_XVFILTER


def seqsum_jax(a, axis=0):
    a = jnp.moveaxis(a, axis, 0)
    s, _ = lax.scan(lambda c, x: (c + x, None), a[0], a[1:])
    return s


def _shapiro_x(q):
    x = q
    for _ in range(NSHAP):
        x = ((jnp.roll(x, 1, axis=0) - x) - x) + jnp.roll(x, -1, axis=0)
    return x


def fltruv_jax(u, v, ma, dxyn, dxys):
    usave = u
    xu = (DT / DT_XUFILTER) * BY4TON
    xv = (DT / DT_XVFILTER) * BY4TON
    sl = slice(1, JM)
    u = u.at[:, sl, :].set(u[:, sl, :] - _shapiro_x(u[:, sl, :]) * xu)
    v = v.at[:, sl, :].set(v[:, sl, :] - _shapiro_x(v[:, sl, :]) * xv)
    ip1 = np.roll(np.arange(IM), -1)
    ms = (ma[:, ip1, 0:JM - 1] + ma[:, :, 0:JM - 1]) * dxyn[0:JM - 1][None, None, :]      # (LM,IM,JM-1)
    mn = (ma[:, ip1, 1:JM] + ma[:, :, 1:JM]) * dxys[1:JM][None, None, :]
    mmuv = (.5 * (ms + mn)).transpose(1, 2, 0)                                           # (IM,JM-1,LM)
    du = u[:, 1:JM, :] - usave[:, 1:JM, :]
    order = np.array([IM - 1] + list(range(IM - 1)))

    def body(c, xs):
        angm, mmuvs = c
        mm, d = xs
        return (angm - mm * d, mmuvs + mm), None
    z = jnp.zeros((JM - 1, LM))
    (angm, mmuvs), _ = lax.scan(body, (z, z), (mmuv[order], du[order]))
    return u.at[:, 1:JM, :].set(u[:, 1:JM, :] + (angm / mmuvs)[None, :, :]), v


def fltry2_jax(q, strength=1.0):
    yv = min(strength, 1.0) * BY4TON * ((-1) ** NSHAP)
    yn = q[:, 1:, :]
    h = IM // 2
    for _ in range(NSHAP):
        south = -jnp.concatenate([yn[h:, 0, :], yn[:h, 0, :]], axis=0)
        left = jnp.concatenate([south[:, None, :], yn[:, :-1, :]], axis=1)
        inner = ((left[:, :-1, :] - yn[:, :-1, :]) - yn[:, :-1, :]) + yn[:, 1:, :]
        yj = yn[:, -1, :]
        partner = jnp.concatenate([yj[h:], yj[:h]], axis=0)
        north = ((left[:, -1, :] - yj) - yj) - partner
        yn = jnp.concatenate([inner, north[:, None, :]], axis=1)
    return q.at[:, 1:, :].set(q[:, 1:, :] - yn * yv)


def conserv_amb_ext_jax(u, ma, dxyn, dxys, cosv, radius, omega):
    ip1 = np.roll(np.arange(IM), -1)
    a = (ma[:, :, 0:JM - 1] + ma[:, ip1, 0:JM - 1]) * dxyn[0:JM - 1][None, None, :]    # (LM,IM,JM-1)
    b = (ma[:, :, 1:JM] + ma[:, ip1, 1:JM]) * dxys[1:JM][None, None, :]
    cj = cosv[1:JM][None, None, :]
    w = (a + b) * (u.transpose(2, 0, 1)[:, :, 1:JM] + cj * radius * omega)
    s = seqsum_jax(w, axis=0)                                                           # (IM,JM-1)
    return jnp.zeros((IM, JM)).at[:, 1:JM].set(s * .5 * cj[0] * radius)


def add_am_jax(u, dam, masum, dxyn, dxys, cosv, radius):
    masumj = seqsum_jax(masum, axis=0)
    xj = jnp.zeros(JM).at[1:].set(cosv[1:] ** 2 * (masumj[0:JM - 1] * dxyn[0:JM - 1] + masumj[1:] * dxys[1:]))
    xglob = seqsum_jax(xj, axis=0)
    dueq = dam / (radius * xglob)
    return u.at[:, 1:, :].set(u[:, 1:, :] + (dueq * cosv[1:])[None, :, None])


@jax.jit
def filter_chain_jax(u, v, ma, masum, dxyn, dxys, cosv, radius, omega):
    u1, v1 = fltruv_jax(u, v, ma, dxyn, dxys)
    am1 = conserv_amb_ext_jax(u1, ma, dxyn, dxys, cosv, radius, omega)
    u2 = fltry2_jax(u1, 1.0)
    v2 = fltry2_jax(v1, 1.0)
    am2 = conserv_amb_ext_jax(u2, ma, dxyn, dxys, cosv, radius, omega)
    d = (am1 - am2).at[:, 0].set(am2[:, 0])
    damsum = seqsum_jax(seqsum_jax(d, axis=0), axis=0)
    u3 = add_am_jax(u2, damsum, masum, dxyn, dxys, cosv, radius)
    return u3, v2, damsum


# ----------------------------------------------------------------------------- calc_kea_3d, compute_wsave
@jax.jit
def calc_kea_3d_jax(u, v, byim):
    k = jnp.zeros_like(u).at[:, 1:, :].set(.5 * (u[:, 1:, :] * u[:, 1:, :] + v[:, 1:, :] * v[:, 1:, :]))
    r0 = k[:, 1:JM - 1, :]; r1 = k[:, 2:JM, :]
    new = k.at[:, 1:JM - 1, :].set(.25 * (((jnp.roll(r0, 1, axis=0) + r0) + jnp.roll(r1, 1, axis=0)) + r1))
    new = new.at[:, 0, :].set(seqsum_jax(k[:, 1, :], axis=0) * byim)
    return new.at[:, JM - 1, :].set(seqsum_jax(k[:, JM - 1, :], axis=0) * byim)


@jax.jit
def compute_wsave_jax(mws, t, pk, pedn, byaxyp, rgas, bygrav, dtsrc):
    pkt = pk.transpose(1, 2, 0)
    s = t[:, :, :LM - 1] * pkt[:, :, :LM - 1] + t[:, :, 1:] * pkt[:, :, 1:]
    pe = pedn.transpose(1, 2, 0)[:, :, 1:LM]
    return (((((mws[:, :, :LM - 1] * byaxyp[:, :, None]) * rgas) * 0.5) * s) * bygrav) / (dtsrc * pe)
