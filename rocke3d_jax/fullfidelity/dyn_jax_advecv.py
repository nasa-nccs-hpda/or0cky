"""D139: ADVECV in JAX (jitted whole-field jnp port of dyn_advecv_ff.advecv, same operation order).

Differences from the numpy port are structural only: in-place slice assignment -> .at[].set, the level loop of the
vertical advection (each level reads the old DUT of its own level only) -> three whole-array slice expressions with
the same per-element operation sequence, python loops over the two poles kept (static).  Fortran ordering of the
per-cell DUT/DVT updates is reproduced exactly as in the numpy port (IP1-role then I-role, cell IM reversed).
Geometry arrays are passed as a pytree (`advecv_geo(g)`), do_polefix is static.
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dyn_aflux_ff import IM, JM, LM


def advecv_geo(g):
    k = ('polwt', 'acor', 'dxyn', 'dxys', 'dxv', 'fcor', 'ravpn', 'ravps', 'siniv', 'cosiv')
    d = {n: jnp.asarray(np.asarray(g[n], dtype=np.float64)) for n in k}
    return d


def _roll_p(a):
    return jnp.roll(a, 1, axis=0)


def _hadv_component(wE, nS, sW, sE, d0):
    zero = jnp.zeros((IM, 1, wE.shape[2]))
    carry_ns = jnp.concatenate([zero, nS], axis=1)
    carry_sw = jnp.concatenate([zero, sW], axis=1)
    carry_se = jnp.concatenate([zero, sE], axis=1)
    nS_new = jnp.concatenate([nS, zero], axis=1)
    sW_new = jnp.concatenate([sW, zero], axis=1)
    sE_new = jnp.concatenate([sE, zero], axis=1)

    def ip1_seq(r):
        r = r + _roll_p(wE)
        r = r + _roll_p(carry_sw)
        r = r - _roll_p(sE_new)
        return r

    def i_seq(r):
        r = r - wE
        r = r + carry_ns
        r = r + carry_se
        r = r - nS_new
        r = r - sW_new
        return r
    a = i_seq(ip1_seq(d0))
    b = ip1_seq(i_seq(d0))
    return a.at[IM - 1].set(b[IM - 1])


def _coriolis_acc(alph, ur, vr):
    an = jnp.roll(alph, -1, axis=0)
    ud = (0. + alph * vr) + an * vr
    vd = (0. - alph * ur) - an * ur
    ud = ud.at[IM - 1].set((0. + alph[0] * vr[IM - 1]) + alph[IM - 1] * vr[IM - 1])
    vd = vd.at[IM - 1].set((0. - alph[0] * ur[IM - 1]) - alph[IM - 1] * ur[IM - 1])
    return ud, vd


def _acc2(wf, snf, twice):
    def ip1_seq(r):
        r = r + _roll_p(wf)
        return r + _roll_p(wf) if twice else r

    def i_seq(r):
        r = r - wf
        r = (r - wf) if twice else r
        r = r + snf
        return (r + snf) if twice else r
    zz = jnp.zeros_like(wf)
    a = i_seq(ip1_seq(zz)); b = ip1_seq(i_seq(zz))
    return a.at[IM - 1].set(b[IM - 1])


@jax.jit
def advecv_jax(dt1, u, v, mmean, mbefor, ut, vt, mafter, pu, pv, sd, spa, geo):
    """Same arguments/returns as dyn_advecv_ff.advecv with polefix=True (do_polefix=1)."""
    polwt, acor = geo['polwt'], geo['acor']
    dxyn, dxys, dxv = geo['dxyn'], geo['dxys'], geo['dxv']
    fcor, ravpn, ravps = geo['fcor'], geo['ravpn'], geo['ravps']
    sini, cosi = geo['siniv'], geo['cosiv']
    dt2 = dt1 / 2.
    dt8 = dt1 / 8.
    dt12 = dt1 / 12.
    dt24 = dt1 / 24.
    ip1 = np.roll(np.arange(IM), -1)

    def vmass_of(m):
        mt = m.transpose(1, 2, 0)
        return .5 * (((mt[:, 0:JM - 1, :] + mt[ip1][:, 0:JM - 1, :]) * dxyn[None, 0:JM - 1, None])
                     + ((mt[:, 1:JM, :] + mt[ip1][:, 1:JM, :]) * dxys[None, 1:JM, None]))
    vm0 = vmass_of(mbefor)
    ut = ut.at[:, 1:JM, :].set(ut[:, 1:JM, :] * vm0)
    vt = vt.at[:, 1:JM, :].set(vt[:, 1:JM, :] * vm0)
    uc = u.at[:, 1, :].set(polwt * u[:, 1, :] + (1 - polwt) * u[:, 2, :])
    vc = v.at[:, 1, :].set(polwt * v[:, 1, :] + (1 - polwt) * v[:, 2, :])
    uc = uc.at[:, JM - 1, :].set(polwt * u[:, JM - 1, :] + (1 - polwt) * u[:, JM - 2, :])
    vc = vc.at[:, JM - 1, :].set(polwt * v[:, JM - 1, :] + (1 - polwt) * v[:, JM - 2, :])
    j = slice(1, JM)
    jm1 = slice(0, JM - 1)
    flux = dt12 * (((pu[ip1][:, j, :] + pu[ip1][:, jm1, :]) + pu[:, j, :]) + pu[:, jm1, :])
    wE_u = flux * (uc[ip1][:, j, :] + uc[:, j, :])
    wE_v = flux * (vc[ip1][:, j, :] + vc[:, j, :])
    jr = slice(1, JM - 1)
    jr1 = slice(2, JM)
    f_ns = dt12 * (((pv[:, jr, :] + pv[ip1][:, jr, :]) + pv[:, jr1, :]) + pv[ip1][:, jr1, :])
    nS_u = f_ns * (uc[:, jr, :] + uc[:, jr1, :])
    nS_v = f_ns * (vc[:, jr, :] + vc[:, jr1, :])
    f_sw = dt24 * (((pu[ip1][:, jr, :] + pu[:, jr, :]) + pv[ip1][:, jr, :]) + pv[ip1][:, jr1, :])
    sW_u = f_sw * (uc[ip1][:, jr1, :] + uc[:, jr, :])
    sW_v = f_sw * (vc[ip1][:, jr1, :] + vc[:, jr, :])
    f_se = dt24 * ((((-pu[ip1][:, jr, :]) - pu[:, jr, :]) + pv[ip1][:, jr, :]) + pv[ip1][:, jr1, :])
    sE_u = f_se * (uc[:, jr1, :] + uc[ip1][:, jr, :])
    sE_v = f_se * (vc[:, jr1, :] + vc[ip1][:, jr, :])
    z = jnp.zeros((IM, JM - 1, LM))
    dut = jnp.zeros((IM, JM, LM)); dvt = jnp.zeros((IM, JM, LM))
    dut = dut.at[:, 1:JM, :].set(_hadv_component(wE_u, nS_u, sW_u, sE_u, z))
    dvt = dvt.at[:, 1:JM, :].set(_hadv_component(wE_v, nS_v, sW_v, sE_v, z))
    # polar rows (do_polefix=1)
    for pole in (0, 1):
        if pole == 0:
            hemi, jpo, jns, jv, jvs, jvn, wts = -1, 0, 1, 1, 1, 2, polwt
        else:
            hemi, jpo, jns, jv, jvs, jvn, wts = 1, JM - 1, JM - 2, JM - 1, JM - 2, JM - 1, 1. - polwt
        c = cosi[:, None]; s = (hemi * sini)[:, None]
        UP = {}; VP = {}
        for jj in (jvs, jvn):
            UP[jj] = c * u[:, jj, :] - s * v[:, jj, :]
            VP[jj] = c * v[:, jj, :] + s * u[:, jj, :]
        UP[jv] = wts * UP[jvs] + (1. - wts) * UP[jvn]
        VP[jv] = wts * VP[jvs] + (1. - wts) * VP[jvn]
        fl = dt8 * (((pu[ip1][:, jpo, :] + pu[:, jpo, :]) + pu[ip1][:, jns, :]) + pu[:, jns, :])
        fu = fl * (UP[jv][ip1] + UP[jv])
        fv = fl * (VP[jv][ip1] + VP[jv])
        sn = dt8 * (((pv[:, jvs, :] + pv[ip1][:, jvs, :]) + pv[:, jvn, :]) + pv[ip1][:, jvn, :])
        sn = sn * hemi
        snu = sn * (UP[jvs] + UP[jvn])
        snv = sn * (VP[jvs] + VP[jvn])
        du = _acc2(fu, snu, False)
        dv = _acc2(fv, snv, False)
        dmt = _acc2(fl, sn, True)
        du = du + (acor - 1.) * (du - dmt * UP[jv])
        dv = dv + (acor - 1.) * (dv - dmt * VP[jv])
        dut = dut.at[:, jv, :].set(c * du + s * dv)
        dvt = dvt.at[:, jv, :].set(c * dv - s * du)
    # vertical advection: each level reads only its own old DUT/DVT
    sdi = jnp.roll(sd, -1, axis=0)
    asdu = jnp.zeros((IM, JM, LM - 1))
    asdu = asdu.at[:, 1:JM, :].set(dt2 * (((sd[:, 0:JM - 1, :] + sdi[:, 0:JM - 1, :]) * ravpn[None, 0:JM - 1, None])
                                          + ((sd[:, 1:JM, :] + sdi[:, 1:JM, :]) * ravps[None, 1:JM, None])))
    res = []
    for dd, xx in ((dut, u), (dvt, v)):
        r = slice(1, JM)
        d0 = dd[:, r, 0] + asdu[:, r, 0] * (xx[:, r, 0] + xx[:, r, 1])
        dm = (dd[:, r, 1:LM - 1] - asdu[:, r, 0:LM - 2] * (xx[:, r, 0:LM - 2] + xx[:, r, 1:LM - 1]))
        dm = dm + asdu[:, r, 1:LM - 1] * (xx[:, r, 1:LM - 1] + xx[:, r, 2:LM])
        dl = dd[:, r, LM - 1] - asdu[:, r, LM - 2] * (xx[:, r, LM - 2] + xx[:, r, LM - 1])
        res.append(jnp.concatenate([d0[:, :, None], dm, dl[:, :, None]], axis=2))
    ut = ut.at[:, 1:JM, :].set(ut[:, 1:JM, :] + res[0])
    vt = vt.at[:, 1:JM, :].set(vt[:, 1:JM, :] + res[1])
    # Coriolis
    mm = mmean.transpose(1, 2, 0)
    spa_m = jnp.roll(spa, 1, axis=0)
    fd = jnp.zeros((IM, JM, LM))
    fd = fd.at[:, 0, :].set((-.5 * (spa_m[:, 0, :] + spa[:, 0, :])) * dxv[1])
    fd = fd.at[:, JM - 1, :].set((.5 * (spa_m[:, JM - 1, :] + spa[:, JM - 1, :])) * dxv[JM - 1])
    jj = slice(1, JM - 1)
    fd = fd.at[:, jj, :].set(fcor[None, jj, None] + (.25 * (spa_m[:, jj, :] + spa[:, jj, :]))
                             * (dxv[None, 1:JM - 1, None] - dxv[None, 2:JM, None]))
    pdt4 = dt8 * (mm[:, 0:JM - 1, :] + mm[:, 1:JM, :])
    alph = pdt4 * (fd[:, 1:JM, :] + fd[:, 0:JM - 1, :])
    ud, vd = _coriolis_acc(alph, u[:, 1:JM, :], v[:, 1:JM, :])
    dut = jnp.zeros((IM, JM, LM)).at[:, 1:JM, :].set(ud)
    dvt = jnp.zeros((IM, JM, LM)).at[:, 1:JM, :].set(vd)
    for jpo, jns, jrow in ((0, 1, 1), (JM - 1, JM - 2, JM - 1)):
        pdt4p = dt8 * (mm[:, jpo, :] + mm[:, jns, :])
        alp = pdt4p * (2 * fcor[jpo] + fcor[jns])
        udp, vdp = _coriolis_acc(alp[:, None, :], u[:, jrow:jrow + 1, :], v[:, jrow:jrow + 1, :])
        dut = dut.at[:, jrow, :].set(udp[:, 0, :]); dvt = dvt.at[:, jrow, :].set(vdp[:, 0, :])
    vm1 = vmass_of(mafter)
    ut = ut.at[:, 1:JM, :].set((ut[:, 1:JM, :] + dut[:, 1:JM, :]) / vm1)
    vt = vt.at[:, 1:JM, :].set((vt[:, 1:JM, :] + dvt[:, 1:JM, :]) / vm1)
    return ut, vt
