"""D143: AADVT (QUS_DRV.f temperature advection: pole fill, X(half) Y Z X(half) sweeps) and the 1-D QUS kernels adv1d /
advection_1D_custom in JAX (jitted port of dyn_aadvt_ff.py / dyn_adv1d_ff.py, qlimit=.false. only: the qlimit path of AADVT
is dead and stays in numpy).

Layout.  The numpy port batches lines as (rows, cells); here the cell axis is simply the array axis of the (IM,JM,LM)
fields (X: axis 0 over rows J=2..JM-1, Y: axis 1, Z: axis 2), so no transposes.  The upwind cell n(n) is selected with
jnp.where(dm<0, a_next, a) with a_next the (cyclic / clamped) neighbour: a pure selection, no gather.
Per-row Courant count (QUS_DRV.f:237-258, 508-530): lax.while_loop over trial = 1..20 with an inner fori_loop over the
sub-steps and an `active` mask; the sub-stepping loop is a lax.fori_loop to the largest nstep with a per-row mask
(k < nstep) so that rows with a smaller nstep stop updating (an unmasked max-nstep run changes the result; `masked=False`
reproduces that error for the mutation tests).  Reductions (the polar-box sums, the fqu/fqv accumulation over l) are
lax.scan carries in Fortran order, never jnp.sum / cumsum.  Constants (the cube exponent) are traced arguments (XLA
simplifies pow(x,3) on closure constants).  Needs the FMA-free XLA flag (dyn_jax_env).

aadvt_jax(dt, mm, rm, rmom, mu, mv, mw, masked=True) -> dict(mm, rm, rmom, fqu, fqv, nsx1, nsx2, nsz, bad, stages)
(jax arrays; `bad` True where the Fortran would stop_model: courmax>1 at nstep=20).
"""
import functools

import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from dyn_adv1d_ff import XDIR, YDIR, ZDIR, MX, MZ, MZZ, IHMOMS

IM, JM, LM = 72, 46, 40
BYIM = 1.0 / 72.0
NSTEP_MAX = 20


def seqsum(a, axis=0):
    a = jnp.moveaxis(a, axis, 0)
    s, _ = lax.scan(lambda c, x: (c + x, None), a[0], a[1:])
    return s


# ------------------------------------------------------------------------------------------ adv1d kernels
def _next(a, ax, cyclic):
    if cyclic:
        return jnp.roll(a, -1, axis=ax)
    return jnp.concatenate([lax.slice_in_dim(a, 1, a.shape[ax], axis=ax), lax.slice_in_dim(a, a.shape[ax] - 1, a.shape[ax], axis=ax)], axis=ax)


def _prev(a, ax, cyclic):
    if cyclic:
        return jnp.roll(a, 1, axis=ax)
    z = jnp.zeros_like(lax.slice_in_dim(a, 0, 1, axis=ax))
    return jnp.concatenate([z, lax.slice_in_dim(a, 0, a.shape[ax] - 1, axis=ax)], axis=ax)


def adv1d_jax(s, sm, mass, dm, dirv, ax, cyclic, c3):
    """One adv1d / advection_1D_custom step (qlimit False).  s, mass, dm: arrays with the cell axis `ax`; sm (9,...) the
    moments.  Returns (s, sm, mass, f)."""
    mx, my, mz, mxx, myy, mzz, mxy, myz, mzx = dirv
    neg = dm < 0.
    sel = lambda a: jnp.where(neg, _next(a, ax, cyclic), a)
    frac1 = jnp.where(neg, 1., -1.)
    ms = sel(mass)
    fracm = dm / ms
    fracm = jnp.where(ms <= 0., 0., fracm)
    frac1 = fracm + frac1
    sn = sel(s)
    sx = sel(sm[mx])
    sxx = sel(sm[mxx])
    f = fracm * (sn - frac1 * (sx - (frac1 + fracm) * sxx))
    # slope stage
    fm = [None] * 9
    fm[mx] = dm * (fracm * fracm * (sx - 3. * frac1 * sxx) - 3. * f)
    fm[mxx] = dm * (dm * jnp.power(fracm, c3) * sxx - 5. * (dm * f + fm[mx]))
    sxy = sel(sm[mxy])
    fm[my] = fracm * (sel(sm[my]) - frac1 * sxy)
    fm[mxy] = dm * (fracm * fracm * sxy - 3. * fm[my])
    szx = sel(sm[mzx])
    fm[mz] = fracm * (sel(sm[mz]) - frac1 * szx)
    fm[mzx] = dm * (fracm * fracm * szx - 3. * fm[mz])
    fm[myy] = fracm * sel(sm[myy])
    fm[mzz] = fracm * sel(sm[mzz])
    fm[myz] = fracm * sel(sm[myz])
    # update stage
    P = lambda a: _prev(a, ax, cyclic)
    dmm, fmm = P(dm), P(f)
    tmp = mass + dmm
    mnew = tmp - dm
    bymnew = 1. / mnew
    dm2 = dmm + dm
    tmp = s + fmm
    s2 = tmp - f
    m0 = mass
    fmx_m, fmxx_m = P(fm[mx]), P(fm[mxx])
    n = [None] * 9
    n[mx] = (sm[mx] * m0 - 3. * (-dm2 * s2 + m0 * (fmm + f)) + (fmx_m - fm[mx])) * bymnew
    n[mxx] = (sm[mxx] * m0 * m0
              + 2.5 * s2 * (m0 * m0 - mnew * mnew - 3. * dm2 * dm2)
              + 5. * (m0 * (m0 * (fmm - f) - fmx_m - fm[mx]) + dm2 * n[mx] * mnew)
              + (fmxx_m - fm[mxx])) * (bymnew * bymnew)
    fmy_m = P(fm[my])
    n[my] = sm[my] + fmy_m - fm[my]
    n[mxy] = (sm[mxy] * m0 - 3. * (-dm2 * n[my] + m0 * (fmy_m + fm[my])) + (P(fm[mxy]) - fm[mxy])) * bymnew
    fmz_m = P(fm[mz])
    n[mz] = sm[mz] + fmz_m - fm[mz]
    n[mzx] = (sm[mzx] * m0 - 3. * (-dm2 * n[mz] + m0 * (fmz_m + fm[mz])) + (P(fm[mzx]) - fm[mzx])) * bymnew
    n[myy] = sm[myy] + P(fm[myy]) - fm[myy]
    n[mzz] = sm[mzz] + P(fm[mzz]) - fm[mzz]
    n[myz] = sm[myz] + P(fm[myz]) - fm[myz]
    dead = mnew <= 0.
    s2 = jnp.where(dead, 0., s2)
    smn = jnp.stack([jnp.where(dead, 0., x) for x in n])
    return s2, smn, mnew, f


# ------------------------------------------------------------------------------------------ Courant counts
def _trials(count_fn, shape_rows):
    """while_loop over trial=1..20: count_fn(trial_float) -> courmax per row (all rows); the first trial with courmax<=1
    fixes a row's nstep and courmax (rows already settled are not touched)."""
    def cond(c):
        trial, nstep, courmax, active = c
        return (trial <= NSTEP_MAX) & jnp.any(active)

    def body(c):
        trial, nstep, courmax, active = c
        cm = count_fn(trial.astype(jnp.float64))
        nstep = jnp.where(active, trial, nstep)
        courmax = jnp.where(active, cm, courmax)
        active = active & (cm > 1.)
        return trial + 1, nstep, courmax, active
    init = (jnp.asarray(1), jnp.zeros(shape_rows, jnp.int64), jnp.full(shape_rows, 2.), jnp.ones(shape_rows, bool))
    _, nstep, courmax, _ = lax.while_loop(cond, body, init)
    return nstep, courmax


def courant_x(mu, mass):
    """mu, mass (IM, R) (rows on axis 1) -> nstep (R,), courmax (R,)."""
    def count(trial):
        am = mu * (1. / trial)

        def sub(ns, c):
            mi, cm = c
            mip1 = jnp.roll(mi, -1, axis=0)
            cand = jnp.where(am > 0., am / mi, -am / mip1)
            cm = jnp.maximum(cm, jnp.max(cand, axis=0))
            mi2 = mi + (jnp.roll(am, 1, axis=0) - am)
            return jnp.where(ns < trial, mi2, mi), cm
        # ns runs 1..trial: in the numpy port the mass update follows the courmax of every sub-step except the last
        return lax.fori_loop(1, trial.astype(jnp.int64) + 1, lambda ns, c: _subx(ns, c, am, trial), (mass, jnp.zeros(mu.shape[1:])))[1]
    return _trials(count, mu.shape[1:])


def _subx(ns, c, am, trial):
    mi, cm = c
    mip1 = jnp.roll(mi, -1, axis=0)
    cand = jnp.where(am > 0., am / mi, -am / mip1)
    cm = jnp.maximum(cm, jnp.max(cand, axis=0))
    mi2 = mi + (jnp.roll(am, 1, axis=0) - am)
    return jnp.where(ns < trial, mi2, mi), cm


def courant_z(cmz, mass):
    """cmz (R, LM-1) vertical flux (columns on axis 0), mass (R, LM) -> nstep, courmax."""
    def count(trial):
        c = cmz * (1. / trial)

        def sub(ns, cc):
            ml, cmax = cc
            cand = jnp.where(c > 0., c / ml[:, :LM - 1], -c / ml[:, 1:])
            cmax = jnp.maximum(cmax, jnp.max(cand, axis=1))
            ml2 = ml.at[:, 1:].set(ml[:, 1:] + c)
            ml2 = ml2.at[:, :LM - 1].set(ml2[:, :LM - 1] - c)
            return jnp.where(ns < trial, ml2, ml), cmax
        return lax.fori_loop(1, trial.astype(jnp.int64) + 1, lambda ns, cc: sub_z(ns, cc, c, trial), (mass, jnp.zeros(mass.shape[0])))[1]
    return _trials(count, mass.shape[:1])


def sub_z(ns, cc, c, trial):
    ml, cmax = cc
    cand = jnp.where(c > 0., c / ml[:, :LM - 1], -c / ml[:, 1:])
    cmax = jnp.maximum(cmax, jnp.max(cand, axis=1))
    ml2 = ml.at[:, 1:].set(ml[:, 1:] + c)
    ml2 = ml2.at[:, :LM - 1].set(ml2[:, :LM - 1] - c)
    return jnp.where(ns < trial, ml2, ml), cmax


# ------------------------------------------------------------------------------------------ sweeps
@functools.partial(jax.jit, static_argnames=('masked',))
def aadvtx_jax(rm, rmom, mass, mu, fqu, c3, masked=True):
    """QUS_DRV.f:198-316 on rows J=2..JM-1.  Returns (rm, rmom, mass, fqu, nstep (nj,LM), bad)."""
    nj = JM - 2
    js = slice(1, JM - 1)
    mu_r = mu[:, js, :]
    s = rm[:, js, :]
    sm = rmom[:, :, js, :]
    ma = mass[:, js, :]
    nstep, courmax = courant_x(mu_r.reshape(IM, nj * LM), ma.reshape(IM, nj * LM))
    nstep = nstep.reshape(nj, LM)
    courmax = courmax.reshape(nj, LM)
    bad = jnp.any(courmax > 1.)
    nused = nstep if masked else jnp.full_like(nstep, jnp.max(nstep))
    by = 1. / nused.astype(jnp.float64)
    am = mu_r * by[None]

    def body(k, c):
        s, sm, ma, h = c
        s2, sm2, ma2, f = adv1d_jax(s, sm, ma, am, XDIR, 0, True, c3)
        act = (k < nused)[None]
        return (jnp.where(act, s2, s), jnp.where(act[None], sm2, sm), jnp.where(act, ma2, ma), jnp.where(act, h + f, h))
    s, sm, ma, h = lax.fori_loop(0, jnp.max(nused), body, (s, sm, ma, jnp.zeros_like(s)))
    # fqu(:,j) += hfqu(:,j,l), l sequential
    f0 = fqu[:, js]
    f1, _ = lax.scan(lambda c, x: (c + x, None), f0, jnp.moveaxis(h, 2, 0))
    return (rm.at[:, js, :].set(s), rmom.at[:, :, js, :].set(sm), mass.at[:, js, :].set(ma), fqu.at[:, js].set(f1),
            nstep, bad)


@functools.partial(jax.jit, static_argnames=('masked',))
def aadvtz_jax(rm, rmom, mass, mw, c3, masked=True):
    """QUS_DRV.f:474-573: all IM*JM columns.  mw (IM,JM,LM) with mw[:,:,LM-1]=0.  Returns (rm, rmom, mass, nstep (IM,JM), bad)."""
    B = IM * JM
    cm0 = mw.reshape(B, LM)
    nstep, courmax = courant_z(cm0[:, :LM - 1], mass.reshape(B, LM))
    bad = jnp.any(courmax > 1.)
    nused = nstep if masked else jnp.full_like(nstep, jnp.max(nstep))
    by = 1. / nused.astype(jnp.float64)
    cm = (cm0 * by[:, None]).at[:, LM - 1].set(0.).reshape(IM, JM, LM)
    nu = nused.reshape(IM, JM)

    def body(k, c):
        s, sm, ma = c
        s2, sm2, ma2, _ = adv1d_jax(s, sm, ma, cm, ZDIR, 2, True, c3)
        act = (k < nu)[:, :, None]
        return jnp.where(act, s2, s), jnp.where(act[None], sm2, sm), jnp.where(act, ma2, ma)
    s, sm, ma = lax.fori_loop(0, jnp.max(nused), body, (rm, rmom, mass))
    return s, sm, ma, nstep.reshape(IM, JM), bad


@jax.jit
def aadvty_jax(rm, rmom, mass, bm, c3):
    """QUS_DRV.f:318-471: Y sweep with polar-box scaling / averaging.  bm = MFLX (IM,JM,LM) (last row zeroed here).
    Returns (rm, rmom, mass, fqv)."""
    S, N = 0, JM - 1
    mass = mass.at[:, S, :].set(mass[:, S, :] * IM)
    m_sp = mass[0, S, :]
    rm = rm.at[:, S, :].set(rm[:, S, :] * IM)
    rm_sp = rm[0, S, :]
    rmom = rmom.at[:, :, S, :].set(rmom[:, :, S, :] * IM)
    rzm_sp = rmom[MZ, 0, S, :]
    rzzm_sp = rmom[MZZ, 0, S, :]
    mass = mass.at[:, N, :].set(mass[:, N, :] * IM)
    m_np = mass[0, N, :]
    rm = rm.at[:, N, :].set(rm[:, N, :] * IM)
    rm_np = rm[0, N, :]
    rmom = rmom.at[:, :, N, :].set(rmom[:, :, N, :] * IM)
    rzm_np = rmom[MZ, 0, N, :]
    rzzm_np = rmom[MZZ, 0, N, :]
    for n in IHMOMS:
        rmom = rmom.at[n, :, S, :].set(0.)
        rmom = rmom.at[n, :, N, :].set(0.)
    bm = bm.at[:, N, :].set(0.)
    s2, sm2, ma2, f_j = adv1d_jax(rm, rmom, mass, bm, YDIR, 1, False, c3)
    rm, rmom, mass = s2, sm2, ma2
    for n in IHMOMS:
        rmom = rmom.at[n, :, S, :].set(0.)
        rmom = rmom.at[n, :, N, :].set(0.)

    def avg(x, ref):                                            # (ref + sum_i (x_i - ref)) * BYIM, left to right
        return (ref + seqsum(x - ref[None], 0)) * BYIM
    mass = mass.at[:, S, :].set(avg(mass[:, S, :], m_sp))
    rm = rm.at[:, S, :].set(avg(rm[:, S, :], rm_sp))
    rmom = rmom.at[MZ, :, S, :].set(avg(rmom[MZ, :, S, :], rzm_sp))
    rmom = rmom.at[MZZ, :, S, :].set(avg(rmom[MZZ, :, S, :], rzzm_sp))
    mass = mass.at[:, N, :].set(avg(mass[:, N, :], m_np))
    rm = rm.at[:, N, :].set(avg(rm[:, N, :], rm_np))
    rmom = rmom.at[MZ, :, N, :].set(avg(rmom[MZ, :, N, :], rzm_np))
    rmom = rmom.at[MZZ, :, N, :].set(avg(rmom[MZZ, :, N, :], rzzm_np))
    fqv, _ = lax.scan(lambda c, x: (c + x, None), jnp.zeros((IM, JM)), jnp.moveaxis(f_j, 2, 0))
    fqv = fqv.at[:, N].set(0.)
    return rm, rmom, mass, fqv


@jax.jit
def aadvt_pre(dt, mm, rm, rmom, mu, mv, mw):
    """Pole fill, concentration -> mass units, and the three flux arrays (QUS_DRV.f:70-196)."""
    for j in (0, JM - 1):
        rm = rm.at[1:, j, :].set(rm[0, j, :][None, :])
        rmom = rmom.at[:, 1:, j, :].set(rmom[:, 0, j, :][:, None, :])
    rm = rm * mm
    rmom = rmom * mm[None]
    mflx_x = mu * (.5 * dt)
    mflx_y = jnp.zeros((IM, JM, LM)).at[:, :JM - 1, :].set(mv[:, 1:JM, :] * dt)
    mflx_z = jnp.zeros((IM, JM, LM)).at[:, :, :LM - 1].set(mw[:, :, :LM - 1] * (-dt))
    return rm, rmom, mflx_x, mflx_y, mflx_z


@jax.jit
def aadvt_post(rm, rmom, mm):
    by = 1. / mm
    return rm * by, rmom * by[None]


def aadvt_jax(dt, mm, rm, rmom, mu, mv, mw, masked=True):
    """QUS_DRV.f:70-196 AADVT (qlimit=.false.).  Inputs not modified."""
    c3 = jnp.asarray(3.)
    fqu0 = jnp.zeros((IM, JM))
    rm, rmom, mfx, mfy, mfz = aadvt_pre(float(dt), mm, rm, rmom, mu, mv, mw)
    rm, rmom, mm, fqu, nsx1, b1 = aadvtx_jax(rm, rmom, mm, mfx, fqu0, c3, masked=masked)
    st1 = (rm, rmom, mm)
    rm, rmom, mm, fqv = aadvty_jax(rm, rmom, mm, mfy, c3)
    st2 = (rm, rmom, mm)
    rm, rmom, mm, nsz, b3 = aadvtz_jax(rm, rmom, mm, mfz, c3, masked=masked)
    st3 = (rm, rmom, mm)
    rm, rmom, mm, fqu, nsx2, b2 = aadvtx_jax(rm, rmom, mm, mfx, fqu, c3, masked=masked)
    rm, rmom = aadvt_post(rm, rmom, mm)
    return dict(mm=mm, rm=rm, rmom=rmom, fqu=fqu, fqv=fqv, nsx1=nsx1, nsx2=nsx2, nsz=nsz, bad=b1 | b2 | b3,
                stages=dict(x1=st1, y=st2, z=st3))
