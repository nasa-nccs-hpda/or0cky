"""D144: QDYNAM (moisture advection, QUS3D.f AADVQ0 + AADVQ + QDYNAM glue) in JAX: jitted port of dyn_aadvq_ff.py.

What is JAX
  AADVQ0 (cycle counts): the data-dependent NCYC trial loop (lax.while_loop over trials with a nested while over the NCYC
  sub-cycles, REAL*4 `1./ncyc`), the per-level NCYCXY search (while loops, vmapped over the 40 levels: levels are independent),
  the per-level/per-row XSTEP counts (NSTEPX), the flux scalings by ncyc/ncycxy and the flow-out-both-sides masks
  (the numpy i_y/i_z lists become boolean masks).
  AADVQ: the NCYC cycle loop, the L=1..LM+1 level loop (fori_loop) with the per-level NCYCXY inner while (y checkflux,
  aadvqy, aadvqx with per-row NSTEPX masked sub-steps up to the largest count), the z checkflux and the interface sweep
  aadvqz whose carry (mwdn, fdn, fdn0, fmomdn) lives in the level loop (the "per-level vertical carry").
  Reductions (polar sums, running accumulations) are lax.scan carries in Fortran order, never jnp.sum on floats.
What stays numpy
  The extra-column vertical advection branch (AADVQ0 lminzij<LM columns: nstepz_extra / mw_extra partition, zstep, and the
  aadvqz_column advection after every cycle).  If AADVQ0 flags `do_z_extra` (never in the real windows, reached by the x8
  stress dumps) the call falls back: numpy AADVQ0 for the counts and the extra-column bookkeeping, the main cycle sweeps
  still JAX (one jitted cycle per call, numpy aadvqz_column between cycles).  Diagnostics sbf/sbm/sfbm/scf/scm/sfcm/scf3d
  of the numpy port are not computed (no state feedback).
Same numerical conventions as the numpy port (see its docstring: REAL*4 reciprocal traps, np.power(fracm,3) as pow with the
exponent a traced argument).  Needs the FMA-free XLA flag (dyn_jax_env).

qdynam_jax(qgeo, q, qmom, maold, mus, mvs, mws) -> dict(q, qmom, mus, mvs, mws, ncyc, ncycxy, nstepx, do_z_extra, err)
(numpy arrays in, numpy arrays out; err is None or the numpy port's QusError text, raised by the caller).
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import dyn_aadvq_ff as aq

IM, JM, LM = 72, 46, 40
MX, MY, MZ, MXX, MYY, MZZ, MXY, MZX, MYZ = range(9)
IHMOMS = aq.IHMOMS
XSPEC, YSPEC, ZSPEC = aq.XSPEC, aq.YSPEC, aq.ZSPEC
NCMAX = aq.NCMAX
NSTEPMAX_X = aq.NSTEPMAX_X
MRAT_LIMH = aq.MRAT_LIMH
MRAT_LIMY = aq.MRAT_LIMY
F32 = jnp.float32


def inv32(n):
    """REAL*8 = 1./n with n integer: the quotient is formed in REAL*4."""
    return (F32(1.0) / jnp.asarray(n).astype(F32)).astype(jnp.float64)


def seqsum(a, axis=0):
    a = jnp.moveaxis(a, axis, 0)
    s, _ = lax.scan(lambda c, x: (c + x, None), a[0], a[1:])
    return s


def seqacc(init, xs, sign=1.0):
    """init + xs[0] + xs[1] + ... (sign=1) or init - xs[0] - xs[1] ... (sign=-1), left to right."""
    if sign > 0:
        return lax.scan(lambda c, x: (c + x, None), init, xs)[0]
    return lax.scan(lambda c, x: (c - x, None), init, xs)[0]


# ---------------------------------------------------------------------------------------------- kernels
def checkflux(aml, amr, m, rm, rxm, rxxm):
    a = amr / m
    fr = a * (rm + (1. - a) * (rxm + (1. - 2. * a) * rxxm))
    a = aml / m
    fl = a * (rm - (1. + a) * (rxm - (1. + 2. * a) * rxxm))
    fire = (rm + fl - fr) <= 0.
    return jnp.where(fire, 0., rxm), jnp.where(fire, 0., rxxm), fire


def face_flux(a, mL, rmL, momL, mR, rmR, momR, rm_left, rm_right, spec, c3, strict_neg=True):
    p, pp, aa, aap, bb, bbp, q1, q2, q3 = spec
    neg = a < 0.
    mass = jnp.where(neg, mR, mL)
    rmu = jnp.where(neg, rmR, rmL)
    mom = jnp.where(neg[None], momR, momL)
    frac1 = jnp.where(neg, 1., -1.)
    fracm = a / mass
    frac1 = fracm + frac1
    mp, mpp = mom[p], mom[pp]
    fe = fracm * (rmu - frac1 * (mp - (frac1 + fracm) * mpp))
    fm = [None] * 9
    fm[p] = a * (fracm * fracm * (mp - 3. * frac1 * mpp) - 3. * fe)
    fm[pp] = a * (a * jnp.power(fracm, c3) * mpp - 5. * (a * fe + fm[p]))
    fm[aa] = fracm * (mom[aa] - frac1 * mom[aap])
    fm[aap] = a * (fracm * fracm * mom[aap] - 3. * fm[aa])
    fm[bb] = fracm * (mom[bb] - frac1 * mom[bbp])
    fm[bbp] = a * (fracm * fracm * mom[bbp] - 3. * fm[bb])
    fm[q1] = fracm * mom[q1]
    fm[q2] = fracm * mom[q2]
    fm[q3] = fracm * mom[q3]
    fe0 = fe
    fe_pass = fe
    fpp_ = fm[p]
    fppp = fm[pp]
    pos = a > 0.
    negm = neg if strict_neg else ~pos
    c1 = pos & (fe < 0.)
    c2 = pos & ~(fe < 0.) & (fe > rm_left)
    c3_ = negm & (fe > 0.)
    c4 = negm & ~(fe > 0.) & (fe < -rm_right)
    fe_o = fe
    fe = jnp.where(c1, 0., fe_o)
    fe_pass = jnp.where(c1, 0., fe_pass)
    fpp_ = jnp.where(c1, 0., fpp_)
    fppp = jnp.where(c1, 0., fppp)
    fe = jnp.where(c2, rm_left, fe)
    fe_pass = jnp.where(c2, rm_left, fe_pass)
    fpp_ = jnp.where(c2, a * (-3. * rm_left), fpp_)
    fppp = jnp.where(c2, a * (-5. * (a * rm_left + (a * (-3. * rm_left)))), fppp)
    fe = jnp.where(c3_, 0., fe)
    fe0 = jnp.where(c3_, 0., fe0)
    fm[p] = jnp.where(c3_, 0., fm[p])
    fm[pp] = jnp.where(c3_, 0., fm[pp])
    fe = jnp.where(c4, -rm_right, fe)
    fe0 = jnp.where(c4, -rm_right, fe0)
    fp4 = a * (-3. * (-rm_right))
    fm[p] = jnp.where(c4, fp4, fm[p])
    fm[pp] = jnp.where(c4, a * (-5. * (a * (-rm_right) + fp4)), fm[pp])
    return dict(fe=fe, fe0=fe0, fe_pass=fe_pass, fmom=jnp.stack(fm), fp_pass=fpp_, fpp_pass=fppp)


def cell_update(rm, mom, mass, aw, fw, fw0, fmw, ae, fe, fe0, fme, spec):
    p, pp, aa, aap, bb, bbp, q1, q2, q3 = spec
    mold = mass
    mnew = mold + aw - ae
    bymnew = 1. / mnew
    dm2 = aw + ae
    rm0 = rm + fw0 - fe0
    rmn = rm + fw - fe
    new = [None] * 9
    new[p] = (mom[p] * mold - 3. * (-dm2 * rm0 + mold * (fw0 + fe0)) + (fmw[p] - fme[p])) * bymnew
    new[pp] = (mom[pp] * mold * mold + 2.5 * rm0 * (mold * mold - mnew * mnew - 3. * dm2 * dm2)
               + 5. * (mold * (mold * (fw0 - fe0) - fmw[p] - fme[p]) + dm2 * new[p] * mnew)
               + (fmw[pp] - fme[pp])) * (bymnew * bymnew)
    new[aa] = mom[aa] + fmw[aa] - fme[aa]
    new[aap] = (mom[aap] * mold - 3. * (-dm2 * new[aa] + mold * (fmw[aa] + fme[aa])) + (fmw[aap] - fme[aap])) * bymnew
    new[bb] = mom[bb] + fmw[bb] - fme[bb]
    new[bbp] = (mom[bbp] * mold - 3. * (-dm2 * new[bb] + mold * (fmw[bb] + fme[bb])) + (fmw[bbp] - fme[bbp])) * bymnew
    for q in (q1, q2, q3):
        new[q] = mom[q] + fmw[q] - fme[q]
    clean = rmn <= 0.
    rmn = jnp.where(clean, 0., rmn)
    new = jnp.where(clean[None], 0., jnp.stack(new))
    return rmn, new, mnew


def west_moments(f, spec):
    return f['fmom'].at[spec[0]].set(f['fp_pass']).at[spec[1]].set(f['fpp_pass'])


# ---------------------------------------------------------------------------------------------- sweeps (one level)
def aadvqx_j(rm, rmom, mass, mu, nstep, c3):
    """aadvqx for one level: rows 2..JM-1 only, per-row nstep via masked sub-steps up to the largest count."""
    jj = jnp.arange(JM)
    rowact = (jj >= 1) & (jj <= JM - 2)
    nst = jnp.where(rowact, nstep, 1).astype(jnp.float64)
    nmax = jnp.max(jnp.where(rowact, nstep, 0))
    am = mu / nst[None, :]
    amw = jnp.roll(am, 1, axis=0)
    sel = (am > 0.) & (amw < 0.)

    def body(ns, c):
        rm, rmom, mass = c
        act = rowact & (nstep >= ns)
        rxm, rxxm, _ = checkflux(amw, am, mass, rm, rmom[MX], rmom[MXX])
        mom_ = rmom.at[MX].set(jnp.where(sel, rxm, rmom[MX])).at[MXX].set(jnp.where(sel, rxxm, rmom[MXX]))
        r1 = jnp.roll(rm, -1, axis=0)
        f = face_flux(am, mass, rm, mom_, jnp.roll(mass, -1, axis=0), r1, jnp.roll(mom_, -1, axis=1), rm, r1, XSPEC, c3,
                      strict_neg=False)
        fmw = west_moments(f, XSPEC)
        rmn, momn, massn = cell_update(rm, mom_, mass, amw, jnp.roll(f['fe'], 1, axis=0), jnp.roll(f['fe_pass'], 1, axis=0),
                                       jnp.roll(fmw, 1, axis=1), am, f['fe'], f['fe0'], f['fmom'], XSPEC)
        a = act[None, :]
        return jnp.where(a, rmn, rm), jnp.where(a[None], momn, rmom), jnp.where(a, massn, mass)
    return lax.fori_loop(1, nmax + 1, body, (rm, rmom, mass))


def aadvqy_j(rm, rmom, mass, mv, byim, c3):
    """aadvqy for one level (polar caps, faces, cell update).  mv(:,j) = north-edge flux of row j."""
    im, jm = IM, JM
    m_sp = mass[0, 0] * im; rm_sp = rm[0, 0] * im
    rzm_sp = rmom[MZ, 0, 0] * im; rzzm_sp = rmom[MZZ, 0, 0] * im
    ks = mv[:, 0] < 0.                                   # `if (mv(i,1) < 0) cycle`: keep
    mass = mass.at[:, 0].set(jnp.where(ks, mass[:, 0], m_sp))
    rm = rm.at[:, 0].set(jnp.where(ks, rm[:, 0], rm_sp))
    rmom = rmom.at[MZ, :, 0].set(jnp.where(ks, rmom[MZ, :, 0], rzm_sp))
    rmom = rmom.at[MZZ, :, 0].set(jnp.where(ks, rmom[MZZ, :, 0], rzzm_sp))
    for q in IHMOMS:
        rmom = rmom.at[q, :, 0].set(jnp.where(ks, rmom[q, :, 0], 0.))
    m_np = mass[0, jm - 1] * im; rm_np = rm[0, jm - 1] * im
    rzm_np = rmom[MZ, 0, jm - 1] * im; rzzm_np = rmom[MZZ, 0, jm - 1] * im
    kn = mv[:, jm - 2] >= 0.
    mass = mass.at[:, jm - 1].set(jnp.where(kn, mass[:, jm - 1], m_np))
    rm = rm.at[:, jm - 1].set(jnp.where(kn, rm[:, jm - 1], rm_np))
    rmom = rmom.at[MZ, :, jm - 1].set(jnp.where(kn, rmom[MZ, :, jm - 1], rzm_np))
    rmom = rmom.at[MZZ, :, jm - 1].set(jnp.where(kn, rmom[MZZ, :, jm - 1], rzzm_np))
    for q in IHMOMS:
        rmom = rmom.at[q, :, jm - 1].set(jnp.where(kn, rmom[q, :, jm - 1], 0.))
    mvf = mv[:, :jm - 1]
    f = face_flux(mvf, mass[:, :jm - 1], rm[:, :jm - 1], rmom[:, :, :jm - 1], mass[:, 1:], rm[:, 1:], rmom[:, :, 1:],
                  rm[:, :jm - 1], rm[:, 1:], YSPEC, c3, strict_neg=True)
    f0 = face_flux(mvf[:, :1], mass[:, :1], rm[:, :1], rmom[:, :, :1], mass[:, 1:2], rm[:, 1:2], rmom[:, :, 1:2],
                   rm[:, :1], rm[:, 1:2], YSPEC, c3, strict_neg=False)
    fe = f['fe'].at[:, :1].set(f0['fe'])
    fe0 = f['fe0'].at[:, :1].set(f0['fe0'])
    fe_pass = f['fe_pass'].at[:, :1].set(f0['fe_pass'])
    fp_pass = f['fp_pass'].at[:, :1].set(f0['fp_pass'])
    fpp_pass = f['fpp_pass'].at[:, :1].set(f0['fpp_pass'])
    fmom = f['fmom'].at[:, :, :1].set(f0['fmom'])
    fmw_all = fmom.at[MY].set(fp_pass).at[MYY].set(fpp_pass)
    # south polar cap
    m_sp = m_sp - seqsum(mvf[:, 0])
    rm_sp = rm_sp - seqsum(fe[:, 0])
    rzm_sp = seqacc(rzm_sp, fmom[MZ, :, 0], -1.)
    rzzm_sp = seqacc(rzzm_sp, fmom[MZZ, :, 0], -1.)
    mass = mass.at[0, 0].set(m_sp * byim)
    rm = rm.at[0, 0].set(rm_sp * byim)
    rmom = rmom.at[MZ, 0, 0].set(rzm_sp * byim).at[MZZ, 0, 0].set(rzzm_sp * byim)
    for q in IHMOMS:
        rmom = rmom.at[q, 0, 0].set(0.)
    z = rm[0, 0] <= 0.
    rm = rm.at[0, 0].set(jnp.where(z, 0., rm[0, 0]))
    rmom = rmom.at[:, 0, 0].set(jnp.where(z, 0., rmom[:, 0, 0]))
    js = slice(1, jm - 1)
    rmn, momn, massn = cell_update(rm[:, js], rmom[:, :, js], mass[:, js], mvf[:, 0:jm - 2], fe[:, 0:jm - 2],
                                   fe_pass[:, 0:jm - 2], fmw_all[:, :, 0:jm - 2], mvf[:, 1:jm - 1], fe[:, 1:jm - 1],
                                   fe0[:, 1:jm - 1], fmom[:, :, 1:jm - 1], YSPEC)
    # north polar cap (last face values)
    m_np = m_np + seqsum(mvf[:, jm - 2])
    rm_np = rm_np + seqsum(fe[:, jm - 2])
    rzm_np = seqacc(rzm_np, fmom[MZ, :, jm - 2], 1.)
    rzzm_np = seqacc(rzzm_np, fmom[MZZ, :, jm - 2], 1.)
    rm = rm.at[:, js].set(rmn)
    rmom = rmom.at[:, :, js].set(momn)
    mass = mass.at[:, js].set(massn)
    mass = mass.at[0, jm - 1].set(m_np * byim)
    rm = rm.at[0, jm - 1].set(rm_np * byim)
    rmom = rmom.at[MZ, 0, jm - 1].set(rzm_np * byim).at[MZZ, 0, jm - 1].set(rzzm_np * byim)
    for q in IHMOMS:
        rmom = rmom.at[q, 0, jm - 1].set(0.)
    z = rm[0, jm - 1] <= 0.
    rm = rm.at[0, jm - 1].set(jnp.where(z, 0., rm[0, jm - 1]))
    rmom = rmom.at[:, 0, jm - 1].set(jnp.where(z, 0., rmom[:, 0, jm - 1]))
    return rm, rmom, mass


def aadvqz_j(rm1, rmom1, mass1, rm2, rmom2, mass2, mw, carry, act, c3):
    """One interface sweep (layer 1 updated, layer 2 read only); carry = (mwdn, fdn, fdn0, fmomdn)."""
    mwdn, fdn, fdn0, fmomdn = carry
    f = face_flux(mw, mass1, rm1, rmom1, mass2, rm2, rmom2, rm1, rm2, ZSPEC, c3, strict_neg=True)
    rmn, momn, massn = cell_update(rm1, rmom1, mass1, mwdn, fdn, fdn0, fmomdn, mw, f['fe'], f['fe0'], f['fmom'], ZSPEC)
    rm1 = jnp.where(act, rmn, rm1)
    rmom1 = jnp.where(act[None], momn, rmom1)
    mass1 = jnp.where(act, massn, mass1)
    fmw = f['fmom'].at[MZ].set(f['fp_pass']).at[MZZ].set(f['fpp_pass'])
    carry = (jnp.where(act, mw, mwdn), jnp.where(act, f['fe'], fdn), jnp.where(act, f['fe_pass'], fdn0),
             jnp.where(act[None], fmw, fmomdn))
    for j in (0, JM - 1):
        mass1 = mass1.at[1:, j].set(mass1[0, j])
        rm1 = rm1.at[1:, j].set(rm1[0, j])
        rmom1 = rmom1.at[:, 1:, j].set(rmom1[:, 0:1, j])
    return rm1, rmom1, mass1, carry


# ---------------------------------------------------------------------------------------------- AADVQ0
def _xstep(m, mu):
    """XSTEP for a batch of rows.  m, mu (IM, B).  Returns nstep (B,), mi (IM,B) mass after nstep sub-steps, ierr."""
    B = m.shape[1]

    def cond(c):
        trial, nstep, mi_out, done, ierr = c
        return (trial <= NSTEPMAX_X) & jnp.any(~done) & (ierr == 0)

    def body(c):
        trial, nstep, mi_out, done, ierr = c
        last = trial == NSTEPMAX_X
        nstep = jnp.where(~done, trial, nstep)
        am = mu / trial.astype(jnp.float64)

        def sub(ns, cc):
            mi, cm = cc
            mip1 = jnp.roll(mi, -1, axis=0)
            cand = jnp.where(am > 0., am / mi, -am / mip1)
            cm = jnp.maximum(cm, jnp.max(cand, axis=0))
            return (mi + jnp.roll(am, 1, axis=0)) - am, cm
        mi, cm = lax.fori_loop(0, trial, sub, (m, jnp.zeros(B)))
        ok = ~(cm > 1.) & ~done & ~last
        mi_out = jnp.where(ok[None, :], mi, mi_out)
        done = done | ok
        return trial + 1, nstep, mi_out, done, jnp.where(last & jnp.any(~done), 1, ierr)
    init = (jnp.asarray(1), jnp.zeros(B, jnp.int64), m, jnp.zeros(B, bool), jnp.asarray(0))
    _, nstep, mi_out, _, ierr = lax.while_loop(cond, body, init)
    return nstep, mi_out, ierr


def _level_ncycxy(mb_l, mu_l, mv_l, dz_l, ncyc, byim):
    """NCYCXY search of one level (QUS3D.f:556-700).  Returns (ncycxy, error flag)."""
    def nc3_cond(c):
        nc3, nxy, mb2d, brk = c
        return (nc3 <= ncyc) & ~brk

    def nc3_body(c):
        nc3, nxy, mb2d, brk = c
        start = jnp.where(nc3 == 1, mb_l, mb2d)

        def w_cond(c2):
            ma2d, nxy, nbad, brk = c2
            return (nbad > 0) & ~brk

        def w_body(c2):
            ma2d, nxy, nbad, brk = c2
            brk = nxy > NCMAX
            by = inv32(nxy)

            def n_cond(c3):
                nc, ma2d, nbad = c3
                return (nc <= nxy) & (nbad == 0)

            def n_body(c3):
                nc, ma2d, nbad = c3
                mvbyn = mv_l[:, 1:JM - 2] * by
                nbad = nbad + jnp.sum((ma2d[:, 1:JM - 2] - mvbyn) * (ma2d[:, 2:JM - 1] + mvbyn) < 0.)
                mpol = ma2d[0, 0] * IM
                mvbyn = mv_l[:, 0] * by
                nbad = nbad + jnp.sum((mpol - mvbyn) * (ma2d[:, 1] + mvbyn) < 0.)
                mpol = ma2d[0, JM - 1] * IM
                mvbyn = mv_l[:, JM - 2] * by
                nbad = nbad + jnp.sum((ma2d[:, JM - 2] - mvbyn) * (mpol + mvbyn) < 0.)
                ma2d = ma2d.at[:, 1:JM - 1].set(ma2d[:, 1:JM - 1] + (mv_l[:, 0:JM - 2] - mv_l[:, 1:JM - 1]) * by)
                nbad = nbad + jnp.sum(ma2d[:, 1:JM - 1] < MRAT_LIMY * mb_l[:, 1:JM - 1])
                ma2d = ma2d.at[:, 1:JM - 1].set(ma2d[:, 1:JM - 1] + (jnp.roll(mu_l[:, 1:JM - 1], 1, axis=0) - mu_l[:, 1:JM - 1]) * by)
                ssp = seqsum(ma2d[:, 0] - mv_l[:, 0] * by) * byim
                ma2d = ma2d.at[:, 0].set(ssp)
                nbad = nbad + (ma2d[0, 0] < MRAT_LIMY * mb_l[0, 0])
                snp = seqsum(ma2d[:, JM - 1] + mv_l[:, JM - 2] * by) * byim
                ma2d = ma2d.at[:, JM - 1].set(snp)
                nbad = nbad + (ma2d[0, JM - 1] < MRAT_LIMY * mb_l[0, JM - 1])
                return nc + 1, ma2d, nbad.astype(jnp.int64)
            _, ma_new, nb = lax.while_loop(n_cond, n_body, (jnp.asarray(1), ma2d, jnp.asarray(0, jnp.int64)))
            fail = (nb > 0) & ~brk
            ma_out = jnp.where(brk, ma2d, jnp.where(fail, start, ma_new))
            return ma_out, jnp.where(fail, nxy + 1, nxy), jnp.where(brk, 0, nb), brk
        ma2d, nxy, _, brk = lax.while_loop(w_cond, w_body, (start, nxy, jnp.asarray(1, jnp.int64), jnp.asarray(False)))
        return nc3 + 1, nxy, ma2d + dz_l, brk
    _, nxy, _, brk = lax.while_loop(nc3_cond, nc3_body, (jnp.asarray(1), jnp.asarray(1), mb_l, jnp.asarray(False)))
    return nxy, brk


def _level_nstepx(mb_l, mu_l, mv_l, dz_l, ncyc, nxy):
    """NSTEPX of one level (QUS3D.f:704-760): mu_l, mv_l already scaled by ncyc and 1/ncycxy."""
    def nc3_body(nc3, c):
        ma2d, nst, ie = c

        def nc_body(nc, c2):
            ma2d, nst, ie = c2
            rows = ma2d[:, 1:JM - 1] + (mv_l[:, 0:JM - 2] - mv_l[:, 1:JM - 1])
            nsd, mi, e = _xstep(rows, mu_l[:, 1:JM - 1])
            return ma2d.at[:, 1:JM - 1].set(mi), jnp.maximum(nst, nsd), ie + e
        ma2d, nst, ie = lax.fori_loop(1, nxy + 1, nc_body, (ma2d, nst, ie))
        ma2d = jnp.where(nc3 < ncyc, ma2d.at[:, 1:JM - 1].set(ma2d[:, 1:JM - 1] + dz_l[:, 1:JM - 1]), ma2d)
        return ma2d, nst, ie
    ma2d, nst, ie = lax.fori_loop(1, ncyc + 1, nc3_body, (mb_l, jnp.ones(JM - 2, jnp.int64), jnp.asarray(0)))
    return jnp.zeros(JM, jnp.int64).at[1:JM - 1].set(nst), ie


def aadvq0_j(mu, mv, mw, mb, imaxj, byim):
    """AADVQ0 (cycle counts) without the z-extra partition.  Returns dict (see module docstring) with `do_z_extra`
    (True: the caller must use the numpy AADVQ0), `err` (0 ok, 1 ncyc>ncmax, 2 ncycxy>ncmax, 3 xstep too many steps)."""
    pv_south = mv[:, 0, :]
    mu = mu.at[:, 0, :].set(0.)
    mv = jnp.concatenate([mv[:, 1:, :], jnp.zeros((IM, 1, LM))], axis=1)
    mu = mu.at[:, JM - 1, :].set(0.)
    mw = mw.at[:, :, LM - 1].set(0.)

    def trial_cond(c):
        ncyc, ok, lmin, lmax, err = c
        return ~ok & (err == 0)

    def trial_body(c):
        ncyc = c[0] + 1
        byn = inv32(ncyc)

        def nc_cond(cc):
            nc, mma, nbad, lmin, lmax = cc
            return (nc <= ncyc) & (nbad == 0)

        def nc_body(cc):
            nc, mma, nbad, lmin, lmax = cc
            muw = jnp.roll(mu[:, 1:JM - 1, :], 1, axis=0)
            d = ((muw - mu[:, 1:JM - 1, :]) + mv[:, 0:JM - 2, :]) - mv[:, 1:JM - 1, :]
            mid = mma[:, 1:JM - 1, :] + byn * d
            nb = jnp.sum(mid < MRAT_LIMH * mb[:, 1:JM - 1, :])
            ssp = seqsum(mma[:, 0, :] - mv[:, 0, :] * byn) * byim                        # (LM,)
            snp = seqsum(mma[:, JM - 1, :] + mv[:, JM - 2, :] * byn) * byim
            mma = mma.at[:, 1:JM - 1, :].set(mid)
            mma = mma.at[:, 0, :].set(ssp[None, :])
            nb = nb + jnp.sum(mma[0, 0, :] < MRAT_LIMH * mb[0, 0, :])
            mma = mma.at[:, JM - 1, :].set(snp[None, :])
            nb = nb + jnp.sum(mma[0, JM - 1, :] < MRAT_LIMH * mb[0, JM - 1, :])
            nbad2 = nbad + nb
            mwbyn = -(mw * byn)
            viol = (mma[:, :, :LM - 1] - mwbyn[:, :, :LM - 1]) * (mma[:, :, 1:] + mwbyn[:, :, :LM - 1]) < 0.
            lv = jnp.arange(LM - 1)[None, None, :]
            lmin2 = jnp.minimum(lmin, jnp.min(jnp.where(viol, lv + 1, LM + 1), axis=2))
            lmax2 = jnp.maximum(lmax, jnp.max(jnp.where(viol, lv + 2, 0), axis=2))
            znew = jnp.concatenate([
                (mma[:, :, 0] + mw[:, :, 0] * byn)[:, :, None],
                mma[:, :, 1:LM - 1] + (mw[:, :, 1:LM - 1] - mw[:, :, 0:LM - 2]) * byn,
                (mma[:, :, LM - 1] - mw[:, :, LM - 2] * byn)[:, :, None]], axis=2)
            mma = jnp.where(nc < ncyc, znew, mma)
            return nc + 1, mma, nbad2.astype(jnp.int64), lmin2, lmax2
        init = (jnp.asarray(1), mb, jnp.asarray(0, jnp.int64), jnp.full((IM, JM), LM + 1, jnp.int64), jnp.zeros((IM, JM), jnp.int64))
        _, _, nbad, lmin, lmax = lax.while_loop(nc_cond, nc_body, init)
        return ncyc, nbad == 0, lmin, lmax, jnp.where((nbad != 0) & (ncyc >= NCMAX), 1, 0)
    ncyc, _, lminzij, lmaxzij, err1 = lax.while_loop(
        trial_cond, trial_body, (jnp.asarray(0), jnp.asarray(False), jnp.full((IM, JM), LM + 1, jnp.int64), jnp.zeros((IM, JM), jnp.int64), jnp.asarray(0)))
    do_z = jnp.any(lminzij < LM)
    byn = inv32(ncyc)
    scale = ncyc > 1
    pv_south = jnp.where(scale, pv_south * byn, pv_south)
    mu = jnp.where(scale, mu * byn, mu)
    mv = jnp.where(scale, mv * byn, mv)
    # per-level z term of the mass update between the ncyc sub-cycles (as the numpy port)
    mwl = jnp.moveaxis(mw, 2, 0)                                                     # (LM, IM, JM)
    dz = jnp.concatenate([(mwl[0] * byn)[None], (mwl[1:LM - 1] - mwl[0:LM - 2]) * byn, (-(mwl[LM - 2] * byn))[None]], axis=0)
    mbl, mul, mvl = (jnp.moveaxis(x, 2, 0) for x in (mb, mu, mv))
    nxy, brk = jax.vmap(_level_ncycxy, in_axes=(0, 0, 0, 0, None, None))(mbl, mul, mvl, dz, ncyc, byim)
    err2 = jnp.any(brk | (nxy > NCMAX))
    bynxy = inv32(nxy)
    sc = (nxy > 1)
    pv_south = jnp.where(sc[None, :], pv_south * bynxy[None, :], pv_south)
    mul = jnp.where(sc[:, None, None], mul * bynxy[:, None, None], mul)
    mvl = jnp.where(sc[:, None, None], mvl * bynxy[:, None, None], mvl)
    nstepx, ie = jax.vmap(_level_nstepx, in_axes=(0, 0, 0, 0, None, 0))(mbl, mul, mvl, dz, ncyc, nxy)
    err = jnp.where(err1 > 0, 1, jnp.where(err2, 2, jnp.where(jnp.sum(ie) > 0, 3, 0)))
    return dict(ncyc=ncyc, ncycxy=nxy, nstepx=nstepx.T, do_z_extra=do_z, err=err, mu=jnp.moveaxis(mul, 0, 2),
                mv=jnp.moveaxis(mvl, 0, 2), mw=mw, pv_south=pv_south, lminzij=lminzij, lmaxzij=lmaxzij)


# ---------------------------------------------------------------------------------------------- AADVQ cycle
def _tables(mu, mv, mw, imaxj):
    """Flow-out-both-sides masks (numpy i_y / i_z lists) as boolean arrays (IM,JM,LM)."""
    my = jnp.zeros((IM, JM, LM), bool).at[:, 1:JM - 1, :].set((mv[:, 1:JM - 1, :] > 0.) & (mv[:, 0:JM - 2, :] < 0.))
    act = jnp.arange(IM)[:, None] < imaxj[None, :]
    mz = jnp.zeros((IM, JM, LM), bool).at[:, :, 1:LM - 1].set((mw[:, :, 0:LM - 2] > 0.) & (mw[:, :, 1:LM - 1] < 0.))
    mz = mz & act[:, :, None]
    return my, mz


def make_cycle(imaxj, byim, c3):
    """Returns cycle(rm, rmom, mma, pu, pv, sd, nstepx, ncycxy, mask_y, mask_z): one nc cycle of AADVQ on level-leading
    arrays rm (LM,IM,JM), rmom (LM,9,IM,JM), mma (LM,IM,JM), pu/pv/sd (LM,IM,JM), nstepx (LM,JM), ncycxy (LM,)."""
    act = jnp.arange(IM)[:, None] < imaxj[None, :]

    def cycle(rm, rmom, mma, pu, pv, sd, nstepx, ncycxy, my, mz):
        zero = jnp.zeros((IM, JM))
        carry0 = (zero, zero, zero, jnp.zeros((9, IM, JM)))

        def level_xy(l, st):
            rm, rmom, mma = st
            def one(k, s):
                r, mo, ma = s
                aml = jnp.roll(pv[l], 1, axis=1)
                a, b, _ = checkflux(aml, pv[l], ma, r, mo[MY], mo[MYY])
                m_ = my[l]
                mo = mo.at[MY].set(jnp.where(m_, a, mo[MY])).at[MYY].set(jnp.where(m_, b, mo[MYY]))
                r, mo, ma = aadvqy_j(r, mo, ma, pv[l], byim, c3)
                r, mo, ma = aadvqx_j(r, mo, ma, pu[l], nstepx[l], c3)
                return r, mo, ma
            r, mo, ma = lax.fori_loop(0, ncycxy[l], one, (rm[l], rmom[l], mma[l]))
            return rm.at[l].set(r), rmom.at[l].set(mo), mma.at[l].set(ma)

        def zcheck(l, st):
            rm, rmom, mma = st
            a, b, _ = checkflux(sd[l - 1], sd[l], mma[l], rm[l], rmom[l, MZ], rmom[l, MZZ])
            m_ = mz[l]
            mo = rmom[l].at[MZ].set(jnp.where(m_, a, rmom[l, MZ])).at[MZZ].set(jnp.where(m_, b, rmom[l, MZZ]))
            return rm, rmom.at[l].set(mo), mma

        def zsweep(L, st, carry):
            rm, rmom, mma = st
            l1 = L - 2
            l2 = jnp.minimum(L - 1, LM - 1)
            r1, mo1, m1, carry = aadvqz_j(rm[l1], rmom[l1], mma[l1], rm[l2], rmom[l2], mma[l2], sd[l1], carry, act, c3)
            return (rm.at[l1].set(r1), rmom.at[l1].set(mo1), mma.at[l1].set(m1)), carry

        def body(L, c):
            st, carry = c
            st = lax.cond(L <= LM, lambda s: level_xy(L - 1, s), lambda s: s, st)
            st = lax.cond((L > 1) & (L < LM), lambda s: zcheck(L - 1, s), lambda s: s, st)
            st, carry = lax.cond(L > 1, lambda a: zsweep(L, a[0], a[1]), lambda a: a, (st, carry))
            return st, carry
        (rm, rmom, mma), _ = lax.fori_loop(1, LM + 2, body, ((rm, rmom, mma), carry0))
        return rm, rmom, mma
    return cycle


def _to_lead(x):
    return jnp.moveaxis(x, -1, 0)


def _from_lead(x):
    return jnp.moveaxis(x, 0, -1)


class QdynamKit:
    def __init__(self, qg):
        self.qg = qg
        self.axyp = jnp.asarray(np.asarray(qg['axyp'], dtype=np.float64))
        self.imaxj = jnp.asarray(np.asarray(qg['imaxj'], dtype=np.int64))
        self.kg2mb = jnp.asarray(float(qg['kg2mb']))
        self.byim_geom = jnp.asarray(float(qg['byim_geom']))
        self.byim_qus = jnp.asarray(float(qg['byim_qus']))
        self.c3 = jnp.asarray(3.)
        self.n_fallback = 0

        @jax.jit
        def prep(q, qmom, maold, mus, mvs, mws, axyp, kg2mb, byim_geom, imaxj):
            mb = jnp.moveaxis((maold * kg2mb) * axyp[None], 0, 2)
            q0 = aadvq0_j(mus, mvs, mws, mb, imaxj, byim_geom)
            rm = q * mb
            rmom = qmom * mb[None]
            my, mz = _tables(q0['mu'], q0['mv'], q0['mw'], imaxj)
            return mb, rm, rmom, q0, my, mz
        self._prep = prep

        @jax.jit
        def prep_given(q, qmom, maold, mu, mv, mw, axyp, kg2mb, imaxj):
            """Same without AADVQ0 (the counts / scaled fluxes come from the numpy AADVQ0 in the z-extra fallback)."""
            mb = jnp.moveaxis((maold * kg2mb) * axyp[None], 0, 2)
            my, mz = _tables(mu, mv, mw, imaxj)
            return mb, q * mb, qmom * mb[None], my, mz
        self._prep_given = prep_given

        @jax.jit
        def one_cycle(rm, rmom, mma, pu, pv, mw, ncyc, nstepx, ncycxy, my, mz, imaxj, byim, c3):
            sd = -(mw * (1. / ncyc.astype(jnp.float64)))
            cyc = make_cycle(imaxj, byim, c3)
            return cyc(_to_lead(rm), jnp.moveaxis(rmom, (0, 1, 2, 3), (1, 2, 3, 0)), _to_lead(mma), _to_lead(pu), _to_lead(pv),
                       _to_lead(sd), nstepx.T, ncycxy, _to_lead(my), _to_lead(mz))
        self._one_cycle_raw = one_cycle

        @jax.jit
        def all_cycles(rm, rmom, mma, pu, pv, mw, ncyc, nstepx, ncycxy, my, mz, imaxj, byim, c3):
            sd = _to_lead(-(mw * (1. / ncyc.astype(jnp.float64))))
            cyc = make_cycle(imaxj, byim, c3)
            st = (_to_lead(rm), jnp.moveaxis(rmom, 3, 0), _to_lead(mma))
            args = (_to_lead(pu), _to_lead(pv), sd, nstepx.T, ncycxy, _to_lead(my), _to_lead(mz))
            st = lax.fori_loop(0, ncyc, lambda i, s: cyc(*s, *args), st)
            bymma = 1. / _from_lead(st[2])
            return _from_lead(st[0]) * bymma, jnp.moveaxis(st[1], 0, 3) * bymma[None]
        self._all_cycles = all_cycles

        @jax.jit
        def finish(rm, rmom, mma):
            bymma = 1. / mma
            return rm * bymma, rmom * bymma[None]
        self._finish = finish

    def _cycle_np_layout(self, rm, rmom, mma, pu, pv, mw, ncyc, nstepx, ncycxy, my, mz):
        r, mo, ma = self._one_cycle_raw(rm, rmom, mma, pu, pv, mw, jnp.asarray(ncyc), jnp.asarray(nstepx), jnp.asarray(ncycxy),
                                        my, mz, self.imaxj, self.byim_qus, self.c3)
        return _from_lead(r), jnp.moveaxis(mo, (1, 2, 3, 0), (0, 1, 2, 3)), _from_lead(ma)

    def __call__(self, q, qmom, maold, mus, mvs, mws):
        """numpy in / numpy out.  Raises aq.QusError where the Fortran stops."""
        a = [jnp.asarray(np.ascontiguousarray(x, dtype=np.float64)) for x in (q, qmom, maold, mus, mvs, mws)]
        mb, rm, rmom, q0, my, mz = self._prep(*a, self.axyp, self.kg2mb, self.byim_geom, self.imaxj)
        err = int(q0['err'])
        if err:
            raise aq.QusError({1: 'AADVQ0: ncyc>ncmax', 2: 'AADVQ0: ncycxy>ncmax', 3: 'too many steps in xstep'}[err])
        if not bool(q0['do_z_extra']):
            qo, qmo = self._all_cycles(rm, rmom, mb, q0['mu'], q0['mv'], q0['mw'], q0['ncyc'], q0['nstepx'], q0['ncycxy'], my, mz,
                                       self.imaxj, self.byim_qus, self.c3)
            return dict(q=np.array(qo), qmom=np.array(qmo), mus=np.array(q0['mu']), mvs=np.array(q0['mv']), mws=np.array(q0['mw']),
                        ncyc=int(q0['ncyc']), ncycxy=np.array(q0['ncycxy']), nstepx=np.array(q0['nstepx']), do_z_extra=False)
        return self._z_extra_fallback(a, np.array(mb))

    def _z_extra_fallback(self, a, mb):
        """AADVQ0 incl. the extra-column partition in numpy (the Fortran control flow, dyn_aadvq_ff.aadvq0); the main cycle
        sweeps stay JAX, the extra-column vertical advection between the cycles is numpy (aadvqz_column)."""
        self.n_fallback += 1
        q, qmom, maold, mus, mvs, mws = (np.asarray(x) for x in a)
        qg = self.qg
        q0 = aq.aadvq0(mus, mvs, mws, mb, qg['imaxj'], qg['byim_geom'])
        ncyc = q0['ncyc']
        pu, pv, mw = (jnp.asarray(q0[k]) for k in ('mu', 'mv', 'mw'))
        my, mz = _tables(pu, pv, mw, self.imaxj)
        mbj = jnp.asarray(mb)
        rm = jnp.asarray(q) * mbj
        rmom = jnp.asarray(qmom) * mbj[None]
        mma = mbj
        nst = jnp.asarray(q0['nstepx']); nxy = jnp.asarray(q0['ncycxy'])
        for nc in range(1, ncyc + 1):
            rm, rmom, mma = self._cycle_np_layout(rm, rmom, mma, pu, pv, mw, ncyc, nst, nxy, my, mz)
            rm_n, rmom_n, mma_n = np.array(rm), np.array(rmom), np.array(mma)
            for j in range(JM):
                for i in range(IM):
                    ns_ = int(q0['nstepz_extra'][i, j])
                    if ns_ == 0:
                        continue
                    lmin = int(q0['lminzij'][i, j]); lmax = int(q0['lmaxzij'][i, j]); nl = lmax - lmin + 1
                    mw1d = np.zeros(nl)
                    mw1d[:nl - 1] = -q0['mw_extra'][i, j, lmin - 1:lmax - 1] / (ncyc * ns_)
                    ma1d = mma_n[i, j, lmin - 1:lmax].copy()
                    rm1d = rm_n[i, j, lmin - 1:lmax].copy()
                    rmom1d = rmom_n[:, i, j, lmin - 1:lmax].copy()
                    for istep in range(ns_):
                        rm1d, rmom1d, ma1d = aq.aadvqz_column(rm1d, rmom1d, ma1d, mw1d)
                    mma_n[i, j, lmin - 1:lmax] = ma1d; rm_n[i, j, lmin - 1:lmax] = rm1d; rmom_n[:, i, j, lmin - 1:lmax] = rmom1d
            rm, rmom, mma = jnp.asarray(rm_n), jnp.asarray(rmom_n), jnp.asarray(mma_n)
        qo, qmo = self._finish(rm, rmom, mma)
        return dict(q=np.array(qo), qmom=np.array(qmo), mus=q0['mu'], mvs=q0['mv'], mws=q0['mw'], ncyc=ncyc,
                    ncycxy=q0['ncycxy'], nstepx=q0['nstepx'], do_z_extra=True)
