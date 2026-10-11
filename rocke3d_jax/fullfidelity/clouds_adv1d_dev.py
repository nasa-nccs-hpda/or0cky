"""D214: the QUS subsidence advection of MSTCNV (ADV1D with the positivity limiter) as a DEVICE program.

Replaces the host callback `clouds_mstcnv_dev.adv_cb_dev` -> `clouds_mstcnv_jax._host_adv` -> `clouds_mstcnv_batch._adv_groups` (NumPy).  Measured on the V100 (D214,
`gpu_mstcnv_bench.py phases`): without these callbacks one cloud-base event block takes 0.043 s instead of 0.60 s, i.e. the callbacks are 93% of the cloud-base stage.

Same arithmetic, same operation order as the NumPy reference (dyn_adv1d_ff._flux_stage / limitq / _slope_stage / _update_stage / _qlimit_cyclic, clouds_mstcnv_batch.adv_batch):
each selected column is a cyclic line of nx = lmax - ldmin + 1 cells (layers ldmin..lmax); lines are independent, so all selected columns run at once on a fixed window of
NXW = LM cells with a validity mask (cell j < nx); the cyclic neighbours are (j+1) mod nx and (j-1) mod nx per column; the limiter is the only sequential part (loop over
the cells with per-line early exit when limitq returns ierr = 2).  A line with ierr == 2 skips the update stage and keeps s / mass and the moments as left by the limiter.

Interface = that of the host callback:  adv_dev(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit)  ->  (arr, mom, ierr, lerr)
  sel (n,) bool; ldmin, lmax (n,) int; arr (L1, n); mom (9, L1, n); ml, cmneg (L1, n) [mass and dm, rows = layer number]; ierr, lerr (n,) int.
"""
import clouds_jax_env  # noqa: F401  (XLA flags BEFORE jax: no FMA contraction, no algebraic simplifier; needed for bitwise equality with the NumPy reference)
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dyn_adv1d_ff import ZDIR

jax.config.update("jax_enable_x64", True)

NXW = 40          # window: the largest slice (LM layers)


def _where(c, a, b):
    return jnp.where(c, a, b)


def limitq_vec(anm1, an, fnm1, fn, sn, sx, sxx):
    """dyn_adv1d_ff.limitq (QUSDEF.f:639-758) for arrays of lines; returns (fnm1, fn, sx, sxx, ierr) with ierr int."""
    f64 = anm1.dtype
    zero_i = jnp.zeros(anm1.shape, jnp.int64)
    # ------------------------------------------------------------------ case A: no air leaving the box
    caseA = (anm1 >= 0.) & (an <= 0.)
    # ------------------------------------------------------------------ case B: air leaving through both edges
    caseB = (~caseA) & (anm1 < 0.) & (an > 0.)
    sl = -fnm1
    sr = +fn
    sc = sn - sl
    sc = sc - sr
    b_ret0 = (sl >= 0.) & (sr >= 0.) & (sc >= 0.)
    frl = anm1
    frl1 = frl + 1.
    frr = an
    frr1 = frr - 1.
    # center division negative
    gamma = 1. + (frl - frr)
    t = (frl + frr)
    g13ab = gamma * gamma - 1. + 3. * (t * t)
    den = gamma * (12. * (t * t) + 5. * g13ab * g13ab)
    sxx_c = sxx - sc * 10. * g13ab / den
    sx_c = sx + sc * 12. * (frl + frr) / den
    sl_c = -frl * (sn - frl1 * (sx_c - (frl + frl1) * sxx_c))
    sr_c = sn - sl_c
    # leftmost division
    u = (frl + frl1)
    sxx_l = sxx + sl * (frl + frl1) / (frl * frl1 * (.6 + u * u))
    sx_l = sn / frl1 + (frl + frl1) * sxx_l
    sr_l = frr * (sn - frr1 * (sx_l - (frr + frr1) * sxx_l))
    sl_l = jnp.zeros_like(sl)
    # rightmost division
    u2 = (frr + frr1)
    sxx_r = sxx - sr * (frr + frr1) / (frr * frr1 * (.6 + u2 * u2))
    sx_r = sn / frr1 + (frr + frr1) * sxx_r
    sl_r = -frl * (sn - frl1 * (sx_r - (frl + frl1) * sxx_r))
    sr_r = jnp.zeros_like(sr)
    sel_c = (sl >= 0.) & (sr >= 0.)
    sel_l = (~sel_c) & (sr >= 0.)
    sel_r = (~sel_c) & (~sel_l) & (sl >= 0.)
    bsxx = _where(sel_c, sxx_c, _where(sel_l, sxx_l, _where(sel_r, sxx_r, sxx)))
    bsx = _where(sel_c, sx_c, _where(sel_l, sx_l, _where(sel_r, sx_r, sx)))
    bsl = _where(sel_c, sl_c, _where(sel_l, sl_l, _where(sel_r, sl_r, sl)))
    bsr = _where(sel_c, sr_c, _where(sel_l, sr_l, _where(sel_r, sr_r, sr)))
    sc2 = sn - bsl
    sc2 = sc2 - bsr
    gamma2 = 1. + (frl - frr)
    # two outer divisions
    sxx_o = sn * (1. + gamma2) / (2. * gamma2 * frl1 * frr1)
    sx_o = sn * (frl + frr) * (1. + 2. * gamma2) / (2. * gamma2 * frl1 * frr1)
    # center/left divisions
    sxx_cl = sn / (2. * frr * frl1)
    sx_cl = sn * (frl + frr + .5) / (frr * frl1)
    # center/right divisions
    sxx_cr = sn / (2. * frl * frr1)
    sx_cr = sn * (frl + frr - .5) / (frl * frr1)
    f_o = (bsl <= 0.) & (bsr <= 0.)
    f_cl = (~f_o) & (bsl <= 0.) & (sc2 <= 0.)
    f_cr = (~f_o) & (~f_cl) & (bsr <= 0.) & (sc2 <= 0.)
    zl = jnp.zeros_like(bsl)
    csxx = _where(f_o, sxx_o, _where(f_cl, sxx_cl, _where(f_cr, sxx_cr, bsxx)))
    csx = _where(f_o, sx_o, _where(f_cl, sx_cl, _where(f_cr, sx_cr, bsx)))
    csl = _where(f_o, zl, _where(f_cl, zl, _where(f_cr, sn, bsl)))
    csr = _where(f_o, zl, _where(f_cl, sn, _where(f_cr, zl, bsr)))
    B_fnm1, B_fn = -csl, +csr
    # ------------------------------------------------------------------ case C: air leaving through one edge
    right = an > 0.
    fr = _where(right, an, anm1)
    sd = _where(right, fn, -fnm1)
    fsign = _where(right, -1.0, 1.0)
    warn = jnp.abs(fr) > 1.
    err2 = warn & ((sn + fnm1 - fn) < 0)
    ierrC = _where(err2, 2, _where(warn, 1, 0)).astype(jnp.int64)
    su = sn - sd
    c_ret = (~err2) & (sd >= 0.) & (su >= 0.)
    fr1 = fr + fsign
    w = (fr + fr1)
    sxx_d = sxx + fsign * sd * (fr + fr1) / (fr * fr1 * (.6 + w * w))
    sx_d = sn / fr1 + (fr + fr1) * sxx_d
    su_d = sn
    sxx_u = sxx - fsign * su * (fr + fr1) / (fr * fr1 * (.6 + w * w))
    sx_u = sn / fr + (fr + fr1) * sxx_u
    su_u = jnp.zeros_like(su)
    dneg = sd < 0.
    Csxx = _where(dneg, sxx_d, sxx_u)
    Csx = _where(dneg, sx_d, sx_u)
    Csu = _where(dneg, su_d, su_u)
    Csd = sn - Csu
    C_fn = _where(right, Csd, fn)
    C_fnm1 = _where(right, fnm1, -Csd)
    c_unchanged = err2 | c_ret
    # ------------------------------------------------------------------ assemble
    out_fnm1 = _where(caseA, fnm1, _where(caseB, _where(b_ret0, fnm1, B_fnm1), _where(c_unchanged, fnm1, C_fnm1)))
    out_fn = _where(caseA, fn, _where(caseB, _where(b_ret0, fn, B_fn), _where(c_unchanged, fn, C_fn)))
    out_sx = _where(caseA, sx, _where(caseB, _where(b_ret0, sx, csx), _where(c_unchanged, sx, Csx)))
    out_sxx = _where(caseA, sxx, _where(caseB, _where(b_ret0, sxx, csxx), _where(c_unchanged, sxx, Csxx)))
    out_ierr = _where(caseA | caseB, zero_i, ierrC)
    return out_fnm1, out_fn, out_sx, out_sxx, out_ierr


def adv_dev(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit):
    mx, my, mz, mxx, myy, mzz, mxy, myz, mzx = ZDIR
    L1, n = arr.shape
    nx = jnp.where(sel, lmax - ldmin + 1, 1).astype(jnp.int64)
    j = jnp.arange(NXW, dtype=jnp.int64)[None, :]                      # (1, NXW)
    jm = (j < nx[:, None]) & sel[:, None]                               # valid cells
    rows = jnp.clip(ldmin[:, None] + j, 0, L1 - 1)                      # (n, NXW) layer index of cell j
    gl = lambda a: jnp.take_along_axis(a.T, rows, axis=1)               # noqa: E731  (L1, n) -> (n, NXW)
    s = gl(arr)
    mm = jnp.stack([gl(mom[i]) for i in range(9)])                      # (9, n, NXW)
    mass = gl(ml)
    dm = gl(cmneg)
    jj = jnp.broadcast_to(j, (n, NXW))
    nnx = nx[:, None]
    nxt = jnp.where(dm < 0., (jj + 1) % nnx, jj)
    nxt = jnp.where(jm, nxt, 0)
    prv = (jj - 1) % nnx
    prv = jnp.where(jm, prv, 0)
    g = lambda a, idx: jnp.take_along_axis(a, idx, axis=1)              # noqa: E731
    # ------------------------------------------------------------ flux stage (dyn_adv1d_ff._flux_stage)
    frac1 = jnp.where(dm < 0., 1., -1.)
    ms = g(mass, nxt)
    fracm = dm / ms
    fracm = jnp.where(ms <= 0., 0., fracm)
    frac1 = fracm + frac1
    sn = g(s, nxt)
    sx = g(mm[mx], nxt)
    sxx = g(mm[mxx], nxt)
    f = fracm * (sn - frac1 * (sx - (frac1 + fracm) * sxx))
    # ------------------------------------------------------------ qlimit loop (dyn_adv1d_ff._qlimit_cyclic), per line, early exit at ierr == 2
    ie_out = jnp.zeros(n, jnp.int64)
    ne_out = jnp.zeros(n, jnp.int64)
    skip = jnp.zeros(n, bool)
    smx, smxx = mm[mx], mm[mxx]
    if qlimit:
        ar = jnp.arange(n)

        def body(c_n, st):
            f_, smx_, smxx_, nm1, ie_o, ne_o, done = st
            act = (c_n < nx) & sel & ~done
            an = fracm[:, c_n]
            anm1 = jnp.take_along_axis(fracm, nm1[:, None], axis=1)[:, 0]
            fn = f_[:, c_n]
            fnm1 = jnp.take_along_axis(f_, nm1[:, None], axis=1)[:, 0]
            snn = s[:, c_n]
            fnm1n, fnn, sxn, sxxn, ie = limitq_vec(anm1, an, fnm1, fn, snn, smx_[:, c_n], smxx_[:, c_n])
            ie_o = jnp.where(act, ie, ie_o)
            ne_o = jnp.where(act & (ie > 0), c_n + 1, ne_o)
            stop = act & (ie == 2)
            done = done | stop
            commit = act & ~stop
            f_ = f_.at[:, c_n].set(jnp.where(commit, fnn, f_[:, c_n]))
            f_ = f_.at[ar, nm1].set(jnp.where(commit, fnm1n, f_[ar, nm1]))
            smx_ = smx_.at[:, c_n].set(jnp.where(commit, sxn, smx_[:, c_n]))
            smxx_ = smxx_.at[:, c_n].set(jnp.where(commit, sxxn, smxx_[:, c_n]))
            nm1 = jnp.where(act, c_n, nm1)
            return f_, smx_, smxx_, nm1, ie_o, ne_o, done

        st0 = (f, smx, smxx, nx - 1, ie_out, ne_out, jnp.zeros(n, bool))
        f, smx, smxx, _, ie_out, ne_out, skip = lax.fori_loop(0, NXW, body, st0)
        mm = mm.at[mx].set(smx).at[mxx].set(smxx)
    # ------------------------------------------------------------ slope stage (_slope_stage)
    fm = [None] * 9
    sx = g(mm[mx], nxt)            # recomputed from the moments AFTER the limiter, as _slope_stage does
    sxx = g(mm[mxx], nxt)
    fm[mx] = dm * (fracm * fracm * (sx - 3. * frac1 * sxx) - 3. * f)
    fm[mxx] = dm * (dm * jnp.power(fracm, 3.) * sxx - 5. * (dm * f + fm[mx]))
    sxy = g(mm[mxy], nxt)
    fm[my] = fracm * (g(mm[my], nxt) - frac1 * sxy)
    fm[mxy] = dm * (fracm * fracm * sxy - 3. * fm[my])
    szx = g(mm[mzx], nxt)
    fm[mz] = fracm * (g(mm[mz], nxt) - frac1 * szx)
    fm[mzx] = dm * (fracm * fracm * szx - 3. * fm[mz])
    fm[myy] = fracm * g(mm[myy], nxt)
    fm[mzz] = fracm * g(mm[mzz], nxt)
    fm[myz] = fracm * g(mm[myz], nxt)
    # ------------------------------------------------------------ update stage (_update_stage)
    pv = lambda a: g(a, prv)                                            # noqa: E731
    dmm, fmm = pv(dm), pv(f)
    tmp = mass + dmm
    mnew = tmp - dm
    bymnew = 1. / mnew
    dm2 = dmm + dm
    tmp = s + fmm
    s_new = tmp - f
    m0 = mass
    fmx_m, fmxx_m = pv(fm[mx]), pv(fm[mxx])
    n_mx = (mm[mx] * m0 - 3. * (-dm2 * s_new + m0 * (fmm + f)) + (fmx_m - fm[mx])) * bymnew
    n_mxx = (mm[mxx] * m0 * m0
             + 2.5 * s_new * (m0 * m0 - mnew * mnew - 3. * dm2 * dm2)
             + 5. * (m0 * (m0 * (fmm - f) - fmx_m - fm[mx]) + dm2 * n_mx * mnew)
             + (fmxx_m - fm[mxx])) * (bymnew * bymnew)
    fmy_m = pv(fm[my])
    n_my = mm[my] + fmy_m - fm[my]
    n_mxy = (mm[mxy] * m0 - 3. * (-dm2 * n_my + m0 * (fmy_m + fm[my])) + (pv(fm[mxy]) - fm[mxy])) * bymnew
    fmz_m = pv(fm[mz])
    n_mz = mm[mz] + fmz_m - fm[mz]
    n_mzx = (mm[mzx] * m0 - 3. * (-dm2 * n_mz + m0 * (fmz_m + fm[mz])) + (pv(fm[mzx]) - fm[mzx])) * bymnew
    n_myy = mm[myy] + pv(fm[myy]) - fm[myy]
    n_mzz = mm[mzz] + pv(fm[mzz]) - fm[mzz]
    n_myz = mm[myz] + pv(fm[myz]) - fm[myz]
    new = [None] * 9
    new[mx], new[mxx], new[my], new[mxy], new[mz], new[mzx], new[myy], new[mzz], new[myz] = n_mx, n_mxx, n_my, n_mxy, n_mz, n_mzx, n_myy, n_mzz, n_myz
    mnew_f = mnew
    dead = mnew_f <= 0.
    s_new = jnp.where(dead, 0., s_new)
    new = [jnp.where(dead, 0., x) for x in new]
    # lines with ierr == 2 keep s / mass / moments as left by the limiter (post-limiter mx, mxx)
    keep = skip[:, None]
    s_out = jnp.where(keep, s, s_new)
    mass_out = jnp.where(keep, mass, mnew_f)
    mom_out = jnp.stack([jnp.where(keep, mm[i], new[i]) for i in range(9)])
    # ------------------------------------------------------------ write back (valid cells of selected columns only)
    cols = jnp.broadcast_to(jnp.arange(n)[:, None], (n, NXW))
    ridx = jnp.where(jm, rows, L1)                                      # L1 = out of range -> dropped
    arr2 = arr.at[ridx, cols].set(s_out, mode="drop")
    mom2 = mom
    for i in range(9):
        mom2 = mom2.at[i, ridx, cols].set(mom_out[i], mode="drop")
    ierr2 = jnp.where(sel, jnp.maximum(ie_out, ierr), ierr)
    lerr2 = jnp.where(sel, jnp.maximum(ne_out + ldmin - 1, lerr), lerr)
    # the mass array `ml` is an INPUT of the callback (the host updated its own copy only); the callback returned no mass
    return arr2, mom2, ierr2, lerr2
