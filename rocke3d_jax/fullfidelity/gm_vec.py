"""Batched Gent-McWilliams/Redi port (Stage 2, D83): numpy versions of gmredi_ff.isoslope4,
gmkdif and gmfexp (with its helpers) with the (i, j, l) loops replaced by masked array operations.

Arrays are 1-based (IM+1, JM+1, LMO+1), as in gmredi_ff. Every arithmetic expression keeps the
scalar port's operand order so the results are bitwise identical to the scalar port.

Genuinely sequential pieces that stay as loops / cumulative sums, replicating the scalar order:
  * _add_fluxes (enthalpy path): the horizontal flux-divergence is accumulated into TRM cell by
    cell in the scalar's visiting order (per cell: west-wrap term first at i=IM, then +FX, +FY,
    -FX(i+1), -FY(j+1), then the pole-box terms last), written term by term; the vertical
    redistribution is a loop over the 13 layers (trm[l] += f[l]; trm[l+1] -= f[l]).
  * _wrap_adjust_fluxes (salt limiter): pole sums use np.cumsum(...)[-1] (= _fortran_sum), the
    global CONVPOS/CONVADJ sums accumulate in the scalar's (l, i) order per J and then over J.
Validated by gm_vec_compare.py and tests/test_gm_vec.py.
"""
import numpy as np

from odhorz_ff import geomo_dyn_arrays
from gmredi_ff import IM, JM, LMO, DTS

_NAMES_ISO = ["aix0", "aix1", "aix2", "aix3", "aiy0", "aiy1", "aiy2", "aiy3",
              "asx0", "asx1", "asx2", "asx3", "asy0", "asy1", "asy2", "asy3",
              "s2x0", "s2x1", "s2x2", "s2x3", "s2y0", "s2y1", "s2y2", "s2y3"]
_NAMES_KDIF = ["bxx", "byy", "bzz", "azx", "bzx", "czx", "aezx", "ezx", "cezx",
               "azy", "bzy", "czy", "aezy", "ezy", "cezy"]


# ---------------------------------------------------------------- shift helpers (zero filled)
def _sl(axis, ndim, s):
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


def _west(a):
    """out[i] = a[im1(i)] for i = 1..IM (periodic), out[0] = 0."""
    o = np.zeros_like(a)
    o[1] = a[IM]
    o[2:IM + 1] = a[1:IM]
    return o


def _east(a):
    """out[i] = a[ip1(i)] for i = 1..IM (periodic), out[0] = 0."""
    o = np.zeros_like(a)
    o[1:IM] = a[2:IM + 1]
    o[IM] = a[1]
    return o


def _jm1(a, axis=1):
    """out[..., j, ...] = a[..., j-1, ...] (out at j=0 is 0)."""
    o = np.zeros_like(a)
    o[_sl(axis, a.ndim, slice(1, None))] = a[_sl(axis, a.ndim, slice(None, -1))]
    return o


def _jp1(a, axis=1):
    """out[..., j, ...] = a[..., j+1, ...] (out at the last index is 0)."""
    o = np.zeros_like(a)
    o[_sl(axis, a.ndim, slice(None, -1))] = a[_sl(axis, a.ndim, slice(1, None))]
    return o


def _lm1(a):
    return _jm1(a, axis=2)


def _lp1(a):
    return _jp1(a, axis=2)


def _active(lmm):
    """act[i, j, l] = l <= lmm[i, j] for 1-based i, j, l (index 0 entries False)."""
    lidx = np.arange(LMO + 1)[None, None, :]
    act = lmm[:, :, None] >= lidx
    act[:, :, 0] = False
    act[0, :, :] = False
    act[:, 0, :] = False
    return act


def _jrows(lo, hi):
    """Boolean (1, JM+1, 1) mask for j in lo..hi inclusive."""
    j = np.arange(JM + 1)
    return ((j >= lo) & (j <= hi))[None, :, None]


# ---------------------------------------------------------------- ISOSLOPE4
def isoslope4_vec(lmm, rhox, rhoy, rhomz, byrhoz, bydh, dzv, k3d):
    _, _, dxpo, dypo, _, dyvo, _, _ = geomo_dyn_arrays()
    bydyp = np.zeros_like(dypo)
    bydyp[1:] = 1.0 / dypo[1:]
    byp = bydyp[None, :, None]
    dyv0 = dyvo[None, :, None]
    dyv1 = _jm1(dyvo, axis=0)[None, :, None]            # dyvo[j-1]

    act = _active(lmm) & _jrows(2, JM)
    ll = np.arange(LMO + 1)[None, None, :]
    lm3 = lmm[:, :, None]
    ariv = k3d
    rhox_w = _west(rhox)
    rhoy_s = _jm1(rhoy)                                  # rhoy[i, j-1, l]
    byrhoz_lm1 = _lm1(byrhoz)
    rhomz_lm1 = _lm1(rhomz)
    dzv_lm1 = _lm1(dzv)

    out = {}
    with np.errstate(all="ignore"):
        byaidt = 1.0 / (4.0 * DTS * (ariv + ariv))
        pos = ariv > 0.0

        def block(branch, six_a, six_b, siy_a, siy_b, dz):
            """branch: bool mask of the 'else' (non-zero) branch; returns the four slopes' state."""
            sx_a = np.where(branch, six_a, 0.0)
            sx_b = np.where(branch, six_b, 0.0)
            sy_a = np.where(branch, siy_a, 0.0)
            sy_b = np.where(branch, siy_b, 0.0)
            dsq = dz ** 2 * byaidt
            m = branch & pos
            ast = []
            for s in (sx_a, sx_b, sy_a, sy_b):
                a = np.where(branch, ariv, 0.0)
                a = np.where(m & (s ** 2 > dsq), a * dsq / s ** 2, a)
                ast.append(a)
            return sx_a, sx_b, sy_a, sy_b, ast

        # slopes 0,2 (RHOMZ(L))
        br0 = act & ~((ll == lm3) | (rhomz == 0.0))
        six0, six2, siy0, siy2, (aix0st, aix2st, aiy0st, aiy2st) = block(
            br0, rhox * byrhoz, rhox_w * byrhoz, rhoy * byrhoz, rhoy_s * byrhoz, dzv)
        # slopes 1,3 (RHOMZ(L-1))
        br1 = act & ~((ll == 1) | (rhomz_lm1 == 0.0))
        six1, six3, siy1, siy3, (aix1st, aix3st, aiy1st, aiy3st) = block(
            br1, rhox * byrhoz_lm1, rhox_w * byrhoz_lm1, rhoy * byrhoz_lm1, rhoy_s * byrhoz_lm1, dzv_lm1)

        z = np.zeros_like(ariv)
        for name, ast, br, dz in (("aix0", aix0st, br0, dzv), ("aix2", aix2st, br0, dzv),
                                  ("aiy0", aiy0st, br0, dzv), ("aiy2", aiy2st, br0, dzv),
                                  ("aix1", aix1st, br1, dzv_lm1), ("aix3", aix3st, br1, dzv_lm1),
                                  ("aiy1", aiy1st, br1, dzv_lm1), ("aiy3", aiy3st, br1, dzv_lm1)):
            out[name] = np.where(br, ast * dz * bydh, z)

        def w(a):
            return np.where(act, a, 0.0)

        out["asx0"] = w(aix0st * six0); out["asx1"] = w(aix1st * six1)
        out["asx2"] = w(aix2st * six2); out["asx3"] = w(aix3st * six3)
        out["asy0"] = w(aiy0st * siy0); out["asy1"] = w(aiy1st * siy1)
        out["asy2"] = w(aiy2st * siy2); out["asy3"] = w(aiy3st * siy3)
        out["s2x0"] = w(aix0st * six0 * six0); out["s2x1"] = w(aix1st * six1 * six1)
        out["s2x2"] = w(aix2st * six2 * six2); out["s2x3"] = w(aix3st * six3 * six3)
        out["s2y0"] = w(aiy0st * siy0 * siy0 * byp * dyv0)
        out["s2y1"] = w(aiy1st * siy1 * siy1 * byp * dyv0)
        out["s2y2"] = w(aiy2st * siy2 * siy2 * byp * dyv1)
        out["s2y3"] = w(aiy3st * siy3 * siy3 * byp * dyv1)
    return out


# ---------------------------------------------------------------- GMKDIF
def gmkdif_vec(lmm, kpl, aix0, aix1, aix2, aix3, aiy0, aiy1, aiy2, aiy3,
               asx0, asx1, asx2, asx3, asy0, asy1, asy2, asy3,
               s2x0, s2x1, s2x2, s2x3, s2y0, s2y1, s2y2, s2y3):
    act = _active(lmm) & _jrows(2, JM)
    deep = act & (np.arange(LMO + 1)[None, None, :] > kpl[:, :, None])
    z = np.zeros_like(aix0)
    out = {}

    # BXX is written at (IM1, J, L) from the loop's own I; BYY at (I, J-1, L)
    bx = np.where(act, aix2 + _west(aix0) + aix3 + _west(aix1), z)
    out["bxx"] = _east(bx)                                  # out[i-1] = bx[i]
    by = np.where(act, aiy2 + _jm1(aiy0) + aiy3 + _jm1(aiy1), z)
    out["byy"] = _jp1(by)                                   # out[j-1] = by[j]

    # BZZ(l-1) from layer l (l > 1), zero where the column is active but l <= KPL
    bzz_src = (s2x1 + s2x3 + _lm1(s2x0) + _lm1(s2x2) + s2y1 + s2y3 + _lm1(s2y0) + _lm1(s2y2))
    bz = np.where(deep, bzz_src, z)
    bz[:, :, 1] = 0.0                                       # l = 1 writes nothing
    out["bzz"] = _lp1(bz)                                   # out[l-1] = bz[l]

    def d(a):
        return np.where(deep, a, z)
    out["azx"] = d(asx2); out["bzx"] = d(asx0 - asx2); out["czx"] = d(-asx0)
    out["aezx"] = d(asx3); out["ezx"] = d(asx1 - asx3); out["cezx"] = d(-asx1)
    out["azy"] = d(asy2); out["bzy"] = d(asy0 - asy2); out["czy"] = d(-asy0)
    out["aezy"] = d(asy3); out["ezy"] = d(asy1 - asy3); out["cezy"] = d(-asy1)
    return out


# ---------------------------------------------------------------- GMFEXP
def _fxx_fyy_fzz_fzx_fzy_vec(lmm, lmu, lmv, kpl, tr, bxx, byy, bzz, azx, bzx, czx, aezx, ezx,
                             cezx, azy, bzy, czy, aezy, ezy, cezy, dt4, dt4dx_of_j, dt4dy_of_j,
                             bydyv, bydzv):
    ll = np.arange(LMO + 1)[None, None, :]
    rows = _jrows(2, JM - 1)
    ok = (lmm > 0)[:, :, None] & rows & (ll >= 1)
    z = np.zeros_like(tr)
    dx = dt4dx_of_j[None, :, None]
    dy = dt4dy_of_j[None, :, None]
    lmm3 = lmm[:, :, None]

    tr_w = _west(tr); tr_e = _east(tr)
    tr_lp = _lp1(tr)
    with np.errstate(all="ignore"):
        # FXX(im1) = dt4dx*bxx(im1)*(tr(im1)-tr(i)), written at im1 from the loop's own I
        cx = ok & (_west(lmu)[:, :, None] >= ll)
        vx = dx * _west(bxx) * (tr_w - tr)
        fxx = _east(np.where(cx, vx, z))
        # FYY(i,j)
        cy = ok & (lmv[:, :, None] >= ll)
        fyy = np.where(cy, dt4 * byy * (tr - _jp1(tr)) * bydyv[None, :, None], z)
        # FZZ, FZX, FZY need lmm > l
        cz = ok & (lmm3 > ll)
        fzz = np.where(cz & (kpl[:, :, None] <= ll), dt4 * bzz * (tr_lp - tr) * bydzv, z)
        fzx = dx * (bzx * tr + azx * tr_w + czx * tr_e + _lp1(ezx) * tr_lp)
        fzx = np.where(cz & (_west(lmm)[:, :, None] > ll), fzx + dx * _lp1(aezx) * _lp1(tr_w), fzx)
        fzx = np.where(cz & (_east(lmm)[:, :, None] > ll), fzx + dx * _lp1(cezx) * _lp1(tr_e), fzx)
        fzx = np.where(cz, fzx, z)
        tr_s = _jm1(tr); tr_n = _jp1(tr)
        fzy = dy * (bzy * tr + azy * tr_s + czy * tr_n + _lp1(ezy) * tr_lp)
        fzy = np.where(cz & (_jm1(lmm, axis=1)[:, :, None] > ll), fzy + dy * _lp1(aezy) * _lp1(tr_s), fzy)
        fzy = np.where(cz & (_jp1(lmm, axis=1)[:, :, None] > ll), fzy + dy * _lp1(cezy) * _lp1(tr_n), fzy)
        fzy = np.where(cz, fzy, z)
    return fxx, fyy, fzz, fzx, fzy


def _compute_fluxes_vec(lmm, lmu, lmv, mo, fxx, fyy, fzz, fzx, fzy, bxx, byy, txm0, tym0,
                        dt4, bydxp, bydyp, bydh, dxypo):
    RGMI = 1.0
    ll = np.arange(LMO + 1)[None, None, :]
    rows = _jrows(2, JM - 1)
    ok = (lmm > 0)[:, :, None] & rows & (ll >= 1)
    lmm3 = lmm[:, :, None]
    z = np.zeros_like(mo)
    dxy = dxypo[None, :, None]
    bx = bydxp[None, :, None]
    by = bydyp[None, :, None]
    flux_x = z.copy(); flux_y = z.copy(); flux_z = z.copy()
    txm = txm0.copy(); tym = tym0.copy()
    fxx_w = _west(fxx); bxx_w = _west(bxx); mo_w = _west(mo)
    fyy_s = _jm1(fyy); byy_s = _jm1(byy)

    with np.errstate(all="ignore"):
        mofx = (mo_w + mo) * dxy * bx * 0.5
        flux_x = np.where(ok & (_west(lmu)[:, :, None] >= ll), fxx_w * mofx, z)
        wet = ok & (ll <= lmm3)
        txm = np.where(wet, (txm0 - 3.0 * (fxx_w + fxx) * mo * dxy * bx) /
                       (1.0 + 6.0 * dt4 * (bxx_w + bxx) * bx ** 2), txm0)

        dxy_s = _jm1(dxypo, axis=0)[None, :, None]
        by_s = _jm1(bydyp, axis=0)[None, :, None]
        mofy = ((_jm1(mo) * by_s * dxy_s) + (mo * by * dxy)) * 0.5
        flux_y = np.where(ok & (_jm1(lmv, axis=1)[:, :, None] >= ll), fyy_s * mofy, z)
        tym = np.where(wet, (tym0 - 3.0 * (fyy_s + fyy) * mo * dxy * by) /
                       (1.0 + 6.0 * dt4 * (byy_s + byy) * by ** 2), tym0)

        mofz = ((_lp1(mo) * _lp1(bydh)) + (mo * bydh)) * dxy * 0.5
        flux_z = np.where(ok & (lmm3 > ll), (fzz + (fzx + fzy) * (1.0 + RGMI)) * mofz, z)

        # pole boxes (one set of arrays per pole; flux_y at J=JM / J=1 for every I, flux_z at I=1)
        for jp, jn in ((JM, JM - 1), (1, 2)):               # (pole row, adjacent row)
            sel = (lmv[1:IM + 1, jn][:, None] >= ll[0, :, :])           # (IM, LMO+1)
            sel[:, 0] = False
            mofy_p = ((mo[1:IM + 1, jn, :] * bydyp[jn] * dxypo[jn]) +
                      (mo[1, jp, :][None, :] * bydyp[jp] * dxypo[jp])) * 0.5
            flux_y[1:IM + 1, jp, :] = np.where(sel, fyy[1:IM + 1, jn, :] * mofy_p, 0.0)
            mofz_p = ((_jp1(mo[1, jp, :], axis=0) * _jp1(bydh[1, jp, :], axis=0)) +
                      (mo[1, jp, :] * bydh[1, jp, :])) * dxypo[jp] * 0.5
            fz_p = (fzz[1, jp, :] + fzy[1, jp, :] * (1.0 + RGMI)) * mofz_p
            ok_z = (lmm[1, jp] > np.arange(LMO + 1)) & (np.arange(LMO + 1) >= 1)
            flux_z[1, jp, :] = np.where(ok_z, fz_p, 0.0)
    return flux_x, flux_y, flux_z, txm, tym


def _wrap_adjust_fluxes_vec(lmm, trm0, flux_x, flux_y, flux_z):
    trm = trm0.copy()
    ll = np.arange(LMO + 1)
    conv = np.zeros_like(trm0)
    # interior xy convergence (j = 2..JM-1): ((fx - fx_east) + fy) - fy_north
    xy = ((flux_x - _east(flux_x)) + flux_y) - _jp1(flux_y)
    rows = _jrows(2, JM - 1)
    conv = np.where(rows, xy, 0.0)
    # pole boxes: sequential Fortran SUM over I, divided by IM (cumsum = left-to-right)
    conv[1, JM, :] = np.cumsum(flux_y[1:IM + 1, JM, :], axis=0)[-1] / IM
    conv[1, 1, :] = np.cumsum(flux_y[1:IM + 1, 1, :], axis=0)[-1] / IM
    # vertical: conv[l] = xy[l] (+ fz[l] if l < LMO) (- fz[l-1] if l > 1), in that order
    fz = flux_z.copy()
    fz[:, :, LMO] = 0.0
    zmask = np.zeros(conv.shape, dtype=bool)
    zmask[:, 2:JM, :] = True
    zmask[1, JM, :] = True
    zmask[1, 1, :] = True
    zmask[:, :, 0] = False
    add = np.where(zmask, fz, 0.0)
    sub = np.where(zmask, _lm1(flux_z), 0.0)
    conv = np.where(zmask, (conv + add) - sub, conv)
    conv[:, :, 0] = 0.0
    for j in (JM, 1):
        trm[2:IM + 1, j, 1:] = trm[1, j, 1:][None, :]
        conv[2:IM + 1, j, 1:] = conv[1, j, 1:][None, :]

    act = _active(lmm)
    plus = act & (conv > 0.0)
    neg = act & ~(conv > 0.0) & (conv < -trm)
    posval = np.where(plus, conv, 0.0)
    adjval = np.where(neg, -trm - conv, 0.0)
    conv = np.where(neg, -trm, conv)
    # per-J sequential sums in the scalar's (l outer, i inner) order
    def jsum(v):
        flat = v[1:IM + 1, :, 1:].transpose(1, 2, 0).reshape(JM + 1, LMO * IM)   # (j, l*IM+i)
        return np.cumsum(flat, axis=1)[:, -1]
    convpos_j = jsum(posval)
    convadj_j = jsum(adjval)
    sumpos = np.cumsum(convpos_j[1:JM + 1])[-1]
    sumadj = np.cumsum(convadj_j[1:JM + 1])[-1]
    posadj = 1.0 - sumadj / sumpos if sumpos > 0.0 else 1.0

    conv = np.where(act & (conv > 0.0), conv * posadj, conv)
    trm = np.where(act, np.maximum(0.0, trm + conv), trm)
    return trm


def _add_fluxes_vec(lmm, lmu, lmv, trm0, flux_x, flux_y, flux_z):
    trm = trm0.copy()
    ll = np.arange(LMO + 1)[None, None, :]
    rows = _jrows(2, JM - 1)
    z = np.zeros_like(trm)
    wet = (lmm > 0)[:, :, None] & (ll >= 1)
    # masks of the contributions made at the loop's own (i, j, l)
    mx = wet & rows & (_west(lmu)[:, :, None] >= ll)
    my = wet & rows & (_jm1(lmv, axis=1)[:, :, None] >= ll)
    fx = np.where(mx, flux_x, z)                      # contribution of face (i) (+ to i, - to i-1)
    fy = np.where(my, flux_y, z)                      # contribution of face (j) (+ to j, - to j-1)
    # cell-by-cell term order (see module docstring)
    t = trm
    # west-wrap term at the first iteration, only cell IM:  trm[IM] -= fx[1]
    first = z.copy()
    first[IM] = fx[1]
    t = t - first
    t = t + fx
    t = t + fy
    t = t - _east_noWrap(fx)
    t = t - _jp1(fy)
    # pole boxes: north pole -> trm[i, JM-1, :], south pole -> trm[i, 2, :]
    selN = (lmv[1:IM + 1, JM - 1][:, None] >= ll[0]) & (ll[0] >= 1)
    selS = (lmv[1:IM + 1, 2][:, None] >= ll[0]) & (ll[0] >= 1)
    fN = np.where(selN, flux_y[1:IM + 1, JM, :], 0.0)
    fS = np.where(selS, flux_y[1:IM + 1, 1, :], 0.0)
    t[1:IM + 1, JM - 1, :] = t[1:IM + 1, JM - 1, :] - fN
    t[1:IM + 1, 2, :] = t[1:IM + 1, 2, :] - fS
    t[1, JM, :] = t[1, JM, :] + np.cumsum(fN, axis=0)[-1] / IM
    t[1, 1, :] = t[1, 1, :] + np.cumsum(fS, axis=0)[-1] / IM
    t[:, :, 0] = trm0[:, :, 0]
    trm = t
    # vertical redistribution: sequential over layers
    for l in range(1, LMO):
        m = (_jrows(2, JM - 1)[:, :, 0] & (lmm > 0)[:, :] & (lmm > l))
        rf = np.where(m, flux_z[:, :, l], 0.0)
        trm[:, :, l] = trm[:, :, l] + rf
        trm[:, :, l + 1] = trm[:, :, l + 1] - rf
        for jp in (JM, 1):
            if lmm[1, jp] >= l and lmm[1, jp] > l:
                rfzt = flux_z[1, jp, l]
                trm[1, jp, l] = trm[1, jp, l] + rfzt
                trm[1, jp, l + 1] = trm[1, jp, l + 1] - rfzt
    return trm


def _east_noWrap(a):
    """out[i] = a[i+1] for i = 1..IM-1, out[IM] = 0 (the wrap term is applied first, separately)."""
    o = np.zeros_like(a)
    o[1:IM] = a[2:IM + 1]
    return o


def gmfexp_vec(lmm, lmu, lmv, mo, trm0, txm0, tym0, tzm0, qlimit,
               bxx, byy, bzz, azx, bzx, czx, aezx, ezx, cezx,
               azy, bzy, czy, aezy, ezy, cezy, kpl, bydh, bydzv):
    _, _, dxpo, dypo, _, dyvo, _, dxypo = geomo_dyn_arrays()
    bydxp = np.zeros_like(dxpo); bydxp[1:] = 1.0 / dxpo[1:]
    bydyp = np.zeros_like(dypo); bydyp[1:] = 1.0 / dypo[1:]
    bydyv = np.zeros_like(dyvo); bydyv[1:JM] = 1.0 / dyvo[1:JM]
    dt4 = 0.25 * DTS
    dt4dx_of_j = dt4 * bydxp
    dt4dy_of_j = dt4 * bydyp

    act = _active(lmm)
    with np.errstate(all="ignore"):
        tr = np.where(act, trm0 / (dxypo[None, :, None] * mo), 0.0)

    fxx, fyy, fzz, fzx, fzy = _fxx_fyy_fzz_fzx_fzy_vec(
        lmm, lmu, lmv, kpl, tr, bxx, byy, bzz, azx, bzx, czx, aezx, ezx, cezx,
        azy, bzy, czy, aezy, ezy, cezy, dt4, dt4dx_of_j, dt4dy_of_j, bydyv, bydzv)
    flux_x, flux_y, flux_z, txm, tym = _compute_fluxes_vec(
        lmm, lmu, lmv, mo, fxx, fyy, fzz, fzx, fzy, bxx, byy, txm0, tym0,
        dt4, bydxp, bydyp, bydh, dxypo)
    if qlimit:
        trm = _wrap_adjust_fluxes_vec(lmm, trm0, flux_x, flux_y, flux_z)
    else:
        trm = _add_fluxes_vec(lmm, lmu, lmv, trm0, flux_x, flux_y, flux_z)
    return trm, txm, tym, tzm0.copy()
