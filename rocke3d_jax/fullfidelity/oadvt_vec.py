"""Batched OADVTX2 (D81), OADVTY2 and OADVTZ2 (D80): numpy ports of oadvt2_ff.oadvtx2 / oadvty2 /
oadvtz2 with the independent grid dimensions vectorized.

OADVTX2: each (l, j) pass is independent in the tracer state, but the Fortran's single MUDT array
carries stale values between passes (see oadvt2_ff.oadvtx2). MUDT, NCOURANT and the skip logic depend
only on MU, MM-in and the masks, so they are precomputed per pass with the scalar logic (`_x_prepass`,
cheap), then the sequential-in-i flux sweep runs over all (l, j) lanes at once.

OADVTY2: the north-south sweep is sequential in j but independent across (i, l), so the j loop is
kept (46 steps) and (i, l) is vectorized. OADVTZ2: the vertical sweep is sequential in l but
independent across (i, j), so the l loop is kept (13 steps) and (i, j) is vectorized. The
accumulation order of every update is the scalar port's, so results agree to rounding. The pole
average uses cumsum (sequential, as the Fortran SUM under -fp-model strict). Arrays are 1-based,
(IM+1, JM+1, LMO+1), as in oadvt2_ff. Validated by oadvt_vec_compare.py and tests/test_oadvt_vec.py.
"""
import numpy as np

from oadvt2_ff import IM, JM, LMO, _get_i1i2


def _sign(a, b):
    """Fortran SIGN(A,B) elementwise (B=0 treated as positive)."""
    return np.where(b != 0.0, np.copysign(a, b), np.abs(a))


def _x_prepass(mm, mu, dt, lmu, lmm):
    """The MUDT / courant part of oadvt2_ff.oadvtx2 for every (l, j) pass, in the scalar order.
    Returns mudt_snap (LMO+1, JM+1, IM+1) = MUDT as seen by each pass's flux sweep (after courant
    scaling, including its stale entries), ncourant (LMO+1, JM+1) and lane (bool, pass has any
    U-active cell)."""
    mudt = np.zeros(IM + 1)
    snap = np.zeros((LMO + 1, JM + 1, IM + 1))
    ncour = np.zeros((LMO + 1, JM + 1), int)
    lane = np.zeros((LMO + 1, JM + 1), bool)
    for l in range(1, LMO + 1):
        for j in range(2, JM):
            active_u = np.zeros(IM + 1, dtype=bool)
            active_u[1:] = lmu[1:IM + 1, j] >= l
            if not np.any(active_u):
                continue
            lane[l, j] = True
            courmax = 0.0
            mudt[1] = mu[1, j, l] * dt
            mudt[2] = mu[2, j, l] * dt
            mudt[IM] = mu[IM, j, l] * dt
            i = 1
            if lmu[i, j] >= l:
                if mudt[i] >= 0.0:
                    mcheck = mm[i, j, l] + min(0.0, mudt[IM] - mudt[i])
                else:
                    mcheck = -mm[i + 1, j, l] - min(0.0, mudt[i] - mudt[i + 1])
                courmax = max(courmax, mudt[i] / mcheck)
            u_segs = _get_i1i2(active_u)
            for (i1, i2) in u_segs:
                i = i1
                if i > 1:
                    mudt[i - 1] = 0.0
                mudt[i] = mu[i, j, l] * dt
                for i in range(max(2, i1), min(i2, IM - 1) + 1):
                    mudt[i + 1] = mu[i + 1, j, l] * dt
                    if mudt[i] >= 0.0:
                        mcheck = mm[i, j, l] + min(0.0, mudt[i - 1] - mudt[i])
                    else:
                        mcheck = -mm[i + 1, j, l] - min(0.0, mudt[i] - mudt[i + 1])
                    courmax = max(courmax, mudt[i] / mcheck)
            i = IM
            if lmu[i, j] >= l:
                if mudt[i] >= 0.0:
                    mcheck = mm[i, j, l] + min(0.0, mudt[i - 1] - mudt[i])
                else:
                    mcheck = -mm[1, j, l] - min(0.0, mudt[i] - mudt[1])
                courmax = max(courmax, mudt[i] / mcheck)
            if courmax > 1.0:
                ncourant = 1 + int(courmax)
                zcourant = 1.0 / ncourant
                for (i1, i2) in u_segs:
                    for i in range(i1, i2 + 1):
                        mudt[i] *= zcourant
            else:
                ncourant = 1
            ncour[l, j] = ncourant
            snap[l, j] = mudt
    return snap, ncour, lane


def oadvtx2_vec(rm, rx, ry, rz, mm, mu, dt, qlimit, lmu, lmm):
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy(); mm = mm.copy()
    rxlimit = 1.0 if qlimit else 0.0
    snap, ncour, lane_all = _x_prepass(mm, mu, dt, lmu, lmm)

    # lanes = (l, j) passes with any U-active cell; arrays become (nlane, IM+1) with 1-based i
    ls, js = np.nonzero(lane_all)
    n = len(ls)
    if n == 0:
        return rm, rx, ry, rz, mm
    def pull(a):
        return a[:, js, ls].T.copy()          # (n, IM+1)
    R, X, Y, Z, M = pull(rm), pull(rx), pull(ry), pull(rz), pull(mm)
    mudt = snap[ls, js]                       # (n, IM+1)
    nc = ncour[ls, js]
    lm_m = (lmm[:, js] >= ls[None, :]).T      # (n, IM+1) M-active
    lm_u = (lmu[:, js] >= ls[None, :]).T
    lm_m[:, 0] = False

    # which i are processed in the basins loop: M-active, i <= IM-1, not a skipped single-cell segment
    left = np.zeros_like(lm_m); left[:, 2:] = lm_m[:, 1:-1]       # active at i-1
    right = np.zeros_like(lm_m); right[:, 1:-1] = lm_m[:, 2:]     # active at i+1
    single = lm_m & ~left & ~right
    single[:, 1] = False                                           # i1 == 1 segments are processed
    proc = lm_m & ~single
    proc[:, IM] = False

    rows = np.arange(n)
    def sign(a, b):
        return np.where(b != 0.0, np.copysign(a, b), np.abs(a))

    def flux(am, c_pos, c_neg, act):
        """Flux through the face of the donor cell (c_pos if am >= 0 else c_neg), with the in-place
        RX limiter. Returns (fm, fx, fy, fz) and writes the limited RX back."""
        pos = am >= 0.0
        cell = np.where(pos, c_pos, c_neg)
        rm_c, rx_c = R[rows, cell], X[rows, cell]
        a = am / M[rows, cell]
        lim = rx_c - rxlimit * sign(np.minimum(0.0, rm_c - np.abs(rx_c)), rx_c)
        X[rows, cell] = np.where(act, lim, rx_c)
        rx_c = np.where(act, lim, rx_c)
        fm = np.where(pos, a * (rm_c + (1 - a) * rx_c), a * (rm_c - (1 + a) * rx_c))
        fx = am * (a * a * rx_c - 3 * fm)
        fy = a * Y[rows, cell]
        fz = a * Z[rows, cell]
        return fm, fx, fy, fz

    def update(i, am, fm, fx, fy, fz, amim1, fmim1, fxim1, fyim1, fzim1, act):
        mi, ri, xi = M[rows, i], R[rows, i], X[rows, i]
        yi, zi = Y[rows, i], Z[rows, i]
        mmnew = mi + (amim1 - am)
        rnew = ri + (fmim1 - fm)
        xnew = (xi * mi + (fxim1 - fx) + 3.0 * ((amim1 + am) * rnew - mi * (fmim1 + fm))) / mmnew
        R[rows, i] = np.where(act, rnew, ri)
        X[rows, i] = np.where(act, xnew, xi)
        Y[rows, i] = np.where(act, yi + (fyim1 - fy), yi)
        Z[rows, i] = np.where(act, zi + (fzim1 - fz), zi)
        M[rows, i] = np.where(act, mmnew, mi)

    maxnc = int(nc.max())
    IMv = np.full(n, IM)
    ones = np.ones(n, int)
    with np.errstate(all="ignore"):
        for k in range(maxnc):
            lane = nc > k
            # dateline flux (I = IM)
            du = lane & lm_u[:, IM]
            am = np.where(du, mudt[:, IM], 0.0)
            fm, fx, fy, fz = flux(am, IMv, ones, du)
            prev = [np.where(du, v, 0.0) for v in (am, fm, fx, fy, fz)]
            dl = list(prev)
            for i in range(1, IM):
                act = lane & proc[:, i]
                ii = np.full(n, i)
                am = mudt[:, i]
                fm, fx, fy, fz = flux(am, ii, ii + 1, act)
                update(ii, am, fm, fx, fy, fz, *prev, act)
                prev = [np.where(act, v, p) for v, p in zip((am, fm, fx, fy, fz), prev)]
            # dateline update
            act = lane & lm_m[:, IM]
            update(IMv, dl[0], dl[1], dl[2], dl[3], dl[4], *prev, act)

    for a, A in ((rm, R), (rx, X), (ry, Y), (rz, Z), (mm, M)):
        a[:, js, ls] = A.T
    return rm, rx, ry, rz, mm


def oadvty2_vec(rm, rx, ry, rz, mo, mv, dt, qlimit, lmm, lmv):
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy(); mo = mo.copy()
    rylimit = 1.0 if qlimit else 0.0
    lidx = np.arange(1, LMO + 1)[None, :]
    S = (slice(1, IM + 1), slice(None), slice(1, LMO + 1))

    # pole fill (row JM copied across longitude, rx/ry zeroed) for every layer at once
    for a in (mo, rm, rz):
        a[2:IM + 1, JM, 1:] = a[1, JM, 1:][None, :]
    rx[1:IM + 1, JM, 1:] = 0.0
    ry[1:IM + 1, JM, 1:] = 0.0

    z = np.zeros((IM, LMO))
    bmjm1, fmjm1, fxjm1, fyjm1, fzjm1 = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()

    with np.errstate(all="ignore"):
        for j in range(1, JM + 1):
            jp = min(j + 1, JM)
            act = (lmv if j == 1 else lmm)[1:IM + 1, j][:, None] >= lidx
            bm = mv[1:IM + 1, j, 1:] * dt
            pos = bm >= 0.0
            neg = ~pos
            if j == JM:
                neg = np.zeros_like(neg)       # SMV(:, JM, :) == 0, the j+1 branch is unreachable
                pos = np.ones_like(pos)
            # limiter on the donor cell's RY (in place, as in the Fortran)
            ry_j = ry[1:IM + 1, j, 1:]
            ry_p = ry[1:IM + 1, jp, 1:]
            ry_s = np.where(pos, ry_j, ry_p)
            rm_s = np.where(pos, rm[1:IM + 1, j, 1:], rm[1:IM + 1, jp, 1:])
            lim = ry_s - rylimit * _sign(np.minimum(0.0, rm_s - np.abs(ry_s)), ry_s)
            ry[1:IM + 1, j, 1:] = np.where(act & pos, lim, ry_j)
            if jp != j:
                ry[1:IM + 1, jp, 1:] = np.where(act & neg, lim, ry_p)
            ry_s = np.where(act, lim, ry_s)
            mo_s = np.where(pos, mo[1:IM + 1, j, 1:], mo[1:IM + 1, jp, 1:])
            b = bm / mo_s
            fm = np.where(pos, b * (rm_s + (1 - b) * ry_s), b * (rm_s - (1 + b) * ry_s))
            fy = bm * (b * b * ry_s - 3 * fm)
            fx = b * np.where(pos, rx[1:IM + 1, j, 1:], rx[1:IM + 1, jp, 1:])
            fz = b * np.where(pos, rz[1:IM + 1, j, 1:], rz[1:IM + 1, jp, 1:])

            if j >= 2:
                mo_j = mo[1:IM + 1, j, 1:]
                rm_j = rm[1:IM + 1, j, 1:]
                ry_j = ry[1:IM + 1, j, 1:]
                rx_j = rx[1:IM + 1, j, 1:]
                rz_j = rz[1:IM + 1, j, 1:]
                mnew = mo_j + (bmjm1 - bm)
                rm_new = rm_j + (fmjm1 - fm)
                ry_new = (ry_j * mo_j + (fyjm1 - fy) +
                          3.0 * ((bmjm1 + bm) * rm_new - mo_j * (fmjm1 + fm))) / mnew  # uses the updated RM, as the Fortran
                rx[1:IM + 1, j, 1:] = np.where(act, rx_j + (fxjm1 - fx), rx_j)
                rz[1:IM + 1, j, 1:] = np.where(act, rz_j + (fzjm1 - fz), rz_j)
                rm[1:IM + 1, j, 1:] = np.where(act, rm_new, rm_j)
                ry[1:IM + 1, j, 1:] = np.where(act, ry_new, ry_j)
                mo[1:IM + 1, j, 1:] = np.where(act, mnew, mo_j)
            bmjm1 = np.where(act, bm, bmjm1)
            fmjm1 = np.where(act, fm, fmjm1)
            fxjm1 = np.where(act, fx, fxjm1)
            fyjm1 = np.where(act, fy, fyjm1)
            fzjm1 = np.where(act, fz, fzjm1)

    # average the pole (sequential accumulation over i, as Fortran SUM)
    pole = lmm[1, JM] >= np.arange(1, LMO + 1)
    for a in (mo, rm, rz):
        avg = np.cumsum(a[1:IM + 1, JM, 1:], axis=0)[-1] / IM
        a[1:IM + 1, JM, 1:] = np.where(pole[None, :], avg[None, :], a[1:IM + 1, JM, 1:])
    rx[1:IM + 1, JM, 1:] = np.where(pole[None, :], 0.0, rx[1:IM + 1, JM, 1:])
    ry[1:IM + 1, JM, 1:] = np.where(pole[None, :], 0.0, ry[1:IM + 1, JM, 1:])
    return rm, rx, ry, rz, mo


def oadvtz2_vec(rm, rx, ry, rz, mo, mw, dt, qlimit, lmm):
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy(); mo = mo.copy()
    rzlim = 1.0 if qlimit else 0.0
    no_rzlim = 1.0 - rzlim
    edgmax = 1.5
    shp = (IM + 1, JM + 1)
    cmup, fmup, fxup, fyup, fzup, fmup_ctr = [np.zeros(shp) for _ in range(6)]
    I = np.arange(IM + 1)[:, None]
    J = np.arange(JM + 1)[None, :]
    pole_cols = (J == JM)

    def at(a, idx):
        return np.take_along_axis(a, idx[:, :, None], axis=2)[:, :, 0]

    with np.errstate(all="ignore"):
        for l in range(1, LMO + 1):
            lp = min(l + 1, LMO)       # layer l+1 is only read where cm < 0, unreachable at l == LMO
            M = np.where(pole_cols, (I == 1) & (lmm[1, JM] >= l), lmm >= l)
            M[0, :] = False
            M[:, 0] = False
            mo_l, rm_l = mo[:, :, l], rm[:, :, l]
            cm = dt * mw[:, :, l]
            ldn = np.minimum(l + 1, lmm)
            mo_dn, rm_dn = at(mo, ldn), at(rm, ldn)
            wtdn = mo_l / (mo_l + mo_dn)
            rdn = rm_dn / mo_dn
            rup = rm_l / mo_l
            r_edge0 = wtdn * rdn + (1.0 - wtdn) * rup
            pos = cm >= 0.0
            neg = ~pos

            # limiter on the donor cell's RZ (layer l if cm >= 0, layer l+1 otherwise), in place
            rz_l = rz[:, :, l]
            rz_n = rz[:, :, lp]
            rz_s = np.where(pos, rz_l, rz_n)
            rm_s = np.where(pos, rm_l, rm[:, :, lp])
            lim = rz_s - rzlim * _sign(np.minimum(0.0, rm_s - np.abs(rz_s)), rz_s)
            rz[:, :, l] = np.where(M & pos, lim, rz_l)
            if lp != l:
                rz[:, :, lp] = np.where(M & neg, lim, rz_n)
            rz_s = np.where(M, lim, rz_s)
            mo_s = np.where(pos, mo_l, mo[:, :, lp])
            c = cm / mo_s
            fm = np.where(pos, c * (rm_s + (1.0 - c) * rz_s), c * (rm_s - (1.0 + c) * rz_s))
            fx = c * np.where(pos, rx[:, :, l], rx[:, :, lp])
            fy = c * np.where(pos, ry[:, :, l], ry[:, :, lp])
            fz = cm * (c * c * rz_s - 3.0 * fm)
            r_edge = np.where(pos,
                              r_edge0 * no_rzlim + rzlim * np.minimum(r_edge0, edgmax * rup),
                              r_edge0 * no_rzlim + rzlim * np.minimum(r_edge0, edgmax * rdn))
            fm_ctr = np.where(pos, cm * (c * rup + (1.0 - c) * r_edge),
                              cm * (-c * rdn + (1.0 + c) * r_edge))

            mo_l = mo[:, :, l]
            rm_l = rm[:, :, l]
            rz_l = rz[:, :, l]
            rm_lus = rm_l + (fmup - fm)
            rm_new = rm_l + (fmup_ctr - fm_ctr)
            mnew = mo_l + cmup - cm
            rz_new = (rz_l * mo_l + (fzup - fz) + 3.0 *
                      ((cmup + cm) * rm_lus - mo_l * (fmup + fm))) / mnew
            rm[:, :, l] = np.where(M, rm_new, rm_l)
            rz[:, :, l] = np.where(M, rz_new, rz_l)
            mo[:, :, l] = np.where(M, mnew, mo_l)
            rx[:, :, l] = np.where(M, rx[:, :, l] + (fxup - fx), rx[:, :, l])
            ry[:, :, l] = np.where(M, ry[:, :, l] + (fyup - fy), ry[:, :, l])
            cmup = np.where(M, cm, cmup)
            fmup = np.where(M, fm, fmup)
            fmup_ctr = np.where(M, fm_ctr, fmup_ctr)
            fzup = np.where(M, fz, fzup)
            fxup = np.where(M, fx, fxup)
            fyup = np.where(M, fy, fyup)
    return rm, rx, ry, rz, mo


def oadvt2_vec(mmi, rm, rx, ry, rz, dt, qlimit, smu, smv, smw, lmu, lmv, lmm):
    """OADVT2 with the X, Y and Z sweeps batched."""
    ma = mmi.copy()
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy()
    rm, rx, ry, rz, ma = oadvtx2_vec(rm, rx, ry, rz, ma, smu, 0.5 * dt, qlimit, lmu, lmm)
    rm, rx, ry, rz, ma = oadvty2_vec(rm, rx, ry, rz, ma, smv, dt, qlimit, lmm, lmv)
    rm, rx, ry, rz, ma = oadvtz2_vec(rm, rx, ry, rz, ma, smw, dt, qlimit, lmm)
    rm, rx, ry, rz, ma = oadvtx2_vec(rm, rx, ry, rz, ma, smu, 0.5 * dt, qlimit, lmu, lmm)
    rm[1:IM + 1, JM, 1:] = rm[1, JM, 1:][None, :]
    rz[1:IM + 1, JM, 1:] = rz[1, JM, 1:][None, :]
    return ma, rm, rx, ry, rz
