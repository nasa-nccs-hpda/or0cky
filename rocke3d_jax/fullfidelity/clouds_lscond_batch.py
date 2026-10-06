"""Batched LSCOND (CLOUDS2.F90:3211-5286): all columns of a step at once with numpy -- D131.

The validated per-column port is clouds_lscond_ff.py (+ clouds_lscond_size_ff.py).  This module repeats the same statements, in the
same order, with every scalar replaced by a vector over the N columns of the batch (the columns are independent; the only coupling
between columns is the module-array carry, see below).  The sequential structure is kept exactly:
  * the top-down layer loop `do L=LMCLD,1,-1` (with the downward precipitation carry PREBAR/PREICE/LHP) is a Python loop over L, each
    statement being a vector operation; every `if` of the Fortran is an `np.where`/mask on the column axis;
  * the CTEI loop `do L=LMCLD-1,1,-1` filters the columns through the chain of `cycle` tests with masks and then COMPACTS the columns that
    really mix (a handful per layer) before running the bounded `do ITER=1,9` iteration with a per-column active mask (a column that
    exits keeps the values of its last executed iteration);
  * the particle-size / optical-thickness tail (D107) is the same two L loops, vectorised.
Exactness rules (why the result can be bit-for-bit with the per-column port):
  * +, -, *, / and sqrt are IEEE-exact in numpy, and the parenthesisation / left-to-right order of every expression is copied from the
    scalar port (which copied the Fortran, -fp-model strict, no FMA).  Python `max(a,b)` / `min(a,b)` (first argument wins ties) are
    reproduced by `pymax`/`pymin` (np.maximum/np.minimum differ in the sign of zero and NaN handling);
  * exp and pow are NOT vectorised through numpy ufuncs: they are evaluated element by element by the same scalar function the per-column
    port uses (`clouds_lscond_size_ff.ex/pw`: glibc libm, or Intel libimf in the imf mode) and ONLY on the columns on which the
    Fortran reaches that statement (so no transient overflow in an untaken branch).  The one exception is get_dq_cond/get_dq_evap in libm
    mode: the per-column port calls the numpy helper clouds_dq_ff there, and so does this module (same function, array arguments);
  * np.where evaluates both branches, so untaken branches may produce inf/nan transients which are discarded (errstate silenced); every
    exp/pow and every divide that could raise in Python is guarded by the mask.
Module-array carry: the Fortran module keeps CSIZELIP, TAUSSLIP, CLDSAL, CLDSV1, DQLSC from the previous column.  TAUSSLIP/CLDSAL/CLDSV1 are
rewritten for every layer L<=LMCLD of each column (layers above keep the value they had at entry of the batch), CSIZELIP is written only
on some layers: the carry is resolved by a forward fill along the column axis (`_ffill`).  DQLSC accumulates over columns in the Fortran
(a diagnostic never read by the model); here it is the column-local increment (documented; not part of any exit field).
Not batched: columns with KMAX>4 (the two poles, UM/VM carry 72 rows): the caller runs them with the per-column port.
Array layout inside: S[name] is (LM, N) ((LM+1, N) for lhp/prebar1/precnvl, (LM, 9, N) for moments, (LM, 4, N) for um/vm).
"""
import numpy as np

import clouds_dq_ff as dq
import clouds_lscond_ff as L0
import clouds_lscond_size_ff as sz
import clouds_massflux_ff as mf

LM = 40
NMOM = 9
TEENY, LHE, LHS, LHM, TF = L0.TEENY, L0.LHE, L0.LHS, L0.LHM, L0.TF
SHA, BYSHA, BYGRAV, BY3, TWOPI, RGAS, GRAV = L0.SHA, L0.BYSHA, L0.BYGRAV, L0.BY3, L0.TWOPI, L0.RGAS, L0.GRAV
RVAP, BYMRAT, DELTX, SLHE = L0.RVAP, L0.BYMRAT, L0.DELTX, L0.SLHE
COESIG, COEEC, TMAX_ICE, TMIN_WATER = L0.COESIG, L0.COEEC, L0.TMAX_ICE, L0.TMIN_WATER
CM00LIQ, CM00ICE, GBYAIRM0 = L0.CM00LIQ, L0.CM00ICE, L0.GBYAIRM0
_F02, _F23816, _F20783, _F999999 = L0._F02, L0._F23816, L0._F20783, L0._F999999
_A, _B, _C = L0._A, L0._B, L0._C
XYMOMS, ZMOMS = L0.XYMOMS, L0.ZMOMS
SNDL, SNDI = 174.0, 0.06417127


# ---------------------------------------------------------------------------------------------- scalar-semantics helpers
def pymax(a, b):
    """Python max(a, b): a unless b > a."""
    return np.where(b > a, b, a)


def pymin(a, b):
    """Python min(a, b): a unless b < a."""
    return np.where(b < a, b, a)


def exm(x, m):
    """exp(x) element by element (scalar libm / libimf of the per-column port) on the columns where m; 0 elsewhere."""
    r = np.zeros(x.shape)
    idx = np.flatnonzero(m)
    if idx.size:
        ex = sz.ex
        r[idx] = [ex(v) for v in x[idx].tolist()]
    return r


def pwm(x, y, m):
    """x ** y element by element (scalar pow) on the columns where m; y scalar or array; 0 elsewhere."""
    r = np.zeros(x.shape)
    idx = np.flatnonzero(m)
    if idx.size:
        pw = sz.pw
        if np.ndim(y):
            r[idx] = [pw(a, b) for a, b in zip(x[idx].tolist(), np.asarray(y)[idx].tolist())]
        else:
            r[idx] = [pw(a, y) for a in x[idx].tolist()]
    return r


def qsat_b(tm, lh, pr, m=None):
    """Utilities.F90 QSAT = A*exp(LH*(B-C/max(130,TM)))/PR (scalar order); exp only where m."""
    if m is None:
        m = np.ones(np.shape(tm), bool)
    return _A * exm(lh * (_B - _C / pymax(130.0, tm)), m) / pr


def dqsatdt_b(tm, lh):
    return lh * _C / (tm * tm)


def _dq_adjust_imf(sm, qm, plk, mass, lhx, pl, sign):
    slh = lhx * BYSHA
    qmt = qm
    tp = sm * plk / mass
    dqsum = np.zeros(np.shape(qm))
    allm = np.ones(np.shape(qm), bool)
    for _ in range(dq.NITER):
        qst = _A * exm(lhx * (_B - _C / pymax(130.0, tp)), allm) / pl
        d = (qmt - mass * qst) / (1.0 + slh * qst * (lhx * _C / (tp * tp)))
        tp = tp + slh * d / mass
        qmt = qmt - d
        dqsum = dqsum + sign * d
    return dqsum


def _bc(a, shp):
    return np.broadcast_to(np.asarray(a, float), shp)


def dq_b(kind, sm, qm, plk, mass, lhx, pl, cond, m):
    """get_dq_cond (kind 'c', cond = qm) / get_dq_evap (kind 'e') on the columns where m -> (dqsum, f) (zeros elsewhere)."""
    shp = np.shape(m)
    outd, outf = np.zeros(shp), np.zeros(shp)
    idx = np.flatnonzero(m)
    if not idx.size:
        return outd, outf
    g = lambda a: _bc(a, shp)[idx]  # noqa: E731
    args = [g(a) for a in (sm, qm, plk, mass, lhx, pl)]
    if sz._IMF["on"]:
        ref = g(qm) if kind == "c" else g(cond)
        act = ref > 0.0
        d = np.zeros(idx.size)
        f = np.zeros(idx.size)
        if act.any():
            sub = [a[act] for a in args]
            dd = _dq_adjust_imf(*sub, +1.0 if kind == "c" else -1.0)
            r = ref[act]
            dd = pymax(0.0, pymin(dd, r))
            d[act] = dd
            f[act] = dd / r
    else:
        with np.errstate(all="ignore"):
            if kind == "c":
                d, f = dq.get_dq_cond(*args)
            else:
                d, f = dq.get_dq_evap(*args, g(cond))
    outd[idx] = d
    outf[idx] = f
    return outd, outf


def clear_b(rh, rh00, c):
    """_clear_from_rh vectorised."""
    with np.errstate(all="ignore"):
        s = np.sqrt((1.0 - rh) / ((1.0 - rh00) + TEENY))
    m1 = rh <= 1.0
    c = np.where(m1, np.where(rh00 < 1.0, s, 1.0), c)
    c = np.where(c > 1.0, 1.0, c)
    c = np.where(rh > 1.0, 0.0, c)
    return c


def _ffill(vals, setm, init):
    """Forward fill along the column axis: vals (LM,N), setm (LM,N) bool, init (LM,) value before the first column."""
    n = vals.shape[1]
    idx = np.where(setm, np.arange(n)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    got = np.take_along_axis(vals, np.maximum(idx, 0), axis=1)
    return np.where(idx >= 0, got, np.asarray(init)[:, None])


# ---------------------------------------------------------------------------------------------- main layer loop
def lscond_main_b(S, P):
    N = S["tl"].shape[1]
    dtsrc, bydtsrc = P["dtsrc"], P["bydtsrc"]
    lmcld = int(P["lmcld"])
    dcl = np.asarray(P["dcl"], int)
    cols = np.arange(N)
    pearth, wconst, scdncw = P["pearth"], P["wconst"], P["scdncw"]
    wmui, cmx, u00a, scdnci, rimax, rcldlx, rcldix = P["wmui"], P["cmx"], P["u00a"], P["scdnci"], P["rimax"], P["rcldlx"], P["rcldix"]
    zl = lambda: np.zeros((LM, N))  # noqa: E731
    z1 = lambda: np.zeros((LM + 1, N))  # noqa: E731
    er, ec, prep, qheat, qheatl, qheati = zl(), zl(), zl(), zl(), zl(), zl()
    prebar, preice, lhp = z1(), z1(), z1()
    S["cldssl"], S["taussl"], S["wmpr"], S["rh1"], S["qlss"], S["qiss"] = zl(), zl(), zl(), zl(), zl(), zl()
    S["prebar1"] = z1()
    S["sshr"], S["dctei"] = zl(), zl()
    cleara = zl()
    rhf, rh00, qsatl, sq, ath = zl(), zl(), zl(), zl(), zl()
    for i in range(lmcld):
        cleara[i] = 1.0 - S["cldsavl"][i]
        cleara[i] = np.where(S["qclx"][i] + S["qcix"][i] <= 0.0, 1.0, cleara[i])
    hcndss = np.zeros(N)
    ierr, lerr, wmerr = np.zeros(N, int), np.zeros(N, int), np.zeros(N)
    tl, ql, th, rh = S["tl"], S["ql"], S["th"], S["rh"]
    qclx, qcix, svlhxl = S["qclx"], S["qcix"], S["svlhxl"]
    fssl, airm, byam, pl, plk = S["fssl"], S["airm"], S["byam"], S["pl"], S["plk"]
    svlatl, svwmxl, qcll, qcil = S["svlatl"], S["svwmxl"], S["qcll"], S["qcil"]
    sdl, vsubl, precnvl = S["sdl"], S["vsubl"], S["precnvl"]
    pl_dcl = pl[dcl - 1, cols]
    qmom = S["qmom"]
    with np.errstate(all="ignore"):
        for L in range(lmcld, 0, -1):
            i = L - 1
            ip = i + 1
            told, qold = tl[i].copy(), ql[i].copy()
            oldlhx, oldlat = svlhxl[i].copy(), svlatl[i]
            temp = 100.0 * RGAS * tl[i] / (pl[i] * GRAV)
            if L == 1:
                vvel = -sdl[ip] * temp
            else:
                vvel = -0.5 * (sdl[i] + sdl[ip]) * temp
            vdef = vvel - vsubl[i]
            fcld = (1.0 - cleara[i]) * fssl[i] + TEENY
            r00 = np.full(N, u00a)
            r00 = np.where(pl[i] < pl_dcl, r00 / (r00 + (1.0 - r00) * S["pdsigl00"][i] / 35.0), r00)
            r00 = np.where(S["u00l"][i] > r00, S["u00l"][i], r00)
            r00 = np.where(r00 < 0.0, 0.0, r00)
            r00 = np.where(r00 > 1.0, 1.0, r00)
            rh00[i] = r00
            rhf[i] = r00 + (1.0 - cleara[i]) * (1.0 - r00)
            # ---- phase (use_vmp)
            ice = tl[i] <= TMIN_WATER
            lhx = np.where(ice, LHS, LHE)
            lhp[i] = np.where(ice, LHS, np.where(tl[i] < TF, LHS, LHE))
            isE = lhx == LHE
            qsatl[i] = qsat_b(tl[i], lhx, pl[i])
            rh1 = ql[i] / qsatl[i]
            m1 = (lhx == LHS) & (qclx[i] + qcix[i] <= 0.0)
            qsate = qsat_b(tl[i], LHE, pl[i], m1)
            rhw = (2.583 - tl[i] / _F20783) * (qsat_b(tl[i], LHS, pl[i], m1) / qsate)
            rh1 = np.where(m1 & (tl[i] < _F23816), ql[i] / (qsate * rhw), rh1)
            hchang = np.zeros(N)
            c1 = (oldlhx == LHE) & (lhx == LHS)
            hchang = np.where(c1, qcll[i] * LHM, hchang)
            c2 = (oldlhx == LHS) & (lhx == LHE)
            hchang = np.where(c2, -qcil[i] * LHM, hchang)
            c3 = (oldlat == LHE) & (lhx == LHS)
            hchang = np.where(c3, hchang + svwmxl[i] * LHM, hchang)
            c4 = (oldlat == LHS) & (lhx == LHE)
            hchang = np.where(c4, hchang - svwmxl[i] * LHM, hchang)
            # the Fortran modifies QCIX/QCLX in the four blocks above and then overwrites both in the elif chain below
            # (one branch always applies, lhx being LHE or LHS), so only the final assignments matter
            qcix[i] = np.where(isE, 0.0, np.where(c1, qcll[i] + svwmxl[i], qcil[i] + svwmxl[i]))
            qclx[i] = np.where(isE, np.where(c2, qcil[i] + svwmxl[i], qcll[i] + svwmxl[i]), 0.0)
            svlhxl[i] = lhx
            tl[i] = tl[i] + hchang / (SHA * fssl[i] + TEENY)
            th[i] = tl[i] / plk[i]
            rhi = ql[i] / qsat_b(tl[i], LHS, pl[i])
            lhp[i] = np.where((lhp[ip] == LHS) & (tl[i] < TF + dtsrc * LHM * preice[ip] * GRAV * byam[i] * BYSHA), LHS, lhp[i])
            # ---- autoconversion
            qx = np.where(isE, qclx[i], qcix[i])
            qcx = qclx[i] + qcix[i]
            mq = qcx > 0.0
            rho = 1.0e5 * pl[i] / (RGAS * tl[i])
            wtliq = np.where(tl[i] > TMAX_ICE, 1.0, np.where(tl[i] < TMIN_WATER, 0.0, (tl[i] - TMIN_WATER) / (TMAX_ICE - TMIN_WATER)))
            wtliq = np.where((oldlat == LHS) & (svwmxl[i] > 0.0), 0.0, wtliq)
            tem = wconst * wtliq + wmui * (1.0 - wtliq)
            cm0 = CM00LIQ * wtliq + CM00ICE * (1.0 - wtliq)
            mv = mq & (vdef > 0.0) & (rho * qcx < 10.0)
            cm0 = np.where(mv, cm0 * pwm(np.full(N, 10.0), -_F02 * vdef, mv), cm0)
            tem = rho * qcx / (tem * fcld + TEENY)
            tem = tem * tem
            tem = np.where(tem > 10.0, 10.0, tem)
            cm = cm0 * (1.0 - 1.0 / exm(tem * tem, mq)) + 100.0 * (prebar[ip] + precnvl[ip] * bydtsrc)
            cm = cm * cmx
            cm = np.where(cm > bydtsrc, bydtsrc, cm)
            prep[i] = np.where(mq, qcx * cm, prep[i])
            # ---- form clouds?
            low = rh1 < rh00[i]
            sq[i] = np.where(low, sq[i], lhx * qsatl[i] * dqsatdt_b(tl[i], lhx) * BYSHA)
            tem = -lhx * S["dpdt"][i] / pl[i]
            ath[i] = np.where(low, ath[i], (th[i] - S["ttoldl"][i]) * bydtsrc)
            qconv = lhx * S["aq"][i] - rh[i] * sq[i] * SHA * plk[i] * ath[i] - tem * qsatl[i] * rh[i]
            form = ~low & ((qconv > 0.0) | (qx > 0.0))
            ermax = lhx * prebar[ip] * GRAV * byam[i]
            # form branch
            rhn = pymin(rh[i], rhf[i])
            er_f = np.where(qx > 0.0, (1.0 - rhn) * (1.0 - rhn) * lhx * prebar[ip] * GBYAIRM0,
                            np.where((preice[ip] > 0.0) & (tl[i] < TF), (1.0 - rhi) * (1.0 - rhi) * lhx * prebar[ip] * GBYAIRM0,
                                     (1.0 - rh[i]) * (1.0 - rh[i]) * lhx * prebar[ip] * GBYAIRM0))
            # no-form branch
            nf = ~form
            mev = nf & (qx > 0.0)
            dqs, _ = dq_b("e", tl[i] * rh00[i] / plk[i], ql[i] * rh00[i], plk[i], rh00[i], lhx, pl[i], qx / (fssl[i] * rh00[i]), mev)
            dwdt_e = dqs * rh00[i] * fssl[i]
            qh_nf = np.where(mev, -dwdt_e * lhx * bydtsrc, 0.0)
            prep_nf = pymax(0.0, (qx - dwdt_e) * bydtsrc)
            prep[i] = np.where(mev, prep_nf, prep[i])
            S["wmpr"][i] = np.where(mev, prep[i] * dtsrc, S["wmpr"][i])
            er_n = (1.0 - rh[i]) * (1.0 - rh[i]) * lhx * prebar[ip] * GBYAIRM0
            er_n = np.where((preice[ip] > 0.0) & (tl[i] < TF), (1.0 - rhi) * (1.0 - rhi) * lhx * prebar[ip] * GBYAIRM0, er_n)
            e_raw = np.where(form, er_f, er_n)
            er[i] = pymax(0.0, pymin(e_raw, ermax))
            # cleara>0 block (form only)
            fc = form & (cleara[i] > 0.0)
            wtem = 1.0e5 * qx * pl[i] / (fcld * tl[i] * RGAS + TEENY)
            wtem = np.where(isE & (qclx[i] / fcld >= wconst * 1.0e-3), 1.0e2 * wconst * pl[i] / (tl[i] * RGAS), wtem)
            wtem = np.where(wtem < 1.0e-10, 1.0e-10, wtem)
            pe = pwm(wtem / (2.0 * BY3 * TWOPI * scdncw), BY3, fc & isE)
            pi_ = pwm(wtem / (2.0 * BY3 * TWOPI * scdnci), BY3, fc & ~isE)
            rcld = np.where(isE, rcldlx * 1.0e-6 * 100.0 * pe, pymin(rcldix * 100.0e-6 * pi_, rimax))
            ck1 = 1000.0 * lhx * lhx / (2.4e-2 * RVAP * tl[i] * tl[i])
            ck2 = 1000.0 * RGAS * tl[i] / (2.4e-3 * qsatl[i] * pl[i])
            tevap = COEEC * (ck1 + ck2) * rcld * rcld
            wmx1 = qx - prep[i] * dtsrc
            S["wmpr"][i] = np.where(fc, prep[i] * dtsrc, S["wmpr"][i])
            ecrate = (1.0 - rhf[i]) / (tevap * fcld + TEENY)
            ecrate = np.where(ecrate > bydtsrc, bydtsrc, ecrate)
            ec[i] = np.where(fc, wmx1 * ecrate * lhx, ec[i])
            # drhdt / qheat (form)
            drhdt = 2.0 * cleara[i] * cleara[i] * (1.0 - rh00[i]) * (qconv + er[i]) / lhx / (
                qx / (fcld + TEENY) + 2.0 * cleara[i] * qsatl[i] * (1.0 - rh00[i]) + TEENY)
            drhdt = np.where((er[i] == 0.0) & (qx <= 0.0), 0.0, drhdt)
            qh_f = fssl[i] * (qconv - lhx * drhdt * qsatl[i]) / (1.0 + rh[i] * sq[i])
            dwdt = qh_f / lhx - prep[i] + cleara[i] * fssl[i] * er[i] / lhx
            qxn_f = qx + dwdt * dtsrc
            neg = qxn_f < 0.0
            qxn_f = np.where(neg, 0.0, qxn_f)
            qh_f = np.where(neg, (-qx * bydtsrc + prep[i]) * lhx - cleara[i] * fssl[i] * er[i], qh_f)
            # no-form final qheat / qxnew
            qh_n = qh_nf - cleara[i] * fssl[i] * er[i]
            qh = np.where(form, qh_f, qh_n)
            qxnew = np.where(form, qxn_f, 0.0)
            qheatl[i] = np.where(isE, qh, np.where(form, qheatl[i], 0.0))
            qheati[i] = np.where(~isE, qh, np.where(form, qheati[i], 0.0))
            # ---- phase of precipitation, precip carry
            hphase = np.zeros(N)
            ch1 = (lhp[ip] == LHS) & (lhp[i] == LHE) & (preice[ip] > 0.0)
            hphase = np.where(ch1, hphase + LHM * preice[ip] * GRAV * byam[i], hphase)
            preice[ip] = np.where(ch1, 0.0, preice[ip])
            ch2 = (lhp[ip] == LHE) & (lhp[i] == LHS) & (prebar[ip] > 0.0)
            hphase = np.where(ch2, hphase - LHM * prebar[ip] * GRAV * byam[i], hphase)
            ch3 = lhp[i] != lhx
            hphase = np.where(ch3, hphase + (er[i] * cleara[i] * fssl[i] / lhx - prep[i]) * LHM, hphase)
            prebar[i] = np.where(er[i] == ermax, prebar[ip] * (1.0 - cleara[i] * fssl[i]) + airm[i] * prep[i] * BYGRAV,
                                 pymax(0.0, prebar[ip] + airm[i] * (prep[i] - er[i] * cleara[i] * fssl[i] / lhx) * BYGRAV))
            qnew = ql[i] - dtsrc * qh / (lhx * fssl[i] + TEENY)
            qn = qnew < 0.0
            qnew = np.where(qn, 0.0, qnew)
            qh = np.where(qn, ql[i] * lhx * bydtsrc * fssl[i], qh)
            dwdt1 = qh / lhx - prep[i] + cleara[i] * fssl[i] * er[i] / lhx
            qxn1 = qx + dwdt1 * dtsrc
            qxnew = np.where(qn, qxn1, qxnew)
            ie = qn & (qxn1 < 0.0)
            ierr = np.where(ie, 1, ierr)
            lerr = np.where(ie, L, lerr)
            wmerr = np.where(ie, qxn1, wmerr)
            qxnew = np.where(ie, 0.0, qxnew)
            qheatl[i] = np.where(isE & qn, qh, qheatl[i])
            qheati[i] = np.where(~isE & qn, qh, qheati[i])
            fqtow = np.zeros(N)
            pf = fssl[i] > 0.0
            qheat[i] = np.where(pf, qh, qheat[i])
            c5 = pf & (qh + cleara[i] * fssl[i] * er[i] > 0.0) & (lhx * ql[i] + dtsrc * cleara[i] * er[i] > 0.0)
            fqtow = np.where(c5, (qh + cleara[i] * fssl[i] * er[i]) * dtsrc / ((lhx * ql[i] + dtsrc * cleara[i] * er[i]) * fssl[i]), fqtow)
            ql[i] = qnew
            qmom[i] = qmom[i] * (1.0 - fqtow)[None, :]
            qclx[i] = np.where(isE, qxnew, qclx[i])
            qcix[i] = np.where(isE, qcix[i], qxnew)
            tl[i] = tl[i] + dtsrc * (qh - hphase) / (SHA * fssl[i] + TEENY)
            th[i] = tl[i] / plk[i]
            qsatc = qsat_b(tl[i], lhx, pl[i])
            rh[i] = ql[i] / qsatc
            rh1 = ql[i] / qsatc
            mS = lhx == LHS
            ca = clear_b(rh[i], rh00[i], cleara[i])
            ca = np.where(qclx[i] + qcix[i] <= 0.0, 1.0, ca)
            cleara[i] = np.where(mS, ca, cleara[i])
            qf = (ql[i] - qsatc * (1.0 - cleara[i])) / (cleara[i] + TEENY)
            qsate = qsat_b(tl[i], LHE, pl[i], mS)
            rhw = (2.583 - tl[i] / _F20783) * (qsat_b(tl[i], LHS, pl[i], mS) / qsate)
            rh1 = np.where(mS & (tl[i] < _F23816) & (qclx[i] + qcix[i] <= 0.0), qf / (qsate * rhw), rh1)
            S["rh1"][i] = rh1
            mC = rh1 > 1.0
            slh = lhx * BYSHA
            dqsum, fcond = dq_b("c", tl[i], ql[i], 1.0, 1.0, lhx, pl[i], None, mC)
            mD = mC & (dqsum > 0.0)
            tl[i] = np.where(mD, tl[i] + slh * dqsum, tl[i])
            ql[i] = np.where(mD, ql[i] - dqsum, ql[i])
            qclx[i] = np.where(mD & isE, qclx[i] + dqsum * fssl[i], qclx[i])
            qcix[i] = np.where(mD & ~isE, qcix[i] + dqsum * fssl[i], qcix[i])
            qmom[i] = np.where(mD[None, :], qmom[i] * (1.0 - fcond)[None, :], qmom[i])
            rh[i] = np.where(mC, ql[i] / qsat_b(tl[i], lhx, pl[i], mC), rh[i])
            th[i] = np.where(mC, tl[i] / plk[i], th[i])
            cleara[i] = clear_b(rh[i], rh00[i], cleara[i])
            qx2 = np.where(isE, qclx[i], qcix[i])
            cleara[i] = np.where(qx2 <= 0.0, 1.0, cleara[i])
            cleara[i] = np.where(cleara[i] < 0.0, 0.0, cleara[i])
            rhf[i] = rh00[i] + (1.0 - cleara[i]) * (1.0 - rh00[i])
            ro = (rh[i] <= rhf[i]) & (rh[i] < _F999999) & (qx2 > 0.0)
            prebar[i] = np.where(ro, prebar[i] + qx2 * airm[i] * BYGRAV * bydtsrc, prebar[i])
            mm = ro & isE & (lhp[i] == LHS)
            hch = qclx[i] * LHM
            tl[i] = np.where(mm, tl[i] + hch / (SHA * fssl[i] + TEENY), tl[i])
            th[i] = np.where(mm, tl[i] / plk[i], th[i])
            qclx[i] = np.where(ro & isE, 0.0, qclx[i])
            qcix[i] = np.where(ro & ~isE, 0.0, qcix[i])
            S["prebar1"][i] = prebar[i]
            preice[i] = np.where((prebar[i] > 0.0) & (lhp[i] == LHS), prebar[i], 0.0)
            lhp[i] = np.where(prebar[i] <= 0.0, 0.0, lhp[i])
            cleara[i] = clear_b(rh[i], rh00[i], cleara[i])
            qclx[i] = np.where(isE & (qclx[i] <= TEENY), 0.0, qclx[i])
            qcix[i] = np.where(~isE & (qcix[i] <= TEENY), 0.0, qcix[i])
            qx3 = np.where(isE, qclx[i], qcix[i])
            cleara[i] = np.where(qx3 <= 0.0, 1.0, cleara[i])
            cleara[i] = np.where(cleara[i] < 0.0, 0.0, cleara[i])
            S["cldssl"][i] = fssl[i] * (1.0 - cleara[i])
            S["cldsavl"][i] = 1.0 - cleara[i]
            hcndss = hcndss + fssl[i] * (tl[i] - told) * airm[i]
            S["sshr"][i] = S["sshr"][i] + fssl[i] * (tl[i] - told) * airm[i]
            S["dqlsc"][i] = S["dqlsc"][i] + fssl[i] * (ql[i] - qold)
    prcpss = pymax(0.0, prebar[0] * GRAV * dtsrc)
    S["lhp"] = lhp
    return dict(cleara=cleara, rhf=rhf, rh00=rh00, er=er, ec=ec, prep=prep, qheatl=qheatl, qheati=qheati, qheat=qheat, prebar=prebar,
                preice=preice, prcpss=prcpss, hcndss=hcndss, ierr=ierr, lerr=lerr, wmerr=wmerr, ckij=np.ones(N))


# ---------------------------------------------------------------------------------------------- CTEI
def _ctmix_b(rm0, rm1, mom0, mom1, fmair, fmix, frat):
    """QUSDEF CTMIX on a layer pair for M columns; mom0/mom1 (9, M) modified in place; returns new (rm0, rm1)."""
    rtemp = rm0 * (1.0 - fmix) + frat * rm1
    n1 = rm1 * (1.0 - frat) + fmix * rm0
    for m in XYMOMS:
        rt = mom0[m] * (1.0 - fmix) + frat * mom1[m]
        mom1[m] = mom1[m] * (1.0 - frat) + fmix * mom0[m]
        mom0[m] = rt
    for m in ZMOMS:
        mom0[m] = mom0[m] * (1.0 - fmair)
        mom1[m] = mom1[m] * (1.0 - fmair)
    return rtemp, n1


def lscond_ctei_b(S, W, P):
    N = S["tl"].shape[1]
    lmcld, dtsrc = int(P["lmcld"]), P["dtsrc"]
    tl, ql, th, rh = S["tl"], S["ql"], S["th"], S["rh"]
    qclx, qcix, svlhxl = S["qclx"], S["qcix"], S["svlhxl"]
    fssl, airm, byam, pl, plk = S["fssl"], S["airm"], S["byam"], S["pl"], S["plk"]
    cleara, rh00 = W["cleara"], W["rh00"]
    sm, qm = S["sm"], S["qm"]
    smom, qmom, um, vm = S["smom"], S["qmom"], S["um"], S["vm"]
    ra = P["ra"]                                   # (4, N)
    wmxm = np.zeros((LM + 1, N))
    ckij = np.ones(N)
    hcndss = W["hcndss"]
    with np.errstate(all="ignore"):
        for L in range(lmcld - 1, 0, -1):
            i = L - 1
            ip = i + 1
            lhx = svlhxl[i]
            isE = lhx == LHE
            sm[i] = th[i] * airm[i]
            qm[i] = ql[i] * airm[i]
            wmxm[i] = np.where(isE, qclx[i], qcix[i]) * airm[i]
            sm[ip] = th[ip] * airm[ip]
            qm[ip] = ql[ip] * airm[ip]
            wmxm[ip] = np.where(isE, qclx[ip], qcix[ip]) * airm[i]
            alive = ~(qclx[ip] + qcix[ip] > TEENY)
            alive &= ~((cleara[i] == 1.0) | ((cleara[i] < 1.0) & (cleara[ip] < 1.0)))
            a = np.flatnonzero(alive)
            if not a.size:
                continue
            g = lambda arr: arr[a]  # noqa: E731
            lhx_a, isE_a = g(lhx), g(isE)
            th_i, th_ip, plk_i, plk_ip = g(th[i]), g(th[ip]), g(plk[i]), g(plk[ip])
            ql_i, ql_ip, tl_i, rh_i = g(ql[i]), g(ql[ip]), g(tl[i]), g(rh[i])
            fcld = (1.0 - g(cleara[i])) * g(fssl[i]) + TEENY
            sedge = np.asarray(mf.thbar(th_ip, th_i), float)
            dse = (th_ip - sedge) * plk_ip + (sedge - th_i) * plk_i + SLHE * (ql_ip - ql_i)
            dwm = np.where(isE_a, ql_ip - ql_i + (g(qclx[ip]) - g(qclx[i])) / fcld, ql_ip - ql_i + (g(qcix[ip]) - g(qcix[i])) / fcld)
            dqsdt = dqsatdt_b(tl_i, LHE) * ql_i / (rh_i + 1.0e-30)
            beta = (1.0 + BYMRAT * tl_i * dqsdt) / (1.0 + SLHE * dqsdt)
            ckm = (1.0 + SLHE * dqsdt) * (1.0 + (1.0 - DELTX) * tl_i / SLHE) / (2.0 + (1.0 + BYMRAT * tl_i / SLHE) * SLHE * dqsdt)
            ckr = tl_i / (beta * SLHE)
            ck = dse / (SLHE * dwm + TEENY)
            ok3 = ~(ckr > ckm)
            sigk = np.zeros(a.size)
            mk5 = ok3 & (ck > ckr)
            sigk = np.where(mk5, COESIG * pwm((ck - ckr) / ((ckm - ckr) + TEENY), 5.0, mk5), 0.0)
            expst = exm(-sigk * dtsrc, ok3)
            if L <= 1:
                ckij[a[ok3]] = expst[ok3]
            dsec = dwm * tl_i / beta
            fpmax = pymin(1.0, 1.0 - expst)
            mix = ok3 & ~(ck < ckr) & ~(fpmax <= 0.0) & ~(dse >= dsec)
            if not mix.any():
                continue
            M = a[mix]                                  # global column indices of the mixing columns
            s = lambda arr: arr[mix]  # noqa: E731
            fpmax, fcld = s(fpmax), s(fcld)
            lhx_m, isE_m = s(lhx_a), s(isE_a)
            airm_i, airm_ip, byam_i, byam_ip = airm[i, M], airm[ip, M], byam[i, M], byam[ip, M]
            plk_i, plk_ip, pl_i = plk[i, M], plk[ip, M], pl[i, M]
            fssl_i, fssl_ip = fssl[i, M], fssl[ip, M]
            told, toldu, qold, qoldu = tl[i, M], tl[ip, M], ql[i, M], ql[ip, M]
            airmr = (airm_ip + airm_i) * byam_ip * byam_i
            smo1, qmo1, wmo1 = sm[i, M], qm[i, M], wmxm[i, M]
            smo2, qmo2, wmo2 = sm[ip, M], qm[ip, M], wmxm[ip, M]
            smo12 = smo1 * plk_i + smo2 * plk_ip
            umo1, vmo1, umo2, vmo2 = um[i][:, M], vm[i][:, M], um[ip][:, M], vm[ip][:, M]       # (4, Mn)
            Mn = M.size
            fplume = fpmax * fssl_i
            dfx = fplume.copy()
            active = np.ones(Mn, bool)
            R = {k: np.zeros(Mn) for k in ("smn1", "qmn1", "wmn1", "smn2", "qmn2", "wmn2", "fmix", "frat", "fmass")}
            for it in range(1, 10):
                b = np.flatnonzero(active)
                if not b.size:
                    break
                h = lambda arr: arr[b]  # noqa: E731
                dfx_b = h(dfx) * 0.5
                dfx[b] = dfx_b
                fmix = h(fplume) * h(fcld)
                fmass = fmix * h(airm_i)
                fmass = pymin(fmass, (h(airm_ip) * h(airm_i)) / (h(airm_ip) + h(airm_i)))
                fmix = fmass * h(byam_i)
                frat = fmass * h(byam_ip)
                so1, so2, qo1, qo2, wo1, wo2 = h(smo1), h(smo2), h(qmo1), h(qmo2), h(wmo1), h(wmo2)
                smn1 = so1 * (1.0 - fmix) + frat * so2
                qmn1 = qo1 * (1.0 - fmix) + frat * qo2
                wmn1 = wo1 * (1.0 - fmix) + frat * wo2
                smn2 = so2 * (1.0 - frat) + fmix * so1
                qmn2 = qo2 * (1.0 - frat) + fmix * qo1
                wmn2 = wo2 * (1.0 - frat) + fmix * wo1
                bi, bip = h(byam_i), h(byam_ip)
                tht1 = smn1 * bi / h(plk_i)
                qlt1 = qmn1 * bi
                tlt1 = tht1 * h(plk_i)
                rht1 = qlt1 / qsat_b(tlt1, h(lhx_m), h(pl_i))
                wmt1 = wmn1 * bi
                tht2 = smn2 * bip / h(plk_ip)
                qlt2 = qmn2 * bip
                wmt2 = wmn2 * bip
                sedge = np.asarray(mf.thbar(tht2, tht1), float)
                dse = (tht2 - sedge) * h(plk_ip) + (sedge - tht1) * h(plk_i) + SLHE * (qlt2 - qlt1)
                dwm = qlt2 - qlt1 + (wmt2 - wmt1) / h(fcld)
                dqsdt = dqsatdt_b(tlt1, LHE) * qlt1 / (rht1 + 1.0e-30)
                beta = (1.0 + BYMRAT * tlt1 * dqsdt) / (1.0 + SLHE * dqsdt)
                dsec = dwm * tlt1 / beta
                dsedif = dse - dsec
                fp = h(fplume)
                fp = np.where(dsedif > 1.0e-3, fp - dfx_b, fp)
                fp = np.where(dsedif < -1.0e-3, fp + dfx_b, fp)
                fplume[b] = fp
                for k, v in (("smn1", smn1), ("qmn1", qmn1), ("wmn1", wmn1), ("smn2", smn2), ("qmn2", qmn2), ("wmn2", wmn2),
                             ("fmix", fmix), ("frat", frat), ("fmass", fmass)):
                    R[k][b] = v
                ex_ = (np.abs(dsedif) <= 1.0e-3) | (fp > h(fpmax) * h(fssl_i))
                active[b[ex_]] = False
            smn1, qmn1, wmn1, smn2, qmn2, wmn2 = R["smn1"], R["qmn1"], R["wmn1"], R["smn2"], R["qmn2"], R["wmn2"]
            fmix, frat, fmass = R["fmix"], R["frat"], R["fmass"]
            smn12 = smn1 * plk_i + smn2 * plk_ip
            smn1 = smn1 - (smn12 - smo12) * airm_i / ((airm_i + airm_ip) * plk_i)
            smn2 = smn2 - (smn12 - smo12) * airm_ip / ((airm_i + airm_ip) * plk_ip)
            th[i, M] = smn1 * byam_i
            tl[i, M] = th[i, M] * plk_i
            ql[i, M] = qmn1 * byam_i
            rh[i, M] = ql[i, M] / qsat_b(tl[i, M], lhx_m, pl_i)
            qclx[i, M] = np.where(isE_m, wmn1 * byam_i, qclx[i, M])
            qcix[i, M] = np.where(isE_m, qcix[i, M], wmn1 * byam_i)
            fsslrat = fssl_i / fssl_ip
            mk = fsslrat != 1.0
            sm_ip0, qm_ip0 = sm[ip, M], qm[ip, M]
            smn2 = np.where(mk, sm_ip0 + (smn2 - sm_ip0) * fsslrat, smn2)
            qmn2 = np.where(mk, qm_ip0 + (qmn2 - qm_ip0) * fsslrat, qmn2)
            smom2_sv, qmom2_sv = smom[ip][:, M].copy(), qmom[ip][:, M].copy()
            for (rm, mom, nm) in ((sm, smom, "s"), (qm, qmom, "q")):
                m0, m1 = mom[i][:, M].copy(), mom[ip][:, M].copy()
                r0, r1 = _ctmix_b(rm[i, M], rm[ip, M], m0, m1, fmass * airmr, fmix, frat)
                rm[i, M], rm[ip, M] = r0, r1
                mom[i][:, M], mom[ip][:, M] = m0, m1
            sm[ip, M] = np.where(mk, smn2, sm[ip, M])
            qm[ip, M] = np.where(mk, qmn2, qm[ip, M])
            for mom, sv in ((smom, smom2_sv), (qmom, qmom2_sv)):
                cur = mom[ip][:, M]
                mom[ip][:, M] = np.where(mk[None, :], sv + (cur - sv) * fsslrat[None, :], cur)
            th[ip, M] = smn2 * byam_ip
            ql[ip, M] = qmn2 * byam_ip
            qclx[ip, M] = np.where(isE_m, wmn2 * byam_ip, qclx[ip, M])
            qcix[ip, M] = np.where(isE_m, qcix[ip, M], wmn2 * byam_ip)
            raM = ra[:, M]
            umn1 = umo1 * (1.0 - fmix)[None, :] + frat[None, :] * umo2
            vmn1 = vmo1 * (1.0 - fmix)[None, :] + frat[None, :] * vmo2
            umn2 = umo2 * (1.0 - frat)[None, :] + fmix[None, :] * umo1
            vmn2 = vmo2 * (1.0 - frat)[None, :] + fmix[None, :] * vmo1
            um[i][:, M] = um[i][:, M] + (umn1 - umo1) * raM
            vm[i][:, M] = vm[i][:, M] + (vmn1 - vmo1) * raM
            um[ip][:, M] = um[ip][:, M] + (umn2 - umo2) * raM
            vm[ip][:, M] = vm[ip][:, M] + (vmn2 - vmo2) * raM
            qxip = np.where(isE_m, qclx[ip, M], qcix[ip, M])
            ql[ip, M] = ql[ip, M] + qxip / (fssl_ip + TEENY)
            th[ip, M] = th[ip, M] - ((svlhxl[ip, M] - svlhxl[i, M]) * BYSHA) * wmxm[ip, M] / airm_ip / (plk_ip * fssl_ip + TEENY)
            th[ip, M] = th[ip, M] - (lhx_m * BYSHA) * np.where(isE_m, qclx[ip, M], qcix[ip, M]) / (plk_ip * fssl_ip + TEENY)
            tl[ip, M] = th[ip, M] * plk_ip
            rh[ip, M] = ql[ip, M] / qsat_b(tl[ip, M], lhx_m, pl[ip, M])
            qclx[ip, M] = np.where(isE_m, 0.0, qclx[ip, M])
            qcix[ip, M] = np.where(isE_m, qcix[ip, M], 0.0)
            ca = clear_b(rh[i, M], rh00[i, M], cleara[i, M])
            cleara[i, M] = ca
            S["cldssl"][i, M] = fssl_i * (1.0 - ca)
            S["cldsavl"][i, M] = 1.0 - ca
            tnew, tnewu, qnew, qnewu = tl[i, M], tl[ip, M], ql[i, M], ql[ip, M]
            hcndss[M] = hcndss[M] + fssl_i * (tnew - told) * airm_i + fssl_ip * (tnewu - toldu) * airm_ip
            S["sshr"][i, M] = S["sshr"][i, M] + fssl_i * (tnew - told) * airm_i
            S["sshr"][ip, M] = S["sshr"][ip, M] + fssl_ip * (tnewu - toldu) * airm_ip
            S["dqlsc"][i, M] = S["dqlsc"][i, M] + fssl_i * (qnew - qold)
            S["dqlsc"][ip, M] = S["dqlsc"][ip, M] + fssl_ip * (qnewu - qoldu)
            S["dctei"][i, M] = S["dctei"][i, M] + fssl_i * (qnew - qold) * airm_i * lhx_m * BYSHA
            S["dctei"][ip, M] = S["dctei"][ip, M] + fssl_ip * (qnewu - qoldu) * airm_ip * lhx_m * BYSHA
    W["hcndss"] = hcndss
    W["ckij"] = ckij


# ---------------------------------------------------------------------------------------------- tail (D107)
def lscond_tail_b(S, W, P):
    N = S["tl"].shape[1]
    lmcld = int(P["lmcld"])
    dcl = np.asarray(P["dcl"], int)
    pearth, ckij = P["pearth"], W["ckij"]
    bybr, rimax, rwmax, rwcldox, rcldlx, rcldix = P["bybr"], P["rimax"], P["rwmax"], P["rwcldox"], P["rcldlx"], P["rcldix"]
    sndo = 59.68 / (rwcldox ** 3)
    scdncw = sndo * (1.0 - pearth) + SNDL * pearth
    scdnci = SNDI
    lhp = S["lhp"]
    zl = lambda: np.zeros((LM, N))  # noqa: E731
    cldssl, qclx, qcix, svlhxl, tl, pl, airm, fssl = S["cldssl"], S["qclx"], S["qcix"], S["svlhxl"], S["tl"], S["pl"], S["airm"], S["fssl"]
    wmpr, taumcl, taussl = S["wmpr"], S["taumcl"], S["taussl"]
    csizel = S["csizel"]
    tausslip = S["tausslip"].copy()
    qlss, qiss = S["qlss"], S["qiss"]
    cs_val, cs_set = zl(), np.zeros((LM, N), bool)
    cleara, qheatl, ec, er, prep = W["cleara"], W["qheatl"], W["ec"], W["er"], W["prep"]
    wmsum = np.zeros(N)
    with np.errstate(all="ignore"):
        for L in range(1, lmcld + 1):
            i = L - 1
            fcld = cldssl[i] + TEENY
            lhx = svlhxl[i]
            isE = lhx == LHE
            qx = np.where(isE, qclx[i], qcix[i])
            wtem = 1.0e5 * qx * pl[i] / (fcld * tl[i] * RGAS + TEENY)
            wtem = np.where(wtem < 1.0e-10, 1.0e-10, wtem)
            wmpr[i] = pymax(wmpr[i], 0.0)
            pe = pwm(wtem / (2.0 * BY3 * TWOPI * scdncw), BY3, isE)
            rcld_e = rcldlx * 100.0 * pe
            qheatc = (qheatl[i] + fssl[i] * cleara[i] * (ec[i] + er[i])) / lhx
            rcld_e = np.where(isE & (rcld_e > rwmax) & (prep[i] > qheatc), rwmax, rcld_e)
            rclde_e = rcld_e / bybr
            mI = isE & (lhp[i] == LHS) & (wmpr[i] > 0.0)
            r1 = 1.0e5 * wmpr[i] * pl[i] / (fcld * tl[i] * RGAS + TEENY)
            r1 = rcldix * 100.0 * pwm(r1 / (2.0 * BY3 * TWOPI * scdnci), BY3, mI)
            r1 = pymin(r1, rimax) / bybr
            pi_ = pwm(wtem / (2.0 * BY3 * TWOPI * scdnci), BY3, ~isE)
            rcld_i = pymin(rcldix * 100.0 * pi_, rimax)
            rclde_i = rcld_i / bybr
            rclde = np.where(isE, rclde_e, rclde_i)
            rclde1 = np.where(isE, np.where(mI, r1, rclde_e), rclde_i)
            sset = mI | (~isE & (cldssl[i] > 0))
            cs_val[i] = np.where(mI, r1, rclde_i)
            cs_set[i] = sset
            rclde1 = 5.0 * rclde1
            cz = np.where((fcld <= TEENY) & (rclde > 25.0), 25.0, rclde)
            csizel[i] = cz
            tem = airm[i] * qx * 1.0e2 * BYGRAV
            ts = 1.5e3 * tem / (fcld * rclde + TEENY)
            tem1 = airm[i] * wmpr[i] * 1.0e2 * BYGRAV
            tem1 = 1.5e3 * tem1 / (fcld * rclde1 + TEENY)
            eq = lhp[i] == lhx
            if (~eq & (lhp[i] == LHE)).any():
                raise RuntimeError("VMP: should not be here")
            ts = np.where(eq, ts + tem1, ts)
            tip = np.where(eq, 0.0, np.where(fcld <= TEENY, 0.0, np.where(tem1 > 100.0, 100.0, tem1)))
            tausslip[i] = tip
            ts = np.where(fcld <= TEENY, 0.0, ts)
            ts = np.where(ts > 100.0, 100.0, ts)
            taussl[i] = ts
            wmsum = np.where(isE, wmsum + tem, wmsum)
        cldsal = cldssl.copy()
        cldsv1 = S["cldsv1"].copy()
        by3_2 = 2.0 * BY3
        for L in range(1, lmcld + 1):
            i = L - 1
            cldsv1[i] = cldssl[i]
            lhx = svlhxl[i]
            isE = lhx == LHE
            svlhxl[i] = np.where(isE, np.where(qclx[i] <= 0.0, 0.0, lhx), np.where(qcix[i] <= 0.0, 0.0, lhx))
            r = (taumcl[i] == 0.0) | (ckij != 1.0)
            bmax = 1.0 - exm(-(cldsv1[i] / 0.3), r)
            bmax = np.where(cldsv1[i] >= 0.95, cldsv1[i], bmax)
            bl = r & ((L == 1) | (L <= dcl))
            new_c = pymin(cldssl[i] + (bmax - cldssl[i]) * ckij, fssl[i])
            cold = cldssl[i].copy()
            cldssl[i] = np.where(bl, new_c, cldssl[i])
            taussl[i] = np.where(bl, taussl[i] * cldsv1[i] / (cldssl[i] + TEENY), taussl[i])
            tausslip[i] = np.where(bl, tausslip[i] * cldsv1[i] / (cldssl[i] + TEENY), tausslip[i])
            cldsal[i] = np.where(bl, cldssl[i], cldsal[i])
            cldssl[i] = np.where(r & (taussl[i] <= 0.0), 0.0, cldssl[i])
            fr = r & (L > dcl) & (taumcl[i] <= 0.0)
            cldssl[i] = np.where(fr, pymin(pwm(cldssl[i], by3_2, fr), fssl[i]), cldssl[i])
            p3 = pwm(cldsv1[i], BY3, fr)
            taussl[i] = np.where(fr, taussl[i] * p3, taussl[i])
            tausslip[i] = np.where(fr, tausslip[i] * p3, tausslip[i])
            cldsal[i] = np.where(fr, pwm(cldsal[i], 2.0 / 3.0, fr), cldsal[i])
            neg = taussl[i] < 0.0
            taussl[i] = np.where(neg, 0.0, taussl[i])
            cldssl[i] = np.where(neg, 0.0, cldssl[i])
            qclx[i] = np.where(neg & isE, 0.0, qclx[i])
            qcix[i] = np.where(neg & ~isE, 0.0, qcix[i])
            ok = ~neg
            qlss[i] = np.where(ok & isE, qlss[i] + qclx[i], qlss[i])
            qiss[i] = np.where(ok & ~isE, qiss[i] + qcix[i], qiss[i])
            lE = lhp[i] == LHE
            qlss[i] = np.where(ok & lE, qlss[i] + wmpr[i], qlss[i])
            qiss[i] = np.where(ok & ~lE, qiss[i] + wmpr[i], qiss[i])
            tausslip[i] = np.where(tausslip[i] < 0.0, 0.0, tausslip[i])
    S["tausslip"], S["cldsal"], S["cldsv1"] = tausslip, cldsal, cldsv1
    S["csizelip"] = _ffill(cs_val, cs_set, S["csizelip_init"]) if "csizelip_init" in S else _ffill(cs_val, cs_set, np.zeros(LM))
    # layers above LMCLD keep the carried values (tausslip/cldsal/cldsv1 are not rewritten there)
    W["wmsum"] = wmsum


def lscond_batch(S, P):
    """Whole LSCOND for N columns.  S: dict of arrays (layout in the module docstring, modified in place), P: dict of per-column arrays
    (pearth, wconst, scdncw, dcl, ra (4,N)) and scalars.  -> (S, W)."""
    W = lscond_main_b(S, P)
    lscond_ctei_b(S, W, P)
    lscond_tail_b(S, W, P)
    return S, W


# ---------------------------------------------------------------------------------------------- packing from per-column dicts
_KEYS_LM = ("qcll qcil svlatl svlat1 svwmxl sdl vsubl fssl ttoldl aq dpdt pl plk airm byam u00l taumcl tl ql th rh qclx qcix svlhxl "
            "cldsavl csizel sm qm").split()
_KEYS_LM1 = ("precnvl",)
_KEYS_MOM = ("qmom", "smom")
_KEYS_UV = ("um", "vm")


def pack(S_list, P_list, carry=None):
    """Per-column dicts (the S/P of clouds_condse_ff / the recorded inputs, lists) -> batch arrays.  carry: dict of (LM,) arrays or
    lists (tausslip csizelip cldsal cldsv1) left by the column before the first one (default zeros)."""
    N = len(S_list)
    S = {}
    for k in _KEYS_LM:
        S[k] = np.array([c[k] for c in S_list], float).T.copy()
    for k in _KEYS_LM1:
        S[k] = np.array([c[k] for c in S_list], float).T.copy()
    for k in _KEYS_MOM:
        S[k] = np.array([c[k] for c in S_list], float).transpose(1, 2, 0).copy()
    for k in _KEYS_UV:
        a = np.array([[row[:4] + [0.0] * (4 - len(row[:4])) for row in c[k]] for c in S_list], float)
        S[k] = a.transpose(1, 2, 0).copy()
    S["pdsigl00"] = list(S_list[0]["pdsigl00"])
    S["dqlsc"] = np.zeros((LM, N))
    carry = carry or {}
    for k in ("tausslip", "csizelip", "cldsal", "cldsv1"):
        S[k + "_init"] = np.asarray(carry.get(k, np.zeros(LM)), float)
    S["tausslip"] = np.tile(S["tausslip_init"][:, None], (1, N))
    S["cldsv1"] = np.tile(S["cldsv1_init"][:, None], (1, N))
    p0 = P_list[0]
    P = {k: p0[k] for k in ("wmui", "cmx", "u00a", "scdnci", "rimax", "rwmax", "rwcldox", "rcldlx", "rcldix", "bybr", "bydtsrc", "dtsrc", "lmcld")}
    for k in ("pearth", "wconst", "scdncw"):
        P[k] = np.array([p[k] for p in P_list], float)
    P["dcl"] = np.array([p["dcl"] for p in P_list], int)
    P["ra"] = np.array([list(p["ra"][:4]) + [0.0] * (4 - len(p["ra"][:4])) for p in P_list], float).T.copy()
    for p in P_list:                                  # the scalars must be the same for every column of the batch
        for k in ("wmui", "cmx", "u00a", "scdnci", "rimax", "rcldlx", "rcldix", "bybr", "bydtsrc", "dtsrc", "lmcld"):
            assert p[k] == P[k], k
    return S, P


OUT_LM = ("tl ql th rh qclx qcix svlhxl cldsavl cldssl taussl tausslip csizel csizelip cldsal cldsv1 qlss qiss sshr dctei sm qm "
          "rh1 wmpr").split()


def unpack(S, W, n):
    """Column n of the batch -> (S_col dict of lists in the per-column layout, W_col dict)."""
    o = {k: S[k][:, n].tolist() for k in OUT_LM}
    o["prebar1"] = S["prebar1"][:, n].tolist()
    o["lhp"] = S["lhp"][:, n].tolist()
    o["qmom"] = S["qmom"][:, :, n].tolist()
    o["smom"] = S["smom"][:, :, n].tolist()
    o["um"] = S["um"][:, :, n].tolist()
    o["vm"] = S["vm"][:, :, n].tolist()
    w = {k: (W[k][:, n].tolist() if np.ndim(W[k]) == 2 else W[k][n]) for k in W}
    return o, w
