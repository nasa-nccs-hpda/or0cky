"""D145: batched LSCOND in JAX (clouds_lscond_batch.py converted; same statements, same operation order).

All N columns at once.  Structure:
  * main layer loop `do L=LMCLD,1,-1` (downward PREBAR/PREICE/LHP carry): lax.scan over the layers (descending), every `if` a jnp.where;
  * CTEI loop `do L=LMCLD-1,1,-1`: lax.scan over the layer pairs (the modified upper layer is the scan carry); the bounded iteration
    `do ITER=1,9` with exit is a lax.fori_loop(0, 9) with a per-column active mask (a column that exits keeps the values of its last
    executed iteration), run for ALL columns and selected with the alive/mix masks (the numpy batch compacts the mixing columns);
  * tail (particle size / optical thickness): per-layer independent, done as whole-array (LMCLD, N) operations; the running WMSUM sum is
    a python loop of vector adds (order kept).
Exactness (what is and is not claimed -- see Reports ledger D145):
  * + - * / sqrt, max/min and where are IEEE exact.  Needs --xla_cpu_max_isa=AVX (dyn_jax_env, imported first) because XLA:CPU otherwise
    fuses a*b+c into FMAs.  Every named physical constant (and the REAL(4) literals, already rounded by the numpy code's f4) is passed
    as a TRACED argument (class K) so XLA cannot re-associate/fold operations on closure constants; plain literals (1.0, 0.5, 100.0 ...) stay
    in the source because they are exact doubles.
  * exp and pow are the problem: XLA's exp/pow are not glibc libm.  Two modes (static argument `mode`):
      mode="libm" : exp/pow/get_dq_* go through jax.pure_callback to the SAME scalar functions the numpy batch uses (clouds_lscond_size_ff.ex/pw,
                    clouds_dq_ff), only on the masked columns => bit-identical by construction; this is NOT accelerator-resident (host callback).
      mode="xla"  : jnp.exp / jnp.power; fully on-device, differs from libm by ulps in exp/pow and therefore (threshold sensitivity) can
                    flip a branch in a few columns.
  The Fortran stop in the tail ("VMP: should not be here") is returned as the flag W['vmp_err'] instead of raising.
Usage: S2, W = lscond_jax(S, P, mode="libm"|"xla")  with S, P exactly as clouds_lscond_batch.pack / condse_batch build them (numpy dicts);
returns numpy dicts with the same keys as lscond_batch (S modified copy, W).
"""
import clouds_jax_env  # noqa: F401  (must come before jax: --xla_cpu_max_isa=AVX, --xla_disable_hlo_passes=algsimp)
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import clouds_dq_ff as dq
import clouds_lscond_batch as lb
import clouds_lscond_ff as L0
import clouds_lscond_size_ff as sz
import clouds_massflux_ff as mf

jax.config.update("jax_enable_x64", True)

LM = 40
XYMOMS, ZMOMS = L0.XYMOMS, L0.ZMOMS

_KNAMES = ("TEENY LHE LHS LHM TF SHA BYSHA BYGRAV BY3 TWOPI RGAS GRAV RVAP BYMRAT DELTX SLHE COESIG COEEC TMAX_ICE TMIN_WATER CM00LIQ CM00ICE "
           "GBYAIRM0 _F02 _F23816 _F20783 _F999999 _A _B _C").split()


def make_K():
    """Traced-constant dict (see module docstring)."""
    k = {n.lstrip("_"): np.float64(getattr(L0, n)) for n in _KNAMES}
    for n in "ABCDEFG":
        k["TB_" + n] = np.float64(getattr(mf, "_" + n))
    k["SNDL"] = np.float64(174.0)
    k["BY3_2"] = np.float64(2.0 * L0.BY3)
    k["TWO3"] = np.float64(2.0 / 3.0)
    k["C35"] = np.float64(35.0)
    k["C03"] = np.float64(0.3)
    k["DQ_BYSHA"] = np.float64(dq.BYSHA)
    return k


class _Ns:
    def __init__(self, d):
        self.__dict__.update(d)


# ---------------------------------------------------------------------------------------------- exp / pow / dq backends
def _host_exp(x, m):
    x = np.asarray(x)
    r = np.zeros(x.size)
    xf, mf_ = x.ravel(), np.asarray(m).ravel()
    idx = np.flatnonzero(mf_)
    if idx.size:
        ex = sz.ex
        out = []
        for v in xf[idx].tolist():
            try:
                out.append(ex(v))
            except OverflowError:
                out.append(float("inf"))
        r[idx] = out
    return r.reshape(x.shape)


def _host_pow(x, y, m):
    x = np.asarray(x)
    r = np.zeros(x.size)
    xf, yf, mf_ = x.ravel(), np.broadcast_to(np.asarray(y), x.shape).ravel(), np.asarray(m).ravel()
    idx = np.flatnonzero(mf_)
    if idx.size:
        pw = sz.pw
        r[idx] = [pw(a, b) for a, b in zip(xf[idx].tolist(), yf[idx].tolist())]
    return r.reshape(x.shape)


def _host_dq(kind, sm, qm, plk, mass, lhx, pl, cond, m):
    d, f = lb.dq_b(kind, sm, qm, plk, mass, lhx, pl, cond, m)
    return d, f


class OpsLibm:
    name = "libm"

    @staticmethod
    def exp(x, m):
        return jax.pure_callback(_host_exp, jax.ShapeDtypeStruct(x.shape, jnp.float64), x, m)

    @staticmethod
    def pw(x, y, m):
        y = jnp.broadcast_to(jnp.asarray(y, jnp.float64), x.shape)
        return jax.pure_callback(_host_pow, jax.ShapeDtypeStruct(x.shape, jnp.float64), x, y, m)

    @staticmethod
    def dq(kind, sm, qm, plk, mass, lhx, pl, cond, m, k):
        shp = m.shape
        bc = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float64), shp)  # noqa: E731
        sh = jax.ShapeDtypeStruct(shp, jnp.float64)
        return jax.pure_callback(functools.partial(_host_dq, kind), (sh, sh), bc(sm), bc(qm), bc(plk), bc(mass), bc(lhx), bc(pl), bc(cond), m)


class OpsXla:
    name = "xla"

    @staticmethod
    def exp(x, m):
        return jnp.where(m, jnp.exp(x), 0.0)

    @staticmethod
    def pw(x, y, m):
        return jnp.where(m, jnp.power(x, y), 0.0)

    @staticmethod
    def dq(kind, sm, qm, plk, mass, lhx, pl, cond, m, k):
        bc = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float64), m.shape)  # noqa: E731
        sm, qm, plk, mass, lhx, pl = (bc(a) for a in (sm, qm, plk, mass, lhx, pl))
        sign = 1.0 if kind == "c" else -1.0
        slh = lhx * k.DQ_BYSHA
        qmt = qm
        tp = sm * plk / mass
        dqsum = jnp.zeros(m.shape)
        for _ in range(dq.NITER):
            qst = k.A * jnp.exp(lhx * (k.B - k.C / jnp.maximum(130.0, tp))) / pl
            d = (qmt - mass * qst) / (1.0 + slh * qst * (lhx * k.C / (tp * tp)))
            tp = tp + slh * d / mass
            qmt = qmt - d
            dqsum = dqsum + sign * d
        ref = qm if kind == "c" else bc(cond)
        dqs = jnp.maximum(0.0, jnp.minimum(dqsum, ref))
        fc = dqs / ref
        act = ref > 0
        return jnp.where(m & act, dqs, 0.0), jnp.where(m & act, fc, 0.0)


OPS = {"libm": OpsLibm, "xla": OpsXla}


# ---------------------------------------------------------------------------------------------- helpers
def pymax(a, b):
    return jnp.where(b > a, b, a)


def pymin(a, b):
    return jnp.where(b < a, b, a)


def _qsat(ops, k, tm, lh, pr, m):
    return k.A * ops.exp(lh * (k.B - k.C / pymax(130.0, tm)), m) / pr


def _dqsatdt(k, tm, lh):
    return lh * k.C / (tm * tm)


def _clear(rh, rh00, c, k):
    s = jnp.sqrt((1.0 - rh) / ((1.0 - rh00) + k.TEENY))
    m1 = rh <= 1.0
    c = jnp.where(m1, jnp.where(rh00 < 1.0, s, 1.0), c)
    c = jnp.where(c > 1.0, 1.0, c)
    c = jnp.where(rh > 1.0, 0.0, c)
    return c


def _thbar(k, x, y):
    q = x / y
    al = (k.TB_A + q * (k.TB_B + q * (k.TB_C + q * (k.TB_D + q)))) / (k.TB_E + q * (k.TB_F + k.TB_G * q))
    return x * al


# ---------------------------------------------------------------------------------------------- main layer loop
def _main(ops, k, S, P, W0, lmcld):
    N = S["tl"].shape[1]
    ones = jnp.ones(N, bool)
    dtsrc, bydtsrc = P["dtsrc"], P["bydtsrc"]
    cols = jnp.arange(N)
    pl_dcl = S["pl"][P["dcl"] - 1, cols]
    zero = jnp.zeros(N)
    names = ("tl ql th rh qclx qcix svlhxl svlatl svwmxl qcll qcil fssl airm byam pl plk vsubl u00l dpdt ttoldl aq cldsavl").split()
    xs = {n: S[n][:lmcld][::-1] for n in names}
    xs["sdl_i"] = S["sdl"][:lmcld][::-1]
    xs["sdl_ip"] = S["sdl"][1:lmcld + 1][::-1]
    xs["prec_ip"] = S["precnvl"][1:lmcld + 1][::-1]
    xs["pdsig"] = P["pdsigl00"][:lmcld][::-1]
    xs["qmom"] = S["qmom"][:lmcld][::-1]
    xs["L"] = jnp.arange(lmcld, 0, -1)

    def layer(carry, x):
        prebar_ip, preice_ip, lhp_ip, hcnd, ierr, lerr, wmerr = carry
        L = x["L"]
        tl, ql, th, rh = x["tl"], x["ql"], x["th"], x["rh"]
        qclx, qcix, svlhxl = x["qclx"], x["qcix"], x["svlhxl"]
        fssl, airm, byam, pl, plk = x["fssl"], x["airm"], x["byam"], x["pl"], x["plk"]
        svlatl, svwmxl, qcll, qcil = x["svlatl"], x["svwmxl"], x["qcll"], x["qcil"]
        vsubl = x["vsubl"]
        qmom = x["qmom"]
        cleara = 1.0 - x["cldsavl"]
        cleara = jnp.where(qclx + qcix <= 0.0, 1.0, cleara)
        told, qold = tl, ql
        oldlhx, oldlat = svlhxl, svlatl
        temp = 100.0 * k.RGAS * tl / (pl * k.GRAV)
        vvel = jnp.where(L == 1, -x["sdl_ip"] * temp, -0.5 * (x["sdl_i"] + x["sdl_ip"]) * temp)
        vdef = vvel - vsubl
        fcld = (1.0 - cleara) * fssl + k.TEENY
        r00 = jnp.full(N, P["u00a"])
        r00 = jnp.where(pl < pl_dcl, r00 / (r00 + (1.0 - r00) * x["pdsig"] / k.C35), r00)
        r00 = jnp.where(x["u00l"] > r00, x["u00l"], r00)
        r00 = jnp.where(r00 < 0.0, 0.0, r00)
        r00 = jnp.where(r00 > 1.0, 1.0, r00)
        rh00 = r00
        rhf = r00 + (1.0 - cleara) * (1.0 - r00)
        ice = tl <= k.TMIN_WATER
        lhx = jnp.where(ice, k.LHS, k.LHE)
        lhp = jnp.where(ice, k.LHS, jnp.where(tl < k.TF, k.LHS, k.LHE))
        isE = lhx == k.LHE
        qsatl = _qsat(ops, k, tl, lhx, pl, ones)
        rh1 = ql / qsatl
        m1 = (lhx == k.LHS) & (qclx + qcix <= 0.0)
        qsate = _qsat(ops, k, tl, k.LHE, pl, m1)
        rhw = (2.583 - tl / k.F20783) * (_qsat(ops, k, tl, k.LHS, pl, m1) / qsate)
        rh1 = jnp.where(m1 & (tl < k.F23816), ql / (qsate * rhw), rh1)
        hchang = zero
        c1 = (oldlhx == k.LHE) & (lhx == k.LHS)
        hchang = jnp.where(c1, qcll * k.LHM, hchang)
        c2 = (oldlhx == k.LHS) & (lhx == k.LHE)
        hchang = jnp.where(c2, -qcil * k.LHM, hchang)
        c3 = (oldlat == k.LHE) & (lhx == k.LHS)
        hchang = jnp.where(c3, hchang + svwmxl * k.LHM, hchang)
        c4 = (oldlat == k.LHS) & (lhx == k.LHE)
        hchang = jnp.where(c4, hchang - svwmxl * k.LHM, hchang)
        qcix = jnp.where(isE, 0.0, jnp.where(c1, qcll + svwmxl, qcil + svwmxl))
        qclx = jnp.where(isE, jnp.where(c2, qcil + svwmxl, qcll + svwmxl), 0.0)
        svlhxl = lhx
        tl = tl + hchang / (k.SHA * fssl + k.TEENY)
        th = tl / plk
        rhi = ql / _qsat(ops, k, tl, k.LHS, pl, ones)
        lhp = jnp.where((lhp_ip == k.LHS) & (tl < k.TF + dtsrc * k.LHM * preice_ip * k.GRAV * byam * k.BYSHA), k.LHS, lhp)
        # autoconversion
        qx = jnp.where(isE, qclx, qcix)
        qcx = qclx + qcix
        mq = qcx > 0.0
        rho = 1.0e5 * pl / (k.RGAS * tl)
        wtliq = jnp.where(tl > k.TMAX_ICE, 1.0, jnp.where(tl < k.TMIN_WATER, 0.0, (tl - k.TMIN_WATER) / (k.TMAX_ICE - k.TMIN_WATER)))
        wtliq = jnp.where((oldlat == k.LHS) & (svwmxl > 0.0), 0.0, wtliq)
        tem = P["wconst"] * wtliq + P["wmui"] * (1.0 - wtliq)
        cm0 = k.CM00LIQ * wtliq + k.CM00ICE * (1.0 - wtliq)
        mv = mq & (vdef > 0.0) & (rho * qcx < 10.0)
        cm0 = jnp.where(mv, cm0 * ops.pw(jnp.full(N, 10.0), -k.F02 * vdef, mv), cm0)
        tem = rho * qcx / (tem * fcld + k.TEENY)
        tem = tem * tem
        tem = jnp.where(tem > 10.0, 10.0, tem)
        cm = cm0 * (1.0 - 1.0 / ops.exp(tem * tem, mq)) + 100.0 * (prebar_ip + x["prec_ip"] * bydtsrc)
        cm = cm * P["cmx"]
        cm = jnp.where(cm > bydtsrc, bydtsrc, cm)
        prep = jnp.where(mq, qcx * cm, zero)
        # form clouds?
        low = rh1 < rh00
        sq = jnp.where(low, zero, lhx * qsatl * _dqsatdt(k, tl, lhx) * k.BYSHA)
        tem = -lhx * x["dpdt"] / pl
        ath = jnp.where(low, zero, (th - x["ttoldl"]) * bydtsrc)
        qconv = lhx * x["aq"] - rh * sq * k.SHA * plk * ath - tem * qsatl * rh
        form = ~low & ((qconv > 0.0) | (qx > 0.0))
        ermax = lhx * prebar_ip * k.GRAV * byam
        rhn = pymin(rh, rhf)
        er_f = jnp.where(qx > 0.0, (1.0 - rhn) * (1.0 - rhn) * lhx * prebar_ip * k.GBYAIRM0,
                         jnp.where((preice_ip > 0.0) & (tl < k.TF), (1.0 - rhi) * (1.0 - rhi) * lhx * prebar_ip * k.GBYAIRM0,
                                   (1.0 - rh) * (1.0 - rh) * lhx * prebar_ip * k.GBYAIRM0))
        nf = ~form
        mev = nf & (qx > 0.0)
        dqs, _ = ops.dq("e", tl * rh00 / plk, ql * rh00, plk, rh00, lhx, pl, qx / (fssl * rh00), mev, k)
        dwdt_e = dqs * rh00 * fssl
        qh_nf = jnp.where(mev, -dwdt_e * lhx * bydtsrc, 0.0)
        prep_nf = pymax(0.0, (qx - dwdt_e) * bydtsrc)
        prep = jnp.where(mev, prep_nf, prep)
        wmpr = jnp.where(mev, prep * dtsrc, zero)
        er_n = (1.0 - rh) * (1.0 - rh) * lhx * prebar_ip * k.GBYAIRM0
        er_n = jnp.where((preice_ip > 0.0) & (tl < k.TF), (1.0 - rhi) * (1.0 - rhi) * lhx * prebar_ip * k.GBYAIRM0, er_n)
        e_raw = jnp.where(form, er_f, er_n)
        er = pymax(0.0, pymin(e_raw, ermax))
        fc = form & (cleara > 0.0)
        wtem = 1.0e5 * qx * pl / (fcld * tl * k.RGAS + k.TEENY)
        wtem = jnp.where(isE & (qclx / fcld >= P["wconst"] * 1.0e-3), 1.0e2 * P["wconst"] * pl / (tl * k.RGAS), wtem)
        wtem = jnp.where(wtem < 1.0e-10, 1.0e-10, wtem)
        pe = ops.pw(wtem / (2.0 * k.BY3 * k.TWOPI * P["scdncw"]), k.BY3, fc & isE)
        pi_ = ops.pw(wtem / (2.0 * k.BY3 * k.TWOPI * P["scdnci"]), k.BY3, fc & ~isE)
        rcld = jnp.where(isE, P["rcldlx"] * 1.0e-6 * 100.0 * pe, pymin(P["rcldix"] * 100.0e-6 * pi_, P["rimax"]))
        ck1 = 1000.0 * lhx * lhx / (2.4e-2 * k.RVAP * tl * tl)
        ck2 = 1000.0 * k.RGAS * tl / (2.4e-3 * qsatl * pl)
        tevap = k.COEEC * (ck1 + ck2) * rcld * rcld
        wmx1 = qx - prep * dtsrc
        wmpr = jnp.where(fc, prep * dtsrc, wmpr)
        ecrate = (1.0 - rhf) / (tevap * fcld + k.TEENY)
        ecrate = jnp.where(ecrate > bydtsrc, bydtsrc, ecrate)
        ec = jnp.where(fc, wmx1 * ecrate * lhx, zero)
        drhdt = 2.0 * cleara * cleara * (1.0 - rh00) * (qconv + er) / lhx / (qx / (fcld + k.TEENY) + 2.0 * cleara * qsatl * (1.0 - rh00) + k.TEENY)
        drhdt = jnp.where((er == 0.0) & (qx <= 0.0), 0.0, drhdt)
        qh_f = fssl * (qconv - lhx * drhdt * qsatl) / (1.0 + rh * sq)
        dwdt = qh_f / lhx - prep + cleara * fssl * er / lhx
        qxn_f = qx + dwdt * dtsrc
        neg = qxn_f < 0.0
        qxn_f = jnp.where(neg, 0.0, qxn_f)
        qh_f = jnp.where(neg, (-qx * bydtsrc + prep) * lhx - cleara * fssl * er, qh_f)
        qh_n = qh_nf - cleara * fssl * er
        qh = jnp.where(form, qh_f, qh_n)
        qxnew = jnp.where(form, qxn_f, 0.0)
        qheatl = jnp.where(isE, qh, 0.0)
        qheati = jnp.where(~isE, qh, 0.0)
        # phase of precipitation
        hphase = zero
        ch1 = (lhp_ip == k.LHS) & (lhp == k.LHE) & (preice_ip > 0.0)
        hphase = jnp.where(ch1, hphase + k.LHM * preice_ip * k.GRAV * byam, hphase)
        preice_ip_mod = jnp.where(ch1, 0.0, preice_ip)
        ch2 = (lhp_ip == k.LHE) & (lhp == k.LHS) & (prebar_ip > 0.0)
        hphase = jnp.where(ch2, hphase - k.LHM * prebar_ip * k.GRAV * byam, hphase)
        ch3 = lhp != lhx
        hphase = jnp.where(ch3, hphase + (er * cleara * fssl / lhx - prep) * k.LHM, hphase)
        prebar = jnp.where(er == ermax, prebar_ip * (1.0 - cleara * fssl) + airm * prep * k.BYGRAV,
                           pymax(0.0, prebar_ip + airm * (prep - er * cleara * fssl / lhx) * k.BYGRAV))
        qnew = ql - dtsrc * qh / (lhx * fssl + k.TEENY)
        qn = qnew < 0.0
        qnew = jnp.where(qn, 0.0, qnew)
        qh = jnp.where(qn, ql * lhx * bydtsrc * fssl, qh)
        dwdt1 = qh / lhx - prep + cleara * fssl * er / lhx
        qxn1 = qx + dwdt1 * dtsrc
        qxnew = jnp.where(qn, qxn1, qxnew)
        ie = qn & (qxn1 < 0.0)
        ierr = jnp.where(ie, 1, ierr)
        lerr = jnp.where(ie, L, lerr)
        wmerr = jnp.where(ie, qxn1, wmerr)
        qxnew = jnp.where(ie, 0.0, qxnew)
        qheatl = jnp.where(isE & qn, qh, qheatl)
        qheati = jnp.where(~isE & qn, qh, qheati)
        fqtow = zero
        pf = fssl > 0.0
        qheat = jnp.where(pf, qh, zero)
        c5 = pf & (qh + cleara * fssl * er > 0.0) & (lhx * ql + dtsrc * cleara * er > 0.0)
        fqtow = jnp.where(c5, (qh + cleara * fssl * er) * dtsrc / ((lhx * ql + dtsrc * cleara * er) * fssl), fqtow)
        ql = qnew
        qmom = qmom * (1.0 - fqtow)[None, :]
        qclx = jnp.where(isE, qxnew, qclx)
        qcix = jnp.where(isE, qcix, qxnew)
        tl = tl + dtsrc * (qh - hphase) / (k.SHA * fssl + k.TEENY)
        th = tl / plk
        qsatc = _qsat(ops, k, tl, lhx, pl, ones)
        rh = ql / qsatc
        rh1 = ql / qsatc
        mS = lhx == k.LHS
        ca = _clear(rh, rh00, cleara, k)
        ca = jnp.where(qclx + qcix <= 0.0, 1.0, ca)
        cleara = jnp.where(mS, ca, cleara)
        qf = (ql - qsatc * (1.0 - cleara)) / (cleara + k.TEENY)
        qsate = _qsat(ops, k, tl, k.LHE, pl, mS)
        rhw = (2.583 - tl / k.F20783) * (_qsat(ops, k, tl, k.LHS, pl, mS) / qsate)
        rh1 = jnp.where(mS & (tl < k.F23816) & (qclx + qcix <= 0.0), qf / (qsate * rhw), rh1)
        mC = rh1 > 1.0
        slh = lhx * k.BYSHA
        dqsum, fcond = ops.dq("c", tl, ql, jnp.ones(N), jnp.ones(N), lhx, pl, ql, mC, k)
        mD = mC & (dqsum > 0.0)
        tl = jnp.where(mD, tl + slh * dqsum, tl)
        ql = jnp.where(mD, ql - dqsum, ql)
        qclx = jnp.where(mD & isE, qclx + dqsum * fssl, qclx)
        qcix = jnp.where(mD & ~isE, qcix + dqsum * fssl, qcix)
        qmom = jnp.where(mD[None, :], qmom * (1.0 - fcond)[None, :], qmom)
        rh = jnp.where(mC, ql / _qsat(ops, k, tl, lhx, pl, mC), rh)
        th = jnp.where(mC, tl / plk, th)
        cleara = _clear(rh, rh00, cleara, k)
        qx2 = jnp.where(isE, qclx, qcix)
        cleara = jnp.where(qx2 <= 0.0, 1.0, cleara)
        cleara = jnp.where(cleara < 0.0, 0.0, cleara)
        rhf = rh00 + (1.0 - cleara) * (1.0 - rh00)
        ro = (rh <= rhf) & (rh < k.F999999) & (qx2 > 0.0)
        prebar = jnp.where(ro, prebar + qx2 * airm * k.BYGRAV * bydtsrc, prebar)
        mm = ro & isE & (lhp == k.LHS)
        hch = qclx * k.LHM
        tl = jnp.where(mm, tl + hch / (k.SHA * fssl + k.TEENY), tl)
        th = jnp.where(mm, tl / plk, th)
        qclx = jnp.where(ro & isE, 0.0, qclx)
        qcix = jnp.where(ro & ~isE, 0.0, qcix)
        prebar1 = prebar
        preice = jnp.where((prebar > 0.0) & (lhp == k.LHS), prebar, 0.0)
        lhp = jnp.where(prebar <= 0.0, 0.0, lhp)
        cleara = _clear(rh, rh00, cleara, k)
        qclx = jnp.where(isE & (qclx <= k.TEENY), 0.0, qclx)
        qcix = jnp.where(~isE & (qcix <= k.TEENY), 0.0, qcix)
        qx3 = jnp.where(isE, qclx, qcix)
        cleara = jnp.where(qx3 <= 0.0, 1.0, cleara)
        cleara = jnp.where(cleara < 0.0, 0.0, cleara)
        cldssl = fssl * (1.0 - cleara)
        cldsavl = 1.0 - cleara
        dT = fssl * (tl - told) * airm
        hcnd = hcnd + dT
        sshr = zero + dT
        dqlsc = zero + fssl * (ql - qold)
        y = dict(tl=tl, ql=ql, th=th, rh=rh, qclx=qclx, qcix=qcix, svlhxl=svlhxl, cleara=cleara, rhf=rhf, rh00=rh00, er=er, ec=ec, prep=prep,
                 qheatl=qheatl, qheati=qheati, qheat=qheat, prebar=prebar, preice=preice, preice_ip=preice_ip_mod, lhp=lhp, cldssl=cldssl,
                 cldsavl=cldsavl, wmpr=wmpr, rh1=rh1, sshr=sshr, dqlsc=dqlsc, qmom=qmom, prebar1=prebar1)
        return (prebar, preice, lhp, hcnd, ierr, lerr, wmerr), y

    c0 = (zero, zero, zero, zero, jnp.zeros(N, jnp.int64), jnp.zeros(N, jnp.int64), zero)
    (pb0, pi0, lh0, hcnd, ierr, lerr, wmerr), ys = lax.scan(layer, c0, xs)
    return (pb0, pi0, lh0, hcnd, ierr, lerr, wmerr), ys


def _assemble(ys, lmcld, S, name, last=None):
    """scan outputs (descending layers) -> (LM, ...) array: ascending layers, rest from S[name] (or zeros if S has no such entry)."""
    a = ys[name][::-1]
    rest = S[name][lmcld:] if name in S else jnp.zeros((LM - lmcld,) + a.shape[1:])
    return jnp.concatenate([a, rest], axis=0)


# ---------------------------------------------------------------------------------------------- CTEI
def _ctmix(rm0, rm1, mom0, mom1, fmair, fmix, frat, k):
    rtemp = rm0 * (1.0 - fmix) + frat * rm1
    n1 = rm1 * (1.0 - frat) + fmix * rm0
    for m in XYMOMS:
        rt = mom0[m] * (1.0 - fmix) + frat * mom1[m]
        mom1 = mom1.at[m].set(mom1[m] * (1.0 - frat) + fmix * mom0[m])
        mom0 = mom0.at[m].set(rt)
    for m in ZMOMS:
        mom0 = mom0.at[m].set(mom0[m] * (1.0 - fmair))
        mom1 = mom1.at[m].set(mom1[m] * (1.0 - fmair))
    return rtemp, n1, mom0, mom1


def _ctei(ops, k, S, P, W, lmcld, st):
    """st: dict of post-main layer arrays (LM, ...).  Returns updated dict and W entries."""
    N = S["tl"].shape[1]
    dtsrc = P["dtsrc"]
    ra = P["ra"]
    ones = jnp.ones(N, bool)
    zero = jnp.zeros(N)
    keys_i = "tl ql th rh qclx qcix cleara cldssl cldsavl sshr dqlsc dctei".split()
    # per-layer arrays for the pair loop; L = lmcld-1 .. 1, i = L-1, ip = L
    Ls = jnp.arange(lmcld - 1, 0, -1)

    def sl(a):          # layer i for L=lmcld-1..1 -> layers lmcld-2..0 descending
        return a[:lmcld - 1][::-1]

    xs = {n: sl(st[n]) for n in "tl ql th rh qclx qcix cleara cldssl cldsavl sshr dqlsc dctei svlhxl airm byam pl plk fssl rh00 smom qmom um vm".split()}
    xs["L"] = Ls
    # carry = the (already modified) ip layer; initial: layer lmcld-1
    ip0 = {n: st[n][lmcld - 1] for n in "tl ql th rh qclx qcix cleara cldssl cldsavl sshr dqlsc dctei svlhxl airm byam pl plk fssl smom qmom um vm sm qm".split()}
    carry0 = (ip0, W["hcndss"], jnp.ones(N))

    def step(carry, x):
        ip, hcnd, ckij = carry
        L = x["L"]
        lhx = x["svlhxl"]
        isE = lhx == k.LHE
        i_tl, i_ql, i_th, i_rh, i_qclx, i_qcix = x["tl"], x["ql"], x["th"], x["rh"], x["qclx"], x["qcix"]
        airm_i, byam_i, pl_i, plk_i, fssl_i, rh00_i = x["airm"], x["byam"], x["pl"], x["plk"], x["fssl"], x["rh00"]
        airm_ip, byam_ip, plk_ip, fssl_ip = ip["airm"], ip["byam"], ip["plk"], ip["fssl"]
        cleara_i, cleara_ip = x["cleara"], ip["cleara"]
        sm_i = i_th * airm_i
        qm_i = i_ql * airm_i
        wm_i = jnp.where(isE, i_qclx, i_qcix) * airm_i
        sm_ip = ip["th"] * airm_ip
        qm_ip = ip["ql"] * airm_ip
        wm_ip = jnp.where(isE, ip["qclx"], ip["qcix"]) * airm_i
        alive = ~(ip["qclx"] + ip["qcix"] > k.TEENY)
        alive &= ~((cleara_i == 1.0) | ((cleara_i < 1.0) & (cleara_ip < 1.0)))
        fcld = (1.0 - cleara_i) * fssl_i + k.TEENY
        sedge = _thbar(k, ip["th"], i_th)
        dse = (ip["th"] - sedge) * plk_ip + (sedge - i_th) * plk_i + k.SLHE * (ip["ql"] - i_ql)
        dwm = jnp.where(isE, ip["ql"] - i_ql + (ip["qclx"] - i_qclx) / fcld, ip["ql"] - i_ql + (ip["qcix"] - i_qcix) / fcld)
        dqsdt = _dqsatdt(k, i_tl, k.LHE) * i_ql / (i_rh + 1.0e-30)
        beta = (1.0 + k.BYMRAT * i_tl * dqsdt) / (1.0 + k.SLHE * dqsdt)
        ckm = (1.0 + k.SLHE * dqsdt) * (1.0 + (1.0 - k.DELTX) * i_tl / k.SLHE) / (2.0 + (1.0 + k.BYMRAT * i_tl / k.SLHE) * k.SLHE * dqsdt)
        ckr = i_tl / (beta * k.SLHE)
        ck = dse / (k.SLHE * dwm + k.TEENY)
        ok3 = alive & ~(ckr > ckm)
        mk5 = ok3 & (ck > ckr)
        sigk = jnp.where(mk5, k.COESIG * ops.pw((ck - ckr) / ((ckm - ckr) + k.TEENY), 5.0, mk5), 0.0)
        expst = ops.exp(-sigk * dtsrc, ok3)
        ckij = jnp.where((L <= 1) & ok3, expst, ckij)
        dsec = dwm * i_tl / beta
        fpmax = pymin(1.0, 1.0 - expst)
        mix = ok3 & ~(ck < ckr) & ~(fpmax <= 0.0) & ~(dse >= dsec)
        # ---- mixing (all columns, selected by mix)
        airmr = (airm_ip + airm_i) * byam_ip * byam_i
        smo1, qmo1, wmo1 = sm_i, qm_i, wm_i
        smo2, qmo2, wmo2 = sm_ip, qm_ip, wm_ip
        smo12 = smo1 * plk_i + smo2 * plk_ip
        umo1, vmo1, umo2, vmo2 = x["um"], x["vm"], ip["um"], ip["vm"]
        fplume0 = fpmax * fssl_i
        pl_ip = ip["pl"]

        def it_body(_, c):
            fplume, dfx, active, R = c
            dfx_b = dfx * 0.5
            fmix = fplume * fcld
            fmass = fmix * airm_i
            fmass = pymin(fmass, (airm_ip * airm_i) / (airm_ip + airm_i))
            fmix = fmass * byam_i
            frat = fmass * byam_ip
            smn1 = smo1 * (1.0 - fmix) + frat * smo2
            qmn1 = qmo1 * (1.0 - fmix) + frat * qmo2
            wmn1 = wmo1 * (1.0 - fmix) + frat * wmo2
            smn2 = smo2 * (1.0 - frat) + fmix * smo1
            qmn2 = qmo2 * (1.0 - frat) + fmix * qmo1
            wmn2 = wmo2 * (1.0 - frat) + fmix * wmo1
            tht1 = smn1 * byam_i / plk_i
            qlt1 = qmn1 * byam_i
            tlt1 = tht1 * plk_i
            rht1 = qlt1 / _qsat(ops, k, tlt1, lhx, pl_i, ones)
            wmt1 = wmn1 * byam_i
            tht2 = smn2 * byam_ip / plk_ip
            qlt2 = qmn2 * byam_ip
            wmt2 = wmn2 * byam_ip
            sedge_ = _thbar(k, tht2, tht1)
            dse_ = (tht2 - sedge_) * plk_ip + (sedge_ - tht1) * plk_i + k.SLHE * (qlt2 - qlt1)
            dwm_ = qlt2 - qlt1 + (wmt2 - wmt1) / fcld
            dqsdt_ = _dqsatdt(k, tlt1, k.LHE) * qlt1 / (rht1 + 1.0e-30)
            beta_ = (1.0 + k.BYMRAT * tlt1 * dqsdt_) / (1.0 + k.SLHE * dqsdt_)
            dsec_ = dwm_ * tlt1 / beta_
            dsedif = dse_ - dsec_
            fp = jnp.where(dsedif > 1.0e-3, fplume - dfx_b, fplume)
            fp = jnp.where(dsedif < -1.0e-3, fp + dfx_b, fp)
            new = dict(smn1=smn1, qmn1=qmn1, wmn1=wmn1, smn2=smn2, qmn2=qmn2, wmn2=wmn2, fmix=fmix, frat=frat, fmass=fmass)
            R = {n: jnp.where(active, new[n], R[n]) for n in R}
            fplume_n = jnp.where(active, fp, fplume)
            dfx_n = jnp.where(active, dfx_b, dfx)
            ex_ = (jnp.abs(dsedif) <= 1.0e-3) | (fp > fpmax * fssl_i)
            return fplume_n, dfx_n, active & ~ex_, R

        R0 = {n: zero for n in "smn1 qmn1 wmn1 smn2 qmn2 wmn2 fmix frat fmass".split()}
        _, _, _, R = lax.fori_loop(0, 9, it_body, (fplume0, fplume0, mix, R0))
        smn1, qmn1, wmn1, smn2, qmn2, wmn2 = R["smn1"], R["qmn1"], R["wmn1"], R["smn2"], R["qmn2"], R["wmn2"]
        fmix, frat, fmass = R["fmix"], R["frat"], R["fmass"]
        smn12 = smn1 * plk_i + smn2 * plk_ip
        smn1 = smn1 - (smn12 - smo12) * airm_i / ((airm_i + airm_ip) * plk_i)
        smn2 = smn2 - (smn12 - smo12) * airm_ip / ((airm_i + airm_ip) * plk_ip)
        n_th_i = smn1 * byam_i
        n_tl_i = n_th_i * plk_i
        n_ql_i = qmn1 * byam_i
        n_rh_i = n_ql_i / _qsat(ops, k, n_tl_i, lhx, pl_i, mix)
        n_qclx_i = jnp.where(isE, wmn1 * byam_i, i_qclx)
        n_qcix_i = jnp.where(isE, i_qcix, wmn1 * byam_i)
        fsslrat = fssl_i / fssl_ip
        mk = fsslrat != 1.0
        smn2 = jnp.where(mk, sm_ip + (smn2 - sm_ip) * fsslrat, smn2)
        qmn2 = jnp.where(mk, qm_ip + (qmn2 - qm_ip) * fsslrat, qmn2)
        smom2_sv, qmom2_sv = ip["smom"], ip["qmom"]
        r0s, r1s, sm0n, sm1n = _ctmix(sm_i, sm_ip, x["smom"], ip["smom"], fmass * airmr, fmix, frat, k)
        r0q, r1q, qm0n, qm1n = _ctmix(qm_i, qm_ip, x["qmom"], ip["qmom"], fmass * airmr, fmix, frat, k)
        sm_i_f = r0s
        sm_ip_f = jnp.where(mk, smn2, r1s)
        qm_i_f = r0q
        qm_ip_f = jnp.where(mk, qmn2, r1q)
        sm1n = jnp.where(mk[None, :], smom2_sv + (sm1n - smom2_sv) * fsslrat[None, :], sm1n)
        qm1n = jnp.where(mk[None, :], qmom2_sv + (qm1n - qmom2_sv) * fsslrat[None, :], qm1n)
        th_ip = smn2 * byam_ip
        ql_ip = qmn2 * byam_ip
        qclx_ip = jnp.where(isE, wmn2 * byam_ip, ip["qclx"])
        qcix_ip = jnp.where(isE, ip["qcix"], wmn2 * byam_ip)
        raM = ra
        umn1 = umo1 * (1.0 - fmix)[None, :] + frat[None, :] * umo2
        vmn1 = vmo1 * (1.0 - fmix)[None, :] + frat[None, :] * vmo2
        umn2 = umo2 * (1.0 - frat)[None, :] + fmix[None, :] * umo1
        vmn2 = vmo2 * (1.0 - frat)[None, :] + fmix[None, :] * vmo1
        um_i = x["um"] + (umn1 - umo1) * raM
        vm_i = x["vm"] + (vmn1 - vmo1) * raM
        um_ip = ip["um"] + (umn2 - umo2) * raM
        vm_ip = ip["vm"] + (vmn2 - vmo2) * raM
        qxip = jnp.where(isE, qclx_ip, qcix_ip)
        ql_ip = ql_ip + qxip / (fssl_ip + k.TEENY)
        th_ip = th_ip - ((ip["svlhxl"] - lhx) * k.BYSHA) * wm_ip / airm_ip / (plk_ip * fssl_ip + k.TEENY)
        th_ip = th_ip - (lhx * k.BYSHA) * jnp.where(isE, qclx_ip, qcix_ip) / (plk_ip * fssl_ip + k.TEENY)
        tl_ip = th_ip * plk_ip
        rh_ip = ql_ip / _qsat(ops, k, tl_ip, lhx, pl_ip, mix)
        qclx_ip = jnp.where(isE, 0.0, qclx_ip)
        qcix_ip = jnp.where(isE, qcix_ip, 0.0)
        ca = _clear(n_rh_i, rh00_i, cleara_i, k)
        cldssl_i = fssl_i * (1.0 - ca)
        cldsavl_i = 1.0 - ca
        tnew, tnewu, qnew, qnewu = n_tl_i, tl_ip, n_ql_i, ql_ip
        told, toldu, qold, qoldu = i_tl, ip["tl"], i_ql, ip["ql"]
        hcnd_n = hcnd + fssl_i * (tnew - told) * airm_i + fssl_ip * (tnewu - toldu) * airm_ip
        sshr_i = x["sshr"] + fssl_i * (tnew - told) * airm_i
        sshr_ip = ip["sshr"] + fssl_ip * (tnewu - toldu) * airm_ip
        dql_i = x["dqlsc"] + fssl_i * (qnew - qold)
        dql_ip = ip["dqlsc"] + fssl_ip * (qnewu - qoldu)
        dct_i = x["dctei"] + fssl_i * (qnew - qold) * airm_i * lhx * k.BYSHA
        dct_ip = ip["dctei"] + fssl_ip * (qnewu - qoldu) * airm_ip * lhx * k.BYSHA
        w = lambda new, old: jnp.where(mix, new, old)  # noqa: E731
        wm = lambda new, old: jnp.where(mix[None, :], new, old)  # noqa: E731
        ip_new = dict(ip)
        ip_new.update(tl=w(tl_ip, ip["tl"]), ql=w(ql_ip, ip["ql"]), th=w(th_ip, ip["th"]), rh=w(rh_ip, ip["rh"]), qclx=w(qclx_ip, ip["qclx"]),
                      qcix=w(qcix_ip, ip["qcix"]), sshr=w(sshr_ip, ip["sshr"]), dqlsc=w(dql_ip, ip["dqlsc"]), dctei=w(dct_ip, ip["dctei"]),
                      um=wm(um_ip, ip["um"]), vm=wm(vm_ip, ip["vm"]), smom=wm(sm1n, ip["smom"]), qmom=wm(qm1n, ip["qmom"]),
                      sm=sm_ip_f, qm=qm_ip_f)
        # sm/qm of layer ip are always (re)set: unmixed columns get the freshly computed th*airm / ql*airm values
        ip_new["sm"] = jnp.where(mix, sm_ip_f, sm_ip)
        ip_new["qm"] = jnp.where(mix, qm_ip_f, qm_ip)
        # the pair's lower layer i becomes the next ip: it is the new carry; the upper layer ip is final -> output
        i_new = dict(tl=w(n_tl_i, i_tl), ql=w(n_ql_i, i_ql), th=w(n_th_i, i_th), rh=w(n_rh_i, i_rh), qclx=w(n_qclx_i, i_qclx), qcix=w(n_qcix_i, i_qcix),
                     cleara=w(ca, cleara_i), cldssl=w(cldssl_i, x["cldssl"]), cldsavl=w(cldsavl_i, x["cldsavl"]), sshr=w(sshr_i, x["sshr"]),
                     dqlsc=w(dql_i, x["dqlsc"]), dctei=w(dct_i, x["dctei"]), svlhxl=lhx, airm=airm_i, byam=byam_i, pl=pl_i, plk=plk_i, fssl=fssl_i,
                     smom=wm(sm0n, x["smom"]), qmom=wm(qm0n, x["qmom"]), um=wm(um_i, x["um"]), vm=wm(vm_i, x["vm"]),
                     sm=jnp.where(mix, sm_i_f, sm_i), qm=jnp.where(mix, qm_i_f, qm_i))
        hcnd2 = jnp.where(mix, hcnd_n, hcnd)
        return (i_new, hcnd2, ckij), ip_new

    (i_last, hcnd, ckij), ys = lax.scan(step, carry0, xs)
    # ys[j] = final ip layer for L = lmcld-1-j  => layer index lmcld-1-j ; i_last = layer 0
    out = {}
    for n in "tl ql th rh qclx qcix cleara cldssl cldsavl sshr dqlsc dctei sm qm smom qmom um vm".split():
        top = ys[n][::-1]                       # layers 1 .. lmcld-1
        low = i_last[n][None]                   # layer 0
        out[n] = jnp.concatenate([low, top], axis=0)
    return out, hcnd, ckij


# ---------------------------------------------------------------------------------------------- tail
def _tail(ops, k, S, P, W, lmcld, st, ckij, lhp):
    N = S["tl"].shape[1]
    Lr = jnp.arange(1, lmcld + 1)[:, None]
    dcl = P["dcl"][None, :]
    pearth = P["pearth"]
    bybr, rimax, rwmax, rcldlx, rcldix = P["bybr"], P["rimax"], P["rwmax"], P["rcldlx"], P["rcldix"]
    scdncw = P["sndo"] * (1.0 - pearth) + k.SNDL * pearth
    scdnci = P["scdnci_t"]
    g = lambda n: st[n][:lmcld]  # noqa: E731
    cldssl, qclx, qcix, svlhxl, tl, pl, airm, fssl = g("cldssl"), g("qclx"), g("qcix"), g("svlhxl"), g("tl"), g("pl"), g("airm"), g("fssl")
    wmpr, taumcl = g("wmpr"), S["taumcl"][:lmcld]
    cleara, qheatl, ec, er, prep = g("cleara"), g("qheatl"), g("ec"), g("er"), g("prep")
    lhp_l = lhp[:lmcld]
    fcld = cldssl + k.TEENY
    lhx = svlhxl
    isE = lhx == k.LHE
    qx = jnp.where(isE, qclx, qcix)
    wtem = 1.0e5 * qx * pl / (fcld * tl * k.RGAS + k.TEENY)
    wtem = jnp.where(wtem < 1.0e-10, 1.0e-10, wtem)
    wmpr = pymax(wmpr, 0.0)
    pe = ops.pw(wtem / (2.0 * k.BY3 * k.TWOPI * scdncw[None, :]), k.BY3, isE)
    rcld_e = rcldlx * 100.0 * pe
    qheatc = (qheatl + fssl * cleara * (ec + er)) / lhx
    rcld_e = jnp.where(isE & (rcld_e > rwmax) & (prep > qheatc), rwmax, rcld_e)
    rclde_e = rcld_e / bybr
    mI = isE & (lhp_l == k.LHS) & (wmpr > 0.0)
    r1 = 1.0e5 * wmpr * pl / (fcld * tl * k.RGAS + k.TEENY)
    r1 = rcldix * 100.0 * ops.pw(r1 / (2.0 * k.BY3 * k.TWOPI * scdnci), k.BY3, mI)
    r1 = pymin(r1, rimax) / bybr
    pi_ = ops.pw(wtem / (2.0 * k.BY3 * k.TWOPI * scdnci), k.BY3, ~isE)
    rcld_i = pymin(rcldix * 100.0 * pi_, rimax)
    rclde_i = rcld_i / bybr
    rclde = jnp.where(isE, rclde_e, rclde_i)
    rclde1 = jnp.where(isE, jnp.where(mI, r1, rclde_e), rclde_i)
    sset = mI | (~isE & (cldssl > 0))
    cs_val = jnp.where(mI, r1, rclde_i)
    rclde1 = 5.0 * rclde1
    cz = jnp.where((fcld <= k.TEENY) & (rclde > 25.0), 25.0, rclde)
    csizel = jnp.concatenate([cz, S["csizel"][lmcld:]], axis=0)
    tem = airm * qx * 1.0e2 * k.BYGRAV
    ts = 1.5e3 * tem / (fcld * rclde + k.TEENY)
    tem1 = airm * wmpr * 1.0e2 * k.BYGRAV
    tem1 = 1.5e3 * tem1 / (fcld * rclde1 + k.TEENY)
    eq = lhp_l == lhx
    vmp_err = (~eq & (lhp_l == k.LHE)).any()
    ts = jnp.where(eq, ts + tem1, ts)
    tausslip = jnp.where(eq, 0.0, jnp.where(fcld <= k.TEENY, 0.0, jnp.where(tem1 > 100.0, 100.0, tem1)))
    ts = jnp.where(fcld <= k.TEENY, 0.0, ts)
    taussl = jnp.where(ts > 100.0, 100.0, ts)
    wmsum = jnp.zeros(N)
    for i in range(lmcld):
        wmsum = jnp.where(isE[i], wmsum + tem[i], wmsum)
    # ---- second loop
    cldsal = cldssl
    cldsv1 = cldssl
    svl = jnp.where(isE, jnp.where(qclx <= 0.0, 0.0, lhx), jnp.where(qcix <= 0.0, 0.0, lhx))
    r = (taumcl == 0.0) | (ckij[None, :] != 1.0)
    bmax = 1.0 - ops.exp(-(cldsv1 / k.C03), r)
    bmax = jnp.where(cldsv1 >= 0.95, cldsv1, bmax)
    bl = r & ((Lr == 1) | (Lr <= dcl))
    new_c = pymin(cldssl + (bmax - cldssl) * ckij[None, :], fssl)
    cldssl2 = jnp.where(bl, new_c, cldssl)
    taussl = jnp.where(bl, taussl * cldsv1 / (cldssl2 + k.TEENY), taussl)
    tausslip = jnp.where(bl, tausslip * cldsv1 / (cldssl2 + k.TEENY), tausslip)
    cldsal = jnp.where(bl, cldssl2, cldsal)
    cldssl2 = jnp.where(r & (taussl <= 0.0), 0.0, cldssl2)
    fr = r & (Lr > dcl) & (taumcl <= 0.0)
    cldssl2 = jnp.where(fr, pymin(ops.pw(cldssl2, k.BY3_2, fr), fssl), cldssl2)
    p3 = ops.pw(cldsv1, k.BY3, fr)
    taussl = jnp.where(fr, taussl * p3, taussl)
    tausslip = jnp.where(fr, tausslip * p3, tausslip)
    cldsal = jnp.where(fr, ops.pw(cldsal, k.TWO3, fr), cldsal)
    neg = taussl < 0.0
    taussl = jnp.where(neg, 0.0, taussl)
    cldssl2 = jnp.where(neg, 0.0, cldssl2)
    qclx2 = jnp.where(neg & isE, 0.0, qclx)
    qcix2 = jnp.where(neg & ~isE, 0.0, qcix)
    ok = ~neg
    qlss = jnp.where(ok & isE, qclx2, 0.0)
    qiss = jnp.where(ok & ~isE, qcix2, 0.0)
    lE = lhp_l == k.LHE
    qlss = jnp.where(ok & lE, qlss + wmpr, qlss)
    qiss = jnp.where(ok & ~lE, qiss + wmpr, qiss)
    tausslip = jnp.where(tausslip < 0.0, 0.0, tausslip)
    # csizelip forward fill along the column axis
    n = N
    idx = jnp.where(sset, jnp.arange(n)[None, :], -1)
    idx = lax.cummax(idx, axis=1)
    got = jnp.take_along_axis(cs_val, jnp.maximum(idx, 0), axis=1)
    cs_f = jnp.where(idx >= 0, got, S["csizelip_init"][:lmcld, None])
    csizelip = jnp.concatenate([cs_f, jnp.broadcast_to(S["csizelip_init"][lmcld:, None], (LM - lmcld, N))], axis=0)
    cat = lambda a, name: jnp.concatenate([a, st[name][lmcld:]], axis=0)  # noqa: E731
    out = dict(cldssl=cat(cldssl2, "cldssl"), taussl=cat(taussl, "taussl"), tausslip=jnp.concatenate([tausslip, S["tausslip"][lmcld:]], axis=0),
               cldsal=jnp.concatenate([cldsal, jnp.zeros((LM - lmcld, N))], axis=0),
               cldsv1=jnp.concatenate([cldsv1, S["cldsv1"][lmcld:]], axis=0), csizel=csizel, csizelip=csizelip,
               qclx=cat(qclx2, "qclx"), qcix=cat(qcix2, "qcix"), svlhxl=cat(svl, "svlhxl"), qlss=cat(qlss, "qlss"), qiss=cat(qiss, "qiss"),
               wmpr=cat(wmpr, "wmpr"))
    return out, wmsum, vmp_err


# ---------------------------------------------------------------------------------------------- driver
@functools.partial(jax.jit, static_argnames=("lmcld", "mode"))
def _core(S, P, K, lmcld, mode):
    ops = OPS[mode]
    k = _Ns(K)
    N = S["tl"].shape[1]
    # ---- main
    (pb0, pi0, lh0, hcnd, ierr, lerr, wmerr), ys = _main(ops, k, S, P, None, lmcld)
    z1 = jnp.zeros((1, N))
    full = lambda name: _assemble(ys, lmcld, S, name)  # noqa: E731

    def full0(name, extra=0):
        a = ys[name][::-1]
        return jnp.concatenate([a, jnp.zeros((LM - lmcld + extra, N))], axis=0)
    st = {n: full(n) for n in "tl ql th rh qclx qcix svlhxl".split()}
    st["airm"], st["byam"], st["pl"], st["plk"], st["fssl"] = S["airm"], S["byam"], S["pl"], S["plk"], S["fssl"]
    for n in "cleara rhf rh00 er ec prep qheatl qheati qheat cldssl cldsavl wmpr rh1 sshr dqlsc".split():
        st[n] = full0(n)
    st["dqlsc"] = S["dqlsc"][:lmcld] * 0 + ys["dqlsc"][::-1]
    st["dqlsc"] = jnp.concatenate([st["dqlsc"], S["dqlsc"][lmcld:]], axis=0)
    st["dctei"] = jnp.zeros((LM, N))
    st["taussl"] = jnp.zeros((LM, N))
    st["qlss"] = jnp.zeros((LM, N))
    st["qiss"] = jnp.zeros((LM, N))
    st["qmom"] = jnp.concatenate([ys["qmom"][::-1], S["qmom"][lmcld:]], axis=0)
    st["smom"], st["um"], st["vm"] = S["smom"], S["um"], S["vm"]
    st["sm"], st["qm"] = S["sm"], S["qm"]
    prebar = jnp.concatenate([ys["prebar"][::-1], jnp.zeros((LM + 1 - lmcld, N))], axis=0)
    preice = jnp.concatenate([pi0[None], ys["preice_ip"][::-1]], axis=0)       # index 0 = final carry, 1..lmcld from the modified ip values
    preice = jnp.concatenate([preice, jnp.zeros((LM + 1 - preice.shape[0], N))], axis=0)
    lhp = jnp.concatenate([ys["lhp"][::-1], jnp.zeros((LM + 1 - lmcld, N))], axis=0)
    prebar1 = jnp.concatenate([ys["prebar1"][::-1], jnp.zeros((LM + 1 - lmcld, N))], axis=0)
    prcpss = pymax(0.0, prebar[0] * k.GRAV * P["dtsrc"])
    W = dict(hcndss=hcnd)
    # ---- CTEI
    c_out, hcnd2, ckij = _ctei(ops, k, S, P, W, lmcld, st)
    for n, a in c_out.items():
        st[n] = jnp.concatenate([a, st[n][lmcld:]], axis=0)
    st["sm"] = jnp.concatenate([c_out["sm"], S["sm"][lmcld:]], axis=0)
    st["qm"] = jnp.concatenate([c_out["qm"], S["qm"][lmcld:]], axis=0)
    # ---- tail
    t_out, wmsum, vmp_err = _tail(ops, k, S, P, W, lmcld, st, ckij, lhp)
    st.update(t_out)
    Sout = {n: st[n] for n in ("tl ql th rh qclx qcix svlhxl cldsavl cldssl taussl tausslip csizel csizelip cldsal cldsv1 qlss qiss sshr dctei sm qm "
                               "rh1 wmpr smom qmom um vm dqlsc").split()}
    Sout["lhp"], Sout["prebar1"] = lhp, prebar1
    Wout = dict(cleara=st["cleara"], rhf=st["rhf"], rh00=st["rh00"], er=st["er"], ec=st["ec"], prep=st["prep"], qheatl=st["qheatl"],
                qheati=st["qheati"], qheat=st["qheat"], prebar=prebar, preice=preice, prcpss=prcpss, hcndss=hcnd2, ierr=ierr, lerr=lerr,
                wmerr=wmerr, ckij=ckij, wmsum=wmsum, vmp_err=vmp_err)
    return Sout, Wout


_SKEYS = ("tl ql th rh qclx qcix svlhxl svlatl svwmxl qcll qcil fssl airm byam pl plk vsubl u00l dpdt ttoldl aq cldsavl sdl precnvl "
          "qmom smom um vm sm qm csizel taumcl tausslip cldsv1 dqlsc csizelip_init cldsal_init").split()
_PSCAL = "cmx u00a rimax rwmax rwcldox rcldix rcldlx wmui scdnci bybr bydtsrc dtsrc".split()


def prepare(S, P):
    """numpy dicts (lscond_batch layout) -> (Sj, Pj, lmcld) with jnp arrays; the host-side scalars that the numpy code evaluates with python
    pow (sndo = 59.68/rwcldox**3) are computed here in python."""
    Sj = {n: jnp.asarray(S[n], jnp.float64) for n in _SKEYS if n in S}
    Sj["csizelip_init"] = jnp.asarray(S["csizelip_init"], jnp.float64)
    Sj["cldsal_init"] = jnp.asarray(S["cldsal_init"], jnp.float64)
    Pj = {n: jnp.asarray(P[n], jnp.float64) for n in _PSCAL}
    for n in ("pearth", "wconst", "scdncw"):
        Pj[n] = jnp.asarray(P[n], jnp.float64)
    Pj["dcl"] = jnp.asarray(P["dcl"], jnp.int64)
    Pj["ra"] = jnp.asarray(P["ra"], jnp.float64)
    Pj["pdsigl00"] = jnp.asarray(S["pdsigl00"], jnp.float64)
    Pj["sndo"] = jnp.float64(59.68 / (P["rwcldox"] ** 3))
    Pj["scdnci_t"] = jnp.float64(0.06417127)
    return Sj, Pj, int(P["lmcld"])


def lscond_jax(S, P, mode="libm", return_jax=False):
    """-> (S_out dict, W dict) as numpy arrays (S_out has the output fields only; moments (LM,9,N), um/vm (LM,4,N))."""
    Sj, Pj, lmcld = prepare(S, P)
    K = {n: jnp.asarray(v) for n, v in make_K().items()}
    Sout, Wout = _core(Sj, Pj, K, lmcld, mode)
    if return_jax:
        return Sout, Wout
    Sout = {n: np.asarray(v) for n, v in Sout.items()}
    Wout = {n: np.asarray(v) for n, v in Wout.items()}
    if Wout["vmp_err"]:
        raise RuntimeError("VMP: should not be here")
    return Sout, Wout
