"""D190 (stage S5): ADVSI (sea-ice advection, ICEDYN_DRV.f:880-1636) as a device program; port of advsi_ff.advsi (D166) with the SAME arithmetic order.

NEW module; advsi_ff.py / seaice_core_ff.py are used for constants and comparison but not edited.

Vectorisation (why it is not a loop on the device): in advsi_ff the fluxes of one sweep (north-south, then east-west) are computed from the state at the start of the sweep
and every cell update then reads only its own cell and the (already computed) fluxes, so the cell loops are independent and are done for all cells at once, with the
GOTO case structure of the Fortran kept as masks (labels 220/230/250/260/270/285 and 520/530/550/560/570/585, crunch 320/620/350).  The two sequential accumulations
(the north-polar box sums over i, and the 6/6/6-term sums of the flux components) are written left to right.  Every cell sees the same operations in the same
order as the scalar code, so the result is bitwise equal wherever the elementary operations are (IEEE +,-,*,/,sqrt; no libm functions are used).
The helpers get_snow_ice_layer / relayer / relayer_12 / set_snow_ice_layer are transcribed from the SCALAR seaice_core_ff versions (NOT seaice_core_jax: its closed
forms associate some products differently, e.g. h + (f-m)*(h'/m') versus h + (f-m)*h'/m').
REAL*16: Ti2b in get_snow_ice_layer is binary128 in the Fortran (SEAICE_FIXES_2022).  Here it is evaluated in DOUBLE-DOUBLE arithmetic (about 106 bits, error-free
transformations without FMA, Dekker splitting) with the Fortran typing rules of advsi_ff.ti2b_quad, and rounded to double at the end.  This is NOT binary128: it is
checked against the mpmath 113-bit reference of advsi_ff on the inputs of the real dumps and on random inputs (d190_test_advsi.py); the number of differing results
is reported there.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import advsi_ff as A  # noqa: E402
import seaice_core_ff as S  # noqa: E402

IM, JM, LMI = 72, 46, 4
NTR = 3 * (LMI + 2)
ACE1I, XSI = A.ACE1I, A.XSI
MU, SHW, SHI, LHM, BYSHI = S.MU, S.SHW, S.SHI, S.LHM, S.BYSHI
SPLIT = 134217729.0          # 2^27 + 1


# ---------------------------------------------------------------------------------------------------------------- double-double (REAL*16 substitute)
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def _split(a):
    t = SPLIT * a
    hi = t - (t - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_add(a, b):
    s1, s2 = _two_sum(a[0], b[0])
    t1, t2 = _two_sum(a[1], b[1])
    s2 = s2 + t1
    s1, s2 = _quick_two_sum(s1, s2)
    s2 = s2 + t2
    return _quick_two_sum(s1, s2)


def dd_neg(a):
    return -a[0], -a[1]


def dd_mul(a, b):
    p1, p2 = _two_prod(a[0], b[0])
    p2 = p2 + (a[0] * b[1] + a[1] * b[0])
    return _quick_two_sum(p1, p2)


def dd_mul_d(a, d):
    p1, p2 = _two_prod(a[0], d)
    p2 = p2 + a[1] * d
    return _quick_two_sum(p1, p2)


def dd_sqrt(a):
    """sqrt of a positive double-double (Newton step on 1/sqrt(hi))."""
    x = 1.0 / jnp.sqrt(a[0])
    ax = a[0] * x
    s = dd_mul((ax, jnp.zeros_like(ax)), (ax, jnp.zeros_like(ax)))
    diff = dd_add(a, dd_neg(s))
    return dd_add((ax, jnp.zeros_like(ax)), (diff[0] * (x * 0.5), jnp.zeros_like(ax)))


def ti2b_dd(eit, si, snowl, mice):
    """advsi_ff.ti2b_quad in double-double (see the module docstring): SEAICE.f:2403 Ti2b with REAL*16 b, c, det, tm."""
    pos = si > 0.0
    tm = -MU * si                                    # double (Fortran: REAL*8 product, then widened)
    frac = mice / (mice + snowl)
    z = jnp.zeros_like(tm)
    p = _two_prod(frac, tm)                          # frac * tm exact
    b = dd_add(dd_mul_d(p, SHW - SHI), (-(eit + LHM), z))
    c = _two_prod(frac * LHM, tm)                    # (frac*lhm in double) * tm
    b2 = dd_mul(b, b)
    k4c = dd_mul_d(c, 4.0 * SHI)
    det = dd_add(b2, dd_neg(k4c))
    det = (jnp.where(pos, det[0], 1.0), jnp.where(pos, det[1], 0.0))      # keep the sqrt argument positive in unused lanes
    sq = dd_sqrt(det)
    s = dd_add(b, sq)
    r = dd_mul_d(dd_mul_d(s, -0.5), BYSHI)
    hi = r[0] + r[1]
    t_lo = (eit + LHM) * BYSHI
    t_lo = jnp.where(jnp.abs(eit + LHM) < 1e-10, 0.0, t_lo)
    return jnp.where(pos, hi, t_lo)


# ---------------------------------------------------------------------------------------------------------------- scalar-order ports of the layer helpers
def Ei(t, si):
    ts = jnp.where(t == 0.0, 1.0, t)
    hi = (t + MU * si) * SHI - LHM * (1.0 + MU * si / ts) - SHW * MU * si
    return jnp.where((si > 0.0) & (t != 0.0), hi, t * SHI - LHM)


def Em(si):
    return -MU * si * SHW


def get_mhs(snow, msi2, hsil, ssil, ti2b=ti2b_dd):
    """advsi_ff.get_snow_ice_layer for every cell: returns the (18, ...) MHS column (mice 0:4, snowl 4:6, hice 6:10, hsnow 10:12, sice 12:16, 0 16:18)."""
    msi1 = snow + ACE1I
    P = ACE1I > XSI[1] * msi1
    z = jnp.zeros_like(snow)
    h0, h1, h2, h3 = (hsil[..., k] for k in range(4))
    s0, s1, s2, s3 = (ssil[..., k] for k in range(4))
    # ---- branch P
    miceP0 = ACE1I - XSI[1] * msi1
    miceP1 = XSI[1] * msi1
    slP0 = snow
    miceP0s = jnp.where(P, miceP0, 1.0)
    siP = 1e3 * s0 / miceP0s
    tiP = ti2b(h0 / (XSI[0] * msi1), siP, slP0, miceP0s)
    tiP = jnp.where((tiP > -1e-15) & (tiP < 0.0), 0.0, tiP)
    hiP0 = jnp.minimum(jnp.maximum(miceP0s * Ei(tiP, siP), h0), miceP0s * Em(siP))
    hsP0 = h0 - hiP0
    hsP0 = jnp.where((slP0 == 0.0) | (jnp.abs(h0 - hiP0) < 1e-8), 0.0, hsP0)
    hsP0 = jnp.where((tiP < 0) & (slP0 > 0) & (hiP0 != miceP0s * Em(siP)), jnp.minimum(hsP0, (tiP * SHI - LHM) * slP0), hsP0)
    # ---- branch Q
    slQ0 = XSI[0] * msi1
    slQ1 = XSI[1] * msi1 - ACE1I
    miceQ1 = ACE1I
    siQ = 1e3 * s1 / miceQ1
    tiQ = ti2b(h1 / (XSI[1] * msi1), siQ, slQ1, miceQ1 + z)
    tiQ = jnp.where((tiQ > -1e-15) & (tiQ < 0.0), 0.0, tiQ)
    hiQ1 = jnp.minimum(jnp.maximum(miceQ1 * Ei(tiQ, siQ), h1), miceQ1 * Em(siQ))
    hsQ1 = h1 - hiQ1
    hsQ1 = jnp.where((slQ1 == 0.0) | (jnp.abs(h1 - hiQ1) < 1e-8), 0.0, hsQ1)
    hsQ1 = jnp.where((tiQ < 0) & (slQ1 > 0) & (hiQ1 != miceQ1 * Em(siQ)), jnp.minimum(hsQ1, (tiQ * SHI - LHM) * slQ1), hsQ1)
    sel = lambda a, b: jnp.where(P, a, b)       # noqa: E731
    mice0, mice1 = sel(miceP0, z), sel(miceP1, miceQ1 + z)
    snowl0, snowl1 = sel(slP0, slQ0), sel(z, slQ1)
    hice0, hice1 = sel(hiP0, z), sel(h1, hiQ1)
    hsnow0, hsnow1 = sel(hsP0, h0), sel(z, hsQ1)
    sice0, sice1 = sel(s0, z), s1
    mice2, mice3 = XSI[2] * msi2, XSI[3] * msi2
    return jnp.stack([mice0, mice1, mice2, mice3, snowl0, snowl1, hice0, hice1, h2, h3, hsnow0, hsnow1, sice0, sice1, s2, s3, z, z])


def relayer(fmsi2, mice, hice, sice):
    """seaice_core_ff.relayer on lists of 4 arrays; returns new lists."""
    m, h, s = list(mice), list(hice), list(sice)
    msi2 = m[2] + m[3]
    fm = [None] * 4
    fh = [None] * 4
    fs = [None] * 4
    for l in (1, 2):
        sx = 1.0 if l == 1 else 0.5                               # sum(XSI[l+1:4])
        sm = (m[2] + m[3]) if l == 1 else m[3]                    # sum(mice[l+1:4])
        f = sx * (msi2 + fmsi2) - sm
        gt = f > m[l]
        a_h = h[l] + (f - m[l]) * h[l - 1] / m[l - 1]
        a_s = s[l] + (f - m[l]) * s[l - 1] / m[l - 1]
        b_h = h[l] * f / m[l]
        b_s = s[l] * f / m[l]
        c_h = h[l + 1] * f / m[l + 1]
        c_s = s[l + 1] * f / m[l + 1]
        pos = f > 0
        fm[l] = f
        fh[l] = jnp.where(pos, jnp.where(gt, a_h, b_h), c_h)
        fs[l] = jnp.where(pos, jnp.where(gt, a_s, b_s), c_s)
    for l in range(1, LMI):
        if l > 1:
            m[l] = m[l] + fm[l - 1]
            h[l] = h[l] + fh[l - 1]
            s[l] = s[l] + fs[l - 1]
        if l < LMI - 1:
            m[l] = m[l] - fm[l]
            h[l] = h[l] - fh[l]
            s[l] = s[l] - fs[l]
    return m, h, s


def relayer_12(hsnow, hice, sice, mice, snowl):
    """seaice_core_ff.relayer_12 on arrays; hsnow/snowl lists of 2, hice/sice/mice lists of 4.  Returns the new lists (layers 2, 3 pass through)."""
    hs0, hs1 = hsnow
    hi0, hi1 = hice[0], hice[1]
    si0, si1 = sice[0], sice[1]
    mi0, mi1 = mice[0], mice[1]
    sl0, sl1 = snowl
    z = jnp.zeros_like(mi0)
    fmsi1 = sl0 + mi0 - XSI[0] * (sl0 + sl1 + ACE1I)
    fmsi1 = jnp.where(jnp.abs(fmsi1) < 1e-14, 0.0, fmsi1)
    fmsi0 = XSI[0] * (sl0 + sl1 + ACE1I)
    A_ = (sl1 > 0.0) & (mi0 + mi1 > fmsi0)
    B_ = ~A_ & (mi0 > 0.0) & (mi0 + mi1 < fmsi0)
    # ---- A
    gA = fmsi0 > mi1
    fssA = jnp.where(gA, (fmsi0 - mi1) * si0 / mi0, (mi1 - fmsi0) * si1 / mi1)
    fhsA = jnp.where(gA, (fmsi0 - mi1) * hi0 / mi0, (mi1 - fmsi0) * hi1 / mi1)
    A_si0 = jnp.where(gA, si0 - fssA, si0 + fssA)
    A_si1 = jnp.where(gA, si1 + fssA, si1 - fssA)
    A_hi0 = jnp.where(gA, hi0 - fhsA, hi0 + fhsA)
    A_hi1 = jnp.where(gA, hi1 + fhsA, hi1 - fhsA)
    A_mi0 = mi0 + mi1 - fmsi0
    A_mi1 = fmsi0
    A_sl0 = sl0 + sl1
    A_sl1 = z
    A_hs0 = hs0 + hs1
    A_hs1 = z
    # ---- B
    lB = fmsi0 < sl0
    fhB = jnp.where(lB, (sl0 - fmsi0) * hs0 / sl0, (fmsi0 - sl0) * hs1 / sl1)
    B_hs1 = jnp.where(lB, hs1 + fhB, hs1 - fhB)
    B_hs0 = jnp.where(lB, hs0 - fhB, hs0 + fhB)
    B_sl1 = sl0 + sl1 - fmsi0
    B_sl0 = fmsi0
    B_hi1, B_hi0 = hi1 + hi0, z
    B_si1, B_si0 = si1 + si0, z
    B_mi1, B_mi0 = mi1 + mi0, z
    # ---- C
    c_pos = fmsi1 > 0
    c_m = mi0 > 0
    c1 = c_pos & c_m & (fmsi1 > mi0)
    c2 = c_pos & c_m & ~(fmsi1 > mi0)
    c3 = c_pos & ~c_m
    c4 = ~c_pos & c_m
    c5 = ~c_pos & ~c_m
    c5a = c5 & (sl1 + fmsi1 < 0)
    c5b = c5 & ~(sl1 + fmsi1 < 0)
    # C1
    C1_hs1 = hs1 + (fmsi1 - mi0) * hs0 / sl0
    C1_hs0 = hs0 - (fmsi1 - mi0) * hs0 / sl0
    C1_sl1 = sl1 + (fmsi1 - mi0)
    C1_sl0 = sl0 - (fmsi1 - mi0)
    # C2 (and C4 shares the structure with different fluxes)
    fh2 = fmsi1 * hi0 / mi0
    fs2 = fmsi1 * si0 / mi0
    fh4 = fmsi1 * hi1 / mi1
    fs4 = fmsi1 * si1 / mi1
    fhx = jnp.where(c4, fh4, fh2)
    fsx = jnp.where(c4, fs4, fs2)
    C24_mi0, C24_mi1 = mi0 - fmsi1, mi1 + fmsi1
    C24_hi0, C24_hi1 = hi0 - fhx, hi1 + fhx
    C24_si0, C24_si1 = si0 - fsx, si1 + fsx
    # C3 (and C5b share structure)
    fh3 = fmsi1 * hs0 / sl0
    fh5 = fmsi1 * hs1 / sl1
    fhy = jnp.where(c5b, fh5, fh3)
    C35_sl0, C35_sl1 = sl0 - fmsi1, sl1 + fmsi1
    C35_hs0, C35_hs1 = hs0 - fhy, hs1 + fhy
    # C5a
    t5 = -(sl1 + fmsi1)
    C5a_hi0 = t5 * hi1 / mi1
    C5a_hi1 = hi1 - C5a_hi0
    C5a_si0 = t5 * si1 / mi1
    C5a_si1 = si1 - C5a_si0
    C5a_mi0 = t5
    C5a_mi1 = mi1 - C5a_mi0
    C5a_hs0, C5a_hs1 = hs0 + hs1, z
    C5a_sl0, C5a_sl1 = sl0 + sl1, z
    # ---- merge (A, B, then the C leaves)
    def pick(a, b, c1_, c24, c35, c5a_, keep):
        r = keep
        r = jnp.where(c5a, c5a_, r) if c5a_ is not None else r
        r = jnp.where(c3 | c5b, c35, r) if c35 is not None else r
        r = jnp.where(c2 | c4, c24, r) if c24 is not None else r
        r = jnp.where(c1, c1_, r) if c1_ is not None else r
        r = jnp.where(B_, b, r) if b is not None else r
        r = jnp.where(A_, a, r) if a is not None else r
        return r
    hs0n = pick(A_hs0, B_hs0, C1_hs0, None, C35_hs0, C5a_hs0, hs0)
    hs1n = pick(A_hs1, B_hs1, C1_hs1, None, C35_hs1, C5a_hs1, hs1)
    sl0n = pick(A_sl0, B_sl0, C1_sl0, None, C35_sl0, C5a_sl0, sl0)
    sl1n = pick(A_sl1, B_sl1, C1_sl1, None, C35_sl1, C5a_sl1, sl1)
    # C1 also moves the ice layer 0 into layer 1
    hi0n = pick(A_hi0, B_hi0, z, C24_hi0, None, C5a_hi0, hi0)
    hi1n = pick(A_hi1, B_hi1, hi1 + hi0, C24_hi1, None, C5a_hi1, hi1)
    si0n = pick(A_si0, B_si0, z, C24_si0, None, C5a_si0, si0)
    si1n = pick(A_si1, B_si1, si1 + si0, C24_si1, None, C5a_si1, si1)
    mi0n = pick(A_mi0, B_mi0, z, C24_mi0, None, C5a_mi0, mi0)
    mi1n = pick(A_mi1, B_mi1, mi1 + mi0, C24_mi1, None, C5a_mi1, mi1)
    return ([hs0n, hs1n], [hi0n, hi1n, hice[2], hice[3]], [si0n, si1n, sice[2], sice[3]], [mi0n, mi1n, mice[2], mice[3]], [sl0n, sl1n])


# ---------------------------------------------------------------------------------------------------------------- sums in Fortran order
def _lsum(xs):
    s = xs[0]
    for x in xs[1:]:
        s = s + x
    return s


def _crunch(amsi, asi, byfoa):
    """label 320/620/350 for arrays: amsi (18, ...) -> new mhs (18, ...)."""
    out = []
    for k in range(3):
        b = 6 * k
        a = [amsi[b + q] for q in range(6)]
        s = a[0] + a[1] + a[2] + a[3]
        dm = s * (byfoa - 1.0 / asi)
        out += [a[0] / asi, a[1] / asi, a[2] / asi + XSI[2] * dm, a[3] / asi + XSI[3] * dm, a[4] * byfoa, a[5] * byfoa]
    return jnp.stack(out)


def _limit(rsi, rx, ry):
    rx = jnp.where(rsi - rx < 0.0, rsi, rx)
    rx = jnp.where(rsi + rx < 0.0, -rsi, rx)
    rx = jnp.where(rsi - rx > 1.0, rsi - 1.0, rx)
    rx = jnp.where(rsi + rx > 1.0, 1.0 - rsi, rx)
    ry = jnp.where(rsi - ry < 0.0, rsi, ry)
    ry = jnp.where(rsi + ry < 0.0, -rsi, ry)
    ry = jnp.where(rsi - ry > 1.0, rsi - 1.0, ry)
    ry = jnp.where(rsi + ry > 1.0, 1.0 - rsi, ry)
    return rx, ry


# ---------------------------------------------------------------------------------------------------------------- statics
def make_static(focean, geo):
    """Host-side constants: focean, CONNECT (from focean), the bit fields of CONNECT used by the velocity limiter, the five geometry vectors."""
    if np.any(focean[:, 0] > 0.0):
        raise NotImplementedError('south-pole row has ocean')
    foc = np.asarray(focean, float)
    conn = A.connect_from_focean(foc)
    ip1 = np.roll(np.arange(IM), -1)
    ci = conn.astype(np.int64)
    K = dict(foc=foc, conn=conn, geo={k: np.asarray(geo[k], float) for k in ('dxyp', 'dyp', 'dxp', 'dxv', 'bydxyp')})
    K['cxi'] = ci % 2
    K['cxip1'] = (ci[ip1] % 4) // 2
    K['u_lim'] = (conn + conn[ip1]) < 30.0
    K['cyj'] = (ci // 4) % 2
    K['cyjp1'] = np.zeros_like(ci)
    K['cyjp1'][:, :JM - 1] = (conn[:, 1:] / 8.0).astype(np.int64)
    K['v_lim'] = np.zeros((IM, JM), bool)
    K['v_lim'][:, :JM - 1] = (conn[:, :JM - 1] + conn[:, 1:]) < 30.0
    return K


# ---------------------------------------------------------------------------------------------------------------- ADVSI
def advsi(K, st, ausi, avsi, dts=1800.0, ti2b=ti2b_dd):
    """advsi_ff.advsi on device arrays.  st: dict rsi, rsix, rsiy, rsisave, msi, snowi (IM,JM), hsi, ssi (IM,JM,4).  Returns (new state, out)."""
    foc = jnp.asarray(K['foc'])
    g = {k: jnp.asarray(v) for k, v in K['geo'].items()}
    dxyp, dyp, dxp, dxv, bydxyp = g['dxyp'], g['dyp'], g['dxp'], g['dxv'], g['bydxyp']
    rsi, rsix, rsiy, rsisave = st['rsi'], st['rsix'], st['rsiy'], st['rsisave']
    msi, snowi, hsi, ssi = st['msi'], st['snowi'], st['hsi'], st['ssi']
    z2 = jnp.zeros((IM, JM))
    # ---- regularise ice concentration gradients (J = 2..JM-1)
    r = rsi
    rowm = jnp.asarray((np.arange(JM) >= 1) & (np.arange(JM) <= JM - 2))[None, :]
    big = r > 1e-4
    frsi = (rsisave - r) / rsisave
    shr = rsisave > r
    rx = jnp.where(shr, rsix * (1.0 - frsi), rsix)
    ry = jnp.where(shr, rsiy * (1.0 - frsi), rsiy)
    rx, ry = _limit(r, rx, ry)
    rsix = jnp.where(rowm, jnp.where(big, rx, 0.0), rsix)
    rsiy = jnp.where(rowm, jnp.where(big, ry, 0.0), rsiy)
    # ---- MHS
    mhs = get_mhs(snowi, msi, hsi, ssi, ti2b)
    byfoa = jnp.where(foc > 0.0, bydxyp[None, :] / jnp.where(foc > 0.0, foc, 1.0), 0.0)
    hsicnv0 = jnp.where(foc > 0.0, rsi * (((hsi[..., 0] + hsi[..., 1]) + hsi[..., 2]) + hsi[..., 3]), 0.0)
    out_hsicnv = hsicnv0
    # ---- velocities times dt
    ip1 = np.roll(np.arange(IM), -1)
    im1 = np.roll(np.arange(IM), 1)
    jidx = np.arange(JM)
    cf_ok = jnp.asarray((jidx != 0) & (jidx != JM - 1))
    cfx = jnp.where(cf_ok, 1e-3 * 1e-1 * 1e5 / jnp.where(dxp == 0, 1.0, dxp), 0.0)[None, :]
    cfy = jnp.where(cf_ok, 1e-3 * 1e-1 * 1e5 / jnp.where(dyp == 0, 1.0, dyp), 0.0)[None, :]
    ausi_jm1 = jnp.roll(ausi, 1, axis=1)               # ausi[:, j-1]
    cond_u = (foc > 0.0) & (foc[ip1] > 0.0) & (rsisave + rsisave[ip1] > 1e-4)
    uv = 0.5 * (ausi_jm1 + ausi) * dts
    du = (msi - msi[ip1]) * cfx
    du = jnp.minimum(10.0, jnp.maximum(-10.0, du))
    cxi, cxip1 = jnp.asarray(K['cxi']), jnp.asarray(K['cxip1'])
    du = jnp.where(cxi < cxip1, jnp.maximum(0.0, du), jnp.where(cxi > cxip1, jnp.minimum(0.0, du), du))
    usidt = jnp.where(cond_u, jnp.where(jnp.asarray(K['u_lim']), uv + dts * du, uv), 0.0)
    fsh = lambda a: jnp.concatenate([a[:, 1:], jnp.zeros((IM, 1))], axis=1)       # a[:, j+1]  # noqa: E731
    foc_jp1 = fsh(foc)
    cond_v = (foc_jp1 > 0.0) & (foc > 0.0) & (rsisave + fsh(rsisave) > 1e-4)
    vv = 0.5 * (avsi[im1] + avsi) * dts
    dv = (msi - fsh(msi)) * cfy
    dv = jnp.minimum(10.0, jnp.maximum(-10.0, dv))
    cyj, cyjp1 = jnp.asarray(K['cyj']), jnp.asarray(K['cyjp1'])
    dv = jnp.where(cyj < cyjp1, jnp.maximum(0.0, dv), jnp.where(cyj > cyjp1, jnp.minimum(0.0, dv), dv))
    vsidt = jnp.where(cond_v, jnp.where(jnp.asarray(K['v_lim']), vv + dts * dv, vv), 0.0)
    jlast = jnp.asarray(jidx == JM - 1)[None, :]
    vsidt = jnp.where(jlast, 0.0, vsidt)
    usidt = jnp.where(jlast, ausi[0, JM - 1] * dts, usidt)
    # (rows j <= JM-2 of usidt/vsidt come from the loop `for j in range(JM-1)`)
    rsisave = rsi
    # ---- north-south fluxes, J = 2..JM-2 (0-based j = 1 .. JM-3) and the row next to the pole (j = JM-2)
    J = slice(1, JM - 2)
    J1 = slice(2, JM - 1)
    v = vsidt[:, J]
    f = v * dxv[None, 2:JM - 1]
    neg = v <= 0.0
    byj, byj1 = bydxyp[None, 1:JM - 2], bydxyp[None, 2:JM - 1]
    fa_n = f * (rsi[:, J1] - (1.0 + f * byj1) * rsiy[:, J1]) * foc[:, J1]
    fx_n = f * rsix[:, J1] * foc[:, J1]
    fy_n = f * (f * byj1 * f * rsiy[:, J1] * foc[:, J1] - 3.0 * fa_n)
    fa_p = f * (rsi[:, J] + (1.0 - f * byj) * rsiy[:, J]) * foc[:, J]
    fx_p = f * rsix[:, J] * foc[:, J]
    fy_p = f * (f * byj * f * rsiy[:, J] * foc[:, J] - 3.0 * fa_p)
    nz = v != 0.0
    fa = jnp.where(nz, jnp.where(neg, fa_n, fa_p), 0.0)
    fx = jnp.where(nz, jnp.where(neg, fx_n, fx_p), 0.0)
    fy = jnp.where(nz, jnp.where(neg, fy_n, fy_p), 0.0)
    fm = jnp.where(nz[None], jnp.where(neg[None], fa_n[None] * mhs[:, :, J1], fa_p[None] * mhs[:, :, J]), 0.0)    # (18, IM, JM-3)
    faw_ = jnp.where(nz, f, 0.0)
    # pole-adjacent row j = JM-2
    jn = JM - 2
    vn = vsidt[:, jn]
    fn = vn * dxv[JM - 1]
    negn = vn <= 0.0
    nzn = vn != 0.0
    fa_pn = fn * foc[:, jn] * (rsi[:, jn] + (1.0 - fn * bydxyp[jn]) * rsiy[:, jn])
    fa_nn = fn * rsi[0, JM - 1] * foc[0, JM - 1]
    fan = jnp.where(negn, fa_nn, fa_pn)
    fxn = jnp.where(negn, 0.0, fn * rsix[:, jn] * foc[:, jn])
    fyn = jnp.where(negn, -fn * fan, fn * (fn * bydxyp[jn] * fn * rsiy[:, jn] * foc[:, jn] - 3.0 * fan))
    fmn = jnp.where(negn[None], fan[None] * mhs[:, 0, JM - 1][:, None], fan[None] * mhs[:, :, jn])           # (18, IM)
    fan = jnp.where(nzn, fan, 0.0)
    fxn = jnp.where(nzn, fxn, 0.0)
    fyn = jnp.where(nzn, fyn, 0.0)
    fmn = jnp.where(nzn[None], fmn, 0.0)
    fawn = jnp.where(nzn, fn, 0.0)
    # sequential sums over i (north polar box)
    def over_i(a):
        s = a[..., 0]
        for i in range(1, IM):
            s = s + a[..., i]
        return s
    sfasi = over_i(fan)
    sfmsi = over_i(fmn)
    # assemble full-grid flux arrays
    faw = z2.at[:, J].set(faw_).at[:, jn].set(fawn)
    fasi = z2.at[:, J].set(fa).at[:, jn].set(fan)
    fxsi = z2.at[:, J].set(fx).at[:, jn].set(fxn)
    fysi = z2.at[:, J].set(fy).at[:, jn].set(fyn)
    fmsj = jnp.zeros((NTR, IM, JM)).at[:, :, J].set(fm).at[:, :, jn].set(fmn)
    # diagnostics mvsi, hvsi, svsi (rows with a flux)
    def comp_sums(fmx):
        return _lsum([fmx[k] for k in range(0, 6)]), _lsum([fmx[k] for k in range(6, 12)]), _lsum([fmx[k] for k in range(12, 18)])
    mvsi, hvsi, svsi = comp_sums(fmsj)
    # ---- update for south-north fluxes (J = 2..JM-1: 0-based j = 1 .. JM-2)
    U = slice(1, JM - 1)
    Um = slice(0, JM - 2)
    dx = dxyp[None, U]
    fo = foc[:, U]
    by = byfoa[:, U]
    bd = bydxyp[None, U]
    vm, vc = vsidt[:, Um], vsidt[:, U]
    rs, rxx, ryy = rsi[:, U], rsix[:, U], rsiy[:, U]
    mh = mhs[:, :, U]
    fa_c, fa_m = fasi[:, U], fasi[:, Um]
    fw_c, fw_m = faw[:, U], faw[:, Um]
    fx_c, fx_m = fxsi[:, U], fxsi[:, Um]
    fy_c, fy_m = fysi[:, U], fysi[:, Um]
    fm_c, fm_m = fmsj[:, :, U], fmsj[:, :, Um]
    foc_m = foc[:, Um]
    c260 = ((vm < 0.0) & (vc < 0.0)) | ((vm > 0.0) & (vc != 0.0))
    c250 = (vm < 0.0) & (vc == 0.0)
    c270 = (vm < 0.0) & (vc > 0.0)
    c220 = (vm == 0.0) & (vc < 0.0)
    c230 = (vm == 0.0) & (vc > 0.0)
    c285 = (vm > 0.0) & (vc == 0.0)
    act = c260 | c250 | c270 | c220 | c230 | c285
    # asi / amsi / yrsi of the three flux-update cases
    base = rs * dx * fo
    amb = (rs * dx)[None] * mh * fo[None]
    asi_220 = base - fa_c
    am_220 = amb - fm_c
    asi_260 = base + (fa_m - fa_c)
    am_260 = amb + (fm_m - fm_c)
    asi_285 = base + fa_m
    am_285 = amb + fm_m
    yr_220 = (ryy * dx * dx * fo - fy_c + 3.0 * (fw_c * asi_220 - dx * fa_c)) / (dx - fw_c)
    yr_260 = (ryy * dx * dx * fo + (fy_m - fy_c) + 3.0 * ((fw_m + fw_c) * asi_260 - dx * (fa_m + fa_c))) / (dx + (fw_m - fw_c))
    yr_285 = (ryy * dx * dx * fo + fy_m + 3.0 * (fw_m * asi_285 - dx * fa_m)) / (dx + fw_m)
    fl3 = c220 | c260 | c285
    asi = jnp.where(c220, asi_220, jnp.where(c260, asi_260, asi_285))
    amsi = jnp.where(c220[None], am_220, jnp.where(c260[None], am_260, am_285))
    yrsi = jnp.where(c220, yr_220, jnp.where(c260, yr_260, yr_285))
    crunch = fl3 & (asi > dx * fo)
    fine3 = fl3 & ~crunch
    # new values, non-crunch
    n_rsi = jnp.where(fine3, asi * by, rs)
    n_rsi = jnp.where(fine3 & ~c285 & (n_rsi > 1.0), 1.0, n_rsi)
    n_rsiy = jnp.where(fine3, yrsi * by, ryy)
    n_rsix = rxx
    n_rsix = jnp.where(fine3 & c220, rxx - fx_c * by, n_rsix)
    n_rsix = jnp.where(fine3 & c260, rxx + (fx_m - fx_c) * by, n_rsix)
    n_rsix = jnp.where(fine3 & c285, rxx + fx_m * by, n_rsix)
    n_mh = jnp.where((fine3 & (asi > 0.0))[None], amsi / jnp.where(asi == 0.0, 1.0, asi), mh)
    # simple cases
    t230 = (1.0 - fw_c * bd)
    n_rsi = jnp.where(c230, rs - fa_c * by, n_rsi)
    n_rsix = jnp.where(c230, rxx * t230, n_rsix)
    n_rsiy = jnp.where(c230, ryy * (t230 * t230), n_rsiy)
    t250 = (1.0 + fw_m * foc_m * by)
    n_rsi = jnp.where(c250, rs + fa_m * by, n_rsi)
    n_rsix = jnp.where(c250, rxx * t250, n_rsix)
    n_rsiy = jnp.where(c250, ryy * (t250 * t250), n_rsiy)
    t270 = (1.0 + (fw_m * foc_m - fw_c * fo) * by)
    n_rsi = jnp.where(c270, rs + (fa_m - fa_c) * by, n_rsi)
    n_rsix = jnp.where(c270, rxx * t270, n_rsix)
    n_rsiy = jnp.where(c270, ryy * (t270 * t270), n_rsiy)
    # finish: crunch or max(0, .) and the limits
    n_rsi2 = jnp.maximum(0.0, n_rsi)
    lx, ly = _limit(n_rsi2, n_rsix, n_rsiy)
    cm = _crunch(amsi, jnp.where(crunch, asi, 1.0), by)
    f_rsi = jnp.where(crunch, 1.0, jnp.where(act, n_rsi2, rs))
    f_rsix = jnp.where(crunch, 0.0, jnp.where(act, lx, rxx))
    f_rsiy = jnp.where(crunch, 0.0, jnp.where(act, ly, ryy))
    f_mh = jnp.where(crunch[None], cm, n_mh)
    rsi = rsi.at[:, U].set(f_rsi)
    rsix = rsix.at[:, U].set(f_rsix)
    rsiy = rsiy.at[:, U].set(f_rsiy)
    mhs = mhs.at[:, :, U].set(f_mh)
    # ---- north pole box
    j = JM - 1
    asi_p = rsi[0, j] * dxyp[j] * foc[0, j] + sfasi / IM
    amsi_p = rsi[0, j] * dxyp[j] * mhs[:, 0, j] * foc[0, j] + sfmsi / IM
    crp = asi_p > dxyp[j] * foc[0, j]
    cmp_ = _crunch(amsi_p, jnp.where(crp, asi_p, 1.0), byfoa[0, j])
    rsi_p = jnp.where(crp, 1.0, asi_p * byfoa[0, j])
    mh_p = jnp.where(crp, cmp_, jnp.where(asi_p > 0.0, amsi_p / jnp.where(asi_p == 0.0, 1.0, asi_p), mhs[:, 0, j]))
    rsi = rsi.at[0, j].set(rsi_p)
    mhs = mhs.at[:, 0, j].set(mh_p)
    # ---- east-west advection (J = 2..JM-1)
    E = slice(1, JM - 1)
    u = usidt[:, E]
    dyp_j = dyp[None, E]
    dxj = dxyp[None, E]
    bdj = bydxyp[None, E]
    f = u * dyp_j
    negu = u <= 0.0
    rsi_e, rsix_e, rsiy_e, foc_e = rsi[:, E], rsix[:, E], rsiy[:, E], foc[:, E]
    rsi_i, rsix_i, rsiy_i, foc_i = rsi_e[ip1], rsix_e[ip1], rsiy_e[ip1], foc_e[ip1]
    mh_e = mhs[:, :, E]
    fa_n = f * (rsi_i - (1.0 + f * bdj) * rsix_i) * foc_i
    fx_n = f * (f * bdj * f * rsix_i * foc_i - 3.0 * fa_n)
    fy_n = f * rsiy_i * foc_i
    fm_n = fa_n[None] * mh_e[:, ip1]
    fa_p = f * (rsi_e + (1.0 - f * bdj) * rsix_e) * foc_e
    fx_p = f * (f * bdj * f * rsix_e * foc_e - 3.0 * fa_p)
    fy_p = f * rsiy_e * foc_e
    fm_p = fa_p[None] * mh_e
    nzu = u != 0.0
    faE = jnp.where(nzu, jnp.where(negu, fa_n, fa_p), 0.0)
    fxE = jnp.where(nzu, jnp.where(negu, fx_n, fx_p), 0.0)
    fyE = jnp.where(nzu, jnp.where(negu, fy_n, fy_p), 0.0)
    fmE = jnp.where(nzu[None], jnp.where(negu[None], fm_n, fm_p), 0.0)
    fwE = jnp.where(nzu, f, 0.0)
    musi_E = _lsum([fmE[k] for k in range(0, 6)])
    husi_E = _lsum([fmE[k] for k in range(6, 12)])
    susi_E = _lsum([fmE[k] for k in range(12, 18)])
    um, uc = usidt[im1][:, E], usidt[:, E]
    e560 = ((um < 0.0) & (uc < 0.0)) | ((um > 0.0) & (uc != 0.0))
    e550 = (um < 0.0) & (uc == 0.0)
    e570 = (um < 0.0) & (uc > 0.0)
    e520 = (um == 0.0) & (uc < 0.0)
    e530 = (um == 0.0) & (uc > 0.0)
    e585 = (um > 0.0) & (uc == 0.0)
    actE = e560 | e550 | e570 | e520 | e530 | e585
    fo = foc_e
    by = byfoa[:, E]
    fa_c, fa_m = faE, faE[im1]
    fw_c, fw_m = fwE, fwE[im1]
    fx_c, fx_m = fxE, fxE[im1]
    fy_c, fy_m = fyE, fyE[im1]
    fm_c, fm_m = fmE, fmE[:, im1]
    foc_m = foc_e[im1]
    base = rsi_e * dxj * fo
    amb = (rsi_e * dxj)[None] * mh_e * fo[None]
    asi_520 = base - fa_c
    am_520 = amb - fm_c
    asi_560 = base + (fa_m - fa_c)
    am_560 = amb + (fm_m - fm_c)
    asi_585 = base + fa_m
    am_585 = amb + fm_m
    xr_520 = (rsix_e * dxj * dxj * fo - fx_c + 3.0 * (fw_c * asi_520 - dxj * fa_c)) / (dxj - fw_c)
    xr_560 = (rsix_e * dxj * dxj * fo + (fx_m - fx_c) + 3.0 * ((fw_m + fw_c) * asi_560 - dxj * (fa_m + fa_c))) / (dxj + (fw_m - fw_c))
    xr_585 = (rsix_e * dxj * dxj * fo + fx_m + 3.0 * (fw_m * asi_585 - dxj * fa_m)) / (dxj + fw_m)
    fl3 = e520 | e560 | e585
    asi = jnp.where(e520, asi_520, jnp.where(e560, asi_560, asi_585))
    amsi = jnp.where(e520[None], am_520, jnp.where(e560[None], am_560, am_585))
    xrsi = jnp.where(e520, xr_520, jnp.where(e560, xr_560, xr_585))
    crunch = fl3 & (asi > dxj * fo)
    fine3 = fl3 & ~crunch
    n_rsi = jnp.where(fine3, asi * by, rsi_e)
    n_rsix = jnp.where(fine3, xrsi * by, rsix_e)
    n_rsiy = rsiy_e
    n_rsiy = jnp.where(fine3 & e520, rsiy_e - fy_c * by, n_rsiy)
    n_rsiy = jnp.where(fine3 & e560, rsiy_e + (fy_m - fy_c) * by, n_rsiy)
    n_rsiy = jnp.where(fine3 & e585, rsiy_e + fy_m * by, n_rsiy)
    n_mh = jnp.where((fine3 & (asi > 0.0))[None], amsi / jnp.where(asi == 0.0, 1.0, asi), mh_e)
    t530 = (1.0 - fw_c * bdj)
    n_rsi = jnp.where(e530, rsi_e - fa_c * by, n_rsi)
    n_rsix = jnp.where(e530, rsix_e * (t530 * t530), n_rsix)
    n_rsiy = jnp.where(e530, rsiy_e * t530, n_rsiy)
    t550 = (1.0 + fw_m * foc_m * by)
    n_rsi = jnp.where(e550, rsi_e + fa_m * by, n_rsi)
    n_rsix = jnp.where(e550, rsix_e * (t550 * t550), n_rsix)
    n_rsiy = jnp.where(e550, rsiy_e * t550, n_rsiy)
    t570 = (1.0 + (fw_m * foc_m - fw_c * fo) * by)
    n_rsi = jnp.where(e570, rsi_e + (fa_m - fa_c) * by, n_rsi)
    n_rsix = jnp.where(e570, rsix_e * (t570 * t570), n_rsix)
    n_rsiy = jnp.where(e570, rsiy_e * t570, n_rsiy)
    n_rsi2 = jnp.maximum(0.0, n_rsi)
    lx, ly = _limit(n_rsi2, n_rsix, n_rsiy)
    n_rsi3 = jnp.where(n_rsi2 > 1.0, 1.0, n_rsi2)
    cm = _crunch(amsi, jnp.where(crunch, asi, 1.0), by)
    f_rsi = jnp.where(crunch, 1.0, jnp.where(actE, n_rsi3, rsi_e))
    f_rsix = jnp.where(crunch, 0.0, jnp.where(actE, lx, rsix_e))
    f_rsiy = jnp.where(crunch, 0.0, jnp.where(actE, ly, rsiy_e))
    f_mh = jnp.where(crunch[None], cm, n_mh)
    rsi = rsi.at[:, E].set(f_rsi)
    rsix = rsix.at[:, E].set(f_rsix)
    rsiy = rsiy.at[:, E].set(f_rsiy)
    mhs = mhs.at[:, :, E].set(f_mh)
    # diagnostics musi/husi/susi (rows 1..JM-2) and mvsi/hvsi/svsi
    musi = z2.at[:, E].set(musi_E)
    husi = z2.at[:, E].set(husi_E)
    susi = z2.at[:, E].set(susi_E)
    # ---- back to the thermal layers (KOCEAN >= 1)
    vis = np.zeros((IM, JM), bool)
    vis[:, 1:JM - 1] = True
    vis[0, 0] = vis[0, JM - 1] = True
    vis = jnp.asarray(vis) & (foc > 0.0)
    m = mhs
    s_all = _lsum([m[k] for k in range(0, 6)])
    s_salt = _lsum([m[k] for k in range(12, 16)])
    sold = ((ssi[..., 0] + ssi[..., 1]) + ssi[..., 2]) + ssi[..., 3]
    msicnv = rsi * (s_all - s_salt) - rsisave * (ACE1I + snowi + msi - sold)
    mice = [m[0], m[1], m[2], m[3]]
    snowl = [m[4], m[5]]
    hice = [m[6], m[7], m[8], m[9]]
    hsnow = [m[10], m[11]]
    sice = [m[12], m[13], m[14], m[15]]
    fmsi2 = mice[0] + mice[1] - ACE1I
    mice, hice, sice = relayer(fmsi2, mice, hice, sice)
    hsnow, hice, sice, mice, snowl = relayer_12(hsnow, hice, sice, mice, snowl)
    snow_n = snowl[0] + snowl[1]
    msi2_n = mice[2] + mice[3]
    hsil_n = jnp.stack([hsnow[0] + hice[0], hsnow[1] + hice[1], hice[2], hice[3]], axis=-1)
    ssil_n = jnp.stack(sice, axis=-1)
    snowi_o = jnp.where(vis, snow_n, snowi)
    msi_o = jnp.where(vis, msi2_n, msi)
    hsi_o = jnp.where(vis[..., None], hsil_n, hsi)
    ssi_o = jnp.where(vis[..., None], ssil_n, ssi)
    hs = ((hsi_o[..., 0] + hsi_o[..., 1]) + hsi_o[..., 2]) + hsi_o[..., 3]
    ss = ((ssi_o[..., 0] + ssi_o[..., 1]) + ssi_o[..., 2]) + ssi_o[..., 3]
    out = dict(musi=musi, husi=husi, susi=susi, mvsi=jnp.where(jnp.asarray(_rowmask_v()), mvsi, 0.0), hvsi=jnp.where(jnp.asarray(_rowmask_v()), hvsi, 0.0),
               svsi=jnp.where(jnp.asarray(_rowmask_v()), svsi, 0.0),
               msicnv=jnp.where(vis, msicnv, 0.0), hsicnv=jnp.where(vis, foc * (rsi * hs - hsicnv0), out_hsicnv),
               fwsim=jnp.where(vis, rsi * (ACE1I + snowi_o + msi_o - ss), 0.0))
    new = dict(rsi=rsi, rsix=rsix, rsiy=rsiy, rsisave=rsisave, msi=msi_o, snowi=snowi_o, hsi=hsi_o, ssi=ssi_o)
    return new, out


def _rowmask_v():
    m = np.zeros((IM, JM), bool)
    m[:, 1:JM - 1] = True
    return m
