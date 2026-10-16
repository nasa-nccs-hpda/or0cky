"""D95: SDRAG (default, non-linear branch, ANG_SDRAG=1) of ATMDYN.f ported to numpy.

Fortran: SDRAG, model/ATMDYN.f 1993-2141 (linear_sdrag/`rtau` branch 2038-2058 is dead for this
rundeck and NOT ported; UNRDRAG is dead; the AJL / DIAGCD diagnostic accumulations are not ported).
Parameters of the run (recorded in ffd_sdrag_consts.bin): LS1=24, LSDRAG=LPSDRAG=37, X_SDRAG=(.002,.0002),
Wc_JDRAG=30, WMAX=200, ANG_SDRAG=1, CSDRAGL(LS1:LM), VSDRAGL(LS1:LM), RGAS (planet parameter).

SDRAG acts independently on every velocity column (I,J), J=2..JM, so the port works on a set of
columns: arrays of shape (R, LM) (levels 1..LM along axis 1) plus per-column latitude index J.
Operation order follows the Fortran statements (left-to-right, `**2` as x*x, the single-precision
literals .15/100./373. compare exactly as in Fortran: .15 is the float32 value only in the cosv test,
which is why that test is written with the float32 constant below).
"""
import numpy as np

LM = 40
F32_015 = float(np.float32(.15))   # `COSV(J).LE..15` compares real*8 with the REAL*4 literal .15


def sdrag_columns(u, v, t, pk, pedn1, ma_ip1_jm1, ma_i_jm1, ma_ip1_j, ma_i_j, jcol, dt1, p, geo):
    """u,v,t,pk,pedn1(=PEDN(L+1)),ma_*: (R, LM).  jcol: Fortran J of each column (R,).  p: parameter dict
    (ls1,lsdrag,lpsdrag,ang_sdrag,wc_jdrag,wmax,x_sdrag(2),csdragl(LM),vsdragl(LM),rgas).
    geo: cosv, rapvn, rapvs, dxyv, dxyn, dxys (JM arrays, index J-1).  Returns (u_new, v_new, extra)."""
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    R = u.shape[0]
    jcol = np.asarray(jcol, dtype=int)
    cosv = geo['cosv'][jcol - 1]
    rapvn = geo['rapvn'][jcol - 2]          # RAPVN(J-1)
    rapvs = geo['rapvs'][jcol - 1]
    dxyv = geo['dxyv'][jcol - 1]
    dxyn = geo['dxyn'][jcol - 2]            # DXYN(J-1)
    dxys = geo['dxys'][jcol - 1]
    ls1, lsdrag, lpsdrag = p['ls1'], p['lsdrag'], p['lpsdrag']
    wmax = p['wmax']
    wmaxp = wmax * 3. / 4.
    polar = cosv <= F32_015
    wmaxj = np.where(polar, wmaxp, wmax)
    ang = np.zeros(R)
    nclamp = 0
    for L in range(ls1, LM + 1):            # Fortran L
        l = L - 1
        cd_lin = (L >= lsdrag) | ((L >= lpsdrag) & polar)
        tl = t[:, l] * pk[:, l]
        if np.any((tl < 100.) | (tl > 373.)):
            raise RuntimeError("SDRAG: T outside 100-373 K (Fortran calls stop_model)")
        rho = 100. * pedn1[:, l] / (p['rgas'] * tl)
        wl = np.sqrt(u[:, l] * u[:, l] + v[:, l] * v[:, l])
        xjud = np.ones(R)
        if p['wc_jdrag'] > 0.:
            q = p['wc_jdrag'] / (p['wc_jdrag'] + np.minimum(wl, wmaxj))
            xjud = q * q
        cdn = p['csdragl'][l] * xjud
        cdn_lin = (p['x_sdrag'][0] + p['x_sdrag'][1] * np.minimum(wl, wmaxj)) * xjud
        cdn = np.where(cd_lin, cdn_lin, cdn)
        mauv = (ma_ip1_jm1[:, l] + ma_i_jm1[:, l]) * rapvn + (ma_ip1_j[:, l] + ma_i_j[:, l]) * rapvs
        x = dt1 * rho * cdn * np.minimum(wl, wmaxj) * p['vsdragl'][l] / mauv
        over = wl > wmaxj
        nclamp += int(over.sum())
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = 1. - (1. - x) * wmaxj / wl
        x = np.where(over, xc, x)
        dut = -x * mauv * dxyv * u[:, l]
        ang = ang - dut
        u[:, l] = u[:, l] * (1. - x)
        v[:, l] = v[:, l] * (1. - x)
    if p['ang_sdrag'] > 0:
        lmax = ls1 - 1
        if p['ang_sdrag'] > 1:
            lmax = LM
        mm = np.zeros((R, lmax))
        s = np.zeros(R)
        for l in range(lmax):
            mm[:, l] = .5 * ((ma_ip1_jm1[:, l] + ma_i_jm1[:, l]) * dxyn + (ma_ip1_j[:, l] + ma_i_j[:, l]) * dxys)
            s = s + mm[:, l]
        du = ang / s
        for l in range(lmax):
            u[:, l] = u[:, l] + du
    return u, v, dict(nclamp=nclamp, ang=ang)
