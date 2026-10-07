"""D166: ADVSI (sea-ice advection) -- full-fidelity numpy port of ICEDYN_DRV.f:880-1636 (latlon, no tracers, KOCEAN >= 1,
EXPEL_COASTAL_ICEXS defined: the P2SAoM40 build).  Operation order, parenthesisation and the GOTO case structure of the Fortran are
kept; all arithmetic is IEEE double with no fused multiply-add (numpy float64 scalars).

libm: none.  ADVSI itself has no pow/exp/log/sin/cos (``x**2`` is x*x); the helpers it calls (get_snow_ice_layer, relayer,
relayer_12, set_snow_ice_layer) are the validated seaice_core_ff versions, which use only +,-,*,/,sqrt.  The only transcendental
functions are in the one-time grid geometry (sin/cos in ICEDYN.f:1022-1082), which this module does NOT recompute: the five
geometry vectors (DXYP, DYP, DXP, DXV, BYDXYP) are read from the real model's dump (see ``read_dump``) or passed in.

Array convention (like surface_loop.py): 0-based numpy, [i, j(, l)], i = longitude (IM = 72), j = latitude (JM = 46), ice layers
l = 0..3.  Fortran (I, J) = python [I-1, J-1].  MHS = (18, IM, JM): 0:4 ice mass, 4:6 snow mass, 6:10 ice heat, 10:12 snow heat,
12:16 ice salt, 16:18 snow salt (always 0).

Known inputs not computed here: AUSI/AVSI (DYNSI result on the atmosphere grid) and RSISAVE (RSI at DYNSI entry) are inputs;
CONNECT is computed from FOCEAN by ``connect_from_focean`` (ICEDYN_DRV.f:1832-1894).
Limitation: the south-pole row (j = 0) must have FOCEAN = 0 (true for this grid; the Fortran would read the halo row ausi(i,0) otherwise);
this is checked and raises NotImplementedError.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S  # noqa: E402

IM, JM, LMI = 72, 46, 4
NTR = 3 * (LMI + 2)
ACE1I = S.ACE1I
XSI = S.XSI


def connect_from_focean(foc):
    """ICEDYN_DRV.f:1832-1894 (EXPEL_COASTAL_ICEXS, lat-lon): ocean-connectedness code W=1 E=2 S=4 N=8; row 1 = 0, row JM = 15."""
    c = np.zeros((IM, JM))
    for j in range(1, JM - 1):
        for i in range(IM):
            if foc[i, j] > 0.0:
                v = 0
                if foc[i - 1, j] > 0.0:
                    v += 1
                if foc[(i + 1) % IM, j] > 0.0:
                    v += 2
                if foc[i, j - 1] > 0.0:
                    v += 4
                if foc[i, j + 1] > 0.0:
                    v += 8
                c[i, j] = v
    c[:, 0] = 0.0
    c[:, JM - 1] = 15.0
    return c


# ---------------------------------------------------------------------------------------------------- dump reader
_IN = ['focean', 'rsi', 'rsix', 'rsiy', 'rsisave', 'msi', 'snowi']
_OUT2 = ['musi', 'husi', 'susi', 'mvsi', 'hvsi', 'svsi', 'msicnv', 'hsicnv', 'fwsim']


def read_dump(path_in, path_out=None):
    """Read ffadv_in_<it>.bin (and ffadv_out_<it>.bin) written by instrumentation/ICEDYN_DRV_advsi.f.patch (big-endian float64 streams)."""
    a = np.fromfile(path_in, '>f8').astype(np.float64)
    pos = 0

    def take(n, shape, order='F'):
        nonlocal pos
        x = a[pos:pos + n].reshape(shape, order=order)
        pos += n
        return x

    d = {}
    for k in _IN:
        d[k] = take(IM * JM, (IM, JM)).copy()
    # (LMI, IM, JM) Fortran order -> [i, j, l]
    for k in ('hsi', 'ssi'):
        d[k] = np.transpose(take(LMI * IM * JM, (LMI, IM, JM)), (1, 2, 0)).copy()
    for k in ('ausi', 'avsi', 'connect'):
        d[k] = take(IM * JM, (IM, JM)).copy()
    d['geo'] = {k: take(JM, (JM,)).copy() for k in ('dxyp', 'dyp', 'dxp', 'dxv', 'bydxyp')}
    assert pos == a.size, (pos, a.size)
    out = None
    if path_out is not None:
        b = np.fromfile(path_out, '>f8').astype(np.float64)
        pos = 0
        a = b
        out = {}
        for k in ('rsi', 'rsix', 'rsiy', 'rsisave', 'msi', 'snowi'):
            out[k] = take(IM * JM, (IM, JM)).copy()
        for k in ('hsi', 'ssi'):
            out[k] = np.transpose(take(LMI * IM * JM, (LMI, IM, JM)), (1, 2, 0)).copy()
        for k in _OUT2:
            out[k] = take(IM * JM, (IM, JM)).copy()
        assert pos == b.size
    return d, out



# ---------------------------------------------------------------------------------------------------- REAL*16 Ti2b + get_snow_ice_layer
from mpmath import mp, mpf  # noqa: E402
mp.prec = 113           # IEEE binary128 significand
_Q1E10 = mpf(1) / mpf(10 ** 10)   # 1q-10


def ti2b_quad(eit, si, snowl, mice):
    """SEAICE.f:2403 Ti2b (SEAICE_FIXES_2022, seaice_thermo = BP): quad (REAL*16) b, c, det, tm; sub-expressions of two REAL*8 operands
    (mu*Si, MICE/(MICE+SNOWL), frac*lhm, Eit+lhm, shw-shi) are evaluated in double first, exactly as Fortran's typing rules require."""
    if si > 0.0:
        tm = mpf(-S.MU * si)
        frac = mice / (mice + snowl)
        b = (mpf(frac) * tm) * mpf(S.SHW - S.SHI) - mpf(eit + S.LHM)
        c = mpf(frac * S.LHM) * tm
        det = b * b - (mpf(4) * mpf(S.SHI)) * c
        return float((mpf(-0.5) * (b + mp.sqrt(det))) * mpf(S.BYSHI))
    t = (eit + S.LHM) * S.BYSHI
    if mpf(abs(eit + S.LHM)) < _Q1E10:
        t = 0.0
    return t


def get_snow_ice_layer(snow, msi2, hsil, ssil):
    """SEAICE.f:1875 get_snow_ice_layer (needtemp = .false., no tracers) with the REAL*16 Ti2b.  Returns snowl, hsnow, hice, sice, mice."""
    ei, em = S.Ei, S.Em
    msi1 = snow + ACE1I
    mice = [0.0] * LMI; hice = [0.0] * LMI; sice = [0.0] * LMI
    snowl = [0.0, 0.0]; hsnow = [0.0, 0.0]
    if ACE1I > XSI[1] * msi1:
        mice[0] = ACE1I - XSI[1] * msi1
        mice[1] = XSI[1] * msi1
        snowl[0] = snow
        si1 = 1e3 * ssil[0] / mice[0]
        ti1 = ti2b_quad(hsil[0] / (XSI[0] * msi1), si1, snowl[0], mice[0])
        if ti1 < 0.0 and ti1 > -1e-15:
            ti1 = 0.0
        hice[0] = min(max(mice[0] * ei(ti1, si1), hsil[0]), mice[0] * em(si1))
        hsnow[0] = hsil[0] - hice[0]
        if snowl[0] == 0.0 or abs(hsil[0] - hice[0]) < 1e-8:
            hsnow[0] = 0.0
        hsnow[1] = 0.0
        if ti1 < 0 and snowl[0] > 0 and hice[0] != mice[0] * em(si1):
            hsnow[0] = min(hsnow[0], (ti1 * S.SHI - S.LHM) * snowl[0])
        hice[1] = hsil[1]
        sice[0] = ssil[0]; sice[1] = ssil[1]
    else:
        mice[0] = 0.0
        mice[1] = ACE1I
        snowl[0] = XSI[0] * msi1
        snowl[1] = XSI[1] * msi1 - ACE1I
        hsnow[0] = hsil[0]
        si1 = 1e3 * ssil[1] / mice[1]
        ti1 = ti2b_quad(hsil[1] / (XSI[1] * msi1), si1, snowl[1], mice[1])
        if ti1 < 0.0 and ti1 > -1e-15:
            ti1 = 0.0
        hice[0] = 0.0
        hice[1] = min(max(mice[1] * ei(ti1, si1), hsil[1]), mice[1] * em(si1))
        hsnow[1] = hsil[1] - hice[1]
        if snowl[1] == 0.0 or abs(hsil[1] - hice[1]) < 1e-8:
            hsnow[1] = 0.0
        if ti1 < 0 and snowl[1] > 0 and hice[1] != mice[1] * em(si1):
            hsnow[1] = min(hsnow[1], (ti1 * S.SHI - S.LHM) * snowl[1])
        sice[0] = 0.0
        sice[1] = ssil[1]
    for l in range(2, LMI):
        mice[l] = XSI[l] * msi2
        hice[l] = hsil[l]
        sice[l] = ssil[l]
    return snowl, hsnow, hice, sice, mice

# ---------------------------------------------------------------------------------------------------- the port
def _crunch(mhs, amsi, asi, byfoa):
    """label 320/620/350: ice crunches into itself and completely covers the box; excess ice to the lower layers, snow piles up."""
    for k in range(3):
        b = 6 * k
        mhs[b + 0] = amsi[b + 0] / asi
        mhs[b + 1] = amsi[b + 1] / asi
        s = amsi[b + 0] + amsi[b + 1] + amsi[b + 2] + amsi[b + 3]
        dmhsi = s * (byfoa - 1.0 / asi)
        mhs[b + 2] = amsi[b + 2] / asi + XSI[2] * dmhsi
        mhs[b + 3] = amsi[b + 3] / asi + XSI[3] * dmhsi
        mhs[b + 4] = amsi[b + 4] * byfoa
        mhs[b + 5] = amsi[b + 5] * byfoa


def _limit(rsi, rx, ry):
    """The eight sequential 'limit RSIX and RSIY' IFs (labels 310 / 610)."""
    if rsi - rx < 0.0:
        rx = rsi
    if rsi + rx < 0.0:
        rx = -rsi
    if rsi - rx > 1.0:
        rx = rsi - 1.0
    if rsi + rx > 1.0:
        rx = 1.0 - rsi
    if rsi - ry < 0.0:
        ry = rsi
    if rsi + ry < 0.0:
        ry = -rsi
    if rsi - ry > 1.0:
        ry = rsi - 1.0
    if rsi + ry > 1.0:
        ry = 1.0 - rsi
    return rx, ry


def advsi(st, ausi, avsi, focean, geo, connect=None, dts=1800.0, kocean=1, stats=None):
    """One ADVSI call.  st: dict rsi, rsix, rsiy, rsisave, msi, snowi (IM, JM), hsi, ssi (IM, JM, LMI) (state at entry; not modified).
    ausi/avsi: (IM, JM) DYNSI sea-ice velocities on the atmosphere grid (ATMICE%USI/VSI).  geo: dict dxyp, dyp, dxp, dxv, bydxyp (JM,).
    stats: optional dict, filled with branch counters ('ns<label>', 'ew<label>', 'ns_crunch', 'ew_crunch', 'pole_crunch').
    Returns (new state dict, outputs dict musi, husi, susi, mvsi, hvsi, svsi, msicnv, hsicnv, fwsim) (outputs defined as in the Fortran:
    fluxes zero where not computed, msicnv/hsicnv/fwsim only where FOCEAN > 0, zero elsewhere)."""
    if np.any(focean[:, 0] > 0.0):
        raise NotImplementedError('south-pole row has ocean: the Fortran would read the ausi halo row')
    if connect is None:
        connect = connect_from_focean(focean)
    dxyp, dyp, dxp, dxv, bydxyp = (np.asarray(geo[k], float) for k in ('dxyp', 'dyp', 'dxp', 'dxv', 'bydxyp'))
    rsi = st['rsi'].copy(); rsix = st['rsix'].copy(); rsiy = st['rsiy'].copy(); rsisave = st['rsisave'].copy()
    msi = st['msi'].copy(); snowi = st['snowi'].copy(); hsi = st['hsi'].copy(); ssi = st['ssi'].copy()
    foc = focean
    out = {k: np.zeros((IM, JM)) for k in _OUT2}

    # ---- regularise ice concentration gradients (J = 2..JM-1)
    for j in range(1, JM - 1):
        for i in range(IM):
            r = rsi[i, j]
            if r > 1e-4:
                if rsisave[i, j] > r:
                    frsi = (rsisave[i, j] - r) / rsisave[i, j]
                    rsix[i, j] = rsix[i, j] * (1.0 - frsi)
                    rsiy[i, j] = rsiy[i, j] * (1.0 - frsi)
                rx, ry = rsix[i, j], rsiy[i, j]
                if r - rx < 0.0:
                    rx = r
                if r + rx < 0.0:
                    rx = -r
                if r - rx > 1.0:
                    rx = r - 1.0
                if r + rx > 1.0:
                    rx = 1.0 - r
                if r - ry < 0.0:
                    ry = r
                if r + ry < 0.0:
                    ry = -r
                if r - ry > 1.0:
                    ry = r - 1.0
                if r + ry > 1.0:
                    ry = 1.0 - r
                rsix[i, j], rsiy[i, j] = rx, ry
            else:
                rsix[i, j] = 0.0
                rsiy[i, j] = 0.0

    # ---- MHS
    mhs = np.zeros((NTR, IM, JM))
    for j in range(JM):
        for i in range(IM):
            snowl, hsnow, hice, sice, mice = get_snow_ice_layer(
                float(snowi[i, j]), float(msi[i, j]), [float(x) for x in hsi[i, j]], [float(x) for x in ssi[i, j]])
            mhs[0:4, i, j] = mice
            mhs[4:6, i, j] = snowl
            mhs[6:10, i, j] = hice
            mhs[10:12, i, j] = hsnow
            mhs[12:16, i, j] = sice
            mhs[16:18, i, j] = 0.0
    byfoa = np.zeros((IM, JM))
    hsicnv = np.zeros((IM, JM))
    for j in range(JM):
        for i in range(IM):
            if foc[i, j] > 0.0:
                byfoa[i, j] = bydxyp[j] / foc[i, j]
                hsicnv[i, j] = rsi[i, j] * (((hsi[i, j, 0] + hsi[i, j, 1]) + hsi[i, j, 2]) + hsi[i, j, 3])
            else:
                byfoa[i, j] = 0.0
    out['hsicnv'] = hsicnv.copy()    # the Fortran leaves the pre-advection RSI*sum(HSI) in cells the final loop does not visit (pole i > 1)

    # ---- velocities times dt (USIDT, VSIDT), J = 1..JM-1
    usidt = np.zeros((IM, JM)); vsidt = np.zeros((IM, JM))
    for j in range(JM - 1):               # Fortran J = j+1
        jf = j + 1
        if jf != 1 and jf != JM:
            cfx = 1e-3 * 1e-1 * 1e5 / dxp[j]
            cfy = 1e-3 * 1e-1 * 1e5 / dyp[j]
        else:
            cfx = 0.0
            cfy = 0.0
        for i in range(IM):
            ip1 = (i + 1) % IM
            im1 = (i - 1) % IM
            usidt[i, j] = 0.0
            if foc[i, j] > 0.0 and foc[ip1, j] > 0.0 and rsisave[i, j] + rsisave[ip1, j] > 1e-4:
                usidt[i, j] = 0.5 * (ausi[i, j - 1] + ausi[i, j]) * dts
                if connect[i, j] + connect[ip1, j] < 30.0:
                    cxi = int(connect[i, j]) % 2
                    cxip1 = (int(connect[ip1, j]) % 4) // 2
                    du = (msi[i, j] - msi[ip1, j]) * cfx
                    du = min(10.0, max(-10.0, du))
                    if cxi < cxip1:
                        du = max(0.0, du)
                    elif cxi > cxip1:
                        du = min(0.0, du)
                    usidt[i, j] = usidt[i, j] + dts * du
            vsidt[i, j] = 0.0
            if foc[i, j + 1] > 0.0 and foc[i, j] > 0.0 and rsisave[i, j] + rsisave[i, j + 1] > 1e-4:
                vsidt[i, j] = 0.5 * (avsi[im1, j] + avsi[i, j]) * dts
                if connect[i, j] + connect[i, j + 1] < 30.0:
                    cyj = (int(connect[i, j]) // 4) % 2
                    cyjp1 = int(connect[i, j + 1] / 8.0)
                    dv = (msi[i, j] - msi[i, j + 1]) * cfy
                    dv = min(10.0, max(-10.0, dv))
                    if cyj < cyjp1:
                        dv = max(0.0, dv)
                    elif cyj > cyjp1:
                        dv = min(0.0, dv)
                    vsidt[i, j] = vsidt[i, j] + dts * dv
    vsidt[:, JM - 1] = 0.0
    usidt[:, JM - 1] = ausi[0, JM - 1] * dts
    rsisave = rsi.copy()                  # update RSISAVE for diagnostics

    nan = np.nan
    faw = np.full((IM, JM), nan); fasi = np.full((IM, JM), nan); fxsi = np.full((IM, JM), nan); fysi = np.full((IM, JM), nan)
    fmsj = np.full((IM, NTR, JM), nan)

    def _sum(a, lo, hi):
        s = 0.0
        for k in range(lo, hi):
            s = s + a[k]
        return s

    # ---- north-south fluxes (J = 2..JM-2) and near the north pole
    sfasi = 0.0
    sfmsi = np.zeros(NTR)
    for i in range(IM):
        for j in range(1, JM - 2):
            v = vsidt[i, j]
            if v == 0.0:
                continue
            faw[i, j] = v * dxv[j + 1]
            f = faw[i, j]
            if v <= 0.0:
                fasi[i, j] = f * (rsi[i, j + 1] - (1.0 + f * bydxyp[j + 1]) * rsiy[i, j + 1]) * foc[i, j + 1]
                fxsi[i, j] = f * rsix[i, j + 1] * foc[i, j + 1]
                fysi[i, j] = f * (f * bydxyp[j + 1] * f * rsiy[i, j + 1] * foc[i, j + 1] - 3.0 * fasi[i, j])
                fmsj[i, :, j] = fasi[i, j] * mhs[:, i, j + 1]
            else:
                fasi[i, j] = f * (rsi[i, j] + (1.0 - f * bydxyp[j]) * rsiy[i, j]) * foc[i, j]
                fxsi[i, j] = f * rsix[i, j] * foc[i, j]
                fysi[i, j] = f * (f * bydxyp[j] * f * rsiy[i, j] * foc[i, j] - 3.0 * fasi[i, j])
                fmsj[i, :, j] = fasi[i, j] * mhs[:, i, j]
            out['mvsi'][i, j] = _sum(fmsj[i, :, j], 0, LMI + 2)
            out['hvsi'][i, j] = _sum(fmsj[i, :, j], 6, 12)
            out['svsi'][i, j] = _sum(fmsj[i, :, j], 12, 18)
        j = JM - 2                        # Fortran J = JM-1
        v = vsidt[i, j]
        if v == 0.0:
            continue
        faw[i, j] = v * dxv[JM - 1]
        f = faw[i, j]
        if v <= 0.0:
            fasi[i, j] = f * rsi[0, JM - 1] * foc[0, JM - 1]
            fxsi[i, j] = 0.0
            fysi[i, j] = -f * fasi[i, j]
            fmsj[i, :, j] = fasi[i, j] * mhs[:, 0, JM - 1]
        else:
            fasi[i, j] = f * foc[i, j] * (rsi[i, j] + (1.0 - f * bydxyp[j]) * rsiy[i, j])
            fxsi[i, j] = f * rsix[i, j] * foc[i, j]
            fysi[i, j] = f * (f * bydxyp[j] * f * rsiy[i, j] * foc[i, j] - 3.0 * fasi[i, j])
            fmsj[i, :, j] = fasi[i, j] * mhs[:, i, j]
        sfasi = sfasi + fasi[i, j]
        sfmsi = sfmsi + fmsj[i, :, j]
        out['mvsi'][i, j] = _sum(fmsj[i, :, j], 0, LMI + 2)
        out['hvsi'][i, j] = _sum(fmsj[i, :, j], 6, 12)
        out['svsi'][i, j] = _sum(fmsj[i, :, j], 12, 18)

    # ---- update for south-north fluxes (J = 2..JM-1)
    for i in range(IM):
        for j in range(1, JM - 1):
            dx = dxyp[j]
            fo = foc[i, j]
            by = byfoa[i, j]
            bd = bydxyp[j]
            vm = vsidt[i, j - 1]
            vc = vsidt[i, j]
            amsi = None
            asi = 0.0
            crunch = False
            case = None
            if vm < 0.0:
                case = 260 if vc < 0.0 else (250 if vc == 0.0 else 270)
            elif vm == 0.0:
                if vc < 0.0:
                    case = 220
                elif vc == 0.0:
                    continue
                else:
                    case = 230
            else:
                case = 260 if vc != 0.0 else 285
            if stats is not None:
                stats['ns%d' % case] = stats.get('ns%d' % case, 0) + 1
            if case == 220:
                asi = rsi[i, j] * dx * fo - fasi[i, j]
                amsi = rsi[i, j] * dx * mhs[:, i, j] * fo - fmsj[i, :, j]
                if asi > dx * fo:
                    crunch = True
                else:
                    yrsi = (rsiy[i, j] * dx * dx * fo - fysi[i, j] + 3.0 * (faw[i, j] * asi - dx * fasi[i, j])) / (dx - faw[i, j])
                    rsi[i, j] = asi * by
                    if rsi[i, j] > 1.0:
                        rsi[i, j] = 1.0
                    rsiy[i, j] = yrsi * by
                    rsix[i, j] = rsix[i, j] - fxsi[i, j] * by
                    if asi > 0.0:
                        mhs[:, i, j] = amsi / asi
            elif case == 230:
                rsi[i, j] = rsi[i, j] - fasi[i, j] * by
                t = (1.0 - faw[i, j] * bd)
                rsix[i, j] = rsix[i, j] * t
                rsiy[i, j] = rsiy[i, j] * (t * t)
            elif case == 250:
                rsi[i, j] = rsi[i, j] + fasi[i, j - 1] * by
                tmp = (1.0 + faw[i, j - 1] * foc[i, j - 1] * by)
                rsix[i, j] = rsix[i, j] * tmp
                rsiy[i, j] = rsiy[i, j] * (tmp * tmp)
            elif case == 260:
                asi = rsi[i, j] * dx * fo + (fasi[i, j - 1] - fasi[i, j])
                amsi = rsi[i, j] * dx * mhs[:, i, j] * fo + (fmsj[i, :, j - 1] - fmsj[i, :, j])
                if asi > dx * fo:
                    crunch = True
                else:
                    yrsi = (rsiy[i, j] * dx * dx * fo + (fysi[i, j - 1] - fysi[i, j]) + 3.0 * ((faw[i, j - 1] + faw[i, j]) * asi
                            - dx * (fasi[i, j - 1] + fasi[i, j]))) / (dx + (faw[i, j - 1] - faw[i, j]))
                    rsi[i, j] = asi * by
                    if rsi[i, j] > 1.0:
                        rsi[i, j] = 1.0
                    rsiy[i, j] = yrsi * by
                    rsix[i, j] = rsix[i, j] + (fxsi[i, j - 1] - fxsi[i, j]) * by
                    if asi > 0.0:
                        mhs[:, i, j] = amsi / asi
            elif case == 270:
                rsi[i, j] = rsi[i, j] + (fasi[i, j - 1] - fasi[i, j]) * by
                t = (1.0 + (faw[i, j - 1] * foc[i, j - 1] - faw[i, j] * foc[i, j]) * by)
                rsix[i, j] = rsix[i, j] * t
                rsiy[i, j] = rsiy[i, j] * (t * t)
            else:  # 285
                asi = rsi[i, j] * dx * fo + fasi[i, j - 1]
                amsi = rsi[i, j] * dx * mhs[:, i, j] * fo + fmsj[i, :, j - 1]
                if asi > dx * fo:
                    crunch = True
                else:
                    yrsi = (rsiy[i, j] * dx * dx * fo + fysi[i, j - 1] + 3.0 * (faw[i, j - 1] * asi - dx * fasi[i, j - 1])) / (dx + faw[i, j - 1])
                    rsi[i, j] = asi * by
                    rsiy[i, j] = yrsi * by
                    rsix[i, j] = rsix[i, j] + fxsi[i, j - 1] * by
                    if asi > 0.0:
                        mhs[:, i, j] = amsi / asi
            if crunch:
                if stats is not None:
                    stats['ns_crunch'] = stats.get('ns_crunch', 0) + 1
                rsi[i, j] = 1.0
                rsix[i, j] = 0.0
                rsiy[i, j] = 0.0
                col = mhs[:, i, j].copy()
                _crunch(col, amsi, asi, by)
                mhs[:, i, j] = col
            else:
                rsi[i, j] = max(0.0, rsi[i, j])
                rsix[i, j], rsiy[i, j] = _limit(rsi[i, j], rsix[i, j], rsiy[i, j])

    # ---- north pole box
    j = JM - 1
    asi = rsi[0, j] * dxyp[j] * foc[0, j] + sfasi / IM
    amsi = rsi[0, j] * dxyp[j] * mhs[:, 0, j] * foc[0, j] + sfmsi / IM
    if asi > dxyp[j] * foc[0, j]:
        if stats is not None:
            stats['pole_crunch'] = stats.get('pole_crunch', 0) + 1
        rsi[0, j] = 1.0
        col = mhs[:, 0, j].copy()
        _crunch(col, amsi, asi, byfoa[0, j])
        mhs[:, 0, j] = col
    else:
        rsi[0, j] = asi * byfoa[0, j]
        if asi > 0.0:
            mhs[:, 0, j] = amsi / asi

    # ---- east-west advection (J = 2..JM-1)
    for j in range(1, JM - 1):
        dx = dxyp[j]
        bd = bydxyp[j]
        fmsi = np.full((NTR, IM), nan)
        for i in range(IM):
            ip1 = (i + 1) % IM
            u = usidt[i, j]
            if u == 0.0:
                continue
            faw[i, j] = u * dyp[j]
            f = faw[i, j]
            if u <= 0.0:
                fasi[i, j] = f * (rsi[ip1, j] - (1.0 + f * bd) * rsix[ip1, j]) * foc[ip1, j]
                fxsi[i, j] = f * (f * bd * f * rsix[ip1, j] * foc[ip1, j] - 3.0 * fasi[i, j])
                fysi[i, j] = f * rsiy[ip1, j] * foc[ip1, j]
                fmsi[:, i] = fasi[i, j] * mhs[:, ip1, j]
            else:
                fasi[i, j] = f * (rsi[i, j] + (1.0 - f * bd) * rsix[i, j]) * foc[i, j]
                fxsi[i, j] = f * (f * bd * f * rsix[i, j] * foc[i, j] - 3.0 * fasi[i, j])
                fysi[i, j] = f * rsiy[i, j] * foc[i, j]
                fmsi[:, i] = fasi[i, j] * mhs[:, i, j]
            out['musi'][i, j] = _sum(fmsi[:, i], 0, LMI + 2)
            out['husi'][i, j] = _sum(fmsi[:, i], 6, 12)
            out['susi'][i, j] = _sum(fmsi[:, i], 12, 18)
        for i in range(IM):
            im1 = (i - 1) % IM
            fo = foc[i, j]
            by = byfoa[i, j]
            um = usidt[im1, j]
            uc = usidt[i, j]
            if um < 0.0:
                case = 560 if uc < 0.0 else (550 if uc == 0.0 else 570)
            elif um == 0.0:
                if uc < 0.0:
                    case = 520
                elif uc == 0.0:
                    continue
                else:
                    case = 530
            else:
                case = 560 if uc != 0.0 else 585
            crunch = False
            asi = 0.0
            amsi = None
            if stats is not None:
                stats['ew%d' % case] = stats.get('ew%d' % case, 0) + 1
            if case == 520:
                asi = rsi[i, j] * dx * fo - fasi[i, j]
                amsi = rsi[i, j] * dx * mhs[:, i, j] * fo - fmsi[:, i]
                if asi > dx * fo:
                    crunch = True
                else:
                    xrsi = (rsix[i, j] * dx * dx * fo - fxsi[i, j] + 3.0 * (faw[i, j] * asi - dx * fasi[i, j])) / (dx - faw[i, j])
                    rsi[i, j] = asi * by
                    rsix[i, j] = xrsi * by
                    rsiy[i, j] = rsiy[i, j] - fysi[i, j] * by
                    if asi > 0.0:
                        mhs[:, i, j] = amsi / asi
            elif case == 530:
                rsi[i, j] = rsi[i, j] - fasi[i, j] * by
                t = (1.0 - faw[i, j] * bd)
                rsix[i, j] = rsix[i, j] * (t * t)
                rsiy[i, j] = rsiy[i, j] * t
            elif case == 550:
                rsi[i, j] = rsi[i, j] + fasi[im1, j] * by
                t = (1.0 + faw[im1, j] * foc[im1, j] * by)
                rsix[i, j] = rsix[i, j] * (t * t)
                rsiy[i, j] = rsiy[i, j] * t
            elif case == 560:
                asi = rsi[i, j] * dx * fo + (fasi[im1, j] - fasi[i, j])
                amsi = rsi[i, j] * dx * mhs[:, i, j] * fo + (fmsi[:, im1] - fmsi[:, i])
                if asi > dx * fo:
                    crunch = True
                else:
                    xrsi = (rsix[i, j] * dx * dx * fo + (fxsi[im1, j] - fxsi[i, j]) + 3.0 * ((faw[im1, j] + faw[i, j]) * asi
                            - dx * (fasi[im1, j] + fasi[i, j]))) / (dx + (faw[im1, j] - faw[i, j]))
                    rsi[i, j] = asi * by
                    rsix[i, j] = xrsi * by
                    rsiy[i, j] = rsiy[i, j] + (fysi[im1, j] - fysi[i, j]) * by
                    if asi > 0.0:
                        mhs[:, i, j] = amsi / asi
            elif case == 570:
                rsi[i, j] = rsi[i, j] + (fasi[im1, j] - fasi[i, j]) * by
                t = (1.0 + (faw[im1, j] * foc[im1, j] - faw[i, j] * foc[i, j]) * by)
                rsix[i, j] = rsix[i, j] * (t * t)
                rsiy[i, j] = rsiy[i, j] * t
            else:  # 585
                asi = rsi[i, j] * dx * fo + fasi[im1, j]
                amsi = rsi[i, j] * dx * mhs[:, i, j] * fo + fmsi[:, im1]
                if asi > dx * fo:
                    crunch = True
                else:
                    xrsi = (rsix[i, j] * dx * dx * fo + fxsi[im1, j] + 3.0 * (faw[im1, j] * asi - dx * fasi[im1, j])) / (dx + faw[im1, j])
                    rsi[i, j] = asi * by
                    rsix[i, j] = xrsi * by
                    rsiy[i, j] = rsiy[i, j] + fysi[im1, j] * by
                    if asi > 0.0:
                        mhs[:, i, j] = amsi / asi
            if crunch:
                if stats is not None:
                    stats['ew_crunch'] = stats.get('ew_crunch', 0) + 1
                rsi[i, j] = 1.0
                rsix[i, j] = 0.0
                rsiy[i, j] = 0.0
                col = mhs[:, i, j].copy()
                _crunch(col, amsi, asi, by)
                mhs[:, i, j] = col
            else:
                rsi[i, j] = max(0.0, rsi[i, j])
                rsix[i, j], rsiy[i, j] = _limit(rsi[i, j], rsix[i, j], rsiy[i, j])
                if rsi[i, j] > 1.0:
                    rsi[i, j] = 1.0

    # ---- back to the thermal layers (KOCEAN >= 1)
    if kocean < 1:
        raise NotImplementedError('fixed-SST branch not ported')
    for j in range(JM):
        imaxj = 1 if j in (0, JM - 1) else IM
        for i in range(imaxj):
            if foc[i, j] > 0.0:
                m = mhs[:, i, j]
                s_all = _sum(m, 0, LMI + 2)
                s_salt = _sum(m, 12, 16)
                sold = ((ssi[i, j, 0] + ssi[i, j, 1]) + ssi[i, j, 2]) + ssi[i, j, 3]
                out['msicnv'][i, j] = rsi[i, j] * (s_all - s_salt) - rsisave[i, j] * (ACE1I + snowi[i, j] + msi[i, j] - sold)
                mice = [float(x) for x in m[0:4]]
                snowl = [float(x) for x in m[4:6]]
                hice = [float(x) for x in m[6:10]]
                hsnow = [float(x) for x in m[10:12]]
                sice = [float(x) for x in m[12:16]]
                fmsi2 = mice[0] + mice[1] - ACE1I
                S.relayer(fmsi2, mice, hice, sice)
                S.relayer_12(hsnow, hice, sice, mice, snowl)
                snow, _msi1, msi2, hsil, ssil = S.set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
                snowi[i, j] = snow
                msi[i, j] = msi2
                hsi[i, j, :] = hsil
                ssi[i, j, :] = ssil
                hs = ((hsi[i, j, 0] + hsi[i, j, 1]) + hsi[i, j, 2]) + hsi[i, j, 3]
                out['hsicnv'][i, j] = foc[i, j] * (rsi[i, j] * hs - hsicnv[i, j])
                ss = ((ssi[i, j, 0] + ssi[i, j, 1]) + ssi[i, j, 2]) + ssi[i, j, 3]
                out['fwsim'][i, j] = rsi[i, j] * (ACE1I + snowi[i, j] + msi[i, j] - ss)
    new = dict(rsi=rsi, rsix=rsix, rsiy=rsiy, rsisave=rsisave, msi=msi, snowi=snowi, hsi=hsi, ssi=ssi)
    return new, out
