"""Reference transcription (plain Python, float64) of the ModelE sea-ice ground thermodynamics
(SEAICE.f `SEA_ICE`, `SSIDEC`, `snowice`, `TICE` and their helpers) -- Track B.

This is GROUND_SI's core: given the surface fluxes SURFACE.f already computed (F0DT, F1DT, EVAP,
solar), it diffuses heat through the 4 thermal layers (LMI=4: layer 1/2 = snow+first-year ice,
layer 3/4 = deeper ice), handles melt/freeze, brine drainage (SSIDEC) and snow-to-ice conversion
(snowice). `seaice_thermo="BP"` (brine-pocket formulation, the P2SAoM40 default, not overridden by
the rundeck) throughout.

Known, documented approximation: the real Fortran computes `Ti`/`Ti2b` (temperature from enthalpy) in
REAL*16 (quad precision) internally, only the final REAL*8 result is returned (SEAICE_FIXES_2022 is
always defined in this build). This port uses float64 (REAL*8) throughout; validated empirically
against real dumps rather than assumed exact -- see FULL_FIDELITY_DELTAS.md.
"""
import math

LHM = 3.34e5
SHW = 4185.0
SHI = 2060.0
BYSHI = 1.0 / SHI
BYLHM = 1.0 / LHM
MU = 0.054
RHOI = 916.6
RHOW = 1000.0
RHOWS = 1030.0
RHOS = 300.0
LMI = 4
XSI = (0.5, 0.5, 0.5, 0.5)
ACE1I = 0.1 * RHOI
AC2OIM = 0.1 * RHOI
ALPHA = 1.0
SSI0 = 0.0032
FSSS = 8.0 / 35.0
SSIMIN = 1e-6


def tfrez(sss):
    return (-.0575 + (-2.154996e-4) * sss) * sss + 1.710523e-3 * sss * math.sqrt(sss)


def Ti(Eit, Si):
    if Si > 1e-10:
        tm = -MU * Si
        if Eit >= SHW * tm:
            return tm
        b = tm * (SHW - SHI) - (Eit + LHM)
        c = LHM * tm
        det = b * b - 4.0 * SHI * c
        return -0.5 * (b + math.sqrt(det)) * BYSHI
    else:
        t = (Eit + LHM) * BYSHI
        return 0.0 if abs(Eit + LHM) < 1e-10 else t


def Ti2b(Eit, Si, snowl, mice):
    if Si > 1e-10:
        tm = -MU * Si
        frac = mice / (mice + snowl)
        b = frac * tm * (SHW - SHI) - (Eit + LHM)
        c = frac * LHM * tm
        det = b * b - 4.0 * SHI * c
        return -0.5 * (b + math.sqrt(det)) * BYSHI
    else:
        t = (Eit + LHM) * BYSHI
        return 0.0 if abs(Eit + LHM) < 1e-10 else t


def Ei(t, si):
    if si > 0.0 and t != 0.0:
        return (t + MU * si) * SHI - LHM * (1.0 + MU * si / t) - SHW * MU * si
    return t * SHI - LHM


def dEidTi(t, si):
    if si < 1e-10:
        return SHI
    return SHI if t == 0.0 else SHI + LHM * MU * si / (t * t)


def Mi(hsi, ssi, msi):
    if 1e3 * ssi / msi > 1e-10:
        if hsi + SHW * MU * 1e3 * ssi > 0.0 or abs(hsi + SHW * MU * 1e3 * ssi) < 1e-12:
            return msi
        return 0.0
    return max(0.0, msi + hsi * BYLHM)


def Em(si):
    return -MU * si * SHW


def solar_ice_frac_full(snow, msi2, wetsnow):
    """SEAICE.f solar_ice_frac extended to all LMI=4 layers (SEA_ICE calls it with lmax=LMI, not 2)."""
    kiextvis, kiextnir1 = 1.5, 18.0
    dsnow = snow / RHOS
    hice12 = ACE1I / RHOI
    deep = dsnow > 0.02
    if deep:
        fracvis, fracnir1, ksextvis, ksextnir1 = ((0.20, 0.33, 10.7, 118.0) if wetsnow
                                                   else (0.06, 0.31, 19.6, 196.0))
    else:
        fracvis, fracnir1, ksextvis, ksextnir1 = 0.24, 0.43, 10.7, 118.0
    fsri = [0.0] * LMI
    if ACE1I * XSI[0] > snow * XSI[1]:
        hice1 = (ACE1I - XSI[1] * (snow + ACE1I)) / RHOI
        fv1 = math.exp(-ksextvis * dsnow - kiextvis * hice1)
        fn1 = math.exp(-ksextnir1 * dsnow - kiextnir1 * hice1)
    else:
        dsnow1 = (ACE1I + snow) * XSI[0] / RHOS
        fv1 = math.exp(-ksextvis * dsnow1)
        fn1 = math.exp(-ksextnir1 * dsnow1)
    fsri[0] = fracvis * fv1 + fracnir1 * fn1
    fv2 = math.exp(-ksextvis * dsnow - kiextvis * hice12)
    fn2 = math.exp(-ksextnir1 * dsnow - kiextnir1 * hice12)
    fsri[1] = fracvis * fv2 + fracnir1 * fn2
    fvp, fnp = fv2, fn2
    for l in range(2, LMI):
        hicel = XSI[l] * msi2 / RHOI
        fvp = math.exp(-kiextvis * hicel) * fvp
        fnp = math.exp(-kiextnir1 * hicel) * fnp
        fsri[l] = fracvis * fvp + fracnir1 * fnp
    return fsri


def get_snow_ice_layer(snow, msi2, hsil, ssil, needtemp):
    """Returns snowl(2), hsnow(2), hice(LMI), sice(LMI), tsnw(2), tsil(LMI), mice(LMI)."""
    msi1 = snow + ACE1I
    mice = [0.0] * LMI
    hice = [0.0] * LMI
    sice = [0.0] * LMI
    snowl = [0.0, 0.0]
    hsnow = [0.0, 0.0]
    if ACE1I > XSI[1] * msi1:
        mice[0] = ACE1I - XSI[1] * msi1
        mice[1] = XSI[1] * msi1
        snowl[0] = snow
        snowl[1] = 0.0
        si1 = 1e3 * ssil[0] / mice[0]
        ti1 = Ti2b(hsil[0] / (XSI[0] * msi1), si1, snowl[0], mice[0])
        if -1e-15 < ti1 < 0.0:
            ti1 = 0.0
        hice[0] = min(max(mice[0] * Ei(ti1, si1), hsil[0]), mice[0] * Em(si1))
        hsnow[0] = hsil[0] - hice[0]
        if snowl[0] == 0.0 or abs(hsil[0] - hice[0]) < 1e-8:
            hsnow[0] = 0.0
        hsnow[1] = 0.0
        if ti1 < 0 and snowl[0] > 0 and hice[0] != mice[0] * Em(si1):
            hsnow[0] = min(hsnow[0], (ti1 * SHI - LHM) * snowl[0])
        hice[1] = hsil[1]
        sice[0] = ssil[0]
        sice[1] = ssil[1]
    else:
        mice[0] = 0.0
        mice[1] = ACE1I
        snowl[0] = XSI[0] * msi1
        snowl[1] = XSI[1] * msi1 - ACE1I
        hsnow[0] = hsil[0]
        si1 = 1e3 * ssil[1] / mice[1]
        ti1 = Ti2b(hsil[1] / (XSI[1] * msi1), si1, snowl[1], mice[1])
        if -1e-15 < ti1 < 0.0:
            ti1 = 0.0
        hice[0] = 0.0
        hice[1] = min(max(mice[1] * Ei(ti1, si1), hsil[1]), mice[1] * Em(si1))
        hsnow[1] = hsil[1] - hice[1]
        if snowl[1] == 0.0 or abs(hsil[1] - hice[1]) < 1e-8:
            hsnow[1] = 0.0
        if ti1 < 0 and snowl[1] > 0 and hice[1] != mice[1] * Em(si1):
            hsnow[1] = min(hsnow[1], (ti1 * SHI - LHM) * snowl[1])
        sice[0] = 0.0
        sice[1] = ssil[1]
    for l in range(2, LMI):
        mice[l] = XSI[l] * msi2
        hice[l] = hsil[l]
        sice[l] = ssil[l]
    tsnw = [0.0, 0.0]
    tsil = [0.0] * LMI
    if needtemp:
        for l in range(2):
            tsnw[l] = Ti(hsnow[l] / snowl[l], 0.0) if snowl[l] > 0 else 0.0
        for l in range(LMI):
            tsil[l] = Ti(hice[l] / mice[l], 1e3 * sice[l] / mice[l]) if mice[l] > 0 else 0.0
    return snowl, hsnow, hice, sice, tsnw, tsil, mice


def set_snow_ice_layer(hsnow, hice, sice, mice, snowl):
    snow = snowl[0] + snowl[1]
    msi1 = snow + ACE1I
    msi2 = sum(mice[2:LMI])
    hsil = [0.0] * LMI
    ssil = [0.0] * LMI
    hsil[0] = hsnow[0] + hice[0]; ssil[0] = sice[0]
    hsil[1] = hsnow[1] + hice[1]; ssil[1] = sice[1]
    for l in range(2, LMI):
        hsil[l] = hice[l]; ssil[l] = sice[l]
    return snow, msi1, msi2, hsil, ssil


def relayer(fmsi2, mice, hice, sice):
    """In place update of mice/hice/sice (lists), matching SEAICE.f `relayer`."""
    msi2 = sum(mice[2:LMI])
    fmsi = [0.0] * LMI; fhsi = [0.0] * LMI; fssi = [0.0] * LMI
    for l in range(1, LMI - 1):    # Fortran L=2..LMI-1, 0-based l=1..LMI-2
        fmsi[l] = sum(XSI[l + 1:LMI]) * (msi2 + fmsi2) - sum(mice[l + 1:LMI])
        if fmsi[l] > 0:
            if fmsi[l] > mice[l]:
                fhsi[l] = hice[l] + (fmsi[l] - mice[l]) * hice[l - 1] / mice[l - 1]
                fssi[l] = sice[l] + (fmsi[l] - mice[l]) * sice[l - 1] / mice[l - 1]
            else:
                fhsi[l] = hice[l] * fmsi[l] / mice[l]
                fssi[l] = sice[l] * fmsi[l] / mice[l]
        else:
            fhsi[l] = hice[l + 1] * fmsi[l] / mice[l + 1]
            fssi[l] = sice[l + 1] * fmsi[l] / mice[l + 1]
    for l in range(1, LMI):        # Fortran L=2..LMI, 0-based l=1..LMI-1
        if l > 1:
            mice[l] += fmsi[l - 1]; hice[l] += fhsi[l - 1]; sice[l] += fssi[l - 1]
        if l < LMI - 1:
            mice[l] -= fmsi[l]; hice[l] -= fhsi[l]; sice[l] -= fssi[l]


def relayer_12(hsnow, hice, sice, mice, snowl):
    """In place update, matching SEAICE.f `relayer_12` (no tracers)."""
    fmsi1 = snowl[0] + mice[0] - XSI[0] * (snowl[0] + snowl[1] + ACE1I)
    if abs(fmsi1) < 1e-14:
        fmsi1 = 0.0
    fmsi0 = XSI[0] * (snowl[0] + snowl[1] + ACE1I)
    if snowl[1] > 0.0 and mice[0] + mice[1] > fmsi0:
        if fmsi0 > mice[1]:
            fssi1 = (fmsi0 - mice[1]) * sice[0] / mice[0]
            fhsi1 = (fmsi0 - mice[1]) * hice[0] / mice[0]
            sice[0] -= fssi1; sice[1] += fssi1
            hice[0] -= fhsi1; hice[1] += fhsi1
        else:
            fssi1 = (mice[1] - fmsi0) * sice[1] / mice[1]
            fhsi1 = (mice[1] - fmsi0) * hice[1] / mice[1]
            sice[0] += fssi1; sice[1] -= fssi1
            hice[0] += fhsi1; hice[1] -= fhsi1
        mice[0] = mice[0] + mice[1] - fmsi0
        mice[1] = fmsi0
        snowl[0] = snowl[0] + snowl[1]; snowl[1] = 0.0
        hsnow[0] = hsnow[0] + hsnow[1]; hsnow[1] = 0.0
    elif mice[0] > 0.0 and mice[0] + mice[1] < fmsi0:
        if fmsi0 < snowl[0]:
            fhsi1 = (snowl[0] - fmsi0) * hsnow[0] / snowl[0]
            hsnow[1] += fhsi1; hsnow[0] -= fhsi1
        else:
            fhsi1 = (fmsi0 - snowl[0]) * hsnow[1] / snowl[1]
            hsnow[1] -= fhsi1; hsnow[0] += fhsi1
        snowl[1] = snowl[0] + snowl[1] - fmsi0
        snowl[0] = fmsi0
        hice[1] += hice[0]; hice[0] = 0.0
        sice[1] += sice[0]; sice[0] = 0.0
        mice[1] += mice[0]; mice[0] = 0.0
    else:
        if fmsi1 > 0:
            if mice[0] > 0:
                if fmsi1 > mice[0]:
                    hsnow[1] += (fmsi1 - mice[0]) * hsnow[0] / snowl[0]
                    hsnow[0] -= (fmsi1 - mice[0]) * hsnow[0] / snowl[0]
                    snowl[1] += (fmsi1 - mice[0]); snowl[0] -= (fmsi1 - mice[0])
                    hice[1] += hice[0]; hice[0] = 0.0
                    sice[1] += sice[0]; sice[0] = 0.0
                    mice[1] += mice[0]; mice[0] = 0.0
                else:
                    fhsi1 = fmsi1 * hice[0] / mice[0]
                    fssi1 = fmsi1 * sice[0] / mice[0]
                    mice[0] -= fmsi1; mice[1] += fmsi1
                    hice[0] -= fhsi1; hice[1] += fhsi1
                    sice[0] -= fssi1; sice[1] += fssi1
            else:
                fhsi1 = fmsi1 * hsnow[0] / snowl[0]
                snowl[0] -= fmsi1; snowl[1] += fmsi1
                hsnow[0] -= fhsi1; hsnow[1] += fhsi1
        else:
            if mice[0] > 0:
                fhsi1 = fmsi1 * hice[1] / mice[1]
                fssi1 = fmsi1 * sice[1] / mice[1]
                mice[0] -= fmsi1; mice[1] += fmsi1
                hice[0] -= fhsi1; hice[1] += fhsi1
                sice[0] -= fssi1; sice[1] += fssi1
            else:
                if snowl[1] + fmsi1 < 0:
                    hice[0] = -(snowl[1] + fmsi1) * hice[1] / mice[1]
                    hice[1] -= hice[0]
                    sice[0] = -(snowl[1] + fmsi1) * sice[1] / mice[1]
                    sice[1] -= sice[0]
                    mice[0] = -(snowl[1] + fmsi1)
                    mice[1] -= mice[0]
                    hsnow[0] += hsnow[1]; hsnow[1] = 0.0
                    snowl[0] += snowl[1]; snowl[1] = 0.0
                else:
                    fhsi1 = fmsi1 * hsnow[1] / snowl[1]
                    snowl[0] -= fmsi1; snowl[1] += fmsi1
                    hsnow[0] -= fhsi1; hsnow[1] += fhsi1


def tice(hsil, ssil, msi1, msi2):
    if ACE1I > XSI[1] * msi1:
        mice = [ACE1I - XSI[1] * msi1, XSI[1] * msi1]
        snowl = [msi1 - ACE1I, 0.0]
    else:
        mice = [0.0, ACE1I]
        snowl = [XSI[0] * msi1, XSI[1] * msi1 - ACE1I]
    tsil = [0.0] * LMI
    if mice[0] != 0.0:
        tsil[0] = Ti2b(hsil[0] / (XSI[0] * msi1), 1e3 * ssil[0] / mice[0], snowl[0], mice[0])
    else:
        tsil[0] = Ti(hsil[0] / (XSI[0] * msi1), 0.0)
    tsil[1] = Ti2b(hsil[1] / (XSI[1] * msi1), 1e3 * ssil[1] / mice[1], snowl[1], mice[1])
    for l in range(2, LMI):
        tsil[l] = Ti(hsil[l] / (XSI[l] * msi2), 1e3 * ssil[l] / (XSI[l] * msi2))
    return tsil


def alami(t, si):
    """SEAICE.f Alami (thermal diffusion of ice, Pringle et al 2007)."""
    alami0, alamdt, alamds = 2.11, -0.011, 0.09
    if si < 1e-10:
        return alami0
    if t != 0.0:
        a = alami0 + alamdt * t + alamds * si / t
        return a if a > 0.0 else alami0
    return alami0


def sea_ice(dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow):
    """SEA_ICE. Returns dict: snow, hsil, ssil, msi2, srox2, run, erun, srun, wetsnow, melt12, cmprs."""
    hsil = list(hsil); ssil = list(ssil)
    msi1 = snow + ACE1I
    fsri = solar_ice_frac_full(snow, msi2, wetsnow) if srox0 > 0 else [0.0] * LMI
    srox2 = srox0 * fsri[LMI - 1]
    f = [0.0] * LMI
    f[0] = f1dt
    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, True)
    for l in range(1, LMI - 1):
        hc = dEidTi(tsil[l], 1e3 * (sice[l] / mice[l])) * mice[l]
        tavg = (tsil[l] * mice[l + 1] + tsil[l + 1] * mice[l]) / (mice[l + 1] + mice[l])
        savg = 1e3 * (sice[l] * mice[l + 1] / mice[l] + sice[l + 1] * mice[l] / mice[l + 1]) / (mice[l + 1] + mice[l])
        alam = alami(tavg, savg)
        dfdti = 2.0 * alam * RHOI * dtsrce / (mice[l] + mice[l + 1])
        f[l] = (dfdti * (hc * (tsil[l] - tsil[l + 1]) + ALPHA * f[l - 1]) + hc * srox0 * fsri[l]) / (hc + ALPHA * dfdti)
    hsil[0] += f0dt - f1dt
    hsil[1] += f1dt
    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, False)
    for l in range(1, LMI - 1):
        hice[l] -= f[l]; hice[l + 1] += f[l]
    hice[LMI - 1] -= srox2 + fhoc
    sice[LMI - 1] -= fsoc
    mice[LMI - 1] -= fmoc
    dew = -evap
    dewi = [max(0.0, dew), 0.0]
    dews = dew - dewi[0]
    if snowl[0] + dews <= 0:
        dews = -snowl[0]
        dewi[0] = dew - dews
        mice[0] += dewi[0]
    else:
        if mice[0] == 0.0:
            dewi[1] = dewi[0]; dewi[0] = 0.0
            mice[1] += dewi[1]
        else:
            mice[0] += dewi[0]
    melts = max(0.0, hsnow[0] * BYLHM + snowl[0] + dews)
    melts2 = max(0.0, hsnow[1] * BYLHM + snowl[1])
    melti = [0.0] * LMI
    smelti = [0.0] * LMI
    hmelti = [0.0] * LMI
    for l in range(LMI):
        if mice[l] > 0:
            mi = Mi(hice[l], sice[l], mice[l])
            smelti[l] = mi * sice[l] / mice[l]
            hmelti[l] = mi * Em(1e3 * sice[l] / mice[l])
            melti[l] = mi
            mice[l] -= mi
            sice[l] = 0.0 if (mice[l] == 0.0 and abs(sice[l] - smelti[l]) < 1e-12) else sice[l] - smelti[l]
            hice[l] = 0.0 if (mice[l] == 0.0 and abs(hice[l] - hmelti[l]) < 1e-12) else hice[l] - hmelti[l]
    snowx = snowl[0] + dews - melts
    if snowx > 0.0:
        cmprs = min(0.0, snowx)   # dSNdML = 0 -> CMPRS = min(0*melts, snowx) = 0
        hcmprs = hsnow[0] * cmprs / snowx
        snowl[0] = snowx - cmprs
        hsnow[0] -= hcmprs
        if mice[0] > 0:
            mice[0] += cmprs; hice[0] += hcmprs
        else:
            mice[1] += cmprs; hice[1] += hcmprs
    else:
        cmprs = 0.0
        hice[0] += hsnow[0]
        snowl[0] = 0.0
        hsnow[0] = 0.0
    if melts2 > 0.0:
        if snowl[1] > melts2:
            snowl[1] -= melts2
        else:
            hice[1] += hsnow[1]
            snowl[1] = 0.0
            hsnow[1] = 0.0
    fmsi2 = -melti[0] - melti[1] + dewi[0] + dewi[1] + cmprs
    relayer(fmsi2, mice, hice, sice)
    relayer_12(hsnow, hice, sice, mice, snowl)
    snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
    run = melts + melts2 + sum(melti)
    srun = sum(smelti)
    erun = srox2 + sum(hmelti)
    tsil = tice(hsil, ssil, msi1, msi2)
    melt12 = melts + melti[0]
    return dict(snow=snow, hsil=hsil, ssil=ssil, msi2=msi2, srox2=srox2, run=run, srun=srun, erun=erun,
                wetsnow=(wetsnow or melt12 > 0), melt12=melt12, cmprs=cmprs, tsil=tsil)


# --------------------------------------------------------------- SSIDEC / snowice
SECONDS_PER_DAY = 86400.0


def ssidec(snow, msi2, hsil, ssil, dt, melt12):
    """SEAICE.f SSIDEC (seaice_thermo='BP'). Returns dict: snow,msi1,msi2,hsil,ssil,melt12,mflux,hflux,sflux."""
    hsil = list(hsil); ssil = list(ssil)
    dtssi = 30.0
    bydtssi = 1.0 / (dtssi * SECONDS_PER_DAY)
    dssi = [0.0] * LMI
    dmsi = [0.0] * LMI
    dhsi = [0.0] * LMI
    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, True)
    for l in range(LMI):
        rate = 0.0
        brine_frac = 0.0
        if sice[l] > 0.0:
            if 1e3 * sice[l] / mice[l] > 1e-10:
                brine_frac = -MU * 1e3 * (sice[l] / tsil[l]) / mice[l]
            if mice[l] > AC2OIM and brine_frac != 0.0:
                if brine_frac > 0.05 and msi2 > 2.0 * RHOI:
                    rate = min(1.0, 0.3 * melt12 / (mice[l] * brine_frac))
                else:
                    rate = min(0.3, 0.3 * melt12 / (mice[l] * brine_frac))
            if brine_frac > 0.05:
                rate = min(rate + dt * bydtssi * 5.0 * (brine_frac - 0.05), 1.0)
            if sice[l] < SSIMIN * mice[l]:
                rate = 1.0
                dmsi[l] = brine_frac * mice[l]
                dhsi[l] = max(-MU * 1e3 * sice[l] * SHW, hice[l] - (tsil[l] * SHI - LHM) * (mice[l] - dmsi[l]))
                dssi[l] = rate * sice[l]
            else:
                dmsi[l] = rate * brine_frac * mice[l]
                dhsi[l] = -rate * MU * 1e3 * sice[l] * SHW
                dssi[l] = rate * sice[l]
            sice[l] = max(0.0, sice[l] - dssi[l])
            hice[l] -= dhsi[l]
            mice[l] -= dmsi[l]
            if snow == 0.0 and l <= 1:
                melt12 += dmsi[l]
    fmsi = [0.0] * (LMI - 1)
    fhsi = [0.0] * (LMI - 1)
    fssi = [0.0] * (LMI - 1)
    fmsi[0] = -dmsi[0]
    fhsi[0] = fmsi[0] * (tsil[1] * SHI - LHM)
    fmsi[1] = -dmsi[0] - dmsi[1]
    fhsi[1] = fmsi[1] * (tsil[2] * SHI - LHM)
    if dmsi[0] + dmsi[1] > 0:
        hice[0] -= fhsi[0]
        hice[1] += fhsi[0] - fhsi[1]
        mice[0] -= fmsi[0]
        mice[1] += fmsi[0] - fmsi[1]
    snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
    fmsi2 = XSI[2] * dmsi[3] + XSI[3] * (fmsi[1] - dmsi[2])
    if fmsi2 > 0:
        fhsi2 = fmsi2 * hsil[2] / mice[2]
        fssi2 = fmsi2 * ssil[2] / mice[2]
    else:
        fhsi2 = fmsi2 * (tsil[3] * SHI - LHM)
        fssi2 = 0.0
    hsil[2] += fhsi[1] - fhsi2
    hsil[3] += fhsi2
    ssil[2] += fssi[1] - fssi2
    ssil[3] += fssi2
    msi2 += fmsi[1]
    mflux = sum(dmsi)
    hflux = sum(dhsi)
    sflux = sum(dssi)
    tsil = tice(hsil, ssil, msi1, msi2)
    return dict(snow=snow, msi1=msi1, msi2=msi2, hsil=hsil, ssil=ssil, melt12=melt12,
                mflux=mflux, hflux=hflux, sflux=sflux)


def snowice(tm, sm, snow, msi2, hsil, ssil, qsfix):
    """SEAICE.f snowice. Returns dict: snow,msi2,hsil,ssil,msnwic,hsnwic,ssnwic,dsnow (no-op if inactive)."""
    hsil = list(hsil); ssil = list(ssil)
    if not (RHOI * snow > (ACE1I + msi2) * (RHOWS - RHOI)):
        return dict(snow=snow, msi2=msi2, hsil=hsil, ssil=ssil, msnwic=0.0, hsnwic=0.0, ssnwic=0.0, dsnow=0.0)
    msi1 = snow + ACE1I
    z0 = (msi1 + msi2) / RHOWS - (ACE1I + msi2) / RHOI
    maxm = z0 * RHOWS * (RHOI - RHOS) / (RHOWS + RHOS - RHOI)
    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, True)
    si = 1e3 * SSI0 if qsfix else FSSS * sm
    tf = tfrez(sm)
    eic = Ei(tf, si)
    eoc = tm * SHW
    esnow1 = hsnow[0] / snowl[0]
    erat1 = (esnow1 - Ei(tf, 0.0)) / (eic - eoc)
    if snowl[1] > 0:
        esnow2 = hsnow[1] / snowl[1]
        erat2 = (esnow2 - Ei(tf, 0.0)) / (eic - eoc)
    else:
        esnow2 = 0.0
        erat2 = 0.0
    eratd = (esnow1 - esnow2) / (eic - eoc)
    maxme = (z0 * RHOI * erat1 - snowl[1] * eratd) / (1.0 + erat1 * (RHOWS - RHOI) / RHOWS)
    maxm = max(0.0, min(maxme, maxm))
    maxm = max(0.0, min(0.9 * RHOWS * (mice[1] / RHOI - z0), maxm))
    dsnow = z0 * RHOI - maxm * (RHOWS - RHOI) / RHOWS
    if maxm == 0.0:
        dsnow = min(0.9 * mice[1], dsnow)
    if dsnow > snowl[1]:
        hsnow[0] = hsnow[0] * (snowl[0] + snowl[1] - dsnow) / snowl[0]
        hsnow[1] = 0.0
        mice[0] += maxm + dsnow
        sice[0] += maxm * 0.001 * si
        hice[0] += maxm * eoc + snowl[1] * esnow2 + (dsnow - snowl[1]) * esnow1
        snowl[0] = snowl[0] + snowl[1] - dsnow
        snowl[1] = 0.0
    else:
        hsnow[1] = hsnow[1] * (snowl[1] - dsnow) / snowl[1]
        mice[1] += dsnow + maxm
        sice[1] += maxm * 0.001 * si
        hice[1] += maxm * eoc + dsnow * esnow2
        snowl[1] -= dsnow
    fmsi2 = maxm + dsnow
    relayer(fmsi2, mice, hice, sice)
    relayer_12(hsnow, hice, sice, mice, snowl)
    snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
    msnwic = -maxm
    hsnwic = msnwic * eoc
    ssnwic = 0.001 * si * msnwic
    return dict(snow=snow, msi2=msi2, hsil=hsil, ssil=ssil, msnwic=msnwic, hsnwic=hsnwic, ssnwic=ssnwic, dsnow=dsnow)


def ground_si_ocean(dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow,
                    tm, sm, snow_ice_flag=1, qsfix=False):
    """Full GROUND_SI cell update for domain='OCEAN' (SEA_ICE -> SSIDEC -> snowice), matching the
    Fortran call order. Returns dict with final snow/hsil/ssil/msi2 and the coupling fluxes
    runosi/erunosi/srunosi/solar_io."""
    si = sea_ice(dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)
    dec = ssidec(si["snow"], si["msi2"], si["hsil"], si["ssil"], dtsrce, si["melt12"])
    if snow_ice_flag == 1:
        sic = snowice(tm, sm, dec["snow"], dec["msi2"], dec["hsil"], dec["ssil"], qsfix)
    else:
        sic = dict(snow=dec["snow"], msi2=dec["msi2"], hsil=dec["hsil"], ssil=dec["ssil"],
                  msnwic=0.0, hsnwic=0.0, ssnwic=0.0, dsnow=0.0)
    runosi = fmoc + si["run"] + dec["mflux"] + sic["msnwic"]
    erunosi = fhoc + si["erun"] + dec["hflux"] + sic["hsnwic"]
    srunosi = fsoc + si["srun"] + dec["sflux"] + sic["ssnwic"]
    return dict(snow=sic["snow"], hsil=sic["hsil"], ssil=sic["ssil"], msi2=sic["msi2"],
                runosi=runosi, erunosi=erunosi, srunosi=srunosi, solar_io=si["srox2"],
                mflux=dec["mflux"], hflux=dec["hflux"], sflux=dec["sflux"])


def ground_si_other(dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow):
    """Full GROUND_SI cell update for domain != 'OCEAN' (lakes): SEA_ICE only, no SSIDEC/snowice."""
    si = sea_ice(dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)
    return dict(snow=si["snow"], hsil=si["hsil"], ssil=si["ssil"], msi2=si["msi2"],
                runosi=fmoc + si["run"], erunosi=fhoc + si["erun"], srunosi=fsoc + si["srun"],
                solar_io=si["srox2"], mflux=0.0, hflux=0.0, sflux=0.0)


# --------------------------------------------------------------------- ADDICE
FLEADMX = 5.0
BYHREF = 1.1


def addice(snow, roice, hsil, ssil, msi2, enrgfo, acefo, acefi, enrgfi, salto, salti, flead, qfixr):
    """SEAICE.f ADDICE (no tracers). Returns dict: snow, roice, hsil, ssil, msi2, tsil, dmimp, dhimp, dsimp."""
    hsil = list(hsil); ssil = list(ssil)
    dmimp = dhimp = dsimp = 0.0
    msi1 = snow + ACE1I
    if not qfixr:
        if roice <= 0.0 and acefo > 0.0:
            roice = min(1.0, acefo / (ACE1I + AC2OIM))
            msi1 = ACE1I
            msi2 = max(AC2OIM, acefo - ACE1I)
            snow = 0.0
            for l in (0, 1):
                hsil[l] = (enrgfo / acefo) * XSI[l] * msi1
                ssil[l] = (salto / acefo) * XSI[l] * msi1
            for l in (2, 3):
                hsil[l] = (enrgfo / acefo) * XSI[l] * msi2
                ssil[l] = (salto / acefo) * XSI[l] * msi2
        elif roice > 0.0:
            if acefi > 0.0:
                if XSI[2] * acefi > XSI[3] * msi2:
                    fhsi3 = -hsil[3] - (XSI[2] * acefi - XSI[3] * msi2) * enrgfi / acefi
                    fssi3 = -ssil[3] - (XSI[2] * acefi - XSI[3] * msi2) * salti / acefi
                else:
                    fhsi3 = -hsil[3] * acefi * (XSI[2] / XSI[3]) / msi2
                    fssi3 = -ssil[3] * acefi * (XSI[2] / XSI[3]) / msi2
            else:
                fhsi3 = fssi3 = 0.0
            if acefo == 0.0:
                hsil[2] -= fhsi3
                hsil[3] += fhsi3 + enrgfi
                ssil[2] -= fssi3
                ssil[3] += fssi3 + salti
                msi2 += acefi
            else:
                drsi = min((1.0 - roice) * acefo / (ACE1I + AC2OIM), 1.0 - roice)
                roicen = roice + drsi
                msi2no = max(AC2OIM, acefo - ACE1I)
                snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, False)
                snowl = [x * (roice / roicen) for x in snowl]
                hsnow = [x * (roice / roicen) for x in hsnow]
                for l in (0, 1):
                    hice[l] = ((1.0 - roice) * enrgfo * XSI[l] * ACE1I / (ACE1I + msi2no) + roice * hice[l]) / roicen
                    sice[l] = ((1.0 - roice) * salto * XSI[l] * ACE1I / (ACE1I + msi2no) + roice * sice[l]) / roicen
                    mice[l] = ((1.0 - roice) * acefo * XSI[l] * ACE1I / (ACE1I + msi2no) + roice * mice[l]) / roicen
                relayer_12(hsnow, hice, sice, mice, snowl)
                snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
                msi2 = (drsi * max(AC2OIM, acefo - ACE1I) + roice * (msi2 + acefi)) / roicen
                hsil[2] = ((1.0 - roice) * enrgfo * XSI[2] * msi2no / (ACE1I + msi2no) + roice * (hsil[2] - fhsi3)) / roicen
                hsil[3] = ((1.0 - roice) * enrgfo * XSI[3] * msi2no / (ACE1I + msi2no)
                          + roice * (hsil[3] + fhsi3 + enrgfi)) / roicen
                ssil[2] = ((1.0 - roice) * salto * XSI[2] * msi2no / (ACE1I + msi2no) + roice * (ssil[2] - fssi3)) / roicen
                ssil[3] = ((1.0 - roice) * salto * XSI[3] * msi2no / (ACE1I + msi2no)
                          + roice * (ssil[3] + fssi3 + salti)) / roicen
                roice = roicen

        havg = roice * (ACE1I + msi2) / RHOI
        opnocn = min(0.0, flead * math.exp(-BYHREF * (havg - 1.0)))
        if roice * (ACE1I + msi2) > FLEADMX * RHOI:
            opnocn = 0.0
        if msi2 < AC2OIM or roice > 1.0 - opnocn:
            snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, False)
            roicen = min(roice * (ACE1I + msi2) / (ACE1I + AC2OIM), 1.0 - opnocn)
            drsi = roicen - roice
            fmsi1 = -mice[0] * drsi / roicen
            fmsi2 = -(mice[0] + mice[1]) * drsi / roicen
            fmsi3 = fmsi2 * XSI[3]
            fhsi1 = hice[0] * fmsi1 / (mice[0] + 1e-30)
            fhsi2 = hice[1] * fmsi2 / mice[1]
            fhsi3 = hice[2] * fmsi3 / (msi2 * XSI[2])
            fssi1 = sice[0] * fmsi1 / (mice[0] + 1e-30)
            fssi2 = sice[1] * fmsi2 / mice[1]
            fssi3 = sice[2] * fmsi3 / (msi2 * XSI[2])
            hice[1] = hice[1] * roice / roicen + fhsi1 - fhsi2
            hice[2] = hice[2] * roice / roicen + fhsi2 - fhsi3
            hice[3] = hice[3] * roice / roicen + fhsi3
            sice[1] = sice[1] * roice / roicen + fssi1 - fssi2
            sice[2] = sice[2] * roice / roicen + fssi2 - fssi3
            sice[3] = sice[3] * roice / roicen + fssi3
            msi2 = msi2 * roice / roicen + fmsi2
            snowl = [x * roice / roicen for x in snowl]
            hsnow = [x * roice / roicen for x in hsnow]
            roice = roicen
            relayer_12(hsnow, hice, sice, mice, snowl)
            snow, msi1, msi2xx, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)

        if roice > 0.0:
            havg = roice * (ACE1I + msi2) / RHOI
            opnocn = min(0.0, flead * math.exp(-BYHREF * (havg - 1.0)))
            if roice * (ACE1I + msi2) > FLEADMX * RHOI:
                opnocn = 0.0
            if roice > (1.0 - opnocn) - 1e-3:
                roicen = 1.0 - opnocn
                drsi = max(0.0, roicen - roice)
                if drsi > 0.0:
                    fmsi4 = (ACE1I + msi2) * (drsi / roicen)
                    fhsi4 = hsil[3] * fmsi4 / (XSI[3] * msi2)
                    fssi4 = ssil[3] * fmsi4 / (XSI[3] * msi2)
                    fhsi3 = hsil[2] * fmsi4 / msi2
                    fssi3 = ssil[2] * fmsi4 / msi2
                    msi2 -= fmsi4
                    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, False)
                    fri = [mice[0] / (ACE1I + msi2), mice[1] / (ACE1I + msi2),
                           XSI[2] * msi2 / (ACE1I + msi2), XSI[3] * msi2 / (ACE1I + msi2)]
                    snowl = [x * (roice / roicen) for x in snowl]
                    hsnow = [x * (roice / roicen) for x in hsnow]
                    for l in (0, 1):
                        hice[l] = (roice / roicen) * (fhsi4 * fri[l] + hice[l])
                        sice[l] = (roice / roicen) * (fssi4 * fri[l] + sice[l])
                        mice[l] = (roice / roicen) * (fmsi4 * fri[l] + mice[l])
                    relayer_12(hsnow, hice, sice, mice, snowl)
                    snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
                    hsil[2] = (roice / roicen) * (hsil[2] + fhsi4 * fri[2] - fhsi3)
                    hsil[3] = (roice / roicen) * (hsil[3] + fhsi4 * fri[3] + fhsi3 - fhsi4)
                    ssil[2] = (roice / roicen) * (ssil[2] + fssi4 * fri[2] - fssi3)
                    ssil[3] = (roice / roicen) * (ssil[3] + fssi4 * fri[3] + fssi3 - fssi4)
                    roice = roicen
    else:
        if roice > 0.0 and msi2 < AC2OIM:
            dmimp = AC2OIM - msi2
            dhimp = sum(hsil[l] * XSI[l] * dmimp for l in (2, 3))
            dsimp = sum(ssil[l] * XSI[l] * dmimp for l in (2, 3))
            for l in (2, 3):
                hsil[l] *= AC2OIM / msi2
                ssil[l] *= AC2OIM / msi2
            msi2 = AC2OIM
    tsil = tice(hsil, ssil, msi1, msi2)
    return dict(snow=snow, roice=roice, hsil=hsil, ssil=ssil, msi2=msi2, tsil=tsil,
                dmimp=dmimp, dhimp=dhimp, dsimp=dsimp)


# --------------------------------------------------------------------- SIMELT
SILMFAC = 1.0e-7
SILMPOW = 1.36


def simelt(dt, roice, snow, msi2, hsil, ssil, pocean, tm, tfo, enrgmax):
    """SEAICE.f SIMELT (no tracers). Returns dict: roice, snow, msi2, hsil, ssil, tsil, enrgused, run0, salt."""
    hsil = list(hsil); ssil = list(ssil)
    if roice < 1e-3:
        drsi = roice
    else:
        dtemp = max(tm - tfo, 0.0)
        drsi = dt * SILMFAC * dtemp ** SILMPOW
        if roice - drsi < 1e-3:
            drsi = roice
        if enrgmax + drsi * sum(hsil) < 0:
            drsi = -enrgmax / sum(hsil)
        if roice - drsi > 1:
            drsi = 1 - roice
    enrgused = -drsi * sum(hsil)
    run0 = drsi * (snow + ACE1I + msi2)
    salt = drsi * sum(ssil)
    roice = min(1.0, roice - drsi)
    if roice < 1e-10:
        roice = 0.0
        snow = 0.0
        msi2 = AC2OIM
        if pocean > 0.0:
            for l in (0, 1):
                ssil[l] = SSI0 * XSI[l] * ACE1I
            for l in (2, 3):
                ssil[l] = SSI0 * XSI[l] * AC2OIM
        else:
            ssil = [0.0] * LMI
        for l in (0, 1):
            hsil[l] = (XSI[l] * ACE1I) * Ei(tfo, 1e3 * ssil[l] / (XSI[l] * ACE1I))
        for l in (2, 3):
            hsil[l] = (XSI[l] * AC2OIM) * Ei(tfo, 1e3 * ssil[l] / (XSI[l] * AC2OIM))
        tsil = [tfo] * LMI
        if tfo > Ti(0.0, 1e3 * SSI0) and tfo != 0.0:
            hsil = [0.0] * LMI
    else:
        # SEAICE.f leaves TSIL (intent(out)) UNASSIGNED in this branch -- there is no well-defined
        # Fortran reference value to validate against here, so this port does not claim one either.
        tsil = None
    return dict(roice=roice, snow=snow, msi2=msi2, hsil=hsil, ssil=ssil, tsil=tsil,
                enrgused=enrgused, run0=run0, salt=salt)
