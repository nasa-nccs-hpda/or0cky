"""Full-fidelity port of SEAICE.f's iceocean_fluxes/icelake_fluxes -- the core physics of
SEAICE_DRV.f's UNDERICE (Stage 1 of the DYNSI/ocean port, D32). Computes the mass/salt/heat fluxes
at the base of sea ice (ocean domain) or lake ice (lakes domain).

UNDERICE itself is not separately modeled here: its own body is a thin per-cell wrapper (compute
Tic via the already-ported `Ti`, set icefrac=0 fluxes trivially, else call one of these two
functions and scale by DTsrc) with no additional arithmetic beyond what's below. `seaice_thermo=
"BP"` (brine-pocket, this project's established default) and `qsfix=.false.` (SEAICE.f's default,
not overridden by this rundeck) throughout -- KOCEAN=1 for this rundeck (decks/P2SAoM40.R), so the
OCEAN-domain "fixed SST" fallback branch (KOCEAN<1) is dead code, not ported.

`Tm`/`Sm`/`mlsh`/`Ustar`/`Coriol` (ocean domain) and `Tm`/`Dlake`/`Glake` (lakes domain) are real,
recorded inputs from the not-yet-ported ocean model / already-ported lake state -- same "record the
external forcing" pattern as DYNSI's GAIRX/GWATX etc. (D29).
"""
import math

from seaice_core_ff import LHM, SHW, SHI, MU, FSSS, RHOWS, RHOW, tfrez, alami, dEidTi

ALAMI0 = 2.11
BYSHI = 1.0 / SHI
G_MOLE_T, G_MOLE_S = 65.9, 2255.0
NITER = 5
QSFIX = False  # SEAICE.f default, not overridden by this rundeck


def iceocean_fluxes(Ti, Si, Tm, Sm, dh, ustar, coriol, dtsrc, mlsh):
    """SEAICE.f iceocean_fluxes (seaice_thermo='BP', qsfix=.false.). Returns dict: mflux, sflux,
    hflux (all positive down, kg/m^2/s, kg/m^2/s, J/m^2/s)."""
    g_turb = 2.5 * math.log(5300.0 * ustar * ustar / coriol) + 7.12
    g_T = ustar / (g_turb + G_MOLE_T)
    g_S = ustar / (g_turb + G_MOLE_S)
    rsg = RHOWS * SHW * g_T

    if Si == 0.0:  # seaice_thermo='BP', so only Si==0 triggers the no-salinity-effect branch
        alamdh = ALAMI0 / (dh + 1.0 * dtsrc * ALAMI0 * BYSHI / (2.0 * dh * 916.6))
    else:
        alamdh = alami(Ti, Si) / (dh + 1.0 * dtsrc * alami(Ti, Si) / (dEidTi(Ti, Si) * 2.0 * 916.6 * dh))

    Sb0 = 0.75 * Sm + 0.25 * Si
    Sb = Sb0
    Sib = Si
    m = 0.0
    lh = LHM
    for _ in range(NITER):
        Tb = tfrez(Sb0)
        left2 = -alamdh * (Ti - Tb) + rsg * (Tb - Tm)
        if left2 > 0.0:  # freezing
            Sib = Sb0 * FSSS  # qsfix=.false.
            if Sib > 0.0:
                lh = LHM * (1.0 + MU * Sib / Tb) + (Tb + MU * Sib) * (SHW - SHI)
            else:
                lh = LHM + Tb * (SHW - SHI)
            m = -left2 / lh
            if Sib > 0.0:
                dmdTb = (left2 * (-LHM * MU * Sib / Tb ** 2 + SHW - SHI) - lh * (alamdh + rsg)) / (lh * lh)
                dmdSi = (left2 * MU * (LHM / Tb + SHW - SHI)) / (lh * lh)
            else:
                dmdTb = (left2 * (SHW - SHI) - lh * (alamdh + rsg)) / (lh * lh)
                dmdSi = 0.0
            Sb = RHOWS * g_S * Sm / (RHOWS * g_S + m * (1.0 - FSSS))
            df3dm = -RHOWS * g_S * Sm * (1.0 - FSSS) / (RHOWS * g_S + m * (1.0 - FSSS)) ** 2
            dSbdSb = (FSSS * dmdSi - MU * dmdTb) * df3dm
        else:  # melting
            Sib = Si
            if Sib > 0.0 and Ti != 0.0:
                lh = LHM * (1.0 + MU * Sib / Ti) + (Ti + MU * Sib) * (SHW - SHI) - SHW * (Ti - Tb)
            else:
                lh = LHM + Tb * SHW - Ti * SHI
            m = -left2 / lh
            Sb = (m * Sib + RHOWS * g_S * Sm) / (RHOWS * g_S + m)
            df3dm = (Sib * (RHOWS * g_S + m) - (m * Sib + RHOWS * g_S * Sm)) / (RHOWS * g_S + m) ** 2
            dmdTb = -(alamdh + rsg) / lh + left2 * SHW / lh ** 2
            dSbdSb = -MU * dmdTb * df3dm

        f0 = Sb - Sb0
        df = dSbdSb - 1.0 + 1e-20
        Sb = min(max(Sib, Sb0 - f0 / df), 40.0)
        Sb0 = Sb

    m = max(min(0.9 * 2.0 * dh * 916.6 / dtsrc, m), -0.9 * 2.0 * dh * 916.6 / dtsrc)
    # Tb/lh are deliberately NOT recomputed here -- the real Fortran uses whatever Tb/lh were last
    # set inside the loop (from the iteration-5 Sb0, one step behind the final post-loop Sb0), not
    # a fresh tfrez(Sb0) call. Found by comparing against ffz_undocn_*.bin: mflux/sflux matched
    # immediately, only hflux (which alone depends on this specific Tb) was wrong.
    mflux = m
    sflux = 1e-3 * m * Sib
    hflux = alamdh * (Ti - Tb) - m * lh + m * SHW * Tb
    return dict(mflux=mflux, sflux=sflux, hflux=hflux)


def icelake_fluxes(Ti, Tm, dh, dtsrc, mlsh):
    """SEAICE.f icelake_fluxes. Returns dict: mflux, hflux."""
    rsg = RHOW * SHW * 1.3e-5
    alamdh = ALAMI0 / (dh + 1.0 * dtsrc * BYSHI * ALAMI0 / (2.0 * dh * 916.6))
    left2 = -alamdh * Ti - rsg * Tm
    if left2 > 0.0:
        lh = LHM
    else:
        lh = LHM - Ti * SHI
    m = -left2 / lh
    m = max(min(0.9 * 2.0 * dh * 916.6 / dtsrc, m), -0.9 * 2.0 * dh * 916.6 / dtsrc)
    mflux = m
    hflux = alamdh * Ti - m * lh
    return dict(mflux=mflux, hflux=hflux)


def icelake_fluxes_limited(Ti, Tm, dh, dtsrc, mlsh, dlake, glake):
    """UNDERICE's shallow-lake flux-limiting wrapper around icelake_fluxes."""
    out = icelake_fluxes(Ti, Tm, dh, dtsrc, mlsh)
    mflux, hflux = out["mflux"], out["hflux"]
    if dlake < 0.4:
        fluxlim = -glake / dtsrc
        if hflux < fluxlim:
            hflux = fluxlim
        if mflux < 0:
            mflux = 0.0
    return dict(mflux=mflux, hflux=hflux)
