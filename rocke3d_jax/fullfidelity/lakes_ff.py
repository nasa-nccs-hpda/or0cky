"""Reference transcription (plain Python, float64) of the ModelE lake physics (LAKES.f `LKSOURC`,
`LKMIX`) -- Track B. No tracers.

Track A's `lakes_jax.py` `lkmix` is a documented no-op placeholder; the REAL Fortran `LKMIX` is not --
it is a genuine two-layer lake model (static-stability mixing against a parabolic density-temperature
relation, implicit vertical heat diffusion, TKE-driven entrainment). This module ports the real thing.
"""
import math

SHW = 4185.0
SHI = 2060.0 * 1000.0 / 1000.0  # SHI is per-kg here (LAKES.f uses CONSTANT's shi directly, J/(kg C))
SHI = 2060.0
LHM = 3.34e5
RHOW = 1000.0
GRAV = 9.80664999999999942
BYGRAV = 1.0 / GRAV
TF = 273.15

MINMLD = 1.0
TMAXRHO = 4.0
KVLAKE = 1e-5
TFL = 0.0
BYZETA = 1.0 / 0.35
EMIN = -1e-10
MAXRHO, RHO0 = 1e3, 999.842594
BFAC = (MAXRHO - RHO0) / 16.0


def _lksourc_core(roice, mlake, elake, runi, enrgo, enrgo2, enrgi, enrgi2, runo):
    dm2 = dh2 = 0.0
    if mlake[0] + runo < MINMLD * RHOW and mlake[1] > 0.0:
        dm2 = min(mlake[1], MINMLD * RHOW - (mlake[0] + runo))
        dh2 = dm2 * (elake[1] + (1.0 - roice) * enrgo2 + roice * enrgi2) / mlake[1]
    if dm2 < mlake[1]:
        mlake[1] = mlake[1] - dm2
        elake[1] = elake[1] - dh2 + (1.0 - roice) * enrgo2 + roice * enrgi2
    else:
        mlake[1] = 0.0
        elake[1] = 0.0

    e2o = e2i = 0.0
    enrgfo = acefo = 0.0
    enrgfi = acefi = 0.0
    if roice < 1.0:
        fho = elake[0] + enrgo + dh2 - (mlake[0] + dm2 + runo) * TFL * SHW
        if fho < EMIN:
            acefo = fho / (TFL * (SHI - SHW) - LHM)
            acefo = min(acefo, max(mlake[0] + dm2 + runo - MINMLD * RHOW, 0.0))
            enrgfo = acefo * (TFL * SHI - LHM)
            e2o = fho - enrgfo
    if roice > 0.0:
        fhi = elake[0] + dh2 + enrgi - (mlake[0] + dm2 + runi) * TFL * SHW
        if fhi < EMIN:
            acefi = fhi / (TFL * (SHI - SHW) - LHM)
            acefi = min(acefi, max(mlake[0] + dm2 + runi - MINMLD * RHOW, 0.0))
            enrgfi = acefi * (TFL * SHI - LHM)
            e2i = fhi - enrgfi

    mlake[0] = mlake[0] + dm2 + (1.0 - roice) * (runo - acefo) + roice * (runi - acefi)
    elake[0] = elake[0] + dh2 + (1.0 - roice) * (enrgo - enrgfo) + roice * (enrgi - enrgfi)

    acef1 = acef2 = enrgf1 = enrgf2 = 0.0
    fh2 = elake[0] - mlake[0] * TFL * SHW
    if fh2 < EMIN:
        if mlake[1] > 0.0:
            tlk2 = elake[1] / (mlake[1] * SHW)
            acef2 = -fh2 / (tlk2 * SHW - TFL * SHI + LHM)
            acef2 = min(acef2, mlake[1])
            enrgf2 = acef2 * (TFL * SHI - LHM)
            elake[0] = elake[0] + acef2 * tlk2 * SHW - enrgf2
            elake[1] = elake[1] - acef2 * tlk2 * SHW
            mlake[1] = mlake[1] - acef2
        fh1 = elake[0] - mlake[0] * TFL * SHW
        if fh1 < EMIN:
            acef1 = fh1 / (TFL * (SHI - SHW) - LHM)
            if mlake[0] - acef1 < 0.5 * RHOW:
                enrgf1 = fh1
                acef1 = min(mlake[0] - 0.2 * RHOW, max(0.4 * mlake[0] + 0.6 * acef1 - 0.2 * RHOW, 0.0))
                min_ice_t = -100.0    # SEAICE's minIceTemperature default (not overridden by rundeck)
                if enrgf1 < acef1 * (min_ice_t * SHI - LHM):
                    acef1 = enrgf1 / (min_ice_t * SHI - LHM)
                if acef1 > mlake[0]:
                    raise RuntimeError("Lake water too cold during LKSOURC")
            else:
                enrgf1 = acef1 * (TFL * SHI - LHM)
            mlake[0] = mlake[0] - acef1
            elake[0] = mlake[0] * TFL * SHW

    frato = frati = 1.0
    if e2i + e2o < 0:
        frato = e2o / (e2i * roice + e2o * (1.0 - roice))
        frati = e2i / (e2i * roice + e2o * (1.0 - roice))
    acefo = acefo + (acef1 + acef2) * frato
    acefi = acefi + (acef1 + acef2) * frati
    enrgfo = enrgfo + (enrgf1 + enrgf2) * frato
    enrgfi = enrgfi + (enrgf1 + enrgf2) * frati
    return dict(mlake=mlake, elake=elake, enrgfo=enrgfo, acefo=acefo, acefi=acefi, enrgfi=enrgfi)


def lksourc_full(roice, mlake, elake, run0, fodt, fidt, srox, fsr2, evapo):
    """Full LKSOURC call (matches the real argument list: RUNO=-EVAPO, RUNI=RUN0)."""
    mlake = list(mlake); elake = list(elake)
    enrgo = fodt - srox[0] * fsr2
    enrgo2 = srox[0] * fsr2
    enrgi = fidt - srox[1] * fsr2
    enrgi2 = srox[1] * fsr2
    runo = -evapo
    return _lksourc_core(roice, mlake, elake, run0, enrgo, enrgo2, enrgi, enrgi2, runo)


def lkmix(mlake, elake, hlake, tke, roice, dtsrc):
    """Returns dict: mlake, elake (length-2 lists)."""
    mlake = list(mlake); elake = list(elake)
    if mlake[1] > 0.0:
        tlk1 = elake[0] / (mlake[0] * SHW)
        tlk2 = elake[1] / (mlake[1] * SHW)
        hlt = elake[0] + elake[1]
        mlt = mlake[0] + mlake[1]
        if (TMAXRHO - 0.5 * (tlk1 + tlk2)) * (tlk2 - tlk1) < 0:
            mlake[0] = min(mlt, max(MINMLD * RHOW, mlt - hlake * RHOW))
            mlake[1] = mlt - mlake[0]
            elake[0] = hlt * mlake[0] / mlt
            elake[1] = hlt * mlake[1] / mlt
        else:
            dtk = 2.0 * KVLAKE * (1.0 - roice) * dtsrc * RHOW ** 2
            e1n = (elake[0] + dtk * hlt / (mlt * mlake[1])) / (1.0 + dtk / (mlake[0] * mlake[1]))
            e2n = (elake[1] + dtk * hlt / (mlt * mlake[0])) / (1.0 + dtk / (mlake[0] * mlake[1]))
            elake[0] = e1n
            elake[1] = e2n
            if tke > 0.0:
                atke = 0.2 * tke
                h1 = mlake[0] / RHOW
                h2 = mlake[1] / RHOW
                drho = (tlk2 - tlk1) * 2.0 * BFAC * (TMAXRHO - 0.5 * (tlk1 + tlk2))
                dml = atke * BYGRAV / (drho * 0.5 * h1)
                if dml * RHOW < mlake[1]:
                    dhml = dml * elake[1] / h2
                    elake[0] += dhml
                    elake[1] -= dhml
                    mlake[0] += dml * RHOW
                    mlake[1] -= dml * RHOW
                else:
                    mlake[0] = mlt
                    mlake[1] = 0.0
                    elake[0] = hlt
                    elake[1] = 0.0
    return dict(mlake=mlake, elake=elake)


TEENY = 1e-30


def precip_lk(flake, flice, rsi, prcp, enrgp, runpsi, runo_li, melti, emelti, axyp, mwl0, gml0, tlake0, mldlk0,
             gtemp0, gtemp20, gtempr0):
    """LAKES.f PRECIP_LK (no tracers, no SCM, no irrigation): applies precipitation/land-ice runoff/lake-ice
    melt to the lake mass/energy reservoir. Runs (per real Fortran) at every land+lake grid box (FLAKE+FLICE>0);
    a genuinely inactive cell passes every state field through unchanged. Returns dict: mwl, gml, tlake, mldlk,
    dlake, glake, gtemp, gtemp2, gtempr, run0, erun0."""
    if flake + flice <= 0.0:
        return dict(mwl=mwl0, gml=gml0, tlake=tlake0, mldlk=mldlk0, dlake=0.0, glake=0.0, gtemp=gtemp0,
                    gtemp2=gtemp20, gtempr=gtempr0, run0=0.0, erun0=0.0)
    polake = (1.0 - rsi) * flake
    plkice = rsi * flake
    plice = flice
    run0 = polake * prcp + plkice * runpsi + plice * runo_li
    erun0 = polake * enrgp
    if flake > 0.0:
        run0 += melti
        erun0 += emelti
    mwl = mwl0 + run0 * axyp
    gml = gml0 + erun0 * axyp
    if flake > 0.0:
        hlk1 = tlake0 * mldlk0 * RHOW * SHW
        mldlk = mldlk0 + run0 / (flake * RHOW)
        tlake = (hlk1 * flake + erun0) / (mldlk * flake * RHOW * SHW)
        dlake = mwl / (RHOW * flake * axyp)
        glake = gml / (flake * axyp)
        gtemp = tlake
        gtempr = tlake + TF
        if mwl > (1e-10 + mldlk) * RHOW * flake * axyp:
            gtemp2 = (gml - tlake * SHW * mldlk * RHOW * flake * axyp) / (SHW * (mwl - mldlk * RHOW * flake * axyp))
        else:
            gtemp2 = tlake
    else:
        tlake = gml / (mwl * SHW + TEENY)
        mldlk = mldlk0
        dlake = 0.0
        glake = 0.0
        gtemp, gtemp2, gtempr = gtemp0, gtemp20, gtempr0
    return dict(mwl=mwl, gml=gml, tlake=tlake, mldlk=mldlk, dlake=dlake, glake=glake, gtemp=gtemp, gtemp2=gtemp2,
                gtempr=gtempr, run0=run0, erun0=erun0)
