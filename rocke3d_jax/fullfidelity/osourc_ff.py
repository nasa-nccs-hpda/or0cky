"""Full-fidelity port of OCNDYN.f's OSOURC (called from GROUND_OC) -- Stage 2 of the DYNSI/ocean
port, D34. Applies surface mass/heat/salt fluxes (river+ice-melt runoff, evaporation, solar
insolation) to the ocean's water column, separately over the open-ocean and ice-covered fractions
of a cell, checks each fraction for below-freezing conditions (frazil-ice formation), and
distributes the recombined fluxes across the water column with an exponential (two-band Jerlov)
solar-penetration profile. TRACERS_OCEAN undefined for this rundeck (dead code, not ported).

FSR/FSRZ/LSRPD (the solar-penetration profile) are derived analytically here from OCEAN_COM.f's
`init_solar` -- RFRAC/ZETA1/ZETA2/ZMAX_SOLAR are hardcoded PARAMETERs (not runtime/file-dependent),
and the L13 layering (OCN_LAYERING L13, this rundeck's build flag) gives a fixed dZO array -- so
this is fully re-derivable rather than needing a recorded dump, unlike D29's ice-dyn RADIUS/GRAV
(which were genuine runtime USE_PLANET_RAD parameters). Validated implicitly: if OSOURC's own
output matches the real dump using these derived FSR/FSRZ/LSRPD values, the derivation is correct.
"""
import math

from seaice_core_ff import Ei, FSSS

# ---- OCEAN_COM.f init_solar: analytic, no external data ----
RFRAC, ZETA1, ZETA2 = 0.62, 1.5, 20.0
ZMAX_SOLAR = 92.0
DZO_L13 = (12.0, 18.0, 27.0, 40.5, 60.75, 91.125, 136.6875, 205.03125, 307.546875,
           461.3203125, 691.98046875, 1037.970703125, 1556.9560546875)
LMO = 13


def _ef(z):
    return RFRAC * math.exp(-z / ZETA1) + (1.0 - RFRAC) * math.exp(-z / ZETA2)


def _efz(z):
    return ZETA1 * RFRAC * math.exp(-z / ZETA1) + ZETA2 * (1.0 - RFRAC) * math.exp(-z / ZETA2)


def _init_solar():
    """OCEAN_COM.f init_solar, specialized to this rundeck's L13 layering. Returns (lsrpd, fsr,
    fsrz), fsr/fsrz 1-indexed dicts (Fortran L=1..LSRPD) to mirror the source directly."""
    ze = [0.0] * (LMO + 1)
    for l in range(1, LMO + 1):
        ze[l] = ze[l - 1] + DZO_L13[l - 1]

    lsrpd = LMO - 1
    for l in range(1, LMO):
        if ze[l + 1] > ZMAX_SOLAR:
            lsrpd = l
            break

    fsr, fsrz = {}, {}
    for l in range(1, lsrpd + 1):
        fsr[l] = _ef(ze[l - 1])
        fsrz[l] = (-3.0 * (_ef(ze[l - 1]) + _ef(ze[l]))
                   + 6.0 * (_efz(ze[l - 1]) - _efz(ze[l])) / (ze[l] - ze[l - 1]))
    return lsrpd, fsr, fsrz


LSRPD, FSR, FSRZ = _init_solar()


def gfrezs(s):
    """OCNFUNTAB.f GFREZS: linear interpolation of a hardcoded 41-point table (J/kg)."""
    f = (0.000, -232.482, -461.136, -688.288, -914.774, -1141.078, -1367.530, -1594.374,
         -1821.798, -2049.957, -2278.978, -2508.971, -2740.030, -2972.237, -3205.667, -3440.386,
         -3676.454, -3913.926, -4152.851, -4393.287, -4635.259, -4878.815, -5123.994, -5370.830,
         -5619.358, -5869.609, -6121.615, -6375.402, -6631.001, -6888.436, -7147.733, -7408.917,
         -7672.010, -7937.037, -8204.019, -8472.976, -8743.931, -9016.917, -9291.927, -9568.992,
         -9848.131)
    ss = s * 1000.0
    js = int(ss)
    if js >= 40:
        js = 39
    return (js - ss + 1) * f[js] + (ss - js) * f[js + 1]


def tfrezs(sin_):
    """OCNFUNTAB.f TFREZS: closed-form freezing temperature (C); same formula as
    seaice_core_ff.tfrez but takes salinity in kg/kg (0 to .04), not PSU."""
    s = sin_ * 1e3
    s32 = s * math.sqrt(s)
    return (-0.0575 + (-2.154996e-4) * s) * s + 1.710523e-3 * s32


def osourc(roice, mo, g0ml, gzml, s0m, dxypj, bydxypj, lmij, runo, runi, eruno, eruni, sruno,
           sruni, srox):
    """OCNDYN.f OSOURC. g0ml/gzml are length-LMO lists/arrays (Fortran G0ML(LSRPD)/GZML(LSRPD),
    only 1..LSRPD touched); returns dict: mo, s0m, g0ml, gzml (all updated), dmoo, deoo, dmoi,
    deoi, dsoo, dsoi."""
    g0ml = list(g0ml)
    gzml = list(gzml)
    dmoo = deoo = dmoi = deoi = dsoo = dsoi = 0.0
    lsr = min(LSRPD, lmij)

    # ---- open ocean ----
    moo = mo + runo
    gmoo = g0ml[0] * bydxypj + eruno
    smoo = s0m * bydxypj + sruno
    if roice < 1.0:
        if lsr > 1:
            gmoo -= srox[0] * FSR[2]
        goo = gmoo / moo
        soo = smoo / moo
        gfoo = gfrezs(soo)
        if goo < gfoo:
            tfoo = tfrezs(soo)
            sioo = FSSS * soo
            eioo = Ei(tfoo, sioo * 1e3)
            dmoo = moo * (goo - gfoo) / (eioo - gfoo)
            deoo = eioo * dmoo
            dsoo = sioo * dmoo

    # ---- ocean under ice ----
    moi = mo + runi
    gmoi = g0ml[0] * bydxypj + eruni
    smoi = s0m * bydxypj + sruni
    if roice > 0.0:
        if lsr > 1:
            gmoi -= srox[1] * FSR[2]
        goi = gmoi / moi
        soi = smoi / moi
        gfoi = gfrezs(soi)
        if goi < gfoi:
            tfoi = tfrezs(soi)
            sioi = FSSS * soi
            eioi = Ei(tfoi, sioi * 1e3)
            dmoi = moi * (goi - gfoi) / (eioi - gfoi)
            deoi = eioi * dmoi
            dsoi = sioi * dmoi

    # ---- recombine, update layer 1 ----
    mo_new = (moi - dmoi) * roice + (1.0 - roice) * (moo - dmoo)
    g0ml[0] = ((gmoi - deoi) * roice + (1.0 - roice) * (gmoo - deoo)) * dxypj
    s0m_new = ((smoi - dsoi) * roice + (1.0 - roice) * (smoo - dsoo)) * dxypj

    # ---- distribute insolation to lower layers ----
    tsol = (srox[0] * (1.0 - roice) + srox[1] * roice) * dxypj
    for l in range(2, lsr):
        g0ml[l - 1] += tsol * (FSR[l] - FSR[l + 1])
        gzml[l - 1] += tsol * FSRZ[l]
    g0ml[lsr - 1] += tsol * FSR[lsr]
    gzml[lsr - 1] += tsol * FSRZ[lsr]

    return dict(mo=mo_new, s0m=s0m_new, g0ml=g0ml, gzml=gzml, dmoo=dmoo, deoo=deoo, dmoi=dmoi,
                deoi=deoi, dsoo=dsoo, dsoi=dsoi)
