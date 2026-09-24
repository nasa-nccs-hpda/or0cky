"""Sea-ice thermal properties used by the SURFACE ice tile -- Track B (float64).

Transcription of SEAICE.f `Alami`, `dEidTiws`, `solar_ice_frac` (brine-pocket "BP" thermodynamics,
the default) and the ice-layer bookkeeping at the top of the SURFACE.f ice tile (MICE, SNOWL, HCG1, HCG2,
dF1dTG, FSRI). Constants from SEAICE.f / Constants_mod.F90.
"""
import numpy as np
import jax.numpy as jnp

SHI, LHM, MU = 2060.0, 3.34e5, 0.054
RHOI, RHOS = 916.6, 300.0
BYRHOI = 1.0 / RHOI
Z1I = 0.1
ACE1I = Z1I * RHOI
XSI = (0.5, 0.5)
ALAMI0, ALAMS, ALAMDS, ALAMDT = 2.11, 0.35, 0.09, -0.011
BYRLS = 1.0 / (RHOS * ALAMS)


def alami(ti, si):
    tis = jnp.where(ti != 0.0, ti, 1.0)
    a = ALAMI0 + ALAMDT * ti + ALAMDS * si / tis
    a = jnp.where(a <= 0.0, ALAMI0, a)
    a = jnp.where(ti != 0.0, a, ALAMI0)
    return jnp.where(si < 1e-10, ALAMI0, a)


def deidtiws(ti, si, snowl, mice):
    tis = jnp.where(ti != 0.0, ti, 1.0)
    frac = mice / (mice + snowl)
    brine = SHI + frac * LHM * MU * si / (tis * tis)
    use = (ti != 0.0) | (mice == 0.0)
    return jnp.where(si < 1e-10, SHI, jnp.where(use, brine, SHI))


def solar_ice_frac(snow, wetsnow):
    """fsri(1), fsri(2) (lmax = 2)."""
    kiextvis, kiextnir1 = 1.5, 18.0
    dsnow = snow / RHOS
    hice12 = ACE1I * BYRHOI
    deep = dsnow > 0.02
    fracvis = jnp.where(deep, jnp.where(wetsnow, 0.20, 0.06), 0.24)
    fracnir1 = jnp.where(deep, jnp.where(wetsnow, 0.33, 0.31), 0.43)
    ksextvis = jnp.where(deep, jnp.where(wetsnow, 10.7, 19.6), 10.7)
    ksextnir1 = jnp.where(deep, jnp.where(wetsnow, 118.0, 196.0), 118.0)
    layer_ice = ACE1I * XSI[0] > snow * XSI[1]
    hice1 = (ACE1I - XSI[1] * (snow + ACE1I)) * BYRHOI
    fv1a = jnp.exp(-ksextvis * dsnow - kiextvis * hice1)
    fn1a = jnp.exp(-ksextnir1 * dsnow - kiextnir1 * hice1)
    dsnow1 = (ACE1I + snow) * XSI[0] / RHOS
    fv1b = jnp.exp(-ksextvis * dsnow1)
    fn1b = jnp.exp(-ksextnir1 * dsnow1)
    fsri1 = fracvis * jnp.where(layer_ice, fv1a, fv1b) + fracnir1 * jnp.where(layer_ice, fn1a, fn1b)
    fv2 = jnp.exp(-ksextvis * dsnow - kiextvis * hice12)
    fn2 = jnp.exp(-ksextnir1 * dsnow - kiextnir1 * hice12)
    fsri2 = fracvis * fv2 + fracnir1 * fn2
    return fsri1, fsri2


def ice_tile_props(tg1, tg2, snow, ssi1, ssi2, srheat, wetsnow):
    """Returns dF1dTG, HCG1, HCG2, FSRI1, FSRI2 for the SURFACE.f ice tile (inputs at PBL entry)."""
    msi1 = snow + ACE1I
    df1dtg = 2. / (ACE1I / (RHOI * alami(tg1, 1e3 * ((ssi1 + ssi2) / ACE1I))) + snow * BYRLS)
    some_ice = ACE1I > XSI[1] * msi1
    mice1 = jnp.where(some_ice, ACE1I - XSI[1] * msi1, 0.0)
    mice2 = jnp.where(some_ice, XSI[1] * msi1, ACE1I)
    snowl1 = jnp.where(some_ice, msi1 - ACE1I, XSI[0] * msi1)
    snowl2 = jnp.where(some_ice, 0.0, XSI[1] * msi1 - ACE1I)
    m1s = jnp.where(mice1 != 0.0, mice1, 1.0)
    hcg1 = jnp.where(mice1 != 0.0,
                     deidtiws(tg1, 1e3 * (ssi1 / m1s), snowl1, mice1),
                     deidtiws(tg1, 0.0, snowl1, 0.0)) * XSI[0] * msi1
    hcg2 = deidtiws(tg2, 1e3 * (ssi2 / mice2), snowl2, mice2) * XSI[1] * msi1
    f1, f2 = solar_ice_frac(snow, wetsnow)
    sun = srheat > 0
    return df1dtg, hcg1, hcg2, jnp.where(sun, f1, 0.0), jnp.where(sun, f2, 0.0)
