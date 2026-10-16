"""D177 item 3: sea-ice thermal columns of the ffs ice-tile record (dF1dTG, HCG1, HCG2, FSRI1, FSRI2 = columns 10..14, 0-based) computed from the
tile's own ice state, as SURFACE.f:560-598 does at the start of every substep (ITYPE == ITYPE_OCEANICE branch):

    dF1dTG = 2/(ACE1I/(RHOI*alami(TG1, 1e3*(SSI1+SSI2)/ACE1I)) + SNOW*BYRLS)
    MICE/SNOWL split of the first two layers, HCG1/HCG2 = dEidTiws(...) * XSI * MSI1
    FSRI(1:2) = solar_ice_frac(SNOW, MSI2, FLAG_DSWS) when SRHEAT > 0 else 0   (SRHEAT = FSF*COSZ1 is a radiation-derived column, taken from the row)

The arithmetic is the existing `ice_props_ff.ice_tile_props` (SEAICE.f alami, dEidTiws, solar_ice_frac); this module only applies it to
ffs-layout rows from the row's own state columns so that a record built from OUR ice state (apply_state_to_records, predict_ns2) no longer keeps
the recorded, stale values.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ice_props_ff as IP
import surface_tile_ff as ST

COLS = ('dF1dTG', 'hcg1', 'hcg2', 'fsri1', 'fsri2')


def solar_ice_frac_np(snow, wetsnow, use_imf=True):
    """SEAICE.f solar_ice_frac (lmax = 2) in numpy with the Intel libimf exp (the real build's library; jnp.exp differs by 1 ulp in ~0.1 % of the rows)."""
    import drv_zenith as Z
    ex = np.vectorize(Z._libm(use_imf)['exp'], otypes=[float])
    kiextvis, kiextnir1 = 1.5, 18.0
    dsnow = snow / IP.RHOS
    hice12 = IP.ACE1I * IP.BYRHOI
    deep = dsnow > 0.02
    fracvis = np.where(deep, np.where(wetsnow, 0.20, 0.06), 0.24)
    fracnir1 = np.where(deep, np.where(wetsnow, 0.33, 0.31), 0.43)
    ksv = np.where(deep, np.where(wetsnow, 10.7, 19.6), 10.7)
    ksn = np.where(deep, np.where(wetsnow, 118.0, 196.0), 118.0)
    layer_ice = IP.ACE1I * IP.XSI[0] > snow * IP.XSI[1]
    hice1 = (IP.ACE1I - IP.XSI[1] * (snow + IP.ACE1I)) * IP.BYRHOI
    dsnow1 = (IP.ACE1I + snow) * IP.XSI[0] / IP.RHOS
    fv1 = np.where(layer_ice, ex(-ksv * dsnow - kiextvis * hice1), ex(-ksv * dsnow1))
    fn1 = np.where(layer_ice, ex(-ksn * dsnow - kiextnir1 * hice1), ex(-ksn * dsnow1))
    fsri1 = fracvis * fv1 + fracnir1 * fn1
    fsri2 = fracvis * ex(-ksv * dsnow - kiextvis * hice12) + fracnir1 * ex(-ksn * dsnow - kiextnir1 * hice12)
    return fsri1, fsri2


def ice_thermal_columns(tile_rows, use_imf=True):
    """Returns (rows with the five columns recomputed for the ice rows (itype == 2), the five computed arrays for those rows)."""
    t = np.array(tile_rows, dtype=np.float64)
    m = t[:, ST.IN['itype']] == 2
    r = t[m]
    f = IP.ice_tile_props(jnp.asarray(r[:, ST.IN['tg1']]), jnp.asarray(r[:, ST.IN['tg2']]), jnp.asarray(r[:, ST.IN['snow']]),
                          jnp.asarray(r[:, ST.IN['ssi1']]), jnp.asarray(r[:, ST.IN['ssi2']]), jnp.asarray(r[:, ST.IN['srheat']]),
                          jnp.asarray(r[:, ST.IN['flag_dsws']] > 0.5))
    f = [np.asarray(x) for x in f]
    f1, f2 = solar_ice_frac_np(r[:, ST.IN['snow']], r[:, ST.IN['flag_dsws']] > 0.5, use_imf)
    sun = r[:, ST.IN['srheat']] > 0
    f[3], f[4] = np.where(sun, f1, 0.0), np.where(sun, f2, 0.0)
    for nm, v in zip(COLS, f):
        t[m, ST.IN[nm]] = v
    return t, f


def compare(tile_rows):
    """Max |computed - recorded| per column over the ice rows and the number of unequal rows."""
    t, f = ice_thermal_columns(tile_rows)
    m = np.asarray(tile_rows)[:, ST.IN['itype']] == 2
    out = {}
    for nm, v in zip(COLS, f):
        d = np.abs(v - np.asarray(tile_rows)[m, ST.IN[nm]])
        out[nm] = (float(d.max()) if d.size else 0.0, int((d != 0).sum()), int(d.size))
    return out
