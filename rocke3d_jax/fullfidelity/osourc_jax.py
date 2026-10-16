"""Batched JAX port of osourc_ff.py (OCNDYN.f OSOURC, called from GROUND_OC) -- Stage 2 of the
DYNSI/ocean port, D34. FSR/FSRZ/LSRPD are derived analytically at import time (same closed-form
init_solar reproduction as osourc_ff.py; LSRPD is fixed at 3 for this rundeck's L13 layering, so
the layer loop below is a plain Python-unrolled range over LMO, not a jnp/lax construct -- no
data-dependent trip count). The two below-freezing branches (open ocean / under ice) are each
computed both ways and merged with jnp.where, the established pattern."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from seaice_core_jax import Ei
from osourc_ff import FSSS, LSRPD, FSR, FSRZ, LMO

_GFREZS_TABLE = jnp.array([
    0.000, -232.482, -461.136, -688.288, -914.774, -1141.078, -1367.530, -1594.374,
    -1821.798, -2049.957, -2278.978, -2508.971, -2740.030, -2972.237, -3205.667, -3440.386,
    -3676.454, -3913.926, -4152.851, -4393.287, -4635.259, -4878.815, -5123.994, -5370.830,
    -5619.358, -5869.609, -6121.615, -6375.402, -6631.001, -6888.436, -7147.733, -7408.917,
    -7672.010, -7937.037, -8204.019, -8472.976, -8743.931, -9016.917, -9291.927, -9568.992,
    -9848.131])


def gfrezs(s):
    """Batched OCNFUNTAB.f GFREZS: linear interpolation of the hardcoded 41-point table."""
    ss = s * 1000.0
    js = jnp.minimum(ss.astype(jnp.int32), 39)  # matches the real Fortran: only the upper bound
                                                 # (JS.GE.40) is active; the lower-bound clip is
                                                 # commented out in the source, not applied
    f_js = _GFREZS_TABLE[js]
    f_js1 = _GFREZS_TABLE[js + 1]
    return (js - ss + 1) * f_js + (ss - js) * f_js1


def tfrezs(sin_):
    """Batched OCNFUNTAB.f TFREZS: closed-form freezing temperature (C)."""
    s = sin_ * 1e3
    s32 = s * jnp.sqrt(s)
    return (-0.0575 + (-2.154996e-4) * s) * s + 1.710523e-3 * s32


def osourc(roice, mo, g0ml, gzml, s0m, dxypj, bydxypj, lmij, runo, runi, eruno, eruni, sruno,
           sruni, srox1, srox2):
    """Batched version of osourc_ff.osourc. g0ml/gzml: shape (n, LMO) arrays (Fortran
    G0ML(LSRPD)/GZML(LSRPD), only columns 0..LSRPD-1 updated -- matches lmij>=LSRPD always true
    for real ocean columns in this rundeck, checked in tests). Returns dict: mo, s0m, g0ml, gzml
    (updated), dmoo, deoo, dmoi, deoi, dsoo, dsoi."""
    lsr = jnp.minimum(LSRPD, lmij)
    zeros = jnp.zeros_like(mo)

    # ---- open ocean ----
    moo = mo + runo
    gmoo0 = g0ml[:, 0] * bydxypj + eruno
    gmoo = jnp.where(lsr > 1, gmoo0 - srox1 * FSR[2], gmoo0)
    goo = gmoo / moo
    smoo = s0m * bydxypj + sruno
    soo = smoo / moo
    gfoo = gfrezs(soo)
    freezing_o = (roice < 1.0) & (goo < gfoo)
    tfoo = tfrezs(soo)
    sioo = FSSS * soo
    eioo = Ei(tfoo, sioo * 1e3)
    dmoo_f = moo * (goo - gfoo) / (eioo - gfoo)
    deoo_f = eioo * dmoo_f
    dsoo_f = sioo * dmoo_f
    dmoo = jnp.where(freezing_o, dmoo_f, zeros)
    deoo = jnp.where(freezing_o, deoo_f, zeros)
    dsoo = jnp.where(freezing_o, dsoo_f, zeros)

    # ---- ocean under ice ----
    moi = mo + runi
    gmoi0 = g0ml[:, 0] * bydxypj + eruni
    gmoi = jnp.where(lsr > 1, gmoi0 - srox2 * FSR[2], gmoi0)
    goi = gmoi / moi
    smoi = s0m * bydxypj + sruni
    soi = smoi / moi
    gfoi = gfrezs(soi)
    freezing_i = (roice > 0.0) & (goi < gfoi)
    tfoi = tfrezs(soi)
    sioi = FSSS * soi
    eioi = Ei(tfoi, sioi * 1e3)
    dmoi_f = moi * (goi - gfoi) / (eioi - gfoi)
    deoi_f = eioi * dmoi_f
    dsoi_f = sioi * dmoi_f
    dmoi = jnp.where(freezing_i, dmoi_f, zeros)
    deoi = jnp.where(freezing_i, deoi_f, zeros)
    dsoi = jnp.where(freezing_i, dsoi_f, zeros)

    mo_new = (moi - dmoi) * roice + (1.0 - roice) * (moo - dmoo)
    g0ml1_0 = ((gmoi - deoi) * roice + (1.0 - roice) * (gmoo - deoo)) * dxypj
    s0m_new = ((smoi - dsoi) * roice + (1.0 - roice) * (smoo - dsoo)) * dxypj

    tsol = (srox1 * (1.0 - roice) + srox2 * roice) * dxypj
    g0ml_cols = [g0ml1_0] + [g0ml[:, l] for l in range(1, LMO)]
    gzml_cols = [gzml[:, l] for l in range(LMO)]
    for l in range(2, LSRPD):  # Fortran L=2..LSRPD-1 (fixed trip count, LSRPD=3 for this rundeck)
        in_range = lsr > l  # Fortran "DO L=2,LSR-1" only runs for L < LSR (empty if LSR<=2)
        g0ml_cols[l - 1] = jnp.where(in_range, g0ml_cols[l - 1] + tsol * (FSR[l] - FSR[l + 1]),
                                      g0ml_cols[l - 1])
        gzml_cols[l - 1] = jnp.where(in_range, gzml_cols[l - 1] + tsol * FSRZ[l], gzml_cols[l - 1])
    # last active layer (L=LSR, data-dependent per-column -- LSR is either LSRPD or lmij<LSRPD)
    for l in range(1, LSRPD + 1):
        at_lsr = lsr == l
        g0ml_cols[l - 1] = jnp.where(at_lsr, g0ml_cols[l - 1] + tsol * FSR[l], g0ml_cols[l - 1])
        gzml_cols[l - 1] = jnp.where(at_lsr, gzml_cols[l - 1] + tsol * FSRZ[l], gzml_cols[l - 1])

    g0ml_new = jnp.stack(g0ml_cols, axis=1)
    gzml_new = jnp.stack(gzml_cols, axis=1)

    return dict(mo=mo_new, s0m=s0m_new, g0ml=g0ml_new, gzml=gzml_new, dmoo=dmoo, deoo=deoo,
                dmoi=dmoi, deoi=deoi, dsoo=dsoo, dsoi=dsoi)
