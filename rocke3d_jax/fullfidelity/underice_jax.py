"""Batched JAX port of underice_ff.py (SEAICE.f iceocean_fluxes/icelake_fluxes) -- Stage 1 of the
DYNSI/ocean port, D32. The 5-iteration Newton solve in iceocean_fluxes is a fixed trip count
(NITER=5, not data-dependent), so it unrolls into a plain Python for-loop over batched jnp arrays --
no lax.scan/while_loop needed. The freezing/melting branch inside each iteration is computed for
both lanes and merged with jnp.where, the same pattern used throughout this project."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from seaice_core_jax import tfrez, alami, dEidTi
from seaice_core_ff import LHM, SHW, SHI, MU, FSSS, RHOWS, RHOW

ALAMI0 = 2.11
BYSHI = 1.0 / SHI
RHOI = 916.6
G_MOLE_T, G_MOLE_S = 65.9, 2255.0
NITER = 5


def iceocean_fluxes(Ti, Si, Tm, Sm, dh, ustar, coriol, dtsrc, mlsh):
    """Batched version of underice_ff.iceocean_fluxes (seaice_thermo='BP', qsfix=.false.)."""
    g_turb = 2.5 * jnp.log(5300.0 * ustar * ustar / coriol) + 7.12
    g_T = ustar / (g_turb + G_MOLE_T)
    g_S = ustar / (g_turb + G_MOLE_S)
    rsg = RHOWS * SHW * g_T

    si_zero = Si == 0.0
    alamdh_zero = ALAMI0 / (dh + 1.0 * dtsrc * ALAMI0 * BYSHI / (2.0 * dh * RHOI))
    alami_ti_si = alami(Ti, Si)
    alamdh_nonzero = alami_ti_si / (dh + 1.0 * dtsrc * alami_ti_si / (dEidTi(Ti, Si) * 2.0 * RHOI * dh))
    alamdh = jnp.where(si_zero, alamdh_zero, alamdh_nonzero)

    Sb0 = 0.75 * Sm + 0.25 * Si
    Sib = Si
    lh = jnp.full_like(Ti, LHM)
    Tb = jnp.zeros_like(Ti)
    for _ in range(NITER):
        Tb = tfrez(Sb0)
        left2 = -alamdh * (Ti - Tb) + rsg * (Tb - Tm)
        freezing = left2 > 0.0

        # ---- freezing branch ----
        Sib_f = Sb0 * FSSS
        sib_f_pos = Sib_f > 0.0
        lh_f_pos = LHM * (1.0 + MU * Sib_f / Tb) + (Tb + MU * Sib_f) * (SHW - SHI)
        lh_f_neg = LHM + Tb * (SHW - SHI)
        lh_f = jnp.where(sib_f_pos, lh_f_pos, lh_f_neg)
        m_f = -left2 / lh_f
        dmdTb_f_pos = (left2 * (-LHM * MU * Sib_f / Tb ** 2 + SHW - SHI) - lh_f * (alamdh + rsg)) / (lh_f * lh_f)
        dmdSi_f_pos = (left2 * MU * (LHM / Tb + SHW - SHI)) / (lh_f * lh_f)
        dmdTb_f_neg = (left2 * (SHW - SHI) - lh_f * (alamdh + rsg)) / (lh_f * lh_f)
        dmdSi_f_neg = jnp.zeros_like(Ti)
        dmdTb_f = jnp.where(sib_f_pos, dmdTb_f_pos, dmdTb_f_neg)
        dmdSi_f = jnp.where(sib_f_pos, dmdSi_f_pos, dmdSi_f_neg)
        Sb_f = RHOWS * g_S * Sm / (RHOWS * g_S + m_f * (1.0 - FSSS))
        df3dm_f = -RHOWS * g_S * Sm * (1.0 - FSSS) / (RHOWS * g_S + m_f * (1.0 - FSSS)) ** 2
        dSbdSb_f = (FSSS * dmdSi_f - MU * dmdTb_f) * df3dm_f

        # ---- melting branch ----
        Sib_m = Si
        sib_m_pos = (Sib_m > 0.0) & (Ti != 0.0)
        lh_m_pos = LHM * (1.0 + MU * Sib_m / Ti) + (Ti + MU * Sib_m) * (SHW - SHI) - SHW * (Ti - Tb)
        lh_m_neg = LHM + Tb * SHW - Ti * SHI
        lh_m = jnp.where(sib_m_pos, lh_m_pos, lh_m_neg)
        m_m = -left2 / lh_m
        Sb_m = (m_m * Sib_m + RHOWS * g_S * Sm) / (RHOWS * g_S + m_m)
        df3dm_m = (Sib_m * (RHOWS * g_S + m_m) - (m_m * Sib_m + RHOWS * g_S * Sm)) / (RHOWS * g_S + m_m) ** 2
        dmdTb_m = -(alamdh + rsg) / lh_m + left2 * SHW / lh_m ** 2
        dSbdSb_m = -MU * dmdTb_m * df3dm_m

        Sib = jnp.where(freezing, Sib_f, Sib_m)
        lh = jnp.where(freezing, lh_f, lh_m)
        m = jnp.where(freezing, m_f, m_m)
        Sb = jnp.where(freezing, Sb_f, Sb_m)
        dSbdSb = jnp.where(freezing, dSbdSb_f, dSbdSb_m)

        f0 = Sb - Sb0
        df = dSbdSb - 1.0 + 1e-20
        Sb0 = jnp.clip(Sb0 - f0 / df, Sib, 40.0)

    m = jnp.clip(m, -0.9 * 2.0 * dh * RHOI / dtsrc, 0.9 * 2.0 * dh * RHOI / dtsrc)
    mflux = m
    sflux = 1e-3 * m * Sib
    hflux = alamdh * (Ti - Tb) - m * lh + m * SHW * Tb
    return dict(mflux=mflux, sflux=sflux, hflux=hflux)


def icelake_fluxes(Ti, Tm, dh, dtsrc, mlsh):
    """Batched version of underice_ff.icelake_fluxes."""
    rsg = RHOW * SHW * 1.3e-5
    alamdh = ALAMI0 / (dh + 1.0 * dtsrc * BYSHI * ALAMI0 / (2.0 * dh * RHOI))
    left2 = -alamdh * Ti - rsg * Tm
    lh = jnp.where(left2 > 0.0, LHM, LHM - Ti * SHI)
    m = -left2 / lh
    m = jnp.clip(m, -0.9 * 2.0 * dh * RHOI / dtsrc, 0.9 * 2.0 * dh * RHOI / dtsrc)
    mflux = m
    hflux = alamdh * Ti - m * lh
    return dict(mflux=mflux, hflux=hflux)


def icelake_fluxes_limited(Ti, Tm, dh, dtsrc, mlsh, dlake, glake):
    """Batched version of underice_ff.icelake_fluxes_limited."""
    out = icelake_fluxes(Ti, Tm, dh, dtsrc, mlsh)
    mflux, hflux = out["mflux"], out["hflux"]
    shallow = dlake < 0.4
    fluxlim = -glake / dtsrc
    hflux = jnp.where(shallow & (hflux < fluxlim), fluxlim, hflux)
    mflux = jnp.where(shallow & (mflux < 0), 0.0, mflux)
    return dict(mflux=mflux, hflux=hflux)
