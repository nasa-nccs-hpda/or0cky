"""JAX-vectorized (batched-array) port of lakes_ff.py's LKSOURC/LKMIX -- Track B.

Same physics as lakes_ff.py (validated bitwise against real Fortran, FULL_FIDELITY_DELTAS.md D13);
this module operates on whole arrays of cells at once instead of one Python call per cell, using
jnp.where in place of Python if/else. No dynamic-shape or per-cell-loop concerns here (unlike GHY or
sea-ice) -- lakes_ff.py is a small, fixed two-layer (upper/lower) model with no data-dependent
iteration, so this is a direct branch-by-branch transcription, not a novel technique.
"""
import jax
jax.config.update("jax_enable_x64", True)  # float32 is insufficient to match lakes_ff.py's Fortran-derived ops
import jax.numpy as jnp

SHW = 4185.0
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


def _safe_div(a, b):
    return a / jnp.where(b == 0.0, 1.0, b)


def _lksourc_core(roice, mlake0, mlake1, elake0, elake1, runi, enrgo, enrgo2, enrgi, enrgi2, runo):
    cond_a = (mlake0 + runo < MINMLD * RHOW) & (mlake1 > 0.0)
    dm2_a = jnp.minimum(mlake1, MINMLD * RHOW - (mlake0 + runo))
    dh2_a = dm2_a * (elake1 + (1.0 - roice) * enrgo2 + roice * enrgi2) / jnp.where(mlake1 == 0.0, 1.0, mlake1)
    dm2 = jnp.where(cond_a, dm2_a, 0.0)
    dh2 = jnp.where(cond_a, dh2_a, 0.0)

    cond_keep = dm2 < mlake1
    mlake1_b = jnp.where(cond_keep, mlake1 - dm2, 0.0)
    elake1_b = jnp.where(cond_keep, elake1 - dh2 + (1.0 - roice) * enrgo2 + roice * enrgi2, 0.0)
    mlake1, elake1 = mlake1_b, elake1_b

    fho = elake0 + enrgo + dh2 - (mlake0 + dm2 + runo) * TFL * SHW
    cond_o = (roice < 1.0) & (fho < EMIN)
    acefo_a = fho / (TFL * (SHI - SHW) - LHM)
    acefo_a = jnp.minimum(acefo_a, jnp.maximum(mlake0 + dm2 + runo - MINMLD * RHOW, 0.0))
    enrgfo_a = acefo_a * (TFL * SHI - LHM)
    e2o_a = fho - enrgfo_a
    acefo = jnp.where(cond_o, acefo_a, 0.0)
    enrgfo = jnp.where(cond_o, enrgfo_a, 0.0)
    e2o = jnp.where(cond_o, e2o_a, 0.0)

    fhi = elake0 + dh2 + enrgi - (mlake0 + dm2 + runi) * TFL * SHW
    cond_i = (roice > 0.0) & (fhi < EMIN)
    acefi_a = fhi / (TFL * (SHI - SHW) - LHM)
    acefi_a = jnp.minimum(acefi_a, jnp.maximum(mlake0 + dm2 + runi - MINMLD * RHOW, 0.0))
    enrgfi_a = acefi_a * (TFL * SHI - LHM)
    e2i_a = fhi - enrgfi_a
    acefi = jnp.where(cond_i, acefi_a, 0.0)
    enrgfi = jnp.where(cond_i, enrgfi_a, 0.0)
    e2i = jnp.where(cond_i, e2i_a, 0.0)

    mlake0 = mlake0 + dm2 + (1.0 - roice) * (runo - acefo) + roice * (runi - acefi)
    elake0 = elake0 + dh2 + (1.0 - roice) * (enrgo - enrgfo) + roice * (enrgi - enrgfi)

    fh2 = elake0 - mlake0 * TFL * SHW
    cond_fh2 = fh2 < EMIN

    cond_m1pos = mlake1 > 0.0
    tlk2 = _safe_div(elake1, mlake1 * SHW)
    acef2_a = -fh2 / (tlk2 * SHW - TFL * SHI + LHM)
    acef2_a = jnp.minimum(acef2_a, mlake1)
    enrgf2_a = acef2_a * (TFL * SHI - LHM)
    elake0_after2 = elake0 + acef2_a * tlk2 * SHW - enrgf2_a
    elake1_after2 = elake1 - acef2_a * tlk2 * SHW
    mlake1_after2 = mlake1 - acef2_a
    cond2 = cond_fh2 & cond_m1pos
    acef2 = jnp.where(cond2, acef2_a, 0.0)
    enrgf2 = jnp.where(cond2, enrgf2_a, 0.0)
    elake0_2 = jnp.where(cond2, elake0_after2, elake0)
    elake1_2 = jnp.where(cond2, elake1_after2, elake1)
    mlake1_2 = jnp.where(cond2, mlake1_after2, mlake1)

    fh1 = elake0_2 - mlake0 * TFL * SHW
    cond1_outer = cond_fh2 & (fh1 < EMIN)
    acef1_raw = fh1 / (TFL * (SHI - SHW) - LHM)
    cond_thin = (mlake0 - acef1_raw) < 0.5 * RHOW
    enrgf1_thin = fh1
    acef1_thin = jnp.minimum(mlake0 - 0.2 * RHOW, jnp.maximum(0.4 * mlake0 + 0.6 * acef1_raw - 0.2 * RHOW, 0.0))
    min_ice_t = -100.0
    cond_cap = enrgf1_thin < acef1_thin * (min_ice_t * SHI - LHM)
    acef1_thin = jnp.where(cond_cap, enrgf1_thin / (min_ice_t * SHI - LHM), acef1_thin)
    # NB: the original raises RuntimeError here if acef1 > mlake0 ("lake water too cold") -- a real
    # invariant violation, not expected with physical inputs; not replicated (can't raise inside a
    # traced function), matching how other such assertions are treated elsewhere in this project.
    enrgf1_thick = acef1_raw * (TFL * SHI - LHM)
    acef1_a = jnp.where(cond_thin, acef1_thin, acef1_raw)
    enrgf1_a = jnp.where(cond_thin, enrgf1_thin, enrgf1_thick)
    mlake0_after1 = mlake0 - acef1_a
    elake0_after1 = mlake0_after1 * TFL * SHW

    acef1 = jnp.where(cond1_outer, acef1_a, 0.0)
    enrgf1 = jnp.where(cond1_outer, enrgf1_a, 0.0)
    mlake0 = jnp.where(cond1_outer, mlake0_after1, mlake0)
    elake0 = jnp.where(cond1_outer, elake0_after1, elake0_2)
    elake1 = elake1_2
    mlake1 = mlake1_2

    denom = e2i * roice + e2o * (1.0 - roice)
    cond_neg = (e2i + e2o) < 0.0
    frato = jnp.where(cond_neg, _safe_div(e2o, denom), 1.0)
    frati = jnp.where(cond_neg, _safe_div(e2i, denom), 1.0)
    acefo = acefo + (acef1 + acef2) * frato
    acefi = acefi + (acef1 + acef2) * frati
    enrgfo = enrgfo + (enrgf1 + enrgf2) * frato
    enrgfi = enrgfi + (enrgf1 + enrgf2) * frati
    return dict(mlake0=mlake0, mlake1=mlake1, elake0=elake0, elake1=elake1, enrgfo=enrgfo, acefo=acefo,
               acefi=acefi, enrgfi=enrgfi)


def lksourc_full(roice, mlake0, mlake1, elake0, elake1, run0, fodt, fidt, srox0, srox1, fsr2, evapo):
    """Full LKSOURC call (matches the real argument list: RUNO=-EVAPO, RUNI=RUN0)."""
    enrgo = fodt - srox0 * fsr2
    enrgo2 = srox0 * fsr2
    enrgi = fidt - srox1 * fsr2
    enrgi2 = srox1 * fsr2
    runo = -evapo
    return _lksourc_core(roice, mlake0, mlake1, elake0, elake1, run0, enrgo, enrgo2, enrgi, enrgi2, runo)


def lkmix(mlake0, mlake1, elake0, elake1, hlake, tke, roice, dtsrc):
    """Returns dict: mlake0, mlake1, elake0, elake1."""
    cond_m1pos = mlake1 > 0.0
    mlake1_safe = jnp.where(mlake1 == 0.0, 1.0, mlake1)
    mlake0_safe = jnp.where(mlake0 == 0.0, 1.0, mlake0)
    tlk1 = elake0 / (mlake0_safe * SHW)
    tlk2 = elake1 / (mlake1_safe * SHW)
    hlt = elake0 + elake1
    mlt = mlake0 + mlake1
    mlt_safe = jnp.where(mlt == 0.0, 1.0, mlt)

    cond_unstable = (TMAXRHO - 0.5 * (tlk1 + tlk2)) * (tlk2 - tlk1) < 0.0

    mlake0_a = jnp.minimum(mlt, jnp.maximum(MINMLD * RHOW, mlt - hlake * RHOW))
    mlake1_a = mlt - mlake0_a
    elake0_a = hlt * mlake0_a / mlt_safe
    elake1_a = hlt * mlake1_a / mlt_safe

    dtk = 2.0 * KVLAKE * (1.0 - roice) * dtsrc * RHOW ** 2
    denom_b = 1.0 + dtk / (mlake0_safe * mlake1_safe)
    e1n = (elake0 + dtk * hlt / (mlt_safe * mlake1_safe)) / denom_b
    e2n = (elake1 + dtk * hlt / (mlt_safe * mlake0_safe)) / denom_b

    cond_tke = tke > 0.0
    atke = 0.2 * tke
    h1 = mlake0 / RHOW
    h2 = mlake1 / RHOW
    h1_safe = jnp.where(h1 == 0.0, 1.0, h1)
    h2_safe = jnp.where(h2 == 0.0, 1.0, h2)
    drho = (tlk2 - tlk1) * 2.0 * BFAC * (TMAXRHO - 0.5 * (tlk1 + tlk2))
    drho_safe = jnp.where(drho == 0.0, 1.0, drho)
    dml = atke * BYGRAV / (drho_safe * 0.5 * h1)

    cond_small_dml = dml * RHOW < mlake1
    dhml = dml * e2n / h2_safe
    elake0_tke_small = e1n + dhml
    elake1_tke_small = e2n - dhml
    mlake0_tke_small = mlake0 + dml * RHOW
    mlake1_tke_small = mlake1 - dml * RHOW

    elake0_tke_big, elake1_tke_big = hlt, jnp.zeros_like(hlt)
    mlake0_tke_big, mlake1_tke_big = mlt, jnp.zeros_like(mlt)

    elake0_tke = jnp.where(cond_small_dml, elake0_tke_small, elake0_tke_big)
    elake1_tke = jnp.where(cond_small_dml, elake1_tke_small, elake1_tke_big)
    mlake0_tke = jnp.where(cond_small_dml, mlake0_tke_small, mlake0_tke_big)
    mlake1_tke = jnp.where(cond_small_dml, mlake1_tke_small, mlake1_tke_big)

    elake0_b = jnp.where(cond_tke, elake0_tke, e1n)
    elake1_b = jnp.where(cond_tke, elake1_tke, e2n)
    mlake0_b = jnp.where(cond_tke, mlake0_tke, mlake0)
    mlake1_b = jnp.where(cond_tke, mlake1_tke, mlake1)

    mlake0_out = jnp.where(cond_m1pos, jnp.where(cond_unstable, mlake0_a, mlake0_b), mlake0)
    mlake1_out = jnp.where(cond_m1pos, jnp.where(cond_unstable, mlake1_a, mlake1_b), mlake1)
    elake0_out = jnp.where(cond_m1pos, jnp.where(cond_unstable, elake0_a, elake0_b), elake0)
    elake1_out = jnp.where(cond_m1pos, jnp.where(cond_unstable, elake1_a, elake1_b), elake1)
    return dict(mlake0=mlake0_out, mlake1=mlake1_out, elake0=elake0_out, elake1=elake1_out)
