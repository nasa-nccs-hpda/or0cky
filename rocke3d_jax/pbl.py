"""
JAX Implementation of ROCKE-3D PBL (Planetary Boundary Layer)
================================================================

This module provides JAX-based implementations of the following PBL subroutines:
- find_dpsim: Monin-Obukhov similarity function for momentum
- find_dpsih: Monin-Obukhov similarity function for heat/moisture
- getcm: Drag coefficient for momentum flux
- getchq: Stanton/Dalton number for heat/moisture flux
- simil: Similarity solutions for wind, temperature, and moisture

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Usage:
    from rocke3d_jax.pbl import simil_jit
    u, t, q, dpsim, dpsih, dpsiq = simil_jit(z, z0m, z0h, z0q, lmonin, ustar, tstar, qstar, tg, qg)
"""

import math

import jax
import jax.numpy as jnp
from jax import jit


# Constants from PBL.f
kappa = 0.4  # von Karman constant
zet1 = 1.0   # Critical Monin-Obukhov length
slope1 = 5.0 # Slope for stable conditions
gamams = 4.7  # Gamma for momentum (stable)
gamamu = 16.0 # Gamma for momentum (unstable)
gamahs = 4.7  # Gamma for heat (stable)
gamahu = 16.0 # Gamma for heat (unstable)
sigma = 1.0   # Sigma for heat/moisture
sigma1 = 1.0  # Sigma1 for heat/moisture
by3 = 1.0 / 3.0
cmax = 1.0    # Max drag coefficient
cmin = 0.001  # Min drag coefficient
zetm = -1.0  # Critical Monin-Obukhov length for unstable conditions


# ---------------------------------------------------------------------------
# Fast elementwise math (Round 2 optimization, 2026-09)
#
# On XLA-CPU, pow(x, 0.25), pow(x, 1/3)/cbrt and arctan are scalar libm calls
# (~50-70 us per 3312 elements) while sqrt/log/exp are vectorized (~4 us). The
# original find_dpsim spent ~285 us per call, almost all in 2x pow(.25),
# 2x arctan and 1x pow(1/3). The replacements below are algebraically the same
# functions, accurate to float32 rounding (verified against the original and
# against real Fortran; see STATUS.md "Round 2 optimization").
# ---------------------------------------------------------------------------

def _atan(x):
    """Vectorized float32-accurate arctan (Cephes atanf: range reduction +
    minimax polynomial). Pure elementwise jnp ops, so XLA vectorizes it."""
    ax = jnp.abs(x)
    big = ax > 2.414213562373095
    mid = ax > 0.4142135623730951
    xr = jnp.where(big, -1.0 / jnp.where(big, ax, 1.0),
                   jnp.where(mid, (ax - 1.0) / (ax + 1.0), ax))
    y0 = jnp.where(big, 1.5707963267948966, jnp.where(mid, 0.7853981633974483, 0.0))
    z = xr * xr
    p = (((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z
         - 3.33329491539e-1) * z * xr + xr
    return jnp.sign(x) * (y0 + p)


def _quarter_pow(a):
    """a**0.25 as sqrt(sqrt(a)) (NaN for a<0, same as pow with a fractional exponent)."""
    return jnp.sqrt(jnp.sqrt(a))


@jit
def find_dpsim(zet: jnp.ndarray, zet0: jnp.ndarray) -> jnp.ndarray:
    """
    Monin-Obukhov similarity function for momentum (find_dpsim in PBL.f).

    Args:
        zet: Non-dimensional height (z / L) for momentum.
        zet0: Non-dimensional roughness height (z0 / L) for momentum.

    Returns:
        dpsim: Similarity function for momentum (dimensionless).
    """
    stable = zet >= 0.0

    # Unstable (zet < 0). For zet <= zetm the "gt" formula is evaluated at
    # zet = zetm (so x -> xm) and the extra strongly-unstable terms are added.
    zc = jnp.maximum(zet, zetm)
    x = _quarter_pow(1.0 - gamamu * zc)
    x0 = _quarter_pow(1.0 - gamamu * zet0)
    term1 = jnp.log(((1 + x) * (1 + x) * (1 + x * x)) /
                    ((1 + x0) * (1 + x0) * (1 + x0 * x0)))
    # 2*(atan(x) - atan(x0)) == 2*atan((x-x0)/(1+x*x0)) for x, x0 > 0
    term3 = 2 * _atan((x - x0) / (1 + x * x0))
    dpsim_unstable = term1 - term3
    w = jnp.maximum(-zet, -zetm)          # >= -zetm; = -zet where zet <= zetm
    lw = jnp.log(w)
    extra = (lw - math.log(-zetm)) - 1.140125 * (jnp.exp(lw * by3) - (-zetm) ** by3)
    dpsim_unstable = dpsim_unstable + jnp.where(zet <= zetm, extra, 0.0)

    # Stable: 0 <= zet <= zet1 and zet > zet1
    lstab = jnp.log(jnp.maximum(zet, zet1) / zet1)
    dpsim_s2 = (-gamams * (zet1 - zet0) + zet1 * (slope1 - gamams) * lstab
                - slope1 * (zet - zet1))

    return jnp.where(
        stable & (zet <= zet1),
        -gamams * (zet - zet0),
        jnp.where(stable, dpsim_s2, dpsim_unstable),
    )


@jit
def find_dpsih(zet: jnp.ndarray, zet0: jnp.ndarray, z: jnp.ndarray, z0: jnp.ndarray,
               logzz0=None) -> jnp.ndarray:
    """
    Monin-Obukhov similarity function for heat/moisture (find_dpsih in PBL.f).

    Args:
        zet: Non-dimensional height (z / L) for heat/moisture.
        zet0: Non-dimensional roughness height (z0 / L) for heat/moisture.
        z: Height (m).
        z0: Roughness height (m).
        logzz0: optional precomputed log(z / z0) (loop-invariant in callers).

    Returns:
        dpsih: Similarity function for heat/moisture (dimensionless).
    """
    stable = zet >= 0.0
    if logzz0 is None:
        logzz0 = jnp.log(z / z0)
    lstab = jnp.log(jnp.maximum(zet, zet1) / zet1)
    return jnp.where(
        stable & (zet <= zet1),
        sigma1 * logzz0 - sigma * gamahs * (zet - zet0),
        jnp.where(
            stable,
            sigma1 * jnp.log(zet1 / z0) - sigma * gamahs * (zet1 - zet0) +
            (1 + sigma * (zet1 * (slope1 - gamahs) - 1)) * lstab -
            sigma * slope1 * (zet - zet1),
            # Unstable conditions (zet < 0)
            sigma1 * logzz0 - sigma * gamahu * (zet - zet0),
        ),
    )


@jit
def getcm(z: jnp.ndarray, z0: jnp.ndarray, lmonin: jnp.ndarray, logzz0=None
          ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Drag coefficient for momentum flux (getcm in PBL.f).

    Args:
        z: Height (m).
        z0: Roughness height for momentum (m).
        lmonin: Monin-Obukhov length (m).
        logzz0: optional precomputed log(z / z0).

    Returns:
        dm: Logarithmic term for momentum.
        dpsim: Similarity function for momentum.
        cm: Drag coefficient (dimensionless).
    """
    zet = z / lmonin
    zet0 = z0 / lmonin
    dpsim = find_dpsim(zet, zet0)
    if logzz0 is None:
        logzz0 = jnp.log(z / z0)
    dm = jnp.maximum(logzz0 - dpsim, 1e-3)
    cm = (kappa ** 2) / (dm ** 2)
    cm = jnp.clip(cm, cmin, cmax)
    return dm, dpsim, cm


@jit
def getchq(z: jnp.ndarray, z0: jnp.ndarray, lmonin: jnp.ndarray, dm: jnp.ndarray,
           logzz0=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Stanton/Dalton number for heat/moisture flux (getchq in PBL.f).

    Args:
        z: Height (m).
        z0: Roughness height for heat/moisture (m).
        lmonin: Monin-Obukhov length (m).
        dm: Logarithmic term for momentum (from getcm).
        logzz0: optional precomputed log(z / z0).

    Returns:
        dpsih: Similarity function for heat/moisture.
        ch: Stanton/Dalton number (dimensionless).
    """
    zet = z / lmonin
    zet0 = z0 / lmonin
    if logzz0 is None:
        logzz0 = jnp.log(z / z0)
    dpsih = find_dpsih(zet, zet0, z, z0, logzz0)
    dh = jnp.maximum(logzz0 - dpsih, 1e-3)
    ch = (kappa ** 2) / (dm * dh)
    ch = jnp.clip(ch, cmin, cmax)
    return dpsih, ch


@jit
def simil(z: jnp.ndarray, z0m: jnp.ndarray, z0h: jnp.ndarray, z0q: jnp.ndarray,
           lmonin: jnp.ndarray, ustar: jnp.ndarray, tstar: jnp.ndarray, qstar: jnp.ndarray,
           tg: jnp.ndarray, qg: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                                       jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of simil (Monin-Obukhov similarity solutions).
    
    Args:
        z: Height above ground (m).
        z0m: Momentum roughness height (m).
        z0h: Temperature roughness height (m).
        z0q: Moisture roughness height (m).
        lmonin: Monin-Obukhov length (m).
        ustar: Friction speed (m/s).
        tstar: Temperature scale (K).
        qstar: Moisture scale (kg/kg).
        tg: Ground temperature (K).
        qg: Ground moisture mixing ratio (kg/kg).
    
    Returns:
        u: Wind speed at height z (m/s).
        t: Virtual potential temperature at height z (K).
        q: Moisture mixing ratio at height z (kg/kg).
        dpsim: Similarity function for momentum.
        dpsih: Similarity function for heat.
        dpsiq: Similarity function for moisture.
    """
    # Compute drag coefficients (log(z/z0*) computed once each and reused)
    lzm = jnp.log(z / z0m)
    lzh = jnp.log(z / z0h)
    lzq = jnp.log(z / z0q)
    dm, dpsim, _ = getcm(z, z0m, lmonin, lzm)
    dpsih, _ = getchq(z, z0h, lmonin, dm, lzh)
    dpsiq, _ = getchq(z, z0q, lmonin, dm, lzq)

    # Compute similarity solutions
    u = (ustar / kappa) * (lzm - dpsim)
    t = tg + (tstar / kappa) * (lzh - dpsih)
    q = qg + (qstar / kappa) * (lzq - dpsiq)
    
    return u, t, q, dpsim, dpsih, dpsiq


# JIT-compiled versions for performance
find_dpsim_jit = jit(find_dpsim)
find_dpsih_jit = jit(find_dpsih)
getcm_jit = jit(getcm)
getchq_jit = jit(getchq)
simil_jit = simil  # already @jit
