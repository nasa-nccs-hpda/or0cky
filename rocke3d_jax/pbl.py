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


@jit
def find_dpsim(zet: jnp.ndarray, zet0: jnp.ndarray) -> jnp.ndarray:
    """
    JAX implementation of find_dpsim (Monin-Obukhov similarity for momentum).
    
    Args:
        zet: Non-dimensional height (z / L) for momentum.
        zet0: Non-dimensional roughness height (z0 / L) for momentum.
    
    Returns:
        dpsim: Similarity function for momentum (dimensionless).
    """
    # Stable conditions (zet >= 0)
    stable = zet >= 0.0
    
    # Unstable branch: zet < 0
    x = (1.0 - gamamu * zet)**0.25
    x0 = (1.0 - gamamu * zet0)**0.25
    xm = (1.0 - gamamu * zetm)**0.25
    
    # Unstable: zet > zetm
    term1_unstable = jnp.log(((1 + x) * (1 + x) * (1 + x * x)) / 
               ((1 + x0) * (1 + x0) * (1 + x0 * x0)))
    term3_unstable = 2 * (jnp.arctan(x) - jnp.arctan(x0))
    dpsim_unstable_gt = term1_unstable - term3_unstable
    
    # Unstable: zet <= zetm
    term1_unstable_le = jnp.log(((1 + xm) * (1 + xm) * (1 + xm * xm)) / 
                   ((1 + x0) * (1 + x0) * (1 + x0 * x0)))
    term3_unstable_le = 2 * (jnp.arctan(xm) - jnp.arctan(x0))
    term4_unstable_le = jnp.log(zet / zetm)
    term5_unstable_le = 1.140125 * ((-zet)**by3 - (-zetm)**by3)
    dpsim_unstable_le = term1_unstable_le - term3_unstable_le + term4_unstable_le - term5_unstable_le
    
    dpsim = jnp.where(
        stable & (zet <= zet1),
        -gamams * (zet - zet0),
        jnp.where(
            stable,
            -gamams * (zet1 - zet0) +
            zet1 * (slope1 - gamams) * jnp.log(zet / zet1) -
            slope1 * (zet - zet1),
            jnp.where(
                zet > zetm,
                dpsim_unstable_gt,
                dpsim_unstable_le
            )
        )
    )
    return dpsim


@jit
def find_dpsih(zet: jnp.ndarray, zet0: jnp.ndarray, z: jnp.ndarray, z0: jnp.ndarray) -> jnp.ndarray:
    """
    JAX implementation of find_dpsih (Monin-Obukhov similarity for heat/moisture).
    
    Args:
        zet: Non-dimensional height (z / L) for heat/moisture.
        zet0: Non-dimensional roughness height (z0 / L) for heat/moisture.
        z: Height (m).
        z0: Roughness height (m).
    
    Returns:
        dpsih: Similarity function for heat/moisture (dimensionless).
    """
    # Stable conditions (zet >= 0)
    stable = zet >= 0.0
    dpsih = jnp.where(
        stable & (zet <= zet1),
        sigma1 * jnp.log(z / z0) - sigma * gamahs * (zet - zet0),
        jnp.where(
            stable,
            sigma1 * jnp.log(zet1 / z0) - sigma * gamahs * (zet1 - zet0) +
            (1 + sigma * (zet1 * (slope1 - gamahs) - 1)) * jnp.log(zet / zet1) -
            sigma * slope1 * (zet - zet1),
            # Unstable conditions (zet < 0)
            sigma1 * jnp.log(z / z0) - sigma * gamahu * (zet - zet0)
        )
    )
    return dpsih


@jit
def getcm(z: jnp.ndarray, z0: jnp.ndarray, lmonin: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of getcm (drag coefficient for momentum flux).
    
    Args:
        z: Height (m).
        z0: Roughness height for momentum (m).
        lmonin: Monin-Obukhov length (m).
    
    Returns:
        dm: Logarithmic term for momentum.
        dpsim: Similarity function for momentum.
        cm: Drag coefficient (dimensionless).
    """
    zet = z / lmonin
    zet0 = z0 / lmonin
    dpsim = find_dpsim(zet, zet0)
    dm = jnp.maximum(jnp.log(z / z0) - dpsim, 1e-3)
    cm = (kappa ** 2) / (dm ** 2)
    cm = jnp.clip(cm, cmin, cmax)
    return dm, dpsim, cm


@jit
def getchq(z: jnp.ndarray, z0: jnp.ndarray, lmonin: jnp.ndarray, dm: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of getchq (Stanton/Dalton number for heat/moisture flux).
    
    Args:
        z: Height (m).
        z0: Roughness height for heat/moisture (m).
        lmonin: Monin-Obukhov length (m).
        dm: Logarithmic term for momentum (from getcm).
    
    Returns:
        dpsih: Similarity function for heat/moisture.
        ch: Stanton/Dalton number (dimensionless).
    """
    zet = z / lmonin
    zet0 = z0 / lmonin
    dpsih = find_dpsih(zet, zet0, z, z0)
    dh = jnp.maximum(jnp.log(z / z0) - dpsih, 1e-3)
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
    # Compute drag coefficients
    dm, dpsim, _ = getcm(z, z0m, lmonin)
    dpsih, _ = getchq(z, z0h, lmonin, dm)
    dpsiq, _ = getchq(z, z0q, lmonin, dm)
    
    # Compute similarity solutions
    u = (ustar / kappa) * (jnp.log(z / z0m) - dpsim)
    t = tg + (tstar / kappa) * (jnp.log(z / z0h) - dpsih)
    q = qg + (qstar / kappa) * (jnp.log(z / z0q) - dpsiq)
    
    return u, t, q, dpsim, dpsih, dpsiq


# JIT-compiled versions for performance
find_dpsim_jit = jit(find_dpsim)
find_dpsih_jit = jit(find_dpsih)
getcm_jit = jit(getcm)
getchq_jit = jit(getchq)
simil_jit = jit(simil)
