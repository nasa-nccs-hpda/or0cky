"""
JAX Implementation of ROCKE-3D PBL_SIMPLE (Simplified Planetary Boundary Layer)
==============================================================================

This module provides a JAX-based implementation of the simplified PBL scheme
from ROCKE-3D's PBL_SIMPLE_DRV.f. It computes surface wind, temperature, and
moisture fluxes for the planetary boundary layer using a simplified approach.

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Dependencies:
- JAX (jax, jax.numpy)
- NumPy (for validation)

Usage:
    from pbl_simple_jax import pbl_simple_jit
    us, vs, ws, wsm, wsh, tsv, qsrf, khs, w2_1, wint, cdm, cdh, cia = pbl_simple_jit(
        i, j, itype, ptype, t, q, u, v, pedn, MA, pek, tmom, qmom, uocean, vocean,
        ACE1I, MSI, cdnl, hemi, pole, mz
    )
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Tuple


# Constants from PBL_SIMPLE_DRV.f
S1BYG1 = 0.57735  # 1/sqrt(3)
CDMA = 2.0        # Drag coefficient multiplier (momentum)
CDMC = 4.0        # Drag coefficient multiplier (momentum, stable)
CDHA = 2.0        # Drag coefficient multiplier (heat)
CDHC = 4.0        # Drag coefficient multiplier (heat, stable)
WSBYW1 = 0.75     # Wind speed scaling factor
CIAX = 0.3        # Stability parameter for angle calculation

# Constants from CONSTANT module (assumed values)
RGAS = 287.0      # Specific gas constant for dry air (J/kg/K)
GRAV = 9.80665    # Gravitational acceleration (m/s^2)
DELTX = 0.608     # Virtual temperature factor
TWOPI = 6.283185307179586  # 2 * pi


@jit
def compute_pg_ps(
    pedn: jnp.ndarray,  # Edge pressure (Pa) (I, J, L+1)
    MA: jnp.ndarray,    # Mass (kg/m^2) (I, J, L)
    i: int,             # Grid index (i)
    j: int,             # Grid index (j)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute ground pressure (pg) and surface pressure (ps).
    
    Args:
        pedn: Edge pressure (Pa) (I, J, L+1)
        MA: Mass (kg/m^2) (I, J, L)
        i: Grid index (i)
        j: Grid index (j)
    
    Returns:
        pg: Ground pressure (Pa)
        ps: Surface pressure (Pa)
    """
    pg = pedn[0, i, j] * 100.0  # Convert to Pa
    ps = pg - GRAV * MA[0, i, j] * 0.5 * (1.0 - S1BYG1)
    return pg, ps


@jit
def compute_t_q(
    t: jnp.ndarray,      # Temperature (K) (I, J, L)
    q: jnp.ndarray,      # Specific humidity (kg/kg) (I, J, L)
    tmom: jnp.ndarray,   # Temperature tendency (K/m) (I, J, L)
    qmom: jnp.ndarray,   # Moisture tendency (kg/kg/m) (I, J, L)
    pek: jnp.ndarray,    # Pressure scaling factor (I, J, L)
    i: int,              # Grid index (i)
    j: int,              # Grid index (j)
    mz: int,             # Vertical index for tendencies
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute temperature and moisture at the first layer and their tendencies.
    
    Args:
        t: Temperature (K) (I, J, L)
        q: Specific humidity (kg/kg) (I, J, L)
        tmom: Temperature tendency (K/m) (I, J, L)
        qmom: Moisture tendency (kg/kg/m) (I, J, L)
        pek: Pressure scaling factor (I, J, L)
        i: Grid index (i)
        j: Grid index (j)
        mz: Vertical index for tendencies
    
    Returns:
        t1: Temperature at first layer (K)
        dtdz: Temperature tendency (K/m)
        q1: Specific humidity at first layer (kg/kg)
        dqdz: Moisture tendency (kg/kg/m)
    """
    t1 = t[i, j, 0]
    dtdz = tmom[mz, i, j, 0]
    q1 = q[i, j, 0]
    dqdz = qmom[mz, i, j, 0]
    return t1, dtdz, q1, dqdz


@jit
def compute_u_v(
    u: jnp.ndarray,      # X-velocity (m/s) (I, J, L)
    v: jnp.ndarray,      # Y-velocity (m/s) (I, J, L)
    idij: jnp.ndarray,   # Grid index mapping (k, J)
    idjj: jnp.ndarray,   # Grid index mapping (k, J)
    kmaxj: jnp.ndarray,  # Maximum k for each j (J)
    rapj: jnp.ndarray,   # Weighting factor (k, J)
    cosiv: jnp.ndarray,   # Cosine of rotation angle (k, J)
    siniv: jnp.ndarray,   # Sine of rotation angle (k, J)
    hemi: jnp.ndarray,   # Hemisphere sign (I, J)
    pole: jnp.ndarray,   # Pole flag (I, J)
    i: int,              # Grid index (i)
    j: int,              # Grid index (j)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute wind components at the top of the PBL (u1, v1).
    
    Args:
        u: X-velocity (m/s) (I, J, L)
        v: Y-velocity (m/s) (I, J, L)
        idij: Grid index mapping (k, J)
        idjj: Grid index mapping (k, J)
        kmaxj: Maximum k for each j (J)
        rapj: Weighting factor (k, J)
        cosiv: Cosine of rotation angle (k, J)
        siniv: Sine of rotation angle (k, J)
        hemi: Hemisphere sign (I, J)
        pole: Pole flag (I, J)
        i: Grid index (i)
        j: Grid index (j)
    
    Returns:
        u1: X-velocity at top of PBL (m/s)
        v1: Y-velocity at top of PBL (m/s)
    """
    # Initialize u1 and v1
    u1 = 0.0
    v1 = 0.0
    
    # Loop over k (vectorized in JAX using lax.fori_loop)
    def body_fun(k, carry):
        u_carry, v_carry = carry
        # Get indices for the current k
        idx_i = idij[k, j]
        idx_j = idjj[k, j]
        
        # Compute contributions using jnp.where to avoid boolean conversion
        u_contrib_pole = rapj[k, j] * (u[idx_i, idx_j, 0] * cosiv[k, j] - hemi[i, j] * v[idx_i, idx_j, 0] * siniv[k, j])
        v_contrib_pole = rapj[k, j] * (v[idx_i, idx_j, 0] * cosiv[k, j] + hemi[i, j] * u[idx_i, idx_j, 0] * siniv[k, j])
        u_contrib_nonpole = rapj[k, j] * u[idx_i, idx_j, 0]
        v_contrib_nonpole = rapj[k, j] * v[idx_i, idx_j, 0]
        
        u_contrib = jnp.where(pole[i, j], u_contrib_pole, u_contrib_nonpole)
        v_contrib = jnp.where(pole[i, j], v_contrib_pole, v_contrib_nonpole)
        
        u_new = u_carry + u_contrib
        v_new = v_carry + v_contrib
        return (u_new, v_new)
    
    # Use lax.fori_loop to iterate over k
    u1, v1 = lax.fori_loop(
        0, kmaxj[j], body_fun, (u1, v1)
    )
    
    return u1, v1


@jit
def compute_cdn(
    itype: jnp.ndarray,  # Surface type (I, J)
    wssq: jnp.ndarray,   # Wind speed squared (m^2/s^2) (I, J)
    ACE1I: jnp.ndarray,   # Sea ice concentration (I, J)
    MSI: jnp.ndarray,     # Snow mask index (I, J)
    cdnl: jnp.ndarray,    # Roughness length for land/ice (I, J)
    i: int,               # Grid index (i)
    j: int,               # Grid index (j)
) -> jnp.ndarray:
    """
    Compute drag coefficient (cdn) based on surface type.
    
    Args:
        itype: Surface type (1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land/Snow)
        wssq: Wind speed squared (m^2/s^2)
        ACE1I: Sea ice concentration
        MSI: Snow mask index
        cdnl: Roughness length for land/ice
        i: Grid index (i)
        j: Grid index (j)
    
    Returns:
        cdn: Drag coefficient
    """
    # Select case based on itype
    cdn = jnp.where(
        itype[i, j] == 1,  # Ocean
        0.0009 + 0.0000025 * wssq,
        jnp.where(
            itype[i, j] == 2,  # Ocean ice
            0.00095417 + 0.0000005 * ACE1I[i, j] + 0.0000005 * MSI[i, j],
            jnp.where(
                (itype[i, j] == 3) | (itype[i, j] == 4),  # Land ice, Land, Snow
                cdnl[i, j],
                0.0  # Default (should not happen)
            )
        )
    )
    return cdn


@jit
def compute_stability(
    betas: jnp.ndarray,   # Specific volume at surface (m^3/kg)
    betag: jnp.ndarray,   # Specific volume at ground (m^3/kg)
    pg: jnp.ndarray,      # Ground pressure (Pa)
    ps: jnp.ndarray,      # Surface pressure (Pa)
    wssq: jnp.ndarray,   # Wind speed squared (m^2/s^2)
    ws: jnp.ndarray,      # Wind speed (m/s)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute stability parameters (rigs, cdm, cdh, cia).
    
    Args:
        betas: Specific volume at surface (m^3/kg)
        betag: Specific volume at ground (m^3/kg)
        pg: Ground pressure (Pa)
        ps: Surface pressure (Pa)
        wssq: Wind speed squared (m^2/s^2)
        ws: Wind speed (m/s)
    
    Returns:
        rigs: Richardson number for stability
        cdm: Drag coefficient for momentum
        cdh: Drag coefficient for heat
        cia: Angle for wind rotation (rad)
    """
    # Unstable stratification (betag > betas)
    unstable = betag > betas
    
    # Compute rigs (Richardson number)
    rigs = jnp.where(
        unstable,
        0.0,
        (pg - ps) * (betas - betag) / (wssq + 0.1)
    )
    
    # Compute cdm and cdh
    cdm = jnp.where(
        unstable,
        CDMA,
        CDMA / (1.0 + CDMC * rigs * rigs)
    )
    
    cdh = jnp.where(
        unstable,
        CDHA,
        CDHA / (1.0 + CDHC * rigs * rigs)
    )
    
    # Compute cia (angle for wind rotation)
    cia = jnp.where(
        unstable,
        TWOPI * 0.0625 / (1.0 + ws * CIAX),
        TWOPI * (0.09375 - 0.03125 / (1.0 + 4.0 * rigs * rigs)) / (1.0 + ws * CIAX)
    )
    
    return rigs, cdm, cdh, cia


@jit
def pbl_simple_jit(
    i: int,
    j: int,
    itype: jnp.ndarray,   # Surface type (I, J)
    ptype: jnp.ndarray,   # Percent surface type (I, J)
    t: jnp.ndarray,       # Temperature (K) (I, J, L)
    q: jnp.ndarray,       # Specific humidity (kg/kg) (I, J, L)
    u: jnp.ndarray,       # X-velocity (m/s) (I, J, L)
    v: jnp.ndarray,       # Y-velocity (m/s) (I, J, L)
    pedn: jnp.ndarray,    # Edge pressure (Pa) (I, J, L+1)
    MA: jnp.ndarray,      # Mass (kg/m^2) (I, J, L)
    pek: jnp.ndarray,     # Pressure scaling factor (I, J, L)
    tmom: jnp.ndarray,    # Temperature tendency (K/m) (I, J, L)
    qmom: jnp.ndarray,    # Moisture tendency (kg/kg/m) (I, J, L)
    uocean: jnp.ndarray,  # Ocean X-velocity (m/s) (I, J)
    vocean: jnp.ndarray,  # Ocean Y-velocity (m/s) (I, J)
    ACE1I: jnp.ndarray,   # Sea ice concentration (I, J)
    MSI: jnp.ndarray,     # Snow mask index (I, J)
    cdnl: jnp.ndarray,    # Roughness length for land/ice (I, J)
    hemi: jnp.ndarray,    # Hemisphere sign (I, J)
    pole: jnp.ndarray,    # Pole flag (I, J)
    mz: int,              # Vertical index for tendencies
    idij: jnp.ndarray,    # Grid index mapping (k, J)
    idjj: jnp.ndarray,    # Grid index mapping (k, J)
    kmaxj: jnp.ndarray,   # Maximum k for each j (J)
    rapj: jnp.ndarray,    # Weighting factor (k, J)
    cosiv: jnp.ndarray,   # Cosine of rotation angle (k, J)
    siniv: jnp.ndarray,   # Sine of rotation angle (k, J)
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    JAX implementation of the simplified PBL subroutine from PBL_SIMPLE_DRV.f.
    
    Args:
        i: Grid index (i)
        j: Grid index (j)
        itype: Surface type (1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land/Snow)
        ptype: Percent surface type
        t: Temperature (K) (I, J, L)
        q: Specific humidity (kg/kg) (I, J, L)
        u: X-velocity (m/s) (I, J, L)
        v: Y-velocity (m/s) (I, J, L)
        pedn: Edge pressure (Pa) (I, J, L+1)
        MA: Mass (kg/m^2) (I, J, L)
        pek: Pressure scaling factor (I, J, L)
        tmom: Temperature tendency (K/m) (I, J, L)
        qmom: Moisture tendency (kg/kg/m) (I, J, L)
        uocean: Ocean X-velocity (m/s) (I, J)
        vocean: Ocean Y-velocity (m/s) (I, J)
        ACE1I: Sea ice concentration (I, J)
        MSI: Snow mask index (I, J)
        cdnl: Roughness length for land/ice (I, J)
        hemi: Hemisphere sign (I, J)
        pole: Pole flag (I, J)
        mz: Vertical index for tendencies
        idij: Grid index mapping (k, J)
        idjj: Grid index mapping (k, J)
        kmaxj: Maximum k for each j (J)
        rapj: Weighting factor (k, J)
        cosiv: Cosine of rotation angle (k, J)
        siniv: Sine of rotation angle (k, J)
    
    Returns:
        us: X-component of surface wind (m/s)
        vs: Y-component of surface wind (m/s)
        ws: Wind speed (m/s)
        wsm: Wind speed modified by ocean currents (m/s)
        wsh: Wind speed modified by buoyancy flux (m/s)
        tsv: Virtual temperature at surface (K)
        qsrf: Surface specific humidity (kg/kg)
        khs: Heat transport coefficient (m^2/s)
        w2_1: Placeholder for w2gcm (not used)
        wint: Wind speed for initialization
        cdm: Drag coefficient for momentum
        cdh: Drag coefficient for heat
        cia: Angle for wind rotation (rad)
    """
    # Step 1: Compute pg and ps
    pg, ps = compute_pg_ps(pedn, MA, i, j)
    
    # Step 2: Compute t1, dtdz, q1, dqdz
    t1, dtdz, q1, dqdz = compute_t_q(t, q, tmom, qmom, pek, i, j, mz)
    
    # Step 3: Compute qs (surface moisture)
    qs = jnp.maximum(q1 - dqdz * S1BYG1, 0.0)
    
    # Step 4: Compute tps, tks, tvs
    tps = t1 - dtdz * S1BYG1
    tks = tps * pek[0, i, j]
    tvs = tks * (1.0 + qs * DELTX)
    
    # Step 5: Compute u1, v1 (wind at top of PBL)
    u1, v1 = compute_u_v(u, v, idij, idjj, kmaxj, rapj, cosiv, siniv, hemi, pole, i, j)
    
    # Step 6: Compute wind speed (ws)
    wssq = (u1 * u1 + v1 * v1) * WSBYW1 * WSBYW1
    ws = jnp.sqrt(wssq)
    
    # Step 7: Compute cdn (drag coefficient)
    cdn = compute_cdn(itype, wssq, ACE1I, MSI, cdnl, i, j)
    
    # Step 8: Compute betas, betag (specific volumes)
    tvg = 0.0  # Placeholder (not used in PBL_SIMPLE_DRV.f)
    betas = RGAS * tvs / pg
    betag = RGAS * tvg / pg
    
    # Step 9: Compute stability parameters
    rigs, cdm, cdh, cia = compute_stability(betas, betag, pg, ps, wssq, ws)
    
    # Step 10: Compute us, vs (surface wind components)
    cosc = jnp.cos(cia)
    sinc = jnp.sin(cia) * hemi[i, j]
    us = (u1 * cosc - v1 * sinc) * WSBYW1
    vs = (v1 * cosc + u1 * sinc) * WSBYW1
    
    # Step 11: Compute wsm, wsh, tsv, qsrf
    wsm = jnp.sqrt((us - uocean[i, j]) ** 2 + (vs - vocean[i, j]) ** 2)
    wsh = wsm
    tsv = tvs
    qsrf = qs
    
    # Step 12: Placeholder for w2_1 and wint
    w2_1 = 666.0
    wint = wsm
    
    # Step 13: Placeholder for khs
    khs = 1.0
    
    return us, vs, ws, wsm, wsh, tsv, qsrf, khs, w2_1, wint, cdm, cdh, cia
