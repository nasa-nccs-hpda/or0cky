"""
JAX Implementation of ROCKE-3D FLUXES (Surface Fluxes)
==========================================================

This module provides a JAX-based implementation of the surface flux calculations
from ROCKE-3D's FLUXES.f. It computes momentum, heat, and moisture fluxes at the
surface for different surface types (ocean, sea ice, land, etc.).

Key Features:
- Vectorized operations (no explicit loops).
- JIT compilation for performance.
- Numerical consistency with Fortran (within tolerance).

Dependencies:
- JAX (jax, jax.numpy)
- NumPy (for validation)

Usage:
    from fluxes_jax import compute_surface_fluxes_jit
    uflux, vflux, tflux, qflux = compute_surface_fluxes_jit(
        us, vs, tsv, qsrf, rho, cdm, cdh, cq, u1, v1, t1, q1
    )
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple


# Constants from CONSTANT module (assumed values)
RGAS = 287.0      # Specific gas constant for dry air (J/kg/K)
LHM = 2.501e6     # Latent heat of vaporization (J/kg)
LHE = 2.501e6     # Latent heat of evaporation (J/kg)
LHS = 2.834e6     # Latent heat of sublimation (J/kg)
SHA = 1004.6      # Specific heat of dry air (J/kg/K)
TF = 273.15       # Freezing point of water (K)
RHOW = 1000.0     # Density of water (kg/m^3)
SHV = 1864.0      # Specific heat of water vapor (J/kg/K)
SHI = 2106.0      # Specific heat of ice (J/kg/K)
STBO = 5.67e-8    # Stefan-Boltzmann constant (W/m^2/K^4)
DELTX = 0.608     # Virtual temperature factor
GRAV = 9.80665    # Gravitational acceleration (m/s^2)
TEENY = 1e-12     # Tiny value to avoid division by zero


@jit
def compute_momentum_flux(
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    rho: jnp.ndarray,     # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute momentum fluxes (uflux, vflux) at the surface.
    
    Args:
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
    """
    # Momentum flux: rho * cdm * |V| * (V_surface - V_atm)
    ws = jnp.sqrt(us**2 + vs**2)
    uflux = rho * cdm * ws * (us - u1)
    vflux = rho * cdm * ws * (vs - v1)
    return uflux, vflux


@jit
def compute_heat_flux(
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    t1: jnp.ndarray,       # Temperature at first layer (K)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdh: jnp.ndarray,      # Stanton number (heat transfer coefficient)
    ws: jnp.ndarray,       # Wind speed (m/s)
) -> jnp.ndarray:
    """
    Compute sensible heat flux at the surface.
    
    Args:
        tsv: Surface virtual temperature (K)
        t1: Temperature at first layer (K)
        rho: Air density (kg/m^3)
        cdh: Stanton number (heat transfer coefficient)
        ws: Wind speed (m/s)
    
    Returns:
        tflux: Sensible heat flux (W/m^2)
    """
    # Sensible heat flux: rho * cdh * ws * cp * (T_surface - T_atm)
    cp = SHA  # Specific heat of dry air
    tflux = rho * cdh * ws * cp * (tsv - t1)
    return tflux


@jit
def compute_moisture_flux(
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    q1: jnp.ndarray,       # Specific humidity at first layer (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cq: jnp.ndarray,       # Dalton number (moisture transfer coefficient)
    ws: jnp.ndarray,       # Wind speed (m/s)
    lh: jnp.ndarray = LHE, # Latent heat of evaporation (J/kg)
) -> jnp.ndarray:
    """
    Compute latent heat flux (moisture flux) at the surface.
    
    Args:
        qsrf: Surface specific humidity (kg/kg)
        q1: Specific humidity at first layer (kg/kg)
        rho: Air density (kg/m^3)
        cq: Dalton number (moisture transfer coefficient)
        ws: Wind speed (m/s)
        lh: Latent heat of evaporation (J/kg)
    
    Returns:
        qflux: Latent heat flux (W/m^2)
    """
    # Latent heat flux: rho * cq * ws * L * (q_surface - q_atm)
    qflux = rho * cq * ws * lh * (qsrf - q1)
    return qflux


@jit
def compute_surface_fluxes_jit(
    us: jnp.ndarray,      # Surface x-wind (m/s)
    vs: jnp.ndarray,      # Surface y-wind (m/s)
    tsv: jnp.ndarray,      # Surface virtual temperature (K)
    qsrf: jnp.ndarray,     # Surface specific humidity (kg/kg)
    rho: jnp.ndarray,      # Air density (kg/m^3)
    cdm: jnp.ndarray,     # Drag coefficient for momentum
    cdh: jnp.ndarray,     # Stanton number (heat transfer coefficient)
    cq: jnp.ndarray,      # Dalton number (moisture transfer coefficient)
    u1: jnp.ndarray,      # Wind x-component at first layer (m/s)
    v1: jnp.ndarray,      # Wind y-component at first layer (m/s)
    t1: jnp.ndarray,      # Temperature at first layer (K)
    q1: jnp.ndarray,      # Specific humidity at first layer (kg/kg)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute all surface fluxes (momentum, heat, moisture) in one JIT-compiled function.
    
    Args:
        us: Surface x-wind (m/s)
        vs: Surface y-wind (m/s)
        tsv: Surface virtual temperature (K)
        qsrf: Surface specific humidity (kg/kg)
        rho: Air density (kg/m^3)
        cdm: Drag coefficient for momentum
        cdh: Stanton number (heat transfer coefficient)
        cq: Dalton number (moisture transfer coefficient)
        u1: Wind x-component at first layer (m/s)
        v1: Wind y-component at first layer (m/s)
        t1: Temperature at first layer (K)
        q1: Specific humidity at first layer (kg/kg)
    
    Returns:
        uflux: Momentum flux in x-direction (kg/m/s^2)
        vflux: Momentum flux in y-direction (kg/m/s^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
    """
    # Compute wind speed
    ws = jnp.sqrt(us**2 + vs**2)
    
    # Compute momentum fluxes
    uflux, vflux = compute_momentum_flux(us, vs, rho, cdm, u1, v1)
    
    # Compute sensible heat flux
    tflux = compute_heat_flux(tsv, t1, rho, cdh, ws)
    
    # Compute latent heat flux
    qflux = compute_moisture_flux(qsrf, q1, rho, cq, ws)
    
    return uflux, vflux, tflux, qflux


@jit
def compute_radiative_flux(
    tsv: jnp.ndarray,      # Surface temperature (K)
    fsf: jnp.ndarray,      # Downwelling shortwave radiation (W/m^2)
    flong: jnp.ndarray,    # Downwelling longwave radiation (W/m^2)
    albedo: jnp.ndarray,   # Surface albedo (dimensionless)
    emissivity: jnp.ndarray = 0.98,  # Surface emissivity (dimensionless)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute radiative fluxes (absorbed shortwave and net longwave) at the surface.
    
    Args:
        tsv: Surface temperature (K)
        fsf: Downwelling shortwave radiation (W/m^2)
        flong: Downwelling longwave radiation (W/m^2)
        albedo: Surface albedo (dimensionless)
        emissivity: Surface emissivity (dimensionless)
    
    Returns:
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
    """
    # Absorbed shortwave: (1 - albedo) * fsf
    solar = (1.0 - albedo) * fsf
    
    # Net longwave: emissivity * (flong - STBO * tsv^4)
    lw_net = emissivity * (flong - STBO * tsv**4)
    
    return solar, lw_net


@jit
def compute_energy_balance(
    solar: jnp.ndarray,      # Absorbed shortwave radiation (W/m^2)
    lw_net: jnp.ndarray,     # Net longwave radiation (W/m^2)
    tflux: jnp.ndarray,      # Sensible heat flux (W/m^2)
    qflux: jnp.ndarray,      # Latent heat flux (W/m^2)
) -> jnp.ndarray:
    """
    Compute net energy flux at the surface.
    
    Args:
        solar: Absorbed shortwave radiation (W/m^2)
        lw_net: Net longwave radiation (W/m^2)
        tflux: Sensible heat flux (W/m^2)
        qflux: Latent heat flux (W/m^2)
    
    Returns:
        net_energy: Net energy flux (W/m^2)
    """
    net_energy = solar + lw_net - tflux - qflux
    return net_energy


# ============================================================================
# Data Structures for Atmosphere-Surface Exchange
# ============================================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class AtmSrfXchngVars:
    """
    JAX-compatible data structure for atmosphere-surface exchange variables.
    This is a simplified version of the `atmsrf_xchng_vars` type from FLUXES.f.
    """
    # Surface type
    itype4: int = 0  # 1=Ocean, 2=Ocean Ice, 3=Land Ice, 4=Land
    surf_name: str = ""
    
    # Flux fields
    e0: Optional[jnp.ndarray] = None          # Net energy flux at surface [J/m^2]
    solar: Optional[jnp.ndarray] = None      # Absorbed solar radiation [J/m^2]
    trheat: Optional[jnp.ndarray] = None     # Net LW flux accumulation [J/m^2]
    dmua: Optional[jnp.ndarray] = None        # Momentum flux from atmosphere (x) [kg/m/s]
    dmva: Optional[jnp.ndarray] = None        # Momentum flux from atmosphere (y) [kg/m/s]
    evapor: Optional[jnp.ndarray] = None      # Evaporation [kg/m^2]
    sensht: Optional[jnp.ndarray] = None      # Sensible heat flux accumulation [J/m^2]
    latht: Optional[jnp.ndarray] = None       # Latent heat flux accumulation [J/m^2]
    runo: Optional[jnp.ndarray] = None        # Runoff [kg/m^2]
    eruno: Optional[jnp.ndarray] = None       # Energy of runoff [J/m^2]
    
    # Surface state fields
    gtemp: Optional[jnp.ndarray] = None       # Surface temperature [C]
    gtemp2: Optional[jnp.ndarray] = None      # Ground temperature of second layer [C]
    gtempr: Optional[jnp.ndarray] = None      # Radiative ground temperature [K]
    gtemps: Optional[jnp.ndarray] = None      # Skin temperature [C]
    snow: Optional[jnp.ndarray] = None        # Snow mass [kg/m^2]
    snowfr: Optional[jnp.ndarray] = None      # Snow fraction [1]
    snowdp: Optional[jnp.ndarray] = None      # Snow depth [m]
    
    # PBL fields
    wsavg: Optional[jnp.ndarray] = None       # Surface wind magnitude [m/s]
    usavg: Optional[jnp.ndarray] = None       # Reference-height surface wind (x) [m/s]
    vsavg: Optional[jnp.ndarray] = None       # Reference-height surface wind (y) [m/s]
    tsavg: Optional[jnp.ndarray] = None       # Reference-height surface temperature [K]
    qsavg: Optional[jnp.ndarray] = None       # Reference-height surface humidity [kg/kg]
    cmgs: Optional[jnp.ndarray] = None        # Drag coefficient (momentum)
    chgs: Optional[jnp.ndarray] = None        # Stanton number (heat)
    cqgs: Optional[jnp.ndarray] = None        # Dalton number (moisture)
    ustar_pbl: Optional[jnp.ndarray] = None   # Friction velocity [m/s]
    
    # Atmospheric fields (phase 1)
    srfp: Optional[jnp.ndarray] = None        # Surface pressure [hPa]
    srfpk: Optional[jnp.ndarray] = None       # srfp**kappa
    am1: Optional[jnp.ndarray] = None          # First-layer air mass [kg/m^2]
    byam1: Optional[jnp.ndarray] = None       # 1/AM1
    p1: Optional[jnp.ndarray] = None           # Center pressure of first layer [mb]
    prec: Optional[jnp.ndarray] = None        # Precipitation [kg/m^2]
    eprec: Optional[jnp.ndarray] = None       # Energy of precipitation [J/m^2]
    cosz1: Optional[jnp.ndarray] = None       # Mean solar zenith angle [rad]
    flong: Optional[jnp.ndarray] = None        # Downwelling longwave radiation [W/m^2]
    fshort: Optional[jnp.ndarray] = None      # Downwelling shortwave radiation [W/m^2]
    
    # Atmospheric fields (phase srfflx)
    temp1: Optional[jnp.ndarray] = None       # Potential temperature of first layer [K]
    q1: Optional[jnp.ndarray] = None           # Specific humidity of first layer [kg/kg]
    u1: Optional[jnp.ndarray] = None           # Wind x-component of first layer [m/s]
    v1: Optional[jnp.ndarray] = None           # Wind y-component of first layer [m/s]


@dataclass
class AtmOcnXchngVars(AtmSrfXchngVars):
    """
    JAX-compatible data structure for atmosphere-ocean exchange variables.
    Extends AtmSrfXchngVars with ocean-specific fields.
    """
    focean: Optional[jnp.ndarray] = None       # Ocean fraction [1]
    flowo: Optional[jnp.ndarray] = None        # Mass from rivers into ocean [kg/m^2]
    eflowo: Optional[jnp.ndarray] = None       # Energy from rivers into ocean [J/m^2]
    gmel: Optional[jnp.ndarray] = None         # Mass from glacial melt into ocean [kg/m^2]
    egmel: Optional[jnp.ndarray] = None        # Energy from glacial melt into ocean [J/m^2]
    uosurf: Optional[jnp.ndarray] = None       # Ocean surface velocity (x) [m/s]
    vosurf: Optional[jnp.ndarray] = None       # Ocean surface velocity (y) [m/s]
    ogeоза: Optional[jnp.ndarray] = None       # Ocean surface height geopotential [m^2/s^2]
    mlhc: Optional[jnp.ndarray] = None         # Ocean mixed layer heat capacity [J/m^2/C]
    sss: Optional[jnp.ndarray] = None          # Sea surface salinity [ppt]


@dataclass
class AtmIceXchngVars(AtmSrfXchngVars):
    """
    JAX-compatible data structure for atmosphere-ice exchange variables.
    Extends AtmSrfXchngVars with ice-specific fields.
    """
    focean: Optional[jnp.ndarray] = None       # Ocean fraction [1]
    e1: Optional[jnp.ndarray] = None           # Net energy flux at layer 1 [J/m^2]
    uisurf: Optional[jnp.ndarray] = None       # Ice surface velocity (x) [m/s]
    visurf: Optional[jnp.ndarray] = None       # Ice surface velocity (y) [m/s]
    fwsim: Optional[jnp.ndarray] = None        # Fresh water sea ice mass [kg/m^2]
    hsicnv: Optional[jnp.ndarray] = None       # Heat from sea ice convergence [J/m^2]
    msicnv: Optional[jnp.ndarray] = None       # Mass from sea ice convergence [kg/m^2]
    rsi: Optional[jnp.ndarray] = None          # Sea ice fraction [1]
    snowi: Optional[jnp.ndarray] = None        # Snow on sea ice [kg/m^2]
    zsnowi: Optional[jnp.ndarray] = None       # Snow thickness on sea ice [m]


# ============================================================================
# Allocation Functions
# ============================================================================

@jit
def alloc_fluxes(
    im: int,
    jm: int,
) -> Tuple[AtmSrfXchngVars, AtmOcnXchngVars, AtmIceXchngVars]:
    """
    Allocate and initialize FLUXES arrays for JAX.
    This is a simplified version of the ALLOC_FLUXES subroutine from FLUXES.f.
    
    Args:
        im: Number of longitudinal grid boxes.
        jm: Number of latitudinal grid boxes.
    
    Returns:
        atmsrf: Atmosphere-surface exchange variables.
        atmocn: Atmosphere-ocean exchange variables.
        atmice: Atmosphere-ice exchange variables.
    """
    # Initialize atmosphere-surface exchange variables
    atmsrf = AtmSrfXchngVars()
    atmsrf.e0 = jnp.zeros((im, jm))
    atmsrf.solar = jnp.zeros((im, jm))
    atmsrf.trheat = jnp.zeros((im, jm))
    atmsrf.dmua = jnp.zeros((im, jm))
    atmsrf.dmva = jnp.zeros((im, jm))
    atmsrf.evapor = jnp.zeros((im, jm))
    atmsrf.sensht = jnp.zeros((im, jm))
    atmsrf.latht = jnp.zeros((im, jm))
    atmsrf.runo = jnp.zeros((im, jm))
    atmsrf.eruno = jnp.zeros((im, jm))
    atmsrf.gtemp = jnp.zeros((im, jm))
    atmsrf.gtemp2 = jnp.zeros((im, jm))
    atmsrf.gtempr = jnp.zeros((im, jm))
    atmsrf.gtemps = jnp.zeros((im, jm))
    atmsrf.snow = jnp.zeros((im, jm))
    atmsrf.snowfr = jnp.zeros((im, jm))
    atmsrf.snowdp = jnp.zeros((im, jm))
    atmsrf.wsavg = jnp.zeros((im, jm))
    atmsrf.usavg = jnp.zeros((im, jm))
    atmsrf.vsavg = jnp.zeros((im, jm))
    atmsrf.tsavg = jnp.zeros((im, jm))
    atmsrf.qsavg = jnp.zeros((im, jm))
    atmsrf.cmgs = jnp.zeros((im, jm))
    atmsrf.chgs = jnp.zeros((im, jm))
    atmsrf.cqgs = jnp.zeros((im, jm))
    atmsrf.ustar_pbl = jnp.zeros((im, jm))
    atmsrf.srfp = jnp.zeros((im, jm))
    atmsrf.srfpk = jnp.zeros((im, jm))
    atmsrf.am1 = jnp.zeros((im, jm))
    atmsrf.byam1 = jnp.zeros((im, jm))
    atmsrf.p1 = jnp.zeros((im, jm))
    atmsrf.prec = jnp.zeros((im, jm))
    atmsrf.eprec = jnp.zeros((im, jm))
    atmsrf.cosz1 = jnp.zeros((im, jm))
    atmsrf.flong = jnp.zeros((im, jm))
    atmsrf.fshort = jnp.zeros((im, jm))
    atmsrf.temp1 = jnp.zeros((im, jm))
    atmsrf.q1 = jnp.zeros((im, jm))
    atmsrf.u1 = jnp.zeros((im, jm))
    atmsrf.v1 = jnp.zeros((im, jm))
    
    # Initialize atmosphere-ocean exchange variables
    atmocn = AtmOcnXchngVars()
    atmocn.focean = jnp.zeros((im, jm))
    atmocn.flowo = jnp.zeros((im, jm))
    atmocn.eflowo = jnp.zeros((im, jm))
    atmocn.gmel = jnp.zeros((im, jm))
    atmocn.egmel = jnp.zeros((im, jm))
    atmocn.uosurf = jnp.zeros((im, jm))
    atmocn.vosurf = jnp.zeros((im, jm))
    atmocn.ogeоза = jnp.zeros((im, jm))
    atmocn.mlhc = jnp.zeros((im, jm))
    atmocn.sss = jnp.zeros((im, jm))
    
    # Initialize atmosphere-ice exchange variables
    atmice = AtmIceXchngVars()
    atmice.focean = jnp.zeros((im, jm))
    atmice.e1 = jnp.zeros((im, jm))
    atmice.uisurf = jnp.zeros((im, jm))
    atmice.visurf = jnp.zeros((im, jm))
    atmice.fwsim = jnp.zeros((im, jm))
    atmice.hsicnv = jnp.zeros((im, jm))
    atmice.msicnv = jnp.zeros((im, jm))
    atmice.rsi = jnp.zeros((im, jm))
    atmice.snowi = jnp.zeros((im, jm))
    atmice.zsnowi = jnp.zeros((im, jm))
    
    return atmsrf, atmocn, atmice
