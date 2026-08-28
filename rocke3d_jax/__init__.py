"""
ROCKE-3D JAX Implementation
============================

This package provides JAX-based implementations of ROCKE-3D's core modules,
optimized for performance on CPU/GPU/TPU.

Modules:
- drycnv: Dry convection mixing (DRYCNV.f)
- pbl: Planetary Boundary Layer (PBL.f)
- radiation: Radiative transfer (RADIATION.f)
- surface: Surface fluxes (SURFACE.f)
- clouds: Cloud microphysics (CLOUDS2.F90)
- dynamical_core: Finite-volume dynamical core (FV_LatLon_Mod.F90)

Usage:
    from rocke3d_jax.drycnv import dry_convection_mixing_jit
    from rocke3d_jax.pbl import simil_jit
"""

# Set JAX to use CPU by default (for compatibility)
import os
os.environ["JAX_PLATFORMS"] = "cpu"

# Import all submodules
from . import drycnv, pbl

__all__ = ["drycnv", "pbl"]
