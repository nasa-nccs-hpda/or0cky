"""Batched JAX port of apress_ff.py (SEAICE_DRV.f CALC_APRESS) -- Stage 1 of the DYNSI/ocean port,
D30. Pure per-cell arithmetic, no branches -- nothing to merge with jnp.where."""
import jax
jax.config.update("jax_enable_x64", True)

from apress_ff import ACE1I, GRAV_DEFAULT


def calc_apress(srfp, rsi, snowi, msi, grav=GRAV_DEFAULT):
    """SEAICE_DRV.f CALC_APRESS, batched. srfp in hPa; returns APRESS in Pa."""
    return 100.0 * (srfp - 1013.25) + rsi * (snowi + ACE1I + msi) * grav
