"""Full-fidelity port of SEAICE_DRV.f CALC_APRESS -- Stage 1 of the DYNSI/ocean port, D30. Computes
the total atmosphere+sea-ice pressure anomaly at the ocean surface (used as a forcing input to
DYNSI's pressure-gradient term, OGEOZA-free approximation). Pure per-cell arithmetic, no branches.

GRAV is a *runtime* planet parameter under USE_PLANET_RAD (like RADIUS in icedyn_geom_ff.py) --
callers should infer it from a real dump rather than assume the hardcoded 9.80665 m/s^2 default.
"""
from ice_props_ff import ACE1I

GRAV_DEFAULT = 9.80665


def calc_apress(srfp, rsi, snowi, msi, grav=GRAV_DEFAULT):
    """SEAICE_DRV.f CALC_APRESS. srfp in hPa; returns APRESS in Pa."""
    return 100.0 * (srfp - 1013.25) + rsi * (snowi + ACE1I + msi) * grav
