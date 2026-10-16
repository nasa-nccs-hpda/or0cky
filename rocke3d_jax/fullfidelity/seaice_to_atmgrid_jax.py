"""Batched JAX port of seaice_to_atmgrid_ff.py (SEAICE_DRV.f seaice_to_atmgrid's GTEMP/GTEMP2/
GTEMPR/ZSNOWI/ZSI/FWSIM derivation) -- Stage 1 of the DYNSI/ocean port, D31. The MICE1/SNOWL1
branch (some ice in first layer vs. some snow in second layer) is computed for both lanes and
merged with jnp.where; Ti/Ti2b are already-batched (seaice_core_jax.py)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ice_props_ff import RHOI, RHOS, ACE1I
from seaice_core_jax import Ti, Ti2b
from seaice_core_ff import XSI

TF = 273.15


def seaice_to_atmgrid_cell(rsi, snowi, msi, hsi1, hsi2, ssi1, ssi2, ssi3, ssi4):
    """Batched version of seaice_to_atmgrid_ff.seaice_to_atmgrid_cell."""
    msi1 = snowi + ACE1I

    some_ice_in_layer1 = ACE1I > XSI[1] * msi1
    mice1_a = ACE1I - XSI[1] * msi1
    mice2_a = XSI[1] * msi1
    snowl1_a = msi1 - ACE1I
    snowl2_a = jnp.zeros_like(msi1)

    mice1_b = jnp.zeros_like(msi1)
    mice2_b = jnp.full_like(msi1, ACE1I)
    snowl1_b = XSI[0] * msi1
    snowl2_b = XSI[1] * msi1 - ACE1I

    mice1 = jnp.where(some_ice_in_layer1, mice1_a, mice1_b)
    mice2 = jnp.where(some_ice_in_layer1, mice2_a, mice2_b)
    snowl1 = jnp.where(some_ice_in_layer1, snowl1_a, snowl1_b)
    snowl2 = jnp.where(some_ice_in_layer1, snowl2_a, snowl2_b)

    has_mice1 = mice1 != 0.0
    gtemp_a = Ti2b(hsi1 / (XSI[0] * msi1), 1e3 * ssi1 / jnp.where(has_mice1, mice1, 1.0), snowl1,
                   jnp.where(has_mice1, mice1, 1.0))
    gtemp_b = Ti(hsi1 / (XSI[0] * msi1), jnp.zeros_like(msi1))
    gtemp = jnp.where(has_mice1, gtemp_a, gtemp_b)
    gtemp2 = Ti2b(hsi2 / (XSI[1] * msi1), 1e3 * ssi2 / mice2, snowl2, mice2)

    gtempr = gtemp + TF
    zsnowi = snowi / RHOS
    zsi = (ACE1I + msi) / RHOI
    fwsim = rsi * (msi1 + msi - (ssi1 + ssi2 + ssi3 + ssi4))

    return dict(gtemp=gtemp, gtemp2=gtemp2, gtempr=gtempr, zsnowi=zsnowi, zsi=zsi, fwsim=fwsim)
