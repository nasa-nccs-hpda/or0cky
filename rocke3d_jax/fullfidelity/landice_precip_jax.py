"""Batched JAX port of landice_precip_ff.py (LANDICE_DRV.f PRECIP_LI / LANDICE.f PRECLI) -- Track B, Stage 1
of the DYNSI/ocean port, D28. Both branches computed for every lane and merged with jnp.where, the same
pattern used throughout this project's batched ports."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from landice_precip_ff import LHM, SHI, ACE1LI, ACE2LI, HC1LI


def precli(snow0, tg10, tg20, prcp, enrgp):
    """LANDICE.f PRECLI, batched. Returns dict: snow, tg1, tg2, run0, edifs, difs, erun2."""
    hc1 = HC1LI + snow0 * SHI
    is_rain = enrgp >= 0.0

    # ---- rain branch ----
    heats_to_freezing = enrgp >= -tg10 * hc1
    dwater = (tg10 * hc1 + enrgp) / LHM
    run0_melt = dwater + prcp
    some_snow_left = dwater < snow0
    snow_melt_some = snow0 - dwater
    difs_all = snow0 - dwater
    snow_melt_all = jnp.zeros_like(snow0)
    tg1_melt_all = -tg20 * difs_all / ACE1LI
    edifs_melt_all = difs_all * (tg20 * SHI - LHM)
    snow_A1 = jnp.where(some_snow_left, snow_melt_some, snow_melt_all)
    tg1_A1 = jnp.where(some_snow_left, jnp.zeros_like(tg10), tg1_melt_all)
    difs_A1 = jnp.where(some_snow_left, jnp.zeros_like(snow0), difs_all)
    edifs_A1 = jnp.where(some_snow_left, jnp.zeros_like(snow0), edifs_melt_all)
    erun2_A1 = edifs_A1
    tg2_A1 = tg20
    run0_A1 = run0_melt

    tg1_A2 = tg10 + enrgp / hc1
    run0_A2 = prcp
    snow_A2, tg2_A2, difs_A2, edifs_A2, erun2_A2 = snow0, tg20, jnp.zeros_like(snow0), jnp.zeros_like(snow0), jnp.zeros_like(snow0)

    snow_A = jnp.where(heats_to_freezing, snow_A1, snow_A2)
    tg1_A = jnp.where(heats_to_freezing, tg1_A1, tg1_A2)
    tg2_A = jnp.where(heats_to_freezing, tg2_A1, tg2_A2)
    run0_A = jnp.where(heats_to_freezing, run0_A1, run0_A2)
    difs_A = jnp.where(heats_to_freezing, difs_A1, difs_A2)
    edifs_A = jnp.where(heats_to_freezing, edifs_A1, edifs_A2)
    erun2_A = jnp.where(heats_to_freezing, erun2_A1, erun2_A2)

    # ---- snow branch ----
    tg1_snow = (tg10 * hc1 + enrgp + LHM * prcp) / (hc1 + prcp * SHI)
    snow_snow = snow0 + prcp
    compacts = snow_snow > ACE1LI
    difs_c = snow_snow - 0.9 * ACE1LI
    edifs_c = difs_c * (tg1_snow * SHI - LHM)
    erun2_c = difs_c * (tg20 * SHI - LHM)
    tg2_c = tg20 + (tg1_snow - tg20) * difs_c / ACE2LI
    snow_c = jnp.full_like(snow0, 1.0) * 0.9 * ACE1LI

    snow_B = jnp.where(compacts, snow_c, snow_snow)
    tg2_B = jnp.where(compacts, tg2_c, tg20)
    difs_B = jnp.where(compacts, difs_c, jnp.zeros_like(snow0))
    edifs_B = jnp.where(compacts, edifs_c, jnp.zeros_like(snow0))
    erun2_B = jnp.where(compacts, erun2_c, jnp.zeros_like(snow0))
    tg1_B = tg1_snow
    run0_B = jnp.zeros_like(snow0)

    snow = jnp.where(is_rain, snow_A, snow_B)
    tg1 = jnp.where(is_rain, tg1_A, tg1_B)
    tg2 = jnp.where(is_rain, tg2_A, tg2_B)
    run0 = jnp.where(is_rain, run0_A, run0_B)
    difs = jnp.where(is_rain, difs_A, difs_B)
    edifs = jnp.where(is_rain, edifs_A, edifs_B)
    erun2 = jnp.where(is_rain, erun2_A, erun2_B)
    return dict(snow=snow, tg1=tg1, tg2=tg2, run0=run0, edifs=edifs, difs=difs, erun2=erun2)


def precip_li(ftype, prcp, enrgp, snow0, tg10, tg20):
    """LANDICE_DRV.f PRECIP_LI, batched."""
    active = (ftype > 0.0) & (prcp > 0.0)
    out = precli(snow0, tg10, tg20, prcp, enrgp)
    zeros = jnp.zeros_like(snow0)
    return dict(snow=jnp.where(active, out["snow"], snow0), tg1=jnp.where(active, out["tg1"], tg10),
                tg2=jnp.where(active, out["tg2"], tg20), runo=jnp.where(active, out["run0"], zeros),
                implm=jnp.where(active, out["difs"], zeros), implh=jnp.where(active, out["erun2"], zeros),
                e1=jnp.where(active, out["edifs"], zeros))
