"""D186 (stage S2): MELT_SI (ATM_DRV.f:257-258, SEAICE_DRV.f:377-562, both domains) as a device program.

Same arithmetic as surface_loop.melt_si: seaice_core_jax.simelt on the cells of the mask `dop`; there the NumPy driver gathers the cells with np.nonzero, here simelt
runs on the whole (IM,JM) grid and the results are selected with jnp.where (simelt is elementwise, so each selected cell sees the same operations).
Inputs: ice dict (rsi, snowi, msi (IM,JM); hsi, ssi (IM,JM,4); pond_melt, flag_dsws passed through), gtemp, sss, mlhc (IM,JM), geo (fwater, is_ocean, valid).
Not changed by MELT_SI: the atmosphere.  CONDSE reads RSI of the result (the NumPy chain reads the recorded CONDSE-entry RSI; the test compares both).
"""
import clouds_jax_env_fast  # noqa: F401
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import seaice_core_jax as SI

IM, JM = 72, 46


def geo_device(geo):
    return dict(fwater=jnp.asarray(geo['fwater']), is_ocean=jnp.asarray(geo['is_ocean']), valid=jnp.asarray(geo['valid']))


@jax.jit
def melt_si(ice, gtemp, sss, mlhc, geo, dtsrc):
    fw, oc, valid = geo['fwater'], geo['is_ocean'], geo['valid']
    roice = ice['rsi']
    dop = (fw * roice > 0) & (oc | (roice < 1.0)) & valid
    pocean = jnp.where(oc, fw, 0.0)
    tfo = jnp.where(oc, SI.tfrez(sss), 0.0)
    enrgmax = jnp.maximum(gtemp - tfo, 0.0) * mlhc
    r = SI.simelt(dtsrc, roice, ice['snowi'], ice['msi'], ice['hsi'], ice['ssi'], pocean, gtemp, tfo, enrgmax)
    d3 = dop[:, :, None]
    new = dict(ice)
    new['rsi'] = jnp.where(dop, r['roice'], ice['rsi'])
    new['msi'] = jnp.where(dop, r['msi2'], ice['msi'])
    new['snowi'] = jnp.where(dop, r['snow'], ice['snowi'])
    new['hsi'] = jnp.where(d3, r['hsil'], ice['hsi'])
    new['ssi'] = jnp.where(d3, r['ssil'], ice['ssi'])
    for j in (0, JM - 1):                                     # _pole_replicate: ocean-domain pole rows from i = 1
        flag = oc[0, j]
        for k in ('rsi', 'snowi', 'msi'):
            new[k] = new[k].at[1:, j].set(jnp.where(flag, new[k][0, j], new[k][1:, j]))
        for k in ('hsi', 'ssi'):
            new[k] = new[k].at[1:, j, :].set(jnp.where(flag, new[k][0, j, :][None, :], new[k][1:, j, :]))
    melt = dict(melti=jnp.where(dop, r['run0'] * fw, 0.0), emelti=jnp.where(dop, -r['enrgused'] * fw, 0.0),
                smelti=jnp.where(dop, r['salt'] * fw, 0.0), dop=dop)
    return new, melt
