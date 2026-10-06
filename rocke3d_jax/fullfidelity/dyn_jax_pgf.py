"""D139: PGF in JAX (jitted port of dyn_pgf_ff.pgf, numpy-pow semantics).

  * the top-down pressure/geopotential column recursion (loop over L, carrying M,PU,PKU,PKPU,PKPPU) is a lax.scan over
    L = LM..1 with the numpy body, the bottom-up GZ integration is a lax.scan over L = 1..LM;
  * x**KAPA is jnp.power (on this CPU backend it reproduced numpy's pow in the D139 test; the Intel libimf
    bitwise mode of the numpy port cannot be reproduced in JAX and is NOT claimed for this path);
  * everything after the recursion (N-S derivative, E-W derivative, AVRX through dyn_jax_fft, polar scaling, UT/VT
    update) is whole-field jnp with the numpy operation order.
"""
import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from dyn_aflux_ff import IM, JM, LM
import dyn_jax_fft as jf


def pgf_geo(g, tab):
    d = {k: jnp.asarray(np.asarray(g[k], dtype=np.float64)) for k in ('dxv', 'dyv', 'dxyn', 'dxys', 'zatmo', 'acor', 'acor2')}
    for k in ('kapa', 'grav', 'rgas', 'bykapa', 'bykapap1', 'bykapap2', 'bygrav', 'mtop'):
        d[k] = jnp.asarray(float(g[k]))
    act, coef, mask = jf.avrx_plan(range(1, JM - 1), g, tab)
    d['avrx_coef'] = jnp.asarray(coef); d['avrx_mask'] = jnp.asarray(mask)
    return d, act


def make_pgf(g, tab):
    """-> (pgf_jax(dt1, mam, ut, vt, mafter, s0, sz, dut, dvt) -> dict, geo)."""
    geo, act = pgf_geo(g, tab)
    assert g['do_polefix'] == 1

    @jax.jit
    def pgf_jax(dt1, mam, ut, vt, mafter, s0, sz, dut, dvt, geo):
        # scalar constants are traced arguments (not closure constants): XLA's algebraic simplifier folds
        # (x*c1)*c2 -> x*(c1*c2) for constants, which is not bit-identical (seen on ADM = (RGAS*(..))*BYGRAV)
        kapa, grav, rgas = geo['kapa'], geo['grav'], geo['rgas']
        zK, zKp1, zKp2, byGRAV = geo['bykapa'], geo['bykapap1'], geo['bykapap2'], geo['bygrav']
        dt4 = dt1 / 4.
        hk = .01 ** g['kapa']                                          # HUNDREDTHeKAPA (host double pow)
        mam_t = mam.transpose(1, 2, 0)
        m0 = jnp.full((IM, JM), geo['mtop'])
        pu0 = m0 * grav
        pku0 = jnp.power(pu0, kapa); pkpu0 = pku0 * pu0; pkppu0 = pkpu0 * pu0

        def down(c, xs):
            m, pu, pku, pkpu, pkppu = c
            mam_l, s0_l, sz_l = xs
            dp = mam_l * grav
            zdp = 1 / dp
            y = sz_l * 2 * zdp * hk
            x = s0_l * hk + y * (pu + .5 * dp)
            pd = pu + dp
            pkd = jnp.power(pd, kapa); pkpd = pkd * pd; pkppd = pkpd * pd
            dgzu = rgas * (x * (pkd - pku) * zK - y * (pkpd - pkpu) * zKp1)
            dgza = rgas * (x * (dp * pkd - (pkpd - pkpu) * zKp1) * zK
                           - y * (dp * pkpd - (pkppd - pkppu) * zKp2) * zKp1) * zdp
            adm_l = dgzu * byGRAV
            p_l = grav * (m + .5 * mam_l)
            return (m + mam_l, pd, pkd, pkpd, pkppd), (dgzu, dgza, adm_l, p_l)
        rev = lambda a: jnp.moveaxis(a, 2, 0)[::-1]                   # (IM,JM,LM) -> (LM,IM,JM) top-down
        _, (dgzu, dgza, adm, p) = lax.scan(down, (m0, pu0, pku0, pkpu0, pkppu0),
                                           (jnp.moveaxis(mam_t, 2, 0)[::-1], rev(s0), rev(sz)))
        dgzu, dgza, adm, p = dgzu[::-1], dgza[::-1], adm[::-1], p[::-1]   # (LM,IM,JM) L=1..LM

        def up(gzd, xs):
            dza, dzu = xs
            return gzd + dzu, gzd + dza
        _, gz = lax.scan(up, geo['zatmo'], (dgza, dgzu))
        adm = jnp.moveaxis(adm, 0, 2); p = jnp.moveaxis(p, 0, 2); gz = jnp.moveaxis(gz, 0, 2)
        # polar columns from I=1
        pol = lambda a: a.at[1:, 0, :].set(a[0, 0, :][None, :]).at[1:, JM - 1, :].set(a[0, JM - 1, :][None, :])
        adm, p, gz = pol(adm), pol(p), pol(gz)
        phi = gz
        fac = (dt4 * geo['dxv'])[None, 1:JM, None]
        flux = (((adm[:, 1:JM, :] + adm[:, 0:JM - 1, :]) * (p[:, 1:JM, :] - p[:, 0:JM - 1, :])
                 + (mam_t[:, 1:JM, :] + mam_t[:, 0:JM - 1, :]) * (gz[:, 1:JM, :] - gz[:, 0:JM - 1, :])) * fac)
        d0 = dvt[:, 1:JM, :]
        f_next = jnp.roll(flux, -1, axis=0)
        dv = (d0 - flux) - f_next
        dv = dv.at[IM - 1].set((d0[IM - 1] - flux[0]) - flux[IM - 1])
        dvt = dvt.at[:, 1:JM, :].set(dv)
        ip1 = np.roll(np.arange(IM), -1)
        pgfu = jnp.zeros((IM, JM, LM)).at[:, 1:JM, :].set(
            (adm[ip1][:, 1:JM, :] + adm[:, 1:JM, :]) * (p[ip1][:, 1:JM, :] - p[:, 1:JM, :])
            + (mam_t[ip1][:, 1:JM, :] + mam_t[:, 1:JM, :]) * (gz[ip1][:, 1:JM, :] - gz[:, 1:JM, :]))
        pgfu0 = pgfu
        pgfu = jf.avrx_field_jax(pgfu, act, geo['avrx_coef'], geo['avrx_mask'])
        facu = (-dt4 * geo['dyv'])[None, 1:JM, None]
        dut = dut.at[:, 1:JM, :].set(dut[:, 1:JM, :] + facu * (pgfu[:, 1:JM, :] + pgfu[:, 0:JM - 1, :]))
        for j in (1, JM - 1):
            dut = dut.at[:, j, :].set(dut[:, j, :] * geo['acor'])
            dvt = dvt.at[:, j, :].set(dvt[:, j, :] * geo['acor2'])
        mn = mafter.transpose(1, 2, 0)
        vmass = .5 * (((mn[:, 0:JM - 1, :] + mn[ip1][:, 0:JM - 1, :]) * geo['dxyn'][None, 0:JM - 1, None])
                      + ((mn[:, 1:JM, :] + mn[ip1][:, 1:JM, :]) * geo['dxys'][None, 1:JM, None]))
        ut = ut.at[:, 1:JM, :].set(ut[:, 1:JM, :] + dut[:, 1:JM, :] / vmass)
        vt = vt.at[:, 1:JM, :].set(vt[:, 1:JM, :] + dvt[:, 1:JM, :] / vmass)
        return dict(ut=ut, vt=vt, dut=dut, dvt=dvt, gz=gz, phi=phi, adm=adm, pgfu0=pgfu0, pgfu=pgfu)
    return pgf_jax, geo
