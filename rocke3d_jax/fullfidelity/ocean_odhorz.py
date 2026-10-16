"""D119: copy of odhorz_jax (D79) whose layer step additionally returns the layer mass fluxes MU, MV (needed to
accumulate SMU/SMV on the even leapfrog calls, OCNDYN2.f:1351,1443). Arithmetic unchanged."""
"""JAX ODHORZ (Stage 2, D79): odhorz_vec.odhorz_vec with the layer body under jax.jit.

The layer loop (L = LMO..1, carrying PDN, OGEOZ, OPBOT) stays a Python loop of 13 steps; each step is
one jitted call on that layer's (IM+1, JM+1) slices. The `if lmm[1, JM] >= l` polar branches of the
numpy version become masked jnp.where updates so that the layer index can be traced (one compile
for all 13 layers). Arrays are 1-based (IM+1, JM+1), as in odhorz_ff / odhorz_vec.
Validated against odhorz_vec (odhorz_jax_compare.py).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from odhorz_ff import geomo_dyn_arrays, IM, JM, LMO, GRAV, OMEGA
from polerelax_ff import geomo_pole_arrays

# D119: the real OMEGA is 2*pi/siderealRotationPeriod with Earth365DayOrbit's period 86400*365/366 s = 86163.934 s
# (shared/Earth365DayOrbit.F90:101, shared/Constants_mod.F90:282); odhorz_ff.OMEGA used 86164.09054 s (rel. 1.8e-6 high).
OMEGA = 2.0 * np.pi / (86400.0 * 365.0 / 366.0)

_SRC_M1 = ((np.arange(1, IM + 1) - 1 - 1) % IM) + 1   # i-1 wrapped to 1..IM
_SRC_P1 = ((np.arange(1, IM + 1) - 1 + 1) % IM) + 1   # i+1 wrapped to 1..IM


def _shift_im1(a):
    return jnp.zeros_like(a).at[1:IM + 1].set(a[_SRC_M1])


def _shift_ip1(a):
    return jnp.zeros_like(a).at[1:IM + 1].set(a[_SRC_P1])


def make_layer_step(lmm, lmu, lmv):
    sinvo, sinpo, dxpo, dypo, dxvo, dyvo, dxyvo, dxypo = [jnp.asarray(a) for a in geomo_dyn_arrays()]
    cosic, sinic, cosu, sinu = [jnp.asarray(a) for a in geomo_pole_arrays()]
    lmm = jnp.asarray(lmm); lmu = jnp.asarray(lmu); lmv = jnp.asarray(lmv)
    I = jnp.arange(IM + 1)[:, None]
    J = jnp.arange(JM + 1)[None, :]
    notpole = (J != JM)
    pole_row = (J == JM) & (I == 1)
    jidx = jnp.arange(JM + 1)
    bydy = 1.0 / dyvo
    bydy = jnp.where(jidx == JM - 1, bydy * 2.0 / 3.0, bydy)
    corofj_u = 2.0 * OMEGA * sinpo
    corofj_v = 2.0 * OMEGA * sinvo
    bydx = 1.0 / dxpo
    mvfac = 0.5 * dxvo
    pgfac = jnp.where(jidx == JM - 1, 0.5, 0.25)
    convfac = 1.0 / dxypo
    jj = np.arange(2, JM)
    jv = np.arange(1, JM)

    @jax.jit
    def step(l, dt, pdn, ogeoz, opbot,
             moh, uoh, voh, uodh, vodh, vbar, dzgdp, us, pgfx,
             mo0, uo0, vo0, uod0, vod0):
        pc = lmm[1, JM] >= l
        ma = jnp.where(notpole, lmm >= l, pole_row & pc)
        ua_act = notpole & (lmu >= l)
        va_act = notpole & (lmv >= l)

        dp = moh * GRAV
        dh = jnp.where(ma, moh * vbar, 0.0)
        p = jnp.where(ma, pdn - 0.5 * dp, 0.0)
        zg = jnp.where(ma, ogeoz + dp * 0.5 * dzgdp, 0.0)
        pdn_new = jnp.where(ma, pdn - dp, pdn)
        ogeoz_new = jnp.where(ma, ogeoz + dh * GRAV, ogeoz)

        uasmooth = 0.5 * (_shift_im1(us) + us)
        ua = jnp.where(ma, 0.5 * (_shift_im1(uoh) + uoh), 0.0)
        va_raw = jnp.zeros((IM + 1, JM + 1)).at[:, 1:].set(0.5 * (voh[:, :-1] + voh[:, 1:]))
        va = jnp.where(ma, va_raw, 0.0)
        ke = jnp.where(ma, 0.5 * (ua * uasmooth + va ** 2), 0.0)

        def polefill(a, new=None):
            row = a[1, JM] if new is None else new
            return a.at[2:, JM].set(jnp.where(pc, row, a[2:, JM]))
        dh = polefill(dh)
        p = polefill(p)
        zg = polefill(zg)
        ke = polefill(ke)
        ua = polefill(ua, 0.5 * (us[1:IM, JM] + us[2:IM + 1, JM]))

        # south-north pressure gradient force
        mmid_v = moh + jnp.roll(moh, -1, axis=1)
        sel = va_act[:, jj]
        tmp = ((zg[:, jj] - zg[:, jj + 1]) + (p[:, jj] - p[:, jj + 1]) *
               (dh[:, jj] + dh[:, jj + 1]) / mmid_v[:, jj]) * bydy[jj][None, :]
        pgfy = jnp.zeros((IM + 1, JM + 1)).at[:, jj].set(jnp.where(sel, tmp, 0.0))

        # vorticity at cell corners
        voh_ip1 = _shift_ip1(voh)
        vort = jnp.zeros((IM + 1, JM + 1)).at[:, jv].set(
            (dxpo[jv][None, :] * us[:, jv] - dxpo[jv + 1][None, :] * us[:, jv + 1] +
             dyvo[jv][None, :] * (voh_ip1[:, jv] - voh[:, jv])) / dxyvo[jv][None, :])

        # update UO, VOD (j = 2..JM-1)
        vq_full = jnp.zeros((IM + 1, JM + 1)).at[:, jj].set(
            0.25 * (va[:, jj] + _shift_ip1(va)[:, jj]) * (vort[:, jj - 1] + vort[:, jj]))
        upd_u = uo0 + dt * (pgfx + (ke - _shift_ip1(ke)) * bydx[None, :] +
                            vodh * corofj_u[None, :] + vq_full)
        pgfy_jm1 = jnp.roll(pgfy, 1, axis=1)
        pgf4 = 0.25 * (pgfy + _shift_ip1(pgfy) + pgfy_jm1 + _shift_ip1(pgfy_jm1))
        upd_vod = vod0 + dt * (pgf4 - uoh * corofj_u[None, :])
        uo = uo0.at[:, jj].set(jnp.where(ua_act[:, jj], upd_u[:, jj], uo0[:, jj]))
        vod = vod0.at[:, jj].set(jnp.where(ua_act[:, jj], upd_vod[:, jj], vod0[:, jj]))

        # update VO, UOD (j = 1..JM-1)
        vort_im1 = _shift_im1(vort)
        mv_new = mvfac[None, :] * voh * mmid_v
        mv = jnp.zeros((IM + 1, JM + 1)).at[:, jv].set(jnp.where(va_act[:, jv], mv_new[:, jv], 0.0))
        uq = jnp.zeros((IM + 1, JM + 1)).at[:, jv].set(
            0.25 * (ua[:, jv] + jnp.roll(ua, -1, axis=1)[:, jv]) * (vort_im1[:, jv] + vort[:, jv]))
        vo_upd = vo0 + dt * (pgfy + (ke - jnp.roll(ke, -1, axis=1)) * bydy[None, :] -
                             uodh * corofj_v[None, :] - uq)
        pgfx_jp1 = jnp.roll(pgfx, -1, axis=1)
        pgf4v = pgfac[None, :] * (_shift_im1(pgfx) + pgfx + _shift_im1(pgfx_jp1) + pgfx_jp1)
        uod_upd = uod0 + dt * (pgf4v + voh * corofj_v[None, :])
        vo = vo0.at[:, jv].set(jnp.where(va_act[:, jv], vo_upd[:, jv], vo0[:, jv]))
        uod = uod0.at[:, jv].set(jnp.where(va_act[:, jv], uod_upd[:, jv], uod0[:, jv]))

        # polar velocities (polevel)
        w = (lmv[1:IM + 1, JM - 1] >= l)
        unp = -jnp.sum(jnp.where(w, sinic[1:] * vo[1:IM + 1, JM - 1], 0.0)) * (2.0 / IM)
        vnp = jnp.sum(jnp.where(w, cosic[1:] * vo[1:IM + 1, JM - 1], 0.0)) * (2.0 / IM)
        uo = uo.at[1:IM + 1, JM].set(unp * cosu[1:] + vnp * sinu[1:])
        vo = vo.at[1:IM + 1, JM].set(vnp * cosic[1:] - unp * sinic[1:])

        # MO, OPBOT
        mmid_u = moh + _shift_ip1(moh)
        mu = jnp.where(notpole & (lmu >= l), 0.5 * dypo[None, :] * us * mmid_u, 0.0)
        convij = (dt * convfac)[None, :] * (_shift_im1(mu) - mu + jnp.roll(mv, 1, axis=1) - mv)
        upd = ma[:, jj]
        mo = mo0.at[:, jj].set(jnp.where(upd, mo0[:, jj] + convij[:, jj], mo0[:, jj]))
        opbot_new = opbot.at[:, jj].set(
            jnp.where(upd, opbot[:, jj] + convij[:, jj] * GRAV, opbot[:, jj]))
        cij = dt * jnp.sum(mv[1:IM + 1, JM - 1]) / (IM * dxypo[JM])
        mo_pole = mo0[1, JM] + cij
        mo = mo.at[1:IM + 1, JM].set(jnp.where(pc, mo_pole, mo[1:IM + 1, JM]))
        opbot_new = opbot_new.at[1, JM].set(
            jnp.where(pc, opbot_new[1, JM] + cij * GRAV, opbot_new[1, JM]))
        return pdn_new, ogeoz_new, opbot_new, mo, uo, vo, uod, vod, mu, mv

    return step


def odhorz_jax(lmm, lmu, lmv, hocean, dt, moh, uoh, voh, uodh, vodh, opboth,
               mo0, uo0, vo0, uod0, vod0, opbot0, vbar, dzgdp, usmooth, pgfx, step=None):
    if step is None:
        step = make_layer_step(lmm, lmu, lmv)
    I = np.arange(IM + 1)[:, None]
    J = np.arange(JM + 1)[None, :]
    ma1 = np.where(J != JM, lmm >= 1, (J == JM) & (I == 1) & (lmm[1, JM] >= 1))
    pdn = jnp.asarray(np.where(ma1, opboth, 0.0))
    ogeoz = jnp.asarray(np.where(ma1, -hocean * GRAV, 0.0))
    opbot = jnp.asarray(opbot0)
    mo = np.array(mo0); uo = np.array(uo0); vo = np.array(vo0)
    mu3 = np.zeros_like(mo); mv3 = np.zeros_like(mo)
    uod = np.array(uod0); vod = np.array(vod0)
    for l in range(LMO, 0, -1):
        pdn, ogeoz, opbot, mo_l, uo_l, vo_l, uod_l, vod_l, mu_l, mv_l = step(
            l, dt, pdn, ogeoz, opbot,
            moh[:, :, l], uoh[:, :, l], voh[:, :, l], uodh[:, :, l], vodh[:, :, l],
            vbar[:, :, l], dzgdp[:, :, l], usmooth[:, :, l], pgfx[:, :, l],
            mo0[:, :, l], uo0[:, :, l], vo0[:, :, l], uod0[:, :, l], vod0[:, :, l])
        mo[:, :, l] = mo_l; uo[:, :, l] = uo_l; vo[:, :, l] = vo_l
        uod[:, :, l] = uod_l; vod[:, :, l] = vod_l
        mu3[:, :, l] = np.asarray(mu_l); mv3[:, :, l] = np.asarray(mv_l)
    return mo, uo, vo, uod, vod, np.asarray(opbot), mu3, mv3
