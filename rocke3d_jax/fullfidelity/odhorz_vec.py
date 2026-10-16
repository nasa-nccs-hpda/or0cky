"""Vectorized ODHORZ (Stage 2, D78): the layer-by-layer horizontal momentum and mass solve of
odhorz_ff.odhorz, with the (i, j) loops replaced by masked array operations.

The sequential structure is kept: layers run top-down (L = LMO..1), carrying PDN and OGEOZ across
layers. Within a layer every grid point is independent except for the polar row, which is handled
by the same polevel call as the scalar port. Arrays are 1-based (IM+1, JM+1), as in odhorz_ff.
Validated against odhorz_ff.odhorz (odhorz_vec_compare.py).
"""
import numpy as np

from odhorz_ff import geomo_dyn_arrays, IM, JM, LMO, GRAV, OMEGA
from polerelax_ff import polevel, geomo_pole_arrays


def _shift_i(a, k):
    """a[ip(i), j] for 1-based i: returns b with b[i, :] = a[i+k wrapped to 1..IM, :]."""
    b = np.zeros_like(a)
    idx = np.arange(1, IM + 1)
    src = ((idx - 1 + k) % IM) + 1
    b[1:IM + 1] = a[src]
    return b


def odhorz_vec(lmm, lmu, lmv, hocean, dt, moh, uoh, voh, uodh, vodh, opboth,
               mo0, uo0, vo0, uod0, vod0, opbot0, vbar, dzgdp, usmooth, pgfx):
    sinvo, sinpo, dxpo, dypo, dxvo, dyvo, dxyvo, dxypo = geomo_dyn_arrays()
    cosic, sinic, cosu, sinu = geomo_pole_arrays()
    mo = mo0.copy(); uo = uo0.copy(); vo = vo0.copy()
    uod = uod0.copy(); vod = vod0.copy(); opbot = opbot0.copy()

    I = np.arange(IM + 1)[:, None]
    J = np.arange(JM + 1)[None, :]
    notpole = (J != JM)
    pole_row = (J == JM) & (I == 1)

    def m_act(l):
        return np.where(notpole, lmm >= l, pole_row & (lmm[1, JM] >= l))

    def u_act(l):
        return notpole & (lmu >= l)

    def v_act(l):
        return notpole & (lmv >= l)

    # bottom pressure and geopotential (once)
    ma = np.zeros((IM + 1, JM + 1), bool)
    pdn = np.zeros((IM + 1, JM + 1))
    ogeoz = np.zeros((IM + 1, JM + 1))
    ma1 = m_act(1)
    pdn = np.where(ma1, opboth, pdn)
    ogeoz = np.where(ma1, -hocean * GRAV, ogeoz)

    for l in range(LMO, 0, -1):
        ma = m_act(l)
        dp = moh[:, :, l] * GRAV
        dh = np.where(ma, moh[:, :, l] * vbar[:, :, l], 0.0)
        p = np.where(ma, pdn - 0.5 * dp, 0.0)
        zg = np.where(ma, ogeoz + dp * 0.5 * dzgdp[:, :, l], 0.0)
        pdn = np.where(ma, pdn - dp, pdn)
        ogeoz = np.where(ma, ogeoz + dh * GRAV, ogeoz)

        us = usmooth[:, :, l]
        # ke, ua, va at active points
        us_im1 = _shift_i(us, -1)
        uasmooth = 0.5 * (us_im1 + us)
        uoh_im1 = _shift_i(uoh[:, :, l], -1)
        ua = np.where(ma, 0.5 * (uoh_im1 + uoh[:, :, l]), 0.0)
        va_raw = np.zeros((IM + 1, JM + 1))
        va_raw[:, 1:] = 0.5 * (voh[:, :-1, l] + voh[:, 1:, l])
        va = np.where(ma, va_raw, 0.0)
        ke = np.where(ma, 0.5 * (ua * uasmooth + va ** 2), 0.0)
        # pole row: the scalar port fills i=2..IM from i=1 for the pole row
        if lmm[1, JM] >= l:
            j = JM
            dh[2:, j] = dh[1, j]
            p[2:, j] = p[1, j]
            zg[2:, j] = zg[1, j]
            ke[2:, j] = ke[1, j]
            ua[2:, j] = 0.5 * (us[1:IM, j] + us[2:IM + 1, j])

        # south-north pressure gradient force
        pgfy = np.zeros((IM + 1, JM + 1))
        mv = np.zeros((IM + 1, JM + 1))
        bydy_v = 1.0 / dyvo
        bydy_v = np.where(np.arange(JM + 1) == JM - 1, bydy_v * 2.0 / 3.0, bydy_v)
        va_act = v_act(l)
        mmid_v = moh[:, :, l] + np.roll(moh[:, :, l], -1, axis=1)
        sel = np.zeros((IM + 1, JM + 1), bool)
        sel[:, 2:JM] = va_act[:, 2:JM]
        jj = np.arange(2, JM)
        tmp = ((zg[:, jj] - zg[:, jj + 1]) + (p[:, jj] - p[:, jj + 1]) *
               (dh[:, jj] + dh[:, jj + 1]) / mmid_v[:, jj]) * bydy_v[jj][None, :]
        pgfy[:, jj] = np.where(sel[:, jj], tmp, 0.0)

        # vorticity at cell corners
        vort = np.zeros((IM + 1, JM + 1))
        jv = np.arange(1, JM)
        voh_ip1 = _shift_i(voh[:, :, l], 1)
        vort[:, jv] = (dxpo[jv][None, :] * us[:, jv] - dxpo[jv + 1][None, :] * us[:, jv + 1] +
                       dyvo[jv][None, :] * (voh_ip1[:, jv] - voh[:, jv, l])) / dxyvo[jv][None, :]

        # update UO, VOD (j = 2..JM-1)
        ju = np.arange(2, JM)
        ua_act = u_act(l)
        vq_full = np.zeros((IM + 1, JM + 1))
        va_ip1 = _shift_i(va, 1)
        vq_full[:, ju] = 0.25 * (va[:, ju] + va_ip1[:, ju]) * (vort[:, ju - 1] + vort[:, ju])
        corofj_u = 2.0 * OMEGA * sinpo
        bydx = 1.0 / dxpo
        ke_ip1 = _shift_i(ke, 1)
        pgfx_l = pgfx[:, :, l]
        uo_new = uo[:, :, l].copy()
        vod_new = vod[:, :, l].copy()
        upd_u = (uo0[:, :, l] + dt * (pgfx_l + (ke - ke_ip1) * bydx[None, :] +
                 vodh[:, :, l] * corofj_u[None, :] + vq_full))
        pgf4 = 0.25 * (pgfy + _shift_i(pgfy, 1) + np.roll(pgfy, 1, axis=1) +
                       _shift_i(np.roll(pgfy, 1, axis=1), 1))
        upd_vod = vod0[:, :, l] + dt * (pgf4 - uoh[:, :, l] * corofj_u[None, :])
        uo_new[:, ju] = np.where(ua_act[:, ju], upd_u[:, ju], uo_new[:, ju])
        vod_new[:, ju] = np.where(ua_act[:, ju], upd_vod[:, ju], vod_new[:, ju])
        uo[:, :, l] = uo_new
        vod[:, :, l] = vod_new

        # update VO, UOD (j = 1..JM-1)
        jv2 = np.arange(1, JM)
        va2_act = v_act(l)
        corofj_v = 2.0 * OMEGA * sinvo
        bydy_c = 1.0 / dyvo
        bydy_c = np.where(np.arange(JM + 1) == JM - 1, bydy_c * 2.0 / 3.0, bydy_c)
        mvfac = 0.5 * dxvo
        pgfac = np.where(np.arange(JM + 1) == JM - 1, 0.5, 0.25)
        vort_im1 = _shift_i(vort, -1)
        mmid_v2 = moh[:, :, l] + np.roll(moh[:, :, l], -1, axis=1)
        mv_new = mvfac[None, :] * voh[:, :, l] * mmid_v2
        mv[:, jv2] = np.where(va2_act[:, jv2], mv_new[:, jv2], 0.0)
        uq = np.zeros((IM + 1, JM + 1))
        ua_jp1 = np.roll(ua, -1, axis=1)
        uq[:, jv2] = 0.25 * (ua[:, jv2] + ua_jp1[:, jv2]) * (vort_im1[:, jv2] + vort[:, jv2])
        ke_jp1 = np.roll(ke, -1, axis=1)
        vo_upd = vo0[:, :, l] + dt * (pgfy + (ke - ke_jp1) * bydy_c[None, :] -
                                      uodh[:, :, l] * corofj_v[None, :] - uq)
        pgfx_im1 = _shift_i(pgfx_l, -1)
        pgfx_jp1 = np.roll(pgfx_l, -1, axis=1)
        pgfx_im1_jp1 = _shift_i(pgfx_jp1, -1)
        pgf4v = pgfac[None, :] * (pgfx_im1 + pgfx_l + pgfx_im1_jp1 + pgfx_jp1)
        uod_upd = uod0[:, :, l] + dt * (pgf4v + voh[:, :, l] * corofj_v[None, :])
        vo[:, jv2, l] = np.where(va2_act[:, jv2], vo_upd[:, jv2], vo[:, jv2, l])
        uod[:, jv2, l] = np.where(va2_act[:, jv2], uod_upd[:, jv2], uod[:, jv2, l])

        # polar velocities (scalar routine on this layer)
        polevel(uo, vo, l, lmv, cosic, sinic, cosu, sinu)

        # MO, OPBOT
        mu = np.zeros((IM + 1, JM + 1))
        mmid_u = moh[:, :, l] + _shift_i(moh[:, :, l], 1)
        mu_full = 0.5 * dypo[None, :] * us * mmid_u
        mu = np.where(u_act(l), mu_full, 0.0)
        jc = np.arange(2, JM)
        convfac = dt / dxypo
        mu_im1 = _shift_i(mu, -1)
        mv_jm1 = np.roll(mv, 1, axis=1)
        convij = convfac[None, :] * (mu_im1 - mu + mv_jm1 - mv)
        upd = np.zeros((IM + 1, JM + 1), bool)
        upd[:, jc] = m_act(l)[:, jc]
        mo[:, jc, l] = np.where(upd[:, jc], mo0[:, jc, l] + convij[:, jc], mo[:, jc, l])
        opbot[:, jc] = np.where(upd[:, jc], opbot[:, jc] + convij[:, jc] * GRAV, opbot[:, jc])
        if lmm[1, JM] >= l:
            j = JM
            cij = dt * np.sum(mv[1:IM + 1, JM - 1]) / (IM * dxypo[JM])
            mo[1, j, l] = mo0[1, j, l] + cij
            opbot[1, j] = opbot[1, j] + cij * GRAV
            mo[2:IM + 1, j, l] = mo[1, j, l]
    return mo, uo, vo, uod, vod, opbot
