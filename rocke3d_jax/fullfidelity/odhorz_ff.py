"""Full-fidelity port of OCNDYN2.f's ODHORZ -- Stage 2 of the DYNSI/ocean port, D42 (D44 added
optional SMU/SMV accumulation, OCEAN_DYN's integrated horizontal mass fluxes consumed by
OFLUXV/OADVT2's tracer advection -- see `odhorz()`'s docstring and `odhorz_smuv_compare.py`).

ODHORZ is the actual horizontal momentum + mass-continuity solve that ODHORZ0 (D40) prepares
pressure/equation-of-state inputs for. Called several times per DTsrc step (twice for the initial
forward/backward half-steps, then once per leapfrog sub-step) with a "H" (history/flux-source)
state and a separate current (INOUT) state -- a standard leapfrog/Euler-predictor pattern.

Per call, top-down across layers (L=LMO..1), carrying P/ZG/DH/PDN/OGEOZ state across layers:
1. Smooth the west-east velocity (`USMOOTH`, via `OPFIL2`) for the mass flux `MU` and later terms.
2. Accumulate pressure `P`, geopotential `ZG`, thickness `DH` using `VBAR`/`dZGdP` (from `ODHORZ0`).
3. Compute kinetic energy `KE` at cell centers.
4. Compute east-west pressure-gradient force `PGFX` (`OPFIL2`-smoothed) and south-north `PGFY`.
5. Compute relative vorticity `VORT` at cell corners.
6. Update `UO`/`VOD` and `VO`/`UOD` (pressure-gradient force + KE gradient + Coriolis + vorticity
   advection, cross-registered).
7. Reconstruct the polar velocities via `polevel()` (D39, reused unchanged).
8. Update `MO` (mass continuity) and `OPBOT`.

`OPFIL2` (polar Fourier filter, external `AVR` file -- same "new architecture" item as `OFLUXV`)
is decoupled via the established "record what's not yet ported" pattern: its two per-layer
outputs (`USMOOTH`, `PGFX`) are recorded directly as real inputs (D34's solar profile, D35's
SHCGS, D40's VOLGSP are the same pattern). `HOCEAN` (bathymetry, needed for `OGEOZ`'s per-call
initialization) is recorded once as static geometry (real file-sourced data, not derivable).

`OMEGA` (planetary rotation rate) is a genuine runtime planet parameter
(`omega = 2*pi/rotationPeriod`, not a hardcoded constant) -- used here at Earth's standard
sidereal value and validated empirically against real Fortran output (same approach as D29's
RADIUS/GRAV, both confirmed Earth-standard for this rundeck via exact real-data matches).
"""
import numpy as np
from polerelax_ff import polevel, geomo_pole_arrays
from ostres2_ff import geomo_arrays

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665
RADIUS = 6371000.0
OMEGA = 2.0 * np.pi / 86164.09054  # sidereal day, s -- Earth-standard; validated empirically


def geomo_dyn_arrays():
    """Analytically derive SINVO, SINPO, DXPO, DYPO, DXVO, DYVO, DXYVO, DXYPO (length JM+1,
    1-indexed), matching OGEOM.f's GEOMO exactly (re-verified line-by-line against the source
    for this delta). DXYVO/DXYPO reuse D36's already-validated ostres2_ff.geomo_arrays()."""
    twopi = 2.0 * np.pi
    dlon = twopi / IM
    fjeq = 0.5 * (1 + JM)
    odlat_dg = round(180.0 / (JM - 1))
    dlat = odlat_dg * np.pi / 180.0

    cosv = np.zeros(JM + 1)  # cosv[0]=cosv[JM]=0 by construction (GEOMO's COSV(0)=COSV(JM)=0)
    sinvo = np.zeros(JM + 1)
    dxvo = np.zeros(JM + 1)
    for j in range(1, JM):
        cosv[j] = np.cos(dlat * (j + 0.5 - fjeq))
        sinvo[j] = np.sin(dlat * (j + 0.5 - fjeq))
        dxvo[j] = RADIUS * dlon * cosv[j]
    dyvo = np.full(JM + 1, RADIUS * dlat)
    dyvo[JM] = 0.0
    dxvo[JM] = 0.0

    sinpo = np.zeros(JM + 1)
    dxpo = np.zeros(JM + 1)
    dypo = np.zeros(JM + 1)
    for j in range(1, JM + 1):
        latn = dlat * (j + 0.5 - fjeq) if j != JM else twopi / 4
        lats = dlat * (j - 0.5 - fjeq) if j != 1 else -twopi / 4
        rlat = dlat * (j - fjeq)
        sinpo[j] = np.sin(rlat)
        dxpo[j] = 0.5 * RADIUS * dlon * (cosv[j - 1] + cosv[j])
        dypo[j] = RADIUS * (latn - lats)
    sinpo[1] = -1.0
    sinpo[JM] = 1.0

    dxys, dxyn, dxyvo, _, _ = geomo_arrays()
    dxypo = dxys + dxyn
    return sinvo, sinpo, dxpo, dypo, dxvo, dyvo, dxyvo, dxypo


def odhorz(lmm, lmu, lmv, hocean, dt,
           moh, uoh, voh, uodh, vodh, opboth,
           mo0, uo0, vo0, uod0, vod0, opbot0,
           vbar, dzgdp, usmooth, pgfx,
           qeven=None, smu0=None, smv0=None):
    """Direct port of ODHORZ (OCNDYN2.f:1185-1560ish). All 2D fields shape (IM+1,JM+1); all 3D
    fields shape (IM+1,JM+1,LMO+1); 1-indexed. `usmooth`,`pgfx` are OPFIL2's recorded real
    per-layer outputs (shape (IM+1,JM+1,LMO+1), used directly). Returns
    (mo, uo, vo, uod, vod, opbot), fresh copies.

    D44 addition: `qeven`/`smu0`/`smv0` are optional -- when given, also accumulates and returns
    SMU/SMV (OCEAN_DYN's integrated horizontal mass fluxes, `Use OCEAN_DYN, Only: SMU,SMV` in the
    real Fortran), reusing the same `mu`/`mv` arrays already computed (and already validated via
    D42's MO/OPBOT match) for the mass-continuity update: `smu[i,j,l] = smu0[i,j,l] +
    mu[i,j]*xeven`, `xeven = 1.0 if qeven else 0.0` (OCNDYN2.f:1351,1443 -- SMU accumulates in the
    east-west-pressure-gradient block, SMV in the "update VO,UOD" block, both reusing the single
    MU/MV computation also read later by the mass-continuity update). Returns
    (mo, uo, vo, uod, vod, opbot, smu, smv) when qeven is not None, else the original 6-tuple.
    """
    track_sm = qeven is not None
    if track_sm:
        xeven = 1.0 if qeven else 0.0
        smu = smu0.copy()
        smv = smv0.copy()
    sinvo, sinpo, dxpo, dypo, dxvo, dyvo, dxyvo, dxypo = geomo_dyn_arrays()
    cosic, sinic, cosu, sinu = geomo_pole_arrays()

    mo = mo0.copy()
    uo = uo0.copy()
    vo = vo0.copy()
    uod = uod0.copy()
    vod = vod0.copy()
    opbot = opbot0.copy()

    def m_active(i, j, l):
        if j == JM:
            return i == 1 and lmm[1, JM] >= l
        return lmm[i, j] >= l

    def u_active(i, j, l):
        if j == JM:
            return False
        return lmu[i, j] >= l

    def v_active(i, j, l):
        if j == JM:
            return False
        return lmv[i, j] >= l

    # ---- Initialize pressure and geopotential at the ocean bottom (once, before the layer loop) ----
    pdn = np.zeros((IM + 1, JM + 1))
    ogeoz = np.zeros((IM + 1, JM + 1))
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            if m_active(i, j, 1):
                pdn[i, j] = opboth[i, j]
                ogeoz[i, j] = -hocean[i, j] * GRAV

    for l in range(LMO, 0, -1):
        p = np.zeros((IM + 1, JM + 1))
        zg = np.zeros((IM + 1, JM + 1))
        dh = np.zeros((IM + 1, JM + 1))
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if m_active(i, j, l):
                    dp = moh[i, j, l] * GRAV
                    dh[i, j] = moh[i, j, l] * vbar[i, j, l]
                    p[i, j] = pdn[i, j] - 0.5 * dp
                    zg[i, j] = ogeoz[i, j] + dp * 0.5 * dzgdp[i, j, l]
                    pdn[i, j] = pdn[i, j] - dp
                    ogeoz[i, j] = ogeoz[i, j] + dh[i, j] * GRAV

        us = usmooth[:, :, l]
        ke = np.zeros((IM + 1, JM + 1))
        ua = np.zeros((IM + 1, JM + 1))
        va = np.zeros((IM + 1, JM + 1))
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if not m_active(i, j, l):
                    continue
                im1 = IM if i == 1 else i - 1
                uasmooth = 0.5 * (us[im1, j] + us[i, j])
                ua[i, j] = 0.5 * (uoh[im1, j, l] + uoh[i, j, l])
                va[i, j] = 0.5 * (voh[i, j - 1, l] + voh[i, j, l])
                ke[i, j] = 0.5 * (ua[i, j] * uasmooth + va[i, j] ** 2)

        # fill pole
        if lmm[1, JM] >= l:
            j = JM
            for i in range(2, IM + 1):
                dh[i, j] = dh[1, j]
                p[i, j] = p[1, j]
                zg[i, j] = zg[1, j]
                ke[i, j] = ke[1, j]
                ua[i, j] = 0.5 * (us[i - 1, j] + us[i, j])

        # east-west pressure gradient force (already OPFIL2-smoothed via recorded pgfx)
        pgfx_l = pgfx[:, :, l]

        # south-north pressure gradient force
        pgfy = np.zeros((IM + 1, JM + 1))
        mv = np.zeros((IM + 1, JM + 1))
        for j in range(max(2, 1), JM):
            bydy = 1.0 / dyvo[j]
            if j == JM - 1:
                bydy *= 2.0 / 3.0
            for i in range(1, IM + 1):
                if v_active(i, j, l):
                    mmid = moh[i, j, l] + moh[i, j + 1, l]
                    pgfy[i, j] = ((zg[i, j] - zg[i, j + 1]) + (p[i, j] - p[i, j + 1]) *
                                  (dh[i, j] + dh[i, j + 1]) / mmid) * bydy

        # vorticity at cell corners
        vort = np.zeros((IM + 1, JM + 1))
        for j in range(1, JM):
            for i in range(1, IM + 1):
                ip1 = 1 if i == IM else i + 1
                vort[i, j] = (dxpo[j] * us[i, j] - dxpo[j + 1] * us[i, j + 1] +
                              dyvo[j] * (voh[ip1, j, l] - voh[i, j, l])) / dxyvo[j]

        # update UO, VOD
        for j in range(2, JM):
            bydx = 1.0 / dxpo[j]
            corofj = 2.0 * OMEGA * sinpo[j]
            for i in range(1, IM + 1):
                if not u_active(i, j, l):
                    continue
                ip1 = 1 if i == IM else i + 1
                vq = 0.25 * (va[i, j] + va[ip1, j]) * (vort[i, j - 1] + vort[i, j])
                uo[i, j, l] = uo0[i, j, l] + dt * (
                    pgfx_l[i, j] + (ke[i, j] - ke[ip1, j]) * bydx + vodh[i, j, l] * corofj + vq)
                pgf4pt = 0.25 * (pgfy[i, j] + pgfy[ip1, j] + pgfy[i, j - 1] + pgfy[ip1, j - 1])
                vod[i, j, l] = vod0[i, j, l] + dt * (pgf4pt - uoh[i, j, l] * corofj)

        # update VO, UOD
        for j in range(1, JM):
            bydy = 1.0 / dyvo[j]
            if j == JM - 1:
                bydy *= 2.0 / 3.0
            corofj = 2.0 * OMEGA * sinvo[j]
            mvfac = 0.5 * dxvo[j]
            pgfac = 0.5 if j == JM - 1 else 0.25
            for i in range(1, IM + 1):
                if not v_active(i, j, l):
                    continue
                im1 = IM if i == 1 else i - 1
                mmid = moh[i, j, l] + moh[i, j + 1, l]
                mv[i, j] = mvfac * voh[i, j, l] * mmid
                if track_sm:
                    smv[i, j, l] = smv0[i, j, l] + mv[i, j] * xeven
                uq = 0.25 * (ua[i, j] + ua[i, j + 1]) * (vort[im1, j] + vort[i, j])
                vo[i, j, l] = vo0[i, j, l] + dt * (
                    pgfy[i, j] + (ke[i, j] - ke[i, j + 1]) * bydy - uodh[i, j, l] * corofj - uq)
                pgf4pt = pgfac * (pgfx_l[im1, j] + pgfx_l[i, j] + pgfx_l[im1, j + 1] + pgfx_l[i, j + 1])
                uod[i, j, l] = uod0[i, j, l] + dt * (pgf4pt + voh[i, j, l] * corofj)

        # update polar velocities
        polevel(uo, vo, l, lmv, cosic, sinic, cosu, sinu)

        # update MO, OPBOT
        mu = np.zeros((IM + 1, JM + 1))
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if u_active(i, j, l):
                    mmid = moh[i, j, l] + moh[1 if i == IM else i + 1, j, l]
                    mu[i, j] = 0.5 * dypo[j] * us[i, j] * mmid
                    if track_sm:
                        smu[i, j, l] = smu0[i, j, l] + mu[i, j] * xeven
        for j in range(2, JM):
            convfac = dt / dxypo[j]
            for i in range(1, IM + 1):
                if m_active(i, j, l):
                    im1 = IM if i == 1 else i - 1
                    convij = convfac * (mu[im1, j] - mu[i, j] + mv[i, j - 1] - mv[i, j])
                    mo[i, j, l] = mo0[i, j, l] + convij
                    opbot[i, j] = opbot[i, j] + convij * GRAV
        if lmm[1, JM] >= l:
            j = JM
            convij = dt * np.sum(mv[1:IM + 1, JM - 1]) / (IM * dxypo[JM])
            mo[1, j, l] = mo0[1, j, l] + convij
            opbot[1, j] = opbot[1, j] + convij * GRAV
            for i in range(2, IM + 1):
                mo[i, j, l] = mo[1, j, l]

    if track_sm:
        return mo, uo, vo, uod, vod, opbot, smu, smv
    return mo, uo, vo, uod, vod, opbot
