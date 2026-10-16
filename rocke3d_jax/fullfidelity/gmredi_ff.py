"""Full-fidelity port of OCNGM.f's Gent-McWilliams/Redi mesoscale-mixing scheme -- Stage 2, D49+.
Scoped in D46/D48: `ocnmeso_drv`'s live "skew-GM" branch calls `GMKDIF` (density-gradient-
derived isoneutral slopes, itself calling `ISOSLOPE4`; `GET_PSI_DIAG` confirmed purely
diagnostic, D48, not ported) then `GMFEXP` (applies the skew flux to G0M/S0M).

`QCROSS` is always false for this rundeck's actual call (`ocnmeso_drv`'s `call gmkdif(k3d,1d0)`
hardcodes `RGMI_in=1d0`, and `QCROSS = .NOT.(RGMI.eq.1d0)`, D48) -- every `IF(QCROSS)` branch in
both `GMKDIF` and `GMFEXP` (the cross-term coefficients coupling isoneutral/thickness diffusion)
is dead code for this build, not ported.

This module starts with `ISOSLOPE4` (OCNGM.f:997-1130): the isopycnal-slope-derived diffusion
coefficients. Embarrassingly parallel per-cell -- no sequential dependency across I/J/L, unlike
almost everything else ported in Stage 2. Real inputs are exactly D47's `densgrad` outputs
(RHOX/RHOY/RHOMZ/BYRHOZ/BYDH/DZV) plus K3D (D47's `get_1d_mesodiff`, a constant=800 broadcast)
-- no new input instrumentation needed; `ffdump_isoslope4` (D49) records only ISOSLOPE4's own
24 output arrays.

AINV = RGMI*ARIV = ARIV exactly (RGMI=1.0 for this rundeck's call), so both are simply K3D --
no need to track them as separate quantities.
"""
import numpy as np
from odhorz_ff import geomo_dyn_arrays
from oadvt2_ff import _fortran_sum

IM, JM, LMO = 72, 46, 13
DTS = 1800.0  # OCEAN_COM.f default short ocean dynamics timestep, validated empirically D29-style


def isoslope4(lmm, rhox, rhoy, rhomz, byrhoz, bydh, dzv, k3d):
    """OCNGM.f:997-1130, QCROSS=false (always, D48). All fields (IM+1,JM+1,LMO+1), 1-indexed;
    `rhox`,`rhoy` already translated to this project's (i,j,l) convention (not the Fortran's
    native LMO-first ordering) by gmredi_compare.py. Returns a dict with aix0..aix3, aiy0..aiy3,
    asx0..asx3, asy0..asy3, s2x0..s2x3, s2y0..s2y3 (each (IM+1,JM+1,LMO+1))."""
    _, _, dxpo, dypo, _, dyvo, _, _ = geomo_dyn_arrays()
    bydyp = np.zeros_like(dypo)
    bydyp[1:] = 1.0 / dypo[1:]  # index 0 is the unused dummy row (1-indexed convention)

    shape = (IM + 1, JM + 1, LMO + 1)
    out = {}
    for name in ["aix0", "aix1", "aix2", "aix3", "aiy0", "aiy1", "aiy2", "aiy3",
                 "asx0", "asx1", "asx2", "asx3", "asy0", "asy1", "asy2", "asy3",
                 "s2x0", "s2x1", "s2x2", "s2x3", "s2y0", "s2y1", "s2y2", "s2y3"]:
        out[name] = np.zeros(shape)

    for l in range(1, LMO + 1):
        for j in range(2, JM + 1):  # J_STRT_STGR..J_STOP_STGR, serial: 2..JM (includes pole)
            for i in range(1, IM + 1):
                if lmm[i, j] < l:
                    continue
                im1 = IM if i == 1 else i - 1
                ariv = k3d[i, j, l]

                # SIX0,SIY0,SIX2,SIY2: four slopes using RHOMZ(L)
                if l == lmm[i, j] or rhomz[i, j, l] == 0.0:
                    aix0st = aix2st = aiy0st = aiy2st = 0.0
                    six0 = six2 = siy0 = siy2 = 0.0
                else:
                    aix0st = aix2st = aiy0st = aiy2st = ariv
                    six0 = rhox[i, j, l] * byrhoz[i, j, l]
                    six2 = rhox[im1, j, l] * byrhoz[i, j, l]
                    siy2 = rhoy[i, j - 1, l] * byrhoz[i, j, l]
                    siy0 = rhoy[i, j, l] * byrhoz[i, j, l]
                    if ariv > 0.0:
                        byaidt = 1.0 / (4.0 * DTS * (ariv + ariv))
                        dsq = dzv[i, j, l] ** 2 * byaidt
                        if six0 ** 2 > dsq:
                            aix0st = aix0st * dsq / six0 ** 2
                        if six2 ** 2 > dsq:
                            aix2st = aix2st * dsq / six2 ** 2
                        if siy0 ** 2 > dsq:
                            aiy0st = aiy0st * dsq / siy0 ** 2
                        if siy2 ** 2 > dsq:
                            aiy2st = aiy2st * dsq / siy2 ** 2
                    out["aix0"][i, j, l] = aix0st * dzv[i, j, l] * bydh[i, j, l]
                    out["aix2"][i, j, l] = aix2st * dzv[i, j, l] * bydh[i, j, l]
                    out["aiy0"][i, j, l] = aiy0st * dzv[i, j, l] * bydh[i, j, l]
                    out["aiy2"][i, j, l] = aiy2st * dzv[i, j, l] * bydh[i, j, l]

                # SIX1,SIY1,SIX3,SIY3: four slopes using RHOMZ(L-1)
                if l == 1 or rhomz[i, j, l - 1] == 0.0:
                    aix1st = aix3st = aiy1st = aiy3st = 0.0
                    six1 = six3 = siy1 = siy3 = 0.0
                else:
                    aix1st = aix3st = aiy1st = aiy3st = ariv
                    six1 = rhox[i, j, l] * byrhoz[i, j, l - 1]
                    six3 = rhox[im1, j, l] * byrhoz[i, j, l - 1]
                    siy1 = rhoy[i, j, l] * byrhoz[i, j, l - 1]
                    siy3 = rhoy[i, j - 1, l] * byrhoz[i, j, l - 1]
                    if ariv > 0.0:
                        byaidt = 1.0 / (4.0 * DTS * (ariv + ariv))
                        dsq = dzv[i, j, l - 1] ** 2 * byaidt
                        if six1 ** 2 > dsq:
                            aix1st = aix1st * dsq / six1 ** 2
                        if six3 ** 2 > dsq:
                            aix3st = aix3st * dsq / six3 ** 2
                        if siy1 ** 2 > dsq:
                            aiy1st = aiy1st * dsq / siy1 ** 2
                        if siy3 ** 2 > dsq:
                            aiy3st = aiy3st * dsq / siy3 ** 2
                    out["aix1"][i, j, l] = aix1st * dzv[i, j, l - 1] * bydh[i, j, l]
                    out["aix3"][i, j, l] = aix3st * dzv[i, j, l - 1] * bydh[i, j, l]
                    out["aiy1"][i, j, l] = aiy1st * dzv[i, j, l - 1] * bydh[i, j, l]
                    out["aiy3"][i, j, l] = aiy3st * dzv[i, j, l - 1] * bydh[i, j, l]

                out["asx0"][i, j, l] = aix0st * six0
                out["asx1"][i, j, l] = aix1st * six1
                out["asx2"][i, j, l] = aix2st * six2
                out["asx3"][i, j, l] = aix3st * six3
                out["asy0"][i, j, l] = aiy0st * siy0
                out["asy1"][i, j, l] = aiy1st * siy1
                out["asy2"][i, j, l] = aiy2st * siy2
                out["asy3"][i, j, l] = aiy3st * siy3

                out["s2x0"][i, j, l] = aix0st * six0 * six0
                out["s2x1"][i, j, l] = aix1st * six1 * six1
                out["s2x2"][i, j, l] = aix2st * six2 * six2
                out["s2x3"][i, j, l] = aix3st * six3 * six3
                out["s2y0"][i, j, l] = aiy0st * siy0 * siy0 * bydyp[j] * dyvo[j]
                out["s2y1"][i, j, l] = aiy1st * siy1 * siy1 * bydyp[j] * dyvo[j]
                out["s2y2"][i, j, l] = aiy2st * siy2 * siy2 * bydyp[j] * dyvo[j - 1]
                out["s2y3"][i, j, l] = aiy3st * siy3 * siy3 * bydyp[j] * dyvo[j - 1]

    return out


def gmkdif(lmm, kpl, aix0, aix1, aix2, aix3, aiy0, aiy1, aiy2, aiy3,
           asx0, asx1, asx2, asx3, asy0, asy1, asy2, asy3,
           s2x0, s2x1, s2x2, s2x3, s2y0, s2y1, s2y2, s2y3):
    """GMKDIF's remaining (post-QCROSS, D48) coefficient-setting logic (OCNGM.f:194-319, this
    delta's D50 numbering). Real inputs are D49's already-validated ISOSLOPE4 outputs plus `kpl`
    (mixed-layer-depth index, `OCEAN_COM.f`, set by `OCNKPP.f`'s `OCONV` -- not yet ported,
    recorded directly). Main loop's J range matches D49's finding (J=2..JM, including the North
    Pole row) -- the separate "J=J_1STG+1" block in the real source is a pure domain-decomposition
    (MPI halo) artifact that never fires for this rundeck's serial/single-process execution
    (J_STOP_STGR already equals JM here), so it's not ported, matching this project's established
    precedent for skipping inactive parallel-domain-decomposition-only code paths.

    Note the real Fortran's write-target offsets: `BXX` is written at `(IM1,J,L)` (the WEST
    neighbor of the loop's own `I`), and `BYY` at `(I,J-1,L)` -- both intentional, not typos.
    Returns a dict with bxx,byy,bzz,azx,bzx,czx,aezx,ezx,cezx,azy,bzy,czy,aezy,ezy,cezy (each
    (IM+1,JM+1,LMO+1))."""
    shape = (IM + 1, JM + 1, LMO + 1)
    out = {}
    for name in ["bxx", "byy", "bzz", "azx", "bzx", "czx", "aezx", "ezx", "cezx",
                 "azy", "bzy", "czy", "aezy", "ezy", "cezy"]:
        out[name] = np.zeros(shape)

    for l in range(1, LMO + 1):
        for j in range(2, JM + 1):
            im1 = IM
            for i in range(1, IM + 1):
                if lmm[i, j] >= l:
                    out["bxx"][im1, j, l] = (aix2[i, j, l] + aix0[im1, j, l] +
                                              aix3[i, j, l] + aix1[im1, j, l])
                    out["byy"][i, j - 1, l] = (aiy2[i, j, l] + aiy0[i, j - 1, l] +
                                                aiy3[i, j, l] + aiy1[i, j - 1, l])

                    if l > kpl[i, j]:
                        if l > 1:
                            out["bzz"][i, j, l - 1] = (
                                s2x1[i, j, l] + s2x3[i, j, l] + s2x0[i, j, l - 1] + s2x2[i, j, l - 1] +
                                s2y1[i, j, l] + s2y3[i, j, l] + s2y0[i, j, l - 1] + s2y2[i, j, l - 1])
                        out["azx"][i, j, l] = asx2[i, j, l]
                        out["bzx"][i, j, l] = asx0[i, j, l] - asx2[i, j, l]
                        out["czx"][i, j, l] = -asx0[i, j, l]
                        out["aezx"][i, j, l] = asx3[i, j, l]
                        out["ezx"][i, j, l] = asx1[i, j, l] - asx3[i, j, l]
                        out["cezx"][i, j, l] = -asx1[i, j, l]
                        out["azy"][i, j, l] = asy2[i, j, l]
                        out["bzy"][i, j, l] = asy0[i, j, l] - asy2[i, j, l]
                        out["czy"][i, j, l] = -asy0[i, j, l]
                        out["aezy"][i, j, l] = asy3[i, j, l]
                        out["ezy"][i, j, l] = asy1[i, j, l] - asy3[i, j, l]
                        out["cezy"][i, j, l] = -asy1[i, j, l]
                    elif l > 1:
                        out["bzz"][i, j, l - 1] = 0.0
                im1 = i

    return out


def _compute_fxx_fyy_fzz_fzx_fzy(lmm, lmu, lmv, kpl, tr, bxx, byy, bzz, azx, bzx, czx, aezx, ezx,
                                  cezx, azy, bzy, czy, aezy, ezy, cezy, dt4, dt4dx_of_j, dt4dy_of_j,
                                  bydyv, bydzv):
    """GMFEXP's own main loop (OCNGM.f:361-463, QCROSS branches excluded -- FXZ/FYZ stay 0).
    Never touches the pole rows (its own J range is 2..JM-1, unlike ISOSLOPE4/GMKDIF)."""
    shape = (IM + 1, JM + 1, LMO + 1)
    fxx = np.zeros(shape); fyy = np.zeros(shape); fzz = np.zeros(shape)
    fzx = np.zeros(shape); fzy = np.zeros(shape)

    for l in range(1, LMO + 1):
        for j in range(2, JM):  # J_STRT_SKP..J_STOP_SKP, serial: 2..JM-1 (excludes poles)
            dt4dx = dt4dx_of_j[j]
            dt4dy = dt4dy_of_j[j]
            for i in range(1, IM + 1):
                if lmm[i, j] <= 0:
                    continue
                im1 = IM if i == 1 else i - 1
                ip1 = 1 if i == IM else i + 1
                if lmu[im1, j] >= l:
                    fxx[im1, j, l] = dt4dx * bxx[im1, j, l] * (tr[im1, j, l] - tr[i, j, l])
                if lmv[i, j] >= l:
                    fyy[i, j, l] = dt4 * byy[i, j, l] * (tr[i, j, l] - tr[i, j + 1, l]) * bydyv[j]
                if lmm[i, j] > l:
                    if kpl[i, j] <= l:
                        fzz[i, j, l] = dt4 * bzz[i, j, l] * (tr[i, j, l + 1] - tr[i, j, l]) * bydzv[i, j, l]
                    fzx[i, j, l] = dt4dx * (bzx[i, j, l] * tr[i, j, l] + azx[i, j, l] * tr[im1, j, l] +
                                            czx[i, j, l] * tr[ip1, j, l] + ezx[i, j, l + 1] * tr[i, j, l + 1])
                    if lmm[im1, j] > l:
                        fzx[i, j, l] += dt4dx * aezx[i, j, l + 1] * tr[im1, j, l + 1]
                    if lmm[ip1, j] > l:
                        fzx[i, j, l] += dt4dx * cezx[i, j, l + 1] * tr[ip1, j, l + 1]
                    fzy[i, j, l] = dt4dy * (bzy[i, j, l] * tr[i, j, l] + azy[i, j, l] * tr[i, j - 1, l] +
                                            czy[i, j, l] * tr[i, j + 1, l] + ezy[i, j, l + 1] * tr[i, j, l + 1])
                    if lmm[i, j - 1] > l:
                        fzy[i, j, l] += dt4dy * aezy[i, j, l + 1] * tr[i, j - 1, l + 1]
                    if lmm[i, j + 1] > l:
                        fzy[i, j, l] += dt4dy * cezy[i, j, l + 1] * tr[i, j + 1, l + 1]
    return fxx, fyy, fzz, fzx, fzy


def _compute_fluxes(lmm, lmu, lmv, mo, fxx, fyy, fzz, fzx, fzy, bxx, byy, txm0, tym0,
                     dt4, bydxp, bydyp, bydh, dxypo):
    """OCNGM.f's computeFluxes (496-658, QCROSS excluded -- FXZ/FYZ are 0). Converts
    FXX/FYY/FZZ/FZX/FZY into flux_x/flux_y/flux_z and adjusts TXM/TYM by the diagonal terms.
    Handles both poles explicitly (unlike GMFEXP's own main loop). RGMI=1.0 (D48).
    Returns flux_x, flux_y, flux_z, txm, tym."""
    RGMI = 1.0
    shape = (IM + 1, JM + 1, LMO + 1)
    flux_x = np.zeros(shape); flux_y = np.zeros(shape); flux_z = np.zeros(shape)
    txm = txm0.copy(); tym = tym0.copy()

    for l in range(1, LMO + 1):
        for j in range(2, JM):  # J_STRT_SKP..J_STOP_SKP, serial: 2..JM-1
            for i in range(1, IM + 1):
                if lmm[i, j] <= 0:
                    continue
                im1 = IM if i == 1 else i - 1
                if lmu[im1, j] >= l:
                    mofx = (mo[im1, j, l] + mo[i, j, l]) * dxypo[j] * bydxp[j] * 0.5
                    flux_x[i, j, l] = fxx[im1, j, l] * mofx
                if l <= lmm[i, j]:
                    txm[i, j, l] = (txm[i, j, l] - 3.0 * (fxx[im1, j, l] + fxx[i, j, l]) *
                                    mo[i, j, l] * dxypo[j] * bydxp[j]) / (
                        1.0 + 6.0 * dt4 * (bxx[im1, j, l] + bxx[i, j, l]) * bydxp[j] ** 2)
                if lmv[i, j - 1] >= l:
                    mofy = ((mo[i, j - 1, l] * bydyp[j - 1] * dxypo[j - 1]) +
                            (mo[i, j, l] * bydyp[j] * dxypo[j])) * 0.5
                    flux_y[i, j, l] = fyy[i, j - 1, l] * mofy
                if l <= lmm[i, j]:
                    tym[i, j, l] = (tym[i, j, l] - 3.0 * (fyy[i, j - 1, l] + fyy[i, j, l]) *
                                    mo[i, j, l] * dxypo[j] * bydyp[j]) / (
                        1.0 + 6.0 * dt4 * (byy[i, j - 1, l] + byy[i, j, l]) * bydyp[j] ** 2)
                if lmm[i, j] > l:
                    mofz = ((mo[i, j, l + 1] * bydh[i, j, l + 1]) +
                            (mo[i, j, l] * bydh[i, j, l])) * dxypo[j] * 0.5
                    flux_z[i, j, l] = (fzz[i, j, l] + (fzx[i, j, l] + fzy[i, j, l]) * (1.0 + RGMI)) * mofz

        # North polar box
        for i in range(1, IM + 1):
            if lmv[i, JM - 1] >= l:
                mofy = ((mo[i, JM - 1, l] * bydyp[JM - 1] * dxypo[JM - 1]) +
                        (mo[1, JM, l] * bydyp[JM] * dxypo[JM])) * 0.5
                flux_y[i, JM, l] = fyy[i, JM - 1, l] * mofy
        if lmm[1, JM] > l:
            mofz = ((mo[1, JM, l + 1] * bydh[1, JM, l + 1]) +
                    (mo[1, JM, l] * bydh[1, JM, l])) * dxypo[JM] * 0.5
            flux_z[1, JM, l] = (fzz[1, JM, l] + fzy[1, JM, l] * (1.0 + RGMI)) * mofz

        # South polar box
        for i in range(1, IM + 1):
            if lmv[i, 2] >= l:
                mofy = ((mo[i, 2, l] * bydyp[2] * dxypo[2]) +
                        (mo[1, 1, l] * bydyp[1] * dxypo[1])) * 0.5
                flux_y[i, 1, l] = fyy[i, 2, l] * mofy
        if lmm[1, 1] > l:
            mofz = ((mo[1, 1, l + 1] * bydh[1, 1, l + 1]) +
                    (mo[1, 1, l] * bydh[1, 1, l])) * dxypo[1] * 0.5
            flux_z[1, 1, l] = (fzz[1, 1, l] + fzy[1, 1, l] * (1.0 + RGMI)) * mofz

    return flux_x, flux_y, flux_z, txm, tym


def _wrap_adjust_fluxes(lmm, trm0, flux_x, flux_y, flux_z):
    """OCNGM.f's wrapAdjustFluxes (660-810): the QLIMIT=.TRUE. (salt) path -- a global-sum-based
    limiter preventing TRM from going negative, applied via a multiplicative correction to
    positive flux convergences. GIJL diagnostics not ported. Returns trm."""
    trm = trm0.copy()
    shape = (IM + 1, JM + 1, LMO + 1)
    conv = np.zeros(shape)

    for l in range(LMO, 0, -1):
        for j in range(2, JM):
            for i in range(1, IM + 1):
                ip1 = 1 if i == IM else i + 1
                conv[i, j, l] = (flux_x[i, j, l] - flux_x[ip1, j, l] +
                                 flux_y[i, j, l] - flux_y[i, j + 1, l])
        conv[1, JM, l] = _fortran_sum(flux_y[1:IM + 1, JM, l]) / IM
        conv[1, 1, l] = _fortran_sum(flux_y[1:IM + 1, 1, l]) / IM
        if l < LMO:
            for j in range(2, JM):
                for i in range(1, IM + 1):
                    conv[i, j, l] += flux_z[i, j, l]
                    conv[i, j, l + 1] -= flux_z[i, j, l]
            conv[1, JM, l] += flux_z[1, JM, l]
            conv[1, JM, l + 1] -= flux_z[1, JM, l]
            conv[1, 1, l] += flux_z[1, 1, l]
            conv[1, 1, l + 1] -= flux_z[1, 1, l]

    for l in range(1, LMO + 1):
        trm[2:IM + 1, JM, l] = trm[1, JM, l]
        conv[2:IM + 1, JM, l] = conv[1, JM, l]
        trm[2:IM + 1, 1, l] = trm[1, 1, l]
        conv[2:IM + 1, 1, l] = conv[1, 1, l]

    convadj_j = np.zeros(JM + 1)
    convpos_j = np.zeros(JM + 1)
    for l in range(1, LMO + 1):
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if lmm[i, j] < l:
                    continue
                if conv[i, j, l] > 0.0:
                    convpos_j[j] += conv[i, j, l]
                elif conv[i, j, l] < -trm[i, j, l]:
                    convadj_j[j] += -trm[i, j, l] - conv[i, j, l]
                    conv[i, j, l] = -trm[i, j, l]

    sumpos = _fortran_sum(convpos_j[1:JM + 1])
    sumadj = _fortran_sum(convadj_j[1:JM + 1])
    posadj = 1.0 - sumadj / sumpos if sumpos > 0.0 else 1.0

    for l in range(1, LMO + 1):
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if lmm[i, j] < l:
                    continue
                if conv[i, j, l] > 0.0:
                    conv[i, j, l] *= posadj
                trm[i, j, l] = max(0.0, trm[i, j, l] + conv[i, j, l])

    return trm


def _add_fluxes(lmm, lmu, lmv, trm0, flux_x, flux_y, flux_z):
    """OCNGM.f's addFluxes (813-995): the QLIMIT=.FALSE. (enthalpy) path -- straightforward
    flux-divergence application, no limiter. GIJL diagnostics not ported. Returns trm."""
    trm = trm0.copy()

    for l in range(1, LMO + 1):
        for j in range(2, JM):
            im1 = IM
            for i in range(1, IM + 1):
                if lmm[i, j] > 0:
                    if lmu[im1, j] >= l:
                        trm[i, j, l] += flux_x[i, j, l]
                        trm[im1, j, l] -= flux_x[i, j, l]
                    if lmv[i, j - 1] >= l:
                        trm[i, j, l] += flux_y[i, j, l]
                        trm[i, j - 1, l] -= flux_y[i, j, l]
                im1 = i

        strnp = 0.0
        for i in range(1, IM + 1):
            if lmv[i, JM - 1] >= l:
                strnp += flux_y[i, JM, l]
                trm[i, JM - 1, l] -= flux_y[i, JM, l]
        trm[1, JM, l] += strnp / IM

        strsp = 0.0
        for i in range(1, IM + 1):
            if lmv[i, 2] >= l:
                strsp += flux_y[i, 1, l]
                trm[i, 2, l] -= flux_y[i, 1, l]
        trm[1, 1, l] += strsp / IM

    for l in range(1, LMO + 1):
        for j in range(2, JM):
            for i in range(1, IM + 1):
                if lmm[i, j] <= 0 or lmm[i, j] < l:
                    continue
                if lmm[i, j] > l:
                    rfzt = flux_z[i, j, l]
                    trm[i, j, l] += rfzt
                    trm[i, j, l + 1] -= rfzt
        if lmm[1, JM] >= l and lmm[1, JM] > l:
            rfzt = flux_z[1, JM, l]
            trm[1, JM, l] += rfzt
            trm[1, JM, l + 1] -= rfzt
        if lmm[1, 1] >= l and lmm[1, 1] > l:
            rfzt = flux_z[1, 1, l]
            trm[1, 1, l] += rfzt
            trm[1, 1, l + 1] -= rfzt

    return trm


def gmfexp(lmm, lmu, lmv, mo, trm0, txm0, tym0, tzm0, qlimit,
           bxx, byy, bzz, azx, bzx, czx, aezx, ezx, cezx,
           azy, bzy, czy, aezy, ezy, cezy, kpl, bydh, bydzv):
    """OCNGM.f's GMFEXP (330-499) + computeFluxes + wrapAdjustFluxes/addFluxes -- the actual
    Gent-McWilliams skew-flux application to a tracer (G0M with qlimit=False, S0M with
    qlimit=True). TZM is never updated (the real Fortran's own TZM-update code is commented out
    in full -- verified from source, not an oversight here). GIJL diagnostics not ported.
    Returns trm, txm, tym, tzm (tzm returned unchanged)."""
    _, _, dxpo, dypo, _, dyvo, _, dxypo = geomo_dyn_arrays()
    bydxp = np.zeros_like(dxpo); bydxp[1:] = 1.0 / dxpo[1:]
    bydyp = np.zeros_like(dypo); bydyp[1:] = 1.0 / dypo[1:]
    # DYVO[JM]=0 by construction (no V-points at the pole, established in odhorz_ff.py's
    # geomo_dyn_arrays) -- guard to avoid a real-but-harmless 1/0 (BYDYV[JM] is never read by
    # any active computation, matching the established pattern throughout this project).
    bydyv = np.zeros_like(dyvo); bydyv[1:JM] = 1.0 / dyvo[1:JM]

    dt4 = 0.25 * DTS
    dt4dx_of_j = dt4 * bydxp
    dt4dy_of_j = dt4 * bydyp

    tr = np.zeros((IM + 1, JM + 1, LMO + 1))
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            for l in range(1, lmm[i, j] + 1):
                tr[i, j, l] = trm0[i, j, l] / (dxypo[j] * mo[i, j, l])

    fxx, fyy, fzz, fzx, fzy = _compute_fxx_fyy_fzz_fzx_fzy(
        lmm, lmu, lmv, kpl, tr, bxx, byy, bzz, azx, bzx, czx, aezx, ezx, cezx,
        azy, bzy, czy, aezy, ezy, cezy, dt4, dt4dx_of_j, dt4dy_of_j, bydyv, bydzv)

    flux_x, flux_y, flux_z, txm, tym = _compute_fluxes(
        lmm, lmu, lmv, mo, fxx, fyy, fzz, fzx, fzy, bxx, byy, txm0, tym0,
        dt4, bydxp, bydyp, bydh, dxypo)

    if qlimit:
        trm = _wrap_adjust_fluxes(lmm, trm0, flux_x, flux_y, flux_z)
    else:
        trm = _add_fluxes(lmm, lmu, lmv, trm0, flux_x, flux_y, flux_z)

    return trm, txm, tym, tzm0.copy()
