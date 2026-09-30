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
