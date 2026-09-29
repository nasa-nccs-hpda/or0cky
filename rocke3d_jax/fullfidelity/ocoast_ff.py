"""Full-fidelity port of OCNDYN.f's OCOAST -- Stage 2 of the DYNSI/ocean port, D37.

OCOAST damps the horizontal (X and Y) gradient moments of the ocean's enthalpy/salt tracer
fields (GXMO/SXMO, GYMO/SYMO -- the linear-in-cell moments used by the QUS advection scheme)
in coastal grid boxes, where a shallower neighbor makes the deeper layers' horizontal gradient
estimate less reliable. For each cell I with neighbors at IM1 (west) and IP1 (east) [or J-1/J+1
for the Y-pass], layers from `LMIN = MIN(depth(neighbor1), depth(neighbor2)) + 1` down to the
cell's own depth get multiplied by a single relaxation factor `REDUCE = 1 - DTS/(86400*20)`
(a ~20-day damping time constant). Layers shallower than `LMIN` (i.e. where both neighbors are
at least as deep) are left untouched.

Live physics: called unconditionally from `OCEANS` (`OCNDYN2.f`) when `OCoastal_drag == 1`
(true for this rundeck). No external files; `DTS = DTSRC = 1800.0` (`OCNDYN.f:405`, this
rundeck's confirmed timestep) and `SECONDS_PER_DAY = 86400.0` (standard GISS `TimeConstants_mod`
value) are both compile-time-known constants, not recorded inputs -- `REDUCE` is fully analytic.
LMM (the ocean depth mask) is recorded directly from the dump (same file-sourced-bathymetry
treatment as D36's LMU/LMV).
"""
import numpy as np

IM, JM, LMO = 72, 46, 13
DTS = 1800.0
SECONDS_PER_DAY = 86400.0
REDUCE = 1.0 - DTS / (SECONDS_PER_DAY * 20.0)


def ocoast(lmm, gxmo0, sxmo0, gymo0, symo0):
    """Direct port of OCOAST (OCNDYN.f:4514-4570).

    `lmm`: integer depth mask, shape (IM+1, JM+1), 1-indexed (row/col 0 unused).
    `gxmo0`,`sxmo0`,`gymo0`,`symo0`: real fields, shape (IM+1, JM+1, LMO+1), 1-indexed in I,J,L.
    Returns updated (gxmo, sxmo, gymo, symo), each a fresh copy (untouched cells retain their
    input value exactly, matching Fortran's in-place update leaving other cells unchanged).
    """
    gxmo = gxmo0.copy()
    sxmo = sxmo0.copy()
    gymo = gymo0.copy()
    symo = symo0.copy()

    # ---- Reduce West-East gradient of tracers ----
    for j in range(2, JM):  # Fortran J_0S..J_1S = 2..JM-1
        im1 = IM - 1
        i = IM
        for ip1 in range(1, IM + 1):
            lmin = min(lmm[im1, j], lmm[ip1, j]) + 1
            lmax = lmm[i, j]
            for l in range(lmin, lmax + 1):
                gxmo[i, j, l] = gxmo0[i, j, l] * REDUCE
                sxmo[i, j, l] = sxmo0[i, j, l] * REDUCE
            im1 = i
            i = ip1

    # ---- Reduce South-North gradient of tracers ----
    for j in range(2, JM):
        for i in range(1, IM + 1):
            lmin = min(lmm[i, j - 1], lmm[i, j + 1]) + 1
            lmax = lmm[i, j]
            for l in range(lmin, lmax + 1):
                gymo[i, j, l] = gymo0[i, j, l] * REDUCE
                symo[i, j, l] = symo0[i, j, l] * REDUCE

    return gxmo, sxmo, gymo, symo
