"""Full-fidelity port of OCNDYN2.f's OBDRAG2 -- Stage 2 of the DYNSI/ocean port, D38.

OBDRAG2 applies an implicit bottom-drag deceleration to the ocean's BOTTOM layer at each
column -- the layer index varies per cell (`L=LMU(I,J)` for the east-edge U/VOD pair,
`L=LMV(I,J)` for the north-edge V/UOD pair; everywhere else is untouched). The drag factor is
`bdragfac = BDRAGX*sqrt(WSQ)` (BDRAGX=1.0, a quadratic-drag coefficient) where WSQ is the local
squared current speed (UO^2+VOD^2 at east edges, VO^2+UOD^2 at north edges, plus a 1e-20 floor
to avoid a zero-speed singularity), and the velocity is scaled by
`(MO_l+MO_r) / (MO_l+MO_r + 2*DTS*bdragfac)` -- an implicit (unconditionally stable) linear drag.

Live physics: called unconditionally from `OCEANS` (`OCNDYN2.f`) when `OBottom_drag == 1` (true
for this rundeck). `OCN_GISS_TURB` is NOT `#define`d for this build (confirmed from the compiled
rundeck_opts.h's active flag list), so the `taubx`/`tauby`/`rhobot`/`idrag` tidal-enhancement
branch never compiles in -- only the simple `bdragfac=BDRAGX*sqrt(WSQ)` path is live. `LMU`/`LMV`
(same static depth masks validated in D36/D37) are recorded directly from this delta's own dump.
"""
import numpy as np

IM, JM, LMO = 72, 46, 13
BDRAGX = 1.0
DTS = 1800.0


def obdrag2(lmu, lmv, mo, uo0, vo0, uod0, vod0):
    """Direct port of OBDRAG2 (OCNDYN2.f:2592-2682).

    `lmu`,`lmv`: integer depth masks, shape (IM+1, JM+1), 1-indexed.
    `mo`,`uo0`,`vo0`,`uod0`,`vod0`: real fields, shape (IM+1, JM+1, LMO+1), 1-indexed.
    Returns (uo, vo, uod, vod), each a fresh copy (untouched cells retain their input value).
    """
    uo = uo0.copy()
    vo = vo0.copy()
    uod = uod0.copy()
    vod = vod0.copy()

    # ---- Reduce ocean current at east edges of cells (Fortran J=Max(J1O,J1)..JNP = 2..JM-1) ----
    for j in range(2, JM):
        i = IM
        for ip1 in range(1, IM + 1):
            if lmu[i, j] > 0:
                l = int(lmu[i, j])
                wsq = uo0[i, j, l] ** 2 + vod0[i, j, l] ** 2 + 1e-20
                bdragfac = BDRAGX * np.sqrt(wsq)
                denom = mo[i, j, l] + mo[ip1, j, l]
                factor = denom / (denom + DTS * bdragfac * 2.0)
                uo[i, j, l] = uo0[i, j, l] * factor
                vod[i, j, l] = vod0[i, j, l] * factor
            i = ip1

    # ---- Reduce ocean current at north edges of cells ----
    for j in range(2, JM):
        for i in range(1, IM + 1):
            if lmv[i, j] > 0:
                l = int(lmv[i, j])
                wsq = vo0[i, j, l] ** 2 + uod0[i, j, l] ** 2 + 1e-20
                bdragfac = BDRAGX * np.sqrt(wsq)
                denom = mo[i, j, l] + mo[i, j + 1, l]
                factor = denom / (denom + DTS * bdragfac * 2.0)
                vo[i, j, l] = vo0[i, j, l] * factor
                uod[i, j, l] = uod0[i, j, l] * factor

    return uo, vo, uod, vod
