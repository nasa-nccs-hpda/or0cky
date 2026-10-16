"""Full-fidelity port of OCNDYN2.f's polar UOD/VOD relaxation block (inside `OCEANS`, the "relax
UOD,VOD toward 4-pt avgs of UO,VO" comment) plus its `polevel()` helper -- Stage 2 of the
DYNSI/ocean port, D39.

Each ocean layer L relaxes the D-grid velocities (UOD, defined at V-points; VOD, defined at
U-points -- the same cross-registration established in D36's OSTRES2/D38's OBDRAG2) toward a
4-point average of the C-grid velocities (UO, VO) at neighboring grid boxes, with a small
relaxation factor RELFAC=0.005 (a ~200-step, i.e. ~4-day, damping time constant at DTsrc=1800s).
Before relaxing, `polevel()` reconstructs the North Pole row of UO/VO (row JM) from the ring of V
velocities at J=JM-1, via a discrete wavenumber-1 (Fourier mode 1) projection -- the same
trigonometric pole-reconstruction idiom seen in D29's DYNSI pole handling and OVtoM/OMtoV's dead
south-pole code. South Pole: no special case exists (this ocean grid's south pole sits inside
Antarctic land, confirmed since D36).

`nbyzu`/`nbyzv`/`i1yzu`/`i2yzu`/`i1yzv`/`i2yzv` (the Fortran's precomputed per-(J,L) contiguous-
I-segment lists used to loop over ocean cells) are confirmed, by reading their construction site
(`OCNDYN.f:505-518`: `qexist(:) = (l <= lmu(:,j))` fed through a run-length-encoding helper), to
be an exact cached representation of "cells where LMU(I,J)>=L" (or LMV) -- so this port uses the
LMU/LMV point-masks directly (already validated in D36-D38) rather than reconstructing the
segment-list structure.
"""
import numpy as np

IM, JM, LMO = 72, 46, 13
RELFAC = 0.005


def geomo_pole_arrays():
    """Analytically derive COSIC, SINIC, COSU, SINU (length IM, 1-indexed, [0] unused) exactly as
    OGEOM.f's GEOMO does. COSIC/SINIC match D36's ostres2_ff.geomo_arrays(); COSU/SINU
    (`OGEOM.f`: SINU(I)=Sin(I*TWOPI/IM), COSU(I)=Cos(I*TWOPI/IM), SINU(IM)=0, COSU(IM)=1) are new
    to this delta."""
    twopi = 2.0 * np.pi
    i1 = np.arange(1, IM + 1, dtype=np.float64)
    cosic = np.zeros(IM + 1)
    sinic = np.zeros(IM + 1)
    cosic[1:] = np.cos((i1 - 0.5) * twopi / IM)
    sinic[1:] = np.sin((i1 - 0.5) * twopi / IM)
    cosu = np.zeros(IM + 1)
    sinu = np.zeros(IM + 1)
    cosu[1:] = np.cos(i1 * twopi / IM)
    sinu[1:] = np.sin(i1 * twopi / IM)
    sinu[IM] = 0.0
    cosu[IM] = 1.0
    return cosic, sinic, cosu, sinu


def polevel(u, v, l, lmv, cosic, sinic, cosu, sinu):
    """Direct port of polevel (OCNDYN2.f:1530-1564). Modifies u, v IN PLACE at row JM only
    (1-indexed (IM+1,JM+1,LMO+1) arrays). `l` is the 1-indexed layer being processed."""
    j = JM - 1
    unp = 0.0
    vnp = 0.0
    for i in range(1, IM + 1):
        if lmv[i, j] >= l:
            unp -= sinic[i] * v[i, j, l]
            vnp += cosic[i] * v[i, j, l]
    unp *= 2.0 / IM
    vnp *= 2.0 / IM
    for i in range(1, IM + 1):
        u[i, JM, l] = unp * cosu[i] + vnp * sinu[i]
        v[i, JM, l] = vnp * cosic[i] - unp * sinic[i]


def polerelax(lmu, lmv, uo0, vo0, uod0, vod0):
    """Direct port of the "relax UOD,VOD toward 4-pt avgs of UO,VO" block (OCNDYN2.f:179-228),
    including polevel(). All arrays (IM+1,JM+1,LMO+1), 1-indexed. Returns (uo,vo,uod,vod), fresh
    copies (uo/vo change only at row JM; uod/vod change only where the L-loop's masks fire)."""
    cosic, sinic, cosu, sinu = geomo_pole_arrays()

    uo = uo0.copy()
    vo = vo0.copy()
    uod = uod0.copy()
    vod = vod0.copy()

    for l in range(1, LMO + 1):
        polevel(uo, vo, l, lmv, cosic, sinic, cosu, sinu)

        # ---- UOD, pole-adjacent row J=JM-1 (doubled uo(:,JM,l) term) ----
        j = JM - 1
        if lmv[1, j] >= l:
            uod[1, j, l] = ((1.0 - RELFAC) * uod0[1, j, l] + RELFAC * 0.25 *
                            (uo[IM, j, l] + uo[1, j, l] + 2.0 * uo[1, j + 1, l]))
        for i in range(2, IM + 1):
            if lmv[i, j] >= l:
                uod[i, j, l] = ((1.0 - RELFAC) * uod0[i, j, l] + RELFAC * 0.25 *
                                (uo[i - 1, j, l] + uo[i, j, l] + 2.0 * uo[i, j + 1, l]))

        # ---- UOD, interior rows J=2..JM-2 ----
        for j in range(2, JM - 1):
            if lmv[1, j] >= l:
                uod[1, j, l] = ((1.0 - RELFAC) * uod0[1, j, l] + RELFAC * 0.25 *
                                (uo[IM, j, l] + uo[1, j, l] + uo[IM, j + 1, l] + uo[1, j + 1, l]))
            for i in range(2, IM + 1):
                if lmv[i, j] >= l:
                    uod[i, j, l] = ((1.0 - RELFAC) * uod0[i, j, l] + RELFAC * 0.25 *
                                    (uo[i - 1, j, l] + uo[i, j, l] + uo[i - 1, j + 1, l] + uo[i, j + 1, l]))

        # ---- VOD, all rows J=2..JM-1 (never touches the pole row) ----
        for j in range(2, JM):
            for i in range(1, IM):
                if lmu[i, j] >= l:
                    vod[i, j, l] = ((1.0 - RELFAC) * vod0[i, j, l] + RELFAC * 0.25 *
                                    (vo[i, j - 1, l] + vo[i + 1, j - 1, l] + vo[i, j, l] + vo[i + 1, j, l]))
            i = IM
            if lmu[i, j] >= l:
                vod[i, j, l] = ((1.0 - RELFAC) * vod0[i, j, l] + RELFAC * 0.25 *
                                (vo[i, j - 1, l] + vo[1, j - 1, l] + vo[i, j, l] + vo[1, j, l]))

    return uo, vo, uod, vod
