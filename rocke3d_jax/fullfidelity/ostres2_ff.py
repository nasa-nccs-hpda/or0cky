"""Full-fidelity port of OCNDYN2.f's OSTRES2 -- Stage 2 of the DYNSI/ocean port, D36.

OSTRES2 applies the atmospheric surface stress (oDMUA/oDMVA, momentum flux down into open
ocean, recorded as a real input -- not yet ported upstream) and the sea-ice stress (oDMUI/oDMVI,
also recorded) to the ocean's layer-1 U/V velocities on the C grid (UO/VO) and their D-grid
relaxation targets (UOD/VOD). Live physics: called unconditionally once per DTsrc step from the
real `OCEANS` driver in OCNDYN2.f, right after GROUND_OC.

This is the first delta from OCNDYN2.f rather than OCNDYN.f: while scoping OFLUX/OPFIL as the
next Stage 2 candidate, every call site of OCNDYN.f's OFLUX/OADVM/OADVV/OPGF/OVtoM/OMtoV/OSTRES/
OBDRAG/OPFIL was found to be dead code -- OCNDYN.f's entire driver is literally named
`OCEANS_old` and commented out in full, superseded by a live rewrite in OCNDYN2.f (`OCEANS`,
`OFLUXV`, `ODHORZ`/`ODHORZ0`, `OPFIL2`, `OSTRES2`, `OBDRAG2`, the `OADVT2` family). OSTRES2 was
picked as the smallest live piece: no external files, no FFT, ~100 lines.

Static ocean-grid geometry (DXYSO/DXYNO/DXYVO/COSIC/SINIC) is fully analytic -- the same
standard lat-lon formulas as the atmosphere grid (OGEOM.f's GEOMO, RADIUS=6371000.0 per D29),
validated exactly against the dumped geometry record before use. LMU/LMV (ocean depth masks at
the U/V staggered points) are `MIN(LMM(i,j),LMM(i+1,j))` / `MIN(LMM(i,j),LMM(i,j+1))`
(OCNDYN.f:488,499) -- simple derived quantities from file-sourced bathymetry (LMM itself) -- but
are taken here as a recorded input directly (not re-derived from LMM), since LMM's full 2D grid
is not independently available in this delta's dump and LMU/LMV are exactly what OSTRES2 itself
consumes.

South Pole: OSTRES2 has no south-pole special case at all (unlike OVtoM/OMtoV's dead, commented-
out QSP branches) -- this ocean grid's south pole sits inside Antarctic land, so there is no
polar-ocean singularity there. Only the North Pole (Arctic Ocean) needs special handling, via
IVNP (the "virtual" V-as-U index, confirmed IVNP=IM/4=18 from the dump) and the COSIC/SINIC
rotation.
"""
import numpy as np

IM, JM = 72, 46
RADIUS = 6371000.0


def geomo_arrays():
    """Analytically derive DXYSO, DXYNO, DXYVO (length JM, 1-indexed as [0]=unused),
    COSIC, SINIC (length IM, 1-indexed as [0]=unused) exactly as OGEOM.f's GEOMO does.
    Validated bitwise-exact against ffz_ostres2_geom.bin (D36)."""
    twopi = 2.0 * np.pi
    dlon = twopi / IM
    fjeq = 0.5 * (1 + JM)
    odlat_dg = round(180.0 / (JM - 1))  # JM==46 -> "half polar box" branch
    dlat = odlat_dg * np.pi / 180.0

    dxyp = np.zeros(JM + 1)
    for j in range(1, JM + 1):
        latn = dlat * (j + 0.5 - fjeq) if j != JM else twopi / 4
        lats = dlat * (j - 0.5 - fjeq) if j != 1 else -twopi / 4
        dxyp[j] = RADIUS * RADIUS * dlon * (np.sin(latn) - np.sin(lats))
    dxys = 0.5 * dxyp
    dxyn = 0.5 * dxyp

    dxyvo = np.zeros(JM + 1)
    for j in range(1, JM):
        dxyvo[j] = dxyn[j] + dxys[j + 1]

    cosic = np.zeros(IM + 1)
    sinic = np.zeros(IM + 1)
    for i in range(1, IM + 1):
        cosic[i] = np.cos((i - 0.5) * twopi / IM)
        sinic[i] = np.sin((i - 0.5) * twopi / IM)

    return dxys, dxyn, dxyvo, cosic, sinic


def ostres2(lmu, lmv, dmua, dmva, dmui, dmvi, mo1, uo0, vo0, uod0, vod0, ivnp=18):
    """Direct port of OSTRES2 (OCNDYN2.f:2478-2577).

    All 2D arrays are (IM+1, JM+1), 1-indexed (row/col 0 unused), matching Fortran (I,J).
    `lmu`, `lmv`: integer depth masks. `dmua`,`dmva`,`dmui`,`dmvi`,`mo1`: recorded real inputs
    (atmosphere/ice momentum flux, ocean layer-1 mass). `uo0`,`vo0`,`uod0`,`vod0`: layer-1
    state before the call. Returns updated (uo, vo, uod, vod), each a fresh (IM+1,JM+1) array
    (unmodified cells retain their input value, exactly as Fortran's in-place update leaves
    untouched cells unchanged).
    """
    dxyso, dxyno, dxyvo, cosic, sinic = geomo_arrays()

    uo = uo0.copy()
    vo = vo0.copy()
    uod = uod0.copy()
    vod = vod0.copy()

    # Surface stress applied to U component (J=2..JM-1, i.e. J_0S..J_1S for a single-process run)
    for j in range(2, JM):
        im1 = IM
        for i in range(1, IM + 1):
            ip1 = i
            i_ = im1
            if lmu[i_, j] > 0:
                uo[i_, j] = uo0[i_, j] + (dmua[i_, j] + dmua[ip1, j]
                                           + 2.0 * dmui[i_, j]) / (mo1[i_, j] + mo1[ip1, j])
            im1 = ip1

    # North Pole special case for UO
    uo[IM, JM] = uo0[IM, JM] + dmua[1, JM] / mo1[1, JM]
    uo[ivnp, JM] = uo0[ivnp, JM] + dmva[1, JM] / mo1[1, JM]

    # VOD update (uses dmvi at J-1, halo already resident since this is a full serial grid)
    for j in range(2, JM):
        im1 = IM
        for i in range(1, IM + 1):
            ip1 = i
            i_ = im1
            if lmu[i_, j] > 0:
                jm1 = j - 1
                vod[i_, j] = vod0[i_, j] + (
                    dmva[i_, j] + dmva[ip1, j]
                    + 0.5 * (dmvi[i_, jm1] + dmvi[ip1, jm1] + dmvi[i_, j] + dmvi[ip1, j])
                ) / (mo1[i_, j] + mo1[ip1, j])
            im1 = ip1

    # Surface stress applied to V component (J=2..JM-2)
    for j in range(2, JM - 1):
        for i in range(1, IM + 1):
            if lmv[i, j] > 0:
                jp1 = j + 1
                vo[i, j] = vo0[i, j] + (
                    dmva[i, j] * dxyno[j] + dmva[i, jp1] * dxyso[jp1]
                    + dmvi[i, j] * dxyvo[j]
                ) / (mo1[i, j] * dxyno[j] + mo1[i, jp1] * dxyso[jp1])

    # V stress at the North Pole (J=JM-1)
    for i in range(1, IM + 1):
        vo[i, JM - 1] = vo0[i, JM - 1] + (
            dmva[i, JM - 1] * dxyno[JM - 1]
            + (dmva[1, JM] * cosic[i] - dmua[1, JM] * sinic[i]) * dxyso[JM]
            + dmvi[i, JM - 1] * dxyvo[JM - 1]
        ) / (mo1[i, JM - 1] * dxyno[JM - 1] + mo1[1, JM] * dxyso[JM])

    # UOD update (J=2..JM-2)
    for j in range(2, JM - 1):
        im1 = IM
        for i in range(1, IM + 1):
            if lmv[i, j] > 0:
                jp1 = j + 1
                uod[i, j] = uod0[i, j] + (
                    dmua[i, j] + dmua[i, jp1]
                    + 0.5 * (dmui[im1, j] + dmui[i, j] + dmui[im1, jp1] + dmui[i, jp1])
                ) / (mo1[i, j] + mo1[i, jp1])
            im1 = i

    return uo, vo, uod, vod
