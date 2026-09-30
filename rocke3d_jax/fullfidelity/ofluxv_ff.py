"""Full-fidelity port of OCNDYN2.f's OFLUXV + OADVUZ -- Stage 2, D43.

OFLUXV is the "long-timestep vertical redistribution of mass" step, called once per NOCEAN
iteration from OCEANS right after the leapfrog ODHORZ loop closes. Per column (with at least 2
real layers -- single-layer columns are left untouched, see below):

1. Snapshots MO into MB (pre-redistribution mass, at every real layer including the pole row).
2. Computes MSUM = (OPBOT-OPRESS)/GRAV -- the true total-column mass implied by the bottom
   pressure -- and rescales each layer L's mass to MSUM*DZO(L)/ZE(LMM(I,J)) (the L13
   fixed-layering fraction of the true total; the bottom layer uses DZO(LMM(I,J)) itself, so the
   layers sum back to exactly MSUM), accumulating the vertical mass flux this rescaling implies
   into SMW.
3. Uses that implied vertical mass flux (via `OADVUZ`, a "simplest upstream scheme") to
   vertically advect the horizontal velocities UO and VO -- their column-integrated momentum is
   preserved even though the layer thicknesses just changed.

**Single-layer columns are a genuine edge case, verified from the source, not assumed**: both the
layer-1 redistribution and the bottom-layer update are gated on "column reaches layer 2"
(`nbyzm(j,2)` in the Fortran) -- for LMM(I,J)==1, MO(I,J,1) is never touched by this routine at
all.

Unlike OFLUXV's neighbors (ODHORZ/ODHORZ0), this delta needs no external file or polar filter --
`DZO`/`ZE` (the L13 fixed layer thicknesses/edges) are already analytically available (D34's
`DZO_L13`). `OPRESS` (atmosphere/ice pressure load on the ocean surface) is a recorded real input,
not yet ported upstream, same pattern as D29's `GAIRX`/`GWATX`.
"""
import numpy as np
from osourc_ff import DZO_L13

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665


def _ze_array():
    ze = [0.0] * (LMO + 1)
    for l in range(1, LMO + 1):
        ze[l] = ze[l - 1] + DZO_L13[l - 1]
    return ze


ZE = _ze_array()                 # 1-indexed: ZE[0..LMO]
DZO = [0.0] + list(DZO_L13)      # 1-indexed: DZO[1..LMO]


def oadvuz(r0, m0, mw, dt, active):
    """Direct port of OADVUZ (OCNDYN2.f:2471-2521): simplest upstream vertical-mass-flux
    advection. `r0`,`m0`,`mw` shape (IM+1,JM+1,LMO+1), 1-indexed. `active(i,j,l)` is the caller's
    mask (U- or V-domain). Returns r (fresh copy); the caller's local M working array's final
    value is not separately validated, so it is not returned."""
    r = r0.copy()
    m = m0.copy()
    for j in range(1, JM + 1):
        cmup = np.zeros(IM + 1)
        fmup = np.zeros(IM + 1)
        for l in range(1, LMO + 1):
            for i in range(1, IM + 1):
                if not active(i, j, l):
                    continue
                cm = dt * mw[i, j, l]
                fm = cm * r[i, j, l] if cm >= 0.0 else cm * r[i, j, l + 1]
                mnew = m[i, j, l] + cmup[i] - cm
                r[i, j, l] = (r[i, j, l] * m[i, j, l] + (fmup[i] - fm)) / mnew
                cmup[i] = cm
                fmup[i] = fm
                m[i, j, l] = mnew
    return r


def ofluxv(lmm, lmu, lmv, dtolf, opbot0, opress0, mo0, uo0, vo0):
    """Direct port of OFLUXV (OCNDYN2.f:711-800). All 2D fields shape (IM+1,JM+1); all 3D fields
    shape (IM+1,JM+1,LMO+1); 1-indexed. Returns (mo, uo, vo), fresh copies."""

    def m_active(i, j, l):
        """nbyzm(j,l) mask: ordinary LMM(I,J)>=L everywhere except J=JM, restricted to I=1 only
        (the D40 North-Pole finding, confirmed to apply uniformly to nbyzm)."""
        if j == JM:
            return i == 1 and lmm[1, JM] >= l
        return lmm[i, j] >= l

    def u_active(i, j, l):
        return j != JM and lmu[i, j] >= l

    def v_active(i, j, l):
        return j != JM and lmv[i, j] >= l

    # ---- Snapshot MO into MB (including the pole row, unconditionally at real layers) ----
    mb = mo0.copy()
    if lmm[1, JM] >= 1:
        for l in range(1, int(lmm[1, JM]) + 1):
            for i in range(2, IM + 1):
                mb[i, JM, l] = mo0[i, JM, l]

    mo = mo0.copy()
    msum = np.zeros((IM + 1, JM + 1))
    smw = np.zeros((IM + 1, JM + 1, LMO + 1))

    # ---- Layer 1: rescale to the L13 fraction of the true column mass MSUM ----
    for j in range(2, JM + 1):
        for i in range(1, IM + 1):
            if m_active(i, j, 2):  # column reaches layer 2 (real gate, verified from source)
                msum[i, j] = (opbot0[i, j] - opress0[i, j]) / GRAV
                lm = int(lmm[i, j])
                mfinal = msum[i, j] * DZO[1] / ZE[lm]
                smw[i, j, 1] = mo0[i, j, 1] - mfinal
                mo[i, j, 1] = mfinal

    # ---- Layers 2..LMO-1: same rescale, gated on "not yet the bottom layer" ----
    for l in range(2, LMO):
        for j in range(2, JM + 1):
            for i in range(1, IM + 1):
                if m_active(i, j, l + 1):  # column reaches layer l+1, i.e. l is not the bottom
                    lm = int(lmm[i, j])
                    mfinal = msum[i, j] * DZO[l] / ZE[lm]
                    smw[i, j, l] = smw[i, j, l - 1] + (mo0[i, j, l] - mfinal)
                    mo[i, j, l] = mfinal

    # ---- Bottom layer: rescale using its own thickness (conserves the column total exactly) ----
    for j in range(2, JM + 1):
        for i in range(1, IM + 1):
            if m_active(i, j, 2):
                lm = int(lmm[i, j])
                mfinal = msum[i, j] * DZO[lm] / ZE[lm]
                mo[i, j, lm] = mfinal

    # ---- Fill pole (uniform across longitudes) ----
    if lmm[1, JM] >= 1:
        for l in range(1, int(lmm[1, JM]) + 1):
            for i in range(2, IM + 1):
                mo[i, JM, l] = mo[1, JM, l]
                smw[i, JM, l] = smw[1, JM, l]

    # ---- Vertical advection of UO (U-points: I-neighbor-averaged MB/SMW) ----
    # NOTE on the DXYPO/DTOLF bookkeeping: the Fortran multiplies SMW's accumulation by
    # DXYPO(J)/DTOLF at write time, then divides by DXYPO(J) again (or DXYPO(J+1) for the V-point
    # average's second term) when averaging it onto U/V-points. At U-points both neighbor terms
    # share the SAME J, so DXYPO(J) cancels completely, leaving a residual /DTOLF; at V-points
    # each term's own DXYPO(J) or DXYPO(J+1) cancels individually against its own factor, leaving
    # the same residual /DTOLF in both terms. Tracked here as raw (mwfac-free) SMW above, so only
    # the shared /DTOLF remains to apply at the U/V-point averaging step below.
    mtmp_u = np.zeros((IM + 1, JM + 1, LMO + 1))
    mwtmp_u = np.zeros((IM + 1, JM + 1, LMO + 1))
    for l in range(1, LMO + 1):
        for j in range(2, JM):
            for i in range(1, IM + 1):
                if u_active(i, j, l):
                    ip1 = 1 if i == IM else i + 1
                    mtmp_u[i, j, l] = 0.5 * (mb[i, j, l] + mb[ip1, j, l])
                    mwtmp_u[i, j, l] = 0.5 * (smw[i, j, l] + smw[ip1, j, l]) / dtolf
    for j in range(2, JM):
        for i in range(1, IM + 1):
            if lmu[i, j] >= 1:
                mwtmp_u[i, j, int(lmu[i, j])] = 0.0
    uo = oadvuz(uo0, mtmp_u, mwtmp_u, dtolf, u_active)

    # ---- Vertical advection of VO (V-points: J-neighbor-averaged MB/SMW) ----
    mtmp_v = np.zeros((IM + 1, JM + 1, LMO + 1))
    mwtmp_v = np.zeros((IM + 1, JM + 1, LMO + 1))
    for l in range(1, LMO + 1):
        for j in range(2, JM):
            for i in range(1, IM + 1):
                if v_active(i, j, l):
                    mtmp_v[i, j, l] = 0.5 * (mb[i, j, l] + mb[i, j + 1, l])
                    mwtmp_v[i, j, l] = 0.5 * (smw[i, j, l] + smw[i, j + 1, l]) / dtolf
    for j in range(2, JM):
        for i in range(1, IM + 1):
            if lmv[i, j] >= 1:
                mwtmp_v[i, j, int(lmv[i, j])] = 0.0
    vo = oadvuz(vo0, mtmp_v, mwtmp_v, dtolf, v_active)

    return mo, uo, vo
