"""Full-fidelity port of OCNDYN2.f's ocnstate_derived + OCNMESO_DRV.f's densgrad/get_1d_mesodiff
-- Stage 2, D47. Cell-centered intensive thermodynamic state (G3D/S3D/P3D/VBAR/RHO) and the
horizontal/vertical density gradients (RHOX/RHOY/RHOMZ/BYRHOZ/DZV/BYDZV/BYDH) that feed the
Gent-McWilliams mesoscale-mixing scheme (GMKDIF/GMFEXP, OCNGM.f, scoped D46, not yet ported).

USE_OPGFQ=0 confirmed (D40) -- ocnstate_derived's QUS branch (OCNDYN2.f's `if(use_opgfq==1)`)
is dead code for this rundeck, not ported.

VOLGSP is the seawater-EOS lookup table (OFTAB, same external-file dependency as D35/D40); its
outputs (VUP/VDN for ocnstate_derived, plus VUPU/VDNU for densgrad's vertical gradient) are
recorded directly rather than re-implementing the table -- the established "record what's not
yet ported" pattern. TEMGSP (in-situ temperature, T3D) is similarly a lookup-table function;
T3D is dumped for completeness but not independently re-derived or consumed here, since nothing
in this delta's scope (densgrad) reads it.

RHOX/RHOY need two additional VOLGSP evaluations each (at inter-cell average pressures) --
recorded as final outputs rather than further decomposed, since nothing downstream in this
delta needs the individual terms; revisit if a future delta needs to validate them more finely.

densgrad reads OCEAN_DYN's module-level DH array, which is NOT recomputed by densgrad itself --
it's ODHORZ0's already-validated (D40) DH3D output, persisting unchanged from ODHORZ0's single
call earlier in the same OCEANS invocation (NOCEAN=1 confirmed D44). Reused directly rather than
re-dumped.
"""
import numpy as np
from odhorz_ff import geomo_dyn_arrays

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665
MESO_DIFFUSIVITY_CONST = 800.0  # decks/P2SAoM40.R: meso_diffusivity_const=800.


def ocnstate_derived(mo, g0m, gzm, s0m, szm, opress, lmm, vup, vdn):
    """OCNDYN2.f:1568-1706 (this delta's line numbering), live USE_OPGFQ=0 branch. All 3D
    fields shape (IM+1,JM+1,LMO+1), 2D (IM+1,JM+1); 1-indexed. `vup`,`vdn` (VOLGSP's recorded
    real outputs) shape (IM+1,JM+1,LMO+1) in this project's usual (i,j,l) convention -- NOT the
    Fortran's native (LMO,IM,JM) ordering (translated by ocnmeso_compare.py). Returns
    g3d, s3d, p3d, vbar, rho (each (IM+1,JM+1,LMO+1))."""
    dxypo = geomo_dyn_arrays()[-1]
    g3d = np.zeros((IM + 1, JM + 1, LMO + 1))
    s3d = np.zeros((IM + 1, JM + 1, LMO + 1))
    p3d = np.zeros((IM + 1, JM + 1, LMO + 1))
    vbar = np.zeros((IM + 1, JM + 1, LMO + 1))
    rho = np.zeros((IM + 1, JM + 1, LMO + 1))

    # The real main loop is nbyzm-driven: at J=JM (North Pole), nbyzm restricts iteration to
    # I=1 only (D40's established finding), so VUP/VDN/RHO are never touched there for I>1 --
    # they're filled purely by the pole-copy step below. Looping every I pointwise via LMM(i,j)
    # (ignoring this restriction) computes a transient, discarded 1/0 at those cells.
    def m_active(i, j, l):
        if j == JM:
            return i == 1 and l <= lmm[1, JM]
        return l <= lmm[i, j]

    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            pe_prev = opress[i, j]
            for l in range(1, lmm[i, j] + 1):
                pe_l = pe_prev + mo[i, j, l] * GRAV
                if m_active(i, j, l):
                    bym = 1.0 / (mo[i, j, l] * dxypo[j])
                    g3d[i, j, l] = g0m[i, j, l] * bym
                    s3d[i, j, l] = s0m[i, j, l] * bym
                    p3d[i, j, l] = 0.5 * (pe_l + pe_prev)
                    vbar[i, j, l] = (vup[i, j, l] + vdn[i, j, l]) * 0.5
                    rho[i, j, l] = 1.0 / vbar[i, j, l]
                pe_prev = pe_l

    # Copy to all longitudes at both poles (J=1 south, J=JM north)
    for j in (1, JM):
        for l in range(1, lmm[1, j] + 1):
            rho[2:IM + 1, j, l] = rho[1, j, l]
            vbar[2:IM + 1, j, l] = vbar[1, j, l]
            g3d[2:IM + 1, j, l] = g3d[1, j, l]
            s3d[2:IM + 1, j, l] = s3d[1, j, l]
            p3d[2:IM + 1, j, l] = p3d[1, j, l]

    return g3d, s3d, p3d, vbar, rho


def densgrad_vertical(lmm, dh, vbar, vup, vdn, vupu, vdnu):
    """The vertical-gradient portion of densgrad (OCNMESO_DRV.f:479-515, this delta's
    numbering): DZV/BYDZV/BYDH/RHOMZ/BYRHOZ. `dh` is ODHORZ0's already-validated DH3D (D40).
    Returns dzv, bydzv, bydh, rhomz, byrhoz (each (IM+1,JM+1,LMO+1))."""
    dzv = np.zeros((IM + 1, JM + 1, LMO + 1))
    bydzv = np.zeros((IM + 1, JM + 1, LMO + 1))
    bydh = np.zeros((IM + 1, JM + 1, LMO + 1))
    rhomz = np.zeros((IM + 1, JM + 1, LMO + 1))
    byrhoz = np.zeros((IM + 1, JM + 1, LMO + 1))

    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            for l in range(1, lmm[i, j] + 1):
                if l > 1:
                    dvbardz = 0.5 * (vup[i, j, l] + vdn[i, j, l] - vupu[i, j, l] - vdnu[i, j, l])
                    dzvlm1 = 0.5 * (dh[i, j, l] + dh[i, j, l - 1])
                    dzv[i, j, l - 1] = dzvlm1
                    bydzv[i, j, l - 1] = 1.0 / dzvlm1
                    rhomz[i, j, l - 1] = max(0.0, -dvbardz * bydzv[i, j, l - 1] / vbar[i, j, l - 1] ** 2)
                    if rhomz[i, j, l - 1] != 0.0:
                        byrhoz[i, j, l - 1] = 1.0 / rhomz[i, j, l - 1]
                bydh[i, j, l] = 1.0 / dh[i, j, l]

    for j in (1, JM):
        for l in range(1, lmm[1, j] + 1):
            dzv[2:IM + 1, j, l] = dzv[1, j, l]
            bydzv[2:IM + 1, j, l] = bydzv[1, j, l]
            bydh[2:IM + 1, j, l] = bydh[1, j, l]
            rhomz[2:IM + 1, j, l] = rhomz[1, j, l]
            byrhoz[2:IM + 1, j, l] = byrhoz[1, j, l]

    return dzv, bydzv, bydh, rhomz, byrhoz


def get_1d_mesodiff(lmm):
    """OCNMESO_DRV.f:1187-1216, live branch (CONSTANT_MESO_DIFFUSIVITY implies USE_1D_MESODIFF,
    D46's finding). k3d(i,j,l) = meso_diffusivity_const (800., the rundeck's fixed value,
    verified from decks/P2SAoM40.R directly -- same precedent as D29's RADIUS/GRAV) at every
    active cell. Returns k3d (IM+1,JM+1,LMO+1)."""
    k3d = np.zeros((IM + 1, JM + 1, LMO + 1))
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            k3d[i, j, 1:lmm[i, j] + 1] = MESO_DIFFUSIVITY_CONST
    return k3d
