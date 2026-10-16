"""Batched OCNMESO inputs (Stage 2, D82): numpy ports of ocnmeso_ff.ocnstate_derived,
densgrad_vertical and get_1d_mesodiff with the (i, j) loops replaced by masked array operations.

ocnstate_derived keeps a sequential loop over the 13 layers: the interface pressure is accumulated
layer by layer (pe_l = pe_prev + mo*GRAV) in the scalar port's order, so the rounding is unchanged.
densgrad_vertical and get_1d_mesodiff have no cross-layer recurrence and are fully vectorized.
Arrays are 1-based (IM+1, JM+1, LMO+1), as in ocnmeso_ff. Validated by ocnmeso_vec_compare.py.
"""
import numpy as np

from odhorz_ff import geomo_dyn_arrays
from ocnmeso_ff import IM, JM, LMO, GRAV, MESO_DIFFUSIVITY_CONST


def _active(lmm):
    """act[i, j, l] = l <= lmm[i, j] for 1-based i, j, l (index 0 entries False)."""
    lidx = np.arange(LMO + 1)[None, None, :]
    act = lmm[:, :, None] >= lidx
    act[:, :, 0] = False
    act[0, :, :] = False
    act[:, 0, :] = False
    return act


def _pole_copy(a, lmm):
    """Copy longitude 1 to 2..IM at both poles (J = 1 and J = JM), for l <= lmm[1, j]."""
    for j in (1, JM):
        n = lmm[1, j]
        if n >= 1:
            a[2:IM + 1, j, 1:n + 1] = a[1, j, 1:n + 1][None, :]


def ocnstate_derived_vec(mo, g0m, gzm, s0m, szm, opress, lmm, vup, vdn):
    dxypo = geomo_dyn_arrays()[-1]
    shp = (IM + 1, JM + 1, LMO + 1)
    g3d, s3d, p3d, vbar, rho = [np.zeros(shp) for _ in range(5)]
    act = _active(lmm)
    # the North Pole row is driven by nbyzm: only I = 1 is active there (see ocnmeso_ff)
    act[2:, JM, :] = False
    pe_prev = opress.copy()
    with np.errstate(all="ignore"):
        for l in range(1, LMO + 1):
            within = lmm >= l                                  # pe_l is accumulated for l <= lmm
            pe_l = np.where(within, pe_prev + mo[:, :, l] * GRAV, pe_prev)
            a = act[:, :, l]
            bym = 1.0 / (mo[:, :, l] * dxypo[None, :])
            vb = (vup[:, :, l] + vdn[:, :, l]) * 0.5
            g3d[:, :, l] = np.where(a, g0m[:, :, l] * bym, 0.0)
            s3d[:, :, l] = np.where(a, s0m[:, :, l] * bym, 0.0)
            p3d[:, :, l] = np.where(a, 0.5 * (pe_l + pe_prev), 0.0)
            vbar[:, :, l] = np.where(a, vb, 0.0)
            rho[:, :, l] = np.where(a, 1.0 / vb, 0.0)
            pe_prev = pe_l
    for arr in (rho, vbar, g3d, s3d, p3d):
        _pole_copy(arr, lmm)
    return g3d, s3d, p3d, vbar, rho


def densgrad_vertical_vec(lmm, dh, vbar, vup, vdn, vupu, vdnu):
    shp = (IM + 1, JM + 1, LMO + 1)
    dzv, bydzv, bydh, rhomz, byrhoz = [np.zeros(shp) for _ in range(5)]
    act = _active(lmm)
    with np.errstate(all="ignore"):
        bydh[:] = np.where(act, 1.0 / dh, 0.0)
        # interface quantities at l-1 for every active l > 1
        up = act[:, :, 2:]                                     # cells with layer l = 2..LMO active
        dvbardz = 0.5 * (vup[:, :, 2:] + vdn[:, :, 2:] - vupu[:, :, 2:] - vdnu[:, :, 2:])
        dzvlm1 = 0.5 * (dh[:, :, 2:] + dh[:, :, 1:-1])
        by = 1.0 / dzvlm1
        rz = np.maximum(0.0, -dvbardz * by / vbar[:, :, 1:-1] ** 2)
        brz = np.where(rz != 0.0, 1.0 / rz, 0.0)
        dzv[:, :, 1:-1] = np.where(up, dzvlm1, 0.0)
        bydzv[:, :, 1:-1] = np.where(up, by, 0.0)
        rhomz[:, :, 1:-1] = np.where(up, rz, 0.0)
        byrhoz[:, :, 1:-1] = np.where(up, brz, 0.0)
    for arr in (dzv, bydzv, bydh, rhomz, byrhoz):
        _pole_copy(arr, lmm)
    return dzv, bydzv, bydh, rhomz, byrhoz


def get_1d_mesodiff_vec(lmm):
    return np.where(_active(lmm), MESO_DIFFUSIVITY_CONST, 0.0)
