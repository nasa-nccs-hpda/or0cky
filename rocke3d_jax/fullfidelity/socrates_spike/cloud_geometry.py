"""Cloud geometry of radiance_calc for this rundeck: set_cloud_geometry (n_cloud_top), aggregate_cloud (CSIW -> stratiform / convective
regions) and overlap_coupled (i_cloud = ip_cloud_triple: corr_factor = 1, three regions clear / stratiform / convective).
Layout: layer index 0-based in the layer axis of w_cloud (n_profile, n_layer); cloud_overlap (n_profile, n_layer+1, 18) with the
second axis = Fortran layer 0..n_layer (id_ct - 1 = 0), third axis = coefficient 1..18 (v11..v33, u11..u33).
Fortran: radiance_core/set_cloud_geometry.F90, aggregate_cloud.F90, overlap_coupled.F90."""
import numpy as np

TOL = 1.0e2 * np.finfo(np.float64).eps
IP_CLOUD_TYPE_SW, IP_CLOUD_TYPE_SI, IP_CLOUD_TYPE_CW, IP_CLOUD_TYPE_CI = 1, 2, 3, 4
REGION_FOR_TYPE = {1: 2, 2: 2, 3: 3, 4: 3}      # stratiform (2) for sw, si; convective (3) for cw, ci  (i_region_cloud, dumped tag 45/3)


def set_cloud_top(w_cloud, l_global_cloud_top=False):
    """n_cloud_top (1-based) of a call: first layer with cloud in any profile, else n_layer (the loop stops at n_layer)."""
    n_layer = w_cloud.shape[1]
    nct = 1
    if l_global_cloud_top:
        return nct
    has = np.any(w_cloud > 0.0, axis=0)
    while nct < n_layer and not has[nct - 1]:
        nct += 1
    return nct


def aggregate_csiw(frac_cloud):
    """frac_cloud (n_profile, n_layer, 4) -> frac_region (n_profile, n_layer, 3) (clear entry unused, set 0)."""
    fr = np.zeros(frac_cloud.shape[:2] + (3,))
    fr[:, :, 1] = frac_cloud[:, :, IP_CLOUD_TYPE_SW - 1] + frac_cloud[:, :, IP_CLOUD_TYPE_SI - 1]
    fr[:, :, 2] = frac_cloud[:, :, IP_CLOUD_TYPE_CW - 1] + frac_cloud[:, :, IP_CLOUD_TYPE_CI - 1]
    return fr


def overlap_coupled_triple(w_cloud, frac_region, n_cloud_top):
    """Returns cloud_overlap (n_profile, n_layer+1, 18), w_free (n_profile, n_layer) (valid for layers >= n_cloud_top),
    tot_cloud_cover (n_profile,)."""
    n_prof, n_layer = w_cloud.shape
    nr = 3
    co = np.zeros((n_prof, n_layer + 1, 18))
    w_free = 1.0 - w_cloud
    w_free[:, :n_cloud_top - 1] = np.nan                       # not defined by the Fortran
    wf = lambda i: w_free[:, i - 1]                            # Fortran layer i
    wc = lambda i: w_cloud[:, i - 1]
    tcc = np.where(wf(n_cloud_top) > TOL, wf(n_cloud_top), 0.0)
    area_upper = np.zeros((n_prof, nr)); area_upper[:, 0] = 1.0
    for i in range(n_cloud_top - 1, n_layer + 1):
        corr = np.ones((n_prof, nr))
        area_lower = np.zeros((n_prof, nr))
        if i < n_layer:
            area_lower[:, 0] = wf(i + 1)
            for k in (1, 2):
                area_lower[:, k] = wc(i + 1) * frac_region[:, i, k]
        else:
            area_lower[:, 0] = 1.0
        ao = np.zeros((n_prof, nr, nr))
        for k in range(nr):
            ao[:, k, k] = corr[:, k] * np.minimum(area_lower[:, k], area_upper[:, k])
        if n_cloud_top <= i < n_layer:
            den = 1.0 - ao[:, 1, 1] - ao[:, 2, 2]
            ok = wf(i + 1) > TOL
            tcc = np.where(ok, tcc * wf(i + 1) / np.where(ok, den, 1.0), 0.0)
        area_random_tot = 1.0 - ao[:, 0, 0]
        for k in (1, 2):
            area_random_tot = area_random_tot - ao[:, k, k]
        ar_up = area_upper - np.einsum("lkk->lk", ao)
        ar_lo = area_lower - np.einsum("lkk->lk", ao)
        ok = area_random_tot > TOL
        for k in range(nr):
            for j in range(nr):
                add = ar_up[:, k] * ar_lo[:, j] / np.where(ok, area_random_tot, 1.0)
                ao[:, k, j] = np.where(ok, ao[:, k, j] + add, ao[:, k, j])
        # cloud_overlap(l, i, nr*(j-1)+k): v-part (1..9), then u-part nr*(nr+j-1)+k (10..18); index (1-based) -> column idx-1
        for k in range(nr):
            co[:, i, nr * k + k] = 1.0
            co[:, i, nr * (nr + k) + k] = 1.0
        for k in range(nr):
            ok_u = area_upper[:, k] > TOL
            ok_l = area_lower[:, k] > TOL
            for j in range(nr):
                co[:, i, nr * j + k] = np.where(ok_u, ao[:, k, j] / np.where(ok_u, area_upper[:, k], 1.0), co[:, i, nr * j + k])
                co[:, i, nr * (nr + j) + k] = np.where(ok_l, ao[:, j, k] / np.where(ok_l, area_lower[:, k], 1.0),
                                                       co[:, i, nr * (nr + j) + k])
        # the Fortran zeroes the off-diagonals of the initial identity first, then overwrites where area > tol: identical to the above
        # because only diagonal entries are preset to 1 (off-diagonals start at 0 here)
        if i < n_layer:
            area_upper = area_lower.copy()
    return co, w_free, 1.0 - tcc
