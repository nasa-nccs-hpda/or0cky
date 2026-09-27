"""Tile aggregation (FLUXES.f `avg_patches_pbl_exports`/`avg_patches_srfflx_exports`) -- Track B.

The composite (grid-cell-mean) surface flux/state fields that ATURB and PBL actually read
(atmsrf%uflux1, vflux1, dth1, dq1, tsavg, qsavg) are a simple area-fraction-weighted sum over the
4 surface-type patches (ocean, sea ice, land ice, land): `sum_k patch_k * ftype_k`. No special-casing:
FLUXES.f's `avg_patches_*` is exactly this loop (`avg%field(i,j) += patches(k)%field(i,j)*ftype(i,j,k)`).
"""
import numpy as np
import jax.numpy as jnp

NREC = 40


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


FIELDS = ("uflux1", "vflux1", "dth1", "dq1", "tsavg", "qsavg")


def unpack(rec):
    """rec: (N,40). Returns dict: ftype (N,4), {field: (N,4)} per patch, {field: (N,)} composite ref."""
    ftype = rec[:, 2:30:7]
    patch = {}
    for j, f in enumerate(FIELDS):
        patch[f] = rec[:, 3 + j:30:7]
    composite_ref = {f: rec[:, 30 + j] for j, f in enumerate(FIELDS)}
    return ftype, patch, composite_ref


def aggregate(ftype, patch):
    """ftype: (...,4), patch[field]: (...,4). Returns {field: (...)}."""
    return {f: jnp.sum(patch[f] * ftype, axis=-1) for f in FIELDS}
