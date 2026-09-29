"""Full-fidelity port of ICEDYN.f's GEOMICDYN/ICDYN_MASKS -- Track B, float64 (Stage 1 of the
DYNSI/ocean port, D29). This is the ice-dynamics B-grid's own geometry setup: purely analytic
lat-lon geometry (no dependence on RSI/MSI/SNOWI or any prognostic field) plus a land/ocean-derived
velocity/tracer mask. Both are computed ONCE at model init and held fixed for the whole run, so this
module is called once (not per step) and its outputs (DynsiGeom) are reused every DYNSI call.

For this rundeck (P2SAoM40, non-cubed-sphere, IMICDYN=IM=72, JMICDYN=JM=46), NX1=IMICDYN+2=74,
NY1=JMICDYN=46: 2 ghost columns wrap the 72 real longitudes cyclically (col 1 mirrors the last real
column, col NX1 mirrors the first). RADIUS is a *runtime* planet parameter under USE_PLANET_RAD (not
the hardcoded 6371000 m default) -- the compare/test code infers it exactly from the real dump's DXT
column (DXT = DLON*RADIUS, DLON known exactly from IMICDYN), rather than assuming Earth's radius.
"""
import numpy as np


def geomicdyn(imicdyn, jmicdyn, radius):
    """Port of ICEDYN.f GEOMICDYN. Returns a dict of 1-D and 2-D geometry arrays, all 0-based numpy
    arrays indexed [0..NX1-1] / [0..NY1-1] (Fortran i=1..NX1 -> python index i-1)."""
    nx1 = imicdyn + 2
    ny1 = jmicdyn

    twopi = 2.0 * np.pi
    dlon = twopi / imicdyn
    dlat_dg = 180.0 / (jmicdyn - 1)
    if jmicdyn == 90:
        dlat_dg = 2.0
    if jmicdyn == 180:
        dlat_dg = 1.0
    if jmicdyn == 24:
        dlat_dg = 180.0 / (jmicdyn - 1.5)
    dlat = np.deg2rad(dlat_dg)

    acor, acoru = 1.0, 1.0
    if ny1 in (90, 180):
        acor, acoru = 1.5, 2.0
    if ny1 == 24:
        acor, acoru = 0.75, 0.5

    dyt = np.full(ny1, dlat * radius)
    dyu = np.full(ny1, dlat * radius)
    dyt[ny1 - 1] *= acor
    dyu[ny1 - 1] *= acoru
    dyt[0] *= acor
    dyu[0] *= acoru

    dxt = np.full(nx1, dlon * radius)
    dxu = np.full(nx1, dlon * radius)
    # ghost columns: col0(=i=1) mirrors col(nx1-2)(=i=NX1-1); col(nx1-1)(=i=NX1) mirrors col1(=i=2)
    dxt[0] = dxt[nx1 - 2]
    dxt[nx1 - 1] = dxt[1]
    dxu[0] = dxu[nx1 - 2]
    dxu[nx1 - 1] = dxu[1]

    bydx2 = 0.5 / (dxu * dxu)
    bydxr = 0.5 / (dxu * radius)
    bydy2 = 0.5 / (dyu * dyu)
    bydyr = 0.5 / (dyu * radius)

    bydxdy = np.zeros((nx1, ny1))
    for j in range(ny1):
        bydxdy[:, j] = 0.5 / (dxu * dyu[j])

    fjeq = 0.5 * (ny1 + 1)  # 1-based equatorial index
    cst = np.zeros(ny1)
    tngt = np.zeros(ny1)
    cst[ny1 - 1] = np.cos(np.deg2rad(90.0))
    cst[0] = np.cos(np.deg2rad(-90.0))
    for j1 in range(2, ny1):  # Fortran j=2..ny1-1 -> python index j1 (1-based) in [2,ny1-1]
        phit = (j1 - fjeq) * dlat
        cst[j1 - 1] = np.cos(phit)
        tngt[j1 - 1] = np.sin(phit) / cst[j1 - 1]

    csu = np.zeros(ny1)
    bycsu = np.zeros(ny1)
    tng = np.zeros(ny1)
    for j1 in range(1, ny1 + 1):
        phiu = (j1 - fjeq + 0.5) * dlat
        csu[j1 - 1] = np.cos(phiu)
        bycsu[j1 - 1] = 1.0 / csu[j1 - 1]
        tng[j1 - 1] = np.sin(phiu) / csu[j1 - 1]

    sinen = np.zeros((nx1, ny1))
    for j1 in range(1, ny1 + 1):
        phiu = (j1 - fjeq + 0.5) * dlat
        sinen[:, j1 - 1] = np.sin(phiu)

    # polar fixups
    tngt[ny1 - 1] = tngt[ny1 - 2]
    tng[ny1 - 1] = tng[ny1 - 2]
    csu[ny1 - 1] = csu[ny1 - 2]
    bycsu[ny1 - 1] = 1.0 / csu[ny1 - 1]
    tngt[0] = tngt[1]
    tng[0] = tng[1]
    csu[0] = csu[1]
    bycsu[0] = 1.0 / csu[0]

    return dict(nx1=nx1, ny1=ny1, dlon=dlon, dlat=dlat, dxt=dxt, dxu=dxu, dyt=dyt, dyu=dyu,
                bydx2=bydx2, bydxr=bydxr, bydy2=bydy2, bydyr=bydyr, bydxdy=bydxdy,
                cst=cst, csu=csu, tngt=tngt, tng=tng, bycsu=bycsu, sinen=sinen)


def icdyn_masks(focean, nx1, ny1):
    """Port of ICEDYN.f ICDYN_MASKS. `focean` is the ice-grid land/ocean fraction, shape
    (imicdyn=nx1-2, ny1), values in [0,1]. Returns (heffm, uvm), both shape (nx1, ny1).
    Non-cubed-sphere branch: heffm(i,j) = ceiling(focean(i-1,j)) for real columns i=2..nx1-1
    (0-based: heffm[:, j] for cols 1..nx1-2 = ceil(focean[0:nx1-2, j])), then two masking passes."""
    imicdyn = nx1 - 2
    heffm = np.zeros((nx1, ny1))
    heffm[1:nx1 - 1, :] = np.ceil(focean)
    heffm[0, :] = heffm[nx1 - 2, :]
    heffm[nx1 - 1, :] = heffm[1, :]

    # uvm(i,j) for i=1..nx1-1 (0-based 0..nx1-2), j=1..ny1-1 (0-based 0..ny1-2):
    # min(heffm(i,j),heffm(i+1,j),heffm(i,j+1),heffm(i+1,j+1)) -- needs heffm at j+1 (HALO from north)
    # here we have the full column range already (no real MPI halo needed, single-domain case).
    uvm = np.zeros((nx1, ny1))
    for j in range(ny1 - 1):
        uvm[0:nx1 - 1, j] = np.rint(np.minimum.reduce([
            heffm[0:nx1 - 1, j], heffm[1:nx1, j], heffm[0:nx1 - 1, j + 1], heffm[1:nx1, j + 1]]))

    # reset tracer points to surround velocity points (except poles): j=1..ny1-2 (0-based),
    # i=2..nx1-1 (0-based 1..nx1-2), using uvm at j-1 (already computed above).
    for j in range(1, ny1 - 1):
        for i in range(1, nx1 - 1):
            k = np.rint(max(uvm[i, j], uvm[i - 1, j], uvm[i, j - 1], uvm[i - 1, j - 1]))
            heffm[i, j] = k

    heffm[0, :] = heffm[nx1 - 2, :]
    heffm[nx1 - 1, :] = heffm[1, :]

    uvm = np.zeros((nx1, ny1))
    for j in range(ny1 - 1):
        uvm[0:nx1 - 1, j] = np.rint(np.minimum.reduce([
            heffm[0:nx1 - 1, j], heffm[1:nx1, j], heffm[0:nx1 - 1, j + 1], heffm[1:nx1, j + 1]]))

    # cyclic east-west boundary on uvm
    uvm[0, :] = uvm[nx1 - 2, :]
    uvm[nx1 - 1, :] = uvm[1, :]

    return heffm, uvm
