"""Full-fidelity port of ICEDYN.f's FORM/PLAST/RELAX/VPICEDYN -- Track B, float64 (Stage 1 of the
DYNSI/ocean port, D29). This is the real ice-dynamics numerical core: a 2-step ADI (Alternating
Direction Implicit) viscous-plastic solve, iterated to convergence via an outer pseudo-timestep loop
(VPICEDYN). Ported plain-Python first (this module); a batched/JAX version with jax.lax.while_loop
for the outer loop is the next increment.

Index convention: every whole-grid field is a numpy array of shape (NX1+1, NY1+1), where Fortran
index (I,J), I=1..NX1, J=1..NY1, maps DIRECTLY to python index [I,J] (row/col 0 unused/padding).
This mirrors the Fortran source 1:1 and avoids off-by-one translation errors in the dense stencils
below -- verified by exact validation against real Fortran ffy_<itime>_{in,out}.bin dumps (D29).

Not yet ported here (recorded/external inputs to this stage): the atm-grid<->ice-grid regrid of wind
stress (GAIRX/GAIRY, from DMUA/DMVA) and ocean current/pressure-gradient (GWATX/GWATY/PGFUB/PGFVB,
from UOSURF/VOSURF/OGEOZA) done in DYNSI's own body before VPICEDYN -- these depend on fields the
ocean model (not yet ported) supplies, so per this project's established pattern they are taken as
recorded inputs (dumped post-regrid) rather than re-derived. HEFF/AREA/AMASS/COR (also computed in
DYNSI's own body, from iRSI/iMSI which ARE already-ported sea-ice fields) are recorded as inputs too,
for the same reason it's cleaner to validate VPICEDYN as one unit against its real Fortran boundary.
"""
import numpy as np

RADIUS = 6371000.0  # inferred exactly from ffz_geom.bin's DXT column (D29); this rundeck's runtime
                     # planet radius (USE_PLANET_RAD) happens to equal Earth's, per icedyn_geom_ff.
BYRAD2 = 1.0 / (RADIUS * RADIUS)
PSTAR = 27500.0      # ICEDYN.f PARAMETER, ice strength coefficient (Hibler 1979)
ECCEN = 2.0          # ICEDYN.f PARAMETER, ellipse eccentricity for the plastic yield curve
NPOL = 1.0           # ICEDYN.f PARAMETER (FLOAT(NPOL-0) pole-boundary multiplier in RELAX)

# Grid geometry constants: like ICEDYN.f, these are module-level ("host-associated" via the ICEDYN
# module in the real source) rather than passed as arguments to every stencil function -- set once
# per grid via init_geometry(). All are 1-padded arrays (index 0 unused) to mirror Fortran 1-based
# indexing directly; see icedyn_geom_ff.geomicdyn for how they are computed and validated (D29).
DXT = DXU = DYT = DYU = None
BYDX2 = BYDXR = BYDY2 = BYDYR = BYDXDY = None
CST = CSU = TNGT = TNG = BYCSU = SINEN = None
HEFFM = UVM = None


def init_geometry(geom, heffm, uvm):
    """Install the (NX1,NY1)-shaped 0-based geometry from icedyn_geom_ff into this module's 1-padded
    (NX1+1,NY1+1) globals, once per grid (matching how ICEDYN.f's module arrays are set by GEOMICDYN/
    ICDYN_MASKS once at init and then simply host-associated by FORM/PLAST/RELAX)."""
    global DXT, DXU, DYT, DYU, BYDX2, BYDXR, BYDY2, BYDYR, BYDXDY
    global CST, CSU, TNGT, TNG, BYCSU, SINEN, HEFFM, UVM
    nx1, ny1 = geom["nx1"], geom["ny1"]

    def pad1d(a):
        p = np.zeros(len(a) + 1)
        p[1:] = a
        return p

    def pad2d(a):
        p = np.zeros((nx1 + 1, ny1 + 1))
        p[1:, 1:] = a
        return p

    DXT, DXU = pad1d(geom["dxt"]), pad1d(geom["dxu"])
    DYT, DYU = pad1d(geom["dyt"]), pad1d(geom["dyu"])
    BYDX2, BYDXR = pad1d(geom["bydx2"]), pad1d(geom["bydxr"])
    BYDY2, BYDYR = pad1d(geom["bydy2"]), pad1d(geom["bydyr"])
    BYDXDY = pad2d(geom["bydxdy"])
    CST, CSU = pad1d(geom["cst"]), pad1d(geom["csu"])
    TNGT, TNG, BYCSU = pad1d(geom["tngt"]), pad1d(geom["tng"]), pad1d(geom["bycsu"])
    SINEN = pad2d(geom["sinen"])
    HEFFM, UVM = pad2d(heffm), pad2d(uvm)


def tridiag_thomas(a, b, c, r):
    """Plain (non-cyclic) Thomas algorithm, TRIDIAG_MOD's TRIDIAG/TRIDIAG_new. a,b,c,r are 1-D arrays
    over the SAME index range (already sliced to [j_lower..j_upper]); a[0] and c[-1] are unused."""
    n = len(b)
    gam = np.zeros(n)
    u = np.zeros(n)
    bet = b[0]
    u[0] = r[0] / bet
    for j in range(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def tridiag_cyclic(a, b, c, r):
    """TRIDIAG_MOD's TRIDIAG_cyclic: Thomas + Sherman-Morrison for a cyclic system with nonzero
    corner terms a[0] (=A_1, coupling to u_N) and c[-1] (=C_N, coupling to u_1). 1-D arrays length N."""
    n = len(b)
    gam = np.zeros(n)
    q = np.zeros(n)
    u = np.zeros(n)
    a1, b1, c1, r1 = a[0], b[0], c[0], r[0]
    if b1 == 1.0:
        a1, b1, c1, r1 = a1 * 2.0, b1 * 2.0, c1 * 2.0, r1 * 2.0
    bet = b1 - 1.0
    u[0] = r1 / bet
    q[0] = 1.0 / bet
    gam[1] = c1 / bet
    for j in range(1, n - 1):
        bet = b[j] - a[j] * gam[j]
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
        q[j] = -a[j] * q[j - 1] / bet
        gam[j + 1] = c[j] / bet
    j = n - 1
    bet = b[j] - a[j] * gam[j] - a1 * c[n - 1]
    u[j] = (r[j] - a[j] * u[j - 1]) / bet
    q[j] = (c[n - 1] - a[j] * q[j - 1]) / bet
    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
        q[j] = q[j] - gam[j + 1] * q[j + 1]
    bet = 1.0 + q[0] + a1 * q[n - 1]
    qcoeff = (u[0] + a1 * u[n - 1]) / bet
    for j in range(n):
        u[j] = u[j] - qcoeff * q[j]
    return u


def _pad(nx1, ny1):
    return np.zeros((nx1 + 1, ny1 + 1))


def plast(nx1, ny1, uice1, vice1, press):
    """ICEDYN.f PLAST. Returns (eta, zeta), shape (NX1+1, NY1+1). uice1/vice1 are UICE(:,:,1)/VICE(:,:,1)
    (the *predictor* velocity going into this FORM/PLAST call), press is the just-computed ice pressure."""
    nypole = ny1 - 1
    ecm2 = 1.0 / (ECCEN ** 2)
    gmin = 1e-20
    e11 = _pad(nx1, ny1)
    e22 = _pad(nx1, ny1)
    e12 = _pad(nx1, ny1)
    zeta = _pad(nx1, ny1)
    zmax = (5e12 / 2e4) * press
    zmin = np.full_like(press, 4e8)

    for j in range(2, ny1):  # J_0S..J_1S = 2..NYPOLE
        for i in range(2, nx1):  # I=2..NX1-1
            e11[i, j] = 0.5 / (DXT[i] * CST[j]) * (uice1[i, j] + uice1[i, j - 1]
                        - uice1[i - 1, j] - uice1[i - 1, j - 1]) - 0.25 * (vice1[i, j]
                        + vice1[i - 1, j] + vice1[i - 1, j - 1] + vice1[i, j - 1]) * TNGT[j] / RADIUS
            e22[i, j] = 0.5 / DYT[j] * (vice1[i, j] + vice1[i - 1, j] - vice1[i, j - 1] - vice1[i - 1, j - 1])
            e12[i, j] = 0.5 * (0.5 / DYT[j] * (uice1[i, j] + uice1[i - 1, j] - uice1[i, j - 1]
                        - uice1[i - 1, j - 1]) + 0.5 / (DXT[i] * CST[j]) * (vice1[i, j] + vice1[i, j - 1]
                        - vice1[i - 1, j] - vice1[i - 1, j - 1]) + 0.25 * (uice1[i, j] + uice1[i - 1, j]
                        + uice1[i - 1, j - 1] + uice1[i, j - 1]) * TNGT[j] / RADIUS)
            delt = (e11[i, j] ** 2 + e22[i, j] ** 2) * (1.0 + ecm2) + 4.0 * ecm2 * e12[i, j] ** 2 \
                + 2.0 * e11[i, j] * e22[i, j] * (1.0 - ecm2)
            delt1 = max(gmin, np.sqrt(delt))
            zeta[i, j] = 0.5 * press[i, j] / delt1

    for j in range(2, ny1):
        for i in range(2, nx1):
            zeta[i, j] = min(zmax[i, j], zeta[i, j])
            zeta[i, j] = max(zmin[i, j], zeta[i, j])

    aaa = sum(zeta[i, nypole] for i in range(2, nx1)) / (nx1 - 2)
    zeta[1:nx1 + 1, ny1] = aaa
    aaa = sum(zeta[i, 2] for i in range(2, nx1)) / (nx1 - 2)
    zeta[1:nx1 + 1, 1] = aaa

    zeta[1, :] = zeta[nx1 - 1, :]
    zeta[nx1, :] = zeta[2, :]

    eta = ecm2 * zeta
    return eta, zeta


def form(nx1, ny1, uice1, vice1, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, dragsym_consts,
         osurf_tilt, pgfub, pgfvb):
    """ICEDYN.f FORM. dragsym_consts = (sinwat, coswat, dts). Returns dict with dwatn, dragS, dragA,
    forcex, forcey, press, eta, zeta, all shape (NX1+1, NY1+1)."""
    sinwat, coswat = dragsym_consts
    nypole = ny1 - 1

    dwatn = _pad(nx1, ny1)
    for j in range(1, ny1 + 1):
        for i in range(1, nx1):
            dwatn[i, j] = 5.5 * np.sqrt((uice1[i, j] - gwatx[i, j]) ** 2 + (vice1[i, j] - gwaty[i, j]) ** 2)

    drags = _pad(nx1, ny1)
    draga = _pad(nx1, ny1)
    forcex = _pad(nx1, ny1)
    forcey = _pad(nx1, ny1)
    half = ny1 // 2
    for j in range(1, nypole + 1):
        for i in range(1, nx1):
            drags[i, j] = dwatn[i, j] * coswat
            if j > half:
                draga[i, j] = dwatn[i, j] * sinwat + cor[i, j]
            else:
                draga[i, j] = -dwatn[i, j] * sinwat + cor[i, j]

            forcex[i, j] = gairx[i, j]
            forcey[i, j] = gairy[i, j]
            if j > half:
                forcex[i, j] += dwatn[i, j] * (coswat * gwatx[i, j] - sinwat * gwaty[i, j])
                forcey[i, j] += dwatn[i, j] * (sinwat * gwatx[i, j] + coswat * gwaty[i, j])
            else:
                forcex[i, j] += dwatn[i, j] * (coswat * gwatx[i, j] + sinwat * gwaty[i, j])
                forcey[i, j] += dwatn[i, j] * (-sinwat * gwatx[i, j] + coswat * gwaty[i, j])

            if osurf_tilt == 1:
                forcex[i, j] += amass[i, j] * pgfub[i, j]
                forcey[i, j] += amass[i, j] * pgfvb[i, j]
            else:
                forcex[i, j] -= cor[i, j] * gwaty[i, j]
                forcey[i, j] += cor[i, j] * gwatx[i, j]

    press = _pad(nx1, ny1)
    for j in range(1, ny1 + 1):
        for i in range(1, nx1 + 1):
            press[i, j] = PSTAR * heff[i, j] * np.exp(-20.0 * (1.0 - area[i, j]))

    eta, zeta = plast(nx1, ny1, uice1, vice1, press)

    aaa = sum(press[i, nypole] for i in range(2, nx1)) / (nx1 - 2)
    press[1:nx1 + 1, ny1] = aaa

    press[1, :] = press[nx1 - 1, :]
    press[nx1, :] = press[2, :]

    press *= HEFFM
    eta *= HEFFM
    zeta *= HEFFM

    for j in range(1, nypole + 1):
        for i in range(1, nx1):
            forcex[i, j] -= (0.25 / (DXU[i] * CSU[j])) * (press[i + 1, j] + press[i + 1, j + 1]
                             - press[i, j] - press[i, j + 1])
            forcey[i, j] -= 0.25 / DYU[j] * (press[i, j + 1] + press[i + 1, j + 1]
                            - press[i, j] - press[i + 1, j])

    forcex[1, :] = forcex[nx1 - 1, :]
    forcey[1, :] = forcey[nx1 - 1, :]
    forcex[nx1, :] = forcex[2, :]
    forcey[nx1, :] = forcey[2, :]
    dwatn[1, :] = dwatn[nx1 - 1, :]
    dwatn[nx1, :] = dwatn[2, :]

    return dict(dwatn=dwatn, drags=drags, draga=draga, forcex=forcex, forcey=forcey,
                press=press, eta=eta, zeta=zeta)


def relax(nx1, ny1, uice, vice, uicec, vicec, forcex, forcey, draga, drags, eta, zeta, amass, cor,
          bydts, _debug_stage1_only=False):
    """ICEDYN.f RELAX: the 2-step ADI viscous-plastic solve. `uice`/`vice` are dicts {1:.., 2:.., 3:..}
    holding UICE(:,:,{1,2,3})/VICE(:,:,{1,2,3}); mutated and also returned. `uicec`/`vicec` are mutated
    in place (boundary/pole fills), matching the real subroutine's side effects on its host-associated
    module arrays. Single-domain (no MPI halo needed: this rundeck's ice-dyn grid is one process)."""
    nxlcyc = nx1 - 1
    nypole = ny1 - 1
    half = nx1 // 2  # NX1/2 (integer)
    pad = (nx1 - 2) // 2  # (NX1-2)/2

    for j in range(1, nypole + 1):
        for i in range(1, nx1 + 1):
            forcex[i, j] *= UVM[i, j]
            forcey[i, j] *= UVM[i, j]

    for j in range(1, nypole + 1):
        for i in range(1, nx1 + 1):
            uice[2][i, j] = uice[1][i, j]
            vice[2][i, j] = vice[1][i, j]
            uice[1][i, j] = uice[3][i, j] * UVM[i, j]
            vice[1][i, j] = vice[3][i, j] * UVM[i, j]

    # north pole reflection (this grid always has a north pole row at j=NY1, single global domain)
    for i in range(1, half + 1):
        src = i + pad
        uice[1][i, ny1] = -uicec[src, nypole]
        vice[1][i, ny1] = -vicec[src, nypole]
        uice[3][i, ny1] = -uicec[src, nypole]
        vice[3][i, ny1] = -vicec[src, nypole]
        uicec[i, ny1] = -uicec[src, nypole]
        vicec[i, ny1] = -vicec[src, nypole]
    for i in range(half + 1, nx1):
        src = i - pad
        uice[1][i, ny1] = -uicec[src, nypole]
        vice[1][i, ny1] = -vicec[src, nypole]
        uice[3][i, ny1] = -uicec[src, nypole]
        vice[3][i, ny1] = -vicec[src, nypole]
        uicec[i, ny1] = -uicec[src, nypole]
        vicec[i, ny1] = -vicec[src, nypole]

    for j in range(1, nypole + 1):
        uice[1][1, j] = uicec[nx1 - 1, j]
        vice[1][1, j] = vicec[nx1 - 1, j]
        uice[1][nx1, j] = uicec[2, j]
        vice[1][nx1, j] = vicec[2, j]
        uice[3][1, j] = uicec[nx1 - 1, j]
        vice[3][1, j] = vicec[nx1 - 1, j]
        uice[3][nx1, j] = uicec[2, j]
        vice[3][nx1, j] = vicec[2, j]
        uicec[1, j] = uicec[nx1 - 1, j]
        vicec[1, j] = vicec[nx1 - 1, j]
        uicec[nx1, j] = uicec[2, j]
        vicec[nx1, j] = vicec[2, j]

    # ---- FIRST DO UICE: the first half (I-direction, cyclic) ----
    fxy = _pad(nx1, ny1)
    for j in range(2, nypole + 1):
        for i in range(2, nxlcyc + 1):
            delxy = BYDXDY[i, j]
            delxr = BYDXR[i]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            fxy[i, j] = draga[i, j] * vicec[i, j] + forcex[i, j] \
                + 0.5 * (zeta[i + 1, j + 1] * (vicec[i + 1, j + 1] + vicec[i, j + 1] - vicec[i + 1, j]
                         - vicec[i, j]) + zeta[i + 1, j] * (vicec[i + 1, j] + vicec[i, j]
                         - vicec[i + 1, j - 1] - vicec[i, j - 1]) + zeta[i, j + 1] * (vicec[i, j]
                         + vicec[i - 1, j] - vicec[i, j + 1] - vicec[i - 1, j + 1]) + zeta[i, j] *
                         (vicec[i, j - 1] + vicec[i - 1, j - 1] - vicec[i, j] - vicec[i - 1, j])) \
                * delxy * BYCSU[j] \
                - 0.5 * (eta[i + 1, j + 1] * (vicec[i + 1, j + 1] + vicec[i, j + 1] - vicec[i + 1, j]
                         - vicec[i, j]) + eta[i + 1, j] * (vicec[i + 1, j] + vicec[i, j]
                         - vicec[i + 1, j - 1] - vicec[i, j - 1]) + eta[i, j + 1] * (vicec[i, j]
                         + vicec[i - 1, j] - vicec[i, j + 1] - vicec[i - 1, j + 1]) + eta[i, j] *
                         (vicec[i, j - 1] + vicec[i - 1, j - 1] - vicec[i, j] - vicec[i - 1, j])) \
                * delxy * BYCSU[j] \
                + 0.5 * (vicec[i + 1, j] - vicec[i - 1, j]) * (eta[i, j + 1] + eta[i + 1, j + 1]
                         - eta[i, j] - eta[i + 1, j]) * delxy * BYCSU[j] \
                + 0.5 * e_mean * ((vicec[i + 1, j + 1] - vicec[i - 1, j + 1]) * BYCSU[j + 1]
                         - (vicec[i + 1, j - 1] - vicec[i - 1, j - 1]) * BYCSU[j - 1]) * delxy \
                - ((zeta[i + 1, j + 1] + zeta[i + 1, j] - zeta[i, j] - zeta[i, j + 1])
                   + (eta[i + 1, j + 1] + eta[i + 1, j] - eta[i, j] - eta[i, j + 1])) \
                * TNG[j] * vicec[i, j] * delxr * BYCSU[j] \
                - (e_mean + 0.25 * (zeta[i, j + 1] + zeta[i + 1, j + 1] + zeta[i, j] + zeta[i + 1, j])) \
                * TNG[j] * (vicec[i + 1, j] - vicec[i - 1, j]) * delxr * BYCSU[j] \
                - e_mean * 2.0 * TNG[j] * (vicec[i + 1, j] - vicec[i - 1, j]) * delxr * BYCSU[j]

    au = _pad(nx1, ny1)
    bu = _pad(nx1, ny1)
    cu = _pad(nx1, ny1)
    for j in range(2, nypole + 1):
        for i in range(2, nxlcyc + 1):
            delx2 = BYDX2[i]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            aa1 = ((eta[i + 1, j] + zeta[i + 1, j]) * BYCSU[j]
                   + (eta[i + 1, j + 1] + zeta[i + 1, j + 1]) * BYCSU[j]) * BYCSU[j]
            aa2 = ((eta[i, j] + zeta[i, j]) * BYCSU[j]
                   + (eta[i, j + 1] + zeta[i, j + 1]) * BYCSU[j]) * BYCSU[j]
            aa6 = 2.0 * e_mean * TNG[j] * TNG[j]
            au[i, j] = -aa2 * delx2 * UVM[i, j]
            bu[i, j] = ((aa1 + aa2) * delx2 + aa6 * BYRAD2 + amass[i, j] * bydts * 2.0
                        + drags[i, j]) * UVM[i, j] + (1.0 - UVM[i, j])
            cu[i, j] = -aa1 * delx2 * UVM[i, j]

    for j in range(2, nypole + 1):
        urt = np.zeros(nx1 + 1)
        for i in range(2, nxlcyc + 1):
            delyr = BYDYR[j]
            aa3 = eta[i, j + 1] + eta[i + 1, j + 1]
            aa4 = eta[i, j] + eta[i + 1, j]
            aa5 = -(eta[i, j + 1] + eta[i + 1, j + 1] - eta[i, j] - eta[i + 1, j]) * TNG[j]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            delnn_y2 = BYDY2[j]
            urt[i] = fxy[i, j] - aa5 * delyr * uicec[i, j] - (aa3 + aa4) * delnn_y2 * uicec[i, j] \
                + (eta[i, j + 1] + eta[i + 1, j + 1]) * uicec[i, j + 1] * delnn_y2 \
                + (eta[i, j] + eta[i + 1, j]) * uicec[i, j - 1] * delnn_y2 \
                + e_mean * delyr * (uicec[i, j + 1] * TNG[j + 1] - uicec[i, j - 1] * TNG[j - 1]) \
                - e_mean * delyr * 2.0 * TNG[j] * (uicec[i, j + 1] - uicec[i, j - 1])
            urt[i] = (urt[i] + amass[i, j] * bydts * uice[2][i, j] * 2.0) * UVM[i, j]
        sl = slice(2, nxlcyc + 1)
        sol = tridiag_cyclic(au[sl, j], bu[sl, j], cu[sl, j], urt[sl])
        uice[1][sl, j] = sol
        uice[1][1, j] = uice[1][nx1 - 1, j]
        uice[1][nx1, j] = uice[1][2, j]

    if _debug_stage1_only:
        return uice, vice

    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            uice[3][i, j] = uice[1][i, j]

    # ---- second half (J-direction) ----
    av = _pad(nx1, ny1)
    bv = _pad(nx1, ny1)
    cv = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            dely2 = BYDY2[j]
            delyr = BYDYR[j]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            aa1 = eta[i, j + 1] + eta[i + 1, j + 1]
            aa2 = eta[i, j] + eta[i + 1, j]
            aa5 = -(eta[i, j + 1] + eta[i + 1, j + 1] - eta[i, j] - eta[i + 1, j]) * TNG[j]
            aa6 = 2.0 * e_mean * TNG[j] * TNG[j]
            av[i, j] = (-aa2 * dely2 + e_mean * delyr * (TNG[j - 1] - 2.0 * TNG[j])) * UVM[i, j]
            bv[i, j] = ((aa1 + aa2) * dely2 + aa5 * delyr + aa6 * BYRAD2 + amass[i, j] * bydts * 2.0
                        + drags[i, j]) * UVM[i, j] + (1.0 - UVM[i, j])
            cv[i, j] = (-aa1 * dely2 - e_mean * delyr * (TNG[j + 1] - 2.0 * TNG[j])) * UVM[i, j]
    for i in range(2, nxlcyc + 1):
        av[i, 2] = 0.0
        cv[i, nypole] = 0.0

    fxy1a = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            delx2 = BYDX2[i]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            aa1 = ((eta[i + 1, j] + zeta[i + 1, j]) * BYCSU[j]
                   + (eta[i + 1, j + 1] + zeta[i + 1, j + 1]) * BYCSU[j]) * BYCSU[j]
            aa2 = ((eta[i, j] + zeta[i, j]) * BYCSU[j]
                   + (eta[i, j + 1] + zeta[i, j + 1]) * BYCSU[j]) * BYCSU[j]
            if j == nypole:
                dely2 = BYDY2[j]
                delyr = BYDYR[j]
                aa9 = ((eta[i, j + 1] + eta[i + 1, j + 1]) * dely2 * uicec[i, j + 1]
                       + e_mean * delyr * (TNG[j + 1] - 2.0 * TNG[j]) * uicec[i, j + 1]) \
                    * UVM[i, j] * NPOL
            else:
                aa9 = 0.0
            fxy1a[i, j] = aa9 + amass[i, j] * bydts * uice[1][i, j] * 2.0 - (aa1 + aa2) * delx2 * uice[1][i, j] \
                + ((eta[i + 1, j] + zeta[i + 1, j] + eta[i + 1, j + 1] + zeta[i + 1, j + 1]) * uice[1][i + 1, j]
                   + (eta[i, j] + zeta[i, j] + eta[i, j + 1] + zeta[i, j + 1]) * uice[1][i - 1, j]) \
                * delx2 * BYCSU[j] * BYCSU[j]

    vrt = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            vrt[i, j] = (fxy[i, j] + fxy1a[i, j]) * UVM[i, j]

    u_tmp = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        sl = slice(2, nypole + 1)
        u_tmp[i, sl] = tridiag_thomas(av[i, sl], bv[i, sl], cv[i, sl], vrt[i, sl])
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            uice[1][i, j] = u_tmp[i, j]

    # ---- NOW DO VICE: first half (J-direction) ----
    fxya = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            delxy = BYDXDY[i, j]
            delxr = BYDXR[i]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            z_mean = 0.25 * (zeta[i, j + 1] + zeta[i + 1, j + 1] + zeta[i, j] + zeta[i + 1, j])
            fxya[i, j] = -draga[i, j] * uicec[i, j] + forcey[i, j] \
                + (0.5 * (uicec[i + 1, j] - uicec[i - 1, j]) * (zeta[i, j + 1] + zeta[i + 1, j + 1]
                          - zeta[i, j] - zeta[i + 1, j]) * delxy * BYCSU[j]
                   + 0.5 * z_mean * ((uicec[i + 1, j + 1] - uicec[i - 1, j + 1]) * BYCSU[j + 1]
                          - (uicec[i + 1, j - 1] - uicec[i - 1, j - 1]) * BYCSU[j - 1]) * delxy) \
                - (0.5 * (uicec[i + 1, j] - uicec[i - 1, j]) * (eta[i, j + 1] + eta[i + 1, j + 1]
                          - eta[i, j] - eta[i + 1, j]) * delxy * BYCSU[j]
                   + 0.5 * e_mean * ((uicec[i + 1, j + 1] - uicec[i - 1, j + 1]) * BYCSU[j + 1]
                          - (uicec[i + 1, j - 1] - uicec[i - 1, j - 1]) * BYCSU[j - 1]) * delxy) \
                + 0.5 * (eta[i + 1, j + 1] * (uicec[i + 1, j + 1] + uicec[i, j + 1] - uicec[i + 1, j]
                         - uicec[i, j]) + eta[i + 1, j] * (uicec[i + 1, j] + uicec[i, j]
                         - uicec[i + 1, j - 1] - uicec[i, j - 1]) + eta[i, j + 1] * (uicec[i, j]
                         + uicec[i - 1, j] - uicec[i, j + 1] - uicec[i - 1, j + 1]) + eta[i, j] *
                         (uicec[i, j - 1] + uicec[i - 1, j - 1] - uicec[i, j] - uicec[i - 1, j])) \
                * delxy * BYCSU[j] \
                + (eta[i + 1, j + 1] + eta[i + 1, j] - eta[i, j] - eta[i, j + 1]) \
                * TNG[j] * uicec[i, j] * delxr * BYCSU[j] \
                + e_mean * TNG[j] * (uicec[i + 1, j] - uicec[i - 1, j]) * delxr * BYCSU[j] \
                + e_mean * 2.0 * TNG[j] * (uicec[i + 1, j] - uicec[i - 1, j]) * delxr * BYCSU[j]

    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            dely2 = BYDY2[j]
            delyr = BYDYR[j]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            z_mean = 0.25 * (zeta[i, j + 1] + zeta[i + 1, j + 1] + zeta[i, j] + zeta[i + 1, j])
            aa1 = eta[i, j + 1] + zeta[i, j + 1] + eta[i + 1, j + 1] + zeta[i + 1, j + 1]
            aa2 = eta[i, j] + zeta[i, j] + eta[i + 1, j] + zeta[i + 1, j]
            aa5 = ((zeta[i, j + 1] - eta[i, j + 1]) + (zeta[i + 1, j + 1] - eta[i + 1, j + 1])
                   - (zeta[i, j] - eta[i, j]) - (zeta[i + 1, j] - eta[i + 1, j])) * TNG[j]
            aa6 = 2.0 * e_mean * TNG[j] * TNG[j]
            av[i, j] = (-aa2 * dely2 - (z_mean - e_mean) * TNG[j - 1] * delyr
                        - e_mean * 2.0 * TNG[j] * delyr) * UVM[i, j]
            bv[i, j] = ((aa1 + aa2) * dely2 + aa5 * delyr + aa6 * BYRAD2 + amass[i, j] * bydts * 2.0
                        + drags[i, j]) * UVM[i, j] + (1.0 - UVM[i, j])
            cv[i, j] = (-aa1 * dely2 + (z_mean - e_mean) * TNG[j + 1] * delyr
                        + e_mean * 2.0 * TNG[j] * delyr) * UVM[i, j]
    for i in range(2, nxlcyc + 1):
        av[i, 2] = 0.0
        cv[i, nypole] = 0.0

    vrt2 = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            dely2 = BYDY2[j]
            delyr = BYDYR[j]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            z_mean = 0.25 * (zeta[i, j + 1] + zeta[i + 1, j + 1] + zeta[i, j] + zeta[i + 1, j])
            aa1 = eta[i, j + 1] + zeta[i, j + 1] + eta[i + 1, j + 1] + zeta[i + 1, j + 1]
            aa3 = (eta[i + 1, j] * BYCSU[j] + eta[i + 1, j + 1] * BYCSU[j]) * BYCSU[j]
            aa4 = (eta[i, j] * BYCSU[j] + eta[i, j + 1] * BYCSU[j]) * BYCSU[j]
            if j == nypole:
                aa9 = (aa1 * dely2 - (z_mean - e_mean) * TNG[j + 1] * delyr
                       - e_mean * 2.0 * TNG[j] * delyr) * vicec[i, j + 1] * UVM[i, j] * NPOL
            else:
                aa9 = 0.0
            vrt2[i, j] = aa9 + fxya[i, j] - (aa3 + aa4) * BYDX2[i] * vicec[i, j] \
                + ((eta[i + 1, j] * BYCSU[j] + eta[i + 1, j + 1] * BYCSU[j]) * vicec[i + 1, j] * BYDX2[i]
                   + (eta[i, j] * BYCSU[j] + eta[i, j + 1] * BYCSU[j]) * vicec[i - 1, j] * BYDX2[i]) \
                * BYCSU[j]
            vrt2[i, j] = (vrt2[i, j] + amass[i, j] * bydts * vice[2][i, j] * 2.0) * UVM[i, j]

    u_tmp2 = _pad(nx1, ny1)
    for i in range(2, nxlcyc + 1):
        sl = slice(2, nypole + 1)
        u_tmp2[i, sl] = tridiag_thomas(av[i, sl], bv[i, sl], cv[i, sl], vrt2[i, sl])
    for i in range(2, nxlcyc + 1):
        for j in range(2, nypole + 1):
            vice[1][i, j] = u_tmp2[i, j]
            vice[3][i, j] = vice[1][i, j]

    # ---- VICE second half (I-direction, cyclic) ----
    au2 = _pad(nx1, ny1)
    bu2 = _pad(nx1, ny1)
    cu2 = _pad(nx1, ny1)
    for j in range(2, nypole + 1):
        for i in range(2, nxlcyc + 1):
            delx2 = BYDX2[i]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            aa3 = (eta[i + 1, j] * BYCSU[j] + eta[i + 1, j + 1] * BYCSU[j]) * BYCSU[j]
            aa4 = (eta[i, j] * BYCSU[j] + eta[i, j + 1] * BYCSU[j]) * BYCSU[j]
            aa6 = 2.0 * e_mean * TNG[j] * TNG[j]
            au2[i, j] = -aa4 * delx2 * UVM[i, j]
            bu2[i, j] = ((aa3 + aa4) * delx2 + aa6 * BYRAD2 + amass[i, j] * bydts * 2.0
                         + drags[i, j]) * UVM[i, j] + (1.0 - UVM[i, j])
            cu2[i, j] = -aa3 * delx2 * UVM[i, j]

    fxy1 = _pad(nx1, ny1)
    for j in range(2, nypole + 1):
        for i in range(2, nxlcyc + 1):
            dely2 = BYDY2[j]
            delyr = BYDYR[j]
            e_mean = 0.25 * (eta[i, j + 1] + eta[i + 1, j + 1] + eta[i, j] + eta[i + 1, j])
            z_mean = 0.25 * (zeta[i, j + 1] + zeta[i + 1, j + 1] + zeta[i, j] + zeta[i + 1, j])
            aa1 = eta[i, j + 1] + zeta[i, j + 1] + eta[i + 1, j + 1] + zeta[i + 1, j + 1]
            aa2 = eta[i, j] + zeta[i, j] + eta[i + 1, j] + zeta[i + 1, j]
            aa5 = ((zeta[i, j + 1] - eta[i, j + 1]) + (zeta[i + 1, j + 1] - eta[i + 1, j + 1])
                   - (zeta[i, j] - eta[i, j]) - (zeta[i + 1, j] - eta[i + 1, j])) * TNG[j]
            fxy1[i, j] = amass[i, j] * bydts * vice[1][i, j] * 2.0 - aa5 * delyr * vice[1][i, j] \
                - (aa1 + aa2) * dely2 * vice[1][i, j] \
                + aa1 * dely2 * vice[1][i, j + 1] - ((z_mean - e_mean) * TNG[j + 1] * delyr
                    + e_mean * 2.0 * TNG[j] * delyr) * vice[1][i, j + 1] \
                + aa2 * dely2 * vice[1][i, j - 1] + ((z_mean - e_mean) * TNG[j - 1] * delyr
                    + e_mean * 2.0 * TNG[j] * delyr) * vice[1][i, j - 1]

    for j in range(2, nypole + 1):
        urt2 = np.zeros(nx1 + 1)
        for i in range(2, nxlcyc + 1):
            urt2[i] = (fxya[i, j] + fxy1[i, j]) * UVM[i, j]
        sl = slice(2, nxlcyc + 1)
        vice[1][sl, j] = tridiag_cyclic(au2[sl, j], bu2[sl, j], cu2[sl, j], urt2[sl])

    for j in range(2, nypole + 1):
        for i in range(2, nxlcyc + 1):
            uice[1][i, j] *= UVM[i, j]
            vice[1][i, j] *= UVM[i, j]

    return uice, vice


def vpicedyn(nx1, ny1, usi0, vsi0, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, sinwat, coswat,
             bydts, osurf_tilt, pgfub, pgfvb, max_kki=20, conv_tol=25e-6):
    """ICEDYN.f VPICEDYN: the outer pseudo-timestep loop. `usi0`/`vsi0` are the previous step's ice
    velocities on the (IMICDYN,NY1) grid (no ghost columns); this function builds the padded, cyclic
    UICE(:,:,1..3)/VICE(:,:,1..3) state, runs FORM+RELAX to convergence, and returns the new
    UICE(:,:,1)/VICE(:,:,1) fields (shape (NX1+1,NY1+1), still 1-padded)."""
    imicdyn = nx1 - 2
    uice = {k: _pad(nx1, ny1) for k in (1, 2, 3)}
    vice = {k: _pad(nx1, ny1) for k in (1, 2, 3)}
    for j in range(1, ny1 + 1):
        for i in range(1, imicdyn + 1):
            uice[1][i + 1, j] = usi0[i - 1, j - 1]
            vice[1][i + 1, j] = vsi0[i - 1, j - 1]
        uice[1][1, j] = usi0[imicdyn - 1, j - 1]
        uice[1][nx1, j] = usi0[0, j - 1]
        vice[1][1, j] = vsi0[imicdyn - 1, j - 1]
        vice[1][nx1, j] = vsi0[0, j - 1]
    # uice[2],uice[3] start at zero (Fortran: UICE(I,J,2)=0.; UICE(I,J,3)=0. initial)

    rms, rms0, area_tot = 0.0, 0.0, 0.0
    usave = vsave = None
    kki = 0
    while True:
        kki += 1
        uice[3] = uice[1].copy()
        vice[3] = vice[1].copy()
        uicec = uice[1].copy()
        vicec = vice[1].copy()

        f = form(nx1, ny1, uice[1], vice[1], gairx, gairy, gwatx, gwaty, heff, area, amass, cor,
                 (sinwat, coswat), osurf_tilt, pgfub, pgfvb)
        uice, vice = relax(nx1, ny1, uice, vice, uicec, vicec, f["forcex"], f["forcey"], f["draga"],
                            f["drags"], f["eta"], f["zeta"], amass, cor, bydts)

        for j in range(1, ny1 + 1):
            uice[1][1, j] = uice[1][nx1 - 1, j]
            vice[1][1, j] = vice[1][nx1 - 1, j]
            uice[1][nx1, j] = uice[1][2, j]
            vice[1][nx1, j] = vice[1][2, j]

        for j in range(1, ny1 + 1):
            for i in range(1, nx1 + 1):
                uice[1][i, j] = 0.5 * (uice[1][i, j] + uice[2][i, j])
                vice[1][i, j] = 0.5 * (vice[1][i, j] + vice[2][i, j])

        f = form(nx1, ny1, uice[1], vice[1], gairx, gairy, gwatx, gwaty, heff, area, amass, cor,
                 (sinwat, coswat), osurf_tilt, pgfub, pgfvb)
        last_dwatn = f["dwatn"]  # DMU/DMV in DYNSI use DWATN as left by VPICEDYN's last FORM call,
                                  # i.e. from the Euler-averaged UICE (this call), not the final
                                  # post-RELAX velocity -- matches the real module-array side effect.

        uice[3] = uice[1].copy()
        vice[3] = vice[1].copy()
        uicec = uice[1].copy()
        vicec = vice[1].copy()
        uice[1] = uice[2].copy()
        vice[1] = vice[2].copy()

        uice, vice = relax(nx1, ny1, uice, vice, uicec, vicec, f["forcex"], f["forcey"], f["draga"],
                            f["drags"], f["eta"], f["zeta"], amass, cor, bydts)

        for j in range(1, ny1 + 1):
            uice[1][1, j] = uice[1][nx1 - 1, j]
            vice[1][1, j] = vice[1][nx1 - 1, j]
            uice[1][nx1, j] = uice[1][2, j]
            vice[1][nx1, j] = vice[1][2, j]

        if kki > 1:
            rms_acc, area_acc = 0.0, 0.0
            for i in range(1, nx1 + 1):
                for j in range(1, ny1 + 1):
                    if amass[i, j] * UVM[i, j] > 0:
                        w = DXU[i] * DYU[j]
                        rms_acc += w * ((usave[i, j] - uice[1][i, j]) ** 2
                                        + (vsave[i, j] - vice[1][i, j]) ** 2)
                        area_acc += w
            rms, area_tot = rms_acc, area_acc

        if kki == max_kki:
            break
        elif kki == 1 or rms > conv_tol * area_tot:
            usave = uice[1].copy()
            vsave = vice[1].copy()
            rms0 = rms
        else:
            break

    return uice[1], vice[1], kki, last_dwatn
