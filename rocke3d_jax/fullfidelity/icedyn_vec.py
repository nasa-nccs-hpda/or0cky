"""Batched (numpy) sea-ice dynamics core (D87): array-operation versions of icedyn_dynsi_ff.py's
PLAST, FORM, RELAX, TRIDIAG (Thomas / cyclic) and VPICEDYN.

Same index convention as the scalar port (1-padded (NX1+1, NY1+1) arrays; Fortran (I,J) -> [I,J]) and
the same host-associated geometry globals: call icedyn_dynsi_ff.init_geometry (and set D.RADIUS /
D.BYRAD2) before using these functions; they read the geometry from that module at call time.

Batching: PLAST/FORM and every coefficient/RHS stencil in RELAX are pointwise stencils and are done as
shifted-slice array expressions (term order copied from the scalar code so rounding stays at ulp
level). The tridiagonal solves are batched across lines (all rows j, or all columns i, are independent)
with the recurrence kept as a Python loop along the line (axis 0 of the (n, nlines) work arrays).
The outer VPICEDYN convergence loop stays a Python while loop. Sequential reductions (pole-row mean,
RMS accumulators) use np.cumsum so the summation order equals the scalar port's left-to-right sum.
Nothing is left scalar.
"""
import numpy as np

import icedyn_dynsi_ff as D

init_geometry = D.init_geometry


def _pad(nx1, ny1):
    return np.zeros((nx1 + 1, ny1 + 1))


def _seqsum(x):
    """Left-to-right sum (what Python's sum() does), vectorised via cumsum."""
    return np.cumsum(x)[-1]


# ----------------------------------------------------------------------------- tridiagonal solves
def tridiag_thomas_batch(a, b, c, r):
    """Thomas algorithm for many independent systems. a,b,c,r have shape (n, L); recurrence along
    axis 0, systems along axis 1 (a[0] and c[-1] unused). Same operation order as tridiag_thomas."""
    n = b.shape[0]
    gam = np.zeros_like(b)
    u = np.zeros_like(b)
    bet = b[0].copy()
    u[0] = r[0] / bet
    for j in range(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def tridiag_cyclic_batch(a, b, c, r):
    """Cyclic tridiagonal (Thomas + Sherman-Morrison) for many systems; shapes (n, L), recurrence
    along axis 0. Same operation order as tridiag_cyclic, including the per-line b[0]==1 doubling."""
    n = b.shape[0]
    gam = np.zeros_like(b)
    q = np.zeros_like(b)
    u = np.zeros_like(b)
    a1, b1, c1, r1 = a[0], b[0], c[0], r[0]
    dbl = (b1 == 1.0)
    a1 = np.where(dbl, a1 * 2.0, a1)
    c1 = np.where(dbl, c1 * 2.0, c1)
    r1 = np.where(dbl, r1 * 2.0, r1)
    b1 = np.where(dbl, b1 * 2.0, b1)
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
    u = u - qcoeff * q
    return u


def tridiag_thomas(a, b, c, r):
    """1-D convenience wrapper (same signature as the scalar tridiag_thomas)."""
    return tridiag_thomas_batch(a[:, None], b[:, None], c[:, None], r[:, None])[:, 0]


def tridiag_cyclic(a, b, c, r):
    return tridiag_cyclic_batch(a[:, None], b[:, None], c[:, None], r[:, None])[:, 0]


# ----------------------------------------------------------------------------------------- PLAST
def plast(nx1, ny1, uice1, vice1, press):
    nypole = ny1 - 1
    ecm2 = 1.0 / (D.ECCEN ** 2)
    gmin = 1e-20
    radius = D.RADIUS

    def S(F, di, dj):
        return F[2 + di:nx1 + di, 2 + dj:ny1 + dj]

    dxt = D.DXT[2:nx1, None]
    cst = D.CST[None, 2:ny1]
    dyt = D.DYT[None, 2:ny1]
    tngt = D.TNGT[None, 2:ny1]
    U = lambda di, dj: S(uice1, di, dj)
    V = lambda di, dj: S(vice1, di, dj)
    e11 = 0.5 / (dxt * cst) * (U(0, 0) + U(0, -1) - U(-1, 0) - U(-1, -1)) \
        - 0.25 * (V(0, 0) + V(-1, 0) + V(-1, -1) + V(0, -1)) * tngt / radius
    e22 = 0.5 / dyt * (V(0, 0) + V(-1, 0) - V(0, -1) - V(-1, -1))
    e12 = 0.5 * (0.5 / dyt * (U(0, 0) + U(-1, 0) - U(0, -1) - U(-1, -1))
                 + 0.5 / (dxt * cst) * (V(0, 0) + V(0, -1) - V(-1, 0) - V(-1, -1))
                 + 0.25 * (U(0, 0) + U(-1, 0) + U(-1, -1) + U(0, -1)) * tngt / radius)
    delt = (e11 ** 2 + e22 ** 2) * (1.0 + ecm2) + 4.0 * ecm2 * e12 ** 2 \
        + 2.0 * e11 * e22 * (1.0 - ecm2)
    delt1 = np.maximum(gmin, np.sqrt(delt))
    zin = 0.5 * S(press, 0, 0) / delt1
    zmax = (5e12 / 2e4) * press
    zmin = np.full_like(press, 4e8)
    zin = np.minimum(S(zmax, 0, 0), zin)
    zin = np.maximum(S(zmin, 0, 0), zin)
    zeta = _pad(nx1, ny1)
    zeta[2:nx1, 2:ny1] = zin

    aaa = _seqsum(zeta[2:nx1, nypole]) / (nx1 - 2)
    zeta[1:nx1 + 1, ny1] = aaa
    aaa = _seqsum(zeta[2:nx1, 2]) / (nx1 - 2)
    zeta[1:nx1 + 1, 1] = aaa
    zeta[1, :] = zeta[nx1 - 1, :]
    zeta[nx1, :] = zeta[2, :]
    eta = ecm2 * zeta
    return eta, zeta


# ------------------------------------------------------------------------------------------ FORM
def form(nx1, ny1, uice1, vice1, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, dragsym_consts,
         osurf_tilt, pgfub, pgfvb):
    sinwat, coswat = dragsym_consts
    nypole = ny1 - 1
    HEFFM = D.HEFFM

    dwatn = _pad(nx1, ny1)
    rr = (slice(1, nx1), slice(1, ny1 + 1))
    dwatn[rr] = 5.5 * np.sqrt((uice1[rr] - gwatx[rr]) ** 2 + (vice1[rr] - gwaty[rr]) ** 2)

    drags = _pad(nx1, ny1)
    draga = _pad(nx1, ny1)
    forcex = _pad(nx1, ny1)
    forcey = _pad(nx1, ny1)
    half = ny1 // 2
    for (j0, j1, north) in ((1, half + 1, False), (half + 1, nypole + 1, True)):
        r = (slice(1, nx1), slice(j0, j1))
        dw, cr, gx, gy = dwatn[r], cor[r], gwatx[r], gwaty[r]
        drags[r] = dw * coswat
        fx = gairx[r].copy()
        fy = gairy[r].copy()
        if north:
            draga[r] = dw * sinwat + cr
            fx += dw * (coswat * gx - sinwat * gy)
            fy += dw * (sinwat * gx + coswat * gy)
        else:
            draga[r] = -dw * sinwat + cr
            fx += dw * (coswat * gx + sinwat * gy)
            fy += dw * (-sinwat * gx + coswat * gy)
        if osurf_tilt == 1:
            fx += amass[r] * pgfub[r]
            fy += amass[r] * pgfvb[r]
        else:
            fx -= cr * gy
            fy += cr * gx
        forcex[r] = fx
        forcey[r] = fy

    ra = (slice(1, nx1 + 1), slice(1, ny1 + 1))
    press = _pad(nx1, ny1)
    press[ra] = D.PSTAR * heff[ra] * np.exp(-20.0 * (1.0 - area[ra]))

    eta, zeta = plast(nx1, ny1, uice1, vice1, press)

    aaa = _seqsum(press[2:nx1, nypole]) / (nx1 - 2)
    press[1:nx1 + 1, ny1] = aaa
    press[1, :] = press[nx1 - 1, :]
    press[nx1, :] = press[2, :]

    press *= HEFFM
    eta *= HEFFM
    zeta *= HEFFM

    i0, j0 = slice(1, nx1), slice(1, nypole + 1)
    dxu = D.DXU[1:nx1, None]
    csu = D.CSU[None, 1:nypole + 1]
    dyu = D.DYU[None, 1:nypole + 1]
    p = lambda di, dj: press[1 + di:nx1 + di, 1 + dj:nypole + 1 + dj]
    forcex[i0, j0] -= (0.25 / (dxu * csu)) * (p(1, 0) + p(1, 1) - p(0, 0) - p(0, 1))
    forcey[i0, j0] -= 0.25 / dyu * (p(0, 1) + p(1, 1) - p(0, 0) - p(1, 0))

    forcex[1, :] = forcex[nx1 - 1, :]
    forcey[1, :] = forcey[nx1 - 1, :]
    forcex[nx1, :] = forcex[2, :]
    forcey[nx1, :] = forcey[2, :]
    dwatn[1, :] = dwatn[nx1 - 1, :]
    dwatn[nx1, :] = dwatn[2, :]

    return dict(dwatn=dwatn, drags=drags, draga=draga, forcex=forcex, forcey=forcey,
                press=press, eta=eta, zeta=zeta)


# ----------------------------------------------------------------------------------------- RELAX
def relax(nx1, ny1, uice, vice, uicec, vicec, forcex, forcey, draga, drags, eta, zeta, amass, cor,
          bydts):
    """Batched RELAX; same contract as the scalar relax (mutates uice/vice dicts, uicec/vicec and
    forcex/forcey in place)."""
    nxlcyc = nx1 - 1
    nypole = ny1 - 1
    half = nx1 // 2
    pad = (nx1 - 2) // 2
    UVM = D.UVM
    BYRAD2 = D.BYRAD2
    NPOL = D.NPOL

    ra = (slice(1, nx1 + 1), slice(1, nypole + 1))
    forcex[ra] *= UVM[ra]
    forcey[ra] *= UVM[ra]
    uice[2][ra] = uice[1][ra]
    vice[2][ra] = vice[1][ra]
    uice[1][ra] = uice[3][ra] * UVM[ra]
    vice[1][ra] = vice[3][ra] * UVM[ra]

    # north pole reflection (reads uicec row nypole only; writes row ny1)
    ii = np.arange(1, nx1)
    src = np.where(ii <= half, ii + pad, ii - pad)
    for fld, fldc in ((uice, uicec), (vice, vicec)):
        val = -fldc[src, nypole]
        fld[1][ii, ny1] = val
        fld[3][ii, ny1] = val
        fldc[ii, ny1] = val

    jj = slice(1, nypole + 1)
    for fld, fldc in ((uice, uicec), (vice, vicec)):
        fld[1][1, jj] = fldc[nx1 - 1, jj]
        fld[1][nx1, jj] = fldc[2, jj]
        fld[3][1, jj] = fldc[nx1 - 1, jj]
        fld[3][nx1, jj] = fldc[2, jj]
        fldc[1, jj] = fldc[nx1 - 1, jj]
        fldc[nx1, jj] = fldc[2, jj]

    # ---- shifted views over the solve region i=2..nxlcyc, j=2..nypole ----
    def S(F, di, dj):
        return F[2 + di:nx1 + di, 2 + dj:ny1 + dj]

    def Ci(A, di):
        return A[2 + di:nx1 + di, None]

    def Cj(A, dj):
        return A[None, 2 + dj:ny1 + dj]

    E = lambda di, dj: S(eta, di, dj)
    Z = lambda di, dj: S(zeta, di, dj)
    Uc = lambda di, dj: S(uicec, di, dj)
    Vc = lambda di, dj: S(vicec, di, dj)

    uvm = S(UVM, 0, 0)
    am = S(amass, 0, 0)
    drg = S(drags, 0, 0)
    dga = S(draga, 0, 0)
    delxy = S(D.BYDXDY, 0, 0)
    delxr = Ci(D.BYDXR, 0)
    delx2 = Ci(D.BYDX2, 0)
    dely2 = Cj(D.BYDY2, 0)
    delyr = Cj(D.BYDYR, 0)
    bycsu0, bycsu1, bycsum1 = Cj(D.BYCSU, 0), Cj(D.BYCSU, 1), Cj(D.BYCSU, -1)
    tng0, tng1, tngm1 = Cj(D.TNG, 0), Cj(D.TNG, 1), Cj(D.TNG, -1)
    e_mean = 0.25 * (E(0, 1) + E(1, 1) + E(0, 0) + E(1, 0))
    z_mean = 0.25 * (Z(0, 1) + Z(1, 1) + Z(0, 0) + Z(1, 0))
    one_m_uvm = 1.0 - uvm
    aa6 = 2.0 * e_mean * tng0 * tng0
    interior = (slice(2, nx1), slice(2, ny1))

    def put(F, val):
        F[interior] = val

    # ---- FIRST DO UICE: first half (I-direction, cyclic) ----
    fxy = (dga * Vc(0, 0) + S(forcex, 0, 0)
           + 0.5 * (Z(1, 1) * (Vc(1, 1) + Vc(0, 1) - Vc(1, 0) - Vc(0, 0))
                    + Z(1, 0) * (Vc(1, 0) + Vc(0, 0) - Vc(1, -1) - Vc(0, -1))
                    + Z(0, 1) * (Vc(0, 0) + Vc(-1, 0) - Vc(0, 1) - Vc(-1, 1))
                    + Z(0, 0) * (Vc(0, -1) + Vc(-1, -1) - Vc(0, 0) - Vc(-1, 0))) * delxy * bycsu0
           - 0.5 * (E(1, 1) * (Vc(1, 1) + Vc(0, 1) - Vc(1, 0) - Vc(0, 0))
                    + E(1, 0) * (Vc(1, 0) + Vc(0, 0) - Vc(1, -1) - Vc(0, -1))
                    + E(0, 1) * (Vc(0, 0) + Vc(-1, 0) - Vc(0, 1) - Vc(-1, 1))
                    + E(0, 0) * (Vc(0, -1) + Vc(-1, -1) - Vc(0, 0) - Vc(-1, 0))) * delxy * bycsu0
           + 0.5 * (Vc(1, 0) - Vc(-1, 0)) * (E(0, 1) + E(1, 1) - E(0, 0) - E(1, 0)) * delxy * bycsu0
           + 0.5 * e_mean * ((Vc(1, 1) - Vc(-1, 1)) * bycsu1
                             - (Vc(1, -1) - Vc(-1, -1)) * bycsum1) * delxy
           - ((Z(1, 1) + Z(1, 0) - Z(0, 0) - Z(0, 1)) + (E(1, 1) + E(1, 0) - E(0, 0) - E(0, 1)))
           * tng0 * Vc(0, 0) * delxr * bycsu0
           - (e_mean + 0.25 * (Z(0, 1) + Z(1, 1) + Z(0, 0) + Z(1, 0)))
           * tng0 * (Vc(1, 0) - Vc(-1, 0)) * delxr * bycsu0
           - e_mean * 2.0 * tng0 * (Vc(1, 0) - Vc(-1, 0)) * delxr * bycsu0)

    aa1 = ((E(1, 0) + Z(1, 0)) * bycsu0 + (E(1, 1) + Z(1, 1)) * bycsu0) * bycsu0
    aa2 = ((E(0, 0) + Z(0, 0)) * bycsu0 + (E(0, 1) + Z(0, 1)) * bycsu0) * bycsu0
    au = -aa2 * delx2 * uvm
    bu = ((aa1 + aa2) * delx2 + aa6 * BYRAD2 + am * bydts * 2.0 + drg) * uvm + one_m_uvm
    cu = -aa1 * delx2 * uvm

    aa3 = E(0, 1) + E(1, 1)
    aa4 = E(0, 0) + E(1, 0)
    aa5 = -(E(0, 1) + E(1, 1) - E(0, 0) - E(1, 0)) * tng0
    urt = fxy - aa5 * delyr * Uc(0, 0) - (aa3 + aa4) * dely2 * Uc(0, 0) \
        + (E(0, 1) + E(1, 1)) * Uc(0, 1) * dely2 \
        + (E(0, 0) + E(1, 0)) * Uc(0, -1) * dely2 \
        + e_mean * delyr * (Uc(0, 1) * tng1 - Uc(0, -1) * tngm1) \
        - e_mean * delyr * 2.0 * tng0 * (Uc(0, 1) - Uc(0, -1))
    urt = (urt + am * bydts * S(uice[2], 0, 0) * 2.0) * uvm
    put(uice[1], tridiag_cyclic_batch(au, bu, cu, urt))
    uice[1][1, 2:ny1] = uice[1][nx1 - 1, 2:ny1]
    uice[1][nx1, 2:ny1] = uice[1][2, 2:ny1]

    uice[3][interior] = uice[1][interior]

    # ---- second half (J-direction) ----
    aa1 = E(0, 1) + E(1, 1)
    aa2 = E(0, 0) + E(1, 0)
    aa5 = -(E(0, 1) + E(1, 1) - E(0, 0) - E(1, 0)) * tng0
    av = (-aa2 * dely2 + e_mean * delyr * (tngm1 - 2.0 * tng0)) * uvm
    bv = ((aa1 + aa2) * dely2 + aa5 * delyr + aa6 * BYRAD2 + am * bydts * 2.0 + drg) * uvm + one_m_uvm
    cv = (-aa1 * dely2 - e_mean * delyr * (tng1 - 2.0 * tng0)) * uvm
    av[:, 0] = 0.0
    cv[:, -1] = 0.0

    aa1c = ((E(1, 0) + Z(1, 0)) * bycsu0 + (E(1, 1) + Z(1, 1)) * bycsu0) * bycsu0
    aa2c = ((E(0, 0) + Z(0, 0)) * bycsu0 + (E(0, 1) + Z(0, 1)) * bycsu0) * bycsu0
    aa9 = np.zeros_like(uvm)
    aa9[:, -1] = (((E(0, 1) + E(1, 1)) * dely2 * Uc(0, 1)
                   + e_mean * delyr * (tng1 - 2.0 * tng0) * Uc(0, 1)) * uvm * NPOL)[:, -1]
    U1 = lambda di, dj: S(uice[1], di, dj)
    fxy1a = aa9 + am * bydts * U1(0, 0) * 2.0 - (aa1c + aa2c) * delx2 * U1(0, 0) \
        + ((E(1, 0) + Z(1, 0) + E(1, 1) + Z(1, 1)) * U1(1, 0)
           + (E(0, 0) + Z(0, 0) + E(0, 1) + Z(0, 1)) * U1(-1, 0)) \
        * delx2 * bycsu0 * bycsu0
    vrt = (fxy + fxy1a) * uvm
    put(uice[1], tridiag_thomas_batch(av.T, bv.T, cv.T, vrt.T).T)

    # ---- NOW DO VICE: first half (J-direction) ----
    fxya = (-dga * Uc(0, 0) + S(forcey, 0, 0)
            + (0.5 * (Uc(1, 0) - Uc(-1, 0)) * (Z(0, 1) + Z(1, 1) - Z(0, 0) - Z(1, 0)) * delxy * bycsu0
               + 0.5 * z_mean * ((Uc(1, 1) - Uc(-1, 1)) * bycsu1
                                 - (Uc(1, -1) - Uc(-1, -1)) * bycsum1) * delxy)
            - (0.5 * (Uc(1, 0) - Uc(-1, 0)) * (E(0, 1) + E(1, 1) - E(0, 0) - E(1, 0)) * delxy * bycsu0
               + 0.5 * e_mean * ((Uc(1, 1) - Uc(-1, 1)) * bycsu1
                                 - (Uc(1, -1) - Uc(-1, -1)) * bycsum1) * delxy)
            + 0.5 * (E(1, 1) * (Uc(1, 1) + Uc(0, 1) - Uc(1, 0) - Uc(0, 0))
                     + E(1, 0) * (Uc(1, 0) + Uc(0, 0) - Uc(1, -1) - Uc(0, -1))
                     + E(0, 1) * (Uc(0, 0) + Uc(-1, 0) - Uc(0, 1) - Uc(-1, 1))
                     + E(0, 0) * (Uc(0, -1) + Uc(-1, -1) - Uc(0, 0) - Uc(-1, 0))) * delxy * bycsu0
            + (E(1, 1) + E(1, 0) - E(0, 0) - E(0, 1)) * tng0 * Uc(0, 0) * delxr * bycsu0
            + e_mean * tng0 * (Uc(1, 0) - Uc(-1, 0)) * delxr * bycsu0
            + e_mean * 2.0 * tng0 * (Uc(1, 0) - Uc(-1, 0)) * delxr * bycsu0)

    aa1 = E(0, 1) + Z(0, 1) + E(1, 1) + Z(1, 1)
    aa2 = E(0, 0) + Z(0, 0) + E(1, 0) + Z(1, 0)
    aa5 = ((Z(0, 1) - E(0, 1)) + (Z(1, 1) - E(1, 1))
           - (Z(0, 0) - E(0, 0)) - (Z(1, 0) - E(1, 0))) * tng0
    av = (-aa2 * dely2 - (z_mean - e_mean) * tngm1 * delyr - e_mean * 2.0 * tng0 * delyr) * uvm
    bv = ((aa1 + aa2) * dely2 + aa5 * delyr + aa6 * BYRAD2 + am * bydts * 2.0 + drg) * uvm + one_m_uvm
    cv = (-aa1 * dely2 + (z_mean - e_mean) * tng1 * delyr + e_mean * 2.0 * tng0 * delyr) * uvm
    av[:, 0] = 0.0
    cv[:, -1] = 0.0

    aa3 = (E(1, 0) * bycsu0 + E(1, 1) * bycsu0) * bycsu0
    aa4 = (E(0, 0) * bycsu0 + E(0, 1) * bycsu0) * bycsu0
    aa9 = np.zeros_like(uvm)
    aa9[:, -1] = ((aa1 * dely2 - (z_mean - e_mean) * tng1 * delyr
                   - e_mean * 2.0 * tng0 * delyr) * Vc(0, 1) * uvm * NPOL)[:, -1]
    bydx2 = delx2
    vrt2 = aa9 + fxya - (aa3 + aa4) * bydx2 * Vc(0, 0) \
        + ((E(1, 0) * bycsu0 + E(1, 1) * bycsu0) * Vc(1, 0) * bydx2
           + (E(0, 0) * bycsu0 + E(0, 1) * bycsu0) * Vc(-1, 0) * bydx2) * bycsu0
    vrt2 = (vrt2 + am * bydts * S(vice[2], 0, 0) * 2.0) * uvm
    sol = tridiag_thomas_batch(av.T, bv.T, cv.T, vrt2.T).T
    put(vice[1], sol)
    put(vice[3], sol)

    # ---- VICE second half (I-direction, cyclic) ----
    au2 = -aa4 * delx2 * uvm
    bu2 = ((aa3 + aa4) * delx2 + aa6 * BYRAD2 + am * bydts * 2.0 + drg) * uvm + one_m_uvm
    cu2 = -aa3 * delx2 * uvm
    V1 = lambda di, dj: S(vice[1], di, dj)
    fxy1 = am * bydts * V1(0, 0) * 2.0 - aa5 * delyr * V1(0, 0) \
        - (aa1 + aa2) * dely2 * V1(0, 0) \
        + aa1 * dely2 * V1(0, 1) - ((z_mean - e_mean) * tng1 * delyr
                                    + e_mean * 2.0 * tng0 * delyr) * V1(0, 1) \
        + aa2 * dely2 * V1(0, -1) + ((z_mean - e_mean) * tngm1 * delyr
                                     + e_mean * 2.0 * tng0 * delyr) * V1(0, -1)
    urt2 = (fxya + fxy1) * uvm
    put(vice[1], tridiag_cyclic_batch(au2, bu2, cu2, urt2))

    uice[1][interior] *= uvm
    vice[1][interior] *= uvm
    return uice, vice


# --------------------------------------------------------------------------------------- VPICEDYN
def vpicedyn(nx1, ny1, usi0, vsi0, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, sinwat, coswat,
             bydts, osurf_tilt, pgfub, pgfvb, max_kki=20, conv_tol=25e-6):
    """Batched VPICEDYN. Same contract/returns as the scalar vpicedyn: (UICE1, VICE1, kki, last_dwatn).
    The outer convergence loop is the same data-dependent Python while loop."""
    imicdyn = nx1 - 2
    uice = {k: _pad(nx1, ny1) for k in (1, 2, 3)}
    vice = {k: _pad(nx1, ny1) for k in (1, 2, 3)}
    uice[1][2:imicdyn + 2, 1:ny1 + 1] = usi0[0:imicdyn, 0:ny1]
    vice[1][2:imicdyn + 2, 1:ny1 + 1] = vsi0[0:imicdyn, 0:ny1]
    uice[1][1, 1:ny1 + 1] = usi0[imicdyn - 1, 0:ny1]
    uice[1][nx1, 1:ny1 + 1] = usi0[0, 0:ny1]
    vice[1][1, 1:ny1 + 1] = vsi0[imicdyn - 1, 0:ny1]
    vice[1][nx1, 1:ny1 + 1] = vsi0[0, 0:ny1]

    UVM = D.UVM
    jall = slice(1, ny1 + 1)
    rall = (slice(1, nx1 + 1), slice(1, ny1 + 1))

    def wrap():
        for fld in (uice[1], vice[1]):
            fld[1, jall] = fld[nx1 - 1, jall]
            fld[nx1, jall] = fld[2, jall]

    rms, area_tot = 0.0, 0.0
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
        wrap()

        uice[1][rall] = 0.5 * (uice[1][rall] + uice[2][rall])
        vice[1][rall] = 0.5 * (vice[1][rall] + vice[2][rall])

        f = form(nx1, ny1, uice[1], vice[1], gairx, gairy, gwatx, gwaty, heff, area, amass, cor,
                 (sinwat, coswat), osurf_tilt, pgfub, pgfvb)
        last_dwatn = f["dwatn"]

        uice[3] = uice[1].copy()
        vice[3] = vice[1].copy()
        uicec = uice[1].copy()
        vicec = vice[1].copy()
        uice[1] = uice[2].copy()
        vice[1] = vice[2].copy()

        uice, vice = relax(nx1, ny1, uice, vice, uicec, vicec, f["forcex"], f["forcey"], f["draga"],
                           f["drags"], f["eta"], f["zeta"], amass, cor, bydts)
        wrap()

        if kki > 1:
            sel = (amass[rall] * UVM[rall]) > 0
            w = D.DXU[1:nx1 + 1, None] * D.DYU[None, 1:ny1 + 1]
            term = w * ((usave[rall] - uice[1][rall]) ** 2 + (vsave[rall] - vice[1][rall]) ** 2)
            # scalar order: i outer, j inner, left-to-right accumulation
            wsel = np.broadcast_to(w, term.shape)[sel]
            tsel = term[sel]
            rms = _seqsum(tsel) if tsel.size else 0.0
            area_tot = _seqsum(wsel) if wsel.size else 0.0

        if kki == max_kki:
            break
        elif kki == 1 or rms > conv_tol * area_tot:
            usave = uice[1].copy()
            vsave = vice[1].copy()
        else:
            break

    return uice[1], vice[1], kki, last_dwatn
