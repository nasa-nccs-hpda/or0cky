"""JAX sea-ice dynamics (D88): icedyn_vec's PLAST/FORM/RELAX/TRIDIAG under jax.jit (float64).

Same stencils and term order as icedyn_vec; the pointwise stencils are shifted-slice jnp expressions and
the tridiagonal solves are batched across lines with lax.scan along the line (Thomas forward/backward;
the Sherman-Morrison cyclic variant for the longitude solves). FORM and RELAX are each one jitted
function; the data-dependent outer VPICEDYN convergence loop stays a Python while loop around them (the
RMS test is evaluated on the host). Geometry is read from icedyn_dynsi_ff's module globals (call
init_geometry first, as for the scalar and numpy versions) and passed to the jitted functions as a pytree,
so changing grids/radius needs no recompile beyond a shape change.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import icedyn_dynsi_ff as D
import icedyn_vec as V   # only for the host-side input packing contract


def geometry():
    """Snapshot of the scalar module's geometry globals as a pytree of jnp arrays."""
    names = ("DXT", "DXU", "DYT", "DYU", "BYDX2", "BYDXR", "BYDY2", "BYDYR", "BYDXDY", "CST", "CSU",
             "TNGT", "TNG", "BYCSU", "HEFFM", "UVM")
    G = {n: jnp.asarray(getattr(D, n)) for n in names}
    G["RADIUS"] = jnp.asarray(D.RADIUS)
    G["BYRAD2"] = jnp.asarray(D.BYRAD2)
    return G


# ------------------------------------------------------------------------------ tridiagonal solves
def tridiag_thomas_batch(a, b, c, r):
    """(n, L) systems, recurrence along axis 0 via lax.scan."""
    bet0 = b[0]
    u0 = r[0] / bet0

    def fwd(carry, x):
        bet, uprev = carry
        aj, bj, cjm1, rj = x
        gam = cjm1 / bet
        bet = bj - aj * gam
        uj = (rj - aj * uprev) / bet
        return (bet, uj), (gam, uj)

    _, (gam, uf) = lax.scan(fwd, (bet0, u0), (a[1:], b[1:], c[:-1], r[1:]))
    # gam[j] for j=1..n-1 ; uf[j-1] = u[j]
    def bwd(unext, x):
        uj, gnext = x
        uj = uj - gnext * unext
        return uj, uj

    ulast = uf[-1]
    _, ub = lax.scan(bwd, ulast, (uf[:-1], gam[1:]), reverse=True)  # u[0..n-2] excluding u0 handled below
    # u[0] needs its own back-substitution step with gam[1]
    u_head = u0 - gam[0] * ub[0]
    return jnp.concatenate([u_head[None], ub, ulast[None]], axis=0)


def tridiag_cyclic_batch(a, b, c, r):
    n = b.shape[0]
    a1, b1, c1, r1 = a[0], b[0], c[0], r[0]
    dbl = (b1 == 1.0)
    a1 = jnp.where(dbl, a1 * 2.0, a1)
    c1 = jnp.where(dbl, c1 * 2.0, c1)
    r1 = jnp.where(dbl, r1 * 2.0, r1)
    b1 = jnp.where(dbl, b1 * 2.0, b1)
    bet = b1 - 1.0
    u0 = r1 / bet
    q0 = 1.0 / bet
    gam1 = c1 / bet

    def fwd(carry, x):
        gam, uprev, qprev = carry
        aj, bj, cj, rj = x
        bet = bj - aj * gam
        uj = (rj - aj * uprev) / bet
        qj = -aj * qprev / bet
        return (cj / bet, uj, qj), (uj, qj, cj / bet)

    (gam_l, up, qp), (um, qm, gnext) = lax.scan(
        fwd, (gam1, u0, q0), (a[1:n - 1], b[1:n - 1], c[1:n - 1], r[1:n - 1]))
    # gam[j] for j=1..n-1 : [gam1, gnext...]
    gam = jnp.concatenate([gam1[None], gnext], axis=0)       # index k -> gam[k+1]
    bet = b[n - 1] - a[n - 1] * gam_l - a1 * c[n - 1]
    ul = (r[n - 1] - a[n - 1] * up) / bet
    ql = (c[n - 1] - a[n - 1] * qp) / bet
    uf = jnp.concatenate([u0[None], um], axis=0)             # u[0..n-2]
    qf = jnp.concatenate([q0[None], qm], axis=0)

    def bwd(carry, x):
        un, qn = carry
        uj, qj, gn = x
        uj = uj - gn * un
        qj = qj - gn * qn
        return (uj, qj), (uj, qj)

    _, (ub, qb) = lax.scan(bwd, (ul, ql), (uf, qf, gam), reverse=True)
    u = jnp.concatenate([ub, ul[None]], axis=0)
    q = jnp.concatenate([qb, ql[None]], axis=0)
    bet = 1.0 + q[0] + a1 * q[n - 1]
    qcoeff = (u[0] + a1 * u[n - 1]) / bet
    return u - qcoeff * q


# ------------------------------------------------------------------------------------------- PLAST
def _seqsum(x):
    return jnp.cumsum(x)[-1]


def _plast(G, uice1, vice1, press):
    nx1, ny1 = uice1.shape[0] - 1, uice1.shape[1] - 1
    nypole = ny1 - 1
    ecm2 = 1.0 / (D.ECCEN ** 2)
    gmin = 1e-20
    radius = G["RADIUS"]
    S = lambda F, di, dj: F[2 + di:nx1 + di, 2 + dj:ny1 + dj]
    dxt = G["DXT"][2:nx1, None]
    cst = G["CST"][None, 2:ny1]
    dyt = G["DYT"][None, 2:ny1]
    tngt = G["TNGT"][None, 2:ny1]
    U = lambda di, dj: S(uice1, di, dj)
    Vv = lambda di, dj: S(vice1, di, dj)
    e11 = 0.5 / (dxt * cst) * (U(0, 0) + U(0, -1) - U(-1, 0) - U(-1, -1)) \
        - 0.25 * (Vv(0, 0) + Vv(-1, 0) + Vv(-1, -1) + Vv(0, -1)) * tngt / radius
    e22 = 0.5 / dyt * (Vv(0, 0) + Vv(-1, 0) - Vv(0, -1) - Vv(-1, -1))
    e12 = 0.5 * (0.5 / dyt * (U(0, 0) + U(-1, 0) - U(0, -1) - U(-1, -1))
                 + 0.5 / (dxt * cst) * (Vv(0, 0) + Vv(0, -1) - Vv(-1, 0) - Vv(-1, -1))
                 + 0.25 * (U(0, 0) + U(-1, 0) + U(-1, -1) + U(0, -1)) * tngt / radius)
    delt = (e11 ** 2 + e22 ** 2) * (1.0 + ecm2) + 4.0 * ecm2 * e12 ** 2 \
        + 2.0 * e11 * e22 * (1.0 - ecm2)
    delt1 = jnp.maximum(gmin, jnp.sqrt(delt))
    zin = 0.5 * S(press, 0, 0) / delt1
    zmax = (5e12 / 2e4) * press
    zin = jnp.minimum(S(zmax, 0, 0), zin)
    zin = jnp.maximum(4e8, zin)
    zeta = jnp.zeros_like(press).at[2:nx1, 2:ny1].set(zin)
    zeta = zeta.at[1:nx1 + 1, ny1].set(_seqsum(zeta[2:nx1, nypole]) / (nx1 - 2))
    zeta = zeta.at[1:nx1 + 1, 1].set(_seqsum(zeta[2:nx1, 2]) / (nx1 - 2))
    zeta = zeta.at[1, :].set(zeta[nx1 - 1, :])
    zeta = zeta.at[nx1, :].set(zeta[2, :])
    return ecm2 * zeta, zeta


# -------------------------------------------------------------------------------------------- FORM
@jax.jit
def _form(G, uice1, vice1, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, pgfub, pgfvb,
          sinwat, coswat):
    """FORM with OSURF_TILT == 1 (the validated configuration)."""
    nx1, ny1 = uice1.shape[0] - 1, uice1.shape[1] - 1
    nypole = ny1 - 1
    z = jnp.zeros_like(uice1)
    rr = (slice(1, nx1), slice(1, ny1 + 1))
    dwatn = z.at[rr].set(5.5 * jnp.sqrt((uice1[rr] - gwatx[rr]) ** 2 + (vice1[rr] - gwaty[rr]) ** 2))
    drags, draga, forcex, forcey = z, z, z, z
    half = ny1 // 2
    for (j0, j1, north) in ((1, half + 1, False), (half + 1, nypole + 1, True)):
        r = (slice(1, nx1), slice(j0, j1))
        dw, cr, gx, gy = dwatn[r], cor[r], gwatx[r], gwaty[r]
        drags = drags.at[r].set(dw * coswat)
        fx, fy = gairx[r], gairy[r]
        if north:
            draga = draga.at[r].set(dw * sinwat + cr)
            fx = fx + dw * (coswat * gx - sinwat * gy)
            fy = fy + dw * (sinwat * gx + coswat * gy)
        else:
            draga = draga.at[r].set(-dw * sinwat + cr)
            fx = fx + dw * (coswat * gx + sinwat * gy)
            fy = fy + dw * (-sinwat * gx + coswat * gy)
        fx = fx + amass[r] * pgfub[r]
        fy = fy + amass[r] * pgfvb[r]
        forcex = forcex.at[r].set(fx)
        forcey = forcey.at[r].set(fy)
    ra = (slice(1, nx1 + 1), slice(1, ny1 + 1))
    press = z.at[ra].set(D.PSTAR * heff[ra] * jnp.exp(-20.0 * (1.0 - area[ra])))
    eta, zeta = _plast(G, uice1, vice1, press)
    press = press.at[1:nx1 + 1, ny1].set(_seqsum(press[2:nx1, nypole]) / (nx1 - 2))
    press = press.at[1, :].set(press[nx1 - 1, :])
    press = press.at[nx1, :].set(press[2, :])
    HEFFM = G["HEFFM"]
    press, eta, zeta = press * HEFFM, eta * HEFFM, zeta * HEFFM
    i0, j0 = slice(1, nx1), slice(1, nypole + 1)
    dxu = G["DXU"][1:nx1, None]
    csu = G["CSU"][None, 1:nypole + 1]
    dyu = G["DYU"][None, 1:nypole + 1]
    p = lambda di, dj: press[1 + di:nx1 + di, 1 + dj:nypole + 1 + dj]
    forcex = forcex.at[i0, j0].set(forcex[i0, j0] - (0.25 / (dxu * csu)) * (p(1, 0) + p(1, 1) - p(0, 0) - p(0, 1)))
    forcey = forcey.at[i0, j0].set(forcey[i0, j0] - 0.25 / dyu * (p(0, 1) + p(1, 1) - p(0, 0) - p(1, 0)))
    forcex = forcex.at[1, :].set(forcex[nx1 - 1, :]).at[nx1, :].set(forcex[2, :])
    forcey = forcey.at[1, :].set(forcey[nx1 - 1, :]).at[nx1, :].set(forcey[2, :])
    dwatn = dwatn.at[1, :].set(dwatn[nx1 - 1, :]).at[nx1, :].set(dwatn[2, :])
    return dict(dwatn=dwatn, drags=drags, draga=draga, forcex=forcex, forcey=forcey,
                press=press, eta=eta, zeta=zeta)


def form(nx1, ny1, uice1, vice1, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, dragsym_consts,
         osurf_tilt, pgfub, pgfvb, G=None):
    assert osurf_tilt == 1, "icedyn_jax.form supports the validated OSURF_TILT=1 configuration only"
    sinwat, coswat = dragsym_consts
    out = _form(G or geometry(), jnp.asarray(uice1), jnp.asarray(vice1), jnp.asarray(gairx),
                jnp.asarray(gairy), jnp.asarray(gwatx), jnp.asarray(gwaty), jnp.asarray(heff),
                jnp.asarray(area), jnp.asarray(amass), jnp.asarray(cor), jnp.asarray(pgfub),
                jnp.asarray(pgfvb), sinwat, coswat)
    return {k: np.asarray(v) for k, v in out.items()}


def plast(nx1, ny1, uice1, vice1, press, G=None):
    eta, zeta = jax.jit(_plast)(G or geometry(), jnp.asarray(uice1), jnp.asarray(vice1),
                                jnp.asarray(press))
    return np.asarray(eta), np.asarray(zeta)


# ------------------------------------------------------------------------------------------- RELAX
@jax.jit
def _relax(G, u1, u2, u3, v1, v2, v3, uicec, vicec, forcex, forcey, draga, drags, eta, zeta, amass,
           bydts):
    nx1, ny1 = u1.shape[0] - 1, u1.shape[1] - 1
    nypole = ny1 - 1
    half = nx1 // 2
    pad = (nx1 - 2) // 2
    UVM = G["UVM"]
    BYRAD2 = G["BYRAD2"]
    NPOL = D.NPOL

    ra = (slice(1, nx1 + 1), slice(1, nypole + 1))
    forcex = forcex.at[ra].set(forcex[ra] * UVM[ra])
    forcey = forcey.at[ra].set(forcey[ra] * UVM[ra])
    u2 = u2.at[ra].set(u1[ra])
    v2 = v2.at[ra].set(v1[ra])
    u1 = u1.at[ra].set(u3[ra] * UVM[ra])
    v1 = v1.at[ra].set(v3[ra] * UVM[ra])

    ii = np.arange(1, nx1)
    src = np.where(ii <= half, ii + pad, ii - pad)
    val = -uicec[src, nypole]
    u1 = u1.at[ii, ny1].set(val)
    u3 = u3.at[ii, ny1].set(val)
    uicec = uicec.at[ii, ny1].set(val)
    val = -vicec[src, nypole]
    v1 = v1.at[ii, ny1].set(val)
    v3 = v3.at[ii, ny1].set(val)
    vicec = vicec.at[ii, ny1].set(val)
    jj = slice(1, nypole + 1)
    u1 = u1.at[1, jj].set(uicec[nx1 - 1, jj]).at[nx1, jj].set(uicec[2, jj])
    u3 = u3.at[1, jj].set(uicec[nx1 - 1, jj]).at[nx1, jj].set(uicec[2, jj])
    v1 = v1.at[1, jj].set(vicec[nx1 - 1, jj]).at[nx1, jj].set(vicec[2, jj])
    v3 = v3.at[1, jj].set(vicec[nx1 - 1, jj]).at[nx1, jj].set(vicec[2, jj])
    uicec = uicec.at[1, jj].set(uicec[nx1 - 1, jj]).at[nx1, jj].set(uicec[2, jj])
    vicec = vicec.at[1, jj].set(vicec[nx1 - 1, jj]).at[nx1, jj].set(vicec[2, jj])

    S = lambda F, di, dj: F[2 + di:nx1 + di, 2 + dj:ny1 + dj]
    Ci = lambda A, di: A[2 + di:nx1 + di, None]
    Cj = lambda A, dj: A[None, 2 + dj:ny1 + dj]
    E = lambda di, dj: S(eta, di, dj)
    Z = lambda di, dj: S(zeta, di, dj)
    Uc = lambda di, dj: S(uicec, di, dj)
    Vc = lambda di, dj: S(vicec, di, dj)

    uvm = S(UVM, 0, 0)
    am = S(amass, 0, 0)
    drg = S(drags, 0, 0)
    dga = S(draga, 0, 0)
    delxy = S(G["BYDXDY"], 0, 0)
    delxr = Ci(G["BYDXR"], 0)
    delx2 = Ci(G["BYDX2"], 0)
    dely2 = Cj(G["BYDY2"], 0)
    delyr = Cj(G["BYDYR"], 0)
    bycsu0, bycsu1, bycsum1 = Cj(G["BYCSU"], 0), Cj(G["BYCSU"], 1), Cj(G["BYCSU"], -1)
    tng0, tng1, tngm1 = Cj(G["TNG"], 0), Cj(G["TNG"], 1), Cj(G["TNG"], -1)
    e_mean = 0.25 * (E(0, 1) + E(1, 1) + E(0, 0) + E(1, 0))
    z_mean = 0.25 * (Z(0, 1) + Z(1, 1) + Z(0, 0) + Z(1, 0))
    one_m_uvm = 1.0 - uvm
    aa6 = 2.0 * e_mean * tng0 * tng0
    interior = (slice(2, nx1), slice(2, ny1))

    # ---- UICE first half (I, cyclic) ----
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
    urt = (urt + am * bydts * S(u2, 0, 0) * 2.0) * uvm
    u1 = u1.at[interior].set(tridiag_cyclic_batch(au, bu, cu, urt))
    u1 = u1.at[1, 2:ny1].set(u1[nx1 - 1, 2:ny1]).at[nx1, 2:ny1].set(u1[2, 2:ny1])
    u3 = u3.at[interior].set(u1[interior])

    # ---- UICE second half (J) ----
    aa1 = E(0, 1) + E(1, 1)
    aa2 = E(0, 0) + E(1, 0)
    aa5 = -(E(0, 1) + E(1, 1) - E(0, 0) - E(1, 0)) * tng0
    av = (-aa2 * dely2 + e_mean * delyr * (tngm1 - 2.0 * tng0)) * uvm
    bv = ((aa1 + aa2) * dely2 + aa5 * delyr + aa6 * BYRAD2 + am * bydts * 2.0 + drg) * uvm + one_m_uvm
    cv = (-aa1 * dely2 - e_mean * delyr * (tng1 - 2.0 * tng0)) * uvm
    av = av.at[:, 0].set(0.0)
    cv = cv.at[:, -1].set(0.0)
    aa1c = ((E(1, 0) + Z(1, 0)) * bycsu0 + (E(1, 1) + Z(1, 1)) * bycsu0) * bycsu0
    aa2c = ((E(0, 0) + Z(0, 0)) * bycsu0 + (E(0, 1) + Z(0, 1)) * bycsu0) * bycsu0
    aa9 = jnp.zeros_like(uvm).at[:, -1].set(
        (((E(0, 1) + E(1, 1)) * dely2 * Uc(0, 1)
          + e_mean * delyr * (tng1 - 2.0 * tng0) * Uc(0, 1)) * uvm * NPOL)[:, -1])
    U1 = lambda di, dj: S(u1, di, dj)
    fxy1a = aa9 + am * bydts * U1(0, 0) * 2.0 - (aa1c + aa2c) * delx2 * U1(0, 0) \
        + ((E(1, 0) + Z(1, 0) + E(1, 1) + Z(1, 1)) * U1(1, 0)
           + (E(0, 0) + Z(0, 0) + E(0, 1) + Z(0, 1)) * U1(-1, 0)) \
        * delx2 * bycsu0 * bycsu0
    vrt = (fxy + fxy1a) * uvm
    u1 = u1.at[interior].set(tridiag_thomas_batch(av.T, bv.T, cv.T, vrt.T).T)

    # ---- VICE first half (J) ----
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
    av = av.at[:, 0].set(0.0)
    cv = cv.at[:, -1].set(0.0)
    aa3 = (E(1, 0) * bycsu0 + E(1, 1) * bycsu0) * bycsu0
    aa4 = (E(0, 0) * bycsu0 + E(0, 1) * bycsu0) * bycsu0
    aa9 = jnp.zeros_like(uvm).at[:, -1].set(
        ((aa1 * dely2 - (z_mean - e_mean) * tng1 * delyr
          - e_mean * 2.0 * tng0 * delyr) * Vc(0, 1) * uvm * NPOL)[:, -1])
    vrt2 = aa9 + fxya - (aa3 + aa4) * delx2 * Vc(0, 0) \
        + ((E(1, 0) * bycsu0 + E(1, 1) * bycsu0) * Vc(1, 0) * delx2
           + (E(0, 0) * bycsu0 + E(0, 1) * bycsu0) * Vc(-1, 0) * delx2) * bycsu0
    vrt2 = (vrt2 + am * bydts * S(v2, 0, 0) * 2.0) * uvm
    sol = tridiag_thomas_batch(av.T, bv.T, cv.T, vrt2.T).T
    v1 = v1.at[interior].set(sol)
    v3 = v3.at[interior].set(sol)

    # ---- VICE second half (I, cyclic) ----
    au2 = -aa4 * delx2 * uvm
    bu2 = ((aa3 + aa4) * delx2 + aa6 * BYRAD2 + am * bydts * 2.0 + drg) * uvm + one_m_uvm
    cu2 = -aa3 * delx2 * uvm
    V1 = lambda di, dj: S(v1, di, dj)
    fxy1 = am * bydts * V1(0, 0) * 2.0 - aa5 * delyr * V1(0, 0) \
        - (aa1 + aa2) * dely2 * V1(0, 0) \
        + aa1 * dely2 * V1(0, 1) - ((z_mean - e_mean) * tng1 * delyr
                                    + e_mean * 2.0 * tng0 * delyr) * V1(0, 1) \
        + aa2 * dely2 * V1(0, -1) + ((z_mean - e_mean) * tngm1 * delyr
                                     + e_mean * 2.0 * tng0 * delyr) * V1(0, -1)
    urt2 = (fxya + fxy1) * uvm
    v1 = v1.at[interior].set(tridiag_cyclic_batch(au2, bu2, cu2, urt2))
    u1 = u1.at[interior].set(u1[interior] * uvm)
    v1 = v1.at[interior].set(v1[interior] * uvm)
    return u1, u2, u3, v1, v2, v3, uicec, vicec


def relax(nx1, ny1, uice, vice, uicec, vicec, forcex, forcey, draga, drags, eta, zeta, amass, cor,
          bydts, G=None):
    """Dict-in/dict-out wrapper with the scalar relax's signature (returns new numpy dicts; the
    inputs are not mutated)."""
    a = lambda x: jnp.asarray(x)
    out = _relax(G or geometry(), a(uice[1]), a(uice[2]), a(uice[3]), a(vice[1]), a(vice[2]),
                 a(vice[3]), a(uicec), a(vicec), a(forcex), a(forcey), a(draga), a(drags), a(eta),
                 a(zeta), a(amass), bydts)
    u1, u2, u3, v1, v2, v3 = [np.asarray(x) for x in out[:6]]
    return {1: u1, 2: u2, 3: u3}, {1: v1, 2: v2, 3: v3}


# ----------------------------------------------------------------------------------------- VPICEDYN
def vpicedyn(nx1, ny1, usi0, vsi0, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, sinwat, coswat,
             bydts, osurf_tilt, pgfub, pgfvb, max_kki=20, conv_tol=25e-6):
    """VPICEDYN with jitted FORM/RELAX and the convergence loop in Python. Returns
    (UICE1, VICE1, kki, last_dwatn) as numpy arrays, like the scalar port."""
    assert osurf_tilt == 1
    G = geometry()
    imicdyn = nx1 - 2
    a = jnp.asarray
    z = jnp.zeros((nx1 + 1, ny1 + 1))
    u1 = z.at[2:imicdyn + 2, 1:ny1 + 1].set(a(usi0[0:imicdyn, 0:ny1]))
    v1 = z.at[2:imicdyn + 2, 1:ny1 + 1].set(a(vsi0[0:imicdyn, 0:ny1]))
    u1 = u1.at[1, 1:ny1 + 1].set(usi0[imicdyn - 1, 0:ny1]).at[nx1, 1:ny1 + 1].set(usi0[0, 0:ny1])
    v1 = v1.at[1, 1:ny1 + 1].set(vsi0[imicdyn - 1, 0:ny1]).at[nx1, 1:ny1 + 1].set(vsi0[0, 0:ny1])
    u2, v2, u3, v3 = z, z, z, z
    fargs = tuple(a(x) for x in (gairx, gairy, gwatx, gwaty, heff, area, amass, cor, pgfub, pgfvb))
    amass_j = a(amass)
    jall = slice(1, ny1 + 1)
    rall = (slice(1, nx1 + 1), slice(1, ny1 + 1))

    def wrap(x):
        return x.at[1, jall].set(x[nx1 - 1, jall]).at[nx1, jall].set(x[2, jall])

    def run_form(uu, vv):
        f = _form(G, uu, vv, fargs[0], fargs[1], fargs[2], fargs[3], fargs[4], fargs[5], fargs[6],
                  fargs[7], fargs[8], fargs[9], sinwat, coswat)
        return f

    kki = 0
    usave = vsave = None
    w = G["DXU"][1:nx1 + 1, None] * G["DYU"][None, 1:ny1 + 1]
    sel_all = (amass_j[rall] * G["UVM"][rall]) > 0
    while True:
        kki += 1
        u3, v3 = u1, v1
        uc, vc = u1, v1
        f = run_form(u1, v1)
        u1, u2, u3, v1, v2, v3, _, _ = _relax(G, u1, u2, u3, v1, v2, v3, uc, vc, f["forcex"],
                                              f["forcey"], f["draga"], f["drags"], f["eta"],
                                              f["zeta"], amass_j, bydts)
        u1, v1 = wrap(u1), wrap(v1)
        u1 = u1.at[rall].set(0.5 * (u1[rall] + u2[rall]))
        v1 = v1.at[rall].set(0.5 * (v1[rall] + v2[rall]))
        f = run_form(u1, v1)
        last_dwatn = f["dwatn"]
        u3, v3, uc, vc = u1, v1, u1, v1
        u1, v1 = u2, v2
        u1, u2, u3, v1, v2, v3, _, _ = _relax(G, u1, u2, u3, v1, v2, v3, uc, vc, f["forcex"],
                                              f["forcey"], f["draga"], f["drags"], f["eta"],
                                              f["zeta"], amass_j, bydts)
        u1, v1 = wrap(u1), wrap(v1)
        rms, area_tot = 0.0, 0.0
        if kki > 1:
            term = w * ((usave[rall] - u1[rall]) ** 2 + (vsave[rall] - v1[rall]) ** 2)
            rms = float(jnp.sum(jnp.where(sel_all, term, 0.0)))
            area_tot = float(jnp.sum(jnp.where(sel_all, w, 0.0)))
        if kki == max_kki:
            break
        elif kki == 1 or rms > conv_tol * area_tot:
            usave, vsave = u1, v1
        else:
            break
    return np.asarray(u1), np.asarray(v1), kki, np.asarray(last_dwatn)
