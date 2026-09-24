"""Velocity-grid (B-grid) part of ATURB `atm_diffus` + `recalc_agrid_uv` -- Track B.

Faithful port of the U,V diffusion in ATURB.f (non-SCM branch), ATMDYN.f
`regrid_atov_1d`, `get_regrid_info_for_n`, `recalc_agrid_uv`, and the geometry
of GEOM_B.f that these use (area-ratio weights rapvs/rapvn, polar rotation
cos/sin). Level-last (J, I, L) arrays; 0-based rows: Fortran velocity rows
J=2..JM are indices 1..JM-1.

Requires `aturb_ff` (float64 / x64).
"""
import numpy as np
import jax
import jax.numpy as jnp
import aturb_ff as A


def geometry(im=72, jm=46):
    """GEOM_B.f area ratios and polar rotation factors. The Earth radius and
    dlon cancel in these ratios, so it is not needed. Returns dict of numpy arrays."""
    twopi = 2.0 * np.pi
    dlon = twopi / im
    fjeq = 0.5 * (1 + jm)
    radian = np.pi / 180.0
    dlat = (180.0 / (jm - 1)) * radian
    j = np.arange(1, jm + 1)
    dxyp = np.zeros(jm)
    dxyp[0] = dlon * (np.sin(dlat * (1 + .5 - fjeq)) + 1)
    dxyp[jm - 1] = dlon * (1 - np.sin(dlat * (jm - .5 - fjeq)))
    for jj in range(2, jm):
        dxyp[jj - 1] = dlon * (np.sin(dlat * (jj + .5 - fjeq)) - np.sin(dlat * (jj - .5 - fjeq)))
    dxys = np.zeros(jm); dxyn = np.zeros(jm)
    dxys[jm - 1] = dxyp[jm - 1]; dxyn[0] = dxyp[0]
    dxys[1:jm - 1] = .5 * dxyp[1:jm - 1]; dxyn[1:jm - 1] = .5 * dxyp[1:jm - 1]
    rapvs = np.zeros(jm); rapvn = np.zeros(jm); ravps = np.zeros(jm); ravpn = np.zeros(jm)
    for jj in range(2, jm + 1):
        dxyv = dxyn[jj - 2] + dxys[jj - 1]
        rapvs[jj - 1] = .5 * dxys[jj - 1] / dxyv
        rapvn[jj - 2] = .5 * dxyn[jj - 2] / dxyv
    i = np.arange(im)
    siniv = np.sin(i * dlon)
    cosiv = np.cos(i * twopi * (1.0 / im))
    return dict(rapvs=rapvs, rapvn=rapvn, cosiv=cosiv, siniv=siniv, byim=1.0 / im)


def _v_of_a(x, geo, hemi_s=-1.0, hemi_n=1.0):
    """regrid_atov_1d for one 2-D A-grid field pair handled by caller; helper for sums."""
    raise NotImplementedError


def regrid_atov(ua, va, geo):
    """A-grid (J,I) vector -> velocity-grid rows 1..JM-1: returns (uv_u, uv_v), each (JM, I)
    (row 0 unused). ua,va: (J,I); only column 0 of the two polar rows is meaningful."""
    jm, im = ua.shape
    cosiv = jnp.asarray(geo["cosiv"]); siniv = jnp.asarray(geo["siniv"])
    rapvn = jnp.asarray(geo["rapvn"]); rapvs = jnp.asarray(geo["rapvs"])
    ip1 = (jnp.arange(im) + 1) % im
    su = ua + ua[:, ip1]                         # (u_a(i,j)+u_a(ip1,j))
    sv = va + va[:, ip1]
    # polar rows replaced by rotated pole vector (hemi = -1 south, +1 north)
    def pole(row, hemi):
        u1, v1 = ua[row, 0], va[row, 0]
        return (2. * (u1 * cosiv + v1 * siniv * hemi), 2. * (v1 * cosiv - u1 * siniv * hemi))
    us, vs = pole(0, -1.0)
    un, vn = pole(jm - 1, 1.0)
    su = su.at[0].set(us).at[jm - 1].set(un)
    sv = sv.at[0].set(vs).at[jm - 1].set(vn)
    south_u, south_v = su[:-1], sv[:-1]          # row j-1 for velocity row j=1..jm-1
    north_u, north_v = su[1:], sv[1:]            # row j
    wn = rapvn[:-1][:, None]                     # rapvn(j-1)
    ws = rapvs[1:][:, None]                      # rapvs(j)
    uu = wn * south_u + ws * north_u
    vv = wn * south_v + ws * north_v
    pad = jnp.zeros((1, im), ua.dtype)
    return jnp.concatenate([pad, uu]), jnp.concatenate([pad, vv])


def _interp_to_v(x, geo):
    """Sequential 4-neighbour interpolation of an A-grid (J,I,...) field to velocity
    points rows 1..JM-1 (Fortran get_regrid_info_for_n accumulation order)."""
    jm, im = x.shape[:2]
    rapvn = jnp.asarray(geo["rapvn"]); rapvs = jnp.asarray(geo["rapvs"])
    ip1 = (jnp.arange(im) + 1) % im
    sh = (1,) * (x.ndim - 1)
    w_n = rapvn[:-1].reshape((jm - 1,) + sh)
    w_s = rapvs[1:].reshape((jm - 1,) + sh)
    s_i, s_ip = x[:-1], x[:-1][:, ip1]
    n_i, n_ip = x[1:], x[1:][:, ip1]
    acc = w_n * s_i
    acc = acc + w_n * s_ip
    acc = acc + w_s * n_i
    acc = acc + w_s * n_ip
    return acc                                   # (JM-1, I, ...)


def _fill_poles(x):
    """Copy column i=0 of the two polar rows across i (ATURB 'fill poles')."""
    x = x.at[0].set(jnp.broadcast_to(x[0, :1], x[0].shape))
    x = x.at[-1].set(jnp.broadcast_to(x[-1, :1], x[-1].shape))
    return x


def _column_uv(uv0, km, uw_nl, rho, rhoe, dz, dze, flux_surf, dtime):
    bydzerho = 1.0 / (dze * rho)
    rhoebydz = jnp.concatenate([jnp.zeros(1, uv0.dtype), rhoe[1:] / dz[:-1]])
    flux_bot = rhoe[0] * flux_surf + rhoe[1] * uw_nl[1]
    p4 = jnp.zeros_like(uv0).at[1:-1].set(
        -(rhoe[2:] * uw_nl[2:] - rhoe[1:-1] * uw_nl[1:-1]) * bydzerho[1:-1])
    return A.de_solver_main(uv0, km, p4, rhoebydz, bydzerho, flux_bot, 0.0, dtime, False)


def diffuse_uv(U, V, uflxa, vflxa, km, uw_nl, vw_nl, rho, rhoe, dz, dze, dtime, geo):
    """U,V: B-grid (J,I,L) at atmosphere entry. km.. (J,I,L) from aturb_grid. Returns new U,V."""
    km, uw_nl, vw_nl, rho, rhoe, dz, dze = (_fill_poles(x) for x in (km, uw_nl, vw_nl, rho, rhoe, dz, dze))
    fu, fv = regrid_atov(uflxa, vflxa, geo)
    kmv, uwv, vwv, rhov, rhoev, dzv, dzev = (_interp_to_v(x, geo) for x in (km, uw_nl, vw_nl, rho, rhoe, dz, dze))
    col = jax.vmap(jax.vmap(_column_uv, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None)), in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))
    un = col(U[1:], kmv, uwv, rhov, rhoev, dzv, dzev, fu[1:], dtime)
    vn = col(V[1:], kmv, vwv, rhov, rhoev, dzv, dzev, fv[1:], dtime)
    return U.at[1:].set(un), V.at[1:].set(vn)


def recalc_agrid_uv(U, V, geo):
    """B-grid (J,I,L) -> A-grid winds (J,I,L): ATMDYN.f recalc_agrid_uv. Polar rows only
    i=0 is meaningful (others returned equal to i=0)."""
    jm, im, L = U.shape
    cosiv = jnp.asarray(geo["cosiv"])[:, None]; siniv = jnp.asarray(geo["siniv"])[:, None]
    rak_p = geo["byim"]
    im1 = (jnp.arange(im) - 1) % im
    i0 = jnp.arange(im)
    ra = 0.25

    def acc4(X):
        # k=1: (im1, J), k=2: (i, J), k=3: (im1, J+1), k=4: (i, J+1); sequential accumulation
        s = X[:-1][:, im1] * ra
        s = s + X[:-1][:, i0] * ra
        s = s + X[1:][:, im1] * ra
        s = s + X[1:][:, i0] * ra
        return s                                  # rows 0..jm-2 ; valid for interior rows 1..jm-2
    ua = jnp.zeros_like(U).at[:-1].set(acc4(U))
    va = jnp.zeros_like(V).at[:-1].set(acc4(V))

    def pole(row_b, hemi):
        u_t = jnp.zeros(L, U.dtype); v_t = jnp.zeros(L, U.dtype)
        for k in range(im):                       # sequential in K like the Fortran
            uk, vk = U[row_b, k], V[row_b, k]
            ck, sk = cosiv[k, 0], siniv[k, 0]
            u_t = u_t + rak_p * (uk * ck - hemi * vk * sk)
            v_t = v_t + rak_p * (vk * ck + hemi * uk * sk)
        return u_t, v_t
    us, vs = pole(1, -1.0)                        # south pole uses velocity row J=2 (index 1)
    un, vn = pole(jm - 1, 1.0)                    # north pole uses row JM
    ua = ua.at[0].set(jnp.broadcast_to(us, (im, L))).at[jm - 1].set(jnp.broadcast_to(un, (im, L)))
    va = va.at[0].set(jnp.broadcast_to(vs, (im, L))).at[jm - 1].set(jnp.broadcast_to(vn, (im, L)))
    return ua, va
