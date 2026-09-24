"""Full-fidelity JAX port of ModelE PBL.f `advanc` (multi-sublayer surface-layer solve) -- Track B.

Faithful transcription (float64) of modelE2_planet_2.0/model/PBL.f as compiled for
P2SAoM40 (no tracers, no SCM, xdelt=0, PBL_USES_GCM_TENDENCIES and SNOW_SKIN_TEMP
undefined), plus the PBL_DRV.f `pbl` post-processing (psi). The routine integrates
prognostic wind/T/q/TKE profiles over npbl=8 sublayers between the surface and the
middle of GCM layer 1 with a <=5-iteration fixed-point on ustar, using a Newton-mapped
log-linear grid (griddr/NewtonMethod). It is what produces the surface fluxes and drag
coefficients for every tile (ocean, sea ice, land ice, land).

Public: `advanc(rec_in)` for one call (dict of scalars / 8-vectors) and `advanc_batch`
for N calls (vmapped). Dump-record layout: see instrumentation (ffp_*.bin, 154 doubles).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import aturb_ff as A          # enables x64; shares SOCPBL closure constants

N = 8                          # npbl
KAPPA, ZGS = A.KAPPA, A.ZGS
GRAV, RGAS, DELTX, TEENY = A.GRAV, A.RGAS, A.DELTX, A.TEENY
BYGRAV = 1.0 / GRAV
SHA, SHV, LHE, LHS = A.SHA, 0.0, 2.5e6, 2.834e6
STBO, TF = 5.67037321e-8, 273.15
RHOWS = 1030.0
BYRHOWS = 1.0 / RHOWS
VISC_WTR_KIN = 1.05e-6
MRAT, RVAP = 0.621946798777856413, 461.532611712461858
TWOBY3, BY3 = 2.0 / 3.0, 1.0 / 3.0
SIGMA, SIGMA1 = 0.95, 1. - 0.95
GAMAMU, GAMAHU, GAMAMS, GAMAHS = 19.0, 11.6, 5.3, 8.0 / 0.95
ZET1, SLOPE1, ZETM, ZETH = 0.5, 0.1, -1.464, -1.072
SMAX, SMIN = 0.25, 0.005
CMAX, CMIN = SMAX * SMAX, SMIN * SMIN
USTAR_MIN = A.USTAR_MIN
LMONIN_MIN, LMONIN_MAX = A.LMONIN_MIN, A.LMONIN_MAX
K_MAX, KMMIN, KHMIN, KQMIN, KEMIN, EMIN, EMAX = 500.0, 1.5e-5, 2.5e-5, 2.5e-5, 1.5e-5, 1e-6, 1.0e5
SE = 0.1
TOL, WBLEND, ITMAX = 1e-3, 0.5, 5
B1, S0, S1, S2, S4, S5, S6, S7, S8 = A.B1, A.S0, A.S1, A.S2, A.S4, A.S5, A.S6, A.S7, A.S8
# b123=b1**(2./3.) in PBL.f: the exponent is a REAL*4 literal expression
B123 = A.B1 ** float(np.float32(2.) / np.float32(3.))
D1, D2, D3, D4, D5, GHMIN, GHMAX = A.D1, A.D2, A.D3, A.D4, A.D5, A.GHMIN, A.GHMAX
_QA = 6.108 * MRAT
_QB = 1.0 / (RVAP * TF)
_QC = 1.0 / RVAP


def _sign(a, b):
    return jnp.where(b < 0, -jnp.abs(a), jnp.abs(a))


def qsat(tm, lh, pr):
    return _QA * jnp.exp(lh * (_QB - _QC / jnp.maximum(130.0, tm))) / pr


def visc_air_kin(t):
    tc = t - TF
    return 1.326e-5 * (1. + tc * (6.542e-3 + tc * (8.301e-6 - 4.84e-9 * tc)))


def tfrez(sss):
    return (-.0575 + (-2.154996e-4) * sss) * sss + 1.710523e-3 * sss * jnp.sqrt(sss)


def delta_sst(qnet, qsol, ustar_oc):
    byk, lam = 1.677, 2.4
    dl = lam * VISC_WTR_KIN / jnp.maximum(ustar_oc, 0.00098)
    fc = jnp.where(dl < 1e-8, 0.0545 + 11. * dl,
                   0.137 + 11. * dl - 6.6e-5 * (1. - jnp.exp(-dl * 1250.)) / dl)
    fc = jnp.minimum(0.24, fc)
    return (qnet + fc * qsol) * dl * byk


# ------------------------------------------------ similarity functions (float64)
def find_dpsim(zet, zet0):
    s_lo = -GAMAMS * (zet - zet0)
    s_hi = (-GAMAMS * (ZET1 - zet0) + ZET1 * (SLOPE1 - GAMAMS) * jnp.log(jnp.maximum(zet, 1e-300) / ZET1)
            - SLOPE1 * (zet - ZET1))
    stable = jnp.where(zet <= ZET1, s_lo, s_hi)
    zu = jnp.minimum(zet, 0.0)
    x = (1. - GAMAMU * zu) ** 0.25
    x0 = (1. - GAMAMU * jnp.minimum(zet0, 0.0)) ** 0.25
    xm = (1. - GAMAMU * ZETM) ** 0.25
    u1 = (jnp.log((1 + x) * (1 + x) * (1 + x * x) / ((1 + x0) * (1 + x0) * (1 + x0 * x0)))
          - 2. * (jnp.arctan(x) - jnp.arctan(x0)))
    zz = jnp.minimum(zu, -1e-300)
    u2 = (jnp.log((1 + xm) * (1 + xm) * (1 + xm * xm) / ((1 + x0) * (1 + x0) * (1 + x0 * x0)))
          - 2. * (jnp.arctan(xm) - jnp.arctan(x0)) + jnp.log(zz / ZETM)
          - 1.140125 * ((-zz) ** BY3 - (-ZETM) ** BY3))
    unstable = jnp.where(zet > ZETM, u1, u2)
    return jnp.where(zet >= 0.0, stable, unstable)


def find_dpsih(zet, zet0, z, z0):
    lz = jnp.log(z / z0)
    s_lo = SIGMA1 * lz - SIGMA * GAMAHS * (zet - zet0)
    s_hi = (SIGMA1 * jnp.log(ZET1 / zet0) - SIGMA * GAMAHS * (ZET1 - zet0)
            + (1 + SIGMA * (ZET1 * (SLOPE1 - GAMAHS) - 1)) * jnp.log(jnp.maximum(zet, 1e-300) / ZET1)
            - SIGMA * SLOPE1 * (zet - ZET1))
    stable = jnp.where(zet <= ZET1, s_lo, s_hi)
    zu = jnp.minimum(zet, 0.0)
    x = (1. - GAMAHU * zu) ** 0.5
    x0 = (1. - GAMAHU * jnp.minimum(zet0, 0.0)) ** 0.5
    xh = (1. - GAMAHU * ZETH) ** 0.5
    rat_small = z0 / z * (1 - .5 * GAMAHU * (zu - zet0))
    xs = jnp.where(x == 1.0, 0.5, x)
    rat_big = (1 + x) * (1 - x0) / ((1 - xs) * (1 + x0))
    rat = jnp.where(-GAMAHU * zu < 1e-5, rat_small, rat_big)
    u1 = lz + SIGMA * jnp.log(rat)
    zz = jnp.minimum(zu, -1e-300)
    u2 = (lz + SIGMA * jnp.log((1 + xh) * (1 - x0) / ((1 - xh) * (1 + x0)))
          - 0.7957508 * ((-ZETH) ** (-BY3) - (-zz) ** (-BY3)))
    unstable = jnp.where(zet > ZETH, u1, u2)
    return jnp.where(zet >= 0.0, stable, unstable)


def getcm(z, z0, lmonin):
    dpsim = find_dpsim(z / lmonin, z0 / lmonin)
    dm = jnp.maximum(jnp.log(z / z0) - dpsim, 1e-3)
    cm = KAPPA * KAPPA / (dm * dm)
    cm = jnp.where(cm > CMAX, CMAX, cm)
    cm = jnp.where(cm < CMIN, CMIN, cm)
    return dm, cm


def getchq(z, z0, lmonin, dm):
    dpsih = find_dpsih(z / lmonin, z0 / lmonin, z, z0)
    dh = jnp.maximum(jnp.log(z / z0) - dpsih, 1e-3)
    ch = KAPPA * KAPPA / (dm * dh)
    ch = jnp.where(ch > CMAX, CMAX, ch)
    ch = jnp.where(ch < CMIN, CMIN, ch)
    return ch


def getzhq(ustar, z0m, scpr, nu, z0min):
    fac_smooth = 30. * jnp.exp(-13.6 * KAPPA * scpr ** TWOBY3)
    smooth = nu * fac_smooth / ustar + z0min
    fac_rough = -7.3 * KAPPA * jnp.sqrt(scpr)
    r0q = jnp.sqrt(jnp.sqrt(ustar * z0m / nu))
    rough = 7.4 * z0m * jnp.exp(fac_rough * r0q)
    return jnp.where(ustar <= 0.20, smooth, rough)


def dflux(lmonin, ustar0, ts, z0m_in, itype, zgs):
    """Returns cm,ch,cq,dm,z0m,z0h,z0q."""
    nu = visc_air_kin(ts)
    ustar = jnp.maximum(ustar0, 1.0125e-5)
    z0m_w = 0.135 * nu / ustar + 0.018 * ustar * ustar * BYGRAV
    z0h_w = getzhq(ustar, z0m_w, 0.71, nu, 1.4e-5)
    z0q_w = getzhq(ustar, z0m_w, 0.595, nu, 1.3e-4)
    water = (itype == 1) | (itype == 2)
    z0m = jnp.where(water, z0m_w, z0m_in)
    z0h = jnp.where(water, z0h_w, z0m_in * .13533528)
    z0q = jnp.where(water, z0q_w, z0m_in * .13533528)
    dm, cm = getcm(zgs, z0m, lmonin)
    ch = getchq(zgs, z0h, lmonin, dm)
    cq = getchq(zgs, z0q, lmonin, dm)
    return cm, ch, cq, dm, z0m, z0h, z0q


# --------------------------------------------------------------- tridiagonal --
tridiag = A.tridiag


# ---------------------------------------------------------------- grid (griddr)
def _newton(fg, a, b, accuracy):
    fa, _ = fg(a)
    fb, _ = fg(b)
    xa0 = jnp.where(fa < 0.0, a, b)
    xb0 = jnp.where(fa < 0.0, b, a)
    root0 = .5 * (a + b)
    dxold0 = jnp.abs(b - a)
    f0, df0 = fg(root0)

    def body(_, st):
        root, dx, dxold, xa, xb, f, df, done = st
        bis = (((root - xb) * df - f) * ((root - xa) * df - f) > 0.0) | (jnp.abs(2.0 * f) > jnp.abs(dxold * df))
        dx_b = .5 * (xb - xa)
        root_b = xa + dx_b
        dx_n = f / df
        root_n = root - dx_n
        dx_new = jnp.where(bis, dx_b, dx_n)
        root_new = jnp.where(bis, root_b, root_n)
        ret1 = jnp.where(bis, xa == root_b, root == root_n)
        ret2 = (~ret1) & (jnp.abs(dx_new) < accuracy)
        fin = ret1 | ret2
        f_new, df_new = fg(root_new)
        xa_new = jnp.where(f_new < 0.0, root_new, xa)
        xb_new = jnp.where(f_new < 0.0, xb, root_new)
        take = ~done
        upd = lambda new, old: jnp.where(take, new, old)
        return (upd(root_new, root), upd(dx_new, dx), upd(dx, dxold),
                jnp.where(take & ~fin, xa_new, xa), jnp.where(take & ~fin, xb_new, xb),
                jnp.where(take & ~fin, f_new, f), jnp.where(take & ~fin, df_new, df), done | fin)

    st = (root0, dxold0, dxold0, xa0, xb0, f0, df0, jnp.zeros((), bool))
    # early-exit loop (data dependent, <=100 iterations like NewtonMethod's maxNumIterations)
    st = lax.while_loop(lambda c: (~c[1][7]) & (c[0] < 100), lambda c: (c[0] + 1, body(c[0], c[1])), (0, st))[1]
    return st[0]


def griddr(z1, zn):
    """Returns z(8), zhat(7), dz(8), dzh(7)."""
    n = N
    fn1 = float(n - 1)
    byzs, bydzs = 1.0 / 10.0, 1.0 / 4.7914
    dxi = (zn - z1) / fn1
    bgrid = jnp.maximum((dxi * bydzs - 1.) / ((zn - z1) * byzs - jnp.log(zn / z1)), 0.0)
    lznbyz1 = jnp.log(zn / z1)

    def fg_for(xi):
        def fg(z):
            f = z + bgrid * ((zn - z1) * jnp.log(z / z1) - (z - z1) * lznbyz1) - xi
            df = 1. + bgrid * ((zn - z1) / z - lznbyz1)
            return f, df
        return fg

    idx = np.arange(1, n)                                  # Fortran i = 1..n-1
    xi = jnp.stack([z1 + (zn - z1) * float(i - 1) / fn1 for i in idx])
    xihat = jnp.stack([z1 + (zn - z1) * (float(i) - 0.5) / fn1 for i in idx])
    solve = lambda x: _newton(fg_for(x), z1, zn, 1e-3)
    zhat = jax.vmap(solve)(xihat)
    zmain = jax.vmap(solve)(xi[1:])                        # z(2..n-1)
    z = jnp.concatenate([jnp.array([z1]) * jnp.ones(()), zmain, jnp.array([zn]) * jnp.ones(())])
    dxidz = 1. + bgrid * ((zn - z1) / z - lznbyz1)
    dz = dxi / dxidz
    dxidzh = 1. + bgrid * ((zn - z1) / zhat - lznbyz1)
    dzh = dxi / dxidzh
    return z, zhat, dz, dzh


# ------------------------------------------------------ turbulence sub-steps --
def get_tv(t, q):
    return t * (1. + DELTX * q)


def getl1(e, zhat, dzh):
    sum1 = 0.0
    sum2 = 0.0
    for j in range(N - 1):
        sum1 = sum1 + jnp.sqrt(e[j]) * zhat[j] * dzh[j]
        sum2 = sum2 + jnp.sqrt(e[j]) * dzh[j]
    l0 = 0.2 * sum1 / sum2
    l0 = jnp.where(l0 < zhat[0], zhat[0], l0)
    l1 = KAPPA * zhat
    return l0 * l1 / (l0 + l1)


def getl(e, u, v, t, zhat, dzh, lmonin, ustar, dbl):
    l0 = .3 * dbl
    l0 = jnp.where(l0 < zhat[0], zhat[0], l0)
    kz1 = KAPPA * zhat[0]
    ls1 = l0 * kz1 / (l0 + kz1)
    kz = KAPPA * zhat[1:]
    zeta = zhat[1:] / lmonin
    zneg = jnp.minimum(zeta, 0.0)
    ls = jnp.where(zeta >= 1., kz / 3.7,
                   jnp.where(zeta >= 0., kz / (1. + 2.7 * zeta), kz * (1. - 100. * zneg) ** 0.2))
    tt, tn = t[1:N - 1], t[2:N]
    stab = tn > tt
    an2 = 2. * GRAV * (tn - tt) / ((tn + tt) * dzh[1:])
    an = jnp.sqrt(jnp.where(stab, an2, 1.0))
    qturb = jnp.sqrt(2 * e[1:])
    base = jnp.maximum(-KAPPA * lmonin * l0 * l0, 1e-300)
    qty = (ustar / (base ** BY3 * an)) ** 0.5
    lb = jnp.where(stab, jnp.where(zeta >= 0., qturb / an, qturb * (1. + 5. * qty) / an), 1.0e30)
    lsc = l0 * ls * lb / (l0 * ls + l0 * lb + ls * lb)
    return jnp.concatenate([ls1[None], lsc])


def getk(u, v, t, e, lscale, dzh):
    tt, tn = t[:-1], t[1:]
    an2 = 2. * GRAV * (tn - tt) / ((tn + tt) * dzh)
    dudz = (u[1:] - u[:-1]) / dzh
    dvdz = (v[1:] - v[:-1]) / dzh
    as2 = dudz * dudz + dvdz * dvdz
    qturb = jnp.sqrt(2. * e)
    tau = B1 * lscale / jnp.maximum(qturb, TEENY)
    gh = tau * tau * an2
    gm = tau * tau * as2
    gh = jnp.where(gh < GHMIN, GHMIN, gh)
    gh = jnp.where(gh > GHMAX, GHMAX, gh)
    gmmax = (1 + D1 * gh + D3 * gh * gh) / (D2 + D4 * gh)
    gm = jnp.where(gm > gmmax, gmmax, gm)
    den = 1. + D1 * gh + D2 * gm + D3 * gh * gh + D4 * gh * gm + D5 * gm * gm
    sm = (S0 + S1 * gh + S2 * gm) / den
    sh = (S4 + S5 * gh + S6 * gm) / den
    taue = tau * e
    km = jnp.minimum(jnp.maximum(taue * sm, KMMIN), K_MAX)
    kh = jnp.minimum(jnp.maximum(taue * sh, KHMIN), K_MAX)
    kq = jnp.minimum(jnp.maximum(taue * sh, KQMIN), K_MAX)
    ke = jnp.minimum(jnp.maximum(taue * SE, KEMIN), K_MAX)
    return km, kh, kq, ke


def stars(tgrnd, qgrnd, ts, u, v, t, q, z, z0m, cm_in, ch_in, cq_in, km, kh, kq, dzh, itype):
    tgrndv = tgrnd * (1. + DELTX * qgrnd)
    tv = get_tv(t[:2], q[:2])
    dz = dzh[0]
    vel1 = jnp.sqrt(u[0] * u[0] + v[0] * v[0])
    du1, dv1 = u[1] - u[0], v[1] - v[0]
    dudz = jnp.sqrt(du1 * du1 + dv1 * dv1) / dz
    tflx = kh[0] * (t[1] - t[0]) / dz
    qflx = kq[0] * (q[1] - q[0]) / dz
    tvflx = tflx * (1. + DELTX * q[0]) + t[0] * DELTX * qflx
    ustar = jnp.maximum(jnp.sqrt(km[0] * dudz), USTAR_MIN)
    tstar = tflx / ustar
    tstar = jnp.where(jnp.abs(tstar) > SMAX * jnp.abs(t[0] - tgrnd), SMAX * (t[0] - tgrnd), tstar)
    tstar = jnp.where(jnp.abs(tstar) < SMIN * jnp.abs(t[0] - tgrnd), SMIN * (t[0] - tgrnd), tstar)
    qstar = qflx / ustar
    qstar = jnp.where(jnp.abs(qstar) > SMAX * jnp.abs(q[0] - qgrnd), SMAX * (q[0] - qgrnd), qstar)
    qstar = jnp.where(jnp.abs(qstar) < SMIN * jnp.abs(q[0] - qgrnd), SMIN * (q[0] - qgrnd), qstar)
    tstarv = tvflx / ustar
    dtv1 = tv[0] - tgrndv
    tstarv = jnp.where(jnp.abs(tstarv) > SMAX * jnp.abs(dtv1), SMAX * dtv1, tstarv)
    tstarv = jnp.where(jnp.abs(tstarv) < SMIN * jnp.abs(dtv1), SMIN * dtv1, tstarv)
    ustar = jnp.where(ustar > SMAX * vel1, SMAX * vel1, ustar)
    ustar = jnp.where(ustar < SMIN * vel1, SMIN * vel1, ustar)
    tstar = jnp.where(tstar == 0.0, TEENY, tstar)
    tstarv = jnp.where(tstarv == 0.0, TEENY, tstarv)
    ustar = jnp.maximum(ustar, USTAR_MIN)
    lmonin_dry = ustar * ustar * tgrnd / (KAPPA * GRAV * tstar)
    lmonin_dry = jnp.where(jnp.abs(lmonin_dry) < LMONIN_MIN, _sign(LMONIN_MIN, lmonin_dry), lmonin_dry)
    lmonin_dry = jnp.where(jnp.abs(lmonin_dry) > LMONIN_MAX, _sign(LMONIN_MAX, lmonin_dry), lmonin_dry)
    lmonin = ustar * ustar * tgrndv / (KAPPA * GRAV * tstarv)
    lmonin = jnp.where(jnp.abs(lmonin) < LMONIN_MIN, _sign(LMONIN_MIN, lmonin), lmonin)
    lmonin = jnp.where(jnp.abs(lmonin) > LMONIN_MAX, _sign(LMONIN_MAX, lmonin), lmonin)
    cm, ch, cq, dm, z0m_o, z0h, z0q = dflux(lmonin, ustar, ts, z0m, itype, z[0])
    return ustar, tstar, qstar, lmonin, lmonin_dry, cm, ch, cq, z0m_o, z0h, z0q


def e_eqn(esave, e, u, v, t, km, kh, ke, lscale, dz, dzh, ustar, dtime):
    n1 = N - 1
    j = np.arange(1, n1 - 1)                       # 0-based interior j = 1..n-3
    qturb = jnp.sqrt(2. * e[j])
    sub_i = -dtime * 0.5 * (ke[j] + ke[j - 1]) / (dzh[j] * dz[j - 0])
    # Fortran: sub(j)=-dtime*.5*(ke(j)+ke(j-1))/(dzh(j)*dz(j)); sup(j)=... /(dzh(j)*dz(j+1))
    sup_i = -dtime * 0.5 * (ke[j] + ke[j + 1]) / (dzh[j] * dz[j + 1])
    dia_i = 1. - (sub_i + sup_i) + dtime * 2 * qturb / (B1 * lscale[j])
    an2 = 2 * GRAV * (t[j + 1] - t[j]) / ((t[j + 1] + t[j]) * dzh[j])
    dudz = (u[j + 1] - u[j]) / dzh[j]
    dvdz = (v[j + 1] - v[j]) / dzh[j]
    as2 = dudz * dudz + dvdz * dvdz
    rhs_i = esave[j] + dtime * (km[j] * as2 - kh[j] * an2)
    sub = jnp.zeros(n1).at[j].set(sub_i).at[n1 - 1].set(-1.)
    dia = jnp.ones(n1).at[j].set(dia_i)
    sup = jnp.zeros(n1).at[j].set(sup_i)
    rhs = jnp.zeros(n1).at[j].set(rhs_i).at[0].set(0.5 * B123 * ustar * ustar)
    en = tridiag(sub, dia, sup, rhs)
    return jnp.minimum(jnp.maximum(en, TEENY), EMAX)


def q_eqn(q0, kq, dz, dzh, cq, usurf, qgrnd, qtop, dtime, flux_max, fr_sat, gusti, qprime, qdns, ddml):
    i = np.arange(1, N - 1)
    sub_i = -dtime / (dz[i] * dzh[i - 1]) * kq[i - 1]
    sup_i = -dtime / (dz[i] * dzh[i]) * kq[i]
    dia_i = 1. - (sub_i + sup_i)
    factq0 = cq * dzh[0] / kq[0]
    factq = usurf * factq0
    dia1 = 1. + factq
    rhs1 = jnp.where(ddml, factq0 * (usurf * qgrnd - gusti * (qdns - qtop)), factq * qgrnd)
    sub = jnp.zeros(N).at[i].set(sub_i).at[N - 1].set(0.)
    dia = jnp.ones(N).at[i].set(dia_i).at[0].set(dia1)
    sup = jnp.zeros(N).at[i].set(sup_i).at[0].set(-1.)
    rhs = jnp.zeros(N).at[i].set(q0[i]).at[0].set(rhs1).at[N - 1].set(qtop)
    q = tridiag(sub, dia, sup, rhs)
    # over-limit evaporation from unsaturated soil: recompute with flux-limited BC
    need = (fr_sat < 1.) & (cq * usurf * (qgrnd - q[0]) - cq * gusti * qprime > flux_max)
    dia1b = 1. + fr_sat * factq
    rhs1b = jnp.where(ddml,
                      fr_sat * factq0 * (usurf * qgrnd - gusti * (qdns - qtop)) + (1. - fr_sat) * flux_max * dzh[0] / kq[0],
                      fr_sat * factq * qgrnd + (1. - fr_sat) * flux_max * dzh[0] / kq[0])
    q2 = tridiag(sub, dia.at[0].set(dia1b), sup, rhs.at[0].set(rhs1b))
    return jnp.where(need, q2, q)


def t_eqn(u, v, t0, t, z, kh, dz, dzh, ch, usurf, tgrnd, ttop, qtop, dtime,
          dpdxr, dpdyr, dpdxr0, dpdyr0, gusti, tdns, ddml):
    i = np.arange(1, N - 1)
    sub_i = -dtime / (dz[i] * dzh[i - 1]) * kh[i - 1]
    sup_i = -dtime / (dz[i] * dzh[i]) * kh[i]
    dia_i = 1. - (sub_i + sup_i)
    factx = (dpdxr - dpdxr0) / (z[N - 1] - z[0])
    facty = (dpdyr - dpdyr0) / (z[N - 1] - z[0])
    rhs_i = t0[i] - dtime * t[i] * BYGRAV * (v[i] * facty + u[i] * factx)
    facth0 = ch * dzh[0] / kh[0]
    facth = usurf * facth0
    rhs1 = jnp.where(ddml, facth0 * (usurf * tgrnd - gusti * ((1. + 0.0 * qtop) * tdns - ttop)), facth * tgrnd)
    sub = jnp.zeros(N).at[i].set(sub_i).at[N - 1].set(0.)
    dia = jnp.ones(N).at[i].set(dia_i).at[0].set(1 + facth)
    sup = jnp.zeros(N).at[i].set(sup_i).at[0].set(-1.)
    rhs = jnp.zeros(N).at[i].set(rhs_i).at[0].set(rhs1).at[N - 1].set(ttop)
    return tridiag(sub, dia, sup, rhs)


def uv_eqn(u0, v0, u, v, z, km, dz, dzh, cm, utop, vtop, dtime, coriol, uocean, vocean,
           dpdxr, dpdyr, dpdxr0, dpdyr0):
    i = np.arange(1, N - 1)
    sub_i = -dtime / (dz[i] * dzh[i - 1]) * km[i - 1]
    sup_i = -dtime / (dz[i] * dzh[i]) * km[i]
    dia_i = 1. - (sub_i + sup_i)
    factx = (dpdxr - dpdxr0) / (z[N - 1] - z[0])
    facty = (dpdyr - dpdyr0) / (z[N - 1] - z[0])
    dpdx = factx * (z[i] - z[0]) + dpdxr0
    dpdy = facty * (z[i] - z[0]) + dpdyr0
    rhs_u = u0[i] + dtime * (coriol * v[i] - dpdx)
    rhs_v = v0[i] - dtime * (coriol * u[i] + dpdy)
    usurf = jnp.sqrt((u[0] - uocean) ** 2 + (v[0] - vocean) ** 2)
    factor = cm * usurf * dzh[0] / km[0]
    sub = jnp.zeros(N).at[i].set(sub_i)
    dia = jnp.ones(N).at[i].set(dia_i).at[0].set(1. + factor)
    sup = jnp.zeros(N).at[i].set(sup_i).at[0].set(-1.)
    ru = jnp.zeros(N).at[i].set(rhs_u).at[0].set(factor * uocean).at[N - 1].set(utop)
    rv = jnp.zeros(N).at[i].set(rhs_v).at[0].set(factor * vocean).at[N - 1].set(vtop)
    return tridiag(sub, dia, sup, ru), tridiag(sub, dia, sup, rv)


def tfix(t, z, ttop, tgrnd, lmonin_dry, tstar, ustar, khs):
    tn = tgrnd + (ttop - tgrnd) * z[:N - 1] / z[N - 1]
    t_new = t.at[:N - 1].set(tn)
    dtdz = (t_new[1] - t_new[0]) / (z[1] - z[0])
    ts_ = khs * dtdz / ustar
    ts_ = jnp.where(jnp.abs(ts_) > SMAX * jnp.abs(t_new[0] - tgrnd), SMAX * (t_new[0] - tgrnd), ts_)
    ts_ = jnp.where(jnp.abs(ts_) < SMIN * jnp.abs(t_new[0] - tgrnd), SMIN * (t_new[0] - tgrnd), ts_)
    lm = ustar * ustar * tgrnd / (KAPPA * GRAV * ts_)
    lm = jnp.where(jnp.abs(lm) < LMONIN_MIN, _sign(LMONIN_MIN, lm), lm)
    lm = jnp.where(jnp.abs(lm) > LMONIN_MAX, _sign(LMONIN_MAX, lm), lm)
    return t_new, ts_, lm


# ------------------------------------------------------------------ advanc ----
def advanc(d):
    """One PBL call. `d`: dict with scalars dtsurf,tgv,tkv,qg_sat,qg_aver,evap_max,fr_sat,uocean,
    vocean,psurf,trhr0,tg,tr4,elhx,qsol,sss_loc,ocean(0/1),ddml(0/1),gusti,tdns,qdns,snow,dbl,ug,vg,
    cm,ch,cq,coriol,utop,vtop,qtop,ztop,dpdxr,dpdyr,dpdxr0,dpdyr0,z0m,itype and 8-vectors u,v,t,q
    and 7-vector e. Returns dict of outputs."""
    dtime = d["dtsurf"]; tgrnd0 = d["tgv"]; ttop = d["tkv"]; qgrnd_sat = d["qg_sat"]; qgrnd0 = d["qg_aver"]
    evap_max, fr_sat, uocean, vocean = d["evap_max"], d["fr_sat"], d["uocean"], d["vocean"]
    psurf, trhr0, tg, tr4, elhx, qsol, sss_loc = d["psurf"], d["trhr0"], d["tg"], d["tr4"], d["elhx"], d["qsol"], d["sss_loc"]
    ocean = d["ocean"] > 0.5
    ddml = d["ddml"] > 0.5
    gusti, snow, dbl = d["gusti"], d["snow"], d["dbl"]
    itype = d["itype"]
    qtop, utop, vtop, coriol = d["qtop"], d["utop"], d["vtop"], d["coriol"]
    dpdxr, dpdyr, dpdxr0, dpdyr0 = d["dpdxr"], d["dpdyr"], d["dpdxr0"], d["dpdyr0"]
    u, v, t, q, e = d["u"], d["v"], d["t"], d["q"], d["e"]

    z, zhat, dz, dzh = griddr(ZGS, d["ztop"])
    usave, vsave, tsave, qsave, esave = u, v, t, q, e
    usave1, vsave1, tsave1, qsave1, esave1 = u[:N - 1], v[:N - 1], t[:N - 1], q[:N - 1], e
    tgrnd, qgrnd = tgrnd0, qgrnd0
    tgskin, tgr4skin, dskin = tg, tr4, jnp.zeros(())
    ts = t[0]
    lscale = getl1(e, zhat, dzh)
    tdns_ = jnp.where(ddml, d["tdns"], 0.0)
    qdns_ = jnp.where(ddml, d["qdns"], 0.0)
    tprime = jnp.where(ddml, tdns_ - ttop, 0.0)
    qprime = jnp.where(ddml, qdns_ - qtop, 0.0)

    ustar0 = jnp.zeros(())
    lmonin = jnp.ones(())
    ustar = jnp.ones(())
    ws = jnp.zeros(()); ws0 = jnp.zeros(())
    cm, ch, cq = d["cm"], d["ch"], d["cq"]
    z0m, z0h, z0q = d["z0m"], jnp.zeros(()), jnp.zeros(())
    km = kh = kq = jnp.zeros(N - 1)
    done = jnp.zeros((), bool)
    skin_active = ((itype == 1) | ((itype == 2) & (snow > 0)))
    for it in range(1, ITMAX + 1):
        tv = get_tv(t, q)
        if it > 1:
            lscale_n = getl(e, u, v, tv, zhat, dzh, lmonin, ustar, dbl)
            # skin effect for ocean (and snow-covered sea ice: qgrnd/tgr4skin refresh only)
            rhosrf = 100. * psurf / (RGAS * t[0])
            ts_s = t[0]
            qnet = ((LHE + tgskin * SHV) * cq * rhosrf * (ws * (q[0] - qgrnd) + gusti * qprime)
                    + SHA * ch * rhosrf * (ws * (ts_s - tgskin) + gusti * tprime) + trhr0 - STBO * tgr4skin)
            ustar_oc = ustar * jnp.sqrt(rhosrf * BYRHOWS)
            dsk = delta_sst(qnet, qsol, ustar_oc)
            tgs_new = 0.5 * (tgskin + (tg + dsk))
            tgs_new = jnp.maximum(tgs_new, TF + tfrez(sss_loc))
            tgskin_s = jnp.where(itype == 1, tgs_new, tgskin)
            dskin_s = tgskin_s - tg
            tgr4_s = (jnp.sqrt(jnp.sqrt(tr4)) + dskin_s) ** 4
            qg_s = qsat(tgskin_s, elhx, psurf)
            qg_s = jnp.where(ocean, 0.98 * qg_s, qg_s)
            lscale = jnp.where(done, lscale, lscale_n)
            sk = skin_active & ~done
            ts = jnp.where(sk, ts_s, ts)
            tgskin = jnp.where(sk, tgskin_s, tgskin)
            dskin = jnp.where(sk, dskin_s, dskin)
            tgr4skin = jnp.where(sk, tgr4_s, tgr4skin)
            qgrnd = jnp.where(sk, qg_s, qgrnd)
            tgrnd = jnp.where(sk, tgskin_s, tgrnd)
        km_n, kh_n, kq_n, ke_n = getk(u, v, tv, e, lscale, dzh)
        (ustar_n, tstar_n, qstar_n, lmonin_n, lmd_n, cm_n, ch_n, cq_n, z0m_n, z0h_n, z0q_n) = stars(
            tgrnd, qgrnd, ts, u, v, t, q, z, z0m, cm, ch, cq, km_n, kh_n, kq_n, dzh, itype)
        conv = tv[1] < tv[0]
        wstar3 = -dbl * GRAV * 2. * (tv[1] - tv[0]) * kh_n[0] / ((tv[1] + tv[0]) * dzh[0])
        wstar2h = jnp.where(conv, jnp.maximum(wstar3, 0.0) ** TWOBY3, 0.0)
        e_n = e_eqn(esave, e, u, v, tv, km_n, kh_n, ke_n, lscale, dz, dzh, ustar_n, dtime)
        ws02 = (u[0] - uocean) ** 2 + (v[0] - vocean) ** 2 + wstar2h
        ws0_n = jnp.sqrt(ws02)
        ws_n = jnp.sqrt(ws02 + gusti * gusti)
        q_n = q_eqn(qsave, kq_n, dz, dzh, cq_n, ws_n, qgrnd_sat, qtop, dtime, evap_max, fr_sat, gusti, qprime, qdns_, ddml)
        t_n = t_eqn(u, v, tsave, t, z, kh_n, dz, dzh, ch_n, ws_n, tgrnd, ttop, qtop, dtime,
                    dpdxr, dpdyr, dpdxr0, dpdyr0, gusti, tdns_, ddml)
        u_n, v_n = uv_eqn(usave, vsave, u, v, z, km_n, dz, dzh, cm_n, utop, vtop, dtime, coriol, uocean, vocean,
                          dpdxr, dpdyr, dpdxr0, dpdyr0)
        need_fix = ((ttop >= tgrnd) & (lmd_n <= 0.)) | ((ttop <= tgrnd) & (lmd_n >= 0.))
        t_fx, _, _ = tfix(t_n, z, ttop, tgrnd, lmd_n, tstar_n, ustar_n, kh_n[0])
        t_n = jnp.where(need_fix, t_fx, t_n)
        test = jnp.abs(2. * (ustar_n - ustar0) / (ustar_n + ustar0))
        exit_now = test < TOL
        # commit this iteration's results (unless an earlier iteration already exited)
        act = ~done
        sel = lambda new, old: jnp.where(act, new, old)
        km, kh, kq = sel(km_n, km), sel(kh_n, kh), sel(kq_n, kq)
        ustar, lmonin = sel(ustar_n, ustar), sel(lmonin_n, lmonin)
        cm, ch, cq = sel(cm_n, cm), sel(ch_n, ch), sel(cq_n, cq)
        z0m, z0h, z0q = sel(z0m_n, z0m), sel(z0h_n, z0h), sel(z0q_n, z0q)
        ws, ws0 = sel(ws_n, ws), sel(ws0_n, ws0)
        u_c, v_c, t_c, q_c, e_c = sel(u_n, u), sel(v_n, v), sel(t_n, t), sel(q_n, q), sel(e_n, e)
        blend = act & ~exit_now & (it < ITMAX)
        w = WBLEND
        ub = w * usave1 + (1. - w) * u_c[:N - 1]
        vb = w * vsave1 + (1. - w) * v_c[:N - 1]
        tb = w * tsave1 + (1. - w) * t_c[:N - 1]
        qb = w * qsave1 + (1. - w) * q_c[:N - 1]
        eb = w * esave1 + (1. - w) * e_c
        u = jnp.where(blend, u_c.at[:N - 1].set(ub), u_c)
        v = jnp.where(blend, v_c.at[:N - 1].set(vb), v_c)
        t = jnp.where(blend, t_c.at[:N - 1].set(tb), t_c)
        q = jnp.where(blend, q_c.at[:N - 1].set(qb), q_c)
        e = jnp.where(blend, eb, e_c)
        usave1 = jnp.where(blend, ub, usave1); vsave1 = jnp.where(blend, vb, vsave1)
        tsave1 = jnp.where(blend, tb, tsave1); qsave1 = jnp.where(blend, qb, qsave1)
        esave1 = jnp.where(blend, eb, esave1)
        ustar0 = jnp.where(blend, ustar_n, ustar0)
        done = done | (act & exit_now)

    tv = get_tv(t, q)
    us, vs, tsv, qsrf = u[0], v[0], t[0], q[0]
    ufluxs = km[0] * (u[1] - u[0]) / dzh[0]
    vfluxs = km[0] * (v[1] - v[0]) / dzh[0]
    tfluxs = kh[0] * (t[1] - t[0]) / dzh[0]
    qfluxs = kq[0] * (q[1] - q[0]) / dzh[0]
    n = N
    an2 = 2. * GRAV * (tv[n - 1] - tv[n - 2]) / ((tv[n - 1] + tv[n - 2]) * dzh[n - 2])
    dudz = (u[n - 1] - u[n - 2]) / dzh[n - 2]
    dvdz = (v[n - 1] - v[n - 2]) / dzh[n - 2]
    as2 = dudz * dudz + dvdz * dvdz
    tau = B1 * lscale[n - 2] / jnp.maximum(jnp.sqrt(2. * e[n - 2]), TEENY)
    w2_1 = TWOBY3 * e[n - 2] - tau * BY3 * (S7 * km[n - 2] * as2 + S8 * kh[n - 2] * an2)
    w2_1 = jnp.maximum(0.24 * e[n - 2], w2_1)
    psitop = jnp.arctan2(d["vg"], d["ug"] + TEENY)
    psisrf = jnp.arctan2(vs, us + TEENY)
    return dict(us=us, vs=vs, ws=ws, tsv=tsv, qsrf=qsrf, cm=cm, ch=ch, cq=cq, dskin=dskin, ws0=ws0,
                ustar=ustar, lmonin=lmonin, khs=kh[0], kms=km[0], kqs=kq[0], z0m=z0m, z0h=z0h, z0q=z0q,
                w2_1=w2_1, ufluxs=ufluxs, vfluxs=vfluxs, tfluxs=tfluxs, qfluxs=qfluxs, psi=psisrf - psitop,
                u=u, v=v, t=t, q=q, e=e)


def advanc_batch(d):
    """d: dict of arrays with leading batch dim (scalars (B,), profiles (B,8)/(B,7))."""
    return jax.vmap(advanc)(d)


def unpack_records(rec):
    """ffp record array (B,154) -> input dict for advanc_batch (+ raw outputs dict for comparison)."""
    c = lambda k: jnp.asarray(rec[:, k - 1])
    names = {5: "dtsurf", 6: "zs1", 7: "tgv", 8: "tkv", 9: "qg_sat", 10: "qg_aver", 11: "hemi", 12: "tr4",
             13: "evap_max", 14: "fr_sat", 15: "uocean", 16: "vocean", 17: "psurf", 18: "trhr0", 19: "tg",
             20: "elhx", 21: "qsol", 22: "sss_loc", 23: "ocean", 24: "ddml", 25: "gusti", 26: "tdns",
             27: "qdns", 28: "snow", 29: "dskin_in", 30: "dbl", 31: "khs_in", 32: "ug", 33: "vg", 34: "cm",
             35: "ch", 36: "cq", 37: "coriol", 38: "utop", 39: "vtop", 40: "qtop", 41: "ztop", 42: "mdf",
             43: "dpdxr", 44: "dpdyr", 45: "dpdxr0", 46: "dpdyr0", 47: "dtdt_gcm", 50: "z0m", 3: "itype"}
    d = {n: c(k) for k, n in names.items()}
    d["u"] = jnp.asarray(rec[:, 50:58]); d["v"] = jnp.asarray(rec[:, 58:66])
    d["t"] = jnp.asarray(rec[:, 66:74]); d["q"] = jnp.asarray(rec[:, 74:82]); d["e"] = jnp.asarray(rec[:, 82:89])
    return d
