"""Full-fidelity JAX port of ModelE ATURB.f (`atm_diffus` A-grid part) -- Track B.

Faithful transcription of the real Fortran (modelE2_planet_2.0/model/ATURB.f,
PBL.f, solvers/TRIDIAG.f) for the P2SAoM40 configuration, float64 throughout.
NOT derived from the earlier `aturb_jax.py` placeholder.

Scope of this file: the per-column A-grid computation of T, Q, TKE (e), w2, PBL
diagnostics, plus the turbulent-flux fields the velocity-grid U/V solve needs
(km, uw_nl, vw_nl, rho, rhoe, dz, dze). The U/V B-grid diffusion is in
`aturb_uv_ff.py`.

Conventions: per-column vectors are layer index 0..L-1 (Fortran 1..LM). Array
API: (J, I, L) level-last, matching `p2saom40_io` (dumps use (I,J,L)/(L,I,J);
convert with `ffdump_reader`).

float64 is required (the Fortran is REAL*8); this module enables jax x64 at
import -- import it in a process of its own, not alongside float32 Track A code.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------- constants --
# CONSTANT / RESOLUTION values for P2SAoM40 (verified against the running
# model's own values in tests via ffa_consts.txt). GRAV/RGAS/DELTX are the
# planet-config values, NOT the 9.81/287/0.608 approximations of Track A.
GRAV = 9.80664999999999942
RGAS = 287.048730386149032
DELTX = 0.607854565639744715
TEENY = 1.0e-30
SHA = 1002.88097573814161
MB2KG = 10.1971621297792829
BY3 = 1.0 / 3.0
PSF = 984.0
PMTOP = 0.1

# SOCPBL (PBL.f)
KAPPA, ZGS = 0.40, 10.0
SIGMA = 0.95
GAMAMU, GAMAHU, GAMAMS = 19.0, 11.6, 5.3
GAMAHS = 8.0 / SIGMA
ZET1, SLOPE1 = 0.5, 0.1
ZETM, ZETH = -1.464, -1.072
K_MAX, KMMIN, KHMIN, EMIN = 500.0, 1.5e-5, 2.5e-5, 1e-6
EMAX, USTAR_MIN = 1.0e5, 1e-2
LMONIN_MIN, LMONIN_MAX = 1e-6, 1e6


def _ccoeff0():
    """PBL.f ccoeff0, including its REAL*4 int() truncation quirks."""
    prt, b1 = 0.82, 19.3
    g1, g2, g3, g4, g5, g6, g7, g8 = .1070, .0032, .0864, .1000, 11.04, .786, .643, .547
    d1 = (7. * g4 / 3 + g8) / g5
    d2 = (g3 ** 2 - g2 ** 2 / 3.) - 1. / (4. * g5 ** 2) * (g6 ** 2 - g7 ** 2)
    d3 = g4 / (3. * g5 ** 2) * (4. * g4 + 3. * g8)
    d4 = (g4 / (3. * g5 ** 2) * (g2 * g6 - 3. * g3 * g7 - g5 * (g2 ** 2 - g3 ** 2))
          + g8 / g5 * (g3 ** 2 - g2 ** 2 / 3.))
    d5 = -1. / (4. * g5 ** 2) * (g3 ** 2 - g2 ** 2 / 3) * (g6 ** 2 - g7 ** 2)
    s0 = g1 / 2.
    s1 = (-g4 / (3. * g5 ** 2) * (g6 + g7) + 2. * g4 / (3. * g5) * (g1 - g2 / 3. - g3)
          + g1 / (2. * g5) * g8)
    s2 = -g1 / (8. * g5 ** 2) * (g6 ** 2 - g7 ** 2)
    s4 = 2. / (3. * g5)
    s5 = 2. * g4 / (3. * g5 ** 2)
    s6 = (2. / (3. * g5) * (g3 ** 2 - g2 ** 2 / 3) - g1 / (2. * g5) * (g3 - g2 / 3.)
          + g1 / (4 * g5 ** 2) * (g6 - g7))
    s7, s8 = 3. * g3 - g2, 4. * g4
    c1, c2, c3, c4, c5 = s5 + 2 * d3, s1 - s6 - 2 * d4, -s2 + 2. * d5, s4 + 2. * d1, -s0 + 2. * d2
    rimax = (c2 + np.sqrt(c2 ** 2 - 4. * c1 * c3)) / (2. * c1)
    # rimax=int(rimax*1000.)/1000. : integer / REAL*4 literal -> REAL*4 division
    rimax = float(np.float32(int(rimax * 1000.0)) / np.float32(1000.0))
    aa = c1 * rimax * rimax - c2 * rimax + c3
    bb = c4 * rimax + c5
    cc = 2.0
    if abs(aa) < 1e-8:
        gm_at_rimax = -cc / bb
    else:
        gm_at_rimax = (-bb - np.sqrt(bb * bb - 4. * aa * cc)) / (2. * aa)
    dl = (s4 + 2. * d1) ** 2 - 8. * (s5 + 2. * d3)
    ghmin = (-s4 - 2. * d1 + np.sqrt(dl)) / (2. * (s5 + 2. * d3))
    ghmin = float(np.float32(int(ghmin * 10000.0)) / np.float32(10000.0))
    ghmax = (b1 * 0.53) ** 2
    return dict(prt=prt, b1=b1, g5=g5, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, s0=s0, s1=s1, s2=s2,
                s4=s4, s5=s5, s6=s6, s7=s7, s8=s8, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
                rimax=rimax, gm_at_rimax=gm_at_rimax, ghmin=ghmin, ghmax=ghmax)


C = _ccoeff0()
PRT, B1 = C["prt"], C["b1"]
D1, D2, D3, D4, D5 = C["d1"], C["d2"], C["d3"], C["d4"], C["d5"]
S0, S1, S2, S4, S5, S6, S7, S8 = (C[k] for k in ("s0", "s1", "s2", "s4", "s5", "s6", "s7", "s8"))
C1, C2, C3, C4, C5 = (C[k] for k in ("c1", "c2", "c3", "c4", "c5"))
RIMAX, GM_AT_RIMAX, GHMIN, GHMAX = C["rimax"], C["gm_at_rimax"], C["ghmin"], C["ghmax"]


def _sign(a, b):
    """Fortran SIGN(a,b): |a| with the sign of b (b>=0 -> +)."""
    return jnp.where(b < 0, -jnp.abs(a), jnp.abs(a))


# ------------------------------------------------------- PBL.f phi functions --
def find_phim0(zet):
    stable = jnp.where(zet <= ZET1, 1. + GAMAMS * zet,
                       1. + GAMAMS * ZET1 + SLOPE1 * (zet - ZET1))
    zu = jnp.minimum(zet, 0.0)                      # keep dead branch finite
    unstable = (1. - GAMAMU * zu) ** (-.25)
    return jnp.where(zet >= 0.0, stable, unstable)


def find_phih(zet):
    stable = SIGMA * jnp.where(zet <= ZET1, 1. + GAMAHS * zet,
                               1. + GAMAHS * ZET1 + SLOPE1 * (zet - ZET1))
    zu = jnp.minimum(zet, 0.0)
    un1 = SIGMA * (1 - GAMAHU * jnp.maximum(zu, ZETH)) ** (-.5)
    zz = jnp.minimum(zu, -1e-30)
    un2 = .9 * KAPPA ** (4. / 3.) * (-zz) ** (-BY3)
    return jnp.where(zet >= 0.0, stable, jnp.where(zet >= ZETH, un1, un2))


# --------------------------------------------------------------- solvers -----
def tridiag(a, b, c, r):
    """solvers/TRIDIAG.f (Numerical Recipes / Thomas), non-optimized variant."""
    n = a.shape[0]

    def fwd(carry, xs):
        bet, u_prev = carry
        aj, bj, cjm1, rj = xs
        gam = cjm1 / bet
        bet_n = bj - aj * gam
        u = (rj - aj * u_prev) / bet_n
        return (bet_n, u), (u, gam)

    u1 = r[0] / b[0]
    (_, _), (us, gams) = lax.scan(fwd, (b[0], u1), (a[1:], b[1:], c[:-1], r[1:]))
    u = jnp.concatenate([u1[None], us])
    gam = jnp.concatenate([jnp.zeros(1, a.dtype), gams])   # gam[j] for j=1..n-1

    def bwd(u_next, xs):
        uj, gj1 = xs
        un = uj - gj1 * u_next
        return un, un

    _, ub = lax.scan(bwd, u[-1], (u[:-1], gam[1:]), reverse=True)
    return jnp.concatenate([ub, u[-1:]])


def de_solver_main(x0, p1, p4, rhoebydz, bydzerho, flux_bot, flux_top, dtime, qlimit):
    n = x0.shape[0]
    j = jnp.arange(n)
    p1p = jnp.concatenate([p1[1:], p1[-1:]])
    rhp = jnp.concatenate([rhoebydz[1:], rhoebydz[-1:]])
    sub = -dtime * p1 * rhoebydz * bydzerho
    sup = -dtime * p1p * rhp * bydzerho
    dia = 1.0 - (sub + sup)
    rhs = x0 + dtime * p4
    if qlimit:
        rhs = jnp.where(rhs < 0, 0.0, rhs)
    a1 = dtime * p1[1] * rhoebydz[1] * bydzerho[0]
    an = dtime * p1[n - 1] * rhoebydz[n - 1] * bydzerho[n - 1]
    dia = dia.at[0].set(1.0 + a1)
    sup = sup.at[0].set(-a1)
    rhs = rhs.at[0].set(x0[0] - dtime * bydzerho[0] * flux_bot)
    sub = sub.at[n - 1].set(-an)
    dia = dia.at[n - 1].set(1.0 + an)
    rhs = rhs.at[n - 1].set(x0[n - 1] + dtime * bydzerho[n - 1] * flux_top)
    return tridiag(sub, dia, sup, rhs)


# ----------------------------------------------------------------- pieces ----
def getdz(tv, pmid, pedn, pk, tvsurf):
    """ATURB.f getdz for one column. tv,pmid,pk: (L,), pedn: (L+1,).
    Returns dz, dze, rho, rhoe (L,), dz0 scalar."""
    L = tv.shape[0]
    pl, pl1 = pmid[:-1], pmid[1:]
    ple, pl1e = pedn[:-2], pedn[1:-1]
    temp0 = tv[:-1] * pk[:-1]
    temp1 = tv[1:] * pk[1:]
    temp1e = 0.5 * (temp0 + temp1)
    r_g = RGAS / GRAV
    dz_i = -r_g * temp1e * jnp.log(pl1 / pl)
    dze_i = -r_g * temp0 * jnp.log(pl1e / ple)
    rhoe_i = 100.0 * (pl - pl1) / (GRAV * dz_i)          # rhoe(l+1), l=1..L-1
    rho_i = 100.0 * (ple - pl1e) / (GRAV * dze_i)        # rho(l),   l=1..L-1
    dz0 = -r_g * .5 * (temp0[0] + tvsurf) * jnp.log(pl[0] / ple[0])
    rhoe0 = 100.0 * ple[0] / (tvsurf * RGAS)
    plm1e = pedn[L]
    dze_last = -r_g * temp1[-1] * jnp.log(plm1e / pl1e[-1])
    rho_last = 100.0 * (pl1e[-1] - plm1e) / (GRAV * dze_last)
    dz = jnp.concatenate([dz_i, jnp.zeros(1, tv.dtype)])
    dze = jnp.concatenate([dze_i, dze_last[None]])
    rho = jnp.concatenate([rho_i, rho_last[None]])
    rhoe = jnp.concatenate([rhoe0[None], rhoe_i])
    return dz, dze, rho, rhoe, dz0


def zze(dz, dze, dz0):
    """z (L,), ze (L+1,)."""
    ze = ZGS + jnp.concatenate([jnp.zeros(1, dz.dtype), jnp.cumsum(dze)])
    z = ZGS + dz0 + jnp.concatenate([jnp.zeros(1, dz.dtype), jnp.cumsum(dz[:-1])])
    return z, ze


def find_pbl_top(z, u, v, t, ustar, ustar2, tvflx, lmonin, ldbl_max):
    """Bulk-Richardson PBL height. ldbl_max: traced int in [1, L]. Returns dbl, ldbl (1-based)."""
    L = z.shape[0]
    FAC, RI_CR, BB, DBL_MAX = 100., 0.50, 8.5, 4000.
    idx = jnp.arange(L)                     # 0-based l; Fortran l = idx+1
    in_rng = (idx >= 1) & (idx < ldbl_max)  # l = 2..ldbl_max
    v2l = jnp.maximum((u - u[0]) ** 2 + (v - v[0]) ** 2 + FAC * ustar2, TEENY)

    def scan_loop(tref):
        ri = jnp.where(idx == 0, 0.0, (z - z[0]) * GRAV * (t - tref) / (t[0] * v2l))
        ri_prev = jnp.concatenate([jnp.zeros(1, ri.dtype), ri[:-1]])
        hit = in_rng & (ri >= RI_CR)
        any_hit = jnp.any(hit)
        l = jnp.argmax(hit)                 # first True
        return ri, ri_prev, any_hit, l

    def interp(ri, ri_prev, l, den_add):
        den = ri[l] - ri_prev[l]
        return z[l - 1] + (z[l] - z[l - 1]) * (RI_CR - ri_prev[l]) / den, den

    ri, ri_prev, any_hit, l = scan_loop(t[0])
    den = ri[l] - ri_prev[l]
    den = jnp.where(den == 0.0, TEENY, den)
    dbl_hit = z[l - 1] + (z[l] - z[l - 1]) * (RI_CR - ri_prev[l]) / den
    dbl_nohit = z[jnp.maximum(ldbl_max - 1, 0)]          # l==ldbl_max -> dbl=z(ldbl_max)
    dbl = jnp.where(ldbl_max <= 1, z[0], jnp.where(any_hit, dbl_hit, dbl_nohit))

    wtvs = -tvflx
    zet = .1 * dbl / lmonin
    phim = find_phim0(zet)
    wm = ustar / phim
    t1x = t[0] + BB * wtvs / (wm + TEENY)
    ri2, ri2_prev, any_hit2, l2 = scan_loop(t1x)
    den2 = ri2[l2] - ri2_prev[l2] + TEENY
    dbl_hit2 = z[l2 - 1] + (z[l2] - z[l2 - 1]) * (RI_CR - ri2_prev[l2]) / den2
    dbl2 = jnp.where(ldbl_max <= 1, dbl, jnp.where(any_hit2, dbl_hit2, dbl_nohit))
    dbl = jnp.where(wtvs > 0.0, dbl2, dbl)
    dbl = jnp.minimum(dbl, DBL_MAX)

    # ldbl: first l in 2..ldbl_max with dbl>z(l-1) and dbl<=z(l); else ldbl_max; 1 if dbl<=z(1)
    cond = in_rng & (dbl > jnp.concatenate([z[:1], z[:-1]])) & (dbl <= z)
    lf = jnp.where(jnp.any(cond), jnp.argmax(cond) + 1, ldbl_max)
    ldbl = jnp.where(dbl <= z[0], 1, lf)
    return dbl, ldbl


def l_gcm(ze, dbl, lmonin, ustar, qturb, an2):
    """ze: (L+1,) -> uses ze[:L]."""
    FAC, L0MIN = 0.3, 30.0
    z = ze[:-1]
    kz = KAPPA * z
    zeta = z / lmonin
    l0 = .3 * dbl
    ls = jnp.where(zeta >= 1., kz / 3.7,
                   jnp.where(zeta >= 0., kz / (1. + 2.7 * zeta), kz))
    an = jnp.sqrt(jnp.where(an2 > 0., an2, 1.0))
    base = jnp.maximum(-KAPPA * lmonin * l0 * l0, 1e-300)
    qty = (ustar / (base ** BY3 * an)) ** 0.5
    lb_pos = jnp.where(zeta >= 0., qturb / an, qturb * (1. + 5. * qty) / an)
    lb = jnp.where(an2 > 0., lb_pos, 1.0e30)
    l_in = l0 * ls * lb / (l0 * ls + l0 * lb + ls * lb)
    l1 = L0MIN + jnp.maximum(FAC * dbl - L0MIN, 0.0) * jnp.exp(1. - z / dbl)
    l_out = l1 * kz / (l1 + kz)
    return jnp.where(z < dbl, l_in, l_out)


def e_gcm(wstar, ustar, dbl, lmonin, ze, an2, as2, lscale):
    z = ze[:-1]
    kz = KAPPA * z
    ustar3 = ustar ** 3
    wstar3 = wstar ** 3
    zet = z / lmonin
    phim = find_phim0(zet)
    eps = .4 * wstar3 / dbl + ustar3 * (1. - z / dbl) * phim / kz
    ej = .5 * jnp.maximum(19.3 * lscale * eps, 0.0) ** (2. * BY3)
    e_in = jnp.minimum(jnp.maximum(ej, EMIN), EMAX)
    ri = an2 / jnp.maximum(as2, TEENY)
    ri_c = jnp.minimum(ri, RIMAX)
    aa = C1 * ri_c * ri_c - C2 * ri_c + C3
    bb = C4 * ri_c + C5
    cc = 2.0
    disc = jnp.maximum(bb * bb - 4. * aa * cc, 0.0)
    aa_safe = jnp.where(jnp.abs(aa) < 1e-8, 1.0, aa)
    gm_q = (-bb - jnp.sqrt(disc)) / (2. * aa_safe)
    gm = jnp.where(ri < RIMAX, jnp.where(jnp.abs(aa) < 1e-8, -cc / bb, gm_q), GM_AT_RIMAX)
    tmp = 0.5 * (B1 * lscale) ** 2 * as2 / jnp.maximum(gm, TEENY)
    e_out = jnp.minimum(jnp.maximum(tmp, EMIN), EMAX)
    return jnp.where(z <= dbl, e_in, e_out)


def k_gcm(tvflx, ustar, wstar, dbl, lmonin, ze, lscale, e, qturb, an2, as2, dtdz, dqdz, dudz, dvdz):
    """Returns kh,kq,km,ke,wt,wq,w2,uw,vw,wt_nl,wq_nl,uw_nl,vw_nl (each (L,))."""
    ustar2 = ustar * ustar
    wstar3 = wstar ** 3
    z = ze[:-1]
    zet0 = .1 * dbl / lmonin
    unst0 = zet0 < 0.0
    phim = find_phim0(jnp.minimum(zet0, 0.0))
    phih1 = find_phih(jnp.minimum(zet0, 0.0))
    by_phim1 = jnp.where(unst0, 1. / phim, 0.0)
    wm1 = ustar * by_phim1
    pr1 = jnp.where(unst0, phih1 * by_phim1 + .72 * KAPPA * wstar / jnp.where(unst0, wm1, 1.0), 0.0)
    cgh1 = jnp.where(unst0, 7.2 * wstar * (-tvflx) / (jnp.where(unst0, wm1, 1.0) ** 2 * dbl), 0.0)

    kz = KAPPA * z
    tau = B1 * lscale / (qturb + TEENY)
    gh = tau * tau * an2
    gm = tau * tau * as2
    gh = jnp.where(gh < GHMIN, GHMIN, gh)
    gh = jnp.where(gh > GHMAX, GHMAX, gh)
    gmmax = (1. + D1 * gh + D3 * gh * gh) / (D2 + D4 * gh)
    gm = jnp.where(gm > gmmax, gmmax, gm)
    byden = 1. / (1. + D1 * gh + D2 * gm + D3 * gh * gh + D4 * gh * gm + D5 * gm * gm)
    sm = (S0 + S1 * gh + S2 * gm) * byden
    sh = (S4 + S5 * gh + S6 * gm) * byden
    km = jnp.minimum(jnp.maximum(tau * e * sm, KMMIN), K_MAX)
    kh = jnp.minimum(jnp.maximum(tau * e * sh, KHMIN), K_MAX)

    zzi = z / dbl
    zet = z / lmonin
    unst = (zzi <= 1.0) & (zet < 0.0)
    surf = unst & (zzi < 0.1)
    outer = unst & (zzi >= 0.1)
    zet_u = jnp.minimum(zet, 0.0)
    phim_j = find_phim0(zet_u)
    phih_j = find_phih(zet_u)
    by_phim = 1. / phim_j
    wm = ustar * by_phim
    km_s = kz * wm * (1. - zzi) ** 2
    kh_s = km_s / (phih_j * by_phim)
    km_o = kz * wm1 * (1. - zzi) ** 2
    kh_o = km_o / jnp.where(outer, pr1, 1.0)
    km_n = jnp.where(surf, km_s, km_o)
    kh_n = jnp.where(surf, kh_s, kh_o)
    wt_nl = jnp.where(outer, kh_o * cgh1, 0.0)
    zzu = jnp.clip(zzi, 0.0, 1.0)
    tmp_u = (1.6 * ustar2 * (1. - zzu) + TEENY) ** 1.5 + 1.2 * wstar3 * zzu * (1. - .9 * zzu) ** 1.5
    w2j_u = tmp_u ** (2. * BY3)
    w2j_a = BY3 * (2. * e - tau * (S7 * km * as2 + S8 * kh * an2))
    km = jnp.where(unst, jnp.minimum(jnp.maximum(km_n, KMMIN), K_MAX), km)
    kh = jnp.where(unst, jnp.minimum(jnp.maximum(kh_n, KHMIN), K_MAX), kh)
    w2j = jnp.where(unst, w2j_u, w2j_a)
    kq = kh
    wq_nl = jnp.zeros_like(kh)
    uw_nl = jnp.zeros_like(kh)      # cgu1 = cgv1 = 0 (counter-gradient uw,vw turned off)
    vw_nl = jnp.zeros_like(kh)
    ke = 5. * km
    wt = -kh * dtdz + wt_nl
    wq = -kq * dqdz
    w2 = jnp.minimum(jnp.maximum(0.24 * e, w2j), 2. * e)
    uw = -km * dudz + uw_nl
    vw = -km * dvdz + vw_nl
    return kh, kq, km, ke, wt, wq, w2, uw, vw, wt_nl, wq_nl, uw_nl, vw_nl


# ------------------------------------------------------------- column driver --
def aturb_column(t3, q3, ua, va, e_old, pmid, pedn, pk, pek1, pdsig,
                 uflux1, vflux1, tflux1, qflux1, tsavg, qsavg, dtime, ldbl_max):
    """One ATURB column (ATURB.f loop_j_tq body). Inputs are the model's T
    (potential-temperature convention), Q, A-grid winds, TKE, layer pressures/
    exner factors and the four layer-1 fluxes. Returns a dict of results."""
    L = t3.shape[0]
    tvsurf = tsavg * (1.0 + DELTX * qsavg)
    t = t3 * (1.0 + DELTX * q3)                      # virtual potential T
    q = q3
    dz, dze, rho, rhoe, dz0 = getdz(t, pmid, pedn, pk, tvsurf)
    qturb = jnp.sqrt(2.0 * e_old)
    bydzerho = 1.0 / (dze * rho)
    rhoebydz = jnp.concatenate([jnp.zeros(1, t.dtype), rhoe[1:] / dz[:-1]])

    tvs = tvsurf / pek1
    uflx = uflux1 / rhoe[0]
    vflx = vflux1 / rhoe[0]
    qflx = qflux1 / rhoe[0]
    tvflx = tflux1 * (1.0 + DELTX * qsavg) / (rhoe[0] * pek1) + DELTX * tsavg / pek1 * qflx
    uflxa, vflxa = uflx, vflx
    uflx = jnp.where(jnp.abs(uflx) < TEENY, _sign(TEENY, uflx), uflx)
    vflx = jnp.where(jnp.abs(vflx) < TEENY, _sign(TEENY, vflx), vflx)
    ustar = jnp.maximum((uflx * uflx + vflx * vflx) ** 0.25, USTAR_MIN)
    ustar2 = ustar * ustar

    tmp = 1.0 / (ustar * KAPPA * ZGS)
    g_alpha1 = GRAV / tvs
    den = KAPPA * g_alpha1 * tvflx
    den = jnp.where(den == 0.0, TEENY, den)
    lmonin = ustar ** 3 / den
    lmonin = jnp.where(jnp.abs(lmonin) < LMONIN_MIN, _sign(LMONIN_MIN, lmonin), lmonin)
    lmonin = jnp.where(jnp.abs(lmonin) > LMONIN_MAX, _sign(LMONIN_MAX, lmonin), lmonin)

    dudz = jnp.concatenate([(uflx * tmp)[None], (ua[1:] - ua[:-1]) / dz[:-1]])
    dvdz = jnp.concatenate([(vflx * tmp)[None], (va[1:] - va[:-1]) / dz[:-1]])
    dtdz = jnp.concatenate([(tvflx * PRT * tmp)[None], (t[1:] - t[:-1]) / dz[:-1]])
    dqdz = jnp.concatenate([(qflx * PRT * tmp)[None], (q[1:] - q[:-1]) / dz[:-1]])
    g_alpha = jnp.concatenate([g_alpha1[None], GRAV * 2.0 / (t[1:] + t[:-1])])
    an2 = g_alpha * dtdz
    an2 = jnp.where(jnp.abs(an2) < TEENY, _sign(TEENY, an2), an2)
    as2 = jnp.maximum(dudz * dudz + dvdz * dvdz, TEENY)

    z, ze = zze(dz, dze, dz0)
    dbl, ldbl = find_pbl_top(z, ua, va, t, ustar, ustar2, tvflx, lmonin, ldbl_max)
    wstar = jnp.where(tvflx < 0.0, (jnp.maximum(-g_alpha1 * tvflx * dbl, 0.0)) ** BY3, TEENY)

    lscale = l_gcm(ze, dbl, lmonin, ustar, qturb, an2)
    e = e_gcm(wstar, ustar, dbl, lmonin, ze, an2, as2, lscale)
    (kh, kq, km, ke, wt, wq, w2, uw, vw, wt_nl, wq_nl, uw_nl, vw_nl) = k_gcm(
        tvflx, ustar, wstar, dbl, lmonin, ze, lscale, e, qturb, an2, as2, dtdz, dqdz, dudz, dvdz)

    # --- T equation
    p4 = jnp.zeros_like(t).at[1:-1].set(
        -(rhoe[2:] * wt_nl[2:] - rhoe[1:-1] * wt_nl[1:-1]) * bydzerho[1:-1])
    flux_bot = rhoe[0] * tvflx + rhoe[1] * wt_nl[1]
    flux_top = rhoe[L - 1] * wt_nl[L - 1]
    t_new = de_solver_main(t, kh, p4, rhoebydz, bydzerho, flux_bot, flux_top, dtime, False)

    # --- Q equation (with the sequential non-negativity fixes)
    q0 = q
    flux_bot_q = rhoe[0] * qflx + rhoe[1] * wq_nl[1]
    fix1 = (q0[0] - dtime * bydzerho[0] * flux_bot_q) < 0.0
    flux_bot_q = jnp.where(fix1, q0[0] / (dtime * bydzerho[0]), flux_bot_q)
    wq_nl = wq_nl.at[1].set(jnp.where(fix1, (flux_bot_q - rhoe[0] * qflx) / rhoe[1], wq_nl[1]))

    def qbody(w_l, xs):                    # w_l = wq_nl(l) (possibly modified at previous step)
        rhoe_l, rhoe_l1, bydz_l, q0_l, q_l, w_l1_in = xs
        p4l = -(rhoe_l1 * w_l1_in - rhoe_l * w_l) * bydz_l
        bad = (p4l * dtime + q0_l) < 0.0
        p4l = jnp.where(bad, -q_l / dtime, p4l)
        w_next = jnp.where(bad, (q0_l / (dtime * bydz_l) + rhoe_l * w_l) / rhoe_l1, w_l1_in)
        return w_next, (p4l, w_next)

    xs = (rhoe[1:-1], rhoe[2:], bydzerho[1:-1], q0[1:-1], q[1:-1], wq_nl[2:])
    _, (p4q, wq_next) = lax.scan(qbody, wq_nl[1], xs)
    wq_nl = wq_nl.at[2:].set(wq_next)
    p4q = jnp.concatenate([jnp.zeros(1, t.dtype), p4q, jnp.zeros(1, t.dtype)])
    flux_top_q = rhoe[L - 1] * wq_nl[L - 1]
    q_new = de_solver_main(q0, kq, p4q, rhoebydz, bydzerho, flux_bot_q, flux_top_q, dtime, True)

    # --- PBL diagnostics
    li = ldbl - 1                                            # 0-based
    pblptop = jnp.where(ldbl <= 1, pmid[0],
                        pmid[jnp.maximum(li - 1, 0)] + (pmid[li] - pmid[jnp.maximum(li - 1, 0)])
                        * (dbl - z[jnp.maximum(li - 1, 0)]) / (z[li] - z[jnp.maximum(li - 1, 0)]))

    # --- energy correction (tflux1 term uses the incoming actual T)
    tpe0 = -tflux1 * dtime * SHA + jnp.sum(t3 * pk * pdsig * SHA * MB2KG)
    tpe1 = jnp.sum(t_new * pk * pdsig * SHA * MB2KG / (1.0 + DELTX * q_new))
    ediff = (tpe1 - tpe0) / ((PSF - PMTOP) * SHA * MB2KG)
    t_fix = t_new - ediff * (1.0 + DELTX * q_new) / pk
    t3_new = t_fix / (1.0 + DELTX * q_new)
    return dict(t=t3_new, q=q_new, e=e, w2=w2, km=km, uw_nl=uw_nl, vw_nl=vw_nl,
                rho=rho, rhoe=rhoe, dz=dz, dze=dze, pblht=dbl, dclev=ldbl.astype(t.dtype),
                pblptop=pblptop, uflxa=uflxa, vflxa=vflxa, ustar=ustar, lmonin=lmonin,
                tvflx=tvflx)


def aturb_grid(T, Q, UA, VA, E, PMID, PEDN, PK, PEK1, PDSIG, UFLUX1, VFLUX1, TFLUX1, QFLUX1,
               TSAVG, QSAVG, dtime):
    """Vectorised over (J, I) columns. Level arrays are (J, I, L) [PEDN: (J,I,L+1)],
    surface arrays (J, I). Returns dict of (J, I, L) / (J, I) results."""
    L = T.shape[-1]
    ldbl_max = 1 + jnp.sum(PMID[..., 1:] >= 400.0, axis=-1)   # p decreases with height
    f = jax.vmap(jax.vmap(aturb_column, in_axes=(0,) * 16 + (None, 0)), in_axes=(0,) * 16 + (None, 0))
    return f(T, Q, UA, VA, E, PMID, PEDN, PK, PEK1, PDSIG,
             UFLUX1, VFLUX1, TFLUX1, QFLUX1, TSAVG, QSAVG, dtime, ldbl_max)
