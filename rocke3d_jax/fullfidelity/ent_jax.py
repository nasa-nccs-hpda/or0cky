"""D219: JAX (device, batched over land cells) port of the Ent export path of ent_ff.py (D169): clim_stats, photosynth_cond (canopy integration qsimp/trapzd,
Photosynth_analyticsoln, ci_cubic / rtsafe), summarize_patch / summarize_entcell, for the sub-iteration calls of GHY.

ent_ff.py (scalar NumPy mirror of the Fortran, validated D169/D171) is the REFERENCE here; this module is validated against it in tests/test_ent_jax.py.
Same scope as ent_ff: soil_bgc, carbon pools, the daily prescribed update (ent_daily_ff, host, once per day) are not part of this module.

Layout (const = static structure from `pack_cells`, state = what clim_stats advances):
  cells N (one per land cell with an Ent state), patches P (flat), cohorts C (flat); cohort_patch[C], patch_cell[P], cohort_cell[C].
  const: cohort pft, n, lai, h, fracroot[C,6]; patch area[P], albedo[P,6]; has_cohorts[P]; heat_capacity[N]; the PFT tables.
  state: cell airtemp_10d, par_10d, gdd, ncd, daylength[N,2], fall; cohort Sacclim[C].
ent_step(const, state, f, dts, update_day, active) -> (state', exports): f = forcings of ent_set_forcings per cell (already converted: TairC, TcanopyC, Qf,
P_mbar, Ca (mol conc), Ch, U, IPARdif, IPARdir, CosZen, fwet, Soilmoist[N,6], fice[N,6]); dts, update_day, active: per cell.  Cells with active False keep
their state and return exports of the last call (zeros here: the caller masks).
Fortran evaluation order is followed term by term where it can change a rounded result; exp/pow are the XLA backend's (libm mode), so agreement with the
libimf-based reference is rounding-level, not bitwise.  Data-dependent loops: rtsafe is a lax.while_loop with per-element masks (at most MAXIT = 100 iterations,
failure sets a flag), qsimp is the Fortran iteration IT = 1..6 unrolled with a per-cohort convergence mask (a later iteration is skipped by lax.cond when every
cohort has converged).
"""
import jax
import jax.numpy as jnp
import os
import numpy as np

import ent_ff as E

F32 = E.F32
KELVIN, GASC, EPS, EPS2, UNDEF = E.KELVIN, E.GASC, E.EPS, E.EPS2, E.UNDEF
N_DEPTH = E.N_DEPTH

PFPAR_LEAFTYPE = jnp.asarray(E.PFPAR_LEAFTYPE, dtype=jnp.int32)
PFPAR_SSTAR = jnp.asarray(E.PFPAR_SSTAR)
PFPAR_SWILT = jnp.asarray(E.PFPAR_SWILT)
PFPAR_PHENOTYPE = jnp.asarray(E.PFPAR_PHENOTYPE, dtype=jnp.int32)
PFTPAR_VCMAX = jnp.asarray(E.PFTPAR_VCMAX)
PFTPAR_M = jnp.asarray(E.PFTPAR_M)
PFTPAR_B = jnp.asarray(E.PFTPAR_B)


# ----------------------------------------------------------------------------------------------------------- host: packing of the restart cells
def pack_cells(cells, keys):
    """cells: {(i,j): ent_ff.EntCell} (1-based keys); keys: ordered list of the (i,j) of the batch (all must be in `cells`).
    Returns (const, state) of numpy arrays."""
    pat_cell, pat_area, pat_alb, coh_pft, coh_patch, coh_cell = [], [], [], [], [], []
    coh_n, coh_lai, coh_h, coh_fr, coh_sacc = [], [], [], [], []
    for ic, k in enumerate(keys):
        cell = cells.get(k)
        if cell is None:                                   # land cell without an Ent state (fv = 0): placeholder, one patch without cohorts
            cell = E.EntCell()
            cell.heat_capacity = 0.0
            pp = E.Patch()
            pp.area, pp.albedo, pp.cohorts, pp.soil_type = 1.0, np.zeros(E.N_BANDS), [], 1
            cell.patches = [pp]
            cells = dict(cells)
            cells[k] = cell
        for p in cell.patches:
            ip = len(pat_cell)
            pat_cell.append(ic)
            pat_area.append(p.area)
            pat_alb.append(np.asarray(p.albedo, dtype=float))
            for c in p.cohorts:
                coh_pft.append(c.pft)
                coh_patch.append(ip)
                coh_cell.append(ic)
                coh_n.append(c.n)
                coh_lai.append(c.lai)
                coh_h.append(c.h)
                coh_fr.append(np.asarray(c.fracroot, dtype=float))
                coh_sacc.append(c.Sacclim)
    P, C = len(pat_cell), len(coh_pft)
    has = np.zeros(P, bool)
    has[np.asarray(coh_patch, dtype=int)] = True
    const = dict(patch_cell=np.asarray(pat_cell, dtype=np.int32), patch_area=np.asarray(pat_area), patch_albedo=np.asarray(pat_alb),
                 patch_has=has, cohort_patch=np.asarray(coh_patch, dtype=np.int32), cohort_cell=np.asarray(coh_cell, dtype=np.int32),
                 cohort_pft=np.asarray(coh_pft, dtype=np.int32), cohort_n=np.asarray(coh_n), cohort_lai=np.asarray(coh_lai),
                 cohort_h=np.asarray(coh_h), cohort_fracroot=np.asarray(coh_fr), heat_capacity=np.asarray([cells[k].heat_capacity for k in keys]),
                 N=len(keys), P=P, C=C)
    state = dict(airtemp_10d=np.asarray([cells[k].airtemp_10d for k in keys]), par_10d=np.asarray([cells[k].par_10d for k in keys]),
                 gdd=np.asarray([cells[k].gdd for k in keys]), ncd=np.asarray([cells[k].ncd for k in keys]),
                 daylength=np.asarray([cells[k].daylength for k in keys], dtype=float), fall=np.asarray([cells[k].fall for k in keys], dtype=np.int32),
                 Sacclim=np.asarray(coh_sacc))
    return const, state


# ----------------------------------------------------------------------------------------------------------- photosynthesis pieces (vectorised)
def _q10(q10, t):
    return jnp.power(q10, (t - 25.0) / 10.0)


def _frost(sacc):
    tacclim, a_const = -5.93, 0.1
    f = jnp.minimum(a_const * (sacc - tacclim), 1.0)
    return jnp.where(sacc > tacclim, f, jnp.where(sacc == UNDEF, 1.0, 0.01))


def _a_eqn(A, Ra, b, K1, gamol, ca, a1, f1, Rd, want_df):
    K = jnp.where(A > 0.0, K1, 0.0)
    cs = ca - A * Ra
    byAKbcs = 1.0 / (A * K + b * cs)
    ci = cs * (1.0 - A * byAKbcs)
    bycif1 = 1.0 / (ci + f1)
    f = A - (a1 * (ci - gamol) * bycif1 - Rd)
    if not want_df:
        return f, jnp.zeros_like(f)
    dci = -Ra * (1.0 - A * byAKbcs) + cs * (-byAKbcs + A * byAKbcs * byAKbcs * (K - b * Ra))
    df = 1 - a1 * (f1 + gamol) * bycif1 * bycif1 * dci
    return f, df


def _rtsafe(x1, x2, xacc, Ra, b, K, gamol, ca, a1, f1, Rd):
    """ent_ff._rtsafe element-wise: returns (root, failed).  All arguments broadcast to one shape."""
    MAXIT = 100
    shape = jnp.broadcast_shapes(*(jnp.shape(v) for v in (x1, x2, xacc, Ra, b, K, gamol, ca, a1, f1, Rd)))
    bc = lambda v: jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), shape)    # noqa: E731
    x1, x2, xacc, Ra, b, K, gamol, ca, a1, f1, Rd = (bc(v) for v in (x1, x2, xacc, Ra, b, K, gamol, ca, a1, f1, Rd))
    args = (Ra, b, K, gamol, ca, a1, f1, Rd)
    fl, _ = _a_eqn(x1, *args, False)
    fh, _ = _a_eqn(x2, *args, False)
    nosign = ((fl > 0.0) & (fh > 0.0)) | ((fl < 0.0) & (fh < 0.0))
    z1 = fl == 0.0
    z2 = (fh == 0.0) & ~z1
    xl = jnp.where(fl < 0.0, x1, x2)
    xh = jnp.where(fl < 0.0, x2, x1)
    rts = x1
    dxold = jnp.abs(x2 - x1)
    dx = dxold
    f, df = _a_eqn(rts, *args, True)
    result = jnp.where(nosign, -1.0e30, jnp.where(z1, x1, jnp.where(z2, x2, 0.0)))
    done = nosign | z1 | z2

    def cond(c):
        return (c['j'] < MAXIT) & jnp.any(~c['done'])

    def body(c):
        rts, dxold, dx, xl, xh, f, df, done, result = (c[k] for k in ('rts', 'dxold', 'dx', 'xl', 'xh', 'f', 'df', 'done', 'result'))
        bis = ((((rts - xh) * df - f) * ((rts - xl) * df - f)) >= 0.0) | (jnp.abs(2.0 * f) > jnp.abs(dxold * df))
        dx_b = 0.5 * (xh - xl)
        rts_b = xl + dx_b
        dx_n = f / df
        rts_n = rts - dx_n
        dxold_new = dx
        dx_new = jnp.where(bis, dx_b, dx_n)
        rts_new = jnp.where(bis, rts_b, rts_n)
        stuck = jnp.where(bis, xl == rts_b, rts == rts_n)
        conv = stuck | (jnp.abs(dx_new) < xacc)
        fn, dfn = _a_eqn(rts_new, *args, True)
        xl_new = jnp.where(fn < 0.0, rts_new, xl)
        xh_new = jnp.where(fn < 0.0, xh, rts_new)
        act = ~done
        u = lambda new, old: jnp.where(act, new, old)       # noqa: E731
        return dict(rts=u(rts_new, rts), dxold=u(dxold_new, dxold), dx=u(dx_new, dx), xl=u(jnp.where(conv, xl, xl_new), xl), xh=u(jnp.where(conv, xh, xh_new), xh),
                    f=u(fn, f), df=u(dfn, df), done=done | (act & conv), result=jnp.where(act & conv, rts_new, result), j=c['j'] + 1)

    import while_unroll as _WU      # ROCKE_WHILE_UNROLL (default 1: the plain lax.while_loop)
    c = _WU.while_loop(cond, body, dict(rts=rts, dxold=dxold, dx=dx, xl=xl, xh=xh, f=f, df=df, done=done, result=result, j=jnp.zeros((), dtype=jnp.int32)))
    return c['result'], ~c['done']


class _Leaf:
    """per-cohort constants of the leaf photosynthesis (the SAVEd state of ci_cubic / Photosynth_analyticsoln, valid for one cohort call)"""


def _leaf_setup(Vcmax, Kc, Ko, Gam, m, b, ca, rh, gb, Pa):
    L = _Leaf()
    L.Vcmax, L.Gam, L.m, L.b, L.ca, L.rh, L.gb, L.Pa = Vcmax, Gam, m, b, ca, rh, gb, Pa
    O2pres = 20900.0
    S_ATM, S_STOM = 1.37, 1.65
    L.Rd = 0.015 * Vcmax
    L.Ra = 1 / gb * S_ATM
    L.bb = b / S_STOM
    L.K = m * rh / S_STOM
    L.gamol = Gam * 1.0e06 / Pa
    L.A_d_asymp = -L.bb * ca / (L.K - L.bb * L.Ra)
    L.x1 = -L.Rd
    x2 = ca / L.Ra
    t = L.bb * ca / (1.0 - L.K + L.bb * L.Ra)
    x2 = jnp.where(t > 0.0, jnp.minimum(x2, t), x2)
    t = L.A_d_asymp
    x2 = jnp.where(t > 0.0, jnp.minimum(x2, t), x2)
    L.x2save = x2 - .0000001
    L.xacc = .0001
    L.a1c = Vcmax
    L.f1c = Kc * (1.0 + O2pres / Ko) * 1.0e06 / Pa
    L.Ac, L.fail = _ci_cubic(L, L.a1c, L.f1c)
    L.As = Vcmax / 2.0 - L.Rd
    return L


def _ci_cubic(L, a1, f1):
    x2 = jnp.minimum(L.x2save, a1 - L.Rd)
    return _rtsafe(L.x1, x2, L.xacc, L.Ra, L.bb, L.K, L.gamol, L.ca, a1, f1, L.Rd)


def _ball_berry(L, Anet, cs):
    gsw = L.m * Anet * L.rh / cs + L.b
    return jnp.where(gsw < L.b, L.b, gsw)


def _leaf(L, IPAR):
    """Photosynth_analyticsoln for IPAR of shape (C, n) with the per-cohort constants (C,) -> (ci, gs, Atot, flag)"""
    e = lambda v: v[:, None]                              # noqa: E731
    alpha = .08
    Rd = e(L.Rd)
    dark = IPAR < .000001
    a1e = IPAR * alpha
    f1e = 2 * e(L.Gam) * 1.0e06 / e(L.Pa)
    need = ((a1e < e(L.a1c)) | (f1e > e(L.f1c))) & ~dark
    x2 = jnp.minimum(e(L.x2save), a1e - Rd)
    Ae_r, fail = _rtsafe(e(L.x1), x2, L.xacc, e(L.Ra), e(L.bb), e(L.K), e(L.gamol), e(L.ca), a1e, f1e, Rd)
    Ae = jnp.where(need, Ae_r, 1.0e30)
    Anet = jnp.minimum(jnp.minimum(Ae, e(L.Ac)), jnp.minimum(e(L.As), 0.8 * e(L.ca) * e(L.gb) / 1.37))
    Atot = Anet + Rd
    neg = Atot < 0.0
    Atot = jnp.where(neg, 0.0, Atot)
    Anet = jnp.where(neg, -Rd, Anet)
    Atot = jnp.where(dark, 0.0, Atot)
    Anet = jnp.where(dark, -Rd, Anet)
    cs = e(L.ca) - Anet * 1.37 / e(L.gb)
    gs = _ball_berry_e(L, Anet, cs)
    ci = cs - Anet / (gs / 1.65)
    return ci, gs, Atot, jnp.any(fail & need)


def _ball_berry_e(L, Anet, cs):
    e = lambda v: v[:, None]                              # noqa: E731
    gsw = e(L.m) * Anet * e(L.rh) / cs + e(L.b)
    return jnp.where(gsw < e(L.b), e(L.b), gsw)


def _canopy_rad(Lc, crp):
    """Isl, Ish, fsl at cumulative LAI Lc (C, n); crp: dict of per-cohort (C,) scalars"""
    e = lambda v: v[:, None]                              # noqa: E731
    sb = e(crp['CosZen'])
    pos = sb > EPS
    Idfa = (1.0 - e(crp['rhor'])) * e(crp['I0df']) * crp['kdf'] * jnp.exp(-crp['kdf'] * Lc)
    Idra = (1.0 - e(crp['rhor'])) * e(crp['I0dr']) * crp['sqrtexpr'] * e(crp['kbl']) * jnp.exp(-crp['sqrtexpr'] * e(crp['kbl']) * Lc)
    Idrdra = (1.0 - crp['sigma']) * e(crp['I0dr']) * e(crp['kbl']) * jnp.exp(-crp['sqrtexpr'] * e(crp['kbl']) * Lc)
    Isha = Idfa + (Idra - Idrdra)
    Isla = Isha + (1.0 - crp['sigma']) * e(crp['kbl']) * e(crp['I0dr'])
    fsl = jnp.exp(-e(crp['kbl']) * Lc)
    Isha = jnp.where(Isha <= 0.0, 0.0, Isha)
    Isla = jnp.where(Isla <= 0.0, 0.0, Isla)
    fsl = jnp.where(fsl < 0.0, 0.0, fsl)
    z = jnp.zeros_like(Isha)
    return jnp.where(pos, Isla, z), jnp.where(pos, Isha, z), jnp.where(pos, fsl, z)


def _sunshd(L, Lc, crp):
    Isl, Ish, fsl = _canopy_rad(Lc, crp)
    ci1, gs1, A1, f1 = _leaf(L, Isl)
    ci2, gs2, A2, f2 = _leaf(L, Ish)
    Rd = L.Rd[:, None]
    Al = fsl * A1 + (1.0 - fsl) * A2
    gl = fsl * gs1 + (1.0 - fsl) * gs2
    Rl = fsl * Rd + (1.0 - fsl) * Rd
    return Al, gl, Rl, ci2, f1 | f2


_ENT_SKIP = os.environ.get('ROCKE_ENT_SKIP', '0') == '1'


def _qsimp_iter(L, crp, Xlim, Bc, A0, IT, layers, MAXIT, ERRLIM, c):
    """One qsimp iteration IT >= 2 (the body of the loop in _qsimp) as a function of the carried values."""
    (ST, STg, STr, OS, OST, OSTg, OSTr, resS, resSg, resSr, ci_last, active, fail) = c
    Cn = Xlim.shape[0]
    DEL = (Xlim - A0) / float(layers)
    X = A0[:, None] + 0.5 * DEL[:, None] + DEL[:, None] * jnp.arange(layers, dtype=jnp.float64)[None, :]
    Al, gl, Rl, ci, f = _sunshd(L, X, crp)
    sm = smg = smr = jnp.zeros(Cn)
    for k in range(layers):                               # sequential accumulation, as the Fortran loop
        sm, smg, smr = sm + Al[:, k], smg + gl[:, k], smr + Rl[:, k]
    RCL = float(layers)
    ST = 0.5 * (ST + (Xlim - A0) * sm / RCL)
    STg = 0.5 * (STg + (Xlim - A0) * smg / RCL)
    STr = 0.5 * (STr + (Xlim - A0) * smr / RCL)
    ci_it = ci[:, layers - 1]
    fail = fail | f
    S = (4.0 * ST - OST) / 3.0
    Sg = (4.0 * STg - OSTg) / 3.0
    Sr = (4.0 * STr - OSTr) / 3.0
    conv = jnp.abs(S - OS) < ERRLIM
    take = active & (conv | (IT == MAXIT))
    resS, resSg, resSr = (jnp.where(take, v, r) for v, r in ((S, resS), (Sg, resSg), (Sr, resSr)))
    ci_last = jnp.where(take, ci_it, ci_last)
    active = active & ~take
    OS, OST, OSTg, OSTr = S, ST, STg, STr
    return (ST, STg, STr, OS, OST, OSTg, OSTr, resS, resSg, resSr, ci_last, active, fail)


def _qsimp(L, crp, Xlim, Bc):
    """ent_ff.qsimp for all cohorts at once: S, Sg, Sr, Si, ci of the last leaf call, failure flag"""
    MAXIT, ERRLIM = 6, 0.1
    Cn = Xlim.shape[0]
    OST = OS = OSTg = OSTr = OSTi = jnp.full(Cn, -1.0e30)
    ST = STg = STr = STi = jnp.zeros(Cn)
    active = jnp.ones(Cn, dtype=bool)
    resS = resSg = resSr = resSi = jnp.zeros(Cn)
    ci_last = jnp.zeros(Cn)
    fail = jnp.zeros((), dtype=bool)
    layers = 1
    A0 = jnp.zeros(Cn)
    for IT in range(1, MAXIT + 1):
        if IT > 1 and _ENT_SKIP:
            # speed track (ROCKE_ENT_SKIP=1): when every cohort has converged the iteration changes nothing that is kept (take = active & ..., results
            # and ci are only written where `take`), so it is skipped; only the `fail` diagnostic would no longer see the leaf calls of converged cohorts
            def _it(c, IT=IT, layers=layers):
                return _qsimp_iter(L, crp, Xlim, Bc, A0, IT, layers, MAXIT, ERRLIM, c)
            carry = (ST, STg, STr, OS, OST, OSTg, OSTr, resS, resSg, resSr, ci_last, active, fail)
            carry = jax.lax.cond(jnp.any(active), _it, lambda c: c, carry)
            (ST, STg, STr, OS, OST, OSTg, OSTr, resS, resSg, resSr, ci_last, active, fail) = carry
            layers *= 2
            continue
        if IT == 1:
            Lc = jnp.stack([A0, Xlim], axis=1)
            Al, gl, Rl, ci, f = _sunshd(L, Lc, crp)
            ST = 0.5 * (Bc - A0) * (Al[:, 0] + Al[:, 1])
            STg = 0.5 * (Bc - A0) * (gl[:, 0] + gl[:, 1])
            STr = 0.5 * (Bc - A0) * (Rl[:, 0] + Rl[:, 1])
            STi = jnp.zeros(Cn)
            ci_it = ci[:, 1]
            fail = fail | f
        else:
            DEL = (Xlim - A0) / float(layers)
            X = A0[:, None] + 0.5 * DEL[:, None] + DEL[:, None] * jnp.arange(layers, dtype=jnp.float64)[None, :]
            Al, gl, Rl, ci, f = _sunshd(L, X, crp)
            sm = smg = smr = jnp.zeros(Cn)
            for k in range(layers):                               # sequential accumulation, as the Fortran loop
                sm, smg, smr = sm + Al[:, k], smg + gl[:, k], smr + Rl[:, k]
            RCL = float(layers)
            ST = 0.5 * (ST + (Xlim - A0) * sm / RCL)
            STg = 0.5 * (STg + (Xlim - A0) * smg / RCL)
            STr = 0.5 * (STr + (Xlim - A0) * smr / RCL)
            ci_it = ci[:, layers - 1]
            fail = fail | f
        S = (4.0 * ST - OST) / 3.0
        Sg = (4.0 * STg - OSTg) / 3.0
        Sr = (4.0 * STr - OSTr) / 3.0
        conv = jnp.abs(S - OS) < ERRLIM
        take = active & (conv | (IT == MAXIT))
        resS, resSg, resSr = (jnp.where(take, v, r) for v, r in ((S, resS), (Sg, resSg), (Sr, resSr)))
        ci_last = jnp.where(take, ci_it, ci_last)
        active = active & ~take
        OS, OST, OSTg, OSTr = S, ST, STg, STr
        if IT > 1:
            layers *= 2
    return resS, resSg, resSr, ci_last, fail


# ----------------------------------------------------------------------------------------------------------- the Ent step
def _qsat(TM, LH, PR):
    MWAT, MAIR = 18.015, 28.9655
    MRAT = MWAT / MAIR
    RVAP = 1e3 * GASC / MWAT
    A = 6.108 * MRAT
    B = 1.0 / (RVAP * E.TFRZ)
    C = 1.0 / RVAP
    return A * jnp.exp(LH * (B - C / jnp.maximum(130.0, TM))) / PR


def ent_step(const, state, f, dts, update_day, active):
    N, P, C = const['heat_capacity'].shape[0], const['patch_cell'].shape[0], const['cohort_pft'].shape[0]
    cc, cp, pc = const['cohort_cell'], const['cohort_patch'], const['patch_cell']
    pft = const['cohort_pft'] - 1
    # ---- clim_stats
    zw = jnp.exp(-1.0 / (10.0 * 86400.0 / dts))
    air10 = zw * state['airtemp_10d'] + (1.0 - zw) * f['TairC']
    par = f['IPARdif'] + f['IPARdir']
    par10 = zw * state['par_10d'] + (1.0 - zw) * par
    day = f['CosZen'] > 0.0
    dl1 = state['daylength'][:, 1] + jnp.where(day, dts / 60.0, 0.0)
    gdd, ncd, fall = state['gdd'], state['ncd'], state['fall']
    gdd = jnp.where(update_day & (air10 >= 5.0), gdd + (air10 - 5.0), gdd)
    ncd = jnp.where(update_day & (air10 < 5.0), ncd + 1.0, ncd)
    r1, r0 = jnp.round(dl1), jnp.round(state['daylength'][:, 0])
    fall = jnp.where(update_day & (r1 < r0), 1, jnp.where(update_day & (r1 > r0), 0, fall))
    ph, lt = PFPAR_PHENOTYPE[pft], PFPAR_LEAFTYPE[pft]
    acc_mask = ((ph == 1) & (lt == 2)) | (ph == 2) | (ph == 4)
    tau_inv = F32(2.22222e-6)
    sacc_new = jnp.where(acc_mask, state['Sacclim'] + dts[cc] * tau_inv * (air10[cc] - state['Sacclim']), 25.0)
    # ---- cell level quantities of photosynth_cond
    IPAR = f['IPARdir'] + f['IPARdif']
    fdir = jnp.where(f['IPARdir'] == 0.0, 0.0, f['IPARdir'] / jnp.where(IPAR == 0.0, 1.0, IPAR))
    Pa = f['P_mbar'] * 100.0
    Gb = f['Ch'] * f['U'] * Pa / (GASC * (f['TairC'] + KELVIN))
    molc = GASC * (f['TcanopyC'] + KELVIN) / Pa * 1.0e6
    ca_u = f['Ca'] * molc
    TcanK = f['TcanopyC'] + KELVIN
    TsurfK = f['TairC'] + KELVIN
    rh = jnp.minimum(1.0, jnp.maximum(f['Qf'], 0.0) / _qsat(TcanK, 2500800.0 - 2360.0 * (TsurfK - KELVIN), Pa / 100.0))
    lai = const['cohort_lai']
    pLAI = jax.ops.segment_sum(lai, cp, num_segments=P)
    # ---- cohort level
    sm, fice = f['Soilmoist'][cc], f['fice'][cc]
    sstar, swilt = PFPAR_SSTAR[pft][:, None], PFPAR_SWILT[pft][:, None]
    betak = jnp.where(sm >= sstar, 1.0, jnp.where((sm < sstar) & (sm > swilt), (sm - swilt) / (sstar - swilt), 0.0))
    bl = (1.0 - fice) * const['cohort_fracroot'] * betak
    betad = bl[:, 0] + bl[:, 1] + bl[:, 2] + bl[:, 3] + bl[:, 4] + bl[:, 5]
    betad = jnp.where(betad < EPS2, 0.0, betad)
    Pa_c, Tc_c = Pa[cc], f['TcanopyC'][cc]
    facc = _frost(sacc_new)
    Vcmax = PFTPAR_VCMAX[pft] * _q10(2.21, Tc_c) * facc * 1.0
    Kc = E.KC * _q10(E.KCQ10, Tc_c)
    Ko = E.KO * _q10(E.KOQ10, Tc_c)
    O2pres = E.O2FRAC * Pa_c
    Gam = 0.5 * (Kc / Ko) * F32(0.21) * O2pres
    m = betad * PFTPAR_M[pft]
    b = PFTPAR_B[pft]
    L = _leaf_setup(Vcmax, Kc, Ko, Gam, m, b, ca_u[cc], rh[cc], Gb[cc], Pa_c)
    # canopy radiation set-up (per cohort; LAI of the patch, albedo of the patch)
    alb0 = const['patch_albedo'][:, 0][cp]
    cz = f['CosZen'][cc]
    IP = IPAR[cc] * E.SWTOPAR
    sqrtexpr = np.sqrt(1.0 - E.SIGMA)
    kbl = 0.5 * E.KDF / (0.8 * sqrtexpr * cz + EPS)
    rhor = jnp.where(alb0 == 0.0, ((1.0 - sqrtexpr) / (1.0 + sqrtexpr)) * (2.0 / (1.0 + 1.6 * cz)), alb0)
    crp = dict(CosZen=cz, I0dr=fdir[cc] * IP, I0df=(1.0 - fdir[cc]) * IP, sigma=E.SIGMA, kdf=E.KDF, sqrtexpr=sqrtexpr, kbl=kbl, rhor=rhor)
    Xlim = pLAI[cp]
    S, Sg, Sr, ci_last, fail = _qsimp(L, crp, Xlim, Xlim)
    fdry = jnp.where(lt == 2, 1.0 - jnp.minimum(f['fwet'][cc], 0.333), 1.0)
    on = lai > 0.0
    GC = jnp.where(on, Sg * fdry * (GASC * TsurfK[cc]) / Pa_c, 0.0)
    Ci = jnp.where(on, ci_last * Pa_c / (GASC * (Tc_c + KELVIN)), EPS)
    GPP = jnp.where(on, S * fdry * 0.012e-6, 0.0)
    IPP = jnp.zeros(C)                     # Si (the "Iint" of qsimp) is identically 0 in this build (Iel/Ies = 0.0)
    stress_l = jnp.where(on[:, None], bl, 0.0)
    # transmittance per patch
    cz_p, fd_p, IPAR_p = f['CosZen'][pc], fdir[pc], IPAR[pc]
    alb_p = const['patch_albedo'][:, 0]
    rhor_p = jnp.where(alb_p == 0.0, ((1.0 - sqrtexpr) / (1.0 + sqrtexpr)) * (2.0 / (1.0 + 1.6 * cz_p)), alb_p)
    kbl_p = 0.5 * E.KDF / (0.8 * sqrtexpr * cz_p + EPS)
    al = pLAI
    absdf = (1 - fd_p) * (1.0 - rhor_p) * E.KDF * jnp.exp(-E.KDF * al)
    absdr = fd_p * (1.0 - rhor_p) * sqrtexpr * kbl_p * jnp.exp(-sqrtexpr * kbl_p * al)
    absdrdr = absdr * (1.0 - E.SIGMA) * kbl_p * jnp.exp(-sqrtexpr * kbl_p * al)
    abssh = absdf + (absdr - absdrdr)
    abssl = abssh + (1.0 - E.SIGMA) * kbl_p * fd_p
    fracsl = jnp.exp(-kbl_p * al)
    abssh0 = (1.0 - rhor_p) * E.KDF * jnp.exp(-E.KDF * al)
    pos = cz_p > 0.0
    abssh = jnp.where(pos, abssh, abssh0)
    abssl = jnp.where(pos, abssl, 0.0)
    fracsl = jnp.where(pos, fracsl, 0.0)
    abssh = jnp.clip(abssh, 0.0, 1.0)
    abssl = jnp.clip(abssl, 0.0, 1.0)
    trans = (1.0 - fracsl) * abssh + fracsl * abssl
    trans = jnp.where(const['patch_has'], trans, 1.0)
    # ---- summarize_patch / summarize_entcell
    ncoh = const['cohort_n']
    seg = lambda v, idx, n: jax.ops.segment_sum(v, idx, num_segments=n)         # noqa: E731
    nsum = seg(ncoh, cp, P)
    pl_ = pLAI
    ph_ = seg(const['cohort_h'] * ncoh, cp, P)
    ph_ = jnp.where(nsum > 0.0, ph_ / jnp.where(nsum > 0.0, nsum, 1.0), ph_)
    pCi = seg(Ci * lai, cp, P)
    pCi = jnp.where(pl_ > 0.0, pCi / jnp.where(pl_ > 0.0, pl_, 1.0), 0.0)
    pGC, pGPP, pIPP = seg(GC, cp, P), seg(GPP, cp, P), seg(IPP, cp, P)
    pbl = seg(stress_l * lai[:, None], cp, P)
    pbl = jnp.where((pl_ > 0.0)[:, None], pbl / jnp.where(pl_ > 0.0, pl_, 1.0)[:, None], 0.0)
    area = const['patch_area']
    fa = seg(area, pc, N)
    laifa = area * pl_
    laifasum = seg(laifa, pc, N)
    eLAI = seg(pl_ * area, pc, N) / fa
    eCi = (0.0127 + seg(pCi * area, pc, N)) / fa
    eGC = seg(pGC * area, pc, N) / fa
    eGPP = seg(pGPP * area, pc, N) / fa
    eIPP = seg(pIPP * area, pc, N) / fa
    ebl = seg(pbl * area[:, None], pc, N) / fa[:, None]
    eTR = seg(trans * area, pc, N) / fa
    out = dict(cnc=eGC, betadl=ebl, trans_sw=eTR, ci=eCi, gpp=eGPP, lai=eLAI, ipp=eIPP, fail=fail)
    a = active
    st = dict(airtemp_10d=jnp.where(a, air10, state['airtemp_10d']), par_10d=jnp.where(a, par10, state['par_10d']),
              gdd=jnp.where(a, gdd, state['gdd']), ncd=jnp.where(a, ncd, state['ncd']),
              daylength=jnp.where(a[:, None], jnp.stack([state['daylength'][:, 0], dl1], axis=1), state['daylength']),
              fall=jnp.where(a, fall, state['fall']), Sacclim=jnp.where(a[cc], sacc_new, state['Sacclim']))
    return st, out


# ----------------------------------------------------------------------------------------------------------- host side: restart cells, daily update (ROCKE_ENT_LIVE)
REG = {}                    # path of the restart (set_restart), cached host cells, packed batches by key tuple, LAI table


def set_restart(path):
    REG.clear()
    REG['path'] = path


def host_cells():
    if 'cells' not in REG:
        REG['cells'] = E.load_restart_cells(REG['path'])
    return REG['cells']


def to_device(const, state):
    return ({k: jnp.asarray(v) for k, v in const.items() if isinstance(v, np.ndarray)}, {k: jnp.asarray(v) for k, v in state.items()})


def packed(keys):
    """(const, state) of the Ent cells of `keys` (list of 1-based (i, j)) from the restart, as device arrays; memoised"""
    key = tuple(keys)
    pk = REG.setdefault('pk', {})
    if key not in pk:
        c, s = pack_cells(host_cells(), list(keys))
        pk[key] = to_device(c, s)
    return pk[key]


def daily_update(keys, const, state, jday):
    """Day boundary (GHY_DRV daily_earth -> update_vegetation_data): the Ent state is pulled from the device into host EntCells, the prescribed LAI / albedo update of
    ent_daily_ff is applied, and (const, state) are packed again.  One small host round trip per day."""
    import ent_daily_ff as D
    cells = {k: v for k, v in host_cells().items()}
    c = {k: np.asarray(v) for k, v in const.items()}
    st = {k: np.asarray(v) for k, v in state.items()}
    ic = 0
    for n, k in enumerate(keys):
        cell = cells.get(k)
        if cell is None:
            continue
        cell.airtemp_10d, cell.par_10d = float(st['airtemp_10d'][n]), float(st['par_10d'][n])
        cell.gdd, cell.ncd = float(st['gdd'][n]), float(st['ncd'][n])
        cell.daylength = [float(st['daylength'][n, 0]), float(st['daylength'][n, 1])]
        cell.fall = int(st['fall'][n])
    # cohort order of pack_cells: cells in `keys` order, patches, cohorts
    ic = 0
    for k in keys:
        cell = cells.get(k)
        if cell is None:
            continue
        for p in cell.patches:
            for co in p.cohorts:
                co.Sacclim = float(st['Sacclim'][ic])
                ic += 1
    if 'lai_qty' not in REG:
        REG['lai_qty'] = D.read_lai_file()
    la = D.lai_linm2m(REG['lai_qty'], jday)
    for (ci, cj) in keys:
        cell = cells.get((ci, cj))
        if cell is not None:
            D.daily_update(cell, la[:, cj - 1, ci - 1], D.hemi_of_j(cj), jday)
    REG['exports'] = {k: E.call_exports(cells[k]) for k in keys if cells.get(k) is not None}      # D224: Ent exports of the updated cells (ws_can, shc_can, fv, height, albedo)
    pc, ps = pack_cells(cells, list(keys))
    return to_device(pc, ps)
