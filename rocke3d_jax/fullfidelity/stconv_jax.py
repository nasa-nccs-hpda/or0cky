"""JAX (batched) STCONV -- straits convection, Stage 2, D68 (speed-first JAX port; see D62).

STCONV (OCNKPP.f:3032-3229 and the combine step to 790) runs once per model step over the NMST
straits. Each strait has two half-boxes (IQ=1,2, RI=-beta,+beta). For each half-box it runs the
KPP diffusivities (KPPMIX with zero surface forcing, LDD false), the momentum and G/S diffusion
(OVDIFF/OVDIFFS with GHAT=0, DTP4=0), an ITER loop (up to 4, with the same HBL convergence rule as
OCONV), and an implicit application of the surface-free G/S tendencies. The half-boxes are then
combined into the strait's prognostic state (MUST, G0MST, GXMST, GZMST, S0MST, SXMST, SZMST).

Batched over H = 2*NMST half-boxes (h = 2*n + iq, n = strait index 0-based). Layer indices 1-based,
padded to LMO+2, as in setup_jax/ocnhbl_jax. Straits differ from OCONV in three ways that matter
here: no ZSCALE (zgrid comes from ZE directly), no RAVM weighting of the shears, and zero surface
forcing (Ustar = Bo = Bosol = 0).

EOS: VOLGSP (the seawater table) is external. The per-ITER BYRHO, DBLOC, DBSFC and RITOP are passed
in, as in D59/D66 (recorded from the instrumented run).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import kppmix_jax as KP
from ovdiffs_jax import ovdiff_jax, ovdiffs_jax

LMO = 13
ITMAX = 4
BETA = 0.5
BYBETA = 1.0 / BETA
EPSLN = 1e-20


def exponent(x):
    """Fortran EXPONENT: e with x = f*2**e, f in [0.5,1); 0 for x == 0."""
    _, e = jnp.frexp(x)
    return jnp.where(x == 0.0, 0, e)


def reduce_fig(nsig, rx):
    """Vectorized REDUCE_FIG (reduce_fig_ff.py, D57)."""
    nint = jnp.sign(rx) * jnp.floor(jnp.abs(rx) * jnp.exp2(-nsig.astype(jnp.float64)) + 0.5)
    red = jnp.ldexp(nint, nsig)
    return jnp.where(nsig + 30 > exponent(rx), red, rx)


def static_grid(ze):
    """STCONV's IFIRST block: zgrid, hwide, byhwide from ZE (no ZSCALE). Shapes (LMO+2,)."""
    zg = jnp.zeros(LMO + 2)
    zg = zg.at[0].set(EPSLN)
    zg = zg.at[1:LMO + 1].set(-0.5 * (ze[0:LMO] + ze[1:LMO + 1]))
    zg = zg.at[LMO + 1].set(-ze[LMO])
    hw = jnp.zeros(LMO + 2).at[0].set(EPSLN)
    hw = hw.at[1:LMO + 1].set(zg[0:LMO] - zg[1:LMO + 1])
    hw = hw.at[LMO + 1].set(EPSLN)
    byhw = jnp.zeros(LMO + 2)
    byhw = byhw.at[1:LMO + 1].set(1.0 / hw[1:LMO + 1])
    return zg, hw, byhw


def geometry(grav, dts, mmst, dist, wist, lmst):
    """Per-strait geometry (OCNKPP.f:3064-3080). mmst (N, LMO+1) 1-indexed, dist/wist/lmst (N,).
    Returns po, dtbydz, bydz2, mml, bymml with shape (N, LMO+1) (index 1..LMO)."""
    N = mmst.shape[0]
    L = jnp.arange(LMO + 1)[None, :]
    lm = lmst[:, None]
    dw = (dist * wist)[:, None]
    po1 = 0.5 * grav * mmst[:, 1:2] / dw
    step = 0.5 * grav * (mmst[:, 1:LMO] + mmst[:, 2:LMO + 1]) / dw       # MMST(L)+MMST(L+1), L=1..LMO-1
    po = jnp.concatenate([jnp.zeros((N, 1)), po1, po1 + jnp.cumsum(step, axis=1)], axis=1)[:, :LMO + 1]
    po = jnp.where(L >= 1, po, 0.0)
    mml = jnp.where((L >= 1) & (L <= lm), 0.5 * mmst, 0.0)
    bymml = jnp.where(mml > 0, 1.0 / jnp.where(mml > 0, mml, 1.0), 0.0)
    dtbydz = jnp.where((L >= 1) & (L <= lm), (dts * dw) / jnp.where(mmst > 0, mmst, 1.0), 0.0)
    bydz2_full = 2.0 * dw / (mmst[:, 2:LMO + 1] + mmst[:, 1:LMO])   # MMST(L+1)+MMST(L), L=1..LMO-1
    bydz2 = jnp.zeros((N, LMO + 1)).at[:, 1:LMO].set(bydz2_full)
    bydz2 = jnp.where((L >= 1) & (L <= lm - 1), bydz2, 0.0)
    return po, dtbydz, bydz2, mml, bymml


def stconv_jax(ze, grav, dts, nmst, lmst, mmst, dist, wist, jst, sinpo,
               must, g0mst, gxmst, gzmst, s0mst, sxmst, szmst, eos, tabs, debug=None):
    """Batched STCONV. Inputs with leading dimension NMST (per-strait), 1-indexed arrays (N, LMO+1).
    eos: dict with byrho, dbloc, dbsfc, ritop, each (ITMAX, 2N, LMO+1) per (ITER, half-box).
    Returns dict of the post-state (must, g0mst, gxmst, gzmst, s0mst, sxmst, szmst), (N, LMO+1)."""
    N = nmst
    H = 2 * N
    zg, hw, byhw = static_grid(ze)
    zg = jnp.broadcast_to(zg, (H, LMO + 2))
    hw = jnp.broadcast_to(hw, (H, LMO + 2))
    byhw = jnp.broadcast_to(byhw, (H, LMO + 2))

    po, dtbydz_n, bydz2_n, mml_n, bymml_n = geometry(grav, dts, mmst, dist, wist, lmst)
    # half-box arrays: h = 2*n + iq
    rep = lambda a: jnp.repeat(a, 2, axis=0)  # noqa: E731
    lm = rep(lmst)
    dtbydz = rep(dtbydz_n)
    bydz2 = rep(bydz2_n)
    mml = rep(mml_n)
    bymml = rep(bymml_n)
    iq = jnp.tile(jnp.array([1, 2]), N)
    ri = BETA * (2.0 * iq - 3.0)
    L = jnp.arange(LMO + 1)[None, :]
    act = (L >= 1) & (L <= lm[:, None])
    ul_init = jnp.where(act, 0.5 * rep(must) * rep(dist)[:, None] * bymml, 0.0)
    g0 = jnp.where(act, 0.5 * (rep(g0mst) + ri[:, None] * rep(gxmst)), 0.0)
    s0 = jnp.where(act, 0.5 * (rep(s0mst) + ri[:, None] * rep(sxmst)), 0.0)
    gz = jnp.where(act, 0.5 * rep(gzmst), 0.0)
    sz = jnp.where(act, 0.5 * rep(szmst), 0.0)
    ul0 = ul_init
    g0ml0 = g0
    s0ml0 = s0
    ul = ul_init
    g0ml = g0
    s0ml = s0
    done = jnp.zeros(H, dtype=bool)
    hbl = jnp.zeros(H)
    kbl = jnp.zeros(H, dtype=jnp.int64)
    fl_g = jnp.zeros((H, LMO + 1))
    fl_s = jnp.zeros((H, LMO + 1))
    akvg_last = jnp.zeros((H, LMO + 1))
    akvs_last = jnp.zeros((H, LMO + 1))
    zeros = jnp.zeros((H, LMO + 1))
    for k in range(1, ITMAX + 1):
        act_h = ~done
        byrho = eos['byrho'][k - 1]
        dbloc = eos['dbloc'][k - 1]
        dbsfc = eos['dbsfc'][k - 1]
        ritop = eos['ritop'][k - 1]
        # shears: plain (no RAVM), interface and tracer-point forms
        Lm1 = jnp.arange(LMO + 1)[None, :]
        ul_p = jnp.concatenate([ul, jnp.zeros((H, 1))], axis=1)            # index L+1 for L=LMO
        shsq = jnp.where((Lm1 >= 1) & (Lm1 <= lm[:, None] - 1),
                         (ul_p[:, :LMO + 1] - ul_p[:, 1:LMO + 2]) ** 2, 0.0)
        dvsq = jnp.where((Lm1 >= 1) & (Lm1 <= lm[:, None]),
                         (ul[:, 1:2] - ul[:, :LMO + 1]) ** 2, 0.0)
        shsq_pad = jnp.concatenate([shsq, jnp.zeros((H, 1))], axis=1)
        dvsq_pad = jnp.concatenate([dvsq, jnp.zeros((H, 1))], axis=1)
        zero = jnp.zeros(H)
        kpp = KP.kppmix_jax(ze, zg, hw, byhw, lm, shsq, dvsq, zero, zero, zero,
                            dbloc, ritop,
                            tabs['wmt'], tabs['wst'], tabs['fz500'], tabs['vtc'], tabs['cg'],
                            tabs['difmiw'], tabs['difsiw'], tabs['lsrpd'], tabs['fsr'],
                            tabs['dfsrdz'], tabs['dfsrdzb'])
        visc, difs, dift, ghat, hbl_k, kbl_k = kpp
        if debug is not None:
            debug.append(dict(k=k, shsq=shsq, dvsq=dvsq, visc=visc, difs=difs, dift=dift, ghat=ghat, hbl=hbl_k, kbl=kbl_k, ul=ul, lm=lm))
        # density rescale (OCNKPP.f:3305-3314): R = 0.5*(RHO(L)+RHO(L+1)), RHO = 1/BYRHO
        rho = jnp.concatenate([1.0 / byrho[:, 1:], jnp.zeros((H, 1))], axis=1)
        rho = jnp.concatenate([jnp.zeros((H, 1)), rho, jnp.zeros((H, 1))], axis=1)
        R = 0.5 * (rho[:, 1:LMO + 1] + rho[:, 2:LMO + 2])
        R2 = jnp.concatenate([jnp.zeros((H, 1)), R * R], axis=1)
        mask_l = (Lm1 >= 1) & (Lm1 <= lm[:, None] - 1)
        akvm = jnp.where(mask_l, visc[:, :LMO + 1] * R2, 0.0)
        akvg = jnp.where(mask_l, dift[:, :LMO + 1] * R2, 0.0)
        akvs = jnp.where(mask_l, difs[:, :LMO + 1] * R2, 0.0)
        # momentum OVDIFF (one vector per half-box), then G and S OVDIFFS: GHAT = 0, DTP4 = 0
        ul_new = ovdiff_jax(akvm, zeros, zeros, dtbydz, bydz2, lm, ul0)
        ul_new = jnp.where(act_h[:, None], ul_new, ul)
        u_g, fl_g_k = ovdiffs_jax(akvg, zeros, zeros, dtbydz, bydz2, dts, lm, g0ml0)
        u_s, fl_s_k = ovdiffs_jax(akvs, zeros, zeros, dtbydz, bydz2, dts, lm, s0ml0)
        ul = ul_new
        g0ml = jnp.where(act_h[:, None], u_g[:, :LMO + 1], g0ml)
        s0ml = jnp.where(act_h[:, None], u_s[:, :LMO + 1], s0ml)
        fl_g = jnp.where(act_h[:, None], fl_g_k[:, :LMO + 1], fl_g)
        fl_s = jnp.where(act_h[:, None], fl_s_k[:, :LMO + 1], fl_s)
        akvg_last = jnp.where(act_h[:, None], akvg, akvg_last)
        akvs_last = jnp.where(act_h[:, None], akvs, akvs_last)
        hblp = hbl
        hbl = jnp.where(act_h, hbl_k, hbl)
        kbl = jnp.where(act_h, kbl_k, kbl)
        dz_kbl = ze[kbl] - ze[kbl - 1]   # OCNKPP.f:3320 uses ZE, not zgrid
        moved = jnp.abs(hblp - hbl) > (dz_kbl * 0.25)
        cont = ((k == 1) | moved) & (k < ITMAX)
        done = done | (act_h & ~cont)
    # ---- implicit application of the surface-free tendencies (OCNKPP.f:3340-3380) ----
    bydts = 1.0 / dts
    dtb = dtbydz

    def apply(gz_in, fl, akv):
        out = gz_in
        d1 = 12.0 * dtb[:, 1] ** 2 * bydts
        out = out.at[:, 1].set((gz_in[:, 1] + 3.0 * fl[:, 1]) / (1.0 + d1 * akv[:, 1]))
        Lr = jnp.arange(LMO + 1)[None, :]
        dmid = 6.0 * dtb ** 2 * bydts
        mid_new = (gz_in + 3.0 * (jnp.concatenate([jnp.zeros((H, 1)), fl[:, :LMO]], axis=1) + fl)) / \
                  (1.0 + dmid * (jnp.concatenate([jnp.zeros((H, 1)), akv[:, :LMO]], axis=1) + akv))
        sel_mid = (Lr >= 2) & (Lr <= lm[:, None] - 1)
        out = jnp.where(sel_mid, mid_new, out)
        last = lm
        dl = 12.0 * _take(dtb, last) ** 2 * bydts
        val = (_take(gz_in, last) + 3.0 * _take(fl, last - 1)) / (1.0 + dl * _take(akv, last - 1))
        out = out.at[jnp.arange(H), last].set(val)
        return out

    gz_fin = apply(gz, fl_g, akvg_last)
    sz_fin = apply(sz, fl_s, akvs_last)

    # ---- combine half-boxes (OCNKPP.f:3390-3425) ----
    def pair(x):
        return x.reshape(N, 2, LMO + 1)
    ul2 = pair(ul)
    g2 = pair(g0ml)
    s2 = pair(s0ml)
    gz2 = pair(gz_fin)
    sz2 = pair(sz_fin)
    Lr = jnp.arange(LMO + 1)[None, :]
    actn = (Lr >= 1) & (Lr <= lmst[:, None])
    mml_p = mml_n
    must_n = jnp.where(actn, (ul2[:, 1] + ul2[:, 0]) * mml_p / dist[:, None], must)
    g0_n = jnp.where(actn, g2[:, 1] + g2[:, 0], g0mst)
    nsigg = exponent(g0_n) - 1 - 42
    gx_raw = (g2[:, 1] - g2[:, 0]) * BYBETA
    gx_n = jnp.where(actn, reduce_fig(nsigg, gx_raw), gxmst)
    gz_n = jnp.where(actn, gz2[:, 1] + gz2[:, 0], gzmst)
    s0_n = jnp.where(actn, s2[:, 1] + s2[:, 0], s0mst)
    nsigs = exponent(s0_n) - 1 - 42 + 4
    sx_raw = (s2[:, 1] - s2[:, 0]) * BYBETA
    sx_n = jnp.where(actn, reduce_fig(nsigs, sx_raw), sxmst)
    sz_n = jnp.where(actn, sz2[:, 1] + sz2[:, 0], szmst)
    # limit salinity gradients (OCNKPP.f:3418-3421)
    sx_n = jnp.where(actn & (jnp.abs(sx_n) > s0_n), jnp.sign(sx_n) * jnp.abs(s0_n), sx_n)
    sz_n = jnp.where(actn & (jnp.abs(sz_n) > s0_n), jnp.sign(sz_n) * jnp.abs(s0_n), sz_n)
    return dict(ul=ul, g0ml=g0ml, s0ml=s0ml, gz=gz_fin, sz=sz_fin, hbl=hbl, kbl=kbl,
                must=must_n, g0mst=g0_n, gxmst=gx_n, gzmst=gz_n, s0mst=s0_n, sxmst=sx_n,
                szmst=sz_n)


def _take(arr, idx):
    return jnp.take_along_axis(arr, idx[:, None], axis=1)[:, 0]
