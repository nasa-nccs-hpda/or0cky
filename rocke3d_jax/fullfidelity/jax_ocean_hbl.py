"""D190: NEW copy of ocean_hbl.py (D119) whose only change is a trace-safe `_native` (see below), so that the OCONV HBL loop can run inside a jit.
   The original docstring follows.
"""
"""D119: copy of ocnhbl_jax.hbl_loop (D66/D76) with three changes, everything else unchanged:
  1. ALPHAGSP/BETAGSP/SHCGS of layer 1 come from the OFTAB tables (agsp, bgsp, cgs arguments) instead of
     the recorded setup dump (`eos['alpha'|'beta'|'shc']` no longer read);
  2. the last-iteration AKVG/AKVS (after density rescale) are returned (`akvg`, `akvs`), needed for the
     OCONV tail (GZMO/SZMO flux update, GX/GY/SX/SY vertical diffusion);
  3. `kbl` of the final iteration is returned unchanged (KPL = KBL, OCNKPP.f post-loop).
"""
"""JAX (batched) OCONV HBL iteration loop -- Stage 2, D66 (speed-first JAX port).

Batched over N columns. Mirrors OCNKPP.f's ITER loop (OCNKPP.f:1979-2338) and the post-loop
pass (2339-2410), with per-column masks in place of GO TO 510:
  per ITER k (k = 1..4, unrolled):
    G, S = G0ML, S0ML * BYMML             (BYMML: pre-source at k=1, post-source after)
    setup_jax           (D64, OCONV per-column setup)
    kppmix_jax          (D63, KPPMIX)
    density rescale of AKVM/AKVG/AKVS    (OCNKPP.f:2240-2270)
    GHATG / GHATS        (OCNKPP.f:2250-2252)
    momentum OVDIFF per (column, K)     (ovdiff_jax, D62/D64)
    G and S OVDIFFS                      (ovdiffs_jax, D62)
    convergence: GO TO 510 while (k==1 or |HBLP-HBL| > layer*0.25) and k < 4
  post-loop: D-grid OVDIFF (non-pole) with the last ITER's AKVM, then the flux save
  (DM, FLG3D, FLS3D; OCNKPP.f:2399-2405).

Inputs that are not yet ported in JAX are passed in, as in D54-D63:
  - the seawater EOS (VOLGSP, ALPHAGSP, ...): per-ITER values, from the setup dump.
  - S0M1(I,J) and the per-column entry state (UL, UL0, ULD, ULD0, G0ML, G0ML0, S0ML, S0ML0).
Index conventions: 1-indexed padded arrays, as in setup_jax.py (LMO+2 for L, KMAX+1 for K).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import kppmix_jax as KP
from eos_jax import volgsp
from setup_jax import setup_jax
from ovdiffs_jax import ovdiff_jax, ovdiffs_jax

LMO = 13
KM = 74
ITMAX = 4


def _native(x):
    """D190: trace-safe version of ocean_hbl._native (the original calls np.asarray, which fails on tracers inside a jit).  Concrete numpy inputs are converted to
    native-endian device arrays as before; tracers and device arrays pass through; ints stay int64."""
    if isinstance(x, (jnp.ndarray, jax.core.Tracer)):
        return x
    x = np.asarray(x)
    if x.dtype.kind == 'f':
        return jnp.asarray(x.astype(np.float64))
    return jnp.asarray(x.astype(np.int64)) if x.dtype.kind in 'iu' else jnp.asarray(x)


def _shcgs(cgs, g, s):
    gg = g * 2.5e-4
    ss = s * 1000.0
    ig = jnp.clip(jnp.trunc(gg + 2.0).astype(jnp.int64) - 2, -2, 39)
    js = jnp.trunc(ss).astype(jnp.int64)
    js = jnp.where(js >= 40, 39, js)
    c = lambda i, j: cgs[i + 2, j]   # noqa: E731
    return ((js - ss + 1) * ((ig - gg + 1) * c(ig, js) + (gg - ig) * c(ig + 1, js))
            + (ss - js) * ((ig - gg + 1) * c(ig, js + 1) + (gg - ig) * c(ig + 1, js + 1)))


def _take(arr, idx):
    return jnp.take_along_axis(arr, idx[:, None], axis=1)[:, 0]


def hbl_loop(ze, grav, lmij, kmuv, pole, dts, dxypo, mo, mo1, deltae, deltas, deltam, deltasr,
             u2rho, ogeoz, hocean, s0m1, ravm, lmuv, dtbydz, bydz2,
             ul0, ulm, uld0, uld, g0ml0, s0ml0, g0ml, s0ml, eos, tabs, itmax=ITMAX,
             po=None, vgsp=None, agsp=None, bgsp=None, cgs=None):
    """Batched ITER loop + post-loop pass.

    Shapes: lmij, kmuv (N,) int64; pole (N,) bool; dxypo, mo1, deltae, deltas, deltam, deltasr,
    u2rho, ogeoz, hocean, s0m1 (N,); mo, dtbydz, bydz2, g0ml, g0ml0, s0ml, s0ml0 (N, LMO+1);
    ravm, lmuv (N, KM+1); ul0, ulm (N, LMO+1, KM+1) where ulm is the entry UL (setup reads it);
    uld0, uld (N, LMO+1, KM+1); eos: dict with byrho, rhom, rho1 (ITMAX, N, LMO+1),
    alpha, beta, shc (ITMAX, N). tabs: dict with the kmixinit/init_solar tables.
    Returns dict: ul, uld, g0ml, s0ml (final), hbl, kbl (N,), iters (N,), flg3d, fls3d (N, LMO+2),
    dm (N,).
    """
    cv = _native
    ze, lmij, kmuv, pole, dxypo, mo, mo1 = map(cv, (ze, lmij, kmuv, pole, dxypo, mo, mo1))
    deltae, deltas, deltam, deltasr, u2rho, ogeoz, hocean, s0m1 = map(
        cv, (deltae, deltas, deltam, deltasr, u2rho, ogeoz, hocean, s0m1))
    ravm, lmuv, dtbydz, bydz2 = map(cv, (ravm, lmuv, dtbydz, bydz2))
    ul0, ulm, uld0, uld, g0ml0, s0ml0, g0ml, s0ml = map(cv, (ul0, ulm, uld0, uld, g0ml0, s0ml0, g0ml, s0ml))
    eos = {k: cv(v) for k, v in eos.items()}
    tabs = {k: (cv(v) if not isinstance(v, int) else v) for k, v in tabs.items()}
    N = lmij.shape[0]
    Lidx = jnp.arange(LMO + 1)[None, :]
    done = jnp.zeros(N, dtype=bool)
    iters = jnp.zeros(N, dtype=jnp.int64)
    hbl = jnp.zeros(N)
    kbl = jnp.zeros(N, dtype=jnp.int64)
    ul = ulm
    g_cur = g0ml
    s_cur = s0ml
    fl_g = jnp.zeros((N, LMO + 1))
    fl_s = jnp.zeros((N, LMO + 1))
    akvm_last = jnp.zeros((N, LMO + 1))
    akvg_last = jnp.zeros((N, LMO + 1))
    akvs_last = jnp.zeros((N, LMO + 1))
    ulx = ul

    for k in range(1, itmax + 1):
        act = ~done
        # BYMML(L): pre-source at k=1 for L=1 (MMLT), post-source otherwise (MML0)
        mm = mo * dxypo[:, None]
        bymml = 1.0 / mm
        if k == 1:
            bymml = bymml.at[:, 1].set(1.0 / (mo1 * dxypo))
        g = jnp.concatenate([jnp.zeros((N, 1)), g_cur[:, 1:]], axis=1) * bymml
        s = jnp.concatenate([jnp.zeros((N, 1)), s_cur[:, 1:]], axis=1) * bymml
        g = g.at[:, 0].set(0.0)
        s = s.at[:, 0].set(0.0)
        if vgsp is not None:
            # EOS from the OFTAB table (OCNDYN/OCNKPP VOLGSP): BYRHO, RHOM, RHO1 from G, S, PO
            byrho = volgsp(vgsp, g, s, po)
            g_prev = jnp.concatenate([jnp.zeros((N, 1)), g[:, :LMO]], axis=1)
            s_prev = jnp.concatenate([jnp.zeros((N, 1)), s[:, :LMO]], axis=1)
            rhom = 1.0 / volgsp(vgsp, g_prev, s_prev, po)
            rho1 = 1.0 / volgsp(vgsp, jnp.broadcast_to(g[:, 1:2], (N, LMO + 1)),
                                jnp.broadcast_to(s[:, 1:2], (N, LMO + 1)), po)
        else:
            byrho = eos['byrho'][k - 1]
            rhom = eos['rhom'][k - 1]
            rho1 = eos['rho1'][k - 1]
        alpha_k = volgsp(agsp, g[:, 1], s[:, 1], po[:, 1])
        beta_k = volgsp(bgsp, g[:, 1], s[:, 1], po[:, 1]) * 1e3
        shc_k = _shcgs(cgs, g[:, 1], s[:, 1])
        ul_pad = jnp.concatenate([ulx, jnp.zeros((N, 1, KM + 1))], axis=1)   # (N, LMO+2, KM+1)
        su = setup_jax(ze, lmij, kmuv, ogeoz, hocean, grav, ul_pad, ravm, g, s, byrho, rhom, rho1,
                       alpha_k, beta_k, shc_k, u2rho,
                       deltae, deltas, deltam, deltasr)
        visc, difs, dift, ghat, hbl_k, kbl_k = KP.kppmix_jax(
            ze, su['zgrid'], su['hwide'], su['byhwide'], lmij, su['shsq'], su['dvsq'],
            su['ustar'], su['bo'], su['bosol'], su['dbloc'], su['ritop'],
            tabs['wmt'], tabs['wst'], tabs['fz500'], tabs['vtc'], tabs['cg'],
            tabs['difmiw'], tabs['difsiw'], tabs['lsrpd'], tabs['fsr'], tabs['dfsrdz'],
            tabs['dfsrdzb'])
        # density rescale, R = 0.5*(RHO(L)+RHO(L+1)), RHO = 1/BYRHO   (OCNKPP.f:2240-2270)
        rho = jnp.concatenate([1.0 / byrho[:, 1:], jnp.zeros((N, 1))], axis=1)  # rho[L-1] = RHO(L)
        rho = jnp.concatenate([jnp.zeros((N, 1)), rho], axis=1)              # index L: RHO(L)
        rho = jnp.concatenate([rho, jnp.zeros((N, 1))], axis=1)              # RHO(LMO+2)=0
        R = 5e-1 * (rho[:, 1:LMO + 1] + rho[:, 2:LMO + 2])                   # R(L), L = 1..LMO
        R2 = jnp.concatenate([jnp.zeros((N, 1)), R * R], axis=1)             # index L, 0..LMO
        akvm = visc[:, :LMO + 1] * R2[:, :LMO + 1]
        akvg = dift[:, :LMO + 1] * R2[:, :LMO + 1]
        akvs = difs[:, :LMO + 1] * R2[:, :LMO + 1]
        # GHATG / GHATS (OCNKPP.f:2250-2252): raw AKVG/AKVS (pre-rescale) times GHAT
        ghatg = dift[:, :LMO + 1] * ghat[:, :LMO + 1] * deltae[:, None] * dxypo[:, None]
        ghats = difs[:, :LMO + 1] * ghat[:, :LMO + 1] * (
            deltas[:, None] - s0ml0[:, 1:2] * bymml[:, 1:2] * deltam[:, None]) * dxypo[:, None]
        mask_l = (Lidx >= 1) & (Lidx <= (lmij[:, None] - 1))
        akvm = jnp.where(mask_l, akvm, 0.0)
        akvg = jnp.where(mask_l, akvg, 0.0)
        akvs = jnp.where(mask_l, akvs, 0.0)
        ghatg = jnp.where(mask_l, ghatg, 0.0)
        ghats = jnp.where(mask_l, ghats, 0.0)

        # momentum OVDIFF: one solve per (column, K); ghat_m = 0, dtp4 = 0 (DTP4UV is zero here)
        kidx = jnp.arange(KM + 1)[None, :]
        lm_k = lmuv                                              # (N, KM+1) per-K active length
        act_k = (kidx >= 1) & (kidx <= kmuv[:, None]) & (lm_k > 1) & act[:, None]
        n_flat = N * (KM + 1)
        k_m = jnp.repeat(akvm[:, None, :], KM + 1, axis=1).reshape(n_flat, LMO + 1)
        dtb = jnp.repeat(dtbydz[:, None, :], KM + 1, axis=1).reshape(n_flat, LMO + 1)
        byd = jnp.repeat(bydz2[:, None, :], KM + 1, axis=1).reshape(n_flat, LMO + 1)
        zero_flat = jnp.zeros((n_flat, LMO + 1))
        u0_flat = jnp.transpose(ul0[:, :LMO + 1, :], (0, 2, 1)).reshape(n_flat, LMO + 1)
        lm_flat = lm_k.reshape(n_flat)
        u_m = ovdiff_jax(k_m, zero_flat, zero_flat, dtb, byd, jnp.maximum(lm_flat, 1), u0_flat)
        u_m = u_m.reshape(N, KM + 1, LMO + 1).transpose(0, 2, 1)     # (N, LMO+1, KM+1)
        ul_new = jnp.where(act_k[:, None, :], u_m, ul)

        # G and S OVDIFFS (OCNKPP.f:2332-2336)
        u_g, fl_g_k = ovdiffs_jax(akvg, ghatg, jnp.zeros((N, LMO + 1)), dtbydz, bydz2,
                                  dts, lmij, g0ml0)
        u_s, fl_s_k = ovdiffs_jax(akvs, ghats, jnp.zeros((N, LMO + 1)), dtbydz, bydz2,
                                  dts, lmij, s0ml0)
        g_new = jnp.where(act[:, None], u_g[:, :LMO + 1], g_cur)
        s_new = jnp.where(act[:, None], u_s[:, :LMO + 1], s_cur)
        ul = ul_new
        g_cur, s_cur = g_new, s_new
        fl_g = jnp.where(act[:, None], fl_g_k[:, :LMO + 1], fl_g)
        fl_s = jnp.where(act[:, None], fl_s_k[:, :LMO + 1], fl_s)
        akvm_last = jnp.where(act[:, None], akvm, akvm_last)
        akvg_last = jnp.where(act[:, None], akvg, akvg_last)
        akvs_last = jnp.where(act[:, None], akvs, akvs_last)

        # convergence (OCNKPP.f:2337-2338)
        hblp = hbl
        hbl = jnp.where(act, hbl_k, hbl)
        kbl = jnp.where(act, kbl_k, kbl)
        iters = jnp.where(act, k, iters)
        dz_kbl = ze[kbl] - ze[kbl - 1]
        moved = jnp.abs(hblp - hbl) > dz_kbl * 0.25
        cont = (k == 1) | moved
        cont = cont & (k < itmax)
        done = done | (act & ~cont)
        ulx = ul

    # ---- post-loop: D-grid OVDIFF (non-pole), with the last ITER's AKVM ----
    kidx = jnp.arange(KM + 1)[None, :]
    act_k = (kidx >= 1) & (kidx <= kmuv[:, None]) & (lmuv > 1) & (~pole)[:, None]
    n_flat = N * (KM + 1)
    k_m = jnp.repeat(akvm_last[:, None, :], KM + 1, axis=1).reshape(n_flat, LMO + 1)
    dtb = jnp.repeat(dtbydz[:, None, :], KM + 1, axis=1).reshape(n_flat, LMO + 1)
    byd = jnp.repeat(bydz2[:, None, :], KM + 1, axis=1).reshape(n_flat, LMO + 1)
    zero_flat = jnp.zeros((n_flat, LMO + 1))
    ud0_flat = jnp.transpose(uld0[:, :LMO + 1, :], (0, 2, 1)).reshape(n_flat, LMO + 1)
    lm_flat = lmuv.reshape(n_flat)
    ud_m = ovdiff_jax(k_m, zero_flat, zero_flat, dtb, byd, jnp.maximum(lm_flat, 1), ud0_flat)
    ud_m = ud_m.reshape(N, KM + 1, LMO + 1).transpose(0, 2, 1)
    uld_out = jnp.where(act_k[:, None, :], ud_m, uld)

    # ---- flux save (OCNKPP.f:2399-2405) ----
    dm = deltam * (dts / mo[:, 1])
    flg3d = jnp.zeros((N, LMO + 2))
    fls3d = jnp.zeros((N, LMO + 2))
    flg3d = flg3d.at[:, :LMO + 1].set(jnp.where(mask_l, fl_g, 0.0))
    fls3d = fls3d.at[:, :LMO + 1].set(jnp.where(mask_l, fl_s, 0.0))
    flg3d = flg3d.at[:, 0].set(dts * deltae * dxypo)
    fls3d = fls3d.at[:, 0].set(-dts * deltas * dxypo * (1 - dm) + dm * s0m1)
    return dict(ul=ul, uld=uld_out, g0ml=g_cur, s0ml=s_cur, hbl=hbl, kbl=kbl, iters=iters,
                flg3d=flg3d, fls3d=fls3d, dm=dm, akvm=akvm_last, akvg=akvg_last, akvs=akvs_last)
