"""JAX (batched) port of KPPMIX -- Stage 2, D63 (speed-first JAX port; see D62).

Batched over N columns, each with its own active depth `lmij` (= KMTJ). Mirrors kppmix_ff.kppmix
(D54) step for step, restructured for jit:
  - z121 smoothing: lax.scan over the layer index, since the v[0] carry is sequential.
  - bulk-Richardson search: rib_ka is the previous level's rib_ku, so every level is computed at
    once and the first crossing is found with argmax.
  - _wscale: both branches computed, gathers clipped to valid indices, merged with where.
  - the per-level coefficient loop (ki < kbl) and the kbl correction run as masked vector ops.

Index convention (same as kppmix_ff.py): ze, shsq, dvsq, dbloc, ritop, fsr, dfsrdz, dfsrdzb have
Fortran indices 0..LMO (length LMO+1); zgrid, hwide, byhwide, visc, difs, dift have indices
0..LMO+1 (length LMO+2). Outputs: visc (=AKVM), difs (=AKVS), dift (=AKVG), ghats (=GHAT), hbl, kbl.

Accuracy is checked to a stated tolerance against the recorded KPPMIX calls (kppmix_jax_compare.py),
not bitwise: the speed-first policy allows reordered floating-point operations.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import kppmix_ff as K

LMO = K.LMO
NNI, NNJ = K.NNI, K.NNJ


def _sign(a, b):
    """Fortran SIGN(a,b): |a| if b >= 0, else -|a|."""
    return jnp.where(b >= 0.0, jnp.abs(a), -jnp.abs(a))


def _take(arr, idx):
    """Gather one value per column: arr (N, M), idx (N,) -> (N,)."""
    return jnp.take_along_axis(arr, idx[:, None], axis=1)[:, 0]


def _wscale(sigma, depth, ustar, bfsfc, wmt, wst):
    """Vectorized kppmix_ff._wscale. Arguments broadcast together; returns (wm, ws)."""
    zehat = K.VONK * sigma * depth * bfsfc
    small = zehat <= K.ZMAX
    zdiff = zehat - K.ZMIN
    iz = jnp.clip(jnp.floor(zdiff * K.RDELTAZ).astype(jnp.int64), 0, NNI)
    izp1 = iz + 1
    udiff = ustar - K.UMIN
    ju = jnp.clip(jnp.floor(udiff * K.RDELTAU).astype(jnp.int64), 0, NNJ)
    jup1 = ju + 1
    zfrac = zdiff * K.RDELTAZ - iz.astype(jnp.float64)
    ufrac = udiff * K.RDELTAU - ju.astype(jnp.float64)
    fzfrac = 1.0 - zfrac

    wam = fzfrac * wmt[iz, jup1] + zfrac * wmt[izp1, jup1]
    wbm = fzfrac * wmt[iz, ju] + zfrac * wmt[izp1, ju]
    wm = (1.0 - ufrac) * wbm + ufrac * wam
    wm = jnp.where((ju == NNJ) & (wm < wam), wam, wm)

    was = fzfrac * wst[iz, jup1] + zfrac * wst[izp1, jup1]
    wbs = fzfrac * wst[iz, ju] + zfrac * wst[izp1, ju]
    ws = (1.0 - ufrac) * wbs + ufrac * was
    ws = jnp.where((ju == NNJ) & (ws < was), was, ws)

    u3 = ustar * ustar * ustar
    wm_big = K.VONK * ustar * u3 / (u3 + K.CONC1 * zehat)
    return jnp.where(small, wm, wm_big), jnp.where(small, ws, wm_big)


def _swfrac_search(ze, zg_kl, kl, kmax, fsr, dfsrdz, dfsrdzb):
    """Vectorized kppmix_ff._bfsfc_search. kl is an int array; kmax per column."""
    kt = kl + jnp.round(0.5 + _sign(0.5, -(ze[kl] + zg_kl))).astype(jnp.int64)
    ktc = jnp.clip(kt, 1, LMO)
    val_eq = fsr[ktc] + (zg_kl + ze[jnp.clip(ktc - 1, 0, LMO)]) * dfsrdzb[ktc]
    val_lt = fsr[ktc] + (zg_kl + ze[jnp.clip(ktc - 1, 0, LMO)]) * dfsrdz[ktc]
    out = jnp.where(kt == kmax, val_eq, val_lt)
    return jnp.where(kt > kmax, 0.0, out)


def _swfrac_at_hbl(ze, hbl, kbl, kmax, fsr, dfsrdz, dfsrdzb):
    """Vectorized kppmix_ff._bfsfc_at_hbl. kbl is per column."""
    km1 = kbl - 1
    kt = km1 + jnp.round(0.5 + _sign(0.5, -(ze[km1] - hbl))).astype(jnp.int64)
    ktc = jnp.clip(kt, 1, LMO)
    val_eq = fsr[ktc] + (ze[jnp.clip(ktc - 1, 0, LMO)] - hbl) * dfsrdzb[ktc]
    val_lt = fsr[ktc] + (ze[jnp.clip(ktc - 1, 0, LMO)] - hbl) * dfsrdz[ktc]
    out = jnp.where(kt == kmax, val_eq, val_lt)
    return jnp.where(kt > kmax, 0.0, out)


def _coeffs(sig, hbl, stable, ustar, bfsfc, wmt, wst, gat1, dat1):
    """Shared boundary-layer shape coefficients at normalized depth sig (all arrays broadcast).
    Returns (wm, ws, gm, gs, gt) pieces used by the blmc and kbl-correction formulas."""
    sigma = stable * sig + (1.0 - stable) * jnp.minimum(sig, K.EPSILON)
    wm, ws = _wscale(sigma, hbl, ustar, bfsfc, wmt, wst)
    a1 = sig - 2.0
    a2 = 3.0 - 2.0 * sig
    a3 = sig - 1.0
    gm = a1 + a2 * gat1[0] + a3 * dat1[0]
    gs = a1 + a2 * gat1[1] + a3 * dat1[1]
    gt = a1 + a2 * gat1[2] + a3 * dat1[2]
    return wm, ws, gm, gs, gt


@jax.jit
def kppmix_jax(ze, zgrid, hwide, byhwide, lmij, shsq, dvsq, ustar, bo, bosol, dbloc, ritop,
               wmt, wst, fz500, vtc, cg, difmiw, difsiw, lsrpd, fsr, dfsrdz, dfsrdzb):
    """Batched KPPMIX. Per-column inputs have leading dimension N. Returns
    (visc, difs, dift, ghats, hbl, kbl) with shapes (N, LMO+2), (N, LMO+2), (N, LMO+2),
    (N, LMO+1), (N,), (N,)."""
    N = zgrid.shape[0]
    km = lmij                                              # (N,) int, KMTJ
    kmax = jnp.minimum(lsrpd, km)
    kk = jnp.arange(LMO + 2)[None, :]                      # Fortran index 0..LMO+1
    act = (kk >= 1) & (kk <= km[:, None])

    # pad the LMO+1 arrays to LMO+2 so every array shares the same index range
    def pad2(x):
        return jnp.concatenate([x, jnp.zeros((N, 1))], axis=1)

    dbloc2 = pad2(dbloc)
    shsq2 = pad2(shsq)

    # ---- initial visc/dift (OCNKPP.f:~550), then z121 smoothing (NUM_V_SMOOTH_RI = 1) ----
    visc = jnp.where(act, dbloc2 * (zgrid - jnp.roll(zgrid, -1, axis=1)) / (shsq2 + K.EPSL), 0.0)
    dift = jnp.where(act, dbloc2 * jnp.roll(byhwide, -1, axis=1), 0.0)

    v_km = _take(visc, km)
    vnext = jnp.where(kk == km[:, None], v_km[:, None], jnp.roll(visc, -1, axis=1))

    def zstep(v0, xs):
        vk, vkp1, active = xs
        new = v0 + 0.5 * vk + 0.25 * vkp1
        v0n = jnp.where(active, 0.25 * vk, v0)
        return v0n, jnp.where(active, new, 0.0)

    k_idx = jnp.arange(1, LMO + 2)
    xs = (visc[:, 1:LMO + 2].T, vnext[:, 1:LMO + 2].T, (k_idx[:, None] <= km[None, :]))
    _, zvals = jax.lax.scan(zstep, 0.25 * visc[:, 1], xs)      # (LMO+1, N), k = 1..LMO+1
    visc = jnp.concatenate([jnp.zeros((N, 1)), zvals.T], axis=1)     # (N, LMO+2)

    # ---- diffusivity closure (OCNKPP.f:~560-580): uses post-smoothing visc, pre-update dift ----
    rigg = jnp.maximum(dift, K.BVSQCON)
    ratio = jnp.minimum((K.BVSQCON - rigg) * K.RBVSQCON, 1.0)
    fcon = 1.0 - ratio * ratio
    fcon = fcon * fcon * fcon
    ratio2 = jnp.minimum(jnp.maximum(visc, 0.0) * K.RRIINFTY, 1.0)
    fri = 1.0 - ratio2 * ratio2
    fri = fri * fri * fri
    ftop = jnp.concatenate([fz500.T[km], jnp.zeros((N, 1))], axis=1)[:, :LMO + 2]  # fz500[ki,kmtj]
    visc_n = difmiw + fcon * K.DIFMCON + fri * K.DIFM0
    difs_n = difsiw + fcon * K.DIFSCON + fri * K.DIFS0 + ftop * K.DIFTOP
    visc = jnp.where(act, visc_n, visc)
    difs = jnp.where(act, difs_n, 0.0)
    dift = jnp.where(act, difs_n, 0.0)
    visc = jnp.where((kk == 0) | (kk >= km[:, None]), 0.0, visc)
    difs = jnp.where((kk == 0) | (kk >= km[:, None]), 0.0, difs)
    dift = jnp.where((kk == 0) | (kk >= km[:, None]), 0.0, dift)

    # ---- bulk Richardson search over kl = 2..LMO (OCNKPP.f:~422-527) ----
    kl = jnp.arange(2, LMO + 1)                            # (LMO-1,)
    zg_kl = zgrid[:, 2:LMO + 1]                            # zgrid(kl)
    sw = _swfrac_search(ze, zg_kl, kl[None, :], kmax[:, None], fsr, dfsrdz, dfsrdzb)
    bfs_s = bo[:, None] + bosol[:, None] * (1.0 - sw)
    stable_s = 0.5 + _sign(0.5, bfs_s)
    sigma_s = stable_s + (1.0 - stable_s) * K.EPSILON
    wm_s, ws_s = _wscale(sigma_s, -zg_kl, ustar[:, None], bfs_s, wmt, wst)
    bvsq = 0.5 * (dbloc2[:, 1:LMO] * byhwide[:, 2:LMO + 1] + dbloc2[:, 2:LMO + 1] * byhwide[:, 3:LMO + 2])
    vtsq = -zg_kl * ws_s * jnp.sqrt(jnp.abs(bvsq)) * vtc
    rib_ku = ritop[:, 2:LMO + 1] / (dvsq[:, 2:LMO + 1] + vtsq + K.EPSL)
    rib_ka = jnp.concatenate([jnp.zeros((N, 1)), rib_ku[:, :-1]], axis=1)
    cross = (kl[None, :] <= km[:, None]) & (rib_ku > K.RICR)
    any_cross = jnp.any(cross, axis=1)
    first = jnp.argmax(cross, axis=1)
    kbl_f = kl[first]
    rka_f = _take(rib_ka, first)
    rku_f = _take(rib_ku, first)
    zg_km1 = _take(zgrid, kbl_f - 1)
    zg_k = _take(zgrid, kbl_f)
    hbl_f = -zg_km1 + (zg_km1 - zg_k) * (K.RICR - rka_f) / (rku_f - rka_f)
    hbl = jnp.where(any_cross, hbl_f, -_take(zgrid, km))
    kbl = jnp.where(any_cross, kbl_f, km)

    # ---- post-search quantities (OCNKPP.f:~530-560) ----
    bfsfc = _swfrac_at_hbl(ze, hbl, kbl, kmax, fsr, dfsrdz, dfsrdzb)
    bfsfc = bo + bosol * (1.0 - bfsfc)
    stable = 0.5 + _sign(0.5, bfsfc)
    bfsfc = bfsfc + stable * K.EPSL
    zg_kbl = _take(zgrid, kbl)
    hw_kbl = _take(hwide, kbl)
    caseA = 0.5 + _sign(0.5, -zg_kbl - 0.5 * hw_kbl - hbl)
    byhbl = 1.0 / hbl
    sigma = stable + (1.0 - stable) * K.EPSILON
    wm, ws = _wscale(sigma, hbl, ustar, bfsfc, wmt, wst)
    byws = 1.0 / (ws + K.EPSL)
    caseA_i = jnp.round(caseA + K.EPSL).astype(jnp.int64)
    kn = caseA_i * (kbl - 1) + (1 - caseA_i) * kbl

    delhat = 0.5 * _take(hwide, kn) - _take(zgrid, kn) - hbl
    r = 1.0 - delhat * _take(byhwide, kn)

    def deriv(arr):
        dvdzup = (_take(arr, kn - 1) - _take(arr, kn)) * _take(byhwide, kn)
        dvdzdn = (_take(arr, kn) - _take(arr, kn + 1)) * _take(byhwide, kn + 1)
        return 0.5 * ((1.0 - r) * (dvdzup + jnp.abs(dvdzup)) + r * (dvdzdn + jnp.abs(dvdzdn)))

    viscp = deriv(visc)
    difsp = deriv(difs)
    diftp = deriv(dift)
    visch = _take(visc, kn) + viscp * delhat
    difsh = _take(difs, kn) + difsp * delhat
    difth = _take(dift, kn) + diftp * delhat

    f1 = stable * K.CONC1 * bfsfc / (ustar ** 4 + K.EPSL)
    gat1 = [visch * byhbl * (1.0 / (wm + K.EPSL)),
            difsh * byhbl * byws,
            difth * byhbl * byws]
    dat1 = [jnp.minimum(-viscp * (1.0 / (wm + K.EPSL)) + f1 * visch, 0.0),
            jnp.minimum(-difsp * byws + f1 * difsh, 0.0),
            jnp.minimum(-diftp * byws + f1 * difth, 0.0)]

    # ---- boundary-layer coefficients for ki = 1..kbl-1 (OCNKPP.f:~580-620) ----
    ki = jnp.arange(1, LMO + 1)[None, :]
    act_ki = ki < kbl[:, None]
    zg_ki = zgrid[:, 1:LMO + 1]
    hw_ki = hwide[:, 1:LMO + 1]
    sig_ki = (-zg_ki + 0.5 * hw_ki) * byhbl[:, None]
    gat1_b = [g[:, None] for g in gat1]
    dat1_b = [d[:, None] for d in dat1]
    wm_b, ws_b, gm_b, gs_b, gt_b = _coeffs(sig_ki, hbl[:, None], stable[:, None],
                                           ustar[:, None], bfsfc[:, None], wmt, wst,
                                           gat1_b, dat1_b)
    b1 = hbl[:, None] * wm_b * sig_ki * (1.0 + sig_ki * gm_b)
    b2 = hbl[:, None] * ws_b * sig_ki * (1.0 + sig_ki * gs_b)
    b3 = hbl[:, None] * ws_b * sig_ki * (1.0 + sig_ki * gt_b)
    ghat_ki = (1.0 - stable[:, None]) * cg * byws[:, None] * byhbl[:, None]
    blmc1 = jnp.where(act_ki, b1, 0.0)
    blmc2 = jnp.where(act_ki, b2, 0.0)
    blmc3 = jnp.where(act_ki, b3, 0.0)
    ghats = jnp.concatenate([jnp.zeros((N, 1)), jnp.where(act_ki, ghat_ki, 0.0),
                             jnp.zeros((N, 1))], axis=1)[:, :LMO + 1]

    # ---- kbl-level correction (OCNKPP.f:~620-650); kbl <= kmtj always holds here ----
    sig_b = -_take(zgrid, kbl - 1) * byhbl
    wm_c, ws_c, gm_c, gs_c, gt_c = _coeffs(sig_b, hbl, stable, ustar, bfsfc, wmt, wst,
                                           gat1, dat1)
    dkm1_1 = hbl * wm_c * sig_b * (1.0 + sig_b * gm_c)
    dkm1_2 = hbl * ws_c * sig_b * (1.0 + sig_b * gs_c)
    dkm1_3 = hbl * ws_c * sig_b * (1.0 + sig_b * gt_c)
    kb1 = kbl - 1
    delta = (hbl + _take(zgrid, kb1)) * _take(byhwide, kb1 + 1)
    bl1 = _take(jnp.concatenate([jnp.zeros((N, 1)), blmc1], axis=1), kb1)
    bl2 = _take(jnp.concatenate([jnp.zeros((N, 1)), blmc2], axis=1), kb1)
    bl3 = _take(jnp.concatenate([jnp.zeros((N, 1)), blmc3], axis=1), kb1)
    v_kb1 = _take(visc, kb1)
    d_kb1 = _take(difs, kb1)
    t_kb1 = _take(dift, kb1)

    def corr(dkm1, own, old, bl):
        dkmp5 = caseA * old + (1.0 - caseA) * bl
        dstar = (1.0 - delta) ** 2 * dkm1 + delta ** 2 * dkmp5
        return (1.0 - delta) * own + delta * dstar

    new1 = corr(dkm1_1, v_kb1, v_kb1, bl1)
    new2 = corr(dkm1_2, d_kb1, d_kb1, bl2)
    new3 = corr(dkm1_3, t_kb1, t_kb1, bl3)
    rows = jnp.arange(N)
    blmc1 = blmc1.at[rows, kb1 - 1].set(new1)
    blmc2 = blmc2.at[rows, kb1 - 1].set(new2)
    blmc3 = blmc3.at[rows, kb1 - 1].set(new3)
    ghats = ghats.at[rows, kb1].set((1.0 - caseA) * _take(ghats, kb1))

    # ---- final assembly (OCNKPP.f:~650-660) ----
    keep = ki < kbl[:, None]
    # levels ki >= kbl keep their closure values (Fortran only assigns ki < kbl)
    visc_out = visc.at[:, 1:LMO + 1].set(jnp.where(keep, blmc1, visc[:, 1:LMO + 1]))
    difs_out = difs.at[:, 1:LMO + 1].set(jnp.where(keep, blmc2, difs[:, 1:LMO + 1]))
    dift_out = dift.at[:, 1:LMO + 1].set(jnp.where(keep, blmc3, dift[:, 1:LMO + 1]))
    ghats = jnp.where(jnp.arange(LMO + 1)[None, :] >= kbl[:, None], 0.0, ghats)
    return visc_out, difs_out, dift_out, ghats, hbl, kbl
