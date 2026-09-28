"""JAX-vectorized (batched-array) port of ghy_ref.py's GHY land model -- Track B.

Same physics as ghy_ref.py (validated against real Fortran dumps, FULL_FIDELITY_DELTAS.md D9); this
module operates on whole arrays of cells at once instead of one GhyColumn object per cell, using
jnp.where in place of Python if/else and fixed-size arrays plus masks in place of the plain-Python
port's per-cell variable-length slicing (variable active soil-layer count `n`, variable snow-layer
count `nsn`). See FULL_FIDELITY_PLAN.md's "JAX-vectorization of GHY" section for the scoping notes
this implementation follows, including two empirically-checked findings that changed the approach:
the soil-hydraulics bisection's "exact table hit" branch is common (~7% of real calls, not rare), and
naive dts=0/dts~0 padding of the per-cell variable substep count is NOT a safe no-op (it either
crashes or silently perturbs accumulator outputs) -- this module instead gates each substep with an
explicit per-lane "run vs. keep prior state" mask around the WHOLE substep body.

Because the only real-Fortran ground truth available is the FINAL state after the whole per-cell
substep loop (see ghy_compare.py's `refs`, which has no intermediate per-substep dump), individual
functions here (reth, retp, hydra, ...) are validated by cross-check against the already-validated
plain-Python ghy_ref.py on the same inputs, not against Fortran directly; only the full `advnc`
pipeline is checked against real Fortran.

State is a dict of jnp arrays with a leading batch axis (N,). ibv (0=bare,1=vegetated) and k (soil
layer, 0=canopy..NGM=6) are always full-size axes; `n` (active layer count, per-cell, invariant
across a cell's own timesteps -- set from the static soil-depth config) and `nsn` (active snow-layer
count per ibv, dynamic within a cell, only ever 0/1/3) are handled with masks, not dynamic shapes.
"""
import jax.numpy as jnp

import ghy_ref as R   # reuse constants and the once-computed THM/HLM/XKLM/DLM soil tables verbatim

TFRZ, SHA, LHE, LHM, RHOW = R.TFRZ, R.SHA, R.LHE, R.LHM, R.RHOW
STBO, GRAV = R.STBO, R.GRAV
SHW, SHI, FSN, ELH, SHV, PRFR = R.SHW, R.SHI, R.FSN, R.ELH, R.SHV, R.PRFR
NGM, IMT, NLSN, NTH = R.NGM, R.IMT, R.NLSN, R.NTH
MRAT, RVAP = R.MRAT, R.RVAP
SHC_SOIL_TEXTURE = jnp.asarray(R.SHC_SOIL_TEXTURE)
RHO_ICE, RHO_WATER, RHO_FRESH_SNOW = R.RHO_ICE, R.RHO_WATER, R.RHO_FRESH_SNOW
LAT_FUSION, LAT_EVAP = R.LAT_FUSION, R.LAT_EVAP
MAX_FRACT_WATER, EPS_SNOW, MIN_SNOW_THICKNESS, MIN_FRACT_COVER = (
    R.MAX_FRACT_WATER, R.EPS_SNOW, R.MIN_SNOW_THICKNESS, R.MIN_FRACT_COVER)
TOTAL_NL = R.TOTAL_NL
THM = jnp.asarray(R.THM)   # (NTH+1, IMT-1)
HLM = jnp.asarray(R.HLM)   # (NTH+1,)
XKLM = jnp.asarray(R.XKLM)
DLM = jnp.asarray(R.DLM)


def _safe_div(a, b):
    return a / jnp.where(b == 0.0, 1.0, b)


def _dotn(a, b, n):
    """Explicit left-to-right accumulation over the last axis (size n) of a*b, broadcasting a and b
    first. Verified empirically (200k random trials, 0 mismatches) to match Python's `sum(a[i]*b[i]
    for i in range(n))` bit-for-bit; jnp.sum/einsum do NOT (confirmed ~29% mismatch rate on the same
    trials, from XLA's reduction using a different accumulation order) -- since these are tiny (4-5
    term) dot products with real Fortran-matching precedent (ghy_ref.py's plain Python loops), forcing
    the same left-to-right order here is a real correctness fix, not premature micro-optimization."""
    a, b = jnp.broadcast_arrays(a, b)
    out = a[..., 0] * b[..., 0]
    for i in range(1, n):
        out = out + a[..., i] * b[..., i]
    return out


def qsat(tm, lh, pr):
    _QA = 6.108 * MRAT
    _QB = 1.0 / (RVAP * TFRZ)
    _QC = 1.0 / RVAP
    return _QA * jnp.exp(lh * (_QB - _QC / jnp.maximum(130.0, tm))) / pr


def dqsatdt(tm, lh):
    _QC = 1.0 / RVAP
    return lh * _QC / (tm * tm)


def init_static(dz, q, qk, fb, fv):
    """dz: (N,NGM); q,qk: (N,IMT,NGM); fb,fv: (N,) bare/vegetated area fractions. Returns dict: n,
    zb(N,NGM+1), zc(N,NGM), thets/thetm/shc/ws (N,NGM+1,2) -- k=0 slot left at 0 here (canopy values
    are set by the caller from Ent's exports), process_bare/process_vege (N,) bool."""
    N = dz.shape[0]
    active = dz > 0.0                                  # (N, NGM) -- Fortran's "break on first dz<=0"
    idx = jnp.arange(NGM)[None, :]
    # first non-active index (or NGM if all active); n = that index (count of active layers)
    first_inactive = jnp.where(active, NGM, idx)
    n = jnp.min(first_inactive, axis=-1)                # (N,) int, in [0, NGM]

    dz_pad = jnp.where(jnp.arange(NGM)[None, :] < n[:, None], dz, 0.0)
    zb = jnp.concatenate([jnp.zeros((N, 1)), -jnp.cumsum(dz_pad, axis=-1)], axis=-1)   # (N, NGM+1)
    zc = 0.5 * (zb[:, :-1] + zb[:, 1:])                                                # (N, NGM)

    q_t = q.transpose(0, 2, 1)                          # (N, NGM, IMT)
    ts_ = _dotn(q_t[..., :IMT - 1], THM[0, :], IMT - 1)
    tm_ = _dotn(q_t[..., :IMT - 1], THM[NTH, :], IMT - 1)
    sh_ = _dotn(q_t, SHC_SOIL_TEXTURE, IMT)
    sh_ = (1.0 - ts_) * sh_ * dz                        # (N, NGM), Fortran k=1..NGM -> array pos 0..NGM-1
    ws_ = ts_ * dz

    kmask = (jnp.arange(NGM)[None, :] < n[:, None])     # (N, NGM) -- valid Fortran k=1..n
    zeros_col = jnp.zeros((N, 1))
    thets = jnp.stack([jnp.concatenate([zeros_col, jnp.where(kmask, ts_, 0.0)], axis=-1)] * 2, axis=-1)
    thetm = jnp.stack([jnp.concatenate([zeros_col, jnp.where(kmask, tm_, 0.0)], axis=-1)] * 2, axis=-1)
    shc = jnp.stack([jnp.concatenate([zeros_col, jnp.where(kmask, sh_, 0.0)], axis=-1)] * 2, axis=-1)
    ws = jnp.stack([jnp.concatenate([zeros_col, jnp.where(kmask, ws_, 0.0)], axis=-1)] * 2, axis=-1)
    return dict(n=n, zb=zb, zc=zc, thets=thets, thetm=thetm, shc=shc, ws=ws, dz=dz, q=q, qk=qk,
                kmask=kmask, process_bare=fb > 0.0, process_vege=fv > 0.0, fb=fb, fv=fv)


def reth(static, w, nsn, wsn, fr_snow, snowm):
    """GHY.f reth. w:(N,NGM+1,2); nsn:(N,2) int; wsn:(N,NLSN,2); fr_snow,snowm:(N,).
    Returns dict: theta(N,NGM+1,2), snowd(N,2), fw,fm,fd,fw0,fd0 (N,)."""
    dz = static['dz']; kmask = static['kmask']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)   # (N,2) -- an inactive ibv's theta
    # is simply never written by the real Fortran loop (gated by i_bare..i_vege) and stays at its
    # initial value (0, from GhyColumn.__init__'s np.zeros) for the whole advnc() call.
    dz_safe = jnp.where(dz == 0.0, 1.0, dz)
    theta_soil = w[:, 1:, :] / dz_safe[:, :, None]
    theta_soil = jnp.where(kmask[:, :, None] & ibv_active[:, None, :], theta_soil, 0.0)

    ws0_1 = static['ws'][:, 0, 1]
    w0_1 = w[:, 0, 1]
    theta01 = jnp.where((process_vege) & (ws0_1 > 0.0), _safe_div(w0_1, ws0_1) ** (2.0 / 3.0), 0.0)
    theta01 = jnp.minimum(theta01, 1.0)
    theta_k0 = jnp.stack([jnp.zeros_like(theta01), theta01], axis=-1)[:, None, :]
    theta = jnp.concatenate([theta_k0, theta_soil], axis=1)

    lsn_idx = jnp.arange(NLSN)[None, :, None]
    lsn_mask = lsn_idx < nsn[:, None, :]
    snowd = jnp.sum(jnp.where(lsn_mask, wsn, 0.0), axis=1) * fr_snow
    snowd = jnp.where(ibv_active, snowd, 0.0)   # inactive ibv: self.snowd[ibv] reset to 0, never added to

    fw = theta01
    fm = 1.0 - jnp.exp(-snowd[:, 1] / (snowm + 1e-12))
    fm = jnp.where(fm < 1e-3, 0.0, fm)
    fd = jnp.ones_like(fw)
    fw0 = fw
    fd0 = 1.0 - fw
    return dict(theta=theta, snowd=snowd, fw=fw, fm=fm, fd=fd, fw0=fw0, fd0=fd0)


def _xklh_constants():
    """GHY.f _xklh_warm's cell-INDEPENDENT part -- hcwtw/hcwti/hcwtb/hcwt/ba are pure functions of
    fixed constants (gabc, alamw, alami, alams), not of any per-cell state, even though ghy_ref.py
    recomputes them once per cell (in __init__) for structural convenience. Computed here in plain
    Python to guarantee bit-identical values to ghy_ref's own plain-Python computation."""
    gabc = [.125, .125, 1.0 - .125 - .125]
    alamw, alami = .573345, 2.1762
    alams = [8.8, 2.9, 2.9, .25]
    hcwtw = 1.0
    hcwti = sum(1.0 / (1.0 + (alami / alamw - 1.0) * g) for g in gabc) / 3.0
    hcwtb = 1.0
    hcwt = [sum(1.0 / (1.0 + (alams[i] / alamw - 1.0) * g) for g in gabc) / 3.0 for i in range(IMT - 1)]
    ba = .025 / alamw - 1.0
    return hcwtw, hcwti, hcwtb, jnp.asarray(hcwt), ba, jnp.asarray(alams)


HCWTW, HCWTI, HCWTB, HCWT, BA, ALAMS = _xklh_constants()
ALAMW, ALAMI, ALAMA, ALAMBR = .573345, 2.1762, .025, 2.9


def init_xklh_static(static):
    """GHY.f _xklh_warm's per-cell part (xsha, xsh). Adds them to `static`, shape (N,NGM+1,2), same
    Fortran-k-direct indexing as thets/thetm/etc, broadcast identically over both ibv (never gated by
    process_bare/process_vege -- ghy_ref computes this in __init__, before i_bare/i_vege even exist)."""
    N = static['dz'].shape[0]
    kmask = static['kmask']
    q_t = static['q'].transpose(0, 2, 1)[..., :IMT - 1]     # (N,NGM,IMT-1)
    xs = (1.0 - THM[0, :])[None, None, :] * q_t             # (N,NGM,IMT-1)
    xsha_soil = _dotn(xs, HCWT * ALAMS, IMT - 1)
    xsh_soil = _dotn(xs, HCWT, IMT - 1)
    xsha_soil = jnp.where(kmask, xsha_soil, 0.0)
    xsh_soil = jnp.where(kmask, xsh_soil, 0.0)
    xsha = jnp.broadcast_to(xsha_soil[:, :, None], xsha_soil.shape + (2,))
    xsh = jnp.broadcast_to(xsh_soil[:, :, None], xsh_soil.shape + (2,))
    zero_col = jnp.zeros((N, 1, 2))
    static = dict(static)
    static['xsha'] = jnp.concatenate([zero_col, xsha], axis=1)
    static['xsh'] = jnp.concatenate([zero_col, xsh], axis=1)
    return static


def xklh(static, w, fice, theta):
    """GHY.f xklh (soil thermal conductivity). Returns xkh, xkhm (N,NGM+1,2), Fortran-k-direct."""
    N = theta.shape[0]
    kmask = static['kmask']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    active = kmask[:, :, None] & ibv_active[:, None, :]

    thets = static['thets'][:, 1:, :]
    theta_k = theta[:, 1:, :]
    fice_k = fice[:, 1:, :]
    w_k = w[:, 1:, :]
    dz = static['dz']
    xsha = static['xsha'][:, 1:, :]
    xsh = static['xsh'][:, 1:, :]
    q_last = static['q'][:, IMT - 1, :]                      # (N,NGM), the "xb" texture fraction

    gaa = .298 * theta_k / (thets + 1e-6) + .035
    gca = 1.0 - 2.0 * gaa
    hcwta = (2.0 / (1.0 + BA * gaa) + 1.0 / (1.0 + BA * gca)) / 3.0
    dz_safe = jnp.where(dz == 0.0, 1.0, dz)[:, :, None]
    xw = w_k * (1.0 - fice_k) / dz_safe
    xi = w_k * fice_k / dz_safe
    xa = thets - theta_k
    xb = q_last[:, :, None]
    xnum = xw * HCWTW * ALAMW + xi * HCWTI * ALAMI + xa * hcwta * ALAMA + xsha + xb * HCWTB * ALAMBR
    xden = xw * HCWTW + xi * HCWTI + xa * hcwta + xsh + xb * HCWTB
    xkh_soil = jnp.where(active, xnum / jnp.where(xden == 0.0, 1.0, xden), 0.0)
    zero_col = jnp.zeros((N, 1, 2))
    xkh = jnp.concatenate([zero_col, xkh_soil], axis=1)       # (N,NGM+1,2)

    kk = jnp.arange(2, NGM + 1)                               # Fortran k=2..NGM
    zb_km1 = static['zb'][:, kk - 1]
    zc_km2 = static['zc'][:, kk - 2]
    zc_km1 = static['zc'][:, kk - 1]
    xkh_k = xkh[:, kk, :]
    xkh_km1 = xkh[:, kk - 1, :]
    denom = zc_km1 - zc_km2
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[:, :, None]
    xkhm_tail = ((zb_km1 - zc_km2)[:, :, None] * xkh_k + (zc_km1 - zb_km1)[:, :, None] * xkh_km1) / denom_safe
    active_tail = kmask[:, 1:, None] & ibv_active[:, None, :]  # position j=1..NGM-1 == Fortran k=2..NGM
    xkhm_tail = jnp.where(active_tail, xkhm_tail, 0.0)
    zero2 = jnp.zeros((N, 2, 2))
    xkhm = jnp.concatenate([zero2, xkhm_tail], axis=1)
    return dict(xkh=xkh, xkhm=xkhm)


_JCM = 6   # round(log2(NTH)) with NTH=64, a compile-time constant (not data-dependent)


def hydra(static, theta, fice):
    """GHY.f hydra: bisection into THM/HLM/XKLM/DLM. theta,fice: (N,NGM+1,2) (Fortran-k-direct
    indexing, position 0 = canopy/unused here). Returns h,d,xku,xkus (N,NGM+1,2, same indexing,
    position 0 always 0), xk (N,NGM+1,2, xk[:,0,:] unused/0, xk[:,k,:] valid for k=1..n), xkusa(N,2).

    Checked empirically against real data (FULL_FIDELITY_PLAN.md's GHY scoping note): the bisection's
    `exact` branch (a midpoint landing exactly on thr0) fires ~7% of the time, not rarely -- so it is
    handled here as a normal per-iteration outcome (frozen-state tracking across all 6 iterations),
    not approximated away."""
    N = theta.shape[0]
    kmask = static['kmask']                                            # (N,NGM), Fortran k=1..n
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)      # (N,2)
    active = kmask[:, :, None] & ibv_active[:, None, :]                 # (N,NGM,2)

    thets = static['thets'][:, 1:, :]     # (N,NGM,2), position j = Fortran k=j+1
    thetm = static['thetm'][:, 1:, :]
    theta_k = theta[:, 1:, :]
    fice_k = fice[:, 1:, :]
    q_soil = static['q'][:, :IMT - 1, :].transpose(0, 2, 1)             # (N,NGM,IMT-1)

    thr1 = thets                                       # initial bisection bounds: [thetm, thets]
    thr2 = thetm
    thr0 = jnp.minimum(thets, jnp.maximum(thetm, theta_k))   # bisection TARGET: theta clamped into that range
    shape = thr0.shape
    q_bcast = jnp.broadcast_to(q_soil[:, :, None, :], shape + (IMT - 1,))    # (N,NGM,2,IMT-1)
    j1 = jnp.zeros(shape, dtype=jnp.int32)
    j2 = jnp.full(shape, NTH, dtype=jnp.int32)
    frozen = jnp.zeros(shape, dtype=bool)

    for _ in range(_JCM):
        j = (j1 + j2) // 2
        THM_g = THM[j]                                                  # (N,NGM,2,IMT-1)
        thr = _dotn(THM_g, q_bcast, IMT - 1)                             # (N,NGM,2)
        d = thr - thr0
        is_exact = (d == 0.0) & (~frozen)
        go_lo = (d < 0.0) & (~frozen) & (~is_exact)
        go_hi = (d > 0.0) & (~frozen) & (~is_exact)
        j2 = jnp.where(go_lo, j, j2); thr2 = jnp.where(go_lo, thr, thr2)
        j1 = jnp.where(go_hi, j, j1); thr1 = jnp.where(go_hi, thr, thr1)
        j1 = jnp.where(is_exact, j, j1)
        thr1 = jnp.where(is_exact, thr0, thr1)
        thr2 = jnp.where(is_exact, -10.0, thr2)
        frozen = frozen | is_exact

    ith = j1
    thr1_m_thr2 = jnp.where(thr1 == thr2, 1.0, thr1 - thr2)   # guard (inactive lanes only; see kmask)
    hl = (HLM[ith] * (thr0 - thr2) + HLM[j2] * (thr1 - thr0)) / thr1_m_thr2
    temp = (thr1 - thr0) / thr1_m_thr2

    XKLM_lo, XKLM_hi = XKLM[ith], XKLM[ith + 1]
    DLM_lo, DLM_hi = DLM[ith], DLM[ith + 1]
    d1 = _dotn(DLM_lo, q_bcast, IMT - 1); d2 = _dotn(DLM_hi, q_bcast, IMT - 1)
    xku1 = _dotn(XKLM_lo, q_bcast, IMT - 1); xku2 = _dotn(XKLM_hi, q_bcast, IMT - 1)
    xkus_noibv = _dotn(XKLM[0], q_soil, IMT - 1)                        # (N,NGM), ibv-independent formula
    xkus_k = jnp.broadcast_to(xkus_noibv[:, :, None], shape)

    dl = ((1.0 - temp) * d1 + temp * d2) * (1.0 - fice_k)
    xklu = ((1.0 - temp) * xku1 + temp * xku2) * (1.0 - fice_k)

    xkud = 2.78e-5
    zc0 = static['zc'][:, 0]                                            # (N,) Fortran zc(1) i.e. position 0
    xk1 = _dotn(static['qk'][:, :IMT - 1, 0], XKLM[0], IMT - 1)   # (N,), no ibv dependence
    xkl = xk1 / (1.0 + xk1 / (-zc0 * xkud))
    thets1 = thets[:, 0, :]; theta1 = theta_k[:, 0, :]; fice1 = fice_k[:, 0, :]   # (N,2), k=1 slice
    thets1_safe = jnp.where(thets1 == 0.0, 1.0, thets1)
    xkl = (1.0 - fice1 * theta1 / thets1_safe) * xkl[:, None]
    xkl = jnp.maximum(0.0, xkl)                                         # (N,2), the k=1 special case

    xk_rest = jnp.sqrt(jnp.maximum(xklu[:, :-1, :] * xklu[:, 1:, :], 0.0))   # k=3..NGM (uses k-1,k pairs)
    xk_soil = jnp.concatenate([xkl[:, None, :], xk_rest], axis=1)        # (N,NGM,2): k=1 special, k=2..NGM sqrt

    dz = static['dz']
    dz_total = jnp.sum(jnp.where(kmask, dz, 0.0), axis=1, keepdims=True)          # (N,1)
    dz_total_safe = jnp.where(dz_total == 0.0, 1.0, dz_total)
    xkusa_num = jnp.sum(jnp.where(kmask[:, :, None], xkus_k * dz[:, :, None], 0.0), axis=1)   # (N,2)
    xkusa = xkusa_num / dz_total_safe
    xkusa = jnp.where(ibv_active, xkusa, 0.0)

    grav_adj = static['zc'] * GRAV / 9.80665                            # (N,NGM)
    hl_adj = hl + grav_adj[:, :, None]

    def _pad(x):
        return jnp.where(active, x, 0.0)

    zero_col = jnp.zeros((N, 1, 2))
    h = jnp.concatenate([zero_col, _pad(hl_adj)], axis=1)
    d = jnp.concatenate([zero_col, _pad(dl)], axis=1)
    xku = jnp.concatenate([zero_col, _pad(xklu)], axis=1)
    xkus = jnp.concatenate([zero_col, _pad(xkus_k)], axis=1)
    xk = jnp.concatenate([zero_col, _pad(xk_soil)], axis=1)
    return dict(h=h, d=d, xku=xku, xkus=xkus, xkusa=xkusa, xk=xk)


def retp(static, w, ht):
    """GHY.f retp. Returns tp(N,NGM+1,2), fice(N,NGM+1,2) -- k=0 valid only for ibv=1 (kk=1-ibv in
    Fortran terms: bare starts at k=1, vegetated includes canopy k=0)."""
    shc = static['shc']; kmask = static['kmask']    # kmask: (N,NGM) for k=1..n
    k_full_mask = jnp.concatenate([jnp.zeros((kmask.shape[0], 1), dtype=bool), kmask], axis=1)  # (N,NGM+1), k=1..n
    is_k0 = jnp.arange(k_full_mask.shape[1]) == 0
    ibv0_mask = k_full_mask                              # bare: k=1..n only
    ibv1_mask = k_full_mask | is_k0[None, :]             # vegetated: k=0..n (adds canopy)
    ibv_mask = jnp.stack([ibv0_mask, ibv1_mask], axis=-1)
    active = ibv_mask & jnp.stack([static['process_bare'], static['process_vege']], axis=-1)[:, None, :]

    w_safe = jnp.where(w == 0.0, 1.0, w)
    cond_ice = (FSN * w + ht) < 0.0
    cond_warm = ht > 0.0
    cond_partial = w >= 1e-12

    tp_ice = (ht + w * FSN) / (shc + w * SHI)
    tp_warm = ht / (shc + w * SHW)
    fice_partial = jnp.where(cond_partial, -ht / (FSN * w_safe), 0.0)

    tp = jnp.where(cond_ice, tp_ice, jnp.where(cond_warm, tp_warm, 0.0))
    fice = jnp.where(cond_ice, 1.0, jnp.where(cond_warm, 0.0, fice_partial))
    tp = jnp.where(active, tp, 0.0)
    fice = jnp.where(active, fice, 0.0)
    return dict(tp=tp, fice=fice)
