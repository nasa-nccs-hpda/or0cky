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

`advnc()`'s per-cell substep loop uses `jax.lax.scan`, not a Python-level unroll -- an earlier
unrolled version was correct but measured as slower than plain Python in eager mode and
impractically slow to `jax.jit`-compile (duplicating the whole per-substep graph 11 times), fixed
by switching to scan (compiles the substep body once); see FULL_FIDELITY_DELTAS.md D15 for the
measurements. Every other bounded loop in this module (hydra's bisection, tridiag, relayer_12, ...)
stays a small Python unroll deliberately -- those are 2-6 iterations of a small body, not 11
iterations of a huge one, so they don't hit the same problem.
"""
import jax
jax.config.update("jax_enable_x64", True)  # float64 required to match ghy_ref.py's Fortran-derived ops
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


def evap_limits(static, w, theta, d, tp, fice, tsn1, nsn, wsn, fr_snow, dt, pr, betadl, cnc, ch, vs, rho,
                pres, qs, gusti, qprime, qm1, lai, fm):
    """GHY.f evap_limits(compute_evap=True) -- the only call site in advnc() always passes True, so
    that is the only path implemented. Returns a dict with everything the flux chain needs:
    evapb/evapbs/evapvw/evapvd/evapvs/evapvg (N,), evap_min(N,), devapbs_dt/devapvs_dt(N,),
    evapdl(N,NGM), betad/abetad/acna/acnc(N,), evap_max_out/fr_sat(N,)."""
    n = static['n']; kmask = static['kmask']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    fb, fv = static['fb'], static['fv']
    dz = static['dz']; thetm = static['thetm'][:, 1:, :]
    dt_safe = jnp.where(dt == 0.0, 1.0, dt)

    w_soil = w[:, 1:, :]
    evap_max_terms = jnp.where(kmask[:, :, None], (w_soil - dz[:, :, None] * thetm) / dt_safe[:, None, None], 0.0)
    evap_max = jnp.sum(evap_max_terms, axis=1)                          # (N,2)
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    evap_max = jnp.where(ibv_active, evap_max, 0.0)

    lsn_idx = jnp.arange(NLSN)[None, :, None]
    lsn_mask = lsn_idx < nsn[:, None, :]
    evap_max_snow = pr[:, None] + jnp.sum(jnp.where(lsn_mask, wsn, 0.0), axis=1) / dt_safe[:, None]
    evap_max_snow = jnp.where(ibv_active, evap_max_snow, 0.0)

    dz0 = dz[:, 0]; dz0_safe = jnp.where(dz0 == 0.0, 1.0, dz0)
    d1 = d[:, 1, :]; theta1 = theta[:, 1, :]; thetm1 = thetm[:, 0, :]
    evap_max_wet0 = evap_max[:, 0] + pr
    evap_max_dry0 = jnp.minimum(evap_max[:, 0],
                                2.467 * d1[:, 0] * (theta1[:, 0] - thetm1[:, 0]) / dz0_safe + pr)
    evap_max_wet0 = jnp.where(process_bare, evap_max_wet0, 0.0)
    evap_max_dry0 = jnp.where(process_bare, evap_max_dry0, 0.0)

    cna = ch * vs
    evap_max_vegsoil = jnp.minimum(evap_max[:, 1],
                                   2.467 * d1[:, 1] * (theta1[:, 1] - thetm1[:, 1]) / dz0_safe + pr)
    evap_max_vegsoil = jnp.where(process_vege, evap_max_vegsoil, 0.0)
    evap_max_wet1 = jnp.where(process_vege, w[:, 0, 1] / dt_safe, 0.0)

    betad_raw = jnp.sum(jnp.where(kmask, betadl, 0.0), axis=1)
    betad = jnp.where(betad_raw < 1e-12, 0.0, betad_raw)
    betad = jnp.where(process_vege, betad, 0.0)
    abetad = betad
    acna = jnp.where(process_vege, cna, 0.0)
    acnc = jnp.where(process_vege, cnc, 0.0)
    betat = cnc / (cnc + cna + 1e-12)
    tp01 = tp[:, 0, 1]
    pot_evap_can = betat * (rho / RHOW) * ch * (vs * (qsat(tp01 + TFRZ, LHE, pres) - qs) - gusti * qprime)

    betad_safe = jnp.where(betad == 0.0, 1.0, betad)
    dry_terms = jnp.minimum(pot_evap_can[:, None] * betadl / betad_safe[:, None],
                            (w_soil[:, :, 1] - dz * thetm[:, :, 1]) / dt_safe[:, None])
    dry_terms = jnp.where(kmask, dry_terms, 0.0)
    cond_dry = (betad > 0.0) & (pot_evap_can > 0.0) & process_vege
    evap_max_dry1 = jnp.where(cond_dry, jnp.sum(dry_terms, axis=1), 0.0)

    fr_snow0, fr_snow1 = fr_snow[:, 0], fr_snow[:, 1]
    theta01 = theta[:, 0, 1]
    evap_max_sat = fb * fr_snow0 * evap_max_snow[:, 0]
    evap_max_nsat = fb * (1.0 - fr_snow0) * evap_max_dry0
    vege_sat = fv * (fr_snow1 * fm * evap_max_snow[:, 1] + (1.0 - fr_snow1 * fm) * theta01 * evap_max_wet1)
    vege_nsat = fv * ((1.0 - fr_snow1 * fm) * (1.0 - theta01) * evap_max_dry1)
    evap_max_sat = evap_max_sat + jnp.where(process_vege, vege_sat, 0.0)
    evap_max_nsat = evap_max_nsat + jnp.where(process_vege, vege_nsat, 0.0)
    fr_sat = fb * fr_snow0 + jnp.where(process_vege, fv * (fr_snow1 * fm + (1.0 - fr_snow1 * fm) * theta01), 0.0)

    qm1dt = .001 * qm1 / dt_safe
    evap_min = -qm1dt
    qb = qsat(tp[:, 1, 0] + TFRZ, LHE, pres)
    qv = qsat(tp01 + TFRZ, LHE, pres)
    qbs = qsat(tsn1[:, 0] + TFRZ, LHE, pres)
    qvs = qsat(tsn1[:, 1] + TFRZ, LHE, pres)
    qvg = qsat(tp[:, 1, 1] + TFRZ, LHE, pres)
    rho3 = rho / RHOW
    v_qprime = gusti * qprime
    epb = rho3 * ch * (vs * (qb - qs) - v_qprime)
    epbs = rho3 * ch * (vs * (qbs - qs) - v_qprime)
    epv = rho3 * ch * (vs * (qv - qs) - v_qprime)
    epvs = rho3 * ch * (vs * (qvs - qs) - v_qprime)
    fw = theta01; fd = jnp.ones_like(fw)   # matches reth's fw,fd (GHY_FD_1_HACK -> fd always 1)
    epv1 = epv * (1.0 - fw) / (fd + 1e-12)
    ch_dense_veg = 0.01 * cna
    eta = jnp.exp(-(lai))
    ch_vg = ch * eta + ch_dense_veg * (1.0 - eta)
    epvg = rho3 * ch_vg * (vs * (qvg - qs) - v_qprime)

    evapb = jnp.where(process_bare, jnp.maximum(jnp.minimum(epb, evap_max_dry0), -qm1dt), 0.0)
    evapbs = jnp.where(process_bare, jnp.maximum(jnp.minimum(epbs, evap_max_snow[:, 0]), -qm1dt), 0.0)

    evapvw_raw = jnp.maximum(jnp.minimum(epv, evap_max_wet1), -qm1dt)
    evapvd_raw = jnp.maximum(jnp.minimum(epv1, evap_max_dry1), 0.0)
    evapvs_raw = jnp.maximum(jnp.minimum(epvs, evap_max_snow[:, 1]), -qm1dt)
    evapvg_a = jnp.minimum(epvg, evap_max_vegsoil)
    evapvg_b = jnp.minimum(evapvg_a, evap_max[:, 1] - evapvd_raw * fd)
    evapvg_c = jnp.minimum(evapvg_b, epv - evapvd_raw * fd - evapvw_raw * fw)
    evapvg_raw = jnp.maximum(evapvg_c, 0.0)
    # GHY.f: "if evapvw<0: fw=1,fd=0" -- an fw/fd MUTATION that carries forward into every later
    # substep method (drip_from_canopy, flg, flhg, runoff, fllmt, apply_fluxes all read self.fw/fd).
    cond_wet_override = process_vege & (evapvw_raw < 0.0)
    fw = jnp.where(cond_wet_override, 1.0, fw)
    fd = jnp.where(cond_wet_override, 0.0, fd)
    evapvw = jnp.where(process_vege, evapvw_raw, 0.0)
    evapvd = jnp.where(process_vege, evapvd_raw, 0.0)
    evapvs = jnp.where(process_vege, evapvs_raw, 0.0)
    evapvg = jnp.where(process_vege, evapvg_raw, 0.0)

    devapbs_dt = rho3 * cna * qsat(tsn1[:, 0] + TFRZ, LHE, pres) * dqsatdt(tsn1[:, 0] + TFRZ, LHE)
    devapvs_dt = rho3 * cna * qsat(tsn1[:, 1] + TFRZ, LHE, pres) * dqsatdt(tsn1[:, 1] + TFRZ, LHE)

    evapdl = jnp.where(kmask & (betad > 0.0)[:, None], evapvd[:, None] * betadl / betad_safe[:, None], 0.0)

    return dict(evapb=evapb, evapbs=evapbs, evapvw=evapvw, evapvd=evapvd, evapvs=evapvs, evapvg=evapvg,
                evap_min=evap_min, devapbs_dt=devapbs_dt, devapvs_dt=devapvs_dt, evapdl=evapdl,
                betad=betad, abetad=abetad, acna=acna, acnc=acnc, evap_max_out=evap_max_nsat,
                fr_sat=fr_sat, fw=fw, fd=fd)


def sensible_heat(tp, tsn1, ts, vs, ch, rho, gusti, tprime):
    """GHY.f sensible_heat (SNSH_VEG_GROUND not defined -> eta=0). Returns snshg,snshv,snshs (N,2),
    dsnsh_dt (N,)."""
    cna = ch * vs
    v_tprime = gusti * tprime
    snshg0 = SHA * rho * ch * (vs * (tp[:, 1, 0] - ts + TFRZ) - v_tprime)
    snshg1 = SHA * rho * ch * (vs * (tp[:, 1, 1] - ts + TFRZ) - v_tprime) * 0.0   # eta=0
    snshg = jnp.stack([snshg0, snshg1], axis=-1)
    snshv1 = SHA * rho * ch * (vs * (tp[:, 0, 1] - ts + TFRZ) - v_tprime) * 1.0   # (1-eta)=1
    snshv = jnp.stack([jnp.zeros_like(snshv1), snshv1], axis=-1)
    snshs0 = SHA * rho * ch * (vs * (tsn1[:, 0] - ts + TFRZ) - v_tprime)
    snshs1 = SHA * rho * ch * (vs * (tsn1[:, 1] - ts + TFRZ) - v_tprime)
    snshs = jnp.stack([snshs0, snshs1], axis=-1)
    dsnsh_dt = SHA * rho * cna
    return dict(snshg=snshg, snshv=snshv, snshs=snshs, dsnsh_dt=dsnsh_dt)


def drip_from_canopy(static, w, htpr, htprs, pr, prs, evapvw, fw, fm, fr_snow, fd0, dts, tp):
    """GHY.f drip_from_canopy. Returns dripw,htdripw,drips,htdrips,dripw_scale (N,2)."""
    process_vege = static['process_vege']
    ws01 = static['ws'][:, 0, 1]
    w01 = w[:, 0, 1]
    snowf = jnp.where(htpr < 0.0, jnp.minimum(-htpr / FSN, pr), 0.0)
    snowfs = jnp.where(htprs < 0.0, jnp.minimum(-htprs / FSN, prs), 0.0)

    can_evap = evapvw * fw * (1.0 - fm * fr_snow[:, 1])
    ptmps = jnp.maximum(prs - snowfs - can_evap, 0.0)
    ptmp = pr - prs - (snowf - snowfs)
    pr_dry0 = fd0 * ptmps
    wc_add0 = ws01 - w01
    wc_new0 = w01 + jnp.minimum(pr_dry0 * dts, wc_add0)
    ws01_safe = jnp.where(ws01 == 0.0, 1.0, ws01)
    fw_new = jnp.where(ws01 > 1e-12, _safe_div(wc_new0, ws01) ** (2.0 / 3.0), 0.0)
    fd_new = 1.0 - fw_new
    dts_safe = jnp.where(dts == 0.0, 1.0, dts)
    dr_scale = ptmps - (wc_new0 - w01) / dts_safe
    tau_storm = 3600.0
    f_prev_wet = 1.0 - (dts / tau_storm)
    pr_dry1 = jnp.where(fw_new > PRFR, (1.0 - f_prev_wet) * fd_new * ptmp,
                        (1.0 - f_prev_wet * fw_new / PRFR) * fd_new * ptmp)
    wc_add1 = (1.0 - f_prev_wet) * PRFR * (ws01 - wc_new0)
    wc_new1 = wc_new0 + jnp.minimum(pr_dry1 * dts, wc_add1)
    dr = ptmp - (wc_new1 - w01) / dts_safe
    dr = jnp.minimum(dr, pr - snowf - can_evap)
    dr = jnp.maximum(dr, pr - snowf - can_evap - (ws01 - w01) / dts_safe)
    dr = jnp.maximum(dr, 0.0)

    dripw1 = jnp.where(process_vege, dr, 0.0)
    dripw_scale1 = jnp.where(process_vege, dr_scale, 0.0)
    htdripw1 = jnp.where(process_vege, SHW * dr * jnp.maximum(tp[:, 0, 1], 0.0), 0.0)
    drips1 = jnp.where(process_vege, snowf, 0.0)
    htdrips1 = jnp.where(process_vege, jnp.minimum(htpr, 0.0), 0.0)

    drips0 = snowf
    htdrips0 = jnp.minimum(htpr, 0.0)
    dripw0 = pr - drips0
    dripw_scale0 = prs - snowfs
    htdripw0 = htpr - htdrips0

    dripw = jnp.stack([dripw0, dripw1], axis=-1)
    htdripw = jnp.stack([htdripw0, htdripw1], axis=-1)
    drips = jnp.stack([drips0, drips1], axis=-1)
    htdrips = jnp.stack([htdrips0, htdrips1], axis=-1)
    dripw_scale = jnp.stack([dripw_scale0, dripw_scale1], axis=-1)
    return dict(dripw=dripw, htdripw=htdripw, drips=drips, htdrips=htdrips, dripw_scale=dripw_scale)


def fl(static, h, xk):
    """GHY.f fl (soil moisture diffusion flux). f uses the OFFSET convention (position j = Fortran
    f(j+1)), size (N,NGM+1,2); position 0 is a placeholder filled in later by flg(). Returns f, xinfc."""
    n = static['n']; zc = static['zc']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    N = h.shape[0]
    k_range = jnp.arange(2, NGM + 1)          # Fortran k=2..NGM
    pos = k_range - 1                          # offset position (=k-1), values 1..NGM-1
    xk_k = xk[:, k_range, :]
    h_km1 = h[:, k_range - 1, :]
    h_k = h[:, k_range, :]
    zc_km2 = zc[:, k_range - 2]; zc_km1 = zc[:, k_range - 1]
    denom = zc_km2 - zc_km1
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[:, :, None]
    f_tail = -xk_k * (h_km1 - h_k) / denom_safe

    f = jnp.zeros((N, NGM + 1, 2))
    f = f.at[:, 1:NGM, :].set(f_tail)
    pos_full = jnp.arange(NGM + 1)[None, :]
    is_boundary = pos_full == n[:, None]
    beyond = pos_full > n[:, None]
    f = jnp.where(is_boundary[:, :, None], 0.0, f)
    f = jnp.where(beyond[:, :, None], 0.0, f)
    f = jnp.where(ibv_active[:, None, :], f, 0.0)

    zc0 = zc[:, 0]
    xinfc = xk[:, 1, :] * h[:, 1, :] / zc0[:, None]
    return dict(f=f, xinfc=xinfc)


def flh(static, xkhm, tp, f, geothermal_heat):
    """GHY.f flh (soil heat diffusion + upwind advection). fh: same OFFSET convention as f."""
    n = static['n']; zc = static['zc']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    N = tp.shape[0]
    k_range = jnp.arange(2, NGM + 1)
    pos = k_range - 1
    xkhm_k = xkhm[:, k_range, :]
    tp_km1 = tp[:, k_range - 1, :]
    tp_k = tp[:, k_range, :]
    zc_km2 = zc[:, k_range - 2]; zc_km1 = zc[:, k_range - 1]
    denom = zc_km2 - zc_km1
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[:, :, None]
    val = -xkhm_k * (tp_km1 - tp_k) / denom_safe
    f_km1 = f[:, pos, :]
    upwind = jnp.where(f_km1 > 0.0, f_km1 * tp_k * SHW, f_km1 * tp_km1 * SHW)
    val = val + upwind

    fh = jnp.zeros((N, NGM + 1, 2))
    fh = fh.at[:, 1:NGM, :].set(val)
    pos_full = jnp.arange(NGM + 1)[None, :]
    is_boundary = pos_full == n[:, None]
    beyond = pos_full > n[:, None]
    fh = jnp.where(is_boundary[:, :, None], geothermal_heat[:, None, None], fh)
    fh = jnp.where(beyond[:, :, None], 0.0, fh)
    fh = jnp.where(ibv_active[:, None, :], fh, 0.0)
    return dict(fh=fh)


def flg(static, f, flmlt, flmlt_scale, dripw, drips, evapb, evapvg, fr_snow, pr, evapvw, evapbs,
       evapvs, evapvd, fw, fd, fm, irrig=None):
    """GHY.f flg. Fills the OFFSET position 0 (Fortran f(1)) of `f`, plus canopy flux `fc`(N,2) and
    evap_tot(N,2). irrig is always 0 in this rundeck (ghy_ref hardcodes self.irrig=zeros in __init__
    regardless of the forcing dict's irrig value -- dead input, not used, so omitted here)."""
    process_bare, process_vege = static['process_bare'], static['process_vege']
    # D136: GHY.f flg subtracts irrig(ibv) (irrig(1)=0, irrig(2)=irrig_in/fv); irrig is (N,2) or None
    ir0 = 0.0 if irrig is None else irrig[:, 0]
    ir1 = 0.0 if irrig is None else irrig[:, 1]
    f0_bare = (-flmlt[:, 0] * fr_snow[:, 0] - flmlt_scale[:, 0]
              - (dripw[:, 0] - evapb) * (1.0 - fr_snow[:, 0]) - ir0)
    f0_vege = (-flmlt[:, 1] * fr_snow[:, 1] - flmlt_scale[:, 1]
              - (dripw[:, 1] - evapvg) * (1.0 - fr_snow[:, 1]) - ir1)
    f0 = jnp.stack([jnp.where(process_bare, f0_bare, f[:, 0, 0]),
                    jnp.where(process_vege, f0_vege, f[:, 0, 1])], axis=-1)
    f = f.at[:, 0, :].set(f0)

    fc0 = jnp.where(process_vege, -pr + evapvw * fw * (1.0 - fm * fr_snow[:, 1]), 0.0)
    fc1 = jnp.where(process_vege, -dripw[:, 1] - drips[:, 1], 0.0)
    fc = jnp.stack([fc0, fc1], axis=-1)

    evap_tot0 = evapb * (1.0 - fr_snow[:, 0]) + evapbs * fr_snow[:, 0]
    evap_tot1 = ((evapvw * fw + evapvd * fd) * (1.0 - fr_snow[:, 1] * fm)
                + evapvs * fr_snow[:, 1] * fm + evapvg * (1.0 - fr_snow[:, 1]))
    evap_tot = jnp.stack([evap_tot0, evap_tot1], axis=-1)
    return dict(f=f, fc=fc, evap_tot=evap_tot)


def runoff(static, w, f, xinfc, dripw, dripw_scale, evapb, evapvg, fr_snow, pr, xku, sl):
    """GHY.f runoff. rnf(N,2), rnff(N,NGM,2) -- rnff uses the REDUCED convention (position j =
    Fortran k=j+1), same alignment as kmask/thets, no extra padding slot (ghy_ref's own array has an
    unused trailing slot from its dynamic n+1 sizing; nothing ever reads it)."""
    n = static['n']; dz = static['dz']; ws = static['ws']; kmask = static['kmask']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    N = w.shape[0]
    rosmp = 8.0

    rnf_default = jnp.broadcast_to(pr[:, None], (N, 2))   # irrig always 0 in this rundeck

    f1_conv0 = jnp.where(process_bare, -(dripw[:, 0] - evapb - dripw_scale[:, 0]), 0.0)
    f1_conv1 = jnp.where(process_vege, -(dripw[:, 1] - evapvg - dripw_scale[:, 1]), 0.0)
    f1_conv = jnp.stack([f1_conv0, f1_conv1], axis=-1)

    w1 = w[:, 1, :]; ws1 = ws[:, 1, :]
    satfrac = jnp.minimum(_safe_div(w1, ws1) ** rosmp, 0.6)
    f0 = f[:, 0, :]
    rnf_a = satfrac * jnp.maximum(-f0, 0.0)

    water_down1 = jnp.maximum(0.0, -f1_conv)
    safe1 = jnp.where(water_down1 <= 1e-30, 1.0, water_down1)
    term1 = (1.0 - fr_snow) * (1.0 - satfrac) * water_down1 * jnp.exp(-xinfc * PRFR / safe1)
    rnf_a = rnf_a + jnp.where(water_down1 > 1e-30, term1, 0.0)

    water_down2 = jnp.maximum(-f0 - water_down1, 0.0)
    safe2 = jnp.where(water_down2 <= 1e-30, 1.0, water_down2)
    term2 = (1.0 - satfrac) * water_down2 * jnp.exp(-xinfc / safe2)
    rnf_a = rnf_a + jnp.where(water_down2 > 1e-30, term2, 0.0)

    rnf = jnp.where(ibv_active, rnf_a, rnf_default)

    rnff = xku[:, 1:, :] * sl[:, None, None] * dz[:, :, None] / 100.0
    rnff = jnp.where(kmask[:, :, None] & ibv_active[:, None, :], rnff, 0.0)
    return dict(rnf=rnf, rnff=rnff)


def flhg(static, fh, tp, fhsng, fhsng_scale, htdripw, htdrips, evapb, evapvg, evapvw, evapvd, snshg, snshv,
        snshs, thrmsn, fr_snow, srht, trht, htpr, fw, fd, fm, htirrig=None):
    """GHY.f flhg. Takes the `fh` array flh() built (position 0 still a placeholder there, since flh()
    runs before flhg() in advnc()'s real order) and fills OFFSET position 0 (Fortran fh(1)), returning
    the merged fh -- same pattern as flg() merging into `f`. Also returns canopy heat flux `fch`(N,2)
    and thrm_tot/snsh_tot(N,2)."""
    process_bare, process_vege = static['process_bare'], static['process_vege']
    thrm_can = STBO * (tp[:, 0, 1] + TFRZ) ** 4
    thrm_soil0 = STBO * (tp[:, 1, 0] + TFRZ) ** 4
    thrm_soil1 = STBO * (tp[:, 1, 1] + TFRZ) ** 4

    fh0_bare = (-fhsng[:, 0] * fr_snow[:, 0] - fhsng_scale[:, 0]
               + (-htdripw[:, 0] + evapb * (ELH + SHV * tp[:, 1, 0]) + snshg[:, 0]
                  + thrm_soil0 - srht - trht) * (1.0 - fr_snow[:, 0])
               - (0.0 if htirrig is None else htirrig[:, 0]))
    fh0_vege = (-fhsng[:, 1] * fr_snow[:, 1] - fhsng_scale[:, 1]
               + (-htdripw[:, 1] + evapvg * (ELH + SHV * tp[:, 1, 1]) + snshg[:, 1]
                  + thrm_soil1 - thrm_can) * (1.0 - fr_snow[:, 1])
               - (0.0 if htirrig is None else htirrig[:, 1]))
    fh0 = jnp.stack([jnp.where(process_bare, fh0_bare, fh[:, 0, 0]),
                    jnp.where(process_vege, fh0_vege, fh[:, 0, 1])], axis=-1)
    fh = fh.at[:, 0, :].set(fh0)
    snsh_vapor10 = jnp.where(process_bare, evapb * SHV * tp[:, 1, 0], 0.0)
    snsh_vapor11 = jnp.where(process_vege, evapvg * SHV * tp[:, 1, 1], 0.0)

    fch0 = jnp.where(process_vege,
                     (-htpr + (evapvw * (ELH + SHV * tp[:, 0, 1]) * fw + snshv[:, 1] + thrm_can
                               - srht - trht + evapvd * (ELH + SHV * tp[:, 0, 1]) * fd)
                      * (1.0 - fm * fr_snow[:, 1])), 0.0)
    snsh_vapor01 = jnp.where(process_vege, (evapvw * fw + evapvd * fd) * SHV * tp[:, 0, 1], 0.0)
    fch1 = jnp.where(process_vege,
                     (-(thrm_can - thrm_soil1) * (1.0 - fr_snow[:, 1])
                      - (thrm_can - thrmsn[:, 1]) * fr_snow[:, 1] * (1.0 - fm)
                      - htdripw[:, 1] - htdrips[:, 1]), 0.0)
    fch = jnp.stack([fch0, fch1], axis=-1)

    thrm_tot0 = thrm_soil0 * (1.0 - fr_snow[:, 0]) + thrmsn[:, 0] * fr_snow[:, 0]
    thrm_tot1 = thrm_can * (1.0 - fr_snow[:, 1] * fm) + thrmsn[:, 1] * fr_snow[:, 1] * fm
    thrm_tot = jnp.stack([thrm_tot0, thrm_tot1], axis=-1)

    snsh_tot0 = (snshg[:, 0] + snsh_vapor10) * (1.0 - fr_snow[:, 0]) + snshs[:, 0] * fr_snow[:, 0]
    snsh_tot1 = ((snshv[:, 1] + snsh_vapor01) * (1.0 - fr_snow[:, 1] * fm)
                + snshs[:, 1] * fr_snow[:, 1] * fm
                + (snshg[:, 1] + snsh_vapor11) * (1.0 - fr_snow[:, 1]))
    snsh_tot = jnp.stack([snsh_tot0, snsh_tot1], axis=-1)
    return dict(fh=fh, fch=fch, thrm_tot=thrm_tot, snsh_tot=snsh_tot)


def fllmt(static, w, f, rnff, rnf, evapdl, fr_snow, fm, dts):
    """GHY.f fllmt (moisture-flux truncation to keep w within [thetm*dz, ws]). Mutates f, rnff, rnf.
    Three passes, transcribed in order: (1) a top-down k=n..2 sequential recurrence (each step reads
    the flux the previous step may have just written), (2) a fixed k=1 correction, (3) a k=1..n
    while-loop (bounded by NGM, frozen once rnf>=0 per lane -- same technique as hydra's bisection
    freeze and relayer_12's branch selection). Transcribed exactly as written, including the (odd but
    faithful) use of fr_snow[1]/fm inside the bare (ibv=0) formula -- it is multiplied by evapdl,
    which is always 0 for ibv=0, so it has no effect there, but is kept literal rather than "fixed"."""
    n = static['n']; dz = static['dz']; ws = static['ws']; thetm = static['thetm']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    trunc = 0.0
    dts_b = dts[:, None]
    evapdl_full = jnp.stack([jnp.zeros_like(evapdl), evapdl], axis=-1)   # (N,NGM,2), position=k-1
    fd_factor = (1.0 - fr_snow[:, 1] * fm)[:, None]

    for k in range(NGM, 1, -1):
        active_k = (k <= n)[:, None] & ibv_active
        w_k = w[:, k, :]
        f_k = f[:, k, :]; f_km1 = f[:, k - 1, :]; rnff_km1 = rnff[:, k - 1, :]
        evd = evapdl_full[:, k - 1, :]
        wn = w_k + (f_k - f_km1 - rnff_km1 - fd_factor * evd) * dts_b
        ws_k = ws[:, k, :]; thetm_k = thetm[:, k, :]; dz_km1 = dz[:, k - 1][:, None]

        cond1 = (wn - ws_k) > trunc
        f_km1_a = jnp.where(cond1, f_km1 + (wn - ws_k + trunc) / dts_b, f_km1)
        cond2 = (wn - dz_km1 * thetm_k) < trunc
        rnff_km1_a = jnp.where(cond2, rnff_km1 + (wn - dz_km1 * thetm_k - trunc) / dts_b, rnff_km1)
        cond3 = cond2 & (rnff_km1_a < 0.0)
        f_km1_a = jnp.where(cond3, f_km1_a + rnff_km1_a, f_km1_a)
        rnff_km1_a = jnp.where(cond3, 0.0, rnff_km1_a)

        f = f.at[:, k - 1, :].set(jnp.where(active_k, f_km1_a, f_km1))
        rnff = rnff.at[:, k - 1, :].set(jnp.where(active_k, rnff_km1_a, rnff_km1))

    evd1 = evapdl_full[:, 0, :]
    wn1 = w[:, 1, :] + (f[:, 1, :] - f[:, 0, :] - rnf - rnff[:, 0, :] - fd_factor * evd1) * dts_b
    ws1 = ws[:, 1, :]; thetm1 = thetm[:, 1, :]; dz0 = dz[:, 0][:, None]
    cond_a = (wn1 - ws1) > trunc
    rnf = jnp.where(cond_a & ibv_active, rnf + (wn1 - ws1 + trunc) / dts_b, rnf)
    cond_b = (wn1 - dz0 * thetm1) < trunc
    rnf = jnp.where(cond_b & ibv_active, rnf + (wn1 - dz0 * thetm1 - trunc) / dts_b, rnf)

    for k in range(1, NGM + 1):
        active_lane = (rnf < 0.0) & (k <= n)[:, None] & ibv_active
        if k > 1:
            evdl = evapdl_full[:, k - 1, :]
            dz_km1 = dz[:, k - 1][:, None]
            f_k = f[:, k, :]; w_k = w[:, k, :]; thetm_k = thetm[:, k, :]
            f_km1 = f[:, k - 1, :]; rnff_km1 = rnff[:, k - 1, :]
            dflux = f_k + (w_k - dz_km1 * thetm_k) / dts_b - f_km1 - rnff_km1 - fd_factor * evdl
            f_km1_new = f_km1 - rnf
            rnf_new = rnf + jnp.minimum(-rnf, dflux)
            f = f.at[:, k - 1, :].set(jnp.where(active_lane, f_km1_new, f_km1))
            rnf = jnp.where(active_lane, rnf_new, rnf)
        rnff_km1_cur = rnff[:, k - 1, :]
        drnf = jnp.minimum(-rnf, rnff_km1_cur)
        rnf2 = rnf + drnf
        rnff_km1_new = rnff_km1_cur - drnf
        rnf = jnp.where(active_lane, rnf2, rnf)
        rnff = rnff.at[:, k - 1, :].set(jnp.where(active_lane, rnff_km1_new, rnff_km1_cur))
    rnf = jnp.maximum(rnf, 0.0)
    return dict(f=f, rnff=rnff, rnf=rnf)


def apply_fluxes(static, w, ht, f, fh, fc, fch, rnf, rnff, tp, evapdl, fr_snow, fm, dts):
    """GHY.f apply_fluxes. Returns updated w, ht (N,NGM+1,2)."""
    n = static['n']; dz = static['dz']; ws = static['ws']; thetm = static['thetm']
    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    kmask = static['kmask']
    dts_b = dts[:, None]
    evapdl_full = jnp.stack([jnp.zeros_like(evapdl), evapdl], axis=-1)
    fd_factor = (1.0 - fr_snow[:, 1] * fm)[:, None]

    w01 = w[:, 0, 1] + (fc[:, 1] - fc[:, 0]) * dts
    ht01 = ht[:, 0, 1] + (fch[:, 1] - fch[:, 0]) * dts
    w = w.at[:, 0, 1].set(jnp.where(process_vege, w01, w[:, 0, 1]))
    ht = ht.at[:, 0, 1].set(jnp.where(process_vege, ht01, ht[:, 0, 1]))

    w1 = w[:, 1, :] - rnf * dts_b
    ht1 = ht[:, 1, :] - SHW * jnp.maximum(tp[:, 1, :], 0.0) * rnf * dts_b
    w = w.at[:, 1, :].set(jnp.where(ibv_active, w1, w[:, 1, :]))
    ht = ht.at[:, 1, :].set(jnp.where(ibv_active, ht1, ht[:, 1, :]))

    k_idx = jnp.arange(1, NGM + 1)                    # Fortran k=1..NGM (direct positions)
    w_k = w[:, k_idx, :]; f_k = f[:, k_idx, :]; f_km1 = f[:, k_idx - 1, :]; rnff_km1 = rnff[:, k_idx - 1, :]
    evdl_k = evapdl_full[:, k_idx - 1, :]
    w_new = w_k + (f_k - f_km1 - rnff_km1 - fd_factor[:, None, :] * evdl_k) * dts_b[:, None, :]
    fh_k = fh[:, k_idx, :]; fh_km1 = fh[:, k_idx - 1, :]
    ht_new = (ht[:, k_idx, :] + (fh_k - fh_km1 - SHW * jnp.maximum(tp[:, k_idx, :], 0.0) * rnff_km1)
             * dts_b[:, None, :])
    active_k = kmask[:, :, None] & ibv_active[:, None, :]
    w = w.at[:, 1:, :].set(jnp.where(active_k, w_new, w[:, 1:, :]))
    ht = ht.at[:, 1:, :].set(jnp.where(active_k, ht_new, ht[:, 1:, :]))

    w = w.at[:, 0, 1].set(jnp.where((process_vege) & (w[:, 0, 1] < 0.0), 0.0, w[:, 0, 1]))

    # final clamp: GHY.f loops `for ibv in (0,1)` here, NOT gated by process_bare/vege (unlike
    # everything else in this function) -- an inactive ibv's stale w still gets clamped into
    # [dz*thetm, ws], which is harmless (those bounds are static, computed identically either way)
    # but must not be skipped to stay faithful to the real control flow.
    dz_padded = jnp.concatenate([jnp.zeros((w.shape[0], 1)), dz], axis=1)[:, :, None]
    w_clamped = jnp.clip(w, thetm * dz_padded, ws)
    clamp_mask = jnp.concatenate([jnp.zeros((w.shape[0], 1), dtype=bool), kmask], axis=1)[:, :, None]
    w = jnp.where(jnp.broadcast_to(clamp_mask, w.shape), w_clamped, w)
    return dict(w=w, ht=ht)


MIN_SNOW_THICKNESS_ = MIN_SNOW_THICKNESS   # (kept name-matched to ghy_ref for cross-reading)


def snow_pass_water(wsn, hsn, dz, nl, water_down, heat_down):
    """SNOW.f pass_water, batched over cells with a fixed TOTAL_NL=3 layer axis; `nl` gates which
    layers are active (n<nl). This is a genuine top-down SEQUENTIAL cascade (each layer's water_down/
    heat_down output feeds the next), so it is a Python-unrolled loop over the 3 fixed layers, not a
    single vectorized expression -- matching the same technique used for GHY's other small bounded
    recurrences. Returns dz, wsn, hsn (mutated), water_down, heat_down (the final carry, N,)."""
    N = wsn.shape[0]
    wsn = wsn + 0.0; hsn = hsn + 0.0; dz = dz + 0.0
    for n in range(TOTAL_NL):
        active_n = n < nl
        ice_old = jnp.minimum(wsn[:, n], -hsn[:, n] / LAT_FUSION)
        wsn_n = wsn[:, n] + water_down
        hsn_n = hsn[:, n] + heat_down
        wd_next = jnp.zeros(N); hd_next = jnp.zeros(N)

        cond_empty = (hsn_n >= 0.0) | (wsn_n <= 0.0)
        wd_a = wsn_n; hd_a = hsn_n
        wsn_a = jnp.zeros(N); hsn_a = jnp.zeros(N); dz_a = jnp.zeros(N)

        cond_partial = hsn_n > -wsn_n * LAT_FUSION
        ice_b = -hsn_n / LAT_FUSION
        free_water = wsn_n - ice_b
        wd_b = jnp.maximum(0.0, free_water - ice_b * MAX_FRACT_WATER)
        wsn_b = wsn_n - wd_b
        hsn_b = hsn_n
        dz_b = jnp.minimum(dz[:, n], ice_b * RHO_WATER / RHO_FRESH_SNOW)
        hd_b = jnp.zeros(N)

        ice_old_safe = jnp.where(ice_old == 0.0, 1.0, ice_old)
        dz_c = jnp.where(wsn_n + EPS_SNOW < ice_old, dz[:, n] * wsn_n / ice_old_safe, dz[:, n])
        dz_c = jnp.minimum(dz_c, wsn_n * RHO_WATER / RHO_FRESH_SNOW)
        wsn_c = wsn_n; hsn_c = hsn_n; wd_c = jnp.zeros(N); hd_c = jnp.zeros(N)

        wsn_new = jnp.where(cond_empty, wsn_a, jnp.where(cond_partial, wsn_b, wsn_c))
        hsn_new = jnp.where(cond_empty, hsn_a, jnp.where(cond_partial, hsn_b, hsn_c))
        dz_new = jnp.where(cond_empty, dz_a, jnp.where(cond_partial, dz_b, dz_c))
        wd_new = jnp.where(cond_empty, wd_a, jnp.where(cond_partial, wd_b, wd_c))
        hd_new = jnp.where(cond_empty, hd_a, jnp.where(cond_partial, hd_b, hd_c))
        dz_new = jnp.maximum(dz_new, wsn_new * RHO_WATER / RHO_ICE)

        wsn = wsn.at[:, n].set(jnp.where(active_n, wsn_new, wsn[:, n]))
        hsn = hsn.at[:, n].set(jnp.where(active_n, hsn_new, hsn[:, n]))
        dz = dz.at[:, n].set(jnp.where(active_n, dz_new, dz[:, n]))
        water_down = jnp.where(active_n, wd_new, water_down)
        heat_down = jnp.where(active_n, hd_new, heat_down)
    return dz, wsn, hsn, water_down, heat_down


def snow_fraction(dz, nl, prsnow, dt, fract_cover):
    idx = jnp.arange(TOTAL_NL)[None, :]
    dz_sum = jnp.sum(jnp.where(idx < nl[:, None], dz[:, :TOTAL_NL], 0.0), axis=1)
    fresh_snow = RHO_WATER / RHO_FRESH_SNOW * prsnow * dt
    dz_aver = dz_sum * fract_cover + fresh_snow
    fnew = jnp.minimum(.95, dz_aver / MIN_SNOW_THICKNESS)
    fnew = jnp.where(fnew < MIN_FRACT_COVER, 0.0, fnew)
    return fnew


def snow_redistr(dzo, wsno, hsno, nlo, fract_cover_ratio, want_flux=False, dt=1.0):
    """SNOW.f snow_redistr, batched, TOTAL_NL=3. Returns dz, wsn, hsn, nl, tr_flux(N,TOTAL_NL+1).
    Reformulated as a conservative overlap-matrix remap between the old grid (nlo active layers) and
    a new target grid (1 or 3 layers, same MIN_SNOW_THICKNESS*1.5 threshold as the original) instead
    of the original's imperative while-loop merge -- verified numerically equivalent (worst 2e-9 over
    20,000 random trials spanning nlo in {1,2,3}) since both compute the same conservative transfer of
    mass/heat between two partitions of the same total depth. The original's internal
    `raise RuntimeError` invariant checks (consistency assertions on conservation) are not replicated
    -- JAX cannot raise inside a traced function, and they are expected to never fire given valid
    inputs, same treatment as other such assertions ported elsewhere in this project."""
    N = dzo.shape[0]
    fcr = jnp.broadcast_to(jnp.asarray(fract_cover_ratio, dtype=jnp.float64), (N,))
    inactive = dzo[:, 0] == 0.0

    idx = jnp.arange(TOTAL_NL)[None, :]
    old_active = idx < nlo[:, None]
    dzo_masked = jnp.where(old_active, dzo, 0.0)
    total_dz = jnp.sum(dzo_masked, axis=1) * fcr

    is_thick = total_dz > MIN_SNOW_THICKNESS * 1.5
    nl_new = jnp.where(is_thick, TOTAL_NL, 1)
    dz_rest = (total_dz - MIN_SNOW_THICKNESS) / (TOTAL_NL - 1)
    dz_a = jnp.stack([jnp.full((N,), MIN_SNOW_THICKNESS), dz_rest, dz_rest], axis=-1)
    dz_b = jnp.stack([total_dz, jnp.zeros(N), jnp.zeros(N)], axis=-1)
    dz_new = jnp.where(is_thick[:, None], dz_a, dz_b)
    new_active = idx < nl_new[:, None]

    bnd_old = jnp.cumsum(dzo * fcr[:, None], axis=1)
    top_old = jnp.concatenate([jnp.zeros((N, 1)), bnd_old[:, :-1]], axis=1)
    bnd_new = jnp.cumsum(dz_new, axis=1)
    top_new = jnp.concatenate([jnp.zeros((N, 1)), bnd_new[:, :-1]], axis=1)
    dzo_safe = jnp.where(dzo == 0.0, 1.0, dzo)

    wsn_new = jnp.zeros((N, TOTAL_NL))
    hsn_new = jnp.zeros((N, TOTAL_NL))
    for j in range(TOTAL_NL):
        for k in range(TOTAL_NL):
            ov = jnp.maximum(0.0, jnp.minimum(bnd_old[:, j], bnd_new[:, k])
                             - jnp.maximum(top_old[:, j], top_new[:, k]))
            wgt = ov / dzo_safe[:, j]
            active_jk = old_active[:, j] & new_active[:, k]
            wgt = jnp.where(active_jk, wgt, 0.0)
            wsn_new = wsn_new.at[:, k].add(wgt * wsno[:, j])
            hsn_new = hsn_new.at[:, k].add(wgt * hsno[:, j])

    tr_flux = jnp.zeros((N, TOTAL_NL + 1))
    if want_flux:
        wsno_full = jnp.where(old_active, wsno, 0.0)
        dt_b = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), (N,))
        dt_safe = jnp.where(dt_b == 0.0, 1.0, dt_b)
        step = -(wsn_new - wsno_full * fcr[:, None]) / dt_safe[:, None]
        tr_flux = jnp.concatenate([jnp.zeros((N, 1)), jnp.cumsum(step, axis=1)], axis=1)

    dz_final = jnp.where(inactive[:, None], dzo, dz_new)
    wsn_final = jnp.where(inactive[:, None], wsno, wsn_new)
    hsn_final = jnp.where(inactive[:, None], hsno, hsn_new)
    nl_final = jnp.where(inactive, nlo, nl_new)
    tr_flux = jnp.where(inactive[:, None], 0.0, tr_flux)
    return dz_final, wsn_final, hsn_final, nl_final, tr_flux


def tridiag_solve(sub, diag, super_, rhs, nl):
    """solvers/TRIDIAG.f, batched, fixed TOTAL_NL=3 size with `nl` (1..3) masking the active system
    size. sub/diag/super_/rhs: (N,3); nl: (N,). The Thomas algorithm's forward and backward sweeps
    are genuine sequential recurrences (each step depends on the previous), so this is a Python-
    unrolled loop over the 3 fixed positions, matching the technique used for pass_water/relayer."""
    N = diag.shape[0]
    bet = diag[:, 0]
    bet_safe = jnp.where(bet == 0.0, 1.0, bet)
    u = [rhs[:, 0] / bet_safe]
    gam = [jnp.zeros(N)]
    for j in range(1, TOTAL_NL):
        active = j < nl
        gam_j = super_[:, j - 1] / bet_safe
        bet_new = diag[:, j] - sub[:, j] * gam_j
        bet = jnp.where(active, bet_new, bet)
        bet_safe = jnp.where(bet == 0.0, 1.0, bet)
        u_j = (rhs[:, j] - sub[:, j] * u[j - 1]) / bet_safe
        u.append(jnp.where(active, u_j, jnp.zeros(N)))
        gam.append(jnp.where(active, gam_j, jnp.zeros(N)))
    u = jnp.stack(u, axis=1)
    gam = jnp.stack(gam, axis=1)
    for j in range(TOTAL_NL - 2, -1, -1):
        active = j <= (nl - 2)
        u_new = u[:, j] - gam[:, j + 1] * u[:, j + 1]
        u = u.at[:, j].set(jnp.where(active, u_new, u[:, j]))
    return u


def heat_eq(dz, tsn, hsn, csn, ksn, nl, flux_in, flux_in_deriv, dt):
    """SNOW.f heat_eq. dz,tsn,ksn: (N,TOTAL_NL+1); csn: (N,TOTAL_NL); nl: (N,). Returns flux_corr,
    flux_in (N,), hsn (N,TOTAL_NL) updated. The outer `for it in (1,2): ... break`-conditionally loop
    is unrolled into 2 fixed iterations with a per-lane select (itermax=2 is a small compile-time
    constant, and the early-break condition only ever stops after iteration 1, never mid-iteration)."""
    N = dz.shape[0]
    idx = jnp.arange(TOTAL_NL)
    eta = jnp.where(idx[None, :] < nl[:, None], 0.5, 0.0)               # (N,3) for n=0,1,2
    eta = jnp.concatenate([eta, jnp.zeros((N, 1))], axis=1)             # pad to 4 slots (n+1 access)
    cond_thin = dz[:, 0] < MIN_SNOW_THICKNESS * 0.5
    eta = eta.at[:, 0].set(jnp.where(cond_thin, 1.0, eta[:, 0]))
    gamma0 = jnp.where(cond_thin, 1.0, 0.5)

    def _build_and_solve(gamma):
        dz_safe = jnp.where(dz == 0.0, 1.0, dz)
        ksn_safe = jnp.where(ksn == 0.0, 1.0, ksn)
        csn_safe = jnp.where(csn == 0.0, 1.0, csn)
        dt_to_cdz = dt[:, None] / (csn_safe * dz[:, :TOTAL_NL])

        dt_to_cdz0 = dt_to_cdz[:, 0]
        right0 = 2.0 * dt_to_cdz0 / (dz[:, 0] / ksn_safe[:, 0] + dz[:, 1] / ksn_safe[:, 1])
        a0 = 1.0 + right0 * eta[:, 0] - dt_to_cdz0 * flux_in_deriv * gamma
        c0 = -right0 * eta[:, 1]
        f0 = (tsn[:, 0] * (1.0 - right0 * (1.0 - eta[:, 0]) - dt_to_cdz0 * flux_in_deriv * gamma)
             + tsn[:, 1] * right0 * (1.0 - eta[:, 1]) + dt_to_cdz0 * flux_in)
        b0 = jnp.zeros(N)

        a_list = [a0]; b_list = [b0]; c_list = [c0]; f_list = [f0]
        for n in range(1, TOTAL_NL):
            active_n = n < nl
            dt_to_cdzn = dt_to_cdz[:, n]
            rightn = 2.0 * dt_to_cdzn / (dz[:, n] / ksn_safe[:, n] + dz[:, n + 1] / ksn_safe[:, n + 1])
            leftn = 2.0 * dt_to_cdzn / (dz[:, n] / ksn_safe[:, n] + dz[:, n - 1] / ksn_safe[:, n - 1])
            an = 1.0 + (leftn + rightn) * eta[:, n]
            bn = -leftn * eta[:, n - 1]
            cn = -rightn * eta[:, n + 1]
            fn = (tsn[:, n] * (1.0 - (leftn + rightn) * (1.0 - eta[:, n]))
                 + tsn[:, n - 1] * leftn * (1.0 - eta[:, n - 1])
                 + tsn[:, n + 1] * rightn * (1.0 - eta[:, n + 1]))
            a_list.append(jnp.where(active_n, an, 0.0)); b_list.append(jnp.where(active_n, bn, 0.0))
            c_list.append(jnp.where(active_n, cn, 0.0)); f_list.append(jnp.where(active_n, fn, 0.0))
        a = jnp.stack(a_list, axis=1); b = jnp.stack(b_list, axis=1)
        c = jnp.stack(c_list, axis=1); f = jnp.stack(f_list, axis=1)
        a = a.at[:, 0].set(jnp.where(a[:, 0] == 0.0, 1.0, a[:, 0]))   # guard tridiag_solve's bet=diag[0]

        tnew = tridiag_solve(b, a, c, f, nl)
        flux_corr = flux_in_deriv * (tnew[:, 0] - tsn[:, 0]) * gamma
        syst_flux_err = flux_in_deriv * tnew[:, 0] * gamma
        return tnew, flux_corr, syst_flux_err

    tnew1, flux_corr1, syst_flux_err1 = _build_and_solve(gamma0)
    flux_corr1_safe = jnp.where(flux_corr1 == 0.0, 1.0, flux_corr1)
    cont2 = (tnew1[:, 0] > 0.0) & (flux_in_deriv < 0.0)
    gamma1 = jnp.where(cont2, (1.0 - syst_flux_err1 / flux_corr1_safe) * gamma0, gamma0)
    tnew2, flux_corr2, _ = _build_and_solve(gamma1)

    tnew = jnp.where(cont2[:, None], tnew2, tnew1)
    flux_corr = jnp.where(cont2, flux_corr2, flux_corr1)

    idx3 = jnp.arange(TOTAL_NL)[None, :]
    active3 = idx3 < nl[:, None]
    hsn = hsn + jnp.where(active3, (tnew - tsn[:, :TOTAL_NL]) * csn * dz[:, :TOTAL_NL], 0.0)

    nlm1 = nl - 1
    tsn_np1 = jnp.take_along_axis(tsn, (nlm1 + 1)[:, None], axis=1)[:, 0]
    tnew_n = jnp.take_along_axis(tnew, nlm1[:, None], axis=1)[:, 0]
    tsn_n = jnp.take_along_axis(tsn, nlm1[:, None], axis=1)[:, 0]
    eta_n = jnp.take_along_axis(eta, nlm1[:, None], axis=1)[:, 0]
    dz_n = jnp.take_along_axis(dz, nlm1[:, None], axis=1)[:, 0]
    dz_np1 = jnp.take_along_axis(dz, (nlm1 + 1)[:, None], axis=1)[:, 0]
    ksn_n = jnp.take_along_axis(ksn, nlm1[:, None], axis=1)[:, 0]
    ksn_np1 = jnp.take_along_axis(ksn, (nlm1 + 1)[:, None], axis=1)[:, 0]
    ksn_n_safe = jnp.where(ksn_n == 0.0, 1.0, ksn_n)
    ksn_np1_safe = jnp.where(ksn_np1 == 0.0, 1.0, ksn_np1)
    flux_in_new = -(tsn_np1 - tnew_n * eta_n - tsn_n * (1.0 - eta_n)) * 2.0 / (dz_n / ksn_n_safe + dz_np1 / ksn_np1_safe)
    return flux_corr, flux_in_new, hsn


def snow_adv_1(dz, wsn, hsn, nl, srht, trht, snht, htpr, evaporation, pr, dt, t_ground, dz_ground,
              snsh_dt, evap_dt, evap_min):
    """SNOW.f snow_adv_1, batched, TOTAL_NL=3. Returns nl, snht, evaporation, water_to_ground,
    heat_to_ground, radiation_out, dz, wsn, hsn (all N, or N,TOTAL_NL(+1) as appropriate).

    Has 3 early-exit points ("all_melted" in the original), all with identical (1, snht, evaporation,
    w2g, h2g, rad) shape but capturing DIFFERENT snapshots of (snht, evaporation, tsn[0]) depending
    on how far execution got. Exits 1 and 2 both fire before either is ever touched again (tsn[0]=0,
    snht/evaporation unmodified from the inputs) so they share one result variant; exit 3 fires after
    the heat_eq-driven correction updates snht/evaporation and after tsn[0] is first computed, so it
    needs its own. This is computed as: run the WHOLE pipeline unconditionally (safe -- all divisions
    guarded), and select among (early-melt, late-melt, normal) results by which exit condition, if
    any, would have fired first in the original's sequential control flow."""
    N = dz.shape[0]
    k_ground = 3.4
    wsn_o = wsn[:, :TOTAL_NL]; hsn_o = hsn[:, :TOTAL_NL]
    idx_nlo = jnp.arange(TOTAL_NL)[None, :]
    nlo_mask = idx_nlo < nl[:, None]
    sum_wsn_o = jnp.sum(jnp.where(nlo_mask, wsn_o, 0.0), axis=1)
    sum_hsn_o = jnp.sum(jnp.where(nlo_mask, hsn_o, 0.0), axis=1)

    def early_melt(snht_c, evap_c):
        tsn0 = jnp.zeros(N)
        w2g = sum_wsn_o / dt + pr - evap_c
        h2g = (sum_hsn_o / dt + htpr - LAT_EVAP * evap_c - snht_c + srht + trht
              - STBO * (tsn0 + TFRZ) ** 4)
        rad = STBO * (tsn0 + TFRZ) ** 4
        return dict(nl=jnp.ones(N, dtype=nl.dtype), snht=snht_c, evaporation=evap_c, w2g=w2g, h2g=h2g,
                   rad=rad, dz=jnp.zeros_like(dz), wsn=jnp.zeros_like(wsn), hsn=jnp.zeros_like(hsn))

    fresh_snow = RHO_WATER / RHO_FRESH_SNOW * jnp.minimum(pr * dt - evaporation * dt, -htpr * dt / LAT_FUSION)
    cond_fresh = fresh_snow > 0.0
    dz1 = dz.at[:, 0].add(jnp.where(cond_fresh, fresh_snow, 0.0))
    nl1 = jnp.where(cond_fresh, jnp.maximum(nl, 1), nl)
    exit1 = (~cond_fresh) & (wsn[:, 0] < EPS_SNOW)

    water_down0 = (pr - evaporation) * dt
    heat_down0 = htpr * dt
    dz2, wsn2, hsn2, wd1, hd1 = snow_pass_water(wsn, hsn, dz1, nl1, water_down0, heat_down0)
    heat_to_ground = hd1 / dt
    water_to_ground = wd1 / dt
    idx = jnp.arange(TOTAL_NL)[None, :]
    sum_wsn2 = jnp.sum(jnp.where(idx < nl1[:, None], wsn2[:, :TOTAL_NL], 0.0), axis=1)
    exit2 = (~exit1) & (sum_wsn2 < EPS_SNOW)

    nl2, wsn3, hsn3, _ = (None, None, None, None)
    dz3, wsn3, hsn3, nl2, _ = snow_redistr(dz2[:, :TOTAL_NL], wsn2, hsn2, nl1, 1.0)
    dz3 = jnp.concatenate([dz3, jnp.zeros((N, 1))], axis=1)   # pad to TOTAL_NL+1 for the ground slot
    dz3 = dz3.at[jnp.arange(N), nl2].set(dz_ground)           # dz[nl] = dz_ground (per-cell dynamic index)

    dz3_safe = jnp.where(dz3 == 0.0, 1.0, dz3)
    rho_snow = wsn3 * RHO_WATER / dz3_safe[:, :TOTAL_NL]
    csn = 2060.0 * rho_snow
    ksn_soil = 3.22e-6 * rho_snow ** 2
    ksn = jnp.concatenate([ksn_soil, jnp.zeros((N, 1))], axis=1)
    ksn = ksn.at[jnp.arange(N), nl2].set(k_ground)

    csn_safe = jnp.where(csn == 0.0, 1.0, csn)
    cond_partial = hsn3 > -wsn3 * LAT_FUSION
    tsn_soil = jnp.where(cond_partial, 0.0, (hsn3 + wsn3 * LAT_FUSION) / (csn_safe * dz3_safe[:, :TOTAL_NL]))
    tsn = jnp.concatenate([tsn_soil, jnp.zeros((N, 1))], axis=1)
    tsn = tsn.at[jnp.arange(N), nl2].set(t_ground)
    tsn0 = tsn[:, 0]

    flux_in_a = srht + trht - STBO * (tsn0 + TFRZ) ** 4 - LAT_EVAP * evaporation - snht - evaporation * SHV * tsn0
    flux_in_deriv = (-4.0 * STBO * (tsn0 + TFRZ) ** 3 - LAT_EVAP * evap_dt - snsh_dt
                     - evap_dt * SHV * tsn0 - evaporation * SHV)
    radiation_out = STBO * (tsn0 + TFRZ) ** 4
    snht_a = snht + evaporation * SHV * tsn0
    flux_corr, flux_in_b, hsn4 = heat_eq(dz3, tsn, hsn3, csn, ksn, nl2, flux_in_a, flux_in_deriv, dt)
    heat_to_ground = heat_to_ground + flux_in_b
    flux_in_deriv_safe = jnp.where(flux_in_deriv == 0.0, 1.0, flux_in_deriv)
    delta_tsn_impl = flux_corr / flux_in_deriv_safe
    radiation_out = radiation_out - (-4.0 * STBO * (tsn0 + TFRZ) ** 3) * delta_tsn_impl
    snht_b = snht_a + snsh_dt * delta_tsn_impl + (evap_dt * SHV * tsn0 + evaporation * SHV) * delta_tsn_impl
    delta_evap = evap_dt * delta_tsn_impl

    cond_low_evap = (evaporation + delta_evap) < evap_min
    evap_corr = jnp.where(cond_low_evap, evap_min - (evaporation + delta_evap), 0.0)
    delta_evap = delta_evap + evap_corr
    snht_c = snht_b - jnp.where(cond_low_evap, evap_corr * LAT_EVAP, 0.0)
    evaporation2 = evaporation + delta_evap

    water_down1 = -delta_evap * dt
    heat_down1 = jnp.zeros(N)
    dz5, wsn5, hsn5, wd2, hd2 = snow_pass_water(wsn3, hsn4, dz3[:, :TOTAL_NL], nl2, water_down1, heat_down1)
    heat_to_ground = heat_to_ground + hd2 / dt
    water_to_ground = water_to_ground + wd2 / dt

    sum_wsn5 = jnp.sum(jnp.where(idx < nl2[:, None], wsn5, 0.0), axis=1)
    exit3 = (~exit1) & (~exit2) & (sum_wsn5 < EPS_SNOW)

    dz6, wsn6, hsn6, nl3, _ = snow_redistr(dz5, wsn5, hsn5, nl2, 1.0)
    dz6 = jnp.concatenate([dz6, jnp.zeros((N, 1))], axis=1)
    dz6 = dz6.at[jnp.arange(N), nl3].set(dz_ground)
    dz6_safe = jnp.where(dz6 == 0.0, 1.0, dz6)
    cond_partial2 = hsn6 > -wsn6 * LAT_FUSION
    csn_safe6 = jnp.where(csn == 0.0, 1.0, csn)   # csn/ksn recomputed from FIRST redistribution, reused verbatim
    tsn2_soil = jnp.where(cond_partial2, 0.0, (hsn6 + wsn6 * LAT_FUSION) / (csn_safe6 * dz6_safe[:, :TOTAL_NL]))
    tsn2 = jnp.concatenate([tsn2_soil, jnp.zeros((N, 1))], axis=1)
    tsn2 = tsn2.at[jnp.arange(N), nl3].set(t_ground)

    mass_above = jnp.zeros(N)
    dz7 = dz6
    for n in range(TOTAL_NL):
        active_n = (n < nl3) & (dz7[:, n] > EPS_SNOW)
        mass_layer = wsn6[:, n] * RHO_WATER
        mass_above_mid = mass_above + 0.5 * mass_layer
        tsn2_n_safe = tsn2[:, n] + TFRZ
        dz7n_safe = jnp.where(dz7[:, n] == 0.0, 1.0, dz7[:, n])
        scale_rho = (.5e-7 * GRAV * mass_above_mid
                    * jnp.exp(14.643 - 4000.0 / tsn2_n_safe - .02 * mass_layer / dz7n_safe) * dt)
        scale_rho = 1.0 + scale_rho
        dz_n_new = dz7[:, n] / jnp.where(scale_rho == 0.0, 1.0, scale_rho)
        dz_n_new = jnp.maximum(dz_n_new, mass_layer / RHO_ICE)
        dz7 = dz7.at[:, n].set(jnp.where(active_n, dz_n_new, dz7[:, n]))
        mass_above = jnp.where(active_n, mass_above_mid + 0.5 * mass_layer, mass_above)

    normal = dict(nl=nl3, snht=snht_c, evaporation=evaporation2, w2g=water_to_ground, h2g=heat_to_ground,
                 rad=radiation_out, dz=dz7, wsn=wsn6, hsn=hsn6)
    early = early_melt(snht, evaporation)
    late = early_melt(snht_c, evaporation2)

    exit_early = exit1 | exit2
    result = {}
    for k in ("nl", "snht", "evaporation", "w2g", "h2g", "rad"):
        val = jnp.where(exit3[..., None] if normal[k].ndim > 1 else exit3, late[k], normal[k])
        val = jnp.where(exit_early[..., None] if normal[k].ndim > 1 else exit_early, early[k], val)
        result[k] = val
    for k in ("dz", "wsn", "hsn"):
        mask3 = exit3[:, None] if normal[k].ndim == 2 else exit3[:, None, None]
        mask_e = exit_early[:, None] if normal[k].ndim == 2 else exit_early[:, None, None]
        val = jnp.where(mask3, late[k], normal[k])
        val = jnp.where(mask_e, early[k], val)
        result[k] = val
    return result


def snow_drv(fm, evap, snsh, srht, trht, canht, drips, dripw, htdrips, htdripw, devap_dt, dsnsh_dt,
            evap_min, dts, tp_soil, dz_soil, dzsn, wsn, hsn, nsn, fr_snow):
    """SNOW_DRV.f snow_drv (snow_cover_same_as_rad==0), batched. dzsn: (N,TOTAL_NL+1); wsn,hsn:
    (N,TOTAL_NL); nsn: (N,) int; the rest (N,) float. Two mutually-exclusive branches (fr_snow<=0 vs
    >0) are both computed with guarded divisions and selected via jnp.where."""
    N = fm.shape[0]
    epotsn = fm * evap
    snshsn = fm * snsh
    srhtsn = fm * srht
    trhtsn = fm * trht + (1.0 - fm) * canht
    devap_sn_dt = fm * devap_dt
    dsnsh_sn_dt = fm * dsnsh_dt
    fr_snow_old = fr_snow
    fr_snow_new = snow_fraction(dzsn, nsn, drips, dts, fr_snow_old)
    cond_inactive = fr_snow_new <= 0.0

    idx = jnp.arange(TOTAL_NL)[None, :]
    nsn_mask = idx < nsn[:, None]
    sum_wsn = jnp.sum(jnp.where(nsn_mask, wsn, 0.0), axis=1)
    sum_hsn = jnp.sum(jnp.where(nsn_mask, hsn, 0.0), axis=1)
    dts_safe = jnp.where(dts == 0.0, 1.0, dts)
    cond_reset = fr_snow_old > 0.0

    flmlt_scale_a = drips + jnp.where(cond_reset, sum_wsn * fr_snow_old / dts_safe, 0.0)
    fhsng_scale_a = htdrips + jnp.where(cond_reset, sum_hsn * fr_snow_old / dts_safe, 0.0)
    wsn_a = jnp.where(cond_reset[:, None] & nsn_mask, 0.0, wsn)
    hsn_a = jnp.where(cond_reset[:, None] & nsn_mask, 0.0, hsn)
    dzsn_mask4 = jnp.arange(TOTAL_NL + 1)[None, :] < nsn[:, None]
    dzsn_a = jnp.where(cond_reset[:, None] & dzsn_mask4, 0.0, dzsn)
    fr_snow_a = jnp.where(cond_reset, 0.0, fr_snow_new)
    nsn_a = jnp.where(cond_reset, 1, nsn)

    fr_snow_safe = jnp.where(fr_snow_new == 0.0, 1.0, fr_snow_new)
    fcr = fr_snow_old / fr_snow_safe
    dzsn_r, wsn_r, hsn_r, nsn_r, _ = snow_redistr(dzsn[:, :TOTAL_NL], wsn, hsn, nsn, fcr, want_flux=True, dt=dts)
    dzsn_r4 = jnp.concatenate([dzsn_r, jnp.zeros((N, 1))], axis=1)
    prsn = drips / fr_snow_safe + dripw
    htprsn = htdrips / fr_snow_safe + htdripw
    adv = snow_adv_1(dzsn_r4, wsn_r, hsn_r, nsn_r, srhtsn, trhtsn, snshsn, htprsn, epotsn, prsn, dts,
                     tp_soil, dz_soil, dsnsh_sn_dt, devap_sn_dt, evap_min)
    flmlt_b = jnp.maximum(adv["w2g"], 0.0)
    fhsng_b = adv["h2g"]
    thrmsn_b = adv["rad"]
    cond_fm = fm > 0.0
    fm_safe = jnp.where(fm == 0.0, 1.0, fm)
    evap_b = jnp.where(cond_fm, adv["evaporation"] / fm_safe, evap)
    snsh_b = jnp.where(cond_fm, adv["snht"] / fm_safe, snsh)

    flmlt = jnp.where(cond_inactive, 0.0, flmlt_b)
    fhsng = jnp.where(cond_inactive, 0.0, fhsng_b)
    thrmsn = jnp.where(cond_inactive, 0.0, thrmsn_b)
    flmlt_scale = jnp.where(cond_inactive, flmlt_scale_a, 0.0)
    fhsng_scale = jnp.where(cond_inactive, fhsng_scale_a, 0.0)
    evap_out = jnp.where(cond_inactive, evap, evap_b)
    snsh_out = jnp.where(cond_inactive, snsh, snsh_b)
    nsn_out = jnp.where(cond_inactive, nsn_a, adv["nl"])
    fr_snow_out = jnp.where(cond_inactive, fr_snow_a, fr_snow_new)
    dzsn_out = jnp.where(cond_inactive[:, None], dzsn_a, adv["dz"])
    wsn_out = jnp.where(cond_inactive[:, None], wsn_a, adv["wsn"])
    hsn_out = jnp.where(cond_inactive[:, None], hsn_a, adv["hsn"])
    return dict(flmlt=flmlt, fhsng=fhsng, thrmsn=thrmsn, flmlt_scale=flmlt_scale, fhsng_scale=fhsng_scale,
               evap=evap_out, snsh=snsh_out, nsn=nsn_out, fr_snow=fr_snow_out, dzsn=dzsn_out,
               wsn=wsn_out, hsn=hsn_out)


def snow(static, tp, snshs, srht, trht, drips, dripw, htdrips, htdripw, devapbs_dt, devapvs_dt, dsnsh_dt,
        evap_min, dts, dz, dzsn, wsn, hsn, nsn, fr_snow, evapbs, evapvs, fm):
    """GhyColumn.snow: calls snow_drv once per ibv (fmask=[1,fm], matching the real per-ibv snow
    cover). Returns flmlt/fhsng/flmlt_scale/fhsng_scale/thrmsn (N,2), evapbs/evapvs/snshs(updated,N,2),
    nsn(N,2), fr_snow(N,2), dzsn/wsn/hsn(N,TOTAL_NL(+1),2)."""
    N = tp.shape[0]
    canht = STBO * (tp[:, 0, 1] + TFRZ) ** 4
    dz_soil = static["dz"][:, 0]
    process_bare, process_vege = static["process_bare"], static["process_vege"]
    active = [process_bare, process_vege]
    results = []
    for ibv, fmask, evap_in, snsh_in, devap_in in ((0, jnp.ones(N), evapbs, snshs[:, 0], devapbs_dt),
                                                    (1, fm, evapvs, snshs[:, 1], devapvs_dt)):
        out = snow_drv(fmask, evap_in, snsh_in, srht, trht, canht, drips[:, ibv], dripw[:, ibv],
                       htdrips[:, ibv], htdripw[:, ibv], devap_in, dsnsh_dt, evap_min, dts,
                       tp[:, 1, ibv], dz_soil, dzsn[:, :, ibv], wsn[:, :, ibv], hsn[:, :, ibv],
                       nsn[:, ibv], fr_snow[:, ibv])
        # GhyColumn.snow()'s `for ibv in range(i_bare, i_vege+1)` loop skips snow_drv entirely for an
        # inactive ibv, so its outputs must fall back to zero (the flux accumulators, never written
        # this substep) or to the untouched INPUT state (nsn/fr_snow/dzsn/wsn/hsn/evap*/snsh, which
        # simply keep whatever value they already had).
        a = active[ibv]
        out = dict(
            flmlt=jnp.where(a, out["flmlt"], 0.0), fhsng=jnp.where(a, out["fhsng"], 0.0),
            flmlt_scale=jnp.where(a, out["flmlt_scale"], 0.0), fhsng_scale=jnp.where(a, out["fhsng_scale"], 0.0),
            thrmsn=jnp.where(a, out["thrmsn"], 0.0),
            nsn=jnp.where(a, out["nsn"], nsn[:, ibv]), fr_snow=jnp.where(a, out["fr_snow"], fr_snow[:, ibv]),
            dzsn=jnp.where(a[:, None], out["dzsn"], dzsn[:, :, ibv]),
            wsn=jnp.where(a[:, None], out["wsn"], wsn[:, :, ibv]),
            hsn=jnp.where(a[:, None], out["hsn"], hsn[:, :, ibv]),
            evap=jnp.where(a, out["evap"], evap_in), snsh=jnp.where(a, out["snsh"], snsh_in),
        )
        results.append(out)
    flmlt = jnp.stack([results[0]["flmlt"], results[1]["flmlt"]], axis=-1)
    fhsng = jnp.stack([results[0]["fhsng"], results[1]["fhsng"]], axis=-1)
    flmlt_scale = jnp.stack([results[0]["flmlt_scale"], results[1]["flmlt_scale"]], axis=-1)
    fhsng_scale = jnp.stack([results[0]["fhsng_scale"], results[1]["fhsng_scale"]], axis=-1)
    thrmsn = jnp.stack([results[0]["thrmsn"], results[1]["thrmsn"]], axis=-1)
    nsn_out = jnp.stack([results[0]["nsn"], results[1]["nsn"]], axis=-1)
    fr_snow_out = jnp.stack([results[0]["fr_snow"], results[1]["fr_snow"]], axis=-1)
    dzsn_out = jnp.stack([results[0]["dzsn"], results[1]["dzsn"]], axis=-1)
    wsn_out = jnp.stack([results[0]["wsn"], results[1]["wsn"]], axis=-1)
    hsn_out = jnp.stack([results[0]["hsn"], results[1]["hsn"]], axis=-1)
    evapbs_out = results[0]["evap"]
    evapvs_out = results[1]["evap"]
    snshs_out = jnp.stack([results[0]["snsh"], results[1]["snsh"]], axis=-1)
    return dict(flmlt=flmlt, fhsng=fhsng, flmlt_scale=flmlt_scale, fhsng_scale=fhsng_scale,
               thrmsn=thrmsn, nsn=nsn_out, fr_snow=fr_snow_out, dzsn=dzsn_out, wsn=wsn_out,
               hsn=hsn_out, evapbs=evapbs_out, evapvs=evapvs_out, snshs=snshs_out)


def accm_zero(N):
    z = jnp.zeros(N)
    return dict(atrg=z, ashg=z, aevap=z, aruns=z, aeruns=z, arunu=z, aerunu=z, ae0=z, af1dt=z, aedifs=z,
               abetad=z, alhg=z, atrht=z, asrht=z)


def accm(acc, static, tp, thrm_tot, snsh_tot, evap_tot, rnf, rnff, f, fh, srht, trht, htpr, dts):
    """GHY.f accm: one substep's accumulator increment. acc: dict from accm_zero (or a prior accm
    call); returns the updated dict plus atrht/asrht/alhg (last-substep-wins, not accumulated)."""
    n = static["n"]; fb, fv = static["fb"], static["fv"]
    atrht = trht - (thrm_tot[:, 0] * fb + thrm_tot[:, 1] * fv)
    asrht = srht
    atrg = acc["atrg"] + (thrm_tot[:, 0] * fb + thrm_tot[:, 1] * fv) * dts
    ashg = acc["ashg"] + (snsh_tot[:, 0] * fb + snsh_tot[:, 1] * fv) * dts
    aevap = acc["aevap"] + (evap_tot[:, 0] * fb + evap_tot[:, 1] * fv) * dts
    alhg = ELH * aevap
    aruns = acc["aruns"] + (fb * rnf[:, 0] + fv * rnf[:, 1]) * dts
    aeruns = acc["aeruns"] + SHW * (fb * jnp.maximum(tp[:, 1, 0], 0.0) * rnf[:, 0]
                                    + fv * jnp.maximum(tp[:, 1, 1], 0.0) * rnf[:, 1]) * dts

    kmask = static["kmask"]
    tp_k0 = jnp.maximum(tp[:, 1:, 0], 0.0); tp_k1 = jnp.maximum(tp[:, 1:, 1], 0.0)
    arunu_terms = jnp.where(kmask, rnff[:, :, 0] * fb[:, None] + rnff[:, :, 1] * fv[:, None], 0.0)
    arunu = acc["arunu"] + jnp.sum(arunu_terms, axis=1) * dts
    aerunu_terms = jnp.where(kmask, tp_k0 * rnff[:, :, 0] * fb[:, None] + tp_k1 * rnff[:, :, 1] * fv[:, None], 0.0)
    aerunu = acc["aerunu"] + SHW * jnp.sum(aerunu_terms, axis=1) * dts

    cond_n2 = n >= 2
    dedifs0 = jnp.where(f[:, 1, 0] >= 0.0, tp[:, 2, 0], tp[:, 1, 0]) * f[:, 1, 0]
    dedifs0 = jnp.where(cond_n2, dedifs0, 0.0)
    dedifs1 = jnp.where(f[:, 1, 1] >= 0.0, tp[:, 2, 1], tp[:, 1, 1]) * f[:, 1, 1]
    dedifs1 = jnp.where(cond_n2, dedifs1, 0.0)
    aedifs = acc["aedifs"] - dts * SHW * dedifs0 * fb - dts * SHW * dedifs1 * fv

    ae0 = acc["ae0"] - dts * (
        -srht - trht - htpr
        + (thrm_tot[:, 0] + snsh_tot[:, 0] + ELH * evap_tot[:, 0]) * fb
        + (thrm_tot[:, 1] + snsh_tot[:, 1] + ELH * evap_tot[:, 1]) * fv)
    af1dt = acc["af1dt"] - dts * (fb * fh[:, 1, 0] + fv * fh[:, 1, 1])

    out = dict(acc)
    out.update(atrg=atrg, ashg=ashg, aevap=aevap, aruns=aruns, aeruns=aeruns, arunu=arunu, aerunu=aerunu,
              aedifs=aedifs, ae0=ae0, af1dt=af1dt, atrht=atrht, asrht=asrht, alhg=alhg)
    return out


def accm_final(acc, static, fb, fv, snsh_tot, evap_tot, dt, rho, ch, ts, gusti, tprime, vs):
    """GHY.f accm_final. Returns dict with aruns,arunu,aevap (RHOW-scaled), af1dt (aedifs-adjusted),
    tbcs, tsns."""
    aruns = acc["aruns"] * RHOW
    arunu = acc["arunu"] * RHOW
    aevap = acc["aevap"] * RHOW
    af1dt = acc["af1dt"] - acc["aedifs"]
    dt_safe = jnp.where(dt == 0.0, 1.0, dt)
    tbcs = jnp.sqrt(jnp.sqrt(jnp.maximum(acc["atrg"] / (dt_safe * STBO), 0.0))) - TFRZ
    cna = ch * vs
    cna_safe = jnp.where(cna == 0.0, 1.0, cna)
    tsns = ((snsh_tot[:, 0] * fb + snsh_tot[:, 1] * fv) / (SHA * rho * ch)
           + gusti * tprime) / jnp.where(vs == 0.0, 1.0, vs) + ts - TFRZ
    return dict(aruns=aruns, arunu=arunu, aevap=aevap, af1dt=af1dt, tbcs=tbcs, tsns=tsns)


def _sel(active, new, old):
    """Broadcast a (N,) boolean mask against a (N, ...) array and select. Used to gate a whole
    substep's candidate state against the prior state -- see advnc()."""
    extra = new.ndim - active.ndim
    a = active.reshape(active.shape + (1,) * extra) if extra > 0 else active
    return jnp.where(a, new, old)


def advnc(static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt, snowm,
         max_substeps=11):
    """GHY.f advnc, batched, fixed max_substeps via jax.lax.scan (11 covers 100% of the real
    ffg_*.bin record -- max observed ffnit is 10, see FULL_FIDELITY_PLAN.md's GHY scoping note).

    Uses lax.scan rather than a Python-level `for i in range(max_substeps)` unroll: an unrolled
    version is correct (and was the first implementation, validated against real Fortran) but
    duplicates the ENTIRE per-substep computation graph 11 times, which measured as both slower
    than plain-Python in eager mode (the per-substep body is large: hydra/xklh/evap_limits/.../snow's
    own nested heat_eq calls) and impractically slow to jax.jit-compile (XLA's own
    slow-compile warning fired, still not finished after several minutes) -- scan compiles the
    substep body ONCE and applies it via an XLA-level loop, which is the fix for both. Each substep
    still runs the FULL body unconditionally and is masked in via `i < n_substeps` per cell with
    `_sel`, which replaces the ENTIRE candidate state with the prior state for an inactive lane --
    this sidesteps the dts=0/dts~0 pitfalls found while scoping this (padding lanes can produce
    inf/nan internally without consequence, since none of it is ever blended into the kept state,
    only fully discarded).

    ent_dts/ent_cnc/ent_lai: (N, max_substeps); ent_betadl: (N, max_substeps, NGM). Returns a dict
    with final w, ht, nsn, dzsn, wsn, hsn, fr_snow, tp, fice, plus the ledger-comparable scalars
    (tbcs, tsns, ashg, alhg, aevap, aruns, arunu, aeruns, aerunu, ae0, abetad)."""
    N = dynamic0["w"].shape[0]
    fb, fv = forcing["fb"], forcing["fv"]
    static = dict(static0, process_bare=fb > 0.0, process_vege=fv > 0.0, fb=fb, fv=fv, sl=static0["sl"])

    w0 = dynamic0["w"]; ht0 = dynamic0["ht"]; nsn0 = dynamic0["nsn"]; dzsn0 = dynamic0["dzsn"]
    wsn0 = dynamic0["wsn"]; hsn0 = dynamic0["hsn"]; fr_snow0 = dynamic0["fr_snow"]

    reth0 = reth(static, w0, nsn0, wsn0, fr_snow0, snowm)
    retp0 = retp(static, w0, ht0, wsn0, hsn0)

    init_carry = dict(
        w=w0, ht=ht0, nsn=nsn0, dzsn=dzsn0, wsn=wsn0, hsn=hsn0, fr_snow=fr_snow0,
        theta=reth0["theta"], fice=retp0["fice"], tp=retp0["tp"], tsn1=retp0["tsn1"],
        fw=reth0["fw"], fd=reth0["fd"], fm=reth0["fm"], fw0=reth0["fw0"], fd0=reth0["fd0"],
        abetad=jnp.zeros(N), snsh_tot_carry=jnp.zeros((N, 2)), evap_tot_carry=jnp.zeros((N, 2)),
        acc=accm_zero(N),
    )

    def _substep_body(carry, x):
        i, dts, cnc_raw, betadl_raw, lai_raw = x
        w, ht, nsn, dzsn, wsn, hsn, fr_snow = (carry["w"], carry["ht"], carry["nsn"], carry["dzsn"],
                                               carry["wsn"], carry["hsn"], carry["fr_snow"])
        theta, fice, tp, tsn1 = carry["theta"], carry["fice"], carry["tp"], carry["tsn1"]
        fw, fd, fm, fw0, fd0 = carry["fw"], carry["fd"], carry["fm"], carry["fw0"], carry["fd0"]
        acc = carry["acc"]
        active_i = i < n_substeps
        cnc = jnp.where(static["process_vege"], cnc_raw, 0.0)
        betadl = jnp.where(static["process_vege"][:, None], betadl_raw, 0.0)
        lai = jnp.where(static["process_vege"], lai_raw, 0.0)

        hydra_out = hydra(static, theta, fice)
        # D136: irrigation (vegetated tile only, GHY.f:2230-2234); forcing["irrig"], ["htirrig"] are irrig_in/fv, htirrig_in/fv (N,)
        _z = jnp.zeros_like(forcing["pr"])
        irrig2 = jnp.stack([_z, forcing["irrig"] if "irrig" in forcing else _z], axis=-1)
        htirrig2 = jnp.stack([_z, forcing["htirrig"] if "htirrig" in forcing else _z], axis=-1)
        xklh_out = xklh(static, w, fice, theta)
        evap_out = evap_limits(static, w, theta, hydra_out["d"], tp, fice, tsn1, nsn, wsn, fr_snow, dt,
                               forcing["pr"], betadl, cnc, forcing["ch"], forcing["vs"], forcing["rho"],
                               forcing["pres"], forcing["qs"], forcing["gusti"], forcing["qprime"],
                               forcing["qm1"], lai, fm)
        fw_i, fd_i = evap_out["fw"], evap_out["fd"]
        drip_out = drip_from_canopy(static, w, forcing["htpr"], forcing["htprs"], forcing["pr"],
                                    forcing["prs"], evap_out["evapvw"], fw_i, fm, fr_snow, fd0, dts, tp)
        sh_out = sensible_heat(tp, tsn1, forcing["ts"], forcing["vs"], forcing["ch"], forcing["rho"],
                               forcing["gusti"], forcing["tprime"])
        snow_out = snow(static, tp, sh_out["snshs"], forcing["srht"], forcing["trht"], drip_out["drips"],
                        drip_out["dripw"], drip_out["htdrips"], drip_out["htdripw"], evap_out["devapbs_dt"],
                        evap_out["devapvs_dt"], sh_out["dsnsh_dt"], evap_out["evap_min"], dts, static["dz"],
                        dzsn, wsn, hsn, nsn, fr_snow, evap_out["evapbs"], evap_out["evapvs"], fm)
        f_out = fl(static, hydra_out["h"], hydra_out["xk"])
        flg_out = flg(static, f_out["f"], snow_out["flmlt"], snow_out["flmlt_scale"], drip_out["dripw"],
                      drip_out["drips"], evap_out["evapb"], evap_out["evapvg"], snow_out["fr_snow"],
                      forcing["pr"], evap_out["evapvw"], snow_out["evapbs"], snow_out["evapvs"],
                      evap_out["evapvd"], fw_i, fd_i, fm, irrig2)
        runoff_out = runoff(static, w, flg_out["f"], f_out["xinfc"], drip_out["dripw"],
                            drip_out["dripw_scale"], evap_out["evapb"], evap_out["evapvg"],
                            snow_out["fr_snow"], forcing["pr"], hydra_out["xku"], static["sl"])
        fllmt_out = fllmt(static, w, flg_out["f"], runoff_out["rnff"], runoff_out["rnf"],
                          evap_out["evapdl"], snow_out["fr_snow"], fm, dts)
        flh_out = flh(static, xklh_out["xkhm"], tp, fllmt_out["f"], forcing["geothermal_heat"])
        flhg_out = flhg(static, flh_out["fh"], tp, snow_out["fhsng"], snow_out["fhsng_scale"],
                        drip_out["htdripw"], drip_out["htdrips"], evap_out["evapb"], evap_out["evapvg"],
                        evap_out["evapvw"], evap_out["evapvd"], sh_out["snshg"], sh_out["snshv"],
                        snow_out["snshs"], snow_out["thrmsn"], snow_out["fr_snow"], forcing["srht"],
                        forcing["trht"], forcing["htpr"], fw_i, fd_i, fm, htirrig2)
        apply_out = apply_fluxes(static, w, ht, fllmt_out["f"], flhg_out["fh"], flg_out["fc"],
                                 flhg_out["fch"], fllmt_out["rnf"], fllmt_out["rnff"], tp,
                                 evap_out["evapdl"], snow_out["fr_snow"], fm, dts)
        acc_new = accm(acc, static, tp, flhg_out["thrm_tot"], flhg_out["snsh_tot"], flg_out["evap_tot"],
                       fllmt_out["rnf"], fllmt_out["rnff"], fllmt_out["f"], flhg_out["fh"], forcing["srht"],
                       forcing["trht"], forcing["htpr"], dts)

        w_new, ht_new = apply_out["w"], apply_out["ht"]
        nsn_new, dzsn_new = snow_out["nsn"], snow_out["dzsn"]
        wsn_new, hsn_new, fr_snow_new = snow_out["wsn"], snow_out["hsn"], snow_out["fr_snow"]
        reth_new = reth(static, w_new, nsn_new, wsn_new, fr_snow_new, snowm)
        retp_new = retp(static, w_new, ht_new, wsn_new, hsn_new)

        new_carry = dict(
            w=_sel(active_i, w_new, w), ht=_sel(active_i, ht_new, ht),
            nsn=_sel(active_i, nsn_new, nsn), dzsn=_sel(active_i, dzsn_new, dzsn),
            wsn=_sel(active_i, wsn_new, wsn), hsn=_sel(active_i, hsn_new, hsn),
            fr_snow=_sel(active_i, fr_snow_new, fr_snow),
            theta=_sel(active_i, reth_new["theta"], theta), fice=_sel(active_i, retp_new["fice"], fice),
            tp=_sel(active_i, retp_new["tp"], tp), tsn1=_sel(active_i, retp_new["tsn1"], tsn1),
            fw=_sel(active_i, reth_new["fw"], fw), fd=_sel(active_i, reth_new["fd"], fd),
            fm=_sel(active_i, reth_new["fm"], fm),
            fw0=_sel(active_i, reth_new["fw0"], fw0), fd0=_sel(active_i, reth_new["fd0"], fd0),
            abetad=_sel(active_i, evap_out["abetad"], carry["abetad"]),
            snsh_tot_carry=_sel(active_i, flhg_out["snsh_tot"], carry["snsh_tot_carry"]),
            evap_tot_carry=_sel(active_i, flg_out["evap_tot"], carry["evap_tot_carry"]),
            acc={k: _sel(active_i, acc_new[k], acc[k]) for k in acc},
        )
        return new_carry, None

    xs = (jnp.arange(max_substeps), ent_dts.T, ent_cnc.T, ent_betadl.transpose(1, 0, 2), ent_lai.T)
    final_carry, _ = jax.lax.scan(_substep_body, init_carry, xs, length=max_substeps)

    acc = final_carry["acc"]
    final = accm_final(acc, static, fb, fv, final_carry["snsh_tot_carry"], final_carry["evap_tot_carry"],
                       dt, forcing["rho"], forcing["ch"], forcing["ts"], forcing["gusti"], forcing["tprime"],
                       forcing["vs"])
    # GHY_DRV.f calls evap_limits(.false.) after advnc (on the final state, after a fresh hydra, with the LAST
    # substep's Ent conductances) to get the evap_max_ij / fr_sat_ij that the next PBL call receives.
    rows = jnp.arange(N)
    last = jnp.clip(n_substeps - 1, 0, max_substeps - 1)
    pv = static["process_vege"]
    hyd_f = hydra(static, final_carry["theta"], final_carry["fice"])
    ev_f = evap_limits(static, final_carry["w"], final_carry["theta"], hyd_f["d"], final_carry["tp"],
                       final_carry["fice"], final_carry["tsn1"], final_carry["nsn"], final_carry["wsn"],
                       final_carry["fr_snow"], dt, forcing["pr"],
                       jnp.where(pv[:, None], ent_betadl[rows, last], 0.0), jnp.where(pv, ent_cnc[rows, last], 0.0),
                       forcing["ch"], forcing["vs"], forcing["rho"], forcing["pres"], forcing["qs"],
                       forcing["gusti"], forcing["qprime"], forcing["qm1"], jnp.where(pv, ent_lai[rows, last], 0.0),
                       final_carry["fm"])
    bad = jnp.isnan(ev_f["evap_max_out"])
    evap_max_ij = jnp.where(bad, 0.0, ev_f["evap_max_out"])
    fr_sat_ij = jnp.where(bad, 0.0, ev_f["fr_sat"])
    return dict(w=final_carry["w"], ht=final_carry["ht"], nsn=final_carry["nsn"], dzsn=final_carry["dzsn"],
               wsn=final_carry["wsn"], hsn=final_carry["hsn"], fr_snow=final_carry["fr_snow"],
               tp=final_carry["tp"], fice=final_carry["fice"],
               tbcs=final["tbcs"], tsns=final["tsns"], ashg=acc["ashg"], alhg=acc["alhg"],
               aevap=final["aevap"], aruns=final["aruns"], arunu=final["arunu"], aeruns=acc["aeruns"],
               aerunu=acc["aerunu"], ae0=acc["ae0"], abetad=final_carry["abetad"],
               evap_max_ij=evap_max_ij, fr_sat_ij=fr_sat_ij)


def retp(static, w, ht, wsn, hsn):
    """GHY.f retp. Returns tp(N,NGM+1,2), fice(N,NGM+1,2) -- k=0 valid only for ibv=1 (kk=1-ibv in
    Fortran terms: bare starts at k=1, vegetated includes canopy k=0) -- and tsn1(N,2), the top
    snow-layer temperature used by sensible_heat, NOT gated by process_bare/vege (ghy_ref's
    `for ibv in range(i_bare,i_vege+1)` here still only fires for active ibv, so inactive stays 0;
    tsn1 has its own condition on wsn[0,ibv] independent of ibv-activity in general)."""
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

    process_bare, process_vege = static['process_bare'], static['process_vege']
    ibv_active = jnp.stack([process_bare, process_vege], axis=-1)
    wsn0 = wsn[:, 0, :]; hsn0 = hsn[:, 0, :]
    wsn0_safe = jnp.where(wsn0 == 0.0, 1.0, wsn0)
    cond_tsn1 = (wsn0 > 1e-6) & ((hsn0 + wsn0 * FSN) < 0.0) & ibv_active
    tsn1 = jnp.where(cond_tsn1, (hsn0 + wsn0 * FSN) / (wsn0_safe * SHI), 0.0)
    return dict(tp=tp, fice=fice, tsn1=tsn1)
