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
       evapvs, evapvd, fw, fd, fm):
    """GHY.f flg. Fills the OFFSET position 0 (Fortran f(1)) of `f`, plus canopy flux `fc`(N,2) and
    evap_tot(N,2). irrig is always 0 in this rundeck (ghy_ref hardcodes self.irrig=zeros in __init__
    regardless of the forcing dict's irrig value -- dead input, not used, so omitted here)."""
    process_bare, process_vege = static['process_bare'], static['process_vege']
    f0_bare = (-flmlt[:, 0] * fr_snow[:, 0] - flmlt_scale[:, 0]
              - (dripw[:, 0] - evapb) * (1.0 - fr_snow[:, 0]))
    f0_vege = (-flmlt[:, 1] * fr_snow[:, 1] - flmlt_scale[:, 1]
              - (dripw[:, 1] - evapvg) * (1.0 - fr_snow[:, 1]))
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
        snshs, thrmsn, fr_snow, srht, trht, htpr, fw, fd, fm):
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
                  + thrm_soil0 - srht - trht) * (1.0 - fr_snow[:, 0]))
    fh0_vege = (-fhsng[:, 1] * fr_snow[:, 1] - fhsng_scale[:, 1]
               + (-htdripw[:, 1] + evapvg * (ELH + SHV * tp[:, 1, 1]) + snshg[:, 1]
                  + thrm_soil1 - thrm_can) * (1.0 - fr_snow[:, 1]))
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
