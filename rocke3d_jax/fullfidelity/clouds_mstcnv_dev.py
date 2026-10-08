"""D189: MSTCNV as ONE device program (no host loop, no per-iteration host sync) with FUSED libimf callbacks.

Same arithmetic, same operation order, same REAL(4) literals as clouds_mstcnv_jax (D146) with the D183 libimf semantics (every exp / pow the real build
routes through libimf is evaluated by the libimf callback).  What changed, and nothing else:

 1. The cloud-base loop LMIN = 1..LMCM-1 is a lax.fori_loop INSIDE one jit (`_events_device`).  Per iteration: the base test (`_base_dev`), the column mask,
    the compaction (`jnp.nonzero(size=)`) and a lax.switch over bucket sizes (branch 0 = no event column: skip) replace D146's host loop with its mask read
    (np.asarray(mask)) and the per-iteration 'negative cloud' read (bool(err)).  The error flag is carried and returned; the caller reads it once.
 2. libimf calls are FUSED (libimf_fused.fx): independent exp / pow of one dependency level go into ONE host callback, evaluated only on the lanes that can still
    matter (lane masks), and the callback is skipped on the device (lax.cond) when no lane is selected.  Dependency levels are those of the arithmetic:
      ascent level L:   CB0 [E1 = exp of the plume-saturation test, exp of the downdraft-trigger qsat, exp of the first condensation iteration, pow(pl/1000,.4),
                              pow(1000/pl,.4)], CB1/CB2 [2nd and 3rd condensation iteration, sequential by construction], CB3 [FLAMW/G/I and the four
                              diameter pows of conv_micro], CB4 [the three flam**4 and six exp(-flam*dc) of the Marshall-Palmer integrals]  = 5 per level
                              (D183: about 37).
      downdraft / precipitation layer: 3 per layer (the three sequential iterations of get_dq), only on the lanes that are active and have cond > 0.
      mass flux:         1 callback per LMIN for the four trip-invariant exp (all lanes) + 1 per trip (only the active lanes).
      post:              1 callback for the cloud-radius pow of all layers (D183: 56).
    Speculation is bitwise safe: a request is evaluated with the lane mask of the level BEFORE the lane death test of that level; lanes that die afterwards
    are discarded by the same selects as before, lanes that survive have exactly the arguments of the sequential code.
 3. Dead code removed: the second qsat of the ascent (`qsatmp = wb(ms_, ...)`) is never read afterwards.  Duplicates removed: pow(pl/1000, .4) (4 identical
    evaluations), the 8 Marshall-Palmer calls of conv_micro have 6 distinct (exp) arguments (two pairs of identical arguments).  The trip-invariant qsat of
    MASS_FLUX are evaluated once (their arguments do not depend on the trip).
Values are those of libimf_ops (scalar libimf exp / pow per element) because the host loop calls the same scalar functions.  The QUS subsidence
(clouds_mstcnv_batch._adv_groups) is still a NumPy host callback inside the event branch (see the D189 ledger).
Usage:  o = mstcnv_dev(R, c)  (dict like clouds_mstcnv_jax.mstcnv_jax);  device pieces: _events_device / _post_dev for the D186 phase-1 wiring.
"""
import clouds_jax_env  # noqa: F401  (XLA flags before jax)
import functools
import threading
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import clouds_dq_ff as dq
import clouds_helpers_ff as hp
import clouds_mstcnv_ff as mc
import clouds_mstcnv_jax as cmj
import libimf_fused as FX
from clouds_mstcnv_jax import (F, I64, LM, NK, NMOM, S_NAMES, SC_NAMES, WORK_2, WORK_K, WORK_M, XYM, XYMa, ZM, ZMa, _init_V, _mg, _put, _setup,  # noqa: F401
                               _take, dqsatdt_j, make_K, pymax, pymin, thbar_j)

jax.config.update("jax_enable_x64", True)

BUCKETS = (32, 64, 128, 256, 512, 1024)      # plus N (the largest); measured nov26 event counts per LMIN: 9..1006 (see the ledger)

# ---------------------------------------------------------------------------------------------- QUS host callback (counted)
QUS = dict(calls=0, bytes=0, seconds=0.0)
_qlk = threading.Lock()


def reset_qus():
    with _qlk:
        QUS.update(calls=0, bytes=0, seconds=0.0)


def _host_adv_dev(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit):
    t0 = time.perf_counter()
    r = cmj._host_adv(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit)
    nb = sum(np.asarray(x).nbytes for x in (sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr)) + sum(x.nbytes for x in r)
    with _qlk:
        QUS["calls"] += 1
        QUS["bytes"] += nb
        QUS["seconds"] += time.perf_counter() - t0
    return r


def adv_cb_dev(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit):
    out = (jax.ShapeDtypeStruct(arr.shape, jnp.float64), jax.ShapeDtypeStruct(mom.shape, jnp.float64),
           jax.ShapeDtypeStruct(ierr.shape, ierr.dtype), jax.ShapeDtypeStruct(lerr.shape, lerr.dtype))
    return jax.pure_callback(functools.partial(_host_adv_dev, qlimit=qlimit), out, sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr)


# ---------------------------------------------------------------------------------------------- libimf-fused physics helpers
def qarg(k, tm, lh):
    """The exp argument of cmj.qsat_j:  qsat = QA_ * exp(qarg) / pr."""
    return lh * (k.QB - k.QC / jnp.maximum(130.0, tm))


def dq_dev(k, kind, sm, qm, plk, mass, lhx, pl, cond=None, sel=None, e_first=None, tag="dq"):
    """cmj.dq_j (3 fixed iterations) with the exp of every iteration a fused callback on the lanes `sel` (the lanes whose result the caller uses AND ref > 0).
    On lanes outside `sel` the exp is 0.0 (not evaluated) and the returned value is meaningless (the callers select it away).  Lanes with ref <= 0 return 0 as
    dq_j does.  e_first: the exp of iteration 1 when it was evaluated earlier (same argument)."""
    sign = 1.0 if kind == "c" else -1.0
    shp = jnp.broadcast_shapes(*(jnp.shape(a) for a in (sm, qm, plk, mass, lhx, pl)))
    bc = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float64), shp)  # noqa: E731
    sm, qm, plk, mass, lhx, pl = (bc(a) for a in (sm, qm, plk, mass, lhx, pl))
    slh = lhx * k.DQ_BYSHA
    qmt = qm
    tp = sm * plk / mass
    dqsum = jnp.zeros(shp)
    ref = qm if kind == "c" else bc(cond)
    if sel is None:
        sel = ref > 0
    for it in range(dq.NITER):
        if it == 0 and e_first is not None:
            e = e_first
        else:
            e = FX.fx([("e", qarg(k, tp, lhx), sel)], tag)[0]
        qst = k.QA_ * e / pl
        d = (qmt - mass * qst) / (1.0 + slh * qst * dqsatdt_j(k, tp, lhx))
        tp = tp + slh * d / mass
        qmt = qmt - d
        dqsum = dqsum + sign * d
    dqs = jnp.maximum(0.0, jnp.minimum(dqsum, ref))
    fc = dqs / ref
    act = ref > 0
    return jnp.where(act, dqs, 0.0), jnp.where(act, fc, 0.0)


def dq_sum_dev(k, sm, qm, plk, mass, lhx, pl, sel, tag="dq"):
    """The three iterations of cmj.dq_j("e") up to dqsum (before the clamp against `cond`), for arrays of any common shape; exp as one fused callback per
    iteration on the lanes `sel`."""
    slh = lhx * k.DQ_BYSHA
    qmt = qm
    tp = sm * plk / mass
    dqsum = jnp.zeros(jnp.shape(sm))
    for _ in range(dq.NITER):
        e = FX.fx([("e", qarg(k, tp, lhx), sel)], tag)[0]
        qst = k.QA_ * e / pl
        d = (qmt - mass * qst) / (1.0 + slh * qst * dqsatdt_j(k, tp, lhx))
        tp = tp + slh * d / mass
        qmt = qmt - d
        dqsum = dqsum + (-1.0) * d
    return dqsum


def mass_flux_dev(k, cand, e_inv, lhx0, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg, airm0, airm1, byam0, byam1, byam2, sm2, qm2, plk0, plk1, pl0, pl1):
    """cmj.mass_flux_j; the three trip-invariant qsat come from e_inv (their exp were evaluated once), the per-trip qsat is a fused callback on the lanes that
    are still iterating.  Only the lanes `cand` are iterated (others are never used by the caller)."""
    shp = qmo1.shape
    fplume = jnp.full(shp, 0.25)
    dfp = jnp.full(shp, 0.25)
    fmp2 = jnp.zeros(shp)
    lhx = jnp.full(shp, 1.0) * lhx0
    tp = smo1 * plk1 * byam0
    qsat_tp = k.QA_ * e_inv[0] / pl1
    tnx1 = smo2 * plk1 * byam1
    qsatc = k.QA_ * e_inv[1] / pl1
    tnx0 = smo1 * plk0 * byam0
    qsatc2 = k.QA_ * e_inv[2] / pl0

    def cond_f(c):
        i, fplume, dfp, fmp2, active = c
        return (i < 8) & active.any()

    def body(c):
        i, fplume, dfp, fmp2, active = c
        a = active
        dfp_n = dfp * 0.5
        fmp2_n = fplume * airm0
        frat1 = fmp2_n * byam1
        frat2 = fmp2_n * byam2
        smn1 = smo1 * (1.0 - fplume) + frat1 * smo2
        qmn1 = qmo1 * (1.0 - fplume) + frat1 * qmo2
        smn2 = smo2 * (1.0 - frat1) + frat2 * sm2
        qmn2 = qmo2 * (1.0 - frat1) + frat2 * qm2
        smp = smo1 * fplume
        qmp = qmo1 * fplume
        qsatmp = fmp2_n * qsat_tp
        gama = slh * qsatmp * dqsatdt_j(k, tp, lhx) / fmp2_n
        dqsum_n = (qmp - qsatmp) / (1.0 + gama)
        blk = dqsum_n > 0.0
        fevap = 0.5 * fplume
        mcloud = fevap * airm1
        dqe = mcloud * (qsatc - qmo2 * byam1) / (1.0 + slh * qsatc * dqsatdt_j(k, tnx1, lhx))
        dqe = jnp.where(dqe > dqsum_n, dqsum_n, dqe)
        smn2 = jnp.where(blk, smn2 - slh * dqe / plk1, smn2)
        qmn2 = jnp.where(blk, qmn2 + dqe, qmn2)
        dqsum_b = jnp.where(blk, dqsum_n - dqe, dqsum_n)
        blk2 = blk & (dqsum_b > 0.0)
        mcloud2 = fevap * airm0
        qnx0 = qmo1 * byam0
        dq2 = mcloud2 * (qsatc2 - qnx0) / (1.0 + slh * qsatc2 * dqsatdt_j(k, tnx0, lhx))
        dq2 = jnp.where(dq2 > dqsum_b, dqsum_b, dq2)
        smn1 = jnp.where(blk2, smn1 - slh * dq2 / plk0, smn1)
        qmn1 = jnp.where(blk2, qmn1 + dq2, qmn1)
        sdn = smn1 * byam0
        sup = smn2 * byam1
        sedge = thbar_j(k, sup, sdn)
        qdn = qmn1 * byam0
        qup = qmn2 * byam1
        svdn = sdn * (1.0 + k.DELTX * qdn - wmdn)
        svup = sup * (1.0 + k.DELTX * qup - wmup)
        qedge = 0.5 * (qup + qdn)
        svedg = sedge * (1.0 + k.DELTX * qedge - wmedg)
        e_t = FX.fx([("e", qarg(k, sup * plk1, lhx), a)], "mf_trip")[0]
        dmse = ((svup - svedg) * plk1 + (svedg - svdn) * plk0 + k.SLHE * (k.QA_ * e_t / pl1 - qdn))
        dfp = jnp.where(a, dfp_n, dfp)
        fmp2 = jnp.where(a, fmp2_n, fmp2)
        ex = a & (jnp.abs(dmse) <= 1.0e-3)
        up = a & ~ex
        fplume = jnp.where(up & (dmse > 1.0e-3), fplume - dfp, fplume)
        fplume = jnp.where(up & (dmse < -1.0e-3), fplume + dfp, fplume)
        return i + 1, fplume, dfp, fmp2, a & ~ex

    _, fplume, dfp, fmp2, active = lax.while_loop(cond_f, body, (jnp.zeros((), I64), fplume, dfp, fmp2, cand))
    return fplume, fmp2


def conv_micro_dev(k, pl, wcu, dwcu, lfrz, wcufrz, tp, pland, condmu, tlmin, tlmin1, condip_in, condgp_in, a, p04, pfac1):
    """cmj.conv_micro_j with the libimf calls fused (two callbacks: [flam x3, diameters x4] and [flam**4 x3, exp x6]).  flam* are computed here from condmu
    (cmj computes them in the caller under the mask a; lanes outside a are unused by the caller).  p04 = pow(pl/1000, F04), pfac1 = pow(1000/pl, P4) were
    evaluated with the ascent's first callback (identical arguments)."""
    n = pl.shape
    wv = jnp.maximum(wcu - dwcu, 0.0)
    wvu = wcu + dwcu
    r = FX.fx([("p", 1000.0 * k.PI * k.CN0 / (condmu + k.TEENY), k.QA, a), ("p", 400.0 * k.PI * k.CN0G / (condmu + k.TEENY), k.QA, a),
               ("p", 100.0 * k.PI * k.CN0I / (condmu + k.TEENY), k.QA, a),
               ("p", (wv / k.F193) * p04, k.F27, a), ("p", (wv / k.F1172) * p04, k.F2439, a),
               ("p", (wvu / k.F1172) * p04, k.F2439, a), ("p", (wvu / k.F193) * p04, k.F27, a)], "cm_flam_dc")
    flamw, flamg, flami = r[0], r[1], r[2]
    dcg = jnp.minimum(r[3], 1e-2)
    dci = jnp.minimum(r[4], 1e-2)
    dci_u = jnp.minimum(r[5], 1e-2)
    dcg_u = jnp.minimum(r[6], 1e-2)
    tig = jnp.where(lfrz == 0, k.TF, k.TF - 4.0 * wcufrz)
    tig = jnp.where(tig < k.TI - 10.0, k.TI - 10.0, tig)
    water = tp >= k.TF
    ice = ~water & (tp <= tig)
    mixed = ~water & ~ice
    ddcw = jnp.where(pland < 0.5, 1.5e-3 * k.FITMAX, 6e-3 * k.FITMAX)
    wv2 = jnp.concatenate([wv, wvu])
    ddcw2 = jnp.concatenate([ddcw, ddcw])
    wmax2 = jnp.full(wv2.shape, 1.0) * k.WMAX
    pfac = jnp.concatenate([pfac1, pfac1])

    def sbody(_, c):
        dcw, active = c
        vt = (-0.267 + dcw * (5.15e3 - dcw * (1.0225e6 - 7.55e7 * dcw))) * pfac
        ex1 = (vt >= 0.0) & (vt >= wv2)
        ex2 = ~ex1 & (vt > wmax2)
        ex = active & (ex1 | ex2)
        dcw = jnp.where(active & ~ex, dcw + ddcw2, dcw)
        return dcw, active & ~ex

    dcw_both, _ = lax.fori_loop(0, int(mc.ITMAX) - 1, sbody, (jnp.zeros(wv2.shape), jnp.ones(wv2.shape, bool)))
    dcw1, dcw2 = dcw_both[:n[0]], dcw_both[n[0]:]
    r2 = FX.fx([("p", flamw, 4.0, a), ("p", flami, 4.0, a), ("p", flamg, 4.0, a),
                ("e", -flamw * dcw1, a), ("e", -flamw * dcw2, a), ("e", -flami * dci, a), ("e", -flami * dci_u, a),
                ("e", -flamg * dcg, a), ("e", -flamg * dcg_u, a)], "cm_mp")
    f4w, f4i, f4g = r2[0], r2[1], r2[2]
    ex_w1, ex_w2, ex_i, ex_iu, ex_g, ex_gu = r2[3:9]

    def mp(rho, flam, dc, cn, f4p, e):
        return (rho * (k.PI * hp.BY6) * cn * e
                * (dc * dc * dc / flam + 3.0 * dc * dc / (flam * flam) + k.SIX * dc / (flam * flam * flam) + 6.0 / f4p))

    condp1_w = mp(k.RHOW, flamw, dcw1, k.CN0, f4w, ex_w1)
    condp_w = mp(k.RHOW, flamw, dcw2, k.CN0, f4w, ex_w2)
    condp1_i = mp(k.RHOIP, flami, dci, k.CN0I, f4i, ex_i)
    condp_i = mp(k.RHOIP, flami, dci_u, k.CN0I, f4i, ex_iu)
    fg = (tp - tig) / ((k.TF - tig) + k.TEENY)
    fg = jnp.where(fg > 1.0, 1.0, fg)
    fg = jnp.where(fg < 0.0, 0.0, fg)
    fg = jnp.where((tlmin <= k.TF) | (tlmin1 <= k.TF), 0.0, fg)
    fi = 1.0 - fg
    cip_l = condp1_i          # mp(RHOIP, flami, dci, CN0I): identical arguments to condp1_i
    cgp_l = mp(k.RHOG, flamg, dcg, k.CN0G, f4g, ex_g)
    condp1_m = fg * cgp_l + fi * cip_l
    cip_u = condp_i           # mp(RHOIP, flami, dci_u, CN0I): identical arguments to condp_i
    cgp_u = mp(k.RHOG, flamg, dcg_u, k.CN0G, f4g, ex_gu)
    condp_m = fg * cgp_u + fi * cip_u
    condp = jnp.where(water, condp_w, jnp.where(ice, condp_i, condp_m))
    condp1 = jnp.where(water, condp1_w, jnp.where(ice, condp1_i, condp1_m))
    condip = jnp.where(mixed, cip_u, condip_in)
    condgp = jnp.where(mixed, cgp_u, condgp_in)
    return condp, condp1, condip, condgp


def _event_dev(K, lmin, ic, fmp0, fmp2, valid, S, Sc, I, V, KC):
    k = SimpleNamespace(**K)
    n = fmp0.shape[0]
    cols = jnp.arange(n)
    dcl, dwcu, pland = KC["dcl"], KC["dwcu"], KC["pland"]
    pl, ple, plk, airm, byam, etal = I["pl"], I["ple"], I["plk"], I["airm"], I["byam"], I["etal"]
    tl, tvl, qcll, qcil, wturb, gzl = I["tl"], I["tvl"], I["qcll"], I["qcil"], I["wturb"], I["gzl"]
    smold, qmold, smomold, qmomold, u_0, v_0, ra = I["smold"], I["qmold"], I["smomold"], I["qmomold"], I["u_0"], I["v_0"], I["ra"]
    dtsrc, bydtsrc, xmass = K["dtsrc"], K["bydtsrc"], K["xmass"]
    T = {}
    T.update(S)
    T.update(Sc)
    T.update(V)
    T["err_neg_cloud"] = jnp.zeros((), bool)
    wb = jnp.where
    lm1 = lmin + 1

    T["mc1"] = jnp.zeros(n, bool)
    mplume0 = pymin(pymin(airm[lmin], airm[lm1]), fmp2)
    fctype = wb(mplume0 > fmp0, fmp0 / mplume0, 1.0)
    fctype = wb(ic == 2, 1. - fctype, fctype)
    live = valid & ~(fctype < k.f001)
    T["mplume"], T["fctype"] = mplume0, fctype
    mplum1 = mplume0
    contce = wb(ic == 1, k.contce1, k.contce2)

    # ================================================================ NPPL loop (area partitions)
    def nppl_body(nppl, carry):
        T, cyc = carry
        s = SimpleNamespace(**T)
        run = live & ~cyc
        run = wb(nppl == 2, run & (s.mc1 & (s.mccont >= 2)), run)
        s.mplume = wb(run, mplum1 * s.fctype, s.mplume)
        c1 = run & ((s.mccont == 0) | s.mc1)
        fsub_tmp = wb(s.mccont == 0, 1.0 + (airm[lm1] - 100.0) / 200.0, 1.0 + (pl[lmin] - pl[s.lmax, cols] - 100.0) / 200.0)
        fconv_tmp = pymin(mplum1 * byam[lm1] * (k.C1800 / dtsrc), 1.0)
        t2 = 1.0 / (fconv_tmp + 1.0e-20) - 1.0
        fsub_tmp = wb(fsub_tmp > t2, t2, fsub_tmp)
        fsub_tmp = pymax(1.0, pymin(fsub_tmp, 5.0))
        fssl_tmp = 1.0 - (1.0 + fsub_tmp) * fconv_tmp
        fssl_tmp = pymax(k.CLDMIN, pymin(fssl_tmp, 1.0 - k.CLDMIN))
        s.fmc1 = wb(c1, (1.0 - fssl_tmp) + k.TEENY, s.fmc1)
        m2 = run & (s.mc1 | (s.mccont > 0))
        s.mplume = wb(m2, pymin(0.95 * airm[lmin] * s.fmc1, s.mplume), s.mplume)
        for w in WORK_2:
            setattr(s, w, wb(run[None, :], 0.0, getattr(s, w)))
        for w in WORK_M + WORK_K:
            setattr(s, w, wb(run[None, None, :], 0.0, getattr(s, w)))
        s.mplume = wb(run, pymin(s.mplume / s.fmc1, airm[lmin] * 0.95 * s.qm[lmin] / (qmold[lmin] + k.TEENY)), s.mplume)
        small = run & (s.mplume <= k.f001 * airm[lmin])
        cyc = cyc | small
        r2 = run & ~small
        # ---- plume set-up
        s.fplume = wb(r2, s.mplume * byam[lmin], s.fplume)
        s.smp = wb(r2, smold[lmin] * s.fplume, s.smp)
        new = jnp.zeros((NMOM, n)).at[XYMa].set(smomold[XYMa, lmin] * s.fplume[None, :])
        s.smomp = wb(r2[None, :], new, s.smomp)
        s.qmp = wb(r2, qmold[lmin] * s.fplume, s.qmp)
        new = jnp.zeros((NMOM, n)).at[XYMa].set(qmomold[XYMa, lmin] * s.fplume[None, :])
        s.qmomp = wb(r2[None, :], new, s.qmomp)
        s.tpsav = s.tpsav.at[lmin].set(wb(r2 & (s.tpsav[lmin] == 0), s.smp * plk[lmin] / s.mplume, s.tpsav[lmin]))
        s.dmr = s.dmr.at[lmin].set(wb(r2, -s.mplume, s.dmr[lmin]))
        s.dsmr = s.dsmr.at[lmin].set(wb(r2, -s.smp, s.dsmr[lmin]))
        for mm in XYM:
            s.dsmomr = s.dsmomr.at[mm, lmin].set(wb(r2, -s.smomp[mm], s.dsmomr[mm, lmin]))
            s.dqmomr = s.dqmomr.at[mm, lmin].set(wb(r2, -s.qmomp[mm], s.dqmomr[mm, lmin]))
        for mm in ZM:
            s.dsmomr = s.dsmomr.at[mm, lmin].set(wb(r2, -smomold[mm, lmin] * s.fplume, s.dsmomr[mm, lmin]))
            s.dqmomr = s.dqmomr.at[mm, lmin].set(wb(r2, -qmomold[mm, lmin] * s.fplume, s.dqmomr[mm, lmin]))
        s.dqmr = s.dqmr.at[lmin].set(wb(r2, -s.qmp, s.dqmr[lmin]))
        s.ump = wb(r2[None, :], s.um[:, lmin] * s.fplume, s.ump)
        s.dum = s.dum.at[:, lmin].set(wb(r2[None, :], -s.ump, s.dum[:, lmin]))
        s.vmp = wb(r2[None, :], s.vm[:, lmin] * s.fplume, s.vmp)
        s.dvm = s.dvm.at[:, lmin].set(wb(r2[None, :], -s.vmp, s.dvm[:, lmin]))
        s.cdhsum, s.cdhsum1, s.cdhdrt = wb(r2, 0.0, s.cdhsum), wb(r2, 0.0, s.cdhsum1), wb(r2, 0.0, s.cdhdrt)
        s.etadn = wb(r2, 0.0, s.etadn)
        s.ldraft = wb(r2, LM, s.ldraft)
        s.evpsum = wb(r2, 0.0, s.evpsum)
        s.ddraft = wb(r2, 0.0, s.ddraft)
        s.lfrz = wb(r2, 0, s.lfrz)
        s.lmax = wb(r2, lmin, s.lmax)
        wc0 = pymax(.5, wturb[lm1])
        wc0 = wb(ic == 1, pymax(.5, 2.0 * wturb[lm1]), wc0)
        s.wcu = s.wcu.at[lmin].set(wb(r2, wc0, s.wcu[lmin]))
        s.wcu2 = s.wcu2.at[lmin].set(wb(r2, s.wcu[lmin] * s.wcu[lmin], s.wcu2[lmin]))
        s.mpmax, s.smpmax, s.qmpmax = wb(r2, 0.0, s.mpmax), wb(r2, 0.0, s.smpmax), wb(r2, 0.0, s.qmpmax)
        s.smompmax = wb(r2[None, :], 0.0, s.smompmax)
        s.qmompmax = wb(r2[None, :], 0.0, s.qmompmax)
        # ---- plume ascent
        Ts = vars(s)

        def asc_cond(c):
            L, a, Tc = c
            return (L <= LM) & a.any()

        def asc_body(c):
            L, a, Tc = c
            Tn, a = _ascent_step_dev(k, Tc, L, a, lmin, contce, ic, K, I, KC, cols, n)
            return L + 1, a, Tn

        L0 = jnp.asarray(lmin + 1, I64)
        _, _, Ts = lax.while_loop(asc_cond, asc_body, (L0, r2, Ts))
        s = SimpleNamespace(**Ts)
        cyc = cyc | (r2 & (s.lmax == lmin))
        return vars(s), cyc

    T, cyc = lax.fori_loop(1, 3, nppl_body, (T, jnp.zeros(n, bool)))
    s = SimpleNamespace(**T)
    ev = live & ~cyc
    # ================================================================ after the plume partitions
    def tm_body(L, tm):
        return wb(ev & (L <= s.lmax), tm + s.taumc1[L], tm)
    # taumcl[L] = taumcl[L] + taumc1[L] for L in lmin..LM (layer-wise independent)
    Ls = jnp.arange(LM + 2)[:, None]
    s.taumcl = wb(ev[None, :] & (Ls >= lmin) & (Ls <= s.lmax[None, :]) & (Ls <= LM), s.taumcl + s.taumc1, s.taumcl)
    mA = pl[lmin] < pl[dcl, cols]
    s.u00l = s.u00l.at[lmin].set(wb(ev & mA, k.u00_mc, s.u00l[lmin]))
    s.u00l = wb(ev[None, :] & ~mA[None, :] & (Ls >= 1) & (Ls <= lmin), k.u00_mc, s.u00l)
    s.lfrz = wb(ev & (s.tpsav[s.lmax, cols] >= k.TF), s.lmax, s.lfrz)
    lme = s.lmax
    s.u00l = s.u00l.at[lme, cols].set(wb(ev, k.u00a, s.u00l[lme, cols]))
    s.dm = s.dm.at[lme, cols].set(wb(ev, s.dm[lme, cols] + s.mpmax, s.dm[lme, cols]))
    s.dsm = s.dsm.at[lme, cols].set(wb(ev, s.dsm[lme, cols] + s.smpmax, s.dsm[lme, cols]))
    xr = XYMa[:, None]
    s.dsmom = s.dsmom.at[xr, lme[None, :], cols[None, :]].set(wb(ev[None, :], s.dsmom[xr, lme[None, :], cols[None, :]] + s.smompmax[XYMa],
                                                                  s.dsmom[xr, lme[None, :], cols[None, :]]))
    s.dqm = s.dqm.at[lme, cols].set(wb(ev, s.dqm[lme, cols] + s.qmpmax, s.dqm[lme, cols]))
    s.dqmom = s.dqmom.at[xr, lme[None, :], cols[None, :]].set(wb(ev[None, :], s.dqmom[xr, lme[None, :], cols[None, :]] + s.qmompmax[XYMa],
                                                                  s.dqmom[xr, lme[None, :], cols[None, :]]))
    s.ccm = s.ccm.at[lme, cols].set(wb(ev, 0., s.ccm[lme, cols]))
    kr = jnp.arange(NK)[:, None]
    s.dum = s.dum.at[kr, lme[None, :], cols[None, :]].set(wb(ev[None, :], s.dum[kr, lme[None, :], cols[None, :]] + s.ump, s.dum[kr, lme[None, :], cols[None, :]]))
    s.dvm = s.dvm.at[kr, lme[None, :], cols[None, :]].set(wb(ev[None, :], s.dvm[kr, lme[None, :], cols[None, :]] + s.vmp, s.dvm[kr, lme[None, :], cols[None, :]]))
    s.lmcmin = wb(ev & (s.lmcmin == 0), lmin, s.lmcmin)
    s.lmcmax = wb(ev & (s.lmcmax < s.lmax), s.lmax, s.lmcmax)
    # ================================================================ downdraft
    s.ldmin = wb(ev, s.ldraft - 1, s.ldmin)
    s.llmin = wb(ev, s.ldmin, s.llmin)
    ddm_ev = ev & (s.etadn > 1e-10)
    ld = s.ldraft
    ddraft = wb(ddm_ev, s.ddr[ld, cols], s.ddraft)
    ddrold = wb(ddm_ev, ddraft, jnp.zeros(n))
    smdn = wb(ddm_ev, s.smdnl[ld, cols], 0.0)
    qmdn = wb(ddm_ev, s.qmdnl[ld, cols], 0.0)
    smomdn = wb(ddm_ev[None, :], jnp.zeros((NMOM, n)).at[XYMa].set(_mg(s.smomdnl, ld, cols)[XYMa]), 0.0)
    qmomdn = wb(ddm_ev[None, :], jnp.zeros((NMOM, n)).at[XYMa].set(_mg(s.qmomdnl, ld, cols)[XYMa]), 0.0)
    umdn = wb(ddm_ev[None, :], s.umdnl[:, ld, cols], 0.0)
    vmdn = wb(ddm_ev[None, :], s.vmdnl[:, ld, cols], 0.0)
    edraft = jnp.zeros(n)
    DD = dict(ddraft=ddraft, ddrold=ddrold, smdn=smdn, qmdn=qmdn, smomdn=smomdn, qmomdn=qmomdn, umdn=umdn, vmdn=vmdn, edraft=edraft,
              cond=s.cond, taumcl=s.taumcl, cdheat=s.cdheat, evpsum=s.evpsum, dsmr=s.dsmr, dsmomr=s.dsmomr, dqmr=s.dqmr, dqmomr=s.dqmomr,
              dum=s.dum, dvm=s.dvm, dsm=s.dsm, dsmom=s.dsmom, dqm=s.dqm, dqmom=s.dqmom, dmr=s.dmr, dm=s.dm, ddm=s.ddm,
              ldmin=s.ldmin, llmin=s.llmin, alive=ddm_ev)

    def dd_body(j, D):
        L = LM - j
        d = SimpleNamespace(**D)
        dact = d.alive & (L <= s.ldraft)
        lhx = s.vlat[L]
        slh = lhx * k.BYSHA
        dqsum, fq1 = dq_dev(k, "e", d.smdn, d.qmdn, plk[L], d.ddraft, lhx, pl[L], d.cond[L], dact & (d.cond[L] > 0), tag="dd_dq")
        dqevp = k.fddrt * d.cond[L]
        dqevp = wb(dqevp > dqsum, dqsum, dqevp)
        dqevp = wb(dqevp > d.smdn * plk[L] / slh, d.smdn * plk[L] / slh, dqevp)
        dqevp = wb(L < lmin, 0.0, dqevp)
        fsevp = wb(plk[L] * d.smdn > k.TEENY, slh * dqevp / (plk[L] * d.smdn), 0.0)
        d.smdn = wb(dact, d.smdn - slh * dqevp / plk[L], d.smdn)
        d.smomdn = wb(dact[None, :], d.smomdn * (1. - fsevp)[None, :], d.smomdn)
        d.smomdn = d.smomdn   # (only XYMOMS are ever non-zero; same expression on all rows as numpy applies to XYMa rows)
        d.qmdn = wb(dact, d.qmdn + dqevp, d.qmdn)
        d.cond = d.cond.at[L].set(wb(dact, d.cond[L] - dqevp, d.cond[L]))
        d.taumcl = d.taumcl.at[L].set(wb(dact, d.taumcl[L] - dqevp * s.fmc1, d.taumcl[L]))
        d.cdheat = d.cdheat.at[L].set(wb(dact, d.cdheat[L] - dqevp * slh, d.cdheat[L]))
        d.evpsum = wb(dact, d.evpsum + dqevp * slh, d.evpsum)
        mu = dact & (L < s.ldraft) & (L > 1)
        ddrup = d.ddraft
        ddr_new = d.ddraft + d.ddrold * etal[L]
        ddr_new = wb(ddrup > ddr_new, ddrup, ddr_new)
        smix = d.smdn / (ddrup + k.TEENY)
        qmix = d.qmdn / (ddrup + k.TEENY)
        wmix = d.cond[L] / (ddrup + k.TEENY)
        svmix = smix * plk[L - 1] * (1. + k.DELTX * qmix - wmix)
        svm1 = s.sm1[L - 1] * byam[L - 1] * plk[L - 1] * (1. + k.DELTX * s.qm1[L - 1] * byam[L - 1] - qcll[L - 1] - qcil[L - 1])
        ddr_new = wb(svmix - svm1 >= k.DTMIN1, k.FDDET * ddrup, ddr_new)
        ddr_new = wb(ddr_new > .95 * (airm[L - 1] + d.dmr[L - 1]), .95 * (airm[L - 1] + d.dmr[L - 1]), ddr_new)
        d.ddraft = wb(mu, ddr_new, d.ddraft)
        d.edraft = wb(mu, d.ddraft - ddrup, d.edraft)
        e1 = mu & (d.edraft > 0)
        e2 = mu & ~(d.edraft > 0)
        # e1 branch
        fentra = d.edraft * byam[L]
        senv = s.sm[L] * byam[L]
        qenv = s.qm[L] * byam[L]
        smdn1 = d.smdn + d.edraft * senv
        qmdn1 = d.qmdn + d.edraft * qenv
        smomdn1 = d.smomdn.at[XYMa].set(d.smomdn[XYMa] + s.smom[XYMa, L] * fentra)
        qmomdn1 = d.qmomdn.at[XYMa].set(d.qmomdn[XYMa] + s.qmom[XYMa, L] * fentra)
        umtemp = k.pgrad * (d.ddraft * d.ddraft) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
        vmtemp = k.pgrad * (d.ddraft * d.ddraft) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
        d.dsmr = d.dsmr.at[L].set(wb(e1, d.dsmr[L] - d.edraft * senv, d.dsmr[L]))
        d.dsmomr = d.dsmomr.at[:, L].set(wb(e1, d.dsmomr[:, L] - s.smom[:, L] * fentra, d.dsmomr[:, L]))
        d.dqmr = d.dqmr.at[L].set(wb(e1, d.dqmr[L] - d.edraft * qenv, d.dqmr[L]))
        d.dqmomr = d.dqmomr.at[:, L].set(wb(e1, d.dqmomr[:, L] - s.qmom[:, L] * fentra, d.dqmomr[:, L]))
        d.dmr = d.dmr.at[L].set(wb(e1, d.dmr[L] - d.edraft, d.dmr[L]))
        umdn1 = d.umdn + fentra * s.um[:, L] - umtemp
        vmdn1 = d.vmdn + fentra * s.vm[:, L] - vmtemp
        dum1 = d.dum[:, L] - fentra * s.um[:, L] + umtemp
        dvm1 = d.dvm[:, L] - fentra * s.vm[:, L] + vmtemp
        # e2 branch
        fentra2 = d.edraft / (ddrup + k.TEENY)
        dsm2 = d.dsm[L] - fentra2 * d.smdn
        dsmom2 = d.dsmom[XYMa, L] - d.smomdn[XYMa] * fentra2
        dqm2 = d.dqm[L] - fentra2 * d.qmdn
        dqmom2 = d.dqmom[XYMa, L] - d.qmomdn[XYMa] * fentra2
        smdn2 = d.smdn * (1 + fentra2)
        qmdn2 = d.qmdn * (1 + fentra2)
        smomdn2 = d.smomdn.at[XYMa].set(d.smomdn[XYMa] * (1 + fentra2))
        qmomdn2 = d.qmomdn.at[XYMa].set(d.qmomdn[XYMa] * (1 + fentra2))
        dm2 = d.dm[L] - d.edraft
        dum2 = d.dum[:, L] - fentra2 * d.umdn + umtemp
        dvm2 = d.dvm[:, L] - fentra2 * d.vmdn + vmtemp
        umdn2 = d.umdn * (1. + fentra2) - umtemp
        vmdn2 = d.vmdn * (1. + fentra2) - vmtemp
        d.dsm = d.dsm.at[L].set(wb(e2, dsm2, d.dsm[L]))
        d.dsmom = d.dsmom.at[XYMa, L].set(wb(e2, dsmom2, d.dsmom[XYMa, L]))
        d.dqm = d.dqm.at[L].set(wb(e2, dqm2, d.dqm[L]))
        d.dqmom = d.dqmom.at[XYMa, L].set(wb(e2, dqmom2, d.dqmom[XYMa, L]))
        d.dm = d.dm.at[L].set(wb(e2, dm2, d.dm[L]))
        d.dum = d.dum.at[:, L].set(wb(e1[None, :], dum1, wb(e2[None, :], dum2, d.dum[:, L])))
        d.dvm = d.dvm.at[:, L].set(wb(e1[None, :], dvm1, wb(e2[None, :], dvm2, d.dvm[:, L])))
        d.smdn = wb(e1, smdn1, wb(e2, smdn2, d.smdn))
        d.qmdn = wb(e1, qmdn1, wb(e2, qmdn2, d.qmdn))
        d.smomdn = wb(e1[None, :], smomdn1, wb(e2[None, :], smomdn2, d.smomdn))
        d.qmomdn = wb(e1[None, :], qmomdn1, wb(e2[None, :], qmomdn2, d.qmomdn))
        d.umdn = wb(e1[None, :], umdn1, wb(e2[None, :], umdn2, d.umdn))
        d.vmdn = wb(e1[None, :], vmdn1, wb(e2[None, :], vmdn2, d.vmdn))
        d.ldmin = wb(dact, L, d.ldmin)
        d.llmin = wb(dact, d.ldmin, d.llmin)
        Lm = jnp.maximum(L - 1, 0)
        smix = d.smdn / (d.ddraft + k.TEENY)
        qmix = d.qmdn / (d.ddraft + k.TEENY)
        wmix = d.cond[Lm] / (d.ddraft + k.TEENY)
        svmix = smix * plk[Lm] * (1. + k.DELTX * qmix - wmix)
        svm1 = s.sm1[Lm] * byam[Lm] * plk[Lm] * (1. + k.DELTX * s.qm1[Lm] * byam[Lm] - qcll[Lm] - qcil[Lm])
        brk = dact & (L > 1) & (L <= lmin) & (svmix >= svm1)
        cont = dact & (L > 1) & ~brk
        d.ddm = d.ddm.at[Lm].set(wb(cont, d.ddraft, d.ddm[Lm]))
        d.ddrold = wb(cont, d.ddraft, d.ddrold)
        d.ddraft = wb(cont, d.ddraft + s.ddr[Lm], d.ddraft)
        d.smdn = wb(cont, d.smdn + s.smdnl[Lm], d.smdn)
        d.qmdn = wb(cont, d.qmdn + s.qmdnl[Lm], d.qmdn)
        d.smomdn = d.smomdn.at[XYMa].set(wb(cont, d.smomdn[XYMa] + s.smomdnl[XYMa, Lm], d.smomdn[XYMa]))
        d.qmomdn = d.qmomdn.at[XYMa].set(wb(cont, d.qmomdn[XYMa] + s.qmomdnl[XYMa, Lm], d.qmomdn[XYMa]))
        d.umdn = wb(cont[None, :], d.umdn + s.umdnl[:, Lm], d.umdn)
        d.vmdn = wb(cont[None, :], d.vmdn + s.vmdnl[:, Lm], d.vmdn)
        d.alive = d.alive & ~brk
        return vars(d)

    D = lax.fori_loop(0, LM, dd_body, DD)
    d = SimpleNamespace(**D)
    for nm in ("cond", "taumcl", "cdheat", "evpsum", "dsmr", "dsmomr", "dqmr", "dqmomr", "dum", "dvm", "dsm", "dsmom", "dqm", "dqmom", "dmr", "dm", "ddm",
               "ldmin", "llmin"):
        setattr(s, nm, getattr(d, nm))
    s.ddraft = d.ddraft
    ld_ = s.ldmin
    s.dsm = s.dsm.at[ld_, cols].set(wb(ddm_ev, s.dsm[ld_, cols] + d.smdn, s.dsm[ld_, cols]))
    s.dsmom = s.dsmom.at[xr, ld_[None, :], cols[None, :]].set(wb(ddm_ev[None, :], s.dsmom[xr, ld_[None, :], cols[None, :]] + d.smomdn[XYMa],
                                                                  s.dsmom[xr, ld_[None, :], cols[None, :]]))
    s.dqm = s.dqm.at[ld_, cols].set(wb(ddm_ev, s.dqm[ld_, cols] + d.qmdn, s.dqm[ld_, cols]))
    s.dqmom = s.dqmom.at[xr, ld_[None, :], cols[None, :]].set(wb(ddm_ev[None, :], s.dqmom[xr, ld_[None, :], cols[None, :]] + d.qmomdn[XYMa],
                                                                  s.dqmom[xr, ld_[None, :], cols[None, :]]))
    s.tdnl = s.tdnl.at[ld_, cols].set(wb(ddm_ev, d.smdn * plk[ld_, cols] / (d.ddraft + k.TEENY), s.tdnl[ld_, cols]))
    s.qdnl = s.qdnl.at[ld_, cols].set(wb(ddm_ev, d.qmdn / (d.ddraft + k.TEENY), s.qdnl[ld_, cols]))
    s.dum = s.dum.at[kr, ld_[None, :], cols[None, :]].set(wb(ddm_ev[None, :], s.dum[kr, ld_[None, :], cols[None, :]] + d.umdn,
                                                              s.dum[kr, ld_[None, :], cols[None, :]]))
    s.dvm = s.dvm.at[kr, ld_[None, :], cols[None, :]].set(wb(ddm_ev[None, :], s.dvm[kr, ld_[None, :], cols[None, :]] + d.vmdn,
                                                              s.dvm[kr, ld_[None, :], cols[None, :]]))
    s.dm = s.dm.at[ld_, cols].set(wb(ddm_ev, s.dm[ld_, cols] + d.ddraft, s.dm[ld_, cols]))
    # ================================================================ subsidence
    s.ldmin = wb(ev & (s.ldmin > lmin), lmin, s.ldmin)
    lmax = s.lmax
    Lr = jnp.arange(LM + 2)[:, None]
    inr = ev[None, :] & (Lr >= s.ldmin[None, :]) & (Lr <= lmax[None, :]) & (Lr >= 1) & (Lr <= LM)

    def cm_body(L, c):
        cm, smt, qmt = c
        m = ev & (L >= s.ldmin) & (L <= lmax)
        cm = cm.at[L].set(wb(m, cm[L - 1] - s.dm[L] - s.dmr[L], cm[L]))
        smt = smt.at[L].set(wb(m, s.sm[L], smt[L]))
        qmt = qmt.at[L].set(wb(m, s.qm[L], qmt[L]))
        return cm, smt, qmt

    z2 = jnp.zeros((LM + 2, n))
    cm, smt, qmt = lax.fori_loop(1, LM + 1, cm_body, (z2, z2, z2))
    cm = wb(ev[None, :] & (Lr >= lmax[None, :]) & (Lr >= 1) & (Lr <= LM), 0., cm)

    def ks_body(l, ksub):
        m = ev & (l >= s.ldmin) & (l < lmax)
        c_a = m & (+cm[l] > airm[l + 1] + s.dmr[l + 1])
        c_b = m & ~c_a & (-cm[l] > airm[l] + s.dmr[l])
        va = wb(c_a, (+cm[l] - s.dmr[l + 1]) / airm[l + 1], 0.0)
        vb = wb(c_b, (-cm[l] - s.dmr[l]) / airm[l], 0.0)
        ksub = wb(c_a, jnp.maximum(ksub, 1 + va.astype(I64)), ksub)
        ksub = wb(c_b, jnp.maximum(ksub, 1 + vb.astype(I64)), ksub)
        return ksub

    ksub = lax.fori_loop(1, LM, ks_body, jnp.ones(n, I64))
    ksub = jnp.minimum(ksub, 2)
    byksub = 1.0 / ksub

    def su_body(L, c):
        sumu, sumv, sumdp = c
        m = ev & (L >= s.ldmin) & (L <= lmax)
        return wb(m[None, :], sumu + s.um[:, L], sumu), wb(m[None, :], sumv + s.vm[:, L], sumv), wb(m, sumdp + airm[L], sumdp)

    sumu, sumv, sumdp = lax.fori_loop(1, LM + 1, su_body, (jnp.zeros((NK, n)), jnp.zeros((NK, n)), jnp.zeros(n)))

    def al_body(L, c):
        alpha, um_, vm_ = c
        m = ev & (L >= s.ldmin) & (L <= lmax)
        cldm = ccm_ = s.ccm[L]
        cldm = wb(m & (L < s.ldraft) & (L >= s.llmin) & (s.etadn > 1e-10), s.ccm[L] - s.ddm[L], cldm)
        vsubl = s.vsubl.at[L].set(wb(m & s.mc1, 100. * cldm * k.RGAS * tl[L] / (pl[L] * k.GRAV * dtsrc), s.vsubl[L]))
        beta = cldm * byam[L + 1]
        beta = wb(cldm < 0., cldm * byam[L], beta)
        betau = wb(beta < 0., 0.0, beta)
        alphau = wb(alpha < 0., 0.0, alpha)
        um_ = um_.at[:, L].set(wb(m[None, :], um_[:, L] + ra * (-alphau * um_[:, L] + betau * um_[:, L + 1] + s.dum[:, L]), um_[:, L]))
        vm_ = vm_.at[:, L].set(wb(m[None, :], vm_[:, L] + ra * (-alphau * vm_[:, L] + betau * vm_[:, L + 1] + s.dvm[:, L]), vm_[:, L]))
        return wb(m, beta, alpha), um_, vm_, vsubl

    def al_body2(L, c):
        alpha, um_, vm_, vsubl = c
        m = ev & (L >= s.ldmin) & (L <= lmax)
        cldm = s.ccm[L]
        cldm = wb(m & (L < s.ldraft) & (L >= s.llmin) & (s.etadn > 1e-10), s.ccm[L] - s.ddm[L], cldm)
        vsubl = vsubl.at[L].set(wb(m & s.mc1, 100. * cldm * k.RGAS * tl[L] / (pl[L] * k.GRAV * dtsrc), vsubl[L]))
        beta = cldm * byam[L + 1]
        beta = wb(cldm < 0., cldm * byam[L], beta)
        betau = wb(beta < 0., 0.0, beta)
        alphau = wb(alpha < 0., 0.0, alpha)
        um_ = um_.at[:, L].set(wb(m[None, :], um_[:, L] + ra * (-alphau * um_[:, L] + betau * um_[:, L + 1] + s.dum[:, L]), um_[:, L]))
        vm_ = vm_.at[:, L].set(wb(m[None, :], vm_[:, L] + ra * (-alphau * vm_[:, L] + betau * vm_[:, L + 1] + s.dvm[:, L]), vm_[:, L]))
        return wb(m, beta, alpha), um_, vm_, vsubl

    _, s.um, s.vm, s.vsubl = lax.fori_loop(1, LM + 1, al_body2, (jnp.zeros(n), s.um, s.vm, s.vsubl))

    def s1_body(L, c):
        su, sv = c
        m = ev & (L >= s.ldmin) & (L <= lmax)
        return wb(m[None, :], su + s.um[:, L], su), wb(m[None, :], sv + s.vm[:, L], sv)

    sumu1, sumv1 = lax.fori_loop(1, LM + 1, s1_body, (jnp.zeros((NK, n)), jnp.zeros((NK, n))))
    inr1 = ev[None, :] & (Lr >= s.ldmin[None, :]) & (Lr <= lmax[None, :]) & (Lr >= 1) & (Lr <= LM)
    s.um = wb(inr1[None], s.um - (sumu1 - sumu)[:, None, :] * airm[None] / sumdp[None, None, :], s.um)
    s.vm = wb(inr1[None], s.vm - (sumv1 - sumv)[:, None, :] * airm[None] / sumdp[None, None, :], s.vm)
    cmneg = wb(ev[None, :] & (Lr >= s.ldmin[None, :]) & (Lr < lmax[None, :]) & (Lr >= 1) & (Lr <= LM), -cm * byksub[None, :], 0.0)
    for it in (1, 2):
        selm = ev & (ksub >= it)
        mm_ = selm[None, :] & (Lr >= s.ldmin[None, :]) & (Lr <= lmax[None, :]) & (Lr >= 1) & (Lr <= LM)
        ml = wb(mm_, airm + s.dmr * byksub[None, :], 0.0)
        s.sm = wb(mm_, s.sm + s.dsmr * byksub[None, :], s.sm)
        s.smom = wb(mm_[None], s.smom + s.dsmomr * byksub[None, None, :], s.smom)
        s.sm, s.smom, s.ierr, s.lerr = adv_cb_dev(selm, s.ldmin, lmax, s.sm, s.smom, ml, cmneg, s.ierr, s.lerr, False)
        s.sm = wb(mm_, s.sm + s.dsm * byksub[None, :], s.sm)
        s.smom = wb(mm_[None], s.smom + s.dsmom * byksub[None, None, :], s.smom)
        ml2 = wb(mm_, airm + s.dmr * byksub[None, :], 0.0)
        s.qm = wb(mm_, s.qm + s.dqmr * byksub[None, :], s.qm)
        s.qmom = wb(mm_[None], s.qmom + s.dqmomr * byksub[None, None, :], s.qmom)
        s.qm, s.qmom, s.ierr, s.lerr = adv_cb_dev(selm, s.ldmin, lmax, s.qm, s.qmom, ml2, cmneg, s.ierr, s.lerr, True)
        s.qm = wb(mm_, s.qm + s.dqm * byksub[None, :], s.qm)
        s.qmom = wb(mm_[None], s.qmom + s.dqmom * byksub[None, None, :], s.qmom)
    # ================================================================ diagnostics that feed exit fields
    fcdh = wb(Lr == lmax[None, :], s.cdhsum - s.cdhsum1 + 0.0, 0.0)
    fcdh1 = wb(Lr == s.llmin[None, :], s.cdhsum1 - s.evpsum, 0.0)
    s.mcflx = wb(inr1, s.mcflx + s.ccm * s.fmc1, s.mcflx)
    s.dgdsm = wb(inr1, s.dgdsm + (plk * (s.sm - smt) - fcdh - fcdh1) * s.fmc1, s.dgdsm)
    s.ddmflx = wb(inr1, s.ddmflx + s.ddm * s.fmc1, s.ddmflx)
    s.sm1 = wb(ev[None, :] & (Lr >= 1) & (Lr <= LM), s.sm, s.sm1)
    s.qm1 = wb(ev[None, :] & (Lr >= 1) & (Lr <= LM), s.qm, s.qm1)
    # ================================================================ precipitation
    s.cond = s.cond.at[lme, cols].set(wb(ev, s.cond[lme, cols] + s.condv[lme, cols], s.cond[lme, cols]))
    deep = (ple[lmin] - ple[lmax + 1, cols]) >= 450.

    def pr_body(j, c):
        cond, condp, heat1, svlatl, svwmxl, condpt = c
        L = LM - j
        m = ev & deep & (L <= lmax) & (L >= lmin)
        condp = condp.at[L].set(wb(m & (cond[L] < condp[L]), cond[L], condp[L]))
        fclw = wb(cond[L] > 0, (cond[L] - condp[L]) / cond[L], 0.0)
        ph = m & (svlatl[L] > 0) & (svlatl[L] != s.vlat[L])
        heat1 = heat1.at[L].set(wb(ph, heat1[L] + (svlatl[L] - s.vlat[L]) * svwmxl[L] * airm[L] * k.BYSHA / s.fmc1, heat1[L]))
        svlatl = svlatl.at[L].set(wb(m, s.vlat[L], svlatl[L]))
        svwmxl = svwmxl.at[L].set(wb(m, svwmxl[L] + fclw * cond[L] * byam[L] * s.fmc1, svwmxl[L]))
        cond = cond.at[L].set(wb(m, condp[L], cond[L]))
        condpt = condpt.at[L].set(wb(m, condpt[L] + condp[L], condpt[L]))
        return cond, condp, heat1, svlatl, svwmxl, condpt

    s.cond, s.condp, s.heat1, s.svlatl, s.svwmxl, s.condpt = lax.fori_loop(
        0, LM, pr_body, (s.cond, s.condp, s.heat1, s.svlatl, s.svwmxl, s.condpt))
    prcp = s.cond[lmax, cols]
    s.prheat = wb(ev, s.cdheat[lmax, cols], s.prheat)
    told = smold[lmax, cols] * plk[lmax, cols] * byam[lmax, cols]
    lh_new = wb(told <= k.TF, k.LHS, k.LHE)
    s.lhp = s.lhp.at[lme, cols].set(wb(ev, lh_new, s.lhp[lme, cols]))
    vl_top = s.vlat[lmax, cols]
    fix = ev & (((told > k.TF) & (vl_top == k.LHS)) | ((told <= k.TF) & (vl_top == k.LHE)))
    lhpl = s.lhp[lmax, cols]
    pks = plk[lmax, cols] * s.sm[lmax, cols]
    fs = wb((jnp.abs(pks) > k.TEENY) & ((lhpl - vl_top) * prcp * k.BYSHA < 0), -((lhpl - vl_top) * prcp * k.BYSHA / pks), 0.0)
    s.sm = s.sm.at[lmax, cols].set(wb(fix, s.sm[lmax, cols] + (lhpl - vl_top) * prcp * k.BYSHA / plk[lmax, cols], s.sm[lmax, cols]))
    mrow = jnp.arange(NMOM)[:, None]
    s.smom = s.smom.at[mrow, lmax[None, :], cols[None, :]].set(
        wb(fix[None, :], s.smom[mrow, lmax[None, :], cols[None, :]] * (1. - fs)[None, :], s.smom[mrow, lmax[None, :], cols[None, :]]))
    plemin = ple[lmin]
    plemax = ple[lmax + 1, cols]

    # ---- precipitation-evaporation pre-pass (D189).  The get_dq("e") iterations of layer L use only layer-L quantities (smold, qmold, plk, airm, pl) and the
    # phase lhx = lhp[L]; neither depends on the precipitation `prcp` that the layer loop carries (prcp only enters the final clamp min(dqsum, prcp*airm/mcloud)).
    # So the lhp[L] chain (no libimf) is run first, then the three sequential exp of get_dq for ALL layers are three fused callbacks, instead of 3 per layer.
    def lh_body(j, lh):
        L = LM - 1 - j
        mL = ev & (L <= lmax - 1)
        told_ = smold[L] * plk[L] * byam[L]
        told1_ = smold[L + 1] * plk[L + 1] * byam[L + 1]
        lh_ = lh[L + 1]
        lh_ = wb((lh_ == k.LHS) & (told_ > k.TF) & (told1_ <= k.TF), k.LHE, lh_)
        lh_ = wb((lh_ == k.LHE) & (told_ <= k.TF) & (told1_ > k.TF), k.LHS, lh_)
        return lh.at[L].set(wb(mL, lh_, lh[L]))

    lhp_all = lax.fori_loop(0, LM - 1, lh_body, s.lhp)
    Lx = jnp.arange(LM + 1)[:, None]
    fev_all = .5 * s.ccm[:LM + 1] * byam[1:LM + 2]
    fev_all = wb(Lx < lmin, .5 * s.ccm[lmin] * byam[lmin + 1], fev_all)
    fev_all = wb(fev_all > .5, 0.5, fev_all)
    mcl_all = 2.0 * fev_all * airm[:LM + 1]
    mcl_all = wb(mcl_all > airm[:LM + 1], airm[:LM + 1], mcl_all)
    sel_all = ev & (Lx <= lmax - 1) & (Lx >= 1) & (mcl_all > 0)
    dqsum_pre = dq_sum_dev(k, smold[:LM + 1], qmold[:LM + 1], plk[:LM + 1], airm[:LM + 1], lhp_all[:LM + 1], pl[:LM + 1], sel_all, tag="pp_dq")

    def pp_body(j, c):
        (prcp, prheat, cldmcl, cldslwij, clddepij, precnvl, lhp, heat1, qm, sm, smom, errf) = c
        L = LM - 1 - j
        m = ev & (L <= lmax - 1)
        plel2 = ple[L + 2]
        rho = pl[L] / (k.RGAS * tl[L])
        fcloud = k.CCMUL * s.ccm[L] / (rho * k.GRAV * s.wcu[L] * dtsrc + k.TEENY)
        fcloud = wb(plemin - plel2 >= 450.0, 5.0 * fcloud, fcloud)
        below = L < lmin
        fcloud = wb(below, k.CCMUL * s.ccm[lmin] / (rho * k.GRAV * s.wcu[lmin] * dtsrc + k.TEENY), fcloud)
        shallow = (plemin - plemax) < 450.0
        fcloud = wb(shallow & (L == lmax - 1), k.CCMUL2 * s.ccm[L] / (rho * k.GRAV * s.wcu[L] * dtsrc + k.TEENY), fcloud)
        fcloud = wb(shallow & below, 0.0, fcloud)
        fcloud = wb(fcloud > 1.0, 1.0, fcloud)
        errf = errf | (m & (fcloud < 0.)).any()
        fevap = .5 * s.ccm[L] * byam[L + 1]
        fevap = wb(L < lmin, .5 * s.ccm[lmin] * byam[lmin + 1], fevap)
        fevap = wb(fevap > .5, 0.5, fevap)
        cl_old = cldmcl[L + 1]
        cl_new = wb(m, pymin(cl_old + fcloud * s.fmc1, s.fmc1), cl_old)
        cldmcl = cldmcl.at[L + 1].set(cl_new)
        cldref = cl_new
        cldslwij = wb(m & (plemax > 700) & (cldref > cldslwij), cldref, cldslwij)
        clddepij = wb(m & (plemin - plemax >= 450) & (cldref > clddepij), cldref, clddepij)
        told = smold[L] * plk[L] * byam[L]
        told1 = smold[L + 1] * plk[L + 1] * byam[L + 1]
        precnvl = precnvl.at[L + 1].set(wb(m, precnvl[L + 1] + prcp * k.BYGRAV, precnvl[L + 1]))
        # mc_precip_phase
        mcloud = 2.0 * fevap * airm[L]
        mcloud = wb(mcloud > airm[L], airm[L], mcloud)
        lhp_ = lhp[L + 1]
        h1 = heat1[L]
        m1 = (lhp_ == k.LHS) & (told > k.TF) & (told1 <= k.TF)
        h1 = wb(m1, h1 + k.LHM * prcp * k.BYSHA, h1)
        lhp_ = wb(m1, k.LHE, lhp_)
        m2_ = (lhp_ == k.LHE) & (told <= k.TF) & (told1 > k.TF)
        h1 = wb(m2_, h1 - k.LHM * prcp * k.BYSHA, h1)
        lhp_ = wb(m2_, k.LHS, lhp_)
        m3 = (lhp_ != s.vlat[L]) & (s.cond[L] > 0)
        h1 = wb(m3, h1 + (s.vlat[L] - lhp_) * s.cond[L] * k.BYSHA, h1)
        lhp = lhp.at[L].set(wb(m, lhp_, lhp[L]))
        heat1 = heat1.at[L].set(wb(m, h1, heat1[L]))
        lhx = lhp[L]
        slh = lhx * k.BYSHA
        pp = m & (prcp > 0.)
        me = pp & (mcloud > 0)
        ref_pp = prcp * airm[L] / mcloud
        dqs = jnp.where(ref_pp > 0, jnp.maximum(0.0, jnp.minimum(dqsum_pre[L], ref_pp)), 0.0)      # the clamp of cmj.dq_j
        dqsum = wb(me, dqs, 0.0)
        dqsum = wb(pp, dqsum * mcloud * byam[L], dqsum)
        prcp = wb(pp, prcp - dqsum, prcp)
        qm = qm.at[L].set(wb(pp, qm[L] + dqsum, qm[L]))
        fssum = wb((jnp.abs(plk[L] * sm[L]) > k.TEENY) & (slh * dqsum + heat1[L] > 0), (slh * dqsum + heat1[L]) / (plk[L] * sm[L]), 0.0)
        sm = sm.at[L].set(wb(m, sm[L] - (slh * dqsum + heat1[L]) / plk[L], sm[L]))
        smom = smom.at[:, L].set(wb(m[None, :], smom[:, L] * (1. - fssum)[None, :], smom[:, L]))
        prheat = wb(m, s.cdheat[L] + slh * prcp, prheat)
        prcp = wb(m, prcp + s.cond[L], prcp)
        return prcp, prheat, cldmcl, cldslwij, clddepij, precnvl, lhp, heat1, qm, sm, smom, errf

    (prcp, s.prheat, s.cldmcl, s.cldslwij, s.clddepij, s.precnvl, s.lhp, s.heat1, s.qm, s.sm, s.smom, errf) = lax.fori_loop(
        0, LM - 1, pp_body, (prcp, s.prheat, s.cldmcl, s.cldslwij, s.clddepij, s.precnvl, s.lhp, s.heat1, s.qm, s.sm, s.smom, jnp.zeros((), bool)))
    s.err_neg_cloud = s.err_neg_cloud | errf
    pos = ev & (prcp > 0.)
    shal = (ple[lmin] - ple[lmax + 1, cols]) < 450.
    s.cldmcl = s.cldmcl.at[1].set(wb(pos & shal, pymin(s.cldmcl[1], s.fmc1), s.cldmcl[1]))
    rho = pl[1] / (k.RGAS * tl[1])
    s.cldmcl = s.cldmcl.at[1].set(wb(pos & ~shal, pymin(s.cldmcl[1] + s.fmc1 * s.ccm[lmin] / (rho * k.GRAV * s.wcu[lmin] * dtsrc + k.TEENY), s.fmc1),
                                     s.cldmcl[1]))
    s.prcpmc = wb(ev, s.prcpmc + prcp * s.fmc1, s.prcpmc)
    s.lmcmin = wb(ev & (s.lmcmin > s.ldmin), s.ldmin, s.lmcmin)
    s.mc1 = wb(ev, False, s.mc1)
    Tf = vars(s)
    return ({n: Tf[n] for n in S_NAMES}, {n: Tf[n] for n in SC_NAMES}, {n: Tf[n] for n in V.keys()}, Tf["err_neg_cloud"])


# ---------------------------------------------------------------------------------------------- one level of the plume ascent


def _ascent_step_dev(k, T, L, a, lmin, contce, ic, K, I, KC, cols, n):
    s = SimpleNamespace(**T)
    wb = jnp.where
    pl, ple, plk, airm, byam = I["pl"], I["ple"], I["plk"], I["airm"], I["byam"]
    tl, tvl, qcll, qcil, gzl, u_0, v_0 = I["tl"], I["tvl"], I["qcll"], I["qcil"], I["gzl"], I["u_0"], I["v_0"]
    smom, qmom = s.smom, s.qmom
    pland, dwcu = KC["pland"], KC["dwcu"]
    sm1, qm1 = s.sm1, s.qm1
    mplume = s.mplume
    a = a & ~(mplume <= k.f001 * airm[L])
    sdn = s.smp / mplume
    sup = sm1[L] * byam[L]
    qdn = s.qmp / mplume
    qup = qm1[L] * byam[L]
    wmdn = 0.0
    wmup = qcll[L] + qcil[L]
    svdn = sdn * (1. + k.DELTX * qdn - wmdn)
    svup = sup * (1. + k.DELTX * qup - wmup)
    a = a & ~(plk[L - 1] * (svup - svdn) + k.SLHE * (qup - qdn) >= 0.)
    s.tpsav = s.tpsav.at[L].set(wb(a & (s.tpsav[L] == 0), s.smp * plk[L] / mplume, s.tpsav[L]))
    tp = s.tpsav[L]
    s.lfrz = wb(a & (s.tpsav[L - 1] >= k.TF) & (s.tpsav[L] < k.TF), L - 1, s.lfrz)
    lhx0 = wb(tp < k.TI, k.LHS, k.LHE)
    # ---- callback CB0 (all requests below depend only on values known here; lane mask = a BEFORE the saturation test, see the module doc)
    ms_s = a & (tp < k.TF) & (lhx0 == k.LHE)
    lhx_s = wb(ms_s, k.LHS, lhx0)
    lhx_s = wb(a & (s.vlat[L] == k.LHS), k.LHS, lhx_s)
    cap_s = a & (mplume > k.f95 * airm[L])
    delta_s = (mplume - k.f95 * airm[L]) / mplume
    mpl_s = wb(cap_s, k.f95 * airm[L], mplume)
    smp_s = wb(cap_s, s.smp * (1. - delta_s), s.smp)
    qmp_s = wb(cap_s, s.qmp * (1. - delta_s), s.qmp)
    cb0 = FX.fx([("e", qarg(k, tp, lhx0), a), ("e", qarg(k, sup * plk[L], lhx_s), a),
                 ("e", qarg(k, smp_s * plk[L] / mpl_s, lhx_s), a & (qmp_s > 0)),
                 ("p", pl[L] / 1000.0, k.F04, a), ("p", 1000.0 / pl[L], k.P4, a)], "asc_cb0")
    qsatmp = mplume * (k.QA_ * cb0[0] / pl[L])
    a = a & ~(s.qmp < qsatmp)
    ms_ = a & (tp < k.TF) & (lhx0 == k.LHE)
    lhx = wb(ms_, k.LHS, lhx0)
    lhx = wb(a & (s.vlat[L] == k.LHS), k.LHS, lhx)
    s.vlat = s.vlat.at[L].set(wb(a, lhx, s.vlat[L]))
    slh = lhx * k.BYSHA
    s.u00l = s.u00l.at[L].set(wb(a & (tl[L] >= k.TF) & (s.u00l[L] != k.u00a), k.u00_mc, s.u00l[L]))
    s.mccont = wb(a, s.mccont + 1, s.mccont)
    s.mc1 = s.mc1 | (a & (s.mccont == 1))
    cap = a & (mplume > k.f95 * airm[L])
    delta = (mplume - k.f95 * airm[L]) / mplume
    s.dm = s.dm.at[L - 1].set(wb(cap, s.dm[L - 1] + delta * mplume, s.dm[L - 1]))
    mplume = wb(cap, k.f95 * airm[L], mplume)
    s.dsm = s.dsm.at[L - 1].set(wb(cap, s.dsm[L - 1] + delta * s.smp, s.dsm[L - 1]))
    smp = wb(cap, s.smp * (1. - delta), s.smp)
    s.dsmom = s.dsmom.at[XYMa, L - 1].set(wb(cap, s.dsmom[XYMa, L - 1] + delta * s.smomp[XYMa], s.dsmom[XYMa, L - 1]))
    s.smomp = s.smomp.at[XYMa].set(wb(cap, s.smomp[XYMa] * (1. - delta), s.smomp[XYMa]))
    s.dqm = s.dqm.at[L - 1].set(wb(cap, s.dqm[L - 1] + delta * s.qmp, s.dqm[L - 1]))
    qmp = wb(cap, s.qmp * (1. - delta), s.qmp)
    s.dqmom = s.dqmom.at[XYMa, L - 1].set(wb(cap, s.dqmom[XYMa, L - 1] + delta * s.qmomp[XYMa], s.dqmom[XYMa, L - 1]))
    s.qmomp = s.qmomp.at[XYMa].set(wb(cap, s.qmomp[XYMa] * (1. - delta), s.qmomp[XYMa]))
    s.dum = s.dum.at[:, L - 1].set(wb(cap, s.dum[:, L - 1] + s.ump * delta, s.dum[:, L - 1]))
    s.dvm = s.dvm.at[:, L - 1].set(wb(cap, s.dvm[:, L - 1] + s.vmp * delta, s.dvm[:, L - 1]))
    ump = wb(cap, s.ump - s.ump * delta, s.ump)
    vmp = wb(cap, s.vmp - s.vmp * delta, s.vmp)
    wk = mplume * (sup - sdn) * (plk[L - 1] - plk[L]) / plk[L - 1]
    s.dsm = s.dsm.at[L - 1].set(wb(a, s.dsm[L - 1] - wk, s.dsm[L - 1]))
    s.ccm = s.ccm.at[L - 1].set(wb(a, mplume, s.ccm[L - 1]))
    dqsum, fqcond = dq_dev(k, "c", smp, qmp, plk[L], mplume, lhx, pl[L], None, a & (qmp > 0), e_first=cb0[2], tag="asc_dq")
    m3 = a & (dqsum > 0.) & (qmp > k.TEENY)
    s.qmomp = s.qmomp.at[XYMa].set(wb(m3, s.qmomp[XYMa] * (1. - fqcond), s.qmomp[XYMa]))
    smp = wb(m3, smp + slh * dqsum / plk[L], smp)
    qmp = wb(m3, qmp - dqsum, qmp)
    cond_L = wb(a, dqsum, s.cond[L])
    s.condmmr = s.condmmr.at[L].set(wb(a, s.condmmr[L] + cond_L * byam[L] * s.fmc1, s.condmmr[L]))
    cdheat_L = wb(a, slh * cond_L, s.cdheat[L])
    cdhsum = wb(a, s.cdhsum + cdheat_L, s.cdhsum)
    cond_L = wb(a, cond_L + s.condv[L - 1], cond_L)
    mv = a & (s.vlat[L - 1] != s.vlat[L])
    smp = wb(mv, smp - (s.vlat[L - 1] - s.vlat[L]) * s.condv[L - 1] * k.BYSHA / plk[L], smp)
    cdheat_L = wb(mv, cdheat_L - (s.vlat[L - 1] - s.vlat[L]) * s.condv[L - 1] * k.BYSHA, cdheat_L)
    cdhsum = wb(mv, cdhsum - (s.vlat[L - 1] - s.vlat[L]) * s.condv[L - 1] * k.BYSHA, cdhsum)
    condmu = 100. * cond_L * pl[L] / (s.ccm[L - 1] * tl[L] * k.RGAS)
    s.taumc1 = s.taumc1.at[L].set(wb(a, s.taumc1[L] + cond_L * s.fmc1, s.taumc1[L]))
    tvp = (smp / mplume) * plk[L] * (1. + k.DELTX * qmp / mplume)
    buoy_L = wb(a, (tvp - tvl[L]) / tvl[L] - cond_L / mplume, s.buoy[L])
    s.buoy = s.buoy.at[L].set(buoy_L)
    ent_L = wb(a, .16667 * contce * k.GRAV * buoy_L / (s.wcu[L - 1] * s.wcu[L - 1] + k.TEENY), s.ent[L])
    ng = a & (ent_L < 0.)
    det_L = wb(ng, -ent_L, s.det[L])
    ent_L = wb(ng, 0., ent_L)
    pe = a & (ent_L > 0.)
    fplume = s.fplume
    fentr = 1000. * ent_L * gzl[L] * fplume
    cap2 = pe & (fentr + fplume > 1.)
    fentr = wb(cap2, 1. - fplume, fentr)
    ent_L = wb(cap2, 0.001 * fentr / (gzl[L] * fplume), ent_L)
    ee = pe & (fentr >= k.TEENY)
    mpold, fpold = mplume, fplume
    etal1 = fentr / (fplume + k.TEENY)
    eplume = mplume * etal1
    cap3 = ee & (eplume > airm[L] * 0.975 - mplume)
    eplume = wb(cap3, airm[L] * 0.975 - mplume, eplume)
    mplume = wb(ee, mplume + eplume, mplume)
    etal1 = eplume / mpold
    fentr = wb(ee, etal1 * fpold, fentr)
    ent_L = wb(ee, 0.001 * fentr / (gzl[L] * fpold), ent_L)
    fplume = wb(ee, mplume * byam[L], fplume)          # mc_entr_mass_lim_plume = 1 (asserted by the driver)
    fentra = eplume * byam[L]
    s.dsmr = s.dsmr.at[L].set(wb(ee, s.dsmr[L] - eplume * sup, s.dsmr[L]))
    s.dsmomr = s.dsmomr.at[:, L].set(wb(ee, s.dsmomr[:, L] - smom[:, L] * fentra, s.dsmomr[:, L]))
    s.dqmr = s.dqmr.at[L].set(wb(ee, s.dqmr[L] - eplume * qup, s.dqmr[L]))
    s.dqmomr = s.dqmomr.at[:, L].set(wb(ee, s.dqmomr[:, L] - qmom[:, L] * fentra, s.dqmomr[:, L]))
    s.dmr = s.dmr.at[L].set(wb(ee, s.dmr[L] - eplume, s.dmr[L]))
    smp = wb(ee, smp + eplume * sup, smp)
    s.smomp = s.smomp.at[XYMa].set(wb(ee, s.smomp[XYMa] + smom[XYMa, L] * fentra, s.smomp[XYMa]))
    qmp = wb(ee, qmp + eplume * qup, qmp)
    s.qmomp = s.qmomp.at[XYMa].set(wb(ee, s.qmomp[XYMa] + qmom[XYMa, L] * fentra, s.qmomp[XYMa]))
    umtemp = k.pgrad * (mplume * mplume) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
    vmtemp = k.pgrad * (mplume * mplume) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
    ump = wb(ee, ump + u_0[:, L] * eplume + umtemp, ump)
    s.dum = s.dum.at[:, L].set(wb(ee, s.dum[:, L] - u_0[:, L] * eplume - umtemp, s.dum[:, L]))
    vmp = wb(ee, vmp + v_0[:, L] * eplume + vmtemp, vmp)
    s.dvm = s.dvm.at[:, L].set(wb(ee, s.dvm[:, L] - v_0[:, L] * eplume - vmtemp, s.dvm[:, L]))
    dd = a & (det_L > 0.)
    delta = 1000. * det_L * gzl[L]
    cap4 = dd & (delta > .95)
    delta = wb(cap4, 0.95, delta)
    det_L = wb(cap4, .001 * delta / gzl[L], det_L)
    s.dm = s.dm.at[L].set(wb(dd, s.dm[L] + delta * mplume, s.dm[L]))
    mplume = wb(dd, mplume * (1. - delta), mplume)
    s.dsm = s.dsm.at[L].set(wb(dd, s.dsm[L] + delta * smp, s.dsm[L]))
    smp = wb(dd, smp * (1. - delta), smp)
    s.dsmom = s.dsmom.at[XYMa, L].set(wb(dd, s.dsmom[XYMa, L] + delta * s.smomp[XYMa], s.dsmom[XYMa, L]))
    s.smomp = s.smomp.at[XYMa].set(wb(dd, s.smomp[XYMa] * (1. - delta), s.smomp[XYMa]))
    s.dqm = s.dqm.at[L].set(wb(dd, s.dqm[L] + delta * qmp, s.dqm[L]))
    qmp = wb(dd, qmp * (1. - delta), qmp)
    s.dqmom = s.dqmom.at[XYMa, L].set(wb(dd, s.dqmom[XYMa, L] + delta * s.qmomp[XYMa], s.dqmom[XYMa, L]))
    s.qmomp = s.qmomp.at[XYMa].set(wb(dd, s.qmomp[XYMa] * (1. - delta), s.qmomp[XYMa]))
    umtemp = k.pgrad * (mplume * mplume) * (u_0[:, L + 1] - u_0[:, L]) / (pl[L] - pl[L + 1])
    vmtemp = k.pgrad * (mplume * mplume) * (v_0[:, L + 1] - v_0[:, L]) / (pl[L] - pl[L + 1])
    s.dum = s.dum.at[:, L].set(wb(dd, s.dum[:, L] + ump * delta - umtemp, s.dum[:, L]))
    s.dvm = s.dvm.at[:, L].set(wb(dd, s.dvm[:, L] + vmp * delta - vmtemp, s.dvm[:, L]))
    ump = wb(dd, ump - ump * delta + umtemp, ump)
    vmp = wb(dd, vmp - vmp * delta + vmtemp, vmp)
    # ---- downdraft trigger (only for L - lmin > 1)
    gate = (L - lmin) > 1
    smix = .5 * (sup + smp / mplume)
    qmix = .5 * (qup + qmp / mplume)
    wmix = .5 * (wmup + cond_L / mplume)
    svmix = smix * (1. + k.DELTX * qmix - wmix)
    svup = sup * (1. + k.DELTX * qup - wmup)
    dmmix = (svup - svmix) * plk[L] + k.SLHE * (k.QA_ * cb0[1] / pl[L] - qmix)
    cdhdrt = wb(gate & a & (dmmix < 1e-10), s.cdhdrt + cdheat_L, s.cdhdrt)
    tr = gate & a & (dmmix >= 1e-10)
    s.ldraft = wb(tr, L, s.ldraft)
    etad = F(1.0) / F(3.0)
    etadn = wb(tr, etad, s.etadn)
    fleft = 1. - .5 * etad
    ddraft = wb(tr, etad * mplume, s.ddraft)
    s.ddr = s.ddr.at[L].set(wb(tr, ddraft, s.ddr[L]))
    cdhsum1 = wb(tr, s.cdhsum1 + cdhdrt * .5 * etad, s.cdhsum1)
    cdhdrt = wb(tr, cdhdrt - cdhdrt * .5 * etad + cdheat_L, cdhdrt)
    fddp = .5 * ddraft
    fddp = fddp / mplume
    fddl = .5 * ddraft * byam[L]
    mplume = wb(tr, fleft * mplume, mplume)
    s.smdnl = s.smdnl.at[L].set(wb(tr, ddraft * smix, s.smdnl[L]))
    s.smomdnl = s.smomdnl.at[XYMa, L].set(wb(tr, smom[XYMa, L] * fddl + s.smomp[XYMa] * fddp, s.smomdnl[XYMa, L]))
    smp = wb(tr, fleft * smp, smp)
    s.smomp = s.smomp.at[XYMa].set(wb(tr, s.smomp[XYMa] * fleft, s.smomp[XYMa]))
    s.qmdnl = s.qmdnl.at[L].set(wb(tr, ddraft * qmix, s.qmdnl[L]))
    s.qmomdnl = s.qmomdnl.at[XYMa, L].set(wb(tr, qmom[XYMa, L] * fddl + s.qmomp[XYMa] * fddp, s.qmomdnl[XYMa, L]))
    qmp = wb(tr, fleft * qmp, qmp)
    s.qmomp = s.qmomp.at[XYMa].set(wb(tr, s.qmomp[XYMa] * fleft, s.qmomp[XYMa]))
    s.dmr = s.dmr.at[L].set(wb(tr, s.dmr[L] - .5 * ddraft, s.dmr[L]))
    s.dsmr = s.dsmr.at[L].set(wb(tr, s.dsmr[L] - .5 * ddraft * sup, s.dsmr[L]))
    s.dsmomr = s.dsmomr.at[:, L].set(wb(tr, s.dsmomr[:, L] - smom[:, L] * fddl, s.dsmomr[:, L]))
    s.dqmr = s.dqmr.at[L].set(wb(tr, s.dqmr[L] - .5 * ddraft * qup, s.dqmr[L]))
    s.dqmomr = s.dqmomr.at[:, L].set(wb(tr, s.dqmomr[:, L] - qmom[:, L] * fddl, s.dqmomr[:, L]))
    s.umdnl = s.umdnl.at[:, L].set(wb(tr, .5 * (etad * ump + ddraft * u_0[:, L]), s.umdnl[:, L]))
    ump = wb(tr, ump * fleft, ump)
    s.dum = s.dum.at[:, L].set(wb(tr, s.dum[:, L] - .5 * ddraft * u_0[:, L], s.dum[:, L]))
    s.vmdnl = s.vmdnl.at[:, L].set(wb(tr, .5 * (etad * vmp + ddraft * v_0[:, L]), s.vmdnl[:, L]))
    vmp = wb(tr, vmp * fleft, vmp)
    s.dvm = s.dvm.at[:, L].set(wb(tr, s.dvm[:, L] - .5 * ddraft * v_0[:, L], s.dvm[:, L]))
    # ---- cloud-top velocity
    w2tem = .16667 * k.GRAV * buoy_L - s.wcu[L - 1] * s.wcu[L - 1] * (.66667 * det_L + ent_L)
    hdep = airm[L] * tl[L] * k.RGAS / (k.GRAV * pl[L])
    wcu2_L = wb(a, s.wcu2[L - 1] + 2. * hdep * w2tem, s.wcu2[L])
    s.wcu2 = s.wcu2.at[L].set(wcu2_L)
    wl = wb(wcu2_L > 0., jnp.sqrt(wb(wcu2_L > 0., wcu2_L, 1.0)), 0.)
    wl = wb(wl >= 0., pymin(50., wl), wl)
    wl = wb(wl < 0., pymax(-50., wl), wl)
    wcu_L = wb(a, wl, s.wcu[L])
    s.wcu = s.wcu.at[L].set(wcu_L)
    s.smpmax = wb(a, smp, s.smpmax)
    s.smompmax = wb(a, s.smomp, s.smompmax)
    s.qmpmax = wb(a, qmp, s.qmpmax)
    s.qmompmax = wb(a, s.qmomp, s.qmompmax)
    s.mpmax = wb(a, mplume, s.mpmax)
    s.lmax = wb(a, s.lmax + 1, s.lmax)
    a = a & ~(wcu2_L < 0.)
    # ---- convective microphysics
    wcufrz = wb(s.lfrz > 0, s.wcu[s.lfrz, cols], 0.0)
    cp, cp1, cip, cgp = conv_micro_dev(k, pl[L], wcu_L, dwcu, s.lfrz.astype(jnp.float64), wcufrz, tp, pland, condmu, I["tl"][lmin],
                                       I["tl"][lmin + 1], s.condip[L], s.condgp[L], a, cb0[3], cb0[4])
    condp_L = wb(a, cp, s.condp[L])
    condp1_L = wb(a, cp1, s.condp1[L])
    s.condip = s.condip.at[L].set(wb(a, cip, s.condip[L]))
    s.condgp = s.condgp.at[L].set(wb(a, cgp, s.condgp[L]))
    condp_L = wb(a, .01 * condp_L * s.ccm[L - 1] * tl[L] * k.RGAS / pl[L], condp_L)
    condp1_L = wb(a, .01 * condp1_L * s.ccm[L - 1] * tl[L] * k.RGAS / pl[L], condp1_L)
    condp1_L = wb(a & (condp1_L > cond_L), cond_L, condp1_L)
    condp_L = wb(a & (condp_L > condp1_L), condp1_L, condp_L)
    condv_L = wb(a, cond_L - condp1_L, s.condv[L])
    cond_L = wb(a, cond_L - condv_L, cond_L)
    taumc1_L = wb(a, s.taumc1[L] - condv_L * s.fmc1, s.taumc1[L])
    s.condp = s.condp.at[L].set(condp_L)
    s.condp1 = s.condp1.at[L].set(condp1_L)
    s.condv = s.condv.at[L].set(condv_L)
    s.cond = s.cond.at[L].set(cond_L)
    s.taumc1 = s.taumc1.at[L].set(taumc1_L)
    s.cdheat = s.cdheat.at[L].set(cdheat_L)
    s.ent = s.ent.at[L].set(ent_L)
    s.det = s.det.at[L].set(det_L)
    s.mplume, s.smp, s.qmp, s.fplume, s.ump, s.vmp = mplume, smp, qmp, fplume, ump, vmp
    s.cdhsum, s.cdhsum1, s.cdhdrt, s.etadn, s.ddraft = cdhsum, cdhsum1, cdhdrt, etadn, ddraft
    return vars(s), a


# ---------------------------------------------------------------------------------------------- set-up, base test, post stage, driver

def _base_dev(K, lmin, S, I):
    k = SimpleNamespace(**K)
    lm1 = lmin + 1
    pl, plk, airm, byam, sdl = I["pl"], I["plk"], I["airm"], I["byam"], I["sdl"]
    fmp0 = -(10. * 1.0 * sdl[lm1] * k.BYGRAV * k.xmass)
    fmp0 = jnp.where(fmp0 <= 0., 0.0, fmp0)
    smo1, qmo1, smo2, qmo2 = S["sm"][lmin], S["qm"][lmin], S["sm"][lm1], S["qm"][lm1]
    sdn = smo1 * byam[lmin]
    sup = smo2 * byam[lm1]
    sedge = thbar_j(k, sup, sdn)
    qdn = qmo1 * byam[lmin]
    qup = qmo2 * byam[lm1]
    wmdn = I["qcll"][lmin] + I["qcil"][lmin]
    wmup = I["qcll"][lm1] + I["qcil"][lm1]
    svdn = sdn * (1. + k.DELTX * qdn - wmdn)
    svup = sup * (1. + k.DELTX * qup - wmup)
    qedge = .5 * (qup + qdn)
    wmedg = .5 * (wmup + wmdn)
    svedg = sedge * (1. + k.DELTX * qedge - wmedg)
    slh0 = k.LHE * k.BYSHA
    lhxv = jnp.full(smo1.shape, 1.0) * k.LHE
    e_inv = FX.fx([("e", qarg(k, sup * plk[lm1], k.LHE), None), ("e", qarg(k, smo1 * plk[lm1] * byam[lmin], lhxv), None),
                   ("e", qarg(k, smo2 * plk[lm1] * byam[lm1], lhxv), None), ("e", qarg(k, smo1 * plk[lmin] * byam[lmin], lhxv), None)], "base_cb")
    dmse = (svup - svedg) * plk[lm1] + (svedg - svdn) * plk[lmin] + slh0 * (k.QA_ * e_inv[0] / pl[lm1] - qdn)
    sub0 = ~(dmse > -1e-10)
    fplume, fmp2 = mass_flux_dev(k, sub0, e_inv[1:], k.LHE, qmo1, qmo2, smo1, smo2, slh0, wmdn, wmup, wmedg, airm[lmin], airm[lm1], byam[lmin], byam[lm1],
                                 byam[lmin + 2], S["sm"][lmin + 2], S["qm"][lmin + 2], plk[lmin], plk[lm1], pl[lmin], pl[lm1])
    mask = sub0 & ~(fplume <= k.f001)
    return mask, fmp0, fmp2 * k.fmpscale


def _event_body(K, lmin, fmp0, fmp2, idxg, idxs, valid, S, Sc, I, aux):
    nb = idxg.shape[0]
    Sg, Scg, Ig = _take(S, idxg), _take(Sc, idxg), _take(I, idxg)
    KC = dict(dcl=aux["dcl"][idxg], dwcu=aux["dwcu"][idxg], pland=aux["pland"][idxg])
    f0, f2 = fmp0[idxg], fmp2[idxg]
    V = _init_V(nb, lmin)

    def ic_body(ic, c):
        Sg, Scg, V, err = c
        Sg, Scg, V, e = _event_dev(K, lmin, ic, f0, f2, valid, Sg, Scg, Ig, V, KC)
        return Sg, Scg, V, err | e

    Sg, Scg, V, err = lax.fori_loop(1, 3, ic_body, (Sg, Scg, V, jnp.zeros((), bool)))
    return _put(S, Sg, idxs), _put(Sc, Scg, idxs), err


def _buckets_for(N, buckets):
    bk = sorted({b for b in buckets if b < N} | {N})
    return tuple(bk)


@functools.partial(jax.jit, static_argnames=("lmcm", "buckets"))
def _events_device(K, S, Sc, I, aux, lmcm, buckets):
    """The whole cloud-base loop on the device: returns (S, Sc, err, n_events_per_lmin)."""
    N = I["pl"].shape[1]
    bk = _buckets_for(N, buckets)
    bka = jnp.asarray(bk)

    def skip(args):
        S, Sc, *_ = args
        return S, Sc, jnp.zeros((), bool)

    def make(nb):
        def run(args):
            S, Sc, idx, cnt, lmin, fmp0, fmp2 = args
            ar = jnp.arange(nb)
            valid = ar < cnt
            idxg = idx[:nb]
            idxs = jnp.where(valid, idxg, N)
            return _event_body(K, lmin, fmp0, fmp2, idxg, idxs, valid, S, Sc, I, aux)
        return run

    branches = [skip] + [make(nb) for nb in bk]

    def body(lmin, c):
        S, Sc, err, nev = c
        mask, fmp0, fmp2 = _base_dev(K, lmin, S, I)
        cnt = jnp.sum(mask.astype(I64))
        idx = jnp.nonzero(mask, size=N, fill_value=0)[0]
        bi = jnp.where(cnt == 0, 0, 1 + jnp.searchsorted(bka, cnt, side="left"))
        S, Sc, e = lax.switch(bi, branches, (S, Sc, idx, cnt, lmin, fmp0, fmp2))
        return S, Sc, err | e, nev.at[lmin].set(cnt)

    S, Sc, err, nev = lax.fori_loop(1, lmcm, body, (S, Sc, jnp.zeros((), bool), jnp.zeros(lmcm + 1, I64)))
    return S, Sc, err, nev



@jax.jit
def _post_dev(K, S, Sc, I, aux):
    k = SimpleNamespace(**K)
    N = I["pl"].shape[1]
    ca = jnp.arange(N)
    pl, ple, plk, airm, byam = I["pl"], I["ple"], I["plk"], I["airm"], I["byam"]
    sm, qm = S["sm"], S["qm"]
    lmcmin, lmcmax, fmc1 = Sc["lmcmin"], Sc["lmcmax"], Sc["fmc1"]
    conv = lmcmin > 0
    Lr = jnp.arange(LM + 2)[:, None]
    rng = (Lr >= 1) & (Lr <= LM)
    fssl = jnp.where(conv[None, :] & rng & (Lr <= lmcmax[None, :]), 1 - fmc1[None, :], 1.0)
    inb = conv[None, :] & rng & (Lr >= lmcmin[None, :]) & (Lr <= lmcmax[None, :])

    def acc(L, c):
        sumdp, sumaj, airxl = c
        m = conv & (L >= lmcmin) & (L <= lmcmax)
        return (jnp.where(m, sumdp + airm[L] * fmc1, sumdp), jnp.where(m, sumaj + S["dgdsm"][L], sumaj), jnp.where(m, airxl + S["mcflx"][L], airxl))

    sumdp, sumaj, airxl = lax.fori_loop(1, LM + 1, acc, (jnp.zeros(N), jnp.zeros(N), jnp.zeros(N)))
    dgdsm = jnp.where(inb, S["dgdsm"] - sumaj * airm * fmc1 / sumdp, S["dgdsm"])
    sm = jnp.where(inb, sm - sumaj * airm / (sumdp * plk), sm)
    wconst = k.wmu_mult * (k.WMU * (1. - aux["pearth"]) + k.WMUL * aux["pearth"])
    pearth = aux["pearth"]
    dcl = aux["dcl"]
    dpl = ple[lmcmin, ca] - ple[lmcmax + 1, ca]
    mcdncw = k.mndo * (1. - pearth) + k.MNDL * pearth

    Lall = jnp.arange(LM + 2)[:, None]
    m_a = Lall <= lmcmax[None, :]
    tl_a = jnp.where(m_a, (sm * byam) * plk, I["tl"])
    z0_a = m_a & (S["svlatl"] == 0.)
    sv_a = jnp.where(z0_a, jnp.where(((S["tpsav"] > 0.) & (S["tpsav"] < k.TF)) | ((S["tpsav"] == 0.) & (tl_a < k.TF)), k.LHS, k.LHE), S["svlatl"])
    an_a = m_a & (S["svwmxl"] > 0.)
    fcld_a = S["cldmcl"] + k.F1E20
    wtem_a = 1e5 * S["svwmxl"] * pl / (fcld_a * tl_a * k.RGAS)
    wtem_a = jnp.where((sv_a == k.LHE) & (S["svwmxl"] / fcld_a >= wconst[None, :] * 1e-3), 1e2 * wconst[None, :] * pl / (tl_a * k.RGAS), wtem_a)
    wtem_a = jnp.where(wtem_a < 1e-10, 1e-10, wtem_a)
    pw_liq, pw_ice = FX.fx([("p", wtem_a / (2.0 * k.BY3 * k.TWOPI * mcdncw[None, :]), k.BY3, an_a & (sv_a == k.LHE)),
                            ("p", wtem_a / (2.0 * k.BY3 * k.TWOPI * k.MNDI), k.BY3, an_a & (sv_a != k.LHE))], "post_radius")

    def ot(L, c):
        tl, taumcl, svlat1, svlatl, csizel, qlmc, qimc, wmsum, wmctwp, wmclwp = c
        m = L <= lmcmax
        tl_L = jnp.where(m, (sm[L] * byam[L]) * plk[L], tl[L])
        tl = tl.at[L].set(tl_L)
        temwm = (taumcl[L] - S["svwmxl"][L] * airm[L]) * 1e2 * k.BYGRAV
        warm = m & (tl_L >= k.TF)
        wmsum = jnp.where(warm, wmsum + temwm, wmsum)
        wmctwp = jnp.where(m, wmctwp + temwm, wmctwp)
        wmclwp = jnp.where(warm, wmclwp + temwm, wmclwp)
        temwm = taumcl[L] - S["condpt"][L] * fmc1 - S["svwmxl"][L] * airm[L]
        ql_ = jnp.where(m & (svlatl[L] == k.LHE), temwm / airm[L] + S["svwmxl"][L], 0.0)
        qi_ = jnp.where(m & (svlatl[L] == k.LHS), temwm / airm[L] + S["svwmxl"][L], 0.0)
        ql_ = jnp.where(m & (S["lhp"][L] == k.LHE), ql_ + S["condpt"][L] * fmc1 / airm[L], ql_)
        qi_ = jnp.where(m & (S["lhp"][L] == k.LHS), qi_ + S["condpt"][L] * fmc1 / airm[L], qi_)
        qlmc = qlmc.at[L].set(ql_)
        qimc = qimc.at[L].set(qi_)
        cp = m & (S["cldmcl"][L] > 0.)
        t_ = taumcl[L]
        t_ = jnp.where(cp, airm[L] * k.COETAU, t_)
        t_ = jnp.where(cp & (L == lmcmax) & (dpl < 450), airm[L] * .02, t_)
        t_ = jnp.where(cp & (L <= lmcmin) & (dpl >= 450), airm[L] * .02, t_)
        svlat1 = svlat1.at[L].set(jnp.where(m, svlatl[L], 0.0))
        z0 = m & (svlatl[L] == 0.)
        sv_ = jnp.where(z0, jnp.where(((S["tpsav"][L] > 0.) & (S["tpsav"][L] < k.TF)) | ((S["tpsav"][L] == 0.) & (tl_L < k.TF)), k.LHS, k.LHE), svlatl[L])
        svlatl = svlatl.at[L].set(sv_)
        an = m & (S["svwmxl"][L] > 0.)
        fcld = S["cldmcl"][L] + k.F1E20
        tem = 1e5 * S["svwmxl"][L] * airm[L] * k.BYGRAV
        wtem = 1e5 * S["svwmxl"][L] * pl[L] / (fcld * tl_L * k.RGAS)
        wtem = jnp.where((sv_ == k.LHE) & (S["svwmxl"][L] / fcld >= wconst * 1e-3), 1e2 * wconst * pl[L] / (tl_L * k.RGAS), wtem)
        wtem = jnp.where(wtem < 1e-10, 1e-10, wtem)
        r_liq = k.rcldlx * 100.0 * pw_liq[L]
        r_ice = k.rcldix * 100.0 * pw_ice[L]
        r_ice = jnp.minimum(r_ice, k.rimax)
        rcld = jnp.where(sv_ == k.LHE, r_liq, r_ice)
        rclde = rcld / k.bybr
        taumc = 1.5 * tem / (fcld * rclde + k.F1E20)
        taumc = jnp.where(taumc > 100.0, 100.0, taumc)
        t_ = jnp.where(an, taumc, t_)
        csizel = csizel.at[L].set(jnp.where(an, rcld / k.bybr, csizel[L]))
        t_ = jnp.where(m & (t_ < 0.) & (S["cldmcl"][L] <= 0.), 0., t_)
        taumcl = taumcl.at[L].set(t_)
        return tl, taumcl, svlat1, svlatl, csizel, qlmc, qimc, wmsum, wmctwp, wmclwp

    z = jnp.zeros((LM + 2, N))
    tl, taumcl, svlat1, svlatl, csizel, qlmc, qimc, wmsum, wmctwp, wmclwp = lax.fori_loop(
        1, LM + 1, ot, (I["tl"], S["taumcl"], z, S["svlatl"], aux["csizel"], z, z, jnp.zeros(N), jnp.zeros(N), jnp.zeros(N)))
    low = lmcmax <= 1
    pl_dcl = pl[dcl, ca]
    u00l = jnp.where(low[None, :] & (pl < pl_dcl[None, :]) & rng, 0., S["u00l"])
    cnvmmrl = jnp.where(S["cldmcl"] > 0.0, S["condmmr"], 0.0).at[0].set(0.0).at[LM + 1].set(0.0)
    o = {n: Sc[n] for n in ("ierr", "lerr", "lmcmin", "lmcmax", "prcpmc", "cldslwij", "clddepij", "prheat", "fmc1", "mccont")}
    o.update(airxl=airxl, wmsum=wmsum, wmctwp=wmctwp, wmclwp=wmclwp)
    named = dict(tl=tl, sm=sm, qm=qm, fssl=fssl, cldmcl=S["cldmcl"], taumcl=taumcl, svlatl=svlatl, svlat1=svlat1, svwmxl=S["svwmxl"], csizel=csizel,
                 condpt=S["condpt"], vsubl=S["vsubl"], tpsav=S["tpsav"], mcflx=S["mcflx"], dgdsm=dgdsm, ddmflx=S["ddmflx"], tdnl=S["tdnl"],
                 qdnl=S["qdnl"], u00l=u00l, qlmc=qlmc, qimc=qimc, cnvmmrl=cnvmmrl, condmmr=S["condmmr"])
    for n_, a_ in named.items():
        o[n_] = a_[1:LM + 1].T
    o["lhp"] = S["lhp"][1:LM + 2].T
    o["precnvl"] = S["precnvl"][1:LM + 2].T
    for n_ in ("smom", "qmom", "um", "vm"):
        o[n_] = S[n_][:, 1:LM + 1].transpose(2, 0, 1)
    return o



def mstcnv_dev(R, c, return_jax=False, stats=None, buckets=BUCKETS):
    """MSTCNV for N columns (KMAX=4): same inputs / outputs as clouds_mstcnv_jax.mstcnv_jax; the cloud-base loop is on the device (one host read at the end:
    the 'negative cloud cover' flag)."""
    assert c["mc_new_ddrft_thetav"] != 0 and c["mc_entr_mass_lim_plume"] != 0
    lmcm = int(R["lmcm"][0])
    for n_ in ("xmass", "bydtsrc", "dtsrc", "bybr"):
        assert (R[n_] == R[n_][0]).all(), n_
    K = make_K(c)
    for n_ in ("xmass", "bydtsrc", "dtsrc", "bybr"):
        K[n_] = np.float64(R[n_][0])
    K["fmpscale"] = np.float64(min(1.0, F(R["dtsrc"][0]) / (F(1.0) * mc.SECONDS_PER_HOUR)))
    Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
    S, Sc, I, aux = _setup_jit(Kj, R, lmcm)
    S, Sc, err, nev = _events_device(Kj, S, Sc, I, aux, lmcm, tuple(buckets))
    o = _post_dev(Kj, S, Sc, I, aux)
    if return_jax:
        return o, err
    if stats is not None:
        stats["events_per_lmin"] = np.asarray(nev)[1:lmcm]
        stats["events"] = int(np.asarray(nev).sum())
    if bool(err):
        raise RuntimeError("MSTCNV: negative cloud cover")
    return {n_: np.asarray(v) for n_, v in o.items()}


@functools.partial(jax.jit, static_argnums=2)
def _setup_jit(K, R, lmcm):
    return _setup(SimpleNamespace(**K), R, lmcm)
