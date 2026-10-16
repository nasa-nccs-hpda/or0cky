"""D189 tests of libimf_fused and the restructured helpers of clouds_mstcnv_dev (own unit only; no real data, runs in seconds on cores 0-2).
Run in its own process:  taskset -c 0-2 env OMP_NUM_THREADS=1 python -m pytest tests/test_libimf_fused.py
(host callbacks inside jit need >= 2 cores).  The stage-level test on real CONDSE inputs is d189_run_dev.py / d189_condse_check.py (long cold compile)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: E402,F401
import intel_libm_ff  # noqa: E402

pytestmark = pytest.mark.skipif(not intel_libm_ff.available(), reason="Intel libimf not available")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import libimf_fused as FX  # noqa: E402

TUNE = dict(entrainment_cont1=0.3, entrainment_cont2=0.6, radiusl_multiplier=1.0, radiusi_multiplier=1.0, u00a=1.0, u00b=0.9, wmu_multiplier=1.0,
            rwcldox=1.0, rimax=100.0, mc_fddrt=1.0, mc_entr_mass_lim_plume=1, mc_new_ddrft_thetav=1, mc_revp_abv_cldbase=0)   # test inputs, not model values


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, float)).view(np.int64)


def test_shim_equals_ctypes_and_differs_from_numpy():
    FX.set_mode("libimf")
    r = np.random.default_rng(1)
    x = np.concatenate([r.random(20000) * 900 + 1e-3, [0.0, 1.0, 1e-300, 1e300, 5e-324]])
    y = np.concatenate([r.random(20000) * 3 - 1, [0.4, 0.4, -0.3, 2.0, 0.5]])
    ex = r.standard_normal(20000) * 30
    f = intel_libm_ff._load()
    ref_p = np.array([f(a, b) for a, b in zip(x.tolist(), y.tolist())])
    ref_e = FX._ctypes_exp()
    ref_e = np.array([ref_e(v) for v in ex.tolist()])
    with np.errstate(all="ignore"):
        assert (_bits(FX.host_pow(x, y)) == _bits(ref_p)).all()
        assert (_bits(FX.host_exp(ex)) == _bits(ref_e)).all()
        assert (_bits(ref_p) != _bits(np.power(x, y))).any()           # the comparison can fail (libimf != numpy)
        assert (_bits(ref_e) != _bits(np.exp(ex))).any()
    assert FX.status()["backend"] in ("c-shim", "ctypes-loop")


def test_fx_masks_skip_counters_in_jit_and_while():
    FX.set_mode("libimf")
    r = np.random.default_rng(2)
    x = r.random((6, 50)) * 100 + 0.1
    m = r.random((6, 50)) > 0.5
    FX.reset_counters()

    @jax.jit
    def g(x, m):
        a, b = FX.fx([("e", -x * 0.01, m), ("p", x, 0.2857, None)], "t1")
        return a, b

    a, b = (np.asarray(v) for v in g(x, m))
    assert (a[~m] == 0).all()
    assert (_bits(a[m]) == _bits([FX._ctypes_exp()(v) for v in (-x * 0.01)[m].tolist()])).all()
    assert (_bits(b) == _bits(intel_libm_ff.pow_imf(x, 0.2857))).all()
    c = FX.counters()
    assert c["total"]["calls"] == 1 and c["total"]["lanes"] == int(m.sum()) + x.size and c["tags"]["t1"]["calls"] == 1
    # no selected lane: the callback is skipped on the device
    FX.reset_counters()
    z = np.asarray(jax.jit(lambda x: FX.fx([("e", x, jnp.zeros(x.shape, bool))], "t2")[0])(x))
    assert (z == 0).all() and FX.counters()["total"]["calls"] == 0
    # inside a while loop: one callback per trip while lanes are active (trips 0..2), none afterwards
    FX.reset_counters()

    def body(c):
        i, v = c
        return i + 1, FX.fx([("p", v, 1.1, jnp.full(v.shape, i < 3))], "w")[0] + jnp.where(i < 3, 0.0, v)

    out = np.asarray(jax.jit(lambda v: lax.while_loop(lambda c: c[0] < 5, body, (0, v))[1])(x))
    assert FX.counters()["tags"]["w"]["calls"] == 3 and np.isfinite(out).all()


def test_dq_dev_and_conv_micro_dev_bitwise_vs_cmj_libm_and_mutation():
    import clouds_mstcnv_jax as cmj
    import clouds_mstcnv_dev as md
    FX.set_mode("libm")
    k = SimpleNamespace(**{n: jnp.asarray(v) for n, v in cmj.make_K(TUNE).items()})
    r = np.random.default_rng(3)
    n = 400
    sm = r.random(n) * 1e5 + 1e4
    qm = r.random(n) * 1e-2
    plk, mass, pl = r.random(n) * 0.5 + 0.5, r.random(n) * 3000 + 100, r.random(n) * 900 + 50
    lhx = np.where(r.random(n) > .5, 2.5e6, 2.83e6)
    cond = r.random(n) * 1e-3
    for kind in ("c", "e"):
        ref = cmj.dq_j(k, kind, sm, qm, plk, mass, lhx, pl, cond)
        got = md.dq_dev(k, kind, sm, qm, plk, mass, lhx, pl, cond, None)
        for u, v in zip(ref, got):
            assert (_bits(u) == _bits(v)).all()
    # first-iteration exp supplied from outside (the ascent's CB0 path)
    tp = sm * plk / mass
    e1 = FX.fx([("e", md.qarg(k, tp, lhx), None)])[0]
    got = md.dq_dev(k, "c", sm, qm, plk, mass, lhx, pl, None, None, e_first=e1)
    ref = cmj.dq_j(k, "c", sm, qm, plk, mass, lhx, pl)
    assert all((_bits(u) == _bits(v)).all() for u, v in zip(ref, got))
    # conv_micro
    pl_ = r.random(n) * 900 + 50
    wcu, dwcu = r.random(n) * 8 + 0.5, r.random(n) * 0.3
    tp_ = 230 + r.random(n) * 70
    pland = r.random(n)
    condmu = r.random(n) * 3 + 0.01
    flamw = jnp.power(1000.0 * k.PI * k.CN0 / (condmu + k.TEENY), k.QA)
    flamg = jnp.power(400.0 * k.PI * k.CN0G / (condmu + k.TEENY), k.QA)
    flami = jnp.power(100.0 * k.PI * k.CN0I / (condmu + k.TEENY), k.QA)
    lfrz = np.floor(r.random(n) * 3)
    wcufrz = r.random(n) * 4
    tl1, tl2 = 250 + r.random(n) * 40, 250 + r.random(n) * 40
    ci, cg = r.random(n), r.random(n)
    ref = cmj.conv_micro_j(k, pl_, wcu, dwcu, lfrz, wcufrz, tp_, pland, flamw, flamg, flami, tl1, tl2, ci, cg)
    p04 = FX.fx([("p", pl_ / 1000.0, k.F04, None)])[0]
    pf = FX.fx([("p", 1000.0 / pl_, k.P4, None)])[0]
    got = md.conv_micro_dev(k, pl_, wcu, dwcu, lfrz, wcufrz, tp_, pland, condmu, tl1, tl2, ci, cg, jnp.ones(n, bool), p04, pf)
    for u, v in zip(ref, got):
        assert (_bits(u) == _bits(v)).all()
    # mutation: one constant perturbed -> detected
    k2 = SimpleNamespace(**{**vars(k), "F27": k.F27 * (1 + 1e-12)})
    got2 = md.conv_micro_dev(k2, pl_, wcu, dwcu, lfrz, wcufrz, tp_, pland, condmu, tl1, tl2, ci, cg, jnp.ones(n, bool), p04, pf)
    assert any((_bits(u) != _bits(v)).any() for u, v in zip(ref, got2))


def test_dq_dev_libimf_equals_scalar_reference_and_differs_from_libm():
    import clouds_mstcnv_jax as cmj
    import clouds_mstcnv_dev as md
    FX.set_mode("libimf")
    k = SimpleNamespace(**{n: jnp.asarray(v) for n, v in cmj.make_K(TUNE).items()})
    r = np.random.default_rng(4)
    n = 20000
    sm, qm = r.random(n) * 1e5 + 1e4, r.random(n) * 1e-2
    plk, mass, pl = r.random(n) * 0.5 + 0.5, r.random(n) * 3000 + 100, r.random(n) * 900 + 50
    lhx = np.where(r.random(n) > .5, 2.5e6, 2.83e6)
    sel = r.random(n) > 0.3
    got = np.asarray(jax.jit(lambda *a: md.dq_dev(k, "c", *a[:6], None, jnp.asarray(sel), tag="u")[0])(sm, qm, plk, mass, lhx, pl))
    # scalar reference of cmj.dq_j with the libimf exp, python floats
    K = {a: float(b) for a, b in cmj.make_K(TUNE).items()}
    ex = FX._ctypes_exp()
    out = np.zeros(n)
    for i in range(n):
        slh = lhx[i] * K["DQ_BYSHA"]
        qmt, tp, dqsum = qm[i], sm[i] * plk[i] / mass[i], 0.0
        for _ in range(3):
            e = ex(lhx[i] * (K["QB"] - K["QC"] / max(130.0, tp)))
            qst = K["QA_"] * e / pl[i]
            d = (qmt - mass[i] * qst) / (1.0 + slh * qst * (lhx[i] * K["QC"] / (tp * tp)))
            tp = tp + slh * d / mass[i]
            qmt = qmt - d
            dqsum = dqsum + d
        out[i] = max(0.0, min(dqsum, qm[i])) if qm[i] > 0 else 0.0
    assert (_bits(got[sel]) == _bits(out[sel])).all()      # lanes outside sel are unused by the callers (their exp was not evaluated)
    FX.set_mode("libm")
    lm = np.asarray(md.dq_dev(k, "c", sm, qm, plk, mass, lhx, pl, None, None)[0])
    assert (_bits(lm[sel]) != _bits(out[sel])).any()          # libimf and libm differ in the last bit somewhere


def test_stacked_and_unstacked_host_forms_equal():
    FX.set_mode("libimf")
    r = np.random.default_rng(5)
    x1, x2, x3 = (r.random((4, 30)) * 50 + 0.1 for _ in range(3))
    y2 = r.random((4, 30)) + 0.1
    m = [r.random((4, 30)) > .4 for _ in range(3)]
    kinds = ("e", "p", "p")
    a = FX._host(kinds, "u", x1, m[0], x2, y2, m[1], x3, np.full((4, 30), 0.25), m[2])
    b = FX._host_stacked(kinds, "u", np.stack([x1, x2, x3]), np.stack(m), np.stack([y2, np.full((4, 30), 0.25)]))
    assert all((_bits(a[i]) == _bits(b[i])).all() for i in range(3))
    assert (b[0][~m[0]] == 0).all() and (b[1][~m[1]] == 0).all()
    # through jit: stacked path (same shapes) equals the lane-by-lane scalar libimf
    FX.reset_counters()
    o = jax.jit(lambda x1, x2, x3: FX.fx([("e", -x1 * .01, m[0]), ("p", x2, 1.7, m[1]), ("p", x3, 0.25, None)], "s"))(x1, x2, x3)
    assert FX.counters()["total"]["calls"] == 1
    f = intel_libm_ff._load()
    assert (_bits(o[2]) == _bits(np.array([f(v, 0.25) for v in x3.ravel().tolist()]).reshape(x3.shape))).all()
    assert (_bits(o[1][m[1]]) == _bits(np.array([f(v, 1.7) for v in x2[m[1]].tolist()]))).all()
