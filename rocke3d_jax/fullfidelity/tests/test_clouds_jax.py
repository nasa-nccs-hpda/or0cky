"""Tests for the JAX cloud physics (D145-D147): clouds_lscond_jax.py, clouds_mstcnv_jax.py, clouds_condse_jax.py.

Reference = the numpy batch (itself bit-identical to the validated per-column ports in libm mode).  Real inputs are captured from one chained
step of the real dumps ff_data/<date>/ffc_cse_* (nov26, step 0; skipped when absent).  All comparisons are BITWISE (value and bit pattern).
Runtime is dominated by the one-off XLA compile of the MSTCNV event kernel (~2.5 min per fresh process; set CLOUDS_JAX_CACHE=<dir> to reuse).
Mutation tests change one input/constant of the JAX path and require the comparison to get worse (the comparison is not vacuous)."""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402

import clouds_condse_batch as cb  # noqa: E402
import clouds_condse_ff as cf  # noqa: E402
import clouds_condse_io as cio  # noqa: E402

DATE, IT0 = dict(cio.DATES).get("nov26") and ("nov26", dict(cio.DATES)["nov26"])
HAVE = cio.have_dumps(DATE)
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data dumps not present on this host")
_cache = {}


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, float)).view(np.int64)


def ndiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int((_bits(a) != _bits(b)).sum() - ((np.isnan(a) & np.isnan(b)) & (_bits(a) != _bits(b))).sum())


def _capture():
    """Run the numpy batched CONDSE on step 0 and record the inputs/outputs of the LSCOND and MSTCNV calls."""
    if "cap" in _cache:
        return _cache["cap"]
    cfg = cf.make_cfg(DATE)
    inp, ref = cio.load_step(DATE, IT0)
    store = {}
    o_ls, o_mc = cb.lb.lscond_batch, cb.mb.mstcnv_batch

    def ls(S, P):
        store["ls_in"] = copy.deepcopy((S, P))
        S2, W = o_ls(S, P)
        store["ls_out"] = copy.deepcopy((S2, W))
        return S2, W

    def mcw(R, tune):
        store["mc_in"] = copy.deepcopy((R, tune))
        o = o_mc(R, tune)
        store["mc_out"] = copy.deepcopy(o)
        return o

    cb.lb.lscond_batch, cb.mb.mstcnv_batch = ls, mcw
    try:
        X, _ = cb.condse_step_batch(inp, cfg, ms={})
    finally:
        cb.lb.lscond_batch, cb.mb.mstcnv_batch = o_ls, o_mc
    store.update(inp=inp, cfg=cfg, X=X, ref=ref)
    _cache["cap"] = store
    return store


# ---------------------------------------------------------------------------- environment / primitives (synthetic)
def test_xla_flags_and_division_rewrite_synthetic():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    assert "xla_disable_hlo_passes=algsimp" in clouds_jax_env.flags() and "xla_cpu_max_isa=AVX" in clouds_jax_env.flags()
    r = np.random.default_rng(1)
    x, y, z = r.random(20000) * 3, r.random(20000) + .1, r.random(20000) + .1
    jx, jy, jz = map(jnp.asarray, (x, y, z))
    assert np.array_equal(np.asarray(jax.jit(lambda a: a / 35.0)(jx)), x / 35.0)
    assert np.array_equal(np.asarray(jax.jit(lambda a, b, c: a / (b / c))(jx, jy, jz)), x / (y / z))
    assert np.array_equal(np.asarray(jax.jit(lambda a, b, c: a * b + c)(jx, jy, jz)), x * y + z)


def test_xla_exp_pow_equal_libm_on_this_backend_synthetic():
    import math
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    r = np.random.default_rng(0)
    x = r.uniform(-30, 3, 50000)
    assert np.array_equal(np.asarray(jax.jit(jnp.exp)(jnp.asarray(x))), np.array([math.exp(v) for v in x.tolist()]))
    y = r.random(20000) + 1e-3
    assert np.array_equal(np.asarray(jax.jit(lambda u, e: jnp.power(u, e))(jnp.asarray(y), jnp.asarray(1.0 / 3.0))), np.array([v ** (1.0 / 3.0) for v in y.tolist()]))


# ---------------------------------------------------------------------------- LSCOND
@needs_dumps
@pytest.mark.parametrize("mode", ["libm", "xla"])
def test_lscond_jax_bitwise_vs_numpy_batch_real_step(mode):
    import clouds_lscond_jax as lj
    c = _capture()
    S, P = c["ls_in"]
    So, Wo = c["ls_out"]
    S2, W2 = lj.lscond_jax(copy.deepcopy(S), P, mode=mode)
    bad = {k: ndiff(So[k], S2[k]) for k in S2 if k in So and ndiff(So[k], S2[k])}
    bad.update({"W." + k: ndiff(Wo[k], W2[k]) for k in Wo if k in W2 and ndiff(Wo[k], W2[k])})
    assert not bad, bad


@needs_dumps
def test_lscond_jax_mutation_detected():
    import clouds_lscond_jax as lj
    c = _capture()
    S, P = c["ls_in"]
    So, Wo = c["ls_out"]
    S1 = copy.deepcopy(S)
    S1["ql"] = S1["ql"] * (1 + 1e-15)                      # last-bit perturbation of the water vapour input
    S2, W2 = lj.lscond_jax(S1, P, mode="xla")
    assert ndiff(So["rh"], S2["rh"]) > 0 and ndiff(So["tl"], S2["tl"]) + ndiff(So["ql"], S2["ql"]) > 0
    P2 = dict(P, dtsrc=P["dtsrc"] * (1 + 1e-12))
    S3, _ = lj.lscond_jax(copy.deepcopy(S), P2, mode="xla")
    assert ndiff(So["tl"], S3["tl"]) > 0


# ---------------------------------------------------------------------------- MSTCNV
@needs_dumps
def test_mstcnv_jax_bitwise_vs_numpy_batch_real_step():
    import clouds_mstcnv_jax as mj
    c = _capture()
    R, tune = c["mc_in"]
    o = mj.mstcnv_jax(R, tune)
    bad = {k: ndiff(v, o[k]) for k, v in c["mc_out"].items() if ndiff(v, o[k])}
    assert not bad, bad
    assert (o["lmcmin"] > 0).sum() > 100                   # real convection is present in the step (the test is not vacuous)


@needs_dumps
def test_mstcnv_jax_mutation_detected():
    import clouds_mstcnv_jax as mj
    c = _capture()
    R, tune = c["mc_in"]
    o = mj.mstcnv_jax(R, dict(tune, entrainment_cont1=tune["entrainment_cont1"] * 1.5))
    assert sum(ndiff(v, o[k]) > 0 for k, v in c["mc_out"].items()) >= 5


# ---------------------------------------------------------------------------- CONDSE
@needs_dumps
def test_condse_jax_bitwise_vs_numpy_batch_and_real_exit_state():
    import clouds_condse_compare as cc
    import clouds_condse_jax as cj
    c = _capture()
    Xj, _ = cj.condse_step_jax(c["inp"], c["cfg"], ms={})
    bad = {k: ndiff(c["X"][k], Xj[k]) for k in c["X"] if k in Xj and ndiff(c["X"][k], Xj[k])}
    assert not bad, bad
    G = c["cfg"]["geom"]
    n_fail = 0
    for n in cc.ALL_FIELDS:
        if n in Xj and n in c["ref"] and n not in ("UALIJ", "VALIJ"):
            n_fail += cc.stat(Xj[n], c["ref"][n], None)["status"] == "FAIL"
    # same statistical status as the numpy batch (libm mode flips some threshold columns against the real model: documented in D126/D132)
    n_fail_np = sum(cc.stat(c["X"][n], c["ref"][n], None)["status"] == "FAIL" for n in cc.ALL_FIELDS if n in c["X"] and n in c["ref"] and n not in ("UALIJ", "VALIJ"))
    assert n_fail == n_fail_np
