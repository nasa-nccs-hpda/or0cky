"""D183 tests for jax_imf.py (libimf host callbacks for the JAX stages).  Own process, XLA flags first:
    RUN_XLA_FLAG_TESTS=1 taskset -c 5-7 env OMP_NUM_THREADS=1 python -m pytest tests/test_jax_imf.py
Skipped when libimf is missing, when the XLA flags are not in effect (jax imported earlier without them) or when ff_data is absent.
The slow MSTCNV stage test (about 5 min of XLA compile) additionally needs RUN_SLOW_JAX_IMF=1.
Reference everywhere = the libimf-mode NumPy chain.  No C2 claim."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402

import intel_libm_ff as il  # noqa: E402

flags_ok = 'jax' not in sys.modules or 'algsimp' in os.environ.get('XLA_FLAGS', '')
pytestmark = [pytest.mark.skipif(not flags_ok, reason="XLA flags not in effect (jax imported earlier); run in a fresh process"),
              pytest.mark.skipif(not il.available(), reason="Intel libimf not available on this host")]

DATE = 'nov26'


def _have():
    import clouds_condse_io as cio
    import dyn_step as ds
    return cio.have_dumps(DATE) and ds.available(DATE)


needs_dumps = pytest.mark.skipif(not _have(), reason="ff_data dumps not present on this host") if il.available() else pytest.mark.skip("no libimf")


def _scalar_ref(x, y):
    f = il._load()
    return np.array([f(float(a), float(y)) for a in x.ravel()]).reshape(x.shape)


def test_pow_callback_bitwise_and_not_vacuous():
    import jax
    import jax.numpy as jnp
    import jax_imf as JI
    r = np.random.default_rng(3)
    x = 100.0 + 900.0 * r.random((400, 500))                       # PMID-like pressures
    kapa = 0.2857
    ref = _scalar_ref(x, kapa)
    got = np.asarray(jax.jit(lambda a: JI.pow_imf(a, kapa))(jnp.asarray(x)))
    assert np.array_equal(got, ref)
    got2 = np.asarray(jax.jit(lambda a, k: JI.pow_imf(a, k))(jnp.asarray(x), jnp.asarray(kapa)))    # traced scalar exponent
    assert np.array_equal(got2, ref)
    plain = np.asarray(jax.jit(lambda a: jnp.power(a, kapa))(jnp.asarray(x)))
    assert (plain != ref).sum() > 0                                  # libimf differs from XLA on this sample: the comparison is not vacuous
    ya = 0.2 + r.random(x.shape)
    assert np.array_equal(np.asarray(jax.jit(JI.pow_imf)(jnp.asarray(x), jnp.asarray(ya))),
                          np.array([il._load()(float(a), float(b)) for a, b in zip(x.ravel(), ya.ravel())]).reshape(x.shape))


def test_exp_callback_bitwise():
    import jax
    import jax.numpy as jnp
    import jax_imf as JI
    lib = JI._lib()
    x = np.random.default_rng(4).uniform(-30.0, 3.0, 100000)
    got = np.asarray(jax.jit(JI.exp_imf)(jnp.asarray(x)))
    assert np.array_equal(got, np.array([lib.exp(float(v)) for v in x]))


def test_meter_counts_calls_and_bytes():
    import jax
    import jax.numpy as jnp
    import jax_imf as JI
    JI.METER.reset()
    x = jnp.full((10, 20), 3.0)
    jax.jit(lambda a: JI.pow_imf(a, 0.5, site='pow@test'))(x).block_until_ready()
    t = JI.METER.totals()
    assert t['calls'] == 1 and t['elements'] == 200 and t['bytes_in'] == 1600 and t['bytes_out'] == 1600


def test_imfkapa_uses_libimf_in_reflected_pow():
    import jax_imf as JI
    k = JI.ImfKapa(0.2857)
    assert .01 ** k == il._load()(0.01, 0.2857)
    assert isinstance(k, float) and float(k) == 0.2857


def test_conv_micro_source_rewrite_finds_exactly_four():
    import jax_imf as JI
    src = JI._conv_micro_imf_source()
    assert '**' not in src and src.count('jnp.power(((') == 4


def test_patch_contexts_restore_modules():
    import dyn_jax_aflux as jaf
    import dyn_jax_pgf as jpg
    import clouds_mstcnv_jax as mj
    import jax.numpy as jnp
    import jax_imf as JI
    f0 = mj.conv_micro_j
    with JI.patch_dyn():
        assert isinstance(jaf.jnp, JI.JnpImf) and isinstance(jpg.jnp, JI.JnpImf)
    assert jaf.jnp is jnp and jpg.jnp is jnp
    with JI.patch_mstcnv():
        assert isinstance(mj.jnp, JI.JnpImf) and mj.conv_micro_j is not f0
    assert mj.jnp is jnp and mj.conv_micro_j is f0


@needs_dumps
def test_pek_real_state_bitwise_to_libimf_numpy():
    import dyn_step as ds
    import jax_imf as JI
    it0 = dict(ds.DATES)[DATE]
    s = ds.load_state(ds.state_path(DATE, it0, 1))
    pedn = np.ascontiguousarray(s['pedn'], dtype=np.float64)
    ctx = ds.load_ctx(DATE, imf_pow=True)
    kapa = float(ctx.gg['kapa'])
    ref = il.pow_imf(pedn, kapa)
    assert np.array_equal(np.array(JI.pek(pedn, kapa)), ref)


@needs_dumps
def test_advecm_matop_stage_bitwise_to_libimf_numpy_and_plain_is_not():
    """AFLUX -> ADVECM stage of pass 1 from the real step-0 state: JAX with callbacks == libimf NumPy (all written fields), plain JAX is not."""
    import dyn_step as ds
    import dyn_step_jax2 as dj2
    import jax_imf as JI
    it0 = dict(ds.DATES)[DATE]
    ctx = ds.load_ctx(DATE, imf_pow=True)
    with JI.patch_dyn():
        kit_imf = dj2.Kit(ctx)
    kit_plain = dj2.Kit(ctx)
    s1 = ds.load_state(ds.state_path(DATE, it0, 1))
    seen = {}

    def hook(when, st, w, c):
        if st.kind != 'advecm':
            return
        if when == 'pre':
            if seen:
                return
            seen['w1'] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            seen['w2'] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            dj2.exec_stage_jax(st, seen['w1'], c, kit_plain, ('advecm',))
            with JI.patch_dyn():
                dj2.exec_stage_jax(st, seen['w2'], c, kit_imf, ('advecm',))
        elif 'w2' in seen and 'plain_pk_diff' not in seen:
            for k in ('PEDN', 'PMID', 'PDSIG', 'PK', 'P'):
                assert np.array_equal(seen['w2'][k], w[k]), k
            seen['plain_pk_diff'] = int((seen['w1']['PK'] != w['PK']).sum())

    plan = [st for st in ds.step_plan(nstep=None) if st.kind in ('aflux', 'advecm')][:2]
    ds.dyn_step(s1, ctx, itime=it0, plan=plan, hook=hook)
    assert seen and seen['plain_pk_diff'] > 0


@needs_dumps
def test_lscond_stage_bitwise_to_libimf_numpy_batch():
    import clouds_lscond_jax as lj
    import jax_imf_stage_compare as SC
    import jax_imf as JI
    c = SC._capture(DATE)
    S, P = c['ls_in']
    So, Wo = c['ls_out']
    JI.METER.reset()
    with JI.patch_lscond_meter():
        S2, W2 = lj.lscond_jax(copy.deepcopy(S), P, mode='libm')
    bad = {k: v for k, v in {**SC._cmp_dict(S2, So), **{'W.' + k: v for k, v in SC._cmp_dict(W2, Wo).items()}}.items() if v.get('n_diff')}
    assert not bad, bad
    assert JI.METER.totals()['calls'] > 100
    S3, W3 = lj.lscond_jax(copy.deepcopy(S), P, mode='xla')            # plain XLA differs from the libimf reference (not vacuous)
    assert sum(v['n_diff'] for v in SC._cmp_dict(S3, So).values() if 'n_diff' in v) > 0


@needs_dumps
@pytest.mark.skipif(not os.environ.get('RUN_SLOW_JAX_IMF'), reason="slow (about 5 min of XLA compile); set RUN_SLOW_JAX_IMF=1")
def test_mstcnv_stage_bitwise_to_libimf_numpy_batch():
    import clouds_mstcnv_jax as mj
    import jax_imf_stage_compare as SC
    import jax_imf as JI
    c = SC._capture(DATE)
    R, tune = c['mc_in']
    JI.METER.reset()
    with JI.patch_mstcnv():
        o = mj.mstcnv_jax(copy.deepcopy(R), tune)
    bad = {k: v for k, v in SC._cmp_dict(o, c['mc_out']).items() if v.get('n_diff')}
    assert not bad, bad
    assert JI.METER.totals()['calls'] > 100
