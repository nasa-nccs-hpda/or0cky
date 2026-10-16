"""D183 tests of libimf_ops (own unit only).  Run in its own process:  python -m pytest tests/test_libimf_ops.py
Skipped when the Intel libimf is absent.  Checks: callback pow/exp equal intel_libm_ff / clouds_lscond_size_ff bitwise (inside jit and lax.scan),
mask semantics, counters, mode lock, and that the comparison can fail (libimf differs from numpy on random data)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402
import intel_libm_ff  # noqa: E402

pytestmark = pytest.mark.skipif(not intel_libm_ff.available(), reason="Intel libimf not available")


def test_ops():
    import jax
    import libimf_ops as L
    import clouds_lscond_size_ff as sz
    L.set_mode('libimf')
    assert L.mode() == 'libimf'
    r = np.random.default_rng(3)
    x = r.random((6, 72, 46)) * 1000 + 1e-3
    L.reset_counters()
    g = np.asarray(jax.jit(lambda a: L.pow(a, 0.2857))(x))
    assert np.array_equal(g, intel_libm_ff.pow_imf(x, 0.2857))
    assert (g != np.power(x, 0.2857)).any()          # the comparison can fail
    c = L.counters()
    assert c['pow']['calls'] == 1 and c['pow']['elements'] == x.size and c['pow']['bytes_out'] == x.nbytes
    e = r.standard_normal(2000) * 30
    sz.use_imf(True)
    m = r.random(2000) > .5
    h = np.asarray(jax.jit(L.exp)(e, m))
    assert np.array_equal(h[m], np.array([sz.ex(v) for v in e[m]])) and (h[~m] == 0).all()
    s = np.asarray(jax.jit(lambda a: jax.lax.scan(lambda c_, v: (c_, L.pow(v, c_)), 0.4, a)[1])(x))
    assert np.array_equal(s, intel_libm_ff.pow_imf(x, 0.4))
    with pytest.raises(RuntimeError):
        L.set_mode('libm')                           # mode locked after first trace
    assert L.status()['effective'] == 'libimf'
