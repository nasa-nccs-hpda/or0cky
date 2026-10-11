"""D214: the device OADVT2 X pre-pass (oadvt_xpre_dev.xpre_dev) against the NumPy original (oadvt_vec._x_prepass), bitwise, including the stale entries."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import clouds_jax_env  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402

import oadvt_vec as OV  # noqa: E402
import oadvt_xpre_dev as XD  # noqa: E402
from oadvt2_ff import IM, JM, LMO  # noqa: E402


def _make(seed, mu_scale, fill=0.7):
    rng = np.random.default_rng(seed)
    shp = (IM + 1, JM + 1, LMO + 1)
    mm = rng.uniform(0.5, 2.0, shp) * 1e13
    mu = rng.normal(size=shp) * mu_scale * 1e12
    # land/sea structure with several segments per row: lmu[i, j] in 0..LMO, run lengths vary, sea fraction `fill`
    lmu = np.zeros((IM + 1, JM + 1), int)
    for j in range(JM + 1):
        i = 1
        while i <= IM:
            run = int(rng.integers(1, 12))
            if rng.random() < fill:
                lmu[i:i + run, j] = rng.integers(1, LMO + 1)
            i += run
    lmm = np.maximum(lmu, 1)
    return mm, mu, lmu, lmm


@pytest.mark.parametrize("seed,scale", [(0, 0.5), (1, 3.0), (2, 12.0), (3, 1.0)])
def test_xpre_dev_bitwise(seed, scale):
    mm, mu, lmu, lmm = _make(seed, scale)
    dt = 3600.0
    s0, n0, l0 = OV._x_prepass(mm, mu, dt, lmu, lmm)
    s1, n1, l1 = XD.xpre_dev(jnp.asarray(mm), jnp.asarray(mu), dt, jnp.asarray(lmu))
    s1, n1, l1 = np.asarray(s1), np.asarray(n1), np.asarray(l1)
    assert np.array_equal(l0, l1), 'lane'
    assert np.array_equal(n0, n1), ('ncour', int((n0 != n1).sum()))
    assert np.array_equal(s0, s1), ('snap', int((s0 != s1).sum()), float(np.abs(s0 - s1).max()))
    assert n0.max() >= 1


def test_xpre_dev_exercises_courant_scaling():
    mm, mu, lmu, lmm = _make(5, 12.0)
    s0, n0, l0 = OV._x_prepass(mm, mu, 3600.0, lmu, lmm)
    assert n0.max() > 1, 'the test data must contain lanes with courant substeps'
