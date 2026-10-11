"""D214: the device ADV1D of the MSTCNV subsidence (clouds_adv1d_dev.adv_dev) against the NumPy reference (clouds_mstcnv_batch._adv_groups), bitwise,
on random columns that exercise every branch of the limiter and variable slice lengths."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import clouds_jax_env  # noqa: E402,F401  (before jax)
import jax.numpy as jnp  # noqa: E402

import clouds_adv1d_dev as AD  # noqa: E402
import clouds_mstcnv_batch as mb  # noqa: E402


def _make(n, seed, L1=42, scale=1.0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(L1, n)) * scale
    arr[rng.random((L1, n)) < 0.1] *= -1.0           # sign changes: negative tracer values make the limiter act
    mom = rng.normal(size=(9, L1, n)) * 0.3 * scale
    ml = rng.uniform(0.5, 2.0, size=(L1, n))
    ml[rng.random((L1, n)) < 0.03] = 0.0              # dead cells
    cm = rng.normal(size=(L1, n)) * rng.choice([0.2, 1.0, 2.0], size=(L1, n))   # |dm| / mass above 1 triggers the warning branches
    ld = rng.integers(1, 20, size=n)
    lm = np.minimum(ld + rng.integers(0, 20, size=n), 40)
    sel = rng.random(n) < 0.8
    ierr = np.zeros(n, np.int64)
    lerr = np.zeros(n, np.int64)
    return sel, ld, lm, arr, mom, ml, cm, ierr, lerr


def _ref(args, qlimit):
    sel, ld, lm, arr, mom, ml, cm, ierr, lerr = (np.array(a) for a in args)
    mb._adv_groups(np.flatnonzero(sel), ld, lm, arr, mom, ml, cm, qlimit, ierr, lerr)
    return arr, mom, ierr, lerr


@pytest.mark.parametrize("qlimit", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adv_dev_bitwise_vs_numpy(qlimit, seed):
    args = _make(300, seed)
    want = _ref(args, qlimit)
    got = AD.adv_dev(*(jnp.asarray(a) for a in args), qlimit)
    for name, g, w in zip(("arr", "mom", "ierr", "lerr"), got, want):
        g = np.asarray(g)
        assert g.shape == w.shape, name
        assert np.array_equal(g, w, equal_nan=True), (name, qlimit, seed, int((g != w).sum()), float(np.nanmax(np.abs(g - w))))
