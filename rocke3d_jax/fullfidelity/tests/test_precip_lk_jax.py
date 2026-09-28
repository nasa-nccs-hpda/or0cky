"""Tests (D27): lakes_core_jax.precip_lk (batched PRECIP_LK, Stage 1 of the DYNSI/ocean port) vs real Fortran,
the plain-Python reference, and mutation checks. gtemp/gtemp2/gtempr are only checked for flake>0 cells --
see precip_lk_compare.py's module docstring for why the flake<=0 pass-through case can't be independently
checked with this dump."""
import os, sys, glob
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffv_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran PRECIP_LK dumps not available")

import lakes_core_jax as J        # noqa: E402
import lakes_ff as L              # noqa: E402
import precip_lk_compare as C     # noqa: E402
import precip_lk_jax_compare as JC  # noqa: E402


@pytest.fixture(scope="module")
def rec():
    return JC.load_all(FILES)


def test_matches_fortran(rec):
    out = JC.batched_precip_lk(rec)
    for k, c in (("mwl", 16), ("gml", 17), ("tlake", 18), ("mldlk", 19), ("dlake", 20), ("glake", 21)):
        d = np.abs(np.asarray(out[k]) - rec[:, c])
        assert d.max() == 0.0, k
    flake_pos = rec[:, 2] > 0
    for k, c in (("gtemp", 22), ("gtemp2", 23), ("gtempr", 24)):
        d = np.abs(np.asarray(out[k])[flake_pos] - rec[flake_pos, c])
        assert d.max() == 0.0, k
    assert len(rec) > 15000
    assert int(flake_pos.sum()) > 10000


def test_non_vacuous(rec):
    flake_pos = rec[:, 2] > 0
    flice_only = (rec[:, 2] <= 0) & (rec[:, 3] > 0)
    assert flake_pos.sum() > 10000
    assert flice_only.sum() > 1000
    assert (rec[:, 9] != 0).sum() > 5     # real lake-ice melt (MELTI) cells


def test_matches_plain_python_reference_row_for_row(rec):
    out = JC.batched_precip_lk(rec)
    idx = np.linspace(0, len(rec) - 1, 300).astype(int)
    worst = 0.0
    for i in idx:
        r = rec[i]
        py = L.precip_lk(r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
                         r[15], 0.0, 0.0, 0.0)
        for k in ("mwl", "gml", "tlake", "mldlk"):
            got = float(np.asarray(out[k])[i])
            worst = max(worst, abs(got - py[k]) / max(abs(py[k]), 1e-6))
    assert worst < 1e-9


def test_jit_compiles_and_matches_eager(rec):
    a = lambda c: jnp.asarray(rec[:, c])
    z = jnp.zeros(len(rec))
    args = (a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10), a(11), a(12), a(13), a(14), a(15), z, z, z)
    eager = J.precip_lk(*args)
    jitted = jax.jit(J.precip_lk)(*args)
    for k in ("mwl", "gml", "tlake"):
        d = jnp.abs(eager[k] - jitted[k])
        scale = jnp.maximum(jnp.abs(eager[k]), 1e-6)
        assert float(jnp.max(d / scale)) < 1e-9, k   # relative: mwl/gml scales span ~O(1) to ~1e21


def test_no_nan_or_inf(rec):
    out = JC.batched_precip_lk(rec)
    for k, v in out.items():
        v = np.asarray(v)
        assert not np.isnan(v).any(), k
        assert not np.isinf(v).any(), k


def test_mutations_are_detected(rec):
    out = JC.batched_precip_lk(rec)
    old = J.SHW
    try:
        J.SHW = old * 1.01
        out2 = JC.batched_precip_lk(rec)
        d = np.max(np.abs(np.asarray(out2["tlake"]) - np.asarray(out["tlake"])) /
                   np.maximum(np.abs(np.asarray(out["tlake"])), 1e-6))
        assert d > 1e-4
    finally:
        J.SHW = old
