"""Tests for the full-fidelity PBL `advanc` port against REAL-Fortran call records (ffp_*.bin).

Records are stratified samples (per surface type) of every PBL call in one DTsrc step from three
dates. Skipped when the data are absent. x64 is enabled on import; run in its own process:
    JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest fullfidelity/tests/test_pbl_ff.py -q
"""
import os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = [f"{FF}/nov26/ffp_33312.bin", f"{FF}/dec01/ffp_33552.bin", f"{FF}/jan01/ffp_17520.bin"]
pytestmark = pytest.mark.skipif(not all(os.path.exists(f) for f in FILES), reason="real-Fortran PBL records not available")

import pbl_compare as C   # noqa: E402  (imports pbl_ff -> aturb_ff -> enables x64)
import pbl_ff as P        # noqa: E402


def sample(path, per_type=48, seed=0):
    rec = C.load(path)
    rng = np.random.default_rng(seed)
    it = rec[:, 2].astype(int)
    idx = np.concatenate([rng.choice(np.where(it == k)[0], min(per_type, (it == k).sum()), replace=False) for k in (1, 2, 3, 4)])
    return rec[np.sort(idx)]


@pytest.fixture(scope="module")
def results():
    out = []
    for f in FILES:
        rec = sample(f)
        out.append((rec, C.run(rec)))
    return out


def rel(got, ref):
    return np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)


KEY = ["us", "vs", "ws", "ustar", "cm", "ch", "cq", "khs", "kms", "z0m", "z0h", "z0q", "ufluxs", "vfluxs", "qfluxs"]


def test_outputs_match_fortran_at_float64_sensitivity_floor(results):
    for rec, got in results:
        for k in KEY:
            r = rel(got[k], rec[:, C.OUT[k]])
            assert np.median(r) < 1e-11, (k, np.median(r))
            assert r.max() < 5e-8, (k, r.max())          # observed <= 8e-11 (48 records/type); generous
        # temperature / humidity profiles and fluxes with cancellation: absolute vs field scale
        assert np.abs(got["tsv"] - rec[:, C.OUT["tsv"]]).max() < 1e-8
        assert np.abs(got["tfluxs"] - rec[:, C.OUT["tfluxs"]]).max() < 1e-9


def test_not_vacuous(results):
    for rec, got in results:
        it = rec[:, 2].astype(int)
        assert set(np.unique(it)) == {1, 2, 3, 4}
        for k in ("ustar", "cm", "khs", "ufluxs", "tfluxs", "qfluxs", "us"):
            ref = rec[:, C.OUT[k]]
            assert ref.std() > 0
            # port error is orders of magnitude below the natural spread of the output
            assert np.abs(got[k] - ref).max() < 1e-6 * ref.std()
        # the fixed-point iteration and skin effect are actually active: ocean dskin != 0
        assert np.abs(rec[it == 1, C.OUT["dskin"]]).max() > 1e-3


def test_iteration_and_skin_paths_are_exercised(results):
    rec, got = results[0]
    # ustar varies across surface types by >10x -> both smooth/rough z0h branches occur
    ust = rec[:, C.OUT["ustar"]]
    assert ust.max() / ust.min() > 10
    assert (rec[:, 23] > 0.5).any() and (rec[:, 23] < 0.5).any()      # ddml gusti path on and off


def test_mutations_are_detected():
    import jax
    rec = sample(FILES[0], per_type=16)
    base = C.run(rec)
    b0 = max(rel(base[k], rec[:, C.OUT[k]]).max() for k in ("ustar", "cm", "khs"))
    assert b0 < 1e-7
    for name, bad in (("KAPPA", 0.41), ("B1", 19.0), ("GRAV", 9.81), ("SE", 0.11), ("ITMAX", 4)):
        old = getattr(P, name)
        try:
            jax.clear_caches()
            setattr(P, name, bad)
            got = C.run(rec)
            worst = max(rel(got[k], rec[:, C.OUT[k]]).max() for k in ("ustar", "cm", "khs", "us"))
            assert worst > 1e-6, (name, worst)
        finally:
            setattr(P, name, old)
            jax.clear_caches()
