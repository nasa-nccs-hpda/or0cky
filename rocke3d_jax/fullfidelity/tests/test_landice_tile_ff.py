"""Tests: SURFACE_LANDICE.f tile flux logic (landice_tile_ff) vs real-Fortran tile records (ffl_*.bin).
Coverage note: the dew-limit branch never triggers in these 6 steps (transcribed, not validated)."""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffl_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran land-ice tile records not available")

import landice_tile_ff as L   # noqa: E402


@pytest.mark.parametrize("path", FILES)
def test_landice_tile_matches_fortran(path):
    r = L.load(path)
    assert len(r) > 600
    g = L.tile_fluxes(L.to_inputs(r))
    for k, c in L.OUT.items():
        ref = r[:, c]
        scale = max(np.sqrt((ref ** 2).mean()), 1e-30)
        assert np.abs(np.asarray(g[k]) - ref).max() < 1e-9 * scale + 1e-18, k
    for k in ("shdt", "evhdt", "trhdt", "dth1", "uflux1", "f1dt"):
        ref = r[:, L.OUT[k]]
        assert ref.std() > 0


def test_mutations_are_detected():
    r = L.load(FILES[0])
    d = L.to_inputs(r)
    for name, val in (("HC1LI", 3.0e5), ("Z2LI3L", 0.5), ("STBO", 5.67e-8)):
        old = getattr(L, name)
        try:
            setattr(L, name, val)
            g = L.tile_fluxes(d)
            worst = max(np.abs(np.asarray(g[k]) - r[:, L.OUT[k]]).max() / r[:, L.OUT[k]].std() for k in ("shdt", "f1dt", "dth1"))
            assert worst > 1e-6, name
        finally:
            setattr(L, name, old)
