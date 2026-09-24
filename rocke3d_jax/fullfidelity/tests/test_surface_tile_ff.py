"""Tests: SURFACE.f ocean/lake + sea-ice tile flux logic (surface_tile_ff) vs real-Fortran tile records (ffs_*.bin).
Skipped when data absent. Coverage note: the lake-evaporation, dew, lake heat-flux and ice-melt-clip limiter
branches are not triggered by these 6 steps, so they are transcribed but NOT validated against Fortran."""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffs_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran SURFACE tile records not available")

import surface_tile_ff as S   # noqa: E402


def check(rec, got, tol_rel=1e-9):
    for k, c in S.OUT.items():
        m = rec[:, 2] == 2 if k == "tg2" else np.ones(len(rec), bool)
        ref = rec[m, c]
        d = np.abs(np.asarray(got[k])[m] - ref)
        scale = max(np.sqrt((ref ** 2).mean()), 1e-30)
        assert d.max() < tol_rel * scale + 1e-18, (k, d.max(), scale)


@pytest.mark.parametrize("path", FILES)
def test_tile_fluxes_match_fortran(path):
    rec = S.load(path)
    it = rec[:, 2].astype(int)
    assert (it == 1).sum() > 1000 and (it == 2).sum() > 300               # both tile types present
    got = S.tile_fluxes(S.to_inputs(rec))
    check(rec, got)
    for k in ("shdt", "evhdt", "trhdt", "dth1", "dq1", "dmua"):           # non-vacuous: outputs vary widely
        ref = rec[:, S.OUT[k]]
        assert ref.std() > 0
        assert np.abs(np.asarray(got[k]) - ref).max() < 1e-9 * ref.std()


def test_mutations_are_detected():
    rec = S.load(FILES[0])
    base = S.to_inputs(rec)
    for name, val in (("STBO", 5.67e-8), ("SHA", 1004.0), ("LHE", 2.51e6)):
        old = getattr(S, name)
        try:
            setattr(S, name, val)
            got = S.tile_fluxes(base)
            worst = max(np.abs(np.asarray(got[k]) - rec[:, S.OUT[k]]).max() / max(rec[:, S.OUT[k]].std(), 1e-30)
                        for k in ("shdt", "evhdt", "trhdt", "dth1"))
            assert worst > 1e-6, name
        finally:
            setattr(S, name, old)
