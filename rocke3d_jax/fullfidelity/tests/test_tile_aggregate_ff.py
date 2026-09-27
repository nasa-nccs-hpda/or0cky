"""Tests: tile aggregation (tile_aggregate_ff) vs real-Fortran composite fields (fft_<itime>.bin)."""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/fft_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran tile-aggregation dumps not available")

import tile_aggregate_ff as T   # noqa: E402


def test_matches_fortran():
    worst = {}
    total = 0
    for f in FILES:
        rec = T.load(f)
        total += len(rec)
        ftype, patch, ref = T.unpack(rec)
        assert np.allclose(ftype.sum(-1), 1.0, atol=1e-12)   # the 4 patch fractions always cover the cell
        got = T.aggregate(ftype, patch)
        for k in T.FIELDS:
            d = np.abs(np.asarray(got[k]) - ref[k])
            worst[k] = max(worst.get(k, 0.0), float(d.max() / max(np.sqrt((ref[k] ** 2).mean()), 1e-30)))
    assert total > 30000
    for k, v in worst.items():
        assert v < 1e-5, (k, v)


def test_not_vacuous():
    rec = T.load(FILES[0])
    ftype, patch, ref = T.unpack(rec)
    for k in T.FIELDS:
        assert ref[k].std() > 0
    # not all cells are single-type: at least some have a genuine multi-patch mix
    assert ((ftype > 0.001) & (ftype < 0.999)).any(axis=-1).sum() > 100


def test_mutations_are_detected():
    rec = T.load(FILES[0])
    ftype, patch, ref = T.unpack(rec)
    bad = T.aggregate(ftype * 1.1, patch)   # break the weighting -> should no longer match
    worst = max(float(np.abs(np.asarray(bad[k]) - ref[k]).max() / max(np.sqrt((ref[k] ** 2).mean()), 1e-30))
                for k in T.FIELDS)
    assert worst > 1e-3
