"""Tests: sea-ice formation (ADDICE) and lateral/complete melt (SIMELT) vs real-Fortran dumps
(ffn_<itime>.bin, ffm_<itime>.bin) -- completes the sea-ice ground-thermodynamics port (see also
test_seaice_core_ff.py for SEA_ICE/SSIDEC/snowice)."""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
N_FILES = sorted(glob.glob(f"{FF}/*/ffn_*.bin"))
M_FILES = sorted(glob.glob(f"{FF}/*/ffm_*.bin"))
pytestmark = pytest.mark.skipif(len(N_FILES) < 6 or len(M_FILES) < 6,
                                reason="real-Fortran ADDICE/SIMELT dumps not available")

import seaice_core_ff as S     # noqa: E402
import addice_compare as A     # noqa: E402
import simelt_compare as M     # noqa: E402


def relerr(v):
    ref = abs(v.get("ref", 0))
    return v["abs"] / ref if ref > 1e-6 else v["abs"]


def test_addice_matches_fortran():
    worst = {}
    active = 0
    for f in N_FILES:
        rec = A.load(f)
        for r in rec:
            if abs(r[17]) > 1e-12 or abs(r[15]) > 1e-12:   # acefo or acefi
                active += 1
            rows = A.compare_row(r)
            for k, v in rows.items():
                worst[k] = max(worst.get(k, 0), relerr(v))
    for k, v in worst.items():
        assert v < 1e-6, (k, v)
    assert active > 500   # new-ice formation genuinely exercised, not a rare edge case


def test_simelt_matches_fortran():
    worst = {}
    melted_out = 0
    for f in M_FILES:
        rec = M.load(f)
        for r in rec:
            rows = M.compare_row(r)
            if rows["melted_out"]:
                melted_out += 1
            for k, v in rows.items():
                if k == "melted_out":
                    continue
                worst[k] = max(worst.get(k, 0), relerr(v))
    for k, v in worst.items():
        assert v < 1e-10, (k, v)
    assert melted_out > 50   # the complete-melt-out branch (TSIL well-defined there) is exercised


def test_no_exceptions_over_full_files():
    for f in N_FILES:
        rec = A.load(f)
        for r in rec:
            A.run_row(r)
    for f in M_FILES:
        rec = M.load(f)
        for r in rec:
            M.run_row(r)


def test_mutations_are_detected():
    n_rec = A.load(N_FILES[0])
    n_rec = n_rec[np.argsort(-np.abs(n_rec[:, 17]))][:60]   # bias toward active new-ice cells
    m_rec = M.load(M_FILES[0])[:200]

    for fname, factor in (("Ei", 1.2),):
        old = getattr(S, fname)
        try:
            setattr(S, fname, lambda *a, _old=old, _f=factor: _old(*a) * _f)
            worst = max(relerr(A.compare_row(r)["hsil"]) for r in n_rec)
            assert worst > 1e-4, fname
        finally:
            setattr(S, fname, old)

    old = S.SILMFAC
    try:
        S.SILMFAC = old * 5
        worst = max(relerr(M.compare_row(r)["roice"]) for r in m_rec)
        assert worst > 1e-6
    finally:
        S.SILMFAC = old
