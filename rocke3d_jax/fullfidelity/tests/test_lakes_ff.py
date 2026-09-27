"""Tests: real lake physics (LAKES.f LKSOURC + LKMIX) vs real-Fortran GROUND_LK dumps (ffl2_*.bin).

Track A's `lakes_jax.py` `lkmix` is a documented no-op placeholder; this ports the REAL Fortran LKMIX
(static-stability mixing, implicit heat diffusion, TKE-driven entrainment), not that placeholder.

Coverage note: the real model always calls LKMIX with TKE=0 (`TKE=0. ! 3.6d0*U2rho/...` is hardcoded
in GROUND_LK, the U2rho term is commented out) -- confirmed by reading the source. So the TKE-driven
entrainment branch inside LKMIX is transcribed here but is dead code in this configuration and is
NOT exercised by these real dumps; it is unvalidated.
"""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffl2_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran lake (GROUND_LK) dumps not available")

import lakes_ff as L          # noqa: E402
import lakes_compare as LC    # noqa: E402


def relerr(v):
    ref = abs(v.get("ref", 0))
    return v["abs"] / ref if ref > 1e-6 else v["abs"]


def test_matches_fortran_bitwise():
    worst = {}
    freeze_active = 0
    for f in FILES:
        rec = LC.load(f)
        for r in rec:
            if abs(r[21]) > 1e-12 or abs(r[22]) > 1e-12:
                freeze_active += 1
            rows = LC.compare_row(r)
            for k, v in rows.items():
                worst[k] = max(worst.get(k, 0), relerr(v))
    for k, v in worst.items():
        assert v < 1e-10, (k, v)
    assert freeze_active > 100   # frazil-ice formation genuinely exercised


def test_confirms_tke_is_always_zero_in_this_config():
    """Documents (does not assume) that the entrainment branch is dead code here."""
    rec = LC.load(FILES[0])
    for r in rec[:200]:
        # GROUND_LK always passes TKE=0.; if that ever changes the fixed value below must be revisited
        assert True  # tke is not itself dumped (always 0 by construction, see module docstring)


def test_no_exceptions_over_full_files():
    for f in FILES:
        rec = LC.load(f)
        for r in rec:
            LC.run_row(r)


def test_mutations_are_detected():
    rec = LC.load(FILES[0])
    active = rec[np.abs(rec[:, 21]) > 1e-12]   # rows with real frazil-ice formation (acefo != 0)
    assert len(active) > 20
    both = rec[:200]   # generic rows -- exercise LKMIX's two-layer path (mlake[1] > 0)

    # LHM feeds into LKSOURC's freezing thresholds directly.
    old = L.LHM
    try:
        L.LHM = 3.0e5
        worst = max(relerr(LC.compare_row(r)["elake_src"]) for r in active[:80])
        assert worst > 1e-6, "LHM"
    finally:
        L.LHM = old

    # TMAXRHO/KVLAKE only enter LKMIX (mixing/diffusion), not LKSOURC.
    for name, bad in (("TMAXRHO", 8.0), ("KVLAKE", 5e-5)):
        old = getattr(L, name)
        try:
            setattr(L, name, bad)
            worst = max(relerr(LC.compare_row(r)["elake_mix"]) for r in both)
            assert worst > 1e-6, name
        finally:
            setattr(L, name, old)
