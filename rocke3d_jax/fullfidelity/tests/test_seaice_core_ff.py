"""Tests: sea-ice ground thermodynamics port (seaice_core_ff) vs real-Fortran GROUND_SI dumps (ffi_*.bin).

Covers SEA_ICE (heat diffusion, melt, dew, compression, relayering), SSIDEC (brine drainage, ocean
domain only) and snowice (snow-to-ice conversion). `seaice_thermo="BP"` (the default, not overridden
by the rundeck). Known documented approximation: the real Ti/Ti2b use REAL*16 internally; this port
uses float64 throughout -- validated empirically here, not assumed exact.
"""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffi_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran GROUND_SI dumps not available")

import seaice_core_ff as S    # noqa: E402
import seaice_compare as C    # noqa: E402


def relerr(v):
    ref = abs(v.get("ref", 0))
    return v["abs"] / ref if ref > 1e-6 else v["abs"]


def test_sea_ice_core_matches_fortran():
    """SEA_ICE alone (before SSIDEC/snowice), all real cells, both ocean and lake-ice."""
    worst = {}
    n_melt = 0
    for f in FILES:
        rec = C.load(f)
        for r in rec:
            rows = C.compare_row(r)
            if r[35] > 1e-8 or r[39] > 1e-8:
                n_melt += 1
            for k in ("msi2", "run", "erun", "srun", "melt12", "cmprs", "srox2"):
                worst[k] = max(worst.get(k, 0), relerr(rows[k]))
            worst["hsil"] = max(worst.get("hsil", 0), rows["hsil"]["abs"])
            assert rows["wetsnow_match"]
    for k, v in worst.items():
        assert v < 1e-4, (k, v)
    assert n_melt > 50   # melt/refreeze path genuinely exercised, not vacuous


def test_full_ground_si_matches_fortran():
    """Full GROUND_SI (SEA_ICE -> SSIDEC -> snowice for ocean; SEA_ICE only for lakes)."""
    worst = {}
    for f in FILES:
        rec = C.load(f)
        for r in rec:
            rows = C.compare_full(r)
            for k, v in rows.items():
                worst[k] = max(worst.get(k, 0), relerr(v))
    for k, v in worst.items():
        assert v < 1e-3, (k, v)


def test_ssidec_and_snowice_are_exercised():
    """Brine drainage and snow-ice conversion both fire in a large fraction of ocean cells -- these
    are not rare edge cases in this dataset."""
    ocean = ssidec_active = snowice_active = 0
    for f in FILES:
        rec = C.load(f)
        for r in rec:
            if r[2] > 0.5:
                ocean += 1
                if abs(r[55]) > 1e-12 or abs(r[56]) > 1e-12 or abs(r[57]) > 1e-12:
                    ssidec_active += 1
                snow, msi2 = r[25], r[34]
                if S.RHOI * snow > (S.ACE1I + msi2) * (S.RHOWS - S.RHOI):
                    snowice_active += 1
    assert ssidec_active / ocean > 0.3
    assert snowice_active / ocean > 0.05


def test_no_exceptions_over_full_files():
    for f in FILES:
        rec = C.load(f)
        for r in rec:
            C.run_full(r)   # raises on failure


def test_mutations_are_detected():
    """Monkeypatch core physics functions directly (not module constants, several of which have
    derived counterparts cached at import time, e.g. BYSHI=1/SHI, ACE1I=0.1*RHOI -- mutating the
    base constant alone leaves those unchanged and understates sensitivity)."""
    rec = C.load(FILES[0])
    rec = rec[np.argsort(-rec[:, 34])][:80]     # bias toward large-msi2 ocean cells
    def scale_hsil(old, factor):
        def wrapped(*a, **kw):
            snow, msi1, msi2, hsil, ssil = old(*a, **kw)
            return snow, msi1, msi2, [h * factor for h in hsil], ssil
        return wrapped

    for fname, factor, make in (("Ei", 1.2, None), ("set_snow_ice_layer", 1.0001, scale_hsil)):
        old = getattr(S, fname)
        try:
            if make is not None:
                setattr(S, fname, make(old, factor))
            else:
                setattr(S, fname, lambda *a, _old=old, _f=factor: _old(*a) * _f)
            worst = 0.0
            for r in rec:
                rows = C.compare_full(r)
                for k in ("msi2", "hsil", "erunosi"):
                    worst = max(worst, relerr(rows[k]))
            assert worst > 1e-4, fname
        finally:
            setattr(S, fname, old)
