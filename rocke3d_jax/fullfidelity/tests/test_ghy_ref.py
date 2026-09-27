"""Tests: GHY land-surface port (ghy_ref.GhyColumn) vs real-Fortran land-tile dumps (ffg_<itime>.bin).

Reference implementation is plain Python/NumPy (not yet vectorized/JAX), so this suite runs on a
stratified sample of cells per file rather than all ~1500/file, to keep runtime reasonable.
Coverage note: TRACERS_WATER, dynamic vegetation structure updates, and the fllmt/runoff
"negative-rnf redistribution" branch (k>1 while-loop) are exercised implicitly wherever they occur
in the real data but not isolated in a dedicated test.
"""
import os, sys, glob
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffg_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran GHY (land) dumps not available")

import ghy_ref as G          # noqa: E402
import ghy_compare as C      # noqa: E402


def sample(path, n=120, seed=0):
    rec = C.load(path)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rec), min(n, len(rec)), replace=False))
    return rec[idx]


@pytest.fixture(scope="module")
def all_rows():
    rows = []
    for f in FILES:
        for rec in sample(f):
            rows.append((f, C.compare_cell(rec)))
    return rows


def test_matches_fortran_relative_to_field_scale(all_rows):
    """No bitwise requirement (this is a plain-Python reference, not yet float64-optimized like ATURB/PBL);
    checks each output is within a small fraction of that field's natural scale across all sampled cells."""
    ref_rms = {k: 0.0 for k in C.SCALARS}
    max_abs = {k: 0.0 for k in C.SCALARS}
    for _, rows in all_rows:
        for k in C.SCALARS:
            ref_rms[k] = max(ref_rms[k], abs(rows[k]["ref"]))
            max_abs[k] = max(max_abs[k], rows[k]["abs"])
    for k in C.SCALARS:
        if ref_rms[k] < 1e-10:
            continue
        assert max_abs[k] / ref_rms[k] < 1e-2, (k, max_abs[k], ref_rms[k])
    # the core temperature/heat-flux fields (what matters most) should be much tighter
    for k in ("tbcs", "tsns", "ashg", "alhg", "ae0"):
        assert max_abs[k] / max(ref_rms[k], 1e-10) < 1e-4, k


def test_not_vacuous(all_rows):
    """The compared fields actually vary and the port actually computes something (not all zeros)."""
    vals = {k: [] for k in C.SCALARS}
    for _, rows in all_rows:
        for k in C.SCALARS:
            vals[k].append(rows[k]["ref"])
    for k in C.SCALARS:
        assert float(np.std(vals[k])) > 0, k


def test_snow_model_is_exercised():
    """At least a third of land cells in these dumps carry snow -- the snow_drv/snow_adv path is real."""
    total = active = 0
    for f in FILES:
        rec = C.load(f)
        for i in range(len(rec)):
            static, dynamic, forcing, ent_iters, refs, snowm = C.unpack(rec[i])
            total += 1
            if dynamic["nsn"].max() > 0 and np.abs(dynamic["wsn"]).max() > 1e-6:
                active += 1
    assert active / total > 0.3, (active, total)


def test_no_exceptions_over_full_files():
    """Run every real cell in one file end to end (not just the sample) -- guards against rare code paths."""
    rec = C.load(FILES[0])
    n_fail = 0
    for i in range(len(rec)):
        try:
            C.compare_cell(rec[i])
        except Exception:
            n_fail += 1
    assert n_fail == 0, n_fail


def test_mutations_are_detected():
    """The relative-scale test above must fail if the physics is wrong."""
    rec = sample(FILES[0], n=40)
    # NB: FSN/ELH are derived constants (FSN=LHM*RHOW, ELH=LHE*RHOW) computed once at import, so
    # mutating LHM/LHE after import has no effect -- mutate the derived constants directly.
    for name, bad in (("STBO", 5.0e-8), ("FSN", 2.5e8), ("SHW", 4000.0 * G.RHOW)):
        old = getattr(G, name)
        try:
            setattr(G, name, bad)
            worst = 0.0
            for r in rec:
                rows = C.compare_cell(r)
                for k in ("tbcs", "ashg", "ae0", "alhg", "aevap"):
                    ref = abs(rows[k]["ref"])
                    if ref > 1e-6:
                        worst = max(worst, rows[k]["abs"] / ref)
            assert worst > 1e-4, name
        finally:
            setattr(G, name, old)
