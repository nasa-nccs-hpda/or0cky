"""Tests (D22): land tile (our PBL itype 4 -> our JAX GHY -> land patch) in the surface chain, single substep and chained
over two substeps with land included, vs the real dumps. Ent exports, forcing and TRUP_in_rad(land) are recorded/inferred."""
import os, sys
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = {"nov26": 33312, "dec01": 33552, "jan01": 17520}


def _have(d, it):
    need = [f"fft_{it}.bin", f"ffs_{it}.bin", f"ffp_{it}.bin", f"ffl_{it}.bin", f"ffg_{it}.bin"] + \
           [f"ffa_{it}_c{n}_{s}.bin" for n in (1, 2) for s in ("in", "out")]
    return all(os.path.exists(f"{FF}/{d}/{f}") for f in need)


CASES = [(d, it) for d, it in DATES.items() if _have(d, it)]
pytestmark = pytest.mark.skipif(len(CASES) < 3, reason="real-Fortran dumps not available")

import land_chain as LC              # noqa: E402
import chain_two_substeps as T2      # noqa: E402
import pbl_compare as PC             # noqa: E402
import ghy_compare as GC             # noqa: E402
import tile_aggregate_ff as TA       # noqa: E402
from ffdump_reader import read_dump  # noqa: E402

_cache = {}


def _one(d, it):
    if (d, it) in _cache:
        return _cache[(d, it)]
    dd = f"{FF}/{d}"
    p = PC.load(f"{dd}/ffp_{it}.bin"); g = GC.load(f"{dd}/ffg_{it}.bin"); fft = TA.load(f"{dd}/fft_{it}.bin")
    n = len(p) // 2; ng = len(g) // 2; B = len(fft) // 2
    a4 = p[:n][p[:n, 2] == 4]; g1, g2 = g[:ng], g[ng:]
    blk = fft[:B]; _, patch, _ = TA.unpack(blk)
    lut = {(int(a), int(b)): k for k, (a, b) in enumerate(blk[:, :2])}
    idx = np.array([lut[(int(a), int(b))] for a, b in g1[:, :2]])
    trup = LC.infer_trup(g1, patch["dth1"][idx, 3], 900.0)
    din = read_dump(f"{dd}/ffa_{it}_c1_in.bin", 1)
    q1 = np.asarray(din["Q"])[g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1, 0]
    land = LC.land_substep(a4, g1, q1, trup)
    _cache[(d, it)] = (land, patch, idx, g1, g2, trup)
    return _cache[(d, it)]


@pytest.mark.parametrize("d,it", CASES)
def test_land_patch_matches_recorded(d, it):
    land, patch, idx, g1, g2, _ = _one(d, it)
    tol = dict(uflux1=1e-11, vflux1=1e-11, tsavg=1e-11, qsavg=1e-11)
    for k in ("uflux1", "vflux1", "tsavg", "qsavg"):
        ref = patch[k][idx, 3]
        assert np.abs(land["patch"][k] - ref).max() <= tol[k] * max(np.abs(ref).max(), 1.0), k
    # GHY-limited fields: bulk agrees tightly; the max is set by known threshold-crossing cells (D9 aruns/aeruns)
    for k, q99, rel_max in (("dth1", 1e-6, 1e-3), ("dq1", 1e-6, 5e-3)):
        e = np.abs(land["patch"][k] - patch[k][idx, 3])
        scale = np.abs(patch[k][idx, 3]).max()
        assert np.quantile(e, 0.99) < q99 * scale + 1e-12, k
        assert e.max() < rel_max * scale, k
    assert np.abs(patch["dth1"][idx, 3]).max() > 0.1      # non-vacuous


@pytest.mark.parametrize("d,it", CASES)
def test_trup_is_constant_across_substeps(d, it):
    """The reconstructed land TRUP_in_rad must be the same at substep 2 (a radiation input fixed for the step)."""
    _, patch1, idx1, g1, g2, trup1 = _one(d, it)
    fft = TA.load(f"{FF}/{d}/fft_{it}.bin"); B = len(fft) // 2
    blk2 = fft[B:]; _, patch2, _ = TA.unpack(blk2)
    lut = {(int(a), int(b)): k for k, (a, b) in enumerate(blk2[:, :2])}
    idx2 = np.array([lut[(int(a), int(b))] for a, b in g2[:, :2]])
    trup2 = LC.infer_trup(g2, patch2["dth1"][idx2, 3], 900.0)
    assert np.abs(trup1 - trup2).max() < 1e-9


@pytest.mark.parametrize("d,it", CASES)
def test_evap_max_and_frsat_outputs(d, it):
    land, _, _, g1, _, _ = _one(d, it)
    assert np.abs(land["evap_max_ij"] - g1[:, 286]).max() < 1e-5 * np.abs(g1[:, 286]).max()
    bad = np.abs(land["fr_sat_ij"] - g1[:, 287]) > 1e-9
    assert bad.mean() < 0.1     # a minority of cells sit on a threshold (documented D9-type sensitivity)


@pytest.mark.parametrize("d,it", CASES)
def test_two_substeps_with_land_chained(d, it):
    if (d, it, "2") not in _cache:
        _cache[(d, it, "2")] = T2.run_two_substeps(f"{FF}/{d}", it, land=True)
    rows, diag = _cache[(d, it, "2")]
    for k, r in rows.items():
        scale = r.get("fortran_change_rms", 1.0)
        if k == "pblht":
            assert r["max_abs"] < 5e-2 and r["rms"] < 1e-3
            continue
        assert r["rms"] < 2e-3 * scale, (k, r)          # bulk of the grid: 0.2% of the substep signal
        assert r["max_abs"] < 0.1 * scale, (k, r)       # worst cell: 10% (threshold-flip land cells, D9)
        assert scale > 1e-6
    assert diag["itype4.cm"] < 1e-11 and diag["profiles4"] < 1e-9
    assert diag["itype4.tg"] < 1e-3 and diag["itype4.qg_aver"] < 1e-4


def test_land_state_carried_over_four_substeps_stays_small():
    """D25: land state (GHY prognostic state, PBL land profiles/columns) carried by our code over two consecutive steps (4
    substeps), atmosphere/forcing/Ent recorded. Errors amplify at first then saturate far below the signal; the prognostic
    soil water drifts only slowly."""
    d, it = "dec01", 33552
    if not os.path.exists(f"{FF}/{d}/ffg_{it + 1}.bin") or not os.path.exists(f"{FF}/{d}/ffp_{it + 1}.bin"):
        pytest.skip("second-step dumps not available")
    rows = LC.run_land_multistep(f"{FF}/{d}", [it, it + 1])
    assert len(rows) == 4
    for r in rows:
        assert r["tbcs"][1] < 1e-3 and r["tbcs"][0] < 5e-2          # rms/max in K on a ~50 C field
        assert r["w"][1] < 1e-6 and r["w"][0] < 1e-4                # soil water (m), scale ~0.8
        assert r["alhg"][1] < 1e-4 * r["alhg"][2]
    assert rows[0]["tbcs"][1] < 1e-7                                 # first substep: recorded state, model error only
    assert rows[3]["w"][1] >= rows[0]["w"][1]                        # non-vacuous: carried error is visible and grows
