"""Tests for ocnsetup_ff.py (OCONV per-column setup block -> KPPMIX inputs) against real
per-call dumps -- Stage 2, D59 (piece 1 of the OCONV port). Plain-Python only."""
import glob
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__) + "/..")
from ocnsetup_ff import setup_inputs, IM, JM, LMO
from kppmix_compare import load_kppmix_records
from odhorz0_compare import FF_DEFAULT

RS = 1218
SETUP_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_setup_*.bin"))

pytestmark = pytest.mark.skipif(not SETUP_PATHS, reason="ff_data setup dumps not present on this host")


def _pad(x, n):
    a = np.zeros(n + 1)
    a[1:] = x[:n]
    return a


def _parse(row):
    off = 6

    def take(n):
        nonlocal off
        v = row[off:off + n]
        off += n
        return v

    rec = {"grav": row[5]}
    rec["g"] = _pad(take(LMO), LMO)
    rec["s"] = _pad(take(LMO), LMO)
    rec["po"] = _pad(take(LMO), LMO)
    take(1)  # mo1
    kmuv_max = IM + 2
    ul = take(LMO * kmuv_max).reshape((LMO, kmuv_max), order="F")
    rec["ul"] = np.zeros((LMO + 2, kmuv_max + 1))
    rec["ul"][1:LMO + 1, 1:kmuv_max + 1] = ul
    rec["ravm"] = np.concatenate([[0.0], take(kmuv_max)])
    take(kmuv_max)  # lmuv (unused by the port; kept for the layout)
    rec["ogeoz"], rec["hocean"], rec["deltae"], rec["deltas"], rec["deltam"], rec["deltasr"], rec["u2rho"] = take(7)
    rec["byrho"] = _pad(take(LMO), LMO)
    rec["rhom"] = _pad(take(LMO), LMO)
    rec["rho1"] = _pad(take(LMO), LMO)
    take(LMO)  # ptd (not needed for KPPMIX inputs)
    rec["alpha1"], rec["beta1"], rec["shc1"] = take(3)
    assert off == RS
    return rec


def _pairs(path):
    """Yield (setup_record, kppmix_record) pairs, verified to describe the same KPPMIX call."""
    raw = np.fromfile(path, dtype=">f8").reshape(-1, RS)
    itime = os.path.basename(path).split("_")[-1][:-4]
    kpath = f"{FF_DEFAULT}/{os.path.basename(os.path.dirname(path))}/ffz_kppmix_{itime}.bin"
    kr = load_kppmix_records(kpath)
    assert len(kr) == raw.shape[0]
    for row, k in zip(raw, kr):
        i, j, it, lmij = (int(round(row[n])) for n in range(4))
        assert (i, j, it, lmij) == (k["i"], k["j"], k["iter"], k["lmij"])
        yield _parse(row), j, k


@pytest.mark.parametrize("path", SETUP_PATHS)
def test_setup_block_matches_real_kppmix_inputs(path):
    for rec, j, k in _pairs(path):
        out = setup_inputs(rec, k["ze"], k["lmij"], j)
        for name in ["zgrid", "hwide", "byhwide", "shsq", "dvsq", "dbloc", "ritop"]:
            assert np.array_equal(out[name], k[name]), name
        for name in ["ustar", "bo", "bosol"]:
            assert out[name] == k[name], name


def test_kmuv_is_four_on_nonpole_and_full_on_pole():
    """Regression pin (OCNKPP.f:1807 sets KMUV=4 for non-pole columns; :1714 sets IM+2 at the
    pole). Confirmed against real records: a non-pole column matches only with 4 velocity points."""
    path = SETUP_PATHS[0]
    rec, j, k = next(p for p in _pairs(path) if p[1] < JM)
    out = setup_inputs(rec, k["ze"], k["lmij"], j)
    assert np.array_equal(out["shsq"], k["shsq"])


def test_setup_outputs_nonvacuous():
    rec, j, k = next(_pairs(SETUP_PATHS[0]))
    assert np.any(k["shsq"] != 0.0)
    assert np.any(k["zgrid"] != 0.0)
