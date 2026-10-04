"""Tests for kvinit_ff.py (OCNKPP.f KVINIT) against real per-step dumps -- Stage 2, D58."""
import glob

import numpy as np
import pytest

from kvinit_ff import kvinit
from odhorz0_compare import FF_DEFAULT

IM, JM = 72, 46
KVINIT_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_kvinit_*.bin"))

pytestmark = pytest.mark.skipif(not KVINIT_PATHS, reason="ff_data kvinit dumps not present on this host")


def _load(path):
    raw = np.fromfile(path, dtype=">f8")
    lsrpd = int(round(raw[1]))
    off = 2

    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        a = raw[off:off + n].reshape(shape, order="F")
        off += n
        return a

    g0m = take((IM, JM, lsrpd))
    g0m1 = take((IM, JM, lsrpd))
    rec = {"g0m": g0m, "g0m1": g0m1, "lsrpd": lsrpd}
    for name in ["s0m", "mo", "gxmo", "gymo", "sxmo", "symo", "uo", "vo", "uod", "vod"]:
        rec[name] = take((IM, JM))
        rec[name + "1"] = take((IM, JM))
    assert off == raw.size
    return rec


@pytest.mark.parametrize("path", KVINIT_PATHS)
def test_kvinit_matches_real_fortran(path):
    r = _load(path)
    # Source fields as the 3-D arrays kvinit expects; the 2-D fields only need layer 1 populated.
    lsrpd = r["lsrpd"]
    def field3(name):
        a = np.zeros((IM, JM, lsrpd))
        a[:, :, 0] = r[name]
        return a
    g0m = r["g0m"]
    snap = kvinit(g0m, field3("s0m"), field3("mo"), field3("gxmo"), field3("gymo"),
                  field3("sxmo"), field3("symo"), field3("uo"), field3("vo"),
                  field3("uod"), field3("vod"), lsrpd)
    assert np.array_equal(snap["g0m1"], r["g0m1"])
    dumped_for = {"s0m1": "s0m1", "mo1": "mo1", "gxm1": "gxmo1", "gym1": "gymo1",
                  "sxm1": "sxmo1", "sym1": "symo1", "uo1": "uo1", "vo1": "vo1",
                  "uod1": "uod1", "vod1": "vod1"}
    for port_name, dump_name in dumped_for.items():
        assert np.array_equal(snap[port_name], r[dump_name]), port_name


def test_kvinit_snapshot_is_nonvacuous():
    """Regression pin: the real fields are not all zero, so an exact-equality match is not
    trivially satisfied by empty arrays."""
    r = _load(KVINIT_PATHS[0])
    assert np.any(r["g0m1"] != 0.0)
    assert np.any(r["mo1"] != 0.0)
