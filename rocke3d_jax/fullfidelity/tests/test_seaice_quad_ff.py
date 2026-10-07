"""D173: binary128 Ti/Ti2b emulation (seaice_quad_ff).  Data-dependent tests skip when the real dumps are absent.
Measured 2026-10-07: with quad Ti/Ti2b seaice_to_atmgrid gtemp/gtemp2/gtempr are bitwise on all 18 dumps (float64: 21780/26395/982 elements off),
ADDICE hsil bitwise on all (float64: 187 elements off)."""
import glob
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")

import seaice_quad_ff as Q  # noqa: E402


def test_matches_advsi_emulation_and_differs_from_float64():
    import advsi_ff as A
    import seaice_core_ff as S
    random.seed(1)
    nd = 0
    for _ in range(3000):
        e = -random.uniform(1e5, 3.34e5) * random.random(); si = random.uniform(0, 10); sn = random.uniform(0, 3); mi = random.uniform(1, 90)
        q = Q.ti2b_quad(e, si, sn, mi)
        assert q == A.ti2b_quad(e, si, sn, mi)
        nd += q != S.Ti2b(e, si, sn, mi)
    assert nd > 100      # the float64 shortcut really differs in the last bits


def test_s2ag_bitwise_with_quad():
    import seaice_to_atmgrid_compare as C
    from seaice_to_atmgrid_ff import seaice_to_atmgrid_cell
    fs = sorted(glob.glob(f"{FF}/nov26/ffz_s2ag_*.bin"))[:1]
    if not fs:
        pytest.skip("ffz_s2ag dumps absent")
    d = C.read_s2ag(fs[0])
    with Q.patched():
        for i in range(0, len(d["i"]), 7):
            o = seaice_to_atmgrid_cell(d["rsi"][i], d["snowi"][i], d["msi"][i], d["hsi1"][i], d["hsi2"][i], d["ssi1"][i], d["ssi2"][i], d["ssi3"][i], d["ssi4"][i])
            assert o["gtemp"] == d["gtemp"][i] and o["gtemp2"] == d["gtemp2"][i]


def test_addice_hsil_bitwise_with_quad():
    import addice_compare as AD
    fs = sorted(glob.glob(f"{FF}/*/ffn_*.bin"))
    if not fs:
        pytest.skip("ffn dumps absent")
    with Q.patched():
        for fn in fs:
            for r in AD.load(fn):
                o = AD.run_row(r)
                assert np.array_equal(np.array(o["hsil"]), r[24:28])
