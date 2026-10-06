"""Regression test for D136: GHY.f applies irrigation to the vegetated tile (giss_LSM/GHY.f:2230-2234, subtracted in flg/flhg at 1283, 1298,
1334, 1349); the ports used to drop it ("irrig is always 0" was false: IRRIGATION_ON is defined and 704 of 3012 cell-substeps of nov26 33312 +
dec01 33552 carry irrigation). With the term, ghy_jax (via land_chain.run_ghy) matches every output of every cell of the first ffg file of each
date to 1e-11 of field scale (measured 2e-13; before the fix aruns was off by up to 2.1e-3 of scale). Skips without the dumps."""
import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ghy_compare as GC
import land_chain as LC

FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
K = ("tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad")
FILES = {d: sorted(glob.glob(f"{FF}/{d}/ffg_*.bin"))[:1] for d in ("nov26", "dec01", "jan01")}

pytestmark = pytest.mark.skipif(not any(FILES.values()), reason="ff_data ffg dumps not present on this host")


@pytest.mark.parametrize("date", ["nov26", "dec01", "jan01"])
@pytest.mark.parametrize("half", [0, 1])
def test_all_outputs_match_real(date, half):
    if not FILES[date]:
        pytest.skip("no dumps for " + date)
    g = GC.load(FILES[date][0])
    n = len(g) // 2
    gg = g[half * n:(half + 1) * n]
    out, refs = LC.run_ghy(gg)
    for k in K:
        scale = max(float(np.abs(refs[k]).max()), 1e-30)
        assert float(np.abs(out[k] - refs[k]).max()) <= 1e-11 * scale, (date, half, k)


def test_irrigation_is_exercised():
    """Non-vacuity: the nov26 file has vegetated cells with nonzero irrigation (otherwise the fix could not matter)."""
    if not FILES["nov26"]:
        pytest.skip("no dumps")
    g = GC.load(FILES["nov26"][0])
    assert int((g[:, 147] != 0.0).sum()) > 50
