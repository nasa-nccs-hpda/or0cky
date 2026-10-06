"""Regression test for D135: land_chain.run_ghy must condition the precipitation forcing like ghy_ref.GhyColumn (pr>=0, 0<=prs<=pr,
htprs=htpr/pr*prs). Before the fix the JAX GHY got htprs=0 and the snowfall cell (62,34) of nov26 33312 (both substeps) was wrong
(dripw[veg] 0 instead of 6.7e-9; aruns 3.0e-4 vs real 3.9e-4). Skips without the ffg dumps."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ghy_compare as GC
import land_chain as LC

FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
P = f"{FF}/nov26/ffg_33312.bin"
K = ("tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad")

pytestmark = pytest.mark.skipif(not os.path.exists(P), reason="ff_data ffg dumps not present on this host")


@pytest.mark.parametrize("half", [1, 2])
def test_snowfall_cell_matches_real(half):
    g = GC.load(P)
    n = len(g) // 2
    g = g[:n] if half == 1 else g[n:]
    out, refs = LC.run_ghy(g)
    c = int(np.where((g[:, 0] == 62) & (g[:, 1] == 34))[0][0])
    for k in K:
        scale = max(abs(refs[k][c]), 1e-30)
        assert abs(out[k][c] - refs[k][c]) <= 1e-10 * scale + 1e-18, (half, k, out[k][c], refs[k][c])

