"""D158: cells with ffnit >= 12 sub-iterations. The ffg record holds <= 11 Ent/dts entries, so the old ports advanced those cells over < dt.
ghy_ref_nit.advnc_full runs the real time loop (gdtm -> dts); these tests check it against the real record."""
import glob, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DAY = sorted(glob.glob(f"{FF}/nov26_day/ffg_*.bin"))
pytestmark = pytest.mark.skipif(len(DAY) < 54, reason="nov26_day ffg dumps not available")

import ghy_compare as GC       # noqa: E402
import ghy_ref_nit as N        # noqa: E402

# (itime, 0-based record index, ffnit) of every cell with ffnit >= 12 in nov26_day, jan01, dec01, nov26 (15 cells, found by scanning all ffg files)
STIFF = [(33319, 837), (33320, 799), (33321, 46), (33329, 112), (33329, 865), (33330, 865), (33331, 863), (33332, 863),
         (33333, 863), (33335, 864), (33336, 111), (33336, 864), (33337, 111), (33338, 864), (33339, 864)]


def _rec(itime, i):
    return GC.load(f"{FF}/nov26_day/ffg_{itime}.bin")[i]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-6)


@pytest.mark.parametrize("itime,i", STIFF)
def test_stiff_cell_matches_real_with_full_time_loop(itime, i):
    rec = _rec(itime, i)
    ffnit = int(round(rec[289]))
    assert ffnit >= 12
    col, refs, info = N.run_cell_full(rec, use_recorded_dts=False)     # dts from gdtm, NOT the record, and nit not imposed
    assert info['nit'] == ffnit
    assert max(abs(a - b) for a, b in zip(info['dts'], info['dts_rec']) if b is not None) < 1e-9   # gdtm reproduces the 11 recorded dts
    assert _rel(col.ashg, refs['ashg']) < 1e-5
    assert abs(col.tbcs - refs['tbcs']) < 1e-5
    assert _rel(col.alhg, refs['alhg']) < 1e-4
    assert _rel(col.aevap, refs['aevap']) < 1e-4


@pytest.mark.parametrize("itime,i", STIFF[:4])
def test_old_path_fails_non_vacuous(itime, i):
    col, refs = GC.run_cell(_rec(itime, i))        # the 11-record path: dt = sum(recorded dts) < 900
    assert _rel(col.ashg, refs['ashg']) > 1e-2 or abs(col.tbcs - refs['tbcs']) > 1e-2


def test_previously_matching_cells_still_match():
    """Every cell with ffnit 8..11 of five day files (gdtm-derived dts, not the recorded ones) still matches the real record."""
    n = 0
    for itime in (33312, 33321, 33329, 33336, 33337):
        rec = GC.load(f"{FF}/nov26_day/ffg_{itime}.bin")
        for i in np.where(np.round(rec[:, 289]) >= 8)[0]:
            if int(round(rec[i, 289])) > 11:
                continue
            col, refs, info = N.run_cell_full(rec[i], use_recorded_dts=False)
            assert info['nit'] == int(round(rec[i, 289]))
            assert _rel(col.ashg, refs['ashg']) < 1e-6, (itime, i)
            assert abs(col.tbcs - refs['tbcs']) < 1e-6, (itime, i)
            n += 1
    assert n >= 10
