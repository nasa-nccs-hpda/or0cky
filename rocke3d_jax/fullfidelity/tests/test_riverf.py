"""D167: tests of riverf_ff.py / riverf_loop.py.  Skipped when the dumps, restart or the river/topography input files are absent.
Bounds are the measured values (exact where the measurement was exact)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import riverf_ff as RF  # noqa: E402
import ocean_chain_io as C  # noqa: E402

D = C.FF_DEFAULT + '/nov26'
HAVE_DATA = RF.available() and all(os.path.exists(f'{D}/{f}') for f in ('ffo_state_33312.bin', 'ffs_33312.bin', 'ffs_33313.bin', 'ffc_cse_out_33313.bin')) \
    and os.path.exists(C.FF_DEFAULT + '/_pristine_restarts/fort1_nov26_itime33312.nc')
EXACT = RF.libm_mode() == 'intel libimf'


def test_get_dir_and_lonlat():
    assert RF.lonlat_to_ij(-177.5, -90.0) == (1, 1)
    assert RF.lonlat_to_ij(177.5, 90.0) == (72, 46)
    # (DI, DJ) = (I-ID, J-JD) table of LAKES.f get_dir
    assert RF.get_dir(10, 10, 9, 9) == 5 and RF.get_dir(10, 10, 9, 10) == 4 and RF.get_dir(10, 10, 11, 10) == 8
    assert RF.get_dir(10, 10, 10, 10) == 0 and RF.get_dir(10, 10, 10, 11) == 2 and RF.get_dir(10, 10, 10, 9) == 6
    assert RF.get_dir(1, 10, 72, 10) == 4      # date-line wrap: DI = 1-IM -> +1
    assert RF.get_dir(5, 1, 5, 2) == 2 and RF.get_dir(5, 46, 5, 45) == 6


@pytest.mark.skipif(not RF.available(), reason='river / topography input files missing')
def test_statics_and_rate():
    import ocean_step as O
    rs = RF.load_statics(O._DXYPO)
    assert rs['no_direction_land'] == []
    assert set(np.unique(rs['kdirec'])) <= set(range(10))
    assert (rs['rate'] > 0).sum() == 1075
    # non-vacuity: the REAL*4 literal DZDH1 = .00005 (not the double 5e-5) is what the model uses; the two rates differ at 2.5e-8
    r8 = RF.compute_rate(rs, RF.DZDH1_R8)
    m = rs['rate'] > 0
    rel = np.abs(r8[m] / rs['rate'][m] - 1).max()
    assert 1e-8 < rel < 1e-7


@pytest.fixture(scope='module')
def run2():
    import riverf_loop as RL
    rows, S = RL.run_free(2, 'riverf', log=lambda *_: None)
    return rows


@pytest.mark.skipif(not HAVE_DATA, reason='dumps/restart missing')
def test_flows_against_ffo_tag1(run2):
    f0, f1 = run2[0]['flows'], run2[1]['flows']
    assert f0['nonzero_real'] == 159 and f0['nonzero_ours'] == 159
    if EXACT:
        assert f0['bitwise_oflowo'] == f0['cells'] and f0['bitwise_oeflowo'] == f0['cells']
        assert f1['bitwise_oflowo'] == f1['cells']
        assert f1['maxabs_eo'] < 1e-17           # measured 6.9e-18 on a scale of 6.4e4 (one cell, one ulp)
    else:
        assert f0['maxabs_o'] < 1e-12 * f0['scale_o'] and f0['maxabs_eo'] < 1e-12 * f0['scale_eo']


@pytest.mark.skipif(not HAVE_DATA, reason='dumps/restart missing')
def test_lake_state_at_next_step_equals_real_records(run2):
    # step 1 entry state includes RIVERF of step 0: lake tg1, MWL, GML equal to the real tile records
    for n in ('lake.tg1', 'lake.mwl', 'lake.gml'):
        assert run2[0]['pre'][n][0] == 0.0 and run2[1]['pre'][n][0] == 0.0
