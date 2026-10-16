"""D166: surface loop with ADVSI (surface_loop_advsi.run_free), nov26, first two steps.  Skip if dumps/restart are missing.
Measured (2026-10-07): step-1 ice entry errors relative to scale: snow 1.7e-16, msi2 4.0e-10, ssi1 1.0e-16, tg1 1.9e-15, ptype 1.9e-8; tile sets
identical; ocean exit uo 6.4e-9, vo 3.5e-8 (absolute, ocean_step_chain_compare.errs metric).  Without ADVSI the same step gives
snow 9.7e-4, msi2 9.5e-4, ptype 1.1e-3, 14 mismatched ocean tiles, uo 2.3e-3.  Bounds below are ~3-10x the measured with-ADVSI values."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocean_chain_io as C  # noqa: E402

D = C.FF_DEFAULT + '/nov26'
HAVE = all(os.path.exists(f) for f in (D + '/ffo_state_33313.bin', D + '/ffy_33312_out.bin', C.FF_DEFAULT + '/advsi_dumps/nov26/ffadv_in_33312.bin',
                                       C.FF_DEFAULT + '/_pristine_restarts/fort1_nov26_itime33312.nc'))
pytestmark = pytest.mark.skipif(not HAVE, reason='dumps/restart missing')


@pytest.fixture(scope='module')
def rows():
    import surface_loop_advsi as LA
    return LA.run_free(2, with_advsi=True, log=lambda *a: None)


def test_ice_drift_removed(rows):
    e = rows[1]['entry']
    assert rows[1]['tileset'] == (0.0, 0.0)
    for k, b in (('ice.snow', 1e-14), ('ice.msi2', 4e-9), ('ice.ssi1', 1e-14), ('ice.tg1', 1e-13), ('ice.tg2', 1e-13), ('ice.ptype', 2e-7)):
        assert e[k] < b, (k, e[k])


def test_ocean_exit_close(rows):
    assert rows[1]['ocean_exit_abs']['uo'] < 1e-7 and rows[1]['ocean_exit_abs']['vo'] < 3e-7
    assert rows[1]['advsi_in_usi_equals_ffy']
