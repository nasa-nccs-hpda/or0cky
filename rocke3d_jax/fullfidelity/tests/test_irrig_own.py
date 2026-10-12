"""D225: irrigation from the IRRIG file and our lake state reproduces the recorded actual irrigation (ffg 147 water, 148 energy) of the first step BITWISE
(all 753 land rows) and the potential demand reproduces the record at later steps (unlimited cells), including both day boundaries.  Skipped without records."""
import os

import numpy as np
import pytest

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day2'
HAVE = os.path.exists(f'{FF}/ffg_33312.bin')


def _pot(q12, it):
    import ent_daily_ff as D
    return np.maximum(D.lai_linm2m(q12, (it // 48) % 365 + 1), 0.0).T


def _q12():
    import netCDF4 as nc
    import jax_static as JST
    q = np.array(nc.Dataset(os.path.join(JST.PROD, 'Irrig72x46_1848to2100_FixedFuture_v3.nc')).variables['irrigation_per_m2'][:], dtype=np.float64)
    return q[(1850 - 1848) * 12:(1850 - 1848) * 12 + 12]


@pytest.mark.skipif(not HAVE, reason='records not available')
def test_first_step_actual_irrigation_bitwise():
    import atm_step as A
    import jax_static as JST
    import surface_loop as SL
    GC = A._surf_mods()['GC']
    st, SS = SL.load_statics('nov26'), SL.init_surface_state('nov26')
    axyp, fearth, fland = (np.asarray(st[k]) for k in ('axyp', 'fearth', 'fland'))
    fv = JST.ent_static('nov26')['fv']
    fv = np.where(fv < 1e-6, 0.0, fv)
    pot = _pot(_q12(), 33312)
    pc = np.where((pot > 1e-30) & (fearth > 1e-30) & (fv > 1e-30), pot, 0.0)
    lake = {k: np.array(SS['lake'][k], copy=True) for k in ('mwl', 'gml', 'tlake', 'mldlk')}
    _, irr = SL.irrig_lk(lake, pc, fland, axyp, st['geo'])
    m2kg = SL.RHOW * axyp
    mi = irr['mwl_to_irrig']
    act = np.where((mi >= pc * m2kg * SL.DTSRC) & (pc > 0), pc, mi / (m2kg * SL.DTSRC))
    en = irr['gml_to_irrig'] * SL.RHOW / (m2kg * SL.DTSRC)
    g = np.asarray(GC.load(f'{FF}/ffg_33312.bin'))
    g = g[:len(g) // 2]
    i, j = g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1
    assert np.array_equal(act[i, j] / fearth[i, j], g[:, 147])
    assert np.array_equal(en[i, j] / fearth[i, j], g[:, 148])


@pytest.mark.skipif(not HAVE, reason='records not available')
@pytest.mark.parametrize('it', [33359, 33360, 33408])
def test_potential_matches_record_across_day_boundaries(it):
    import atm_step as A
    import clouds_condse_io as cio
    GC = A._surf_mods()['GC']
    g = np.asarray(GC.load(f'{FF}/ffg_{it}.bin'))
    g = g[:len(g) // 2]
    fe = np.asarray(cio.read_cse(f'{FF}/ffc_cse_in_{it}.bin')['FEARTH'])
    i, j = g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1
    rec = g[:, 147] * fe[i, j]
    pot = _pot(_q12(), it)[i, j]
    m = rec > 0
    assert int(m.sum()) == 176
    assert not np.any(rec[m] > pot[m] * (1 + 1e-14))          # actual never exceeds the potential
    assert int((rec[m] == pot[m]).sum()) >= 150                # the others are lake-limited
