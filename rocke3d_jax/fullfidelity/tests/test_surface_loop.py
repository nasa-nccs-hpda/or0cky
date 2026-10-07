"""D164: tests of surface_loop.py (step 0 of nov26 from the real restart). Skip if the dumps are missing.
Bounds are the measured values (exact where the measurement was exact), not loosened."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocean_chain_io as C  # noqa: E402

D = C.FF_DEFAULT + '/nov26'
HAVE = all(os.path.exists(f'{D}/{f}') for f in ('ffo_state_33312.bin', 'ffs_33312.bin', 'ffm_33312.bin', 'ffc_cse_out_33312.bin')) \
    and os.path.exists(C.FF_DEFAULT + '/_pristine_restarts/fort1_nov26_itime33312.nc')
pytestmark = pytest.mark.skipif(not HAVE, reason='dumps/restart missing')


@pytest.fixture(scope='module')
def env():
    import surface_loop as L
    st = L.load_statics('nov26')
    st['ctx'] = L.make_ocean_ctx('nov26')
    S = L.init_surface_state('nov26', st=st)
    inp = L.replay_inputs('nov26', 33312)
    S1, mid = L.surface_pre(S, st, inp)
    return L, st, S, inp, S1, mid


def test_toc2sst_equals_restart_exports(env):
    L, st, S, *_ = env
    t = L.toc2sst(S['ocean'], st['ctx'])
    foc = st['geo']['is_ocean']
    import surface_loop as sl
    R = sl.load_restart('nov26')
    for k, rk in (('gtemp', 'asst'), ('sss', 'sss'), ('mlhc', 'mlhc')):
        assert np.array_equal(t[k][foc], R['atm'][rk][foc])


def test_pre_surface_state_matches_real_tile_records_exactly(env):
    L, st, S, inp, S1, mid = env
    cmp = L.compare_pre_records(S1, mid, st, 'nov26', 33312)
    for n in ('ocean.tg1', 'ocean.sss', 'lake.tg1', 'lake.mwl', 'lake.gml', 'ice.snow', 'ice.msi2', 'ice.ssi1', 'ice.flag_dsws',
              'landice.tg1', 'landice.tg2', 'landice.snow'):
        assert cmp[n][0] == 0.0, n
    assert cmp['ice.tg1'][0] < 1e-13 and cmp['ice.tg2'][0] < 1e-13
    assert cmp['tileset.ocean_mismatch'][0] == 0 and cmp['tileset.ice_mismatch'][0] == 0


def test_irrigation_reconstruction_is_needed(env):
    """non-vacuity: without IRRIG_LK the lake mass of the irrigated cells differs from the record."""
    L, st, S, inp, S1, mid = env
    assert mid['irr']['mwl_to_irrig'].max() > 1e6


def test_ocean_fluxes_from_our_state_equal_recorded(env):
    L, st, S, inp, S1, mid = env
    S2, post = L.surface_post(S1, st, inp, mid)
    sn = C.load_step(D, 33312)
    ocn = st['geo']['is_ocean'] & st['geo']['valid']
    for k, tol in (('oprec', 0), ('oeprec', 0), ('orsi', 0), ('omelti', 0), ('oemelti', 0), ('oe0', 0), ('oevapor', 0), ('osolarw', 0),
                   ('odmua', 0), ('odmva', 0)):
        ref = sn[0][k] if k in ('oprec', 'oeprec') else sn[1][k]
        assert np.abs(post['fx'][k] - ref)[ocn].max() <= tol, k
    for k, rel in (('orunosi', 1e-6), ('oerunosi', 1e-7), ('osrunosi', 1e-4), ('osolari', 1e-12)):
        assert np.abs(post['fx'][k] - sn[1][k])[ocn].max() <= rel * np.abs(sn[1][k]).max(), k
    from ocean_step_chain_compare import errs
    e = errs(S2['ocean'], sn[14])
    assert e['g0m'] < 1e-10 and e['mo'] < 1e-11 and e['uo'] < 1e-7 and e['vo'] < 1e-7
