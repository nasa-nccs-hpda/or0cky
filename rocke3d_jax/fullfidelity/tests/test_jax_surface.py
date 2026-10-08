"""D188: quick tests of jax_surface (fixed-shape SURFACE stage).  Unit tests of the ported glue and of the template builder run in seconds on the
real nov26 step-0 records; the whole-stage C1 test (about 4 min: it compiles the stage and runs the NumPy stage) runs only with D188_FULL=1.
Run pinned: taskset -c 3-5 env OMP_NUM_THREADS=1 pytest tests/test_jax_surface.py"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import atm_step as A  # noqa: E402
import jax_surface as JS  # noqa: E402

DATE, IT = 'nov26', 33312
pytestmark = pytest.mark.skipif(not A.have_dumps(DATE, IT), reason='real nov26 step-0 dumps not available')


@pytest.fixture(scope='module')
def world():
    R = A.Real(DATE, IT)
    rec = A.surface_records(R)
    S = A.real_state_at(R, 'surface', None)
    A._native(S)
    tpl, host = JS.build_template(rec)
    return dict(R=R, rec=rec, S=S, tpl=tpl, host=host)


def test_constants_equal_numpy_stage():
    assert JS.PBL_COLS == A.PBL_COLS and JS.PBL_GUSTI_OUT == A.PBL_GUSTI_OUT and JS.FLOAT4_4375 == A.FLOAT4_4375


def test_template_slots_match_record_rowsets(world):
    h, rec = world['host'], world['rec']
    pa = rec['pa']
    assert h['Nl'] == (pa[:, 2] == 3).sum() and h['Ne'] == (pa[:, 2] == 4).sum()
    v = h['wvalid'][0]
    assert v[0].sum() == (pa[:, 2] == 1).sum() and v[1].sum() == (pa[:, 2] == 2).sum()
    assert h['Nw'] == len({(a, b) for a, b in pa[pa[:, 2] <= 2][:, :2]})
    assert len(h['bcells']) == 3170 and h['max_substeps'] >= 11
    # the slot arrays have fixed shapes independent of which tiles exist
    t = world['tpl']['ns'][0]
    assert t['pw'].shape == (2 * h['Nw'], 154) and t['tw'].shape == (2 * h['Nw'], 90) and t['pl'].shape == (h['Nl'], 154)
    assert t['pe'].shape == (h['Ne'], 154) and t['ftype'].shape == (72 * 46, 4) and t['wvalid'].shape == (2, h['Nw'])
    # valid slots hold exactly the recorded rows
    ocean_rows = pa[pa[:, 2] == 1]
    cells = (ocean_rows[:, 1].astype(int) - 1) * 72 + ocean_rows[:, 0].astype(int) - 1
    slot = np.searchsorted(h['wcells'], cells)
    assert np.array_equal(np.asarray(t['pw'])[slot], ocean_rows)


def test_pow4_matches_numpy_power():
    x = np.random.default_rng(0).uniform(200.0, 330.0, 4000)
    assert np.array_equal(np.asarray(JS._pow4(jnp.asarray(x))), x ** 4)


def test_first_layer_update_equals_numpy(world):
    S = world['S']
    dth1 = np.random.default_rng(1).normal(0, 1e-3, (72, 46))
    dq1 = np.random.default_rng(2).normal(0, 1e-7, (72, 46))
    ref = A.first_layer_update(S['TMOM'], S['QMOM'], dth1, dq1, S['T'][:, :, 0], S['Q'][:, :, 0], S['PK'][0])
    got = JS.first_layer_update(jnp.asarray(S['TMOM']), jnp.asarray(S['QMOM']), jnp.asarray(dth1), jnp.asarray(dq1), jnp.asarray(S['T'][:, :, 0]),
                                jnp.asarray(S['Q'][:, :, 0]), jnp.asarray(S['PK'][0]))
    for a, b in zip(ref, got):
        assert np.array_equal(a, np.asarray(b))


def test_cell_inputs_and_overrides_equal_numpy(world):
    S, rec, tpl, h = world['S'], world['rec'], world['tpl'], world['host']
    atm = A.atm_layout(S, None)
    pb = rec['pb']
    cor = np.zeros(atm['T'].shape[:2])
    cor[pb[:, 1].astype(int) - 1, pb[:, 0].astype(int) - 1] = pb[:, 36]
    ref = A.cell_inputs(S, atm, None, S['USTARPBL'].T, S['LMONINPBL'].T, S['PBLHT'].T, S['DCLEV'].T, cor, S['T1AA'].T, S['U1AA'].T, S['V1AA'].T)
    Sd = JS.state_from_numpy(S)
    got = JS.cell_inputs(Sd, JS.atm_layout(Sd), jnp.asarray(cor))
    for k, v in ref.items():
        assert np.array_equal(v, np.asarray(got[k])), k
    # override_pbl on the land rows equals the NumPy override of the same rows
    p4 = rec['pa'][rec['pa'][:, 2] == 4]
    ref4 = A.override_pbl(p4, ref)
    got4 = JS.override_pbl(jnp.asarray(p4), got, jnp.asarray(h['ej']), jnp.asarray(h['ei']))
    assert np.array_equal(ref4, np.asarray(got4))


@pytest.mark.skipif(os.environ.get('D188_FULL') != '1', reason='whole-stage C1 (about 4 min); set D188_FULL=1')
def test_whole_stage_c1_bitwise_vs_numpy(world):
    R, S, rec, tpl, h = world['R'], world['S'], world['rec'], world['tpl'], world['host']
    S2 = A.real_state_at(R, 'surface', None)
    A._native(S2)
    A.stage_surface(S2, R, None, rec=rec, land_mode='ghy')
    out, _ = JS.make_stage(h)(JS.state_from_numpy(S), tpl)
    for k in ('T', 'Q', 'U', 'V', 'UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'TSAVG', 'QSAVG',
              'USTARPBL', 'LMONINPBL', 'TMOM', 'QMOM'):
        assert np.array_equal(np.asarray(out[k]), S2[k]), k


# ---- D195: the GHY substep schedule of cells with recorded ffnit > 11 follows OUR precipitation (as the NumPy chain), not the recorded one
DAYDIR, IT_NIT = 'nov26_day', 33319          # step 7 of the nov26 day: first step with a cell of ffnit 12, cell (64,19)


def test_nit_rule_ffnit_gt_11_uses_our_precipitation():
    import ghy_advnc_test as AT
    import os.path as op
    if not op.exists(f'{A.FF}/{DAYDIR}/ffg_{IT_NIT}.bin') and not op.exists(f'{A.FF}/{DAYDIR}/ffg_{IT_NIT}'):
        pytest.skip('nov26_day step 7 records not available')
    rec = A.surface_records(A.Real(DAYDIR, IT_NIT))
    tpl, host = JS.build_template(rec)
    hit = host['nit_hit']
    assert len(hit[0]) + len(hit[1]) >= 1, 'no ffnit > 11 cell in this record (the test needs one)'
    k = 0 if len(hit[0]) else 1
    idx = hit[k]
    g0 = host['nit_rows'][k]
    assert (np.round(g0[idx, 289]).astype(int) > 11).all()
    # grid precipitation in the units of PREC/EPREC/PRECSS; recorded = grid values reproduce the record exactly
    P, E, Ps = (np.zeros((72, 46)) for _ in range(3))
    i, j = g0[:, 0].astype(int) - 1, g0[:, 1].astype(int) - 1
    P[i, j], E[i, j], Ps[i, j] = g0[:, 143] * 1800.0 * 1000.0, g0[:, 144] * 1800.0, g0[:, 145] * 1800.0 * 1000.0
    JS.NIT_LOG.clear()
    t_same, ncell, _ = JS.nit_rebuild(tpl, host, jnp.asarray(P), jnp.asarray(E), jnp.asarray(Ps))
    assert ncell >= 1
    # (1) with our precipitation == the record the schedule equals the recorded-row schedule
    for key in JS.NIT_KEYS:
        a, b = np.asarray(t_same['ghy'][k][key]), np.asarray(tpl['ghy'][k][key])
        assert np.allclose(a[idx], b[idx], rtol=0, atol=1e-9), key
    # (2) with a different precipitation the schedule of exactly the ffnit > 11 cells changes, and equals the NumPy build_batch of the row with our precipitation
    P2, E2 = P * (1 + 1e-3) + 1e-9, E * (1 + 1e-3)
    t_new, ncell2, _ = JS.nit_rebuild(tpl, host, jnp.asarray(P2), jnp.asarray(E2), jnp.asarray(Ps))
    g = np.array(g0, dtype=np.float64, copy=True)
    g[:, 143], g[:, 144], g[:, 145] = P2[i, j] / 1.8e6, E2[i, j] / 1800.0, Ps[i, j] / 1.8e6
    ref = AT.build_batch(g)
    edts_ref, nsub_ref = np.asarray(ref[3]), np.asarray(ref[7])
    got_e, got_n = np.asarray(t_new['ghy'][k]['edts']), np.asarray(t_new['ghy'][k]['nsub'])
    w = got_e.shape[1]
    ref_pad = np.pad(edts_ref, ((0, 0), (0, w - edts_ref.shape[1])))
    assert np.array_equal(got_e[idx], ref_pad[idx]) and np.array_equal(got_n[idx], nsub_ref[idx])
    assert not np.array_equal(got_e[idx], np.asarray(tpl['ghy'][k]['edts'])[idx]), 'non-vacuity: our precipitation must change the schedule'
    other = np.setdiff1d(np.arange(got_e.shape[0]), idx)
    assert np.array_equal(got_e[other], np.asarray(tpl['ghy'][k]['edts'])[other])      # all other cells keep the recorded schedule
    # (2b) the explicit non-strict option (iteration count reported, not asserted) gives the same schedule when the count matches the record
    JS.NIT_MISMATCH.clear()
    t_free, _, _ = JS.nit_rebuild(tpl, host, jnp.asarray(P2), jnp.asarray(E2), jnp.asarray(Ps), strict=False)
    assert np.array_equal(np.asarray(t_free['ghy'][k]['edts']), got_e) and np.array_equal(np.asarray(t_free['ghy'][k]['nsub']), got_n)
    assert JS.NIT_MISMATCH == []
    # (3) no ffnit > 11 cell: no-op and no device->host read
    host0 = dict(host, nit_hit=[np.zeros(0, int), np.zeros(0, int)])
    t0, n0, d0 = JS.nit_rebuild(tpl, host0, None, None, None)
    assert t0 is tpl and n0 == 0 and d0 == 0
