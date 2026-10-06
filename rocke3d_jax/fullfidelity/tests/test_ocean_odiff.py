"""D137/D138 tests: ported ODIFF vs ffo_state tag 12 -> 13, and calc_opfil2_coeffs vs ffo_opcoef.bin.
Skipped when the ff_data dumps are missing. Runtime ~1-2 min (one ODHORZ0 jit)."""
import os
import sys

import numpy as np
import pytest

FF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # fullfidelity/
sys.path.insert(0, FF)
import ocean_chain_io as C  # noqa: E402

DATA = C.FF_DEFAULT
FIRST = {'jan01': 17520, 'nov26': 33312, 'dec01': 33552}
pytestmark = pytest.mark.skipif(not all(os.path.exists(f'{DATA}/{d}/ffo_state_{i}.bin') for d, i in FIRST.items())
                                or not os.path.exists(f'{DATA}/jan01/ffo_opcoef.bin'), reason='ff_data ocean dumps missing')

_CACHE = {}


def _case(date):
    if date not in _CACHE:
        import ocean_step as O
        import ocean_odiff as D
        from ocean_step_compare import ctx_for
        d = f'{DATA}/{date}'
        ctx = ctx_for(d)
        sn = C.load_step(d, FIRST[date])
        dh = O.run_odhorz0(sn[12], ctx)['dh3d']
        _CACHE[date] = (sn, ctx, dh)
    return _CACHE[date]


def _run(date, **kw):
    import ocean_odiff as D
    sn, ctx, dh = _case(date)
    a = sn[12]
    return D.odiff(a['mo'], a['uo'], a['vo'], dh, ctx['lmu'], ctx['lmv'], **kw), sn


def _rel(g, r):
    return float(np.abs(g - r).max() / np.abs(r).max())


@pytest.mark.parametrize('date', list(FIRST))
def test_odiff_matches_dump(date):
    (uo, vo, vonp), sn = _run(date)
    b = sn[13]
    assert _rel(uo, b['uo']) < 5e-16 and _rel(vo, b['vo']) < 5e-16     # ~1 ulp of the field maximum
    assert np.array_equal(vonp, b['vonp'])
    assert (uo != b['uo']).mean() < 0.05 and (vo != b['vo']).mean() < 0.05   # >95% of values bitwise equal


def test_odiff_not_vacuous():
    (uo, vo, _), sn = _run('jan01')
    assert np.abs(sn[12]['uo'] - sn[13]['uo']).max() > 1e-3              # ODIFF really changes UO
    assert np.abs(sn[12]['vo'] - sn[13]['vo']).max() > 1e-3
    assert np.abs(uo - sn[12]['uo']).max() > 1e-3


def test_odiff_mutations_detected():
    import ocean_odiff as D
    base, sn = _run('jan01')
    ref = sn[13]
    (u1, v1, _), _ = _run('jan01', dtdiff=D.DTDIFF * 0.5)               # wrong time step
    assert _rel(u1, ref['uo']) > 1e-4
    (u2, v2, _), _ = _run('jan01', tridiag_x=D._tridiag_recip)           # wrong tridiagonal variant: still close, more bits off
    assert (u2 != ref['uo']).sum() > 2 * (base[0] != ref['uo']).sum()
    old = D.AKHMIN
    try:
        D.AKHMIN = old * 1.01; D._TAB.clear()                            # wrong minimum viscosity
        (u3, v3, _), _ = _run('jan01')
        assert _rel(u3, ref['uo']) > 1e-6
    finally:
        D.AKHMIN = old; D._TAB.clear()
    old3 = D.SQRT3_F32
    try:
        D.SQRT3_F32 = float(np.sqrt(3.0)); D._TAB.clear()                # double sqrt(3) instead of the single-precision literal
        (u4, v4, _), _ = _run('jan01')
        assert (u4 != ref['uo']).sum() > (base[0] != ref['uo']).sum()     # KHP changes only where the Munk term is active
    finally:
        D.SQRT3_F32 = old3; D._TAB.clear()


def test_odiff_stage_idle_and_fired():
    import ocean_step_odiff as OD
    sn, ctx, dh = _case('jan01')
    s = dict(sn[12]); s['dh3d'] = dh
    idle = OD.stage_odiff_ported(s, {'itime': 17521}, ctx)
    assert np.array_equal(idle['uo'], s['uo'])
    fired = OD.stage_odiff_ported(s, {'itime': 17520}, ctx)
    assert _rel(fired['uo'], sn[13]['uo']) < 5e-16 and np.array_equal(fired['vonp'], sn[13]['vonp'])


@pytest.mark.parametrize('date', list(FIRST))
def test_opfil2_coeffs(date):
    import ocean_opfil2_coeffs as P
    d = f'{DATA}/{date}'
    g = C.load_geom(d)
    ref = np.fromfile(d + '/ffo_opcoef.bin', dtype='>f8').astype(np.float64)
    v, sset = P.calc_opfil2_coeffs(g['lmu'])
    assert len(v) == len(ref)
    nred, nfft, nfil, nmn, nsm = [int(x) for x in v[:5]]
    m = np.ones(len(v), bool); m[5:5 + nsm] = sset
    o = 5 + nsm + nmn + nfft + 26 + 4 * nfil + 26
    assert np.array_equal(v[:o][m[:o]], ref[:o][m[:o]])                  # smooth, nmin and all segment/index tables exact
    assert np.abs(v[o:] - ref[o:]).max() < 2.5e-16                       # reduco: libm sin / fma level
    assert (v[o:] != ref[o:]).mean() < 0.02


def test_opfil2_coeffs_mutation():
    import ocean_opfil2_coeffs as P
    d = f'{DATA}/jan01'
    g = C.load_geom(d)
    ref = np.fromfile(d + '/ffo_opcoef.bin', dtype='>f8').astype(np.float64)
    old = P.HWID_MAX
    try:
        P.HWID_MAX = 14
        v, _ = P.calc_opfil2_coeffs(g['lmu'])
        assert len(v) != len(ref)
    finally:
        P.HWID_MAX = old
    lm = g['lmu'].copy(); lm[:, 5] = 0                                    # a polar-row land mask change must change the tables
    v, _ = P.calc_opfil2_coeffs(lm)
    assert len(v) != len(ref) or not np.array_equal(v, ref)
