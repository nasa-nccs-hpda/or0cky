"""D176 tests: radiation-derived tile columns and the surface side of the radiation packet (skipped when the nov26_day data are absent).

The bounds are the MEASURED values of the D176 runs (not the tolerances of an earlier gate); where a bound is looser than bitwise it says why.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import drv_radcols as DR  # noqa: E402

rs = pytest.importorskip('radiation_server')
FF = rs.FF
DAYDIR = f"{FF}/nov26_day"
LEGS6 = f"{DAYDIR}/ours_d176/lake_legs_replay6.npz"


def _need(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} absent")


def test_ghg_ca_matches_record_bitwise():
    _need(DR.GHG_FILE)
    _need(f"{DAYDIR}/ffg_33312.bin")
    import ghy_compare as GC
    g = GC.load(f"{DAYDIR}/ffg_33312.bin")
    assert np.all(g[:, DR.FFG['Ca']] == DR.ghg_ca())


def test_reset_algebra():
    rng = np.random.default_rng(1)
    F = rng.uniform(100, 500, (4, 72, 46))
    old = rng.uniform(0.1, 0.9, (72, 46))
    new = old + rng.uniform(-0.05, 0.05, (72, 46))
    m = np.ones((72, 46), bool)
    o = DR.reset_surf_fluxes_ice(F, old, new, m)
    up = new > old
    exp2 = np.where(up, (F[1] * old + F[0] * (new - old)) / new, F[1])
    exp1 = np.where(~up, (F[0] * (1 - old) + F[1] * ((1 - new) - (1 - old))) / (1 - new), F[0])
    assert np.array_equal(o[1], exp2) and np.array_equal(o[0], exp1)
    assert np.array_equal(o[2], F[2]) and np.array_equal(o[3], F[3])
    # general form with types (4,1): FSF1' = (FSF1*Fo + FSF4*(Fn-Fo))/Fn
    fo = rng.uniform(0.1, 0.5, (72, 46)); fn = fo + 0.01
    o2 = DR.reset_surf_fluxes(F, 4, 1, fo, fn, m)
    assert np.array_equal(o2[0], (F[0] * fo + F[3] * (fn - fo)) / fn)


@pytest.fixture(scope='module')
def day_rows():
    _need(f"{DAYDIR}/rsv_n26_33362_out.bin")
    _need(f"{rs.FF}/advsi_dumps/nov26/ffadv_out_33365.bin")
    import drv_radcols_day as D
    return D.run(54, lake='record', log=lambda *a: None)


def _agg(rows, name, steps=None):
    mx, nd, sc = 0.0, 0, 0.0
    for r in rows:
        if steps is not None and r['itime'] not in steps:
            continue
        v = r['cols'][name]
        mx, nd, sc = max(mx, v['max']), nd + v['ndiff'], max(sc, v['scale'])
    return mx, nd, sc


def test_columns_bitwise_ocean_land_landice_ent(day_rows):
    boundary = {33360, 33361}
    nob = set(r['itime'] for r in day_rows) - boundary
    for sub in ('', '_sub2'):
        for nm in ('srheat', 'trhr0', 'trup_in_rad'):
            assert _agg(day_rows, f'ffs{sub}.{nm}.ocean_domain')[:2] == (0.0, 0), nm
        assert _agg(day_rows, f'ffs{sub}.trhr0.lake')[:2] == (0.0, 0)
        assert _agg(day_rows, f'ffp{sub}.trhr0.ocean_domain_and_land')[:2] == (0.0, 0)
        assert _agg(day_rows, f'ffp{sub}.trhr0.lake')[:2] == (0.0, 0)
        for nm in ('srheat', 'flong', 'trup_in_rad'):
            assert _agg(day_rows, f'ffl{sub}.{nm}')[:2] == (0.0, 0), nm
        for nm in ('Ca', 'cosz1', 'vis_rad', 'dvis', 'trheat'):
            assert _agg(day_rows, f'ffg{sub}.{nm}')[:2] == (0.0, 0), nm
        # land solar heating and PBL qsol: bitwise except at the day boundary (daily_LAKE reset uses an approximate RSI_old, measured 4.6e-8)
        assert _agg(day_rows, f'ffg{sub}.srheat', nob)[:2] == (0.0, 0)
        assert _agg(day_rows, f'ffp{sub}.qsol.ocean_domain_and_land', nob)[:2] == (0.0, 0)
        assert _agg(day_rows, f'ffg{sub}.srheat', boundary)[0] <= 1e-7
    # COSZ1 input on radiation steps is the server output and equals the record
    for r in day_rows:
        assert r['cols']['cosz1_input_vs_record']['max'] == 0.0


def test_land_trup_equals_inferred_to_inversion_noise(day_rows):
    nob = set(r['itime'] for r in day_rows) - {33360, 33361}
    mx = _agg(day_rows, 'land.trup_vs_inferred', nob)[0]
    assert mx <= 3e-13               # measured 2.3e-13: the inferred value is a reconstruction with rounding noise; ours is TRSURF(4) itself
    assert _agg(day_rows, 'land.trup_vs_inferred', {33360, 33361})[0] <= 2e-7      # day boundary, RSI_old approximated (measured 1.3e-7)


def test_lake_composite_leg_is_only_approximate(day_rows):
    # lake cells with ONE composite leg between record entries (melt/form order lost): measured relative 4.7e-4 max; NOT bitwise.
    for nm in ('ffs.srheat.lake', 'ffs.trup_in_rad.lake'):
        mx, nd, sc = _agg(day_rows, nm)
        assert mx / sc <= 6e-4
        assert nd > 0


def test_lake_exact_with_our_replay_legs():
    _need(LEGS6)
    _need(f"{DAYDIR}/rsv_n26_33317_out.bin")
    import drv_radcols_day as D
    rows = D.run(6, lake=f"ours:{LEGS6}", log=lambda *a: None, daily_lake=False)
    for nm in ('ffs.srheat.lake', 'ffp.qsol.lake', 'ffs_sub2.srheat.lake'):
        assert _agg(rows, nm)[:2] == (0.0, 0), nm                 # bitwise
    mx, nd, sc = _agg(rows, 'ffs.trup_in_rad.lake')
    assert mx <= 6e-14 and nd <= 4          # measured: 4 values at 1 ulp (5.7e-14 K-flux units at scale 491) from step 33315, all others bitwise
    for nm in ('ffs.srheat.ocean_domain', 'ffs.trup_in_rad.ocean_domain'):
        assert _agg(rows, nm)[:2] == (0.0, 0)


def test_packet_surface_side_against_live_packets():
    _need(f"{DAYDIR}/rsv_n26_33362_in.bin")
    _need(f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc")
    import drv_radpacket_check as K
    import surface_loop as L
    import netCDF4 as nc
    import drv_radpacket as RP
    st = L.load_statics('nov26', L.FF)
    S0 = L.init_surface_state('nov26', L.FF, st=st)
    r0 = K.check_step(33312, S0, st)
    for k, v in r0.items():
        if k == 'GTEMPR2':
            assert v['max'] <= 1e-13 and v['ndiff'] <= 2         # seaice_to_atmgrid Ti: 2 cells at 1 ulp (5.7e-14 K)
        else:
            assert v['max'] == 0.0 and v['ndiff'] == 0, k
    d0 = nc.Dataset(f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc")
    sbv = np.transpose(np.array(d0.variables['snowbv'][:])[:, :, :2], (2, 1, 0))
    d0.close()
    import ghy_compare as GC
    for it in range(33313, 33363):
        g2, tbcs, w, nsn, dz, wsn, fr, ts, ws = K.land_from_record(it - 1)
        lf, sbv = RP.land_fields(g2, tbcs, w, nsn, dz, wsn, fr, sbv)
        if it in K.RAD_STEPS:
            res = K.check_step(it, None, st, snowbv_prev=None) if False else None
            lv = K.live(it)
            land = np.zeros((72, 46), bool)
            land[g2[:, 0].astype(int) - 1, g2[:, 1].astype(int) - 1] = True
            ok = ~RP.pole_mask()
            for k in ('GTEMPR4', 'BARESW'):
                assert np.array_equal(lf[k][land & ok], lv[k][land & ok]), (it, k)
            assert np.array_equal(lf['SNOWD'][:, land & ok], lv['SNOWD'][:, land & ok]), it
            assert np.abs(lf['FRSNOW'][:, land & ok] - lv['FRSNOW'][:, land & ok]).max() <= 2.3e-16, it       # measured: 1-2 ulp in <= 30 cells
            assert np.array_equal(ts[ok], lv['TSAVG'][ok]) and np.array_equal(ws[ok], lv['WSAVG'][ok]), it
