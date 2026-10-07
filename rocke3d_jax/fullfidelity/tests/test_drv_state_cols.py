"""D177 tests: COSZ1 from the clock, persistent PBL state carry, sea-ice thermal columns, land leftovers. Skip when the dumps are absent."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
cio = pytest.importorskip('clouds_condse_io')
FF = cio.FF_DEFAULT


def _have(d, it, name):
    return os.path.exists(f"{FF}/{d}/{name}_{it}.bin")


# ---------------------------------------------------------------- item 1: COSZ1
def test_orbpar_1850_matches_model_print():
    import drv_zenith as Z
    e, o, w = Z.orbpar(1850.0)
    # P2SAoM40.PRT of the real run prints 1.676429465128236E-002, 23.4592765450604, 280.326871404745
    assert abs(e - 1.676429465128236e-2) < 1e-17 and abs(o - 23.4592765450604) < 1e-12 and abs(w - 280.326871404745) < 1e-11


@pytest.mark.parametrize('d,it0,n', [('nov26_day', 33312, 54), ('dec01', 33552, 6), ('jan01', 17520, 6)])
def test_cosz1_vs_record(d, it0, n):
    import drv_zenith as Z
    if not os.path.exists(f"{FF}/{d}/ffa_step_{it0}_r.bin"):
        pytest.skip('dump absent')
    zn = Z.Zenith(use_imf=True)
    tol = 0.0 if Z._F.get('imf') else 2e-16          # bitwise with the Intel libimf (the real build's library), 1 ulp without it
    for it in range(it0, it0 + n):
        p = f"{FF}/{d}/ffa_step_{it}_r.bin"
        if not os.path.exists(p):
            continue
        r = np.array(cio.read_cse(p)['COSZ1'])
        assert np.abs(zn.cosz1(it) - r).max() <= tol


# ---------------------------------------------------------------- item 2: persistent PBL state
@pytest.mark.parametrize('d,date,it0,n', [('nov26_day', 'nov26', 33312, 54), ('dec01', 'dec01', 33552, 6), ('jan01', 'jan01', 17520, 6)])
def test_pbl_carry_bitwise(d, date, it0, n):
    import drv_state_cols as D
    if not _have(d, it0, 'ffp'):
        pytest.skip('dump absent')
    C = D.PBLCarry.from_restart(date)
    donors = 0
    for it in range(it0, it0 + n):
        if not _have(d, it, 'ffp'):
            continue
        p = np.fromfile(f"{FF}/{d}/ffp_{it}.bin", '>f8').reshape(-1, 154)
        h = len(p) // 2
        a, b = p[:h], p[h:]
        C.begin_step()
        r = D.diff_persistent(C.fill(a), a)
        assert max(v[0] for v in r.values()) == 0.0, (it, r)
        C.update(a, D.PBLCarry.out_from_records(a))
        r = D.diff_persistent(C.fill(b), b)
        assert max(v[0] for v in r.values()) == 0.0, (it, r)
        C.update(b, D.PBLCarry.out_from_records(b))


def test_pbl_carry_donor_rule_exercised():
    """The restart has 53 ocean rows with ipbl == 0 at nov26 step 0: they must come from a donor (ice, else land) and still match the record."""
    import drv_state_cols as D
    if not _have('nov26_day', 33312, 'ffp'):
        pytest.skip('dump absent')
    C = D.PBLCarry.from_restart('nov26')
    p = np.fromfile(f"{FF}/nov26_day/ffp_33312.bin", '>f8').reshape(-1, 154)
    a = p[:len(p) // 2]
    oc = a[a[:, 2] == 1]
    i, j = oc[:, 0].astype(int) - 1, oc[:, 1].astype(int) - 1
    missing = C.st[1]['ipbl'][j, i] == 0
    assert missing.sum() == 53
    raw = D.PBLCarry(C.st).fill(oc)                  # without begin_step(): the stale restart values
    assert np.abs(raw[missing][:, 66:74] - oc[missing][:, 66:74]).max() > 0
    C.begin_step()
    assert np.abs(C.fill(oc)[missing][:, 66:74] - oc[missing][:, 66:74]).max() == 0.0


# ---------------------------------------------------------------- item 3: sea-ice thermal columns
@pytest.mark.parametrize('d,it0,n', [('nov26_day', 33312, 54), ('dec01', 33552, 6), ('jan01', 17520, 6)])
def test_ice_thermal_columns_bitwise(d, it0, n):
    import drv_ice_cols as I
    import surface_tile_ff as ST
    if not _have(d, it0, 'ffs'):
        pytest.skip('dump absent')
    rows = 0
    for it in range(it0, it0 + n, 1 if n < 10 else 7):
        if not _have(d, it, 'ffs'):
            continue
        t = ST.load(f"{FF}/{d}/ffs_{it}.bin")
        h = len(t) // 2
        for part in (t[:h], t[h:]):
            r = I.compare(part)
            for k, v in r.items():
                assert v[0] == 0.0, (it, k, v)
                rows += v[2]
    assert rows > 0


# ---------------------------------------------------------------- item 4: land leftovers
def test_land_leftover_identities():
    import drv_land_cols as L
    n = 0
    for d, it0, nn in (('nov26_day', 33312, 54), ('dec01', 33552, 6), ('jan01', 17520, 6)):
        for it in range(it0, it0 + nn, 1 if nn < 10 else 5):
            if not (_have(d, it, 'ffp') and _have(d, it, 'ffg')):
                continue
            p = np.fromfile(f"{FF}/{d}/ffp_{it}.bin", '>f8').reshape(-1, 154)
            g = np.fromfile(f"{FF}/{d}/ffg_{it}.bin", '>f8').reshape(-1, 450)
            h, hg = len(p) // 2, len(g) // 2
            for pp, gg in ((p[:h], g[:hg]), (p[h:], g[hg:])):
                p4 = pp[pp[:, 2] == 4]
                assert np.array_equal(p4[:, :2], gg[:, :2])
                assert np.array_equal(gg[:, L.G_PRES], p4[:, L.P_PSURF])
                assert np.array_equal(gg[:, L.G_VS0], p4[:, L.P_OUT_WS0])
                assert np.array_equal(gg[:, L.G_GUSTI], p4[:, L.P_OUT_GUSTI])
                assert np.array_equal(gg[:, L.G_GUSTI], p4[:, L.P_GUSTI_IN])
                assert np.array_equal(gg[:, L.G_VS], p4[:, L.P_OUT_WS])
                assert np.array_equal(L.land_elhx_from_tg(p4[:, L.P_TG]), p4[:, L.P_ELHX])
                n += len(p4)
    if n == 0:
        pytest.skip('dump absent')


def test_land_ma1_is_atmosphere_ma1():
    import atm_step as A
    for d, it in (('nov26_day', 33312), ('nov26_day', 33340), ('dec01', 33552), ('jan01', 17521)):
        if not (_have(d, it, 'ffg') and os.path.exists(f"{FF}/{d}/ffa_step_{it}_r.bin")):
            continue
        R = A.Real(d, it)
        g = np.fromfile(f"{FF}/{d}/ffg_{it}.bin", '>f8').reshape(-1, 450)
        h = len(g) // 2
        ma = np.array(R.site('r')['MA'])[0]                          # (IM, JM)
        for gg in (g[:h], g[h:]):
            assert np.array_equal(ma[gg[:, 0].astype(int) - 1, gg[:, 1].astype(int) - 1], gg[:, 165])
        return
    pytest.skip('dump absent')


def test_land_substep_v2_equals_recorded_path():
    """The wired substep (leftovers + elhx computed) reproduces land_chain.land_substep with the recorded columns exactly (nov26 step 0, substep 1)."""
    import atm_step as A
    import land_chain as LC
    import tile_aggregate_ff as TA
    import drv_land_cols as L
    if not (_have('nov26_day', 33312, 'ffg') and os.path.exists(f"{FF}/nov26_day/ffa_step_33312_r.bin")):
        pytest.skip('dump absent')
    R = A.Real('nov26_day', 33312)
    rec = A.surface_records(R)
    pa, g1, blk1 = rec['pa'], rec['g1'], rec['blk1']
    p4 = pa[pa[:, 2] == 4]
    ma = np.array(R.site('r')['MA'])[0].T
    q1 = np.array(R.site('r')['Q'])[..., 0]
    gi, gj = g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1
    _, patch1, _ = TA.unpack(blk1)
    lut = {(int(a), int(b)): k for k, (a, b) in enumerate(blk1[:, :2])}
    idx = np.array([lut[(int(a), int(b))] for a, b in g1[:, :2]])
    trup = LC.infer_trup(g1, patch1['dth1'][idx, 3], 900.0)
    r0 = LC.land_substep(p4, g1, q1[gi, gj], trup, 900.0, None)
    r1 = L.land_substep_v2(p4, g1, q1[gi, gj], trup, ma, 900.0, None, set_elhx=True)
    for k in r0['patch']:
        assert np.array_equal(r0['patch'][k], r1['patch'][k]), k
