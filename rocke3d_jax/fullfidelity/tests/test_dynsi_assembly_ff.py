"""D168: dynsi_ff (DYNSI input assembly + post-processing around VPICEDYN) against the recorded DYNSI boundary, nov26, 6 steps.
Inputs are the recorded/real ones (ffy ice state, ffs ice-tile DMUA, restart/ffo ocean exports); USI/VSI carried from OUR output.
Bounds are the measured values rounded up by ~3x (residual comes from the existing non-bitwise VPICEDYN, D87/D29), not loosened. Skip if data absent."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocean_chain_io as OC  # noqa: E402

FF = OC.FF_DEFAULT
D = FF + '/nov26'
RST = FF + '/_pristine_restarts/fort1_nov26_itime33312.nc'
HAVE = all(os.path.exists(f'{D}/{f}') for f in ('ffz_geom.bin', 'ffy_33312_in.bin', 'ffs_33312.bin', 'ffo_state_33312.bin', 'ffz_undocn_33312.bin')) and os.path.exists(RST)
pytestmark = pytest.mark.skipif(not HAVE, reason='dumps/restart missing')


def _run():
    import dynsi_ff as DF
    import dynsi_compare as C
    import dynsi_loop_ff as LF
    import underice_compare as UC
    from odhorz_ff import geomo_dyn_arrays
    from icedyn_geom_compare import read_geom
    import netCDF4 as nc
    G = DF.Geom(read_geom(f'{D}/ffz_geom.bin')['gfocean'])
    sinvo, sinpo = geomo_dyn_arrays()[:2]
    R = nc.Dataset(RST)
    r2 = lambda k: np.array(R.variables[k][:], float).T
    foc = G.focean > 0
    usi, vsi, ogz_sv = r2('usi'), r2('vsi'), r2('ogeoz_sv')
    rows = []
    for it in range(33312, 33318):
        sn = OC.load_step(D, it)
        din = C.read_dynsi_in(f'{D}/ffy_{it}_in.bin'); dout = C.read_dynsi_out(f'{D}/ffy_{it}_out.bin')
        s0 = sn[0]
        oga = np.where(foc, 0.5 * (s0['ogeoz'] + ogz_sv), 0.0)
        for j in (0, 45):
            if foc[0, j]:
                oga[1:, j] = oga[0, j]
        us, vs = DF.uosurf_from_ocean(s0['uo'][:, :, 0], s0['vo'][:, :, 0], sinpo, sinvo)
        dmua, dmva = LF.ice_tile_dmua('nov26', it, FF)
        import surface_tile_ff as ST
        t = ST.load(f'{D}/ffs_{it}.bin'); n = len(t) // 2; r = t[:n][t[:n][:, 2] == 2]
        snow = np.zeros((72, 46)); snow[r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1] = r[:, ST.IN['snow']]
        out = DF.dynsi(G, dmua, dmva, din['irsi'][1:73, 1:], din['imsi'][1:73, 1:], snow, oga, us, vs, usi, vsi)
        u = UC.read_undocn(f'{D}/ffz_undocn_{it}.bin'); ii, jj = u['i'].astype(int) - 1, u['j'].astype(int) - 1
        rows.append(dict(it=it, din=din, dout=dout, out=out, sn=sn, us=us, vs=vs, oga=oga, usi_in=usi, ustar=(out['ustar'][ii, jj], u['ustar'])))
        usi, vsi, ogz_sv = out['usi'], out['vsi'], s0['ogeoz']
    return rows, r2


@pytest.fixture(scope='module')
def res():
    return _run()


def test_assembly_inputs_exact(res):
    rows, r2 = res
    for r in rows:
        for k in ('gairx', 'gairy', 'pgfub', 'pgfvb', 'heff', 'area', 'amass', 'cor'):
            assert np.array_equal(r['din'][k], r['out']['inputs'][k]), (r['it'], k)
        for k in ('gwatx', 'gwaty'):
            assert np.abs(r['din'][k] - r['out']['inputs'][k]).max() < 3e-17, (r['it'], k)   # measured <= 6.9e-18 (sum order of the polar vector)


def test_uosurf_ogeoza_equal_restart_exports(res):
    rows, r2 = res
    r = rows[0]
    assert np.abs(r['us'] - r2('uosurf')).max() < 1e-16 and np.abs(r['vs'] - r2('vosurf')).max() < 1e-16
    assert np.array_equal(r['oga'], r2('ogeoza'))


def test_state_carry_and_outputs(res):
    rows, _ = res
    for r in rows:
        assert np.abs(r['din']['uice0'][2:74, 1:47] - r['usi_in']).max() < 3e-12        # measured <= 1.8e-12 (carried from our output)
        for k, scale in (('dmui', 160.0), ('dmvi', 176.0)):
            assert np.abs(r['out']['post'][k] - r['dout'][k][1:73, 1:47]).max() < 1e-9 * scale, (r['it'], k)   # measured <= 1.4e-10 / 3e-11
        assert np.abs(r['out']['post']['dmu'] - r['dout']['dmu']).max() < 1e-9 * 214      # includes the polar row
        assert np.abs(r['out']['usi'] - r['dout']['usi'][1:73, 1:47]).max() < 3e-12


def test_ocean_boundary_and_ustar(res):
    rows, _ = res
    for r in rows:
        o = r['sn'][1]
        assert np.abs(r['out']['odmui'] - o['odmui']).max() < 1e-9 * 160       # measured <= 1.6e-10
        assert np.abs(r['out']['odmvi'] - o['odmvi']).max() < 1e-9 * 176
        mine, ref = r['ustar']
        assert (np.abs(mine - ref) / ref).max() < 5e-11                         # measured <= 1.3e-11 relative, 482 cells
