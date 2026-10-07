"""D170: combined surface loop (surface_loop_v2.py).  Skips when the nov26 dumps / restart / ADVSI dumps are absent.
Bounds are the measured values rounded up (see scoping/D170_SURFACE_LOOP_V2_ENTRY.md)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocean_chain_io as C  # noqa: E402

D = C.FF_DEFAULT + '/nov26'
HAVE = all(os.path.exists(p) for p in (f'{D}/ffo_state_33312.bin', f'{D}/ffs_33312.bin', f'{D}/ffy_33312_out.bin', f'{D}/ffl_33317.bin',
                                       C.FF_DEFAULT + '/advsi_dumps/nov26/ffadv_in_33312.bin',
                                       C.FF_DEFAULT + '/_pristine_restarts/fort1_nov26_itime33312.nc'))
pytestmark = pytest.mark.skipif(not HAVE, reason='dumps/restart missing')


def test_landice_chain_bitexact_without_precip_edifs():
    """Measured cause of the D164/D166 land-ice tg1 jump: PRECIP_LI's EDIFS must not be added to E1 (SURFACE_LANDICE.f zeroes igla%e1)."""
    import surface_loop as L
    import landice_tile_ff as LIT
    st = L.load_statics('nov26'); geo = st['geo']
    li = {k: v.copy() for k, v in L.load_restart('nov26')['landice'].items()}
    for add in (False, True):
        l = {k: v.copy() for k, v in li.items()}
        worst = 0.0
        for k in range(5):
            it = 33312 + k
            inp = L.replay_inputs('nov26', it)
            l, pli = L.precip_li(l, inp['prec'], inp['eprec'], st['flice'], geo)
            r = LIT.load(f'{D}/ffl_{it}.bin'); r = r[:len(r) // 2]
            i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
            worst = max(worst, np.abs(l['tlandi'][i, j, 0] - r[:, LIT.IN['tg1']]).max())
            e1 = inp['acc']['e1_li'] + (pli['e1'] if add else 0.0)
            l, _ = L.ground_li(l, inp['acc']['e0_li'], e1, inp['acc']['evap_li'], st['flice'], geo)
        if add:
            assert worst > 1.0            # non-vacuity: the old behaviour gives the 9 K error
        else:
            assert worst == 0.0


def test_free_loop_v2_three_steps():
    import surface_loop_v2 as V2
    rows, S, V = V2.run_free_v2(nsteps=3, log=lambda *a: None)
    for r in rows:
        assert r['landice']['landice.tg1'] == 0.0 and r['landice']['landice.tg2'] == 0.0
        assert r['tileset'] == (0.0, 0.0)
        assert r['entry']['ice.msi2'] < 1e-8 and r['entry']['ice.tg1'] < 1e-9
        assert r['lake']['lake.tg1'] < 1e-15 and r['lake']['lake.mwl'] < 1e-15
        assert r['ocean_exit']['uo'] < 1e-6 and r['ocean_exit']['vo'] < 1e-6 and r['ocean_exit']['g0m'] < 1e-9
        assert r['flows']['maxabs_o'] < 1e-15 and r['flows']['maxabs_eo'] < 1e-11 * r['flows']['scale_eo']
        assert r['dyn_vs_rec']['odmui'] < 1e-6 and r['dyn_vs_rec']['ustar'] < 1e-9
    assert rows[0]['advsi_entry_vs_real']['msi'] < 2e-6      # the single-cell GROUND_SI residual (cell 65,38), recorded in the ledger
