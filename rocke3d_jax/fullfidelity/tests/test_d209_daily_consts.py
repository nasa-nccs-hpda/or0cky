"""D209: the computed day-boundary constants (MDRYA, ch4ox water mass, SNOAGE aging) that d209_day.py applies by default at the assembled step's day boundary,
against the real records of the nov26 -> nov27 boundary (it 33360).  Skips when the dump is absent."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
cio = pytest.importorskip('clouds_condse_io')
FF = cio.FF_DEFAULT
need = pytest.mark.skipif(not os.path.exists(f"{FF}/nov26_day/ffa_step_33360_a.bin"), reason='dump absent')


@need
def test_mdrya_and_ch4ox_dm_vs_records():
    import drv_daily as DD
    import atm_day_open_loop as OL
    assert DD.mdrya() == OL.daily_mdrya(FF, 'nov26')[0]                       # bitwise
    dm_rec, _ = OL.ch4ox_increment(FF, 'nov26_day', 33360)
    yr, mo = DD.date_of_itime(33360)
    assert (yr, mo) == (1950, 11)
    dm = DD.ch4ox_dm(yr, mo)
    assert np.abs(dm - dm_rec).max() <= 1e-12 * np.abs(dm_rec).max()          # the record value is a difference of two float arrays: rounding level


@need
def test_snoage_computed_is_bitwise_vs_entry_record():
    import drv_daily as DD
    r = DD.snoage_check()
    assert r['bitwise'] and r['n_changed_by_aging'] > 0


@need
def test_computed_boundary_on_real_end_state_matches_recorded_boundary():
    """DAILY_ATMDYN + ch4ox with the computed constants on the real end state of step 33359 versus the real start state of 33360 (same check as the recorded form)."""
    import drv_daily as DD
    r = DD.replay_check()
    for k in ('MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'P', 'PEK', 'Q'):
        assert r['computed']['fields'][k]['n_diff'] == 0, k      # bitwise vs the real step-start state, Q included (the recorded-DM form leaves 673 Q elements at 4e-22)
    assert r['recorded']['fields']['Q']['max_abs'] < 1e-18
