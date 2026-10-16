"""D166: advsi_ff.advsi against the real ADVSI entry/exit dumps (ff_data/advsi_dumps/<date>/ffadv_{in,out}_<it>.bin).  Skip when absent.
Measured result: every field of every dumped call (nov26 54 steps, dec01 48, jan01 48 where present) is bitwise equal: tolerance is exactly 0."""
import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import advsi_compare as AC  # noqa: E402
import advsi_ff as A  # noqa: E402

DIRS = sorted(glob.glob(AC.FF + '/advsi_dumps/*'))
pytestmark = pytest.mark.skipif(not DIRS, reason='ADVSI dumps missing')


def _calls():
    out = []
    for dd in DIRS:
        its = sorted(int(os.path.basename(p)[9:-4]) for p in glob.glob(f'{dd}/ffadv_in_*.bin'))
        out += [(dd, it) for it in its[::6]]       # every 6th call keeps the test fast; the full set is advsi_compare.py
    return out


@pytest.mark.parametrize('dd,it', _calls())
def test_advsi_bitwise(dd, it):
    res, _ = AC.compare_step(f'{dd}/ffadv_in_{it}.bin', f'{dd}/ffadv_out_{it}.bin')
    bad = {k: v for k, v in res.items() if v[1] != 0}
    assert not bad, bad


def test_non_vacuous_and_branches():
    dd = DIRS[0]
    its = sorted(int(os.path.basename(p)[9:-4]) for p in glob.glob(f'{dd}/ffadv_in_*.bin'))[:6]
    stats = {}
    dr = 0.0
    for it in its:
        d, o = A.read_dump(f'{dd}/ffadv_in_{it}.bin', f'{dd}/ffadv_out_{it}.bin')
        st = {k: d[k] for k in AC.STATE}
        new, _ = A.advsi(st, d['ausi'], d['avsi'], d['focean'], d['geo'], stats=stats)
        dr = max(dr, float(np.abs(new['rsi'] - d['rsi']).max()))
    assert dr > 1e-4                      # the call really moves the ice
    assert stats.get('ns260', 0) > 0 and stats.get('ew560', 0) > 0 and stats.get('ns_crunch', 0) > 0


def test_connect_formula():
    d, _ = A.read_dump(f'{DIRS[0]}/ffadv_in_{sorted(glob.glob(DIRS[0] + "/ffadv_in_*.bin"))[0].split("_")[-1][:-4]}.bin')
    assert np.array_equal(A.connect_from_focean(d['focean']), d['connect'])


def test_south_pole_guard():
    d, _ = A.read_dump(sorted(glob.glob(DIRS[0] + '/ffadv_in_*.bin'))[0])
    foc = d['focean'].copy()
    foc[:, 0] = 1.0
    st = {k: d[k] for k in AC.STATE}
    with pytest.raises(NotImplementedError):
        A.advsi(st, d['ausi'], d['avsi'], foc, d['geo'])
