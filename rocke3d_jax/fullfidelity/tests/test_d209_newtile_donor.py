"""D209: the loadbl donor rule for tiles that APPEAR (PBL_DRV.f:1098-1237; drv_state_cols.PBLCarry.begin_step) is exercised by the real nov26 day and is needed there.
Skips when the dump is absent.  Counts: (a) tile appearances in the 54 steps of the ffp records; (b) with the donor rule the carried PBL state equals the record at EVERY
appearing tile (bitwise, the 8 columns groups u v t q e cm ch cq); (c) without the donor rule (stale state kept, only ipbl cleared) the appearing tiles do NOT match."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
cio = pytest.importorskip('clouds_condse_io')
FF = cio.FF_DEFAULT
D, IT0, N = 'nov26_day', 33312, 54


def _rows(it):
    p = np.fromfile(f"{FF}/{D}/ffp_{it}.bin", '>f8').reshape(-1, 154)
    return p[:len(p) // 2], p[len(p) // 2:]


def _run(use_donor):
    import drv_state_cols as S
    C = S.PBLCarry.from_restart('nov26')
    prev = None
    n_new = {1: 0, 2: 0, 3: 0, 4: 0}
    bad_new = {1: 0, 2: 0, 3: 0, 4: 0}
    bad_old = 0
    for it in range(IT0, IT0 + N):
        a, b = _rows(it)
        if use_donor:
            C.begin_step()
        else:
            for t in C.st:
                C.st[t]['ipbl'][:] = 0
        built = C.fill(a)
        keys = {(int(r[0]), int(r[1]), int(r[2])) for r in a}
        for r, br in zip(a, built):
            k = (int(r[0]), int(r[1]), int(r[2]))
            if prev is None:
                continue
            ok = all(np.array_equal(br[sl], r[sl]) for sl in S.IN_PROF.values()) and all(br[c] == r[c] for c in S.IN_C.values())
            if k not in prev:
                n_new[k[2]] += 1
                bad_new[k[2]] += (not ok)
            else:
                bad_old += (not ok)
        C.update(a, S.PBLCarry.out_from_records(a))
        C.update(b, S.PBLCarry.out_from_records(b))
        prev = keys
    return n_new, bad_new, bad_old


@pytest.mark.skipif(not os.path.exists(f"{FF}/{D}/ffp_{IT0}.bin"), reason='dump absent')
def test_donor_rule_needed_and_exact_on_real_day():
    n_new, bad_new, bad_old = _run(True)
    assert n_new[2] == 8 and sum(n_new.values()) > 8       # 8 new ice tiles (D190) plus new ocean tiles; no new land / land-ice tiles
    assert sum(bad_new.values()) == 0 and bad_old == 0
    n_new2, bad_new2, bad_old2 = _run(False)
    assert n_new2 == n_new
    assert bad_new2[2] > 0 and sum(bad_new2.values()) > 0    # without the rule the appearing tiles carry stale state
