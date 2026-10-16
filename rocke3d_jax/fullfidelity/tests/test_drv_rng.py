import os
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import drv_rng as R
cio = pytest.importorskip('clouds_condse_io')
FF = cio.FF_DEFAULT


def _s(d, it):
    p = f"{FF}/{d}/ffc_cse_in_{it}.bin"
    if not os.path.exists(p):
        pytest.skip('dump absent')
    return [int(round(x)) & 0xFFFFFFFF for x in cio.read_cse(p)['SEEDS']]


def test_jump_matches_iteration():
    ix = 12345
    for _ in range(1000):
        ix = (ix * 69069 + 1) % R.M
    assert R.jump(12345, 1000) == ix


@pytest.mark.parametrize('d,its', [('nov26_day', range(33312, 33330)), ('jan01', range(17520, 17525))])
def test_seed_chain(d, its):
    for it in its:
        a, b = _s(d, it), _s(d, it + 1)
        assert R.radia_seed(a[0]) == a[1]
        assert R.next_seed(a[0], (it - 16032) % 5 == 0) == b[0]
