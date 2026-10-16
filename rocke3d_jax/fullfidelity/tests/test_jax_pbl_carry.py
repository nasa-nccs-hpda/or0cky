"""D222: jax_pbl_carry (device persistent PBL state) reproduces the recorded substep-1 PBL columns of step k+1 BITWISE from the recorded substep-2
OUTPUT columns of step k, including tiles that are new at k+1 (loadbl donors).  Real records of ff_data/nov26_day; skipped without them."""
import os

import numpy as np
import pytest

D = __import__("os").environ.get("FF_DATA", "ff_data") + '/nov26_day'
IT0 = 33312
OUT = dict(u=slice(115, 123), v=slice(123, 131), t=slice(131, 139), q=slice(139, 147), e=slice(147, 154))
OUTC = dict(cm=94, ch=95, cq=96)


def _tpl(it):
    import atm_step as A
    import jax_surface as JS
    R = A.Real('nov26', it, os.path.dirname(D))
    R.__dict__['date'] = 'nov26_day'
    rec = A.surface_records(R)
    tpl, host = JS.build_template(rec)
    return tpl, host


def _outs(rows):
    o = {k: np.asarray(rows)[:, sl] for k, sl in OUT.items()}
    o.update({k: np.asarray(rows)[:, c] for k, c in OUTC.items()})
    return o


def _pairs():
    import atm_step as A
    # consecutive steps; include the transitions where the tile row sets change (found by the Nw/ice counts of the templates)
    return [(IT0, IT0 + 1), (IT0 + 5, IT0 + 6), (IT0 + 34, IT0 + 35), (IT0 + 35, IT0 + 36), (IT0 + 47, IT0 + 48)]


@pytest.mark.skipif(not os.path.exists(f'{D}/ffp_{IT0 + 1}.bin'), reason='records not available')
@pytest.mark.parametrize('k', range(5))
def test_carry_reproduces_next_step_pbl_columns(k):
    import jax_pbl_carry as PC
    import jax.numpy as jnp
    a, b = _pairs()[k]
    ta, ha = _tpl(a)
    tb, hb = _tpl(b)
    # state at the end of step a: the real substep-2 outputs of the rows of step a
    ixa = PC.index_arrays(ha, ta['ns'][1]['wvalid'])
    c = PC.loadbl(PC.empty())
    c = PC.update(c, {kk: jnp.asarray(v) for kk, v in _outs(ta['ns'][1]['pw']).items()},
                  {kk: jnp.asarray(v) for kk, v in _outs(ta['ns'][1]['pl']).items()},
                  {kk: jnp.asarray(v) for kk, v in _outs(ta['ns'][1]['pe']).items()}, ixa)
    # start of step b: loadbl, then fill the substep-1 template rows of b
    ixb = PC.index_arrays(hb, tb['ns'][0]['wvalid'])
    pw, pl, pe = PC.fill_template(tb['ns'][0]['pw'], tb['ns'][0]['pl'], tb['ns'][0]['pe'], PC.loadbl(c), ixb)
    cols = np.r_[33:36, 50:89]
    v = np.asarray(ixb['wvalid2'])
    assert np.array_equal(np.asarray(pw)[v][:, cols], np.asarray(tb['ns'][0]['pw'])[v][:, cols])
    assert np.array_equal(np.asarray(pl)[:, cols], np.asarray(tb['ns'][0]['pl'])[:, cols])
    assert np.array_equal(np.asarray(pe)[:, cols], np.asarray(tb['ns'][0]['pe'])[:, cols])
