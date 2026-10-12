"""D227: the Ent end-of-day flag of the template under ROCKE_TPL_OWN follows the step (it % 48 == 0), not the first-step base."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_eod_follows_step():
    import jax.numpy as jnp
    import jax_coupled as C
    base = dict(ghy=[dict(entf=dict(eod=jnp.ones(3, bool), Ca=jnp.zeros(3))), dict(entf=dict(eod=jnp.ones(2, bool), Ca=jnp.zeros(2))), dict(other=1)],
                keep='x')
    for it, want in ((33312, True), (33313, False), (33359, False), (33360, True)):
        t = C.tpl_own_eod(base, it)
        assert t['keep'] == 'x' and t['ghy'][2] == dict(other=1)
        for b, n in zip(t['ghy'][:2], (3, 2)):
            e = np.asarray(b['entf']['eod'])
            assert e.dtype == bool and e.shape == (n,) and bool(e.all()) == want and bool(e.any()) == want
    assert bool(np.asarray(base['ghy'][0]['entf']['eod']).all())          # the base is not modified
