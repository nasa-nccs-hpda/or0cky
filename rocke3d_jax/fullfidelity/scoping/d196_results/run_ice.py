"""D196 wrapper (scratchpad only): d195_day with allarr_of replaced by a small dict of the ice/lake/ocean surface state per step."""
import sys
sys.path.insert(0, '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity')
import clouds_jax_env  # noqa
import functools, json, os
import numpy as np
import jax_coupled as C
import jax_surface as JS
import d193_day as D
import d187_common as K

def small(arrays):
    SS2 = C.tree_np(arrays['SS2']); V2n = C.tree_np(arrays['V2'])
    out = {}
    for grp in ('ice', 'lake'):
        out.update(K.flatten(SS2[grp], f'surf/{grp}/'))
    out.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy']), 'surf/v2/'))
    return out
D.allarr_of = small
C.Coupled = functools.partial(C.Coupled, nit_strict=False)
if __name__ == '__main__':
    D.main()
