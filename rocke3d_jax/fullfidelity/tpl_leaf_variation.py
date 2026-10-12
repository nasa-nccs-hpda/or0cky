"""D222: which leaves of the jax_surface template pytree change over a record set (per leaf: constant / varying, max |change|).  python tpl_leaf_variation.py [DAYDIR] [STEP_STRIDE]"""
import json
import sys
import numpy as np
import jax
import atm_step as A
import jax_surface as JS

DAY = sys.argv[1] if len(sys.argv) > 1 else 'nov26_day2'
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
it0 = 33312
base = None
const = {}
maxd = {}
for k in range(0, 108, STRIDE):
    R = A.Real('nov26', it0 + k)
    R.__dict__['date'] = DAY
    tpl, host = JS.build_template(A.surface_records(R))
    leaves = {jax.tree_util.keystr(p): np.asarray(x) for p, x in jax.tree_util.tree_flatten_with_path(tpl)[0]}
    if base is None:
        base = leaves
        const = {n: True for n in leaves}
        maxd = {n: 0.0 for n in leaves}
        continue
    for n, x in leaves.items():
        b = base[n]
        if x.shape != b.shape:
            const[n] = False
            maxd[n] = float('inf')
        elif x.dtype.kind == 'f':
            d = float(np.nanmax(np.abs(x - b))) if x.size else 0.0
            if d != 0.0:
                const[n] = False
                maxd[n] = max(maxd[n], d)
        elif not np.array_equal(x, b):
            const[n] = False
out = {n: dict(shape=list(base[n].shape), constant=const[n], max_change=maxd[n]) for n in base}
json.dump(out, open('tpl_leaf_variation.json', 'w'), indent=1)
print('leaves', len(out), 'constant', sum(const.values()))
for n, v in out.items():
    if not v['constant'] and "['ns'][1]" not in n and "['ghy'][1]" not in n:
        print('VARIES', n, v['shape'], f"{v['max_change']:.3g}")
