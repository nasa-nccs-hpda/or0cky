"""D187: first failing stage -- the hybrid step's stage snapshots against the real stage-exit records (atm_step.stage_references), common fields, category per field.
python d187_stage_vs_real.py DATE HYBDIR"""
import clouds_jax_env  # noqa: F401
import sys, json
import numpy as np
import atm_step as A, jax_harness as H, d187_common as K
date, hyb = sys.argv[1], sys.argv[2]
it0 = dict(A.DATES)[date]
R = A.Real(date, it0)
refs = A.stage_references(R)
z = H.load_npz(f'{hyb}/{date}_hyb.npz')
out = {}
for st in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter'):
    rows = {}
    for k, v in refs[st].items():
        key = f'{st}/{k}'
        if key in z and isinstance(v, np.ndarray) and v.dtype.kind == 'f' and z[key].shape == v.shape:
            rows[k] = K.slim(H.field_category(z[key], v))
    cats = {c: sum(1 for r in rows.values() if r['cat'] == c) for c in 'ABCD'}
    worst = sorted(rows.items(), key=lambda kv: -kv[1]['rel'])[:4]
    out[st] = dict(n=len(rows), categories=cats, worst=[(k, v['cat'], v['rel']) for k, v in worst])
    print(st, out[st]['n'], cats, [(k, c, '%.1e' % r) for k, c, r in out[st]['worst']])
json.dump(out, open(f'{hyb}/{date}_stage_vs_real.json', 'w'), indent=1, default=str)
