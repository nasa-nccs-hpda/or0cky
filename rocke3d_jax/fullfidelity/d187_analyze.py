"""D187 C1 analysis: the hybrid coupled step (d187_coupled_step.py) against the NumPy libimf reference with the same closed surface (d187_ref_numpy.py),
field by field (categories A bitwise / B <= 1e-12 / C <= 1e-6 / D worse of jax_harness.field_category), per stage group.
    python d187_analyze.py DATE HYBDIR REFDIR [--server]
Writes HYBDIR/<date>_c1.json and prints a summary.  Both npz files must come from runs with the same core pinning (the runners record it)."""
import json
import os
import sys

import numpy as np

import jax_harness as H
import d187_common as K

GROUPS = ('dyn', 'condse', 'radia', 'X', 'surface', 'dissip', 'filter', 'surf', 'land')


def compare(a, b):
    rows = {}
    for g in GROUPS:
        ka = {k[len(g) + 1:] for k in a if k.startswith(g + '/')}
        kb = {k[len(g) + 1:] for k in b if k.startswith(g + '/')}
        common = sorted(ka & kb)
        res = {k: H.field_category(a[f'{g}/{k}'], b[f'{g}/{k}']) for k in common if a[f'{g}/{k}'].dtype.kind in 'fiub' and b[f'{g}/{k}'].dtype.kind in 'fiub'}
        cats = {c: sum(1 for v in res.values() if v['cat'] == c) for c in 'ABCD'}
        notA = {k: K.slim(v) for k, v in res.items() if v['cat'] != 'A'}
        rows[g] = dict(n_fields=len(res), categories=cats, only_in_hybrid=sorted(ka - kb), only_in_reference=sorted(kb - ka), not_A=notA)
    return rows


def main(date, hybdir, refdir, server=False):
    ref = H.load_npz(os.path.join(refdir, f'{date}_ref.npz'))
    out = {}
    for tag, fn in (('replay', f'{date}_hyb.npz'),) + ((('server', f'{date}_hyb_server.npz'),) if server else ()):
        hyb = H.load_npz(os.path.join(hybdir, fn))
        rows = compare(hyb, ref)
        out[tag] = rows
        print(f'== {date} C1 [{tag}] hybrid vs NumPy libimf reference')
        for g, r in rows.items():
            print(f'  {g:8s} n={r["n_fields"]:3d} {r["categories"]} not-A: {list(r["not_A"])[:10]}  only_hyb {r["only_in_hybrid"][:4]} only_ref {r["only_in_reference"][:4]}')
    # non-vacuity: the comparison can fail (a copy of a reference array with one element changed by 1 ulp is category B)
    k0 = 'filter/T'
    pert = ref[k0].copy()
    pert.flat[0] = np.nextafter(pert.flat[0], np.inf)
    out['non_vacuity_1ulp_detected_as'] = H.field_category(pert, ref[k0])['cat']
    json.dump(out, open(os.path.join(hybdir, f'{date}_c1.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], '--server' in sys.argv)
