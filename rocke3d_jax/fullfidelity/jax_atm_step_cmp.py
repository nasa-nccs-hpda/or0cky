"""D180 comparer.  python jax_atm_step_cmp.py DIR [nsteps]: per step and stage, per field: ref (NumPy chain) vs jax equal/unequal, max abs diff, location;
and jax (and ref) vs the real end state (F1 gate fields, categories A/B/C/D, atm_step_compare.gate_summary)."""
import sys
import numpy as np
import atm_step as A
import atm_step_compare as C

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def load(m, k):
    return dict(np.load(f'{d}/{m}_step{k}.npz'))


tot = {}
for k in range(n):
    r, j = load('ref', k), load('jax', k)
    uneq = []
    neq_total = 0
    nf = 0
    for key in sorted(r):
        if key.startswith('real/') or key not in j:
            continue
        nf += 1
        a, b = j[key], r[key]
        ne = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if ne.any():
            df = np.where(ne, np.abs(a - b), 0)
            w = tuple(int(x) + 1 for x in np.unravel_index(np.argmax(df), df.shape))
            uneq.append((key, int(ne.sum()), float(df.max()), float(df.max() / max(np.abs(b).max(), 1e-300)), w))
    print(f"step {k}: {nf} fields compared jax vs numpy chain, {nf - len(uneq)} bitwise equal, {len(uneq)} unequal")
    for u in uneq[:40]:
        print(f"    {u[0]:18s} n_unequal {u[1]:7d} max {u[2]:.3e} rel {u[3]:.2e} worst(1-based) {u[4]}")
    for who, S in (('jax', j), ('ref', r)):
        st = {}
        for f in C.GATE_FIELDS:
            if f'filter/{f}' in S and f'real/{f}' in S:
                st[f] = A.field_stats(S[f'filter/{f}'], S[f'real/{f}'])
        g = C.gate_summary(st)
        print(f"    vs real end state [{who}, libm]: {g['verdict']}; cats {g['categories']}; worst {g['worst_field']} rel {g['worst_rel']:.2e}")
