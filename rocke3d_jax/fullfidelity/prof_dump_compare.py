"""Compare two gpu_mstcnv_bench BENCH_DUMP npz files: count of bitwise-different leaves, max abs / relative difference per leaf (worst 12).  python prof_dump_compare.py A.npz B.npz"""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
rows = []
for k in a.files:
    x, y = a[k], b[k]
    if x.dtype.kind in 'fc':
        d = np.abs(x - y); m = np.abs(x).max() if x.size else 0
        rows.append((float(np.nanmax(d)) if d.size else 0.0, float(np.nanmax(d) / m) if m > 0 and d.size else 0.0, int((x != y).sum()), k))
    else:
        rows.append((0.0 if np.array_equal(x, y) else 1.0, 0.0, int((x != y).sum()), k))
print('leaves', len(rows), 'bitwise-equal', sum(r[2] == 0 for r in rows), 'different', sum(r[2] > 0 for r in rows))
for r in sorted(rows, reverse=True)[:12]:
    print(f'  maxabs {r[0]:.3e} maxrel-to-leaf-max {r[1]:.3e} n_unequal {r[2]:6d} {r[3][:90]}')
