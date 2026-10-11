import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ks = [k for k in a.files if k in b.files]
bad = []
for k in ks:
    x, y = a[k], b[k]
    if x.shape != y.shape:
        bad.append((k, 'shape', x.shape, y.shape)); continue
    if x.dtype.kind == 'f':
        same = np.array_equal(x, y, equal_nan=True)
        if not same:
            d = np.abs(x - y); sc = np.abs(y).max() or 1.0
            bad.append((k, int((x != y).sum()), float(np.nanmax(d)), float(np.nanmax(d) / sc)))
    elif not np.array_equal(x, y):
        bad.append((k, 'int/bool', int((x != y).sum())))
print(f'{len(ks)} leaves compared; {len(bad)} differ')
for r in sorted(bad, key=lambda r: -(r[3] if len(r) > 3 and isinstance(r[3], float) else 0))[:25]:
    print('  ', r)
