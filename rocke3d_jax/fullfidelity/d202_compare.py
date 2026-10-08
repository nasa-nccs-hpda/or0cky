"""D202: gusti-fixed ON run (run2) versus the D201 ON (elhx fix, NaN at 39) and OFF (D195 bitwise) saved states and the real record of land cell 112.
    python d202_compare.py RUN2DIR D201DIR OUT.md"""
import sys, json, glob, os
import numpy as np
run2, d201, outp = sys.argv[1:4]
F = ('T', 'U', 'V', 'Q', 'P', 'QCL', 'QCI')
real = json.load(open('scoping/d201_results/real_rows.json'))
lines = ['| step | finite T U V Q P QCL QCI | max|T-T_ON(D201)| | max|T-T_OFF| | cell112 tsns fixed / ON / OFF / real (C) | aevap fixed / real | ch fixed (ON) | ws fixed (ON) | ddml gusti |', '|---|---|---|---|---|---|---|---|---|']
n = len(glob.glob(f'{run2}/ours_d193/step_*.npz'))
for k in range(n):
    a = np.load(f'{run2}/ours_d193/step_{33312+k}.npz')
    fin = all(np.isfinite(a[f]).all() for f in F)
    row = [str(k), 'yes' if fin else 'NO']
    for tag in ('on', 'off'):
        fn = f'{d201}/{tag}/ours_d193/step_{33312+k}.npz'
        if os.path.exists(fn):
            b = np.load(fn)
            d = np.abs(a['T'] - b['T'])
            row.append('bitwise equal' if np.array_equal(a['T'], b['T'], equal_nan=True) else ('%.3g' % np.nanmax(d) if np.isfinite(d).any() else 'nan'))
        else:
            row.append('-')
    lf = f'{run2}/cap/land_{k}.npz'
    if os.path.exists(lf):
        l = np.load(lf); r = real[f'{k}/112/2']
        on = np.load(f'{d201}/on/cap/land_{k}.npz') if os.path.exists(f'{d201}/on/cap/land_{k}.npz') else None
        off = np.load(f'{d201}/off/cap/land_{k}.npz') if os.path.exists(f'{d201}/off/cap/land_{k}.npz') else None
        g = lambda z, x: float(z[x][112]) if z is not None else float('nan')
        row.append('%.3f / %.3f / %.3f / %.3f' % (g(l, 'land/ghy/tsns'), g(on, 'land/ghy/tsns'), g(off, 'land/ghy/tsns'), r['tsns']))
        row.append('%.3e / %.3e' % (g(l, 'land/ghy/aevap'), r['aevap']))
        row.append('%.4f (%.4f)' % (g(l, 'land/pbl/ch'), g(on, 'land/pbl/ch')))
        row.append('%.3f (%.3f)' % (g(l, 'land/pbl/ws'), g(on, 'land/pbl/ws')))
    else:
        row += ['-'] * 4
    ic = f'{run2}/cap/in_{k}.npz'
    if os.path.exists(ic):
        z = np.load(ic); row.append('%g %.2f' % (z['rows/rows2'][112][23], z['rows/rows2'][112][113]))
    else:
        row.append('-')
    lines.append('| ' + ' | '.join(row) + ' |')
open(outp, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
