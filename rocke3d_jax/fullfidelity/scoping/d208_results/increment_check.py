"""D208: the INCREMENT of the transfer in our run (after - before at the boundary) against the REAL increment (entry of step 48 - exit of step 47 of the record), per field.
usage: python increment_check.py OUT_ON OUT.json"""
import json, sys
import numpy as np
sys.path.insert(0, '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity')
import ghy_compare as GC
FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
on, outp = sys.argv[1:3]
R3 = lambda r, s, e: np.stack([x[s - 1:e].reshape(7, 3, order='F') for x in r])[:, :, :2]      # noqa: E731
a = GC.load(f'{FF}/ffg_33359.bin'); n = len(a) // 2; a2 = a[n:]
b = GC.load(f'{FF}/ffg_33360.bin')[:n]
real = dict(w=R3(b, 9, 29) - R3(a2, 181, 201), ht=R3(b, 30, 50) - R3(a2, 202, 222), fr_snow=b[:, 70:72] - a2[:, 242:244])
bf, af = np.load(f'{on}/land_boundary_before.npz'), np.load(f'{on}/land_boundary_after.npz')
res = {}
for k in ('w', 'ht', 'fr_snow'):
    ours = (af[k][:, :7, :] - bf[k][:, :7, :]) if k != 'fr_snow' else af[k] - bf[k]
    r = real[k]
    ax = tuple(range(1, r.ndim))
    nr, no = np.abs(r).max(axis=ax), np.abs(ours).max(axis=ax)
    ch = nr > 0
    err = np.abs(ours - r).max(axis=ax)
    rel = err[ch] / nr[ch]
    res[k] = dict(real_changed=int(ch.sum()), ours_changed=int((no > 0).sum()), both=int(((no > 0) & ch).sum()), median_rel_increment_error=float(np.median(rel)), p90=float(np.percentile(rel, 90)),
                  max=float(rel.max()), frac_cells_within_1pct=float((rel < 0.01).mean()), rms_real_increment=float(np.sqrt(np.mean(r ** 2))), rms_increment_error=float(np.sqrt(np.mean((ours - r) ** 2))))
    print(k, res[k])
json.dump(res, open(outp, 'w'), indent=1)
