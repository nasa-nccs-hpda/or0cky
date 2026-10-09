"""Compare the inputs of the first OADVT2 X sweep captured on the CPU run and on the GPU run (numpy only; D214 follow-up).

usage: python gpu_xsweep_compare_inputs.py CPU_INPUTS.npz GPU_INPUTS.npz
Both files come from  XSWEEP_CAPTURE=<file> python gpu_oadvt_probe.py  (one on each device).  The sweep alone is fine on the GPU with the CPU's inputs
(job 58803559), so the NaN of the full GPU run must come from different inputs.  Prints, per input array, the number of differing values, the largest
absolute and relative difference and where, and for the Courant data (mudt, nc, lane_pass) the structural differences that can trigger a division by zero.
"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print('cpu', sys.argv[1], '\ngpu', sys.argv[2])
for k in ('rm', 'rx', 'ry', 'rz', 'mm', 'mudt', 'nc', 'lane_pass', 'lmu_l', 'lmm_l'):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print(f'{k:10s} SHAPE differs {x.shape} vs {y.shape}')
        continue
    if x.dtype.kind in 'fiub':
        xf, yf = x.astype(float), y.astype(float)
        d = np.abs(xf - yf)
        nd = int((xf != yf).sum())
        scale = np.abs(xf).max() or 1.0
        idx = tuple(int(i) for i in np.unravel_index(np.argmax(d), d.shape)) if d.size else ()
        print(f'{k:10s} differing {nd:7d} of {x.size:7d}   max|diff| {d.max():.3e} (rel to field max {d.max() / scale:.2e}) at {idx}   non-finite cpu/gpu {int((~np.isfinite(xf)).sum())}/{int((~np.isfinite(yf)).sum())}')
nc1, nc2 = a['nc'], b['nc']
print('\nnc (Courant sub-steps): cpu max', int(nc1.max()), 'gpu max', int(nc2.max()), '; lanes where they differ', int((nc1 != nc2).sum()))
m = b['mm']
print('lanes in which any |mudt| is non-zero and the mass of the same row is zero somewhere (potential 0 division), cpu/gpu inputs:')
for tag, z in (('cpu', a), ('gpu', b)):
    mu, mm, lane = z['mudt'], z['mm'], z['lane_pass']
    print(f'  {tag}: lanes {int(lane.sum())}, rows with mudt != 0 : {int((np.abs(mu).max(axis=1) > 0).sum())}, mass min over the whole mm field {float(mm.min()):.3e}, count mm == 0: {int((mm == 0).sum())}, mm < 0: {int((mm < 0).sum())}')
