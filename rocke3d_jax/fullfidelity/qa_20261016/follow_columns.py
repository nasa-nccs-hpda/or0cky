"""Follow selected columns step by step: when does JAX leave the Fortran members' range, in which field and layer?"""
import os, sys, json
import numpy as np

PKG = os.environ.get('REVIEW_PKG', '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/rocke3d_review_package_20261016')
sys.path.insert(0, PKG + '/scoring_tool')
import atm_day_report as RP

FF, DAY, IT0 = PKG + '/fortran_reference', 'nov26_day2', 33312
NAMES = ['ctrl', 'p1', 'p2', 'p3', 'p4', 'p5']
ours_dir = PKG + '/jax_run/ours_d193'
COLS = {'A (-157.5,-10)': (4, 20), 'B (-162.5,-14)': (3, 19), 'C (-162.5,-6)': (3, 21), 'D (137.5,34)': (63, 31)}
FIELDS = ['T', 'Q', 'U', 'V', 'QCL', 'QCI']
NSTEP = 7
hist = {c: [] for c in COLS}
prof = {}
for k in range(NSTEP):
    it = IT0 + k
    runs = [RP.load_member(FF, DAY, nm, it) for nm in NAMES]
    jax = RP.load_ours(ours_dir, it)
    for c, (i, j) in COLS.items():
        row = {}
        for f in FIELDS:
            A = np.stack([r[f][i, j, :] for r in runs])   # (6, LM)
            C, J = A[0], jax[f][i, j, :]
            scale = np.abs(C).max() + 1e-300
            width = A.max(axis=0) - A.min(axis=0)
            diff = np.abs(J - C)
            ratio = diff / np.maximum(width, 1e-12 * scale)
            l = int(np.argmax(ratio))
            row[f] = dict(maxdiff=float(diff.max()), rel=float(diff.max() / scale), layer_maxdiff=int(np.argmax(diff) + 1),
                          worst_ratio=float(ratio[l]), worst_layer=l + 1, width_at_worst=float(width[l]))
            if c.startswith('A') and f in ('T', 'Q', 'QCL', 'QCI') and k in (2, 3, 4):
                prof[(k, f)] = dict(ctrl=C[14:21].tolist(), jax=J[14:21].tolist(), width=width[14:21].tolist())
        hist[c].append(row)

for c in COLS:
    print(f'== column {c}: per step, per field: max |JAX-ctrl| (relative to column max) at layer L; worst ratio to member range at layer')
    print('step ' + ' '.join(f'{f:>34s}' for f in FIELDS))
    for k in range(NSTEP):
        print(f'{k:4d} ' + ' '.join(
            f'{hist[c][k][f]["maxdiff"]:.1e}({hist[c][k][f]["rel"]:.0e})L{hist[c][k][f]["layer_maxdiff"]:<2d} r={hist[c][k][f]["worst_ratio"]:.0e}'.rjust(34)
            for f in FIELDS))
print('== column A, layers 15-21, steps 2-4 (ctrl, JAX, member range width)')
for (k, f), d in sorted(prof.items()):
    print(f'step {k} {f}: ctrl', ' '.join(f'{x:.3e}' for x in d['ctrl']))
    print(f'        jax ', ' '.join(f'{x:.3e}' for x in d['jax']))
    print(f'       width', ' '.join(f'{x:.1e}' for x in d['width']))
