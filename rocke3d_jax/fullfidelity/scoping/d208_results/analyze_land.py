"""D208 (b2): post-boundary carried land state with and without the transfer against the REAL records.
usage: python analyze_land.py OUT_ON OUT_OFF OUT.json
 OUT_ON/OUT_OFF: d208_day.py output dirs (--land-fractions 1 / 0): land_boundary_before.npz, land_boundary_after.npz, land_step44..53.npz (carried dyn_next w ht fr_snow, rows = 753 land cells)
 real: boundary: ffg_33360 first substep entry (w_in, ht_in, fr_snow_in) -- the real state after update_land_fractions; steps k = 48..52: entry of step k+1 (first substep)
 against our end state of step k (an entry record of step k+1 is the real end state of step k, no land processing in between); step 53: w_out of the second substep of 53."""
import json
import sys

import numpy as np

sys.path.insert(0, '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity')
import ghy_compare as GC

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
on, off, outp = sys.argv[1:4]
IT0 = 33312
R3 = lambda r, s, e: np.stack([x[s - 1:e].reshape(7, 3, order='F') for x in r])[:, :, :2]      # noqa: E731


def real(k, kind):
    """real state at the END of step k (rows of the 753 cells): from the entry of step k+1 (first substep) or the exit of the second substep of step k."""
    if kind == 'entry':
        g = GC.load(f'{FF}/ffg_{IT0 + k + 1}.bin')
        g = g[:len(g) // 2]
        return R3(g, 9, 29), R3(g, 30, 50), g[:, 70:72]
    g = GC.load(f'{FF}/ffg_{IT0 + k}.bin')
    g = g[len(g) // 2:]
    return R3(g, 181, 201), R3(g, 202, 222), g[:, 242:244]


def stat(ours, ref, base=None):
    d = np.abs(ours - ref)
    rms = float(np.sqrt(np.mean((ours - ref) ** 2)))
    return rms


res = {}
b = np.load(f'{on}/land_boundary_before.npz')
a = np.load(f'{on}/land_boundary_after.npz')
rw, rh, rf = real(47, 'exit') if False else (None, None, None)
g = GC.load(f'{FF}/ffg_{IT0 + 48}.bin')
g = g[:len(g) // 2]
rw, rh, rf = R3(g, 9, 29), R3(g, 30, 50), g[:, 70:72]          # REAL state right after the boundary (entry of step 48)
out = {}
for nm, bef, aft, ref in (('w', b['w'][:, :7, :], a['w'][:, :7, :], rw), ('ht', b['ht'][:, :7, :], a['ht'][:, :7, :], rh), ('fr_snow', b['fr_snow'], a['fr_snow'], rf)):
    eb, ea = np.abs(bef - ref), np.abs(aft - ref)
    ax = tuple(range(1, eb.ndim))
    cb, ca = eb.max(axis=ax), ea.max(axis=ax)
    moved = (np.abs(aft - bef).max(axis=ax) > 0)
    out[nm] = dict(n_cells_moved=int(moved.sum()), cells_nearer_to_real=int((ca < cb)[moved].sum()), cells_farther=int((ca > cb)[moved].sum()), cells_equal=int((ca == cb)[moved].sum()),
                   max_abs_err_without=float(cb.max()), max_abs_err_with=float(ca.max()),
                   rms_without=float(np.sqrt(np.mean(eb ** 2))), rms_with=float(np.sqrt(np.mean(ea ** 2))),
                   rms_without_moved=float(np.sqrt(np.mean(eb[moved] ** 2))), rms_with_moved=float(np.sqrt(np.mean(ea[moved] ** 2))),
                   not_improved_cells=[int(i) for i in np.nonzero(moved & (ca >= cb))[0][:20]])
    print(nm, {k: v for k, v in out[nm].items() if k != 'not_improved_cells'})
res['boundary'] = out
steps = {}
for k in range(48, 54):
    kind = 'entry' if k < 53 else 'exit'
    rw_, rh_, rf_ = real(k, kind)
    row = {}
    for nm, ref in (('w', rw_), ('ht', rh_), ('fr_snow', rf_)):
        e = {}
        for tag, d in (('on', on), ('off', off)):
            z = np.load(f'{d}/land_step{k:02d}.npz')
            x = z[nm][:, :7, :] if nm != 'fr_snow' else z[nm]
            e[tag] = float(np.sqrt(np.mean((x - ref) ** 2)))
        row[nm] = e
    steps[k] = row
    print(k, {nm: (f"{v['off']:.4e}", f"{v['on']:.4e}") for nm, v in row.items()})
res['steps'] = steps
json.dump(res, open(outp, 'w'), indent=1)
