"""D190: how do tiles that APPEAR between steps get their template columns?  Source rule (PBL_DRV.f loadbl, called at SURFACE.f:409 at the start of every step):
a tile type whose PBL did not run in the previous step (ipbl = 0) is initialised from the donor type of the same cell: ice <- ocean (first instance), ocean <- ice, else land;
land ice <- land; land <- land ice, else ocean.  This script TESTS the rule on the real PBL-entry records ffp_<it> of the 54-step nov26 day (all 154 columns, bitwise):
for every (step, cell) where an ice tile exists at step k and did not at step k-1, compare the ice row with the ocean row of the same cell at step k; as control, the same
comparison for ice tiles that existed at k-1."""
import sys, glob, os
import numpy as np
import clouds_jax_env  # noqa
import pbl_compare as PC

date = sys.argv[1] if len(sys.argv) > 1 else 'nov26_day'
FF = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
its = sorted(int(os.path.basename(p)[4:-4]) for p in glob.glob(f'{FF}/{date}/ffp_[0-9]*.bin'))
print(date, 'ffp steps', len(its), its[0], its[-1])
def rows(it):
    p = PC.load(f'{FF}/{date}/ffp_{it}.bin')
    d = {}
    for r in p:
        d[(int(r[0]), int(r[1]), int(r[2]))] = r
    return d
prev = rows(its[0])
new_eq = np.zeros(154, int); new_n = 0
old_eq = np.zeros(154, int); old_n = 0
newcells = []
vanish = 0
for it in its[1:]:
    cur = rows(it)
    for (i, j, t), r in cur.items():
        if t != 2:
            continue
        o = cur.get((i, j, 1))
        if o is None:
            continue
        eq = (r == o)
        if (i, j, 2) not in prev:
            new_eq += eq; new_n += 1; newcells.append((it, i, j))
        else:
            old_eq += eq; old_n += 1
    vanish += sum(1 for k in prev if k[2] == 2 and k not in cur)
    prev = cur
print('new ice tiles (absent at the previous step):', new_n, ' existing ice tiles (control):', old_n, ' ice tiles that vanished:', vanish)
if new_n:
    cols_all_eq = [c for c in range(154) if new_eq[c] == new_n]
    cols_none_eq = [c for c in range(154) if new_eq[c] == 0]
    print('columns equal to the OCEAN row of the same cell in ALL new tiles:', cols_all_eq)
    ctl_ne = [c for c in cols_all_eq if old_eq[c] != old_n]
    print('  of these, NOT always equal for the existing (control) ice tiles:', ctl_ne)
    print('columns equal in no new tile:', cols_none_eq)
    print('new tile examples (step, i, j):', newcells[:10])
