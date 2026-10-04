"""Validate the HBL convergence decision and the post-loop flux save (D61) against real dumps.

Per column (consecutive ffz_kppmix records with the same i,j), the Fortran ran ITER=1..n. The
decision to continue after each ITER must match the recorded HBL/KBL sequence (convergence check
reproduced, bitwise since it uses the recorded values). Then the post-loop flux save from the
last ITER's OVDIFFS fluxes must match the ffz_post record bitwise.
Run directory layout: ffz_kppmix_*, ffz_ovdiffs_*, ffz_setup_*, ffz_post_* (per itime).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kppmix_compare import load_kppmix_records  # noqa: E402
from ovdiffs_compare import load_ovdiffs_records  # noqa: E402
from oconv_save_ff import convergence_continue, flux_save  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402

RS_SETUP = 1218
RS_POST = 4 + 2 + 14 + 14  # i,j,iter,lmij, s0m1,dm, flg(0:13), fls(0:13)


def column_groups(kr):
    groups, cur = [], []
    for r in kr:
        if cur and (r['i'], r['j']) == (cur[-1]['i'], cur[-1]['j']) and r['iter'] == cur[-1]['iter'] + 1:
            cur.append(r)
        else:
            if cur:
                groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)
    return groups


def run(run_dir):
    dxypo = geomo_dyn_arrays()[-1]
    stats = dict(cols=0, conv_bad=0, conv_checks=0, save_cols=0, save_bad=0, dm_bad=0, flg_bad=0,
                 fls_bad=0)
    for pf in sorted(glob.glob(f'{run_dir}/ffz_post_*.bin')):
        itime = pf.split('_')[-1][:-4]
        kr = load_kppmix_records(f'{run_dir}/ffz_kppmix_{itime}.bin')
        ov = load_ovdiffs_records(f'{run_dir}/ffz_ovdiffs_{itime}.bin')
        S = np.fromfile(f'{run_dir}/ffz_setup_{itime}.bin', dtype='>f8').reshape(-1, RS_SETUP)
        P = np.fromfile(pf, dtype='>f8').reshape(-1, RS_POST)
        groups = column_groups(kr)
        assert len(groups) == len(P), (len(groups), len(P))
        pos = 0
        for gi, grp in enumerate(groups):
            stats['cols'] += 1
            # convergence decision after each ITER
            for k, r in enumerate(grp):
                hblp = grp[k - 1]['hbl'] if k > 0 else 0.0
                cont = convergence_continue(r['iter'], hblp, r['hbl'], r['ze'], r['kbl'])
                actual = k + 1 < len(grp)
                stats['conv_checks'] += 1
                stats['conv_bad'] += int(cont != actual)
            # post-loop save from the last ITER
            last = grp[-1]
            n = len(grp)
            g_rec, s_rec = ov[2 * (pos + n - 1)], ov[2 * (pos + n - 1) + 1]
            srow = S[pos + n - 1]
            p = P[gi]
            assert (int(p[0]), int(p[1]), int(p[2]), int(p[3])) == (last['i'], last['j'], last['iter'], last['lmij'])
            lmij = last['lmij']
            dm, flg3d, fls3d = flux_save(lmij, g_rec['fl_real'], s_rec['fl_real'], srow[1158],
                                         srow[1159], srow[1160], dxypo[last['j']], srow[45], p[4])
            stats['save_cols'] += 1
            stats['dm_bad'] += int(dm != p[5])
            stats['flg_bad'] += int(np.sum(np.array(flg3d[0:lmij + 1]) != p[6:6 + lmij + 1]))
            stats['fls_bad'] += int(np.sum(np.array(fls3d[0:lmij + 1]) != p[20:20 + lmij + 1]))
            pos += n
    return stats


if __name__ == '__main__':
    tot = None
    for rd in sys.argv[1:]:
        st = run(rd)
        print(rd, st)
        tot = st if tot is None else {k: tot[k] + st[k] for k in st}
    print('TOTAL', tot)
