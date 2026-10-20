"""Stage-by-stage localisation of the early-step JAX-minus-Fortran difference (steps 0-3 of nov26_day2).
    python stage_localise.py --c1dir RUN_C1DIR --out stage_localise.txt [--steps 0,1,2,3]      (FF_DATA set; rocke3d-jax environment)
RUN_C1DIR holds nov26_day_step<k>.npz written by `gpu_day_run.py ... --c1dir` (GPU_DAY_DRIVER=d212_day, --rad native); its per-stage arrays are
compared with the recorded Fortran dumps of the same step, mapping found by matching step 0 (T rms difference):
  dyn <-> ffd_<it>_pre_condse (state entering CONDSE)   condse <-> ffd_post_condse   radia <-> ffd_post_radia   surface <-> ffd_post_surface
  dissip <-> ffa_step_<it>_d   filter <-> ffa_step_<it>_e (end of step)
Per step, field and stage: rms over valid cells of JAX - Fortran, and the cell of the largest |difference| (i, j, level; 0-based).  The step-end member spread
(max over p1-p5 of rms(member - control), from ffpt_*_<it>.bin) is printed beside it.  An inspection tool, not the acceptance score.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atm_day_report as RP
import clouds_condse_io as cio

STAGES = (('dyn', 'ffd', 'pre_condse'), ('condse', 'ffd', 'post_condse'), ('radia', 'ffd', 'post_radia'), ('surface', 'ffd', 'post_surface'),
          ('dissip', 'ffa', 'd'), ('filter', 'ffa', 'e'))
FIELDS = ('T', 'U', 'V', 'Q', 'QCL', 'QCI')


def fortran(ff, day, it, kind, tag):
    if kind == 'ffd':
        return cio.old_state(day, it, tag, ff)
    d = cio.read_cse(f'{ff}/{day}/ffa_step_{it}_{tag}.bin')
    return {k: d[k] for k in FIELDS if k in d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c1dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', default='0,1,2,3')
    ap.add_argument('--daydir', default='nov26_day2')
    a = ap.parse_args()
    ff, IT0 = RP.FF, 33312
    out = ['stage localisation; c1dir=%s; daydir=%s; rms over valid cells; "worst" = (i,j,level) 0-based and the signed difference' % (a.c1dir, a.daydir), '']
    for k in [int(x) for x in a.steps.split(',')]:
        it = IT0 + k
        z = np.load(os.path.join(a.c1dir, f'nov26_day_step{k}.npz'))
        real = RP.load_real_e(ff, a.daydir, it)
        mem = [RP.load_member(ff, a.daydir, m, it) for m in ('p1', 'p2', 'p3', 'p4', 'p5')]
        F = {s: fortran(ff, a.daydir, it, kd, tg) for s, kd, tg in STAGES}
        out.append('== step %d (it %d)' % (k, it))
        for f in FIELDS:
            msk = np.broadcast_to(RP.valid_mask(f)[:, :, None], real[f].shape)
            rm = max(np.sqrt(np.mean((m[f] - real[f])[msk] ** 2)) for m in mem)
            row = []
            for s, _, _ in STAGES:
                d = (z[s + '/' + f] - F[s][f]) * msk
                r = np.sqrt(np.mean(d[msk] ** 2))
                w = np.unravel_index(np.argmax(np.abs(d)), d.shape)
                row.append('%s %.2e (%d,%d,%d %+.2e)' % (s, r, w[0], w[1], w[2], d[w]))
            out.append('%-3s member spread at step end %.2e | ' % (f, rm) + ' | '.join(row))
        out.append('')
    open(a.out, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
