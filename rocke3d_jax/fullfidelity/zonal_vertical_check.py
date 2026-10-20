"""Zonal-mean, per-level check of JAX-minus-Fortran against the five one-ulp members (reviewer question: a wave-like vertical pattern above layer ~25 in U and T).
    python zonal_vertical_check.py --ours RUN/fid_fast_jax/ours_d193 --out OUTDIR [--daydir nov26_day2] [--steps 1,3,11,23,35,53]
For each step and field: zonal mean over longitude (U, V skip the polar row j=0; T, Q, ... use all defined cells), then
  D_jax(l,j) = zm(JAX) - zm(Fortran control), D_m(l,j) = zm(member m) - zm(Fortran control).
Per level l: rms over latitude of D_jax and of each D_m (area-unweighted, an inspection tool, not the acceptance score); ratio = rms(D_jax) / max_m rms(D_m).
Also the level-mean signed offset bias(l) = mean_j D_jax(l,j) against the members' range, and the lat-lev maps of D_jax and of one member.
Writes OUTDIR/zonal_<field>_step<k>.png, zonal_profile_<field>.png and OUTDIR/zonal_vertical.txt.  Not part of the acceptance rule; no thresholds are applied.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atm_day_report as RP

IT0 = 33312
MEMBERS = ('p1', 'p2', 'p3', 'p4', 'p5')
FIELDS = ('U', 'T')


def zmean(a, name):
    m = RP.valid_mask(name)[:, :, None] & np.ones((1, 1, a.shape[2]), bool)
    s = np.where(m, a, 0.0).sum(0)
    n = m.sum(0)
    z = np.where(n > 0, s / np.maximum(n, 1), np.nan)  # (JM, LM)
    return z.T  # (LM, JM)


def rms_lat(d):
    return np.sqrt(np.nanmean(d ** 2, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--daydir', default='nov26_day2')
    ap.add_argument('--steps', default='1,3,11,23,35,53')
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(',')]
    os.makedirs(a.out, exist_ok=True)
    ff = RP.FF
    L = np.arange(1, RP.LM + 1)
    lines = ['zonal-mean per-level check; ours=%s; daydir=%s; levels are 1-based (1 = bottom layer in the model\'s own ordering)' % (a.ours, a.daydir), '']
    prof = {f: {} for f in FIELDS}
    for k in steps:
        it = IT0 + k
        real = RP.load_real_e(ff, a.daydir, it)
        ours = RP.load_ours(a.ours, it)
        mem = [RP.load_member(ff, a.daydir, m, it) for m in MEMBERS]
        for f in FIELDS:
            zr = zmean(real[f], f)
            dj = zmean(ours[f], f) - zr
            dm = [zmean(m[f], f) - zr for m in mem]
            rj = rms_lat(dj)
            rm = np.array([rms_lat(d) for d in dm])
            bj = np.nanmean(dj, axis=1)
            bm = np.array([np.nanmean(d, axis=1) for d in dm])
            prof[f][k] = (rj, rm, bj, bm, dj, dm[0])
            ratio = rj / np.maximum(rm.max(0), 1e-300)
            lo, hi = slice(0, 25), slice(25, RP.LM)
            lines.append('%s step %d: rms-over-lat ratio (JAX/max member) median levels 1-25: %.2f, levels 26-40: %.2f; max in 26-40: %.2f at level %d; levels with ratio>2: %s'
                         % (f, k, np.median(ratio[lo]), np.median(ratio[hi]), ratio[hi].max(), 26 + int(np.argmax(ratio[hi])), [int(l) for l in L[ratio > 2]]))
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
            v = np.nanmax(np.abs(np.concatenate([dj.ravel(), dm[0].ravel()])))
            for axx, d, t in ((ax[0], dj, 'JAX - Fortran'), (ax[1], dm[0], 'member p1 - Fortran')):
                im = axx.pcolormesh(np.arange(RP.JM), L, d, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
                axx.set_title('%s zonal mean, %s, step %d' % (f, t, k)); axx.set_xlabel('latitude index j'); axx.set_ylabel('level')
                axx.axhline(25, color='k', lw=0.6, ls=':')
                fig.colorbar(im, ax=axx)
            ax[2].fill_betweenx(L, rm.min(0), rm.max(0), color='#cfd8e3', label='5 members (min-max)')
            ax[2].plot(rj, L, color='#d9822b', lw=2, label='JAX')
            ax[2].set_xscale('log'); ax[2].set_xlabel('rms over latitude'); ax[2].set_ylabel('level'); ax[2].axhline(25, color='k', lw=0.6, ls=':'); ax[2].legend()
            ax[2].set_title('per-level rms')
            fig.savefig(os.path.join(a.out, 'zonal_%s_step%d.png' % (f, k)), dpi=90); plt.close(fig)
    for f in FIELDS:
        fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for k in steps:
            rj, rm, bj, bm, _, _ = prof[f][k]
            ax[0].plot(rj / rm.max(0), L, label='step %d' % k)
            ax[1].plot(bj, L, label='step %d' % k)
        ax[0].axvline(1, color='k', lw=0.6); ax[0].axvline(2, color='r', lw=0.6, ls='--'); ax[0].set_xlabel('rms(JAX-F)/max member rms'); ax[0].set_ylabel('level'); ax[0].set_title('%s: ratio by level' % f)
        ax[1].set_xlabel('lat-mean signed JAX-F'); ax[1].set_title('%s: signed offset by level' % f); ax[1].legend(fontsize=7)
        for x in ax: x.axhline(25, color='k', lw=0.6, ls=':')
        fig.savefig(os.path.join(a.out, 'zonal_profile_%s.png' % f), dpi=90); plt.close(fig)
    # vertical roughness: second difference of the signed level offset, JAX vs members (is the oscillation larger than members')
    lines.append('')
    for f in FIELDS:
        for k in steps:
            rj, rm, bj, bm, dj, d1 = prof[f][k]
            rough = lambda b: np.sqrt(np.mean(np.diff(b[24:], 2) ** 2))
            rr = lambda D: np.sqrt(np.nanmean(np.diff(np.nan_to_num(D[24:]), 2, axis=0) ** 2))
            lines.append('%s step %d: vertical second-difference rms above level 25, JAX %.3g, member p1 %.3g (ratio %.2f)' % (f, k, rr(dj), rr(d1), rr(dj) / max(rr(d1), 1e-300)))
    open(os.path.join(a.out, 'zonal_vertical.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
