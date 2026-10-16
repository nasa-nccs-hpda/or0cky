"""Validate dyn_isotropuv_ff.py (isotropuv + shap1, D95) against real-Fortran dumps.

Dumps per date in ff_data/<date>/ (big-endian float64):
  ffd_isotr_geom.bin  once: jm, im, dt, fjeq, radius; COSV(1:jm), DXV(1:jm), COSIV(1:im), SINIV(1:im)
  ffd_isotr_stat.bin  EVERY processed (j,l) row of every isotropuv call in the window (8 doubles):
                      itime, ncall(1..5 within the step), j, l, k, fac, n=int(fac)+1, max|u_in|
  ffd_isotr.bin       sampled rows (l = 1,5,9,..,37 and every row with n>1), 440 doubles:
                      itime, ncall, j, l, k, fac, n, hemi, U_in(72), V_in(72), UA(72), VA(72)
                      (x-y velocities after the two shap1 calls, before the pole FFT), U_out(72), V_out(72)
  ffd_avrx_consts.bin (D94) supplies the FFT C,S tables.
The 5 calls per step are, in order of occurrence in DYNAM: ncall=1 initial forward (UX,VX), 2 backward
(UT,VT), 3 odd/even leapfrog... (the call-site mapping is inferred from the source, only the counter is recorded).

Usage: python3 dyn_isotropuv_compare.py
"""
import numpy as np

import dyn_avrx_compare as ac
import dyn_geom_ff as gm
import dyn_isotropuv_ff as fi

FF_DEFAULT = ac.FF_DEFAULT
DATES = ac.DATES
NSTEP = 6
IM, JM = 72, 46


def available(date, ff=FF_DEFAULT):
    import os
    return all(os.path.exists(f"{ff}/{date}/{n}") for n in
               ("ffd_isotr.bin", "ffd_isotr_stat.bin", "ffd_isotr_geom.bin", "ffd_avrx_consts.bin"))


def load_geom(date, ff=FF_DEFAULT):
    g = np.fromfile(f"{ff}/{date}/ffd_isotr_geom.bin", dtype='>f8')
    assert int(g[0]) == JM and int(g[1]) == IM and g.size == 5 + 2 * JM + 2 * IM
    return dict(dt=g[2], fjeq=g[3], radius=g[4], cosv=g[5:5 + JM], dxv=g[5 + JM:5 + 2 * JM],
                cosiv=g[5 + 2 * JM:5 + 2 * JM + IM], siniv=g[5 + 2 * JM + IM:])


def load_stat(date, ff=FF_DEFAULT):
    s = np.fromfile(f"{ff}/{date}/ffd_isotr_stat.bin", dtype='>f8').reshape(-1, 8)
    return dict(itime=s[:, 0].astype(int), ncall=s[:, 1].astype(int), j=s[:, 2].astype(int),
                l=s[:, 3].astype(int), k=s[:, 4], fac=s[:, 5], n=s[:, 6].astype(int), umax=s[:, 7])


def load_rows(date, ff=FF_DEFAULT):
    r = np.fromfile(f"{ff}/{date}/ffd_isotr.bin", dtype='>f8').reshape(-1, 440)
    return dict(itime=r[:, 0].astype(int), ncall=r[:, 1].astype(int), j=r[:, 2].astype(int),
                l=r[:, 3].astype(int), k=r[:, 4], fac=r[:, 5], n=r[:, 6].astype(int), hemi=r[:, 7].astype(int),
                u=r[:, 8:80], v=r[:, 80:152], ua=r[:, 152:224], va=r[:, 224:296],
                uo=r[:, 296:368], vo=r[:, 368:440])


def run_date(date, ff=FF_DEFAULT, analytic=False, tables_dump=True):
    geo = load_geom(date, ff)
    k = ac.load_consts(date, ff)
    C, S = (k['C'], k['S']) if tables_dump else __import__('dyn_avrx_ff').make_tables()
    if analytic:
        a = gm.geometry(geo['radius'])
        geo = dict(cosv=a['cosv'], dxv=a['dxv'], cosiv=a['cosiv'], siniv=a['siniv'], fjeq=a['fjeq'])
    r = load_rows(date, ff)
    Uo, Vo, aux = fi.iso_rows(r['u'], r['v'], r['j'], geo, C, S, return_all=True)
    res = dict(n=len(r['j']),
               k=float(np.max(np.abs(aux['k'] - r['k']))), fac=float(np.max(np.abs(aux['fac'] - r['fac']))),
               n_ok=bool(np.array_equal(aux['n'], r['n'])),
               ua=float(np.max(np.abs(aux['ua'] - r['ua']))), va=float(np.max(np.abs(aux['va'] - r['va']))),
               uo=float(np.max(np.abs(Uo - r['uo']))), vo=float(np.max(np.abs(Vo - r['vo']))),
               n_exact=int(np.sum(Uo == r['uo']) + np.sum(Vo == r['vo'])), n_total=2 * Uo.size,
               chg=float(np.max(np.abs(r['uo'] - r['u']))), scale=float(np.max(np.abs(r['u']))))
    # all rows (stat file): k, fac, n recomputed from the stored max|u_in| / dxv
    st = load_stat(date, ff)
    dxv = geo['dxv'][st['j'] - 1]
    kk = fi._k_of_u(st['umax'], dxv)[1]
    fac = kk * fi.DT / (dxv * dxv)
    res['stat_k'] = float(np.max(np.abs(kk - st['k'])))
    res['stat_fac'] = float(np.max(np.abs(fac - st['fac'])))
    res['stat_n_ok'] = bool(np.array_equal(fac.astype(int) + 1, st['n']))
    return res


def n_histogram(date, ff=FF_DEFAULT):
    st = load_stat(date, ff)
    u, c = np.unique(st['n'], return_counts=True)
    by_row = {int(j): sorted(set(st['n'][st['j'] == j].tolist())) for j in sorted(set(st['j'].tolist()))}
    return dict(zip(u.tolist(), c.tolist())), by_row, st


if __name__ == "__main__":
    for date, it0 in DATES:
        for an in (False, True):
            r = run_date(date, analytic=an)
            print(f"{date} {'analytic geom' if an else 'dumped geom  '}: rows={r['n']} k={r['k']:.1e} fac={r['fac']:.1e} "
                  f"n_ok={r['n_ok']} ua={r['ua']:.1e} va={r['va']:.1e} U_out={r['uo']:.1e} V_out={r['vo']:.1e} "
                  f"exact={r['n_exact']}/{r['n_total']} chg={r['chg']:.2e} scale={r['scale']:.1f}")
        print(f"   all-row stats: k={r['stat_k']:.1e} fac={r['stat_fac']:.1e} n_ok={r['stat_n_ok']}")
        h, byrow, st = n_histogram(date)
        print(f"   sub-iteration counts n over all {len(st['j'])} (j,l) rows of 30 calls: {h}")
        print(f"   n values by row j: {byrow}")
