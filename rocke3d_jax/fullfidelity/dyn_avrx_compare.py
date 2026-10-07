"""Validate dyn_avrx_ff.py (FFT72 FFT/FFTI + AVRX, D94) against real-Fortran dumps.

Dumps (instrumentation/ATMDYN_dynA.f.patch + ATM_DRV_dynA.f.patch), per date in ff_data/<date>/
(all big-endian float64 streams):
  ffd_avrx_consts.bin  once: jm, dlon, bydyp(3); DRAT(1:jm), NMIN(1:jm) (as doubles), DXP(1:jm),
                       DYP(1:jm), BYSN(1:36), C(0:72), S(0:72)     (AVRX SAVE tables + FFT0 tables)
  ffd_avrx_calls.bin   every AVRX call in the 6-step window: [itime, site, ncall, sampled]
                       (site 1 = AFLUX, 3 = PGF; site 2 (ADVECV-labelled call inside the dead V2 PGF) never runs)
  ffd_avrx.bin         sampled calls (every 17th per (itime,site)), one 222-double record per active
                       row (DRAT<=1): itime, site, ncall, j, X_in(72), AN(0:36), BN(0:36) (FFT of X_in,
                       before truncation scaling), X_out(72)

Usage: python3 dyn_avrx_compare.py
"""
import os
import numpy as np

import dyn_avrx_ff as fa
import dyn_geom_ff as gm

FF_DEFAULT = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6
JM = 46


def available(date, ff=FF_DEFAULT):
    import os
    return all(os.path.exists(f"{ff}/{date}/{n}") for n in
               ("ffd_avrx.bin", "ffd_avrx_consts.bin", "ffd_avrx_calls.bin", "ffd_isotr_geom.bin"))


def load_consts(date, ff=FF_DEFAULT):
    cs = np.fromfile(f"{ff}/{date}/ffd_avrx_consts.bin", dtype='>f8')
    jm = int(cs[0])
    assert jm == JM and cs.size == 3 + 4 * jm + 36 + 146
    o = 3
    d = dict(dlon=cs[1], bydyp3=cs[2])
    d['drat'] = cs[o:o + jm]; o += jm
    d['nmin'] = cs[o:o + jm].astype(int); o += jm
    d['dxp'] = cs[o:o + jm]; o += jm
    d['dyp'] = cs[o:o + jm]; o += jm
    d['bysn'] = np.concatenate([[0.], cs[o:o + 36]]); o += 36
    d['C'] = cs[o:o + 73]; d['S'] = cs[o + 73:o + 146]
    return d


def load_rows(date, ff=FF_DEFAULT):
    r = np.fromfile(f"{ff}/{date}/ffd_avrx.bin", dtype='>f8').reshape(-1, 222)
    return dict(itime=r[:, 0].astype(int), site=r[:, 1].astype(int), ncall=r[:, 2].astype(int),
                j=r[:, 3].astype(int), x=r[:, 4:76], an=r[:, 76:113], bn=r[:, 113:150], xo=r[:, 150:222])


def load_calls(date, ff=FF_DEFAULT):
    c = np.fromfile(f"{ff}/{date}/ffd_avrx_calls.bin", dtype='>f8').reshape(-1, 4)
    return dict(itime=c[:, 0].astype(int), site=c[:, 1].astype(int), ncall=c[:, 2].astype(int),
                sampled=c[:, 3].astype(int))


def radius(date, ff=FF_DEFAULT):
    return np.fromfile(f"{ff}/{date}/ffd_isotr_geom.bin", dtype='>f8')[4]


def run_date(date, ff=FF_DEFAULT, tables="dump"):
    """tables: 'dump' (recorded C,S,DRAT,NMIN,BYSN), 'analytic' (make_tables + avrx_tables from the analytic
    geometry).  Returns dict of results."""
    k = load_consts(date, ff)
    rows = load_rows(date, ff)
    if tables == "analytic":
        C, S = fa.make_tables()
        g = gm.geometry(radius(date, ff))
        drat, nmin, bysn = fa.avrx_tables(g['dxp'], g['bydyp'][2], g['dlon'])
    else:
        C, S, drat, nmin, bysn = k['C'], k['S'], k['drat'], k['nmin'], k['bysn']
    out, A, B = fa.avrx_rows(rows['x'], rows['j'], drat, nmin, bysn, C, S, return_spec=True)
    res = dict(n=len(rows['j']),
               fft_A=float(np.max(np.abs(A - rows['an']))), fft_B=float(np.max(np.abs(B - rows['bn']))),
               avrx=float(np.max(np.abs(out - rows['xo']))),
               n_exact=int(np.sum(out == rows['xo'])), n_total=out.size,
               chg=float(np.max(np.abs(rows['xo'] - rows['x']))), scale=float(np.max(np.abs(rows['x']))))
    # inverse transform checked on its own: FFTI(AN,BN) of the recorded spectrum reproduces X_in
    xr = fa.ffti72(rows['an'], rows['bn'], C, S)
    res['roundtrip'] = float(np.max(np.abs(xr - rows['x'])))
    # numpy rfft alternative
    A2, B2 = fa.fft72_np(rows['x'])
    res['np_fft_A'] = float(np.max(np.abs(A2 - rows['an']))); res['np_fft_B'] = float(np.max(np.abs(B2 - rows['bn'])))
    o2 = fa.avrx_rows_np(rows['x'], rows['j'], drat, nmin, bysn)
    res['np_avrx'] = float(np.max(np.abs(o2 - rows['xo'])))
    res['np_avrx_rel'] = float(np.max(np.abs(o2 - rows['xo'])) / np.max(np.abs(rows['xo'])))
    # tables
    g = gm.geometry(radius(date, ff))
    dr2, nm2, by2 = fa.avrx_tables(g['dxp'], g['bydyp'][2], g['dlon'])
    res['tab_drat'] = float(np.max(np.abs(dr2 - k['drat']) / np.maximum(k['drat'], 1e-300)))
    res['tab_nmin_ok'] = bool(np.array_equal(nm2[1:JM - 1], k['nmin'][1:JM - 1]))
    res['tab_bysn'] = float(np.max(np.abs(by2 - k['bysn'])))
    res['tab_dxp'] = float(np.max(np.abs(g['dxp'] - k['dxp'])))
    res['tab_C'] = float(np.max(np.abs(fa.make_tables()[0] - k['C'])))
    res['tab_S'] = float(np.max(np.abs(fa.make_tables()[1] - k['S'])))
    return res


if __name__ == "__main__":
    for date, it0 in DATES:
        k = load_calls(date)
        per = {s: int(np.sum((k['site'] == s) & (k['itime'] == it0))) for s in (1, 2, 3)}
        rows = load_rows(date)
        print(f"== {date} ==  calls/step by site {per}; sampled calls {int(k['sampled'].sum())}; "
              f"rows dumped {len(rows['j'])}; active J = {sorted(set(rows['j'].tolist()))}")
        for tb in ("dump", "analytic"):
            r = run_date(date, tables=tb)
            print(f"  tables={tb}: fft_A={r['fft_A']:.1e} fft_B={r['fft_B']:.1e} avrx={r['avrx']:.1e} "
                  f"exact={r['n_exact']}/{r['n_total']} roundtrip(FFTI(AN,BN) vs X_in)={r['roundtrip']:.1e} "
                  f"chg={r['chg']:.2e} scale={r['scale']:.2e}")
        print(f"  numpy rfft alternative: spectra |dA|={r['np_fft_A']:.1e} |dB|={r['np_fft_B']:.1e}; "
              f"AVRX max|diff|={r['np_avrx']:.2e} (rel to max {r['np_avrx_rel']:.1e})")
        print(f"  analytic tables vs dump: DRAT rel={r['tab_drat']:.1e} NMIN_equal={r['tab_nmin_ok']} "
              f"BYSN={r['tab_bysn']:.1e} DXP={r['tab_dxp']:.1e} C={r['tab_C']:.1e} S={r['tab_S']:.1e}")
