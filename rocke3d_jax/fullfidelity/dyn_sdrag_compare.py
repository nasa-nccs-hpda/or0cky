"""Validate dyn_sdrag_ff.py (SDRAG, D95) against real-Fortran dumps.

Dumps per date in ff_data/<date>/ (big-endian float64):
  ffd_sdrag_consts.bin once: lm, ls1, lsdrag, lpsdrag, ang_sdrag, wc_jdrag, wmax, x_sdrag(1), x_sdrag(2),
                       linear_sdrag(0/1); jm; CSDRAGL(1:lm), VSDRAGL(1:lm) (only L>=LS1 are defined in
                       the Fortran arrays; lower entries are memory beyond the array and ignored);
                       COSV, RAPVN, RAPVS, DXYV, DXYN, DXYS (each 1:jm); RGAS
  ffd_sdrag.bin        sampled columns, 5+11*LM = 445 doubles per column per call:
                       itime, ncall(1..2 within step), i, j, dt1; then U,V,T,PK(L,I,J),PEDN(L+1,I,J),
                       MA(L,I+1,J-1), MA(L,I,J-1), MA(L,I+1,J), MA(L,I,J) (each length LM; inputs), then
                       U,V after SDRAG (length LM each).  Columns: all J=2..JM, I=1,25,49.
  ffd_sdrag_stat.bin   2 records of 10 doubles per call: [itime, ncall, dt1, n_levels(L>=LS1,J=2..JM,all I),
                       n(wl>wmaxj), n(cd_lin), max wl, 0, 0, n(T out of range)] from the FULL-grid inputs, and
                       [itime, ncall, dt1, -1,-1,-1,-1, max|dU|, max|dV|, -1] over the full grid.
Usage: python3 dyn_sdrag_compare.py
"""
import numpy as np

import dyn_avrx_compare as ac
import dyn_sdrag_ff as sd

FF_DEFAULT = ac.FF_DEFAULT
DATES = ac.DATES
LM, JM = 40, 46
NREC = 5 + 11 * LM


def available(date, ff=FF_DEFAULT):
    import os
    return all(os.path.exists(f"{ff}/{date}/{n}") for n in
               ("ffd_sdrag.bin", "ffd_sdrag_stat.bin", "ffd_sdrag_consts.bin"))


def load_consts(date, ff=FF_DEFAULT):
    c = np.fromfile(f"{ff}/{date}/ffd_sdrag_consts.bin", dtype='>f8')
    assert int(c[0]) == LM and int(c[10]) == JM and c.size == 11 + 2 * LM + 6 * JM + 1, c.size
    p = dict(ls1=int(c[1]), lsdrag=int(c[2]), lpsdrag=int(c[3]), ang_sdrag=int(c[4]), wc_jdrag=c[5], wmax=c[6],
             x_sdrag=(c[7], c[8]), linear=bool(c[9]))
    o = 11
    p['csdragl'] = c[o:o + LM]; o += LM
    p['vsdragl'] = c[o:o + LM]; o += LM
    geo = {}
    for nm in ('cosv', 'rapvn', 'rapvs', 'dxyv', 'dxyn', 'dxys'):
        geo[nm] = c[o:o + JM]; o += JM
    p['rgas'] = c[o]
    assert not p['linear']
    return p, geo


def load_cols(date, ff=FF_DEFAULT):
    r = np.fromfile(f"{ff}/{date}/ffd_sdrag.bin", dtype='>f8')
    assert r.size % NREC == 0
    r = r.reshape(-1, NREC)
    hdr = r[:, :5]
    d = dict(itime=hdr[:, 0].astype(int), ncall=hdr[:, 1].astype(int), i=hdr[:, 2].astype(int),
             j=hdr[:, 3].astype(int), dt1=hdr[:, 4])
    b = r[:, 5:]
    names = ('u', 'v', 't', 'pk', 'pedn1', 'ma_ip1_jm1', 'ma_i_jm1', 'ma_ip1_j', 'ma_i_j', 'uo', 'vo')
    for k, nm in enumerate(names):
        d[nm] = b[:, k * LM:(k + 1) * LM]
    return d


def load_stat(date, ff=FF_DEFAULT):
    s = np.fromfile(f"{ff}/{date}/ffd_sdrag_stat.bin", dtype='>f8').reshape(-1, 10)
    return s[0::2], s[1::2]


def run(cols, p, geo, dt1=None):
    """Apply sdrag_columns to every recorded column (all share dt1 per call; group by it)."""
    uo = np.zeros_like(cols['u']); vo = np.zeros_like(cols['v'])
    ncl = 0
    for dt in np.unique(cols['dt1']):
        m = cols['dt1'] == dt
        a = {k: cols[k][m] for k in ('u', 'v', 't', 'pk', 'pedn1', 'ma_ip1_jm1', 'ma_i_jm1', 'ma_ip1_j', 'ma_i_j')}
        un, vn, ex = sd.sdrag_columns(a['u'], a['v'], a['t'], a['pk'], a['pedn1'], a['ma_ip1_jm1'], a['ma_i_jm1'],
                                      a['ma_ip1_j'], a['ma_i_j'], cols['j'][m], dt, p, geo)
        uo[m] = un; vo[m] = vn; ncl += ex['nclamp']
    return uo, vo, ncl


def run_date(date, ff=FF_DEFAULT):
    p, geo = load_consts(date, ff)
    cols = load_cols(date, ff)
    uo, vo, ncl = run(cols, p, geo)
    st_in, st_out = load_stat(date, ff)
    return dict(n=len(cols['j']), du=float(np.max(np.abs(uo - cols['uo']))), dv=float(np.max(np.abs(vo - cols['vo']))),
                n_exact=int(np.sum(uo == cols['uo']) + np.sum(vo == cols['vo'])), n_total=2 * uo.size,
                chg_u=float(np.max(np.abs(cols['uo'] - cols['u']))), chg_v=float(np.max(np.abs(cols['vo'] - cols['v']))),
                nclamp_cols=ncl, full_nlev=int(st_in[0, 3]), full_nwl=int(st_in[:, 4].max()),
                full_ncd=int(st_in[0, 5]), full_maxwl=float(st_in[:, 6].max()), full_bad=int(st_in[:, 9].max()),
                full_maxdu=float(st_out[:, 7].max()), p=p, ncalls=len(st_in))


if __name__ == "__main__":
    for date, it0 in DATES:
        r = run_date(date)
        p = r['p']
        print(f"{date}: columns={r['n']} (12 calls) dU={r['du']:.1e} dV={r['dv']:.1e} exact={r['n_exact']}/{r['n_total']} "
              f"max change U={r['chg_u']:.3f} V={r['chg_v']:.3f}; clamp(wl>wmaxj) hits in sampled cols={r['nclamp_cols']}")
        print(f"   full-grid stats over {r['ncalls']} calls: levels L>=LS1 columns/call={r['full_nlev']} cd_lin={r['full_ncd']} "
              f"wl>wmaxj max over calls={r['full_nwl']} max wl={r['full_maxwl']:.2f} T-bad={r['full_bad']} max|dU|={r['full_maxdu']:.3f}")
    print("params:", {k: v for k, v in p.items() if k not in ('csdragl', 'vsdragl')})
    print("CSDRAGL(LS1:LM):", p['csdragl'][p['ls1'] - 1:], "VSDRAGL(LS1:LM):", p['vsdragl'][p['ls1'] - 1:])
