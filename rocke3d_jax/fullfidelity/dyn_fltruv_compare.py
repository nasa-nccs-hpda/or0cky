"""Validate dyn_fltruv_ff.py against real-Fortran dumps (D90).

Dumps (instrumentation/ATMDYN_fltruv.f.patch + ATM_DRV_fltruv.f.patch), per date in
ff_data/<date>/:  ffd_fltruv_geom.bin (once) and, per DYNAM call (1 per step, 6 steps),
ffd_fltruv_<itime>_{in,flt,ny,out}.bin.  All big-endian float64 streams.

Usage: python3 dyn_fltruv_compare.py [--analytic]
  default : geometry (DXYN, DXYS, COSV, RADIUS, OMEGA) taken from the recorded dump.
  --analytic : geometry recomputed analytically from GEOM_B.f (RADIUS, OMEGA still read
               from the dump header since they are runtime planet parameters).
"""
import sys
import numpy as np
from dyn_fltruv_ff import (IM, JM, LM, geometry, fltruv, fltry2, conserv_amb_ext,
                           add_am_as_solidbody_rotation, filter_chain)

FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6


def _f3(flat):      # Fortran (IM,JM,LM) -> numpy (IM,JM,LM)
    return flat.reshape(LM, JM, IM).transpose(2, 1, 0).copy()


def load_geom(path):
    raw = np.fromfile(path, dtype='>f8')
    assert int(raw[0]) == JM and raw.size == 3 + 3 * JM
    return dict(radius=raw[1], omega=raw[2], dxyn=raw[3:3 + JM],
                dxys=raw[3 + JM:3 + 2 * JM], cosv=raw[3 + 2 * JM:3 + 3 * JM])


def load_in(path):
    raw = np.fromfile(path, dtype='>f8')
    n = IM * JM * LM
    assert raw.size == 1 + 3 * n + IM * JM, raw.size
    u = _f3(raw[1:1 + n]); v = _f3(raw[1 + n:1 + 2 * n])
    ma = raw[1 + 2 * n:1 + 3 * n].reshape(JM, IM, LM).transpose(2, 1, 0).copy()  # (LM,IM,JM)
    masum = raw[1 + 3 * n:].reshape(JM, IM).T.copy()
    return dict(itime=raw[0], u=u, v=v, ma=ma, masum=masum)


def load_uv(path):
    raw = np.fromfile(path, dtype='>f8')
    n = IM * JM * LM
    assert raw.size == 1 + 2 * n, raw.size
    return dict(itime=raw[0], u=_f3(raw[1:1 + n]), v=_f3(raw[1 + n:]))


def load_ny(path):
    raw = np.fromfile(path, dtype='>f8')
    n = IM * JM * LM
    assert raw.size == 2 + IM * JM + 2 * n, raw.size
    am1 = raw[2:2 + IM * JM].reshape(JM, IM).T.copy()
    o = 2 + IM * JM
    return dict(itime=raw[0], damsum=raw[1], am1=am1, u=_f3(raw[o:o + n]),
                v=_f3(raw[o + n:o + 2 * n]))


def load_call(date, itime, ff=FF_DEFAULT):
    d = f"{ff}/{date}/ffd_fltruv_{itime}"
    return (load_in(d + "_in.bin"), load_uv(d + "_flt.bin"), load_ny(d + "_ny.bin"),
            load_uv(d + "_out.bin"))


def available(date, ff=FF_DEFAULT):
    import os
    return os.path.exists(f"{ff}/{date}/ffd_fltruv_geom.bin")


def run_call(date, itime, analytic=False, ff=FF_DEFAULT):
    """Returns dict of max-abs differences per stage/field and the reference ranges."""
    geo = load_geom(f"{ff}/{date}/ffd_fltruv_geom.bin")
    omega = geo['omega']
    if analytic:
        geo = geometry(geo['radius'])
    rin, rflt, rny, rout = load_call(date, itime, ff)
    u, v, damsum, st = filter_chain(rin['u'], rin['v'], rin['ma'], rin['masum'], geo,
                                    omega, return_stages=True)
    res = {}

    def cmp(name, a, b):
        res[name] = float(np.max(np.abs(a - b)))
    cmp('flt_u', st['flt'][0], rflt['u']); cmp('flt_v', st['flt'][1], rflt['v'])
    cmp('am1', st['am1'], rny['am1'])
    # 'ny' stage = after both fltry2 calls (before solid-body fix)
    cmp('ny_u', st['ny'][0], rny['u']); cmp('ny_v', st['ny'][1], rny['v'])
    res['damsum'] = float(abs(damsum - rny['damsum']))
    res['damsum_rel'] = res['damsum'] / abs(rny['damsum']) if rny['damsum'] != 0 else 0.
    cmp('out_u', u, rout['u']); cmp('out_v', v, rout['v'])
    res['n_out_u_exact'] = int(np.sum(u == rout['u']))
    res['n_total'] = u.size
    # how much the chain changed the field (non-vacuity)
    res['chg_u'] = float(np.max(np.abs(rout['u'] - rin['u'])))
    res['chg_v'] = float(np.max(np.abs(rout['v'] - rin['v'])))
    res['scale_u'] = float(np.max(np.abs(rin['u'])))
    return res


def geometry_check(date, ff=FF_DEFAULT):
    g = load_geom(f"{ff}/{date}/ffd_fltruv_geom.bin")
    a = geometry(g['radius'])
    return {k: float(np.max(np.abs(a[k] - g[k]) / np.maximum(np.abs(g[k]), 1e-300)))
            for k in ('dxyn', 'dxys', 'cosv')}


if __name__ == "__main__":
    analytic = "--analytic" in sys.argv
    worst = {}
    for date, it0 in DATES:
        print(f"== {date} ==  geometry analytic-vs-dump max rel diff: {geometry_check(date)}")
        for k in range(NSTEP):
            r = run_call(date, it0 + k, analytic)
            print(f"  itime={it0 + k}: " + " ".join(
                f"{n}={r[n]:.2e}" for n in ('flt_u', 'flt_v', 'am1', 'ny_u', 'ny_v',
                                              'damsum_rel', 'out_u', 'out_v'))
                  + f"  exact_u={r['n_out_u_exact']}/{r['n_total']}  chg_u={r['chg_u']:.2e}")
            for n in ('flt_u', 'flt_v', 'am1', 'ny_u', 'ny_v', 'damsum_rel', 'out_u', 'out_v'):
                worst[n] = max(worst.get(n, 0.), r[n])
    print("WORST over all calls:", {k: f"{v:.2e}" for k, v in worst.items()})
