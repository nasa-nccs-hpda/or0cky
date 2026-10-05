"""Validate dyn_advecv_ff.py (ADVECV, D98) against real-Fortran per-call dumps.

Dumps (instrumentation/MOMEN2ND_advecv.f.patch + ATM_DRV_dynB.f.patch, unit 1085), per date in ff_data/<date>/
(big-endian float64, Fortran order), 6 steps x 5 passes (k as in dyn_aflux_compare.py):
  ffd_advecv_<itime>_p<k>_in.bin   [itime,pass,DT1,MRCH] U V (IM,JM,LM) MMEAN MBEFOR MAFTER (LM,IM,JM) UT VT (IM,JM,LM)
                                   then module MU MV (IM,JM,LM) MW (IM,JM,LM-1) SPA (IM,JM,LM) of this pass's AFLUX
  ffd_advecv_<itime>_p<k>_out.bin  [itime,pass] UT VT (IM,JM,LM)
Geometry/constants from ffd_aflux_geom.bin (D96).  Usage: python3 dyn_advecv_compare.py
"""
import os
import numpy as np
from dyn_aflux_ff import IM, JM, LM
from dyn_aflux_compare import (FF_DEFAULT, DATES, NSTEP, NPASS, load_geom_date, fname, _r, maxdiff, calls)
from dyn_advecv_ff import advecv


def load_advecv_in(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1]), dt1=raw[2], mrch=int(raw[3])); o = 4
    for k, sh in (('u', (IM, JM, LM)), ('v', (IM, JM, LM)), ('mmean', (LM, IM, JM)), ('mbefor', (LM, IM, JM)),
                  ('mafter', (LM, IM, JM)), ('ut', (IM, JM, LM)), ('vt', (IM, JM, LM)), ('mu', (IM, JM, LM)),
                  ('mv', (IM, JM, LM)), ('mw', (IM, JM, LM - 1)), ('spa', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_advecv_out(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1])); o = 2
    for k in ('ut', 'vt'):
        d[k], o = _r(raw, o, (IM, JM, LM))
    assert o == raw.size
    return d


def available(date, ff=FF_DEFAULT):
    return os.path.exists(fname(date, dict(DATES)[date] + NSTEP - 1, NPASS, 'advecv', ff) + "_out.bin")


def run_advecv(date, itime, pas, g=None, ff=FF_DEFAULT, stages=None, **kw):
    g = g or load_geom_date(date, ff)
    i = load_advecv_in(fname(date, itime, pas, 'advecv', ff) + "_in.bin")
    o = load_advecv_out(fname(date, itime, pas, 'advecv', ff) + "_out.bin")
    ut, vt = advecv(i['dt1'], i['u'], i['v'], i['mmean'], i['mbefor'], i['ut'], i['vt'], i['mafter'],
                    i['mu'], i['mv'], i['mw'], i['spa'], g, stages=stages, **kw)
    return i, o, dict(ut=ut, vt=vt)


def advecv_compare(i, o, r):
    return dict(ut=maxdiff(r['ut'], o['ut']), vt=maxdiff(r['vt'], o['vt']),
                ut_rows2_JM_n_exact=int(np.sum(r['ut'][:, 1:, :] == o['ut'][:, 1:, :])),
                n_total=int(r['ut'][:, 1:, :].size))


if __name__ == "__main__":
    worst = {}
    for date, it0 in DATES:
        g = load_geom_date(date)
        for k in range(NSTEP):
            for p in range(1, NPASS + 1):
                i, o, r = run_advecv(date, it0 + k, p, g)
                d = advecv_compare(i, o, r)
                print(date, it0 + k, p, 'mrch', i['mrch'], f"ut={d['ut']:.1e} vt={d['vt']:.1e} exact_ut={d['ut_rows2_JM_n_exact']}/{d['n_total']}")
                for a in ('ut', 'vt'):
                    worst[a] = max(worst.get(a, 0.), d[a])
    print("WORST", {a: f"{b:.2e}" for a, b in worst.items()})
