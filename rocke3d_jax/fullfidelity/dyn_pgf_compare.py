"""Validate dyn_pgf_ff.py (PGF, D97) against real-Fortran per-call dumps.

Dumps (instrumentation/ATMDYN_aflux_pgf_advecv.f.patch + ATM_DRV_dynB.f.patch, unit 1084), per date in
ff_data/<date>/ (big-endian float64, Fortran order), 6 steps x 5 passes (see dyn_aflux_compare.py for k):
  ffd_pgf_<itime>_p<k>_in.bin   [itime,pass,DT1,MRCH] MAM (LM,IM,JM) MAFTER (LM,IM,JM) S0 SZ UT VT DUT DVT (IM,JM,LM)
  ffd_pgf_<itime>_p<k>_out.bin  [itime,pass] UT VT DUT DVT GZ PHI SPA(=AdM) PGFU0 (before AVRX; 0 outside the
                                computed rows) PGFU (after AVRX), all (IM,JM,LM)
Geometry/constants come from ffd_aflux_geom.bin (D96).

Usage: python3 dyn_pgf_compare.py [--imf-pow]
"""
import sys
import numpy as np
from dyn_aflux_ff import IM, JM, LM, avrx_tables
from dyn_aflux_compare import (FF_DEFAULT, DATES, NSTEP, NPASS, load_geom_date, fname, _r, maxdiff, calls)
from dyn_pgf_ff import pgf


def load_pgf_in(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1]), dt1=raw[2], mrch=int(raw[3])); o = 4
    for k, sh in (('mam', (LM, IM, JM)), ('mafter', (LM, IM, JM)), ('s0', (IM, JM, LM)), ('sz', (IM, JM, LM)),
                  ('ut', (IM, JM, LM)), ('vt', (IM, JM, LM)), ('dut', (IM, JM, LM)), ('dvt', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_pgf_out(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1])); o = 2
    for k in ('ut', 'vt', 'dut', 'dvt', 'gz', 'phi', 'adm', 'pgfu0', 'pgfu'):
        d[k], o = _r(raw, o, (IM, JM, LM))
    assert o == raw.size
    return d


def run_pgf(date, itime, pas, g=None, tab=None, imf_pow=False, ff=FF_DEFAULT, **kw):
    g = g or load_geom_date(date, ff)
    i = load_pgf_in(fname(date, itime, pas, 'pgf', ff) + "_in.bin")
    o = load_pgf_out(fname(date, itime, pas, 'pgf', ff) + "_out.bin")
    r = pgf(i['dt1'], i['mam'], i['ut'], i['vt'], i['mafter'], i['s0'], i['sz'], i['dut'], i['dvt'], g,
            tab=tab, imf_pow=imf_pow, **kw)
    return i, o, r


def pgf_compare(i, o, r):
    d = {}
    d['gz'] = maxdiff(r['gz'], o['gz'])
    d['phi'] = maxdiff(r['phi'], o['phi'])
    d['adm'] = maxdiff(r['adm'], o['adm'])
    d['pgfu0'] = maxdiff(r['pgfu0'][:, 1:, :], o['pgfu0'][:, 1:, :])
    d['pgfu'] = maxdiff(r['pgfu'][:, 1:, :], o['pgfu'][:, 1:, :])
    d['dut'] = maxdiff(r['dut'], o['dut'])
    d['dvt'] = maxdiff(r['dvt'], o['dvt'])
    d['ut'] = maxdiff(r['ut'], o['ut'])
    d['vt'] = maxdiff(r['vt'], o['vt'])
    return d


def available(date, ff=FF_DEFAULT):
    import os
    return os.path.exists(fname(date, dict(DATES)[date] + NSTEP - 1, NPASS, 'pgf', ff) + "_out.bin")


if __name__ == "__main__":
    imf = "--imf-pow" in sys.argv
    worst = {}
    for date, it0 in DATES:
        g = load_geom_date(date); tab = avrx_tables(g)
        for k in range(NSTEP):
            for p in range(1, NPASS + 1):
                i, o, r = run_pgf(date, it0 + k, p, g, tab, imf_pow=imf)
                d = pgf_compare(i, o, r)
                print(date, it0 + k, p, 'mrch', i['mrch'], ' '.join(f"{a}={b:.1e}" for a, b in d.items()))
                for a, b in d.items():
                    worst[a] = max(worst.get(a, 0.), b)
    print("WORST", {a: f"{b:.2e}" for a, b in worst.items()})
