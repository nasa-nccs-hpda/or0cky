"""Validate dyn_aflux_ff.py (AFLUX, ADVECM, MAtoP, AVRX) against real-Fortran per-call dumps (D96).

Dumps (instrumentation/ATMDYN_aflux_pgf_advecv.f.patch + ATM_DRV_dynB.f.patch, unit 1081/1082/1083), per
date in ff_data/<date>/ (all big-endian float64 streams, Fortran order):
  ffd_aflux_geom.bin                       once: constants, geometry, ZATMO, topography patches
  ffd_aflux_<itime>_p<k>_in.bin            AFLUX in : [itime,pass,NS,DT,MRCH] U V (IM,JM,LM) MA (LM,IM,JM)
                                           MASUM (IM,JM) ME (LM,IM,JM) MESUM (IM,JM)
  ffd_aflux_<itime>_p<k>_out.bin           AFLUX out: [itime,pass] MU MV (IM,JM,LM) MW (IM,JM,LM-1) CONV SPA
                                           SPA0 (SPA before AVRX; rows 2..JM-1 only)
  ffd_aflux_advecm_<itime>_p<k>_in.bin     ADVECM in : [itime,pass,DT1,MRCH] MOLD (LM,IM,JM) CONV MW
  ffd_aflux_advecm_<itime>_p<k>_out.bin    ADVECM(+MAtoP) out: [itime,pass] MNEW (LM,IM,JM) MSUM (IM,JM)
                                           PEDN PMID PDSIG PK (LM,IM,JM) P (IM,JM)
k = leapfrog pass within DYNAM (1 fwd MRCH=0, 2 bwd -1, 3 even 2, 4 odd -2, 5 even 2); 6 steps x 5 passes
per date.

Usage: python3 dyn_aflux_compare.py [--recorded-avrx] [--imf-pow]
  --imf-pow: PK = PMID**KAPA through the Intel libimf pow (bitwise; needs the Intel runtime)
"""
import os
import sys
import numpy as np
from dyn_aflux_ff import (IM, JM, LM, load_geom, aflux, advecm, matop, avrx_tables, avrx_field)

FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6
NPASS = 5


def _r(raw, o, shape):
    n = int(np.prod(shape))
    return raw[o:o + n].reshape(shape, order='F').copy(), o + n


def load_aflux_in(path):
    raw = np.fromfile(path, dtype='>f8')
    h = raw[:5]; o = 5
    d = dict(itime=int(h[0]), pas=int(h[1]), ns=int(h[2]), dt=h[3], mrch=int(h[4]))
    for k, sh in (('u', (IM, JM, LM)), ('v', (IM, JM, LM)), ('ma', (LM, IM, JM)), ('masum', (IM, JM)),
                  ('me', (LM, IM, JM)), ('mesum', (IM, JM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_aflux_out(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1])); o = 2
    for k, sh in (('mu', (IM, JM, LM)), ('mv', (IM, JM, LM)), ('mw', (IM, JM, LM - 1)), ('conv', (IM, JM, LM)),
                  ('spa', (IM, JM, LM)), ('spa0', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_advecm_in(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1]), dt1=raw[2], mrch=int(raw[3])); o = 4
    for k, sh in (('mold', (LM, IM, JM)), ('conv', (IM, JM, LM)), ('mw', (IM, JM, LM - 1))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_advecm_out(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), pas=int(raw[1])); o = 2
    for k, sh in (('mnew', (LM, IM, JM)), ('msum', (IM, JM)), ('pedn', (LM, IM, JM)), ('pmid', (LM, IM, JM)),
                  ('pdsig', (LM, IM, JM)), ('pk', (LM, IM, JM)), ('p', (IM, JM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def fname(date, itime, pas, kind, ff=FF_DEFAULT):
    return f"{ff}/{date}/ffd_{kind}_{itime}_p{pas}"


def load_geom_date(date, ff=FF_DEFAULT):
    return load_geom(f"{ff}/{date}/ffd_aflux_geom.bin")


def available(date, ff=FF_DEFAULT):
    return os.path.exists(f"{ff}/{date}/ffd_aflux_geom.bin") and \
        os.path.exists(fname(date, dict(DATES)[date] + NSTEP - 1, NPASS, 'aflux', ff) + "_out.bin")


def calls(ff=FF_DEFAULT):
    return [(d, it0 + k, p) for d, it0 in DATES for k in range(NSTEP) for p in range(1, NPASS + 1)]


def maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def run_aflux(date, itime, pas, g=None, recorded_avrx=False, ff=FF_DEFAULT, stats=None, tab=None, **kw):
    g = g or load_geom_date(date, ff)
    i = load_aflux_in(fname(date, itime, pas, 'aflux', ff) + "_in.bin")
    o = load_aflux_out(fname(date, itime, pas, 'aflux', ff) + "_out.bin")
    r = aflux(i['ns'], i['u'], i['v'], i['ma'], i['masum'], i['me'], i['mesum'], g,
              avrx_post=(o['spa'] if recorded_avrx else None), stats=stats, tab=tab, **kw)
    return i, o, r


# rows of each output that Fortran actually writes (the rest keep stale/zero module values)
def aflux_compare(i, o, r):
    d = {}
    d['mu'] = maxdiff(r['mu'], o['mu'])
    d['mv_J2..JM'] = maxdiff(r['mv'][:, 1:, :], o['mv'][:, 1:, :])
    d['mv_J1_unwritten_real'] = float(np.max(np.abs(o['mv'][:, 0, :])))
    d['mw'] = maxdiff(r['mw'], o['mw'])
    d['conv'] = maxdiff(r['conv'], o['conv'])
    d['spa'] = maxdiff(r['spa'], o['spa'])
    d['spa0'] = maxdiff(r['spa0'][:, 1:JM - 1, :], o['spa0'][:, 1:JM - 1, :])
    return d


def run_advecm(date, itime, pas, g=None, ff=FF_DEFAULT, imf_pow=False):
    g = g or load_geom_date(date, ff)
    i = load_advecm_in(fname(date, itime, pas, 'aflux_advecm', ff) + "_in.bin")
    o = load_advecm_out(fname(date, itime, pas, 'aflux_advecm', ff) + "_out.bin")
    r = advecm(i['dt1'], i['mold'], i['conv'], i['mw'], g, imf_pow=imf_pow)
    return i, o, r


def advecm_compare(o, r):
    return {k: maxdiff(r[k], o[k]) for k in ('mnew', 'msum', 'pedn', 'pmid', 'pdsig', 'pk', 'p')}


if __name__ == "__main__":
    rec = "--recorded-avrx" in sys.argv
    imf = "--imf-pow" in sys.argv
    worst = {}
    for date, it0 in DATES:
        g = load_geom_date(date)
        tab = avrx_tables(g)
        for k in range(NSTEP):
            for p in range(1, NPASS + 1):
                it = it0 + k
                i, o, r = run_aflux(date, it, p, g, recorded_avrx=rec, tab=tab)
                d = aflux_compare(i, o, r)
                im_, om_, rm_ = run_advecm(date, it, p, g, imf_pow=imf)
                d.update({'M_' + a: b for a, b in advecm_compare(om_, rm_).items()})
                print(date, it, p, 'mrch', i['mrch'], 'ns', i['ns'], ' '.join(f"{a}={b:.1e}" for a, b in d.items()))
                for a, b in d.items():
                    worst[a] = max(worst.get(a, 0.), b)
    print("WORST", {a: f"{b:.2e}" for a, b in worst.items()})
