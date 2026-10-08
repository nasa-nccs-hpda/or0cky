"""Validate dyn_aadvt_ff.py / dyn_adv1d_ff.py (AADVT, D99/D100) against real-Fortran per-call dumps.

Dumps (instrumentation/ATMDYN_aadvt.f.patch + QUS_DRV_aadvt.f.patch + ATM_DRV_dynC.f.patch, units 1100-1102), per
date in ff_data/<date>/ (big-endian float64 streams, Fortran order), 6 steps x 2 AADVT calls (k=1,2 = the two
even leapfrog passes of one DYNAM) per date:
  ffd_aadvt_<itime>_c<k>_in.bin    [itime,k,DT,qlimit(0)] MMA T (IM,JM,LM) TMOM (9,IM,JM,LM) MU MV (IM,JM,LM) MW (IM,JM,LM-1)
  ffd_aadvt_<itime>_c<k>_out.bin   [itime,k] MMA T (IM,JM,LM) TMOM (9,IM,JM,LM) FPEU FPEV (IM,JM)
  ffd_aadvt_<itime>_c<k>_s<1|2|3>.bin  [itime,k,stage] RM (IM,JM,LM) RMOM (9,IM,JM,LM) MM (IM,JM,LM) in MASS units,
                                   stage 1 after X sweep 1, 2 after Y, 3 after Z (the final X sweep = out, after the
                                   mass->concentration conversion)
  ffd_aadvt_<itime>_c<k>_ns.bin    [itime,k] NSX (JM,LM,2) per-row Courant nstep of X sweep 1 and 2 (rows 1 and JM are
                                   never advected in x and stay 0), NSZ (IM,JM), as doubles
Usage: python3 dyn_aadvt_compare.py [aadvt|aadvtS4|aadvtS12]
"""
import os
import sys
import numpy as np
from dyn_aadvt_ff import aadvt, IM, JM, LM

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6
NCALL = 2


def _r(raw, o, shape):
    n = int(np.prod(shape))
    return raw[o:o + n].reshape(shape, order='F').copy(), o + n


def fname(date, itime, k, ph, ff=FF_DEFAULT, tag="aadvt"):
    """tag 'aadvt' = real windows; 'aadvtS4'/'aadvtS12' = STRESS runs (MU and MW scaled x4 / x12 inside the AADVT
    call by the helper when env FFD_AADVT_STRESS is set) so that Courant nstep>1 occurs.  The stress runs stopped
    in the real model (aadvtx/aadvtz courmax>1 at nstep=20 -> stop_model) after a few steps; only calls with all
    six files (ns present) are complete."""
    return f"{ff}/{date}/ffd_{tag}_{itime}_c{k}_{ph}.bin"


def stress_calls(tag, ff=FF_DEFAULT):
    out = []
    for d, it0 in DATES:
        for k in range(NSTEP):
            for c in range(1, NCALL + 1):
                if os.path.exists(fname(d, it0 + k, c, 'ns', ff, tag)):
                    out.append((d, it0 + k, c))
    return out


def load_in(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), call=int(raw[1]), dt=raw[2], qlimit=bool(raw[3])); o = 4
    for k, sh in (('mma', (IM, JM, LM)), ('t', (IM, JM, LM)), ('tmom', (9, IM, JM, LM)), ('mu', (IM, JM, LM)),
                  ('mv', (IM, JM, LM)), ('mw', (IM, JM, LM - 1))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_out(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), call=int(raw[1])); o = 2
    for k, sh in (('mma', (IM, JM, LM)), ('t', (IM, JM, LM)), ('tmom', (9, IM, JM, LM)), ('fpeu', (IM, JM)),
                  ('fpev', (IM, JM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_ckpt(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), call=int(raw[1]), stage=int(raw[2])); o = 3
    for k, sh in (('rm', (IM, JM, LM)), ('rmom', (9, IM, JM, LM)), ('mm', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_ns(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), call=int(raw[1])); o = 2
    for k, sh in (('nsx', (JM, LM, 2)), ('nsz', (IM, JM))):
        d[k], o = _r(raw, o, sh)
        d[k] = d[k].astype(int)
    assert o == raw.size
    return d


def available(date, ff=FF_DEFAULT):
    it = dict(DATES)[date] + NSTEP - 1
    return os.path.exists(fname(date, it, NCALL, 'ns', ff)) and os.path.exists(fname(date, it, NCALL, 'out', ff))


def calls():
    return [(d, it0 + k, c) for d, it0 in DATES for k in range(NSTEP) for c in range(1, NCALL + 1)]


def maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def run_aadvt(date, itime, call, ff=FF_DEFAULT, stats=None, tag="aadvt", **kw):
    i = load_in(fname(date, itime, call, 'in', ff, tag))
    o = load_out(fname(date, itime, call, 'out', ff, tag))
    st = {}
    r = aadvt(i['dt'], i['mma'], i['t'], i['tmom'], i['mu'], i['mv'], i['mw'], i['qlimit'], stages=st, stats=stats, **kw)
    return i, o, r, st


def aadvt_compare(i, o, r):
    return dict(mm=maxdiff(r['mm'], o['mma']), t=maxdiff(r['rm'], o['t']), tmom=maxdiff(r['rmom'], o['tmom']),
                fqu=maxdiff(r['fqu'], o['fpeu']), fqv=maxdiff(r['fqv'], o['fpev']),
                n_exact=int(np.sum(r['rm'] == o['t']) + np.sum(r['rmom'] == o['tmom'])),
                n_total=int(o['t'].size + o['tmom'].size))


def stage_compare(date, itime, call, st, ff=FF_DEFAULT, tag="aadvt"):
    d = {}
    for k, ph in (('x1', 's1'), ('y', 's2'), ('z', 's3')):
        c = load_ckpt(fname(date, itime, call, ph, ff, tag))
        d[k] = max(maxdiff(st[k][0], c['rm']), maxdiff(st[k][1], c['rmom']), maxdiff(st[k][2], c['mm']))
    ns = load_ns(fname(date, itime, call, 'ns', ff, tag))
    nsx1, nsx2, nsz = st['ns']
    d['ns_equal'] = bool(np.array_equal(nsx1, ns['nsx'][:, :, 0]) and np.array_equal(nsx2, ns['nsx'][:, :, 1])
                         and np.array_equal(nsz, ns['nsz']))
    return d


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "aadvt"
    worst = {}
    nsx_hist, nsz_hist = {}, {}
    for date, itime, call in (calls() if tag == "aadvt" else stress_calls(tag)):
        i, o, r, st = run_aadvt(date, itime, call, tag=tag)
        d = aadvt_compare(i, o, r); d.update(stage_compare(date, itime, call, st, tag=tag))
        ns = load_ns(fname(date, itime, call, 'ns', FF_DEFAULT, tag))
        for v, c in zip(*np.unique(ns['nsx'][1:JM - 1], return_counts=True)): nsx_hist[int(v)] = nsx_hist.get(int(v), 0) + int(c)
        for v, c in zip(*np.unique(ns['nsz'], return_counts=True)): nsz_hist[int(v)] = nsz_hist.get(int(v), 0) + int(c)
        print(date, itime, call, {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
        for k, v in d.items():
            if isinstance(v, float):
                worst[k] = max(worst.get(k, 0.), v)
    print("worst per field:", worst)
    print("nstep histogram X (rows j=2..JM-1, l=1..LM; both sweeps):", dict(sorted(nsx_hist.items())))
    print("nstep histogram Z (columns incl. poles):", dict(sorted(nsz_hist.items())))
