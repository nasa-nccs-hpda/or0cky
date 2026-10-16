"""Validate dyn_aadvq_ff.py (AADVQ0, AADVQ + sweeps, QDYNAM; D103-D106) against real-Fortran per-call dumps.

Dump layout: see dyn_qdynam_io.py (ffd_qdyn_<itime>_{in,q0,ain,ck,out,fin}.bin, ffd_qdyn_geom.bin; stress runs use the
prefix ffd_qdynS<f>[<comp>]_).  1 QDYNAM call per physics step, 6 steps x 3 dates (nov26 33312, dec01 33552, jan01
17520).  Everything is replayed from the recorded INPUT of the call and compared with every recorded intermediate:
  q0   : the AADVQ0 outputs (ncyc, ncycxy, nstepx, nstepz_extra, lminzij, lmaxzij, scaled MUs/MVs/MWs, pv_south,
         checkflux tables, mw_extra on its defined cells)
  ain  : mass-unit state at AADVQ entry
  ck   : every instrumented stage inside AADVQ (after y-checkflux, Y, X per (cycle,sub-cycle,level); after z-checkflux;
         the vertical carry after each AADVQZ; full state after the L loop and after the extra z advection)
  out  : AADVQ outputs (RM, RMOM, MMA, the six zonal diagnostics, SCF3D);  fin : QDYNAM outputs (Q, QMOM).
Usage: python3 dyn_qdynam_compare.py [qdyn|qdynS8|...] [--limit N]
"""
import sys
import numpy as np
import dyn_aadvq_ff as aq
import dyn_qdynam_io as io
from dyn_qdynam_io import IM, JM, LM


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0


class CkComparer:
    """Hook for aadvq: compares each instrumented stage with the next record of the ck stream."""

    def __init__(self, recs):
        self.recs = recs
        self.k = 0
        self.worst = {}
        self.count = {}
        self.mismatch_records = 0

    def __call__(self, stage, nc, ncxy, l, payload):
        r = self.recs[self.k]; self.k += 1
        assert (r['stage'], r['nc'], r['ncxy'], r['l']) == (stage, nc, ncxy, l), ((r['stage'], r['nc'], r['ncxy'], r['l']), (stage, nc, ncxy, l))
        if stage in (0, 3):
            d = max(maxdiff(payload[0], r['a']), maxdiff(payload[1], r['b']))
        elif stage in (1, 2, 6, 7):
            d = max(maxdiff(payload[0], r['rm']), maxdiff(payload[1], r['rmom']), maxdiff(payload[2], r['mma']))
        else:
            d = max(maxdiff(payload[0], r['mwdn']), maxdiff(payload[1], r['fdn']), maxdiff(payload[2], r['fdn0']),
                    maxdiff(payload[3], r['fmomdn']))
        self.worst[stage] = max(self.worst.get(stage, 0.), d)
        self.count[stage] = self.count.get(stage, 0) + 1
        if d != 0.:
            self.mismatch_records += 1


def q0_compare(r, q):
    """Compare an aadvq0 result with the recorded q0 dump (mw_extra only on its defined cells)."""
    d = {}
    d['ncyc'] = int(r['ncyc'] != q['ncyc'])
    d['do_z'] = int(bool(r['do_z_extra']) != bool(q['do_z_extra']))
    d['ncycxy'] = int(np.sum(r['ncycxy'] != q['ncycxy']))
    d['nstepx'] = int(np.sum(r['nstepx'][1:JM - 1] != q['nstepx'][1:JM - 1]))
    d['nstepz'] = int(np.sum(r['nstepz_extra'] != q['nstepz_extra']))
    d['lmin'] = int(np.sum(r['lminzij'] != q['lminzij']))
    d['lmax'] = int(np.sum(r['lmaxzij'] != q['lmaxzij']))
    d['ni_y'] = int(np.sum(r['ni_y'][1:JM - 1] != q['ni_y'][1:JM - 1]))
    d['ni_z'] = int(np.sum(r['ni_z'] != q['ni_z']))
    ly = sorted((j, l, i) for (j, l), lst in r['i_y'].items() for i in lst)
    lz = sorted((j, l, i) for (j, l), lst in r['i_z'].items() for i in lst)
    qy = sorted(map(tuple, q['list_y'].tolist())); qz = sorted(map(tuple, q['list_z'].tolist()))
    d['list_y'] = int(ly != [t for t in qy if 2 <= t[0] <= JM - 1])
    d['list_z'] = int(lz != qz)
    for k in ('mu', 'mv', 'mw', 'pv_south'):
        d[k] = maxdiff(r[k], q[k])
    # mw_extra on defined cells: (i,j) with nstepz_extra>0, l in lmin..lmax-1
    me = 0.
    for j in range(JM):
        for i in range(IM):
            if q['nstepz_extra'][i, j] > 0:
                lo, hi = int(q['lminzij'][i, j]), int(q['lmaxzij'][i, j]) - 1
                me = max(me, maxdiff(r['mw_extra'][i, j, lo - 1:hi], q['mw_extra'][i, j, lo - 1:hi]))
    d['mw_extra'] = me
    return d


def run_call(date, itime, ff=io.FF_DEFAULT, tag="qdyn", stats=None, geom=None):
    g = geom if geom is not None else io.load_geom(io.gname(date, ff, tag if tag == "qdyn" else "qdyn"))
    i = io.load_in(io.fname(date, itime, 'in', ff, tag))
    q0 = io.load_q0(io.fname(date, itime, 'q0', ff, tag))
    ain = io.load_ain(io.fname(date, itime, 'ain', ff, tag))
    out = io.load_out(io.fname(date, itime, 'out', ff, tag))
    fin = io.load_fin(io.fname(date, itime, 'fin', ff, tag))
    ck = io.load_ck(io.fname(date, itime, 'ck', ff, tag))
    cc = CkComparer(ck)
    res = aq.qdynam(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'], g['axyp'], g['imaxj'], g['kg2mb'],
                    g['byim_geom'], g['byim_qus'], stats, cc)
    assert cc.k == len(ck), (cc.k, len(ck))
    d = dict(mb=maxdiff(res['mb'], i['mb']))
    d.update({'q0_' + k: v for k, v in q0_compare(res['q0'], q0).items()})
    d['ain'] = max(maxdiff(res['ain'][0], ain['rm']), maxdiff(res['ain'][1], ain['rmom']))
    a = res['aad']
    d['out_rm'] = maxdiff(a['rm'], out['rm']); d['out_rmom'] = maxdiff(a['rmom'], out['rmom']); d['out_mma'] = maxdiff(a['mma'], out['mma'])
    for k in ('sbf', 'sbm', 'sfbm', 'scf', 'scm', 'sfcm', 'scf3d'):
        d['out_' + k] = maxdiff(a[k], out[k])
    d['fin_q'] = maxdiff(res['q'], fin['q']); d['fin_qmom'] = maxdiff(res['qmom'], fin['qmom'])
    d['n_exact'] = int(np.sum(res['q'] == fin['q']) + np.sum(res['qmom'] == fin['qmom']))
    d['n_total'] = int(fin['q'].size + fin['qmom'].size)
    for s in sorted(cc.worst):
        d[f'ck{s}'] = cc.worst[s]
    d['ck_records'] = len(ck)
    return d, dict(i=i, q0=q0, out=out, fin=fin, ck=ck, res=res)


if __name__ == "__main__":
    tag = "qdyn"; lim = None
    args = sys.argv[1:]
    if args and not args[0].startswith('--'):
        tag = args.pop(0)
    if '--limit' in args:
        lim = int(args[args.index('--limit') + 1])
    worst = {}
    stats = {}
    n = 0
    for date, itime in io.calls(tag):
        if lim is not None and n >= lim:
            break
        n += 1
        d, _ = run_call(date, itime, tag=tag, stats=stats)
        bad = {k: v for k, v in d.items() if isinstance(v, (int, float)) and k not in ('n_exact', 'n_total', 'ck_records') and v != 0}
        print(date, itime, 'ALL ZERO' if not bad else bad, 'exact', d['n_exact'], '/', d['n_total'], 'ck', d['ck_records'], flush=True)
        for k, v in d.items():
            if isinstance(v, (int, float)) and k not in ('n_exact', 'n_total', 'ck_records'):
                worst[k] = max(worst.get(k, 0), v)
    print("calls:", n, "worst per field (nonzero only):", {k: v for k, v in worst.items() if v})
    print("branch statistics (summed over calls):", dict(sorted(stats.items())))
