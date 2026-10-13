"""D230: shadow comparison of the native (JAX RADIA + SOCRATES) radiation against the unmodified Fortran radiation, on the packets the free-running model visited.
   python shadow_compare.py SHADOW_DIR [OUT_JSON] [--date nov26]
SHADOW_DIR holds shadow_<itime>.npz written by d212_day.py with ROCKE_SHADOW_DIR (in_<field> = the packet the native radiation received, out_<field> = what it returned,
seed, itime).  Each packet is sent, in time order, to ONE persistent Fortran radiation server (radiation_server_persist; real RADIA + SOCRATES, unmodified; the server
steps its own clock through the day boundaries) and the outputs are compared field by field.  Same packet, two radiations: no trajectory is compared.
Threshold (SOCRATES report 14.3/14.6): a difference above 1e-6 W/m2 in FSF, SRDN or SRHR is an 'albedo-threshold excursion' (the 5.4 W/m2 single-step event of report 13.1)."""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import radiation_server as rs                       # noqa: E402
import radiation_server_persist as RP               # noqa: E402

FIELDS = [f for f in rs.OUTPUT_FIELDS if f != 'AIJ']
EXC_FIELDS = ('FSF', 'SRDN', 'SRHR')
EXC_TOL = 1e-6


def compare_one(nat, ref):
    row = {}
    for f in FIELDS:
        if ('out_' + f) not in nat or f not in ref:
            continue
        a, b = np.asarray(nat['out_' + f], float), np.asarray(ref[f], float)
        if a.shape != b.shape:
            row[f] = dict(shape_native=list(a.shape), shape_fortran=list(b.shape))
            continue
        d = np.abs(a - b)
        i = int(np.argmax(d))
        row[f] = dict(max_abs=float(d.max()), at=[int(x) for x in np.unravel_index(i, d.shape)], scale=float(np.abs(b).max()),
                      n_unequal=int((a != b).sum()), n=int(a.size), n_above_tol=int((d > EXC_TOL).sum()))
    return row


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    date = sys.argv[sys.argv.index('--date') + 1] if '--date' in sys.argv else 'nov26'
    if date in args:
        args.remove(date)
    sdir = args[0]
    outj = args[1] if len(args) > 1 else os.path.join(sdir, 'shadow_compare.json')
    files = sorted(glob.glob(os.path.join(sdir, 'shadow_*.npz')), key=lambda p: int(os.path.basename(p)[7:-4]))
    print('shadow packets:', len(files), flush=True)
    rows = []
    srv = RP.PersistentServer(date).start()
    try:
        for p in files:
            nat = dict(np.load(p))
            it, seed = int(nat['itime']), int(nat['seed'])
            state = {k: nat['in_' + k] for k in rs.INPUT_FIELDS}
            t0 = time.time()
            ref = srv.request(state, it, seed)
            row = compare_one(nat, ref)
            exc = {f: row[f]['max_abs'] for f in EXC_FIELDS if f in row and 'max_abs' in row[f] and row[f]['max_abs'] > EXC_TOL}
            rows.append(dict(itime=it, day=it // 48, seed=seed, server_s=float(ref.get('_server_s', 0.0)), fields=row, excursion=exc))
            w = max((v.get('max_abs', 0.0) for k, v in row.items() if k in ('FSF', 'SRDN', 'SRHR', 'TRHR', 'SRVISSURF')), default=0.0)
            print('it %d day %d  %.1f s  worst(FSF,SRDN,SRHR,TRHR,SRVISSURF) %.3g  excursion %s' % (it, it // 48, time.time() - t0, w, exc or 'none'), flush=True)
            json.dump(rows, open(outj, 'w'), indent=1)
    finally:
        srv.stop()
    summ = {}
    for f in FIELDS:
        v = [r['fields'][f]['max_abs'] for r in rows if f in r['fields'] and 'max_abs' in r['fields'][f]]
        if v:
            k = int(np.argmax(v))
            summ[f] = dict(max_abs=float(max(v)), itime=rows[k]['itime'])
    json.dump(dict(date=date, n_packets=len(rows), excursion_tol=EXC_TOL, n_excursion_packets=sum(1 for r in rows if r['excursion']), summary=summ, packets=rows),
              open(outj, 'w'), indent=1)
    print('SUMMARY', json.dumps(summ, indent=1))
    print('packets with an excursion (> %g W/m2 in FSF/SRDN/SRHR): %d of %d' % (EXC_TOL, sum(1 for r in rows if r['excursion']), len(rows)))


if __name__ == '__main__':
    main()
