"""Speed track: ROCKE_QDYN_VMAP=1 (level-batched y/x sweeps in the QUS moisture advection cycle) against the original
level loop and against the recorded Fortran QDYNAM output, on recorded calls.  CPU, graphcast-env python.
    python speed_qdyn_vmap_test.py [NCALLS=3] [TAG=qdyn]"""
import os, sys, time
import numpy as np
import dyn_qdynam_io as io
import dyn_jax_qdynam as Q
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
tag = sys.argv[2] if len(sys.argv) > 2 else 'qdyn'
calls = io.calls(tag)[:n]
for date, it in calls:
    g = io.load_geom(io.gname(date, tag="qdyn"))
    i = io.load_in(io.fname(date, it, 'in', tag=tag))
    fin = io.load_fin(io.fname(date, it, 'fin', tag=tag))
    res = {}
    for mode in ('0', '1'):
        os.environ['ROCKE_QDYN_VMAP'] = mode
        kit = Q.QdynamKit(g)
        t = time.perf_counter()
        r = kit(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'])
        t1 = time.perf_counter() - t
        t = time.perf_counter()
        r = kit(i['q'], i['qmom'], i['maold'], i['mu'], i['mv'], i['mw'])
        res[mode] = r
        print(f'{date} {it} vmap={mode}: first {t1:.1f} s, second {time.perf_counter()-t:.2f} s (CPU)', flush=True)
    a, b = res['0'], res['1']
    print('  orig vs vmap: q equal', np.array_equal(a['q'], b['q']), 'qmom equal', np.array_equal(a['qmom'], b['qmom']),
          '| vs Fortran fin: orig q exact', int(np.sum(a['q'] == fin['q'])), '/', fin['q'].size, ' vmap q exact', int(np.sum(b['q'] == fin['q'])))
