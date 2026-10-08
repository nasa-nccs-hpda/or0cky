"""D190: jax_advsi vs advsi_ff.advsi (NumPy, mpmath REAL*16 Ti2b) and vs the real ffadv_out dumps.  usage: d190_test_advsi.py [calls_per_date] [stride]"""
import sys, time, glob, os
import d190_util as U
import numpy as np, jax, jax.numpy as jnp
import advsi_ff as A
import jax_advsi as JA

nper = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/advsi_dumps'
tot = {}
first = True
for date in ('nov26', 'dec01', 'jan01'):
    its = sorted(int(os.path.basename(p)[9:-4]) for p in glob.glob(f'{FF}/{date}/ffadv_in_*.bin'))
    pick = its[::max(1, len(its) // nper)][:nper]
    d0, _ = A.read_dump(f'{FF}/{date}/ffadv_in_{its[0]}.bin')
    K = JA.make_static(d0['focean'], d0['geo'])
    f = jax.jit(lambda st, ausi, avsi: JA.advsi(K, st, ausi, avsi))
    for it in pick:
        d, o = A.read_dump(f'{FF}/{date}/ffadv_in_{it}.bin', f'{FF}/{date}/ffadv_out_{it}.bin')
        st = {k: d[k] for k in ('rsi', 'rsix', 'rsiy', 'rsisave', 'msi', 'snowi', 'hsi', 'ssi')}
        t0 = time.perf_counter()
        new_np, out_np = A.advsi(st, d['ausi'], d['avsi'], d['focean'], d['geo'])
        t_np = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = f(U.to_dev(st), jnp.asarray(d['ausi']), jnp.asarray(d['avsi'])); U.sync(r)
        t1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = f(U.to_dev(st), jnp.asarray(d['ausi']), jnp.asarray(d['avsi'])); U.sync(r)
        t_jx = time.perf_counter() - t0
        new_j, out_j = U.to_np(r[0]), U.to_np(r[1])
        bad = []
        n = 0
        for k, v in new_np.items():
            c = U.cat(new_j[k], v); n += 1
            if c[0] != 'A': bad.append(('new.' + k, c))
        for k, v in out_np.items():
            c = U.cat(out_j[k], v); n += 1
            if c[0] != 'A': bad.append(('out.' + k, c))
        # vs the real output (pole rows i>1 stale in the Fortran: not compared)
        oc = d['focean'] > 0; oc = oc.copy(); oc[1:, 0] = False; oc[1:, -1] = False
        real = {}
        for k in ('rsi', 'msi', 'snowi', 'hsi', 'ssi', 'rsix', 'rsiy'):
            m = oc[..., None] if o[k].ndim == 3 else oc
            real[k] = U.cat(np.where(m, new_j[k], 0.0), np.where(m, o[k], 0.0))
        print(date, it, 'np %.2fs jax first %.1fs steady %.4fs' % (t_np, t1, t_jx), 'compared', n, 'not-A vs numpy:', bad if bad else 'none', '| vs real out nonA:', {k: v for k, v in real.items() if v[0] != 'A'} or 'none', flush=True)
