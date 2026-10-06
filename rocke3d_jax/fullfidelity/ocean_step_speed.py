"""Warm CPU timing of the chained ocean step, per stage (D120). usage: python ocean_step_speed.py [date] [repeats]"""
import sys
import time
import numpy as np
import ocean_chain_io as C
import ocean_step as O
from ocean_step_compare import ctx_for
from ocean_step_chain_compare import fx_step

date = sys.argv[1] if len(sys.argv) > 1 else 'nov26'
nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
d = C.FF_DEFAULT + '/' + date
ctx = ctx_for(d)
it = C.list_steps(d)[1]            # a non-ODIFF step
sn = C.load_step(d, it)
fx = fx_step(sn, it)
O.ocean_step(sn[0], fx, ctx)       # compile / table warm-up
tot = []; per = {}
for _ in range(nrep):
    t0 = time.perf_counter(); s = sn[0]
    for name, fn, _t in O.STAGES:
        t1 = time.perf_counter(); s = fn(s, fx, ctx); per.setdefault(name, []).append(time.perf_counter() - t1)
    tot.append(time.perf_counter() - t0)
print(f'{date} step {it}: warm total {np.mean(tot):.2f} s (min {np.min(tot):.2f}) over {nrep} runs, CPU')
for k, v in per.items():
    print(f'  {k:9s} {np.mean(v):.3f} s')
