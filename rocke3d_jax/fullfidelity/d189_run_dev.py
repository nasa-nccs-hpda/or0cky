"""D189: run the device MSTCNV (clouds_mstcnv_dev) stand-alone on the MSTCNV arguments captured from the real step-0 CONDSE of a date (d189_profile.py)
and compare ALL outputs bitwise with the existing libimf implementation (clouds_mstcnv_jax + D183 install), whose outputs the profile run saved.
Reports timing (cold, steady), fused-callback counters, QUS counters, jit executions and host syncs.
Usage: taskset -c 0-2 env OMP_NUM_THREADS=1 python d189_run_dev.py nov26 [libimf|libm]"""
import clouds_jax_env  # noqa: F401
import sys
import time

import numpy as np

import d189_common as C

date = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'libimf'
import os
BK = tuple(int(v) for v in os.environ['D189_BUCKETS'].split(',')) if os.environ.get('D189_BUCKETS') else None
tag = os.environ.get('D189_TAG', '')
import clouds_condse_ff as cf
cf.set_backend('imf')            # constants of make_K (mc._pow) as in the baseline run
import libimf_fused as FX
import clouds_mstcnv_dev as md
FX.set_mode(mode)
print('fused status', FX.status(), flush=True)
b = C.load(f'base_{date}.pkl')
R, c, base = b['R'], b['c'], b['out']
FX.reset_counters(); md.reset_qus()
t0 = time.perf_counter()
st = {}
o = md.mstcnv_dev(R, c, stats=st, **({'buckets': BK} if BK else {}))
cold = time.perf_counter() - t0
print(f'cold {cold:.1f} s  events {st["events"]}  per lmin {st["events_per_lmin"].tolist()}', flush=True)
FX.reset_counters(); md.reset_qus()
t0 = time.perf_counter()
o = md.mstcnv_dev(R, c, **({'buckets': BK} if BK else {}))
steady = time.perf_counter() - t0
cn = FX.counters()
print(f'steady {steady:.2f} s   (baseline steady {b["steady"]:.2f} s)')
print('fused total', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cn['total'].items()})
for t, d in sorted(cn['tags'].items(), key=lambda kv: -kv[1]['seconds']):
    print(f'   {t:14s} calls {d["calls"]:6d} lanes {d["lanes"]:9d} MB_in {d["bytes_in"]/1e6:8.1f} s {d["seconds"]:6.2f}')
print('QUS', md.QUS)
nf, neq, bad = C.compare_dicts(o, base)
print(f'outputs compared {nf}, bitwise equal {neq}, unequal {len(bad)}')
for x in bad[:20]:
    print('   UNEQUAL', x)
C.dump(dict(out=o, cold=cold, steady=steady, counters=cn, qus=dict(md.QUS), stats={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in st.items()}), f'dev_{date}_{mode}{tag}.pkl')
