"""D183 driver.  python d183_run.py {ref|jax} OUTDIR [nsteps=2] [date=nov26]
Run as: taskset -c 5-6 env OMP_NUM_THREADS=1 python d183_run.py ...  (same affinity for both modes, separate processes, one after the other).
ref = NumPy chained step in libimf mode (atm_step_fast.run_chain, ctx imf=True);
jax = D180 JAX chain with libimf host callbacks (jax_atm_step_imf.install('libimf') + jax_atm_step.run_chain_jax, ctx imf=True).
Saves per-step stage snapshots (OUTDIR/<mode>_step<k>.npz), timings and (jax) libimf_ops counters per step (OUTDIR/<mode>_timing.json)."""
import clouds_jax_env  # noqa: F401
import json
import os
import sys
import numpy as np
import atm_step as A
import atm_step_compare as C

mode, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
date = sys.argv[4] if len(sys.argv) > 4 else 'nov26'
it0 = dict(A.DATES)[date]
os.makedirs(out, exist_ok=True)
assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
ctx = A.make_ctx(date, imf=True)


def save(k, snaps, R):
    d = {}
    for st, fl in C.STAGE_FIELDS.items():
        for f in fl:
            if f in snaps[st]:
                d[f'{st}/{f}'] = np.asarray(snaps[st][f], float)
    ref = A.end_reference(R)
    for f in A.END_FIELDS:
        if f in ref:
            d[f'real/{f}'] = np.asarray(ref[f], float)
    np.savez(f'{out}/{mode}_step{k}.npz', **d)


if mode == 'ref':
    import atm_step_fast as F

    def cb(k, it, R, sn, S, tm):
        save(k, sn, R)
    t = F.run_chain(date, it0, n, ctx, land_mode='recorded', on_step=cb)
else:
    import jax_atm_step_imf as I
    import jax_atm_step as J
    import libimf_ops as L
    inst = I.install('libimf')
    cnts = []
    prev = [L.counters()]

    def cb(k, it, R, info, S):
        save(k, info['snaps'], R)
        c = L.counters()
        cnts.append({f: {x: c[f][x] - prev[0][f][x] for x in c[f]} for f in c})
        prev[0] = c
    t, prov = J.run_chain_jax(date, it0, n, ctx, land_mode='recorded', on_step=cb)
    for r, c in zip(t, cnts):
        r['libimf_callbacks'] = c
    json.dump(dict(installed=inst, status=L.status(), callback_sites=I.CALLBACK_SITES, recorded=prov.recorded_inputs()),
              open(f'{out}/jax_header.json', 'w'), indent=1, default=str)
json.dump(t, open(f'{out}/{mode}_timing.json', 'w'), indent=1, default=float)
for r in t:
    print(mode, r['itime'], f"{r['wall']:.1f}s", {k: round(v, 1) for k, v in r['timing'].items()}, r.get('compiles', ''),
          r.get('libimf_callbacks', {}).get('total', ''), flush=True)
