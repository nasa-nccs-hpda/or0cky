"""D180 driver.  python jax_atm_step_run.py {ref|jax} OUTDIR [nsteps=6] [date=nov26]
Run under: taskset -c 0-2 (or 0) env OMP_NUM_THREADS=1 ...  Saves per-step, per-stage snapshots (stage fields) to OUTDIR/<mode>_step<k>.npz
and timings to OUTDIR/<mode>_timing.json.  ref = atm_step_fast.run_chain (NumPy dynamics numpy-pow, batched NumPy CONDSE, libm);
jax = jax_atm_step.run_chain_jax.  Both import clouds_jax_env first (same XLA flags).  No compile cache."""
import clouds_jax_env  # noqa: F401
import json
import os
import sys
import numpy as np
import atm_step as A
import atm_step_compare as C

mode, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
date = sys.argv[4] if len(sys.argv) > 4 else 'nov26'
it0 = dict(A.DATES)[date]
os.makedirs(out, exist_ok=True)
assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
ctx = A.make_ctx(date, imf=False)


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
    import jax_atm_step as J
    def cb(k, it, R, info, S):
        save(k, info['snaps'], R)
    t, prov = J.run_chain_jax(date, it0, n, ctx, land_mode='recorded', on_step=cb)
    json.dump(prov.recorded_inputs(), open(f'{out}/jax_recorded.json', 'w'), indent=1)
json.dump(t, open(f'{out}/{mode}_timing.json', 'w'), indent=1, default=float)
for r in t:
    print(mode, r['itime'], f"{r['wall']:.1f}s", {k: round(v, 1) for k, v in r['timing'].items()}, r.get('compiles', ''), flush=True)
