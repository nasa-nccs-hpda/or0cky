"""Ocean stage 'dynamics' alone (D214): capture its inputs once, then time it truncated after each phase (ROCKE_OCEAN_STOP semantics) on any device.

  capture (run on the device you like, takes a whole step compile):  python -u gpu_ocean_dyn_bench.py capture OUT.pkl [STEP]   (default STEP 1)
  phases (compiles only this stage):                                    python -u gpu_ocean_dyn_bench.py phases IN.pkl [REPEATS]
Phases: 1 = ODHORZ0, 2 = + the five ODHORZ calls with the prefilter, 3 = + OFLUXV, 4 = + OADVT2 of G0M (heat), 99 = + OADVT2 of S0M (the whole stage).
Cumulative steady seconds and the delta per phase are printed.  Diagnostic only (libm mode on a GPU, replay).
"""
import pickle
import sys
import time

mode = sys.argv[1]
_ARGV = list(sys.argv)
sys.argv = [_ARGV[0], 'nov26', '/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/_import_scratch']      # gpu_step_run parses argv at import and would create a directory named after our output path
if mode == 'capture':
    import gpu_step_run as G
    import jax
    sys.argv = _ARGV
    out_path = sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    cp = G.C.Coupled('nov26')
    orig = cp.ob
    cur = {'k': 0, 'done': False}

    def wrapped(oc, fx, Kb):
        if cur['k'] == step and not cur['done']:
            cur['done'] = True
            pickle.dump(dict(oc=jax.device_get(oc), fx=jax.device_get(fx), step=step), open(out_path, 'wb'), protocol=4)
            print('captured', out_path, flush=True)
        return orig(oc, fx, Kb)

    cp.ob = wrapped
    sr = G.build_registry()
    state, rec, dev = cp.initial_state()
    for k in range(step + 1):
        cur['k'] = k
        if k > 0:
            rec = cp.ph.load_records(cp.it0 + k, first=False)
            dev = cp.ph.to_device(rec, first=False)
        out, state, arrays, sr = G.run_step(cp, k, state, rec, dev, sr, False)
        print(f'[step {k}] wall {out["step_seconds"]:.2f} s', flush=True)
        if cur['done']:
            break
else:
    import numpy as np
    import gpu_step_run as G
    import jax
    import jax.numpy as jnp
    import jax_ocean as JO
    sys.argv = _ARGV
    d = pickle.load(open(sys.argv[2], 'rb'))
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cp = G.C.Coupled('nov26')
    oc = jax.tree_util.tree_map(jnp.asarray, d['oc'])
    fx = jax.tree_util.tree_map(jnp.asarray, d['fx'])
    names = {1: 'ODHORZ0', 2: '+ five ODHORZ calls (prefilter)', 3: '+ OFLUXV', 4: '+ OADVT2 G0M (heat)', 99: '+ OADVT2 S0M (salt) = whole stage'}
    prev = 0.0
    print('backend', jax.default_backend(), flush=True)
    for stop in (1, 2, 3, 4, 99):
        JO.OC_STOP = stop
        f = jax.jit(lambda s, fx_, Kb: JO.stage_dynamics(cp.K, Kb, s, fx_))
        ts = []
        for r in range(rep + 1):
            t0 = time.perf_counter()
            out = f(oc, fx, cp.Kbd)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        cur = float(np.mean(ts[1:]))
        print(f'stop {stop:2d} {names[stop]:40s} cumulative {cur:7.3f} s   delta {cur - prev:7.3f} s   (compile+first {ts[0]:.1f} s)', flush=True)
        prev = cur
    JO.OC_STOP = 99
