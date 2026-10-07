"""D182: cold-process run of dyn_step_jax2 (or a variant module) for N steps from the real nov26 states (33312..), saving every workspace
array to OUT.npz and the timing / compile statistics to OUT.json.  Used for the bitwise old-flags vs new-flags comparison.
python d182b_dyn_digest.py OUT {old|new|avx} [--mod dyn_step_jax2] [--steps 2]
old = clouds_jax_env defaults (--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp); new = old + reshape-mover disabled;
avx = dyn_jax_env only (algsimp ON).  XLA flags are set before jax is imported.  No persistent compile cache."""
import os, sys, json, time, argparse, re, logging
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('flags'); ap.add_argument('--mod', default='dyn_step_jax2')
ap.add_argument('--steps', type=int, default=2); a = ap.parse_args()
for k in ('JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE'):
    assert k not in os.environ
if a.flags == 'new':
    os.environ['XLA_FLAGS'] = '--xla_disable_hlo_passes=algsimp,reshape-mover --xla_cpu_max_isa=AVX'
    import clouds_jax_env  # noqa
elif a.flags == 'old':
    import clouds_jax_env  # noqa
else:
    import dyn_jax_env  # noqa
import importlib, numpy as np
import jax
jax.config.update('jax_log_compiles', True)
n_xla = [0]; t_xla = [0.0]
class H(logging.Handler):
    def emit(self, r):
        m = re.search(r'Finished XLA compilation of (\S+) in ([\d.e+-]+) sec', r.getMessage())
        if m: n_xla[0] += 1; t_xla[0] += float(m.group(2))
logging.getLogger('jax').addHandler(H())
import dyn_step as ds
dj = importlib.import_module(a.mod)
ctx = ds.load_ctx('nov26', imf_pow=False); kit = dj.Kit(ctx)
res = {}; steps = []
for k in range(a.steps):
    it = 33312 + k
    s1 = ds.load_state(ds.state_path('nov26', it, 1))
    tm = {}; t0 = time.perf_counter(); w = dj.dyn_step_jax(s1, ctx, kit, itime=it, timing=tm); wall = time.perf_counter() - t0
    steps.append(dict(itime=it, wall=wall, stage=tm, xla_compiles_cum=n_xla[0], xla_s_cum=t_xla[0]))
    print('step', it, round(wall, 1), 'compiles', n_xla[0], 'xla_s', round(t_xla[0], 1), {s: round(v, 1) for s, v in tm.items()}, flush=True)
    for name, v in w.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f': res[f's{k}/{name}'] = np.ascontiguousarray(v, dtype=np.float64)
        elif isinstance(v, (float, np.floating)): res[f's{k}/{name}'] = np.float64(v)
np.savez(a.out + '.npz', **res)
json.dump(dict(args=vars(a), flags=os.environ.get('XLA_FLAGS'), jax=jax.__version__, steps=steps, nfields=len(res), compiles=n_xla[0], xla_s=t_xla[0],
               affinity=sorted(os.sched_getaffinity(0))), open(a.out + '.json', 'w'), indent=1)
