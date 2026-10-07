"""D182 harness: cold-process timing of the dynamics step (dyn_step_jax2 or the D182 variant), per jitted function.
python d182_cold_compile.py OUT.json --env {dyn|clouds|none} [--mod dyn_step_jax2] [--steps 1] [--hlo]
Sets XLA flags BEFORE importing jax, no persistent compile cache (asserted).  Logs jax.log_compiles lines and parses them:
'Finished tracing + transforming <f> for pjit in X sec', 'Finished jaxpr to MLIR module conversion jit(<f>) in X sec',
'Finished XLA compilation of jit(<f>) in X sec'."""
import os, sys, json, time, re, logging, argparse
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--env', default='clouds'); ap.add_argument('--mod', default='dyn_step_jax2')
ap.add_argument('--steps', type=int, default=1); ap.add_argument('--date', default='nov26'); ap.add_argument('--stages', default='')
a = ap.parse_args()
for k in ('JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE'):
    assert k not in os.environ
if a.env == 'clouds':
    import clouds_jax_env  # noqa
elif a.env == 'dyn':
    import dyn_jax_env  # noqa
elif a.env == 'avx_only_clouds_off':
    pass
import importlib
import numpy as np
import jax
jax.config.update('jax_log_compiles', True)
assert not jax.config.jax_enable_compilation_cache or not jax.config.jax_compilation_cache_dir
recs = []
class H(logging.Handler):
    def emit(self, r):
        recs.append((time.perf_counter(), r.getMessage()))
logging.getLogger('jax').addHandler(H()); logging.getLogger('jax').setLevel(logging.DEBUG)
import dyn_step as ds
dj = importlib.import_module(a.mod)
date = 'nov26'; itime = 33312
ctx = ds.load_ctx(date, imf_pow=False)
t0 = time.perf_counter(); kit = dj.Kit(ctx); tkit = time.perf_counter() - t0
s1 = ds.load_state(ds.state_path(date, itime, 1))
steps = []
for k in range(a.steps):
    tm = {}; t0 = time.perf_counter(); w = dj.dyn_step_jax(s1, ctx, kit, itime=itime + k, timing=tm); steps.append(dict(wall=time.perf_counter() - t0, stage=tm))
fn = {}
pat = [('trace', r'Finished tracing \+ transforming (\S+) for pjit in ([\d.e+-]+) sec'),
       ('lower', r'Finished jaxpr to MLIR module conversion (\S+) in ([\d.e+-]+) sec'),
       ('xla', r'Finished XLA compilation of (\S+) in ([\d.e+-]+) sec')]
for t, m in recs:
    for kind, p in pat:
        g = re.search(p, m)
        if g:
            d = fn.setdefault(g.group(1), dict(trace=0., lower=0., xla=0., n=0)); d[kind] += float(g.group(2))
            if kind == 'xla': d['n'] += 1
tot = {k: sum(v[k] for v in fn.values()) for k in ('trace', 'lower', 'xla', 'n')}
json.dump(dict(args=vars(a), flags=os.environ.get('XLA_FLAGS'), jax=jax.__version__, kit_s=tkit, steps=steps, per_fn=fn, total=tot,
               affinity=sorted(os.sched_getaffinity(0)), nlog=len(recs)), open(a.out, 'w'), indent=1)
print('steps', [round(s['wall'], 1) for s in steps], 'totals', tot, 'kit', round(tkit, 1))
for k, v in sorted(fn.items(), key=lambda kv: -(kv[1]['trace'] + kv[1]['lower'] + kv[1]['xla']))[:10]:
    print(k[:60], {x: round(y, 2) for x, y in v.items()})
