"""D182 per-jit-unit profile of the dynamics step.  python d182_stage_profile.py OUT.json --env {dyn|clouds|none} [--mod dyn_step_jax2]
Patches jax.jit BEFORE the stage modules are imported: on the first call of each jit unit with a new abstract signature it records
the StableHLO text size / op counts / while-loops / dense constants of `.lower(...)`, then delegates to the real jitted function.
Compile times come from jax_log_compiles (Finished XLA compilation).  No persistent compile cache (asserted)."""
import os, sys, json, time, re, logging, argparse, functools
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--env', default='clouds'); ap.add_argument('--mod', default='dyn_step_jax2')
ap.add_argument('--steps', type=int, default=1); ap.add_argument('--nolower', action='store_true')
a = ap.parse_args()
for k in ('JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE'):
    assert k not in os.environ
if a.env == 'clouds':
    import clouds_jax_env  # noqa
elif a.env == 'dyn':
    import dyn_jax_env  # noqa
import importlib
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_log_compiles', True)
stats = []
_real_jit = jax.jit


class Probe:
    def __init__(self, f, j):
        self._f, self._j, self._seen = f, j, set()
        functools.update_wrapper(self, f)

    def __getattr__(self, n):
        return getattr(self._j, n)

    def __call__(self, *args, **kw):
        flat = jax.tree_util.tree_leaves((args, kw))
        if not a.nolower and not any(isinstance(x, jax.core.Tracer) for x in flat):
            key = tuple((getattr(x, 'shape', None), str(getattr(x, 'dtype', type(x).__name__))) for x in flat) + (repr([x for x in flat if not hasattr(x, 'shape')][:6]),)
            if key not in self._seen:
                self._seen.add(key)
                try:
                    t0 = time.perf_counter(); txt = self._j.lower(*args, **kw).as_text(); dt = time.perf_counter() - t0
                    ops = re.findall(r'= "?(stablehlo\.\w+|func\.call|chlo\.\w+)', txt)
                    from collections import Counter
                    c = Counter(ops)
                    big = [int(m) for m in re.findall(r'dense<[^>]{200,}>\s*:\s*tensor<([\d]+)x', txt)][:5]
                    stats.append(dict(name=getattr(self._f, '__name__', '?'), lower_s=dt, chars=len(txt), nops=len(ops), nwhile=c.get('stablehlo.while', 0),
                                      nconst=c.get('stablehlo.constant', 0), ndense_big=len(re.findall(r'dense<[^>]{200,}>', txt)),
                                      nfunc=len(re.findall(r'func\.func', txt)), top=c.most_common(6)))
                except Exception as e:
                    stats.append(dict(name=getattr(self._f, '__name__', '?'), err=repr(e)[:200]))
        dd = os.environ.get('D182_DUMP')
        if dd and not any(isinstance(x, jax.core.Tracer) for x in flat):
            nm = getattr(self._f, '__name__', '?')
            fp = os.path.join(dd, nm + '.pkl')
            if not os.path.exists(fp):
                import pickle
                tonp = lambda x: np.asarray(x) if hasattr(x, 'shape') else x
                pickle.dump(jax.tree_util.tree_map(tonp, (args, kw)), open(fp, 'wb'))
        return self._j(*args, **kw)


def jit(fun=None, **kw):
    if fun is None:
        return lambda f: jit(f, **kw)
    return Probe(fun, _real_jit(fun, **kw))


jax.jit = jit
recs = []
class H(logging.Handler):
    def emit(self, r):
        recs.append((time.perf_counter(), r.getMessage()))
logging.getLogger('jax._src.dispatch').addHandler(H())
import dyn_step as ds
dj = importlib.import_module(a.mod)
date = 'nov26'; itime = 33312
ctx = ds.load_ctx(date, imf_pow=False)
kit = dj.Kit(ctx)
s1 = ds.load_state(ds.state_path(date, itime, 1))
steps = []
for k in range(a.steps):
    tm = {}; t0 = time.perf_counter(); w = dj.dyn_step_jax(s1, ctx, kit, itime=itime + k, timing=tm); steps.append(dict(wall=time.perf_counter() - t0, stage=tm))
fn = {}
for t, m in recs:
    g = re.search(r'Finished XLA compilation of (\S+) in ([\d.e+-]+) sec', m)
    if g:
        fn.setdefault(g.group(1), []).append(float(g.group(2)))
json.dump(dict(args=vars(a), flags=os.environ.get('XLA_FLAGS'), jax=jax.__version__, steps=steps, xla_compile=fn, units=stats,
               affinity=sorted(os.sched_getaffinity(0))), open(a.out, 'w'), indent=1, default=str)
print('steps', [round(s['wall'], 1) for s in steps], 'xla total', round(sum(sum(v) for v in fn.values()), 1), 'n', sum(len(v) for v in fn.values()))
for s in stats:
    nm = s['name']
    print(nm, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k not in ('name', 'top')}, 'xla', [round(x, 2) for x in fn.get('jit(%s)' % nm, [])])
