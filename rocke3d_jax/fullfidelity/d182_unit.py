"""D182 unit runner: cold compile time (+ second-call time + output digest) of single jit units of the dynamics step on the args dumped by
d182_stage_profile.py (D182_DUMP).  python d182_unit.py OUT.json ARGDIR UNIT[,UNIT..] --env {dyn|clouds|none} [--mod dyn_step_jax2]
[--patch MODULE] [--save DIR]
MODULE (e.g. d182_variants) is imported after the stock modules; its apply(dj, PROBES) may replace jit units by variants.
Digest = sha1 over the bytes of every output leaf (bitwise identity check between variants run with the same XLA flags/cores).
No persistent compile cache (asserted)."""
import os, sys, json, time, argparse, hashlib, pickle, importlib
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('argdir'); ap.add_argument('units'); ap.add_argument('--env', default='clouds')
ap.add_argument('--mod', default='dyn_step_jax2'); ap.add_argument('--patch', default='')
ap.add_argument('--save', default='')
a = ap.parse_args()
for k in ('JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE'):
    assert k not in os.environ
if a.env == 'clouds':
    import clouds_jax_env  # noqa
elif a.env == 'dyn':
    import dyn_jax_env  # noqa
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
PROBES = {}
_real_jit = jax.jit


def jit(fun=None, **kw):
    if fun is None:
        return lambda f: jit(f, **kw)
    j = _real_jit(fun, **kw)
    PROBES.setdefault(getattr(fun, '__name__', '?'), []).append(j)
    return j


jax.jit = jit
import dyn_step as ds
dj = importlib.import_module(a.mod)
ctx = ds.load_ctx('nov26', imf_pow=False)
kit = dj.Kit(ctx)
if a.patch:
    importlib.import_module(a.patch).apply(dj, kit, PROBES)
res = {}
for u in a.units.split(','):
    args, kw = pickle.load(open(os.path.join(a.argdir, u + '.pkl'), 'rb'))
    f = PROBES[u][-1]
    c0 = time.process_time(); t0 = time.perf_counter(); out = jax.block_until_ready(f(*args, **kw)); t1 = time.perf_counter() - t0; c1 = time.process_time() - c0
    t0 = time.perf_counter(); out = jax.block_until_ready(f(*args, **kw)); t2 = time.perf_counter() - t0
    leaves = jax.tree_util.tree_leaves(out)
    h = hashlib.sha1()
    for l in leaves:
        h.update(np.ascontiguousarray(np.asarray(l)).tobytes())
    res[u] = dict(first_s=t1, first_cpu_s=c1, second_s=t2, digest=h.hexdigest(), nleaves=len(leaves))
    if a.save:
        np.savez(os.path.join(a.save, u + '.npz'), **{str(i): np.asarray(l) for i, l in enumerate(leaves)})
    print(u, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[u].items()}, flush=True)
json.dump(dict(args=vars(a), flags=os.environ.get('XLA_FLAGS'), jax=jax.__version__, units=res, affinity=sorted(os.sched_getaffinity(0))),
          open(a.out, 'w'), indent=1)
