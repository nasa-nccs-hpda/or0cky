"""Which OCEAN stage first produces non-finite values at step 0? (D214 follow-up; run it like gpu_step_run.py, on the GPU and, as a control, on the CPU)

usage: python -u gpu_ocean_stage_probe.py [DATE] [STEPS]       (default nov26, 1 step; about the cost of one gpu_step_run step: ~12 min cold on the A100)

It builds the assembled step exactly as gpu_step_run does, then wraps the two ocean jits so that each also returns, per ocean stage
(ground, ostres, oconv, drag, polar | dynamics, straits, post, odiff, meso), the NUMBER OF NON-FINITE VALUES of every ocean state array after that stage.
Nothing else changes (same jit boundaries otherwise, same inputs). Prints a table: the first stage with a non-zero count is the stage that creates the NaN;
the arrays listed there are the first victims.  Labels: libm mode, radiation replayed, hybrid; this is a diagnostic, not a fidelity result.
"""
import sys
import gpu_step_run as G           # sets the backend/flags before jax is used (same as the runner)
import jax
import jax.numpy as jnp
import jax_ocean as JO

DATE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else 'nov26'
NSTEPS = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 1
C = G.C
cp = C.Coupled(DATE)
rows = []


def counts(s):
    return {k: jnp.sum(~jnp.isfinite(v)).astype(jnp.int64) for k, v in s.items() if hasattr(v, 'dtype') and jnp.issubdtype(v.dtype, jnp.floating)}


def make(which):
    K = cp.K

    def f(oc, fx, Kb):
        tr = {}
        out = JO.ocean_stages(K, Kb, oc, fx, which=which, trace=tr)
        return out, {n: counts(s) for n, s in tr.items()}
    return jax.jit(f)


oa2, ob2 = make('a'), make('b')
cur = {'k': 0}


def wrap(fn, tag):
    def g(oc, fx, Kb):
        out, c = fn(oc, fx, Kb)
        rows.append((cur['k'], tag, jax.device_get(c)))
        return out
    return g


cp.oa, cp.ob = wrap(oa2, 'a'), wrap(ob2, 'b')
sr = G.build_registry()
state, rec, dev = cp.initial_state()
for k in range(NSTEPS):
    cur['k'] = k
    if k > 0:
        rec = cp.ph.load_records(cp.it0 + k, first=False)
        dev = cp.ph.to_device(rec, first=False)
    out, state, arrays, sr = G.run_step(cp, k, state, rec, dev, sr, False)
    print(f'[step {k}] done; backend {jax.default_backend()}', flush=True)
first = None
print('\nstep  stage        arrays with non-finite values (count)')
ORDER = ['ground', 'ostres', 'oconv', 'drag', 'polar', 'dynamics', 'straits', 'post', 'odiff', 'meso']   # execution order (a jit output dict comes back key-sorted)
for k, tag, c in sorted(rows, key=lambda r: (r[0], r[1])):
    for stage in [x for x in ORDER if x in c]:
        d = c[stage]
        bad = {a: int(n) for a, n in d.items() if int(n)}
        print(f'{k:4d}  {stage:10s}  {bad if bad else "-"}')
        if bad and first is None:
            first = (k, stage, bad)
print('\nFIRST stage with non-finite values:', first)
