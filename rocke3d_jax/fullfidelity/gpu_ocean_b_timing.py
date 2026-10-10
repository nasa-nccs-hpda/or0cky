"""Where do the seconds of ocean stage b go? Each ocean-b stage as its OWN jit, blocked and timed per step (D214 follow-up; run on the GPU and on the CPU).

usage: python -u gpu_ocean_b_timing.py [DATE] [NSTEPS]       (default nov26, 4 steps; ~12 min cold on the A100)

Replaces the single ocean-b jit of the assembled step by one jit per stage (dynamics, straits, post, odiff, meso, in ocean_stages order) with
block_until_ready and perf_counter around each, then prints the seconds per stage and step. Results are the same arrays (stage by stage composition),
but XLA can no longer fuse across stage boundaries, so the sum can differ a little from the single-jit ob time. Step 0 includes compilation: read steps >= 2.
Labels: libm mode, radiation replayed, hybrid; a profiling diagnostic, not a fidelity result.
"""
import sys
import time
import gpu_step_run as G
import jax
import jax_ocean as JO

DATE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else 'nov26'
NSTEPS = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 4
C = G.C
cp = C.Coupled(DATE)
K = cp.K
SUB = JO.STAGES[5:]                      # ocean stage b: dynamics, straits, post, odiff, meso
jits = {n: jax.jit((lambda fn: (lambda s, fx, Kb: fn(K, Kb, s, fx)))(f)) for n, f in SUB}
rows = []
cur = {'k': 0}


def ob_timed(oc, fx, Kb):
    s = oc
    jax.block_until_ready(s)
    t = {}
    for n, _ in SUB:
        t0 = time.perf_counter()
        s = jits[n](s, fx, Kb)
        jax.block_until_ready(s)
        t[n] = time.perf_counter() - t0
    rows.append((cur['k'], t))
    return s


cp.ob = ob_timed
sr = G.build_registry()
state, rec, dev = cp.initial_state()
for k in range(NSTEPS):
    cur['k'] = k
    if k > 0:
        rec = cp.ph.load_records(cp.it0 + k, first=False)
        dev = cp.ph.to_device(rec, first=False)
    out, state, arrays, sr = G.run_step(cp, k, state, rec, dev, sr, True)
    print(f'[step {k}] wall {out["step_seconds"]:.2f} s', flush=True)
print(f'\nbackend {jax.default_backend()}; seconds per ocean-b stage (step 0 includes compilation)')
names = [n for n, _ in SUB]
print('step ' + ''.join(f'{n:>11s}' for n in names) + f'{"sum":>9s}')
for k, t in rows:
    print(f'{k:4d} ' + ''.join(f'{t[n]:11.3f}' for n in names) + f'{sum(t.values()):9.3f}')
