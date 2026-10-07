"""D182 micro-experiments: which jnp pattern makes XLA:CPU emit huge bitcast chains when the algsimp pass is disabled.
python d182b_micro.py {clouds|dyn} -> for each pattern: trace+lower+compile time, instruction counts of the optimized HLO, bitcast count."""
import sys, time, re
env = sys.argv[1]
if env == 'clouds': import clouds_jax_env
else: import dyn_jax_env
import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
IM, JM, LM = 72, 46, 40
I, J = np.meshgrid(np.arange(IM), np.arange(1, JM), indexing='ij'); I = I.ravel(); J = J.ravel()
r = np.random.default_rng(0)
u = jnp.asarray(r.standard_normal((IM, JM, LM)))
def A(f):  # f(u) -> out
    return f
pats = {}
def gather_only(u): return u[I, J, :] * 2.0
def any_loop(u):
    uc = u[I, J, :]; bad = jnp.zeros((), bool)
    for l in range(LM): bad = bad | jnp.any((uc[:, l] < -3.) | (uc[:, l] > 3.))
    return bad
def any_loop_nogather(u):
    uc = u.reshape(-1, LM); bad = jnp.zeros((), bool)
    for l in range(LM): bad = bad | jnp.any((uc[:, l] < -3.) | (uc[:, l] > 3.))
    return bad
def any_once(u):
    uc = u[I, J, :]; return jnp.any((uc < -3.) | (uc > 3.))
def set_loop(u):
    uc = u[I, J, :]
    for l in range(LM): uc = uc.at[:, l].set(uc[:, l] * 1.5)
    return u.at[I, J, :].set(uc)
def arith_loop(u):
    uc = u[I, J, :]; s = jnp.zeros(uc.shape[0])
    for l in range(LM): s = s + uc[:, l] * uc[:, l]
    return s
for name in sys.argv[2:] or ['gather_only', 'any_once', 'any_loop_nogather', 'any_loop', 'arith_loop', 'set_loop']:
    f = jax.jit(globals()[name])
    t0 = time.perf_counter(); lo = f.lower(u); t1 = time.perf_counter(); co = lo.compile(); t2 = time.perf_counter()
    txt = co.as_text()
    nb = len(re.findall(r'= \S+ bitcast', txt))
    print(f"{env} {name}: lower {t1-t0:.2f}s compile {t2-t1:.2f}s instr {txt.count(' = ')} bitcast {nb}", flush=True)
