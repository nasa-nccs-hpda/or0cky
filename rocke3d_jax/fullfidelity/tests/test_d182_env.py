"""D182: clouds_jax_env_fast sets the three flags before jax and keeps the arithmetic of clouds_jax_env (numpy-identical a/c and a*b+c) while the
optimized HLO of a scatter loop stays small.  Runs the checks in a fresh subprocess (XLA flags are read at jax import)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = r'''
import sys, re
import clouds_jax_env_fast as e
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
fl = e.flags()
assert "algsimp,reshape-mover" in fl and "xla_cpu_max_isa=AVX" in fl, fl
r = np.random.default_rng(0); x, y, z = r.standard_normal((3, 20000))
assert np.array_equal(np.asarray(jax.jit(lambda a: a / 35.0)(x)), x / 35.0)
assert np.array_equal(np.asarray(jax.jit(lambda a, b, c: a * b + c)(x, y, z)), x * y + z)
u = jnp.zeros((72, 46, 40))
def f(u):
    uc = u.reshape(-1, 40)
    for l in range(40):
        uc = uc.at[:, l].set(uc[:, l] * 1.5)
    return uc
n = jax.jit(f).lower(u).compile().as_text().count(" = ")
print("instr", n)
assert n < 4000, n   # clouds_jax_env flags: ~8,700 for the same function (measured, D182)
'''


def test_env_fast_flags_arithmetic_and_hlo_size():
    env = dict(os.environ, PYTHONPATH=HERE, OMP_NUM_THREADS='1')
    env.pop('XLA_FLAGS', None)
    r = subprocess.run([sys.executable, '-c', CODE], env=env, capture_output=True, text=True, timeout=300, cwd=HERE)
    assert r.returncode == 0, r.stdout + r.stderr
