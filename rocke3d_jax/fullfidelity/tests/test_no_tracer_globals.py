"""D214: modules that jax_ocean imports lazily from inside a jit trace must not keep JAX tracers as module globals
(a module-level jnp.asarray executed during a trace yields a tracer that escapes it; seen as UnexpectedTracerError and a NaN-producing
ocean stage on the Discover GPU, job 58799715).  Runs in a subprocess so the import really happens inside a trace."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.path.dirname(HERE)

CODE = r'''
import importlib, sys
import jax, jax.numpy as jnp
from jax._src import core
mods = "ocean_ofluxv ocnmeso_jax ovdiffs_jax oadvt_jax odhorz0_jax oadvt_vec straits_jax stconv_jax ocean_odiff gm_jax ocean_odhorz".split()
def f(x):
    for m in mods:
        importlib.import_module(m)
    return x
jax.jit(f)(jnp.ones(2))
bad = []
for m in mods:
    for k, v in vars(sys.modules[m]).items():
        if isinstance(v, core.Tracer):
            bad.append(m + "." + k)
print("BAD", bad)
'''


def test_lazily_imported_ocean_modules_hold_no_tracers():
    r = subprocess.run([sys.executable, '-c', CODE], cwd=FF, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert 'BAD []' in r.stdout, r.stdout[-400:]
