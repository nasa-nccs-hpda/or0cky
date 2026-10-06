"""D139: XLA CPU flag for the JAX dynamics path.  Must be imported before `import jax`.

XLA:CPU contracts a*b+c into fused multiply-adds on FMA-capable hosts (measured: 23% of random a*b+c results differ from numpy
in jax.jit), which is not the unfused IEEE arithmetic of the numpy ports and of the real build (ifort -fp-model strict, no FMA)
and costs ~1e-13 relative on PGF (cancellation in the E-W derivative).  `--xla_cpu_max_isa=AVX` removes the FMA instructions
(a*b+c then matches numpy exactly).  Set DYN_JAX_ALLOW_FMA=1 to keep the default ISA (faster, ~1e-13 relative differences).
`fma_free()` probes the active backend.
"""
import os
import sys

ALLOW_FMA = os.environ.get('DYN_JAX_ALLOW_FMA', '0') == '1'
if not ALLOW_FMA and 'jax' not in sys.modules and 'xla_cpu_max_isa' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_cpu_max_isa=AVX').strip()


def fma_free():
    """True if jitted a*b+c is bit-identical to numpy's on random data (no FMA contraction)."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    r = np.random.default_rng(0)
    a, b, c = r.standard_normal((3, 20000))
    return bool(np.array_equal(np.asarray(jax.jit(lambda x, y, z: x * y + z)(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c))),
                               a * b + c))
