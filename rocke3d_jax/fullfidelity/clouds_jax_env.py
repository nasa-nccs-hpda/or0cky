"""D145: XLA CPU flags for the JAX cloud physics.  Import BEFORE jax.

Two XLA:CPU behaviours change IEEE results of the unfused Fortran arithmetic (measured on random data against numpy, jax 0.5.3):
  * FMA contraction of a*b+c  -> `--xla_cpu_max_isa=AVX` (as dyn_jax_env, D139);
  * the HLO algebraic simplifier rewrites a/c to a*(1/c) for constants c that are not powers of two (x/35.0: 20% of results differ;
    x/10.0: 33%), a/(b/c) to a*c/b and (a/b)/c to a/(b*c) (35% differ)  -> `--xla_disable_hlo_passes=algsimp` removes all of these
    (0 differences in the same micro tests).  This is the cause behind the "XLA rewrites operations on closure constants" workaround of D139.
CLOUDS_JAX_KEEP_ALGSIMP=1 keeps the simplifier (for the comparison).
"""
import os
import sys

if 'jax' not in sys.modules:
    fl = os.environ.get('XLA_FLAGS', '')
    add = []
    if 'xla_cpu_max_isa' not in fl and os.environ.get('DYN_JAX_ALLOW_FMA', '0') != '1':
        add.append('--xla_cpu_max_isa=AVX')
    if 'xla_disable_hlo_passes' not in fl and os.environ.get('CLOUDS_JAX_KEEP_ALGSIMP', '0') != '1':
        # D182 (2026-10-07): 'reshape-mover' is disabled as well. With algsimp off, XLA:CPU's post_scatter_expansion_simplification loop grows ~150 reshape/bitcast
        # pairs per gather/scatter/reduce; the cold compile of the dynamics step took ~2,500 s instead of ~90 s. Results are bitwise identical (D182; test_dyn_jax,
        # test_dyn_jax2, test_clouds_jax pass under these flags). CLOUDS_JAX_OLD_PASSES=1 restores the old flag.
        add.append('--xla_disable_hlo_passes=' + ('algsimp' if os.environ.get('CLOUDS_JAX_OLD_PASSES', '0') == '1' else 'algsimp,reshape-mover'))
    os.environ['XLA_FLAGS'] = (fl + ' ' + ' '.join(add)).strip()


def flags():
    return os.environ.get('XLA_FLAGS', '')
