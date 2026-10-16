"""D182: drop-in for clouds_jax_env with the XLA:CPU compile-time fix.  Import BEFORE jax.

Same two arithmetic flags as clouds_jax_env (D145): `--xla_cpu_max_isa=AVX` (no FMA) and the algebraic simplifier disabled
(`algsimp`, keeps a/c as a division).  Disabling algsimp alone makes the XLA:CPU fixpoint pipeline `post_scatter_expansion_simplification`
grow ~150 reshape/bitcast pairs per gather / scatter / reduce (nothing cancels them any more), which inflates the HLO 10-20x and makes the
cold compile of the dynamics step ~30x slower (D182).  Additionally disabling `reshape-mover` (a pure data-movement pass) stops the growth.
Bitwise equality of results with the clouds_jax_env flags is validated in D182 (see D182_COMPILE_TIME_ENTRY.md).
CLOUDS_JAX_KEEP_ALGSIMP=1 and DYN_JAX_ALLOW_FMA=1 behave as in clouds_jax_env; D182_KEEP_RESHAPE_MOVER=1 restores the old pass list.
"""
import os
import sys

if 'jax' not in sys.modules:
    fl = os.environ.get('XLA_FLAGS', '')
    add = []
    if 'xla_cpu_max_isa' not in fl and os.environ.get('DYN_JAX_ALLOW_FMA', '0') != '1':
        add.append('--xla_cpu_max_isa=AVX')
    if 'xla_disable_hlo_passes' not in fl and os.environ.get('CLOUDS_JAX_KEEP_ALGSIMP', '0') != '1':
        passes = 'algsimp' if os.environ.get('D182_KEEP_RESHAPE_MOVER', '0') == '1' else 'algsimp,reshape-mover'
        add.append('--xla_disable_hlo_passes=' + passes)
    os.environ['XLA_FLAGS'] = (fl + ' ' + ' '.join(add)).strip()


def flags():
    return os.environ.get('XLA_FLAGS', '')
