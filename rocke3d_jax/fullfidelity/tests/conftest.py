"""Test isolation for the JAX tests whose numerics depend on process-global XLA flags.

dyn_jax_env.py / clouds_jax_env.py set XLA_FLAGS (--xla_cpu_max_isa=AVX, --xla_disable_hlo_passes=algsimp) BEFORE jax is imported, because XLA:CPU
otherwise fuses a*b+c into FMAs and rewrites x/c into x*(1/c), which breaks bitwise agreement with numpy (D139, D145). In a full `pytest tests` run jax
is already imported by earlier modules, so the flags cannot take effect and these bitwise tests fail (22 failures in the 2026-10-06 regression), and
if the flags did take effect they would change the numerics of every other JAX test in the process. So these files are skipped from the default run
and must be run each in its own process: see run_all_tests.sh (RUN_XLA_FLAG_TESTS=1).
"""
import os

collect_ignore = [] if os.environ.get("RUN_XLA_FLAG_TESTS") else ["test_dyn_jax.py", "test_dyn_jax2.py", "test_clouds_jax.py"]
