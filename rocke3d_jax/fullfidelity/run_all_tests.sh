#!/bin/bash
# Full regression: the main suite in one process, then each XLA-flag-sensitive JAX test file in its own fresh process (see tests/conftest.py).
# Run from fullfidelity/ with the graphcast-env python on PATH (or set PY).
# The bitwise-vs-NumPy tests need XLA_FLAGS exported BEFORE the run (otherwise 20+ of them fail at rounding level, 2026-10-11):
#   XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover" bash run_all_tests.sh
PY=${PY:-python}            # the python of an environment with jax, numpy, pytest (see qa_20261016/ENVIRONMENT.md)
rc=0
$PY -m pytest tests -q -p no:cacheprovider "$@" || rc=1
for f in test_dyn_jax test_dyn_jax2 test_clouds_jax; do
  RUN_XLA_FLAG_TESTS=1 $PY -m pytest tests/$f.py -q -p no:cacheprovider || rc=1
done
for f in test_daily_lake_hook test_d187_imf_p1; do
  RUN_ISOLATED_TESTS=1 $PY -m pytest tests/$f.py -q -p no:cacheprovider || rc=1
done
exit $rc
