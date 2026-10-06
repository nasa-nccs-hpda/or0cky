#!/bin/bash
# Full regression: the main suite in one process, then each XLA-flag-sensitive JAX test file in its own fresh process (see tests/conftest.py).
# Run from fullfidelity/ with the graphcast-env python on PATH (or set PY).
PY=${PY:-/home/gtamkin/.conda/envs/graphcast-env/bin/python}
rc=0
$PY -m pytest tests -q -p no:cacheprovider "$@" || rc=1
for f in test_dyn_jax test_dyn_jax2 test_clouds_jax; do
  RUN_XLA_FLAG_TESTS=1 $PY -m pytest tests/$f.py -q -p no:cacheprovider || rc=1
done
exit $rc
