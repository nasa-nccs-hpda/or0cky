#!/bin/bash
# Flag-specific checks of the adopted speed flags (SPEED_RUN_RECIPE_20261013.md). They are standalone scripts, not part of tests/ or run_all_tests.sh.
# Run from fullfidelity/ :  bash run_speed_flag_tests.sh        Exit code 0 only if every check exits 0. CPU; a few minutes.
# Needs the MSTCNV capture pickle (gpu_data_stage/mstcnv_inputs_nov26_step1.pkl) for the two MSTCNV checks; they are skipped (rc reported as SKIP) if absent.
PY=${PY:?set PY to a python with jax 0.5.3, numpy, mpmath (see qa_20261016/ENVIRONMENT.md)}
PYNB=${PYNB:?set PYNB to a python with numba (the ROCKE_MSTCNV_PY interpreter, see qa_20261016/ENVIRONMENT.md)}
PKL=${PKL:-}   # path of mstcnv_inputs_nov26_step1.pkl (recorded cloud-base inputs); the two MSTCNV checks are skipped when it is not set or absent
export XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover"
export OMP_NUM_THREADS=1 ROCKE_MSTCNV_PY="$PYNB"
unset JAX_COMPILATION_CACHE_DIR CLOUDS_JAX_CACHE
rc=0
check() {  # name, command...
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "speed_flag_test_${name}.log" 2>&1; local r=$?
  echo "$name rc=$r ($(( $(date +%s) - t0 )) s)  log: speed_flag_test_${name}.log  last: $(tail -1 speed_flag_test_${name}.log | cut -c1-120)"
  [ $r -eq 0 ] || rc=1
}
check ent_batch   $PY speed_ent_batch_test.py
check qdyn_vmap   $PY speed_qdyn_vmap_test.py 3
check seqsum      $PY speed_seqsum_test.py
if [ -n "$PKL" ] && [ -f "$PKL" ]; then
  check mstcnv_col  $PYNB speed_mstcnv_col_test.py "$PKL"
  check mstcnv_host $PY   speed_mstcnv_host_test.py "$PKL" 64 2
else
  echo "mstcnv_col SKIP (PKL unset or missing: '$PKL')"; echo "mstcnv_host SKIP"
fi
echo "speed flag tests rc=$rc"
exit $rc
