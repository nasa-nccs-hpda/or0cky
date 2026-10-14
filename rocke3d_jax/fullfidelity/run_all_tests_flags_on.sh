#!/bin/bash
# Full regression with the nine speed flags ON (SPEED_RUN_RECIPE_20261013.md). Wraps run_all_tests.sh; flags-off baseline: 3329 passed, 1 failed (known order-dependent libimf test), 16 skipped, 1 h 42 min.
# Run from fullfidelity/ :   bash run_all_tests_flags_on.sh [pytest args]     (log: tests_flags_on_<date>.log, summary at the end)
# STAGED=1 also sets the ten staged/standalone flags of the recipe command (default: speed flags only).
# Expected by design, not defects: tests that assert bitwise equality with the level-by-level Ent (ROCKE_ENT_BATCH: ~1e-15 in 16-36 of 1,146 cells).
export XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp,reshape-mover"   # as in run_all_tests.sh
export ROCKE_MSTCNV_HOST=1 ROCKE_MSTCNV_IMF=1 ROCKE_MSTCNV_HOST_IO=1 ROCKE_QDYN_VMAP=1 ROCKE_POLE_FAST=2 ROCKE_POLE_ASYNC=1 ROCKE_SEQSUM_BLK=32 ROCKE_GLUE_JIT=1 ROCKE_ENT_BATCH=1
export ROCKE_MSTCNV_PY="${ROCKE_MSTCNV_PY:-/home/gtamkin/.conda/envs/DEV-Kernel/bin/python}"   # numba worker interpreter
if [ "${STAGED:-0}" = "1" ]; then
  export ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1
fi
unset JAX_COMPILATION_CACHE_DIR CLOUDS_JAX_CACHE
LOG="tests_flags_on_$(date +%Y%m%d_%H%M).log"
echo "flags-on regression start $(date -Is) host $(hostname) git $(git rev-parse --short HEAD) STAGED=${STAGED:-0}" | tee "$LOG"
env | grep '^ROCKE_' | sort | tee -a "$LOG"
bash run_all_tests.sh "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "flags-on regression end $(date -Is) rc=$rc" | tee -a "$LOG"
grep -E "passed|failed|FAILED|ERROR" "$LOG" | tail -40
exit $rc
