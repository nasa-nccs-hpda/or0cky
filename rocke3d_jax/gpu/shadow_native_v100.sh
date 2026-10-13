#!/bin/bash
# D230: staged shadow comparison. 30-day free-running native-radiation run on one V100 (1440 steps from nov26 itime 33312; products for 30 model days),
# the packet and the native outputs of every SHADOW_EVERY-th radiation step are dumped, then the SAME packets go through the unmodified Fortran radiation
# server (persistent, one core) and are compared (fullfidelity/shadow_compare.py). Flags and boundary as gpu/benchmark_native_v100.sh (all default-off flags on).
# usage (inside the job, from gpu/v100_run.sh):  bash gpu/shadow_native_v100.sh OUTNAME
P=${ROCKE_PROJ:-/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax}
OUT=$P/gpu_runs/${1:-shadow30}
NSTEPS=${NSTEPS:-1440}
NRAD=$(( NSTEPS / 5 ))
cd $P/fullfidelity
ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1 \
  ROCKE_NATIVE_NRAD=$NRAD ROCKE_SHADOW_DIR=$OUT/shadow ROCKE_SHADOW_EVERY=${SHADOW_EVERY:-4} \
  GPU_DAYDIR=nov26_day2 GPU_DAY_DRIVER=d212_day python -u gpu_day_run.py $OUT --rad native --surf-fields computed --seed computed --cosz computed --nsteps $NSTEPS --timed 0
echo "run exit $?"
python -u shadow_compare.py $OUT/shadow $OUT/shadow_compare.json
echo "compare exit $?"
