#!/bin/bash
# D231: the 30-day native run of gpu/shadow_native_v100.sh WITHOUT the radiation shadow dump, with ROCKE_ZEXTRA_DUMP: every step on which the QDYNAM
# extra-column vertical branch fires writes its columns (lminzij, lmaxzij, ncyc, mw, mu, mv, ma, q) to gpu_runs/<OUT>/zextra/zextra_NNNN.npz.
# usage (inside the job, from gpu/v100_run.sh):  bash <abs path>/gpu/zextra_diag_v100.sh OUTNAME
P=${ROCKE_PROJ:-/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax}
OUT=$P/gpu_runs/${1:-zextra30}
NSTEPS=${NSTEPS:-1440}
cd $P/fullfidelity
ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1 \
  ROCKE_NATIVE_NRAD=$(( NSTEPS / 5 )) ROCKE_ZEXTRA_DUMP=$OUT/zextra \
  GPU_DAYDIR=nov26_day2 GPU_DAY_DRIVER=d212_day python -u gpu_day_run.py $OUT --rad native --surf-fields computed --seed computed --cosz computed --nsteps $NSTEPS --timed 0
echo "run exit $?"
