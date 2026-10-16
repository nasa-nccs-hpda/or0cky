#!/bin/bash
# V100 recipe for the 6-step coupled nov26 run (LIBM, radiation replayed) in the graphcast-env conda env (CUDA JAX). Same command as gpu/step_command.txt.
# Submit from forest204 (no local SLURM): ssh gpulogin1 "cd <project dir> && sbatch gpu/v100_step6_graphcast.sbatch"
# Cluster facts as found 2026-10-13: partition compute, feature v100 (gpu_a100 / rome / the QEFM container path in run_gpu_job.sbatch do not exist here).
# Run at 061486d: jobs 37942390 and 37942591 (different nodes) gave bitwise-identical step 0 and step 5 end states, steady mean about 2.6 s/step, about 12-13 min wall.
# Outputs go to gpu_runs/ (git-ignored); compare with: python fullfidelity/gpu_step_run.py --compare CAND.npz REF.npz OUTDIR
#SBATCH --job-name=v100_rerun6
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/v100_rerun6_%j.out
PROJ=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
OUT=$PROJ/gpu_runs/v100_rerun6_${SLURM_JOB_ID}
mkdir -p $OUT
export FF_DATA=$PROJ/ff_data
export MODELE_PROD_INPUT=$PROJ/modelE2_planet_2.0/ModelE_Support/prod_input_files
unset JAX_COMPILATION_CACHE_DIR CLOUDS_JAX_CACHE XLA_FLAGS
export OMP_NUM_THREADS=1
echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)  git: $(cd $PROJ && git rev-parse --short HEAD)" | tee $OUT/metadata.txt
cd $PROJ/fullfidelity
$CONDA run --no-capture-output -n graphcast-env python -u gpu_step_run.py nov26 $OUT/step --steps 6 --save first-last > $OUT/stdout.txt 2> $OUT/stderr.txt
echo "exit_code: $?" >> $OUT/metadata.txt
