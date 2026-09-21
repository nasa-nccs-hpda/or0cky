#!/bin/bash
#SBATCH --job-name=p2saom40_jax_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=compare_data/slurm_gpu_%j.out
#
# GPU leg of the P2SAoM40 Fortran-vs-JAX comparison.
#
# EDIT BEFORE SUBMITTING: --partition/--gres above and the environment
# setup below are placeholders -- I don't have access to this cluster's
# actual GPU partition name or how JAX-with-CUDA is normally activated
# here (module load? conda env? the project's own Dockerfile.gpu via
# singularity/apptainer?). Adjust to match your real setup; the only hard
# requirement is that `python compare_jax.py` run at the bottom sees a
# CUDA-visible JAX install and the same repo checkout (this comparison
# reads/writes files under ./compare_data/ using paths relative to this
# script's directory).
#
# Example adjustments, pick what matches your cluster:
#   module load cuda/12.1 anaconda && conda activate jax-gpu
#   OR: singularity exec --nv rocke3d-jax-gpu.sif python compare_jax.py
#
# Prerequisite: run compare_generate_inputs.py once beforehand (on any
# node) so compare_data/*.npy exists for this job to read -- this job
# does NOT regenerate inputs, so it uses the exact same values as the
# CPU/Fortran legs.

set -euo pipefail
cd "$(dirname "$0")"

# --- EDIT: environment activation for this cluster's GPU nodes ---
# module load cuda/12.1
# conda activate jax-gpu
# -------------------------------------------------------------------

if [ ! -d compare_data ] || [ -z "$(ls -A compare_data/*.npy 2>/dev/null)" ]; then
  echo "compare_data/*.npy not found -- run compare_generate_inputs.py first" >&2
  exit 1
fi

JAX_PLATFORMS=cuda python compare_jax.py

echo "Done. Results written to compare_data/timing_jax_gpu.json and"
echo "compare_data/{drycnv,pbl}_jax_gpu_*_out.npy"
echo "Copy compare_data/ back to the CPU-side checkout (or confirm it's the"
echo "same shared filesystem) before running compare_report.py."
