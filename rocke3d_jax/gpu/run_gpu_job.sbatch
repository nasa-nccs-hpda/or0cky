#!/bin/bash
# One-command version of the manual GPU session (salloc + singularity exec).
# DRAFT, NOT YET RUN ON THE CLUSTER: written from the transcript of the owner's manual session (2026-10-07).
#
# Usage (from the Discover checkout of this repository, project directory):
#   sbatch gpu/run_gpu_job.sbatch                       # runs benchmark_all.py like the manual session
#   CMD="python -u my_script.py args" sbatch --export=ALL gpu/run_gpu_job.sbatch
# Then fold the result back with:  python gpu/collect_gpu_results.py gpu_runs/<jobid>
#
# Resources and container are copied from the manual session; override on the sbatch command line if needed.
#SBATCH --job-name=rocke3d_gpu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --constraint=rome
#SBATCH --output=gpu_runs/%j/slurm.out

set -uo pipefail
PROJ="${SLURM_SUBMIT_DIR:?run with sbatch from the project directory}"
OUT="$PROJ/gpu_runs/${SLURM_JOB_ID}"
mkdir -p "$OUT"
# container as used in the manual session (path copied verbatim; verify it on the cluster)
CONTAINER="${CONTAINER:-/discover/nobackup/projects/QEFM/qefm-core/../containers/qefm-core-sandbox}"
# default command = the manual session's command (note: it launches through torch.distributed.run)
CMD="${CMD:-python -u -m torch.distributed.run $PROJ/benchmark_all.py}"

{
  echo "# metadata (written by run_gpu_job.sbatch)"
  echo "date_start: $(date -Is)"
  echo "job_id: ${SLURM_JOB_ID}"
  echo "node: $(hostname)"
  echo "partition: ${SLURM_JOB_PARTITION:-?}  constraint: rome"
  echo "container: $CONTAINER"
  echo "command: $CMD"
  echo "git_head: $(cd "$PROJ" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files: $(cd "$PROJ" && git status --short 2>/dev/null | grep -v '^??' | wc -l)"
  echo "nvidia_smi:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
} > "$OUT/metadata.txt"

# same call as the manual session; stdout and stderr are kept separately (the manual run mixed warnings into the .out file)
/usr/bin/time -v -o "$OUT/time.txt" \
  singularity exec --nv -B "$PROJ" "$CONTAINER" bash -c "$CMD" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
rc=$?
echo "exit_code: $rc" >> "$OUT/metadata.txt"
echo "date_end: $(date -Is)" >> "$OUT/metadata.txt"
exit $rc
