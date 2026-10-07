#!/bin/bash
# One-command version of the manual GPU session (salloc + singularity exec). Paths from the owner, 2026-10-07.
# DRAFT, NOT YET RUN ON THE CLUSTER.
#
# Submit from the project directory on Discover (/discover/nobackup/gtamkin/dev/ilab-agentic-ai/projects/imvi/rocke3d_jax):
#   MODE=smoke  sbatch gpu/run_gpu_job.sbatch     # gpu/smoke_test_env.py (about 1 minute): is JAX with CUDA in the container?
#   MODE=probe  sbatch gpu/run_gpu_job.sbatch     # gpu/gpu_probe.py: devices, float64 timing, device math functions vs NumPy
#   MODE=bench  sbatch gpu/run_gpu_job.sbatch     # your manual command: torch.distributed.run benchmark_all.py
#   MODE=atm    sbatch gpu/run_gpu_job.sbatch     # the JAX atmosphere step, 6 steps of nov26 (needs the data: gpu/DATA_SUBSET_6STEP.md)
#   CMD="..."   MODE=custom sbatch gpu/run_gpu_job.sbatch
# Results land in gpu_runs/<jobid>/ ; fold them back with:  python gpu/collect_gpu_results.py gpu_runs/<jobid>
#
# Resources copied from the manual session; override on the command line (e.g. sbatch --time=3:00:00 ...).
#SBATCH --job-name=rocke3d_gpu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --constraint=rome
#SBATCH --output=gpu_runs/%j/slurm.out

set -uo pipefail
PROJ="${SLURM_SUBMIT_DIR:?run with sbatch from the project directory}"
OUT="$PROJ/gpu_runs/${SLURM_JOB_ID}"
mkdir -p "$OUT"
# the owner's container (path copied verbatim)
CONTAINER="${CONTAINER:-/discover/nobackup/projects/QEFM/qefm-core/../containers/qefm-core-sandbox}"
MODE="${MODE:-bench}"

# where the data and support files sit on Discover (the owner's choice: inside the project directory)
export FF_DATA="${FF_DATA:-$PROJ/ff_data}"
export MODELE_PROD_INPUT="${MODELE_PROD_INPUT:-$PROJ/modelE2_planet_2.0/ModelE_Support/prod_input_files}"
# no Intel libimf in the container unless the owner copies it and sets INTEL_LIBIMF_DIR (then bitwise-with-Fortran results are possible)
unset JAX_COMPILATION_CACHE_DIR CLOUDS_JAX_CACHE     # a warm compile cache changed results (D175): never for validated runs

case "$MODE" in
  smoke) CMD="python -u $PROJ/gpu/smoke_test_env.py --repo $PROJ/fullfidelity" ;;
  probe) CMD="python -u $PROJ/gpu/gpu_probe.py" ;;
  bench) CMD="python -u -m torch.distributed.run $PROJ/benchmark_all.py" ;;     # exactly the manual command
  atm)   CMD="cd $PROJ/fullfidelity && OMP_NUM_THREADS=1 python -u jax_atm_step_run.py jax $OUT/atm 6 nov26" ;;
  custom) CMD="${CMD:?set CMD}" ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac

{
  echo "# metadata (written by run_gpu_job.sbatch)"
  echo "date_start: $(date -Is)"
  echo "job_id: ${SLURM_JOB_ID}"
  echo "node: $(hostname)"
  echo "partition: ${SLURM_JOB_PARTITION:-?}  constraint: rome"
  echo "mode: $MODE"
  echo "container: $CONTAINER"
  echo "command: $CMD"
  echo "FF_DATA: $FF_DATA"
  echo "MODELE_PROD_INPUT: $MODELE_PROD_INPUT"
  echo "git_head: $(cd "$PROJ" && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files: $(cd "$PROJ" && git status --short 2>/dev/null | grep -v '^??' | wc -l)"
  echo "nvidia_smi:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
} > "$OUT/metadata.txt"

# same call as the manual session; stdout and stderr kept separately (the manual run mixed warnings into the .out file)
/usr/bin/time -v -o "$OUT/time.txt" \
  singularity exec --nv -B "$PROJ" "$CONTAINER" bash -c "$CMD" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
rc=$?
echo "exit_code: $rc" >> "$OUT/metadata.txt"
echo "date_end: $(date -Is)" >> "$OUT/metadata.txt"
exit $rc
