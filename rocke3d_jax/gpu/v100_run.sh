#!/bin/bash
# Submit one job on the ADAPT V100 partition from this host (forest204) over ssh gpulogin1; the batch file and the output live in the shared project directory.
# ROCKE_PROJ: absolute path of the checkout whose code the job runs (required).
# usage: gpu/v100_run.sh NAME MINUTES 'shell command run inside fullfidelity/ with graphcast-env'     (example below)
#   gpu/v100_run.sh mstb_seq 30 'python -u gpu_mstcnv_bench.py bench <path>/mstcnv_inputs_nov26_step1.pkl 3'
# Prints the job id; output: gpu_runs/NAME_<jobid>.out .  Environment variables for the job can be put in front of the command (VAR=1 python ...).
# Platform facts (2026-10-10): partition compute, constraint v100, Tesla V100-SXM2-32GB, graphcast-env (jax 0.5.3, python 3.10, no container), no Intel libimf (libm mode).
NAME=${1:?name}; MIN=${2:?minutes}; CMD=${3:?command}
PROJ=${ROCKE_PROJ:?set ROCKE_PROJ to the absolute path of this rocke3d_jax checkout}
mkdir -p "$PROJ/gpu_runs"
F="$PROJ/gpu_runs/${NAME}_$$.sbatch"
cat > "$F" <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${V100_CPUS:-4}
#SBATCH --mem=24G
#SBATCH --time=00:$(printf '%02d' "$MIN"):00
#SBATCH --output=$PROJ/gpu_runs/${NAME}_%j.out
${V100_EXTRA:+#SBATCH $V100_EXTRA}
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
export FF_DATA="${FF_DATA:?set FF_DATA to the ff_data directory}"
echo "host: \$(hostname)  gpu: \$(nvidia-smi --query-gpu=name --format=csv,noheader)  git: \$(cd $PROJ && git log --oneline -1 | cut -c1-60)  dirty: \$(cd $PROJ && git status --short | grep -v '^??' | wc -l)"
cd $PROJ/fullfidelity
export OMP_NUM_THREADS=1
\$CONDA run --no-capture-output -n graphcast-env bash ${F%.sbatch}.cmd.sh
EOF
printf '%s\n' "$CMD" > "${F%.sbatch}.cmd.sh"      # the command goes into its own file: no quoting problems with quotes inside CMD
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 gpulogin1 "sbatch $F" 2>&1 | grep -i "submitted\|error"
