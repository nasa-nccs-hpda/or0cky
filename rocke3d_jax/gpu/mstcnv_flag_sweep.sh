#!/bin/bash
# D214: benchmark the cloud-base stage (condse_mstcnv) ALONE on the GPU under several XLA flag sets (no code change).  Each variant compiles only this stage.
# usage (from the repo root on Discover):
#   CMD="bash $P/gpu/mstcnv_flag_sweep.sh $P/mstcnv_inputs_nov26_step1.pkl" MODE=custom sbatch --time=1:30:00 gpu/run_gpu_job.sbatch
# Prints, per variant, setup / events / post / total seconds (first run includes the compile), the steady mean and the result digest (equal digests across
# variants on the GPU = the flag did not change the numbers).  A variant whose flag XLA does not know fails within seconds and is reported as such.
IN=${1:?input pkl}
cd "$(dirname "$0")/../fullfidelity" || exit 1
run() {
  echo "=================================================================== variant: $1   XLA_FLAGS='$2'"
  XLA_FLAGS="$2" OMP_NUM_THREADS=1 python -u gpu_mstcnv_bench.py bench "$IN" 3 2>&1 | grep -v "^\*\*\*\|slow_operation_alarm\|Very slow compile\|^$" | cut -c1-220
}
run baseline ""
run command_buffers_while "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL,COLLECTIVES,CONDITIONAL,WHILE"
run graph_min_size_1 "--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1"
