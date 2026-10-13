#!/bin/bash
# Speed track: ent_jax.ent_step steady time under XLA flag sets (same program, no code change).
cd "$(dirname "$0")/../fullfidelity" || exit 1
run() {
  echo "=== $1   XLA_FLAGS='$2'"
  XLA_FLAGS="$2" ROCKE_WHILE_UNROLL=${3:-1} python -u gpu_ent_bench.py 20 2>&1 | grep -v "^\*\*\*\|slow_operation\|Very slow\|^$" | cut -c1-200
}
run baseline ""
run cmdbuf_while "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE"
run cmdbuf_while_min1 "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1"
run no_cmdbuf "--xla_gpu_enable_command_buffer="
run autotune0 "--xla_gpu_autotune_level=0"
run while_unroll4_cmdbuf "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE" 4
