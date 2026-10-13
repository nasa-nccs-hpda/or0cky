#!/bin/bash
# Speed track: ent_jax.ent_step steady time under ROCKE_WHILE_UNROLL = 1, 2, 4, 8 (the rtsafe while loop); same cnc sum = same result.
cd "$(dirname "$0")/../fullfidelity" || exit 1
for k in 1 2 4 8; do
  echo "=== ROCKE_WHILE_UNROLL=$k"
  ROCKE_WHILE_UNROLL=$k python -u gpu_ent_bench.py 20 2>&1 | grep -v "^\*\*\*\|slow_operation\|Very slow\|^$"
done
