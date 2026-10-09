#!/bin/bash
cd /panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity
for m in "p1|T 36 23 10 1" "ctrl|" "p2|T 20 12 20 -1" "p3|Q 50 30 5 1"; do
  n=${m%%|*}; p=${m#*|}
  if [ -n "$p" ]; then export D213_PERT="$p"; else unset D213_PERT; fi
  s=$(date +%s)
  taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d213/tmp D189_SHIM_DIR=/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d213/../d209 /home/gtamkin/.conda/envs/graphcast-env/bin/python d213_day.py /panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d213/run_$n --nit-strict 0 --daily-lake 1 > /panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d213/run_$n.log 2>&1
  echo "$n rc=$? wall=$(($(date +%s)-s))" >> /panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d213/chain.status
done
