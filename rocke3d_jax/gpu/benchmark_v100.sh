#!/bin/bash
# V100 benchmark, ONE node, strictly sequential (no concurrent load): (1) real Fortran 54-step nov26 day x3 (1 core), (2) JAX hybrid day with the radiation COMPUTED
# by the Fortran persistent server, 108 steps (day 1 + day 2), cold compile and steady steps logged separately in d193_run.json.
# usage: gpu/v100_run.sh bench_v100 150 'bash ../gpu/benchmark_v100.sh'   (V100_CPUS=8)
P=/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax
echo "node $(hostname) cpu $(lscpu | grep 'Model name' | cut -c1-90) load $(cut -d' ' -f1-3 /proc/loadavg)"
for i in 1 2 3; do bash $P/gpu/fortran_day.sh b$i ${FCORE:-0}; done
cd $P/fullfidelity
GPU_DAY_DRIVER=d212_day python -u gpu_day_run.py $P/gpu_runs/bench_jax_server --rad server --surf-fields computed --seed computed --cosz ${COSZ:-computed} --nsteps 108 --timed 0
