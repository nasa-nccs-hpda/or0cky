#!/bin/bash
# Matched benchmark, ONE exclusive V100 node, strictly sequential: (1) real Fortran ModelE nov26 two-day window (108 steps, end 1950-11-28 03:00) x3, one core;
# (2) JAX hybrid with the SOCRATES JAX radiation (--rad native, day products), same restart, 108 steps, same node.  Cold compile is in step 0 of d193_run.json.
# usage: V100_CPUS=8 V100_EXTRA=--exclusive gpu/v100_run.sh bench_native 100 'bash ../gpu/benchmark_native_v100.sh'
P=${ROCKE_PROJ:?set ROCKE_PROJ to the absolute path of this rocke3d_jax checkout}
echo "node $(hostname) cpu $(lscpu | grep 'Model name' | cut -c1-90) load $(cut -d' ' -f1-3 /proc/loadavg)"
for i in $(seq 1 ${NFORT:-3}); do FORT_DATEE=28 bash $P/gpu/fortran_day.sh ${FTAG:-n}$i ${FCORE:-0}; done
cd $P/fullfidelity
# SPEED_FLAGS (speed track, default empty = the frozen benchmark): extra environment words, e.g. 'ROCKE_MSTCNV_HOST=1 ROCKE_QDYN_VMAP=1 ...'
env ROCKE_TPL_OWN=1 ROCKE_OWN_GHY=1 ROCKE_IRRIG_OWN=1 ROCKE_PBL_CARRY=1 ROCKE_REC_LAZY=1 ROCKE_CSE_IN_OWN=1 ROCKE_CARRY_CSE_OUT=1 ROCKE_ENT_LIVE=1 ROCKE_RADCOLS=1 ROCKE_STAGE_ARGS=1 \
GPU_DAYDIR=nov26_day2 GPU_DAY_DRIVER=d212_day $SPEED_FLAGS python -u gpu_day_run.py $P/gpu_runs/${JAXOUT:-bench_native_jax} --rad native --surf-fields computed --seed computed --cosz computed --nsteps 108 --timed 0
