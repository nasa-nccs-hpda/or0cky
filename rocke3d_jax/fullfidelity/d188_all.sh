#!/bin/bash
# D188: NumPy reference then JAX run for one date, same core pinning on both sides.  usage: d188_all.sh DATE OUTDIR
D=$1; O=$2; mkdir -p $O
PY=/home/gtamkin/.conda/envs/graphcast-env/bin/python
cd "$(dirname "$0")"
[ -s $O/ref_$D.npz ] || taskset -c 3-5 env OMP_NUM_THREADS=1 $PY d188_ref_numpy.py $D $O/ref_$D.npz > $O/ref_$D.log 2>&1
taskset -c 3-5 env OMP_NUM_THREADS=1 $PY d188_run.py $D $O/ref_$D.npz $O/run_$D.json > $O/run_$D.log 2>&1
