#!/bin/bash
# Real Fortran ModelE, nov26 54-step day, ONE process pinned to one core, production hook-free binary, as D215/one.sh but with output under gpu_runs/.
# usage: gpu/fortran_day.sh TAG CORE      prints real/user/sys and the model log's timing lines
FF=/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data
HS=/explore/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/huge_space/P2SAoM40
BIN=$HS/P2SAoM40.bin
export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
R=${FORTRAN_RUNS:-/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/fortran}/run_$1; rm -rf $R; mkdir -p $R; cd $R
for f in I P2SAoM40ln P2SAoM40uln runtime_opts; do cp $HS/$f .; done
\cp $BIN P2SAoM40.bin; \cp $BIN P2SAoM40
sed -i "109s/YEARE=1950,MONTHE=12,DATEE=1,HOURE=0,/YEARE=1950,MONTHE=11,DATEE=${FORT_DATEE:-27},HOURE=3,/" I
sed -n 109p I
cp $FF/_pristine_restarts/fort1_nov26_itime33312.nc fort.1.nc; cp $FF/_pristine_restarts/fort1_nov26_itime33312.nc fort.2.nc
sh P2SAoM40ln > /dev/null 2>&1
unset FFD_START FFD_NSTEP FFPT_START FFPT_NSTEP FFPT_TAG FFPT_PERT RADSRV_MODE
( time taskset -c $2 ./P2SAoM40 -i I > run.PRT 2>&1 ; echo rc=$? ) > run.time 2>&1
echo "fortran $1 core $2: $(tr '\n' ' ' < run.time)"; grep -i "main loop\|RADIA " run.PRT | head -4
