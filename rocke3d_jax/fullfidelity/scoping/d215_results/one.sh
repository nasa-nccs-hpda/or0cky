#!/bin/bash
# usage: one.sh <tag> <binary> <core>   -- 54-step nov26 day, single process pinned to <core>
SC=/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d215
FF=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data
HS=/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/huge_space/P2SAoM40
export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
R=$SC/run_$1; rm -rf $R; mkdir -p $R; cd $R
for f in I P2SAoM40ln P2SAoM40uln runtime_opts; do cp $HS/$f .; done
\cp $2 P2SAoM40.bin; \cp $2 P2SAoM40
sed -i '109s/YEARE=1950,MONTHE=12,DATEE=1,HOURE=0,/YEARE=1950,MONTHE=11,DATEE=27,HOURE=3,/' I
sed -n 109p I > I109.txt
cp $FF/_pristine_restarts/fort1_nov26_itime33312.nc fort.1.nc; cp $FF/_pristine_restarts/fort1_nov26_itime33312.nc fort.2.nc
sh P2SAoM40ln > /dev/null 2>&1
unset FFD_START FFD_NSTEP FFPT_START FFPT_NSTEP FFPT_TAG FFPT_PERT RADSRV_MODE
( time taskset -c $3 ./P2SAoM40 -i I > run.PRT 2>&1 ; echo rc=$? ) > run.time 2>&1
