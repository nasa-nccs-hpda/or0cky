#!/bin/bash
# D207: two-day (108 step) re-run of the D149 instrumented real model, same binary as D149. usage: run108.sh name core pert full
SC=/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d207
SRC=/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0
FF=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data
HS=$SRC/ModelE_Support/huge_space/P2SAoM40
BIN=/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/ac69365f-34cd-40d2-965e-805e4c93b52b/scratchpad/mE_day1/mE2/model/P2SAoM40.bin
export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
NST=108
name=$1; core=$2; pert=$3; full=$4
R=$SC/run_$name; rm -rf $R; mkdir -p $R; cd $R
for f in I P2SAoM40ln P2SAoM40uln runtime_opts; do cp $HS/$f .; done
\cp $BIN P2SAoM40.bin; \cp $BIN P2SAoM40
sed -i '109s/YEARE=1950,MONTHE=12,DATEE=1,HOURE=0,/YEARE=1950,MONTHE=11,DATEE=28,HOURE=6,/' I
grep -n "YEARE=1950,MONTHE=11" I | head -1
rm -f fort.1.nc fort.2.nc; cp $FF/_pristine_restarts/fort1_nov26_itime33312.nc fort.1.nc; cp $FF/_pristine_restarts/fort1_nov26_itime33312.nc fort.2.nc
sh P2SAoM40ln > /dev/null 2>&1
export FFPT_START=33312 FFPT_NSTEP=$NST FFPT_TAG=$name
if [ "$full" = full ]; then export FFD_START=33312 FFD_NSTEP=$NST; else unset FFD_START FFD_NSTEP; fi
if [ -n "$pert" ]; then export FFPT_PERT="$pert"; else unset FFPT_PERT; fi
( time taskset -c $core ./P2SAoM40 -i I > run.PRT 2>&1 ; echo rc=$? ) > run.time 2>&1
