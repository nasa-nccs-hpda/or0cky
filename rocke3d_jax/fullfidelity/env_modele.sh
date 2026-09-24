# Toolchain used to build/run real ModelE (P2SAoM40): Intel ifort 19.1.3 + NetCDF 4.9.3s.
# Source this file:  source fullfidelity/env_modele.sh
# Notes: use Lmod directly (the legacy /usr/share/Modules init hides these modulefiles);
# the intel module's compilervars step does not put ifort on PATH in non-login shells,
# so prepend the compiler bin dir explicitly.
source /usr/share/lmod/lmod/init/bash
module load netcdf4/4.9.3s intel/2020Update4
export PATH=/panfs/ccds02/app/modules/intel/platform/x86_64/rhel/8.6/2020Update4/compilers_and_libraries_2020.4.304/linux/bin/intel64:$PATH
