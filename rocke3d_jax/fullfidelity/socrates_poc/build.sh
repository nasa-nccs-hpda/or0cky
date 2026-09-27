#!/bin/bash
# Build gauss_angle_driver: links real SOCRATES object code (libsocrates.a) into a small
# stdin/stdout-driven executable. Run from this directory after `source ../env_modele.sh`.
set -e
LIBDIR=/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model/socrates
ifort -O2 -I"$LIBDIR" gauss_angle_driver.f90 "$LIBDIR/libsocrates.a" -o gauss_angle_driver
echo "built gauss_angle_driver -- run: LD_LIBRARY_PATH=$LIBDIR python3 socrates_py.py"
