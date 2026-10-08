"""D194 driver: d193_day.main (unchanged) with d194_nit_fix installed (--nitfix 1, default) or not (--nitfix 0).  Usage as d193_day.py plus --nitfix.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d194_run.py OUTDIR --nsteps 11 --c1dir DIR [--nitfix 0|1]"""
import clouds_jax_env  # noqa: F401
import sys
import json
import os
import d194_nit_fix as NF
import d193_day as D

if __name__ == '__main__':
    fix = 1
    if '--nitfix' in sys.argv:
        k = sys.argv.index('--nitfix')
        fix = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    if fix:
        NF.install()
    try:
        D.main()
    finally:
        out = sys.argv[1]
        json.dump(NF.CTX['log'], open(os.path.join(out, 'd194_nitfix_log.json'), 'w'), indent=1)
