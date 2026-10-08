"""D195 driver: d193_day.main on the assembled step with the D194 nit rule merged (Coupled(nit_fix=True), default).  --nit-strict 1 (default) = the
assertion nit == ffnit is not caught (stops the day at step 23, it 33335, cell 111); --nit-strict 0 = the mismatch is REPORTED (jax_surface.NIT_MISMATCH) and the
day continues.  Writes OUTDIR/d195_nit.json (rebuild log, mismatches).
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d195_day.py OUTDIR --c1dir DIR [--nit-strict 0|1]"""
import clouds_jax_env  # noqa: F401
import functools
import json
import os
import sys

import jax_coupled as C
import jax_surface as JS
import d193_day as D

if __name__ == '__main__':
    strict = 1
    if '--nit-strict' in sys.argv:
        k = sys.argv.index('--nit-strict')
        strict = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    C.Coupled = functools.partial(C.Coupled, nit_strict=bool(strict))
    try:
        D.main()
    finally:
        json.dump(dict(strict=strict, rebuild_log=JS.NIT_LOG, mismatch=JS.NIT_MISMATCH), open(os.path.join(sys.argv[1], 'd195_nit.json'), 'w'), indent=1)
