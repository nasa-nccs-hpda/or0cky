"""D206 driver (copy of d205_day.py; run on HEAD 256aa41 = D204 computed GHY schedule default + D205 lake hook: the combined run): d193_day.main on the assembled step with the D194 nit rule merged (Coupled(nit_fix=True)), the nit assertion
REPORTED not raised (--nit-strict 0), radiation REPLAYED from the records, and at the day boundary (step 48, it 33360), after DAILY_ATMDYN + ch4ox + SNOAGE of d193_day.daily_on_state,
the Fortran daily_LAKE (daily_lake.py via jax_coupled.Coupled.daily_lake_update) when --daily-lake 1 (default 1).  --daily-lake 0 reproduces d203_day.py.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d206_day.py OUTDIR [--c1dir DIR] [--nit-strict 0|1] [--daily-lake 0|1] [--nsteps N]
With --c1dir the FULL C1 arrays of every step are saved (about 24 GB; D206 change vs d205_day.py).
Writes OUTDIR/d206_nit.json (rebuild log, nit mismatches) and OUTDIR/d206_lake.json + d206_lake_hook.npz (the hook's info and arrays)."""
import clouds_jax_env  # noqa: F401
import functools
import json
import os
import sys

import numpy as np

import jax_coupled as C
import jax_surface as JS
import d193_day as D
import d187_common as K
import d191_run as R191

HOOK = dict(cp=None, on=True, info=None)


class CoupledD206(C.Coupled):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        HOOK['cp'] = self


def allarr_light(arrays):
    allarr = {}
    SS2 = C.tree_np(arrays['SS2'])
    for grp in ('ice', 'lake'):
        allarr.update(K.flatten(SS2[grp], f'surf/{grp}/'))
    V2n = C.tree_np(arrays['V2'])
    allarr.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy']), 'surf/v2/'))
    return allarr


_orig_daily = D.daily_on_state


def daily_with_lake(state, dev, ctx_imf, mdrya, it):
    state, rec = _orig_daily(state, dev, ctx_imf, mdrya, it)
    if HOOK['on']:
        cp = HOOK['cp']
        state, info = cp.daily_lake_update(state)
        HOOK['info'] = info
        rec['daily_lake'] = dict(n_flake_changed=info['n_flake_changed'], max_abs_dflake=info['max_abs_dflake'], n_rsi_changed=info['n_rsi_changed'],
                                 n_dmwldf_pos=info['n_dmwldf_pos'], counters=info['counters'], seconds=info['seconds'], pow=info['pow'])
    return state, rec


if __name__ == '__main__':
    strict, on = 1, 1
    for flag in ('--nit-strict', '--daily-lake'):
        if flag in sys.argv:
            k = sys.argv.index(flag)
            v = int(sys.argv[k + 1])
            del sys.argv[k:k + 2]
            if flag == '--nit-strict':
                strict = v
            else:
                on = v
    HOOK['on'] = bool(on)
    C.Coupled = functools.partial(CoupledD206, nit_strict=bool(strict))
    D.daily_on_state = daily_with_lake
    # D206: --c1dir keeps the FULL per-step arrays (needed for C1 against the NumPy chain); allarr_light unused
    out = sys.argv[1]
    try:
        D.main()
    finally:
        json.dump(dict(strict=strict, daily_lake=on, rebuild_log=JS.NIT_LOG, mismatch=JS.NIT_MISMATCH), open(os.path.join(out, 'd206_nit.json'), 'w'), indent=1)
        info = HOOK['info']
        if info is not None:
            json.dump({k: v for k, v in info.items() if not isinstance(v, np.ndarray)}, open(os.path.join(out, 'd206_lake.json'), 'w'), indent=1, default=str)
            np.savez(os.path.join(out, 'd206_lake_hook.npz'), dmwldf=info['dmwldf'], dgml=info['dgml'], svflake=info['svflake'], mdwnimp=info['mdwnimp'], edwnimp=info['edwnimp'],
                     flake=HOOK['cp'].st['geo']['flake'], fland=HOOK['cp'].st['fland'], fearth=HOOK['cp'].st['fearth'])
