"""D207 driver (copy of d206_day.py; no tracked file edited): the D206 combined day generalised to a TWO-DAY window (default 108 steps, it 33312-33419) against the
second-day record set ff_data/nov26_day2 (instrumented real model, FFD_NSTEP=108).  Changes against d206_day.py:
  * record directory: d193_day.DAYDIR is set to --daydir (default nov26_day2) before d193_day.main (DayReal, ch4ox_increment read it at call time);
  * --nsteps default 108; the day-boundary test of d193_day (`it % 48 == 0`, k > 0) fires at step 48 (it 33360) AND step 96 (it 33408);
  * at EVERY boundary: DAILY_ATMDYN with MDRYA (the dry-mass constant; the only recorded value is the nov26 ffd_glue_daily dump, the day-2 records carry no glue dump),
    ch4ox from the record pair (ffa_step_<it-1>_e, ffa_step_<it>_a) of the NEW record set, recorded SNOAGE, then the Fortran daily_LAKE hook;
  * per boundary, a diagnostic against the real records: our MA/Q/P change across the boundary versus the real change (ffa_step_<it>_a - ffa_step_<it-1>_e);
  * outputs d207_nit.json, d207_lake_<it>.json / d207_lake_hook_<it>.npz (one per boundary), d207_boundaries.json.
    taskset -c 3-5 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d207_day.py OUTDIR [--daydir nov26_day2] [--c1dir DIR] [--nit-strict 0|1] [--daily-lake 0|1] [--nsteps N]
Writes OUTDIR/ours_d193/step_<it>.npz (kept name of d193_day), OUTDIR/d193_run.json."""
import clouds_jax_env  # noqa: F401
import functools
import json
import os
import sys

import numpy as np

import jax_coupled as C
import jax_surface as JS
import d193_day as D
import atm_step as A
import clouds_condse_io as cio

HOOK = dict(cp=None, on=True, infos={})
BOUNDARIES = {}


class CoupledD207(C.Coupled):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        HOOK['cp'] = self


_orig_daily = D.daily_on_state


def _boundary_check(state_before, state_after, it):
    """our change across the boundary vs the real change between ffa_step_<it-1>_e and ffa_step_<it>_a (MA, Q, P, PEDN where present)."""
    e = cio.read_cse(f"{A.FF}/{D.DAYDIR}/ffa_step_{it - 1}_e.bin")
    a = cio.read_cse(f"{A.FF}/{D.DAYDIR}/ffa_step_{it}_a.bin")
    out = {}
    for k in ('MA', 'Q', 'P', 'PEDN', 'PK', 'PMID'):
        if k not in e or k not in a or k not in state_before['S']:
            continue
        real_d = np.asarray(a[k], dtype=np.float64) - np.asarray(e[k], dtype=np.float64)
        ours_d = np.asarray(state_after['S'][k], dtype=np.float64) - np.asarray(state_before['S'][k], dtype=np.float64)
        if real_d.shape != ours_d.shape:
            out[k] = dict(shape_real=list(real_d.shape), shape_ours=list(ours_d.shape))
            continue
        out[k] = dict(real_rms=float(np.sqrt((real_d ** 2).mean())), real_max=float(np.abs(real_d).max()), ours_rms=float(np.sqrt((ours_d ** 2).mean())),
                      ours_max=float(np.abs(ours_d).max()), diff_rms=float(np.sqrt(((ours_d - real_d) ** 2).mean())), diff_max=float(np.abs(ours_d - real_d).max()))
    return out


def daily_with_lake(state, dev, ctx_imf, mdrya, it):
    before = state
    state, rec = _orig_daily(state, dev, ctx_imf, mdrya, it)
    try:
        chk = _boundary_check(before, state, it)
    except Exception as e:                                   # report, do not hide
        chk = dict(error=repr(e))
    rec['boundary_vs_real'] = chk
    if HOOK['on']:
        cp = HOOK['cp']
        state, info = cp.daily_lake_update(state)
        HOOK['infos'][it] = info
        rec['daily_lake'] = dict(n_flake_changed=info['n_flake_changed'], max_abs_dflake=info['max_abs_dflake'], n_rsi_changed=info['n_rsi_changed'],
                                 n_dmwldf_pos=info['n_dmwldf_pos'], counters=info['counters'], seconds=info['seconds'], pow=info['pow'])
    BOUNDARIES[it] = rec
    return state, rec


def _pop(flag, default, cast):
    if flag in sys.argv:
        k = sys.argv.index(flag)
        v = cast(sys.argv[k + 1])
        del sys.argv[k:k + 2]
        return v
    return default


if __name__ == '__main__':
    strict = _pop('--nit-strict', 1, int)
    on = _pop('--daily-lake', 1, int)
    D.DAYDIR = _pop('--daydir', 'nov26_day2', str)
    if '--nsteps' not in sys.argv:
        sys.argv += ['--nsteps', '108']
    HOOK['on'] = bool(on)
    C.Coupled = functools.partial(CoupledD207, nit_strict=bool(strict))
    D.daily_on_state = daily_with_lake
    out = sys.argv[1]
    try:
        D.main()
    finally:
        os.makedirs(out, exist_ok=True)
        json.dump(dict(strict=strict, daily_lake=on, daydir=D.DAYDIR, rebuild_log=JS.NIT_LOG, mismatch=JS.NIT_MISMATCH), open(os.path.join(out, 'd207_nit.json'), 'w'), indent=1)
        json.dump(BOUNDARIES, open(os.path.join(out, 'd207_boundaries.json'), 'w'), indent=1, default=str)
        for it, info in HOOK['infos'].items():
            json.dump({k: v for k, v in info.items() if not isinstance(v, np.ndarray)}, open(os.path.join(out, f'd207_lake_{it}.json'), 'w'), indent=1, default=str)
            np.savez(os.path.join(out, f'd207_lake_hook_{it}.npz'), dmwldf=info['dmwldf'], dgml=info['dgml'], svflake=info['svflake'], mdwnimp=info['mdwnimp'], edwnimp=info['edwnimp'])
        if HOOK['cp'] is not None:
            np.savez(os.path.join(out, 'd207_final_statics.npz'), flake=HOOK['cp'].st['geo']['flake'], fland=HOOK['cp'].st['fland'], fearth=HOOK['cp'].st['fearth'])
