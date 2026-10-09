"""D209 driver (copy of d208_day.py; adds --daily-consts computed|recorded: day-boundary MDRYA/ch4ox/SNOAGE computed by drv_daily.daily_on_state_computed (default) or read from records (D208 behaviour). D208 text follows: HEAD abcb0eb, land-fraction transfer in jax_coupled.daily_lake_update, flag --land-fractions, default 1; also saves the carried GHY state after steps 44-53 and before/after the boundary transfer): d193_day.main on the assembled step with the D194 nit rule merged (Coupled(nit_fix=True)), the nit assertion
REPORTED not raised (--nit-strict 0), radiation REPLAYED from the records, and at the day boundary (step 48, it 33360), after DAILY_ATMDYN + ch4ox + SNOAGE of d193_day.daily_on_state,
the Fortran daily_LAKE (daily_lake.py via jax_coupled.Coupled.daily_lake_update) when --daily-lake 1 (default 1).  --daily-lake 0 reproduces d203_day.py.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d209_day.py OUTDIR [--c1dir DIR] [--nit-strict 0|1] [--daily-lake 0|1] [--land-fractions 0|1] [--daily-consts computed|recorded] [--nsteps N]
With --c1dir the FULL C1 arrays of every step are saved (about 24 GB; D209 change vs d205_day.py).
Writes OUTDIR/d209_nit.json (rebuild log, nit mismatches) and OUTDIR/d209_lake.json + d209_lake_hook.npz (the hook's info and arrays)."""
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

SNO_DRIFT = []
HOOK = dict(cp=None, on=True, info=None, land=True, outdir=None)


class CoupledD209(C.Coupled):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        HOOK['cp'] = self

    def step(self, k, state, *a, **kw):
        out = super().step(k, state, *a, **kw)
        if k >= 44 and HOOK['outdir']:                       # D209: the carried GHY state after steps 44-53 (small), for the comparison with the entry records
            dn = out[0]['land_prev']['dyn_next']
            np.savez(os.path.join(HOOK['outdir'], f'land_step{k:02d}.npz'), **{kk: np.asarray(dn[kk]) for kk in ('w', 'ht', 'fr_snow')})
        if HOOK['outdir'] and 'SNOAGE' in out[0]['carry']:       # D209 diagnostic: our carried SNOAGE (CONDSE exit) vs the recorded CONDSE exit of this step
            import clouds_condse_io as cio
            ro = np.asarray(cio.read_cse(f"{cio.FF_DEFAULT}/nov26_day/ffc_cse_out_{33312 + k}.bin")['SNOAGE'])
            d = np.abs(np.asarray(out[0]['carry']['SNOAGE']) - ro)
            SNO_DRIFT.append(dict(k=k, max_abs=float(d.max()), n_diff=int((d > 0).sum()), by_type=[float(d[t].max()) for t in range(d.shape[0])]))
        return out


def allarr_light(arrays):
    allarr = {}
    SS2 = C.tree_np(arrays['SS2'])
    for grp in ('ice', 'lake'):
        allarr.update(K.flatten(SS2[grp], f'surf/{grp}/'))
    V2n = C.tree_np(arrays['V2'])
    allarr.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy']), 'surf/v2/'))
    return allarr


_orig_daily = D.daily_on_state
CONSTS = dict(mode='computed', dh2o=None)


def daily_with_lake(state, dev, ctx_imf, mdrya, it):
    if CONSTS['mode'] == 'computed':                 # D209: MDRYA, ch4ox water mass and SNOAGE computed (drv_daily), no record read
        import drv_daily as DD
        if CONSTS['dh2o'] is None:
            CONSTS['dh2o'] = DD.getqma()
        state, rec = DD.daily_on_state_computed(state, dev, ctx_imf, it, dh2o=CONSTS['dh2o'])
        if 'SNOAGE' in state['carry']:                # comparison only (the record is NOT used for the state): computed vs CONDSE entry record
            rs = np.asarray(dev['cse_in']['SNOAGE']); cs = np.asarray(state['carry']['SNOAGE'])
            rec['snoage_computed_vs_record_max_abs'] = float(np.abs(rs - cs).max()); rec['snoage_computed_vs_record_bitwise'] = bool(np.array_equal(rs, cs))
    else:
        state, rec = _orig_daily(state, dev, ctx_imf, mdrya, it)
    if HOOK['on']:
        cp = HOOK['cp']
        dn0 = state['land_prev']['dyn_next']
        np.savez(os.path.join(HOOK['outdir'], 'land_boundary_before.npz'), **{kk: np.asarray(dn0[kk]) for kk in ('w', 'ht', 'fr_snow')})
        state, info = cp.daily_lake_update(state, land_fractions=HOOK['land'])
        dn1 = state['land_prev']['dyn_next']
        np.savez(os.path.join(HOOK['outdir'], 'land_boundary_after.npz'), **{kk: np.asarray(dn1[kk]) for kk in ('w', 'ht', 'fr_snow')},
                 land3_w=np.asarray(cp._land3['w']) if cp._land3 else 0, land3_ht=np.asarray(cp._land3['ht']) if cp._land3 else 0)
        HOOK['info'] = info
        rec['daily_lake'] = dict(n_flake_changed=info['n_flake_changed'], max_abs_dflake=info['max_abs_dflake'], n_rsi_changed=info['n_rsi_changed'],
                                 n_dmwldf_pos=info['n_dmwldf_pos'], counters=info['counters'], seconds=info['seconds'], pow=info['pow'])
    return state, rec


if __name__ == '__main__':
    strict, on, landf = 1, 1, 1
    for flag in ('--nit-strict', '--daily-lake', '--land-fractions', '--daily-consts'):
        if flag in sys.argv:
            k = sys.argv.index(flag)
            v = sys.argv[k + 1] if flag == '--daily-consts' else int(sys.argv[k + 1])
            del sys.argv[k:k + 2]
            if flag == '--daily-consts':
                assert v in ('computed', 'recorded')
                CONSTS['mode'] = v
            elif flag == '--nit-strict':
                strict = v
            elif flag == '--land-fractions':
                landf = v
            else:
                on = v
    HOOK['on'] = bool(on)
    HOOK['land'] = bool(landf)
    C.Coupled = functools.partial(CoupledD209, nit_strict=bool(strict))
    D.daily_on_state = daily_with_lake
    # D209: --c1dir keeps the FULL per-step arrays (needed for C1 against the NumPy chain); allarr_light unused
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    HOOK['outdir'] = out
    try:
        D.main()
    finally:
        json.dump(dict(daily_consts=CONSTS['mode'], strict=strict, daily_lake=on, land_fractions=landf, rebuild_log=JS.NIT_LOG, mismatch=JS.NIT_MISMATCH), open(os.path.join(out, 'd209_nit.json'), 'w'), indent=1)
        json.dump(SNO_DRIFT, open(os.path.join(out, 'd209_snoage_drift.json'), 'w'), indent=1)
        info = HOOK['info']
        if info is not None:
            json.dump({k: v for k, v in info.items() if not isinstance(v, np.ndarray)}, open(os.path.join(out, 'd209_lake.json'), 'w'), indent=1, default=str)
            np.savez(os.path.join(out, 'd209_lake_hook.npz'), dmwldf=info['dmwldf'], dgml=info['dgml'], svflake=info['svflake'], mdwnimp=info['mdwnimp'], edwnimp=info['edwnimp'],
                     flake=HOOK['cp'].st['geo']['flake'], fland=HOOK['cp'].st['fland'], fearth=HOOK['cp'].st['fearth'])
