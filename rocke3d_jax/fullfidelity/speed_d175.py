"""D175: timed, optionally parallel coupled step loop (surface_loop_v2.Loop2 + Ent land + batched CONDSE), per-stage wall times and a
state dump per step for bitwise comparison of the serial and the parallel variants.

  python speed_d175.py MODE NPROC NSTEPS OUTDIR     MODE = ref (existing land_chain_ent + atm_step_fast.fast_condse, serial)
                                                           par (land_ent_par + clouds_condse_par with NPROC workers each)
Writes OUTDIR/step_<k>.pkl (atmosphere end state + whole surface state) and OUTDIR/timing.json.  Core limit is the caller's (taskset).
Same recorded inputs as surface_loop_v2.run_coupled_v2 (nov26, recorded radiation frozen as in D150).
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(mode, nproc, nsteps, outdir, date='nov26', daydir='nov26_day', it0=33312):
    import surface_loop as L
    import surface_loop_v2 as V2
    import atm_step as A_
    import atm_step_fast as F
    import atm_day_open_loop as OL
    import land_chain_ent as LE
    os.makedirs(outdir, exist_ok=True)
    ff = L.FF
    t00 = time.perf_counter()
    ctx = A_.make_ctx(date, imf=True, ff=ff)
    F.ensure_backend(ctx)
    st = L.load_statics(date, ff); st['ctx'] = L.make_ocean_ctx(date, ff)
    SS = L.init_surface_state(date, ff, st)
    loop = V2.Loop2(date, daydir, ff, st, SS, flags=None, it0=it0)
    A_.stage_surface = loop.stage_surface
    if mode == 'par':
        import land_ent_par as LP
        import clouds_condse_par as CP
        ent = LP.ParEntLand(LE.RESTART['nov26_day'], nproc)
        undo = LP.install_par(A_, ent)
        cs = lambda: CP.par_condse(nproc, 'imf')          # noqa: E731
    else:
        ent = LE.EntLand(LE.RESTART['nov26_day'])
        undo = LE.install(A_, ent)
        cs = F.fast_condse
    setup = time.perf_counter() - t00
    S, ms, rad, rows = None, {}, None, []
    for k in range(nsteps):
        it = it0 + k
        if rad is None or A_.is_radiation_step(it):
            rr = A_.Real(daydir, it, ff).site('r'); rad = dict(SRHR=np.array(rr['SRHR']), TRHR=np.array(rr['TRHR']))
        R = OL.RealRad(daydir, it, ff, rad)
        tm = {}
        t0 = time.perf_counter()
        loop.pre_cse(R)
        with cs():
            S, sn = A_.run_step(date, it, ctx, R=R, S=S, ms=ms, land_mode='ghy', timing=tm)
        wall = time.perf_counter() - t0
        X = S.get('_condse_X')
        carry = {key: np.array(X[key], copy=True) for key in A_.CARRY_KEYS if X is not None and key in X}
        if '_cloud_rad' in S:
            carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
        S = {key: v for key, v in S.items() if not key.startswith('_')}
        if carry:
            S['_carry'] = carry
        pickle.dump(dict(atm={f: np.asarray(S[f]) for f in ('T', 'Q', 'U', 'V', 'P', 'QCL', 'QCI')}, surf=loop.SS), open(f"{outdir}/step_{k}.pkl", 'wb'))
        r = dict(step=k, wall=wall, dyn=tm.get('stage_dyn'), condse=tm.get('stage_condse'), stage_surface=tm.get('stage_surface'),
                 surface_tiles_land=tm.get('surface'), filter=tm.get('stage_filter'))
        rows.append(r)
        print(r, flush=True)
    undo()
    json.dump(dict(mode=mode, nproc=nproc, setup=setup, rows=rows), open(f"{outdir}/timing.json", 'w'))
    return rows


def compare(dir_a, dir_b, nsteps):
    """Bitwise comparison of the dumped states; returns (n_arrays, n_differing, list of (step, name, max|diff|))."""
    def flat(d, pre=''):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, pre + k + '.')
            elif isinstance(v, np.ndarray):
                yield pre + k, v
    n = nd = 0
    diffs = []
    for k in range(nsteps):
        a = pickle.load(open(f"{dir_a}/step_{k}.pkl", 'rb')); b = pickle.load(open(f"{dir_b}/step_{k}.pkl", 'rb'))
        fa, fb = dict(flat(a)), dict(flat(b))
        for name, x in fa.items():
            y = fb[name]
            n += 1
            if not np.array_equal(x, y, equal_nan=True):
                nd += 1
                diffs.append((k, name, float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))))
    return n, nd, diffs


if __name__ == '__main__':
    if sys.argv[1] == 'cmp':
        print(compare(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
