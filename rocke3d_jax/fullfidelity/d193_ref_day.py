"""D193 C1 reference for the one-day run: the NumPy libimf chain with the closed surface (as d191_ref_chain.py), steps 0..N-1 of nov26 from the records of
ff_data/nov26_day, radiation frozen between radiation steps (atm_day_open_loop.RealRad, the convention of d193_day.py), the day boundary (it % 48 == 0) handled with
the same DAILY_ATMDYN + ch4ox + SNOAGE reset.  Each step is compared on the spot with the arrays saved by d193_day.py --c1dir (harness categories, d187_analyze.compare);
the reference arrays themselves are not stored.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 python d193_ref_day.py OUTDIR C1DIR NSTEPS"""
import clouds_jax_env  # noqa: F401
import json
import os
import sys
import time

import numpy as np

import atm_step as A
import atm_step_fast as F
import atm_day_open_loop as OL
import surface_loop as L
import surface_loop_v2 as V2
import jax_harness as H
import d187_common as K
import d187_analyze as AN

DATE, DAYDIR = 'nov26', 'nov26_day'


def main(outdir, c1dir, n):
    os.makedirs(outdir, exist_ok=True)
    assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
    it0 = dict(A.DATES)[DATE]
    hdr = H.provenance_header(extra=dict(task='D193 NumPy libimf chain reference (closed surface), one day', date=DATE), require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    ctx = A.make_ctx(DATE, imf=True)
    F.ensure_backend(ctx)
    mdrya, _, _ = OL.daily_mdrya(A.FF, DATE)
    st = L.load_statics(DATE)
    st['ctx'] = L.make_ocean_ctx(DATE)
    SS = L.init_surface_state(DATE, st=st)
    loop = V2.Loop2(DATE, DAYDIR, L.FF, st, SS, flags=None, it0=it0)
    orig_surface = A.stage_surface
    A.stage_surface = loop.stage_surface
    S, ms, rad, out = None, {}, None, []
    try:
        for k in range(n):
            it = it0 + k
            if rad is None or A.is_radiation_step(it):
                rr = A.Real(DAYDIR, it, A.FF).site('r')
                rad = dict(SRHR=np.array(rr['SRHR']), TRHR=np.array(rr['TRHR']))
            R = OL.RealRad(DAYDIR, it, A.FF, rad)
            loop.pre_cse(R)
            rec_daily = None
            if S is not None and it % 48 == 0:
                rec_daily = OL.apply_daily(S, ctx, mdrya)
                dm, _ = OL.ch4ox_increment(A.FF, DAYDIR, it)
                OL.apply_ch4ox(S, ctx, dm)
                carry = S.get('_carry', {})
                if 'SNOAGE' in carry:
                    carry['SNOAGE'] = np.array(R.cse_in['SNOAGE'], copy=True)
            tm = {}
            t0 = time.perf_counter()
            with F.fast_condse():
                S, sn = A.run_step(DATE, it, ctx, R=R, S=S, ms=ms, land_mode='ghy', timing=tm)
            wall = time.perf_counter() - t0
            arrs = {}
            for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter'):
                for kk, v in sn[stg].items():
                    if isinstance(v, np.ndarray) and v.dtype.kind in 'fiub':
                        arrs[f'{stg}/{kk}'] = np.asarray(v)
            for kk, v in S['_condse_X'].items():
                if isinstance(v, np.ndarray):
                    arrs[f'X/{kk}'] = np.asarray(v)
            arrs.update(K.surface_snapshot(loop))
            row = dict(k=k, itime=it, wall=wall, daily=rec_daily)
            p = os.path.join(c1dir, f'{DATE}_day_step{k}.npz')
            if os.path.exists(p):
                z = np.load(p)
                ours = {kk: z[kk] for kk in z.files}
                rows = AN.compare(ours, arrs)
                tot = {c: sum(r['categories'][c] for r in rows.values()) for c in 'ABCD'}
                row['c1_total'] = tot
                row['n_arrays'] = sum(r['n_fields'] for r in rows.values())
                row['not_A'] = {g: r['not_A'] for g, r in rows.items() if r['not_A']}
                only = sorted(set(ours) ^ set(arrs))
                row['keys_on_one_side_only'] = only[:20]
            else:
                row['c1_total'] = None
            X = S.get('_condse_X')
            carry = {key: np.array(X[key], copy=True) for key in A.CARRY_KEYS if X is not None and key in X}
            if '_cloud_rad' in S:
                carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
            S = {key: v for key, v in S.items() if not key.startswith('_')}
            if carry:
                S['_carry'] = carry
            out.append(row)
            print(f'ref step {k} it {it} wall {wall:.1f} s C1 {row["c1_total"]} n={row.get("n_arrays")} not-A groups {list(row.get("not_A", {}))[:8]}', flush=True)
            json.dump(dict(date=DATE, n=n, header=hdr, steps=out, taskset=sorted(os.sched_getaffinity(0))), open(os.path.join(outdir, 'd193_c1.json'), 'w'), indent=1, default=str)
    finally:
        A.stage_surface = orig_surface
    print('done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
