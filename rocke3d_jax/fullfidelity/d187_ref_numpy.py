"""D187 C1 reference: the NumPy chained coupled step in libimf mode, step 0 of a date, with the closed surface of surface_loop_v2 (Loop2).

Same stages as the hybrid step (d187_coupled_step.py): dyn, condse (batched NumPy), radia (recorded SRHR/TRHR/COSZ1), surface (PBL, tiles, GHY land with
recorded Ent exports, GROUND_*, RIVERF, DYNSI, ocean, FORM_SI, ADVSI), dissip, filter; ctx imf=True (Intel libimf pow/exp through ctypes).  MELT_SI is the NumPy
surface_loop.melt_si (Loop2.pre_cse), as in model_driver.ModelDriver.step.
Run (same pinning as the hybrid run; the hybrid needs >= 2 cores, so use three):
    taskset -c 0-2 env OMP_NUM_THREADS=1 python d187_ref_numpy.py DATE OUTDIR
Writes OUTDIR/<date>_ref.npz (stage snapshots dyn/condse/radia/surface/dissip/filter, X/ = CONDSE outputs, surf/ and land/ = surface state after the step),
OUTDIR/<date>_ref.json (timing, C2 numbers of this chain: atmosphere gate and 30 end fields, surface state vs the real records).
No compile cache; SOCRATES/RADIA not involved (radiation recorded)."""
import clouds_jax_env  # noqa: F401  (XLA flags BEFORE jax)
import json
import os
import sys
import time

import numpy as np

import atm_step as A
import atm_step_fast as F
import surface_loop as L
import surface_loop_v2 as V2
import jax_harness as H
import d187_common as K


def main(date, outdir):
    os.makedirs(outdir, exist_ok=True)
    assert 'JAX_COMPILATION_CACHE_DIR' not in os.environ and not os.environ.get('CLOUDS_JAX_CACHE')
    it0 = dict(A.DATES)[date]
    hdr = H.provenance_header(extra=dict(task='D187 NumPy libimf reference with closed surface', date=date), require_xla_flags=True)
    print(H.header_text(hdr), flush=True)
    t_all = time.perf_counter()
    ctx = A.make_ctx(date, imf=True)
    F.ensure_backend(ctx)
    t0 = time.perf_counter()
    st = L.load_statics(date)
    st['ctx'] = L.make_ocean_ctx(date)
    SS = L.init_surface_state(date, st=st)
    loop = V2.Loop2(date, date, L.FF, st, SS, flags=None, it0=it0)
    t_setup = time.perf_counter() - t0
    R = A.Real(date, it0)
    orig_surface = A.stage_surface
    A.stage_surface = loop.stage_surface
    tm = {}
    try:
        t0 = time.perf_counter()
        loop.pre_cse(R)                                           # MELT_SI (NumPy)
        t_melt = time.perf_counter() - t0
        t0 = time.perf_counter()
        with F.fast_condse():
            S, sn = A.run_step(date, it0, ctx, R=R, ms={}, land_mode='ghy', timing=tm)
        t_step = time.perf_counter() - t0
    finally:
        A.stage_surface = orig_surface
    arrs = {}
    for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter'):
        for k, v in sn[stg].items():
            if isinstance(v, np.ndarray) and v.dtype.kind in 'fiub':
                arrs[f'{stg}/{k}'] = np.asarray(v)
    for k, v in S['_condse_X'].items():
        if isinstance(v, np.ndarray):
            arrs[f'X/{k}'] = np.asarray(v)
    arrs.update(K.surface_snapshot(loop))
    np.savez(os.path.join(outdir, f'{date}_ref.npz'), **arrs)
    # ---- C2 of this chain (atmosphere + surface), same code as the hybrid
    end = K.numeric_state(sn['filter'])
    ref = {k: np.asarray(v) for k, v in A.end_reference(R).items() if isinstance(v, (np.ndarray, np.generic))}
    gate = H.compare_end_state(end, ref)
    end30 = {f: K.slim(H.field_category(end[f], ref[f])) for f in A.END_FIELDS if f in end and f in ref}
    t0 = time.perf_counter()
    c2s = K.surface_c2(date, it0, loop)
    t_c2s = time.perf_counter() - t0
    out = dict(date=date, itime=it0, header=hdr, mode='NumPy chain, libimf (ctx imf=True), closed surface (surface_loop_v2.Loop2), land ghy, radiation recorded',
               seconds=dict(setup_surface=t_setup, melt_si=t_melt, run_step=t_step, total=time.perf_counter() - t_all, c2_surface_compare=t_c2s),
               stage_seconds={k: v for k, v in tm.items()},
               atm_gate=dict(verdict=gate['verdict'], categories=gate['categories'], gate_fields={f: K.slim(v) for f, v in gate['gate_fields'].items()},
                             exception_columns=gate['exception_columns']),
               atm_end30=dict(categories=K.count_categories(end30) if False else {c: sum(1 for v in end30.values() if v['cat'] == c) for c in 'ABCD'}, fields=end30),
               surface_c2=c2s, surface_c2_categories=K.summarize_surface_c2(c2s),
               taskset=sorted(os.sched_getaffinity(0)), XLA_FLAGS=os.environ.get('XLA_FLAGS'))
    json.dump(out, open(os.path.join(outdir, f'{date}_ref.json'), 'w'), indent=1, default=str)
    print(date, 'ref done: gate', gate['verdict'], gate['categories'], 'end30', out['atm_end30']['categories'], 'step %.1f s' % t_step, flush=True)
    print('surface C2 categories', out['surface_c2_categories'], flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
