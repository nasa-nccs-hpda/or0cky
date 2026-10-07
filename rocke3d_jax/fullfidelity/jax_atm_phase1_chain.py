"""D186: phase 1 on the device for SEVERAL consecutive steps, with the surface half on the record boundary / host:
  per step k:  [device] MELT_SI (step 0 only), DYNAM, CONDSE, RADIA apply   ->   [host, existing NumPy code, DECLARED] atm_step.stage_surface (land recorded),
  stage_dissip, stage_filter   ->   next step's state.
So the state crosses the host at every step (device->host after phase 1, host->device before the next): this hybrid chain is NOT a device-resident multi-step run;
it exists to test C1 over several steps (the NumPy libm chain of jax_p1_ref_chain.py is the reference).  The surface half runs the SAME code on both sides, so a
difference can only come from phase 1.
Run: taskset -c 0-1 env OMP_NUM_THREADS=1 python jax_atm_phase1_chain.py DATE OUTDIR REFDIR NSTEPS"""
import clouds_jax_env_fast  # noqa: F401
import json
import os
import sys
import time

import numpy as np

import jax_atm_phase1 as P1
import jax_atm_phase1_run as RUN
import jax_harness as H
import jax
import jax.numpy as jnp
import atm_step as A


def main(date, outdir, refdir, n):
    os.makedirs(outdir, exist_ok=True)
    ph = P1.Phase1(date)
    it0 = ph.it0
    ctx = ph.ctx
    carry, ms, hold, ice, S_np = {}, P1.CS.ms_zero(), None, None, None
    out = dict(date=date, steps=[])
    for k in range(n):
        it = it0 + k
        ref = H.load_npz(os.path.join(refdir, f'{date}_chain_step{k}.npz'))
        rec = ph.load_records(it, first=(k == 0))
        dev = ph.to_device(rec, first=(k == 0))
        if k == 0:
            S_dev, hold, ice = dev['S'], ph.new_hold(dev['rad']), dev['ice']
        else:
            S_dev = {key: jnp.asarray(v) for key, v in S_np.items() if isinstance(v, np.ndarray)}
        t0 = time.perf_counter()
        r = ph.step(it, dev, S_dev, carry, ms, hold, ice, rec=rec, do_melt=(k == 0))
        jax.block_until_ready((r['S'], r['X']))
        t_dev = time.perf_counter() - t0
        carry, ms, hold, ice = r['carry'], r['ms'], r['hold'], r['ice']
        fl = P1.D.flags_to_host(r['flags'])
        comp = {}
        for st in ('dyn', 'condse', 'radia'):
            res, miss = RUN.compare_stage(RUN.to_np(r['snaps'][st]), ref, st)
            cats, notA = RUN.summarize(res)
            comp[st] = dict(n=len(res), categories=cats, not_A=notA, missing=miss)
        resX, miss = RUN.compare_stage(RUN.to_np(r['X']), ref, 'X')
        cats, notA = RUN.summarize(resX)
        comp['X'] = dict(n=len(resX), categories=cats, not_A=notA, missing=miss)
        if 'rad/CLDSS' in ref and A.is_radiation_step(it):
            comp['rad_cloud_mask'] = {kk: H.field_category(np.asarray(r['cloud_masked'][kk]), ref['rad/' + kk])['cat'] for kk in ('CLDSS', 'CLDMC')}
        # host surface half (declared)
        t0 = time.perf_counter()
        S = {key: np.array(v) for key, v in r['S'].items() if hasattr(v, 'shape')}
        Rr = rec['R']
        tm = {}
        A.stage_surface(S, Rr, ctx, tm=tm, land_mode='recorded')
        A.stage_dissip(S, ctx)
        A.stage_filter(S, ctx)
        A._native(S)
        S_np = {key: v for key, v in S.items() if not key.startswith('_')}
        t_host = time.perf_counter() - t0
        out['steps'].append(dict(k=k, itime=it, radiation_step=bool(A.is_radiation_step(it)), flags=fl, device_phase1_seconds=t_dev, host_surface_half_seconds=t_host,
                                 comparison=comp))
        print(k, it, 'dev %.1f s host %.1f s' % (t_dev, t_host), {s: comp[s]['categories'] for s in ('dyn', 'condse', 'radia', 'X')}, flush=True)
        json.dump(out, open(os.path.join(outdir, f'{date}_chain_phase1.json'), 'w'), indent=1, default=str)
    print('done')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
