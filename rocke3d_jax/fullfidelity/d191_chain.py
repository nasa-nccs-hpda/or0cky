"""D191 part 4: chain steps 0..N-1 of one date with the state DEVICE-RESIDENT between the steps (jax_coupled.Coupled; the state pytree of step k is the input of
step k+1, no NumPy round trip), and compare every step with the NumPy chain of d191_ref_chain.py (C1, harness categories).  Only the records of each step (declared
recorded inputs: CONDSE entry set, recorded radiation, SURFACE templates) are read from files.  No claim beyond C1 (ACCEPTANCE section 4).
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 python d191_chain.py DATE OUTDIR REFDIR NSTEPS"""
import clouds_jax_env  # noqa: F401
import json
import os
import sys
import time

import numpy as np

import jax_coupled as C
import jax
import d191_run as R
import jax_harness as H
import d187_common as K
import d187_analyze as AN


def main(date, outdir, refdir, n):
    os.makedirs(outdir, exist_ok=True)
    cp = C.Coupled(date)
    sr = R.build_registry('replay')
    state, rec, dev = cp.initial_state()
    res = dict(date=date, steps=[])
    for k in range(n):
        if k > 0:
            it = cp.it0 + k
            rec = cp.ph.load_records(it, first=False)
            dev = cp.ph.to_device(rec, first=False)
        t0 = time.perf_counter()
        state, info, arrays = cp.step(k, state, rec, dev, sr, keep=True)
        jax.block_until_ready(state)
        wall = time.perf_counter() - t0
        allarr = {}
        for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter', 'X'):
            for kk, v in R.snap_np(arrays[stg]).items():
                allarr[f'{stg}/{kk}'] = v
        SS2, V2n = C.tree_np(arrays['SS2']), C.tree_np(arrays['V2'])
        for grp in ('ocean', 'ice', 'lake', 'li', 'atm'):
            allarr.update(K.flatten(SS2[grp], f'surf/{grp}/'))
        allarr.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy'], usi=V2n['usi'], vsi=V2n['vsi']), 'surf/v2/'))
        allarr.update(K.flatten(C.tree_np(arrays['aux_land']), 'land/'))
        ref = H.load_npz(os.path.join(refdir, f'{date}_chain_ref_step{k}.npz'))
        rows = AN.compare(allarr, ref)
        tot = {c: sum(r['categories'][c] for r in rows.values()) for c in 'ABCD'}
        m = np.asarray(arrays['masks'])
        res['steps'].append(dict(k=k, itime=info['itime'], wall=wall, c1_total=tot, c1=rows, flags=info['flags'], stage_rebuilt=info.get('stage_rebuilt', False),
                                 tile_mask_mismatch=[int(m[0]), int(m[1])], host=info['host'], jit_by_stage=info['jit_by_stage']))
        print(f'step {k} it {info["itime"]} wall {wall:.1f} s  C1 {tot}  not-A: {[(g, list(r["not_A"])[:6]) for g, r in rows.items() if r["not_A"]]}  '
              f'mask mismatch {int(m[0])},{int(m[1])} flags {sum(info["flags"].values())}', flush=True)
        json.dump(res, open(os.path.join(outdir, f'{date}_chain.json'), 'w'), indent=1, default=str)
    print('done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
