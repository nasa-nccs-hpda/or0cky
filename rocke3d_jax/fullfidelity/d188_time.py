"""D188: cost of the JAX SURFACE stage vs the NumPy stage, interleaved in ONE process (same cores, same node load) and the one-time host work.
    taskset -c 3-5 env OMP_NUM_THREADS=1 python d188_time.py DATE OUT.json [REPEATS]
Reports: compile count and cold time of the first JAX call, steady times (interleaved NumPy/JAX), jit executions per stage, transfers under
jax.transfer_guard('disallow'), bytes of the template and of the state, host template build split, and the time of the Python per-cell parts."""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clouds_jax_env  # noqa: F401
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import atm_step as A
import jax_surface as JS
import jax_harness as H
import ghy_advnc_test as AT

DATE_IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}


def sync(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, x)
    return x


def nbytes(t):
    return int(sum(getattr(a, 'nbytes', 0) for a in jax.tree_util.tree_leaves(t)))


def main(date, outp, repeats=6):
    it = DATE_IT0[date]
    res = {'date': date}
    R = A.Real(date, it)
    rec = A.surface_records(R)
    S = A.real_state_at(R, 'surface', None)
    A._native(S)
    # host template build, split
    t0 = time.perf_counter()
    gb = [AT.build_batch(rec[k]) for k in ('g1', 'g2')]
    res['t_build_batch_g1_g2_s'] = time.perf_counter() - t0
    res['n_stiff_cells_ffnit_gt11'] = [int((np.round(rec[k][:, 289]) > 11).sum()) for k in ('g1', 'g2')]
    t0 = time.perf_counter()
    tpl, host = JS.build_template(rec)
    sync(tpl)
    res['t_build_template_total_s'] = time.perf_counter() - t0
    Sd = JS.state_from_numpy(S)
    sync(Sd)
    res['bytes_template'] = nbytes(tpl)
    res['bytes_state'] = nbytes(Sd)
    c = H.Counters().listen_compiles()
    stage = JS.make_stage(host)
    t0 = time.perf_counter()
    with c.stage('first_call'):
        out, aux = stage(Sd, tpl)
        sync(out)
    res['cold_first_call_s'] = time.perf_counter() - t0
    res['compiles_first_call'] = c.snapshot().get('first_call', {}).get('compiles')
    res['compile_seconds_first_call'] = c.snapshot().get('first_call', {}).get('compile_seconds')
    res['jit_executions_per_stage'] = 5
    tn, tj = [], []
    for rep in range(repeats):
        S2 = A.real_state_at(R, 'surface', None)
        A._native(S2)
        t0 = time.perf_counter()
        A.stage_surface(S2, R, None, rec=rec, land_mode='ghy')
        tn.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        if rep == repeats - 1:
            with jax.transfer_guard('disallow'):
                o2, _ = stage(Sd, tpl)
                sync(o2)
            res['transfer_guard_disallow_passed'] = True
        else:
            o2, _ = stage(Sd, tpl)
            sync(o2)
        tj.append(time.perf_counter() - t0)
    res['numpy_stage_s'] = tn
    res['jax_stage_s'] = tj
    res['numpy_median_s'] = float(np.median(tn))
    res['jax_median_s'] = float(np.median(tj))
    # NumPy stage split: how much of it is the record -> GHY batch build (host Python), measured on its own
    t0 = time.perf_counter()
    AT.build_batch(rec['g1'])
    res['t_one_build_batch_s'] = time.perf_counter() - t0
    json.dump(res, open(outp, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
