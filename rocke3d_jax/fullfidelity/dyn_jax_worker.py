"""D175: the JAX dynamics step (dyn_step_jax2, D139-D144, bitwise vs the numpy-pow dynamics only under the FMA-free XLA flag) in a
dedicated worker process, so that the flag `--xla_cpu_max_isa=AVX` (dyn_jax_env) does NOT leak into the other JAX stages of the coupled
step (PBL, tile chain, ocean), which run in the caller without it exactly as before.

stage_dyn drop-in: make_stage_dyn(date, ff, imf) -> (stage_dyn(S, R, ctx, tm=None), close).  Install with atm_step.stage_dyn = stage_dyn.
Cores: 1 worker process (XLA may use more threads inside it; pin with taskset).  First call compiles (~tens of seconds), then steady.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _worker(conn, date, ff, imf):
    import dyn_jax_env  # noqa: F401  (sets XLA_FLAGS before jax is imported)
    import dyn_step as ds
    import dyn_step_jax2 as dj2
    ctx = ds.load_ctx(date, ff, imf_pow=bool(imf))
    kit = dj2.Kit(ctx)
    conn.send("ready")
    while True:
        m = conn.recv()
        if m is None:
            return
        state, itime = m
        w = dj2.dyn_step_jax(state, ctx, kit, itime=itime)
        conn.send({k: np.asarray(v) for k, v in w.items()})


def make_stage_dyn(date, ff, imf=True):
    import multiprocessing as mp
    import atm_step as A
    c = mp.get_context("spawn")
    a, b = c.Pipe()
    p = c.Process(target=_worker, args=(b, date, ff, imf), daemon=True)
    p.start()
    assert a.recv() == "ready"

    def stage_dyn(S, R, ctx, tm=None):
        a.send(({k: S[k.upper()] for k in A.ds.STATE_KEYS}, R.itime))
        w = a.recv()
        for k in A.DYN_OUT:
            if k in w:
                S[k] = np.array(w[k], copy=True)
        S['PEK'] = A._pow(ctx, S['PEDN'], ctx.kapa)
        return S

    def close():
        try:
            a.send(None)
        except Exception:
            pass
        p.join(timeout=10)
    return stage_dyn, close
