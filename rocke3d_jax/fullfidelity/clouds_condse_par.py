"""D175: drop-in parallel variant of clouds_condse_batch.condse_step_batch.

MSTCNV (clouds_mstcnv_batch.mstcnv_batch) is 9-10 s of the ~11 s batched CONDSE step.  Its columns are independent (lock-step masks,
elementwise numpy; the only cross-column object is the LSCOND carry `ms`, which MSTCNV does not touch), so the N non-polar columns are
dealt out round-robin (column n -> worker n % nproc, which balances the convecting tropical columns) to persistent worker processes, each
runs the unchanged mstcnv_batch on its slice, and the outputs are re-interleaved.  Everything else (set-up, LSCOND, poles, hand-off) is
the existing condse_step_batch, called unchanged; this module only substitutes clouds_mstcnv_batch.mstcnv_batch for the duration of
the call.  Results are expected bitwise equal (elementwise IEEE operations do not depend on the batch composition); this is checked in
tests/test_speed_d175.py and must be re-checked if the backend (libimf/numpy) changes.

Cores: nproc worker processes (spawn); the caller waits.  nproc<=1 -> existing code path.
Usage: with clouds_condse_par.par_condse(ctx_backend='imf', nproc=4): ... ; or condse_step_batch_par(inp, cfg, ms, nproc, backend).
"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_POOL = {}


def _worker(conn, backend):
    import clouds_condse_ff as cf
    import clouds_mstcnv_batch as mb
    cf.set_backend(backend)
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            return
        R, tune = msg
        conn.send(mb.mstcnv_batch(R, tune))


def _get_pool(nproc, backend):
    key = (nproc, backend)
    if key not in _POOL:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        ws = []
        for _ in range(nproc):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, backend), daemon=True)
            p.start()
            ws.append((p, a))
        for p, a in ws:
            assert a.recv() == "ready"
        _POOL[key] = ws
    return _POOL[key]


def close_pools():
    for ws in _POOL.values():
        for p, a in ws:
            try:
                a.send(None)
            except Exception:
                pass
    _POOL.clear()


def _axis(a, N):
    for ax, s in enumerate(a.shape):
        if s == N:
            return ax
    raise ValueError(f"no column axis in shape {a.shape}")


def mstcnv_par(R, tune, nproc, backend):
    N = R["pl"].shape[1]
    ws = _get_pool(nproc, backend)
    sl = [np.arange(w, N, nproc) for w in range(nproc)]
    for w, (p, a) in enumerate(ws):
        Rw = {}
        for k, v in R.items():
            v = np.asarray(v)
            Rw[k] = np.take(v, sl[w], axis=_axis(v, N))
        a.send((Rw, tune))
    outs = [a.recv() for p, a in ws]
    o = {}
    for k, v0 in outs[0].items():
        v0 = np.asarray(v0)
        ax = _axis(v0, len(sl[0]) if v0.ndim else 0) if v0.ndim else None
        full = np.empty(v0.shape[:ax] + (N,) + v0.shape[ax + 1:], v0.dtype)
        for w in range(nproc):
            idx = [slice(None)] * full.ndim
            idx[ax] = sl[w]
            full[tuple(idx)] = outs[w][k]
        o[k] = full
    return o


def condse_step_batch_par(inp, cfg, ms=None, with_momentum=True, cnt=None, nproc=4, backend="imf"):
    import clouds_condse_batch as cb
    if nproc <= 1:
        return cb.condse_step_batch(inp, cfg, ms=ms, with_momentum=with_momentum, cnt=cnt)
    orig = cb.mb.mstcnv_batch
    cb.mb.mstcnv_batch = lambda R, c, mut=None: mstcnv_par(R, c, nproc, backend)
    try:
        return cb.condse_step_batch(inp, cfg, ms=ms, with_momentum=with_momentum, cnt=cnt)
    finally:
        cb.mb.mstcnv_batch = orig


@contextlib.contextmanager
def par_condse(nproc=4, backend="imf"):
    """Like atm_step_fast.fast_condse() but with the parallel MSTCNV."""
    import atm_step as A
    import atm_step_fast as F
    orig = A.stage_condse

    def stage(S, R, ctx, ms=None, cols=None, tm=None):
        if cols is not None:
            raise NotImplementedError
        inp = A.condse_inputs(S, R)
        X, cnt = condse_step_batch_par(inp, ctx.cfg, ms=ms if ms is not None else {}, nproc=nproc, backend='imf' if ctx.imf else 'numpy')
        for k in A.CONDSE_OUT:
            if k in X:
                S[k] = np.array(X[k], copy=True)
        S['_condse_counts'] = cnt
        S['_condse_X'] = X
        return S
    A.stage_condse = stage
    try:
        yield
    finally:
        A.stage_condse = orig
