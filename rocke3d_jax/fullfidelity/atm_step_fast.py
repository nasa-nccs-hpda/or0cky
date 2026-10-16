"""D133: the chained atmosphere step of atm_step.py with the CONDSE stage replaced by the batched CONDSE (clouds_condse_batch, D132).

Nothing in atm_step.py is changed: this module provides `stage_condse_fast` (same inputs/outputs as atm_step.stage_condse, but calling
clouds_condse_batch.condse_step_batch) and `run_step` / `run_chain`, which run atm_step.run_step / the atm_step.run_free chain with
`atm_step.stage_condse` temporarily bound to the fast stage (atm_step.run_step looks the stage function up at call time), so every other
stage, the recorded-input policy (scoping/ATM_STEP_PLAN.md) and the state hand-over are exactly those of atm_step.py.

The batched CONDSE keeps the two pole columns on the per-column code (south pole, batch, north pole: the LSCOND module-array carry `ms`
passes through them in Fortran order).  `cols` (a column subset) is not supported by the batch; use atm_step.stage_condse for that.

Usage:  import atm_step_fast as F;  S, snaps = F.run_step(date, itime, ctx, ms={})       # one step from the real start state
        out = F.run_chain(date, it0, 6, ctx, land_mode='recorded', on_step=callback)        # multi-step chain from OUR own end state
IMPORTANT: clouds_condse_ff's libm/libimf backend is a process-global switch (atm_step.make_ctx sets it); `ensure_backend(ctx)` re-applies
the mode of a ctx before running, so contexts of both modes can be alternated safely.
"""
import contextlib
import time

import numpy as np

import atm_step as A
import clouds_condse_batch as cb
import clouds_condse_ff as cf

CARRY_KEYS = A.CARRY_KEYS


def ensure_backend(ctx):
    cf.set_backend('imf' if ctx.imf else 'numpy')


def stage_condse_fast(S, R, ctx, ms=None, cols=None, tm=None):
    if cols is not None:
        raise NotImplementedError("the batched CONDSE has no column-subset mode; use atm_step.stage_condse")
    inp = A.condse_inputs(S, R)
    X, cnt = cb.condse_step_batch(inp, ctx.cfg, ms=ms if ms is not None else {})
    for k in A.CONDSE_OUT:
        if k in X:
            S[k] = np.array(X[k], copy=True)
    S['_condse_counts'] = cnt
    S['_condse_X'] = X
    return S


@contextlib.contextmanager
def fast_condse():
    orig = A.stage_condse
    A.stage_condse = stage_condse_fast
    try:
        yield
    finally:
        A.stage_condse = orig


def run_step(date, itime, ctx, **kw):
    """atm_step.run_step with the batched CONDSE (same arguments and return value)."""
    ensure_backend(ctx)
    with fast_condse():
        return A.run_step(date, itime, ctx, **kw)


def run_chain(date, it0, nsteps, ctx, land_mode='recorded', ff=A.FF, on_step=None, ms=None):
    """Chain nsteps steps; step k starts from OUR end state of step k-1 (atmosphere + ATURB/PBL hidden state + CONDSE cloud/precip carry +
    LSCOND module arrays), exactly as atm_step.run_free, but returns every stage snapshot.  Recorded (non-atmosphere) inputs of step k:
    radiation SRHR/TRHR/COSZ1, SURFACE tile/PBL/land-ice/GHY/Ent records, sea-ice/lake/ocean, the non-atmosphere CONDSE entry fields.
    on_step(k, itime, R, snaps, S, timing) is called after each step (S still holds the '_' keys).  Returns list of dict(itime, timing, wall)."""
    ensure_backend(ctx)
    S, out = None, []
    ms = ms if ms is not None else {}
    for k in range(nsteps):
        R = A.Real(date, it0 + k, ff)
        tm = {}
        t0 = time.perf_counter()
        with fast_condse():
            S, sn = A.run_step(date, it0 + k, ctx, R=R, S=S, ms=ms, land_mode=land_mode, timing=tm)
        wall = time.perf_counter() - t0
        if on_step is not None:
            on_step(k, it0 + k, R, sn, S, tm)
        X = S.get('_condse_X')
        carry = {key: np.array(X[key], copy=True) for key in CARRY_KEYS if X is not None and key in X}
        if '_cloud_rad' in S:
            carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
        out.append(dict(itime=it0 + k, timing={a: b for a, b in tm.items() if a.startswith('stage_')}, wall=wall))
        S = {key: v for key, v in S.items() if not key.startswith('_')}
        if carry:
            S['_carry'] = carry
    return out
