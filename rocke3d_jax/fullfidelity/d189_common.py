"""D189 shared helpers (own unit): scratch path, the step-0 CONDSE inputs of a date (as in D183 d183_stage_condse), exit-field comparison,
and a recorder that captures the (R, tune) arguments of every clouds_mstcnv_jax.mstcnv_jax call so MSTCNV can be run stand-alone on real inputs."""
import os
import pickle
import copy

SCR = ("/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-"
       "projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d189")
DATES = {'nov26': 33312, 'dec01': 33552, 'jan01': 17522}


def step0_inputs(date, it, imf=True):
    """NumPy-dynamics step of the real start state; returns (ctx, CONDSE input dict)."""
    import atm_step as A
    ctx = A.make_ctx(date, imf=imf)
    R = A.Real(date, it, A.FF)
    S = A.init_state(R)
    A._native(S)
    A.stage_dyn(S, R, ctx)
    return ctx, A.condse_inputs(S, R)


def compare_dicts(a, b):
    """-> (n_compared, n_equal, [(key, n_unequal, maxabs)]) for ndarray entries present in both with equal shape (NaN == NaN)."""
    import numpy as np
    bad, nf = [], 0
    for k in sorted(a):
        if k not in b or not isinstance(a[k], np.ndarray):
            continue
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if x.shape != y.shape:
            continue
        nf += 1
        ne = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        if ne.any():
            bad.append((k, int(ne.sum()), float(np.abs(np.where(ne, x - y, 0)).max())))
    return nf, nf - len(bad), bad


class Recorder:
    """Wrap clouds_mstcnv_jax.mstcnv_jax: keep deep copies of (R, tune) and the output dict of each call."""

    def __init__(self, mj):
        self.mj, self.orig, self.calls = mj, mj.mstcnv_jax, []

    def __enter__(self):
        def wrapped(R, c, *a, **k):
            self.calls.append((copy.deepcopy(R), copy.deepcopy(c)))
            return self.orig(R, c, *a, **k)
        self.mj.mstcnv_jax = wrapped
        return self

    def __exit__(self, *e):
        self.mj.mstcnv_jax = self.orig


def dump(obj, name):
    with open(os.path.join(SCR, name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(name):
    with open(os.path.join(SCR, name), 'rb') as f:
        return pickle.load(f)
