"""D221: which columns of the recorded SURFACE template does the coupled step actually consume?  Step 0 runs normally (replay radiation, as d193_day);
then step 1 is run repeatedly from the same state with one template leaf (or one column of a leaf) set to NaN BEFORE the device applies our state.
A leaf/column whose NaN reaches the step output (new state or filter arrays) is consumed and not overwritten; otherwise it is overwritten or unused
(at this step, with this data).  Usage: python tpl_consumption_probe.py OUT.json [--max-cols N]   (V100 through gpu/v100_run.sh)"""
import argparse
import faulthandler
import json
import sys
import time

import numpy as np

import os
_A = list(sys.argv)                     # same GPU preamble as gpu_day_run.py: backend and libm wiring BEFORE anything else imports jax
sys.argv = [_A[0], 'nov26', os.environ.get('GPU_IMPORT_SCRATCH', '/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/_import_scratch')]
import gpu_step_run as G  # noqa: E402,F401
sys.argv = _A
import atm_step as A  # noqa: E402
import jax_harness as H  # noqa: E402
_mk = A.make_ctx
A.make_ctx = lambda date, imf=False, *a, **k: _mk(date, False, *a, **k)
import d193_day as D  # noqa: E402
import jax
import jax.numpy as jnp
import jax_coupled as C
import jax_harness as H
import d191_run as R191


def nan_leaves(tree):
    out = []
    for pth, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if hasattr(x, 'dtype') and np.issubdtype(np.asarray(x).dtype, np.floating):
            if bool(jnp.any(jnp.isnan(x))):
                out.append(jax.tree_util.keystr(pth))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--max-cols', type=int, default=100000)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(1200, repeat=True, file=sys.stdout)
    A.Real = D.DayReal
    t00 = time.perf_counter()
    cp = C.Coupled(D.DATE)
    print(f'setup {cp.setup_seconds:.0f} s', flush=True)
    sr = R191.build_registry('replay')
    it0 = dict(A.DATES)[D.DATE]
    state, rec, dev = cp.initial_state()
    state, info, _ = cp.step(0, state, rec, dev, sr, timed=False, keep=False)
    jax.block_until_ready(state)
    print(f'step 0 done at {time.perf_counter() - t00:.0f} s', flush=True)
    rec1 = cp.ph.load_records(it0 + 1, first=False)
    state0 = state

    def copy_state():
        return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True) if hasattr(x, 'dtype') else x, state0)

    orig = cp._build_tpl
    cfg = dict(leaf=None, col=None)

    def bt(R):
        tpl, host, recs, irrig = orig(R)
        if cfg['leaf'] is None:
            return tpl, host, recs, irrig
        flat, tdef = jax.tree_util.tree_flatten_with_path(tpl)
        new = []
        for pth, x in flat:
            if jax.tree_util.keystr(pth) == cfg['leaf']:
                x = jnp.asarray(x)
                if cfg['col'] is None:
                    x = jnp.full_like(x, jnp.nan)
                else:
                    x = x.at[..., cfg['col']].set(jnp.nan)
            new.append(x)
        return jax.tree_util.tree_unflatten(tdef, new), host, recs, irrig
    cp._build_tpl = bt

    def run():
        dev1 = cp.ph.to_device(rec1, first=False)
        cp._tpl_prefetch.clear()
        s, inf, arr = cp.step(1, copy_state(), rec1, dev1, sr, timed=False, keep=True)
        jax.block_until_ready(s)
        return nan_leaves((s, arr['filter']))

    base = run()
    print('baseline NaN leaves:', base, flush=True)
    tpl, _, _, _ = orig(rec1['R'])
    leaves = [(jax.tree_util.keystr(p), np.asarray(x)) for p, x in jax.tree_util.tree_flatten_with_path(tpl)[0]]
    res = dict(baseline_nan=base, leaves={})
    ncols = 0
    t0 = time.perf_counter()
    for name, x in leaves:
        if x.dtype.kind != 'f' or x.size == 0:
            continue
        cfg.update(leaf=name, col=None)
        hit = run()
        ent = dict(shape=list(x.shape), whole_leaf_nan=sorted(set(hit) - set(base))[:6], n_hit=len(set(hit) - set(base)), consumed_cols=None)
        if ent['n_hit'] and x.ndim == 2 and x.shape[1] > 1:
            cons = []
            for c in range(x.shape[1]):
                if ncols >= a.max_cols:
                    break
                cfg.update(col=c)
                ncols += 1
                if set(run()) - set(base):
                    cons.append(c)
            ent['consumed_cols'] = cons
        res['leaves'][name] = ent
        print(f"{name} {x.shape}: whole-leaf hit {ent['n_hit']}  consumed cols {ent['consumed_cols'] if ent['consumed_cols'] is None else len(ent['consumed_cols'])}  [{time.perf_counter() - t0:.0f} s]", flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
