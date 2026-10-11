"""D220: capture of the SURFACE template before and after the device-side rewrite (jax_tpl_state.apply) at one step of the assembled coupled step, to see which template
columns the step writes from OUR state and which stay as read from the real records.   python tpl_audit_capture.py OUTDIR [--steps 3] [--capture-k 2]
CPU or GPU, replay radiation, libm mode (gpu_step_run wiring).  Writes OUTDIR/tpl_before.npz and tpl_after.npz (flattened leaves, paths as keys)."""
import os
import sys

_A = list(sys.argv)
OUT = _A[1]
os.makedirs(OUT, exist_ok=True)
NST = int(_A[_A.index('--steps') + 1]) if '--steps' in _A else 3
CK = int(_A[_A.index('--capture-k') + 1]) if '--capture-k' in _A else 2
sys.argv = [_A[0], 'nov26', os.path.join(OUT, 'run'), '--steps', str(NST), '--save', 'none']
import gpu_step_run as G   # noqa: E402  (wiring + runner)
import numpy as np   # noqa: E402
import jax   # noqa: E402
import jax_tpl_state as TS   # noqa: E402

_orig = TS.make_apply_state
CNT = {'n': 0}


def _flat(t):
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(t)[0]}


def _wrapped(host, K):
    f = _orig(host, K)

    def apply(tpl, *a, **kw):
        out = f(tpl, *a, **kw)
        if CNT['n'] == CK:
            np.savez(os.path.join(OUT, 'tpl_before.npz'), **_flat(tpl))
            np.savez(os.path.join(OUT, 'tpl_after.npz'), **_flat(out))
            print('captured template at call', CNT['n'], flush=True)
        CNT['n'] += 1
        return out
    return apply


TS.make_apply_state = _wrapped
G.main()
