"""Speed track: `ROCKE_SCAN_UNROLL=N` (default 0 = off) fully unrolls every lax.scan / fori_loop with a static trip count <= N.

On the GPU a scan is a while loop whose predicate is read back by the host every iteration; the model has many 40- to 72-step scans of a few
kernels each (layer sweeps, polar sums).  Unrolled, the same operations in the same order are straight-line code of one program and are
fused / launched without a host round trip per step.  The values are those of the loop (same operations, same order); XLA may fuse them
differently, so rounding-level differences against the loop version are possible and are judged by the band score.  Scans with a longer or
data-dependent trip count are left alone.  `install()` replaces jax.lax.scan (and fori_loop with static bounds) once.
"""
import os

N = int(os.environ.get('ROCKE_SCAN_UNROLL', '0'))
_installed = False


def install():
    global _installed
    if _installed or N <= 0:
        return
    import jax
    from jax import lax
    _scan = lax.scan
    _fori = lax.fori_loop

    def scan(f, init, xs=None, length=None, reverse=False, unroll=1, **kw):
        n = length
        if n is None and xs is not None:
            leaves = jax.tree_util.tree_leaves(xs)
            if leaves:
                n = leaves[0].shape[0]
        if n is not None and unroll == 1 and 0 < n <= N:
            unroll = n
        return _scan(f, init, xs, length=length, reverse=reverse, unroll=unroll, **kw)

    def fori_loop(lower, upper, body_fun, init_val, *, unroll=None):
        try:
            lo, hi = int(lower), int(upper)
            static = isinstance(lower, (int,)) or hasattr(lower, '__index__')
            static = static and (isinstance(upper, (int,)) or hasattr(upper, '__index__'))
        except Exception:
            static = False
        if static and unroll is None and 0 < hi - lo <= N:
            unroll = hi - lo
        return _fori(lower, upper, body_fun, init_val, unroll=unroll)

    lax.scan = scan
    lax.fori_loop = fori_loop
    jax.lax.scan = scan
    jax.lax.fori_loop = fori_loop
    _installed = True
