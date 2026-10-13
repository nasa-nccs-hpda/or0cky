"""Speed track: strictly sequential (left-to-right) sums with far fewer device kernels.  `ROCKE_SEQSUM_BLK=N` (default 0 = off).

The model's Fortran-order sums are lax.scan carries `c + x`; on the GPU every scan iteration is a few kernel launches, so a sum over
the 3,404 cells of the sea-ice grid costs thousands of launches (post_a trace: 23,000 add kernels, 0.13 s).  With N > 1 the scan runs
over blocks of N elements whose N additions are written out in order inside the body (XLA fuses them into one kernel); the tail
(< N elements) is added in order after the loop.  Same additions in the same order, so the result is bitwise the same as the
plain scan.  Only the number of kernels changes.
"""
import os

import jax.numpy as jnp
from jax import lax

BLK = int(os.environ.get('ROCKE_SEQSUM_BLK', '0'))


def _fast(a, axis=0, blk=BLK):
    a = jnp.moveaxis(a, axis, 0)
    n = a.shape[0]
    s = a[0]
    m = (n - 1) // blk
    if m:
        rest = a[1:1 + m * blk].reshape((m, blk) + a.shape[1:])

        def body(c, b):
            for t in range(blk):
                c = c + b[t]
            return c, None
        s, _ = lax.scan(body, s, rest)
    for t in range(1 + m * blk, n):
        s = s + a[t]
    return s


def wrap(fn):
    """fn(a, axis=0) or fn(a): the fast version when ROCKE_SEQSUM_BLK > 1, else fn itself."""
    if BLK <= 1:
        return fn

    def f(a, axis=0):
        return _fast(a, axis)
    f.__name__ = fn.__name__
    return f
