"""Speed track: lax.while_loop whose body runs K masked iterations per loop test (`ROCKE_WHILE_UNROLL=K`, default 1 = plain loop).

On the GPU each while iteration ends with a device->host read of the loop predicate, so a loop of many tiny iterations is latency-bound.  Each of
the K inner iterations is applied only where the loop condition still holds (a select over the whole carry), so the final carry is exactly that of
the plain loop; only the number of host-visible loop tests drops (to ceil(n/K)).  Costs up to K-1 wasted body executions at the end.
"""
import os

import jax
import jax.numpy as jnp
from jax import lax

K = int(os.environ.get('ROCKE_WHILE_UNROLL', '1'))


def while_loop(cond, body, init, k=None):
    k = K if k is None else k
    if k <= 1:
        return lax.while_loop(cond, body, init)

    def multi(c):
        for _ in range(k):
            go = cond(c)
            c2 = body(c)
            c = jax.tree_util.tree_map(lambda a, b: jnp.where(go, b, a), c, c2)
        return c
    return lax.while_loop(cond, multi, init)
