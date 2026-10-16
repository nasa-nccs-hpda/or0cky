"""Batched lakes_core_jax.precip_lk vs real-Fortran PRECIP_LK dumps (D27)."""
import sys, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lakes_core_jax as J
import precip_lk_compare as C


def load_all(files):
    return np.concatenate([C.load(f) for f in files], axis=0)


def batched_precip_lk(rec):
    a = lambda c: jnp.asarray(rec[:, c])
    z = jnp.zeros(len(rec))
    return J.precip_lk(a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10), a(11), a(12), a(13), a(14),
                       a(15), z, z, z)
