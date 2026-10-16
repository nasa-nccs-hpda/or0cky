"""Batched landice_precip_jax.precip_li vs real-Fortran PRECIP_LI dumps (D28)."""
import sys, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landice_precip_jax as J
import precli_compare as C


def load_all(files):
    return np.concatenate([C.load(f) for f in files], axis=0)


def batched_precip_li(rec):
    a = lambda c: jnp.asarray(rec[:, c])
    return J.precip_li(a(3), a(4), a(5), a(6), a(7), a(8))
