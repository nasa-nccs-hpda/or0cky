"""Batched seaice_core_jax.prec_si vs real-Fortran PRECIP_SI dumps (D26). Mirrors seaice_jax_compare.py's
batched_ground_si style: load records, run the whole file as one batched call."""
import sys, os, glob
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_jax as J
import precsi_compare as C


def load_all(files):
    return np.concatenate([C.load(f) for f in files], axis=0)


def batched_prec_si(rec):
    a = lambda c: jnp.asarray(rec[:, c])
    return J.prec_si(a(3), a(4), jnp.asarray(rec[:, 5:9]), jnp.asarray(rec[:, 9:13]), a(13), a(14))
