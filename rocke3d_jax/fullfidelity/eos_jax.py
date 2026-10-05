"""Seawater specific volume VOLGSP (OCNFUNTAB.f:21-63) from the OFTABLE_NEW table, Stage 2, D72.

VOLGSP is a trilinear interpolation of VGSP(-2:40, 0:40, 0:39), read from the OFTAB record 1
(Fortran sequential, big-endian: 80-byte TITLE then the array in column-major order). The index
arithmetic (INT truncation, clamps) follows the Fortran exactly. Used by the straits port in place
of recorded EOS values.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

OFTAB = ('/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/'
         'ModelE_Support/prod_input_files/OFTABLE_NEW')
NI, NJ, NK = 43, 41, 40       # VGSP(-2:40, 0:40, 0:39)


def load_vgsp(path=OFTAB):
    raw = open(path, 'rb').read()
    n = int.from_bytes(raw[0:4], 'big', signed=True)
    assert n == 80 + 8 * NI * NJ * NK, n
    body = raw[4:4 + n]
    arr = np.frombuffer(body[80:], dtype='>f8').reshape((NI, NJ, NK), order='F')
    # index [ig+2, js, kp] for ig = -2..40
    return np.ascontiguousarray(arr.astype(np.float64))


def volgsp(vgsp, g, s, p):
    """Vectorized VOLGSP. vgsp: (43, 41, 40) array from load_vgsp. g, s, p: arrays of any shape."""
    gg = g * 2.5e-4
    ss = s * 1000.0
    pp = p * 5e-7
    ig = jnp.trunc(gg + 2.0).astype(jnp.int64) - 2
    ig = jnp.clip(ig, -2, 39)
    js = jnp.trunc(ss).astype(jnp.int64)
    js = jnp.where(js >= 40, 39, js)
    kp = jnp.trunc(pp).astype(jnp.int64)
    kp = jnp.where(kp < 0, 0, kp)
    kp = jnp.where(kp >= 39, 38, kp)
    V = jnp.asarray(vgsp)
    def v(i, j, k):
        return V[i + 2, j, k]
    ig1 = ig + 1
    js1 = js + 1
    kp1 = kp + 1
    t = ((kp - pp + 1) * ((js - ss + 1) * ((ig - gg + 1) * v(ig, js, kp) + (gg - ig) * v(ig1, js, kp))
                          + (ss - js) * ((ig - gg + 1) * v(ig, js1, kp) + (gg - ig) * v(ig1, js1, kp)))
         + (pp - kp) * ((js - ss + 1) * ((ig - gg + 1) * v(ig, js, kp1) + (gg - ig) * v(ig1, js, kp1))
                        + (ss - js) * ((ig - gg + 1) * v(ig, js1, kp1) + (gg - ig) * v(ig1, js1, kp1))))
    return t
