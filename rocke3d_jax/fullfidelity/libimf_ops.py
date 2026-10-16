"""D183: jax-compatible elementwise exp / pow with a LABELLED host callback into the Intel libimf of the real build.

What this is.  The real ModelE build (ifort 19.1.3) calls the Intel libimf `pow` / `exp`; XLA's (and glibc's) differ from it in the last
bit in a small fraction of the arguments, which flips CLOUDS threshold tests in a few percent of columns (D126/D129/D134).  This module
provides `exp(x, mask=None)` and `pow(x, y, mask=None)` that work inside `jax.jit` / `lax.scan` / `lax.while_loop` and, in mode 'libimf',
evaluate the values on the HOST by `jax.pure_callback` into libimf (ctypes, via intel_libm_ff for pow; same shared object for exp).
THIS IS NOT DEVICE-RESIDENT: the operand arrays (float64, and the optional bool mask) are copied device -> host, evaluated element by
element with the scalar libimf function (one callback per array operation, not per element), and the float64 result is copied back.
The element-by-element loop is the same one the NumPy 'imf' chain uses (intel_libm_ff.pow_imf; clouds_lscond_size_ff.ex/pw), so the
values are bit-identical to it by construction.

Modes (module state, `set_mode`):
  'libimf' : host callback into libimf (requires the Intel runtime; raises RuntimeError if absent unless fallback=True was given).
  'libm'   : plain jnp.exp / jnp.power on the XLA backend (what the existing JAX code does); no callback.
  With set_mode('libimf', fallback=True) and no libimf on the host the callback evaluates numpy exp/power instead and `status()` reports
  effective == 'libm-numpy-fallback' (never 'libimf').
The mode is read when an op is TRACED; after the first trace `set_mode` to a different mode raises (jit caches would silently keep the old
mode).  Start a new process to change mode.

Counters (readable from outside, thread-safe): `counters()` -> {'exp': {calls, elements, bytes_in, bytes_out, seconds}, 'pow': {...},
'total': {...}}; `reset_counters()`.  Counted when the callback RUNS on the host (not when traced), i.e. executed calls.
bytes_in = bytes of all operands received by the callback (x, y, mask as passed), bytes_out = bytes of the returned array.
"""
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

import intel_libm_ff

jax.config.update("jax_enable_x64", True)

_lock = threading.Lock()
_state = {"mode": "libm", "requested": "libm", "fallback": False, "traced": False}
_ZERO = {"calls": 0, "elements": 0, "bytes_in": 0, "bytes_out": 0, "seconds": 0.0}
_C = {"exp": dict(_ZERO), "pow": dict(_ZERO)}
_exp_fn = None


def available():
    """True when the Intel libimf (exp and pow) can be loaded on this host."""
    return _load_exp() is not None and intel_libm_ff.available()


def _load_exp():
    global _exp_fn
    if _exp_fn is None:
        import ctypes
        import os
        if not intel_libm_ff.available():
            _exp_fn = False
        else:
            d = os.environ.get("INTEL_LIBIMF_DIR", intel_libm_ff._DEFAULT)
            lib = ctypes.CDLL(os.path.join(d, "libimf.so"))
            lib.exp.restype = ctypes.c_double
            lib.exp.argtypes = [ctypes.c_double]
            _exp_fn = lib.exp
    return _exp_fn or None


def set_mode(mode, fallback=False):
    """mode 'libimf' or 'libm'.  'libimf' without the Intel runtime raises RuntimeError unless fallback=True."""
    if mode not in ("libimf", "libm"):
        raise ValueError(mode)
    if mode == "libimf" and not available() and not fallback:
        raise RuntimeError("Intel libimf not available on this host; mode 'libimf' refused (use mode 'libm', or fallback=True to run the "
                           "numpy fallback, which is reported as 'libm-numpy-fallback' and is NOT libimf)")
    eff = mode if (mode == "libm" or available()) else "libm-numpy-fallback"
    if _state["traced"] and eff != _state["mode"]:
        raise RuntimeError("libimf_ops: mode cannot change after the first op was traced (jit caches); start a new process")
    _state.update(mode=eff, requested=mode, fallback=bool(fallback))
    return eff


def mode():
    """Effective mode: 'libimf', 'libm' or 'libm-numpy-fallback'."""
    return _state["mode"]


def status():
    return dict(requested=_state["requested"], effective=_state["mode"], libimf_available=available(), fallback_allowed=_state["fallback"],
                traced=_state["traced"], libimf_dir=getattr(intel_libm_ff, "_DEFAULT", None))


def reset_counters():
    with _lock:
        for k in _C:
            _C[k] = dict(_ZERO)


def counters():
    with _lock:
        out = {k: dict(v) for k, v in _C.items()}
    out["total"] = {f: sum(out[k][f] for k in ("exp", "pow")) for f in _ZERO}
    return out


def _count(name, n, bin_, bout, dt):
    with _lock:
        c = _C[name]
        c["calls"] += 1
        c["elements"] += int(n)
        c["bytes_in"] += int(bin_)
        c["bytes_out"] += int(bout)
        c["seconds"] += dt


# ------------------------------------------------------------------------------------------------ host side
def host_exp(x):
    """Elementwise exp, libimf scalar exp (numpy fallback in 'libm-numpy-fallback' mode).  No counting."""
    x = np.asarray(x, dtype=np.float64)
    if _state["mode"] == "libm-numpy-fallback":
        with np.errstate(all="ignore"):
            return np.exp(x)
    f = _load_exp()
    return np.array([f(v) for v in x.ravel().tolist()]).reshape(x.shape)


def host_pow(x, y):
    """Elementwise x**y, libimf scalar pow (numpy fallback in 'libm-numpy-fallback' mode).  No counting."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if _state["mode"] == "libm-numpy-fallback":
        with np.errstate(all="ignore"):
            return np.power(x, y)
    f = intel_libm_ff._load()
    return np.array([f(u, v) for u, v in zip(x.ravel().tolist(), y.ravel().tolist())]).reshape(x.shape)


def _cb_exp(x, m):
    t0 = time.perf_counter()
    x = np.asarray(x)
    m = np.asarray(m)
    r = np.zeros(x.shape)
    idx = np.flatnonzero(m.ravel())
    if idx.size:
        r.ravel()[idx] = host_exp(x.ravel()[idx])
    _count("exp", idx.size, x.nbytes + m.nbytes, r.nbytes, time.perf_counter() - t0)
    return r


def _cb_pow(x, y, m):
    t0 = time.perf_counter()
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.asarray(m)
    r = np.zeros(x.shape)
    idx = np.flatnonzero(m.ravel())
    if idx.size:
        r.ravel()[idx] = host_pow(x.ravel()[idx], np.broadcast_to(y, x.shape).ravel()[idx])
    _count("pow", idx.size, x.nbytes + y.nbytes + m.nbytes, r.nbytes, time.perf_counter() - t0)
    return r


# ------------------------------------------------------------------------------------------------ jax side
def exp(x, mask=None):
    """exp(x).  mask (bool, same shape as x) given: evaluated only where mask, 0.0 elsewhere (the masked-lane convention of the LSCOND ops);
    mask None: all elements."""
    _state["traced"] = True
    x = jnp.asarray(x, jnp.float64)
    if _state["mode"] == "libm":
        r = jnp.exp(x)
        return r if mask is None else jnp.where(mask, r, 0.0)
    m = jnp.ones(x.shape, bool) if mask is None else jnp.broadcast_to(jnp.asarray(mask, bool), x.shape)
    return jax.pure_callback(_cb_exp, jax.ShapeDtypeStruct(x.shape, jnp.float64), x, m, vmap_method="broadcast_all")


def pow(x, y, mask=None):  # noqa: A001
    """x ** y (y scalar or array broadcastable to x).  mask as in exp."""
    _state["traced"] = True
    x = jnp.asarray(x, jnp.float64)
    y = jnp.asarray(y, jnp.float64)
    shp = jnp.broadcast_shapes(x.shape, y.shape)
    if _state["mode"] == "libm":
        r = jnp.power(x, y)
        return r if mask is None else jnp.where(mask, r, 0.0)
    x = jnp.broadcast_to(x, shp)
    y = jnp.broadcast_to(y, shp)
    m = jnp.ones(shp, bool) if mask is None else jnp.broadcast_to(jnp.asarray(mask, bool), shp)
    return jax.pure_callback(_cb_pow, jax.ShapeDtypeStruct(shp, jnp.float64), x, y, m, vmap_method="broadcast_all")


class JnpProxy:
    """Stand-in for a module-level `jnp` of an existing module: everything is forwarded to jax.numpy except `power` and `exp`, which go
    through this module (mode-dependent).  Only meaningful for modules whose every jnp.power / jnp.exp is a call the real build routes through
    libimf (the inventory in the D183 ledger lists them); `**` operators on tracers are NOT intercepted."""

    def __getattr__(self, name):
        return getattr(jnp, name)

    @staticmethod
    def power(x, y):
        return pow(x, y)

    @staticmethod
    def exp(x):
        return exp(x)
