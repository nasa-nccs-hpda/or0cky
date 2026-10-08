"""D189: FUSED libimf host callback.  One `jax.pure_callback` evaluates a whole LIST of exp / pow requests (each with its own lane mask) with the
Intel libimf of the real build.  Same function, same arguments, same bits as D183's libimf_ops (one request per callback); what changes is how many callbacks
are made and how many lanes are evaluated.

  fx(reqs, tag=None) -> list of arrays
      reqs: list of ('e', x, mask) or ('p', x, y, mask); x, y float64 arrays (y broadcast to x), mask bool (None = all lanes).
      Result i has the shape of request i; lanes with mask False are 0.0 (the masked-lane convention of libimf_ops).
      The callback is skipped (lax.cond on the device, no host round trip) when no lane of any request is selected.

Host evaluation: lanes of all requests are compacted and evaluated in ONE loop per function.  The loop is a small C shim (gcc, built once into the
directory D189_SHIM_DIR or a temp dir) that dlopens the same libimf.so and calls the same scalar `exp` / `pow` per element -- no arithmetic of its own,
so the values are those of the ctypes call of D183 (checked bitwise in tests/test_libimf_fused.py).  Without gcc the ctypes per-element loop is used
(`status()['backend']` says which).

Modes (module state): 'libimf' (host callback; RuntimeError if the Intel runtime is absent) and 'libm' (jnp.exp / jnp.power on the XLA backend, no callback,
all lanes: the arithmetic of the existing libm JAX code).  The mode is read when a request is traced.

Counters (`counters()`): calls, lanes (selected elements), bytes_in (all operand arrays passed to the host), bytes_out, seconds (host-side), total and per tag.
"""
import ctypes
import functools
import os
import subprocess
import tempfile
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import intel_libm_ff

jax.config.update("jax_enable_x64", True)

_state = {"mode": "libm", "backend": None}
_lock = threading.Lock()
_ZERO = dict(calls=0, lanes=0, bytes_in=0, bytes_out=0, seconds=0.0)
_C = {"total": dict(_ZERO)}
_TAGS = {}
_shim = None
_fexp = None

_CSRC = r"""
#include <dlfcn.h>
typedef double (*f2)(double, double);
typedef double (*f1)(double);
static f2 P; static f1 E;
int shim_init(const char *dir) {
    char b[4096];
    snprintf(b, sizeof b, "%s/libintlc.so.5", dir);
    if (!dlopen(b, RTLD_NOW | RTLD_GLOBAL)) return 1;
    snprintf(b, sizeof b, "%s/libimf.so", dir);
    void *h = dlopen(b, RTLD_NOW | RTLD_GLOBAL);
    if (!h) return 2;
    P = (f2)dlsym(h, "pow"); E = (f1)dlsym(h, "exp");
    return (P && E) ? 0 : 3;
}
void shim_pow(const double *x, const double *y, double *r, long n) { for (long i = 0; i < n; i++) r[i] = P(x[i], y[i]); }
void shim_exp(const double *x, double *r, long n) { for (long i = 0; i < n; i++) r[i] = E(x[i]); }
"""


def _libdir():
    return os.environ.get("INTEL_LIBIMF_DIR", intel_libm_ff._DEFAULT)


def _load_shim():
    """Build (once) and load the C loop; False if gcc / libimf are unavailable (ctypes fallback)."""
    global _shim
    if _shim is not None:
        return _shim
    _shim = False
    if os.environ.get("D189_NO_SHIM") == "1" or not intel_libm_ff.available():
        return _shim
    try:
        d = os.environ.get("D189_SHIM_DIR") or tempfile.mkdtemp(prefix="d189shim")
        os.makedirs(d, exist_ok=True)
        src, so = os.path.join(d, "shim.c"), os.path.join(d, "libd189shim.so")
        if not os.path.exists(so):
            with open(src, "w") as f:
                f.write("#include <stdio.h>\n" + _CSRC)
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so + ".tmp", src, "-ldl"], check=True, capture_output=True)
            os.replace(so + ".tmp", so)
        lib = ctypes.CDLL(so)
        lib.shim_init.argtypes = [ctypes.c_char_p]
        lib.shim_init.restype = ctypes.c_int
        if lib.shim_init(_libdir().encode()) != 0:
            return _shim
        dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        lib.shim_pow.argtypes = [dp, dp, dp, ctypes.c_long]
        lib.shim_exp.argtypes = [dp, dp, ctypes.c_long]
        _shim = lib
    except Exception:  # noqa: BLE001
        _shim = False
    return _shim


def _ctypes_exp():
    global _fexp
    if _fexp is None:
        intel_libm_ff._load()
        lib = ctypes.CDLL(os.path.join(_libdir(), "libimf.so"))
        lib.exp.restype = ctypes.c_double
        lib.exp.argtypes = [ctypes.c_double]
        _fexp = lib.exp
    return _fexp


def available():
    return intel_libm_ff.available()


def set_mode(mode):
    if mode not in ("libimf", "libm"):
        raise ValueError(mode)
    if mode == "libimf" and not available():
        raise RuntimeError("Intel libimf not available on this host")
    _state["mode"] = mode
    if mode == "libimf":
        _state["backend"] = "c-shim" if _load_shim() else "ctypes-loop"
    return mode


def mode():
    return _state["mode"]


def status():
    return dict(mode=_state["mode"], backend=_state["backend"], libimf_available=available())


def reset_counters():
    with _lock:
        _C["total"] = dict(_ZERO)
        _TAGS.clear()


def counters():
    with _lock:
        return dict(total=dict(_C["total"]), tags={k: dict(v) for k, v in _TAGS.items()})


# ------------------------------------------------------------------------------------------------ host side
def host_exp(x):
    """Elementwise exp with libimf (no mask, no counting)."""
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    sh = _load_shim()
    if sh:
        r = np.empty_like(x)
        sh.shim_exp(x, r, x.size)
        return r
    f = _ctypes_exp()
    return np.array([f(v) for v in x.tolist()])


def host_pow(x, y):
    """Elementwise pow with libimf (no mask, no counting)."""
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    sh = _load_shim()
    if sh:
        r = np.empty_like(x)
        sh.shim_pow(x, y, r, x.size)
        return r
    f = intel_libm_ff._load()
    return np.array([f(u, v) for u, v in zip(x.tolist(), y.tolist())])


def _host_core(kinds, tag, items, bin_, t0):
    """items: list of (kind, x, y or None, mask) NumPy arrays.  Evaluates all selected lanes (one loop per function) and returns the list of results."""
    outs = [np.zeros(it[1].shape) for it in items]
    ex_x, ex_pos = [], []
    pw_x, pw_y, pw_pos = [], [], []
    for n, (kd, x, y, m) in enumerate(items):
        idx = np.flatnonzero(m.ravel())
        if not idx.size:
            continue
        if kd == "e":
            ex_x.append(x.ravel()[idx])
            ex_pos.append((n, idx))
        else:
            pw_x.append(x.ravel()[idx])
            pw_y.append(y.ravel()[idx])
            pw_pos.append((n, idx))
    lanes = 0
    if ex_x:
        r = host_exp(np.concatenate(ex_x))
        lanes += r.size
        o = 0
        for n, idx in ex_pos:
            outs[n].ravel()[idx] = r[o:o + idx.size]
            o += idx.size
    if pw_x:
        r = host_pow(np.concatenate(pw_x), np.concatenate(pw_y))
        lanes += r.size
        o = 0
        for n, idx in pw_pos:
            outs[n].ravel()[idx] = r[o:o + idx.size]
            o += idx.size
    bout = sum(o_.nbytes for o_ in outs)
    dt = time.perf_counter() - t0
    with _lock:
        for c in (_C["total"], _TAGS.setdefault(tag, dict(_ZERO))):
            c["calls"] += 1
            c["lanes"] += lanes
            c["bytes_in"] += bin_
            c["bytes_out"] += bout
            c["seconds"] += dt
    return outs


def _host(kinds, tag, *ops):
    """Un-stacked form: operands (x, m) per exp request and (x, y, m) per pow request."""
    t0 = time.perf_counter()
    ops = [np.asarray(o_) for o_ in ops]          # the callback may receive jax arrays: everything below is NumPy on the host
    i = 0
    items = []
    for kd in kinds:
        if kd == "e":
            items.append(("e", ops[i], None, ops[i + 1]))
            i += 2
        else:
            items.append(("p", ops[i], ops[i + 1], ops[i + 2]))
            i += 3
    return tuple(_host_core(kinds, tag, items, sum(o_.nbytes for o_ in ops), t0))


def _host_stacked(kinds, tag, X, M, *Y):
    """Stacked form (all requests of one shape): X (R, ...) values, M (R, ...) masks, Y (P, ...) the y of the pow requests in order."""
    t0 = time.perf_counter()
    X, M = np.asarray(X), np.asarray(M)
    Y = [np.asarray(y_) for y_ in Y]
    items, j = [], 0
    for r_, kd in enumerate(kinds):
        if kd == "e":
            items.append(("e", X[r_], None, M[r_]))
        else:
            items.append(("p", X[r_], Y[0][j], M[r_]))
            j += 1
    bin_ = X.nbytes + M.nbytes + sum(y_.nbytes for y_ in Y)
    return np.stack(_host_core(kinds, tag, items, bin_, t0))


# ------------------------------------------------------------------------------------------------ device side
def fx(reqs, tag=None):
    """List of exp / pow requests -> list of results (see module doc)."""
    kinds, ops, shapes, masks = [], [], [], []
    libimf = _state["mode"] == "libimf"
    res_libm = []
    for r in reqs:
        if r[0] == "e":
            _, x, m = r
            x = jnp.asarray(x, jnp.float64)
            shp = x.shape
            if not libimf:
                res_libm.append(jnp.exp(x))
                continue
            m = jnp.ones(shp, bool) if m is None else jnp.broadcast_to(jnp.asarray(m, bool), shp)
            kinds.append("e")
            ops += [x, m]
        else:
            _, x, y, m = r
            x = jnp.asarray(x, jnp.float64)
            y = jnp.asarray(y, jnp.float64)
            shp = jnp.broadcast_shapes(x.shape, y.shape)
            if not libimf:
                res_libm.append(jnp.power(x, y))
                continue
            x, y = jnp.broadcast_to(x, shp), jnp.broadcast_to(y, shp)
            m = jnp.ones(shp, bool) if m is None else jnp.broadcast_to(jnp.asarray(m, bool), shp)
            kinds.append("p")
            ops += [x, y, m]
        shapes.append(shp)
        masks.append(m)
    if not libimf:
        return res_libm
    anyl = functools.reduce(jnp.logical_or, [jnp.any(m) for m in masks])
    if all(sh == shapes[0] for sh in shapes) and os.environ.get("D189_NO_STACK") != "1":
        shp = shapes[0]
        X = jnp.stack(_xs(ops, kinds))
        M = jnp.stack(masks)
        Ys = _ys(ops, kinds)
        sops = (X, M) + ((jnp.stack(Ys),) if Ys else ())
        spec = jax.ShapeDtypeStruct((len(kinds),) + shp, jnp.float64)
        cb = functools.partial(_host_stacked, tuple(kinds), tag)

        def run(o):
            return jax.pure_callback(cb, spec, *o)

        def skip(o):
            return jnp.zeros(spec.shape, jnp.float64)

        out = lax.cond(anyl, run, skip, sops)
        return [out[i] for i in range(len(kinds))]
    spec = tuple(jax.ShapeDtypeStruct(s, jnp.float64) for s in shapes)
    cb = functools.partial(_host, tuple(kinds), tag)

    def run(o):
        return jax.pure_callback(cb, spec, *o)

    def skip(o):
        return tuple(jnp.zeros(s.shape, jnp.float64) for s in spec)

    return list(lax.cond(anyl, run, skip, tuple(ops)))


def _xs(ops, kinds):
    out, i = [], 0
    for kd in kinds:
        out.append(ops[i])
        i += 2 if kd == "e" else 3
    return out


def _ys(ops, kinds):
    out, i = [], 0
    for kd in kinds:
        if kd == "p":
            out.append(ops[i + 1])
        i += 2 if kd == "e" else 3
    return out
