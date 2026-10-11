"""D183: Intel libimf semantics for the JAX stages through a LABELLED HOST CALLBACK (ACCEPTANCE_CRITERIA section 8 item 1).

What this module is: drop-in variants of the call sites at which the JAX dynamics and the JAX CONDSE kernels evaluate `pow` / `exp`, routed
through `jax.pure_callback` to the Intel libimf scalar functions (the same ctypes bridge as the libimf-mode NumPy chain,
intel_libm_ff / clouds_lscond_size_ff / clouds_mstcnv_ff).  Nothing existing is edited: every variant is installed at run time by a
context manager that rebinds module attributes and restores them on exit.

HYBRID ITEM (must be reported with any result that uses it): these evaluations are NOT device resident.  Each callback moves its operands
device -> host and the result host -> device, evaluates the scalar libimf function element by element on one host core (GIL held), and is
a synchronisation point.  `METER` counts calls, elements, bytes in/out and host seconds per callback site.

Call sites covered (see scoping/D183_LIBIMF_CALLBACK_ENTRY.md for the audit that selected them):
  dyn_jax_aflux.matop_jax        PK = PMID**KAPA             (inside ADVECM, lax.scan over the layers)
  dyn_jax_pgf.make_pgf           PKU/PKD = P**KAPA           (lax.scan over the layers) and HUNDREDTHeKAPA = .01**KAPA (host constant)
  jax_atm_step._pek_jax          PEK = PEDN**KAPA
  clouds_mstcnv_jax              qsat exp (dq_j), precip_mp exp/pow(flam,4), pfac pow(1000/pl,.4), dcg/dci pow, flam pow(.25), anvil pow(1/3)
                                 (jnp.exp / jnp.power by proxy; the four `**` of conv_micro_j by a textual rewrite of its source)
  clouds_lscond_jax              already routes exp/pow/dq through callbacks in mode="libm" to clouds_lscond_size_ff.ex/pw, which are libimf
                                 once clouds_condse_ff.set_backend('imf') ran; this module only METERS those callbacks (patch_lscond_meter).
Not covered, deliberately: sqrt (IEEE correctly rounded, equal everywhere), the integer-power `fracm**3` of AADVT/QDYNAM and the
`mplume**2.0` forms (the libimf-mode NumPy chain evaluates those with numpy, not libimf, so JAX/XLA is the matching reference), NumPy-side
stages (TROP, MAtoPMB, filters, pole columns, snow-age exp) which already call libimf.
"""
import clouds_jax_env  # noqa: F401  (XLA flags before jax)
import contextlib
import functools
import inspect
import itertools
import re
import threading
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import intel_libm_ff as il


def available():
    return il.available()


# ------------------------------------------------------------------------------------------------ metering
class Meter:
    """Per-site counters of the host callbacks (calls, elements, bytes device->host and host->device, host seconds)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.sites = {}

    def add(self, site, elements, b_in, b_out, secs):
        with self.lock:
            s = self.sites.setdefault(site, dict(calls=0, elements=0, bytes_in=0, bytes_out=0, host_seconds=0.0))
            s['calls'] += 1
            s['elements'] += int(elements)
            s['bytes_in'] += int(b_in)
            s['bytes_out'] += int(b_out)
            s['host_seconds'] += secs

    def snapshot(self):
        with self.lock:
            return {k: dict(v) for k, v in self.sites.items()}

    def totals(self, prefix=None):
        t = dict(calls=0, elements=0, bytes_in=0, bytes_out=0, host_seconds=0.0)
        for k, v in self.snapshot().items():
            if prefix is None or k.startswith(prefix):
                for n in t:
                    t[n] += v[n]
        t['MiB_total'] = (t['bytes_in'] + t['bytes_out']) / 2 ** 20
        return t


METER = Meter()


def _nbytes(*a):
    return sum(np.asarray(x).nbytes for x in a)


# ------------------------------------------------------------------------------------------------ host functions
def _libimf():
    f = il._load()
    if f is None:
        raise RuntimeError("Intel libimf not available")
    import ctypes
    import os
    d = os.environ.get("INTEL_LIBIMF_DIR", il._DEFAULT)
    lib = ctypes.CDLL(os.path.join(d, "libimf.so"))
    lib.exp.restype = ctypes.c_double
    lib.exp.argtypes = [ctypes.c_double]
    lib.pow.restype = ctypes.c_double
    lib.pow.argtypes = [ctypes.c_double, ctypes.c_double]
    return lib


_LIB = {}


def _lib():
    if 'l' not in _LIB:
        _LIB['l'] = _libimf()
    return _LIB['l']


def host_pow(x, y, site='pow'):
    """Elementwise libimf pow(x, y) (y scalar or array broadcastable to x).  float64 in, float64 out."""
    t0 = time.perf_counter()
    f = _lib().pow
    x = np.asarray(x, dtype=np.float64)
    yb = np.asarray(y, dtype=np.float64)
    if yb.ndim == 0:
        out = np.fromiter(map(f, x.ravel().tolist(), itertools.repeat(float(yb))), np.float64, x.size)
    else:
        yb = np.broadcast_to(yb, x.shape)
        out = np.fromiter(map(f, x.ravel().tolist(), yb.ravel().tolist()), np.float64, x.size)
    out = out.reshape(x.shape)
    METER.add(site, x.size, _nbytes(x) + (0 if isinstance(y, float) else _nbytes(yb)), out.nbytes, time.perf_counter() - t0)
    return out


def host_exp(x, site='exp'):
    t0 = time.perf_counter()
    f = _lib().exp
    x = np.asarray(x, dtype=np.float64)
    out = np.fromiter(map(f, x.ravel().tolist()), np.float64, x.size).reshape(x.shape)
    METER.add(site, x.size, x.nbytes, out.nbytes, time.perf_counter() - t0)
    return out


def _site_name():
    """caller of pow_imf/exp_imf outside this module (python frame at trace time), for the meter."""
    f = inspect.currentframe().f_back.f_back
    while f is not None and (f.f_code.co_filename.endswith('jax_imf.py') or 'site-packages' in f.f_code.co_filename):
        f = f.f_back
    if f is None:
        return 'unknown'
    import os
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}"


def pow_imf(x, y, site=None):
    """jnp.power replacement: libimf pow through a host callback.  x: array/tracer; y: python float or array (broadcast)."""
    x = jnp.asarray(x, jnp.float64)
    site = site or 'pow@' + _site_name()
    if isinstance(y, (int, float, np.floating)):
        return jax.pure_callback(functools.partial(host_pow, y=float(y), site=site), jax.ShapeDtypeStruct(x.shape, jnp.float64), x)
    y = jnp.asarray(y, jnp.float64)
    if y.ndim == 0:                      # traced scalar exponent (e.g. KAPA passed as a jit argument): one scalar operand
        return jax.pure_callback(functools.partial(host_pow, site=site), jax.ShapeDtypeStruct(x.shape, jnp.float64), x, y)
    shp = jnp.broadcast_shapes(x.shape, y.shape)
    x = jnp.broadcast_to(x, shp)
    y = jnp.broadcast_to(y, shp)
    return jax.pure_callback(functools.partial(host_pow, site=site), jax.ShapeDtypeStruct(shp, jnp.float64), x, y)


def exp_imf(x):
    x = jnp.asarray(x, jnp.float64)
    return jax.pure_callback(functools.partial(host_exp, site='exp@' + _site_name()), jax.ShapeDtypeStruct(x.shape, jnp.float64), x)


class JnpImf:
    """Stand-in for the `jnp` name of one module: power and exp go through libimf callbacks, everything else is jax.numpy."""

    def __getattr__(self, name):
        return getattr(jnp, name)

    @staticmethod
    def power(x, y):
        return pow_imf(x, y)

    @staticmethod
    def exp(x):
        return exp_imf(x)


# ------------------------------------------------------------------------------------------------ dynamics
class ImfKapa(float):
    """float whose reflected power is libimf's: `.01 ** ImfKapa(kapa)` -> libimf pow(.01, kapa) (dyn_jax_pgf evaluates HUNDREDTHeKAPA =
    .01**g['kapa'] on the host with the Python float pow; the libimf-mode NumPy chain uses libimf there)."""

    def __rpow__(self, other):
        return float(_lib().pow(float(other), float(self)))


@contextlib.contextmanager
def patch_dyn():
    """Within the context: dyn_jax_aflux and dyn_jax_pgf evaluate PK/PKU/PKD through libimf callbacks, and HUNDREDTHeKAPA through libimf.
    Kits/jits must be CREATED AND FIRST CALLED inside the context (jit caches are per function object and trace once)."""
    import dyn_jax_aflux as jaf
    import dyn_jax_pgf as jpg
    saved = (jaf.jnp, jpg.jnp, jpg.make_pgf)
    jaf.jnp = JnpImf()
    jpg.jnp = JnpImf()
    orig_make = jpg.make_pgf

    def make_pgf(g, tab):
        g2 = dict(g)
        g2['kapa'] = ImfKapa(g['kapa'])
        return orig_make(g2, tab)
    jpg.make_pgf = make_pgf
    try:
        yield
    finally:
        jaf.jnp, jpg.jnp, jpg.make_pgf = saved


@jax.jit
def _pek_imf(pedn, kapa):
    return pow_imf(pedn, kapa, site='pow@PEK')


def pek(pedn, kapa):
    """PEK = PEDN**KAPA through the libimf callback (replacement of jax_atm_step._pek_jax)."""
    return _pek_imf(jnp.asarray(pedn, jnp.float64), float(kapa))


@contextlib.contextmanager
def patch_pek():
    import jax_atm_step as J
    saved = J._pek_jax
    J._pek_jax = lambda pedn, kapa: pek(pedn, kapa)
    try:
        yield
    finally:
        J._pek_jax = saved


# ------------------------------------------------------------------------------------------------ CONDSE
_DC_RE = re.compile(r"(\(\(\w+ / k\.\w+\) \* jnp\.power\(pl / 1000\.0, k\.F04\)\)) \*\* (k\.\w+)")


def _conv_micro_imf_source():
    import clouds_mstcnv_jax as mj
    src = inspect.getsource(mj.conv_micro_j)
    new, n = _DC_RE.subn(r"jnp.power(\1, \2)", src)
    if n != 4:
        raise RuntimeError(f"conv_micro_j source changed: expected 4 `**` rewrites, found {n}")
    return new


@contextlib.contextmanager
def patch_mstcnv():
    """Within the context: every jnp.exp / jnp.power of clouds_mstcnv_jax (and the four `**` of conv_micro_j, rewritten to jnp.power) goes
    through a libimf callback.  Must wrap the first call of mstcnv_jax in the process (module-level jits trace once)."""
    import clouds_mstcnv_jax as mj
    saved_jnp, saved_cm = mj.jnp, mj.conv_micro_j
    proxy = JnpImf()
    mj.jnp = proxy
    exec(compile(_conv_micro_imf_source(), 'jax_imf:conv_micro_j_imf', 'exec'), mj.__dict__)   # rebinds mj.conv_micro_j (same globals)
    try:
        yield
    finally:
        mj.jnp, mj.conv_micro_j = saved_jnp, saved_cm


@contextlib.contextmanager
def patch_lscond_meter():
    """Meter the existing LSCOND libm-mode callbacks (clouds_lscond_jax._host_exp/_host_pow/_host_dq).  They evaluate through
    clouds_lscond_size_ff.ex / pw, i.e. libimf when clouds_condse_ff.set_backend('imf') (or clouds_lscond_size_ff.use_imf(True)) is active."""
    import clouds_lscond_jax as lj
    saved = (lj._host_exp, lj._host_pow, lj._host_dq)

    def wrap(name, f):
        def g(*a):
            t0 = time.perf_counter()
            r = f(*a)
            rr = r if isinstance(r, tuple) else (r,)
            METER.add('lscond.' + name, np.asarray(a[0]).size, _nbytes(*a), _nbytes(*rr), time.perf_counter() - t0)
            return r
        return g
    lj._host_exp = wrap('exp', saved[0])
    lj._host_pow = wrap('pow', saved[1])

    def hdq(kind, *a):
        return wrap('dq_' + kind, lambda *b: saved[2](kind, *b))(*a)
    lj._host_dq = hdq
    try:
        yield
    finally:
        lj._host_exp, lj._host_pow, lj._host_dq = saved


@contextlib.contextmanager
def patch_adv_meter():
    """Meter the pre-existing ADV1D host callback of MSTCNV (not a libimf item; reported because it adds to the transfer volume)."""
    import clouds_mstcnv_jax as mj
    saved = mj._host_adv

    def g(*a, **k):
        t0 = time.perf_counter()
        r = saved(*a, **k)
        METER.add('adv1d(existing,not libimf)', np.asarray(a[3]).size, _nbytes(*a), _nbytes(*r), time.perf_counter() - t0)
        return r
    mj._host_adv = g
    try:
        yield
    finally:
        mj._host_adv = saved


@contextlib.contextmanager
def condse_imf(ls_meter=True):
    """Everything needed for the JAX CONDSE in libimf mode: the NumPy parts (set-up, post-processing, poles, snow age) and the LSCOND callbacks
    use libimf through clouds_condse_ff.set_backend('imf'); MSTCNV pow/exp by callbacks; metering of all callbacks."""
    import clouds_condse_ff as cf
    cf.set_backend('imf')
    with patch_mstcnv(), patch_adv_meter():
        if ls_meter:
            with patch_lscond_meter():
                yield
        else:
            yield
