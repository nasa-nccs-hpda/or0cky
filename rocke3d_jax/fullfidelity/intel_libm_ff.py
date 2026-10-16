"""Optional access to the Intel libimf `pow` that the real ModelE build (ifort 19.1.3, -O2) calls for
`x**KAPA` (D96, found while validating MAtoP's PK = PMID**KAPA).

numpy's `**` and a correctly rounded pow both differ from ifort's libimf pow by 1 ulp in about 0.04 % of
the PMID values (47 of 132,480 in the first checked call), because libimf pow is not correctly rounded.
Calling the same libimf.so scalar `pow` through ctypes reproduces every dumped PK bit-for-bit (0 of
132,480 mismatches).  This only works on hosts that have the Intel runtime (path below, override with the
environment variable INTEL_LIBIMF_DIR); everywhere else `available()` is False and callers fall back to
numpy (1-ulp tolerance).
"""
import ctypes
import os

import numpy as np

_DEFAULT = ("/panfs/ccds02/app/modules/intel/platform/x86_64/rhel/8.6/2020Update4/"
            "compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin")
_pow = None
_tried = False


def _load():
    global _pow, _tried
    if _tried:
        return _pow
    _tried = True
    d = os.environ.get("INTEL_LIBIMF_DIR", _DEFAULT)
    try:
        ctypes.CDLL(os.path.join(d, "libintlc.so.5"), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(os.path.join(d, "libimf.so"))
        lib.pow.restype = ctypes.c_double
        lib.pow.argtypes = [ctypes.c_double, ctypes.c_double]
        _pow = lib.pow
    except OSError:
        _pow = None
    return _pow


def available():
    return _load() is not None


def pow_imf(x, y):
    """Elementwise x**y with the Intel libimf scalar pow (x array, y scalar)."""
    f = _load()
    if f is None:
        raise RuntimeError("Intel libimf not available")
    x = np.asarray(x, dtype=float)
    y = float(y)
    return np.array([f(v, y) for v in x.ravel()]).reshape(x.shape)
