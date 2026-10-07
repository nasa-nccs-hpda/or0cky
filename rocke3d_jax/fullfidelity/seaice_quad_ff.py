"""REAL*16 (IEEE binary128) emulation of SEAICE.f `Ti` and `Ti2b` (D173).

SEAICE.f:2363-2440 (SEAICE_FIXES_2022, always defined in this build; seaice_thermo = "BP") declares `real*16 b,c,det,tm` and uses
quad literals (`0q0`, `4q0`, `5q-1`, `1q-10`).  seaice_core_ff.Ti / Ti2b are float64.

Emulation: mpmath at 113 bits (binary128 significand), correctly rounded.  np.longdouble on this host is x87 80-bit extended
(64-bit significand, `np.finfo(np.longdouble).precision == 18`, eps 1.08e-19), NOT binary128, so it is NOT used.
Fortran typing rules honoured: a sub-expression of two REAL*8 operands is evaluated in double first (`mu*Si`, `shw-shi`,
`Eit+lhm`, `MICE/(MICE+SNOWL)`, `frac*lhm`); `Tm` (quad) is the double value of -mu*Si widened; the result is narrowed to REAL*8 by
round-to-nearest.  Differences from the float64 port beyond precision: the real tests `Si.gt.0q0` (float64 port: Si > 1e-10) and
`abs(Ei+lhm).LT.1q-10` (quad 1e-10, not the double 1e-10).

Nothing in an existing module is modified: `patched()` swaps seaice_core_ff.Ti/Ti2b (and the names bound by `from ... import` in
seaice_to_atmgrid_ff) for the duration of a `with` block.
"""
import contextlib
import os
import sys

from mpmath import mp, mpf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S  # noqa: E402

PREC = 113  # binary128 significand bits


def _q(x):
    return mpf(x)


def ti_quad(ei, si):
    """SEAICE.f:2363 Ti (BP)."""
    with mp.workprec(PREC):
        if si > 0.0:
            tm = _q(-S.MU * si)                      # double product, widened
            if ei >= S.SHW * tm:                     # quad compare (product of two doubles is exact in 113 bits)
                return float(tm)
            b = tm * _q(S.SHW - S.SHI) - _q(ei + S.LHM)
            c = _q(S.LHM) * tm
            det = b * b - (_q(4) * _q(S.SHI)) * c
            return float((_q(-0.5) * (b + mp.sqrt(det))) * _q(S.BYSHI))
        t = (ei + S.LHM) * S.BYSHI                   # double
        if _q(abs(ei + S.LHM)) < _q(1) / _q(10 ** 10):
            t = 0.0
        return t


def ti2b_quad(eit, si, snowl, mice):
    """SEAICE.f:2403 Ti2b (BP)."""
    with mp.workprec(PREC):
        if si > 0.0:
            tm = _q(-S.MU * si)
            frac = mice / (mice + snowl)             # double
            b = (_q(frac) * tm) * _q(S.SHW - S.SHI) - _q(eit + S.LHM)
            c = _q(frac * S.LHM) * tm
            det = b * b - (_q(4) * _q(S.SHI)) * c
            return float((_q(-0.5) * (b + mp.sqrt(det))) * _q(S.BYSHI))
        t = (eit + S.LHM) * S.BYSHI
        if _q(abs(eit + S.LHM)) < _q(1) / _q(10 ** 10):
            t = 0.0
        return t


@contextlib.contextmanager
def patched():
    """Temporarily replace Ti/Ti2b by the binary128 versions in seaice_core_ff and in seaice_to_atmgrid_ff."""
    import seaice_to_atmgrid_ff as A
    saved = [(S, "Ti", S.Ti), (S, "Ti2b", S.Ti2b), (A, "Ti", A.Ti), (A, "Ti2b", A.Ti2b)]
    try:
        S.Ti, S.Ti2b = ti_quad, ti2b_quad
        A.Ti, A.Ti2b = ti_quad, ti2b_quad
        yield
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)
