"""Full-fidelity port of OCNKPP.f's REDUCE_FIG -- Stage 2, D57 (OCNKPP.f:3517-3530).

Reduces a real*8 value to NSIG significant bits when it carries more precision than the
calculation warrants. Uses Fortran's EXPONENT, SCALE and NINT semantics: EXPONENT(x) is the e
with x = f*2**e, f in [0.5, 1) (and 0 for x == 0); NINT rounds half away from zero.
"""
import math


def _exponent(x):
    if x == 0.0:
        return 0
    return math.frexp(x)[1]


def _nint(x):
    return math.copysign(math.floor(abs(x) + 0.5), x)


def reduce_fig(nsig, rx):
    if nsig + 30 > _exponent(rx):
        return math.ldexp(_nint(math.ldexp(rx, -nsig)), nsig)
    return rx
