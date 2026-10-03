"""Full-fidelity port of OCNKPP.f's OVDIFF (momentum) -- Stage 2, D56.

OVDIFF (OCNKPP.f:3410-3469) is the implicit vertical-diffusion solver for velocity (UL/ULD in
OCONV). Structurally different from OVDIFFS (D55): its off-diagonal coefficients use DTBYDZ(L),
its bottom RHS reads DTP4(LMIJ) (not LMIJ-1), and it takes no separate DT argument. Built on the
same plain TRIDIAG as D55 (ovdiffs_ff.tridiag). Arrays are 1-based, length LMO+1.
"""
import numpy as np

from ovdiffs_ff import tridiag, LMO


def ovdiff(k, ghat, dtp4, dtbydz, bydz2, lmij, u0):
    a = np.zeros(LMO + 1); b = np.zeros(LMO + 1)
    c = np.zeros(LMO + 1); r = np.zeros(LMO + 1)
    c[1] = -dtbydz[1] * bydz2[1] * k[1]
    b[1] = 1.0 - c[1]
    tmp = u0[1]
    r[1] = tmp - dtbydz[1] * ghat[1] + dtp4[1]
    for L in range(2, lmij):
        a[L] = -dtbydz[L] * bydz2[L - 1] * k[L - 1]
        c[L] = -dtbydz[L] * bydz2[L] * k[L]
        b[L] = 1.0 + dtbydz[L] * (bydz2[L - 1] * k[L - 1] + bydz2[L] * k[L])
        tmp = u0[L]
        r[L] = tmp + dtbydz[L] * (ghat[L - 1] - ghat[L]) + dtp4[L]
    a[lmij] = -dtbydz[lmij] * bydz2[lmij - 1] * k[lmij - 1]
    b[lmij] = 1.0 - a[lmij]
    c[lmij] = 0.0
    tmp = u0[lmij]
    r[lmij] = tmp + dtbydz[lmij] * ghat[lmij - 1] + dtp4[lmij]
    return tridiag(a, b, c, r, lmij)
