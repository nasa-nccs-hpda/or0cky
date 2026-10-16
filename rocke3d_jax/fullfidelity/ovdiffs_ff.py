import numpy as np

LMO = 13


def tridiag(a, b, c, r, n):
    u = np.zeros(LMO + 1)
    gam = np.zeros(LMO + 1)
    bet = b[1]
    u[1] = r[1] / bet
    for j in range(2, n + 1):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 1, 0, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def ovdiffs(k, ghat, dtp4, dtbydz, bydz2, dt, lmij, u0):
    a = np.zeros(LMO + 1); b = np.zeros(LMO + 1)
    c = np.zeros(LMO + 1); r = np.zeros(LMO + 1)
    a[1] = 0.0
    b[1] = 1.0 + dtbydz[1] * bydz2[1] * k[1]
    c[1] = -dtbydz[2] * bydz2[1] * k[1]
    r[1] = u0[1] - dt * ghat[1] + dtp4[1]
    for L in range(2, lmij):
        a[L] = -dtbydz[L - 1] * bydz2[L - 1] * k[L - 1]
        b[L] = 1.0 + dtbydz[L] * (bydz2[L - 1] * k[L - 1] + bydz2[L] * k[L])
        c[L] = -dtbydz[L + 1] * bydz2[L] * k[L]
        r[L] = u0[L] + dt * (ghat[L - 1] - ghat[L]) + dtp4[L]
    a[lmij] = -dtbydz[lmij - 1] * bydz2[lmij - 1] * k[lmij - 1]
    b[lmij] = 1.0 + dtbydz[lmij] * bydz2[lmij - 1] * k[lmij - 1]
    c[lmij] = 0.0
    r[lmij] = u0[lmij] + dt * ghat[lmij - 1] + dtp4[lmij - 1]
    u = tridiag(a, b, c, r, lmij)
    fl = np.zeros(LMO + 1)
    for L in range(1, lmij):
        fl[L] = k[L] * (dtbydz[L + 1] * u[L + 1] - dtbydz[L] * u[L]) * bydz2[L] - dt * ghat[L]
    return u, fl
