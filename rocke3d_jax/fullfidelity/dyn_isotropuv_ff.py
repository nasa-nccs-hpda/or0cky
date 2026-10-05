"""D95: isotropuv + shap1 (+ Hemisphere / at_pole / far_from_pole) of ATMDYN.f ported to numpy.

Fortran (model/ATMDYN.f of modelE2_planet_2.0): isotropuv 1838-1900, Hemisphere 1902-1910,
at_pole 1913-1923, far_from_pole 1925-1931, shap1 1934-1955; FFT/FFTI from dyn_avrx_ff.py (bitwise
port of FFT72.f).  COS_LIMIT=0.15 (ATMDYN_COM.F90:33), DT=450 (ATMDYN_COM.F90:39), serial domain
(J_0STG=2, J_1STG=JM).

Rows with COSV(J) < 0.15 (J=2,3,JM-1,JM for the 4x5 grid) are processed for every layer L:
  ua = cosi*u - hemi*sini*v ; va = cosi*v + hemi*sini*u                  (x-y components)
  k  = max|u(:,j,l)|*2.*dt/dxv(j) -> klo=1e3 (k<0.5), khi=1e7 (k>1), else linear in between
  fac = k*dt/(dxv*dxv); shap1(ua,fac); shap1(va,fac)    (n=int(fac)+1 sub-iterations of
                                                           x += fac/(4n)*(x(i-1)-2x(i)+x(i+1)), cyclic)
  at the pole rows (J=2, JM): FFT, zero harmonics 2..IMH, FFTI (for ua and va)
  u = cosi*ua + hemi*sini*va ; v = cosi*va - hemi*sini*ua
Everything is elementwise IEEE double arithmetic in the Fortran statement order, vectorised over the
(j,l) rows; sub-iteration counts differ per row, handled with per-row masks.
"""
import numpy as np

import dyn_avrx_ff as fa

IM, JM, LM = 72, 46, 40
KLO, KHI = 1e3, 1e7
COS_LIMIT = 0.15
DT = 450.0


def hemisphere(j, fjeq):
    return -1 if j < fjeq else 1


def at_pole(j, jm=JM):
    return j == jm or j == 2


def far_from_pole(j, cosv, coscut=COS_LIMIT):
    return cosv[j - 1] >= coscut


def shap1_rows(x, fac):
    """shap1 on rows x (R, IM) with per-row fac (R,).  Returns new array and the n per row.
    n = int(fac)+1; facby4 = fac*.25d0/n; each sub-iteration uses the pre-pass neighbours."""
    x = np.array(x, dtype=float)
    fac = np.asarray(fac, dtype=float)
    n = fac.astype(int) + 1                       # int() truncates toward zero; fac >= 0 here
    facby4 = fac * .25 / n
    for nn in range(1, int(n.max()) + 1):
        act = n >= nn
        xs = x[act]
        new = xs + facby4[act][:, None] * (((np.roll(xs, 1, axis=1) - xs) - xs) + np.roll(xs, -1, axis=1))
        x[act] = new
    return x, n


def _k_of_u(umax, dxv, dt=DT):
    k = umax * 2. * dt / dxv
    kk = np.where(k < 0.5, KLO, np.where(k > 1.0, KHI, KLO + 2. * (k - 0.5) * (KHI - KLO)))
    return k, kk


def iso_rows(U, V, jj, geo, C, S, dt=DT, return_all=False):
    """Core: U,V (R, IM) rows at Fortran latitude indices jj (R,).  Returns (Uo, Vo) and, with
    return_all, a dict with k, fac, n, ua, va (after shap1, before the pole FFT)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    jj = np.asarray(jj, dtype=int)
    cosi, sini = geo['cosiv'], geo['siniv']
    hemi = np.array([hemisphere(j, geo['fjeq']) for j in jj], dtype=float)[:, None]
    dxv = geo['dxv'][jj - 1]
    ua = cosi[None, :] * U - (hemi * sini[None, :]) * V
    va = cosi[None, :] * V + (hemi * sini[None, :]) * U
    umax = np.max(np.abs(U), axis=1)
    k, kk = _k_of_u(umax, dxv, dt)
    fac = kk * dt / (dxv * dxv)
    ua, n = shap1_rows(ua, fac)
    va, _ = shap1_rows(va, fac)
    ua1, va1 = ua.copy(), va.copy()
    pole = np.array([at_pole(j) for j in jj])
    if pole.any():
        for arr in (ua, va):
            A, B = fa.fft72(arr[pole], C, S)
            A[:, 2:fa.IMH + 1] = 0.
            B[:, 2:fa.IMH + 1] = 0.
            arr[pole] = fa.ffti72(A, B, C, S)
    Uo = cosi[None, :] * ua + (hemi * sini[None, :]) * va
    Vo = cosi[None, :] * va - (hemi * sini[None, :]) * ua
    if return_all:
        return Uo, Vo, dict(k=kk, kraw=k, fac=fac, n=n, ua=ua1, va=va1)
    return Uo, Vo


def isotropuv(u, v, geo, C, S, dt=DT, coscut=COS_LIMIT):
    """Full-field isotropuv on u,v (IM,JM,LM) (numpy 0-based, Fortran (I,J,L)); returns new copies."""
    u = u.copy()
    v = v.copy()
    rows = [j for j in range(2, JM + 1) if not far_from_pole(j, geo['cosv'], coscut)]
    if not rows:
        return u, v
    jj = np.array([j for l in range(LM) for j in rows])
    ll = np.array([l for l in range(LM) for j in rows])
    U = u[:, jj - 1, ll].T
    V = v[:, jj - 1, ll].T
    Uo, Vo = iso_rows(U, V, jj, geo, C, S, dt)
    u[:, jj - 1, ll] = Uo.T
    v[:, jj - 1, ll] = Vo.T
    return u, v
