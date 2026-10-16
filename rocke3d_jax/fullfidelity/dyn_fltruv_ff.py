"""D90: end-of-DYNAM velocity filter chain of ATMDYN.f ported to numpy.

Ports (Fortran line numbers in model/ATMDYN.f of modelE2_planet_2.0):
  FLTRUV                       1702-1812  (8th-order E-W Shapiro on U and V, then
                                           per-row angular-momentum fix, ANG_UV=1)
  fltry2 (for U, then V)       1627-1700  (8-pass N-S filter with pole crossing)
  CONSERV_AMB_EXT              2181-2211
  ADD_AM_AS_SOLIDBODY_ROTATION 2146-2179
  glue in DYNAM                379-388
DIAGCD calls (diagnostic accumulation only) are not ported.

Array conventions (0-based numpy): U,V (IM,JM,LM); MA (LM,IM,JM); MASUM (IM,JM);
geometry vectors indexed by Fortran J-1.  Rundeck constants: NSHAP=8, DT=450,
DT_XUfilter=DT_XVfilter=450, DT_YU/YVfilter=0 (fltry2 strength fixed at 1d0 in the
call), ANG_UV=1, serial domain (J_0STG=2, J_1STG=JM, both poles on one rank).

Reductions follow the Fortran sequential order (explicit loops): the I loop in FLTRUV
starts at I=IM then 1..IM-1.  The Fortran SUM intrinsics over L / I / J in
CONSERV_AMB_EXT, GLOBALSUM and ADD_AM_AS_SOLIDBODY_ROTATION are replaced by
`seqsum`; the `order` argument is only there to diagnose compiler reassociation.
"""
import numpy as np

IM, JM, LM = 72, 46, 40
NSHAP = 8
BY4TON = 1.0 / (4.0 ** NSHAP)          # real*8 parameter from real expr, exact (2**-16)
DT, DT_XUFILTER, DT_XVFILTER = 450.0, 450.0, 450.0


def geometry(radius, im=IM, jm=JM):
    """Analytic GEOM_B.f (lines 150-262, half-polar-box 4x5 case): DXYN, DXYS, COSV.
    Returns dict of length-JM arrays (index = Fortran J - 1).  RADIUS/OMEGA are
    runtime planet parameters (USE_PLANET_RAD) and are taken from the dump header."""
    dlon = 2.0 * np.pi / im
    fjeq = 0.5 * (1 + jm)
    radian = np.pi / 180.0
    dlat = (180.0 / (jm - 1)) * radian
    cosp = np.zeros(jm)
    for j in range(2, jm):
        cosp[j - 1] = np.cos(dlat * (j - fjeq))
    cosp1 = np.cos(dlat * (1. - fjeq))
    cosv = np.zeros(jm)
    for j in range(2, jm + 1):
        cosv[j - 1] = .5 * (cosp[j - 2] + cosp[j - 1])
        if j == 2:
            cosv[j - 1] = .5 * (cosp1 + cosp[j - 1])
        if j == jm:
            cosv[j - 1] = .5 * (cosp[j - 2] + cosp1)
    dxyp = np.zeros(jm)
    dxyp[0] = radius * radius * dlon * (np.sin(dlat * (1 + .5 - fjeq)) + 1)
    dxyp[jm - 1] = radius * radius * dlon * (1 - np.sin(dlat * (jm - .5 - fjeq)))
    for j in range(2, jm):
        dxyp[j - 1] = radius * radius * dlon * (np.sin(dlat * (j + .5 - fjeq))
                                                - np.sin(dlat * (j - .5 - fjeq)))
    dxys = np.zeros(jm)
    dxyn = np.zeros(jm)
    dxys[jm - 1] = dxyp[jm - 1]
    dxyn[0] = dxyp[0]
    dxys[1:jm - 1] = .5 * dxyp[1:jm - 1]
    dxyn[1:jm - 1] = .5 * dxyp[1:jm - 1]
    return dict(dxyn=dxyn, dxys=dxys, cosv=cosv, radius=radius)


def seqsum(a, axis=0):
    """Strictly sequential (left-to-right) sum along `axis`."""
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    s = a[0].copy()
    for k in range(1, a.shape[0]):
        s = s + a[k]
    return s


def _shapiro_x(q):
    """8 passes of X(I)=X(I-1)-2X(I)+X(I+1) (cyclic, all right-hand sides use the
    pre-pass values, as in the in-place Fortran loop); q is (IM, ...)."""
    x = q.copy()
    for _ in range(NSHAP):
        x = ((np.roll(x, 1, axis=0) - x) - x) + np.roll(x, -1, axis=0)
    return x


def fltruv(u, v, ma, geo, ang_uv=1, nshap_check=True):
    """FLTRUV: E-W Shapiro filter of U,V at J=2..JM, then per-row angular-momentum fix
    on U.  Returns new (u, v); inputs are not modified."""
    u = u.copy()
    v = v.copy()
    usave = u.copy()
    xu = (DT / DT_XUFILTER) * BY4TON if DT_XUFILTER > 0 else 0.0
    xv = (DT / DT_XVFILTER) * BY4TON if DT_XVFILTER > 0 else 0.0
    dxyn, dxys = geo['dxyn'], geo['dxys']
    sl = slice(1, JM)                       # J = 2..JM
    u[:, sl, :] = u[:, sl, :] - _shapiro_x(u[:, sl, :]) * xu
    v[:, sl, :] = v[:, sl, :] - _shapiro_x(v[:, sl, :]) * xv
    # angular momentum fix: sums run I=IM,1,2,..,IM-1 (Fortran loop order)
    order = [IM - 1] + list(range(IM - 1))
    angm = np.zeros((JM - 1, LM))
    mmuvs = np.zeros((JM - 1, LM))
    # ma: (LM, IM, JM) -> arrays over (J,L)
    for i in order:
        ip1 = (i + 1) % IM
        ms = (ma[:, ip1, 0:JM - 1] + ma[:, i, 0:JM - 1]) * dxyn[0:JM - 1][None, :]
        mn = (ma[:, ip1, 1:JM] + ma[:, i, 1:JM]) * dxys[1:JM][None, :]
        mmuv = .5 * (ms + mn)               # (LM, JM-1)
        mmuv = mmuv.T                       # (JM-1, LM)
        mmuvs = mmuvs + mmuv
        angm = angm - mmuv * (u[i, 1:JM, :] - usave[i, 1:JM, :])
    if ang_uv == 1:
        u[:, 1:JM, :] = u[:, 1:JM, :] + (angm / mmuvs)[None, :, :]
    return u, v


def fltry2(q, strength=1.0):
    """fltry2: 8-pass N-S filter on a staggered field at J=2..JM (q is (IM,JM,LM))."""
    yv = min(strength, 1.0) * BY4TON * ((-1) ** NSHAP)
    q = q.copy()
    yn = q[:, 1:, :].copy()                 # J = 2..JM -> index 0..JM-2
    h = IM // 2
    for _ in range(NSHAP):
        new = np.empty_like(yn)
        # south pole crossing: yjm1(i) = -yn(i+im/2, J=2)  (i<=im/2), -yn(i-im/2,J=2) else
        south = -np.concatenate([yn[h:, 0, :], yn[:h, 0, :]], axis=0)
        left = np.concatenate([south[:, None, :], yn[:, :-1, :]], axis=1)   # yjm1
        # interior J=2..JM-1: yjm1 - yj - yj + yn(j+1)
        new[:, :-1, :] = ((left[:, :-1, :] - yn[:, :-1, :]) - yn[:, :-1, :]) + yn[:, 1:, :]
        # north pole J=JM: yjm1 - yj - yj - (partner across the pole)
        yj = yn[:, -1, :]
        partner = np.concatenate([yj[h:], yj[:h]], axis=0)
        new[:, -1, :] = ((left[:, -1, :] - yj) - yj) - partner
        yn = new
    q[:, 1:, :] = q[:, 1:, :] - yn * yv
    return q


def conserv_amb_ext(u, ma, geo, omega):
    """CONSERV_AMB_EXT: column angular momentum on the B grid, AM (IM,JM), AM(:,1)=0."""
    dxyn, dxys, cosv, radius = geo['dxyn'], geo['dxys'], geo['cosv'], geo['radius']
    am = np.zeros((IM, JM))
    ip1 = (np.arange(IM) + 1) % IM
    for j in range(2, JM + 1):              # Fortran J
        jj = j - 1
        a = (ma[:, :, jj - 1] + ma[:, ip1, jj - 1]) * dxyn[jj - 1]    # (LM, IM)
        b = (ma[:, :, jj] + ma[:, ip1, jj]) * dxys[jj]
        w = (a + b) * (u[:, jj, :].T + cosv[jj] * radius * omega)     # (LM, IM)
        s = seqsum(w, axis=0)                                         # sum over L
        am[:, jj] = s * .5 * cosv[jj] * radius
    return am


def add_am_as_solidbody_rotation(u, dam, masum, geo):
    """ADD_AM_AS_SOLIDBODY_ROTATION."""
    dxyn, dxys, cosv, radius = geo['dxyn'], geo['dxys'], geo['cosv'], geo['radius']
    u = u.copy()
    masumj = seqsum(masum, axis=0)          # (JM,)
    xj = np.zeros(JM)
    for j in range(2, JM + 1):
        jj = j - 1
        xj[jj] = cosv[jj] ** 2 * (masumj[jj - 1] * dxyn[jj - 1] + masumj[jj] * dxys[jj])
    xglob = seqsum(xj, axis=0)
    dueq = dam / (radius * xglob)
    u[:, 1:, :] = u[:, 1:, :] + (dueq * cosv[1:])[None, :, None]
    return u


def filter_chain(u, v, ma, masum, geo, omega, ang_uv=1, return_stages=False):
    """Full end-of-DYNAM sequence (ATMDYN.f:379-388).  Returns (u, v, damsum) and, if
    requested, a dict of intermediate stages ('flt', 'ny' (+am1))."""
    u1, v1 = fltruv(u, v, ma, geo, ang_uv=ang_uv)
    am1 = conserv_amb_ext(u1, ma, geo, omega)
    u2 = fltry2(u1, 1.0)
    v2 = fltry2(v1, 1.0)
    am2 = conserv_amb_ext(u2, ma, geo, omega)
    d = am1 - am2
    d[:, 0] = am2[:, 0]                     # J=1 row keeps am2(:,1) (=0): not overwritten
    # GLOBALSUM_IJ: zonal sum over I per J (J=1..JM), then sum over J
    zon = seqsum(d, axis=0)
    damsum = seqsum(zon, axis=0)
    u3 = add_am_as_solidbody_rotation(u2, damsum, masum, geo)
    if return_stages:
        return u3, v2, damsum, dict(flt=(u1, v1), ny=(u2, v2), am1=am1)
    return u3, v2, damsum
