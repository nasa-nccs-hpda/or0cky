"""D137: numpy port of ODIFF (ocean horizontal momentum diffusion, Wajsowicz ADI) and its one-time setup init_ODIFF.

Sources (read in full, ModelE2 planet 2.0 tree, read-only):
  OCNDYN.f:5092-5575   SUBROUTINE ODIFF (x semi-implicit sweep per layer, then y sweep on all layers)
  OCNDYN.f:1236-1498   SUBROUTINE init_odiff (KHP/KHV/TANP/TANV and the fixed operators UXA..VYC)
  OCNDYN2.f:1530-1564  polevel (called inside ODIFF under ODIFF_FIXES_2017)
  solvers/TRIDIAG.f    TRIDIAG (x sweep: division form, OPTIMIZED_TRIDIAG undefined) and TRIDIAG_3D_DIST_new
                       (y sweep: reciprocal form), OGEOM.f (geometry), OCEAN_COM.f:170 (FSLIP=0)
  OCNDYN2.f:556-564    call site: every 6th step (mod(itime, int(10800/1800)) == 0), DTDIFF = 10800 s.
Preprocessor state for P2SAoM40: ODIFF_FIXES_2017 (rundeck) and ODIFF_FIXES_2022 (OCNDYN.f:10) defined,
ODIFF_TRIDIAG_CYCLIC not defined (explicit cyclic terms + plain TRIDIAG), OPTIMIZED_TRIDIAG assumed undefined.

Fidelity notes: SQRT(3.) in the Munk-length line is a single-precision literal in the Fortran (promoted to double
afterwards), reproduced here; DLAT = oDLAT_DG*radian with radian = pi/180 (MathematicalConstants, aux files), RADIUS =
6371000 and OMEGA = 2*pi/(86400*365/366) as used by the other ocean ports; AKHFAC=1 (no rundeck override assumed),
AKHMIN = 1.5e8 for JM=46. State arrays are 0-based (IM, JM, LMO) as in ocean_chain_io.
"""
import numpy as np

IM, JM, LMO = 72, 46, 13
IVNP0 = IM // 4 - 1
RADIUS = 6371000.0
OMEGA = 2.0 * np.pi / (86400.0 * 365.0 / 366.0)
RHOWS = 1030.0
TWOPI = 2.0 * np.pi
FSLIP = 0.0
AKHFAC = 1.0
AKHMIN = 1.5e8
DTDIFF = 3.0 * 3600.0
SQRT3_F32 = float(np.sqrt(np.float32(3.0)))      # SQRT(3.) in single precision


def geometry():
    """OGEOM.f quantities ODIFF needs, 1-based arrays of length JM+1 ([0] unused/zero) so J reads like the Fortran."""
    dlon = TWOPI / IM
    fjeq = 0.5 * (1 + JM)
    dlat = 4.0 * (np.pi / 180.0)          # oDLAT_DG = NINT(180/(JM-1)) = 4, times radian
    cosv = np.zeros(JM + 1); dxvo = np.zeros(JM + 1); dyvo = np.zeros(JM + 1)
    for j in range(1, JM):
        cosv[j] = np.cos(dlat * (j + .5 - fjeq))
        dxvo[j] = RADIUS * dlon * cosv[j]
        dyvo[j] = RADIUS * dlat
    rlat = np.zeros(JM + 1); cospo = np.zeros(JM + 1); dxypo = np.zeros(JM + 1)
    dxpo = np.zeros(JM + 1); dypo = np.zeros(JM + 1)
    for j in range(1, JM + 1):
        latn = dlat * (j + .5 - fjeq)
        if j == JM:
            latn = TWOPI / 4
        lats = dlat * (j - .5 - fjeq)
        if j == 1:
            lats = -TWOPI / 4
        sinn, sins = np.sin(latn), np.sin(lats)
        rlat[j] = dlat * (j - fjeq)
        cospo[j] = np.cos(rlat[j])
        dxypo[j] = RADIUS * RADIUS * dlon * (sinn - sins)
        dxpo[j] = .5 * RADIUS * dlon * (cosv[j - 1] + cosv[j])
        dypo[j] = RADIUS * (latn - lats)
    rlat[1] = -TWOPI / 4
    rlat[JM] = TWOPI / 4
    dxyvo = np.zeros(JM + 1)
    for j in range(1, JM):
        dxyvo[j] = .5 * dxypo[j] + .5 * dxypo[j + 1]       # DXYN(J)+DXYS(J+1)
    return dict(dlat=dlat, cosv=cosv, dxvo=dxvo, dyvo=dyvo, rlat=rlat, cospo=cospo, dxypo=dxypo, dxpo=dxpo,
                dypo=dypo, dxyvo=dxyvo)


def trig_tables():
    """COSIC, SINIC, COSU, SINU (length IM, 0-based) as OGEOM.f."""
    i1 = np.arange(1, IM + 1, dtype=np.float64)
    cosic = np.cos((i1 - 0.5) * TWOPI / IM); sinic = np.sin((i1 - 0.5) * TWOPI / IM)
    cosu = np.cos(i1 * TWOPI / IM); sinu = np.sin(i1 * TWOPI / IM)
    sinu[IM - 1] = 0.0; cosu[IM - 1] = 1.0
    return cosic, sinic, cosu, sinu


_TAB = {}


def init_odiff(lmu, lmv):
    """init_ODIFF (OCNDYN.f:1236-1498): returns dict of 1-based (length JM+1) line arrays (KHP, KHV, TANP, TANV, BYD*)
    and 0-based (IM, JM, LMO) operators UXA..VYC. lmu, lmv: (IM, JM) integer depth indices."""
    key = (lmu.tobytes(), lmv.tobytes())
    if key in _TAB:
        return _TAB[key]
    g = geometry()
    dxpo, dypo, dxvo, dyvo, dxyvo, dxypo = g['dxpo'], g['dypo'], g['dxvo'], g['dyvo'], g['dxyvo'], g['dxypo']
    dlat = g['dlat']
    n = JM + 1
    khp = np.zeros(n); khv = np.zeros(n); bydxyv = np.zeros(n); bydxv = np.zeros(n); bydxp = np.zeros(n)
    bydyv = np.zeros(n); bydyp = np.zeros(n); kypxp = np.zeros(n); kxpyv = np.zeros(n); kxvyp = np.zeros(n)
    kyvxv = np.zeros(n); tanp = np.zeros(n); tanv = np.zeros(n)
    bydxypo = np.zeros(n)
    for j in range(1, JM + 1):
        bydxypo[j] = 1.0 / dxypo[j]
    for j in range(1, JM):                         # J_0=1 .. J_1S=JM-1
        dsp = np.minimum(dxpo[j], dypo[j]) * 2.0 * SQRT3_F32 / TWOPI
        dsv = np.minimum(dxvo[j], dyvo[j]) * 2.0 * SQRT3_F32 / TWOPI
        khp[j] = AKHFAC * 2.0 * RHOWS * abs(OMEGA) * g['cospo'][j] * (dsp * dsp * dsp) / RADIUS
        cosvo_j = g['cosv'][j]
        khv[j] = AKHFAC * 2.0 * RHOWS * abs(OMEGA) * cosvo_j * (dsv * dsv * dsv) / RADIUS
        khp[j] = max(khp[j], AKHFAC * AKHMIN)
        khv[j] = max(khv[j], AKHFAC * AKHMIN)
        bydxyv[j] = 1.0 / dxyvo[j]; bydxv[j] = 1.0 / dxvo[j]; bydxp[j] = 1.0 / dxpo[j]
        bydyv[j] = 1.0 / dyvo[j]; bydyp[j] = 1.0 / dypo[j]
        kypxp[j] = khp[j] * dypo[j] * bydxp[j]
        kxpyv[j] = khv[j] * dxvo[j] * bydyv[j]       # ODIFF_FIXES_2017
        kxvyp[j] = khp[j] * dxpo[j] * bydyp[j]
        kyvxv[j] = khv[j] * dyvo[j] * bydxv[j]
        tanp[j] = np.tan(g['rlat'][j]) * np.tan(0.5 * dlat) / (RADIUS * 0.5 * dlat)
        vlat = dlat * (j + 0.5 - 0.5 * (1 + JM))
        tanv[j] = np.tan(vlat) * np.sin(dlat) / (dlat * RADIUS)
    khv[1] = khv[2]
    bydxv[1] = 1.0 / dxvo[1]; bydyv[1] = 1.0 / dyvo[1]; bydyp[1] = 1.0 / dypo[1]
    tanp[1] = 0.0
    bydxp[JM] = 1.0 / dxpo[JM]
    tanp[JM] = 0.0
    # ---- fixed operators, 1-based internally (IM, JM+2) views are built per layer, stored 0-based
    ops = {k: np.zeros((IM, JM, LMO)) for k in ('uxa', 'uxb', 'uxc', 'uya', 'uyb', 'uyc', 'vxa', 'vxb', 'vxc', 'vya', 'vyb', 'vyc')}
    lmu1 = np.zeros((IM, JM + 2), dtype=np.int64); lmv1 = np.zeros((IM, JM + 2), dtype=np.int64)
    lmu1[:, 1:JM + 1] = lmu; lmv1[:, 1:JM + 1] = lmv          # [i, J] with J 1-based
    for l in range(1, LMO + 1):
        mu = lmu1 >= l; mv = lmv1 >= l
        z = lambda: np.zeros((IM, JM + 2))            # noqa: E731
        dudx1, dudx2, dudy1, dudy2, dvdx1, dvdx2, dvdy1, dvdy2 = z(), z(), z(), z(), z(), z(), z(), z()
        for j in range(2, JM):
            # DUDX(IP1,J,1)=KYPXP if L<=LMU(IP1,J); DUDX(IP1,J,2)=-KYPXP if L<=LMU(I,J)
            dudx1[:, j] = np.where(mu[:, j], kypxp[j], 0.0)
            dudx2[:, j] = np.where(np.roll(mu[:, j], 1), -kypxp[j], 0.0)
            a = kxpyv[j] * (1. + 0.5 * tanv[j] * dyvo[j]); b = -kxpyv[j] * (1. - 0.5 * tanv[j] * dyvo[j])
            both = mu[:, j] & mu[:, j + 1]
            dudy1[:, j] = np.where(both, a, np.where(~mu[:, j] & mu[:, j + 1], (1. - FSLIP) * 2.0 * kxpyv[j], 0.0))
            dudy2[:, j] = np.where(both, b, np.where(mu[:, j] & ~mu[:, j + 1], -(1. - FSLIP) * 2.0 * kxpyv[j], 0.0))
            # DVDY(I,J+1,*) (ODIFF_FIXES_2017 pole branch)
            r = j + 1
            rr = r if r != JM else j
            dvdy1[:, r] = np.where(mv[:, r], kxvyp[rr] * (1. + 0.5 * tanp[rr] * dypo[rr]), 0.0)
            dvdy2[:, r] = np.where(mv[:, j], -kxvyp[rr] * (1. - 0.5 * tanp[rr] * dypo[rr]), 0.0)
            nxt = np.roll(mv[:, j], -1)
            dvdx1[:, j] = np.where(mv[:, j] & nxt, kyvxv[j], np.where(~mv[:, j] & nxt, (1. - FSLIP) * 2.0 * kyvxv[j], 0.0))
            dvdx2[:, j] = np.where(mv[:, j] & nxt, -kyvxv[j], np.where(mv[:, j] & ~nxt, -(1. - FSLIP) * 2.0 * kyvxv[j], 0.0))
        dudy1[:, 1] = np.where(mu[:, 2], (1. - FSLIP) * 2.0 * kxpyv[1], 0.0)     # south pole as a 1-point island
        for j in range(2, JM):
            by = bydxypo[j]; byv = bydxyv[j]
            im1 = lambda a: np.roll(a, 1, axis=0)       # noqa: E731  a[I-1]
            ip1 = lambda a: np.roll(a, -1, axis=0)      # noqa: E731  a[I+1]
            # U: operator at cell k=IM1 uses DUDX at k and k+1
            m = mu[:, j]
            ops['uxa'][:, j - 1, l - 1] = np.where(m, -dudx2[:, j] * by, 0.0)
            ops['uxb'][:, j - 1, l - 1] = np.where(m, (ip1(dudx2[:, j]) - dudx1[:, j]) * by, 0.0)
            ops['uxc'][:, j - 1, l - 1] = np.where(m, ip1(dudx1[:, j]) * by, 0.0)
            ops['uya'][:, j - 1, l - 1] = np.where(m, -dudy2[:, j - 1] * by + 0.5 * tanp[j] * khp[j] * bydyp[j], 0.0)
            ops['uyb'][:, j - 1, l - 1] = np.where(m, (dudy2[:, j] - dudy1[:, j - 1]) * by - tanp[j] * tanp[j] * khp[j], 0.0)
            ops['uyc'][:, j - 1, l - 1] = np.where(m, dudy1[:, j] * by - 0.5 * tanp[j] * khp[j] * bydyp[j], 0.0)
            m = mv[:, j]
            ops['vxa'][:, j - 1, l - 1] = np.where(m, -im1(dvdx2[:, j]) * byv, 0.0)
            ops['vxb'][:, j - 1, l - 1] = np.where(m, (dvdx2[:, j] - im1(dvdx1[:, j])) * byv, 0.0)
            ops['vxc'][:, j - 1, l - 1] = np.where(m, dvdx1[:, j] * byv, 0.0)
            ops['vya'][:, j - 1, l - 1] = np.where(m, -dvdy2[:, j] * byv + 0.5 * tanv[j] * khv[j] * bydyv[j], 0.0)
            ops['vyb'][:, j - 1, l - 1] = np.where(m, (dvdy2[:, j + 1] - dvdy1[:, j]) * byv - tanv[j] * tanv[j] * khv[j], 0.0)
            ops['vyc'][:, j - 1, l - 1] = np.where(m, dvdy1[:, j + 1] * byv - 0.5 * tanv[j] * khv[j] * bydyv[j], 0.0)
    out = dict(g=g, khp=khp, khv=khv, tanp=tanp, tanv=tanv, bydxyv=bydxyv, bydxv=bydxv, bydxp=bydxp, bydyv=bydyv,
               bydyp=bydyp, bydxypo=bydxypo, **ops)
    _TAB[key] = out
    return out


def _tridiag_div(a, b, c, r):
    """TRIDIAG (division form) vectorised over trailing axes; leading axis is the system (length n)."""
    n = a.shape[0]
    u = np.zeros_like(r); gam = np.zeros_like(r)
    bet = b[0].copy()
    if (bet == 0).any():
        raise ZeroDivisionError('TRIDIAG: denominator = zero')
    u[0] = r[0] / bet
    for j in range(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        if (bet == 0).any():
            raise ZeroDivisionError('TRIDIAG: denominator = zero')
        u[j] = (r[j] - a[j] * u[j - 1]) / bet
    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def _tridiag_recip(a, b, c, r):
    """TRIDIAG_3D_DIST_new (reciprocal form)."""
    n = a.shape[0]
    u = np.zeros_like(r); gam = np.zeros_like(r)
    bet = b[0]
    if (bet == 0).any():
        raise ZeroDivisionError('TRIDIAG_new: denominator = zero')
    bybet = 1.0 / bet
    u[0] = r[0] * bybet
    for j in range(1, n):
        gam[j] = c[j - 1] * bybet
        bet = b[j] - a[j] * gam[j]
        if (bet == 0).any():
            raise ZeroDivisionError('TRIDIAG_new: denominator = zero')
        bybet = 1.0 / bet
        u[j] = (r[j] - a[j] * u[j - 1]) * bybet
    for j in range(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def polevel_all(uo, vo, lmv, cosic, sinic, cosu, sinu):
    """polevel for every layer (in place on uo, vo row JM-1 0-based)."""
    j = JM - 2
    mv = (lmv[:, j][:, None] >= np.arange(1, LMO + 1)[None, :])
    unp = np.zeros(LMO); vnp = np.zeros(LMO)
    for i in range(IM):                       # sequential sum, order as the Fortran
        unp = unp - np.where(mv[i], sinic[i] * vo[i, j, :], 0.0)
        vnp = vnp + np.where(mv[i], cosic[i] * vo[i, j, :], 0.0)
    unp = unp * 2 / IM; vnp = vnp * 2 / IM
    uo[:, JM - 1, :] = unp[None, :] * cosu[:, None] + vnp[None, :] * sinu[:, None]
    vo[:, JM - 1, :] = vnp[None, :] * cosic[:, None] - unp[None, :] * sinic[:, None]


def _fluxes(uo, vo, lmu, lmv, T):
    """Wajsowicz cross-term fluxes (FSLIP=0 branch), arrays (IM, JM, LMO), nonzero for J=1..JM-1 (index 0..JM-2)."""
    g = T['g']; khp, khv, tanp, tanv = T['khp'], T['khv'], T['tanp'], T['tanv']
    bydxp, bydyv, bydyp, bydxv = T['bydxp'], T['bydyv'], T['bydyp'], T['bydxv']
    L = np.arange(1, LMO + 1)[None, None, :]
    mu = lmu[:, :, None] >= L; mv = lmv[:, :, None] >= L
    fux = np.zeros_like(uo); fuy = np.zeros_like(uo); fvx = np.zeros_like(uo); fvy = np.zeros_like(uo)
    for J in range(1, JM):
        j = J - 1                                  # 0-based row of J
        u_jp1, u_j = uo[:, j + 1], uo[:, j]
        m1, m0 = mu[:, j + 1], mu[:, j]
        mim1 = np.roll(m1, 1, axis=0)
        ut = np.where(m1, u_jp1 * tanp[J + 1], 0.0)
        ux = np.where(m1, u_jp1, 0.0)
        uy = np.where(m1, u_jp1, 0.0)
        ut = ut + np.where(m0, u_j * tanp[J], 0.0)
        uy = uy - np.where(m0, u_j, 0.0)
        ux = ux - np.where(mim1, np.roll(u_jp1, 1, axis=0), 0.0)
        ut = 0.5 * ut
        ux = ux * bydxp[J + 1]
        uy = uy * bydyv[J]
        v_j = vo[:, j]
        mj = mv[:, j]
        vt = np.where(mj, v_j * tanv[J], 0.0)
        vx = np.where(mj, v_j, 0.0)
        vy = np.where(mj, v_j, 0.0)
        if J > 1:
            mjm = mv[:, j - 1]
            vt = vt + np.where(mjm, vo[:, j - 1] * tanv[J - 1], 0.0)
            vy = vy - np.where(mjm, vo[:, j - 1], 0.0)
        mim1v = np.roll(mj, 1, axis=0)
        vx = vx - np.where(mim1v, np.roll(v_j, 1, axis=0), 0.0)
        vt = 0.5 * vt
        vy = vy * bydyp[J]
        vx = vx * bydxv[J]
        fuy[:, j] = np.roll(khv[J] * vx, -1, axis=0)         # FUY(IM1,J)
        fvx[:, j] = khv[J] * (uy + ut)
        fux[:, j] = np.roll(khp[J] * (vy + vt), -1, axis=0)  # FUX(IM1,J)
        if J < JM - 1:
            fvy[:, j] = khp[J + 1] * ux
    return fux, fuy, fvx, fvy


def odiff(mo, uo_in, vo_in, dh, lmu, lmv, dtdiff=DTDIFF, tridiag_x=_tridiag_div):
    """ODIFF on one state. mo, dh: (IM, JM, LMO) (dh = DH from the second ODHORZ0 of the step); uo_in, vo_in as the
    state at the call (UO/VO incl. pole row). Returns (uo, vo, vonp) with vonp = UO(IVNP, JM, :) of the input."""
    T = init_odiff(lmu, lmv)
    g = T['g']
    uo = uo_in.copy(); vo = vo_in.copy()
    dt2 = dtdiff * 5e-1
    uonp = uo[IM - 1, JM - 1, :].copy(); vonp = uo[IVNP0, JM - 1, :].copy()
    cosic, sinic, cosu, sinu = trig_tables()
    L = np.arange(1, LMO + 1)[None, None, :]
    mu = lmu[:, :, None] >= L; mv = lmv[:, :, None] >= L
    # pole row from UONP/VONP, then polevel (ODIFF_FIXES_2017) overwrites it from VO(:,JM-1)
    uo[:, JM - 1, :] = uonp[None, :] * cosu[:, None] + vonp[None, :] * sinu[:, None]
    vo[:, JM - 1, :] = vonp[None, :] * cosic[:, None] - uonp[None, :] * sinic[:, None]
    polevel_all(uo, vo, lmv, cosic, sinic, cosu, sinu)

    mo_ip1 = np.roll(mo, -1, axis=0)
    mo_jp1 = np.zeros_like(mo); mo_jp1[:, :JM - 1] = mo[:, 1:]
    dh_ip1 = np.roll(dh, -1, axis=0)
    dh_jp1 = np.zeros_like(dh); dh_jp1[:, :JM - 1] = dh[:, 1:]
    with np.errstate(all='ignore'):
        bymu = np.where(mu, 1.0 / (mo + mo_ip1), 0.0)
        bymv = np.where(mv, 1.0 / (mo + mo_jp1), 0.0)
    dtu = dt2 * (dh + dh_ip1) * bymu
    dtv = dt2 * (dh + dh_jp1) * bymv
    ops = {k: T[k] for k in ('uxa', 'uxb', 'uxc', 'uya', 'uyb', 'uyc', 'vxa', 'vxb', 'vxc', 'vya', 'vyb', 'vyc')}
    dxvo, dypo, dxpo, dyvo = g['dxvo'], g['dypo'], g['dxpo'], g['dyvo']
    bydxypo, bydxyv, tanp, tanv = T['bydxypo'], T['bydxyv'], T['tanp'], T['tanv']
    jr = slice(1, JM - 1)                         # J = 2..JM-1  (0-based rows 1..JM-2)
    J1 = np.arange(2, JM)                         # 1-based J for those rows

    def col(v):                                   # per-J line array -> (1, JM-2, 1)
        return v[J1][None, :, None]

    def col_m1(v):
        return v[J1 - 1][None, :, None]

    def col_p1(v):
        return v[J1 + 1][None, :, None]

    def cross_u(fux, fuy):
        fux_im1 = np.roll(fux, 1, axis=0)
        return (col(dypo) * (fux_im1[:, jr] - fux[:, jr]) + col(dxvo) * fuy[:, jr] - col_m1(dxvo) * fuy[:, 0:JM - 2]) \
            * col(bydxypo) - 0.5 * (col_m1(tanv) * fuy[:, 0:JM - 2] + col(tanv) * fuy[:, jr])

    def cross_v(fvx, fvy):
        fvx_im1 = np.roll(fvx, 1, axis=0)
        return (col(dyvo) * (fvx[:, jr] - fvx_im1[:, jr]) + col(dxpo) * fvy[:, 0:JM - 2] - col_p1(dxpo) * fvy[:, jr]) \
            * col(bydxyv) + 0.5 * (col(tanp) * fvy[:, 0:JM - 2] + col_p1(tanp) * fvy[:, jr])

    # ------------------------------------------------ x sweep (all layers independent)
    fux, fuy, fvx, fvy = _fluxes(uo, vo, lmu, lmv, T)
    sl = lambda a: a[:, jr]                       # noqa: E731
    au = np.zeros_like(uo); bu = np.ones_like(uo); cu = np.zeros_like(uo); ru = np.zeros_like(uo)
    av = np.zeros_like(uo); bv = np.ones_like(uo); cv = np.zeros_like(uo); rv = np.zeros_like(uo)
    mus, mvs = sl(mu), sl(mv)
    DTU, DTV = sl(dtu), sl(dtv)
    uj, ujm, ujp = sl(uo), uo[:, 0:JM - 2], uo[:, 2:JM]
    vj, vjm, vjp = sl(vo), vo[:, 0:JM - 2], vo[:, 2:JM]
    au_ = -DTU * sl(ops['uxa']); bu_ = 1.0 - DTU * sl(ops['uxb']); cu_ = -DTU * sl(ops['uxc'])
    ru_ = uj + DTU * (sl(ops['uya']) * ujm + sl(ops['uyb']) * uj + sl(ops['uyc']) * ujp)
    ru_ = ru_ + DTU * cross_u(fux, fuy)
    av_ = -DTV * sl(ops['vxa']); bv_ = 1.0 - DTV * sl(ops['vxb']); cv_ = -DTV * sl(ops['vxc'])
    rv_ = vj + DTV * (sl(ops['vya']) * vjm + sl(ops['vyb']) * vj + sl(ops['vyc']) * vjp)
    rv_ = rv_ + DTV * cross_v(fvx, fvy)
    au[:, jr] = np.where(mus, au_, 0.0); bu[:, jr] = np.where(mus, bu_, 1.0); cu[:, jr] = np.where(mus, cu_, 0.0)
    ru[:, jr] = np.where(mus, ru_, 0.0)
    av[:, jr] = np.where(mvs, av_, 0.0); bv[:, jr] = np.where(mvs, bv_, 1.0); cv[:, jr] = np.where(mvs, cv_, 0.0)
    rv[:, jr] = np.where(mvs, rv_, 0.0)
    # explicit cyclic terms (ODIFF_TRIDIAG_CYCLIC undefined)
    m0u = mu[0, jr]; m0v = mv[0, jr]
    au[0, jr] = np.where(m0u, 0.0, au[0, jr])
    ru[0, jr] = np.where(m0u, ru[0, jr] + dtu[0, jr] * ops['uxa'][0, jr] * uo[IM - 1, jr], ru[0, jr])
    av[0, jr] = np.where(m0v, 0.0, av[0, jr])
    rv[0, jr] = np.where(m0v, rv[0, jr] + dtv[0, jr] * ops['vxa'][0, jr] * vo[IM - 1, jr], rv[0, jr])
    mLu = mu[IM - 1, jr]; mLv = mv[IM - 1, jr]
    cu[IM - 1, jr] = np.where(mLu, 0.0, cu[IM - 1, jr])
    ru[IM - 1, jr] = np.where(mLu, ru[IM - 1, jr] + dtu[IM - 1, jr] * ops['uxc'][IM - 1, jr] * uo[0, jr], ru[IM - 1, jr])
    cv[IM - 1, jr] = np.where(mLv, 0.0, cv[IM - 1, jr])
    rv[IM - 1, jr] = np.where(mLv, rv[IM - 1, jr] + dtv[IM - 1, jr] * ops['vxc'][IM - 1, jr] * vo[0, jr], rv[IM - 1, jr])
    uo[:, jr] = tridiag_x(au[:, jr], bu[:, jr], cu[:, jr], ru[:, jr])
    vo[:, jr] = tridiag_x(av[:, jr], bv[:, jr], cv[:, jr], rv[:, jr])

    # ------------------------------------------------ y sweep
    fux, fuy, fvx, fvy = _fluxes(uo, vo, lmu, lmv, T)
    uj, ujm, ujp = sl(uo), None, None
    uxa, uxb, uxc = sl(ops['uxa']), sl(ops['uxb']), sl(ops['uxc'])
    vxa, vxb, vxc = sl(ops['vxa']), sl(ops['vxb']), sl(ops['vxc'])
    uim1, uip1 = np.roll(uo, 1, axis=0)[:, jr], np.roll(uo, -1, axis=0)[:, jr]
    vim1, vip1 = np.roll(vo, 1, axis=0)[:, jr], np.roll(vo, -1, axis=0)[:, jr]
    vj = sl(vo)
    au3 = -DTU * sl(ops['uya']); bu3 = 1.0 - DTU * sl(ops['uyb']); cu3 = -DTU * sl(ops['uyc'])
    cu3[:, JM - 3] = 0.0
    ru3 = uj + DTU * (uxa * uim1 + uxb * uj + uxc * uip1)
    ru3[:, JM - 3] = ru3[:, JM - 3] + DTU[:, JM - 3] * sl(ops['uyc'])[:, JM - 3] * uo[:, JM - 1]
    ru3 = ru3 + DTU * cross_u(fux, fuy)
    av3 = -DTV * sl(ops['vya']); bv3 = 1.0 - DTV * sl(ops['vyb']); cv3 = -DTV * sl(ops['vyc'])
    cv3[:, JM - 3] = 0.0
    rv3 = vj + DTV * (vxa * vim1 + vxb * vj + vxc * vip1)
    rv3[:, JM - 3] = rv3[:, JM - 3] + DTV[:, JM - 3] * sl(ops['vyc'])[:, JM - 3] * vo[:, JM - 1]
    rv3 = rv3 + DTV * cross_v(fvx, fvy)
    au3 = np.where(mus, au3, 0.0); bu3 = np.where(mus, bu3, 1.0); cu3 = np.where(mus, cu3, 0.0); ru3 = np.where(mus, ru3, 0.0)
    av3 = np.where(mvs, av3, 0.0); bv3 = np.where(mvs, bv3, 1.0); cv3 = np.where(mvs, cv3, 0.0); rv3 = np.where(mvs, rv3, 0.0)
    # TRIDIAG_new runs along J: move J to the leading axis
    mv_ax = lambda a: np.moveaxis(a, 1, 0)        # noqa: E731
    uu = _tridiag_recip(mv_ax(au3), mv_ax(bu3), mv_ax(cu3), mv_ax(ru3))
    vv = _tridiag_recip(mv_ax(av3), mv_ax(bv3), mv_ax(cv3), mv_ax(rv3))
    uo[:, jr] = np.moveaxis(uu, 0, 1); vo[:, jr] = np.moveaxis(vv, 0, 1)
    uo[IM - 1, JM - 1, :] = uonp
    uo[IVNP0, JM - 1, :] = vonp
    return uo, vo, vonp.copy()
