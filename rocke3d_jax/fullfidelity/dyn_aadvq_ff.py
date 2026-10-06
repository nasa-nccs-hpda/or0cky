"""Full-fidelity numpy port of the moisture-advection family of QUS3D.f (D103-D105) and the QDYNAM glue of ATMDYN.f (D105/D106).

Ported (ModelE2_planet_2.0/model/QUS3D.f): AADVQ0 292-829 (cycle-count preparation), XSTEP 835-893, ZSTEP 896-928,
AADVQ 53-289 (cycle loop, per-level Y then X sweeps, vertical interface sweep with the per-level carry, extra
vertical column advection), aadvqx 978-1234, aadvqy 1236-1509, aadvqz 1511-1670, checkflux 1672-1688,
aadvqz_column 2180-2301; ATMDYN.f QDYNAM 3039-3101 (mass-unit conversion, call sequence, back conversion).
Dead for this rundeck and NOT ported: the unlimited *2 variants (qlimit is always .TRUE. from QDYNAM), the
halo/pack/unpack code (serial run: both poles local, halo updates are no-ops), the tracer-only TrDYNAM.

Array conventions (Fortran index order, 0-based): rm, mass, mu, mv, mw (IM,JM,LM); rmom (9,IM,JM,LM).  Serial run:
J_0=1, J_1=JM, J_0S=2, J_1S=JM-1, J_0H=1, J_1H=JM (Fortran numbering).  After AADVQ0 `mv(:,j)` holds the flux through
the NORTH edge of row j (AADVQ0 shifts MVs by one row and keeps the original south edge in pv_south).

Style: the 1-D flux/moment algebra of X, Y and Z is the same expression tree written once (`face_flux`, `cell_update`)
with the direction permutation as data; the three sweeps are vectorised over rows/columns, which is exact because
every face flux is computed from the pre-update state of its two neighbour cells (checked against literal scalar
transcriptions `aadvqx_rowloop`, `aadvqy_loop`, `aadvqz_loop` in the tests).  AADVQ0 follows the Fortran control flow
literally (the cycle loops are data dependent).  Single-precision literal traps replicated: `byn = 1./ncyc` and
`byNXY = 1./ncycxy(l)` are computed in REAL*4 and then promoted; `mrat_limy = 0.20` is a REAL*4 literal stored in a
REAL*8 parameter (0.20000000298023224); `fracm**3` is np.power (ifort calls pow; x*x*x differs in a few cells).
Reductions (polar sums, per-row flux diagnostics) are left to right (np.cumsum), as ifort -fp-model strict does.
"""
import numpy as np

IM, JM, LM = 72, 46, 40
NMOM = 9
MX, MY, MZ, MXX, MYY, MZZ, MXY, MZX, MYZ = range(9)
IHMOMS = (MX, MY, MXX, MYY, MXY, MYZ, MZX)
# direction specs: (p, pp, a, ap, b, bp, plain1, plain2, plain3)
XSPEC = (MX, MXX, MY, MXY, MZ, MZX, MYY, MZZ, MYZ)
YSPEC = (MY, MYY, MZ, MYZ, MX, MXY, MZZ, MXX, MZX)
ZSPEC = (MZ, MZZ, MY, MYZ, MX, MZX, MYY, MXX, MXY)

NCMAX = 10
NSTEPMAX_X = 60
MRAT_LIMH = 0.25
MRAT_LIMY = float(np.float32(0.20))          # real*8, parameter :: mrat_limy=0.20 (REAL*4 literal)
F32 = np.float32


def inv32(n):
    """REAL*8 variable assigned `1./n` with n integer: the quotient is formed in REAL*4."""
    return float(F32(1.0) / F32(n))


def ordered_sum(a, axis=-1):
    a = np.moveaxis(np.asarray(a), axis, -1)
    return np.cumsum(a, axis=-1)[..., -1]


def seq_add(acc, xs):
    """acc + xs[0] + xs[1] + ... strictly left to right (acc, xs[..., k]); acc shape = xs.shape[:-1]."""
    st = np.concatenate([np.asarray(acc)[..., None], xs], axis=-1)
    return np.cumsum(st, axis=-1)[..., -1]


class QusError(RuntimeError):
    """The Fortran calls stop_model at these points."""


# ------------------------------------------------------------------------------------------------ checkflux
def checkflux(aml, amr, m, rm, rxm, rxxm):
    """QUS3D.f:1672-1688, elementwise (arrays); returns (rxm, rxxm, fired_mask)."""
    a = amr / m
    fr = a * (rm + (1. - a) * (rxm + (1. - 2. * a) * rxxm))
    a = aml / m
    fl = a * (rm - (1. + a) * (rxm - (1. + 2. * a) * rxxm))
    fire = (rm + fl - fr) <= 0.
    return np.where(fire, 0., rxm), np.where(fire, 0., rxxm), fire


# ------------------------------------------------------------------------------------------------ XSTEP / ZSTEP
def xstep_rows(m, mu, nmax=NSTEPMAX_X):
    """XSTEP (QUS3D.f:835-893) for a batch of rows.  m, mu: (B, IM).  Returns (nstep (B,), mi (B, IM), ierr).

    Per row: smallest nstep with courmax<=1 over nstep sub-steps of am=mu/nstep; mi is the mass after those
    nstep sub-steps (updated unconditionally, also after the last).  nstep reaching 60 sets ierr=1 and returns."""
    B, im = m.shape
    nstep = np.zeros(B, dtype=int)
    mi_out = m.copy()
    done = np.zeros(B, dtype=bool)
    ierr = 0
    with np.errstate(all='ignore'):
        for trial in range(1, nmax + 1):
            idx = np.nonzero(~done)[0]
            if idx.size == 0:
                break
            if trial == nmax:
                ierr = 1
                nstep[idx] = trial
                break
            nstep[idx] = trial
            am = mu[idx] / trial
            mi = m[idx].copy()
            cm = np.zeros(idx.size)
            for ns in range(trial):
                mip1 = np.roll(mi, -1, axis=1)
                cand = np.where(am > 0., am / mi, -am / mip1)
                cm = np.maximum(cm, np.max(cand, axis=1))
                mi = (mi + np.roll(am, 1, axis=1)) - am
            ok = ~(cm > 1.)
            mi_out[idx[ok]] = mi[ok]
            done[idx[ok]] = True
    return nstep, mi_out, ierr


def xstep_row_loop(m, mu, nmax=NSTEPMAX_X):
    """Literal scalar transcription of XSTEP for one row (testing reference).  Returns (nstep, mi, ierr)."""
    im = m.size
    nstep = 0
    courmax = 2.
    mi = np.array(m, dtype=float)
    while courmax > 1.:
        nstep += 1
        if nstep == nmax:
            return nstep, mi, 1
        am = np.array([mu[i] / nstep for i in range(im)])
        mi = np.array(m, dtype=float)
        courmax = 0.
        for ns in range(nstep):
            if am[0] > 0.:
                courmax = max(courmax, +am[0] / mi[0])
            else:
                courmax = max(courmax, -am[0] / mi[1])
            i = im - 1
            if am[i] > 0.:
                courmax = max(courmax, +am[i] / mi[i])
            else:
                courmax = max(courmax, -am[i] / mi[0])
            for i in range(1, im - 1):
                if am[i] > 0.:
                    courmax = max(courmax, +am[i] / mi[i])
                else:
                    courmax = max(courmax, -am[i] / mi[i + 1])
                mi[i] = mi[i] + am[i - 1] - am[i]
            mi[0] = mi[0] + am[im - 1] - am[0]
            mi[im - 1] = mi[im - 1] + am[im - 2] - am[im - 1]
    return nstep, mi, 0


def zstep(m0, cm0):
    """ZSTEP (QUS3D.f:896-928): returns (nstep, m0_updated).  m0, cm0: 1-D arrays of length nl (cm0[nl-1] unused)."""
    nl = m0.size
    m0 = np.array(m0, dtype=float)
    nstep = 0
    done = False
    while not done:
        nstep += 1
        if nstep > 200000:      # Fortran has no cap (int32 wrap-around on degenerate input, e.g. a layer emptied to exactly 0)
            raise QusError('ZSTEP: no convergence')
        cm = cm0[:nl - 1] / nstep
        ml = m0.copy()
        done = True
        for ns in range(1, nstep + 1):
            bad = False
            for l in range(nl - 1):
                if (ml[l] - cm[l]) * (ml[l + 1] + cm[l]) < 0.:
                    bad = True
                    break
            if bad:
                done = False
                break
            if ns < nstep:
                ml[:nl - 1] = ml[:nl - 1] - cm
                ml[1:nl] = ml[1:nl] + cm
    m0[:nl - 1] = m0[:nl - 1] - cm0[:nl - 1]
    m0[1:nl] = m0[1:nl] + cm0[:nl - 1]
    return nstep, m0


# ------------------------------------------------------------------------------------------------ AADVQ0
def aadvq0(mu, mv, mw, mb, imaxj, byim, stats=None, nmax=NCMAX):
    """AADVQ0 (QUS3D.f:292-829), serial.  mu,mv,mw,mb (IM,JM,LM) are NOT modified; returns a dict with the AADVQ0
    outputs: ncyc, ncycxy(LM), nstepx(JM,LM), nstepz_extra(IM,JM), lminzij, lmaxzij (Fortran 1-based values),
    do_z_extra, mw_extra, mu, mv, mw (scaled / z-extra-subtracted), pv_south(IM,LM), ni_y, ni_z (JM,LM) and the
    lists i_y, i_z (dict (j,l) -> list of 1-based i).  `stats`: optional dict of branch counters."""
    def hit(k, n=1):
        if stats is not None:
            stats[k] = stats.get(k, 0) + n
    im, jm, lm = IM, JM, LM
    mu = mu.copy(); mv = mv.copy(); mw = mw.copy()
    # ---- shift of MV, polar zeroing (QUS3D.f:332-345)
    pv_south = mv[:, 0, :].copy()
    mu[:, 0, :] = 0.
    mv[:, :jm - 1, :] = mv[:, 1:, :].copy()
    mu[:, jm - 1, :] = 0.
    mv[:, jm - 1, :] = 0.
    mw[:, :, lm - 1] = 0.
    # ---- horizontal-vertical ncyc (QUS3D.f:347-446)
    nbad = 1
    ncyc = 0
    lminzij = np.full((im, jm), lm + 1, dtype=int)
    lmaxzij = np.zeros((im, jm), dtype=int)
    byn = None
    while nbad > 0:
        ncyc += 1
        hit('ncyc_trials')
        if ncyc > nmax:
            raise QusError('AADVQ0: ncyc>ncmax')
        byn = inv32(ncyc)
        nbad_loc = 0
        mma = mb.copy()
        lminzij = np.full((im, jm), lm + 1, dtype=int)
        lmaxzij = np.zeros((im, jm), dtype=int)
        for nc in range(1, ncyc + 1):
            for l in range(lm):
                # non-polar rows j=2..jm-1
                muw = np.roll(mu[:, 1:jm - 1, l], 1, axis=0)
                d = ((muw - mu[:, 1:jm - 1, l]) + mv[:, 0:jm - 2, l]) - mv[:, 1:jm - 1, l]
                mma[:, 1:jm - 1, l] = mma[:, 1:jm - 1, l] + byn * d
                nbad_loc += int(np.sum(mma[:, 1:jm - 1, l] < MRAT_LIMH * mb[:, 1:jm - 1, l]))
                # south pole
                ssp = ordered_sum(mma[:, 0, l] - mv[:, 0, l] * byn) * byim
                mma[:, 0, l] = ssp
                if mma[0, 0, l] < MRAT_LIMH * mb[0, 0, l]:
                    nbad_loc += 1
                snp = ordered_sum(mma[:, jm - 1, l] + mv[:, jm - 2, l] * byn) * byim
                mma[:, jm - 1, l] = snp
                if mma[0, jm - 1, l] < MRAT_LIMH * mb[0, jm - 1, l]:
                    nbad_loc += 1
            nbad = nbad_loc
            if nbad > 0:
                hit('ncyc_horizontal_fail')
                break
            # z courant check and mass update from z fluxes (QUS3D.f:408-444)
            mwbyn = -(mw * byn)                              # mwbyn(l) = -mw(l)*byn  (l<lm)
            viol = (mma[:, :, :lm - 1] - mwbyn[:, :, :lm - 1]) * (mma[:, :, 1:] + mwbyn[:, :, :lm - 1]) < 0.
            for l in range(lm - 1):
                v = viol[:, :, l]
                lminzij = np.where(v, np.minimum(lminzij, l + 1), lminzij)
                lmaxzij = np.where(v, np.maximum(lmaxzij, l + 2), lmaxzij)
            if nc < ncyc:
                new = mma.copy()
                new[:, :, 0] = mma[:, :, 0] + mw[:, :, 0] * byn
                new[:, :, lm - 1] = mma[:, :, lm - 1] - mw[:, :, lm - 2] * byn
                new[:, :, 1:lm - 1] = mma[:, :, 1:lm - 1] + (mw[:, :, 1:lm - 1] - mw[:, :, 0:lm - 2]) * byn
                mma = new
    hit('ncyc_%d' % ncyc)
    # ---- divide the horizontal mass fluxes by ncyc (QUS3D.f:449-457)
    if ncyc > 1:
        byn = inv32(ncyc)
        pv_south = pv_south * byn
        mu = mu * byn
        mv = mv * byn
    # ---- nstepz_extra and the partition of the vertical flux (QUS3D.f:459-554)
    do_z_extra = False
    nstepz_extra = np.zeros((im, jm), dtype=int)
    mw_extra = np.zeros((im, jm, lm))
    for j in range(jm):
        for i in range(im):
            if lminzij[i, j] >= lm:
                continue
            im1 = i - 1 if i > 0 else im - 1
            do_z_extra = True
            lmin, lmax = int(lminzij[i, j]), int(lmaxzij[i, j])          # 1-based
            nl = lmax - lmin + 1
            div1d = np.empty(nl)
            for l in range(lmin, lmax + 1):
                if j == 0:
                    mvjm1 = 0. if l > 1 else np.nan                          # out-of-bounds read mv(i,0,l) = mv(i,JM,l-1) = 0
                else:
                    mvjm1 = mv[i, j - 1, l - 1]
                div1d[l - lmin] = ((mu[im1, j, l - 1] - mu[i, j, l - 1]) + mvjm1) - mv[i, j, l - 1]
            mb1d = mb[i, j, lmin - 1:lmax].copy()
            mw1d = -(mw[i, j, lmin - 1:lmax] * byn)
            ma1d = mb1d + div1d
            mamin = ma1d.copy()
            for nc3d in range(2, ncyc + 1):
                ma1d[:nl - 1] = ma1d[:nl - 1] - mw1d[:nl - 1]
                ma1d[1:nl] = ma1d[1:nl] + mw1d[:nl - 1]
                if lmin > 1:
                    ma1d[0] = ma1d[0] - mw[i, j, lmin - 2] * byn
                if lmax < lm:
                    ma1d[nl - 1] = ma1d[nl - 1] + mw[i, j, lmax - 1] * byn
                ma1d = ma1d + div1d
                mamin = np.minimum(ma1d, mamin)
            for l in range(nl - 1):
                if mw1d[l] > 0.:
                    mwlim = min(mw1d[l], +mamin[l])
                else:
                    mwlim = max(mw1d[l], -mamin[l + 1])
                div1d[l] = div1d[l] - mwlim
                div1d[l + 1] = div1d[l + 1] + mwlim
                mw1d[l] = mw1d[l] - mwlim
                mw_extra[i, j, l + lmin - 1] = -(mw1d[l] * ncyc)
            if lmin > 1:
                div1d[0] = div1d[0] - mw[i, j, lmin - 2] * byn
            if lmax < lm:
                div1d[nl - 1] = div1d[nl - 1] + mw[i, j, lmax - 1] * byn
            ma1d = mb1d.copy()
            for nc3d in range(1, ncyc + 1):
                ma1d = ma1d + div1d
                nsd, ma1d = zstep(ma1d, mw1d)
                nstepz_extra[i, j] = max(nstepz_extra[i, j], nsd)
    hit('z_extra_columns', int(np.sum(nstepz_extra > 0)))
    # ---- ncycxy per level (QUS3D.f:556-700)
    ncycxy = np.ones(lm, dtype=int)
    ma2d = np.zeros((im, jm))
    mb2d = np.zeros((im, jm))
    for l in range(lm):
        ncycxy[l] = 1
        broke = False
        for nc3d in range(1, ncyc + 1):
            ma2d = mb[:, :, l].copy() if nc3d == 1 else mb2d.copy()
            nbad = 1
            while nbad > 0:
                if ncycxy[l] > nmax:
                    broke = True
                    break
                byNXY = inv32(int(ncycxy[l]))
                nbad = 0
                for nc in range(1, ncycxy[l] + 1):
                    # y courant numbers
                    mvbyn = mv[:, 1:jm - 2, l] * byNXY
                    nbad += int(np.sum((ma2d[:, 1:jm - 2] - mvbyn) * (ma2d[:, 2:jm - 1] + mvbyn) < 0.))
                    mpol = ma2d[0, 0] * im
                    mvbyn = mv[:, 0, l] * byNXY
                    nbad += int(np.sum((mpol - mvbyn) * (ma2d[:, 1] + mvbyn) < 0.))
                    mpol = ma2d[0, jm - 1] * im
                    mvbyn = mv[:, jm - 2, l] * byNXY
                    nbad += int(np.sum((ma2d[:, jm - 2] - mvbyn) * (mpol + mvbyn) < 0.))
                    # y update with mass-ratio check, then x update (rows 2..jm-1)
                    ma2d[:, 1:jm - 1] = ma2d[:, 1:jm - 1] + (mv[:, 0:jm - 2, l] - mv[:, 1:jm - 1, l]) * byNXY
                    nbad += int(np.sum(ma2d[:, 1:jm - 1] < MRAT_LIMY * mb[:, 1:jm - 1, l]))
                    ma2d[:, 1:jm - 1] = ma2d[:, 1:jm - 1] + (np.roll(mu[:, 1:jm - 1, l], 1, axis=0) - mu[:, 1:jm - 1, l]) * byNXY
                    ssp = ordered_sum(ma2d[:, 0] - mv[:, 0, l] * byNXY) * byim
                    ma2d[:, 0] = ssp
                    if ma2d[0, 0] < MRAT_LIMY * mb[0, 0, l]:
                        nbad += 1
                    snp = ordered_sum(ma2d[:, jm - 1] + mv[:, jm - 2, l] * byNXY) * byim
                    ma2d[:, jm - 1] = snp
                    if ma2d[0, jm - 1] < MRAT_LIMY * mb[0, jm - 1, l]:
                        nbad += 1
                    if nbad > 0:
                        ncycxy[l] += 1
                        hit('ncycxy_increment')
                        ma2d = mb[:, :, l].copy() if nc3d == 1 else mb2d.copy()
                        break
            if broke:
                break
            if nc3d < ncyc:
                if l == 0:
                    mb2d = ma2d + mw[:, :, l] * byn
                elif l == lm - 1:
                    mb2d = ma2d - mw[:, :, l - 1] * byn
                else:
                    mb2d = ma2d + (mw[:, :, l] - mw[:, :, l - 1]) * byn
        # (the Fortran exits nc3dloop when ncycxy>ncmax; the stop follows below)
    # ---- globalmax no-op; scaling of the xy fluxes by ncycxy, nstepx via XSTEP (QUS3D.f:704-760)
    nstepx = np.zeros((jm, lm), dtype=int)
    ierr = 0
    for l in range(lm):
        if ncycxy[l] > nmax:
            raise QusError('AADVQ0: ncycxy>ncmax')
        if ncycxy[l] > 1:
            byNXY = inv32(int(ncycxy[l]))
            pv_south[:, l] = pv_south[:, l] * byNXY
            mu[:, :, l] = mu[:, :, l] * byNXY
            mv[:, :, l] = mv[:, :, l] * byNXY
        ma2d = mb[:, :, l].copy()
        nstepx[1:jm - 1, l] = 1
        for nc3d in range(1, ncyc + 1):
            for nc in range(1, ncycxy[l] + 1):
                ma2d[:, 1:jm - 1] = ma2d[:, 1:jm - 1] + (mv[:, 0:jm - 2, l] - mv[:, 1:jm - 1, l])
                nsd, mi, ie = xstep_rows(ma2d[:, 1:jm - 1].T.copy(), mu[:, 1:jm - 1, l].T.copy())
                ierr += ie
                nstepx[1:jm - 1, l] = np.maximum(nstepx[1:jm - 1, l], nsd)
                ma2d[:, 1:jm - 1] = mi.T
            if nc3d < ncyc:
                if l == 0:
                    ma2d[:, 1:jm - 1] = ma2d[:, 1:jm - 1] + mw[:, 1:jm - 1, l] * byn
                elif l == lm - 1:
                    ma2d[:, 1:jm - 1] = ma2d[:, 1:jm - 1] - mw[:, 1:jm - 1, l - 1] * byn
                else:
                    ma2d[:, 1:jm - 1] = ma2d[:, 1:jm - 1] + (mw[:, 1:jm - 1, l] - mw[:, 1:jm - 1, l - 1]) * byn
    if ierr > 0:
        raise QusError('too many steps in xstep')
    # ---- subtract mw_extra from mw (QUS3D.f:785-799)
    if do_z_extra:
        for j in range(jm):
            for i in range(im):
                if nstepz_extra[i, j] > 0:
                    lmin = int(lminzij[i, j]); lmax = int(lmaxzij[i, j]) - 1
                    for l in range(lmin, lmax + 1):
                        mw[i, j, l - 1] = mw[i, j, l - 1] - mw_extra[i, j, l - 1]
    # ---- flow-out-both-sides tables (QUS3D.f:838-866); upwind-halo tables are serial no-ops
    ni_z = np.zeros((jm, lm), dtype=int)
    ni_y = np.zeros((jm, lm), dtype=int)
    i_z, i_y = {}, {}
    for l in range(2, lm):                                            # Fortran l=2..lm-1
        for j in range(jm):
            lst = [i + 1 for i in range(int(imaxj[j])) if (mw[i, j, l - 2] > 0 and mw[i, j, l - 1] < 0)]
            ni_z[j, l - 1] = len(lst)
            if lst:
                i_z[(j + 1, l)] = lst
    for l in range(1, lm + 1):
        for j in range(2, jm):                                         # Fortran j=2..jm-1
            lst = [i + 1 for i in range(im) if (mv[i, j - 1, l - 1] > 0. and mv[i, j - 2, l - 1] < 0.)]
            ni_y[j - 1, l - 1] = len(lst)
            if lst:
                i_y[(j, l)] = lst
    hit('checkfobs_y_cells', int(ni_y.sum())); hit('checkfobs_z_cells', int(ni_z.sum()))
    return dict(ncyc=ncyc, ncycxy=ncycxy, nstepx=nstepx, nstepz_extra=nstepz_extra, lminzij=lminzij, lmaxzij=lmaxzij,
                do_z_extra=do_z_extra, mw_extra=mw_extra, mu=mu, mv=mv, mw=mw, pv_south=pv_south, ni_y=ni_y, ni_z=ni_z,
                i_y=i_y, i_z=i_z)


# ------------------------------------------------------------------------------------------------ flux kernels
def face_flux(a, mL, rmL, momL, mR, rmR, momR, rm_left, rm_right, spec, strict_neg=True):
    """Flux through a face (QUS3D.f aadvqx/aadvqy/aadvqz, same expression tree), elementwise on arrays.

    a: face mass flux; mL/rmL/momL(9,..) the cell on the low side, mR/rmR/momR the high side (upwind cell chosen by
    the sign of a: negative flux -> high side).  rm_left/rm_right: tracer mass of the two cells used by the flux
    limiter.  Returns dict: fe (limited tracer flux), fe0, fe_pass, fmom (9,...) east-role moments (after the
    negative-branch limits), fp_pass, fpp_pass (primary moments handed to the downstream cell)."""
    p, pp, aa, aap, bb, bbp, q1, q2, q3 = spec
    neg = a < 0.
    mass = np.where(neg, mR, mL)
    rmu = np.where(neg, rmR, rmL)
    mom = np.where(neg[None], momR, momL)
    frac1 = np.where(neg, 1., -1.)
    with np.errstate(all='ignore'):
        fracm = a / mass
        frac1 = fracm + frac1
        mp, mpp = mom[p], mom[pp]
        fe = fracm * (rmu - frac1 * (mp - (frac1 + fracm) * mpp))
        fm = np.empty_like(mom)
        fm[p] = a * (fracm * fracm * (mp - 3. * frac1 * mpp) - 3. * fe)
        fm[pp] = a * (a * np.power(fracm, 3.) * mpp - 5. * (a * fe + fm[p]))
        fm[aa] = fracm * (mom[aa] - frac1 * mom[aap])
        fm[aap] = a * (fracm * fracm * mom[aap] - 3. * fm[aa])
        fm[bb] = fracm * (mom[bb] - frac1 * mom[bbp])
        fm[bbp] = a * (fracm * fracm * mom[bbp] - 3. * fm[bb])
        fm[q1] = fracm * mom[q1]
        fm[q2] = fracm * mom[q2]
        fm[q3] = fracm * mom[q3]
        # flux limitations
        fe0 = fe.copy()
        fe_pass = fe.copy()
        fpp_ = fm[p].copy()
        fppp = fm[pp].copy()
        pos = a > 0.
        negm = neg if strict_neg else ~pos
        c1 = pos & (fe < 0.)
        c2 = pos & ~(fe < 0.) & (fe > rm_left)
        c3 = negm & (fe > 0.)
        c4 = negm & ~(fe > 0.) & (fe < -rm_right)
        fe_orig = fe
        fe = np.where(c1, 0., fe)
        fe_pass = np.where(c1, 0., fe_pass)
        fpp_ = np.where(c1, 0., fpp_)
        fppp = np.where(c1, 0., fppp)
        fe = np.where(c2, rm_left, fe)
        fe_pass = np.where(c2, rm_left, fe_pass)
        fpp_ = np.where(c2, a * (-3. * rm_left), fpp_)
        fppp = np.where(c2, a * (-5. * (a * rm_left + (a * (-3. * rm_left)))), fppp)
        fe = np.where(c3, 0., fe)
        fe0 = np.where(c3, 0., fe0)
        fm[p] = np.where(c3, 0., fm[p])
        fm[pp] = np.where(c3, 0., fm[pp])
        fe = np.where(c4, -rm_right, fe)
        fe0 = np.where(c4, -rm_right, fe0)
        fp4 = a * (-3. * (-rm_right))
        fm[p] = np.where(c4, fp4, fm[p])
        fm[pp] = np.where(c4, a * (-5. * (a * (-rm_right) + fp4)), fm[pp])
    return dict(fe=fe, fe0=fe0, fe_pass=fe_pass, fmom=fm, fp_pass=fpp_, fpp_pass=fppp, c=(c1, c2, c3, c4))


def cell_update(rm, mom, mass, aw, fw, fw0, fmw, ae, fe, fe0, fme, spec):
    """The shared cell update of aadvqx/aadvqy/aadvqz (west/east = low/high side faces).  fmw: west-role moments
    (9,...) with the primary moments already replaced by the pass-through values.  Returns (rm, mom, mass)."""
    p, pp, aa, aap, bb, bbp, q1, q2, q3 = spec
    with np.errstate(all='ignore'):
        mold = mass
        mnew = mold + aw - ae
        bymnew = 1. / mnew
        dm2 = aw + ae
        rm0 = rm + fw0 - fe0
        rmn = rm + fw - fe
        new = mom.copy()
        new[p] = (mom[p] * mold - 3. * (-dm2 * rm0 + mold * (fw0 + fe0)) + (fmw[p] - fme[p])) * bymnew
        new[pp] = (mom[pp] * mold * mold + 2.5 * rm0 * (mold * mold - mnew * mnew - 3. * dm2 * dm2)
                   + 5. * (mold * (mold * (fw0 - fe0) - fmw[p] - fme[p]) + dm2 * new[p] * mnew)
                   + (fmw[pp] - fme[pp])) * (bymnew * bymnew)
        new[aa] = mom[aa] + fmw[aa] - fme[aa]
        new[aap] = (mom[aap] * mold - 3. * (-dm2 * new[aa] + mold * (fmw[aa] + fme[aa])) + (fmw[aap] - fme[aap])) * bymnew
        new[bb] = mom[bb] + fmw[bb] - fme[bb]
        new[bbp] = (mom[bbp] * mold - 3. * (-dm2 * new[bb] + mold * (fmw[bb] + fme[bb])) + (fmw[bbp] - fme[bbp])) * bymnew
        for q in (q1, q2, q3):
            new[q] = mom[q] + fmw[q] - fme[q]
        clean = rmn <= 0.
        rmn = np.where(clean, 0., rmn)
        new = np.where(clean[None], 0., new)
    return rmn, new, mnew


def _west_moments(f, spec):
    fmw = f['fmom'].copy()
    fmw[spec[0]] = f['fp_pass']
    fmw[spec[1]] = f['fpp_pass']
    return fmw


# ------------------------------------------------------------------------------------------------ aadvqx
def aadvqx(rm, rmom, mass, mu, nstep, stats=None, jmin=1, jmax=JM - 2):
    """aadvqx (QUS3D.f:978-1234) for one level.  rm, mass, mu: (IM,JM); rmom (9,IM,JM); nstep (JM,) (rows jmin..jmax,
    0-based).  Rows are independent; every row runs exactly nstep(j) sub-steps with am=mu/nstep.  Returns new
    (rm, rmom, mass) copies."""
    rm = rm.copy(); rmom = rmom.copy(); mass = mass.copy()
    nmax = int(np.max(nstep[jmin:jmax + 1]))
    for ns in range(1, nmax + 1):
        rows = [j for j in range(jmin, jmax + 1) if nstep[j] >= ns]
        if not rows:
            continue
        rows = np.array(rows)
        am = mu[:, rows] / nstep[rows][None, :].astype(float)           # (IM, R)
        m_ = mass[:, rows]; r_ = rm[:, rows]; mom_ = rmom[:, :, rows]
        # checkflux where flow leaves the cell on both sides
        amw = np.roll(am, 1, axis=0)
        sel = (am > 0.) & (amw < 0.)
        if stats is not None:
            stats['x_checkflux_cells'] = stats.get('x_checkflux_cells', 0) + int(sel.sum())
        rxm, rxxm, fire = checkflux(amw, am, m_, r_, mom_[MX], mom_[MXX])
        if stats is not None:
            stats['x_checkflux_fired'] = stats.get('x_checkflux_fired', 0) + int((sel & fire).sum())
        mom_ = mom_.copy()
        mom_[MX] = np.where(sel, rxm, mom_[MX]); mom_[MXX] = np.where(sel, rxxm, mom_[MXX])
        # face i = east face of cell i (between i and i+1, periodic)
        f = face_flux(am, m_, r_, mom_, np.roll(m_, -1, axis=0), np.roll(r_, -1, axis=0), np.roll(mom_, -1, axis=1),
                      r_, np.roll(r_, -1, axis=0), XSPEC, strict_neg=False)
        if stats is not None:
            for k, c in zip(('x_lim_pos_neg', 'x_lim_pos_gt_rm', 'x_lim_neg_pos', 'x_lim_neg_lt_rm'), f['c']):
                stats[k] = stats.get(k, 0) + int(c.sum())
        fmw = _west_moments(f, XSPEC)
        rmn, momn, massn = cell_update(r_, mom_, m_, amw, np.roll(f['fe'], 1, axis=0), np.roll(f['fe_pass'], 1, axis=0),
                                       np.roll(fmw, 1, axis=1), am, f['fe'], f['fe0'], f['fmom'], XSPEC)
        rm[:, rows] = rmn; rmom[:, :, rows] = momn; mass[:, rows] = massn
    return rm, rmom, mass


def aadvqx_rowloop(rm, rmom, mass, mu_row, nstep, j):
    """Literal scalar transcription of aadvqx for ONE row j (0-based); testing reference.  Returns new copies of
    the full (rm (IM,JM), rmom, mass)."""
    rm = rm.copy(); rmom = rmom.copy(); mass = mass.copy()
    im = IM
    am = mu_row / nstep
    for ns in range(nstep):
        for i in range(im):
            iw = i - 1 if i > 0 else im - 1
            if am[i] > 0. and am[iw] < 0.:
                a = am[i] / mass[i, j]
                fr = a * (rm[i, j] + (1. - a) * (rmom[MX, i, j] + (1. - 2. * a) * rmom[MXX, i, j]))
                a = am[iw] / mass[i, j]
                fl = a * (rm[i, j] - (1. + a) * (rmom[MX, i, j] - (1. + 2. * a) * rmom[MXX, i, j]))
                if rm[i, j] + fl - fr <= 0.:
                    rmom[MX, i, j] = 0.; rmom[MXX, i, j] = 0.

        def flux(i):
            if am[i] < 0.:
                ii = (i + 1) % im; frac1 = 1.
            else:
                ii = i; frac1 = -1.
            fracm = am[i] / mass[ii, j]
            frac1 = fracm + frac1
            fe = fracm * (rm[ii, j] - frac1 * (rmom[MX, ii, j] - (frac1 + fracm) * rmom[MXX, ii, j]))
            fm = np.zeros(9)
            fm[MX] = am[i] * (fracm * fracm * (rmom[MX, ii, j] - 3. * frac1 * rmom[MXX, ii, j]) - 3. * fe)
            fm[MXX] = am[i] * (am[i] * np.power(fracm, 3.) * rmom[MXX, ii, j] - 5. * (am[i] * fe + fm[MX]))
            fm[MY] = fracm * (rmom[MY, ii, j] - frac1 * rmom[MXY, ii, j])
            fm[MXY] = am[i] * (fracm * fracm * rmom[MXY, ii, j] - 3. * fm[MY])
            fm[MZ] = fracm * (rmom[MZ, ii, j] - frac1 * rmom[MZX, ii, j])
            fm[MZX] = am[i] * (fracm * fracm * rmom[MZX, ii, j] - 3. * fm[MZ])
            fm[MYY] = fracm * rmom[MYY, ii, j]
            fm[MZZ] = fracm * rmom[MZZ, ii, j]
            fm[MYZ] = fracm * rmom[MYZ, ii, j]
            fe0 = fe; fe_pass = fe; fex_pass = fm[MX]; fexx_pass = fm[MXX]
            if am[i] > 0.:
                if fe < 0.:
                    fe = 0.; fe_pass = 0.; fex_pass = 0.; fexx_pass = 0.
                elif fe > rm[i, j]:
                    fe = rm[i, j]; fe_pass = fe
                    fex_pass = am[i] * (-3. * fe)
                    fexx_pass = am[i] * (-5. * (am[i] * fe + fex_pass))
            else:
                if fe > 0.:
                    fe = 0.; fe0 = 0.; fm[MX] = 0.; fm[MXX] = 0.
                elif fe < -rm[(i + 1) % im, j]:
                    fe = -rm[(i + 1) % im, j]; fe0 = fe
                    fm[MX] = am[i] * (-3. * fe)
                    fm[MXX] = am[i] * (-5. * (am[i] * fe + fm[MX]))
            return fe, fe0, fe_pass, fm, fex_pass, fexx_pass
        i = im - 1
        fe, fe0, fe_pass, fm, fex_pass, fexx_pass = flux(i)
        feim, feim0, fmomeim = fe, fe0, fm.copy()
        amw = am[im - 1]; fw = fe; fw0 = fe_pass; fmomw = fm.copy(); fmomw[MX] = fex_pass; fmomw[MXX] = fexx_pass

        def update(i, fe, fe0, fm):
            mold = mass[i, j]
            mnew = mold + amw - am[i]
            bymnew = 1. / mnew
            dm2 = amw + am[i]
            rm0 = rm[i, j] + fw0 - fe0
            rm[i, j] = rm[i, j] + fw - fe
            r = rmom[:, i, j]
            r[MX] = (r[MX] * mold - 3. * (-dm2 * rm0 + mold * (fw0 + fe0)) + (fmomw[MX] - fm[MX])) * bymnew
            r[MXX] = (r[MXX] * mold * mold + 2.5 * rm0 * (mold * mold - mnew * mnew - 3. * dm2 * dm2)
                      + 5. * (mold * (mold * (fw0 - fe0) - fmomw[MX] - fm[MX]) + dm2 * r[MX] * mnew)
                      + (fmomw[MXX] - fm[MXX])) * (bymnew * bymnew)
            r[MY] = r[MY] + fmomw[MY] - fm[MY]
            r[MXY] = (r[MXY] * mold - 3. * (-dm2 * r[MY] + mold * (fmomw[MY] + fm[MY])) + (fmomw[MXY] - fm[MXY])) * bymnew
            r[MZ] = r[MZ] + fmomw[MZ] - fm[MZ]
            r[MZX] = (r[MZX] * mold - 3. * (-dm2 * r[MZ] + mold * (fmomw[MZ] + fm[MZ])) + (fmomw[MZX] - fm[MZX])) * bymnew
            r[MYY] = r[MYY] + fmomw[MYY] - fm[MYY]
            r[MZZ] = r[MZZ] + fmomw[MZZ] - fm[MZZ]
            r[MYZ] = r[MYZ] + fmomw[MYZ] - fm[MYZ]
            mass[i, j] = mnew
            if rm[i, j] <= 0.:
                rm[i, j] = 0.; rmom[:, i, j] = 0.
        for i in range(im - 1):
            fe, fe0, fe_pass, fm, fex_pass, fexx_pass = flux(i)
            update(i, fe, fe0, fm)
            amw = am[i]; fw = fe; fmomw = fm.copy(); fw0 = fe_pass; fmomw[MX] = fex_pass; fmomw[MXX] = fexx_pass
        update(im - 1, feim, feim0, fmomeim)
    return rm, rmom, mass


# ------------------------------------------------------------------------------------------------ aadvqy
def aadvqy(rm, rmom, mass, mv, byim, sbf, sbm, sfbm, stats=None):
    """aadvqy (QUS3D.f:1236-1509) for one level.  rm, mass, mv (IM,JM) (mv(:,j) = north-edge flux of row j), rmom
    (9,IM,JM).  sbf, sbm, sfbm (JM,) are the running diagnostics (returned updated).  Returns (rm, rmom, mass,
    sbf, sbm, sfbm)."""
    rm = rm.copy(); rmom = rmom.copy(); mass = mass.copy()
    sbf = sbf.copy(); sbm = sbm.copy(); sfbm = sfbm.copy()
    im, jm = IM, JM
    # polar boxes scaled to full extent
    m_sp = mass[0, 0] * im; rm_sp = rm[0, 0] * im
    rzm_sp = rmom[MZ, 0, 0] * im; rzzm_sp = rmom[MZZ, 0, 0] * im
    for i in range(im):
        if mv[i, 0] < 0.:
            continue
        mass[i, 0] = m_sp; rm[i, 0] = rm_sp
        rmom[MZ, i, 0] = rzm_sp; rmom[MZZ, i, 0] = rzzm_sp
        for q in IHMOMS:
            rmom[q, i, 0] = 0.
    m_np = mass[0, jm - 1] * im; rm_np = rm[0, jm - 1] * im
    rzm_np = rmom[MZ, 0, jm - 1] * im; rzzm_np = rmom[MZZ, 0, jm - 1] * im
    for i in range(im):
        if mv[i, jm - 2] >= 0.:
            continue
        mass[i, jm - 1] = m_np; rm[i, jm - 1] = rm_np
        rmom[MZ, i, jm - 1] = rzm_np; rmom[MZZ, i, jm - 1] = rzzm_np
        for q in IHMOMS:
            rmom[q, i, jm - 1] = 0.
    # faces j = 0..jm-2 (face j is the north edge of row j): all from the pre-update state
    mvf = mv[:, :jm - 1]
    f = face_flux(mvf, mass[:, :jm - 1], rm[:, :jm - 1], rmom[:, :, :jm - 1], mass[:, 1:], rm[:, 1:], rmom[:, :, 1:],
                  rm[:, :jm - 1], rm[:, 1:], YSPEC, strict_neg=True)
    # the south-edge face of the SP cap (face 0) uses the 'else' branch of the Fortran limiter (mv<=0)
    f0 = face_flux(mvf[:, :1], mass[:, :1], rm[:, :1], rmom[:, :, :1], mass[:, 1:2], rm[:, 1:2], rmom[:, :, 1:2],
                   rm[:, :1], rm[:, 1:2], YSPEC, strict_neg=False)
    if stats is not None:
        for k, c in zip(('y_lim_pos_neg', 'y_lim_pos_gt_rm', 'y_lim_neg_pos', 'y_lim_neg_lt_rm'), f['c']):
            stats[k] = stats.get(k, 0) + int(c[:, 1:].sum())
    fe = f['fe'].copy(); fe0 = f['fe0'].copy(); fe_pass = f['fe_pass'].copy(); fmom = f['fmom'].copy()
    fp_pass = f['fp_pass'].copy(); fpp_pass = f['fpp_pass'].copy()
    for dst, src in ((fe, f0['fe']), (fe0, f0['fe0']), (fe_pass, f0['fe_pass']), (fp_pass, f0['fp_pass']),
                     (fpp_pass, f0['fpp_pass'])):
        dst[:, :1] = src
    fmom[:, :, :1] = f0['fmom']
    fmw_all = fmom.copy()
    fmw_all[MY] = fp_pass; fmw_all[MYY] = fpp_pass
    # south polar cap (uses face 0 values)
    m_sp = m_sp - ordered_sum(mvf[:, 0])
    rm_sp = rm_sp - ordered_sum(fe[:, 0])
    rzm_sp = np.subtract.accumulate(np.concatenate([[rzm_sp], fmom[MZ, :, 0]]))[-1]
    rzzm_sp = np.subtract.accumulate(np.concatenate([[rzzm_sp], fmom[MZZ, :, 0]]))[-1]
    sbf[0] = seq_add(sbf[0], fe[:, 0]); sbm[0] = seq_add(sbm[0], mv[:, 0])
    nz = mv[:, 0] != 0.
    with np.errstate(all='ignore'):
        sfbm[0] = seq_add(sfbm[0], np.where(nz, fe[:, 0] / mv[:, 0], 0.))
    mass[0, 0] = m_sp * byim; rm[0, 0] = rm_sp * byim
    rmom[MZ, 0, 0] = rzm_sp * byim; rmom[MZZ, 0, 0] = rzzm_sp * byim
    for q in IHMOMS:
        rmom[q, 0, 0] = 0.
    if rm[0, 0] <= 0.:
        rm[0, 0] = 0.; rmom[:, 0, 0] = 0.
    # non-polar rows 1..jm-2 (0-based): west face = j-1, east face = j
    js = slice(1, jm - 1)
    rmn, momn, massn = cell_update(rm[:, js], rmom[:, :, js], mass[:, js], mvf[:, 0:jm - 2], fe[:, 0:jm - 2],
                                   fe_pass[:, 0:jm - 2], fmw_all[:, :, 0:jm - 2], mvf[:, 1:jm - 1], fe[:, 1:jm - 1],
                                   fe0[:, 1:jm - 1], fmom[:, :, 1:jm - 1], YSPEC)
    # per-row accumulations (left to right over i)
    sbf[js] = seq_add(sbf[js], fe[:, 1:jm - 1].T)
    sbm[js] = seq_add(sbm[js], mv[:, 1:jm - 1].T)
    nzr = mv[:, 1:jm - 1] != 0.
    with np.errstate(all='ignore'):
        sfbm[js] = seq_add(sfbm[js], np.where(nzr, fe[:, 1:jm - 1] / mv[:, 1:jm - 1], 0.).T)
    # north polar cap: uses the last face (jm-2) values, as left in mvj/fs/fmoms by the Fortran loop
    m_np = m_np + ordered_sum(mvf[:, jm - 2])
    rm_np = rm_np + ordered_sum(fe[:, jm - 2])
    rzm_np = np.add.accumulate(np.concatenate([[rzm_np], fmom[MZ, :, jm - 2]]))[-1]
    rzzm_np = np.add.accumulate(np.concatenate([[rzzm_np], fmom[MZZ, :, jm - 2]]))[-1]
    rm[:, js] = rmn; rmom[:, :, js] = momn; mass[:, js] = massn
    mass[0, jm - 1] = m_np * byim; rm[0, jm - 1] = rm_np * byim
    rmom[MZ, 0, jm - 1] = rzm_np * byim; rmom[MZZ, 0, jm - 1] = rzzm_np * byim
    for q in IHMOMS:
        rmom[q, 0, jm - 1] = 0.
    if rm[0, jm - 1] <= 0.:
        rm[0, jm - 1] = 0.; rmom[:, 0, jm - 1] = 0.
    return rm, rmom, mass, sbf, sbm, sfbm


def aadvqy_loop(rm, rmom, mass, mv, byim, sbf, sbm, sfbm):
    """Literal scalar transcription of aadvqy (testing reference): same signature/returns as aadvqy."""
    rm = rm.copy(); rmom = rmom.copy(); mass = mass.copy()
    sbf = sbf.copy(); sbm = sbm.copy(); sfbm = sfbm.copy()
    im, jm = IM, JM
    m_sp = mass[0, 0] * im; rm_sp = rm[0, 0] * im; rzm_sp = rmom[MZ, 0, 0] * im; rzzm_sp = rmom[MZZ, 0, 0] * im
    for i in range(im):
        if mv[i, 0] < 0.:
            continue
        mass[i, 0] = m_sp; rm[i, 0] = rm_sp; rmom[MZ, i, 0] = rzm_sp; rmom[MZZ, i, 0] = rzzm_sp
        for q in IHMOMS:
            rmom[q, i, 0] = 0.
    m_np = mass[0, jm - 1] * im; rm_np = rm[0, jm - 1] * im; rzm_np = rmom[MZ, 0, jm - 1] * im; rzzm_np = rmom[MZZ, 0, jm - 1] * im
    for i in range(im):
        if mv[i, jm - 2] >= 0.:
            continue
        mass[i, jm - 1] = m_np; rm[i, jm - 1] = rm_np; rmom[MZ, i, jm - 1] = rzm_np; rmom[MZZ, i, jm - 1] = rzzm_np
        for q in IHMOMS:
            rmom[q, i, jm - 1] = 0.
    mvj = np.zeros(im); fs = np.zeros(im); fs0 = np.zeros(im); fmoms = np.zeros((9, im))

    def fluxes(i, j, polar):
        if mv[i, j] < 0.:
            jj = j + 1; frac1 = 1.
        else:
            jj = j; frac1 = -1.
        v = mv[i, j]
        fracm = v / mass[i, jj]
        frac1 = fracm + frac1
        fn = fracm * (rm[i, jj] - frac1 * (rmom[MY, i, jj] - (frac1 + fracm) * rmom[MYY, i, jj]))
        fm = np.zeros(9)
        fm[MY] = v * (fracm * fracm * (rmom[MY, i, jj] - 3. * frac1 * rmom[MYY, i, jj]) - 3. * fn)
        fm[MYY] = v * (v * np.power(fracm, 3.) * rmom[MYY, i, jj] - 5. * (v * fn + fm[MY]))
        fm[MZ] = fracm * (rmom[MZ, i, jj] - frac1 * rmom[MYZ, i, jj])
        fm[MYZ] = v * (fracm * fracm * rmom[MYZ, i, jj] - 3. * fm[MZ])
        fm[MX] = fracm * (rmom[MX, i, jj] - frac1 * rmom[MXY, i, jj])
        fm[MXY] = v * (fracm * fracm * rmom[MXY, i, jj] - 3. * fm[MX])
        fm[MZZ] = fracm * rmom[MZZ, i, jj]; fm[MXX] = fracm * rmom[MXX, i, jj]; fm[MZX] = fracm * rmom[MZX, i, jj]
        fn0 = fn; fn_pass = fn; fny = fm[MY]; fnyy = fm[MYY]
        if v > 0.:
            if fn < 0.:
                fn = 0.; fn_pass = 0.; fny = 0.; fnyy = 0.
            elif fn > rm[i, j]:
                fn = rm[i, j]; fn_pass = fn; fny = v * (-3. * fn); fnyy = v * (-5. * (v * fn + fny))
        elif (v < 0.) or polar:
            if fn > 0.:
                fn = 0.; fn0 = 0.; fm[MY] = 0.; fm[MYY] = 0.
            elif fn < -rm[i, j + 1]:
                fn = -rm[i, j + 1]; fn0 = fn; fm[MY] = v * (-3. * fn); fm[MYY] = v * (-5. * (v * fn + fm[MY]))
        return fn, fn0, fn_pass, fm, fny, fnyy
    # south edge of the SP cap
    for i in range(im):
        fn, fn0, fn_pass, fm, fny, fnyy = fluxes(i, 0, True)
        mvj[i] = mv[i, 0]; fs[i] = fn; fs0[i] = fn_pass; fmoms[:, i] = fm
        fmoms[MY, i] = fny; fmoms[MYY, i] = fnyy
    m_sp = m_sp - ordered_sum(mvj); rm_sp = rm_sp - ordered_sum(fs)
    for i in range(im):
        rzm_sp = rzm_sp - fmoms[MZ, i]; rzzm_sp = rzzm_sp - fmoms[MZZ, i]
        sbf[0] = sbf[0] + fs[i]; sbm[0] = sbm[0] + mv[i, 0]
        if mv[i, 0] != 0.:
            sfbm[0] = sfbm[0] + fs[i] / mv[i, 0]
    mass[0, 0] = m_sp * byim; rm[0, 0] = rm_sp * byim; rmom[MZ, 0, 0] = rzm_sp * byim; rmom[MZZ, 0, 0] = rzzm_sp * byim
    for q in IHMOMS:
        rmom[q, 0, 0] = 0.
    if rm[0, 0] <= 0.:
        rm[0, 0] = 0.; rmom[:, 0, 0] = 0.
    for j in range(1, jm - 1):
        for i in range(im):
            fn, fn0, fn_pass, fmomn, fny, fnyy = fluxes(i, j, False)
            v = mv[i, j]
            mold = mass[i, j]; mnew = mold + mvj[i] - v; bymnew = 1. / mnew; dm2 = mvj[i] + v
            rm0 = rm[i, j] + fs0[i] - fn0
            rm[i, j] = rm[i, j] + fs[i] - fn
            r = rmom[:, i, j]
            r[MY] = (r[MY] * mold - 3. * (-dm2 * rm0 + mold * (fs0[i] + fn0)) + (fmoms[MY, i] - fmomn[MY])) * bymnew
            r[MYY] = (r[MYY] * mold * mold + 2.5 * rm0 * (mold * mold - mnew * mnew - 3. * dm2 * dm2)
                      + 5. * (mold * (mold * (fs0[i] - fn0) - fmoms[MY, i] - fmomn[MY]) + dm2 * r[MY] * mnew)
                      + (fmoms[MYY, i] - fmomn[MYY])) * (bymnew * bymnew)
            r[MZ] = r[MZ] + fmoms[MZ, i] - fmomn[MZ]
            r[MYZ] = (r[MYZ] * mold - 3. * (-dm2 * r[MZ] + mold * (fmoms[MZ, i] + fmomn[MZ])) + (fmoms[MYZ, i] - fmomn[MYZ])) * bymnew
            r[MX] = r[MX] + fmoms[MX, i] - fmomn[MX]
            r[MXY] = (r[MXY] * mold - 3. * (-dm2 * r[MX] + mold * (fmoms[MX, i] + fmomn[MX])) + (fmoms[MXY, i] - fmomn[MXY])) * bymnew
            r[MZZ] = r[MZZ] + fmoms[MZZ, i] - fmomn[MZZ]
            r[MXX] = r[MXX] + fmoms[MXX, i] - fmomn[MXX]
            r[MZX] = r[MZX] + fmoms[MZX, i] - fmomn[MZX]
            mass[i, j] = mnew
            if rm[i, j] <= 0.:
                rm[i, j] = 0.; rmom[:, i, j] = 0.
            mvj[i] = v; fs[i] = fn; fmoms[:, i] = fmomn; fs0[i] = fn_pass
            fmoms[MY, i] = fny; fmoms[MYY, i] = fnyy
            sbf[j] = sbf[j] + fn; sbm[j] = sbm[j] + v
            if v != 0.:
                sfbm[j] = sfbm[j] + fn / v
    m_np = m_np + ordered_sum(mvj); rm_np = rm_np + ordered_sum(fs)
    for i in range(im):
        rzm_np = rzm_np + fmoms[MZ, i]; rzzm_np = rzzm_np + fmoms[MZZ, i]
    mass[0, jm - 1] = m_np * byim; rm[0, jm - 1] = rm_np * byim; rmom[MZ, 0, jm - 1] = rzm_np * byim; rmom[MZZ, 0, jm - 1] = rzzm_np * byim
    for q in IHMOMS:
        rmom[q, 0, jm - 1] = 0.
    if rm[0, jm - 1] <= 0.:
        rm[0, jm - 1] = 0.; rmom[:, 0, jm - 1] = 0.
    return rm, rmom, mass, sbf, sbm, sfbm


# ------------------------------------------------------------------------------------------------ aadvqz
def aadvqz(rm1, rmom1, mass1, rm2, rmom2, mass2, mw, mwdn, fdn, fdn0, fmomdn, imaxj, scf, scm, sfcm, stats=None):
    """aadvqz (QUS3D.f:1511-1670): one interface sweep.  Layer 1 = l-1 (updated), layer 2 = l (read only).
    rm* (IM,JM), rmom* (9,IM,JM), mw (IM,JM) is the interface flux sd(:,:,l-1); the carry (mwdn, fdn, fdn0,
    fmomdn (9,IM,JM)) is returned updated; scf/scm/sfcm (JM,) running diagnostics.  Only i < imaxj(j) is
    advected (poles: i=1) and the pole rows are then filled.  Returns dict."""
    rm1 = rm1.copy(); rmom1 = rmom1.copy(); mass1 = mass1.copy()
    im, jm = IM, JM
    act = np.arange(im)[:, None] < np.asarray(imaxj)[None, :]            # (IM, JM)
    f = face_flux(mw, mass1, rm1, rmom1, mass2, rm2, rmom2, rm1, rm2, ZSPEC, strict_neg=True)
    if stats is not None:
        for k, c in zip(('z_lim_pos_neg', 'z_lim_pos_gt_rm', 'z_lim_neg_pos', 'z_lim_neg_lt_rm'), f['c']):
            stats[k] = stats.get(k, 0) + int((c & act).sum())
    rmn, momn, massn = cell_update(rm1, rmom1, mass1, mwdn, fdn, fdn0, fmomdn, mw, f['fe'], f['fe0'], f['fmom'], ZSPEC)
    rm1 = np.where(act, rmn, rm1); rmom1 = np.where(act[None], momn, rmom1); mass1 = np.where(act, massn, mass1)
    fmw = f['fmom'].copy(); fmw[MZ] = f['fp_pass']; fmw[MZZ] = f['fpp_pass']
    mwdn2 = np.where(act, mw, mwdn); fdn_new = np.where(act, f['fe'], fdn)
    fdn0_new = np.where(act, f['fe_pass'], fdn0)
    fmomdn_new = np.where(act[None], fmw, fmomdn)
    scf = seq_add(scf, np.where(act, f['fe'], 0.).T)
    scm = seq_add(scm, np.where(act, mw, 0.).T)
    with np.errstate(all='ignore'):
        scr = np.where(act & (mw != 0.), f['fe'] / mw, 0.)
    sfcm = seq_add(sfcm, scr.T)
    for j in (0, jm - 1):
        mass1[1:, j] = mass1[0, j]; rm1[1:, j] = rm1[0, j]; rmom1[:, 1:, j] = rmom1[:, 0:1, j]
    return dict(rm=rm1, rmom=rmom1, mass=mass1, mwdn=mwdn2, fdn=fdn_new, fdn0=fdn0_new, fmomdn=fmomdn_new,
                scf=scf, scm=scm, sfcm=sfcm)


def aadvqz_loop(rm1, rmom1, mass1, rm2, rmom2, mass2, mw, mwdn, fdn, fdn0, fmomdn, imaxj, scf, scm, sfcm):
    """Literal scalar transcription of aadvqz (testing reference)."""
    rm1 = rm1.copy(); rmom1 = rmom1.copy(); mass1 = mass1.copy()
    mwdn = mwdn.copy(); fdn = fdn.copy(); fdn0 = fdn0.copy(); fmomdn = fmomdn.copy()
    scf = scf.copy(); scm = scm.copy(); sfcm = sfcm.copy()
    im, jm = IM, JM
    for j in range(jm):
        for i in range(int(imaxj[j])):
            if mw[i, j] < 0.:
                m_ = mass2[i, j]; r_ = rm2[i, j]; mom = rmom2[:, i, j]; frac1 = 1.
            else:
                m_ = mass1[i, j]; r_ = rm1[i, j]; mom = rmom1[:, i, j]; frac1 = -1.
            w = mw[i, j]
            fracm = w / m_
            frac1 = fracm + frac1
            fup = fracm * (r_ - frac1 * (mom[MZ] - (frac1 + fracm) * mom[MZZ]))
            fm = np.zeros(9)
            fm[MZ] = w * (fracm * fracm * (mom[MZ] - 3. * frac1 * mom[MZZ]) - 3. * fup)
            fm[MZZ] = w * (w * np.power(fracm, 3.) * mom[MZZ] - 5. * (w * fup + fm[MZ]))
            fm[MY] = fracm * (mom[MY] - frac1 * mom[MYZ])
            fm[MYZ] = w * (fracm * fracm * mom[MYZ] - 3. * fm[MY])
            fm[MX] = fracm * (mom[MX] - frac1 * mom[MZX])
            fm[MZX] = w * (fracm * fracm * mom[MZX] - 3. * fm[MX])
            fm[MYY] = fracm * mom[MYY]; fm[MXX] = fracm * mom[MXX]; fm[MXY] = fracm * mom[MXY]
            fup0 = fup; fup_pass = fup; fz_pass = fm[MZ]; fzz_pass = fm[MZZ]
            if w > 0.:
                if fup < 0.:
                    fup = 0.; fup_pass = 0.; fz_pass = 0.; fzz_pass = 0.
                elif fup > rm1[i, j]:
                    fup = rm1[i, j]; fup_pass = fup; fz_pass = w * (-3. * fup); fzz_pass = w * (-5. * (w * fup + fz_pass))
            elif w < 0.:
                if fup > 0.:
                    fup = 0.; fup0 = 0.; fm[MZ] = 0.; fm[MZZ] = 0.
                elif fup < -rm2[i, j]:
                    fup = -rm2[i, j]; fup0 = fup; fm[MZ] = w * (-3. * fup); fm[MZZ] = w * (-5. * (w * fup + fm[MZ]))
            mold = mass1[i, j]; mnew = mold + mwdn[i, j] - w; bymnew = 1. / mnew; dm2 = mwdn[i, j] + w
            rm0 = rm1[i, j] + fdn0[i, j] - fup0
            rm1[i, j] = rm1[i, j] + fdn[i, j] - fup
            r = rmom1[:, i, j]; fd = fmomdn[:, i, j]
            r[MZ] = (r[MZ] * mold - 3. * (-dm2 * rm0 + mold * (fdn0[i, j] + fup0)) + (fd[MZ] - fm[MZ])) * bymnew
            r[MZZ] = (r[MZZ] * mold * mold + 2.5 * rm0 * (mold * mold - mnew * mnew - 3. * dm2 * dm2)
                      + 5. * (mold * (mold * (fdn0[i, j] - fup0) - fd[MZ] - fm[MZ]) + dm2 * r[MZ] * mnew)
                      + (fd[MZZ] - fm[MZZ])) * (bymnew * bymnew)
            r[MY] = r[MY] + fd[MY] - fm[MY]
            r[MYZ] = (r[MYZ] * mold - 3. * (-dm2 * r[MY] + mold * (fd[MY] + fm[MY])) + (fd[MYZ] - fm[MYZ])) * bymnew
            r[MX] = r[MX] + fd[MX] - fm[MX]
            r[MZX] = (r[MZX] * mold - 3. * (-dm2 * r[MX] + mold * (fd[MX] + fm[MX])) + (fd[MZX] - fm[MZX])) * bymnew
            r[MYY] = r[MYY] + fd[MYY] - fm[MYY]
            r[MXX] = r[MXX] + fd[MXX] - fm[MXX]
            r[MXY] = r[MXY] + fd[MXY] - fm[MXY]
            mass1[i, j] = mnew
            if rm1[i, j] <= 0.:
                rm1[i, j] = 0.; rmom1[:, i, j] = 0.
            mwdn[i, j] = w; fdn[i, j] = fup; fmomdn[:, i, j] = fm
            fdn0[i, j] = fup_pass; fmomdn[MZ, i, j] = fz_pass; fmomdn[MZZ, i, j] = fzz_pass
            scf[j] = scf[j] + fup; scm[j] = scm[j] + w
            if w != 0.:
                sfcm[j] = sfcm[j] + fup / w
    for j in (0, jm - 1):
        for i in range(1, im):
            mass1[i, j] = mass1[0, j]; rm1[i, j] = rm1[0, j]; rmom1[:, i, j] = rmom1[:, 0, j]
    return dict(rm=rm1, rmom=rmom1, mass=mass1, mwdn=mwdn, fdn=fdn, fdn0=fdn0, fmomdn=fmomdn, scf=scf, scm=scm, sfcm=sfcm)


def aadvqz_column(rm, rmom, mass, mw):
    """aadvqz_column (QUS3D.f:2180-2301): one pass of the limited vertical advection of ONE column of nl layers.
    rm, mass, mw: (nl,), rmom (9,nl); mw(nl) must be 0 (set by the caller).  Returns new (rm, rmom, mass)."""
    nl = rm.size
    rm = rm.copy(); rmom = rmom.copy(); mass = mass.copy()
    mwdn = 0.; fdn = 0.; fdn0 = 0.; fmomdn = np.zeros(9)
    for l in range(nl):
        w = mw[l]
        if w < 0.:
            ll = l + 1; frac1 = 1.
        else:
            ll = l; frac1 = -1.
        fracm = w / mass[ll]
        frac1 = fracm + frac1
        mom = rmom[:, ll]
        fup = fracm * (rm[ll] - frac1 * (mom[MZ] - (frac1 + fracm) * mom[MZZ]))
        fm = np.zeros(9)
        fm[MZ] = w * (fracm * fracm * (mom[MZ] - 3. * frac1 * mom[MZZ]) - 3. * fup)
        fm[MZZ] = w * (w * np.power(fracm, 3.) * mom[MZZ] - 5. * (w * fup + fm[MZ]))
        fm[MY] = fracm * (mom[MY] - frac1 * mom[MYZ])
        fm[MYZ] = w * (fracm * fracm * mom[MYZ] - 3. * fm[MY])
        fm[MX] = fracm * (mom[MX] - frac1 * mom[MZX])
        fm[MZX] = w * (fracm * fracm * mom[MZX] - 3. * fm[MX])
        fm[MYY] = fracm * mom[MYY]; fm[MXX] = fracm * mom[MXX]; fm[MXY] = fracm * mom[MXY]
        fup0 = fup; fup_pass = fup; fz_pass = fm[MZ]; fzz_pass = fm[MZZ]
        if w > 0.:
            if fup < 0.:
                fup = 0.; fup_pass = 0.; fz_pass = 0.; fzz_pass = 0.
            elif fup > rm[l]:
                fup = rm[l]; fup_pass = fup; fz_pass = w * (-3. * fup); fzz_pass = w * (-5. * (w * fup + fz_pass))
        elif w < 0.:
            if fup > 0.:
                fup = 0.; fup0 = 0.; fm[MZ] = 0.; fm[MZZ] = 0.
            elif fup < -rm[l + 1]:
                fup = -rm[l + 1]; fup0 = fup; fm[MZ] = w * (-3. * fup); fm[MZZ] = w * (-5. * (w * fup + fm[MZ]))
        mold = mass[l]; mnew = mold + mwdn - w; bymnew = 1. / mnew; dm2 = mwdn + w
        rm0 = rm[l] + fdn0 - fup0
        rm[l] = rm[l] + fdn - fup
        r = rmom[:, l]; fd = fmomdn
        r[MZ] = (r[MZ] * mold - 3. * (-dm2 * rm0 + mold * (fdn0 + fup0)) + (fd[MZ] - fm[MZ])) * bymnew
        r[MZZ] = (r[MZZ] * mold * mold + 2.5 * rm0 * (mold * mold - mnew * mnew - 3. * dm2 * dm2)
                  + 5. * (mold * (mold * (fdn0 - fup0) - fd[MZ] - fm[MZ]) + dm2 * r[MZ] * mnew)
                  + (fd[MZZ] - fm[MZZ])) * (bymnew * bymnew)
        r[MY] = r[MY] + fd[MY] - fm[MY]
        r[MYZ] = (r[MYZ] * mold - 3. * (-dm2 * r[MY] + mold * (fd[MY] + fm[MY])) + (fd[MYZ] - fm[MYZ])) * bymnew
        r[MX] = r[MX] + fd[MX] - fm[MX]
        r[MZX] = (r[MZX] * mold - 3. * (-dm2 * r[MX] + mold * (fd[MX] + fm[MX])) + (fd[MZX] - fm[MZX])) * bymnew
        r[MYY] = r[MYY] + fd[MYY] - fm[MYY]
        r[MXX] = r[MXX] + fd[MXX] - fm[MXX]
        r[MXY] = r[MXY] + fd[MXY] - fm[MXY]
        mass[l] = mnew
        if rm[l] <= 0.:
            rm[l] = 0.; rmom[:, l] = 0.
        mwdn = w; fdn = fup; fmomdn = fm.copy()
        fdn0 = fup_pass; fmomdn[MZ] = fz_pass; fmomdn[MZZ] = fzz_pass
    return rm, rmom, mass


# ------------------------------------------------------------------------------------------------ AADVQ
def aadvq(rm, rmom, mb, q0, imaxj, byim, stats=None, hooks=None):
    """AADVQ (QUS3D.f:53-289) with qlimit=.TRUE. and tname='q'.  rm (IM,JM,LM), rmom (9,IM,JM,LM) in MASS units,
    mb the mass at entry; q0 = aadvq0 output dict.  hooks: optional callback(stage, nc, ncxy, l, payload) called at
    the instrumented points (stages as in dyn_qdynam_io).  Returns dict(rm, rmom, mma, sbf..sfcm (JM,LM), scf3d)."""
    im, jm, lm = IM, JM, LM
    rm = rm.copy(); rmom = rmom.copy()
    mma = mb.copy()
    scf3d = np.zeros((im, jm, lm))
    ncyc = q0['ncyc']; ncycxy = q0['ncycxy']
    pu, pv, mws = q0['mu'], q0['mv'], q0['mw']
    sbf = np.zeros((jm, lm)); sbm = np.zeros((jm, lm)); sfbm = np.zeros((jm, lm))
    scf = np.zeros((jm, lm)); scm = np.zeros((jm, lm)); sfcm = np.zeros((jm, lm))
    byNCYC = 1. / ncyc
    sd = -(mws * byNCYC)
    ly = {}
    for (j, l), lst in q0['i_y'].items():
        ly[(j, l)] = np.array(lst) - 1
    lz = {}
    for (j, l), lst in q0['i_z'].items():
        lz[(j, l)] = np.array(lst) - 1
    do_z = q0['do_z_extra']

    def hk(stage, nc, ncxy, l, payload):
        if hooks is not None:
            hooks(stage, nc, ncxy, l, payload)
    for nc in range(1, ncyc + 1):
        mwdn = np.zeros((im, jm)); fdn = np.zeros((im, jm)); fdn0 = np.zeros((im, jm)); fmomdn = np.zeros((9, im, jm))
        for L in range(1, lm + 2):
            if L <= lm:
                l = L - 1
                for ncxy in range(1, int(ncycxy[l]) + 1):
                    # y checkflux (rows 2..jm-1)
                    for j in range(2, jm):
                        ii = ly.get((j, L))
                        if ii is None:
                            continue
                        a, b, fire = checkflux(pv[ii, j - 2, l], pv[ii, j - 1, l], mma[ii, j - 1, l], rm[ii, j - 1, l],
                                               rmom[MY, ii, j - 1, l], rmom[MYY, ii, j - 1, l])
                        if stats is not None:
                            stats['y_checkflux_cells'] = stats.get('y_checkflux_cells', 0) + len(ii)
                            stats['y_checkflux_fired'] = stats.get('y_checkflux_fired', 0) + int(fire.sum())
                        rmom[MY, ii, j - 1, l] = a; rmom[MYY, ii, j - 1, l] = b
                    hk(0, nc, ncxy, L, (rmom[MY, :, :, l], rmom[MYY, :, :, l]))
                    r, m, ma, sbf[:, l], sbm[:, l], sfbm[:, l] = aadvqy(rm[:, :, l], rmom[:, :, :, l], mma[:, :, l], pv[:, :, l], byim,
                                                                        sbf[:, l], sbm[:, l], sfbm[:, l], stats)
                    rm[:, :, l] = r; rmom[:, :, :, l] = m; mma[:, :, l] = ma
                    hk(1, nc, ncxy, L, (r, m, ma))
                    r, m, ma = aadvqx(rm[:, :, l], rmom[:, :, :, l], mma[:, :, l], pu[:, :, l], q0['nstepx'][:, l], stats)
                    rm[:, :, l] = r; rmom[:, :, :, l] = m; mma[:, :, l] = ma
                    hk(2, nc, ncxy, L, (r, m, ma))
            if 1 < L < lm:
                for j in range(1, jm + 1):
                    ii = lz.get((j, L))
                    if ii is None:
                        continue
                    a, b, fire = checkflux(sd[ii, j - 1, L - 2], sd[ii, j - 1, L - 1], mma[ii, j - 1, L - 1],
                                           rm[ii, j - 1, L - 1], rmom[MZ, ii, j - 1, L - 1], rmom[MZZ, ii, j - 1, L - 1])
                    if stats is not None:
                        stats['z_checkflux_cells'] = stats.get('z_checkflux_cells', 0) + len(ii)
                        stats['z_checkflux_fired'] = stats.get('z_checkflux_fired', 0) + int(fire.sum())
                    rmom[MZ, ii, j - 1, L - 1] = a; rmom[MZZ, ii, j - 1, L - 1] = b
                hk(3, nc, 0, L, (rmom[MZ, :, :, L - 1], rmom[MZZ, :, :, L - 1]))
            if L > 1:
                l1 = L - 2
                l2 = min(L - 1, lm - 1)                                 # layer 2 (read only; unused when L=lm+1: mw=0)
                z = aadvqz(rm[:, :, l1], rmom[:, :, :, l1], mma[:, :, l1], rm[:, :, l2], rmom[:, :, :, l2],
                           mma[:, :, l2], sd[:, :, l1], mwdn, fdn, fdn0, fmomdn, imaxj, scf[:, l1], scm[:, l1],
                           sfcm[:, l1], stats)
                rm[:, :, l1] = z['rm']; rmom[:, :, :, l1] = z['rmom']; mma[:, :, l1] = z['mass']
                mwdn, fdn, fdn0, fmomdn = z['mwdn'], z['fdn'], z['fdn0'], z['fmomdn']
                scf[:, l1], scm[:, l1], sfcm[:, l1] = z['scf'], z['scm'], z['sfcm']
                scf3d[:, :, l1] = scf3d[:, :, l1] + fdn
                hk(4, nc, 0, L, (mwdn, fdn, fdn0, fmomdn))
        hk(6, nc, 0, 0, (rm, rmom, mma))
        if do_z:
            for j in range(jm):
                for i in range(im):
                    ns_ = int(q0['nstepz_extra'][i, j])
                    if ns_ == 0:
                        continue
                    lmin = int(q0['lminzij'][i, j]); lmax = int(q0['lmaxzij'][i, j]); nl = lmax - lmin + 1
                    mw1d = np.zeros(nl)
                    mw1d[:nl - 1] = -q0['mw_extra'][i, j, lmin - 1:lmax - 1] / (ncyc * ns_)
                    ma1d = mma[i, j, lmin - 1:lmax].copy()
                    rm1d = rm[i, j, lmin - 1:lmax].copy()
                    rmom1d = rmom[:, i, j, lmin - 1:lmax].copy()
                    for istep in range(ns_):
                        rm1d, rmom1d, ma1d = aadvqz_column(rm1d, rmom1d, ma1d, mw1d)
                    mma[i, j, lmin - 1:lmax] = ma1d; rm[i, j, lmin - 1:lmax] = rm1d; rmom[:, i, j, lmin - 1:lmax] = rmom1d
            hk(7, nc, 0, 0, (rm, rmom, mma))
    return dict(rm=rm, rmom=rmom, mma=mma, sbf=sbf, sbm=sbm, sfbm=sfbm, scf=scf, scm=scm, sfcm=sfcm, scf3d=scf3d)


# ------------------------------------------------------------------------------------------------ QDYNAM
def qdynam(q, qmom, maold, mus, mvs, mws, axyp, imaxj, kg2mb, byim_geom, byim_qus, stats=None, hooks=None):
    """QDYNAM (ATMDYN.f:3039-3101): q (IM,JM,LM), qmom (9,IM,JM,LM) concentration units; maold (LM,IM,JM) as in
    ATM_COM; mus, mvs, mws (IM,JM,LM).  Returns dict(q, qmom, mb, q0, ain (rm, rmom), aad (AADVQ output))."""
    mb = np.empty((IM, JM, LM))
    for l in range(LM):
        mb[:, :, l] = maold[l] * kg2mb * axyp                        # MAOLD(L,:,:)*KG2MB*AXYP(:,:)
    q0 = aadvq0(mus, mvs, mws, mb, imaxj, byim_geom, stats)
    rm = q * mb
    rmom = qmom * mb[None]
    aad = aadvq(rm, rmom, mb, q0, imaxj, byim_qus, stats, hooks)
    bymma = 1. / aad['mma']
    return dict(q=aad['rm'] * bymma, qmom=aad['rmom'] * bymma[None], mb=mb, q0=q0, ain=(rm, rmom), aad=aad)
