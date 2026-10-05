"""Full-fidelity numpy port of the AADVT family of QUS_DRV.f (D99/D100): temperature advection driver.

Ported (QUS_DRV.f, ModelE2_planet_2.0): AADVT 70-196 (driver: pole fill, mass-unit conversion, X(half) Y Z
X(half) sweeps, back-conversion), AADVTX 198-316, AADVTY 318-471, AADVTZ 474-573.  The 1-D kernels are in
dyn_adv1d_ff.py (adv1d, advection_1d_custom, limitq; shared with moist convection).

Array conventions (the Fortran index order, 0-based): rm, mm, mu, mv (IM,JM,LM); rmom (9,IM,JM,LM); mw
(IM,JM,LM-1) as in ATMDYN_COM (MW is allocated (IM,JM,LM-1)); fqu, fqv (IM,JM).  JM=46, serial (one MPI rank;
J_0=1, J_1=JM, J_0S=2, J_1S=JM-1 in Fortran numbering; the halo calls are no-ops).

Courant sub-stepping (the scoping-doc warning): AADVTX/AADVTZ pick, per row (j,l) / per column (i,j), the
smallest nstep<=20 for which the sub-stepped Courant number is <=1, and then run adv1d nstep times with
dm=mu/nstep.  Rows are independent, so the batched implementation groups rows by their own nstep value and
runs exactly nstep sub-steps for each group (a fixed max-nstep trip without masking would change the result;
`unmasked_max_nstep=True` reproduces that error and is used only by the mutation tests).  The scalar
per-row/column reference loops (`*_rowloop`) follow the Fortran control flow literally and are used by tests.
Reductions (the polar-box `sum`, the fqu/fqv accumulation over l) follow the Fortran order (left to right).
"""
import numpy as np
from dyn_adv1d_ff import (adv1d, advection_1d_custom, XDIR, YDIR, ZDIR, MX, MZ, MZZ, IHMOMS, ordered_sum)

IM, JM, LM = 72, 46, 40
BYIM = 1.0 / 72.0          # BYIM = 1.D0/REAL(IM,KIND=8)
NSTEP_MAX = 20


# ------------------------------------------------------------------------------------- Courant counts
def courant_nstep_x(mu, mass, nmax=NSTEP_MAX):
    """Per-row nstep of AADVTX (QUS_DRV.f:237-258), batched over rows.

    mu, mass: (B, IM) (west-east flux, mass) -> (nstep (B,) int, courmax (B,)).
    """
    B, im = mu.shape
    nstep = np.zeros(B, dtype=int)
    courmax = np.full(B, 2.)
    active = np.ones(B, dtype=bool)
    with np.errstate(all='ignore'):
        for trial in range(1, nmax + 1):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            nstep[idx] = trial
            bynstep = 1. / float(trial)
            am = mu[idx] * bynstep
            mi = mass[idx].copy()
            cm = np.zeros(idx.size)
            for ns in range(1, trial + 1):
                mip1 = np.roll(mi, -1, axis=1)
                cand = np.where(am > 0., am / mi, -am / mip1)
                cm = np.maximum(cm, np.max(cand, axis=1))
                if ns < trial:
                    mi = mi + (np.roll(am, 1, axis=1) - am)
            courmax[idx] = cm
            active[idx] = cm > 1.
    return nstep, courmax


def courant_nstep_z(mw, mass, nmax=NSTEP_MAX):
    """Per-column nstep of AADVTZ (QUS_DRV.f:508-530), batched over columns.

    mw (B, LM-1) vertical flux (cm(lm)=0 appended internally), mass (B, LM) -> (nstep, courmax).
    """
    B, lm = mass.shape
    mwf = np.zeros((B, lm))
    mwf[:, :lm - 1] = mw
    nstep = np.zeros(B, dtype=int)
    courmax = np.full(B, 2.)
    active = np.ones(B, dtype=bool)
    with np.errstate(all='ignore'):
        for trial in range(1, nmax + 1):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            nstep[idx] = trial
            bynstep = 1. / float(trial)
            cm = mwf[idx] * bynstep
            cm[:, lm - 1] = 0.
            ml = mass[idx].copy()
            cmax = np.zeros(idx.size)
            for ns in range(1, trial + 1):
                c = cm[:, :lm - 1]
                cand = np.where(c > 0., c / ml[:, :lm - 1], -c / ml[:, 1:])
                cmax = np.maximum(cmax, np.max(cand, axis=1))
                if ns < trial:
                    ml[:, 1:] = ml[:, 1:] + c            # iteration l: mass(l+1) += cm(l) ...
                    ml[:, :lm - 1] = ml[:, :lm - 1] - c  # ... then iteration l+1: mass(l+1) -= cm(l+1)
            courmax[idx] = cmax
            active[idx] = cmax > 1.
    return nstep, courmax


def courant_nstep_x_rowloop(mu_row, mass_row):
    """Literal scalar transcription of the Fortran do-while for ONE row (testing reference)."""
    im = mu_row.size
    nstep = 0
    courmax = 2.
    while courmax > 1. and nstep < NSTEP_MAX:
        nstep += 1
        bynstep = 1. / float(nstep)
        am = mu_row * bynstep
        mass_i = mass_row.copy()
        courmax = 0.
        for ns in range(1, nstep + 1):
            i = im - 1
            for ip1 in range(im):
                if am[i] > 0.:
                    courmax = max(courmax, +am[i] / mass_i[i])
                else:
                    courmax = max(courmax, -am[i] / mass_i[ip1])
                i = ip1
            if ns < nstep:
                i = im - 1
                for ip1 in range(im):
                    mass_i[ip1] = mass_i[ip1] + (am[i] - am[ip1])
                    i = ip1
    return nstep, courmax


def courant_nstep_z_rowloop(mw_col, mass_col):
    lm = mass_col.size
    nstep = 0
    courmax = 2.
    while courmax > 1. and nstep < NSTEP_MAX:
        nstep += 1
        bynstep = 1. / float(nstep)
        cm = np.zeros(lm)
        cm[:lm - 1] = mw_col * bynstep
        cm[lm - 1] = 0.
        mass_l = mass_col.copy()
        courmax = 0.
        for ns in range(1, nstep + 1):
            for l in range(lm - 1):
                if cm[l] > 0.:
                    courmax = max(courmax, +cm[l] / mass_l[l])
                else:
                    courmax = max(courmax, -cm[l] / mass_l[l + 1])
            if ns < nstep:
                for l in range(lm - 1):
                    mass_l[l] = mass_l[l] - cm[l]
                    mass_l[l + 1] = mass_l[l + 1] + cm[l]
    return nstep, courmax


# ----------------------------------------------------------------------------------------- X and Z
def _run_groups(nstep, nmax_override, apply_group):
    """Run apply_group(rows, nstep_value, bynstep) once per distinct per-row nstep (masked per-row trip count)."""
    if nmax_override:
        rows = np.arange(nstep.size)
        apply_group(rows, int(nstep.max()), 1. / float(nstep.max()))
        return
    for v in np.unique(nstep):
        rows = np.nonzero(nstep == v)[0]
        apply_group(rows, int(v), 1. / float(v))


def aadvtx(rm, rmom, mass, mu, qlimit=False, fqu=None, stats=None, unmasked_max_nstep=False, ns_out=None):
    """QUS_DRV.f:198-316.  rm, mass, mu (IM,JM,LM); rmom (9,IM,JM,LM); updated in place (rm, rmom, mass).

    Rows J_0S..J_1S (0-based 1..JM-2) only.  fqu (IM,JM) is accumulated (fqu += sum over l of the per-row
    sub-step flux sums, in l order).  ns_out: optional (JM,LM) int array receiving each row's nstep.
    Returns fqu.  Raises RuntimeError where the Fortran calls stop_model (courmax>1 at nstep=20, qlimit error).
    """
    if fqu is None:
        fqu = np.zeros((IM, JM))
    js = slice(1, JM - 1)
    nj = JM - 2
    # lines ordered (j, l) -> batch index b = j*LM + l ; cell axis last (i)
    mu_b = np.ascontiguousarray(np.moveaxis(mu[:, js, :], 0, -1)).reshape(nj * LM, IM)
    mass_b = np.ascontiguousarray(np.moveaxis(mass[:, js, :], 0, -1)).reshape(nj * LM, IM)
    s_b = np.ascontiguousarray(np.moveaxis(rm[:, js, :], 0, -1)).reshape(nj * LM, IM)
    m_b = np.ascontiguousarray(np.moveaxis(rmom[:, :, js, :], 1, -1)).reshape(9, nj * LM, IM)
    nstep, courmax = courant_nstep_x(mu_b, mass_b)
    if np.any(courmax > 1.):
        raise RuntimeError("aadvtx: courmax>1 after nstep=20 (Fortran stop_model)")
    if ns_out is not None:
        ns_out[js, :] = nstep.reshape(nj, LM)
    hf = np.zeros((nj * LM, IM))
    err = []

    def grp(rows, v, by):
        am = mu_b[rows] * by
        s, sm, ma = s_b[rows], m_b[:, rows], mass_b[rows]
        h = np.zeros_like(s)
        for _ in range(v):
            f, _, ierr, nerr = adv1d(s, sm, ma, am, qlimit, XDIR, stats)
            if ierr == 2:
                err.append(ierr)
            h = h + f
        s_b[rows], m_b[:, rows], mass_b[rows], hf[rows] = s, sm, ma, h
    _run_groups(nstep, unmasked_max_nstep, grp)
    if err:
        raise RuntimeError("aadvtx: qlimit error (Fortran ICKERR>0)")
    rm[:, js, :] = np.moveaxis(s_b.reshape(nj, LM, IM), -1, 0)
    mass[:, js, :] = np.moveaxis(mass_b.reshape(nj, LM, IM), -1, 0)
    rmom[:, :, js, :] = np.moveaxis(m_b.reshape(9, nj, LM, IM), -1, 1)
    hf3 = np.moveaxis(hf.reshape(nj, LM, IM), -1, 0)          # (IM, nj, LM)
    for l in range(LM):                                       # fqu(:,j) += hfqu(:,j,l), l sequential
        fqu[:, js] = fqu[:, js] + hf3[:, :, l]
    return fqu


def aadvtz(rm, rmom, mass, mw, qlimit=False, stats=None, unmasked_max_nstep=False, ns_out=None):
    """QUS_DRV.f:474-573.  mw is the MFLX array of AADVT: (IM,JM,LM) with mflx(:,:,lm)=0 (or (IM,JM,LM-1))."""
    mwf = np.zeros((IM, JM, LM))
    mwf[:, :, :mw.shape[2]] = mw
    mwf[:, :, LM - 1] = 0.
    B = IM * JM
    # lines ordered (i, j) -> b = i*JM + j ; cell axis last (l)
    cm_b = mwf.reshape(B, LM)
    mass_b = np.ascontiguousarray(mass.reshape(B, LM))
    s_b = np.ascontiguousarray(rm.reshape(B, LM))
    m_b = np.ascontiguousarray(rmom.reshape(9, B, LM))
    nstep, courmax = courant_nstep_z(cm_b[:, :LM - 1], mass_b)
    if np.any(courmax > 1.):
        raise RuntimeError("aadvtz: courmax>1 after nstep=20 (Fortran stop_model)")
    if ns_out is not None:
        ns_out[...] = nstep.reshape(IM, JM)
    err = []

    def grp(rows, v, by):
        cm = cm_b[rows] * by
        cm[:, LM - 1] = 0.
        s, sm, ma = s_b[rows], m_b[:, rows], mass_b[rows]
        for _ in range(v):
            f, _, ierr, nerr = adv1d(s, sm, ma, cm, qlimit, ZDIR, stats)
            if ierr == 2:
                err.append(ierr)
        s_b[rows], m_b[:, rows], mass_b[rows] = s, sm, ma
    _run_groups(nstep, unmasked_max_nstep, grp)
    if err:
        raise RuntimeError("aadvtz: qlimit error (Fortran ICKERR>0)")
    rm[...] = s_b.reshape(IM, JM, LM)
    mass[...] = mass_b.reshape(IM, JM, LM)
    rmom[...] = m_b.reshape(9, IM, JM, LM)


def aadvtx_rowloop(rm, rmom, mass, mu, qlimit=False, fqu=None, stats=None):
    """Literal per-row Fortran control flow (slow reference): one row (j,l) at a time, own nstep."""
    if fqu is None:
        fqu = np.zeros((IM, JM))
    hfqu = np.zeros((IM, JM, LM))
    for l in range(LM):
        for j in range(1, JM - 1):
            nstep, courmax = courant_nstep_x_rowloop(mu[:, j, l], mass[:, j, l])
            if courmax > 1.:
                raise RuntimeError("aadvtx courmax")
            am = (mu[:, j, l] * (1. / float(nstep)))[None, :]
            s = rm[:, j, l].copy()[None, :]
            sm = rmom[:, :, j, l].copy()[:, None, :]
            ma = mass[:, j, l].copy()[None, :]
            for _ in range(nstep):
                f, _, ierr, _ = adv1d(s, sm, ma, am, qlimit, XDIR, stats)
                hfqu[:, j, l] = hfqu[:, j, l] + f[0]
            rm[:, j, l], mass[:, j, l], rmom[:, :, j, l] = s[0], ma[0], sm[:, 0, :]
    for j in range(1, JM - 1):
        for l in range(LM):
            fqu[:, j] = fqu[:, j] + hfqu[:, j, l]
    return fqu


def aadvtz_rowloop(rm, rmom, mass, mw, qlimit=False, stats=None):
    mwf = np.zeros((IM, JM, LM))
    mwf[:, :, :mw.shape[2]] = mw
    mwf[:, :, LM - 1] = 0.
    for j in range(JM):
        for i in range(IM):
            nstep, courmax = courant_nstep_z_rowloop(mwf[i, j, :LM - 1], mass[i, j, :])
            if courmax > 1.:
                raise RuntimeError("aadvtz courmax")
            cm = (mwf[i, j, :] * (1. / float(nstep)))[None, :]
            cm[0, LM - 1] = 0.
            s = rm[i, j, :].copy()[None, :]
            sm = rmom[:, i, j, :].copy()[:, None, :]
            ma = mass[i, j, :].copy()[None, :]
            for _ in range(nstep):
                adv1d(s, sm, ma, cm, qlimit, ZDIR, stats)
            rm[i, j, :], mass[i, j, :], rmom[:, i, j, :] = s[0], ma[0], sm[:, 0, :]


# ------------------------------------------------------------------------------------------------ Y
def aadvty(rm, rmom, mass, mv, qlimit=False):
    """QUS_DRV.f:318-471 (all lines/levels at once through advection_1d_custom).

    mv (IM,JM,LM): the MFLX array of AADVT (mflx(:,jm,:)=0).  Updated in place.  Returns fqv (IM,JM).
    qlimit must be False here: advection_1D_custom is the no-limiter variant and the Fortran ignores qlimit in
    the sense that the limiter calls in it would need halo data; AADVT is only ever called with .false..
    """
    if qlimit:
        raise NotImplementedError("aadvty qlimit=.true. (apply_limiter in advection_1D_custom) not ported: dead path")
    fqv = np.zeros((IM, JM))
    m_sp = np.zeros(LM); rm_sp = np.zeros(LM); rzm_sp = np.zeros(LM); rzzm_sp = np.zeros(LM)
    m_np = np.zeros(LM); rm_np = np.zeros(LM); rzm_np = np.zeros(LM); rzzm_np = np.zeros(LM)
    bm = np.array(mv, dtype=float, copy=True)
    for l in range(LM):
        # scale polar boxes to their full extent
        mass[:, 0, l] = mass[:, 0, l] * IM
        m_sp[l] = mass[0, 0, l]
        rm[:, 0, l] = rm[:, 0, l] * IM
        rm_sp[l] = rm[0, 0, l]
        rmom[:, :, 0, l] = rmom[:, :, 0, l] * IM
        rzm_sp[l] = rmom[MZ, 0, 0, l]
        rzzm_sp[l] = rmom[MZZ, 0, 0, l]
        mass[:, JM - 1, l] = mass[:, JM - 1, l] * IM
        m_np[l] = mass[0, JM - 1, l]
        rm[:, JM - 1, l] = rm[:, JM - 1, l] * IM
        rm_np[l] = rm[0, JM - 1, l]
        rmom[:, :, JM - 1, l] = rmom[:, :, JM - 1, l] * IM
        rzm_np[l] = rmom[MZ, 0, JM - 1, l]
        rzzm_np[l] = rmom[MZZ, 0, JM - 1, l]
        # poles: horizontal moments are zero; no flux through the north pole
        for n in IHMOMS:
            rmom[n, :, 0, l] = 0.
        bm[:, JM - 1, l] = 0.
        for n in IHMOMS:
            rmom[n, :, JM - 1, l] = 0.
    # lines along J: batch (i, l), cell axis J
    s_b = np.ascontiguousarray(np.moveaxis(rm, 1, -1))                # (IM, LM, JM)
    ma_b = np.ascontiguousarray(np.moveaxis(mass, 1, -1))
    dm_b = np.ascontiguousarray(np.moveaxis(bm, 1, -1))
    sm_b = np.ascontiguousarray(np.moveaxis(rmom, 2, -1))             # (9, IM, LM, JM)
    f_b, _ = advection_1d_custom(s_b, sm_b, ma_b, dm_b, YDIR)
    rm[...] = np.moveaxis(s_b, -1, 1)
    mass[...] = np.moveaxis(ma_b, -1, 1)
    rmom[...] = np.moveaxis(sm_b, -1, 2)
    f_j = np.moveaxis(f_b, -1, 1)                                     # (IM, JM, LM)
    for n in IHMOMS:                                                  # horizontal moments are zero at pole
        rmom[n, :, 0, :] = 0.
        rmom[n, :, JM - 1, :] = 0.
    for l in range(LM):
        # average and unscale polar boxes (Fortran sum() order: left to right)
        mass[:, 0, l] = (m_sp[l] + ordered_sum(mass[:, 0, l] - m_sp[l])) * BYIM
        rm[:, 0, l] = (rm_sp[l] + ordered_sum(rm[:, 0, l] - rm_sp[l])) * BYIM
        rmom[MZ, :, 0, l] = (rzm_sp[l] + ordered_sum(rmom[MZ, :, 0, l] - rzm_sp[l])) * BYIM
        rmom[MZZ, :, 0, l] = (rzzm_sp[l] + ordered_sum(rmom[MZZ, :, 0, l] - rzzm_sp[l])) * BYIM
        mass[:, JM - 1, l] = (m_np[l] + ordered_sum(mass[:, JM - 1, l] - m_np[l])) * BYIM
        rm[:, JM - 1, l] = (rm_np[l] + ordered_sum(rm[:, JM - 1, l] - rm_np[l])) * BYIM
        rmom[MZ, :, JM - 1, l] = (rzm_np[l] + ordered_sum(rmom[MZ, :, JM - 1, l] - rzm_np[l])) * BYIM
        rmom[MZZ, :, JM - 1, l] = (rzzm_np[l] + ordered_sum(rmom[MZZ, :, JM - 1, l] - rzzm_np[l])) * BYIM
    for l in range(LM):
        fqv = fqv + f_j[:, :, l]                                      # fqv(:,j) += f_j(:,j,l), l sequential
    fqv[:, JM - 1] = 0.
    return fqv


# --------------------------------------------------------------------------------------------- driver
def aadvt(dt, mm, rm, rmom, mu, mv, mw, qlimit=False, stages=None, stats=None, **kw):
    """QUS_DRV.f:70-196 AADVT.  Inputs are NOT modified; returns dict(mm, rm, rmom, fqu, fqv).

    dt (s); mm (IM,JM,LM) air mass (kg) -> updated mass; rm (T), rmom (TMOM) concentration units in/out;
    mu, mv (IM,JM,LM), mw (IM,JM,LM-1) kg/s fluxes.  stages: optional dict that receives mass-unit checkpoints
    'x1','y','z' (rm, rmom, mm copies) and 'ns' (nsx1, nsx2 (JM,LM), nsz (IM,JM)).
    """
    mm = np.array(mm, dtype=float, copy=True)
    rm = np.array(rm, dtype=float, copy=True)
    rmom = np.array(rmom, dtype=float, copy=True)
    fqu = np.zeros((IM, JM))
    # fill in values at the poles (rows 1 and JM, I=2..IM)
    for j in (0, JM - 1):
        rm[1:, j, :] = rm[0, j, :][None, :]
        rmom[:, 1:, j, :] = rmom[:, 0, j, :][:, None, :]
    # concentration -> mass units
    for l in range(LM):
        rm[:, :, l] = rm[:, :, l] * mm[:, :, l]
        rmom[:, :, :, l] = rmom[:, :, :, l] * mm[:, :, l][None]
    nsx1 = np.zeros((JM, LM), dtype=int); nsx2 = np.zeros((JM, LM), dtype=int); nsz = np.zeros((IM, JM), dtype=int)
    mflx = mu * (.5 * dt)
    aadvtx(rm, rmom, mm, mflx, qlimit, fqu, stats, ns_out=nsx1, **kw)
    if stages is not None:
        stages['x1'] = (rm.copy(), rmom.copy(), mm.copy())
    mflx = np.zeros((IM, JM, LM))
    mflx[:, :JM - 1, :] = mv[:, 1:JM, :] * dt
    mflx[:, JM - 1, :] = 0.
    fqv = aadvty(rm, rmom, mm, mflx, qlimit)
    if stages is not None:
        stages['y'] = (rm.copy(), rmom.copy(), mm.copy())
    mflx = np.zeros((IM, JM, LM))
    mflx[:, :, :LM - 1] = mw[:, :, :LM - 1] * (-dt)
    mflx[:, :, LM - 1] = 0.
    aadvtz(rm, rmom, mm, mflx, qlimit, stats, ns_out=nsz, **kw)
    if stages is not None:
        stages['z'] = (rm.copy(), rmom.copy(), mm.copy())
    mflx = mu * (.5 * dt)
    aadvtx(rm, rmom, mm, mflx, qlimit, fqu, stats, ns_out=nsx2, **kw)
    if stages is not None:
        stages['ns'] = (nsx1, nsx2, nsz)
    # mass -> concentration units (multiply by the reciprocal, as the Fortran does)
    for l in range(LM):
        by = 1. / mm[:, :, l]
        rm[:, :, l] = rm[:, :, l] * by
        rmom[:, :, :, l] = rmom[:, :, :, l] * by[None]
    return dict(mm=mm, rm=rm, rmom=rmom, fqu=fqu, fqv=fqv)
