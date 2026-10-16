"""Full-fidelity numpy port of the 1-D quadratic-upstream (QUS) kernels of QUSDEF.f (D99/D100).

Ported (QUSDEF.f line ranges, ModelE2_planet_2.0):
  adv1d               43-221   cyclic 1-D flux + moment update; used by AADVTX (x) and AADVTZ (z, cm(lm)=0)
  advection_1D_custom 224-635  non-cyclic all-lines variant used by AADVTY (y); no qlimit/limiter inside
  limitq              639-758  positivity limiter (live in CLOUDS2 moist convection; dead on the T path
                               of AADVT because AADVT is called with qlimit=.false.)
These are SHARED with moist convection (CLOUDS2.F90:2385,2426), so they are plain importable functions with
no AADVT-specific state.

Array conventions (0-based; Fortran 1-based in comments):
  lines are batched on leading axes, the cell axis is the LAST axis of s/mass/dm (length nx);
  smom is (9, ..., nx) with moment axis first.  Moment numbering (QUSDEF): MX=0 MY=1 MZ=2 MXX=3 MYY=4 MZZ=5
  MXY=6 MZX=7 MYZ=8; XDIR/YDIR/ZDIR are the Fortran direction permutations (adv1d's local mx,my,mz,mxx,myy,
  mzz,mxy,myz,mzx = dir(1..9)).

Bitwise-faithfulness notes (ifort flags -O2 -fp-model strict -assume protect_parens: no FMA, no
reassociation): every expression keeps the Fortran evaluation order (a*b*c = (a*b)*c, A-3*B+C = (A-3*B)+C);
`fracm**3` is np.power(fracm,3.) (matches ifort bitwise on 38448 harness cells; x*x*x differs in 516).  Within one 1-D line the flux/update loops of the Fortran read only
quantities that are not overwritten before use (f, fmom, dm are never modified; mass/s/smom of cell n are
read before being overwritten and only for cell n), so the cell loops are exact as whole-axis vector
operations.  The only genuinely sequential piece is the qlimit loop (limitq modifies f(n-1)), which is
implemented as a scalar loop (and validated only against a standalone ifort build of the real adv1d, see
fullfidelity/dyn_adv1d_compare.py; it is NOT exercised by the AADVT dumps).
prather_limits (QUSDEF.f:208-216, 551-557, ...) is dead (default 0, not set in the rundeck) and not ported.
"""
import numpy as np

NMOM = 9
MX, MY, MZ, MXX, MYY, MZZ, MXY, MZX, MYZ = range(9)
XDIR = (MX, MY, MZ, MXX, MYY, MZZ, MXY, MYZ, MZX)
YDIR = (MY, MX, MZ, MYY, MXX, MZZ, MXY, MZX, MYZ)
ZDIR = (MZ, MY, MX, MZZ, MYY, MXX, MYZ, MXY, MZX)
# moments with a horizontal component (QUSDEF ihmoms) and the two vertical-only ones
IHMOMS = (MX, MY, MXX, MYY, MXY, MYZ, MZX)


def _g(a, idx):
    return np.take_along_axis(a, idx, axis=-1)


def ordered_sum(a, axis=-1):
    """Left-to-right (Fortran/ifort strict) sum along `axis` (np.sum is pairwise; np.cumsum is sequential)."""
    a = np.moveaxis(np.asarray(a), axis, -1)
    return np.cumsum(a, axis=-1)[..., -1]


# --------------------------------------------------------------------------------------------- limitq
def limitq(anm1, an, fnm1, fn, sn, sx, sxx, stats=None):
    """QUSDEF.f:639-758.  Scalars in, returns (fnm1, fn, sx, sxx, ierr).  stats: optional dict of branch counts.

    ierr=1: |fr|>1 warning only; ierr=2: |fr|>1 and the net new tracer is negative (adv1d returns at once).
    """
    def hit(k):
        if stats is not None:
            stats[k] = stats.get(k, 0) + 1
    ierr = 0
    # no air leaving the box
    if anm1 >= 0. and an <= 0.:
        hit('no_outflow')
        return fnm1, fn, sx, sxx, ierr
    if anm1 < 0. and an > 0.:
        # air is leaving through both the left and right edges
        sl = -fnm1
        sr = +fn
        sc = sn - sl
        sc = sc - sr
        if sl >= 0. and sr >= 0. and sc >= 0.:
            hit('both_all_nonneg')
            return fnm1, fn, sx, sxx, ierr
        frl = anm1
        frl1 = frl + 1.
        frr = an
        frr1 = frr - 1.
        if sl >= 0. and sr >= 0.:                       # center division negative
            hit('both_center_neg')
            gamma = 1. + (frl - frr)
            t = (frl + frr)
            g13ab = gamma * gamma - 1. + 3. * (t * t)
            den = gamma * (12. * (t * t) + 5. * g13ab * g13ab)
            sxx = sxx - sc * 10. * g13ab / den
            sx = sx + sc * 12. * (frl + frr) / den
            sl = -frl * (sn - frl1 * (sx - (frl + frl1) * sxx))
            sr = sn - sl
        elif sr >= 0.:                                   # leftmost division
            hit('both_left_neg')
            u = (frl + frl1)
            sxx = sxx + sl * (frl + frl1) / (frl * frl1 * (.6 + u * u))
            sx = sn / frl1 + (frl + frl1) * sxx
            sr = frr * (sn - frr1 * (sx - (frr + frr1) * sxx))
            sl = 0.
        elif sl >= 0.:                                   # rightmost division
            hit('both_right_neg')
            u = (frr + frr1)
            sxx = sxx - sr * (frr + frr1) / (frr * frr1 * (.6 + u * u))
            sx = sn / frr1 + (frr + frr1) * sxx
            sl = -frl * (sn - frl1 * (sx - (frl + frl1) * sxx))
            sr = 0.
        else:
            hit('both_two_or_more_neg_initial')
        sc = sn - sl
        sc = sc - sr
        if sl <= 0. and sr <= 0.:                        # two outer divisions
            hit('both_fix_two_outer')
            gamma = 1. + (frl - frr)
            sxx = sn * (1. + gamma) / (2. * gamma * frl1 * frr1)
            sx = sn * (frl + frr) * (1. + 2. * gamma) / (2. * gamma * frl1 * frr1)
            sl = 0.
            sr = 0.
        elif sl <= 0. and sc <= 0.:                      # center/left divs
            hit('both_fix_center_left')
            sxx = sn / (2. * frr * frl1)
            sx = sn * (frl + frr + .5) / (frr * frl1)
            sl = 0.
            sr = sn
        elif sr <= 0. and sc <= 0.:                      # center/right divs
            hit('both_fix_center_right')
            sxx = sn / (2. * frl * frr1)
            sx = sn * (frl + frr - .5) / (frl * frr1)
            sl = sn
            sr = 0.
        fnm1 = -sl
        fn = +sr
        return fnm1, fn, sx, sxx, ierr
    # air is leaving only through one edge
    if an > 0.:
        fr = an
        sd = fn
        fsign = -1.
        hit('one_right')
    else:
        fr = anm1
        sd = -fnm1
        fsign = 1.
        hit('one_left')
    if abs(fr) > 1.:
        ierr = 1
        hit('one_warn_abs_gt_1')
        if sn + fnm1 - fn < 0:
            ierr = 2
            hit('one_error_new_sn_neg')
            return fnm1, fn, sx, sxx, ierr
    su = sn - sd
    if sd >= 0. and su >= 0.:
        hit('one_all_nonneg')
        return fnm1, fn, sx, sxx, ierr
    fr1 = fr + fsign
    w = (fr + fr1)
    if sd < 0.:                                          # downstream division negative
        hit('one_down_neg')
        sxx = sxx + fsign * sd * (fr + fr1) / (fr * fr1 * (.6 + w * w))
        sx = sn / fr1 + (fr + fr1) * sxx
        su = sn
    else:                                                # upstream division negative
        hit('one_up_neg')
        sxx = sxx - fsign * su * (fr + fr1) / (fr * fr1 * (.6 + w * w))
        sx = sn / fr + (fr + fr1) * sxx
        su = 0.
    sd = sn - su
    if an > 0.:
        fn = sd
    else:
        fnm1 = -sd
    return fnm1, fn, sx, sxx, ierr


# ------------------------------------------------------------------------------------- shared stages
def _flux_stage(s, smom, mass, dm, nn, dirv):
    mx, my, mz, mxx, myy, mzz, mxy, myz, mzx = dirv
    frac1 = np.where(dm < 0., 1., -1.)
    ms = _g(mass, nn)
    with np.errstate(all='ignore'):
        fracm = dm / ms
    fracm = np.where(ms <= 0., 0., fracm)
    frac1 = fracm + frac1
    sn = _g(s, nn)
    sx = _g(smom[mx], nn)
    sxx = _g(smom[mxx], nn)
    with np.errstate(all='ignore'):
        f = fracm * (sn - frac1 * (sx - (frac1 + fracm) * sxx))
    return f, fracm, frac1


def _slope_stage(smom, dm, f, fracm, frac1, nn, dirv):
    mx, my, mz, mxx, myy, mzz, mxy, myz, mzx = dirv
    fm = np.empty_like(smom)
    sx = _g(smom[mx], nn)
    sxx = _g(smom[mxx], nn)
    with np.errstate(all='ignore'):
        fm[mx] = dm * (fracm * fracm * (sx - 3. * frac1 * sxx) - 3. * f)
        fm[mxx] = dm * (dm * np.power(fracm, 3.) * sxx - 5. * (dm * f + fm[mx]))
        sxy = _g(smom[mxy], nn)
        fm[my] = fracm * (_g(smom[my], nn) - frac1 * sxy)
        fm[mxy] = dm * (fracm * fracm * sxy - 3. * fm[my])
        szx = _g(smom[mzx], nn)
        fm[mz] = fracm * (_g(smom[mz], nn) - frac1 * szx)
        fm[mzx] = dm * (fracm * fracm * szx - 3. * fm[mz])
        fm[myy] = fracm * _g(smom[myy], nn)
        fm[mzz] = fracm * _g(smom[mzz], nn)
        fm[myz] = fracm * _g(smom[myz], nn)
    return fm


def _update_stage(s, smom, mass, dm, f, fm, prev, dirv):
    """In-place update of s, smom, mass (all cells at once); `prev(a)` gives a(n-1) (cyclic or zero-padded)."""
    mx, my, mz, mxx, myy, mzz, mxy, myz, mzx = dirv
    dmm, fmm = prev(dm), prev(f)
    with np.errstate(all='ignore'):
        tmp = mass + dmm
        mnew = tmp - dm
        bymnew = 1. / mnew
        dm2 = dmm + dm
        tmp = s + fmm
        s[...] = tmp - f
        m0 = mass            # old mass (not yet overwritten)
        fmx_m, fmxx_m = prev(fm[mx]), prev(fm[mxx])
        smom[mx] = (smom[mx] * m0 - 3. * (-dm2 * s + m0 * (fmm + f)) + (fmx_m - fm[mx])) * bymnew
        smom[mxx] = (smom[mxx] * m0 * m0
                     + 2.5 * s * (m0 * m0 - mnew * mnew - 3. * dm2 * dm2)
                     + 5. * (m0 * (m0 * (fmm - f) - fmx_m - fm[mx]) + dm2 * smom[mx] * mnew)
                     + (fmxx_m - fm[mxx])) * (bymnew * bymnew)
        fmy_m = prev(fm[my])
        smom[my] = smom[my] + fmy_m - fm[my]
        smom[mxy] = (smom[mxy] * m0 - 3. * (-dm2 * smom[my] + m0 * (fmy_m + fm[my]))
                     + (prev(fm[mxy]) - fm[mxy])) * bymnew
        fmz_m = prev(fm[mz])
        smom[mz] = smom[mz] + fmz_m - fm[mz]
        smom[mzx] = (smom[mzx] * m0 - 3. * (-dm2 * smom[mz] + m0 * (fmz_m + fm[mz]))
                     + (prev(fm[mzx]) - fm[mzx])) * bymnew
        smom[myy] = smom[myy] + prev(fm[myy]) - fm[myy]
        smom[mzz] = smom[mzz] + prev(fm[mzz]) - fm[mzz]
        smom[myz] = smom[myz] + prev(fm[myz]) - fm[myz]
    mass[...] = mnew
    dead = mass <= 0.
    if dead.any():
        s[dead] = 0.
        smom[:, dead] = 0.


def _cyc_prev(a):
    return np.roll(a, 1, axis=-1)


def _zero_prev(a):
    out = np.zeros_like(a)
    out[..., 1:] = a[..., :-1]
    return out


# ---------------------------------------------------------------------------------------------- adv1d
def adv1d(s, smom, mass, dm, qlimit=False, dirv=XDIR, stats=None):
    """QUSDEF.f:43-221 for a batch of independent cyclic lines.

    s (...,nx), smom (9,...,nx), mass (...,nx), dm (...,nx): float64, updated IN PLACE (s, smom, mass), dm read-only.
    Returns (f, fmom, ierr, nerr): f (...,nx) tracer flux (diagnostic), fmom (9,...,nx), and ierr/nerr as the
    Fortran out-arguments (qlimit path: ierr is the status of the last limitq call, nerr the last 1-based n with ierr>0, exactly as the Fortran out-arguments; for a batch of lines, the values of the last line).
    ierr>0 only occurs on the qlimit path.
    """
    nx = s.shape[-1]
    k = np.arange(nx)
    nn = np.broadcast_to(np.where(dm < 0., (k + 1) % nx, k), dm.shape)
    f, fracm, frac1 = _flux_stage(s, smom, mass, dm, nn, dirv)
    mx, mxx = dirv[0], dirv[3]
    ierr, nerr = 0, 0
    if qlimit:
        ierr, nerr = _qlimit_cyclic(f, fracm, s, smom, mx, mxx, stats)
        if ierr == 2:
            return f, None, ierr, nerr
    fm = _slope_stage(smom, dm, f, fracm, frac1, nn, dirv)
    _update_stage(s, smom, mass, dm, f, fm, _cyc_prev, dirv)
    return f, fm, ierr, nerr


def _qlimit_cyclic(f, fracm, s, smom, mx, mxx, stats):
    """qlimit loop of adv1d (QUSDEF.f:107-130), sequential over n, scalar over lines."""
    nx = s.shape[-1]
    lead = s.shape[:-1]
    ierr_out, nerr_out = 0, 0
    for idx in np.ndindex(*lead) if lead else [()]:
        nm1 = nx - 1
        for n in range(nx):
            an = fracm[idx + (n,)]
            anm1 = fracm[idx + (nm1,)]
            fn = f[idx + (n,)]
            fnm1 = f[idx + (nm1,)]
            sn = s[idx + (n,)]
            sxn = smom[(mx,) + idx + (n,)]
            sxxn = smom[(mxx,) + idx + (n,)]
            fnm1, fn, sxn, sxxn, ierr = limitq(anm1, an, fnm1, fn, sn, sxn, sxxn, stats)
            ierr_out = ierr                  # Fortran ierr is the status of the LAST limitq call ...
            if ierr > 0:
                nerr_out = n + 1             # ... and nerr the last n with ierr>0 (set only then)
                if ierr == 2:
                    return ierr_out, nerr_out
            f[idx + (n,)] = fn
            f[idx + (nm1,)] = fnm1
            smom[(mx,) + idx + (n,)] = sxn
            smom[(mxx,) + idx + (n,)] = sxxn
            nm1 = n
    return ierr_out, nerr_out


# ------------------------------------------------------------------------------------ advection_1D_custom
def advection_1d_custom(s, smom, mass, dm, dirv=YDIR):
    """QUSDEF.f:224-635 (qlimit=.false. path, the only one AADVTY uses) for batched NON-cyclic lines.

    Same array convention as adv1d (cell axis last = J).  Interface n (between cells n and n+1) takes its upwind
    cell nn=n+1 if dm(n)<0 else n; the caller must make dm of the last cell zero (AADVTY sets bm(:,jm)=0), so nn
    never leaves the line (a defensive clamp is applied).  The Fortran special-cases the first cell (south pole,
    update_tracer_mass) by dropping the nm1 terms; those are exactly the generic formulas with a zero neighbour
    (a-0=a, 0-f=-f in IEEE), so the zero-padded generic update is bitwise identical.
    Returns (f, fmom); s, smom, mass updated in place.
    """
    nx = s.shape[-1]
    k = np.arange(nx)
    nn = np.broadcast_to(np.minimum(np.where(dm < 0., k + 1, k), nx - 1), dm.shape)
    f, fracm, frac1 = _flux_stage(s, smom, mass, dm, nn, dirv)
    fm = _slope_stage(smom, dm, f, fracm, frac1, nn, dirv)
    _update_stage(s, smom, mass, dm, f, fm, _zero_prev, dirv)
    return f, fm
