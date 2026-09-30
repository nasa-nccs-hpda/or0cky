"""Full-fidelity port of OCNDYN2.f's OADVT2 dispatcher + OADVTX2/OADVTY2/OADVTZ2 -- Stage 2 of
the DYNSI/ocean port, D45. The long-timestep advection of potential enthalpy (G0M) and salt
(S0M), called twice per OCEANS invocation (QLIMIT=.FALSE. for G0M, .TRUE. for S0M), each call
re-deriving MO1(=MA) identically from the same SMU/SMV/SMW flux fields (D43/D44).

Linear-upstream-with-limited-moments scheme (the "linear" path -- USE_QUS=0 for this rundeck,
confirmed dead D41, so OADVT3's full quadratic-moments scheme is never used). OADVT2 itself is a
thin Strang-splitting dispatcher: X(half dt), Y(dt), Z(dt), X(half dt) again, then fills the pole
row uniformly.

OADVTX2/OADVTY2 never touch J=1 or J=JM directly in their own update loops (J1P=2..JNP=JM-1) --
X advection is genuinely periodic in longitude (dateline wraparound at I=1/I=IM, handled here via
circular neighbor indexing, verified equivalent to the Fortran's explicit dateline special-casing
since MUDT is already zero at inactive cells either way, the same masking-equivalence argument
established in D40/42/43/44). OADVTY2 handles the pole row itself (copy across longitudes before
the Y sweep, area-average after). OADVTZ2 uses a centered "R_edge" flux value (LUS_VERT_ADV not
defined for this rundeck, confirmed via the preprocessor block -- the simpler upstream-only path
is dead code here) requiring a `wtdn`-weighted edge reconstruction between adjacent layers.

OIJL (a pure diagnostic mass-flux accumulator, `OIJL(I,J,L) = OIJL(I,J,L) + FM`) is deliberately
not ported -- same "skip pure diagnostics" pattern as D31's RESET_SURF_FLUXES.

`MA` (mass advected) is NOT `MO1`'s carried-in value -- OADVT2 overwrites it unconditionally from
`OCEAN_DYN`'s `MMI` (`MA = MB` where `MB=>mmi`) at the top of every call. `MMI = MO0*DXYPO(J)`
using `ODHORZ0`'s (D40) already-validated `mo0` input, frozen from before any dynamics
sub-stepping this OCEANS call (see `oadvt2_compare.py`'s `load_mmi`).
"""
import math
import numpy as np

IM, JM, LMO = 72, 46, 13


def _sign(a, b):
    """Fortran SIGN(A,B): magnitude of A, sign of B (B=0 treated as positive, matching ifort)."""
    return math.copysign(a, b) if b != 0.0 else abs(a)


def _fortran_sum(a):
    """Sequential left-to-right accumulation, matching ifort's `-fp-model strict` SUM intrinsic
    (no reassociation). `np.sum()` uses pairwise/blocked reduction -- different rounding, which
    was the actual source of a real (if small, ~1e-6 relative) pole-row mismatch found by
    debugging D45's OADVTY2 pole-averaging step (`mo(:,j,l)=sum(mo(:,j,l))/im`) against real
    Fortran output."""
    total = 0.0
    for x in a:
        total += x
    return total


def _get_i1i2(active):
    """OCNDYN.f:1494 `get_i1i2`, transliterated: `active` is a 1-indexed bool array (size IM+1,
    index 0 unused). Finds maximal runs of True, LINEARLY (wraparound explicitly disabled in the
    real Fortran -- longitude periodicity at the dateline is handled by OADVTX2's own explicit
    i=1/i=IM special-casing below, not by the segment structure). Returns a list of (i1,i2)."""
    segs = []
    i = 1
    while i <= IM:
        if not active[i]:
            i += 1
            continue
        i1 = i
        while i <= IM and active[i]:
            i += 1
        segs.append((i1, i - 1))
    return segs


def oadvtx2(rm, rx, ry, rz, mm, mu, dt, qlimit, lmu, lmm):
    """OCNDYN2.f:1916-2128. All fields 1-indexed (IM+1,JM+1,LMO+1); operates across all layers.
    `mu` is the (already dt-independent) mass flux MU(I,J,L). Returns fresh (rm,rx,ry,rz,mm).

    `mudt` is a SINGLE array reused across every (l,j) pass in the real Fortran (`real*8,
    dimension(im) :: mudt`, declared once for the whole subroutine call, never reset): indices 1,
    2 and IM are unconditionally refreshed from `mu` at the top of every pass with any U-active
    cell (regardless of whether 1/2/IM are themselves active), while every other index is only
    refreshed within this pass's own U-active segments (built via `_get_i1i2` on `lmu`, LINEAR,
    not circular) -- everywhere else it deliberately retains whatever an EARLIER, unrelated (l,j)
    pass last left there. Both quirks are exercised by real data and were found by debugging a
    real mismatch, not assumed: (1) a cell can be M-active while U-inactive at the same (i,j,l)
    (e.g. a coastal cell whose eastward neighbor is land), so the basins-update loop below reads a
    stale `mudt(i)`; (2) the single-cell-segment skip (`i1yzm>1 .and. i1yzm==i2yzm`) leaves an
    isolated M-active cell with inactive neighbors on both sides entirely untouched by this pass.
    """
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy(); mm = mm.copy()
    rxlimit = 1.0 if qlimit else 0.0
    mudt = np.zeros(IM + 1)
    for l in range(1, LMO + 1):
        for j in range(2, JM):  # J1P=2 .. JNP=JM-1
            active_u = np.zeros(IM + 1, dtype=bool)
            for i in range(1, IM + 1):
                active_u[i] = lmu[i, j] >= l
            if not np.any(active_u):
                continue

            courmax = 0.0
            mudt[1] = mu[1, j, l] * dt
            mudt[2] = mu[2, j, l] * dt
            mudt[IM] = mu[IM, j, l] * dt

            i = 1
            if lmu[i, j] >= l:
                if mudt[i] >= 0.0:
                    mcheck = mm[i, j, l] + min(0.0, mudt[IM] - mudt[i])
                else:
                    mcheck = -mm[i + 1, j, l] - min(0.0, mudt[i] - mudt[i + 1])
                courmax = max(courmax, mudt[i] / mcheck)

            u_segs = _get_i1i2(active_u)
            for (i1, i2) in u_segs:
                i = i1
                if i > 1:
                    mudt[i - 1] = 0.0
                mudt[i] = mu[i, j, l] * dt
                for i in range(max(2, i1), min(i2, IM - 1) + 1):
                    mudt[i + 1] = mu[i + 1, j, l] * dt
                    if mudt[i] >= 0.0:
                        mcheck = mm[i, j, l] + min(0.0, mudt[i - 1] - mudt[i])
                    else:
                        mcheck = -mm[i + 1, j, l] - min(0.0, mudt[i] - mudt[i + 1])
                    courmax = max(courmax, mudt[i] / mcheck)

            i = IM
            if lmu[i, j] >= l:
                if mudt[i] >= 0.0:
                    mcheck = mm[i, j, l] + min(0.0, mudt[i - 1] - mudt[i])
                else:
                    mcheck = -mm[1, j, l] - min(0.0, mudt[i] - mudt[1])
                courmax = max(courmax, mudt[i] / mcheck)

            if courmax > 1.0:
                ncourant = 1 + int(courmax)
                zcourant = 1.0 / ncourant
                for (i1, i2) in u_segs:
                    for i in range(i1, i2 + 1):
                        mudt[i] *= zcourant
            else:
                ncourant = 1

            active_m = np.zeros(IM + 1, dtype=bool)
            for i in range(1, IM + 1):
                active_m[i] = lmm[i, j] >= l
            m_segs = _get_i1i2(active_m)

            for _nc in range(ncourant):
                # dateline flux (I=IM), computed first -- seeds the sequential sweep
                i = IM
                if lmu[i, j] >= l:
                    am = mudt[i]
                    if am >= 0.0:
                        a = am / mm[i, j, l]
                        rx[i, j, l] -= rxlimit * _sign(min(0.0, rm[i, j, l] - abs(rx[i, j, l])), rx[i, j, l])
                        fm = a * (rm[i, j, l] + (1 - a) * rx[i, j, l])
                        fx = am * (a * a * rx[i, j, l] - 3 * fm)
                        fy = a * ry[i, j, l]
                        fz = a * rz[i, j, l]
                    else:
                        a = am / mm[1, j, l]
                        rx[1, j, l] -= rxlimit * _sign(min(0.0, rm[1, j, l] - abs(rx[1, j, l])), rx[1, j, l])
                        fm = a * (rm[1, j, l] - (1 + a) * rx[1, j, l])
                        fx = am * (a * a * rx[1, j, l] - 3 * fm)
                        fy = a * ry[1, j, l]
                        fz = a * rz[1, j, l]
                else:
                    am = fm = fx = fy = fz = 0.0
                amim1, fmim1, fxim1, fyim1, fzim1 = am, fm, fx, fy, fz
                am_im, fm_im, fx_im, fy_im, fz_im = am, fm, fx, fy, fz

                # basins loop: M-active segments, capped below IM (dateline handled separately),
                # single-cell segments with i1>1 skipped entirely (OCNDYN2.f:2066)
                for (i1, i2) in m_segs:
                    if i1 > 1 and i1 == i2:
                        continue
                    for i in range(i1, min(i2, IM - 1) + 1):
                        am = mudt[i]
                        if am >= 0.0:
                            a = am / mm[i, j, l]
                            rx[i, j, l] -= rxlimit * _sign(min(0.0, rm[i, j, l] - abs(rx[i, j, l])), rx[i, j, l])
                            fm = a * (rm[i, j, l] + (1 - a) * rx[i, j, l])
                            fx = am * (a * a * rx[i, j, l] - 3 * fm)
                            fy = a * ry[i, j, l]
                            fz = a * rz[i, j, l]
                        else:
                            a = am / mm[i + 1, j, l]
                            rx[i + 1, j, l] -= rxlimit * _sign(min(0.0, rm[i + 1, j, l] - abs(rx[i + 1, j, l])), rx[i + 1, j, l])
                            fm = a * (rm[i + 1, j, l] - (1 + a) * rx[i + 1, j, l])
                            fx = am * (a * a * rx[i + 1, j, l] - 3 * fm)
                            fy = a * ry[i + 1, j, l]
                            fz = a * rz[i + 1, j, l]
                        mmnew = mm[i, j, l] + (amim1 - am)
                        rm[i, j, l] = rm[i, j, l] + (fmim1 - fm)
                        rx[i, j, l] = (rx[i, j, l] * mm[i, j, l] + (fxim1 - fx) +
                                       3.0 * ((amim1 + am) * rm[i, j, l] - mm[i, j, l] * (fmim1 + fm))) / mmnew
                        ry[i, j, l] = ry[i, j, l] + (fyim1 - fy)
                        rz[i, j, l] = rz[i, j, l] + (fzim1 - fz)
                        mm[i, j, l] = mmnew
                        amim1, fmim1, fxim1, fyim1, fzim1 = am, fm, fx, fy, fz

                # dateline update, using the stored original dateline flux (am_im/fm_im/...)
                # against amim1/fmim1/... left over from the basins loop's last iteration
                i = IM
                if lmm[i, j] >= l:
                    am, fm, fx, fy, fz = am_im, fm_im, fx_im, fy_im, fz_im
                    mmnew = mm[i, j, l] + (amim1 - am)
                    rm[i, j, l] = rm[i, j, l] + (fmim1 - fm)
                    rx[i, j, l] = (rx[i, j, l] * mm[i, j, l] + (fxim1 - fx) +
                                   3.0 * ((amim1 + am) * rm[i, j, l] - mm[i, j, l] * (fmim1 + fm))) / mmnew
                    ry[i, j, l] = ry[i, j, l] + (fyim1 - fy)
                    rz[i, j, l] = rz[i, j, l] + (fzim1 - fz)
                    mm[i, j, l] = mmnew
    return rm, rx, ry, rz, mm


def oadvty2(rm, rx, ry, rz, mo, mv, dt, qlimit, lmm, lmv):
    """OCNDYN2.f:2130-2302. `mv` bound to SMV -- SMV(i,JM,l)==0 identically (D44's accumulation
    loop never touches j=JM), so the update loop below needs no j=JM special-casing: it naturally
    degenerates to a zero-flux update there, matching the Fortran exactly. Returns fresh
    (rm,rx,ry,rz,mo)."""
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy(); mo = mo.copy()
    rylimit = 1.0 if qlimit else 0.0
    for l in range(1, LMO + 1):
        # pole fill (copy row JM across longitude before this layer's Y sweep)
        for i in range(2, IM + 1):
            mo[i, JM, l] = mo[1, JM, l]
            rm[i, JM, l] = rm[1, JM, l]
            rz[i, JM, l] = rz[1, JM, l]
        for i in range(1, IM + 1):
            rx[i, JM, l] = 0.0
            ry[i, JM, l] = 0.0

        bmjm1 = np.zeros(IM + 1); fmjm1 = np.zeros(IM + 1)
        fxjm1 = np.zeros(IM + 1); fyjm1 = np.zeros(IM + 1); fzjm1 = np.zeros(IM + 1)

        j = 1
        for i in range(1, IM + 1):
            if lmv[i, j] < l:
                continue
            bm = mv[i, j, l] * dt
            if bm >= 0.0:
                b = bm / mo[i, j, l]
                ry[i, j, l] -= rylimit * _sign(min(0.0, rm[i, j, l] - abs(ry[i, j, l])), ry[i, j, l])
                fm = b * (rm[i, j, l] + (1 - b) * ry[i, j, l])
                fy = bm * (b * b * ry[i, j, l] - 3 * fm)
                fx = b * rx[i, j, l]
                fz = b * rz[i, j, l]
            else:
                b = bm / mo[i, j + 1, l]
                ry[i, j + 1, l] -= rylimit * _sign(min(0.0, rm[i, j + 1, l] - abs(ry[i, j + 1, l])), ry[i, j + 1, l])
                fm = b * (rm[i, j + 1, l] - (1 + b) * ry[i, j + 1, l])
                fy = bm * (b * b * ry[i, j + 1, l] - 3 * fm)
                fx = b * rx[i, j + 1, l]
                fz = b * rz[i, j + 1, l]
            bmjm1[i], fmjm1[i], fxjm1[i], fyjm1[i], fzjm1[i] = bm, fm, fx, fy, fz

        for j in range(2, JM + 1):  # j = 2 .. JM (inclusive)
            for i in range(1, IM + 1):
                if lmm[i, j] < l:
                    continue
                bm = mv[i, j, l] * dt
                if bm >= 0.0:
                    b = bm / mo[i, j, l]
                    ry[i, j, l] -= rylimit * _sign(min(0.0, rm[i, j, l] - abs(ry[i, j, l])), ry[i, j, l])
                    fm = b * (rm[i, j, l] + (1 - b) * ry[i, j, l])
                    fy = bm * (b * b * ry[i, j, l] - 3 * fm)
                    fx = b * rx[i, j, l]
                    fz = b * rz[i, j, l]
                else:
                    b = bm / mo[i, j + 1, l]
                    ry[i, j + 1, l] -= rylimit * _sign(min(0.0, rm[i, j + 1, l] - abs(ry[i, j + 1, l])), ry[i, j + 1, l])
                    fm = b * (rm[i, j + 1, l] - (1 + b) * ry[i, j + 1, l])
                    fy = bm * (b * b * ry[i, j + 1, l] - 3 * fm)
                    fx = b * rx[i, j + 1, l]
                    fz = b * rz[i, j + 1, l]
                mnew = mo[i, j, l] + (bmjm1[i] - bm)
                rm[i, j, l] = rm[i, j, l] + (fmjm1[i] - fm)
                ry[i, j, l] = (ry[i, j, l] * mo[i, j, l] + (fyjm1[i] - fy) +
                               3.0 * ((bmjm1[i] + bm) * rm[i, j, l] - mo[i, j, l] * (fmjm1[i] + fm))) / mnew
                rx[i, j, l] = rx[i, j, l] + (fxjm1[i] - fx)
                rz[i, j, l] = rz[i, j, l] + (fzjm1[i] - fz)
                mo[i, j, l] = mnew
                bmjm1[i], fmjm1[i], fxjm1[i], fyjm1[i], fzjm1[i] = bm, fm, fx, fy, fz

        # average the pole
        if lmm[1, JM] >= l:
            j = JM
            mo[1:IM + 1, j, l] = _fortran_sum(mo[1:IM + 1, j, l]) / IM
            rm[1:IM + 1, j, l] = _fortran_sum(rm[1:IM + 1, j, l]) / IM
            rz[1:IM + 1, j, l] = _fortran_sum(rz[1:IM + 1, j, l]) / IM
            rx[1:IM + 1, j, l] = 0.0
            ry[1:IM + 1, j, l] = 0.0
    return rm, rx, ry, rz, mo


def oadvtz2(rm, rx, ry, rz, mo, mw, dt, qlimit, lmm):
    """OCNDYN2.f:2304-2423. `mw` bound to SMW -- SMW(i,j,LMM(i,j))==0 identically (proven from
    source: OFLUXV's SMW accumulation loop, OCNDYN2.f:766-780, only ever assigns l=1..lm-1 per
    column via direct '=' assignment, and SMW is zero at allocation, `OCEAN_COM.f:505`, never
    touched anywhere else -- so the column's own bottom-layer entry is permanently, structurally
    zero, not just usually so). This makes the l==LMO / cm<0 out-of-bounds case (which would read
    RM(...,L+1) at L+1=LMO+1) unreachable: whenever l==LMO for an active cell, l==lmm(i,j)==LMO,
    so cm=dt*mw[...,LMO]=0 exactly, taking the cm>=0 branch, never the cm<0 one. `LUS_VERT_ADV`
    is confirmed not defined for this rundeck (the preprocessor block), so the R_edge-weighted
    `fm_ctr` path is always the live one (never just `fm_ctr=fm`).

    `nbyzm` restricts J=JM (the North Pole row) to I=1 only (D40's established finding, reused
    throughout D40/42/43/44 as the `m_active` override) -- OADVTZ2's `cmup`/`fmup`/... arrays are
    PERSISTENT per-(i,j) state carried across layers, so processing every I at J=JM pointwise via
    `lmm[i,JM]` (rather than restricting to I=1) lets each I independently accumulate its OWN
    cmup/fmup history, while the real Fortran leaves I=2..IM's cmup/fmup frozen at 0 forever
    (never touched) -- a real, accumulating-with-depth divergence found by isolating D45's
    residual MA mismatch to exactly this routine, at exactly the pole row. Returns fresh
    (rm,rx,ry,rz,mo)."""
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy(); mo = mo.copy()
    rzlim = 1.0 if qlimit else 0.0
    no_rzlim = 1.0 - rzlim
    edgmax = 1.5

    cmup = np.zeros((IM + 1, JM + 1)); fmup = np.zeros((IM + 1, JM + 1))
    fxup = np.zeros((IM + 1, JM + 1)); fyup = np.zeros((IM + 1, JM + 1))
    fzup = np.zeros((IM + 1, JM + 1)); fmup_ctr = np.zeros((IM + 1, JM + 1))

    def m_active(i, j, l):
        if j == JM:
            return i == 1 and lmm[1, JM] >= l
        return lmm[i, j] >= l

    for l in range(1, LMO + 1):
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if not m_active(i, j, l):
                    continue
                cm = dt * mw[i, j, l]
                ldn = min(l + 1, lmm[i, j])
                wtdn = mo[i, j, l] / (mo[i, j, l] + mo[i, j, ldn])
                rdn = rm[i, j, ldn] / mo[i, j, ldn]
                rup = rm[i, j, l] / mo[i, j, l]
                r_edge = wtdn * rdn + (1.0 - wtdn) * rup
                if cm >= 0.0:
                    c = cm / mo[i, j, l]
                    rz[i, j, l] -= rzlim * _sign(min(0.0, rm[i, j, l] - abs(rz[i, j, l])), rz[i, j, l])
                    fm = c * (rm[i, j, l] + (1.0 - c) * rz[i, j, l])
                    fx = c * rx[i, j, l]
                    fy = c * ry[i, j, l]
                    fz = cm * (c * c * rz[i, j, l] - 3.0 * fm)
                    r_edge = r_edge * no_rzlim + rzlim * min(r_edge, edgmax * rup)
                    fm_ctr = cm * (c * rup + (1.0 - c) * r_edge)
                else:
                    c = cm / mo[i, j, l + 1]
                    rz[i, j, l + 1] -= rzlim * _sign(min(0.0, rm[i, j, l + 1] - abs(rz[i, j, l + 1])), rz[i, j, l + 1])
                    fm = c * (rm[i, j, l + 1] - (1.0 + c) * rz[i, j, l + 1])
                    fx = c * rx[i, j, l + 1]
                    fy = c * ry[i, j, l + 1]
                    fz = cm * (c * c * rz[i, j, l + 1] - 3.0 * fm)
                    r_edge = r_edge * no_rzlim + rzlim * min(r_edge, edgmax * rdn)
                    fm_ctr = cm * (-c * rdn + (1.0 + c) * r_edge)

                rm_lus = rm[i, j, l] + (fmup[i, j] - fm)
                rm[i, j, l] = rm[i, j, l] + (fmup_ctr[i, j] - fm_ctr)
                mnew = mo[i, j, l] + cmup[i, j] - cm
                rz[i, j, l] = (rz[i, j, l] * mo[i, j, l] + (fzup[i, j] - fz) + 3.0 *
                               ((cmup[i, j] + cm) * rm_lus - mo[i, j, l] * (fmup[i, j] + fm))) / mnew
                mo[i, j, l] = mnew
                cmup[i, j] = cm
                fmup[i, j] = fm
                fmup_ctr[i, j] = fm_ctr
                fzup[i, j] = fz
                rx[i, j, l] = rx[i, j, l] + (fxup[i, j] - fx)
                fxup[i, j] = fx
                ry[i, j, l] = ry[i, j, l] + (fyup[i, j] - fy)
                fyup[i, j] = fy
    return rm, rx, ry, rz, mo


def oadvt2(mmi, rm, rx, ry, rz, dt, qlimit, smu, smv, smw, lmu, lmv, lmm):
    """OCNDYN2.f:1853-1914, the Strang-splitting dispatcher: X(half dt), Y(dt), Z(dt), X(half dt)
    again, then fills the pole row's RM/RZ uniformly across longitude. `mmi` seeds MA (overwrites
    whatever mass state the caller carried in, matching `MA = MB` in the real Fortran). Returns
    fresh (ma, rm, rx, ry, rz)."""
    ma = mmi.copy()
    rm = rm.copy(); rx = rx.copy(); ry = ry.copy(); rz = rz.copy()
    rm, rx, ry, rz, ma = oadvtx2(rm, rx, ry, rz, ma, smu, 0.5 * dt, qlimit, lmu, lmm)
    rm, rx, ry, rz, ma = oadvty2(rm, rx, ry, rz, ma, smv, dt, qlimit, lmm, lmv)
    rm, rx, ry, rz, ma = oadvtz2(rm, rx, ry, rz, ma, smw, dt, qlimit, lmm)
    rm, rx, ry, rz, ma = oadvtx2(rm, rx, ry, rz, ma, smu, 0.5 * dt, qlimit, lmu, lmm)

    for l in range(1, LMO + 1):
        for i in range(1, IM + 1):
            rm[i, JM, l] = rm[1, JM, l]
            rz[i, JM, l] = rz[1, JM, l]
    return ma, rm, rx, ry, rz
