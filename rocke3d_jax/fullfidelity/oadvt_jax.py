"""JAX OADVT2 sweeps (Stage 2, D84): oadvt_vec's X/Y/Z sweeps under jax.jit.

Structure follows oadvt_vec: Y is sequential in j (lax.fori_loop over j = 2..JM, after a j = 1 step),
Z is sequential in l (13 unrolled layers), X is sequential in i (lax.fori_loop over i, wrapped in a loop
over courant sub-steps) with all (l, j) passes as lanes. The X MUDT/NCOURANT/skip pre-pass depends on
the input MM and is still computed in numpy (oadvt_vec._x_prepass), so `oadvt2_jax` is a thin Python
driver around three jitted sweeps. Arrays are 1-based (IM+1, JM+1, LMO+1). Validated against the scalar
port and the real Fortran dumps by oadvt_jax_compare.py.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from oadvt2_ff import IM, JM, LMO
from oadvt_vec import _x_prepass


def _sign(a, b):
    return jnp.where(b != 0.0, jnp.copysign(a, b), jnp.abs(a))


# ----------------------------------------------------------------------------------------- X sweep
_LS, _JS = np.meshgrid(np.arange(1, LMO + 1), np.arange(2, JM), indexing="ij")
_LS, _JS = _LS.ravel(), _JS.ravel()                 # all (l, j) passes as lanes, static shape
_N = len(_LS)


@jax.jit
def _x_sweep(rm, rx, ry, rz, mm, mudt, nc, lane_pass, lmu_l, lmm_l, rxlimit):
    """mudt (N, IM+1), nc (N,), lane_pass (N,) bool, lmu_l / lmm_l (N, IM+1) bool activity of the
    pass's cells. rm..mm are (IM+1, JM+1, LMO+1)."""
    pull = lambda a: a[:, _JS, _LS].T
    R, X, Y, Z, M = pull(rm), pull(rx), pull(ry), pull(rz), pull(mm)
    nc = jnp.where(lane_pass, nc, 0)
    lm_m = lmm_l.at[:, 0].set(False)
    left = jnp.zeros_like(lm_m).at[:, 2:].set(lm_m[:, 1:-1])
    right = jnp.zeros_like(lm_m).at[:, 1:-1].set(lm_m[:, 2:])
    single = (lm_m & ~left & ~right).at[:, 1].set(False)
    proc = (lm_m & ~single).at[:, IM].set(False)
    rows = jnp.arange(_N)

    def flux(R, X, Y, Z, M, am, c_pos, c_neg, act):
        pos = am >= 0.0
        cell = jnp.where(pos, c_pos, c_neg)
        rm_c, rx_c = R[rows, cell], X[rows, cell]
        a = am / M[rows, cell]
        lim = rx_c - rxlimit * _sign(jnp.minimum(0.0, rm_c - jnp.abs(rx_c)), rx_c)
        X = X.at[rows, cell].set(jnp.where(act, lim, rx_c))
        rx_c = jnp.where(act, lim, rx_c)
        fm = jnp.where(pos, a * (rm_c + (1 - a) * rx_c), a * (rm_c - (1 + a) * rx_c))
        fx = am * (a * a * rx_c - 3 * fm)
        fy = a * Y[rows, cell]
        fz = a * Z[rows, cell]
        return X, (fm, fx, fy, fz)

    def update(R, X, Y, Z, M, i, am, fm, fx, fy, fz, amim1, fmim1, fxim1, fyim1, fzim1, act):
        mi, ri, xi = M[rows, i], R[rows, i], X[rows, i]
        yi, zi = Y[rows, i], Z[rows, i]
        mmnew = mi + (amim1 - am)
        rnew = ri + (fmim1 - fm)
        xnew = (xi * mi + (fxim1 - fx) + 3.0 * ((amim1 + am) * rnew - mi * (fmim1 + fm))) / mmnew
        R = R.at[rows, i].set(jnp.where(act, rnew, ri))
        X = X.at[rows, i].set(jnp.where(act, xnew, xi))
        Y = Y.at[rows, i].set(jnp.where(act, yi + (fyim1 - fy), yi))
        Z = Z.at[rows, i].set(jnp.where(act, zi + (fzim1 - fz), zi))
        M = M.at[rows, i].set(jnp.where(act, mmnew, mi))
        return R, X, Y, Z, M

    IMv = jnp.full(_N, IM)
    ones = jnp.ones(_N, int)

    def substep(k, st):
        R, X, Y, Z, M = st
        lane = nc > k
        du = lane & lmu_l[:, IM]
        am = jnp.where(du, mudt[:, IM], 0.0)
        X, (fm, fx, fy, fz) = flux(R, X, Y, Z, M, am, IMv, ones, du)
        dl = tuple(jnp.where(du, v, 0.0) for v in (am, fm, fx, fy, fz))

        def body(i, c):
            R, X, Y, Z, M, prev = c
            act = lane & proc[:, i]
            ii = jnp.full(_N, i)
            am = mudt[:, i]
            X, fl = flux(R, X, Y, Z, M, am, ii, ii + 1, act)
            R, X, Y, Z, M = update(R, X, Y, Z, M, ii, am, *fl, *prev, act)
            prev = tuple(jnp.where(act, v, p) for v, p in zip((am,) + tuple(fl), prev))
            return R, X, Y, Z, M, prev

        R, X, Y, Z, M, prev = lax.fori_loop(1, IM, body, (R, X, Y, Z, M, dl))
        act = lane & lm_m[:, IM]
        R, X, Y, Z, M = update(R, X, Y, Z, M, IMv, *dl, *prev, act)
        return R, X, Y, Z, M

    R, X, Y, Z, M = lax.fori_loop(0, jnp.max(nc), substep, (R, X, Y, Z, M))
    push = lambda a, A: a.at[:, _JS, _LS].set(A.T)
    return push(rm, R), push(rx, X), push(ry, Y), push(rz, Z), push(mm, M)


def oadvtx2_jax(rm, rx, ry, rz, mm, mu, dt, qlimit, lmu, lmm):
    mm_np = np.asarray(mm)
    snap, ncour, lane_all = _x_prepass(mm_np, np.asarray(mu), dt, lmu, lmm)
    ls, js = _LS, _JS
    lmu_l = (lmu[:, js] >= ls[None, :]).T
    lmm_l = (lmm[:, js] >= ls[None, :]).T
    return _x_sweep(jnp.asarray(rm), jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(rz),
                    jnp.asarray(mm), jnp.asarray(snap[ls, js]), jnp.asarray(ncour[ls, js]),
                    jnp.asarray(lane_all[ls, js]), jnp.asarray(lmu_l), jnp.asarray(lmm_l),
                    1.0 if qlimit else 0.0)


# ----------------------------------------------------------------------------------------- Y sweep
@jax.jit
def _y_sweep(rm, rx, ry, rz, mo, mv, dt, rylimit, lmm, lmv):
    lidx = jnp.arange(1, LMO + 1)[None, :]
    SI = slice(1, IM + 1)
    for a_name in ("mo", "rm", "rz"):
        a = {"mo": mo, "rm": rm, "rz": rz}[a_name]
        a = a.at[2:IM + 1, JM, 1:].set(a[1, JM, 1:][None, :])
        if a_name == "mo": mo = a
        elif a_name == "rm": rm = a
        else: rz = a
    rx = rx.at[SI, JM, 1:].set(0.0)
    ry = ry.at[SI, JM, 1:].set(0.0)

    def sweep(j, act, rm, rx, ry, rz, mo, prev, store_only):
        jp = jnp.minimum(j + 1, JM)
        bm = mv[SI, j, 1:] * dt
        pos = (bm >= 0.0) | (j == JM)
        neg = ~pos
        ry_j, ry_p = ry[SI, j, 1:], ry[SI, jp, 1:]
        ry_s = jnp.where(pos, ry_j, ry_p)
        rm_s = jnp.where(pos, rm[SI, j, 1:], rm[SI, jp, 1:])
        lim = ry_s - rylimit * _sign(jnp.minimum(0.0, rm_s - jnp.abs(ry_s)), ry_s)
        ry = ry.at[SI, j, 1:].set(jnp.where(act & pos, lim, ry_j))
        ry = ry.at[SI, jp, 1:].set(jnp.where(act & neg & (jp != j), lim, ry[SI, jp, 1:]))
        ry_s = jnp.where(act, lim, ry_s)
        mo_s = jnp.where(pos, mo[SI, j, 1:], mo[SI, jp, 1:])
        b = bm / mo_s
        fm = jnp.where(pos, b * (rm_s + (1 - b) * ry_s), b * (rm_s - (1 + b) * ry_s))
        fy = bm * (b * b * ry_s - 3 * fm)
        fx = b * jnp.where(pos, rx[SI, j, 1:], rx[SI, jp, 1:])
        fz = b * jnp.where(pos, rz[SI, j, 1:], rz[SI, jp, 1:])
        bmp, fmp, fxp, fyp, fzp = prev
        if not store_only:
            mo_j, rm_j, ry_j, rx_j, rz_j = (mo[SI, j, 1:], rm[SI, j, 1:], ry[SI, j, 1:],
                                            rx[SI, j, 1:], rz[SI, j, 1:])
            mnew = mo_j + (bmp - bm)
            rm_new = rm_j + (fmp - fm)
            ry_new = (ry_j * mo_j + (fyp - fy) +
                      3.0 * ((bmp + bm) * rm_new - mo_j * (fmp + fm))) / mnew
            rx = rx.at[SI, j, 1:].set(jnp.where(act, rx_j + (fxp - fx), rx_j))
            rz = rz.at[SI, j, 1:].set(jnp.where(act, rz_j + (fzp - fz), rz_j))
            rm = rm.at[SI, j, 1:].set(jnp.where(act, rm_new, rm_j))
            ry = ry.at[SI, j, 1:].set(jnp.where(act, ry_new, ry_j))
            mo = mo.at[SI, j, 1:].set(jnp.where(act, mnew, mo_j))
        prev = tuple(jnp.where(act, v, p) for v, p in zip((bm, fm, fx, fy, fz), prev))
        return rm, rx, ry, rz, mo, prev

    z = jnp.zeros((IM, LMO))
    prev = (z, z, z, z, z)
    act1 = lmv[SI, 1][:, None] >= lidx
    rm, rx, ry, rz, mo, prev = sweep(1, act1, rm, rx, ry, rz, mo, prev, True)

    def body(j, c):
        rm, rx, ry, rz, mo, prev = c
        act = lmm[SI, j][:, None] >= lidx
        return sweep(j, act, rm, rx, ry, rz, mo, prev, False)

    rm, rx, ry, rz, mo, prev = lax.fori_loop(2, JM + 1, body, (rm, rx, ry, rz, mo, prev))

    pole = lmm[1, JM] >= jnp.arange(1, LMO + 1)
    out = []
    for a in (mo, rm, rz):
        avg = jnp.cumsum(a[SI, JM, 1:], axis=0)[-1] / IM
        out.append(a.at[SI, JM, 1:].set(jnp.where(pole[None, :], avg[None, :], a[SI, JM, 1:])))
    mo, rm, rz = out
    rx = rx.at[SI, JM, 1:].set(jnp.where(pole[None, :], 0.0, rx[SI, JM, 1:]))
    ry = ry.at[SI, JM, 1:].set(jnp.where(pole[None, :], 0.0, ry[SI, JM, 1:]))
    return rm, rx, ry, rz, mo


def oadvty2_jax(rm, rx, ry, rz, mo, mv, dt, qlimit, lmm, lmv):
    return _y_sweep(*(jnp.asarray(a) for a in (rm, rx, ry, rz, mo, mv)), dt,
                    1.0 if qlimit else 0.0, jnp.asarray(lmm), jnp.asarray(lmv))


# ----------------------------------------------------------------------------------------- Z sweep
@jax.jit
def _z_sweep(rm, rx, ry, rz, mo, mw, dt, rzlim, lmm):
    no_rzlim = 1.0 - rzlim
    edgmax = 1.5
    shp = (IM + 1, JM + 1)
    cmup, fmup, fxup, fyup, fzup, fmup_ctr = [jnp.zeros(shp) for _ in range(6)]
    I = jnp.arange(IM + 1)[:, None]
    J = jnp.arange(JM + 1)[None, :]
    at = lambda a, idx: jnp.take_along_axis(a, idx[:, :, None], axis=2)[:, :, 0]
    for l in range(1, LMO + 1):
        lp = min(l + 1, LMO)
        M = jnp.where(J == JM, (I == 1) & (lmm[1, JM] >= l), lmm >= l)
        M = M.at[0, :].set(False).at[:, 0].set(False)
        mo_l, rm_l = mo[:, :, l], rm[:, :, l]
        cm = dt * mw[:, :, l]
        ldn = jnp.minimum(l + 1, lmm)
        mo_dn, rm_dn = at(mo, ldn), at(rm, ldn)
        wtdn = mo_l / (mo_l + mo_dn)
        rdn = rm_dn / mo_dn
        rup = rm_l / mo_l
        r_edge0 = wtdn * rdn + (1.0 - wtdn) * rup
        pos = cm >= 0.0
        neg = ~pos
        rz_l, rz_n = rz[:, :, l], rz[:, :, lp]
        rz_s = jnp.where(pos, rz_l, rz_n)
        rm_s = jnp.where(pos, rm_l, rm[:, :, lp])
        lim = rz_s - rzlim * _sign(jnp.minimum(0.0, rm_s - jnp.abs(rz_s)), rz_s)
        rz = rz.at[:, :, l].set(jnp.where(M & pos, lim, rz_l))
        if lp != l:
            rz = rz.at[:, :, lp].set(jnp.where(M & neg, lim, rz_n))
        rz_s = jnp.where(M, lim, rz_s)
        mo_s = jnp.where(pos, mo_l, mo[:, :, lp])
        c = cm / mo_s
        fm = jnp.where(pos, c * (rm_s + (1.0 - c) * rz_s), c * (rm_s - (1.0 + c) * rz_s))
        fx = c * jnp.where(pos, rx[:, :, l], rx[:, :, lp])
        fy = c * jnp.where(pos, ry[:, :, l], ry[:, :, lp])
        fz = cm * (c * c * rz_s - 3.0 * fm)
        r_edge = jnp.where(pos,
                           r_edge0 * no_rzlim + rzlim * jnp.minimum(r_edge0, edgmax * rup),
                           r_edge0 * no_rzlim + rzlim * jnp.minimum(r_edge0, edgmax * rdn))
        fm_ctr = jnp.where(pos, cm * (c * rup + (1.0 - c) * r_edge),
                           cm * (-c * rdn + (1.0 + c) * r_edge))
        mo_l, rm_l, rz_l = mo[:, :, l], rm[:, :, l], rz[:, :, l]
        rm_lus = rm_l + (fmup - fm)
        rm_new = rm_l + (fmup_ctr - fm_ctr)
        mnew = mo_l + cmup - cm
        rz_new = (rz_l * mo_l + (fzup - fz) + 3.0 *
                  ((cmup + cm) * rm_lus - mo_l * (fmup + fm))) / mnew
        rm = rm.at[:, :, l].set(jnp.where(M, rm_new, rm_l))
        rz = rz.at[:, :, l].set(jnp.where(M, rz_new, rz_l))
        mo = mo.at[:, :, l].set(jnp.where(M, mnew, mo_l))
        rx = rx.at[:, :, l].set(jnp.where(M, rx[:, :, l] + (fxup - fx), rx[:, :, l]))
        ry = ry.at[:, :, l].set(jnp.where(M, ry[:, :, l] + (fyup - fy), ry[:, :, l]))
        cmup = jnp.where(M, cm, cmup)
        fmup = jnp.where(M, fm, fmup)
        fmup_ctr = jnp.where(M, fm_ctr, fmup_ctr)
        fzup = jnp.where(M, fz, fzup)
        fxup = jnp.where(M, fx, fxup)
        fyup = jnp.where(M, fy, fyup)
    return rm, rx, ry, rz, mo


def oadvtz2_jax(rm, rx, ry, rz, mo, mw, dt, qlimit, lmm):
    return _z_sweep(*(jnp.asarray(a) for a in (rm, rx, ry, rz, mo, mw)), dt,
                    1.0 if qlimit else 0.0, jnp.asarray(lmm))


def oadvt2_jax(mmi, rm, rx, ry, rz, dt, qlimit, smu, smv, smw, lmu, lmv, lmm):
    """OADVT2 with the X, Y and Z sweeps jitted (X pre-pass in numpy)."""
    rm, rx, ry, rz, ma = oadvtx2_jax(rm, rx, ry, rz, mmi, smu, 0.5 * dt, qlimit, lmu, lmm)
    rm, rx, ry, rz, ma = oadvty2_jax(rm, rx, ry, rz, ma, smv, dt, qlimit, lmm, lmv)
    rm, rx, ry, rz, ma = oadvtz2_jax(rm, rx, ry, rz, ma, smw, dt, qlimit, lmm)
    rm, rx, ry, rz, ma = oadvtx2_jax(rm, rx, ry, rz, ma, smu, 0.5 * dt, qlimit, lmu, lmm)
    rm = rm.at[1:IM + 1, JM, 1:].set(rm[1, JM, 1:][None, :])
    rz = rz.at[1:IM + 1, JM, 1:].set(rz[1, JM, 1:][None, :])
    return ma, rm, rx, ry, rz
