"""Batched (all columns at once, jit-able, device-resident) radiance_calc band computation for this rundeck: the validated pieces of
R3-R5 rewritten without per-column Python control flow.

Differences from the validation chain (all shown to be rounding-level by r2_batch_check.py):
  * n_cloud_top = 1 for every column (r2_e2e_check.py --nct1: identical to the Fortran per-call cloud top);
  * the major absorber is chosen per column and the absorbers are permuted with a gather; k-term axes are padded to the largest k-term count
    of the band (weight 0, so padded terms contribute exactly 0);
  * sums over k-terms are jnp sums / cumulative products over layers (not the Fortran's sequential order);
  * recurrences over layers are unrolled Python loops (43 layers) or lax.scan.
Inputs: dict of device arrays with a leading column axis N (see `columns_from_calls`), static spectral data from spec_data.load.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import kernels as K
import gas_absorption as GA
import planck as PL
from solver_triple_hogan import solver_triple_hogan

TINY = np.finfo(np.float64).tiny
EPS = np.finfo(np.float64).eps
TEMP_MAX = float(np.log(1.0 / EPS))
TRANS_DEF_MAJOR = float(np.exp(-1.0))
DIFF_MINOR = 1.66
MOL_WEIGHT_AIR = 28.966e-03
REPSILON = 1.0 / (28.966 / 18.015)
R_NCT1 = 1


# ---------------------------------------------------------------- grey optical properties (grey_opt_prop.py, jnp)
def _cloud_comp(kind, lwc, r, p, rel_var, n_order_forward=2):
    if kind == "water":
        k_ext = lwc * (p[0] + r * (p[1] + r * p[2])) / (1.0 + r * (p[3] + r * (p[4] + r * p[5])))
        k_sc = k_ext * (1.0 - (p[6] + r * (p[7] + r * p[8])) / (1.0 + r * (p[9] + r * p[10])))
        g = (p[11] + r * (p[12] + r * p[13])) / (1.0 + r * (p[14] + r * p[15]))
    else:
        x = p[3] / r
        k_ext = lwc * ((p[2] * x + p[1]) * x + p[0])
        x = r / p[8]
        k_sc = k_ext * (1.0 - (p[4] + x * (p[5] + x * (p[6] + x * p[7]))))
        x = r / p[13]
        g = p[9] + x * (p[10] + x * (p[11] + x * p[12]))
    ph = k_sc * g
    fwd = ph * g
    corr = 1.0 / (1.0 + rel_var)
    ok = k_sc > TINY
    asym = ph / jnp.where(ok, k_sc, 1.0)
    omega = k_sc / jnp.where(ok, k_ext, 1.0)
    om_corr = 1.0 / (1.0 + rel_var * (1.0 - omega))
    as_corr = (1.0 + rel_var * (1.0 - omega)) / (1.0 + rel_var * (1.0 - omega * asym))
    k_sc2 = k_sc * corr * om_corr
    ph2 = ph * corr * om_corr * as_corr
    fwd2 = fwd * corr * om_corr * as_corr ** n_order_forward
    return k_ext * corr, jnp.where(ok, k_sc2, k_sc), jnp.where(ok, ph2, ph), jnp.where(ok, fwd2, fwd)


def grey_props(iso, rc, mode_mix, mode_abs, mode_scat, mode_asym, n_mode, cont_terms, comps, w_cloud, n_type=4):
    """-> kg, kes, ph, fwd each (N, nl, n_type+1), normalised.  cont_terms: list of (k_cont, amount (N, nl))."""
    kes = jnp.full(w_cloud.shape, rc if iso == 1 else 0.0)
    kg = jnp.zeros_like(kes); ph = jnp.zeros_like(kes); fwd = jnp.zeros_like(kes)
    for j in range(n_mode):
        mix, a, s, g = mode_mix[:, :, j], mode_abs[:, :, j], mode_scat[:, :, j], mode_asym[:, :, j]
        kg = kg + mix * a
        ks = mix * s
        kes = kes + ks
        ph = ph + ks * g
        fwd = fwd + (ks * g) * g
    for kc, am in cont_terms:
        kg = kg + kc * am
    kg = kg + kes
    cl = w_cloud > 0.0
    outs = [(kg, kes, ph, fwd)]
    for k in range(1, n_type + 1):
        a, b, c, d = kg, kes, ph, fwd
        for (ct, kind, lwc, r, param, rv) in comps:
            if ct != k:
                continue
            e, s, p1, f1 = _cloud_comp(kind, lwc, r, param, rv)
            a = jnp.where(cl, a + e, a); b = jnp.where(cl, b + s, b); c = jnp.where(cl, c + p1, c); d = jnp.where(cl, d + f1, d)
        outs.append((a, b, c, d))
    KG, KES, PH, FW = [], [], [], []
    for (a, b, c, d) in outs:
        ok = b > TINY
        inv = 1.0 / jnp.where(ok, b, 1.0)
        KG.append(a); KES.append(b)
        PH.append(jnp.where(ok, c * inv, c)); FW.append(jnp.where(ok, d * inv, d))
    st = lambda L: jnp.stack(L, axis=2)
    return st(KG), st(KES), st(PH), st(FW)


# ---------------------------------------------------------------- cloud geometry (overlap_coupled, ip_cloud_triple, n_cloud_top = 1)
def overlap_triple(w_cloud, fr):
    """w_cloud (N, nl); fr (N, nl, 3) region fractions (clear unused).  Returns co (N, nl+1, 18), w_free (N, nl)."""
    N, nl = w_cloud.shape
    tol = 1.0e2 * EPS
    w_free = 1.0 - w_cloud
    zeros = jnp.zeros(N)
    area_upper = [jnp.ones(N), zeros, zeros]
    co = []
    for i in range(0, nl + 1):                                   # Fortran layer index i = nct-1 .. nl with nct = 1
        if i < nl:
            area_lower = [w_free[:, i], w_cloud[:, i] * fr[:, i, 1], w_cloud[:, i] * fr[:, i, 2]]
        else:
            area_lower = [jnp.ones(N), zeros, zeros]
        ao = [[zeros] * 3 for _ in range(3)]
        for k in range(3):
            ao[k][k] = jnp.minimum(area_lower[k], area_upper[k])
        art = 1.0 - ao[0][0]
        art = art - ao[1][1]
        art = art - ao[2][2]
        ar_up = [area_upper[k] - ao[k][k] for k in range(3)]
        ar_lo = [area_lower[k] - ao[k][k] for k in range(3)]
        ok = art > tol
        for k in range(3):
            for j in range(3):
                ao[k][j] = jnp.where(ok, ao[k][j] + ar_up[k] * ar_lo[j] / jnp.where(ok, art, 1.0), ao[k][j])
        c = [jnp.zeros(N) for _ in range(18)]
        for k in range(3):
            c[3 * k + k] = jnp.ones(N)
            c[3 * (3 + k) + k] = jnp.ones(N)
        for k in range(3):
            oku = area_upper[k] > tol
            okl = area_lower[k] > tol
            for j in range(3):
                c[3 * j + k] = jnp.where(oku, ao[k][j] / jnp.where(oku, area_upper[k], 1.0), c[3 * j + k])
                c[3 * (3 + j) + k] = jnp.where(okl, ao[j][k] / jnp.where(okl, area_lower[k], 1.0), c[3 * (3 + j) + k])
        co.append(jnp.stack(c, axis=1))
        if i < nl:
            area_upper = list(area_lower)
    return jnp.stack(co, axis=1), w_free


# ---------------------------------------------------------------- gas absorption with per-column major absorber
def band_gas(spec, ib, p, t, mix, mass):
    """Returns kperm (N, nl, NK, n_gas), wperm (N, NK, n_gas), nkperm (N, n_gas) with the major absorber of each column first."""
    n = int(spec["n_band_absorb"][ib - 1])
    gases = [int(spec["index_absorb"][j, ib - 1]) for j in range(n)]
    nks = [int(spec["i_band_k"][ib - 1, g - 1]) for g in gases]
    NK = max(nks)
    ip = GA.inter_pt_lookup(p, t, spec["p_lookup"], spec["t_lookup"])
    N, nl = p.shape
    slots, ws = [], []
    for j, g in enumerate(gases):
        nk = nks[j]
        m = mix[:, :, g - 1]
        fn = int(spec["i_scale_fnc"][ib - 1, g - 1])
        if fn == GA.IP_SCALE_LOOKUP:
            ke = GA.k_esft_lookup(ip, spec["k_lookup"], ib, g, nk)
            kk = jnp.maximum(0.0, ke * m[:, :, None])
        elif fn == 2:
            cols = []
            for k in range(nk):
                fr = GA.scale_absorb_power_quad(p, t, m, spec["scale"][:3, k, ib - 1, g - 1], float(spec["p_ref"][g - 1, ib - 1]),
                                                float(spec["t_ref"][g - 1, ib - 1]))
                cols.append(spec["k"][k, ib - 1, g - 1] * fr)
            kk = jnp.stack(cols, axis=2)
        else:
            raise NotImplementedError(fn)
        slots.append(jnp.pad(kk, ((0, 0), (0, 0), (0, NK - nk))))
        w = np.zeros(NK); w[:nk] = spec["w"][:nk, ib - 1, g - 1]
        ws.append(w)
    kslots = jnp.stack(slots, axis=3)                              # (N, nl, NK, n)
    wslots = jnp.asarray(np.stack(ws, axis=1))                     # (NK, n)
    nkv = jnp.asarray(np.array(nks))
    # major absorber: column transmission test (planet_rad.F90 2780-2842), per column
    colm = [jnp.zeros(N) for _ in range(n)]
    trans = [jnp.ones(N) for _ in range(n)]
    done = jnp.zeros(N, dtype=bool)
    for lay in range(nl):
        nc, nt = [], []
        for j, g in enumerate(gases):
            nk = nks[j]
            c = colm[j] + mix[:, lay, g - 1] * mass[:, lay]
            w = jnp.asarray(spec["w"][:nk, ib - 1, g - 1]); k = jnp.asarray(spec["k"][:nk, ib - 1, g - 1])
            tr = jnp.sum(w[None, :] * jnp.exp(-k[None, :] * c[:, None]), axis=1)
            nc.append(c); nt.append(tr)
        upd = ~done
        colm = [jnp.where(upd, nc[j], colm[j]) for j in range(n)]
        trans = [jnp.where(upd, nt[j], trans[j]) for j in range(n)]
        prod = trans[0]
        for j in range(1, n):
            prod = prod * trans[j]
        done = done | (prod <= TRANS_DEF_MAJOR)
    nmaj = jnp.argmin(jnp.stack(trans, axis=1), axis=1)            # (N,)
    ar = jnp.arange(n)[None, :]
    order = jnp.where(ar == 0, nmaj[:, None], jnp.where(ar == nmaj[:, None], 0, ar))      # (N, n)
    kperm = jnp.take_along_axis(kslots, order[:, None, None, :], axis=3)
    wperm = jnp.take_along_axis(jnp.broadcast_to(wslots[None], (N,) + wslots.shape), order[:, None, :], axis=2)
    nkperm = nkv[order]
    return kperm, wperm, nkperm


def k_eqv_solar(kperm, wperm, nkperm, dmass, zen):
    N, nl, NK, n = kperm.shape
    k_eqv = jnp.zeros((N, nl)); adj = jnp.ones((N, nl))
    kidx = jnp.arange(NK)
    for j in range(1, n):
        k = kperm[..., j]; w = wperm[:, :, j]
        valid = kidx[None, :] < nkperm[:, j][:, None]
        tr = jnp.exp(-k * dmass[:, :, None] * zen[:, None, None])
        ft = w[:, None, :] * jnp.cumprod(tr, axis=1)               # (N, nl, NK)
        flux_gas = jnp.concatenate([jnp.ones((N, 1)), jnp.sum(ft, axis=2)], axis=1)       # (N, nl+1)
        fend = ft[:, -1, :]
        sum_flux = jnp.sum(fend, axis=1)
        sum_k = jnp.sum(k * fend[:, None, :], axis=2)
        kmin = jnp.min(jnp.where(valid[:, None, :], k, jnp.finfo(jnp.float64).max), axis=2)
        pos = sum_flux > 0.0
        k_eqv = k_eqv + jnp.where(pos[:, None], sum_k / jnp.where(pos, sum_flux, 1.0)[:, None], kmin)
        okf = flux_gas[:, :-1] > 0.0
        adj = jnp.where(okf, adj * flux_gas[:, 1:] / jnp.where(okf, flux_gas[:, :-1], 1.0), adj)
    adj = adj * jnp.exp(jnp.minimum(k_eqv * dmass * zen[:, None], TEMP_MAX))
    return k_eqv, adj


def k_eqv_ir(kperm, wperm, nkperm, dmass, pfb, dpb, alb, dsurf_src):
    N, nl, NK, n = kperm.shape
    sq = float(np.sqrt(EPS))
    finc = -pfb[:, 0]
    dsurf = dsurf_src - pfb[:, nl]
    k_eqv = jnp.zeros((N, nl))
    for j in range(1, n):
        k = kperm[..., j]; w = wperm[:, :, j]
        tau = k * dmass[:, :, None]
        trans = jnp.exp(-DIFF_MINOR * tau)
        s_up = (1.0 - trans + sq) * dpb[:, :, None] / (DIFF_MINOR * tau + sq)
        s_dn = -s_up
        def down(c, xs):
            tr, sd = xs
            c = tr * c + sd
            return c, c
        c0 = jnp.broadcast_to(finc[:, None], (N, NK))
        _, D = lax.scan(down, c0, (jnp.moveaxis(trans, 1, 0), jnp.moveaxis(s_dn, 1, 0)))     # (nl, N, NK): fd_{2i+2}, i = 1..nl
        U_nl = dsurf[:, None] + alb[:, None] * D[-1]
        def up(c, xs):
            tr, su = xs
            c = tr * c + su
            return c, c
        _, Ur = lax.scan(up, U_nl, (jnp.moveaxis(trans, 1, 0)[::-1], jnp.moveaxis(s_up, 1, 0)[::-1]))   # fd_{2i-1}, i = nl..1
        U = Ur[::-1]                                               # (nl, N, NK): fd_{2i-1}, i = 1..nl  -> layer-top upward flux
        # layer i (1-based) uses fluxes 2i = D[i-1] and 2i+1 = fd_{2(i+1)-1} = U[i] (U[nl] is the surface value U_nl)
        Uc = jnp.concatenate([U[1:], U_nl[None]], axis=0)          # (nl, N, NK): fd_{2i+1}
        Dtop = jnp.concatenate([c0[None], D[:-1]], axis=0)       # downward flux at the top of layer i: fd_{2i}
        a = jnp.abs(Dtop) ; b = jnp.abs(Uc)
        km = jnp.moveaxis(k, 1, 0)
        lk = jnp.sum(km * w[None] * a, axis=2) + jnp.sum(km * w[None] * b, axis=2)    # (nl, N)
        lf = jnp.sum(w[None] * a, axis=2) + jnp.sum(w[None] * b, axis=2)
        k_eqv = k_eqv + (lk / lf).T
    return k_eqv


# ---------------------------------------------------------------- the band
def band_fluxes(spec, iso, ib, X, scheme):
    """X: dict of device arrays (N columns): p, t, mix, mass, w_cloud, frac_cloud, zen_0, solar_irrad, rho (N,2), t_level, t_ground, tiles...
    plus grey inputs built by `grey_inputs`.  Returns (flux_down, flux_up, flux_direct|None), each (N, nl+1), band-weighted, IR with the
    Planck add-back."""
    solar = iso == 1
    p, t, mix, mass = X["p"], X["t"], X["mix"], X["mass"]
    N, nl = p.shape
    kperm, wperm, nkperm = band_gas(spec, ib, p, t, mix, mass)
    NK = kperm.shape[2]
    zen = X["zen_0"]
    fr = jnp.stack([jnp.zeros_like(X["w_cloud"]), X["frac_cloud"][:, :, 0] + X["frac_cloud"][:, :, 1],
                    X["frac_cloud"][:, :, 2] + X["frac_cloud"][:, :, 3]], axis=2)
    co, w_free = overlap_triple(X["w_cloud"], fr)
    kg, kes, ph, fwd = X["grey"][ib]
    if solar:
        k_eqv, adj = k_eqv_solar(kperm, wperm, nkperm, mass, zen)
        finc = X["solar_irrad_band"][ib] / zen
    else:
        pf, dp, dp2, pg = X["planck"][ib]
        k_eqv = k_eqv_ir(kperm, wperm, nkperm, mass, pf, dp, X["rho"][:, 0, ib - 1], pg)
        adj = None
        finc = -pf[:, 0]
    kgt = kg + k_eqv[:, :, None]                                   # k_grey_tot incl. k_eqv (all layers: nct = 1)
    kgas = kperm[..., 0]                                           # (N, nl, NK) major absorber k-terms
    ktot = kgt[:, :, None, :] + kgas[..., None]                    # (N, nl, NK, 5)
    tau = ktot * mass[:, :, None, None]
    omega = jnp.minimum(kes[:, :, None, :] / (ktot + TINY), 1.0 - 32.0 * EPS)
    f = fwd[:, :, None, :]
    tau, omega = tau * (1.0 - omega * f), omega * (1.0 - f) / (1.0 - omega * f)
    g = ph[:, :, None, :]                                          # post-rescale phase function (N, nl, 1, 5)
    g = (g - f) / (1.0 - f)
    sec0 = zen[:, None, None, None]
    cf = K.two_coeff(tau, tau, omega, g, scheme, solar, sec0 if solar else None, True)
    names = ["trans", "reflect"] + (["trans_0", "s_up", "s_down"] if solar else ["ir_1d", "ir_2d"])
    irc = [2, 2, 3, 3]
    reg = {}
    fc = X["frac_cloud"][:, :, None, :]                            # (N, nl, 1, 4)
    for nme in names:
        c = cf[nme]                                                # (N, nl, NK, 5)
        r = [c[..., 0], jnp.zeros_like(c[..., 0]), jnp.zeros_like(c[..., 0])]
        for ik in range(1, 5):
            f_ = fc[..., ik - 1]
            use = f_ > TINY
            r[irc[ik - 1] - 1] = r[irc[ik - 1] - 1] + jnp.where(use, f_ * c[..., ik], 0.0)
        for rr in (1, 2):
            fr_ = fr[:, :, rr][:, :, None]
            use = fr_ > TINY
            r[rr] = jnp.where(use, r[rr] / jnp.where(use, fr_, 1.0), 0.0)
        reg[nme] = jnp.stack(r, axis=-1)                           # (N, nl, NK, 3)
    B = N * NK
    flat = lambda a: jnp.transpose(a, (0, 2, 1) + tuple(range(3, a.ndim))).reshape((B, nl) + a.shape[3:])
    trans, refl = flat(reg["trans"]), flat(reg["reflect"])         # (B, nl, 3)
    rep = lambda a: jnp.repeat(a, NK, axis=0)
    v = rep(co[:, :, 0:9].reshape(N, nl + 1, 3, 3)); u = rep(co[:, :, 9:18].reshape(N, nl + 1, 3, 3))
    alb_d = rep(X["rho"][:, 0, ib - 1]); alb_dir = rep(X["rho"][:, 1, ib - 1])
    if solar:
        t0 = flat(reg["trans_0"]); sc_up = flat(reg["s_up"]); sc_dn = flat(reg["s_down"])
        fd0 = rep(finc); adjb = rep(adj)
        base = [fd0, jnp.zeros(B), jnp.zeros(B)]
        s_up = [None] * nl; s_dn = [None] * nl; fdir = [fd0]
        for i in range(1, nl + 1):
            c = i - 1
            top = [v[:, c, a, 0] * base[0] + v[:, c, a, 1] * base[1] + v[:, c, a, 2] * base[2] for a in range(3)]
            nb = [top[k] * t0[:, c, k] * adjb[:, c] for k in range(3)]
            s_up[c] = jnp.stack([sc_up[:, c, k] * top[k] for k in range(3)], axis=1)
            s_dn[c] = jnp.stack([(sc_dn[:, c, k] - t0[:, c, k]) * top[k] + nb[k] for k in range(3)], axis=1)
            base = nb
            fdir.append(base[0] + base[1] + base[2])
        s_up = jnp.stack(s_up, axis=1); s_dn = jnp.stack(s_dn, axis=1)
        sg = [(alb_dir - alb_d) * base[k] for k in range(3)]
        finc_b = rep(finc)
        fdir = jnp.stack(fdir, axis=1)
    else:
        dp_, dp2_ = rep(X["planck"][ib][1]), rep(X["planck"][ib][2])
        w_cloud_b = rep(X["w_cloud"]); w_free_b = rep(w_free); fr_b = rep(fr)
        s1, s2 = flat(reg["ir_1d"]), flat(reg["ir_2d"])
        s_up = s1 * dp_[:, :, None] + s2 * dp2_[:, :, None]
        s_dn = -s1 * dp_[:, :, None] + s2 * dp2_[:, :, None]
        wgt = jnp.stack([w_free_b, w_cloud_b * fr_b[:, :, 1], w_cloud_b * fr_b[:, :, 2]], axis=2)
        s_up, s_dn = wgt * s_up, wgt * s_dn
        pfb = X["planck"][ib][0]
        d_surf = rep(X["planck"][ib][3] - pfb[:, nl])
        sg = [v_ for v_ in (u[:, nl, 0, 0] * (1.0 - alb_d) * d_surf, u[:, nl, 1, 1] * (1.0 - alb_d) * d_surf, u[:, nl, 2, 2] * (1.0 - alb_d) * d_surf)]
        finc_b = rep(-pfb[:, 0])
        fdir = None
    ft = solver_triple_hogan(R_NCT1, trans[:, :, 0], refl[:, :, 0], s_dn[:, :, 0], s_up[:, :, 0], trans[:, :, 1], refl[:, :, 1],
                             s_dn[:, :, 1], s_up[:, :, 1], trans[:, :, 2], refl[:, :, 2], s_dn[:, :, 2], s_up[:, :, 2], v, u,
                             finc_b, sg[0], sg[1], sg[2], alb_d)                          # (B, 2nl+2)
    ft = ft.reshape(N, NK, 2 * nl + 2)
    w = wperm[:, :, 0]                                              # (N, NK) ESFT weights of the major absorber (0 where padded)
    up = jnp.einsum("nk,nkl->nl", w, ft[:, :, 0::2][:, :, :nl + 1])
    dn = jnp.einsum("nk,nkl->nl", w, ft[:, :, 1::2][:, :, :nl + 1])
    dr = None
    if solar:
        dr = jnp.einsum("nk,nkl->nl", w, fdir.reshape(N, NK, nl + 1))
    else:
        up = up + X["planck"][ib][0]; dn = dn + X["planck"][ib][0]
    return dn, up, dr


# ---------------------------------------------------------------- driver helpers
def columns_from_calls(calls):
    """Concatenate the tag-0 inputs of several one-column calls (same isolir) into one batch; arrays without a column axis from call 0."""
    r0 = calls[0]["r"]
    out = {}
    for k, v in r0.items():
        if k in (21, 22, 23, 24, 25, 26, 31, 20, 32):         # call-independent or scalar-like
            out[k] = v
        elif v.ndim >= 1 and v.shape[0] == 1 and all(c["r"][k].shape == v.shape for c in calls if k in c["r"]):
            out[k] = np.concatenate([c["r"][k] for c in calls], axis=0)
        else:
            out[k] = v
    return out


def prepare(spec, iso, r, tile=None):
    """Build the X dict of device arrays for the batch r (tag-0 style arrays with a column axis)."""
    X = dict(p=jnp.asarray(r[1]), t=jnp.asarray(r[2]), mix=jnp.asarray(r[4]), mass=jnp.asarray(r[5]),
             w_cloud=jnp.asarray(r[10]), frac_cloud=jnp.asarray(r[11]), zen_0=jnp.asarray(r[15]), rho=jnp.asarray(r[17]))
    n_band = spec["n_band_absorb"].shape[0]
    n_mode = int(r[31][2])
    X["grey"], X["planck"], X["solar_irrad_band"] = {}, {}, {}
    for ib in range(1, n_band + 1):
        cont = []
        for j in range(int(spec["n_band_continuum"][ib - 1])):
            ic = int(spec["index_continuum"][ib - 1, j])
            sc = spec["scale_cont"][:3, ib - 1, ic - 1]
            pr, tr = float(spec["p_ref_cont"][ic - 1, ib - 1]), float(spec["t_ref_cont"][ic - 1, ib - 1])
            pwk = jnp.power(X["p"] / pr, sc[0]); tt = X["t"] / tr - 1.0
            amount = pwk * (1.0 + sc[1] * tt + sc[2] * tt ** 2)
            water = X["mix"][:, :, 0]
            mdw = jnp.asarray(r[3]) * water / (REPSILON * MOL_WEIGHT_AIR)
            cont.append((float(spec["k_cont"][ib - 1, ic - 1]), amount * mdw * water))
        comps = []
        for kk, (ty, ip) in enumerate(zip(r[24].astype(int), r[21].astype(int))):
            kind = "water" if ty in (1, 3) else "ice"
            comps.append((kk + 1, kind, jnp.asarray(r[12][:, :, kk]), jnp.asarray(r[13][:, :, kk]),
                          [float(x) for x in r[23][:16 if kind == "water" else 14, kk, ib - 1]], jnp.asarray(r[14][:, :, kk])))
        rc = float(spec["rayleigh_coeff"][ib - 1]) if iso == 1 else 0.0
        X["grey"][ib] = grey_props(iso, rc, jnp.asarray(r[27]), jnp.asarray(r[28][..., ib - 1]), jnp.asarray(r[29][..., ib - 1]),
                                   jnp.asarray(r[30][..., ib - 1]), n_mode, cont, comps, X["w_cloud"])
        if iso == 1:
            X["solar_irrad_band"][ib] = jnp.asarray(r[16]) * float(spec["solar_flux_band"][ib - 1])
        else:
            pf, dp, dp2, pg = PL.band_planck(spec, ib, jnp.asarray(r[7]), X["t"], jnp.asarray(r[18]))
            if r[32][0] > 0.5:
                pg = jnp.asarray(PL.surface_planck(spec, ib, np.asarray(r[18]), dict(n_tile=int(r[32][1]), list_tile=r[33],
                                                                                        frac_tile=r[34], t_tile=r[35])))
            X["planck"][ib] = (pf, dp, dp2, pg)
    return X


def radiance(spec, iso, r, jit=True):
    """All bands of one isolir for the batch r.  Returns list of per-band (flux_down, flux_up, flux_direct) and, for the LW, the cumulative
    channel flux (down, up) after each band."""
    scheme = K.IP_PIFM80 if iso == 1 else K.IP_ELSASSER
    X = prepare(spec, iso, r)
    n_band = spec["n_band_absorb"].shape[0]
    res = []
    for ib in range(1, n_band + 1):
        f = (lambda ib_: jax.jit(lambda XX: band_fluxes(spec, iso, ib_, XX, scheme)))(ib) if jit else (lambda XX, ib_=ib: band_fluxes(spec, iso, ib_, XX, scheme))
        res.append(f(X))
    return res
