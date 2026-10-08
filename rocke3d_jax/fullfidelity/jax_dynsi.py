"""D190 (stage S5): DYNSI (sea-ice dynamics glue + VPICEDYN, ICEDYN_DRV.f:328-877) as a device program; port of dynsi_ff.py (glue) with the convergence loop of
icedyn_vec.vpicedyn as a lax.while_loop.

NEW module; dynsi_ff / icedyn_dynsi_ff / icedyn_vec / icedyn_jax are used (geometry, `_relax`, tridiagonal solvers) but not edited.

Summation order (the reason this module copies `_plast` and `_form` of icedyn_jax): the NumPy path (icedyn_vec) takes every sequential sum with np.cumsum(x)[-1]
(left to right); icedyn_jax uses jnp.cumsum, which on XLA:CPU is an associative scan and differs from the left-to-right sum in the last bit in ~80% of random cases
(measured here).  Here `seqsum` is a lax.scan of plain additions.  np.sum of the 72-element polar row in uosurf_from_ocean is NumPy's pairwise sum (8 accumulators);
`np_sum72` reproduces its order.  Squares are written x * x; no power above 2 occurs.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import dynsi_ff as DF  # noqa: E402
import icedyn_dynsi_ff as D  # noqa: E402
import icedyn_jax as JI  # noqa: E402

IM, JM = 72, 46
NX1, NY1 = IM + 2, JM
GRAV, RHOI, RHOWS, ACE1I, BYRHOI, OMEGA = DF.GRAV, DF.RHOI, DF.RHOWS, DF.ACE1I, DF.BYRHOI, DF.OMEGA
SINWAT, COSWAT, DTS, BYDTS = DF.SINWAT, DF.COSWAT, DF.DTS, DF.BYDTS


def seqsum(x):
    """Left-to-right sum of a 1-D array (the order of a Fortran DO loop / np.cumsum(x)[-1])."""
    s, _ = lax.scan(lambda c, v: (c + v, None), x[0], x[1:])
    return s


def np_sum72(x):
    """np.sum of a contiguous float64 vector of length 72 (NumPy's pairwise sum: 8 interleaved accumulators over the blocks of 8, then
    ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)); there is no remainder for n = 72)."""
    b = x.reshape(9, 8)
    r = b[0]
    for k in range(1, 9):
        r = r + b[k]
    return ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))


# ---------------------------------------------------------------------------------------------------------------- statics
def make_static(focean, sinpo, sinvo, ivnpo=IM // 4):
    """Host-side constants: the DYNSI Geom (this also fills the module globals of icedyn_dynsi_ff that icedyn_jax.geometry() snapshots), the
    jnp geometry pytree for _form/_relax, and the per-row weights of uosurf_from_ocean."""
    G = DF.Geom(focean)
    Kd = dict(G=G, foc=np.asarray(focean, float), Gj=JI.geometry(), heffm=np.asarray(D.HEFFM), sinen=np.asarray(D.SINEN),
              uvm=np.asarray(D.UVM), dxu=np.asarray(D.DXU), dyu=np.asarray(D.DYU))
    awt1 = np.zeros(JM + 1)
    for j in range(2, JM):
        awt1[j] = (sinpo[j] - sinvo[j - 1]) / (sinvo[j] - sinvo[j - 1])
    Kd['awt1'] = awt1
    Kd['awt2'] = 1.0 - awt1
    Kd['trig'] = DF.ocean_polar_trig()
    Kd['ivnpo'] = ivnpo
    return Kd


# ---------------------------------------------------------------------------------------------------------------- ocean -> ice grid
def uosurf_from_ocean(Kd, uo1, vo1):
    """dynsi_ff.uosurf_from_ocean on device arrays (uo1, vo1: layer-1 ocean velocities (IM, JM))."""
    cos_o, sin_o, cos_a, sin_a = (jnp.asarray(a) for a in Kd['trig'])
    awt1 = jnp.asarray(Kd['awt1'][2:JM])[None, :]
    awt2 = jnp.asarray(Kd['awt2'][2:JM])[None, :]
    um1 = jnp.roll(uo1, 1, axis=0)
    u = jnp.zeros((IM, JM)).at[:, 1:JM - 1].set(.5 * (uo1[:, 1:JM - 1] + um1[:, 1:JM - 1]))
    v = jnp.zeros((IM, JM)).at[:, 1:JM - 1].set(vo1[:, 0:JM - 2] * awt1 + vo1[:, 1:JM - 1] * awt2)
    iv = Kd['ivnpo']
    u = u.at[:, JM - 1].set(uo1[IM - 1, JM - 1] * cos_o + uo1[iv - 1, JM - 1] * sin_o)
    v = v.at[:, JM - 1].set(uo1[iv - 1, JM - 1] * cos_o - uo1[IM - 1, JM - 1] * sin_o)
    unp = np_sum72(u[:, JM - 1] * cos_a) * 2 / IM
    vnp = np_sum72(u[:, JM - 1] * sin_a) * 2 / IM
    u = u.at[:, JM - 1].set(unp)
    v = v.at[:, JM - 1].set(vnp)
    return u, v


# ---------------------------------------------------------------------------------------------------------------- input assembly
def _pad():
    return jnp.zeros((NX1 + 1, NY1 + 1))


def assemble_inputs(Kd, dmua, dmva, rsi, msi, snowi, ogeoza, uosurf, vosurf):
    G = Kd['G']
    foc = jnp.asarray(Kd['foc'])
    rsi, msi, snowi, dmua, dmva = (a.at[1:, JM - 1].set(a[0, JM - 1]) for a in (rsi, msi, snowi, dmua, dmva))
    nodata = foc * rsi <= 0
    dmua = jnp.where(nodata, 0.0, dmua)
    dmva = jnp.where(nodata, 0.0, dmva)

    def avg4(a):
        am1 = jnp.roll(a, 1, axis=0)
        return 0.25 * (a[:, :JM - 1] + am1[:, :JM - 1] + am1[:, 1:JM] + a[:, 1:JM]) * BYDTS

    gairx, gairy = _pad(), _pad()
    outs = []
    for gg, a in ((gairx, dmua), (gairy, dmva)):
        gg = gg.at[1:IM + 1, 1:JM].set(avg4(a))
        gg = gg.at[IM + 1, 1:JM].set(gg[1, 1:JM])
        gg = gg.at[IM + 2, 1:JM].set(gg[2, 1:JM])
        gg = gg.at[1:NX1 + 1, JM].set(a[0, JM - 1] * BYDTS)
        outs.append(gg)
    gairx, gairy = outs
    ptmp = ogeoza + (rsi * (msi + snowi + ACE1I)) * GRAV / RHOWS
    ip1 = np.roll(np.arange(IM), -1)
    dxp = jnp.asarray(G.dxp[2:JM])[None, :]            # J = 2..JM-1
    c = (foc[:, 1:JM - 1] > 0) & (foc[ip1, 1:JM - 1] > 0) & (rsi[:, 1:JM - 1] + rsi[ip1, 1:JM - 1] > 0)
    val = -(ptmp[ip1, 1:JM - 1] - ptmp[:, 1:JM - 1]) / dxp
    pgfu = _pad().at[1:IM + 1, 2:JM].set(jnp.where(c, val, 0.0))
    dyv = jnp.asarray(G.dyv[2:JM + 1])[None, :]        # j0 = 0..JM-2 -> dyv[j0 + 2]
    c = (foc[:, 1:JM] > 0) & (foc[:, 0:JM - 1] > 0) & (rsi[:, 0:JM - 1] + rsi[:, 1:JM] > 0)
    val = -(ptmp[:, 1:JM] - ptmp[:, 0:JM - 1]) / dyv
    pgfv = _pad().at[1:IM + 1, 1:JM].set(jnp.where(c, val, 0.0))
    heffm = jnp.asarray(Kd['heffm'])
    heff, area = _pad(), _pad()
    h = (rsi * (ACE1I + msi)) * BYRHOI
    heff = heff.at[2:NX1, 1:JM + 1].set(h)
    heff = heff.at[2:NX1, 1:JM + 1].set(heff[2:NX1, 1:JM + 1] * heffm[2:NX1, 1:JM + 1])
    area = area.at[2:NX1, 1:JM + 1].set(rsi)
    heff = heff.at[1, 1:JM + 1].set(heff[NX1 - 1, 1:JM + 1])
    area = area.at[1, 1:JM + 1].set(area[NX1 - 1, 1:JM + 1])
    heff = heff.at[NX1, 1:JM + 1].set(heff[2, 1:JM + 1])
    area = area.at[NX1, 1:JM + 1].set(area[2, 1:JM + 1])
    sinen = jnp.asarray(Kd['sinen'])
    a4 = RHOI * 0.25 * (heff[1:NX1, 1:JM] + heff[2:NX1 + 1, 1:JM] + heff[1:NX1, 2:JM + 1] + heff[2:NX1 + 1, 2:JM + 1])
    amass = _pad().at[1:NX1, 1:JM].set(a4)
    cor = _pad().at[1:NX1, 1:JM].set(a4 * 2.0 * OMEGA * sinen[1:NX1, 1:JM])
    amass = amass.at[1:NX1 + 1, JM].set(RHOI * heff[1, JM])
    cor = cor.at[1:NX1 + 1, JM].set(amass[1:NX1 + 1, JM] * 2.0 * OMEGA * sinen[1, JM])
    um1 = jnp.roll(uosurf, 1, axis=0)
    vm1 = jnp.roll(vosurf, 1, axis=0)
    gwatx = _pad().at[1:IM + 1, 1:JM].set(0.25 * (um1[:, :JM - 1] + um1[:, 1:JM] + uosurf[:, :JM - 1] + uosurf[:, 1:JM]))
    gwaty = _pad().at[1:IM + 1, 1:JM].set(0.25 * (vm1[:, :JM - 1] + vm1[:, 1:JM] + vosurf[:, :JM - 1] + vosurf[:, 1:JM]))
    pum1 = jnp.roll(pgfu[1:IM + 1, :], 1, axis=0)
    pvm1 = jnp.roll(pgfv[1:IM + 1, :], 1, axis=0)
    pgfub = _pad().at[1:IM + 1, 1:JM].set(0.5 * (pum1[:, 1:JM] + pum1[:, 2:JM + 1]))
    pgfvb = _pad().at[1:IM + 1, 1:JM].set(0.5 * (pvm1[:, 1:JM] + pgfv[1:IM + 1, 1:JM]))
    gw = []
    for a in (gwatx, gwaty, pgfub, pgfvb):
        a = a.at[IM + 1, 1:JM + 1].set(a[1, 1:JM + 1])
        a = a.at[IM + 2, 1:JM + 1].set(a[2, 1:JM + 1])
        gw.append(a)
    gwatx, gwaty, pgfub, pgfvb = gw
    return dict(gairx=gairx, gairy=gairy, gwatx=gwatx, gwaty=gwaty, pgfub=pgfub, pgfvb=pgfvb, heff=heff, area=area, amass=amass, cor=cor,
                irsi=rsi, imsi=msi, rsi=rsi, msi=msi, snowi=snowi)


# ---------------------------------------------------------------------------------------------------------------- VPICEDYN (FORM / PLAST copied with a left-to-right sum)
def _plast(G, uice1, vice1, press):
    nx1, ny1 = uice1.shape[0] - 1, uice1.shape[1] - 1
    nypole = ny1 - 1
    ecm2 = 1.0 / (D.ECCEN ** 2)
    gmin = 1e-20
    radius = G["RADIUS"]
    S = lambda F, di, dj: F[2 + di:nx1 + di, 2 + dj:ny1 + dj]
    dxt = G["DXT"][2:nx1, None]
    cst = G["CST"][None, 2:ny1]
    dyt = G["DYT"][None, 2:ny1]
    tngt = G["TNGT"][None, 2:ny1]
    U = lambda di, dj: S(uice1, di, dj)
    Vv = lambda di, dj: S(vice1, di, dj)
    e11 = 0.5 / (dxt * cst) * (U(0, 0) + U(0, -1) - U(-1, 0) - U(-1, -1)) \
        - 0.25 * (Vv(0, 0) + Vv(-1, 0) + Vv(-1, -1) + Vv(0, -1)) * tngt / radius
    e22 = 0.5 / dyt * (Vv(0, 0) + Vv(-1, 0) - Vv(0, -1) - Vv(-1, -1))
    e12 = 0.5 * (0.5 / dyt * (U(0, 0) + U(-1, 0) - U(0, -1) - U(-1, -1))
                 + 0.5 / (dxt * cst) * (Vv(0, 0) + Vv(0, -1) - Vv(-1, 0) - Vv(-1, -1))
                 + 0.25 * (U(0, 0) + U(-1, 0) + U(-1, -1) + U(0, -1)) * tngt / radius)
    delt = (e11 ** 2 + e22 ** 2) * (1.0 + ecm2) + 4.0 * ecm2 * e12 ** 2 \
        + 2.0 * e11 * e22 * (1.0 - ecm2)
    delt1 = jnp.maximum(gmin, jnp.sqrt(delt))
    zin = 0.5 * S(press, 0, 0) / delt1
    zmax = (5e12 / 2e4) * press
    zin = jnp.minimum(S(zmax, 0, 0), zin)
    zin = jnp.maximum(4e8, zin)
    zeta = jnp.zeros_like(press).at[2:nx1, 2:ny1].set(zin)
    zeta = zeta.at[1:nx1 + 1, ny1].set(seqsum(zeta[2:nx1, nypole]) / (nx1 - 2))
    zeta = zeta.at[1:nx1 + 1, 1].set(seqsum(zeta[2:nx1, 2]) / (nx1 - 2))
    zeta = zeta.at[1, :].set(zeta[nx1 - 1, :])
    zeta = zeta.at[nx1, :].set(zeta[2, :])
    return ecm2 * zeta, zeta


def _form(G, uice1, vice1, gairx, gairy, gwatx, gwaty, heff, area, amass, cor, pgfub, pgfvb, sinwat, coswat):
    nx1, ny1 = uice1.shape[0] - 1, uice1.shape[1] - 1
    nypole = ny1 - 1
    z = jnp.zeros_like(uice1)
    rr = (slice(1, nx1), slice(1, ny1 + 1))
    dwatn = z.at[rr].set(5.5 * jnp.sqrt((uice1[rr] - gwatx[rr]) ** 2 + (vice1[rr] - gwaty[rr]) ** 2))
    drags, draga, forcex, forcey = z, z, z, z
    half = ny1 // 2
    for (j0, j1, north) in ((1, half + 1, False), (half + 1, nypole + 1, True)):
        r = (slice(1, nx1), slice(j0, j1))
        dw, cr, gx, gy = dwatn[r], cor[r], gwatx[r], gwaty[r]
        drags = drags.at[r].set(dw * coswat)
        fx, fy = gairx[r], gairy[r]
        if north:
            draga = draga.at[r].set(dw * sinwat + cr)
            fx = fx + dw * (coswat * gx - sinwat * gy)
            fy = fy + dw * (sinwat * gx + coswat * gy)
        else:
            draga = draga.at[r].set(-dw * sinwat + cr)
            fx = fx + dw * (coswat * gx + sinwat * gy)
            fy = fy + dw * (-sinwat * gx + coswat * gy)
        fx = fx + amass[r] * pgfub[r]
        fy = fy + amass[r] * pgfvb[r]
        forcex = forcex.at[r].set(fx)
        forcey = forcey.at[r].set(fy)
    ra = (slice(1, nx1 + 1), slice(1, ny1 + 1))
    press = z.at[ra].set(D.PSTAR * heff[ra] * jnp.exp(-20.0 * (1.0 - area[ra])))
    eta, zeta = _plast(G, uice1, vice1, press)
    press = press.at[1:nx1 + 1, ny1].set(seqsum(press[2:nx1, nypole]) / (nx1 - 2))
    press = press.at[1, :].set(press[nx1 - 1, :])
    press = press.at[nx1, :].set(press[2, :])
    HEFFM = G["HEFFM"]
    press, eta, zeta = press * HEFFM, eta * HEFFM, zeta * HEFFM
    i0, j0 = slice(1, nx1), slice(1, nypole + 1)
    dxu = G["DXU"][1:nx1, None]
    csu = G["CSU"][None, 1:nypole + 1]
    dyu = G["DYU"][None, 1:nypole + 1]
    p = lambda di, dj: press[1 + di:nx1 + di, 1 + dj:nypole + 1 + dj]
    forcex = forcex.at[i0, j0].set(forcex[i0, j0] - (0.25 / (dxu * csu)) * (p(1, 0) + p(1, 1) - p(0, 0) - p(0, 1)))
    forcey = forcey.at[i0, j0].set(forcey[i0, j0] - 0.25 / dyu * (p(0, 1) + p(1, 1) - p(0, 0) - p(1, 0)))
    forcex = forcex.at[1, :].set(forcex[nx1 - 1, :]).at[nx1, :].set(forcex[2, :])
    forcey = forcey.at[1, :].set(forcey[nx1 - 1, :]).at[nx1, :].set(forcey[2, :])
    dwatn = dwatn.at[1, :].set(dwatn[nx1 - 1, :]).at[nx1, :].set(dwatn[2, :])
    return dict(dwatn=dwatn, drags=drags, draga=draga, forcex=forcex, forcey=forcey, press=press, eta=eta, zeta=zeta)


def vpicedyn(Kd, usi0, vsi0, inp, max_kki=20, conv_tol=25e-6):
    """icedyn_vec.vpicedyn as a device program.  Returns (uice1, vice1, kki, last_dwatn)."""
    G = Kd['Gj']
    nx1, ny1 = NX1, NY1
    imicdyn = nx1 - 2
    z = jnp.zeros((nx1 + 1, ny1 + 1))
    u1 = z.at[2:imicdyn + 2, 1:ny1 + 1].set(usi0[0:imicdyn, 0:ny1])
    v1 = z.at[2:imicdyn + 2, 1:ny1 + 1].set(vsi0[0:imicdyn, 0:ny1])
    u1 = u1.at[1, 1:ny1 + 1].set(usi0[imicdyn - 1, 0:ny1]).at[nx1, 1:ny1 + 1].set(usi0[0, 0:ny1])
    v1 = v1.at[1, 1:ny1 + 1].set(vsi0[imicdyn - 1, 0:ny1]).at[nx1, 1:ny1 + 1].set(vsi0[0, 0:ny1])
    fargs = tuple(inp[k] for k in ('gairx', 'gairy', 'gwatx', 'gwaty', 'heff', 'area', 'amass', 'cor', 'pgfub', 'pgfvb'))
    amass = inp['amass']
    jall = slice(1, ny1 + 1)
    rall = (slice(1, nx1 + 1), slice(1, ny1 + 1))
    w = G["DXU"][1:nx1 + 1, None] * G["DYU"][None, 1:ny1 + 1]
    sel = (amass[rall] * G["UVM"][rall]) > 0
    wsel = jnp.where(sel, jnp.broadcast_to(w, sel.shape), 0.0)
    area_tot_c = seqsum(wsel.reshape(-1))

    def wrap(x):
        return x.at[1, jall].set(x[nx1 - 1, jall]).at[nx1, jall].set(x[2, jall])

    def run_form(uu, vv):
        return _form(G, uu, vv, *fargs, SINWAT, COSWAT)

    def body(c):
        u1, v1, u2, v2, u3, v3, usave, vsave, kki, go, dwn = c
        kki = kki + 1
        f = run_form(u1, v1)
        u1, u2, u3, v1, v2, v3, _, _ = JI._relax(G, u1, u2, u1, v1, v2, v1, u1, v1, f["forcex"], f["forcey"], f["draga"], f["drags"], f["eta"],
                                                 f["zeta"], amass, BYDTS)
        u1, v1 = wrap(u1), wrap(v1)
        u1 = u1.at[rall].set(0.5 * (u1[rall] + u2[rall]))
        v1 = v1.at[rall].set(0.5 * (v1[rall] + v2[rall]))
        f = run_form(u1, v1)
        dwn = f["dwatn"]
        uc, vc = u1, v1
        u1, v1 = u2, v2
        u1, u2, u3, v1, v2, v3, _, _ = JI._relax(G, u1, u2, uc, v1, v2, vc, uc, vc, f["forcex"], f["forcey"], f["draga"], f["drags"], f["eta"],
                                                 f["zeta"], amass, BYDTS)
        u1, v1 = wrap(u1), wrap(v1)
        term = w * ((usave[rall] - u1[rall]) ** 2 + (vsave[rall] - v1[rall]) ** 2)
        rms = seqsum(jnp.where(sel, term, 0.0).reshape(-1))
        # kki == max_kki: stop; kki == 1 or rms > tol * area: iterate again (usave updated); else stop
        again = (kki == 1) | (rms > conv_tol * area_tot_c)
        go = (kki != max_kki) & again
        upd = (kki == 1) | (rms > conv_tol * area_tot_c)
        usave = jnp.where(upd & (kki != max_kki), u1, usave)
        vsave = jnp.where(upd & (kki != max_kki), v1, vsave)
        return (u1, v1, u2, v2, u3, v3, usave, vsave, kki, go, dwn)

    c0 = (u1, v1, z, z, z, z, z, z, jnp.asarray(0, jnp.int64), jnp.asarray(True), z)
    c = lax.while_loop(lambda c: c[9], body, c0)
    return c[0], c[1], c[8], c[10]


# ---------------------------------------------------------------------------------------------------------------- post-processing
def post_vpicedyn(Kd, uice1, vice1, dwatn, gwatx, gwaty, irsi, usi_old, vsi_old):
    G = Kd['G']
    foc = jnp.asarray(Kd['foc'])
    half = NY1 // 2
    jj = np.arange(1, JM + 1)
    hemi = jnp.asarray(np.where(jj <= half, -1.0, 1.0))[None, :]
    sl = (slice(1, NX1), slice(1, JM + 1))
    dw = dwatn[sl]
    du = uice1[sl] - gwatx[sl]
    dv = vice1[sl] - gwaty[sl]
    dmu = _pad().at[sl].set(DTS * dw * (COSWAT * du - hemi * SINWAT * dv))
    dmv = _pad().at[sl].set(DTS * dw * (hemi * SINWAT * du + COSWAT * dv))
    dmu = dmu.at[1, :].set(dmu[NX1 - 1, :]).at[NX1, :].set(dmu[2, :])
    u = uice1[2:IM + 2, 2:JM]
    v = vice1[2:IM + 2, 2:JM]
    usi = usi_old.at[:, 1:JM - 1].set(jnp.where(jnp.abs(u) < 1e-10, 0.0, u)).at[:, JM - 1].set(0.0)
    vsi = vsi_old.at[:, 1:JM - 1].set(jnp.where(jnp.abs(v) < 1e-10, 0.0, v)).at[:, JM - 1].set(0.0)
    Iidx = np.arange(1, IM + 1)
    ip1 = Iidx % IM + 1
    fr = foc * irsi
    dxyn, dxys, dxyv, byd = (jnp.asarray(a) for a in (G.dxyn, G.dxys, G.dxyv, G.bydxyp))
    dmui = jnp.zeros((IM, JM))
    dmvi = jnp.zeros((IM, JM))
    Jr = np.arange(2, JM)
    d = 0.5 * (dmu[Iidx + 1][:, Jr - 1] + dmu[Iidx + 1][:, Jr])
    dmui = dmui.at[:, Jr - 1].set(0.5 * d * (fr[Iidx - 1][:, Jr - 1] + fr[ip1 - 1][:, Jr - 1]))
    e = 0.5 * (dmv[Iidx][:, Jr] + dmv[Iidx + 1][:, Jr])
    dmvi = dmvi.at[:, Jr - 1].set(0.5 * e * (fr[Iidx - 1][:, Jr - 1] * dxyn[Jr][None, :] + fr[Iidx - 1][:, Jr] * dxys[Jr + 1][None, :]) / dxyv[Jr + 1][None, :])
    dmui = dmui.at[:, 0].set(0.0)
    J = JM
    d = 0.5 * (dmu[Iidx + 1, J - 1] + dmu[Iidx + 1, J])
    npcol = 0.5 * d * (fr[Iidx - 1, J - 1] + fr[ip1 - 1, J - 1])
    dmuinp = seqsum(npcol) / IM
    dmui = dmui.at[:, J - 1].set(dmuinp)
    dmvi = dmvi.at[:, J - 1].set(0.0)
    # UI2rho
    ui2 = jnp.zeros((IM, JM))
    i1 = Iidx
    for j in range(1, JM + 1):
        if j > 1:
            tu = dxys[j] * (dmu[i1 + 1, j - 1] + dmu[i1, j - 1])
            tv = dxys[j] * (dmv[i1 + 1, j - 1] + dmv[i1, j - 1])
        else:
            tu = tv = 0.0
        dua = 0.5 * (dxyn[j] * (dmu[i1 + 1, j] + dmu[i1, j]) + tu) * byd[j]
        dva = 0.5 * (dxyn[j] * (dmv[i1 + 1, j] + dmv[i1, j]) + tv) * byd[j]
        for_ = foc[:, j - 1] * irsi[:, j - 1] > 0
        ui2 = ui2.at[:, j - 1].set(jnp.where(for_, jnp.sqrt(dua * dua + dva * dva) * BYDTS, 0.0))
    return dict(usi=usi, vsi=vsi, dmui=dmui, dmvi=dmvi, dmu=dmu, dmv=dmv, ui2rho=ui2)


def get_uisurf(Kd, usi, vsi):
    G = Kd['G']
    siniu, cosiu = jnp.asarray(G.siniu), jnp.asarray(G.cosiu)
    u = jnp.zeros((IM, JM))
    v = jnp.zeros((IM, JM))
    ums1 = jnp.roll(usi, 1, axis=0)
    vms1 = jnp.roll(vsi, 1, axis=0)
    u = u.at[:, 1:JM - 1].set(0.25 * (usi[:, 1:JM - 1] + usi[:, 0:JM - 2] + ums1[:, 1:JM - 1] + ums1[:, 0:JM - 2]))
    v = v.at[:, 1:JM - 1].set(0.25 * (vsi[:, 1:JM - 1] + vsi[:, 0:JM - 2] + vms1[:, 1:JM - 1] + vms1[:, 0:JM - 2]))
    i = np.arange(1, IM + 1)
    s_u = seqsum((vsi[i - 1, 0] * siniu[i]) * 2. / IM)
    s_v = seqsum((vsi[i - 1, 0] * cosiu[i]) * 2. / IM)
    n_u = seqsum(-((vsi[i - 1, JM - 2] * siniu[i]) * 2. / IM))
    n_v = seqsum((vsi[i - 1, JM - 2] * cosiu[i]) * 2. / IM)
    u = u.at[:, 0].set(s_u).at[:, JM - 1].set(n_u)
    v = v.at[:, 0].set(s_v).at[:, JM - 1].set(n_v)
    return u, v


def dynsi(Kd, dmua, dmva, rsi, msi, snowi, ogeoza, uo1, vo1, usi, vsi):
    """dynsi_ff.dynsi on device arrays.  rsi/msi/snowi: ocean-domain ice (zero in lake cells).  Returns the dict of the NumPy version (odmui, odmvi, ui2rho,
    ustar, usi, vsi, uisurf, visurf, kki)."""
    us, vs = uosurf_from_ocean(Kd, uo1, vo1)
    inp = assemble_inputs(Kd, dmua, dmva, rsi, msi, snowi, ogeoza, us, vs)
    uice1, vice1, kki, dwatn = vpicedyn(Kd, usi, vsi, inp)
    post = post_vpicedyn(Kd, uice1, vice1, dwatn, inp['gwatx'], inp['gwaty'], inp['irsi'], usi, vsi)
    odmui = post['dmui'].at[:, JM - 1].set(0.0)
    odmvi = post['dmvi'].at[:, JM - 1].set(0.0)
    uis, vis = get_uisurf(Kd, post['usi'], post['vsi'])
    ustar = jnp.maximum(5e-4, jnp.sqrt(post['ui2rho'] / RHOWS))
    return dict(odmui=odmui, odmvi=odmvi, ui2rho=post['ui2rho'], ustar=ustar, usi=post['usi'], vsi=post['vsi'], uisurf=uis, visurf=vis, kki=kki,
                us=us, vs=vs, inp=inp, uice1=uice1, vice1=vice1, dwatn=dwatn)
