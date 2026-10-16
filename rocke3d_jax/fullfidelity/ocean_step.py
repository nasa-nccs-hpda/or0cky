"""Chained live ocean step (D119): OCEANS (OCNDYN2.f) for P2SAoM40, built from the validated ports.

State container: dict of 0-based float64 numpy arrays, 3-D fields (IM, JM, LMO), 2-D (IM, JM), keys as in
ocean_chain_io (g0m, s0m, gxmo, ..., mo, uo, vo, uod, vod, mmi, smu, smv, smw, opbot, opress, ogeoz, kpl, vonp).
Each stage is a function state -> state (inputs are copied, never mutated). STAGES lists them in call order;
see scoping/OCEAN_CHAIN_PLAN.md for the call table and the recorded boundaries.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from eos_jax import OFTAB, volgsp  # noqa: E402
from osourc_jax import osourc, gfrezs  # noqa: E402
from ground_oc_sweep_jax import ground_oc_sweep_layer  # noqa: E402
from precip_oc_jax import precip_oc_cell  # noqa: E402
from ostres2_jax import ostres2_jax, geomo_arrays_jax  # noqa: E402
from obdrag2_jax import obdrag2_jax  # noqa: E402
from ocoast_jax import ocoast_jax  # noqa: E402
from polerelax_jax import polerelax_jax  # noqa: E402
from kvinit_ff import kvinit  # noqa: E402

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665
DTS = 1800.0
LSRPD = 3
IVNP0 = IM // 4 - 1          # 0-based column of Fortran IVNP = IM/4
F3 = ['g0m', 's0m', 'gxmo', 'gymo', 'gzmo', 'sxmo', 'symo', 'szmo', 'mo', 'uo', 'vo', 'uod', 'vod']
_DXYPO = np.asarray(sum(geomo_arrays_jax()[:2]))      # dxys + dxyn, length JM


def oftab_record(k, path=OFTAB):
    """k-th (0-based) record of OFTABLE_NEW (80-byte title + array), big-endian sequential."""
    raw = open(path, 'rb').read()
    o = 0
    for i in range(k + 1):
        n = int.from_bytes(raw[o:o + 4], 'big')
        body = raw[o + 4:o + 4 + n]
        o += n + 8
    return np.frombuffer(body[80:], dtype='>f8').astype(np.float64)


_CGS = None


def shcgs(g, s):
    """OCNFUNTAB.f SHCGS: bilinear lookup in OFTAB record 3 (CGS(-2:40, 0:40))."""
    global _CGS
    if _CGS is None:
        _CGS = oftab_record(2).reshape((43, 41), order='F')
    gg = g * 2.5e-4
    ss = s * 1000.0
    ig = np.clip(np.trunc(gg + 2.0).astype(np.int64) - 2, -2, 39)
    js = np.trunc(ss).astype(np.int64)
    js = np.where(js >= 40, 39, js)
    c = lambda i, j: _CGS[i + 2, j]   # noqa: E731
    return ((js - ss + 1) * ((ig - gg + 1) * c(ig, js) + (gg - ig) * c(ig + 1, js))
            + (ss - js) * ((ig - gg + 1) * c(ig, js + 1) + (gg - ig) * c(ig + 1, js + 1)))


def imaxj_mask():
    m = np.zeros((IM, JM), bool)
    m[:, 1:JM - 1] = True
    m[0, 0] = m[0, JM - 1] = True
    return m


def copy_state(s):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}


# ---------------------------------------------------------------- stage 0: KVINIT + PRECIP_OC
def stage_precip(s, fx, ctx):
    s = copy_state(s)
    snap = kvinit(s['g0m'], s['s0m'], s['mo'], s['gxmo'], s['gymo'], s['sxmo'], s['symo'],
                  s['uo'], s['vo'], s['uod'], s['vod'], LSRPD)
    s.update(snap)
    foc = ctx['focean']
    act = imaxj_mask() & (foc > 0) & (fx['oprec'] > 0)
    dx = _DXYPO[None, :]
    r = precip_oc_cell(foc, fx['oprec'], fx['orsi'], fx['orunpsi'], fx['oeprec'], fx['oerunpsi'],
                       fx['osrunpsi'], dx, s['mo'][:, :, 0], s['g0m'][:, :, 0], s['s0m'][:, :, 0])
    for k, a in (('mo', 'mo'), ('g0m', 'g0m'), ('s0m', 's0m')):
        s[k][:, :, 0] = np.where(act, np.asarray(r[a]), s[k][:, :, 0])
    return s


# ---------------------------------------------------------------- stage 1: GROUND_OC
def stage_ground(s, fx, ctx):
    s = copy_state(s)
    foc, lmm = ctx['focean'], ctx['lmm']
    sel = imaxj_mask() & (foc > 0)
    ii, jj = np.nonzero(sel)
    dxypo = _DXYPO[jj]
    # DXYPJ = DXYPJ*FOCEAN(I,J) accumulates along I within a row (OCNDYN.f:4690-4692); BYDXYPJ likewise by division
    dxypj = np.zeros(len(ii)); bydxypj = np.zeros(len(ii))
    for j in np.unique(jj):
        d, b = _DXYPO[j], 1.0 / _DXYPO[j]
        for n in np.nonzero(jj == j)[0]:
            d = d * foc[ii[n], j]; b = b / foc[ii[n], j]
            dxypj[n] = d; bydxypj[n] = b
    f = lambda k: fx[k][ii, jj]      # noqa: E731
    runo = (f('oflowo') + f('omelti')) - f('oevapor')
    runi = (f('oflowo') + f('omelti')) + f('orunosi')
    eruno = (f('oeflowo') + f('oemelti')) + f('oe0')
    eruni = (f('oeflowo') + f('oemelti')) + f('oerunosi')
    sruno = f('osmelti'); sruni = f('osmelti') + f('osrunosi')
    g0ml = s['g0m'][ii, jj, :].copy(); gzml = s['gzmo'][ii, jj, :].copy()
    lm = lmm[ii, jj]
    g0ml[np.arange(len(ii)), lm - 1] += DTS * ctx['fgeotherm'][ii, jj] * dxypo
    r = osourc(f('orsi'), s['mo'][ii, jj, 0], g0ml, gzml, s['s0m'][ii, jj, 0], dxypj, bydxypj, lm,
               runo, runi, eruno, eruni, sruno, sruni, f('osolarw'), f('osolari'))
    r = {k: np.asarray(v) for k, v in r.items()}
    mo = s['mo'][ii, jj, :].copy(); g0 = r['g0ml'].copy(); s0 = s['s0m'][ii, jj, :].copy()
    mo[:, 0] = r['mo']; s0[:, 0] = r['s0m']
    gz = r['gzml']
    # sweep over lower layers (OCNDYN.f:4786-4806)
    p0l = mo[:, 0] * GRAV
    sdm = np.zeros(len(ii)); sde = np.zeros(len(ii)); sds = np.zeros(len(ii))
    for l in range(2, LMO + 1):
        act = lm >= l
        mol = np.where(act, mo[:, l - 1], 1.0)
        g0l = np.where(act, g0[:, l - 1], 0.0) / (mol * dxypj); s0l = np.where(act, s0[:, l - 1], 0.0) / (mol * dxypj)
        p0l_a = p0l + mol * GRAV * 0.5
        gf00 = np.asarray(gfrezs(jnp.asarray(s0l)))
        pcorr = (shcgs(gf00, s0l) * 8.19e-8) * p0l_a
        o = {k: np.asarray(v) for k, v in ground_oc_sweep_layer(mol, np.where(act, g0[:, l - 1], 0.0), np.where(act, s0[:, l - 1], 0.0), dxypj, pcorr, p0l_a).items()}
        mo[:, l - 1] = np.where(act, o['mo'], mo[:, l - 1])
        g0[:, l - 1] = np.where(act, o['g0m'], g0[:, l - 1])
        s0[:, l - 1] = np.where(act, o['s0m'], s0[:, l - 1])
        sdm = sdm + np.where(act, o['dm0'], 0.0); sde = sde + np.where(act, o['de0'], 0.0)
        sds = sds + np.where(act, o['ds0'], 0.0)
        p0l = np.where(act, p0l_a + mo[:, l - 1] * GRAV * 0.5, p0l)
    s['mo'][ii, jj, :] = mo; s['g0m'][ii, jj, :] = g0; s['s0m'][ii, jj, :] = s0; s['gzmo'][ii, jj, :] = gz
    dmsi = np.zeros((2, IM, JM)); dhsi = np.zeros((2, IM, JM)); dssi = np.zeros((2, IM, JM))
    dmsi[0, ii, jj] = r['dmoo'] + sdm; dmsi[1, ii, jj] = r['dmoi'] + sdm
    dhsi[0, ii, jj] = r['deoo'] + sde; dhsi[1, ii, jj] = r['deoi'] + sde
    dssi[0, ii, jj] = r['dsoo'] + sds; dssi[1, ii, jj] = r['dsoi'] + sds
    rsi = fx['orsi']
    op = fx['oapress'] + GRAV * ((1.0 - rsi) * dmsi[0] + rsi * dmsi[1])
    s['opress'] = np.where(sel, op, s['opress'])
    s['opress'][1:, 0] = s['opress'][0, 0]
    s['opress'][1:, JM - 1] = s['opress'][0, JM - 1]
    s['odmsi'], s['odhsi'], s['odssi'] = dmsi, dhsi, dssi
    return s


# ---------------------------------------------------------------- stage 2: OSTRES2
def stage_ostres(s, fx, ctx):
    s = copy_state(s)
    uo, vo, uod, vod = ostres2_jax(ctx['lmu'], ctx['lmv'], fx['odmua'], fx['odmva'], fx['odmui'], fx['odmvi'],
                                   s['mo'][:, :, 0], s['uo'][:, :, 0], s['vo'][:, :, 0], s['uod'][:, :, 0],
                                   s['vod'][:, :, 0], IVNP0)
    for k, a in (('uo', uo), ('vo', vo), ('uod', uod), ('vod', vod)):
        s[k][:, :, 0] = np.asarray(a)
    return s


# ---------------------------------------------------------------- stage 4: OBDRAG2 + OCOAST
def stage_drag(s, fx, ctx):
    s = copy_state(s)
    lmu, lmv, lmm = [np.asarray(ctx[k], dtype=np.float64) for k in ('lmu', 'lmv', 'lmm')]
    uo, vo, uod, vod = obdrag2_jax(lmu, lmv, s['mo'], s['uo'], s['vo'], s['uod'], s['vod'])
    gx, sx, gy, sy = ocoast_jax(lmm, s['gxmo'], s['sxmo'], s['gymo'], s['symo'])
    for k, a in (('uo', uo), ('vo', vo), ('uod', uod), ('vod', vod), ('gxmo', gx), ('sxmo', sx), ('gymo', gy), ('symo', sy)):
        s[k] = np.asarray(a).copy()
    return s


# ---------------------------------------------------------------- stage 5: polar relax
def stage_polar(s, fx, ctx):
    s = copy_state(s)
    lmu, lmv = [np.asarray(ctx[k], dtype=np.float64) for k in ('lmu', 'lmv')]
    uo, vo, uod, vod = polerelax_jax(lmu, lmv, s['uo'], s['vo'], s['uod'], s['vod'])
    for k, a in (('uo', uo), ('vo', vo), ('uod', uod), ('vod', vod)):
        s[k] = np.asarray(a).copy()
    return s


# (name, function, snapshot tag reached afterwards)
STAGES = [('precip', stage_precip, 1), ('ground', stage_ground, 2), ('ostres', stage_ostres, 3),
          ('drag', stage_drag, 5), ('polar', stage_polar, 6)]


# ---------------------------------------------------------------- stage 3: OCONV (D119: preamble + hbl loop + tail)
_OC = {}


def _oconv_tables():
    if not _OC:
        import kppmix_ff as K
        from ofluxv_jax import ZE
        from stconv_compare import tabs_and_grid
        ze = np.asarray(ZE)
        _OC.update(ze=ze, tabs=tabs_and_grid(ze), vg=oftab_record(0).reshape((43, 41, 40), order='F'),
                   ag=oftab_record(4).reshape((43, 41, 40), order='F'),
                   bg=oftab_record(5).reshape((43, 41, 40), order='F'), cg=oftab_record(2).reshape((43, 41), order='F'))
    return _OC


def oconv_columns(s, fx, ctx):
    """OCONV per-column preamble (OCNKPP.f:1648-1973), vectorised over all processed columns.
    Returns (arguments dict for ocean_hbl.hbl_loop, column index arrays)."""
    T = _oconv_tables()
    ze = T['ze']; fsr2 = T['tabs']['fsr'][2]
    lmm, lmu, lmv = ctx['lmm'], ctx['lmu'], ctx['lmv']
    msk = (lmm > 1); msk[:, 0] = False; msk[:, JM - 1] = False
    jj, ii = np.nonzero(msk.T)
    jj = np.append(jj, JM - 1); ii = np.append(ii, 0)           # pole column last
    N = len(ii); pole = np.zeros(N, bool); pole[-1] = True
    im1 = (ii - 1) % IM
    jm1 = np.maximum(jj - 1, 0)
    dxys, dxyn = [np.asarray(a) for a in geomo_arrays_jax()[:2]]
    dxypo = _DXYPO[jj]
    KM = 74
    L1 = np.arange(1, LMO + 1)
    lmij = lmm[ii, jj].astype(np.int64)
    kmuv = np.where(pole, IM + 2, 4).astype(np.int64)
    ut, vt, utd, vtd = s['uo'], s['vo'], s['uod'], s['vod']

    def stack(a):   # (N,13) layer-first view from grid (n cells)
        return a
    ul0 = np.zeros((N, LMO + 1, KM + 1)); ul = np.zeros_like(ul0); uld0 = np.zeros_like(ul0); uld = np.zeros_like(ul0)
    lmuv = np.zeros((N, KM + 1), np.int64); ravm = np.zeros((N, KM + 1))
    npn = ~pole
    n = np.nonzero(npn)[0]
    srcs0 = [ut[im1[n], jj[n], :], ut[ii[n], jj[n], :], vt[ii[n], jm1[n], :], vt[ii[n], jj[n], :]]
    srcs1 = [s['uo1'][im1[n], jj[n]], s['uo1'][ii[n], jj[n]], s['vo1'][ii[n], jm1[n]], s['vo1'][ii[n], jj[n]]]
    dsrc0 = [vtd[im1[n], jj[n], :], vtd[ii[n], jj[n], :], utd[ii[n], jm1[n], :], utd[ii[n], jj[n], :]]
    dsrc1 = [s['vod1'][im1[n], jj[n]], s['vod1'][ii[n], jj[n]], s['uod1'][ii[n], jm1[n]], s['uod1'][ii[n], jj[n]]]
    for k in range(4):
        ul0[n, 1:, k + 1] = srcs0[k]; ul[n, 1:, k + 1] = srcs0[k]; ul[n, 1, k + 1] = srcs1[k]
        uld0[n, 1:, k + 1] = dsrc0[k]; uld[n, 1:, k + 1] = dsrc0[k]; uld[n, 1, k + 1] = dsrc1[k]
    lmuv[n, 1] = lmu[im1[n], jj[n]]; lmuv[n, 2] = lmu[ii[n], jj[n]]
    lmuv[n, 3] = lmv[ii[n], jm1[n]]; lmuv[n, 4] = lmv[ii[n], jj[n]]
    ravm[n, 1:5] = 0.5
    p = N - 1
    ul0[p, 1:, 1:IM + 1] = vt[:, JM - 2, :].T; ul[p, 1:, 1:IM + 1] = vt[:, JM - 2, :].T
    ul[p, 1, 1:IM + 1] = s['vo1'][:, JM - 2]
    ul0[p, 1:, IM + 1] = ut[IM - 1, JM - 1, :]; ul[p, 1:, IM + 1] = ut[IM - 1, JM - 1, :]; ul[p, 1, IM + 1] = s['uo1'][IM - 1, JM - 1]
    ul0[p, 1:, IM + 2] = ut[IVNP0, JM - 1, :]; ul[p, 1:, IM + 2] = ut[IVNP0, JM - 1, :]; ul[p, 1, IM + 2] = s['uo1'][IVNP0, JM - 1]
    lmuv[p, 1:IM + 1] = lmv[:, JM - 2]; lmuv[p, IM + 1] = lmuv[p, IM + 2] = lmm[0, JM - 1]
    ravm[p, 1:IM + 1] = 1.0 / IM; ravm[p, IM + 1] = ravm[p, IM + 2] = 1.0
    mo = np.zeros((N, LMO + 1)); mo[:, 1:] = s['mo'][ii, jj, :]
    g0ml0 = np.zeros((N, LMO + 1)); g0ml0[:, 1:] = s['g0m'][ii, jj, :]
    s0ml0 = np.zeros((N, LMO + 1)); s0ml0[:, 1:] = s['s0m'][ii, jj, :]
    g0ml = g0ml0.copy(); s0ml = s0ml0.copy()
    g0ml[:, 1:LSRPD + 1] = s['g0m1'][ii, jj, :]; s0ml[:, 1] = s['s0m1'][ii, jj]
    lay = np.arange(LMO + 1)[None, :]
    for a in (mo, g0ml0, s0ml0, g0ml, s0ml):
        a[lay > lmij[:, None]] = 0.0
    mo1 = s['mo1'][ii, jj]
    with np.errstate(divide='ignore', invalid='ignore'):
        dtbydz = np.where(mo > 0, DTS / np.where(mo > 0, mo, 1), 0.0)
        bydz2 = np.zeros((N, LMO + 1))
        bydz2[:, 1:LMO] = np.where((mo[:, 2:] > 0), 2.0 / (mo[:, 2:] + mo[:, 1:LMO]), 0.0)
    bydts = 1.0 / DTS
    # surface stresses
    cosic = np.cos((np.arange(1, IM + 1) - 0.5) * 2 * np.pi / IM); sinic = np.sin((np.arange(1, IM + 1) - 0.5) * 2 * np.pi / IM)
    sg = lambda lm: np.where(lm > 0, 0.25, -0.25)   # noqa: E731
    uistr = np.zeros(N); vistr = np.zeros(N)
    rsi = fx['orsi'][ii, jj]
    a_u = (np.where(lmu[ii, jj] > 0, fx['odmui'][ii, jj], 0.0) + np.where(lmu[im1, jj] > 0, fx['odmui'][im1, jj], 0.0))
    anu = 1.0 - sg(lmu[ii, jj]) - sg(lmu[im1, jj])
    a_v = (np.where(lmv[ii, jj] > 0, fx['odmvi'][ii, jj], 0.0) + np.where(lmv[ii, jm1] > 0, fx['odmvi'][ii, jm1], 0.0))
    anv = 1.0 - sg(lmv[ii, jj]) - sg(lmv[ii, jm1])
    uistr = np.where(rsi > 0, a_u * anu, 0.0); vistr = np.where(rsi > 0, a_v * anv, 0.0)
    pu = 0.0; pv = 0.0
    for i in range(IM):        # sequential accumulation, OCNKPP.f:1757-1761
        pu = pu + fx['odmvi'][i, JM - 2] * cosic[i]; pv = pv - fx['odmvi'][i, JM - 2] * sinic[i]
    uistr[p] = pu / IM; vistr[p] = pv / IM
    u2rho = np.sqrt((fx['odmua'][ii, jj] + uistr) ** 2 + (fx['odmva'][ii, jj] + vistr) ** 2) * bydts
    deltam = (s['mo'][ii, jj, 0] - mo1) * bydts
    bydx = 1.0 / dxypo
    deltae = ((g0ml0[:, 1] - g0ml[:, 1]) * bydx) * bydts
    deltas = ((s0ml0[:, 1] - s0ml[:, 1]) * bydx) * bydts
    deltasr = (fx['osolarw'][ii, jj] * (1.0 - rsi) + fx['osolari'][ii, jj] * rsi) * bydts
    deltae = deltae - (1.0 - fsr2) * deltasr
    # PO (OCNKPP.f:1949-1957), sequential
    po = np.zeros((N, LMO + 1)); po[:, 1] = 0.5 * mo[:, 1] * GRAV
    for l in range(1, LMO):
        po[:, l + 1] = po[:, l] + 0.5 * GRAV * (mo[:, l] + mo[:, l + 1])
    args = dict(lmij=lmij, kmuv=kmuv, pole=pole, dts=DTS, dxypo=dxypo, mo=mo, mo1=mo1, deltae=deltae, deltas=deltas,
                deltam=deltam, deltasr=deltasr, u2rho=u2rho, ogeoz=s['ogeoz'][ii, jj], hocean=ctx['hocean'][ii, jj],
                s0m1=s['s0m1'][ii, jj], ravm=ravm, lmuv=lmuv, dtbydz=dtbydz, bydz2=bydz2, ul0=ul0, ulm=ul, uld0=uld0,
                uld=uld, g0ml0=g0ml0, s0ml0=s0ml0, g0ml=g0ml, s0ml=s0ml, po=po)
    return args, ii, jj


def stage_oconv(s, fx, ctx):
    import ocean_hbl
    s = copy_state(s)
    T = _oconv_tables()
    ut, vt, utd, vtd = s['uo'].copy(), s['vo'].copy(), s['uod'].copy(), s['vod'].copy()
    lmm, lmu, lmv = ctx['lmm'], ctx['lmu'], ctx['lmv']
    # entry slope limit (OCNKPP.f:1450-1462): |SZMO| > S0M, rows J=2..JM-1
    act3 = (np.arange(1, LMO + 1)[None, None, :] <= lmm[:, :, None])
    rows = np.zeros((IM, JM, 1), bool); rows[:, 1:JM - 1] = True
    big = act3 & rows & (np.abs(s['szmo']) > s['s0m'])
    s['szmo'] = np.where(big, np.copysign(s['s0m'], s['szmo']), s['szmo'])
    a, ii, jj = oconv_columns(s, fx, ctx)
    ze = T['ze']
    out = ocean_hbl.hbl_loop(ze, GRAV, a['lmij'], a['kmuv'], a['pole'], DTS, a['dxypo'], a['mo'], a['mo1'], a['deltae'],
                             a['deltas'], a['deltam'], a['deltasr'], a['u2rho'], a['ogeoz'], a['hocean'], a['s0m1'],
                             a['ravm'], a['lmuv'], a['dtbydz'], a['bydz2'], a['ul0'], a['ulm'], a['uld0'], a['uld'],
                             a['g0ml0'], a['s0ml0'], a['g0ml'], a['s0ml'], {}, T['tabs'], po=a['po'], vgsp=T['vg'],
                             agsp=T['ag'], bgsp=T['bg'], cgs=T['cg'])
    out = {k: np.asarray(v) for k, v in out.items()}
    N = len(ii); pole = a['pole']; lmij = a['lmij']
    dxys, dxyn = [np.asarray(x) for x in geomo_arrays_jax()[:2]]
    # G0M,S0M write-back
    for n in range(N):
        s['g0m'][ii[n], jj[n], :lmij[n]] = out['g0ml'][n, 1:lmij[n] + 1]
        s['s0m'][ii[n], jj[n], :lmij[n]] = out['s0ml'][n, 1:lmij[n] + 1]
    s['kpl'][ii, jj] = out['kbl']
    # velocity increments UKM / UKMD (non-pole columns)
    npn = np.nonzero(~pole)[0]
    im1 = (ii - 1) % IM; jm1 = np.maximum(jj - 1, 0)
    ramv = np.zeros((N, 5)); ramv[:, 1] = ramv[:, 2] = 0.5
    ramv[:, 3] = dxys[jj] / (dxyn[jm1] + dxys[jj]); ramv[:, 4] = dxyn[jj] / (dxyn[jj] + dxys[np.minimum(jj + 1, JM - 1)])
    Tk = [np.zeros((IM, JM, LMO)) for _ in range(4)]; Td = [np.zeros((IM, JM, LMO)) for _ in range(4)]
    src = [(ut, im1, jj), (ut, ii, jj), (vt, ii, jm1), (vt, ii, jj)]
    srcd = [(vtd, im1, jj), (vtd, ii, jj), (utd, ii, jm1), (utd, ii, jj)]
    lay = np.arange(1, LMO + 1)[None, :]
    for k in range(4):
        m = lay <= a['lmuv'][:, k + 1][:, None]
        uk = ramv[:, k + 1][:, None] * (out['ul'][:, 1:, k + 1] - src[k][0][src[k][1], src[k][2], :])
        ud = ramv[:, k + 1][:, None] * (out['uld'][:, 1:, k + 1] - srcd[k][0][srcd[k][1], srcd[k][2], :])
        Tk[k][ii[npn], jj[npn], :] = np.where(m, uk, 0.0)[npn]
        Td[k][ii[npn], jj[npn], :] = np.where(m, ud, 0.0)[npn]

    def add_i(f, T1, T2):      # (f + T2[i]) + T1[i+1]; wrap column: (f + T1[0]) + T2[IM-1]
        r = (f + T2) + np.roll(T1, -1, axis=0)
        r[IM - 1] = (f[IM - 1] + T1[0]) + T2[IM - 1]
        return r

    def add_j(f, T3, T4):      # (f + T4[j]) + T3[j+1]
        t3 = np.zeros_like(T3); t3[:, :JM - 1] = T3[:, 1:]
        return (f + T4) + t3
    s['uo'] = add_i(s['uo'], Tk[0], Tk[1]); s['vod'] = add_i(s['vod'], Td[0], Td[1])
    s['vo'] = add_j(s['vo'], Tk[2], Tk[3]); s['uod'] = add_j(s['uod'], Td[2], Td[3])
    # GZMO/SZMO flux update (adjust_zslope_using_flux, OCNKPP.f:2540-2565)
    akg = out['akvg'].copy(); aks = out['akvs'].copy()
    for ak in (akg, aks):
        ak[:, 0] = ak[:, 1]
        ak[np.arange(N), lmij] = ak[np.arange(N), lmij - 1]
    flg = out['flg3d']; fls = out['fls3d']
    dtb = a['dtbydz']
    mask = (np.arange(LMO + 1)[None, :] >= 1) & (np.arange(LMO + 1)[None, :] <= lmij[:, None])
    dtbydz2 = 6.0 * dtb ** 2 * (1.0 / DTS)
    gz = np.zeros((N, LMO + 1)); sz = np.zeros((N, LMO + 1))
    gz[:, 1:] = s['gzmo'][ii, jj, :]; sz[:, 1:] = s['szmo'][ii, jj, :]
    gzn = (gz[:, 1:] + 3.0 * (flg[:, 0:LMO] + flg[:, 1:LMO + 1])) / (1.0 + dtbydz2[:, 1:] * (akg[:, 0:LMO] + akg[:, 1:LMO + 1]))
    szn = (sz[:, 1:] + 3.0 * (fls[:, 0:LMO] + fls[:, 1:LMO + 1])) / (1.0 + dtbydz2[:, 1:] * (aks[:, 0:LMO] + aks[:, 1:LMO + 1]))
    gzn = np.where(mask[:, 1:], gzn, gz[:, 1:]); szn = np.where(mask[:, 1:], szn, sz[:, 1:])
    s['gzmo'][ii, jj, :] = gzn; s['szmo'][ii, jj, :] = szn
    # horizontal-moment vertical diffusion (non-pole rows), OVDIFFS with zero GHAT/DTP4
    from ovdiffs_jax import ovdiffs_jax
    zero = np.zeros((len(npn), LMO + 1))
    for fld, ak in (('gxmo', akg), ('gymo', akg), ('sxmo', aks), ('symo', aks)):
        u0 = np.zeros((len(npn), LMO + 1)); u0[:, 1:] = s[fld][ii[npn], jj[npn], :]
        u, _ = ovdiffs_jax(ak[npn], zero, zero, dtb[npn], a['bydz2'][npn], DTS, lmij[npn], u0)
        u = np.asarray(u)[:, 1:LMO + 1]
        old = s[fld][ii[npn], jj[npn], :]
        s[fld][ii[npn], jj[npn], :] = np.where(mask[npn, 1:], u, old)
    # final slope limits (OCNKPP.f:2830-2860), rows J=2..JM-1
    tiny = np.finfo(np.float64).tiny
    txy = np.abs(s['sxmo']) + np.abs(s['symo'])
    lim = act3 & rows & (txy > s['s0m'])
    f = s['s0m'] / (txy + tiny)
    s['sxmo'] = np.where(lim, s['sxmo'] * f, s['sxmo']); s['symo'] = np.where(lim, s['symo'] * f, s['symo'])
    big = act3 & rows & (np.abs(s['szmo']) > s['s0m'])
    s['szmo'] = np.where(big, np.copysign(s['s0m'], s['szmo']), s['szmo'])
    return s


STAGES.insert(2, ('oconv', stage_oconv, 4))


# ---------------------------------------------------------------- stages 6-11: dynamics (NO loop)
DTOLF, DTO, DTOFS = 900.0, 450.0, 300.0
Z12EH = 0.28867513


def to1(a):
    """0-based (IM,JM[,LMO]) -> 1-based zero-padded (IM+1,JM+1[,LMO+1]) used by the odhorz/oadvt/gm ports."""
    out = np.zeros((IM + 1, JM + 1) + ((LMO + 1,) if a.ndim == 3 else ()))
    out[(slice(1, None), slice(1, None)) + ((slice(1, None),) if a.ndim == 3 else ())] = a
    return out


def from1(a):
    return np.ascontiguousarray(a[(slice(1, None), slice(1, None)) + ((slice(1, None),) if a.ndim == 3 else ())])


def m_active(lmm):
    """nbyzm mask (IM,JM,LMO): lmm>=l, North Pole row only at I=1 (D40)."""
    l = np.arange(1, LMO + 1)[None, None, :]
    m = l <= lmm[:, :, None]
    m[1:, JM - 1, :] = False
    return m


def eos_vup_vdn(s, ctx, vg):
    """ODHORZ0 (OCNDYN2.f:1760-1790): VUP/VDN from the VGSP table; P by sequential top-down accumulation."""
    mo = s['mo']; lmm = ctx['lmm']
    mk = m_active(lmm)
    dx = _DXYPO[None, :, None]
    opb = np.where(mk[:, :, 0], s['opress'], 0.0)
    p = np.zeros_like(mo)
    for l in range(LMO):
        p[:, :, l] = np.where(mk[:, :, l], opb + mo[:, :, l] * GRAV * 0.5, 0.0)
        opb = np.where(mk[:, :, l], opb + mo[:, :, l] * GRAV, opb)
    mmi = np.where(mk, mo * dx, 1.0)
    gup = (s['g0m'] - 2 * Z12EH * s['gzmo']) / mmi; gdn = (s['g0m'] + 2 * Z12EH * s['gzmo']) / mmi
    sup = (s['s0m'] - 2 * Z12EH * s['szmo']) / mmi; sdn = (s['s0m'] + 2 * Z12EH * s['szmo']) / mmi
    smean = s['s0m'] / mmi
    sup = np.maximum(sup, 0.5 * smean); sdn = np.maximum(sdn, 0.5 * smean)
    pup = p - mo * GRAV * Z12EH; pdn = p + mo * GRAV * Z12EH
    vup = np.asarray(volgsp(vg, jnp.asarray(gup), jnp.asarray(sup), jnp.asarray(pup)))
    vdn = np.asarray(volgsp(vg, jnp.asarray(gdn), jnp.asarray(sdn), jnp.asarray(pdn)))
    return np.where(mk, vup, 0.0), np.where(mk, vdn, 0.0), mk


def run_odhorz0(s, ctx):
    from odhorz0_jax import odhorz0_jax
    T = _oconv_tables()
    s = copy_state(s)
    vup, vdn, mk = eos_vup_vdn(s, ctx, T['vg'])
    lmm = np.asarray(ctx['lmm'], dtype=np.float64); lmv = np.asarray(ctx['lmv'], dtype=np.float64)
    r = {k: np.asarray(v) for k, v in odhorz0_jax(lmm, lmv, s['opress'], s['g0m'], s['gzmo'], s['s0m'], s['szmo'],
                                                 s['mo'], s['uo'], s['vo'], vup, vdn).items()}
    s['mmi'] = np.where(mk, s['mo'] * _DXYPO[None, :, None], s['mmi'])
    for k in ('opbot', 'vbar', 'dzgdp', 'dh3d', 'mo', 'uo', 'vo', 'gup', 'gdn', 'sup', 'sdn'):
        s[k] = r[k].copy()
    return s


def stage_odhorz0(s, fx, ctx):
    return run_odhorz0(s, ctx)


def _opfil_ops(ctx):
    if 'opf_T' not in _OC:
        import opfil2_jax as OJ
        from opfil2_ff import coef_from_dump
        v = np.fromfile(ctx['opcoef'], dtype='>f8').astype(np.float64)
        _OC['opf_T'] = OJ.build_operators(coef_from_dump(v))
    return _OC['opf_T']


def odhorz_prefilter(ctx, moh, uoh, voh, opboth, vbar, dzgdp, hocean):
    """USMOOTH and PGFX as ODHORZ leaves them for each layer (OCNDYN2.f:1265-1375), including the OPFIL2 calls and the
    array persistence across layers. Replaces the recorded post-OPFIL2 arrays. Inputs 0-based (IM,JM[,LMO])."""
    import opfil2_jax as OJ
    Top = _opfil_ops(ctx)
    lmm, lmu = ctx['lmm'], ctx['lmu']
    dxpo = geo1()[2][1:]
    ma1 = m_active(lmm)[:, :, 0]
    pdn = np.where(ma1, opboth, 0.0); ogeoz = np.where(ma1, -hocean * GRAV, 0.0)
    us = np.zeros((IM, JM)); pg = np.zeros((IM, JM))
    us_out = np.zeros((IM, JM, LMO)); pg_out = np.zeros((IM, JM, LMO))
    rows = np.zeros(JM, bool); rows[1:JM - 1] = True
    for l in range(LMO, 0, -1):
        ma = m_active(lmm)[:, :, l - 1]
        ma = ma.copy(); ma[:, JM - 1] = ma[:, JM - 1]
        m = moh[:, :, l - 1]
        dp = m * GRAV; dh = np.where(ma, m * vbar[:, :, l - 1], 0.0)
        p = np.where(ma, pdn - 0.5 * dp, 0.0)
        zg = np.where(ma, ogeoz + dp * 0.5 * dzgdp[:, :, l - 1], 0.0)
        pdn = np.where(ma, pdn - dp, pdn); ogeoz = np.where(ma, ogeoz + dh * GRAV, ogeoz)
        uact = (lmu >= l) & rows[None, :]
        anyu = uact.any()
        if anyu:
            us = np.where(uact, uoh[:, :, l - 1], us)
            us = np.asarray(OJ.apply_operators(Top, us, l, 2, JM - 1)).copy()
        us[:, JM - 1] = uoh[:, JM - 1, l - 1]
        mmid = m + np.roll(m, -1, axis=0)
        with np.errstate(all='ignore'):
            new = (((zg - np.roll(zg, -1, axis=0)) + (p - np.roll(p, -1, axis=0)) * (dh + np.roll(dh, -1, axis=0)) / mmid)
                   * (1.0 / dxpo)[None, :])
        pg = np.where(uact, new, pg)
        if anyu:
            pg = np.asarray(OJ.apply_operators(Top, pg, l, 2, JM - 1)).copy()
        us_out[:, :, l - 1] = us; pg_out[:, :, l - 1] = pg
    return us_out, pg_out, ogeoz


_GEO = {}


def geo1():
    if not _GEO:
        from odhorz_ff import geomo_dyn_arrays
        _GEO['g'] = geomo_dyn_arrays()
    return _GEO['g']


def stage_dynamics(s, fx, ctx):
    """ODHORZ0 (first), SMU/SMV reset and state copies, five ODHORZ calls, VONP, OFLUXV, CONSERV (diagnostic only),
    OADVT2 for G0M and S0M. Returns the state before the straits."""
    from ocean_odhorz import odhorz_jax, make_layer_step
    from ocean_ofluxv import ofluxv_jax
    from oadvt_jax import oadvt2_jax
    s = run_odhorz0(s, ctx)
    lmm1, lmu1, lmv1 = to1(ctx['lmm']).astype(np.int64), to1(ctx['lmu']).astype(np.int64), to1(ctx['lmv']).astype(np.int64)
    hoc1 = to1(ctx['hocean'])
    s['smu'] = np.zeros_like(s['mo']); s['smv'] = np.zeros_like(s['mo'])
    mk = m_active(ctx['lmm']); ku = ctx['lmu'][:, :, None] >= np.arange(1, LMO + 1)[None, None, :]
    kv = ctx['lmv'][:, :, None] >= np.arange(1, LMO + 1)[None, None, :]
    ku[:, JM - 1] = False; kv[:, JM - 1] = False

    def odd_state():
        d = {}
        d['mo'] = np.where(mk, s['mo'], 0.0); d['uo'] = np.where(ku, s['uo'], 0.0); d['vod'] = np.where(ku, s['vod'], 0.0)
        d['vo'] = np.where(kv, s['vo'], 0.0); d['uod'] = np.where(kv, s['uod'], 0.0)
        for k in ('mo', 'uo', 'vo'):
            d[k][:, JM - 1, :] = s[k][:, JM - 1, :]
        d['opbot'] = np.where(mk[:, :, 0], s['opbot'], 0.0)
        return d
    st1 = odd_state(); st2 = {k: v.copy() for k, v in st1.items()}
    main = {k: s[k].copy() for k in ('mo', 'uo', 'vo', 'uod', 'vod', 'opbot')}
    step = make_layer_step(lmm1, lmu1, lmv1)

    def call(H, IO, dt, qeven):
        us, pg, og = odhorz_prefilter(ctx, H['mo'], H['uo'], H['vo'], H['opbot'], s['vbar'], s['dzgdp'], ctx['hocean'])
        a = lambda d, k: to1(d[k])   # noqa: E731
        mo, uo, vo, uod, vod, opbot, mu3, mv3 = odhorz_jax(
            lmm1, lmu1, lmv1, hoc1, dt, a(H, 'mo'), a(H, 'uo'), a(H, 'vo'), a(H, 'uod'), a(H, 'vod'), to1(H['opbot']),
            a(IO, 'mo'), a(IO, 'uo'), a(IO, 'vo'), a(IO, 'uod'), a(IO, 'vod'), to1(IO['opbot']),
            to1(s['vbar']), to1(s['dzgdp']), to1(us), to1(pg), step=step)
        s['ogeoz'] = np.where(m_active(ctx['lmm'])[:, :, 0], og, s['ogeoz'])   # OGEOZ module array: last call wins
        out = dict(mo=from1(mo), uo=from1(uo), vo=from1(vo), uod=from1(uod), vod=from1(vod), opbot=from1(opbot))
        if qeven:
            s['smu'] = s['smu'] + from1(mu3) * 1.0; s['smv'] = s['smv'] + from1(mv3) * 1.0
        return out
    st2 = call(main, st2, DTOFS, False)
    st1 = call(st2, st1, DTO, False)
    neven = 2
    for n in range(1, neven + 1):
        main = call(st1, main, DTOLF, True)
        if n == neven:
            break
        st1 = call(main, st1, DTOLF, False)
    for k in ('mo', 'uo', 'vo', 'uod', 'vod', 'opbot'):
        s[k] = main[k]
    s['uo'][IVNP0, JM - 1, :] = s['vonp']
    mo, uo, vo, smw = ofluxv_jax(np.asarray(ctx['lmm'], dtype=np.float64), np.asarray(ctx['lmu'], dtype=np.float64),
                                  np.asarray(ctx['lmv'], dtype=np.float64), DTOLF, s['opbot'], s['opress'], s['mo'], s['uo'], s['vo'])
    s['mo'], s['uo'], s['vo'] = [np.asarray(x).copy() for x in (mo, uo, vo)]
    lidx = np.arange(1, LMO + 1)[None, None, :]
    wr = (lidx + 1 <= ctx['lmm'][:, :, None]); wr[:, JM - 1, :] = (lidx[0] + 1 <= ctx['lmm'][0, JM - 1])
    s['smw'] = np.where(wr, np.asarray(smw) * (_DXYPO[None, :, None] / DTOLF), 0.0)   # OFLUXV's SMW carries DXYPO(J)/DTOLF (D43 note)
    # stale entries of the Fortran SMW (cells OFLUXV does not write) are not reproduced
    for r, qk in ((('g0m', 'gxmo', 'gymo', 'gzmo'), False), (('s0m', 'sxmo', 'symo', 'szmo'), True)):
        ma, rm, rx, ry, rz = oadvt2_jax(to1(s['mmi']), *[to1(s[k]) for k in r], DTOLF, qk, to1(s['smu']), to1(s['smv']),
                                        to1(s['smw']), lmu1, lmv1, lmm1)
        for k, v in zip(r, (rm, rx, ry, rz)):
            s[k] = from1(np.asarray(v))
    return s


def stage_straits(s, fx, ctx):
    from straits_step_jax import straits_step
    from stconv_jax import stconv_jax  # noqa: F401
    T = _oconv_tables()
    s = copy_state(s)
    ist, jst = ctx['ist'] - 1, ctx['jst'] - 1
    nm = ctx['nmst']
    names = [('moe', 'mo'), ('g0me', 'g0m'), ('gxme', 'gxmo'), ('gyme', 'gymo'), ('gzme', 'gzmo'),
             ('s0me', 's0m'), ('sxme', 'sxmo'), ('syme', 'symo'), ('szme', 'szmo')]
    me = {}
    for a, b in names:      # gather: (N,2,LMO+1)
        arr = np.zeros((nm, 2, LMO + 1)); arr[:, :, 1:] = np.stack([s[b][ist[:, k], jst[:, k], :] for k in (0, 1)], axis=1)
        me[a] = arr
    pad = lambda a: np.concatenate([np.zeros((nm, 1)), a.T], axis=1)   # noqa: E731
    state = dict(must=pad(s['must']), mmst=pad(s['mmst']), g0=pad(s['g0mst']), gx=pad(s['gxmst']), gz=pad(s['gzmst']),
                 s0=pad(s['s0mst']), sx=pad(s['sxmst']), sz=pad(s['szmst']), lmst=ctx['lmst'],
                 lmme=np.stack([ctx['lmm'][ist[:, k], jst[:, k]] for k in (0, 1)], axis=1),
                 oprese=np.stack([s['opress'][ist[:, k], jst[:, k]] for k in (0, 1)], axis=1),
                 hoceane=np.stack([ctx['hocean'][ist[:, k], jst[:, k]] for k in (0, 1)], axis=1),
                 distpg=ctx['distpg'], wist=ctx['wist'], dist=ctx['dist'], dts=DTS, nmst=nm, sinpo=geo1()[1][1:])
    geo = dict(ist=ctx['ist'], jst=ctx['jst'], xst=ctx['xst'], yst=ctx['yst'])
    dxypo1 = np.concatenate([[0.0], _DXYPO])
    st, me2 = straits_step(T['vg'], T['ze'], geo, dxypo1, state, me, T['tabs'])
    for k, kk in (('must', 'must'), ('g0', 'g0mst'), ('gx', 'gxmst'), ('gz', 'gzmst'), ('s0', 's0mst'), ('sx', 'sxmst'), ('sz', 'szmst')):
        s[kk] = np.asarray(st[k])[:, 1:].T.copy()
    for a, b in names:      # scatter, pairs in order (n,k)
        arr = np.asarray(me2[a])
        for n in range(nm):
            for k in (0, 1):
                s[b][ist[n, k], jst[n, k], :] = arr[n, k, 1:]
    return s


STAGES += [('dynamics', stage_dynamics, 10), ('straits', stage_straits, 11)]


# ---------------------------------------------------------------- stages 12-14: post-dynamics, ODIFF (recorded), OCNMESO
def stage_post(s, fx, ctx):
    """Second ODHORZ0 (recalculate vbar etc.) -> state at the ODIFF point (snapshot 12)."""
    return run_odhorz0(s, ctx)


def stage_odiff(s, fx, ctx):
    """ODIFF is NOT ported: on steps with mod(itime,6)==0 the recorded post-ODIFF UO,VO,VONP replace the state."""
    s = copy_state(s)
    rec = fx.get('odiff')
    if fx.get('itime', 1) % 6 == 0:
        if rec is None:
            raise RuntimeError('ODIFF step without a recorded boundary (ODIFF is not ported)')
        s['uo'], s['vo'], s['vonp'] = rec['uo'].copy(), rec['vo'].copy(), rec['vonp'].copy()
    return s


def stage_meso(s, fx, ctx):
    from ocnmeso_jax import ocnstate_derived_jax, densgrad_vertical_jax
    from gm_jax import isoslope4_jax, gmkdif_jax, gmfexp_jax
    s = copy_state(s)
    if 'dh3d' not in s:      # replay from a bare snapshot: DH is the ODHORZ0 output of the same state
        s['dh3d'] = run_odhorz0(s, ctx)['dh3d']
    T = _oconv_tables(); vg = T['vg']
    lmm1 = to1(ctx['lmm']).astype(np.int64); lmu1 = to1(ctx['lmu']).astype(np.int64); lmv1 = to1(ctx['lmv']).astype(np.int64)
    mo, g0m, s0m = s['mo'], s['g0m'], s['s0m']
    mk = m_active(ctx['lmm'])
    with np.errstate(all='ignore'):
        bym = np.where(mk, 1.0 / (mo * _DXYPO[None, :, None]), 0.0)
    pe = np.zeros((IM, JM, LMO + 1)); pe[:, :, 0] = s['opress']
    for l in range(LMO):
        pe[:, :, l + 1] = np.where(mk[:, :, l], pe[:, :, l] + mo[:, :, l] * GRAV, pe[:, :, l])
    pm = 0.5 * (pe[:, :, 1:] + pe[:, :, :-1])
    gup = (g0m - 2 * Z12EH * s['gzmo']) * bym; gdn = (g0m + 2 * Z12EH * s['gzmo']) * bym
    sup = np.maximum(0.0, (s0m - 2 * Z12EH * s['szmo']) * bym); sdn = np.maximum(0.0, (s0m + 2 * Z12EH * s['szmo']) * bym)
    V = lambda g, sa, p: np.asarray(volgsp(vg, jnp.asarray(g), jnp.asarray(sa), jnp.asarray(p)))   # noqa: E731
    vup = np.where(mk, V(gup, sup, pm), 0.0); vdn = np.where(mk, V(gdn, sdn, pm), 0.0)
    vupu = np.zeros_like(vup); vdnu = np.zeros_like(vup)
    vupu[:, :, 1:] = np.where(mk[:, :, 1:], V(gup[:, :, :-1], sup[:, :, :-1], pm[:, :, 1:]), 0.0)
    vdnu[:, :, 1:] = np.where(mk[:, :, 1:], V(gdn[:, :, :-1], sdn[:, :, :-1], pm[:, :, 1:]), 0.0)
    g3d, s3d, p3d, vbar, rho = [np.asarray(x) for x in ocnstate_derived_jax(
        to1(mo), to1(g0m), to1(s['gzmo']), to1(s0m), to1(s['szmo']), to1(s['opress']), lmm1, to1(vup), to1(vdn))]
    dzv, bydzv, bydh, rhomz, byrhoz = [np.asarray(x) for x in densgrad_vertical_jax(lmm1, to1(s['dh3d']), vbar, to1(vup), to1(vdn), to1(vupu), to1(vdnu))]
    # horizontal density gradients (OCNMESO_DRV.f:520-575), 1-based arrays
    g1 = geo1(); dxpo, dyvo = g1[2], g1[5]
    rhox = np.zeros((IM + 1, JM + 1, LMO + 1)); rhoy = np.zeros_like(rhox)
    Lr = np.arange(LMO + 1)[None, None, :]
    jr_ = slice(2, JM)
    X = lambda g, sa, p_: 1.0 / np.asarray(volgsp(vg, jnp.asarray(g), jnp.asarray(sa), jnp.asarray(p_)))   # noqa: E731
    gl, sl, pl = g3d[1:, jr_, :], s3d[1:, jr_, :], p3d[1:, jr_, :]
    gr, sr, pr = np.roll(gl, -1, axis=0), np.roll(sl, -1, axis=0), np.roll(pl, -1, axis=0)
    with np.errstate(all='ignore'):
        p12 = 0.5 * (pl + pr)
        rx = (X(gr, sr, p12) - X(gl, sl, p12)) * (1.0 / dxpo[jr_])[None, :, None]
        gn, sn_, pn = g3d[1:, 3:JM + 1, :], s3d[1:, 3:JM + 1, :], p3d[1:, 3:JM + 1, :]
        p12y = 0.5 * (pl + pn)
        ry = (X(gn, sn_, p12y) - X(gl, sl, p12y)) * (1.0 / dyvo[jr_])[None, :, None]
    u_act = (lmu1[1:, jr_, None] >= Lr) & (Lr >= 1)
    v_act = (lmv1[1:, jr_, None] >= Lr) & (Lr >= 1)
    rhox[1:, jr_, :] = np.where(u_act, rx, 0.0); rhoy[1:, jr_, :] = np.where(v_act, ry, 0.0)
    k3d = np.where(np.arange(LMO + 1)[None, None, :] <= lmm1[:, :, None], 800.0, 0.0); k3d[:, :, 0] = 0.0
    iso = isoslope4_jax(lmm1, rhox, rhoy, rhomz, byrhoz, bydh, dzv, k3d)
    ISO = ["aix0", "aix1", "aix2", "aix3", "aiy0", "aiy1", "aiy2", "aiy3", "asx0", "asx1", "asx2", "asx3",
           "asy0", "asy1", "asy2", "asy3", "s2x0", "s2x1", "s2x2", "s2x3", "s2y0", "s2y1", "s2y2", "s2y3"]
    from gm_vec_compare import ISO_ORDER
    kpl1 = to1(s['kpl']).astype(np.int64)
    gmk = gmkdif_jax(lmm1, kpl1, *[iso[n] for n in ISO_ORDER])
    names = ("bxx", "byy", "bzz", "azx", "bzx", "czx", "aezx", "ezx", "cezx", "azy", "bzy", "czy", "aezy", "ezy", "cezy")
    for fields, ql in ((('g0m', 'gxmo', 'gymo', 'gzmo'), False), (('s0m', 'sxmo', 'symo', 'szmo'), True)):
        r = gmfexp_jax(lmm1, lmu1, lmv1, to1(mo), *[to1(s[k]) for k in fields], ql, *[gmk[n] for n in names], kpl1, bydh, bydzv)
        for k, v in zip(fields, r):
            s[k] = from1(np.asarray(v))
    return s


STAGES += [('post', stage_post, 12), ('odiff', stage_odiff, 13), ('meso', stage_meso, 14)]


ORDER = ['precip', 'ground', 'ostres', 'oconv', 'drag', 'polar', 'dynamics', 'straits', 'post', 'odiff', 'meso']
STAGES = sorted(STAGES, key=lambda t: ORDER.index(t[0]))


def ocean_step(state, fx, ctx, trace=None, stages=None):
    """One live ocean step (PRECIP_OC + OCEANS) from `state` (state at PRECIP_OC entry, i.e. snapshot tag 0).
    fx: recorded atm/ice-to-ocean fluxes (ffo tags 0 and 1), 'itime', and 'odiff' (recorded ODIFF result, only read when
    mod(itime,6)==0). `stages` optionally overrides the stage list (used by the mutation tests).
    trace(name, state) is called after every stage."""
    s = state
    for name, fn, _tag in (stages if stages is not None else STAGES):
        s = fn(s, fx, ctx)
        if trace is not None:
            trace(name, s)
    return s
