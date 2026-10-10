"""D190 (stage S5): the ocean step (PRECIP_OC + OCEANS, the stages of ocean_step.STAGES with the ported ODIFF of ocean_step_odiff) as device-resident jax.numpy functions.

NEW module; ocean_step.py and everything it imports are used but not edited.  Each stage mirrors the NumPy glue of ocean_step.py / ocean_step_odiff.py statement by
statement; the jitted kernels (osourc, ground_oc_sweep, ostres2, obdrag2, ocoast, polerelax, odhorz0, odhorz layer step, ofluxv, oadvt2 sweeps, the OCONV HBL loop, ovdiffs,
straits kernels, ocnmeso/gm) are called unchanged.  What changes:
  (1) np -> jnp, with `where` for the selections of the NumPy code;
  (2) the cell lists of np.nonzero become static index arrays/masks computed once on the host from the static ocean geometry (`make_static`; they depend on LMM/LMU/LMV/FOCEAN
      only, never on the state);
  (3) in-place assignments become `.at[].set`; host loops over columns/layers keep their order (python loops unrolled at trace time, lax.scan/fori where the trip count is long);
  (4) branches that depend only on static geometry (e.g. `anyu`) are evaluated at trace time; the only state-dependent Python branch of the originals, `itime % 6 == 0` (ODIFF),
      is a lax.cond on the traced step number;
  (5) the one remaining host stage inside the ocean step is the X pre-pass of OADVT2 (oadvt_vec._x_prepass: the MUDT/courant bookkeeping of the east-west advection); it is a
      jax.pure_callback to the original NumPy function (4 calls per ocean step), declared in the report.
Exponents: every `x ** n` with n >= 3 of the NumPy glue would be written jnp.power(x, float(n)) (D188); the glue only has `** 2` (identical in both).
The ocean state is a dict of arrays in the layout of ocean_step.py ((IM,JM,LMO) etc.); stage functions are pure (K, s, fx) -> s.
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

import ocean_step as O  # noqa: E402
from precip_oc_jax import precip_oc_cell  # noqa: E402
from osourc_jax import osourc, gfrezs  # noqa: E402
from ground_oc_sweep_jax import ground_oc_sweep_layer  # noqa: E402
from ostres2_jax import ostres2_jax, geomo_arrays_jax  # noqa: E402
from obdrag2_jax import obdrag2_jax  # noqa: E402
from ocoast_jax import ocoast_jax  # noqa: E402
from polerelax_jax import polerelax_jax  # noqa: E402
from eos_jax import volgsp  # noqa: E402
import jax_ocean_hbl as HB  # noqa: E402

IM, JM, LMO = 72, 46, 13
LSRPD = 3
GRAV = O.GRAV
DTS = O.DTS
IVNP0 = O.IVNP0
_DXYPO = O._DXYPO
DTOLF, DTO, DTOFS = O.DTOLF, O.DTO, O.DTOFS
Z12EH = O.Z12EH


# ====================================================================================================== statics
def make_static(st, K):
    """Host-side constants of the ocean step from surface_loop's statics `st` (ctx = the ocean ctx) and the post-tile statics K.  Returns (Ko, Kb): Ko = small numpy
    constants closed over by the jitted functions, Kb = big numpy arrays that are passed as jit ARGUMENTS (not embedded as constants)."""
    ctx = st['ctx']
    Ko, Kb = {}, {}
    lmm, lmu, lmv = (np.asarray(ctx[k]) for k in ('lmm', 'lmu', 'lmv'))
    Ko['ctx'] = ctx
    Ko['lmm'], Ko['lmu'], Ko['lmv'] = lmm, lmu, lmv
    Ko['valid'] = np.asarray(K['valid'])
    foc = np.asarray(ctx['focean'])
    # ---- GROUND_OC (stage_ground): selected cells, DXYPJ / BYDXYPJ accumulated along I within a row (OCNDYN.f:4690-4692)
    sel = O.imaxj_mask() & (foc > 0)
    ii, jj = np.nonzero(sel)
    dxypj = np.ones((IM, JM))
    bydxypj = np.ones((IM, JM))
    for j in np.unique(jj):
        d, b = _DXYPO[j], 1.0 / _DXYPO[j]
        for n in np.nonzero(jj == j)[0]:
            d = d * foc[ii[n], j]
            b = b / foc[ii[n], j]
            dxypj[ii[n], j] = d
            bydxypj[ii[n], j] = b
    Ko['g_sel'], Ko['g_dxypj'], Ko['g_bydxypj'] = sel, dxypj, bydxypj
    Ko['dxypo_grid'] = np.repeat(_DXYPO[None, :], IM, axis=0)
    Ko['fgeotherm'] = np.asarray(ctx['fgeotherm'])
    Ko['cgs'] = tables()['cg']
    _static_oconv(Ko, ctx)
    _static_dynamics(Ko, Kb, ctx)
    _static_straits(Ko, ctx)
    _static_odiff(Ko, Kb, ctx)
    return Ko, Kb


# ====================================================================================================== stage 0: KVINIT + PRECIP_OC
def stage_precip(K, Kb, s, fx):
    """ocean_step.stage_precip: KVINIT snapshot (copies) and PRECIP_OC on layer 1."""
    s = dict(s)
    s.update(g0m1=s['g0m'][:, :, :LSRPD], s0m1=s['s0m'][:, :, 0], mo1=s['mo'][:, :, 0], gxm1=s['gxmo'][:, :, 0], gym1=s['gymo'][:, :, 0],
             sxm1=s['sxmo'][:, :, 0], sym1=s['symo'][:, :, 0], uo1=s['uo'][:, :, 0], vo1=s['vo'][:, :, 0], uod1=s['uod'][:, :, 0],
             vod1=s['vod'][:, :, 0])
    foc = jnp.asarray(K['focean'])
    act = jnp.asarray(K['valid']) & (foc > 0) & (fx['oprec'] > 0)
    dx = jnp.asarray(K['dxypo'])[None, :]
    r = precip_oc_cell(foc, fx['oprec'], fx['orsi'], fx['orunpsi'], fx['oeprec'], fx['oerunpsi'], fx['osrunpsi'], dx,
                       s['mo'][:, :, 0], s['g0m'][:, :, 0], s['s0m'][:, :, 0])
    for k in ('mo', 'g0m', 's0m'):
        s[k] = s[k].at[:, :, 0].set(jnp.where(act, r[k], s[k][:, :, 0]))
    return s


# ====================================================================================================== table lookups (jnp copies of ocean_step.shcgs)
def _shcgs(cgs, g, s):
    gg = g * 2.5e-4
    ss = s * 1000.0
    ig = jnp.clip(jnp.trunc(gg + 2.0).astype(jnp.int64) - 2, -2, 39)
    js = jnp.trunc(ss).astype(jnp.int64)
    js = jnp.where(js >= 40, 39, js)
    c = lambda i, j: cgs[i + 2, j]   # noqa: E731
    return ((js - ss + 1) * ((ig - gg + 1) * c(ig, js) + (gg - ig) * c(ig + 1, js))
            + (ss - js) * ((ig - gg + 1) * c(ig, js + 1) + (gg - ig) * c(ig + 1, js + 1)))


_TAB = {}


def tables():
    """OFTAB tables used by the ocean (numpy, host): VGSP, AGSP, BGSP, CGS and the kmixinit/init_solar tables of OCONV (ocean_step._oconv_tables)."""
    if not _TAB:
        T = O._oconv_tables()
        _TAB.update(T)
    return _TAB


# ====================================================================================================== stage 1: GROUND_OC
def stage_ground(K, Kb, s, fx):
    Ko = K['oc']
    s = dict(s)
    NC = IM * JM
    sel = jnp.asarray(Ko['g_sel'])
    dxypj = jnp.asarray(Ko['g_dxypj']).reshape(NC)
    bydxypj = jnp.asarray(Ko['g_bydxypj']).reshape(NC)
    dxypo = jnp.asarray(Ko['dxypo_grid'])
    lm = jnp.asarray(Ko['lmm']).reshape(NC)
    fl = lambda k: fx[k].reshape(NC)   # noqa: E731
    runo = (fl('oflowo') + fl('omelti')) - fl('oevapor')
    runi = (fl('oflowo') + fl('omelti')) + fl('orunosi')
    eruno = (fl('oeflowo') + fl('oemelti')) + fl('oe0')
    eruni = (fl('oeflowo') + fl('oemelti')) + fl('oerunosi')
    sruno = fl('osmelti')
    sruni = fl('osmelti') + fl('osrunosi')
    g0ml = s['g0m'].reshape(NC, LMO)
    gzml = s['gzmo'].reshape(NC, LMO)
    onehot = jnp.arange(LMO)[None, :] == (lm - 1)[:, None]
    geo_e = (DTS * jnp.asarray(Ko['fgeotherm']) * dxypo).reshape(NC)
    g0ml = g0ml + jnp.where(onehot, geo_e[:, None], 0.0)
    r = osourc(fl('orsi'), s['mo'][:, :, 0].reshape(NC), g0ml, gzml, s['s0m'][:, :, 0].reshape(NC), dxypj, bydxypj, lm,
               runo, runi, eruno, eruni, sruno, sruni, fl('osolarw'), fl('osolari'))
    mo = s['mo'].reshape(NC, LMO)
    g0 = r['g0ml']
    s0 = s['s0m'].reshape(NC, LMO)
    mo = mo.at[:, 0].set(r['mo'])
    s0 = s0.at[:, 0].set(r['s0m'])
    gz = r['gzml']
    p0l = mo[:, 0] * GRAV
    sdm = jnp.zeros(NC)
    sde = jnp.zeros(NC)
    sds = jnp.zeros(NC)
    cgs = jnp.asarray(Ko['cgs'])
    for l in range(2, LMO + 1):
        act = lm >= l
        mol = jnp.where(act, mo[:, l - 1], 1.0)
        g0l = jnp.where(act, g0[:, l - 1], 0.0) / (mol * dxypj)
        s0l = jnp.where(act, s0[:, l - 1], 0.0) / (mol * dxypj)
        p0l_a = p0l + mol * GRAV * 0.5
        gf00 = gfrezs(s0l)
        pcorr = (_shcgs(cgs, gf00, s0l) * 8.19e-8) * p0l_a
        o = ground_oc_sweep_layer(mol, jnp.where(act, g0[:, l - 1], 0.0), jnp.where(act, s0[:, l - 1], 0.0), dxypj, pcorr, p0l_a)
        mo = mo.at[:, l - 1].set(jnp.where(act, o['mo'], mo[:, l - 1]))
        g0 = g0.at[:, l - 1].set(jnp.where(act, o['g0m'], g0[:, l - 1]))
        s0 = s0.at[:, l - 1].set(jnp.where(act, o['s0m'], s0[:, l - 1]))
        sdm = sdm + jnp.where(act, o['dm0'], 0.0)
        sde = sde + jnp.where(act, o['de0'], 0.0)
        sds = sds + jnp.where(act, o['ds0'], 0.0)
        p0l = jnp.where(act, p0l_a + mo[:, l - 1] * GRAV * 0.5, p0l)
    sel3 = sel[:, :, None]
    s['mo'] = jnp.where(sel3, mo.reshape(IM, JM, LMO), s['mo'])
    s['g0m'] = jnp.where(sel3, g0.reshape(IM, JM, LMO), s['g0m'])
    s['s0m'] = jnp.where(sel3, s0.reshape(IM, JM, LMO), s['s0m'])
    s['gzmo'] = jnp.where(sel3, gz.reshape(IM, JM, LMO), s['gzmo'])
    g2 = lambda a: a.reshape(IM, JM)   # noqa: E731
    z2 = jnp.zeros((IM, JM))
    dmsi = jnp.stack([jnp.where(sel, g2(r['dmoo'] + sdm), z2), jnp.where(sel, g2(r['dmoi'] + sdm), z2)])
    dhsi = jnp.stack([jnp.where(sel, g2(r['deoo'] + sde), z2), jnp.where(sel, g2(r['deoi'] + sde), z2)])
    dssi = jnp.stack([jnp.where(sel, g2(r['dsoo'] + sds), z2), jnp.where(sel, g2(r['dsoi'] + sds), z2)])
    rsi = fx['orsi']
    op = fx['oapress'] + GRAV * ((1.0 - rsi) * dmsi[0] + rsi * dmsi[1])
    opress = jnp.where(sel, op, s['opress'])
    opress = opress.at[1:, 0].set(opress[0, 0])
    opress = opress.at[1:, JM - 1].set(opress[0, JM - 1])
    s['opress'] = opress
    s['odmsi'], s['odhsi'], s['odssi'] = dmsi, dhsi, dssi
    return s


# ====================================================================================================== stage 2: OSTRES2, stage 4: OBDRAG2 + OCOAST, stage 5: polar
def stage_ostres(K, Kb, s, fx):
    Ko = K['oc']
    s = dict(s)
    uo, vo, uod, vod = ostres2_jax(Ko['lmu'], Ko['lmv'], fx['odmua'], fx['odmva'], fx['odmui'], fx['odmvi'], s['mo'][:, :, 0], s['uo'][:, :, 0],
                                   s['vo'][:, :, 0], s['uod'][:, :, 0], s['vod'][:, :, 0], IVNP0)
    for k, a in (('uo', uo), ('vo', vo), ('uod', uod), ('vod', vod)):
        s[k] = s[k].at[:, :, 0].set(a)
    return s


def stage_drag(K, Kb, s, fx):
    Ko = K['oc']
    s = dict(s)
    lmu, lmv, lmm = (np.asarray(Ko[k], dtype=np.float64) for k in ('lmu', 'lmv', 'lmm'))
    uo, vo, uod, vod = obdrag2_jax(lmu, lmv, s['mo'], s['uo'], s['vo'], s['uod'], s['vod'])
    gx, sx, gy, sy = ocoast_jax(lmm, s['gxmo'], s['sxmo'], s['gymo'], s['symo'])
    for k, a in (('uo', uo), ('vo', vo), ('uod', uod), ('vod', vod), ('gxmo', gx), ('sxmo', sx), ('gymo', gy), ('symo', sy)):
        s[k] = a
    return s


def stage_polar(K, Kb, s, fx):
    Ko = K['oc']
    s = dict(s)
    lmu, lmv = (np.asarray(Ko[k], dtype=np.float64) for k in ('lmu', 'lmv'))
    uo, vo, uod, vod = polerelax_jax(lmu, lmv, s['uo'], s['vo'], s['uod'], s['vod'])
    for k, a in (('uo', uo), ('vo', vo), ('uod', uod), ('vod', vod)):
        s[k] = a
    return s


# ====================================================================================================== stage 3: OCONV
KM = 74


def _static_oconv(Ko, ctx):
    """Column lists and the state-independent parts of ocean_step.oconv_columns / stage_oconv (OCNKPP.f:1648-2860)."""
    lmm, lmu, lmv = Ko['lmm'], Ko['lmu'], Ko['lmv']
    msk = (lmm > 1)
    msk[:, 0] = False
    msk[:, JM - 1] = False
    jj, ii = np.nonzero(msk.T)
    jj = np.append(jj, JM - 1)
    ii = np.append(ii, 0)
    N = len(ii)
    pole = np.zeros(N, bool)
    pole[-1] = True
    im1 = (ii - 1) % IM
    jm1 = np.maximum(jj - 1, 0)
    dxys, dxyn = [np.asarray(a) for a in geomo_arrays_jax()[:2]]
    n = np.nonzero(~pole)[0]
    lmij = lmm[ii, jj].astype(np.int64)
    kmuv = np.where(pole, IM + 2, 4).astype(np.int64)
    lmuv = np.zeros((N, KM + 1), np.int64)
    ravm = np.zeros((N, KM + 1))
    lmuv[n, 1] = lmu[im1[n], jj[n]]
    lmuv[n, 2] = lmu[ii[n], jj[n]]
    lmuv[n, 3] = lmv[ii[n], jm1[n]]
    lmuv[n, 4] = lmv[ii[n], jj[n]]
    ravm[n, 1:5] = 0.5
    p = N - 1
    lmuv[p, 1:IM + 1] = lmv[:, JM - 2]
    lmuv[p, IM + 1] = lmuv[p, IM + 2] = lmm[0, JM - 1]
    ravm[p, 1:IM + 1] = 1.0 / IM
    ravm[p, IM + 1] = ravm[p, IM + 2] = 1.0
    lay = np.arange(LMO + 1)[None, :]
    zmask = lay > lmij[:, None]                      # entries set to zero beyond the column depth
    cosic = np.cos((np.arange(1, IM + 1) - 0.5) * 2 * np.pi / IM)
    sinic = np.sin((np.arange(1, IM + 1) - 0.5) * 2 * np.pi / IM)
    sg = lambda lm: np.where(lm > 0, 0.25, -0.25)   # noqa: E731
    anu = 1.0 - sg(lmu[ii, jj]) - sg(lmu[im1, jj])
    anv = 1.0 - sg(lmv[ii, jj]) - sg(lmv[ii, jm1])
    ramv = np.zeros((N, 5))
    ramv[:, 1] = ramv[:, 2] = 0.5
    ramv[:, 3] = dxys[jj] / (dxyn[jm1] + dxys[jj])
    ramv[:, 4] = dxyn[jj] / (dxyn[jj] + dxys[np.minimum(jj + 1, JM - 1)])
    Ko['oc_'] = dict(ii=ii, jj=jj, N=N, pole=pole, im1=im1, jm1=jm1, n=n, lmij=lmij, kmuv=kmuv, lmuv=lmuv, ravm=ravm, zmask=zmask, cosic=cosic, sinic=sinic,
                     anu=anu, anv=anv, u_ok=lmu[ii, jj] > 0, um_ok=lmu[im1, jj] > 0, v_ok=lmv[ii, jj] > 0, vm_ok=lmv[ii, jm1] > 0,
                     dxypo=_DXYPO[jj], ramv=ramv, lmu_ij=lmu[ii, jj], hocean=np.asarray(ctx['hocean'])[ii, jj], npn=n)
    act3 = (np.arange(1, LMO + 1)[None, None, :] <= lmm[:, :, None])
    rows = np.zeros((IM, JM, 1), bool)
    rows[:, 1:JM - 1] = True
    Ko['oc_']['act3rows'] = act3 & rows


def _seqsum(x):
    """Left-to-right sum of a 1-D array (a Fortran DO loop)."""
    s, _ = lax.scan(lambda c, v: (c + v, None), x[0], x[1:])
    return s


def oconv_columns(K, s, fx):
    """ocean_step.oconv_columns on device arrays.  Returns the argument dict of hbl_loop."""
    Ko = K['oc']
    C = Ko['oc_']
    T = tables()
    ii, jj, N = C['ii'], C['jj'], C['N']
    im1, jm1, n = C['im1'], C['jm1'], C['n']
    fsr2 = T['tabs']['fsr'][2]
    ut, vt, utd, vtd = s['uo'], s['vo'], s['uod'], s['vod']
    KMp = KM + 1
    ul0 = jnp.zeros((N, LMO + 1, KMp))
    ul = jnp.zeros_like(ul0)
    uld0 = jnp.zeros_like(ul0)
    uld = jnp.zeros_like(ul0)
    srcs0 = [ut[im1[n], jj[n], :], ut[ii[n], jj[n], :], vt[ii[n], jm1[n], :], vt[ii[n], jj[n], :]]
    srcs1 = [s['uo1'][im1[n], jj[n]], s['uo1'][ii[n], jj[n]], s['vo1'][ii[n], jm1[n]], s['vo1'][ii[n], jj[n]]]
    dsrc0 = [vtd[im1[n], jj[n], :], vtd[ii[n], jj[n], :], utd[ii[n], jm1[n], :], utd[ii[n], jj[n], :]]
    dsrc1 = [s['vod1'][im1[n], jj[n]], s['vod1'][ii[n], jj[n]], s['uod1'][ii[n], jm1[n]], s['uod1'][ii[n], jj[n]]]
    for k in range(4):
        ul0 = ul0.at[n, 1:, k + 1].set(srcs0[k])
        ul = ul.at[n, 1:, k + 1].set(srcs0[k])
        ul = ul.at[n, 1, k + 1].set(srcs1[k])
        uld0 = uld0.at[n, 1:, k + 1].set(dsrc0[k])
        uld = uld.at[n, 1:, k + 1].set(dsrc0[k])
        uld = uld.at[n, 1, k + 1].set(dsrc1[k])
    p = N - 1
    ul0 = ul0.at[p, 1:, 1:IM + 1].set(vt[:, JM - 2, :].T)
    ul = ul.at[p, 1:, 1:IM + 1].set(vt[:, JM - 2, :].T)
    ul = ul.at[p, 1, 1:IM + 1].set(s['vo1'][:, JM - 2])
    ul0 = ul0.at[p, 1:, IM + 1].set(ut[IM - 1, JM - 1, :])
    ul = ul.at[p, 1:, IM + 1].set(ut[IM - 1, JM - 1, :])
    ul = ul.at[p, 1, IM + 1].set(s['uo1'][IM - 1, JM - 1])
    ul0 = ul0.at[p, 1:, IM + 2].set(ut[IVNP0, JM - 1, :])
    ul = ul.at[p, 1:, IM + 2].set(ut[IVNP0, JM - 1, :])
    ul = ul.at[p, 1, IM + 2].set(s['uo1'][IVNP0, JM - 1])
    zero1 = jnp.zeros((N, 1))
    mo = jnp.concatenate([zero1, s['mo'][ii, jj, :]], axis=1)
    g0ml0 = jnp.concatenate([zero1, s['g0m'][ii, jj, :]], axis=1)
    s0ml0 = jnp.concatenate([zero1, s['s0m'][ii, jj, :]], axis=1)
    g0ml = g0ml0.at[:, 1:LSRPD + 1].set(s['g0m1'][ii, jj, :])
    s0ml = s0ml0.at[:, 1].set(s['s0m1'][ii, jj])
    zm = jnp.asarray(C['zmask'])
    mo, g0ml0, s0ml0, g0ml, s0ml = (jnp.where(zm, 0.0, a) for a in (mo, g0ml0, s0ml0, g0ml, s0ml))
    mo1 = s['mo1'][ii, jj]
    dtbydz = jnp.where(mo > 0, DTS / jnp.where(mo > 0, mo, 1), 0.0)
    bydz2 = jnp.zeros((N, LMO + 1)).at[:, 1:LMO].set(jnp.where((mo[:, 2:] > 0), 2.0 / (mo[:, 2:] + mo[:, 1:LMO]), 0.0))
    bydts = 1.0 / DTS
    rsi = fx['orsi'][ii, jj]
    odmui, odmvi = fx['odmui'], fx['odmvi']
    a_u = (jnp.where(C['u_ok'], odmui[ii, jj], 0.0) + jnp.where(C['um_ok'], odmui[im1, jj], 0.0))
    a_v = (jnp.where(C['v_ok'], odmvi[ii, jj], 0.0) + jnp.where(C['vm_ok'], odmvi[ii, jm1], 0.0))
    uistr = jnp.where(rsi > 0, a_u * C['anu'], 0.0)
    vistr = jnp.where(rsi > 0, a_v * C['anv'], 0.0)
    pu = _seqsum(odmvi[:, JM - 2] * C['cosic'])
    pv = _seqsum(-(odmvi[:, JM - 2] * C['sinic']))
    uistr = uistr.at[N - 1].set(pu / IM)
    vistr = vistr.at[N - 1].set(pv / IM)
    u2rho = jnp.sqrt((fx['odmua'][ii, jj] + uistr) ** 2 + (fx['odmva'][ii, jj] + vistr) ** 2) * bydts
    deltam = (s['mo'][ii, jj, 0] - mo1) * bydts
    bydx = 1.0 / C['dxypo']
    deltae = ((g0ml0[:, 1] - g0ml[:, 1]) * bydx) * bydts
    deltas = ((s0ml0[:, 1] - s0ml[:, 1]) * bydx) * bydts
    deltasr = (fx['osolarw'][ii, jj] * (1.0 - rsi) + fx['osolari'][ii, jj] * rsi) * bydts
    deltae = deltae - (1.0 - fsr2) * deltasr
    po = jnp.zeros((N, LMO + 1)).at[:, 1].set(0.5 * mo[:, 1] * GRAV)
    for l in range(1, LMO):
        po = po.at[:, l + 1].set(po[:, l] + 0.5 * GRAV * (mo[:, l] + mo[:, l + 1]))
    return dict(lmij=C['lmij'], kmuv=C['kmuv'], pole=C['pole'], dts=DTS, dxypo=C['dxypo'], mo=mo, mo1=mo1, deltae=deltae, deltas=deltas,
                deltam=deltam, deltasr=deltasr, u2rho=u2rho, ogeoz=s['ogeoz'][ii, jj], hocean=C['hocean'],
                s0m1=s['s0m1'][ii, jj], ravm=C['ravm'], lmuv=C['lmuv'], dtbydz=dtbydz, bydz2=bydz2, ul0=ul0, ulm=ul, uld0=uld0,
                uld=uld, g0ml0=g0ml0, s0ml0=s0ml0, g0ml=g0ml, s0ml=s0ml, po=po)


def stage_oconv(K, Kb, s, fx):
    """ocean_step.stage_oconv on device arrays."""
    Ko = K['oc']
    C = Ko['oc_']
    s = dict(s)
    T = tables()
    ii, jj, N, n, im1, jm1 = C['ii'], C['jj'], C['N'], C['n'], C['im1'], C['jm1']
    ut, vt, utd, vtd = s['uo'], s['vo'], s['uod'], s['vod']
    act3 = jnp.asarray(C['act3rows'])
    big = act3 & (jnp.abs(s['szmo']) > s['s0m'])
    s['szmo'] = jnp.where(big, jnp.copysign(s['s0m'], s['szmo']), s['szmo'])
    a = oconv_columns(K, s, fx)
    out = HB.hbl_loop(jnp.asarray(T['ze']), GRAV, a['lmij'], a['kmuv'], a['pole'], DTS, a['dxypo'], a['mo'], a['mo1'], a['deltae'],
                      a['deltas'], a['deltam'], a['deltasr'], a['u2rho'], a['ogeoz'], a['hocean'], a['s0m1'],
                      a['ravm'], a['lmuv'], a['dtbydz'], a['bydz2'], a['ul0'], a['ulm'], a['uld0'], a['uld'],
                      a['g0ml0'], a['s0ml0'], a['g0ml'], a['s0ml'], {}, T['tabs'], po=a['po'], vgsp=jnp.asarray(T['vg']),
                      agsp=jnp.asarray(T['ag']), bgsp=jnp.asarray(T['bg']), cgs=jnp.asarray(T['cg']))
    lmij = C['lmij']
    lay1 = np.arange(1, LMO + 1)[None, :]
    wm = lay1 <= lmij[:, None]
    s['g0m'] = s['g0m'].at[ii, jj, :].set(jnp.where(wm, out['g0ml'][:, 1:], s['g0m'][ii, jj, :]))
    s['s0m'] = s['s0m'].at[ii, jj, :].set(jnp.where(wm, out['s0ml'][:, 1:], s['s0m'][ii, jj, :]))
    s['kpl'] = s['kpl'].at[ii, jj].set(out['kbl'].astype(s['kpl'].dtype))
    ramv = C['ramv']
    Tk = [jnp.zeros((IM, JM, LMO)) for _ in range(4)]
    Td = [jnp.zeros((IM, JM, LMO)) for _ in range(4)]
    src = [(ut, im1, jj), (ut, ii, jj), (vt, ii, jm1), (vt, ii, jj)]
    srcd = [(vtd, im1, jj), (vtd, ii, jj), (utd, ii, jm1), (utd, ii, jj)]
    for k in range(4):
        m = lay1 <= C['lmuv'][:, k + 1][:, None]
        uk = ramv[:, k + 1][:, None] * (out['ul'][:, 1:, k + 1] - src[k][0][src[k][1], src[k][2], :])
        ud = ramv[:, k + 1][:, None] * (out['uld'][:, 1:, k + 1] - srcd[k][0][srcd[k][1], srcd[k][2], :])
        Tk[k] = Tk[k].at[ii[n], jj[n], :].set(jnp.where(m, uk, 0.0)[n])
        Td[k] = Td[k].at[ii[n], jj[n], :].set(jnp.where(m, ud, 0.0)[n])

    def add_i(f, T1, T2):
        r = (f + T2) + jnp.roll(T1, -1, axis=0)
        return r.at[IM - 1].set((f[IM - 1] + T1[0]) + T2[IM - 1])

    def add_j(f, T3, T4):
        t3 = jnp.zeros_like(T3).at[:, :JM - 1].set(T3[:, 1:])
        return (f + T4) + t3
    s['uo'] = add_i(s['uo'], Tk[0], Tk[1])
    s['vod'] = add_i(s['vod'], Td[0], Td[1])
    s['vo'] = add_j(s['vo'], Tk[2], Tk[3])
    s['uod'] = add_j(s['uod'], Td[2], Td[3])
    akg = out['akvg']
    aks = out['akvs']
    ar = np.arange(N)
    res = []
    for ak in (akg, aks):
        ak = ak.at[:, 0].set(ak[:, 1])
        ak = ak.at[ar, lmij].set(ak[ar, lmij - 1])
        res.append(ak)
    akg, aks = res
    flg, fls = out['flg3d'], out['fls3d']
    dtb = a['dtbydz']
    mask = (np.arange(LMO + 1)[None, :] >= 1) & (np.arange(LMO + 1)[None, :] <= lmij[:, None])
    dtbydz2 = 6.0 * dtb ** 2 * (1.0 / DTS)
    zc = jnp.zeros((N, 1))
    gz = jnp.concatenate([zc, s['gzmo'][ii, jj, :]], axis=1)
    sz = jnp.concatenate([zc, s['szmo'][ii, jj, :]], axis=1)
    gzn = (gz[:, 1:] + 3.0 * (flg[:, 0:LMO] + flg[:, 1:LMO + 1])) / (1.0 + dtbydz2[:, 1:] * (akg[:, 0:LMO] + akg[:, 1:LMO + 1]))
    szn = (sz[:, 1:] + 3.0 * (fls[:, 0:LMO] + fls[:, 1:LMO + 1])) / (1.0 + dtbydz2[:, 1:] * (aks[:, 0:LMO] + aks[:, 1:LMO + 1]))
    gzn = jnp.where(mask[:, 1:], gzn, gz[:, 1:])
    szn = jnp.where(mask[:, 1:], szn, sz[:, 1:])
    s['gzmo'] = s['gzmo'].at[ii, jj, :].set(gzn)
    s['szmo'] = s['szmo'].at[ii, jj, :].set(szn)
    from ovdiffs_jax import ovdiffs_jax
    zero = jnp.zeros((len(n), LMO + 1))
    for fld, ak in (('gxmo', akg), ('gymo', akg), ('sxmo', aks), ('symo', aks)):
        u0 = jnp.concatenate([jnp.zeros((len(n), 1)), s[fld][ii[n], jj[n], :]], axis=1)
        u, _ = ovdiffs_jax(ak[n], zero, zero, dtb[n], a['bydz2'][n], DTS, lmij[n], u0)
        u = u[:, 1:LMO + 1]
        old = s[fld][ii[n], jj[n], :]
        s[fld] = s[fld].at[ii[n], jj[n], :].set(jnp.where(mask[n, 1:], u, old))
    tiny = np.finfo(np.float64).tiny
    txy = jnp.abs(s['sxmo']) + jnp.abs(s['symo'])
    lim = act3 & (txy > s['s0m'])
    f = s['s0m'] / (txy + tiny)
    s['sxmo'] = jnp.where(lim, s['sxmo'] * f, s['sxmo'])
    s['symo'] = jnp.where(lim, s['symo'] * f, s['symo'])
    big = act3 & (jnp.abs(s['szmo']) > s['s0m'])
    s['szmo'] = jnp.where(big, jnp.copysign(s['s0m'], s['szmo']), s['szmo'])
    return s


# ====================================================================================================== stages 6-11: dynamics (ODHORZ0, 5 x ODHORZ, OFLUXV, OADVT2), post
Z12 = 2 * Z12EH


def _static_dynamics(Ko, Kb, ctx):
    from ocean_odhorz import make_layer_step
    lmm, lmu, lmv = Ko['lmm'], Ko['lmu'], Ko['lmv']
    Ko['to1'] = lambda a: np.pad(a, [(1, 0)] * a.ndim)
    to1n = Ko['to1']
    Ko['lmm1'], Ko['lmu1'], Ko['lmv1'] = (to1n(a).astype(np.int64) for a in (lmm, lmu, lmv))
    Ko['hoc1'] = to1n(np.asarray(ctx['hocean']))
    Ko['mk'] = O.m_active(lmm)
    ku = lmu[:, :, None] >= np.arange(1, LMO + 1)[None, None, :]
    kv = lmv[:, :, None] >= np.arange(1, LMO + 1)[None, None, :]
    ku[:, JM - 1] = False
    kv[:, JM - 1] = False
    Ko['ku'], Ko['kv'] = ku, kv
    lidx = np.arange(1, LMO + 1)[None, None, :]
    wr = (lidx + 1 <= lmm[:, :, None])
    wr[:, JM - 1, :] = (lidx[0] + 1 <= lmm[0, JM - 1])
    Ko['wr'] = wr
    Ko['lmm_f'], Ko['lmu_f'], Ko['lmv_f'] = (np.asarray(a, dtype=np.float64) for a in (lmm, lmu, lmv))
    Ko['step'] = make_layer_step(Ko['lmm1'], Ko['lmu1'], Ko['lmv1'])
    Ko['dxpo'] = O.geo1()[2][1:]
    T = O._opfil_ops(ctx)                                     # (LMO+1, JM+1, IM, IM) numpy, host (cached in ocean_step._OC)
    Kb['opfT'] = np.ascontiguousarray(T[:, 2:JM, :, :])       # rows J = 2..JM-1 (1-based): the only ones apply_operators uses
    for l in range(1, LMO + 1):
        assert (Ko['lmu'] >= l)[:, 1:JM - 1].any(), 'a layer without U points: the traced-layer prefilter loop assumes anyu = True'
    I = np.arange(IM + 1)[:, None]
    J = np.arange(JM + 1)[None, :]
    lmm1 = Ko['lmm1']
    Ko['ma1_1'] = np.where(J != JM, lmm1 >= 1, (J == JM) & (I == 1) & (lmm1[1, JM] >= 1))
    # shapes of the X pre-pass of OADVT2
    Ko['x_lmu1'], Ko['x_lmm1'] = Ko['lmu1'], Ko['lmm1']
    from oadvt_jax import _LS, _JS
    Ko['x_ls'], Ko['x_js'] = _LS, _JS
    Ko['x_lmu_l'] = (Ko['lmu1'][:, _JS] >= _LS[None, :]).T
    Ko['x_lmm_l'] = (Ko['lmm1'][:, _JS] >= _LS[None, :]).T


def _to1(a):
    pad = [(1, 0)] * a.ndim
    return jnp.pad(a, pad)


def _from1(a):
    return a[(slice(1, None), slice(1, None)) + ((slice(1, None),) if a.ndim == 3 else ())]


def _eos_vup_vdn(Ko, s, vg):
    mk = jnp.asarray(Ko['mk'])
    mo = s['mo']
    opb = jnp.where(mk[:, :, 0], s['opress'], 0.0)
    ps = []
    for l in range(LMO):
        ps.append(jnp.where(mk[:, :, l], opb + mo[:, :, l] * GRAV * 0.5, 0.0))
        opb = jnp.where(mk[:, :, l], opb + mo[:, :, l] * GRAV, opb)
    p = jnp.stack(ps, axis=2)
    dx = jnp.asarray(_DXYPO)[None, :, None]
    mmi = jnp.where(mk, mo * dx, 1.0)
    gup = (s['g0m'] - 2 * Z12EH * s['gzmo']) / mmi
    gdn = (s['g0m'] + 2 * Z12EH * s['gzmo']) / mmi
    sup = (s['s0m'] - 2 * Z12EH * s['szmo']) / mmi
    sdn = (s['s0m'] + 2 * Z12EH * s['szmo']) / mmi
    smean = s['s0m'] / mmi
    sup = jnp.maximum(sup, 0.5 * smean)
    sdn = jnp.maximum(sdn, 0.5 * smean)
    pup = p - mo * GRAV * Z12EH
    pdn = p + mo * GRAV * Z12EH
    vup = volgsp(vg, gup, sup, pup)
    vdn = volgsp(vg, gdn, sdn, pdn)
    return jnp.where(mk, vup, 0.0), jnp.where(mk, vdn, 0.0), mk


def run_odhorz0(K, s):
    from odhorz0_jax import odhorz0_jax
    Ko = K['oc']
    T = tables()
    s = dict(s)
    vup, vdn, mk = _eos_vup_vdn(Ko, s, jnp.asarray(T['vg']))
    r = odhorz0_jax(Ko['lmm_f'], Ko['lmv_f'], s['opress'], s['g0m'], s['gzmo'], s['s0m'], s['szmo'], s['mo'], s['uo'], s['vo'], vup, vdn)
    s['mmi'] = jnp.where(mk, s['mo'] * jnp.asarray(_DXYPO)[None, :, None], s['mmi'])
    for k in ('opbot', 'vbar', 'dzgdp', 'dh3d', 'mo', 'uo', 'vo', 'gup', 'gdn', 'sup', 'sdn'):
        s[k] = r[k]
    return s


def stage_odhorz0(K, Kb, s, fx):
    return run_odhorz0(K, s)


def _apply_ops(Tl, x, l):
    """opfil2_jax.apply_operators(T, x, l, 2, JM-1) with the layer index l traced; Tl = Kb['opfT'] (LMO+1, JM-2, IM, IM)."""
    xr = x[:, 1:JM - 1].T
    yr = jnp.einsum('rij,rj->ri', Tl[l], xr)
    return x.at[:, 1:JM - 1].set(yr.T)


def odhorz_prefilter(Ko, Kb, moh, uoh, voh, opboth, vbar, dzgdp, hocean):
    """ocean_step.odhorz_prefilter as a fori_loop over the layers (13 .. 1)."""
    Tl = Kb['opfT']
    lmu = jnp.asarray(Ko['lmu'])
    ma_all = jnp.asarray(Ko['mk'])
    dxpo = jnp.asarray(Ko['dxpo'])
    ma1 = ma_all[:, :, 0]
    pdn = jnp.where(ma1, opboth, 0.0)
    ogeoz = jnp.where(ma1, -hocean * GRAV, 0.0)
    rows = jnp.asarray(np.arange(JM) >= 1) & jnp.asarray(np.arange(JM) <= JM - 2)
    us0 = jnp.zeros((IM, JM))
    pg0 = jnp.zeros((IM, JM))
    us_out = jnp.zeros((IM, JM, LMO))
    pg_out = jnp.zeros((IM, JM, LMO))

    def body(k, c):
        pdn, ogeoz, us, pg, us_out, pg_out = c
        l = LMO - k
        ma = lax.dynamic_index_in_dim(ma_all, l - 1, axis=2, keepdims=False)
        m = lax.dynamic_index_in_dim(moh, l - 1, axis=2, keepdims=False)
        vb = lax.dynamic_index_in_dim(vbar, l - 1, axis=2, keepdims=False)
        dz = lax.dynamic_index_in_dim(dzgdp, l - 1, axis=2, keepdims=False)
        uo_l = lax.dynamic_index_in_dim(uoh, l - 1, axis=2, keepdims=False)
        dp = m * GRAV
        dh = jnp.where(ma, m * vb, 0.0)
        p = jnp.where(ma, pdn - 0.5 * dp, 0.0)
        zg = jnp.where(ma, ogeoz + dp * 0.5 * dz, 0.0)
        pdn = jnp.where(ma, pdn - dp, pdn)
        ogeoz = jnp.where(ma, ogeoz + dh * GRAV, ogeoz)
        uact = (lmu >= l) & rows[None, :]
        us = jnp.where(uact, uo_l, us)
        us = _apply_ops(Tl, us, l)
        us = us.at[:, JM - 1].set(uo_l[:, JM - 1])
        mmid = m + jnp.roll(m, -1, axis=0)
        new = (((zg - jnp.roll(zg, -1, axis=0)) + (p - jnp.roll(p, -1, axis=0)) * (dh + jnp.roll(dh, -1, axis=0)) / mmid) * (1.0 / dxpo)[None, :])
        pg = jnp.where(uact, new, pg)
        pg = _apply_ops(Tl, pg, l)
        us_out = lax.dynamic_update_index_in_dim(us_out, us, l - 1, axis=2)
        pg_out = lax.dynamic_update_index_in_dim(pg_out, pg, l - 1, axis=2)
        return pdn, ogeoz, us, pg, us_out, pg_out

    pdn, ogeoz, us, pg, us_out, pg_out = lax.fori_loop(0, LMO, body, (pdn, ogeoz, us0, pg0, us_out, pg_out))
    return us_out, pg_out, ogeoz


def odhorz_dev(Ko, dt, H, IO, vbar1, dzgdp1, us1, pg1):
    """ocean_odhorz.odhorz_jax on device arrays with the layer loop as a fori_loop (all arrays 1-based (IM+1, JM+1, LMO+1))."""
    step = Ko['step']
    ma1 = jnp.asarray(Ko['ma1_1'])
    pdn = jnp.where(ma1, H['opbot1'], 0.0)
    ogeoz = jnp.where(ma1, -jnp.asarray(Ko['hoc1']) * GRAV, 0.0)
    opbot = IO['opbot1_in']
    z = jnp.zeros_like(IO['mo1'])

    def body(k, c):
        pdn, ogeoz, opbot, mo, uo, vo, uod, vod, mu3, mv3 = c
        l = LMO - k
        at = lambda a: lax.dynamic_index_in_dim(a, l, axis=2, keepdims=False)    # noqa: E731
        pdn, ogeoz, opbot, mo_l, uo_l, vo_l, uod_l, vod_l, mu_l, mv_l = step(
            l, dt, pdn, ogeoz, opbot, at(H['mo1']), at(H['uo1']), at(H['vo1']), at(H['uod1']), at(H['vod1']), at(vbar1), at(dzgdp1), at(us1), at(pg1),
            at(IO['mo1']), at(IO['uo1']), at(IO['vo1']), at(IO['uod1']), at(IO['vod1']))
        put = lambda a, v: lax.dynamic_update_index_in_dim(a, v, l, axis=2)      # noqa: E731
        return (pdn, ogeoz, opbot, put(mo, mo_l), put(uo, uo_l), put(vo, vo_l), put(uod, uod_l), put(vod, vod_l), put(mu3, mu_l), put(mv3, mv_l))

    c0 = (pdn, ogeoz, opbot, IO['mo1'], IO['uo1'], IO['vo1'], IO['uod1'], IO['vod1'], z, z)
    pdn, ogeoz, opbot, mo, uo, vo, uod, vod, mu3, mv3 = lax.fori_loop(0, LMO, body, c0)
    return mo, uo, vo, uod, vod, opbot, mu3, mv3


# ---- OADVT2 with the X pre-pass as a host callback ------------------------------------------------
XPRE_STATS = {'calls': 0, 'seconds': 0.0, 'bytes_in': 0, 'bytes_out': 0}


def _xpre_np(mm, mu, dt, lmu1, lmm1):
    import time as _t
    from oadvt_vec import _x_prepass
    t0 = _t.perf_counter()
    snap, ncour, lane = _x_prepass(np.asarray(mm), np.asarray(mu), dt, lmu1, lmm1)
    ncour = ncour.astype(np.int64)
    XPRE_STATS['calls'] += 1
    XPRE_STATS['seconds'] += _t.perf_counter() - t0
    XPRE_STATS['bytes_in'] += np.asarray(mm).nbytes + np.asarray(mu).nbytes
    XPRE_STATS['bytes_out'] += snap.nbytes + ncour.nbytes + lane.nbytes
    return snap, ncour, lane


def oadvtx2_dev(Ko, rm, rx, ry, rz, mm, mu, dt, qlimit):
    from oadvt_jax import _x_sweep
    ls, js = Ko['x_ls'], Ko['x_js']
    shp = (jax.ShapeDtypeStruct((LMO + 1, JM + 1, IM + 1), jnp.float64), jax.ShapeDtypeStruct((LMO + 1, JM + 1), jnp.int64),
           jax.ShapeDtypeStruct((LMO + 1, JM + 1), jnp.bool_))
    lmu1, lmm1 = Ko['x_lmu1'], Ko['x_lmm1']
    snap, ncour, lane = jax.pure_callback(lambda a, b: _xpre_np(a, b, dt, lmu1, lmm1), shp, mm, mu)
    args = (rm, rx, ry, rz, mm, snap[ls, js], ncour[ls, js], lane[ls, js], jnp.asarray(Ko['x_lmu_l']), jnp.asarray(Ko['x_lmm_l']), 1.0 if qlimit else 0.0)
    if os.environ.get('ROCKE_XSWEEP_BARRIER') == '1':      # D214 experiment: stop XLA fusing/reordering across the sweep (GPU NaN, job 58812322); default off
        out = lax.optimization_barrier(_x_sweep(*lax.optimization_barrier(args[:-1]), args[-1]))
        return out
    return _x_sweep(*args)


def oadvt2_dev(Ko, mmi, rm, rx, ry, rz, dt, qlimit, smu, smv, smw):
    from oadvt_jax import oadvty2_jax, oadvtz2_jax
    lmu1, lmv1, lmm1 = Ko['lmu1'], Ko['lmv1'], Ko['lmm1']
    rm, rx, ry, rz, ma = oadvtx2_dev(Ko, rm, rx, ry, rz, mmi, smu, 0.5 * dt, qlimit)
    rm, rx, ry, rz, ma = oadvty2_jax(rm, rx, ry, rz, ma, smv, dt, qlimit, lmm1, lmv1)
    rm, rx, ry, rz, ma = oadvtz2_jax(rm, rx, ry, rz, ma, smw, dt, qlimit, lmm1)
    rm, rx, ry, rz, ma = oadvtx2_dev(Ko, rm, rx, ry, rz, ma, smu, 0.5 * dt, qlimit)
    rm = rm.at[1:IM + 1, JM, 1:].set(rm[1, JM, 1:][None, :])
    rz = rz.at[1:IM + 1, JM, 1:].set(rz[1, JM, 1:][None, :])
    return ma, rm, rx, ry, rz


def stage_dynamics(K, Kb, s, fx):
    """ocean_step.stage_dynamics: ODHORZ0, five ODHORZ calls with the OPFIL2 prefilter, VONP, OFLUXV, OADVT2 of G0M and S0M."""
    from ocean_ofluxv import ofluxv_jax
    Ko = K['oc']
    s = run_odhorz0(K, s)
    s['smu'] = jnp.zeros_like(s['mo'])
    s['smv'] = jnp.zeros_like(s['mo'])
    mk, ku, kv = (jnp.asarray(Ko[k]) for k in ('mk', 'ku', 'kv'))
    hocean = jnp.asarray(Ko['ctx']['hocean'])

    def odd_state():
        d = {}
        d['mo'] = jnp.where(mk, s['mo'], 0.0)
        d['uo'] = jnp.where(ku, s['uo'], 0.0)
        d['vod'] = jnp.where(ku, s['vod'], 0.0)
        d['vo'] = jnp.where(kv, s['vo'], 0.0)
        d['uod'] = jnp.where(kv, s['uod'], 0.0)
        for k in ('mo', 'uo', 'vo'):
            d[k] = d[k].at[:, JM - 1, :].set(s[k][:, JM - 1, :])
        d['opbot'] = jnp.where(mk[:, :, 0], s['opbot'], 0.0)
        return d
    st1 = odd_state()
    st2 = dict(st1)
    main = {k: s[k] for k in ('mo', 'uo', 'vo', 'uod', 'vod', 'opbot')}
    vbar1, dzgdp1 = _to1(s['vbar']), _to1(s['dzgdp'])

    def call(H, IO, dt, qeven):
        us, pg, og = odhorz_prefilter(Ko, Kb, H['mo'], H['uo'], H['vo'], H['opbot'], s['vbar'], s['dzgdp'], hocean)
        Hn = dict(opbot1=_to1(H['opbot']), mo1=_to1(H['mo']), uo1=_to1(H['uo']), vo1=_to1(H['vo']), uod1=_to1(H['uod']), vod1=_to1(H['vod']))
        IOn = dict(opbot1_in=_to1(IO['opbot']), mo1=_to1(IO['mo']), uo1=_to1(IO['uo']), vo1=_to1(IO['vo']), uod1=_to1(IO['uod']), vod1=_to1(IO['vod']))
        mo, uo, vo, uod, vod, opbot, mu3, mv3 = odhorz_dev(Ko, dt, Hn, IOn, vbar1, dzgdp1, _to1(us), _to1(pg))
        s['ogeoz'] = jnp.where(mk[:, :, 0], og, s['ogeoz'])
        out = dict(mo=_from1(mo), uo=_from1(uo), vo=_from1(vo), uod=_from1(uod), vod=_from1(vod), opbot=_from1(opbot))
        if qeven:
            s['smu'] = s['smu'] + _from1(mu3) * 1.0
            s['smv'] = s['smv'] + _from1(mv3) * 1.0
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
    s['uo'] = s['uo'].at[IVNP0, JM - 1, :].set(s['vonp'])
    mo, uo, vo, smw = ofluxv_jax(Ko['lmm_f'], Ko['lmu_f'], Ko['lmv_f'], DTOLF, s['opbot'], s['opress'], s['mo'], s['uo'], s['vo'])
    s['mo'], s['uo'], s['vo'] = mo, uo, vo
    s['smw'] = jnp.where(jnp.asarray(Ko['wr']), smw * (jnp.asarray(_DXYPO)[None, :, None] / DTOLF), 0.0)
    for r, qk in ((('g0m', 'gxmo', 'gymo', 'gzmo'), False), (('s0m', 'sxmo', 'symo', 'szmo'), True)):
        ma, rm, rx, ry, rz = oadvt2_dev(Ko, _to1(s['mmi']), *[_to1(s[k]) for k in r], DTOLF, qk, _to1(s['smu']), _to1(s['smv']), _to1(s['smw']))
        for k, v in zip(r, (rm, rx, ry, rz)):
            s[k] = _from1(v)
    return s


def stage_post(K, Kb, s, fx):
    return run_odhorz0(K, s)


# ====================================================================================================== stage 7: straits
def _static_straits(Ko, ctx):
    from straits_jax import partners
    ist, jst = np.asarray(ctx['ist']) - 1, np.asarray(ctx['jst']) - 1
    nm = ctx['nmst']
    lmm = Ko['lmm']
    Ko['st'] = dict(ist=ist, jst=jst, nm=nm, links=partners(ist, jst), lmst=np.asarray(ctx['lmst']),
                    lmme=np.stack([lmm[ist[:, k], jst[:, k]] for k in (0, 1)], axis=1),
                    hoceane=np.stack([np.asarray(ctx['hocean'])[ist[:, k], jst[:, k]] for k in (0, 1)], axis=1),
                    distpg=np.asarray(ctx['distpg']), wist=np.asarray(ctx['wist']), dist=np.asarray(ctx['dist']),
                    xst=np.asarray(ctx['xst']), yst=np.asarray(ctx['yst']), jst1=np.asarray(ctx['jst']),
                    dxyp=np.concatenate([[0.0], _DXYPO])[np.asarray(ctx['jst'])], sinpo=O.geo1()[1][1:])
    pairs = [(int(ist[n, k]), int(jst[n, k])) for n in range(nm) for k in (0, 1)]
    Ko['st']['unique_cells'] = len(set(pairs)) == len(pairs)


def stadv_seq_dev(dts, lmst, mmst, must, dxyp1, dxyp2, xst, yst, links, me, mst):
    from straits_jax import stadv_jax
    me = dict(me)
    mst = dict(mst)
    N = lmst.shape[0]
    key_of = {'g0mst': 'g0', 'gxmst': 'gx', 'gzmst': 'gz', 's0mst': 's0', 'sxmst': 'sx', 'szmst': 'sz'}
    for n in range(N):
        sl = slice(n, n + 1)
        out = stadv_jax(dts, lmst[sl], mmst[sl], must[sl], dxyp1[sl], dxyp2[sl], xst[sl], yst[sl], me['moe'][sl], me['g0me'][sl], me['gxme'][sl],
                        me['gyme'][sl], me['gzme'][sl], me['s0me'][sl], me['sxme'][sl], me['syme'][sl], me['szme'][sl], mst['g0'][sl], mst['gx'][sl],
                        mst['gz'][sl], mst['s0'][sl], mst['sx'][sl], mst['sz'][sl])
        for k, v in out.items():
            if k in me:
                me[k] = me[k].at[sl].set(v)
            else:
                kk = key_of[k]
                mst[kk] = mst[kk].at[sl].set(v)
        m_lim = int(lmst[n])
        for k in range(2):
            for (m, j) in links.get((n, k), []):
                for name in ['moe', 'g0me', 'gxme', 'gyme', 'gzme', 's0me', 'sxme', 'syme', 'szme']:
                    me[name] = me[name].at[m, j, 1:m_lim + 1].set(me[name][n, k, 1:m_lim + 1])
    return me, mst


def stage_straits(K, Kb, s, fx):
    from straits_jax import stpgf_jax, stbdra_jax
    from stconv_jax import stconv_jax
    Ko = K['oc']
    S_ = Ko['st']
    T = tables()
    s = dict(s)
    ist, jst, nm = S_['ist'], S_['jst'], S_['nm']
    names = [('moe', 'mo'), ('g0me', 'g0m'), ('gxme', 'gxmo'), ('gyme', 'gymo'), ('gzme', 'gzmo'),
             ('s0me', 's0m'), ('sxme', 'sxmo'), ('syme', 'symo'), ('szme', 'szmo')]
    me = {}
    for a, b in names:
        arr = jnp.zeros((nm, 2, LMO + 1)).at[:, :, 1:].set(jnp.stack([s[b][ist[:, k], jst[:, k], :] for k in (0, 1)], axis=1))
        me[a] = arr
    pad = lambda a: jnp.concatenate([jnp.zeros((nm, 1)), a.T], axis=1)   # noqa: E731
    vg = jnp.asarray(T['vg'])
    ze = jnp.asarray(T['ze'])
    tabs = {k: (jnp.asarray(v) if not isinstance(v, int) else v) for k, v in T['tabs'].items()}
    dts = DTS
    must, mmst = pad(s['must']), pad(s['mmst'])
    g0, gx, gz = pad(s['g0mst']), pad(s['gxmst']), pad(s['gzmst'])
    s0, sx, sz = pad(s['s0mst']), pad(s['sxmst']), pad(s['szmst'])
    oprese = jnp.stack([s['opress'][ist[:, k], jst[:, k]] for k in (0, 1)], axis=1)
    # 1. STPGF
    must = stpgf_jax(vg, GRAV, dts, S_['lmst'], S_['lmme'], oprese, S_['hoceane'], me['moe'], me['g0me'], me['gzme'], me['s0me'], me['szme'],
                     S_['dxyp'], must, S_['distpg'], S_['wist'])
    # 2. STADV
    me, mst = stadv_seq_dev(dts, S_['lmst'], mmst, must, S_['dxyp'][:, 0], S_['dxyp'][:, 1], S_['xst'], S_['yst'], S_['links'], me,
                            {'g0': g0, 'gx': gx, 'gz': gz, 's0': s0, 'sx': sx, 'sz': sz})
    # 3. STCONV
    res = stconv_jax(ze, GRAV, dts, nm, S_['lmst'], mmst, S_['dist'], S_['wist'], S_['jst1'], S_['sinpo'], vg, must, mst['g0'], mst['gx'], mst['gz'],
                     mst['s0'], mst['sx'], mst['sz'], tabs)
    # 4. STBDRA
    must2, gx2, sx2 = stbdra_jax(dts, nm, S_['lmst'], res['must'], mmst, S_['wist'], S_['dist'], res['gxmst'], res['sxmst'])
    st = dict(must=must2, g0=res['g0mst'], gx=gx2, gz=res['gzmst'], s0=res['s0mst'], sx=sx2, sz=res['szmst'])
    for k, kk in (('must', 'must'), ('g0', 'g0mst'), ('gx', 'gxmst'), ('gz', 'gzmst'), ('s0', 's0mst'), ('sx', 'sxmst'), ('sz', 'szmst')):
        s[kk] = st[k][:, 1:].T
    for a, b in names:
        arr = me[a]
        if S_['unique_cells']:
            for k in (0, 1):
                s[b] = s[b].at[ist[:, k], jst[:, k], :].set(arr[:, k, 1:])
        else:
            for n in range(nm):
                for k in (0, 1):
                    s[b] = s[b].at[ist[n, k], jst[n, k], :].set(arr[n, k, 1:])
    return s


# ====================================================================================================== stage 9: ODIFF (ocean_odiff.odiff, every 6th step)
def _static_odiff(Ko, Kb, ctx):
    import ocean_odiff as D
    T = D.init_odiff(Ko['lmu'], Ko['lmv'])
    g = T['g']
    J1 = np.arange(2, JM)
    cosic, sinic, cosu, sinu = D.trig_tables()
    od = dict(cosic=cosic, sinic=sinic, cosu=cosu, sinu=sinu, dtdiff=D.DTDIFF)
    for k in ('khp', 'khv', 'tanp', 'tanv', 'bydxp', 'bydyv', 'bydyp', 'bydxv', 'bydxypo', 'bydxyv'):
        od[k] = T[k]
    for k in ('dxvo', 'dypo', 'dxpo', 'dyvo'):
        od[k] = g[k]
    od['J1'] = J1
    Ko['od'] = od
    for k in ('uxa', 'uxb', 'uxc', 'uya', 'uyb', 'uyc', 'vxa', 'vxb', 'vxc', 'vya', 'vyb', 'vyc'):
        Kb['od_' + k] = T[k]


def _tri_fwd_bwd(a, b, c, r, recip):
    """TRIDIAG (division form, recip=False: OCNDYN x sweep) or TRIDIAG_3D_DIST_new (reciprocal form) along axis 0, as two lax.scans."""
    def fwd(carry, x):
        bet, uprev = carry
        aj, bj, cjm1, rj = x
        gam = cjm1 * bet if recip else cjm1 / bet            # `bet` is bybet in the reciprocal form
        bet_n = bj - aj * gam
        if recip:
            bybet = 1.0 / bet_n
            uj = (rj - aj * uprev) * bybet
            return (bybet, uj), (gam, uj)
        uj = (rj - aj * uprev) / bet_n
        return (bet_n, uj), (gam, uj)
    if recip:
        bet0 = 1.0 / b[0]
        u0 = r[0] * bet0
    else:
        bet0 = b[0]
        u0 = r[0] / bet0
    _, (gams, us) = lax.scan(fwd, (bet0, u0), (a[1:], b[1:], c[:-1], r[1:]))
    uf = jnp.concatenate([u0[None], us], axis=0)
    gam_full = jnp.concatenate([jnp.zeros_like(u0)[None], gams], axis=0)

    def bwd(unext, x):
        uj, gnext = x
        uj = uj - gnext * unext
        return uj, uj
    _, ub = lax.scan(bwd, uf[-1], (uf[:-1], gam_full[1:]), reverse=True)
    return jnp.concatenate([ub, uf[-1][None]], axis=0)


def _polevel_all(Od, uo, vo, lmv):
    j = JM - 2
    mv = jnp.asarray(lmv[:, j][:, None] >= np.arange(1, LMO + 1)[None, :])
    cosic, sinic, cosu, sinu = (jnp.asarray(Od[k]) for k in ('cosic', 'sinic', 'cosu', 'sinu'))

    def body(c, x):
        unp, vnp = c
        mvi, si, ci, voi = x
        return (unp - jnp.where(mvi, si * voi, 0.0), vnp + jnp.where(mvi, ci * voi, 0.0)), None
    (unp, vnp), _ = lax.scan(body, (jnp.zeros(LMO), jnp.zeros(LMO)), (mv, sinic, cosic, vo[:, j, :]))
    unp = unp * 2 / IM
    vnp = vnp * 2 / IM
    uo = uo.at[:, JM - 1, :].set(unp[None, :] * cosu[:, None] + vnp[None, :] * sinu[:, None])
    vo = vo.at[:, JM - 1, :].set(vnp[None, :] * cosic[:, None] - unp[None, :] * sinic[:, None])
    return uo, vo


def _fluxes_dev(Od, uo, vo, mu, mv):
    """ocean_odiff._fluxes with the J loop vectorised (arrays over j = 0 .. JM-2)."""
    khp, khv, tanp, tanv = Od['khp'], Od['khv'], Od['tanp'], Od['tanv']
    bydxp, bydyv, bydyp, bydxv = Od['bydxp'], Od['bydyv'], Od['bydyp'], Od['bydxv']
    Jr = np.arange(1, JM)
    c3 = lambda v: jnp.asarray(v[Jr])[None, :, None]            # coefficient at J      # noqa: E731
    c3p = lambda v: jnp.asarray(v[Jr + 1])[None, :, None]       # coefficient at J + 1  # noqa: E731
    c3m = lambda v: jnp.asarray(v[Jr - 1])[None, :, None]       # coefficient at J - 1  # noqa: E731
    u_jp1, u_j = uo[:, 1:JM], uo[:, 0:JM - 1]
    m1, m0 = mu[:, 1:JM], mu[:, 0:JM - 1]
    mim1 = jnp.roll(m1, 1, axis=0)
    ut = jnp.where(m1, u_jp1 * c3p(tanp), 0.0)
    ux = jnp.where(m1, u_jp1, 0.0)
    uy = jnp.where(m1, u_jp1, 0.0)
    ut = ut + jnp.where(m0, u_j * c3(tanp), 0.0)
    uy = uy - jnp.where(m0, u_j, 0.0)
    ux = ux - jnp.where(mim1, jnp.roll(u_jp1, 1, axis=0), 0.0)
    ut = 0.5 * ut
    ux = ux * c3p(bydxp)
    uy = uy * c3(bydyv)
    v_j = vo[:, 0:JM - 1]
    mj = mv[:, 0:JM - 1]
    vt = jnp.where(mj, v_j * c3(tanv), 0.0)
    vx = jnp.where(mj, v_j, 0.0)
    vy = jnp.where(mj, v_j, 0.0)
    v_m = jnp.concatenate([jnp.zeros_like(vo[:, 0:1]), vo[:, 0:JM - 2]], axis=1)
    mjm = jnp.concatenate([jnp.zeros_like(mv[:, 0:1]), mv[:, 0:JM - 2]], axis=1)
    vt = vt + jnp.where(mjm, v_m * c3m(tanv), 0.0)
    vy = vy - jnp.where(mjm, v_m, 0.0)
    mim1v = jnp.roll(mj, 1, axis=0)
    vx = vx - jnp.where(mim1v, jnp.roll(v_j, 1, axis=0), 0.0)
    vt = 0.5 * vt
    vy = vy * c3(bydyp)
    vx = vx * c3(bydxv)
    fuy = jnp.roll(c3(khv) * vx, -1, axis=0)
    fvx = c3(khv) * (uy + ut)
    fux = jnp.roll(c3(khp) * (vy + vt), -1, axis=0)
    fvy_core = c3p(khp)[:, :JM - 2] * ux[:, :JM - 2]
    fvy = jnp.concatenate([fvy_core, jnp.zeros_like(ux[:, :1])], axis=1)
    pad = lambda a: jnp.concatenate([a, jnp.zeros_like(a[:, :1])], axis=1)      # noqa: E731
    return pad(fux), pad(fuy), pad(fvx), pad(fvy)


def odiff_dev(Ko, Kb, mo, uo_in, vo_in, dh):
    Od = Ko['od']
    lmu, lmv = Ko['lmu'], Ko['lmv']
    dt2 = Od['dtdiff'] * 5e-1
    uo, vo = uo_in, vo_in
    uonp = uo[IM - 1, JM - 1, :]
    vonp = uo[IVNP0, JM - 1, :]
    cosic, sinic, cosu, sinu = (jnp.asarray(Od[k]) for k in ('cosic', 'sinic', 'cosu', 'sinu'))
    L = np.arange(1, LMO + 1)[None, None, :]
    mu = jnp.asarray(lmu[:, :, None] >= L)
    mv = jnp.asarray(lmv[:, :, None] >= L)
    uo = uo.at[:, JM - 1, :].set(uonp[None, :] * cosu[:, None] + vonp[None, :] * sinu[:, None])
    vo = vo.at[:, JM - 1, :].set(vonp[None, :] * cosic[:, None] - uonp[None, :] * sinic[:, None])
    uo, vo = _polevel_all(Od, uo, vo, lmv)
    mo_ip1 = jnp.roll(mo, -1, axis=0)
    mo_jp1 = jnp.zeros_like(mo).at[:, :JM - 1].set(mo[:, 1:])
    dh_ip1 = jnp.roll(dh, -1, axis=0)
    dh_jp1 = jnp.zeros_like(dh).at[:, :JM - 1].set(dh[:, 1:])
    bymu = jnp.where(mu, 1.0 / (mo + mo_ip1), 0.0)
    bymv = jnp.where(mv, 1.0 / (mo + mo_jp1), 0.0)
    dtu = dt2 * (dh + dh_ip1) * bymu
    dtv = dt2 * (dh + dh_jp1) * bymv
    ops = {k: Kb['od_' + k] for k in ('uxa', 'uxb', 'uxc', 'uya', 'uyb', 'uyc', 'vxa', 'vxb', 'vxc', 'vya', 'vyb', 'vyc')}
    dxvo, dypo, dxpo, dyvo = Od['dxvo'], Od['dypo'], Od['dxpo'], Od['dyvo']
    bydxypo, bydxyv, tanp, tanv = Od['bydxypo'], Od['bydxyv'], Od['tanp'], Od['tanv']
    jr = slice(1, JM - 1)
    J1 = Od['J1']
    col = lambda v: jnp.asarray(v[J1])[None, :, None]            # noqa: E731
    col_m1 = lambda v: jnp.asarray(v[J1 - 1])[None, :, None]     # noqa: E731
    col_p1 = lambda v: jnp.asarray(v[J1 + 1])[None, :, None]     # noqa: E731

    def cross_u(fux, fuy):
        fux_im1 = jnp.roll(fux, 1, axis=0)
        return (col(dypo) * (fux_im1[:, jr] - fux[:, jr]) + col(dxvo) * fuy[:, jr] - col_m1(dxvo) * fuy[:, 0:JM - 2]) * col(bydxypo) \
            - 0.5 * (col_m1(tanv) * fuy[:, 0:JM - 2] + col(tanv) * fuy[:, jr])

    def cross_v(fvx, fvy):
        fvx_im1 = jnp.roll(fvx, 1, axis=0)
        return (col(dyvo) * (fvx[:, jr] - fvx_im1[:, jr]) + col(dxpo) * fvy[:, 0:JM - 2] - col_p1(dxpo) * fvy[:, jr]) * col(bydxyv) \
            + 0.5 * (col(tanp) * fvy[:, 0:JM - 2] + col_p1(tanp) * fvy[:, jr])

    # ---------------- x sweep
    fux, fuy, fvx, fvy = _fluxes_dev(Od, uo, vo, mu, mv)
    sl = lambda a: a[:, jr]                       # noqa: E731
    mus, mvs = sl(mu), sl(mv)
    DTU, DTV = sl(dtu), sl(dtv)
    uj, ujm, ujp = sl(uo), uo[:, 0:JM - 2], uo[:, 2:JM]
    vj, vjm, vjp = sl(vo), vo[:, 0:JM - 2], vo[:, 2:JM]
    au_ = -DTU * sl(ops['uxa'])
    bu_ = 1.0 - DTU * sl(ops['uxb'])
    cu_ = -DTU * sl(ops['uxc'])
    ru_ = uj + DTU * (sl(ops['uya']) * ujm + sl(ops['uyb']) * uj + sl(ops['uyc']) * ujp)
    ru_ = ru_ + DTU * cross_u(fux, fuy)
    av_ = -DTV * sl(ops['vxa'])
    bv_ = 1.0 - DTV * sl(ops['vxb'])
    cv_ = -DTV * sl(ops['vxc'])
    rv_ = vj + DTV * (sl(ops['vya']) * vjm + sl(ops['vyb']) * vj + sl(ops['vyc']) * vjp)
    rv_ = rv_ + DTV * cross_v(fvx, fvy)
    au = jnp.where(mus, au_, 0.0)
    bu = jnp.where(mus, bu_, 1.0)
    cu = jnp.where(mus, cu_, 0.0)
    ru = jnp.where(mus, ru_, 0.0)
    av = jnp.where(mvs, av_, 0.0)
    bv = jnp.where(mvs, bv_, 1.0)
    cv = jnp.where(mvs, cv_, 0.0)
    rv = jnp.where(mvs, rv_, 0.0)
    m0u, m0v = mus[0], mvs[0]
    au = au.at[0].set(jnp.where(m0u, 0.0, au[0]))
    ru = ru.at[0].set(jnp.where(m0u, ru[0] + dtu[0, jr] * sl(ops['uxa'])[0] * uo[IM - 1, jr], ru[0]))
    av = av.at[0].set(jnp.where(m0v, 0.0, av[0]))
    rv = rv.at[0].set(jnp.where(m0v, rv[0] + dtv[0, jr] * sl(ops['vxa'])[0] * vo[IM - 1, jr], rv[0]))
    mLu, mLv = mus[IM - 1], mvs[IM - 1]
    cu = cu.at[IM - 1].set(jnp.where(mLu, 0.0, cu[IM - 1]))
    ru = ru.at[IM - 1].set(jnp.where(mLu, ru[IM - 1] + dtu[IM - 1, jr] * sl(ops['uxc'])[IM - 1] * uo[0, jr], ru[IM - 1]))
    cv = cv.at[IM - 1].set(jnp.where(mLv, 0.0, cv[IM - 1]))
    rv = rv.at[IM - 1].set(jnp.where(mLv, rv[IM - 1] + dtv[IM - 1, jr] * sl(ops['vxc'])[IM - 1] * vo[0, jr], rv[IM - 1]))
    uo = uo.at[:, jr].set(_tri_fwd_bwd(au, bu, cu, ru, False))
    vo = vo.at[:, jr].set(_tri_fwd_bwd(av, bv, cv, rv, False))
    # ---------------- y sweep
    fux, fuy, fvx, fvy = _fluxes_dev(Od, uo, vo, mu, mv)
    uj = sl(uo)
    uxa, uxb, uxc = sl(ops['uxa']), sl(ops['uxb']), sl(ops['uxc'])
    vxa, vxb, vxc = sl(ops['vxa']), sl(ops['vxb']), sl(ops['vxc'])
    uim1, uip1 = jnp.roll(uo, 1, axis=0)[:, jr], jnp.roll(uo, -1, axis=0)[:, jr]
    vim1, vip1 = jnp.roll(vo, 1, axis=0)[:, jr], jnp.roll(vo, -1, axis=0)[:, jr]
    vj = sl(vo)
    au3 = -DTU * sl(ops['uya'])
    bu3 = 1.0 - DTU * sl(ops['uyb'])
    cu3 = -DTU * sl(ops['uyc'])
    cu3 = cu3.at[:, JM - 3].set(0.0)
    ru3 = uj + DTU * (uxa * uim1 + uxb * uj + uxc * uip1)
    ru3 = ru3.at[:, JM - 3].set(ru3[:, JM - 3] + DTU[:, JM - 3] * sl(ops['uyc'])[:, JM - 3] * uo[:, JM - 1])
    ru3 = ru3 + DTU * cross_u(fux, fuy)
    av3 = -DTV * sl(ops['vya'])
    bv3 = 1.0 - DTV * sl(ops['vyb'])
    cv3 = -DTV * sl(ops['vyc'])
    cv3 = cv3.at[:, JM - 3].set(0.0)
    rv3 = vj + DTV * (vxa * vim1 + vxb * vj + vxc * vip1)
    rv3 = rv3.at[:, JM - 3].set(rv3[:, JM - 3] + DTV[:, JM - 3] * sl(ops['vyc'])[:, JM - 3] * vo[:, JM - 1])
    rv3 = rv3 + DTV * cross_v(fvx, fvy)
    au3 = jnp.where(mus, au3, 0.0)
    bu3 = jnp.where(mus, bu3, 1.0)
    cu3 = jnp.where(mus, cu3, 0.0)
    ru3 = jnp.where(mus, ru3, 0.0)
    av3 = jnp.where(mvs, av3, 0.0)
    bv3 = jnp.where(mvs, bv3, 1.0)
    cv3 = jnp.where(mvs, cv3, 0.0)
    rv3 = jnp.where(mvs, rv3, 0.0)
    mv_ax = lambda a: jnp.moveaxis(a, 1, 0)       # noqa: E731
    uu = _tri_fwd_bwd(mv_ax(au3), mv_ax(bu3), mv_ax(cu3), mv_ax(ru3), True)
    vv = _tri_fwd_bwd(mv_ax(av3), mv_ax(bv3), mv_ax(cv3), mv_ax(rv3), True)
    uo = uo.at[:, jr].set(jnp.moveaxis(uu, 0, 1))
    vo = vo.at[:, jr].set(jnp.moveaxis(vv, 0, 1))
    uo = uo.at[IM - 1, JM - 1, :].set(uonp)
    uo = uo.at[IVNP0, JM - 1, :].set(vonp)
    return uo, vo, vonp


def stage_odiff(K, Kb, s, fx):
    """ocean_step_odiff.stage_odiff_ported: ODIFF every 6th step (mod(itime, 6) == 0), a lax.cond on the traced step number."""
    Ko = K['oc']
    s = dict(s)
    do = (fx['itime'] % 6) == 0

    def yes(args):
        mo, uo, vo, dh = args
        return odiff_dev(Ko, Kb, mo, uo, vo, dh)

    def no(args):
        mo, uo, vo, dh = args
        return uo, vo, s['vonp']
    uo, vo, vonp = lax.cond(do, yes, no, (s['mo'], s['uo'], s['vo'], s['dh3d']))
    s['uo'], s['vo'], s['vonp'] = uo, vo, vonp
    return s


# ====================================================================================================== stage 10: OCNMESO (GM / Redi)
def stage_meso(K, Kb, s, fx):
    from ocnmeso_jax import ocnstate_derived_jax, densgrad_vertical_jax
    from gm_jax import isoslope4_jax, gmkdif_jax, gmfexp_jax
    from gm_vec_compare import ISO_ORDER
    Ko = K['oc']
    s = dict(s)
    T = tables()
    vg = jnp.asarray(T['vg'])
    lmm1, lmu1, lmv1 = Ko['lmm1'], Ko['lmu1'], Ko['lmv1']
    mo, g0m, s0m = s['mo'], s['g0m'], s['s0m']
    mk = jnp.asarray(Ko['mk'])
    dxypo3 = jnp.asarray(_DXYPO)[None, :, None]
    bym = jnp.where(mk, 1.0 / (mo * dxypo3), 0.0)
    pes = [s['opress']]
    for l in range(LMO):
        pes.append(jnp.where(mk[:, :, l], pes[l] + mo[:, :, l] * GRAV, pes[l]))
    pe = jnp.stack(pes, axis=2)
    pm = 0.5 * (pe[:, :, 1:] + pe[:, :, :-1])
    gup = (g0m - 2 * Z12EH * s['gzmo']) * bym
    gdn = (g0m + 2 * Z12EH * s['gzmo']) * bym
    sup = jnp.maximum(0.0, (s0m - 2 * Z12EH * s['szmo']) * bym)
    sdn = jnp.maximum(0.0, (s0m + 2 * Z12EH * s['szmo']) * bym)
    V = lambda g, sa, p: volgsp(vg, g, sa, p)   # noqa: E731
    vup = jnp.where(mk, V(gup, sup, pm), 0.0)
    vdn = jnp.where(mk, V(gdn, sdn, pm), 0.0)
    vupu = jnp.zeros_like(vup).at[:, :, 1:].set(jnp.where(mk[:, :, 1:], V(gup[:, :, :-1], sup[:, :, :-1], pm[:, :, 1:]), 0.0))
    vdnu = jnp.zeros_like(vup).at[:, :, 1:].set(jnp.where(mk[:, :, 1:], V(gdn[:, :, :-1], sdn[:, :, :-1], pm[:, :, 1:]), 0.0))
    g3d, s3d, p3d, vbar, rho = ocnstate_derived_jax(_to1(mo), _to1(g0m), _to1(s['gzmo']), _to1(s0m), _to1(s['szmo']), _to1(s['opress']), lmm1,
                                                    _to1(vup), _to1(vdn))
    dzv, bydzv, bydh, rhomz, byrhoz = densgrad_vertical_jax(lmm1, _to1(s['dh3d']), vbar, _to1(vup), _to1(vdn), _to1(vupu), _to1(vdnu))
    g1 = O.geo1()
    dxpo, dyvo = g1[2], g1[5]
    Lr = np.arange(LMO + 1)[None, None, :]
    jr_ = slice(2, JM)
    X = lambda g, sa, p_: 1.0 / volgsp(vg, g, sa, p_)   # noqa: E731
    gl, sl, pl = g3d[1:, jr_, :], s3d[1:, jr_, :], p3d[1:, jr_, :]
    gr, sr, pr = jnp.roll(gl, -1, axis=0), jnp.roll(sl, -1, axis=0), jnp.roll(pl, -1, axis=0)
    p12 = 0.5 * (pl + pr)
    rx = (X(gr, sr, p12) - X(gl, sl, p12)) * jnp.asarray(1.0 / dxpo[jr_])[None, :, None]
    gn, sn_, pn = g3d[1:, 3:JM + 1, :], s3d[1:, 3:JM + 1, :], p3d[1:, 3:JM + 1, :]
    p12y = 0.5 * (pl + pn)
    ry = (X(gn, sn_, p12y) - X(gl, sl, p12y)) * jnp.asarray(1.0 / dyvo[jr_])[None, :, None]
    u_act = (lmu1[1:, jr_, None] >= Lr) & (Lr >= 1)
    v_act = (lmv1[1:, jr_, None] >= Lr) & (Lr >= 1)
    rhox = jnp.zeros((IM + 1, JM + 1, LMO + 1)).at[1:, jr_, :].set(jnp.where(u_act, rx, 0.0))
    rhoy = jnp.zeros((IM + 1, JM + 1, LMO + 1)).at[1:, jr_, :].set(jnp.where(v_act, ry, 0.0))
    k3d = np.where(np.arange(LMO + 1)[None, None, :] <= lmm1[:, :, None], 800.0, 0.0)
    k3d[:, :, 0] = 0.0
    iso = isoslope4_jax(lmm1, rhox, rhoy, rhomz, byrhoz, bydh, dzv, k3d)
    kpl1 = _to1(s['kpl']).astype(jnp.int64)
    gmk = gmkdif_jax(lmm1, kpl1, *[iso[n] for n in ISO_ORDER])
    names = ("bxx", "byy", "bzz", "azx", "bzx", "czx", "aezx", "ezx", "cezx", "azy", "bzy", "czy", "aezy", "ezy", "cezy")
    for fields, ql in ((('g0m', 'gxmo', 'gymo', 'gzmo'), False), (('s0m', 'sxmo', 'symo', 'szmo'), True)):
        r = gmfexp_jax(lmm1, lmu1, lmv1, _to1(mo), *[_to1(s[k]) for k in fields], ql, *[gmk[n] for n in names], kpl1, bydh, bydzv)
        for k, v in zip(fields, r):
            s[k] = _from1(v)
    return s


STAGES = [('ground', stage_ground), ('ostres', stage_ostres), ('oconv', stage_oconv), ('drag', stage_drag), ('polar', stage_polar),
          ('dynamics', stage_dynamics), ('straits', stage_straits), ('post', stage_post), ('odiff', stage_odiff), ('meso', stage_meso)]


def ocean_stages(K, Kb, s, fx, trace=None, which=None):
    """The ten stages after PRECIP_OC in the order of ocean_step.ORDER (which = 'a': ground..polar, 'b': dynamics..meso, None: all)."""
    sub = STAGES if which is None else (STAGES[:5] if which == 'a' else STAGES[5:])
    for name, fn in sub:
        s = fn(K, Kb, s, fx)
        if trace is not None:
            trace[name] = s
    return s
