"""D183: the D180 JAX atmosphere chain with LABELLED host callbacks into the real build's Intel libimf at exactly the places where the NumPy
'imf' chain (atm_step.make_ctx(imf=True)) routes exp/pow through libimf.  jax_atm_step.py and the stage modules are NOT edited: `install()`
rebinds names inside the already imported modules (listed in INSTALLED) and swaps three functions of jax_atm_step.

Where libimf is used (inventory, details in scoping/D183_LIBIMF_CALLBACK_ENTRY.md):
  dynamics   pow(x, KAPA): MAtoP in ADVECM (dyn_jax_aflux.matop_jax, jnp.power -> libimf_ops.pow), PGF (dyn_jax_pgf: PKU/PKD pow, and the
             host-constant HUNDREDTHeKAPA = .01**KAPA, evaluated with the libimf scalar pow), PEK = PEDN**KAPA (this module).
             NumPy parts of the dynamics (CALC_TROP, MAtoPMB, filter SLP) already use libimf through ctx.imf (unchanged, host).
             AADVT / QDYNAM `fracm**3` stay jnp.power: the NumPy imf chain uses plain np.power there (not libimf).
  LSCOND     exp / pow / get_dq_* of the batch kernel (clouds_lscond_jax mode 'imf': OpsImf below).
  MSTCNV     qsat exp, the 4 `**` and jnp.power of the precipitation size search, pfac, FLAMW/G/I (.25), ANVIL radius (**BY3), MP exp/pow(,4.0)
             (clouds_mstcnv_jax with its `jnp` rebound to libimf_ops.JnpProxy, and conv_micro_j replaced by the copy below in which the four
             `** k.F27/F2439` operators are explicit libimf pow calls).
  host side  the NumPy parts of CONDSE (pole columns, snow-age exp) are unchanged and already use libimf (cf.set_backend('imf')).
Everything else is the D180 code.  `install('libm')` keeps the original XLA exp/pow (regression of D180).
"""
import clouds_jax_env  # noqa: F401  (XLA flags before jax)
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import libimf_ops as L
import intel_libm_ff
import jax_atm_step as J
import atm_step as A
import clouds_condse_jax as ccj
import clouds_lscond_jax as lj
import clouds_mstcnv_jax as cmj
import dyn_jax_aflux as jaf
import dyn_jax_pgf as jpg
import dyn_step_jax2 as dj2
import clouds_dq_ff as dq

INSTALLED = {}


# ------------------------------------------------------------------------------------------------ LSCOND ops
class OpsImf:
    """clouds_lscond_jax ops with exp / pow / get_dq via libimf_ops (host callback, evaluated only on the masked lanes, 0 elsewhere)."""
    name = "imf"

    @staticmethod
    def exp(x, m):
        return L.exp(x, m)

    @staticmethod
    def pw(x, y, m):
        return L.pow(x, y, m)

    @staticmethod
    def dq(kind, sm, qm, plk, mass, lhx, pl, cond, m, k):
        # same statements as lj.OpsXla.dq; only the exp is the libimf callback (on the lanes that are selected at the end)
        bc = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float64), m.shape)  # noqa: E731
        sm, qm, plk, mass, lhx, pl = (bc(a) for a in (sm, qm, plk, mass, lhx, pl))
        sign = 1.0 if kind == "c" else -1.0
        ref = qm if kind == "c" else bc(cond)
        sel = m & (ref > 0)
        slh = lhx * k.DQ_BYSHA
        qmt = qm
        tp = sm * plk / mass
        dqsum = jnp.zeros(m.shape)
        for _ in range(dq.NITER):
            qst = k.A * L.exp(lhx * (k.B - k.C / jnp.maximum(130.0, tp)), sel) / pl
            d = (qmt - mass * qst) / (1.0 + slh * qst * (lhx * k.C / (tp * tp)))
            tp = tp + slh * d / mass
            qmt = qmt - d
            dqsum = dqsum + sign * d
        dqs = jnp.maximum(0.0, jnp.minimum(dqsum, ref))
        fc = dqs / ref
        return jnp.where(sel, dqs, 0.0), jnp.where(sel, fc, 0.0)


# ------------------------------------------------------------------------------------------------ MSTCNV conv_micro_j (copy, libimf pow)
def conv_micro_j_imf(k, pl, wcu, dwcu, lfrz, wcufrz, tp, pland, flamw, flamg, flami, tlmin, tlmin1, condip_in, condgp_in):
    """clouds_mstcnv_jax.conv_micro_j with every pow / exp explicit through libimf_ops (the original has `** k.F27` operators that a module
    level rebinding cannot intercept).  Statement for statement the original otherwise."""
    from jax import lax
    hp, mc = cmj.hp, cmj.mc
    jnp_ = jnp
    pw, ex = L.pow, L.exp
    n = pl.shape
    wv = jnp_.maximum(wcu - dwcu, 0.0)
    dcg = jnp_.minimum(pw((wv / k.F193) * pw(pl / 1000.0, k.F04), k.F27), 1e-2)
    dci = jnp_.minimum(pw((wv / k.F1172) * pw(pl / 1000.0, k.F04), k.F2439), 1e-2)
    tig = jnp_.where(lfrz == 0, k.TF, k.TF - 4.0 * wcufrz)
    tig = jnp_.where(tig < k.TI - 10.0, k.TI - 10.0, tig)
    water = tp >= k.TF
    ice = ~water & (tp <= tig)
    mixed = ~water & ~ice
    ddcw = jnp_.where(pland < 0.5, 1.5e-3 * k.FITMAX, 6e-3 * k.FITMAX)
    wvu = wcu + dwcu
    wv2 = jnp_.concatenate([wv, wvu])
    pl2 = jnp_.concatenate([pl, pl])
    ddcw2 = jnp_.concatenate([ddcw, ddcw])
    wmax2 = jnp_.full(wv2.shape, 1.0) * k.WMAX
    pfac = pw(1000.0 / pl2, k.P4)

    def sbody(_, c):
        dcw, active = c
        vt = (-0.267 + dcw * (5.15e3 - dcw * (1.0225e6 - 7.55e7 * dcw))) * pfac
        ex1 = (vt >= 0.0) & (vt >= wv2)
        ex2 = ~ex1 & (vt > wmax2)
        exm = active & (ex1 | ex2)
        dcw = jnp_.where(active & ~exm, dcw + ddcw2, dcw)
        return dcw, active & ~exm

    dcw_both, _ = lax.fori_loop(0, int(mc.ITMAX) - 1, sbody, (jnp_.zeros(wv2.shape), jnp_.ones(wv2.shape, bool)))
    dcw1, dcw2 = dcw_both[:n[0]], dcw_both[n[0]:]

    def mp(rho, flam, dc, cn):
        f4p = pw(flam, k.FOUR)
        return (rho * (k.PI * hp.BY6) * cn * ex(-flam * dc)
                * (dc * dc * dc / flam + 3.0 * dc * dc / (flam * flam) + k.SIX * dc / (flam * flam * flam) + 6.0 / f4p))

    condp1_w = mp(k.RHOW, flamw, dcw1, k.CN0)
    condp_w = mp(k.RHOW, flamw, dcw2, k.CN0)
    condp1_i = mp(k.RHOIP, flami, dci, k.CN0I)
    dci_u = jnp_.minimum(pw((wvu / k.F1172) * pw(pl / 1000.0, k.F04), k.F2439), 1e-2)
    condp_i = mp(k.RHOIP, flami, dci_u, k.CN0I)
    fg = (tp - tig) / ((k.TF - tig) + k.TEENY)
    fg = jnp_.where(fg > 1.0, 1.0, fg)
    fg = jnp_.where(fg < 0.0, 0.0, fg)
    fg = jnp_.where((tlmin <= k.TF) | (tlmin1 <= k.TF), 0.0, fg)
    fi = 1.0 - fg
    cip_l = mp(k.RHOIP, flami, dci, k.CN0I)
    cgp_l = mp(k.RHOG, flamg, dcg, k.CN0G)
    condp1_m = fg * cgp_l + fi * cip_l
    dcg_u = jnp_.minimum(pw((wvu / k.F193) * pw(pl / 1000.0, k.F04), k.F27), 1e-2)
    cip_u = mp(k.RHOIP, flami, dci_u, k.CN0I)
    cgp_u = mp(k.RHOG, flamg, dcg_u, k.CN0G)
    condp_m = fg * cgp_u + fi * cip_u
    condp = jnp_.where(water, condp_w, jnp_.where(ice, condp_i, condp_m))
    condp1 = jnp_.where(water, condp1_w, jnp_.where(ice, condp1_i, condp1_m))
    condip = jnp_.where(mixed, cip_u, condip_in)
    condgp = jnp_.where(mixed, cgp_u, condgp_in)
    return condp, condp1, condip, condgp


# ------------------------------------------------------------------------------------------------ dynamics kit
class _KapaF(float):
    """KAPA whose reflected pow gives the libimf scalar pow: `.01 ** g['kapa']` in dyn_jax_pgf (HUNDREDTHeKAPA, a trace-time host constant)
    is then pow_imf(.01, KAPA), as in dyn_pgf_ff.pgf(imf_pow=True)."""

    def __rpow__(self, other):
        if L.mode() == "libm":
            return float(other) ** float(self)
        return float(L.host_pow(np.array([float(other)]), float(self))[0])


class KitImf(dj2.Kit):
    def __init__(self, ctx):
        import copy
        c2 = copy.copy(ctx)
        g = dict(ctx.g)
        g['kapa'] = _KapaF(float(g['kapa']))
        c2.g = g
        super().__init__(c2)


def make_kit(ctx):
    return KitImf(ctx.dyn)


@jax.jit
def _pek_imf(pedn, kapa):
    return L.pow(pedn, kapa)


def stage_dyn(S, R, ctx, kit, tm):
    import dyn_step as ds
    w = dj2.dyn_step_jax({k: S[k.upper()] for k in ds.STATE_KEYS}, ctx.dyn, kit, itime=R.itime, timing=tm)
    for k in A.DYN_OUT:
        if k in w:
            S[k] = np.array(w[k], copy=True)
    S['PEK'] = np.array(_pek_imf(jnp.asarray(S['PEDN'], dtype=jnp.float64), ctx.kapa))
    return S


def stage_condse(S, R, ctx, ms):
    inp = A.condse_inputs(S, R)
    X, cnt = ccj.condse_step_jax(inp, ctx.cfg, ms=ms, ls_mode=('imf' if L.mode() != 'libm' else 'xla'))
    for k in A.CONDSE_OUT:
        if k in X:
            S[k] = np.array(X[k], copy=True)
    S['_condse_counts'] = cnt
    S['_condse_X'] = X
    return S


# ------------------------------------------------------------------------------------------------ install
def install(mode='libimf'):
    """Set the libimf_ops mode and rebind the names below.  Call once, before any JAX stage is traced (fresh process)."""
    if mode == 'libimf' and not intel_libm_ff.available():
        raise RuntimeError("Intel libimf not available on this host")
    L.set_mode(mode)
    prox = L.JnpProxy()
    lj.OPS['imf'] = OpsImf
    INSTALLED.update({
        'clouds_lscond_jax.OPS[imf]': 'OpsImf (exp, pw, dq through libimf_ops)',
        'clouds_mstcnv_jax.jnp': 'libimf_ops.JnpProxy (power, exp -> libimf_ops; everything else jax.numpy)',
        'clouds_mstcnv_jax.conv_micro_j': 'conv_micro_j_imf',
        'dyn_jax_pgf.jnp': 'JnpProxy (PKU/PKD pow)',
        'dyn_jax_aflux.jnp': 'JnpProxy (MAtoP pk pow)',
        'dyn_jax_pgf hk': 'KAPA subclass: .01**KAPA = libimf scalar pow',
        'jax_atm_step.stage_dyn': 'stage_dyn (PEK pow via libimf_ops)',
        'jax_atm_step.stage_condse': 'stage_condse (LSCOND mode imf)',
        'jax_atm_step.make_kit': 'KitImf'})
    cmj.jnp = prox
    cmj.conv_micro_j = conv_micro_j_imf
    jpg.jnp = prox
    jaf.jnp = prox
    J.stage_dyn, J.stage_condse, J.make_kit = stage_dyn, stage_condse, make_kit
    return dict(INSTALLED)


CALLBACK_SITES = {
    'dyn': ['dyn_jax_aflux.matop_jax: pow(PMID, KAPA), LM scan x (IM,JM), once per ADVECM call (5 per step)',
            'dyn_jax_pgf: pow(PU, KAPA) (1 + LM scan), per PGF call', 'jax_atm_step_imf._pek_imf: pow(PEDN, KAPA) (LM,IM,JM) once per step'],
    'condse': ['LSCOND (clouds_lscond_jax OpsImf): exp, pow, get_dq on masked lanes', 'MSTCNV (clouds_mstcnv_jax): qsat exp, size-search pow, '
               'FLAM pow, microphysics exp/pow, anvil pow (inside lax loops)'],
    'host numpy (unchanged, already libimf in imf mode)': ['CALC_TROP pow, MAtoPMB pow, SLP filter pow, pole columns, snow-age exp, '
                                                           'MSTCNV make_K constants'],
}
