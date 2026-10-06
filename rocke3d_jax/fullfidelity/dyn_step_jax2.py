"""D145 (D142-D144): the chained dynamics step with AFLUX, ADVECM(+MAtoP), AADVT and QDYNAM converted to JAX in addition to
the D140 stages (dyn_step_jax.py is not modified; this module extends its kit and its stage executor).

JAX stages (kind): advecv, pgf, iso, sdrag, filter_chain, kea, wsave (D140), aflux, advecm (dyn_jax_aflux), aadvt
(dyn_jax_aadvt, qlimit=.false.), qdynam (dyn_jax_qdynam; the extra-column z branch falls back to numpy for the
bookkeeping, see that module).  numpy stages: trop, matopmb (MAtoPMB), se/ke bookkeeping, efix, pgrad, the glue copies.

Workspace and conversion conventions as dyn_step_jax: the workspace is a dict of numpy arrays, every JAX stage is one
(or a few) jitted call(s) fed from / returning to that dict.  PK = PMID**KAPA is numpy-pow semantics (jnp.power).
"""
import time

import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dyn_step as ds
import dyn_step_jax as dj
import dyn_jax_aflux as jaf
import dyn_jax_aadvt as jat
import dyn_jax_qdynam as jqd

NEW_KINDS = ('aflux', 'advecm', 'aadvt', 'qdynam')
JAX_KINDS = dj.JAX_KINDS + NEW_KINDS
_N, _F, _NativeView = dj._N, dj._F, dj._NativeView


class Kit(dj.Kit):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.afl = jaf.make_aflux(ctx.g, ctx.tab)
        self.qd = jqd.QdynamKit(ctx.qg)


def exec_stage_jax(st, w, ctx, kit, kinds=JAX_KINDS):
    k, a = st.kind, st.a
    if k not in NEW_KINDS or k not in kinds:
        return dj.exec_stage_jax(st, w, ctx, kit, tuple(x for x in kinds if x in dj.JAX_KINDS))
    w = _NativeView(w)
    if k == 'aflux':
        r = kit.afl.aflux(a['ns'], w[a['u']], w[a['v']], w[a['ma']], w[a['masum']], w[a['me']], w[a['mesum']])
        w['MU'] = _N(r['mu']); w['MV'] = _N(r['mv']); w['MW'] = _N(r['mw']); w['CONV'] = _N(r['conv']); w['SPA'] = _N(r['spa'])
    elif k == 'advecm':
        r = kit.afl.advecm(a['dt'], w[a['mold']], w['CONV'], w['MW'])
        if int(r['n_exception']) == 2:
            raise RuntimeError('ADVECM: Mass diagnostic error (stop_model 11)')
        w[a['mnew']] = _N(r['mnew']); w[a['msum']] = _N(r['msum'])
        ds._set_matop(w, {x: _N(r[x]) for x in ('pedn', 'pmid', 'pdsig', 'pk', 'p')})
    elif k == 'aadvt':
        r = jat.aadvt_jax(a['dt'], w['MMA'], w['T'], w['TMOM'], w['MU'], w['MV'], w['MW'])
        if bool(r['bad']):
            raise RuntimeError("aadvt: courmax>1 after nstep=20 (Fortran stop_model)")
        w['T'] = _N(r['rm']); w['TMOM'] = _N(r['rmom']); w['MMA'] = _N(r['mm']); w['FPEU'] = _N(r['fqu']); w['FPEV'] = _N(r['fqv'])
    elif k == 'qdynam':
        r = kit.qd(w['Q'], w['QMOM'], w['MAOLD'], w['MUS'], w['MVS'], w['MWS'])
        w['Q'] = r['q']; w['QMOM'] = r['qmom']; w['MUS'] = r['mus']; w['MVS'] = r['mvs']; w['MWS'] = r['mws']


def run_plan_jax(plan, w, ctx, kit, kinds=JAX_KINDS, hook=None, timing=None):
    for st in plan:
        if hook is not None:
            hook('pre', st, w, ctx)
        t0 = time.perf_counter()
        exec_stage_jax(st, w, ctx, kit, kinds)
        if timing is not None:
            timing[st.kind] = timing.get(st.kind, 0.0) + time.perf_counter() - t0
        if hook is not None:
            hook('post', st, w, ctx)
    return w


def dyn_step_jax(state, ctx, kit, itime=None, plan=None, kinds=JAX_KINDS, hook=None, timing=None):
    w = ds.workspace(state)
    if plan is None:
        plan = ds.step_plan(nstep=None if itime is None else (itime - ds.ITIMEI) * 4)
    return run_plan_jax(plan, w, ctx, kit, kinds, hook, timing)
