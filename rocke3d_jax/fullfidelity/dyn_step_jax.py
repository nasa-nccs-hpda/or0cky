"""D140: the chained dynamics step (dyn_step.py plan) with the stages converted to JAX (dyn_jax_*.py) and the rest
left on the numpy ports.

JAX stages (kind in the plan): advecv, pgf (incl. AVRX), iso, sdrag, filter_chain (FLTRUV, fltry2 x2, CONSERV_AMB_EXT x2,
ADD_AM_AS_SOLIDBODY_ROTATION), kea (calc_kea_3d), wsave (COMPUTE_WSAVE).  numpy stages: aflux, advecm, aadvt, qdynam,
trop, matopmb, se/ke bookkeeping, efix, pgrad and the glue copies.

The workspace stays a dict of numpy arrays (the numpy stages in between make a whole-step jit impossible, see
D141); every JAX stage is one jitted call fed from / returning to that dict.  pow(x, KAPA) is numpy-pow semantics
(jnp.power); the Intel libimf bitwise mode is not available in this path and is not claimed.
"""
import time

import numpy as np
import dyn_jax_env  # noqa: F401  (XLA flag, before jax)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dyn_step as ds
import dyn_jax_advecv as jav
import dyn_jax_pgf as jpg
import dyn_jax_filter as jfl
import dyn_jax_pointwise as jpw

JAX_KINDS = ('advecv', 'pgf', 'iso', 'sdrag', 'filter_chain', 'kea', 'wsave')


class Kit:
    """Jitted stage functions bound to one run context (geometry / tables)."""

    def __init__(self, ctx):
        g = ctx.g
        self.ctx = ctx
        self.adv_geo = jav.advecv_geo(g)
        self.pgf, self.pgf_geo = jpg.make_pgf(g, ctx.tab)
        self.iso = jpw.make_iso(ctx.iso_geo)
        self.sdrag = jpw.make_sdrag(ctx.sdp, ctx.sdgeo)
        f = ctx.flt_geo
        self.flt = tuple(jnp.asarray(np.asarray(f[k], dtype=np.float64)) for k in ('dxyn', 'dxys', 'cosv')) + (float(f['radius']), float(ctx.omega))
        self.byim = float(ctx.gg['byim'])
        self.ws = tuple(float(ctx.gg[k]) for k in ('rgas', 'bygrav', 'dtsrc'))
        self.byaxyp = jnp.asarray(np.asarray(ctx.gg['byaxyp'], dtype=np.float64))


_N = lambda x: np.array(x)          # jax -> writable numpy copy
_F = lambda x: np.ascontiguousarray(x, dtype=np.float64)   # recorded dumps are big-endian '>f8': JAX needs native


class _NativeView(dict):
    """Read access to the workspace returns native-endian float64 arrays; writes go to the underlying dict."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def __getitem__(self, k):
        x = self.w[k]
        return _F(x) if isinstance(x, np.ndarray) else x

    def __setitem__(self, k, v):
        self.w[k] = v


def exec_stage_jax(st, w, ctx, kit, kinds=JAX_KINDS):
    k, a = st.kind, st.a
    if k not in kinds:
        return ds.exec_stage(st, w, ctx)
    w = _NativeView(w)
    if k == 'advecv':
        ut, vt = jav.advecv_jax(a['dt'], w[a['u']], w[a['v']], w[a['mmean']], w[a['mbefor']], w[a['ut']], w[a['vt']],
                                w[a['mafter']], w['MU'], w['MV'], w['MW'], w['SPA'], kit.adv_geo)
        w[a['ut']] = _N(ut); w[a['vt']] = _N(vt)
        w['DUT'] = np.zeros((ds.IM, ds.JM, ds.LM)); w['DVT'] = np.zeros((ds.IM, ds.JM, ds.LM))
    elif k == 'pgf':
        r = kit.pgf(a['dt'], w[a['mam']], w[a['ut']], w[a['vt']], w[a['mafter']], w[a['s0']], w[a['sz']],
                    w['DUT'], w['DVT'], kit.pgf_geo)
        w[a['ut']] = _N(r['ut']); w[a['vt']] = _N(r['vt']); w['DUT'] = _N(r['dut']); w['DVT'] = _N(r['dvt'])
        w['GZ'] = _N(r['gz']); w['PHI'] = _N(r['phi']); w['SPA'] = _N(r['adm'])
    elif k == 'iso':
        x, y = kit.iso(w[a['a']], w[a['b']])
        w[a['a']], w[a['b']] = _N(x), _N(y)
    elif k == 'sdrag':
        u, v, bad = kit.sdrag(w['U'], w['V'], w['T'], w['PK'], w['PEDN'], w['MA'], a['dt'])
        if bool(bad):
            raise RuntimeError("SDRAG: T outside 100-373 K (Fortran calls stop_model)")
        w['U'], w['V'] = _N(u), _N(v)
    elif k == 'filter_chain':
        u, v, dam = jfl.filter_chain_jax(w['U'], w['V'], w['MA'], w['MASUM'], *kit.flt)
        w['U'] = _N(u); w['V'] = _N(v); w['DAMSUM'] = float(dam)
    elif k == 'kea':
        w['KEA'] = _N(jfl.calc_kea_3d_jax(w['U'], w['V'], kit.byim))
    elif k == 'wsave':
        w['WSAVE'] = _N(jfl.compute_wsave_jax(w['MWS'], w['T'], w['PK'], w['PEDN'], kit.byaxyp, *kit.ws))


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
