"""D186 (stage S2): the DYNAMICS block of phase 1 (atm_phase1 ATM_DRV.f:88-235) as a DEVICE-RESIDENT program.

The plan of dyn_step.step_plan (the real call order, ~130 stages) is executed on a workspace dict of DEVICE arrays; nothing is copied to the host between
stages.  Every stage of dyn_step_jax2 keeps its JAX kernel (advecv, pgf, iso, sdrag, filter_chain, kea, wsave, aflux, advecm, aadvt, qdynam); the stages that
were NumPy in D180 are now jnp (jax_p1_glue): reinit (MASUM sum), pscale, accum, mma, avg, flux scaling, matopmb, DIAGA pole fix, QCL/QCI rescale,
conserv_se/ke, energy fix, CALC_TROP, PGRAD_PBL.  PEK = PEDN**KAPA is appended (as atm_step.stage_dyn does).

Control checks that the NumPy/Fortran code RAISES (sdrag T range, ADVECM mass diagnostic, AADVT courant, QUS errors) are returned as device FLAGS in
`flags` and read by the caller once per step (a declared host read).  The QDYNAM extra-column vertical branch (`do_z_extra`), which the real windows never
reach and which the D144 module runs as a NumPy fallback, is NOT executed here: it is flagged (`qdynam_do_z_extra`) and the step must then be declared invalid.

Two execution modes of the SAME code: eager (each kernel is one jit dispatch on device arrays) and fused (`make_fused`: the whole block traced under one jax.jit).
Needs the XLA flags of clouds_jax_env_fast (imported first).  Libm mode (numpy-pow semantics via jnp.power); the Intel libimf is not used.
"""
import clouds_jax_env_fast  # noqa: F401  (XLA flags BEFORE jax)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import dyn_step as ds
import dyn_step_jax2 as dj2
import dyn_jax_aadvt as jat
import dyn_jax_advecv as jav
import dyn_jax_filter as jfl
import jax_p1_glue as G

IM, JM, LM = 72, 46, 40
MZ = ds.MZ
DT = ds.DT
DTLF = 2. * DT

# dynamics outputs returned to the caller (atm_step.DYN_OUT + PEK)
OUT_KEYS = ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'MASUM', 'TMOM', 'QMOM', 'MUS', 'MVS', 'MWS', 'GZ',
            'KEA', 'PTROPO', 'LTROPO', 'WSAVE', 'DPDX', 'DPDY', 'DPDX0', 'DPDY0', 'PHI', 'PMIDOLD', 'PDSIG')
FLAG_KEYS = ('sdrag_bad', 'advecm_exc2', 'aadvt_bad', 'qdynam_err', 'qdynam_do_z_extra')


# ----------------------------------------------------------------------------- small jitted glue units
@jax.jit
def _reinit(ma, u, v, tmom):
    return G.seqsum(ma, 0), u, v, tmom[MZ]


@jax.jit
def _pscale(mu, mv, mw, kg2mb):
    return mu * kg2mb, mv * kg2mb, mw * kg2mb


@jax.jit
def _accum(mus, mvs, mws, pu, pv, sd):
    mws = mws.at[:, :, :LM - 1].set(mws[:, :, :LM - 1] + sd)
    return mus + pu, mvs + pv, mws


@jax.jit
def _mma(meven, axyp):
    return jnp.transpose(meven * axyp[None], (1, 2, 0))


@jax.jit
def _avg(t, tt, tz, tzt):
    return .5 * (t + tt), .5 * (tz + tzt)


@jax.jit
def _flux_scale(mus, mvs, mws):
    mws = mws.at[:, :, :LM - 1].set(mws[:, :, :LM - 1] * DTLF)
    return mus * DTLF, mvs * DTLF, mws


@jax.jit
def _set_matop(pedn_full, r_pedn):
    return pedn_full.at[:LM].set(r_pedn)


class DynDevice:
    """Constants + kernels of one run (built once per date): the D140/D145 kit (jitted kernels with traced geometry), the glue constants and the plan."""

    def __init__(self, ctx, counters=None):
        self.ctx = ctx
        self.kit = dj2.Kit(ctx.dyn)
        self.K = G.make_consts(ctx)
        self.C = counters
        self.kpl = {}

    def plan(self, itime):
        return ds.step_plan(nstep=(itime - ds.ITIMEI) * 4)

    # ---- counted kernel call
    def _k(self, name, fn):
        return fn if self.C is None else self.C.count_jit(fn, name)

    def qdynam(self, w):
        kit = self.kit.qd
        a = (w['Q'], w['QMOM'], w['MAOLD'], w['MUS'], w['MVS'], w['MWS'])
        mb, rm, rmom, q0, my, mz = self._k('qdynam.prep', kit._prep)(*a, kit.axyp, kit.kg2mb, kit.byim_geom, kit.imaxj)
        qo, qmo = self._k('qdynam.cycles', kit._all_cycles)(rm, rmom, mb, q0['mu'], q0['mv'], q0['mw'], q0['ncyc'], q0['nstepx'], q0['ncycxy'],
                                                         my, mz, kit.imaxj, kit.byim_qus, kit.c3)
        return qo, qmo, q0

    def run(self, state, itime):
        """state: dict of device arrays with ds.STATE_KEYS (upper-case names).  Returns (out dict of device arrays, flags dict of device scalars)."""
        K, kit, ctx = self.K, self.kit, self.ctx
        k = self._k
        g = ctx.dyn.g
        w = {n: state[n] for n in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'MASUM', 'TMOM', 'QMOM', 'MUS', 'MVS', 'MWS', 'GZ')}
        w['DUT'] = jnp.zeros((IM, JM, LM))
        w['DVT'] = jnp.zeros((IM, JM, LM))
        fl = {n: jnp.zeros((), jnp.int32) for n in FLAG_KEYS}
        for st in self.plan(itime):
            kd, a = st.kind, st.a
            if kd == 'save_old':
                w['MAOLD'] = w['MA']
                w['PMIDOLD'] = w['PMID']
            elif kd == 'se_init':
                w['SEINIT'] = k('conserv_se', G.conserv_se)(w['MA'], w['MASUM'], w['PK'], w['T'], w['Q'], w['QCI'], K)
            elif kd == 'ke_init':
                w['KEINIT'] = k('conserv_ke', G.conserv_ke)(w['MA'], w['U'], w['V'], K)
            elif kd == 'flux_zero':
                for n in ('MUS', 'MVS', 'MWS'):
                    w[n] = jnp.zeros((IM, JM, LM))
            elif kd == 'reinit':
                w['MASUM'], _, _, w['TZ'] = k('reinit', _reinit)(w['MA'], w['U'], w['V'], w['TMOM'])
                w['UX'] = w['U']; w['UT'] = w['U']
                w['VX'] = w['V']; w['VT'] = w['V']
            elif kd == 'aflux':
                r = k('aflux', kit.afl.aflux)(a['ns'], w[a['u']], w[a['v']], w[a['ma']], w[a['masum']], w[a['me']], w[a['mesum']])
                w['MU'], w['MV'], w['MW'], w['CONV'], w['SPA'] = r['mu'], r['mv'], r['mw'], r['conv'], r['spa']
            elif kd == 'advecm':
                r = k('advecm', kit.afl.advecm)(a['dt'], w[a['mold']], w['CONV'], w['MW'])
                fl['advecm_exc2'] = jnp.maximum(fl['advecm_exc2'], (r['n_exception'] == 2).astype(jnp.int32))
                w[a['mnew']] = r['mnew']; w[a['msum']] = r['msum']
                w['PEDN'] = k('set_matop', _set_matop)(w['PEDN'], r['pedn'])
                w['PMID'], w['PDSIG'], w['PK'], w['P'] = r['pmid'], r['pdsig'], r['pk'], r['p']
            elif kd == 'advecv':
                ut, vt = k('advecv', jav.advecv_jax)(a['dt'], w[a['u']], w[a['v']], w[a['mmean']], w[a['mbefor']], w[a['ut']], w[a['vt']],
                                                     w[a['mafter']], w['MU'], w['MV'], w['MW'], w['SPA'], kit.adv_geo)
                w[a['ut']], w[a['vt']] = ut, vt
                w['DUT'] = jnp.zeros((IM, JM, LM)); w['DVT'] = jnp.zeros((IM, JM, LM))
            elif kd == 'pgf':
                r = k('pgf', kit.pgf)(a['dt'], w[a['mam']], w[a['ut']], w[a['vt']], w[a['mafter']], w[a['s0']], w[a['sz']], w['DUT'], w['DVT'],
                                      kit.pgf_geo)
                w[a['ut']], w[a['vt']], w['DUT'], w['DVT'] = r['ut'], r['vt'], r['dut'], r['dvt']
                w['GZ'], w['PHI'], w['SPA'] = r['gz'], r['phi'], r['adm']
            elif kd == 'pscale':
                w['PU'], w['PV'], w['SD'] = k('pscale', _pscale)(w['MU'], w['MV'], w['MW'], K['kg2mb'])
            elif kd == 'iso':
                w[a['a']], w[a['b']] = k('iso', kit.iso)(w[a['a']], w[a['b']])
            elif kd == 'copy':
                w[a['dst']] = w[a['src']]
            elif kd == 'accum':
                w['MUS'], w['MVS'], w['MWS'] = k('accum', _accum)(w['MUS'], w['MVS'], w['MWS'], w['PU'], w['PV'], w['SD'])
            elif kd == 'mma':
                w['MMA'] = k('mma', _mma)(w['MEVEN'], K['axyp'])
            elif kd == 'aadvt':
                r = jat.aadvt_jax(a['dt'], w['MMA'], w['T'], w['TMOM'], w['MU'], w['MV'], w['MW'])
                fl['aadvt_bad'] = jnp.maximum(fl['aadvt_bad'], r['bad'].astype(jnp.int32))
                w['T'], w['TMOM'], w['MMA'], w['FPEU'], w['FPEV'] = r['rm'], r['rmom'], r['mm'], r['fqu'], r['fqv']
            elif kd == 'tz':
                w['TZ'] = w['TMOM'][MZ]
            elif kd == 'avg':
                w['TT'], w['TZT'] = k('avg', _avg)(w['T'], w['TT'], w['TZ'], w['TZT'])
            elif kd == 'sdrag':
                u, v, bad = k('sdrag', kit.sdrag)(w['U'], w['V'], w['T'], w['PK'], w['PEDN'], w['MA'], a['dt'])
                fl['sdrag_bad'] = jnp.maximum(fl['sdrag_bad'], jnp.asarray(bad).astype(jnp.int32))
                w['U'], w['V'] = u, v
            elif kd == 'matopmb':
                r = k('matopmb', G.matopmb)(w['MA'], K)
                w['MASUM'], w['PEDN'], w['PMID'], w['PK'], w['PDSIG'], w['P'] = r['masum'], r['pedn'], r['pmid'], r['pk'], r['pdsig'], r['p']
            elif kd == 'flux_scale':
                w['MUS'], w['MVS'], w['MWS'] = k('flux_scale', _flux_scale)(w['MUS'], w['MVS'], w['MWS'])
            elif kd == 'diaga':
                w['Q'] = k('diaga', G.diaga_poles)(w['Q'])
            elif kd == 'filter_chain':
                u, v, dam = k('filter_chain', jfl.filter_chain_jax)(w['U'], w['V'], w['MA'], w['MASUM'], *kit.flt)
                w['U'], w['V'], w['DAMSUM'] = u, v, dam
            elif kd == 'wsave':
                w['WSAVE'] = k('wsave', jfl.compute_wsave_jax)(w['MWS'], w['T'], w['PK'], w['PEDN'], kit.byaxyp, *kit.ws)
            elif kd == 'qscale':
                for n in ('QCL', 'QCI'):
                    w[n] = k('qscale', G.qscale)(w[n], w['MAOLD'], w['MA'])
            elif kd == 'qdynam':
                qo, qmo, q0 = self.qdynam(w)
                w['Q'], w['QMOM'] = qo, qmo
                w['MUS'], w['MVS'], w['MWS'] = q0['mu'], q0['mv'], q0['mw']
                fl['qdynam_err'] = jnp.maximum(fl['qdynam_err'], q0['err'].astype(jnp.int32))
                fl['qdynam_do_z_extra'] = jnp.maximum(fl['qdynam_do_z_extra'], q0['do_z_extra'].astype(jnp.int32))
            elif kd == 'se_final':
                w['SEFINAL'] = k('conserv_se', G.conserv_se)(w['MA'], w['MASUM'], w['PK'], w['T'], w['Q'], w['QCI'], K)
            elif kd == 'ke_final':
                w['KEFINAL'] = k('conserv_ke', G.conserv_ke)(w['MA'], w['U'], w['V'], K)
            elif kd == 'efix':
                w['T'], w['DSEPKE'], w['MMGLOB'] = k('efix', G.energy_fix)(w['SEINIT'], w['KEINIT'], w['SEFINAL'], w['KEFINAL'], w['MASUM'], w['T'], w['PK'], K)
            elif kd == 'trop':
                w['PTROPO'], w['LTROPO'] = k('trop', G.calc_trop)(w['T'], w['PK'], w['PMID'], K)
            elif kd == 'pgrad':
                w['DPDX'], w['DPDY'], w['DPDX0'], w['DPDY0'] = k('pgrad', G.pgrad_pbl)(w['T'][:, :, 0], w['PK'][0], w['PMID'][0], w['PEDN'][0], w['PHI'][:, :, 0], K)
            elif kd == 'kea':
                w['KEA'] = k('kea', jfl.calc_kea_3d_jax)(w['U'], w['V'], kit.byim)
            else:
                raise ValueError(kd)
        out = {n: w[n] for n in OUT_KEYS}
        out['PEK'] = k('pek', G.pek_of)(w['PEDN'], K)
        return out, fl

    def make_fused(self, itime):
        """The same block traced under ONE jax.jit (one compiled program per distinct plan, i.e. per distinct DIAGA phase of itime)."""
        names = ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'MASUM', 'TMOM', 'QMOM', 'MUS', 'MVS', 'MWS', 'GZ')
        C0, self.C = self.C, None

        def f(*arrs):
            return self.run(dict(zip(names, arrs)), itime)
        jf = jax.jit(f)
        self.C = C0
        return lambda state: jf(*[state[n] for n in names])


def flags_to_host(fl):
    """The one host read of the step: returns dict of python ints."""
    return {k: int(v) for k, v in jax.device_get(fl).items()}
