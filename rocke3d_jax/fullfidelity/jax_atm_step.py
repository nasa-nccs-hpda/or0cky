"""D180 (stage 1 of the JAX-driven coupled step): the ATMOSPHERE half of one 30-minute step as a JAX-executed chain.

Stages, in the real call order (atm_step.py, D128), and what executes them here:
  dyn      dyn_step_jax2.dyn_step_jax: JAX for advecv, pgf, iso, sdrag, filter_chain, kea, wsave, aflux, advecm(+MAtoP), aadvt, qdynam
           (one jitted call per stage); NumPy for trop, MAtoPMB, se/ke bookkeeping, energy fix, pgrad_pbl, glue copies, the
           extra-column z branch of QDYNAM (see dyn_step_jax2 docstring).  PEK = PEDN**KAPA: JAX (jnp.power).
  condse   clouds_condse_jax.condse_step_jax: LSCOND and MSTCNV batch kernels in JAX (jit); the column set-up / post-processing /
           bookkeeping / two pole columns / snow-age exp / QUS advection of the subsidence (host callback) stay NumPy.
  radia    boundary provider (recorded SRHR/TRHR/COSZ1); the T update RAD_DRV.f:5474-5478 is a jitted JAX function here.
  surface  atm_step.stage_surface UNCHANGED (SURFACE.f x2: PBL, tile fluxes, land-ice, land patch, aggregation, ATURB+UV).
           JAX-jitted: PBL (pbl_ff), tile fluxes, aggregation, ATURB+UV (chain_two_substeps._aturb_uv_jit), get_dbl, landice chain.
           NumPy/eager Python: override_pbl/override_tiles record rewriting, first_layer_update (TMOM/QMOM), the land patch
           (land_mode 'recorded': recorded GHY outputs; 'ghy': per-cell land_chain), composite ustar/lmonin, merges.
  dissip   NumPy (dyn_glue_ff.dissip)
  filter   NumPy (dyn_filter_ff.filter_slp, MAtoPMB)
The state lives as NumPy arrays between stages (host round trips at every stage boundary): this is NOT a device-resident program.

Boundary provider: `RecordBoundary` wraps atm_step.Real and logs every record file the step reads.  Nothing is computed for those.
"""
import clouds_jax_env  # noqa: F401  (XLA flags --xla_cpu_max_isa=AVX, --xla_disable_hlo_passes=algsimp; BEFORE jax)
import os
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import atm_step as A
import clouds_condse_ff as cf
import clouds_condse_jax as ccj
import dyn_step as ds
import dyn_step_jax2 as dj2

CARRY_KEYS = A.CARRY_KEYS

JAX_STAGES = {
    'dyn': 'JAX: advecv, pgf, iso, sdrag, filter_chain, kea, wsave, aflux, advecm(+MAtoP), aadvt, qdynam (one jit call per stage); PEK pow',
    'condse': 'JAX: LSCOND batch, MSTCNV batch kernels',
    'radia': 'JAX: T update RAD_DRV.f:5474-5478 (jitted)',
    'surface': 'JAX (jit): PBL advance, tile fluxes, aggregation, ATURB+UV, get_dbl, land-ice chain',
}
NON_JAX_STAGES = [
    'dyn: NumPy trop (CALC_TROP), MAtoPMB, SE/KE bookkeeping, energy fix, PGRAD_PBL, workspace glue copies, QDYNAM extra-column z branch',
    'condse: NumPy column set-up and post-processing arithmetic, bookkeeping/hand-off, the 2 pole columns (per-column port), snow-age exp loop, '
    'QUS advection of MSTCNV subsidence (host callback), recalc_agrid_uv, condse_inputs/replicate_uv_to_agrid',
    'radia: SOCRATES/RADIA not run here (recorded outputs); RADIA cloud masking CLDSS/CLDMC (NumPy where)',
    'surface: NumPy record rewriting (override_pbl/override_tiles), first-layer TMOM/QMOM update, layer1_exports, composite ustar/lmonin, '
    'merges of ATURB output, land patch (recorded GHY outputs, or per-cell Python GHY+Ent loop when land_mode=ghy), '
    'run_chain glue (ice tile properties) and record readers',
    'dissip: NumPy (dyn_glue_ff.dissip)',
    'filter: NumPy (dyn_filter_ff.filter_slp + matopmb)',
    'all stage hand-overs: NumPy arrays on the host (no device-resident state)',
]


class RecordBoundary:
    """Record-backed boundary provider for one run: serves Real(date, itime) objects and logs the record files read."""

    def __init__(self, date, ff=A.FF):
        self.date, self.ff = date, ff
        self.log = {}          # itime -> set of loaded record keys
        self._real = {}

    def real(self, itime):
        if itime not in self._real:
            r = A.Real(self.date, itime, self.ff)
            log = self.log.setdefault(itime, set())
            orig = r._get

            def _get(key, fn, _o=orig, _l=log):
                _l.add(key)
                return _o(key, fn)
            r._get = _get
            self._real[itime] = r
        return self._real[itime]

    def radiation(self, itime):
        r = self.real(itime).r
        return np.array(r['SRHR']), np.array(r['TRHR']), np.array(r['COSZ1'])

    def available(self, itime):
        return self.real(itime).available()

    def recorded_inputs(self):
        names = {'sitea': 'ffa_step_<it>_a hidden state (step 0 only is used for the start; PDSIG, PEK, hidden fields)',
                 'siter': 'ffa_step_<it>_r SRHR/TRHR/COSZ1 (radiation output, recorded)',
                 'ci': 'ffc_cse_in non-dynamic CONDSE entry fields (cloud/precip carry, ground/ocean fields, tuning)',
                 'co': 'ffc_cse_out (only read for the RADIA cloud masking on radiation steps)',
                 's1': 'ffd_state s1 (start state, step 0 only)'}
        keys = sorted({k for s in self.log.values() for k in s})
        return dict(record_keys_read=keys, descriptions={k: names.get(k, k) for k in keys},
                    also_recorded='SURFACE records ffp/ffs/ffl/ffg/fft (tiles, PBL, land-ice, GHY outputs, Ent exports, land forcing) '
                                  'read through atm_step.surface_records; CONDSE module constants and geometry (make_cfg)')


# ---------------------------------------------------------------- JAX pieces
@jax.jit
def _pek_jax(pedn, kapa):
    return jnp.power(pedn, kapa)


@jax.jit
def _radia_T_jax(T, srhr, trhr, cosz1, ma, pk, mask, dtsrc, bysha):
    byma = 1.0 / ma
    h = jnp.transpose(srhr[1:], (1, 2, 0)) * cosz1[:, :, None] + jnp.transpose(trhr[1:], (1, 2, 0))
    inc = h * dtsrc * bysha * jnp.transpose(byma, (1, 2, 0)) / jnp.transpose(pk, (1, 2, 0))
    return jnp.where(mask, T + inc, T)


def make_kit(ctx):
    return dj2.Kit(ctx.dyn)


# ---------------------------------------------------------------- stages
def stage_dyn(S, R, ctx, kit, tm):
    w = dj2.dyn_step_jax({k: S[k.upper()] for k in ds.STATE_KEYS}, ctx.dyn, kit, itime=R.itime, timing=tm)
    for k in A.DYN_OUT:
        if k in w:
            S[k] = np.array(w[k], copy=True)
    S['PEK'] = np.array(_pek_jax(jnp.asarray(S['PEDN'], dtype=jnp.float64), ctx.kapa))
    return S


def stage_condse(S, R, ctx, ms):
    inp = A.condse_inputs(S, R)
    X, cnt = ccj.condse_step_jax(inp, ctx.cfg, ms=ms)
    for k in A.CONDSE_OUT:
        if k in X:
            S[k] = np.array(X[k], copy=True)
    S['_condse_counts'] = cnt
    S['_condse_X'] = X
    return S


def stage_radia(S, R, ctx, prov):
    S['SRHR'], S['TRHR'], S['COSZ1'] = prov.radiation(R.itime)
    m = A.imaxj_mask(ctx)
    S['T'] = np.array(_radia_T_jax(jnp.asarray(S['T']), jnp.asarray(S['SRHR']), jnp.asarray(S['TRHR']), jnp.asarray(S['COSZ1']),
                                   jnp.asarray(S['MA']), jnp.asarray(S['PK']), jnp.asarray(m)[:, :, None], ctx.dtsrc, 1.0 / ctx.sha))
    if A.is_radiation_step(R.itime):
        X = S.get('_condse_X') or R.cse_out
        S['_cloud_rad'] = A.radia_cloud_masking(X['CLDSS'], X['CLDMC'], X['TAUSS'], X['TAUMC'])
    return S


def atm_step_jax(S, prov, itime, ctx, kit, ms, land_mode='recorded', timing=None):
    """One atmosphere step from state S (the atm_step state dict, or None for the real start state at itime).
    Returns (S_end, info).  info: timing per stage, compile count, jax stages, non-jax stages, recorded inputs."""
    R = prov.real(itime)
    tm = timing if timing is not None else {}
    S = S if S is not None else A.init_state(R)
    A._native(S)
    snaps = {}
    ncomp = [0]

    def _cb(name, *a, **k):
        if 'backend_compile' in name:
            ncomp[0] += 1
    try:
        jax.monitoring.register_event_duration_secs_listener(_cb)
    except Exception:
        pass
    for st in A.STAGES:
        t0 = time.perf_counter()
        if st == 'dyn':
            stage_dyn(S, R, ctx, kit, tm)
        elif st == 'condse':
            stage_condse(S, R, ctx, ms)
        elif st == 'radia':
            stage_radia(S, R, ctx, prov)
        elif st == 'surface':
            A.stage_surface(S, R, ctx, tm=tm, land_mode=land_mode)
        elif st == 'dissip':
            A.stage_dissip(S, ctx)
        elif st == 'filter':
            A.stage_filter(S, ctx)
        tm['stage_' + st] = tm.get('stage_' + st, 0.0) + time.perf_counter() - t0
        A._native(S)
        snaps[st] = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in S.items() if not k.startswith('_')}
    info = dict(itime=itime, timing={k: v for k, v in tm.items() if k.startswith('stage_')}, snaps=snaps, compiles=ncomp[0],
                jax_stages=JAX_STAGES, non_jax_stages=NON_JAX_STAGES, recorded_inputs=prov.recorded_inputs(), land_mode=land_mode,
                radiation='recorded SRHR/TRHR/COSZ1 (no radiation computed; no Fortran callback in this stage)')
    return S, info


def run_chain_jax(date, it0, nsteps, ctx, land_mode='recorded', ff=A.FF, on_step=None):
    """nsteps consecutive steps; step k starts from OUR end state of step k-1 (same hand-over as atm_step_fast.run_chain)."""
    cf.set_backend('imf' if ctx.imf else 'numpy')
    prov = RecordBoundary(date, ff)
    kit = make_kit(ctx)
    S, ms, out = None, {}, []
    for k in range(nsteps):
        t0 = time.perf_counter()
        S, info = atm_step_jax(S, prov, it0 + k, ctx, kit, ms, land_mode)
        info['wall'] = time.perf_counter() - t0
        if on_step is not None:
            on_step(k, it0 + k, prov.real(it0 + k), info, S)
        X = S.get('_condse_X')
        carry = {key: np.array(X[key], copy=True) for key in CARRY_KEYS if X is not None and key in X}
        if '_cloud_rad' in S:
            carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
        out.append(dict(itime=it0 + k, wall=info['wall'], timing=info['timing'], compiles=info['compiles']))
        S = {key: v for key, v in S.items() if not key.startswith('_')}
        if carry:
            S['_carry'] = carry
    return out, prov
