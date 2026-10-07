"""D186 (build stage S2 of JAX_COVERAGE_MATRIX section 6): ATMOSPHERE PHASE 1 of the coupled step as a device-resident JAX program.

Phase 1 = MELT_SI -> DYNAM (+QDYNAM, energy fix, TROP, PGRAD_PBL, KEA) -> CONDSE -> RADIA apply (ATM_DRV.f:88-274).  The state lives in device arrays between
the stages; the only host reads inside a step are the declared ones listed in `DECLARED_HOST` (flag read, MSTCNV LMIN-loop mask reads, the callbacks).
This module does NOT run the surface half: SURFACE, dissip, filter, ocean, ice dynamics, lakes stay on the record boundary (declared) and are not part of this stage.

Units (what is jitted): see `UNITS`.  Libm mode (numpy-pow semantics via XLA); the Intel libimf is NOT used.  Radiation: 'replay' (recorded SRHR/TRHR/COSZ1 of
ffa_step_<it>_r served through the SAME hand-off interface as the real server: jax_radiation.make_handoff_step + RadiationHandoff) or 'server' (the real persistent
Fortran server; "radiation computed by the original Fortran (hybrid component)").
"""
import clouds_jax_env_fast  # noqa: F401  (XLA flags BEFORE jax)
import jax_p1_count as CNT
CNT.install()                                   # execution counters: must precede the import of every module that defines @jax.jit functions

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import atm_step as A
import dyn_step as ds
import surface_loop as SL
import jax_harness as H
import jax_p1_glue as G
import jax_p1_dyn as D
import jax_p1_condse as CS
import jax_p1_melt as M
import jax_radiation as JR
import radiation_server as rs

IM, JM, LM = 72, 46, 40

UNITS = OrderedDict([
    ('melt_si', 'jit: seaice simelt on the full grid + selects'),
    ('dyn', 'DynDevice.run: eager sequence of ~100 jitted kernels on device arrays (or ONE fused jit with fused_dyn=True)'),
    ('condse_entry', 'jit: A-grid replication of U,V and the entry-array merge'),
    ('condse_setup', 'jit: column set-up + south-pole callback'),
    ('condse_mstcnv', 'host LMIN loop over jitted MSTCNV kernels (+ QUS subsidence callback)'),
    ('condse_post', 'jit: post-processing + LSCOND core + stores + north-pole callback + momentum back-transfer + recalc_agrid_uv'),
    ('radia', 'jit: lax.cond hand-off (packet assembly, io_callback, T update) + jit cloud masking'),
])

# stage -> (kind, description) for the harness StageRegistry (FORT/REC/NP/EJ/JJ)
STAGE_TABLE = OrderedDict([
    ('record_load', ('REC', 'host file reads of the recorded entry data of the step through the registry, then device_put (CONDSE entry set, radiation record, restart ice state)')),
    ('melt_si', ('JJ', 'MELT_SI')),
    ('dyn', ('JJ/NP', 'DYNAM+QDYNAM+energy fix+TROP+PGRAD_PBL+KEA, all kernels and glue on device; NumPy part: none; host part: dispatch of ~100 kernel calls from Python, '
                      'QDYNAM extra-column branch not executed (flagged)')),
    ('condse_entry', ('JJ', 'CONDSE entry set')),
    ('condse_setup', ('JJ/NP', 'CONDSE column set-up (JJ) + south pole column: per-column NumPy port clouds_condse_ff.condse_column through pure_callback (NP)')),
    ('condse_mstcnv', ('JJ/NP', 'MSTCNV: jitted kernels; host loop over LMIN with a per-iteration device->host mask read and bucket compaction (NP); NumPy QUS ADV1D subsidence callback (NP)')),
    ('condse_post', ('JJ/NP', 'CONDSE post-processing + LSCOND (JJ); north pole column through pure_callback (NP)')),
    ('radia', ('JJ/REC', 'RADIA apply: hand-off interface of D185; in replay mode the packet is served from the recorded ffa_step_r record (REC); with the real server FORT')),
    ('flag_read', ('NP', 'one host read of the stop-model flags per step')),
])

DECLARED_HOST = [
    'record_load: file reads (recorded inputs) + device_put at step start',
    'condse pole columns (2 pure_callbacks to the NumPy per-column port clouds_condse_ff.condse_column; not ported to JAX)',
    'MSTCNV LMIN loop (host loop, 22 iterations at LMCM=23, one device->host read of a 3,168-element mask each, bucketed jit calls) -- D146 design, unchanged',
    'MSTCNV QUS ADV1D subsidence (NumPy callback inside the jitted event block)',
    'flag_read: one device->host read of the stop-model flags per step',
    'radiation: replay server stand-in (host callback, recorded SRHR/TRHR/COSZ1) or the real persistent Fortran server (host callback)',
    'QDYNAM extra-column z branch: NOT executed (flag qdynam_do_z_extra; never reached in the real windows)',
]

STATE_DYN = ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'MASUM', 'TMOM', 'QMOM', 'MUS', 'MVS', 'MWS', 'GZ')
HIDDEN_ENTRY = ('EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'TSAVG', 'QSAVG')
DYN_TO_ENTRY = ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'PK', 'PMID', 'PEDN', 'PDSIG', 'PMIDOLD', 'GZ', 'MWS', 'PEK')
CARRY_KEYS = tuple(A.CARRY_KEYS)


class ReplayServer:
    """Stand-in for the radiation server that serves the RECORDED RADIA outputs of ffa_step_<it>_r (SRHR, TRHR, COSZ1) through RadiationHandoff.request.
    Declared limits: Q, T, RQT, KLIQ, SNOAGE, CLDSS, CLDMC are echoed from the packet (the NumPy chain does not change them at RADIA); the surface-facing
    outputs (FSF, TRSURF, ALB, FSRDIR, SRVISSURF, FSRDIF, DIRVIS, DIRNIR, DIFNIR, SRDN, CFRAC) and AIJ are ZEROS (phase 1 does not use them); no radiation is computed."""
    binary = 'none (recorded replay)'
    restart = 'none'

    def __init__(self, rrec):
        self.r = rrec
        self.calls = 0

    def request(self, state, itime, seed):
        self.calls += 1
        out = {k: np.zeros(JR.OUT_SHAPES[k]) for k in JR.OUT_FIELDS}
        for k in ('T', 'Q', 'RQT', 'KLIQ', 'SNOAGE', 'CLDSS', 'CLDMC'):
            out[k] = np.array(state[k])
        out['SRHR'], out['TRHR'], out['COSZ1'] = (np.array(self.r[k], dtype=np.float64) for k in ('SRHR', 'TRHR', 'COSZ1'))
        out['AIJ'] = np.zeros((IM, JM, 400))
        out['_wall_s'] = 0.0
        out['_server_s'] = 0.0
        return out


class Phase1:
    def __init__(self, date, rad='replay', fused_dyn=False, registry=None, server=None, live_packet=None):
        self.date = date
        self.it0 = dict(A.DATES)[date]
        self.reg = registry or H.RecordedInputRegistry()
        if not self.reg.declared('sitea'):
            H.declare_d174_inputs(self.reg, date)
        for name, src, desc in (('surface_restart_for_melt_si', 'ff_data/_pristine_restarts/<restart>.nc + ocean geometry (surface_loop.init_surface_state)',
                                 'sea-ice, lake and ocean-derived exchange state (GTEMP, SSS, MLHC) at step 0 for MELT_SI'),
                                ('live_radiation_packet', 'ff_data/nov26_day/rsv_n26_<it>_in.bin',
                                 'real-server variant only: RQT, KLIQ and the 21 surface fields of the radiation packet (D185 practice)')):
            if not self.reg.declared(name):
                self.reg.declare(name, src, description=desc, origin='D186')
        self.ctx = self.reg.read('ctx_static', lambda: A.make_ctx(date, imf=False), stage='setup')
        self.rad_mode = rad
        self.server = server
        self.live_packet = live_packet
        self.dyn = D.DynDevice(self.ctx)
        self.fused_dyn = fused_dyn
        self._fused = {}
        self.cs = CS.CondseDevice(self.ctx)
        self.const = JR.RadConst.from_ctx(self.ctx)
        self.stages = H.StageRegistry()
        for n, (k, d) in STAGE_TABLE.items():
            self.stages.register(n, k, d)
        self.ct = H.Counters()
        self.rad_log = H.RadiationCallbackLog('fortran' if rad == 'server' else 'replay')
        self.timing = {}
        self.cs_stats = {}
        self.qus = dict(calls=0, seconds=0.0, bytes=0)               # QUS ADV1D subsidence callback accounting (wrapper only counts; results untouched)
        import clouds_mstcnv_jax as _mj
        _orig = _mj._host_adv
        q = self.qus

        def _counted(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit):
            t0 = time.perf_counter()
            res = _orig(sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr, qlimit)
            q['calls'] += 1
            q['seconds'] += time.perf_counter() - t0
            q['bytes'] += int(sum(np.asarray(x).nbytes for x in (sel, ldmin, lmax, arr, mom, ml, cmneg, ierr, lerr))) + int(sum(np.asarray(x).nbytes for x in res))
            return res
        _mj._host_adv = _counted
        self._build_jits()

    # ------------------------------------------------------------------------------------------------ jits
    def _build_jits(self):
        keys_o = D.OUT_KEYS + ('PEK',)

        def entry(cse_in, S, carry):
            Aa = dict(cse_in)
            for k in DYN_TO_ENTRY:
                Aa[k] = S[k]
            for k in HIDDEN_ENTRY:
                Aa[k] = S[k]
            Aa.update(carry)
            ukm, vkm, usp, vsp, unp, vnp = G.replicate_uv_to_agrid(S['U'], S['V'])
            Aa.update(UKM=ukm, VKM=vkm, UKMSP=usp, VKMSP=vsp, UKMNP=unp, VKMNP=vnp)
            return Aa
        self._entry = jax.jit(entry)

    # ------------------------------------------------------------------------------------------------ host side: recorded data
    def load_records(self, itime, first=True):
        """All host reads of one step, through the registry (GuardedReal).  Returns a dict of numpy arrays (see keys)."""
        R = self.reg.guard_real(A.Real, stage='record_load')(self.date, itime, A.FF)
        rec = {'R': R}
        rec['cse_in'] = {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in R.cse_in.items() if hasattr(v, 'shape')}
        rec['rad'] = {k: np.ascontiguousarray(R.r[k], dtype=np.float64) for k in ('SRHR', 'TRHR', 'COSZ1')}
        if first:
            rec['S0'] = A._native(A.init_state(R))
            rec['ice'] = self.reg.read('surface_restart_for_melt_si', lambda: self._melt_inputs(), stage='record_load')
        rec['seed'] = int(round(float(R.cse_out['SEEDS'][1])))       # RADIA seed of the step = SEEDS[1] of the CONDSE exit record (jax_radiation.seed_from_record)
        return rec

    def _melt_inputs(self):
        SS = SL.init_surface_state(self.date)
        st = SL.load_statics(self.date)
        return dict(ice=SS['ice'], gtemp=SS['atm']['gtemp'], sss=SS['atm']['sss'], mlhc=SS['atm']['mlhc'], geo=st['geo'])

    def to_device(self, rec, first=True):
        out = {}
        out['cse_in'] = {k: jnp.asarray(v) for k, v in rec['cse_in'].items()}
        out['rad'] = {k: jnp.asarray(v) for k, v in rec['rad'].items()}
        if first:
            out['S'] = {k: jnp.asarray(v) for k, v in rec['S0'].items() if isinstance(v, np.ndarray)}
            mi = rec['ice']
            out['ice'] = {k: jnp.asarray(v) for k, v in mi['ice'].items()}
            out['melt_in'] = dict(gtemp=jnp.asarray(mi['gtemp']), sss=jnp.asarray(mi['sss']), mlhc=jnp.asarray(mi['mlhc']), geo=M.geo_device(mi['geo']))
        return out

    # ------------------------------------------------------------------------------------------------ one step
    def new_hold(self, rad_dev):
        h = JR.empty_hold()
        h['SRHR'], h['TRHR'] = rad_dev['SRHR'], rad_dev['TRHR']
        return h

    def step(self, itime, dev, S, carry, ms, hold, ice, timed=False, rec=None, do_melt=True):
        """One phase-1 step.  dev: device records of this step (cse_in, rad[, melt_in]); S: device state dict (atm_step names); carry: device dict (CONDSE
        carry from the previous step, {} at step 0); ms: LSCOND module vectors; hold: radiation hold; ice: ice dict.  Returns dict with the new state and snapshots.
        timed=True blocks the device after every stage to attribute wall time (adds host syncs; for the per-stage table only)."""
        snaps = {}
        info = {'stage_seconds': {}}

        def fin(name, x):
            if timed:
                jax.block_until_ready(x)
            return x

        def stage(name):
            return _Stage(self, name, info, timed)

        S = dict(S)
        with stage('melt_si'):
            if do_melt:
                ice_new, melt = M.melt_si(ice, dev['melt_in']['gtemp'], dev['melt_in']['sss'], dev['melt_in']['mlhc'], dev['melt_in']['geo'], 1800.0)
                fin('melt_si', ice_new)
            else:                                       # steps after the first: the ice state is not advanced by this stage (surface half = record boundary)
                ice_new, melt = ice, None
        with stage('dyn'):
            sd = {k: S[k] for k in STATE_DYN}
            if self.fused_dyn:
                key = ((itime - ds.ITIMEI) * 4) % 54            # the plan depends on itime only through the DIAGA phase (MODDA)
                if key not in self._fused:
                    self._fused[key] = self.dyn.make_fused(itime)
                out, fl = self._fused[key](sd)
            else:
                out, fl = self.dyn.run(sd, itime)
            fin('dyn', out)
        S.update(out)
        snaps['dyn'] = dict(S)
        with stage('condse_entry'):
            Aa = self._entry(dev['cse_in'], S, carry)
            fin('condse_entry', Aa)
        if self.cs.host is None:
            self.cs.bind(Aa)
        with stage('condse_setup'):
            R_, W0, Xs, ms_s = self.cs.setup(Aa, ms)
            fin('condse_setup', R_)
        with stage('condse_mstcnv'):
            o, err = self.cs.mstcnv(R_, self.cs_stats)
            fin('condse_mstcnv', o)
        with stage('condse_post'):
            X, ms_n, cfl, extra = self.cs.post(Aa, W0, o, ms_s, Xs)
            fin('condse_post', X)
        cfl['mstcnv_negative_cloud'] = err.astype(jnp.int32)
        for k in A.CONDSE_OUT:
            if k in X:
                S[k] = X[k]
        snaps['condse'] = dict(S)
        with stage('radia'):
            S = self.radia(itime, dev, rec, S, X, hold)
            S, hold = S
            fin('radia', S['T'])
        snaps['radia'] = dict(S)
        flags = dict(fl)
        flags.update(cfl)
        newcarry = {k: X[k] for k in CARRY_KEYS if k in X}
        if A.is_radiation_step(itime):
            newcarry['CLDSS'], newcarry['CLDMC'] = G.radia_cloud_masking(X['CLDSS'], X['CLDMC'], X['TAUSS'], X['TAUMC'])
        return dict(S=S, X=X, carry=newcarry, ms=ms_n, hold=hold, ice=ice_new, melt=melt, flags=flags, snaps=snaps, info=info, cloud_masked=newcarry)

    def radia(self, itime, dev, rec, S, X, hold):
        if not hasattr(self, '_handoff'):
            srv = ReplayServer(None) if self.rad_mode == 'replay' else self.server
            self._handoff = JR.RadiationHandoff(srv, log=JR.RadLog(getattr(srv, 'binary', None), getattr(srv, 'restart', None)))
            self._hstep = JR.make_handoff_step(self._handoff, self.const)
        if self.rad_mode == 'replay':
            self._handoff.server.r = rec['rad']
        hold = OrderedDict(hold)
        hold['SNOAGE'] = X['SNOAGE']                        # the packet's SNOAGE is the CONDSE-exit value (D185)
        cloud = {k: X[k] for k in JR.CLOUD_IN}
        atm = {k: S[k] for k in ('T', 'Q', 'PK', 'PMID', 'PDSIG', 'PEDN', 'MA')}
        atm['LTROPO'] = S['LTROPO']
        pk_state = {**atm, **cloud}
        if self.rad_mode == 'replay':
            surf = {k: jnp.zeros(rs.INPUT_FIELDS[k]) for k in JR.SURF_IN}      # replay: the 21 surface fields are zeros (unused by the replay server; declared)
        else:
            surf = {k: jnp.asarray(self.live_packet[k]) for k in JR.SURF_IN}
        T, Q, hold_new, cosz = self._hstep(pk_state, surf, hold, jnp.int32(itime), jnp.int32(rec['seed']) if rec else jnp.int32(0), dev['rad']['COSZ1'])
        S = dict(S)
        S['T'], S['Q'] = T, Q
        S['SRHR'], S['TRHR'], S['COSZ1'] = hold_new['SRHR'], hold_new['TRHR'], cosz
        return S, hold_new


class _Stage:
    def __init__(self, ph, name, info, timed):
        self.ph, self.name, self.info, self.timed = ph, name, info, timed

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c = CNT.stage(self.name)
        self.res = self.c.__enter__()
        return self

    def __exit__(self, *e):
        self.c.__exit__(*e)
        dt = time.perf_counter() - self.t0
        self.info['stage_seconds'][self.name] = dt
        self.info.setdefault('stage_jit_calls', {})[self.name] = self.res.get('jit_calls')
        self.info.setdefault('stage_eager_prims', {})[self.name] = self.res.get('eager_primitive_calls')
        return False
