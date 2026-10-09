"""D191 (build stage S6, numbers of S7): ONE coupled step assembled from the verified pieces, DEVICE-RESIDENT between the stages.

    J1  phase 1     MELT_SI -> DYNAM (+QDYNAM, energy fix, TROP, PGRAD_PBL, KEA) -> CONDSE (D189 device MSTCNV, fused libimf callbacks; pole columns and QUS
                    subsidence are host callbacks) -> RADIA hand-off (D185: replay of the recorded SRHR/TRHR/COSZ1, or the real persistent Fortran server) -> carry
                    write-back (CLDSS/CLDMC masking, held SNOAGE/RQT/KLIQ of the server)          [jax_atm_phase1 + d187_imf_p1 + clouds_mstcnv_dev]
    J2  surface     PRECIP_*/TOC2SST (surface_pre_dev) -> device rewrite of the state columns of the recorded SURFACE templates (jax_tpl_state) -> SURFACE stage
                    (D188 jax_surface: PBL, tiles, land GHY with RECORDED Ent exports, land ice, 2 substeps, ATURB) -> tile accumulators -> GROUND_*, RIVERF, DYNSI,
                    ocean (OADVT2 X pre-pass = host callback), FORM_SI, ADVSI     [jax_posttile + jax_ocean + jax_advsi ...]
    J3  phase 2     DISSIP + FILTER (jax_phase2), libimf pow through the labelled host callback
    host            record read + template upload (declared recorded inputs), seeds/clock, flag read, logging.
The state is a pytree whose groups follow jax_state_d181 (atm, atm_carry, atm_ms, rad_hold, ocean, ice, ice_dyn, lake, landice, exch, land, tile, clock); the
tile validity masks are checked against jax_state_d181.tile_masks of the MELT_SI result.  Nothing existing is edited.  SOCRATES/RADIA never ported or modified.

The host/device interface per step is: records (uploaded), template (uploaded), callbacks (labelled), flag read (36 B), and for validation runs ONLY a final
device_get of the stage snapshots.  No state array is copied to NumPy between the stages (checked with jax.transfer_guard in d191_run.py).

Not implemented / declared: the loadbl donor rule for NEW ice tiles (D190 section 6): the template of every step is read from that step's record (so a tile that
appears has its columns), the tile-set equality with the MELT_SI result is checked and reported each step.
"""
import clouds_jax_env  # noqa: F401  (XLA flags BEFORE jax)
import contextlib
import copy
import os
import sys
import time
import types

import numpy as np

import d187_imf_p1 as W                 # imports jax_atm_phase1 first (execution counters), wires libimf into the D186 modules
INST = W.install()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_atm_phase1 as P1  # noqa: E402
import jax_p1_count as CNT  # noqa: E402
import jax_p1_melt as M  # noqa: E402
import jax_harness as H  # noqa: E402
import libimf_ops as LI  # noqa: E402
import libimf_fused as FX  # noqa: E402
import clouds_mstcnv_jax as mj  # noqa: E402
import clouds_mstcnv_dev as md  # noqa: E402
import atm_step as A  # noqa: E402
import surface_loop as L  # noqa: E402
import surface_loop_v2 as V2  # noqa: E402
import jax_surface as JS  # noqa: E402
import jax_posttile as PT  # noqa: E402
import jax_ocean as JO  # noqa: E402
import jax_phase2 as P2  # noqa: E402
import jax_tpl_state as TS  # noqa: E402
import jax_state_d181 as S181  # noqa: E402
import d190_util as U  # noqa: E402

IM, JM, LM = 72, 46, 40
F = np.float64


def mstcnv_new(self, R, stats=None):
    """CondseDevice.mstcnv with the cloud-base loop on the device (clouds_mstcnv_dev, D189); same function as d189_hybrid_phase1.mstcnv_new (instance-level
    replacement, no file edited)."""
    c = self.cfg['tune']
    lmcm = int(self.cfg['lmcm'])
    K = mj.make_K(c)
    for n_ in ("xmass", "bydtsrc", "dtsrc", "bybr"):
        K[n_] = np.float64(self.cfg[n_])
    K['fmpscale'] = np.float64(min(1.0, F(self.cfg['dtsrc']) / (F(1.0) * mj.mc.SECONDS_PER_HOUR)))
    Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
    S, Sc, I_, aux = self._mc_setup(Kj, R, lmcm)
    S, Sc, err, nev = md._events_device(Kj, S, Sc, I_, aux, lmcm, md.BUCKETS)
    if stats is not None:
        stats['events_per_lmin_device'] = nev
        stats['lmin_host_syncs'] = 0
    o = md._post_dev(Kj, S, Sc, I_, aux)
    return o, err


def tree_dev(d):
    """nested dict of numpy leaves -> nested dict of device arrays (numeric leaves only)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = tree_dev(v)
        elif isinstance(v, (np.ndarray, np.generic)) and np.asarray(v).dtype.kind in 'fiub':
            out[k] = jnp.asarray(v)
    return out


def tree_np(d):
    if isinstance(d, dict):
        return {k: tree_np(v) for k, v in d.items()}
    return np.asarray(d)


STAGES2 = [
    ('template_build', 'REC/NP', 'host: scatter of the recorded SURFACE rows (ffp/ffs/ffl/ffg/fft) into the fixed slots, per-cell Python GHY batch build (ghy_advnc_test.build_batch), '
                                  'upload of the template (D188 jax_surface.build_template). Disappears only when Ent and the template columns come from the pytree'),
    ('melt_si_dev', 'JJ', 'MELT_SI on the device state (jit) and replacement of the CONDSE-entry RSI by its result'),
    ('carry_writeback', 'JJ', 'CONDSE exit -> carry (CLDSS/CLDMC masking, held SNOAGE/RQT/KLIQ of the radiation server written back into the carry; one small jit)'),
    ('tile_mask_check', 'JJ', 'tile masks from the MELT_SI result (jax_state_d181.tile_masks), compared with the template slot validity on the device'),
    ('surface_pre', 'JJ', 'PRECIP_SI, AG2OG, PRECIP_OC (+TOC2SST), PRECIP_LI, PRECIP_LK, seaice_to_atmgrid (jax_posttile.surface_pre_dev)'),
    ('nit_rebuild', 'REC/NP', 'host (D195, ON by default, Coupled(nit_fix=True)): only on steps with a cell of recorded ffnit > 11: ONE declared device->host read of PREC/EPREC/PRECSS (3 small arrays), '
                              'ghy_advnc_test.build_batch (NumPy, ghy_ref_nit.run_cell_full) of those rows with OUR precipitation, device replacement of edts/ecnc/elai/ebet/nsub/dt of those cells '
                              '(jax_surface.nit_rebuild); a no-op, no transfer, on the other steps'),
    ('template_apply', 'JJ', 'device rewrite of the state columns of the SURFACE templates and of the GHY precipitation forcing (jax_tpl_state)'),
    ('surface_tiles_land', 'JJ/REC', 'SURFACE stage (D188): PBL, tile fluxes, aggregation, ATURB, land GHY (Ent exports, land forcing and tile radiation columns are RECORDED inputs), '
                                      'land ice, 2 substeps; 5 jit executions'),
    ('tile_acc', 'JJ', 'tile accumulators of the post-tile stage from the device tile outputs'),
    ('post_a', 'JJ', 'DYNSI, GROUND_LI, lake chain, RIVERF, ocean-driver head (jit)'),
    ('ocean_a', 'JJ', 'OCEANS: ground, ostres, oconv, drag, polar (jit)'),
    ('ocean_b', 'JJ/NP', 'OCEANS: dynamics, straits, post, odiff, meso (jit) with the OADVT2 X pre-pass as a host pure_callback (4 calls/step, NumPy)'),
    ('post_b', 'JJ', 'FORM_SI, TOC2SST, ADVSI (double-double Ti2b), pole replication (jit)'),
    ('dissip_filter', 'JJ/NP', 'DISSIP + FILTER (J3, jax_phase2); the pow of SLP, MAtoPMB and PEK through the libimf host callback'),
    ('state_assembly', 'JJ', 'assembly of the next-step state pytree (device dict operations, no jit)'),
]


class Coupled:
    def __init__(self, date, mstcnv='dev', log=print, nit_fix=True, nit_strict=True):
        self.nit_strict = bool(nit_strict)   # True (default): the build_batch_nit assertion nit == ffnit is NOT caught (as the NumPy chain); False: reported in jax_surface.NIT_MISMATCH
        self.nit_fix = bool(nit_fix)      # D195: GHY dts of ffnit>11 cells from OUR precipitation (as the NumPy chain); False = D193 behaviour (recorded precipitation)
        self.date = date
        self.it0 = dict(A.DATES)[date]
        self.log = log
        FX.set_mode('libimf')
        t0 = time.perf_counter()
        self.ph = W.make_phase1(date, rad='replay')
        if mstcnv == 'dev':
            self.ph.cs.mstcnv = types.MethodType(mstcnv_new, self.ph.cs)
        self.mstcnv = mstcnv
        self.ctx = self.ph.ctx
        H.declare_d174_surface_items(self.ph.reg, date)
        self.ph.reg.declare('surface_init', f'ff_data/{date}: ffc_cse_in (FLAKE...), ffp (coriolis), ffl2 (lake depth), ffo_geom.bin, ffo_state_<it0> tag 0 (MMST), restart nc '
                            f'(ocean, ice, lake, land ice, ADVSI rsix/rsiy), advsi_dumps/{date}/ffadv_in_<it0> (ADVSI geometry)',
                            description='statics and initial state of the surface (surface_loop.load_statics, init_surface_state, make_static_all, V2State)',
                            origin='D187/D190; D174 items mmst, advsi_geometry, straits_start')
        # ---- statics of the surface half (host, once per run)
        self.st = L.load_statics(date)
        self.st['ctx'] = L.make_ocean_ctx(date)
        self.SS0 = L.init_surface_state(date, st=self.st)
        self.V0 = V2.V2State(self.st, date, L.FF, self.it0)
        self.K, self.Kb = PT.make_static_all(self.st, date, self.it0)
        self.Kbd = U.to_dev(self.Kb)
        self.static181, self.src181 = S181.build_static(date)
        self.sp181 = S181.static_pytree(self.static181)
        self.melt_geo = M.geo_device(self.st['geo'])
        # ---- J3
        self.K2, self.static2 = P2.make_consts(self.ctx)
        self.j3 = P2.make_unit(self.static2)
        # ---- J2 jits (D190 split variant: 5 executions)
        self._build_jits()
        self.writeback = jax.jit(self._writeback, static_argnums=(3,))
        self._stage_cache = None
        self.ocean_keys = list(self.SS0['ocean'].keys())
        self._last_ghy = None             # D205: (ffg rows of the substep 2 of the last step, their flat cells) for the DMWLDF of the day-boundary lake update
        self.daily_lake_info = None
        self.setup_seconds = time.perf_counter() - t0

    def _build_jits(self, only=None):
        """The jitted functions of the surface half; they close over the statics self.K (compile-time constants).  `only` = names to (re)build (default all).
        D205: after the lake fractions change only 'pre' and 'pa' are rebuilt: the ocean stages (oa, ob: K keys focean, dxypo, oc, valid) and post_b (adv, is_ocean,
        valid_ocean) read no lake fraction, and re-tracing `ob` raises jax UnexpectedTracerError (ocean_ofluxv builds module-level jnp constants (ZE, DZO) at its first
        import, which happens inside the first trace of `ob`; D205 run 1)."""
        K = self.K
        mk = dict(pre=lambda: jax.jit(lambda S, mi, me, inp: PT.surface_pre_dev(K, S, mi, me, inp)),
                  pa=lambda: jax.jit(lambda S1, mid, acc, srfp, itime, V: PT.post_a(K, S1, mid, acc, srfp, itime, V)),
                  oa=lambda: jax.jit(lambda oc, fx, Kb: JO.ocean_stages(K, Kb, oc, fx, which='a')),
                  ob=lambda: jax.jit(lambda oc, fx, Kb: JO.ocean_stages(K, Kb, oc, fx, which='b')),
                  pb=lambda: jax.jit(lambda S1, c, oc, V: PT.post_b(K, S1, c, oc, V)))
        for name in (only or tuple(mk)):
            setattr(self, name, mk[name]())
        self.mask_check = jax.jit(self._mask_check)

    def daily_lake_update(self, state, log=print):
        """D205: the end-of-day lake update daily_LAKE (LAKES.f:2492; daily_lake.py, bitwise equal to the compiled Fortran on the nov26 case and on 12 stress cases) applied to
        the state at the day boundary (call between the last step of a day and the first of the next, after DAILY_ATMDYN).  ONE declared host round trip: the lake, ice and
        atmosphere-grid lake exports (device -> host), the soil water of the last land step (for DMWLDF, GHY_DRV.f:4054) -> NumPy daily_lake -> device.  Then the lake
        fractions FLAKE/FLAND/FEARTH of the statics change: self.st / the geometry / K / Kb / static181 / melt_geo are rebuilt and the jitted functions and the SURFACE
        stage cache are dropped (recompiled at the next step).  NOT applied (declared, in self.daily_lake_info['not_applied']): the GHY water/heat transfer for the changed
        lake fraction (GHY_DRV.f:4531, uses DMWLDF, DGML, svflake - returned in the info), FSF/TRSURF reset (RESET_SURF_FLUXES, radiation is replayed), MDWNIMP/EDWNIMP
        (daily_LI), the soil-moisture bookkeeping of daily_EARTH.  Returns (new state, info)."""
        import daily_lake as DL
        import ghy_ref as GR
        import jax_static as JST
        t0 = time.perf_counter()
        sf = state['surf']
        host = lambda d, ks: {k: np.array(np.asarray(d[k]), dtype=np.float64, copy=True) for k in ks}    # noqa: E731
        ice = host(sf['ice'], ('rsi', 'msi', 'snowi', 'hsi'))
        lake = host(sf['lake'], ('mwl', 'gml', 'tlake', 'mldlk'))
        assert self._last_ghy is not None and state['land_prev'] is not None, 'daily_lake_update needs a completed land step'
        rows, ecells = self._last_ghy
        w = np.asarray(state['land_prev']['dyn_next']['w'], dtype=np.float64)
        geo = self.st['geo']
        fearth0, fland0 = np.array(self.st['fearth'], copy=True), np.array(self.st['fland'], copy=True)
        dm = DL.water_deficit(rows, w, ecells, fearth0, GR.THM[0, :])
        topo = JST.topography()
        hlake, tn = DL.lake_statics(topo, np.asarray(geo['flake']), np.asarray(self.st['axyp']))
        Sin = dict(flake=np.array(geo['flake'], copy=True), fearth=fearth0, fland=fland0, **ice, **lake)
        out = DL.daily_lake(Sin, np.asarray(self.st['flice']), np.asarray(geo['focean']), tn, hlake, np.asarray(self.st['axyp']), dm,
                            valid=np.asarray(geo['valid']))
        dflake = out['flake'] - geo['flake']
        info = dict(counters=out['counters'], pow=out['pow'], n_flake_changed=int((dflake != 0).sum()), max_abs_dflake=float(np.abs(dflake).max()),
                    n_rsi_changed=int((out['rsi'] != ice['rsi']).sum()), n_dmwldf_pos=int((dm > 0).sum()),
                    reset_surf_fluxes=len(out['reset_surf_fluxes']), dmwldf=out['dmwldf'], dgml=out['dgml'], svflake=out['svflake'],
                    mdwnimp=out['mdwnimp'], edwnimp=out['edwnimp'],
                    not_applied=['GHY dfrac water/heat transfer (GHY_DRV.f:4531)', 'RESET_SURF_FLUXES (FSF/TRSURF)', 'MDWNIMP/EDWNIMP into daily_LI', 'daily_EARTH'])
        # ---- device state
        newice = dict(sf['ice'])
        for k in ('rsi', 'msi', 'snowi', 'hsi'):
            newice[k] = jnp.asarray(out[k], dtype=sf['ice'][k].dtype)
        newlake = dict(sf['lake'])
        for k in ('mwl', 'gml', 'tlake', 'mldlk'):
            newlake[k] = jnp.asarray(out[k], dtype=sf['lake'][k].dtype)
        atm = dict(sf['atm'])
        for k, v in (('gtemp', out['gtemp']), ('gtempr', out['gtempr']), ('mlhc', out['mlhc'])):
            m = np.isfinite(v)
            if k in atm and m.any():
                atm[k] = jnp.where(jnp.asarray(m), jnp.asarray(np.where(m, v, 0.0), dtype=atm[k].dtype), atm[k])
        surf = dict(sf, ice=newice, lake=newlake, atm=atm)
        # ---- statics: the lake fractions (FOCEAN, FLICE do not change)
        st = self.st
        st['fland'], st['fearth'] = out['fland'], out['fearth']
        st['geo'] = L.make_geo(st['ctx'], out['flake'])
        t1 = time.perf_counter()
        self.K, self.Kb = PT.make_static_all(st, self.date, self.it0)
        self.Kbd = U.to_dev(self.Kb)
        s181 = dict(self.static181)
        fl = np.asarray(out['flake'], dtype=np.float64)
        s181.update(flake=fl, fland=np.asarray(out['fland']), fearth=np.asarray(out['fearth']), fwater=np.asarray(st['geo']['fwater']), is_lake=fl > 0)
        self.static181 = s181
        self.sp181 = S181.static_pytree(s181)
        self.melt_geo = M.geo_device(st['geo'])
        self._build_jits(only=('pre', 'pa'))
        self._stage_cache = None
        info['seconds'] = dict(total=time.perf_counter() - t0, statics_rebuild=time.perf_counter() - t1)
        self.daily_lake_info = info
        new = dict(state)
        new['surf'] = surf
        log(f"  daily_LAKE at the day boundary: {info['n_flake_changed']} cells with FLAKE changed (max |dFLAKE| {info['max_abs_dflake']:.3e}), {info['n_rsi_changed']} RSI changed, "
            f"DMWLDF>0 in {info['n_dmwldf_pos']} cells, counters {info['counters']}")
        return new, info

    # ------------------------------------------------------------------------------------------------ small jitted helpers
    @staticmethod
    def _mask_check(rsi, wvalid, wj, wi, sp):
        """tile masks of the fixed layout (jax_state_d181.tile_masks: ptype > 0 inside the IMAXJ domain) from the MELT_SI RSI, against the validity of the
        water slots of the (recorded) template.  Returns the number of cells where the two disagree for the ocean/lake-water and the ice type, and the number
        of tiles of each type in the mask."""
        m = S181.tile_masks(sp, rsi, jnp)                           # (4,IM,JM)
        m0, m1 = m[0][wi, wj], m[1][wi, wj]
        mism0 = jnp.sum(m0 != wvalid[0]) + (jnp.sum(m[0]) - jnp.sum(m0))     # slots that disagree + mask cells outside every slot
        mism1 = jnp.sum(m1 != wvalid[1]) + (jnp.sum(m[1]) - jnp.sum(m1))
        return jnp.stack([mism0, mism1, jnp.sum(m[0]), jnp.sum(m[1]), jnp.sum(wvalid[0]), jnp.sum(wvalid[1])])

    @staticmethod
    def _writeback(carry, hold, rad_step, server):
        """CONDSE exit -> carry.  `carry` (from the phase-1 program) already holds the device-masked CLDSS/CLDMC on a radiation step.  With the REAL server (server=True)
        the server's own masked CLDSS/CLDMC and its SNOAGE are written back (D185 left this to the driver); RQT and KLIQ of the server are returned in `hold` and
        are inputs of the next radiation packet.  In replay mode the server stand-in echoes the packet, so nothing is overwritten."""
        out = dict(carry)
        if server:
            take = {k: jnp.where(rad_step, hold[k], carry[k]) for k in ('CLDSS', 'CLDMC', 'SNOAGE') if k in carry and k in hold}
            out.update(take)
        return out

    # ------------------------------------------------------------------------------------------------ state
    def initial_state(self):
        """Device state of the first step: atmosphere from the real start state, surface from the restart (all uploaded once)."""
        rec = self.ph.load_records(self.it0, first=True)
        dev = self.ph.to_device(rec, first=True)
        SS = tree_dev({g: self.SS0[g] for g in ('ocean', 'ice', 'lake', 'li', 'atm')})
        V = dict(usi=jnp.asarray(self.V0.dyn.usi), vsi=jnp.asarray(self.V0.dyn.vsi), rsix=jnp.asarray(self.V0.adv['rsix']), rsiy=jnp.asarray(self.V0.adv['rsiy']))
        state = dict(S=dev['S'], carry={}, ms=P1.CS.ms_zero(), hold=self.ph.new_hold(dev['rad']), surf=SS, V=V, land_prev=None, clock=dict(itime=jnp.asarray(self.it0)))
        return state, rec, dev

    def groups(self, state):
        """The state as the pytree of jax_state_d181 groups (names of JAX_COVERAGE_MATRIX section 4); used for sizes and descriptions."""
        sf = state['surf']
        return dict(atm=state['S'], atm_carry=state['carry'], rad_hold={k: v for k, v in state['hold'].items() if hasattr(v, 'shape')},
                    ocean=sf['ocean'], ice=sf['ice'], ice_dyn=state['V'], lake=sf['lake'], landice=sf['li'], exch=sf['atm'],
                    land=(state['land_prev'] or {}), clock=state['clock'])

    # ------------------------------------------------------------------------------------------------ one step
    def step(self, k, state, rec, dev, sr, timed=False, keep=False, server=None, lv=None, io=None):
        """One coupled step from `state` (device pytree).  rec/dev: records of this step (host dict / device dict).  Returns (new state, info, arrays)."""
        it = self.it0 + k
        ph = self.ph
        R = rec['R']
        info = dict(k=k, itime=it, jit_by_stage={}, eager_by_stage={})
        arrays = {}

        @contextlib.contextmanager
        def stage(name):
            with sr.time(name), CNT.stage(name) as res:
                yield
            info['jit_by_stage'][name] = info['jit_by_stage'].get(name, 0) + (res.get('jit_calls') or 0)
            info['eager_by_stage'][name] = info['eager_by_stage'].get(name, 0) + (res.get('eager_primitive_calls') or 0)

        def fin(x):
            if timed:
                jax.block_until_ready(x)
            return x

        S, SS, V = state['S'], state['surf'], state['V']
        # ---------------------------------------------------------------- host: template of the SURFACE stage (recorded rows, uploaded)
        with stage('template_build'):
            reg_fn = ph.reg.guard_function('surface_records', A.surface_records, stage='surface')
            recs = reg_fn(R)
            tpl, host = JS.build_template(recs)
            self._last_ghy = (host['nit_rows'][1], host['ecells'])
            g1 = recs['g1']
            irrig = np.zeros((IM, JM))
            irrig[g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1] = g1[:, 147]
            irrig_act = jnp.asarray(irrig * self.st['fearth'])
            fin(tpl)
        if self._stage_cache is None or not self._same_host(self._stage_cache['host'], host):
            self._stage_cache = dict(host=host, stage=JS.make_stage(host), apply=TS.make_apply_state(host, self.K),
                                     accf=jax.jit(lambda aux_, tpl_, _h=host: PT.tile_accumulators_dev(_h, tpl_, aux_)))
            info['stage_rebuilt'] = True
        stg, apply = self._stage_cache['stage'], self._stage_cache['apply']
        # ---------------------------------------------------------------- J1: phase 1
        with stage('melt_si_dev'):
            ice_new, melt = M.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], self.melt_geo, 1800.0)
            dev['cse_in']['RSI'] = ice_new['rsi']                       # as surface_loop.Loop.pre_cse: our RSI replaces the recorded CONDSE-entry RSI
            fin(ice_new)
        hold = state['hold']
        if server is not None and k == 0:
            for kk in ('RQT', 'KLIQ', 'SNOAGE'):
                hold[kk] = jnp.asarray(lv[kk])
        t0 = time.perf_counter()
        r = ph.step(it, dev, S, state['carry'], state['ms'], hold, ice_new, timed=timed, rec=rec, do_melt=False)
        jax.block_until_ready((r['S'], r['X'], r['flags']))
        with sr.time('flag_read'):
            fl = P1.D.flags_to_host(r['flags'])
        info['phase1_seconds'] = time.perf_counter() - t0
        info['flags'] = {kk: int(v) for kk, v in fl.items()}
        info['phase1_stage_seconds'] = r['info']['stage_seconds']
        info['phase1_stage_jit'] = r['info']['stage_jit_calls']
        info['phase1_stage_eager'] = r['info']['stage_eager_prims']
        if timed:
            for n, s in r['info']['stage_seconds'].items():
                if n in ('dyn', 'condse_entry', 'condse_setup', 'condse_mstcnv', 'condse_post', 'radia'):
                    sr.add_seconds(n, s)
        with stage('carry_writeback'):
            rad_step = jnp.asarray(bool(A.is_radiation_step(it)))
            carry = self.writeback(r['carry'], r['hold'], rad_step, server is not None)
            fin(carry)
        S1a = r['S']
        if keep:
            arrays['dyn'] = r['snaps']['dyn']
            arrays['condse'] = r['snaps']['condse']
            arrays['radia'] = r['snaps']['radia']
            arrays['X'] = r['X']
        # ---------------------------------------------------------------- J2: surface
        prec, eprec, precss = S1a['PREC'], S1a['EPREC'], S1a['PRECSS']
        with stage('surface_pre'):
            inp = dict(prec=prec, eprec=eprec, irrig_act=irrig_act)
            ice_c = {kk: v.astype(SS['ice'][kk].dtype) if kk in SS['ice'] else v for kk, v in ice_new.items()}
            melt_c = melt
            S1s, mid = self.pre(SS, ice_c, melt_c, inp)
            fin(S1s)
        with stage('tile_mask_check'):
            masks = self.mask_check(ice_new['rsi'], tpl['ns'][0]['wvalid'], jnp.asarray(host['wj']), jnp.asarray(host['wi']), self.sp181)
            fin(masks)
        with stage('nit_rebuild'):
            n_nit, n_d2h = 0, 0
            if self.nit_fix:
                tpl, n_nit, n_d2h = JS.nit_rebuild(tpl, host, prec, eprec, precss, strict=self.nit_strict)
                fin(tpl)
            info['nit_rebuild'] = dict(cells=n_nit, device_to_host_arrays=n_d2h)
        with stage('template_apply'):
            lp = state['land_prev']
            tpl_m = apply(tpl, S1s, mid['ag'], S1a['PEDN'][0], prec, eprec, precss, None if lp is None else lp['dyn_next'], lp)
            fin(tpl_m)
        with stage('surface_tiles_land'):
            Sd = {kk: S1a[kk] for kk in JS.ATM_KEYS}
            out_s, aux = stg(Sd, tpl_m)
            fin(out_s)
        with stage('tile_acc'):
            acc = self._stage_cache['accf'](aux, tpl_m)
            fin(acc)
        S2a = dict(S1a)
        S2a.update(out_s)
        if keep:
            arrays['surface'] = S2a
        itime_d = state['clock']['itime']
        with stage('post_a'):
            c = self.pa(S1s, mid, acc, S1a['PEDN'][0], itime_d, V)
            fin(c)
        with stage('ocean_a'):
            oc_a = self.oa(S1s['ocean'], c['fx'], self.Kbd)
            fin(oc_a)
        with stage('ocean_b'):
            oc_b = self.ob(oc_a, c['fx'], self.Kbd)
            fin(oc_b)
        with stage('post_b'):
            SSn, V2n, post = self.pb(S1s, c, oc_b, V)
            fin(SSn)
        # ---------------------------------------------------------------- J3: phase 2
        with stage('dissip_filter'):
            Sin = {kk: S2a[kk] for kk in P2.IN_KEYS}
            o3, extra3 = self.j3(Sin, self.K2)
            fin(o3)
        with stage('state_assembly'):
            Sf = dict(S2a)
            Sf.update({kk: o3[kk] for kk in P2.OUT_KEYS if kk in o3})
            Sf['KEA_NEW'] = o3['KEA_NEW']
            oc = {kk: (SSn['ocean'][kk] if kk in SSn['ocean'] else SS['ocean'][kk]) for kk in self.ocean_keys}
            surf_n = dict(ocean=oc, ice=SSn['ice'], lake=SSn['lake'], li=SSn['li'], atm=SSn['atm'])
            Vn = {kk: V2n[kk] for kk in ('usi', 'vsi', 'rsix', 'rsiy')}
            new = dict(S=Sf, carry=carry, ms=r['ms'], hold=r['hold'], surf=surf_n, V=Vn, land_prev=aux['r2']['land'],
                       clock=dict(itime=itime_d + 1))
        if keep:
            arrays['dissip'] = dict(S2a, T=extra3['T_dissip'], KEA_NEW=o3['KEA_NEW'])
            arrays['filter'] = Sf
            arrays['SS2'] = SSn
            arrays['V2'] = V2n
            arrays['post'] = post
            arrays['aux_land'] = aux['r2']['land']
            arrays['masks'] = masks
            arrays['tpl_wvalid'] = tpl['ns'][0]['wvalid']
            arrays['host'] = host
            arrays['n_slp_exp'] = extra3['n_slp_exp_branch']
        info['host'] = dict(Nw=host['Nw'], Nl=host['Nl'], Ne=host['Ne'], max_substeps=host['max_substeps'])
        return new, info, arrays

    @staticmethod
    def _same_host(a, b):
        for k in ('wj', 'wi', 'lj', 'li', 'ej', 'ei', 'bj', 'bi', 'wcells', 'lcells', 'ecells', 'bcells'):
            if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                return False
        return a['max_substeps'] == b['max_substeps'] and a['Nw'] == b['Nw']
