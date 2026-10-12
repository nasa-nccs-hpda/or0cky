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
import drv_radcols_dev as DD  # noqa: E402
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
    S, Sc, err, nev = md.events(Kj, S, Sc, I_, aux, lmcm, md.BUCKETS)
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
    ('radcols', 'JJ', 'D218 (ROCKE_RADCOLS=1, server mode): the radiation-derived template columns (srheat, trhr0, qsol, trup_in_rad of the tiles and the land patch) from OUR radiation state (server output, ice legs), replacing the recorded columns (drv_radcols_dev)'),
    ('radcols_legs', 'JJ', 'D218: RESET_SURF_FLUXES ice legs (b), (c), (F) at the end of the step on the carried FSF/TRSURF'),
    ('ent_daily', 'NP', 'D219 (ROCKE_ENT_LIVE=1): once per day, at the step whose records carry the end-of-day flag: the Ent state is pulled to the host, the prescribed LAI / albedo update of ent_daily_ff is applied, the Ent arrays are packed and uploaded again'),
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


IRRIG_OWN = os.environ.get('ROCKE_IRRIG_OWN', '0') == '1'    # D225, default off: irrigation from the IRRIG file and our lake state (no record read)
RHOW_, DTSRC_ = 1000.0, 1800.0
TPL_OWN = os.environ.get('ROCKE_TPL_OWN', '0') == '1'      # D224, default off: SURFACE template of steps after the first from the first-step base + own state (needs ROCKE_ENT_LIVE=1)


class Coupled:
    def __init__(self, date, mstcnv='dev', log=print, nit_fix=True, nit_strict=True):
        self.nit_strict = bool(nit_strict)   # True (default): the build_batch_nit assertion nit == ffnit is NOT caught (as the NumPy chain); False: reported in jax_surface.NIT_MISMATCH
        self.nit_fix = bool(nit_fix)      # D195: GHY dts of ffnit>11 cells from OUR precipitation (as the NumPy chain); False = D193 behaviour (recorded precipitation)
        self.date = date
        self.it0 = dict(A.DATES)[date]
        if os.environ.get('ROCKE_ENT_LIVE', '0') == '1':       # D219: Ent computed inside the GHY loop (ent_jax); initial Ent state = the restart of this date
            import ent_jax as EJ_
            EJ_.set_restart(f"{A.FF}/_pristine_restarts/fort1_{date}_itime{self.it0}.nc")
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
        self._tpl_prefetch = {}
        self.ocean_keys = list(self.SS0['ocean'].keys())
        self._last_ghy = None             # D205: (ffg rows of the substep 2 of the last step, their flat cells) for the DMWLDF of the day-boundary lake update
        self.daily_lake_info = None
        self._land3 = None                # D208: underwater-fraction soil state (w, ht of GHY ibv = 3) carried across day boundaries (first boundary: from the recorded rows)
        self.setup_seconds = time.perf_counter() - t0

    @staticmethod
    def _make_ob(K):
        """Ocean stage b. D214: on a non-CPU backend each of its five stages is its OWN jit, called in sequence (mathematically the same program): measured on the A100
        the single-jit version took 4.11 s per steady step against 1.46 s for the five jits (whole step 11.96 s -> about 8.3 s, jobs 58814135 / 58815058); the CPU keeps the
        single jit (bitwise results unchanged). ROCKE_OCEAN_B_SPLIT=1 forces the split, =0 forces the single jit."""
        mode = os.environ.get('ROCKE_OCEAN_B_SPLIT')
        if not (mode == '1' or (mode is None and jax.default_backend() != 'cpu')):
            return jax.jit(lambda oc, fx, Kb: JO.ocean_stages(K, Kb, oc, fx, which='b'))
        subs = [jax.jit((lambda fn: (lambda s, fx, Kb: fn(K, Kb, s, fx)))(f)) for _, f in JO.STAGES[5:]]

        def ob(oc, fx, Kb):
            for j in subs:
                oc = j(oc, fx, Kb)
            return oc
        return ob

    def _k_arg_jit(self, name, fn, K):
        """ROCKE_STAGE_ARGS=1 (D217): fn(K, *args) jitted with every ARRAY leaf of the statics K as a jit argument (not a compile-time constant), so the program survives the
        daily_LAKE rebuild of K without recompiling.  The non-array leaves (python numbers, strings) stay closed over; the cached program is reused only while they are equal,
        otherwise it is rebuilt (never a silent stale constant).  Default (flag off): the program closing over K, as before."""
        if os.environ.get('ROCKE_STAGE_ARGS', '0') != '1':
            return jax.jit(lambda *a: fn(K, *a))
        leaves, treedef = jax.tree_util.tree_flatten(K)
        idx = [i for i, l in enumerate(leaves) if isinstance(l, (np.ndarray, jax.Array))]
        arrs = [jnp.asarray(leaves[i]) for i in idx]
        static = tuple((i, leaves[i]) for i in range(len(leaves)) if i not in set(idx))
        key = (name, treedef, tuple((a.shape, str(a.dtype)) for a in arrs))
        if not hasattr(self, '_karg_cache'):
            self._karg_cache = {}
        hit = self._karg_cache.get(key)
        def _same(a, b):             # numbers, strings, bools exactly; the structural objects of the ocean/dyn statics (Geom, counted jit, lambda: rebuilt with K, independent of the lake fractions) by type
            if isinstance(a, (int, float, str, bool, type(None))) or isinstance(b, (int, float, str, bool, type(None))):
                return type(a) is type(b) and a == b
            return type(a) is type(b)
        same = hit is not None and len(hit[1]) == len(static) and all(i == j and _same(a, b) for (i, a), (j, b) in zip(hit[1], static))
        if hit is not None and not same:
            print(f'  _k_arg_jit {name}: non-array statics changed ({[i for (i, a), (_, b) in zip(hit[1], static) if not _same(a, b)][:6]}): program rebuilt', flush=True)
        if not same:
            def prog(arrs_, *a, _leaves=list(leaves), _idx=idx, _td=treedef):
                ls = list(_leaves)
                for i, v in zip(_idx, arrs_):
                    ls[i] = v
                return fn(jax.tree_util.tree_unflatten(_td, ls), *a)
            hit = (jax.jit(prog), static)
            self._karg_cache[key] = hit
        j = hit[0]
        return lambda *a, _j=j, _arrs=arrs: _j(_arrs, *a)

    def _build_jits(self, only=None):
        """The jitted functions of the surface half; they close over the statics self.K (compile-time constants).  `only` = names to (re)build (default all).
        D205: after the lake fractions change only 'pre' and 'pa' are rebuilt: the ocean stages (oa, ob: K keys focean, dxypo, oc, valid) and post_b (adv, is_ocean,
        valid_ocean) read no lake fraction, and re-tracing `ob` raises jax UnexpectedTracerError (ocean_ofluxv builds module-level jnp constants (ZE, DZO) at its first
        import, which happens inside the first trace of `ob`; D205 run 1)."""
        K = self.K
        mk = dict(pre=lambda: self._k_arg_jit('pre', lambda K_, S, mi, me, inp: PT.surface_pre_dev(K_, S, mi, me, inp), K),
                  pa=lambda: self._k_arg_jit('pa', lambda K_, S1, mid, acc, srfp, itime, V: PT.post_a(K_, S1, mid, acc, srfp, itime, V), K),
                  oa=lambda: jax.jit(lambda oc, fx, Kb: JO.ocean_stages(K, Kb, oc, fx, which='a')),
                  ob=lambda: self._make_ob(K),
                  pb=lambda: jax.jit(lambda S1, c, oc, V: PT.post_b(K, S1, c, oc, V)))
        for name in (only or tuple(mk)):
            setattr(self, name, mk[name]())
        self.mask_check = jax.jit(self._mask_check)

    def daily_lake_update(self, state, log=print, land_fractions=True):
        """D205: the end-of-day lake update daily_LAKE (LAKES.f:2492; daily_lake.py, bitwise equal to the compiled Fortran on the nov26 case and on 12 stress cases) applied to
        the state at the day boundary (call between the last step of a day and the first of the next, after DAILY_ATMDYN).  ONE declared host round trip: the lake, ice and
        atmosphere-grid lake exports (device -> host), the soil water of the last land step (for DMWLDF, GHY_DRV.f:4054) -> NumPy daily_lake -> device.  Then the lake
        fractions FLAKE/FLAND/FEARTH of the statics change: self.st / the geometry / K / Kb / static181 / melt_geo are rebuilt and the jitted functions and the SURFACE
        stage cache are dropped (recompiled at the next step).  NOT applied (declared, in self.daily_lake_info['not_applied']): the GHY water/heat transfer for the changed
        lake fraction (GHY_DRV.f:4531, uses DMWLDF, DGML, svflake - returned in the info), FSF/TRSURF reset (RESET_SURF_FLUXES, radiation is replayed), MDWNIMP/EDWNIMP
        (daily_LI), the soil-moisture bookkeeping of daily_EARTH.  Returns (new state, info).
        D208: with land_fractions=True (default) the GHY transfer IS applied (GHY_DRV.f:4367 update_land_fractions, daily_land_fractions.py, bitwise equal to the compiled
        Fortran subroutine; vs the real step-48 entry record: scoping/D208_LAND_FRACTIONS_ENTRY.md) to land_prev['dyn_next'] w, ht, fr_snow; the underwater-fraction state
        (ibv = 3, not carried by the land chain) comes from the recorded rows of the last step at the first boundary and from self._land3 afterwards."""
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
        # ---- D208: the land half of the daily lake update (water/heat between the lake and the soil fractions)
        land_new = state['land_prev']
        if land_fractions:
            import daily_land_fractions as DLF
            lp = state['land_prev']
            dyn = lp['dyn_next']
            ii, jj = (ecells % IM).astype(int), (ecells // IM).astype(int)
            wd, hd = np.asarray(dyn['w'], dtype=np.float64), np.asarray(dyn['ht'], dtype=np.float64)
            if self._land3 is None:
                self._land3 = dict(w=np.stack([r[8:29].reshape(7, 3, order='F')[:, 2] for r in rows]), ht=np.stack([r[29:50].reshape(7, 3, order='F')[:, 2] for r in rows]))
            w7 = np.concatenate([wd[:, :7, :2], self._land3['w'][:, :, None]], axis=2)
            h7 = np.concatenate([hd[:, :7, :2], self._land3['ht'][:, :, None]], axis=2)
            dz_r, q_r, fv_r = DLF.rows_from_ffg(rows)
            SL = dict(w=w7, ht=h7, fr_snow=np.asarray(dyn['fr_snow'], dtype=np.float64), dz=dz_r, q=q_r, fv=fv_r,
                      fearth=out['fearth'][ii, jj], flake=out['flake'][ii, jj], svflake=out['svflake'][ii, jj], focean=np.asarray(geo['focean'])[ii, jj],
                      dmwldf=out['dmwldf'][ii, jj], dgml=out['dgml'][ii, jj], byaxyp=1.0 / np.asarray(self.st['axyp'])[ii, jj])
            lout, linfo = DLF.update_land_fractions(SL, GR.THM[0, :])
            self._land3 = dict(w=lout['w'][:, :, 2].copy(), ht=lout['ht'][:, :, 2].copy())
            wn, hn = np.array(wd, copy=True), np.array(hd, copy=True)
            wn[:, :7, :2], hn[:, :7, :2] = lout['w'][:, :, :2], lout['ht'][:, :, :2]
            conv = lambda new, old: jnp.asarray(new, dtype=old.dtype) if hasattr(old, 'devices') else np.asarray(new, dtype=old.dtype)    # noqa: E731
            dyn_n = dict(dyn, w=conv(wn, dyn['w']), ht=conv(hn, dyn['ht']), fr_snow=conv(lout['fr_snow'], dyn['fr_snow']))
            land_new = dict(lp, dyn_next=dyn_n)
            chg = (out['flake'] != out['svflake'])
            rowmask = np.zeros(chg.shape, bool)
            rowmask[ii, jj] = True
            info['land_fractions'] = dict(n_shrunk=linfo['n_shrunk'], n_expanded=linfo['n_expanded'], new_cell_rows=linfo['new_cell_rows'].tolist(),
                                          n_changed_cells_without_land_row=int((chg & ~rowmask & (out['fearth'] > 0) & (np.asarray(geo['focean']) < 1)).sum()))
            info['not_applied'] = [x for x in info['not_applied'] if 'GHY dfrac' not in x] + ['atmlnd fr_snow_rad / set_new_ghy_cells_outputs (radiation replayed)']
        if getattr(self, '_rs', None) is not None:                # D218: RESET_SURF_FLUXES of daily_LAKE on the carried FSF / TRSURF (host RadSurf.daily_lake, once per day)
            import drv_radcols as DR_
            rsn = DR_.RadSurf()
            rsn.FSF, rsn.TRSURF = np.array(self._rs['FSF'], dtype=np.float64), np.array(self._rs['TRSURF'], dtype=np.float64)
            rsn.daily_lake(geo['flake'], out['flake'], ice['rsi'], out['rsi'], fearth0, out['fearth'])
            self._rs = dict(self._rs, FSF=jnp.asarray(rsn.FSF), TRSURF=jnp.asarray(rsn.TRSURF))
            info['radcols_daily_lake_legs'] = rsn.n_legs
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
        new['land_prev'] = land_new
        log(f"  daily_LAKE at the day boundary: {info['n_flake_changed']} cells with FLAKE changed (max |dFLAKE| {info['max_abs_dflake']:.3e}), {info['n_rsi_changed']} RSI changed, "
            f"DMWLDF>0 in {info['n_dmwldf_pos']} cells, counters {info['counters']}"
            + (f"; land fractions: {info['land_fractions']['n_shrunk']} shrunk, {info['land_fractions']['n_expanded']} expanded rows" if 'land_fractions' in info else ''))
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
    def _build_tpl(self, R):
        """Host build of the SURFACE template of one step from its recorded rows (depends only on rec['R'])."""
        reg_fn = self.ph.reg.guard_function('surface_records', A.surface_records, stage='surface')
        recs = reg_fn(R)
        tpl, host = JS.build_template(recs)
        g1 = recs['g1']
        irrig = np.zeros((IM, JM))
        irrig[g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1] = g1[:, 147]
        irrig_act = jnp.asarray(irrig * self.st['fearth'])
        return tpl, host, recs, irrig_act

    # ------------------------------------------------------------------------------------------------ D224: the SURFACE template without per-step records
    def _tpl_own_base(self, tpl, host):
        """Once, at the first step: the slots that are invalid in the first-step template (ice tiles that do not exist yet) get their STATIC columns from
        jax_static.build_static (validated bitwise against the records, D181); everything else of those rows is written by the step's own overrides."""
        import jax_static as JST
        b = JST.build_static(self.date)
        Nw = host['Nw']
        wi, wj = np.asarray(host['wi']), np.asarray(host['wj'])
        out = dict(tpl)
        nsl = []
        for d in tpl['ns']:
            d = dict(d)
            wv = np.asarray(d['wvalid'])
            pw, tw = np.array(d['pw']), np.array(d['tw'])
            for ty in (0, 1):
                inv = np.nonzero(~wv[ty])[0]
                if not len(inv):
                    continue
                rows = ty * Nw + inv
                for arr, fam, cols in ((pw, b['pbl'][ty], b['built']['pbl']), (tw, b['tile'][ty], b['built']['tile'])):
                    src = fam[wi[inv], wj[inv]]
                    for c in cols:
                        arr[rows, c] = src[:, c]
            for ty, el in ((0, 2.5e6), (1, 2.834e6)):          # elhx (pbl col 19) of the water tiles: LHE for open water, LHS for ice (constants)
                inv = np.nonzero(~wv[ty])[0]
                pw[ty * Nw + inv, 19] = el
            d['pw'], d['tw'] = jnp.asarray(pw), jnp.asarray(tw)
            nsl.append(d)
        out['ns'] = nsl
        return out

    def _tpl_own(self, R, it):
        """Template of a step after the first from the first-step base: the slot validity and the type fractions are refreshed from OUR sea-ice state in
        step() (_tpl_own_state); here the recorded land irrigation columns of THIS step (ffg 147, 148: the demand file is not available) are the only record read."""
        tpl, host, recs, _ = self._tpl_base
        if IRRIG_OWN:
            return self._tpl_own_exports(tpl, host), host, recs, jnp.zeros((IM, JM))
        import ghy_compare as GC_
        path = f"{R.ff}/{R.date}/ffg_{it}.bin"
        g = self.ph.reg.read('irrigation_demand', lambda: np.asarray(GC_.load(path)), stage='surface')
        h = len(g) // 2
        gs = (g[:h], g[h:])
        irrig = np.zeros((IM, JM))
        irrig[gs[0][:, 0].astype(int) - 1, gs[0][:, 1].astype(int) - 1] = gs[0][:, 147]
        irrig_act = jnp.asarray(irrig * self.st['fearth'])
        ghy = []
        for k, b in enumerate(tpl['ghy']):
            b = dict(b)
            f = dict(b['forcing'])
            fv = np.asarray(f['fv'])
            pos = fv > 0.0
            f['irrig'] = jnp.asarray(np.where(pos, gs[k][:, 147] / np.where(pos, fv, 1.0), 0.0))
            f['htirrig'] = jnp.asarray(np.where(pos, gs[k][:, 148] / np.where(pos, fv, 1.0), 0.0))
            b['forcing'] = f
            ghy.append(b)
        return self._tpl_own_exports(dict(tpl, ghy=ghy), host), host, recs, irrig_act

    def _irrig_pot(self, it):
        """D225 (ROCKE_IRRIG_OWN=1): potential irrigation (m/s, IRRIGMOD irrig_water_pot) of day `jday` from Irrig72x46_1848to2100_FixedFuture_v3.nc, monthly data with
        timestream 'linm2m' (ent_daily_ff.lai_linm2m), year irrig_yr (default master_yr 1850, ROCKE_IRRIG_YR), negatives set to 0.  Returns (IM,JM) float64 and the
        Ent fv grid (conditioned as GHY)."""
        import netCDF4 as nc_
        import ent_daily_ff as ED_
        import jax_static as JST_
        if getattr(self, '_irr_q', None) is None:
            yr = int(os.environ.get('ROCKE_IRRIG_YR', '1850'))
            f_ = os.path.join(JST_.PROD, 'Irrig72x46_1848to2100_FixedFuture_v3.nc')
            q = np.array(nc_.Dataset(f_).variables['irrigation_per_m2'][:], dtype=np.float64)       # REAL*8 as read_stream holds it
            self._irr_q = q[(yr - 1848) * 12:(yr - 1848) * 12 + 12]
            es = JST_.ent_static(self.date)
            fv = np.where(es['fv'] < 1e-6, 0.0, es['fv'])
            self._irr_fv = np.where(fv > 1.0 - 1e-6, 1.0, fv)
            self._irr_cache = {}
        jday = (it // 48) % 365 + 1
        if jday not in self._irr_cache:
            self._irr_cache = {jday: np.maximum(ED_.lai_linm2m(self._irr_q, jday), 0.0).T.copy()}
        return self._irr_cache[jday], self._irr_fv

    def _tpl_own_state(self, tpl, host, rsi):
        """Slot validity and type fractions of both substeps from OUR RSI (jax_state_d181.tile_ptype: [(1-RSI) FWATER, RSI FWATER, FLICE, FEARTH])."""
        p = S181.tile_ptype(self.sp181, rsi, jnp)                      # (4, IM, JM)
        wi, wj = jnp.asarray(host['wi']), jnp.asarray(host['wj'])
        wv = jnp.stack([p[0][wi, wj] > 0.0, p[1][wi, wj] > 0.0])
        ft = jnp.stack([p[t].T.reshape(-1) for t in range(4)], axis=1)
        nsl = [dict(d, wvalid=wv, ftype=ft) for d in tpl['ns']]
        return dict(tpl, ns=nsl)

    def _tpl_own_exports(self, tpl, host, refresh=False):
        """ws_can, shc_can and height (snowm = 0.1 height) of the Ent cells into the GHY static leaves.  They change only at the daily Ent update: the arrays are
        taken from ent_jax.REG['exports'] there (refresh=True) and applied to every later step's template."""
        import ent_jax as EJ_
        if refresh:
            ex = EJ_.REG.get('exports')
            if ex:
                keys = [(int(i_) + 1, int(j_) + 1) for i_, j_ in zip(host['ei'], host['ej'])]
                self._own_ex = tuple(jnp.asarray(np.array([ex[k][n] for k in keys]) * m_) for n, m_ in (('ws_can', 1.0), ('shc_can', 1.0), ('height', 0.1)))
        if getattr(self, '_own_ex', None) is None:
            return tpl
        ws, shc, sm = self._own_ex
        ghy = []
        for b in tpl['ghy']:
            b = dict(b)
            st = dict(b['static'])
            st['ws'] = st['ws'].at[:, 0, 1].set(ws)
            st['shc'] = st['shc'].at[:, 0, 1].set(shc)
            b['static'] = st
            b['snowm'] = sm
            ghy.append(b)
        return dict(tpl, ghy=ghy)

    def prefetch_template(self, rec_next):
        """D214 (replay scaffolding): start building the template of the NEXT step in a background thread; step() picks it up by the identity of rec['R']."""
        import concurrent.futures as cf
        if not hasattr(self, '_tpl_pool'):
            self._tpl_pool = cf.ThreadPoolExecutor(max_workers=1)
        R = rec_next['R']
        self._tpl_prefetch[id(R)] = self._tpl_pool.submit(self._build_tpl, R)

    def _own_cse_in(self, it, dev):
        """D221 (ROCKE_CSE_IN_OWN=1): the three CONDSE entry fields that the device CONDSE reads and the chain does not produce, from our own state:
        FLAND/FEARTH from the statics (updated by daily_LAKE), RNDSS from the LCG chain of the first entry seed (drv_rng, D174).  With
        ROCKE_CSE_IN_OWN_CHECK=1 the recorded values are still read and the differences are logged in self.cse_in_own_log."""
        import drv_rng
        s0 = self.__dict__.setdefault('_cse_s0', {})
        if not s0:
            s0[self.it0] = int(round(float(np.asarray(dev['cse_in']['SEEDS'])[0]))) & 0xFFFFFFFF
        last = max(s0)
        while last < it:
            s0[last + 1] = drv_rng.next_seed(s0[last], bool(A.is_radiation_step(last)))
            last += 1
        ctx = self.ph.ctx
        rnd = drv_rng.rndss(s0[it], ctx.imaxj, int(ctx.cfg['lmcld']))
        own = dict(FLAND=np.asarray(self.st['fland'], dtype=np.float64), FEARTH=np.asarray(self.st['fearth'], dtype=np.float64), RNDSS=rnd)
        if P1.CSE_IN_OWN_CHECK:
            mask = np.zeros(rnd.shape, bool)
            for j in range(rnd.shape[3]):
                mask[:, :int(ctx.cfg['lmcld']), :int(ctx.imaxj[j]), j] = True
            rec_ = {k_: np.asarray(dev['cse_in'][k_]) for k_ in own}
            lg = dict(itime=int(it), FLAND=float(np.abs(own['FLAND'] - rec_['FLAND']).max()), FEARTH=float(np.abs(own['FEARTH'] - rec_['FEARTH']).max()),
                      RNDSS=float(np.abs(own['RNDSS'] - rec_['RNDSS'])[mask].max()))
            self.__dict__.setdefault('cse_in_own_log', []).append(lg)
            print('CSE_IN_OWN_CHECK ' + repr(lg), flush=True)
        for k_, v in own.items():
            dev['cse_in'][k_] = jnp.asarray(v)

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
            pf = self._tpl_prefetch.pop(id(R), None)
            if pf is not None:                                   # D214: built for this step in a background thread while the previous step ran (identical call, identical result)
                tpl, host, recs, irrig_act = pf.result()
            elif TPL_OWN and getattr(self, '_tpl_base', None) is not None:
                tpl, host, recs, irrig_act = self._tpl_own(R, it)
            else:
                tpl, host, recs, irrig_act = self._build_tpl(R)
                if TPL_OWN:                       # D224: the first-step template is the base of all later steps
                    tpl = self._tpl_own_base(tpl, host)
                    self._tpl_base = (tpl, host, recs, irrig_act)
            if not (TPL_OWN and getattr(self, '_tpl_last_ghy', None) is not None):
                self._last_ghy = (host['nit_rows'][1], host['ecells'])
                self._tpl_last_ghy = self._last_ghy
            fin(tpl)
        if self._stage_cache is None or not self._same_host(self._stage_cache['host'], host):
            if os.environ.get('ROCKE_STAGE_ARGS', '0') == '1':       # D217: cell index arrays as jit arguments (no recompile when daily_LAKE changes them)
                _cells = {k: jnp.asarray(host[k]) for k in ('wcells', 'lcells', 'ecells')}
                _ak = (host['Nw'], tuple(_cells[k].shape for k in sorted(_cells)))
                if not hasattr(self, '_accj_cache'):
                    self._accj_cache = {}
                if _ak not in self._accj_cache:                  # compiled once, reused after daily_LAKE rebuilds
                    self._accj_cache[_ak] = jax.jit(lambda aux_, tpl_, c_, _h=host: PT.tile_accumulators_dev(_h, tpl_, aux_, c_))
                _accj = self._accj_cache[_ak]
                _accf = lambda aux_, tpl_, _j=_accj, _c=_cells: _j(aux_, tpl_, _c)
            else:
                _accf = jax.jit(lambda aux_, tpl_, _h=host: PT.tile_accumulators_dev(_h, tpl_, aux_))
            self._stage_cache = dict(host=host, stage=JS.make_stage(host), apply=TS.make_apply_state(host, self.K), accf=_accf)
            info['stage_rebuilt'] = True
        stg, apply = self._stage_cache['stage'], self._stage_cache['apply']
        # ---------------------------------------------------------------- J1: phase 1
        with stage('melt_si_dev'):
            ice_new, melt = M.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], self.melt_geo, 1800.0)
            dev['cse_in']['RSI'] = ice_new['rsi']                       # as surface_loop.Loop.pre_cse: our RSI replaces the recorded CONDSE-entry RSI
            if TPL_OWN:
                tpl = self._tpl_own_state(tpl, host, ice_new['rsi'])
            if P1.CSE_IN_OWN:
                self._own_cse_in(it, dev)
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
        # ---------------------------------------------------------------- D218: radiation-derived template columns from OUR radiation state (ROCKE_RADCOLS=1, server mode)
        use_rc = os.environ.get('ROCKE_RADCOLS', '0') == '1' and server is not None
        if use_rc:
            with stage('radcols'):
                geo_ = self.st['geo']
                if A.is_radiation_step(it):
                    assert DD.LAST.get('it') == it, f"radiation server output of step {it} not captured (last {DD.LAST.get('it')})"
                    self._rs = DD.rs_from_server(DD.LAST['out'])
                    cz_ = jnp.asarray(np.asarray(DD.LAST['out']['COSZ1'], dtype=np.float64))
                else:
                    assert getattr(self, '_rs', None) is not None, 'ROCKE_RADCOLS needs a radiation step first'
                    self._rs = DD.ice_leg(self._rs, SS['ice']['rsi'], ice_new['rsi'], jnp.asarray((np.asarray(geo_['focean']) > 0) | (np.asarray(geo_['flake']) > 0)))   # leg (a)
                    cz_ = jnp.asarray(np.asarray(rec['rad']['COSZ1'], dtype=np.float64))
                tpl_new = DD.fill_tpl(tpl, self._rs, cz_, host)
                if os.environ.get('ROCKE_RADCOLS_CHECK', '0') == '1':            # diagnostic: computed columns against the RECORDED template (host read, slow)
                    lo, ln = jax.tree_util.tree_flatten_with_path(tpl)[0], jax.tree_util.tree_leaves(tpl_new)
                    chk = {}
                    for (pth, a), b in zip(lo, ln):
                        a_, b_ = np.asarray(a), np.asarray(b)
                        if a_.dtype.kind == 'f' and not np.array_equal(a_, b_, equal_nan=True):
                            chk[jax.tree_util.keystr(pth)] = (int((a_ != b_).sum()), float(np.nanmax(np.abs(a_ - b_))))
                    info['radcols_check'] = chk
                    print(f'  radcols_check it {it}: {len(chk)} leaves differ from the recorded template: ' + '; '.join(f'{k}: n={v[0]} max={v[1]:.3g}' for k, v in chk.items()), flush=True)
                tpl = tpl_new
                fin(tpl)
            info['radcols'] = True
        # ---------------------------------------------------------------- J2: surface
        prec, eprec, precss = S1a['PREC'], S1a['EPREC'], S1a['PRECSS']
        with stage('surface_pre'):
            if IRRIG_OWN:
                pot_, fvg_ = self._irrig_pot(it)
                fe_ = np.asarray(self.st['fearth'])
                irrig_act = jnp.asarray(np.where((pot_ > 1e-30) & (fe_ > 1e-30) & (fvg_ > 1e-30), pot_, 0.0))       # irrig_water_pot where IRRIGMOD acts
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
            if self.nit_fix and not TPL_OWN:         # D224: the recorded iteration counts of the step are not read
                tpl, n_nit, n_d2h = JS.nit_rebuild(tpl, host, prec, eprec, precss, strict=self.nit_strict)
                fin(tpl)
            info['nit_rebuild'] = dict(cells=n_nit, device_to_host_arrays=n_d2h)
        eod_ = (it % 48 == 0) if TPL_OWN else bool(host['nit_rows'][0][0, 163] == 1.0)       # end-of-day flag of the records == itime % 48 == 0 (steps 33312, 33360, 33408)
        if os.environ.get('ROCKE_ENT_LIVE', '0') == '1' and state['land_prev'] is not None and 'ent_state' in state['land_prev']['dyn_next'] and eod_:
            with stage('ent_daily'):
                import ent_jax as EJ_
                lp_ = state['land_prev']
                dn_ = lp_['dyn_next']
                keys_ = [(int(i_) + 1, int(j_) + 1) for i_, j_ in zip(host['ei'], host['ej'])]
                jday_ = (it // 48) % 365 + 1
                ec_, es_ = EJ_.daily_update(keys_, dn_['ent_const'], dn_['ent_state'], jday_)
                state = dict(state, land_prev=dict(lp_, dyn_next=dict(dn_, ent_const=ec_, ent_state=es_)))
                info['ent_daily'] = int(jday_)
                if TPL_OWN:
                    tpl = self._tpl_own_exports(tpl, host, refresh=True)
        if TPL_OWN and os.environ.get('ROCKE_TPL_OWN_CHECK', '0') == '1' and k > 0:       # diagnostic: own template against the RECORDED one of this step (host read, slow)
            with contextlib.nullcontext():
                rt, _, _, _ = self._build_tpl(R)
                fl = {jax.tree_util.keystr(p_): np.asarray(x) for p_, x in jax.tree_util.tree_flatten_with_path(tpl)[0]}
                fr = {jax.tree_util.keystr(p_): np.asarray(x) for p_, x in jax.tree_util.tree_flatten_with_path(rt)[0]}
                bad = {}
                for n_, a_ in fl.items():
                    b_ = fr.get(n_)
                    if b_ is None or a_.shape != b_.shape or a_.dtype.kind != 'f':
                        continue
                    if "['ns'][" in n_ and n_.endswith("['pw']") or n_.endswith("['tw']"):
                        wv_ = np.asarray(tpl['ns'][int(n_[7])]['wvalid']).reshape(-1)
                        a_, b_ = a_[wv_], b_[wv_]
                    d_ = np.abs(a_ - b_)
                    if d_.size and d_.max() > 0:
                        bad[n_] = (float(d_.max()), int((d_ > 0).sum()))
                imp = {n_: v_ for n_, v_ in bad.items() if any(t_ in n_ for t_ in ('wvalid', 'ftype', "['ws']", "['shc']", 'snowm', 'irrig', "['fb']", "['fv']", 'geothermal'))}
                print(f'  tpl_own_check it {it}: {len(bad)} float leaves differ from the recorded template; static/forcing: ' + '; '.join(f'{n_}: max {v_[0]:.3g} n {v_[1]}' for n_, v_ in imp.items()), flush=True)
        if IRRIG_OWN:                      # actual irrigation (water, energy) per land fraction -> GHY forcing irrig/htirrig of the vegetated tile (GHY_DRV.f:1223, GHY.f:2230)
            m2kg_ = RHOW_ * jnp.asarray(self.st['axyp'])
            mi_, gi_ = mid['irr']['mwl_to_irrig'], mid['irr']['gml_to_irrig']
            pot_g_ = irrig_act
            full_ = (mi_ >= pot_g_ * m2kg_ * DTSRC_) & (pot_g_ > 0)
            act_ = jnp.where(full_, pot_g_, mi_ / (m2kg_ * DTSRC_))
            en_ = gi_ * RHOW_ / (m2kg_ * DTSRC_)
            fe_ = jnp.asarray(self.st['fearth'])
            ej_, ei_ = jnp.asarray(host['ej']), jnp.asarray(host['ei'])
            ghy_ = []
            for b_ in tpl['ghy']:
                b_ = dict(b_)
                f_ = dict(b_['forcing'])
                fvr_ = f_['fv']
                pos_ = fvr_ > 0.0
                f_['irrig'] = jnp.where(pos_, (act_[ei_, ej_] / fe_[ei_, ej_]) / jnp.where(pos_, fvr_, 1.0), 0.0)
                f_['htirrig'] = jnp.where(pos_, (en_[ei_, ej_] / fe_[ei_, ej_]) / jnp.where(pos_, fvr_, 1.0), 0.0)
                b_['forcing'] = f_
                ghy_.append(b_)
            tpl = dict(tpl, ghy=ghy_)
        with stage('template_apply'):
            lp = state['land_prev']
            tpl_m = apply(tpl, S1s, mid['ag'], S1a['PEDN'][0], prec, eprec, precss, None if lp is None else lp['dyn_next'], lp)
            pcarry_b = None
            if os.environ.get('ROCKE_PBL_CARRY', '0') == '1':      # D222: substep-1 PBL columns of this step from OUR substep-2 outputs of the previous step
                import jax_pbl_carry as PC_
                ix_ = PC_.index_arrays(host, tpl_m['ns'][0]['wvalid'])
                pcarry_b = PC_.loadbl(state['pbl_carry']) if state.get('pbl_carry') is not None else PC_.empty()
                if state.get('pbl_carry') is not None:
                    n0 = dict(tpl_m['ns'][0])
                    npw, npl, npe = PC_.fill_template(n0['pw'], n0['pl'], n0['pe'], pcarry_b, ix_)
                    if os.environ.get('ROCKE_PBL_CARRY_CHECK', '0') == '1':      # diagnostic: carried columns against the RECORDED template (host read)
                        cols = np.r_[33:36, 50:89]
                        wv_ = np.asarray(ix_['wvalid2'])
                        chk = {}
                        for nm_, new_, old_, v_ in (('pw', npw, n0['pw'], wv_), ('pl', npl, n0['pl'], None), ('pe', npe, n0['pe'], None)):
                            a_, b_ = np.asarray(new_)[:, cols], np.asarray(old_)[:, cols]
                            if v_ is not None:
                                a_, b_ = a_[v_], b_[v_]
                            chk[nm_] = (float(np.abs(a_ - b_).max()) if a_.size else 0.0, int((a_ != b_).sum()), int(a_.size))
                        print(f'  pbl_carry_check it {it}: ' + '; '.join(f'{k}: max {v[0]:.3g} n_unequal {v[1]}/{v[2]}' for k, v in chk.items()), flush=True)
                    n0['pw'], n0['pl'], n0['pe'] = npw, npl, npe
                    tpl_m = dict(tpl_m, ns=[n0, tpl_m['ns'][1]])
            fin(tpl_m)
            if TPL_OWN and os.environ.get('ROCKE_TPL_OWN_CHECK2', '0') == '1' and k > 0 and k < 4:      # diagnostic: the same pipeline on the RECORDED template, compared leaf by leaf after apply
                rt_, _, _, _ = self._build_tpl(R)
                if use_rc:
                    rt_ = DD.fill_tpl(rt_, self._rs, cz_, host)
                rt_m = apply(rt_, S1s, mid['ag'], S1a['PEDN'][0], prec, eprec, precss, None if lp is None else lp['dyn_next'], lp)
                if pcarry_b is not None and state.get('pbl_carry') is not None:
                    r0 = dict(rt_m['ns'][0])
                    r0['pw'], r0['pl'], r0['pe'] = PC_.fill_template(r0['pw'], r0['pl'], r0['pe'], pcarry_b, ix_)
                    rt_m = dict(rt_m, ns=[r0, rt_m['ns'][1]])
                A_ = {jax.tree_util.keystr(p_): np.asarray(x) for p_, x in jax.tree_util.tree_flatten_with_path(tpl_m)[0]}
                B_ = {jax.tree_util.keystr(p_): np.asarray(x) for p_, x in jax.tree_util.tree_flatten_with_path(rt_m)[0]}
                out_ = []
                for n_, a_ in A_.items():
                    b_ = B_.get(n_)
                    if b_ is None or a_.shape != b_.shape or a_.dtype.kind != 'f':
                        continue
                    if n_.endswith("['pw']") or n_.endswith("['tw']"):
                        v_ = np.asarray(tpl_m['ns'][int(n_[7])]['wvalid']).reshape(-1)
                        a_, b_ = a_[v_], b_[v_]
                    if a_.ndim == 2 and a_.size:
                        d_ = np.abs(a_ - b_)
                        cs_ = np.nonzero(d_.max(0) > 0)[0]
                        if len(cs_):
                            out_.append((n_, [(int(c_), float(d_[:, c_].max())) for c_ in cs_][:12]))
                    elif a_.size and np.abs(a_ - b_).max() > 0:
                        out_.append((n_, float(np.abs(a_ - b_).max())))
                print(f'  tpl_own_check2 it {it}: ' + ' | '.join(f'{n_}: {v_}' for n_, v_ in out_), flush=True)
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
        if use_rc:
            with stage('radcols_legs'):                          # legs (b), (c) ocean cells and (F) lake cells; they act on the NEXT step's columns
                geo_ = self.st['geo']
                ocn_, lake_ = jnp.asarray(np.asarray(geo_['focean']) > 0), jnp.asarray(np.asarray(geo_['flake']) > 0)
                s1_, pre_, gsi_, end_ = S1s['ice']['rsi'], post['ice_pre_adv']['rsi'], post['ice_after_gsi']['rsi'], SSn['ice']['rsi']
                rs_ = DD.ice_leg(self._rs, s1_, pre_, ocn_)
                rs_ = DD.ice_leg(rs_, pre_, end_, ocn_)
                self._rs = DD.ice_leg(rs_, s1_, gsi_, lake_)
                fin(self._rs['FSF'])
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
            if pcarry_b is not None:
                r2_ = aux['r2']
                new['pbl_carry'] = PC_.update(pcarry_b, r2_['pbl'], r2_['pbl_li'], r2_['pbl_land'], ix_)
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
