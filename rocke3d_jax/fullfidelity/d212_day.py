"""D212 driver (copy of d210_day.py; adds --surf-fields recorded|computed, --seed recorded|computed, --cosz recorded|computed: the three recorded inputs of the radiation packet of D210
built from OUR state/functions. surf-fields computed: the 21 surface fields of the packet from our ice/lake/land-ice state (jax_radiation.surface_fields_jax after our MELT_SI and seaice_to_atmgrid),
our carried GHY state of the last substep (drv_radpacket.land_fields; snowbv carried from the restart), our composite TSAVG and WSAVG (sum ftype*ws of the last PBL substep); seed computed: the D174 LCG chain
(drv_rng) from the recorded SEEDS[0] of the FIRST step only; cosz computed: drv_zenith.Zenith (libimf) for every step. Each switch is independent. Not computed here: see scoping/D212_RADIATION_PACKET_OWN_STATE_ENTRY.md.
--check-packet 1: in replay mode also compute the surface fields every radiation step and compare with the recorded live packet (diagnostic, not used). D210 text follows:  (copy of d209_day.py; adds --rad replay|server: with server, the radiation of every radiation step (itime-ITIMEI)%5==0 is COMPUTED by the real persistent Fortran radiation server (radiation_server_persist.PersistentServer; real RADIA/SOCRATES, unmodified; "radiation computed by the original Fortran (hybrid)"), the other 4 of 5 steps apply the HELD server SRHR/TRHR with the recorded per-step COSZ1 (as RADIA does). Server packet: atmosphere, clouds, SNOAGE from OUR state; RQT/KLIQ from the server hold (first call: live packet of step 0); the 21 SURFACE fields and the seed are RECORDED (live packets rsv_n26_<it>_in.bin, recorded SEEDS). Per call the packet/outputs are compared with the recorded real packet/output (d212_server_calls.json). --rad replay = D209 behaviour. D209 text follows:  (copy of d208_day.py; adds --daily-consts computed|recorded: day-boundary MDRYA/ch4ox/SNOAGE computed by drv_daily.daily_on_state_computed (default) or read from records (D208 behaviour). D208 text follows: HEAD abcb0eb, land-fraction transfer in jax_coupled.daily_lake_update, flag --land-fractions, default 1; also saves the carried GHY state after steps 44-53 and before/after the boundary transfer): d193_day.main on the assembled step with the D194 nit rule merged (Coupled(nit_fix=True)), the nit assertion
REPORTED not raised (--nit-strict 0), radiation REPLAYED from the records, and at the day boundary (step 48, it 33360), after DAILY_ATMDYN + ch4ox + SNOAGE of d193_day.daily_on_state,
the Fortran daily_LAKE (daily_lake.py via jax_coupled.Coupled.daily_lake_update) when --daily-lake 1 (default 1).  --daily-lake 0 reproduces d203_day.py.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=<scratch> D189_SHIM_DIR=<scratch> python d209_day.py OUTDIR [--c1dir DIR] [--nit-strict 0|1] [--daily-lake 0|1] [--land-fractions 0|1] [--daily-consts computed|recorded] [--nsteps N]
With --c1dir the FULL C1 arrays of every step are saved (about 24 GB; D209 change vs d205_day.py).
Writes OUTDIR/d212_nit.json (rebuild log, nit mismatches) and OUTDIR/d212_lake.json + d212_lake_hook.npz (the hook's info and arrays)."""
import clouds_jax_env  # noqa: F401
import functools
import json
import os
import sys

import numpy as np

import jax_coupled as C
import jax_surface as JS
import jax.numpy as jnp
import jax_p1_melt as M
import d193_day as D
import d187_common as K
import d191_run as R191

def rs_FF():
    import radiation_server as rs
    return rs.FF


SNO_DRIFT = []
OWN = dict(surf='computed', seed='computed', cosz='computed', check=False, log=[], prev=None, lf=None, sbv=None, wsavg=None, thets_dz=None, rest=None, seeds={}, zen=None, s0_first=None)
HOOK = dict(cp=None, on=True, info=None, land=True, outdir=None)


RAD = dict(mode='replay', srv=None, wrap=None, lv=None, calls=[], cpus=(10, 11), save_dir=None)


class SrvWrap:
    """Forwards request() to the persistent server; compares the packet we send with the recorded real packet of that step and the server output
    with the recorded real server output (rsv_n26_<it>_{in,out}.bin), keeps the output fields for saving."""

    def __init__(self, srv):
        self.srv = srv
        self.binary, self.restart, self.rundir = srv.binary, getattr(srv, 'restart', None), srv.rundir

    def request(self, state, itime, seed=None):
        import radiation_server as rs
        import jax_harness as H
        out = self.srv.request(state, itime, seed)
        d = f'{rs.FF}/nov26_day'
        rin = rs.read_packet(f'{d}/rsv_n26_{itime}_in.bin')
        rout = rs.read_packet(f'{d}/rsv_n26_{itime}_out.bin')
        rec = dict(itime=int(itime), seed=int(seed), server_s=float(out['_server_s']), wall_s=float(out['_wall_s']))
        rec['packet_vs_recorded_packet'] = {k: K.slim(H.field_category(np.asarray(state[k]), rin[k])) for k in rs.INPUT_FIELDS}
        rec['output_vs_recorded_output'] = {k: K.slim(H.field_category(np.asarray(out[k]), rout[k])) for k in rs.OUTPUT_FIELDS if k != 'AIJ' and k in out and k in rout}
        RAD['calls'].append(rec)
        if RAD['save_dir']:
            np.savez(os.path.join(RAD['save_dir'], f'server_out_{itime}.npz'), **{k: np.asarray(out[k]) for k in rs.OUTPUT_FIELDS if k != 'AIJ'},
                     **{'IN_' + k: np.asarray(state[k]) for k in ('T', 'Q', 'SNOAGE', 'RQT', 'CLDSS', 'CLDMC', 'TAUSS', 'TAUMC')})
        return out


def _affine_child():
    import ctypes, signal
    try:
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:
        pass
    try:
        os.sched_setaffinity(0, set(RAD['cpus']))
    except Exception:
        pass



_orig_make_stage = JS.make_stage


def _make_stage_capture(host):
    stg = _orig_make_stage(host)

    def wrapped(Sd, tpl_m, *a, **k):
        out, aux = stg(Sd, tpl_m, *a, **k)
        OWN['prev'] = dict(aux=aux, tpl=tpl_m, host=host)       # kept only to build the next radiation packet from OUR substep-2 PBL/GHY outputs
        return out, aux
    wrapped.units = getattr(stg, 'units', None)
    return wrapped


def _signed32(u):
    u = int(u) % (1 << 32)
    return u - (1 << 32) if u >= (1 << 31) else u


class CoupledD209(C.Coupled):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        HOOK['cp'] = self
        self._wrap_records()
        if RAD['mode'] == 'server':
            import radiation_server as rs
            import radiation_server_persist as RP
            RP._pdeathsig = _affine_child                    # the server child runs on its own cores (RAD['cpus']) and still dies with this process
            srv = RP.PersistentServer('nov26').start()
            RAD['srv'] = srv
            RAD['wrap'] = SrvWrap(srv)
            RAD['lv'] = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{self.it0}_in.bin')
            ph = self.ph
            ph.rad_mode, ph.server, ph.live_packet = 'server', RAD['wrap'], RAD['lv']
            for att in ('_handoff', '_hstep'):
                if hasattr(ph, att):
                    delattr(ph, att)


    # ------------------------------------------------------------------------------------------------ D212: seed and COSZ1 from our own functions
    def _wrap_records(self):
        import drv_rng
        orig = self.ph.load_records
        it0 = self.it0

        def seed_for(itime, rec):
            if not OWN['seeds']:
                OWN['s0_first'] = int(round(float(rec['R'].cse_out['SEEDS'][0]))) % drv_rng.M       # the ONE recorded value: SEEDS[0] of the first step
                OWN['seeds']['s0'] = {it0: OWN['s0_first']}
            s0 = OWN['seeds']['s0']
            i = max(s0)
            while i < itime:                                   # advance the chain to this step (steps are visited in order)
                s0[i + 1] = drv_rng.next_seed(s0[i], A_is_rad(i))
                i += 1
            return _signed32(drv_rng.radia_seed(s0[itime]))

        def A_is_rad(i):
            import atm_step as A
            return A.is_radiation_step(i)

        def lr(itime, first=True):
            rec = orig(itime, first)
            rec['seed_recorded'] = rec['seed']
            if OWN['seed'] == 'computed':
                rec['seed'] = seed_for(itime, rec)
            rec['rad']['cosz1_recorded'] = rec['rad']['COSZ1']
            if OWN['cosz'] == 'computed':
                if OWN['zen'] is None:
                    import drv_zenith
                    OWN['zen'] = drv_zenith.Zenith()
                rec['rad']['COSZ1'] = np.ascontiguousarray(OWN['zen'].cosz1(itime), dtype=np.float64)
            OWN['log'].append(dict(kind='seed_cosz', itime=int(itime), seed_used=int(rec['seed']), seed_recorded=int(rec['seed_recorded']), seed_equal=bool(rec['seed'] == rec['seed_recorded']),
                                   cosz_max_abs_vs_recorded=float(np.abs(rec['rad']['COSZ1'] - rec['rad']['cosz1_recorded']).max()),
                                   cosz_n_unequal=int((rec['rad']['COSZ1'] != rec['rad']['cosz1_recorded']).sum())))
            del rec['rad']['cosz1_recorded']
            return rec
        self.ph.load_records = lr

    # ------------------------------------------------------------------------------------------------ D212: the 21 surface fields of the packet from OUR state
    def _static_land_rows(self, it):
        import ghy_compare as GC
        g = GC.load(f"{rs_FF()}/nov26_day/ffg_{it}.bin")          # static columns of the land rows only (cell index, top_dev, fv, soil), see the entry
        return g[len(g) // 2:]

    def _land_update(self, it, new_state):
        """After step `it`: land group of the NEXT packet (GTEMPR4 BARESW SNOWD FRSNOW) from OUR carried GHY state of the last substep, snowbv carried; WSAVG of the next packet."""
        import drv_radpacket as RP
        pv = OWN['prev']
        host, aux, tpl = pv['host'], pv['aux'], pv['tpl']
        lp = new_state['land_prev']
        g2 = self._static_land_rows(it)
        cells = (g2[:, 1].astype(int) - 1) * 72 + (g2[:, 0].astype(int) - 1)
        assert np.array_equal(cells, host['ecells']), 'land row order of the static rows differs from the template cells'
        dn = lp['dyn_next']
        tbcs = np.asarray(lp['ghy']['tbcs'], dtype=np.float64)
        arr = lambda k: np.asarray(dn[k], dtype=np.float64)
        lf, sbv = RP.land_fields(g2, tbcs, arr('w'), np.asarray(dn['nsn']).astype(int), arr('dzsn'), arr('wsn'), arr('fr_snow'), OWN['sbv'], thets_dz=OWN['thets_dz'])
        OWN['lf'], OWN['sbv'] = lf, sbv
        # WSAVG: sum over the four types of ftype*ws of the last (second) PBL substep (PBL_DRV.f:420)
        r2 = aux['r2']
        Nw = host['Nw']
        ft = np.asarray(tpl['ns'][1]['ftype'])
        wv = np.asarray(tpl['ns'][1]['wvalid'])
        patch = np.zeros((72 * 46, 4))
        ws_w = np.asarray(r2['pbl']['ws'])
        patch[host['wcells'], 0] = np.where(wv[0], ws_w[:Nw], 0.0)
        patch[host['wcells'], 1] = np.where(wv[1], ws_w[Nw:], 0.0)
        patch[host['lcells'], 2] = np.asarray(r2['pbl_li']['ws'])
        patch[host['ecells'], 3] = np.asarray(r2['pbl_land']['ws'])
        comp = (patch * ft).sum(axis=1)
        w = np.zeros((72, 46))
        w[host['bi'], host['bj']] = comp[host['bcells']]
        OWN['wsavg'] = w

    def _land_restart(self):
        import drv_radpacket as RP
        g = self._static_land_rows_first()
        ra = RP.restart_land_arrays(f"{rs_FF()}/_pristine_restarts/fort1_{self.date}_itime{self.it0}.nc", g)
        OWN['thets_dz'] = RP._static_baresw(g)
        lf, _ = RP.land_fields(g, ra['tbcs'], ra['w'], ra['nsn'], ra['dzsn'], ra['wsn'], ra['fr_snow'], ra['snowbv'], thets_dz=OWN['thets_dz'], update_snowbv=False)
        OWN['sbv'] = np.array(ra['snowbv'], float)
        lm = np.zeros((72, 46), bool); lm[g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1] = True
        fvv = np.zeros((72, 46)); fvv[g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1] = np.where(g[:, 169] < 1e-6, 0.0, np.where(g[:, 169] > 1 - 1e-6, 1.0, g[:, 169]))
        OWN['land_mask'], OWN['fv'] = lm, fvv
        OWN['rest'] = dict(lf=lf, tsavg=ra['tsavg'], wsavg=ra['wsavg'])

    def _static_land_rows_first(self):
        import ghy_compare as GC
        g = GC.load(f"{rs_FF()}/nov26_day/ffg_{self.it0}.bin")
        return g[:len(g) // 2]

    def own_surface(self, k, it, state):
        """The 21 fields from OUR state at the beginning of step k (= end of step k-1) + MELT_SI of step k."""
        import jax_radiation as JR
        import jax_seaice_lake as SLk
        SS = state['surf']
        ice_new, _m = M.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], self.melt_geo, 1800.0)
        ag = SLk.seaice_to_atmgrid(self.K, ice_new)
        geo, st = self.st['geo'], self.st
        jf = JR.surface_fields_jax(ice_new['rsi'], ice_new['snowi'], ice_new['pond_melt'], ice_new['flag_dsws'], ag['zsi'], ag['zsnowi'], ag['gtempr'],
                                   jnp.asarray(geo['fwater']), jnp.asarray(geo['flake']), SS['lake']['mwl'], jnp.asarray(st['axyp']), jnp.asarray(st['flice']),
                                   jnp.asarray(st['fland']), jnp.asarray(st['fearth']), SS['atm']['gtempr'], SS['li']['tlandi'][..., 0], SS['li']['snowli'])
        out = {kk: np.asarray(v, dtype=np.float64) for kk, v in jf.items()}
        if k == 0:
            if OWN['rest'] is None:
                self._land_restart()
            lf, tsavg, wsavg = OWN['rest']['lf'], OWN['rest']['tsavg'], OWN['rest']['wsavg']
        else:
            lf, tsavg, wsavg = OWN['lf'], np.asarray(state['S']['TSAVG'], dtype=np.float64), OWN['wsavg']
        out.update({kk: np.asarray(lf[kk], dtype=np.float64) for kk in ('GTEMPR4', 'BARESW', 'SNOWD', 'FRSNOW')})
        out['TSAVG'], out['WSAVG'] = np.asarray(tsavg, dtype=np.float64), np.asarray(wsavg, dtype=np.float64)
        return out

    def _compare_surface(self, it, own, save_dir=None):
        """Own packet surface fields vs the recorded live packet.  Masks as D176 (fields whose value is unused/undefined outside their domain are compared only inside it):
        ZSI ZSNOWI GTEMPR2 where ice; GTEMPR1 where water; GTEMPR3 SNOWLI where land ice; land group on land rows; FRSNOW bare/veg by fv; poles: only the first longitude."""
        import radiation_server as rs
        import drv_radpacket as RP
        import d212_inputs_check as IC
        lv = rs.read_packet(f"{rs.FF}/nov26_day/rsv_n26_{it}_in.bin")
        ok = ~RP.pole_mask()
        geo, st = self.st['geo'], self.st
        poice = (np.asarray(lv['RSI']) * np.asarray(geo['fwater']) > 0) & ok
        fw = (np.asarray(geo['fwater']) > 0) & ok
        fli = (np.asarray(st['flice']) > 0) & ok
        land = OWN['land_mask'] & ok
        fv = OWN['fv']
        sel = dict(RSI=ok, SNOWI=ok, POND_MELT=ok, FLAG_DSWS=ok, ZSI=poice, ZSNOWI=poice, GTEMPR2=poice, FLAKE=ok, DLAKE=ok, FLICE=ok, FLAND=ok, FEARTH=ok, GTEMPR1=fw,
                   GTEMPR3=fli, SNOWLI=fli, GTEMPR4=land, BARESW=land, TSAVG=ok, WSAVG=ok)
        rows = {}
        for kk in RP.FIELDS_SURFACE:
            a, b = np.asarray(own[kk]), np.asarray(lv[kk])
            if kk in ('SNOWD', 'FRSNOW'):
                for ib in (0, 1):
                    m = land & ((fv < 1.0) if (kk == 'FRSNOW' and ib == 0) else (fv > 0.0) if (kk == 'FRSNOW') else land)
                    r = IC.cmp(a[ib], b[ib], m); r['cat'] = IC.cat(r)
                    r['rms'] = float(np.sqrt(np.mean((a[ib] - b[ib])[m] ** 2))) if m.any() else 0.0
                    rows[f'{kk}[{ib}]'] = r
                continue
            m = sel[kk]
            r = IC.cmp(a, b, m); r['cat'] = IC.cat(r)
            r['rms'] = float(np.sqrt(np.mean((a - b)[m] ** 2))) if m.any() else 0.0
            rows[kk] = r
        OWN['log'].append(dict(kind='surface', itime=int(it), fields=rows))
        if OWN.get('save_dir'):
            np.savez(os.path.join(OWN['save_dir'], f'own_surface_{it}.npz'), **{kk: np.asarray(v) for kk, v in own.items()})
        return rows

    def step(self, k, state, *a, **kw):
        it = self.it0 + k
        if RAD['mode'] == 'server':
            import radiation_server as rs
            import atm_step as A
            it = self.it0 + k
            if A.is_radiation_step(it):                      # live packet of THIS radiation step: its 21 surface fields are read only with --surf-fields recorded (and, at step 0, RQT/KLIQ/SNOAGE)
                pk = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it}_in.bin')
                if OWN['surf'] == 'computed':
                    own = self.own_surface(k, it, state)
                    self._compare_surface(it, own)
                    pk = dict(pk)
                    pk.update(own)
                self.ph.live_packet = pk
            kw['server'], kw['lv'] = RAD['wrap'], RAD['lv']
        elif OWN['check']:
            import atm_step as A
            if A.is_radiation_step(it):
                self._compare_surface(it, self.own_surface(k, it, state))
        out = super().step(k, state, *a, **kw)
        if OWN['surf'] == 'computed' or OWN['check']:
            self._land_update(it, out[0])
        if k >= 44 and HOOK['outdir']:                       # D209: the carried GHY state after steps 44-53 (small), for the comparison with the entry records
            dn = out[0]['land_prev']['dyn_next']
            np.savez(os.path.join(HOOK['outdir'], f'land_step{k:02d}.npz'), **{kk: np.asarray(dn[kk]) for kk in ('w', 'ht', 'fr_snow')})
        if HOOK['outdir'] and 'SNOAGE' in out[0]['carry']:       # D209 diagnostic: our carried SNOAGE (CONDSE exit) vs the recorded CONDSE exit of this step
            import clouds_condse_io as cio
            ro = np.asarray(cio.read_cse(f"{cio.FF_DEFAULT}/nov26_day/ffc_cse_out_{33312 + k}.bin")['SNOAGE'])
            d = np.abs(np.asarray(out[0]['carry']['SNOAGE']) - ro)
            SNO_DRIFT.append(dict(k=k, max_abs=float(d.max()), n_diff=int((d > 0).sum()), by_type=[float(d[t].max()) for t in range(d.shape[0])]))
        return out


def allarr_light(arrays):
    allarr = {}
    SS2 = C.tree_np(arrays['SS2'])
    for grp in ('ice', 'lake'):
        allarr.update(K.flatten(SS2[grp], f'surf/{grp}/'))
    V2n = C.tree_np(arrays['V2'])
    allarr.update(K.flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy']), 'surf/v2/'))
    return allarr


_orig_daily = D.daily_on_state
CONSTS = dict(mode='computed', dh2o=None)


def daily_with_lake(state, dev, ctx_imf, mdrya, it):
    if CONSTS['mode'] == 'computed':                 # D209: MDRYA, ch4ox water mass and SNOAGE computed (drv_daily), no record read
        import drv_daily as DD
        if CONSTS['dh2o'] is None:
            CONSTS['dh2o'] = DD.getqma()
        state, rec = DD.daily_on_state_computed(state, dev, ctx_imf, it, dh2o=CONSTS['dh2o'])
        if 'SNOAGE' in state['carry']:                # comparison only (the record is NOT used for the state): computed vs CONDSE entry record
            rs = np.asarray(dev['cse_in']['SNOAGE']); cs = np.asarray(state['carry']['SNOAGE'])
            rec['snoage_computed_vs_record_max_abs'] = float(np.abs(rs - cs).max()); rec['snoage_computed_vs_record_bitwise'] = bool(np.array_equal(rs, cs))
    else:
        state, rec = _orig_daily(state, dev, ctx_imf, mdrya, it)
    if HOOK['on']:
        cp = HOOK['cp']
        dn0 = state['land_prev']['dyn_next']
        np.savez(os.path.join(HOOK['outdir'], 'land_boundary_before.npz'), **{kk: np.asarray(dn0[kk]) for kk in ('w', 'ht', 'fr_snow')})
        state, info = cp.daily_lake_update(state, land_fractions=HOOK['land'])
        dn1 = state['land_prev']['dyn_next']
        np.savez(os.path.join(HOOK['outdir'], 'land_boundary_after.npz'), **{kk: np.asarray(dn1[kk]) for kk in ('w', 'ht', 'fr_snow')},
                 land3_w=np.asarray(cp._land3['w']) if cp._land3 else 0, land3_ht=np.asarray(cp._land3['ht']) if cp._land3 else 0)
        HOOK['info'] = info
        rec['daily_lake'] = dict(n_flake_changed=info['n_flake_changed'], max_abs_dflake=info['max_abs_dflake'], n_rsi_changed=info['n_rsi_changed'],
                                 n_dmwldf_pos=info['n_dmwldf_pos'], counters=info['counters'], seconds=info['seconds'], pow=info['pow'])
    return state, rec


if __name__ == '__main__':
    strict, on, landf = 1, 1, 1
    if '--rad' in sys.argv:
        k = sys.argv.index('--rad')
        RAD['mode'] = sys.argv[k + 1]
        assert RAD['mode'] in ('replay', 'server')
        del sys.argv[k:k + 2]
    for flag in ('--surf-fields', '--seed', '--cosz'):
        if flag in sys.argv:
            k = sys.argv.index(flag)
            v = sys.argv[k + 1]
            assert v in ('recorded', 'computed')
            OWN[flag[2:].replace('-fields', '')] = v
            del sys.argv[k:k + 2]
    if '--check-packet' in sys.argv:
        k = sys.argv.index('--check-packet')
        OWN['check'] = bool(int(sys.argv[k + 1]))
        del sys.argv[k:k + 2]
    JS.make_stage = _make_stage_capture
    for flag in ('--nit-strict', '--daily-lake', '--land-fractions', '--daily-consts'):
        if flag in sys.argv:
            k = sys.argv.index(flag)
            v = sys.argv[k + 1] if flag == '--daily-consts' else int(sys.argv[k + 1])
            del sys.argv[k:k + 2]
            if flag == '--daily-consts':
                assert v in ('computed', 'recorded')
                CONSTS['mode'] = v
            elif flag == '--nit-strict':
                strict = v
            elif flag == '--land-fractions':
                landf = v
            else:
                on = v
    HOOK['on'] = bool(on)
    HOOK['land'] = bool(landf)
    C.Coupled = functools.partial(CoupledD209, nit_strict=bool(strict))
    D.daily_on_state = daily_with_lake
    # D209: --c1dir keeps the FULL per-step arrays (needed for C1 against the NumPy chain); allarr_light unused
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    HOOK['outdir'] = out
    OWN['save_dir'] = out
    RAD['save_dir'] = out
    try:
        D.main()
    finally:
        try:
            hf = getattr(HOOK['cp'].ph, '_handoff', None)
            json.dump(dict(rad_mode=RAD['mode'], sentence=(hf.log.sentence() if hf is not None and RAD['mode'] == 'server' else 'radiation replayed from the real record (not computed)'),
                           log=(hf.log.totals() if hf is not None else None), calls=RAD['calls']), open(os.path.join(out, 'd212_server_calls.json'), 'w'), indent=1, default=float)
            if RAD['srv'] is not None:
                RAD['srv'].stop()
        except Exception as e:
            print('d210 radiation log failed', repr(e), flush=True)
        json.dump(dict(daily_consts=CONSTS['mode'], strict=strict, daily_lake=on, land_fractions=landf, rebuild_log=JS.NIT_LOG, mismatch=JS.NIT_MISMATCH), open(os.path.join(out, 'd212_nit.json'), 'w'), indent=1)
        json.dump(SNO_DRIFT, open(os.path.join(out, 'd210_snoage_drift.json'), 'w'), indent=1)
        json.dump(dict(surf_fields=OWN['surf'], seed=OWN['seed'], cosz=OWN['cosz'], check_packet=OWN['check'], log=OWN['log']), open(os.path.join(out, 'd212_own_inputs.json'), 'w'), indent=1, default=float)
        info = HOOK['info']
        if info is not None:
            json.dump({k: v for k, v in info.items() if not isinstance(v, np.ndarray)}, open(os.path.join(out, 'd212_lake.json'), 'w'), indent=1, default=str)
            np.savez(os.path.join(out, 'd212_lake_hook.npz'), dmwldf=info['dmwldf'], dgml=info['dgml'], svflake=info['svflake'], mdwnimp=info['mdwnimp'], edwnimp=info['edwnimp'],
                     flake=HOOK['cp'].st['geo']['flake'], fland=HOOK['cp'].st['fland'], fearth=HOOK['cp'].st['fearth'])
