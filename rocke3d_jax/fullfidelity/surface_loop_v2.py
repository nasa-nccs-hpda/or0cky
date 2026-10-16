"""D170: ONE combined surface loop with ADVSI (D166), RIVERF (D167) and the DYNSI glue (D168) all computed, for both the replay-mode
free loop (R1 of D164) and the closed/coupled path.  New module; surface_loop.py, surface_loop_advsi.py, riverf_loop.py and
dynsi_loop_ff.py are imported, not modified.

``surface_post_v2`` is surface_loop.surface_post with exactly these differences (each stated, each switchable for isolation runs):
  1. DYNSI (dynsi_ff.dynsi) is COMPUTED at the head of the ocean driver from OUR state (ice RSI/MSI/SNOWI of the ocean domain, ocean
     surface velocity, OGEOZA) and the ice-tile DMUA/DMVA of the step; its odmui/odmvi/ustar replace the recorded boundary; its USI/VSI
     (the persistent B-grid ice velocity) are carried; uisurf/visurf (GET_UISURF) are computed and returned (UOdrag = 0 in this build, so
     the tiles do not read them).  dynsi='recorded' uses the recorded odmui/odmvi/ustar and the recorded ffy USI/VSI instead.
  2. GROUND_LK gets the FULL geo (land runoff added to every FLAND > 0 cell, D167 finding), RIVERF is applied between GROUND_LK and
     FORM_SI, oflowo/oeflowo are COMPUTED, and GTEMP/GTEMPR/MLHC of lake cells are reset after RIVERF (riverf_loop.surface_post_riverf).
     riverf=False restores the D164 recorded flows and lake-only runoff.
  3. ADVSI is applied last on the ocean-domain ice (surface_loop_advsi.advsi_on_state), RSISAVE = RSI at DYNSI entry, RSIX/RSIY carried.
     advsi=False skips it.
Still recorded in every mode (stated, not hidden): radiation, Ent exports and the ffg land forcing columns, TRUP_in_rad, PBL profile
columns, MMST (init_STRAITS), the five static ADVSI geometry vectors (read from a real ffadv_in dump; static), the irrigation demand
(reconstructed from the recorded actual flux).  In the replay mode additionally the real SURFACE tile outputs.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface_loop as L  # noqa: E402
import surface_loop_advsi as AL  # noqa: E402
import dynsi_loop_ff as DL  # noqa: E402
import dynsi_ff as DF  # noqa: E402
import riverf_ff as RF  # noqa: E402
import advsi_ff as A  # noqa: E402

FF = L.FF
IM, JM = L.IM, L.JM


class V2State:
    """Persistent state of the three ported pieces that the restart does not make part of the (ocean, ice, lake, land-ice) state:
    USI/VSI (via Dynsi), RSIX/RSIY (ADVSI) and the static tables (RIVERF river network, ADVSI geometry)."""

    def __init__(self, st, date='nov26', ff=FF, it0=33312, dynsi='computed', advsi=True, riverf=True, advsi_usi='dynsi', li_e1_precip=False):
        self.li_e1_precip = li_e1_precip
        self.dynsi_mode, self.advsi_on, self.riverf_on, self.advsi_usi = dynsi, advsi, riverf, advsi_usi
        self.rs = RF.load_statics(st['axyp'][0]) if riverf else None
        self.dyn = DL.Dynsi(date, ff)
        self.adv = AL.init_adv(date, ff)
        self.advgeo = AL.advsi_geo_from_dump(date, it0, ff) if advsi else None
        self.date, self.ff = date, ff
        self.uisurf = self.visurf = None


def ice_tile_dmua_from_tiles(sd):
    """Ice-tile (type 2) DMUA/DMVA accumulated over the two substeps (x DTSURF) from OUR tile outputs (closed path)."""
    import surface_tile_ff as ST
    a = np.zeros((IM, JM)); b = np.zeros((IM, JM))
    for rr, rows in ((sd['r1'], sd['ta']), (sd['r2'], sd['tnew2'])):
        got = {k: np.asarray(v) for k, v in rr['tile'].items()}
        i, j = L._rows_ij(rows)
        dts = rows[:, ST.IN['dtsurf']]
        q = rows[:, 2] == 2
        np.add.at(a, (i[q], j[q]), got['dmua'][q] * dts[q]); np.add.at(b, (i[q], j[q]), got['dmva'][q] * dts[q])
    return a, b


def surface_post_v2(S, st, inp, mid, V, ocean_stages=None):
    """See the module docstring.  inp: acc (tile accumulators incl. dmua_i/dmva_i), srfp, itime, and for the replay/recorded switches
    flows, dynsi, ffy_usi/ffy_vsi.  V: V2State (mutated: USI/VSI, RSIX/RSIY).  Returns (S', post)."""
    geo, ctx = st['geo'], st['ctx']
    acc = inp['acc']
    gl, gol = L.geo_sub(geo, 'lake'), L.geo_sub(geo, 'ocean')
    ice, atm, lake, li = S['ice'], {k: v.copy() for k, v in S['atm'].items()}, S['lake'], S['li']
    rsisave = ice['rsi'].copy()                                   # RSI at DYNSI entry (ocean domain unchanged by the lake chain)
    # ---- DYNSI (head of ocean_driver, ICEDYN_DRV.f) -----------------------------------------------------------------------
    if V.dynsi_mode == 'computed':
        d = V.dyn(S, st, acc['dmua_i'], acc['dmva_i'])
        dyn = dict(odmui=d['odmui'], odmvi=d['odmvi'], ustar=d['ustar'], ui2rho=d['ui2rho'])
        V.uisurf, V.visurf = d['uisurf'], d['visurf']
        usi_adv, vsi_adv = d['usi'], d['vsi']
    else:
        dyn = inp['dynsi']
        usi_adv, vsi_adv = inp['ffy_usi'], inp['ffy_vsi']
        d = None
    if V.advsi_usi == 'ffy' and 'ffy_usi' in inp:
        usi_adv, vsi_adv = inp['ffy_usi'], inp['ffy_vsi']
    # D170: PRECIP_LI's EDIFS is NOT added to E1 (SURFACE_LANDICE.f:132 zeroes igla%e1 at the first substep); measured bit-exact, see ledger
    li, gli = L.ground_li(li, acc['e0_li'], acc['e1_li'] + (mid['pli']['e1'] if V.li_e1_precip else 0.0), acc['evap_li'], st['flice'], geo)
    fm, fh, fs = L.underice(ice, atm['gtemp'], atm['sss'], atm['mlhc'], np.zeros((IM, JM)), st['coriol'], lake['mldlk'], mid['dl']['dlake'],
                            mid['dl']['glake'], gl)
    ice, gs_l = L.ground_si(ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], gl)
    lake, gt_l, dm = L.ground_lk(lake, ice, st['hlake'], st['axyp'], st['fland'], st['flice'], st['fearth'], st['fgeotherm'], gli['runo'],
                                 acc['rune'], acc['erune'], acc['e0_o'], acc['evap_o'], acc['solar_o'], gs_l['runosi'], gs_l['erunosi'],
                                 gs_l['solar_io'], geo if V.riverf_on else gl)
    for k in ('gtemp', 'gtemp2', 'gtempr'):
        atm[k] = np.where(geo['is_lake'], gt_l[k], atm[k])
    if V.riverf_on:
        lk_rv, flowo, eflowo, gtm, gtr, mlh, rdiag = RF.riverf(lake, geo['flake'], st['fland'], st['fearth'], V.rs)
        lake = {k: lk_rv[k] for k in ('mwl', 'gml', 'tlake', 'mldlk')}
        isl = geo['is_lake']
        atm['gtemp'] = np.where(isl, gtm, atm['gtemp']); atm['gtempr'] = np.where(isl, gtr, atm['gtempr']); atm['mlhc'] = np.where(isl, mlh, atm['mlhc'])
    else:
        flowo, eflowo, rdiag = inp['flows']['oflowo'], inp['flows']['oeflowo'], None
    ice = L.form_si(ice, dm['dmsi'], dm['dhsi'], dm['dssi'], gl)
    oc = S['ocean']
    fm, fh, fs = L.underice(ice, atm['gtemp'], atm['sss'], atm['mlhc'], dyn.get('ui2rho', np.zeros((IM, JM))), st['coriol'], lake['mldlk'],
                            mid['dl']['dlake'], mid['dl']['glake'], gol, ustar_override=dyn.get('ustar'))
    ice, gs_o = L.ground_si(ice, acc['e0_i'], acc['e1_i'], acc['evap_i'], acc['solar_i'], fm, fh, fs, atm['gtemp'], atm['sss'], gol)
    ice_after_gsi = L.copy_ice(ice)
    apress = L.calc_apress(inp['srfp'], ice, geo)
    rsi = ice['rsi']
    foc = geo['focean']
    sel = geo['is_ocean'] & geo['valid']
    inv_f = 1.0 / np.where(foc > 0, foc, 1.0)
    mt = mid['melt']
    fx = dict(mid['fxp'])
    fx.update(oflowo=flowo, oeflowo=eflowo,
              omelti=np.where(sel, mt['melti'] * inv_f, 0.0), oemelti=np.where(sel, mt['emelti'] * inv_f, 0.0),
              osmelti=np.where(sel, mt['smelti'] * inv_f, 0.0),
              oevapor=np.where(sel, acc['evap_o'], 0.0), oe0=np.where(sel, acc['e0_o'], 0.0), osolarw=np.where(sel, acc['solar_o'], 0.0),
              orunosi=np.where(sel, gs_o['runosi'], 0.0), oerunosi=np.where(sel, gs_o['erunosi'], 0.0),
              osrunosi=np.where(sel, gs_o['srunosi'], 0.0), osolari=np.where(sel, gs_o['solar_io'], 0.0),
              oapress=np.where(geo['is_ocean'], apress, 0.0),
              odmua=np.where(sel, acc['dmua_o'] * (1.0 - rsi), 0.0), odmva=np.where(sel, acc['dmva_o'] * (1.0 - rsi), 0.0),
              odmui=dyn['odmui'], odmvi=dyn['odmvi'], orsi=rsi.copy(), itime=inp['itime'])
    ogeoz_entry = oc['ogeoz'].copy()
    for name, fn, _t in (ocean_stages or L.ocean_stages_after_precip()):
        oc = fn(oc, fx, ctx)
    oc['ogeoz_sv'] = ogeoz_entry
    ice = L.form_si(ice, oc['odmsi'], oc['odhsi'], oc['odssi'], gol)
    ice_pre_adv = L.copy_ice(ice)
    t = L.toc2sst(oc, ctx)
    for k in ('gtemp', 'gtemp2', 'gtempr', 'sss', 'mlhc'):
        atm[k] = np.where(geo['is_ocean'], t[k], atm[k])
    aout = None
    if V.advsi_on:
        ice, V.adv, aout = AL.advsi_on_state(ice, V.adv, rsisave, usi_adv, vsi_adv, geo, V.advgeo)
    S2 = dict(S, ocean=oc, ice=ice, lake=lake, li=li, atm=atm, itime=S['itime'] + 1)
    return S2, dict(fx=fx, flowo=flowo, eflowo=eflowo, rdiag=rdiag, dyn=d, ice_after_gsi=ice_after_gsi, ice_pre_adv=ice_pre_adv,
                    advsi_out=aout, gs_l=gs_l, gs_o=gs_o, gli=gli, apress=apress, dm=dm)


# ------------------------------------------------------------------------------------------------ replay inputs
def replay_inputs_v2(date, it, ff=FF):
    inp = L.replay_inputs(date, it, ff)
    a, b = DL.ice_tile_dmua(date, it, ff)
    inp['acc']['dmua_i'], inp['acc']['dmva_i'] = a, b
    inp['ffy_usi'], inp['ffy_vsi'] = AL.usi_vsi_from_ffy(date, it, ff)
    return inp


def _rel(cmp, names):
    return {n: (cmp[n][0] / cmp[n][1] if cmp[n][1] else cmp[n][0]) for n in names}


ICE_NAMES = ('ice.snow', 'ice.msi2', 'ice.ssi1', 'ice.tg1', 'ice.tg2', 'ice.ptype', 'ocean.ptype')


def run_free_v2(nsteps=6, date='nov26', it0=33312, ff=FF, log=print, **flags):
    """Free loop in replay mode R1 (real SURFACE tile outputs as the flux input, real PREC/EPREC/SRFP; ocean, ice, lake, land-ice state
    carried from OUR computation) with the v2 post stage.  flags: dynsi ('computed'|'recorded'), advsi, riverf, advsi_usi.
    Per step: entry-record errors relative to the field scale (as surface_loop_advsi), ocean exit errors vs ffo tag 14 (relative, errs),
    our post-ADVSI ice vs the real ffadv_out (abs), our DYNSI vs the recorded result, our RIVERF flows vs the recorded ones.
    Returns (rows, S, V)."""
    import ocean_chain_io as C
    from ocean_step_chain_compare import errs
    st = L.load_statics(date, ff)
    st['ctx'] = L.make_ocean_ctx(date, ff)
    S = L.init_surface_state(date, ff, st=st)
    V = V2State(st, date, ff, it0, **flags)
    advdir = f"{ff}/advsi_dumps/{date}"
    rows = []
    for k in range(nsteps):
        it = it0 + k
        inp = replay_inputs_v2(date, it, ff)
        S1, mid = L.surface_pre(S, st, inp)
        cmp = L.compare_pre_records(S1, mid, st, date, it, ff)
        row = dict(step=k, itime=it, entry=_rel(cmp, ICE_NAMES), tileset=(cmp['tileset.ocean_mismatch'][0], cmp['tileset.ice_mismatch'][0]),
                   lake=_rel(cmp, ('lake.tg1', 'lake.mwl', 'lake.gml')), landice=_rel(cmp, ('landice.tg1', 'landice.tg2', 'landice.snow')),
                   cmp_raw={n: cmp[n] for n in cmp})
        S2, post = surface_post_v2(S1, st, inp, mid, V)
        rec = inp['dynsi']
        ocn = st['geo']['is_ocean'] & st['geo']['valid']
        if post['dyn'] is not None:
            dd = post['dyn']
            u = rec['undocn']; ii, jj = u['i'].astype(int) - 1, u['j'].astype(int) - 1
            row['dyn_vs_rec'] = dict(odmui=float(np.abs(dd['odmui'] - rec['odmui'])[ocn].max()), odmvi=float(np.abs(dd['odmvi'] - rec['odmvi'])[ocn].max()),
                                     odmui_scale=float(np.abs(rec['odmui']).max()), ustar=float(np.abs(dd['ustar'][ii, jj] - u['ustar']).max()),
                                     ustar_scale=float(np.abs(u['ustar']).max()),
                                     usi=float(np.abs(dd['usi'] - inp['ffy_usi']).max()), vsi=float(np.abs(dd['vsi'] - inp['ffy_vsi']).max()),
                                     usi_scale=float(np.abs(inp['ffy_usi']).max()))
        if V.riverf_on:
            r1, r2 = inp['flows']['oflowo'], inp['flows']['oeflowo']
            row['flows'] = dict(maxabs_o=float(abs(post['flowo'] - r1).max()), scale_o=float(abs(r1).max()),
                                maxabs_eo=float(abs(post['eflowo'] - r2).max()), scale_eo=float(abs(r2).max()))
        dout = f"{advdir}/ffadv_out_{it}.bin"
        if V.advsi_on and os.path.exists(dout):
            _, o = A.read_dump(f"{advdir}/ffadv_in_{it}.bin", dout)
            oc_ = st['geo']['is_ocean'].copy(); oc_[1:, 0] = False; oc_[1:, -1] = False
            row['advsi_vs_real_out'] = {kk: float(np.abs(np.where(oc_[..., None] if S2['ice'][kk].ndim == 3 else oc_, S2['ice'][kk] - o[kk], 0.0)).max())
                                        for kk in ('rsi', 'msi', 'snowi', 'hsi', 'ssi')}
            din, _ = A.read_dump(f"{advdir}/ffadv_in_{it}.bin")
            row['advsi_entry_vs_real'] = {kk: float(np.abs(np.where(oc_[..., None] if post['ice_pre_adv'][kk].ndim == 3 else oc_,
                                                                   post['ice_pre_adv'][kk] - din[kk], 0.0)).max()) for kk in ('rsi', 'msi', 'snowi', 'hsi', 'ssi')}
        sn = C.load_step(f"{ff}/{date}", it)
        row['ocean_exit'] = {kk: float(v) for kk, v in errs(S2['ocean'], sn[14]).items()}
        rows.append(row)
        log(f"step {k} it={it}: ice {row['entry']} lake {row['lake']} landice {row['landice']} tileset {row['tileset']} ocean_exit {row['ocean_exit']}")
        sys.stdout.flush()
        S = S2
    return rows, S, V


# ------------------------------------------------------------------------------------------------ closed / coupled path
class Loop2(L.Loop):
    """surface_loop.Loop with stage_surface using surface_post_v2 (DYNSI/RIVERF/ADVSI computed; no recorded boundary is read except the
    ones listed in the module docstring).  Everything before the post stage is identical to Loop.stage_surface."""

    def __init__(self, date, daydir, ff, st, SS, rec_dir=None, closed_land=True, bnd=None, flags=None, it0=33312):
        super().__init__(date, daydir, ff, st, SS, rec_dir=rec_dir, closed_land=closed_land, bnd=bnd)
        self.V = V2State(st, date, ff, it0, **(flags or {}))

    def stage_surface(self, S, R, ctx, rec=None, tm=None, land_mode='ghy'):
        import land_chain as LCm
        A_ = L.A
        st = self.st
        it = R.itime
        inp = dict(prec=np.asarray(S['PREC']), eprec=np.asarray(S['EPREC']))
        rec0 = A_.surface_records(R)
        g1 = rec0['g1']
        irrig = np.zeros((IM, JM)); irrig[g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1] = g1[:, 147]
        inp['irrig_act'] = irrig * st['fearth']
        S1, mid = L.surface_pre(self.SS, st, inp, melt_done=self.melt)
        rec, rep = L.apply_state_to_records(rec0, S1, mid, st, np.asarray(S['PEDN'][0]))
        for key in ('g1', 'g2'):
            g = rec[key]
            i, j = L._rows_ij(g)
            g[:, 143] = np.asarray(S['PREC'])[i, j] / (L.DTSRC * L.RHOW)
            g[:, 144] = np.asarray(S['EPREC'])[i, j] / L.DTSRC
            g[:, 145] = np.asarray(S['PRECSS'])[i, j] / (L.DTSRC * L.RHOW)
        land_dyn = None
        if self.closed_land and self.land_prev is not None:
            lp = self.land_prev
            pa = rec['pa']
            m4 = pa[:, 2] == 4
            assert np.array_equal(lp['p4_ij'], pa[m4][:, :2]), 'land tile set changed between steps'
            pa4 = LCm.next_land_pbl_columns(pa[m4], lp)
            pa[m4] = pa4
            land_dyn = lp['dyn_next']
        S = L.stage_surface_closed(S, R, ctx, rec, land_dyn=land_dyn, tm=tm)
        sd = S['_surface']
        acc = L.tile_outputs_to_acc(sd, st)
        acc['dmua_i'], acc['dmva_i'] = ice_tile_dmua_from_tiles(sd)
        post_in = dict(acc=acc, srfp=np.asarray(S['PEDN'][0]), itime=it)
        if self.V.dynsi_mode != 'computed' or not self.V.riverf_on:
            post_in.update(self.bnd(it))                       # recorded boundaries only when a piece is switched off
            post_in['ffy_usi'], post_in['ffy_vsi'] = AL.usi_vsi_from_ffy(self.date, it, self.ff)
        SSn, post = surface_post_v2(S1, st, post_in, mid, self.V)
        land2 = dict(sd['r2']['land'])
        land2['p4_ij'] = sd['r2']['land']['p4_ij'] if 'p4_ij' in sd['r2']['land'] else rec['pa'][rec['pa'][:, 2] == 4][:, :2]
        self.land_prev = land2
        self.last = dict(S1=S1, mid=mid, post=post, rec_report=rep, acc=acc, rec=rec, SSn=SSn)
        self.SS = SSn
        S['_surf_state'] = SSn
        return S


def run_coupled_v2(nsteps=6, date='nov26', daydir='nov26_day', it0=33312, ff=FF, out_dir=None, log=print, flags=None, **kw):
    """surface_loop.run_coupled with Loop replaced by Loop2 (no edit of surface_loop.py: the module attribute is swapped for the call)."""
    orig = L.Loop
    L.Loop = lambda date_, daydir_, ff_, st_, SS_, bnd=None, **k: Loop2(date_, daydir_, ff_, st_, SS_, bnd=bnd, flags=flags, it0=it0, **k)
    try:
        return L.run_coupled(nsteps=nsteps, date=date, daydir=daydir, it0=it0, ff=ff, out_dir=out_dir, log=log, **kw)
    finally:
        L.Loop = orig
