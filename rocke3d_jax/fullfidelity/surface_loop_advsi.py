"""D166: optional ADVSI stage for the D164 surface loop (new file; surface_loop.py is not modified).

``surface_post_advsi`` = ``surface_loop.surface_post`` followed by the real model's last ocean_driver call, ADVSI (advsi_ff.advsi), on the
ocean-domain sea ice.  ``run_free`` re-runs the free (state-carrying) surface loop with replayed real atmosphere-side inputs (R1 mode of D164),
with or without ADVSI, and reports the error of our carried state against the real records at every step.

Inputs of ADVSI that are NOT computed here (named boundaries):
  * USI/VSI (DYNSI result on the atmosphere grid): taken from the real ffy_<it>_out.bin dump (DYNSI itself is not ported; D164 section 4).
  * RSIX/RSIY: carried by ADVSI itself (initial values from the real restart).
  * RSISAVE = RSI at DYNSI entry = the ocean-domain RSI of the state handed to surface_post (MELT_SI/PRECIP_SI are the only earlier changes).
Pole rows: after ADVSI the ocean-domain pole rows are re-replicated from i = 1 with surface_loop._pole_replicate (same convention as after FORM_SI).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface_loop as L  # noqa: E402
import advsi_ff as A  # noqa: E402

FF = L.FF
IM, JM = L.IM, L.JM


def usi_vsi_from_ffy(date, it, ff=FF):
    """ATMICE%USI/VSI (B-grid ice velocities handed to ADVSI) from the real DYNSI exit dump ffy_<it>_out.bin: (IM, JM) arrays [i, j]."""
    import dynsi_compare as DC
    o = DC.read_dynsi_out(f"{ff}/{date}/ffy_{it}_out.bin")
    return o['usi'][1:IM + 1, 1:JM + 1].copy(), o['vsi'][1:IM + 1, 1:JM + 1].copy()


def advsi_geo_from_dump(date, it, ff=FF, adv_dir=None):
    """The five ADVSI geometry vectors from a real ffadv_in dump (static; any step)."""
    d, _ = A.read_dump(f"{adv_dir or ff + '/advsi_dumps/' + date}/ffadv_in_{it}.bin")
    return d['geo']


def init_adv(date, ff=FF):
    """RSIX/RSIY at the start of the window from the real restart (ocean domain; zero elsewhere)."""
    import netCDF4 as nc
    R = nc.Dataset(f"{ff}/_pristine_restarts/{L.RESTART[date]}")
    return dict(rsix=np.array(R.variables['rsix'][:], float).T.copy(), rsiy=np.array(R.variables['rsiy'][:], float).T.copy())


def advsi_on_state(ice, adv, rsisave, ausi, avsi, geo, advgeo):
    """Apply ADVSI to the combined (ocean + lake domain) ice dict `ice`; only FOCEAN > 0 cells are touched.  Returns (ice', adv', outputs)."""
    st = dict(rsi=ice['rsi'], rsix=adv['rsix'], rsiy=adv['rsiy'], rsisave=rsisave, msi=ice['msi'], snowi=ice['snowi'], hsi=ice['hsi'], ssi=ice['ssi'])
    new, out = A.advsi(st, ausi, avsi, geo['focean'], advgeo)
    ice2 = L.copy_ice(ice)
    for k in ('rsi', 'msi', 'snowi', 'hsi', 'ssi'):
        ice2[k] = np.where(geo['is_ocean'][..., None], new[k], ice[k]) if new[k].ndim == 3 else np.where(geo['is_ocean'], new[k], ice[k])
    L._pole_replicate(ice2, geo)
    return ice2, dict(rsix=new['rsix'], rsiy=new['rsiy']), out


def surface_post_advsi(S, st, inp, mid, adv, ausi, avsi, advgeo, ocean_stages=None, dynsi=None):
    """surface_loop.surface_post then ADVSI.  S: state after surface_pre.  Returns (S', post, adv', advsi_outputs)."""
    rsisave = S['ice']['rsi'].copy()
    S2, post = L.surface_post(S, st, inp, mid, ocean_stages=ocean_stages, dynsi=dynsi)
    ice2, adv2, out = advsi_on_state(S2['ice'], adv, rsisave, ausi, avsi, st['geo'], advgeo)
    return dict(S2, ice=ice2), post, adv2, out


def _rel(cmp, names):
    return {n: (cmp[n][0] / cmp[n][1] if cmp[n][1] else cmp[n][0]) for n in names}


ICE_NAMES = ('ice.snow', 'ice.msi2', 'ice.ssi1', 'ice.tg1', 'ice.tg2', 'ice.ptype', 'ocean.ptype')


def run_free(nsteps=6, date='nov26', it0=33312, with_advsi=True, ff=FF, adv_dir=None, log=print):
    """Free surface loop (R1: real tile outputs / RIVERF / DYNSI boundary replayed each step; ocean, ice, lake, land-ice state carried from
    OUR computation).  Per step: errors of the carried state against the real step-entry records (surface_loop.compare_pre_records, relative
    to the field scale) and of the ocean exit state against the real ffo exit (tag 14); with ADVSI also the direct comparison of our post-ADVSI
    ice with the real ffadv_out dump.  Returns a list of dicts."""
    import ocean_chain_io as C
    from ocean_step_chain_compare import errs
    st = L.load_statics(date, ff)
    st['ctx'] = L.make_ocean_ctx(date, ff)
    S = L.init_surface_state(date, ff, st=st)
    adv = init_adv(date, ff)
    advdir = adv_dir or f"{ff}/advsi_dumps/{date}"
    advgeo = advsi_geo_from_dump(date, it0, ff, adv_dir=advdir)
    rows = []
    for k in range(nsteps):
        it = it0 + k
        inp = L.replay_inputs(date, it, ff)
        S1, mid = L.surface_pre(S, st, inp)
        cmp = L.compare_pre_records(S1, mid, st, date, it, ff)
        row = dict(step=k, itime=it, entry=_rel(cmp, ICE_NAMES), tileset=(cmp['tileset.ocean_mismatch'][0], cmp['tileset.ice_mismatch'][0]),
                   lake=_rel(cmp, ('lake.tg1', 'lake.mwl')), landice=_rel(cmp, ('landice.tg1', 'landice.tg2', 'landice.snow')))
        if with_advsi:
            ausi, avsi = usi_vsi_from_ffy(date, it, ff)
            S2, post, adv, aout = surface_post_advsi(S1, st, inp, mid, adv, ausi, avsi, advgeo)
            din, dout = f"{advdir}/ffadv_in_{it}.bin", f"{advdir}/ffadv_out_{it}.bin"
            if os.path.exists(dout):
                d, o = A.read_dump(din, dout)
                oc = st['geo']['is_ocean'].copy()
                oc[1:, 0] = False; oc[1:, -1] = False      # pole rows i > 1: stale in the Fortran, re-replicated here -> not compared
                row['advsi_vs_real_out'] = {kk: float(np.abs(np.where(oc[..., None] if S2['ice'][kk].ndim == 3 else oc, S2['ice'][kk] - o[kk], 0.0)).max())
                                            for kk in ('rsi', 'msi', 'snowi', 'hsi', 'ssi')}
                row['advsi_rsix_vs_real'] = float(np.abs(adv['rsix'] - o['rsix']).max())
                row['advsi_in_usi_equals_ffy'] = bool(np.array_equal(d['ausi'], ausi) and np.array_equal(d['avsi'], avsi))
        else:
            S2, post = L.surface_post(S1, st, inp, mid)
        sn = C.load_step(f"{ff}/{date}", it)
        e = errs(S2['ocean'], sn[14])
        row['ocean_exit_abs'] = {kk: float(v) for kk, v in e.items()}
        rows.append(row)
        log(f"step {k} it={it}: entry {row['entry']} tileset {row['tileset']} ocean_exit {row['ocean_exit_abs']}")
        S = S2
    return rows
