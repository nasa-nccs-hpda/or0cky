"""D168: free surface loop (replay mode R1 of D164: real SURFACE tile outputs as the flux input, no ADVSI, RIVERF recorded) with the DYNSI result
(odmui, odmvi, ustar) COMPUTED by dynsi_ff instead of read from the dumps.  New file; surface_loop.py is imported, not modified (surface_post's
existing `dynsi=` argument carries the computed boundary).

run(computed=True/False, nsteps=6) -> list of per-step dicts:
  pre  : surface_loop.compare_pre_records (our state vs the real substep-1 tile records of the step)
  ocean: relative errors of our ocean exit state vs ffo tag 14 (ocean_step_chain_compare.errs)
  dyn  : our DYNSI result vs the recorded one (odmui, odmvi, ustar) for the state of this run
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface_loop as L  # noqa: E402
import dynsi_ff as DF  # noqa: E402

IM, JM = L.IM, L.JM


def ice_tile_dmua(date, it, ff=L.FF):
    """Real ice-tile (type 2) DMUA/DMVA accumulated over the two substeps (kg/m s * s: SURFACE.f:841), replay-mode atmosphere-side input."""
    import surface_tile_ff as ST
    t = ST.load(f"{ff}/{date}/ffs_{it}.bin"); n = len(t) // 2
    a = np.zeros((IM, JM)); b = np.zeros((IM, JM))
    for ns in (0, 1):
        tt = t[ns * n:(ns + 1) * n]
        r = tt[tt[:, 2] == 2]
        i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
        dts = r[:, ST.IN['dtsurf']]
        np.add.at(a, (i, j), r[:, ST.OUT['dmua']] * dts)
        np.add.at(b, (i, j), r[:, ST.OUT['dmva']] * dts)
    return a, b


class Dynsi:
    """DYNSI with its persistent ice-velocity state (USI, VSI of the restart, then our own output)."""

    def __init__(self, date='nov26', ff=L.FF):
        import netCDF4 as nc
        from odhorz_ff import geomo_dyn_arrays
        R = nc.Dataset(f"{ff}/_pristine_restarts/{L.RESTART[date]}")
        self.usi = np.array(R.variables['usi'][:], float).T.copy()
        self.vsi = np.array(R.variables['vsi'][:], float).T.copy()
        self.sinvo, self.sinpo = geomo_dyn_arrays()[:2]
        self.G = None

    def __call__(self, S1, st, dmua, dmva):
        geo = st['geo']
        if self.G is None:
            self.G = DF.Geom(geo['focean'])
        oo = geo['is_ocean']
        ice = S1['ice']
        rsi, msi, snowi = (np.where(oo, ice[k], 0.0) for k in ('rsi', 'msi', 'snowi'))
        oc = S1['ocean']
        oga = L.toc2sst(oc, st['ctx'])['ogeoza']
        us, vs = DF.uosurf_from_ocean(oc['uo'][:, :, 0], oc['vo'][:, :, 0], self.sinpo, self.sinvo)
        out = DF.dynsi(self.G, dmua, dmva, rsi, msi, snowi, oga, us, vs, self.usi, self.vsi)
        self.usi, self.vsi = out['usi'], out['vsi']
        return out


def run(computed=True, nsteps=6, date='nov26', it0=33312, ff=L.FF, log=print):
    import ocean_chain_io as C
    from ocean_step_chain_compare import errs
    st = L.load_statics(date, ff)
    st['ctx'] = L.make_ocean_ctx(date, ff)
    S = L.init_surface_state(date, ff, st)
    dyn = Dynsi(date, ff)
    dyn_rec = Dynsi(date, ff)                       # separate state only used for the diagnostic recorded-vs-computed comparison
    rows = []
    for k in range(nsteps):
        it = it0 + k
        inp = L.replay_inputs(date, it, ff)
        S1, mid = L.surface_pre(S, st, inp)
        pre = L.compare_pre_records(S1, mid, st, date, it, ff)
        dmua, dmva = ice_tile_dmua(date, it, ff)
        d = dyn(S1, st, dmua, dmva)
        rec = inp['dynsi']
        ocn = st['geo']['is_ocean'] & st['geo']['valid']
        u = rec['undocn']; ii, jj = u['i'].astype(int) - 1, u['j'].astype(int) - 1
        diag = dict(odmui=float(np.abs(d['odmui'] - rec['odmui'])[ocn].max()), odmvi=float(np.abs(d['odmvi'] - rec['odmvi'])[ocn].max()),
                    odmui_scale=float(np.abs(rec['odmui']).max()),
                    ustar_abs=float(np.abs(d['ustar'][ii, jj] - u['ustar']).max()), ustar_scale=float(np.abs(u['ustar']).max()),
                    ustar_cells_ours=int((d['ui2rho'] > 0).sum()), ustar_cells_rec=int(len(ii)))
        if computed:
            dynb = dict(odmui=d['odmui'], odmvi=d['odmvi'], ustar=d['ustar'], ui2rho=d['ui2rho'])
        else:
            dynb = rec
        S2, post = L.surface_post(S1, st, inp, mid, dynsi=dynb)
        sn = C.load_step(f"{ff}/{date}", it)
        e = errs(S2['ocean'], sn[14])
        rows.append(dict(step=k, itime=it, pre=pre, ocean=e, dyn=diag, kki=d['kki']))
        log(f"step {k} it {it} computed={computed} kki {d['kki']} dyn {diag}")
        log("   pre " + ' '.join(f"{n}={pre[n][0]:.2e}/{pre[n][1]:.2e}" for n in ('ice.snow', 'ice.msi2', 'ice.tg1', 'ice.ptype')) +
            f" tileset ocean {pre['tileset.ocean_mismatch'][0]:.0f} ice {pre['tileset.ice_mismatch'][0]:.0f}")
        log("   ocean exit " + ' '.join(f"{n}={e[n]:.2e}" for n in ('g0m', 's0m', 'mo', 'uo', 'vo') if n in e))
        sys.stdout.flush()
        S = S2
    return rows


if __name__ == '__main__':
    import json
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    res = {}
    for comp in ((False, True) if which == 'both' else (which == 'computed',)):
        res['computed' if comp else 'recorded'] = run(computed=comp)
    out = os.environ.get('D168_OUT')
    if out:
        json.dump(res, open(out, 'w'), default=float)
