"""D158: GhyColumn.advnc with the Fortran sub-iteration time loop (GHY.f:2389-2416) restored.

The ffg record holds Ent exports and dts for at most 11 sub-iterations (ghy_compare.unpack: n_avail = min(ffnit, 11)); a cell with
ffnit >= 12 is therefore advanced by ghy_ref/ghy_jax for only the first 11 of its ffnit iterations, i.e. over sum(dts) < dt.
advnc_full runs the real loop `do while (dtr > 0)`: hydra, xklh, gdtm -> dts = dtr if dtm >= dtr else min(dtm, dtr/2).
Iterations <= 11 use the recorded Ent exports (and the recorded dts is checked against the recomputed one); iterations > 11 have no
recorded Ent exports and reuse those of the last recorded iteration (ASSUMPTION, flagged in the returned info).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_ref as G
import ghy_compare as GC


def advnc_full(col, ent_iters, dt, snowm, ffnit=None, use_recorded_dts=True, reuse_ent=True):
    """Same body as GhyColumn.advnc but with the dtr loop. Returns info dict."""
    col.dt = dt
    col.snowm = snowm
    col._bounds()
    col.accm_zero()
    col.reth()
    col.retp()
    col.tb0 = col.tp[1, 0]; col.tc0 = col.tp[0, 1]
    col.evapb = col.epb = 1.0
    col.evapvw = col.evapvd = col.epv = 1.0
    dtr = dt
    nit = 0
    info = dict(dts=[], dtm=[], dts_rec=[])
    while dtr > 0.0:
        nit += 1
        col.hydra()
        col.xklh()
        dtm = col.gdtm(nit)
        nrec = len(ent_iters)
        if use_recorded_dts and nit <= nrec:
            dts = ent_iters[nit - 1]['dts']
            dtr = 0.0 if (ffnit is not None and nit == ffnit) else dtr - dts
        elif dtm >= dtr:
            dts = dtr; dtr = 0.0
        else:
            dts = min(dtm, dtr * 0.5); dtr = dtr - dts
        rec = ent_iters[min(nit, nrec) - 1]
        info['dtm'].append(dtm)
        info['dts_rec'].append(ent_iters[nit - 1]['dts'] if nit <= nrec else None)
        col.dts = dts
        info['dts'].append(dts)
        col.cnc = rec['cnc'] if col.process_vege else 0.0
        col.betadl = np.asarray(rec['betadl']) if col.process_vege else np.zeros(col.n)
        col.lai = rec['lai'] if col.process_vege else 0.0
        col.evap_limits(True)
        # ghy_ref.evap_limits stores epb/epv (GHY.f:905,907) for gdtm of the next iteration (applied to ghy_ref.py, D158)
        col.drip_from_canopy()
        col.sensible_heat()
        col.snow()
        col.fl(); col.flg(); col.runoff(); col.fllmt(); col.flh(); col.flhg()
        col.apply_fluxes()
        col.accm()
        col.reth(); col.retp()
    col.accm_final()
    col.hydra()
    info['nit'] = nit
    return info


def run_cell_full(rec, dt=900.0, **kw):
    static, dynamic, forcing, ent_iters, refs, snowm = GC.unpack(rec)
    col = G.GhyColumn(static, dynamic, forcing)
    col.fb, col.fv = forcing['fb'], forcing['fv']
    info = advnc_full(col, ent_iters, dt, snowm, ffnit=refs['ffnit'], **kw)
    return col, refs, info
