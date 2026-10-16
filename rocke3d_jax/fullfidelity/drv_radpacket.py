"""D176: the SURFACE side of the 52-field RADIA packet (radiation_server.INPUT_FIELDS: RSI ZSI SNOWI POND_MELT FLAG_DSWS FLAKE DLAKE FLICE FLAND
FEARTH GTEMPR1-4 TSAVG WSAVG SNOWLI ZSNOWI BARESW FRSNOW SNOWD; 21 arrays) computed from OUR surface state, instead of the 'live' values of the
real model that atm_day_free_rad.assemble_packet replays.

Where RADIA reads them (RAD_DRV.f 3310-3391) and what they are in the model:
  state at the time of RADIA = state at the END of the previous step + MELT_SI + seaice_to_atmgrid of THIS step (ATM_DRV.f 257-259), i.e. BEFORE PRECIP_SI/PRECIP_OC/
  PRECIP_LI/PRECIP_LK and before the SURFACE of this step (found by measurement: the packet RSI/SNOWI equal the melt state, not the post-precipitation state).
  RSI SNOWI POND_MELT FLAG_DSWS : ice state after MELT_SI.            ZSI = (ACE1I+MSI)/RHOI, ZSNOWI = SNOWI/RHOS, GTEMPR2: seaice_to_atmgrid (surface_loop.seaice_to_atmgrid).
  DLAKE = MWL/(RHOW*FLAKE*AXYP) (0 without lake).  GTEMPR1 = atmocn GTEMPR (ocean and lake cells; the default 273.15 elsewhere).
  GTEMPR3 = land-ice TLANDI(1)+TF (273.15 without land ice), SNOWLI = land-ice snow (0 without land ice).
  GTEMPR4 = TEARTH+TF = TBCS of the LAST GHY call (GHY_DRV.f 1388).
  BARESW = w(1,1)/(thets(1,1)*dz(1)) of the last GHY call (GHY_DRV.f 1390); SNOWD(ibv) = sum(dzsn(1:nsn)) if fr_snow(ibv) > 0.001 else 0 (GHY_DRV.f 1345-1351);
  FRSNOW(ibv) = min(snow_cover(snowbv(ibv), top_dev), fr_snow(ibv)) (GHY_DRV.f 1316-1323; snow_cover_same_as_rad = 0 in this build, SNOW_DRV.f:9; snow_cover in
  GHY_DRV.f 1982-1997) with snowbv(ibv) = snowd(ibv) = sum(wsn*fr_snow) (GHY.f 371-377); snowbv(1) is updated only if fb > 0 and snowbv(2) only if fv > 0
  (GHY_DRV.f 1297-1298): for cells with fb = 0 FRSNOW(1) is the PERSISTENT value of an earlier step and is not computable from the last step alone -> a carry
  array `snowbv_prev` is needed (not available at the first step of a run; see the ledger).
  TSAVG = the composite surface air temperature of the last SURFACE substep (S['TSAVG'] of the chained state); WSAVG = sum over the four tile types of ftype*ws of
  the last PBL substep (PBL_DRV.f 420, FLUXES avg_patches_pbl_exports).
  FLAKE, FLICE, FLAND, FEARTH : fractions (FLAKE and FEARTH change at the day boundary with daily_LAKE).

At poles only the first longitude of a pole row is a model cell; the other 71 entries of the packet are unused and not compared.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IM, JM = 72, 46
TF = 273.15
RHOW = 1000.0
SNOW_COVER_COEF = 0.15
TEENY = 1e-30
FIELDS_SURFACE = ("RSI", "ZSI", "SNOWI", "POND_MELT", "FLAG_DSWS", "FLAKE", "DLAKE", "FLICE", "FLAND", "FEARTH", "GTEMPR1", "GTEMPR2", "GTEMPR3",
                  "GTEMPR4", "TSAVG", "WSAVG", "SNOWLI", "ZSNOWI", "BARESW", "FRSNOW", "SNOWD")


def pole_mask():
    m = np.zeros((IM, JM), bool)
    m[1:, 0] = True
    m[1:, -1] = True
    return m


def radia_time_surface_state(S, st):
    """Surface state as RADIA sees it at step k: S = state at the end of step k-1 (surface_loop state dict: ocean, ice, lake, li, atm).
    Returns (ice after MELT_SI, seaice_to_atmgrid dict)."""
    import surface_loop as L
    geo = st['geo']
    ice, _melt = L.melt_si(S['ice'], S['atm']['gtemp'], S['atm']['sss'], S['atm']['mlhc'], geo)
    ag = L.seaice_to_atmgrid(ice, geo)
    return ice, ag


def ice_lake_landice_fields(S, st, ice=None, ag=None):
    """RSI ZSI SNOWI POND_MELT FLAG_DSWS FLAKE DLAKE FLICE FLAND FEARTH GTEMPR1 GTEMPR2 GTEMPR3 SNOWLI ZSNOWI from the surface state S (see module doc)."""
    geo = st['geo']
    if ice is None:
        ice, ag = radia_time_surface_state(S, st)
    fl, fw = geo['flake'], geo['fwater'] > 0
    out = dict(RSI=np.array(ice['rsi']), SNOWI=np.array(ice['snowi']), POND_MELT=np.array(ice['pond_melt']),
               FLAG_DSWS=np.asarray(ice['flag_dsws'], float))
    poice = ice['rsi'] * geo['fwater'] > 0
    out['ZSI'] = np.where(poice, ag['zsi'], 0.2)             # 0.2 = ZIMIN where no ice (value in the live packet; unused by RADIA without ice)
    out['ZSNOWI'] = np.where(poice, ag['zsnowi'], 0.0)
    out['GTEMPR2'] = np.array(ag['gtempr'])
    out['FLAKE'] = np.array(fl)
    out['DLAKE'] = np.where(fl > 0, S['lake']['mwl'] / (RHOW * np.where(fl > 0, fl, 1.0) * st['axyp']), 0.0)
    out['FLICE'], out['FLAND'], out['FEARTH'] = np.array(st['flice']), np.array(st['fland']), np.array(st['fearth'])
    out['GTEMPR1'] = np.where(fw, S['atm']['gtempr'], TF)
    fli = st['flice'] > 0
    out['GTEMPR3'] = np.where(fli, S['li']['tlandi'][..., 0] + TF, TF)
    out['SNOWLI'] = np.where(fli, S['li']['snowli'], 0.0)
    return out


def _static_baresw(g):
    """thets(1,1)*dz(1) (bare-soil layer 1) per ffg row, from the static soil columns of the record (ghy_ref.GhyColumn)."""
    import ghy_compare as GC
    import ghy_ref as G
    out = np.zeros(len(g))
    for n in range(len(g)):
        static, dyn, forc, _ent, _refs, _snowm = GC.unpack(g[n])
        col = G.GhyColumn(static, dyn, forc)
        out[n] = col.thets[1, 0] * static['dz'][0]
    return out


def land_fields(g, tbcs, w, nsn, dzsn, wsn, fr_snow, snowbv_prev=None, thets_dz=None, update_snowbv=True):
    """GTEMPR4 BARESW SNOWD FRSNOW on the (IM,JM) grid from the last GHY call of every land cell.
    g: ffg rows (cell indices, static columns: top_dev col 73, fb/fv via col 141/169); tbcs (N,), w (N,7,2), nsn (N,2) int, dzsn (N,3,2), wsn (N,3,2), fr_snow (N,2).
    snowbv_prev: (2,IM,JM) persistent snowbv (restart variable 'snowbv' at the start of a run, then carried); None -> 0.  update_snowbv=False: use snowbv_prev as is
    (the restart value at the first step).  Otherwise snowbv(ibv) is replaced by sum(wsn*fr_snow) of the END of the step where fb>0 (ibv=1) / fv>0 (ibv=2);
    measured equal to the GHY value within 1 ulp (the Fortran value is computed at the start of the last GHY call).  Returns (fields dict, snowbv (2,IM,JM))."""
    i, j = g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1
    n = len(g)
    thets_dz = _static_baresw(g) if thets_dz is None else thets_dz
    out = dict(GTEMPR4=np.full((IM, JM), TF), BARESW=np.zeros((IM, JM)), SNOWD=np.zeros((2, IM, JM)), FRSNOW=np.zeros((2, IM, JM)))
    sbv = np.zeros((2, IM, JM)) if snowbv_prev is None else np.array(snowbv_prev, float)
    fv = np.where(g[:, 169] < 1e-6, 0.0, np.where(g[:, 169] > 1 - 1e-6, 1.0, g[:, 169]))
    fb = 1.0 - fv
    top_dev = g[:, 73]
    out['GTEMPR4'][i, j] = np.asarray(tbcs) + TF
    out['BARESW'][i, j] = np.asarray(w)[:, 1, 0] / thets_dz
    for b in range(2):
        sd = np.zeros(n)
        snowd_w = np.zeros(n)
        for k in range(n):
            m = int(nsn[k, b])
            sd[k] = np.sum(dzsn[k, :m, b]) if fr_snow[k, b] > 0.001 else 0.0
            snowd_w[k] = np.sum(wsn[k, :m, b]) * fr_snow[k, b]
        out['SNOWD'][b][i, j] = sd
        upd = (fb > 0) if b == 0 else (fv > 0)
        if update_snowbv:
            sbv[b][i[upd], j[upd]] = snowd_w[upd]
        sc = np.sqrt(1000.0 * sbv[b][i, j] / (1000.0 * sbv[b][i, j] + TEENY + SNOW_COVER_COEF * top_dev))
        out['FRSNOW'][b][i, j] = np.minimum(sc, fr_snow[:, b])
    return out, sbv


def wsavg_composite(ftype, ws_tiles, blk_ij):
    """WSAVG on the grid: sum_k ftype_k * ws_k.  ftype (Ncell,4) (tile_aggregate_ff.unpack), ws_tiles (Ncell,4) with 0 for absent types, blk_ij (Ncell,2) 1-based."""
    out = np.zeros((IM, JM))
    out[blk_ij[:, 0].astype(int) - 1, blk_ij[:, 1].astype(int) - 1] = (np.asarray(ftype) * np.asarray(ws_tiles)).sum(1)
    return out


def ws_tiles_from_records(blk, ta2, la2, g2):
    """Per-cell, per-type PBL wind speed ws of substep 2 from the (recorded or our) tile records: ffs col 42 (types 1,2), ffl col 17 (land ice), ffg col 158 (land vs)."""
    cell = {(int(a), int(b)): k for k, (a, b) in enumerate(blk[:, :2])}
    ws = np.zeros((len(blk), 4))
    for r in ta2:
        ws[cell[(int(r[0]), int(r[1]))], int(r[2]) - 1] = r[42]
    for r in la2:
        ws[cell[(int(r[0]), int(r[1]))], 2] = r[17]
    for r in g2:
        ws[cell[(int(r[0]), int(r[1]))], 3] = r[158]
    return ws


def restart_land_arrays(path, g):
    """Land arrays at the start of a run from the restart file (netCDF): tearth, w_ij, nsn_ij, dzsn_ij, wsn_ij, fr_snow_ij, snowbv, tsavg, wsavg.
    Returns dict(tbcs, w, nsn, dzsn, wsn, fr_snow (rows of g), snowbv (2,IM,JM), tsavg (IM,JM), wsavg (IM,JM))."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    i, j = g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1
    rd = lambda k: np.array(d.variables[k][:])
    w = rd("w_ij")                                     # (j,i,ibv(3),layer(7))
    out = dict(tbcs=rd("tearth")[j, i],
               w=np.stack([w[j, i, 0, :], w[j, i, 1, :]], axis=2),
               nsn=rd("nsn_ij")[j, i].astype(int), dzsn=np.transpose(rd("dzsn_ij")[j, i], (0, 2, 1)),
               wsn=np.transpose(rd("wsn_ij")[j, i], (0, 2, 1)), fr_snow=rd("fr_snow_ij")[j, i],
               snowbv=np.transpose(rd("snowbv")[:, :, :2], (2, 1, 0)), tsavg=rd("tsavg").T, wsavg=rd("wsavg").T)
    d.close()
    return out
