"""D208: NumPy port of the land side of the daily lake update, GHY_DRV.f `update_land_fractions` (lines 4367-4645; called from daily_EARTH, GHY_DRV.f:3287, at the
end of every model day after daily_LAKE when variable_lk > 0).  Moves water and heat between the lake (underwater fraction, GHY index ibv = 3) and the soil
(bare = 1, vegetated = 2) in every cell whose lake fraction FLAKE changed at the day boundary:
  * lake shrank (flake < svflake): the underwater state of the freed fraction is mixed into the soil fractions (area weighted), canopy water of the vegetated
    fraction and the snow fractions are diluted by (1 - dfrac/fearth);
  * lake expanded (flake > svflake): soil water and heat of the new underwater area (dfrac*(fb*w1 + fv*w2)) plus the lake share sum_water = DMWLDF*dfrac/rhow with the
    heat DGML*BYAXYP/sum_water per m3 are put into the underwater fraction, layer by layer from the bottom (k = ngm..1), limited by the storage dfrac*w_stor(k)
    (layer 1 unlimited); canopy water is dumped into layer 1; the snow fractions are scaled by (1 + dfrac/fearth), capped at 0.95 where fb or fv is 0.
The statement order and the expressions of the Fortran are kept (scalar loop over the cells; arithmetic in float64 IEEE order, the Fortran is compiled with
-O2 -fp-model strict -assume protect_parens).  Bitwise equality with the verbatim Fortran subroutine compiled standalone: see land_fractions_harness.py.

Layout (one row per land cell, N rows; Fortran index k = 0..6 -> axis 1, ibv 1..3 -> axis 2 index 0..2):
  w, ht (N, 7, 3)   w_ij / ht_ij (slot 0 = canopy; ibv 3 = underwater fraction)     fr_snow (N, 2)   fr_snow_ij(1:2)
  dz (N, 6), q (N, 5, 6) [texture m, layer k]   fv (N,) Ent vegetated fraction (after the 1e-6 round-off rule of get_fb_fv)   thm0 (4,) = thm(0, 1:4)
  flake, svflake, fearth (the NEW FEARTH, after daily_LAKE), focean, dmwldf, dgml, byaxyp (N,).
NOT ported (reason): water tracers (not compiled in), `snow_cover` into atmlnd%fr_snow_rad (a radiation input; radiation is replayed from the records), and
`set_new_ghy_cells_outputs` (qg_ij, tearth, tsns, wearth, aiearth of cells that had no land before: not part of the carried land state; the rows where the Fortran would
call it with an effect are listed in info['new_cell_rows']), `remove_extra_snow_to_ocean` (wsn_max = 0 in the rundeck: off).
"""
import numpy as np

NGM, IMT = 6, 5
RHOW = 1000.0
EPS = 1.0e-12                      # set_new_ghy_cells_outputs: real*8 :: EPS=1.d-12


def fv_from_ent(fv_raw):
    """get_fb_fv (GHY_DRV.f:4881): fv from Ent with the round-off rule; fb = 1 - fv."""
    fv = np.array(fv_raw, dtype=np.float64, copy=True)
    fv = np.where(fv > 1.0 - 1.0e-6, 1.0, fv)
    fv = np.where(fv < 1.0e-6, 0.0, fv)
    return fv


def update_land_fractions(S, thm0):
    """S: dict of the arrays listed in the module docstring (not modified).  Returns (out, info): out = dict(w, ht, fr_snow) updated copies; info = counters, the row
    masks 'shrunk'/'expanded', 'dw_lake' ... (see code), 'new_cell_rows'."""
    w = np.array(S['w'], dtype=np.float64, copy=True)
    ht = np.array(S['ht'], dtype=np.float64, copy=True)
    fr_snow = np.array(S['fr_snow'], dtype=np.float64, copy=True)
    n = w.shape[0]
    thm0 = np.asarray(thm0, dtype=np.float64)
    fv_a = np.asarray(S['fv'], dtype=np.float64)
    flake_a, sv_a, fe_a, fo_a = (np.asarray(S[k], dtype=np.float64) for k in ('flake', 'svflake', 'fearth', 'focean'))
    dm_a, dg_a, by_a = (np.asarray(S[k], dtype=np.float64) for k in ('dmwldf', 'dgml', 'byaxyp'))
    dz_a, q_a = np.asarray(S['dz'], dtype=np.float64), np.asarray(S['q'], dtype=np.float64)
    shrunk = np.zeros(n, bool)
    expanded = np.zeros(n, bool)
    dwlake_used = np.zeros(n)            # total lake water put into the underwater fraction (expansion), m
    new_rows = []
    for r in range(n):
        focean, flake, svflake, fearth = fo_a[r], flake_a[r], sv_a[r], fe_a[r]
        if focean >= 1.0:
            continue
        if svflake == flake:
            continue
        fv = fv_a[r]
        fb = 1.0 - fv
        if flake < svflake:                                      # lake shrunk
            dfrac = svflake - flake
            dfrac = min(dfrac, fearth)
            for k in range(1, NGM + 1):
                dw = dfrac * w[r, k, 2]
                dht = dfrac * ht[r, k, 2]
                for ibv in (0, 1):
                    w[r, k, ibv] = (w[r, k, ibv] * (fearth - dfrac) + dw) / fearth
                    ht[r, k, ibv] = (ht[r, k, ibv] * (fearth - dfrac) + dht) / fearth
            w[r, 0, 1] = (w[r, 0, 1] * (fearth - dfrac)) / fearth
            ht[r, 0, 1] = (ht[r, 0, 1] * (fearth - dfrac)) / fearth
            for ibv in (0, 1):
                fr_snow[r, ibv] = fr_snow[r, ibv] * (1.0 - dfrac / fearth)
            shrunk[r] = True
        elif flake > svflake:                                    # lake expanded
            dz = dz_a[r]
            q = q_a[r]
            w_stor = np.zeros(NGM + 1)
            for k in range(1, NGM + 1):
                w_stor[k] = 0.0
                for m in range(IMT - 1):
                    w_stor[k] = w_stor[k] + q[m, k - 1] * thm0[m] * dz[k - 1]
            w_stor[0] = 0.0                                      # no underlake water in canopy
            w_stor[1] = 1.0e30                                   # any amount of water in the upper soil layer
            dfrac = flake - svflake
            sum_water = dm_a[r] * dfrac / RHOW
            if sum_water > 1.0e-30:
                ht_per_m3 = dg_a[r] * by_a[r] / sum_water
            else:
                ht_per_m3 = 0.0
            for k in range(NGM, 0, -1):                          # do not loop over canopy
                dw_soil = dfrac * (fb * w[r, k, 0] + fv * w[r, k, 1])
                dw = min(dfrac * w_stor[k], sum_water + dw_soil)
                dw_lake = dw - dw_soil
                dht_soil = dfrac * (fb * ht[r, k, 0] + fv * ht[r, k, 1])
                dht_lake = dw_lake * ht_per_m3
                dht = dht_soil + dht_lake
                sum_water = sum_water - dw_lake
                dwlake_used[r] += dw_lake
                w[r, k, 2] = (w[r, k, 2] * svflake + dw) / flake
                ht[r, k, 2] = (ht[r, k, 2] * svflake + dht) / flake
            dw = dfrac * (fv * w[r, 0, 1])                       # dump canopy water into the first layer
            dht = dfrac * (fv * ht[r, 0, 1])
            w[r, 1, 2] = w[r, 1, 2] + dw / flake
            ht[r, 1, 2] = ht[r, 1, 2] + dht / flake
            if fearth <= 0.0:
                raise RuntimeError(f'update_land_fractions: fearth<=0 (row {r}: focean {focean}, fearth {fearth}, flake {flake}, svflake {svflake})')
            for ibv in (0, 1):
                fr_snow[r, ibv] = fr_snow[r, ibv] * (1.0 + dfrac / fearth)
            if fb <= 0.0:
                fr_snow[r, 0] = min(0.95, fr_snow[r, 0])
            if fv <= 0.0:
                fr_snow[r, 1] = min(0.95, fr_snow[r, 1])
            if fr_snow[r, 0] > 1.0 or fr_snow[r, 1] > 1.0:
                raise RuntimeError(f'update_land_fractions: fr_snow_ij > 1 (row {r}: {fr_snow[r]})')
            expanded[r] = True
        # set_new_ghy_cells_outputs (not applied): acts where dfrac = svflake - flake and fearth >= EPS and fearth - dfrac <= EPS
        d2 = svflake - flake
        if not (fearth < EPS or fearth - d2 > EPS):
            new_rows.append(r)
    info = dict(n_shrunk=int(shrunk.sum()), n_expanded=int(expanded.sum()), shrunk=shrunk, expanded=expanded, dw_lake=dwlake_used, new_cell_rows=np.array(new_rows, dtype=int))
    return dict(w=w, ht=ht, fr_snow=fr_snow), info


def rows_from_ffg(g_rows):
    """dz, q, fv (with the get_fb_fv rule) from ffg rows (450 doubles per land cell): dz = cols 75-80, q = 81-110 (5x6, Fortran order), Ent fv = col 170 (1-based)."""
    g = np.asarray(g_rows, dtype=np.float64)
    dz = g[:, 74:80].copy()
    q = np.stack([x[80:110].reshape(5, 6, order='F') for x in g])
    fv = fv_from_ent(g[:, 169])
    return dz, q, fv

