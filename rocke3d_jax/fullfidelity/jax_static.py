"""D181 (stage S1, part 3): the STATIC-FIELD BUILDER that replaces the D174 "template builder".

Until now every step read the real SURFACE records of THAT step (ffp 154 columns, ffs 90, ffl 60, ffg 450, fft 40) as templates and overwrote a
subset of columns; the rows (which tiles exist) and every static column came from the record.  Here the per-tile STATIC columns are built from
the restart, the topography/soil/top-index/GHG input files and analytic geometry only, in the fixed (type, IM, JM) layout of jax_state.py, for ALL
candidate cells (water cells for the ocean/ice types, FLICE > 0 for land ice, FEARTH > 0 for land), independent of the ice state.  Which of those
tiles EXIST at a step is jax_state.tile_masks(rsi, ...).

What is built (and validated against the records, tests/test_jax_static.py):
  geometry     i, j, itype, ihc, dtsurf, hemi, coriol = OMEGA2*sin(lat_j), AXYP = DXYP(j) (analytic, R^2 dlon (sin latn - sin lats)), constants ELHX
               (LHE for ocean, LHS for ice), uocean = vocean = 0, structural zeros, the unset sentinel -1e300 of columns the Fortran dump never writes
  fractions    FOCEAN and FLICE (= FGICE) from Z72X46N_gas.1_nocasp.nc, FLAKE from the restart, FLAND = 1 - FOCEAN - FLAKE, FEARTH = FLAND - FLICE (bitwise
               equal to the CONDSE-entry record on the three dates), land-ice ptype
  soil         dz, q, qk, sl (S72x460098M.ext.nc), top_index, top_dev (top_index_72x46_a.ij.ext.nc), cast to REAL*4 as the Fortran module stores them
  GHG          Ca (ppm CO2) from the CMIP6 table through drv_radcols.ghg_ca (D176)
  Ent          fb, fv, shc_can (ffg 141, 142, 168, 169), from the restart ent_state through ent_ff (host, NumPy; the existing validated unpack)
What is NOT built, with the reason, is tabulated in COLUMN_TABLE / column_report(): state-derived, atmosphere-derived, carried (PBL), radiation-derived,
Ent-derived day-varying, GHY outputs and refs, and the columns that change at the day boundary (daily_LAKE, UPDTYPE) which a single step does not cross.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IM, JM = 72, 46
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
PROD = os.environ.get("MODELE_PROD_INPUT", "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files")
RESTART = {'nov26': 'fort1_nov26_itime33312.nc', 'dec01': 'fort1_dec01_itime33552.nc', 'jan01': 'fort1_jan01_itime17520.nc'}
NCOL = dict(pbl=154, tile=90, landice=60, land=450, blk=40)
SENT = -1e300                      # value of every column the Fortran dump routines never write (srec/lrec/grec/trec initialised to -1d300)
LHE, LHS = 2.5e6, 2.834e6          # constants of ffa_consts.txt (LHE, LHS of the model constants file)
DTSRC, NISURF = 1800.0, 2
RADIUS = 6371000.0


# ---------------------------------------------------------------------------------------------------------------- input files
def _nc(path):
    import netCDF4 as nc
    return nc.Dataset(path)


def topography(prod=PROD):
    """Z72X46N_gas.1_nocasp.nc (TOPO of the rundeck): focean, flake (FLAKE0 of the file; the model uses the restart value), fgrnd, fgice, zatmo,
    hlake, as (IM, JM) float64."""
    R = _nc(prod + '/Z72X46N_gas.1_nocasp.nc')
    return {k: np.array(R.variables[k][:], dtype=np.float64).T.copy() for k in ('focean', 'flake', 'fgrnd', 'fgice', 'zatmo', 'hlake')}


def soil(prod=PROD):
    """S72x460098M.ext.nc (SOIL) and top_index_72x46_a.ij.ext.nc: dz (IM,JM,6), q, qk (IM,JM,5,6), sl (IM,JM), top_index, top_dev (IM,JM).
    The real model stores them as REAL*4 (the record values are exactly float32-representable): the arrays are returned as float64 holding the
    float32 values."""
    R = _nc(prod + '/S72x460098M.ext.nc')
    out = dict(dz=np.transpose(np.array(R.variables['dz'][:]), (2, 1, 0)), q=np.transpose(np.array(R.variables['q'][:]), (3, 2, 1, 0)),
               qk=np.transpose(np.array(R.variables['qk'][:]), (3, 2, 1, 0)), sl=np.array(R.variables['sl'][:]).T)
    T = _nc(prod + '/top_index_72x46_a.ij.ext.nc')
    out['top_index'] = np.array(T.variables['top_index'][:]).T
    out['top_dev'] = np.array(T.variables['top_dev'][:]).T
    return out


def fractions(date, ff=FF, prod=PROD):
    """Surface fractions at the start of the window: FOCEAN, FLICE from the topography, FLAKE from the restart, FLAND, FEARTH derived."""
    t = topography(prod)
    R = _nc(f"{ff}/_pristine_restarts/{RESTART[date]}")
    flake = np.array(R.variables['flake'][:], dtype=np.float64).T.copy()
    foc = t['focean']
    fland = 1.0 - foc - flake
    flice = t['fgice']
    return dict(focean=foc, flake=flake, fland=fland, flice=flice, fearth=fland - flice, hlake=t['hlake'])


def geometry():
    """Analytic latitude geometry of the 72x46 grid (GEOM_B.f): lat(j), sinlat(j), dxyp(j), hemi(j), OMEGA2 (= 2*omega, ModelE: omega = 2 pi (EDPERD+EYEARD)/(EDPERD EYEARD) / 86400
    for EDPERD = 1, EYEARD = 365 -- compared with the recorded constant in the test, not assumed)."""
    pi = np.pi
    dlon = 2.0 * pi / IM
    dlat = round(180.0 / (JM - 1)) * pi / 180.0
    fjeq = 0.5 * (1 + JM)
    j1 = np.arange(1, JM + 1, dtype=np.float64)
    lat = dlat * (j1 - fjeq)
    lat[0], lat[-1] = -2.0 * pi / 4, 2.0 * pi / 4
    latn = np.where(j1 == JM, 2.0 * pi / 4, dlat * (j1 + 0.5 - fjeq))
    lats = np.where(j1 == 1, -2.0 * pi / 4, dlat * (j1 - 0.5 - fjeq))
    dxyp = RADIUS * RADIUS * dlon * (np.sin(latn) - np.sin(lats))
    omega = 2.0 * pi * (1.0 + 365.0) / 365.0 / 86400.0
    hemi = np.where(j1 <= JM // 2, -1.0, 1.0)
    return dict(lat=lat, sinlat=np.sin(lat), dxyp=dxyp, omega2=2.0 * omega, hemi=hemi)


def ent_static(date, ff=FF):
    """fv, shc_can, ws_can, height, (IM,JM) float64 from the restart ent_state through ent_ff (host NumPy); zero where there is no Ent cell.
    ws_can and height change at the day boundary (LAI update), fv and shc_can are static in the 54-step window."""
    import ent_ff as E
    cells = E.load_restart_cells(f"{ff}/_pristine_restarts/{RESTART[date]}")
    out = {k: np.zeros((IM, JM)) for k in ('fv', 'shc_can', 'ws_can', 'height')}
    for (i, j), c in cells.items():
        ce = E.call_exports(c)
        out['fv'][i - 1, j - 1] = ce['fv']
        out['shc_can'][i - 1, j - 1] = ce['shc_can']
        out['ws_can'][i - 1, j - 1] = ce['ws_can']
        out['height'][i - 1, j - 1] = ce['height']
    return out


# ---------------------------------------------------------------------------------------------------------------- the builder
def build_static(date, ff=FF, prod=PROD, with_ent=True):
    """Static columns of the five record families in the fixed layout.  Returns dict with
         pbl (4, IM, JM, 154) tile (2, IM, JM, 90) landice (IM, JM, 60) land (IM, JM, 450) blk (IM, JM, 40)    (values of built columns, 0 elsewhere)
         built   {family: sorted list of column indices that this builder claims}   cand {family: candidate-cell masks}
         fields  the grid fields used (focean, flake, fland, flice, fearth, ...)
       Columns that the real model sets from the state of a step (not static) are NOT touched (value 0.0, not in `built`)."""
    fr = fractions(date, ff, prod)
    geo = geometry()
    so = soil(prod)
    valid = np.zeros((IM, JM), bool)
    valid[:, 1:JM - 1] = True
    valid[0, 0] = valid[0, JM - 1] = True
    fwater = fr['focean'] + fr['flake']
    cand = dict(ocean=(fwater > 0) & valid, ice=(fwater > 0) & valid, landice=(fr['flice'] > 0) & valid, land=(fr['fearth'] > 0) & valid, blk=valid)
    ii, jj = np.meshgrid(np.arange(1, IM + 1, dtype=np.float64), np.arange(1, JM + 1, dtype=np.float64), indexing='ij')
    coriol = (geo['omega2'] * geo['sinlat'])[None, :] * np.ones((IM, 1))
    hemi = geo['hemi'][None, :] * np.ones((IM, 1))
    axyp = geo['dxyp'][None, :] * np.ones((IM, 1))
    dtsurf = DTSRC / NISURF
    built = {}

    # ---- ffp (PBL record) per type
    pbl = np.zeros((4, IM, JM, 154))
    pb = [0, 1, 2, 3, 4, 10, 14, 15, 22, 36, 112]
    for t in range(4):
        pbl[t, :, :, 0], pbl[t, :, :, 1], pbl[t, :, :, 2] = ii, jj, t + 1
        pbl[t, :, :, 3] = 1.0                       # ihc
        pbl[t, :, :, 4] = dtsurf
        pbl[t, :, :, 10] = hemi
        pbl[t, :, :, 14] = pbl[t, :, :, 15] = 0.0   # uocean, vocean
        pbl[t, :, :, 22] = (fr['focean'] > 0) if t == 0 else 0.0     # pbl_args%ocean: true ocean tile (not lake)
        pbl[t, :, :, 36] = coriol
        pbl[t, :, :, 112] = 0.0                     # wspdf (PBL output, identically 0 in this configuration)
    built['pbl'] = pb

    # ---- ffs (ocean + ice tile record): type index 0 ocean, 1 ice
    tile = np.zeros((2, IM, JM, 90))
    tb = [0, 1, 2, 16, 17, 18, 25, 26, 27, 39] + list(range(53, 59)) + [78, 79] + list(range(81, 90))
    for t in range(2):
        tile[t, :, :, 0], tile[t, :, :, 1], tile[t, :, :, 2] = ii, jj, t + 1
        tile[t, :, :, 16] = LHE if t == 0 else LHS
        tile[t, :, :, 17] = tile[t, :, :, 18] = 0.0
        tile[t, :, :, 25] = dtsurf
        tile[t, :, :, 26] = fr['flake']
        tile[t, :, :, 27] = fr['focean']
        tile[t, :, :, 39] = axyp
        tile[t, :, :, 53:59] = SENT
        tile[t, :, :, 78] = tile[t, :, :, 79] = 0.0      # lim_lake_evap, lim_dew (compile-time false in this configuration)
        tile[t, :, :, 81:90] = SENT
    built['tile'] = tb

    # ---- ffl (land-ice record)
    li = np.zeros((IM, JM, 60))
    lb = [0, 1, 2, 3, 13, 24, 28, 45] + list(range(46, 60))
    li[:, :, 0], li[:, :, 1], li[:, :, 2] = ii, jj, 1.0
    li[:, :, 3] = fr['flice']                         # ptype = FLICE
    li[:, :, 13] = dtsurf
    li[:, :, 24] = 0.0                                # dskin (landice has no skin layer)
    li[:, :, 28] = SENT
    li[:, :, 45] = 0.0
    li[:, :, 46:60] = SENT
    built['landice'] = lb

    # ---- ffg (land record)
    from drv_radcols import ghg_ca
    g = np.zeros((IM, JM, 450))
    gb = [0, 1, 3, 8, 22, 29, 43] + list(range(72, 142)) + [146, 151, 164, 177, 178, 180, 194, 201, 215, 244, 258, 272, 288] + list(range(290, 299)) + [310]
    g[:, :, 0], g[:, :, 1] = ii, jj
    g[:, :, 3] = ghg_ca(1850, 182)
    g[:, :, 72] = so['top_index'].astype(np.float32).astype(np.float64)
    g[:, :, 73] = so['top_dev'].astype(np.float32).astype(np.float64)
    g[:, :, 74:80] = so['dz'].astype(np.float32).astype(np.float64)
    # q(i,j,imt,ngm), qk: record order is reshape(q_ij(i,j,1:imt,1:ngm), (/30/)) = Fortran order, imt fastest
    g[:, :, 80:110] = np.transpose(so['q'].astype(np.float32).astype(np.float64), (0, 1, 3, 2)).reshape(IM, JM, 30)
    g[:, :, 110:140] = np.transpose(so['qk'].astype(np.float32).astype(np.float64), (0, 1, 3, 2)).reshape(IM, JM, 30)
    g[:, :, 140] = so['sl'].astype(np.float32).astype(np.float64)
    g[:, :, 177] = g[:, :, 178] = SENT
    g[:, :, 290:299] = SENT                           # grec(291:299) never written
    g[:, :, 310] = 0.0                                # ffent(12,1) slot: identically 0 in the window
    # thets(1,1) (grec(289)): saturated water content of layer 1 = sum_i q(i,1) THM(0,i) over the first IMT-1 textures, same order of summation as GHY
    import ghy_ref as G
    q32 = so['q'].astype(np.float32).astype(np.float64)
    ts = np.zeros((IM, JM))
    for i in range(G.IMT - 1):
        ts = ts + q32[:, :, i, 0] * G.THM[0, i]
    g[:, :, 288] = ts
    gb_ent = []
    if with_ent:
        es = ent_static(date, ff)
        fv = es['fv']
        fv = np.where(fv < 1e-6, 0.0, fv)
        fv = np.where(fv > 1.0 - 1e-6, 1.0, fv)
        g[:, :, 142] = fv
        g[:, :, 141] = 1.0 - fv
        g[:, :, 168] = es['shc_can']
        g[:, :, 169] = es['fv']
        gb_ent = [141, 142, 168, 169]
    built['land'] = sorted(set(gb) | set(gb_ent))

    # ---- fft (per-cell composite): i, j, ftype of the land-ice patch (= FLICE), unset tail
    blk = np.zeros((IM, JM, 40))
    blk[:, :, 0], blk[:, :, 1] = ii, jj
    blk[:, :, 16] = fr['flice']
    blk[:, :, 23] = fr['fearth']                      # ftype of the land patch = FEARTH (changes at the day boundary: UPDTYPE)
    blk[:, :, 36:40] = SENT
    built['blk'] = [0, 1, 16, 23, 36, 37, 38, 39]
    return dict(pbl=pbl, tile=tile, landice=li, land=g, blk=blk, built=built, cand=cand, fields=dict(fr, valid=valid, **{'geometry': geo}))


# ---------------------------------------------------------------------------------------------------------------- comparison with records
def compare_with_records(built, rec, tol=0.0):
    """Compare the built static columns with the records of one step (dict from atm_step.surface_records), both substeps, on the rows of the
    record.  Returns {family: {column: (n_rows_compared, n_unequal, max_abs_diff)}}; a column that is built but not equal is a defect of the
    builder (or a column that changes within the step / at a day boundary)."""
    res = {}
    for fam, keys in (('pbl', ('pa', 'pb')), ('tile', ('ta', 'tb')), ('landice', ('la', 'lb')), ('land', ('g1', 'g2')), ('blk', ('blk1', 'blk2'))):
        res[fam] = {}
        for c in built['built'][fam]:
            n = ne = 0
            mx = 0.0
            for k in keys:
                r = np.asarray(rec[k])
                i, j = r[:, 0].astype(int) - 1, r[:, 1].astype(int) - 1
                if fam in ('pbl', 'tile'):
                    t = r[:, 2].astype(int) - 1
                    b = built[fam][t, i, j, c]
                else:
                    b = built[fam][i, j, c]
                d = np.abs(b - r[:, c])
                n += len(d)
                ne += int((b != r[:, c]).sum())
                mx = max(mx, float(d.max()) if len(d) else 0.0)
            res[fam][c] = (n, ne, mx)
    return res


# ---------------------------------------------------------------------------------------------------------------- column classification
# Kind of EVERY column of the five record families (0-based).  A column listed in build_static()['built'] is 'built' whatever the ranges say.
#   built        produced by build_static (static; validated bitwise against the records of the three dates, step 0)
#   state        function of the surface state (ice, lake, ocean sst, land ice): surface_loop.apply_state_to_records (existing, NumPy)
#   atm          function of the atmosphere state of the step: atm_step.cell_inputs / override_pbl / override_tiles (existing, NumPy)
#   carry        persistent PBL state of the previous substep: drv_state_cols.PBLCarry (D177; bitwise from the record carry, NOT wired)
#   rad          radiation-derived (SRHEAT, TRHR0, QSOL, TRUP, COSZ1, vis): drv_radcols / drv_zenith (D176/D177; not wired, unverified)
#   ice_thermal  sea-ice thermal columns dF1dTG, hcg1, hcg2, fsri1, fsri2: drv_ice_cols (D177)
#   land_ghy     GHY prognostic state / evaporation limits: land_chain (state carried as land_prev)
#   ent          Ent exports and Ent-side state: ent_ff (D169/D171), day-varying LAI-dependent parts need the daily Ent update
#   substep      substep-2 inputs predicted from substep-1 outputs (chain_two_substeps.predict_ns2) / PBL outputs fed to the tile fluxes
#   clock        itime, end-of-day flag, true anomaly
#   output       outputs / reference values of the real call (compared with, never an input)
#   recorded     input still taken from the record, no computed form exists (see COLUMN_NOTES)
#   index        loop index (substep number)
COLUMN_KINDS = dict(
    pbl=[(0, 5, 'built', ''), (5, 6, 'atm', 'zs1'), (6, 7, 'state', 'tgv'), (7, 8, 'atm', 'tkv'), (8, 10, 'state', 'qg_sat, qg_aver'),
         (10, 11, 'built', 'hemi'), (11, 12, 'state', 'tr4'), (12, 14, 'land_ghy', 'evap_max, fr_sat'), (14, 16, 'built', 'uocean, vocean'),
         (16, 17, 'atm', 'psurf'), (17, 18, 'rad', 'trhr0'), (18, 19, 'state', 'tg'), (19, 20, 'state', 'elhx'), (20, 21, 'rad', 'qsol'),
         (21, 22, 'state', 'sss_loc'), (22, 23, 'built', 'ocean flag'), (23, 27, 'atm', 'ddml_eq_1, gusti, tdns, qdns'), (27, 28, 'state', 'snow'),
         (28, 29, 'recorded', 'dskin input (no information, D177)'), (29, 30, 'atm', 'dbl'), (30, 31, 'recorded', 'khs input'),
         (31, 33, 'atm', 'ug, vg'), (33, 36, 'carry', 'cm, ch, cq'), (36, 37, 'built', 'coriol'), (37, 49, 'atm', 'utop..v1aa'),
         (49, 50, 'recorded', 'z0m (local overwritten by DFLUX, no information, D177)'), (50, 89, 'carry', 'u,v,t,q,e profiles'),
         (89, 154, 'output', 'PBL outputs of the real call')],
    tile=[(0, 3, 'built', ''), (3, 4, 'index', 'ns'), (4, 10, 'state', 'ptype, tg1, tg2, tr4, snow, msi2'),
          (10, 15, 'ice_thermal', 'dF1dTG, hcg1, hcg2, fsri1, fsri2'), (15, 16, 'rad', 'srheat'), (16, 19, 'built', 'elhx, uocean, vocean'),
          (19, 20, 'state', 'sss'), (20, 24, 'atm', 'ps, ma1, q1, thv1'), (24, 25, 'rad', 'trhr0'), (25, 28, 'built', 'dtsurf, flake, focean'),
          (28, 30, 'state', 'evaplim, htlim'), (30, 32, 'substep', 'e0, evapor'), (32, 33, 'atm', 'byma1'),
          (33, 37, 'state', 'ssi1, ssi2, flag_dsws, tgo'), (37, 39, 'state', 'mwl, gml'), (39, 40, 'built', 'axyp'),
          (40, 53, 'substep', 'PBL outputs us..tsv fed to the tile fluxes'), (53, 59, 'built', 'unset'), (59, 78, 'output', 'tile flux outputs'),
          (78, 80, 'built', 'lim_lake_evap, lim_dew'), (80, 81, 'rad', 'trup_in_rad'), (81, 90, 'built', 'unset')],
    landice=[(0, 4, 'built', 'i, j, ihc, ptype'), (4, 8, 'state', 'tg1, tg2, tr4, snow'), (8, 9, 'rad', 'srheat'), (9, 12, 'atm', 'ps, ma1, q1'),
             (12, 13, 'rad', 'flong'), (13, 14, 'built', 'dtsurf'), (14, 15, 'rad', 'trup_in_rad'), (15, 28, 'substep', 'PBL outputs'),
             (28, 29, 'built', 'unset'), (29, 45, 'output', 'tile outputs'), (45, 60, 'built', 'unset')],
    land=[(0, 2, 'built', 'i, j'), (2, 3, 'clock', 'itime'), (3, 4, 'built', 'Ca'), (4, 7, 'rad', 'cosz1, vis_rad, dvis'), (7, 8, 'ent', 'Qf_ij entry (carry)'),
          (8, 72, 'land_ghy', 'w, ht, nsn, dzsn, wsn, hsn, fr_snow'), (72, 143, 'built', 'top_index, top_dev, dz, q, qk, sl, fb, fv'),
          (143, 146, 'atm', 'pr, htpr, prs (CONDSE)'), (146, 147, 'built', ''), (147, 149, 'recorded', 'irrigation (demand file not read; reconstructed from the actual flux)'),
          (149, 151, 'rad', 'srheat, trheat'), (151, 152, 'built', 'fgeotherm'), (152, 163, 'atm', 'ts, qs, pres, rho, ch, qm1, vs, vs0, gusti, tprime, qprime'),
          (163, 164, 'clock', 'end_of_day_flag'), (164, 165, 'built', 'mCO2cond'), (165, 166, 'atm', 'ma1'), (166, 167, 'clock', 'true anomaly'),
          (167, 168, 'ent', 'ws_can (LAI, day-varying)'), (168, 170, 'built', 'shc_can, fv(Ent)'), (170, 177, 'ent', 'height, albedo(6) (day-varying)'),
          (177, 179, 'built', 'unset'), (179, 299, 'output', 'GHY outputs and references, thets, ffnit, unset slots'),
          (299, 450, 'ent', 'Ent per-iteration exports (13 x 11) and overflow slots')],
    blk=[(0, 2, 'built', 'i, j'), (2, 3, 'state', 'ftype ocean = (1-RSI) FWATER'), (3, 9, 'output', 'ocean patch fluxes'), (9, 10, 'state', 'ftype ice = RSI FWATER'),
         (10, 16, 'output', 'ice patch fluxes'), (16, 17, 'built', 'ftype landice = FLICE'), (17, 23, 'output', 'landice patch fluxes'),
         (23, 24, 'built', 'ftype land = FEARTH'), (24, 36, 'output', 'land patch fluxes and composites'), (36, 40, 'built', 'unset')],
)
# built columns whose value changes at the day boundary (daily_LAKE / UPDTYPE not ported): valid only inside the first model day
DAY_VARYING = dict(tile=[26], blk=[23])
COLUMN_NOTES = {
    ('pbl', 30): 'khs input of PBL: no override in atm_step, source not identified (recorded)',
    ('land', 147): 'irrigation demand file not read (IRRIGMOD); reconstructed from the recorded actual flux (D164/D170)',
}


def column_kinds(family, built_cols=None):
    """List of the kind of every column of a family; columns in built_cols (default: the builder's) are 'built'."""
    n = NCOL[family]
    kinds = [None] * n
    for lo, hi, k, _ in COLUMN_KINDS[family]:
        for c in range(lo, hi):
            kinds[c] = k
    if built_cols is not None:
        for c in range(n):
            if c in set(built_cols):
                kinds[c] = 'built'
            elif kinds[c] == 'built':
                kinds[c] = 'MISSING'
    return kinds


def column_summary(built):
    """{family: {kind: count}} for the five families given build_static()['built']."""
    out = {}
    for fam in NCOL:
        ks = column_kinds(fam, built['built'][fam])
        d = {}
        for k in ks:
            d[k] = d.get(k, 0) + 1
        out[fam] = d
    return out


def tile_statics(fields):
    """Dict (focean, flake, flice, fearth, valid) for jax_state.tile_masks from build_static()['fields']."""
    return {k: fields[k] for k in ('focean', 'flake', 'flice', 'fearth', 'valid')}


def static_pytree(built):
    """The built static arrays as a pytree of jax.Arrays (constants closed over / passed beside the state): family arrays + (4, IM, JM) candidate masks."""
    import jax.numpy as jnp
    return dict(pbl=jnp.asarray(built['pbl']), tile=jnp.asarray(built['tile']), landice=jnp.asarray(built['landice']), land=jnp.asarray(built['land']),
                blk=jnp.asarray(built['blk']),
                cand=jnp.asarray(np.stack([built['cand']['ocean'], built['cand']['ice'], built['cand']['landice'], built['cand']['land']])),
                fractions={k: jnp.asarray(built['fields'][k]) for k in ('focean', 'flake', 'fland', 'flice', 'fearth', 'valid')})
