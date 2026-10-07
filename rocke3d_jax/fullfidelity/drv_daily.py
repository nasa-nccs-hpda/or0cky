"""D178 part A: the DAY-BOUNDARY items of a multi-day run (MODELE.f:355-366 dailyUpdates after the step that ends a model day, ATM_DRV.f:1004
daily_atm), each validated against the ONE boundary available (nov26 -> nov27, step 33360, ff_data/nov26_day).

Order of the real daily routine (MODELE.f:797-806, ATM_DRV.f:1004-1047) and the state of each item here
  daily_CAL                       calendar only                                                            nothing to do
  daily_OCEAN (OCNDYN.f:1531)     GLMELT (glacial melt added to ocean), TOC2SST                            NOT done (no ocean dump across the boundary)
  DIAG5A / DIAGCA                 diagnostics only                                                         nothing to do
  daily_ATMDYN (ATMDYN_COM:438)   dry-air mass fixer, MDRYA                                               existing port (dyn_glue_ff.daily_atmdyn); MDRYA derived here
  daily_orbit                     orbit/declination for the day (COSZ1: D177 territory)                     not here
  daily_ch4ox (RAD_DRV.f:1600)    Q += xCH4 * dH2O(j,l,month) * byMA                                       COMPUTED here (dH2O file + GHG table), validated
  daily_RAD (RAD_DRV.f:1321)      RCOMPT (radiation parameters: inside the radiation package), CO2 ppm,
                                  SNOAGE aging                                                            SNOAGE aging computed + bitwise validated; RCOMPT/CO2: radiation
                                                                                                          package (never ported); CO2 (Ent's Ca) recorded
  daily_LAKE (LAKES.f:2492)       variable lake fraction FLAKE/FEARTH/FLAND, lake ice, layering             analysed; NOT implemented (see `LAKE_STATUS`)
  daily_EARTH (GHY_DRV)           Ent prescribed LAI/albedo (ent_daily_ff), GHY daily bookkeeping            LAI/albedo exist (EntLand.maybe_daily); the rest NOT done
  daily_LI (LANDICE_DRV.f:982)    land-ice daily (implicit mass/energy flux to the ocean)                   NOT done (no dump to validate)
  UPDTYPE (DIAG.f:6095)           FTYPE for the diagnostics from FOCEAN, RSI, FLAKE, FEARTH, FLICE         computed here (trivial), validated against the fft records

Everything here is a function of plain arrays; `day_boundary(driver, it)` applies the validated items to the driver's state and reports what it did.
Sources are cited per function; nothing here is a result of a model run except the validation numbers in the ledger (D178).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROD = os.environ.get("MODELE_PROD_INPUT", "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files")
DH2O_FILE = f"{PROD}/dH2O_by_CH4_monthly"
GHG_FILE = f"{PROD}/GHG.CMIP6.1-2014.txt"
GRAV = 9.80665
MB2KG = 100.0 / GRAV                       # Constants_mod: MB2KG = 100/GRAV
KG2MB = GRAV / 100.0                       # byMB2KG
IM, JM, LM = 72, 46, 40
DAYS_PER_YEAR = 365.0                      # shared/TimeConstants.F90: DAYS_PER_YEAR = INT_DAYS_PER_YEAR = 365
H2OBYCH4 = 1.0                             # rundeck P2SAoM40.R:211 and I:34
GHG_YR = 1850                              # ghg_yr defaults to master_yr (RAD_DRV.f:328); master_yr=1850 in the rundeck I file (line 35)
PSF = 984.0                                # AtmL40p.F90 (PlanetParams%psf of this rundeck; value checked against the recorded MDRYA)
# AtmL40p.F90:28-35 PLBOT (pratio = 1 for PSF = 984), LS1 = 24
PLBOT = np.array([984., 964., 942., 917., 890., 860., 825., 785., 740., 692., 642., 591., 539., 489., 441., 396., 354., 316., 282., 251., 223.,
                  197., 173., 150., 128., 108., 90., 73., 57., 43., 31., 20., 10., 5.62, 3.16, 1.78, 1., .562, .316, .178, .1])
LS1 = 24

LAKE_STATUS = """daily_LAKE (LAKES.f:2492-3010, variable_lk=1 in the rundeck) changes FLAKE/FEARTH/FLAND in 632 cells at the nov26->nov27 boundary
(max 5.0e-3 of the cell; cse_in 33359 vs 33360) and RSI in 834 cells.  NOT implemented: it needs DMWLDF (the soil-saturation deficit GHY exports,
kg/m2) and TANLK (lake geometry), neither of which is in the dumps or the restart readers used so far; the branch structure (conical lake
cubic root, ice crunch, layering, RESET_SURF_FLUXES, tracer-less) is ~500 lines.  The minimal extra dump to validate a port: at the first
step of a day, per lake cell, DMWLDF, TANLK, FLAKE/FEARTH/FLAND/RSI/MSI/SNOWI/HSI(4)/MWL/GML/TLAKE/MLDLK/T2Lbot before and after daily_LAKE."""


# ------------------------------------------------------------------------------------------------------------------ MDRYA
def mdrya():
    """MDRYA = PSF*MB2KG (AtmL40p.F90:106), the dry-air mass the daily fixer restores (kg/m2).  Equal to the recorded value of the real
    DAILY_ATMDYN dump (ffd_glue_daily_33360.bin) to the last bit: 10034.007535702814."""
    return PSF * MB2KG


# ------------------------------------------------------------------------------------------------------------------ GHG table
def ghg_table(path=GHG_FILE):
    """GHGMOD.f:1103 GHGHST: the table rows (year, CO2, N2O, CH4, CFC11, CFC12, others) after 4 header lines + the column-name line;
    returns (ghgyr1, ghgam (nghg=6, nyear)); -999 -> value of the previous year."""
    rows = []
    with open(path) as f:
        lines = f.read().split("\n")
    # header: 3 text lines + units line = nhead 4 lines (GHGHST reads nhead+1 lines to find the first data line)
    for ln in lines[4:]:
        if not ln.strip():
            continue
        v = ln.split()
        rows.append([float(x) for x in v])
    rows = np.array(rows)
    yr1 = int(rows[0, 0])
    am = rows[:, 1:7].T.copy()
    for n in range(1, am.shape[1]):
        for k in range(am.shape[0]):
            if am[k, n] < 0.0:
                am[k, n] = am[k, n - 1]
    return yr1, am


def ch4_ppm(year, ghg_yr=GHG_YR):
    """RAD_DRV.f:1630-1640: CH4 of the table row iy = year - 2 - ghgyr1 + 1 (clipped to the table) times H2ObyCH4; with ghg_yr > 0 the year
    is ghg_yr (here 1850 from master_yr: the recorded water mass of the real boundary corresponds to table row 1848, CH4 = 0.808092 ppm)."""
    yr1, am = ghg_table()
    iy = year - 2 - yr1 + 1
    if ghg_yr > 0:
        iy = ghg_yr - 2 - yr1 + 1
    iy = max(iy, 1)
    iy = min(iy, am.shape[1])
    return am[2, iy - 1] * H2OBYCH4


# ------------------------------------------------------------------------------------------------------------------ dH2O (getqma)
def reference_pressure_edges():
    """PEDNL00(1:LM+1) of CALC_VERT_AMP (ATMDYN_COM.F90:194-238) with PS = PSF (STDHYB undefined), used as PLBx (RAD_DRV.f:584-588), last = 0."""
    plbot = PLBOT
    delp = plbot[:LM] - plbot[1:LM + 1]
    tropomask = np.array([1.0 if (l + 1) < LS1 else 0.0 for l in range(LM)])
    stratmask = 1.0 - tropomask
    psfmpt = PSF - plbot[LS1 - 1]
    pstrat = plbot[LS1 - 1] - plbot[LM]
    mtop = plbot[LM] * MB2KG
    mfixs = pstrat * MB2KG
    mfix = delp * stratmask * MB2KG
    mfrac = delp * tropomask / psfmpt
    mvar = PSF * MB2KG - mfixs - mtop
    pedn = np.zeros(LM + 1)
    pedn[LM] = mtop * KG2MB
    for l in range(LM - 1, -1, -1):
        ma = mfix[l] + mvar * mfrac[l]
        pedn[l] = pedn[l + 1] + ma * KG2MB
    return pedn


def getqma(path=DH2O_FILE, plb=None, dglat=None):
    """RAD_DRV.f:5667-5766: H2O production rate by CH4 oxidation, dH2O(j, l, month) in kg/m2/ppm_CH4/day, on the model layers and latitudes.
    The Fortran keeps pb, h2o, z, dz, pdn, pup, dh, w1, w2, fracl in REAL*4 (mixed with REAL*8 plb, dglat); this port keeps the same
    precision at the same places (float32 scalars, explicit float64 promotions).  Differences from ifort at the single-ulp level (powf, FMA
    contraction) are possible and are what the validation against the recorded water mass bounds."""
    f4, f8 = np.float32, np.float64
    jma, lma = 18, 24
    plb = np.array(reference_pressure_edges() if plb is None else plb, dtype=f8)
    plb[LM] = 0.0
    if dglat is None:
        dglat = -90.0 + 4.0 * np.arange(JM)
    lines = open(path).read().split("\n")
    xlat = np.array([float(x) for x in lines[2][9:100].split()], dtype=f4)
    assert len(xlat) == jma
    out = np.zeros((JM, LM, 12))
    pos = 3
    for m in range(12):
        pos += 1                                                 # month title line
        z = np.zeros(lma + 2, dtype=f4)
        h2o = np.zeros((jma + 2, lma + 2), dtype=f4)             # h2o(j, 0:lma), 1-based j
        for l in range(lma, 0, -1):
            v = [float(x) for x in lines[pos].split()]
            pos += 1
            z[l] = f4(v[0])
            for j in range(1, jma + 1):
                h2o[j, l] = f4(v[j])
        dz = np.zeros(lma + 2, dtype=f4)
        dz[1] = z[2] - z[1]
        for l in range(2, lma):
            dz[l] = f4(0.5) * (z[l + 1] - z[l - 1])
        dz[lma] = z[lma] - z[lma - 1]
        pb = np.zeros(lma + 2, dtype=f4)
        pb[0] = f4(plb[0])
        for l in range(1, lma + 1):
            pb[l] = f4(1000.0) * f4(10.0) ** (-(z[l] - f4(0.5) * dz[l]) / f4(16.0))
        pb[lma + 1] = f4(0.0)
        ldn = np.zeros(LM, int)
        lup = np.zeros(LM, int)
        for l in range(LM):                                       # model layer l+1
            while f8(pb[ldn[l] + 1]) >= plb[l] and ldn[l] < lma:
                ldn[l] += 1
            lup[l] = ldn[l]
            while f8(pb[lup[l] + 1]) > plb[l + 1] and lup[l] < lma:
                lup[l] += 1
        j2 = 2
        for j in range(JM):
            while j2 < jma and f8(dglat[j]) > f8(xlat[j2 - 1]):
                j2 += 1
            j1 = j2 - 1
            w1 = f4((f8(xlat[j2 - 1]) - dglat[j]) / f8(xlat[j2 - 1] - xlat[j1 - 1]))
            if w1 > f4(1.0):
                w1 = f4(0.5) + f4(0.5) * w1
            if w1 < f4(0.0):
                w1 = f4(0.5) * w1
            w2 = f4(1.0) - w1
            for l in range(LM):
                dh = f4(0.0)
                pdn = f4(plb[l])
                if lup[l] > 0:
                    for ll in range(ldn[l], lup[l] + 1):
                        pup = f4(max(f8(pb[ll + 1]), plb[l + 1]))
                        fracl = (pdn - pup) / (pb[ll] - pb[ll + 1])
                        dh = dh + (w1 * h2o[j1, ll] + w2 * h2o[j2, ll]) * fracl * dz[ll]
                        pdn = pup
                out[j, l, m] = 1.0e-6 * f8(dh) / 1.74 / DAYS_PER_YEAR
    return out


def ch4ox_dm(year, month, dh2o=None):
    """DM(j, l) = xCH4 * dH2O(j, l, month) (kg/m2 of water added per day), RAD_DRV.f:1640-1661."""
    dh2o = getqma() if dh2o is None else dh2o
    return ch4_ppm(year) * dh2o[:, :, month - 1]


# ------------------------------------------------------------------------------------------------------------------ SNOAGE
def snoage_age(snoage, imaxj):
    """daily_RAD (RAD_DRV.f:1405-1412), snoage_def = 0 (the default; the rundeck and its I file do not set it): for itype = 1..3 and
    i <= IMAXJ(j): snoage = 1 + .98*snoage, independent of temperature.  (The D174 inventory assumed TDIURN is needed; it is only used by
    snoage_def = 1.)  snoage (3, IM, JM)."""
    m = np.zeros((IM, JM), bool)
    for j in range(JM):
        m[:imaxj[j], j] = True
    out = np.array(snoage, copy=True)
    for t in range(3):
        out[t][m] = 1.0 + 0.98 * snoage[t][m]
    return out


# ------------------------------------------------------------------------------------------------------------------ UPDTYPE
def updtype(focean, flake, fearth, flice, rsi):
    """DIAG.f:6095-6129 FTYPE(6, IM, JM): [ITOCEAN, ITOICE, ITLANDI, ITEARTH, ITLAKE, ITLKICE] in the order of the DIAG_COM indices used
    here: returns a dict of arrays."""
    ft_oice = focean * rsi
    ft_ocean = focean - ft_oice
    ft_lkice = flake * rsi
    ft_lake = flake - ft_lkice
    return dict(ocean=ft_ocean, oice=ft_oice, lndice=flice + 0.0, earth=fearth, lake=ft_lake, lkice=ft_lkice)


# ------------------------------------------------------------------------------------------------------------------ the boundary
def day_boundary(driver, it):
    """Applied by ModelDriver at the first step of a new model day, to the atmosphere state S (end of the previous step).
    driver.daily_mode 'recorded': the existing D150 treatment (MDRYA, ch4ox water mass and aged SNOAGE from the provider's records);
    'computed': MDRYA derived, ch4ox from the dH2O file and GHG table, SNOAGE aged by the formula; the lake/land-ice/ocean/earth items that
    are not implemented are reported as such (the surface state is not updated for them).  Returns a dict describing what was applied."""
    import numpy as _np
    import atm_day_open_loop as OL
    S, ctx = driver.S, driver.ctx
    rep = dict(mode=driver.daily_mode, not_done=["daily_LAKE (FLAKE/FEARTH/FLAND, lake ice)", "daily_LI", "daily_OCEAN GLMELT",
                                                 "daily_EARTH beyond the Ent LAI/albedo update", "CO2 ppm update (radiation package)"])
    rec = driver.provider.daily_inputs(it)
    mdr = mdrya() if driver.daily_mode == "computed" else rec["mdrya"]
    rep["mdrya"] = mdr
    rep.update(OL.apply_daily(S, ctx, mdr))
    if driver.daily_mode == "computed":
        if not hasattr(driver, "_dh2o"):
            driver._dh2o = getqma()
        yr, month = date_of_itime(it)
        dm = ch4ox_dm(yr, month, driver._dh2o)
        rep["ch4ox_year_month"] = (yr, month)
    else:
        dm = rec["ch4ox_dm"]
    OL.apply_ch4ox(S, ctx, dm)
    carry = S.get("_carry", {})
    if "SNOAGE" in carry:
        if driver.daily_mode == "computed":
            sn = snoage_age(carry["SNOAGE"], ctx.imaxj)
        else:
            sn = _np.array(rec["snoage_after"], copy=True)
        rep["snoage_max_change"] = float(_np.abs(sn - carry["SNOAGE"]).max())
        carry["SNOAGE"] = sn
        if hasattr(driver.provider, "set_snoage"):
            driver.provider.set_snoage(sn)
    return rep


def date_of_itime(it, iyear1=1949, nday=48):
    """(year, month) of the model step `it` (itime counted in 30-minute steps from Jan 1 of iyear1, 365-day years)."""
    day = it // nday                       # 0-based day count since iyear1 Jan 1
    year = iyear1 + day // 365
    doy = day % 365 + 1
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    month = next(m for m in range(1, 13) if doy <= cum[m])
    return year, month


# ------------------------------------------------------------------------------------------------------------------ validation helpers
def replay_check(ff=None, daydir="nov26_day", date="nov26", it_next=33360, ctx=None):
    """DAILY_ATMDYN (MDRYA derived) + ch4ox (computed from the dH2O file and the GHG table) applied to the REAL end state of step it_next-1,
    compared with the REAL step-start state of it_next (ffa_step_<it>_a).  Also the same with the recorded MDRYA / water mass, and the
    unchanged end state (what the daily routine changes).  Returns {field: {...}} for MA PEDN PMID PK PDSIG P PEK Q."""
    import atm_step as A
    import atm_day_open_loop as OL
    import clouds_condse_io as cio
    ff = ff or cio.FF_DEFAULT
    ctx = ctx or A.make_ctx(date, imf=True, ff=ff)
    e = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it_next - 1}_e.bin")
    a = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it_next}_a.bin")
    yr, month = date_of_itime(it_next)
    out = {}
    for label, mdr, dm in (("computed", mdrya(), ch4ox_dm(yr, month)),
                           ("recorded", OL.daily_mdrya(ff, date)[0], OL.ch4ox_increment(ff, daydir, it_next)[0])):
        S = {k: np.array(e[k], copy=True) for k in ("MA", "PEDN", "PMID", "PK", "PDSIG", "P", "PEK")}
        ms_ = np.zeros(e["MA"].shape[1:])
        for l in range(LM - 1, -1, -1):
            ms_ = e["MA"][l] + ms_
        S["MASUM"] = ms_
        info = OL.apply_daily(S, ctx, mdr)
        S["Q"] = np.array(e["Q"], copy=True)
        OL.apply_ch4ox(S, ctx, dm)
        res = {}
        for k in ("MA", "PEDN", "PMID", "PK", "PDSIG", "P", "PEK", "Q"):
            d = np.abs(S[k] - a[k])
            res[k] = dict(max_abs=float(d.max()), n_diff=int((S[k] != a[k]).sum()), scale=float(np.abs(a[k]).max()),
                          change_by_daily=float(np.abs(e[k] - a[k]).max()))
        out[label] = dict(fields=res, deltam=info["deltam"], mdrya=mdr)
    return out


def snoage_check(ff=None, daydir="nov26_day", it_next=33360, ctx=None):
    """snoage_age applied to the CONDSE exit SNOAGE of step it_next-1 versus the CONDSE entry SNOAGE of it_next (bitwise?)."""
    import atm_step as A
    import clouds_condse_io as cio
    ff = ff or cio.FF_DEFAULT
    imaxj = (ctx or A.make_ctx("nov26", imf=True, ff=ff)).imaxj
    co = np.asarray(cio.read_cse(f"{ff}/{daydir}/ffc_cse_out_{it_next - 1}.bin")["SNOAGE"])
    ci = np.asarray(cio.read_cse(f"{ff}/{daydir}/ffc_cse_in_{it_next}.bin")["SNOAGE"])
    pred = snoage_age(co, imaxj)
    return dict(bitwise=bool(np.array_equal(pred, ci)), max_abs=float(np.abs(pred - ci).max()), n_changed_by_aging=int((co != ci).sum()),
                n_elements=int(ci.size))


def updtype_check(ff=None, daydir="nov26_day", it=33360):
    """UPDTYPE from the CONDSE-entry FOCEAN/FLAKE/FEARTH/FLICE/RSI of step `it` versus the FTYPE of the first-substep fft record
    (ocean, ocean ice, land ice, land types; the fft ftype columns are 2+7k of each block)."""
    import clouds_condse_io as cio
    import tile_aggregate_ff as TA
    import f3_diagnostics2 as f3b
    ff = ff or cio.FF_DEFAULT
    ci = cio.read_cse(f"{ff}/{daydir}/ffc_cse_in_{it}.bin")
    fft = TA.load(f"{ff}/{daydir}/fft_{it}.bin")
    blk1 = fft[:len(fft) // 2]
    F = f3b._grid_ftype(blk1)
    ft = updtype(ci["FOCEAN"], ci["FLAKE"], ci["FEARTH"], ci["FLICE"], ci["RSI"])
    i = blk1[:, 0].astype(int) - 1
    j = blk1[:, 1].astype(int) - 1
    res = {}
    # fft types: 0 ocean(+lake open water), 1 ocean ice(+lake ice), 2 land ice, 3 land
    for name, got, ref in (("open water (ocean+lake)", ft["ocean"] + ft["lake"], F[0]), ("ice (ocean+lake)", ft["oice"] + ft["lkice"], F[1]),
                           ("land ice", ft["lndice"], F[2]), ("land", ft["earth"], F[3])):
        d = np.abs(got - ref)[i, j]
        res[name] = dict(max_abs=float(d.max()), n_diff=int((got[i, j] != ref[i, j]).sum()), n=int(len(i)))
    return res
