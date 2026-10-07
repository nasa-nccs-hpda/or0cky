"""D176: the radiation-derived columns of the SURFACE tile records, computed from OUR state plus the radiation-server outputs.

Until D175 the coupled path read the real SURFACE tile records of each step (ffp/ffs/ffl/ffg) as templates and kept from them the columns that the
real model derives from the RADIA outputs.  The real code (read for this entry):

  RAD_DRV.f 4437-4451 (end of RADIA, radiation steps only) stores the model arrays
        FSF(1..4)   = FSRNFG(1,3,4,2)            ocean, ocean ice, land ice, soil   (server output FSF)
        TRSURF(1..4)= STBO*GTEMPR_type**4        (server output TRSURF)
        TRHR(0,I,J) = STBO*sum(P_type*GTEMPR_type**4) - TRNFLB(1)   (server output TRHR(0))
        SRVISSURF, FSRDIR                       (server outputs)
     and these arrays are NOT recomputed on the steps in between (they are model state, RAD_COM.f; they are in the restart).
  ATM_DRV.f 336-350 (atm_phase1_exports): flong = TRHR(0), fshort(type) = FSF(type), trup_in_rad(type) = TRSURF(type), cosz1 = COSZ1.
  SURFACE.f 508/575, SURFACE_LANDICE.f 278, GHY_DRV.f 1061:  SRHEAT = FSF(type)*COSZ1;  PBL_DRV.f 160: QSOL = fshort*cosz1;
  PBL_DRV.f 192: trhr0 = flong;  GHY_DRV.f 1151: trheat = flong;  GHY_DRV.f 1191-1193: vis_rad = SRVISSURF*COSZ1*.82 (the Fortran literal .82 is
  REAL*4: 0.8199999928474426), direct_vis_rad = vis_rad*FSRDIR, cos_zen_angle = COSZ1;  GHY_DRV.f 1176-1187 / RAD_DRV.f 1402: Ca = CO2ppm =
  FULGAS(CO2)*XREF(1) = the GHG-table CO2 interpolated by GTREND (GHGMOD.f 1068) at year ghg_yr (= master_yr = 1850 in the rundeck) and day 182.
  GHY_DRV.f 1422: land TRUP_in_rad = TRSURF(4).

BETWEEN radiation steps FSF and TRSURF are changed by RESET_SURF_FLUXES (RAD_DRV.f 5497-5535) whenever the ice fraction of a cell changes:
    ice grows (RSI_new > RSI_old):  FSF(2) = (FSF(2)*RSI_old + FSF(1)*(RSI_new-RSI_old)) / RSI_new          (call (1,2,RSI_old,RSI_new))
    ice shrinks (RSI_new < RSI_old): FSF(1) = (FSF(1)*(1-RSI_old) + FSF(2)*((1-RSI_new)-(1-RSI_old))) / (1-RSI_new)  (call (2,1,1-RSI_old,1-RSI_new))
  and the same for TRSURF.  The calls (SEAICE_DRV.f 1716-1815 seaice_to_atmgrid for ocean cells; inline in MELT_SI (503) and FORM_SI (1011) for LAKE cells)
  happen at these points of a model step:
      OCEAN cells (FOCEAN>0):  (a) after MELT_SI at the head of the step   RSISAVE (= end of previous step)  -> RSI after melt
                               (b) after FORM_SI of the ocean driver       RSI at DYNSI entry (= after melt)    -> RSI at ADVSI entry   (OCN_DRV.f)
                               (c) after ADVSI (atm_phase2)                RSI at ADVSI entry                   -> RSI at the end of the step
      LAKE cells (FLAKE>0, FOCEAN=0): (a) MELT_SI at the head of the step: RSI before -> after melt; (F) FORM_SI of the ocean driver: after melt -> after form.
  FSF(1)/TRSURF(1) of lake cells also change at the day boundary (daily_LAKE, FLAKE changes: RESET_SURF_FLUXES(4,1,...)): NOT ported here (see the ledger).

What this module does (all new code, nothing existing modified):
  * ghg_ca()                      : Ca from the GHG table (bitwise equal to the recorded Ca).
  * reset_surf_fluxes_ice()       : one RESET_SURF_FLUXES leg for the ice fraction of a set of cells, vectorised on (4,IM,JM).
  * class RadSurf                 : the model arrays FSF, TRSURF, TRHR0, SRVISSURF, FSRDIR; set from a radiation-server output; ice legs; validity.
  * fill_ffs / fill_ffp / fill_ffl / fill_ffg : overwrite the radiation columns of the tile record arrays (copies) from a RadSurf and COSZ1.
  * land_trup()                   : the land TRUP_in_rad (TRSURF(4) at the ffg cells), replacing land_chain.infer_trup.
  * install()                     : context manager that routes atm_step.surface_records and land_chain.infer_trup through the above (no file edited).

COSZ1 is an INPUT here: on radiation steps it is the server output COSZ1; on the other steps it is not computed here (orbit/hour angle, task D177).
"""
import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IM, JM = 72, 46
STBO = 5.67037320999999984e-8
F82 = float(np.float32(0.82))                  # the REAL*4 literal .82 of GHY_DRV.f:1191
GHG_FILE = os.environ.get("MODELE_PROD_INPUT", "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files") + "/GHG.CMIP6.1-2014.txt"

# ---- record column maps (0-based), from surface_tile_ff.IN, landice_tile_ff.IN, pbl_ff.unpack_records (1-based there), ghy_compare.py
FFS = dict(i=0, j=1, itype=2, srheat=15, trhr0=24, trup=80)
FFP = dict(i=0, j=1, itype=2, trhr0=17, qsol=20)
FFL = dict(i=0, j=1, srheat=8, flong=12, trup=14)
FFG = dict(i=0, j=1, Ca=3, cosz1=4, vis_rad=5, dvis=6, srheat=149, trheat=150)


def ghg_ca(ghg_yr=1850, ghg_day=182, path=GHG_FILE):
    """Ca (ppm CO2) as the real model sees it: GTREND (GHGMOD.f:1068) on the CMIP6 GHG table at TNOW = ghg_yr + (ghg_day-0.999)/366.
    FULGAS(CO2)*XREF(1) = (XNOW/XREF)*XREF with XNOW = XREF (same time) = XNOW.  ghg_yr = master_yr = 1850 (rundeck); ghg_day = 182
    reproduces the recorded Ca bitwise (the declaration of the default ghg_day was not located in the source; stated, not assumed)."""
    yrs, co2 = [], []
    for ln in open(path).read().splitlines()[4:]:
        w = ln.split()
        if len(w) >= 2 and w[0].lstrip("-").isdigit():
            yrs.append(int(w[0])); co2.append(float(w[1]))
    ghgyr1, ghgyr2 = yrs[0], yrs[-1]
    tnow = ghg_yr + (ghg_day - 0.999) / 366.0
    year = tnow
    if tnow <= ghgyr1 + .5:
        year = ghgyr1 + .5
    if tnow >= ghgyr2 + .49999:
        year = ghgyr2 + .49999
    dy = year - (ghgyr1 + .5)
    iy = int(dy)
    frac = dy - iy
    return co2[iy] + frac * (co2[iy + 1] - co2[iy])


def reset_surf_fluxes_ice(F, rsi_old, rsi_new, mask):
    """One RESET_SURF_FLUXES leg for a change of the ice fraction rsi_old -> rsi_new on the cells of `mask`.
    F: (4,IM,JM) FSF or TRSURF (a copy is returned).  Only the NEW type is changed (RAD_DRV.f 5527)."""
    rsi_old, rsi_new = np.asarray(rsi_old, float), np.asarray(rsi_new, float)
    up = (rsi_new > rsi_old) & mask
    dn = (rsi_new < rsi_old) & mask
    out = F.copy()
    out[1] = np.where(up, (F[1] * rsi_old + F[0] * (rsi_new - rsi_old)) / np.where(up, rsi_new, 1.0), F[1])
    fo, fn = 1.0 - rsi_old, 1.0 - rsi_new
    out[0] = np.where(dn, (F[0] * fo + F[1] * (fn - fo)) / np.where(dn, fn, 1.0), F[0])
    return out


def reset_surf_fluxes(F, itype_old, itype_new, f_orig, f_now, mask):
    """RESET_SURF_FLUXES(I,J,ITYPE_OLD,ITYPE_NEW,FTYPE_ORIG,FTYPE_NOW) verbatim (RAD_DRV.f 5524-5532), 1-based types, on the cells of `mask`:
    F(new) = (F(new)*f_orig + F(old)*(f_now-f_orig)) / f_now.  Returns a copy of F."""
    f_orig, f_now = np.asarray(f_orig, float), np.asarray(f_now, float)
    out = F.copy()
    ok = mask & (f_now != 0)
    out[itype_new - 1] = np.where(ok, (F[itype_new - 1] * f_orig + F[itype_old - 1] * (f_now - f_orig)) / np.where(ok, f_now, 1.0), F[itype_new - 1])
    return out


def daily_lake_rsi(flake_old, flake_new, rsi_old):
    """RSI of a lake cell right after daily_LAKE (LAKES.f): the ice mass is conserved on the new lake fraction, RSI = PLKIC/FLAKEnew (RSI=1 and the ice
    'crunched up' when PLKIC > FLAKEnew; 0 when the lake disappears).  Only the fraction part of daily_LAKE (the new FLAKE itself is an input).
    daily_LAKE changes RSI WITHOUT a RESET_SURF_FLUXES call for it (only the calls at the end of the cell block, see RadSurf.daily_lake): the melt leg of the
    next step must therefore start from THIS value, not from the end-of-step value."""
    fo, fn, r = np.asarray(flake_old, float), np.asarray(flake_new, float), np.asarray(rsi_old, float)
    plkic = fo * r
    out = np.where(fn > 0, np.minimum(plkic / np.where(fn > 0, fn, 1.0), 1.0), 0.0)
    return np.where(fn == fo, r, out)


class RadSurf:
    """The RAD_COM model arrays that SURFACE reads: FSF(4,IM,JM), TRSURF(4,IM,JM), TRHR(0,IM,JM), SRVISSURF, FSRDIR (frozen between radiation steps)."""

    def __init__(self):
        self.FSF = self.TRSURF = self.TRHR0 = self.SRVIS = self.FSRDIR = None
        self.cosz1_server = None
        self.n_legs = 0

    @property
    def ready(self):
        return self.FSF is not None

    def from_server(self, out):
        """Radiation step: take the server outputs (radiation_server OUTPUT_FIELDS)."""
        self.FSF = np.array(out["FSF"], float)
        self.TRSURF = np.array(out["TRSURF"], float)
        self.TRHR0 = np.array(out["TRHR"][0], float)
        self.SRVIS = np.array(out["SRVISSURF"], float)
        self.FSRDIR = np.array(out["FSRDIR"], float)
        self.cosz1_server = np.array(out["COSZ1"], float)
        return self

    def ice_leg(self, rsi_old, rsi_new, mask):
        """RESET_SURF_FLUXES for the ice fraction change rsi_old -> rsi_new on `mask` cells (FSF and TRSURF)."""
        self.FSF = reset_surf_fluxes_ice(self.FSF, rsi_old, rsi_new, mask)
        self.TRSURF = reset_surf_fluxes_ice(self.TRSURF, rsi_old, rsi_new, mask)
        self.n_legs += 1
        return self

    def reset(self, itype_old, itype_new, f_orig, f_now, mask):
        """General RESET_SURF_FLUXES on FSF and TRSURF."""
        self.FSF = reset_surf_fluxes(self.FSF, itype_old, itype_new, f_orig, f_now, mask)
        self.TRSURF = reset_surf_fluxes(self.TRSURF, itype_old, itype_new, f_orig, f_now, mask)
        self.n_legs += 1
        return self

    def daily_lake(self, flake_old, flake_new, rsi_old, rsi_new, fearth_old, fearth_new):
        """The RESET_SURF_FLUXES calls at the end of the per-cell block of daily_LAKE (LAKES.f 2917-2930), after FLAKE/FEARTH/RSI have been updated:
           lake grows  : RESET(4,1,FLAKE_OLD,FLAKE)
           lake shrinks: if PLAKE>0: RESET(1,4,FEARTH_OLD,FEARTH_OLD+PLAKE-FLAKE*(1-RSI)) ;  if PLKIC>0 and PLKIC/=FLAKE*RSI: RESET(2,4,that,FEARTH)
        with PLAKE = FLAKE_OLD*(1-RSI_old), PLKIC = FLAKE_OLD*RSI_old (LAKES.f 2595).  The new FLAKE/RSI/FEARTH are INPUTS here (daily_LAKE itself is not
        ported)."""
        flake_old, flake_new = np.asarray(flake_old, float), np.asarray(flake_new, float)
        rsi_old, rsi_new = np.asarray(rsi_old, float), np.asarray(rsi_new, float)
        fearth_old, fearth_new = np.asarray(fearth_old, float), np.asarray(fearth_new, float)
        grow = flake_new > flake_old
        self.reset(4, 1, flake_old, flake_new, grow)
        shr = flake_old > flake_new
        plake = flake_old * (1.0 - rsi_old)
        plkic = flake_old * rsi_old
        f1 = fearth_old + plake - flake_new * (1.0 - rsi_new)
        self.reset(1, 4, fearth_old, f1, shr & (plake > 0))
        self.reset(2, 4, f1, fearth_new, shr & (plkic > 0) & (plkic != flake_new * rsi_new))
        return self

    def copy(self):
        o = RadSurf()
        for k in ("FSF", "TRSURF", "TRHR0", "SRVIS", "FSRDIR", "cosz1_server"):
            v = getattr(self, k)
            setattr(o, k, None if v is None else v.copy())
        o.n_legs = self.n_legs
        return o


# ------------------------------------------------------------------------------------------------ column writers
def _ij(rec):
    return rec[:, 0].astype(int) - 1, rec[:, 1].astype(int) - 1


def fill_ffs(ta, rad, cosz1):
    """ffs (ocean/lake and sea-ice tiles, itype 1/2): srheat = FSF(itype)*COSZ1, trhr0 = TRHR(0), trup_in_rad = TRSURF(itype)."""
    t = np.array(ta, copy=True)
    i, j = _ij(t)
    ty = t[:, FFS["itype"]].astype(int) - 1
    t[:, FFS["srheat"]] = rad.FSF[ty, i, j] * cosz1[i, j]
    t[:, FFS["trhr0"]] = rad.TRHR0[i, j]
    t[:, FFS["trup"]] = rad.TRSURF[ty, i, j]
    return t


def fill_ffp(pa, rad, cosz1):
    """ffp (PBL entry, all four tile types): trhr0 = flong = TRHR(0), qsol = fshort*cosz1 = FSF(itype)*COSZ1."""
    p = np.array(pa, copy=True)
    i, j = _ij(p)
    ty = p[:, FFP["itype"]].astype(int) - 1
    p[:, FFP["trhr0"]] = rad.TRHR0[i, j]
    p[:, FFP["qsol"]] = rad.FSF[ty, i, j] * cosz1[i, j]
    return p


def fill_ffl(la, rad, cosz1):
    """ffl (land-ice tiles): srheat = FSF(3)*COSZ1, flong = TRHR(0), trup_in_rad = TRSURF(3)."""
    l = np.array(la, copy=True)
    i, j = _ij(l)
    l[:, FFL["srheat"]] = rad.FSF[2, i, j] * cosz1[i, j]
    l[:, FFL["flong"]] = rad.TRHR0[i, j]
    l[:, FFL["trup"]] = rad.TRSURF[2, i, j]
    return l


def fill_ffg(g, rad, cosz1, ca):
    """ffg (land / Ent): Ca, cosz1, vis_rad = SRVISSURF*COSZ1*.82(real*4), direct_vis_rad = vis_rad*FSRDIR, srheat = FSF(4)*COSZ1, trheat = TRHR(0)."""
    g = np.array(g, copy=True)
    i, j = _ij(g)
    cz = cosz1[i, j]
    vis = rad.SRVIS[i, j] * cz * F82
    g[:, FFG["Ca"]] = ca
    g[:, FFG["cosz1"]] = cz
    g[:, FFG["vis_rad"]] = vis
    g[:, FFG["dvis"]] = vis * rad.FSRDIR[i, j]
    g[:, FFG["srheat"]] = rad.FSF[3, i, j] * cz
    g[:, FFG["trheat"]] = rad.TRHR0[i, j]
    return g


def land_trup(g, rad):
    """Land TRUP_in_rad (GHY_DRV.f:1422) = TRSURF(4) at the ffg cells; replaces land_chain.infer_trup (which inverts a recorded patch value)."""
    i, j = _ij(g)
    return rad.TRSURF[3, i, j]


def fill_records(rec, rad, cosz1, ca=None):
    """All radiation columns of a atm_step.surface_records dict (substep 1 and 2 copies are filled identically).  Returns a new dict."""
    ca = ghg_ca() if ca is None else ca
    o = dict(rec)
    for k in ("ta", "tb"):
        o[k] = fill_ffs(rec[k], rad, cosz1)
    for k in ("pa", "pb"):
        o[k] = fill_ffp(rec[k], rad, cosz1)
    for k in ("la", "lb"):
        o[k] = fill_ffl(rec[k], rad, cosz1)
    for k in ("g1", "g2"):
        o[k] = fill_ffg(rec[k], rad, cosz1, ca)
    return o


@contextlib.contextmanager
def install(get_state):
    """Route the coupled path's recorded radiation columns through this module (no existing file edited; module attributes swapped and restored).
    get_state(itime) -> (RadSurf, cosz1 (IM,JM)) for the step about to run.  Patches atm_step.surface_records (wrapper that overwrites the
    radiation columns) and land_chain.infer_trup (TRSURF(4) instead of the value inferred from the recorded land patch).
    The wrapper is keyed by the R.itime of the record being read."""
    import atm_step as A
    import land_chain as LC
    orig_rec, orig_trup = A.surface_records, LC.infer_trup
    cur = {}

    def wrapped(R):
        rec = orig_rec(R)
        rad, cosz1 = get_state(R.itime)
        cur["rad"] = rad
        return fill_records(rec, rad, cosz1)

    def trup(g, patch_dth1, dtsurf):
        return land_trup(g, cur["rad"])

    A.surface_records, LC.infer_trup = wrapped, trup
    try:
        yield
    finally:
        A.surface_records, LC.infer_trup = orig_rec, orig_trup
