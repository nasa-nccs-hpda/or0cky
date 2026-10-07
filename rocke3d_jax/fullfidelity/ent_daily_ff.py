"""D169 stage 3a (PARTIAL, export-relevant part only): the daily prescribed update of Ent that changes what GHY sees.

Real path (GHY_DRV.f daily_earth -> ENT_DRV.f update_vegetation_data -> ent_prescribe_vegupdate -> entcell_vegupdate,
ent_prescribed_updates.f; rundeck LAI file present, do_phenology_activegrowth=0):
  * laidata(pft): the monthly LAI file (V72x46_EntMM16_lai_trimmed_scaled_ext.nc, one variable per PFT, 12 months) read with
    timestream 'linm2m' (linear between month midpoints, model_com JDmidOfM);
  * entcell_update_lai_poolslitter: cop%LAI = laidata(cop%pft) for every cohort (plus carbon pools and litter: NOT ported here);
  * patch albedo: prescr_veg_albedo(hemi, pft of the tallest cohort, jday) from ALBVND, interpolated between seasons;
  * summarize_entcell.
NOT ported here: allom_plant_cpools / litter_cohort / litter_patch (carbon), set_vegetation_data(reinitialize=.false.) that
runs once on the first day end of every run segment (patch areas from the VEG file x (1-crops)), the crop update, the
height file, update_veg_structure, gdd/ncd bookkeeping at the day boundary (clim_stats does the latter in ent_ff).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ent_ff as E
import ent_tables_ff as T

PROD = os.environ.get("MODELE_PROD_INPUT", "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files")
LAI_FILE = f"{PROD}/V72x46_EntMM16_lai_trimmed_scaled_ext.nc"
ENT_COVER_NAMES = ["ever_br_early", "ever_br_late", "ever_nd_early", "ever_nd_late", "cold_br_early", "cold_br_late",
                   "drought_br", "decid_nd", "cold_shrub", "arid_shrub", "c3_grass_per", "c4_grass", "c3_grass_ann",
                   "c3_grass_arct", "crops_herb", "crops_woody"]
JDEND = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
JDMID = [-15, 16, 45, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350, 381]       # MODEL_COM.f:44
ALBVND = np.array(T.ALBVND_FLAT_F32, dtype=np.float32).astype(np.float64).reshape((18, 4, 6), order="F")
SEASON = [15.0, 105.0, 196.0, 288.0]


def read_lai_file(path=LAI_FILE):
    """qty[m, pft, j, i], m = 0..11."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    q = np.stack([np.array(d[n][:], dtype=np.float64) for n in ENT_COVER_NAMES], axis=1)
    d.close()
    return q


def lai_linm2m(qty, jday):
    """timestream read_stream, tinterp 'linm2m', monthly data: returns laidata[pft, j, i]."""
    jmon = 1
    while jday > JDEND[jmon]:
        jmon += 1
    imon = jmon if jday <= JDMID[jmon] else jmon + 1
    frac = float(JDMID[imon] - jday) / float(JDMID[imon] - JDMID[imon - 1])

    def month(m):                       # 1-based with cyclic extension (0 -> December, 13 -> January)
        return qty[(m - 1) % 12]
    return frac * month(imon - 1) + (1.0 - frac) * month(imon)


def prescr_veg_albedo(hemi, ncov, jday):
    """ent_prescr_veg.f prescr_veg_albedo -> albedo(6)."""
    seasn1 = -77.0
    k = 0
    found = False
    for kk in range(1, 5):
        seasn2 = SEASON[kk - 1]
        if jday <= seasn2:
            k = kk
            found = True
            break
        seasn1 = seasn2
    if not found:
        k = 1
        seasn2 = 380.0
    wt2 = (jday - seasn1) / (seasn2 - seasn1)
    wt1 = 1.0 - wt2
    if hemi == -1:
        kh1 = 1 + (k % 4)
        kh2 = 1 + ((k + 1) % 4)
    else:
        kh1 = 1 + ((k + 2) % 4)
        kh2 = k
    return np.array([wt1 * ALBVND[ncov - 1, kh1 - 1, l] + wt2 * ALBVND[ncov - 1, kh2 - 1, l] for l in range(6)])


def daily_update(cell, laidata, hemi, jday, do_giss_albedo=True):
    """Prescribed LAI and albedo for one Ent cell (see module docstring)."""
    for p in cell.patches:
        for c in p.cohorts:
            c.lai = laidata[c.pft - 1]
    for p in cell.patches:
        if p.cohorts and do_giss_albedo:
            p.albedo = prescr_veg_albedo(hemi, p.cohorts[0].pft, jday)
    E.summarize_entcell(cell)


def hemi_of_j(j):
    """lat2d(i,j) <= 0 -> -1 else +1 on the 72x46 grid (rows 1..23 are south of the equator)."""
    return -1 if j <= 23 else 1
