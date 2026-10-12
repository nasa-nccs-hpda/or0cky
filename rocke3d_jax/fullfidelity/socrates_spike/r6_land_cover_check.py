"""frac_veg of daily_products.LandCover against the model's own (dump tag 50) on all land columns.
   python r6_land_cover_check.py products_step0.npz day0_e.npz RUNDIR DUMP.npz[:YEAR] ...   (DUMP.npz from r6_collect.py, YEAR = the master year)
Prints the largest difference and the number of columns above 1e-12; the run of report 14.8 gave 0.0 / 0 for 1850 (c1_0) and 1990 (c5_1, c5_5)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import planet_side as PS
import radia_jax as RJ
import daily_products as DP

prod = RJ.Products(np.load(sys.argv[1]), np.load(sys.argv[2]))
lc = DP.LandCover(os.path.join(sys.argv[3], "VEG"), os.path.join(sys.argv[3], "CROPS"))
for spec in sys.argv[4:]:
    f, _, y = spec.partition(":")
    fv = PS.unpack(np.load(f)["p50"], PS.PACK50)["frac_veg"]
    land = fv.sum(1) > 0
    m = lc.frac_veg(float(y or 1850), prod.ilon, prod.jlat, land)
    d = np.abs(m - fv)[land]
    print("%s year %s: %d land columns, max|diff| %.3e, columns > 1e-12: %d" % (f, y or 1850, land.sum(), d.max(), (d.max(1) > 1e-12).sum()))
