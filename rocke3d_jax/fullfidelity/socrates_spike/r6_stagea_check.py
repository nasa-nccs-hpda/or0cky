"""R6 stage A check: radia_cols.py (RADIA / RCOMPX set-up) against the arguments of run_planet_rad (tag 50) and the RCOMPX records (tag 60-62)
of the real Fortran, for every column of one radiation step.   JAX_PLATFORMS=cpu python r6_stagea_check.py cols_k.npz ITIME [n_columns]
The recorded input packet of the step (nov26_day/rsv_n26_<ITIME>_in.bin) supplies the model state; the daily products (ozone column, aerosol /
dust / volcanic columns, cloud heterogeneity, vegetation fractions, GHG factors) come from the dump itself (tag 60 / 61 / 62)."""
import sys, os
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import radiation_server as rs
import planet_side as P
import radia_cols as R

z = np.load(sys.argv[1]); itime = int(sys.argv[2])
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else None
pkt = rs.read_packet("%s/nov26_day/rsv_n26_%d_in.bin" % (rs.FF, itime))
ii, jj = R.column_index()
N = len(ii) if nmax is None else min(nmax, len(ii))
ii, jj = ii[:N], jj[:N]
c50 = P.unpack(z["p50"][:N], P.PACK50)
c60 = P.unpack(z["p60"][:N], P.PACK60)
J = lambda a: jnp.asarray(a)
g3 = lambda name: pkt[name][ii - 1, jj - 1, :]                         # (IM, JM, LM) -> (N, LM)
gl = lambda name: pkt[name][:, ii - 1, jj - 1].T                        # (LM, IM, JM) -> (N, LM)
g2 = lambda name: pkt[name][ii - 1, jj - 1]                             # (IM, JM) -> (N,)


def rep(name, mine, ref, sel=None):
    mine = np.asarray(mine); ref = np.asarray(ref)
    if sel is not None:
        mine, ref = mine[sel], ref[sel]
    assert mine.shape == ref.shape, (name, mine.shape, ref.shape)
    d = np.abs(mine - ref)
    rel = d / np.maximum(np.abs(ref), 1e-300)
    print("%-26s n=%-9d bitwise %7.3f %%  max abs %.3e  max rel %.3e" % (name, mine.size, 100.0 * float((mine == ref).mean()), d.max() if d.size else 0.0,
                                                                    rel.max() if d.size else 0.0))


# ---- static products of the dump
st = {k: z["static_%d" % k] for k in (170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180)}
tab = dict(gtau=st[174].reshape(51, 11, 143, order="F"), salbtg=st[175], taugsa=st[176], tautgs=st[177])
albvnh = st[180].reshape(12, 6, 2, order="F")

# ---- vertical set-up
cs = R.column_state(J(g3("T")), J(g3("Q")), J(gl("PK")), J(gl("PMID")), J(gl("PEDN")), J(gl("RQT")), st[171], st[172])
rep("plb (tag60)", cs["plb"], c60["plb"])
rep("tlm (tag60)", cs["tlm"], c60["tlm"])
rep("shl (tag60)", cs["shl"], c60["shl"])
rep("rhl (tag60)", cs["rhl"], c60["rhl"])
rep("p_level (tag50)", cs["plb"][:, ::-1] * 1.0e+02, c50["p_level"])
rep("t_layer (tag50)", cs["tlm"][:, ::-1], c50["t_layer"])
rep("rel_humidity (tag50)", cs["rhl"][:, ::-1], c50["rel_humidity"])

# ---- clouds
zero3 = jnp.zeros((N, 3))
for nm_p, nm_k in (("w_cloud", "W_CLOUD"), ("frac_st_water", "FRAC_ST_WATER"), ("frac_st_ice", "FRAC_ST_ICE"), ("frac_cnv_water", "FRAC_CNV_WATER"),
                   ("frac_cnv_ice", "FRAC_CNV_ICE"), ("mix_st_water", "MIX_ST_WATER"), ("mix_st_ice", "MIX_ST_ICE"), ("mix_cnv_water", "MIX_CNV_WATER"),
                   ("mix_cnv_ice", "MIX_CNV_ICE"), ("dim_st_water", "DIM_ST_WATER"), ("dim_st_ice", "DIM_ST_ICE"), ("dim_cnv_water", "DIM_CNV_WATER"),
                   ("dim_cnv_ice", "DIM_CNV_ICE"), ("frac_area_st", "FRAC_AREA_ST"), ("frac_area_cnv", "FRAC_AREA_CNV")):
    a = jnp.concatenate([zero3, J(gl(nm_k))[:, ::-1]], axis=1)
    rep("cloud " + nm_p, a, c50[nm_p])

# ---- surface
pland = g2("FLAND")
poice = g2("RSI") * (1.0 - pland)
pocean = (1.0 - pland) - poice
plake, plice, pearth = g2("FLAKE"), g2("FLICE"), g2("FEARTH")
rep("frac_sea", pocean, c50["frac_sea"]); rep("frac_land", pearth, c50["frac_land"])
rep("frac_sea_ice", poice, c50["frac_sea_ice"]); rep("frac_land_ice", plice, c50["frac_land_ice"])
rep("t_sea", g2("GTEMPR1"), c50["t_sea"]); rep("t_sea_ice", g2("GTEMPR2"), c50["t_sea_ice"])
rep("t_land_ice", g2("GTEMPR3"), c50["t_land_ice"]); rep("t_land", g2("GTEMPR4"), c50["t_land"])
rep("wind", g2("WSAVG"), c50["wind"])
rep("soil_wetness", np.minimum(g2("BARESW"), 1.0), c50["soil_wetness"])
rep("snow_depth_land", pkt["SNOWD"][:, ii - 1, jj - 1].T, c50["snow_depth_land"])
rep("frac_land_snow", pkt["FRSNOW"][:, ii - 1, jj - 1].T, c50["frac_land_snow"])
rep("snow_depth_sea_ice", g2("ZSNOWI"), c50["snow_depth_sea_ice"])
rep("snow_amnt_sea_lake", g2("SNOWI"), c50["snow_amnt_sea_lake"])
snoage = pkt["SNOAGE"][:, ii - 1, jj - 1].T
ageSN = np.stack([snoage[:, 2], snoage[:, 0], snoage[:, 1]], axis=1)
rep("agesn (tag60)", ageSN, c60["agesn"])
zoice = np.where(poice > 0, g2("ZSI"), 0.0)
fmp = np.where(poice > 0, np.minimum(1.6 * np.sqrt(g2("POND_MELT") / R.RHOW), 1.0), 0.0)
zmp = np.where(poice > 0, np.minimum(0.8 * fmp, 0.9 * zoice), 0.0)
rep("zoice", zoice, c60["zoice"]); rep("fmp", fmp, c60["fmp"]); rep("zmp", zmp, c60["zmp"])
flags = np.where(poice > 0, g2("FLAG_DSWS"), 0.0)
rep("flags", flags, c60["flags"])
rep("zlake", g2("DLAKE"), c60["zlake"])
rep("snowoi", g2("SNOWI"), c60["snowoi"]); rep("snowli", g2("SNOWLI"), c60["snowli"])

# ---- albedo
cosz = c50["cos_zenith"]
prnb, prnx = R.getsur(J(cosz), jnp.asarray(c60["jlat"].astype(int)), jnp.asarray(c60["ilon"].astype(int)), J(ageSN), J(pocean), J(poice), J(pearth),
                      J(plice), J(plake), J(g2("DLAKE")), J(g2("GTEMPR1")), J(zoice), J(fmp), J(g2("ZSNOWI")), J(zmp), J(g2("SNOWI")),
                      J(pkt["SNOWD"][:, ii - 1, jj - 1].T), J(g2("SNOWLI")), J(pkt["FRSNOW"][:, ii - 1, jj - 1].T), J(np.minimum(g2("BARESW"), 1.0)),
                      J(g2("WSAVG")), J(c50["frac_veg"]), J(c60["dalbsn"]), J(flags), J(c60["loc_chl"]), albvnh, tab, ksialb=0)
pn = lambda a: np.transpose(np.asarray(a), (0, 2, 1)).reshape(N, 24)
rep("PRNB (tag50)", pn(prnb), c50["prnb"])
rep("PRNX (tag50)", pn(prnx), c50["prnx"])
for nm, sel in (("land", pearth > 0), ("ocean", pocean > 0), ("seaice", poice > 0), ("landice", plice > 0)):
    t = {"land": 1, "ocean": 0, "seaice": 2, "landice": 3}[nm]
    a = np.asarray(prnx)[:, :, t][sel]; b = c50["prnx"].reshape(N, 4, 6)[:, t, :][sel]
    rep("PRNX %s columns" % nm, a, b)

# ---- gases
lat46 = -94.0 + 4.0 * c60["jlat"]
u0 = c60["u0gas"].reshape(N, 13, 43)
o3col = u0[:, 2, :]
ul, u0m = R.getgas_ulgas(cs["plb"], cs["shl"], J(lat46), J(o3col), c60["fulgas"][0], c60["ppmv80"][0], c60["plb0"][0], c60["hlb0"][0],
                         J(c60["fpxco2"]), c60["ls1"].astype(int))
rep("ULGAS (tag50)", np.asarray(ul).reshape(N, 559), c50["ulgas"])
rep("U0GAS (tag60)", u0m, u0)
for k in range(13):
    rep("   ULGAS gas %2d" % (k + 1), np.asarray(ul)[:, k, :], c50["ulgas"].reshape(N, 13, 43)[:, k, :])

# ---- cloud heterogeneity product -> CLDEPS (KCLDEP = 4)
cl = c60["cldeps"]
plbm = np.asarray(cs["plb"])[:, :43]
epl = np.where((plbm > 750.0).any(axis=1), cl[np.arange(N), np.argmax(plbm > 750.0, axis=1)], 0.0)
midm = (plbm <= 750.0) & (plbm >= 430.0)
epm = np.where(midm.any(axis=1), cl[np.arange(N), np.argmax(midm, axis=1)], 0.0)
hm = plbm < 430.0
eph = np.where(hm.any(axis=1), cl[np.arange(N), np.argmax(hm, axis=1)], 0.0)
rep("CLDEPS (tag60)", R.cloudeps(cs["plb"], J(epl), J(epm), J(eph)), cl)
rep("CLDEPS (tag50)", R.cloudeps(cs["plb"], J(epl), J(epm), J(eph)), c50["cldeps"])

# ---- aerosol arrays
ata = c60["ataulx"].reshape(N, 6, 43).transpose(0, 2, 1)
dtl = z["p61"][:N].transpose(0, 1, 2)
comp, rad, mdd, fsu, ssw, slw = R.aerosol_inputs(J(ata), J(dtl), J(c60["vtaulx"]), J(c60["vreff"]), J(c60["q55h2s"]), z["p62"][0], st[178], st[179],
                                                 {k: (np.asarray(v) if v.ndim == 1 or k in ("fs8opx", "ft8opx") else v) for k, v in c60.items()})
p52, p53 = z["p52"][:N], z["p53"][:N]
rep("aer component", comp[None, :], p52[0:1, :, 0].astype(int))
rep("aer radius", rad, p52[:, :, 1])
rep("aer mass_div_dens", mdd, p53)
rep("aer frac_sulph", fsu, p52[:, :, 2])
rep("aer scaling sw", ssw, p52[:, :, 3])
rep("aer scaling lw", slw, p52[:, :, 4])
