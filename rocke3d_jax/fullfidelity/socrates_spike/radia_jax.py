"""R6: RADIA (RAD_DRV.f) with SOCRATES for this rundeck, in JAX: replaces the persistent radiation server for the 21 non-AIJ output fields of one
radiation step.  Input = the 52-field RADIA input packet (radiation_server.INPUT_FIELDS) of the step; output = the fields of OUTPUT_FIELDS
(except AIJ; the 750 radiation diagnostics are not ported).

   rj = RadiaJax(specdir, products)       # products: daily / static products (ProductsNpz below), see report section 12
   out = rj(pkt, itime)                   # pkt: dict name -> array (read_packet layout); out: dict name -> array

Pipeline: column set-up and albedos and gases and aerosols (radia_cols) -> planet_rad set-up (planet_side) -> all bands (radiance_batch) ->
get_planet_radout (planet_side) -> RADIA's output assembly (below).  Units of work are columns (3170 per step: 72 x 44 + 2 poles)."""
import os, sys
import os as _os, sys as _sys
if _os.environ.get('R6_XLA_FLAGS', '1') == '1':      # XLA:CPU without FMA contraction / HLO algebraic rewrites (as the other ports, D139/D145)
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
    import clouds_jax_env  # noqa: F401  (must precede the jax import)
import numpy as np
import jax
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import planet_side as P
import radia_cols as R
import radiance_batch as RB
import spec_data as SD
import zenith_coszs as ZS

STBO = 5.67037321e-8
DTSRC = 1800.0
NRAD = 5
KAPA = (1.401 - 1.) / 1.401
SHA = P.RGAS / KAPA
BYSHA = 1. / SHA
TAULIM = 0.001
TOL_COS_ZENITH = 0.001


class Products:
    """Daily / static products of one dump (the step-0 npz of r6_collect.py).  Everything here depends on the date and on the grid box only."""

    def __init__(self, z, z_static=None):
        zs = z_static if z_static is not None else z
        N = z["p60"].shape[0]
        c60 = P.unpack(z["p60"], P.PACK60)
        c50 = P.unpack(z["p50"], P.PACK50)
        self.N = N
        self.c60 = c60
        self.o3col = c60["u0gas"].reshape(N, 13, 43)[:, 2, :]
        self.ataulx = c60["ataulx"].reshape(N, 6, 43).transpose(0, 2, 1)
        self.dtaulx = z["p61"]
        self.taucon_dust = z["p62"][0]
        self.vtaulx, self.vreff, self.q55h2s = c60["vtaulx"], c60["vreff"], c60["q55h2s"]
        self.dalbsn = c60["dalbsn"]
        self.rsdist, self.solar_constant = [float(v) for v in z["static_173"][:2]] if "static_173" in z else (None, None)
        self.loc_chl = c60["loc_chl"]
        self.frac_veg = c50["frac_veg"]
        self.fpxco2 = c60["fpxco2"]
        self.ls1 = c60["ls1"]
        self.jlat = c60["jlat"].astype(int)
        self.ilon = c60["ilon"].astype(int)
        # cloud heterogeneity (EPLOW / EPMID / EPHIG of the day, 72 x 46) at the column's 72x46 box
        self.epl = zs["static_181"][self.ilon - 1, self.jlat - 1]
        self.epm = zs["static_182"][self.ilon - 1, self.jlat - 1]
        self.eph = zs["static_183"][self.ilon - 1, self.jlat - 1]
        # scalars (global within the day)
        g = {k: c60[k][0] for k in ("fulgas", "ppmv80", "plb0", "hlb0", "fs8opx", "ft8opx", "denaer", "refdry", "frsulf", "drym2g")}
        for k in ("fstaer", "fttaer", "fsaaer", "ftaaer", "fsdaer", "ftdaer", "fsvaer", "ftvaer"):
            g[k] = float(c60[k][0])
        self.g = g
        self.solar_irrad = float(c50["solar_irrad"][0])
        self.cld_scaling = float(c50["cld_scaling"][0])
        st = {k: z["static_%d" % k] for k in (170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180)}
        self.byaml00 = st[170]
        self.plb0_req, self.shl0_req = st[171], st[172]
        self.tab = dict(gtau=st[174].reshape(51, 11, 143, order="F"), salbtg=st[175], taugsa=st[176], tautgs=st[177])
        self.redust, self.rodust = st[178], st[179]
        self.albvnh = st[180].reshape(12, 6, 2, order="F")
        self.S = P.static_from_npz(z)
        self.p52_ref = z["p52"][0]                                      # component column of the aerosol modes (static)


class RadiaJax:
    """The whole step as ONE jitted function of (packet arrays, zenith grids, per-column products).  Host work per call: three zenith grids
    (drv_zenith / zenith_coszs, exact rational orbit arithmetic) and the packet -> device transfer."""

    DAY_KEYS = ("o3col", "ataulx", "dtaulx", "dalbsn", "epl", "epm", "eph", "albvnh", "fulgas", "ppmv80", "solar_irrad")
    DAY_LEN = 48                                          # model steps per day (DTsrc = 1800 s)

    def __init__(self, specdir, static0_npz, products, zen=None, day_products=None, day0=None):
        self.spec = {1: SD.load(os.path.join(specdir, "spec_sw.npz")), 2: SD.load(os.path.join(specdir, "spec_lw.npz"))}
        self.spec_np = {"lw": np.load(os.path.join(specdir, "spec_lw.npz")), "sw": np.load(os.path.join(specdir, "spec_sw.npz"))}
        self.s0 = dict(np.load(static0_npz)) if isinstance(static0_npz, str) else static0_npz
        self.prod = products
        self.zen = zen
        self.ii, self.jj = R.column_index()
        self.N = len(self.ii)
        pr = products
        self.comp = np.array([7, 14, 30, 25, 4, 4] + [2] * pr.dtaulx.shape[2] + [6])
        self.prod_arrays = {k: jnp.asarray(v) for k, v in dict(
            o3col=pr.o3col, ataulx=pr.ataulx, dtaulx=pr.dtaulx, vtaulx=pr.vtaulx, vreff=pr.vreff, q55h2s=pr.q55h2s, dalbsn=pr.dalbsn,
            loc_chl=pr.loc_chl, frac_veg=pr.frac_veg, fpxco2=pr.fpxco2, jlat=pr.jlat, ilon=pr.ilon, epl=pr.epl, epm=pr.epm, eph=pr.eph,
            albvnh=pr.albvnh, fulgas=pr.g["fulgas"], ppmv80=pr.g["ppmv80"], solar_irrad=np.float64(pr.solar_irrad)).items()}
        # day-varying products of consecutive model days (daily_products.day_arrays), first day = model day number day0 (itime // 48)
        self.day0 = day0
        self.day_arrays = None
        if day_products is not None:
            self.day_arrays = {k: jnp.stack([jnp.asarray(d[k]) for d in day_products]) for k in self.DAY_KEYS}
        self._jit = jax.jit(self._core)

    # ---------------------------------------------------------------------------------------------------------- host side
    def __call__(self, pkt, itime, cosz=None, to_numpy=True):
        if cosz is None:
            cz2, cza = ZS.calc_zenith_angle(self.zen, itime, NRAD)
            cz1 = self.zen.cosz1(itime)
        else:
            cz2, cza, cz1 = cosz
        pj = {k: jnp.asarray(pkt[k]) for k in ("T", "Q", "PK", "PMID", "PEDN", "RQT", "FLAND", "RSI", "FLAKE", "FLICE", "FEARTH", "GTEMPR1",
                                               "GTEMPR2", "GTEMPR3", "GTEMPR4", "SNOAGE", "ZSI", "POND_MELT", "FLAG_DSWS", "SNOWD", "FRSNOW",
                                               "BARESW", "DLAKE", "ZSNOWI", "SNOWI", "WSAVG", "SNOWLI", "KLIQ", "LTROPO", "BYMA", "TAUSS", "TAUMC",
                                               "CLDSS", "CLDMC", "W_CLOUD", "FRAC_ST_WATER", "FRAC_ST_ICE", "FRAC_CNV_WATER", "FRAC_CNV_ICE",
                                               "MIX_ST_WATER", "MIX_ST_ICE", "MIX_CNV_WATER", "MIX_CNV_ICE", "DIM_ST_WATER", "DIM_ST_ICE",
                                               "DIM_CNV_WATER", "DIM_CNV_ICE", "FRAC_AREA_ST", "FRAC_AREA_CNV")}
        out = self._jit(pj, jnp.asarray(cz1), jnp.asarray(cz2), jnp.asarray(cza), self.pa_for_itime(itime))
        if to_numpy:
            out = {k: np.asarray(v) for k, v in out.items()}
        return out

    def pa_for_itime(self, itime):
        """The per-column products of the model day containing `itime` (a Python or traced int).  Without day products: the dump's day.
        Outside the window of the day products the first / last day is used (clipped; check on the host with `day_covers`)."""
        if self.day_arrays is None:
            return self.prod_arrays
        k = jnp.clip(jnp.asarray(itime, jnp.int32) // self.DAY_LEN - self.day0, 0, self.day_arrays["o3col"].shape[0] - 1)
        pa = dict(self.prod_arrays)
        pa.update({n: v[k] for n, v in self.day_arrays.items()})
        return pa

    def day_covers(self, itime):
        return self.day_arrays is None or 0 <= int(itime) // self.DAY_LEN - self.day0 < self.day_arrays["o3col"].shape[0]

    # ---------------------------------------------------------------------------------------------------------- device side
    def _inputs(self, pkt, cosz1_grid, cosz2_grid, cosza_grid, pa):
        pr = self.prod
        N = self.N
        ii, jj = jnp.asarray(self.ii - 1), jnp.asarray(self.jj - 1)
        g3 = lambda n: pkt[n][ii, jj, :]
        gl = lambda n: jnp.transpose(pkt[n][:, ii, jj])
        g2 = lambda n: pkt[n][ii, jj]
        cosza = cosza_grid[ii, jj]
        cosz2 = cosz2_grid[ii, jj]
        cosz1 = cosz1_grid[ii, jj]
        cs = R.column_state(g3("T"), g3("Q"), gl("PK"), gl("PMID"), gl("PEDN"), gl("RQT"), pr.plb0_req, pr.shl0_req)
        plb, tlm, shl, rhl = cs["plb"], cs["tlm"], cs["shl"], cs["rhl"]
        pland = g2("FLAND")
        poice = g2("RSI") * (1.0 - pland)
        pocean = (1.0 - pland) - poice
        plake, plice, pearth = g2("FLAKE"), g2("FLICE"), g2("FEARTH")
        tgo, tgoi, tgli, tge = g2("GTEMPR1"), g2("GTEMPR2"), g2("GTEMPR3"), g2("GTEMPR4")
        snoage = jnp.transpose(pkt["SNOAGE"][:, ii, jj])
        ageSN = jnp.stack([snoage[:, 2], snoage[:, 0], snoage[:, 1]], axis=1)
        zoice = jnp.where(poice > 0, g2("ZSI"), 0.0)
        fmp = jnp.where(poice > 0, jnp.minimum(1.6 * jnp.sqrt(g2("POND_MELT") / R.RHOW), 1.0), 0.0)
        zmp = jnp.where(poice > 0, jnp.minimum(0.8 * fmp, 0.9 * zoice), 0.0)
        flags = jnp.where(poice > 0, g2("FLAG_DSWS"), 0.0)
        snowd = jnp.transpose(pkt["SNOWD"][:, ii, jj])
        frsnow = jnp.transpose(pkt["FRSNOW"][:, ii, jj])
        wearth = jnp.minimum(g2("BARESW"), 1.0)
        prnb, prnx = R.getsur(cosza, pa["jlat"], pa["ilon"], ageSN, pocean, poice, pearth, plice, plake, g2("DLAKE"), tgo, zoice, fmp,
                              g2("ZSNOWI"), zmp, g2("SNOWI"), snowd, g2("SNOWLI"), frsnow, wearth, g2("WSAVG"), pa["frac_veg"], pa["dalbsn"],
                              flags, pa["loc_chl"], pa["albvnh"], pr.tab, ksialb=0)
        lat46 = -94.0 + 4.0 * pa["jlat"]
        ul, _ = R.getgas_ulgas(plb, shl, lat46, pa["o3col"], pa["fulgas"], pa["ppmv80"], pr.g["plb0"], pr.g["hlb0"], pa["fpxco2"],
                               (g2("LTROPO") + 1).astype(jnp.int32))
        comp, rad, mdd, fsu, ssw, slw = R.aerosol_inputs(pa["ataulx"], pa["dtaulx"], pa["vtaulx"], pa["vreff"], pa["q55h2s"], pr.taucon_dust,
                                                        pr.redust, pr.rodust, self._c60g())
        kl_in = jnp.transpose(pkt["KLIQ"][:, :, ii, jj], (2, 0, 1))
        kl_out = R.kdeliq_update(kl_in, rhl, tlm)
        zero3 = jnp.zeros((N, 3))
        top = lambda n: jnp.concatenate([zero3, gl(n)[:, ::-1]], axis=1)
        c = dict(p_level=plb[:, ::-1] * 1.0e+02, t_layer=tlm[:, ::-1], ulgas=ul.reshape(N, 559),
                 cldeps=R.cloudeps(plb, pa["epl"], pa["epm"], pa["eph"]),
                 prnb=jnp.transpose(prnb, (0, 2, 1)).reshape(N, 24), prnx=jnp.transpose(prnx, (0, 2, 1)).reshape(N, 24),
                 cos_zenith=cosza, solar_irrad=jnp.full((N,), pa["solar_irrad"]),
                 w_cloud=top("W_CLOUD"), frac_st_water=top("FRAC_ST_WATER"), frac_st_ice=top("FRAC_ST_ICE"), frac_cnv_water=top("FRAC_CNV_WATER"),
                 frac_cnv_ice=top("FRAC_CNV_ICE"), mix_st_water=top("MIX_ST_WATER"), mix_st_ice=top("MIX_ST_ICE"), mix_cnv_water=top("MIX_CNV_WATER"),
                 mix_cnv_ice=top("MIX_CNV_ICE"), dim_st_water=top("DIM_ST_WATER"), dim_st_ice=top("DIM_ST_ICE"), dim_cnv_water=top("DIM_CNV_WATER"),
                 dim_cnv_ice=top("DIM_CNV_ICE"), frac_area_st=top("FRAC_AREA_ST"), frac_area_cnv=top("FRAC_AREA_CNV"),
                 cld_scaling=jnp.full((N,), pr.cld_scaling), rel_humidity=rhl[:, ::-1],
                 wind=g2("WSAVG"), frac_sea=pocean, frac_land=pearth, frac_sea_ice=poice, frac_land_ice=plice,
                 t_sea=tgo, t_land=tge, t_sea_ice=tgoi, t_land_ice=tgli, soil_wetness=wearth,
                 frac_veg=pa["frac_veg"], frac_land_snow=frsnow, snow_depth_land=snowd, snow_depth_sea_ice=g2("ZSNOWI"),
                 snow_amnt_sea_lake=g2("SNOWI"))
        return c, (comp, rad, mdd, fsu, ssw, slw), (cosz1, cosz2, kl_out, poice, pocean, pearth, plice, tgo, tgoi, tgli, tge)

    def _core(self, pkt, cosz1_grid, cosz2_grid, cosza_grid, pa):
        c, aer, ex = self._inputs(pkt, cosz1_grid, cosz2_grid, cosza_grid, pa)
        res = self._planet_rad(c, *aer)
        return self._assemble(pkt, res, *ex)

    def _c60g(self):
        g = dict(self.prod.g)
        N = self.N
        for k in ("denaer", "refdry", "frsulf", "drym2g"):
            g[k] = np.asarray(g[k])[None, :]
        g["fs8opx"] = np.broadcast_to(np.asarray(g["fs8opx"])[None, :], (N, 8))
        g["ft8opx"] = np.broadcast_to(np.asarray(g["ft8opx"])[None, :], (N, 8))
        for k in ("fstaer", "fttaer", "fsaaer", "ftaaer", "fsdaer", "ftdaer", "fsvaer", "ftvaer"):
            g[k] = float(g[k])
        return g

    def _planet_rad(self, c, comp, rad, mdd, fsu, ssw, slw):
        """planet_rad set-up + all bands (LW all columns, SW all columns with the unlit ones masked) + get_planet_radout."""
        pr = self.prod
        S = pr.S
        N = self.N
        s0 = self.s0
        lit = c["cos_zenith"] > TOL_COS_ZENITH
        # unlit columns: a harmless zenith angle for the arithmetic (their short-wave results are masked by post_sw)
        c_sw = dict(c)
        c_sw["cos_zenith"] = jnp.where(lit, c["cos_zenith"], 1.0)

        def tag0(iso, cc):
            key = "lw" if iso == 2 else "sw"
            nb = 9 if iso == 2 else 6
            p_layer, t_level, mass, dens = P.layers(cc["p_level"], cc["t_layer"])
            ul = cc["ulgas"].reshape(-1, 13, 43)
            mix = P.gas_mix_ratio(ul, mass, self.spec_np[key]["type_absorb"], S[key])
            cc = dict(cc)
            w, frac, cmix, dim, rel = P.cloud_field(cc, S[key])
            scl = ssw if iso == 1 else slw
            mm, ab, sc, asy = P.aerosol_modes(S[key]["aer"], comp, rad, mdd, fsu, scl, cc["rel_humidity"], mass)
            r = {1: p_layer, 2: cc["t_layer"], 3: dens, 4: mix, 5: mass, 7: t_level, 10: w, 11: frac, 12: cmix, 13: dim, 14: rel,
                 27: mm, 28: ab, 29: sc, 30: asy}
            for k in (20, 21, 22, 23, 24, 25, 26, 31):
                r[k] = s0["%s0_%d" % (key, k)]
            r[31] = np.asarray(r[31]).copy(); r[31][2] = len(comp)
            part = ((cc["frac_sea"] > 0) & (cc["frac_sea"] < 1)) | ((cc["frac_land"] > 0) & (cc["frac_land"] < 1)) | \
                   ((cc["frac_sea_ice"] > 0) & (cc["frac_sea_ice"] < 1)) | ((cc["frac_land_ice"] > 0) & (cc["frac_land_ice"] < 1))
            tg = jnp.zeros_like(cc["t_sea"])
            for f, t in (("frac_sea", "t_sea"), ("frac_land", "t_land"), ("frac_sea_ice", "t_sea_ice"), ("frac_land_ice", "t_land_ice")):
                tg = jnp.where(cc[f] > 0, tg + cc[f] * cc[t], tg)
            til_rho = None
            if iso == 2:
                rho_d, tg, til = P.surface_lw(cc, S["lw"], nb)
                rho = jnp.stack([rho_d, jnp.zeros_like(rho_d)], axis=1)
                r[15] = jnp.ones(N); r[16] = jnp.zeros(N)
            else:
                rho, til_rho = P.surface_sw(cc, S["alb_wgt_sw"], nb)
                r[15] = 1.0 / cc["cos_zenith"]; r[16] = cc["solar_irrad"]
            r[17] = rho; r[18] = tg
            fr = jnp.stack([jnp.where(cc["frac_sea"] > 0, cc["frac_sea"], 0.0), jnp.where(cc["frac_sea_ice"] > 0, cc["frac_sea_ice"], 0.0),
                            jnp.where(cc["frac_land"] > 0, cc["frac_land"], 0.0), jnp.where(cc["frac_land_ice"] > 0, cc["frac_land_ice"], 0.0)], axis=1)
            tt = jnp.stack([jnp.where(cc["frac_sea"] > 0, cc["t_sea"], 0.0), jnp.where(cc["frac_sea_ice"] > 0, cc["t_sea_ice"], 0.0),
                            jnp.where(cc["frac_land"] > 0, cc["t_land"], 0.0), jnp.where(cc["frac_land_ice"] > 0, cc["t_land_ice"], 0.0)], axis=1)
            r["tile_flag"] = part; r["tile_n"] = jnp.where(part, 4, 0); r["tile_frac"] = fr; r["tile_t"] = tt
            return r, til_rho

        r_lw, _ = tag0(2, c)
        res_lw = RB.radiance(self.spec[2], 2, r_lw, jit=False)
        r_sw, til_rho = tag0(1, c_sw)
        res_sw = RB.radiance(self.spec[1], 1, r_sw, jit=False)
        tcc = P.total_cloud_cover(r_lw[10], r_lw[11])
        lw = P.post_lw([x[0] for x in res_lw], [x[1] for x in res_lw], r_lw[7])
        nb = 6
        dn_b = [x[0] for x in res_sw]; up_b = [x[1] for x in res_sw]; dr_b = [x[2] for x in res_sw]
        tile_up = jnp.stack([jnp.stack([til_rho[:, 0, k, b] * res_sw[b][0][:, -1] + (til_rho[:, 1, k, b] - til_rho[:, 0, k, b]) * res_sw[b][2][:, -1]
                                        for b in range(nb)], axis=1) for k in range(4)], axis=1)
        tile_up = tile_up / jnp.where(lit, c["cos_zenith"], 1.0)[:, None, None]
        sw = P.post_sw(dn_b, up_b, dr_b, lit, c["cos_zenith"], S["sw"]["weight_blue"], None, None, c["solar_irrad"])
        part = ((c["frac_sea"] > 0) & (c["frac_sea"] < 1)) | ((c["frac_land"] > 0) & (c["frac_land"] < 1)) | \
               ((c["frac_sea_ice"] > 0) & (c["frac_sea_ice"] < 1)) | ((c["frac_land_ice"] > 0) & (c["frac_land_ice"] < 1))
        fs = P.fsrnfg(sw, (c["frac_sea"], c["frac_land"], c["frac_sea_ice"], c["frac_land_ice"]), lit & part, tile_up)
        out = dict(lw)
        out.update(sw)
        out["fsrnfg"] = fs
        out["tot_cloud_cover"] = tcc
        return out

    def _assemble(self, pkt, res, cosz1, cosz2, kl_out, poice, pocean, pearth, plice, tgo, tgoi, tgli, tge):
        IM, JM, LM = 72, 46, 40
        ii, jj = jnp.asarray(self.ii - 1), jnp.asarray(self.jj - 1)
        N = self.N
        pr = self.prod
        srfhrl = res["srfhrl"]; trfcrl = res["trfcrl"]
        trh0 = STBO * (pocean * tgo ** 4 + poice * tgoi ** 4 + plice * tgli ** 4 + pearth * tge ** 4) - res["trnflb"][:, 0]
        srhr = jnp.concatenate([res["srnflb"][:, :1], srfhrl[:, :LM]], axis=1)
        trhr = jnp.concatenate([trh0[:, None], -trfcrl[:, :LM]], axis=1)
        rqt_in = jnp.transpose(pkt["RQT"][:, ii, jj])
        rqt = rqt_in + (srfhrl[:, LM:] * cosz2[:, None] + (-trfcrl[:, LM:])) * NRAD * DTSRC * BYSHA * jnp.asarray(pr.byaml00[LM:LM + 3])[None, :]
        pk = jnp.transpose(pkt["PK"][:, ii, jj])
        byma = jnp.transpose(pkt["BYMA"][:, ii, jj])
        T_in = pkt["T"][ii, jj, :]
        T_new = T_in + (srhr[:, 1:] * cosz1[:, None] + trhr[:, 1:]) * DTSRC * BYSHA * byma / pk
        Q_in = pkt["Q"][ii, jj, :]
        Q_new = jnp.where(Q_in < 0.0, 0.0, Q_in)
        cldss = jnp.where(jnp.transpose(pkt["TAUSS"][:, ii, jj]) <= TAULIM, 0.0, jnp.transpose(pkt["CLDSS"][:, ii, jj]))
        cldmc = jnp.where(jnp.transpose(pkt["TAUMC"][:, ii, jj]) <= TAULIM, 0.0, jnp.transpose(pkt["CLDMC"][:, ii, jj]))
        stb = lambda t: STBO * t ** 4
        fsrnfg = res["fsrnfg"]
        zero = jnp.zeros(N)
        alb = jnp.stack([res["srnflb"][:, 0] / (res["srdflb"][:, 0] + 1.0e-20), res["plavis"], res["planir"], res["albvis"], res["albnir"],
                         zero, zero, zero, zero], axis=1)
        out = {}
        out["T"] = pkt["T"].at[ii, jj, :].set(T_new)
        out["Q"] = pkt["Q"].at[ii, jj, :].set(Q_new)
        out["RQT"] = pkt["RQT"].at[:, ii, jj].set(jnp.transpose(rqt))
        out["KLIQ"] = pkt["KLIQ"].at[:, :, ii, jj].set(jnp.transpose(kl_out, (1, 2, 0)))
        out["SNOAGE"] = pkt["SNOAGE"]
        out["CLDSS"] = pkt["CLDSS"].at[:, ii, jj].set(jnp.transpose(cldss))
        out["CLDMC"] = pkt["CLDMC"].at[:, ii, jj].set(jnp.transpose(cldmc))
        out["SRHR"] = jnp.zeros((41, IM, JM)).at[:, ii, jj].set(jnp.transpose(srhr))
        out["TRHR"] = jnp.zeros((41, IM, JM)).at[:, ii, jj].set(jnp.transpose(trhr))
        fsf = jnp.stack([fsrnfg[:, 0], fsrnfg[:, 2], fsrnfg[:, 3], fsrnfg[:, 1]], axis=1)
        out["FSF"] = jnp.zeros((4, IM, JM)).at[:, ii, jj].set(jnp.transpose(fsf))
        trs = jnp.stack([stb(tgo), stb(tgoi), stb(tgli), stb(tge)], axis=1)
        out["TRSURF"] = jnp.zeros((4, IM, JM)).at[:, ii, jj].set(jnp.transpose(trs))
        out["ALB"] = jnp.zeros((IM, JM, 9)).at[ii, jj, :].set(alb)
        two = lambda v: jnp.zeros((IM, JM)).at[ii, jj].set(v)
        out["FSRDIR"] = two(res["srxvis"]); out["SRVISSURF"] = two(res["srdvis"])
        out["FSRDIF"] = two(res["srdvis"] * (1 - res["srxvis"])); out["DIRVIS"] = two(res["srxvis"] * res["srdvis"])
        out["DIRNIR"] = two(res["srxnir"] * res["srdnir"]); out["DIFNIR"] = two(res["srdnir"] * (1 - res["srxnir"]))
        out["SRDN"] = two(res["srdflb"][:, 0]); out["CFRAC"] = two(res["tot_cloud_cover"]); out["COSZ1"] = two(cosz1)
        # not packet outputs: TOA net fluxes (SNFS(3), TNFS(3) of RADIA) for the AIJ accumulators of the hand-off
        out["_SNFS3"] = two(res["srnflb"][:, -1]); out["_TNFS3"] = two(res["trnflb"][:, -1])
        return out
