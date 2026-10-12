"""R6: RADIA (RAD_DRV.f) with SOCRATES for this rundeck, in JAX: replaces the persistent radiation server for the 21 non-AIJ output fields of one
radiation step.  Input = the 52-field RADIA input packet (radiation_server.INPUT_FIELDS) of the step; output = the fields of OUTPUT_FIELDS
(except AIJ; the 750 radiation diagnostics are not ported).

   rj = RadiaJax(specdir, products)       # products: daily / static products (ProductsNpz below), see report section 12
   out = rj(pkt, itime)                   # pkt: dict name -> array (read_packet layout); out: dict name -> array

Pipeline: column set-up and albedos and gases and aerosols (radia_cols) -> planet_rad set-up (planet_side) -> all bands (radiance_batch) ->
get_planet_radout (planet_side) -> RADIA's output assembly (below).  Units of work are columns (3170 per step: 72 x 44 + 2 poles)."""
import os, sys
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

    def __init__(self, z):
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
        self.loc_chl = c60["loc_chl"]
        self.frac_veg = c50["frac_veg"]
        self.fpxco2 = c60["fpxco2"]
        self.ls1 = c60["ls1"]
        self.jlat = c60["jlat"].astype(int)
        self.ilon = c60["ilon"].astype(int)
        # cloud heterogeneity per column (EPLOW / EPMID / EPHIG): the three values CLDEPS takes with the pressure of the layer edges
        plb = c60["plb"][:, :43]
        cl = c60["cldeps"]
        ar = np.arange(N)
        self.epl = np.where((plb > 750.0).any(axis=1), cl[ar, np.argmax(plb > 750.0, axis=1)], 0.0)
        mid = (plb <= 750.0) & (plb >= 430.0)
        self.epm = np.where(mid.any(axis=1), cl[ar, np.argmax(mid, axis=1)], 0.0)
        hi = plb < 430.0
        self.eph = np.where(hi.any(axis=1), cl[ar, np.argmax(hi, axis=1)], 0.0)
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
    def __init__(self, specdir, static0_npz, products, zen=None):
        self.spec = {1: SD.load(os.path.join(specdir, "spec_sw.npz")), 2: SD.load(os.path.join(specdir, "spec_lw.npz"))}
        self.spec_np = {"lw": np.load(os.path.join(specdir, "spec_lw.npz")), "sw": np.load(os.path.join(specdir, "spec_sw.npz"))}
        self.s0 = np.load(static0_npz) if isinstance(static0_npz, str) else static0_npz
        self.prod = products
        self.zen = zen
        self.ii, self.jj = R.column_index()
        self.N = len(self.ii)

    # ---------------------------------------------------------------------------------------------------------- inputs
    def _gather(self, pkt):
        ii, jj = self.ii - 1, self.jj - 1
        g3 = lambda n: pkt[n][ii, jj, :]
        gl = lambda n: pkt[n][:, ii, jj].T
        g2 = lambda n: pkt[n][ii, jj]
        return g3, gl, g2

    def __call__(self, pkt, itime, cosz=None, collect=None):
        pr = self.prod
        N = self.N
        g3, gl, g2 = self._gather(pkt)
        J = jnp.asarray
        zen = self.zen
        if cosz is None:
            cz2_grid, cza_grid = ZS.calc_zenith_angle(zen, itime, NRAD)
            cosz1_grid = zen.cosz1(itime)
        else:
            cz2_grid, cza_grid, cosz1_grid = cosz
        cosza = cza_grid[self.ii - 1, self.jj - 1]
        cosz2 = cz2_grid[self.ii - 1, self.jj - 1]
        # ---- column state
        cs = R.column_state(J(g3("T")), J(g3("Q")), J(gl("PK")), J(gl("PMID")), J(gl("PEDN")), J(gl("RQT")), pr.plb0_req, pr.shl0_req)
        plb, tlm, shl, rhl = cs["plb"], cs["tlm"], cs["shl"], cs["rhl"]
        # ---- surface
        pland = g2("FLAND")
        poice = g2("RSI") * (1.0 - pland)
        pocean = (1.0 - pland) - poice
        plake, plice, pearth = g2("FLAKE"), g2("FLICE"), g2("FEARTH")
        tgo, tgoi, tgli, tge = g2("GTEMPR1"), g2("GTEMPR2"), g2("GTEMPR3"), g2("GTEMPR4")
        snoage = pkt["SNOAGE"][:, self.ii - 1, self.jj - 1].T
        ageSN = np.stack([snoage[:, 2], snoage[:, 0], snoage[:, 1]], axis=1)
        zoice = np.where(poice > 0, g2("ZSI"), 0.0)
        fmp = np.where(poice > 0, np.minimum(1.6 * np.sqrt(g2("POND_MELT") / R.RHOW), 1.0), 0.0)
        zmp = np.where(poice > 0, np.minimum(0.8 * fmp, 0.9 * zoice), 0.0)
        flags = np.where(poice > 0, g2("FLAG_DSWS"), 0.0)
        snowd = pkt["SNOWD"][:, self.ii - 1, self.jj - 1].T
        frsnow = pkt["FRSNOW"][:, self.ii - 1, self.jj - 1].T
        wearth = np.minimum(g2("BARESW"), 1.0)
        prnb, prnx = R.getsur(J(cosza), J(pr.jlat), J(pr.ilon), J(ageSN), J(pocean), J(poice), J(pearth), J(plice), J(plake), J(g2("DLAKE")),
                              J(tgo), J(zoice), J(fmp), J(g2("ZSNOWI")), J(zmp), J(g2("SNOWI")), J(snowd), J(g2("SNOWLI")), J(frsnow), J(wearth),
                              J(g2("WSAVG")), J(pr.frac_veg), J(pr.dalbsn), J(flags), J(pr.loc_chl), pr.albvnh, pr.tab, ksialb=0)
        # ---- gases
        lat46 = -94.0 + 4.0 * pr.jlat
        ul, _ = R.getgas_ulgas(plb, shl, J(lat46), J(pr.o3col), pr.g["fulgas"], pr.g["ppmv80"], pr.g["plb0"], pr.g["hlb0"], J(pr.fpxco2),
                               (g2("LTROPO") + 1).astype(int))
        # ---- aerosols
        comp, rad, mdd, fsu, ssw, slw = R.aerosol_inputs(J(pr.ataulx), J(pr.dtaulx), J(pr.vtaulx), J(pr.vreff), J(pr.q55h2s), pr.taucon_dust,
                                                        pr.redust, pr.rodust, {k: (v if np.ndim(v) else v) for k, v in self._c60g().items()})
        kl_in = pkt["KLIQ"][:, :, self.ii - 1, self.jj - 1].transpose(2, 0, 1)          # (N, 40, 4)
        kl_out = R.kdeliq_update(J(kl_in), rhl, tlm)
        # ---- planet_rad arguments (same layout as the tag-50 record)
        zero3 = jnp.zeros((N, 3))
        top = lambda n: jnp.concatenate([zero3, J(gl(n))[:, ::-1]], axis=1)
        c = dict(p_level=plb[:, ::-1] * 1.0e+02, t_layer=tlm[:, ::-1], ulgas=ul.reshape(N, 559), cldeps=R.cloudeps(plb, J(pr.epl), J(pr.epm), J(pr.eph)),
                 prnb=jnp.transpose(prnb, (0, 2, 1)).reshape(N, 24), prnx=jnp.transpose(prnx, (0, 2, 1)).reshape(N, 24),
                 cos_zenith=J(cosza), solar_irrad=jnp.full((N,), pr.solar_irrad),
                 w_cloud=top("W_CLOUD"), frac_st_water=top("FRAC_ST_WATER"), frac_st_ice=top("FRAC_ST_ICE"), frac_cnv_water=top("FRAC_CNV_WATER"),
                 frac_cnv_ice=top("FRAC_CNV_ICE"), mix_st_water=top("MIX_ST_WATER"), mix_st_ice=top("MIX_ST_ICE"), mix_cnv_water=top("MIX_CNV_WATER"),
                 mix_cnv_ice=top("MIX_CNV_ICE"), dim_st_water=top("DIM_ST_WATER"), dim_st_ice=top("DIM_ST_ICE"), dim_cnv_water=top("DIM_CNV_WATER"),
                 dim_cnv_ice=top("DIM_CNV_ICE"), frac_area_st=top("FRAC_AREA_ST"), frac_area_cnv=top("FRAC_AREA_CNV"),
                 cld_scaling=jnp.full((N,), pr.cld_scaling), rel_humidity=rhl[:, ::-1],
                 wind=J(g2("WSAVG")), frac_sea=J(pocean), frac_land=J(pearth), frac_sea_ice=J(poice), frac_land_ice=J(plice),
                 t_sea=J(tgo), t_land=J(tge), t_sea_ice=J(tgoi), t_land_ice=J(tgli), soil_wetness=J(wearth),
                 frac_veg=J(pr.frac_veg), frac_land_snow=J(frsnow), snow_depth_land=J(snowd), snow_depth_sea_ice=J(g2("ZSNOWI")),
                 snow_amnt_sea_lake=J(g2("SNOWI")))
        res = self._planet_rad(c, comp, rad, mdd, fsu, ssw, slw)
        if collect is not None:
            collect["c"] = c
            collect["res"] = res
        return self._assemble(pkt, res, c, cosz1_grid, cz2_grid, kl_out, poice, pocean, pearth, plice, tgo, tgoi, tgli, tge)

    def _c60g(self):
        g = dict(self.prod.g)
        N = self.N
        for k in ("fs8opx", "ft8opx", "denaer", "refdry", "frsulf", "drym2g"):
            g[k] = np.asarray(g[k])[None, :]
        for k in ("fstaer", "fttaer", "fsaaer", "ftaaer", "fsdaer", "ftdaer", "fsvaer", "ftvaer"):
            g[k] = float(g[k])
        g["fs8opx"] = np.broadcast_to(g["fs8opx"], (N, 8)); g["ft8opx"] = np.broadcast_to(g["ft8opx"], (N, 8))
        return g

    # ---------------------------------------------------------------------------------------------------------- planet_rad + SOCRATES
    def _planet_rad(self, c, comp, rad, mdd, fsu, ssw, slw):
        pr = self.prod
        S = pr.S
        N = self.N
        lit = np.asarray(c["cos_zenith"]) > TOL_COS_ZENITH
        sw_idx = np.nonzero(lit)[0]
        s0 = self.s0

        def tag0(iso, ci):
            key = "lw" if iso == 2 else "sw"
            nb = 9 if iso == 2 else 6
            sub = {k: v[ci] for k, v in c.items()}
            p_layer, t_level, mass, dens = P.layers(sub["p_level"], sub["t_layer"])
            ul = sub["ulgas"].reshape(-1, 13, 43)
            mix = P.gas_mix_ratio(ul, mass, self.spec_np[key]["type_absorb"], S[key])
            sub["n_aer"] = jnp.full((len(ci),), float(len(comp)))
            w, frac, cmix, dim, rel = P.cloud_field(sub, S[key])
            scl = ssw if iso == 1 else slw
            mm, ab, sc, asy = P.aerosol_modes(S[key]["aer"], comp, rad[ci], mdd[ci], fsu[ci], scl[ci], sub["rel_humidity"], mass)
            r = {1: p_layer, 2: sub["t_layer"], 3: dens, 4: mix, 5: mass, 7: t_level, 10: w, 11: frac, 12: cmix, 13: dim, 14: rel,
                 27: mm, 28: ab, 29: sc, 30: asy}
            for k in (20, 21, 22, 23, 24, 25, 26, 31):
                r[k] = s0["%s0_%d" % (key, k)]
            r[31] = np.asarray(r[31]).copy(); r[31][2] = len(comp)
            part = ((sub["frac_sea"] > 0) & (sub["frac_sea"] < 1)) | ((sub["frac_land"] > 0) & (sub["frac_land"] < 1)) | \
                   ((sub["frac_sea_ice"] > 0) & (sub["frac_sea_ice"] < 1)) | ((sub["frac_land_ice"] > 0) & (sub["frac_land_ice"] < 1))
            tg = jnp.zeros_like(sub["t_sea"])
            for f, t in (("frac_sea", "t_sea"), ("frac_land", "t_land"), ("frac_sea_ice", "t_sea_ice"), ("frac_land_ice", "t_land_ice")):
                tg = jnp.where(sub[f] > 0, tg + sub[f] * sub[t], tg)
            til_rho = None
            if iso == 2:
                rho_d, tg, til = P.surface_lw(sub, S["lw"], nb)
                rho = jnp.stack([rho_d, jnp.zeros_like(rho_d)], axis=1)
                r[15] = jnp.ones(len(ci)); r[16] = jnp.zeros(len(ci))
            else:
                rho, til_rho = P.surface_sw(sub, S["alb_wgt_sw"], nb)
                r[15] = 1.0 / sub["cos_zenith"]; r[16] = sub["solar_irrad"]
            r[17] = rho; r[18] = tg
            fr = jnp.stack([jnp.where(sub["frac_sea"] > 0, sub["frac_sea"], 0.0), jnp.where(sub["frac_sea_ice"] > 0, sub["frac_sea_ice"], 0.0),
                            jnp.where(sub["frac_land"] > 0, sub["frac_land"], 0.0), jnp.where(sub["frac_land_ice"] > 0, sub["frac_land_ice"], 0.0)], axis=1)
            tt = jnp.stack([jnp.where(sub["frac_sea"] > 0, sub["t_sea"], 0.0), jnp.where(sub["frac_sea_ice"] > 0, sub["t_sea_ice"], 0.0),
                            jnp.where(sub["frac_land"] > 0, sub["t_land"], 0.0), jnp.where(sub["frac_land_ice"] > 0, sub["t_land_ice"], 0.0)], axis=1)
            flag = np.asarray(part)
            r["tile_flag"] = flag; r["tile_n"] = np.where(flag, 4, 0); r["tile_frac"] = fr; r["tile_t"] = tt
            return r, til_rho

        r_lw, _ = tag0(2, np.arange(N))
        res_lw = RB.radiance(self.spec[2], 2, r_lw)
        r_sw, til_rho = tag0(1, sw_idx)
        res_sw = RB.radiance(self.spec[1], 1, r_sw)
        tcc = P.total_cloud_cover(r_lw[10], r_lw[11])
        lw = P.post_lw([x[0] for x in res_lw], [x[1] for x in res_lw], r_lw[7])
        nb = 6
        sidx = jnp.asarray(sw_idx)

        def expand(a):
            return jnp.zeros((N,) + a.shape[1:], dtype=a.dtype).at[sidx].set(a)

        dn_b = [expand(x[0]) for x in res_sw]; up_b = [expand(x[1]) for x in res_sw]; dr_b = [expand(x[2]) for x in res_sw]
        litj = jnp.asarray(lit)
        cosz = c["cos_zenith"]
        tile_up_l = jnp.stack([jnp.stack([til_rho[:, 0, k, b] * res_sw[b][0][:, -1] + (til_rho[:, 1, k, b] - til_rho[:, 0, k, b]) * res_sw[b][2][:, -1]
                                          for b in range(nb)], axis=1) for k in range(4)], axis=1)
        tile_up = expand(tile_up_l) / jnp.where(litj, cosz, 1.0)[:, None, None]
        sw = P.post_sw(dn_b, up_b, dr_b, litj, cosz, S["sw"]["weight_blue"], None, None, c["solar_irrad"])
        part = ((c["frac_sea"] > 0) & (c["frac_sea"] < 1)) | ((c["frac_land"] > 0) & (c["frac_land"] < 1)) | \
               ((c["frac_sea_ice"] > 0) & (c["frac_sea_ice"] < 1)) | ((c["frac_land_ice"] > 0) & (c["frac_land_ice"] < 1))
        fs = P.fsrnfg(sw, (c["frac_sea"], c["frac_land"], c["frac_sea_ice"], c["frac_land_ice"]), litj & part, tile_up)
        out = dict(lw)
        out.update(sw)
        out["fsrnfg"] = fs
        out["tot_cloud_cover"] = tcc
        return out

    # ---------------------------------------------------------------------------------------------------------- RADIA output assembly
    def _assemble(self, pkt, res, c, cosz1_grid, cosz2_grid, kl_out, poice, pocean, pearth, plice, tgo, tgoi, tgli, tge):
        IM, JM, LM = 72, 46, 40
        ii, jj = self.ii - 1, self.jj - 1
        N = self.N
        pr = self.prod
        cosz1 = cosz1_grid[ii, jj]
        cosz2 = cosz2_grid[ii, jj]
        srfhrl = res["srfhrl"]; trfcrl = res["trfcrl"]
        trh0 = STBO * (pocean * tgo ** 4 + poice * tgoi ** 4 + plice * tgli ** 4 + pearth * tge ** 4) - res["trnflb"][:, 0]
        srhr = jnp.concatenate([res["srnflb"][:, :1], srfhrl[:, :LM]], axis=1)          # (N, 41): index 0 surface net, 1..40
        trhr = jnp.concatenate([trh0[:, None], -trfcrl[:, :LM]], axis=1)
        # RQT: RADIA label 'Update radiative equilibrium temperatures'
        rqt_in = pkt["RQT"][:, ii, jj].T
        rqt = rqt_in + (srfhrl[:, LM:] * cosz2[:, None] + (-trfcrl[:, LM:])) * NRAD * DTSRC * BYSHA * jnp.asarray(pr.byaml00[LM:LM + 3])[None, :]
        # T (label 900) and Q reset
        pk = pkt["PK"][:, ii, jj].T
        byma = pkt["BYMA"][:, ii, jj].T
        T_in = pkt["T"][ii, jj, :]
        T_new = T_in + (srhr[:, 1:] * cosz1[:, None] + trhr[:, 1:]) * DTSRC * BYSHA * byma / pk
        Q_new = np.where(pkt["Q"][ii, jj, :] < 0.0, 0.0, pkt["Q"][ii, jj, :])
        # clouds seen by radiation
        cldss = np.where(pkt["TAUSS"][:, ii, jj].T <= TAULIM, 0.0, pkt["CLDSS"][:, ii, jj].T)
        cldmc = np.where(pkt["TAUMC"][:, ii, jj].T <= TAULIM, 0.0, pkt["CLDMC"][:, ii, jj].T)
        stb = lambda t: STBO * t ** 4
        fsrnfg = res["fsrnfg"]
        alb = jnp.stack([res["srnflb"][:, 0] / (res["srdflb"][:, 0] + 1.0e-20), res["plavis"], res["planir"], res["albvis"], res["albnir"],
                         jnp.zeros(N), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N)], axis=1)
        col = dict(
            SRHR=srhr, TRHR=trhr, RQT=rqt, T=T_new, Q=Q_new, KLIQ=kl_out, CLDSS=cldss, CLDMC=cldmc,
            FSF=jnp.stack([fsrnfg[:, 0], fsrnfg[:, 2], fsrnfg[:, 3], fsrnfg[:, 1]], axis=1),
            TRSURF=jnp.stack([stb(tgo), stb(tgoi), stb(tgli), stb(tge)], axis=1), ALB=alb,
            FSRDIR=res["srxvis"], SRVISSURF=res["srdvis"], FSRDIF=res["srdvis"] * (1 - res["srxvis"]), DIRVIS=res["srxvis"] * res["srdvis"],
            DIRNIR=res["srxnir"] * res["srdnir"], DIFNIR=res["srdnir"] * (1 - res["srxnir"]), SRDN=res["srdflb"][:, 0],
            CFRAC=res["tot_cloud_cover"], COSZ1=cosz1)
        return self._to_grid(pkt, {k: np.asarray(v) for k, v in col.items()})

    def _to_grid(self, pkt, col):
        """Columns -> packet layout.  Boxes not computed by RADIA (I > IMAXJ at the poles) keep the input value (T, Q, ...) or 0."""
        ii, jj = self.ii - 1, self.jj - 1
        out = {}
        out["T"] = np.array(pkt["T"], copy=True); out["T"][ii, jj, :] = col["T"]
        out["Q"] = np.array(pkt["Q"], copy=True); out["Q"][ii, jj, :] = col["Q"]
        out["RQT"] = np.array(pkt["RQT"], copy=True); out["RQT"][:, ii, jj] = col["RQT"].T
        out["KLIQ"] = np.array(pkt["KLIQ"], copy=True); out["KLIQ"][:, :, ii, jj] = col["KLIQ"].transpose(1, 2, 0)
        out["SNOAGE"] = np.array(pkt["SNOAGE"], copy=True)
        for k in ("CLDSS", "CLDMC"):
            out[k] = np.array(pkt[k], copy=True); out[k][:, ii, jj] = col[k].T
        for k, nlev in (("SRHR", 41), ("TRHR", 41)):
            a = np.zeros((nlev, 72, 46)); a[:, ii, jj] = col[k].T; out[k] = a
        for k, n in (("FSF", 4), ("TRSURF", 4)):
            a = np.zeros((n, 72, 46)); a[:, ii, jj] = col[k].T; out[k] = a
        a = np.zeros((72, 46, 9)); a[ii, jj, :] = col["ALB"]; out["ALB"] = a
        for k in ("FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS", "DIRNIR", "DIFNIR", "SRDN", "CFRAC", "COSZ1"):
            a = np.zeros((72, 46)); a[ii, jj] = col[k]; out[k] = a
        return out
