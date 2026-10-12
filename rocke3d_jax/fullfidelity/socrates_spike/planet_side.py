"""R6 stage B: the planet_rad side of the radiation call, for this rundeck (planet_rad.F90 run_planet_rad / set_atm / set_gas_mix_ratio /
set_cld / set_cloud_field / set_aer / calc_aer_opt_prop / interp_aer / set_bound_lw / set_bound_sw, planet_alb.F90 set_surf_prop_lw / _sw /
set_tiling / calc_alb_wgt / integ_planck, get_planet_radout).

Input: the arguments and module state at the entry of run_planet_rad (dump tag 50, layout PACK50 below), the static records (tags 54, 101-150).
Output: the tag-0 style arrays of the radiance_calc inputs, for LW and SW, as the batched formulation (radiance_batch.prepare) consumes them.
Everything is vectorised over columns (leading axis N) and written with jax.numpy (float64); layer 0 is the top layer as in SOCRATES.

Not ported (not reachable with this rundeck, R1/R6 reports): spherical harmonics, McICA, other cloud representations / inhomogeneity
schemes, the aerosol diagnostics (l_aerosol_diag), tiling with other than four tiles, the clear-sky diagnostic outputs (l_clear)."""
import numpy as np
import jax
import jax.numpy as jnp

NL = 43                                   # npd_layer = lm (40) + lm_req (3)

# ---- constants of ModelE (shared/Constants_mod.F90, no PLANET_PARAMS) and SOCRATES (gas_list_pcf.F90)
GRAV = 9.80665
GASC = 8.314510
MAIR = 28.9655
RGAS = 1.0e3 * GASC / MAIR
AVOG = 6.02214129e23
LOSCHMIDT = 2.6867805e19
PN2, PO2, PAR, PH2 = 0.780840, 0.209476, 0.0093, 0.0
K_BOLTZMANN, H_PLANCK, C_LIGHT = 1.380662e-23, 6.626176e-34, 2.9979245e8
PI = 3.14159265358979323846
TINY = float(np.finfo(np.float64).tiny)
IP_AMMONIUM_SULPHATE = 7
MOLAR_WEIGHT = {1: 18.0153, 2: 44.0100, 3: 47.9982, 4: 44.0128, 6: 16.0430, 7: 31.9988, 9: 64.0628, 10: 46.0055, 13: 28.0134,
                14: 137.3686, 15: 120.9140, 16: 187.3765, 17: 86.46892, 19: 102.03184, 23: 2.01588, 25: 60.075, 36: 39.948}

# ---- packed records (ffsoc/apply_ffsoc_r6.py)
PACK50 = [("p_level", 44), ("t_layer", 43), ("t_level", 44), ("ulgas", 559), ("cldeps", 43), ("prnb", 24), ("prnx", 24),
          ("cos_zenith", 1), ("solar_irrad", 1),
          ("w_cloud", 43), ("frac_st_water", 43), ("frac_st_ice", 43), ("frac_cnv_water", 43), ("frac_cnv_ice", 43),
          ("mix_st_water", 43), ("mix_st_ice", 43), ("mix_cnv_water", 43), ("mix_cnv_ice", 43),
          ("dim_st_water", 43), ("dim_st_ice", 43), ("dim_cnv_water", 43), ("dim_cnv_ice", 43),
          ("frac_area_st", 43), ("frac_area_cnv", 43), ("cld_scaling", 1), ("n_aer", 1), ("n_aer_tracer", 1), ("rel_humidity", 43),
          ("wind", 1), ("frac_sea", 1), ("frac_land", 1), ("frac_sea_ice", 1), ("frac_land_ice", 1),
          ("t_sea", 1), ("t_land", 1), ("t_sea_ice", 1), ("t_land_ice", 1), ("soil_wetness", 1),
          ("frac_veg", 12), ("frac_land_snow", 2), ("snow_depth_land", 2), ("snow_depth_sea_ice", 1), ("snow_amnt_sea_lake", 1)]
PACK60 = [("ilon", 1), ("jlat", 1), ("cosz", 1), ("ls1", 1), ("kfpco2", 1), ("plb", 44), ("plb0", 44), ("hlb0", 44),
          ("tlm", 43), ("shl", 43), ("rhl", 43), ("cldeps", 43), ("u0gas", 559), ("fulgas", 13), ("ppmv80", 13), ("fpxco2", 43),
          ("ataulx", 258), ("vtaulx", 43), ("vreff", 1), ("q55h2s", 1), ("agesn", 3), ("zoice", 1), ("fmp", 1), ("zmp", 1),
          ("zlake", 1), ("snowli", 1), ("dalbsn", 1), ("loc_chl", 1), ("plake", 1), ("flags", 1), ("snowoi", 1), ("zsnwoi", 1),
          ("fstaer", 1), ("fttaer", 1), ("fsaaer", 1), ("ftaaer", 1), ("fsdaer", 1), ("ftdaer", 1), ("fsvaer", 1), ("ftvaer", 1),
          ("fs8opx", 8), ("ft8opx", 8), ("denaer", 8), ("refdry", 8), ("frsulf", 8), ("drym2g", 8)]
PACK51 = [("tlb", 44), ("tlt", 43), ("truflb", 44), ("trdflb", 44), ("trnflb", 44), ("trfcrl", 43), ("sruflb", 44), ("srdflb", 44),
          ("srnflb", 44), ("srfhrl", 43), ("fsrnfg", 4), ("srivis", 1), ("srovis", 1), ("plavis", 1), ("srinir", 1), ("sronir", 1),
          ("planir", 1), ("srdvis", 1), ("sruvis", 1), ("albvis", 1), ("srdnir", 1), ("srunir", 1), ("albnir", 1), ("srxvis", 1),
          ("srxnir", 1), ("s0", 1), ("truptoa", 9), ("sruptoa", 6), ("srdntoa", 6), ("tot_cloud_cover", 1)]
FLAGS = ["l_co2", "l_o3", "l_o2", "l_no2", "l_n2o", "l_ch4", "l_so2", "l_n2", "l_cfc11", "l_cfc12", "l_ar", "l_h2", "l_cloud",
         "l_aerosol", "l_microphysics", "i_sea_ice_alb", "i_cloud_ice_size_scheme", "i_cloud", "i_inhom"]


def unpack(a, layout):
    """a: (N, n) packed -> dict of (N,) or (N, k) arrays."""
    out, o = {}, 0
    for name, n in layout:
        out[name] = a[:, o] if n == 1 else a[:, o:o + n]
        o += n
    assert o == a.shape[1], (o, a.shape)
    return out


def static_from_npz(z):
    """The static records of one dump: control flags, band limits, planet aerosol tables, album weights, cloud-parameter limits."""
    S = {}
    for iso, t in (("lw", 101), ("sw", 102)):
        f = z["static_%d" % t]
        S[iso] = dict(zip(FLAGS, f))
    S["lw"].update(wl_short=z["static_103"], wl_long=z["static_104"], n_excl=z["static_105"].astype(int), idx_excl=z["static_106"].astype(int),
                   dim_lim=z["static_112"])
    S["sw"].update(wl_short=z["static_107"], wl_long=z["static_108"], n_excl=z["static_109"].astype(int), idx_excl=z["static_110"].astype(int),
                   weight_blue=z["static_111"], dim_lim=z["static_113"])
    for iso, b in (("lw", 120), ("sw", 140)):
        nc, nr, nh, nb = [int(v) for v in z["static_%d" % b]]
        S[iso]["aer"] = dict(n_comp=nc, n_rad_max=nr, n_hum=nh, n_band=nb, i_component=z["static_%d" % (b + 1)].astype(int),
                             n_radius=z["static_%d" % (b + 2)].astype(int), comp_map=z["static_%d" % (b + 3)].astype(int),
                             humidity=z["static_%d" % (b + 4)], l_humidity=z["static_%d" % (b + 5)] > 0.5,
                             radius=z["static_%d" % (b + 6)], density=z["static_%d" % (b + 7)],
                             k_abs=z["static_%d" % (b + 8)].reshape(nb, nh, nr, nc, order="F"),
                             k_scat=z["static_%d" % (b + 9)].reshape(nb, nh, nr, nc, order="F"),
                             g_asym=z["static_%d" % (b + 10)].reshape(nb, nh, nr, nc, order="F"))
    S["alb_wgt_sw"] = z["alb_wgt_sw"]
    return S


# ---------------------------------------------------------------------------------------------------------------- atmosphere
def layers(p_level, t_layer):
    """run_planet_rad / set_t_level / set_atm: p_layer, t_level (log10-p interpolation), dry mass, density.  p_level (N,44), t_layer (N,43)."""
    p_layer = (p_level[:, :-1] + p_level[:, 1:]) * 0.5
    lp = jnp.log10(p_layer)
    wgt_up = (lp[:, 1:] - jnp.log10(p_level[:, 1:-1])) / (lp[:, 1:] - lp[:, :-1])
    wgt_dwn = 1.0 - wgt_up
    inner = wgt_up * t_layer[:, :-1] + wgt_dwn * t_layer[:, 1:]
    t_level = jnp.concatenate([t_layer[:, :1], inner, t_layer[:, -1:]], axis=1)
    mass = (p_level[:, 1:] - p_level[:, :-1]) / GRAV
    density = p_layer / (RGAS * t_layer)
    return p_layer, t_level, mass, density


def gas_mix_ratio(ulgas, mass, type_absorb, flags):
    """set_gas_mix_ratio.  ulgas (N, 13, 43) (GISS order: layer 0 = bottom), mass (N, 43) top-first -> (N, 43, n_absorb)."""
    n = len(type_absorb)
    cols = []
    ul = ulgas[:, :, ::-1]                                              # top-first
    def conv(col, ip, extra=None):
        v = ul[:, col, :]
        if extra is not None:
            v = v + ul[:, extra, :]
        return v / (AVOG / (MOLAR_WEIGHT[ip] * LOSCHMIDT * 10.0)) / mass
    zero = jnp.zeros_like(mass)
    for ip in type_absorb.astype(int):
        if ip == 1:
            c = conv(0, 1)
        elif ip == 2:
            c = conv(1, 2) if flags["l_co2"] > 0.5 else zero
        elif ip == 3:
            c = conv(2, 3) if flags["l_o3"] > 0.5 else zero
        elif ip == 7:
            c = conv(3, 7) if flags["l_o2"] > 0.5 else zero
        elif ip == 10:
            c = conv(4, 10) if flags["l_no2"] > 0.5 else zero
        elif ip == 4:
            c = conv(5, 4) if flags["l_n2o"] > 0.5 else zero
        elif ip == 6:
            c = conv(6, 6) if flags["l_ch4"] > 0.5 else zero
        elif ip == 9:
            c = conv(12, 9) if flags["l_so2"] > 0.5 else zero
        elif ip == 13:
            c = zero + PN2 * MOLAR_WEIGHT[13] / MAIR if flags["l_n2"] > 0.5 else zero
        elif ip == 14:
            c = conv(7, 14, extra=10) if flags["l_cfc11"] > 0.5 else zero
        elif ip == 15:
            c = conv(8, 15) if flags["l_cfc12"] > 0.5 else zero
        elif ip == 36:
            c = zero + PAR * MOLAR_WEIGHT[36] / MAIR if flags["l_ar"] > 0.5 else zero
        elif ip == 23:
            c = zero + PH2 * MOLAR_WEIGHT[23] / MAIR if flags["l_h2"] > 0.5 else zero
        else:
            c = zero                                                    # not set by planet_rad (CFC113, HCFC22, HFC134a, OCS, ...)
        cols.append(c)
    return jnp.stack(cols, axis=2)


# ---------------------------------------------------------------------------------------------------------------- clouds
def cloud_field(c, st):
    """set_cloud_field + the Cairns branch of set_cld.  c: unpacked tag-50 dict; st: static dict of the spectrum (flags, dim_lim).
    Returns w_cloud (N,43), frac_cloud (N,43,4), condensed_mix_ratio (N,43,4), condensed_dim_char (N,43,4), rel_var_dens (N,43,4)."""
    w = c["w_cloud"]
    frac = jnp.stack([c["frac_st_water"], c["frac_st_ice"], c["frac_cnv_water"], c["frac_cnv_ice"]], axis=2)
    scl = c["cld_scaling"][:, None]
    fa_st, fa_cnv = c["frac_area_st"], c["frac_area_cnv"]
    mix = jnp.stack([scl * c["mix_st_water"] / (fa_st + TINY), scl * c["mix_st_ice"] / (fa_st + TINY),
                     scl * c["mix_cnv_water"] / (fa_cnv + TINY), scl * c["mix_cnv_ice"] / (fa_cnv + TINY)], axis=2)
    if st["l_microphysics"] > 0.5:
        dw_st, dw_cnv = c["dim_st_water"], c["dim_cnv_water"]
    else:
        dw_st = dw_cnv = jnp.full_like(w, 7.0e-06)
    if int(st["i_cloud_ice_size_scheme"]) != 1:
        raise NotImplementedError("i_cloud_ice_size_scheme /= 1 (UM aggregate size) is not ported")
    dim = jnp.stack([dw_st, 2.0 * c["dim_st_ice"], dw_cnv, 2.0 * c["dim_cnv_ice"]], axis=2)
    tol = 1.0e-10
    mix = jnp.where(frac <= tol, 0.0, mix)
    lim = st["dim_lim"]                                                 # [st_water min, max, cnv_water min, max, st_ice, cnv_ice]
    mn = jnp.asarray([lim[0], lim[4], lim[2], lim[6]]); mx = jnp.asarray([lim[1], lim[5], lim[3], lim[7]])
    dim = jnp.maximum(mn, jnp.minimum(mx, dim))
    eps = c["cldeps"][:, ::-1]
    if int(st["i_inhom"]) != 3:                                          # ip_cairns (rad_pcf)
        raise NotImplementedError("inhomogeneity scheme /= Cairns is not ported")
    rel = jnp.broadcast_to((eps / (1.0 - eps))[:, :, None], mix.shape)
    return w, frac, mix, dim, rel


# ---------------------------------------------------------------------------------------------------------------- aerosols
def interp_aer(A, comp, radius, rh):
    """interp_aer for one component and a column batch.  comp: int component id; radius (N,); rh (N,43) -> k_abs, k_scat, g (N,43,n_band)."""
    nh = A["n_hum"]
    it = int(A["comp_map"][comp - 1]) - 1                               # component index in the table (0-based)
    nr = int(A["n_radius"][it])
    if A["l_humidity"][it]:
        delta = 1.0 / (nh - 1)
        ip = jnp.minimum(1 + jnp.floor(rh * (nh - 1)).astype(jnp.int32), nh - 1)          # 1-based pointer
        hum = jnp.asarray(A["humidity"])
        wgt = (jnp.minimum(1.0, rh) - hum[ip - 1]) / delta
        out = []
        for T in (A["k_abs"], A["k_scat"], A["g_asym"]):
            lo = jnp.asarray(T[:, :, :nr, it])                          # (band, hum, rad)
            a = lo[:, ip - 1, :]                                        # (band, N, 43, rad)
            b = lo[:, ip, :]
            w_ = wgt[None, :, :, None]
            out.append((1.0 - w_) * a + w_ * b)                         # (band, N, 43, rad)
    else:
        out = [jnp.broadcast_to(jnp.asarray(T[:, 0, :nr, it])[:, None, None, :], (T.shape[0],) + rh.shape + (nr,))
               for T in (A["k_abs"], A["k_scat"], A["g_asym"])]
    rad = jnp.asarray(A["radius"][:nr, it])
    below = radius <= rad[0]
    above = radius >= rad[nr - 1]
    i0 = jnp.searchsorted(rad, radius, side="left")                     # first index with rad >= radius (0-based)
    ptr = jnp.where(below, 0, jnp.where(above, nr - 2, i0 - 1))         # 0-based lower index
    w_r = jnp.where(below, 0.0, jnp.where(above, 1.0, (radius - rad[ptr]) / (rad[ptr + 1] - rad[ptr])))
    res = []
    for t in out:                                                       # (band, N, 43, rad)
        lo = jnp.take_along_axis(t, ptr[None, :, None, None].astype(jnp.int32), axis=3)[..., 0]
        hi = jnp.take_along_axis(t, (ptr + 1)[None, :, None, None].astype(jnp.int32), axis=3)[..., 0]
        w_ = w_r[None, :, None]
        r = (1.0 - w_) * lo + w_ * hi                                   # (band, N, 43)
        res.append(jnp.transpose(r, (1, 2, 0)))                         # (N, 43, band)
    return res


def aerosol_modes(A, comp, radius, mass_div_dens, frac_sulph, scaling, rh, atm_mass, n_mode_max=50):
    """calc_aer_opt_prop.  comp (n_aer,) ints; radius (N, n_aer); mass_div_dens (N, n_aer, 43); frac_sulph (N, n_aer); scaling (N, n_aer);
    rh (N, 43); atm_mass (N, 43).  Returns mode_mix_ratio (N,43,n_aer), absorption/scattering/asymmetry (N,43,n_aer,n_band)."""
    n_aer = comp.shape[0]
    mix, ab, sc, asy = [], [], [], []
    dens = A["density"]
    for i in range(n_aer):
        ci = int(comp[i]); it = int(A["comp_map"][ci - 1]) - 1
        m = mass_div_dens[:, i, :] * dens[it] / atm_mass
        m = m * scaling[:, i][:, None]
        mix.append(m)
        k_abs, k_scat, g = interp_aer(A, ci, radius[:, i], rh)
        fs = frac_sulph[:, i][:, None, None]
        # sulphate fraction > 0: add the ammonium sulphate optical properties (ip_ammonium_sulphate = 7 in rad_pcf)
        ka2, ks2, g2 = interp_aer(A, IP_AMMONIUM_SULPHATE, radius[:, i], rh)
        d_s = dens[int(A["comp_map"][IP_AMMONIUM_SULPHATE - 1]) - 1]
        d_i = dens[it]
        use = fs > 0.0
        omf = 1.0 - fs
        ab_s = omf * k_abs + fs * ka2 * d_s / d_i
        sc_s = omf * k_scat + fs * ks2 * d_s / d_i
        g_s = (omf * k_scat * g + fs * ks2 * g2) / (omf * k_scat + fs * ks2)
        ab.append(jnp.where(use, ab_s, k_abs)); sc.append(jnp.where(use, sc_s, k_scat)); asy.append(jnp.where(use, g_s, g))
    return (jnp.stack(mix, axis=2), jnp.stack(ab, axis=2), jnp.stack(sc, axis=2), jnp.stack(asy, axis=2))


# ---------------------------------------------------------------------------------------------------------------- surface
ALB_BAND_MIN_LW = 1.0e2 * np.array([0.0, 100.0, 230.0, 400.0, 570.0, 610.0, 650.0, 690.0, 730.0, 760.0, 790.0, 850.0, 900.0, 970.0,
                                    1010.0, 1070.0, 1210.0, 1330.0, 1450.0, 1590.0, 1720.0, 1900.0, 2020.0, 2080.0, 2130.0, 2210.0,
                                    2250.0, 2300.0, 2380.0, 2390.0])
ALB_BAND_MAX_LW = 1.0e2 * np.array([100.0, 230.0, 400.0, 570.0, 610.0, 650.0, 690.0, 730.0, 760.0, 790.0, 850.0, 900.0, 970.0, 1010.0,
                                    1070.0, 1210.0, 1330.0, 1450.0, 1590.0, 1720.0, 1900.0, 2020.0, 2080.0, 2130.0, 2210.0, 2250.0,
                                    2300.0, 2380.0, 2390.0, 2500.0])
ALB_SEA_LW = np.array([0.150, 0.140, 0.120, 0.110, 0.100, 0.100, 0.097, 0.091, 0.081, 0.070, 0.053, 0.040, 0.036, 0.041, 0.045, 0.052,
                       0.056, 0.058, 0.059, 0.061, 0.062, 0.063, 0.063, 0.064, 0.064, 0.064, 0.064, 0.065, 0.065, 0.065])
ALB_SOIL_LW = np.array([0.002, 0.006, 0.036, 0.130, 0.063, 0.042, 0.030, 0.033, 0.041, 0.045, 0.040, 0.045, 0.076, 0.120, 0.180, 0.190,
                        0.025, 0.021, 0.033, 0.039, 0.043, 0.030, 0.040, 0.051, 0.070, 0.086, 0.095, 0.110, 0.120, 0.130])
ALB_SNOW_LW = np.array([0.280, 0.059, 0.082, 0.120, 0.068, 0.047, 0.031, 0.025, 0.022, 0.020, 0.017, 0.014, 0.013, 0.014, 0.021, 0.020,
                        0.017, 0.012, 0.009, 0.008, 0.017, 0.026, 0.025, 0.022, 0.016, 0.011, 0.009, 0.006, 0.004, 0.0006])
ALB_ICE_LW = ALB_SNOW_LW.copy()
ALB_VEG_LW = np.zeros(30)
VEG_DEPTH = np.array([0.1, 0.2, 0.2, 0.5, 2.0, 5.0, 10.0, 25.0, 0.2, 0.1, 1.0e-05, 0.2])
CHAR_AMOUNT_SNOW = 10.0


def alb_segments(rad_min, rad_max):
    """calc_alb_band_wgt for the Planck weighting: list of (alb_band_index, nu_lo, nu_hi) in the order of accumulation."""
    bmin, bmax = ALB_BAND_MIN_LW, ALB_BAND_MAX_LW
    n = len(bmin)
    if rad_min <= bmin[0] and rad_max <= bmin[0]:
        return [(0, rad_min, rad_max)]
    if rad_min >= bmax[n - 1] and rad_max >= bmax[n - 1]:
        return [(n - 1, rad_min, rad_max)]
    ok1 = rad_min >= bmin
    if ok1.any():                                                       # MINLOC(ABS(rad_min - bmin), mask)
        d = np.where(ok1, np.abs(rad_min - bmin), np.inf)
        ind1 = int(np.argmin(d)) + 1
    else:
        ind1 = 0
    ok2 = rad_max <= bmax
    if ok2.any():
        d = np.where(ok2, np.abs(rad_max - bmax), np.inf)
        ind2 = int(np.argmin(d)) + 1
    else:
        ind2 = 0
    if ind1 != 0 and ind1 == ind2:
        return [(ind1 - 1, rad_min, rad_max)]
    seg = []
    if ind1 == 0:
        seg.append((0, rad_min, bmin[0]))
    else:
        seg.append((ind1 - 1, rad_min, bmax[ind1 - 1]))
    if ind2 == 0:
        seg.append((n - 1, bmax[n - 1], rad_max))
        ind2 = n + 1
    else:
        seg.append((ind2 - 1, bmin[ind2 - 1], rad_max))
    for i in range(ind1 + 1, ind2):
        seg.append((i - 1, bmin[i - 1], bmax[i - 1]))
    return seg


def integ_planck(temp, nu_min, nu_max, t_min=100.0):
    """integ_planck: trapezoid rule with delta_u = 0.1, n_point = max(nint(du/0.1)+1, 2).  The padded number of points is a static bound for
    temperatures >= t_min (jit-able); below t_min the integral would be truncated (the model's Planck table starts at 100 K)."""
    alpha = K_BOLTZMANN / (H_PLANCK * C_LIGHT)
    q = 2.0 * PI * H_PLANCK * alpha ** 4 * C_LIGHT ** 2
    u_min = nu_min / (alpha * temp)
    u_max = nu_max / (alpha * temp)
    x = (u_max - u_min) / 0.1
    n_point = jnp.maximum(jnp.floor(x + 0.5).astype(jnp.int32) + 1, 2)         # Fortran NINT: half away from zero (x > 0)
    du = (u_max - u_min) / (n_point - 1)
    nmax = int(np.ceil((nu_max - nu_min) / (alpha * t_min) / 0.1)) + 3
    i = jnp.arange(nmax)
    u = u_min[:, None] + du[:, None] * i[None, :]
    f = u ** 3 * jnp.exp(-u) / (1.0 - jnp.exp(-u))
    s = f[:, :-1] + f[:, 1:]
    s = jnp.where(i[None, :nmax - 1] < (n_point[:, None] - 1), s, 0.0)
    return q * temp ** 4 * (0.5 * du * jnp.sum(s, axis=1))


def lw_alb_weights(temp, wl_short, wl_long, n_excl, idx_excl):
    """calc_alb_wgt (Planck weighting) for temperatures temp (M,) -> (M, 30, n_band)."""
    nb = len(wl_short)
    out = []
    for ib in range(nb):
        seg = alb_segments(1.0 / wl_long[ib], 1.0 / wl_short[ib])
        w = [jnp.zeros_like(temp) for _ in range(30)]
        for j, lo, hi in seg:
            w[j] = w[j] + integ_planck(temp, lo, hi)
        out.append(jnp.stack(w, axis=1))                                # (M, 30)
    W = jnp.stack(out, axis=2)                                          # (M, 30, nb)
    for ib in range(nb):
        for k in range(int(n_excl[ib])):
            W = W.at[:, :, ib].set(W[:, :, ib] - W[:, :, int(idx_excl[k, ib]) - 1])
    norm = jnp.zeros((W.shape[0], nb))
    for j in range(30):
        norm = norm + W[:, j, :]
    return jnp.where(norm[:, None, :] < 1.0e10 * TINY, 0.0, W / jnp.where(norm == 0.0, 1.0, norm)[:, None, :])


def _wsum(alb, W):
    """SUM(alb*alb_wgt(:,ib)) for each ib: alb (30,) static, W (M, 30, nb) -> (M, nb); sequential over j as the Fortran intrinsic."""
    acc = None
    for j in range(W.shape[1]):
        t = float(alb[j]) * W[:, j, :]
        acc = t if acc is None else acc + t
    return acc


def surface_lw(c, st, n_band):
    """set_surf_prop_lw (Hansen sea-ice scheme 0 / Lacis 1) + set_tiling.  c: unpacked tag-50 dict.
    Returns rho_alb (N, 2, n_band), t_ground (N,), tiles dict(flag (N,), n_tile, frac (N,4), t (N,4), rho (N,2,4,n_band))."""
    N = c["frac_sea"].shape[0]
    fs, fl, fi, fli = c["frac_sea"], c["frac_land"], c["frac_sea_ice"], c["frac_land_ice"]
    ws, wl, we, wi = c["wind"], None, c["soil_wetness"], None
    ts, tl, ti, tli = c["t_sea"], c["t_land"], c["t_sea_ice"], c["t_land_ice"]
    wl_s, wl_l = st["wl_short"], st["wl_long"]
    W = {}
    for key, T in (("sea", ts), ("land", tl), ("ice", ti), ("lice", tli)):
        W[key] = lw_alb_weights(T, wl_s, wl_l, st["n_excl"], st["idx_excl"])
    zero = jnp.zeros((N, n_band))
    rho = zero
    # sea
    x = 1.0 / (1.0 + ws)
    av = -0.0147087 * x ** 2 + 0.0292266 * x - 0.0081079
    bv = 1.01673 - 0.0083652 * ws
    a_sea_raw = _wsum(ALB_SEA_LW, W["sea"])
    a_sea = av[:, None] + bv[:, None] * a_sea_raw
    rho = jnp.where((fs > 0.0)[:, None], rho + a_sea * fs[:, None], rho)
    tg = jnp.where(fs > 0.0, 0.0 + fs * ts, 0.0)
    # land
    sd = c["snow_depth_land"]; fv = c["frac_veg"]; fsn = c["frac_land_snow"]
    snowy = (sd[:, 0] + sd[:, 1]) > 1.0e-03
    soil_s = fv[:, 0] * (1.0 - fsn[:, 0] * (1.0 - jnp.exp(-sd[:, 0] / VEG_DEPTH[0]))) \
        + fv[:, 9] * (1.0 - fsn[:, 0] * (1.0 - jnp.exp(-sd[:, 0] / VEG_DEPTH[9])))
    veg_s = jnp.zeros(N)
    for iv in range(1, 12):
        if iv in (9, 10):
            continue
        veg_s = veg_s + fv[:, iv] * (1.0 - fsn[:, 1] * (1.0 - jnp.exp(-sd[:, 1] / VEG_DEPTH[iv])))
    snow_s = 1.0 - (soil_s + veg_s)
    soil_n = fv[:, 0] + fv[:, 9]
    f_soil = jnp.where(snowy, soil_s, soil_n)
    f_veg = jnp.where(snowy, veg_s, 1.0 - soil_n)
    f_snow = jnp.where(snowy, snow_s, 0.0)
    a_soil = _wsum(ALB_SOIL_LW, W["land"])
    a_veg = _wsum(ALB_VEG_LW, W["land"])
    a_snow = _wsum(ALB_SNOW_LW, W["land"])
    r_land = rho + a_soil * ((1.0 - we) * f_soil * fl)[:, None]
    r_land = r_land + a_veg * (f_veg * fl)[:, None]
    r_land = r_land + a_snow * (f_snow * fl)[:, None]
    rho = jnp.where((fl > 0.0)[:, None], r_land, rho)
    tg = jnp.where(fl > 0.0, tg + fl * tl, tg)
    t_rho_land = (a_soil * ((1.0 - we) * f_soil)[:, None] + a_veg * f_veg[:, None]) + a_snow * f_snow[:, None]
    # sea ice
    if int(st["i_sea_ice_alb"]) == 0:
        zs = c["snow_depth_sea_ice"]
        fsc = jnp.where(zs > 0.0, jnp.where(zs < 0.1, zs / 0.1, 1.0), 0.0)
        fbare = 1.0 - fsc
    else:
        fbare = jnp.exp(-c["snow_amnt_sea_lake"] / CHAR_AMOUNT_SNOW)
        fsc = 1.0 - fbare
    a_isnow = _wsum(ALB_SNOW_LW, W["ice"])
    a_iice = _wsum(ALB_ICE_LW, W["ice"])
    r_ice = rho + a_isnow * (fsc * fi)[:, None]
    r_ice = r_ice + a_iice * (fbare * fi)[:, None]
    rho = jnp.where((fi > 0.0)[:, None], r_ice, rho)
    tg = jnp.where(fi > 0.0, tg + fi * ti, tg)
    t_rho_ice = a_isnow * fsc[:, None]
    t_rho_ice = t_rho_ice + a_iice * fbare[:, None]
    # land ice
    a_li = _wsum(ALB_ICE_LW, W["lice"])
    rho = jnp.where((fli > 0.0)[:, None], rho + a_li * fli[:, None], rho)
    tg = jnp.where(fli > 0.0, tg + fli * tli, tg)
    # tiles
    part = ((fs > 0) & (fs < 1)) | ((fl > 0) & (fl < 1)) | ((fi > 0) & (fi < 1)) | ((fli > 0) & (fli < 1))
    tile_rho = jnp.stack([a_sea, t_rho_ice, t_rho_land, a_li], axis=1)           # (N, 4, nb): ocean, seaice, land, landice
    tile_frac = jnp.where(part[:, None], jnp.stack([jnp.where(fs > 0, fs, 0.0), jnp.where(fi > 0, fi, 0.0),
                                                    jnp.where(fl > 0, fl, 0.0), jnp.where(fli > 0, fli, 0.0)], axis=1), 0.0)
    tile_t = jnp.stack([jnp.where(fs > 0, ts, 0.0), jnp.where(fi > 0, ti, 0.0), jnp.where(fl > 0, tl, 0.0),
                        jnp.where(fli > 0, tli, 0.0)], axis=1)
    tile_rho = jnp.where(jnp.stack([fs > 0, fi > 0, fl > 0, fli > 0], axis=1)[:, :, None], tile_rho, 0.0)
    return rho, tg, dict(flag=part, frac=tile_frac, t=tile_t, rho=tile_rho)


def surface_sw(c, alb_wgt, n_band):
    """set_surf_prop_sw: rho_alb (N, 2, n_band) (basis 0 diffuse, 1 direct), tile rho (N, 2, 4, n_band)."""
    prnb = c["prnb"].reshape(-1, 4, 6)                                  # Fortran (6,4): index [type, band6]
    prnx = c["prnx"].reshape(-1, 4, 6)
    N = prnb.shape[0]
    fr = [c["frac_sea"], c["frac_land"], c["frac_sea_ice"], c["frac_land_ice"]]
    wgt = jnp.asarray(alb_wgt)                                          # (6 alb bands, n_band)
    def band(a):                                                        # (N, 6) -> (N, n_band), sequential over alb bands
        acc = None
        for j in range(6):
            t = a[:, j][:, None] * wgt[j][None, :]
            acc = t if acc is None else acc + t
        return acc
    rd = jnp.zeros((N, n_band)); rx = jnp.zeros((N, n_band))
    tiles_d, tiles_x = [], []
    order = [0, 1, 2, 3]                                                # sea, land, sea ice, land ice
    for t in order:
        ad = band(prnb[:, t, :]); ax = band(prnx[:, t, :])
        on = (fr[t] > 0.0)[:, None]
        rd = jnp.where(on, rd + ad * fr[t][:, None], rd)
        rx = jnp.where(on, rx + ax * fr[t][:, None], rx)
        tiles_d.append(jnp.where(on, ad, 0.0)); tiles_x.append(jnp.where(on, ax, 0.0))
    # tile order of set_tiling: ocean 1, sea ice 2, land 3, land ice 4
    td = jnp.stack([tiles_d[0], tiles_d[2], tiles_d[1], tiles_d[3]], axis=1)
    tx = jnp.stack([tiles_x[0], tiles_x[2], tiles_x[1], tiles_x[3]], axis=1)
    return jnp.stack([rd, rx], axis=1), jnp.stack([td, tx], axis=1)


# ---------------------------------------------------------------------------------------------------------------- post-processing
def total_cloud_cover(w_cloud, frac_cloud):
    """radout%tot_cloud_cover of overlap_coupled (ip_cloud_triple): w_cloud (N, 43), frac_cloud (N, 43, 4) -> (N,).  For any cloud top
    this equals the batched n_cloud_top = 1 evaluation (the factors above the top cloud are exactly 1)."""
    eps = float(np.finfo(np.float64).eps)
    tol = 1.0e2 * eps
    N, nl = w_cloud.shape
    w_free = 1.0 - w_cloud
    fr1 = frac_cloud[:, :, 0] + frac_cloud[:, :, 1]
    fr2 = frac_cloud[:, :, 2] + frac_cloud[:, :, 3]
    a1 = w_cloud * jnp.where(fr1 > TINY, fr1, 0.0)
    a2 = w_cloud * jnp.where(fr2 > TINY, fr2, 0.0)
    z = jnp.zeros((N, 1))
    ao22 = jnp.minimum(a1, jnp.concatenate([z, a1[:, :-1]], axis=1))
    ao33 = jnp.minimum(a2, jnp.concatenate([z, a2[:, :-1]], axis=1))

    def step(tcc, x):
        wf, o2, o3 = x
        upd = tcc * wf / (1.0 - o2 - o3)
        return jnp.where(wf > tol, upd, 0.0), None

    tcc, _ = jax.lax.scan(step, jnp.ones(N), (w_free.T, ao22.T, ao33.T))
    return 1.0 - tcc


def post_lw(dn_b, up_b, t_level):
    """get_planet_radout, long-wave part.  dn_b, up_b: lists of (N, 44) per-band fluxes (Planck add-back included, level 0 = top)."""
    up = None; dn = None
    for u, d in zip(up_b, dn_b):                                        # cumulative channel (adjust_ir_radiance), sequential over bands
        up = u if up is None else up + u
        dn = d if dn is None else dn + d
    truflb, trdflb = up[:, ::-1], dn[:, ::-1]
    trnflb = truflb - trdflb
    return dict(tlb=t_level[:, ::-1], tlt=t_level[:, :-1][:, ::-1], truflb=truflb, trdflb=trdflb, trnflb=trnflb,
                trfcrl=trnflb[:, 1:] - trnflb[:, :-1], truptoa=jnp.stack([u[:, 0] for u in up_b], axis=1))


def post_sw(dn_b, up_b, dr_b, lit, cos_zen, weight_blue, frac, tile_up, solar_irrad, norms0=1):
    """get_planet_radout, short-wave part.  dn_b/up_b/dr_b: lists (per band = channel) of (N, 44); lit (N,) bool; cos_zen (N,);
    frac: dict sea/land/ice/lice fractions; tile_up: (N, 4, nb) or None (tile order ocean, sea ice, land, land ice) already /cos_zen."""
    N = dn_b[0].shape[0]
    nb = len(dn_b)
    cz = jnp.where(lit, cos_zen, 1.0)[:, None]
    if norms0 > 0:
        dn_n = [d / cz for d in dn_b]; up_n = [u / cz for u in up_b]
    else:
        dn_n, up_n = list(dn_b), list(up_b)
    dn_n = [jnp.where(lit[:, None], d, 0.0) for d in dn_n]; up_n = [jnp.where(lit[:, None], u, 0.0) for u in up_n]
    dr_n = [jnp.where(lit[:, None], d, 0.0) for d in dr_b]
    def chsum(lst):
        acc = None
        for x in lst:
            acc = x if acc is None else acc + x
        return acc
    sruflb = chsum(up_n)[:, ::-1]; srdflb = chsum(dn_n)[:, ::-1]
    srnflb = srdflb - sruflb
    srfhrl = srnflb[:, 1:] - srnflb[:, :-1]
    wb = jnp.asarray(weight_blue)
    def wsum(lst, lev, w):
        acc = None
        for b in range(nb):
            t = lst[b][:, lev] * w[b]
            acc = t if acc is None else acc + t
        return acc
    one_m = 1.0 - wb
    srivis = wsum(dn_n, 0, wb); srovis = wsum(up_n, 0, wb)
    plavis = jnp.where(lit, srovis / jnp.where(lit, srivis, 1.0), 1.0)
    srinir = wsum(dn_n, 0, one_m); sronir = wsum(up_n, 0, one_m)
    planir = jnp.where(lit, sronir / jnp.where(lit, srinir, 1.0), 1.0)
    srdvis = wsum(dn_n, -1, wb); sruvis = wsum(up_n, -1, wb)
    albvis = jnp.where(lit, sruvis / (srdvis + TINY), 1.0)
    srdnir = wsum(dn_n, -1, one_m); srunir = wsum(up_n, -1, one_m)
    albnir = jnp.where(lit, srunir / (srdnir + TINY), 1.0)
    d0v = wsum(dr_n, 0, wb); dnv = wsum(dr_n, -1, wb)
    d0n = wsum(dr_n, 0, one_m); dnn = wsum(dr_n, -1, one_m)
    srxvis = jnp.where(lit, dnv / jnp.where(lit, d0v, 1.0), 0.0)
    srxnir = jnp.where(lit, dnn / jnp.where(lit, d0n, 1.0), 0.0)
    return dict(sruflb=sruflb, srdflb=srdflb, srnflb=srnflb, srfhrl=srfhrl, srivis=srivis, srovis=srovis, plavis=plavis, srinir=srinir,
                sronir=sronir, planir=planir, srdvis=srdvis, sruvis=sruvis, albvis=albvis, srdnir=srdnir, srunir=srunir, albnir=albnir,
                srxvis=srxvis, srxnir=srxnir, s0=solar_irrad, sruptoa=jnp.stack([u[:, 0] for u in up_n], axis=1),
                srdntoa=jnp.stack([d[:, 0] for d in dn_n], axis=1))


def fsrnfg(post, frac, tile_flag, tile_up):
    """FSRNFG(1:4) = ocean, land, sea ice, land ice (GISS order).  post: post_sw dict; tile_up (N, 4 tiles, nb) or None."""
    srd0 = post["srdflb"][:, 0]; srn0 = post["srnflb"][:, 0]
    fs, fl, fi, fli = frac
    tot = lambda k: sum_seq([tile_up[:, k, b] for b in range(tile_up.shape[2])])
    if tile_up is not None:
        t_oc = jnp.where(fs > 0.0, srd0 - tot(0), srd0)
        t_la = jnp.where(fl > 0.0, srd0 - tot(2), srd0)
        t_si = jnp.where(fi > 0.0, srd0 - tot(1), srd0)
        t_li = jnp.where(fli > 0.0, srd0 - tot(3), srd0)
    else:
        t_oc = t_la = t_si = t_li = srd0
    n_oc = jnp.where(fs > 0.0, srn0, srd0)
    n_la = jnp.where((fs <= 0.0) & (fl > 0.0), srn0, srd0)
    n_si = jnp.where((fs <= 0.0) & (fl <= 0.0) & (fi > 0.0), srn0, srd0)
    n_li = jnp.where((fs <= 0.0) & (fl <= 0.0) & (fi <= 0.0) & (fli > 0.0), srn0, srd0)
    f = lambda a, b: jnp.where(tile_flag, a, b)
    return jnp.stack([f(t_oc, n_oc), f(t_la, n_la), f(t_si, n_si), f(t_li, n_li)], axis=1)


def sum_seq(lst):
    acc = None
    for x in lst:
        acc = x if acc is None else acc + x
    return acc
