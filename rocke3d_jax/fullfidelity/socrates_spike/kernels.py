"""SOCRATES spike S2: JAX prototypes of two hot kernels of the RADIA call (new file; no SOCRATES/ModelE source is changed).

Kernel A  aer_optical(): ModelE planet_rad.F90 set_aer -> calc_aer_opt_prop -> interp_aer (humidity and effective-radius interpolation of the
          aerosol table, sulphate mixing).  Same arithmetic order as the Fortran; the Fortran interpolates in humidity for ALL radii of
          the table and then picks two, here only the two radii that are used are gathered (same result, since the other radii do not enter).
Kernel B  two_stream_homogen(): clear-region chain two_coeff_basic -> sqrt -> solar_coefficient_basic -> trans_source_coeff ->
          solver_homogen_direct of SOCRATES radiance_core (layer adding method as lax.scan over layers).  Sources of the layers
          (s_up, s_down) are passed in: the real ones come from single_solar_source / ir_source (not ported here).
Both are float64.  Column/term/region dimensions are batch dimensions (leading axis after the layer axis).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ELSASSER = 1.66
IP_PIFM80, IP_ELSASSER = "pifm80", "elsasser"


# ------------------------------------------------------------------------------------------------------------- kernel A
def radius_pointer(r, rad_dry_c, n_radius):
    """Host side, as in interp_aer: returns (0-based lower pointer, weight of the upper radius).  Out-of-table values are clamped
    (the Fortran also prints a warning)."""
    rad = np.asarray(rad_dry_c)[:n_radius]
    if r <= rad[0]:
        return 0, 0.0
    if r >= rad[n_radius - 1]:
        return n_radius - 2, 1.0
    j = int(np.argmax(r <= rad))
    p = j - 1
    return p, (r - rad[p]) / (rad[p + 1] - rad[p])


def aer_table_to_arrays(nc_path):
    """Host: the NetCDF aerosol table -> dict of numpy arrays with the file's layout (component, radius, humidity, band)."""
    import netCDF4 as nc
    d = nc.Dataset(nc_path)
    out = {k: np.array(d.variables[k][:]) for k in
           ("k_abs", "k_scat", "g_asym", "humidity", "n_radius", "radius_eff_dry", "l_humidity", "density")}
    return out


def _interp_one(K, ic, p_r, w_r, ip, w_h):
    """K: [C,R,H,B] table; ic component, p_r/w_r radius pointer/weight (python scalars), ip [N,L] int humidity pointer, w_h [N,L].
    Humidity first, radius second (Fortran order).  Returns [N,L,B]."""
    lo = K[ic, p_r][ip]            # [N,L,B]  (radius p_r, humidity ip)
    hi = K[ic, p_r][ip + 1]
    a = (1.0 - w_h)[..., None] * lo + w_h[..., None] * hi
    lo2 = K[ic, p_r + 1][ip]
    hi2 = K[ic, p_r + 1][ip + 1]
    b = (1.0 - w_h)[..., None] * lo2 + w_h[..., None] * hi2
    return (1.0 - w_r) * a + w_r * b


def aer_optical(tbl, rh, mass, comp_idx, radius, mass_div_dens, frac_sulph, scaling, ic_sulph, hum_flag=None):
    """tbl: dict from aer_table_to_arrays moved to device (k_abs,k_scat,g_asym [C,R,H,B]; humidity [H]; density [C]).
    rh [N,L]; mass [N,L] (column mass of the layer); comp_idx, radius, frac_sulph, scaling: python lists of length A (static);
    mass_div_dens [A,N,L]; ic_sulph: table index of ammonium sulphate.
    Returns mix_ratio [N,L,A], absorption/scattering/asymmetry [N,L,A,B]."""
    H = tbl["humidity"].shape[0]
    dh = 1.0 / (H - 1)
    ip = jnp.minimum(jnp.floor(rh * (H - 1)).astype(jnp.int32), H - 2)          # = MIN(1+INT(rh*(H-1)), H-1) - 1
    w_h = (jnp.minimum(1.0, rh) - tbl["humidity"][ip]) / dh
    rad_np, nrad_np, lhum_np = tbl["_rad_np"], tbl["_nrad_np"], tbl["_lhum_np"]
    mixs, ab, sc, asy = [], [], [], []
    for a in range(len(comp_idx)):
        ic = comp_idx[a]
        def props(c):
            p_r, w_r = radius_pointer(radius[a], rad_np[c], int(nrad_np[c]))
            if lhum_np[c]:
                i_, w_ = ip, w_h
            else:
                i_, w_ = jnp.zeros_like(ip), jnp.zeros_like(w_h)
            return [_interp_one(tbl[k], c, p_r, w_r, i_, w_) for k in ("k_abs", "k_scat", "g_asym")]
        k_abs, k_scat, g = props(ic)
        mix = mass_div_dens[a] * tbl["density"][ic] / mass * scaling[a]
        if frac_sulph[a] > 0.0:
            ks_abs, ks_scat, gs = props(ic_sulph)
            f = frac_sulph[a]
            ratio = tbl["density"][ic_sulph] / tbl["density"][ic]
            ab.append((1.0 - f) * k_abs + f * ks_abs * ratio)
            sc.append((1.0 - f) * k_scat + f * ks_scat * ratio)
            asy.append(((1.0 - f) * k_scat * g + f * ks_scat * gs) / ((1.0 - f) * k_scat + f * ks_scat))
        else:
            ab.append(k_abs); sc.append(k_scat); asy.append(g)
        mixs.append(mix)
    return jnp.stack(mixs, -1), jnp.stack(ab, 2), jnp.stack(sc, 2), jnp.stack(asy, 2)


# ------------------------------------------------------------------------------------------------------------- kernel B
def two_coeff(tau, tau_noscal, omega, g, scheme, solar, sec_0=None, l_ir_quad=True):
    """Port of two_coeff -> two_coeff_basic, solar_coefficient_basic, trans_source_coeff (same operation order), any array shape.
    Returns dict: trans, reflect, and for solar trans_0, trans_0_noscal, s_up / s_down (source_coeff solar up / down); for the infra-red
    ir_1d and (if l_ir_quad) ir_2d (source_coeff 1 and 2).  Used by two_stream_homogen and by r2_kernel_b_check.py (tags 40-41)."""
    sq_eps_r = np.sqrt(np.finfo(np.float64).eps)
    tol = np.sqrt(sq_eps_r)
    if scheme == IP_PIFM80:
        sm = 2.0 - 1.5 * omega * g - 0.5 * omega
        df = 2.0 * (1.0 - omega)
    elif scheme == IP_ELSASSER:
        sm = ELSASSER - 1.5 * omega * g
        df = ELSASSER * (1.0 - omega)
    else:
        raise ValueError(scheme)
    lam = jnp.sqrt(sm * df)
    out = {}
    if solar:
        tol_p = 32.0 * np.finfo(np.float64).eps
        near = jnp.abs(lam - sec_0) < tol_p
        sm = jnp.where(near, (1 + tol_p) * sm, sm)
        df = jnp.where(near, (1 + tol_p) * df, df)
        lam = jnp.where(near, (1 + tol_p) * lam, lam)
        ksi0 = 1.5 * g / sec_0
        fac = 0.5 * omega * sec_0 / ((lam - sec_0) * (lam + sec_0))
        gam_up = fac * (sm - sec_0 - ksi0 * (df - sec_0))
        gam_dn = fac * (sm + sec_0 + ksi0 * (df + sec_0))
    e = jnp.exp(-lam * tau)
    e2 = e * e
    gm = (sm - lam) / (sm + lam)
    gm2 = gm * gm
    inv = 1.0 / (1.0 - e2 * gm2)
    trans = e * (1.0 - gm2) * inv
    refl = gm * (1.0 - e2) * inv
    out["trans"], out["reflect"] = trans, refl
    if solar:
        trans_0 = jnp.exp(-tau * sec_0)
        out["trans_0"] = trans_0
        out["trans_0_noscal"] = jnp.exp(-tau_noscal * sec_0)
        out["s_up"] = (gam_up - refl * (1.0 + gam_dn)) - gam_up * trans * trans_0
        out["s_down"] = trans_0 * (1.0 + gam_dn - gam_up * refl) - (1.0 + gam_dn) * trans
    else:
        out["ir_1d"] = (1.0 - trans + refl + sq_eps_r) / (sq_eps_r + tau * sm)
        if l_ir_quad:
            q = jnp.where(tau > tol, -2.0 * (1.0 - trans - refl + sq_eps_r) / (df * tau + sq_eps_r), -2.0 + df * tau)
            out["ir_2d"] = -(1.0 + refl + trans + q) / (sm * tau + sq_eps_r)
    return out


def two_stream_homogen(tau, omega, g, scheme, solar, sec_0=None, diffuse_albedo=None, direct_albedo=None, flux_inc_down=None,
                       s_up=None, s_down=None, flux_direct_ground=None, d_planck_flux_surface=None):
    """All layer arrays [L, B] (layer axis first, top to bottom, B = flattened column x term x region); per-batch scalars [B].
    Returns flux_total [2L+2, B] in the SOCRATES layout (index 0 = up at TOA, 1 = down at TOA, 2i = up at bottom of layer i, ...).
    solar=True: PIFM80-style coefficients with the solar source; solar=False: infra-red.  s_up/s_down: layer source terms [L,B]."""
    L = tau.shape[0]
    c = two_coeff(tau, tau, omega, g, scheme, solar, sec_0)
    trans, refl = c["trans"], c["reflect"]
    if solar:
        trans_0, s_up_c, s_dn_c = c["trans_0"], c["s_up"], c["s_down"]
        # direct beam through the layers (flux_inc_down is the direct flux at TOA here)
        fdir_bot = flux_inc_down * jnp.cumprod(trans_0, axis=0)
        fdir_top = jnp.concatenate([flux_inc_down[None], fdir_bot[:-1]], 0)
        s_up_l = s_up_c * fdir_top
        s_dn_l = s_dn_c * fdir_top
        fdg = fdir_bot[-1]
        sup_sfc = (direct_albedo - diffuse_albedo) * fdg
    else:
        s_up_l, s_dn_l = s_up, s_down
        sup_sfc = (1.0 - diffuse_albedo) * d_planck_flux_surface

    def back(carry, x):
        alpha, sup = carry
        t, r, sd, su = x
        beta = 1.0 / (1.0 - alpha * r)
        gamma = alpha * t
        h = sup + alpha * sd
        a_new = r + beta * gamma * t
        sup_new = su + beta * t * h
        return (a_new, sup_new), (beta, gamma, h)

    (alpha1, sup1), (beta, gamma, h) = jax.lax.scan(back, (diffuse_albedo, sup_sfc), (trans, refl, s_dn_l, s_up_l), reverse=True)
    f_dn0 = flux_inc_down if not solar else jnp.zeros_like(flux_inc_down) + 0.0
    # in the SOCRATES call flux_inc_down is the diffuse incident flux at TOA (zero for the solar diffuse part)
    f_up0 = alpha1 * f_dn0 + sup1

    def fwd(f_dn, x):
        t, r, sd, b, gm_, hh = x
        f_up = b * (hh + gm_ * f_dn)
        f_dn_new = sd + t * f_dn + r * f_up
        return f_dn_new, (f_up, f_dn_new)

    _, (fu, fd) = jax.lax.scan(fwd, f_dn0, (trans, refl, s_dn_l, beta, gamma, h))
    out = jnp.stack([fu, fd], 1).reshape(2 * L, -1)
    return jnp.concatenate([f_up0[None], f_dn0[None], out], 0)
