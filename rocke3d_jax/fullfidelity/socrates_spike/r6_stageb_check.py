"""R6 stage B check: planet_side.py (planet_rad set-up of one radiation call) against the tag-0 arrays of the real radiance_calc inputs.
   JAX_PLATFORMS=cpu python r6_stageb_check.py cols.npz spec_dir [max_cols]"""
import sys, os
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import planet_side as P

z = np.load(sys.argv[1])
specdir = sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else None
spec = {"lw": np.load(os.path.join(specdir, "spec_lw.npz")), "sw": np.load(os.path.join(specdir, "spec_sw.npz"))}
S = P.static_from_npz(z)
c = P.unpack(z["p50"], P.PACK50)
c60 = P.unpack(z["p60"], P.PACK60)
N = z["p50"].shape[0]
sl = slice(0, nmax)
c = {k: jnp.asarray(v[sl]) for k, v in c.items()}
c["ulgas_g"] = c["ulgas"].reshape(-1, 13, 43) if c["ulgas"].ndim == 2 else None


def rep(name, mine, ref):
    mine = np.asarray(mine); ref = np.asarray(ref)
    assert mine.shape == ref.shape, (name, mine.shape, ref.shape)
    d = np.abs(mine - ref)
    eq = float((mine == ref).mean()) * 100
    rel = d / np.maximum(np.abs(ref), 1e-300)
    print("%-28s n=%-9d bitwise %6.2f %%  max abs %.3e  max rel %.3e" % (name, mine.size, eq, d.max() if d.size else 0, rel.max() if d.size else 0))


p_level = c["p_level"]; t_layer = c["t_layer"]
p_layer, t_level, mass, dens = P.layers(p_level, t_layer)
rep("t_level vs tag50", t_level, c["t_level"])
rep("p_layer (tag0 lw_1)", p_layer, z["lw_1"][sl, 0])
rep("t_level (tag0 lw_7)", t_level, z["lw_7"][sl, 0])
rep("mass (lw_5)", mass, z["lw_5"][sl, 0])
rep("density (lw_3)", dens, z["lw_3"][sl, 0])
for iso, nb in (("lw", 9), ("sw", 6)):
    ref = z[iso + "_4"][:, 0]
    idx = np.arange(N) if iso == "lw" else z["sw_idx"]
    idx = idx[idx < (nmax or N)]
    ci = {k: v[idx] for k, v in c.items() if k != "ulgas_g"}
    ul = ci["ulgas"].reshape(-1, 13, 43)
    _, _, ms, _ = P.layers(ci["p_level"], ci["t_layer"])
    mix = P.gas_mix_ratio(ul, ms, spec[iso]["type_absorb"], S[iso])
    rep("gas mix %s" % iso, mix, ref[:len(idx)])
    w, frac, cmix, dim, rel = P.cloud_field(ci, S[iso])
    rep("w_cloud %s" % iso, w, z[iso + "_10"][:len(idx), 0])
    rep("frac_cloud %s" % iso, frac, z[iso + "_11"][:len(idx), 0])
    rep("cond mix %s" % iso, cmix, z[iso + "_12"][:len(idx), 0])
    rep("cond dim %s" % iso, dim, z[iso + "_13"][:len(idx), 0])
    rep("rel var %s" % iso, rel, z[iso + "_14"][:len(idx), 0])
    n_aer = int(round(float(ci["n_aer"][0])))
    p52 = z["p52"][idx][:len(idx)]; p53 = z["p53"][idx][:len(idx)]
    comp = p52[0, :, 0].astype(int)
    rad = p52[:, :, 1]; fsu = p52[:, :, 2]; sc = p52[:, :, 3 if iso == "sw" else 4]
    rh = ci["rel_humidity"]
    mm, ab, sca, asy = P.aerosol_modes(S[iso]["aer"], comp, jnp.asarray(rad), jnp.asarray(p53), jnp.asarray(fsu), jnp.asarray(sc), rh, ms)
    rep("aer mix_ratio %s" % iso, mm, z[iso + "_27"][:len(idx), 0, :, :n_aer])
    rep("aer absorption %s" % iso, ab, z[iso + "_28"][:len(idx), 0, :, :n_aer, :])
    rep("aer scattering %s" % iso, sca, z[iso + "_29"][:len(idx), 0, :, :n_aer, :])
    rep("aer asymmetry %s" % iso, asy, z[iso + "_30"][:len(idx), 0, :, :n_aer, :])
    if iso == "lw":
        rho, tg, til = P.surface_lw(ci, S["lw"], 9)
        rep("lw rho_alb diffuse", rho, z["lw_17"][idx, 0, 0, :])
        rep("lw t_ground", tg, z["lw_18"][idx, 0])
        fl = z["lw_32"][idx, 0] > 0.5
        print("   tile flag agree: %.3f %%" % (100 * (np.asarray(til["flag"]) == fl).mean()))
        sel = np.asarray(til["flag"]) & fl
        rep("lw tile frac", np.asarray(til["frac"])[sel], z["lw_34"][idx, 0][sel])
        rep("lw tile t", np.asarray(til["t"])[sel], z["lw_35"][idx, 0][sel])
        rep("lw tile rho diffuse", np.asarray(til["rho"])[sel], z["lw_36"][idx, 0, 0][sel])
    else:
        rho, til = P.surface_sw(ci, S["alb_wgt_sw"], 6)
        rep("sw rho_alb (diff, dir)", rho, z["sw_17"][:len(idx), 0])
        rep("sw zen_0", 1.0 / ci["cos_zenith"], z["sw_15"][:len(idx), 0])
        rep("sw solar_irrad", ci["solar_irrad"], z["sw_16"][:len(idx), 0])
