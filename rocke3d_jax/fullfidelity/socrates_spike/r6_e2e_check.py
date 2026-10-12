"""R6 end-to-end check of one radiation step: the arguments of run_planet_rad (dump tag 50) -> planet_side (set-up) -> radiance_batch (all
bands) -> planet_side post-processing  against the outputs of get_planet_radout (dump tag 51) of the real Fortran.
   JAX_PLATFORMS=cpu python r6_e2e_check.py cols.npz static0.npz specdir [n_columns]"""
import sys, os, time
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import spec_data as SD
import radiance_batch as RB
import planet_side as P

z = np.load(sys.argv[1]); s0 = np.load(sys.argv[2]); specdir = sys.argv[3]
nmax = int(sys.argv[4]) if len(sys.argv) > 4 else None
spec = {1: SD.load(os.path.join(specdir, "spec_sw.npz")), 2: SD.load(os.path.join(specdir, "spec_lw.npz"))}
S = P.static_from_npz(z)
N = z["p50"].shape[0] if nmax is None else min(nmax, z["p50"].shape[0])
c = {k: jnp.asarray(v[:N]) for k, v in P.unpack(z["p50"], P.PACK50).items()}
ref = P.unpack(z["p51"][:N], P.PACK51)
sw_idx = z["sw_idx"]; sw_idx = sw_idx[sw_idx < N]
lit = np.zeros(N, bool); lit[sw_idx] = True


def tag0(iso, ci):
    """tag-0 style dict of one isolir for the columns ci (stage B outputs + the call-independent records)."""
    key = "lw" if iso == 2 else "sw"
    nb = 9 if iso == 2 else 6
    sub = {k: v[ci] for k, v in c.items()}
    p_layer, t_level, mass, dens = P.layers(sub["p_level"], sub["t_layer"])
    ul = sub["ulgas"].reshape(-1, 13, 43)
    mix = P.gas_mix_ratio(ul, mass, spec_np[key]["type_absorb"], S[key])
    w, frac, cmix, dim, rel = P.cloud_field(sub, S[key])
    n_aer = int(round(float(sub["n_aer"][0])))
    p52, p53 = z["p52"][:N][ci], z["p53"][:N][ci]
    comp = p52[0, :, 0].astype(int)
    scl = p52[:, :, 3 if iso == 1 else 4]
    mm, ab, sc, asy = P.aerosol_modes(S[key]["aer"], comp, jnp.asarray(p52[:, :, 1]), jnp.asarray(p53), jnp.asarray(p52[:, :, 2]),
                                      jnp.asarray(scl), sub["rel_humidity"], mass)
    r = {1: p_layer, 2: sub["t_layer"], 3: dens, 4: mix, 5: mass, 7: t_level, 10: w, 11: frac, 12: cmix, 13: dim, 14: rel,
         27: mm, 28: ab, 29: sc, 30: asy}
    for k in (20, 21, 22, 23, 24, 25, 26, 31):
        r[k] = s0["%s0_%d" % (key, k)]
    r[31] = np.asarray(r[31]).copy(); r[31][2] = n_aer
    if iso == 2:
        rho_d, tg, til = P.surface_lw(sub, S["lw"], nb)
        rho = jnp.stack([rho_d, jnp.zeros_like(rho_d)], axis=1)
        r[15] = jnp.ones(len(ci)); r[16] = jnp.zeros(len(ci))
        tile_rho = None
    else:
        rho, til_rho = P.surface_sw(sub, S["alb_wgt_sw"], nb)
        part = ((sub["frac_sea"] > 0) & (sub["frac_sea"] < 1)) | ((sub["frac_land"] > 0) & (sub["frac_land"] < 1)) | \
               ((sub["frac_sea_ice"] > 0) & (sub["frac_sea_ice"] < 1)) | ((sub["frac_land_ice"] > 0) & (sub["frac_land_ice"] < 1))
        tg = sum_t(sub)
        til = dict(flag=part, rho=til_rho)
        r[15] = 1.0 / sub["cos_zenith"]; r[16] = sub["solar_irrad"]
        tile_rho = til_rho
    r[17] = rho; r[18] = tg
    # tiles for the IR surface emission (frac, t) as in columns_from_calls
    fr = jnp.stack([jnp.where(sub["frac_sea"] > 0, sub["frac_sea"], 0.0), jnp.where(sub["frac_sea_ice"] > 0, sub["frac_sea_ice"], 0.0),
                    jnp.where(sub["frac_land"] > 0, sub["frac_land"], 0.0), jnp.where(sub["frac_land_ice"] > 0, sub["frac_land_ice"], 0.0)], axis=1)
    tt = jnp.stack([jnp.where(sub["frac_sea"] > 0, sub["t_sea"], 0.0), jnp.where(sub["frac_sea_ice"] > 0, sub["t_sea_ice"], 0.0),
                    jnp.where(sub["frac_land"] > 0, sub["t_land"], 0.0), jnp.where(sub["frac_land_ice"] > 0, sub["t_land_ice"], 0.0)], axis=1)
    flag = np.asarray(((sub["frac_sea"] > 0) & (sub["frac_sea"] < 1)) | ((sub["frac_land"] > 0) & (sub["frac_land"] < 1)) |
                      ((sub["frac_sea_ice"] > 0) & (sub["frac_sea_ice"] < 1)) | ((sub["frac_land_ice"] > 0) & (sub["frac_land_ice"] < 1)))
    r["tile_flag"] = flag; r["tile_n"] = np.where(flag, 4, 0); r["tile_frac"] = fr; r["tile_t"] = tt
    return r, til, flag


def sum_t(sub):
    tg = jnp.zeros_like(sub["t_sea"])
    for f, t in (("frac_sea", "t_sea"), ("frac_land", "t_land"), ("frac_sea_ice", "t_sea_ice"), ("frac_land_ice", "t_land_ice")):
        tg = jnp.where(sub[f] > 0, tg + sub[f] * sub[t], tg)
    return tg


spec_np = {"lw": np.load(os.path.join(specdir, "spec_lw.npz")), "sw": np.load(os.path.join(specdir, "spec_sw.npz"))}

t0 = time.time()
r_lw, til_lw, flag_lw = tag0(2, np.arange(N))
res_lw = RB.radiance(spec[2], 2, r_lw)
jax.block_until_ready(res_lw[-1][0])
print("LW batch %d columns, %.1f s" % (N, time.time() - t0), flush=True)
t0 = time.time()
r_sw, til_sw, flag_sw = tag0(1, sw_idx)
res_sw = RB.radiance(spec[1], 1, r_sw)
jax.block_until_ready(res_sw[-1][0])
print("SW batch %d columns, %.1f s" % (len(sw_idx), time.time() - t0), flush=True)

# ---- post-processing
tcc = P.total_cloud_cover(r_lw[10], r_lw[11])
lw = P.post_lw([x[0] for x in res_lw], [x[1] for x in res_lw], r_lw[7])
nb = 6
def expand(a, fill=0.0):                                  # lit-column array -> all columns
    out = jnp.full((N,) + a.shape[1:], fill, dtype=a.dtype)
    return out.at[jnp.asarray(sw_idx)].set(a)
dn_b = [expand(x[0]) for x in res_sw]; up_b = [expand(x[1]) for x in res_sw]; dr_b = [expand(x[2]) for x in res_sw]
cosz = c["cos_zenith"]
wb = S["sw"]["weight_blue"]
# tile fluxes of the lit columns (augment_tiled_radiance: w_band * (rho_d * F_down + (rho_x - rho_d) * F_direct) at the surface)
til_rho = til_sw["rho"]                                   # (Nlit, 2, 4, nb)
tile_up_l = jnp.stack([jnp.stack([1.0 * (til_rho[:, 0, k, b] * res_sw[b][0][:, -1] + (til_rho[:, 1, k, b] - til_rho[:, 0, k, b]) * res_sw[b][2][:, -1])
                                  for b in range(nb)], axis=1) for k in range(4)], axis=1)      # (Nlit, 4, nb)
tile_up = expand(tile_up_l) / jnp.where(jnp.asarray(lit), cosz, 1.0)[:, None, None]
sw = P.post_sw(dn_b, up_b, dr_b, jnp.asarray(lit), cosz, wb, None, None, c["solar_irrad"])
tile_flag = jnp.asarray(lit) & jnp.asarray(np.asarray(((c["frac_sea"] > 0) & (c["frac_sea"] < 1)) | ((c["frac_land"] > 0) & (c["frac_land"] < 1)) |
                                                     ((c["frac_sea_ice"] > 0) & (c["frac_sea_ice"] < 1)) | ((c["frac_land_ice"] > 0) & (c["frac_land_ice"] < 1))))
fs = P.fsrnfg(sw, (c["frac_sea"], c["frac_land"], c["frac_sea_ice"], c["frac_land_ice"]), tile_flag, tile_up)

mine = dict(tlb=lw["tlb"], tlt=lw["tlt"], truflb=lw["truflb"], trdflb=lw["trdflb"], trnflb=lw["trnflb"], trfcrl=lw["trfcrl"],
            truptoa=lw["truptoa"], tot_cloud_cover=tcc, fsrnfg=fs)
mine.update({k: v for k, v in sw.items()})
print("%-16s %8s %10s %12s %12s" % ("field", "entries", "bitwise %", "max abs", "max rel"))
rows = []
for k in [n for n, _ in P.PACK51]:
    m = np.asarray(mine[k]); rf = np.asarray(ref[k])
    d = np.abs(m - rf)
    sel = np.ones(rf.shape[0], bool) if k not in ("sruflb", "srdflb", "srnflb", "srfhrl") else np.ones(rf.shape[0], bool)
    bit = 100.0 * float((m == rf).mean())
    rel = float((d / np.maximum(np.abs(rf), 1e-300)).max())
    rows.append((k, m.size, bit, float(d.max()), rel))
    print("%-16s %8d %10.3f %12.3e %12.3e" % rows[-1])
