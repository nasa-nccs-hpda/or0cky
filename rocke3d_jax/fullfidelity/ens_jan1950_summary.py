"""D165: summary of the real JAN1950 one-month ensemble (members run by scoping/D165 recipe).

usage: python ens_jan1950_summary.py repro  <run_dir_ctrl>          # control vs stored JAN1950 acc (all arrays)
       python ens_jan1950_summary.py summary <ens_dir> <out_prefix>  # per-member monthly means, spread table

Members are directories <ens_dir>/<name>/ holding the model's end-of-run fort.2.nc (acc block, float64) and, if written,
JAN1950.acc*.nc (float32).  Monthly mean of a column = f3_diagnostics.field_from_aij (DIAG_PRT.f ij_mapk).
"""
import json
import os
import sys

import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f3_diagnostics as f3

STORED = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc"
KEY = ["prsurf", "slp", "t_850", "t_500", "t_200", "z_500", "u_200", "q_850", "omega_500", "qatm", "prec", "evap", "tsurf",
       "srnf_toa", "trnf_toa", "incsw_toa", "srnf_grnd", "trdn_surf", "tauus", "tauvs", "rh_layer1"]


def load_acc(path):
    d = nc.Dataset(path)
    out = dict(aij=np.asarray(d["aij"][:], dtype=np.float64), idacc=np.asarray(d["idacc"][:]),
               aijl=np.asarray(d["aijl"][:], dtype=np.float64), axyp=np.asarray(d["axyp"][:]) if "axyp" in d.variables else None,
               raw={k: np.asarray(d[k][:]) for k in ("aij", "aijl", "aijk", "aj", "ajl", "consrv", "agc", "areg", "tdiurn")
                    if k in d.variables})
    d.close()
    return out


def fields(acc, meta, names):
    res = {}
    for n in names:
        c = next(k for k, v in meta.items() if v["name"].strip() == n)
        m = dict(meta[c])
        m["denom_ia"] = meta[m["denom"]]["ia"] if m["denom"] > 0 else 0
        a, b = f3.field_from_aij(acc["aij"][c - 1], acc["idacc"], m, acc["aij"])
        res[n] = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)   # nan where the denominator is zero (e.g. pressure levels below ground)
    return res


def repro(run, ref=STORED):
    cand = [f for f in os.listdir(run) if f.upper().startswith("JAN1950.ACC")]
    r = {}
    for label, path in [("fort.2.nc", f"{run}/fort.2.nc")] + [(f, f"{run}/{f}") for f in cand]:
        d = nc.Dataset(path)
        s = nc.Dataset(ref)
        rep = {}
        for k in ("idacc", "aij", "aijl", "aijk", "aj", "ajl", "consrv", "agc", "areg", "tdiurn", "asjl", "aisccp", "adiurn",
                  "energy", "ijhc", "oij", "oijl", "icij"):
            if k not in d.variables or k not in s.variables:
                continue
            a = np.asarray(d[k][:]); b = np.asarray(s[k][:])
            if a.shape != b.shape:
                rep[k] = dict(shape_mismatch=[list(a.shape), list(b.shape)]); continue
            a64 = a.astype(np.float64); b64 = b.astype(np.float64)
            neq = int((a64 != b64).sum())
            a32 = a.astype(np.float32); b32 = b.astype(np.float32)
            neq32 = int((a32 != b32).sum())
            mx = float(np.abs(b64).max()) if b64.size else 0.0
            rep[k] = dict(n=int(a.size), n_unequal=neq, n_unequal_as_float32=neq32, maxabs=float(np.abs(a64 - b64).max()),
                          scale=mx, maxrel_of_scale=float(np.abs(a64 - b64).max() / mx) if mx > 0 else 0.0)
        d.close(); s.close()
        r[label] = rep
    return r


def summary(ens, outp):
    meta = f3.aij_names_from_nc(STORED)
    names = sorted(n for n in os.listdir(ens) if os.path.exists(f"{ens}/{n}/JAN1950.accP2SAoM40.nc"))
    allnames = [n for n in KEY]
    # all ported-name set too
    allnames = KEY + [n for n in f3.PORTED_NAMES if n not in KEY]
    mem = {}
    for n in names:
        acc = load_acc(f"{ens}/{n}/JAN1950.accP2SAoM40.nc")
        mem[n] = dict(idacc0=int(acc["idacc"][0]), f=fields(acc, meta, allnames))
    st = load_acc(STORED)
    stored = fields(st, meta, allnames)
    axyp = st["axyp"] if st["axyp"] is not None else np.ones((46, 72))
    w = axyp / axyp.sum()
    rows = []
    perturbed = [n for n in names if n != "ctrl"]
    def wmean(x, ww):                        # nan-aware area-weighted mean over the last two axes
        ok = np.isfinite(x); return np.where(ok, x, 0.0).__mul__(ww).sum(axis=(-2, -1)) / (ok * ww).sum(axis=(-2, -1))
    for fn in allnames:
        X = np.stack([mem[n]["f"][fn] for n in names])                # (M,46,72), nan = undefined
        ok = np.isfinite(X).all(0)
        if not ok.any():
            continue
        Xo = np.where(ok, X, np.nan)
        gm = wmean(Xo, w)
        mu = Xo.mean(0); sd = Xo.std(0, ddof=1)
        zs = np.nanmean(Xo, axis=2).std(0, ddof=1)               # zonal-mean spread (46 lat)
        sto = np.where(ok, stored[fn], np.nan)
        rows.append(dict(field=fn, n_defined_cells=int(ok.sum()), gm_mean=float(gm.mean()), gm_std=float(gm.std(ddof=1)),
                         gm_min=float(gm.min()), gm_max=float(gm.max()), gm_stored=float(wmean(sto, w)),
                         gm_ctrl_minus_stored=float(gm[names.index("ctrl")] - wmean(sto, w)) if "ctrl" in names else None,
                         field_std_space=float(np.sqrt(wmean((mu - wmean(mu, w)) ** 2, w))),
                         grid_sd_rms=float(np.sqrt(wmean(sd ** 2, w))), grid_sd_median=float(np.nanmedian(sd)), grid_sd_max=float(np.nanmax(sd)),
                         zonal_sd_rms=float(np.sqrt(np.nanmean(zs ** 2)))))
    np.savez(outp + ".npz", members=np.array(names), fields=np.array(allnames),
             monthly_mean=np.stack([np.stack([mem[n]["f"][fn] for fn in allnames]) for n in names]),
             stored=np.stack([stored[fn] for fn in allnames]), idacc0=np.array([mem[n]["idacc0"] for n in names]))
    json.dump(dict(members=names, rows=rows), open(outp + ".json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "repro":
        print(json.dumps(repro(sys.argv[2]), indent=1))
    else:
        for r in summary(sys.argv[2], sys.argv[3]):
            print(r["field"], *(f"{r[k]:.4g}" for k in ("gm_mean", "gm_std", "grid_sd_rms", "field_std_space")))
