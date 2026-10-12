"""Daily-products reader (daily_products.py) against the instrumented Fortran of other calendar days (r6_days_oracle.py + r6_collect.py nper).
usage: python r6_days_check.py RUNDIR PREFIX DAYS... [--out result.json]
  PREFIX_0.npz, PREFIX_1.npz, ... are the steps of one chunk; DAYS the day of year of each (330, 331, ..., as given to r6_days_oracle.py)."""
import json, os, sys
import numpy as np
import planet_side as P
import daily_products as D


def day_fields(z):
    N = z["p60"].shape[0]
    c60 = P.unpack(z["p60"], P.PACK60)
    c50 = P.unpack(z["p50"], P.PACK50)
    return dict(o3col=c60["u0gas"].reshape(N, 13, 43)[:, 2, :], ataulx=c60["ataulx"].reshape(N, 6, 43).transpose(0, 2, 1),
                dtaulx=z["p61"], dalbsn=c60["dalbsn"], vtaulx=c60["vtaulx"], vreff=c60["vreff"], q55h2s=c60["q55h2s"],
                loc_chl=c60["loc_chl"], frac_veg=c50["frac_veg"], fpxco2=c60["fpxco2"], fulgas=c60["fulgas"][0], ppmv80=c60["ppmv80"][0],
                solar_irrad=c50["solar_irrad"][0], cld_scaling=c50["cld_scaling"][0],
                ilon=c60["ilon"].astype(int), jlat=c60["jlat"].astype(int))


def main(rundir, prefix, days, out=None):
    z0 = np.load("%s_0.npz" % prefix)
    f0 = day_fields(z0)
    c60 = P.unpack(z0["p60"], P.PACK60)
    # other configurations of the rundeck (the oracle run made with R6_I_EDIT): env R6_MASTER_YR, R6_VOLC_YR
    kw = {}
    if os.environ.get("R6_MASTER_YR"):
        kw["master_yr"] = int(os.environ["R6_MASTER_YR"])
    if os.environ.get("R6_VOLC_YR"):
        kw["volc_yr"] = int(os.environ["R6_VOLC_YR"])
    dp = D.DailyProducts(rundir, c60["plb0"][0], c60["drym2g"][0], c60["ppmv80"][0], z0["p62"][0], hlb0=c60["hlb0"][0], **kw)
    res = {}
    for k, dstr in enumerate(days):
        d, _, ya = str(dstr).partition(":")                       # "jday" or "jday:years after 1950"
        d, year = int(d), 1950 + int(ya or 0)
        z = np.load("%s_%d.npz" % (prefix, k))
        f = day_fields(z)
        il, jl = f["ilon"] - 1, f["jlat"] - 1
        r = {}
        o3 = dp.ozone(d)                                         # (nl, lat, lon)
        r["o3col"] = o3[:, jl, il].T
        a = dp.aerosol(d)                                        # (nl, lat, lon, 6)
        r["ataulx"] = a[:, jl, il, :].transpose(1, 0, 2)
        du = dp.dust_cols(d)
        r["dtaulx"] = du[:, jl, il, :].transpose(1, 0, 2)
        r["dalbsn"] = dp.dalbsn(d)[jl, il] * 1.0
        vt, vr = dp.volcanic(d, year)
        r["vtaulx"], r["vreff"] = vt[jl], vr[jl]
        if dp.volc_tables is not None:
            r["q55h2s"] = dp.volc_tables.q55(vr[jl])
        eps = dp.epsilon(d)
        r["epl"], r["epm"], r["eph"] = eps[0][il, jl], eps[1][il, jl], eps[2][il, jl]
        fu, pp = dp.fulgas_ppmv(d, f0["ppmv80"], f0["fulgas"])
        r["fulgas"], r["ppmv80"] = fu, pp
        r["rsdist"] = np.array(D.rsdist(d, float(dp.year)))
        r["albvnh"] = D.albvnh(d, np.load("data/albvnd.npz")["albvnd"]).reshape(-1, order="F")
        ref = dict(f)
        ref["epl"], ref["epm"], ref["eph"] = z["static_181"][il, jl], z["static_182"][il, jl], z["static_183"][il, jl]
        ref["rsdist"] = np.array(z["static_173"][0]) if "static_173" in z else None
        ref["albvnh"] = np.asarray(z["static_180"]).reshape(-1, order="F") if "static_180" in z else None
        row = {}
        for key, v in r.items():
            w = ref.get(key)
            if w is None:
                row[key] = "no reference"
                continue
            v, w = np.asarray(v, dtype=float).reshape(-1), np.asarray(w, dtype=float).reshape(-1)
            row[key] = dict(maxabs=float(np.abs(v - w).max()), maxrel=float(np.abs(v - w).max() / max(np.abs(w).max(), 1e-300)),
                            bitwise=bool(np.array_equal(v, w)))
        # fields the reader does not produce yet: do they change with the day?
        for key in (("q55h2s",) if dp.volc_tables is None else ()) + ("loc_chl", "frac_veg", "fpxco2", "solar_irrad", "cld_scaling"):
            row["const_" + key] = bool(np.array_equal(np.asarray(f[key]), np.asarray(f0[key])))
        res[str(dstr)] = row
        print(d, {k: (v if isinstance(v, str) else ("=" if v["bitwise"] else "%.1e" % v["maxabs"])) for k, v in row.items()
                  if not k.startswith("const_")}, {k[6:]: v for k, v in row.items() if k.startswith("const_")}, flush=True)
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    o = None
    if "--out" in a:
        i = a.index("--out")
        o = a[i + 1]
        a = a[:i] + a[i + 2:]
    main(a[0], a[1], a[2:], o)
