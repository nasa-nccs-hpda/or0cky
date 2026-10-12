"""R4/R6: n_cloud_top, frac_region, w_free, cloud_overlap rebuilt from w_cloud and frac_cloud (tag 0/10, 0/11) and compared with the
dumps: tag 44 (n_cloud_top), 45/2 (frac_region), 47/3 (w_free), 46/1 (cloud_overlap).   usage: python r2_cloud_check.py DUMP"""
import sys, collections
import numpy as np
import r2_parse as R
import cloud_geometry as CG

if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0))
    atm, seen = {}, set()
    def cmp(key, m, o):
        s = st[key]; ok = np.isfinite(o)
        d = np.abs(m - o)[ok]
        s["n"] += d.size; s["bit"] += int((m[ok] == o[ok]).sum())
        if d.size: s["maxabs"] = max(s["maxabs"], float(d.max()))
    for x in recs:
        t = x["tag"]
        if t == 0:
            if x["k"] == 1:
                atm = {"iso": x["iso"], "ncall": x["ncall"]}
            atm[x["k"]] = x["a"]
        elif t == 44 and x["k"] == 1 and 10 in atm and 11 in atm:
            key = (x["iso"], x["ncall"])
            if key in seen:
                continue
            seen.add(key)
            tag = "SW" if x["iso"] == 1 else "LW"
            nct_f = int(x["a"][0]); n_layer = int(x["a"][3])
            wc, fc = atm[10], atm[11]
            nct = CG.set_cloud_top(wc)
            st[(tag, "n_cloud_top")]["n"] += 1; st[(tag, "n_cloud_top")]["bit"] += int(nct == nct_f)
            fr = CG.aggregate_csiw(fc)
            co, wfree, tcc = CG.overlap_coupled_triple(wc, fr, nct_f)
            atm["_mine"] = (co, wfree, fr, nct_f)
        elif t == 45 and x["k"] == 2 and "_mine" in atm:
            co, wfree, fr, nct = atm["_mine"]
            tag = "SW" if x["iso"] == 1 else "LW"
            cmp((tag, "frac_region(strat,conv)"), fr[:, nct - 1:, 1:], x["a"][:, nct - 1:, 1:])
        elif t == 46 and x["k"] == 1 and "_mine" in atm:
            co, wfree, fr, nct = atm["_mine"]
            tag = "SW" if x["iso"] == 1 else "LW"
            cmp((tag, "cloud_overlap"), co[:, nct - 1:, :], x["a"][:, nct - 1:, :])
        elif t == 47 and x["k"] == 3 and "_mine" in atm:
            co, wfree, fr, nct = atm["_mine"]
            tag = "SW" if x["iso"] == 1 else "LW"
            cmp((tag, "w_free"), wfree[:, nct - 1:], x["a"][:, nct - 1:])
            atm.pop("_mine")
    for k in sorted(st):
        s = st[k]
        print("%-3s %-26s %9d entries  bitwise %8.3f %%  max abs %10.3e" % (k[0], k[1], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"]))
