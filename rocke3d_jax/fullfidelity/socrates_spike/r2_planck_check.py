"""R3b: Planck terms of every dumped LW band (tag 25/5-8) rebuilt from t_level, t, t_ground (tag 0/7, 0/2, 0/18) with planck.py.
usage: python r2_planck_check.py DUMP SPECDIR2"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R, spec_data as SD
import planck as PL

if __name__ == "__main__":
    recs = R.read(sys.argv[1]); spec = SD.load("%s/spec_lw.npz" % sys.argv[2])
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0))
    atm, band, cur = {}, None, {}
    for x in recs:
        t, k = x["tag"], x.get("k")
        if t == 0:
            if k == 1: atm = {"iso": x["iso"]}
            atm[k] = x["a"]
        elif t == 26 and k == 1:
            band = x["band"]
        elif t == 25 and x["iso"] == 2 and k in (5, 6, 7, 8):
            cur[k] = x["a"]
            if k == 8:
                pf, dp, dp2, pg = [np.asarray(v) for v in PL.band_planck(spec, band, atm[7], atm[2], atm[18])]
                if atm[32][0] > 0.5:                                  # control%l_tile
                    tile = dict(n_tile=int(atm[32][1]), list_tile=atm[33], frac_tile=atm[34], t_tile=atm[35])
                    pg = PL.surface_planck(spec, band, atm[18], tile)
                for name, m, o in (("planck_flux_band", pf, cur[5]), ("diff_planck_band", dp, cur[6]), ("diff_planck_band_2", dp2, cur[7]),
                                   ("planck_flux_surface", pg, cur[8])):
                    s = st[name]; d = np.abs(m - o)
                    s["n"] += d.size; s["bit"] += int((m == o).sum()); s["maxabs"] = max(s["maxabs"], float(d.max()))
                    s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(o), 1e-300)).max()))
                cur = {}
    for k in sorted(st):
        s = st[k]
        print("LW  %-20s %8d entries  bitwise %8.3f %%  max abs %10.3e  max rel %10.3e" % (k, s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
