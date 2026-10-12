"""R4: grey optical properties of every dumped band rebuilt from atmosphere / aerosol-mode / cloud inputs (tag 0) and the spectral data
(spec_data npz, SPECDIR2) and compared with tag 10 (k_grey_tot*, k_ext_scat*), tag 49 (forward_scatter) and tag 40/7-8 (phase function
after rescale_phase_fnc).   usage: python r2_grey_check.py DUMP SPECDIR2 [step]"""
import sys, collections
import numpy as np
import r2_parse as R, spec_data as SD
import grey_opt_prop as G
from r2_band_check import bands


def pack(st, key, m, o):
    s = st[key]; ok = np.isfinite(o)
    d = np.abs(m - o)[ok]
    s["n"] += d.size; s["bit"] += int((m[ok] == o[ok]).sum())
    if d.size:
        s["maxabs"] = max(s["maxabs"], float(d.max())); s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(o[ok]), 1e-300)).max()))


if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    spec = {1: SD.load("%s/spec_sw.npz" % sys.argv[2]), 2: SD.load("%s/spec_lw.npz" % sys.argv[2])}
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0))
    nb = 0
    for bi, (atm, band) in enumerate(bands(recs)):
        if bi % step or (10, 1) not in band["t"]:
            continue
        r, T, iso, ib = atm["raw"], band["t"], band["iso"], band["band"]
        sp = spec[iso]
        k0 = band["terms"][0]
        nct = int(k0[(44, 1)].ravel()[0]); tag = "SW" if iso == 1 else "LW"
        n_mode = int(r[31][2])
        # continuum of this band (only the self continuum, index 1, is present in the rundeck's files)
        kc, am = [], []
        for j in range(int(sp["n_band_continuum"][ib - 1])):
            ic = int(sp["index_continuum"][ib - 1, j])
            assert ic == 1 and int(sp["i_scale_fnc_cont"][ib - 1, ic - 1]) == 2
            kc.append(float(sp["k_cont"][ib - 1, ic - 1]))
            am.append(G.continuum_amount_self(r[1], r[2], r[3], r[4][:, :, 0], sp["scale_cont"][:3, ib - 1, ic - 1],
                                              float(sp["p_ref_cont"][ic - 1, ib - 1]), float(sp["t_ref_cont"][ic - 1, ib - 1])))
        comps = []
        types = r[24].astype(int)                            # type_condensed: 1 st water, 2 st ice, 3 cnv water, 4 cnv ice
        ipar = r[21].astype(int)
        for kk, (ty, ip) in enumerate(zip(types, ipar)):
            kind = "water" if ty in (1, 3) else "ice"
            assert ip == (5 if kind == "water" else 12), ip
            npar = 16 if kind == "water" else 14
            comps.append((kk + 1, kind, r[12][:, :, kk], r[13][:, :, kk], r[23][:npar, kk, ib - 1], r[14][:, :, kk]))
        rc = float(sp["rayleigh_coeff"][ib - 1]) if iso == 1 else 0.0
        out = G.grey_props(iso, rc, r[27], r[28][..., ib - 1], r[29][..., ib - 1], r[30][..., ib - 1], n_mode, kc, am, comps, r[10])
        pack(st, (tag, "k_grey_tot_clr"), out["kg"][:, :nct - 1, 0], T[(10, 1)][:, :nct - 1])
        pack(st, (tag, "k_ext_scat_clr"), out["kes"][:, :nct - 1, 0], T[(10, 2)][:, :nct - 1])
        pack(st, (tag, "k_grey_tot"), out["kg"][:, nct - 1:, :], T[(10, 3)][:, nct - 1:, :])
        pack(st, (tag, "k_ext_scat"), out["kes"][:, nct - 1:, :], T[(10, 4)][:, nct - 1:, :])
        pack(st, (tag, "forward_scatter_clr"), out["fwd"][:, :nct - 1, 0], k0[(49, 1)][:, :nct - 1])
        pack(st, (tag, "forward_scatter"), out["fwd"][:, nct - 1:, :], k0[(49, 2)][:, nct - 1:, :])
        # rescale_phase_fnc: (phase - f) / (1 - f)  -> tag 40/7 (clear), 40/8 (cloudy, last axis type)
        phr = (out["ph"] - out["fwd"]) / (1.0 - out["fwd"])
        pack(st, (tag, "phase_fnc_clr(rescaled)"), phr[:, :nct - 1, 0], k0[(40, 7)][:, :nct - 1, 0])
        pack(st, (tag, "phase_fnc(rescaled)"), phr[:, nct - 1:, :], k0[(40, 8)][:, nct - 1:, 0, :])
        nb += 1
    print("bands compared: %d" % nb)
    for k in sorted(st):
        s = st[k]
        print("%-3s %-26s %9d entries  bitwise %8.3f %%  max abs %10.3e  max rel %10.3e" % (k[0], k[1], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
