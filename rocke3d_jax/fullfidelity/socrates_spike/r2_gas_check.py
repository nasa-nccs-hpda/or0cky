"""R3: k_esft_layer (tag 26/1) and k_abs_layer (tag 25/1) of every dumped band rebuilt from p, t, gas_mix_ratio (tag 0) and the spectral
data (spec_data npz) with gas_absorption.py.   usage: python r2_gas_check.py DUMP SPECDIR"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import r2_parse as R, spec_data as SD
import gas_absorption as GA

if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    spec = {1: SD.load("%s/spec_sw.npz" % sys.argv[2]), 2: SD.load("%s/spec_lw.npz" % sys.argv[2])}
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0))
    atm = {}
    cur = {}
    for x in recs:
        t = x["tag"]
        if t == 0:
            if x["k"] == 1:
                atm = {}
            atm[x["k"]] = x["a"]
        elif t == 26 and x["k"] == 1:
            cur = {"band": x["band"], "kesft": x["a"], "iso": x["iso"]}
        elif t == 25 and x["k"] == 1 and cur:
            cur["kabs"] = x["a"]
        elif t == 25 and x["k"] == 4 and cur:
            n_abs = int(x["a"][0]); idx = x["a"].astype(int)[1:]
            sp = spec[cur["iso"]]
            p, tt, mix = jnp.asarray(atm[1]), jnp.asarray(atm[2]), jnp.asarray(atm[4])
            # major gas of this call (last profile), see gas_absorption.major_gas_order; the dumped k_abs_layer is in that order
            order = GA.major_gas_order(sp, cur["band"], np.asarray(atm[4])[-1], np.asarray(atm[5])[-1])
            kabs0, nes0, wes0, gidx0, kesft = GA.band_absorption(sp, cur["band"], p, tt, mix)
            pos = {int(g): j for j, g in enumerate(gidx0)}
            sel = [pos[g] for g in order]
            kabs, nes, gidx = kabs0[:, :, :, sel], nes0[sel], np.array(order)
            tag = "SW" if cur["iso"] == 1 else "LW"
            # tag 26/1: k_esft_layer for lookup gases (compare only entries set by the lookup)
            for j, g in enumerate(gidx):
                if int(sp["i_scale_fnc"][cur["band"] - 1, g - 1]) == GA.IP_SCALE_LOOKUP:
                    nk = int(nes[j])
                    a, b = np.asarray(kesft)[:, :, :nk, g - 1], cur["kesft"][:, :, :nk, g - 1]
                    s = st[(tag, "k_esft_layer")]
                    d = np.abs(a - b)
                    s["n"] += d.size; s["bit"] += int((a == b).sum())
                    s["maxabs"] = max(s["maxabs"], float(d.max())); s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(b), 1e-300)).max()))
            # tag 25/1 k_abs_layer is indexed by absorber j (index_abs may swap the first and a continuum entry; none here)
            for j in range(len(gidx)):
                nk = int(nes[j])
                a, b = np.asarray(kabs)[:, :, :nk, j], cur["kabs"][:, :, :nk, j]
                s = st[(tag, "k_abs_layer")]
                d = np.abs(a - b)
                s["n"] += d.size; s["bit"] += int((a == b).sum())
                s["maxabs"] = max(s["maxabs"], float(d.max())); s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(b), 1e-300)).max()))
            cur = {}
    for k in sorted(st):
        s = st[k]
        print("%-3s %-14s %9d entries  bitwise %8.3f %%  max abs %10.3e  max rel %10.3e" %
              (k[0], k[1], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
