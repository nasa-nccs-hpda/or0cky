"""R5: whole-band chain (band_chain.run_band) against the per-k-term flux_total_part of the dumps (tag 21) and, for solar, flux_direct (tag 48).
Not ported and taken from the dump: grey optical properties of the band (tag 10), phase functions / forward scatter (tags 40, 49), Planck terms
and albedos (tags 25, 46, 47).   usage: python r2_band_check.py DUMP SPECDIR [step]   (step: use every step-th band record)"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R, spec_data as SD
import band_chain as BC


def bands(recs):
    """yield (atm, iso, i_band, band tags, [k-term dicts]) in file order."""
    atm = None; band = None; kt = None
    for x in recs:
        t, k = x["tag"], x["k"] if "k" in x else None
        if t == 0:
            if k == 1:
                if band is not None and band["terms"]:      # the last band of the previous call belongs to the previous atmosphere
                    yield atm, band
                band = None; kt = None
                atm = {"iso": x["iso"], "ncall": x["ncall"], "raw": {}}
            atm["raw"][k] = x["a"]
        elif t == 10 and k == 1:
            if band is not None and band["terms"]:
                yield atm, band
            band = {"iso": x["iso"], "band": x["band"], "t": {}, "terms": []}
        if band is None:
            continue
        if t in (10, 25, 20, 26, 30):
            band["t"][(t, k)] = x["a"]
        if t == 40 and k == 1:
            kt = {}
        if kt is not None and t in (40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 21):
            kt[(t, 1 if t == 21 else k)] = x["a"]
        if t == 22 and kt is not None:
            kt[(22, 1)] = x["a"]
            band["terms"].append(kt); kt = None
    if band is not None and band["terms"]:
        yield atm, band


if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    spec = {1: SD.load("%s/spec_sw.npz" % sys.argv[2]), 2: SD.load("%s/spec_lw.npz" % sys.argv[2])}
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0, nb=0))
    cum = {}
    calls = {}
    for bi, (atm, band) in enumerate(bands(recs)):
        if (25, 1) not in band["t"]:
            continue
        ck = (band["iso"], atm["ncall"])
        if ck not in calls:
            calls[ck] = len(calls)
        if calls[ck] % step:                              # step applies to dumped CALLS (all bands of a call are needed for the LW sum)
            continue
        iso, ib = band["iso"], band["band"]
        r = atm["raw"]
        A = dict(p=r[1], t=r[2], mix=r[4], mass=r[5], w_cloud=r[10], frac_cloud=r[11], zen_0=r[15])
        T = band["t"]
        B = dict(k_grey_tot_clr=T[(10, 1)], k_ext_scat_clr=T[(10, 2)], k_grey_tot=T[(10, 3)], k_ext_scat=T[(10, 4)],
                 planck_flux_band=T[(25, 5)], diff_planck_band=T[(25, 6)], diff_planck_band_2=T[(25, 7)], planck_flux_surface=T[(25, 8)],
                 solar_irrad=T[(25, 9)])
        k0 = band["terms"][0]
        alb = dict(diffuse=k0[(46, 3)], direct=k0[(47, 7)])
        phase = [dict(ph_clr=k[(40, 7)], ph=k[(40, 8)], fwd_clr=k[(49, 1)], fwd=k[(49, 2)]) for k in band["terms"]]
        out = BC.run_band(spec[iso], iso, ib, A, B, phase, alb, l_scale_solar=bool(k0[(47, 9)].ravel()[0] > 0.5))
        tag = "SW" if iso == 1 else "LW"
        st[(tag, "flux_total")]["nb"] += 1
        # band flux = augment_radiance: first k-term sets, the rest add, weight_incr = weight_band * esft_weight (tag 30: down, up)
        wb = float(T[(25, 11)].ravel()[0])
        n_lay = out[0]["flux_total"].shape[1] // 2 - 1
        fd = fu = None
        for o in out:
            wi = wb * o["weight"]
            ft = o["flux_total"]
            up, dn = wi * ft[:, 0::2][:, :n_lay + 1], wi * ft[:, 1::2][:, :n_lay + 1]
            fu = up if fu is None else fu + up
            fd = dn if fd is None else fd + dn
        if iso == 2:                                           # adjust_ir_radiance: + planck_flux_band at every level
            fu = fu + T[(25, 5)]; fd = fd + T[(25, 5)]
            # all LW bands are accumulated in one channel (tag 30 is cumulative over the bands of the call, in band order)
            key = atm["ncall"]
            if ib == 1 or key not in cum:
                cum[key] = (fd, fu)
            else:
                cum[key] = (cum[key][0] + fd, cum[key][1] + fu)
            fd, fu = cum[key]
        if (30, 1) in T:
            for name, m, ref in (("band_flux_down", fd, T[(30, 1)]), ("band_flux_up", fu, T[(30, 2)])):
                s = st[(tag, name)]
                d = np.abs(m - ref)
                s["n"] += d.size; s["bit"] += int((m == ref).sum()); s["nb"] += 1
                s["maxabs"] = max(s["maxabs"], float(d.max()))
                s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(ref), 1e-300)).max()))
        for kt, o in zip(band["terms"], out):
            for name, m, ref in (("flux_total", o["flux_total"], kt[(21, 1)]),) + ((("flux_direct", o["flux_direct"], kt[(48, 1)]),) if iso == 1 else ()):
                s = st[(tag, name)]
                d = np.abs(m - ref)
                s["n"] += d.size; s["bit"] += int((m == ref).sum())
                s["maxabs"] = max(s["maxabs"], float(d.max()))
                s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(ref), 1e-300)).max()))
    for k in sorted(st):
        s = st[k]
        print("%-3s %-12s bands %3d  %8d entries  bitwise %7.3f %%  max abs %10.3e  max rel %10.3e" %
              (k[0], k[1], s["nb"], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
