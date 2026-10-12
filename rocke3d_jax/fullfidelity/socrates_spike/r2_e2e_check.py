"""R5/R6 milestone: radiance_calc from the model-side inputs (tag 0 of the call) and spectral data to the band fluxes, compared with tag 30
(band-end fluxes; LW cumulative over bands in one channel).  Nothing of the optical properties, phase functions, Planck terms or overlap
comes from the dump: only atmosphere, aerosol-mode and cloud-condensate fields, albedos and the cloud top of the call.
usage: python r2_e2e_check.py DUMP SPECDIR2 [step] [calls...]"""
import sys, collections
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R, spec_data as SD
import radiance_calc_jax as RC


def calls(recs):
    cur = None
    for x in recs:
        t = x["tag"]
        if t == 0 and x["k"] == 1:
            if cur is not None:
                yield cur
            cur = dict(iso=x["iso"], ncall=x["ncall"], r={}, ref={})
        if cur is None:
            continue
        if t == 0:
            cur["r"][x["k"]] = x["a"]
        elif t == 30:
            cur["ref"].setdefault(x["band"], {})[x["k"]] = x["a"]
        elif t == 900:
            pass
    if cur is not None:
        yield cur


if __name__ == "__main__":
    recs = R.read(sys.argv[1])
    spec = {1: SD.load("%s/spec_sw.npz" % sys.argv[2]), 2: SD.load("%s/spec_lw.npz" % sys.argv[2])}
    step = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 1
    st = collections.defaultdict(lambda: dict(n=0, bit=0, maxabs=0.0, maxrel=0.0, nc=0))
    for ci, c in enumerate(calls(recs)):
        if ci % step or not c["ref"]:
            continue
        iso = c["iso"]; tag = "SW" if iso == 1 else "LW"
        res = RC.radiance_calc(spec[iso], iso, c["r"], nct=(1 if "--nct1" in sys.argv else None))
        cum_d = cum_u = None
        for rb in res:
            ib = rb["band"]
            if ib not in c["ref"]:
                continue
            fd, fu = rb["flux_down"], rb["flux_up"]
            if iso == 2:                                    # one LW channel: cumulative; the Planck add-back is already in each band
                cum_d = fd if cum_d is None else cum_d + fd
                cum_u = fu if cum_u is None else cum_u + fu
                fd, fu = cum_d, cum_u
            for name, m, ref in (("flux_down", fd, c["ref"][ib][1]), ("flux_up", fu, c["ref"][ib][2])):
                s = st[(tag, name)]
                d = np.abs(m - ref)
                s["n"] += d.size; s["bit"] += int((m == ref).sum()); s["nc"] += 1
                s["maxabs"] = max(s["maxabs"], float(d.max())); s["maxrel"] = max(s["maxrel"], float((d / np.maximum(np.abs(ref), 1e-300)).max()))
    for k in sorted(st):
        s = st[k]
        print("%-3s %-10s band-records %4d %8d entries  bitwise %7.3f %%  max abs %10.3e  max rel %10.3e" %
              (k[0], k[1], s["nc"], s["n"], 100.0 * s["bit"] / max(s["n"], 1), s["maxabs"], s["maxrel"]))
