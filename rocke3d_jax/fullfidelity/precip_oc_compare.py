"""Compare precip_oc_ff.precip_oc_cell against the real Fortran ffz_precoc_<itime>.bin dump (D33)."""
import sys
import numpy as np

from precip_oc_ff import precip_oc_cell


def read_precoc(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 20)
    return dict(i=raw[:, 0], j=raw[:, 1], focean=raw[:, 2], oprec=raw[:, 3], orsi=raw[:, 4],
                orunpsi=raw[:, 5], oeprec=raw[:, 6], oerunpsi=raw[:, 7], osrunpsi=raw[:, 8],
                dxypo=raw[:, 9], mo0=raw[:, 10], g0m0=raw[:, 11], s0m0=raw[:, 12], mo1=raw[:, 13],
                g0m1=raw[:, 14], s0m1=raw[:, 15])


def main(path):
    d = read_precoc(path)
    n = len(d["i"])
    print(f"{n} real cells")
    worst = {"mo": 0.0, "g0m": 0.0, "s0m": 0.0}
    for idx in range(n):
        out = precip_oc_cell(d["focean"][idx], d["oprec"][idx], d["orsi"][idx], d["orunpsi"][idx],
                              d["oeprec"][idx], d["oerunpsi"][idx], d["osrunpsi"][idx],
                              d["dxypo"][idx], d["mo0"][idx], d["g0m0"][idx], d["s0m0"][idx])
        for k, ref_key in (("mo", "mo1"), ("g0m", "g0m1"), ("s0m", "s0m1")):
            ref = d[ref_key][idx]
            rel = abs(out[k] - ref) / max(abs(ref), 1e-10)
            if rel > worst[k]:
                worst[k] = rel
    ok = True
    for k, v in worst.items():
        status = "OK" if v < 1e-9 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{k:4s} max_relerr={v:.3e} {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffz_precoc_33312.bin"
    sys.exit(main(path))
