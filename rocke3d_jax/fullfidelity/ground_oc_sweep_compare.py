"""Compare ground_oc_sweep_ff.ground_oc_sweep_layer against the real Fortran
ffz_gocsw_<itime>.bin dump (D35)."""
import sys
import numpy as np

from ground_oc_sweep_ff import ground_oc_sweep_layer


def read_gocsw(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 20)
    return dict(i=raw[:, 0], j=raw[:, 1], l=raw[:, 2], mo0=raw[:, 3], g0m0=raw[:, 4],
                s0m0=raw[:, 5], dxypj=raw[:, 6], pcorr=raw[:, 7], p0l=raw[:, 8], mo1=raw[:, 9],
                g0m1=raw[:, 10], s0m1=raw[:, 11])


def main(path):
    d = read_gocsw(path)
    n = len(d["i"])
    print(f"{n} real (i,j,l) records")
    worst = {"mo": 0.0, "g0m": 0.0, "s0m": 0.0}
    n_freeze = 0
    for idx in range(n):
        out = ground_oc_sweep_layer(d["mo0"][idx], d["g0m0"][idx], d["s0m0"][idx],
                                     d["dxypj"][idx], d["pcorr"][idx], d["p0l"][idx])
        if out["dm0"] != 0.0:
            n_freeze += 1
        for k, ref in (("mo", d["mo1"][idx]), ("g0m", d["g0m1"][idx]), ("s0m", d["s0m1"][idx])):
            rel = abs(out[k] - ref) / max(abs(ref), 1e-10)
            if rel > worst[k]:
                worst[k] = rel
    print(f"n_freeze={n_freeze}/{n}")
    ok = True
    for k, v in worst.items():
        status = "OK" if v < 1e-8 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{k:4s} max_relerr={v:.3e} {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffz_gocsw_33312.bin"
    sys.exit(main(path))
