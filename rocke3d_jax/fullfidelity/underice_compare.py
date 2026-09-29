"""Compare underice_ff.py against the real Fortran ffz_undocn_<itime>.bin / ffz_undlk_<itime>.bin
dumps (D32)."""
import sys
import numpy as np

from underice_ff import iceocean_fluxes, icelake_fluxes_limited


def read_undocn(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 20)
    return dict(i=raw[:, 0], j=raw[:, 1], tic=raw[:, 2], si=raw[:, 3], tm=raw[:, 4], sm=raw[:, 5],
                dh=raw[:, 6], ustar=raw[:, 7], coriol=raw[:, 8], mlsh=raw[:, 9], mflux=raw[:, 10],
                sflux=raw[:, 11], hflux=raw[:, 12])


def read_undlk(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 20)
    return dict(i=raw[:, 0], j=raw[:, 1], tic=raw[:, 2], tm=raw[:, 3], dh=raw[:, 4], mlsh=raw[:, 5],
                dlake=raw[:, 6], glake=raw[:, 7], mflux=raw[:, 8], hflux=raw[:, 9])


DTSRC = 1800.0  # decks/P2SAoM40.R (same as D29's DYNSI finding)


def check_ocn(path):
    d = read_undocn(path)
    n = len(d["i"])
    worst = {"mflux": 0.0, "sflux": 0.0, "hflux": 0.0}
    for idx in range(n):
        out = iceocean_fluxes(d["tic"][idx], d["si"][idx], d["tm"][idx], d["sm"][idx], d["dh"][idx],
                               d["ustar"][idx], d["coriol"][idx], DTSRC, d["mlsh"][idx])
        for k in worst:
            ref = d[k][idx]
            rel = abs(out[k] - ref) / max(abs(ref), 1e-10)
            if rel > worst[k]:
                worst[k] = rel
    print(f"  OCEAN: {n} cells")
    ok = True
    for k, v in worst.items():
        status = "OK" if v < 1e-6 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"    {k:6s} max_relerr={v:.3e} {status}")
    return ok


def check_lake(path):
    d = read_undlk(path)
    n = len(d["i"])
    worst = {"mflux": 0.0, "hflux": 0.0}
    for idx in range(n):
        out = icelake_fluxes_limited(d["tic"][idx], d["tm"][idx], d["dh"][idx], DTSRC, d["mlsh"][idx],
                                      d["dlake"][idx], d["glake"][idx])
        for k in worst:
            ref = d[k][idx]
            rel = abs(out[k] - ref) / max(abs(ref), 1e-10)
            if rel > worst[k]:
                worst[k] = rel
    print(f"  LAKES: {n} cells")
    ok = True
    for k, v in worst.items():
        status = "OK" if v < 1e-6 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"    {k:6s} max_relerr={v:.3e} {status}")
    return ok


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26"
    itime = sys.argv[2] if len(sys.argv) > 2 else "33312"
    print(f"{base}/{itime}")
    ok1 = check_ocn(f"{base}/ffz_undocn_{itime}.bin")
    ok2 = check_lake(f"{base}/ffz_undlk_{itime}.bin")
    sys.exit(0 if (ok1 and ok2) else 1)
