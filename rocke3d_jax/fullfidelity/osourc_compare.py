"""Compare osourc_ff.osourc against the real Fortran ffz_osourc_<itime>.bin dump (D34)."""
import sys
import numpy as np

from osourc_ff import osourc, LMO

REC = 20 + 4 * LMO + 6  # header + g0ml0 + gzml0 + g0ml1 + gzml1 + (dmoo,deoo,dmoi,deoi,dsoo,dsoi)


def read_osourc(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, REC)
    d = dict(i=raw[:, 0], j=raw[:, 1], roice=raw[:, 2], mo0=raw[:, 3], so0=raw[:, 4],
             dxypj=raw[:, 5], bydxypj=raw[:, 6], lmij=raw[:, 7], runo=raw[:, 8], runi=raw[:, 9],
             eruno=raw[:, 10], eruni=raw[:, 11], sruno=raw[:, 12], sruni=raw[:, 13],
             srox1=raw[:, 14], srox2=raw[:, 15], lmo=raw[:, 16], mo1=raw[:, 17], so1=raw[:, 18])
    p = 20
    d["g0ml0"] = raw[:, p:p + LMO]; p += LMO
    d["gzml0"] = raw[:, p:p + LMO]; p += LMO
    d["g0ml1"] = raw[:, p:p + LMO]; p += LMO
    d["gzml1"] = raw[:, p:p + LMO]; p += LMO
    d["dmoo"] = raw[:, p]; d["deoo"] = raw[:, p + 1]; d["dmoi"] = raw[:, p + 2]
    d["deoi"] = raw[:, p + 3]; d["dsoo"] = raw[:, p + 4]; d["dsoi"] = raw[:, p + 5]
    return d


def main(path):
    d = read_osourc(path)
    n = len(d["i"])
    print(f"{n} real cells")
    worst = {"mo": 0.0, "s0m": 0.0, "g0ml": 0.0, "gzml": 0.0, "dmoo": 0.0, "deoo": 0.0,
             "dmoi": 0.0, "deoi": 0.0, "dsoo": 0.0, "dsoi": 0.0}
    for idx in range(n):
        out = osourc(d["roice"][idx], d["mo0"][idx], d["g0ml0"][idx], d["gzml0"][idx],
                      d["so0"][idx], d["dxypj"][idx], d["bydxypj"][idx], int(d["lmij"][idx]),
                      d["runo"][idx], d["runi"][idx], d["eruno"][idx], d["eruni"][idx],
                      d["sruno"][idx], d["sruni"][idx], (d["srox1"][idx], d["srox2"][idx]))
        for k, ref in (("mo", d["mo1"][idx]), ("s0m", d["so1"][idx]), ("dmoo", d["dmoo"][idx]),
                       ("deoo", d["deoo"][idx]), ("dmoi", d["dmoi"][idx]), ("deoi", d["deoi"][idx]),
                       ("dsoo", d["dsoo"][idx]), ("dsoi", d["dsoi"][idx])):
            rel = abs(out[k] - ref) / max(abs(ref), 1e-10)
            if rel > worst[k]:
                worst[k] = rel
        for k, arr in (("g0ml", d["g0ml1"][idx]), ("gzml", d["gzml1"][idx])):
            lmij = int(d["lmij"][idx])
            mine = np.array(out[k][:lmij])
            ref = arr[:lmij]
            rel = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-10)
            m = float(np.max(rel))
            if m > worst[k]:
                worst[k] = m
    ok = True
    for k, v in worst.items():
        status = "OK" if v < 1e-8 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{k:5s} max_relerr={v:.3e} {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffz_osourc_33312.bin"
    sys.exit(main(path))
