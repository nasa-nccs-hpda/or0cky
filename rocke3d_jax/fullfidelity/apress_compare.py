"""Compare apress_ff.calc_apress against the real Fortran ffz_apress_<itime>.bin dump (D30).
GRAV is inferred exactly from the dump (USE_PLANET_RAD runtime parameter, like RADIUS in
icedyn_geom_compare.py) rather than assumed."""
import sys
import numpy as np

from apress_ff import calc_apress


def read_apress(path):
    raw = np.fromfile(path, dtype=">f8").reshape(-1, 10)
    return dict(i=raw[:, 0], j=raw[:, 1], srfp=raw[:, 2], rsi=raw[:, 3], snowi=raw[:, 4],
                msi=raw[:, 5], apress=raw[:, 6])


def main(path):
    d = read_apress(path)
    n = len(d["i"])
    print(f"{n} real cells")

    # infer GRAV from a cell with RSI > 0 (else APRESS is independent of GRAV)
    has_ice = d["rsi"] > 1e-6
    idx = np.argmax(has_ice)
    if not has_ice[idx]:
        print("no ice-covered cell in this record; cannot infer GRAV, using default")
        grav = 9.80665
    else:
        denom = d["rsi"][idx] * (d["snowi"][idx] + calc_apress.__globals__["ACE1I"] + d["msi"][idx])
        grav = (d["apress"][idx] - 100.0 * (d["srfp"][idx] - 1013.25)) / denom
    print(f"inferred GRAV = {grav!r}")

    mine = calc_apress(d["srfp"], d["rsi"], d["snowi"], d["msi"], grav=grav)
    err = np.abs(mine - d["apress"])
    rel = err / np.maximum(np.abs(d["apress"]), 1e-6)
    print(f"max_abs_err={err.max():.3e} max_relerr={rel.max():.3e}")
    ok = rel.max() < 1e-9
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffz_apress_33312.bin"
    sys.exit(main(path))
