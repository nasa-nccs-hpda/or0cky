"""Compare icedyn_geom_ff.geomicdyn/icdyn_masks against the real Fortran ffz_geom.bin dump (D29).
RADIUS is inferred exactly from the dump's DXT column (this rundeck runs under USE_PLANET_RAD, so
RADIUS/OMEGA are runtime planet parameters, not the hardcoded 6371000 m Earth default)."""
import sys
import numpy as np

from icedyn_geom_ff import geomicdyn, icdyn_masks

FIELDS_1D = [
    ("dxt", "nx1"), ("dxu", "nx1"), ("bydx2", "nx1"), ("bydxr", "nx1"),
    ("dyt", "ny1"), ("dyu", "ny1"), ("bydy2", "ny1"), ("bydyr", "ny1"),
    ("cst", "ny1"), ("csu", "ny1"), ("tngt", "ny1"), ("tng", "ny1"), ("bycsu", "ny1"),
]
FIELDS_2D = ["sinen", "bydxdy", "heffm", "uvm"]


def read_geom(path):
    raw = np.fromfile(path, dtype=">f8")
    pos = 0
    nx1, ny1, imicdyn, jmicdyn = raw[0:4].astype(int)
    pos = 4
    out = {"nx1": nx1, "ny1": ny1, "imicdyn": imicdyn, "jmicdyn": jmicdyn}
    for name, dimname in FIELDS_1D:
        n = nx1 if dimname == "nx1" else ny1
        out[name] = raw[pos:pos + n].copy()
        pos += n
    for name in FIELDS_2D[:2]:  # sinen, bydxdy: (nx1,ny1)
        n = nx1 * ny1
        out[name] = raw[pos:pos + n].reshape(ny1, nx1).T.copy()  # Fortran column-major (nx1,ny1)
        pos += n
    for name in FIELDS_2D[2:]:  # heffm, uvm: (nx1,ny1)
        n = nx1 * ny1
        out[name] = raw[pos:pos + n].reshape(ny1, nx1).T.copy()
        pos += n
    out["gfocean"] = raw[pos:pos + imicdyn * ny1].reshape(ny1, imicdyn).T.copy()
    pos += imicdyn * ny1
    assert pos == raw.size, f"size mismatch: consumed {pos}, have {raw.size}"
    return out


def main(path):
    d = read_geom(path)
    nx1, ny1, imicdyn = d["nx1"], d["ny1"], d["imicdyn"]
    print(f"nx1={nx1} ny1={ny1} imicdyn={imicdyn} jmicdyn={d['jmicdyn']}")

    dlon = 2.0 * np.pi / imicdyn
    # DXT[1] (0-based index 1, a real interior column, i=2 in Fortran) = dlon*radius exactly
    radius = d["dxt"][1] / dlon
    print(f"inferred RADIUS = {radius!r}")

    g = geomicdyn(imicdyn, d["jmicdyn"], radius)
    ok = True
    for name, _ in FIELDS_1D:
        ref, mine = d[name], g[name]
        err = np.max(np.abs(ref - mine))
        rel = err / max(np.max(np.abs(ref)), 1e-30)
        status = "OK" if rel < 1e-12 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{name:8s} max_abs_err={err:.3e} rel={rel:.3e} {status}")

    for name in ("sinen", "bydxdy"):
        ref, mine = d[name], g[name]
        err = np.max(np.abs(ref - mine))
        rel = err / max(np.max(np.abs(ref)), 1e-30)
        status = "OK" if rel < 1e-12 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{name:8s} max_abs_err={err:.3e} rel={rel:.3e} {status}")

    heffm, uvm = icdyn_masks(d["gfocean"], nx1, d["ny1"])
    for name, ref, mine in (("heffm", d["heffm"], heffm), ("uvm", d["uvm"], uvm)):
        err = np.max(np.abs(ref - mine))
        n_mismatch = int(np.sum(ref != mine))
        status = "OK" if n_mismatch == 0 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{name:8s} max_abs_err={err:.3e} n_mismatch={n_mismatch}/{ref.size} {status}")

    print("ALL OK" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffz_geom.bin"
    sys.exit(main(path))
