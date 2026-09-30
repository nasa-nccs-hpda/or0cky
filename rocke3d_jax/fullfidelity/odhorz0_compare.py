"""Validate odhorz0_ff.py's ODHORZ0 port against real Fortran dumps (D40)."""
import sys
import numpy as np
from odhorz0_ff import odhorz0, IM, JM, LMO

FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"


def load_geom(path):
    raw = np.fromfile(path, dtype='>f8')
    im_r, jm_r = raw[0:2].astype(int)
    assert (im_r, jm_r) == (IM, JM)
    off = 2
    lmm_flat = raw[off:off + IM * JM]; off += IM * JM
    assert off == raw.size
    a = np.zeros((IM + 1, JM + 1), dtype=int)
    a[1:, 1:] = lmm_flat.reshape(JM, IM).T.astype(int)
    return a


def load_lmv(path):
    """Reuse LMV from D39's ffz_polerelax_geom.bin (layout: im,jm,LMU,LMV -- 2-double header,
    no ivnp, unlike D36's ffz_ostres2_geom.bin)."""
    raw = np.fromfile(path, dtype='>f8')
    off = 2
    off += IM * JM  # skip LMU
    lmv_flat = raw[off:off + IM * JM]
    a = np.zeros((IM + 1, JM + 1), dtype=int)
    a[1:, 1:] = lmv_flat.reshape(JM, IM).T.astype(int)
    return a


def load_record(path):
    raw = np.fromfile(path, dtype='>f8')
    n2 = IM * JM
    n3 = IM * JM * LMO
    assert raw.size == 1 + 2 * n2 + 19 * n3, raw.size
    itime = raw[0]
    off = 1

    def next2d():
        nonlocal off
        flat = raw[off:off + n2]; off += n2
        a = np.zeros((IM + 1, JM + 1))
        a[1:, 1:] = flat.reshape(JM, IM).T
        return a

    def next3d():
        nonlocal off
        flat = raw[off:off + n3]; off += n3
        a = np.zeros((IM + 1, JM + 1, LMO + 1))
        a[1:, 1:, 1:] = flat.reshape(LMO, JM, IM).transpose(2, 1, 0)
        return a

    out = {"itime": itime}
    out["opress"] = next2d()
    for name in ["g0m", "gzm", "s0m", "szm", "mo0", "uo0", "vo0", "vup", "vdn"]:
        out[name] = next3d()
    out["opbot"] = next2d()
    for name in ["gup", "gdn", "sup", "sdn", "dzgdp", "vbar", "dh3d", "mo1", "uo1", "vo1"]:
        out[name] = next3d()
    assert off == raw.size
    return out


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    rec = load_record(f"{ff}/ffz_odhorz0_{itime}.bin")

    out = odhorz0(lmm, lmv, rec["opress"], rec["g0m"], rec["gzm"], rec["s0m"], rec["szm"],
                  rec["mo0"], rec["uo0"], rec["vo0"], rec["vup"], rec["vdn"])

    results = {}
    for name, computed, expected in [
        ("OPBOT", out["opbot"], rec["opbot"]), ("GUP", out["gup"], rec["gup"]),
        ("GDN", out["gdn"], rec["gdn"]), ("SUP", out["sup"], rec["sup"]),
        ("SDN", out["sdn"], rec["sdn"]), ("dZGdP", out["dzgdp"], rec["dzgdp"]),
        ("VBAR", out["vbar"], rec["vbar"]), ("DH3D", out["dh3d"], rec["dh3d"]),
        ("MO", out["mo"], rec["mo1"]), ("UO", out["uo"], rec["uo1"]), ("VO", out["vo"], rec["vo1"]),
    ]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6))
    return results


if __name__ == "__main__":
    dates = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
    all_ok = True
    for date, itime in dates:
        results = run_one(date, itime)
        print(f"== {date} (itime={itime}) ==")
        for name, (maxdiff, ok) in results.items():
            status = "OK" if ok else "FAIL"
            print(f"  {name}: max_abs_diff={maxdiff:.3e}  [{status}]")
            all_ok = all_ok and ok
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)
