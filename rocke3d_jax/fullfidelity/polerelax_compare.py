"""Validate polerelax_ff.py's polar relax block port against real Fortran dumps (D39)."""
import sys
import numpy as np
from polerelax_ff import polerelax, IM, JM, LMO

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")


def load_geom(path):
    raw = np.fromfile(path, dtype='>f8')
    im_r, jm_r = raw[0:2].astype(int)
    assert (im_r, jm_r) == (IM, JM)
    off = 2
    lmu_flat = raw[off:off + IM * JM]; off += IM * JM
    lmv_flat = raw[off:off + IM * JM]; off += IM * JM
    assert off == raw.size

    def pad(flat):
        a = np.zeros((IM + 1, JM + 1), dtype=int)
        a[1:, 1:] = flat.reshape(JM, IM).T.astype(int)
        return a

    return pad(lmu_flat), pad(lmv_flat)


def load_record(path):
    raw = np.fromfile(path, dtype='>f8')
    n = IM * JM * LMO
    assert raw.size == 1 + 8 * n, raw.size
    itime = raw[0]
    off = 1
    names = ["uo0", "vo0", "uod0", "vod0", "uo1", "vo1", "uod1", "vod1"]
    out = {"itime": itime}
    for name in names:
        flat = raw[off:off + n]; off += n
        a = np.zeros((IM + 1, JM + 1, LMO + 1))
        a[1:, 1:, 1:] = flat.reshape(LMO, JM, IM).transpose(2, 1, 0)
        out[name] = a
    return out


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmu, lmv = load_geom(f"{ff}/ffz_polerelax_geom.bin")
    rec = load_record(f"{ff}/ffz_polerelax_{itime}.bin")

    uo, vo, uod, vod = polerelax(lmu, lmv, rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"])

    results = {}
    for name, computed, expected in [
        ("UO", uo, rec["uo1"]), ("VO", vo, rec["vo1"]),
        ("UOD", uod, rec["uod1"]), ("VOD", vod, rec["vod1"]),
    ]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-9, rtol=1e-9))
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
