"""Validate ostres2_ff.py's OSTRES2 port against real Fortran dumps (D36).

Usage: python3 ostres2_compare.py [ff_data_dir]  (default: nov26)
"""
import sys
import numpy as np
from ostres2_ff import ostres2, geomo_arrays, IM, JM

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")


def load_geom(path):
    raw = np.fromfile(path, dtype='>f8')
    off = 3
    im_r, jm_r, ivnp = raw[0:3].astype(int)
    assert im_r == IM and jm_r == JM
    lmu_flat = raw[off:off + IM * JM]; off += IM * JM
    lmv_flat = raw[off:off + IM * JM]; off += IM * JM
    dxyso_d = raw[off:off + JM]; off += JM
    dxyno_d = raw[off:off + JM]; off += JM
    dxyvo_d = raw[off:off + JM]; off += JM
    cosic_d = raw[off:off + IM]; off += IM
    sinic_d = raw[off:off + IM]; off += IM
    assert off == raw.size

    def pad1(flat2d):
        a = np.zeros((IM + 1, JM + 1))
        a2d = flat2d.reshape(JM, IM)  # J-major (I fastest), matches Fortran column-major write
        a[1:, 1:] = a2d.T
        return a

    lmu = pad1(lmu_flat)
    lmv = pad1(lmv_flat)
    dxyso = np.zeros(JM + 1); dxyso[1:] = dxyso_d
    dxyno = np.zeros(JM + 1); dxyno[1:] = dxyno_d
    dxyvo = np.zeros(JM + 1); dxyvo[1:] = dxyvo_d
    cosic = np.zeros(IM + 1); cosic[1:] = cosic_d
    sinic = np.zeros(IM + 1); sinic[1:] = sinic_d
    return dict(ivnp=ivnp, lmu=lmu, lmv=lmv, dxyso=dxyso, dxyno=dxyno,
                dxyvo=dxyvo, cosic=cosic, sinic=sinic)


def load_record(path):
    raw = np.fromfile(path, dtype='>f8')
    assert raw.size == 1 + 13 * IM * JM, raw.size
    itime = raw[0]
    off = 1
    names = ["dmua", "dmva", "dmui", "dmvi", "mo1",
             "uo0", "vo0", "uod0", "vod0", "uo1", "vo1", "uod1", "vod1"]
    out = {"itime": itime}
    for n in names:
        flat = raw[off:off + IM * JM]; off += IM * JM
        a = np.zeros((IM + 1, JM + 1))
        a[1:, 1:] = flat.reshape(JM, IM).T
        out[n] = a
    return out


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    geom = load_geom(f"{ff}/ffz_ostres2_geom.bin")
    rec = load_record(f"{ff}/ffz_ostres2_{itime}.bin")

    uo, vo, uod, vod = ostres2(
        geom["lmu"], geom["lmv"], rec["dmua"], rec["dmva"], rec["dmui"], rec["dmvi"],
        rec["mo1"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], ivnp=geom["ivnp"])

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
