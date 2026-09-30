"""Validate odhorz_ff.py's ODHORZ port against real Fortran dumps (D42)."""
import sys
import numpy as np
from odhorz_ff import odhorz, IM, JM, LMO
from odhorz0_compare import load_geom as load_lmm, load_lmv, FF_DEFAULT

REC_NAMES_3D = ["moh", "uoh", "voh", "uodh", "vodh", "mo0", "uo0", "vo0", "uod0", "vod0",
                "vbar", "dzgdp", "usmooth", "pgfx", "mo1", "uo1", "vo1", "uod1", "vod1"]
REC_NAMES_2D = ["opboth", "opbot0", "opbot1"]


def load_hocean(path):
    raw = np.fromfile(path, dtype='>f8')
    im_r, jm_r = raw[0:2].astype(int)
    assert (im_r, jm_r) == (IM, JM)
    flat = raw[2:2 + IM * JM]
    a = np.zeros((IM + 1, JM + 1))
    a[1:, 1:] = flat.reshape(JM, IM).T
    return a


def load_lmu(path):
    """Reuse LMU from D40's ffz_odhorz0_geom.bin? No -- that's LMM. LMU comes from D36's
    ffz_ostres2_geom.bin (layout: im,jm,ivnp,LMU,LMV,...)."""
    raw = np.fromfile(path, dtype='>f8')
    off = 3
    lmu_flat = raw[off:off + IM * JM]
    a = np.zeros((IM + 1, JM + 1), dtype=int)
    a[1:, 1:] = lmu_flat.reshape(JM, IM).T.astype(int)
    return a


def load_records(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    n2 = IM * JM
    per_rec = 2 + 5 * n3 + n2 + 5 * n3 + n2 + 2 * n3 + 2 * n3 + 5 * n3 + n2
    assert raw.size % per_rec == 0, (raw.size, per_rec)
    nrec = raw.size // per_rec
    records = []
    off = 0
    for _ in range(nrec):
        rec = {}
        rec["itime"] = raw[off]; off += 1
        rec["dt"] = raw[off]; off += 1
        for name in ["moh", "uoh", "voh", "uodh", "vodh"]:
            a = np.zeros((IM + 1, JM + 1, LMO + 1))
            a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
            rec[name] = a
        a2 = np.zeros((IM + 1, JM + 1)); a2[1:, 1:] = raw[off:off + n2].reshape(JM, IM).T; off += n2
        rec["opboth"] = a2
        for name in ["mo0", "uo0", "vo0", "uod0", "vod0"]:
            a = np.zeros((IM + 1, JM + 1, LMO + 1))
            a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
            rec[name] = a
        a2 = np.zeros((IM + 1, JM + 1)); a2[1:, 1:] = raw[off:off + n2].reshape(JM, IM).T; off += n2
        rec["opbot0"] = a2
        for name in ["vbar", "dzgdp", "usmooth", "pgfx"]:
            a = np.zeros((IM + 1, JM + 1, LMO + 1))
            a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
            rec[name] = a
        for name in ["mo1", "uo1", "vo1", "uod1", "vod1"]:
            a = np.zeros((IM + 1, JM + 1, LMO + 1))
            a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
            rec[name] = a
        a2 = np.zeros((IM + 1, JM + 1)); a2[1:, 1:] = raw[off:off + n2].reshape(JM, IM).T; off += n2
        rec["opbot1"] = a2
        records.append(rec)
    assert off == raw.size
    return records


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    hocean = load_hocean(f"{ff}/ffz_odhorz_hocean.bin")
    records = load_records(f"{ff}/ffz_odhorz_{itime}.bin")

    all_results = []
    for k, rec in enumerate(records):
        mo, uo, vo, uod, vod, opbot = odhorz(
            lmm, lmu, lmv, hocean, rec["dt"],
            rec["moh"], rec["uoh"], rec["voh"], rec["uodh"], rec["vodh"], rec["opboth"],
            rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], rec["opbot0"],
            rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"])
        results = {}
        for name, computed, expected in [
            ("MO", mo, rec["mo1"]), ("UO", uo, rec["uo1"]), ("VO", vo, rec["vo1"]),
            ("UOD", uod, rec["uod1"]), ("VOD", vod, rec["vod1"]), ("OPBOT", opbot, rec["opbot1"]),
        ]:
            diff = np.abs(computed - expected)
            results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6))
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    dates = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
    all_ok = True
    for date, itime in dates:
        all_results = run_one(date, itime)
        print(f"== {date} (itime={itime}), {len(all_results)} records ==")
        for k, results in enumerate(all_results):
            for name, (maxdiff, ok) in results.items():
                status = "OK" if ok else "FAIL"
                print(f"  call {k} {name}: max_abs_diff={maxdiff:.3e}  [{status}]")
                all_ok = all_ok and ok
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)
