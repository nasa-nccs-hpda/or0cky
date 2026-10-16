"""Validate oadvt2_ff.py's OADVT2/OADVTX2/OADVTY2/OADVTZ2 port against real Fortran dumps (D45)."""
import sys
import numpy as np
from oadvt2_ff import oadvt2, IM, JM, LMO
from odhorz_compare import load_lmm, load_lmv, load_lmu, FF_DEFAULT


def load_mmi(date, itime):
    """OCEAN_DYN's MMI, dumped directly (D45). An earlier attempt re-derived it as
    MO0*DXYPO(J) using ODHORZ0's mo0 input -- this does NOT match the real MMI (found via
    mismatch debugging: MMI is a persistent module array that ODHORZ0 only partially
    overwrites, not a dense recomputation from mo0), so it is now recorded directly instead."""
    ff = f"{FF_DEFAULT}/{date}"
    raw = np.fromfile(f"{ff}/ffz_mmi_{itime}.bin", dtype='>f8')
    n3 = IM * JM * LMO
    assert raw.size == 1 + n3, (raw.size, 1 + n3)
    mmi = np.zeros((IM + 1, JM + 1, LMO + 1))
    mmi[1:, 1:, 1:] = raw[1:1 + n3].reshape(LMO, JM, IM).transpose(2, 1, 0)
    return mmi


def load_oadvt2_before(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    expected = 2 + 9 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    dtdum = raw[off]; off += 1
    rec = {"itime": itime, "dtdum": dtdum}
    for name in ["smw", "g0m", "gxmo", "gymo", "gzmo", "s0m", "sxmo", "symo", "szmo"]:
        a = np.zeros((IM + 1, JM + 1, LMO + 1))
        a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
        rec[name] = a
    assert off == raw.size
    return rec


def load_oadvt2_after(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    expected = 1 + 9 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    rec = {"itime": itime}
    for name in ["mo1", "g0m", "gxmo", "gymo", "gzmo", "s0m", "sxmo", "symo", "szmo"]:
        a = np.zeros((IM + 1, JM + 1, LMO + 1))
        a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
        rec[name] = a
    assert off == raw.size
    return rec


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    before = load_oadvt2_before(f"{ff}/ffz_oadvt2_before_{itime}.bin")
    after = load_oadvt2_after(f"{ff}/ffz_oadvt2_after_{itime}.bin")
    mmi = load_mmi(date, itime)

    # G0M advection (QLIMIT=.FALSE.), starting from SMU/SMV real ground-truth (D44's ffz_smfinal)
    smu_final, smv_final = _load_smfinal(ff, itime)

    ma_g, g0m, gxmo, gymo, gzmo = oadvt2(
        mmi, before["g0m"], before["gxmo"], before["gymo"], before["gzmo"],
        before["dtdum"], False, smu_final, smv_final, before["smw"], lmu, lmv, lmm)

    ma_s, s0m, sxmo, symo, szmo = oadvt2(
        mmi, before["s0m"], before["sxmo"], before["symo"], before["szmo"],
        before["dtdum"], True, smu_final, smv_final, before["smw"], lmu, lmv, lmm)

    results = {}
    for name, computed, expected in [
        ("G0M", g0m, after["g0m"]), ("GXMO", gxmo, after["gxmo"]),
        ("GYMO", gymo, after["gymo"]), ("GZMO", gzmo, after["gzmo"]),
        ("S0M", s0m, after["s0m"]), ("SXMO", sxmo, after["sxmo"]),
        ("SYMO", symo, after["symo"]), ("SZMO", szmo, after["szmo"]),
        ("MA_from_S0M_call", ma_s, after["mo1"]),
    ]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6))
    return results


def _load_smfinal(ff, itime):
    raw = np.fromfile(f"{ff}/ffz_smfinal_{int(itime)}.bin", dtype='>f8')
    n3 = IM * JM * LMO
    off = 1
    smu = np.zeros((IM + 1, JM + 1, LMO + 1))
    smu[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
    smv = np.zeros((IM + 1, JM + 1, LMO + 1))
    smv[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
    return smu, smv


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
