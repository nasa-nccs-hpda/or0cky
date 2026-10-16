"""Validate ocnmeso_ff.py's ocnstate_derived/densgrad_vertical/get_1d_mesodiff ports against
real Fortran dumps (D47)."""
import sys
import numpy as np
from ocnmeso_ff import ocnstate_derived, densgrad_vertical, get_1d_mesodiff, IM, JM, LMO
from odhorz_compare import load_lmm, FF_DEFAULT
from odhorz0_compare import load_record as load_odhorz0_record


def _to_ijl(flat_lmo_first):
    """Fortran arrays declared (LMO,IM,JM) -- raw bytes are L-fastest, then I, then J (column-
    major). Returns a (IM+1,JM+1,LMO+1) array in this project's usual (i,j,l) convention."""
    a = np.zeros((IM + 1, JM + 1, LMO + 1))
    a[1:, 1:, 1:] = flat_lmo_first.reshape(JM, IM, LMO).transpose(1, 0, 2)
    return a


def _to_ijl_standard(flat_ij_first):
    """Fortran arrays declared (IM,JM,LMO) -- the project's usual convention."""
    a = np.zeros((IM + 1, JM + 1, LMO + 1))
    a[1:, 1:, 1:] = flat_ij_first.reshape(LMO, JM, IM).transpose(2, 1, 0)
    return a


def _to_ij(flat):
    a = np.zeros((IM + 1, JM + 1))
    a[1:, 1:] = flat.reshape(JM, IM).T
    return a


def load_ocnstate_derived(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    n2 = IM * JM
    expected = 1 + 5 * n3 + n2 + 8 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    rec = {"itime": itime}
    for name in ["mo0", "g0m0", "gzm0", "s0m0", "szm0"]:
        rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
    rec["opress0"] = _to_ij(raw[off:off + n2]); off += n2
    for name in ["vup", "vdn", "g3d", "s3d", "p3d", "t3d", "rho", "vbar"]:
        rec[name] = _to_ijl(raw[off:off + n3]); off += n3
    assert off == raw.size
    return rec


def load_densgrad(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    expected = 1 + 5 * n3 + 4 * n3 + 4 * n3 + 5 * n3 + 2 * n3
    assert raw.size == expected, (raw.size, expected)
    off = 0
    itime = raw[off]; off += 1
    rec = {"itime": itime}
    for name in ["mo0", "g0m0", "gzm0", "s0m0", "szm0"]:
        rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
    for name in ["g3d", "s3d", "p3d", "vbar"]:
        rec[name] = _to_ijl(raw[off:off + n3]); off += n3
    for name in ["vup", "vdn", "vupu", "vdnu"]:
        rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
    for name in ["dzv", "bydzv", "bydh", "rhomz", "byrhoz"]:
        rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
    for name in ["rhox", "rhoy"]:
        rec[name] = _to_ijl(raw[off:off + n3]); off += n3
    assert off == raw.size
    return rec


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    osd = load_ocnstate_derived(f"{ff}/ffz_ocnstate_derived_{itime}.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    odhorz0_rec = load_odhorz0_record(f"{ff}/ffz_odhorz0_{itime}.bin")
    dh = odhorz0_rec["dh3d"]

    g3d, s3d, p3d, vbar, rho = ocnstate_derived(
        osd["mo0"], osd["g0m0"], osd["gzm0"], osd["s0m0"], osd["szm0"],
        osd["opress0"], lmm, osd["vup"], osd["vdn"])

    results = {}
    for name, computed, expected in [
        ("G3D", g3d, osd["g3d"]), ("S3D", s3d, osd["s3d"]), ("P3D", p3d, osd["p3d"]),
        ("VBAR", vbar, osd["vbar"]), ("RHO", rho, osd["rho"]),
    ]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6))

    dzv, bydzv, bydh, rhomz, byrhoz = densgrad_vertical(
        lmm, dh, dg["vbar"], dg["vup"], dg["vdn"], dg["vupu"], dg["vdnu"])

    for name, computed, expected in [
        ("DZV", dzv, dg["dzv"]), ("BYDZV", bydzv, dg["bydzv"]), ("BYDH", bydh, dg["bydh"]),
        ("RHOMZ", rhomz, dg["rhomz"]), ("BYRHOZ", byrhoz, dg["byrhoz"]),
    ]:
        diff = np.abs(computed - expected)
        results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6))

    k3d = get_1d_mesodiff(lmm)
    results["K3D(nonzero-check-only)"] = (0.0, np.any(k3d != 0.0))

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
