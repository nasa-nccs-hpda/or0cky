"""Validate gmredi_ff.py's gmfexp port against real Fortran dumps (D51)."""
import sys
import numpy as np
from gmredi_ff import gmfexp, IM, JM, LMO
from gmkdif_compare import load_gmkdif
from ocnmeso_compare import FF_DEFAULT
from odhorz_compare import load_lmm, load_lmv, load_lmu


def _to_ijl_standard(flat):
    a = np.zeros((IM + 1, JM + 1, LMO + 1))
    a[1:, 1:, 1:] = flat.reshape(LMO, JM, IM).transpose(2, 1, 0)
    return a


def load_gmfexp(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    per_rec = 2 + 9 * n3
    nrec = raw.size // per_rec
    assert raw.size == nrec * per_rec, (raw.size, per_rec)
    records = []
    off = 0
    for _ in range(nrec):
        rec = {}
        rec["itime"] = raw[off]; off += 1
        rec["qlimit"] = raw[off] > 0.5; off += 1
        for name in ["mo0", "trm0", "txm0", "tym0", "tzm0", "trm1", "txm1", "tym1", "tzm1"]:
            rec[name] = _to_ijl_standard(raw[off:off + n3]); off += n3
        records.append(rec)
    assert off == raw.size
    return records


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    gmk = load_gmkdif(f"{ff}/ffz_gmkdif_{itime}.bin")
    records = load_gmfexp(f"{ff}/ffz_gmfexp_{itime}.bin")

    # BYDH is densgrad's own output (D47) -- reuse via ffz_densgrad, same itime
    from ocnmeso_compare import load_densgrad
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    bydh = dg["bydh"]
    bydzv = dg["bydzv"]

    all_results = []
    for rec in records:
        computed_trm, computed_txm, computed_tym, computed_tzm = gmfexp(
            lmm, lmu, lmv, rec["mo0"], rec["trm0"], rec["txm0"], rec["tym0"], rec["tzm0"],
            bool(rec["qlimit"]),
            gmk["bxx"], gmk["byy"], gmk["bzz"],
            gmk["azx"], gmk["bzx"], gmk["czx"], gmk["aezx"], gmk["ezx"], gmk["cezx"],
            gmk["azy"], gmk["bzy"], gmk["czy"], gmk["aezy"], gmk["ezy"], gmk["cezy"],
            gmk["kpl"], bydh, bydzv)

        results = {}
        for name, computed, expected in [
            ("TRM", computed_trm, rec["trm1"]), ("TXM", computed_txm, rec["txm1"]),
            ("TYM", computed_tym, rec["tym1"]), ("TZM", computed_tzm, rec["tzm1"]),
        ]:
            diff = np.abs(computed - expected)
            results[name] = (diff.max(), np.allclose(computed, expected, atol=1e-6, rtol=1e-6),
                              bool(rec["qlimit"]))
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    dates = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
    all_ok = True
    for date, itime in dates:
        all_results = run_one(date, itime)
        print(f"== {date} (itime={itime}) ==")
        for k, results in enumerate(all_results):
            for name, (maxdiff, ok, qlimit) in results.items():
                status = "OK" if ok else "FAIL"
                print(f"  call {k} (qlimit={qlimit}) {name}: max_abs_diff={maxdiff:.3e}  [{status}]")
                all_ok = all_ok and ok
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)
