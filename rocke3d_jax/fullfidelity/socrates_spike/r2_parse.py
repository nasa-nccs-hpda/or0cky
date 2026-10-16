"""R2: read ffsoc_dump.bin (format in ffsoc/ffsoc_mod.F90) -> stage-time table and array inventory.   python r2_parse.py ffsoc_dump.bin [--arrays]"""
import sys, struct, collections
import numpy as np
STAGE = {1: "grey_opt_prop (aerosol/cloud/Rayleigh optical properties)", 2: "band loop total (all bands)",
         3: "k-eqv build in solve_band_k_eqv_scl (gas absorption per k-term, flux sums)", 4: "monochromatic_radiance (two-stream coefficients + solver)",
         5: "augment_radiance (+tiles)", 6: "radiance_calc total",
         7: "solve_band_ses (whole routine)", 8: "solve_band_one_gas (whole routine)", 9: "solve_band_without_gas (whole routine)",
         10: "solve_band_k_eqv_scl (whole routine)"}
def read(path):
    b = open(path, "rb").read(); p = 0; recs = []
    def ints(n):
        nonlocal p
        v = struct.unpack(">%di" % n, b[p:p + 4 * n]); p += 4 * n; return v
    while p < len(b):
        tag, iso, nc, ib, k, rank, n1, n2 = ints(8)
        if tag == 900:
            ints(2)
            acc = np.frombuffer(b[p:p + 96], ">f8").copy(); p += 96
            nacc = np.array(ints(12))
            recs.append(dict(tag=tag, iso=iso, ncall=nc, nband=ib, acc=acc, nacc=nacc))
            continue
        ext = ints(2 * rank)
        shape = ext[:rank]; n = int(np.prod(shape))
        a = np.frombuffer(b[p:p + 8 * n], ">f8").reshape(shape, order="F").astype(float); p += 8 * n
        recs.append(dict(tag=tag, iso=iso, ncall=nc, band=ib, k=k, shape=shape, lb=ext[rank:], a=a))
    return recs
if __name__ == "__main__":
    r = read(sys.argv[1])
    t = [x for x in r if x["tag"] == 900]
    print("records: %d, timer records (radiance_calc calls): %d, array records: %d" % (len(r), len(t), len(r) - len(t)))
    for iso in sorted({x["iso"] for x in t}):
        ts = [x for x in t if x["iso"] == iso]
        acc = sum(x["acc"] for x in ts); tot = acc[5]
        print("\nisolir=%d (%s): %d calls, radiance_calc total %.3f s (%.2f ms/call)" % (iso, {1: "solar/SW", 2: "infra-red/LW"}.get(iso, "?"), len(ts), tot, 1e3 * tot / len(ts)))
        for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            print("   stage %d %-80s %8.3f s  %5.1f %%" % (s, STAGE[s], acc[s - 1], 100 * acc[s - 1] / tot))
        rest = acc[1] - acc[0] - acc[2] - acc[3] - acc[4]
        print("   band loop minus (1+3+4+5) = %.3f s (%.1f %%: band set-up, solver selection, rescale, diagnostics, per-band bookkeeping)" % (rest, 100 * rest / tot))
        print("   radiance_calc outside the band loop = %.3f s (%.1f %%: set-up before the bands, Planck, tails)" % (acc[5] - acc[1], 100 * (acc[5] - acc[1]) / tot))
    if "--check" in sys.argv:
        # tag 43 (solver output inside triple_column) must equal the tag 21 record that follows it (same array handed up); report
        n = bad = 0
        for i, x in enumerate(r):
            if x["tag"] == 21 and i and r[i - 1]["tag"] == 43:
                n += 1
                bad += int(not np.array_equal(r[i - 1]["a"], x["a"]))
        print("tag 43 -> tag 21 pairs: %d, not bitwise equal: %d" % (n, bad))
        # transmission / reflection sanity on the VALID entries only (clear region: all layers; strat/conv: layers >= n_cloud_top;
        # the other entries are uninitialised memory in the Fortran and are not meaningful)
        lo, hi, nrec, nbad = 1e300, -1e300, 0, 0
        meta = None
        for x in r:
            if x["tag"] == 44:
                meta = x["a"].ravel()
            elif x["tag"] == 41 and x["k"] in (1, 2) and meta is not None:
                nct, nreg = int(meta[0]), int(meta[1])
                a = x["a"][0]                      # (layer, region)
                for reg in range(min(a.shape[1], nreg)):
                    v = a[:, reg] if reg == 0 else a[nct - 1:, reg]
                    if v.size:
                        lo, hi = min(lo, float(v.min())), max(hi, float(v.max())); nbad += int(((v < -1e-12) | (v > 1 + 1e-12)).sum())
                nrec += 1
        print("tag 41 trans/reflect, valid entries only: %d records, range [%.6g, %.6g], outside [0,1]: %d" % (nrec, lo, hi, nbad))
    if "--arrays" in sys.argv:
        c = collections.Counter((x["tag"], x["iso"]) for x in r if x["tag"] != 900)
        print("array bytes in file: %d MB" % (sum(x["a"].nbytes for x in r if x["tag"] != 900) // 2**20))
        for k, v in sorted(c.items()): print("tag %3d isolir %d: %d arrays" % (k[0], k[1], v))
