"""Full-record comparison of pbl_ff vs real-Fortran PBL calls; writes JSON summary (quantiles per output)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbl_compare as C

paths = sys.argv[1:-1]
out = sys.argv[-1]
summary = {}
for p in paths:
    rec = C.load(p)
    got = C.run(rec)
    it = rec[:, 2].astype(int)
    s = {"n_records": int(rec.shape[0]), "n_by_itype": {str(k): int((it == k).sum()) for k in (1, 2, 3, 4)}}
    for name, col in C.OUT.items():
        ref = rec[:, col]
        d = np.abs(got[name] - ref)
        rel = d / np.maximum(np.abs(ref), 1e-300)
        s[name] = dict(ref_rms=float(np.sqrt((ref ** 2).mean())), max_abs=float(d.max()),
                       rel_median=float(np.median(rel)), rel_p99=float(np.quantile(rel, .99)), rel_max=float(rel.max()),
                       abs_over_rms_ref_max=float(d.max() / max(np.sqrt((ref ** 2).mean()), 1e-300)))
    summary[os.path.basename(p)] = s
    print(os.path.basename(p), {k: (v["abs_over_rms_ref_max"] if isinstance(v, dict) else v) for k, v in s.items() if k in ("us", "ustar", "cm", "tfluxs", "qfluxs", "ufluxs")}, flush=True)
json.dump(summary, open(out, "w"), indent=1)
