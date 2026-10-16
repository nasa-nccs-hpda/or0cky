"""Compare pbl_ff.advanc against real-Fortran PBL call records (ffp_<itime>.bin)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
import pbl_ff as P

OUT = {"us": 89, "vs": 90, "ws": 91, "tsv": 92, "qsrf": 93, "cm": 94, "ch": 95, "cq": 96, "dskin": 97,
       "ws0": 98, "ustar": 99, "lmonin": 100, "khs": 101, "kms": 102, "kqs": 103, "z0m": 104, "z0h": 105,
       "z0q": 106, "w2_1": 107, "ufluxs": 108, "vfluxs": 109, "tfluxs": 110, "qfluxs": 111, "psi": 114}
PROF = {"u": (115, 123), "v": (123, 131), "t": (131, 139), "q": (139, 147), "e": (147, 154)}


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, 154)


def run(rec, chunk=2048):
    d = P.unpack_records(rec)
    fn = jax.jit(P.advanc_batch)
    outs = []
    for s in range(0, rec.shape[0], chunk):
        sl = {k: v[s:s + chunk] for k, v in d.items()}
        outs.append(jax.tree_util.tree_map(np.asarray, fn(sl)))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def compare(rec, got):
    rows = {}
    it = rec[:, 2].astype(int)
    for k, col in OUT.items():
        ref = rec[:, col]
        d = got[k] - ref
        rows[k] = dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                       ref_rms=float(np.sqrt((ref ** 2).mean())), frac_exact=float((d == 0).mean()),
                       max_rel=float(np.max(np.abs(d) / np.maximum(np.abs(ref), 1e-300))))
    for k, (a, b) in PROF.items():
        ref = rec[:, a:b]
        d = got[k][:, :ref.shape[1]] - ref
        rows[k + "_prof"] = dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                                 ref_rms=float(np.sqrt((ref ** 2).mean())), frac_exact=float((d == 0).mean()),
                                 max_rel=float(np.max(np.abs(d) / np.maximum(np.abs(ref), 1e-300))))
    return rows


if __name__ == "__main__":
    rec = load(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else rec.shape[0]
    rec = rec[:n]
    got = run(rec)
    rows = compare(rec, got)
    for k, r in rows.items():
        print(f"{k:10s} " + "  ".join(f"{a}={b:.3e}" for a, b in r.items()))
