"""Compare aturb_ff (JAX port) with real-Fortran ATURB dumps (ffa_*_in/out.bin)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffdump_reader import read_dump
import aturb_ff as A
import jax.numpy as jnp


def load_inputs(path_in):
    d = read_dump(path_in, 1)
    g3 = lambda k: np.transpose(d[k], (2, 1, 0))           # (L,I,J) -> (J,I,L)
    i3 = lambda k: np.transpose(d[k], (1, 0, 2))           # (I,J,L) -> (J,I,L)
    s2 = lambda k: d[k].T                                  # (I,J) -> (J,I)
    args = dict(T=i3("T"), Q=i3("Q"), UA=g3("UALIJ"), VA=g3("VALIJ"), E=g3("EGCM"),
                PMID=g3("PMID"), PEDN=g3("PEDN"), PK=g3("PK"), PEK1=g3("PEK")[..., 0],
                PDSIG=g3("PDSIG"), UFLUX1=s2("UFLUX1"), VFLUX1=s2("VFLUX1"),
                TFLUX1=s2("TFLUX1"), QFLUX1=s2("QFLUX1"), TSAVG=s2("TSAVG"), QSAVG=s2("QSAVG"))
    return args, float(d["header"][0]), d


def valid_mask(shape_ji):
    m = np.ones(shape_ji, bool)
    m[0, 1:] = False
    m[-1, 1:] = False        # poles: only i=1 is computed by the model
    return m


def run(path_in, path_out):
    args, dt, din = load_inputs(path_in)
    dout = read_dump(path_out, 1)
    m = valid_mask(args["T"].shape[:2])
    # scrub uninitialised garbage in invalid cells so it cannot produce NaN warnings
    for k in ("UFLUX1", "VFLUX1", "TFLUX1", "QFLUX1", "TSAVG", "QSAVG"):
        args[k] = np.where(m, args[k], 0.0 if k not in ("TSAVG",) else 280.0)
    args["QSAVG"] = np.where(m, args["QSAVG"], 0.0)
    for k in ("UFLUX1", "VFLUX1", "TFLUX1", "QFLUX1"):
        args[k] = np.where(m, args[k], 1e-3)
    res = A.aturb_grid(**{k: jnp.asarray(v) for k, v in args.items()}, dtime=dt)
    ref = {
        "t": np.transpose(dout["T"], (1, 0, 2)), "q": np.transpose(dout["Q"], (1, 0, 2)),
        "e": np.transpose(dout["EGCM"], (2, 1, 0)), "w2": np.transpose(dout["W2GCM"], (2, 1, 0)),
        "pblht": dout["PBLHT"].T, "dclev": dout["DCLEV"].T, "pblptop": dout["PBLPTOP"].T,
    }
    ini = {"t": np.transpose(din["T"], (1, 0, 2)), "q": np.transpose(din["Q"], (1, 0, 2)),
           "e": np.transpose(din["EGCM"], (2, 1, 0))}
    rows = {}
    for k, r in ref.items():
        g = np.asarray(res[k])
        mm = m if r.ndim == 2 else m[..., None]
        mm = np.broadcast_to(mm, r.shape)
        d = (g - r)[mm]
        rr = r[mm]
        row = dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                   ref_rms=float(np.sqrt((rr ** 2).mean())), n_exact=float((d == 0).mean()))
        if k in ini:
            ch = (r - ini[k])[mm]
            row["fortran_change_rms"] = float(np.sqrt((ch ** 2).mean()))
        rows[k] = row
    return rows


if __name__ == "__main__":
    dd = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "33312_c1"
    rows = run(f"{dd}/ffa_{tag}_in.bin", f"{dd}/ffa_{tag}_out.bin")
    for k, r in rows.items():
        print(f"{k:8s} " + "  ".join(f"{a}={b:.3e}" for a, b in r.items()))
