"""R6: collect, per radiation column, the planet_rad-side records and the SOCRATES-side records of one radiation step from a dump made with
FFSOC_LIGHT=1 FFSOC_PL=1 (stride 1) into one npz.   python r6_collect.py ffsoc_dump.bin out.npz
Column boundary = tag 60 (RCOMPX, before run_planet_rad).  Keys of the npz:
  p60 p50 p51        (N, n) packed vectors (layouts: README, 'R6 tags')
  p52 (N, n_aer, 5), p53 (N, n_aer, 43), p61 (N, 43, nsized), p62 (N, nsized)   (only if present for every column)
  lw_<k>             tag-0 arrays of the LW call (k = tag-0 array number), shape (N, ...)
  sw_<k>, sw_idx     same for the lit columns; sw_idx = column index
  lwend_<band>_<k>, swend_<band>_<k>   tag 30 (k = 1 down, 2 up, 3 direct (SW), 4 tile up (SW))
  alb_wgt_sw         tag 54 (once);  static_<tag>  tags 101-169 (control flags, band limits, planet aerosol tables; once)"""
import sys
import numpy as np
import r6_stream as S

KEEP0 = {1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}


def main(path, out, nper=None):
    cols = []
    cur = None
    alb_wgt = None
    static = {}
    for r in S.stream(path, tags={0, 30, 50, 51, 52, 53, 54, 60, 61, 62} | set(range(101, 200))):
        t = r["tag"]
        if t == 60:
            cur = dict(p60=r["a"], lw={}, sw={}, lwend={}, swend={}, extra={})
            cols.append(cur)
            continue
        if t == 54:
            alb_wgt = r["a"]
            continue
        if t >= 101:
            static[t] = r["a"]
            continue
        if cur is None:
            continue
        if t in (50, 51):
            cur["p%d" % t] = r["a"]
        elif t in (52, 53, 61, 62):
            cur["p%d" % t] = r["a"]
        elif t == 0 and r["k"] in KEEP0:
            cur["lw" if r["iso"] == 2 else "sw"][r["k"]] = r["a"]
        elif t == 30:
            cur["lwend" if r["iso"] == 2 else "swend"][(r["band"], r["k"])] = r["a"]
    if nper and len(cols) > nper:                                   # several radiation steps in one dump: one npz per step
        base = out[:-4] if out.endswith(".npz") else out
        for k in range(len(cols) // nper):
            _save(cols[k * nper:(k + 1) * nper], "%s_%d.npz" % (base, k), alb_wgt, static)
        return
    _save(cols, out, alb_wgt, static)


def _save(cols, out, alb_wgt, static):
    N = len(cols)
    d = {}
    for key in ("p60", "p50", "p51", "p52", "p53", "p61", "p62"):
        have = [c for c in cols if key in c]
        if len(have) == N and N:
            d[key] = np.stack([c[key] for c in cols])
        elif have:
            print("note: %s present for %d of %d columns" % (key, len(have), N))
    for pre in ("lw", "sw"):
        idx = np.array([i for i, c in enumerate(cols) if c[pre]], dtype=int)
        if pre == "sw":
            d["sw_idx"] = idx
        keys = set().union(*[set(cols[i][pre]) for i in idx]) if len(idx) else set()
        for k in keys:
            try:
                d["%s_%d" % (pre, k)] = np.stack([cols[i][pre][k] for i in idx])
            except (KeyError, ValueError):
                print("note: %s_%d not stackable" % (pre, k))
        ends = set().union(*[set(cols[i][pre + "end"]) for i in idx]) if len(idx) else set()
        for (b, k) in ends:
            have = [i for i in idx if (b, k) in cols[i][pre + "end"]]
            if len(have) == len(idx):
                d["%send_%d_%d" % (pre, b, k)] = np.stack([cols[i][pre + "end"][(b, k)] for i in idx])
            else:
                print("note: %send_%d_%d present for %d of %d" % (pre, b, k, len(have), len(idx)))
    for t, a in static.items():
        d["static_%d" % t] = a
    if alb_wgt is not None:
        d["alb_wgt_sw"] = alb_wgt
    np.savez_compressed(out, **d)
    print("columns:", N, "lit:", len(d.get("sw_idx", [])), "keys:", len(d))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
