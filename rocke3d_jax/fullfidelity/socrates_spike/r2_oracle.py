"""R2: run the persistent radiation server from a given binary (control or instrumented scratch build) on recorded packets and compare
EVERY output field bitwise with the recorded server output (rsv_n26_<it>_out.bin, nov26_day).  The first radiation step is the entry step of the
server; the others are served in order.
usage: python r2_oracle.py BIN RUNDIR [--stride N] [ITIME ...]     (graphcast-env; --stride enables the dumps; default ITIME 33312)
env FF_DATA defaults to the directory with nov26_day (the repo's sibling ff_data)."""
import json, os, sys
os.environ.setdefault("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import radiation_server as rs
import radiation_server_persist as P
bin_, rundir = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
stride = None
if rest and rest[0] == "--stride":
    stride = rest[1]
    rest = rest[2:]
its = [int(x) for x in rest] or [33312]
extra = ({"FFSOC": "1", "FFSOC_STRIDE": stride, **({"FFSOC_LIGHT": "1"} if os.environ.get("R2_LIGHT") else {})} if stride else {})
s = P.PersistentServer("nov26", rundir=rundir, binary=bin_, extra_env=extra).start()
res = {}
try:
    for it in its:
        d = rs.FF + "/nov26_day"
        i, o = rs.read_packet("%s/rsv_n26_%d_in.bin" % (d, it)), rs.read_packet("%s/rsv_n26_%d_out.bin" % (d, it))
        r = s.request(i, it, None)
        keys = [k for k in rs.OUTPUT_FIELDS if k != "AIJ"]
        eq = {k: bool(np.array_equal(r[k], o[k])) for k in keys}
        aij = float(np.abs(r["AIJ"] - o["AIJD"]).max()) if "AIJD" in o else None
        res[it] = dict(server_s=r["_server_s"], aij_maxabs_vs_recorded_delta=aij, equal=eq, all_equal=all(eq.values()),
                       maxabs={k: float(np.abs(r[k] - o[k]).max()) for k, v in eq.items() if not v})
finally:
    s.stop()
print(json.dumps(res, indent=1))
