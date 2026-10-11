"""SOCRATES spike S1: profile the UNMODIFIED Fortran RADIA call of the persistent server with an external profiler (perf, user space).
No SOCRATES / ModelE file is touched.  Requests are the recorded packets ff_data/nov26_day/rsv_n26_<it>_in.bin.
usage: python profile_fortran.py OUTDIR ITIME [ITIME ...]      -> OUTDIR/perf_<itime>.data and perf_<itime>.txt (flat, by symbol)
env PERF_CG=1: also record call graphs (perf --call-graph dwarf, 499 Hz) and write perf_<itime>_callers.txt (inclusive, by caller)
"""
import os, signal, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import radiation_server as rs
import radiation_server_persist as rp

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
FFD = rs.FF + "/nov26_day"
srv = rp.PersistentServer("nov26").start()
print("server start %.1f s, entry itime %d, pid %d" % (srv.start_wall, srv.itime_entry, srv.proc.pid), flush=True)
for it in [int(x) for x in sys.argv[2:]]:
    st = rs.read_packet("%s/rsv_n26_%d_in.bin" % (FFD, it))
    data = os.path.join(out, "perf_%d.data" % it)
    cg = os.environ.get("PERF_CG") == "1"
    cmd = ["perf", "record", "-F", "499" if cg else "999", "-o", data, "-p", str(srv.proc.pid)]
    if cg:
        cmd[2:2] = ["--call-graph", "dwarf,8192"]
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    time.sleep(1.0)
    r = srv.request(st, it)
    p.send_signal(signal.SIGINT)
    p.wait()
    txt = subprocess.run(["perf", "report", "-i", data, "--stdio", "--no-children", "--sort", "sym,dso", "-F", "overhead,sample,sym"],
                         capture_output=True, text=True).stdout
    open(os.path.join(out, "perf_%d.txt" % it), "w").write(txt)
    if cg:
        t2 = subprocess.run(["perf", "report", "-i", data, "--stdio", "--children", "--no-demangle", "-g", "none", "--sort", "sym"],
                            capture_output=True, text=True).stdout
        open(os.path.join(out, "perf_%d_callers.txt" % it), "w").write(t2)
    print("itime %d server_s %.2f wall_s %.2f" % (it, r["_server_s"], r["_wall_s"]), flush=True)
srv.stop()
