"""SOCRATES spike S1b: who calls the hot routine?  Sampled gdb backtraces (attach, bt, detach) of the UNMODIFIED persistent server during
one request; perf's dwarf unwinder does not resolve the callers in this Intel binary.  No source is touched.
usage: python gdb_callers.py ITIME NSAMPLES OUT.txt"""
import os, subprocess, sys, threading, time, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import radiation_server as rs, radiation_server_persist as rp
it, ns, outf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
srv = rp.PersistentServer("nov26").start()
st = rs.read_packet("%s/nov26_day/rsv_n26_%d_in.bin" % (rs.FF, it))
res = {}
th = threading.Thread(target=lambda: res.update(srv.request(st, it)))
th.start()
time.sleep(1.5)
out = open(outf, "w")
n = 0
while th.is_alive() and n < ns:
    time.sleep(random.uniform(0.0, 0.25))
    r = subprocess.run(["gdb", "-p", str(srv.proc.pid), "-batch", "-ex", "bt 12"], capture_output=True, text=True, timeout=120)
    fr = [l.split()[3] if l.startswith("#") and len(l.split()) > 3 and l.split()[2] == "in" else l.split()[1] for l in r.stdout.splitlines() if l.startswith("#")]
    out.write(" < ".join(fr) + "\n")
    out.flush()
    n += 1
th.join()
srv.stop()
