"""Daily-products oracle: serve the recorded nov26 packet (the state is irrelevant for the products) at the first radiation step of other
calendar days; the persistent server steps its clock through the day boundaries and runs the daily updates, and the instrumented build
dumps tags 50-62 and the statics of every request (FFSOC_PL, light).
env R6_NODUMP=1 (with the control binary: no dumps, no size limit); env R6_SAVE_DIR: directory for the server outputs of every request (out_<itime>_0.npz)
usage: python r6_days_oracle.py BIN RUNDIR JDAY[:YEARS_AFTER_1950] ...   (days in increasing order; 330 = nov26 1950 (itime 33312);
JDAY 1..365 after jday 330 is the next calendar year)   prints one line per request: jday, itime, server seconds."""
import os, sys, time
import numpy as np
os.environ.setdefault("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import radiation_server as rs
import radiation_server_persist as P

ITIMEI, NRAD, DAY0 = 16032, 5, 17520            # model step of 1950-01-01 00:00 is 17520 (48 steps per day)


def first_rad_itime(jday, year_after_1950=0):
    it = DAY0 + (jday - 1) * 48 + 365 * 48 * year_after_1950
    while (it - ITIMEI) % NRAD:
        it += 1
    return it


def main(bin_, rundir, days):
    its = []
    for d in days:
        j, _, y = d.partition(":")
        j, y = int(j), int(y or 0)
        it = first_rad_itime(j, y)
        assert not its or it > its[-1][1], "days must be increasing"
        its.append((d, it))
    s = P.PersistentServer("nov26", rundir=rundir, binary=bin_, extra_env=({} if os.environ.get("R6_NODUMP") else
                                                                         {"FFSOC": "1", "FFSOC_STRIDE": "1000000000", "FFSOC_LIGHT": "1",
                                                                          "FFSOC_PL": "1"})).start()
    try:
        i = rs.read_packet(rs.FF + "/nov26_day/rsv_n26_33312_in.bin")
        for d, it in its:
            t = time.time()
            r = s.request(i, it, None)
            if os.environ.get("R6_SAVE_DIR"):                  # the 21 non-AIJ outputs of the Fortran server at this date (end-to-end oracle)
                keys = [k for k in rs.OUTPUT_FIELDS if k != "AIJ"]
                np.savez(os.path.join(os.environ["R6_SAVE_DIR"], "out_%d_%d.npz" % (it, 0)), **{k: r[k] for k in keys})
            print("%s itime %d server %.1f wall %.1f" % (d, it, r["_server_s"], time.time() - t), flush=True)
    finally:
        s.stop()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
