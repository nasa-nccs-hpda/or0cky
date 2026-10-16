"""Daily-products oracle: serve the recorded nov26 packet (the state is irrelevant for the products) at the first radiation step of other
calendar days; the persistent server steps its clock through the day boundaries and runs the daily updates, and the instrumented build
dumps tags 50-62 and the statics of every request (FFSOC_PL, light).
env R6_I_EDIT="name=value;name=value": replace those namelist entries of the run's rundeck copy (e.g. master_yr=1990;volc_yr=0) to test another
configuration of the daily products; env R6_NODUMP=1 (with the control binary: no dumps, no size limit); env R6_SAVE_DIR: directory for the server outputs of every request (out_<itime>_0.npz)
usage: python r6_days_oracle.py BIN RUNDIR JDAY[:YEARS_AFTER_1950] ...   (days in increasing order; 330 = nov26 1950 (itime 33312);
JDAY 1..365 after jday 330 is the next calendar year)   prints one line per request: jday, itime, server seconds."""
import os, sys, time
import numpy as np
os.environ.setdefault("FF_DATA", "ff_data")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import radiation_server as rs
import radiation_server_persist as P

ITIMEI, NRAD, DAY0 = 16032, 5, 17520            # model step of 1950-01-01 00:00 is 17520 (48 steps per day)


def first_rad_itime(jday, year_after_1950=0):
    it = DAY0 + (jday - 1) * 48 + 365 * 48 * year_after_1950
    while (it - ITIMEI) % NRAD:
        it += 1
    return it


def apply_rundeck_edits(lines, edits):
    """Replace the namelist entries `name=value` of a rundeck (list of lines) by `edits` (dict); entries not present are added before the
    terminating `&&END_PARAMETERS` line.  Returns the new list of lines."""
    txt, done = list(lines), set()
    for i, ln in enumerate(txt):
        k = ln.split("=")[0].strip()
        if "=" in ln and k in edits:
            txt[i] = " %s=%s" % (k, edits[k])
            done.add(k)
    missing = set(edits) - done
    if missing:
        at = max(i for i, ln in enumerate(txt) if ln.strip() == "&&END_PARAMETERS")
        txt[at:at] = [" %s=%s" % (k, edits[k]) for k in sorted(missing)]
    return txt


class EditedServer(P.PersistentServer):
    """PersistentServer whose rundeck copy (scratch run directory) has entries replaced from env R6_I_EDIT (name=value;...)."""

    def _prepare(self):
        super()._prepare()
        edits = dict(e.split("=", 1) for e in os.environ.get("R6_I_EDIT", "").split(";") if e)
        if not edits:
            return
        path = os.path.join(self.rundir, "I")
        open(path, "w").write("\n".join(apply_rundeck_edits(open(path).read().split("\n"), edits)))


def main(bin_, rundir, days):
    its = []
    for d in days:
        j, _, y = d.partition(":")
        j, y = int(j), int(y or 0)
        it = first_rad_itime(j, y)
        assert not its or it > its[-1][1], "days must be increasing"
        its.append((d, it))
    s = EditedServer("nov26", rundir=rundir, binary=bin_, extra_env=({} if os.environ.get("R6_NODUMP") else
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
