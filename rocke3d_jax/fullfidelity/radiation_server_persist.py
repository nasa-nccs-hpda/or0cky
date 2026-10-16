"""D159: persistent ModelE radiation server (real RADIA, SOCRATES as a black box), Python side.

The Fortran side is instrumentation/ATM_DRV_radsrv_persist.f.patch (units 1500-1529): the model starts from a restart, reaches the
first radiation step once (so that every non-packet input RADIA reads is initialised), and then the hook never returns: it serves
requests until STOP or until the client goes away.  One process, one start-up (~16 s), then one RADIA call (~11 s) per request instead
of a full model restart per call (27-190 s).

Protocol (text lines; two named pipes in the run directory; the packet files are exchanged through the file system, same packet
format as radiation_server.py):
  request FIFO  (python -> model), one line per request:
      SERVE <itime> <seed|-> <packet_in> <packet_out>      itime must be a radiation step (MOD(itime-ITIMEI,NRAD)=0); any step of the
                                                           entry day in any order; a later day is reached by stepping the model clock
                                                           through the day boundary and running the daily updates (daily_orbit,
                                                           daily_RAD, daily_EARTH); an earlier day than the server's current one is an error
      PING | STOP
  response FIFO (model -> python), one line per request:   READY <itime> <seed> | OK <itime> <seed> <seconds> | ERR <text> | OK STOP
  seed = the random-number seed IX set before RADIA ('-': the value at entry).  RADIA draws random numbers only for the cloud-overlap
  diagnostics; the seed of the real model at a given step is a trajectory property (ffc_cse_out SEEDS), so a client that wants the same
  diagnostics as the one-shot server at that step passes it.
Shutdown: STOP; EOF of the request FIFO (python died, closed fds) ends the model; the child is started with PR_SET_PDEATHSIG=SIGKILL
(Linux) so it also dies if python is killed, and an atexit handler stops every live server.

Functions/classes: PersistentServer (start, request/run_radiation, stop, context manager), run_radiation (module level, lazily
started default server), server_available, DATES (restart registry).
"""
import atexit
import ctypes
import os
import select
import shutil
import signal
import subprocess
import time
import weakref

import numpy as np

import radiation_server as rs

SCRATCH = os.environ.get("RADSRVP_SCRATCH", os.path.join(os.path.dirname(rs.SCRATCH), "mE_persist"))
BIN = os.environ.get("RADSRVP_BIN", os.path.join(SCRATCH, "mE2/model/P2SAoM40.bin"))
# date -> (itime of the restart, restart file, window end for the I file (the window just has to be non-empty),
#          first radiation step = the entry step of the server: MOD(itime-ITIMEI,NRAD)=0; jan01's run starts 1949-12-01 (ITIMEI=16032), so
#          its first radiation step is 17522, two model steps after the restart)
DATES = {
    "nov26": (33312, rs.FF + "/_pristine_restarts/fort1_nov26_itime33312.nc", (1950, 11, 26, 3), 33312),
    "dec01": (33552, rs.FF + "/_pristine_restarts/fort1_dec01_itime33552.nc", (1950, 12, 1, 3), 33552),
    "jan01": (17520, rs.FF + "/_pristine_restarts/fort1_jan01_itime17520.nc", (1950, 1, 1, 3), 17522),
}

_LIVE = weakref.WeakSet()


def server_available(date="nov26"):
    return os.path.exists(BIN) and os.path.exists(DATES[date][1]) and os.path.isdir(rs.HUGE)


def _pdeathsig():
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


@atexit.register
def _stop_all():
    for s in list(_LIVE):
        try:
            s.stop(timeout=10)
        except Exception:
            pass


class ServerError(RuntimeError):
    pass


class PersistentServer:
    def __init__(self, date="nov26", rundir=None, binary=None, daily="ORE", start_timeout=300.0, request_timeout=600.0, extra_env=None):
        self.date = date
        self.itime_restart, self.restart, self.end, self.itime0 = DATES[date]
        self.rundir = rundir or os.path.join(SCRATCH, "run_persist_%s_%d" % (date, os.getpid()))
        self.binary = binary or BIN
        self.daily = daily
        self.start_timeout, self.request_timeout = start_timeout, request_timeout
        self.extra_env = extra_env or {}
        self.proc = None
        self.fd_req = self.fd_rsp = None
        self.buf = b""
        self.itime_entry = None
        self.seed_entry = None
        self.start_wall = None
        self.calls = []

    # ---------------------------------------------------------------------------------------------------------- lifecycle
    def _prepare(self):
        rd = self.rundir
        if os.path.exists(rd):
            shutil.rmtree(rd)
        os.makedirs(rd)
        for fn in ("I", "P2SAoM40ln", "P2SAoM40uln", "runtime_opts"):
            shutil.copy(os.path.join(rs.HUGE, fn), rd)
        shutil.copy(self.binary, os.path.join(rd, "P2SAoM40.bin"))
        shutil.copy(self.binary, os.path.join(rd, "P2SAoM40"))
        txt = open(os.path.join(rd, "I")).read().split("\n")
        for i, ln in enumerate(txt):
            if "YEARE=" in ln:
                y, m, d, h = self.end
                txt[i] = "YEARE=%d,MONTHE=%d,DATEE=%d,HOURE=%d," % (y, m, d, h)
                break
        open(os.path.join(rd, "I"), "w").write("\n".join(txt))
        for fn in ("fort.1.nc", "fort.2.nc"):
            shutil.copy(self.restart, os.path.join(rd, fn))
        subprocess.run(["sh", "P2SAoM40ln"], cwd=rd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.mkfifo(os.path.join(rd, "req.fifo"))
        os.mkfifo(os.path.join(rd, "rsp.fifo"))

    def start(self):
        if self.proc is not None:
            raise ServerError("already started")
        t0 = time.time()
        self._prepare()
        rd = self.rundir
        # O_RDWR on both FIFOs: opening never blocks and the client counts as a writer of the request pipe (EOF reaches the model when we go away)
        self.fd_req = os.open(os.path.join(rd, "req.fifo"), os.O_RDWR)
        self.fd_rsp = os.open(os.path.join(rd, "rsp.fifo"), os.O_RDWR | os.O_NONBLOCK)
        env = rs._env({"RADSRVP": "1", "RADSRVP_REQ": "req.fifo", "RADSRVP_RSP": "rsp.fifo", "RADSRVP_DAILY": self.daily,
                       "RADSRVP_ITIME": str(self.itime0)})
        env.update(self.extra_env)
        self.proc = subprocess.Popen(["./P2SAoM40", "-i", "I"], cwd=rd, env=env, preexec_fn=_pdeathsig,
                                     stdin=subprocess.DEVNULL, stdout=open(os.path.join(rd, "run.PRT"), "w"), stderr=subprocess.STDOUT)
        _LIVE.add(self)
        try:
            ln = self._readline(self.start_timeout)
        except Exception:
            self.stop(timeout=5)
            raise
        w = ln.split()
        if not (w and w[0] == "READY"):
            self.stop(timeout=5)
            raise ServerError("unexpected start reply: %r" % ln)
        self.itime_entry, self.seed_entry = int(w[1]), int(w[2])
        self.start_wall = time.time() - t0
        return self

    def _readline(self, timeout):
        end = time.time() + timeout
        while b"\n" not in self.buf:
            left = end - time.time()
            if left <= 0:
                raise ServerError("timeout waiting for the radiation server (%.0f s); log tail:\n%s" % (timeout, self._tail()))
            r, _, _ = select.select([self.fd_rsp], [], [], min(left, 0.5))
            if r:
                try:
                    self.buf += os.read(self.fd_rsp, 4096)
                except BlockingIOError:
                    pass
            elif self.proc.poll() is not None and b"\n" not in self.buf:
                raise ServerError("radiation server exited (rc=%s):\n%s" % (self.proc.returncode, self._tail()))
        ln, self.buf = self.buf.split(b"\n", 1)
        return ln.decode()

    def _tail(self):
        try:
            return open(os.path.join(self.rundir, "run.PRT"), errors="replace").read()[-1500:]
        except OSError:
            return ""

    def _send(self, line):
        if self.proc is None or self.proc.poll() is not None:
            raise ServerError("radiation server is not running")
        os.write(self.fd_req, (line + "\n").encode())

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout=30.0):
        """Clean shutdown: STOP, wait, escalate to SIGTERM/SIGKILL; closes the pipes.  Idempotent."""
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                self._send("STOP")
                self._readline(min(timeout, 10))
            except Exception:
                pass
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for fd in (self.fd_req, self.fd_rsp):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fd_req = self.fd_rsp = None
        self.proc = None
        _LIVE.discard(self)

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # ---------------------------------------------------------------------------------------------------------- requests
    def request(self, state, itime, seed=None, keep_packets=False):
        """state: dict of rs.INPUT_FIELDS arrays; returns dict of output arrays + '_wall_s' (python wall), '_server_s' (RADIA+IO in the model)."""
        rd = self.rundir
        pin, pout = "rsv_in.bin", "rsv_out.bin"
        t0 = time.time()
        rs.write_packet(os.path.join(rd, pin), state)
        if os.path.exists(os.path.join(rd, pout)):
            os.remove(os.path.join(rd, pout))
        self._send("SERVE %d %s %s %s" % (itime, "-" if seed is None else int(seed), pin, pout))
        ln = self._readline(self.request_timeout)
        w = ln.split()
        if not w or w[0] != "OK":
            raise ServerError("request itime %d failed: %s" % (itime, ln))
        out = rs.read_packet(os.path.join(rd, pout))
        out["_server_s"] = float(w[3])
        out["_wall_s"] = time.time() - t0
        self.calls.append((itime, out["_server_s"], out["_wall_s"]))
        return out

    run_radiation = request

    def ping(self):
        self._send("PING")
        return self._readline(30)


_DEFAULT = {}


def run_radiation(state, itime, date="nov26", seed=None):
    """Module-level convenience: lazily starts one default persistent server per date (stopped at exit)."""
    s = _DEFAULT.get(date)
    if s is None or not s.alive():
        s = _DEFAULT[date] = PersistentServer(date).start()
    return s.request(state, itime, seed)
