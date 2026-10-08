"""D153: Python driver for the ModelE radiation server (real RADIA, SOCRATES as a black box).

The server is the instrumented P2SAoM40.bin built from the pristine tree plus
instrumentation/ATM_DRV_atmstep.f.patch and instrumentation/ATM_DRV_radsrv.f.patch (units 1440-1469).
Protocol (file exchange, "2.D-like"): the model starts from the nov26 restart, runs to the requested
radiation step (RADSRV_ITIME), overwrites every RADIA input listed in INPUT_FIELDS from the packet file
(RADSRV_IN), calls the real RADIA, writes the output packet (RADSRV_OUT) and stops.  Start-up cost of every
call = model initialisation (restart read etc.) + the model steps before the requested one + RADIA itself.

Packet format (big-endian, same family as the other ff dumps): repeated records
  char*16 name, int32 rank, int32 n(1:4), float64 data in Fortran order.
Input records must be in INPUT_FIELDS order (the Fortran reader checks name and shape and stops otherwise).

Functions: read_packet, write_packet, run_radiation(state, itime), run_many (concurrent), dump_state (the model's
own live inputs/outputs via RADSRV_MODE=dump), server_available.
"""
import os
import shutil
import subprocess
import time
from collections import OrderedDict

import numpy as np

IM, JM, LM, LM_REQ = 72, 46, 40, 3
KAIJ = 750

SCRATCH = os.environ.get(
    "RADSRV_SCRATCH",
    "/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/"
    "-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/"
    "ac69365f-34cd-40d2-965e-805e4c93b52b/scratchpad/mE_radsrv")
BIN = os.environ.get("RADSRV_BIN", os.path.join(SCRATCH, "mE2/model/P2SAoM40.bin"))
SRC = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0"
HUGE = SRC + "/ModelE_Support/huge_space/P2SAoM40"
FF = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
RESTARTS = {33312: FF + "/_pristine_restarts/fort1_nov26_itime33312.nc"}
NC_LIB = "/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib"

# (name, shape in the Fortran layout used by the model) in packet order
_L3 = (LM, IM, JM)
INPUT_FIELDS = OrderedDict([
    ("T", (IM, JM, LM)), ("Q", (IM, JM, LM)),
    ("PK", _L3), ("PMID", _L3), ("PDSIG", _L3), ("MA", _L3), ("BYMA", _L3), ("PEDN", (LM + 1, IM, JM)),
    ("W_CLOUD", _L3), ("FRAC_ST_WATER", _L3), ("FRAC_ST_ICE", _L3), ("FRAC_CNV_WATER", _L3),
    ("FRAC_CNV_ICE", _L3), ("MIX_ST_WATER", _L3), ("MIX_ST_ICE", _L3), ("MIX_CNV_WATER", _L3),
    ("MIX_CNV_ICE", _L3), ("DIM_ST_WATER", _L3), ("DIM_ST_ICE", _L3), ("DIM_CNV_WATER", _L3),
    ("DIM_CNV_ICE", _L3), ("FRAC_AREA_ST", _L3), ("FRAC_AREA_CNV", _L3),
    ("TAUSS", _L3), ("TAUMC", _L3), ("CLDSS", _L3), ("CLDMC", _L3),
    ("RQT", (LM_REQ, IM, JM)), ("KLIQ", (LM, 4, IM, JM)), ("SNOAGE", (3, IM, JM)), ("LTROPO", (IM, JM)),
    ("RSI", (IM, JM)), ("ZSI", (IM, JM)), ("SNOWI", (IM, JM)), ("POND_MELT", (IM, JM)), ("FLAG_DSWS", (IM, JM)),
    ("FLAKE", (IM, JM)), ("DLAKE", (IM, JM)), ("FLICE", (IM, JM)), ("FLAND", (IM, JM)), ("FEARTH", (IM, JM)),
    ("GTEMPR1", (IM, JM)), ("GTEMPR2", (IM, JM)), ("GTEMPR3", (IM, JM)), ("GTEMPR4", (IM, JM)),
    ("TSAVG", (IM, JM)), ("WSAVG", (IM, JM)), ("SNOWLI", (IM, JM)), ("ZSNOWI", (IM, JM)), ("BARESW", (IM, JM)),
    ("FRSNOW", (2, IM, JM)), ("SNOWD", (2, IM, JM)),
])
OUTPUT_FIELDS = ["T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "SNOAGE", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB",
                 "FSRDIR", "SRVISSURF", "FSRDIF", "DIRVIS", "DIRNIR", "DIFNIR", "SRDN", "CFRAC", "COSZ1", "AIJ"]


def server_available():
    return os.path.exists(BIN) and os.path.exists(RESTARTS[33312]) and os.path.isdir(HUGE)


def read_packet(path):
    buf = open(path, "rb").read()
    out, off = OrderedDict(), 0
    while off < len(buf):
        name = buf[off:off + 16].decode().strip()
        off += 16
        rank = int(np.frombuffer(buf, ">i4", 1, off)[0])
        off += 4
        n = [int(x) for x in np.frombuffer(buf, ">i4", 4, off)]
        off += 16
        shp = tuple(n[:rank])
        cnt = int(np.prod(shp))
        out[name] = np.frombuffer(buf, ">f8", cnt, off).astype(np.float64).reshape(shp, order="F")
        off += 8 * cnt
    return out


def write_packet(path, state):
    """state: dict name -> array with the Fortran shape of INPUT_FIELDS; written in INPUT_FIELDS order."""
    with open(path, "wb") as f:
        for name, shp in INPUT_FIELDS.items():
            a = np.asarray(state[name], dtype=np.float64)
            if a.shape != shp:
                raise ValueError("%s: shape %s, expected %s" % (name, a.shape, shp))
            n = list(shp) + [1] * (4 - len(shp))
            f.write(name.ljust(16).encode())
            f.write(np.array([len(shp)] + n, ">i4").tobytes())
            f.write(np.asarray(a, ">f8").tobytes(order="F"))


def _prepare(rundir, restart):
    if os.path.exists(rundir):
        shutil.rmtree(rundir)
    os.makedirs(rundir)
    for fn in ("I", "P2SAoM40ln", "P2SAoM40uln", "runtime_opts"):
        shutil.copy(os.path.join(HUGE, fn), rundir)
    shutil.copy(BIN, os.path.join(rundir, "P2SAoM40.bin"))
    shutil.copy(BIN, os.path.join(rundir, "P2SAoM40"))
    # six model steps window (the server stops at the requested radiation step; dump mode runs all six)
    txt = open(os.path.join(rundir, "I")).read().split("\n")
    for i, ln in enumerate(txt):
        if "YEARE=" in ln:
            txt[i] = "YEARE=1950,MONTHE=11,DATEE=26,HOURE=3,"
            break
    open(os.path.join(rundir, "I"), "w").write("\n".join(txt))
    for fn in ("fort.1.nc", "fort.2.nc"):
        shutil.copy(restart, os.path.join(rundir, fn))
    subprocess.run(["sh", "P2SAoM40ln"], cwd=rundir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _env(extra):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = NC_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MP_SET_NUMTHREADS"] = "1"
    env.update(extra)
    return env


def launch(state, itime, rundir, restart=None):
    """Start one server run (non blocking); returns (Popen, rundir, t0)."""
    restart = restart or RESTARTS[33312]
    _prepare(rundir, restart)
    pin = os.path.join(rundir, "rsv_in.bin")
    write_packet(pin, state)
    # relative names: the Fortran hook reads env strings into 256-character buffers (scratch paths are longer)
    env = _env({"RADSRV_MODE": "serve", "RADSRV_ITIME": str(itime), "RADSRV_IN": "rsv_in.bin",
                "RADSRV_OUT": "rsv_out.bin"})
    p = subprocess.Popen(["./P2SAoM40", "-i", "I"], cwd=rundir, env=env,
                         stdout=open(os.path.join(rundir, "run.PRT"), "w"), stderr=subprocess.STDOUT)
    return p, rundir, time.time()


def collect(p, rundir, t0):
    p.wait()
    wall = time.time() - t0
    po = os.path.join(rundir, "rsv_out.bin")
    if not os.path.exists(po):
        tail = open(os.path.join(rundir, "run.PRT"), errors="replace").read()[-1500:]
        raise RuntimeError("radiation server produced no output (rc=%s):\n%s" % (p.returncode, tail))
    out = read_packet(po)
    out["_wall_s"] = wall
    return out


def run_radiation(state, itime=33312, rundir=None, restart=None):
    """state: dict of INPUT_FIELDS arrays (Fortran shapes); returns dict of output arrays (+ '_wall_s')."""
    rundir = rundir or os.path.join(SCRATCH, "run_py_%d" % os.getpid())
    p, rd, t0 = launch(state, itime, rundir, restart)
    return collect(p, rd, t0)


def run_many(jobs, max_parallel=10):
    """jobs: list of (key, state, itime, rundir). Concurrent launch; returns dict key -> outputs."""
    res, running, queue = {}, [], list(jobs)
    while queue or running:
        while queue and len(running) < max_parallel:
            key, st, it, rd = queue.pop(0)
            running.append((key,) + launch(st, it, rd))
        key, p, rd, t0 = running.pop(0)
        res[key] = collect(p, rd, t0)
    return res


def dump_state(rundir, itime, ffd=True, nstep=6):
    """Run the unmodified-physics model in dump mode (RADSRV_MODE=dump): records, for every radiation step in the
    window, the live RADIA inputs (rsv_n26_<it>_in.bin) and outputs (..._out.bin, AIJD = AIJ delta, rounded)."""
    _prepare(rundir, RESTARTS[33312])
    env = _env({"RADSRV_MODE": "dump", "RADSRV_ITIME": str(itime), "RADSRV_NSTEP": str(nstep), "RADSRV_TAG": "n26"})
    t0 = time.time()
    subprocess.run(["./P2SAoM40", "-i", "I"], cwd=rundir, env=env,
                   stdout=open(os.path.join(rundir, "run.PRT"), "w"), stderr=subprocess.STDOUT)
    return time.time() - t0
